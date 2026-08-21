# SPDX-License-Identifier: Apache-2.0

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

# Pre-initialize `vllm_ascend.ops` (which fully loads `device_op` via the
# fused_moe -> experts_selector chain) BEFORE importing `dsa_offload.ops`.
# `dsa_offload/ops.py` eagerly calls `_install_a5_c8_mlapo_cache_guard()`
# which imports `vllm_ascend.device.device_op`; its own docstring requires
# `device_op` to already be loaded, otherwise a pre-existing circular import
# (`device_op` <-> `vllm_ascend.ops.*`) fails. Normal runtime reaches this
# point after `device_op` is loaded; a direct UT import must force the order.
# The order below is intentional; keep isort from reordering it.
# isort: off
import vllm_ascend.ops  # noqa: F401

from vllm_ascend.dsa_offload import ops_nanovllm as onv
from vllm_ascend.dsa_offload.a5_c8_policy import (
    _refresh_fixed_metadata,
    _select_attention_slots,
)
from vllm_ascend.dsa_offload.a5_c8_precision_policy import (
    _pack_c8_rows_as_bytes,
    sparse_flash_attention_for_offload_c8,
)
from vllm_ascend.dsa_offload.contracts import DSA_OP_BACKEND_NANOVLLM_A5
from vllm_ascend.dsa_offload.ops import (
    DSALightningIndexerOutputs,
    DSAOffloadSelectionOutput,
    _normalize_lidu_weights_layout,
)
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)
from vllm_ascend.dsa_offload.runtime import (
    DSALayerOffloadContext,
    DSAOffloadRuntime,
)
# isort: on

_SPARSE_COUNT = onv._SPARSE_COUNT


def test_a5_c8_policy_keeps_kvaf_dump_owners_unpatched() -> None:
    assert DSAOffloadRuntime.dump_prefill_full_blocks.__module__ == (
        "vllm_ascend.dsa_offload.runtime"
    )
    assert DSALayerOffloadContext.dump_full_blocks.__module__ == (
        "vllm_ascend.dsa_offload.runtime"
    )


def test_lidu_weights_normalizes_fused_projection_suffix_view() -> None:
    fused_projection = torch.arange(
        4 * 192,
        dtype=torch.bfloat16,
    ).view(4, 192)
    weights = fused_projection[:, 128:]

    assert weights.shape == (4, 64)
    assert weights.stride() == (192, 1)
    assert not weights.is_contiguous()

    normalized = _normalize_lidu_weights_layout(weights)

    assert normalized.shape == weights.shape
    assert normalized.stride() == (64, 1)
    assert normalized.is_contiguous()
    torch.testing.assert_close(normalized, weights)


def test_lidu_weights_keeps_already_contiguous_storage() -> None:
    weights = torch.empty((1, 64), dtype=torch.bfloat16)

    normalized = _normalize_lidu_weights_layout(weights)

    assert normalized.is_contiguous()
    assert normalized.data_ptr() == weights.data_ptr()


def test_c8_packed_producer_preserves_all_payload_bytes() -> None:
    """The 512/128/16 packed C8 regions retain their source bytes exactly."""

    row_count = 2
    k_nope = torch.arange(row_count * 512, dtype=torch.int16).to(
        torch.uint8
    ).view(row_count, 512)
    k_pe = torch.arange(row_count * 64, dtype=torch.bfloat16).view(
        row_count, 64
    )
    knope_scale = torch.arange(row_count * 4, dtype=torch.float32).view(
        row_count, 4
    )

    packed = _pack_c8_rows_as_bytes(
        k_nope,
        k_pe,
        knope_scale,
        row_count,
    )

    assert packed.dtype == torch.int8
    assert packed.shape == (row_count, 656)
    assert torch.equal(packed[:, :512], k_nope.view(torch.int8))
    assert torch.equal(packed[:, 512:640], k_pe.view(torch.int8))
    assert torch.equal(packed[:, 640:], knope_scale.view(torch.int8))


def test_c8_eager_attention_slots_use_active_contiguous_width() -> None:
    storage = torch.empty(
        (4, 1, _SPARSE_COUNT + 8192),
        dtype=torch.int32,
    )
    runtime = SimpleNamespace(
        graph_capture_row_count=0,
        _a5_c8_attention_slots=storage,
        _a5_c8_max_tail_tokens=8192,
        _a5_c8_active_attention_width=_SPARSE_COUNT + 108,
    )

    slots, max_tail_tokens = _select_attention_slots(runtime, num_reqs=2)

    assert slots.shape == (2, 1, _SPARSE_COUNT + 108)
    assert slots.is_contiguous()
    assert slots.data_ptr() == storage.data_ptr()
    assert max_tail_tokens == 108


def test_c8_metadata_width_covers_sparse_suffix_and_dense_rows() -> None:
    class _Buffer:
        def __init__(self, size: int) -> None:
            self.np = np.zeros(size, dtype=np.int32)
            self.cpu = torch.from_numpy(self.np)
            self.gpu = torch.zeros(size, dtype=torch.int32)

    runtime = SimpleNamespace(
        max_num_reqs=2,
        block_size=128,
        _tokens_after_schedule=np.asarray([5100, 3000], dtype=np.int64),
        _a5_candidate_lens=_Buffer(2),
        _a5_logical_seq_lengths=_Buffer(2),
        _a5_resident_seq_lengths=_Buffer(2),
        _a5_c8_attention_widths=np.empty(2, dtype=np.int32),
        _a5_c8_max_tail_tokens=8192,
        _a5_c8_last_range_log=None,
        keep_decode_kv_in_hbm=True,
    )
    state = SimpleNamespace(
        stages_cpu=np.asarray(
            [
                int(DSARequestCacheStage.ENTER_SPARSE_DECODE),
                int(DSARequestCacheStage.DENSE_DECODE),
            ],
            dtype=np.int32,
        ),
        persistent_suffix_start_tokens_cpu=np.asarray(
            [4992, 0], dtype=np.int32
        ),
        resident_valid_tokens_cpu=np.asarray([4204, -1], dtype=np.int32),
    )

    _refresh_fixed_metadata(runtime, state=state, num_reqs=2)

    assert runtime._a5_candidate_lens.np.tolist() == [4992, 3000]
    assert runtime._a5_logical_seq_lengths.np.tolist() == [5100, 3000]
    assert runtime._a5_resident_seq_lengths.np.tolist() == [4204, 3000]
    assert runtime._a5_c8_attention_widths.tolist() == [2156, 3000]
    assert runtime._a5_c8_active_attention_width == 3000


# ---------------------------------------------------------------------------
# 模块：删除 KSC 的 DDR 搬运职责，保留 C8 Attention 索引语义
# ---------------------------------------------------------------------------


def _make_metadata_fixture(device="cpu", batch=4) -> dict:
    """构造最小 metadata 输入。第 0 行：C=40, K=5, actual=43 -> tail=3，
    期望 attention_slots[0,0,0:5]=destination_slots，[5:8]=[40,41,42]。"""
    dest_slots = torch.arange(
        batch * _SPARSE_COUNT, device=device, dtype=torch.int32
    ).view(batch, 1, _SPARSE_COUNT)
    cache_tokens = torch.tensor([40, 20, 60, 10], dtype=torch.int32, device=device)
    candidate_lens = torch.tensor([5, 20, 4, 0], dtype=torch.int32, device=device)
    actual_seq = torch.tensor([43, 21, 4, 0], dtype=torch.int32, device=device)
    resident_seq_lengths = torch.tensor(
        [43, 21, 4, 0], dtype=torch.int32, device=device
    )
    return {
        "destination_slots": dest_slots,
        "cache_tokens": cache_tokens,
        "candidate_lens": candidate_lens,
        "actual_seq_lengths_kv": actual_seq,
        "resident_seq_lengths": resident_seq_lengths,
    }


def test_build_c8_attention_metadata_writes_only_slots(device="cpu") -> None:
    """前 2048 项 = destination_slots；tail 段 = [cache_count, cache_count+tail)；
    resident_seq_lengths 不被改写；无 DDR tensor 入参。"""
    fx = _make_metadata_fixture(device=device)
    batch = fx["destination_slots"].shape[0]
    max_tail_tokens = 8
    attention_capacity = _SPARSE_COUNT + max_tail_tokens
    # 模拟跨 Decode step 复用的 scratch 中残留的旧索引；builder 必须自行
    # 覆盖完整 buffer，不能依赖调用方预填 -1。
    attention_slots = torch.full(
        (batch, 1, attention_capacity), -123456789, dtype=torch.int32, device=device
    )
    sentinel = fx["resident_seq_lengths"].clone()
    out = onv.build_c8_attention_metadata(
        destination_slots=fx["destination_slots"],
        cache_tokens=fx["cache_tokens"],
        candidate_lens=fx["candidate_lens"],
        actual_seq_lengths_kv=fx["actual_seq_lengths_kv"],
        max_tail_tokens=max_tail_tokens,
        resident_seq_lengths=fx["resident_seq_lengths"],
        attention_slots=attention_slots,
    )

    # 1) 前 2048 项 = destination_slots
    torch.testing.assert_close(
        attention_slots[:, 0, :_SPARSE_COUNT], fx["destination_slots"][:, 0, :]
    )

    # 2) tail 段（按行验证）
    tail_slice = attention_slots[:, 0, _SPARSE_COUNT:]
    tail_count = torch.clamp(
        fx["actual_seq_lengths_kv"] - fx["candidate_lens"],
        min=0,
        max=max_tail_tokens,
    )
    for row in range(batch):
        tc = int(tail_count[row])
        if tc > 0:
            expected = fx["cache_tokens"][row] + torch.arange(
                tc, device=device, dtype=torch.int32
            )
            torch.testing.assert_close(tail_slice[row, :tc], expected)
        if tc < max_tail_tokens:
            assert bool((tail_slice[row, tc:] == -1).all())

    assert not bool((attention_slots == -123456789).any())

    # 3) resident_seq_lengths 不被改写，且返回值为同一 tensor（透传）
    torch.testing.assert_close(fx["resident_seq_lengths"], sentinel)
    assert out is fx["resident_seq_lengths"]

    # 4) 无 DDR tensor 入参
    params = inspect.signature(onv.build_c8_attention_metadata).parameters
    assert "hbm_packed_kv" not in params
    assert "dram_packed_kv" not in params
    assert "dram_block_table" not in params


def test_build_c8_attention_metadata_tail_clamp(device="cpu") -> None:
    """tail 超过 max_tail_tokens 时截断，剩余槽位为 -1。"""
    fx = _make_metadata_fixture(device=device)
    max_tail_tokens = 1  # 第 0 行 actual-candidate=3 > 1 -> 只保留 1 个
    batch = fx["destination_slots"].shape[0]
    attention_slots = torch.full(
        (batch, 1, _SPARSE_COUNT + max_tail_tokens),
        -1,
        dtype=torch.int32,
        device=device,
    )
    onv.build_c8_attention_metadata(
        destination_slots=fx["destination_slots"],
        cache_tokens=fx["cache_tokens"],
        candidate_lens=fx["candidate_lens"],
        actual_seq_lengths_kv=fx["actual_seq_lengths_kv"],
        max_tail_tokens=max_tail_tokens,
        resident_seq_lengths=fx["resident_seq_lengths"],
        attention_slots=attention_slots,
    )
    tail_slice = attention_slots[:, 0, _SPARSE_COUNT:]
    assert int(tail_slice[0, 0]) == 40  # 第 0 行 expected [40]
    assert int(tail_slice[3, 0]) == -1  # 第 3 行 tail=0 -> -1


def test_build_c8_attention_metadata_dense_resident_fallback(
    device="cpu",
) -> None:
    """cache_tokens=0 时恢复连续 resident slots，且逐行覆盖脏缓冲。"""
    max_tail_tokens = 8
    attention_capacity = _SPARSE_COUNT + max_tail_tokens
    destination_slots = torch.full(
        (4, 1, _SPARSE_COUNT),
        -1,
        dtype=torch.int32,
        device=device,
    )
    destination_slots[1, 0] = torch.arange(
        _SPARSE_COUNT, dtype=torch.int32, device=device
    )
    cache_tokens = torch.tensor(
        [0, 2048, 0, 0], dtype=torch.int32, device=device
    )
    candidate_lens = torch.tensor(
        [15, 2048, 0, attention_capacity + 1],
        dtype=torch.int32,
        device=device,
    )
    actual_seq_lengths_kv = candidate_lens.clone()
    actual_seq_lengths_kv[1] = 2051
    resident_seq_lengths = actual_seq_lengths_kv.clone()
    resident_sentinel = resident_seq_lengths.clone()
    attention_slots = torch.full(
        (4, 1, attention_capacity),
        12345,
        dtype=torch.int32,
        device=device,
    )

    onv.build_c8_attention_metadata(
        destination_slots=destination_slots,
        cache_tokens=cache_tokens,
        candidate_lens=candidate_lens,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        max_tail_tokens=max_tail_tokens,
        resident_seq_lengths=resident_seq_lengths,
        attention_slots=attention_slots,
    )

    torch.testing.assert_close(
        attention_slots[0, 0, :15],
        torch.arange(15, dtype=torch.int32, device=device),
    )
    assert bool((attention_slots[0, 0, 15:] == -1).all())
    torch.testing.assert_close(
        attention_slots[1, 0, :_SPARSE_COUNT],
        destination_slots[1, 0],
    )
    torch.testing.assert_close(
        attention_slots[1, 0, _SPARSE_COUNT : _SPARSE_COUNT + 3],
        torch.tensor([2048, 2049, 2050], dtype=torch.int32, device=device),
    )
    assert bool((attention_slots[1, 0, _SPARSE_COUNT + 3 :] == -1).all())
    assert bool((attention_slots[2, 0] == -1).all())
    assert bool((attention_slots[3, 0] == -1).all())
    torch.testing.assert_close(resident_seq_lengths, resident_sentinel)


def test_kvaf_subgraph_c_materializes_subgraph_d_selection(
    monkeypatch,
) -> None:
    """KVAF BatchGet 后必须物化 C8 slots，并返回新版 C→D 字段。"""

    destination_slots = torch.arange(
        _SPARSE_COUNT, dtype=torch.int32
    ).view(1, 1, _SPARSE_COUNT)
    outputs = DSALightningIndexerOutputs(
        topk_index=torch.zeros_like(destination_slots),
        topk_slots=destination_slots,
        miss_count=torch.ones(1, dtype=torch.int32),
        tail_info=torch.zeros((1, 2), dtype=torch.int32),
    )
    cache_tokens = torch.tensor([2048], dtype=torch.int32)
    candidate_lens = torch.tensor([2048], dtype=torch.int32)
    logical_seq_lengths = torch.tensor([2051], dtype=torch.int32)
    resident_seq_lengths = torch.tensor([2051], dtype=torch.int32)
    attention_slots = torch.full(
        (1, 1, _SPARSE_COUNT + 8),
        -1,
        dtype=torch.int32,
    )
    kvaf_adapter = Mock()
    dsa_row_directory = SimpleNamespace(
        addresses=torch.ones(1, dtype=torch.int64),
        valid=torch.ones(1, dtype=torch.uint8),
    )
    runtime = SimpleNamespace(
        op_backend=DSA_OP_BACKEND_NANOVLLM_A5,
        is_c8=True,
        max_num_reqs=1,
        execution_num_reqs=1,
        active_num_reqs=1,
        _a5_c8_lidu_shape_logged=True,
        _a5_candidate_lens=SimpleNamespace(gpu=candidate_lens),
        _a5_logical_seq_lengths=SimpleNamespace(gpu=logical_seq_lengths),
        _a5_resident_seq_lengths=SimpleNamespace(gpu=resident_seq_lengths),
        _a5_c8_attention_slots=attention_slots,
        _a5_c8_max_tail_tokens=8,
        _a5_c8_active_attention_width=_SPARSE_COUNT + 3,
        _req_cache_tokens=cache_tokens,
        _c8_key_dequant_scales={0: torch.ones(1)},
        kvaf_adapter=kvaf_adapter,
        dsa_row_directory=dsa_row_directory,
        _require_kvaf_adapter=lambda: kvaf_adapter,
        _require_dsa_row_directory=lambda: dsa_row_directory,
        resident_token_pool=SimpleNamespace(
            get_cache_slots=lambda _layer_id: torch.zeros(
                1, dtype=torch.int32
            )
        ),
        get_lidu_outputs=lambda *, num_reqs: outputs,
    )
    context = DSALayerOffloadContext(
        layer_id=0,
        indexer_cache=torch.zeros(1),
        runtime=runtime,
    )
    monkeypatch.setattr(
        onv,
        "lightning_indexer_decode_update_c8",
        lambda **_kwargs: None,
    )

    selection = context.execute_decode_selection(
        query=torch.zeros((1, 4)),
        weights=torch.zeros((1, 1)),
        row_modes=torch.zeros(1, dtype=torch.int32),
        resident_pool_indices=torch.zeros(1, dtype=torch.int32),
        actual_seq_lengths_key=logical_seq_lengths,
        indexer_block_table=torch.zeros((1, 1), dtype=torch.int32),
        resident_nope_cache=torch.zeros((1, 128, 1, 656), dtype=torch.int8),
        resident_rope_cache=torch.zeros(1),
        resident_block_table=torch.zeros((1, 1), dtype=torch.int32),
        dram_block_table=torch.zeros((1, 1), dtype=torch.int32),
        actual_seq_lengths_query=torch.ones(1, dtype=torch.int32),
        query_dequant_scale=torch.ones(1),
    )

    kvaf_adapter.execute_batch_get.assert_called_once()
    assert selection._fields == (
        "attention_slots",
        "resident_seq_lengths",
    )
    assert selection.attention_slots.data_ptr() == attention_slots.data_ptr()
    assert (
        selection.resident_seq_lengths.data_ptr()
        == resident_seq_lengths.data_ptr()
    )
    torch.testing.assert_close(
        selection.attention_slots[:, :, :_SPARSE_COUNT],
        destination_slots,
    )
    torch.testing.assert_close(
        selection.attention_slots[0, 0, _SPARSE_COUNT:_SPARSE_COUNT + 3],
        torch.tensor([2048, 2049, 2050], dtype=torch.int32),
    )


def test_c8_kvaf_selection_reaches_sparse_attention(monkeypatch) -> None:
    """已物化的 KVAF C8 selection 必须能直接进入 QSFA。"""

    captured = {}
    expected = torch.ones((1, 1, 512))

    def fake_sfa_c8(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        onv,
        "sparse_flash_attention_for_offload_c8",
        fake_sfa_c8,
    )
    runtime = SimpleNamespace(
        op_backend=DSA_OP_BACKEND_NANOVLLM_A5,
        is_c8=True,
        dsa_row_directory=object(),
    )
    context = DSALayerOffloadContext(
        layer_id=0,
        indexer_cache=torch.zeros(1),
        runtime=runtime,
    )
    attention_slots = torch.zeros(
        (1, 1, _SPARSE_COUNT + 8), dtype=torch.int32
    )
    resident_seq_lengths = torch.tensor([2051], dtype=torch.int32)
    selection = DSAOffloadSelectionOutput(
        attention_slots=attention_slots,
        resident_seq_lengths=resident_seq_lengths,
    )

    result = context.execute_sparse_attention(
        query=torch.zeros((1, 1, 512)),
        resident_nope_cache=torch.zeros(
            (1, 128, 1, 656), dtype=torch.int8
        ),
        selection=selection,
        scale_value=1.0,
        resident_block_table=torch.zeros((1, 1), dtype=torch.int32),
        actual_seq_lengths_query=torch.ones(1, dtype=torch.int32),
        actual_seq_lengths_kv=torch.tensor([2051], dtype=torch.int32),
        query_rope=torch.zeros((1, 1, 64)),
        resident_rope_cache=torch.zeros(1),
    )

    assert result is expected
    assert captured["attention_slots"] is attention_slots
    assert captured["resident_seq_lengths"] is resident_seq_lengths
    assert captured["query"].shape[-1] == 576


def test_c8_sfa_no_clone_no_sync(monkeypatch) -> None:
    """QSFA 热路不调用 packed_kv.clone / torch.npu.synchronize（默认 debug 关闭）。"""
    import vllm_ascend.envs as envs

    monkeypatch.setenv("VLLM_ASCEND_DSA_A5_C8_DEBUG_DIAG", "0")
    assert not envs.VLLM_ASCEND_DSA_A5_C8_DEBUG_DIAG

    clone_called = False
    sync_called = False
    orig_clone = torch.Tensor.clone
    npu = getattr(torch, "npu", None)
    orig_sync = getattr(npu, "synchronize", None) if npu is not None else None

    def fake_clone(self, *args, **kwargs):
        nonlocal clone_called
        clone_called = True
        return orig_clone(self, *args, **kwargs)

    def fake_sync(*args, **kwargs):
        nonlocal sync_called
        sync_called = True

    monkeypatch.setattr(torch.Tensor, "clone", fake_clone)
    if orig_sync is not None:
        monkeypatch.setattr(torch.npu, "synchronize", fake_sync)

    captured = {}

    def fake_sfa_c8(query, packed_kv, attention_slots, block_table,
                    actual_seq_lengths_query, resident_seq_lengths, scale):
        captured["packed_kv"] = packed_kv
        return torch.zeros(*query.shape[:-1], 512, dtype=query.dtype)

    # `sparse_and_tail_attention_c8` is a custom op registered only when the A5
    # native extension is loaded. In UT it may not be registered yet, so use
    # raising=False to create the fake attribute regardless.
    monkeypatch.setattr(
        torch.ops.nanovllm_dsa, "sparse_and_tail_attention_c8", fake_sfa_c8,
        raising=False,
    )

    query = torch.zeros((1, 4, 576), dtype=torch.bfloat16)
    packed_kv = torch.zeros((8, 128, 1, 656), dtype=torch.float8_e4m3fn)
    attention_slots = torch.zeros((1, 1, _SPARSE_COUNT + 8), dtype=torch.int32)
    block_table = torch.zeros((1, 2), dtype=torch.int32)
    actual_q = torch.zeros((1,), dtype=torch.int32)
    resident_l = torch.zeros((1,), dtype=torch.int32)

    out = sparse_flash_attention_for_offload_c8(
        query=query,
        packed_kv=packed_kv,
        attention_slots=attention_slots,
        block_table=block_table,
        actual_seq_lengths_query=actual_q,
        resident_seq_lengths=resident_l,
        scale_value=1.0,
    )

    assert not clone_called, "C8 QSFA 热路不得调用 packed_kv.clone()"
    if orig_sync is not None:
        assert not sync_called, "C8 QSFA 热路不得调用 torch.npu.synchronize()"
    assert captured["packed_kv"] is packed_kv  # 直接使用传入 packed view
    assert out.shape == (1, 4, 512)
