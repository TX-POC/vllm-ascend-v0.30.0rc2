# SPDX-License-Identifier: Apache-2.0

import inspect

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
from vllm_ascend.dsa_offload.ops import _normalize_lidu_weights_layout
# isort: on

_SPARSE_COUNT = onv._SPARSE_COUNT


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
    attention_slots = torch.full(
        (batch, 1, attention_capacity), -1, dtype=torch.int32, device=device
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

    out = onv.sparse_flash_attention_for_offload_c8(
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
