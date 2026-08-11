# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 数据面到 Ascend 自定义算子的薄适配层。

本模块只固定 LIDU、KSC、SFA-Offload 和满块复制的 tensor ABI。请求阶段、
resident 行分配、DRAM block 预留和 attention 时序由上层 runtime 负责；
这里不隐式创建输出 tensor，也不执行 D2H 或 list-to-tensor 转换。
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from vllm.logger import logger


def _install_a5_c8_mlapo_cache_guard() -> None:
    """Install the packed-C8 MLAPO resource guard after DeviceOperator loads.

    ``ascend_config`` imports ``dsa_offload`` while the platform is still being
    initialized, so importing ``device_op`` from ``dsa_offload.__init__`` creates
    an ``ascend_config -> dsa_offload -> device_op -> utils -> ascend_config``
    cycle.  ``dsa_offload.ops`` is imported later by SFA, after ``device_op`` is
    already initialized, which is a safe point to install this narrow guard.

    A5 packed-C8 stores latent KV, RoPE and per-tile scales in ``kv_cache[0]``.
    ``npu_mla_prolog_v3`` still requires a non-aliased empty ``kr_cache`` input,
    and both float/prolog-v3 weight-preparation paths use a stable 0-element BF16
    tensor for that purpose.  Some quantized MLAPO paths do not initialize it,
    so provide it lazily without changing the cache contents or operator ABI.
    """

    from vllm_ascend.device.device_op import DeviceOperator

    if getattr(DeviceOperator, "_dsa_a5_c8_mlapo_cache_guard", False):
        return

    original = DeviceOperator.sfa_preprocess_with_mlapo

    def guarded_sfa_preprocess_with_mlapo(
        sfa_impl,
        hidden_states: torch.Tensor,
        kv_cache: tuple,
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_input_tokens: int,
    ) -> tuple:
        if getattr(sfa_impl, "enable_sparse_sfa_c8", False):
            decode_k_nope = kv_cache[0]
            created_dummy = False
            created_clip = False
            if getattr(sfa_impl, "sfa_qsfa_kr_cache_dummy", None) is None:
                sfa_impl.sfa_qsfa_kr_cache_dummy = torch.empty(
                    0,
                    dtype=torch.bfloat16,
                    device=decode_k_nope.device,
                )
                created_dummy = True
            if getattr(sfa_impl, "sfa_qsfa_k_nope_clip_alpha", None) is None:
                sfa_impl.sfa_qsfa_k_nope_clip_alpha = torch.ones(
                    1,
                    dtype=torch.float32,
                    device=decode_k_nope.device,
                )
                created_clip = True

            if created_dummy or created_clip:
                quant_method = getattr(
                    getattr(
                        getattr(sfa_impl, "fused_qkv_a_proj", None),
                        "quant_method",
                        None,
                    ),
                    "quant_method",
                    None,
                )
                logger.info_once(
                    "[DSA_A5_C8_MLAPO_GUARD] layer=%s quant_method=%s "
                    "created_kr_dummy=%s created_clip_alpha=%s "
                    "packed_shape=%s packed_dtype=%s packed_stride=%s "
                    "kr_dummy_shape=%s kr_dummy_dtype=%s",
                    getattr(sfa_impl, "layer_name", None),
                    type(quant_method).__name__
                    if quant_method is not None
                    else "None",
                    created_dummy,
                    created_clip,
                    tuple(decode_k_nope.shape),
                    decode_k_nope.dtype,
                    tuple(decode_k_nope.stride()),
                    tuple(sfa_impl.sfa_qsfa_kr_cache_dummy.shape),
                    sfa_impl.sfa_qsfa_kr_cache_dummy.dtype,
                )

        return original(
            sfa_impl,
            hidden_states,
            kv_cache,
            cos,
            sin,
            slot_mapping,
            num_input_tokens,
        )

    DeviceOperator.sfa_preprocess_with_mlapo = staticmethod(
        guarded_sfa_preprocess_with_mlapo
    )
    DeviceOperator._dsa_a5_c8_mlapo_cache_guard = True


_install_a5_c8_mlapo_cache_guard()


class DSALightningIndexerOutputs(NamedTuple):
    """LIDU 的四个 caller-owned 原址输出。"""

    topk_index: torch.Tensor
    topk_slots: torch.Tensor
    miss_count: torch.Tensor
    tail_info: torch.Tensor


class DSAOffloadSelectionOutput(NamedTuple):
    """LIDU/KSC 完成后交给 SFA-Offload 的逐行 resident 视图。"""

    sparse_indices: torch.Tensor
    tail_info: torch.Tensor


_REQUIRED_OPS = (
    "npu_lightning_indexer_decode_update_out",
    "npu_kvcache_scatter_copy",
    "npu_sparse_flash_attention_for_offload",
    "kv_cache_full_block_dump",
)


def require_dsa_offload_ops() -> None:
    """在 worker 初始化期确认四个设备算子均已部署。"""

    missing = [
        op_name
        for op_name in _REQUIRED_OPS
        if not hasattr(torch.ops._C_ascend, op_name)
    ]
    if missing:
        raise RuntimeError(
            "DSA sparse offload custom operators are not installed: "
            f"{tuple(missing)}"
        )


def _squeeze_cache_head_dim(
    cache: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if cache.ndim == 4 and cache.shape[2] == 1:
        return cache.squeeze(2)
    if cache.ndim == 3:
        return cache
    raise ValueError(
        f"{name} must be [blocks, block, 1, dim] or "
        f"[blocks, block, dim], got {tuple(cache.shape)}"
    )


def _normalize_lidu_weights_layout(weights: torch.Tensor) -> torch.Tensor:
    """将 v0.23 合并投影产生的 weights view 收敛到 LIDU 连续布局。

    v0.23 的 SFA 通过 ``wk_weights_proj`` 一次生成 key 与 weights，
    ``kw[:, head_dim:]`` 在 batch size 大于 1 时是带较大行 stride 的列后缀
    view。原生 lightning-indexer 能接收该 view，但 LIDU AscendC kernel 按
    紧凑 ``[B, N_idx]`` 布局寻址。这里只规范化这一个小输入；cache 和固定
    容量元数据仍必须由各自 owner 提供连续 tensor，避免在逐层热路径隐式
    复制大张量。
    """

    return weights.contiguous()


def lightning_indexer_decode_update(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots: torch.Tensor,
    row_modes: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
    outputs: DSALightningIndexerOutputs,
) -> None:
    weights = _normalize_lidu_weights_layout(weights)
    torch.ops._C_ascend.npu_lightning_indexer_decode_update_out(
        query,
        key,
        weights,
        req_pool_entries,
        cache_slots,
        row_modes,
        actual_seq_lengths_key,
        block_table,
        outputs.topk_index,
        outputs.topk_slots,
        outputs.miss_count,
        outputs.tail_info,
    )


def kvcache_scatter_copy(
    *,
    resident_nope_cache: torch.Tensor,
    resident_rope_cache: torch.Tensor,
    dram_nope_arena: torch.Tensor,
    dram_rope_arena: torch.Tensor,
    resident_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    src_token_ids: torch.Tensor,
    dst_slots: torch.Tensor,
    copy_counts: torch.Tensor,
) -> None:
    torch.ops._C_ascend.npu_kvcache_scatter_copy(
        _squeeze_cache_head_dim(
            resident_rope_cache,
            name="resident_rope_cache",
        ),
        _squeeze_cache_head_dim(
            resident_nope_cache,
            name="resident_nope_cache",
        ),
        _squeeze_cache_head_dim(
            dram_rope_arena,
            name="dram_rope_arena",
        ),
        _squeeze_cache_head_dim(
            dram_nope_arena,
            name="dram_nope_arena",
        ),
        resident_block_table,
        dram_block_table,
        src_token_ids,
        dst_slots,
        copy_counts,
    )


def sparse_flash_attention_for_offload(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    sparse_indices: torch.Tensor,
    tail_info: torch.Tensor,
    scale_value: float,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C_ascend.npu_sparse_flash_attention_for_offload(
        query,
        key,
        key,
        sparse_indices,
        tail_info,
        float(scale_value),
        1,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        "TND",
        "PA_BSND",
        3,
    )


def _fallback_dump_full_kv_cache_blocks(
    *,
    resident_nope_cache: torch.Tensor,
    resident_rope_cache: torch.Tensor,
    dram_nope_arena: torch.Tensor,
    dram_rope_arena: torch.Tensor,
    src_block_ids: torch.Tensor,
    dst_block_ids: torch.Tensor,
) -> None:
    """A5 fallback for Prefill full-block HBM -> swapped-DRAM copies.

    This path is intentionally limited to final-Prefill offload. It performs one
    batched gather + one batched destination update per cache plane and never
    runs in the steady Decode hot path.
    """

    src_nope = _squeeze_cache_head_dim(
        resident_nope_cache,
        name="resident_nope_cache",
    )
    src_rope = _squeeze_cache_head_dim(
        resident_rope_cache,
        name="resident_rope_cache",
    )
    dst_nope = _squeeze_cache_head_dim(
        dram_nope_arena,
        name="dram_nope_arena",
    )
    dst_rope = _squeeze_cache_head_dim(
        dram_rope_arena,
        name="dram_rope_arena",
    )

    src_index = src_block_ids.to(dtype=torch.int64)
    dst_index = dst_block_ids.to(dtype=torch.int64)

    nope_blocks = torch.index_select(src_nope, 0, src_index)
    rope_blocks = torch.index_select(src_rope, 0, src_index)
    dst_nope.index_copy_(0, dst_index, nope_blocks)
    dst_rope.index_copy_(0, dst_index, rope_blocks)


def dump_full_kv_cache_blocks(
    *,
    resident_nope_cache: torch.Tensor,
    resident_rope_cache: torch.Tensor,
    dram_nope_arena: torch.Tensor,
    dram_rope_arena: torch.Tensor,
    src_block_ids: torch.Tensor,
    dst_block_ids: torch.Tensor,
) -> None:
    if src_block_ids.numel() == 0:
        return

    # Preferred Ascend 950 path. nanovllm_dsa_a5 is imported once during
    # runtime initialization; the new kernel performs direct HBM -> swapped-GM
    # block copies without host staging or temporary KV materialization.
    nanovllm_namespace = getattr(torch.ops, "nanovllm_dsa", None)
    a5_full_block_dump = (
        getattr(nanovllm_namespace, "full_block_dump", None)
        if nanovllm_namespace is not None
        else None
    )
    if a5_full_block_dump is not None:
        a5_full_block_dump(
            _squeeze_cache_head_dim(
                resident_rope_cache,
                name="resident_rope_cache",
            ),
            _squeeze_cache_head_dim(
                resident_nope_cache,
                name="resident_nope_cache",
            ),
            _squeeze_cache_head_dim(
                dram_rope_arena,
                name="dram_rope_arena",
            ),
            _squeeze_cache_head_dim(
                dram_nope_arena,
                name="dram_nope_arena",
            ),
            src_block_ids,
            dst_block_ids,
        )
        return

    try:
        torch.ops._C_ascend.kv_cache_full_block_dump(
            _squeeze_cache_head_dim(
                resident_nope_cache,
                name="resident_nope_cache",
            ),
            _squeeze_cache_head_dim(
                resident_rope_cache,
                name="resident_rope_cache",
            ),
            _squeeze_cache_head_dim(
                dram_nope_arena,
                name="dram_nope_arena",
            ),
            _squeeze_cache_head_dim(
                dram_rope_arena,
                name="dram_rope_arena",
            ),
            src_block_ids,
            dst_block_ids,
        )
        return
    except RuntimeError as exc:
        message = str(exc)
        missing_a5_dump = (
            "aclnnKvCacheFullBlockDump" in message
            or "libopapi.so" in message
        )
        if not missing_a5_dump:
            raise
        logger.warning_once(
            "[DSA_PREFILL_FULL_DUMP] A5 full-block operator is unavailable; "
            "falling back to batched index_select/index_copy_. Rebuild "
            "nanovllm_dsa_a5 to enable the direct AscendC path."
        )

    try:
        _fallback_dump_full_kv_cache_blocks(
            resident_nope_cache=resident_nope_cache,
            resident_rope_cache=resident_rope_cache,
            dram_nope_arena=dram_nope_arena,
            dram_rope_arena=dram_rope_arena,
            src_block_ids=src_block_ids,
            dst_block_ids=dst_block_ids,
        )
    except Exception as fallback_exc:
        raise RuntimeError(
            "DSA Prefill full-block dump failed: neither the A5 nanovllm "
            "full_block_dump operator nor the legacy operator is available, "
            "and the generic swapped-memory fallback also failed."
        ) from fallback_exc
