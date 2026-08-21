# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 DSA registrations backed by the main ``vllm_ascend_C`` extension.

The A5 DSA operators used to require a second editable Python package,
``nanovllm_dsa_a5``, whose ``_C`` module registered the ``nanovllm_dsa`` torch
namespace. Keep that torch namespace for framework ABI compatibility, but load
its C++ registrations from the normal vLLM-Ascend extension instead. The A5
custom OPP package is installed below ``vllm_ascend/_cann_ops_custom`` by the
root build flow.
"""

from __future__ import annotations

import torch
import torch_npu

from vllm_ascend.a5_dsa_registration import (
    local_opapi_path as _registration_opapi_path,
)

BLOCK_SIZE = 128
CKV_DIM = 512
PACKED_KV_DIM = 656
SPARSE_COUNT = 2048

_OPAPI = _registration_opapi_path()

if not hasattr(torch.ops, "nanovllm_dsa"):
    raise RuntimeError(
        "vllm_ascend_C was loaded but A5 DSA torch registrations are absent. "
        "Rebuild vllm-ascend from the current source tree."
    )

lidu_cache_update = torch.ops.nanovllm_dsa.lidu_cache_update
lidu_cache_update_out = torch.ops.nanovllm_dsa.lidu_cache_update_out


def _check_c8_lidu_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> None:
    if (
        query.ndim != 3
        or query.shape[0] <= 0
        or query.shape[1] not in (32, 64)
        or query.shape[2] != 128
    ):
        raise ValueError("C8 LIDU query must be [B,32|64,128]")
    batch, heads, _ = query.shape
    if (
        key.ndim != 4
        or key.shape[0] <= 0
        or tuple(key.shape[1:]) != (BLOCK_SIZE, 1, 128)
    ):
        raise ValueError("C8 LIDU key must be [blocks,128,1,128]")
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None or query.dtype != fp8 or key.dtype != fp8:
        raise TypeError("A5 C8 LIDU query/key must be float8_e4m3fn")
    if (
        weights.shape != (batch, heads)
        or query_dequant_scale.shape != (batch, heads)
        or weights.dtype != torch.bfloat16
        or query_dequant_scale.dtype != torch.float32
    ):
        raise ValueError(
            "C8 LIDU weights must be bfloat16 [B,N] and "
            "query_dequant_scale must be float32 [B,N]"
        )
    if (
        key_dequant_scale.ndim != 3
        or key_dequant_scale.shape != key.shape[:-1]
        or key_dequant_scale.dtype != torch.float32
    ):
        raise ValueError(
            "C8 LIDU key_dequant_scale must be float32 [blocks,128,1]"
        )
    if (
        actual_seq_lengths_query.shape != (batch,)
        or req_pool_entries.shape != (batch,)
        or cache_tokens.shape != (batch,)
        or candidate_lens.shape != (batch,)
        or block_table.ndim != 2
        or block_table.shape[0] != batch
        or cache_slots_pool.ndim != 2
        or cache_slots_pool.shape[0] <= 0
        or block_table.shape[1] * BLOCK_SIZE != cache_slots_pool.shape[1]
    ):
        raise ValueError("C8 LIDU metadata shapes are inconsistent")
    for name, tensor in (
        ("actual_seq_lengths_query", actual_seq_lengths_query),
        ("req_pool_entries", req_pool_entries),
        ("cache_slots_pool", cache_slots_pool),
        ("cache_tokens", cache_tokens),
        ("candidate_lens", candidate_lens),
        ("block_table", block_table),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"C8 LIDU {name} must be int32")
    tensors = (
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        req_pool_entries,
        cache_slots_pool,
        cache_tokens,
        candidate_lens,
        block_table,
    )
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("C8 LIDU tensors must be on one device")
    if not query.is_contiguous() or any(
        not tensor.is_contiguous() for tensor in tensors
    ):
        raise ValueError("C8 LIDU tensors must be contiguous")


def _native_quant_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    op = getattr(torch_npu, "npu_quant_lightning_indexer", None)
    if op is None:
        namespace = getattr(torch.ops, "_C_ascend", None)
        op = (
            getattr(namespace, "npu_lightning_indexer_quant", None)
            if namespace is not None
            else None
        )
    if op is None:
        raise RuntimeError("A5 Quant LightningIndexer is not registered")
    result = op(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_dequant_scale,
        key_dequant_scale=key_dequant_scale,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=candidate_lens,
        block_table=block_table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=3,
    )
    topk = result[0] if isinstance(result, tuple) else result
    if not isinstance(topk, torch.Tensor) or topk.dtype != torch.int32:
        raise RuntimeError("A5 Quant LightningIndexer returned no int32 TopK")
    expected = int(query.shape[0]) * SPARSE_COUNT
    if topk.numel() != expected:
        raise RuntimeError(
            "A5 Quant LightningIndexer returned unexpected TopK shape: "
            f"shape={tuple(topk.shape)}, expected_numel={expected}"
        )
    return topk.reshape(query.shape[0], 1, SPARSE_COUNT).contiguous()


def _lidu_decode_update_c8_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_c8_lidu_inputs(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        req_pool_entries,
        cache_slots_pool,
        cache_tokens,
        candidate_lens,
        block_table,
    )
    topk = _native_quant_lightning_indexer(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        candidate_lens,
        block_table,
    )
    return lidu_cache_update(
        topk,
        req_pool_entries,
        cache_slots_pool,
        cache_tokens,
        candidate_lens,
    )


def _lidu_decode_update_c8_out_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_c8_lidu_inputs(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        req_pool_entries,
        cache_slots_pool,
        cache_tokens,
        candidate_lens,
        block_table,
    )
    topk = _native_quant_lightning_indexer(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        candidate_lens,
        block_table,
    )
    return lidu_cache_update_out(
        topk,
        req_pool_entries,
        cache_slots_pool,
        cache_tokens,
        candidate_lens,
        source_ids,
        destination_slots,
        miss_counts,
    )


def _lidu_decode_update_c8_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        req_pool_entries,
        cache_tokens,
        candidate_lens,
        block_table,
    )
    options = {"dtype": torch.int32, "device": query.device}
    return (
        torch.empty((query.shape[0], 1, SPARSE_COUNT), **options),
        torch.empty((query.shape[0], 1, SPARSE_COUNT), **options),
        torch.empty((query.shape[0],), **options),
        cache_slots_pool,
    )


def _lidu_decode_update_c8_out_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        req_pool_entries,
        cache_tokens,
        candidate_lens,
        block_table,
    )
    return source_ids, destination_slots, miss_counts, cache_slots_pool


_C8_LIDU_NPU_IMPL = torch.library.Library(
    "nanovllm_dsa", "IMPL", "PrivateUse1"
)
_C8_LIDU_NPU_IMPL.impl(
    "lidu_decode_update_c8", _lidu_decode_update_c8_impl
)
_C8_LIDU_NPU_IMPL.impl(
    "lidu_decode_update_c8_out", _lidu_decode_update_c8_out_impl
)
_C8_LIDU_META_IMPL = torch.library.Library("nanovllm_dsa", "IMPL", "Meta")
_C8_LIDU_META_IMPL.impl(
    "lidu_decode_update_c8", _lidu_decode_update_c8_meta
)
_C8_LIDU_META_IMPL.impl(
    "lidu_decode_update_c8_out", _lidu_decode_update_c8_out_meta
)


def _packed_byte_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (
        BLOCK_SIZE,
        1,
        PACKED_KV_DIM,
    ):
        raise ValueError(
            "packed C8 KV cache must have shape [blocks,128,1,656], "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.element_size() != 1:
        raise TypeError(
            "packed C8 KV cache must use a one-byte dtype, "
            f"got {tensor.dtype}"
        )
    if not tensor.is_contiguous():
        raise ValueError("packed C8 KV cache must be contiguous")
    return tensor if tensor.dtype == torch.int8 else tensor.view(torch.int8)


def _packed_value_view(tensor: torch.Tensor) -> torch.Tensor:
    """Legacy diagnostic helper retained for bisecting older A5 runs.

    The supported Ascend 950 QSFA MLA-absorb contract uses the same packed
    656-byte KV tensor for both key and value. Production A5 QSFA must not use
    this 512-byte prefix view; keeping it here avoids changing unrelated debug
    imports while making the production call follow the upstream A5 adaptor.
    """

    _packed_byte_view(tensor)
    value = tensor[..., :CKV_DIM]
    if tuple(value.shape[1:]) != (BLOCK_SIZE, 1, CKV_DIM):
        raise RuntimeError(
            "A5 C8 diagnostic value view must be [blocks,128,1,512], "
            f"got {tuple(value.shape)}"
        )
    return value


def _check_c8_attention_inputs(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    sparse_and_tail_slots: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
) -> None:
    if query.ndim != 3 or query.shape[-1] != 576:
        raise ValueError(
            "C8 QSFA query must be TND [T,Q_HEAD,576], "
            f"got {tuple(query.shape)}"
        )
    if not 1 <= query.shape[1] <= 64:
        raise ValueError("C8 QSFA supports 1 <= Q_HEAD <= 64")
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("C8 QSFA query must be bf16 or fp16")
    _packed_byte_view(packed_kv)
    if (
        sparse_and_tail_slots.ndim != 3
        or sparse_and_tail_slots.shape[0] != query.shape[0]
        or sparse_and_tail_slots.shape[1] != 1
        or sparse_and_tail_slots.shape[2] < SPARSE_COUNT
        or sparse_and_tail_slots.dtype != torch.int32
    ):
        raise ValueError(
            "C8 QSFA sparse_and_tail_slots must be int32 "
            "[T,1,2048+max_tail_tokens]"
        )
    batch = actual_seq_lengths_query.numel()
    if (
        actual_seq_lengths_query.ndim != 1
        or resident_seq_lengths.ndim != 1
        or resident_seq_lengths.numel() != batch
        or block_table.ndim != 2
        or block_table.shape[0] != batch
    ):
        raise ValueError("C8 QSFA batch metadata shapes are inconsistent")
    for name, tensor in (
        ("block_table", block_table),
        ("actual_seq_lengths_query", actual_seq_lengths_query),
        ("resident_seq_lengths", resident_seq_lengths),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"C8 QSFA {name} must be int32")


def _resident_c8_qsfa():
    """Return the repository-pinned A5 resident C8 QSFA binding."""

    op = getattr(
        torch.ops._C_ascend,
        "npu_dsa_a5_sparse_tail_attention_c8",
        None,
    )
    if op is None:
        raise RuntimeError(
            "A5 _C_ascend.npu_dsa_a5_sparse_tail_attention_c8 is not registered"
        )
    return op


@torch.library.custom_op(
    "nanovllm_dsa::sparse_and_tail_attention_c8",
    mutates_args=(),
)
def _sparse_and_tail_attention_c8(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    sparse_and_tail_slots: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    scale_value: float,
) -> torch.Tensor:
    # The A5 QSFA TND kernel reads query as a dense row-major tensor. Match
    # the reference adaptor even when a caller passes a strided query view.
    query = query.contiguous()
    _check_c8_attention_inputs(
        query,
        packed_kv,
        sparse_and_tail_slots,
        block_table,
        actual_seq_lengths_query,
        resident_seq_lengths,
    )

    # Pin resident C8 decode to the repository kernel used by the reference
    # implementation. Its eager ACLNN adapter owns the scale placeholders and
    # the fixed A5 quant/attention attributes.
    op = _resident_c8_qsfa()
    result = op(
        query,
        packed_kv,
        sparse_and_tail_slots,
        float(scale_value),
        block_table,
        actual_seq_lengths_query,
        resident_seq_lengths,
    )
    attention_out = result[0] if isinstance(result, tuple) else result
    if not isinstance(attention_out, torch.Tensor):
        raise RuntimeError("A5 resident C8 QSFA returned no attention tensor")
    expected_shape = (*query.shape[:-1], CKV_DIM)
    if tuple(attention_out.shape) != expected_shape:
        raise RuntimeError(
            "A5 resident C8 QSFA returned unexpected attention shape: "
            f"shape={tuple(attention_out.shape)}, expected={expected_shape}"
        )
    return attention_out


@_sparse_and_tail_attention_c8.register_fake
def _sparse_and_tail_attention_c8_fake(
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    sparse_and_tail_slots: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    scale_value: float,
) -> torch.Tensor:
    del (
        packed_kv,
        sparse_and_tail_slots,
        block_table,
        actual_seq_lengths_query,
        resident_seq_lengths,
        scale_value,
    )
    if query.ndim != 3 or query.shape[-1] != 576:
        raise ValueError("C8 QSFA query must be [T,Q_HEAD,576]")
    if not 1 <= query.shape[1] <= 64:
        raise ValueError("C8 QSFA supports 1 <= Q_HEAD <= 64")
    return query.new_empty((*query.shape[:-1], CKV_DIM))


def local_opapi_path() -> str:
    return _OPAPI


from vllm_ascend.dsa_offload.a5_c8_precision_policy import (
    install_a5_c8_precision_policy,
)

install_a5_c8_precision_policy()

__all__ = ["local_opapi_path"]
