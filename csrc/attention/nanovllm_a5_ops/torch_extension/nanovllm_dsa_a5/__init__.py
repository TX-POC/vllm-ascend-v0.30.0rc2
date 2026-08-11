from __future__ import annotations

import os
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_OPP = _ROOT / "_custom_opp"
_OPAPI_LIBS = tuple(
    (_LOCAL_OPP / "vendors").glob("*/op_api/lib/libcust_opapi.so")
)
if len(_OPAPI_LIBS) != 1:
    raise RuntimeError(
        "nanovllm_dsa_a5 requires exactly one repository-local "
        "libcust_opapi.so. Run bash build.sh first."
    )
_OPAPI = _OPAPI_LIBS[0]
_VENDOR = _OPAPI.parents[2]

_existing = [
    value
    for value in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(":")
    if value
]
_vendor_str = str(_VENDOR)
if _vendor_str not in _existing:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(
        [_vendor_str, *_existing]
    )
os.environ.setdefault(
    "NANOVLLM_A5_INSTALL_OPP_PATH",
    str(_LOCAL_OPP),
)
os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(_OPAPI)

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

from . import _C  # noqa: E402,F401


BLOCK_SIZE = 128
PACKED_KV_DIM = 656
SPARSE_COUNT = 2048

lidu_decode_update = torch.ops.nanovllm_dsa.lidu_decode_update
lidu_decode_update_out = torch.ops.nanovllm_dsa.lidu_decode_update_out
lidu_cache_update = torch.ops.nanovllm_dsa.lidu_cache_update
lidu_cache_update_out = torch.ops.nanovllm_dsa.lidu_cache_update_out
scatter_copy = torch.ops.nanovllm_dsa.scatter_copy
packed_scatter_copy = torch.ops.nanovllm_dsa.packed_scatter_copy
packed_scatter_copy_out = torch.ops.nanovllm_dsa.packed_scatter_copy_out
sparse_and_tail_attention = (
    torch.ops.nanovllm_dsa.sparse_and_tail_attention
)


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
        or block_table.shape[1] * BLOCK_SIZE
        != cache_slots_pool.shape[1]
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
    # Exact A5 C8 call used by vLLM-Ascend's A5DeviceAdaptor. The caller owns
    # GLM preprocessing: RoPE, normalized 128x128 Hadamard, then FP8 E4M3
    # dynamic quantization and its FP32 dequant scales.
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
        raise RuntimeError(
            "official A5 C8 LightningIndexer returned no int32 top-k tensor"
        )
    if topk.numel() != query.shape[0] * SPARSE_COUNT:
        raise RuntimeError(
            "official A5 C8 LightningIndexer returned an unexpected shape: "
            f"{tuple(topk.shape)}"
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

lidu_decode_update_c8 = torch.ops.nanovllm_dsa.lidu_decode_update_c8
lidu_decode_update_c8_out = torch.ops.nanovllm_dsa.lidu_decode_update_c8_out


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


def packed_kvcache_scatter_copy(
    hbm_packed_kv: torch.Tensor,
    dram_packed_kv: torch.Tensor,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    source_token_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    copy_counts: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    max_tail_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy complete packed C8 rows and publish topK+tail QSFA metadata."""

    if hbm_packed_kv.dtype != dram_packed_kv.dtype:
        raise TypeError("HBM and DRAM packed KV dtypes must match")
    hbm_bytes = _packed_byte_view(hbm_packed_kv)
    dram_bytes = _packed_byte_view(dram_packed_kv)
    _, attention_slots, resident_seq_lengths = packed_scatter_copy(
        hbm_bytes,
        dram_bytes,
        hbm_block_table,
        dram_block_table,
        source_token_ids,
        destination_slots,
        copy_counts,
        cache_tokens,
        candidate_lens,
        actual_seq_lengths_kv,
        int(max_tail_tokens),
    )
    return hbm_packed_kv, attention_slots, resident_seq_lengths


def _native_qsfa():
    namespace = getattr(torch.ops, "_C_ascend", None)
    op = (
        getattr(namespace, "npu_kv_quant_sparse_flash_attention", None)
        if namespace is not None
        else None
    )
    if op is not None:
        return op, True
    op = getattr(torch_npu, "npu_kv_quant_sparse_flash_attention", None)
    if op is None:
        raise RuntimeError(
            "A5 npu_kv_quant_sparse_flash_attention is not registered"
        )
    return op, False


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
    """Call native A5 C8 QSFA with GLM-5.1 packed-MLA attributes."""

    _check_c8_attention_inputs(
        query,
        packed_kv,
        sparse_and_tail_slots,
        block_table,
        actual_seq_lengths_query,
        resident_seq_lengths,
    )
    op, supports_lse = _native_qsfa()
    kwargs = dict(
        query=query,
        key=packed_kv,
        value=packed_kv,
        sparse_indices=sparse_and_tail_slots,
        scale_value=float(scale_value),
        key_quant_mode=2,
        value_quant_mode=2,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=resident_seq_lengths,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    if supports_lse:
        kwargs["return_softmax_lse"] = False
    result = op(**kwargs)
    return result[0] if isinstance(result, tuple) else result


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
    return query.new_empty((*query.shape[:-1], 512))


sparse_and_tail_attention_c8 = (
    torch.ops.nanovllm_dsa.sparse_and_tail_attention_c8.default
)


def local_opapi_path() -> str:
    return str(_OPAPI)


__all__ = [
    "lidu_decode_update",
    "lidu_decode_update_out",
    "lidu_cache_update",
    "lidu_cache_update_out",
    "lidu_decode_update_c8",
    "lidu_decode_update_c8_out",
    "scatter_copy",
    "packed_scatter_copy",
    "packed_scatter_copy_out",
    "packed_kvcache_scatter_copy",
    "sparse_and_tail_attention",
    "sparse_and_tail_attention_c8",
    "local_opapi_path",
]
