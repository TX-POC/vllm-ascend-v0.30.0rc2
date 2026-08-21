# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness guards for the Ascend A5 packed-C8 DSA data path.

The weight guard is called directly by ``AscendSFAImpl`` before MLAPO weight
processing. Runtime operator patches are installed later from ``a5_extension``
after the normal vLLM-Ascend modules are imported. Together they fix two
A5-only correctness hazards:

* W4A4 MXFP4 linear weights must not be reinterpreted by the float MLAPO
  weight path. For those layers use the existing native linear path and build
  the exact 656-byte packed KV row with ``custom_kv_rmsnorm_rope``.
* The packed KV tensor uses an FP8 dtype as a byte container; the last 144
  bytes contain BF16 RoPE and FP32 scales. QSFA receives the original storage
  directly so those payload bytes remain bit-for-bit without a hot-path copy.
"""

from __future__ import annotations

from typing import Any

import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

_INSTALLED = False
_PACKED_KV_DIM = 656


def _layer_quant_method(layer: Any | None) -> Any | None:
    linear_method = getattr(layer, "quant_method", None)
    return getattr(linear_method, "quant_method", linear_method)


def _is_w4a4_mxfp4(quant_method: Any | None) -> bool:
    if quant_method is None:
        return False
    from vllm_ascend.quantization.methods import (
        AscendW4A4MXFP4DynamicLinearMethod,
        AscendW4A4MXFP4FlatQuantDynamicLinearMethod,
    )

    return isinstance(
        quant_method,
        (
            AscendW4A4MXFP4DynamicLinearMethod,
            AscendW4A4MXFP4FlatQuantDynamicLinearMethod,
        ),
    )


def _target_quant_methods(impl: Any) -> tuple[Any | None, Any | None]:
    return (
        _layer_quant_method(getattr(impl, "fused_qkv_a_proj", None)),
        _layer_quant_method(getattr(impl, "q_proj", None)),
    )


def _is_target_layer(impl: Any) -> bool:
    return bool(
        get_ascend_device_type() == AscendDeviceType.A5
        and get_ascend_config().dsa_offload_config.enabled
        and getattr(impl, "enable_sparse_sfa_c8", False)
    )


def apply_a5_c8_precision_guard(impl: Any) -> bool:
    """Select the native packed producer before SFA mutates linear weights."""

    if not _is_target_layer(impl):
        return False

    # W4A4 MXFP4 stores packed FP4 weights + MX scales. The A5 float MLAPO
    # path transposes those packed bytes in place, so this guard must run at
    # the beginning of process_weights_after_loading(), not only when the A5
    # operator extension is loaded later by DSAOffloadRuntime.
    impl.enable_mlapo = False
    impl.enable_sfa_prolog_v3 = False
    impl._a5_c8_force_native_packed = True
    impl.c8_k_cache_dtype = torch.float8_e4m3fn
    impl.c8_k_scale_cache_dtype = torch.float32

    fused_quant, q_quant = _target_quant_methods(impl)
    logger.warning_once(
        "[DSA_A5_C8_PRECISION] layer=%s producer=native_packed "
        "mlapo=false fused_quant=%s q_quant=%s "
        "reason=w4a4_mxfp4_weight_contract",
        getattr(impl, "layer_name", None),
        type(fused_quant).__name__ if fused_quant is not None else "None",
        type(q_quant).__name__ if q_quant is not None else "None",
    )
    return True


def _pack_c8_rows_as_bytes(
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    knope_scale: torch.Tensor,
    row_count: int,
) -> torch.Tensor:
    """Pack C8 latent, BF16 RoPE and FP32 scales without dtype conversion."""

    if row_count <= 0:
        raise RuntimeError(
            f"A5 C8 native producer requires at least one row, got {row_count}"
        )

    byte_parts = []
    for name, tensor in (
        ("k_nope", k_nope),
        ("k_pe", k_pe),
        ("knope_scale", knope_scale),
    ):
        if tensor.numel() % row_count != 0:
            raise RuntimeError(
                "A5 C8 native producer cannot split tensor into token rows: "
                f"name={name}, shape={tuple(tensor.shape)}, rows={row_count}"
            )
        byte_parts.append(
            tensor.contiguous().view(torch.int8).view(row_count, -1)
        )

    packed_bytes = torch.cat(byte_parts, dim=-1)
    if int(packed_bytes.shape[-1]) != _PACKED_KV_DIM:
        raise RuntimeError(
            "A5 C8 native producer generated an invalid packed byte row: "
            f"shape={tuple(packed_bytes.shape)}, expected={_PACKED_KV_DIM}"
        )
    return packed_bytes


def sparse_flash_attention_for_offload_c8(
    *,
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    attention_slots: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    scale_value: float,
) -> torch.Tensor:
    """Run C8 QSFA against the canonical byte-packed KV storage."""

    if packed_kv.element_size() != 1:
        raise RuntimeError(
            "A5 C8 QSFA packed cache must be byte-addressable: "
            f"dtype={packed_kv.dtype}"
        )

    return torch.ops.nanovllm_dsa.sparse_and_tail_attention_c8(
        query,
        packed_kv,
        attention_slots,
        block_table,
        actual_seq_lengths_query,
        resident_seq_lengths,
        float(scale_value),
    )


def install_a5_c8_precision_policy() -> None:
    """Install narrow A5 C8 correctness patches once per Python process."""

    global _INSTALLED
    if _INSTALLED:
        return

    import vllm_ascend.attention.sfa_v1 as sfa_mod
    import vllm_ascend.dsa_offload.ops_nanovllm as ops_mod

    AscendSFAImpl = sfa_mod.AscendSFAImpl
    original_process_weights = AscendSFAImpl.process_weights_after_loading
    original_exec_kv = AscendSFAImpl.exec_kv

    def process_weights_after_loading(
        self: Any,
        act_dtype: torch.dtype,
    ) -> Any:
        apply_a5_c8_precision_guard(self)
        return original_process_weights(self, act_dtype)

    def exec_kv(
        self: Any,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata: Any,
    ) -> Any:
        if not getattr(self, "_a5_c8_force_native_packed", False):
            return original_exec_kv(
                self,
                kv_no_split,
                cos,
                sin,
                kv_cache,
                slots,
                attn_metadata,
            )

        # Without an Indexer, the existing native A5 C8 path already calls
        # custom_kv_rmsnorm_rope and performs the final packed scatter in
        # AscendSFAImpl.forward(). Only A5+Indexer needs the missing bridge.
        if not getattr(self, "has_indexer", False):
            return original_exec_kv(
                self,
                kv_no_split,
                cos,
                sin,
                kv_cache,
                slots,
                attn_metadata,
            )

        # DSA-CP already has its own byte-packed gather path; keep this patch
        # scoped to the non-CP A5 DSA path currently under investigation.
        if getattr(self, "enable_dsa_cp", False):
            return original_exec_kv(
                self,
                kv_no_split,
                cos,
                sin,
                kv_cache,
                slots,
                attn_metadata,
            )

        if len(kv_cache) < 1:
            raise RuntimeError("A5 C8 native packed producer has no KV cache")
        packed_cache = kv_cache[0]
        if (
            packed_cache.dtype != torch.float8_e4m3fn
            or packed_cache.ndim != 4
            or int(packed_cache.shape[-1]) != _PACKED_KV_DIM
        ):
            raise RuntimeError(
                "A5 C8 native producer requires [blocks,128,1,656] FP8 cache: "
                f"shape={tuple(packed_cache.shape)}, dtype={packed_cache.dtype}"
            )

        assert self.kv_a_layernorm is not None
        batch = int(kv_no_split.shape[0])
        kv = kv_no_split.view(
            batch,
            self.num_kv_heads,
            1,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )
        k_pe, k_nope, knope_scale = sfa_mod.custom_kv_rmsnorm_rope(
            kv,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            epsilon=self.kv_a_layernorm.variance_epsilon,
            dst_type=torch.float8_e4m3fn,
            tile_size=self.sfa_qsfa_tile_size,
        )
        row_count = int(slots.numel())
        packed_bytes = _pack_c8_rows_as_bytes(
            k_nope,
            k_pe,
            knope_scale,
            row_count,
        )

        # Scatter the full 512B FP8 latent + 128B BF16 RoPE bytes + 16B FP32
        # scale bytes in one 656-byte row. Keeping has_indexer unchanged is
        # important: the independent Indexer cache is still produced/consumed
        # by the normal SFA path.
        torch_npu.npu_scatter_nd_update_(
            packed_cache.view(torch.int8).view(-1, _PACKED_KV_DIM),
            slots.reshape(-1, 1),
            packed_bytes,
        )
        logger.info_once(
            "[DSA_A5_C8_PRECISION] layer=%s packed_producer_write=true "
            "packed_shape=%s packed_dtype=%s packed_stride=%s",
            getattr(self, "layer_name", None),
            tuple(packed_cache.shape),
            packed_cache.dtype,
            tuple(packed_cache.stride()),
        )
        return k_pe, k_nope, knope_scale

    AscendSFAImpl.process_weights_after_loading = process_weights_after_loading
    AscendSFAImpl.exec_kv = exec_kv
    ops_mod.sparse_flash_attention_for_offload_c8 = (
        sparse_flash_attention_for_offload_c8
    )
    _INSTALLED = True
