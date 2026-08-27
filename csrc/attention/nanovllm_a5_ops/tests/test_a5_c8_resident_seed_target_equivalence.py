#!/usr/bin/env python3
"""Strict target-NPU equivalence gate for isolated A5C8ResidentSeed.

This deliberately stops before the P3 raw-Q/K scan or any H2D route.  It
loads only the source-matched official C8 OPP and the independently built
ResidentSeed OPP, then calls both with the *same* resident FP8 index view.

Official C8 does not export its final sortable K-th score in its public ABI.
Accordingly, this program proves the device-visible selection equality and
the ResidentSeed output ABI.  The later P4 candidate audit is the functional
proof that the emitted threshold covers official cold Top-K sources.
"""

from __future__ import annotations

import os

import torch

from fused_streaming_quant_li_get_audit_adapter import (
    RESIDENT_BLOCKS,
    RESIDENT_SEED_OPP_ENV,
    _bind_resident_seed_opp,
    _build_resident_index_view,
    _call_official_resident_reference,
    _launch_resident_seed,
    _resolve_resident_seed_opp,
)
from streaming_quant_li_get_target_adapter import (
    REFERENCE_OPP_ENV,
    _build_official_inputs,
    _official_metadata,
    _prepend_unique_path,
    _require_a5_npu,
    _require_source_matched_official_c8_ops,
    _resolve_single_opp,
)


def _cpu_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        torch.equal(left.detach().cpu().contiguous(), right.detach().cpu().contiguous())
    )


def main() -> int:
    reference_opapi, reference_vendor = _resolve_single_opp(
        os.getenv(REFERENCE_OPP_ENV), "reference"
    )
    seed_opapi, seed_vendor = _resolve_resident_seed_opp(
        os.getenv(RESIDENT_SEED_OPP_ENV)
    )
    # Both vendor paths are installed before importing either typed bridge.
    # This stays process-local and does not select a vLLM product route.
    _prepend_unique_path("ASCEND_CUSTOM_OPP_PATH", (reference_vendor, seed_vendor))
    official, metadata, _ = _require_source_matched_official_c8_ops(
        reference_opapi, reference_vendor
    )
    device = _require_a5_npu()
    inputs = _build_official_inputs(device)
    inputs["metadata"] = _official_metadata(metadata, inputs)
    resident = _build_resident_index_view(inputs, metadata)

    print(
        "A5_C8_RESIDENT_SEED_TARGET_STAGE "
        "stage=resident_view_ready "
        f"key_shape={tuple(inputs['key'].shape)} "
        f"full_block_table_shape={tuple(inputs['block_table'].shape)} "
        "resident_block_table_shape="
        f"{tuple(resident['resident_index_block_table'].shape)} "
        "resident_seq_lengths="
        f"{resident['resident_seq_lengths'].detach().cpu().tolist()}",
        flush=True,
    )
    print(
        "A5_C8_RESIDENT_SEED_TARGET_STAGE "
        "stage=resident_stride_contract "
        "key_stride="
        f"{int(resident['resident_index_key'].stride(0))} "
        "scale_stride="
        f"{int(resident['resident_index_key_dequant_scale'].stride(0))}",
        flush=True,
    )
    _bind_resident_seed_opp(seed_opapi)
    print("A5_C8_RESIDENT_SEED_TARGET_STAGE stage=seed_launch", flush=True)
    seed_indices, seed_values, resident_tau, tau_valid = _launch_resident_seed(
        inputs, resident
    )
    torch.npu.synchronize()
    print("A5_C8_RESIDENT_SEED_TARGET_STAGE stage=seed_complete", flush=True)
    print("A5_C8_RESIDENT_SEED_TARGET_STAGE stage=official_launch", flush=True)
    official_indices, official_values = _call_official_resident_reference(
        official, inputs, resident
    )
    torch.npu.synchronize()
    print("A5_C8_RESIDENT_SEED_TARGET_STAGE stage=official_complete", flush=True)

    if not _cpu_equal(seed_indices, official_indices):
        raise RuntimeError(
            "A5C8ResidentSeed sparse_indices are not bitwise equal to "
            "source-matched official C8 on the same resident FP8 index view"
        )
    if not _cpu_equal(seed_values, official_values):
        raise RuntimeError(
            "A5C8ResidentSeed return_values=False output is not bitwise equal "
            "to source-matched official C8"
        )
    if seed_values.dtype != torch.float32 or seed_values.numel() != 0:
        raise RuntimeError(
            "A5C8ResidentSeed return_values=False ABI must produce an empty "
            f"FP32 tensor, got shape={tuple(seed_values.shape)} dtype={seed_values.dtype}"
        )
    tau_cpu = resident_tau.detach().cpu().contiguous()
    valid_cpu = tau_valid.detach().cpu().contiguous()
    if not bool(torch.all(valid_cpu == 1)):
        raise RuntimeError(
            "resident fixture did not produce a valid exact Top-2048 threshold "
            "for every row"
        )
    if bool(torch.any(torch.bitwise_right_shift(tau_cpu, 16) != 0)):
        raise RuntimeError(
            "resident_tau must carry the sortable C8 threshold only in its "
            "low 16 bits"
        )

    tau_values = tau_cpu.tolist()
    valid_values = valid_cpu.tolist()
    print(
        "A5_C8_RESIDENT_SEED_TARGET_CONFIG "
        f"device={device} physical_npu={device.index} "
        f"reference_opapi={reference_opapi} resident_seed_opapi={seed_opapi} "
        f"resident_blocks={RESIDENT_BLOCKS}",
        flush=True,
    )
    print(
        "A5_C8_RESIDENT_SEED_TARGET_RESULT "
        "resident_indices=bitwise_equal "
        "sparse_values=bitwise_equal_empty_fp32_return_values_false "
        "resident_tau_encoding=sortable_uint16_low_bits "
        f"resident_tau={tau_values} tau_valid={valid_values} "
        "official_tau=not_exported "
        "threshold_functional_proof=next_p4_candidate_audit",
        flush=True,
    )
    print("A5_C8_RESIDENT_SEED_TARGET_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
