#!/usr/bin/env python3
"""Target runner for the P4.1 non-authoritative fused audit harness.

It launches one source-derived raw-Q/K C8 clone with the P4 audit lease
sentinel.  The clone computes its own final Top-K and records score-qualified
cold sources in a fixed [R,S] candidate map, but the test treats those Top-K
indices as non-authoritative.  A source-matched official C8 call is made only
as the strict mathematical reference.
"""

from __future__ import annotations

import os
from time import monotonic
from typing import Any

import torch

from streaming_quant_li_get_target_adapter import (
    BLOCK_SIZE,
    BATCH_SIZE,
    HOOK_CONTROL_MODE_ALL_BUSY,
    INT64_MAX,
    REFERENCE_OPP_ENV,
    SPARSE_COUNT,
    TARGET_OPP_ENV,
    _bind_isolated_target_opp,
    _build_official_inputs,
    _call_official,
    _make_no_post_transaction,
    _official_metadata,
    _prepend_unique_path,
    _require_a5_npu,
    _require_source_matched_official_c8_ops,
    _resolve_single_opp,
)


def _cpu_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(left.detach().cpu().contiguous(), right.detach().cpu().contiguous()))


def _audit_receipt(
    result: dict[str, Any],
    transaction: dict[str, torch.Tensor],
) -> dict[str, int]:
    """Validate audit's side-effect boundary and count fixed-map records."""
    marker = int(result["candidate_marker"])
    sentinel = int(result["audit_lease_sentinel"])
    source_map = result["candidate_source_to_slot"].detach().cpu().contiguous()
    queue_leases = result["queue_leases"].detach().cpu().contiguous()
    ready = result["private_ready_table"].detach().cpu().contiguous()
    arena_head = result["private_arena_head"].detach().cpu().contiguous()
    packed = result["private_packed_output"].detach().cpu().contiguous()
    telemetry = result["telemetry"].detach().cpu().contiguous()

    if not bool(torch.all(queue_leases == sentinel)):
        raise RuntimeError("P4 audit changed the all-sentinel queue lease table")
    if int(torch.count_nonzero(ready).item()) != 0:
        raise RuntimeError("P4 audit wrote the private ready table")
    if int(torch.count_nonzero(arena_head).item()) != 0:
        raise RuntimeError("P4 audit allocated a private HBM arena slot")
    if not bool(torch.all(packed == 0xA5)):
        raise RuntimeError("P4 audit wrote the private packed payload arena")
    if int(torch.count_nonzero(telemetry).item()) != 0:
        raise RuntimeError("P4 audit wrote queue-owned transport telemetry")
    candidate_count = int(torch.count_nonzero(source_map == marker).item())
    if candidate_count <= 0:
        raise RuntimeError("P4 audit produced no score-time candidate markers")
    if bool(torch.any((source_map != -1) & (source_map != marker))):
        raise RuntimeError("P4 audit produced a non-candidate private-slot value")
    return {
        "candidate_count": candidate_count,
        "candidate_overflow": 0,
        "candidate_marker": marker,
        "lease_sentinel": sentinel,
    }


def _assert_clone_candidate_coverage(
    clone_indices: torch.Tensor,
    candidate_map: torch.Tensor,
    score_source_lens: torch.Tensor,
    marker: int,
) -> tuple[int, int]:
    """Check coverage only for the clone's own result, never official Top-K."""
    indices = clone_indices.detach().cpu().contiguous().reshape(BATCH_SIZE, SPARSE_COUNT)
    candidates = candidate_map.detach().cpu().contiguous()
    lengths = score_source_lens.detach().cpu().contiguous()
    selected = 0
    missing = 0
    for row in range(BATCH_SIZE):
        valid = int(lengths[row].item())
        for source in indices[row].tolist():
            if source < 0:
                continue
            if source >= valid:
                raise RuntimeError(
                    f"clone Top-K produced source outside its score length: row={row} "
                    f"source={source} valid={valid}"
                )
            selected += 1
            if int(candidates[row, source].item()) != marker:
                missing += 1
    if missing:
        raise RuntimeError(
            "P4 audit candidate map does not cover clone-selected cold sources: "
            f"selected={selected} missing={missing}"
        )
    return selected, missing


def run_p4_fused_audit() -> dict[str, Any]:
    """Run one P4.1 clone scan and return official/clone outputs for a gate."""
    reference_opapi, reference_vendor = _resolve_single_opp(
        os.getenv(REFERENCE_OPP_ENV), "reference"
    )
    target_opapi, target_vendor = _resolve_single_opp(
        os.getenv(TARGET_OPP_ENV), "target"
    )
    _prepend_unique_path("ASCEND_CUSTOM_OPP_PATH", (reference_vendor, target_vendor))
    official, metadata, _ = _require_source_matched_official_c8_ops(
        reference_opapi, reference_vendor
    )
    device = _require_a5_npu()
    inputs = _build_official_inputs(device)
    inputs["metadata"] = _official_metadata(metadata, inputs)
    official_indices = _call_official(official, inputs)
    torch.npu.synchronize()

    # The P4 bridge loads the existing P3 typed ACLNN resolver after its
    # target opapi path has been set.  It does not load vllm_ascend.
    _bind_isolated_target_opp(target_opapi)
    from nanovllm_p4_fused_streaming_li_get import (  # noqa: PLC0415
        AUDIT_QUEUE_LEASE_SENTINEL,
        launch,
    )

    transaction = _make_no_post_transaction(
        device, BATCH_SIZE, HOOK_CONTROL_MODE_ALL_BUSY
    )
    # This first execution deliberately uses the strongest no-false-negative
    # fixture: tau=0 and every source is nonresident.  It makes every valid
    # source score-qualified, so the fixed-map receipt can prove that the
    # score-time write path covers the clone's selected sources.  It does not
    # claim that zero is a useful production prefetch threshold; production
    # will feed the resident-seed estimate here.
    if not bool(torch.all(transaction["resident_threshold"] == 0)):
        raise RuntimeError("P4 audit fixture requires tau=0")
    if not bool(torch.all(transaction["resident_threshold_valid"] == 1)):
        raise RuntimeError("P4 audit fixture requires valid resident tau rows")
    if not bool(torch.all(transaction["kv_object_slots_snapshot"] < 0)):
        raise RuntimeError("P4 audit fixture requires every source to be cold")
    # The sentinel is a graph input value, deliberately installed before the
    # one cloned LI launch.  There is no second scan or host-side candidate
    # reconstruction.
    transaction["queue_leases"].fill_(AUDIT_QUEUE_LEASE_SENTINEL)
    torch.npu.synchronize()

    print(
        "P4_FUSED_AUDIT_CONFIG "
        f"device={device} physical_npu={device.index} "
        f"reference_opapi={reference_opapi} target_opapi={target_opapi} "
        "raw_qk_scan_count=1 clone_topk_authority=0 "
        "candidate_output=private_source_to_slot[R,S] "
        "mode=audit threshold_fixture=zero_all_cold_coverage "
        "h2d_result=not_exercised",
        flush=True,
    )
    started = monotonic()
    result = launch(
        inputs,
        transaction,
        mode="audit",
        kv_object_domain=0,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        cmp_ratio=1,
        return_values=False,
        stride=int(inputs["key"].stride(0)),
        scale_stride=int(inputs["key_dequant_scale"].stride(0)),
        kvaf_rows_per_entry=BLOCK_SIZE,
    )
    torch.npu.synchronize()
    elapsed_us = (monotonic() - started) * 1_000_000.0

    receipt = _audit_receipt(result, transaction)
    score_source_lens = result["score_source_lens"].detach().cpu().contiguous()
    print(
        "P4_FUSED_AUDIT_MARKERS "
        f"candidate_marker={receipt['candidate_marker']} "
        f"candidate_count={receipt['candidate_count']} "
        f"candidate_overflow={receipt['candidate_overflow']} "
        f"queue_lease_sentinel={receipt['lease_sentinel']} "
        f"score_source_lens={score_source_lens.tolist()} "
        "doorbell_posted=0 private_payload_unchanged=1 telemetry_unchanged=1 "
        f"elapsed_us={elapsed_us:.3f}",
        flush=True,
    )
    selected, missing = _assert_clone_candidate_coverage(
        result["non_authoritative_sparse_indices"],
        result["candidate_source_to_slot"],
        score_source_lens,
        receipt["candidate_marker"],
    )
    print(
        "P4_FUSED_AUDIT_RECEIPT "
        f"clone_selected={selected} clone_missing={missing} "
        "candidate_coverage=clone_only h2d_result=not_exercised",
        flush=True,
    )
    return {
        "official_indices": official_indices,
        "clone_indices": result["non_authoritative_sparse_indices"],
        "candidate_count": receipt["candidate_count"],
        "candidate_marker": receipt["candidate_marker"],
    }


__all__ = ["run_p4_fused_audit"]
