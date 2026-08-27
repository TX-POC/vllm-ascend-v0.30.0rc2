#!/usr/bin/env python3
"""Run the P3 raw-Q/K scanner as a non-authoritative advisory mirror.

The strict P3 matrix established an important boundary: compiling the KVAF
hook into a clone of official C8 can change that clone's final Top-K output.
It must therefore never provide the indices consumed by QSFA.  This adapter
checks the safe architecture instead:

* an unchanged, source-matched official C8 invocation is the only exact
  producer;
* the hook-on C8 clone scans the *same physical Q/K inputs* on another stream
  only to exercise the raw-Q/K score-to-prefetch seam; and
* the official C8 result is bitwise unchanged while that advisory mirror is
  in flight.

The first gate deliberately uses ``reject_all_busy``.  It proves that the
mirror binary contains the hook and reaches its control validation path, but
cannot construct a fabricated Direct-HCCP request.  Thus this is an
architecture-isolation test, not a claim of H2D payload correctness or
end-to-end performance.
"""

from __future__ import annotations

import importlib
import os
import sys
from time import monotonic
from typing import Any

import torch
import torch_npu  # noqa: F401

from streaming_quant_li_get_target_adapter import (
    HOOK_CONTROL_MODE_REJECT_BUSY,
    REFERENCE_OPP_ENV,
    TARGET_OPP_ENV,
    _bind_isolated_target_opp,
    _build_official_inputs,
    _call_official,
    _make_no_post_transaction,
    _official_metadata,
    _prepend_unique_path,
    _prove_safe_hook_control,
    _require_a5_npu,
    _require_source_matched_official_c8_ops,
    _resolve_single_opp,
    _snapshot_no_post_transaction,
)


def _require_fresh_target_bridge() -> Any:
    if "nanovllm_streaming_quant_li_get" in sys.modules:
        raise RuntimeError(
            "run advisory-mirror isolation in a fresh Python process: the "
            "isolated target bridge resolver was already initialized"
        )
    importlib.import_module("nanovllm_streaming_quant_li_get")
    namespace = getattr(torch.ops, "a5_streaming_quant_li_get", None)
    mirror = (
        getattr(namespace, "streaming_quant_li_get", None)
        if namespace is not None
        else None
    )
    if mirror is None:
        raise RuntimeError(
            "isolated StreamingQuantLiGet bridge did not register "
            "torch.ops.a5_streaming_quant_li_get.streaming_quant_li_get"
        )
    return mirror


def _call_advisory_mirror(
    mirror: Any,
    inputs: dict[str, torch.Tensor],
    transaction: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Launch the raw-Q/K mirror; its Top-K output is deliberately ignored."""
    outputs = mirror(
        inputs["query"],
        inputs["key"],
        inputs["weights"],
        inputs["query_dequant_scale"],
        inputs["key_dequant_scale"],
        inputs["actual_seq_lengths_query"],
        inputs["actual_seq_lengths_key"],
        inputs["block_table"],
        inputs["metadata"],
        transaction["resident_threshold"],
        transaction["resident_threshold_valid"],
        transaction["kv_object_slots_snapshot"],
        transaction["req_pool_entries"],
        transaction["kv_object_map"],
        transaction["private_source_to_slot"],
        transaction["private_ready_table"],
        transaction["private_arena_head"],
        transaction["private_packed_output"],
        transaction["queue_control"],
        transaction["queue_leases"],
        transaction["kv_object_lens"],
        kv_object_domain=0,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=2048,
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        cmp_ratio=1,
        return_values=False,
        stride=int(inputs["key"].stride(0)),
        scale_stride=int(inputs["key_dequant_scale"].stride(0)),
        kvaf_rows_per_entry=128,
    )
    if not isinstance(outputs, tuple) or len(outputs) != 9:
        raise RuntimeError(
            "advisory mirror did not return the expected 9-output P3 ABI"
        )
    return outputs


def run_advisory_mirror_isolation() -> dict[str, Any]:
    """Return the exact baseline and exact concurrent C8 output for comparison."""
    reference_opapi, reference_vendor = _resolve_single_opp(
        os.getenv(REFERENCE_OPP_ENV), "reference"
    )
    mirror_opapi, mirror_vendor = _resolve_single_opp(
        os.getenv(TARGET_OPP_ENV), "target"
    )
    _prepend_unique_path(
        "ASCEND_CUSTOM_OPP_PATH", (reference_vendor, mirror_vendor)
    )
    official, metadata, _ = _require_source_matched_official_c8_ops(
        reference_opapi, reference_vendor
    )
    device = _require_a5_npu()
    inputs = _build_official_inputs(device)
    inputs["metadata"] = _official_metadata(metadata, inputs)

    # This output is generated before the mirror bridge is loaded.  It is the
    # mathematical reference, not an old editable vLLM installation.
    baseline_indices = _call_official(official, inputs)
    torch.npu.synchronize()

    _bind_isolated_target_opp(mirror_opapi)
    mirror = _require_fresh_target_bridge()
    transaction = _make_no_post_transaction(
        device, row_count=2, control_mode=HOOK_CONTROL_MODE_REJECT_BUSY
    )
    snapshot = _snapshot_no_post_transaction(transaction)

    # Input construction occurred on the current stream.  The two task
    # streams only wait for this producer event; neither waits for the other.
    # This is the topology that the future graph path needs: the exact C8
    # branch remains independent of an advisory prefetch branch.
    producer_stream = torch.npu.current_stream()
    inputs_ready = torch.npu.Event()
    inputs_ready.record(producer_stream)
    mirror_stream = torch.npu.Stream()
    exact_stream = torch.npu.Stream()
    mirror_start = torch.npu.Event(enable_timing=True)
    mirror_end = torch.npu.Event(enable_timing=True)
    exact_start = torch.npu.Event(enable_timing=True)
    exact_end = torch.npu.Event(enable_timing=True)

    print(
        "P4_ADVISORY_MIRROR_CONFIG "
        f"device={device} physical_npu={device.index} "
        f"reference_opapi={reference_opapi} mirror_opapi={mirror_opapi} "
        "mirror_role=raw_qk_score_scan_non_authoritative "
        "exact_role=official_c8_topk_authoritative "
        "control=reject_all_busy h2d_result=not_exercised",
        flush=True,
    )
    launched = monotonic()
    with torch.npu.stream(mirror_stream):
        mirror_stream.wait_event(inputs_ready)
        mirror_start.record(mirror_stream)
        mirror_outputs = _call_advisory_mirror(mirror, inputs, transaction)
        mirror_end.record(mirror_stream)
    with torch.npu.stream(exact_stream):
        exact_stream.wait_event(inputs_ready)
        exact_start.record(exact_stream)
        concurrent_exact_indices = _call_official(official, inputs)
        exact_end.record(exact_stream)

    mirror_end.synchronize()
    exact_end.synchronize()
    _prove_safe_hook_control(
        mirror_outputs,
        transaction,
        snapshot,
        HOOK_CONTROL_MODE_REJECT_BUSY,
    )
    print(
        "P4_ADVISORY_MIRROR_TIMING "
        f"mirror_us={mirror_start.elapsed_time(mirror_end) * 1000.0:.3f} "
        f"exact_us={exact_start.elapsed_time(exact_end) * 1000.0:.3f} "
        f"host_elapsed_ms={(monotonic() - launched) * 1000.0:.3f}",
        flush=True,
    )
    return {
        "baseline_indices": baseline_indices,
        "concurrent_exact_indices": concurrent_exact_indices,
        "input_aliases": {
            name: (inputs[name], inputs[name])
            for name in (
                "query",
                "key",
                "weights",
                "query_dequant_scale",
                "key_dequant_scale",
                "actual_seq_lengths_query",
                "actual_seq_lengths_key",
                "block_table",
                "metadata",
            )
        },
        "transaction_aliases": {
            "private_source_to_slot": (
                transaction["private_source_to_slot"], mirror_outputs[2]
            ),
            "private_ready_table": (
                transaction["private_ready_table"], mirror_outputs[3]
            ),
            "private_arena_head": (
                transaction["private_arena_head"], mirror_outputs[4]
            ),
            "private_packed_output": (
                transaction["private_packed_output"], mirror_outputs[5]
            ),
            "queue_leases": (transaction["queue_leases"], mirror_outputs[6]),
        },
    }


__all__ = ["run_advisory_mirror_isolation"]
