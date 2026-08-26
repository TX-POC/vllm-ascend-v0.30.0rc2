#!/usr/bin/env python3
"""Measure P1 task-lane overlap without treating a bad payload as success.

This benchmark runs the same 64-AIV P1 custom op in two input placements:

* ``early`` publishes each producer/lane's 32-miss ring chunk near the start
  of its ExactScan work, so the two Get task lanes can issue Direct-HCCP while
  the remaining 62 ExactScan task lanes are still scanning.
* ``late`` moves exactly the same hits, misses, descriptor triples, and output
  rows to the end of each producer scan.  Get task lanes still exist from the
  beginning, but the required chunks become available only near scan finish.

It is deliberately a scheduling/CQ-path benchmark.  The current dense Get
prototype has a known payload-integrity failure (mostly zero returned rows),
so this script verifies only the control plane and never reports payload
bandwidth or a real H2D end-to-end claim.
"""

from __future__ import annotations

import argparse
import statistics
from typing import Any

import torch
import torch_npu  # noqa: F401

from test_exact_scan_speculative_get import (
    CANDIDATE_HBM_HIT,
    ENTRY_BYTES,
    FUSED_CONTROL_MAGIC,
    FUSED_CONTROL_VERSION,
    GET_LANE_COUNT,
    PACKED_ENTRY_BYTES,
    PIPELINE_RING_COUNT,
    PIPELINE_WORDS_PER_RING,
    PRODUCER_COUNT,
    build_ddr_payload,
    build_exact_scan_inputs,
    check_args,
    create_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--transport", choices=("local", "remote"), default="local")
    parser.add_argument("--cpu-service-addr", default="")
    parser.add_argument("--batch-get-kernel", required=True)
    parser.add_argument("--batch-put-kernel", required=True)
    parser.add_argument("--table-capacity", type=int, default=32)
    # 62 * 67,650 is within 4 tokens of B=32 x 128K.  This is still the P1
    # scalar threshold-scan prototype, not the official C8 Lightning Indexer.
    parser.add_argument("--scan-per-producer", type=int, default=67_650)
    parser.add_argument("--misses-per-lane", type=int, default=32)
    parser.add_argument("--hits-per-lane", type=int, default=2)
    parser.add_argument("--frame-epoch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args()


def percentile(samples_us: list[float], fraction: float) -> float:
    ordered = sorted(samples_us)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def report_perf(operation: str, samples_us: list[float], warmup: int) -> None:
    print(
        "PERF_RESULT "
        f"operation={operation} samples={len(samples_us)} "
        f"warmup_excluded={warmup} "
        f"mean_us={statistics.mean(samples_us):.3f} "
        f"p50_us={percentile(samples_us, 0.50):.3f} "
        f"p90_us={percentile(samples_us, 0.90):.3f} "
        f"p99_us={percentile(samples_us, 0.99):.3f}",
        flush=True,
    )


def validate_control(
    fused_get_control: torch.Tensor,
    packed_output: torch.Tensor,
    expected_output_rows: int,
) -> None:
    words = [int(value) for value in fused_get_control.cpu().tolist()]
    mismatches: list[str] = []
    expected = (
        (0, FUSED_CONTROL_MAGIC),
        (1, FUSED_CONTROL_VERSION),
        (5, packed_output.data_ptr()),
        (6, expected_output_rows),
        (7, GET_LANE_COUNT),
        (8, 1),
        (9, PACKED_ENTRY_BYTES),
    )
    for index, value in expected:
        if words[index] != value:
            mismatches.append(f"word{index}")
    for index, name in ((2, "channel_entity"), (3, "queue_state"), (4, "core_status")):
        if words[index] == 0:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(
            "P1 fused control ABI is invalid: " + ",".join(mismatches)
        )


def validate_control_plane(
    label: str,
    outputs: tuple[torch.Tensor, ...],
    packed_output: torch.Tensor,
    pipeline_state: torch.Tensor,
    expected: list[list[Any]],
    expected_misses: int,
    lane_output_rows: int,
) -> dict[str, Any]:
    (
        packed_output_out,
        candidate_indices,
        candidate_kind,
        producer_stats,
        get_stats,
        role_trace,
        _get_indices,
        _get_offsets,
        _get_output_indices,
        _get_mask,
        pipeline_state_out,
    ) = outputs
    if packed_output_out.data_ptr() != packed_output.data_ptr():
        raise AssertionError(f"{label}: packed output lost its bound address")
    if pipeline_state_out.data_ptr() != pipeline_state.data_ptr():
        raise AssertionError(f"{label}: pipeline state lost its stable address")

    producer = producer_stats.cpu()
    get = get_stats.cpu()
    trace = role_trace.cpu()
    candidate_indices_cpu = candidate_indices.cpu()
    candidate_kind_cpu = candidate_kind.cpu()
    expected_selected = len(expected[0])
    expected_hits = sum(item.kind == CANDIDATE_HBM_HIT for item in expected[0])
    expected_miss = expected_selected - expected_hits
    expected_trace = torch.cat(
        (
            torch.arange(2000, 2000 + GET_LANE_COUNT, dtype=torch.int32),
            torch.arange(1000, 1000 + PRODUCER_COUNT, dtype=torch.int32),
        )
    )

    checks = (
        bool(torch.all(producer[:, 0] == expected_selected)),
        bool(torch.all(producer[:, 1] == expected_hits)),
        bool(torch.all(producer[:, 2] == expected_miss)),
        bool(torch.all(producer[:, 3] == expected_miss)),
        bool(torch.all(producer[:, 4] == 0)),
        bool(torch.all(producer[:, 5] > 0)),
        bool(torch.all(producer[:, 6] > 0)),
        bool(torch.all(get[:, 0] == lane_output_rows)),
        int(get[:, 0].sum()) == expected_misses,
        bool(torch.all(get[:, 1] > 0)),
        bool(torch.all(get[:, 2] == 0)),
        bool(torch.all(get[:, 3] > 0)),
        bool(torch.all(get[:, 4] > 0)),
        bool(torch.all(get[:, 6] == PRODUCER_COUNT)),
        bool(torch.equal(trace, expected_trace)),
    )
    if not all(checks):
        raise AssertionError(
            f"{label}: control-plane validation failed "
            f"producer={producer.tolist()} get={get.tolist()} trace={trace.tolist()}"
        )

    # Payload bytes are intentionally not copied: their known failure is the
    # reason this is only a scheduling/CQ-path experiment.  The compact
    # candidate/control plane is still checked exactly, so a broken scan or
    # a malformed ring cannot silently look like a useful overlap result.
    for producer_id, entries in enumerate(expected):
        for rank, item in enumerate(entries):
            slot = rank * PRODUCER_COUNT + producer_id
            if int(candidate_indices_cpu[slot]) != item.source:
                raise AssertionError(
                    f"{label}: candidate source mismatch "
                    f"producer={producer_id} rank={rank}"
                )
            if int(candidate_kind_cpu[slot]) != item.kind:
                raise AssertionError(
                    f"{label}: candidate kind mismatch "
                    f"producer={producer_id} rank={rank}"
                )

    first_doorbell = int(get[:, 3].min())
    completion_done = int(get[:, 4].max())
    scan_done = int(producer[:, 6].max())
    return {
        "first_doorbell_cycle": first_doorbell,
        "completion_done_cycle": completion_done,
        "scan_done_cycle": scan_done,
        "tail_after_doorbell_cycles": max(0, scan_done - first_doorbell),
        "doorbell_before_scan_done": int(first_doorbell < scan_done),
        "requested": int(get[:, 0].sum()),
        "batches": int(get[:, 1].sum()),
        "per_lane_requested": [int(value) for value in get[:, 0].tolist()],
        "per_lane_batches": [int(value) for value in get[:, 1].tolist()],
    }


def validate_all_sample_telemetry(
    label: str,
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    expected: list[list[Any]],
    lane_output_rows: int,
) -> None:
    """Reject an intermittent CQ/control failure without timing host copies."""
    expected_selected = len(expected[0])
    expected_hits = sum(item.kind == CANDIDATE_HBM_HIT for item in expected[0])
    expected_miss = expected_selected - expected_hits
    expected_trace = torch.cat(
        (
            torch.arange(2000, 2000 + GET_LANE_COUNT, dtype=torch.int32),
            torch.arange(1000, 1000 + PRODUCER_COUNT, dtype=torch.int32),
        )
    )
    for sample_index, (producer_stats, get_stats, role_trace) in enumerate(samples):
        producer = producer_stats.cpu()
        get = get_stats.cpu()
        trace = role_trace.cpu()
        valid = (
            bool(torch.all(producer[:, 0] == expected_selected))
            and bool(torch.all(producer[:, 1] == expected_hits))
            and bool(torch.all(producer[:, 2] == expected_miss))
            and bool(torch.all(producer[:, 3] == expected_miss))
            and bool(torch.all(producer[:, 4] == 0))
            and bool(torch.all(get[:, 0] == lane_output_rows))
            and bool(torch.all(get[:, 1] > 0))
            and bool(torch.all(get[:, 2] == 0))
            and bool(torch.all(get[:, 6] == PRODUCER_COUNT))
            and bool(torch.equal(trace, expected_trace))
        )
        if not valid:
            raise AssertionError(
                f"{label}: telemetry failed at replay={sample_index} "
                f"producer={producer.tolist()} get={get.tolist()} "
                f"trace={trace.tolist()}"
            )


def main() -> None:
    args = parse_args()
    check_args(args)
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")

    device = torch.device(args.device)
    torch.npu.set_device(device)
    device_id = device.index if device.index is not None else torch.npu.current_device()
    get_device_name = getattr(
        torch.npu, "get_device_name", torch_npu.npu.get_device_name
    )
    device_name = get_device_name(device_id)
    if "950" not in device_name.lower():
        raise RuntimeError(f"expected Ascend 950, got {device_name!r}")

    cases: dict[str, dict[str, Any]] = {}
    for label in ("early", "late"):
        (
            scores_cpu,
            resident_cpu,
            table_indices_cpu,
            offsets_cpu,
            output_indices_cpu,
            expected,
            expected_misses,
            lane_output_rows,
        ) = build_exact_scan_inputs(args, candidate_placement=label)
        cases[label] = {
            "scores": scores_cpu.to(device),
            "resident": resident_cpu.to(device),
            "table_indices": table_indices_cpu.to(device),
            "offsets": offsets_cpu.to(device),
            "output_indices": output_indices_cpu.to(device),
            "expected": expected,
            "expected_misses": expected_misses,
            "lane_output_rows": lane_output_rows,
        }
    if (
        cases["early"]["expected_misses"] != cases["late"]["expected_misses"]
        or cases["early"]["lane_output_rows"] != cases["late"]["lane_output_rows"]
    ):
        raise AssertionError("early/late fixtures do not have identical Get work")
    early_kind_layout = [
        [item.kind for item in entries] for entries in cases["early"]["expected"]
    ]
    late_kind_layout = [
        [item.kind for item in entries] for entries in cases["late"]["expected"]
    ]
    if early_kind_layout != late_kind_layout:
        raise AssertionError("early/late fixtures changed candidate-kind order")
    torch.npu.synchronize()

    expected_misses = int(cases["early"]["expected_misses"])
    lane_output_rows = int(cases["early"]["lane_output_rows"])
    entry_payload_cpu, _unused_expected_payload = build_ddr_payload(
        args.table_capacity, expected_misses
    )
    keys = list(range(1, args.table_capacity + 1))
    runtime = create_runtime(args, device_id)
    table = None
    control_bound = False
    try:
        print(
            "P1_TASK_LANE_OVERLAP_CONFIG "
            f"device={args.device} device_name={device_name!r} "
            f"transport={args.transport} score_count={PRODUCER_COUNT * args.scan_per_producer} "
            f"scan_per_producer={args.scan_per_producer} "
            f"producer_task_lanes=2..63 get_task_lanes=0,1 "
            "physical_core_affinity_proof=0 "
            f"misses_per_producer_lane={args.misses_per_lane} "
            f"hits_per_producer_lane={args.hits_per_lane} "
            f"total_misses={expected_misses} warmup={args.warmup} iters={args.iters} "
            "placement_pair=early_vs_late"
        )
        print(
            "P1_TASK_LANE_OVERLAP_INTEGRITY "
            "payload_validation=skipped payload_integrity=known_failed "
            "known_dense_get_payload_failure=1 "
            "data_plane_unverified=1 h2d_bandwidth_claim=0 "
            "result_scope=task_lane_scheduling_and_cq_path_only "
            "descriptor_history=not_exported_final_wave_only "
            "exactscan_kind=scalar_threshold_prototype_not_official_c8_li"
        )

        runtime.initialize(args.table_capacity * ENTRY_BYTES, ENTRY_BYTES)
        source = entry_payload_cpu.to(device)
        torch.npu.synchronize()
        runtime.batch_put(
            keys,
            source.reshape(-1),
            [entry * ENTRY_BYTES for entry in range(args.table_capacity)],
            [ENTRY_BYTES] * args.table_capacity,
        )
        torch.npu.synchronize()
        table = runtime.prepare_access_table(keys)
        device_table = runtime.access_table_view(
            table, device="npu", capacity=args.table_capacity
        )
        address_table = device_table.addresses
        valid_table = device_table.valid
        packed_output = torch.empty(
            (expected_misses, PACKED_ENTRY_BYTES), dtype=torch.uint8, device=device
        )
        fused_get_control = torch.zeros(16, dtype=torch.int64, device=device)
        pipeline_state = torch.zeros(
            (PIPELINE_RING_COUNT, PIPELINE_WORDS_PER_RING),
            dtype=torch.int64,
            device=device,
        )
        runtime.bind_fused_batch_get_control(
            packed_output, fused_get_control, torch.npu.current_stream()
        )
        control_bound = True
        torch.npu.synchronize()
        validate_control(fused_get_control, packed_output, expected_misses)

        frame_epoch = torch.empty(1, dtype=torch.int64, device=device)
        threshold = torch.tensor([0.5], dtype=torch.float32, device=device)
        next_epoch = args.frame_epoch - 1
        last_outputs: dict[str, tuple[torch.Tensor, ...]] = {}
        sample_telemetry: dict[
            str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = {"early": [], "late": []}

        def run_one(label: str) -> float:
            nonlocal next_epoch
            next_epoch += 1
            frame_epoch.fill_(next_epoch)
            case = cases[label]
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            outputs = torch.ops.a5_exact_scan_get.exact_scan_speculative_get(
                case["scores"],
                threshold,
                case["resident"],
                case["table_indices"],
                case["offsets"],
                case["output_indices"],
                address_table,
                valid_table,
                packed_output,
                fused_get_control,
                frame_epoch,
                pipeline_state,
            )
            end.record()
            end.synchronize()
            last_outputs[label] = outputs
            # Keep only tiny telemetry from every replay.  It is copied after
            # timing, so an intermittent failed CQ cannot be hidden by a
            # healthy final replay and no host copy enters the NPU event.
            sample_telemetry[label].append((outputs[3], outputs[4], outputs[5]))
            return start.elapsed_time(end) * 1000.0

        # Alternate the warm-up order as well: the Direct-HCCP queue state is
        # intentionally persistent across calls and one placement must not
        # always inherit the warm queue.
        for iteration in range(args.warmup):
            order = ("early", "late") if iteration % 2 == 0 else ("late", "early")
            for label in order:
                run_one(label)

        samples_us: dict[str, list[float]] = {"early": [], "late": []}
        paired_saved_us: list[float] = []
        for iteration in range(args.iters):
            order = ("early", "late") if iteration % 2 == 0 else ("late", "early")
            pair: dict[str, float] = {}
            for label in order:
                elapsed_us = run_one(label)
                samples_us[label].append(elapsed_us)
                pair[label] = elapsed_us
            paired_saved_us.append(pair["late"] - pair["early"])

        telemetry = {
            label: validate_control_plane(
                label,
                last_outputs[label],
                packed_output,
                pipeline_state,
                cases[label]["expected"],
                int(cases[label]["expected_misses"]),
                int(cases[label]["lane_output_rows"]),
            )
            for label in ("early", "late")
        }
        for label in ("early", "late"):
            validate_all_sample_telemetry(
                label,
                sample_telemetry[label],
                cases[label]["expected"],
                int(cases[label]["lane_output_rows"]),
            )
        if telemetry["early"]["doorbell_before_scan_done"] != 1:
            raise AssertionError(
                "early fixture did not create a pre-scan-completion Get window"
            )

        report_perf("p1_exact_scan_speculative_get_early", samples_us["early"], args.warmup)
        report_perf("p1_exact_scan_speculative_get_late", samples_us["late"], args.warmup)
        for label in ("early", "late"):
            info = telemetry[label]
            print(
                "P1_TASK_LANE_OVERLAP_TELEMETRY "
                f"placement={label} requested={info['requested']} "
                f"batches={info['batches']} "
                f"per_lane_requested={info['per_lane_requested']} "
                f"per_lane_batches={info['per_lane_batches']} "
                f"first_doorbell_cycle={info['first_doorbell_cycle']} "
                f"scan_done_cycle={info['scan_done_cycle']} "
                f"completion_done_cycle={info['completion_done_cycle']} "
                f"tail_after_doorbell_cycles={info['tail_after_doorbell_cycles']} "
                f"doorbell_before_scan_done={info['doorbell_before_scan_done']}"
            )

        early_mean = statistics.mean(samples_us["early"])
        late_mean = statistics.mean(samples_us["late"])
        paired_mean = statistics.mean(paired_saved_us)
        paired_reduction_pct = paired_mean * 100.0 / late_mean if late_mean else 0.0
        direction = "positive" if paired_mean > 0.0 else "non_positive"
        baseline_qualified = int(
            telemetry["late"]["doorbell_before_scan_done"] == 0
        )
        batch_shape_equal = int(
            telemetry["early"]["per_lane_requested"]
            == telemetry["late"]["per_lane_requested"]
            and telemetry["early"]["per_lane_batches"]
            == telemetry["late"]["per_lane_batches"]
        )
        pure_overlap_qualified = int(baseline_qualified and batch_shape_equal)
        print(
            "P1_TASK_LANE_OVERLAP_RESULT "
            f"pairs={len(paired_saved_us)} "
            f"early_mean_us={early_mean:.3f} "
            f"late_mean_us={late_mean:.3f} "
            f"observed_early_vs_delayed_mean_us={paired_mean:.3f} "
            f"observed_early_vs_delayed_p50_us={percentile(paired_saved_us, 0.50):.3f} "
            f"observed_early_vs_delayed_p90_us={percentile(paired_saved_us, 0.90):.3f} "
            f"observed_early_vs_delayed_mean_pct={paired_reduction_pct:.2f} "
            f"benefit_direction={direction} "
            f"late_baseline_qualified={baseline_qualified} "
            f"batch_shape_equal={batch_shape_equal} "
            f"pure_overlap_qualified={pure_overlap_qualified} "
            "reference=late_delayed_not_strict_serial"
        )
        print("P1_TASK_LANE_OVERLAP_PASS")
    finally:
        # The event for every replay has already synchronized.  Keep the
        # explicit terminal boundary before invalidating the cold control ABI.
        if control_bound:
            torch.npu.synchronize()
            runtime.release_fused_batch_get_control()
        if table is not None:
            runtime.release_access_table(table)
        runtime.shutdown()


if __name__ == "__main__":
    main()
