#!/usr/bin/env python3
"""Strict same-input P1 comparison: serial Get versus chunk-pipelined Get.

The P1 operator is one AIV-only, 64-block batch-mode launch.  Logical task
lanes 0/1 run the two blocking Direct-HCCP full-Get workers, while task lanes
2..63 run ExactScan.  This script uses one immutable score/resident/descriptor
fixture for both modes and changes only two graph-stable control scalars:

* ``frame_epoch``: monotonically advances the reusable SPSC-ring epoch;
* ``execution_mode``: ``0`` is streaming, ``1`` delays every Get consumer
  until all 124 producer rings have terminally closed.

It therefore measures the causal effect of allowing the existing blocking Get
to consume ready chunks while ExactScan is still running.  The prototype's
payload is a known all-zero failure, so the script validates control/CQ work
only and intentionally makes no H2D-bandwidth or end-to-end claim.

The primary latency is deliberately measured from host submission through a
device-wide terminal fence.  A raw ACLNN custom-op launch can fall outside the
coverage of a ``torch.npu.Event`` even when the event itself synchronizes, so
event time is printed only as a coverage diagnostic.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
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
    PIPELINE_MODE_PARALLEL,
    PIPELINE_MODE_SERIAL_WAIT_ALL_CLOSED,
    PRODUCER_COUNT,
    RECORDS_PER_RING_SLOT,
    RING_SLOTS,
    build_ddr_payload,
    build_exact_scan_inputs,
    check_args,
    create_runtime,
)


GET_STAT_REQUESTED = 0
GET_STAT_BATCHES = 1
GET_STAT_ERROR = 2
GET_STAT_FIRST_DOORBELL = 3
GET_STAT_COMPLETION = 4
GET_STAT_IDLE_POLLS = 5
GET_STAT_CONSUMED_CHUNKS = 6
GET_STAT_FINISH = 7
GET_STAT_MODE = 8
GET_STAT_SERIAL_GATE_OPEN = 9
GET_STAT_FIRST_CONSUMED_CHUNK = 10
GET_STAT_SERIAL_GATE_POLLS = 11
GET_STAT_DESCRIPTOR_SUM = 12
GET_STAT_DESCRIPTOR_XOR = 13

PRODUCER_STAT_SELECTED = 0
PRODUCER_STAT_HIT = 1
PRODUCER_STAT_MISS = 2
PRODUCER_STAT_ENQUEUED = 3
PRODUCER_STAT_DEFERRED = 4
PRODUCER_STAT_FIRST_CHUNK = 5
PRODUCER_STAT_SCAN_DONE = 6
PRODUCER_STAT_CLOSED = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--transport", choices=("local", "remote"), default="local")
    parser.add_argument("--cpu-service-addr", default="")
    parser.add_argument("--batch-get-kernel", required=True)
    parser.add_argument("--batch-put-kernel", required=True)
    parser.add_argument("--table-capacity", type=int, default=32)
    # 62 * 67,650 is within four tokens of B=32 x 128K.  This remains the
    # scalar threshold/resident P1 seam, not an implementation of C8 LI.
    parser.add_argument("--scan-per-producer", type=int, default=67_650)
    parser.add_argument("--misses-per-lane", type=int, default=32)
    parser.add_argument("--hits-per-lane", type=int, default=2)
    parser.add_argument("--frame-epoch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--event-diagnostic",
        action="store_true",
        help=(
            "also emit a fenced torch.npu.Event value for diagnosis; it is "
            "never used as a P1 performance result"
        ),
    )
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
    expected = (
        (0, FUSED_CONTROL_MAGIC),
        (1, FUSED_CONTROL_VERSION),
        (5, packed_output.data_ptr()),
        (6, expected_output_rows),
        (7, GET_LANE_COUNT),
        (8, 1),
        (9, PACKED_ENTRY_BYTES),
    )
    mismatches = [f"word{index}" for index, value in expected if words[index] != value]
    for index, name in ((2, "channel_entity"), (3, "queue_state"), (4, "core_status")):
        if words[index] == 0:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError("P1 fused control ABI is invalid: " + ",".join(mismatches))


def expected_batch_count(per_lane_requested: int) -> int:
    if per_lane_requested <= 0:
        return 0
    if per_lane_requested <= 64:
        return 1
    remaining = per_lane_requested - 64
    return 1 + (remaining + 255) // 256


def candidate_signature(
    label: str,
    candidate_indices: torch.Tensor,
    candidate_kind: torch.Tensor,
    expected: list[list[Any]],
) -> tuple[int, int]:
    """Verify exactly the selected candidate entries and return a signature."""

    indices = candidate_indices.cpu()
    kinds = candidate_kind.cpu()
    signature_sum = 0
    signature_xor = 0
    for producer, entries in enumerate(expected):
        for rank, item in enumerate(entries):
            slot = rank * PRODUCER_COUNT + producer
            actual_source = int(indices[slot])
            actual_kind = int(kinds[slot])
            if actual_source != item.source or actual_kind != item.kind:
                raise AssertionError(
                    f"{label}: candidate mismatch producer={producer} rank={rank} "
                    f"actual=({actual_source},{actual_kind}) "
                    f"expected=({item.source},{item.kind})"
                )
            fingerprint = ((actual_source & 0xFFFFFFFF) << 8) ^ actual_kind
            signature_sum += fingerprint
            signature_xor ^= fingerprint
    return signature_sum, signature_xor


@dataclass(frozen=True)
class Telemetry:
    mode: int
    requested: int
    batches: int
    consumed_chunks: int
    errors: tuple[int, int]
    first_doorbell: int
    completion: int
    first_consumed_chunk: int
    serial_gate_open: int
    serial_gate_polls: int
    producer_scan_done: int
    producer_closed: int
    descriptor_sum: int
    descriptor_xor: int


@dataclass(frozen=True)
class Timing:
    """One replay measured at the host/terminal and stream-event boundaries.

    ``terminal_us`` is the primary P1 comparison metric.  It starts immediately
    before P1 is submitted and ends only after a device-wide NPU fence.  That
    deliberately includes host submission, ACLNN launch, and all device work
    caused by this P1 invocation.  It is the only safe metric while the bridge
    may submit work outside the timing event's visible stream.

    The remaining fields are diagnostics.  In particular, ``event_us`` must
    never be used to claim overlap until it agrees with the terminal fence.
    """

    submit_us: float
    terminal_us: float
    event_us: float | None


def validate_sample(
    label: str,
    outputs: tuple[torch.Tensor, ...],
    packed_output: torch.Tensor,
    pipeline_state: torch.Tensor,
    expected: list[list[Any]],
    expected_misses: int,
    lane_output_rows: int,
    mode: int,
) -> Telemetry:
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
        raise AssertionError(f"{label}: packed output lost its cold-bound address")
    if pipeline_state_out.data_ptr() != pipeline_state.data_ptr():
        raise AssertionError(f"{label}: pipeline state lost its stable address")

    producer = producer_stats.cpu()
    get = get_stats.cpu()
    trace = role_trace.cpu()
    expected_selected = len(expected[0])
    expected_hits = sum(item.kind == CANDIDATE_HBM_HIT for item in expected[0])
    expected_miss_per_producer = expected_selected - expected_hits
    expected_trace = torch.cat(
        (
            torch.arange(2000, 2000 + GET_LANE_COUNT, dtype=torch.int32),
            torch.arange(1000, 1000 + PRODUCER_COUNT, dtype=torch.int32),
        )
    )
    expected_batches = expected_batch_count(lane_output_rows)
    checks = (
        bool(torch.all(producer[:, PRODUCER_STAT_SELECTED] == expected_selected)),
        bool(torch.all(producer[:, PRODUCER_STAT_HIT] == expected_hits)),
        bool(torch.all(producer[:, PRODUCER_STAT_MISS] == expected_miss_per_producer)),
        bool(torch.all(producer[:, PRODUCER_STAT_ENQUEUED] == expected_miss_per_producer)),
        bool(torch.all(producer[:, PRODUCER_STAT_DEFERRED] == 0)),
        bool(torch.all(producer[:, PRODUCER_STAT_FIRST_CHUNK] > 0)),
        bool(torch.all(producer[:, PRODUCER_STAT_SCAN_DONE] > 0)),
        bool(torch.all(producer[:, PRODUCER_STAT_CLOSED] > 0)),
        bool(torch.all(get[:, GET_STAT_REQUESTED] == lane_output_rows)),
        int(get[:, GET_STAT_REQUESTED].sum()) == expected_misses,
        bool(torch.all(get[:, GET_STAT_BATCHES] == expected_batches)),
        bool(torch.all(get[:, GET_STAT_ERROR] == 0)),
        bool(torch.all(get[:, GET_STAT_FIRST_DOORBELL] > 0)),
        bool(torch.all(get[:, GET_STAT_COMPLETION] > 0)),
        bool(torch.all(get[:, GET_STAT_CONSUMED_CHUNKS] == PRODUCER_COUNT)),
        bool(torch.all(get[:, GET_STAT_MODE] == mode)),
        bool(torch.equal(trace, expected_trace)),
    )
    if not all(checks):
        raise AssertionError(
            f"{label}: control/CQ validation failed producer={producer.tolist()} "
            f"get={get.tolist()} trace={trace.tolist()}"
        )

    if mode == PIPELINE_MODE_PARALLEL:
        if not bool(torch.all(get[:, GET_STAT_SERIAL_GATE_OPEN] == 0)):
            raise AssertionError(f"{label}: parallel mode unexpectedly opened serial gate")
    else:
        if not bool(torch.all(get[:, GET_STAT_SERIAL_GATE_OPEN] > 0)):
            raise AssertionError(f"{label}: serial mode did not open both gates")
        if not bool(torch.all(get[:, GET_STAT_FIRST_DOORBELL] >= get[:, GET_STAT_SERIAL_GATE_OPEN])):
            raise AssertionError(f"{label}: serial doorbell preceded all-producer gate")
        if not bool(torch.all(get[:, GET_STAT_FIRST_CONSUMED_CHUNK] >= get[:, GET_STAT_SERIAL_GATE_OPEN])):
            raise AssertionError(f"{label}: serial consumption preceded all-producer gate")

    return Telemetry(
        mode=mode,
        requested=int(get[:, GET_STAT_REQUESTED].sum()),
        batches=int(get[:, GET_STAT_BATCHES].sum()),
        consumed_chunks=int(get[:, GET_STAT_CONSUMED_CHUNKS].sum()),
        errors=tuple(int(value) for value in get[:, GET_STAT_ERROR].tolist()),
        first_doorbell=int(get[:, GET_STAT_FIRST_DOORBELL].min()),
        completion=int(get[:, GET_STAT_COMPLETION].max()),
        first_consumed_chunk=int(get[:, GET_STAT_FIRST_CONSUMED_CHUNK].min()),
        serial_gate_open=int(get[:, GET_STAT_SERIAL_GATE_OPEN].min()),
        serial_gate_polls=int(get[:, GET_STAT_SERIAL_GATE_POLLS].sum()),
        producer_scan_done=int(producer[:, PRODUCER_STAT_SCAN_DONE].max()),
        producer_closed=int(producer[:, PRODUCER_STAT_CLOSED].max()),
        descriptor_sum=int(get[:, GET_STAT_DESCRIPTOR_SUM].sum()),
        descriptor_xor=int(get[0, GET_STAT_DESCRIPTOR_XOR]) ^ int(get[1, GET_STAT_DESCRIPTOR_XOR]),
    )


def validate_cross_mode(
    parallel: list[Telemetry], serial: list[Telemetry]
) -> tuple[Telemetry, Telemetry]:
    if not parallel or not serial:
        raise AssertionError("no timed P1 samples were collected")
    parallel_reference = parallel[-1]
    serial_reference = serial[-1]
    for label, values, expected_mode in (
        ("parallel", parallel, PIPELINE_MODE_PARALLEL),
        ("serial", serial, PIPELINE_MODE_SERIAL_WAIT_ALL_CLOSED),
    ):
        for sample_index, value in enumerate(values):
            if value.mode != expected_mode or any(value.errors):
                raise AssertionError(f"{label}: invalid mode/error at sample={sample_index}")
            if value.requested != parallel_reference.requested:
                raise AssertionError(f"{label}: requested work drifted at sample={sample_index}")
            if value.batches != parallel_reference.batches:
                raise AssertionError(f"{label}: WQE segmentation drifted at sample={sample_index}")
            if value.consumed_chunks != parallel_reference.consumed_chunks:
                raise AssertionError(f"{label}: ring-consumption work drifted at sample={sample_index}")
            if (value.descriptor_sum, value.descriptor_xor) != (
                parallel_reference.descriptor_sum,
                parallel_reference.descriptor_xor,
            ):
                raise AssertionError(f"{label}: descriptor multiset drifted at sample={sample_index}")
    return parallel_reference, serial_reference


def main() -> None:
    args = parse_args()
    check_args(args)
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    # The serial comparator intentionally leaves producer rings untouched
    # until every scan has closed.  It is fair only if one producer/lane can
    # retain all of its selected misses in the two-slot SPSC ring; otherwise
    # serial would defer a descriptor that streaming could have consumed.
    per_producer_lane_ring_capacity = RING_SLOTS * RECORDS_PER_RING_SLOT
    if args.misses_per_lane > per_producer_lane_ring_capacity:
        raise ValueError(
            "strict serial requires --misses-per-lane <= "
            f"{per_producer_lane_ring_capacity}; got {args.misses_per_lane}"
        )

    device = torch.device(args.device)
    torch.npu.set_device(device)
    device_id = device.index if device.index is not None else torch.npu.current_device()
    get_device_name = getattr(torch.npu, "get_device_name", torch_npu.npu.get_device_name)
    device_name = get_device_name(device_id)
    if "950" not in device_name.lower():
        raise RuntimeError(f"expected Ascend 950, got {device_name!r}")

    (
        scores_cpu,
        resident_cpu,
        table_indices_cpu,
        offsets_cpu,
        output_indices_cpu,
        expected,
        expected_misses,
        lane_output_rows,
    ) = build_exact_scan_inputs(args)
    entry_payload_cpu, _ = build_ddr_payload(args.table_capacity, expected_misses)
    runtime = create_runtime(args, device_id)
    table: Any = None
    control_bound = False
    try:
        print(
            "P1_STRICT_SAME_INPUT_CONFIG "
            f"device={args.device} device_name={device_name!r} transport={args.transport} "
            f"score_count={scores_cpu.numel()} scan_per_producer={args.scan_per_producer} "
            f"misses_per_producer_lane={args.misses_per_lane} "
            f"hits_per_producer_lane={args.hits_per_lane} total_misses={expected_misses} "
            f"per_producer_lane_ring_capacity={per_producer_lane_ring_capacity} "
            "serial_no_deferred_required=1 "
            f"warmup={args.warmup} iters={args.iters} semantic_inputs=identical "
            "mutable_control=frame_epoch,execution_mode "
            "frame_epoch_scope=ring_reuse_only comparison_variable=execution_mode"
        )
        print(
            "P1_EXECUTION_SCOPED_AIV_PARTITION "
            "aiv_instances=64 block_dim=64 schedule_mode=batch "
            "get_task_lanes=0,1 exactscan_task_lanes=2..63 "
            "same_invocation_role_exclusive=1 "
            "fixed_physical_aiv_id_mask=unavailable_public_cann_9_1 "
            "fixed_physical_aiv_id_affinity_claim=0"
        )
        print(
            "P1_STRICT_SAME_INPUT_INTEGRITY "
            "payload_validation=skipped payload_integrity=known_failed "
            "h2d_bandwidth_claim=0 end_to_end_claim=0 "
            "result_scope=control_cq_path_and_execution_scoped_overlap_only "
            "exactscan_kind=scalar_threshold_resident_seam_not_official_c8_li"
        )

        runtime.initialize(args.table_capacity * ENTRY_BYTES, ENTRY_BYTES)
        source = entry_payload_cpu.to(device)
        torch.npu.synchronize()
        runtime.batch_put(
            list(range(1, args.table_capacity + 1)),
            source.reshape(-1),
            [entry * ENTRY_BYTES for entry in range(args.table_capacity)],
            [ENTRY_BYTES] * args.table_capacity,
        )
        torch.npu.synchronize()
        table = runtime.prepare_access_table(list(range(1, args.table_capacity + 1)))
        device_table = runtime.access_table_view(
            table, device="npu", capacity=args.table_capacity
        )
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

        scores = scores_cpu.to(device)
        resident = resident_cpu.to(device)
        table_indices = table_indices_cpu.to(device)
        offsets = offsets_cpu.to(device)
        output_indices = output_indices_cpu.to(device)
        threshold = torch.tensor([0.5], dtype=torch.float32, device=device)
        frame_epoch = torch.empty(1, dtype=torch.int64, device=device)
        execution_mode = torch.empty(1, dtype=torch.int64, device=device)
        torch.npu.synchronize()
        next_epoch = args.frame_epoch - 1
        samples: dict[int, list[Telemetry]] = {
            PIPELINE_MODE_PARALLEL: [],
            PIPELINE_MODE_SERIAL_WAIT_ALL_CLOSED: [],
        }
        last_outputs: dict[int, tuple[torch.Tensor, ...]] = {}

        def run_one(
            mode: int, retain_telemetry: bool
        ) -> tuple[Timing, Telemetry | None]:
            nonlocal next_epoch
            # A P1 launch may issue work through an ACLNN path whose full
            # stream visibility is not equivalent to a Python Event.  Fence
            # before reusing the fixed-address input/output state so one
            # replay cannot write over an earlier replay's ring or output.
            # This fence is intentionally outside both compared intervals.
            torch.npu.synchronize()
            next_epoch += 1
            # These writes establish equal cold-bound state for both modes and
            # are deliberately excluded from the measured operator interval.
            # Their addresses remain graph stable across replays.
            frame_epoch.fill_(next_epoch)
            execution_mode.fill_(mode)
            # Equalize the cold-bound destination state without adding this
            # memset to either timing.  Payload content is not validated.
            packed_output.zero_()
            torch.npu.synchronize()

            # A terminal fence is the primary metric because it includes a
            # potentially host-synchronous ACLNN launch and catches work the
            # Event did not order with.  Do not insert Event tasks in a normal
            # performance replay: this bridge deliberately bypasses torch_npu
            # task-queue submission, so Event is opt-in diagnosis only.
            start = (
                torch.npu.Event(enable_timing=True) if args.event_diagnostic else None
            )
            end = (
                torch.npu.Event(enable_timing=True) if args.event_diagnostic else None
            )
            if start is not None:
                start.record()
            host_start_ns = time.perf_counter_ns()
            outputs = torch.ops.a5_exact_scan_get.exact_scan_speculative_get(
                scores,
                threshold,
                resident,
                table_indices,
                offsets,
                output_indices,
                device_table.addresses,
                device_table.valid,
                packed_output,
                fused_get_control,
                frame_epoch,
                pipeline_state,
                execution_mode,
            )
            host_submit_done_ns = time.perf_counter_ns()
            if end is not None:
                end.record()
            # The device-wide fence is the authoritative completion boundary.
            # It must stay outside the optional Event diagnostic path.
            torch.npu.synchronize()
            host_terminal_ns = time.perf_counter_ns()
            event_us: float | None = None
            if start is not None and end is not None:
                end.synchronize()
                event_us = start.elapsed_time(end) * 1000.0
            timing = Timing(
                submit_us=(host_submit_done_ns - host_start_ns) / 1000.0,
                terminal_us=(host_terminal_ns - host_start_ns) / 1000.0,
                event_us=event_us,
            )
            if retain_telemetry:
                # Retain only the final candidate arrays per mode.  Copying
                # their ~20 MiB result after every replay would not enter an
                # NPU event interval, but it would add needless host traffic
                # between paired samples.  Every replay still copies and
                # validates the small producer/Get/role telemetry below.
                last_outputs[mode] = outputs
                telemetry = validate_sample(
                    "parallel" if mode == PIPELINE_MODE_PARALLEL else "serial",
                    outputs,
                    packed_output,
                    pipeline_state,
                    expected,
                    expected_misses,
                    lane_output_rows,
                    mode,
                )
                return timing, telemetry
            return timing, None

        # Alternate mode order.  The two persistent Direct-HCCP queues cannot
        # systematically favor the mode that always follows a warm replay.
        for iteration in range(args.warmup):
            order = (
                (PIPELINE_MODE_PARALLEL, PIPELINE_MODE_SERIAL_WAIT_ALL_CLOSED)
                if iteration % 2 == 0
                else (PIPELINE_MODE_SERIAL_WAIT_ALL_CLOSED, PIPELINE_MODE_PARALLEL)
            )
            for mode in order:
                run_one(mode, retain_telemetry=False)

        timings: dict[int, list[Timing]] = {
            PIPELINE_MODE_PARALLEL: [],
            PIPELINE_MODE_SERIAL_WAIT_ALL_CLOSED: [],
        }
        pair_saved_us: list[float] = []
        for iteration in range(args.iters):
            order = (
                (PIPELINE_MODE_PARALLEL, PIPELINE_MODE_SERIAL_WAIT_ALL_CLOSED)
                if iteration % 2 == 0
                else (PIPELINE_MODE_SERIAL_WAIT_ALL_CLOSED, PIPELINE_MODE_PARALLEL)
            )
            pair: dict[int, Timing] = {}
            for mode in order:
                timing, telemetry = run_one(mode, retain_telemetry=True)
                if telemetry is None:
                    raise AssertionError("timed replay lost its telemetry")
                timings[mode].append(timing)
                samples[mode].append(telemetry)
                pair[mode] = timing
            pair_saved_us.append(
                pair[PIPELINE_MODE_SERIAL_WAIT_ALL_CLOSED].terminal_us
                - pair[PIPELINE_MODE_PARALLEL].terminal_us
            )

        parallel, serial = validate_cross_mode(
            samples[PIPELINE_MODE_PARALLEL],
            samples[PIPELINE_MODE_SERIAL_WAIT_ALL_CLOSED],
        )
        parallel_candidate_signature = candidate_signature(
            "parallel-final",
            last_outputs[PIPELINE_MODE_PARALLEL][1],
            last_outputs[PIPELINE_MODE_PARALLEL][2],
            expected,
        )
        serial_candidate_signature = candidate_signature(
            "serial-final",
            last_outputs[PIPELINE_MODE_SERIAL_WAIT_ALL_CLOSED][1],
            last_outputs[PIPELINE_MODE_SERIAL_WAIT_ALL_CLOSED][2],
            expected,
        )
        if parallel_candidate_signature != serial_candidate_signature:
            raise AssertionError(
                "strict modes produced different selected candidate signatures"
            )
        if parallel.first_doorbell >= parallel.producer_scan_done:
            raise AssertionError("parallel mode did not form a pre-scan Get window")
        if serial.first_doorbell < serial.serial_gate_open:
            raise AssertionError("serial mode opened doorbell before strict gate")

        parallel_timings = timings[PIPELINE_MODE_PARALLEL]
        serial_timings = timings[PIPELINE_MODE_SERIAL_WAIT_ALL_CLOSED]
        print(
            "P1_TIMING_CONTRACT "
            "primary_metric=host_monotonic_submit_to_device_idle "
            "terminal_fence=torch.npu.synchronize "
            f"event_diagnostic={int(args.event_diagnostic)} "
            "event_valid_for_result=0 "
            "pre_replay_fence=enabled cold_state_excluded=1",
            flush=True,
        )
        report_perf(
            "p1_exact_scan_speculative_get_parallel_terminal",
            [sample.terminal_us for sample in parallel_timings],
            args.warmup,
        )
        report_perf(
            "p1_exact_scan_speculative_get_serial_wait_all_closed_terminal",
            [sample.terminal_us for sample in serial_timings],
            args.warmup,
        )
        report_perf(
            "p1_exact_scan_speculative_get_parallel_submit",
            [sample.submit_us for sample in parallel_timings],
            args.warmup,
        )
        report_perf(
            "p1_exact_scan_speculative_get_serial_wait_all_closed_submit",
            [sample.submit_us for sample in serial_timings],
            args.warmup,
        )
        for label, samples_for_mode in (
            ("parallel", parallel_timings),
            ("serial", serial_timings),
        ):
            event_samples = [
                sample.event_us
                for sample in samples_for_mode
                if sample.event_us is not None
            ]
            event_fields = "event_enabled=0"
            if event_samples:
                event_mean = statistics.mean(event_samples)
                terminal_mean = statistics.mean(
                    sample.terminal_us for sample in samples_for_mode
                )
                event_fields = (
                    "event_enabled=1 "
                    f"event_mean_us={event_mean:.3f} "
                    f"terminal_minus_event_mean_us={terminal_mean - event_mean:.3f}"
                )
            print(
                "P1_TIMING_DIAGNOSTIC "
                f"mode={label} {event_fields} "
                f"post_submit_to_terminal_mean_us="
                f"{statistics.mean(sample.terminal_us - sample.submit_us for sample in samples_for_mode):.3f}",
                flush=True,
            )
        for label, telemetry in (("parallel", parallel), ("serial", serial)):
            print(
                "P1_STRICT_SAME_INPUT_TELEMETRY "
                f"mode={label} requested={telemetry.requested} batches={telemetry.batches} "
                f"consumed_chunks={telemetry.consumed_chunks} errors={telemetry.errors} "
                f"first_consumed_chunk_cycle={telemetry.first_consumed_chunk} "
                f"first_doorbell_cycle={telemetry.first_doorbell} "
                f"producer_scan_done_cycle={telemetry.producer_scan_done} "
                f"producer_closed_cycle={telemetry.producer_closed} "
                f"serial_gate_open_cycle={telemetry.serial_gate_open} "
                f"serial_gate_polls={telemetry.serial_gate_polls} "
                f"completion_done_cycle={telemetry.completion} "
                f"descriptor_sum={telemetry.descriptor_sum} "
                f"descriptor_xor={telemetry.descriptor_xor}"
            )

        parallel_mean = statistics.mean(
            sample.terminal_us for sample in parallel_timings
        )
        serial_mean = statistics.mean(
            sample.terminal_us for sample in serial_timings
        )
        saved_mean = statistics.mean(pair_saved_us)
        saved_pct = saved_mean * 100.0 / serial_mean if serial_mean else 0.0
        same_work = int(
            parallel.requested == serial.requested
            and parallel.batches == serial.batches
            and parallel.consumed_chunks == serial.consumed_chunks
            and (parallel.descriptor_sum, parallel.descriptor_xor)
            == (serial.descriptor_sum, serial.descriptor_xor)
            and parallel_candidate_signature == serial_candidate_signature
        )
        print(
            "P1_STRICT_SAME_INPUT_RESULT "
            "measurement=terminal_submit_to_device_idle "
            f"pairs={len(pair_saved_us)} parallel_terminal_mean_us={parallel_mean:.3f} "
            f"serial_terminal_mean_us={serial_mean:.3f} "
            f"serial_minus_parallel_terminal_mean_us={saved_mean:.3f} "
            f"serial_minus_parallel_terminal_p50_us={percentile(pair_saved_us, 0.50):.3f} "
            f"serial_minus_parallel_terminal_p90_us={percentile(pair_saved_us, 0.90):.3f} "
            f"parallel_reduction_vs_serial_pct={saved_pct:.2f} "
            f"same_descriptor_work={same_work} "
            f"candidate_sum={parallel_candidate_signature[0]} "
            f"candidate_xor={parallel_candidate_signature[1]} "
            "physical_id_affinity_claim=0 "
            "result_meaning=execution_scoped_role_partition_control_cq_path_only"
        )
        print("P1_STRICT_SAME_INPUT_PASS")
    finally:
        if control_bound:
            torch.npu.synchronize()
            runtime.release_fused_batch_get_control()
        if table is not None:
            runtime.release_access_table(table)
        runtime.shutdown()


if __name__ == "__main__":
    main()
