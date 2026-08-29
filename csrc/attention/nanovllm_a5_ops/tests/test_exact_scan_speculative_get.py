#!/usr/bin/env python3
"""Validate one real ExactScan -> chunked full SpeculativeGet launch.

This is intentionally a component test, not a vLLM end-to-end benchmark.  It
cold-binds a fixed KVAF Direct-HCCP output, then launches one AIV-only kernel.
Task lanes 0 and 1 run the two current full Get workers through CQ completion;
task lanes 2 through 63 threshold LI scores and publish 32-miss chunks.

The test supports both KVAF transport ABIs.  ``local`` is the default A5 CPU
DDR path; ``remote`` is available once an A5+K5 pair has matching KVAF builds.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch_npu  # noqa: F401

import kvaf
import nanovllm_exact_scan_get  # noqa: F401


PRODUCER_COUNT = 62
GET_LANE_COUNT = 2
GET_DESCRIPTOR_BUFFER_COUNT = 2
GET_WAVE_RECORDS = 256
ROWS_PER_ENTRY = 128
PACKED_ENTRY_BYTES = 656
ENTRY_BYTES = ROWS_PER_ENTRY * PACKED_ENTRY_BYTES
DMA_CACHELINE_BYTES = 64
ROWS_PER_DMA_CACHELINE = DMA_CACHELINE_BYTES // math.gcd(
    DMA_CACHELINE_BYTES, PACKED_ENTRY_BYTES
)
RING_SLOTS = 2
RECORDS_PER_RING_SLOT = 32
CACHE_LINE_WORDS = 8
# state, record count, and resume cursor occupy three cache lines.  This must
# stay aligned with the P5 executor's SPSC ring ABI; the overlap benchmark
# imports this constant directly when it allocates ``pipeline_state``.
RING_SLOT_RECORD_WORD = 3 * CACHE_LINE_WORDS
RING_WORDS_PER_SLOT = RING_SLOT_RECORD_WORD + RECORDS_PER_RING_SLOT * 3
PIPELINE_WORDS_PER_RING = RING_SLOTS * RING_WORDS_PER_SLOT + CACHE_LINE_WORDS
PIPELINE_RING_COUNT = PRODUCER_COUNT * GET_LANE_COUNT

# The two values are device-side scalar inputs to the same P1 operator.  They
# change only when a Get lane may start consuming SPSC rings; ExactScan never
# branches on them.  Keeping the mode as graph-stable data avoids a separate
# operator, stream, or host submission in the strict benchmark.
PIPELINE_MODE_PARALLEL = 0
PIPELINE_MODE_SERIAL_WAIT_ALL_CLOSED = 1

CANDIDATE_HBM_HIT = 0
CANDIDATE_DDR_ENQUEUED = 1
CANDIDATE_DDR_DEFERRED = 2
FUSED_CONTROL_MAGIC = 0x4B56414646555345
FUSED_CONTROL_VERSION = 1


@dataclass(frozen=True)
class Candidate:
    source: int
    kind: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:1")
    parser.add_argument("--transport", choices=("local", "remote"), default="local")
    parser.add_argument("--cpu-service-addr", default="")
    parser.add_argument("--batch-get-kernel", required=True)
    parser.add_argument("--batch-put-kernel", required=True)
    parser.add_argument("--table-capacity", type=int, default=32)
    parser.add_argument("--scan-per-producer", type=int, default=4096)
    parser.add_argument("--misses-per-lane", type=int, default=32)
    parser.add_argument("--hits-per-lane", type=int, default=2)
    parser.add_argument("--frame-epoch", type=int, default=1)
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if args.table_capacity <= 0:
        raise ValueError("--table-capacity must be positive")
    if args.scan_per_producer <= 0:
        raise ValueError("--scan-per-producer must be positive")
    if args.misses_per_lane != RECORDS_PER_RING_SLOT:
        raise ValueError(
            "P1 validation deliberately uses exactly one 32-record chunk per "
            "producer/lane; set --misses-per-lane=32"
        )
    if args.hits_per_lane < 1:
        raise ValueError("--hits-per-lane must be positive")
    if args.frame_epoch <= 0:
        raise ValueError("--frame-epoch must be nonzero")
    needed = 2 * (args.misses_per_lane + args.hits_per_lane)
    if args.scan_per_producer < needed:
        raise ValueError(
            "--scan-per-producer is too small to place the requested two-lane "
            "miss/hit pattern"
        )
    if args.transport == "remote" and not args.cpu_service_addr:
        raise ValueError("--cpu-service-addr is required for --transport=remote")


def create_runtime(args: argparse.Namespace, device_id: int) -> object:
    if args.transport == "local":
        runtime = kvaf.NpuRuntime(
            args.batch_get_kernel,
            device_id,
            2,
            args.batch_put_kernel,
            1,
        )
        if callable(getattr(runtime, "link_to", None)):
            raise RuntimeError(
                "local transport selected, but this KVAF build exposes the "
                "remote CPU-service ABI"
            )
        return runtime

    runtime = kvaf.NpuRuntime(
        args.batch_get_kernel,
        PACKED_ENTRY_BYTES,
        ROWS_PER_ENTRY,
        device_id,
        2,
        args.batch_put_kernel,
        1,
    )
    link_to = getattr(runtime, "link_to", None)
    if not callable(link_to):
        raise RuntimeError(
            "remote transport selected, but this KVAF build does not expose "
            "the CPU-service ABI"
        )
    link_to(args.cpu_service_addr)
    return runtime


def build_exact_scan_inputs(args: argparse.Namespace) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[list[Candidate]],
    int,
    int,
]:
    """Build one deterministic, lane-balanced P1 input fixture.

    The strict parallel-versus-serial benchmark uses this *same* set of NPU
    tensors for both modes.  It intentionally does not move candidates late:
    the execution-mode scalar, rather than a changed input placement, creates
    the serial reference.
    """

    score_count = PRODUCER_COUNT * args.scan_per_producer
    scores = torch.zeros(score_count, dtype=torch.float32)
    resident = torch.ones(score_count, dtype=torch.uint8)
    table_indices = torch.zeros(score_count, dtype=torch.int32)
    offsets = torch.zeros(score_count, dtype=torch.int32)
    output_indices = torch.zeros(score_count, dtype=torch.int32)
    expected: list[list[Candidate]] = [[] for _ in range(PRODUCER_COUNT)]
    # A 656-byte transfer advances the next row by 16 bytes within a 64-byte
    # cache line.  Two independent Direct-HCCP queues must therefore never
    # scatter into interleaved rows: adjacent rows from different queues would
    # concurrently own pieces of the same cache line.  Reserve one aligned,
    # lane-private slab for each Get AIV instead.
    lane_output_rows = PRODUCER_COUNT * args.misses_per_lane
    if lane_output_rows % ROWS_PER_DMA_CACHELINE != 0:
        raise AssertionError(
            "P1 test requires each Get lane's output slab to start on a "
            "DMA cache-line boundary"
        )
    lane_output_counts = [0, 0]

    for producer in range(PRODUCER_COUNT):
        lane_scan_counts = [0, 0]
        lane_seen = [0, 0]
        selected_per_lane = args.misses_per_lane + args.hits_per_lane
        selected_by_step: dict[int, int] = {}
        for step in range(args.scan_per_producer):
            source = producer + step * PRODUCER_COUNT
            lane = (source >> 7) & 1
            lane_scan_counts[lane] += 1
            rank_in_lane = lane_seen[lane]
            lane_seen[lane] += 1
            if rank_in_lane < args.misses_per_lane:
                selected_by_step[step] = CANDIDATE_DDR_ENQUEUED
            elif rank_in_lane < selected_per_lane:
                selected_by_step[step] = CANDIDATE_HBM_HIT
        if min(lane_scan_counts) < selected_per_lane:
            raise AssertionError(
                "scan is too short to place the requested candidates in both "
                f"lanes for producer={producer}: counts={lane_scan_counts}"
            )
        misses = [0, 0]
        hits = [0, 0]
        for step in range(args.scan_per_producer):
            source = producer + step * PRODUCER_COUNT
            lane = (source >> 7) & 1
            kind = selected_by_step.get(step)
            if kind == CANDIDATE_DDR_ENQUEUED:
                output_row = lane * lane_output_rows + lane_output_counts[lane]
                scores[source] = 1.0
                resident[source] = 0
                table_indices[source] = output_row % args.table_capacity
                offsets[source] = (
                    (output_row // args.table_capacity) % ROWS_PER_ENTRY
                ) * PACKED_ENTRY_BYTES
                output_indices[source] = output_row
                expected[producer].append(
                    Candidate(source, CANDIDATE_DDR_ENQUEUED)
                )
                lane_output_counts[lane] += 1
                misses[lane] += 1
            elif kind == CANDIDATE_HBM_HIT:
                scores[source] = 1.0
                resident[source] = 1
                expected[producer].append(Candidate(source, CANDIDATE_HBM_HIT))
                hits[lane] += 1
        if misses != [args.misses_per_lane, args.misses_per_lane] or hits != [
            args.hits_per_lane,
            args.hits_per_lane,
        ]:
            raise AssertionError(
                f"unable to construct balanced producer pattern {producer}: "
                f"misses={misses} hits={hits}"
            )
    if lane_output_counts != [lane_output_rows, lane_output_rows]:
        raise AssertionError(
            "lane-private output slabs did not receive the expected number "
            f"of misses: counts={lane_output_counts} rows={lane_output_rows}"
        )
    return (
        scores,
        resident,
        table_indices,
        offsets,
        output_indices,
        expected,
        sum(lane_output_counts),
        lane_output_rows,
    )


def build_ddr_payload(
    table_capacity: int, output_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-entry DDR bytes and the exact expected recalled rows."""

    physical_rows = torch.arange(
        table_capacity * ROWS_PER_ENTRY, dtype=torch.int64
    ).unsqueeze(1)
    byte_ids = torch.arange(PACKED_ENTRY_BYTES, dtype=torch.int64).unsqueeze(0)
    rows = (
        physical_rows * 29 + byte_ids * 7 + (physical_rows >> 3)
    ).remainder(256).to(torch.uint8)
    entry_payload = rows.reshape(table_capacity, ENTRY_BYTES).contiguous()
    recalled_physical_rows = torch.tensor(
        [
            (output_row % table_capacity) * ROWS_PER_ENTRY
            + ((output_row // table_capacity) % ROWS_PER_ENTRY)
            for output_row in range(output_rows)
        ],
        dtype=torch.int64,
    )
    return entry_payload, rows.index_select(0, recalled_physical_rows)


def validate_result(
    outputs: tuple[torch.Tensor, ...],
    packed_output: torch.Tensor,
    pipeline_state: torch.Tensor,
    expected: list[list[Candidate]],
    expected_misses: int,
    expected_payload: torch.Tensor,
    source_payload: torch.Tensor,
    lane_output_rows: int,
) -> None:
    (
        packed_output_out,
        candidate_indices,
        candidate_kind,
        producer_stats,
        get_stats,
        role_trace,
        get_indices,
        get_offsets,
        get_output_indices,
        get_mask,
        pipeline_state_out,
    ) = outputs
    if packed_output_out.data_ptr() != packed_output.data_ptr():
        raise AssertionError("CANN packed_output_out did not alias the bound HBM output")
    if pipeline_state_out.data_ptr() != pipeline_state.data_ptr():
        raise AssertionError("CANN pipeline_state_out did not alias graph-stable state")
    expected_descriptor_shape = (
        GET_LANE_COUNT,
        GET_DESCRIPTOR_BUFFER_COUNT,
        GET_WAVE_RECORDS,
    )
    for name, value in (
        ("get_indices", get_indices),
        ("get_offsets", get_offsets),
        ("get_output_indices", get_output_indices),
        ("get_mask", get_mask),
    ):
        if tuple(value.shape) != expected_descriptor_shape:
            raise AssertionError(
                f"{name} P2 double-buffer shape={tuple(value.shape)} "
                f"expected={expected_descriptor_shape}"
            )

    candidate_indices_cpu = candidate_indices.cpu()
    candidate_kind_cpu = candidate_kind.cpu()
    stats = producer_stats.cpu()
    get = get_stats.cpu()
    trace = role_trace.cpu()

    # Keep the first device-side Get error visible in the concise test log.
    # This runs only on failure, so it does not add a normal-path host copy.
    if not bool(torch.all(get[:, 2] == 0)):
        pipeline_words = int(torch.count_nonzero(pipeline_state_out.cpu()))
        print(
            "A5_EXACT_SCAN_SPECULATIVE_GET_DIAGNOSTIC "
            f"get_stats={get.tolist()} "
            f"pipeline_nonzero_words={pipeline_words}"
        )

    expected_selected = len(expected[0])
    expected_hits = sum(item.kind == CANDIDATE_HBM_HIT for item in expected[0])
    expected_miss_per_producer = expected_selected - expected_hits
    selected_bad = torch.nonzero(
        stats[:, 0] != expected_selected, as_tuple=False
    ).flatten()
    if selected_bad.numel() != 0:
        producer_rows = stats.index_select(0, selected_bad).tolist()
        get_errors = [int(value) for value in get[:, 2].tolist()]
        get_control_masks = [
            (value >> 16) if (value & 0xFFFF) == 0x1001 else 0
            for value in get_errors
        ]
        print(
            "A5_EXACT_SCAN_SPECULATIVE_GET_DIAGNOSTIC "
            f"producer_selected_bad_count={selected_bad.numel()} "
            f"producer_ids={selected_bad.tolist()} "
            f"selected_values={stats[selected_bad, 0].tolist()} "
            f"producer_rows={producer_rows} "
            f"get_errors={get_errors} "
            f"get_control_masks={get_control_masks} "
            f"role_trace={trace.tolist()}"
        )
    torch.testing.assert_close(
        stats[:, 0], torch.full((PRODUCER_COUNT,), expected_selected, dtype=torch.int64)
    )
    torch.testing.assert_close(
        stats[:, 1], torch.full((PRODUCER_COUNT,), expected_hits, dtype=torch.int64)
    )
    torch.testing.assert_close(
        stats[:, 2],
        torch.full((PRODUCER_COUNT,), expected_miss_per_producer, dtype=torch.int64),
    )
    torch.testing.assert_close(stats[:, 3], stats[:, 2])
    torch.testing.assert_close(stats[:, 4], torch.zeros(PRODUCER_COUNT, dtype=torch.int64))
    if not bool(torch.all(stats[:, 5] > 0)) or not bool(torch.all(stats[:, 6] > 0)):
        raise AssertionError("producer did not report first-chunk and scan-done cycles")

    for producer, entries in enumerate(expected):
        for rank, item in enumerate(entries):
            flat = rank * PRODUCER_COUNT + producer
            if candidate_indices_cpu[flat].item() != item.source:
                raise AssertionError(
                    f"candidate source mismatch producer={producer} rank={rank}"
                )
            if candidate_kind_cpu[flat].item() != item.kind:
                raise AssertionError(
                    f"candidate kind mismatch producer={producer} rank={rank}"
                )

    torch.testing.assert_close(get[:, 2], torch.zeros(GET_LANE_COUNT, dtype=torch.int64))
    torch.testing.assert_close(get[:, 0].sum(), torch.tensor(expected_misses, dtype=torch.int64))
    torch.testing.assert_close(
        get[:, 8], torch.full((GET_LANE_COUNT,), PIPELINE_MODE_PARALLEL, dtype=torch.int64)
    )
    torch.testing.assert_close(get[:, 9], torch.zeros(GET_LANE_COUNT, dtype=torch.int64))
    if not bool(torch.all(get[:, 1] > 0)):
        raise AssertionError("one of the two Get lanes never issued a full Get")
    if not bool(torch.all(get[:, 3] > 0)) or not bool(torch.all(get[:, 4] > 0)):
        raise AssertionError("Get lanes did not report doorbell/CQ completion cycles")
    if int(get[:, 3].min().item()) >= int(stats[:, 6].max().item()):
        raise AssertionError(
            "first Direct-HCCP doorbell was not observed before ExactScan completed; "
            "the chunk pipeline did not form"
        )
    expected_trace = torch.cat(
        (
            torch.arange(2000, 2000 + GET_LANE_COUNT, dtype=torch.int32),
            torch.arange(1000, 1000 + PRODUCER_COUNT, dtype=torch.int32),
        )
    )
    torch.testing.assert_close(trace, expected_trace)

    payload = packed_output[:expected_misses].cpu()
    if not bool(torch.equal(payload, expected_payload)):
        mismatch_rows = torch.nonzero(
            torch.any(payload != expected_payload, dim=1), as_tuple=False
        ).flatten()
        source_rows = source_payload.reshape(-1, PACKED_ENTRY_BYTES)
        samples = []
        for row_tensor in mismatch_rows[:8]:
            row = int(row_tensor)
            actual = payload[row]
            expected_row = expected_payload[row]
            bad_bytes = torch.nonzero(
                actual != expected_row, as_tuple=False
            ).flatten()
            # The generated row pattern is deterministic.  Matching it back
            # to the DDR source distinguishes an incorrect descriptor/WQE
            # route from a missing or torn HCCP destination write.
            prefix_matches = torch.nonzero(
                torch.all(source_rows[:, :32] == actual[:32], dim=1),
                as_tuple=False,
            ).flatten()
            source_matches = [
                int(source_row)
                for source_row in prefix_matches[:8]
                if bool(torch.equal(source_rows[int(source_row)], actual))
            ]
            samples.append(
                {
                    "row": row,
                    "owner_lane": row // lane_output_rows,
                    "row_mod_4": row % ROWS_PER_DMA_CACHELINE,
                    "first_bad_byte": int(bad_bytes[0]),
                    "last_bad_byte": int(bad_bytes[-1]),
                    "actual_first16": actual[:16].tolist(),
                    "expected_first16": expected_row[:16].tolist(),
                    "actual_last16": actual[-16:].tolist(),
                    "expected_last16": expected_row[-16:].tolist(),
                    "source_rows": source_matches,
                }
            )
        descriptor_heads = []
        get_indices_cpu = get_indices.cpu()
        get_offsets_cpu = get_offsets.cpu()
        get_output_indices_cpu = get_output_indices.cpu()
        get_mask_cpu = get_mask.cpu()
        for lane in range(GET_LANE_COUNT):
            descriptor_heads.append(
                {
                    "lane": lane,
                    "buffer0_indices": get_indices_cpu[lane, 0, :8].tolist(),
                    "buffer0_offsets": get_offsets_cpu[lane, 0, :8].tolist(),
                    "buffer0_output_indices": get_output_indices_cpu[
                        lane, 0, :8
                    ].tolist(),
                    "buffer0_mask": get_mask_cpu[lane, 0, :8].tolist(),
                }
            )
        print(
            "A5_EXACT_SCAN_SPECULATIVE_GET_PAYLOAD_DIAGNOSTIC "
            f"mismatch_rows={mismatch_rows.numel()} "
            f"zero_rows={int(torch.all(payload == 0, dim=1).sum())} "
            f"get_stats={get.tolist()} "
            f"samples={samples} "
            f"descriptor_heads={descriptor_heads}"
        )
        raise AssertionError("full Direct-HCCP Get payload mismatch")


def main() -> None:
    args = parse_args()
    check_args(args)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    device_id = device.index if device.index is not None else torch.npu.current_device()
    get_device_name = getattr(
        torch.npu, "get_device_name", torch_npu.npu.get_device_name
    )
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
    output_rows = expected_misses
    entry_payload_cpu, expected_payload_cpu = build_ddr_payload(
        args.table_capacity, output_rows
    )
    keys = list(range(1, args.table_capacity + 1))
    ddr_bytes = args.table_capacity * ENTRY_BYTES
    runtime = create_runtime(args, device_id)
    table = None
    control_bound = False
    try:
        print(
            "A5_EXACT_SCAN_SPECULATIVE_GET_CONFIG "
            f"device={args.device} device_name={device_name!r} "
            f"transport={args.transport} score_count={scores_cpu.numel()} "
            f"table_capacity={args.table_capacity} output_rows={output_rows} "
            f"producer_aiv=62 get_aiv=2 chunk_records=32 "
            f"get_task_lanes=0,1 producer_task_lanes=2..63 "
            f"get_bootstrap_records=64 descriptor_buffers={GET_DESCRIPTOR_BUFFER_COUNT} "
            "parallel_post_only_try_retire=1 serial_blocking_helper=1 "
            f"schedule_mode=batch ring_cacheline_isolated=1 "
            f"output_layout=lane_private_64B_aligned "
            f"lane_rows={lane_output_rows}"
        )
        runtime.initialize(ddr_bytes, ENTRY_BYTES)
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
        torch.npu.synchronize()
        # KVAF owns every byte of this output after the cold bind, and writes
        # every requested row through Direct-HCCP.  Do not queue a pointless
        # torch.zeros on the producer stream: a raw ACL control publication is
        # outside normal PyTorch stream ordering until the bind establishes its
        # cold boundary.
        packed_output = torch.empty(
            (output_rows, PACKED_ENTRY_BYTES), dtype=torch.uint8, device=device
        )
        # Keep this asynchronous zero-fill intentionally.  It is the exact
        # producer-side write that a cold KVAF bind must order before publishing
        # its 16-word ABI; otherwise the producer can overwrite the raw ACL
        # copy after BindFusedBatchGetControl returns.
        fused_get_control = torch.zeros(16, dtype=torch.int64, device=device)
        pipeline_state = torch.zeros(
            (PIPELINE_RING_COUNT, PIPELINE_WORDS_PER_RING),
            dtype=torch.int64,
            device=device,
        )
        bind_stream = torch.npu.current_stream()
        runtime.bind_fused_batch_get_control(
            packed_output, fused_get_control, bind_stream
        )
        control_bound = True
        torch.npu.synchronize()
        control_snapshot = [int(value) for value in fused_get_control.cpu().tolist()]
        print(
            "A5_EXACT_SCAN_SPECULATIVE_GET_CONTROL_BOUND "
            "runtime_get_workers=2 fixed_output=1 control_words=16 "
            "ordinary_get_put_free_blocked=1"
        )
        print(
            "A5_EXACT_SCAN_SPECULATIVE_GET_CONTROL_SNAPSHOT "
            f"words0_9={control_snapshot[:10]} "
            f"output_data_ptr={packed_output.data_ptr()} "
            f"output_rows={packed_output.shape[0]}"
        )
        control_mismatch = []
        if control_snapshot[0] != FUSED_CONTROL_MAGIC:
            control_mismatch.append("magic")
        if control_snapshot[1] != FUSED_CONTROL_VERSION:
            control_mismatch.append("version")
        for index, name in (
            (2, "channel_entity"),
            (3, "queue_state"),
            (4, "core_status"),
        ):
            if control_snapshot[index] == 0:
                control_mismatch.append(name)
        if control_snapshot[5] != packed_output.data_ptr():
            control_mismatch.append("output")
        if control_snapshot[6] != output_rows:
            control_mismatch.append("output_rows")
        if control_snapshot[7] != GET_LANE_COUNT:
            control_mismatch.append("block_count")
        if control_snapshot[8] != 1:
            control_mismatch.append("jettys_per_core")
        if control_snapshot[9] != PACKED_ENTRY_BYTES:
            control_mismatch.append("block_bytes")
        if control_mismatch:
            raise RuntimeError(
                "KVAF fused control was not published before P1 launch: "
                + ",".join(control_mismatch)
            )

        outputs = torch.ops.a5_exact_scan_get.exact_scan_speculative_get(
            scores_cpu.to(device),
            torch.tensor([0.5], dtype=torch.float32, device=device),
            resident_cpu.to(device),
            table_indices_cpu.to(device),
            offsets_cpu.to(device),
            output_indices_cpu.to(device),
            address_table,
            valid_table,
            packed_output,
            fused_get_control,
            torch.tensor([args.frame_epoch], dtype=torch.int64, device=device),
            pipeline_state,
            torch.tensor(
                [PIPELINE_MODE_PARALLEL], dtype=torch.int64, device=device
            ),
        )
        torch.npu.synchronize()
        validate_result(
            outputs,
            packed_output,
            pipeline_state,
            expected,
            expected_misses,
            expected_payload_cpu,
            entry_payload_cpu,
            lane_output_rows,
        )
        producer_stats = outputs[3].cpu()
        get_stats = outputs[4].cpu()
        print(
            "A5_EXACT_SCAN_SPECULATIVE_GET_CORRECTNESS "
            f"selected={int(producer_stats[:, 0].sum())} "
            f"hbm_hit={int(producer_stats[:, 1].sum())} "
            f"ddr_miss={int(producer_stats[:, 2].sum())} "
            f"enqueued={int(producer_stats[:, 3].sum())} "
            f"deferred={int(producer_stats[:, 4].sum())} "
            f"full_get_requested={int(get_stats[:, 0].sum())} "
            f"full_get_batches={int(get_stats[:, 1].sum())}"
        )
        print(
            "A5_EXACT_SCAN_SPECULATIVE_GET_PIPELINE "
            f"first_doorbell_cycle_min={int(get_stats[:, 3].min())} "
            f"producer_scan_done_cycle_max={int(producer_stats[:, 6].max())} "
            "doorbell_before_scan_done=1"
        )
        print("A5_EXACT_SCAN_SPECULATIVE_GET_PASS")
    finally:
        # This is the terminal-reader boundary for the component test.  The
        # Runtime intentionally does not hide a synchronization in release.
        if control_bound:
            torch.npu.synchronize()
            runtime.release_fused_batch_get_control()
        if table is not None:
            runtime.release_access_table(table)
        runtime.shutdown()


if __name__ == "__main__":
    main()
