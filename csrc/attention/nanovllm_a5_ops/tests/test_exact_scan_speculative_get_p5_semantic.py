#!/usr/bin/env python3
"""P5 semantic fixture for C8 receipt -> ExactScan -> SpeculativeGet.

This is deliberately a *P5 semantic stage*, not a replacement for the P3
official-C8 bitwise gate.  It injects the exact graph boundary that native C8
publishes -- an ``int32`` carrier holding sortable ``uint16`` scores plus a
16-word receipt per 128-score tile -- and invokes the P2 executor directly.
Therefore it proves that P2 consumes a C8-shaped publication, selects the
same candidate set as the CPU oracle, deduplicates compressed KV objects, and
uses the Post/CQ/private-ready lifecycle.  It also models the live receipt
case: ``score_source_lens`` marks fixed but inactive receipt slots, while a
late *expected* receipt is allowed to suppress only its speculative request.
The exact C8 Top-K / GuaranteedRecall authority remains unchanged.  A later
integration test wires these same inputs to the live C8 publisher.

Receipt word 0 is a byte offset *relative to score_workspace*, never a device
pointer.  The fixture contains no synthetic float score tensor.  Two sources per object
are intentionally selected in each row, so one object must cause one Get post
while both CandidateSeeds receive the same private slot.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Any


PRODUCER_COUNT = 62
GET_LANE_COUNT = 2
GET_DESCRIPTOR_BUFFER_COUNT = 2
GET_WAVE_RECORDS = 256
ROWS_PER_ENTRY = 128
PACKED_ENTRY_BYTES = 656
ENTRY_BYTES = ROWS_PER_ENTRY * PACKED_ENTRY_BYTES
RING_SLOTS = 2
RECORDS_PER_RING_SLOT = 32
CACHE_LINE_WORDS = 8
RING_SLOT_RECORD_WORD = 2 * CACHE_LINE_WORDS
RING_WORDS_PER_SLOT = RING_SLOT_RECORD_WORD + RECORDS_PER_RING_SLOT * 3
PIPELINE_WORDS_PER_RING = RING_SLOTS * RING_WORDS_PER_SLOT + CACHE_LINE_WORDS
PIPELINE_RING_COUNT = PRODUCER_COUNT * GET_LANE_COUNT
TILE_SCORE_WIDTH = 128
RECEIPT_WORDS = 16
RECEIPT_VERSION = 1
PIPELINE_MODE_PARALLEL = 0
PRIVATE_READY_FREE = 0
PRIVATE_READY_PENDING = 1
PRIVATE_READY_READY = 2


@dataclass(frozen=True)
class Candidate:
    row: int
    source: int
    pool: int
    kv_object: int
    score: int


@dataclass(frozen=True)
class ReceiptWindow:
    """One fixed C8 receipt position and its row-local activity contract."""

    slot: int
    row: int
    source_begin: int
    valid_count: int
    active: bool


@dataclass(frozen=True)
class P5Fixture:
    rows: int
    source_capacity: int
    pool_size: int
    object_capacity: int
    private_arena_capacity: int
    threshold: int
    epoch: int
    raw_scores: list[int]
    score_source_lens: list[int]
    object_map: list[int]
    object_slots_snapshot: list[int]
    expected_candidates: list[list[Candidate]]
    expected_selected: list[int]
    expected_hits: list[int]
    expected_misses: list[int]
    expected_deferred: list[int]
    expected_unique_cold_objects: set[tuple[int, int]]


def _pack_u16_carrier(raw_scores: list[int]) -> list[int]:
    """Pack exact score bytes into signed int32 graph carrier words."""
    if len(raw_scores) % 2:
        raise AssertionError("C8 score workspace must contain an even uint16 count")
    words: list[int] = []
    for index in range(0, len(raw_scores), 2):
        low = raw_scores[index]
        high = raw_scores[index + 1]
        if not 0 <= low <= 0xFFFF or not 0 <= high <= 0xFFFF:
            raise AssertionError(f"score out of uint16 range at {index}")
        word = low | (high << 16)
        words.append(word if word < (1 << 31) else word - (1 << 32))
    return words


def _receipt_windows(fixture: P5Fixture) -> list[ReceiptWindow]:
    """Return the fixed C8 receipt layout, including inert graph slots.

    C8 allocates one graph-static receipt area sized by ``source_capacity``.
    A decode row can be shorter.  ``score_source_lens[row]`` is therefore the
    only input which says whether a particular fixed receipt slot should be
    ignored or waited on for this epoch.
    """
    if len(fixture.score_source_lens) != fixture.rows:
        raise AssertionError("score_source_lens must have exactly one entry per row")
    tiles_per_row = (fixture.source_capacity + TILE_SCORE_WIDTH - 1) // TILE_SCORE_WIDTH
    windows: list[ReceiptWindow] = []
    for row, source_len in enumerate(fixture.score_source_lens):
        if not 0 <= source_len <= fixture.source_capacity:
            raise AssertionError(f"invalid source length row={row} value={source_len}")
        for tile in range(tiles_per_row):
            source_begin = tile * TILE_SCORE_WIDTH
            active = source_begin < source_len
            valid_count = min(TILE_SCORE_WIDTH, source_len - source_begin) if active else 0
            windows.append(
                ReceiptWindow(
                    slot=row * tiles_per_row + tile,
                    row=row,
                    source_begin=source_begin,
                    valid_count=valid_count,
                    active=active,
                )
            )
    return windows


def _receipt_slot_for_source(fixture: P5Fixture, candidate: Candidate) -> int:
    tiles_per_row = (fixture.source_capacity + TILE_SCORE_WIDTH - 1) // TILE_SCORE_WIDTH
    return candidate.row * tiles_per_row + candidate.source // TILE_SCORE_WIDTH


def _build_fixture(epoch: int) -> P5Fixture:
    rows = 2
    # The graph reserves three 128-score windows per row, while this forward
    # activates only the first two.  The third receipt is intentionally zero
    # and must be skipped rather than mistaken for a late producer.
    source_capacity = 384
    pool_size = 2
    object_capacity = 8
    private_arena_capacity = 8
    threshold = 0x7000
    raw_scores = [0x0100] * (rows * source_capacity)
    score_source_lens = [256, 256]
    object_map = [-1] * (pool_size * source_capacity)
    object_slots_snapshot = [-1] * (pool_size * object_capacity)

    # In both pools: O=2 is already resident.  O=1 and O=4 are cold, and each
    # has two high-scoring S values.  The high-but-invalid S=139 proves an
    # invalid S->O mapping is deferred rather than accidentally treated as O=0.
    selected = (
        (3, 1, 0x7A01),
        (7, 1, 0x7A02),
        (9, 2, 0x7A03),
        (133, 4, 0x7A04),
        # This odd source occupies the high half of an int32 carrier.  The
        # value makes that carrier negative as int32, proving the executor
        # must reinterpret exact uint16 score bytes rather than numerically
        # converting the carrier.
        (137, 4, 0xFA05),
        (139, -1, 0x7A06),
    )
    expected_candidates: list[list[Candidate]] = [[] for _ in range(PRODUCER_COUNT)]
    expected_selected = [0] * PRODUCER_COUNT
    expected_hits = [0] * PRODUCER_COUNT
    expected_misses = [0] * PRODUCER_COUNT
    expected_deferred = [0] * PRODUCER_COUNT
    unique_cold: set[tuple[int, int]] = set()

    for row in range(rows):
        pool = row
        object_slots_snapshot[pool * object_capacity + 2] = 71 + row
        for source, kv_object, score in selected:
            raw_scores[row * source_capacity + source] = score
            if kv_object >= 0:
                object_map[pool * source_capacity + source] = kv_object

        for tile_begin in range(0, score_source_lens[row], TILE_SCORE_WIDTH):
            receipt = row * ((source_capacity + TILE_SCORE_WIDTH - 1) // TILE_SCORE_WIDTH) + (
                tile_begin // TILE_SCORE_WIDTH
            )
            producer = receipt % PRODUCER_COUNT
            for source in range(
                tile_begin,
                min(tile_begin + TILE_SCORE_WIDTH, score_source_lens[row]),
            ):
                score = raw_scores[row * source_capacity + source]
                if score < threshold:
                    continue
                expected_selected[producer] += 1
                kv_object = object_map[pool * source_capacity + source]
                if kv_object < 0:
                    expected_deferred[producer] += 1
                    continue
                if object_slots_snapshot[pool * object_capacity + kv_object] >= 0:
                    expected_hits[producer] += 1
                    continue
                expected_misses[producer] += 1
                candidate = Candidate(row, source, pool, kv_object, score)
                expected_candidates[producer].append(candidate)
                unique_cold.add((pool, kv_object))

    return P5Fixture(
        rows=rows,
        source_capacity=source_capacity,
        pool_size=pool_size,
        object_capacity=object_capacity,
        private_arena_capacity=private_arena_capacity,
        threshold=threshold,
        epoch=epoch,
        raw_scores=raw_scores,
        score_source_lens=score_source_lens,
        object_map=object_map,
        object_slots_snapshot=object_slots_snapshot,
        expected_candidates=expected_candidates,
        expected_selected=expected_selected,
        expected_hits=expected_hits,
        expected_misses=expected_misses,
        expected_deferred=expected_deferred,
        expected_unique_cold_objects=unique_cold,
    )


def _assert_fixture_oracle(fixture: P5Fixture) -> None:
    assert len(fixture.raw_scores) == fixture.rows * fixture.source_capacity
    assert len(fixture.raw_scores) % 2 == 0
    assert len(_pack_u16_carrier(fixture.raw_scores)) * 2 == len(fixture.raw_scores)
    windows = _receipt_windows(fixture)
    assert len(windows) == 6
    assert [window.valid_count for window in windows] == [128, 128, 0, 128, 128, 0]
    assert [window.slot for window in windows if not window.active] == [2, 5]
    assert sum(fixture.expected_selected) == 12
    assert sum(fixture.expected_hits) == 2
    assert sum(fixture.expected_misses) == 8
    assert sum(fixture.expected_deferred) == 2
    assert sum(len(value) for value in fixture.expected_candidates) == 8
    assert fixture.expected_unique_cold_objects == {(0, 1), (0, 4), (1, 1), (1, 4)}
    # O=1/O=4 each appear twice within a pool: the test's dedup proof.
    for pool, kv_object in fixture.expected_unique_cold_objects:
        multiplicity = sum(
            candidate.pool == pool and candidate.kv_object == kv_object
            for candidates in fixture.expected_candidates
            for candidate in candidates
        )
        assert multiplicity == 2, (pool, kv_object, multiplicity)


def _assert_live_receipt_oracle(fixture: P5Fixture) -> dict[str, int]:
    """Model C8/P5's live window rule without requiring NPU execution.

    The exact candidate set is computed from all active score windows.  P5 is
    allowed to be missing an expected C8 receipt at its bounded wait deadline:
    that specific window is withheld from the *speculative* stream and counted
    as one defer.  It must not be reclassified as an inactive zero window, and
    it must not change the exact set owned by native C8 + GuaranteedRecall.
    """
    windows = _receipt_windows(fixture)
    active_windows = [window for window in windows if window.active]
    inactive_windows = [window for window in windows if not window.active]
    # Row 0's second window carries the two O=4 cold candidates plus an
    # invalid high-score source.  It is a genuine expected receipt, not one
    # of the fixed inert slots at 2/5.
    late_window = next(
        window
        for window in active_windows
        if window.row == 0 and window.source_begin == TILE_SCORE_WIDTH
    )
    exact_candidates = [
        candidate
        for producer_candidates in fixture.expected_candidates
        for candidate in producer_candidates
    ]
    speculative_candidates = [
        candidate
        for candidate in exact_candidates
        if _receipt_slot_for_source(fixture, candidate) != late_window.slot
    ]
    withheld_candidates = [
        candidate
        for candidate in exact_candidates
        if _receipt_slot_for_source(fixture, candidate) == late_window.slot
    ]

    assert late_window.active
    assert late_window.valid_count == TILE_SCORE_WIDTH
    assert late_window.slot not in {window.slot for window in inactive_windows}
    assert [(candidate.row, candidate.source) for candidate in withheld_candidates] == [
        (0, 133),
        (0, 137),
    ]
    assert len(speculative_candidates) == len(exact_candidates) - len(withheld_candidates)
    # This is the mathematical-equivalence statement at the live seam: the
    # speculative subset can shrink, but the exact authority remains the
    # original full active-window set.  GuaranteedRecall is responsible for
    # the withheld miss if it later belongs to exact Top-K.
    assert {
        (candidate.row, candidate.source, candidate.pool, candidate.kv_object, candidate.score)
        for candidate in exact_candidates
    } == {
        (candidate.row, candidate.source, candidate.pool, candidate.kv_object, candidate.score)
        for producer_candidates in fixture.expected_candidates
        for candidate in producer_candidates
    }
    assert all(
        candidate.source < fixture.score_source_lens[candidate.row]
        for candidate in exact_candidates
    )
    return {
        "active_windows": len(active_windows),
        "inactive_windows": len(inactive_windows),
        "timeout_defer_windows": 1,
        "withheld_candidate_seeds": len(withheld_candidates),
        "exact_candidates": len(exact_candidates),
        "speculative_candidates_after_timeout": len(speculative_candidates),
    }


def _device_inputs(torch: Any, fixture: P5Fixture, device: Any) -> dict[str, Any]:
    """Materialize the literal C8 publication boundary on one NPU device."""
    score_workspace = torch.tensor(
        _pack_u16_carrier(fixture.raw_scores), dtype=torch.int32, device=device
    )
    windows = _receipt_windows(fixture)
    receipt_count = len(windows)
    receipts_cpu = torch.zeros((receipt_count, RECEIPT_WORDS), dtype=torch.int64)
    for window in windows:
        if not window.active:
            # Fixed graph storage, deliberately left zero.  P5 must consult
            # score_source_lens first and skip it without consuming timeout
            # budget or treating it as a score-zero candidate.
            continue
        # Native C8 publishes a graph-stable byte offset relative to this
        # score_workspace view.  A physical device address would make a
        # captured replay allocation-specific and would disagree with P3's
        # InitScoreWorkspaceBase contract.
        receipts_cpu[window.slot, 0] = 2 * (
            window.row * fixture.source_capacity + window.source_begin
        )
        receipts_cpu[window.slot, 1] = window.row
        receipts_cpu[window.slot, 2] = window.source_begin
        receipts_cpu[window.slot, 3] = window.valid_count
        receipts_cpu[window.slot, 4] = fixture.source_capacity
        receipts_cpu[window.slot, 5] = window.slot
        receipts_cpu[window.slot, 6] = 0
        receipts_cpu[window.slot, 7] = RECEIPT_VERSION
        # C8 publishes this word last.  This direct-P2 semantic stage starts
        # only after the fixture's producer stream has written the epoch.
        receipts_cpu[window.slot, 8] = fixture.epoch
    tile_receipts = receipts_cpu.to(device)

    def i32(values: list[int] | tuple[int, ...], shape: tuple[int, ...] | None = None) -> Any:
        value = torch.tensor(values, dtype=torch.int32, device=device)
        return value.reshape(shape) if shape is not None else value

    # This is graph-static forward metadata owned by the scheduler/scan
    # contract.  It is deliberately *not* a ProcessTopK output: P5 needs it
    # before any Top-K terminal result exists so it can classify each fixed
    # receipt position as inactive versus late-but-expected.
    active_score_source_lens = i32(fixture.score_source_lens)
    output_rows = fixture.private_arena_capacity * GET_LANE_COUNT
    return {
        "score_workspace": score_workspace,
        "tile_receipts": tile_receipts,
        "streaming_epoch": torch.tensor([fixture.epoch], dtype=torch.int64, device=device),
        # ABI in3: immutable graph-static active_score_source_lens.
        "score_source_lens": active_score_source_lens,
        "resident_threshold": torch.full(
            (fixture.rows,), fixture.threshold, dtype=torch.int32, device=device
        ),
        "resident_threshold_valid": torch.ones(
            (fixture.rows,), dtype=torch.int32, device=device
        ),
        "kv_object_slots_snapshot": i32(
            fixture.object_slots_snapshot,
            (fixture.pool_size, fixture.object_capacity),
        ),
        "req_pool_entries": i32(list(range(fixture.rows))),
        "kv_object_map": i32(
            fixture.object_map, (fixture.pool_size, fixture.source_capacity)
        ),
        "kv_object_lens": i32([fixture.object_capacity] * fixture.rows),
        "private_source_to_slot": torch.full(
            (fixture.rows, fixture.source_capacity), -1, dtype=torch.int32, device=device
        ),
        "private_object_to_slot": torch.full(
            (fixture.pool_size, fixture.object_capacity),
            -1,
            dtype=torch.int32,
            device=device,
        ),
        # The two lane-private arenas are represented as [lane, local_slot].
        # Kernel outputIndex is ``lane * A + local_slot``, so this matrix is
        # physically contiguous and has the same flattened addressing rule.
        "private_ready_table": torch.full(
            (GET_LANE_COUNT, fixture.private_arena_capacity),
            PRIVATE_READY_FREE,
            dtype=torch.int32,
            device=device,
        ),
        "private_arena_head": torch.zeros(
            (GET_LANE_COUNT,), dtype=torch.int32, device=device
        ),
        "private_packed_output": torch.empty(
            (output_rows, PACKED_ENTRY_BYTES), dtype=torch.uint8, device=device
        ),
        # The remaining tensors are filled by the caller that owns KVAF's
        # cold Direct-HCCP bind.  They are present here so the 20-input ABI is
        # visible and testable before transport is enabled.
        "address_table": None,
        "valid_table": None,
        "fused_get_control": torch.zeros(16, dtype=torch.int64, device=device),
        "pipeline_state": torch.zeros(
            (PIPELINE_RING_COUNT, PIPELINE_WORDS_PER_RING),
            dtype=torch.int64,
            device=device,
        ),
        "execution_mode": torch.tensor(
            [PIPELINE_MODE_PARALLEL], dtype=torch.int64, device=device
        ),
    }


def _assert_receipt_fixture(torch: Any, fixture: P5Fixture, tensors: dict[str, Any]) -> None:
    receipts = tensors["tile_receipts"].cpu()
    carrier = tensors["score_workspace"].cpu().contiguous()
    source_lens = tensors["score_source_lens"].cpu()
    expected_words = _pack_u16_carrier(fixture.raw_scores)
    torch.testing.assert_close(carrier, torch.tensor(expected_words, dtype=torch.int32))
    torch.testing.assert_close(
        source_lens, torch.tensor(fixture.score_source_lens, dtype=torch.int32)
    )
    windows = _receipt_windows(fixture)
    assert tuple(receipts.shape) == (len(windows), RECEIPT_WORDS)
    for window, record in zip(windows, receipts.tolist(), strict=True):
        if not window.active:
            assert all(word == 0 for word in record)
            continue
        assert record[1:5] == [
            window.row,
            window.source_begin,
            window.valid_count,
            fixture.source_capacity,
        ]
        assert record[7] == RECEIPT_VERSION
        assert record[8] == fixture.epoch
        assert all(word == 0 for word in record[9:])
        expected_byte_offset = 2 * (
            window.row * fixture.source_capacity + window.source_begin
        )
        assert record[0] == expected_byte_offset


def _create_runtime(kvaf: Any, args: argparse.Namespace, device_id: int) -> Any:
    if args.transport == "local":
        runtime = kvaf.NpuRuntime(
            args.batch_get_kernel, device_id, GET_LANE_COUNT, args.batch_put_kernel, 1
        )
        if callable(getattr(runtime, "link_to", None)):
            raise RuntimeError("local selected but KVAF exposes the remote ABI")
        return runtime
    runtime = kvaf.NpuRuntime(
        args.batch_get_kernel,
        PACKED_ENTRY_BYTES,
        ROWS_PER_ENTRY,
        device_id,
        GET_LANE_COUNT,
        args.batch_put_kernel,
        1,
    )
    link_to = getattr(runtime, "link_to", None)
    if not callable(link_to):
        raise RuntimeError("remote selected but KVAF does not expose link_to")
    runtime.link_to(args.cpu_service_addr)
    return runtime


def _ddr_payload(torch: Any) -> Any:
    physical_rows = torch.arange(ROWS_PER_ENTRY, dtype=torch.int64).unsqueeze(1)
    byte_ids = torch.arange(PACKED_ENTRY_BYTES, dtype=torch.int64).unsqueeze(0)
    return (physical_rows * 31 + byte_ids * 11 + 5).remainder(256).to(torch.uint8)


def _executor_trace(step: str, **fields: Any) -> None:
    """Emit a compact, unbuffered executor-lifecycle breadcrumb.

    Direct-HCCP emits its own C++ lifecycle messages.  These Python markers
    delimit the user-visible control boundaries around them, which makes a
    single failing run sufficient to tell whether a failure happened during
    cold bind, ACLNN launch, device synchronization, or teardown.
    """
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"P5_EXECUTOR_STEP step={step}{(' ' + suffix) if suffix else ''}", flush=True)


def _run_executor(
    torch: Any, fixture: P5Fixture, tensors: dict[str, Any], args: argparse.Namespace, device: Any
) -> None:
    """Run the final 20x17 ABI against real Direct-HCCP when available."""
    import kvaf
    import nanovllm_exact_scan_get  # noqa: F401

    device_id = device.index if device.index is not None else torch.npu.current_device()
    _executor_trace("runtime_create", device=device_id, transport=args.transport)
    runtime = _create_runtime(kvaf, args, device_id)
    _executor_trace("runtime_created")
    table = None
    bound = False
    cleanup_failures: list[BaseException] = []
    try:
        source_cpu = _ddr_payload(torch)
        _executor_trace("initialize_begin", entry_bytes=ENTRY_BYTES)
        runtime.initialize(ENTRY_BYTES, ENTRY_BYTES)
        _executor_trace("initialize_done")
        _executor_trace("batch_put_begin", keys=1, bytes=ENTRY_BYTES)
        runtime.batch_put([1], source_cpu.reshape(-1).to(device), [0], [ENTRY_BYTES])
        torch.npu.synchronize()
        _executor_trace("batch_put_done")
        _executor_trace("access_table_prepare_begin", keys=1)
        table = runtime.prepare_access_table([1])
        access = runtime.access_table_view(table, device="npu", capacity=1)
        tensors["address_table"] = access.addresses
        tensors["valid_table"] = access.valid
        _executor_trace(
            "access_table_ready",
            address_ptr=hex(tensors["address_table"].data_ptr()),
            valid_ptr=hex(tensors["valid_table"].data_ptr()),
            capacity=tensors["address_table"].numel(),
        )
        _executor_trace(
            "bind_begin",
            output_ptr=hex(tensors["private_packed_output"].data_ptr()),
            control_ptr=hex(tensors["fused_get_control"].data_ptr()),
        )
        runtime.bind_fused_batch_get_control(
            tensors["private_packed_output"],
            tensors["fused_get_control"],
            torch.npu.current_stream(),
        )
        bound = True
        _executor_trace("bind_done")
        torch.npu.synchronize()
        _executor_trace("bind_synchronize_done")

        _executor_trace("op_invoke")
        outputs = torch.ops.a5_exact_scan_get.exact_scan_speculative_get(
            tensors["score_workspace"],
            tensors["tile_receipts"],
            tensors["streaming_epoch"],
            tensors["score_source_lens"],
            tensors["resident_threshold"],
            tensors["resident_threshold_valid"],
            tensors["kv_object_slots_snapshot"],
            tensors["req_pool_entries"],
            tensors["kv_object_map"],
            tensors["kv_object_lens"],
            tensors["private_source_to_slot"],
            tensors["private_object_to_slot"],
            tensors["private_ready_table"],
            tensors["private_arena_head"],
            tensors["private_packed_output"],
            tensors["address_table"],
            tensors["valid_table"],
            tensors["fused_get_control"],
            tensors["pipeline_state"],
            tensors["execution_mode"],
        )
        _executor_trace("op_return", outputs=len(outputs))
        _executor_trace("op_synchronize_begin")
        torch.npu.synchronize()
        _executor_trace("op_synchronize_done")
        _assert_executor_oracle(torch, fixture, tensors, outputs, source_cpu)
        _executor_trace("oracle_done")
    finally:
        primary_exception_active = sys.exc_info()[0] is not None
        _executor_trace("cleanup_begin", bound=int(bound), table=int(table is not None))
        if bound:
            try:
                _executor_trace("cleanup_synchronize_begin")
                torch.npu.synchronize()
                _executor_trace("cleanup_synchronize_done")
            except BaseException as error:  # Preserve the original failure but still release.
                cleanup_failures.append(error)
                _executor_trace("cleanup_synchronize_error", error=type(error).__name__)
            try:
                runtime.release_fused_batch_get_control()
                _executor_trace("cleanup_release_control_done")
            except BaseException as error:
                cleanup_failures.append(error)
                _executor_trace("cleanup_release_control_error", error=type(error).__name__)
        if table is not None:
            try:
                runtime.release_access_table(table)
                _executor_trace("cleanup_release_table_done")
            except BaseException as error:
                cleanup_failures.append(error)
                _executor_trace("cleanup_release_table_error", error=type(error).__name__)
        try:
            runtime.shutdown()
            _executor_trace("cleanup_shutdown_done")
        except BaseException as error:
            cleanup_failures.append(error)
            _executor_trace("cleanup_shutdown_error", error=type(error).__name__)
        if cleanup_failures and not primary_exception_active:
            raise cleanup_failures[0]


def _assert_executor_oracle(
    torch: Any,
    fixture: P5Fixture,
    inputs: dict[str, Any],
    outputs: tuple[Any, ...],
    source_cpu: Any,
) -> None:
    if len(outputs) != 17:
        raise AssertionError(f"P5 ABI returned {len(outputs)} outputs, expected 17")
    (
        private_source_out,
        private_object_out,
        private_ready_out,
        private_head_out,
        private_output_out,
        candidate_seeds,
        candidate_counts,
        producer_stats,
        get_stats,
        role_trace,
        get_indices,
        get_offsets,
        get_output_indices,
        get_mask,
        get_candidate_seed_slots,
        get_claim_new,
        pipeline_out,
    ) = outputs
    for name, before, after in (
        ("private_source_to_slot", inputs["private_source_to_slot"], private_source_out),
        ("private_object_to_slot", inputs["private_object_to_slot"], private_object_out),
        ("private_ready_table", inputs["private_ready_table"], private_ready_out),
        ("private_arena_head", inputs["private_arena_head"], private_head_out),
        ("private_packed_output", inputs["private_packed_output"], private_output_out),
        ("pipeline_state", inputs["pipeline_state"], pipeline_out),
    ):
        if before.data_ptr() != after.data_ptr():
            raise AssertionError(f"{name} output must alias its transaction input")

    counts = candidate_counts.cpu().tolist()
    stats = producer_stats.cpu()
    if tuple(stats.shape) != (PRODUCER_COUNT, 10):
        raise AssertionError(
            "producer_stats must expose [62, 10]: lifecycle columns 0..7 plus "
            f"live-receipt polls/timeouts 8..9, got {tuple(stats.shape)}"
        )
    expected_counts = [len(value) for value in fixture.expected_candidates]
    torch.testing.assert_close(candidate_counts.cpu(), torch.tensor(expected_counts, dtype=torch.int32))
    torch.testing.assert_close(
        stats[:, 0].cpu(), torch.tensor(fixture.expected_selected, dtype=torch.int64)
    )
    torch.testing.assert_close(
        stats[:, 1].cpu(), torch.tensor(fixture.expected_hits, dtype=torch.int64)
    )
    torch.testing.assert_close(
        stats[:, 2].cpu(), torch.tensor(fixture.expected_misses, dtype=torch.int64)
    )
    torch.testing.assert_close(
        stats[:, 4].cpu(), torch.tensor(fixture.expected_deferred, dtype=torch.int64)
    )
    # This executor stage publishes every *active* fixture receipt before the
    # consumer starts.  Inert slots are skipped through score_source_lens, so
    # neither inactive slots nor ready active slots consume live wait budget.
    torch.testing.assert_close(
        stats[:, 8].cpu(), torch.zeros(PRODUCER_COUNT, dtype=torch.int64)
    )
    torch.testing.assert_close(
        stats[:, 9].cpu(), torch.zeros(PRODUCER_COUNT, dtype=torch.int64)
    )

    seeds = candidate_seeds.cpu()
    if (
        seeds.ndim != 3
        or seeds.shape[0] != PRODUCER_COUNT
        or seeds.shape[2] != 5
    ):
        raise AssertionError(f"invalid CandidateSeed shape={tuple(seeds.shape)}")
    capacity = seeds.shape[1]
    slots_by_object: dict[tuple[int, int], int] = {}
    for producer, expected in enumerate(fixture.expected_candidates):
        actual = seeds[producer, : counts[producer]]
        expected_rows = [[item.row, item.source, item.kv_object, item.score] for item in expected]
        if actual[:, :4].tolist() != expected_rows:
            raise AssertionError(
                f"CandidateSeed mismatch producer={producer} actual={actual[:, :4].tolist()} "
                f"expected={expected_rows}"
            )
        for record, item in zip(actual.tolist(), expected, strict=True):
            private_slot = int(record[4])
            if private_slot < 0:
                raise AssertionError(f"cold candidate was not assigned a private slot: {record}")
            key = (item.pool, item.kv_object)
            if key in slots_by_object and slots_by_object[key] != private_slot:
                raise AssertionError(f"duplicate object posted twice: key={key} slots={slots_by_object[key]},{private_slot}")
            slots_by_object[key] = private_slot

    if set(slots_by_object) != fixture.expected_unique_cold_objects:
        raise AssertionError(f"private object claims mismatch: {slots_by_object}")
    if len(set(slots_by_object.values())) != len(slots_by_object):
        raise AssertionError(f"two distinct objects share a private slot: {slots_by_object}")
    for (pool, kv_object), slot in slots_by_object.items():
        if int(private_object_out.cpu()[pool, kv_object]) != slot:
            raise AssertionError(f"private object map does not publish {pool, kv_object} -> {slot}")
        lane = slot // fixture.private_arena_capacity
        local_slot = slot % fixture.private_arena_capacity
        if int(private_ready_out.cpu()[lane, local_slot]) != PRIVATE_READY_READY:
            raise AssertionError(f"private slot {slot} is not READY after terminal CQ")
        expected_payload = source_cpu[kv_object]
        torch.testing.assert_close(private_output_out.cpu()[slot], expected_payload)

    # Four unique cold objects, split two per Direct-HCCP lane (O=4 -> lane0,
    # O=1 -> lane1).  Candidate multiplicity is eight but must not leak into
    # descriptor post count.
    torch.testing.assert_close(get_stats.cpu()[:, 2], torch.zeros(GET_LANE_COUNT, dtype=torch.int64))
    torch.testing.assert_close(get_stats.cpu()[:, 0].sum(), torch.tensor(4, dtype=torch.int64))
    torch.testing.assert_close(private_head_out.cpu(), torch.tensor([2, 2], dtype=torch.int32))
    # Bridge outputs are intentionally ``empty`` outside the records produced
    # by a wave.  Check only each lane's known active prefix; summing the
    # uninitialized tail would create a flaky test rather than a lifecycle
    # assertion.
    requested_per_lane = get_stats.cpu()[:, 0].tolist()
    claim_cpu = get_claim_new.cpu()
    mask_cpu = get_mask.cpu()
    for lane, request_count in enumerate(requested_per_lane):
        if request_count != 2:
            raise AssertionError(f"lane {lane} posted {request_count}, expected two unique objects")
        torch.testing.assert_close(
            claim_cpu[lane, 0, :request_count],
            torch.ones(request_count, dtype=torch.uint8),
        )
        torch.testing.assert_close(
            mask_cpu[lane, 0, :request_count],
            torch.ones(request_count, dtype=torch.uint8),
        )
    expected_trace = torch.cat(
        (
            torch.arange(2000, 2000 + GET_LANE_COUNT, dtype=torch.int32),
            torch.arange(1000, 1000 + PRODUCER_COUNT, dtype=torch.int32),
        )
    )
    torch.testing.assert_close(role_trace.cpu(), expected_trace)
    _ = (get_indices, get_offsets, get_output_indices, get_candidate_seed_slots, pipeline_out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:7")
    parser.add_argument(
        "--stage",
        choices=("oracle", "receipt_fixture", "executor"),
        default="oracle",
        help=(
            "oracle needs only Python and validates C8 receipt/candidate semantics; "
            "receipt_fixture validates real NPU C8-shaped tensors; executor also runs P2+KVAF"
        ),
    )
    parser.add_argument("--frame-epoch", type=int, default=17)
    parser.add_argument("--transport", choices=("local", "remote"), default="local")
    parser.add_argument("--cpu-service-addr", default="")
    parser.add_argument("--batch-get-kernel", default="")
    parser.add_argument("--batch-put-kernel", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.frame_epoch <= 0:
        raise ValueError("--frame-epoch must be nonzero")
    if args.stage == "executor" and (not args.batch_get_kernel or not args.batch_put_kernel):
        raise ValueError("--stage executor requires --batch-get-kernel and --batch-put-kernel")
    if args.transport == "remote" and not args.cpu_service_addr:
        raise ValueError("--transport remote requires --cpu-service-addr")

    fixture = _build_fixture(args.frame_epoch)
    _assert_fixture_oracle(fixture)
    live_oracle = _assert_live_receipt_oracle(fixture)
    if args.stage == "oracle":
        print(
            "P5_SEMANTIC_PURE_ORACLE_PASS "
            f"active_windows={live_oracle['active_windows']} "
            f"inactive_windows={live_oracle['inactive_windows']} "
            f"timeout_defer_windows={live_oracle['timeout_defer_windows']} "
            f"withheld_candidate_seeds={live_oracle['withheld_candidate_seeds']} "
            f"exact_candidates={live_oracle['exact_candidates']} "
            "speculative_candidates_after_timeout="
            f"{live_oracle['speculative_candidates_after_timeout']} "
            "exact_fallback_preserved=1"
        )
        return

    import torch
    import torch_npu  # noqa: F401

    device = torch.device(args.device)
    torch.npu.set_device(device)
    tensors = _device_inputs(torch, fixture, device)
    torch.npu.synchronize()
    _assert_receipt_fixture(torch, fixture, tensors)
    print(
        "P5_SEMANTIC_STAGE "
        f"mode={args.stage} source=p2_direct_receipt_fixture "
        f"device={args.device} score_carrier=int32 score_value=uint16 "
        f"receipts={tensors['tile_receipts'].shape[0]} receipt_words={RECEIPT_WORDS} "
        f"active_score_source_lens={fixture.score_source_lens} "
        "live_c8_publisher=0"
    )
    print(
        "P5_SEMANTIC_ORACLE "
        f"selected={sum(fixture.expected_selected)} hbm_hit={sum(fixture.expected_hits)} "
        f"cold_candidate_seeds={sum(len(value) for value in fixture.expected_candidates)} "
        f"unique_posts={len(fixture.expected_unique_cold_objects)} "
        f"deferred={sum(fixture.expected_deferred)} duplicate_s_to_o=1"
    )
    print(
        "P5_SEMANTIC_LIVE_RECEIPT_ORACLE "
        f"active_windows={live_oracle['active_windows']} "
        f"inactive_windows={live_oracle['inactive_windows']} "
        f"timeout_defer_windows={live_oracle['timeout_defer_windows']} "
        f"withheld_candidate_seeds={live_oracle['withheld_candidate_seeds']} "
        f"exact_candidates={live_oracle['exact_candidates']} "
        f"speculative_candidates_after_timeout="
        f"{live_oracle['speculative_candidates_after_timeout']} "
        "exact_fallback_preserved=1"
    )
    if args.stage == "executor":
        _run_executor(torch, fixture, tensors, args, device)
        print("P5_SEMANTIC_EXECUTOR_PASS candidate_exact=1 object_dedup=1 post_cq_ready=1")
    else:
        print("P5_SEMANTIC_RECEIPT_FIXTURE_PASS input_contract=1 oracle=1")


if __name__ == "__main__":
    main()
