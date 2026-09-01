#!/usr/bin/env python3
"""NPU mathematical gate for the isolated P7.0 receipt candidate oracle.

The fixture models the *published* C8 boundary, rather than importing or
launching C8.  It creates a packed uint16 score workspace (carried by int32),
valid and intentionally terminal receipt records, an immutable resident
snapshot, and an explicit score-source-to-KV-object map.  The CPU oracle
replays the P7.0 predicate exactly:

``score >= resident threshold && object is valid && object is not resident``.

It therefore verifies the part of the future pipeline that is mathematically
allowed to be speculative.  It does not claim C8 overlap, HCCP posting, CQ
handling, private-slot allocation, or final Top-K correctness; those belong to
P7.1 and the subsequent exact reconcile / Guaranteed Recall stages.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu  # type: ignore  # noqa: F401

# Importing this package binds one caller-selected P7.0 OPP and registers the
# standalone torch.library namespace.  It has no C8 or KVAF import path.
import nanovllm_score_receipt_executor  # noqa: F401


TILE_WIDTH = 128
RECEIPT_WORDS = 16
RECEIPT_VERSION = 1
RECEIPT_IDENTITY_WORDS = 4
STATUS_OK = 1
STATUS_INACTIVE = 2
STATUS_EPOCH_MISMATCH = -1
STATUS_RECEIPT_MALFORMED = -2
STATUS_ROW_METADATA_INVALID = -3


@dataclass(frozen=True)
class Fixture:
    rows: int
    source_capacity: int
    pool_size: int
    object_capacity: int
    epoch: int
    score_workspace: torch.Tensor
    tile_receipts: torch.Tensor
    score_source_lens: torch.Tensor
    resident_threshold: torch.Tensor
    resident_threshold_valid: torch.Tensor
    kv_object_slots_snapshot: torch.Tensor
    req_pool_entries: torch.Tensor
    kv_object_map: torch.Tensor
    kv_object_lens: torch.Tensor
    scores_u16: list[list[int]]


def _pack_u16_carrier(scores: list[list[int]]) -> torch.Tensor:
    """Return an int32 carrier whose byte layout is the uint16 score stream."""

    flat = [int(value) for row in scores for value in row]
    if len(flat) % 2:
        flat.append(0)
    words: list[int] = []
    for index in range(0, len(flat), 2):
        word = flat[index] | (flat[index + 1] << 16)
        # torch.int32 is signed while the kernel deliberately reinterprets the
        # same bytes as uint16.  Preserve the carrier's bits exactly.
        words.append(word if word < (1 << 31) else word - (1 << 32))
    return torch.tensor(words, dtype=torch.int32)


def _score(row: int, source: int) -> int:
    # Non-monotonic but deterministic; this catches a wrong score offset,
    # threshold direction, or S->O indexing without a large terminal dump.
    return (row * 97 + source * 29 + (source // 7) * 11 + 31) % 1024


def _fixture() -> Fixture:
    rows, source_capacity, pool_size, object_capacity = 4, 256, 4, 307
    epoch = 37
    scores = [
        [_score(row, source) for source in range(source_capacity)]
        for row in range(rows)
    ]
    source_lens = torch.tensor([256, 173, 81, 0], dtype=torch.int32)
    thresholds = torch.tensor([500, 300, 100, 0], dtype=torch.int32)
    threshold_valid = torch.tensor([1, 1, 0, 1], dtype=torch.int32)
    object_map = torch.empty((pool_size, source_capacity), dtype=torch.int32)
    for pool in range(pool_size):
        object_map[pool] = torch.tensor(
            [((source * 7) + (pool * 11) + 5) % object_capacity
             for source in range(source_capacity)],
            dtype=torch.int32,
        )
    snapshot = torch.full((pool_size, object_capacity), -1, dtype=torch.int32)
    # Some valid objects already resident.  This makes the expected candidate
    # list a compacted subset rather than simply all above-threshold scores.
    for pool in range(pool_size):
        for obj in range(object_capacity):
            if (obj + 3 * pool) % 11 == 0:
                snapshot[pool, obj] = obj % 64

    tiles = (source_capacity + TILE_WIDTH - 1) // TILE_WIDTH
    receipts = torch.zeros((rows * tiles, RECEIPT_WORDS), dtype=torch.int64)
    for row in range(rows):
        active = int(source_lens[row])
        for tile in range(tiles):
            receipt = row * tiles + tile
            source_begin = tile * TILE_WIDTH
            if source_begin >= active:
                # The producer leaves graph-static inactive storage zero.  The
                # oracle must identify it from score_source_lens first.
                continue
            valid = min(TILE_WIDTH, active - source_begin)
            receipts[receipt, 0] = 2 * (row * source_capacity + source_begin)
            receipts[receipt, 1] = row
            receipts[receipt, 2] = source_begin
            receipts[receipt, 3] = valid
            receipts[receipt, 4] = source_capacity
            receipts[receipt, 7] = RECEIPT_VERSION
            receipts[receipt, 8] = epoch

    # This row is active but terminally malformed.  It proves P7.0 never
    # turns arbitrary GM words into a speculative candidate.
    receipts[1 * tiles + 1, 7] = RECEIPT_VERSION + 1

    return Fixture(
        rows=rows,
        source_capacity=source_capacity,
        pool_size=pool_size,
        object_capacity=object_capacity,
        epoch=epoch,
        score_workspace=_pack_u16_carrier(scores),
        tile_receipts=receipts,
        score_source_lens=source_lens,
        resident_threshold=thresholds,
        resident_threshold_valid=threshold_valid,
        kv_object_slots_snapshot=snapshot,
        req_pool_entries=torch.arange(rows, dtype=torch.int32),
        kv_object_map=object_map,
        kv_object_lens=torch.full((rows,), object_capacity, dtype=torch.int32),
        scores_u16=scores,
    )


def _receipt_identity(
    fixture: Fixture, *, graph_replay_generation: int = 11,
    layer_control_index: int = 3,
) -> torch.Tensor:
    """Graph-static full receipt identity with a nonzero per-receipt generation.

    Generation is deliberately distinct across receipt slots so a cursor from
    an earlier replay cannot be adopted merely because it has the same epoch.
    """

    tiles = (fixture.source_capacity + TILE_WIDTH - 1) // TILE_WIDTH
    receipts = fixture.rows * tiles
    identity = torch.empty((receipts, RECEIPT_IDENTITY_WORDS), dtype=torch.int64)
    identity[:, 0] = fixture.epoch
    identity[:, 1] = graph_replay_generation
    identity[:, 2] = layer_control_index
    identity[:, 3] = torch.arange(1, receipts + 1, dtype=torch.int64)
    return identity


def _reference(
    fixture: Fixture, receipt_identity: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """CPU equivalent of the P7.0 device predicate and terminal statuses."""

    tiles = (fixture.source_capacity + TILE_WIDTH - 1) // TILE_WIDTH
    receipts = fixture.rows * tiles
    seeds = torch.full((receipts, TILE_WIDTH, 4), -1, dtype=torch.int32)
    counts = torch.zeros((receipts,), dtype=torch.int32)
    candidate_receipt_identity = receipt_identity.clone()
    status = torch.empty((receipts,), dtype=torch.int32)

    for receipt in range(receipts):
        row, tile = divmod(receipt, tiles)
        source_begin = tile * TILE_WIDTH
        source_len = int(fixture.score_source_lens[row])
        if row >= fixture.rows:
            status[receipt] = STATUS_RECEIPT_MALFORMED
            continue
        if source_len < 0 or source_len > fixture.source_capacity:
            status[receipt] = STATUS_ROW_METADATA_INVALID
            continue
        if source_begin >= source_len:
            status[receipt] = STATUS_INACTIVE
            continue

        record = fixture.tile_receipts[receipt]
        valid = min(TILE_WIDTH, source_len - source_begin)
        expected_offset = 2 * (row * fixture.source_capacity + source_begin)
        malformed = (
            int(record[8]) != fixture.epoch
            or int(record[7]) != RECEIPT_VERSION
            or int(record[0]) != expected_offset
            or int(record[1]) != row
            or int(record[2]) != source_begin
            or int(record[3]) != valid
            or int(record[4]) < fixture.source_capacity
        )
        if int(record[8]) != fixture.epoch:
            status[receipt] = STATUS_EPOCH_MISMATCH
            continue
        if malformed:
            status[receipt] = STATUS_RECEIPT_MALFORMED
            continue
        if int(fixture.resident_threshold_valid[row]) == 0:
            status[receipt] = STATUS_OK
            continue

        pool = int(fixture.req_pool_entries[row])
        object_len = int(fixture.kv_object_lens[row])
        if not 0 <= pool < fixture.pool_size or not 0 <= object_len <= fixture.object_capacity:
            status[receipt] = STATUS_ROW_METADATA_INVALID
            continue
        threshold = int(fixture.resident_threshold[row]) & 0xFFFF
        compact = 0
        for local in range(valid):
            source = source_begin + local
            score = fixture.scores_u16[row][source]
            if score < threshold:
                continue
            obj = int(fixture.kv_object_map[pool, source])
            if not 0 <= obj < object_len or obj >= fixture.object_capacity:
                continue
            if int(fixture.kv_object_slots_snapshot[pool, obj]) >= 0:
                continue
            seeds[receipt, compact] = torch.tensor(
                [row, source, obj, score], dtype=torch.int32
            )
            compact += 1
        counts[receipt] = compact
        status[receipt] = STATUS_OK
    return seeds, counts, candidate_receipt_identity, status


def _digest(tensor: torch.Tensor) -> str:
    """Bounded deterministic summary that keeps failure reports copyable."""

    values = tensor.detach().cpu().reshape(-1).to(torch.int64).tolist()
    digest = 0xCBF29CE484222325
    for value in values:
        digest ^= int(value) & ((1 << 64) - 1)
        digest = (digest * 0x100000001B3) & ((1 << 64) - 1)
    return f"{digest:016x}"


def _assert_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_cpu = actual.detach().cpu().contiguous()
    if torch.equal(actual_cpu, expected):
        return
    mismatches = (actual_cpu != expected).reshape(-1).nonzero(as_tuple=False)
    first = int(mismatches[0].item()) if mismatches.numel() else -1
    raise AssertionError(
        f"{name} mismatch count={int(mismatches.numel())} first_flat={first} "
        f"actual_sha16={_digest(actual_cpu)} expected_sha16={_digest(expected)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    fixture = _fixture()
    receipt_identity = _receipt_identity(fixture)
    expected = _reference(fixture, receipt_identity)
    tensors: list[Any] = [
        fixture.score_workspace,
        fixture.tile_receipts,
        receipt_identity,
        fixture.score_source_lens,
        fixture.resident_threshold,
        fixture.resident_threshold_valid,
        fixture.kv_object_slots_snapshot,
        fixture.req_pool_entries,
        fixture.kv_object_map,
        fixture.kv_object_lens,
    ]
    inputs = [tensor.to(device).contiguous() for tensor in tensors]
    print(
        "P7_ORACLE_CONFIG "
        f"device={device} rows={fixture.rows} source_capacity={fixture.source_capacity} "
        f"receipts={fixture.tile_receipts.size(0)} score_words={fixture.score_workspace.numel()} "
        "c8_called=0 kvaf_called=0 post_cq=0",
        flush=True,
    )
    outputs = torch.ops.a5_score_receipt_executor.score_receipt_candidate_executor(
        *inputs
    )
    torch.npu.synchronize()
    candidate_seeds, candidate_counts, candidate_receipt_identity, receipt_status, _owners = outputs
    for name, actual, wanted in zip(
        ("candidate_seeds", "candidate_counts", "candidate_receipt_identity", "receipt_status"),
        (candidate_seeds, candidate_counts, candidate_receipt_identity, receipt_status),
        expected,
        strict=True,
    ):
        _assert_equal(name, actual, wanted)

    counts_cpu = candidate_counts.cpu()
    status_cpu = receipt_status.cpu()
    print(
        "P7_ORACLE_RESULT "
        f"candidate_total={int(counts_cpu.sum())} "
        f"nonempty_receipts={int((counts_cpu > 0).sum())} "
        f"status_ok={int((status_cpu == STATUS_OK).sum())} "
        f"status_inactive={int((status_cpu == STATUS_INACTIVE).sum())} "
        f"status_malformed={int((status_cpu == STATUS_RECEIPT_MALFORMED).sum())} "
        f"seeds_sha16={_digest(candidate_seeds)} counts_sha16={_digest(candidate_counts)}",
        flush=True,
    )
    print(
        "P7_ORACLE_PASS "
        "candidate_seeds=1 counts=1 receipt_identity=1 status=1 "
        "c8_dependency=0 kvaf_dependency=0",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"P7_ORACLE_FAIL type={type(exc).__name__} text={exc}",
            flush=True,
        )
        raise
