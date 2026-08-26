#!/usr/bin/env python3
"""NPU validation for transaction-local exact C8 Top-K reconciliation.

This target probe deliberately sets ``cmp_ratio=1`` and
``kv_object_domain=RAW_TOKEN``. It therefore tests the legacy raw-token
equivalence case with a non-identity score-source ``S -> O`` map, where the
active C8 score domain S (4096) differs from the raw-token object directory O
(8192). The CPU reference mirrors the device op and compares all twelve
outputs; no persistent cache directory is allowed to change. A compressed
object domain needs an end-to-end QSFA representation proof, not a guessed
token conversion in this test.
"""

from __future__ import annotations

import argparse
from typing import NamedTuple

import torch
import torch_npu  # type: ignore  # noqa: F401

import nanovllm_exact_reconcile_private_view  # noqa: F401


TOPK = 2048
VIEW_RESIDENT_HIT = 0
VIEW_PRIVATE_READY = 1
VIEW_FALLBACK_REQUIRED = 2
PRIVATE_READY_AFTER_CQ = 2


class Expected(NamedTuple):
    source_ids: torch.Tensor
    destination_slots: torch.Tensor
    miss_counts: torch.Tensor
    placement_delta: torch.Tensor
    private_view_descriptors: torch.Tensor
    ready_private_source_ids: torch.Tensor
    ready_private_slots: torch.Tensor
    ready_private_destination_slots: torch.Tensor
    ready_private_counts: torch.Tensor
    fallback_logical_token_ids: torch.Tensor
    fallback_destination_slots: torch.Tensor
    fallback_counts: torch.Tensor


def _mapped_logical(pool: int, source: int, logical_capacity: int) -> int:
    # 3 is coprime to 8192, so the active prefix maps injectively. The map is
    # intentionally non-identity and pool-specific while still remaining in L.
    return (source * 3 + 17 + pool * 97) % logical_capacity


def _install_row(
    snapshot: torch.Tensor, logical_map: torch.Tensor, pool: int,
    hit_count: int, cache_tokens: int,
) -> None:
    topk_logicals = [int(logical_map[pool, source]) for source in range(TOPK)]
    snapshot[pool, topk_logicals[:hit_count]] = torch.arange(
        hit_count, dtype=torch.int32
    )
    selected = set(topk_logicals)
    victims = [
        logical for logical in range(snapshot.size(1)) if logical not in selected
    ][: TOPK - hit_count]
    snapshot[pool, victims] = torch.arange(
        hit_count, TOPK, dtype=torch.int32
    )
    if cache_tokens < TOPK:
        raise AssertionError("test assumes LIDU cache budget >= TOPK")


def build_case() -> tuple[torch.Tensor, ...]:
    """Build twelve actual-C8-shaped rows plus intentional no-cache rows.

    A5 C8 requires N2 == 1.  Its often cited C=4 is an internal S1 score tile,
    so the compatible BSND sparse output is [B, S1, 1, K], not [B, 2, K].
    This test gives each flattened row a distinct pool entry.  That is a
    transaction-isolated planner test: two C=4 rows belonging to the same
    request/pool need an explicit deterministic merge before any placement
    delta is published, and are deliberately not claimed equivalent here.
    """

    batch, s1, n2 = 3, 4, 1
    row_count = batch * s1 * n2
    pool_size = row_count
    logical_capacity, source_capacity = 8192, 4096
    queues, rows_per_queue = 3, 8192
    topk = torch.arange(TOPK, dtype=torch.int32).repeat(row_count).view(
        batch, s1, n2, TOPK
    )
    req_pool_entries = torch.arange(row_count, dtype=torch.int32)
    cache_tokens = torch.tensor(
        [3072, 4096, 0, 3072] * batch, dtype=torch.int32
    )
    candidate_lens = torch.tensor(
        [logical_capacity] * row_count, dtype=torch.int32
    )
    score_source_lens = torch.tensor(
        [source_capacity, source_capacity, 0, source_capacity] * batch,
        dtype=torch.int32,
    )
    cache_slots_snapshot = torch.full(
        (pool_size, logical_capacity), -1, dtype=torch.int32
    )
    logical_token_map = torch.empty(
        (pool_size, source_capacity), dtype=torch.int32
    )
    for pool in range(pool_size):
        logical_token_map[pool] = torch.tensor(
            [_mapped_logical(pool, source, logical_capacity)
             for source in range(source_capacity)],
            dtype=torch.int32,
        )

    hit0, hit1 = 1536, 1200
    hit_counts = [hit0, hit1, 0, hit0] * batch
    for row_index, (pool, budget, hit_count) in enumerate(
        zip(req_pool_entries.tolist(), cache_tokens.tolist(), hit_counts, strict=True)
    ):
        if budget != 0:
            _install_row(cache_slots_snapshot, logical_token_map, pool, hit_count, budget)

    private_source_to_slot = torch.full(
        (row_count, source_capacity), -1, dtype=torch.int32
    )
    private_ready = torch.zeros((queues, rows_per_queue), dtype=torch.int32)
    for row_index, hit_count in enumerate(hit_counts):
        if cache_tokens[row_index].item() == 0:
            continue
        for source in range(hit_count, TOPK):
            if source % 3 == 0:
                private_source_to_slot[row_index, source] = row_index * TOPK + source
                if source % 2 == 0:
                    private_ready.reshape(-1)[row_index * TOPK + source] = PRIVATE_READY_AFTER_CQ

    return (
        topk,
        req_pool_entries,
        cache_slots_snapshot,
        cache_tokens,
        candidate_lens,
        score_source_lens,
        logical_token_map,
        private_source_to_slot,
        private_ready,
    )


def reference(
    exact_topk_indices: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_snapshot: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    score_source_lens: torch.Tensor,
    logical_token_map: torch.Tensor,
    private_source_to_slot: torch.Tensor,
    private_ready: torch.Tensor,
) -> Expected:
    """CPU mirror of source-S -> logical-L reconcile and private split."""

    row_count = exact_topk_indices.numel() // TOPK
    raw_rows = exact_topk_indices.reshape(row_count, TOPK)
    source_rows = source_ids = torch.full_like(exact_topk_indices, -1).reshape(
        row_count, TOPK
    )
    destination_rows = destination_slots = torch.full_like(
        exact_topk_indices, -1
    ).reshape(row_count, TOPK)
    miss_counts = torch.zeros((row_count,), dtype=torch.int32)
    placement_delta = torch.full((row_count, 3, TOPK), -1, dtype=torch.int32)
    view = torch.full((row_count, 4, TOPK), -1, dtype=torch.int32)
    ready_source = torch.full_like(exact_topk_indices, -1).reshape(row_count, TOPK)
    ready_slot = torch.full_like(exact_topk_indices, -1).reshape(row_count, TOPK)
    ready_destination = torch.full_like(exact_topk_indices, -1).reshape(row_count, TOPK)
    ready_counts = torch.zeros((row_count,), dtype=torch.int32)
    fallback_logical = torch.full_like(exact_topk_indices, -1).reshape(row_count, TOPK)
    fallback_destination = torch.full_like(exact_topk_indices, -1).reshape(row_count, TOPK)
    fallback_counts = torch.zeros((row_count,), dtype=torch.int32)
    private_capacity = private_ready.numel()

    for b in range(row_count):
        budget = int(cache_tokens[b])
        if budget == 0:
            continue
        logical_len = int(candidate_lens[b])
        source_len = int(score_source_lens[b])
        pool = int(req_pool_entries[b])
        raw_sources = [int(value) for value in raw_rows[b]]
        logicals = [int(logical_token_map[pool, source]) for source in raw_sources]
        if not all(0 <= source < source_len for source in raw_sources):
            raise AssertionError("fixture contains an inactive score source")
        if not all(0 <= logical < logical_len for logical in logicals):
            raise AssertionError("fixture contains an invalid logical map")
        if len(set(logicals)) != TOPK:
            raise AssertionError("fixture has duplicate selected logical tokens")
        snapshot = cache_slots_snapshot[pool]

        misses: list[tuple[int, int]] = []
        hits: list[tuple[int, int]] = []
        protected: set[int] = set()
        for raw_source, logical in zip(raw_sources, logicals):
            slot = int(snapshot[logical])
            if 0 <= slot < budget:
                hits.append((logical, slot))
                protected.add(slot)
            else:
                misses.append((raw_source, logical))

        victims: list[tuple[int, int]] = []
        for victim_logical in range(logical_len):
            if len(victims) == len(misses):
                break
            slot = int(snapshot[victim_logical])
            if slot < 0 or slot >= budget or slot in protected:
                continue
            protected.add(slot)
            victims.append((victim_logical, slot))
        if len(victims) != len(misses):
            raise AssertionError("fixture has insufficient LIDU victims")

        ordered_logicals = [logical for _, logical in misses] + [
            logical for logical, _ in hits
        ]
        ordered_slots = [slot for _, slot in victims] + [
            slot for _, slot in hits
        ]
        source_rows[b] = torch.tensor(ordered_logicals, dtype=torch.int32)
        destination_rows[b] = torch.tensor(ordered_slots, dtype=torch.int32)
        miss_counts[b] = len(misses)
        for index, ((victim, slot), (_, incoming)) in enumerate(
            zip(victims, misses)
        ):
            placement_delta[b, 0, index] = victim
            placement_delta[b, 1, index] = incoming
            placement_delta[b, 2, index] = slot

        ready_count = 0
        fallback_count = 0
        for index, logical in enumerate(ordered_logicals):
            destination = ordered_slots[index]
            view[b, 0, index] = logical
            view[b, 1, index] = destination
            if index >= len(misses):
                view[b, 2, index] = VIEW_RESIDENT_HIT
                continue
            raw_source = misses[index][0]
            private_slot = int(private_source_to_slot[b, raw_source])
            if 0 <= private_slot < private_capacity and int(
                private_ready.reshape(-1)[private_slot]
            ) == PRIVATE_READY_AFTER_CQ:
                view[b, 2, index] = VIEW_PRIVATE_READY
                view[b, 3, index] = private_slot
                ready_source[b, ready_count] = logical
                ready_slot[b, ready_count] = private_slot
                ready_destination[b, ready_count] = destination
                ready_count += 1
            else:
                view[b, 2, index] = VIEW_FALLBACK_REQUIRED
                fallback_logical[b, fallback_count] = logical
                fallback_destination[b, fallback_count] = destination
                fallback_count += 1
        ready_counts[b] = ready_count
        fallback_counts[b] = fallback_count

    return Expected(
        source_ids.reshape_as(exact_topk_indices),
        destination_slots.reshape_as(exact_topk_indices),
        miss_counts,
        placement_delta,
        view,
        ready_source.reshape_as(exact_topk_indices),
        ready_slot.reshape_as(exact_topk_indices),
        ready_destination.reshape_as(exact_topk_indices),
        ready_counts,
        fallback_logical.reshape_as(exact_topk_indices),
        fallback_destination.reshape_as(exact_topk_indices),
        fallback_counts,
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

    inputs_cpu = build_case()
    expected = reference(*inputs_cpu)
    snapshot_before = inputs_cpu[2].clone()
    inputs_npu = tuple(tensor.to(device).contiguous() for tensor in inputs_cpu)
    outputs = torch.ops.a5_exact_reconcile_private_view.exact_reconcile_private_view(
        *inputs_npu, 1, 0
    )
    torch.npu.synchronize()
    if len(outputs) != len(expected):
        raise AssertionError(f"expected {len(expected)} outputs, got {len(outputs)}")
    for index, (actual, wanted) in enumerate(zip(outputs, expected)):
        if not torch.equal(actual.cpu(), wanted):
            raise AssertionError(
                f"output {index} differs from the logical-domain LIDU reference"
            )
    if not torch.equal(inputs_npu[2].cpu(), snapshot_before):
        raise AssertionError("cache_slots_snapshot changed; reconcile crossed transaction boundary")

    replay = torch.ops.a5_exact_reconcile_private_view.exact_reconcile_private_view(
        *inputs_npu, 1, 0
    )
    torch.npu.synchronize()
    if not all(torch.equal(left.cpu(), right.cpu()) for left, right in zip(outputs, replay)):
        raise AssertionError("transaction-local reconcile is not replay-stable")

    miss = expected.miss_counts.tolist()
    ready = expected.ready_private_counts.tolist()
    fallback = expected.fallback_counts.tolist()
    if any(r + f != m for m, r, f in zip(miss, ready, fallback)):
        raise AssertionError("ready/fallback split does not cover exact miss prefix")
    print(
        "A5_EXACT_RECONCILE_PRIVATE_VIEW_PASS "
        "logical_capacity=8192 score_source_capacity=4096 "
        f"rows={len(miss)} topk_shape={tuple(expected.source_ids.shape)} "
        f"misses={miss} ready={ready} fallback={fallback} "
        "logical_map_nonidentity=1 snapshot_unchanged=1 "
        "kv_object_domain=RAW_TOKEN cmp_ratio=1 "
        "c8_shape=BSND_Bx4x1x2048 distinct_pool_rows=1 "
        "placement_delta_plan_only=1 qsfa_called=0 guaranteed_get_called=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
