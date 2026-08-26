#!/usr/bin/env python3
"""NPU validation for transaction-local exact Top-K reconciliation.

The reference below is deliberately a direct translation of the ordering and
victim-selection portion of ``a5_lidu_cache_update.cpp``.  It verifies the
new boundary separately from QSFA: the operator is allowed to plan a ready
private materialization and a guaranteed-recall fallback, but it must never
change the persistent ``cache_slots_snapshot`` input.
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


def build_case() -> tuple[torch.Tensor, ...]:
    """Build unordered request-pool rows with both ready and fallback misses."""

    batch, pool_size, source_capacity = 3, 5, 4096
    queues, rows_per_queue = 2, 2048
    topk = torch.empty((batch, 1, TOPK), dtype=torch.int32)
    topk[0, 0] = torch.arange(TOPK, dtype=torch.int32)
    topk[1, 0] = torch.arange(TOPK, dtype=torch.int32)
    topk[2, 0] = torch.arange(TOPK, dtype=torch.int32)
    req_pool_entries = torch.tensor([3, 1, 4], dtype=torch.int32)
    cache_tokens = torch.tensor([3072, 4096, 0], dtype=torch.int32)
    candidate_lens = torch.full((batch,), source_capacity, dtype=torch.int32)
    cache_slots_snapshot = torch.full(
        (pool_size, source_capacity), -1, dtype=torch.int32
    )

    # Row 0: top-k has 1536 hits followed by 512 misses.  The first allowed
    # victims are sources 2048..2559, so first-unprotected-victim selection is
    # unambiguous and exercises the legacy scan order.
    row0, hit0 = int(req_pool_entries[0]), 1536
    cache_slots_snapshot[row0, :hit0] = torch.arange(hit0, dtype=torch.int32)
    cache_slots_snapshot[row0, TOPK : TOPK + (TOPK - hit0)] = torch.arange(
        hit0, TOPK, dtype=torch.int32
    )

    # Row 1 has a different hit/miss split and a different pool row, proving
    # the cache snapshot lookup uses req_pool_entries rather than B identity.
    row1, hit1 = int(req_pool_entries[1]), 1200
    cache_slots_snapshot[row1, :hit1] = torch.arange(hit1, dtype=torch.int32)
    cache_slots_snapshot[row1, TOPK : TOPK + (TOPK - hit1)] = torch.arange(
        hit1, TOPK, dtype=torch.int32
    )

    logical_token_map = torch.arange(
        pool_size * source_capacity, dtype=torch.int32
    ).reshape(pool_size, source_capacity)
    # Make logical tokens explicitly pool-row scoped so a wrong map base is
    # caught by the fallback plan assertion.
    logical_token_map.mul_(7).add_(13)

    private_source_to_slot = torch.full(
        (batch, source_capacity), -1, dtype=torch.int32
    )
    private_ready = torch.zeros((queues, rows_per_queue), dtype=torch.int32)
    # Every third miss has a speculative destination.  Even sources become
    # terminal-ready; odd ones deliberately force guaranteed recall.
    for batch_index, hit_count in enumerate((hit0, hit1)):
        for source in range(hit_count, TOPK):
            if source % 3 == 0:
                private_source_to_slot[batch_index, source] = source
                if source % 2 == 0:
                    private_ready.reshape(-1)[source] = PRIVATE_READY_AFTER_CQ

    return (
        topk,
        req_pool_entries,
        cache_slots_snapshot,
        cache_tokens,
        candidate_lens,
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
    logical_token_map: torch.Tensor,
    private_source_to_slot: torch.Tensor,
    private_ready: torch.Tensor,
) -> Expected:
    """CPU mirror of LIDU ordering plus transaction-local plan splitting."""

    batch = exact_topk_indices.size(0)
    source_ids = torch.full_like(exact_topk_indices, -1)
    destination_slots = torch.full_like(exact_topk_indices, -1)
    miss_counts = torch.zeros((batch,), dtype=torch.int32)
    placement_delta = torch.full((batch, 3, TOPK), -1, dtype=torch.int32)
    view = torch.full((batch, 4, TOPK), -1, dtype=torch.int32)
    ready_source = torch.full_like(exact_topk_indices, -1)
    ready_slot = torch.full_like(exact_topk_indices, -1)
    ready_destination = torch.full_like(exact_topk_indices, -1)
    ready_counts = torch.zeros((batch,), dtype=torch.int32)
    fallback_logical = torch.full_like(exact_topk_indices, -1)
    fallback_destination = torch.full_like(exact_topk_indices, -1)
    fallback_counts = torch.zeros((batch,), dtype=torch.int32)
    private_capacity = private_ready.numel()

    for b in range(batch):
        budget = int(cache_tokens[b])
        if budget == 0:
            continue
        candidate_len = int(candidate_lens[b])
        pool = int(req_pool_entries[b])
        topk = [int(value) for value in exact_topk_indices[b, 0]]
        snapshot = cache_slots_snapshot[pool]

        misses: list[int] = []
        hits: list[tuple[int, int]] = []
        protected: set[int] = set()
        for source in topk:
            slot = int(snapshot[source]) if 0 <= source < candidate_len else -1
            if 0 <= slot < budget:
                hits.append((source, slot))
                protected.add(slot)
            else:
                misses.append(source)

        victims: list[tuple[int, int]] = []
        for victim_source in range(candidate_len):
            if len(victims) == len(misses):
                break
            slot = int(snapshot[victim_source])
            if slot < 0 or slot >= budget or slot in protected:
                continue
            protected.add(slot)
            victims.append((victim_source, slot))
        if len(victims) != len(misses):
            raise AssertionError("fixture has insufficient LIDU victims")

        ordered_sources = misses + [source for source, _ in hits]
        ordered_slots = [slot for _, slot in victims] + [slot for _, slot in hits]
        source_ids[b, 0] = torch.tensor(ordered_sources, dtype=torch.int32)
        destination_slots[b, 0] = torch.tensor(ordered_slots, dtype=torch.int32)
        miss_counts[b] = len(misses)
        for index, ((victim, slot), incoming) in enumerate(zip(victims, misses)):
            placement_delta[b, 0, index] = victim
            placement_delta[b, 1, index] = incoming
            placement_delta[b, 2, index] = slot

        ready_count = 0
        fallback_count = 0
        for index, (source, destination) in enumerate(
            zip(ordered_sources, ordered_slots)
        ):
            logical = int(logical_token_map[pool, source])
            view[b, 0, index] = logical
            view[b, 1, index] = destination
            if index >= len(misses):
                view[b, 2, index] = VIEW_RESIDENT_HIT
                continue
            private_slot = int(private_source_to_slot[b, source])
            if 0 <= private_slot < private_capacity and int(
                private_ready.reshape(-1)[private_slot]
            ) == PRIVATE_READY_AFTER_CQ:
                view[b, 2, index] = VIEW_PRIVATE_READY
                view[b, 3, index] = private_slot
                ready_source[b, 0, ready_count] = source
                ready_slot[b, 0, ready_count] = private_slot
                ready_destination[b, 0, ready_count] = destination
                ready_count += 1
            else:
                view[b, 2, index] = VIEW_FALLBACK_REQUIRED
                fallback_logical[b, 0, fallback_count] = logical
                fallback_destination[b, 0, fallback_count] = destination
                fallback_count += 1
        ready_counts[b] = ready_count
        fallback_counts[b] = fallback_count

    return Expected(
        source_ids,
        destination_slots,
        miss_counts,
        placement_delta,
        view,
        ready_source,
        ready_slot,
        ready_destination,
        ready_counts,
        fallback_logical,
        fallback_destination,
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
        *inputs_npu
    )
    torch.npu.synchronize()
    if len(outputs) != len(expected):
        raise AssertionError(f"expected {len(expected)} outputs, got {len(outputs)}")
    for index, (actual, wanted) in enumerate(zip(outputs, expected)):
        if not torch.equal(actual.cpu(), wanted):
            raise AssertionError(
                f"output {index} differs from the LIDU-equivalent reference"
            )
    if not torch.equal(inputs_npu[2].cpu(), snapshot_before):
        raise AssertionError("cache_slots_snapshot changed; reconcile crossed transaction boundary")

    # It must be repeatable against the same immutable snapshot.  In contrast,
    # legacy LIDU's second call sees its published mapping and becomes a
    # different (usually zero-miss) operation.
    replay = torch.ops.a5_exact_reconcile_private_view.exact_reconcile_private_view(
        *inputs_npu
    )
    torch.npu.synchronize()
    if not all(torch.equal(left.cpu(), right.cpu()) for left, right in zip(outputs, replay)):
        raise AssertionError("transaction-local reconcile is not replay-stable")

    miss = expected.miss_counts.tolist()
    ready = expected.ready_private_counts.tolist()
    fallback = expected.fallback_counts.tolist()
    if any(r + f != m for m, r, f in zip(miss, ready, fallback)):
        raise AssertionError("ready/fallback split does not cover the exact miss prefix")
    print(
        "A5_EXACT_RECONCILE_PRIVATE_VIEW_PASS "
        f"batch={len(miss)} misses={miss} ready={ready} fallback={fallback} "
        "lidu_order=miss_prefix_hit_suffix snapshot_unchanged=1 "
        "placement_delta_plan_only=1 qsfa_called=0 guaranteed_get_called=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
