#!/usr/bin/env python3
"""Pure-Python mathematical property test for P3 exact reconciliation.

Run directly; it deliberately imports neither torch nor torch_npu:

    python3 test_exact_reconcile_private_view_math.py

The two reference implementations below are intentionally separate:

* ``legacy_lidu_update`` is the mutation semantics of
  ``a5_lidu_cache_update.cpp``.
* ``reconcile_private_plan`` is the required transaction-local semantics:
  it reads the same snapshot, returns the legacy result plus a placement
  delta, and has optional private-ready classification only *after* those
  first three results are fixed.

Consequently this test guards the central invariant of the P3 design:
private-ready state can change only the extra ready/fallback disposition; it
cannot change ``source_ids``, ``destination_slots``, ``miss_counts``, or the
legacy placement delta.
"""

from __future__ import annotations

import copy
import random
import unittest
from dataclasses import dataclass


TOPK = 2048
MAX_CACHE_TOKENS = 16256


@dataclass(frozen=True)
class LegacyResult:
    source_ids: tuple[int, ...]
    destination_slots: tuple[int, ...]
    miss_count: int
    mutations: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class ReconcileResult:
    source_ids: tuple[int, ...]
    destination_slots: tuple[int, ...]
    miss_count: int
    placement_delta: tuple[tuple[int, int, int], ...]
    ready_plan: tuple[tuple[int, int, int], ...]
    fallback_plan: tuple[tuple[int, int], ...]


def _validate_inputs(
    exact_topk: list[int], cache_slots: list[int], cache_tokens: int,
    candidate_len: int,
) -> None:
    if len(exact_topk) != TOPK:
        raise ValueError("exact_topk must contain 2048 source indices")
    if not (TOPK <= cache_tokens <= MAX_CACHE_TOKENS):
        raise ValueError("cache_tokens is outside legacy LIDU's active range")
    if not (TOPK <= candidate_len <= len(cache_slots)):
        raise ValueError("candidate_len is outside legacy LIDU's active range")


def _classify(
    exact_topk: list[int], cache_slots: list[int], cache_tokens: int,
    candidate_len: int,
) -> tuple[list[int], list[tuple[int, int]], set[int]]:
    """Exact LIDU miss-prefix/hit-suffix classification."""

    misses: list[int] = []
    hits: list[tuple[int, int]] = []
    protected_slots: set[int] = set()
    for source in exact_topk:
        slot = cache_slots[source] if 0 <= source < candidate_len else -1
        if 0 <= slot < cache_tokens:
            hits.append((source, slot))
            protected_slots.add(slot)
        else:
            misses.append(source)
    return misses, hits, protected_slots


def legacy_lidu_update(
    exact_topk: list[int], cache_slots: list[int], cache_tokens: int,
    candidate_len: int,
) -> LegacyResult:
    """Pure Python mirror of the valid-input LIDU mutation loop.

    ``cache_slots`` is intentionally mutated, exactly like the legacy AIV
    kernel.  The return value records the writes in execution order so the P3
    plan can be compared with them without guessing a victim from a changed
    mapping.
    """

    _validate_inputs(exact_topk, cache_slots, cache_tokens, candidate_len)
    misses, hits, protected_slots = _classify(
        exact_topk, cache_slots, cache_tokens, candidate_len
    )
    mutations: list[tuple[int, int, int]] = []
    assigned = 0
    # This deliberately mirrors LIDU's source-order scan.  The legacy kernel
    # checks cache slots in 2048-source DMA chunks, but that chunking does not
    # alter the scalar first-eligible-source order below.
    for victim_source in range(candidate_len):
        if assigned == len(misses):
            break
        slot = cache_slots[victim_source]
        if slot < 0 or slot >= cache_tokens or slot in protected_slots:
            continue
        incoming_source = misses[assigned]
        if incoming_source < 0 or incoming_source >= candidate_len:
            continue
        protected_slots.add(slot)
        cache_slots[victim_source] = -1
        cache_slots[incoming_source] = slot
        mutations.append((victim_source, incoming_source, slot))
        assigned += 1
    if assigned != len(misses) or len(misses) + len(hits) != TOPK:
        raise AssertionError("fixture violates legacy LIDU victim feasibility")
    source_ids = tuple(misses + [source for source, _ in hits])
    destination_slots = tuple(
        [slot for _, _, slot in mutations] + [slot for _, slot in hits]
    )
    return LegacyResult(source_ids, destination_slots, len(misses), tuple(mutations))


def reconcile_private_plan(
    exact_topk: list[int], cache_slots_snapshot: list[int], cache_tokens: int,
    candidate_len: int, logical_token_map: list[int],
    private_source_to_slot: list[int], private_ready: list[int],
) -> ReconcileResult:
    """Required P3 read-only counterpart of ``legacy_lidu_update``.

    This function is intentionally not implemented by calling the legacy
    function on a copy: it models the device operator's snapshot-only victim
    scan.  The private classification begins only after exact source/slot
    ordering has been finalized.
    """

    _validate_inputs(exact_topk, cache_slots_snapshot, cache_tokens, candidate_len)
    if len(logical_token_map) != len(cache_slots_snapshot):
        raise ValueError("logical_token_map shape mismatch")
    if len(private_source_to_slot) != len(cache_slots_snapshot):
        raise ValueError("private_source_to_slot shape mismatch")
    misses, hits, protected_slots = _classify(
        exact_topk, cache_slots_snapshot, cache_tokens, candidate_len
    )
    placement_delta: list[tuple[int, int, int]] = []
    assigned = 0
    for victim_source in range(candidate_len):
        if assigned == len(misses):
            break
        slot = cache_slots_snapshot[victim_source]
        if slot < 0 or slot >= cache_tokens or slot in protected_slots:
            continue
        incoming_source = misses[assigned]
        if incoming_source < 0 or incoming_source >= candidate_len:
            continue
        protected_slots.add(slot)
        # Unlike legacy LIDU, do not touch cache_slots_snapshot here.
        placement_delta.append((victim_source, incoming_source, slot))
        assigned += 1
    if assigned != len(misses) or len(misses) + len(hits) != TOPK:
        raise AssertionError("fixture violates reconcile victim feasibility")

    source_ids = tuple(misses + [source for source, _ in hits])
    destination_slots = tuple(
        [slot for _, _, slot in placement_delta] + [slot for _, slot in hits]
    )
    ready_plan: list[tuple[int, int, int]] = []
    fallback_plan: list[tuple[int, int]] = []
    for index in range(len(misses)):
        source = source_ids[index]
        destination = destination_slots[index]
        private_slot = private_source_to_slot[source]
        if 0 <= private_slot < len(private_ready) and private_ready[private_slot] == 1:
            ready_plan.append((source, private_slot, destination))
        else:
            fallback_plan.append((logical_token_map[source], destination))
    return ReconcileResult(
        source_ids,
        destination_slots,
        len(misses),
        tuple(placement_delta),
        tuple(ready_plan),
        tuple(fallback_plan),
    )


def make_valid_case(rng: random.Random) -> tuple[list[int], list[int], int, int]:
    """Construct one valid cache mapping with a feasible 2048-element Top-K."""

    source_capacity = rng.choice((2048, 2057, 3072, 4096, 6144))
    candidate_len = rng.randint(TOPK, source_capacity)
    cache_tokens = rng.randint(TOPK, min(candidate_len, 6144))
    cache_slots = [-1] * source_capacity
    # A valid resident directory is one-to-one for active slots.  Permuting
    # both source IDs and destination slots is important: it proves the
    # comparison depends on legacy source-order victim selection, not on a
    # convenient slot/source identity mapping.
    resident_sources = rng.sample(range(candidate_len), cache_tokens)
    resident_slots = list(range(cache_tokens))
    rng.shuffle(resident_slots)
    for source, slot in zip(resident_sources, resident_slots):
        cache_slots[source] = slot
    exact_topk = rng.sample(range(candidate_len), TOPK)
    return exact_topk, cache_slots, cache_tokens, candidate_len


class ExactReconcilePrivateViewMathTest(unittest.TestCase):
    def test_random_legacy_equivalence_and_private_independence(self) -> None:
        rng = random.Random(0xA5ECA11)
        for case_id in range(48):
            exact_topk, snapshot, cache_tokens, candidate_len = make_valid_case(rng)
            logical_map = [1000003 + 11 * source for source in range(len(snapshot))]
            private_source_to_slot_a = [-1] * len(snapshot)
            private_source_to_slot_b = [-1] * len(snapshot)
            private_ready_a = [0] * 257
            private_ready_b = [0] * 257
            for source in exact_topk:
                if rng.randrange(4) == 0:
                    slot = rng.randrange(len(private_ready_a))
                    private_source_to_slot_a[source] = slot
                    private_ready_a[slot] = rng.randrange(2)
                if rng.randrange(3) == 0:
                    slot = rng.randrange(len(private_ready_b))
                    private_source_to_slot_b[source] = slot
                    private_ready_b[slot] = rng.randrange(2)

            legacy_mapping = copy.deepcopy(snapshot)
            legacy = legacy_lidu_update(
                exact_topk, legacy_mapping, cache_tokens, candidate_len
            )
            snapshot_before = copy.deepcopy(snapshot)
            plan_a = reconcile_private_plan(
                exact_topk,
                snapshot,
                cache_tokens,
                candidate_len,
                logical_map,
                private_source_to_slot_a,
                private_ready_a,
            )
            plan_b = reconcile_private_plan(
                exact_topk,
                snapshot,
                cache_tokens,
                candidate_len,
                logical_map,
                private_source_to_slot_b,
                private_ready_b,
            )

            with self.subTest(case_id=case_id):
                self.assertEqual(snapshot, snapshot_before)
                self.assertEqual(plan_a.source_ids, legacy.source_ids)
                self.assertEqual(plan_a.destination_slots, legacy.destination_slots)
                self.assertEqual(plan_a.miss_count, legacy.miss_count)
                self.assertEqual(plan_a.placement_delta, legacy.mutations)
                # Private state is only a post-exact disposition; it cannot
                # alter any LIDU-equivalent output or placement choice.
                self.assertEqual(plan_b.source_ids, plan_a.source_ids)
                self.assertEqual(plan_b.destination_slots, plan_a.destination_slots)
                self.assertEqual(plan_b.miss_count, plan_a.miss_count)
                self.assertEqual(plan_b.placement_delta, plan_a.placement_delta)
                self.assertEqual(
                    len(plan_a.ready_plan) + len(plan_a.fallback_plan),
                    plan_a.miss_count,
                )
                self.assertEqual(
                    len(plan_b.ready_plan) + len(plan_b.fallback_plan),
                    plan_b.miss_count,
                )

    def test_duplicate_topk_source_preserves_legacy_order(self) -> None:
        """Duplicates are not expected from official Top-K, but stay exact."""

        source_capacity = 4096
        cache_tokens = 3072
        snapshot = [-1] * source_capacity
        for source, slot in zip(range(cache_tokens), range(cache_tokens)):
            snapshot[source] = slot
        # Repeating a resident source tests that protected-slot semantics—not
        # an accidental unique-source assumption—drive both implementations.
        exact_topk = list(range(TOPK - 1)) + [17]
        logical_map = list(range(source_capacity))
        private_map = [-1] * source_capacity
        snapshot_before = copy.deepcopy(snapshot)
        legacy_mapping = copy.deepcopy(snapshot)
        legacy = legacy_lidu_update(
            exact_topk, legacy_mapping, cache_tokens, source_capacity
        )
        plan = reconcile_private_plan(
            exact_topk,
            snapshot,
            cache_tokens,
            source_capacity,
            logical_map,
            private_map,
            [],
        )
        self.assertEqual(plan.source_ids, legacy.source_ids)
        self.assertEqual(plan.destination_slots, legacy.destination_slots)
        self.assertEqual(plan.miss_count, legacy.miss_count)
        self.assertEqual(plan.placement_delta, legacy.mutations)
        self.assertEqual(snapshot, snapshot_before)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=0)
    if result.result.wasSuccessful():
        print(
            "A5_EXACT_RECONCILE_PRIVATE_VIEW_MATH_PASS "
            "random_cases=48 duplicate_case=1 private_affects_only_disposition=1",
            flush=True,
        )
