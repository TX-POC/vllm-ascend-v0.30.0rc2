#!/usr/bin/env python3
"""Mathematical contract tests for streaming-LI exact reconciliation.

The official C8 Lightning Indexer emits indices in a *score-source* domain S.
The KV cache directory, Direct-HCCP descriptors, and QSFA private view use a
different *logical KV-token* domain L. With ``cmp_ratio == 1`` the two happen
to be identical; treating that coincidence as an ABI invariant is incorrect.

This file deliberately imports neither torch nor torch_npu. It proves two
properties before a target OPP build:

1. With the normal identity map, the immutable reconciliation plan is exactly
   the legacy LIDU mutation plan, only recorded as ``placement_delta``.
2. With ``S_active != L_active``, exact C8 source choices are first translated
   to real logical tokens. The resulting plan equals a logical-domain LIDU
   baseline, while a padded score source, invalid logical map, or duplicate
   logical selection fails closed.
"""

from __future__ import annotations

import copy
import random
import unittest
from dataclasses import dataclass


TOPK = 2048
MAX_CACHE_TOKENS = 16256
PRIVATE_READY_AFTER_CQ = 2


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


def _validate_logical_inputs(
    exact_topk_logical: list[int], cache_slots: list[int], cache_tokens: int,
    logical_candidate_len: int,
) -> None:
    if len(exact_topk_logical) != TOPK:
        raise ValueError("exact_topk must contain 2048 logical token IDs")
    if not (TOPK <= cache_tokens <= MAX_CACHE_TOKENS):
        raise ValueError("cache_tokens is outside legacy LIDU's active range")
    if not (TOPK <= logical_candidate_len <= len(cache_slots)):
        raise ValueError("logical_candidate_len is outside the active directory")
    if any(token < 0 or token >= logical_candidate_len for token in exact_topk_logical):
        raise ValueError("exact logical Top-K contains an invalid token")
    if len(set(exact_topk_logical)) != TOPK:
        raise ValueError("exact logical Top-K must not select a KV twice")


def _classify_logical(
    exact_topk_logical: list[int], cache_slots: list[int], cache_tokens: int,
) -> tuple[list[int], list[tuple[int, int]], set[int]]:
    """The miss-prefix / hit-suffix portion of legacy LIDU in domain L."""

    misses: list[int] = []
    hits: list[tuple[int, int]] = []
    protected_slots: set[int] = set()
    for logical_token in exact_topk_logical:
        slot = cache_slots[logical_token]
        if 0 <= slot < cache_tokens:
            hits.append((logical_token, slot))
            protected_slots.add(slot)
        else:
            misses.append(logical_token)
    return misses, hits, protected_slots


def legacy_logical_lidu_update(
    exact_topk_logical: list[int], cache_slots: list[int], cache_tokens: int,
    logical_candidate_len: int,
) -> LegacyResult:
    """Reference mutation semantics after C8 source IDs are resolved to L."""

    _validate_logical_inputs(
        exact_topk_logical, cache_slots, cache_tokens, logical_candidate_len
    )
    misses, hits, protected_slots = _classify_logical(
        exact_topk_logical, cache_slots, cache_tokens
    )
    mutations: list[tuple[int, int, int]] = []
    assigned = 0
    # LIDU's source-order victim traversal is an ascending scan in its active
    # directory domain. Here that domain is explicitly the real logical KV
    # directory, not C8's compressed score-source space.
    for victim_logical in range(logical_candidate_len):
        if assigned == len(misses):
            break
        slot = cache_slots[victim_logical]
        if slot < 0 or slot >= cache_tokens or slot in protected_slots:
            continue
        incoming_logical = misses[assigned]
        protected_slots.add(slot)
        cache_slots[victim_logical] = -1
        cache_slots[incoming_logical] = slot
        mutations.append((victim_logical, incoming_logical, slot))
        assigned += 1
    if assigned != len(misses) or len(misses) + len(hits) != TOPK:
        raise AssertionError("fixture violates logical-domain LIDU feasibility")
    return LegacyResult(
        tuple(misses + [token for token, _ in hits]),
        tuple([slot for _, _, slot in mutations] + [slot for _, slot in hits]),
        len(misses),
        tuple(mutations),
    )


def _resolve_exact_sources(
    exact_topk_sources: list[int], logical_token_map: list[int],
    score_source_len: int, logical_candidate_len: int,
) -> list[int]:
    """The required source-S -> logical-L boundary, with fail-closed guards."""

    if len(exact_topk_sources) != TOPK:
        raise ValueError("exact C8 Top-K must contain 2048 source IDs")
    if not (TOPK <= score_source_len <= len(logical_token_map)):
        raise ValueError("score_source_len is outside the score-source capacity")
    logical_topk: list[int] = []
    for source in exact_topk_sources:
        if source < 0 or source >= score_source_len:
            raise ValueError("exact C8 Top-K contains a padded/unscored source")
        logical = logical_token_map[source]
        if logical < 0 or logical >= logical_candidate_len:
            raise ValueError("score source cannot resolve to an active logical KV")
        logical_topk.append(logical)
    if len(set(logical_topk)) != TOPK:
        raise ValueError("source-to-logical map duplicates an exact selected KV")
    return logical_topk


def reconcile_private_plan(
    exact_topk_sources: list[int], cache_slots_snapshot: list[int],
    cache_tokens: int, logical_candidate_len: int, score_source_len: int,
    logical_token_map: list[int], private_source_to_slot: list[int],
    private_ready: list[int],
) -> ReconcileResult:
    """Read-only P3 equivalent of logical-domain LIDU plus private split."""

    if len(private_source_to_slot) != len(logical_token_map):
        raise ValueError("private_source_to_slot must use score-source capacity")
    exact_logical = _resolve_exact_sources(
        exact_topk_sources, logical_token_map, score_source_len,
        logical_candidate_len,
    )
    _validate_logical_inputs(
        exact_logical, cache_slots_snapshot, cache_tokens, logical_candidate_len
    )
    misses: list[tuple[int, int]] = []  # (score source, logical KV)
    hits: list[tuple[int, int]] = []  # (logical KV, resident slot)
    protected_slots: set[int] = set()
    for score_source, logical in zip(exact_topk_sources, exact_logical):
        slot = cache_slots_snapshot[logical]
        if 0 <= slot < cache_tokens:
            hits.append((logical, slot))
            protected_slots.add(slot)
        else:
            misses.append((score_source, logical))

    placement_delta: list[tuple[int, int, int]] = []
    assigned = 0
    for victim_logical in range(logical_candidate_len):
        if assigned == len(misses):
            break
        slot = cache_slots_snapshot[victim_logical]
        if slot < 0 or slot >= cache_tokens or slot in protected_slots:
            continue
        incoming_logical = misses[assigned][1]
        protected_slots.add(slot)
        placement_delta.append((victim_logical, incoming_logical, slot))
        assigned += 1
    if assigned != len(misses) or len(misses) + len(hits) != TOPK:
        raise AssertionError("fixture violates reconcile victim feasibility")

    source_ids = tuple(
        [logical for _, logical in misses] + [logical for logical, _ in hits]
    )
    destination_slots = tuple(
        [slot for _, _, slot in placement_delta] + [slot for _, slot in hits]
    )
    ready_plan: list[tuple[int, int, int]] = []
    fallback_plan: list[tuple[int, int]] = []
    for miss_index, (score_source, logical) in enumerate(misses):
        destination = destination_slots[miss_index]
        private_slot = private_source_to_slot[score_source]
        if 0 <= private_slot < len(private_ready) and (
            private_ready[private_slot] == PRIVATE_READY_AFTER_CQ
        ):
            ready_plan.append((logical, private_slot, destination))
        else:
            fallback_plan.append((logical, destination))
    return ReconcileResult(
        source_ids,
        destination_slots,
        len(misses),
        tuple(placement_delta),
        tuple(ready_plan),
        tuple(fallback_plan),
    )


def _make_directory(
    rng: random.Random, logical_capacity: int, logical_candidate_len: int,
    cache_tokens: int,
) -> list[int]:
    snapshot = [-1] * logical_capacity
    logical_residents = rng.sample(range(logical_candidate_len), cache_tokens)
    slots = list(range(cache_tokens))
    rng.shuffle(slots)
    for logical, slot in zip(logical_residents, slots):
        snapshot[logical] = slot
    return snapshot


class ExactReconcilePrivateViewMathTest(unittest.TestCase):
    def test_identity_map_is_legacy_lidu_exact(self) -> None:
        rng = random.Random(0xA5ECA11)
        for case_id in range(32):
            logical_capacity = rng.choice((4096, 6144, 8192))
            logical_len = rng.randint(TOPK, logical_capacity)
            cache_tokens = rng.randint(TOPK, min(logical_len, 4096))
            snapshot = _make_directory(
                rng, logical_capacity, logical_len, cache_tokens
            )
            exact_sources = rng.sample(range(logical_len), TOPK)
            source_map = list(range(logical_capacity))
            private_a = [-1] * logical_capacity
            private_b = [-1] * logical_capacity
            ready_a = [0] * 257
            ready_b = [0] * 257
            for source in exact_sources:
                if rng.randrange(4) == 0:
                    slot = rng.randrange(len(ready_a))
                    private_a[source] = slot
                    ready_a[slot] = rng.randrange(4)
                if rng.randrange(3) == 0:
                    slot = rng.randrange(len(ready_b))
                    private_b[source] = slot
                    ready_b[slot] = rng.randrange(4)

            legacy = legacy_logical_lidu_update(
                exact_sources, copy.deepcopy(snapshot), cache_tokens, logical_len
            )
            snapshot_before = copy.deepcopy(snapshot)
            plan_a = reconcile_private_plan(
                exact_sources, snapshot, cache_tokens, logical_len, logical_len,
                source_map, private_a, ready_a,
            )
            plan_b = reconcile_private_plan(
                exact_sources, snapshot, cache_tokens, logical_len, logical_len,
                source_map, private_b, ready_b,
            )
            with self.subTest(case_id=case_id):
                self.assertEqual(snapshot, snapshot_before)
                self.assertEqual(plan_a.source_ids, legacy.source_ids)
                self.assertEqual(plan_a.destination_slots, legacy.destination_slots)
                self.assertEqual(plan_a.miss_count, legacy.miss_count)
                self.assertEqual(plan_a.placement_delta, legacy.mutations)
                self.assertEqual(plan_b.source_ids, plan_a.source_ids)
                self.assertEqual(plan_b.destination_slots, plan_a.destination_slots)
                self.assertEqual(plan_b.miss_count, plan_a.miss_count)
                self.assertEqual(plan_b.placement_delta, plan_a.placement_delta)
                self.assertEqual(
                    len(plan_a.ready_plan) + len(plan_a.fallback_plan),
                    plan_a.miss_count,
                )

    def test_compressed_score_sources_map_to_logical_lidu_baseline(self) -> None:
        rng = random.Random(0xC8A5BEEF)
        for case_id in range(24):
            logical_capacity = 16384
            logical_len = rng.randint(8192, logical_capacity)
            score_capacity = 6144
            score_len = rng.randint(TOPK, 4096)
            cache_tokens = rng.randint(TOPK, 4096)
            snapshot = _make_directory(
                rng, logical_capacity, logical_len, cache_tokens
            )
            # C8's active score sources cover a non-contiguous subset of the
            # logical KV directory. Sources in [score_len, score_capacity) are
            # storage capacity, not scorable keys for this row.
            logical_map = [-1] * score_capacity
            mapped_logicals = rng.sample(range(logical_len), score_len)
            rng.shuffle(mapped_logicals)
            logical_map[:score_len] = mapped_logicals
            exact_sources = rng.sample(range(score_len), TOPK)
            exact_logicals = [logical_map[source] for source in exact_sources]
            private = [-1] * score_capacity
            ready = [0] * 383
            for source in exact_sources[::5]:
                slot = rng.randrange(len(ready))
                private[source] = slot
                ready[slot] = PRIVATE_READY_AFTER_CQ

            legacy = legacy_logical_lidu_update(
                exact_logicals, copy.deepcopy(snapshot), cache_tokens, logical_len
            )
            snapshot_before = copy.deepcopy(snapshot)
            plan = reconcile_private_plan(
                exact_sources, snapshot, cache_tokens, logical_len, score_len,
                logical_map, private, ready,
            )
            with self.subTest(case_id=case_id):
                self.assertEqual(snapshot, snapshot_before)
                self.assertEqual(plan.source_ids, legacy.source_ids)
                self.assertEqual(plan.destination_slots, legacy.destination_slots)
                self.assertEqual(plan.miss_count, legacy.miss_count)
                self.assertEqual(plan.placement_delta, legacy.mutations)
                self.assertEqual(
                    len(plan.ready_plan) + len(plan.fallback_plan),
                    plan.miss_count,
                )

    def test_invalid_score_or_logical_mapping_fails_closed(self) -> None:
        logical_len, score_len, score_capacity = 4096, 2048, 4096
        snapshot = list(range(logical_len))
        source_map = list(range(score_capacity))
        private = [-1] * score_capacity
        exact = list(range(TOPK))
        with self.assertRaisesRegex(ValueError, "padded/unscored"):
            reconcile_private_plan(
                exact[:-1] + [score_len], snapshot, TOPK, logical_len,
                score_len, source_map, private, [],
            )
        source_map[TOPK - 1] = logical_len
        with self.assertRaisesRegex(ValueError, "active logical"):
            reconcile_private_plan(
                exact, snapshot, TOPK, logical_len, score_len,
                source_map, private, [],
            )
        source_map[TOPK - 1] = source_map[0]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            reconcile_private_plan(
                exact, snapshot, TOPK, logical_len, score_len,
                source_map, private, [],
            )

    def test_c4_rows_sharing_one_pool_are_not_a_global_lidu_baseline(self) -> None:
        """Two C=4 rows may plan the same victim slots from one snapshot.

        Reconcile is intentionally transaction-local and sees an immutable
        directory.  It therefore produces two individually valid proposals
        when C8 output rows share a pool, but they cannot both be published as
        a serial legacy-LIDU update.  The caller needs a distinct-pool proof or
        deterministic cross-row merge before materialization/publication.
        """
        logical_len = score_len = 4096
        cache_tokens = TOPK
        snapshot = [-1] * logical_len
        snapshot[:TOPK] = list(range(TOPK))
        exact_sources = list(range(TOPK, 2 * TOPK))
        identity_map = list(range(score_len))
        private = [-1] * score_len

        first = reconcile_private_plan(
            exact_sources, snapshot, cache_tokens, logical_len, score_len,
            identity_map, private, [],
        )
        second = reconcile_private_plan(
            exact_sources, snapshot, cache_tokens, logical_len, score_len,
            identity_map, private, [],
        )
        self.assertEqual(first.placement_delta, second.placement_delta)
        self.assertEqual(
            {slot for _, _, slot in first.placement_delta}, set(range(TOPK))
        )
        self.assertEqual(first.miss_count, TOPK)
        self.assertEqual(second.miss_count, TOPK)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=0)
    if result.result.wasSuccessful():
        print(
            "A5_EXACT_RECONCILE_PRIVATE_VIEW_MATH_PASS "
            "identity_cases=32 compressed_cases=24 "
            "score_source_bounds=1 logical_map_bounds=1 "
            "private_affects_only_disposition=1 "
            "shared_pool_rows_require_merge=1",
            flush=True,
        )
