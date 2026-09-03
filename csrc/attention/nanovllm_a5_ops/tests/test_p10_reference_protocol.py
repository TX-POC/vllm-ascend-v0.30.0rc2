"""Strict, device-independent P10 score-receipt protocol tests.

The suite intentionally has no torch, CANN, C8, or KVAF dependency.  It proves
the Stage-A/Stage-B boundary before an expensive custom OPP build is allowed:
Stage A publishes raw exact-score descriptors; Stage B alone scans, compacts,
maps S->O, selects immutable canonical resident rows, and posts.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPS_ROOT))

from a5_dsa_c8_exact_speculative_resident.p10_reference_protocol import (  # noqa: E402
    ProtocolViolation,
    ReceiptState,
    ScoreObservation,
    ScoreReceiptResidentProtocol,
    summarize_paths,
    winner_objects_from_exact_sources,
)


class P10ReferenceProtocolTest(unittest.TestCase):
    def _protocol(
        self,
        *,
        workers: int = 1,
        ring_slots: int = 2,
        seed_capacity: int = 64,
        queue_count: int = 2,
        active_rows: int | None = None,
    ) -> ScoreReceiptResidentProtocol:
        # Deliberately non-identity S -> O conversion makes it impossible for
        # the tests to validate source IDs in place of KV objects.
        directory = (
            (17, 4, 31, 8, 22, 10, 41, 6),
            (103, 99, 87, 75, 63, 51, 39, 27),
        )
        canonical_slots = {
            (row, obj): row * 100 + source
            for row, objects in enumerate(directory)
            for source, obj in enumerate(objects)
        }
        return ScoreReceiptResidentProtocol(
            object_directory=directory,
            active_row_count=active_rows,
            workers=workers,
            ring_slots_per_worker=ring_slots,
            seed_capacity=seed_capacity,
            hccp_queue_count=queue_count,
            canonical_resident_slots=canonical_slots,
        )

    @staticmethod
    def _publish(
        protocol: ScoreReceiptResidentProtocol,
        *,
        worker: int = 0,
        tile: int = 0,
        offset: int = 0,
        observations: tuple[ScoreObservation, ...],
    ):
        return protocol.publish_exact_score_window(
            worker=worker,
            tile=tile,
            observations=observations,
            score_surface_offset_bytes=offset,
        )

    @staticmethod
    def _drain_all(
        protocol: ScoreReceiptResidentProtocol,
        *,
        exact_winners: tuple[tuple[int, ...], ...],
        threshold_by_row: dict[int, int] | None = None,
        persistent_object_slots: dict[tuple[int, int], int] | None = None,
    ) -> None:
        thresholds = threshold_by_row or {
            row: 1 for row in range(protocol.active_row_count)
        }
        claimed: list[tuple[object, tuple[object, ...]]] = []
        while True:
            receipt = protocol.claim_ready_receipt()
            if receipt is None:
                break
            seeds = protocol.compact_claimed_receipt(
                receipt,
                threshold_by_row=thresholds,
                persistent_object_slots=persistent_object_slots,
            )
            claimed.append((receipt, seeds))

        # A bounded queue may leave a descriptor CLAIMED.  CQ progress is the
        # only way to make retryable capacity available; no receipt is dropped.
        while claimed:
            waiting: list[tuple[object, tuple[object, ...]]] = []
            for receipt, seeds in claimed:
                if protocol.post_compacted_receipt(receipt, seeds):
                    protocol.recycle_done_receipt(receipt)
                else:
                    waiting.append((receipt, seeds))
            claimed = waiting
            if claimed:
                if protocol.drain_cq(limit=1) != 1:
                    raise AssertionError("bounded queue made no retry progress")

        # This is intentionally before executor terminal.  Any unresolved
        # exact winner enters the same bounded PostOnly/CQ path as speculation.
        protocol.admit_final_winners(
            exact_winners,
            persistent_object_slots=persistent_object_slots,
        )
        protocol.drain_cq()
        for worker in range(protocol.workers):
            protocol.mark_producer_done(worker)
        protocol.mark_executor_terminal()

    def test_stage_a_publishes_a_raw_descriptor_without_stage_b_policy_inputs(self):
        protocol = self._protocol()
        receipt = self._publish(
            protocol,
            tile=11,
            offset=128,
            observations=(
                ScoreObservation(0, 2, 73),
                ScoreObservation(0, 3, 71),
            ),
        )
        self.assertIsNotNone(receipt)
        assert receipt is not None
        self.assertEqual(receipt.descriptor.row, 0)
        self.assertEqual(receipt.descriptor.source_begin, 2)
        self.assertEqual(receipt.descriptor.valid_count, 2)
        self.assertEqual(receipt.descriptor.score_surface_offset_bytes, 128)
        self.assertEqual(receipt.descriptor.score_element_bytes, 2)
        self.assertEqual(receipt.score_surface_values, (73, 71))
        self.assertEqual(protocol.stats.candidates_compacted, 0)

        publish_parameters = inspect.signature(
            protocol.publish_exact_score_window
        ).parameters
        for forbidden in (
            "threshold_by_row",
            "persistent_object_slots",
            "object_directory",
            "resident_slots",
        ):
            self.assertNotIn(forbidden, publish_parameters)

        claimed = protocol.claim_ready_receipt()
        self.assertEqual(claimed, receipt)
        seeds = protocol.compact_claimed_receipt(
            receipt,
            threshold_by_row={0: 72, 1: 1},
        )
        self.assertEqual([(seed.source, seed.kv_object) for seed in seeds], [(2, 31)])
        self.assertEqual(protocol.stats.candidates_compacted, 1)
        protocol.post_compacted_receipt(receipt, seeds)

    def test_exact_winner_objects_are_independent_of_speculation(self) -> None:
        winners = ((6, 2, 0, 5), (1, 4, 7, 3))
        baseline = winner_objects_from_exact_sources(
            ((17, 4, 31, 8, 22, 10, 41, 6), (103, 99, 87, 75, 63, 51, 39, 27)),
            winners,
        )

        without_speculation = self._protocol()
        self._drain_all(without_speculation, exact_winners=winners)
        no_spec_result = without_speculation.finalize(winners)

        with_speculation = self._protocol()
        self.assertIsNotNone(
            self._publish(
                with_speculation,
                tile=0,
                observations=(
                    ScoreObservation(0, 1, 62),  # false positive
                    ScoreObservation(0, 2, 61),
                    ScoreObservation(0, 3, 10),
                    ScoreObservation(0, 4, 10),
                    ScoreObservation(0, 5, 10),
                    ScoreObservation(0, 6, 63),
                ),
            )
        )
        self.assertIsNotNone(
            self._publish(
                with_speculation,
                tile=1,
                offset=12,
                observations=(
                    ScoreObservation(1, 4, 70),
                    ScoreObservation(1, 5, 10),
                    ScoreObservation(1, 6, 10),
                    ScoreObservation(1, 7, 69),
                ),
            )
        )
        self._drain_all(
            with_speculation,
            exact_winners=winners,
            threshold_by_row={0: 60, 1: 60},
        )
        spec_result = with_speculation.finalize(winners)

        self.assertEqual(no_spec_result.winner_objects, baseline)
        self.assertEqual(spec_result.winner_objects, baseline)
        # Transport timing/path cannot change the canonical resident rows seen
        # by downstream QSFA.
        self.assertEqual(no_spec_result.attention_slots, spec_result.attention_slots)
        self.assertEqual(
            summarize_paths(spec_result),
            {"speculative_terminal": 4, "guaranteed_recall": 4},
        )

    def test_stage_b_seed_overflow_is_recalled_without_resident_slot_reuse(self) -> None:
        protocol = self._protocol(
            seed_capacity=2,
        )
        receipt = self._publish(
            protocol,
            tile=4,
            observations=tuple(ScoreObservation(0, source, 90) for source in range(5)),
        )
        self.assertIsNotNone(receipt)
        assert receipt is not None
        claimed = protocol.claim_ready_receipt()
        self.assertEqual(claimed, receipt)
        seeds = protocol.compact_claimed_receipt(
            receipt, threshold_by_row={0: 1, 1: 1}
        )
        self.assertEqual([seed.source for seed in seeds], [0, 1])
        self.assertEqual(protocol.stats.deferred_seed_capacity, 3)
        protocol.post_compacted_receipt(receipt, seeds)
        protocol.recycle_done_receipt(receipt)
        protocol.admit_final_winners(((0, 1, 2, 3, 4), (-1,)))
        protocol.drain_cq()
        protocol.mark_producer_done(0)
        protocol.mark_executor_terminal()

        result = protocol.finalize(((0, 1, 2, 3, 4), (-1,)))
        row_zero = [item for item in result.placements if item.row == 0]
        self.assertEqual([item.kv_object for item in row_zero], [17, 4, 31, 8, 22])
        self.assertEqual(
            [item.path for item in row_zero],
            [
                "speculative_terminal",
                "speculative_terminal",
                "guaranteed_recall",
                "guaranteed_recall",
                "guaranteed_recall",
            ],
        )

    def test_ring_reuses_only_after_stage_b_copies_the_descriptor(self) -> None:
        protocol = self._protocol(
            ring_slots=1,
            seed_capacity=1,
        )
        first = self._publish(
            protocol,
            tile=0,
            observations=(ScoreObservation(0, 0, 10),),
        )
        self.assertIsNotNone(first)
        assert first is not None
        claimed = protocol.claim_ready_receipt()
        self.assertEqual(claimed, first)
        first_seeds = protocol.compact_claimed_receipt(
            claimed, threshold_by_row={0: 1, 1: 1}
        )
        protocol.post_compacted_receipt(claimed, first_seeds)
        self.assertEqual(protocol.receipt_states(), ((ReceiptState.DONE,),))
        protocol.recycle_done_receipt(claimed)
        self.assertEqual(protocol.receipt_states(), ((ReceiptState.FREE,),))

        second = self._publish(
            protocol,
            tile=1,
            offset=2,
            observations=(ScoreObservation(0, 1, 10),),
        )
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(first.ring_slot, second.ring_slot)
        self.assertEqual(first_seeds[0].resident_slot, 0)
        second_claim = protocol.claim_ready_receipt()
        second_seeds = protocol.compact_claimed_receipt(
            second_claim, threshold_by_row={0: 1, 1: 1}
        )
        self.assertEqual(second_seeds[0].resident_slot, 1)

    def test_ring_full_defers_raw_windows_without_scanning_or_slot_allocation(self) -> None:
        protocol = self._protocol(ring_slots=1)
        first = self._publish(
            protocol, observations=(ScoreObservation(0, 0, 10),)
        )
        self.assertIsNotNone(first)
        second = self._publish(
            protocol,
            tile=1,
            offset=2,
            observations=(ScoreObservation(0, 1, 65535),),
        )
        self.assertIsNone(second)
        self.assertEqual(protocol.stats.deferred_ring_full, 1)
        self.assertEqual(protocol.stats.candidates_compacted, 0)

    def test_invalid_claim_is_done_then_explicitly_recycled_not_silently_freed(self) -> None:
        protocol = self._protocol(ring_slots=1)
        receipt = self._publish(
            protocol, observations=(ScoreObservation(0, 0, 10),)
        )
        assert receipt is not None

        # Model a corrupt published descriptor: the claim CAS has already
        # moved the slot out of READY before descriptor validation fails.
        protocol._records[0][0].receipt = None  # pylint: disable=protected-access
        self.assertIsNone(protocol.claim_ready_receipt())
        self.assertEqual(protocol.receipt_states(), ((ReceiptState.DONE,),))
        self.assertEqual(protocol.stats.receipts_rejected, 1)

        self.assertIsNone(
            self._publish(
                protocol,
                tile=1,
                offset=2,
                observations=(ScoreObservation(0, 1, 10),),
            )
        )
        protocol.recycle_done_receipts()
        self.assertEqual(protocol.receipt_states(), ((ReceiptState.FREE,),))

    def test_final_recall_is_admitted_before_terminal_and_shares_bounded_queue(self) -> None:
        protocol = self._protocol(
            ring_slots=2,
            seed_capacity=1,
            queue_count=1,
        )
        first = self._publish(
            protocol, tile=0, observations=(ScoreObservation(0, 0, 10),)
        )
        second = self._publish(
            protocol,
            tile=1,
            offset=2,
            observations=(ScoreObservation(0, 1, 10),),
        )
        assert first is not None and second is not None

        first_claim = protocol.claim_ready_receipt()
        first_seeds = protocol.compact_claimed_receipt(
            first_claim, threshold_by_row={0: 1, 1: 1}
        )
        self.assertTrue(protocol.post_compacted_receipt(first_claim, first_seeds))
        protocol.recycle_done_receipt(first_claim)
        self.assertEqual(protocol.pending_post_count, 1)
        self.assertEqual(protocol.free_queue_count, 0)

        second_claim = protocol.claim_ready_receipt()
        second_seeds = protocol.compact_claimed_receipt(
            second_claim, threshold_by_row={0: 1, 1: 1}
        )
        self.assertFalse(protocol.post_compacted_receipt(second_claim, second_seeds))
        self.assertEqual(protocol.receipt_states(), ((ReceiptState.FREE, ReceiptState.CLAIMED),))

        # Object 2 was not speculatively scheduled.  Its canonical resident
        # row is admitted now, while one speculative CQ is still pending, and
        # waits on the same one-queue transport pool rather than a new path.
        self.assertEqual(protocol.admit_final_winners(((0, 1, 2), (-1,))), 1)
        self.assertEqual(protocol.recall_backlog_count, 1)
        self.assertGreaterEqual(protocol.stats.deferred_queue_busy, 2)

        self.assertEqual(protocol.drain_cq(limit=1), 1)
        self.assertEqual(protocol.recall_backlog_count, 0)
        self.assertEqual(protocol.pending_post_count, 1)
        self.assertFalse(protocol.post_compacted_receipt(second_claim, second_seeds))

        self.assertEqual(protocol.drain_cq(limit=1), 1)
        self.assertTrue(protocol.post_compacted_receipt(second_claim, second_seeds))
        protocol.recycle_done_receipt(second_claim)
        self.assertEqual(protocol.drain_cq(limit=1), 1)
        protocol.mark_producer_done(0)
        protocol.mark_executor_terminal()

        result = protocol.finalize(((0, 1, 2), (-1,)))
        self.assertEqual(
            [placement.path for placement in result.placements if placement.row == 0],
            ["speculative_terminal", "speculative_terminal", "guaranteed_recall"],
        )

    def test_finalizer_refuses_until_all_producers_and_cq_work_are_terminal(self) -> None:
        protocol = self._protocol()
        receipt = self._publish(
            protocol, observations=(ScoreObservation(0, 0, 10),)
        )
        assert receipt is not None
        claimed = protocol.claim_ready_receipt()
        seeds = protocol.compact_claimed_receipt(
            claimed, threshold_by_row={0: 1, 1: 1}
        )
        protocol.post_compacted_receipt(claimed, seeds)
        protocol.recycle_done_receipt(claimed)
        protocol.mark_producer_done(0)
        with self.assertRaisesRegex(ProtocolViolation, "CQ drain"):
            protocol.mark_executor_terminal()
        with self.assertRaisesRegex(ProtocolViolation, "terminal executor drain"):
            protocol.finalize(((0,), (-1,)))
        protocol.drain_cq()
        with self.assertRaisesRegex(ProtocolViolation, "exact winner admission"):
            protocol.mark_executor_terminal()
        protocol.admit_final_winners(((0,), (-1,)))
        protocol.mark_executor_terminal()
        result = protocol.finalize(((0,), (-1,)))
        self.assertEqual(result.winner_objects, ((17,), (-1,)))

    def test_persistent_hit_is_filtered_only_by_stage_b_not_stage_a(self) -> None:
        protocol = self._protocol()
        receipt = self._publish(
            protocol,
            tile=3,
            observations=(ScoreObservation(0, 2, 99), ScoreObservation(0, 3, 98)),
        )
        assert receipt is not None
        self.assertEqual(receipt.score_surface_values, (99, 98))
        claimed = protocol.claim_ready_receipt()
        seeds = protocol.compact_claimed_receipt(
            claimed,
            threshold_by_row={0: 1, 1: 1},
            persistent_object_slots={(0, 31): 2},
        )
        self.assertEqual([seed.source for seed in seeds], [3])
        self.assertEqual(protocol.stats.skipped_persistent, 1)
        protocol.post_compacted_receipt(claimed, seeds)
        protocol.recycle_done_receipt(claimed)
        protocol.admit_final_winners(
            ((2, 3), (-1,)), persistent_object_slots={(0, 31): 2}
        )
        protocol.drain_cq()
        protocol.mark_producer_done(0)
        protocol.mark_executor_terminal()
        result = protocol.finalize(
            ((2, 3), (-1,)), persistent_object_slots={(0, 31): 2}
        )
        self.assertEqual(result.attention_slots[0], (2, 3))
        self.assertEqual(
            [placement.path for placement in result.placements],
            ["persistent", "speculative_terminal"],
        )

    def test_graph_padding_is_not_a_candidate_producer(self) -> None:
        protocol = self._protocol(active_rows=1)
        receipt = self._publish(
            protocol, observations=(ScoreObservation(1, 0, 99),)
        )
        self.assertIsNone(receipt)
        self.assertEqual(protocol.stats.candidates_compacted, 0)
        self._drain_all(protocol, exact_winners=((0,), (-1,)))
        result = protocol.finalize(((0,), (-1,)))
        self.assertEqual(result.winner_objects, ((17,), (-1,)))

    def test_graph_padded_row_cannot_enter_final_qsfa_mapping(self) -> None:
        protocol = self._protocol(active_rows=1)
        self._drain_all(protocol, exact_winners=((0,), (-1,)))
        with self.assertRaisesRegex(ProtocolViolation, "inactive graph-padded row"):
            protocol.finalize(((0,), (0,)))

    def test_one_kv_object_has_one_canonical_slot_even_with_two_sources(self) -> None:
        protocol = ScoreReceiptResidentProtocol(
            object_directory=((41, 41, 99),),
            workers=1,
            ring_slots_per_worker=1,
            seed_capacity=4,
            canonical_resident_slots={(0, 41): 500, (0, 99): 501},
        )
        receipt = self._publish(
            protocol,
            observations=(ScoreObservation(0, 0, 9), ScoreObservation(0, 1, 9)),
        )
        assert receipt is not None
        claimed = protocol.claim_ready_receipt()
        seeds = protocol.compact_claimed_receipt(claimed, threshold_by_row={0: 1})
        self.assertEqual([seed.source for seed in seeds], [0])
        self.assertEqual(protocol.stats.skipped_duplicate, 1)
        protocol.post_compacted_receipt(claimed, seeds)
        protocol.recycle_done_receipt(claimed)
        protocol.admit_final_winners(((0, 1),))
        protocol.drain_cq()
        protocol.mark_producer_done(0)
        protocol.mark_executor_terminal()
        result = protocol.finalize(((0, 1),))
        self.assertEqual(result.winner_objects, ((41, 41),))
        self.assertEqual(result.attention_slots, ((500, 500),))
        self.assertEqual(summarize_paths(result), {"speculative_terminal": 2})

    def test_exact_winner_cannot_name_an_invalid_kv_object(self) -> None:
        protocol = ScoreReceiptResidentProtocol(
            object_directory=((17, -1),),
            workers=1,
            ring_slots_per_worker=1,
            seed_capacity=1,
            canonical_resident_slots={(0, 17): 42},
        )
        with self.assertRaisesRegex(ProtocolViolation, "invalid KV object"):
            protocol.admit_final_winners(((1,),))

    def test_speculative_and_final_recall_use_the_same_canonical_slot(self) -> None:
        winners = ((2,), (-1,))

        speculative = self._protocol()
        receipt = self._publish(
            speculative,
            observations=(ScoreObservation(0, 2, 99),),
        )
        assert receipt is not None
        self._drain_all(
            speculative,
            exact_winners=winners,
            threshold_by_row={0: 1, 1: 1},
        )
        speculative_result = speculative.finalize(winners)

        recall = self._protocol()
        self._drain_all(recall, exact_winners=winners)
        recall_result = recall.finalize(winners)

        self.assertEqual(speculative_result.attention_slots, ((2,), (-1,)))
        self.assertEqual(recall_result.attention_slots, ((2,), (-1,)))
        self.assertEqual(
            [placement.path for placement in speculative_result.placements],
            ["speculative_terminal"],
        )
        self.assertEqual(
            [placement.path for placement in recall_result.placements],
            ["guaranteed_recall"],
        )


if __name__ == "__main__":
    unittest.main()
