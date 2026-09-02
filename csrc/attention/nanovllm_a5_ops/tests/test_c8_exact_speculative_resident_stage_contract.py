#!/usr/bin/env python3
"""Static source gate for the P9 direct-resident C8 target.

The target deliberately remains non-buildable until the new KVAF
direct-resident control ABI is landed. This test prevents the next CANN patch
from inheriting old P7/P8 private transport or silently changing the official
C8 score/Top-K seam.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest


OPS_ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = OPS_ROOT / "a5_c8_exact_speculative_resident_get"
CONTRACT = json.loads(
    (TARGET_ROOT / "stage_a_b_contract.json").read_text(encoding="utf-8")
)
KERNEL_CONTRACT = (
    TARGET_ROOT
    / "op_kernel"
    / "arch35"
    / "a5_c8_exact_speculative_resident_contract.h"
).read_text(encoding="utf-8")
OVERLAY_README = (
    TARGET_ROOT / "source_overlay" / "README.md"
).read_text(encoding="utf-8")
OFFICIAL_VECTOR = (
    OPS_ROOT.parent
    / "vllm_quant_lightning_indexer"
    / "op_kernel"
    / "arch35"
    / "quant_lightning_indexer_service_vector.h"
).read_text(encoding="utf-8")
BUILD_SCRIPT = (OPS_ROOT / "build_nanovllm_ops.sh").read_text(encoding="utf-8")


class C8ExactSpeculativeResidentStageContractTest(unittest.TestCase):
    def test_source_is_derived_from_the_official_score_seam(self) -> None:
        source = CONTRACT["source_derived_from"]
        self.assertEqual(source["operator"], "VllmQuantLightningIndexer")
        seam = source["score_seam"]
        self.assertEqual(seam["function"], "QLIVector::ProcessVec1")
        self.assertEqual(seam["after"], "vector1::BatchMulWeightAndReduceSum")
        self.assertEqual(seam["before"], "DataCopyPad(scoreGm")
        self.assertEqual(source["topk_owner"], "official ProcessTopK unchanged")

        process_vec = OFFICIAL_VECTOR.index("QLIVector<QLIT>::ProcessVec1")
        process_topk = OFFICIAL_VECTOR.index("QLIVector<QLIT>::ProcessTopK")
        score_reduce = OFFICIAL_VECTOR.index(
            "vector1::BatchMulWeightAndReduceSum", process_vec
        )
        score_publish = OFFICIAL_VECTOR.index("DataCopyPad(scoreGm", score_reduce)
        self.assertLess(process_vec, score_reduce)
        self.assertLess(score_reduce, score_publish)
        self.assertLess(score_publish, process_topk)
        self.assertIn("exact checked-out", OVERLAY_README)
        self.assertIn("byte-for-byte", OVERLAY_README)

    def test_direct_resident_candidate_layout_has_no_private_indirection(self) -> None:
        self.assertEqual(CONTRACT["target"], "A5C8ExactSpeculativeResidentGet")
        self.assertEqual(
            CONTRACT["candidate_seed"]["words"],
            ["row", "source", "kv_object", "resident_slot"],
        )
        self.assertEqual(
            CONTRACT["candidate_seed"]["cross_wave_slot_invariant"],
            "CandidateCompact never assigns one resident_slot to two initial "
            "waves; Stage-B may reuse a non-winner slot only after every "
            "initial wave targeting it is terminal",
        )
        resource = CONTRACT["resource_model"]
        fixed = resource["canonical_resident_range"]
        self.assertEqual(fixed["input"], "speculative_resident_slots")
        self.assertEqual(fixed["shape"], "[R, 2048]")
        self.assertEqual(fixed["slot_count"], 2048)
        self.assertFalse(resource["fixed_get_core_partition"])
        self.assertFalse(resource["persistent_slot_map_mutation_during_score_scan"])
        for token in (
            "kCanonicalResidentSlotCount = 2048U",
            "kCandidateResidentSlot = 3U",
            "struct CandidateSeed",
            "struct CanonicalResidentRange",
            "kInitialDrain",
            "kExactRecall",
            "kRecallDrain",
            "kComplete",
            "IsSlotInRange",
        ):
            self.assertIn(token, KERNEL_CONTRACT)
        for retired_token in (
            "p8_",
            "private_",
            "arena_",
            "quarantine",
            "forward_control",
            "replay",
            "epoch",
        ):
            self.assertNotIn(retired_token, KERNEL_CONTRACT.lower())

    def test_stage_b_exposes_only_terminal_direct_slots_to_qsfa(self) -> None:
        stage_b = CONTRACT["stage_b"]
        self.assertEqual(stage_b["name"], "topk_recall_and_terminal_drain")
        self.assertEqual(
            stage_b["winner_resolution_order"],
            [
                "persistent_resident_slot",
                "terminal_speculative_resident_slot",
                "fixed_slot_after_initial_drain_then_exact_recall",
            ],
        )
        self.assertEqual(
            stage_b["completion_condition"],
            "all_initial_and_recall_posts_terminal",
        )
        self.assertEqual(
            stage_b["qsfa_contract"],
            "QSFA consumes only final attention_slots after this condition holds",
        )
        self.assertEqual(stage_b["outputs"], ["attention_slots", "terminal_post_stats"])

    def test_target_is_not_accidentally_build_selectable_before_its_real_closure(self) -> None:
        """A source scaffold must not route through the old streaming OPP."""
        self.assertNotIn("A5C8ExactSpeculativeResidentGet", BUILD_SCRIPT)
        self.assertIn("non-buildable", OVERLAY_README)
        self.assertEqual(
            CONTRACT["non_goals"],
            [
                "cross_forward_transport",
                "cross_layer_transport",
                "private_slot_indirection",
                "persistent_directory_updates_before_exact_topk",
                "fixed_physical_get_core_partition",
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
