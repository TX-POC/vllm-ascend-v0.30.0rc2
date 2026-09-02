#!/usr/bin/env python3
"""Static source gate for P9's one-invocation direct-resident operator.

The gate intentionally describes the product-facing design, not an older P7
or P8 experiment: C8 owns score and exact Top-K; any worker may obtain a
temporary queue lease; all Direct-HCCP work becomes terminal before the
operator returns; QSFA receives only final normal-resident slots.
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
OPS = json.loads(
    (TARGET_ROOT / "source_overlay" / "ops.json").read_text(encoding="utf-8")
)[0]
KERNEL_CONTRACT = (
    TARGET_ROOT
    / "op_kernel"
    / "arch35"
    / "a5_c8_exact_speculative_resident_contract.h"
).read_text(encoding="utf-8")
SCORE_HOOK = (
    TARGET_ROOT
    / "op_kernel"
    / "arch35"
    / "a5_c8_exact_speculative_resident_score_hook.h"
).read_text(encoding="utf-8")
POST_EXECUTOR = (
    TARGET_ROOT
    / "op_kernel"
    / "arch35"
    / "a5_c8_exact_speculative_resident_post_executor.h"
).read_text(encoding="utf-8")
FINALIZER = (
    TARGET_ROOT
    / "op_kernel"
    / "arch35"
    / "a5_c8_exact_speculative_resident_finalizer.h"
).read_text(encoding="utf-8")
SCORE_SEAM = (
    TARGET_ROOT
    / "source_overlay"
    / "op_kernel"
    / "arch35"
    / "a5_c8_exact_speculative_resident_score_seam.h"
).read_text(encoding="utf-8")
STAGE_B_TAIL = (
    TARGET_ROOT
    / "op_kernel"
    / "arch35"
    / "a5_c8_exact_speculative_resident_stage_b_finalizer.h"
).read_text(encoding="utf-8")
OVERLAY_APPLIER = (
    TARGET_ROOT / "source_overlay" / "apply_p9_overlay.py"
).read_text(encoding="utf-8")
OFFICIAL_VECTOR = (
    OPS_ROOT.parent
    / "vllm_quant_lightning_indexer"
    / "op_kernel"
    / "arch35"
    / "quant_lightning_indexer_service_vector.h"
).read_text(encoding="utf-8")
P9_BUILD_SCRIPT = (
    TARGET_ROOT / "build_p9_exact_resident_opp.sh"
).read_text(encoding="utf-8")

OFFICIAL_INPUTS = [
    ("query", "int8"),
    ("key", "int8"),
    ("weights", "float"),
    ("query_dequant_scale", "float"),
    ("key_dequant_scale", "float"),
    ("actual_seq_lengths_query", "int32"),
    ("actual_seq_lengths_key", "int32"),
    ("block_table", "int32"),
    ("metadata", "int32"),
]
P9_INPUTS = [
    ("runtime_enable", "int32"),
    ("resident_threshold", "int32"),
    ("resident_threshold_valid", "int32"),
    ("req_pool_entries", "int32"),
    ("kv_object_map", "int32"),
    ("kv_object_lens", "int32"),
    ("persistent_object_slots_snapshot", "int32"),
    ("canonical_resident_layout", "int32"),
    ("candidate_source_to_slot", "int32"),
    ("candidate_slot_cursor", "int32"),
    ("layer_resident_get_control", "int64"),
    ("rows_per_entry", "int32"),
]
OUTPUTS = [
    ("sparse_indices", "int32"),
    ("sparse_values", "float"),
    ("attention_slots", "int32"),
    ("terminal_post_stats", "int32"),
    ("candidate_source_to_slot_ref", "int32"),
    ("candidate_slot_cursor_ref", "int32"),
]


def op_schema(entries: list[dict[str, object]]) -> list[tuple[str, str]]:
    return [(entry["name"], entry["type"][0]) for entry in entries]


def contract_schema(entries: list[dict[str, object]]) -> list[tuple[str, str]]:
    return [(entry["name"], entry["dtype"]) for entry in entries]


class C8ExactSpeculativeResidentStageContractTest(unittest.TestCase):
    def test_active_source_seam_keeps_official_score_and_topk_ownership(self) -> None:
        source = CONTRACT["source_derived_from"]
        self.assertEqual(source["operator"], "VllmQuantLightningIndexer")
        seam = source["score_seam"]
        self.assertEqual(seam["function"], "QLIVector::ProcessVec1")
        self.assertEqual(seam["after"], "vector1::BatchMulWeightAndReduceSum")
        self.assertEqual(seam["before"], "DataCopyPad(scoreGm")
        self.assertEqual(
            source["topk_owner"],
            "official ProcessTopK selection and original sparse-index writeback stay unchanged; "
            "P9 invokes its terminal direct-resident finalizer after the original "
            "V->MTE3 completion and before that unchanged sparse-index writeback, "
            "without consuming TOPK_MTE3_V",
        )
        self.assertIn("active atomic ABI", CONTRACT["abi_status"])

        process_vec = OFFICIAL_VECTOR.index("QLIVector<QLIT>::ProcessVec1")
        process_topk = OFFICIAL_VECTOR.index("QLIVector<QLIT>::ProcessTopK")
        score_reduce = OFFICIAL_VECTOR.index(
            "vector1::BatchMulWeightAndReduceSum", process_vec
        )
        score_publish = OFFICIAL_VECTOR.index("DataCopyPad(scoreGm", score_reduce)
        self.assertLess(process_vec, score_reduce)
        self.assertLess(score_reduce, score_publish)
        self.assertLess(score_publish, process_topk)

    def test_exact_21_input_6_output_graph_abi_and_aliases(self) -> None:
        abi = CONTRACT["operator_abi"]
        self.assertEqual(abi["input_count"], 21)
        self.assertEqual(abi["output_count"], 6)
        self.assertEqual(contract_schema(abi["inputs"]), OFFICIAL_INPUTS + P9_INPUTS)
        self.assertEqual(contract_schema(abi["outputs"]), OUTPUTS)
        self.assertEqual(op_schema(OPS["input_desc"]), OFFICIAL_INPUTS + P9_INPUTS)
        self.assertEqual(op_schema(OPS["output_desc"]), OUTPUTS)
        self.assertEqual(abi["outputs"][4]["alias_of_input"], 17)
        self.assertEqual(abi["outputs"][5]["alias_of_input"], 18)
        names = [entry["name"] for entry in abi["inputs"] + abi["outputs"]]
        self.assertNotIn("resident_packed_output", names)
        self.assertIn("not an operator input or output", abi["resident_payload"])

    def test_direct_resident_candidate_layout_has_no_private_indirection(self) -> None:
        self.assertEqual(CONTRACT["target"], "A5C8ExactSpeculativeResidentGet")
        self.assertEqual(
            CONTRACT["candidate_seed"]["words"],
            ["row", "source", "kv_object", "resident_slot"],
        )
        self.assertEqual(
            CONTRACT["candidate_seed"]["cross_wave_slot_invariant"],
            "CandidateCompact never assigns one resident_slot to two initial "
            "waves; Stage-B drains every initial wave before it may reuse a "
            "non-winner entry in the same invocation for exact recall",
        )
        resource = CONTRACT["resource_model"]
        layout = resource["canonical_resident_layout"]
        self.assertEqual(layout["input"], "canonical_resident_layout")
        self.assertEqual(layout["shape"], "[first_slot, slots_per_row, row_stride, row_count]")
        self.assertEqual(layout["slot_count"], 2048)
        self.assertIn("widened to uint32", layout["meaning"])
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
        for forbidden in (
            "private_",
            "arena_",
            "quarantine",
            "forward_control",
            "replay",
            "fixed_get_core_partition",
            "31+1",
        ):
            self.assertNotIn(forbidden, KERNEL_CONTRACT.lower())

    def test_score_seam_is_bounded_and_queue_ownership_is_dynamic(self) -> None:
        for required in (
            "A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM 1",
            "P9ObserveScoreSeam",
            "P9PostAtScoreBoundary",
            "kCandidateMicrobatchCapacity",
            "kCandidateMicrobatchCount",
        ):
            self.assertIn(required, SCORE_SEAM + SCORE_HOOK)
        self.assertIn("kCandidateMicrobatchCapacity = 64U", KERNEL_CONTRACT)
        self.assertIn("kCandidateMicrobatchCount = 2U", KERNEL_CONTRACT)
        self.assertNotIn("TBuf<", SCORE_HOOK)
        for forbidden in ("31+1", "fixed_get_core_partition"):
            self.assertNotIn(forbidden, SCORE_HOOK.lower())

        for required in (
            "ReserveCandidateResidentSlotSimt",
            "CommitCandidateResidentSlotSimt",
            "RollbackCandidateResidentSlotSimt",
            "TryAcquireKvafLayerResidentQueueFromAicore",
            "AppendKvafLayerResidentRequest",
            "PublishKvafLayerResidentDescriptor",
            "BeginKvafLayerResidentBatchGet",
            "TryRetireKvafLayerResidentBatchGet",
            "FinishKvafLayerResidentBatchGet",
            "ReturnKvafLayerResidentQueueFromAicore",
            "ReturnKvafLayerResidentUnpostedQueueFromAicore",
            "DrainAllBeforeReturn",
            "TryPostFixedRecallSeeds",
        ):
            self.assertIn(required, POST_EXECUTOR)
        for forbidden in (
            "TBuf<",
            "private_arena",
            "forward_control",
            "replay",
            "epoch",
            "quarantine",
            "fixed_get_core",
            "31+1",
        ):
            self.assertNotIn(forbidden, POST_EXECUTOR.lower())
        self.assertLess(
            POST_EXECUTOR.index("PublishKvafLayerResidentDescriptor"),
            POST_EXECUTOR.index("BeginKvafLayerResidentBatchGet(control_, lease"),
        )
        self.assertLess(
            POST_EXECUTOR.index("BeginKvafLayerResidentBatchGet(control_, lease"),
            POST_EXECUTOR.index("CommitViewClaims(view)"),
        )

    def test_stage_b_only_exposes_terminal_normal_resident_slots(self) -> None:
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
            stage_b["completion_condition"], "all_initial_and_recall_posts_terminal"
        )
        self.assertEqual(
            stage_b["qsfa_contract"],
            "QSFA consumes only final attention_slots after this condition holds",
        )
        self.assertEqual(stage_b["outputs"], ["attention_slots", "terminal_post_stats"])
        for required in (
            "FinalizeExactTopKRow",
            "DeferAllReadyPostViews",
            "DrainAllBeforeReturn",
            "TryPostFixedRecallSeeds",
            "candidateSourceToSlot_",
            "persistentObjectSlots_",
            "attentionSlots_",
            "every selected payload is terminal",
        ):
            self.assertIn(required, FINALIZER)
        self.assertNotIn("TBuf<", FINALIZER)
        for forbidden in (
            "private_arena",
            "forward_control",
            "replay",
            "epoch",
            "quarantine",
            "fixed_get_core",
            "31+1",
        ):
            self.assertNotIn(forbidden, FINALIZER.lower())

    def test_overlay_is_atomic_active_abi_not_a_disabled_or_legacy_route(self) -> None:
        # The source-derived transform must wire every layer of the ABI in one
        # pass.  Running only a seam or only a wrapper creates a mixed binary.
        for required in (
            "inject_kernel_wrapper_abi",
            "inject_preload_abi",
            "inject_vector_runtime",
            "inject_score_seam",
            "seam=active_abi",
            "runtimeEnable",
            "rowsPerEntry",
            "candidateSourceToSlotRef",
            "candidateSlotCursorRef",
        ):
            self.assertIn(required, OVERLAY_APPLIER)
        self.assertNotIn("seam=compile_disabled", OVERLAY_APPLIER)
        self.assertIn("A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER 1", STAGE_B_TAIL)
        self.assertNotIn("compile-disabled", SCORE_SEAM)
        self.assertNotIn("compile-disabled", STAGE_B_TAIL)
        # The active tail may not take the official TOPK event token a second
        # time; its legal boundary must be established by the overlay itself.
        self.assertNotIn(
            "WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT)", OVERLAY_APPLIER
        )
        for forbidden in ("P7", "P8", "31+1", "fixed physical get core"):
            self.assertNotIn(forbidden.lower(), OVERLAY_APPLIER.lower())

    def test_builder_is_fresh_and_provenance_gated(self) -> None:
        for required in (
            "P9_PROVENANCE",
            "fresh_package_root_exists",
            "P9_SCORE_HOOK_HEADER",
            "P9_POST_EXECUTOR_HEADER",
            "P9_FINALIZER_HEADER",
            "P9_STAGE_B_FINALIZER_HEADER",
            "P9_KVAF_SOURCE_CONTRACT_PASS",
            "copy_tree_without_repository_cmake",
        ):
            self.assertIn(required, P9_BUILD_SCRIPT)
        self.assertNotIn("seam=compile_disabled", P9_BUILD_SCRIPT)

    def test_non_goals_remain_explicit(self) -> None:
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
