#!/usr/bin/env python3
"""Static source gate for the P9 direct-resident C8 target.

This test prevents the real CANN target from inheriting old P7/P8 private
transport or silently changing the official C8 score/Top-K seam.
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
CENTRAL_BUILD_SCRIPT = (OPS_ROOT / "build_nanovllm_ops.sh").read_text(encoding="utf-8")
P9_BUILD_SCRIPT = (
    TARGET_ROOT / "build_p9_exact_resident_opp.sh"
).read_text(encoding="utf-8")


class C8ExactSpeculativeResidentStageContractTest(unittest.TestCase):
    def test_source_is_derived_from_the_official_score_seam(self) -> None:
        source = CONTRACT["source_derived_from"]
        self.assertEqual(source["operator"], "VllmQuantLightningIndexer")
        seam = source["score_seam"]
        self.assertEqual(seam["function"], "QLIVector::ProcessVec1")
        self.assertEqual(seam["after"], "vector1::BatchMulWeightAndReduceSum")
        self.assertEqual(seam["before"], "DataCopyPad(scoreGm")
        self.assertEqual(
            source["topk_owner"],
            "official ProcessTopK selection and original writeback are unchanged; "
            "P9 inserts only compile-disabled tails after each original writeback",
        )
        self.assertIn("target ABI", CONTRACT["abi_status"])

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
            "waves; Stage-B drains every initial wave before it may reuse a "
            "non-winner slot for exact recall",
        )
        self.assertEqual(
            CONTRACT["candidate_seed"]["maps"],
            {
                "candidate_source_to_slot": "int32[R, S], initialized to -1; "
                "written only after the fully published descriptor is accepted "
                "by PostOnly",
                "candidate_slot_cursor": "int32[R], invocation-local monotonic "
                "admission cursor; it only prevents duplicate initial posts and is "
                "not used for Stage-B reuse",
            },
        )
        resource = CONTRACT["resource_model"]
        fixed = resource["canonical_resident_layout"]
        self.assertEqual(fixed["input"], "canonical_resident_layout")
        self.assertEqual(
            fixed["shape"],
            "[first_slot, slots_per_row, row_stride, row_count]",
        )
        self.assertIn("widened to uint32", fixed["meaning"])
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

    def test_score_seam_uses_existing_scratch_and_never_mints_a_get_core(self) -> None:
        # Capacity values have one source of truth in the shared contract;
        # the hook imports them rather than duplicating magic numbers.
        self.assertIn("kCandidateMicrobatchCapacity", SCORE_HOOK)
        self.assertIn("kCandidateMicrobatchCount", SCORE_HOOK)
        self.assertIn("kCandidateMicrobatchCapacity = 64U", KERNEL_CONTRACT)
        self.assertIn("kCandidateMicrobatchCount = 2U", KERNEL_CONTRACT)
        self.assertIn("candidateSlotCursor", SCORE_HOOK)
        self.assertIn("rowCount", SCORE_HOOK)
        self.assertIn("BatchMulWeightAndReduceSum", SCORE_HOOK)
        self.assertIn("DataCopyPad(scoreGm", SCORE_HOOK)
        self.assertNotIn("TBuf<", SCORE_HOOK)
        # Do not make comments a source of false failures: the header may
        # explicitly explain why physical-core partitioning is absent.  The
        # stable implementation-level bans below are the actual contract.
        for forbidden in ("31+1", "fixed_get_core_partition"):
            self.assertNotIn(forbidden.lower(), SCORE_HOOK.lower())

    def test_dynamic_post_executor_has_only_invocation_local_queue_ownership(self) -> None:
        # A fixed resident slot does not need a ready/release/ABA protocol.
        # The only dynamic resource is the temporary Direct-HCCP queue.
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
        ):
            self.assertNotIn(forbidden, POST_EXECUTOR.lower())

        # Maps are committed only after the real Begin path; the only early
        # mutation is a short -2 reservation that rollback clears if no
        # doorbell happens.
        self.assertLess(
            POST_EXECUTOR.index("PublishKvafLayerResidentDescriptor"),
            POST_EXECUTOR.index("BeginKvafLayerResidentBatchGet(control_, lease"),
        )
        self.assertLess(
            POST_EXECUTOR.index("BeginKvafLayerResidentBatchGet(control_, lease"),
            POST_EXECUTOR.index("CommitViewClaims(view)"),
        )
        self.assertIn("kCandidateSourceClaiming", POST_EXECUTOR)
        # Descriptor publication owns the DDR barrier in the KVAF ABI.  The
        # P9 executor must call that boundary rather than reimplement it.
        self.assertIn("PublishKvafLayerResidentDescriptor", POST_EXECUTOR)
        self.assertIn("has_wave_ = true;", POST_EXECUTOR)
        self.assertIn("static_cast<event_t>(EVENT_ID0)", POST_EXECUTOR)
        # AICore->SIMT calls must carry scalar layout fields, not an aggregate
        # ABI that CANN may lower differently from normal C++.
        self.assertIn("resident_layout.firstSlot", POST_EXECUTOR)
        self.assertNotIn(
            "source_capacity, resident_layout,\n        result",
            POST_EXECUTOR,
        )

    def test_device_closure_preserves_aicore_domains_and_ub_address_spaces(self) -> None:
        """The compile-disabled seam still parses every P9 device helper."""
        for result_type, helper in (
            ("bool", "IsCanonicalLayout"),
            ("int64_t", "CanonicalRowFirstSlot"),
            ("bool", "IsSlotInCanonicalRow"),
        ):
            self.assertRegex(
                KERNEL_CONTRACT,
                rf"__aicore__\s+constexpr\s+{result_type}\s+{helper}\s*\(",
            )
        # `CandidatePostView.seeds` and Stage-B recall seeds live in UB.  A
        # normal C++ reference silently changes address space under OPC; use
        # explicit UB pointers or scalar register loads instead.
        self.assertNotRegex(
            POST_EXECUTOR, r"(?:const\s+)?CandidateSeed\s*&"
        )
        self.assertIn("__ubuf__ CandidateSeed* const seed", POST_EXECUTOR)
        self.assertIn("__ubuf__ const CandidateSeed* const seed", POST_EXECUTOR)
        self.assertIn("seed_kv_object", POST_EXECUTOR)

    def test_post_executor_resolves_the_staged_kvaf_device_closure_relatively(self) -> None:
        """OPC precompile cannot depend on an implicit op_kernel include path."""
        relative_include = (
            '../kvaf_layer_resident/kvaf_layer_resident_get_device.h'
        )
        self.assertIn(f'#include "{relative_include}"', POST_EXECUTOR)
        # The builder owns the exact staged layout, so assert both the source
        # requirement and its fail-closed file-resolution check here.
        self.assertIn("P9_KVAF_RELATIVE_INCLUDE", P9_BUILD_SCRIPT)
        self.assertIn(
            "p9_post_executor_kvaf_relative_include_unresolvable",
            P9_BUILD_SCRIPT,
        )

    def test_existing_c8_invalid_scratch_has_a_complete_fixed_layout(self) -> None:
        for token in (
            "kExistingC8InvalidScratchBytes",
            "kCandidateSeedScratchBytes",
            "kQueueLeaseReceiptScratchBytes = 16U",
            "kCandidateReservationScratchBytes = 16U",
            "kExactRecallWaveCapacity = 256U",
            "kExactRecallSeedScratchBytes == 4096U",
            "kStageBUsedSlotBitmapBytes",
            "kStageBQueueLeaseReceiptScratchBytes = 16U",
            "kStageBReservationScratchBytes = 16U",
            "kP9ScratchBytes == 4384U",
            "kP9ScratchBytes <= kExistingC8InvalidScratchBytes",
        ):
            self.assertIn(token, KERNEL_CONTRACT)

    def test_stage_b_finalizer_is_a_terminal_exact_recall_tail(self) -> None:
        """Stage B consumes final local Top-K; it never owns C8 math."""
        for required in (
            "FinalizeExactTopKRow",
            "DeferAllReadyPostViews",
            "DrainAllBeforeReturn",
            "TryPostFixedRecallSeeds",
            "kExactRecallWaveCapacity",
            "candidateSourceToSlot_",
            "persistentObjectSlots_",
            "attentionSlots_",
            "kCandidateSourceClaiming",
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
        ):
            self.assertNotIn(forbidden, FINALIZER.lower())
        # The output itself is only consumed after operator completion.  The
        # finalizer may record a recall destination before its CQ terminal,
        # but it must drain before returning success.
        self.assertLess(
            FINALIZER.index("TryPostFixedRecallSeeds"),
            FINALIZER.rindex("DrainAllBeforeReturn()"),
        )

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

    def test_target_has_a_dedicated_fresh_builder_but_no_legacy_route(self) -> None:
        """P9 has one fresh source-derived builder, never a legacy route."""
        self.assertNotIn("A5C8ExactSpeculativeResidentGet", CENTRAL_BUILD_SCRIPT)
        self.assertIn("P9_PROVENANCE", P9_BUILD_SCRIPT)
        self.assertIn("fresh_package_root_exists", P9_BUILD_SCRIPT)
        self.assertIn("legacy_p7_p8_streaming_include", P9_BUILD_SCRIPT)
        self.assertIn("P9_SCORE_HOOK_HEADER", P9_BUILD_SCRIPT)
        self.assertIn("P9_POST_EXECUTOR_HEADER", P9_BUILD_SCRIPT)
        self.assertIn("P9_FINALIZER_HEADER", P9_BUILD_SCRIPT)
        self.assertIn("p9_stage_b_finalizer_include_missing", P9_BUILD_SCRIPT)
        # P9 uses msopgen's standalone CMake scaffold.  Copying the official
        # C8 op_host directory over it revives a monorepo-only macro
        # (add_op_to_compiled_list) and fails before any P9 code compiles.
        # The generated OP_ADD stub is removed only after source files are
        # staged without CMakeLists.txt, leaving one source-derived OpDef.
        self.assertIn("P9_HOST_STAGING", P9_BUILD_SCRIPT)
        self.assertIn("p9_legacy_add_op_to_compiled_list_leaked_after_staging", P9_BUILD_SCRIPT)
        self.assertIn("p9_generated_op_add_stub_count", P9_BUILD_SCRIPT)
        self.assertIn("copy_tree_without_repository_cmake", P9_BUILD_SCRIPT)
        self.assertIn("p9_source_derived_op_add_registration_count", P9_BUILD_SCRIPT)
        self.assertIn("p9_source_derived_op_add_registration_path", P9_BUILD_SCRIPT)
        self.assertNotIn(
            'cp -a "${STAGED_C8_ROOT}/op_host/." "${GENERATED_ROOT}/op_host/"',
            P9_BUILD_SCRIPT,
        )
        self.assertIn("build_p9_exact_resident_opp.sh", (TARGET_ROOT / "README.md").read_text())
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
