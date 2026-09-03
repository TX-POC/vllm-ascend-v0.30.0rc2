"""Static contract for P10's device-side score-receipt Stage B.

It intentionally proves source boundaries before a CANN registration route is
allowed.  This is not a host-path functional test.
"""

from __future__ import annotations

import unittest
import json
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[1]
ROOT = (
    OPS_ROOT
    / "a5_dsa_c8_exact_speculative_resident"
    / "stage_b_score_receipt_forward_resident"
)


class P10StageBSourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = (ROOT / "README.md").read_text()
        cls.cmake = (ROOT / "CMakeLists.txt").read_text()
        cls.definition = (
            ROOT / "op_host" / "a5_p10_score_receipt_forward_resident_def.cpp"
        ).read_text()
        cls.abi = (
            ROOT / "op_kernel" / "a5_p10_score_receipt_forward_resident_abi.h"
        ).read_text()
        cls.tiling = (
            ROOT / "op_kernel" / "a5_p10_score_receipt_forward_resident_tiling.h"
        ).read_text()
        cls.kernel = (
            ROOT / "op_kernel" / "a5_p10_score_receipt_forward_resident_kernel.cpp"
        ).read_text()
        cls.ops_json = (ROOT / "ops.json").read_text()
        cls.ops = json.loads(cls.ops_json)[0]
        cls.builder = (ROOT / "build_stage_b_opp.sh").read_text()
        cls.include_shim = (
            ROOT
            / "op_kernel"
            / "a5_p10_score_receipt_forward_resident_kvaf_include.h"
        ).read_text()
        cls.stage_a_abi = (
            OPS_ROOT
            / "a5_dsa_c8_exact_speculative_resident"
            / "stage_a_w4a4c8_score_receipt"
            / "op_kernel"
            / "a5_p10_w4a4c8_score_receipt_abi.h"
        ).read_text()

    def test_graph_abi_exposes_all_forward_mutation_boundaries(self):
        for name in (
            "score_surface", "receipt_ring", "receipt_state", "forward_control",
            "active_row_pool_entries", "kv_object_map",
            "kv_object_slots_snapshot", "kv_object_to_forward_slot",
            "source_to_forward_slot", "speculative_slot_band",
            "speculative_slot_cursor", "recall_slot_band",
            "canonical_sparse_indices", "canonical_score_source_lens",
            "kvaf_forward_resident_control", "queue_leases",
            "resident_payload", "resident_ready", "attention_slots_buffer",
            "selected_slot_lengths_buffer",
        ):
            self.assertIn(f'Input("{name}")', self.definition)
        self.assertIn('Input("resident_row_to_qsfa_slot")', self.definition)
        for name in (
            "receipt_state_alias", "forward_control_alias",
            "kv_object_to_forward_slot_alias",
            "source_to_forward_slot_alias", "speculative_slot_cursor_alias",
            "queue_leases_alias", "resident_payload_alias",
            "resident_ready_alias", "attention_slots", "selected_slot_lengths",
            "resident_seq_lengths",
        ):
            self.assertIn(f'Output("{name}")', self.definition)
        self.assertNotIn('Input("source_to_kv_object")', self.definition)
        self.assertNotIn('Input("persistent_resident_slots")', self.definition)
        self.assertIn("DT_UINT16", self.definition)
        self.assertIn("DT_INT64", self.definition)
        self.assertIn("DT_UINT8", self.definition)
        self.assertIn('"resident_row_to_qsfa_slot"', self.ops_json)
        self.assertEqual(
            self.ops["input_desc"][18]["name"], "resident_row_to_qsfa_slot"
        )
        self.assertEqual(len(self.ops["input_desc"]), 21)
        self.assertEqual(len(self.ops["output_desc"]), 11)
        self.assertEqual(
            [entry["name"] for entry in self.ops["attr"]],
            [
                "num_layers", "layer_id", "executor_worker_count",
                "executor_worker_limit", "receipt_poll_quantum",
                "bounded_drain_polls",
            ],
        )

    def test_mapping_materializes_row_pool_object_before_slot_claim(self):
        for token in (
            "activeRowPoolEntries_", "kvObjectMap_", "kvObjectSlotsSnapshot_",
            "kvObjectToForwardSlot_", "ResolveActiveRowMapping",
            "object_offset", "ClaimForwardSlotFromSimt", "object_local",
            "materialized_object", "global_object", "numLayers", "layerId",
            "qcap",
        ):
            self.assertIn(token, self.kernel)
        self.assertIn("cold-bound witness", self.kernel)
        self.assertIn("exact token object used by KVAF Append", self.kernel)
        self.assertIn("O_token = ((P * num_layers + layer_id) * qcap * rows_per_object) + S", self.abi)
        self.assertIn("table_index = O_token / 128 = P*num_layers*qcap + layer_id*qcap + (S>>7)", self.abi)
        self.assertIn("byte_offset = (O_token % 128) * 656 = (S&127) * 656", self.abi)
        self.assertIn("local_object", self.kernel)
        self.assertIn("mapping.object_local", self.kernel)
        self.assertNotIn("source_to_kv_object", self.kernel)
        self.assertNotIn("persistent_resident_slots", self.kernel)

    def test_transport_uses_only_the_verified_forward_resident_abi(self):
        for name in (
            "P10_STAGE_B_KVAF_FORWARD_RESIDENT_DEVICE_HEADER",
            "BeginKvafForwardResidentBatchGet",
            "TryRetireKvafForwardResidentBatchGet",
            "FinishKvafForwardResidentBatchGet",
            "TryAcquireKvafForwardResidentQueueFromAicore",
            "ReturnKvafForwardResidentQueueFromAicore",
        ):
            self.assertIn(name, self.kernel)
        for forbidden in (
            "execute_batch_get", "KvafAdapter", "DirectHccp", "P8", "P9",
            "fixed_get_core", "physical_aiv",
        ):
            self.assertNotIn(forbidden, self.kernel)

    def test_slots_are_fixed_bands_not_queue_private_or_recycled(self):
        self.assertIn("A5P10ResidentSlotBand", self.abi)
        self.assertIn("speculative_slot_band", self.abi)
        self.assertIn("recall_slot_band", self.abi)
        self.assertIn("never released or remapped", self.abi)
        self.assertIn("fixed resident rows are never returned to any queue", self.kernel)
        self.assertIn("A queue lease is dynamic", self.readme)
        self.assertIn("AIV-core", self.readme)

    def test_companion_executor_polls_receipts_without_a_stage_a_completion_edge(self):
        for token in (
            "ProducerSealed", "ServiceOwnedReceipts", "A5_P10_STAGE_B_PRODUCER_SEALED",
            "A5P10ProducerSealState::SEALED",
            "while (!producer_sealed || !owned_receipts_done)",
            "boundedDrainPolls", "A5_P10_STAGE_B_FORWARD_STATE",
            "receiptCapacity_ = tiling_->receiptCapacity",
            "state_after != static_cast<int32_t>(A5P10ReceiptState::FREE)",
        ):
            self.assertIn(token, self.kernel)
        self.assertNotIn("A5_P10_STAGE_B_RECEIPT_COUNT", self.kernel)
        self.assertNotIn("receiptCount_", self.kernel)
        self.assertIn("There is deliberately no Stage-A -> Stage-B", self.kernel)
        self.assertIn("graph completion edge", self.kernel)
        self.assertIn("not* connected to Stage A by a normal", self.readme)
        self.assertIn("Stage-B forward_state ->", self.abi)
        self.assertIn("QSFA after the bounded terminal drain", self.abi)
        self.assertIn("Before either C8 or the companion begins", self.readme)
        self.assertIn("must not clear or reinitialize receipt state", self.readme)
        self.assertIn("Overlap witness", self.kernel)
        self.assertIn("ServiceOwnedReceipts(producer_sealed)", self.kernel)

    def test_preseal_overlap_witness_is_diagnostic_and_shared_with_stage_a(self):
        for token in (
            "A5_P10_FORWARD_CONTROL_PRODUCER_SEAL_SEQUENCE_WORD",
            "A5_P10_FORWARD_CONTROL_FIRST_PRESEAL_READY_CLAIM_WORD",
            "A5_P10_STAGE_B_PRODUCER_SEAL_SEQUENCE",
            "A5_P10_STAGE_B_FIRST_PRESEAL_READY_CLAIM_SEQUENCE",
            "observed_seal", "first-preseal",
        ):
            self.assertIn(token, self.kernel + self.abi + self.stage_a_abi)
        self.assertIn("final static receipt sequence", self.stage_a_abi)
        self.assertIn("never used for scheduling decisions", self.stage_a_abi)
        self.assertIn("first_preseal_ready_claim_sequence", self.readme)

    def test_qsfa_abi_keeps_the_full_prefix_and_tail_contract(self):
        for token in (
            "[B,1,K+tail]", "attention_slots_buffer",
            "selected_slot_lengths_buffer", "selected_slot_lengths",
            "resident_seq_lengths",
        ):
            self.assertIn(token, self.abi)
        self.assertIn("replaces only [0,K)", self.abi)
        self.assertIn("[K,K+tail)", self.kernel)
        self.assertIn("selected length is retained verbatim", self.kernel)
        self.assertIn("QSFA", self.readme)
        self.assertIn("resident_row_to_qsfa_slot", self.abi)
        self.assertIn("residentRowToQsfaSlot_", self.kernel)
        self.assertIn("ResolveQsfaSlot", self.kernel)
        self.assertIn("physical resident row number", self.readme)

    def test_kvaf_geometry_is_explicit_and_not_an_invented_remap(self):
        for token in ("numLayers", "layerId", "qcap"):
            self.assertIn(token, self.tiling)
        self.assertIn("kKvafRowsPerObject = 128U", self.kernel)
        self.assertIn("expected_source_capacity", self.kernel)
        self.assertIn("rowsPerObject_ != kKvafRowsPerObject", self.kernel)
        self.assertIn("materialized_object != static_cast<int32_t>(global_object)", self.kernel)
        self.assertIn("mapping.kv_object", self.kernel)
        self.assertIn("mapping.object_local", self.kernel)
        self.assertIn("AppendKvafForwardResidentRequest", self.kernel)

    def test_executor_uses_bounded_logical_polling_not_a_fixed_core_partition(self):
        for token in (
            "executorWorkerCount", "executorWorkerLimit", "receiptPollQuantum",
        ):
            self.assertIn(token, self.tiling)
        self.assertIn("tiling_->executorWorkerCount > tiling_->executorWorkerLimit", self.kernel)
        self.assertIn("PollReceiptQuantumBoundary", self.kernel)
        self.assertIn("poll_turn == tiling_->receiptPollQuantum", self.kernel)
        self.assertIn("not a physical-AIV selection", self.readme)
        self.assertIn("must\nnot mean \"launch on every C8 AIV\"", self.readme)
        self.assertNotIn("physical_aiv", self.kernel)
        self.assertNotIn("fixed_get_core", self.kernel)
        self.assertIn("strictly below available AIV capacity", self.definition)
        self.assertIn("static_cast<uint64_t>(*workerCount) >= aivCount", self.definition)
        self.assertIn("static_cast<uint64_t>(*workerLimit) >= aivCount", self.definition)

    def test_receipt_terminal_transition_is_atomic_after_claim(self):
        for token in (
            "FinishClaimedReceiptFromSimt", "FinishClaimedReceiptSimtService",
            "FinishClaimedReceipt(receipt)", "A5P10ReceiptState::CLAIMED",
            "A5P10ReceiptState::DONE",
        ):
            self.assertIn(token, self.kernel)
        self.assertNotIn("receiptState_[receipt] =", self.kernel)

    def test_kernel_is_a_real_mixed_device_data_plane_not_a_wrapper(self):
        for token in (
            "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)",
            "a5_p10_score_receipt_forward_resident",
            "ConsumeExactScoreReceipt", "BeginForwardResidentMicrobatch",
            "DrainForwardResidentMicrobatch", "FinalizeAssignedRows",
            "EnsureFinalWinnerReady", "A5_P10_STAGE_B_MICROBATCH_CAPACITY",
            "A5_P10_STAGE_B_LOCAL_WAVE_CAPACITY",
        ):
            self.assertIn(token, self.kernel)

    def test_claiming_race_waits_for_durable_object_slot_not_a_reuse(self):
        self.assertIn("ResolveConcurrentForwardSlot", self.kernel)
        self.assertIn("A5_P10_STAGE_B_CLAIMING_SLOT", self.kernel)
        self.assertIn("does not retry/recycle a slot", self.kernel)
        self.assertIn("if (!ResolveConcurrentForwardSlot", self.kernel)

    def test_host_has_real_tiler_infer_and_distinct_registration(self):
        for token in (
            "TilingA5P10ScoreReceiptForwardResident",
            "InferA5P10ScoreReceiptForwardResidentShape",
            "InferA5P10ScoreReceiptForwardResidentDataType",
            "IMPL_OP_INFERSHAPE(A5P10ScoreReceiptForwardResident)",
            "SetTiling(optiling::TilingA5P10ScoreReceiptForwardResident)",
            "OP_ADD(A5P10ScoreReceiptForwardResident)",
            "CalcTschBlockDim",
            "executor_worker_count",
            "resident_row_to_qsfa_slot",
        ):
            self.assertIn(token, self.definition)
        self.assertEqual(
            self.definition.count("OP_ADD(A5P10ScoreReceiptForwardResident)"), 1
        )

    def test_isolated_builder_stages_exact_closure_and_unique_registration(self):
        for token in (
            "P10_STAGE_SOURCE_ROOT", "P10_STAGE_VLLM_SOURCE_ROOT",
            "P10_STAGE_KVAF_SOURCE_ROOT", "P10_STAGE_EXPECTED_VLLM_SHA",
            "P10_STAGE_EXPECTED_KVAF_SHA", "P10_STAGE_PROVENANCE_FILE",
            "P10_STAGE_CANN_3RD_LIB_PATH", "msopgen gen",
            "duplicate_or_missing_op_registration",
            "P10_STAGE_PROVENANCE", "kvaf_forward_resident_get_device.h",
            "execute_batch_get_aiv.asc", "batch_get_abi.h", "direct_hccp_abi.h",
            "a5_p10_w4a4c8_score_receipt_abi.h",
            "generated_kernel_stub_missing",
        ):
            self.assertIn(token, self.builder)
        self.assertIn("P10_STAGE_B_KVAF_FORWARD_RESIDENT_DEVICE_HEADER",
                      self.include_shim)
        self.assertNotIn("P8", self.builder)
        self.assertNotIn("P9", self.builder)

    def test_source_cmake_is_manifest_friendly_not_a_fatal_placeholder(self):
        self.assertIn("build_stage_b_opp.sh", self.cmake)
        self.assertIn("add_subdirectory(op_host)", self.cmake)
        self.assertNotIn("FATAL_ERROR", self.cmake)
        self.assertIn("add_op_to_compiled_list", (ROOT / "op_host" / "CMakeLists.txt").read_text())


if __name__ == "__main__":
    unittest.main()
