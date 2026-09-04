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

    def test_preseal_receipts_do_not_depend_on_a_post_topk_length(self):
        """A READY score receipt is the pre-seal source-range authority.

        The final canonical source length is written only after C8 completes
        Top-K; using it to filter an earlier receipt silently removes the
        speculative side of the target pipeline.
        """
        self.assertIn("require_canonical_source_length", self.kernel)
        self.assertIn("ResolveActiveRowMapping(row, source, &mapping, false)", self.kernel)
        self.assertIn("ResolveActiveRowMapping(row, source, &mapping, true)", self.kernel)
        self.assertIn("post-Top-K output", self.kernel)

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

    def test_begin_returns_only_a_scalar_diagnostic_before_wave_materialization(self):
        """Keep all StageBWave mutation outside the large inline Begin closure.

        A successful PostOnly transaction is copied into the persistent wave
        only in StartMicrobatch.  An unsuccessful Begin returns its exact
        scalar reason directly to the caller; neither its error nor its local
        KVAF record depends on a writeback to a nested executor member.
        """
        begin = self.kernel.split("BeginForwardResidentMicrobatch(", 1)[1].split(
            "// One TryRetire poll", 1
        )[0]
        self.assertIn("inline uint32_t BeginForwardResidentMicrobatch", self.kernel)
        self.assertIn("KvafForwardResidentBatchGetWave* begin_wave", begin)
        self.assertIn("return begin_reason;", begin)
        self.assertNotIn("StageBWave* output", begin)
        self.assertNotIn("output->", begin)
        self.assertNotIn("descriptor, &output->kvaf_wave", self.kernel)

        self.assertIn("KvafForwardResidentBatchGetWave begin_wave{}", self.kernel)
        self.assertIn("const uint32_t begin_reason = BeginForwardResidentMicrobatch(", self.kernel)
        self.assertIn("*failure_reason = begin_reason;", self.kernel)
        self.assertIn(
            "MaterializeForwardResidentWaveForDrain(wave, &begin_wave)",
            self.kernel,
        )
        self.assertIn("wave->posted = 1U;", self.kernel)

        materialize = self.kernel.split(
            "MaterializeForwardResidentWaveForDrain("
        )[1].split("BeginFailureReasonFromKvafTransaction(")[0]
        self.assertNotIn("= *source", materialize)
        for field in (
            "lease.queue_id", "queue_args.address_table",
            "queue_args.output", "transaction.active_count",
            "transaction.result", "transaction.state",
            "transaction.finalized",
        ):
            self.assertIn(field, materialize)

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

    def test_cq_terminal_publishes_ready_and_qsfa_prefix_outside_aicore_dcache(self):
        """The local CQ proof cannot bypass the durable graph-visible state.

        A CQ terminal makes a row READY locally, but the next consumer may be
        SIMT, another AIV, or QSFA.  Therefore the IN_FLIGHT -> READY and
        attention-prefix publications must use uncached GM writes before the
        final COMPLETE release flag.  This specifically covers the first
        recall wave after the 64-entry speculative band.
        """
        for token in (
            "target_slot_ready", "completed_target_slot",
            "successful CQ terminal", "local_terminal_ready",
            "DrainLocalWavesForTarget(\n          true, seed.resident_slot, &local_terminal_ready,",
            "resident_ready=READY", "WriteGmByPassDCache<int32_t>",
            "A5P10ResidentReadyState::IN_FLIGHT",
            "A5P10ResidentReadyState::READY",
            "A5_P10_STAGE_B_FORWARD_STATE_COMPLETE",
            "DataSyncBarrier<MemDsbT::DDR>()",
        ):
            self.assertIn(token, self.kernel)
        self.assertNotIn(
            "residentReady_[wave->seeds[index].resident_slot] =",
            self.kernel,
        )
        self.assertNotIn(
            "residentReady_[wave->seeds[seed].resident_slot] =",
            self.kernel,
        )
        self.assertNotIn("resident_ready[selected] =", self.kernel)
        self.assertIn("ForwardPrepare initializes the full fixed resident table", self.kernel)

    def test_kernel_is_an_aiv_only_dynamic_device_executor_not_a_wrapper(self):
        for token in (
            "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)",
            "a5_p10_score_receipt_forward_resident",
            "ConsumeExactScoreReceipt", "BeginForwardResidentMicrobatch",
            "DrainForwardResidentMicrobatch", "FinalizeAssignedRows",
            "EnsureFinalWinnerReady", "A5_P10_STAGE_B_MICROBATCH_CAPACITY",
            "A5_P10_STAGE_B_LOCAL_WAVE_CAPACITY",
        ):
            self.assertIn(token, self.kernel)
        self.assertNotIn("KERNEL_TYPE_MIX_AIC_1_2", self.kernel)
        self.assertIn("CalcTschBlockDim(", self.definition)
        self.assertIn("0U, aivCount", self.definition)

    def test_first_failure_diagnostic_is_forward_local_and_non_scheduling(self):
        for token in (
            "A5_P10_STAGE_B_DIAGNOSTIC", "INIT_COMMON", "INIT_GEOMETRY",
            "INIT_KVAF", "INIT_UB", "RECEIPT_TERMINAL", "FINALIZE",
            "DRAIN", "FINALIZE_DETAIL_BASE", "FINALIZE_FINAL_QSFA",
            "POST_DETAIL_BASE", "POST_BEGIN_INVALID_ARGS", "CQ_PENDING",
            "POST_BEGIN_STATE_IDLE", "POST_BEGIN_RESULT_UNKNOWN",
            "BeginFailureReasonFromKvafTransaction", "CqFailureReasonFromKvafResult",
            "PostDiagnostic", "failure_reason", "FinalizeDiagnostic",
            "RecordFirstFailure", "MarkRunning",
        ):
            self.assertIn(token, self.abi + self.kernel)
        self.assertIn("forward_control_gm.SetValue(1U, 0)", (
            OPS_ROOT / "a5_dsa_c8_exact_speculative_resident"
            / "p10_forward_bridge" / "op_kernel" / "a5_p10_forward_prepare.cpp"
        ).read_text())

    def test_claiming_race_waits_for_durable_object_slot_not_a_reuse(self):
        self.assertIn("ResolveConcurrentForwardSlot", self.kernel)
        self.assertIn("A5_P10_STAGE_B_CLAIMING_SLOT", self.kernel)
        self.assertIn("does not retry/recycle a slot", self.kernel)
        self.assertIn("if (!ResolveConcurrentForwardSlot", self.kernel)

    def test_aicore_metadata_access_respects_cann_address_spaces(self):
        """Prevent a recurrence of the Stage-B-only 950DT compile failure.

        The transport/lifecycle design is unchanged here: this gate only
        constrains how that design crosses GM and VECCALC UB address spaces.
        """
        self.assertIn("__gm__ int64_t* receipt_ring", self.kernel)
        self.assertNotIn("__gm__ const int64_t* receipt_ring", self.kernel)
        self.assertNotIn(
            "const A5P10ForwardResidentSeed& seed = seeds[index]",
            self.kernel,
        )
        self.assertNotIn(
            "localSeeds_[buffer_index][seed_count++] = {", self.kernel
        )
        self.assertNotIn("localSeeds_[buffer_index][0] = seed;", self.kernel)
        self.assertIn("AscendC::AtomicCas<uint32_t>", self.kernel)
        self.assertIn("reinterpret_cast<__gm__ uint32_t*>", self.kernel)
        self.assertNotIn(
            "asc_atomic_cas(sourceToForwardSlot_ + source_offset",
            self.kernel,
        )

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
