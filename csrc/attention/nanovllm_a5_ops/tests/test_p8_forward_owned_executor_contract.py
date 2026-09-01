#!/usr/bin/env python3
"""Source-closure gate for the isolated P8 forward-owned executor.

This test intentionally does not require Ascend hardware.  It proves the
non-negotiable lifecycle wiring before an OPP build is attempted:

* P8 is a standalone OPP and cannot import/compile C8 or P7 paths;
* Post persists a durable intent.  A proved ``POST_REJECTED`` is reclaimed
  only later by the DRAIN-owned no-post protocol, never in POST;
* Drain atomically claims GM POSTED -> DRAIN_CLAIMED *before* it reloads the
  record, then uses KVAF's nonblocking retire/Finish/private-publish/release
  protocol;
* QSFA terminal uses KVAF's CAS-verified private-slot release path;
* ForwardTerminal quarantines unresolved resources without freeing DMA that
  could still be in flight.

The real KVAF helper is intentionally supplied only by the selected KVAF
revision at OPP build time.  This test does not copy or mutate that repository.
"""

from __future__ import annotations

from pathlib import Path
import unittest


OPS_ROOT = Path(__file__).resolve().parents[1]
P8_ROOT = OPS_ROOT / "a5_p8_forward_owned_executor"
KERNEL = P8_ROOT / "op_kernel" / "a5_p8_forward_owned_executor.cpp"
HOST = P8_ROOT / "op_host" / "a5_p8_forward_owned_executor.cpp"
TILING = P8_ROOT / "op_kernel" / "a5_p8_forward_owned_executor_tiling.h"
ARENA_TAG = P8_ROOT / "op_kernel" / "a5_p8_arena_tag.h"
ADMISSION = P8_ROOT / "op_kernel" / "a5_p8_arena_admission_gate.cpp"
ADVANCE = P8_ROOT / "op_kernel" / "a5_p8_forward_control_advance.cpp"
SPEC = P8_ROOT / "ops.json"
BUILD = OPS_ROOT / "build_nanovllm_ops.sh"
RUNNER = OPS_ROOT / "tests" / "run_p8_forward_owned_executor_npu.sh"
REFERENCE_BUILD = OPS_ROOT / "a5_official_c8_reference" / "build_reference_opp.sh"


class P8ForwardOwnedExecutorContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kernel = KERNEL.read_text(encoding="utf-8")
        cls.host = HOST.read_text(encoding="utf-8")
        cls.tiling = TILING.read_text(encoding="utf-8")
        cls.arena_tag = ARENA_TAG.read_text(encoding="utf-8")
        cls.admission = ADMISSION.read_text(encoding="utf-8")
        cls.advance = ADVANCE.read_text(encoding="utf-8")
        cls.spec = SPEC.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.reference_build = REFERENCE_BUILD.read_text(encoding="utf-8")

    def test_isolated_operator_identity(self) -> None:
        self.assertIn('"op": "A5P8ForwardOwnedExecutor"', self.spec)
        self.assertIn("OP_ADD(A5P8ForwardOwnedExecutor)", self.host)
        self.assertIn("A5P8ForwardOwnedExecutorTilingData", self.tiling)
        self.assertIn("KERNEL_TYPE_AIV_ONLY", self.kernel)
        for legacy_name in (
            "A5ScoreReceiptSpeculativeExecutor",
            "A5StreamingQuantLiGet",
            "A5OfficialC8",
            "quant_lightning",
            "a5_score_receipt_speculative_executor",
        ):
            self.assertNotIn(legacy_name, self.kernel)

    def test_dynamic_resource_model_has_no_fixed_core_partition(self) -> None:
        self.assertIn("TryAcquireKvafStreamingBatchGetQueueFromAicore", self.kernel)
        self.assertIn("ClaimP8GlobalPrivateSlotSimt", self.kernel)
        for forbidden in (
            "kGetLane",
            "LI_CORE",
            "GET_CORE",
            "physical_core",
        ):
            self.assertNotIn(forbidden, self.kernel)

    def test_post_is_durable_and_never_uses_post_intent_rollback(self) -> None:
        for token in (
            "BeginAndPublishKvafP8WaveFilling",
            "ArmKvafP8WaveFillingForExternalWrites",
            "PromoteKvafP8WaveFillingToPostIntent",
            "PostKvafP8Wave",
            "CommitKvafP8WaveMutablePayloadAndStateFromAicore",
            "DataSyncBarrier<MemDsbT::DDR>",
            "identity.post_sequence",
            "ClaimGlobalPrivate",
        ):
            self.assertIn(token, self.kernel)
        self.assertNotIn("PrepareKvafP8WavePostIntent", self.kernel)
        self.assertNotIn("ReleaseKvafP8UnpostedWave", self.kernel)
        self.assertIn("ReloadForwardControlForPhase", self.kernel)
        post = self.kernel[
            self.kernel.index("bool PostReceipt"):self.kernel.index("void RunPostOnly")
        ]
        stage_candidate = self.kernel[
            self.kernel.index("bool StageCandidate"):
            self.kernel.index("bool PostReceipt")
        ]
        self.assertLess(
            post.rindex("ReloadForwardControlForPhase"),
            post.index("PostKvafP8Wave"),
        )
        self.assertLess(
            post.index("BeginAndPublishKvafP8WaveFilling"),
            post.index("StageCandidate"),
        )
        self.assertIn("ArmKvafP8WaveFillingForExternalWrites", stage_candidate)
        self.assertLess(post.index("StageCandidate"),
                        post.index("PromoteKvafP8WaveFillingToPostIntent"))
        self.assertLess(
            post.index("PromoteKvafP8WaveFillingToPostIntent"),
            post.index("PostKvafP8Wave"),
        )

    def test_filling_record_closes_the_lease_acquire_to_descriptor_gap(self) -> None:
        """A pre-descriptor lease has a durable, terminal-race-safe owner."""
        post = self.kernel[
            self.kernel.index("bool PostReceipt"):self.kernel.index("void RunPostOnly")
        ]
        stage_candidate = self.kernel[
            self.kernel.index("bool StageCandidate"):
            self.kernel.index("bool PostReceipt")
        ]
        for token in (
            "kKvafP8WaveFilling",
            "kKvafP8WaveFillingReleaseClaimed",
            "ReleaseKvafP8FillingLeaseAfterProvenNoWork",
            "kKvafP8WaveOutcomeFillingQuarantined",
            "QuarantineKvafP8PostIntentStateFromAicore",
            "QuarantineKvafP8PrivatePendingFromAicore",
        ):
            self.assertIn(token, self.kernel)
        # A no-work release is allowed only before Arm marks any descriptor,
        # private record, or request ledger externally visible.
        self.assertLess(
            post.index("BeginAndPublishKvafP8WaveFilling"),
            post.index("ReleaseFillingAfterProvenNoWork"),
        )
        self.assertIn("ArmKvafP8WaveFillingForExternalWrites", stage_candidate)
        self.assertLess(post.index("StageCandidate"),
                        post.index("PromoteKvafP8WaveFillingToPostIntent"))

    def test_p8_descriptor_publish_uses_the_forward_owned_kvaf_abi_order(self) -> None:
        """Keep a C++ signature drift out of the expensive remote OPP build.

        KVAF deliberately puts the wave identity and private-record arena
        before descriptor scratch in its P8-only overload.  The legacy
        streaming helper has another ordering, so a token-only closure check
        would not catch an accidental call to the wrong ABI.
        """
        post = self.kernel[
            self.kernel.index("bool PostReceipt"):self.kernel.index("void RunPostOnly")
        ]
        expected = (
            "PublishKvafP8BatchGetDescriptorScratch(\n"
            "                control_, lease, identity, privateRecords_,\n"
            "                tiling_->privateRowCount, descriptor)"
        )
        self.assertIn(expected, post)

    def test_deterministic_no_post_is_cleaned_only_by_drain(self) -> None:
        post = self.kernel[
            self.kernel.index("bool PostReceipt"):self.kernel.index("void RunPostOnly")
        ]
        drain = self.kernel[
            self.kernel.index("void DrainOneQueue"):
            self.kernel.index("void RunDrain")
        ]
        no_post = self.kernel[
            self.kernel.index("bool DrainDeterministicNoPostWave"):
            self.kernel.index("void DrainOneQueue")
        ]
        self.assertIn("kKvafP8WaveOutcomePostRejected", post)
        self.assertIn("wave.post_committed == 0U", post)
        self.assertNotIn("ReleaseKvafP8PendingPrivateRecordAfterNoPost", post)
        self.assertNotIn("ReleaseKvafP8ClaimedUnpostedWaveAfterPrivateCleanup", post)
        self.assertIn("TryClaimKvafP8UnpostedWaveDrainFromAicore", drain)
        self.assertIn("DrainDeterministicNoPostWave", drain)
        self.assertIn("ReleaseKvafP8PendingPrivateRecordAfterNoPost", no_post)
        self.assertIn(
            "ReleaseKvafP8ClaimedUnpostedWaveAfterPrivateCleanup", no_post
        )
        self.assertLess(
            no_post.index("ReleaseKvafP8PendingPrivateRecordAfterNoPost"),
            no_post.index("ReleaseKvafP8ClaimedUnpostedWaveAfterPrivateCleanup"),
        )

    def test_drain_claim_precedes_gm_reload(self) -> None:
        drain_start = self.kernel.index("void DrainOneQueue")
        drain_end = self.kernel.index("void RunDrain", drain_start)
        drain = self.kernel[drain_start:drain_end]
        self.assertLess(
            drain.index("TryClaimKvafP8WaveDrainFromAicore"),
            drain.index("LoadKvafP8WaveRecordFromGm"),
        )
        for token in (
            "kKvafP8WaveDrainClaimed",
            "TryRetireKvafP8Wave",
            "FinishKvafP8Wave",
            "PublishTerminalPrivateRows",
            "ReleaseKvafP8WaveAfterPrivateReady",
            "MarkForwardAbort",
        ):
            self.assertIn(token, drain)
        self.assertIn("TryClaimKvafP8UnpostedWaveDrainFromAicore", drain)
        self.assertIn("PublishKvafP8PrivateTerminalRecords", self.kernel)

    def test_happy_post_cq_path_has_no_host_wait(self) -> None:
        """The successful transport path is PostOnly then bounded drain.

        This is intentionally a source-level integration contract until a
        target machine supplies registered Direct-HCCP resources.  It rejects
        the historical error of inserting a Python/host wait between the
        descriptor publish and CQ terminal publication.
        """
        post = self.kernel[
            self.kernel.index("bool PostReceipt"):self.kernel.index("void RunPostOnly")
        ]
        drain = self.kernel[
            self.kernel.index("void DrainOneQueue"):
            self.kernel.index("void RunDrain")
        ]
        self.assertLess(post.index("PromoteKvafP8WaveFillingToPostIntent"),
                        post.index("PostKvafP8Wave"))
        self.assertIn("TryRetireKvafP8Wave", drain)
        self.assertIn("FinishKvafP8Wave", drain)
        self.assertIn("PublishTerminalPrivateRows", drain)
        self.assertIn("ReleaseKvafP8WaveAfterPrivateReady", drain)
        self.assertNotIn("Synchronize", post + drain)

    def test_cq_pending_rehydrates_posted_record_for_later_drain(self) -> None:
        drain = self.kernel[
            self.kernel.index("void DrainOneQueue"):
            self.kernel.index("void RunDrain")
        ]
        # `TryRetire` does not turn an in-flight CQ into a guessed terminal
        # result.  The durable record must return to POSTED through an
        # expected-state commit, so a later graph drain node can CAS-claim it
        # without a stale AIC-local record reviving a terminal quarantine.
        self.assertIn("wave.state == kvaf_fused_get::kKvafP8WavePosted", drain)
        self.assertLess(
            drain.index("TryRetireKvafP8Wave"),
            drain.index("CommitWaveState(", drain.index("TryRetireKvafP8Wave")),
        )

    def test_hot_paths_use_state_last_cas_and_a_minimal_tagged_arena_lease(self) -> None:
        """A terminal CAS winner must never be overwritten by a stale worker."""

        for forbidden in (
            "StoreKvafP8WaveRecordToGm",
            "StoreKvafP8PrivateRecordToGm",
        ):
            self.assertNotIn(forbidden, self.kernel)
        for required in (
            "CommitKvafP8WaveMutablePayloadAndStateFromAicore",
            "CommitKvafP8ClaimedUnpostedWaveReleaseFromAicore",
            # The P8 kernel calls the KVAF publisher; KVAF's own source
            # contract proves its internal PENDING -> READY/FAILED CAS.
            "PublishKvafP8PrivateTerminalRecords",
            "kKvafP8WaveFillingClaimed",
            "kKvafP8PrivateAllocating",
            "a5_p8_arena::MakeTag",
            "a5_p8_arena::ClaimTerminalSimtService",
            "a5_p8_arena::CompleteTerminalSimtService",
        ):
            self.assertIn(required, self.kernel)

        # `p8_arena_health` is one original int32 word, not a new owner
        # tensor or a four-field forward identity.  The low state bits and
        # per-acquire generation form one exact CAS tag.
        for required in (
            "kArenaStateBits = 3U",
            "kArenaGenerationMax",
            "TryClaimAdmissionFromSimt",
            "FinishAdmissionFromSimt",
            "PoisonActiveLeaseFromSimt",
            "TryClaimTerminalFromSimt",
            "CompleteTerminalFromSimt",
            "QuarantineGenerationWrapFromSimt",
        ):
            self.assertIn(required, self.arena_tag)
        for forbidden in (
            "TryTransitionFromSimt",
            "kP8ArenaHealthHealthy",
            "kP8ArenaHealthUnhealthy",
            "kP8ArenaHealthActiveForward",
        ):
            self.assertNotIn(forbidden, self.kernel + self.admission + self.advance)

    def test_arena_release_has_one_terminal_leader_and_no_active_to_idle_shortcut(self) -> None:
        """Arena reuse is closed by terminal-before-reuse, not extra identity words."""
        for required in (
            "ACTIVE -> IDLE shortcut",
            "TryClaimTerminalFromSimt",
            "CompleteTerminalFromSimt",
            "QuarantineGenerationWrapFromSimt",
        ):
            self.assertIn(required, self.arena_tag)
        self.assertIn("ClaimP8ArenaForForwardTerminal", self.kernel)
        self.assertIn("ReleaseP8ArenaAfterCleanForwardTerminal", self.kernel)
        self.assertIn("PoisonP8ArenaActive", self.kernel)
        self.assertIn("CompleteP8ArenaTerminal", self.kernel)
        # Abort/phase control must compare the tag produced by this forward's
        # admission, rather than a raw state value or e/r/l tuple.
        self.assertIn("GetKvafP8ForwardLeaseGeneration", self.advance)
        self.assertIn("MakeTag(", self.advance)
        # The advance node invokes the narrow SIMT service wrapper.  The
        # actual exact ACTIVE(g) -> UNHEALTHY(g) CAS lives in the shared tag
        # header; keeping the wrapper name here prevents a direct/raw store
        # from being substituted in the graph node.
        self.assertIn("PoisonActiveLeaseSimtService", self.advance)

    def test_arena_tag_helpers_are_explicitly_aicore_callable(self) -> None:
        """Prevent CANN from silently treating SIMT helper calls as host calls."""
        for signature in (
            "__aicore__ inline uint32_t MakeTag",
            "__aicore__ inline ArenaState StateOf",
            "__aicore__ inline uint32_t GenerationOf",
        ):
            self.assertIn(signature, self.arena_tag)

    def test_only_admission_writes_the_cached_lease_generation(self) -> None:
        """No generic forward-control store may erase the high-word ticket.

        The arena generation is intentionally small: it shares the existing
        control word rather than adding a new tensor.  That works only if P8
        device hot paths use the narrow low-word flag CAS and leave the high
        word to admission.
        """
        p8_sources = self.kernel + self.admission + self.advance
        self.assertNotIn("StoreKvafP8ForwardControlToGm", p8_sources)
        self.assertEqual(
            self.admission.count("StoreKvafP8ForwardLeaseGenerationToGm"),
            1,
        )

    def test_forward_terminal_quarantines_post_intent_not_release_it(self) -> None:
        terminal_start = self.kernel.index("void RunForwardTerminal")
        terminal = self.kernel[terminal_start:]
        # The KVAF terminal endpoint classifies every non-FREE durable wave,
        # including POST_INTENT.  P8 deliberately does not call a normal
        # release endpoint from this phase because an interrupted doorbell may
        # still be visible to HCCP.
        self.assertIn("QuarantineKvafP8WaveAtForwardTerminal", terminal)
        self.assertNotIn("ReleaseKvafP8WaveAfterPrivateReady", terminal)
        self.assertNotIn("ReleaseKvafP8ClaimedUnpostedWaveAfterPrivateCleanup", terminal)

    def test_failed_cq_is_terminal_private_miss_not_qsfa_ready(self) -> None:
        publish = self.kernel[
            self.kernel.index("bool PublishTerminalPrivateRows"):
            self.kernel.index("bool IsDeterministicNoPostWave")
        ]
        qsfa = self.kernel[
            self.kernel.index("void RunQsfaTerminalRelease"):
            self.kernel.index("void RunForwardTerminal")
        ]
        # KVAF publishes READY or FAILED only after Finish validates every
        # descriptor output.  FAILED remains a terminal record for exact /
        # GuaranteedRecall fallback; it is never promoted to READY by P8.
        self.assertIn("kKvafP8PrivateReady", publish)
        self.assertIn("kKvafP8PrivateFailed", publish)
        self.assertIn("privateReady_[output_index]", publish)
        self.assertIn("kKvafP8PrivateFailed", qsfa)
        self.assertNotIn("kKvafP8PrivateReady);", publish)

    def test_qsfa_terminal_owns_private_release(self) -> None:
        terminal_start = self.kernel.index("void RunQsfaTerminalRelease")
        terminal_end = self.kernel.index("void RunForwardTerminal", terminal_start)
        terminal = self.kernel[terminal_start:terminal_end]
        self.assertIn("IsKvafP8PrivateViewReadyForQsfa", terminal)
        self.assertIn("kKvafP8ForwardFlagQsfaTerminalObserved", terminal)
        self.assertIn("ReleaseKvafP8PrivateRecordAfterQsfaTerminal", terminal)
        self.assertIn("privateRecords_ + slot", terminal)
        self.assertIn("privateReleaseResult_", terminal)
        self.assertNotIn("ClearPrivateMapping", terminal)
        self.assertIn("kKvafP8PrivateReady", terminal)
        self.assertIn("kKvafP8PrivateFailed", terminal)

    def test_abort_is_atomic_and_never_uses_the_legacy_local_helper(self) -> None:
        self.assertIn("MarkKvafP8ForwardAbortFromAicore", self.kernel)
        self.assertIn("KvafP8ForwardAbortResult", self.kernel)
        self.assertNotIn("kvaf_fused_get::MarkKvafP8ForwardAbort(", self.kernel)

    def test_forward_terminal_quarantines_without_queue_release(self) -> None:
        terminal_start = self.kernel.index("void RunForwardTerminal")
        terminal = self.kernel[terminal_start:]
        for token in (
            "kKvafP8ForwardPhaseForwardTerminal",
            "QuarantineKvafP8WaveAtForwardTerminal",
            "QuarantineKvafP8PrivateAtForwardTerminal",
            "KvafP8ForwardTerminalResult",
        ):
            self.assertIn(token, terminal)
        self.assertNotIn("ReleaseKvafP8WaveAfterPrivateReady", terminal)

    def test_candidate_receipt_identity_is_full_and_generation_guarded(self) -> None:
        """Receipt/cursor reuse must be closed beyond an epoch-only check."""
        for token in (
            "kA5P8CandidateReceiptIdentityWords = 4U",
            "LoadStableReceiptIdentity",
            "SameReceiptIdentity",
            "IsCurrentReceiptIdentity",
            "graph_replay_generation",
            "layer_control_index",
            "receipt_generation",
            "candidateReceiptIdentity_",
            "candidateCursorIdentity_",
        ):
            self.assertIn(token, self.kernel + self.tiling)
        consume_start = self.kernel.index("bool LoadCurrentCandidateReceipt")
        consume_end = self.kernel.index("void TombstoneReceiptIdentity", consume_start)
        consume = self.kernel[consume_start:consume_end]
        # Sample generation/prefix before and after status: one read is not
        # enough because terminal/replay invalidates generation concurrently.
        self.assertGreaterEqual(
            consume.count("LoadStableReceiptIdentity(CandidateReceiptIdentity(receipt)"),
            2,
        )
        self.assertIn("SameReceiptIdentity(before, after)", consume)

    def test_cursor_reset_and_terminal_use_full_identity_tombstones(self) -> None:
        post = self.kernel[
            self.kernel.index("bool PostReceipt"):self.kernel.index("void RunPostOnly")
        ]
        self.assertIn("SameReceiptIdentity(cursor_identity, receipt_identity)", post)
        self.assertIn("PublishCursorReceiptIdentity(receipt, receipt_identity)", post)
        terminal_start = self.kernel.index("void TombstoneCurrentCandidateReceipt")
        terminal_end = self.kernel.index("void RunForwardTerminal", terminal_start)
        terminal = self.kernel[terminal_start:terminal_end]
        for token in (
            "TombstoneReceiptIdentity(CandidateReceiptIdentity(receipt))",
            "TombstoneReceiptIdentity(CandidateCursorIdentity(receipt))",
            "kCandidateStatusTombstoned",
        ):
            self.assertIn(token, terminal)

    def test_build_script_stages_only_the_isolated_p8_source(self) -> None:
        self.assertIn("STAGE_P8_FORWARD_OWNED_EXECUTOR_ONLY", self.build)
        self.assertIn("A5P8ForwardOwnedExecutor", self.build)
        self.assertIn("A5P8ForwardControlAdvance", self.build)
        self.assertIn("A5P8ArenaAdmissionGate", self.build)
        self.assertIn("a5_p8_forward_owned_executor", self.build)
        # The tagged-arena helper is part of the isolated OPP source closure.
        # A stale staged copy would compile an old lifecycle even when the
        # Python/runtime checkout has the new graph ordering.
        for token in (
            "P8_ARENA_TAG",
            "P8_ARENA_TAG_HEADER",
            "a5_p8_arena_tag.h",
        ):
            self.assertIn(token, self.build)
        for token in (
            "BeginAndPublishKvafP8WaveFilling",
            "ArmKvafP8WaveFillingForExternalWrites",
            "PromoteKvafP8WaveFillingToPostIntent",
            "ReleaseKvafP8FillingLeaseAfterProvenNoWork",
            "TryClaimKvafP8UnpostedWaveDrainFromAicore",
            "ReleaseKvafP8PendingPrivateRecordAfterNoPost",
            "ReleaseKvafP8ClaimedUnpostedWaveAfterPrivateCleanup",
        ):
            self.assertIn(token, self.build)

    def test_runner_accepts_bash_invoked_non_executable_build_helpers(self) -> None:
        """A normal Git mode-100644 shell helper is valid when invoked by Bash."""
        for invocation in (
            'bash "$P8_BUILD_SCRIPT"',
            'bash "$P8_REFERENCE_BUILD_SCRIPT"',
        ):
            self.assertIn(invocation, self.runner)
        for readable_gate in (
            '! -f "$P8_BUILD_SCRIPT" || ! -r "$P8_BUILD_SCRIPT"',
            '! -f "$P8_REFERENCE_BUILD_SCRIPT" || ! -r "$P8_REFERENCE_BUILD_SCRIPT"',
        ):
            self.assertIn(readable_gate, self.runner)
        self.assertNotIn('! -x "$P8_BUILD_SCRIPT"', self.runner)

    def test_clean_reference_build_uses_an_explicit_offline_tool_cache(self) -> None:
        """Never reuse an old CMakeCache or silently fall back to network."""
        for token in (
            'P8_CANN_3RD_LIB_PATH',
            'cann_third_party_cache_incomplete:',
            'P8_TOOLCHAIN_CACHE cann_3rd_lib_path=%s makeself=1 json=1 abseil=1 protobuf=1',
            'A5_OFFICIAL_C8_CANN_3RD_LIB_PATH="$P8_CANN_3RD_LIB_PATH"',
            'NANOVLLM_A5_CANN_3RD_LIB_PATH="$P8_CANN_3RD_LIB_PATH"',
        ):
            self.assertIn(token, self.runner)
        self.assertLess(
            self.runner.index('cann_third_party_cache_incomplete:'),
            self.runner.index('P8_KVAF_BUILD="$P8_RUN_ROOT/kvaf_build"'),
        )
        for token in (
            'A5_OFFICIAL_C8_CANN_3RD_LIB_PATH',
            'cann_third_party_cache_incomplete',
            '--cann_3rd_lib_path="${CANN_3RD_LIB_PATH}"',
            'toolchain_cann_3rd_lib_path=${CANN_3RD_LIB_PATH}',
            'STAGED_MAKESELF_DIR="${SOURCE_ROOT}/csrc/third_party/makeself"',
            'P3_REFERENCE_TOOLCHAIN_MAKESELF_STAGED',
            'reason=makeself_stage_failed',
            'CMAKE_HOME_DIRECTORY:INTERNAL',
            'reason=foreign_cmake_cache',
            'P3_REFERENCE_INCREMENTAL',
        ):
            self.assertIn(token, self.reference_build)
        self.assertIn(
            'CANN_3RD_LIB_PATH="${NANOVLLM_A5_CANN_3RD_LIB_PATH:-}"',
            self.build,
        )
        self.assertIn(
            '--cann_3rd_lib_path="${CANN_3RD_LIB_PATH}"', self.build
        )
        self.assertIn(
            'STAGED_MAKESELF_DIR="${GENERATED}/third_party/makeself"',
            self.build,
        )
        self.assertIn('CPack makeself staging failed', self.build)
        self.assertLess(
            self.build.index('STAGED_MAKESELF_DIR="${GENERATED}/third_party/makeself"'),
            self.build.index('pushd "${GENERATED}"'),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
