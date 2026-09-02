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
import re
import unittest


OPS_ROOT = Path(__file__).resolve().parents[1]
P8_ROOT = OPS_ROOT / "a5_p8_forward_owned_executor"
KERNEL = P8_ROOT / "op_kernel" / "a5_p8_forward_owned_executor.cpp"
HOST = P8_ROOT / "op_host" / "a5_p8_forward_owned_executor.cpp"
ADMISSION_HOST = P8_ROOT / "op_host" / "a5_p8_arena_admission_gate.cpp"
ADVANCE_HOST = P8_ROOT / "op_host" / "a5_p8_forward_control_advance.cpp"
TILING = P8_ROOT / "op_kernel" / "a5_p8_forward_owned_executor_tiling.h"
ARENA_TAG = P8_ROOT / "op_kernel" / "a5_p8_arena_tag.h"
ADMISSION = P8_ROOT / "op_kernel" / "a5_p8_arena_admission_gate.cpp"
ADVANCE = P8_ROOT / "op_kernel" / "a5_p8_forward_control_advance.cpp"
SPEC = P8_ROOT / "ops.json"
BUILD = OPS_ROOT / "build_nanovllm_ops.sh"
RUNNER = OPS_ROOT / "tests" / "run_p8_forward_owned_executor_npu.sh"
REFERENCE_BUILD = OPS_ROOT / "a5_official_c8_reference" / "build_reference_opp.sh"
BRIDGE = (
    P8_ROOT / "torch_extension" / "csrc" / "p8_forward_owned_executor.cpp"
)
P8_SETUP = P8_ROOT / "torch_extension" / "p8_forward_owned_executor_setup.py"
STAGE_A_BRIDGE = (
    OPS_ROOT
    / "a5_score_receipt_candidate_executor"
    / "torch_extension"
    / "csrc"
    / "score_receipt_candidate_executor.cpp"
)
OP_API_COMMON = OPS_ROOT / "torch_extension" / "csrc" / "op_api_common.h"


class P8ForwardOwnedExecutorContractTest(unittest.TestCase):
    @staticmethod
    def _simt_callee_bodies(source: str) -> list[str]:
        """Return function bodies in AscendC's SIMT callable domain."""
        bodies: list[str] = []
        for match in re.finditer(r"__simt_callee__\s+inline\b", source):
            opening = source.find("{", match.end())
            if opening < 0:
                continue
            depth = 0
            for index in range(opening, len(source)):
                char = source[index]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        bodies.append(source[opening + 1:index])
                        break
        return bodies

    @staticmethod
    def _function_body(source: str, signature: str) -> str:
        start = source.index(signature)
        opening = source.find("{", start)
        if opening < 0:
            raise AssertionError(f"missing body for {signature}")
        depth = 0
        for index in range(opening, len(source)):
            char = source[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return source[opening + 1:index]
        raise AssertionError(f"unterminated body for {signature}")

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
        cls.bridge = BRIDGE.read_text(encoding="utf-8")
        cls.p8_setup = P8_SETUP.read_text(encoding="utf-8")
        cls.stage_a_bridge = STAGE_A_BRIDGE.read_text(encoding="utf-8")
        cls.op_api_common = OP_API_COMMON.read_text(encoding="utf-8")

    def test_p8_acl_host_lifecycle_is_drained_direct_and_tensor_owned(self) -> None:
        """P8 bypasses ReleaseQueue without synchronizing device execution."""
        direct = "EXEC_NPU_CMD_TENSOR_OWNED_DRAINED_DIRECT_ASYNC_ZERO_WORKSPACE("
        typed_direct = (
            "EXEC_NPU_CMD_TENSOR_OWNED_DRAINED_DIRECT_ASYNC_ZERO_WORKSPACE_TYPED("
        )
        deferred = "EXEC_NPU_CMD_TENSOR_OWNED_DEFERRED_ZERO_WORKSPACE("
        self.assertEqual(self.bridge.count(direct), 2)
        self.assertEqual(self.bridge.count(typed_direct), 1)
        self.assertNotIn("EXEC_NPU_CMD_ORDERED(", self.bridge)
        self.assertEqual(self.stage_a_bridge.count(direct), 1)
        self.assertNotIn("EXEC_NPU_CMD_ORDERED(", self.stage_a_bridge)
        self.assertNotIn(deferred, self.bridge)
        self.assertNotIn(deferred, self.stage_a_bridge)
        self.assertNotIn("RunOpApi", self.bridge)
        self.assertNotIn("RunOpApi", self.stage_a_bridge)
        self.assertNotIn("UseStreamResIfNeeded", self.bridge)
        self.assertNotIn("UseStreamResIfNeeded", self.stage_a_bridge)
        self.assertIn(
            "#define EXEC_NPU_CMD_TENSOR_OWNED_DRAINED_DIRECT_ASYNC_ZERO_WORKSPACE",
            self.op_api_common,
        )
        direct_macro = self.op_api_common[
            self.op_api_common.index(
                "#define EXEC_NPU_CMD_TENSOR_OWNED_DRAINED_DIRECT_ASYNC_ZERO_WORKSPACE_IMPL"
            ):self.op_api_common.index(
                "#define EXEC_NPU_CMD_TENSOR_OWNED_DRAINED_DIRECT_ASYNC_ZERO_WORKSPACE_TYPED"
            )
        ]
        for token in (
            "std::make_tuple(__VA_ARGS__)",
            "inline_npu_stream.stream()",
            "RequireDrainedCallerThreadAsyncOpApiContract(inline_npu_stream)",
            "RunDrainedCallerThreadHost(std::move(inline_acl_call))",
            "InitPtaCacheForWorkspace()",
            "ResolveSymbol(\"InitHugeMemThreadLocal\")",
            "ConvertTypes(inline_args_ref...",
            "inline_workspace_size == 0",
            "ReleaseTuple(inline_converted)",
            "UnInitPtaCacheForWorkspace(true)",
            "caller_thread_direct_dispatch_done",
        ):
            self.assertIn(token, direct_macro)
        for forbidden in (
            "at_npu::native::OpCommand::RunOpApi(",
            "UseStreamResIfNeeded",
            "stream(false)",
            "aclrtSynchronizeStream",
        ):
            self.assertNotIn(forbidden, direct_macro)
        self.assertIn("NPUCachingAllocator::recordStream", self.op_api_common)
        self.assertIn("RecordTensorTupleOnNpuStream(", direct_macro)
        self.assertIn("inline_args, inline_npu_stream", direct_macro)
        self.assertIn("ClearTensorTuple(inline_args)", direct_macro)
        record = direct_macro.index("RecordTensorTupleOnNpuStream(")
        release = direct_macro.index("ReleaseTuple(inline_converted)", record)
        clear = direct_macro.index("ClearTensorTuple(inline_args)", release)
        self.assertLess(record, release)
        self.assertLess(release, clear)
        self.assertNotIn("auto keepalive = std::make_tuple", self.bridge)
        self.assertNotIn("auto keepalive = std::make_tuple", self.stage_a_bridge)
        stage_a_host = (
            OPS_ROOT
            / "a5_score_receipt_candidate_executor"
            / "op_host"
            / "a5_score_receipt_candidate_executor.cpp"
        )
        for host in (HOST, ADMISSION_HOST, ADVANCE_HOST, stage_a_host):
            source = host.read_text(encoding="utf-8")
            self.assertIn("GetWorkspaceSizes(1)[0] = 0U", source)

    def test_stage_b_uses_its_generated_28_tensor_workspace_signature(self) -> None:
        """P8 Stage-B must compile against the OPP it resolves at runtime.

        The generated header is the one authoritative declaration of the
        same-name mutable-ref ABI.  It has 28 Tensor inputs, not a synthetic
        47-argument suffix containing the 19 Python aliases.  Binding dlsym
        through ``decltype`` makes a generated-host ABI change fail during the
        scoped bridge build, before task-queue execution can enter CANN.
        """
        for token in (
            '#include "aclnn_a5_p8_forward_owned_executor.h"',
            "EXEC_NPU_CMD_TENSOR_OWNED_DRAINED_DIRECT_ASYNC_ZERO_WORKSPACE_TYPED(",
            "decltype(&aclnnA5P8ForwardOwnedExecutorGetWorkspaceSize)",
            "authoritative contract and deliberately carries no 19-output suffix",
        ):
            self.assertIn(token, self.bridge)
        for token in (
            "NANOVLLM_A5_P8_FORWARD_OWNED_OPP",
            "P8_ACLNN_INCLUDE",
            "aclnn_a5_p8_forward_owned_executor.h",
            'f"-I{P8_ACLNN_INCLUDE}"',
            "CANN_ACLNN_INCLUDE_CANDIDATES",
            '"aclnn" / "acl_meta.h"',
            'f"-I{CANN_ACLNN_INCLUDE}"',
            "freshly generated Stage-B ACLNN header",
        ):
            self.assertIn(token, self.p8_setup)
        for token in (
            "#define EXEC_NPU_CMD_TENSOR_OWNED_DRAINED_DIRECT_ASYNC_ZERO_WORKSPACE_TYPED",
            "reinterpret_cast<workspace_function_type>(inline_workspace_addr_fn)",
            '"caller_thread_workspace_fn_ready"',
            '"caller_thread_workspace_call_begin"',
            '"caller_thread_workspace_call_end"',
        ):
            self.assertIn(token, self.op_api_common)
        export_index = self.runner.index(
            'export NANOVLLM_A5_P8_FORWARD_OWNED_OPP="$P8_STAGE_B_OPP"'
        )
        cann_export_index = self.runner.index("export ASCEND_CANN_PACKAGE_PATH")
        bridge_index = self.runner.index('stage_b_bridge "$P8_RUN_ROOT/stage_b_bridge.log"')
        self.assertLess(export_index, bridge_index)
        self.assertLess(cann_export_index, bridge_index)
        self.assertNotIn("47", self.bridge)

    def test_p8_drained_direct_host_contract_keeps_task_queue_ordering(self) -> None:
        """P8 drains prior producers but owns its own async ACLNN lifecycle."""
        for token in (
            "RequireDrainedCallerThreadAsyncOpApiContract",
            "OptionsManager::GetTaskQueueEnable()",
            "CheckBlockingEnable()",
            "P8_DRAINED_DIRECT_HOST_GATE",
            "device_launch=async",
            "RunDrainedCallerThreadHost",
        ):
            self.assertIn(token, self.op_api_common)
        for token in (
            'P8_TASK_QUEUE_ENABLE="${P8_TASK_QUEUE_ENABLE:-1}"',
            'P8_ASCEND_LAUNCH_BLOCKING="${P8_ASCEND_LAUNCH_BLOCKING:-0}"',
            'runner_fail "unsafe_drained_direct_launch_environment"',
            'export TASK_QUEUE_ENABLE="$P8_TASK_QUEUE_ENABLE"',
            'export ASCEND_LAUNCH_BLOCKING="$P8_ASCEND_LAUNCH_BLOCKING"',
            "P8_DRAINED_DIRECT_HOST_ENV pass=1",
            "torch_npu.npu.get_task_queue_enable()",
        ):
            self.assertIn(token, self.runner)

    def test_acl_tensor_storage_shape_is_owned_by_the_live_aten_tensor(self) -> None:
        """A deferred ACLNN descriptor must not retain a local shape buffer.

        P8's input 26 (`rows_per_entry`) is a real int32[1] GM tensor.  The
        bridge validates it correctly before enqueue, so a later tiler report
        of `storage_rank=<null>` identifies descriptor construction/lifetime,
        not a reason to weaken the P8 ABI.  The launcher uses the TensorImpl-
        backed sizes pointer for both view and ND storage shape; the
        tensor-owned deferred closure keeps that Tensor alive through
        GetWorkspaceSize and launch.
        """
        convert_tensor = self._function_body(
            self.op_api_common,
            "inline aclTensor *ConvertTensor(",
        )
        self.assertIn("const auto sizes = tensor.sizes();", convert_tensor)
        self.assertIn("const auto strides = tensor.strides();", convert_tensor)
        self.assertNotIn("std::vector<int64_t> storage_dims", convert_tensor)
        self.assertIn("sizes.data(),\n      static_cast<uint64_t>(sizes.size()),", convert_tensor)
        self.assertIn("strides.data(),", convert_tensor)
        self.assertIn("aclCreateTensor returned null", convert_tensor)
        self.assertIn("return descriptor;", convert_tensor)
        for token in (
            "P7_1_DESCRIPTOR_ARG api=%s index=%zu descriptor=%p",
            "LogConvertedDescriptorArgument(api, I, std::get<I>(args))",
        ):
            self.assertIn(token, self.op_api_common)

    def test_runner_can_cut_prepost_before_real_post_without_rebuild(self) -> None:
        """One fresh assembly may isolate then exercise the valid POST path."""
        for token in (
            "prepost_then_full",
            "run_p8_runtime_case runtime_prepost prepost_control_reject",
            "run_p8_runtime_case runtime_full full",
            "P8_RUNNER_FAIL stage=runtime_prepost",
            "P8_RUNNER_FAIL stage=runtime_full",
        ):
            self.assertIn(token, self.runner)

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

    def test_stage_b_launch_is_capped_by_receipts_and_queues_not_arena_rows(
        self,
    ) -> None:
        """Private-slot capacity must not manufacture empty AIV workers.

        A worker strides all private rows in QSFA/terminal phases, so a
        64-slot private arena with one Direct-HCCP queue still has exactly one
        concurrent transport service task.  This is a dynamic workload cap,
        not a permanent worker-to-queue or core assignment.
        """
        compact = re.sub(r"\s+", " ", self.host)
        self.assertIn(
            "uint32_t workers = available_aiv < receipts ? available_aiv : receipts;",
            compact,
        )
        self.assertIn("workers = workers < queues ? workers : queues;", compact)
        self.assertNotIn("private_rows > work_items", self.host)
        for marker in ('LogP8Tiling("PREALLOC"', 'LogP8Tiling("READY"'):
            self.assertIn(marker, self.host)
        self.assertIn("const ge::graphStatus schedule_status = context->SetScheduleMode(1U);", self.host)

    def test_stage_b_simt_boundaries_match_cann_address_space_rules(self) -> None:
        """Catch front-end errors before an expensive staged OPP build.

        CANN 9.1 requires the transition-result hand-off of the SIMT service
        to be UB-addressed, while ``ReadGmByPassDCache`` takes a mutable GM
        pointer even for read-only accesses.  Both declarations are semantic
        ABI details, not stylistic constness.
        """
        self.assertIn(
            "uint32_t set_flags, __ubuf__ P8TransitionServiceResult* result)",
            self.advance,
        )
        self.assertNotIn(
            "uint32_t set_flags, P8TransitionServiceResult* result)",
            self.advance,
        )
        self.assertIn("__gm__ int32_t* const leases", self.admission)
        self.assertNotIn("__gm__ const int32_t* const leases", self.admission)

        for vf_signature, callee in (
            (
                "inline void ClaimP8GlobalPrivateSlotSimt(",
                "ClaimP8GlobalPrivateSlotFromSimt(",
            ),
            (
                "inline void CommitP8GlobalPrivateSlotMapsSimt(",
                "CommitP8GlobalPrivateSlotMapsFromSimt(",
            ),
        ):
            vf_body = self._function_body(self.kernel, vf_signature)
            self.assertIn(callee, vf_body)
            self.assertNotIn("asc_atomic_", vf_body)
            self.assertNotIn("PrepareKvafP8PrivateRecord(", vf_body)
            self.assertNotIn("PublishKvafP8PrivateAllocatingPayloadToGm(", vf_body)
            vf_without_comments = re.sub(
                r"//[^\n]*|/\*.*?\*/", "", vf_body, flags=re.DOTALL
            ).strip()
            self.assertRegex(
                vf_without_comments,
                rf"(?s)^{re.escape(callee[:-1])}\(.*\);$",
            )

        for callee_signature in (
            "inline void ClaimP8GlobalPrivateSlotFromSimt(",
            "inline void CommitP8GlobalPrivateSlotMapsFromSimt(",
        ):
            callee_body = self._function_body(self.kernel, callee_signature)
            self.assertNotIn("PrepareKvafP8PrivateRecord(", callee_body)
            self.assertNotIn("PublishKvafP8PrivateAllocatingPayloadToGm(", callee_body)
            self.assertNotIn("FromAicore(", callee_body)

        claim_start = self.kernel.index("bool ClaimGlobalPrivate(")
        claim_end = self.kernel.index("// Each graph phase reloads", claim_start)
        claim = self.kernel[claim_start:claim_end]
        self.assertLess(
            claim.index("VF_CALL<ClaimP8GlobalPrivateSlotSimt>"),
            claim.index("DataSyncBarrier<MemDsbT::DDR>();"),
        )
        self.assertLess(
            claim.index("DataSyncBarrier<MemDsbT::DDR>();"),
            claim.index("VF_CALL<CommitP8GlobalPrivateSlotMapsSimt>"),
        )

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

    def test_arena_tag_helpers_do_not_cross_aicore_simt_callable_domains(self) -> None:
        """A SIMT callee cannot call an AICore helper, even if both inline."""
        for signature in (
            "__aicore__ inline uint32_t MakeTag",
            "__aicore__ inline ArenaState StateOf",
            "__aicore__ inline uint32_t GenerationOf",
            "__simt_callee__ inline uint32_t SimtMakeTag",
            "__simt_callee__ inline ArenaState SimtStateOf",
            "__simt_callee__ inline uint32_t SimtGenerationOf",
        ):
            self.assertIn(signature, self.arena_tag)
        # The named SIMT transitions must call only their SIMT siblings.  The
        # CANN compiler rejects even a trivial call from `__simt_callee__` to
        # `__aicore__`; this gate catches that specific regression before a
        # costly isolated OPP build.
        for signature in (
            "SimtMakeTag(",
            "SimtStateOf(",
            "SimtGenerationOf(",
        ):
            self.assertGreaterEqual(self.arena_tag.count(signature), 2)
        forbidden_call = re.compile(
            r"(?<![A-Za-z0-9_])(?:MakeTag|StateOf|GenerationOf)\s*\("
        )
        simt_bodies = self._simt_callee_bodies(self.arena_tag)
        self.assertGreaterEqual(len(simt_bodies), 9)
        for body in simt_bodies:
            body_without_comments = re.sub(
                r"//[^\n]*|/\*.*?\*/", "", body, flags=re.DOTALL
            )
            self.assertIsNone(forbidden_call.search(body_without_comments))
        # Keep the linter itself honest: a direct AICore helper call must be
        # detected rather than merely relying on a source-token count.
        self.assertIsNotNone(
            forbidden_call.search("return MakeTag(1U, kArenaIdle);")
        )

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

    def test_stage_b_tiler_reports_its_pre_workspace_rejection_reason(self) -> None:
        """A workspace-size failure must identify the selected tiler path.

        These messages are diagnostic-only: they do not relax geometry, add
        Tensor refs, or touch C8/P8 execution ordering.  They turn the next
        hardware run into a single decisive observation instead of a series
        of speculative ABI edits.
        """
        for token in (
            "P8_TILING_DIAG_REVISION=p8_tiling_diag_v1",
            "P8_TILING_ENTER",
            "P8_TILING_REJECT",
            "P8_TILING_GEOMETRY",
            "null_input_shape",
            "null_input_desc",
            "input_dtype",
            "geometry_contract",
        ):
            self.assertIn(token, self.host)

    def test_runner_proves_the_fresh_stage_b_tiler_and_28_ref_abi(self) -> None:
        """Never run Stage-B against a merely existing, stale OPP package."""
        for token in (
            "P8_STAGE_B_TILING_ARTIFACT",
            "P8_STAGE_B_TILING_ARTIFACT_CHECK",
            "stage_b_tiling_artifact_provenance_failed",
            "P8_TILING_DIAG_REVISION=p8_tiling_diag_v1",
            "P8_STAGE_B_ACLNN_TENSOR_REFS",
            "libcust_opmaster_rt2.0.so",
            "aclnn_a5_p8_forward_owned_executor.h",
        ):
            self.assertIn(token, self.runner)
        # Same-name mutable outputs are CANN Ref descriptors; appending them
        # would incorrectly turn the proven 28-ref ABI into a 47-ref ABI.
        self.assertIn('!= "28"', self.runner)
        self.assertNotIn("P8_STAGE_B_ACLNN_TENSOR_REFS\" != \"47", self.runner)

    def test_p8_uses_a_source_owned_lazy_cann_dependency_directory(self) -> None:
        """Dependency downloads are reusable inputs, never stale build outputs."""
        for token in (
            'P8_CANN_3RD_LIB_PATH="${P8_REFERENCE_VLLM_ROOT}/csrc/third_party"',
            'P8_TOOLCHAIN_CACHE mode=source_owned_lazy_fetch root=%s',
            'A5_OFFICIAL_C8_CANN_3RD_LIB_PATH="$P8_CANN_3RD_LIB_PATH"',
            'unset NANOVLLM_A5_CANN_3RD_LIB_PATH',
        ):
            self.assertIn(token, self.runner)
        self.assertNotIn('cann_third_party_cache_incomplete:', self.runner)
        self.assertNotIn(
            'NANOVLLM_A5_CANN_3RD_LIB_PATH="$P8_CANN_3RD_LIB_PATH"',
            self.runner,
        )
        for token in (
            'mkdir -p "${CANN_3RD_LIB_PATH}"',
            'P3_REFERENCE_TOOLCHAIN mode=source_owned_lazy_fetch',
            '--cann_3rd_lib_path="${CANN_3RD_LIB_PATH}"',
            'CMAKE_HOME_DIRECTORY:INTERNAL',
            'reason=foreign_cmake_cache',
            'P3_REFERENCE_INCREMENTAL',
        ):
            self.assertIn(token, self.reference_build)
        for forbidden in (
            'cann_third_party_cache_incomplete',
            'STAGED_MAKESELF_DIR="${SOURCE_ROOT}/csrc/third_party/makeself"',
            'P3_REFERENCE_TOOLCHAIN_MAKESELF_STAGED',
            'reason=makeself_stage_failed',
        ):
            self.assertNotIn(forbidden, self.reference_build)


if __name__ == "__main__":
    unittest.main(verbosity=2)
