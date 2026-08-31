#!/usr/bin/env python3
"""Fast source/ABI gate for the standalone P7.1 executor.

This is intentionally a *closure* contract, not a synthetic HCCP benchmark.
It proves that the generated build will receive the real KVAF PostOnly/CQ
helpers, the 31-tensor/18-Ref ACLNN ABI, two bounded 64-seed local buffers, and no C8
or fixed-core source dependency.  The companion runtime test additionally
executes the bridge in fail-closed and, when supplied, real-control modes.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


REQUIRED_KERNEL = (
    '#include "simt_api/asc_simt.h"',
    "TryAcquireKvafStreamingBatchGetQueueFromAicore",
    "LoadKvafStreamingBatchGetQueueTemplate",
    "PublishKvafStreamingBatchGetDescriptorScratch",
    "BeginKvafStreamingBatchGet",
    "TryRetireKvafStreamingBatchGet",
    "FinishKvafStreamingBatchGet",
    "GetKvafStreamingBatchGetReady",
    "ReleaseKvafStreamingBatchGetQueueFromAicore",
    "DataSyncBarrier<MemDsbT::DDR>",
    "KERNEL_TYPE_AIV_ONLY",
    "privateReadyEpoch_",
    "privateReadyGeneration_",
    "ClearTerminalFailedMappingSimt",
    "LoadGmInt32ForSimt",
    "LoadGmInt64ForSimt",
    "__simt_callee__ inline int32_t LoadGmInt32ForSimt",
    "__simt_callee__ inline int64_t LoadGmInt64ForSimt",
    "event_t simtEvent_ = static_cast<event_t>(EVENT_ID0);",
    "TPipe* pipe_ = nullptr;",
    "TPipe pipe;",
    "A5ScoreReceiptSpeculativeExecutorKernel op(&pipe, &tilingData);",
    "Fail-closed ordering is deliberate.",
    "seedBuffers_[0].Get<int32_t>().GetPhyAddr()",
    "leaseBuffer_.Get<uint8_t>().GetPhyAddr()",
    "claimBuffer_.Get<uint8_t>().GetPhyAddr()",
)
FORBIDDEN_KERNEL = (
    "KERNEL_TYPE_MIX_AIC_1_2",
    "kGetLane",
    "LI_CORE",
    "GET_CORE",
    "quant_lightning",
    "a5_streaming_quant_li_get",
    "a5_official_c8",
    "ReadGmByPassDCache<int64_t>(private_ready_epoch",
    "ReadGmByPassDCache<int64_t>(private_ready_generation",
    "ReadGmByPassDCache<int32_t>(private_ready +",
    "!ASCEND_IS_AIV",
    "static constexpr event_t kSimtEvent",
    "TPipe pipe_;",
    "pipe_.InitBuffer",
    "seedBuffers_[0].Get<int32_t>();",
    "seedBuffers_[1].Get<int32_t>();",
    "leaseBuffer_.Get<uint8_t>());",
    "claimBuffer_.Get<uint8_t>());",
)


def _require(text: str, token: str, path: Path) -> None:
    if token not in text:
        raise AssertionError(f"missing token={token!r} file={path}")


def _forbid(text: str, token: str, path: Path) -> None:
    if token in text:
        raise AssertionError(f"forbidden token={token!r} file={path}")


def _check_fixture_source(path: Path, *, require_fail_closed_shape: bool) -> int:
    """Reject a Python fixture bug before an expensive OPP compilation."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    invalid_tuple_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "tuple"
        and len(node.args) > 1
    ]
    if invalid_tuple_calls:
        first = invalid_tuple_calls[0]
        raise AssertionError(
            f"builtin tuple() accepts one iterable, got {len(first.args)} positional "
            f"arguments at {path}:{first.lineno}"
        )
    if not require_fail_closed_shape:
        return 0

    fixture = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_fail_closed_fixture"
        ),
        None,
    )
    if fixture is None:
        raise AssertionError(f"missing _fail_closed_fixture in {path}")
    tuple_assignments = [
        node
        for node in ast.walk(fixture)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "inputs" for target in node.targets)
        and isinstance(node.value, ast.Tuple)
    ]
    if len(tuple_assignments) != 1 or len(tuple_assignments[0].value.elts) != 31:
        count = len(tuple_assignments[0].value.elts) if tuple_assignments else -1
        raise AssertionError(
            f"_fail_closed_fixture must assign one 31-element tuple literal, got {count}"
        )
    returns_inputs = any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Name)
        and node.value.id == "inputs"
        for node in ast.walk(fixture)
    )
    if not returns_inputs:
        raise AssertionError("_fail_closed_fixture must return its checked inputs tuple")
    return len(tuple_assignments[0].value.elts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ops-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="nanovllm_a5_ops root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ops_root = Path(args.ops_root).resolve()
    p7_root = ops_root / "a5_score_receipt_speculative_executor"
    spec_path = p7_root / "ops.json"
    host_path = p7_root / "op_host" / "a5_score_receipt_speculative_executor.cpp"
    kernel_path = p7_root / "op_kernel" / "a5_score_receipt_speculative_executor.cpp"
    tiling_path = (
        p7_root / "op_kernel" / "a5_score_receipt_speculative_executor_tiling.h"
    )
    bridge_path = (
        p7_root
        / "torch_extension"
        / "csrc"
        / "score_receipt_speculative_executor.cpp"
    )
    op_api_common_path = ops_root / "torch_extension" / "csrc" / "op_api_common.h"
    runtime_package_path = (
        p7_root
        / "torch_extension"
        / "nanovllm_score_receipt_speculative_executor"
        / "__init__.py"
    )
    runtime_test_path = ops_root / "tests" / "test_score_receipt_speculative_executor.py"
    reuse_runner_path = ops_root / "tests" / "run_p7_1_test_only_artifact_reuse.sh"
    full_runner_path = ops_root / "tests" / "run_p7_1_speculative_executor.sh"
    diag_runner_path = ops_root / "tests" / "run_p7_1_fail_closed_dispatch_diag.sh"
    real_fixture_path = p7_root / "torch_extension" / "p7_1_real_direct_hccp_fixture.py"
    abi_path = p7_root / "P7_1_ABI.md"
    for path in (
        spec_path,
        host_path,
        kernel_path,
        tiling_path,
        bridge_path,
        op_api_common_path,
        runtime_package_path,
        runtime_test_path,
        reuse_runner_path,
        full_runner_path,
        diag_runner_path,
        real_fixture_path,
        abi_path,
    ):
        if not path.is_file():
            raise AssertionError(f"missing required P7.1 file={path}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if len(spec) != 1 or spec[0].get("op") != "A5ScoreReceiptSpeculativeExecutor":
        raise AssertionError("ops.json must contain exactly A5ScoreReceiptSpeculativeExecutor")
    inputs = spec[0].get("input_desc", [])
    outputs = spec[0].get("output_desc", [])
    if len(inputs) != 31 or len(outputs) != 18:
        raise AssertionError(
            f"P7.1 ABI must be 31 inputs / 18 outputs, got {len(inputs)}/{len(outputs)}"
        )
    if [inputs[5]["name"], inputs[6]["name"]] != [
        "source_lens",
        "kv_object_address_domain",
    ]:
        raise AssertionError("P7.1 ABI input 5/6 must be source_lens/address_domain")
    mutable_input_indices = (
        11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29,
    )
    expected_ref_names = [inputs[index]["name"] for index in mutable_input_indices]
    actual_ref_names = [output["name"] for output in outputs]
    if actual_ref_names != expected_ref_names:
        raise AssertionError(
            "P7.1 output ports must reuse their mutable input names so CANN emits "
            f"31 tensor Ref ABI, got outputs={actual_ref_names} expected={expected_ref_names}"
        )

    kernel = kernel_path.read_text(encoding="utf-8")
    host = host_path.read_text(encoding="utf-8")
    tiling = tiling_path.read_text(encoding="utf-8")
    bridge = bridge_path.read_text(encoding="utf-8")
    op_api_common = op_api_common_path.read_text(encoding="utf-8")
    reuse_runner = reuse_runner_path.read_text(encoding="utf-8")
    full_runner = full_runner_path.read_text(encoding="utf-8")
    diag_runner = diag_runner_path.read_text(encoding="utf-8")
    abi = abi_path.read_text(encoding="utf-8")
    for token in REQUIRED_KERNEL:
        _require(kernel, token, kernel_path)
    for token in FORBIDDEN_KERNEL:
        _forbid(kernel, token, kernel_path)
    for ref_name in expected_ref_names:
        _require(kernel, f"GM_ADDR {ref_name}_ref", kernel_path)
        _require(kernel, f"(void){ref_name}_ref", kernel_path)
    # The CANN entry macro itself materializes tiling from GM.  A malformed
    # fused_get_control must therefore be rejected in the global entry *before*
    # GET_TILING_DATA, not merely at the beginning of Kernel::Init after the
    # tiling load has already happened.  Init repeats the check defensively,
    # but it is not the primary no-side-effect boundary.
    entry_control_gate = kernel.find("auto* entryControl")
    entry_control_value = kernel.find("entryControlValue", entry_control_gate)
    entry_get_tiling = kernel.find("GET_TILING_DATA(tilingData, tiling)")
    init_control_gate = kernel.find("auto* initControl")
    init_control_validation = kernel.find(
        "IsKvafStreamingBatchGetControlValidForAicore(\n                control_)"
    )
    first_tiling_access = kernel.find("worker_ >= tiling_->workerCount")
    first_epoch_access = kernel.find("ReadGmByPassDCache<int64_t>(streamingEpoch_)")
    first_domain_access = kernel.find("ReadGmByPassDCache<int32_t>(kvObjectAddressDomain_)")
    first_barrier = kernel.find("DataSyncBarrier<MemDsbT::DDR>()")
    first_stat_access = kernel.find("AddStat(kStatControlRejected")
    first_ub_allocation = kernel.find("pipe_->InitBuffer(seedBuffers_[0]")
    if min(entry_control_gate, entry_control_value, entry_get_tiling,
           init_control_gate, init_control_validation, first_tiling_access,
           first_epoch_access, first_domain_access, first_barrier,
           first_stat_access, first_ub_allocation) < 0 or not (
               entry_control_gate < entry_control_value < entry_get_tiling
               and init_control_gate < init_control_validation < first_tiling_access
               and init_control_validation < first_epoch_access
               and init_control_validation < first_domain_access
               and init_control_validation < first_barrier
               and init_control_validation < first_stat_access
               and init_control_validation < first_ub_allocation):
        raise AssertionError(
            "invalid control must be rejected in the global entry before tiling, "
            "then revalidated before GM transaction state, DDR barrier, worker "
            "statistics, or UB allocation"
        )
    for token in (
        "kInputCount",
        "kOutputCount",
        "source_lens",
        "kv_object_address_domain",
        "OP_ADD(A5ScoreReceiptSpeculativeExecutor)",
    ):
        _require(host, token, host_path)
    for token in (
        "kA5ScoreReceiptExecutorMicrobatchCapacity = 64U",
        "kA5ScoreReceiptExecutorMicrobatchSlots = 2U",
    ):
        _require(tiling, token, tiling_path)
    for token in (
        "31tensor_18refout",
        "torch_schema",
        "aclnnA5ScoreReceiptSpeculativeExecutor",
        "P7_1_REF_ABI",
        "mutable_refs=18",
        "Tensor(r!) worker_stats",
        "EXEC_NPU_CMD_ORDERED(",
        "ReserveEnqueued(",
        "MarkHostHandoffAccepted(",
        "P7_1_ORDERED_SUBMISSION",
        "P7_1_ORDERED_HOST_GATE",
        "GetTaskQueueEnable()",
        "isSyncLaunchStream()",
        "p7_submission_plan.bisect",
        "P7_1_BOUNDARY stage=run_npu_after_submit",
        "P7_1_BOUNDARY stage=run_npu_result_ready",
    ):
        _require(bridge, token, bridge_path)
    if "empty_like" in bridge:
        raise AssertionError(
            "P7.1 bridge must return the 18 mutable Torch aliases, not allocate "
            "independent output descriptors"
        )
    for token in (
        "#define EXEC_NPU_CMD_ORDERED",
        "InitPtaCacheForWorkspace()",
        "UnInitPtaCacheForWorkspace(pta_cache_active)",
        "MakeP7DirectLaunchLease",
        "ReserveEnqueued(std::shared_ptr<P7DirectLaunchLeaseBase> lease,",
        "MarkHostHandoffAccepted",
        "MarkHostFailedNoThrow(uint64_t token, aclrtStream stream) noexcept",
        "OrderedLaunchFailureObserver",
        "NotifyOrderedLaunchFailure",
        "PrePublishAbortAction",
        "ResolvePrePublishAbortNoThrow(",
        "kCancelledEnqueued",
        "kFailedBeforePublishQuarantined",
        "kUnexpectedHandoffAccepted",
        "PrePublishAbortActionName(",
        "ReleaseP7DirectLaunchLeasesAtForwardTerminal(",
        "TakeP7DirectLaunchTokenForCurrentThread",
        "ClearP7DirectLaunchTokenForCurrentThread",
        "PublishP7DirectLaunchTokenForCurrentThread",
        "stream_key",
        "handoff_runopapi_by_value",
        "UseStreamResIfNeeded(acl_stream, true)",
        "UseStreamResIfNeeded(acl_stream, false)",
        "torch_npu/csrc/core/npu/register/OptionsManager.h",
        "RTLD_NOW | RTLD_LOCAL",
        "GetAclTensorApiPair()",
        "P7_1_ACL_DSO",
        "P7_1_HOST_DSO",
        "LogConvertedDescriptorSet(#aclnn_api, converted)",
        "P7_1_DESCRIPTOR_LEDGER",
        "P7_1_DESCRIPTOR_LEDGER api=%s tuple_args=%zu acl_tensors=%zu unique_acl_tensors=%zu\\n",
        "static auto *registry = new P7DirectLaunchLeaseRegistry();",
        "process-lifetime instead",
    ):
        _require(op_api_common, token, op_api_common_path)
    _forbid(op_api_common, "aclDestroyAclOpExecutor(executor)", op_api_common_path)
    if "EXEC_NPU_CMD_P7_ZERO_WORKSPACE_DIRECT_ASYNC" in op_api_common or \
       "EXEC_NPU_CMD_P7_ZERO_WORKSPACE_DIRECT_ASYNC" in bridge:
        raise AssertionError(
            "P7 must not retain or invoke a raw-current-stream ACLNN macro; "
            "the only allowed host path is EXEC_NPU_CMD_ORDERED"
        )
    if "pending.token <= token" in op_api_common:
        raise AssertionError(
            "P7 terminal release must reclaim the exact token only; a later "
            "forward must never release an earlier queued callback's inputs"
        )
    abort_pos = bridge.find("ResolvePrePublishAbortNoThrow(")
    abort_log_pos = bridge.find("P7_1_DISPATCH_ABORT", abort_pos)
    publish_pos = bridge.find("PublishP7DirectLaunchTokenForCurrentThread(")
    ordered_submit_pos = bridge.find("EXEC_NPU_CMD_ORDERED(")
    host_gate_pos = bridge.find("P7_1_ORDERED_HOST_GATE")
    reserve_pos = bridge.find("ReserveEnqueued(")
    if min(abort_pos, abort_log_pos, publish_pos, ordered_submit_pos,
           host_gate_pos, reserve_pos) < 0 or not (
        host_gate_pos < reserve_pos < ordered_submit_pos < abort_pos
        < abort_log_pos < publish_pos
    ):
        raise AssertionError(
            "P7 must reject inline host execution before lease reservation, "
            "then resolve a synchronous pre-publish dispatch abort after "
            "ordered submission and before the terminal token is published"
        )
    runtime_package = runtime_package_path.read_text(encoding="utf-8")
    for token in (
        "class ForwardTerminalLease",
        "capture_forward_terminal_lease",
        "release_at_forward_terminal",
        "p7_1_take_launch_token",
        "p7_1_release_after_forward_terminal(int(lease.token))",
        "aclTensor`` descriptors have already been released immediately",
    ):
        _require(runtime_package, token, runtime_package_path)
    for forbidden in ("p7_1_release_after_forward_terminal()",):
        _forbid(runtime_package, forbidden, runtime_package_path)
    runtime_test = runtime_test_path.read_text(encoding="utf-8")
    for token in (
        "P7_1_TEST_EXIT_AFTER_RECEIPT",
        "P7_1_TEST_PROCESS_EXIT_DIAG",
        "P7_1_FINAL_RECEIPT",
        "P7_1_PROCESS_EXIT stage=python_atexit",
        "os._exit(0)",
        "after every P7 assertion, terminal release, and real",
        "_install_valid_zero_candidate_control",
        "_enqueue_invalid_control_predecessor",
        "P7_1_ORDERING_SENTINEL",
        "P7_1_ORDERING_RECEIPT",
        "predecessor=control_zero_queued",
        "required_consumer_observation=invalid_control",
    ):
        _require(runtime_test, token, runtime_test_path)
    snapshot_pos = runtime_test.find("snapshots = _snapshot_mutable_refs(inputs)")
    install_pos = runtime_test.find("_install_valid_zero_candidate_control(inputs)")
    enqueue_pos = runtime_test.find("_enqueue_invalid_control_predecessor(inputs)")
    dispatch_pos = runtime_test.find(
        "torch.ops.a5_score_receipt_speculative_executor.score_receipt_speculative_executor"
    )
    if min(snapshot_pos, install_pos, enqueue_pos, dispatch_pos) < 0 or not (
        snapshot_pos < install_pos < enqueue_pos < dispatch_pos
    ):
        raise AssertionError(
            "fail-closed runtime test must establish valid zero-work control, "
            "queue control.zero_ without synchronizing, then submit P7"
        )
    for token in (
        "grep -Fq 'P7_1_DESCRIPTOR_LEDGER api=aclnnA5ScoreReceiptSpeculativeExecutor tuple_args=33 acl_tensors=31 unique_acl_tensors=31'",
        "DESCRIPTOR_LEDGER",
        "stage=workspace_call_begin",
        "stage=descriptor_release_done",
        "P7_1_ORDERED_SUBMISSION token=",
        "run_ordered_async_fail_closed_case()",
        "run_ordered_async_teardown_probe()",
        "run_real_direct_hccp_case()",
        "P7_1_TEST_EXIT_AFTER_RECEIPT=1",
        "P7_ORDERED_ASYNC_FAIL_CLOSED",
        "P7_ORDERED_ASYNC_TEARDOWN_PROBE",
        "P7_REAL_DIRECT_HCCP",
        "P7_1_TERMINAL_LEASE",
        "P7_1_FAIL_CLOSED_NOOP",
        "P7_1_ORDERING_RECEIPT",
        "P7_1_BOUNDARY stage=torch_op_return",
        "P7_1_BOUNDARY stage=stream_terminal_done",
        "P7_1_FINAL_RECEIPT",
        "P7_1_TEST_PROCESS_EXIT_DIAG=1",
        "P7_REAL_FLOW_FINAL",
        "execution=framework_ordered_submit_then_device_post_cq",
        "real_direct_hccp.log",
        "NANOVLLM_A5_OPAPI_CLEANUP_BISECT",
        "NANOVLLM_A5_OPAPI_DEBUG=1",
        "P7_ARTIFACT_RUNTIME_EXACT",
        "artifact_exact=%s",
        "P7_1_ORDERING_RECEIPT",
        "fail_closed_ordered_async.log",
        "emit_failure_excerpt()",
    ):
        _require(reuse_runner, token, reuse_runner_path)
    _forbid(reuse_runner, "fail_closed_direct_async.log", reuse_runner_path)
    for token in (
        "P7_1_TEST_EXIT_AFTER_RECEIPT=1",
        "P7_1_ORDERING_RECEIPT",
        "P7_FAIL_CLOSED_ORDERING_RECEIPT",
        "NANOVLLM_A5_OPAPI_CLEANUP_BISECT",
        "NANOVLLM_A5_OPAPI_DEBUG=1",
        "emit_failure_excerpt()",
    ):
        _require(full_runner, token, full_runner_path)
    # Every runner that accepts a pre-built OPP has the same exact-provenance
    # rule.  Otherwise a diagnostic invocation can silently restore the old
    # "ancestor OPP + newer bridge" mixed-ABI path that the reuse runner now
    # rejects.
    for token in (
        "P7_ARTIFACT_RUNTIME_EXACT",
        '"${P7_ARTIFACT_VLLM_SHA}" == "${P7_RUNTIME_VLLM_SHA}"',
        "P7_KVAF_BUILD_HEAD_MATCH",
        ".p7_kvaf_source_head",
        "artifact_exact=%s",
        "unset NANOVLLM_A5_OPAPI_CLEANUP_BISECT NANOVLLM_A5_OPAPI_CLEANUP_MODE",
        'contract["mutable_ref_outputs"]',
    ):
        _require(diag_runner, token, diag_runner_path)
    if '"${P7_ARTIFACT_ANCESTOR}" == "1"' in diag_runner:
        raise AssertionError(
            "P7 diagnostic runner must never admit an ancestor OPP; it must "
            "require an exact runtime source match"
        )
    for token in (
        "31tensor_18refout",
        "torch_schema=31in_18aliasout",
        "AIV-only",
        "MAPPED_OBJECT",
        "FREE -> FILLING -> POSTED -> CQ_TERMINAL",
        "framework-ordered ACLNN 入流",
        "forward stream terminal",
    ):
        _require(abi, token, abi_path)

    fail_closed_inputs = _check_fixture_source(
        runtime_test_path, require_fail_closed_shape=True
    )
    _check_fixture_source(real_fixture_path, require_fail_closed_shape=False)

    print(
        "P7_1_SOURCE_ABI "
        "op=A5ScoreReceiptSpeculativeExecutor inputs=31 outputs=18 "
        "candidate_seed_shape=[N,128,4] source_lens=[R] "
        "address_domain=1_mapped_object local_microbatches=2x64",
        flush=True,
    )
    print(
        "P7_1_LIFECYCLE "
        "queue=FREE>FILLING>POSTED>CQ_TERMINAL>FREE_OR_QUARANTINED "
        "private=FREE>PENDING>READY_OR_FAILED "
        "identity=epoch,generation,post_sequence",
        flush=True,
    )
    print(
        "P7_1_TRANSPORT_CONTRACT "
        "lease=real postonly=real try_retire=real finish=real release=real "
        "c8_dependency=0 fixed_get_core=0 task_type=aiv_only",
        flush=True,
    )
    print(
        "P7_1_CANN_COMPILER_CONTRACT "
        "simt_callee=1 aiv_only_no_runtime_branch=1 "
        "ub_physical_address_conversion=1 nonconst_event_t=1 "
        "entry_owned_tpipe=1 fail_closed_before_ub=1 "
        "fail_closed_control_only=1 pta_cache_lifecycle=1 "
        "host_task_queue=ordered workspace_lifetime=framework_owned "
        "loader_scope_local=1 descriptor_pairing=1 descriptor_ledger=1",
        flush=True,
    )
    print(
        "P7_1_DESCRIPTOR_CONTRACT tuple_args=33 acl_tensors=31 "
        "unique_acl_tensors=31 refs=18 gate=before_workspace",
        flush=True,
    )
    print(
        "P7_1_FIXTURE_SOURCE_CONTRACT "
        f"fail_closed_inputs={fail_closed_inputs} tuple_multiarg=0 files=2",
        flush=True,
    )
    print(
        "P7_1_CONTRACT_PASS source=1 abi=1 lifecycle=1 transport=1 "
        "ref_bridge=1 c8_closure=0",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"P7_1_CONTRACT_FAIL type={type(exc).__name__} text={exc}", flush=True)
        raise
