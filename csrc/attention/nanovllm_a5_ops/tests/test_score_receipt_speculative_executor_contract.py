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
    abi = abi_path.read_text(encoding="utf-8")
    for token in REQUIRED_KERNEL:
        _require(kernel, token, kernel_path)
    for token in FORBIDDEN_KERNEL:
        _forbid(kernel, token, kernel_path)
    for ref_name in expected_ref_names:
        _require(kernel, f"GM_ADDR {ref_name}_ref", kernel_path)
        _require(kernel, f"(void){ref_name}_ref", kernel_path)
    control_validation = kernel.find(
        "IsKvafStreamingBatchGetControlValidForAicore(control_)"
    )
    first_ub_allocation = kernel.find("pipe_->InitBuffer(seedBuffers_[0]")
    if control_validation < 0 or first_ub_allocation < 0 or \
       first_ub_allocation <= control_validation:
        raise AssertionError(
            "invalid KVAF control must be rejected before P7.1 initializes UB buffers"
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
        "EXEC_NPU_CMD_P7_ZERO_WORKSPACE_DIRECT_ASYNC(",
    ):
        _require(bridge, token, bridge_path)
    if "empty_like" in bridge:
        raise AssertionError(
            "P7.1 bridge must return the 18 mutable Torch aliases, not allocate "
            "independent output descriptors"
        )
    for token in (
        "#define EXEC_NPU_CMD_P7_ZERO_WORKSPACE_DIRECT_ASYNC",
        "InitPtaCacheForWorkspace()",
        "UnInitPtaCacheForWorkspace(pta_cache_active)",
        "workspace_size != 0",
        "UseStreamResIfNeeded(acl_stream, true)",
        "MakeP7DirectLaunchLease",
        "Commit(std::shared_ptr<P7DirectLaunchLeaseBase> lease,",
        "ReleaseP7DirectLaunchLeasesAtForwardTerminal(",
        "TakeP7DirectLaunchTokenForCurrentThread",
        "ClearP7DirectLaunchTokenForCurrentThread",
        "PublishP7DirectLaunchTokenForCurrentThread",
        "stream_key",
        "UseStreamResIfNeeded(acl_stream, false)",
        "direct_async_workspace_call_begin",
        "direct_async_launch_done",
        "direct_async_descriptor_lease_held",
        "direct_async_submit_done",
        "RTLD_NOW | RTLD_LOCAL",
        "GetAclTensorApiPair()",
        "P7_1_ACL_DSO",
        "P7_1_HOST_DSO",
        "LogConvertedDescriptorSet(#aclnn_api, converted)",
        "P7_1_DESCRIPTOR_LEDGER",
        "P7_1_DESCRIPTOR_LEDGER api=%s tuple_args=%zu acl_tensors=%zu unique_acl_tensors=%zu\\n",
    ):
        _require(op_api_common, token, op_api_common_path)
    _forbid(op_api_common, "aclDestroyAclOpExecutor(executor)", op_api_common_path)
    direct_macro_offset = op_api_common.find(
        "#define EXEC_NPU_CMD_P7_ZERO_WORKSPACE_DIRECT_ASYNC"
    )
    direct_macro = op_api_common[direct_macro_offset:]
    if direct_macro_offset < 0:
        raise AssertionError("missing P7 direct ACLNN macro")
    for forbidden in (
        "OpCommand::RunOpApi",
        "ordered_acl_call",
        "aclrtSynchronizeStream",
    ):
        _forbid(direct_macro, forbidden, op_api_common_path)
    converted_offset = op_api_common.find(
        "auto converted = ConvertTypes(", direct_macro_offset
    )
    ledger_offset = op_api_common.find(
        "LogConvertedDescriptorSet(#aclnn_api, converted);", direct_macro_offset
    )
    workspace_boundary_offset = op_api_common.find(
        '"direct_async_workspace_call_begin"', ledger_offset
    )
    workspace_offset = op_api_common.find(
        "auto workspace_fn = WorkspaceFunction(converted, workspace_addr_fn);", direct_macro_offset
    )
    if not (
        direct_macro_offset >= 0 and converted_offset >= 0
        and converted_offset < ledger_offset < workspace_boundary_offset < workspace_offset
    ):
        raise AssertionError(
            "P7 descriptor ledger/workspace boundary must run after ConvertTypes and before "
            "GetWorkspaceSize"
        )
    direct_token_clear_offset = op_api_common.find(
        "ClearP7DirectLaunchTokenForCurrentThread();", direct_macro_offset
    )
    direct_resolve_offset = op_api_common.find(
        "ResolveSymbol(#aclnn_api \"GetWorkspaceSize\")", direct_macro_offset
    )
    direct_dequeue_offset = op_api_common.find(
        "UseStreamResIfNeeded(acl_stream, false);", direct_macro_offset
    )
    direct_launch_offset = op_api_common.find(
        "const int launch_status =", direct_macro_offset
    )
    direct_commit_offset = op_api_common.find(
        "Commit(", direct_macro_offset
    )
    direct_publish_offset = op_api_common.find(
        "PublishP7DirectLaunchTokenForCurrentThread(direct_async_token);",
        direct_macro_offset,
    )
    if not (
        direct_macro_offset < direct_token_clear_offset < direct_resolve_offset
        and direct_dequeue_offset < direct_launch_offset < direct_commit_offset
        < direct_publish_offset
    ):
        raise AssertionError(
            "P7 direct launch must clear stale tokens before resolver work, bind "
            "torch_npu dequeue stream resources before launch, then publish the "
            "stream/token lease only after a successful enqueue"
        )
    runtime_package = runtime_package_path.read_text(encoding="utf-8")
    for token in (
        "class ForwardTerminalLease",
        "capture_forward_terminal_lease",
        "release_at_forward_terminal",
        "p7_1_take_launch_token",
        "p7_1_release_after_forward_terminal(int(lease.token))",
    ):
        _require(runtime_package, token, runtime_package_path)
    for forbidden in ("p7_1_release_after_forward_terminal()",):
        _forbid(runtime_package, forbidden, runtime_package_path)
    for token in (
        "grep -Fq 'P7_1_DESCRIPTOR_LEDGER api=aclnnA5ScoreReceiptSpeculativeExecutor tuple_args=33 acl_tensors=31 unique_acl_tensors=31'",
        "DESCRIPTOR_LEDGER",
        "direct_async_workspace_call_begin",
        "run_direct_async_fail_closed_case()",
        "P7_DIRECT_ASYNC_FAIL_CLOSED",
        "P7_1_TERMINAL_LEASE",
        "P7_REAL_FLOW_FINAL",
        "execution=direct_current_stream_then_real_hccp",
        "real_direct_hccp.log",
    ):
        _require(reuse_runner, token, reuse_runner_path)
    for token in (
        "31tensor_18refout",
        "torch_schema=31in_18aliasout",
        "AIV-only",
        "MAPPED_OBJECT",
        "FREE -> FILLING -> POSTED -> CQ_TERMINAL",
        "直接 ACLNN 入流",
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
        "direct_current_stream_enqueue=1 pta_cache_lifecycle=1 "
        "host_task_queue=0 zero_workspace_gate=1 "
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
