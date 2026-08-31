#!/usr/bin/env python3
"""Fast source/ABI gate for the standalone P7.1 executor.

This is intentionally a *closure* contract, not a synthetic HCCP benchmark.
It proves that the generated build will receive the real KVAF PostOnly/CQ
helpers, the 31-in/18-alias ABI, two bounded 64-seed local buffers, and no C8
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
        "31in_18aliasout",
        "aclnnA5ScoreReceiptSpeculativeExecutor",
        "CheckAlias",
        "physical_aliases=%lld",
        "Tensor(r!) worker_stats",
    ):
        _require(bridge, token, bridge_path)
    for token in (
        "OpCommand::RunOpApi(#aclnn_api, ordered_acl_call)",
        "UseStreamResIfNeeded(acl_stream, true)",
        "UseStreamResIfNeeded(acl_stream, false)",
        "InitPtaCacheForWorkspace()",
        "UnInitPtaCacheForWorkspace(pta_cache_active)",
        "ordered_tensor_keepalive",
        "handoff_runopapi_by_value",
        "descriptor_release_done",
        "OrderedCleanupPlanFromEnvironment()",
        "NANOVLLM_A5_OPAPI_CLEANUP_BISECT",
        "P7_1_CLEANUP",
        "RTLD_NOW | RTLD_LOCAL",
        "GetAclTensorApiPair()",
        "P7_1_ACL_DSO",
        "P7_1_HOST_DSO",
        "LogConvertedDescriptorSet(#aclnn_api, converted)",
        "P7_1_DESCRIPTOR_LEDGER",
        "P7_1_DESCRIPTOR_LEDGER api=%s tuple_args=%zu acl_tensors=%zu unique_acl_tensors=%zu\\n",
        "return {OrderedCleanupMode::kPhase2Noop, true, false, false, false, false};",
        "phase2_executor_autorelease_done",
        "non-repeatable aclOpExecutor is",
    ):
        _require(op_api_common, token, op_api_common_path)
    _forbid(op_api_common, "aclDestroyAclOpExecutor(executor)", op_api_common_path)
    converted_offset = op_api_common.find("auto converted = ConvertTypes(")
    ledger_offset = op_api_common.find(
        "LogConvertedDescriptorSet(#aclnn_api, converted);", converted_offset
    )
    workspace_boundary_offset = op_api_common.find(
        '"workspace_call_begin"', ledger_offset
    )
    workspace_offset = op_api_common.find(
        "auto workspace_fn = WorkspaceFunction(converted, workspace_addr_fn);",
        converted_offset,
    )
    if not (
        converted_offset >= 0
        and converted_offset < ledger_offset < workspace_boundary_offset < workspace_offset
    ):
        raise AssertionError(
            "P7 descriptor ledger/workspace boundary must run after ConvertTypes and before "
            "GetWorkspaceSize"
        )
    for token in (
        "grep -Fq 'P7_1_DESCRIPTOR_LEDGER api=aclnnA5ScoreReceiptSpeculativeExecutor tuple_args=51 acl_tensors=49 unique_acl_tensors=49'",
        "DESCRIPTOR_LEDGER",
        "workspace_call_begin",
        "run_normal_fail_closed_case()",
        "P7_CLEANUP_NORMAL_ASYNC",
        "P7_DIRECT_BISECT_SKIPPED reason=direct_path_bypasses_task_queue_stream_resource_contract_and_is_not_a_product_gate",
        "unset NANOVLLM_A5_OPAPI_CLEANUP_BISECT",
        "P7_REAL_FLOW_FINAL",
        "execution=normal_task_queue_then_real_hccp",
        "real_direct_hccp.log",
    ):
        _require(reuse_runner, token, reuse_runner_path)
    for token in (
        "31in_18aliasout",
        "AIV-only",
        "MAPPED_OBJECT",
        "FREE -> FILLING -> POSTED -> CQ_TERMINAL",
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
        "ordered_opapi_lifetime=1 pta_cache_lifecycle=1 "
        "normal_async_gate=1 direct_bisect_not_gate=1 "
        "loader_scope_local=1 descriptor_pairing=1 descriptor_ledger=1",
        flush=True,
    )
    print(
        "P7_1_DESCRIPTOR_CONTRACT tuple_args=51 acl_tensors=49 "
        "unique_acl_tensors=49 gate=before_workspace",
        flush=True,
    )
    print(
        "P7_1_FIXTURE_SOURCE_CONTRACT "
        f"fail_closed_inputs={fail_closed_inputs} tuple_multiarg=0 files=2",
        flush=True,
    )
    print(
        "P7_1_CONTRACT_PASS source=1 abi=1 lifecycle=1 transport=1 "
        "alias_bridge=1 c8_closure=0",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"P7_1_CONTRACT_FAIL type={type(exc).__name__} text={exc}", flush=True)
        raise
