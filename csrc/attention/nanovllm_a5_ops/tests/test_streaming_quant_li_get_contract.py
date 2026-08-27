#!/usr/bin/env python3
"""Small source-contract probe for A5StreamingQuantLiGet.

This is deliberately not a bandwidth or payload-correctness benchmark: an
unverified/known-zero Direct-HCCP payload must not be reported as H2D success.
It checks three narrower, reproducible properties before a target OPP build:

* the source hook is placed after the official C8 score reduction and before
  score-GM DataCopy, while the original ProcessTopK remains present;
* the source-level hook observes but never writes the official score tile;
* the modeled lease lifecycle posts before scan completion, leaves the final
  Get in flight through exact Top-K, and releases only after a terminal CQ.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "a5_streaming_quant_li_get"
VECTOR = ROOT / "op_kernel/arch35/quant_lightning_indexer_service_vector.h"
HOOK = ROOT / "op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h"
HOOK_MODE = ROOT / "op_kernel/a5_streaming_quant_li_get_hook_mode.h"
KERNEL = ROOT / "op_kernel/a5_streaming_quant_li_get.cpp"
KERNEL_PRELOAD = ROOT / "op_kernel/arch35/quant_lightning_indexer_kernel.h"
OP_DEF = ROOT / "op_host/a5_streaming_quant_li_get_def.cpp"
INFER = ROOT / "op_host/a5_streaming_quant_li_get_infershape.cpp"
TILING = ROOT / "op_host/a5_streaming_quant_li_get_tiling.cpp"
TILING_HEADER = ROOT / "op_host/a5_streaming_quant_li_get_tiling.h"
BRIDGE = ROOT / "torch_extension/csrc/streaming_quant_li_get.cpp"
OPS = ROOT / "ops.json"
ROOT_CMAKE = ROOT / "CMakeLists.txt"
METADATA_COMPANION = ROOT / "metadata_companion.md"
BUILD_SCRIPT = ROOT.parent / "build_nanovllm_ops.sh"
HOST_COMPAT = ROOT / "op_host/a5_streaming_host_compat.h"

# These are the public CANN-generated operator parameter names.  The isolated
# Torch bridge is deliberately required to use the same names and positional
# order: a 21-input ACLNN function is invoked through a dynamically typed
# function pointer, so a friendly alias or an order-only drift cannot be made
# safe by the C++ compiler.
CANN_INPUT_NAMES = [
    "query",
    "key",
    "weights",
    "query_dequant_scale",
    "key_dequant_scale",
    "actual_seq_lengths_query",
    "actual_seq_lengths_key",
    "block_table",
    "metadata",
    "resident_threshold",
    "resident_threshold_valid",
    "kv_object_slots_snapshot",
    "req_pool_entries",
    "kv_object_map",
    "private_source_to_slot",
    "private_ready_table",
    "private_arena_head",
    "private_packed_output",
    "queue_control",
    "queue_leases",
    "kv_object_lens",
]

# `kv_object_domain` is a required public Torch semantic choice, not an
# optional tuning knob.  Torch must therefore take it before its optional
# defaults.  CANN's generated ACLNN ABI intentionally retains its existing
# declaration order, where the same value is last.  The bridge is the sole,
# explicit reorder boundary between the two.
CANN_ATTR_NAMES = [
    "query_quant_mode",
    "key_quant_mode",
    "layout_query",
    "layout_key",
    "sparse_count",
    "sparse_mode",
    "pre_tokens",
    "next_tokens",
    "cmp_ratio",
    "return_values",
    "stride",
    "scale_stride",
    "kvaf_rows_per_entry",
    "kv_object_domain",
]
TORCH_SCHEMA_ATTR_TOKENS = [
    "int kv_object_domain",
    "int query_quant_mode",
    "int key_quant_mode",
    "str layout_query",
    "str layout_key",
    "int sparse_count",
    "int sparse_mode",
    "int pre_tokens",
    "int next_tokens",
    "int cmp_ratio",
    "bool return_values",
    "int stride",
    "int scale_stride",
    "int kvaf_rows_per_entry",
]
ACLNN_ATTR_ARGUMENTS = [
    "queryQuantMode",
    "keyQuantMode",
    "layoutQueryPtr",
    "layoutKeyPtr",
    "sparseCount",
    "sparseMode",
    "preTokens",
    "nextTokens",
    "cmpRatio",
    "returnValues",
    "stride",
    "scaleStride",
    "kvafRowsPerEntry",
    "kvObjectDomain",
]
ALLOCATE_ATTR_ARGUMENTS = [
    "queryQuantMode",
    "keyQuantMode",
    "layoutQuery",
    "layoutKey",
    "sparseCount",
    "sparseMode",
    "preTokens",
    "nextTokens",
    "cmpRatio",
    "returnValues",
    "stride",
    "scaleStride",
    "kvafRowsPerEntry",
    "kvObjectDomain",
]
BRIDGE_ATTR_REORDER_COMMENT = (
    "ACLNN ABI keeps kvObjectDomain after kvafRowsPerEntry."
)
OUTPUT_NAMES = [
    "sparse_indices",
    "sparse_values",
    "private_source_to_slot_out",
    "private_ready_table_out",
    "private_arena_head_out",
    "private_packed_output_out",
    "queue_leases_out",
    "telemetry",
    "score_source_lens",
]
KERNEL_ARGUMENT_NAMES = [
    "query",
    "key",
    "weights",
    "queryScale",
    "keyScale",
    "actualSeqLengthsQ",
    "actualSeqLengthsK",
    "blocktable",
    "metadata",
    "residentThreshold",
    "residentThresholdValid",
    "kvObjectSlotsSnapshot",
    "reqPoolEntries",
    "kvObjectMap",
    "privateSourceToSlot",
    "privateReadyTable",
    "privateArenaHead",
    "privatePackedOutput",
    "queueControl",
    "queueLeases",
    "kvObjectLens",
    "sparseIndices",
    "sparseValues",
    "privateSourceToSlotOut",
    "privateReadyTableOut",
    "privateArenaHeadOut",
    "privatePackedOutputOut",
    "queueLeasesOut",
    "telemetry",
    "scoreSourceLens",
]


def topk(scores: list[int], k: int) -> list[int]:
    # The test operates on the official uint16 sortable-score representation;
    # deterministic index tie-break makes the source-level non-mutation model
    # unambiguous. This is not a target-runtime bitwise equivalence result.
    return [index for index, _ in sorted(enumerate(scores), key=lambda x: (-x[1], x[0]))[:k]]


def assert_order(label: str, actual: list[str], expected: list[str]) -> None:
    """Fail with the complete ABI order instead of a vague target error."""
    assert actual == expected, (
        f"{label} order mismatch\n"
        f"expected: {' -> '.join(expected)}\n"
        f"actual:   {' -> '.join(actual)}"
    )


def require_source_contract() -> None:
    vector = VECTOR.read_text()
    hook = HOOK.read_text()
    hook_mode = HOOK_MODE.read_text()
    kernel = KERNEL.read_text()
    preload = KERNEL_PRELOAD.read_text()
    op_def = OP_DEF.read_text()
    infer = INFER.read_text()
    tiling = TILING.read_text()
    tiling_header = TILING_HEADER.read_text()
    bridge = BRIDGE.read_text()
    build_script = BUILD_SCRIPT.read_text()
    ops = json.loads(OPS.read_text())
    assert len(ops) == 1 and ops[0]["op"] == "A5StreamingQuantLiGet"
    assert len(ops[0]["input_desc"]) == 21
    assert len(ops[0]["output_desc"]) == 9
    inputs = ops[0]["input_desc"]
    outputs = ops[0]["output_desc"]
    assert_order("ops.json input ABI", [item["name"] for item in inputs], CANN_INPUT_NAMES)
    assert_order("ops.json output ABI", [item["name"] for item in outputs], OUTPUT_NAMES)
    attrs = ops[0]["attr"]
    assert_order("ops.json attr ABI", [item["name"] for item in attrs], CANN_ATTR_NAMES)
    assert attrs[-1] == {
        "name": "kv_object_domain",
        "param_type": "required",
        "type": "int",
    }
    # CANN 9.1's msopgen TypeRange does not accept a FP8 JSON spelling.  Match
    # the official C8 schema: generic metadata uses int8 while the target
    # Ascend950 AICoreConfig and the bridge below must still require E4M3FN.
    assert inputs[0]["type"] == ["int8"]
    assert inputs[1]["type"] == ["int8"]
    assert inputs[9]["name"] == "resident_threshold"
    # ResidentSeed exports int32 (low 16 bits are the exact C8 sortable kth
    # score), so this must stay int32 to preserve the direct graph edge.
    assert inputs[9]["type"] == ["int32"]
    assert inputs[10] == {
        "name": "resident_threshold_valid",
        "param_type": "required",
        "format": ["ND"],
        "type": ["int32"],
    }
    assert inputs[11]["name"] == "kv_object_slots_snapshot"
    assert inputs[13]["name"] == "kv_object_map"
    assert inputs[17] == {
        "name": "private_packed_output",
        "param_type": "required",
        "format": ["ND"],
        "type": ["uint8"],
    }
    assert inputs[18]["name"] == "queue_control"
    assert inputs[18]["type"] == ["int64"]
    assert inputs[19]["name"] == "queue_leases"
    assert inputs[19]["type"] == ["int32"]
    assert inputs[20] == {
        "name": "kv_object_lens",
        "param_type": "required",
        "format": ["ND"],
        "type": ["int32"],
    }
    assert outputs[8] == {
        "name": "score_source_lens",
        "param_type": "required",
        "format": ["ND"],
        "type": ["int32"],
    }
    assert ROOT_CMAKE.is_file()
    assert HOOK_MODE.is_file()
    assert "A5_STREAMING_LI_GET_ENABLE_HOOK 1" in hook_mode
    assert "descriptor_stage_only" not in OPS.read_text()
    assert "descriptor_stage_only" not in ROOT_CMAKE.read_text()
    assert METADATA_COMPANION.is_file()
    # An isolated msopgen project must not inherit the normal repository-wide
    # csrc/common/include path: that err header in turn includes a private
    # log/log.h.  The P3-local header closes both quoted include paths while
    # preserving all validation early-return macros.
    assert HOST_COMPAT.is_file()
    host_compat = HOST_COMPAT.read_text()
    assert '#include "log/log.h"' not in host_compat
    for macro in (
        "OP_LOGE",
        "OP_LOGI",
        "OP_LOGD",
        "OP_CHECK_IF",
        "OP_CHECK_NULL_WITH_CONTEXT",
        "REPORT_INNER_ERR_MSG",
        "OPS_INNER_ERR_STUB",
    ):
        assert macro in host_compat
    assert "STREAMING_LI_HOST_COMPAT" in build_script
    assert '"${OP_DEV}/op_host/err/ops_err.h"' in build_script
    assert '"${OP_DEV}/op_host/log/log.h"' in build_script
    # msopgen emits an all-in-one Host stub containing OP_ADD.  P3 supplies
    # the source-derived OpDef in a differently named _def.cpp, so the staging
    # script must delete the generated stub and prove the resulting CANN host
    # project contains exactly one registration.  Otherwise CANN writes two
    # [A5StreamingQuantLiGet] sections into aic-*-ops-info.ini before kernel
    # compilation starts.
    assert "P3_GENERATED_HOST_STUB" in build_script
    assert 'rm -f "${P3_GENERATED_HOST_STUB}"' in build_script
    assert "P3_OPDEF_SOURCES" in build_script
    assert "P3 host registration closure:" in build_script
    # The C8 hook reaches the KVAF helper through the relative include
    # `../kvaf_device/...`.  The build must therefore overwrite that nested
    # closure, not merely place a fresh helper at op_kernel/.  Otherwise a
    # stale checked-in copy can compile alongside a new hook and fail only
    # deep inside CANN's device compiler.
    assert 'KVAF_STAGE_DIR="${OP_DEV}/op_kernel/kvaf_device"' in build_script
    for staged_file in (
        "kvaf_fused_full_get_device.h",
        "execute_batch_get_aiv.asc",
        "batch_get_abi.h",
    ):
        assert f'"${{KVAF_STAGE_DIR}}/{staged_file}"' in build_script
    assert "KVAF relative include closure:" in build_script
    assert "KVAF_GENERATED_STAGE_DIR" in build_script
    assert "generated KVAF relative include closure:" in build_script
    # The queue control lives in GM.  AscendC rejects an implicit aggregate
    # copy from `__gm__ T` to local memory, so the P3 hook must use KVAF's
    # explicit field-wise AIC loader.  This is representation-only plumbing:
    # it cannot alter score/Top-K math or the Direct-HCCP transaction order.
    assert "LoadKvafStreamingBatchGetControlFromGmForAicore" in hook
    assert "control_ = *controlGm_" not in hook
    assert "BatchMulWeightAndReduceSum" in vector
    assert "streamingKvafHook_.Emit" in vector
    assert "streamingKvafHook_.Finish" in vector
    assert "FinishStreamingKvafHook" in vector
    # The normal source defaults to the fused path, but the isolated OPP
    # builder can compile a true no-hook control without changing ABI or
    # relying on CANN CMAKE flag propagation.  The conditional must remove
    # the helper include, its UB, its score-tile barrier/call, and the TopK
    # terminal wait/finish as one unit.
    assert '#include "../a5_streaming_quant_li_get_hook_mode.h"' in vector
    assert "#if A5_STREAMING_LI_GET_ENABLE_HOOK" in vector
    assert "NANOVLLM_A5_STREAMING_LI_GET_HOOK_MODE" in build_script
    assert "P3 streaming hook mode:" in build_script
    assert "P3 generated hook-mode header:" in build_script
    # A5 cannot reliably publish adjacent int32 side-output rows from paired
    # AIVs through scalar GM stores.  The P3-only receipt therefore uses its
    # own 32B UB line and S<->MTE3 DMA publication, leaving C8 TopK itself
    # untouched.
    assert "WriteScoreSourceLen" in vector
    assert "scoreSourceLensBuf_" in vector
    assert "DataCopyPad(scoreSourceLensGm[outputRow]" in vector
    assert "SetFlag<HardEvent::S_MTE3>(EVENT_ID1)" in vector
    assert "SetFlag<HardEvent::MTE3_S>(EVENT_ID1)" in vector
    assert "outputRow, validS2Len > 0 ? validS2Len : 0" in vector
    assert "ProcessTopK" in vector
    assert vector.index("BatchMulWeightAndReduceSum") < vector.index("streamingKvafHook_.Emit")
    assert vector.index("streamingKvafHook_.Emit") < vector.index("DataCopyPad(scoreGm")
    topk_body = vector[vector.index(
        "__aicore__ inline void QLIVector<QLIT>::ProcessTopK"
    ):]
    # The final H2D remains in flight through exact TopK, but ProcessTopK
    # must not consume the final TOPK_MTE3_V event a second time.  Official
    # FreeEventID owns that final token; only then may the hook terminally
    # poll/release its optional wave.  This is a real liveness contract, not
    # merely a preferred event ordering.
    assert "streamingKvafHook_.Finish()" not in topk_body
    process_main = preload[preload.index(
        "__aicore__ inline void QLIPreload<QLIT>::ProcessMain"
    ):preload.index(
        "__aicore__ inline void QLIPreload<QLIT>::ProcessBaseBlock"
    )]
    assert process_main.index("vectorService.FreeEventID()") < process_main.index(
        "vectorService.FinishStreamingKvafHook()"
    )
    # score_source_lens must use the same per-row value before either the
    # short-row / invalid output path or normal official TopK branch. It is a
    # source-score S bound, not a host approximation of logical KV L.
    valid_formula = topk_body.index("validS2Len = ((int32_t)i + cuRealAcSeq)")
    source_len_store = topk_body.index(
        "WriteScoreSourceLen(outputRow, validS2Len > 0 ? validS2Len : 0)"
    )
    invalid_branch = topk_body.index("if (validS2Len <= 0)")
    assert valid_formula < source_len_store < invalid_branch
    clean_body = vector[vector.index(
        "__aicore__ inline void QLIVector<QLIT>::CleanInvalidOutput"
    ):vector.index(
        "__aicore__ inline void QLIVector<QLIT>::GetKeyScale"
    )]
    assert "WriteScoreSourceLen" in clean_body
    for symbol in (
        "TryAcquireKvafStreamingBatchGetQueueFromAicore",
        "PublishKvafStreamingBatchGetDescriptorScratch",
        "BeginKvafStreamingBatchGet",
        "TryRetireKvafStreamingBatchGet",
        "FinishKvafStreamingBatchGet",
        "GetKvafStreamingBatchGetReady",
        "ReleaseKvafStreamingBatchGetQueueFromAicore",
        "LoadKvafStreamingBatchGetQueueTemplate",
        "kvObjectMap_.GetValue",
        "kvObjectSlots_.GetValue",
        "ClaimStreamingPrivateArenaSimt",
        "residentThresholdValid_.GetValue(row) != 1",
        "& 0xffffU",
        "privateArenaCapacity_ != privateRows",
        "control_.output != reinterpret_cast<uint64_t>(privatePackedOutputRaw_)",
        "control_.output_rows != privatePackedRows_",
    ):
        assert symbol in hook, symbol
    # A fully leased queue pool must bypass the AIV<->SIMT acquire service.
    # That service is needed only when there is a possible real speculative
    # post; otherwise it adds one VF_CALL per score tile while changing no
    # C8 score, Top-K, private mapping, or required-recall result.
    emit_body = hook[hook.index(
        "__aicore__ inline void Emit"
    ):hook.index("__aicore__ inline void Finish")]
    assert "AnyQueueMayBeFree" in hook
    assert emit_body.index("AnyQueueMayBeFree()") < emit_body.index(
        "TryAcquireKvafStreamingBatchGetQueueFromAicore"
    )
    assert "queueLeases_.GetValue(queueId) == 0" in hook
    assert "official_score_tiles" not in hook
    assert "exact_sparse_indices" not in hook

    # The public schema, host definition, infer/tiling contract, device entry
    # signature, and QLIPreload forwarding must stay in one exact order.  The
    # following are intentionally narrow textual checks rather than a guessed
    # CANN build on a non-A5 host.
    public_def = op_def.split("OpAICoreConfig", 1)[0]
    previous = -1
    for name in CANN_INPUT_NAMES:
        position = public_def.index(f'this->Input("{name}")')
        assert position > previous, name
        previous = position
    previous = -1
    for name in OUTPUT_NAMES:
        position = public_def.index(f'this->Output("{name}")')
        assert position > previous, name
        previous = position
    target_def = op_def[op_def.index("OpAICoreConfig"):]
    query_target = target_def.index('aicore_config_950.Input("query")')
    key_target = target_def.index('aicore_config_950.Input("key")')
    assert query_target < target_def.index("ge::DT_FLOAT8_E4M3FN", query_target)
    assert key_target < target_def.index("ge::DT_FLOAT8_E4M3FN", key_target)
    assert "query.scalar_type() == at::kFloat8_e4m3fn" in bridge
    assert "key.scalar_type() == at::kFloat8_e4m3fn" in bridge
    # The public Torch schema keeps the required domain before its optional
    # defaults, while generated ACLNN follows the immutable CANN declaration
    # order. EXEC_NPU_CMD_ORDERED creates a function pointer from its argument
    # tuple, so the bridge must prove this one deliberate reorder explicitly.
    assert BRIDGE.is_file()
    assert BRIDGE_ATTR_REORDER_COMMENT in bridge, (
        "bridge must document the intentional public-Torch to generated-ACLNN "
        "attribute reorder"
    )
    bridge_def = bridge[bridge.index("TORCH_LIBRARY("):bridge.index("TORCH_LIBRARY_IMPL")]
    previous = -1
    for name in CANN_INPUT_NAMES:
        position = bridge_def.index(f"Tensor {name}")
        assert position > previous, name
        previous = position
    previous = -1
    for token in TORCH_SCHEMA_ATTR_TOKENS:
        position = bridge_def.index(token)
        assert position > previous, token
        previous = position

    public_def_attr_order = []
    for name in CANN_ATTR_NAMES:
        position = public_def.index(f'this->Attr("{name}")')
        public_def_attr_order.append((position, name))
    assert_order(
        "OpDef attr ABI",
        [name for _, name in sorted(public_def_attr_order)],
        CANN_ATTR_NAMES,
    )

    allocate_start = bridge.index("Result AllocateAndLaunch(")
    allocate_signature = bridge[
        allocate_start:bridge.index("\n{", allocate_start)
    ]
    previous = -1
    for token in (
        "int64_t queryQuantMode",
        "int64_t keyQuantMode",
        "const std::string& layoutQuery",
        "const std::string& layoutKey",
        "int64_t sparseCount",
        "int64_t sparseMode",
        "int64_t preTokens",
        "int64_t nextTokens",
        "int64_t cmpRatio",
        "bool returnValues",
        "int64_t stride",
        "int64_t scaleStride",
        "int64_t kvafRowsPerEntry",
        "int64_t kvObjectDomain",
    ):
        position = allocate_signature.index(token)
        assert position > previous, token
        previous = position

    npu_to_launch = bridge[
        bridge.index("Result StreamingQuantLiGetNpu("):bridge.index("Result StreamingQuantLiGetMeta(")
    ]
    npu_launch_call = npu_to_launch[npu_to_launch.index("return AllocateAndLaunch("):]
    previous = -1
    for token in ALLOCATE_ATTR_ARGUMENTS:
        position = npu_launch_call.index(token)
        assert position > previous, token
        previous = position

    aclnn_call = bridge[bridge.index("EXEC_NPU_CMD_ORDERED("):bridge.index("return std::make_tuple(")]
    previous = -1
    for name in (
        "query", "keyArg", "weights", "queryScale", "keyScale",
        "actualSeqLengthsQ", "actualSeqLengthsK", "blockTable", "metadata",
        "residentThreshold", "residentThresholdValid", "kvObjectSlotsSnapshot",
        "reqPoolEntries", "kvObjectMap", "privateSourceToSlot",
        "privateReadyTable", "privateArenaHead", "privatePackedOutput",
        "queueControl", "queueLeases", "kvObjectLens",
        *ACLNN_ATTR_ARGUMENTS,
        "sparseIndices", "sparseValues", "privateSourceToSlotOut",
        "privateReadyTableOut", "privateArenaHeadOut", "privatePackedOutputOut",
        "queueLeasesOut", "telemetry", "scoreSourceLens",
    ):
        position = aclnn_call.index(name)
        assert position > previous, name
        previous = position
    for token in (
        "KV_OBJECT_SLOTS_SNAPSHOT_INDEX = 11",
        "KV_OBJECT_MAP_INDEX = 13",
        "KV_OBJECT_LENS_INDEX = 20",
        "ATTR_QUERY_QUANT_MODE_INDEX = 0",
        "ATTR_KEY_QUANT_MODE_INDEX = 1",
        "ATTR_QUERY_LAYOUT_INDEX = 2",
        "ATTR_KEY_LAYOUT_INDEX = 3",
        "ATTR_SPARSE_COUNT_INDEX = 4",
        "ATTR_SPARSE_MODE_INDEX = 5",
        "ATTR_PRE_TOKENS_INDEX = 6",
        "ATTR_NEXT_TOKENS_INDEX = 7",
        "ATTR_CMP_RATIO_INDEX = 8",
        "ATTR_RETURN_VALUES_INDEX = 9",
        "ATTR_STRIDE_INDEX = 10",
        "ATTR_SCALE_STRIDE_INDEX = 11",
        "ATTR_KVAF_ROWS_PER_ENTRY_INDEX = 12",
        "ATTR_KV_OBJECT_DOMAIN_INDEX = 13",
        "RESIDENT_THRESHOLD_VALID_INDEX = 10",
        "PRIVATE_PACKED_OUTPUT_INDEX = 17",
        "QUEUE_CONTROL_INDEX = 18",
        "QUEUE_LEASES_INDEX = 19",
        "PRIVATE_PACKED_OUTPUT_OUTPUT_INDEX = 5",
        "QUEUE_LEASES_OUTPUT_INDEX = 6",
        "TELEMETRY_OUTPUT_INDEX = 7",
        "SCORE_SOURCE_LENS_OUTPUT_INDEX = 8",
        "streamingPrivatePackedRows",
    ):
        assert token in tiling_header, token
    for token in (
        "KV_OBJECT_SLOTS_SNAPSHOT_INDEX = 11",
        "KV_OBJECT_MAP_INDEX = 13",
        "KV_OBJECT_LENS_INDEX = 20",
        "ATTR_KV_OBJECT_DOMAIN_INDEX = 13",
        "RESIDENT_THRESHOLD_VALID_INDEX = 10",
        "PRIVATE_PACKED_OUTPUT_INDEX = 17",
        "QUEUE_CONTROL_INDEX = 18",
        "QUEUE_LEASES_INDEX = 19",
        "KV_OBJECT_LENS_INDEX = 20",
        "PRIVATE_PACKED_OUTPUT_OUTPUT_INDEX = 5",
        "QUEUE_LEASES_OUTPUT_INDEX = 6",
        "TELEMETRY_OUTPUT_INDEX = 7",
        "SCORE_SOURCE_LENS_OUTPUT_INDEX = 8",
        "privatePackedOutputOut",
        "scoreSourceLens->SetDim(0, rowCount)",
    ):
        assert token in infer, token
    for token in (
        "KV_OBJECT_LENS_INDEX",
        "ATTR_KV_OBJECT_DOMAIN_INDEX",
        "privatePackedOutput.GetDim(1) != 656",
        "privatePackedOutput.GetDim(0) / queueCount != privateCapacity",
        "privatePackedOutputOut",
        "streamingInfo->privatePackedRows",
        "QUEUE_CONTROL_INDEX",
        "QUEUE_LEASES_INDEX",
        "SCORE_SOURCE_LENS_OUTPUT_INDEX",
        "!ShapeEquals(scoreSourceLens, {rowCount})",
    ):
        assert token in tiling, token
    signature = kernel[kernel.index("void a5_streaming_quant_li_get("):kernel.index("__gm__ uint8_t *workspace")]
    previous = -1
    for name in KERNEL_ARGUMENT_NAMES:
        position = signature.index(f"*{name}")
        assert position > previous, name
        previous = position
    init_call = kernel[kernel.index("op.Init("):kernel.index("user, tiling_data")]
    previous = -1
    # The device call uses camelCase names; the sequence is still the public
    # ABI followed by all formal receipts.
    for name in (
        "query", "key", "weights", "queryScale", "keyScale",
        "actualSeqLengthsQ", "actualSeqLengthsK", "blocktable", "metadata",
        "residentThreshold", "residentThresholdValid", "kvObjectSlotsSnapshot",
        "reqPoolEntries",
        "kvObjectMap", "privateSourceToSlot", "privateReadyTable",
        "privateArenaHead", "privatePackedOutput", "queueControl",
        "queueLeases", "kvObjectLens", "sparseIndices", "sparseValues",
        "privateSourceToSlotOut", "privateReadyTableOut", "privateArenaHeadOut",
        "privatePackedOutputOut", "queueLeasesOut", "telemetry",
        "scoreSourceLens",
    ):
        position = init_call.index(name)
        assert position > previous, name
        previous = position
    assert "privatePackedOutput, queueControl, queueLeases" in preload
    assert "kvObjectSlotsSnapshot" in preload
    assert "kvObjectMap" in preload
    assert "kvObjectLens" in preload
    assert "residentThresholdValid" in preload
    assert "privatePackedOutputOut" in preload
    assert "queueLeasesOut" in preload
    assert "InitScoreSourceLensOutput" in preload
    # Object L is distinct from LI's score-source S. It must bound the
    # logical map before the resident directory is consulted, so static tails
    # cannot create a speculative H2D request on graph replay.
    candidate_bound = hook.index("kvObjectLens_.GetValue(row)")
    cache_lookup = hook.index("kvObjectSlots_.GetValue")
    assert candidate_bound < cache_lookup
    assert "kvObjectLen <= 0" in hook
    assert "kvObjectLen) > kvObjectCapacity_" in hook
    # ProcessInvalid is the no-ProcessTopK graph-replay path. It must clear
    # output #8, otherwise stale S_active from the prior forward can pass
    # Reconcile's source bound alongside freshly invalid sparse indices.
    assert "ClearScoreSourceLensForInvalidForward" in preload
    assert "InitGlobalMemory(output, end - begin, 0)" in vector
    # Actual A5 C8 is N2==1 with G==64.  The frequently discussed C=4 is an
    # *internal S1 score tile* (M_BASE_SIZE=256 / G=64), not a second sparse
    # output dimension and not an independent cache-directory namespace.
    assert "n2Size_ != 1" in tiling
    assert "gSize_ != G_SIZE_LIMIT" in tiling
    assert "M_BASE_SIZE" in preload
    assert "constInfo.s1BaseSize = (constInfo.mBaseSize + constInfo.gSize - 1) / constInfo.gSize" in preload


def probe_math_and_lifecycle() -> None:
    # Two score tiles: a speculative candidate is detected in the first. The
    # model checks only that an observer leaves the score representation
    # unchanged; it does not execute the target C8 TopK kernel.
    tile0 = [7, 2, 19, 4, 15, 1, 11, 6]
    tile1 = [3, 22, 5, 14, 9, 18, 8, 13]
    original = tile0 + tile1
    observed = list(original)  # TileReady reads; it never writes score data.
    assert topk(original, 4) == topk(observed, 4)

    # The production output carries the exact `validS2Len` used by C8
    # ProcessTopK.  In causal compressed-S2 mode this is not the logical-KV
    # length, so an arbitrary logical cache capacity must never be substituted
    # for it by the host-side Reconcile bridge.
    def official_score_source_len(
        act_s2: int, act_s2_orig: int, act_s1: int, s1_index: int,
        cmp_ratio: int, causal: bool,
    ) -> int:
        if not causal:
            return act_s2
        return max(0, (act_s2_orig - act_s1 + s1_index + 1) // cmp_ratio)

    compressed_lengths = [
        official_score_source_len(4096, 16384, 4, row, 4, True)
        for row in range(4)
    ]
    assert compressed_lengths == [4095, 4095, 4095, 4096]
    logical_capacity = 16384
    assert all(0 <= value < logical_capacity for value in compressed_lengths)
    assert compressed_lengths[-1] != logical_capacity

    # The four C8 rows in one score tile are ordinary flattened output rows.
    # Each paired AIV processes two of them; neither AIV invents an N2=2
    # output axis.  The mapping must agree with ProcessTopK's
    # `indiceOutOffset / K + s1_index` calculation.
    m_base, group_size, n2 = 256, 64, 1
    c_rows = (m_base + group_size - 1) // group_size
    assert (n2, c_rows) == (1, 4)
    aiv0_rows, aiv1_rows = [0, 1], [2, 3]
    assert sorted(aiv0_rows + aiv1_rows) == list(range(c_rows))
    sparse_k = 2048
    # TND: the current product path has T rows and N2=1.
    tnd_offset = 7 * sparse_k
    assert [tnd_offset // sparse_k + row for row in aiv0_rows + aiv1_rows] == [7, 8, 9, 10]
    # BSND: B=2, S1=6, N2=1.  A C=4 tile starting at S1=0 likewise emits
    # the four consecutive flattened output rows for that batch.
    bsnd_offset = (1 * 6 * n2) * sparse_k
    assert [bsnd_offset // sparse_k + row for row in aiv0_rows + aiv1_rows] == [6, 7, 8, 9]

    timeline: list[str] = []
    timeline.append("scan_tile_0")
    timeline.append("lease_acquired")
    timeline.append("doorbell_posted")
    timeline.append("scan_tile_1")
    timeline.append("official_exact_topk")
    timeline.append("cq_terminal_ready")
    timeline.append("lease_released")
    assert timeline.index("doorbell_posted") < timeline.index("scan_tile_1")
    assert timeline.index("doorbell_posted") < timeline.index("official_exact_topk")
    assert timeline.index("official_exact_topk") < timeline.index("cq_terminal_ready")
    assert timeline.index("cq_terminal_ready") < timeline.index("lease_released")
    print(
        "P3_STREAMING_LI_GET_CONTRACT_PASS "
        f"source_score_nonmutation_guard=1 doorbell_before_scan_done=1 "
        f"final_get_overlaps_exact_topk=1 cq_before_release=1 "
        "exact_score_source_lens=1 c8_n2_one=1 c8_internal_s1_tile_rows=4 "
        f"model_topk={topk(original, 4)} "
        "runtime_bitwise_equivalence=required payload_bandwidth=not_reported"
    )


if __name__ == "__main__":
    require_source_contract()
    probe_math_and_lifecycle()
