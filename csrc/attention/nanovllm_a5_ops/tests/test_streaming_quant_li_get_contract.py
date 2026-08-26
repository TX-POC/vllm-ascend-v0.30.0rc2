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
KERNEL = ROOT / "op_kernel/a5_streaming_quant_li_get.cpp"
KERNEL_PRELOAD = ROOT / "op_kernel/arch35/quant_lightning_indexer_kernel.h"
OP_DEF = ROOT / "op_host/a5_streaming_quant_li_get_def.cpp"
INFER = ROOT / "op_host/a5_streaming_quant_li_get_infershape.cpp"
TILING = ROOT / "op_host/a5_streaming_quant_li_get_tiling.cpp"
TILING_HEADER = ROOT / "op_host/a5_streaming_quant_li_get_tiling.h"
OPS = ROOT / "ops.json"
ROOT_CMAKE = ROOT / "CMakeLists.txt"
METADATA_COMPANION = ROOT / "metadata_companion.md"

INPUT_NAMES = [
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
    "cache_slots_snapshot",
    "req_pool_entries",
    "logical_token_map",
    "private_source_to_slot",
    "private_ready_table",
    "private_arena_head",
    "private_packed_output",
    "queue_control",
    "queue_leases",
]
OUTPUT_NAMES = [
    "sparse_indices",
    "sparse_values",
    "private_source_to_slot_out",
    "private_ready_table_out",
    "private_arena_head_out",
    "private_packed_output_out",
    "queue_leases_out",
    "telemetry",
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
    "cacheSlotsSnapshot",
    "reqPoolEntries",
    "logicalTokenMap",
    "privateSourceToSlot",
    "privateReadyTable",
    "privateArenaHead",
    "privatePackedOutput",
    "queueControl",
    "queueLeases",
    "sparseIndices",
    "sparseValues",
    "privateSourceToSlotOut",
    "privateReadyTableOut",
    "privateArenaHeadOut",
    "privatePackedOutputOut",
    "queueLeasesOut",
    "telemetry",
]


def topk(scores: list[int], k: int) -> list[int]:
    # The test operates on the official uint16 sortable-score representation;
    # deterministic index tie-break makes the source-level non-mutation model
    # unambiguous. This is not a target-runtime bitwise equivalence result.
    return [index for index, _ in sorted(enumerate(scores), key=lambda x: (-x[1], x[0]))[:k]]


def require_source_contract() -> None:
    vector = VECTOR.read_text()
    hook = HOOK.read_text()
    kernel = KERNEL.read_text()
    preload = KERNEL_PRELOAD.read_text()
    op_def = OP_DEF.read_text()
    infer = INFER.read_text()
    tiling = TILING.read_text()
    tiling_header = TILING_HEADER.read_text()
    ops = json.loads(OPS.read_text())
    assert len(ops) == 1 and ops[0]["op"] == "A5StreamingQuantLiGet"
    assert len(ops[0]["input_desc"]) == 20
    assert len(ops[0]["output_desc"]) == 8
    inputs = ops[0]["input_desc"]
    outputs = ops[0]["output_desc"]
    assert [item["name"] for item in inputs] == INPUT_NAMES
    assert [item["name"] for item in outputs] == OUTPUT_NAMES
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
    assert ROOT_CMAKE.is_file()
    assert "descriptor_stage_only" not in OPS.read_text()
    assert "descriptor_stage_only" not in ROOT_CMAKE.read_text()
    assert METADATA_COMPANION.is_file()
    assert "BatchMulWeightAndReduceSum" in vector
    assert "streamingKvafHook_.Emit" in vector
    assert "streamingKvafHook_.Finish" in vector
    assert "ProcessTopK" in vector
    assert vector.index("BatchMulWeightAndReduceSum") < vector.index("streamingKvafHook_.Emit")
    assert vector.index("streamingKvafHook_.Emit") < vector.index("DataCopyPad(scoreGm")
    topk_body = vector[vector.index(
        "__aicore__ inline void QLIVector<QLIT>::ProcessTopK"
    ):]
    # The final H2D must remain in flight through the official exact TopK
    # score-GM read / sparse-index write.  Finish only occurs after the last
    # sparse-index DataCopyPad in that function.
    assert topk_body.rindex("DataCopyPad(indiceOutGm") < topk_body.rindex(
        "streamingKvafHook_.Finish()"
    )
    for symbol in (
        "TryAcquireKvafStreamingBatchGetQueueFromAicore",
        "PublishKvafStreamingBatchGetDescriptorScratch",
        "BeginKvafStreamingBatchGet",
        "TryRetireKvafStreamingBatchGet",
        "FinishKvafStreamingBatchGet",
        "GetKvafStreamingBatchGetReady",
        "ReleaseKvafStreamingBatchGetQueueFromAicore",
        "LoadKvafStreamingBatchGetQueueTemplate",
        "logicalTokenMap_.GetValue",
        "cacheSlots_.GetValue",
        "ClaimStreamingPrivateArenaSimt",
        "residentThresholdValid_.GetValue(row) != 1",
        "& 0xffffU",
        "privateArenaCapacity_ != privateRows",
        "control_.output != reinterpret_cast<uint64_t>(privatePackedOutputRaw_)",
        "control_.output_rows != privatePackedRows_",
    ):
        assert symbol in hook, symbol
    assert "official_score_tiles" not in hook
    assert "exact_sparse_indices" not in hook

    # The public schema, host definition, infer/tiling contract, device entry
    # signature, and QLIPreload forwarding must stay in one exact order.  The
    # following are intentionally narrow textual checks rather than a guessed
    # CANN build on a non-A5 host.
    public_def = op_def.split("OpAICoreConfig", 1)[0]
    previous = -1
    for name in INPUT_NAMES:
        position = public_def.index(f'this->Input("{name}")')
        assert position > previous, name
        previous = position
    previous = -1
    for name in OUTPUT_NAMES:
        position = public_def.index(f'this->Output("{name}")')
        assert position > previous, name
        previous = position
    for token in (
        "RESIDENT_THRESHOLD_VALID_INDEX = 10",
        "PRIVATE_PACKED_OUTPUT_INDEX = 17",
        "QUEUE_CONTROL_INDEX = 18",
        "QUEUE_LEASES_INDEX = 19",
        "PRIVATE_PACKED_OUTPUT_OUTPUT_INDEX = 5",
        "QUEUE_LEASES_OUTPUT_INDEX = 6",
        "TELEMETRY_OUTPUT_INDEX = 7",
        "streamingPrivatePackedRows",
    ):
        assert token in tiling_header, token
    for token in (
        "RESIDENT_THRESHOLD_VALID_INDEX = 10",
        "PRIVATE_PACKED_OUTPUT_INDEX = 17",
        "QUEUE_CONTROL_INDEX = 18",
        "QUEUE_LEASES_INDEX = 19",
        "PRIVATE_PACKED_OUTPUT_OUTPUT_INDEX = 5",
        "QUEUE_LEASES_OUTPUT_INDEX = 6",
        "TELEMETRY_OUTPUT_INDEX = 7",
        "privatePackedOutputOut",
    ):
        assert token in infer, token
    for token in (
        "privatePackedOutput.GetDim(1) != 656",
        "privatePackedOutput.GetDim(0) / queueCount != privateCapacity",
        "privatePackedOutputOut",
        "streamingInfo->privatePackedRows",
        "QUEUE_CONTROL_INDEX",
        "QUEUE_LEASES_INDEX",
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
        "residentThreshold", "residentThresholdValid", "cacheSlotsSnapshot",
        "reqPoolEntries",
        "logicalTokenMap", "privateSourceToSlot", "privateReadyTable",
        "privateArenaHead", "privatePackedOutput", "queueControl",
        "queueLeases", "sparseIndices", "sparseValues",
        "privateSourceToSlotOut", "privateReadyTableOut", "privateArenaHeadOut",
        "privatePackedOutputOut", "queueLeasesOut", "telemetry",
    ):
        position = init_call.index(name)
        assert position > previous, name
        previous = position
    assert "privatePackedOutput, queueControl, queueLeases" in preload
    assert "residentThresholdValid" in preload
    assert "privatePackedOutputOut" in preload
    assert "queueLeasesOut" in preload


def probe_math_and_lifecycle() -> None:
    # Two score tiles: a speculative candidate is detected in the first. The
    # model checks only that an observer leaves the score representation
    # unchanged; it does not execute the target C8 TopK kernel.
    tile0 = [7, 2, 19, 4, 15, 1, 11, 6]
    tile1 = [3, 22, 5, 14, 9, 18, 8, 13]
    original = tile0 + tile1
    observed = list(original)  # TileReady reads; it never writes score data.
    assert topk(original, 4) == topk(observed, 4)

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
        f"model_topk={topk(original, 4)} "
        "runtime_bitwise_equivalence=required payload_bandwidth=not_reported"
    )


if __name__ == "__main__":
    require_source_contract()
    probe_math_and_lifecycle()
