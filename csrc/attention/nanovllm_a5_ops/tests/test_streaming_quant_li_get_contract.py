#!/usr/bin/env python3
"""Static ABI and publication-contract gate for A5StreamingQuantLiGet.

P3 no longer embeds a Direct-HCCP/KVAF hook in the C8 score producer.  This
probe deliberately has no NPU dependency.  It makes a stale 21-input bridge,
an old inline-HCCP source seam, or an inconsistent score/receipt ABI fail
before an expensive target-NPU equivalence run.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "a5_streaming_quant_li_get"
VECTOR = ROOT / "op_kernel/arch35/quant_lightning_indexer_service_vector.h"
HOOK_MODE = ROOT / "op_kernel/a5_streaming_quant_li_get_hook_mode.h"
KERNEL = ROOT / "op_kernel/a5_streaming_quant_li_get.cpp"
KERNEL_PRELOAD = ROOT / "op_kernel/arch35/quant_lightning_indexer_kernel.h"
OP_DEF = ROOT / "op_host/a5_streaming_quant_li_get_def.cpp"
INFER = ROOT / "op_host/a5_streaming_quant_li_get_infershape.cpp"
TILING = ROOT / "op_host/a5_streaming_quant_li_get_tiling.cpp"
TILING_HEADER = ROOT / "op_host/a5_streaming_quant_li_get_tiling.h"
BRIDGE = ROOT / "torch_extension/csrc/streaming_quant_li_get.cpp"
OPS = ROOT / "ops.json"

CANN_INPUT_NAMES = [
    "query", "key", "weights", "query_dequant_scale", "key_dequant_scale",
    "actual_seq_lengths_query", "actual_seq_lengths_key", "block_table",
    "metadata", "resident_threshold", "resident_threshold_valid",
    "kv_object_slots_snapshot", "req_pool_entries", "kv_object_map",
    "private_source_to_slot", "private_ready_table", "private_arena_head",
    "private_packed_output", "queue_control", "queue_leases", "kv_object_lens",
    "score_workspace", "tile_receipts", "streaming_epoch",
]
OUTPUT_NAMES = [
    "sparse_indices", "sparse_values", "private_source_to_slot_out",
    "private_ready_table_out", "private_arena_head_out",
    "private_packed_output_out", "queue_leases_out", "telemetry",
    "score_source_lens", "score_workspace_out", "tile_receipts_out",
]
KERNEL_ARGUMENT_NAMES = [
    "query", "key", "weights", "queryScale", "keyScale",
    "actualSeqLengthsQ", "actualSeqLengthsK", "blocktable", "metadata",
    "residentThreshold", "residentThresholdValid", "kvObjectSlotsSnapshot",
    "reqPoolEntries", "kvObjectMap", "privateSourceToSlot",
    "privateReadyTable", "privateArenaHead", "privatePackedOutput",
    "queueControl", "queueLeases", "kvObjectLens", "scoreWorkspace",
    "tileReceipts", "streamingEpoch", "sparseIndices", "sparseValues",
    "privateSourceToSlotOut", "privateReadyTableOut", "privateArenaHeadOut",
    "privatePackedOutputOut", "queueLeasesOut", "telemetry",
    "scoreSourceLens", "scoreWorkspaceOut", "tileReceiptsOut",
]


def assert_order(label: str, actual: list[str], expected: list[str]) -> None:
    assert actual == expected, (
        f"{label} order mismatch\nexpected: {' -> '.join(expected)}\n"
        f"actual:   {' -> '.join(actual)}"
    )


def _assert_in_order(text: str, tokens: list[str], label: str) -> None:
    previous = -1
    for token in tokens:
        position = text.index(token)
        assert position > previous, f"{label}: {token} is out of order"
        previous = position


def require_source_contract() -> None:
    vector = VECTOR.read_text(encoding="utf-8")
    hook_mode = HOOK_MODE.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    preload = KERNEL_PRELOAD.read_text(encoding="utf-8")
    op_def = OP_DEF.read_text(encoding="utf-8")
    infer = INFER.read_text(encoding="utf-8")
    tiling = TILING.read_text(encoding="utf-8")
    tiling_header = TILING_HEADER.read_text(encoding="utf-8")
    bridge = BRIDGE.read_text(encoding="utf-8")
    ops = json.loads(OPS.read_text(encoding="utf-8"))

    assert len(ops) == 1 and ops[0]["op"] == "A5StreamingQuantLiGet"
    assert len(ops[0]["input_desc"]) == 24
    assert len(ops[0]["output_desc"]) == 11
    assert_order(
        "ops.json input ABI",
        [item["name"] for item in ops[0]["input_desc"]],
        CANN_INPUT_NAMES,
    )
    assert_order(
        "ops.json output ABI",
        [item["name"] for item in ops[0]["output_desc"]],
        OUTPUT_NAMES,
    )
    inputs = ops[0]["input_desc"]
    outputs = ops[0]["output_desc"]
    assert inputs[21]["type"] == ["int32"]
    assert inputs[22]["type"] == ["int64"]
    assert inputs[23]["type"] == ["int64"]
    assert outputs[9]["type"] == ["int32"]
    assert outputs[10]["type"] == ["int64"]

    # Publication mode is compiled into each isolated OPP artifact.
    assert "A5_STREAMING_LI_GET_PUBLISH_MODE" in hook_mode
    assert "A5_STREAMING_LI_GET_PUBLISH_MODE >= 0" in hook_mode
    assert "A5_STREAMING_LI_GET_PUBLISH_MODE <= 2" in hook_mode
    assert "A5_STREAMING_LI_GET_ENABLE_HOOK" not in hook_mode
    assert "streamingKvafHook_" not in vector
    assert "FinishStreamingKvafHook" not in vector
    assert "a5_streaming_quant_li_get_tile_ready.h" not in vector

    # Modes 1/2 redirect the official score DMA; mode 2 publishes two fixed
    # 64-byte lines after it, before the existing MTE3/V release.
    for token in (
        "#if A5_STREAMING_LI_GET_PUBLISH_MODE >= 1",
        "scoreGm.SetGlobalBuffer((__gm__ SCORE_T *)(scoreWorkspace",
        "InitScoreWorkspaceBase",
        "#if A5_STREAMING_LI_GET_PUBLISH_MODE >= 2",
    ):
        assert token in preload, token
    for token in (
        "InitScoreTileReceipts",
        "PrepareScoreTileReceipts",
        "EmitScoreTileReceipts",
        "TBuf<TPosition::VECCALC> tileReceiptBuf_",
        "LocalTensor<int64_t> tileReceiptLocal_",
        "kWordsPerReceipt = 16",
        "streamingEpoch_",
        "tileReceiptsGm[receiptBase]",
        "tileReceiptsGm[receiptBase + kWordsPerLine]",
    ):
        assert token in vector, token
    assert "mrgValueLocal_.template ReinterpretCast<int64_t>()" not in vector
    score_dma = vector.index("DataCopyPad(scoreGm")
    receipt_emit = vector.index("EmitScoreTileReceipts", score_dma)
    mte3_release = vector.index("SetFlag<HardEvent::MTE3_V>", score_dma)
    assert score_dma < receipt_emit < mte3_release
    assert "BatchMulWeightAndReduceSum" in vector
    assert "ProcessTopK" in vector

    # Transaction resources are retained as appended graph ABI only.  The C8
    # producer does not read a queue, publish HCCP, or write private state.
    for token in (
        "(void)residentThreshold;", "(void)queueControl;",
        "(void)queueLeases;", "(void)scoreWorkspaceOut;",
        "(void)tileReceiptsOut;",
    ):
        assert token in preload, token

    for token in (
        "SCORE_WORKSPACE_INDEX = 21", "TILE_RECEIPTS_INDEX = 22",
        "STREAMING_EPOCH_INDEX = 23", "SCORE_WORKSPACE_OUTPUT_INDEX = 9",
        "TILE_RECEIPTS_OUTPUT_INDEX = 10", "STREAMING_TILE_RECEIPT_WORDS = 16",
    ):
        assert token in tiling_header, token
    for token in (
        "*scoreWorkspaceOut = *scoreWorkspace",
        "*tileReceiptsOut = *tileReceipts",
        "scoreWorkspace.GetDim(0) < scoreWorkspaceWords",
        "!ShapeEquals(tileReceipts,",
        "!ShapeEquals(streamingEpoch, {1})",
    ):
        assert token in infer or token in tiling, token

    # The bridge must use the same 24/11 public order as CANN.
    for token in (
        "const at::Tensor& scoreWorkspace", "const at::Tensor& tileReceipts",
        "const at::Tensor& streamingEpoch", "scoreWorkspaceOut = scoreWorkspace",
        "tileReceiptsOut = tileReceipts", "in21.score_workspace",
        "in22.tile_receipts", "in23.streaming_epoch",
        "out9.score_workspace_out", "out10.tile_receipts_out",
        "Tensor score_workspace, Tensor tile_receipts, Tensor streaming_epoch",
        "Tensor score_workspace_out", "Tensor tile_receipts_out",
    ):
        assert token in bridge, token

    signature = kernel[
        kernel.index("void a5_streaming_quant_li_get("):
        kernel.index("__gm__ uint8_t *workspace")
    ]
    _assert_in_order(signature, [f"*{name}" for name in KERNEL_ARGUMENT_NAMES], "kernel ABI")
    _assert_in_order(
        op_def,
        [f'this->Input("{name}")' for name in CANN_INPUT_NAMES] +
        [f'this->Output("{name}")' for name in OUTPUT_NAMES],
        "OpDef ABI",
    )


def probe_publication_layout() -> None:
    # Compact host interpretation check. Runtime byte-equivalence remains the
    # separate NPU7 test using this exact 24/11 ABI.
    source, rows, tile_width, aic = 4096, 2, 128, 36
    aligned = ((source + tile_width - 1) // tile_width) * tile_width
    score_words = (aic * 4 * aligned + 1) // 2
    tiles_per_row = source // tile_width
    assert score_words == 294912
    assert rows * tiles_per_row == 64
    row_lengths = (4096, 3968)
    active = sum((length + tile_width - 1) // tile_width for length in row_lengths)
    assert active == 63
    sample_byte_offset = 2 * (7 * tile_width)
    assert sample_byte_offset % 2 == 0
    assert 0 <= sample_byte_offset < score_words * 4
    print(
        "P3_STREAMING_LI_GET_CONTRACT_PASS "
        "abi=24in_11out publication_modes=off_score_workspace_publisher "
        "inline_hccp=absent receipt_words=16 fixed_slot_epoch=required "
        f"score_workspace_words={score_words} active_receipts={active}"
    )


if __name__ == "__main__":
    require_source_contract()
    probe_publication_layout()
