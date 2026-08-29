#!/usr/bin/env python3
"""Static ABI and score-tile-inline contract for A5StreamingQuantLiGet.

This probe has no NPU dependency. It guards the one 24-input / 11-output ABI
shared by four isolated artifacts: strict C8 control, score publisher,
inline-audit, and inline native-enqueue post. It rejects an accidental mixed
publication/HCCP closure before an expensive target-NPU run.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "a5_streaming_quant_li_get"
OPS_ROOT = ROOT.parent
VECTOR = ROOT / "op_kernel/arch35/quant_lightning_indexer_service_vector.h"
HOOK_MODE = ROOT / "op_kernel/a5_streaming_quant_li_get_hook_mode.h"
AUDIT_HOOK = ROOT / "op_kernel/arch35/a5_streaming_quant_li_get_tile_audit.h"
POST_HOOK = ROOT / "op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h"
KERNEL = ROOT / "op_kernel/a5_streaming_quant_li_get.cpp"
KERNEL_PRELOAD = ROOT / "op_kernel/arch35/quant_lightning_indexer_kernel.h"
OP_DEF = ROOT / "op_host/a5_streaming_quant_li_get_def.cpp"
INFER = ROOT / "op_host/a5_streaming_quant_li_get_infershape.cpp"
TILING = ROOT / "op_host/a5_streaming_quant_li_get_tiling.cpp"
TILING_HEADER = ROOT / "op_host/a5_streaming_quant_li_get_tiling.h"
BRIDGE = ROOT / "torch_extension/csrc/streaming_quant_li_get.cpp"
OP_SPEC = ROOT / "ops.json"
POST_PAYLOAD_TEST = OPS_ROOT / "tests/test_streaming_quant_li_get_post_payload.py"

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
    audit_hook = AUDIT_HOOK.read_text(encoding="utf-8")
    post_hook = POST_HOOK.read_text(encoding="utf-8")
    kernel = KERNEL.read_text(encoding="utf-8")
    preload = KERNEL_PRELOAD.read_text(encoding="utf-8")
    op_def = OP_DEF.read_text(encoding="utf-8")
    infer = INFER.read_text(encoding="utf-8")
    tiling = TILING.read_text(encoding="utf-8")
    tiling_header = TILING_HEADER.read_text(encoding="utf-8")
    bridge = BRIDGE.read_text(encoding="utf-8")
    ops = json.loads(OP_SPEC.read_text(encoding="utf-8"))

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

    # Both axes are compiled into each isolated OPP artifact. Post/HCCP and
    # publication remain mutually exclusive. Audit + publisher is the one
    # score-only provenance exception: it compares Scalar audit receipts with
    # the score-GM consumed by C8 Top-K without adding a transport closure.
    assert "A5_STREAMING_LI_GET_PUBLISH_MODE" in hook_mode
    assert "A5_STREAMING_LI_GET_PUBLISH_MODE >= 0" in hook_mode
    assert "A5_STREAMING_LI_GET_PUBLISH_MODE <= 2" in hook_mode
    assert "A5_STREAMING_LI_GET_ENABLE_HOOK" not in hook_mode
    assert "A5_STREAMING_LI_GET_INLINE_MODE" in hook_mode
    assert "A5_STREAMING_LI_GET_INLINE_MODE >= 0" in hook_mode
    assert "A5_STREAMING_LI_GET_INLINE_MODE <= 2" in hook_mode
    assert "A5_STREAMING_LI_GET_POST_CLOSURE_MODE" in hook_mode
    assert "A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE" in hook_mode
    assert "A5_STREAMING_LI_GET_SCORE_SOURCE_LENS_IS_INPUT_ALIAS" in hook_mode
    assert "except audit + publisher provenance" in hook_mode
    assert '#if A5_STREAMING_LI_GET_INLINE_MODE == 1' in vector
    assert '#elif A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE' in vector
    assert 'a5_streaming_quant_li_get_tile_audit.h' in vector
    assert 'a5_streaming_quant_li_get_tile_ready.h' in vector
    assert "StreamingLiTileAuditHook" in vector
    assert "streamingKvafHook_" in vector
    assert "InitStreamingAuditHook" in vector
    assert "InitStreamingKvafHook" in vector
    assert "FinishStreamingKvafHook" in vector
    assert "#if A5_STREAMING_LI_GET_INLINE_MODE == 1" in audit_hook
    assert "#if A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE" in post_hook
    # A strict audit build is physically score-only: the generated staging
    # directory deletes the post hook and all KVAF/SIMT sources, while this
    # remaining audit header has no call-site into that closure.
    for token in (
        "TryAcquireKvafStreamingBatchGet",
        "BeginKvafStreamingBatchGet",
        "PublishKvafStreamingBatchGetDescriptorScratch",
        "TryRetireKvafStreamingBatchGet",
        "FinishKvafStreamingBatchGet",
        "kvaf_fused_get::",
        "#include \"../kvaf_device/",
        "#include \"../simt_api/",
    ):
        assert token not in audit_hook, token
    assert "kAuditScoreReceipt" not in post_hook
    assert "AuditScoreDecision" not in post_hook
    assert "TryAcquireKvafStreamingBatchGetQueueFromAicore" in post_hook
    assert "BeginKvafStreamingBatchGet" in post_hook
    # Empty-shell post keeps the public 24/11 ABI but must not allocate any
    # post-hook handoff UB or emit an HCCP wave.  Its output #8 is a bridge
    # alias of actual_seq_lengths_key, so C8 has no score-lens side write.
    assert "#if A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE" in vector
    assert "#if !A5_STREAMING_LI_GET_SCORE_SOURCE_LENS_IS_INPUT_ALIAS" in vector
    assert "#if !A5_STREAMING_LI_GET_SCORE_SOURCE_LENS_IS_INPUT_ALIAS" in preload
    assert "? actualSeqLengthsK" in bridge
    assert "CheckStrictDecodeScoreSourceLensAlias" in bridge

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
    reduction = vector.index("BatchMulWeightAndReduceSum")
    audit_emit = vector.index("streamingAuditHook_.Emit")
    post_emit = vector.index("streamingKvafHook_.Emit")
    receipt_emit = vector.index("EmitScoreTileReceipts", score_dma)
    mte3_release = vector.index("SetFlag<HardEvent::MTE3_V>", score_dma)
    assert reduction < audit_emit < score_dma
    assert reduction < post_emit < score_dma
    assert score_dma < receipt_emit < mte3_release
    # The audit hook reads a Vector-owned ping-pong UB score tile via Scalar
    # GetValue().  Inspect the *whole* audit branch (rather than ending at
    # Emit) so this source-only test proves both sides of that lifetime:
    # V->S makes the score visible before the Scalar read, and S->V prevents
    # Vector from reusing the ping-pong tile before that read finishes.
    audit_branch_start = vector.rfind(
        "#if A5_STREAMING_LI_GET_INLINE_MODE == 1", 0, audit_emit
    )
    audit_branch_end = vector.index(
        "#elif A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE", audit_branch_start
    )
    audit_seam = vector[audit_branch_start:audit_branch_end]
    _assert_in_order(
        audit_seam,
        [
            "PipeBarrier<PIPE_V>();",
            "FetchEventID(HardEvent::V_S)",
            "SetFlag<HardEvent::V_S>",
            "WaitFlag<HardEvent::V_S>",
            "streamingAuditHook_.Emit",
            "FetchEventID(HardEvent::S_V)",
            "SetFlag<HardEvent::S_V>",
            "WaitFlag<HardEvent::S_V>",
        ],
        "audit Vector/Scalar score-tile lifetime",
    )
    # The immediate P3 Post control remains available in the source, while
    # P6 inserts an additional compile-time CandidateSeed branch before it.
    # Check the direct branch's lifetime using a tail that starts at its own
    # callsite rather than stopping at the first nested preprocessor endif.
    direct_prefix = vector.rfind("PipeBarrier<PIPE_V>();", 0, post_emit)
    direct_suffix = vector.index("int64_t vec1OutGmOffset", post_emit)
    direct_post_seam = vector[direct_prefix:direct_suffix]
    _assert_in_order(
        direct_post_seam,
        [
            "PipeBarrier<PIPE_V>();",
            "FetchEventID(HardEvent::V_S)",
            "SetFlag<HardEvent::V_S>",
            "WaitFlag<HardEvent::V_S>",
            "streamingKvafHook_.Emit",
            "FetchEventID(HardEvent::S_V)",
            "SetFlag<HardEvent::S_V>",
            "WaitFlag<HardEvent::S_V>",
        ],
        "immediate-post Vector/Scalar score-tile lifetime",
    )
    assert "BatchMulWeightAndReduceSum" in vector
    assert "ProcessTopK" in vector

    # The inline=off branch retains all transaction resources only as ABI
    # carriers. Audit initializes only its fixed receipt observer; post alone
    # initializes and drains the native enqueue state after official Top-K.
    for token in (
        "(void)residentThreshold;", "(void)queueControl;",
        "(void)queueLeases;", "(void)scoreWorkspaceOut;",
        "(void)tileReceiptsOut;",
    ):
        assert token in preload, token
    assert "#if A5_STREAMING_LI_GET_INLINE_MODE == 1" in preload
    assert "#elif A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE" in preload
    assert "vectorService.InitStreamingAuditHook(" in preload
    assert "vectorService.InitStreamingKvafHook(" in preload
    free_event = preload.index("vectorService.FreeEventID();")
    finish_hook = preload.index("vectorService.FinishStreamingKvafHook();")
    assert free_event < finish_hook

    for token in (
        "NEEDS_P3_POST_KVAF=0",
        "NANOVLLM_A5_STREAMING_LI_GET_POST_CLOSURE_MODE",
        "NANOVLLM_A5_STREAMING_LI_GET_POST_SCHEDULER",
        "P3 post closure mode:",
        "P6 post scheduler:",
        "post_closure_mode=${STREAMING_LI_POST_CLOSURE_MODE_NAME}",
        "post_scheduler_mode=${STREAMING_LI_POST_SCHEDULER_MODE_NAME}",
        "P3 inline=audit closure must contain only the audit observer",
        "P3 generated audit closure: score-only verified=1",
        "kvaf_closure=${P3_KVAF_CLOSURE}",
        "kvaf_source_head=${P3_KVAF_SOURCE_HEAD}",
    ):
        assert token in (OPS_ROOT / "build_nanovllm_ops.sh").read_text(encoding="utf-8"), token

    for token in (
        "SCORE_WORKSPACE_INDEX = 21", "TILE_RECEIPTS_INDEX = 22",
        "STREAMING_EPOCH_INDEX = 23", "SCORE_WORKSPACE_OUTPUT_INDEX = 9",
        "TILE_RECEIPTS_OUTPUT_INDEX = 10", "STREAMING_TILE_RECEIPT_WORDS = 16",
        "STREAMING_TELEMETRY_COLUMNS = 18",
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

    # 36/72 are the fixed source-C8 metadata capacities.  The A5 runtime
    # reports its active core geometry, so tiling must accept any nonzero
    # geometry that fits in those arrays rather than requiring an exact 36
    # AIC count.  The bridge still reserves the 36-AIC maximum workspace.
    assert "kA5StreamingMaxAicCount = 36" in tiling
    assert "kA5StreamingMaxAivCount = 72" in tiling
    assert "aicCount <= 0 || aicCount > kA5StreamingMaxAicCount" in tiling
    assert "aivCount <= 0 || aivCount > kA5StreamingMaxAivCount" in tiling
    assert "aicCount != kA5StreamingAicCount" not in tiling

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

    target_adapter = (
        OPS_ROOT / "tests/streaming_quant_li_get_target_adapter.py"
    ).read_text(encoding="utf-8")
    assert '"kvaf_closure"' in target_adapter
    assert "POST_CLOSURE_MODE_ENV" in target_adapter
    assert "POST_SCHEDULER_MODE_ENV" in target_adapter
    assert '"staged" if expected_post_closure_mode == "full" else "absent"' in target_adapter
    assert "P3_POST_CLOSURE_MODE=empty_shell" in target_adapter
    assert "P3_PHYSICAL_NPU_ID" in target_adapter
    assert "ASCEND_RT_VISIBLE_DEVICES" in target_adapter
    assert "requested_logical=" in target_adapter
    assert "physical_npu=" in target_adapter
    assert "no-transport artifact unexpectedly records a KVAF" in target_adapter

    empty_shell_runner = (
        OPS_ROOT / "tests/test_streaming_quant_li_get_empty_shell_strict.py"
    ).read_text(encoding="utf-8")
    for token in (
        "run_target_equivalence",
        "P3_EMPTY_SHELL_STRICT_PASS",
        "P3_EMPTY_SHELL_STRICT_FAIL",
        "P3_POST_CLOSURE_MODE=empty_shell",
        "hccp=not_compiled",
    ):
        assert token in empty_shell_runner, token

    # The strict post test is intentionally separate from the former
    # known-failed payload probe.  It must exercise the full 24/11 ABI and
    # reconstruct every payload row via row -> pool -> object -> table -> key.
    post_payload = POST_PAYLOAD_TEST.read_text(encoding="utf-8")
    for token in (
        "P3_POST_PAYLOAD_STRICT_PASS",
        "P3_POST_PAYLOAD_DIAGNOSTIC",
        "ASCEND_RT_VISIBLE_DEVICES",
        "_read_target_provenance(target_vendor, \"off\", \"post\")",
        "bind_streaming_batch_get_control",
        "configure_streaming_batch_get_queue_templates",
        "req_pool_entries",
        "kv_object_map",
        "key = int(keys[table_index])",
        "mapped_sources != len(unique_slots)",
        "P3_POST_ONLINE_OVERLAP_PROOF",
        "score_tiles_with_cq_pending",
        "P3_POST_PAYLOAD_FAIL",
    ):
        assert token in post_payload, token


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
        "inline_modes=off_audit_post post_publication_exclusive=1 "
        "audit_publisher_score_provenance_exception=1 "
        "score_tile_seam=reduction_before_inline_before_score_dma "
        "receipt_words=16 fixed_slot_epoch=required "
        f"score_workspace_words={score_words} active_receipts={active}"
    )


if __name__ == "__main__":
    require_source_contract()
    probe_publication_layout()
