#!/usr/bin/env python3
"""Static P5 contract gate for ExactScoreScan + SpeculativeGet.

This test deliberately has no ``torch`` or NPU dependency.  It protects the
semantic seam between the native C8 score publisher and the independent P5
executor before a target machine spends time building an OPP:

* the executor receives C8's uint16 score workspace and 16-word receipts;
* a candidate is the fixed five-word record
  ``[row, source, kv_object, score_u16, private_slot]``;
* a cold object is claimed once, posted once, and becomes READY only after
  the terminal Direct-HCCP CQ boundary;
* C8 Top-K state is not reimplemented or modified by the executor.

It is intentionally a source/ABI contract, not evidence of NPU performance.
The companion ``test_exact_scan_speculative_get_p5_semantic.py`` constructs a
receipt-shaped device fixture and checks the value-level oracle.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
OP_ROOT = TESTS_ROOT.parent / "a5_exact_scan_speculative_get"
OPS_JSON = OP_ROOT / "ops.json"
HOST_CPP = OP_ROOT / "op_host" / "a5_exact_scan_speculative_get.cpp"
KERNEL_CPP = OP_ROOT / "op_kernel" / "a5_exact_scan_speculative_get.cpp"
TILING_H = OP_ROOT / "op_kernel" / "a5_exact_scan_speculative_get_tiling.h"
BRIDGE_CPP = (
    OP_ROOT / "torch_extension" / "csrc" / "exact_scan_speculative_get.cpp"
)
P3_VECTOR_CPP = (
    TESTS_ROOT.parent
    / "a5_streaming_quant_li_get"
    / "op_kernel"
    / "arch35"
    / "quant_lightning_indexer_service_vector.h"
)


# These lists are the graph ABI.  Keep their ordering in one place: CANN's
# generated host stub and the standalone bridge must agree byte-for-byte with
# this ordering, otherwise a valid tensor can silently be interpreted as an
# unrelated control/state tensor.
INPUTS: tuple[tuple[str, str], ...] = (
    ("score_workspace", "int32"),
    ("tile_receipts", "int64"),
    ("streaming_epoch", "int64"),
    # C8 has a graph-static receipt allocation.  This per-row active source
    # length tells P5 which fixed receipt slots are intentionally inert and
    # which ones must wait for the same forward's ready epoch.
    ("score_source_lens", "int32"),
    ("resident_threshold", "int32"),
    ("resident_threshold_valid", "int32"),
    ("kv_object_slots_snapshot", "int32"),
    ("req_pool_entries", "int32"),
    ("kv_object_map", "int32"),
    ("kv_object_lens", "int32"),
    ("private_source_to_slot", "int32"),
    ("private_object_to_slot", "int32"),
    ("private_ready_table", "int32"),
    ("private_arena_head", "int32"),
    ("private_packed_output", "uint8"),
    ("address_table", "int64"),
    ("valid_table", "uint8"),
    ("fused_get_control", "int64"),
    ("pipeline_state", "int64"),
    ("execution_mode", "int64"),
)

OUTPUTS: tuple[tuple[str, str], ...] = (
    ("private_source_to_slot_out", "int32"),
    ("private_object_to_slot_out", "int32"),
    ("private_ready_table_out", "int32"),
    ("private_arena_head_out", "int32"),
    ("private_packed_output_out", "uint8"),
    ("candidate_seeds", "int32"),
    ("candidate_counts", "int32"),
    ("producer_stats", "int64"),
    ("get_stats", "int64"),
    ("role_trace", "int32"),
    ("get_indices", "int32"),
    ("get_offsets", "int32"),
    ("get_output_indices", "int32"),
    ("get_mask", "uint8"),
    ("get_candidate_seed_slots", "int32"),
    ("get_claim_new", "uint8"),
    ("pipeline_state_out", "int64"),
)

# Torch schemas express aliased outputs with the input's logical name (for
# example ``Tensor(c!) private_ready_table``), while the CANN OpDef needs a
# distinct ``*_out`` port name.  Both spellings are intentional and must stay
# consistent.
BRIDGE_OUTPUT_NAMES: tuple[str, ...] = (
    "private_source_to_slot",
    "private_object_to_slot",
    "private_ready_table",
    "private_arena_head",
    "private_packed_output",
    *(name for name, _ in OUTPUTS[5:-1]),
    "pipeline_state",
)


def _text(path: Path) -> str:
    if not path.is_file():
        raise AssertionError(f"missing P5 contract source: {path}")
    return path.read_text(encoding="utf-8")


def _ordered_names(text: str, direction: str) -> list[str]:
    return re.findall(
        rf'(?:\.|->)\s*{direction}\("([a-z0-9_]+)"\)', text
    )


def _torch_schema_names(bridge: str) -> tuple[list[str], list[str]]:
    """Extract names from the *actual* TORCH_LIBRARY schema declaration.

    Do not grep the full bridge: validation diagnostics and helper parameter
    names legitimately repeat ABI words before ``m.def``.  Only the schema
    is the registration contract consumed by ``torch.ops``.
    """
    start = bridge.find("TORCH_LIBRARY(a5_exact_scan_get, m)")
    end = bridge.find("TORCH_LIBRARY_IMPL(a5_exact_scan_get, PrivateUse1, m)", start)
    if start < 0 or end < 0:
        raise AssertionError("missing a5_exact_scan_get TORCH_LIBRARY schema")
    block = bridge[start:end]
    fragments = re.findall(r'"((?:[^"\\]|\\.)*)"', block)
    schema = "".join(fragments)
    if "exact_scan_speculative_get(" not in schema or "->" not in schema:
        raise AssertionError(f"unparseable exact_scan_speculative_get schema: {schema!r}")
    inputs_text, outputs_text = schema.split("->", 1)
    inputs_text = inputs_text.split("exact_scan_speculative_get(", 1)[1]
    return (
        re.findall(r"Tensor(?:\([a-z]!\))?\s+([a-z0-9_]+)", inputs_text),
        re.findall(r"Tensor(?:\([a-z]!\))?\s+([a-z0-9_]+)", outputs_text),
    )


class P5ExactScanSpeculativeGetContractTest(unittest.TestCase):
    maxDiff = None

    def test_ops_json_is_the_final_20x17_graph_abi(self) -> None:
        document = json.loads(_text(OPS_JSON))
        self.assertEqual(len(document), 1)
        op = document[0]
        self.assertEqual(op.get("op"), "A5ExactScanSpeculativeGet")
        self.assertEqual(
            [(item["name"], item["type"][0]) for item in op["input_desc"]],
            list(INPUTS),
        )
        self.assertEqual(
            [(item["name"], item["type"][0]) for item in op["output_desc"]],
            list(OUTPUTS),
        )
        self.assertEqual(len(op["input_desc"]), 20)
        self.assertEqual(len(op["output_desc"]), 17)

    def test_host_and_bridge_preserve_the_same_abi_order(self) -> None:
        host = _text(HOST_CPP)
        bridge = _text(BRIDGE_CPP)
        self.assertEqual(_ordered_names(host, "Input"), [name for name, _ in INPUTS])
        self.assertEqual(_ordered_names(host, "Output"), [name for name, _ in OUTPUTS])

        bridge_inputs, bridge_outputs = _torch_schema_names(bridge)
        self.assertEqual(bridge_inputs, [name for name, _ in INPUTS])
        self.assertEqual(bridge_outputs, list(BRIDGE_OUTPUT_NAMES))

    def test_host_tiling_keeps_the_64_lane_abi_when_direct_aclnn_rejects_batch_mode(
        self,
    ) -> None:
        """A direct ACLNN probe must not turn a scheduler capability result
        into a false ABI/tiling failure.

        The target graph still requests batch scheduling.  If CANN rejects
        that request in its direct single-op API, P5 reports the explicit
        fallback and retains the same 64 AIV-only logical lanes.  Nothing in
        this launch-mode handling changes C8 scores, candidate selection, or
        the Post/CQ correctness path.
        """
        host = _text(HOST_CPP)
        for required in (
            'LogP5TilingContractFailure("aiv_count"',
            "if (aivCount < kAivCount)",
            "platform.CalcTschBlockDim(kAivCount, 0U, aivCount)",
            "context->SetBlockDim(aivOnlyBlockDim);",
            "const ge::graphStatus scheduleStatus = context->SetScheduleMode(1U);",
            'scheduleStatus == ge::GRAPH_SUCCESS ? "batch" : "default_aiv_only"',
            '"P5_TILING_READY aic=%u aiv=%u requested_aiv=%u block_dim=%u "',
        ):
            self.assertIn(required, host)
        self.assertNotIn(
            "if (context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS)", host
        )
        self.assertLess(
            host.index("const ge::graphStatus scheduleStatus = context->SetScheduleMode(1U);"),
            host.index("context->GetTilingData<"),
        )
        self.assertLess(
            host.index("context->SetBlockDim(aivOnlyBlockDim);"),
            host.index("context->GetTilingData<"),
        )

    def test_host_and_kernel_share_the_standard_pod_tiling_contract(self) -> None:
        """Direct ACLNN must use one standard C++ POD on both sides.

        ``REGISTER_TILING_DEFAULT`` in the device entry point tells CANN the
        exact fixed payload size.  The Host must acquire that payload through
        ``GetTilingData<T>()`` and write the same struct.  Mixing it with the
        older macro/raw-buffer contract can leave direct ACLNN with an
        ``opParaSize`` of only eight and a zero-capacity tiling payload.
        """
        host = _text(HOST_CPP)
        tiling = _text(TILING_H)
        kernel = _text(KERNEL_CPP)
        for required in (
            '#include "../op_kernel/a5_exact_scan_speculative_get_tiling.h"',
            "context->GetTilingData<",
            "A5ExactScanSpeculativeGetTilingData>();",
            "21U * sizeof(uint32_t)",
            "reserved) ==",
            '"P5_TILING_TYPED_READY bytes=%zu alignment=%zu "',
            '"P5_TILING_PREALLOC aic=%u aiv=%u requested_aiv=%u block_dim=%u "',
        ):
            self.assertIn(required, host)
        for forbidden in (
            "BEGIN_TILING_DATA_DEF",
            "REGISTER_TILING_DATA_CLASS",
            "GetRawTilingData",
            "SaveToBuffer",
            "SetDataSize",
        ):
            self.assertNotIn(forbidden, host)

        # The device and host share one 21-word POD.  CANN's runtime header
        # is outside this payload; adding a synthetic tail word changes the
        # registered direct-ACLNN typed allocation contract.
        fields = re.findall(
            r"uint32_t\s+([A-Za-z0-9_]+);",
            tiling,
        )
        self.assertEqual(
            fields,
            [
                "rowCount",
                "sourceCapacity",
                "poolSize",
                "kvObjectCapacity",
                "privateArenaCapacity",
                "tileReceiptCount",
                "scoreWorkspaceWords",
                "rowsPerEntry",
                "candidateCapacityPerProducer",
                "candidateSeedWords",
                "tableCapacity",
                "outputRows",
                "controlWords",
                "producerCount",
                "getLaneCount",
                "ringSlots",
                "recordsPerRingSlot",
                "getBootstrapRecords",
                "getWaveRecords",
                "pipelineWordsPerRing",
                "reserved",
            ],
        )
        self.assertIn(
            "REGISTER_TILING_DEFAULT(A5ExactScanSpeculativeGetTilingData)",
            kernel,
        )

    def test_live_gate_preflights_tiling_before_direct_hccp_setup(self) -> None:
        """A malformed ACLNN tiling closure must fail before KVAF transport.

        The preflight is a real epoch-zero P5 launch, rather than a detached
        workspace call.  It validates the generated executor lifetime while
        deliberately proving that no CandidateSeed or Get request is issued.
        """
        live = _text(TESTS_ROOT / "test_exact_scan_speculative_get_p5_live_c8.py")
        for required in (
            "def _run_epoch_zero_tiling_preflight(",
            'inputs["streaming_epoch"] = torch.zeros_like(',
            'inputs["address_table"] = torch.zeros(',
            'inputs["valid_table"] = torch.zeros(',
            "outputs = _call_p5(inputs)",
            "candidate_counts.count_nonzero()",
            "get_stats[:, :2].count_nonzero()",
            '"P5_TILING_PREFLIGHT_PASS mode=epoch_zero_no_post "',
        ):
            self.assertIn(required, live)
        self.assertLess(
            live.index("_run_epoch_zero_tiling_preflight(publication, device)"),
            live.index("runtime = kvaf.NpuRuntime("),
        )

    def test_tiling_and_kernel_use_c8_score_receipts_not_float_scores(self) -> None:
        tiling = _text(TILING_H)
        kernel = _text(KERNEL_CPP)
        for required in (
            "uint32_t rowCount",
            "uint32_t sourceCapacity",
            "uint32_t poolSize",
            "uint32_t kvObjectCapacity",
            "uint32_t privateArenaCapacity",
            "uint32_t tileReceiptCount",
            "uint32_t scoreWorkspaceWords",
            "uint32_t candidateSeedWords",
        ):
            self.assertIn(required, tiling)
        for required in (
            "constexpr uint32_t kCandidateSeedWords = 5",
            "constexpr uint32_t kReceiptWords = 16",
            "constexpr uint32_t kReceiptContractVersion = 1",
            "scoreWorkspaceU16_ = reinterpret_cast<__gm__ uint16_t*>(scoreWorkspace)",
            "tileReceipts_ = reinterpret_cast<__gm__ int64_t*>(tileReceipts)",
            "scoreSourceLens_ = reinterpret_cast<__gm__ int32_t*>(scoreSourceLens)",
            "LoadReceiptWord(receipt, kReceiptReadyEpochWord) != epoch",
            "LoadReceiptWord(receipt, kReceiptVersionWord) !=",
            "ReadGmByPassDCache<uint16_t>(",
            "scoreWorkspaceU16_ + scoreOffsetU16 + local",
        ):
            self.assertIn(required, kernel)
        # P3's publisher places a graph-stable *relative byte offset* in
        # receipt word 0.  Accepting an allocation-specific device pointer
        # would pass an eager probe yet make a captured replay semantically
        # wrong when workspace storage is rebound.
        if not re.search(
            r"scoreOffsetU16\s*=\s*byteOffset\s*/\s*sizeof\(uint16_t\)",
            kernel,
        ):
            self.fail(
                "receipt word0 must be decoded as relative byteOffset / "
                "sizeof(uint16_t), not as a device pointer"
            )
        self.assertNotIn("workspaceBase =", kernel)
        self.assertNotIn("byteAddress - workspaceBase", kernel)
        self.assertNotIn("liScores_", kernel)
        self.assertNotIn("float*", kernel)

    def test_live_receipt_protocol_is_fail_closed_for_speculation_only(self) -> None:
        """Late expected C8 windows may suppress only speculative prefetch.

        ``score_source_lens`` is essential here: without it, a fixed zeroed
        receipt cannot be distinguished from an expected C8 window which has
        not yet published its ready epoch.  The kernel must skip only truly
        inactive slots, bounded-wait expected slots, and turn a wait expiry
        into telemetry plus a speculative defer.  It must not manufacture a
        fallback score or alter the C8/GuaranteedRecall exact authority.
        """
        kernel = _text(KERNEL_CPP)
        for required in (
            "constexpr uint32_t kProducerStatColumns = 10",
            "constexpr uint32_t kReceiptReadyPollLimit",
            "WaitForReceiptReady(",
            "scoreSourceLens_ + expectedRow",
            "expectedSourceBegin >= static_cast<uint32_t>(sourceLen)",
            "++receiptReadyTimeouts",
            "A timeout only",
            "suppresses a speculative request; Guaranteed Recall retains exactness",
            "Guaranteed Recall retains exactness",
            "stats + 8U",
            "stats + 9U",
        ):
            self.assertIn(required, kernel)

        # The branch must defer before counting the timeout.  That ordering
        # makes timeout a visible non-fatal speculation decision, not a
        # hidden new exact-selection path.
        timeout_block = re.search(
            r"if \(!WaitForReceiptReady\(receipt, epoch, &receiptReadyPolls\)\) \{"
            r"(?P<body>[\s\S]*?)\n            }",
            kernel,
        )
        self.assertIsNotNone(timeout_block)
        body = timeout_block.group("body")
        self.assertLess(body.index("++deferred"), body.index("++receiptReadyTimeouts"))

        inactive = kernel.index(
            "expectedSourceBegin >= static_cast<uint32_t>(sourceLen)"
        )
        wait = kernel.index("WaitForReceiptReady(receipt, epoch, &receiptReadyPolls)")
        self.assertLess(inactive, wait)

    def test_candidate_seed_and_object_dedup_contract(self) -> None:
        kernel = _text(KERNEL_CPP)
        for required in (
            "kCandidateSeedRow = 0U",
            "kCandidateSeedSource = 1U",
            "kCandidateSeedObject = 2U",
            "kCandidateSeedScore = 3U",
            "kCandidateSeedPrivateSlot = 4U",
            "WriteCandidateSeed(producer, candidateCount, row, source,",
            "kvObjectMap_ + mapSourceOffset",
            "kvObjectSlotsSnapshot_ +",
            "privateObjectToSlot_ + objectOffset",
            "privateSourceToSlot_ + sourceOffset",
            "const uint32_t lane =",
            "static_cast<uint32_t>(kvObject) & (kGetLaneCount - 1U)",
        ):
            self.assertIn(required, kernel)

        # CandidateSeed and the SPSC record use bypass GM writes.  The seed
        # has to be written before TryAppendCandidate and that helper's
        # PublishRingSlot release publishes both objects together.  Requiring
        # a DCCI flush here would be wrong: it could clean an older cache line
        # over a newer bypass-GM value.
        seed_write = kernel.index(
            "WriteCandidateSeed(producer, candidateCount, row, source,"
        )
        append = kernel.index("TryAppendCandidate(Ring(producer, lane)")
        publish = kernel.index("DataSyncBarrier<MemDsbT::DDR>();", append)
        self.assertLess(seed_write, append)
        self.assertLess(append, publish)

    def test_post_cq_private_ready_lifecycle_and_c8_noninterference(self) -> None:
        kernel = _text(KERNEL_CPP)
        for required in (
            "BeginKvafDenseBatchGetPostOnly",
            "TryRetireKvafDenseBatchGet",
            "FinishKvafDenseBatchGet",
            "kPrivateReadyPending",
            "kPrivateReadyReady",
            "kPrivateReadyFailed",
            "PublishPrivateReadyForWave",
            "RollBackPrivateClaimsForWave",
            "does not recreate Q/K math",
            "does not mutate",
        ):
            self.assertIn(required, kernel)

        # This regex captures the only legal terminal ordering: CQ finish,
        # status visibility, then private READY/FAILED publication.
        finish_body = re.search(
            r"uint64_t FinishFullGet\([\s\S]*?\n    }\n\n    __aicore__ inline void WriteGetStats",
            kernel,
        )
        self.assertIsNotNone(finish_body)
        body = finish_body.group(0)
        self.assertLess(body.index("FinishKvafDenseBatchGet"), body.index("PublishPrivateReadyForWave"))

    def test_ring_cursor_always_publishes_the_first_unread_record(self) -> None:
        """A descriptor-wave boundary must not replay its final candidate.

        A ring record is a CandidateSeed but ``queued`` counts unique H2D
        objects.  With duplicate source-to-object mappings, the body can make
        ``queued == consumeLimit``.  The cursor must already have advanced at
        that point; otherwise the next wave replays the final record and
        changes the exact Post/CQ descriptor ledger.
        """
        kernel = _text(KERNEL_CPP)
        expected_loop = (
            "while (nextRecord < count && queued < consumeLimit) {\n"
            "                    const uint32_t record = nextRecord++;\n"
            "                    __gm__ int64_t* const words =\n"
            "                        ring.RecordWord(slot, record, 0);"
        )
        self.assertIn(expected_loop, kernel)
        self.assertNotIn(
            "for (; nextRecord < count && queued < consumeLimit;", kernel
        )
        self.assertIn("kGetResultRingGeometryError = 0x1009U", kernel)
        self.assertIn("HasExpectedRingGeometry()", kernel)
        self.assertIn(
            "tiling_->pipelineWordsPerRing == kRingWords", kernel
        )
        self.assertIn("WaitForRingWriteSlot", kernel)
        self.assertIn("allowConsumerProgress", kernel)
        self.assertIn("kProducerRingFreePollLimit", kernel)
        self.assertIn("kPipelineModeParallel", kernel)

        # The legacy overlap benchmark imports its allocation shape from the
        # old test module.  It must include cursor storage as well, otherwise
        # the host/bridge correctly reject a stale 232-word pipeline tensor.
        legacy_test = _text(TESTS_ROOT / "test_exact_scan_speculative_get.py")
        self.assertIn("RING_SLOT_RECORD_WORD = 3 * CACHE_LINE_WORDS", legacy_test)

    def test_inline_c8_score_read_has_bidirectional_ub_lifetime_fence(self) -> None:
        """Full Post may read a C8 tile only inside a V->S->V envelope.

        This is a source-level correctness gate, not a claim that the legacy
        one-wave hook is already the final P5 ring executor.  It prevents a
        future refactor from silently restoring the old `PIPE_V`-only path,
        where Scalar could read unfinished Vector data or Vector/MTE3 could
        recycle the ping-pong tile before CandidateCompact finished.
        """
        vector = _text(P3_VECTOR_CPP)
        marker = "#elif A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE"
        process_vec1 = vector.find("void QLIVector<QLIT>::ProcessVec1")
        self.assertGreaterEqual(process_vec1, 0)
        reduction = vector.find("BatchMulWeightAndReduceSum", process_vec1)
        self.assertGreater(reduction, process_vec1)
        start = vector.find(marker, reduction)
        end = vector.find("#endif", start)
        self.assertGreaterEqual(start, 0)
        self.assertGreater(end, start)
        branch = vector[start:end]
        for required in (
            "GetTPipePtr()->FetchEventID(HardEvent::V_S)",
            "SetFlag<HardEvent::V_S>(scoreVectorToScalar)",
            "WaitFlag<HardEvent::V_S>(scoreVectorToScalar)",
            "streamingKvafHook_.Emit(outBase",
            "GetTPipePtr()->FetchEventID(HardEvent::S_V)",
            "SetFlag<HardEvent::S_V>(scoreScalarToVector)",
            "WaitFlag<HardEvent::S_V>(scoreScalarToVector)",
        ):
            self.assertIn(required, branch)
        self.assertLess(branch.index("WaitFlag<HardEvent::V_S>"),
                        branch.index("streamingKvafHook_.Emit(outBase"))
        self.assertLess(branch.index("streamingKvafHook_.Emit(outBase"),
                        branch.index("WaitFlag<HardEvent::S_V>"))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        P5ExactScanSpeculativeGetContractTest
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print(
            "P5_EXACT_SCAN_SPECULATIVE_GET_CONTRACT_PASS "
            "receipt_u16=1 candidate_seed=1 object_dedup=1 post_cq=1"
        )
    raise SystemExit(0 if result.wasSuccessful() else 1)
