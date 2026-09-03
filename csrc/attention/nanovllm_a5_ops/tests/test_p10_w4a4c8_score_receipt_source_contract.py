"""Fail-closed source contract for P10's real W4A4C8 Stage-A entry.

This is deliberately a cheap pre-CANN gate.  It proves source closure,
graph-alias ABI, direct score-surface mapping, and publication ordering before
an expensive isolated OPP build.  Runtime numerical parity remains a separate
hardware gate.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = OPS_ROOT / "a5_dsa_c8_exact_speculative_resident"
STAGE_ROOT = TARGET_ROOT / "stage_a_w4a4c8_score_receipt"
CONTRACT_PATH = TARGET_ROOT / "P10_SOURCE_CONTRACT.json"


class P10W4A4C8ScoreReceiptSourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT_PATH.read_text())
        cls.ops = json.loads((STAGE_ROOT / "ops.json").read_text())
        cls.definition = (
            STAGE_ROOT / "op_host" / "a5_p10_w4a4c8_score_receipt_def.cpp"
        ).read_text()
        cls.abi = (
            STAGE_ROOT / "op_kernel" / "a5_p10_w4a4c8_score_receipt_abi.h"
        ).read_text()
        cls.qli = (
            STAGE_ROOT / "op_kernel" / "a5_p10_w4a4c8_score_receipt_qli.h"
        ).read_text()
        cls.vector = (
            STAGE_ROOT
            / "op_kernel"
            / "arch35"
            / "quant_lightning_indexer_service_vector.h"
        ).read_text()
        cls.kernel = (
            STAGE_ROOT / "op_kernel" / "a5_p10_w4a4c8_score_receipt.cpp"
        ).read_text()
        cls.build = (
            STAGE_ROOT / "build_stage_a_w4a4c8_score_receipt_opp.sh"
        ).read_text()

    def test_isolated_real_build_route_is_present(self):
        for path in (
            STAGE_ROOT / "ops.json",
            STAGE_ROOT / "build_stage_a_w4a4c8_score_receipt_opp.sh",
            STAGE_ROOT / "op_host" / "a5_p10_w4a4c8_score_receipt_def.cpp",
            STAGE_ROOT / "op_kernel" / "a5_p10_w4a4c8_score_receipt.cpp",
        ):
            self.assertTrue(path.is_file(), path)
        self.assertFalse((STAGE_ROOT / "CMakeLists.txt").exists())
        self.assertIn("msopgen gen", self.build)
        self.assertIn("fresh_package_root_exists", self.build)
        self.assertIn("duplicate_or_missing_op_registration", self.build)
        self.assertIn("CANN_3RD_LIB_PATH", self.build)
        self.assertIn("stage_a_must_not_reset_concurrent_receipt_state", self.build)
        # CANN's source generator normalizes the W4/A4/C8 runs in the
        # operator name.  The standalone route must replace that generated
        # kernel with the checked-in source and remove the original spelling,
        # just like the combined P10 product route.
        for token in (
            "a5_p10_w4_a4_c8_score_receipt.cpp",
            "a5_p10_w4a4c8_score_receipt.cpp",
            'cp -p "${SOURCE_KERNEL}" "${GENERATED_KERNEL}"',
            'rm -f "${SOURCE_KERNEL}"',
            "CANONICAL_OP_FILE=\"a5_p10_w4_a4_c8_score_receipt\"",
            "P10_STAGE_A_OPFILE_GATE op_file=%s source=1 entry=1",
            'GENERATED_STUB="${GENERATED}/op_host/a5_p10_w4_a4_c8_score_receipt.cpp"',
        ):
            self.assertIn(token, self.build)

    def test_operator_abi_is_exact_w4a4c8_graph_alias_abi(self):
        self.assertIn("OP_ADD(A5P10W4A4C8ScoreReceipt)", self.definition)
        self.assertIn(
            'ExtendCfgInfo("opFile.value", "a5_p10_w4_a4_c8_score_receipt")',
            self.definition,
        )
        self.assertNotIn(
            'ExtendCfgInfo("opFile.value", "a5_p10_w4a4c8_score_receipt")',
            self.definition,
        )
        self.assertIn(
            "void a5_p10_w4_a4_c8_score_receipt(", self.kernel
        )
        self.assertNotIn(
            "void a5_p10_w4a4c8_score_receipt(", self.kernel
        )
        for dtype in ("DT_FLOAT8_E4M3FN", "DT_BF16", "DT_FLOAT"):
            self.assertIn(dtype, self.definition)
        for name in (
            "query",
            "key",
            "weights",
            "query_dequant_scale",
            "key_dequant_scale",
            "actual_seq_lengths_query",
            "candidate_lens",
            "block_table",
            "canonical_sparse_indices_buffer",
            "canonical_score_source_lens_buffer",
            "score_surface_buffer",
            "receipt_ring_buffer",
            "receipt_state_buffer",
            "forward_control_buffer",
        ):
            self.assertIn(f'Input("{name}")', self.definition)
        for name in (
            "canonical_sparse_indices_alias",
            "canonical_score_source_lens_alias",
            "score_surface_alias",
            "receipt_ring_alias",
            "receipt_state_alias",
            "forward_control_alias",
        ):
            self.assertIn(f'Output("{name}")', self.definition)
        self.assertIn("scoreSurface.GetDim(1) != s1Base", self.definition)
        self.assertIn("scoreSurface.GetDim(2) != sourceCapacity", self.definition)
        self.assertIn("receiptRing.GetDim(0)", self.definition)
        self.assertIn("forwardControl.GetDim(0) < A5_P10_FORWARD_CONTROL_MIN_WORDS",
                      self.definition)

        # CANN's ops.json spelling for a BF16 tensor is ``bf16``.  This must
        # agree with both the product contract and host DT_BF16 gate: a
        # ``float`` declaration would cause ACLNN to derive an FP32 ABI before
        # the runtime ever reaches the W4A4C8 math check.
        self.assertEqual(len(self.ops), 1)
        input_types = {
            item["name"]: item["type"]
            for item in self.ops[0]["input_desc"]
        }
        self.assertEqual(input_types["weights"], ["bf16"])

    def test_direct_score_map_preserves_original_topk_and_event_ownership(self):
        self.assertIn("fp8_e4m3fn_t, fp8_e4m3fn_t, float, uint16_t, int32_t",
                      self.qli)
        self.assertIn("vectorService_.ProcessVec1(run)", self.qli)
        self.assertIn("vectorService_.ProcessTopK(run)", self.qli)
        self.assertIn("BindGraphScoreSurfaceSlice", self.qli)
        self.assertIn("scoreSurfaceSliceStrideBytes", self.qli)
        self.assertIn("extern \"C\" __global__ __aicore__", self.kernel)
        self.assertIn("KERNEL_TYPE_MIX_AIC_1_2", self.kernel)
        seam = self.vector.index("DataCopyPad(scoreGm[vec1OutGmOffset]")
        publish = self.vector.index("PublishScoreReceiptsAfterNativeScoreDma", seam)
        legacy_event = self.vector.index("SetFlag<HardEvent::MTE3_V>", publish)
        self.assertLess(seam, publish)
        self.assertLess(publish, legacy_event)
        self.assertIn("ProcessTopK(const QLICommon::RunInfo &info)", self.vector)

    def test_receipt_publication_is_ordered_and_stage_a_does_not_reset_it(self):
        for state in ("FREE", "FILLING", "READY", "CLAIMED", "DONE"):
            self.assertIn(state, self.abi)
        self.assertIn("A5_P10_FORWARD_CONTROL_PRODUCER_SEALED_WORD", self.abi)
        self.assertIn("forward_control[PRODUCER_SEALED_WORD]", self.abi)
        self.assertIn("FREE -> FILLING -> READY", self.abi)
        self.assertNotIn("PRODUCER_SEALED = 5", self.abi)
        self.assertIn("WaitForMte3AndPublishDdr", self.vector)
        self.assertIn("DataSyncBarrier<MemDsbT::DDR>()", self.vector)
        self.assertIn("A5P10ReceiptState::FILLING", self.vector)
        self.assertIn("A5P10ReceiptState::READY", self.vector)
        self.assertIn("WriteProducerSealed", self.vector)
        self.assertIn("SyncAll<false>()", self.qli)
        self.assertNotIn("InitializeReceiptStateShard", self.qli)
        self.assertNotIn("InitGlobalMemory", self.vector)
        self.assertIn("Stage-A must never clear receipt_state", self.qli)

    def test_stage_a_excludes_transport_slots_and_legacy_closures(self):
        code = "\n".join(
            path.read_text()
            for path in STAGE_ROOT.rglob("*")
            if path.suffix in {".cpp", ".h"}
        )
        for forbidden in (
            "A5FusedLiManageC8",
            "DirectHccp",
            "KVAF",
            "destinationSlots",
            "missCounts",
            "a5_p7_",
            "a5_p8_",
        ):
            self.assertNotIn(forbidden, code)
        receipt = self.contract["stages"]["producer"]["score_receipt_abi"]
        self.assertEqual(receipt["score_surface"]["dtype"], "uint16")
        self.assertEqual(receipt["receipt_ring"]["dtype"], "int64")
        self.assertEqual(receipt["receipt_state"]["dtype"], "int32")


if __name__ == "__main__":
    unittest.main()
