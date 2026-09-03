"""Fast source gate for the product-derived P10 Stage-A seam.

The test intentionally avoids CANN and NPU initialization.  It verifies that
the checked-in foundation is an explicit source derivation with a bounded GM
receipt protocol, not a hidden binary overlay or a transport closure.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[1]
STAGE_A_ROOT = (
    OPS_ROOT
    / "a5_dsa_c8_exact_speculative_resident"
    / "stage_a_vllm_quant"
)
MANIFEST = STAGE_A_ROOT / "STAGE_A_VLLM_QUANT_MANIFEST.json"
RING = STAGE_A_ROOT / "op_kernel" / "arch35" / "a5_c8_receipt_ring.h"
SEAM = STAGE_A_ROOT / "op_kernel" / "arch35" / "a5_c8_score_receipt_seam.h"
STAGER = STAGE_A_ROOT / "tools" / "stage_product_c8_sibling.sh"


class P10StageASourceDerivationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.ring = RING.read_text()
        cls.seam = SEAM.read_text()
        cls.stager = STAGER.read_text()

    def test_manifest_declares_a_fresh_supported_product_derivation(self):
        self.assertEqual(self.manifest["stage"], "stage_a_exact_score_receipt")
        self.assertEqual(
            self.manifest["operator_family"], "vllm_quant_lightning_indexer"
        )
        self.assertEqual(self.manifest["operator_symbol"], "VllmQuantLightningIndexer")
        derivation = self.manifest["source_derivation"]
        self.assertTrue(derivation["fresh_source_only_clone"])
        self.assertFalse(derivation["original_source_mutated"])
        self.assertFalse(derivation["automatic_text_transform"])
        self.assertFalse(derivation["build_before_formal_seam_derivation"])
        self.assertEqual(
            self.manifest["score_seam"]["anchor"], "BatchMulWeightAndReduceSum"
        )
        self.assertEqual(
            self.manifest["score_seam"]["position"],
            "after_score_surface_visibility_before_original_topk_consumption",
        )

    def test_ring_has_descriptor_only_abi_and_minimal_forward_owned_states(self):
        receipt = self.manifest["receipt_ring"]
        self.assertTrue(receipt["descriptor_only"])
        self.assertEqual(receipt["score_representation"], "native_c8_uint16_score_bits")
        self.assertEqual(
            receipt["states"], ["empty", "filling", "published", "claimed", "done"]
        )
        for required in (
            "kEmpty = 0U",
            "kFilling = 1U",
            "kPublished = 2U",
            "kClaimed = 3U",
            "kDone = 4U",
            "uint64_t score_surface_offset_bytes",
            "uint32_t score_element_bytes",
            "static_assert(sizeof(ScoreReceiptHeader) == 32U",
            "static_assert(sizeof(ScoreReceiptSlot) == 40U",
        ):
            self.assertIn(required, self.ring)
        for forbidden in (
            "struct CandidateSeed",
            "kReceiptSeedCapacity",
            "ddr_offset_bytes",
            "copy_bytes",
            "score_bits;",
        ):
            self.assertNotIn(forbidden, self.ring)

    def test_publish_and_claim_have_explicit_ddr_release_acquire_order(self):
        publish_start = self.ring.index("bool Publish(")
        publish_end = self.ring.index("void AbortBeforePublish", publish_start)
        publish = self.ring[publish_start:publish_end]
        payload = publish.index("slot->header = published_header")
        barrier = publish.index("AtomicOps::DdrPublishBarrier()", payload)
        release = publish.index("ReceiptState::kPublished", barrier)
        self.assertLess(payload, barrier)
        self.assertLess(barrier, release)

        claim_start = self.ring.index("bool TryClaim(")
        claim_end = self.ring.index("bool MarkDone", claim_start)
        claim = self.ring[claim_start:claim_end]
        state_claim = claim.index("ReceiptState::kClaimed")
        acquire = claim.index("AtomicOps::DdrAcquireBarrier()", state_claim)
        sequence_check = claim.index("slot->sequence != expected_sequence", acquire)
        self.assertLess(state_claim, acquire)
        self.assertLess(acquire, sequence_check)

    def test_only_done_can_be_recycled_and_no_hidden_cross_forward_identity_exists(self):
        recycle_start = self.ring.index("bool RecycleDone(")
        recycle = self.ring[recycle_start:]
        self.assertIn("ReceiptState::kDone", recycle)
        self.assertIn("ReceiptState::kEmpty", recycle)
        for forbidden in ("forward_epoch", "graph_replay", "layer_id", "arena_health"):
            self.assertNotIn(forbidden, self.ring)

    def test_seam_publishes_only_a_completed_raw_score_window(self):
        for required in (
            "ExactScoreWindowPlan",
            "ReservedExactScoreWindow",
            "TryReserveScoreWindow",
            "PublishReservedScoreWindowAfterDma",
            "AbortReservedScoreWindowBeforePublish",
            "ExactScoreWindow",
            "score_surface_offset_bytes",
            "score-DMA completion",
            "publisher.TryReserve",
            "publisher.Publish",
            "publisher.AbortBeforePublish",
            "lease.slot_index",
        ):
            self.assertIn(required, self.seam)
        reserve = self.seam.index("bool TryReserveScoreWindow(")
        slot_offset = self.seam.index("lease.slot_index", reserve)
        publish = self.seam.index("bool PublishReservedScoreWindowAfterDma(")
        self.assertLess(reserve, slot_offset)
        self.assertLess(slot_offset, publish)
        for forbidden in (
            "Resolver",
            "TryBuildSeed",
            "CandidateScratch",
            "CandidateSeed",
            "threshold",
            "source_to_object",
            "resident_slot",
            "DirectHCCP",
            "HCCP",
            "CQ",
            "kvaf",
            "ProcessTopK",
            "sparse_indices",
            "runtime_enable",
            "runtime_off",
        ):
            self.assertNotIn(forbidden, self.seam)

    def test_stage_a_implementation_cannot_revive_retired_overlay_routes(self):
        source_text = "\n".join((self.ring, self.seam, self.stager))
        for forbidden in ("P8", "P9", "source_overlay", "runtime_enable", "runtime_off"):
            self.assertNotIn(forbidden, source_text)

    def test_stager_is_fail_closed_source_only_and_does_not_auto_transform(self):
        self.assertIn("probe_product_c8_provenance.py", self.stager)
        self.assertIn("family=vllm_quant_lightning_indexer", self.stager)
        self.assertIn("supported=1", self.stager)
        self.assertIn("source_head_mismatch", self.stager)
        self.assertIn("source_tracked_changes_present", self.stager)
        self.assertIn("fresh_package_root_exists", self.stager)
        self.assertIn("tar -C", self.stager)
        self.assertIn("MANUAL_SEAM_DERIVATION_REQUIRED", self.stager)
        self.assertIn("opp_build_started=0", self.stager)
        for forbidden in ("sed -i", "perl -pi", "apply_patch", "git checkout"):
            self.assertNotIn(forbidden, self.stager)
        result = subprocess.run(
            ["bash", "-n", str(STAGER)], check=False, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_headers_are_host_parseable_before_cann_binding(self):
        compiler = shutil.which("c++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("No host C++ compiler is available.")
        source = f"""
            #include \"{RING}\"
            #include \"{SEAM}\"
            struct MockOps {{
              static unsigned CompareExchange(unsigned* p, unsigned expected, unsigned desired) {{
                const unsigned before = *p; if (before == expected) *p = desired; return before;
              }}
              static void StoreRelease(unsigned* p, unsigned value) {{ *p = value; }}
              static void DdrPublishBarrier() {{}}
              static void DdrAcquireBarrier() {{}}
            }};
            using namespace vllm_ascend::a5_dsa_c8_exact_speculative_resident::stage_a;
            int main() {{
              ScoreReceiptSlot slots[2]{{}};
              ScoreReceiptRing ring{{slots, 2U}};
              ScoreReceiptRingPublisher<MockOps> publisher;
              ReceiptWriteLease lease{{}};
              return publisher.TryReserve(&ring, 1U, &lease) ? 0 : 1;
            }}
        """
        with tempfile.TemporaryDirectory() as temporary:
            cpp = Path(temporary) / "receipt_contract.cc"
            cpp.write_text(source)
            result = subprocess.run(
                [compiler, "-std=c++14", "-fsyntax-only", str(cpp)],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
