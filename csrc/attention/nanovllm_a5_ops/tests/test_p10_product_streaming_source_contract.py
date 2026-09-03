"""Static gate for the product-derived P10 streaming-C8 implementation.

P10 is intentionally a fresh product C8 path.  This test locks down that
choice before any OPP is built: a P9/P8 overlay, an in-kernel disabled branch,
or a fixed-core Get partition would recreate the failed prototype architecture.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = OPS_ROOT / "a5_dsa_c8_exact_speculative_resident"
CONTRACT_PATH = TARGET_ROOT / "P10_SOURCE_CONTRACT.json"
PRODUCT_C8_ROOT = OPS_ROOT.parent / "vllm_quant_lightning_indexer"
GENERIC_C8_ROOT = OPS_ROOT.parent / "lightning_indexer_quant"
PROBE_PATH = TARGET_ROOT / "tools" / "probe_product_c8_provenance.py"
RUNTIME_PROBE_PATH = TARGET_ROOT / "tools" / "probe_runtime_dispatch.py"


class P10ProductStreamingSourceContractTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT_PATH.read_text())

    def _run_probe(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(PROBE_PATH),
                "--source-root",
                str(root),
            ],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_uses_an_explicit_resolved_product_c8_input_with_provenance_anchors(self):
        source = self.contract["source"]
        self.assertEqual(source["kind"], "product_a5_c8_resolved_clone")
        self.assertNotIn("relative_root", source)
        resolved_input = source["resolved_input"]
        self.assertEqual(resolved_input["environment"], "P10_PRODUCT_C8_SOURCE_ROOT")
        self.assertEqual(resolved_input["command_line"], "--source-root")
        self.assertTrue(resolved_input["must_be_exact_operator_root"])
        self.assertTrue(resolved_input["required_before_staging"])

        selection = source["selection"]
        self.assertTrue(selection["must_report_family_root_sha_anchors_and_abi"])
        self.assertTrue(selection["must_report_the_actual_dsa_frontend_and_registered_operator"])
        self.assertTrue(selection["must_prove_product_abi_matches_the_source_family_before_staging"])
        self.assertTrue(selection["ambiguous_or_incomplete_source_fails_closed"])
        self.assertTrue(selection["unsupported_family_reports_supported_zero"])

        families = {item["family"]: item for item in source["families"]}
        self.assertEqual(
            set(families),
            {"vllm_quant_lightning_indexer", "lightning_indexer_quant"},
        )
        provenance = families["vllm_quant_lightning_indexer"]
        self.assertTrue(provenance["p10_execution_supported"])
        self.assertEqual(provenance["operator_symbol"], "VllmQuantLightningIndexer")
        self.assertEqual(
            provenance["source_declared_abi"],
            "fp8_qk_fp32_weights_fp32_scales",
        )
        generic = families["lightning_indexer_quant"]
        self.assertFalse(generic["p10_execution_supported"])
        self.assertEqual(generic["operator_symbol"], "LightningIndexerQuant")

        selected = PRODUCT_C8_ROOT.resolve()
        for relative_path in provenance["required_files"]:
            self.assertTrue(
                (selected / relative_path).is_file(),
                f"missing product C8 provenance file: {relative_path}",
            )
        op_def = (
            selected / "op_host" / "vllm_quant_lightning_indexer_def.cpp"
        ).read_text()
        self.assertIn(provenance["operator_symbol"], op_def)
        vector_service = (
            selected
            / "op_kernel"
            / "arch35"
            / "quant_lightning_indexer_service_vector.h"
        ).read_text()
        self.assertIn(provenance["score_seam"]["anchor"], vector_service)
        self.assertIn(provenance["final_topk"]["anchor"], vector_service)

    def test_provenance_probe_reports_the_supported_vllm_family(self):
        accepted = self._run_probe(PRODUCT_C8_ROOT)
        self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
        self.assertIn("P10_PRODUCT_C8_PROVENANCE status=pass", accepted.stdout)
        self.assertIn("reason=ok", accepted.stdout)
        self.assertIn("family=vllm_quant_lightning_indexer", accepted.stdout)
        self.assertIn("operator=VllmQuantLightningIndexer", accepted.stdout)
        self.assertIn("supported=1", accepted.stdout)
        self.assertIn("source_sha=", accepted.stdout)
        self.assertIn("anchors=BatchMulWeightAndReduceSum,ProcessTopK", accepted.stdout)
        self.assertIn("abi=source:fp8_qk_fp32_weights_fp32_scales", accepted.stdout)

    def test_product_bf16_and_official_reference_fp32_are_explicitly_bridged(self):
        gate = self.contract["product_abi_equivalence_gate"]
        self.assertEqual(
            gate["reference_callable"],
            "a5_official_c8_reference.npu_vllm_quant_lightning_indexer",
        )
        self.assertEqual(gate["reference_weight_dtype"], "float32")
        self.assertTrue(gate["frontend_selection"]["must_be_explicit"])
        self.assertFalse(gate["frontend_selection"]["automatic_callable_inference"])
        frontends = {item["name"]: item for item in gate["frontends"]}
        self.assertEqual(set(frontends), {"dsa_offload_bf16", "main_dsa_v1_fp32"})
        self.assertEqual(
            frontends["dsa_offload_bf16"]["product_callable"],
            "torch_npu.npu_quant_lightning_indexer",
        )
        self.assertEqual(
            frontends["dsa_offload_bf16"]["product_weight_dtype"],
            "bfloat16",
        )
        self.assertEqual(
            frontends["dsa_offload_bf16"]["reference_weight_from_product"],
            "lossless_bfloat16_to_float32_widen",
        )
        self.assertEqual(
            frontends["main_dsa_v1_fp32"]["product_callable"],
            "torch.ops._C_ascend.npu_vllm_quant_lightning_indexer",
        )
        self.assertEqual(
            frontends["main_dsa_v1_fp32"]["product_weight_dtype"], "float32"
        )
        self.assertEqual(
            frontends["main_dsa_v1_fp32"]["reference_weight_from_product"],
            "identity_float32",
        )
        self.assertFalse(gate["product_fallback_callable_is_accepted"])
        self.assertEqual(
            gate["comparison"],
            "bitwise_sparse_indices_on_identical_fp8_qk_and_frontend_declared_weight_values",
        )
        self.assertEqual(
            gate["must_prove_before_npu_compare"],
            [
                "selected_product_frontend_source_contract",
                "reference_source_git_head",
                "reference_opp_p3_provenance",
                "reference_bridge_p10_provenance",
            ],
        )

    def test_provenance_probe_reports_generic_family_without_mislabeling_it(self):
        generic = self._run_probe(GENERIC_C8_ROOT)
        self.assertEqual(generic.returncode, 0, generic.stdout + generic.stderr)
        self.assertIn("P10_PRODUCT_C8_PROVENANCE status=pass", generic.stdout)
        self.assertIn("reason=family_not_implemented", generic.stdout)
        self.assertIn("family=lightning_indexer_quant", generic.stdout)
        self.assertIn("operator=LightningIndexerQuant", generic.stdout)
        self.assertIn("supported=0", generic.stdout)

    def test_provenance_probe_fails_closed_on_ambiguity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            host = root / "op_host"
            host.mkdir()
            (host / "vllm_quant_lightning_indexer_def.cpp").write_text(
                "class VllmQuantLightningIndexer {};\n"
                "OP_ADD(VllmQuantLightningIndexer);\n"
            )
            (host / "lightning_indexer_quant_def.cpp").write_text(
                "class LightningIndexerQuant {};\n"
                "OP_ADD(LightningIndexerQuant);\n"
            )
            ambiguous = self._run_probe(root)
        self.assertEqual(ambiguous.returncode, 2)
        self.assertIn("reason=source_family_ambiguous", ambiguous.stdout)
        self.assertIn(
            "candidates=vllm_quant_lightning_indexer,lightning_indexer_quant",
            ambiguous.stdout,
        )

    def test_provenance_probe_fails_closed_for_unsupported_and_incomplete_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            unsupported = self._run_probe(Path(temporary))
        self.assertEqual(unsupported.returncode, 2)
        self.assertIn("reason=source_family_unsupported", unsupported.stdout)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            host = root / "op_host"
            host.mkdir()
            (host / "vllm_quant_lightning_indexer_def.cpp").write_text(
                "class VllmQuantLightningIndexer {};\n"
                "OP_ADD(VllmQuantLightningIndexer);\n"
            )
            incomplete = self._run_probe(root)
        self.assertEqual(incomplete.returncode, 2)
        self.assertIn("reason=source_family_incomplete", incomplete.stdout)

    def test_runtime_dispatch_probe_is_host_safe_and_names_all_candidates(self):
        text = RUNTIME_PROBE_PATH.read_text()
        self.assertIn("torch_npu.npu_quant_lightning_indexer", text)
        self.assertIn("_C_ascend.npu_lightning_indexer_quant", text)
        self.assertIn("_C_ascend.npu_vllm_quant_lightning_indexer", text)
        for prohibited in (
            "set_device",
            "synchronize",
            "torch.ones",
            "torch.zeros",
            "device=",
        ):
            self.assertNotIn(prohibited, text)

        result = subprocess.run(
            [sys.executable, str(RUNTIME_PROBE_PATH)],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("P10_RUNTIME_DISPATCH_PROBE status=", result.stdout)

    def test_feature_off_is_an_external_dispatch_not_an_in_kernel_noop(self):
        routing = self.contract["routing"]
        self.assertEqual(routing["feature_selection"], "outside_operator")
        self.assertEqual(
            routing["legacy_off_path"], "untouched_native_c8_then_legacy_get"
        )
        self.assertFalse(routing["operator_has_runtime_enable_input"])
        self.assertFalse(routing["operator_has_runtime_off_branch"])

    def test_score_receipt_is_not_a_hidden_hccp_or_topk_finalizer(self):
        producer = self.contract["stages"]["producer"]
        self.assertEqual(producer["role"], "raw_exact_score_receipt_only")
        self.assertFalse(producer["may_issue_hccp"])
        self.assertFalse(producer["may_wait_for_cq"])
        self.assertFalse(producer["may_modify_c8_topk"])
        self.assertFalse(producer["may_read_resident_mapping"])
        self.assertFalse(producer["may_allocate_resident_slot"])
        self.assertEqual(
            producer["required_seam"],
            "after_normal_score_dma_and_streaming_score_window_dma_completion",
        )
        self.assertEqual(
            producer["receipt_payload"],
            "row_source_range_score_surface_offset_score_count_sequence",
        )

    def test_executor_and_finalizer_are_separate_device_stages(self):
        stages = self.contract["stages"]
        executor = stages["executor"]
        finalizer = stages["finalizer"]
        self.assertEqual(
            executor["role"],
            "exact_score_scan_candidate_compact_device_direct_hccp_post_and_nonblocking_cq_poll",
        )
        self.assertEqual(executor["execution_domain"], "aicore_executor")
        self.assertFalse(executor["simt_may_call_hccp_directly"])
        self.assertTrue(executor["owns_dynamic_queue_lease"])
        self.assertFalse(executor["fixed_aiv_core_partition"])
        self.assertEqual(
            finalizer["role"], "separate_device_stage_after_original_topk_output"
        )
        self.assertFalse(finalizer["may_be_injected_into_process_topk"])
        self.assertIn("original_sparse_indices_gm", finalizer["reads"])
        self.assertEqual(
            finalizer["writes"],
            ["resident_slots", "attention_slots", "resident_seq_lengths"],
        )

    def test_resource_model_remains_dynamic_not_a_31_plus_1_partition(self):
        resource_model = self.contract["resource_model"]
        self.assertEqual(resource_model["queue_resources"], "fixed_pool_dynamic_lease")
        self.assertEqual(
            resource_model["stage_a_receipts"],
            "bounded_raw_score_descriptor_ring",
        )
        self.assertEqual(
            resource_model["candidate_microbatches"],
            "two_local_64_seed_buffers_owned_by_stage_b",
        )
        self.assertFalse(resource_model["fixed_get_cores"])
        self.assertFalse(resource_model["fixed_li_get_core_partition"])

    def test_future_p10_implementation_cannot_import_old_prototypes(self):
        # The manifest is intentionally excluded: it records the policy in
        # prose-like keys, while implementation files must not refer to the
        # retired P8/P9 paths or introduce a disabled in-kernel control path.
        forbidden = ("P8", "P9", "runtime_enable", "runtime_off")
        source_suffixes = {".h", ".hpp", ".cc", ".cpp", ".py"}
        for path in TARGET_ROOT.rglob("*"):
            if path == CONTRACT_PATH or path.suffix not in source_suffixes:
                continue
            text = path.read_text(errors="replace")
            for token in forbidden:
                self.assertNotIn(token, text, f"forbidden {token} in {path}")


if __name__ == "__main__":
    unittest.main()
