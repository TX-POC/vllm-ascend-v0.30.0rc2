"""Static contract for the only permitted P10 product C8 route.

The test deliberately does not import torch_npu or initialize an NPU.  It
prevents a generic FP32/INT8 operator from being reintroduced as a seemingly
convenient Stage-A source before the actual W4A4C8 provider is resolved.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = OPS_ROOT / "a5_dsa_c8_exact_speculative_resident"
CONTRACT_PATH = TARGET_ROOT / "P10_SOURCE_CONTRACT.json"
PROBE_PATH = TARGET_ROOT / "tools" / "probe_w4a4c8_runtime.py"
RUNNER_PATH = Path(__file__).resolve().parent / "run_p10_w4a4c8_provenance_npu.sh"
VLLM_ROOT = OPS_ROOT.parents[2]

sys.path.insert(0, str(OPS_ROOT))

from a5_dsa_c8_exact_speculative_resident.product_w4a4c8_abi_gate import (  # noqa: E402
    FALLBACK_CALLABLE,
    PRIMARY_CALLABLE,
    W4A4C8_LAYOUT_ID,
    verify_product_w4a4c8_source,
)


class P10W4A4C8SourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT_PATH.read_text())

    def test_product_contract_has_exactly_one_w4a4c8_route(self) -> None:
        product = self.contract["product"]
        self.assertEqual(product["architecture"], W4A4C8_LAYOUT_ID)
        self.assertEqual(product["runtime_callable"], PRIMARY_CALLABLE)
        self.assertEqual(product["runtime_fallback_callable"], FALLBACK_CALLABLE)
        self.assertTrue(product["primary_callable_required"])
        self.assertTrue(product["fallback_is_not_a_stage_a_derivation_target"])
        self.assertEqual(
            product["input_abi"],
            {
                "query": "float8_e4m3fn",
                "key": "float8_e4m3fn",
                "weights": "bfloat16",
                "query_dequant_scale": "float32_scale_metadata",
                "key_dequant_scale": "float32_scale_metadata",
                "topk_output": "int32",
            },
        )

    def test_fp32_derivation_and_conversion_are_explicitly_forbidden(self) -> None:
        forbidden = set(self.contract["product"]["forbidden_product_paths"])
        self.assertTrue(
            {
                "main_dsa_v1_fp32",
                "torch.ops._C_ascend.npu_vllm_quant_lightning_indexer",
                "bf16_to_fp32_reference_widening",
                "generic_vllm_quant_lightning_indexer_source_derivation",
                "generic_lightning_indexer_quant_source_derivation",
            }.issubset(forbidden)
        )
        math_gate = self.contract["math_gate"]
        self.assertEqual(
            math_gate["reference"],
            "installed_primary_torch_npu_w4a4c8_callable",
        )
        self.assertEqual(
            math_gate["comparison"],
            "exact_ordered_sparse_indices_and_score_source_lens_on_identical_w4a4c8_input_tensors",
        )
        self.assertFalse(math_gate["fp32_conversion_or_reference"])
        self.assertFalse(math_gate["runtime_off_mutated_c8_comparison"])

    def test_runtime_source_gate_proves_the_mixed_w4a4c8_product_abi(self):
        product = verify_product_w4a4c8_source(VLLM_ROOT)
        self.assertEqual(product.architecture, W4A4C8_LAYOUT_ID)
        self.assertEqual(product.product_callable, PRIMARY_CALLABLE)
        self.assertEqual(product.query_key_dtype, "float8_e4m3fn")
        self.assertEqual(product.indexer_weight_dtype, "bfloat16")
        self.assertEqual(product.scale_dtype, "float32_scale_metadata")

    def test_stage_a_is_fail_closed_until_the_w4a4c8_numerical_sibling_passes_parity(self):
        derivation = self.contract["source_derivation"]
        self.assertEqual(
            derivation["kind"],
            "explicit_w4a4c8_numerical_sibling_with_runtime_parity_gate",
        )
        self.assertEqual(derivation["environment"], "P10_W4A4C8_SOURCE_ROOT")
        self.assertFalse(derivation["must_be_exact_operator_source_root"])
        self.assertTrue(derivation["source_is_not_runtime_provider_source"])
        self.assertFalse(derivation["must_match_runtime_primary_callable"])
        self.assertTrue(derivation["must_match_runtime_primary_abi"])
        self.assertTrue(
            derivation[
                "activation_requires_strict_ordered_canonical_sparse_indices_parity"
            ]
        )
        self.assertTrue(
            derivation["activation_requires_canonical_score_source_lens_parity"]
        )
        self.assertEqual(
            derivation["source_unresolved_behavior"],
            "fail_closed_no_stage_a_runtime_activation",
        )
        self.assertTrue(derivation["isolated_sibling_build_is_validation_only"])
        self.assertFalse(derivation["automatic_generic_source_selection"])

    def test_stage_a_activation_requires_canonical_topk_and_score_receipt_proofs(self):
        """The Stage-A recipe is buildable, but runtime activation is parity-gated."""

        activation = self.contract["stage_a_activation"]
        self.assertEqual(
            activation["state"],
            "buildable_but_runtime_fail_closed_until_all_required_proofs",
        )
        self.assertEqual(
            activation["required_proofs"],
            [
                "w4a4c8_numerical_sibling_source_provenance",
                "strict_ordered_canonical_topk_and_source_lens_parity",
                "verified_graph_visible_score_surface_and_receipt_publish",
            ],
        )
        self.assertEqual(
            activation["missing_proof_behavior"],
            "fail_closed_no_runtime_activation",
        )

        producer = self.contract["stages"]["producer"]
        canonical = producer["canonical_topk_export"]
        self.assertEqual(canonical["source"], "unmodified_native_w4a4c8_exact_topk")
        self.assertEqual(
            canonical["outputs"],
            [
                {
                    "name": "canonical_sparse_indices",
                    "dtype": "int32",
                    "shape": "[batch,1,sparse_count]",
                    "ordering": "native_w4a4c8_sparse_indices_order_preserved_byte_for_byte",
                },
                {
                    "name": "canonical_score_source_lens",
                    "dtype": "int32",
                    "shape": "[batch]",
                    "semantics": "native_row_valid_source_length",
                },
            ],
        )
        self.assertTrue(canonical["independent_from_transport_or_executor_outputs"])
        self.assertTrue(canonical["requires_distinct_storage"])
        self.assertTrue(canonical["must_not_be_reordered_compacted_or_overwritten"])
        self.assertTrue(canonical["must_exist_even_when_no_transport_candidate_is_published"])

        receipt = producer["score_receipt_abi"]
        self.assertEqual(receipt["score_surface"], {
            "dtype": "uint16",
            "storage": "caller_owned_graph_visible_gm",
            "must_not_alias_native_opaque_workspace": True,
        })
        self.assertEqual(receipt["receipt_ring"]["dtype"], "int64")
        self.assertEqual(receipt["receipt_ring"]["shape"], "[receipt_capacity,5]")
        self.assertEqual(receipt["receipt_state"]["dtype"], "int32")
        self.assertEqual(
            receipt["receipt_state"]["states"],
            ["FREE", "FILLING", "READY", "CLAIMED", "DONE"],
        )
        self.assertEqual(
            receipt["receipt_state"]["publication_order"],
            "score_surface_visible_before_FILLING_to_READY",
        )

        source_base = self.contract["stage_a_source_base"]
        self.assertEqual(source_base["operator"], "A5P10W4A4C8ScoreReceipt")
        self.assertEqual(source_base["kernel"], "a5_p10_w4a4c8_score_receipt")
        self.assertEqual(source_base["source_root"], "stage_a_w4a4c8_score_receipt")
        self.assertEqual(
            source_base["build_state"],
            "isolated_stage_recipe_requires_exact_source_pair",
        )
        self.assertEqual(
            source_base["outputs"],
            [
                "canonical_sparse_indices",
                "canonical_score_source_lens",
                "score_surface",
                "receipt_ring",
                "receipt_state",
            ],
        )

    def test_forward_resident_layout_is_fixed_double_band_not_per_object_map(self):
        resource = self.contract["resource_model"]
        slots = resource["resident_slots"]
        self.assertEqual(slots["layout"], "fixed_forward_double_band")
        self.assertEqual(
            slots["capacity_invariant"],
            "spec_band_capacity + active_rows * topk_per_row <= resident_row_count",
        )
        self.assertEqual(slots["spec_band"]["assignment"], "monotonic_per_forward_spec_cursor")
        self.assertFalse(slots["spec_band"]["reused_within_forward"])
        self.assertEqual(slots["recall_band"]["topk_per_row"], 2048)
        self.assertEqual(
            slots["recall_band"]["assignment"],
            "deterministic_absolute_row_by_active_row_and_topk_rank",
        )
        self.assertEqual(slots["source_to_forward_slot"]["initial_value"], -1)
        self.assertIn("all_row_object_canonical_slot_map", slots["forbidden"])
        self.assertIn("p8_arena_or_aba_protocol", slots["forbidden"])

    def test_probe_and_runner_never_build_or_select_a_generic_source(self) -> None:
        probe = PROBE_PATH.read_text()
        runner = RUNNER_PATH.read_text()
        self.assertIn("P10_W4A4C8_STAGE_A ready=0", probe)
        self.assertIn("P10_W4A4C8_STAGE_A_SOURCE", probe)
        self.assertIn("P10_W4A4C8_PROVIDER_BINARY", probe)
        self.assertIn("source_derivation_permitted=0", probe)
        self.assertIn("buildable_score_receipt_source_runtime_parity_pending", probe)
        self.assertIn("stage_a_opp=buildable_not_installed", runner)
        self.assertIn("product_route=runtime_fail_closed_until_parity", runner)
        for forbidden in (
            "VllmQuantLightningIndexer",
            "LightningIndexerQuant",
            "main_dsa_v1_fp32",
            "build_nanovllm_ops",
            "cmake --build",
            "pip install",
        ):
            self.assertNotIn(forbidden, probe + runner)

    def test_probe_is_host_safe_and_compiles_as_python(self) -> None:
        with tempfile.TemporaryDirectory() as cache:
            environment = dict(os.environ, PYTHONPYCACHEPREFIX=cache)
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(PROBE_PATH)],
                check=False,
                text=True,
                capture_output=True,
                env=environment,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = subprocess.run(
            ["bash", "-n", str(RUNNER_PATH)],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
