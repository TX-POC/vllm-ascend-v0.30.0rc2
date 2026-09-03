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
        self.assertFalse(math_gate["fp32_conversion_or_reference"])
        self.assertFalse(math_gate["runtime_off_mutated_c8_comparison"])

    def test_runtime_source_gate_proves_the_mixed_w4a4c8_product_abi(self):
        product = verify_product_w4a4c8_source(VLLM_ROOT)
        self.assertEqual(product.architecture, W4A4C8_LAYOUT_ID)
        self.assertEqual(product.product_callable, PRIMARY_CALLABLE)
        self.assertEqual(product.query_key_dtype, "float8_e4m3fn")
        self.assertEqual(product.indexer_weight_dtype, "bfloat16")
        self.assertEqual(product.scale_dtype, "float32_scale_metadata")

    def test_stage_a_is_fail_closed_until_exact_runtime_source_is_proven(self):
        derivation = self.contract["source_derivation"]
        self.assertEqual(
            derivation["kind"], "explicit_w4a4c8_runtime_source_only"
        )
        self.assertEqual(derivation["environment"], "P10_W4A4C8_SOURCE_ROOT")
        self.assertTrue(derivation["must_match_runtime_primary_callable"])
        self.assertEqual(
            derivation["source_unresolved_behavior"],
            "fail_closed_no_stage_a_opp_build",
        )
        self.assertFalse(derivation["automatic_generic_source_selection"])

    def test_probe_and_runner_never_build_or_select_a_generic_source(self) -> None:
        probe = PROBE_PATH.read_text()
        runner = RUNNER_PATH.read_text()
        self.assertIn("P10_W4A4C8_STAGE_A ready=0", probe)
        self.assertIn("source_and_opp_provenance_unresolved", probe)
        self.assertIn("stage_a_build=disabled", runner)
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
