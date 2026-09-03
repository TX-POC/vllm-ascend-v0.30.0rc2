"""Static contract for the sole P10 product build-and-proof runner.

This catches orchestration drift locally, before an A5 host spends time
building a wrong source closure. It checks the concrete P10 path rather than
trying to simulate CANN or Direct-HCCP on a development machine.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[1]
RUNNER = OPS_ROOT / "tests" / "run_p10_w4a4c8_forward_resident_npu.sh"
FIXTURE = OPS_ROOT / "tests" / "test_p10_w4a4c8_forward_resident_runtime.py"


class P10ProductRunnerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = RUNNER.read_text()
        cls.fixture = FIXTURE.read_text()

    def test_runner_is_valid_and_binds_its_own_exact_source_tree(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(RUNNER)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for token in (
            "runner_not_from_exact_vllm_source",
            "require_exact_clean_source vllm",
            "require_exact_clean_source kvaf",
            "cd \"${vllm_root}\"",
            "P10_VLLM_EXPECTED_SHA",
            "P10_KVAF_EXPECTED_SHA",
        ):
            self.assertIn(token, self.runner)

    def test_runner_accepts_only_one_combined_vendor_and_direct_dispatch(self) -> None:
        for token in (
            "build_p10_target_assembly.sh",
            "combined_product_opapi_count_invalid",
            "A5P10ForwardPrepare,A5P10W4A4C8ScoreReceipt,A5P10ScoreReceiptForwardResident",
            "NANOVLLM_A5_P10_PRODUCT_OPAPI_LIB",
            "NANOVLLM_A5_P10_PRODUCT_OPP_VENDOR",
            "TASK_QUEUE_ENABLE=0",
            "ASCEND_LAUNCH_BLOCKING=0",
            "P10_DIRECT_DISPATCH_CONFIG",
            "P10_PRODUCT_BRIDGE_PYTHON",
            "P10_KVAF_BUILD",
        ):
            self.assertIn(token, self.runner)
        self.assertNotIn("P8_", self.runner)
        self.assertNotIn("P9_", self.runner)
        self.assertNotIn("TASK_QUEUE_ENABLE=1", self.runner)
        self.assertNotIn("TQ1", self.runner)
        probe = '"${python_bin}" - <<\'PY\''
        self.assertLess(self.runner.index("TASK_QUEUE_ENABLE=0"), self.runner.index(probe))
        self.assertLess(
            self.runner.index("ASCEND_LAUNCH_BLOCKING=0"), self.runner.index(probe)
        )

    def test_runner_builds_the_complete_sha_keyed_bridge_package(self) -> None:
        """A native .so alone would create a namespace package at runtime.

        ``build_ext`` does not copy the pure-Python graph adapter.  Require a
        separate ``build_py`` and fail closed unless the adapter, package
        marker, and exactly one native extension all share the product root.
        """

        for token in (
            'p10_forward_bridge_setup.py build_py',
            'bridge_package="${bridge_python}/nanovllm_p10_forward_bridge"',
            'bridge_init="${bridge_package}/__init__.py"',
            'bridge_graph="${bridge_package}/p10_forward_graph.py"',
            'bridge_python_package_incomplete',
            'bridge_artifact_count_invalid',
            'P10_BRIDGE_PROVENANCE package=%s init=%s graph=%s bridge=%s',
        ):
            self.assertIn(token, self.runner)

    def test_runner_ignores_stale_interactive_build_root_overrides(self) -> None:
        """Artifact roots are derived from the exact source pair, not a shell.

        A prior P10 command can leave an old package/build path exported.  It
        must not redirect the next source pair into an old product directory.
        """

        for token in (
            'package_parent="${ops_root}/build-p10-product"',
            'package_root="${package_parent}/p10_v${P10_VLLM_EXPECTED_SHA:0:12}_k${P10_KVAF_EXPECTED_SHA:0:12}"',
            'P10_ASSEMBLY_OVERRIDE_IGNORED name=P10_PACKAGE_PARENT',
            'P10_ASSEMBLY_OVERRIDE_IGNORED name=P10_PRODUCT_PACKAGE_ROOT',
            'kvaf_build="${kvaf_root}/build-p10-v${P10_VLLM_EXPECTED_SHA:0:12}-k${P10_KVAF_EXPECTED_SHA:0:12}"',
            'P10_ASSEMBLY_OVERRIDE_IGNORED name=P10_KVAF_BUILD',
        ):
            self.assertIn(token, self.runner)
        self.assertNotIn('package_root="${P10_PRODUCT_PACKAGE_ROOT:-', self.runner)
        self.assertNotIn('kvaf_build="${P10_KVAF_BUILD:-', self.runner)

    def test_fixture_is_real_w4a4c8_direct_hccp_and_qsfa_closure(self) -> None:
        for token in (
            "torch.float8_e4m3fn",
            "dtype=torch.bfloat16",
            "torch_npu.npu_quant_lightning_indexer",
            "bind_layer_resident_batch_get_control",
            "configure_layer_resident_batch_get_queue_templates",
            "launch_p10_forward(inputs)",
            "launch.bind_qsfa()",
            "strict W4A4C8 canonical Top-K mismatch",
            "resident_ready=1 payload=1 queue_leases_zero=1",
            "resident_row_to_qsfa_slot is not a nonidentity permutation",
            "torch.unique(translation)",
            "P10_RUNTIME_PASS",
        ):
            self.assertIn(token, self.fixture)
        self.assertNotIn("runtime.batch_get(", self.fixture)
        self.assertNotIn("P8_", self.fixture)
        self.assertNotIn("P9_", self.fixture)

    def test_fixture_rejects_a_namespace_bridge_before_operator_execution(self) -> None:
        self.assertIn('getattr(module, "__file__", None)', self.fixture)
        self.assertIn('"product_bridge_package_file_missing"', self.fixture)


if __name__ == "__main__":
    unittest.main()
