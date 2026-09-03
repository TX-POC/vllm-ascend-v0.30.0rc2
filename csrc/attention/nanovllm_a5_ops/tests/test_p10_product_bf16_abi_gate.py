"""Host-only contract tests for P10's selected-product C8 ABI gate."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


OPS_ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = OPS_ROOT / "a5_dsa_c8_exact_speculative_resident"
sys.path.insert(0, str(OPS_ROOT))

from a5_dsa_c8_exact_speculative_resident.product_bf16_abi_gate import (  # noqa: E402
    GateFailure,
    REFERENCE_BRIDGE_MARKER,
    REFERENCE_OPAPI_MARKER,
    REFERENCE_OPS,
    REFERENCE_BRIDGE_SOURCE,
    verify_product_c8_frontend_source,
    verify_product_bf16_source,
    verify_reference_artifacts,
)


REPOSITORY_ROOT = OPS_ROOT.parents[2]
PRODUCT_PROBE = TARGET_ROOT / "tools" / "probe_product_bf16_abi.py"
BRIDGE_WRITER = TARGET_ROOT / "tools" / "write_reference_bridge_provenance.py"
RUNTIME_FIXTURE = TARGET_ROOT / "tools" / "run_product_bf16_reference_equivalence.py"
RUNNER = OPS_ROOT / "tests" / "run_p10_product_c8_abi_gate_npu.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


class P10ProductC8AbiGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = REPOSITORY_ROOT.resolve()
        cls.head = _git_head(cls.root)

    def _write_artifacts(
        self, root: Path, *, bridge_marker: bool
    ) -> tuple[Path, Path, Path]:
        vendor = root / "vendors" / "official_c8_reference_transformer"
        opapi = vendor / "op_api" / "lib" / "libcust_opapi.so"
        opapi.parent.mkdir(parents=True)
        opapi.write_bytes(b"opapi")
        (vendor / REFERENCE_OPAPI_MARKER).write_text(
            "\n".join(
                (
                    "format=1",
                    "kind=source_matched_official_c8",
                    f"source_root={self.root}",
                    f"source_head={self.head}",
                    f"operators={REFERENCE_OPS}",
                    "",
                )
            )
        )
        bridge_root = root / "bridge"
        module = bridge_root / "nanovllm_official_c8_reference" / "_C.test.so"
        module.parent.mkdir(parents=True)
        module.write_bytes(b"bridge")
        if bridge_marker:
            bridge_source = self.root / REFERENCE_BRIDGE_SOURCE
            (bridge_root / REFERENCE_BRIDGE_MARKER).write_text(
                "\n".join(
                    (
                        "format=1",
                        "kind=source_matched_official_c8_bridge",
                        f"source_root={self.root}",
                        f"source_head={self.head}",
                        f"opapi={opapi.resolve()}",
                        f"opapi_sha256={_sha256(opapi)}",
                        f"bridge_source={bridge_source.resolve()}",
                        f"bridge_source_sha256={_sha256(bridge_source)}",
                        f"bridge_module={module.resolve()}",
                        f"bridge_module_sha256={_sha256(module)}",
                        "",
                    )
                )
            )
        return opapi, bridge_root, module

    def test_product_source_proves_real_bf16_primary_callable(self):
        product = verify_product_bf16_source(self.root)
        self.assertEqual(product.product_callable, "torch_npu.npu_quant_lightning_indexer")
        self.assertEqual(product.product_weight_dtype, "bfloat16")
        self.assertEqual(product.reference_weight_dtype, "float32")
        self.assertEqual(
            product.weight_conversion, "lossless_bfloat16_to_float32_widen"
        )
        self.assertEqual(product.product_source_sha, self.head)

    def test_product_source_rejects_missing_bf16_anchor_before_any_npu_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "vllm_ascend" / "dsa_offload" / "a5_extension.py"
            source.parent.mkdir(parents=True)
            source.write_text("def _check_c8_lidu_inputs():\n    pass\n")
            with self.assertRaisesRegex(
                GateFailure, "product_bf16_abi_source_anchor_missing"
            ):
                verify_product_bf16_source(root)

    def test_frontend_must_be_explicit_and_each_path_has_its_own_abi(self):
        bf16 = verify_product_c8_frontend_source(self.root, "dsa_offload_bf16")
        fp32 = verify_product_c8_frontend_source(self.root, "main_dsa_v1_fp32")
        self.assertEqual(bf16.product_weight_dtype, "bfloat16")
        self.assertEqual(
            bf16.weight_conversion, "lossless_bfloat16_to_float32_widen"
        )
        self.assertEqual(fp32.product_weight_dtype, "float32")
        self.assertEqual(fp32.weight_conversion, "identity_float32")
        self.assertEqual(
            fp32.product_metadata_callable,
            "torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata",
        )
        with self.assertRaisesRegex(
            GateFailure, "product_frontend_unsupported_or_missing"
        ):
            verify_product_c8_frontend_source(self.root, "")

    def test_reference_artifacts_require_both_opp_and_bridge_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            opapi, bridge_root, module = self._write_artifacts(
                Path(temporary), bridge_marker=False
            )
            with self.assertRaisesRegex(
                GateFailure, "reference_bridge_provenance_missing"
            ):
                verify_reference_artifacts(
                    reference_vllm_root=self.root,
                    reference_opapi=opapi,
                    reference_bridge_root=bridge_root,
                    reference_bridge_module=module,
                    expected_source_sha=self.head,
                )

    def test_reference_artifacts_accept_exact_source_opp_and_bridge_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            opapi, bridge_root, module = self._write_artifacts(
                Path(temporary), bridge_marker=True
            )
            artifacts = verify_reference_artifacts(
                reference_vllm_root=self.root,
                reference_opapi=opapi,
                reference_bridge_root=bridge_root,
                reference_bridge_module=module,
                expected_source_sha=self.head,
            )
        self.assertEqual(artifacts.reference_source_sha, self.head)
        self.assertEqual(artifacts.opapi, opapi.resolve())
        self.assertEqual(artifacts.bridge_root, bridge_root.resolve())

    def test_bridge_marker_rejects_changed_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            opapi, bridge_root, module = self._write_artifacts(
                Path(temporary), bridge_marker=True
            )
            module.write_bytes(b"changed-bridge")
            with self.assertRaisesRegex(GateFailure, "marker_value_mismatch"):
                verify_reference_artifacts(
                    reference_vllm_root=self.root,
                    reference_opapi=opapi,
                    reference_bridge_root=bridge_root,
                    reference_bridge_module=module,
                    expected_source_sha=self.head,
                )

    def test_host_safe_probe_requires_explicit_complete_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            opapi, bridge_root, module = self._write_artifacts(
                Path(temporary), bridge_marker=True
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(PRODUCT_PROBE),
                    "--product-vllm-root",
                    str(self.root),
                    "--frontend",
                    "dsa_offload_bf16",
                    "--reference-vllm-root",
                    str(self.root),
                    "--reference-opapi",
                    str(opapi),
                    "--reference-bridge-root",
                    str(bridge_root),
                    "--reference-bridge-module",
                    str(module),
                    "--reference-source-sha",
                    self.head,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("P10_PRODUCT_C8_ABI_GATE status=pass", result.stdout)
        self.assertIn("frontend=dsa_offload_bf16", result.stdout)
        self.assertIn("product_weights=bfloat16", result.stdout)
        self.assertIn("reference_weights=float32", result.stdout)
        self.assertIn("weight_conversion=lossless_bfloat16_to_float32_widen", result.stdout)

    def test_bridge_writer_is_host_safe_and_write_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            opapi, bridge_root, module = self._write_artifacts(
                Path(temporary), bridge_marker=False
            )
            environment = dict(os.environ)
            environment.update(
                {
                    "P10_REFERENCE_VLLM_ROOT": str(self.root),
                    "P10_REFERENCE_OPAPI": str(opapi),
                    "P10_REFERENCE_BRIDGE_ROOT": str(bridge_root),
                    "P10_REFERENCE_BRIDGE_MODULE": str(module),
                }
            )
            first = subprocess.run(
                [sys.executable, str(BRIDGE_WRITER)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            second = subprocess.run(
                [sys.executable, str(BRIDGE_WRITER)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertIn("P10_REFERENCE_BRIDGE_PROVENANCE status=pass", first.stdout)
        self.assertEqual(second.returncode, 2)
        self.assertIn("bridge_provenance_already_exists", second.stdout)

    def test_runtime_fixture_is_strict_and_never_falls_back_or_builds(self):
        source = RUNTIME_FIXTURE.read_text()
        for required in (
            'getattr(torch_npu, "npu_quant_lightning_indexer", None)',
            "product_primary_callable_unavailable",
            'if product.frontend == "dsa_offload_bf16":',
            'if product.frontend == "main_dsa_v1_fp32":',
            "product_weights.to(torch.float32).contiguous()",
            "reference_weight_widen_not_lossless",
            "reference_bridge_extension_path_mismatch",
            "P10_PRODUCT_C8_REFERENCE_EQ status=pass",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "npu_lightning_indexer_quant",
            "subprocess.run",
            "build_ext",
            "msopgen",
            "P8",
            "P9",
        ):
            self.assertNotIn(forbidden, source)

    def test_runner_is_explicit_fail_closed_and_build_free(self):
        source = RUNNER.read_text()
        for required in (
            "P10_PRODUCT_C8_FRONTEND",
            "P10_PRODUCT_VLLM_ROOT",
            "P10_PRODUCT_VLLM_EXPECTED_SHA",
            "P10_REFERENCE_VLLM_ROOT",
            "P10_REFERENCE_SOURCE_SHA",
            "P10_REFERENCE_OPAPI",
            "P10_REFERENCE_BRIDGE_ROOT",
            "P10_REFERENCE_BRIDGE_MODULE",
            "P10_PHYSICAL_NPU_ID",
            "ASCEND_RT_VISIBLE_DEVICES",
            "P10_ABI_GATE_CONFIG",
            "P10_ABI_GATE_STEP label=source_and_artifact_provenance",
            "P10_ABI_GATE_STEP label=product_vs_reference",
            "P10_ABI_GATE_PASS",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "build_ext",
            "msopgen",
            "cmake --build",
            "pip install",
            "source_overlay",
            "P8",
            "P9",
        ):
            self.assertNotIn(forbidden, source)
        result = subprocess.run(
            ["bash", "-n", str(RUNNER)], check=False, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
