"""Static safety gate for P10's one-vendor product assembly.

The assembly runner validates one clean SHA pair, stages ForwardPrepare, Stage
A, and Stage B independently, and emits one combined product OPP. It reuses
an artifact only after its manifest and the single OpAPI hash verify. This
prevents a stale OPP or a mixed multi-vendor symbol resolution from being
treated as an incremental build result.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
import os
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    OPS_ROOT
    / "a5_dsa_c8_exact_speculative_resident"
    / "build_p10_target_assembly.sh"
)
PRODUCT_SCRIPT = (
    OPS_ROOT
    / "a5_dsa_c8_exact_speculative_resident"
    / "build_p10_product_opp.sh"
)


class P10BuildAssemblyContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text()
        cls.product_source = PRODUCT_SCRIPT.read_text()

    def test_runner_is_syntactically_valid(self) -> None:
        for script in (SCRIPT, PRODUCT_SCRIPT):
            result = subprocess.run(
                ["bash", "-n", str(script)],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                result.returncode, 0, result.stdout + result.stderr
            )

    def test_exact_clean_source_pair_is_a_hard_precondition(self) -> None:
        for token in (
            "P10_BUILD_EXPECTED_VLLM_SHA",
            "P10_BUILD_EXPECTED_KVAF_SHA",
            "require_clean_exact_git_root",
            "tracked_changes_present",
            "legacy_SOC_VERSION_set",
        ):
            self.assertIn(token, self.source)

    def test_staging_is_source_only_and_has_no_legacy_overlay_route(self) -> None:
        for token in (
            "copy_source_tree",
            "stage_a_w4a4c8_score_receipt",
            "stage_b_score_receipt_forward_resident",
            "p10_forward_bridge",
            "build_p10_product_opp.sh",
            "one combined OPP API library",
            "legacy_overlay_reuse=forbidden",
            "a5_c8_exact_speculative_resident_get",
            "a5_streaming_quant_li_get",
            "a5_score_receipt_speculative_executor",
        ):
            self.assertIn(token, self.source)
        self.assertNotIn("build_nanovllm_ops.sh", self.source)
        self.assertNotIn("run_p8_", self.source)
        self.assertNotIn("run_p9_", self.source)

    def test_reuse_requires_manifest_and_exactly_one_hashed_combined_opapi(self) -> None:
        for token in (
            "P10_ASSEMBLY_REUSE_PASS",
            "manifest_has_exact_input",
            "assembly_state=complete",
            "combined_product_opapi_missing",
            "product_opapi_sha256",
            "A5P10ForwardPrepare,A5P10W4A4C8ScoreReceipt,A5P10ScoreReceiptForwardResident",
            "artifact_reuse=manifest_and_opapi_hash_only",
        ):
            self.assertIn(token, self.source)
        self.assertNotIn("fresh_package_root_exists", self.source)

    def test_combined_msopgen_project_keeps_its_own_cmake_and_uses_stage_b_kernel(self) -> None:
        # Repository-level CMake fragments cannot be overlaid on a fresh CANN
        # scaffold, and Stage B's deliberately suffixed source must replace
        # the generated canonical dispatch filename.
        for token in (
            "copy_unique_tree_without_repository_cmake",
            "repository_host_cmake_leaked_into_msopgen_project",
            "stage_b_generated_kernel",
            "stage_b_source_kernel",
            "stage_b_source_kernel_missing_after_copy",
        ):
            self.assertIn(token, self.product_source)

    def test_stage_a_cannot_reinitialize_receipts_after_concurrent_launch(self) -> None:
        for token in (
            "require_forward_preinit_boundary",
            "stage_a_illegal_receipt_init_hook_present",
            "stage_b_forward_control_input_missing",
            "InitializeReceiptStateShard",
        ):
            self.assertIn(token, self.source)

    @staticmethod
    def _commit_repo(root: Path) -> str:
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=P10 Test",
                "-c",
                "user.email=p10@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()

    def test_stage_mode_is_digest_keyed_and_reuses_only_the_complete_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vllm = root / "vllm"
            kvaf = root / "kvaf"
            stage_a = (
                vllm
                / "csrc/attention/nanovllm_a5_ops"
                / "a5_dsa_c8_exact_speculative_resident"
                / "stage_a_w4a4c8_score_receipt"
            )
            stage_b = (
                vllm
                / "csrc/attention/nanovllm_a5_ops"
                / "a5_dsa_c8_exact_speculative_resident"
                / "stage_b_score_receipt_forward_resident"
            )
            forward = (
                vllm
                / "csrc/attention/nanovllm_a5_ops"
                / "a5_dsa_c8_exact_speculative_resident"
                / "p10_forward_bridge"
            )
            for directory in (
                stage_a / "op_host",
                stage_a / "op_kernel/arch35",
                stage_b / "op_host",
                stage_b / "op_kernel",
                forward / "op_host",
                forward / "op_kernel",
                forward / "torch_extension",
                kvaf,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            (stage_a / "VENDORED.md").write_text("source=fixture\n")
            (stage_a / "op_host/a.cpp").write_text("// host\n")
            (stage_a / "op_kernel/a5_p10_w4a4c8_score_receipt_qli.h").write_text(
                "// no receipt init hook\n"
            )
            (stage_a / "op_kernel/arch35/quant_lightning_indexer_service_vector.h").write_text(
                "// vector closure\n"
            )
            (stage_a / "op_kernel/a5_p10_w4a4c8_score_receipt_abi.h").write_text(
                "// forward_control graph alias\n"
            )
            (stage_b / "VENDORED.md").write_text("source=fixture\n")
            (stage_b / "op_host/b.cpp").write_text('Input("forward_control")\n')
            (stage_b / "op_kernel/b.cpp").write_text("// kernel\n")
            (stage_a / "ops.json").write_text("[]\n")
            (stage_b / "ops.json").write_text("[]\n")
            (forward / "ops.json").write_text("[]\n")
            (forward / "P10_FORWARD_BRIDGE_CONTRACT.json").write_text(
                '{"must_run_before": ["stage_a", "stage_b"]}\n'
            )
            (forward / "build_p10_forward_prepare_opp.sh").write_text(
                "#!/usr/bin/env bash\n"
            )
            (forward / "op_host/a5_p10_forward_prepare_def.cpp").write_text(
                "OP_ADD(A5P10ForwardPrepare)\n"
            )
            (forward / "op_kernel/a5_p10_forward_prepare.cpp").write_text(
                "// kernel\n"
            )
            (forward / "torch_extension/p10_forward_bridge_setup.py").write_text(
                "# fixture\n"
            )
            product_recipe = (
                vllm
                / "csrc/attention/nanovllm_a5_ops"
                / "a5_dsa_c8_exact_speculative_resident"
                / "build_p10_product_opp.sh"
            )
            product_recipe.write_text("#!/usr/bin/env bash\n")
            vllm_sha = self._commit_repo(vllm)
            (kvaf / "README").write_text("fixture\n")
            kvaf_sha = self._commit_repo(kvaf)
            package_parent = root / "packages"
            environment = dict(os.environ, **{
                "P10_BUILD_VLLM_ROOT": str(vllm),
                "P10_BUILD_KVAF_ROOT": str(kvaf),
                "P10_BUILD_EXPECTED_VLLM_SHA": vllm_sha,
                "P10_BUILD_EXPECTED_KVAF_SHA": kvaf_sha,
                "P10_BUILD_MODE": "stage",
                "P10_BUILD_PACKAGE_PARENT": str(package_parent),
                "SOC_VERSION": "",
            })
            first = subprocess.run(
                ["bash", str(SCRIPT)],
                check=False,
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertIn("P10_ASSEMBLY_STAGE_ONLY_PASS", first.stdout)
            second = subprocess.run(
                ["bash", str(SCRIPT)],
                check=False,
                text=True,
                capture_output=True,
                env=environment,
            )
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("P10_ASSEMBLY_REUSE_PASS", second.stdout)


if __name__ == "__main__":
    unittest.main()
