#!/usr/bin/env python3
"""Cheap source gate for P9's fresh-artifact NPU acceptance runner."""

from __future__ import annotations

from pathlib import Path
import unittest


OPS_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = OPS_ROOT / "tests"
RUNNER = (TEST_ROOT / "run_p9_exact_speculative_resident_npu.sh").read_text()
RUNTIME = (TEST_ROOT / "test_p9_exact_speculative_resident_runtime.py").read_text()


class P9ExactSpeculativeResidentRunnerContractTest(unittest.TestCase):
    def test_runtime_has_both_c8_math_and_real_terminal_gates(self) -> None:
        for token in (
            "P9_RUNTIME_C8_GATE runtime_enable=0 exact_sparse=1",
            "P9_RUNTIME_REAL_BIND",
            "P9_RUNTIME_ACTIVE",
            "P9_RUNTIME_PASS",
            "runtime_enable=1 changed exact C8 sparse_indices",
            "canonical resident payload mismatch",
            "terminal reported failure",
            "queue lease",
        ):
            self.assertIn(token, RUNTIME)
        self.assertNotIn("p8_real_direct_hccp_fixture", RUNTIME)
        self.assertNotIn("streaming_quant_li_get_target_adapter", RUNTIME)

    def test_runtime_has_a_sparse_only_abi_control_gate(self) -> None:
        """The same-builder control must not be mistaken for resident Get."""
        for token in (
            'choices=("disabled-math", "full")',
            "P9_RUNTIME_DISABLED_MATH",
            "P9_RUNTIME_PASS case=disabled-math runtime_enable_0=1",
            "while leaving P9-only auxiliary outputs unwritten",
        ):
            self.assertIn(token, RUNTIME)

    def test_runtime_uses_explicit_slots_not_an_affine_execution_row_layout(self) -> None:
        self.assertIn("canonical_resident_slots", RUNTIME)
        self.assertIn("torch.empty((rows, TOPK), dtype=torch.int32)", RUNTIME)
        self.assertIn("resident_block_table", RUNTIME)
        self.assertIn("Physical rows are execution-row-owned", RUNTIME)
        self.assertIn("active_destinations_unique=1", RUNTIME)
        self.assertIn("P9_RUNTIME_PADDING_GATE", RUNTIME)
        self.assertIn("req_pool_entries[row]", RUNTIME)
        self.assertIn("torch.arange(rows - 1, -1, -1", RUNTIME)
        self.assertNotIn("canonical_resident_layout", RUNTIME)
        self.assertNotIn("torch.empty((rows, TOPK, 2)", RUNTIME)

    def test_runner_has_one_fresh_provenance_closed_path(self) -> None:
        for token in (
            "mktemp -d",
            "P9_KVAF_BUILD",
            "P9_REFERENCE_OPP",
            "P9_TARGET_PACKAGE_PARENT",
            "P9_TARGET_BRIDGE",
            "P9_OPP_PROVENANCE",
            "source_changed_during_build",
            "P9_RUNNER_PASS physical=7 logical=0",
        ):
            self.assertIn(token, RUNNER)
        self.assertIn("P9_CANN_3RD_LIB_PATH", RUNNER)
        self.assertIn("cann_third_party_cache_path_missing", RUNNER)
        self.assertIn("p9_abi=official9_plus14_inputs_6_outputs", RUNNER)
        self.assertIn("p9_legacy_compile_disabled_provenance", RUNNER)
        self.assertNotIn("export SOC_VERSION=", RUNNER)

    def test_fresh_kvaf_build_path_is_exported_before_the_clean_import_probe(self) -> None:
        """The import probe runs in a child process and needs this exact path."""
        artifact_check = RUNNER.index('runner_blocked "fresh_kvaf_artifacts_missing"')
        export_path = RUNNER.index("export P9_KVAF_BUILD")
        import_probe = RUNNER.index("run_logged kvaf_import")
        self.assertLess(artifact_check, export_path)
        self.assertLess(export_path, import_probe)

    def test_runner_selects_the_installed_p9_receipt_not_the_source_copy(self) -> None:
        """A fresh P9 package intentionally has source and installed receipts.

        The installed receipt is the one adjacent to the OpAPI actually loaded
        by the bridge.  It must match the retained source receipt exactly; a
        generic recursive filename search would make this legitimate pair look
        ambiguous and, worse, would not bind provenance to the loaded library.
        """
        self.assertNotIn(
            'find "$P9_TARGET_PACKAGE_PARENT" -type f -name P9_PROVENANCE',
            RUNNER,
        )
        for token in (
            "P9_PACKAGE_TAG=",
            "-mindepth 1 -maxdepth 1 -type d",
            'runner_blocked "p9_package_root_ambiguous"',
            "P9_PACKAGE_PROVENANCE",
            "P9_VENDOR_PROVENANCE",
            'runner_blocked "p9_provenance_missing"',
            'runner_blocked "p9_provenance_copy_mismatch"',
            'cmp -s "$P9_PACKAGE_PROVENANCE" "$P9_VENDOR_PROVENANCE"',
            "P9_VENDOR_ROOT/op_api/lib/libcust_opapi.so",
        ):
            self.assertIn(token, RUNNER)

    def test_runner_does_not_substitute_a_historical_transport_route(self) -> None:
        self.assertNotIn("p8_real_direct_hccp_fixture", RUNNER)
        self.assertNotIn("a5_p8_forward_owned_executor", RUNNER)
        self.assertNotIn("a5_streaming_quant_li_get", RUNNER)
        self.assertIn("P9_REQUIRE_REAL_HCCP=1", RUNNER)
        self.assertIn("TASK_QUEUE_ENABLE=0", RUNNER)
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES=7", RUNNER)


if __name__ == "__main__":
    unittest.main(verbosity=2)
