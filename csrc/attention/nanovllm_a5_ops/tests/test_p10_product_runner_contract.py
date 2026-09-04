"""Static contract for the sole P10 product build-and-proof runner.

This catches orchestration drift locally, before an A5 host spends time
building a wrong source closure. It checks the concrete P10 path rather than
trying to simulate CANN or Direct-HCCP on a development machine.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[1]
RUNNER = OPS_ROOT / "tests" / "run_p10_w4a4c8_forward_resident_npu.sh"
FIXTURE = OPS_ROOT / "tests" / "test_p10_w4a4c8_forward_resident_runtime.py"
BOOTSTRAP = OPS_ROOT / "tools" / "bootstrap_cann_third_party_cache.sh"


class P10ProductRunnerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = RUNNER.read_text()
        cls.fixture = FIXTURE.read_text()
        cls.bootstrap = BOOTSTRAP.read_text()

    def test_runner_is_valid_and_binds_its_own_exact_source_tree(self) -> None:
        for script in (RUNNER, BOOTSTRAP):
            result = subprocess.run(
                ["bash", "-n", str(script)], capture_output=True, text=True, check=False
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

    def test_runner_bootstraps_a_complete_external_cann_cache_before_build(self) -> None:
        """A clean Git clone must not masquerade as a dependency cache."""

        for token in (
            'cann_cache_default="$(dirname "${vllm_root}")/.cann-third-party-$(basename "${ascend_home}")"',
            "cann_third_party_cache_must_be_external",
            "bootstrap_cann_third_party_cache.sh",
            "P10_CANN_3RD_BOOTSTRAP:-1",
            "cache_complete || fail",
            "P10_CANN_CACHE path=%s source_tree=0 complete=1",
        ):
            self.assertIn(token, self.runner)
        self.assertNotIn(
            'cann_cache="${P10_CANN_3RD_LIB_PATH:-${vllm_root}/csrc/third_party}"',
            self.runner,
        )

        for token in (
            "P10_CANN_3RD_MAKESELF_URL",
            "P10_CANN_3RD_JSON_URL",
            "P10_CANN_3RD_ABSEIL_URL",
            "P10_CANN_3RD_PROTOBUF_URL",
            "P10_CANN_3RD_PROVENANCE",
            "P10_CANN_CACHE_BOOTSTRAP_PASS",
            "download_failed:label=",
            "--netrc-optional",
            "has_makeself || makeself_missing=1",
            "has_json || json_missing=1",
            "has_abseil || abseil_missing=1",
            "has_protobuf || protobuf_missing=1",
            'if [[ "${makeself_missing}" -eq 1 ]]; then',
            'if [[ "${json_missing}" -eq 1 ]]; then',
            'if [[ "${abseil_missing}" -eq 1 ]]; then',
            'if [[ "${protobuf_missing}" -eq 1 ]]; then',
        ):
            self.assertIn(token, self.bootstrap)

    def test_bootstrap_reuses_a_complete_cache_without_touching_the_network(self) -> None:
        """A warm cache must return before curl/download work is considered."""

        with tempfile.TemporaryDirectory(prefix="p10-cann-cache-") as raw_cache:
            cache = Path(raw_cache)
            (cache / "makeself").mkdir()
            (cache / "json" / "include" / "nlohmann").mkdir(parents=True)
            (cache / "pkg").mkdir()
            for relative in (
                "makeself/makeself.sh",
                "makeself/makeself-header.sh",
                "json/include/nlohmann/json.hpp",
                "pkg/abseil-cpp-20230802.1.tar.gz",
                "pkg/protobuf-25.1.tar.gz",
            ):
                (cache / relative).touch()
            result = subprocess.run(
                ["bash", str(BOOTSTRAP), str(cache)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("P10_CANN_CACHE_BOOTSTRAP_PASS", result.stdout)
            self.assertIn("reused=1", result.stdout)
            self.assertNotIn("P10_CANN_CACHE_DOWNLOAD", result.stdout)

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

    def test_runner_forwards_only_the_four_explicit_runtime_cases(self) -> None:
        """Benchmarking must not select a second product artifact or route."""

        for token in (
            'runtime_case="${P10_RUNTIME_CASE:-acceptance}"',
            "acceptance|benchmark|graph_benchmark|graph_compatible_benchmark",
            "invalid_runtime_case:${runtime_case}",
            "runtime_case=%s",
            'P10_RUNTIME_CASE="${runtime_case}" "${python_bin}" "${runtime_fixture}"',
            "label=forward_resident_runtime case=%s status=0",
        ):
            self.assertIn(token, self.runner)

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
            'P10_ASSEMBLY_INVALID_PACKAGE_POLICY=quarantine',
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
            "resident_ready=1 payload=1 ",
            "queue_leases_zero=1 unique_slots=",
            "resident_row_to_qsfa_slot is not a nonidentity permutation",
            "torch.unique(translation)",
            "P10_RUNTIME_PASS",
            "P10_BOOTSTRAP_ROUNDTRIP_PASS ",
            "source 380 is the first",
            "(0, 127, 128, 240, 255, 256, 380, SOURCE_TOKENS - 1)",
            "P10_BENCH_DESTINATION_RESET ",
            "authority=forward_prepare payload_write=none",
        ):
            self.assertIn(token, self.fixture)
        # The sole ordinary BatchGet is a cold source-visibility gate before
        # Layer-resident binding.  It must never become a timed fallback.
        self.assertEqual(self.fixture.count("self.runtime.batch_get("), 1)
        self.assertLess(
            self.fixture.rindex("self._prove_bootstrap_direct_hccp_payload("),
            self.fixture.index("self.runtime.bind_layer_resident_batch_get_control("),
        )
        reset = self.fixture.split("def _reset_benchmark_destination(", 1)[1].split(
            "def _run_timed_forward(", 1
        )[0]
        self.assertIn("authority=forward_prepare payload_write=none", reset)
        self.assertNotIn("resident_payload.narrow", reset)
        self.assertNotIn(".zero_()", reset)
        self.assertNotIn("P8_", self.fixture)
        self.assertNotIn("P9_", self.fixture)

    def test_fixture_rejects_a_namespace_bridge_before_operator_execution(self) -> None:
        self.assertIn('getattr(module, "__file__", None)', self.fixture)
        self.assertIn('"product_bridge_package_file_missing"', self.fixture)

    def test_benchmark_keeps_one_product_route_and_a_qsfa_terminal_boundary(self) -> None:
        """The latency comparison must not quietly degrade into host Get timing."""

        for token in (
            'P10_RUNTIME_CASE", "acceptance")',
            'if runtime_case == "benchmark"',
            'execution_mode=mode',
            "launch.bind_qsfa(producer)",
            "end.synchronize()",
            "require_preseal_overlap=(mode == \"overlap\")",
            "P10_BENCH_DESTINATION_RESET ",
            "authority=forward_prepare payload_write=none",
            "P10_PERF_EQ ",
            "p10_w4a4c8_forward_resident_serial",
            "p10_w4a4c8_forward_resident_overlap",
        ):
            self.assertIn(token, self.fixture)
        self.assertEqual(self.fixture.count("self.runtime.batch_get("), 1)

    def test_graph_benchmark_replays_the_same_product_route_against_eager_serial(self) -> None:
        """Raw ACLGraph timing is replay-only and cannot hide a second ABI."""

        for token in (
            'if runtime_case == "graph_benchmark"',
            "capture_p10_forward_graph",
            "P10_GRAPH_PERF_EQ ",
            "P10_GRAPH_PERF_RESULT ",
            "p10_w4a4c8_forward_resident_eager_serial",
            "p10_w4a4c8_forward_resident_graph_serial",
            "p10_w4a4c8_forward_resident_graph_overlap",
            "aclgraph_segments=2",
            "stage_b_direct_relaunch=1",
            "direct_hccp_captured=0",
            "backend=aclmdlRI_raw_direct_aclnn",
            "workspace_buffers_serial=",
            "workspace_buffers_overlap=",
        ):
            self.assertIn(token, self.fixture)
        self.assertIn("graph_benchmark", self.runner)
        self.assertEqual(self.fixture.count("self.runtime.batch_get("), 1)

    def test_graph_compatible_b16_profile_is_the_only_large_direct_hccp_measurement(self) -> None:
        """The requested capacity case must not be mislabeled raw ACLGraph."""

        for token in (
            'elif P10_BENCH_PROFILE == "b16_s128k_miss5"',
            "BATCH = 16",
            "SOURCE_TOKENS = 131072",
            "KV_MISS_DENOMINATOR = 20",
            "_install_final_hit_payloads",
            "Object ownership is pool-major",
            "Stage B indexes it",
            "``pool * sourceCapacity + source``",
            "def _run_graph_compatible_benchmark",
            "P10_GRAPH_COMPAT_CONFIG ",
            "P10_GRAPH_COMPAT_EQ ",
            "P10_GRAPH_COMPAT_RESULT ",
            "same_source_miss_mask=1 static_aliases=1 raw_aclgraph_capture=0",
            "raw_aclgraph_benchmark_not_valid_for_direct_hccp",
            'if runtime_case == "graph_compatible_benchmark"',
        ):
            self.assertIn(token, self.fixture)
        self.assertIn("graph_compatible_benchmark", self.runner)
        self.assertEqual(self.fixture.count("self.runtime.batch_get("), 1)

    def test_b16_bootstrap_bounds_only_the_pre_timing_batch_put_wave(self) -> None:
        """Large cold loading must not consume the eager AIV CQ budget at once.

        This guard deliberately concerns fixture setup rather than the P10
        forward.  It preserves one source allocation and the original global
        object offsets, while every timed serial/overlap invocation remains
        device-side Stage B Direct-HCCP.
        """

        for token in (
            '"P10_BOOTSTRAP_PUT_ROWS", MAX_REQUESTS_PER_WAVE',
            "invalid_p10_bootstrap_put_rows:",
            "P10_BOOTSTRAP_PUT_CONFIG ",
            "P10_BOOTSTRAP_PUT_PASS ",
            "P10_BOOTSTRAP_PUT_FAIL ",
            "P10_BOOTSTRAP_SOURCE_READY ",
            "P10_BOOTSTRAP_ROUNDTRIP_PASS ",
            "P10_BOOTSTRAP_ROUNDTRIP_FAIL ",
            "keys[start:stop]",
            "offsets[start:stop]",
            "lengths[start:stop]",
            "timed_interval=excluded",
        ):
            self.assertIn(token, self.fixture)
        self.assertLess(
            self.fixture.index("P10_BOOTSTRAP_PUT_PASS "),
            self.fixture.index("self.table = self.runtime.prepare_access_table(keys)"),
        )
        self.assertLess(
            self.fixture.rindex("self._prove_bootstrap_direct_hccp_payload("),
            self.fixture.index("self.runtime.bind_layer_resident_batch_get_control("),
        )

    def test_b16_residency_mask_is_cpu_constructed_before_the_npu_upload(self) -> None:
        """Fixture-only mask construction must not invoke a large NPU int kernel.

        Its bit pattern remains the same deterministic row/source formula;
        only the placement of untimed metadata construction changes.
        """

        for token in (
            "miss_mask_cpu = ",
            "object_snapshot_cpu.masked_fill_(miss_mask_cpu, -1)",
            "miss_mask = miss_mask_cpu.to(device=device).contiguous()",
            "object_snapshot = object_snapshot_cpu.to(device=device).contiguous()",
            "P10_RESIDENCY_MASK_READY ",
            "construction=cpu upload=npu",
        ):
            self.assertIn(token, self.fixture)
        self.assertNotIn(
            "source_ids = torch.arange(SOURCE_TOKENS, dtype=torch.int32, device=device)",
            self.fixture,
        )

    def test_b16_fixture_metadata_and_final_hit_setup_are_cpu_derived(self) -> None:
        """Keep untimed bookkeeping off the fragile ordinary AIVEC path.

        The NPU aliases are still used by the product operator.  This only
        constrains their deterministic fixture construction and preserves the
        exact source/slot arithmetic used by the old setup path.
        """

        for token in (
            "resident_to_qsfa_cpu = (",
            "resident_to_qsfa = resident_to_qsfa_cpu.to(device=device).contiguous()",
            "P10_FIXTURE_STATIC_METADATA_READY ",
            '"object_snapshot_cpu": object_snapshot_cpu',
            '"miss_mask_cpu": miss_mask_cpu',
            "ranked_sources_cpu = (",
            "final_miss_cpu = miss_mask_cpu.gather(1, ranked_sources_cpu)",
            "snapshot.copy_(snapshot_cpu.to(device=snapshot.device))",
            "P10_FINAL_HIT_METADATA_READY ",
            "P10_FINAL_HIT_PAYLOAD_READY ",
        ):
            self.assertIn(token, self.fixture)
        self.assertNotIn(
            "torch.arange(RESIDENT_ROWS, dtype=torch.int32, device=device) * 7",
            self.fixture,
        )
        self.assertNotIn("miss_mask.gather(1, ranked_sources)", self.fixture)

    def test_tnd_query_lengths_are_cumulative_and_reference_indices_fail_closed(self) -> None:
        """Batched W4A4C8 must use the product TND cumulative-length ABI."""

        for token in (
            "actual_query = torch.arange(",
            "1, BATCH + 1, dtype=torch.int32, device=device",
            "product W4A4C8 Top-K source index outside the logical row",
            "invalid = (topk_cpu < 0) | (topk_cpu >= SOURCE_TOKENS)",
            "Do not clamp or",
        ):
            self.assertIn(token, self.fixture)
        self.assertNotIn(
            "actual_query = torch.ones(BATCH, dtype=torch.int32, device=device)",
            self.fixture,
        )

    def test_benchmark_hashes_fp8_storage_without_a_numpy_float8_conversion(self) -> None:
        """The input-immutability gate must work for the real FP8 Q/K ABI."""

        self.assertIn("raw_bytes = tensor.view(torch.uint8)", self.fixture)
        self.assertIn("raw_bytes.numpy().tobytes()", self.fixture)
        self.assertNotIn("return hashlib.sha256(tensor.numpy().tobytes())", self.fixture)
        self.assertIn("P10_FLOAT8_RAW_HASH_PROBE", self.runner)
        self.assertIn("value.reshape(-1).view(torch.uint8)", self.runner)


if __name__ == "__main__":
    unittest.main()
