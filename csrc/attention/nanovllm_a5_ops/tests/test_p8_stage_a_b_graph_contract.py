#!/usr/bin/env python3
"""P8 graph seam gate: score receipt -> CandidateSeed -> forward-owned Get.

This is intentionally a source/ABI test, not a claim that C8 has been
modified.  Stage A is the existing isolated score-receipt consumer.  P8 uses
its published GM ABI exactly as-is and consumes it in a separately built
Stage-B OPP.  Keeping the seam explicit prevents accidental re-entry into
C8 or the retired P7.1 transport executor.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


OPS_ROOT = Path(__file__).resolve().parents[1]
STAGE_A = OPS_ROOT / "a5_score_receipt_candidate_executor"
STAGE_B = OPS_ROOT / "a5_p8_forward_owned_executor"
STAGE_A_SPEC = json.loads((STAGE_A / "ops.json").read_text(encoding="utf-8"))[0]
STAGE_B_SPECS = json.loads((STAGE_B / "ops.json").read_text(encoding="utf-8"))
STAGE_B_SPEC = STAGE_B_SPECS[0]
STAGE_B_ADVANCE_SPEC = STAGE_B_SPECS[1]
STAGE_B_ADMISSION_SPEC = STAGE_B_SPECS[2]
STAGE_A_KERNEL = (
    STAGE_A / "op_kernel" / "a5_score_receipt_candidate_executor.cpp"
).read_text(encoding="utf-8")
STAGE_B_KERNEL = (
    STAGE_B / "op_kernel" / "a5_p8_forward_owned_executor.cpp"
).read_text(encoding="utf-8")
STAGE_B_TILING = (
    STAGE_B / "op_kernel" / "a5_p8_forward_owned_executor_tiling.h"
).read_text(encoding="utf-8")
STAGE_B_BRIDGE = (
    STAGE_B / "torch_extension" / "csrc" / "p8_forward_owned_executor.cpp"
).read_text(encoding="utf-8")
RUNNER = (
    OPS_ROOT / "tests" / "run_p8_forward_owned_executor_npu.sh"
).read_text(encoding="utf-8")
RUNTIME = (
    OPS_ROOT / "tests" / "test_p8_forward_owned_executor_runtime.py"
).read_text(encoding="utf-8")
REAL_FIXTURE = (
    OPS_ROOT / "tests" / "p8_real_direct_hccp_fixture.py"
).read_text(encoding="utf-8")
STAGE_A_INIT = (
    STAGE_A / "torch_extension" / "nanovllm_score_receipt_executor" / "__init__.py"
).read_text(encoding="utf-8")
STAGE_B_INIT = (
    STAGE_B / "torch_extension" / "nanovllm_p8_forward_owned_executor" / "__init__.py"
).read_text(encoding="utf-8")
REFERENCE_BRIDGE_SETUP = (
    OPS_ROOT / "a5_official_c8_reference" / "torch_extension" /
    "official_c8_reference_setup.py"
).read_text(encoding="utf-8")
STAGE_A_BRIDGE_SETUP = (
    STAGE_A / "torch_extension" / "score_receipt_candidate_executor_setup.py"
).read_text(encoding="utf-8")
STAGE_B_BRIDGE_SETUP = (
    STAGE_B / "torch_extension" / "p8_forward_owned_executor_setup.py"
).read_text(encoding="utf-8")


def _schema(bridge: str, opname: str, next_opname: str | None = None) -> str:
    """Extract exactly one TORCH_LIBRARY schema.

    The bridge deliberately contains two P8 graph nodes.  A greedy or
    delimiter-only search can accidentally join their return tuples and hide
    an ABI shift, which is precisely the failure this source test prevents.
    """
    start = bridge.index(f'"{opname}(')
    if next_opname is None:
        end = bridge.index('");', start)
    else:
        end = bridge.index(f'    m.def(\n        "{next_opname}(', start)
    return bridge[start:end]


def _tensor_names(schema: str) -> tuple[list[str], list[str]]:
    args, returns = schema.split(") -> (", maxsplit=1)
    pattern = r"Tensor(?:\([a-z0-9_]+!\))? ([a-z0-9_]+)"
    return re.findall(pattern, args), re.findall(pattern, returns)


class P8StageABGraphContractTest(unittest.TestCase):
    def test_stage_a_is_the_completed_score_receipt_consumer(self) -> None:
        self.assertEqual(STAGE_A_SPEC["op"], "A5ScoreReceiptCandidateExecutor")
        self.assertEqual(
            [item["name"] for item in STAGE_A_SPEC["input_desc"][:6]],
            [
                "score_workspace",
                "tile_receipts",
                "receipt_identity",
                "score_source_lens",
                "resident_threshold",
                "resident_threshold_valid",
            ],
        )
        self.assertIn("ReadGmByPassDCache<uint16_t>", STAGE_A_KERNEL)
        self.assertIn("score < threshold", STAGE_A_KERNEL)
        self.assertIn("kvObjectSlotsSnapshot_", STAGE_A_KERNEL)
        self.assertIn("DataSyncBarrier<MemDsbT::DDR>()", STAGE_A_KERNEL)
        for token in (
            "kCandidateReceiptIdentityWords = 4U",
            "LoadStableInputIdentity",
            "TombstoneOutputIdentity",
            "PublishOutputIdentity",
        ):
            self.assertIn(token, STAGE_A_KERNEL)

    def test_candidate_seed_abi_is_exactly_reused_by_stage_b(self) -> None:
        stage_a_outputs = [item["name"] for item in STAGE_A_SPEC["output_desc"]]
        stage_b_inputs = [item["name"] for item in STAGE_B_SPEC["input_desc"]]
        self.assertEqual(
            stage_a_outputs[:4],
            [
                "candidate_seeds",
                "candidate_counts",
                "candidate_receipt_identity",
                "receipt_status",
            ],
        )
        self.assertEqual(
            stage_b_inputs[:4],
            [
                "candidate_seeds",
                "candidate_counts",
                "candidate_receipt_identity",
                "candidate_status",
            ],
        )
        stage_a_types = [item["type"] for item in STAGE_A_SPEC["output_desc"][:4]]
        stage_b_types = [item["type"] for item in STAGE_B_SPEC["input_desc"][:4]]
        self.assertEqual(stage_a_types, stage_b_types)
        self.assertIn("kA5P8CandidateSeedWords = 4U", STAGE_B_TILING)
        self.assertIn("kA5P8CandidateReceiptIdentityWords = 4U", STAGE_B_TILING)
        self.assertIn("LoadCurrentCandidateReceipt", STAGE_B_KERNEL)
        self.assertIn("candidateReceiptIdentity_", STAGE_B_KERNEL)
        self.assertIn("candidateCursorIdentity_", STAGE_B_KERNEL)
        self.assertIn("candidateStatus_", STAGE_B_KERNEL)

    def test_forward_control_is_a_graph_visible_mutable_alias(self) -> None:
        stage_b_outputs = [item["name"] for item in STAGE_B_SPEC["output_desc"]]
        self.assertEqual(stage_b_outputs[0], "forward_control")
        self.assertIn("kForwardControlOut = 0", (STAGE_B / "op_host" /
                                                   "a5_p8_forward_owned_executor.cpp").read_text(encoding="utf-8"))
        self.assertIn("Tensor(a!) forward_control,", STAGE_B_BRIDGE)

    def test_bridge_schema_is_exactly_28_inputs_and_19_alias_outputs(self) -> None:
        """Reject input/return alias drift before a CANN build.

        In particular, `forward_control` has one input occurrence and one
        return occurrence.  A duplicated schema argument can still compile a
        bridge but shifts every later P8 tensor at ACLNN dispatch time.
        """
        schema = _schema(
            STAGE_B_BRIDGE, "forward_owned_executor",
            "advance_forward_control",
        )
        input_names, output_names = _tensor_names(schema)
        self.assertEqual(len(input_names), 28, input_names)
        self.assertEqual(len(output_names), 19, output_names)
        self.assertEqual(input_names.count("forward_control"), 1)
        self.assertEqual(output_names.count("forward_control"), 1)
        self.assertEqual(input_names[:6], [
            "candidate_seeds", "candidate_counts", "candidate_receipt_identity",
            "candidate_status", "forward_control", "p8_arena_health",
        ])
        self.assertEqual(output_names[0], "forward_control")
        self.assertEqual(
            output_names,
            [item["name"] for item in STAGE_B_SPEC["output_desc"]],
        )

    def test_advance_schema_latches_arena_health_on_abort(self) -> None:
        """Control-transition failure must poison the shared arena as well.

        A forward abort without a sticky health latch would leave later graph
        replays free to lease records whose previous owner failed mid-phase.
        """
        schema = _schema(STAGE_B_BRIDGE, "advance_forward_control")
        input_names, output_names = _tensor_names(schema)
        self.assertEqual(
            input_names,
            [
                "forward_control",
                "p8_arena_health",
                "expected_identity",
                "transition_program",
                "transition_result",
            ],
        )
        self.assertEqual(
            output_names,
            ["forward_control", "p8_arena_health", "transition_result"],
        )
        self.assertEqual(
            [item["name"] for item in STAGE_B_ADVANCE_SPEC["input_desc"]],
            input_names,
        )
        self.assertEqual(
            [item["name"] for item in STAGE_B_ADVANCE_SPEC["output_desc"]],
            output_names,
        )
        self.assertIn("aclnnA5P8ForwardControlAdvance", STAGE_B_BRIDGE)

    def test_admission_gate_is_a_one_time_graph_node_with_three_aliases(self) -> None:
        """Admission protects the fixed arena; it is not a fixed Get core."""
        schema = _schema(STAGE_B_BRIDGE, "arena_admission_gate")
        input_names, output_names = _tensor_names(schema)
        self.assertEqual(
            input_names,
            [
                "forward_control",
                "p8_arena_health",
                "fused_get_control",
                "p8_wave_records",
                "p8_private_records",
                "expected_identity",
                "admission_result",
            ],
        )
        self.assertEqual(
            output_names,
            ["forward_control", "p8_arena_health", "admission_result"],
        )
        self.assertEqual(
            [item["name"] for item in STAGE_B_ADMISSION_SPEC["input_desc"]],
            input_names,
        )
        self.assertEqual(
            [item["name"] for item in STAGE_B_ADMISSION_SPEC["output_desc"]],
            output_names,
        )
        self.assertIn("aclnnA5P8ArenaAdmissionGate", STAGE_B_BRIDGE)

    def test_stage_a_release_marker_precedes_stage_b_consumption(self) -> None:
        # Stage A writes candidate payload/status, publishes identity lanes
        # [epoch,replay,layer], barriers, and writes generation last.  Stage B
        # samples the release generation twice around status consumption.
        publish_start = STAGE_A_KERNEL.index("void PublishOutputIdentity")
        publish_end = STAGE_A_KERNEL.index("void ClearSeeds", publish_start)
        publish = STAGE_A_KERNEL[publish_start:publish_end]
        prefix = publish.index("output + 0U")
        barrier = publish.index("DataSyncBarrier<MemDsbT::DDR>()", prefix)
        generation = publish.index("output + 3U", barrier)
        self.assertLess(prefix, barrier)
        self.assertLess(barrier, generation)
        consume_start = STAGE_B_KERNEL.index("bool LoadCurrentCandidateReceipt")
        consume_end = STAGE_B_KERNEL.index("void TombstoneReceiptIdentity", consume_start)
        consume = STAGE_B_KERNEL[consume_start:consume_end]
        self.assertLess(
            consume.index("LoadStableReceiptIdentity(CandidateReceiptIdentity(receipt)"),
            consume.index("candidateStatus_ + receipt"),
        )
        self.assertGreaterEqual(
            consume.count("LoadStableReceiptIdentity(CandidateReceiptIdentity(receipt)"), 2,
        )

    def test_terminal_tombstones_both_receipt_and_cursor_identity(self) -> None:
        terminal_start = STAGE_B_KERNEL.index("void TombstoneCurrentCandidateReceipt")
        terminal_end = STAGE_B_KERNEL.index("void RunForwardTerminal", terminal_start)
        terminal = STAGE_B_KERNEL[terminal_start:terminal_end]
        for token in (
            "TombstoneReceiptIdentity(CandidateReceiptIdentity(receipt))",
            "candidateStatus_ + receipt",
            "TombstoneReceiptIdentity(CandidateCursorIdentity(receipt))",
            "candidateCursor_ + receipt",
            "kCandidateStatusTombstoned",
        ):
            self.assertIn(token, terminal)

    def test_p8_stage_b_has_its_own_bridge_and_no_c8_or_p7_transport(self) -> None:
        self.assertIn("aclnnA5P8ForwardOwnedExecutor", STAGE_B_BRIDGE)
        self.assertIn("P8_FORWARD_OWNED_BRIDGE_ABI", STAGE_B_BRIDGE)
        self.assertIn("NANOVLLM_A5_P8_FORWARD_OWNED_OPAPI", STAGE_B_BRIDGE)
        for forbidden in (
            "A5ScoreReceiptSpeculativeExecutor",
            "P7DirectLaunchLease",
            "A5StreamingQuantLiGet",
            "A5OfficialC8",
        ):
            self.assertNotIn(forbidden, STAGE_B_BRIDGE)

    def test_single_p8_runner_uses_fresh_exact_paths_without_artifact_reuse(self) -> None:
        """Remote validation has one explicit NPU7 path, not shell folklore."""
        for token in (
            "P8_VLLM_EXPECTED_SHA",
            "P8_KVAF_EXPECTED_SHA",
            "P8_REFERENCE_VLLM_ROOT",
            "! -e \"$repo/.git\"",
            "rev-parse --is-inside-work-tree",
            "git -C \"$repo\" status --porcelain --untracked-files=all",
            "mktemp -d \"$P8_OPS_ROOT/build/p8_npu7_",
            "NANOVLLM_A5_OPS_ONLY=A5ScoreReceiptCandidateExecutor",
            "NANOVLLM_A5_OPS_ONLY=A5P8ForwardOwnedExecutor,A5P8ForwardControlAdvance,A5P8ArenaAdmissionGate",
            "NANOVLLM_A5_P8_KVAF_EXPECTED_SHA",
            "P8_REFERENCE_BRIDGE_ROOT",
            "P8_STAGE_A_BRIDGE_ROOT",
            "P8_STAGE_B_BRIDGE_ROOT",
            "reference_c8_bridge_package_missing",
            "stage_a_bridge_package_missing",
            "stage_b_bridge_package_missing",
            'build_py --force --build-lib "$P8_REFERENCE_BRIDGE"',
            'build_py --force --build-lib "$P8_STAGE_A_BRIDGE"',
            'build_py --force --build-lib "$P8_STAGE_B_BRIDGE"',
            'build_ext --force --build-lib "$P8_REFERENCE_BRIDGE"',
            'build_ext --force --build-lib "$P8_STAGE_A_BRIDGE"',
            'build_ext --force --build-lib "$P8_STAGE_B_BRIDGE"',
            "source_changed_during_scoped_build",
            "P8_PHYSICAL_NPU_ID",
            "ASCEND_RT_VISIBLE_DEVICES",
            "A5_SOC_VERSION=ascend950",
            "unset ASCEND_CUSTOM_OPP_PATH NANOVLLM_CUST_OPAPI_LIB",
            "PYTHONNOUSERSITE=1",
            "test_p8_forward_owned_executor_runtime.py",
        ):
            self.assertIn(token, RUNNER)
        for token in (
            'is_clean_exact_repo vllm_target "$P8_VLLM_ROOT"',
            'is_clean_exact_repo vllm_reference "$P8_REFERENCE_VLLM_ROOT"',
            'A5_OFFICIAL_C8_SOURCE_ROOT="$P8_REFERENCE_VLLM_ROOT"',
            "vllm_target=%s vllm_reference=%s",
        ):
            self.assertIn(token, RUNNER)
        self.assertNotIn("git -C \"$repo\" fetch", RUNNER)
        self.assertNotIn("find ", RUNNER)
        self.assertNotRegex(RUNNER, r"(?m)^\s*(?:export\s+)?SOC_VERSION=")

    def test_fail_closed_fixture_allows_only_control_rejected_diagnostics(self) -> None:
        """Invalid Direct-HCCP control must not look like a no-op or a Post."""
        for token in (
            "FAIL_CLOSED_TRANSPORT_INPUTS",
            "WORKER_STATS_INPUT = 27",
            "receipt_tombstone=1 cursor_tombstone=1",
            "CONTROL_REJECTED_STAT_COLUMN = 10",
            "P8_FAIL_CLOSED_DIAGNOSTIC",
            "non_transport_mutation=0",
        ):
            self.assertIn(token, RUNTIME)

    def test_runtime_orders_prepare_admission_receipt_attach_then_post(self) -> None:
        """No Stage-A publication may race an unadmitted/reset arena.

        The fixture deliberately has a two-step API.  It cold-binds the
        graph-stable arena, admission leases that exact storage, and only then
        does Stage A publish CandidateSeed GM.  Attachment must preserve the
        admitted pointers before Stage B can POST.
        """
        prepare = RUNTIME.index("prepared, attach, real_mode = _prepare_fixture(")
        invoke = RUNTIME.index("admission = _run_arena_admission_gate(shell")
        stage_a = RUNTIME.index("stage_a = _run_stage_a(device, stage_a_fixture)")
        attach = RUNTIME.index("inputs, identity, real_fixture = _attach_stage_a_receipts(")
        post = RUNTIME.index('_run_executor(inputs, "post")')
        self.assertLess(prepare, invoke)
        self.assertLess(invoke, stage_a)
        self.assertLess(stage_a, attach)
        self.assertLess(attach, post)
        for token in (
            "P8_ARENA_ADMISSION",
            "P8_ARENA_ATTACHMENT",
            "admitted_shell_alias=1",
            "stage_a_after_admission=1",
            "dynamic_get_workers=1",
            "ADMISSION_CONTROL_INVALID",
            "arena_admission_gate",
            "prepare_p8_real_fixture",
            "attach_p8_stage_a_receipts",
        ):
            self.assertIn(token, RUNTIME)

    def test_runtime_releases_only_after_qsfa_terminal_and_before_reuse(self) -> None:
        """The single-layer harness is the terminal layer of one forward."""
        qsfa = RUNTIME.index('_run_executor(inputs, "qsfa_terminal")')
        forward = RUNTIME.index('_run_executor(inputs, "forward_terminal")')
        reuse = RUNTIME.rindex("_validate_terminal_reuse_fence(")
        self.assertLess(qsfa, forward)
        self.assertLess(forward, reuse)
        for token in (
            "P8_LAYER_REUSE_CONTRACT",
            "not once per layer",
            "bounded CQ drain *and* QSFA terminal",
            "P8_REUSE_FENCE",
            "next_forward_reset_before_fence=0",
        ):
            self.assertIn(token, RUNTIME)
        for token in (
            "prepare_p8_real_fixture",
            "attach_p8_stage_a_receipts",
            "P8_REAL_ADMISSION",
            "P8_REAL_RECEIPT_ATTACH",
            "P8_REAL_REUSE_FENCE",
            "reuse_fence_validated",
        ):
            self.assertIn(token, REAL_FIXTURE)

    def test_runtime_proves_all_three_imported_bridges_are_fresh(self) -> None:
        for token in (
            "P8_ARTIFACT_PROVENANCE",
            "P8 stale",
            "nanovllm_official_c8_reference",
            "nanovllm_score_receipt_executor",
            "nanovllm_p8_forward_owned_executor",
            "P8_REFERENCE_BRIDGE_ROOT",
            "P8_STAGE_A_BRIDGE_ROOT",
            "P8_STAGE_B_BRIDGE_ROOT",
            "exact=1",
        ):
            self.assertIn(token, RUNTIME)

    def test_runtime_defers_reference_c8_import_until_the_fixture_selects_opp(self) -> None:
        """Provenance preflight must not initialize the C8 resolver itself.

        The source-matched reference bridge binds its selected OPP at its
        first import.  Stage-A/B can be imported at process start, but C8 must
        remain deferred until the real fixture selects the exact reference
        OPP; then the same run-root provenance is checked again while loaded.
        """
        preflight = RUNTIME.index(
            "_assert_fresh_bridge_provenance(reference_must_be_loaded=False)"
        )
        prepare = RUNTIME.index("prepared, attach, real_mode = _prepare_fixture(")
        loaded = RUNTIME.index(
            "_assert_fresh_bridge_provenance(reference_must_be_loaded=real_mode)"
        )
        self.assertLess(preflight, prepare)
        self.assertLess(prepare, loaded)
        self.assertIn("reference_state = \"deferred\"", RUNTIME)
        self.assertIn("reference_state = \"loaded\"", RUNTIME)
        self.assertNotIn(
            'importlib.import_module("nanovllm_official_c8_reference")', RUNTIME
        )

    def test_fresh_bridge_builds_copy_init_from_their_own_source_roots(self) -> None:
        """Reject CWD-dependent `find_packages()` in the per-run bridges.

        Each bridge is invoked by an absolute setup-script path while the P8
        runner remains in another directory.  A bare ``find_packages()`` then
        builds only ``_C.so`` in the run root; Python sees its package as a
        namespace package and provenance loses ``module.__file__``.  This
        exact source mapping keeps the package init, OPP binding, and native
        extension together in every fresh P8 run.
        """
        for setup_source in (
            REFERENCE_BRIDGE_SETUP,
            STAGE_A_BRIDGE_SETUP,
            STAGE_B_BRIDGE_SETUP,
        ):
            self.assertIn("find_packages(where=str(ROOT))", setup_source)
            self.assertIn('package_dir={"": str(ROOT)}', setup_source)
            self.assertNotIn("packages=find_packages(),", setup_source)

    def test_real_hccp_acceptance_requires_an_unchanged_c8_math_gate(self) -> None:
        """Do not mislabel a transport-only fixture as P8 correctness proof."""
        self.assertIn('if args.require_real_hccp and c8_gate != "pass"', RUNTIME)
        self.assertIn("sparse_indices and score_source_lens baselines", RUNTIME)

    def test_real_fixture_is_fresh_kvaf_only_and_checks_actual_hccp_payload(self) -> None:
        """The NPU acceptance path cannot quietly reuse P7's known-failed probe."""
        for token in (
            'P8_KVAF_BUILD="$P8_RUN_ROOT/kvaf_build"',
            'cmake -S "$P8_KVAF_ROOT" -B "$P8_KVAF_BUILD"',
            "P8_KVAF_BATCH_GET_KERNEL",
            "P8_KVAF_BATCH_PUT_KERNEL",
            "stale kvaf import",
            "p8_real_direct_hccp_fixture",
        ):
            self.assertIn(token, RUNNER)
        for token in (
            "build_p8_real_fixture",
            "bind_streaming_batch_get_control",
            "configure_streaming_batch_get_queue_templates",
            "P8_REAL_C8_BEFORE",
            "P8_REAL_C8_AFTER",
            "P8_REAL_PAYLOAD status=mismatch bandwidth_reported=0",
            "P8_REAL_PAYLOAD status=validated bandwidth_reported=0",
            "PostOnly->TryRetire->Finish",
            "happy_post_cq_qsfa_release",
            "link_to",
        ):
            self.assertIn(token, REAL_FIXTURE)
        self.assertNotIn("known_failed_not_checked", REAL_FIXTURE)

    def test_terminal_coverage_is_explicitly_split_between_hardware_and_native_contracts(self) -> None:
        """A happy HCCP run must not pretend to inject impossible CQ faults."""
        self.assertIn('REQUIRED_REAL_CASES = {"happy_post_cq_qsfa_release"}', RUNTIME)
        # The targeted native P8 source contract covers the other terminal
        # branches.  They are deliberately labelled as contracts, not as
        # payload-validated hardware error injection.
        contract = (OPS_ROOT / "tests" / "test_p8_forward_owned_executor_contract.py").read_text(
            encoding="utf-8"
        )
        for test_name in (
            "test_deterministic_no_post_is_cleaned_only_by_drain",
            "test_failed_cq_is_terminal_private_miss_not_qsfa_ready",
            "test_forward_terminal_quarantines_post_intent_not_release_it",
        ):
            self.assertIn(test_name, contract)

    def test_three_opp_vendors_are_composed_not_overwritten_by_import_order(self) -> None:
        """C8, Stage A, and Stage B share one process during real P8 proof."""
        for init in (STAGE_A_INIT, STAGE_B_INIT):
            self.assertIn("_existing_vendors", init)
            self.assertIn("ASCEND_CUSTOM_OPP_PATH", init)
            self.assertNotIn('os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(_VENDOR)', init)
        self.assertIn('c8._prepend_unique_path("ASCEND_CUSTOM_OPP_PATH", (vendor,))', REAL_FIXTURE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
