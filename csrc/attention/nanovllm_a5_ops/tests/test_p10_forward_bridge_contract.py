"""Fail-closed contract for P10's real graph bridge.

The test deliberately checks graph wiring, not numerical output.  Numerical
truth remains the native W4A4C8 Stage-A path; this gate prevents accidental
fallback to a host candidate loop, a Stage-A completion dependency, or an
identity assumption between private resident rows and QSFA slot IDs.
"""

from __future__ import annotations

import json
import py_compile
import subprocess
import tempfile
import unittest
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[1]
ROOT = (
    OPS_ROOT
    / "a5_dsa_c8_exact_speculative_resident"
    / "p10_forward_bridge"
)


class P10ForwardBridgeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads((ROOT / "P10_FORWARD_BRIDGE_CONTRACT.json").read_text())
        cls.prepare_definition = (
            ROOT / "op_host" / "a5_p10_forward_prepare_def.cpp"
        ).read_text()
        cls.prepare_kernel = (
            ROOT / "op_kernel" / "a5_p10_forward_prepare.cpp"
        ).read_text()
        cls.prepare_abi = (
            ROOT / "op_kernel" / "a5_p10_forward_prepare_abi.h"
        ).read_text()
        cls.bridge = (
            ROOT / "torch_extension" / "csrc" / "p10_forward_bridge.cpp"
        ).read_text()
        cls.direct_opapi = (
            ROOT / "torch_extension" / "csrc" / "p10_direct_opapi.h"
        ).read_text()
        cls.graph = (
            ROOT
            / "torch_extension"
            / "nanovllm_p10_forward_bridge"
            / "p10_forward_graph.py"
        ).read_text()
        cls.init = (
            ROOT
            / "torch_extension"
            / "nanovllm_p10_forward_bridge"
            / "__init__.py"
        ).read_text()
        cls.builder = (ROOT / "build_p10_forward_prepare_opp.sh").read_text()
        cls.setup = (ROOT / "torch_extension" / "p10_forward_bridge_setup.py").read_text()
        cls.runtime_fixture = (
            OPS_ROOT / "tests" / "test_p10_w4a4c8_forward_resident_runtime.py"
        ).read_text()
        cls.common_opapi = (
            OPS_ROOT / "torch_extension" / "csrc" / "op_api_common.h"
        ).read_text()

    def test_prepare_is_a_real_graph_node_with_exact_reset_ownership(self) -> None:
        prepare = self.contract["graph_nodes"]["prepare"]
        self.assertTrue(prepare["graph_visible"])
        self.assertEqual(prepare["operator"], "A5P10ForwardPrepare")
        for name in (
            "receipt_state",
            "forward_control",
            "kv_object_to_forward_slot",
            "source_to_forward_slot",
            "speculative_slot_cursor",
            "queue_leases",
            "resident_ready",
        ):
            self.assertIn(f'Input("{name}")', self.prepare_definition)
            self.assertIn(f'Output("{name}_alias")', self.prepare_definition)
        for token in (
            "A5_P10_FORWARD_PREPARE_RECEIPT_FREE",
            "A5_P10_FORWARD_PREPARE_PRODUCER_OPEN",
            "A5_P10_FORWARD_PREPARE_UNMAPPED_SLOT",
            "FillShard<A5_P10_FORWARD_PREPARE_UNMAPPED_SLOT>",
            "FillShard<0>(queue_leases_gm",
            "FillShard<0>(resident_ready_gm",
            "kForwardPrepareInt32PerDCacheLine",
            "DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,",
            "DcciDst::CACHELINE_OUT>(tensor[begin])",
            "WriteGmByPassDCache<int32_t>(speculative_slot_cursor_gm, 0)",
            "A5_P10_FORWARD_PREPARE_PRODUCER_SEAL_SEQUENCE_WORD",
            "A5_P10_FORWARD_PREPARE_FIRST_PRESEAL_READY_CLAIM_WORD",
            "DataSyncBarrier<MemDsbT::DDR>()",
        ):
            self.assertIn(token, self.prepare_kernel)
        for forbidden in (
            "receipt_ring",
            "score_surface",
            "canonical_sparse_indices",
            "attention_slots",
        ):
            self.assertNotIn(forbidden, self.prepare_kernel)

    def test_prepare_publishes_all_sharded_reset_writes_before_stage_b_can_bypass_read(self) -> None:
        """Each full DCache line has one owner before the trailing DDR DSB.

        The historical element-strided reset let several AIVs dirty different
        words in a single cache line.  A later writeback could erase only the
        object half of a freshly published Stage-B mapping, leaving an
        impossible positive source map.  DSB alone cannot repair that.
        """

        publish = "DataSyncBarrier<MemDsbT::DDR>();"
        self.assertEqual(self.prepare_kernel.count(publish), 1)
        self.assertGreater(self.prepare_kernel.rfind(publish), self.prepare_kernel.rfind(
            "A5_P10_FORWARD_PREPARE_FIRST_PRESEAL_READY_CLAIM_WORD,"))
        self.assertIn("line += core_count", self.prepare_kernel)
        self.assertIn("const uint32_t begin = line * kForwardPrepareInt32PerDCacheLine", self.prepare_kernel)
        self.assertIn("DataCacheCleanAndInvalid<int32_t", self.prepare_kernel)
        self.assertNotIn("index = core_index; index < element_count", self.prepare_kernel)
        self.assertIn("unique owner", self.prepare_kernel)
        self.assertNotIn("SyncAll(", self.prepare_kernel)

    def test_no_fixed_physical_core_model_enters_the_forward_bridge(self) -> None:
        combined = "\n".join((self.prepare_definition, self.prepare_kernel, self.bridge, self.graph))
        for forbidden in (
            "physical_aiv_id",
            "physical_core",
            "fixed_get_core",
            "core_id=",
            "aiv_id",
        ):
            self.assertNotIn(forbidden, combined.lower())
        self.assertIn("bounded logical", self.prepare_definition.lower())
        self.assertIn("executor_worker_count", self.bridge)
        self.assertIn("executor_worker_limit", self.bridge)

    def test_stage_b_uses_per_request_22_input_mapping_abi(self) -> None:
        stage_b = self.contract["graph_nodes"]["stage_b"]
        self.assertEqual(stage_b["inputs"], 22)
        self.assertEqual(stage_b["alias_outputs"], 11)
        self.assertIn("resident_row_to_qsfa_slot", self.bridge)
        self.assertIn("resident_payload.size(0)", self.bridge)
        self.assertIn("never assumes resident rows are QSFA logical slot IDs", self.bridge)
        self.assertIn("at::kUInt16", self.bridge)
        self.assertIn("predicted_score_cutoff_u16", self.bridge)
        self.assertIn("predicted_score_cutoff_u16", self.graph)
        self.assertIn("required_cold_bound_uint16_per_active_row_native_sortable_score_threshold", stage_b["predicted_score_cutoff_u16"])
        self.assertIn("case at::kUInt16", self.common_opapi)
        self.assertIn("return ACL_UINT16", self.common_opapi)
        self.assertIn("aclnnA5P10ScoreReceiptForwardResident", self.bridge)

    def test_graph_has_prepare_fanout_but_no_stage_a_to_stage_b_completion_edge(self) -> None:
        # The default graph is the product overlap route.  A distinct
        # benchmark-only serial branch is permitted below, but must not alter
        # the inference default.
        eager = self.graph.split("def launch_p10_forward(", 1)[1]
        overlap = eager.split('if execution_mode == "overlap":', 1)[1].split(
            "    else:\n        # Benchmark-only serial reference.", 1
        )[0]
        self.assertIn('execution_mode: str = "overlap"', self.graph)
        self.assertIn("_launch_prepare(inputs)", self.graph)
        self.assertIn("producer_stream.wait_event(prepared)", overlap)
        self.assertIn("executor_stream.wait_event(prepared)", overlap)
        self.assertIn("_launch_stage_a(inputs)", overlap)
        self.assertIn("_launch_stage_b(inputs)", overlap)
        self.assertNotIn("serial_stage_a_complete", overlap)
        self.assertNotIn("wait_event(serial_stage_a", overlap)
        self.assertIn("stage_b_complete = _record_event(executor_stream)", overlap)
        self.assertIn("qsfa_stream.wait_event(self.stage_b_complete)", self.graph)

    def test_serial_reference_is_explicitly_benchmark_only_and_same_operator_path(self) -> None:
        eager = self.graph.split("def launch_p10_forward(", 1)[1]
        serial = eager.split("# Benchmark-only serial reference.", 1)[1]
        reference = self.contract["benchmark_only_serial_reference"]
        self.assertIn("execution_mode == \"overlap\"", self.graph)
        self.assertIn("serial_stage_a_complete = _record_event(producer_stream)", serial)
        self.assertIn("executor_stream.wait_event(serial_stage_a_complete)", serial)
        self.assertIn("_launch_stage_a(inputs)", serial)
        self.assertIn("_launch_stage_b(inputs)", serial)
        self.assertIn("No host synchronization occurs", serial)
        self.assertEqual(
            reference["sole_added_edge"],
            "stage_a_completion_event_to_stage_b_start",
        )
        self.assertEqual(reference["inference_default"], "overlap")

    def test_companion_is_submitted_before_native_c8_without_a_reverse_dependency(self) -> None:
        """Short forwards need the device poller admitted before C8 can seal.

        This is deliberately host submission order only.  It must never turn
        into a Stage-B completion edge that stalls the mathematical C8 path.
        """

        eager = self.graph.split("def launch_p10_forward(", 1)[1]
        stage_b_submit = eager.index("_launch_stage_b(inputs)")
        stage_a_submit = eager.index("_launch_stage_a(inputs)")
        self.assertLess(stage_b_submit, stage_a_submit)
        self.assertIn("submission-order admission rule", self.graph)
        self.assertNotIn("wait_event(stage_b", self.graph)
        self.assertEqual(
            self.contract["graph_nodes"]["stage_b"]["submission_order"],
            "stage_b_before_stage_a_after_common_prepare",
        )

    def test_aclgraph_mode_captures_only_replay_safe_nodes_and_relaunches_hccp(self) -> None:
        """Graph mode must not pretend Direct-HCCP is a replay-safe graph node.

        P10 bypasses torch_npu's task queue and issues raw ACLNN calls. A
        ``torch.npu.NPUGraph`` can return successfully without retaining those
        calls, leaving capture-time control state but no replayed HCCP DMA.
        Raw ``aclmdlRI`` is used for the replay-safe Prepare and C8 score
        nodes.  Stage B's HCCP doorbell is relaunched on its device executor
        stream for every replay, so a terminal-ready alias cannot be accepted
        without a real replay-time payload transfer.
        """

        self.assertIn("class P10ForwardGraph", self.graph)
        self.assertNotIn("torch.npu.NPUGraph()", self.graph)
        self.assertIn("_require_raw_acl_graph_type", self.graph)
        self.assertIn('getattr(kvaf, "AclGraph", None)', self.graph)
        self.assertIn("aclmdlRI", self.graph)
        capture = self.graph.split("class P10ForwardGraph", 1)[1].split(
            "    def replay(", 1
        )[0]
        replay = self.graph.split("    def replay(", 1)[1].split(
            "\ndef _record_event", 1
        )[0]
        self.assertEqual(capture.count("_capture_raw_acl_segment("), 2)
        self.assertNotIn("submit=lambda: _launch_stage_b(inputs)", capture)
        self.assertNotIn("self._stage_b_graph", self.graph)
        self.assertIn("raw_graph.capture_begin(stream)", self.graph)
        self.assertIn("raw_graph.capture_end()", self.graph)
        self.assertIn("raw_graph.reset()", self.graph)
        self.assertIn("P10_GRAPH_CAPTURE_FAILED", capture)
        self.assertIn("nested_aclgraph_capture_forbidden", capture)
        self.assertNotIn("synchronize()", replay)
        overlap = replay.split('if self._execution_mode == "overlap":', 1)[1].split(
            "        else:", 1
        )[0]
        self.assertLess(
            overlap.index("_launch_stage_b(inputs)"),
            overlap.index("self._stage_a_graph.replay(self._producer_stream)"),
        )
        self.assertIn("self._executor_stream.wait_event(prepared)", overlap)
        self.assertIn("self._producer_stream.wait_event(prepared)", overlap)
        self.assertNotIn("wait_event(stage_b", overlap)
        self.assertIn("Direct-HCCP doorbell must be issued for every replay", self.graph)
        self.assertIn("direct_hccp_captured=0", self.runtime_fixture)
        for token in (
            "tensor_static_alias_mismatch",
            "scalar_tiling_mismatch",
            "capture_p10_forward_graph",
        ):
            self.assertIn(token, self.graph)
        for exported in (
            "P10ForwardGraph",
            "capture_p10_forward_graph",
        ):
            self.assertIn(exported, self.init)

    def test_raw_graph_workspace_is_owned_until_the_graph_is_destroyed(self) -> None:
        """A captured ACLNN workspace cannot be freed between capture/replay."""

        for token in (
            "BeginP10GraphWorkspaceCapture",
            "EndP10GraphWorkspaceCapture",
            "RetainP10GraphWorkspace",
            "ReleaseP10GraphWorkspaceCapture",
            "RetainP10GraphWorkspace(p10_workspace)",
            "begin_graph_workspace_capture",
            "end_graph_workspace_capture",
            "release_graph_workspace_capture",
        ):
            self.assertIn(token, self.direct_opapi + self.bridge + self.graph)
        self.assertIn("workspace_tokens", self.graph)
        self.assertIn("Destroy raw graphs before releasing their captured workspaces", self.graph)
        self.assertLess(
            self.graph.index("graph.reset()", self.graph.index("def close(")),
            self.graph.index(
                "release_graph_workspace_capture", self.graph.index("def close(")
            ),
        )

    def test_overlap_witness_is_explicit_and_not_a_normal_forward_wait(self) -> None:
        self.assertIn("class P10OverlapWitness", self.graph)
        self.assertIn("producer_seal_sequence", self.graph)
        self.assertIn("first_preseal_ready_claim_sequence", self.graph)
        self.assertIn("< self.producer_seal_sequence", self.graph)
        self.assertIn("read_overlap_witness_after_completion", self.graph)
        self.assertIn("Product forward code must use :meth:`bind_qsfa`", self.graph)
        self.assertIn("narrow(0, 8, 2)", self.graph)

    def test_qsfa_boundary_is_only_completed_stage_b_aliases(self) -> None:
        self.assertEqual(
            self.contract["graph_nodes"]["qsfa"]["inputs"],
            ["attention_slots", "resident_seq_lengths"],
        )
        self.assertIn("class P10QsfaAliases", self.graph)
        self.assertIn("attention_slots=self.attention_slots", self.graph)
        self.assertIn("resident_seq_lengths=self.resident_seq_lengths", self.graph)
        self.assertNotIn("torch.npu.synchronize", self.graph)
        self.assertNotIn(".cpu()", self.graph)

    def test_bridge_has_no_python_candidate_or_host_kvaf_submission_path(self) -> None:
        combined = "\n".join((self.bridge, self.graph, self.direct_opapi))
        for forbidden in (
            "batch_get",
            "kvafadapter",
            "candidate_seed_submission",
            "for candidate",
            "for seed",
            ".cpu()",
            "torch.npu.synchronize",
            "aclrtSynchronizeStream",
        ):
            self.assertNotIn(forbidden, combined.lower())
        self.assertIn('"device_only"', self.bridge.lower())

    def test_bridge_uses_the_p10_direct_stream_launcher_not_a_task_queue(self) -> None:
        """The bridge owns three coarse graph submissions, never a callback.

        The direct header may mention the historical launcher in comments to
        document why it is excluded.  Match the call spelling, rather than
        making a brittle text rule that would force a fake compatibility
        include into the bridge.
        """

        self.assertIn('#include "p10_direct_opapi.h"', self.bridge)
        self.assertNotIn('#include "op_api_common.h"', self.bridge)
        self.assertEqual(self.bridge.count("P10_EXEC_NPU_CMD_DIRECT("), 3)
        self.assertNotIn("RunOpApi(", self.direct_opapi)
        self.assertNotIn("aclrtSynchronizeStream(", self.direct_opapi)
        for required in (
            "RequireP10DirectDispatchContract",
            "TASK_QUEUE_ENABLE",
            "ASCEND_LAUNCH_BLOCKING",
            "P10_DIRECT_DISPATCH_CONFIG",
            "c10_npu::getCurrentNPUStream",
            "RecordTensorTupleOnNpuStream",
            "RecordTensorStorageOnNpuStream",
            "ClearTensorTuple",
            "NANOVLLM_A5_P10_PRODUCT_OPAPI_LIB",
        ):
            self.assertIn(required, self.direct_opapi + self.bridge)
        self.assertNotIn("InitPtaCacheForWorkspace(", self.direct_opapi)
        self.assertNotIn("UnInitPtaCacheForWorkspace(", self.direct_opapi)

    def test_direct_dispatch_is_fail_closed_tq0_not_an_implicit_tq1_path(self) -> None:
        init_before_torch_npu = self.init.index(
            "_require_p10_direct_dispatch_contract()\n\n_OPAPI"
        )
        torch_npu_import = self.init.index("import torch_npu")
        self.assertLess(init_before_torch_npu, torch_npu_import)
        for required in (
            "TASK_QUEUE_ENABLE=0",
            "ASCEND_LAUNCH_BLOCKING=0",
            "P10_DIRECT_DISPATCH_BLOCKED",
            "P10_DIRECT_DISPATCH_CONFIG",
            "task_queue_fallback=forbidden",
        ):
            self.assertIn(required, self.init + self.direct_opapi)
        self.assertNotIn("fallback_to_tq1", self.init + self.direct_opapi)

    def test_one_exact_product_opapi_is_required(self) -> None:
        assembly = self.contract["product_assembly"]
        self.assertEqual(assembly["opapi_resolver"], "one_fresh_combined_P10_vendor")
        for operator in assembly["operators"]:
            self.assertIn(operator, self.init + self.bridge + self.contract["product_assembly"]["opapi_resolver"])
        self.assertIn("NANOVLLM_A5_P10_PRODUCT_OPAPI_LIB", self.init)
        self.assertIn("NANOVLLM_A5_P10_PRODUCT_OPP_VENDOR", self.init)
        self.assertIn("same source pair", self.init)

    def test_standalone_prepare_build_is_fail_closed(self) -> None:
        """Only the combined three-operator product vendor is valid."""

        result = subprocess.run(
            ["bash", str(ROOT / "build_p10_forward_prepare_opp.sh")],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("combined_product_vendor_required", result.stderr)
        self.assertIn("build_p10_target_assembly.sh", result.stderr)

    def test_build_closure_and_python_adapter_are_syntax_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(ROOT / "build_p10_forward_prepare_opp.sh")],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with tempfile.TemporaryDirectory() as temp:
            py_compile.compile(
                str(
                    ROOT
                    / "torch_extension"
                    / "nanovllm_p10_forward_bridge"
                    / "p10_forward_graph.py"
                ),
                cfile=str(Path(temp) / "p10_forward_graph.pyc"),
                doraise=True,
            )
        self.assertIn("NpuExtension", self.setup)
        self.assertIn("p10_direct_opapi.h", self.bridge)

    def test_setup_resolves_shared_opapi_headers_from_nanovllm_ops_root(self) -> None:
        """A clean P10 bridge build must not depend on a nonexistent sibling.

        ``torch_extension`` is three levels below ``nanovllm_a5_ops`` when
        counting the path itself, but two ``Path.parents`` hops away.  This
        catches the off-by-one that otherwise fails only after an expensive
        remote OPP build reaches the bridge compilation step.
        """

        self.assertIn("OPS_ROOT = ROOT.parents[2]", self.setup)
        self.assertNotIn("OPS_ROOT = ROOT.parents[3]", self.setup)
        self.assertTrue((OPS_ROOT / "torch_extension" / "csrc" / "op_api_common.h").is_file())


if __name__ == "__main__":
    unittest.main()
