# SPDX-License-Identifier: Apache-2.0
"""Opt-in evidence and injected byte corruption using real PD transactions."""

import importlib
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import test_kvaf_pd_flow as flow
import test_kvaf_pd_integration as integration
import test_kvaf_pd_recovery as recovery
import torch

DIAG = importlib.import_module(f"{flow.PACKAGE}.pd.diagnostics")
AUDIT = integration.load_file("pd_diagnostics_audit_for_ut", "scripts/check_kvaf_pd_diagnostics.py")


def diagnostic(role, verify=True):
    result = DIAG.Diagnostics(1, verify, role)
    result.logger = Mock()
    return result


def events(diag):
    return [json.loads(call.args[1]) for call in diag.logger.info.call_args_list]


class TestPDDiagnosticFlow(unittest.TestCase):
    def setUp(self):
        self.fixture = flow.TestPDFlow()

        def call(endpoint, message, **kwargs):
            return self.fixture.receiver.dispatch(message)

        for target, name, value in (
            (flow.RECEIVER, "ControlServer", recovery._DirectServer),
            (flow.SENDER, "call", call),
        ):
            p = patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.diag = diagnostic("D")
        self.fixture.receiver.diagnostics = self.diag
        self.fixture.pool.policy.diagnostics = self.diag
        for sender in self.fixture.senders:
            sender.diagnostics = diagnostic("P")

    def test_full_dual_pull_hashes_and_committed_ddr_readback(self):
        f = self.fixture
        f.seed(300)
        f.chunk(300)
        f.assert_payload(300)
        self.assertEqual(f.receiver.get_finished(), {"r"})
        checks = [e for e in events(self.diag) if e["event"] == "KV_CHECK"]
        self.assertTrue(checks)
        self.assertTrue(all(e["status"] == "PASS" for e in checks))
        self.assertEqual({e["path"] for e in checks}, {"npu", "cpu"})
        self.assertEqual(sum(f.native.reads), 4 * 2 * 128)
        self.assertTrue(any(e["event"] == "DECODE_READY" for e in events(self.diag)))
        self.assertIsNone(self.diag.retained)
        self.assertEqual(AUDIT.audit(events(self.diag), require_ddr=True), (1, ["cpu", "npu"]))
        with self.assertRaisesRegex(ValueError, "compute evidence"):
            AUDIT.audit(events(self.diag), require_compute=True)
        ready = next(e for e in events(self.diag) if e["event"] == "DECODE_READY")
        compute = [
            dict(event=e, role="D", worker=ready["worker"], request=ready["request"])
            for e in ("MODEL_FORWARD", "MTP_PROPOSE_DONE", "ABC_IMPORT")
        ]
        compute.append(
            dict(
                event="MODEL_FORWARD_DONE",
                graph_mode="FULL",
                role="D",
                worker=ready["worker"],
                request=ready["request"],
            )
        )
        self.assertEqual(AUDIT.audit(events(self.diag) + compute, require_compute=True)[0], 1)
        incomplete = [e for e in events(self.diag) if not (e["event"] == "KV_CHECK" and e.get("path") == "cpu")]
        with self.assertRaisesRegex(ValueError, "missing transfer"):
            AUDIT.audit(incomplete)

    def test_hbm_corruption_fails_request_before_decode(self):
        f = self.fixture
        native = f.native.batch_get_raw_addr

        def corrupt(ready, *args):
            result = native(ready, *args)
            args[3].view(torch.uint8).reshape(-1)[int(args[1][0])] ^= 1
            return result

        f.seed(19)
        with patch.object(f.native, "batch_get_raw_addr", side_effect=corrupt):
            f.chunk(19)
        self.assertTrue(any(e["event"] == "KV_CHECK" and e["status"] == "FAIL" for e in events(self.diag)))
        self.assertFalse(any(e["event"] == "DECODE_READY" for e in events(self.diag)))
        self.assertTrue(f.receiver.get_errors())
        self.assertFalse(f.receiver.requests["r"]["published"])
        with self.assertRaisesRegex(ValueError, "FAIL found"):
            AUDIT.audit(events(self.diag))

    def test_ddr_corruption_is_detected_after_commit(self):
        f = self.fixture
        native = f.native.commit_put

        def corrupt(put):
            result = native(put)
            if result.name == "COMMITTED":
                f.native.pages[put.entries[0].key][5] ^= 1
            return result

        f.seed(129)
        with patch.object(f.native, "commit_put", side_effect=corrupt):
            f.chunk(129)
        self.assertTrue(
            any(e["event"] == "KV_CHECK" and e["path"] == "cpu" and e["status"] == "FAIL" for e in events(self.diag))
        )
        self.assertFalse(any(e["event"] == "DECODE_READY" for e in events(self.diag)))
        self.assertTrue(f.receiver.get_errors())

    def test_non_sender_main_replica_difference_blocks_decode(self):
        f = self.fixture
        native = f.prefill

        def corrupt(sender, tokens, blocks):
            native(sender, tokens, blocks)
            if sender.tp_rank == 1:
                memory = sender.memories["l0"]
                memory.planes["main"][blocks[memory.groups["main"]][0]].view(torch.uint8).reshape(-1)[0] ^= 1

        f.seed(19)
        with patch.object(f, "prefill", side_effect=corrupt):
            f.chunk(19)
        self.assertTrue(any(e["event"] == "MAIN_REPLICA_CHECK" and e["status"] == "FAIL" for e in events(self.diag)))
        self.assertFalse(any(e["event"] == "DECODE_READY" for e in events(self.diag)))
        self.assertTrue(f.receiver.get_errors())

    def test_tail_import_and_native_replay_stage_are_checked(self):
        f = self.fixture
        f.seed(300)
        f.chunk(300)
        f.writer.prepare_requests(
            {"r": NS(num_computed_tokens=299, num_prompt_tokens=300, block_ids=f.blocks)}, {"r": 1}
        )
        f.manager.prepare(("r",), {"r": 1}, decode=False, padded_requests=1)
        checks = [e for e in events(self.diag) if e["event"] == "KV_CHECK"]
        self.assertTrue(all(e["status"] == "PASS" for e in checks))
        self.assertTrue({"ordinary_tail_to_ABC", "ABC_TO_STAGE"} <= {e["path"] for e in checks})

    def test_limit_prevents_second_request_hashes_or_readback(self):
        f = self.fixture
        f.seed(129)
        f.chunk(129)
        f.receiver.get_finished()
        f.receiver.release({"r"})
        reads = len(f.native.reads)
        for sender in f.senders:
            sender.diagnostics.hashes = Mock(side_effect=AssertionError("request limit ignored"))
        f.seed(129, req_id="s", generation="b1")
        f.chunk(129, req_id="s", generation="b1")
        self.assertEqual(f.receiver.get_finished(), {"s"})
        self.assertEqual(len(f.native.reads), reads)
        self.assertFalse(self.diag.active("s"))

    def test_verify_mode_mismatch_fails_before_source_registration(self):
        sender = self.fixture.senders[0]
        sender.diagnostics = None
        with self.assertRaisesRegex(ValueError, "verify_kv modes"):
            sender._register_peer(self.fixture.endpoint)

    def test_disabled_diagnostics_do_not_hash_or_read_back(self):
        f = self.fixture
        f.receiver.diagnostics = None
        f.pool.policy.diagnostics = None
        for sender in f.senders:
            sender.diagnostics = None
        f.seed(300)
        with patch.object(DIAG.Diagnostics, "hashes", side_effect=AssertionError("diagnostics not disabled")):
            f.chunk(300)
        self.assertEqual(f.receiver.get_finished(), {"r"})
        self.assertFalse(f.native.reads)
        self.assertFalse(events(self.diag))

    def test_logging_without_verification_adds_no_hash_or_get(self):
        f = self.fixture
        self.diag.verify = False
        for sender in f.senders:
            sender.diagnostics.verify = False
        f.seed(19)
        with patch.object(DIAG.Diagnostics, "hashes", side_effect=AssertionError("log-only mode copied data")):
            f.chunk(19)
        self.assertEqual(f.receiver.get_finished(), {"r"})
        self.assertFalse(f.native.reads)
        self.assertTrue(any(e["event"] == "PULL_DONE" for e in events(self.diag)))
        with self.assertRaisesRegex(ValueError, "lacks byte verification"):
            AUDIT.audit(events(self.diag))


class TestPDDiagnostics(unittest.TestCase):
    def test_config_defaults_and_invalid_modes(self):
        config = integration.config(producer=True)
        result = flow.CONFIG.PDConfig.from_vllm(config)
        self.assertEqual(result.diagnostics_requests, 0)
        self.assertFalse(result.verify_kv)
        config.kv_transfer_config.kv_connector_extra_config.update(diagnostics_requests=2, verify_kv=True)
        self.assertTrue(flow.CONFIG.PDConfig.from_vllm(config).verify_kv)
        for values in (
            dict(verify_kv=True),
            dict(diagnostics_requests=True),
            dict(diagnostics_requests=17),
            dict(diagnostics_requests=1, verify_kv="true"),
        ):
            config.kv_transfer_config.kv_connector_extra_config = dict(decode_endpoints=["host:31000"], **values)
            with self.subTest(values=values), self.assertRaises(ValueError):
                flow.CONFIG.PDConfig.from_vllm(config)

    def test_hashes_preserve_dtype_bytes_and_exclude_padding(self):
        diag = diagnostic("P")
        value = torch.arange(16, dtype=torch.int8)
        expected = diag.hashes(value, (1,), (5,))
        value[9] ^= 1
        self.assertEqual(diag.hashes(value, (1,), (5,)), expected)
        self.assertEqual(diag.hashes(value.view(torch.uint8), (1,), (5,)), expected)
        with self.assertRaises(ValueError):
            diag.hashes(value, (15,), (2,))

    def test_request_limit_and_duplicate_events(self):
        diag = diagnostic("D")
        self.assertTrue(diag.select("external", "g", alias="internal"))
        self.assertTrue(diag.active("internal"))
        self.assertFalse(diag.select("another", "g"))
        for _ in range(10):
            diag.emit("SFA_CALL", ("internal", "l0"), request="internal")
        self.assertEqual(len(events(diag)), 1)

    def test_independent_pa_formula_detects_a_planner_offset_error(self):
        memory = flow.TestPDFlow.memories_for(decode=False)["l0"]
        tables = memory.tables(((1, 6, 0), (4, 2, 7), (), ()))
        diag = diagnostic("P")
        expected = diag.spans(memory, tables, 300, 0, True, 0, 2)
        layout_module = importlib.import_module(f"{flow.PACKAGE}.pd.layout")
        layout = memory.describe("l0")
        original = layout_module.LayerLayout.offset

        def wrong(instance, *args):
            return original(instance, *args) + 1

        with patch.object(layout_module.LayerLayout, "offset", wrong):
            plan = layout_module.plan_layer(layout, layout, tables, tables, 300, member=0, ratio=2)
        actual = {
            "npu": tuple((b.sources, b.lengths) for b in plan.hbm.batches()),
            "cpu": ((plan.main_sources, (128 * 656,) * len(plan.full_blocks)),),
        }
        with self.assertRaisesRegex(RuntimeError, "actual Tensor PA/block formula"):
            diag.address_check(expected, actual, "job", layer="l0")

    def test_uncertain_diagnostic_get_retains_buffers_and_access_table(self):
        diag = diagnostic("D")
        runtime = Mock()
        runtime.batch_get.side_effect = RuntimeError("Get failed")
        transfer = NS(put=NS(entries=[NS(key=1, length=128 * 656)]))
        with self.assertRaisesRegex(RuntimeError, "Get failed"):
            diag.ddr_hashes(runtime, transfer, "cpu")
        self.assertIsNotNone(diag.retained)
        runtime.release_access_table.assert_not_called()

    def test_indexer_decorator_observes_actual_write_once(self):
        diag = diagnostic("P", verify=False)
        diag.select("r", "g")
        diag.context = ("r",)
        indexer = NS(
            write_cache=Mock(return_value=7),
            _use_c8_reshape_optim=lambda: False,
            k_cache=NS(kv_cache=[torch.empty(4, dtype=torch.float8_e4m3fn)]),
        )
        native = indexer.write_cache
        diag.decorate_indexer(indexer, "l0")
        for _ in range(3):
            self.assertEqual(indexer.write_cache(torch.empty(4, dtype=torch.float8_e4m3fn), torch.ones(1), "slots"), 7)
        self.assertEqual(native.call_count, 3)
        writes = [e for e in events(diag) if e["event"] == "INDEXER_WRITE"]
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0]["operator"], "scatter_cache")


if __name__ == "__main__":
    unittest.main()
