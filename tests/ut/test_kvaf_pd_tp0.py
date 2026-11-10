# SPDX-License-Identifier: Apache-2.0
"""Opt-in TP0 transport without changing cache storage or native model TP."""

import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import test_kvaf_pd_diagnostics as diagnostics
import test_kvaf_pd_flow as flow
import test_kvaf_pd_integration as integration
import test_kvaf_pd_mtp_flow as mtp
import test_kvaf_pd_recovery as recovery
import torch


class TestTP0Config(unittest.TestCase):
    @staticmethod
    def config(producer=True, tp=8):
        value = integration.config(producer=producer, tp=tp)
        value.kv_transfer_config.kv_connector_extra_config["source_tp0_only"] = True
        value.model_config.hf_config = NS(model_type="glm_moe_dsa")
        value.cache_config.cache_dtype = "fp8"
        value.attention_config = NS(indexer_kv_dtype="fp8")
        value.additional_config.update(enable_dsa_cp=False, c8_enable_reshape_optim=False)
        return value

    def test_default_off_preserves_existing_topology_and_flag_is_strict(self):
        self.assertFalse(flow.CONFIG.PDConfig.from_vllm(integration.config(producer=True, tp=8)).source_tp0_only)
        for flag in (1, 0, "true", None):
            value = self.config()
            value.kv_transfer_config.kv_connector_extra_config["source_tp0_only"] = flag
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "boolean"):
                flow.CONFIG.PDConfig.from_vllm(value)

    def test_mode_preserves_model_tp_and_restricts_verified_cache_layout(self):
        for producer, tp, dp in ((True, 8, 2), (False, 1, 16)):
            value = self.config(producer=producer, tp=tp)
            value.parallel_config.data_parallel_size = dp
            self.assertTrue(flow.CONFIG.PDConfig.from_vllm(value).source_tp0_only)
            self.assertEqual(value.parallel_config.tensor_parallel_size, tp)
        for change, expected in (
            (lambda c: setattr(c.model_config.hf_config, "model_type", "other"), "GLM"),
            (lambda c: c.additional_config.update(enable_dsa_cp=True), "enable_dsa_cp"),
            (lambda c: c.additional_config.update(c8_enable_reshape_optim=True), "reshape_optim"),
            (lambda c: setattr(c.cache_config, "cache_dtype", "auto"), "fp8"),
            (lambda c: setattr(c.attention_config, "indexer_kv_dtype", "int8"), "fp8"),
        ):
            value = self.config()
            change(value)
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                flow.CONFIG.PDConfig.from_vllm(value)
        with self.assertRaisesRegex(ValueError, "D TP=1"):
            flow.CONFIG.PDConfig.from_vllm(self.config(producer=False, tp=2))


class TestTP0Flow(unittest.TestCase):
    def setUp(self):
        self.f = flow.TestPDFlow()
        self.f.setUp(source_tp0_only=True, tp_size=8)
        self.addCleanup(self.f.doCleanups)

    def test_79_layer_model_uses_eight_source_pairs_without_combining_storage(self):
        sdk = flow.SDK()
        memories = {
            f"layer.{i}": flow.MEMORY.allocate_layer(1, 0, device="cpu", groups={"main": 0}, offload_main=True)
            for i in range(79)
        }
        endpoints = tuple(f"127.0.0.1:{31000 + i}" for i in range(8))
        mappings = []

        def call(endpoint, message, **kwargs):
            if message["op"] == "hello":
                return dict(
                    peer=endpoint,
                    source_tp0_only=True,
                    source_cleanup_v1=True,
                    pull_api_version=2,
                    ticket=dict(version=2, region_id=1, epoch=1),
                    receiver=dict(version=1, control_address=endpoint, receiver_id=1, receiver_epoch=1),
                )
            if message["op"] == "register":
                mappings.append((endpoint, message["metadata"]))
            return True

        sender = flow.SENDER.Sender(
            sdk,
            memories,
            config=flow.CONFIG.PDConfig(decode_endpoints=endpoints, source_tp0_only=True),
            device_id=0,
            producer="P:0:0",
            host="127.0.0.1",
            activate=lambda: None,
            stream_factory=lambda: None,
            event_factory=lambda: None,
            external_id=lambda req: req,
            tp_rank=0,
            tp_size=8,
            dp_rank=0,
        )
        with patch.object(flow.SENDER, "call", side_effect=call):
            try:
                sender.initialize()
                self.assertEqual(len(sdk.sources), 16)
                self.assertEqual(len(sender.source_groups), 8)
                self.assertEqual(len(mappings), 632)
                for group in sender.source_groups.values():
                    self.assertEqual(len(group.layers), 79)
                    self.assertEqual({view.metadata.mr_id for view in group.layers.values()}, set(range(1, 80)))
                    for view, memory in zip(group.layers.values(), memories.values()):
                        self.assertIs(view.tensor, memory.tensor)
            finally:
                sender.close()

    def test_only_tp0_allocates_sources_and_full_kv_arrives_across_chunks(self):
        f = self.f
        self.assertEqual(len(f.sdk.sources), 2)
        for sender in f.senders:
            self.assertEqual(sender.tp_size, 8)
            self.assertEqual(sender.active, sender.tp_rank == 0)
            if sender.tp_rank:
                self.assertIsNone(sender.bootstrap)
                self.assertIsNone(sender.executor)
                self.assertFalse(sender.registrations)
                self.assertFalse(sender.peers)
                sender.event_factory = Mock(side_effect=AssertionError("inactive rank created an event"))
                sender.activate = Mock(side_effect=AssertionError("inactive rank activated a transport"))
        f.seed(511)
        # A 3-block chunk followed by a non-aligned chunk tests ownership rotation.
        f.chunk(384, final=False)
        self.assertFalse(f.receiver.get_finished())
        f.chunk(511, previous=384, graph=True)
        self.assertEqual(f.receiver.get_finished(), {"r"})
        f.assert_payload(511)
        self.assertEqual(len(f.receiver.requests["r"]["final"]), len(f.memories))
        self.assertTrue(
            all(job.member == 0 and job.ratio == 1 and job.producer == "P:0:0" for job in f.receiver.jobs.values())
        )
        for sender in f.senders:
            self.assertEqual(sender.get_finished({"r"}), {"r"})
            self.assertFalse(sender.forward_pending)

    def test_inactive_ranks_do_not_load_sdk_register_or_connect(self):
        f = self.f
        for sender in f.senders[1:]:
            sender.close()
        with (
            patch.object(flow.SENDER, "ThreadPoolExecutor", side_effect=AssertionError("executor allocated")),
            patch.object(flow.SENDER, "call", side_effect=AssertionError("RPC sent")),
        ):
            for rank in range(1, 8):
                sender = flow.SENDER.Sender(
                    None,
                    {},
                    config=f.senders[0].config,
                    device_id=rank,
                    producer=f"P:9:{rank}",
                    host="127.0.0.1",
                    activate=Mock(),
                    stream_factory=Mock(),
                    event_factory=Mock(),
                    external_id=lambda value: value,
                    tp_rank=rank,
                    tp_size=8,
                    dp_rank=9,
                )
                sender.initialize()
                sender.close()
                sender.close()
                self.assertTrue(sender.closed)

    def test_cancellation_before_forward_reports_all_native_ranks_with_one_remote_vote(self):
        f = self.f
        f.seed(300)
        params = dict(
            remote_host="127.0.0.1", remote_port=f.receiver.server.port, remote_tp_size=1, kvaf_generation="a1"
        )
        for sender in f.senders[1:] + f.senders[:1]:
            sender.bind(NS(requests={}, generations={}, cancellations={"r": params}))
            self.assertEqual(sender.get_finished({"r"}), {"r"})
        self.assertEqual(f.receiver.get_finished(), {"r"})
        self.assertEqual(f.receiver.requests["r"]["cancel_votes"], {0})
        self.assertFalse(f.native.calls)

    def test_inactive_completion_waits_for_native_forward_and_never_contacts_d(self):
        f = self.f
        f.seed(129)
        sender = f.senders[7]
        metadata = mtp.TestPDMTPSkipFlow._metadata(f, sender, 129, 0, True)
        sender.bind(metadata)
        self.assertFalse(sender.get_finished({"r"}))
        with patch.object(flow.SENDER, "call", side_effect=AssertionError("RPC sent")):
            sender.complete_drafting()
            sender.complete_forward()
            self.assertEqual(sender.get_finished(set()), {"r"})
        self.assertFalse(f.receiver.get_finished())  # TP0 has not supplied any KV.

    def test_cancellation_after_partial_prefill_drains_before_freeing_native_blocks(self):
        f = self.f
        f.seed(300)
        f.chunk(128, final=False)
        f.writer.release_requests({"r"})
        f.receiver.release({"r"})
        self.assertIn("r", f.writer.requests)
        for sender in f.senders:
            self.assertEqual(sender.get_finished({"r"}), {"r"})
        self.assertEqual(f.receiver.get_finished(), {"r"})
        self.assertEqual(f.receiver.requests["r"]["cancel_votes"], {0})
        f.receiver.release({"r"})
        self.assertFalse(f.native.pages)
        self.assertFalse(f.pool.policy.receive_slots)
        self.assertFalse(f.receiver.requests)

    def test_d_rejects_wrong_source_mode_rank_and_old_cohort_before_attach(self):
        f = self.f
        sender = f.senders[0]
        message = dict(peer=f.receiver.identity, producer=sender.producer, op="detach")
        for fields, text in (
            ({}, "modes must match"),
            (dict(source_tp0_only=True, source_tp_rank=7), "P TP0 only"),
            (dict(source_tp0_only=True, source_tp_rank=True), "P TP0 only"),
            (dict(source_tp0_only=True, source_tp_rank=0, op="plan", member=0, ratio=8), "one transfer contributor"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, text):
                f.receiver.dispatch(dict(message, **fields))

    def test_mode_mismatch_rejected_before_source_creation_in_either_direction(self):
        f = self.f
        for p_mode, d_mode in ((True, False), (False, True)):
            hello = f.receiver.dispatch({"op": "hello"})
            hello["source_tp0_only"] = d_mode
            sdk = flow.SDK()
            with (
                patch.object(flow.SENDER, "call", return_value=hello),
                self.subTest(p=p_mode, d=d_mode),
                self.assertRaisesRegex(ValueError, "modes must match"),
            ):
                flow.SENDER.Sender(
                    sdk,
                    f.memories_for(decode=False),
                    config=replace(f.senders[0].config, source_tp0_only=p_mode),
                    device_id=0,
                    producer="P:9:0",
                    host="127.0.0.1",
                    activate=lambda: None,
                    stream_factory=lambda: None,
                    event_factory=lambda: None,
                    external_id=lambda value: value,
                    tp_rank=0,
                    tp_size=8,
                    dp_rank=9,
                ).initialize()
            self.assertFalse(sdk.sources)

    def test_diagnostics_verify_both_paths_without_claiming_cross_tp_replica_comparison(self):
        f = self.f
        diag = diagnostics.diagnostic("D")
        f.receiver.diagnostics = f.pool.policy.diagnostics = diag
        for sender in f.senders:
            sender.diagnostics = diagnostics.diagnostic("P")
        f.seed(300)
        f.chunk(300)
        f.assert_payload(300)
        records = diagnostics.events(diag)
        self.assertEqual(diagnostics.AUDIT.audit(records, require_ddr=True), (1, ["cpu", "npu"]))
        replicas = [record for record in records if record["event"] == "MAIN_REPLICA_CHECK"]
        self.assertTrue(replicas)
        self.assertTrue(all(record["member"] == 0 and record["ratio"] == 1 for record in replicas))


class TestTP0MTP(unittest.TestCase):
    def setUp(self):
        self.f = mtp.TestPDMTPSkipFlow()
        flow.TestPDFlow.setUp(self.f, source_tp0_only=True, tp_size=8)
        self.addCleanup(self.f.doCleanups)

    def test_real_graph_mtp_supplies_the_entire_cache_from_tp0(self):
        f = self.f
        f._limits(512)
        f.seed(129)
        f._chunk(129, graph=True)
        self.assertEqual(f.receiver.get_finished(), {"r"})
        flow.TestPDFlow.assert_payload(f, 129)
        self.assertEqual(len(f.receiver.requests["r"]["final"]), len(f.memories))
        self.assertEqual(len(f._mtp_sequences()), 1)
        self.assertTrue(f._mtp_sequences()[0][1])
        self.assertTrue(all(sender.draft_completed for sender in f.senders))

    def test_native_skip_never_sends_dummy_mtp_and_does_not_wait_for_other_tp_ranks(self):
        f = self.f
        f._limits(300)
        f.seed(300)
        f._chunk(300, skipped=True)
        self.assertEqual(f.receiver.get_finished(), {"r"})
        f._assert_target_payload(300)
        self.assertTrue(torch.all(f.memories["mtp"].tensor == -1))
        self.assertEqual(f._mtp_sequences(), ((0, 0),))
        for sender in f.senders:
            self.assertEqual(sender.get_finished({"r"}), {"r"})


class TestTP0Runner(unittest.TestCase):
    def test_native_storage_and_inactive_sdk_loading(self):
        fixture = integration.TestExternalRunner()
        fixture.setUpClass()
        for rank in (0, 7):
            runner, cache = fixture.runner(decode=False)
            runner.pd_config = flow.CONFIG.PDConfig(source_tp0_only=True)
            runner.is_kv_producer, runner.tp_rank = True, rank
            runner.ascend_config = NS(kvaf_config=flow.CONFIG.KVAFConfig(enabled=False, build_dir="/sdk"))
            runner.kvaf_adapter = None
            runner.effective_drafter_max_model_len = 4096
            raw = runner._allocate_kv_cache_tensors(cache)
            runner._reshape_kv_cache_tensors(cache, raw)
            before = {
                name: (memory.tensor.data_ptr(), memory.tensor.clone()) for name, memory in runner.pd_memories.items()
            }
            connector = Mock(diagnostics=None)
            with (
                patch.object(integration.NativeRunnerBoundary, "initialize_kv_cache", create=True),
                patch.object(fixture.runner_module, "get_kv_transfer_group", return_value=connector),
                patch.object(fixture.runner_module, "load_sdk") as load,
            ):
                runner.initialize_kv_cache(cache)
                if rank:
                    load.assert_not_called()
                    self.assertIsNone(connector.bind_pd_backend.call_args.kwargs["sdk"])
                else:
                    load.assert_called_once_with("/sdk")
            for name, memory in runner.pd_memories.items():
                self.assertEqual(memory.tensor.data_ptr(), before[name][0])
                self.assertTrue(torch.equal(memory.tensor, before[name][1]))


class TestTP0Recovery(unittest.TestCase):
    def test_lost_pull_reply_queries_original_ready_without_duplicate_dma(self):
        f = flow.TestPDFlow()
        self.addCleanup(f.doCleanups)
        lost = False
        calls = []

        def call(endpoint, message, **kwargs):
            nonlocal lost
            calls.append(message["op"])
            result = f.receiver.dispatch(message)
            if message["op"] == "pull" and not lost:
                lost = True
                raise ConnectionResetError("reply lost after Pull")
            return result

        with (
            patch.object(flow.RECEIVER, "ControlServer", recovery._DirectServer),
            patch.object(flow.SENDER, "call", call),
        ):
            try:
                f.setUp(source_tp0_only=True, tp_size=8)
                f.seed(300)
                f.chunk(300)
                f.assert_payload(300)
                self.assertEqual(calls.count("status"), 1)
                self.assertEqual(calls.count("pull"), sum(len(job.completed) for job in f.receiver.jobs.values()))
                self.assertTrue(all(source.active is None for source in f.sdk.sources.values()))
                for sender in f.senders:
                    self.assertEqual(sender.get_finished({"r"}), {"r"})
            finally:
                self.assertTrue(f.doCleanups())


if __name__ == "__main__":
    unittest.main()
