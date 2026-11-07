# SPDX-License-Identifier: Apache-2.0
"""Real CPU arena views and end-to-end byte checks through the existing SDK mocks."""

import importlib
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import test_kvaf_pd_flow as flow
import test_kvaf_pd_integration as integration
import torch

MEMORY = flow.MEMORY
REGISTRATION = importlib.import_module(f"{flow.PACKAGE}.pd.registration")
DIAGNOSTICS = importlib.import_module(f"{flow.PACKAGE}.pd.diagnostics")


class TestArenaAllocation(unittest.TestCase):
    def plans(self):
        return {
            name: (3, 2, {"main": 0, "indexer": 1, "scale": 1}, name != "mtp") for name in ("m0", "m1", "m2", "mtp")
        }

    def allocate(self, layers):
        allocator = Mock(side_effect=lambda size: torch.empty(size, dtype=torch.int8))
        memories = MEMORY.allocate_layers(
            self.plans(), device="cpu", allocator=allocator, arena_layers=layers, alignment=4096
        )
        return memories, allocator

    def test_single_layer_default_preserves_independent_allocations(self):
        memories, allocator = self.allocate(1)
        self.assertEqual(allocator.call_count, 4)
        self.assertTrue(all(memory.arena is None for memory in memories.values()))
        self.assertEqual(len({memory.tensor.untyped_storage().data_ptr() for memory in memories.values()}), 4)

    def test_group_is_real_contiguous_storage_with_disjoint_aligned_layer_views(self):
        memories, allocator = self.allocate(16)
        self.assertEqual(allocator.call_count, 2)  # Target and MTP never share a Ready owner.
        arena = memories["m0"].arena
        arena.fill_(11)
        ranges = []
        for name in ("m0", "m1", "m2"):
            memory = memories[name]
            layout = memory.describe(name)
            self.assertIs(memory.arena, arena)
            self.assertEqual(memory.tensor.untyped_storage().data_ptr(), arena.untyped_storage().data_ptr())
            self.assertEqual((layout.pa_base - arena.data_ptr()) % 4096, 0)
            ranges.append((layout.pa_base, layout.pa_base + layout.nbytes))
        for left, right in zip(ranges, ranges[1:]):
            self.assertLessEqual(left[1], right[0])
        memories["m1"].planes["scale"].fill_(7)
        self.assertTrue(torch.all(memories["m0"].tensor == 11))
        self.assertTrue(torch.all(memories["m2"].tensor == 11))
        self.assertIsNot(memories["mtp"].arena, arena)

    def test_independent_storage_cannot_be_forged_into_an_arena(self):
        memories, _ = self.allocate(16)
        memory = memories["m0"]
        memory.arena = torch.empty_like(memory.arena)
        with self.assertRaisesRegex(ValueError, "actual view"):
            memory.describe("m0")

    def test_config_is_explicit_and_p_only(self):
        config = integration.config(producer=True)
        self.assertEqual(flow.CONFIG.PDConfig.from_vllm(config).source_arena_layers, 1)
        for size in (0, -1, True, 1.5, 129):
            config.kv_transfer_config.kv_connector_extra_config["source_arena_layers"] = size
            with self.subTest(size=size), self.assertRaises(ValueError):
                flow.CONFIG.PDConfig.from_vllm(config)
        config.kv_transfer_config.kv_connector_extra_config["source_arena_layers"] = 16
        self.assertEqual(flow.CONFIG.PDConfig.from_vllm(config).source_arena_layers, 16)
        config = integration.config()
        config.kv_transfer_config.kv_connector_extra_config["source_arena_layers"] = 16
        with self.assertRaisesRegex(ValueError, "P producers only"):
            flow.CONFIG.PDConfig.from_vllm(config)

    def test_external_runner_binds_native_layer_views_without_copying(self):
        integration.TestExternalRunner.setUpClass()
        fixture = integration.TestExternalRunner()
        runner, cache = fixture.runner(decode=False)
        runner.pd_config = replace(runner.pd_config, source_arena_layers=16)
        raw = runner._allocate_kv_cache_tensors(cache)
        runner._reshape_kv_cache_tensors(cache, raw)
        first = runner.pd_memories["m"]
        self.assertIs(first.arena, runner.pd_memories["m2"].arena)
        self.assertIsNot(first.arena, runner.pd_memories["draft"].arena)
        self.assertEqual(raw["m"][0].data_ptr(), first.planes["main"].data_ptr())
        self.assertEqual(raw["idx"][0].data_ptr(), first.planes["indexer"].data_ptr())


class TestArenaRegistration(unittest.TestCase):
    def setUp(self):
        memories, _ = TestArenaAllocation().allocate(16)
        self.memories = memories
        self.sdk = flow.SDK()
        self.owner = REGISTRATION.ArenaRegistration(
            self.sdk, "d", "m0", memories["m0"], device_id=0, cpu_eid="", npu_eid=""
        )
        self.owner.register(NS(region_id=1), None, listen_address="127.0.0.1:0", advertised_address="127.0.0.1:0")
        self.layers = {name: self.owner.layer(name, memories[name]) for name in ("m0", "m1", "m2")}

    def test_shared_sources_have_distinct_layer_layouts_and_source_bias(self):
        self.assertEqual(len(self.sdk.sources), 2)
        first = self.layers["m0"].metadata
        for name, layer in self.layers.items():
            self.assertEqual(layer.metadata.binding_key, first.binding_key)
            self.assertEqual(
                layer.metadata.source_bias, self.memories[name].tensor.data_ptr() - self.memories[name].arena.data_ptr()
            )
        invalid = replace(first, registered_range=(first.layout.pa_base + 1, first.layout.nbytes))
        with self.assertRaises(ValueError):
            _ = invalid.source_bias

    def test_one_ready_at_a_time_and_wrong_layer_ack_is_rejected(self):
        first, second = self.layers["m0"], self.layers["m1"]
        event = NS(synchronize=lambda: None)
        ready = first.ready("cpu", event)
        with self.assertRaisesRegex(RuntimeError, "active batch"):
            second.ready("cpu", event)
        with self.assertRaisesRegex(ValueError, "another layer"):
            second.release("cpu", ready)
        with self.assertRaisesRegex(RuntimeError, "active layer batches"):
            first.close()
        with self.assertRaises(AssertionError):  # Simulate missing SDK completion proof.
            first.release("cpu", ready)
        self.assertIn("cpu", self.owner.active_layers)
        self.sdk.acknowledged.add((ready.source.source_id, ready.sequence))
        first.release("cpu", ready)
        next_ready = second.ready("cpu", event)
        self.assertEqual(next_ready.sequence, ready.sequence + 1)
        self.sdk.acknowledged.add((next_ready.source.source_id, next_ready.sequence))
        second.release("cpu", next_ready)

    def test_last_layer_reference_closes_source_once_and_failure_retains_reference(self):
        with patch.object(self.owner.registration, "close") as close:
            self.layers["m0"].close()
            self.layers["m1"].close()
            close.assert_not_called()
            close.side_effect = RuntimeError("detach unconfirmed")
            with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
                self.layers["m2"].close()
            self.assertEqual(self.owner.members, {"m2"})
            close.side_effect = None
            self.layers["m2"].close()
            self.layers["m2"].close()
            self.assertEqual(close.call_count, 2)  # One failed attempt, one successful close.


class TestArenaFlow(unittest.TestCase):
    def fixture(self, layers=4):
        fixture = flow.TestPDFlow()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp(source_arena_layers=layers)
        return fixture

    def test_two_paths_nonzero_arena_offsets_mtp_and_tail_match_bytes(self):
        f = self.fixture()
        self.assertEqual(len(f.receiver.sources), 20)  # Five logical layers x four P ranks.
        self.assertEqual(len(f.receiver.bindings), 8)  # Two real arena sources x four P ranks.
        self.assertEqual(len(f.sdk.sources), 16)  # CPU/NPU source pair for each arena/peer.
        f.seed(300)
        f.chunk(300)
        self.assertEqual(f.receiver.get_finished(), {"r"})
        f.assert_payload(300)
        for sender in f.senders:
            self.assertIs(sender.futures["l0"], sender.futures["l3"])
            self.assertTrue(all(not owner.active_layers for owner in sender.arenas.values()))

    def test_chunks_and_graph_fallback_keep_layer_directories_independent(self):
        f = self.fixture(2)
        f.seed(300)
        f.chunk(256, final=False, graph=True)
        self.assertEqual(f.receiver.get_finished(), set())
        f.chunk(300, previous=256, graph=True)
        self.assertEqual(f.receiver.get_finished(), {"r"})
        f.assert_payload(300)

    def test_no_target_publish_before_group_written_and_no_early_mtp_publish(self):
        f = self.fixture()
        sender = f.senders[0]
        sender.metadata = NS(requests={"r": NS()}, generations={"r": "a1"})
        with patch.object(sender, "_dispatch_arena") as dispatch:
            for name in ("l0", "l1", "l2"):
                sender.written(name)
            dispatch.assert_not_called()
            sender.written("l3")
            dispatch.assert_called_once()
            dispatch.reset_mock()
            sender.written("mtp")
            sender.written("mtp")
            sender.finalize()
            dispatch.assert_not_called()
        sender.metadata = None

    def test_arena_group_stops_after_failed_layer_without_replacing_ready(self):
        f = self.fixture()
        sender = f.senders[0]
        with (
            patch.object(sender, "_send_layer", side_effect=RuntimeError("ACK uncertain")) as send,
            self.assertRaisesRegex(RuntimeError, "ACK uncertain"),
        ):
            sender._send_arena(("l0", "l1", "l2"), (), Mock(), False)
        self.assertEqual(send.call_count, 1)

    def test_diagnostics_hash_the_registered_arena_with_rebased_offsets(self):
        f = self.fixture()
        diag = DIAGNOSTICS.Diagnostics(2, True, "D")
        diag.logger = Mock()
        f.receiver.diagnostics = diag
        for sender in f.senders:
            sender.diagnostics = DIAGNOSTICS.Diagnostics(2, True, "P")
            sender.diagnostics.logger = Mock()
        f.seed(300)
        f.chunk(300)
        self.assertEqual(f.receiver.get_finished(), {"r"})
        f.assert_payload(300)
        self.assertIsNone(diag.retained)
        self.assertTrue(any(call.args[1].find('"event": "KV_CHECK"') >= 0 for call in diag.logger.info.call_args_list))
        self.assertFalse(any('"status": "FAIL"' in call.args[1] for call in diag.logger.info.call_args_list))

    def test_lost_ack_queries_original_ready_without_replaying_cpu_pull(self):
        f = self.fixture()
        sender = f.senders[0]
        original = sender._call
        lost = False

        def call(endpoint, message):
            nonlocal lost
            result = original(endpoint, message)
            if message["op"] == "pull" and message["path"] == "cpu" and not lost:
                lost = True
                raise TimeoutError("lost original ACK")
            return result

        f.seed(300)
        with patch.object(sender, "_call", side_effect=call):
            f.chunk(300)
        self.assertTrue(lost)
        self.assertEqual(f.receiver.get_finished(), {"r"})
        self.assertEqual(f.native.calls.count("submit"), 4)
        f.assert_payload(300)

    def test_detach_is_once_per_arena_not_per_layer(self):
        f = self.fixture()
        with patch.object(f.native, "detach_hbm", wraps=f.native.detach_hbm) as detach:
            for sender in f.senders:
                sender.close()
            self.assertEqual(detach.call_count, 8)
        self.assertEqual(f.receiver.sources, {})
        self.assertEqual(f.receiver.bindings, {})

    def test_mtp_skip_does_not_publish_dummy_writes_from_its_arena(self):
        f = self.fixture()
        f.receiver.draft_limit = 300
        f.seed(300)
        patches = []
        for sender in f.senders:
            sender.draft_limit = 300
            written = sender.written

            def skip(layer, sender=sender, written=written):
                if layer == "mtp":
                    sender.skip_drafting(300)
                else:
                    written(layer)

            patches.append(patch.object(sender, "written", side_effect=skip))
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        f.chunk(300)
        self.assertEqual(f.receiver.get_finished(), {"r"})
        self.assertTrue(torch.all(f.memories["mtp"].tensor == -1))

    def test_old_d_is_rejected_before_creating_any_arena_source(self):
        f = self.fixture()
        original = flow.SENDER.call
        previous_count = len(f.sdk.sources)

        def old_d(*args, **kwargs):
            result = original(*args, **kwargs)
            if args[1]["op"] == "hello":
                result.pop("source_arena_v1")
            return result

        with (
            patch.object(flow.SENDER, "call", side_effect=old_d),
            self.assertRaisesRegex(ValueError, "D does not support source arenas"),
        ):
            flow.SENDER.Sender(
                f.sdk,
                flow.TestPDFlow.memories_for(decode=False, arena_layers=4),
                config=flow.CONFIG.PDConfig(decode_endpoints=(f.endpoint,), source_arena_layers=4),
                device_id=0,
                producer="another:0:0",
                host="127.0.0.1",
                activate=lambda: None,
                stream_factory=lambda: None,
                event_factory=lambda: NS(synchronize=lambda: None),
                external_id=lambda value: value,
                tp_rank=0,
                tp_size=4,
                dp_rank=0,
            )
        self.assertEqual(len(f.sdk.sources), previous_count)

    def test_partial_attach_keeps_descriptors_until_successful_detach(self):
        f = self.fixture()
        memories = flow.TestPDFlow.memories_for(decode=False, arena_layers=4)
        owner = REGISTRATION.ArenaRegistration(
            f.sdk, f.receiver.identity, "l0", memories["l0"], device_id=0, cpu_eid="", npu_eid=""
        )
        owner.register(f.native.offload_region(), None, listen_address="127.0.0.1:0", advertised_address="127.0.0.1:0")
        self.addCleanup(owner.registration.close)
        layer = owner.layer("l0", memories["l0"])
        message = dict(
            op="register", peer=f.receiver.identity, producer="another:0:0", metadata=layer.metadata, offload=True
        )
        with (
            patch.object(f.native, "attach_hbm", side_effect=RuntimeError("attach uncertain")),
            self.assertRaisesRegex(RuntimeError, "attach uncertain"),
        ):
            f.receiver.dispatch(message)
        binding = (message["producer"], layer.metadata.binding_key)
        self.assertIn(binding, f.receiver.bindings)
        self.assertIn(binding, f.receiver.pending_bindings)
        with self.assertRaisesRegex(RuntimeError, "original incomplete"):
            f.receiver.dispatch(message)
        with (
            patch.object(f.native, "detach_hbm", side_effect=RuntimeError("detach uncertain")),
            self.assertRaisesRegex(RuntimeError, "detach uncertain"),
        ):
            f.receiver._detach_sources(message["producer"])
        self.assertIn(binding, f.receiver.bindings)
        f.receiver._detach_sources(message["producer"])
        self.assertNotIn(binding, f.receiver.bindings)
        self.assertNotIn(binding, f.receiver.pending_bindings)
        layer.close()


if __name__ == "__main__":
    unittest.main()
