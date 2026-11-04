# SPDX-License-Identifier: Apache-2.0
"""Byte-level P/4 x D/1 flow over real TCP, with SDK/NPU boundaries substituted."""

import importlib
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import test_kvaf_adapter as helpers
import torch
from test_kvaf_grouped_get import Runtime as DecodeRuntime
from test_kvaf_pd import CpuState
from test_kvaf_pd import Runtime as PullRuntime
from test_kvaf_prefill import Manager

PACKAGE = helpers.PACKAGE.__name__
MEMORY = importlib.import_module(f"{PACKAGE}.pd.memory")
BACKEND = importlib.import_module(f"{PACKAGE}.pd.backend")
RECEIVER = importlib.import_module(f"{PACKAGE}.pd.receiver")
SENDER = importlib.import_module(f"{PACKAGE}.pd.sender")
RUNTIME = importlib.import_module(f"{PACKAGE}.pd.runtime")
TRANSFER = importlib.import_module(f"{PACKAGE}.pd.transfer")
CONFIG = importlib.import_module(f"{PACKAGE}.pd.config")
CONTROL = importlib.import_module(f"{PACKAGE}.pd.control")


class Source:
    def __init__(self, sdk, *, npu=False):
        self.sdk, self.active, self.sequence = sdk, None, 0
        self.npu = npu
        self.source_id = len(sdk.sources) + 1
        sdk.sources[self.source_id] = self

    def register(self, *args):
        self.tensor = args[-2]
        self.descriptor = NS(
            source_id=self.source_id, source_bytes=self.tensor.numel(), region_id=1, region_epoch=1, source_epoch=1
        )
        if self.npu:
            return dict(source_id=self.source_id, epoch=1, bytes=self.tensor.numel(), version=1, token=1)
        self.ticket = args[0]
        self.descriptor.region_id = self.ticket.region_id
        return self.descriptor

    def listen(self, address):
        return 19000 + self.source_id

    def ready(self):
        assert self.active is None
        self.sequence += 1
        self.active = NS(source=self.descriptor, sequence=self.sequence)
        return self.active

    def release_batch(self, ready):
        assert (self.source_id, ready.sequence) in self.sdk.acknowledged
        assert self.active.sequence == ready.sequence
        self.active = None

    def shutdown(self):
        assert self.active is None


class SDK:
    def __init__(self):
        self.sources, self.acknowledged = {}, set()

    def CpuPullHbmSource(self, **kwargs):
        return Source(self)

    def NpuPullHbmSource(self, **kwargs):
        return Source(self, npu=True)

    @staticmethod
    def NpuRuntime(**kwargs):
        return NS(initialize=lambda: None, shutdown=lambda: None)


class Runtime(PullRuntime, DecodeRuntime):
    def __init__(self, sdk):
        PullRuntime.__init__(self, None)
        DecodeRuntime.__init__(self, 1)
        self.sdk = sdk
        self.fail_dma = False

    def offload_region(self):
        return NS(region_id=1, region_epoch=1)

    def attach_cpu_source(self, descriptor):
        assert descriptor.source_id in self.sdk.sources

    def attach_hbm(self, address, ticket, **kwargs):
        assert ticket["source_id"] in self.sdk.sources

    def detach_hbm(self, ticket):
        assert ticket["source_id"] in self.sdk.sources

    def prepare_put(self, keys, lengths):
        result = super().prepare_put(keys, lengths)
        for entry in result.entries:
            entry.address = (entry.key + 1) * 128 * 656
        return result

    def submit_cpu_pull(self, ready, *args):
        self.source = self.sdk.sources[ready.source.source_id].tensor.view(torch.uint8)
        return super().submit_cpu_pull(ready, *args)

    def commit_put(self, put):
        result = super().commit_put(put)
        if result == CpuState.COMMITTED:
            self.sdk.acknowledged.add((self.last_ready.source.source_id, self.last_ready.sequence))
        return result

    def wait_cpu_pull(self, put, timeout):
        if self.fail_dma:
            put.state = CpuState.FAILED
            return put.state
        return super().wait_cpu_pull(put, timeout)

    def abort_put(self, put):
        result = super().abort_put(put)
        self.sdk.acknowledged.add((self.last_ready.source.source_id, self.last_ready.sequence))
        for entry in put.entries:
            self.pages.pop(entry.key, None)
        return result

    def batch_get_raw_addr(self, ready, *args):
        self.source = self.sdk.sources[ready.source.source_id].tensor
        result = super().batch_get_raw_addr(ready, *args)
        self.sdk.acknowledged.add((ready.source.source_id, ready.sequence))
        return result


class RawTransfer(TRANSFER.NpuPullTransfer):
    def _make_arrays(self):
        return tuple(
            torch.tensor(values)
            for values in (self.segments.sources, self.segments.destinations, self.segments.lengths)
        )


class TestPDFlow(unittest.TestCase):
    def setUp(self):
        self.sdk = SDK()
        self.pool = BACKEND.create_decode_cache(
            ("l0", "l1", "l2", "l3"), 2, 128, 8192, "cpu", native=lambda *a: a[14].zero_()
        )
        self.native = Runtime(self.sdk)
        self.runtime = RUNTIME.SerializedRuntime(self.native)
        self.layouts = tuple(
            replace(
                helpers.layout(656, name),
                planes=(replace(helpers.layout(656, name).planes[0], shape=tuple(self.pool.policy.hot[i].shape)),),
            )
            for i, name in enumerate(self.pool.names)
        )
        self.writer = RUNTIME.PDTransfer(
            {(656, 128): self.runtime},
            {(656, 128): 256},
            self.layouts,
            dict(zip(self.pool.names, self.pool.policy.hot)),
        )
        self.manager = Manager(self.writer)
        self.writer.feedback = self.manager
        self.manager.bind_device_pool(self.pool)
        self.addCleanup(self.manager.close)
        self.memories = self.memories_for(decode=True)
        self.receiver = RECEIVER.Receiver(
            self.runtime,
            self.writer,
            self.pool,
            self.memories,
            host="127.0.0.1",
            port=0,
            timeout=3,
            raw_transfer=RawTransfer,
        )
        self.addCleanup(self.receiver.close)
        endpoint = f"127.0.0.1:{self.receiver.server.port}"
        self.senders = []
        for rank in range(4):
            sender = SENDER.Sender(
                self.sdk,
                self.memories_for(decode=False),
                config=CONFIG.PDConfig(decode_endpoints=(endpoint,), timeout=3),
                device_id=rank,
                producer=f"P:0:{rank}",
                host="127.0.0.1",
                activate=lambda: None,
                stream_factory=lambda: None,
                event_factory=lambda: NS(synchronize=lambda: None),
                external_id=lambda value: value,
                tp_rank=rank,
                tp_size=4,
                dp_rank=0,
            )
            self.senders.append(sender)
            self.addCleanup(sender.close)
        self.endpoint = endpoint
        self.blocks = ((100, 101, 102, 103), (5, 1, 0, 6), (2, 0, 5, 6), (7, 3, 1, 2))

    @staticmethod
    def memories_for(*, decode):
        result = {}
        for name in ("l0", "l1", "l2", "l3", "mtp"):
            indexer = name in ("l0", "l3", "mtp")
            groups = {"main": 2 if name == "mtp" else 0}
            if indexer:
                groups.update(indexer=3 if name == "mtp" else 1, scale=3 if name == "mtp" else 1)
            result[name] = MEMORY.allocate_layer(
                2 if decode and name != "mtp" else 8,
                8 if indexer else 0,
                device="cpu",
                groups=groups,
                offload_main=name != "mtp",
            )
            result[name].tensor.fill_(-1)
        return result

    @staticmethod
    def value(layer, plane, token):
        return (layer * 17 + plane * 31 + token) % 113

    def prefill(self, sender, tokens, blocks):
        for i, memory in enumerate(sender.memories.values()):
            for plane, (name, tensor) in enumerate(memory.planes.items()):
                table = blocks[memory.groups[name]]
                for token in range(tokens):
                    tensor[table[token // 128], token % 128].fill_(self.value(i, plane, token))

    def seed(self, tokens, req_id="r", generation="a1"):
        self.receiver.seed(req_id, req_id, NS(generation=generation, tokens=tokens, blocks=self.blocks))

    def chunk(self, tokens, *, previous=0, final=True, req_id="r", generation="a1", participants=None, graph=False):
        for rank, sender in enumerate(self.senders if participants is None else participants):
            rank %= 4  # Physical block permutations are independent of TP membership.
            blocks = ((rank + 1, 6, 0, 5), (4, rank, 7, 6), (3, 0, 6, 1), (5, rank + 1, 0, 7))
            self.prefill(sender, tokens, blocks)
            metadata = NS(
                requests={
                    req_id: NS(
                        remote_tp_size=1,
                        remote_host="127.0.0.1",
                        remote_port=self.receiver.server.port,
                        local_block_ids=blocks,
                        local_transed_tokens=previous,
                        local_computed_tokens=tokens,
                        prompt_len=self.receiver.requests[req_id]["seed"].tokens,
                        chunk_finish=final,
                    )
                },
                generations={req_id: generation},
            )
            sender.bind(metadata)
            for layer in sender.memories:
                if graph and sender.memories[layer].offload_main:
                    continue
                sender.written(layer)
                sender.written(layer)  # Native early-write and layer-end hooks are idempotent.
            sender.finalize()
            sender.complete_forward()
        for sender in self.senders if participants is None else participants:
            for layer in sender.memories:
                sender.wait_layer(layer)

    def assert_payload(self, tokens):
        full, tail = divmod(tokens, 128)
        for i, (name, memory) in enumerate(self.memories.items()):
            if name != "mtp":
                state = self.writer.requests["r"][self.pool.indices[name]]
                self.assertEqual(len(state.keys), full)
                for q, key in enumerate(state.keys):
                    page = self.native.pages[key].view(128, 656)
                    for token in range(128):
                        self.assertTrue(torch.all(page[token] == self.value(i, 0, q * 128 + token)))
                received = memory.planes["main"][self.pool.policy.receive_slots["r"]]
                if tail:
                    for token in range(tail):
                        self.assertTrue(torch.all(received[token] == self.value(i, 0, full * 128 + token)))
                self.assertTrue(torch.all(received[tail:] == -1))
            for plane, (plane_name, tensor) in enumerate(memory.planes.items()):
                if plane_name == "main" and name != "mtp":
                    continue
                table = self.blocks[memory.groups[plane_name]]
                for token in range(tokens):
                    self.assertTrue(torch.all(tensor[table[token // 128], token % 128] == self.value(i, plane, token)))

    def test_four_layer_mtp_two_paths_match_bytes_and_continue_native_decode(self):
        self.seed(300)
        self.chunk(300)
        self.assertEqual(self.receiver.get_finished(), {"r"})
        self.assert_payload(300)
        self.assertEqual(len(self.native.pages), 4 * 2)
        self.assertEqual(self.native.calls.count("submit"), 4)
        self.assertEqual(len(self.receiver.sources), 4 * 5)
        self.assertEqual(self.pool.policy.receive_slot_for("r", "l0", 102), 0)
        self.assertTrue(all(source.active is None for source in self.sdk.sources.values()))
        for sender in self.senders:
            self.assertEqual(sender.get_finished({"r"}), {"r"})
        self.writer.prepare_requests(
            {"r": NS(num_computed_tokens=300, num_prompt_tokens=300, block_ids=self.blocks)}, {"r": 4}
        )
        self.manager.prepare(("r",), {"r": 4}, decode=True, padded_requests=1)
        self.assertTrue(self.native.reads)
        owner = self.pool.policy.owners["r"]
        for i, hot in enumerate(self.pool.policy.hot):
            start = owner * self.pool.policy.slot_capacity + self.pool.policy.history_capacity + 256
            for token in range(44):
                self.assertTrue(torch.all(hot.view(-1, 656)[start + token] == self.value(i, 0, 256 + token)))

    def test_partial_chunks_and_empty_contributors_do_not_release_early(self):
        self.seed(129)
        self.chunk(65, final=False)
        self.assertFalse(self.receiver.get_finished())
        self.assertFalse(self.native.pages)
        self.chunk(128, previous=65, final=False)
        self.assertFalse(self.receiver.get_finished())
        self.chunk(129, previous=128)
        self.assertEqual(self.receiver.get_finished(), {"r"})
        self.assert_payload(129)

    def test_full_block_prompt_has_no_main_hbm_tail(self):
        self.seed(256)
        self.chunk(256)
        self.assertEqual(self.receiver.get_finished(), {"r"})
        self.assert_payload(256)
        self.assertFalse(self.pool.policy._received_tails)

    def test_target_graph_replay_without_python_layer_hooks_uses_successful_forward_fence(self):
        self.seed(129)
        self.chunk(129, graph=True)
        self.assertEqual(self.receiver.get_finished(), {"r"})
        self.assert_payload(129)
        self.assertFalse(any(sender.forward_pending for sender in self.senders))

    def test_first_decode_waits_for_all_tp_members(self):
        self.seed(1)
        self.chunk(1, participants=self.senders[:3])
        self.assertFalse(self.receiver.get_finished())
        self.chunk(1, participants=self.senders[3:])
        self.assertEqual(self.receiver.get_finished(), {"r"})

    def test_commit_ack_loss_reuses_original_transaction(self):
        self.seed(129)
        self.native.commit_lost = True
        self.chunk(129)
        self.assertEqual(self.receiver.get_finished(), {"r"})
        self.assertEqual(self.native.calls.count("prepare"), 4)
        self.assertEqual(self.native.calls.count("submit"), 4)
        self.assertEqual(self.native.calls.count("commit"), 5)

    def test_stale_generation_cannot_replace_live_receive(self):
        self.seed(129)
        with self.assertRaisesRegex(ValueError, "generation"):
            self.receiver._request("r", "dead")
        self.assertEqual(self.pool.policy.receive_slots, {"r": 0})

    def test_release_frees_native_directory_and_compact_receiver(self):
        self.seed(129)
        self.chunk(129)
        self.receiver.release({"r"})
        self.assertFalse(self.native.pages)
        self.assertFalse(self.pool.policy.receive_slots)
        self.assertFalse(self.receiver.requests)
        self.assertFalse(self.receiver.jobs)

    def test_terminal_dma_failure_uses_native_load_errors_after_safe_source_release(self):
        self.seed(129)
        self.native.fail_dma = True
        self.chunk(129)
        self.assertEqual(self.receiver.get_finished(), {"r"})
        self.assertTrue(self.receiver.errors)
        self.assertIsNone(self.runtime.failure)
        self.assertFalse(self.native.pages)
        self.assertTrue(all(source.active is None for source in self.sdk.sources.values()))
        self.assertFalse(self.pool.policy._received_tails)
        self.receiver.release({"r"})
        self.assertFalse(self.writer.slots[656, 128].references)

    def test_native_cancel_holds_destinations_until_all_source_ranks_drain(self):
        self.seed(300)
        self.chunk(128, final=False)
        self.writer.release_requests({"r"})
        self.receiver.release({"r"})
        self.assertIn("r", self.writer.requests)
        self.assertIn("r", self.pool.policy.receive_slots)
        for sender in self.senders[:-1]:
            self.assertEqual(sender.get_finished({"r"}), {"r"})
            self.assertFalse(self.receiver.get_finished())
            self.assertIn("r", self.writer.requests)
        self.assertEqual(self.senders[-1].get_finished({"r"}), {"r"})
        self.assertEqual(self.receiver.get_finished(), {"r"})
        self.receiver.release(set())
        self.assertFalse(self.native.pages)
        self.assertFalse(self.pool.policy.receive_slots)
        self.assertFalse(self.receiver.requests)

    def test_cancellation_before_first_forward_still_releases_remote_allocation(self):
        self.seed(129)
        self.writer.release_requests({"r"})
        for sender in self.senders:
            sender.bind(
                NS(
                    requests={},
                    generations={},
                    cancellations={
                        "r": dict(
                            kvaf_generation="a1",
                            remote_tp_size=1,
                            remote_host="127.0.0.1",
                            remote_port=self.receiver.server.port,
                        )
                    },
                )
            )
            self.assertEqual(sender.get_finished({"r"}), {"r"})
        self.assertEqual(self.receiver.get_finished(), {"r"})
        self.receiver.release({"r"})
        self.assertFalse(self.pool.policy.receive_slots)

    def test_production_tp8_to_tp1_uses_all_eight_participants(self):
        for rank in range(4, 8):
            sender = SENDER.Sender(
                self.sdk,
                self.memories_for(decode=False),
                config=self.senders[0].config,
                device_id=rank,
                producer=f"P:0:{rank}",
                host="127.0.0.1",
                activate=lambda: None,
                stream_factory=lambda: None,
                event_factory=lambda: NS(synchronize=lambda: None),
                external_id=lambda value: value,
                tp_rank=rank,
                tp_size=8,
                dp_rank=0,
            )
            self.senders.append(sender)
            self.addCleanup(sender.close)
        for sender in self.senders:
            sender.tp_size = 8
        self.seed(511)
        self.chunk(511)
        self.assertEqual(self.receiver.get_finished(), {"r"})
        self.assertEqual(len(self.receiver.requests["r"]["final"]), 8 * 5)
        self.assert_payload(511)

    def test_pull_waits_for_compute_fence_before_prepare(self):
        self.runtime.compute_done = Mock()
        original = self.native.prepare_put

        def prepare(*args):
            self.runtime.compute_done.synchronize.assert_called()
            return original(*args)

        self.native.prepare_put = prepare
        self.seed(129)
        self.chunk(129)
        self.assertEqual(self.receiver.get_finished(), {"r"})
        self.assertTrue(all(not job.transfers and job.plan is None for job in self.receiver.jobs.values()))

    def test_backend_handoff_failure_reports_native_error_without_decode(self):
        self.seed(129)
        self.pool.policy.bind_received_tail = Mock(side_effect=RuntimeError("tail handoff rejected"))
        self.chunk(129)
        self.assertEqual(self.receiver.get_finished(), {"r"})
        self.assertTrue(self.receiver.get_errors())
        self.assertFalse(self.native.reads)
        self.assertTrue(all(source.active is None for source in self.sdk.sources.values()))

    def test_control_listener_can_wait_until_native_warmup_finishes(self):
        self.runtime.compute_done = Mock()
        receiver = RECEIVER.Receiver(
            self.runtime,
            self.writer,
            self.pool,
            self.memories,
            host="127.0.0.1",
            port=0,
            timeout=3,
            start_immediately=False,
        )
        self.addCleanup(receiver.close)
        self.assertIsNone(receiver.server)
        receiver.start()
        self.runtime.compute_done.synchronize.assert_called_once()
        self.assertGreater(receiver.server.port, 0)


class TestProductionMesh(unittest.TestCase):
    def test_control_codec_rejects_unregistered_python_globals(self):
        server = CONTROL.ControlServer("127.0.0.1", 0, lambda message: True, timeout=3)
        self.addCleanup(server.close)
        with self.assertRaisesRegex(RuntimeError, "unexpected PD metadata type"):
            CONTROL.call(f"127.0.0.1:{server.port}", {"unexpected": Path("/tmp")}, timeout=3)

    def test_tp8_dp2_sources_register_with_all_tp1_dp16_regions_on_eight_k_pools(self):
        sdk = SDK()
        endpoints, imported = [], []

        def service(worker_id):
            sources = {}
            imported.append(sources)
            peer = f"D-{worker_id}"

            def dispatch(message):
                if message["op"] == "hello":
                    return dict(peer=peer, ticket=NS(region_id=worker_id + 1, region_epoch=1, cpu=f"K-{worker_id % 8}"))
                assert message["peer"] == peer
                if message["op"] == "register":
                    metadata = message["metadata"]
                    assert metadata.peer == peer
                    assert metadata.cpu_descriptor.region_id == worker_id + 1
                    sources[message["producer"]] = metadata
                elif message["op"] == "detach":
                    sources.pop(message["producer"])
                return True

            return dispatch

        for worker in range(16):
            server = CONTROL.ControlServer("127.0.0.1", 0, service(worker), timeout=5)
            self.addCleanup(server.close)
            endpoints.append(f"127.0.0.1:{server.port}")
        for worker in range(16):
            memory = TestPDFlow.memories_for(decode=False)["l0"]
            sender = SENDER.Sender(
                sdk,
                {"l0": memory},
                config=CONFIG.PDConfig(decode_endpoints=tuple(endpoints), timeout=5),
                device_id=worker % 8,
                producer=f"P:{worker // 8}:{worker % 8}",
                host="127.0.0.1",
                activate=lambda: None,
                stream_factory=lambda: None,
                event_factory=lambda: NS(synchronize=lambda: None),
                external_id=lambda value: value,
                tp_rank=worker % 8,
                tp_size=8,
                dp_rank=worker // 8,
            )
            self.addCleanup(sender.close)
        self.assertTrue(all(len(sources) == 16 for sources in imported))
        self.assertEqual(len(sdk.sources), 16 * 16 * 2)
        self.assertEqual(
            {source.ticket.cpu for source in sdk.sources.values() if not source.npu},
            {f"K-{index}" for index in range(8)},
        )


if __name__ == "__main__":
    unittest.main()
