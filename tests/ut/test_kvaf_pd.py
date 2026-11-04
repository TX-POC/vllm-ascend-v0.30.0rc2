# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for PD components; no vLLM, CANN or KVAF binary required.

Run: python -m unittest discover -s tests/ut -p 'test_kvaf_pd.py' -v
Only the package loader and SDK/device boundary are substituted. Layout,
descriptor, transaction and completion implementations execute unchanged.
"""

import importlib
import importlib.util
import subprocess
import sys
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import test_kvaf_adapter as native
import torch

PD_PATH = Path(__file__).resolve().parents[2] / "vllm_ascend/distributed/kv_transfer/kvaf/pd"
SPEC = importlib.util.spec_from_file_location("kvaf_pd_under_test", PD_PATH / "__init__.py")
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
layout = importlib.import_module(f"{SPEC.name}.layout")
transfer = importlib.import_module(f"{SPEC.name}.transfer")
state = importlib.import_module(f"{SPEC.name}.state")
registration = importlib.import_module(f"{SPEC.name}.registration")


def layer_buffer(blocks=8, fill=251):
    # Real shared storage, with nonzero layer offset and gaps between planes.
    size = blocks * 128 * (656 + 128 + 4) + 64
    storage = torch.full((size + 32,), fill, dtype=torch.uint8)
    layer = storage[32:]
    planes, cursor = {}, 0
    for name, width in (("main", 656), ("indexer", 128), ("scale", 4)):
        size = blocks * 128 * width
        planes[name] = layer[cursor : cursor + size].view(blocks, 128, width)
        cursor += size + 32
    return layer, planes, layout.LayerLayout.from_tensors("layer.0", layer, planes)


class CpuState(Enum):
    PREPARED = 1
    SUBMITTED = 2
    RUNNING = 3
    DONE = 4
    FAILED = 5
    IN_DOUBT = 6
    COMMITTED = 7
    ABORTED = 8


class Runtime:
    """Deterministic SDK boundary with independently checked byte destinations."""

    def __init__(self, source):
        self.source = source
        self.calls = []
        self.pages = {}
        self.raw_calls = 0
        self.pending_raw_addr = False
        self.submit_lost = self.commit_lost = self.raw_lost = False
        self.wait_error = False

    def prepare_put(self, keys, lengths):
        self.calls.append("prepare")
        entries = tuple(NS(key=k, length=n, address=1000000 + k * 400000) for k, n in zip(keys, lengths))
        return NS(entries=entries, submitted=False, state=CpuState.PREPARED)

    def submit_cpu_pull(self, ready, put, sources, destinations, lengths):
        self.calls.append("submit")
        put.submitted, put.state = True, CpuState.SUBMITTED
        self.last_put, self.last_ready = put, ready
        for src, dst, size in zip(sources, destinations, lengths):
            entry = next(e for e in put.entries if e.address <= dst and dst + size <= e.address + e.length)
            page = self.pages.setdefault(entry.key, torch.full((entry.length,), 250, dtype=torch.uint8))
            page[dst - entry.address : dst - entry.address + size].copy_(self.source[src : src + size])
        if self.submit_lost:
            raise TimeoutError("submit response lost")

    def wait_cpu_pull(self, put, timeout):
        self.calls.append("wait")
        if self.wait_error:
            raise TimeoutError("DMA unresolved")
        put.state = CpuState.DONE
        return put.state

    def commit_put(self, put):
        self.calls.append("commit")
        if self.commit_lost:
            self.commit_lost = False
            raise TimeoutError("Commit acknowledgment lost")
        put.state = CpuState.COMMITTED
        return put.state

    def query_cpu_pull(self, put):
        self.calls.append("query")
        return put.state

    def abort_put(self, put):
        self.calls.append("abort")
        put.state = CpuState.ABORTED
        return put.state

    def batch_get_raw_addr(self, ready, sources, destinations, lengths, output, stream):
        self.raw_calls += 1
        for src, dst, size in zip(sources.tolist(), destinations.tolist(), lengths.tolist()):
            output[dst : dst + size].copy_(self.source[src : src + size])
        if self.raw_lost:
            self.pending_raw_addr = True
            raise TimeoutError("raw acknowledgment lost")
        return {"state": "done", "sequence": ready.sequence}

    def resolve_raw_addr(self, ready):
        self.calls.append("resolve")
        self.last_ready = ready
        self.pending_raw_addr = False


class TestPDLayout(unittest.TestCase):
    def setUp(self):
        self.source, self.source_planes, self.source_layout = layer_buffer()
        self.output, self.output_planes, self.output_layout = layer_buffer()
        self.p_tables = {"main": (5, 1, 7), "indexer": (3, 2, 4), "scale": (1, 6, 3)}
        self.d_tables = {"main": (2, 6, 0), "indexer": (7, 1, 5), "scale": (0, 4, 2)}
        for name, plane in self.source_planes.items():
            for q, block in enumerate(self.p_tables[name]):
                plane[block].copy_((torch.arange(128).view(128, 1) + q * 128).remainder(249))

    def plan(self, tokens, **kwargs):
        return layout.plan_layer(self.source_layout, self.output_layout, self.p_tables, self.d_tables, tokens, **kwargs)

    def test_offsets_use_layer_base_and_each_plane_block_table(self):
        typed = {**self.source_planes, "scale": self.source_planes["scale"].view(torch.float32)}
        self.assertEqual(layout.LayerLayout.from_tensors("layer.0", self.source, typed), self.source_layout)
        for name, width in (("main", 656), ("indexer", 128), ("scale", 4)):
            self.assertEqual(
                self.source_layout.offset(name, self.p_tables[name], 1),
                self.source_planes[name].data_ptr() - self.source.data_ptr() + self.p_tables[name][1] * 128 * width,
            )

    def test_full_blocks_tail_and_indexer_lengths(self):
        for tokens in (0, 1, 127, 128, 129, 255, 256, 300):
            with self.subTest(tokens=tokens):
                plan = self.plan(tokens)
                full, tail = divmod(tokens, 128)
                self.assertEqual(plan.full_blocks, tuple(range(full)))
                self.assertEqual(sum(plan.hbm.lengths), tokens * 132 + tail * 656)
                self.assertEqual(len(plan.main_sources), full)
                if tokens == 300:
                    self.assertEqual(plan.hbm.lengths[0], 28864)
                    self.assertEqual(full * 128 * 656, 167936)

    def test_byte_transfer_preserves_padding_and_never_writes_full_main_to_hbm(self):
        plan = self.plan(300)
        runtime = Runtime(self.source)
        put = runtime.prepare_put((9, 2), (83968, 83968))
        ready = NS(source=NS(source_bytes=self.source.numel()), sequence=7)
        cpu = transfer.CpuPullTransfer(runtime, ready, put, plan.ddr_segments(put.entries, (9, 2)))
        cpu.submit()
        cpu.finish()
        for q, key in enumerate((9, 2)):
            torch.testing.assert_close(
                runtime.pages[key], self.source_planes["main"][self.p_tables["main"][q]].view(-1)
            )
        raw = transfer.NpuPullTransfer(runtime, ready, plan.hbm, self.output, object())
        arrays = tuple(
            torch.tensor(a, dtype=torch.int64) for a in (plan.hbm.sources, plan.hbm.destinations, plan.hbm.lengths)
        )
        with patch.object(raw, "_make_arrays", return_value=arrays):
            raw.run()
        for name, plane in self.output_planes.items():
            for q, block in enumerate(self.d_tables[name]):
                valid = 128 if q < 2 else 44
                if name == "main" and q < 2:
                    self.assertTrue(torch.all(plane[block] == 251))
                    continue
                expected = self.source_planes[name][self.p_tables[name][q]][:valid]
                torch.testing.assert_close(plane[block, :valid], expected)
                self.assertTrue(torch.all(plane[block, valid:] == 251))

    def test_chunk_transfers_only_new_complete_blocks_and_final_tail(self):
        first = self.plan(200, final=False)
        final = self.plan(300, first_block=1)
        self.assertEqual(first.full_blocks, (0,))
        self.assertEqual(final.full_blocks, (1,))
        self.assertEqual(sum(first.hbm.lengths) + sum(final.hbm.lengths), 300 * 132 + 44 * 656)
        self.assertFalse(set(first.hbm.destinations).intersection(final.hbm.destinations))

    def test_tp8_partition_has_one_main_owner_and_no_duplicate_indexer(self):
        whole = self.plan(300)
        parts = [self.plan(300, ratio=8, member=rank) for rank in range(8)]
        self.assertEqual([p.full_blocks for p in parts], [(0, 1)] + [()] * 7)
        actual = sorted((s, d, n) for p in parts for s, d, n in zip(p.hbm.sources, p.hbm.destinations, p.hbm.lengths))
        self.assertEqual(actual, sorted(zip(whole.hbm.sources, whole.hbm.destinations, whole.hbm.lengths)))
        self.assertEqual(sum(not p.hbm.sources for p in parts), 5)
        self.assertEqual(tuple(layout.contributor_range(1, 3, 8, 3)), (3,))
        self.assertFalse(layout.contributor_range(1, 0, 8, 3))

    def test_wrong_or_short_block_tables_fail(self):
        for table in ((-1,), (8,), (True,), ()):
            with self.subTest(table=table), self.assertRaises(ValueError):
                self.source_layout.offset("main", table, 0)
        for kwargs in ({"first_block": 5}, {"member": 8, "ratio": 8}, {"ratio": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plan(300, **kwargs)

    def test_independent_or_strided_planes_are_not_forged_into_one_registration(self):
        for bad in (self.source_planes["scale"].clone(), self.source_planes["scale"][:, :, ::2]):
            planes = {**self.source_planes, "scale": bad}
            with self.assertRaisesRegex(ValueError, "registered layer storage"):
                layout.LayerLayout.from_tensors("layer.0", self.source, planes)

    def test_grouped_ddr_entries_rejected(self):
        plan = self.plan(300)
        entries = (NS(key=1, address=500000, length=4 * 83968), NS(key=2, address=1000000, length=4 * 83968))
        with self.assertRaisesRegex(ValueError, "grouped DDR"):
            plan.ddr_segments(entries, (1, 2))

    def test_wrong_put_keys_cannot_be_bound_by_entry_index(self):
        plan = self.plan(300)
        entries = (NS(key=2, address=500000, length=83968), NS(key=1, address=1000000, length=83968))
        with self.assertRaises(ValueError):
            plan.ddr_segments(entries, (1, 2))

    def test_segment_limits_and_destination_overlap(self):
        for values in (((0,), (), (1,)), ((0,), (0,), (0,)), ((0,), (0,), (1 << 32,)), ((0, 4), (0, 1), (2, 2))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                layout.Segments(*values)
        with self.assertRaises(ValueError):
            layout.Segments().validate_call(100)
        segments = layout.Segments(tuple(range(4097)), tuple(range(4097)), (1,) * 4097)
        with self.assertRaises(ValueError):
            segments.validate_call(4097)
        self.assertEqual([len(s.sources) for s in segments.batches()], [4096, 1])
        for batch in segments.batches():
            batch.validate_call(4097, 4097)
        with self.assertRaisesRegex(ValueError, "source"):
            layout.Segments((99,), (0,), (2,)).validate_call(100, 100)
        with self.assertRaisesRegex(ValueError, "output"):
            layout.Segments((0,), (99,), (2,)).validate_call(100, 100)

    def test_segments_snapshot_mutable_message_arrays(self):
        sources, destinations, lengths = [0], [8], [4]
        segments = layout.Segments(sources, destinations, lengths)
        sources[0] = destinations[0] = -1
        lengths[0] = 0
        segments.validate_call(32, 32)
        self.assertEqual(segments.sources, (0,))


class TestPDTransactions(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime(torch.arange(32, dtype=torch.uint8))
        self.ready = NS(source=NS(source_bytes=32), sequence=17)
        self.put = self.runtime.prepare_put((7,), (8,))
        self.segments = layout.Segments((3, 19), (self.put.entries[0].address, self.put.entries[0].address + 4), (4, 4))
        self.cpu = transfer.CpuPullTransfer(self.runtime, self.ready, self.put, self.segments)

    def test_submit_wait_commit_are_distinct(self):
        self.cpu.submit()
        self.assertFalse(self.cpu.committed)
        self.cpu.finish()
        self.cpu.finish()
        self.assertEqual(self.runtime.calls, ["prepare", "submit", "wait", "commit"])
        self.assertTrue(self.cpu.committed)
        with self.assertRaisesRegex(RuntimeError, "resubmit"):
            self.cpu.submit()

    def test_submit_response_loss_recovers_original_put(self):
        self.runtime.submit_lost = True
        with self.assertRaises(TimeoutError):
            self.cpu.submit()
        self.cpu.finish()
        self.assertIs(self.runtime.last_put, self.put)
        self.assertIs(self.runtime.last_ready, self.ready)
        self.assertEqual(self.runtime.calls.count("prepare"), 1)
        self.assertEqual(self.runtime.calls.count("submit"), 1)
        self.assertNotIn("abort", self.runtime.calls)

    def test_commit_response_loss_cannot_be_reversed_or_replayed(self):
        self.cpu.submit()
        self.runtime.commit_lost = True
        with self.assertRaises(TimeoutError):
            self.cpu.finish()
        with self.assertRaisesRegex(RuntimeError, "cannot reverse"):
            self.cpu.abort()
        self.cpu.finish()
        self.assertEqual(self.runtime.calls, ["prepare", "submit", "wait", "commit", "commit"])

    def test_wait_timeout_preserves_all_owners(self):
        self.cpu.submit()
        self.runtime.wait_error = True
        with self.assertRaises(TimeoutError):
            self.cpu.finish()
        self.put.state = CpuState.IN_DOUBT
        with self.assertRaisesRegex(RuntimeError, "DMA"):
            self.cpu.abort()
        self.assertIs(self.cpu.put, self.put)
        self.assertFalse(self.cpu.committed)
        self.assertFalse(self.cpu.aborted)
        self.assertNotIn("abort", self.runtime.calls)

    def test_failed_dma_can_abort_but_never_commit(self):
        self.cpu.submit()
        self.put.state = CpuState.FAILED
        self.cpu.abort()
        self.cpu.abort()
        self.assertTrue(self.cpu.aborted)
        with self.assertRaisesRegex(RuntimeError, "Abort"):
            self.cpu.finish()

    def test_unsubmitted_put_aborts_without_wait(self):
        self.cpu.abort()
        self.assertEqual(self.runtime.calls, ["prepare", "abort"])
        with self.assertRaises(RuntimeError):
            self.cpu.submit()

    def test_submit_parameter_failure_does_not_wait_unsubmitted_put(self):
        with (
            patch.object(self.runtime, "submit_cpu_pull", side_effect=ValueError("bad Ready")),
            self.assertRaises(ValueError),
        ):
            self.cpu.submit()
        with self.assertRaisesRegex(RuntimeError, "not submitted"):
            self.cpu.finish()
        self.cpu.abort()
        self.assertNotIn("wait", self.runtime.calls)

    def test_exact_cover_rejects_partial_interleaved_layer_and_cross_slot(self):
        base = self.put.entries[0].address
        for destinations, lengths in (((base,), (4,)), ((base + 1,), (8,)), ((base, base + 5), (4, 3))):
            with self.subTest(destinations=destinations), self.assertRaises(ValueError):
                segments = layout.Segments((0,) * len(lengths), destinations, lengths)
                transfer.CpuPullTransfer(self.runtime, self.ready, self.put, segments)
        put = NS(entries=(NS(address=base, length=4), NS(address=base + 4, length=4)))
        with self.assertRaisesRegex(ValueError, "crossing slots"):
            transfer.CpuPullTransfer(self.runtime, self.ready, put, layout.Segments((0,), (base,), (8,)))

    def make_raw(self):
        segments = layout.Segments((2, 16), (0, 8), (4, 4))
        raw = transfer.NpuPullTransfer(self.runtime, self.ready, segments, torch.zeros(16, dtype=torch.uint8), object())
        arrays = tuple(
            torch.tensor(a, dtype=torch.int64) for a in (segments.sources, segments.destinations, segments.lengths)
        )
        return raw, arrays

    def test_raw_result_is_status_and_data_lands_in_output(self):
        raw, arrays = self.make_raw()
        with patch.object(raw, "_make_arrays", return_value=arrays):
            self.assertIsNone(raw.run())
        torch.testing.assert_close(raw.output[:4], self.runtime.source[2:6])
        torch.testing.assert_close(raw.output[8:12], self.runtime.source[16:20])
        with self.assertRaisesRegex(RuntimeError, "never replay"):
            raw.run()
        self.assertTrue(raw.done)

    def test_raw_ack_loss_resolves_without_re_read(self):
        raw, arrays = self.make_raw()
        self.runtime.raw_lost = True
        with patch.object(raw, "_make_arrays", return_value=arrays), self.assertRaises(TimeoutError):
            raw.run()
        self.assertIs(raw.arrays, arrays)
        self.assertFalse(raw.done)
        raw.recover()
        self.assertTrue(raw.done)
        self.assertEqual(self.runtime.raw_calls, 1)
        self.assertIs(self.runtime.last_ready, self.ready)

    def test_raw_failure_cannot_be_promoted_to_completion(self):
        raw, arrays = self.make_raw()
        with (
            patch.object(raw, "_make_arrays", return_value=arrays),
            patch.object(self.runtime, "batch_get_raw_addr", side_effect=ValueError("bad metadata")),
            self.assertRaises(ValueError),
        ):
            raw.run()
        with self.assertRaisesRegex(RuntimeError, "no recoverable"):
            raw.recover()
        self.assertFalse(raw.done)

    def test_raw_recovery_that_cancels_failed_acquire_does_not_allow_decode(self):
        raw, arrays = self.make_raw()
        self.runtime.raw_lost = True
        with patch.object(raw, "_make_arrays", return_value=arrays), self.assertRaises(TimeoutError):
            raw.run()
        with (
            patch.object(self.runtime, "resolve_raw_addr", side_effect=RuntimeError("cancelled before DMA")),
            self.assertRaises(RuntimeError),
        ):
            raw.recover()
        self.assertFalse(raw.done)
        self.assertIs(raw.arrays, arrays)

    def test_cpu_output_never_implicitly_sent_to_npu_sdk(self):
        raw, _ = self.make_raw()
        with self.assertRaisesRegex(ValueError, "NPU output"):
            raw.run()
        self.assertFalse(raw.attempted)
        self.assertEqual(self.runtime.raw_calls, 0)


class TestPDCompletion(unittest.TestCase):
    def setUp(self):
        self.main = state.Contribution("target.0", "p.dp0.tp0", 0)
        self.mtp = state.Contribution("mtp.0", "p.dp0.tp0", 0)
        self.empty = state.Contribution("target.0", "p.dp0.tp7", 0)
        self.gate = state.CompletionGate(
            "request", 3, {self.main: frozenset(("hbm", "ddr")), self.mtp: frozenset(("hbm",)), self.empty: frozenset()}
        )

    def record(self, key, event):
        self.gate.record("request", 3, key, event)

    def test_first_decode_waits_for_both_paths_publication_mtp_and_empty_contributors(self):
        self.record(self.main, "ack")
        self.record(self.main, "hbm_done")
        self.assertFalse(self.gate.source_reusable(self.main))
        self.record(self.main, "ddr_committed")
        self.assertTrue(self.gate.source_reusable(self.main))
        self.assertFalse(self.gate.decode_ready)
        self.record(self.main, "ddr_published")
        self.record(self.mtp, "hbm_done")
        self.record(self.mtp, "ack")
        self.assertFalse(self.gate.decode_ready)
        self.record(self.empty, "ack")
        self.assertTrue(self.gate.decode_ready)
        self.record(self.empty, "ack")
        self.assertTrue(self.gate.decode_ready)

    def test_stale_generation_unexpected_peer_and_early_publication_rejected(self):
        for request, epoch, key in (
            ("other", 3, self.main),
            ("request", 2, self.main),
            ("request", 3, state.Contribution("target.0", "other", 0)),
        ):
            with self.subTest(request=request, epoch=epoch, key=key), self.assertRaises(ValueError):
                self.gate.record(request, epoch, key, "ack")
        with self.assertRaisesRegex(ValueError, "before Commit"):
            self.record(self.main, "ddr_published")
        with self.assertRaises(ValueError):
            self.record(self.empty, "hbm_done")

    def test_failure_does_not_release_inflight_storage_or_allow_decode(self):
        self.gate.fail()
        self.record(self.main, "ack")
        self.assertFalse(self.gate.source_reusable(self.main))
        self.assertFalse(self.gate.decode_ready)


class TestPDPublication(unittest.TestCase):
    def setUp(self):
        native_transfer = importlib.import_module(f"{native.PACKAGE.__name__}.transfer")
        self.runtime = Mock(spec=("prepare_access_table", "access_table_view", "release_access_table"))
        self.group_state = native_transfer.RequestGroupState(native_transfer.RequestCacheState(128))
        self.writer = native_transfer.KVTransfer(
            {(656, 128): self.runtime},
            {(656, 128): 16},
            (native.layout(656, "target"),),
            {"target": torch.zeros((8, 128, 1, 656), dtype=torch.uint8)},
        )
        self.writer.requests["request"] = [self.group_state]
        self.group = self.writer.groups[0]
        keys = self.writer.slots[(656, 128)].take(1)
        self.put = NS(entries=(NS(key=keys[0], address=1000000, length=83968),))
        ready = NS(source=NS(source_bytes=83968), sequence=1)
        self.cpu = transfer.CpuPullTransfer(self.runtime, ready, self.put, layout.Segments((0,), (1000000,), (83968,)))
        self.runtime.access_table_view.return_value = dict(addresses=[1000000], valid=[1], lengths=[83968])

    def publish(self):
        self.cpu.publish(self.writer, self.group, self.group_state)

    def test_directory_is_published_through_existing_writer_only_after_commit(self):
        with self.assertRaisesRegex(RuntimeError, "Commit"):
            self.publish()
        self.runtime.prepare_access_table.assert_not_called()
        self.cpu.committed = True  # SDK lifecycle is independently exercised above.
        self.publish()
        self.publish()
        self.assertEqual(self.group_state.keys, [0])
        self.assertEqual(self.group_state.addresses, [1000000])
        self.runtime.prepare_access_table.assert_called_once_with([0])
        self.runtime.release_access_table.assert_called_once()
        self.assertEqual(self.writer.slots[(656, 128)].references, {0: 1})

    def test_bad_catalog_keeps_committed_keys_owned_and_retries_only_publication(self):
        self.cpu.committed = True
        self.runtime.access_table_view.return_value = dict(addresses=[0], valid=[1], lengths=[83968])
        with self.assertRaisesRegex(RuntimeError, "address table is invalid"):
            self.publish()
        self.assertEqual(self.group_state.keys, [0])
        self.assertEqual(self.group_state.addresses, [])
        self.assertFalse(self.cpu.published)
        self.runtime.access_table_view.return_value = dict(addresses=[1000000], valid=[1], lengths=[83968])
        self.publish()
        self.assertEqual(self.group_state.keys, [0])
        self.assertEqual(self.group_state.addresses, [1000000])
        self.assertTrue(self.cpu.published)

    def test_wrong_runtime_or_request_cannot_receive_committed_entries(self):
        self.cpu.committed = True
        with patch.dict(self.writer.runtimes, {(656, 128): object()}), self.assertRaisesRegex(ValueError, "region"):
            self.publish()
        self.publish()
        with self.assertRaisesRegex(ValueError, "another request"):
            self.cpu.publish(self.writer, self.group, NS(keys=[], addresses=[]))


class TestPDRegistration(unittest.TestCase):
    def setUp(self):
        self.tensor, self.planes, _ = layer_buffer(2)
        self.sdk = NS(
            CpuPullHbmSource=Mock(side_effect=lambda **kw: Mock()),
            NpuPullHbmSource=Mock(side_effect=lambda **kw: Mock()),
        )

    def source(self, peer="d0"):
        return registration.LayerRegistration(
            self.sdk,
            peer,
            "layer.0",
            self.tensor,
            self.planes,
            device_id=0,
            cpu_eid="cpu-route",
            npu_eid="fullmesh-route",
        )

    def register(self, source):
        return source.register(
            "ddr-ticket", "stream", listen_address="0.0.0.0:29001", advertised_address="10.0.0.1:29001"
        )

    def test_each_d_peer_owns_distinct_single_reader_sources(self):
        sources = [self.source(f"d{rank}") for rank in range(16)]
        self.assertEqual(len({id(source.npu) for source in sources}), 16)
        self.assertEqual(len({id(source.cpu) for source in sources}), 16)
        self.sdk.CpuPullHbmSource.assert_called_with(device_id=0, npu_eid="cpu-route")
        self.sdk.NpuPullHbmSource.assert_called_with(device_id=0, npu_eid="fullmesh-route")

    def test_pa_layout_separate_from_opaque_sdk_tickets(self):
        source = self.source()
        metadata = self.register(source)
        self.assertEqual(metadata.layout.pa_base, self.tensor.data_ptr())
        self.assertIs(metadata.cpu_descriptor, source.cpu.register.return_value)
        self.assertIs(metadata.npu_ticket, source.npu.register.return_value)
        source.cpu.register.assert_called_once_with("ddr-ticket", self.tensor, "stream")
        runtime = Mock()
        metadata.attach(runtime, peer="d0", local_eid="reader-eid")
        runtime.attach_cpu_source.assert_called_once_with(metadata.cpu_descriptor)
        runtime.attach_hbm.assert_called_once_with(metadata.npu_address, metadata.npu_ticket, local_eid="reader-eid")
        with self.assertRaises(ValueError):
            metadata.attach(runtime, peer="d1", local_eid="reader-eid")

    def test_ready_waits_for_production_and_release_failure_retains_batch(self):
        source = self.source()
        self.register(source)
        event = Mock()
        token = NS(source=NS(source_id=9, source_bytes=2048, region_id=3, region_epoch=1), sequence=1)
        source.cpu.ready.side_effect = lambda: (event.synchronize.assert_called_once(), token)[1]
        ready = source.ready("cpu", event)
        with self.assertRaises(RuntimeError):
            source.ready("cpu", event)
        source.cpu.release_batch.side_effect = RuntimeError("still being read")
        with self.assertRaises(RuntimeError):
            source.release("cpu", ready)
        with self.assertRaises(RuntimeError):
            source.close()
        source.cpu.shutdown.assert_not_called()
        source.cpu.release_batch.side_effect = None
        source.release("cpu", ready)
        source.close()
        source.close()
        source.cpu.shutdown.assert_called_once()

    def test_delayed_completion_cannot_release_another_active_batch(self):
        source = self.source()
        self.register(source)
        desc = NS(source_id=9, source_bytes=2048, region_id=3, region_epoch=1)
        current = NS(source=desc, sequence=2)
        source.cpu.ready.return_value = current
        source.ready("cpu", Mock())
        with self.assertRaisesRegex(ValueError, "active source"):
            source.release("cpu", NS(source=desc, sequence=1))
        source.cpu.release_batch.assert_not_called()
        source.cancel("cpu", current)
        with self.assertRaises(RuntimeError):
            source.close()
        source.release("cpu", NS(source=NS(**vars(desc)), sequence=2))
        source.close()

    def test_partial_registration_retains_source_and_tensor_for_cleanup(self):
        source = self.source()
        source.npu.register.side_effect = RuntimeError("registration failed")
        with self.assertRaises(RuntimeError):
            self.register(source)
        with self.assertRaises(RuntimeError):
            self.register(source)
        self.assertIs(source.tensor, self.tensor)
        self.assertIs(source.cpu_descriptor, source.cpu.register.return_value)
        source.npu.shutdown.side_effect = RuntimeError("detach uncertain")
        with self.assertRaises(RuntimeError):
            source.close()
        source.cpu.shutdown.assert_not_called()
        source.npu.shutdown.side_effect = None
        source.close()
        source.cpu.shutdown.assert_called_once()

    def test_no_hardware_or_registration_side_effects_on_component_import(self):
        code = (
            "import importlib.util, sys; "
            f"s=importlib.util.spec_from_file_location('pd', {str(PD_PATH / '__init__.py')!r}); "
            "m=importlib.util.module_from_spec(s); sys.modules['pd']=m; s.loader.exec_module(m); "
            "import pd.layout, pd.transfer, pd.registration, pd.state; "
            "assert not {'torch','torch_npu','kvaf','vllm'} & sys.modules.keys()"
        )
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
