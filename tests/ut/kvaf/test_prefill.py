# SPDX-License-Identifier: Apache-2.0
"""Byte-level chunk, mixed-batch, and completed-prefix lifecycle regression."""

import importlib
import sys
import unittest
from dataclasses import replace
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from . import test_adapter as helpers

package = helpers.PACKAGE.__name__
Window = importlib.import_module(f"{package}.cache_policy").KVCachePolicy
Pool = importlib.import_module(f"{package}.state").DeviceCacheState
Writer = importlib.import_module(f"{package}.transfer").KVTransfer
Manager = importlib.import_module(f"{package}.cache_policy").CacheFeedbackManager
Archive = helpers.load_prefix_module().RequestSnapshotStore


class Runtime:
    def __init__(self):
        self.pages = {}
        self.get_counts = []

    def batch_put(self, keys, source, input_offsets, lengths):
        for key, offset, size in zip(keys, input_offsets, lengths):
            self.pages[key] = source.view(torch.uint8).reshape(-1)[offset : offset + size].clone()

    def prepare_access_table(self, keys):
        return keys

    def access_table_view(self, keys, device=None):
        values = dict(
            addresses=[(k + 1) * 128 * 656 for k in keys],
            valid=[1] * len(keys),
            lengths=[len(self.pages[k]) for k in keys],
        )
        return values

    def release_access_table(self, handle):
        pass

    def batch_get(self, *args, **kwargs):
        if len(args) == 6:
            table, indices, offsets, count, destinations, output = args
            addresses, valid = table["addresses"], table["valid"]
        else:
            addresses, valid, indices, offsets, count, destinations, output = args
        assert output.ndim >= 2 and output[0].numel() == 656
        self.get_counts.append(int(count[0]))
        flat = output.view(-1, 656)
        for i in range(int(count[0])):
            entry = int(indices[i])
            assert valid[entry]
            key = int(addresses[entry]) // (128 * 656) - 1
            offset = int(offsets[i])
            flat[int(destinations[i])].copy_(self.pages[key][offset : offset + 656])

    def batch_free(self, keys):
        for key in keys:
            del self.pages[key]


class TestWindow(unittest.TestCase):
    spill_tokens = 0

    def setUp(self):
        spill_tokens = getattr(self, "spill_tokens", 0)
        pool = Pool(("target",), 3, 16384 + spill_tokens, "cpu")
        self.hot = Window(
            pool,
            512,
            8192,
            native=lambda *args: args[14].zero_(),
            spill_tokens=spill_tokens,
            tail_tokens=getattr(self, "tail_tokens", 384),
            put_interval_steps=getattr(self, "put_interval_steps", 1),
        )
        pool.policy = self.hot
        layout = helpers.layout(656, "target")
        plane = replace(layout.planes[0], shape=tuple(self.hot.hot[0].shape))
        self.runtime = Runtime()
        self.writer = Writer(
            {(656, 128): self.runtime},
            {(656, 128): 256},
            (replace(layout, planes=(plane,)),),
            {"target": self.hot.hot[0]},
        )
        self.manager = Manager(self.writer)
        self.writer.feedback = self.manager
        self.manager.bind_device_pool(pool)
        self.addCleanup(self.manager.close)
        self.requests = {}

    def test_indexer_selector_is_group_owned_and_clears_stale_binding(self):
        metadata = NS(topk_selector="stale")
        self.hot.prepare_indexer_metadata(0, 0, metadata)
        self.assertIsNone(metadata.topk_selector)
        selector = lambda *args: args
        self.hot.indexer_selectors[0] = selector
        self.hot.prepare_indexer_metadata(0, 0, metadata)
        self.assertIs(metadata.topk_selector, selector)
        self.hot.prepare_indexer_metadata(0, 1, metadata)
        self.assertIsNone(metadata.topk_selector)
        self.hot.prepare_indexer_metadata(1, 0, metadata)
        self.assertIsNone(metadata.topk_selector)

    def test_fused_index_storage_is_included_in_hbm_budget(self):
        native = lambda *args: args[14].zero_()
        pool = Pool(("target",), 3, 16384, "cpu")
        baseline = Window(pool, 512, 8192, native=native)
        fused = Window(pool, 512, 8192, native=native, fused_history=16384)
        self.assertEqual(baseline.fused_index_storage, ())
        self.assertEqual(len(fused.fused_index_storage), 1)
        storage = fused.fused_index_storage[0]
        self.assertEqual(
            fused.nbytes - baseline.nbytes,
            storage.nbytes
            + sum(
                t.numel() * t.element_size()
                for t in (fused.fused_errors, fused.fused_reserved_hot, fused.fused_fetch_rows, fused.fused_outputs)
            ),
        )
        # Keys, FP32 scales, addresses, logical IDs, identity, residency.
        expected = 3 * 16384 * (128 + 4 + 8 + 8) + 16384 * 8 + 3 * 8192 * 4
        self.assertEqual(storage.nbytes, expected)

    def test_fused_selection_bypasses_native_prepare_and_get(self):
        slots = torch.full((1, 1, 2048), -1, dtype=torch.int32)
        slots[0, 0, :2] = torch.tensor([3, 11])
        self.hot.indexer_selectors[0] = NS(slots=lambda count: slots[:count])
        self.hot.group_sizes = (4,)
        metadata = NS(num_prefills=0)
        topk = torch.zeros_like(slots)
        self.runtime.get_counts.clear()  # Exclude output registration during bind.
        with patch.object(self.hot, "prepare_op", side_effect=AssertionError("duplicate LIDU")):
            first = self.hot.run(0, topk, metadata)
            later = self.hot.run(0, topk, metadata, local_layer=3)
        self.assertEqual(first[0, 0, :3].tolist(), [12, 44, -1])
        self.assertEqual(later[0, 0, :3].tolist(), [15, 47, -1])
        torch.testing.assert_close(self.hot.destinations[0, :1], slots)
        self.assertFalse(self.runtime.get_counts)

    def run_chunk(self, specs):
        scheduled = {}
        for name, start, count, prompt in specs:
            self.requests[name] = NS(num_computed_tokens=start, num_prompt_tokens=prompt, block_ids=(list(range(64)),))
            scheduled[name] = count
        self.writer.prepare_requests(self.requests, scheduled)
        self.manager.prepare(tuple(scheduled), scheduled, decode=False, padded_requests=len(scheduled))
        flat = self.hot.hot[0].view(-1, 656)
        for p in self.hot.plans.values():
            values = torch.arange(p.start, p.end).remainder(127).to(torch.int8)
            flat[self.hot.write_slots[p.begin : p.stop]] = values[:, None]
        self.writer.put_requests(self.requests, scheduled)

    def assert_prefix(self, request, tokens):
        state = self.writer.requests[request][0]
        data = torch.cat([self.runtime.pages[key] for key in state.keys]).view(-1, 656)
        expected = torch.arange(tokens).remainder(127).to(torch.uint8)
        torch.testing.assert_close(data[:tokens, 0], expected)

    def test_unaligned_prefill_chunks_then_mixed_decode(self):
        self.run_chunk([("a", 0, 129, 260)])
        self.run_chunk([("a", 129, 131, 260)])
        self.assert_prefix("a", 256)
        self.run_chunk([("a", 260, 4, 260), ("b", 0, 200, 260)])
        self.assertEqual(self.hot.decode_frames[0].request_ids, ("a",))
        self.assertFalse(self.hot.only_decode)
        self.run_chunk([("a", 264, 4, 260), ("b", 200, 60, 260)])
        self.assert_prefix("b", 256)
        self.run_chunk([("a", 268, 4, 260), ("b", 260, 4, 260)])
        self.assertTrue(self.hot.only_decode)

    def test_write_slot_address_is_stable_across_capture_and_prepare(self):
        address = self.hot.write_slots.data_ptr()
        self.run_chunk([("a", 0, 260, 260)])
        self.assertEqual(self.hot.write_slots.data_ptr(), address)
        self.run_chunk([("a", 260, 4, 260)])
        self.assertEqual(self.hot.write_slots.data_ptr(), address)

    def test_async_mtp_corrected_positions_choose_tail_slots(self):
        self.run_chunk([("a", 0, 260, 260)])
        self.run_chunk([("a", 263, 4, 260)])  # Optimistic CPU length.
        self.hot.align_decode_writes(torch.arange(260, 264))
        owner = self.hot.owners["a"]
        expected = owner * self.hot.slot_capacity + self.hot.tail_start + torch.arange(260, 264) % 384
        torch.testing.assert_close(self.hot.write_slots[:4], expected)

    def test_prefill_history_read_once_and_mixed_decode_uses_hot(self):
        self.run_chunk([("a", 0, 260, 260), ("b", 0, 129, 260)])
        self.run_chunk([("a", 260, 4, 260), ("b", 129, 131, 260)])
        calls = []

        def attention(impl, q, rope, caches, indices, metadata, query_end, key_end, **kwargs):
            if caches[0].data_ptr() == self.hot.context.data_ptr():
                data = caches[0].view(-1, 656)[:260, 0]
                torch.testing.assert_close(data, torch.arange(260).remainder(127).to(data.dtype))
                self.assertEqual(caches[0].data_ptr(), self.hot.context.data_ptr())
            else:
                self.assertEqual(caches[0].data_ptr(), self.hot.hot[0].data_ptr())
            self.assertEqual(key_end.tolist(), [260] if q.shape[0] == 131 else [264])
            calls.append((q.shape[0], kwargs["sparse_mode"]))
            return q + 1

        module = ModuleType("vllm_ascend.device.device_op")
        module.DeviceOperator = NS(execute_sparse_flash_attention_process=attention)
        topk = torch.zeros(135, 1, 2048, dtype=torch.int64)
        q = torch.zeros(136, 1, 2)  # Includes one TP padding row.
        before = len(self.runtime.get_counts)
        with patch.dict(sys.modules, {module.__name__: module}):
            output = self.hot.attention(
                None, 0, q, q, topk, NS(num_prefills=1, seq_lens=torch.tensor([264, 260], dtype=torch.int32)), None
            )
        self.assertEqual(calls, [(4, 3), (131, 3)])
        self.assertEqual(self.runtime.get_counts[before:], [0, 128])
        torch.testing.assert_close(output[:135], torch.ones_like(q[:135]))
        torch.testing.assert_close(output[135:], torch.zeros_like(q[135:]))

    def test_dp_prefill_padding_excluded_from_local_decode(self):
        self.run_chunk([("a", 0, 260, 260)])
        self.run_chunk([("b", 0, 260, 260)])
        self.run_chunk([("a", 260, 1, 260), ("b", 260, 4, 260)])
        calls = []
        original = self.hot.prepare_op

        def native(*args):
            calls.append((args[0].shape[0], args[1].tolist()))
            self.assertEqual(args[0].shape[0], 5)
            original(*args)

        self.hot.prepare_op = native
        module = ModuleType("vllm_ascend.device.device_op")

        def attention(impl, q, rope, caches, indices, metadata, query_end, key_end, **kwargs):
            self.assertEqual(q.shape[0], 5)
            self.assertEqual(indices.shape[0], 5)
            self.assertEqual(query_end.tolist(), [1, 5])
            self.assertEqual(kwargs["block_table"].shape[0], 2)
            self.assertEqual(kwargs["sparse_mode"], 3)
            self.assertEqual(key_end.tolist(), [261, 264])
            return q + 1

        module.DeviceOperator = NS(execute_sparse_flash_attention_process=attention)
        # Remote DP prefill pads local decode far beyond the 12-row pool.
        # Poison padding indices: they must never reach LIDU or gather.
        topk = torch.full((512, 1, 2048), 999999, dtype=torch.int64)
        topk[:5].zero_()
        q = torch.zeros(512, 1, 2)
        for token_rows, batch_rows in ((512, 2), (512, 128), (9, 2), (5, 3)):
            selected_topk = topk[:token_rows]
            metadata = NS(
                num_prefills=0,
                cum_query_lens=torch.zeros(batch_rows, dtype=torch.int32),
                seq_lens=torch.tensor([261, 264], dtype=torch.int32),
            )
            with patch.dict(sys.modules, {module.__name__: module}):
                result = self.hot.attention(None, 0, q, q, selected_topk, metadata, None)
            torch.testing.assert_close(result[:5], torch.ones_like(q[:5]))
            torch.testing.assert_close(result[5:], torch.zeros_like(q[5:]))
        self.assertEqual(calls, [(5, [1, 5])] * 4)

    def test_full_graph_keeps_capture_shapes_without_reading_host_plans(self):
        self.hot.only_decode = True

        class UnavailablePlans:
            def values(self):
                raise AssertionError("FULL graph must not read dynamic host plans")

            def __len__(self):
                raise AssertionError("FULL graph must not read dynamic host plans")

        self.hot.plans = UnavailablePlans()
        q = torch.zeros(8, 1, 2)
        topk = torch.zeros(8, 1, 2048, dtype=torch.int64)
        metadata = NS(
            num_prefills=0, cum_query_lens=torch.tensor([4, 8]), seq_lens=torch.tensor([60, 130000], dtype=torch.int32)
        )
        module = ModuleType("vllm_ascend.device.device_op")

        def attention(impl, q, rope, caches, indices, meta, query_end, key_end, **kwargs):
            self.assertEqual(kwargs["sparse_mode"], 3)
            self.assertEqual(key_end.tolist(), [60, 130000])
            self.assertEqual(key_end.data_ptr(), metadata.seq_lens.data_ptr())
            # Graph replay reuses baseline device lengths without a host read.
            self.assertEqual(metadata.seq_lens.tolist(), [60, 130000])
            return q

        module.DeviceOperator = NS(execute_sparse_flash_attention_process=attention)
        with (
            patch.dict(sys.modules, {module.__name__: module}),
            patch.object(self.hot, "run", return_value=topk) as run,
        ):
            result = self.hot.attention(None, 0, q, q, topk, metadata, metadata.cum_query_lens, full_graph=True)
        self.assertIs(result, q)
        self.assertIs(run.call_args.args[1], topk)
        self.assertIs(run.call_args.args[2], metadata)

    def test_eager_capture_warmup_keeps_synthetic_queries(self):
        self.assertFalse(self.hot.plans)
        self.assertTrue(self.hot.only_decode)
        calls = []

        def native(*args):
            # Match the original binding's shape validation before launch.
            tokens, batch = args[0].shape[0], args[1].shape[0]
            self.assertGreater(tokens, 0)
            self.assertLessEqual(batch, tokens)
            self.assertLessEqual(tokens, batch * 4)
            calls.append((tokens, batch))
            args[14].zero_()

        self.hot.prepare_op = native
        module = ModuleType("vllm_ascend.device.device_op")
        module.DeviceOperator = NS(execute_sparse_flash_attention_process=lambda impl, q, *a, **kw: q)
        for tokens, batch in ((1, 1), (4, 1), (8, 2), (12, 3)):
            topk = torch.zeros(tokens, 1, 2048, dtype=torch.int32)
            q = torch.zeros(tokens, 1, 2)
            metadata = NS(
                num_prefills=0,
                cum_query_lens=torch.zeros(batch, dtype=torch.int32),
                seq_lens=torch.ones(batch, dtype=torch.int32),
            )
            with patch.dict(sys.modules, {module.__name__: module}):
                result = self.hot.attention(None, 0, q, q, topk, metadata, metadata.cum_query_lens)
            self.assertIs(result, q)
        self.assertEqual(calls, [(1, 1), (4, 1), (8, 2), (12, 3)])

    def test_abc_backpressure_publishes_before_ring_overwrite(self):
        self.run_chunk([("a", 0, 129, 129)])
        owner = self.hot.owners["a"]
        # Delayed Put, but the full uncommitted tail is still resident.
        # Last after_forward already confirmed this boundary after MTP correction.
        self.hot.put_targets["a"] = 256
        tokens = torch.arange(128, 510)
        slots = owner * self.hot.slot_capacity + self.hot.tail_start + tokens % 384
        self.hot.hot[0].view(-1, 656)[slots] = tokens.remainder(127).to(torch.int8)[:, None]
        with self.assertLogs("vllm", level="WARNING") as logged:
            self.run_chunk([("a", 510, 4, 129)])
        self.assertIn("ABC backpressure", logged.output[0])
        self.assert_prefix("a", 256)

    def test_dummy_and_padding_map_to_initialized_guard(self):
        # Warmup has no plans. Exercise the real gather with a -1 LI sentinel
        # and ordinary short indices, rather than mocking its mapped output.
        self.hot.prepare_op = lambda *args: args[14].zero_()
        topk = torch.full((4, 1, 2048), -1, dtype=torch.int32)
        topk[:, 0, :4] = torch.arange(4, dtype=torch.int32)
        meta = NS(num_prefills=0, cum_query_lens=torch.tensor([4]))
        mapped = self.hot.run(0, topk, meta)
        self.assertTrue(torch.all(mapped == 0))
        self.assertTrue(torch.all(self.hot.hot[0][0] == 0))
        # Assigning real owners must not turn padded rows into real KV reads.
        self.run_chunk([("a", 0, 260, 260)])
        self.run_chunk([("a", 260, 4, 260)])
        self.assertTrue(torch.all(self.hot.query_bases[4:] == 1))
        self.assertTrue(torch.all(self.hot.pool.storage["token_to_slot"][:, 0, :-1] == 0))

    def test_grouped_slot_conversion_preserves_invalid_sentinel(self):
        self.hot.group_sizes = (4,)
        self.hot.mapped[0] = torch.tensor([-1, 0, 8192], dtype=torch.int32)
        metadata = NS(num_prefills=0)
        for local in (1, 2, 3):
            result = self.hot.run(0, None, metadata, local)
            torch.testing.assert_close(result, torch.tensor([-1, local, 8192 * 4 + local], dtype=torch.int32))

    def test_short_decode_keeps_dense_history_after_abc_wrap(self):
        self.run_chunk([("a", 0, 260, 260)])
        for start in range(260, 644, 4):
            self.run_chunk([("a", start, 4, 260)])
        owner = self.hot.owners["a"]
        table = self.hot.pool.views["target"].token_to_slot[owner, :640]
        self.assertTrue(torch.all(table >= 0))
        data = self.hot.hot[0].view(-1, 656)[owner * self.hot.slot_capacity + table.long(), 0]
        torch.testing.assert_close(data, torch.arange(640).remainder(127).to(data.dtype))

    def test_optional_put_verification_checks_actual_byte_offsets(self):
        verify = importlib.import_module(f"{package}.transfer").verify_put_block
        self.writer.verify = verify
        self.run_chunk([("a", 0, 260, 260)])
        self.assertEqual(self.writer._verified, {"prefill"})

    def test_archive_survives_request_release_and_shared_restore(self):
        self.run_chunk([("a", 0, 512, 512)])
        aux = NS(aux_planes=(), save=lambda blocks, tokens: NS(keys=()), load=lambda *args: None)
        archive = Archive(self.writer, aux)
        archive.save(NS(entry_id=0, request_id="a", num_tokens=512, block_ids=(tuple(range(4)),)))
        self.assertNotIn("a", self.writer.requests)
        self.assertEqual(len(self.runtime.pages), 4)
        archive.load(NS(entry_id=0, request_id="b", num_tokens=384, block_ids=(tuple(range(4)),)))
        self.assert_prefix("b", 384)
        archive.evict(NS(entry_id=0))
        self.assertEqual(len(self.runtime.pages), 3)
        self.writer.release_requests(("b",))
        self.assertFalse(self.runtime.pages)


class TestWindowWithFetchSpill(TestWindow):
    """The fused fallback must not overlap ABC, Put or restored prefixes."""

    spill_tokens = 8192


if __name__ == "__main__":
    unittest.main()
