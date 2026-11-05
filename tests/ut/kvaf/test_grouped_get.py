# SPDX-License-Identifier: Apache-2.0
"""Four-layer byte layout, request isolation and shared descriptor lifecycle."""

import importlib
import unittest
from dataclasses import replace
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

from . import test_adapter as helpers
from .test_prefill import Archive, Manager, Pool, Window, Writer

Grouping = importlib.import_module(f"{helpers.PACKAGE.__name__}.layout")


class Runtime:
    def __init__(self, width):
        self.width = width
        self.bytes = width * 656
        self.pages = {}
        self.reads = []

    def batch_put(self, keys, source, input_offsets, lengths):
        for key, offset, size in zip(keys, input_offsets, lengths):
            self.pages[key] = source.view(torch.uint8).reshape(-1)[offset : offset + size].clone()

    def batch_free(self, keys):
        for key in keys:
            del self.pages[key]

    def prepare_access_table(self, keys):
        return keys

    def release_access_table(self, handle):
        pass

    def access_table_view(self, keys, device=None):
        return dict(
            addresses=[(k + 1) * 128 * self.bytes for k in keys],
            valid=[1] * len(keys),
            lengths=[len(self.pages[k]) for k in keys],
        )

    def batch_get(self, *args, **kwargs):
        if len(args) == 6:
            table, indices, offsets, count, destinations, output = args
            addresses, valid = table["addresses"], table["valid"]
        else:
            addresses, valid, indices, offsets, count, destinations, output = args
        assert output.is_contiguous() and output.shape[1] == self.bytes
        if not int(count[0]):
            return
        self.reads.append(int(count[0]))
        for i in range(int(count[0])):
            entry = int(indices[i])
            assert valid[entry]
            key = int(addresses[entry]) // (128 * self.bytes) - 1
            offset = int(offsets[i])
            output[int(destinations[i])].copy_(self.pages[key][offset : offset + self.bytes])


class TestGrouped(unittest.TestCase):
    def test_configurable_history_keeps_fixed_abc_and_group_strides(self):
        pool = Pool(("l0", "l4"), 3, 32768, "cpu")
        hot = Window(pool, 1024, 8192, native=lambda *a: None, group_sizes=(4, 1), hot_tokens=16384)
        self.assertEqual(hot.history_capacity, 16384)
        self.assertEqual(hot.slot_capacity, 16384 + 384)
        self.assertEqual(hot.stage_start, 3 * hot.slot_capacity)
        self.assertEqual(hot.resident_tables.shape[1], hot.slot_capacity // 128)
        self.assertTrue(torch.all(hot.group_lengths[4] == hot.slot_capacity * 4))
        self.assertEqual(hot.group_tables[4].shape[1], hot.slot_capacity * 4 // 128)
        for width, tensor in zip((4, 1), hot.hot):
            self.assertGreater(tensor.numel(), 3 * hot.slot_capacity * width * 656)

    def test_split_runtime_budget_admits_two_warmed_prefix_borrowers(self):
        catalog = helpers.load_prefix_module()
        layouts = [helpers.layout(656, str(i)) for i in range(78)]
        auxiliary = replace(helpers.layout(2048, "indexer"), roles=("indexer",))
        groups = [("0",), ("1",)] + [tuple(str(i) for i in range(n, n + 4)) for n in range(2, 78, 4)]
        targets = Grouping.grouped_layouts(layouts, groups)
        weights = Grouping.runtime_weights([*layouts, auxiliary], targets)
        capacities = {}
        for geometry, weight in weights.items():
            slot = ((geometry[0] * geometry[1] + 4095) // 4096) * 4096
            capacities[geometry] = int((16 << 30) * weight / sum(weights.values())) // slot
        capacity = catalog.Capacity(128, capacities, {(656, 128): 2, (2624, 128): 19}, (((656, 128), 128, 2048 * 128),))
        store = catalog.PrefixCatalog(capacity, 1, replay_blocks=1)
        jobs = []
        for name in ("warm-a", "warm-b"):
            self.assertTrue(store.reserve(name, 131328))
            job = store.save(name, (), 130560, (), {})
            store.acknowledge_save(job.entry_id, {0})
            jobs.append(job)
        for name, job in zip(("a", "b"), jobs):
            self.assertTrue(store.reserve(name, 131328, source=job.entry_id, matched_tokens=130432))
        self.assertFalse(store.pending_evictions)

    def setUp(self):
        self.pool = Pool(("l0", "l4"), 3, 16384, "cpu")
        self.hot = Window(self.pool, 1024, 8192, native=lambda *a: a[14].zero_(), group_sizes=(4, 1))
        self.pool.policy = self.hot
        layouts = [helpers.layout(656, f"l{i}") for i in range(5)]
        layouts = [
            replace(layout, planes=(replace(layout.planes[0], shape=tuple(self.hot.hot[0 if i < 4 else 1].shape)),))
            for i, layout in enumerate(layouts)
        ]
        groups = (("l0", "l1", "l2", "l3"), ("l4",))
        packed = Grouping.grouped_layouts(layouts, groups)
        self.runtimes = {w: Runtime(w) for w in (1, 4)}
        self.writer = Writer(
            {(656 * w, 128): r for w, r in self.runtimes.items()},
            {(656 * w, 128): 512 for w in (1, 4)},
            (layouts[0], layouts[4]),
            {"l0": self.hot.hot[0], "l4": self.hot.hot[1]},
        )
        self.writer.groups = [
            replace(layer, layout=layout, geometry=(layout.planes[0].token_bytes, 128))
            for layer, layout in zip(self.writer.groups, packed)
        ]
        self.manager = Manager(self.writer)
        self.writer.feedback = self.manager
        self.manager.bind_device_pool(self.pool)
        self.addCleanup(self.manager.close)
        self.requests = {}

    @staticmethod
    def values(name, tokens, local):
        return (tokens + local * 29 + (53 if name == "b" else 0)).remainder(127).to(torch.int8)

    def prepare(self, specs):
        scheduled = {}
        for name, start, count, prompt in specs:
            self.requests[name] = NS(num_computed_tokens=start, num_prompt_tokens=prompt, block_ids=(list(range(64)),))
            scheduled[name] = count
        self.writer.prepare_requests(self.requests, scheduled)
        self.manager.prepare(tuple(scheduled), scheduled, padded_requests=len(scheduled))
        return scheduled

    def write(self):
        for group, width in enumerate(self.hot.group_sizes):
            flat = self.hot.hot[group].view(-1, 656)
            for local in range(width):
                for p in self.hot.plans.values():
                    slots = self.hot.write_slots_for(group, local)[p.begin : p.stop]
                    flat[slots] = self.values(p.request, torch.arange(p.start, p.end), local)[:, None]

    def chunk(self, specs):
        scheduled = self.prepare(specs)
        self.write()
        self.hot.after_forward(self.requests, scheduled)

    def test_grouped_metrics_count_physical_layers(self):
        stats = importlib.import_module(f"{helpers.PACKAGE.__name__}.metrics").HotMissStats(2, weights=(4, 1))
        stats.record(np.array([[3], [1]]), np.array([[10], [5]]))
        self.assertEqual(stats.misses.tolist(), [3, 3, 3, 3, 1])
        self.assertEqual(stats.references.tolist(), [10, 10, 10, 10, 5])

    def test_groups_follow_actual_index_sharing(self):
        names = [f"l{i}" for i in range(10)]
        ctx = {name: NS(impl=NS(skip_topk=i not in (0, 1, 5, 9))) for i, name in enumerate(names)}
        self.assertEqual(
            Grouping.index_groups(names, ctx), (("l0",), ("l1", "l2", "l3", "l4"), ("l5", "l6", "l7", "l8"), ("l9",))
        )

    def test_put_interleaving_partial_chunk_and_archive_references(self):
        self.chunk([("a", 0, 260, 512), ("b", 0, 260, 512)])
        self.chunk([("a", 260, 252, 512), ("b", 260, 252, 512)])
        for name in ("a", "b"):
            for group, width in enumerate((4, 1)):
                pages = self.writer.requests[name][group].keys
                data = torch.cat([self.runtimes[width].pages[k] for k in pages]).view(512, width, 656)
                for local in range(width):
                    torch.testing.assert_close(
                        data[:, local, 0].view(torch.int8), self.values(name, torch.arange(512), local)
                    )
        aux = NS(aux_planes=(), save=lambda *a: NS(keys=()), load=lambda *a: None)
        archive = Archive(self.writer, aux)
        self.assertEqual(archive.capacity.slots_per_block, {(2624, 128): 1, (656, 128): 1})
        archive.save(NS(entry_id=1, request_id="a", num_tokens=512, block_ids=(tuple(range(4)),)))
        archive.load(NS(entry_id=1, request_id="c", num_tokens=384, block_ids=(tuple(range(4)),)))
        archive.evict(NS(entry_id=1))
        self.writer.release_requests(("b", "c"))
        self.assertTrue(all(not r.pages for r in self.runtimes.values()))

    def test_grouped_short_decode_wrap_and_put_readback(self):
        self.writer.verify = importlib.import_module(f"{helpers.PACKAGE.__name__}.transfer").verify_put_block
        self.chunk([("a", 0, 132, 132)])
        for start in range(132, 648, 4):
            self.chunk([("a", start, 4, 132)])
        self.assertEqual(self.writer._verified, {"prefill", "decode"})
        owner = self.hot.owners["a"]
        maps = self.pool.views["l0"].token_to_slot[owner, :640].long()
        for local in range(4):
            rows = (owner * 8576 + maps) * 4 + local
            data = self.hot.hot[0].view(-1, 656)[rows, 0]
            torch.testing.assert_close(data, self.values("a", torch.arange(640), local))

    def test_prefill_four_layers_reuse_one_workspace_across_requests(self):
        self.chunk([("a", 0, 256, 512)])
        self.chunk([("b", 0, 256, 512)])
        self.assertEqual(self.hot.group_context.shape, (1, 8192, 4, 656))
        address = self.hot.group_context.data_ptr()
        self.runtimes[4].reads.clear()
        module = ModuleType("vllm_ascend.device.device_op")

        for name in ("a", "b"):
            self.prepare([(name, 256, 256, 512)])
            self.write()

            def sfa(impl, q, rope, caches, indices, metadata, query_ends, key_end, name=name, **kw):
                data = caches[0].view(-1, 656)[:512, 0]
                torch.testing.assert_close(data, self.values(name, torch.arange(512), impl.kvaf_group_offset))
                self.assertEqual(kw["sparse_mode"], 3)
                return q

            module.DeviceOperator = NS(execute_sparse_flash_attention_process=sfa)
            with patch.dict("sys.modules", {"vllm_ascend.device.device_op": module}):
                for local in range(4):
                    self.hot.attention(
                        NS(kvaf_group_offset=local),
                        0,
                        torch.zeros(256, 1),
                        torch.zeros(256, 1),
                        torch.zeros(256, 1, 2048, dtype=torch.int32),
                        NS(num_prefills=1, seq_lens=torch.tensor([256], dtype=torch.int32)),
                        None,
                    )
            self.assertEqual(self.hot.group_context.data_ptr(), address)
        self.assertEqual(self.runtimes[4].reads, [256, 256])

    def test_decode_one_lidu_get_and_layer_slot_offsets(self):
        for start in range(0, 4096, 1024):
            self.chunk([("a", start, 1024, 4096)])
        self.prepare([("a", 4096, 4, 4096)])
        calls = []
        token, slot = 37, 7
        owner = self.hot.owners["a"]

        def native(*a):
            calls.append(1)
            a[3][owner, token] = slot
            a[14].zero_()
            a[14][-1] = 1
            a[15][0] = owner * self.hot.group_count * self.hot.qcap + token // 128
            a[16][0] = token % 128 * 656
            a[17][0] = owner * 8576 + slot

        self.hot.prepare_op = native
        topk = torch.full((4, 1, 2048), token, dtype=torch.int32)
        topk[:, :, -1] = -1
        meta = NS(num_prefills=0, cum_query_lens=torch.tensor([4]))
        self.runtimes[4].reads.clear()
        for local in range(4):
            result = self.hot.run(0, topk, meta, local)
            self.assertTrue(torch.all(result[:, :, :-1] == slot * 4 + local))
            self.assertTrue(torch.all(result[:, :, -1] == -1))
            logical = int(result.flatten()[0])
            physical = int(self.hot.group_tables[4][0, logical // 128]) * 128 + logical % 128
            self.assertEqual(physical, (owner * 8576 + slot) * 4 + local)
            row = self.hot.hot[0].view(-1, 656)[(owner * 8576 + slot) * 4 + local]
            self.assertTrue(torch.all(row == self.values("a", torch.tensor(token), local)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.runtimes[4].reads, [1])
        self.hot.align_decode_writes(torch.arange(4097, 4101))
        for local in range(4):
            torch.testing.assert_close(self.hot.write_slots_for(0, local)[:4], self.hot.all_writes[:4] * 4 + local)


if __name__ == "__main__":
    unittest.main()
