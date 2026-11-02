# SPDX-License-Identifier: Apache-2.0
"""Real CPU byte packing across auxiliary planes, slot edges and page gaps."""

import unittest
from types import SimpleNamespace as NS

import torch
from test_kvaf_adapter import layout_module, load_prefix_module

AuxiliarySnapshotStore = load_prefix_module().AuxiliarySnapshotStore


class ByteRuntime:
    def __init__(self):
        self.pages, self.pins = {}, set()
        self.put_calls = 0

    def batch_put(self, keys, source, input_offsets, lengths):
        self.put_calls += 1
        for key, offset, length in zip(keys, input_offsets, lengths):
            assert key not in self.pages
            self.pages[key] = source[offset : offset + length].clone()

    def prepare_access_table(self, keys):
        assert all(key in self.pages for key in keys)
        self.pins.update(keys)
        return tuple(keys)

    def access_table_view(self, handle):
        return {"addresses": [key + 1 for key in handle], "valid": [1] * len(handle)}

    def release_access_table(self, handle):
        self.pins.difference_update(handle)

    def batch_get(self, addresses, valid, indices, offsets, count, output_indices, output, *, synchronize):
        for row in range(int(count)):
            index, offset = int(indices[row]), int(offsets[row])
            assert valid[index]
            key = int(addresses[index]) - 1
            assert key in self.pins
            # Native Get always reads a complete runtime.token_bytes row.
            assert offset + output.shape[1] <= self.pages[key].numel()
            output[int(output_indices[row])].copy_(self.pages[key][offset : offset + output.shape[1]])


class Slots:
    def __init__(self, capacity=100):
        self.capacity, self.next = capacity, 0

    def take(self, count):
        if self.next + count > self.capacity:
            raise RuntimeError("DDR full")
        result = tuple(range(self.next, self.next + count))
        self.next += count
        return result


class TestAuxArchive(unittest.TestCase):
    def make_store(self):
        # Include storage offsets and gaps between pages for the scale plane.
        scale_storage = torch.arange(24 * 8 + 3, dtype=torch.int16)
        scale = scale_storage.as_strided((24, 4, 1, 1), (8, 1, 1, 1), 3)
        caches = {
            "target": torch.arange(24 * 32, dtype=torch.int32).to(torch.int8).view(24, 4, 1, 8),
            "index": (torch.arange(24 * 12).to(torch.uint8).view(24, 4, 1, 3), scale),
            "draft": torch.arange(24 * 32).to(torch.int8).view(24, 4, 1, 8),
            "draft.index": torch.arange(24 * 20).to(torch.uint8).view(24, 4, 1, 5),
        }
        specs = {
            name: NS(
                block_size=4,
                page_size_bytes=sum(
                    t[0].numel() * t.element_size() for t in (cache if isinstance(cache, tuple) else (cache,))
                ),
            )
            for name, cache in caches.items()
        }
        for name in ("index", "draft.index"):
            specs[name].sfa_dcp_replicated_indexer_size = 1
        config = NS(
            kv_cache_groups=[
                NS(layer_names=["target", "index"], kv_cache_spec=NS(kv_cache_specs=specs)),
                NS(layer_names=["draft", "draft.index"], kv_cache_spec=NS(kv_cache_specs=specs)),
            ]
        )
        layouts = layout_module.describe_cache_layouts(caches, config, ("draft", "draft.index"))
        runtime, slots = ByteRuntime(), Slots()
        store = AuxiliarySnapshotStore(
            runtime, slots, layouts, caches, token_bytes=8, tokens_per_block=4, scratch_bytes=128
        )
        return store, runtime, slots, caches, scale_storage

    def test_multiplane_roundtrip_and_shorter_prefix(self):
        store, runtime, slots, caches, scale_storage = self.make_store()
        source_ids = (tuple(range(1, 23, 2)), tuple(range(2, 24, 2)))
        destination_ids = (tuple(range(10, -1, -1)), tuple(range(23, 12, -1)))
        expected = {
            (plane.layer_name, plane.index): plane.cache[list(source_ids[plane.group_ids[0]])].clone()
            for plane in store._planes
        }
        target_before = caches["target"].clone()
        archive = store.save(source_ids, 43)  # Last page is only partially valid.
        self.assertEqual(archive.payload_bytes, 11 * (12 + 8 + 32 + 20))
        self.assertEqual(len(archive.keys), (archive.payload_bytes + 31) // 32)
        self.assertEqual(slots.next, len(archive.keys))
        self.assertGreater(runtime.put_calls, 1)  # Exercise bounded-buffer reuse.
        self.assertLessEqual(max(t.numel() for t in runtime.pages.values()), 32)
        for restored_tokens in (17, 43):
            with self.subTest(restored_tokens=restored_tokens):
                for plane in store._planes:
                    plane.cache.fill_(0xEE)
                gaps = scale_storage.as_strided((24, 4), (8, 1), 7).clone()
                store.load(archive, destination_ids, restored_tokens)
                restored_pages = (restored_tokens + 3) // 4
                for plane in store._planes:
                    ids = destination_ids[plane.group_ids[0]]
                    actual = plane.cache[list(ids[:restored_pages])]
                    self.assertTrue(torch.equal(actual, expected[plane.layer_name, plane.index][:restored_pages]))
                    self.assertTrue(torch.all(plane.cache[list(ids[restored_pages:])] == 0xEE))
                self.assertTrue(torch.equal(gaps, scale_storage.as_strided((24, 4), (8, 1), 7)))
                self.assertFalse(runtime.pins)
        self.assertTrue(torch.equal(caches["target"], target_before))

    def test_invalid_or_unfunded_save_does_not_write_and_restore_is_bounded(self):
        store, runtime, slots, _, _ = self.make_store()
        with self.assertRaises(ValueError):
            store.save(((), ()), 1)
        self.assertEqual(slots.next, 0)
        with self.assertRaises(ValueError):
            store.save(((), ()), -1)
        slots.capacity = 0
        with self.assertRaisesRegex(RuntimeError, "DDR full"):
            store.save(((1,), (2,)), 1)
        self.assertFalse(runtime.pages)
        empty = store.save(((), ()), 0)
        store.load(empty, ((), ()), 0)
        with self.assertRaises(ValueError):
            store.load(empty, ((), ()), 1)


if __name__ == "__main__":
    unittest.main()
