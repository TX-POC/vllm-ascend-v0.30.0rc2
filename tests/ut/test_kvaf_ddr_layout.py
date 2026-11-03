# SPDX-License-Identifier: Apache-2.0
"""DDR allocation locality, fragmentation fallback and reservation ownership."""

import importlib
import unittest

import test_kvaf_adapter as helpers
import test_kvaf_prefill as fixture

Pool = importlib.import_module(f"{helpers.PACKAGE.__name__}.state").DDRSlotPool


class TestExtents(unittest.TestCase):
    def test_layer_regions_and_decode_extension(self):
        pool = Pool(120)
        pool.partition({0: 1, 1: 1, None: 1})
        self.assertEqual(pool.take(10, region=0), list(range(10)))
        self.assertEqual(pool.take(10, region=1), list(range(40, 50)))
        self.assertEqual(pool.take(10), list(range(80, 90)))
        self.assertEqual(pool.take(2, region=0, after=10), [10, 11])

    def test_fragmentation_and_region_exhaustion_do_not_lose_capacity(self):
        pool = Pool(12)
        pool.partition({0: 1, 1: 1})
        keys = pool.take(12)
        retired = pool.release(keys[::2])
        pool.recycle(retired)
        self.assertEqual(set(pool.take(6, region=1)), set(retired))
        with self.assertRaisesRegex(RuntimeError, "slots exhausted"):
            pool.take(1)

    def test_shared_keys_cannot_recycle_before_final_native_free(self):
        pool = Pool(8)
        keys = pool.take(8)
        pool.retain(keys)
        self.assertEqual(pool.release(keys), [])
        retired = pool.release(keys)
        with self.assertRaises(RuntimeError):
            pool.take(1)
        pool.recycle(retired)
        self.assertEqual(pool.extents, [(0, 8)])
        self.assertEqual(pool.take(8), keys)

    def test_coalesce_out_of_order_free_and_atomic_exhaustion(self):
        pool = Pool(10)
        keys = pool.take(10)
        pool.recycle(pool.release(keys[4:8]))
        snapshot = list(pool.extents)
        with self.assertRaises(RuntimeError):
            pool.take(5)
        self.assertEqual(pool.extents, snapshot)
        pool.recycle(pool.release(keys[:4] + keys[8:]))
        self.assertEqual(pool.extents, [(0, 10)])


class TestServingReservation(unittest.TestCase):
    setUp = fixture.TestWindow.setUp
    run_chunk = fixture.TestWindow.run_chunk

    def test_chunked_prompt_stays_contiguous_and_only_written_pages_publish(self):
        self.run_chunk([("a", 0, 128, 512)])
        state = self.writer.requests["a"][0]
        reserved = list(state.reserved)
        self.assertEqual(len(reserved), 3)
        self.assertEqual(len(state.addresses), 1)
        self.run_chunk([("b", 0, 128, 512)])
        for start in (128, 256, 384):
            self.run_chunk([("a", start, 128, 512)])
        self.assertEqual(state.keys, list(range(state.keys[0], state.keys[0] + 4)))
        self.assertEqual(state.keys[1:], reserved)
        self.assertFalse(state.reserved)

    def test_cancel_returns_unwritten_reservation_without_native_free(self):
        self.run_chunk([("a", 0, 128, 512)])
        pool = self.writer.slots[(656, 128)]
        state = self.writer.requests["a"][0]
        self.assertEqual(len(state.reserved), 3)
        self.writer.release_requests(["a"])
        self.assertFalse(self.runtime.pages)
        self.assertFalse(pool.references)
        self.assertEqual(pool.extents, [(0, pool.capacity)])


if __name__ == "__main__":
    unittest.main()
