# SPDX-License-Identifier: Apache-2.0
"""Serving-policy feedback handoff, FIFO overlap, and original LRU correctness."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
from kvip.cache.lru import SlotLRU

from . import test_prefill as fixture


class TestFeedback(unittest.TestCase):
    setUp = fixture.TestWindow.setUp
    run_chunk = fixture.TestWindow.run_chunk
    assert_prefix = fixture.TestWindow.assert_prefix

    def prepare(self, start=4096, ids=("a",)):
        for name in ids:
            if name not in self.writer.requests:
                for begin in range(0, 4096, 512):
                    self.run_chunk([(name, begin, 512, 4096)])
        self.run_chunk([(name, start, 4, 4096) for name in ids])
        return self.manager.groups[0]

    def submit(self, value):
        self.manager.submit_feedback(
            np.full((1, 4, 1), value, np.int32),
            generation=self.manager.generation,
            miss_counts=np.zeros((1, 2), np.int32),
        )

    def test_fused_fetch_counts_reuse_lru_feedback_and_ignore_padding(self):
        frame = self.prepare()
        rows = np.zeros((1, 2, 256), dtype=np.int32)
        rows[0, 0, 9], rows[0, 0, 247] = 3, 1
        rows[0, 1, 9] = 999  # Graph padding is not an active request.
        feedback = np.array([[[1, 2], [2, 1], [1, -1], [2, -1]]], dtype=np.int32)
        self.manager.submit_feedback(feedback, generation=frame.generation, fetch_rows=rows)
        self.manager.wait_pending()
        self.assertEqual(frame.rows[0].lru.unique[0], 2)
        self.assertEqual(self.manager.miss_stats.misses.tolist(), [4])
        self.assertEqual(self.manager.miss_stats.references.tolist(), [2])
        self.assertEqual(self.manager.miss_stats.invalid, 0)
        self.assertEqual(self.manager.miss_stats.get_counts, {1: [4]})

    def test_feedback_mtp_dedup_and_abc_exclusion(self):
        frame = self.prepare()
        tail = self.hot.history_capacity
        feedback = np.array([[[1, 1], [2, 1], [tail, -1], [2, tail + 1]]], dtype=np.int32)
        self.manager.submit_feedback(feedback, generation=frame.generation, miss_counts=np.array([[1, 1]], np.int32))
        self.manager.wait_pending()
        row = frame.rows[0]
        self.assertEqual(row.lru.unique[0], 2)
        self.assertEqual(row.lru.active, self.hot.history_capacity)
        self.assertEqual(self.manager.miss_stats.references.tolist(), [2])
        self.assertEqual(self.manager.miss_stats.misses.tolist(), [1])
        self.assertTrue(np.all(row.lru.hot_snapshot[0] < tail))
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self.submit(1)
        self.prepare(4100)
        self.submit(tail)
        self.manager.wait_pending()
        self.assertEqual(self.manager.miss_stats.references.tolist(), [2])

    def test_invalid_feedback_never_changes_lru(self):
        row = self.prepare().rows[0]
        frames = (
            (
                ((row.lru, self.hot.slot_capacity, self.hot.history_capacity, self.hot.history_capacity),),
                np.array([0, 4], np.int32),
            ),
        )
        before = row.lru.version
        for value, counts, error in ((0, [[-1, -1]], RuntimeError), (self.hot.slot_capacity, [[0, 0]], ValueError)):
            with self.assertRaises(error):
                self.manager._commit(frames, np.full((1, 4, 1), value, np.int32), None, np.array(counts))
        self.assertEqual(row.lru.version, before)

    def test_fused_device_failure_prevents_lru_commit(self):
        row = self.prepare().rows[0]
        frames = ((((row.lru, self.hot.slot_capacity),), np.array([0, 4], np.int32)),)
        before = row.lru.version
        with self.assertRaisesRegex(RuntimeError, "LIFetch Index update failed"):
            self.manager._commit(frames, np.zeros((1, 4, 1), np.int32), None, device_errors=np.array([[2]], np.int32))
        self.assertEqual(row.lru.version, before)

    def test_fused_slot_feedback_does_not_invent_selected_miss_counts(self):
        frame = self.prepare()
        feedback = np.array([[[1, 2], [2, 1], [8192, -1], [1, 2]]], dtype=np.int32)
        with patch.object(self.manager.miss_stats, "record") as record:
            self.manager.submit_feedback(feedback, generation=frame.generation)
            self.manager.wait_pending()
        self.assertEqual(frame.rows[0].lru.unique[0], 2)
        self.assertTrue(np.all(frame.rows[0].lru.hot_snapshot[0] < self.hot.history_capacity))
        record.assert_not_called()

    def test_fifo_feedback_does_not_block_next_prepare(self):
        row = self.prepare().rows[0]
        entered, proceed = threading.Event(), threading.Event()
        original = self.manager._commit

        def delayed(*args):
            entered.set()
            if not proceed.wait(5):
                raise RuntimeError("test feedback stalled")
            return original(*args)

        try:
            with patch.object(self.manager, "_commit", side_effect=delayed):
                self.submit(1)
            self.assertTrue(entered.wait(2))
            with ThreadPoolExecutor(1) as executor:
                job = executor.submit(self.prepare, 4100)
                try:
                    job.result(timeout=1)
                finally:
                    proceed.set()
            self.submit(2)
            self.manager.wait_pending()
            self.assertEqual(row.lru.tail[0], 2)
            self.assertEqual(row.lru.previous[0, 2], 1)
        finally:
            proceed.set()

    def test_reorder_release_and_stale_feedback(self):
        frame = self.prepare(ids=("b", "a"))
        a_lru = frame.rows[1].lru
        frame = self.prepare(4100, ("a", "b"))
        self.assertIs(frame.rows[0].lru, a_lru)
        generation = frame.generation
        self.writer.release_requests(("a",))
        self.assertNotIn(("a", 0), self.manager.owners)
        self.assertNotIn("a", self.hot.owners)
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            self.manager.submit_feedback(np.zeros((1, 8, 1), np.int32), generation=generation)

    def test_summary_one_line_per_card(self):
        stats = type(self.manager.miss_stats)(2, "DP0 TP3 NPU3")
        stats.record(np.array([[1, 0], [0, 1]]), np.array([[2, 2], [4, 4]]))
        with self.assertLogs("vllm", level="INFO") as logs:
            stats.flush()
        self.assertEqual(len(logs.output), 1)
        self.assertIn("overall_miss_pct=16.667", logs.output[0])
        self.assertIn("request_rounds=2", logs.output[0])


class TestSlotLRU(unittest.TestCase):
    def test_grow_preserves_order_and_supports_large_slots(self):
        row = SlotLRU()
        row.resize(131072, 128)
        row.commit(np.array([0, 1], dtype=np.int32))
        row.resize(131072, 130560)
        self.assertEqual(row.slots[0, :3].tolist(), [2, 3, 4])
        self.assertEqual(row.next.dtype, np.int32)
        self.assertEqual(row.bitmap.shape, (1, 4096))
        row.commit(np.array([130000], dtype=np.int32))
        self.assertEqual(row.tail[0], 130000)
        self.assertEqual(row.next[0, 129999], 130001)
        row.resize(131072, 0)
        self.assertTrue(np.all(row.slots == -1))
        self.assertFalse(np.any(row.bitmap))


if __name__ == "__main__":
    unittest.main()
