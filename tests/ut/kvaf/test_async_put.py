# SPDX-License-Identifier: Apache-2.0
"""Exercise live ABC source ownership with a deliberately delayed CPU Put."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from . import test_prefill as fixture
from .test_adapter import KVAFConfig


class TestAsyncPut(unittest.TestCase):
    tail_tokens = 640
    put_interval_steps = 10000
    run_chunk = fixture.TestWindow.run_chunk
    assert_prefix = fixture.TestWindow.assert_prefix

    def setUp(self):
        fixture.TestWindow.setUp(self)
        self.release_put = threading.Event()
        self.started = threading.Event()
        self.writer.put_stream = NS(device="cpu")
        self.writer.put_executor = ThreadPoolExecutor(max_workers=1)
        self.fake_npu = patch.object(torch, "npu", NS(Event=Mock, current_stream=lambda device: None), create=True)
        self.fake_npu.start()
        self.addCleanup(self.fake_npu.stop)
        original = self.writer._execute_put

        def delayed(policy, batches, produced):
            self.started.set()
            if not self.release_put.wait(5):
                raise TimeoutError("test did not release Put")
            return original(policy, batches, "decode"), 0.01

        self.writer._background_put = delayed
        self.addCleanup(self.finish)

    def finish(self):
        self.release_put.set()
        # The failure test consumes its future explicitly.
        self.writer.close_put()

    def fill_decode(self):
        self.run_chunk([("a", 0, 260, 260)])
        self.run_chunk([("b", 0, 260, 260)])
        for start in range(260, 516, 4):
            self.run_chunk([("a", start, 4, 260), ("b", start, 4, 260)])
        self.hot.put_interval_steps = 32
        self.hot.put_step = 31

    def test_batching_publication_and_no_wait_until_wrap(self):
        self.fill_decode()
        with patch.object(self.runtime, "batch_put", wraps=self.runtime.batch_put) as put:
            self.run_chunk([("a", 516, 4, 260), ("b", 516, 1, 260)])
            self.assertTrue(self.started.wait(1))
            self.assertEqual(len(self.writer.requests["a"][0].keys), 3)
            self.assertEqual(len(self.writer.requests["a"][0].addresses), 2)
            original = self.writer.poll_put
            waits = []

            def poll(*, wait=False):
                if wait and self.writer.pending_put is not None:
                    waits.append(True)
                    self.release_put.set()
                return original(wait=wait)

            with patch.object(self.writer, "poll_put", side_effect=poll):
                for start in range(520, 896, 4):
                    self.run_chunk([("a", start, 4, 260)])
                    self.assertFalse(waits)
                # This would overwrite token 256, still read by the pending Put.
                self.run_chunk([("a", 896, 4, 260)])
            self.assertEqual(len(waits), 1)
            self.assertEqual(put.call_count, 1)  # Two requests, one layer-group Put.
            self.assertEqual(len(put.call_args.kwargs["keys"]), 2)
            self.assert_prefix("a", 384)
            self.assert_prefix("b", 384)
            self.assertEqual(len(self.writer.requests["a"][0].addresses), 3)

    def test_release_drains_before_recycling(self):
        self.fill_decode()
        self.run_chunk([("a", 516, 4, 260)])
        self.assertTrue(self.started.wait(1))
        self.release_put.set()
        self.writer.release_requests(("a",))
        self.assertNotIn("a", self.writer.requests)
        self.assertIsNone(self.writer.pending_put)
        self.assert_prefix("b", 256)

    def test_background_failure_does_not_publish(self):
        self.fill_decode()
        with patch.object(self.runtime, "batch_put", side_effect=RuntimeError("transport failed")):
            self.run_chunk([("a", 516, 4, 260)])
            self.release_put.set()
            with self.assertRaisesRegex(RuntimeError, "transport failed"):
                self.writer.poll_put(wait=True)
        self.assertEqual(len(self.writer.requests["a"][0].addresses), 2)
        self.assertEqual(len(self.writer.requests["a"][0].keys), 3)
        self.writer.pending_put = None  # Fatal path: keep ownership, shut down executor.

    def test_rejected_drafts_do_not_advance_submission_boundary(self):
        self.fill_decode()
        self.hot.put_interval_steps = 10000
        self.run_chunk([("a", 516, 4, 260)])
        # Simulate the existing deferred MTP correction before Put selection.
        self.requests["a"].num_computed_tokens = 510
        self.hot.put_interval_steps = 32
        self.hot.put_step = 31
        self.writer.put_requests(self.requests, {"a": 4})
        self.assertIsNone(self.writer.pending_put)
        self.assertEqual(len(self.writer.requests["a"][0].keys), 2)

    def test_mixed_prefill_drains_decode_before_reusing_native_put(self):
        self.fill_decode()
        self.run_chunk([("a", 516, 4, 260)])
        self.assertTrue(self.started.wait(1))
        self.release_put.set()
        # A newly admitted chunk uses the unchanged synchronous prefill path.
        self.writer.release_requests(("b",))
        self.run_chunk([("a", 520, 4, 260), ("c", 0, 129, 260)])
        self.assertIsNone(self.writer.pending_put)
        self.assert_prefix("a", 384)
        self.assert_prefix("c", 128)

    def test_capacity_budget_and_validation(self):
        self.assertEqual(KVAFConfig().abc_tokens(4), 640)
        self.assertEqual(KVAFConfig().abc_tokens(1), 512)
        for field in ("put_interval_steps", "put_overlap_steps"):
            with self.assertRaises(ValueError):
                KVAFConfig(**{field: 0})


if __name__ == "__main__":
    unittest.main()
