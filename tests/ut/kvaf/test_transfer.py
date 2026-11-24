# SPDX-License-Identifier: Apache-2.0
"""Put failure/publication checks through the actual staging/ABC service path."""

import unittest
from unittest.mock import patch

from . import test_prefill as fixture


class TestKVAFPut(unittest.TestCase):
    setUp = fixture.TestWindow.setUp
    run_chunk = fixture.TestWindow.run_chunk
    assert_prefix = fixture.TestWindow.assert_prefix

    def test_failed_put_never_publishes_addresses(self):
        with (
            patch.object(self.runtime, "batch_put", side_effect=RuntimeError("Put failed")),
            self.assertRaisesRegex(RuntimeError, "Put failed"),
        ):
            self.run_chunk([("a", 0, 128, 256)])
        self.assertEqual(self.writer.requests["a"][0].addresses, [])
        self.assertEqual(self.writer.put_pages, 0)
        # Native completion is uncertain after failure: never retry/recycle its keys.
        self.assertFalse(self.runtime.pages)

    def test_invalid_address_table_never_published_and_keys_can_be_released(self):
        with (
            patch.object(
                self.runtime, "access_table_view", return_value=dict(addresses=[0], valid=[1], lengths=[128 * 656])
            ),
            self.assertRaisesRegex(RuntimeError, "address table is invalid"),
        ):
            self.run_chunk([("a", 0, 128, 128)])
        self.assertEqual(self.writer.requests["a"][0].addresses, [])
        self.assertEqual(len(self.runtime.pages), 1)
        self.writer.release_requests(("a",))
        self.assertFalse(self.runtime.pages)

    def test_ddr_full_does_not_publish_extra_pages(self):
        pool_type = type(self.writer.slots[(656, 128)])
        self.writer.slots[(656, 128)] = pool_type(1)
        with self.assertRaisesRegex(RuntimeError, "slots exhausted"):
            self.run_chunk([("a", 0, 128, 256)])
        self.assertEqual(len(self.writer.requests["a"][0].addresses), 0)
        self.assertEqual(self.writer.put_pages, 0)

    def test_failed_verification_preserves_keys_for_cleanup(self):
        self.writer.verify = lambda *args: (_ for _ in ()).throw(RuntimeError("readback mismatch"))
        with self.assertRaisesRegex(RuntimeError, "readback mismatch"):
            self.run_chunk([("a", 0, 256, 256)])
        self.writer.release_requests(("a",))
        self.assertFalse(self.runtime.pages)

    def test_request_release_reuses_slots_without_stale_prefix(self):
        self.run_chunk([("a", 0, 256, 256)])
        keys = set(self.runtime.pages)
        self.writer.release_requests(("a",))
        self.run_chunk([("b", 0, 256, 256)])
        self.assertEqual(set(self.runtime.pages), keys)
        self.assert_prefix("b", 256)

    def test_decode_put_only_at_stable_block_boundary(self):
        self.run_chunk([("a", 0, 260, 260)])
        self.run_chunk([("a", 260, 4, 260)])
        with patch.object(
            self.runtime, "prepare_access_table", side_effect=AssertionError("unexpected catalog lookup")
        ):
            self.run_chunk([("a", 264, 4, 260)])
        self.assertEqual(self.writer.phase_pages, {"prefill": 2, "decode": 0})
        for start in range(268, 516, 4):
            self.run_chunk([("a", start, 4, 260)])
        self.assertEqual(self.writer.phase_pages, {"prefill": 2, "decode": 1})
        self.assert_prefix("a", 384)

    def test_short_logical_table_rejected_before_put(self):
        state_module = __import__(self.writer.__class__.__module__, fromlist=["RequestCacheState"])
        state = state_module.RequestCacheState(128)
        with self.assertRaisesRegex(ValueError, "required=2 available=1"):
            state.prepare((0,), 0, 256)
        self.assertFalse(self.runtime.pages)


if __name__ == "__main__":
    unittest.main()
