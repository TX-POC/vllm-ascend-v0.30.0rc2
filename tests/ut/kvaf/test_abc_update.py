# SPDX-License-Identifier: Apache-2.0
"""Batched ABC updates preserve device-owned history and inactive request rows."""

import unittest
from unittest.mock import patch

import torch
from kvip.cache.abc import update_abc

from . import test_prefill as fixture


class TestABC(unittest.TestCase):
    def test_wrap_full_empty_and_nonidentity_groups(self):
        forward = torch.full((3, 5, 2048), 99, dtype=torch.int32)
        reverse = forward.clone()
        expected_forward, expected_reverse = forward.clone(), reverse.clone()
        desc = torch.tensor(
            [[[2, 4, 350, 734], [2, 1, 768, 768]], [[0, 3, 383, 390], [0, 2, 128, 140]]], dtype=torch.int32
        )
        for group in desc:
            for layer, owner, begin, end in group.tolist():
                for token in range(begin, end):
                    expected_forward[layer, owner, token] = 1024 + token % 384
                expected_reverse[layer, owner, 1024:1408] = -1
                for token in range(begin, end):
                    expected_reverse[layer, owner, 1024 + token % 384] = token
        update_abc(forward, reverse, desc, 2, 1024, 384)
        torch.testing.assert_close(forward, expected_forward)
        torch.testing.assert_close(reverse, expected_reverse)


class TestBatchedPrepare(unittest.TestCase):
    setUp = fixture.TestWindow.setUp
    run_chunk = fixture.TestWindow.run_chunk

    def test_stable_decode_does_not_reupload_tables(self):
        self.run_chunk([("a", 0, 260, 260)])
        self.run_chunk([("b", 0, 260, 260)])
        self.run_chunk([("a", 260, 4, 260), ("b", 260, 4, 260)])
        calls = []
        original = torch.Tensor.copy_

        def copy(target, source, *args, **kwargs):
            calls.append(target.data_ptr())
            return original(target, source, *args, **kwargs)

        with patch.object(torch.Tensor, "copy_", copy):
            self.run_chunk([("a", 264, 4, 260), ("b", 264, 4, 260)])
        self.assertNotIn(self.hot.resident_tables.data_ptr(), calls)
        self.assertNotIn(self.hot.inputs["blocks"].data_ptr(), calls)
        for target in (self.hot.spare_slots, self.hot.spare_bitmap, self.hot.abc_descriptors):
            self.assertEqual(calls.count(target.data_ptr()), 1)
        # Reordering active requests must update request-row page tables.
        with patch.object(torch.Tensor, "copy_", copy):
            self.run_chunk([("b", 268, 4, 260), ("a", 268, 4, 260)])
        self.assertEqual(calls.count(self.hot.resident_tables.data_ptr()), 1)
        self.assertEqual(calls.count(self.hot.inputs["blocks"].data_ptr()), 1)


if __name__ == "__main__":
    unittest.main()
