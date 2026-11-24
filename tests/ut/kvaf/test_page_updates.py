# SPDX-License-Identifier: Apache-2.0
"""Incremental metadata publication preserves resident/permuted history."""

import unittest
from unittest.mock import patch

import torch
from kvip.cache.page_updates import PageUpdates

from . import test_prefill as fixture


class TestPageUpdates(unittest.TestCase):
    def test_batched_groups_clear_and_fused_append(self):
        addresses = torch.zeros((3, 2, 4), dtype=torch.int64)
        valid = torch.zeros_like(addresses, dtype=torch.uint8)
        maps = torch.full((2, 3, 512), 7, dtype=torch.int32)
        fused = tuple(torch.full((3, 512), 999, dtype=torch.int64) for _ in range(2))
        updater = PageUpdates(torch, addresses, valid, maps, fused=fused, widths=(1, 4))
        rows = [(9, 1000000, 1 * 512 + 128), (22, 2000000, -1)]
        updater.submit(rows, 0)
        self.assertEqual(addresses[1, 0, 1], 1000000)
        self.assertEqual(addresses[2, 1, 2], 2000000)
        self.assertEqual(valid.sum(), 2)
        self.assertTrue((maps[0, 1, 128:256] == -1).all())
        self.assertTrue((maps[0, 1, 256:] == 7).all())
        self.assertTrue((maps[1] == 7).all())
        self.assertTrue((fused[0][1, :128] == 999).all())
        self.assertEqual(fused[1][2, 257], 2000000 + 2624)
        before = addresses.clone()
        updater.submit([], 1)
        self.assertTrue(torch.equal(addresses, before))

    def test_npu_matches_reference(self):
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            self.skipTest("requires Ascend NPU")
        addresses = torch.zeros((3, 2, 4), dtype=torch.int64)
        valid = torch.zeros_like(addresses, dtype=torch.uint8)
        maps = torch.full((2, 3, 512), 7, dtype=torch.int32)
        fused = tuple(torch.full((3, 512), 999, dtype=torch.int64) for _ in range(2))
        actual = [t.npu() for t in (addresses, valid, maps)]
        actual_fused = tuple(t.npu() for t in fused)
        cpu = PageUpdates(torch, addresses, valid, maps, fused=fused, widths=(1, 4))
        npu = PageUpdates(torch, *actual, fused=actual_fused, widths=(1, 4))
        for ring, rows in enumerate(([(9, 1000000, 640), (22, 2000000, -1)], [(10, 3000000, 768)])):
            cpu.submit(rows, ring)
            npu.submit(rows, ring)
        for expected, result in zip((addresses, valid, maps, *fused), (*actual, *actual_fused)):
            self.assertTrue(torch.equal(expected, result.cpu()))


class TestServicePublication(unittest.TestCase):
    setUp = fixture.TestWindow.setUp
    run_chunk = fixture.TestWindow.run_chunk

    def test_upload_only_delta_and_skip_unchanged_history(self):
        self.run_chunk([("a", 0, 256, 512)])
        updater = self.hot.page_updates
        with patch.object(updater, "submit", wraps=updater.submit) as submit:
            self.run_chunk([("a", 256, 256, 512)])
            self.assertEqual(len(submit.call_args.args[0]), 2)
            self.run_chunk([("a", 512, 1, 512)])
            self.assertEqual(len(submit.call_args.args[0]), 2)
            self.run_chunk([("a", 513, 1, 512)])
            self.assertEqual(submit.call_args.args[0], [])
        owner = self.hot.owners["a"]
        self.assertEqual(self.hot.valid[owner, 0].sum(), 4)
        self.hot.release("a")
        self.assertNotIn(("a", 0), self.hot.published_pages)
