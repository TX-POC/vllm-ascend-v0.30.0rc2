# SPDX-License-Identifier: Apache-2.0
"""Actual prefill-to-decode policy with an injected native descriptor producer."""

import unittest
from types import SimpleNamespace as NS

import test_kvaf_prefill as fixture
import torch


class TestHotGet(unittest.TestCase):
    setUp = fixture.TestWindow.setUp
    run_chunk = fixture.TestWindow.run_chunk
    assert_prefix = fixture.TestWindow.assert_prefix

    def prepare(self):
        for start in range(0, 4096, 512):
            self.run_chunk([("a", start, 512, 4096)])
        self.run_chunk([("a", 4096, 4, 4096)])

    def test_registration_and_fp8_view_share_storage(self):
        self.assertEqual(self.runtime.get_counts, [0, 0])
        self.writer.groups[0].source = self.writer.groups[0].source.view(torch.float8_e4m3fn)
        self.hot.bind(self.manager)
        self.assertEqual(self.hot.attention_caches[0].dtype, torch.float8_e4m3fn)
        self.assertEqual(self.hot.attention_caches[0].data_ptr(), self.hot.bytes_outputs[0].data_ptr())

    def test_get_result_and_sfa_index_share_hot_storage(self):
        self.prepare()
        owner = self.hot.owners["a"]
        token, slot = 1234, 3
        calls = []

        def produce(*args):
            calls.append(args)
            args[3][owner, token] = slot
            args[14].fill_(1)
            args[15][0] = owner * self.hot.qcap + token // 128
            args[16][0] = token % 128 * 656
            args[17][0] = owner * self.hot.slot_capacity + slot

        self.hot.prepare_op = produce
        topk = torch.full((4, 1, 2048), token, dtype=torch.int32)
        mapped = self.hot.run(0, topk, NS(num_prefills=0, cum_query_lens=torch.tensor([4])))
        self.assertIs(calls[0][0], topk)
        self.assertTrue(torch.all(mapped == slot))
        self.assertTrue(torch.all(self.hot.hot[0].view(-1, 656)[owner * self.hot.slot_capacity + slot] == token % 127))
        self.assertEqual(self.runtime.get_counts[-1], 1)

    def test_padding_and_query_layout(self):
        self.prepare()
        calls = []
        self.hot.prepare_op = lambda *args: (calls.append(args), args[14].zero_())
        self.manager.prepare(("a",), {"a": 4}, decode=True, padded_requests=2)
        topk = torch.full((8, 1, 2048), -1, dtype=torch.int32)
        mapped = self.hot.run(0, topk, NS(num_prefills=0, cum_query_lens=torch.tensor([4, 8])))
        self.assertTrue(torch.all(mapped[:4] == -1))
        self.assertTrue(torch.all(mapped[4:] == 0))
        self.assertEqual(calls[0][1].tolist(), [4, 4])
        self.assertEqual(calls[0][6].tolist(), [self.hot.slot_capacity, 0])

    def test_released_owner_maps_reset_before_reuse(self):
        self.prepare()
        owner = self.hot.owners["a"]
        self.hot.pool.views["target"].token_to_slot[owner, 100] = 7
        self.writer.release_requests(("a",))
        for start in range(0, 4096, 512):
            self.run_chunk([("b", start, 512, 4096)])
        self.run_chunk([("b", 4096, 4, 4096)])
        self.assertEqual(self.hot.owners["b"], owner)
        self.assertEqual(int(self.hot.pool.views["target"].token_to_slot[owner, 100]), -1)


if __name__ == "__main__":
    unittest.main()
