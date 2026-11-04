# SPDX-License-Identifier: Apache-2.0
"""PD singleton DDR layout and native Get/tail continuation on CPU tensors."""

import importlib
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS

import test_kvaf_adapter as helpers
import torch
from test_kvaf_grouped_get import Runtime
from test_kvaf_prefill import Manager, Writer

backend = importlib.import_module(f"{helpers.PACKAGE.__name__}.pd.backend")


class TestPDBackend(unittest.TestCase):
    def setUp(self):
        self.pool = backend.create_decode_cache(("l0", "l1"), 2, 128, 8192, "cpu", native=lambda *a: a[14].zero_())
        self.hot = self.pool.policy
        self.runtime = Runtime(1)
        layouts = tuple(
            replace(
                helpers.layout(656, name),
                planes=(replace(helpers.layout(656, name).planes[0], shape=tuple(self.hot.hot[i].shape)),),
            )
            for i, name in enumerate(self.pool.names)
        )
        self.writer = Writer(
            {(656, 128): self.runtime},
            {(656, 128): 256},
            layouts,
            dict(zip(self.pool.names, self.hot.hot)),
        )
        self.manager = Manager(self.writer)
        self.writer.feedback = self.manager
        self.manager.bind_device_pool(self.pool)
        self.addCleanup(self.manager.close)
        self.requests = {}

    @staticmethod
    def values(tokens, layer):
        return (tokens + 29 * layer).remainder(251).to(torch.int8)

    def receive(self, tokens, *, bind_tail=True):
        self.requests["r"] = NS(num_computed_tokens=tokens, num_prompt_tokens=tokens, block_ids=(list(range(64)),))
        self.writer.prepare_requests(self.requests, {"r": 4})
        for i, (group, state) in enumerate(zip(self.writer.groups, self.writer.requests["r"])):
            # Emulate already committed CPU Pull payloads. SDK transaction and
            # exact-cover tests live in test_kvaf_pd.py; use the real publisher.
            keys = self.writer.slots[group.geometry].take(tokens // 128)
            for q, key in enumerate(keys):
                self.runtime.pages[key] = (
                    self.values(torch.arange(q * 128, (q + 1) * 128), i)[:, None]
                    .expand(-1, 656)
                    .reshape(-1)
                    .clone()
                    .view(torch.uint8)
                )
            state.keys.extend(keys)
            self.writer._publish_addresses(group, keys, [(state, len(keys))])
            if tokens % 128 and bind_tail:
                tail = torch.full((128, 656), 120, dtype=torch.int8)
                tail[: tokens % 128] = self.values(torch.arange(tokens // 128 * 128, tokens), i)[:, None]
                self.hot.bind_received_tail("r", self.pool.names[i], tail, tokens)

    def prepare(self):
        self.manager.prepare(("r",), {"r": 4}, decode=True, padded_requests=1)

    def test_singleton_geometry_does_not_change_global_grouping(self):
        self.assertEqual(self.pool.groups, (("l0",), ("l1",)))
        self.assertEqual(self.hot.group_sizes, (1, 1))
        self.assertIsNone(self.hot.group_context)
        self.assertEqual(self.hot.addresses.shape[1], 2)
        self.assertEqual([g.geometry for g in self.writer.groups], [(656, 128)] * 2)
        native_layout = helpers.layout_module
        self.assertEqual(native_layout.GROUP_WIDTH, 4)
        names = ("a", "b", "c", "d")
        context = {name: NS(impl=NS(skip_topk=i > 0)) for i, name in enumerate(names)}
        self.assertEqual(native_layout.index_groups(names, context), (names,))

    def test_tail_stays_in_receive_buffer_until_first_decode_prepare(self):
        for cache in self.hot.hot:
            cache.fill_(119)
        self.receive(300)
        for cache in self.hot.hot:
            self.assertTrue(torch.all(cache == 119))
        self.prepare()
        owner = self.hot.owners["r"]
        start = owner * self.hot.slot_capacity + self.hot.history_capacity + 256
        for i, cache in enumerate(self.hot.hot):
            rows = cache.view(-1, 656)
            expected = self.values(torch.arange(256, 300), i)
            torch.testing.assert_close(rows[start : start + 44, 0], expected)
            self.assertTrue(torch.all(rows[start + 44 : start + 128] == 119))
            self.assertEqual(self.hot.addresses[owner, i, :2].tolist(), self.writer.requests["r"][i].addresses)

    def test_missing_tail_stops_first_decode_before_using_uninitialized_abc(self):
        self.receive(300, bind_tail=False)
        with self.assertRaisesRegex(RuntimeError, "matching completed PD tail"):
            self.prepare()

    def test_get_uses_per_layer_directory_and_656_byte_offsets(self):
        self.receive(2300)
        self.prepare()
        owner = self.hot.owners["r"]
        token, slot = 37, 7

        def native_get(*args):
            layer, count, _, qcap = args[20:24]
            self.assertEqual(count, 2)
            args[3][owner, token] = slot
            args[14].zero_()
            args[14][-1] = 1
            args[15][0] = (owner * count + layer) * qcap + token // 128
            args[16][0] = token % 128 * 656
            args[17][0] = owner * self.hot.slot_capacity + slot

        self.hot.prepare_op = native_get
        topk = torch.full((4, 1, 2048), token, dtype=torch.int32)
        topk[:, :, -1] = -1
        metadata = NS(num_prefills=0, cum_query_lens=torch.tensor([4]))
        self.runtime.reads.clear()
        for i in range(2):
            mapped = self.hot.run(i, topk, metadata)
            self.assertTrue(torch.all(mapped[:, :, :-1] == slot))
            self.assertTrue(torch.all(mapped[:, :, -1] == -1))
            row = self.hot.hot[i].view(-1, 656)[owner * self.hot.slot_capacity + slot]
            self.assertTrue(torch.all(row == self.values(torch.tensor(token), i)))
        self.assertEqual(self.runtime.reads, [1, 1])

    def test_received_tail_continues_mtp_writes_and_publishes_only_stable_full_block(self):
        self.receive(300)
        for start in range(300, 520, 4):
            self.requests["r"].num_computed_tokens = start
            self.writer.prepare_requests(self.requests, {"r": 4})
            self.prepare()
            for i, cache in enumerate(self.hot.hot):
                slots = self.hot.write_slots_for(i, 0)[:4]
                cache.view(-1, 656)[slots] = self.values(torch.arange(start, start + 4), i)[:, None]
            self.hot.after_forward(self.requests, {"r": 4})
        for i, state in enumerate(self.writer.requests["r"]):
            self.assertEqual(len(state.keys), 3)
            result = torch.cat([self.runtime.pages[key] for key in state.keys]).view(-1, 656)
            torch.testing.assert_close(result[:, 0].view(torch.int8), self.values(torch.arange(384), i))

    def test_release_drops_backend_receive_references_and_reclaims_same_key_pool(self):
        self.receive(300)
        self.prepare()
        self.writer.release_requests(("r",))
        self.assertFalse(self.hot._received_tails)
        self.assertFalse(self.runtime.pages)
        self.assertFalse(self.writer.slots[(656, 128)].references)

    def test_four_layer_group_cannot_bind_to_pd_backend(self):
        self.writer.groups[0] = replace(
            self.writer.groups[0], layout=replace(self.writer.groups[0].layout, layer_names=("l0", "l1"))
        )
        with self.assertRaisesRegex(ValueError, "one distinct Target layer"):
            self.hot.bind(self.manager)


if __name__ == "__main__":
    unittest.main()
