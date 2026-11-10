# SPDX-License-Identifier: Apache-2.0
"""Native MTP completion/skip proofs with byte-accurate dual-Pull substitutes."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_kvaf_pd_flow as flow
import torch


class TestPDMTPSkipFlow(unittest.TestCase):
    setUp = flow.TestPDFlow.setUp
    memories_for = staticmethod(flow.TestPDFlow.memories_for)
    value = staticmethod(flow.TestPDFlow.value)
    prefill = flow.TestPDFlow.prefill
    seed = flow.TestPDFlow.seed

    def _limits(self, limit):
        self.receiver.draft_limit = limit
        for sender in self.senders:
            sender.draft_limit = limit

    def _metadata(self, sender, tokens, previous, final):
        rank = sender.tp_rank % 4
        blocks = ((rank + 1, 6, 0, 5), (4, rank, 7, 6), (3, 0, 6, 1), (5, rank + 1, 0, 7))
        self.prefill(sender, tokens, blocks)
        return NS(
            requests={
                "r": NS(
                    remote_tp_size=1,
                    remote_host="127.0.0.1",
                    remote_port=self.receiver.server.port,
                    local_block_ids=blocks,
                    local_transed_tokens=previous,
                    local_computed_tokens=tokens,
                    prompt_len=self.receiver.requests["r"]["seed"].tokens,
                    chunk_finish=final,
                )
            },
            generations={"r": "a1"},
        )

    def _chunk(self, tokens, *, previous=0, final=True, skipped=False, graph=False, participants=None):
        participants = self.senders if participants is None else participants
        for sender in participants:
            sender.bind(self._metadata(sender, tokens, previous, final))
            if not graph:
                for layer, memory in sender.memories.items():
                    if memory.offload_main:
                        sender.written(layer)
            if skipped:
                sender.skip_drafting(sender.draft_limit)
                # Native DP dummy execution can call the MTP write hook. Those
                # bytes do not belong to the real request, even after success.
                sender.written("mtp")
                sender.written("mtp")
            else:
                if not graph:
                    sender.written("mtp")
                sender.complete_drafting()
            sender.finalize()
            sender.complete_forward()
        for sender in participants:
            sender.wait_all()

    def _mtp_sequences(self):
        return tuple(
            (
                registration.owner.cpu.per_mr.get(registration.metadata.mr_id, 0),
                registration.owner.npu.per_mr.get(registration.metadata.mr_id, 0),
            )
            for sender in self.senders
            for (_, layer), registration in sender.registrations.items()
            if layer == "mtp"
        )

    def _assert_target_payload(self, tokens):
        targets = {name: memory for name, memory in self.memories.items() if memory.offload_main}
        with patch.object(self, "memories", targets):
            flow.TestPDFlow.assert_payload(self, tokens)

    def test_near_limit_dummy_mtp_hooks_issue_no_dma_and_wait_for_all_tp_members(self):
        self._limits(300)
        self.seed(300)
        self._chunk(300, skipped=True, participants=self.senders[:3])
        self.assertFalse(self.receiver.get_finished())
        self.assertEqual(len(self.receiver.requests["r"]["final"]), 3 * len(self.memories))
        self.assertEqual(self._mtp_sequences(), ((0, 0),) * 4)
        self._chunk(300, skipped=True, participants=self.senders[3:])
        self.assertEqual(self.receiver.get_finished(), {"r"})
        self._assert_target_payload(300)
        self.assertTrue(torch.all(self.memories["mtp"].tensor == -1))
        self.assertEqual(self._mtp_sequences(), ((0, 0),) * 4)
        self.assertEqual(self.native.calls.count("submit"), 4)
        mtp_jobs = [job for job in self.receiver.jobs.values() if job.layer == "mtp"]
        self.assertEqual(len(mtp_jobs), 4)
        self.assertTrue(all(job.finished and job.count == {"cpu": 0, "npu": 0} for job in mtp_jobs))
        for sender in self.senders:
            self.assertEqual(sender.get_finished({"r"}), {"r"})

    def test_actual_early_mtp_chunk_then_native_skip_preserves_only_the_valid_prefix(self):
        self._limits(300)
        self.seed(300)
        self._chunk(128, final=False)
        sequences = self._mtp_sequences()
        self.assertTrue(any(npu for _, npu in sequences))
        self.assertFalse(self.receiver.get_finished())
        self._chunk(300, previous=128, skipped=True)
        self.assertEqual(self.receiver.get_finished(), {"r"})
        self.assertEqual(self._mtp_sequences(), sequences)
        self._assert_target_payload(300)
        memory = self.memories["mtp"]
        for plane, (name, tensor) in enumerate(memory.planes.items()):
            table = self.blocks[memory.groups[name]]
            for token in range(128):
                self.assertTrue(torch.all(tensor[table[0], token] == self.value(4, plane, token)))
            for token in range(128, 300):
                self.assertTrue(torch.all(tensor[table[token // 128], token % 128] == -1))

    def test_real_native_graph_proposal_completes_mtp_without_python_write_hooks(self):
        self._limits(512)
        self.seed(129)
        self._chunk(129, graph=True)
        self.assertEqual(self.receiver.get_finished(), {"r"})
        flow.TestPDFlow.assert_payload(self, 129)
        self.assertTrue(any(npu for _, npu in self._mtp_sequences()))
        self.assertFalse(any(sender.draft_written for sender in self.senders))
        self.assertTrue(all(sender.draft_completed for sender in self.senders))

    def test_sender_rejects_skipping_a_prompt_that_native_decode_can_draft(self):
        self._limits(512)
        self.seed(129)
        sender = self.senders[0]
        sender.bind(self._metadata(sender, 129, 0, True))
        with self.assertRaisesRegex(RuntimeError, "future D draft"):
            sender.skip_drafting(512)
        self.assertFalse(sender.draft_skipped)
        self.assertFalse(sender.dispatched)
        self.assertEqual(self._mtp_sequences(), ((0, 0),) * 4)

    def test_missing_required_mtp_is_not_promoted_by_target_graph_fallback(self):
        self._limits(512)
        self.seed(129)
        sender = self.senders[0]
        sender.bind(self._metadata(sender, 129, 0, True))
        with self.assertRaisesRegex(RuntimeError, "required PD layers.*mtp"):
            sender.complete_forward()
        sender.wait_all()
        self.assertFalse(self.receiver.get_finished())
        self.assertEqual(self._mtp_sequences(), ((0, 0),) * 4)
        self.assertIn("r", sender.forward_pending)

    def test_receiver_independently_rejects_skip_for_a_fit_prompt_and_for_target_kv(self):
        self._limits(512)
        self.seed(129)
        sender = self.senders[0]
        for layer in ("mtp", "l0"):
            with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "uninitialized"):
                self.receiver._plan(
                    dict(
                        request="r",
                        generation="a1",
                        producer=sender.producer,
                        cohort="P:0:0",
                        layer=layer,
                        member=0,
                        ratio=4,
                        tokens=129,
                        first=0,
                        final=True,
                        tables=sender.memories[layer].tables(self.blocks),
                        draft_skipped=True,
                    )
                )
        self.assertFalse(self.receiver.jobs)
        self.assertFalse(self.receiver.get_finished())

    def test_p_d_drafter_limit_mismatch_is_rejected_before_memory_registration(self):
        self.receiver.draft_limit = 512
        sender = self.senders[0]
        sender.draft_limit = 300
        before = len(self.sdk.sources), len(self.receiver.sources), dict(sender.registrations)
        with self.assertRaisesRegex(ValueError, "drafter limits"):
            sender._register_peer(self.endpoint)
        self.assertEqual((len(self.sdk.sources), len(self.receiver.sources), sender.registrations), before)

    def test_skip_cannot_bypass_contributor_or_progress_validation(self):
        self._limits(300)
        self.seed(300)
        sender = self.senders[0]
        message = dict(
            request="r",
            generation="a1",
            producer=sender.producer,
            cohort="P:0:0",
            layer="mtp",
            member=0,
            ratio=4,
            tokens=300,
            first=0,
            final=True,
            tables=sender.memories["mtp"].tables(self.blocks),
            draft_skipped=True,
        )
        for invalid in (
            dict(member=4),
            dict(ratio=0),
            dict(first=-1),
            dict(tokens=True),
            dict(first=3),
            dict(draft_skipped="true"),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.receiver._plan(dict(message, **invalid))
        self.assertFalse(self.receiver.jobs)
        self.assertFalse(self.receiver.get_finished())


if __name__ == "__main__":
    unittest.main()
