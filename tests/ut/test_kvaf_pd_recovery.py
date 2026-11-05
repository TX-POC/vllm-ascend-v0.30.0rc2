# SPDX-License-Identifier: Apache-2.0
"""Completion ACK recovery with real address/transaction code and no sockets."""

import unittest
from collections import Counter
from unittest.mock import patch

import test_kvaf_pd_flow as flow


class _DirectServer:
    def __init__(self, host, port, dispatch, **kwargs):
        self.port = 21000

    def close(self):
        pass


class TestPDCompleteRecovery(unittest.TestCase):
    def setUp(self):
        self.fixture = flow.TestPDFlow()
        self.calls, self.dispatched = Counter(), Counter()
        self.before = lambda message: None
        self.after = lambda message, result: None

        def call(endpoint, message, **kwargs):
            self.calls[message["op"]] += 1
            self.before(message)
            result = self.fixture.receiver.dispatch(message)
            self.dispatched[message["op"]] += 1
            self.after(message, result)
            return result

        for target, name, replacement in (
            (flow.RECEIVER, "ControlServer", _DirectServer),
            (flow.SENDER, "call", call),
        ):
            patcher = patch.object(target, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.seed(300)

    def assert_finished(self):
        self.fixture.assert_payload(300)
        for sender in self.fixture.senders:
            self.assertEqual(sender.get_finished({"r"}), {"r"})
        self.assertFalse(any(source.active is not None for source in self.fixture.sdk.sources.values()))

    def test_successful_completion_ack_avoids_terminal_record_growth(self):
        self.fixture.chunk(300)
        self.assert_finished()
        self.assertTrue(all(job.acknowledged for job in self.fixture.receiver.jobs.values()))
        self.fixture.receiver.release({"r"})
        self.assertFalse(self.fixture.receiver.jobs)
        self.assertFalse(self.fixture.receiver.terminal_jobs)

    def test_complete_reply_loss_queries_original_job_without_replaying_pull(self):
        lost = False

        def after(message, result):
            nonlocal lost
            if message["op"] == "complete" and not lost:
                lost = True
                raise TimeoutError("complete executed but its reply was lost")

        self.after = after
        self.fixture.chunk(300)
        self.assert_finished()
        self.assertEqual(self.calls["status"], 1)
        self.assertEqual(self.calls["complete"], len(self.fixture.receiver.jobs))
        self.assertEqual(self.calls["pull"], sum(len(job.completed) for job in self.fixture.receiver.jobs.values()))
        self.assertEqual(self.fixture.receiver.get_finished(), {"r"})

    def test_complete_delivery_loss_retries_only_idempotent_completion(self):
        lost = False

        def before(message):
            nonlocal lost
            if message["op"] == "complete" and not lost:
                lost = True
                raise ConnectionError("complete never reached D")

        self.before = before
        self.fixture.chunk(300)
        self.assert_finished()
        self.assertEqual(self.calls["status"], 1)
        self.assertEqual(self.calls["complete"], self.dispatched["complete"] + 1)
        self.assertEqual(self.calls["pull"], sum(len(job.completed) for job in self.fixture.receiver.jobs.values()))

    def test_native_release_before_lost_reply_recovery_uses_terminal_record(self):
        released = False
        recovered_terminal = False

        def after(message, result):
            nonlocal released, recovered_terminal
            receiver = self.fixture.receiver
            if message["op"] == "complete" and not released and receiver.requests["r"]["published"]:
                released = True
                receiver.release({"r"})
                self.assertIn(message["job"], receiver.terminal_jobs)
                raise TimeoutError("native Decode finished before complete reply arrived")
            if message["op"] == "status" and released:
                recovered_terminal = message["job"] not in receiver.jobs and result["done"]

        self.after = after
        self.fixture.chunk(300)
        for sender in self.fixture.senders:
            self.assertEqual(sender.get_finished({"r"}), {"r"})
        self.assertTrue(released)
        self.assertTrue(recovered_terminal)
        self.assertFalse(self.fixture.receiver.jobs)
        self.assertFalse(self.fixture.receiver.terminal_jobs)

    def test_terminal_identity_duplicate_ack_and_detach_cleanup(self):
        def before(message):
            if message["op"] == "ack_complete":
                raise ConnectionError("cleanup acknowledgement unavailable")

        self.before = before
        self.fixture.chunk(300)
        self.assert_finished()
        receiver = self.fixture.receiver
        receiver.release({"r"})
        key, terminal = next(iter(receiver.terminal_jobs.items()))
        message = dict(job=key, peer=receiver.identity, producer=terminal[0], generation=terminal[1])
        with self.assertRaisesRegex(ValueError, "stale D worker"):
            receiver.dispatch(dict(message, op="status", peer="old-worker"))
        with self.assertRaisesRegex(ValueError, "stale source/job"):
            receiver.dispatch(dict(message, op="status", generation="old-generation"))
        self.assertTrue(receiver.dispatch(dict(message, op="status"))["done"])
        self.assertTrue(receiver.dispatch(dict(message, op="complete")))
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            receiver.dispatch(dict(message, op="pull"))
        self.assertTrue(receiver.dispatch(dict(message, op="ack_complete")))
        self.assertTrue(receiver.dispatch(dict(message, op="ack_complete")))
        self.assertNotIn(key, receiver.terminal_jobs)
        for sender in self.fixture.senders:
            sender.close()
        self.assertFalse(receiver.terminal_jobs)


class TestPDCompleteStatusBoundary(unittest.TestCase):
    def test_incomplete_status_cannot_promote_completion(self):
        sender = flow.SENDER.Sender.__new__(flow.SENDER.Sender)
        operations = []

        def call(endpoint, message):
            operations.append(message["op"])
            if message["op"] == "complete":
                raise TimeoutError("completion delivery unknown")
            return {"done": False}

        sender._call = call
        with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
            sender._complete_job("D:1", "original-job", "original-generation")
        self.assertEqual(operations, ["complete", "status", "complete", "status"])


if __name__ == "__main__":
    unittest.main()
