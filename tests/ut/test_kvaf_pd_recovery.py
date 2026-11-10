# SPDX-License-Identifier: Apache-2.0
"""Completion ACK recovery with real address/transaction code and no sockets."""

import unittest
from collections import Counter
from dataclasses import replace
from unittest.mock import Mock, patch

import test_kvaf_pd_flow as flow


class _DirectServer:
    def __init__(self, host, port, dispatch, **kwargs):
        self.port = 21000

    def close(self):
        pass


class TestPDSourceInitializationRecovery(unittest.TestCase):
    def setUp(self):
        self.fixture = flow.TestPDFlow()
        self.before_call = lambda endpoint, message: None
        self.after_call = lambda endpoint, message, result: None
        self.before_register = lambda source: None
        self.receivers, self.senders, self.created = {}, [], []

        def call(endpoint, message, **kwargs):
            self.before_call(endpoint, message)
            receiver = self.receivers.get(endpoint, self.fixture.receiver)
            result = receiver.dispatch(message)
            self.after_call(endpoint, message, result)
            return result

        for target, name, replacement in (
            (flow.RECEIVER, "ControlServer", _DirectServer),
            (flow.SENDER, "call", call),
        ):
            patcher = patch.object(target, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.fixture.setUp(tp_size=1)
        self.addCleanup(self.fixture.doCleanups)
        self.bootstrap = Mock()

        def source_factory(*, npu=False, **kwargs):
            source = flow.Source(self.fixture.sdk, npu=npu)
            self.created.append(source)
            register = source.register

            def register_source(*args, **kwargs):
                self.before_register(source)
                return register(*args, **kwargs)

            source.register = Mock(side_effect=register_source)
            source.shutdown = Mock(wraps=source.shutdown)
            return source

        for name, replacement in (
            ("CpuPullHbmSource", lambda **kwargs: source_factory(**kwargs)),
            ("NpuPullHbmSource", lambda **kwargs: source_factory(npu=True, **kwargs)),
            ("NpuRuntime", Mock(return_value=self.bootstrap)),
        ):
            patcher = patch.object(self.fixture.sdk, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.before_call = lambda endpoint, message: None
        self.after_call = lambda endpoint, message, result: None
        for source in self.created:
            source.shutdown.side_effect = None
        for sender in self.senders:
            # Tests can model an old D without the new fence. The direct test
            # server has no outstanding handlers once assertions are finished.
            sender._cleanup_barriers.update(sender.peers)
            sender.close()

    def sender(self, *, endpoints=None):
        f = self.fixture
        sender = flow.SENDER.Sender(
            f.sdk,
            f.memories_for(decode=False),
            config=replace(f.senders[0].config, decode_endpoints=endpoints or (f.endpoint,)),
            device_id=0,
            producer="P:init:0",
            host="127.0.0.1",
            activate=lambda: None,
            stream_factory=lambda: None,
            event_factory=lambda: None,
            external_id=lambda value: value,
            tp_rank=0,
            tp_size=8,
            dp_rank=0,
        )
        self.senders.append(sender)
        return sender

    def fail_second_cpu_register(self):
        def fail(source):
            if not source.npu:
                raise RuntimeError("Offload: source CTP query failed")

        self.before_register = fail

    def test_constructor_has_no_sdk_side_effects_and_cpu_failure_rolls_back(self):
        sender = self.sender()
        self.assertFalse(self.created)
        self.bootstrap.initialize.assert_not_called()
        self.fail_second_cpu_register()
        with self.assertLogs("vllm", level="INFO") as logs, self.assertRaisesRegex(RuntimeError, "CTP query failed"):
            sender.initialize()
        text = "\n".join(logs.output)
        for expected in ("mrs_per_path=5", "layer=<group>", "stage=cpu_register", "completed=0", "attempted_mrs=5"):
            self.assertIn(expected, text)
        self.assertTrue(sender.closed)
        self.assertFalse(sender.registrations)
        self.assertFalse(any(key[0] == sender.producer for key in self.fixture.receiver.sources))
        for source in self.created:
            source.shutdown.assert_called_once()
        self.bootstrap.shutdown.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "cannot repeat"):
            sender.initialize()

    def test_npu_registration_and_listen_failures_report_exact_stage(self):
        for stage in ("npu_register",):
            with self.subTest(stage=stage):
                sender = self.sender()
                offset = len(self.created)

                def fail(source, offset=offset):
                    if source is self.created[offset + 1]:
                        raise RuntimeError("NPU setup failed")

                self.before_register = fail
                with self.assertLogs("vllm", level="ERROR") as logs, self.assertRaisesRegex(RuntimeError, "NPU setup"):
                    sender.initialize()
                self.assertIn(f"stage={stage}", "\n".join(logs.output))
                self.assertTrue(sender.closed)

    def test_lost_register_reply_detaches_remote_import_before_local_shutdown(self):
        def lose_reply(endpoint, message, result):
            if message["op"] == "register":
                self.assertIn((message["producer"], message["metadata"].layout.name), self.fixture.receiver.sources)
                raise TimeoutError("register reply lost")

        self.after_call = lose_reply
        sender = self.sender()
        with self.assertLogs("vllm", level="ERROR") as logs, self.assertRaisesRegex(TimeoutError, "reply lost"):
            sender.initialize()
        self.assertIn("stage=D_register", "\n".join(logs.output))
        self.assertTrue(sender.closed)
        self.assertIn(sender.producer, self.fixture.receiver.detached_producers)
        for source in self.created:
            source.shutdown.assert_called_once()

    def test_unconfirmed_detach_retains_sources_and_close_retries(self):
        self.fail_second_cpu_register()

        def fail_detach(endpoint, message):
            if message["op"] == "detach":
                raise TimeoutError("detach delivery unknown")

        self.before_call = fail_detach
        sender = self.sender()
        with self.assertLogs("vllm", level="ERROR") as logs, self.assertRaisesRegex(RuntimeError, "CTP query failed"):
            sender.initialize()
        self.assertIn("cleanup incomplete", "\n".join(logs.output))
        self.assertEqual(len(sender.registrations), 5)
        self.assertFalse(sender.closed)
        for source in self.created:
            source.shutdown.assert_not_called()
        self.bootstrap.shutdown.assert_not_called()
        self.before_call = lambda endpoint, message: None
        sender.close()
        self.assertTrue(sender.closed)
        self.bootstrap.shutdown.assert_called_once()

    def test_lost_detach_reply_retains_sources_until_duplicate_ack(self):
        self.fail_second_cpu_register()

        def lose_reply(endpoint, message, result):
            if message["op"] == "detach":
                raise TimeoutError("detach reply lost")

        self.after_call = lose_reply
        sender = self.sender()
        with self.assertLogs("vllm", level="ERROR"), self.assertRaisesRegex(RuntimeError, "CTP query failed"):
            sender.initialize()
        self.assertFalse(any(key[0] == sender.producer for key in self.fixture.receiver.sources))
        for source in self.created:
            source.shutdown.assert_not_called()
        self.after_call = lambda endpoint, message, result: None
        sender.close()
        self.assertTrue(sender.closed)

    def test_partial_sdk_import_is_shutdown_by_producer_without_reader_detach_api(self):
        sender = self.sender()

        def fail(source):
            if source.npu:
                raise RuntimeError("partial import")

        self.before_register = fail
        with self.assertLogs("vllm", level="ERROR"), self.assertRaisesRegex(RuntimeError, "partial import"):
            sender.initialize()
        self.created[1].shutdown.assert_called_once()
        self.assertTrue(sender.closed)

    def test_late_register_is_rejected_after_detach_barrier(self):
        queued = []

        def hold_register(endpoint, message):
            if message["op"] == "register":
                queued.append(message)
                raise TimeoutError("register handler queued")

        self.before_call = hold_register
        sender = self.sender()
        with self.assertLogs("vllm", level="ERROR"), self.assertRaisesRegex(TimeoutError, "handler queued"):
            sender.initialize()
        self.assertTrue(sender.closed)
        with self.assertRaisesRegex(RuntimeError, "late registration rejected"):
            self.fixture.receiver.dispatch(queued[0])
        self.assertFalse(any(key[0] == sender.producer for key in self.fixture.receiver.sources))

    def test_old_d_without_cleanup_fence_retains_sources(self):
        def old_hello(endpoint, message, result):
            if message["op"] == "hello":
                result.pop("source_cleanup_v1")

        self.after_call = old_hello
        self.fail_second_cpu_register()
        sender = self.sender()
        with self.assertLogs("vllm", level="ERROR") as logs, self.assertRaisesRegex(RuntimeError, "CTP query failed"):
            sender.initialize()
        self.assertIn("D lacks source_cleanup_v1", "\n".join(logs.output))
        self.assertFalse(sender.closed)
        for source in self.created:
            source.shutdown.assert_not_called()

    def test_sdk_shutdown_failure_keeps_owner_and_does_not_block_other_layers(self):
        self.fail_second_cpu_register()
        sender = self.sender()

        def fail(source):
            if not source.npu:
                self.created[1].shutdown.side_effect = RuntimeError("SDK shutdown unresolved")
                raise RuntimeError("Offload: source CTP query failed")

        self.before_register = fail
        with self.assertLogs("vllm", level="ERROR"), self.assertRaisesRegex(RuntimeError, "CTP query failed"):
            sender.initialize()
        self.assertEqual(len(sender.registrations), 5)
        self.created[0].shutdown.assert_not_called()  # Keep CPU source while NPU shutdown is unresolved.
        self.bootstrap.shutdown.assert_not_called()
        self.created[1].shutdown.side_effect = None
        sender.close()
        self.assertTrue(sender.closed)

    def test_one_unreachable_peer_does_not_block_cleanup_of_other_peers(self):
        f = self.fixture
        other_endpoint = "127.0.0.1:21001"
        other = flow.RECEIVER.Receiver(
            f.runtime, f.writer, f.pool, f.memories, host="127.0.0.1", port=0, start_immediately=False
        )
        other.start()
        self.receivers[other_endpoint] = other
        self.addCleanup(other.close)

        def fail(source):
            if len(self.created) > 2 and source is self.created[2]:
                raise RuntimeError("Offload: source CTP query failed")

        def fail_detach(endpoint, message):
            if endpoint == f.endpoint and message["op"] == "detach":
                raise TimeoutError("first peer unreachable")

        self.before_register, self.before_call = fail, fail_detach
        sender = self.sender(endpoints=(f.endpoint, other_endpoint))
        with self.assertLogs("vllm", level="ERROR"), self.assertRaisesRegex(RuntimeError, "CTP query failed"):
            sender.initialize()
        self.assertEqual(len(sender.registrations), 5)
        self.assertTrue(all(key[0] == f.endpoint for key in sender.registrations))
        self.assertFalse(other.sources)
        for source in self.created[:2]:
            source.shutdown.assert_not_called()
        for source in self.created[2:]:
            source.shutdown.assert_called_once()
        self.bootstrap.shutdown.assert_not_called()
        self.before_call = lambda endpoint, message: None
        sender.close()
        self.assertTrue(sender.closed)

    def test_bootstrap_failure_preserves_original_error_and_shuts_down_partial_runtime(self):
        sender = self.sender()
        error = RuntimeError("partial runtime initialization")
        self.bootstrap.initialize.side_effect = error
        with self.assertLogs("vllm", level="ERROR"), self.assertRaises(RuntimeError) as raised:
            sender.initialize()
        self.assertIs(raised.exception, error)
        self.assertFalse(self.created)
        self.bootstrap.shutdown.assert_called_once()
        self.assertTrue(sender.closed)


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

    def test_raw_metadata_failure_is_cancelled_by_reader_before_source_release(self):
        f = self.fixture
        with (
            patch.object(flow.RawTransfer, "_make_arrays", side_effect=ValueError("metadata invalid")),
            patch.object(f.native, "cancel_raw_addr", wraps=f.native.cancel_raw_addr) as cancel,
        ):
            f.chunk(300)
        self.assertGreater(cancel.call_count, 0)
        self.assertEqual(f.native.raw_calls, 0)
        self.assertTrue(all(source.active is None for source in f.sdk.sources.values()))
        self.assertEqual(f.receiver.get_finished(), {"r"})  # Native receives failure completion plus load-error blocks.
        self.assertFalse(f.receiver.requests["r"]["published"])
        self.assertEqual(f.receiver.get_errors(), {block for group in f.blocks for block in group})

    def test_lost_raw_cancellation_ack_resolves_without_read_replay(self):
        f = self.fixture
        cancel = f.native.cancel_raw_addr

        def lose_ack(ready):
            cancel(ready)
            f.native.pending_raw_addr = True
            raise TimeoutError("cancel acknowledgement lost")

        with (
            patch.object(flow.RawTransfer, "_make_arrays", side_effect=ValueError("metadata invalid")),
            patch.object(f.native, "cancel_raw_addr", side_effect=lose_ack),
            patch.object(f.native, "resolve_raw_addr", wraps=f.native.resolve_raw_addr) as resolve,
        ):
            f.chunk(300)
        self.assertGreater(resolve.call_count, 0)
        self.assertEqual(f.native.raw_calls, 0)
        self.assertFalse(f.native.pending_raw_addr)
        self.assertTrue(all(source.active is None for source in f.sdk.sources.values()))

    def test_published_ready_for_failed_job_is_fenced_without_read(self):
        f = self.fixture

        def fail_job(message):
            if message["op"] == "pull":
                f.receiver.jobs[message["job"]].error = RuntimeError("request already failed")

        self.before = fail_job
        with patch.object(f.native, "cancel_raw_addr", wraps=f.native.cancel_raw_addr) as cancel:
            f.chunk(300)
        self.assertGreater(cancel.call_count, 0)
        self.assertEqual(f.native.raw_calls, 0)
        self.assertNotIn("submit", f.native.calls)
        self.assertTrue(all(source.active is None for source in f.sdk.sources.values()))

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
