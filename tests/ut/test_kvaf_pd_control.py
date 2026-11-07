# SPDX-License-Identifier: Apache-2.0
"""Real TCP control bursts and failure context; no NPU or SDK is required."""

import socket
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from unittest.mock import MagicMock, patch

import test_kvaf_pd_flow as flow

CONTROL = flow.CONTROL


class TestPDControl(unittest.TestCase):
    def server(self, dispatch):
        server = CONTROL.ControlServer("127.0.0.1", 0, dispatch, timeout=3)
        self.addCleanup(server.close)
        return server, f"127.0.0.1:{server.port}"

    def test_tp8_six_arenas_can_plan_concurrently_without_replay(self):
        requests = 8 * 6
        barrier = Barrier(requests)
        received = []

        def dispatch(message):
            received.append(message["request"])
            return message["request"]

        server, endpoint = self.server(dispatch)
        self.assertGreaterEqual(server.server.request_queue_size, requests)

        def send(index):
            barrier.wait(timeout=10)
            return CONTROL.call(endpoint, {"op": "plan", "request": index}, timeout=10)

        with ThreadPoolExecutor(max_workers=requests) as executor, patch.object(CONTROL, "_log_failure") as failure:
            self.assertEqual(list(executor.map(send, range(requests))), list(range(requests)))
            failure.assert_not_called()
        self.assertCountEqual(received, range(requests))

    def test_dispatch_failure_is_returned_and_logs_identifiers_only(self):
        def dispatch(message):
            raise ValueError("invalid block table")

        _, endpoint = self.server(dispatch)
        message = dict(op="plan", request="r", producer="P:TP7", generation="g", layer="layer.4", member=7)
        message["tables"] = "DO_NOT_LOG_PAYLOAD"
        with (
            self.assertLogs("vllm", level="ERROR") as captured,
            self.assertRaisesRegex(RuntimeError, "invalid block table"),
        ):
            CONTROL.call(endpoint, message, timeout=3)
        line = "\n".join(captured.output)
        for field in ("side=server", "phase=dispatch", "op=plan", "request=r", "layer=layer.4", "member=7"):
            self.assertIn(field, line)
        self.assertNotIn("DO_NOT_LOG_PAYLOAD", line)

    def test_client_reset_preserves_exception_and_does_not_retry(self):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        error = ConnectionResetError(104, "reset by peer")
        connection.recv.side_effect = error
        message = dict(op="plan", request="r", layer="layer.4")
        with (
            patch.object(CONTROL.socket, "create_connection", return_value=connection) as connect,
            self.assertLogs("vllm", level="ERROR") as captured,
            self.assertRaises(ConnectionResetError) as caught,
        ):
            CONTROL.call("D:31004", message)
        self.assertIs(caught.exception, error)
        connect.assert_called_once()
        connection.sendall.assert_called_once()
        self.assertIn("endpoint=D:31004 phase=receive op=plan", captured.output[0])

    def test_startup_hello_poll_failure_does_not_flood_logs(self):
        with (
            patch.object(CONTROL.socket, "create_connection", side_effect=ConnectionRefusedError),
            patch.object(CONTROL, "_log_failure") as failure,
        ):
            with self.assertRaises(ConnectionRefusedError):
                CONTROL.call("D:31004", {"op": "hello"})
            failure.assert_not_called()

    def test_lost_response_is_logged_without_dispatch_replay(self):
        calls = []
        _, endpoint = self.server(lambda message: calls.append(message["request"]) or True)
        original = CONTROL._send
        logged = Event()
        failures = []

        def send(connection, value):
            if isinstance(value, tuple):
                raise BrokenPipeError("ACK lost")
            return original(connection, value)

        def log(*args):
            failures.append(args)
            if args[2] == "respond":
                logged.set()

        with patch.object(CONTROL, "_send", side_effect=send), patch.object(CONTROL, "_log_failure", side_effect=log):
            with self.assertRaises(ConnectionError):
                CONTROL.call(endpoint, {"op": "plan", "request": "r"}, timeout=3)
            self.assertTrue(logged.wait(3))
        self.assertEqual(calls, ["r"])
        self.assertTrue(
            any(args[:3] == ("server", ("127.0.0.1", int(endpoint.rsplit(":", 1)[1])), "respond") for args in failures)
        )

    def test_partial_header_is_logged_but_empty_tcp_probe_is_quiet(self):
        _, endpoint = self.server(lambda message: True)
        address = "127.0.0.1", int(endpoint.rsplit(":", 1)[1])
        with patch.object(CONTROL, "_log_failure") as failure:
            # Wait for the handler to close the probe, without polling/sleeping.
            with socket.create_connection(address, timeout=3) as connection:
                connection.shutdown(socket.SHUT_WR)
                self.assertEqual(connection.recv(1), b"")
            failure.assert_not_called()
            with socket.create_connection(address, timeout=3) as connection:
                connection.sendall(b"\x00")
                connection.shutdown(socket.SHUT_WR)
                success, error = CONTROL._receive(connection)
            self.assertFalse(success)
            self.assertIn("closed before completion", error)
            failure.assert_called_once()
            self.assertEqual(failure.call_args.args[2], "receive")


if __name__ == "__main__":
    unittest.main()
