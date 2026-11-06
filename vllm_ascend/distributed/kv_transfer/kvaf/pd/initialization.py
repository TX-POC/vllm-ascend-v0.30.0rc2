# SPDX-License-Identifier: Apache-2.0
"""Opt-in initialization admission for trusted D workers sharing a K endpoint.

This module uses only the standard library so its server can run on a CPU host.
The connection owns admission until explicit success. Errors/disconnects block
that endpoint permanently in this server process; a timer never transfers it.
"""

import argparse
import json
import socket
import socketserver
import time
from collections import deque
from contextlib import contextmanager, suppress
from threading import Condition

MAX_MESSAGE_BYTES = 4096
HANDSHAKE_TIMEOUT = 10


def validate_endpoint(endpoint):
    if not isinstance(endpoint, str):
        raise ValueError("initialization endpoint must be host:port")
    host, separator, port = endpoint.rpartition(":")
    if not separator or not host or any(char.isspace() for char in host) or not port.isdecimal():
        raise ValueError("initialization endpoint must be host:port")
    if not 0 < int(port) <= 65535:
        raise ValueError("initialization endpoint port must be in 1..65535")
    return host, int(port)


def _receive(reader):
    data = reader.readline(MAX_MESSAGE_BYTES + 1)
    if not data.endswith(b"\n") or len(data) > MAX_MESSAGE_BYTES:
        raise ConnectionError("incomplete or oversized initialization admission message")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("initialization admission message must be an object")
    return value


def _send(connection, value):
    connection.sendall(json.dumps(value).encode("utf-8") + b"\n")


@contextmanager
def initialization_slot(coordinator, endpoint, *, timeout):
    """Empty address is feature-off: no socket, synchronization or retry."""
    if not coordinator:
        yield
        return
    address = validate_endpoint(coordinator)
    validate_endpoint(endpoint)
    with socket.create_connection(address, timeout=timeout) as connection, connection.makefile("rb") as reader:
        # Waiting for admission does not consume any SDK socket timeout.
        connection.settimeout(timeout + HANDSHAKE_TIMEOUT)
        _send(connection, {"endpoint": endpoint, "timeout": timeout})
        response = _receive(reader)
        if response.get("status") != "granted":
            raise RuntimeError(f"PD initialization admission rejected for {endpoint}: {response.get('error')}")
        # No heartbeat/expiry: the SDK may still be running after a long wait.
        # On exceptional exit, closing without RELEASE poisons the server gate.
        yield
        connection.settimeout(HANDSHAKE_TIMEOUT)
        _send(connection, {"status": "release"})
        if _receive(reader).get("status") != "released":
            raise RuntimeError(f"PD initialization release not acknowledged for {endpoint}")


class _Gate:
    def __init__(self):
        self.condition = Condition()
        self.waiters = deque()
        self.active = False
        self.blocked = False

    def acquire(self, timeout):
        waiter = object()
        deadline = time.monotonic() + timeout
        with self.condition:
            self.waiters.append(waiter)
            try:
                while True:
                    if self.blocked:
                        raise RuntimeError(
                            "previous initializer failed/disconnected; quiesce all D workers before recovery"
                        )
                    if not self.active and self.waiters[0] is waiter:
                        self.active = True
                        return
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("waiting for initialization admission timed out; SDK was not called")
                    self.condition.wait(remaining)
            finally:
                self.waiters.remove(waiter)
                self.condition.notify_all()

    def finish(self, success):
        with self.condition:
            self.blocked = not success
            self.active = False
            self.condition.notify_all()


class InitializationServer(socketserver.ThreadingTCPServer):
    """One shared server for all hosts; endpoints are independent FIFO gates."""

    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False
    request_queue_size = 64

    def __init__(self, address):
        self.gates = {}
        self.gates_condition = Condition()
        super().__init__(address, _Handler)

    def gate(self, endpoint):
        with self.gates_condition:
            return self.gates.setdefault(endpoint, _Gate())


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        gate = None
        acquired = False
        success = False
        try:
            self.request.settimeout(HANDSHAKE_TIMEOUT)
            request = _receive(self.rfile)
            endpoint = request["endpoint"]
            validate_endpoint(endpoint)
            timeout = request["timeout"]
            if type(timeout) not in (int, float) or not 1 <= timeout <= 3600:
                raise ValueError("admission timeout must be in 1..3600 seconds")
            gate = self.server.gate(endpoint)
            gate.acquire(timeout)
            acquired = True
            _send(self.request, {"status": "granted"})
            self.request.settimeout(None)
            if _receive(self.rfile) != {"status": "release"}:
                raise RuntimeError("initializer did not report success")
            _send(self.request, {"status": "released"})
            success = True
        except (Exception, KeyboardInterrupt) as error:
            with suppress(OSError):
                _send(self.request, {"status": "error", "error": str(error)})
        finally:
            if acquired:
                gate.finish(success)


def main():
    parser = argparse.ArgumentParser(description="Serialize PD initialization per K endpoint; no KV data forwarding")
    parser.add_argument("--host", required=True, help="trusted address reachable from every D host")
    parser.add_argument("--port", type=int, default=28998)
    args = parser.parse_args()
    if not 0 < args.port <= 65535:
        parser.error("port must be in 1..65535")
    with InitializationServer((args.host, args.port)) as server:
        print(f"KVAF_INIT_COORDINATOR_LISTEN address={args.host}:{args.port}", flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever()


if __name__ == "__main__":
    main()
