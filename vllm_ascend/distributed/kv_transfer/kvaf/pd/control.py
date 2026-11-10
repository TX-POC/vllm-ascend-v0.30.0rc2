# SPDX-License-Identifier: Apache-2.0
"""Bounded TCP control RPC for trusted P/D peers; KV bytes travel via KVAF."""

import importlib
import io
import pickle
import socket
import socketserver
import struct
from contextlib import suppress
from threading import Thread
from types import SimpleNamespace

from .config import DEFAULT_CONTROL_BACKLOG, validate_control_backlog
from .layout import LayerLayout, Plane
from .registration import SourceMetadata

MAX_MESSAGE_BYTES = 16 * 1024 * 1024
HEADER = struct.Struct("!I")


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        for allowed in (LayerLayout, Plane, SourceMetadata, SimpleNamespace):
            if (module, name) == (allowed.__module__, allowed.__name__):
                return allowed
        if module in ("kvaf._kvaf", "_kvaf") and name in ("CpuPullSourceDesc", "NpuPullSourceDesc", "ReadyToken"):
            # The SDK exposes only pickle constructors for these opaque types.
            # Keep arbitrary Python globals out of the business control codec.
            return getattr(importlib.import_module("kvaf")._native, name)
        raise pickle.UnpicklingError(f"unexpected PD metadata type {module}.{name}")


def _read(connection, size):
    parts = bytearray()
    while len(parts) < size:
        data = connection.recv(size - len(parts))
        if not data:
            raise ConnectionError("PD control connection closed before completion")
        parts.extend(data)
    return parts


def _receive(connection):
    size = HEADER.unpack(_read(connection, HEADER.size))[0]
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError("invalid PD control message length")
    # SDK Ready/descriptor types implement pickle. Listen only on the trusted
    # deployment network, just like the SDK's own source-control listeners.
    return _MetadataUnpickler(io.BytesIO(_read(connection, size))).load()


def _send(connection, value):
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("PD metadata exceeds the control message limit")
    connection.sendall(HEADER.pack(len(data)) + data)


def call(endpoint, message, *, timeout=120.0):
    host, _, port = endpoint.rpartition(":")
    with socket.create_connection((host, int(port)), timeout=timeout) as connection:
        _send(connection, message)
        success, result = _receive(connection)
    if not success:
        raise RuntimeError(f"PD peer rejected operation: {result}")
    return result


class ControlServer:
    def __init__(self, host, port, dispatch, *, timeout=120.0, backlog=DEFAULT_CONTROL_BACKLOG):
        validate_control_backlog(backlog)

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(timeout)
                try:
                    result = (True, dispatch(_receive(self.request)))
                except Exception as exc:
                    result = (False, f"{type(exc).__name__}: {exc}")
                # A lost ACK never permits the caller to reuse its source.
                with suppress(ConnectionError, OSError):
                    _send(self.request, result)

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True
            block_on_close = False
            # Applied by TCPServer before listen(); queued connections do not
            # change dispatch threads, Runtime locks or transfer completion.
            request_queue_size = backlog

        self.server = Server((host, port), Handler)
        self.thread = Thread(target=self.server.serve_forever, name="kvaf-pd-control", daemon=True)
        self.thread.start()

    @property
    def port(self):
        return self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
