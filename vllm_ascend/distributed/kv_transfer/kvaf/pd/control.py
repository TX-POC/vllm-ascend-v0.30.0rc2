# SPDX-License-Identifier: Apache-2.0
"""Bounded TCP control RPC for trusted P/D peers; KV bytes travel via KVAF."""

import importlib
import io
import logging
import pickle
import socket
import socketserver
import struct
from threading import Thread
from types import SimpleNamespace

from .layout import LayerLayout, Plane
from .registration import SourceMetadata

MAX_MESSAGE_BYTES = 16 * 1024 * 1024
HEADER = struct.Struct("!I")
CONTROL_REQUEST_QUEUE_SIZE = 128  # TP workers can publish several arenas concurrently to one D worker.


class _EmptyConnection(ConnectionError):
    """A peer closed before sending any control header (e.g. a TCP probe)."""


def _log_failure(side, endpoint, phase, message, exc):
    # Only bounded identifiers; never log the metadata, descriptors or KV payload.
    fields = message if isinstance(message, dict) else {}

    def identifier(key):
        value = fields.get(key, "-")
        return str(value)[:160] if isinstance(value, (str, int)) else "-"

    logging.getLogger("vllm").error(
        "KVAF_PD_RPC_FAIL side=%s endpoint=%s phase=%s op=%s producer=%s request=%s "
        "generation=%s layer=%s member=%s error=%s detail=%s",
        side,
        endpoint,
        phase,
        identifier("op"),
        identifier("producer"),
        identifier("request"),
        identifier("generation"),
        identifier("layer"),
        identifier("member"),
        type(exc).__name__,
        str(exc)[:256],
        exc_info=side == "server" and phase == "dispatch",
    )


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


def _read(connection, size, *, header=False):
    parts = bytearray()
    while len(parts) < size:
        data = connection.recv(size - len(parts))
        if not data:
            if header and not parts:
                raise _EmptyConnection("PD control peer closed without an RPC")
            raise ConnectionError("PD control connection closed before completion")
        parts.extend(data)
    return parts


def _receive(connection):
    size = HEADER.unpack(_read(connection, HEADER.size, header=True))[0]
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
    phase = "connect"
    try:
        with socket.create_connection((host, int(port)), timeout=timeout) as connection:
            phase = "send"
            _send(connection, message)
            phase = "receive"
            success, result = _receive(connection)
    except OSError as exc:
        # Startup hello polling has its own final timeout; do not log every poll.
        if message.get("op") != "hello":
            _log_failure("client", endpoint, phase, message, exc)
        raise  # Preserve the exception type and the original transfer/recovery ownership.
    if not success:
        raise RuntimeError(f"PD peer rejected operation: {result}")
    return result


class ControlServer:
    def __init__(self, host, port, dispatch, *, timeout=120.0):
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(timeout)
                message, phase = None, "receive"
                try:
                    message = _receive(self.request)
                    phase = "dispatch"
                    result = (True, dispatch(message))
                except Exception as exc:
                    if isinstance(exc, _EmptyConnection):
                        return  # A connect-only health probe has no RPC to acknowledge.
                    _log_failure("server", self.server.server_address, phase, message, exc)
                    result = (False, f"{type(exc).__name__}: {exc}")
                # A lost ACK never permits the caller to reuse its source.
                try:
                    _send(self.request, result)
                except (ConnectionError, OSError) as exc:
                    _log_failure("server", self.server.server_address, "respond", message, exc)

        class Server(socketserver.ThreadingTCPServer):
            request_queue_size = CONTROL_REQUEST_QUEUE_SIZE
            allow_reuse_address = True
            daemon_threads = True
            block_on_close = False

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
