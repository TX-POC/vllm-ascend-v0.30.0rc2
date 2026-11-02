# SPDX-License-Identifier: Apache-2.0
"""Route native KVAF stderr without importing the serving or NPU runtime."""

import ctypes
import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

READ_BYTES = 64 * 1024
RELAY_EXIT_TIMEOUT = 5


def _flush_stderr():
    sys.stderr.flush()
    flush = ctypes.CDLL(None).fflush
    flush.argtypes, flush.restype = [ctypes.c_void_p], ctypes.c_int
    flush(None)


class NativeLog:
    """One worker-local relay, installed once and restored after shutdown.

    KVAF's pybind calls can hold the GIL while writing stderr. A separate
    process must drain the pipe: a Python reader thread could deadlock when
    a native call fills it. Neither Get/Put nor graph replay waits on Python
    logging or changes file descriptors.
    """

    def __init__(self, path: Path):
        self.path = path
        self._process = None
        self._saved_stderr = None

    def start(self):
        if self._process is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _flush_stderr()
        descriptors = []
        process = None
        try:
            saved = os.dup(2)
            descriptors.append(saved)
            read_fd, write_fd = os.pipe()
            descriptors.extend((read_fd, write_fd))
            output = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            descriptors.append(output)
            # Execute this file, not its package: the relay never imports
            # vLLM, torch, torch_npu, or KVAF and owns no device state.
            process = subprocess.Popen(
                [sys.executable, "-S", "-u", str(Path(__file__).resolve()), str(output)],
                stdin=read_fd,
                stdout=subprocess.DEVNULL,
                stderr=saved,
                pass_fds=(output,),
            )
            os.dup2(write_fd, 2)
            self._saved_stderr, self._process = saved, process
            descriptors.remove(saved)
        except BaseException:
            if process is not None:
                process.terminate()
                process.wait()
            raise
        finally:
            for descriptor in descriptors:
                os.close(descriptor)

    def close(self):
        if self._process is None:
            return
        process, saved = self._process, self._saved_stderr
        try:
            _flush_stderr()
        finally:
            # Closing our pipe writer lets the relay drain through EOF before
            # exit. Restore first so any shutdown diagnostics remain visible.
            os.dup2(saved, 2)
            os.close(saved)
            self._process = self._saved_stderr = None
            try:
                process.wait(timeout=RELAY_EXIT_TIMEOUT)
            except subprocess.TimeoutExpired:
                # Another inherited writer must not hang worker shutdown.
                process.kill()
                process.wait()


def _write_all(descriptor, data):
    remaining = memoryview(data)
    while remaining:
        remaining = remaining[os.write(descriptor, remaining) :]


def _relay(output):
    pending = b""

    def emit(line):
        nonlocal output
        native = line.startswith((b"[KVAF_", b"[KVAF]"))
        destination = output if native and output is not None else 2
        try:
            _write_all(destination, line)
        except OSError:
            if destination != 2:
                # A full log filesystem must not kill the relay and break
                # the worker's stderr. Preserve these diagnostics on console.
                output = None
                with suppress(OSError):
                    _write_all(2, b"KVAF native log write failed; forwarding native messages to stderr.\n" + line)

    while chunk := os.read(0, READ_BYTES):
        lines = (pending + chunk).split(b"\n")
        pending = lines.pop()
        for line in lines:
            emit(line + b"\n")
    if pending:
        emit(pending)


if __name__ == "__main__":
    _relay(int(sys.argv[1]))
