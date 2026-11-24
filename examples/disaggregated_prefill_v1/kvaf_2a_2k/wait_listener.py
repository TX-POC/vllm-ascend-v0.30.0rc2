# SPDX-License-Identifier: Apache-2.0
"""Wait for a newly launched Linux service to own its IPv4 listener."""

import os
import socket
import struct
import sys
import time
from contextlib import suppress
from pathlib import Path

pid, host, port, log, marker, timeout = sys.argv[1:]
pid, port, timeout = int(pid), int(port), float(timeout)
if not 0 < timeout <= 3600:
    raise SystemExit("startup_timeout must be in (0, 3600]")
process = Path(f"/proc/{pid}")
deadline = time.monotonic() + timeout
identity = None
announced = False


def owns_listener():
    inodes = set()
    for fd in (process / "fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith("socket:["):
            inodes.add(target[8:-1])
    with open(process / "net/tcp") as table:
        for line in table:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A" or fields[9] not in inodes:
                continue
            address, service_port = fields[1].split(":")
            address = socket.inet_ntoa(struct.pack("=I", int(address, 16)))
            if int(service_port, 16) == port and address in (host, "0.0.0.0"):
                return True
    return False


try:
    while time.monotonic() < deadline:
        state = (process / "stat").read_text().rsplit(")", 1)[1].split()
        if state[0] in ("Z", "X", "T", "t"):
            raise RuntimeError(f"process exited or stopped: state={state[0]}")
        if identity is not None and identity != state[19]:
            raise RuntimeError("PID was reused by another process")
        identity = state[19]
        if not announced:
            with suppress(FileNotFoundError):
                announced = marker in Path(log).read_text(errors="replace")
        if announced and owns_listener():
            current = (process / "stat").read_text().rsplit(")", 1)[1].split()
            if current[19] != identity or current[0] in ("Z", "X", "T", "t"):
                raise RuntimeError("service exited or stopped during readiness check")
            print(f"SERVICE_READY pid={pid} address={host}:{port} log={log}", flush=True)
            break
        time.sleep(0.1)
    else:
        raise TimeoutError("waiting for listener and readiness log")
except (OSError, ValueError, RuntimeError, TimeoutError) as error:
    raise SystemExit(f"SERVICE_START_FAILED pid={pid} address={host}:{port} log={log}: {error}")
