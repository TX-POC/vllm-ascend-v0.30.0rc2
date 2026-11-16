# SPDX-License-Identifier: Apache-2.0
"""Bounded Proxy host observations; no payload, extra requests or token parsing."""

import itertools
import socket
import threading
import time
from pathlib import Path

from kvaf_pd_performance import Capture, _text


class ProxyObserver:
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.requests = {}
        self.identities = self.pending = 0
        self.sequence = itertools.count()
        self.capture = Capture(
            config,
            dict(
                role="Proxy",
                host=socket.gethostname(),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            ),
            self.snapshot,
        )

    def snapshot(self):
        with self.lock:
            return dict(pending_requests=len(self.requests), pending_phase_records=self.pending)

    def record(self, event, **fields):
        if self.capture.stop.is_set():
            return None
        if event == "clock":
            return time.perf_counter_ns()
        if not self.lock.acquire(blocking=False):
            self.capture.drop()
            return None
        try:
            return self._record(event, fields, time.perf_counter_ns())
        finally:
            self.lock.release()

    def _record(self, event, fields, stamp):
        parent = fields.get("parent")
        if event == "bind":
            members = []
            phase = fields.get("request_phase", "unknown")
            if phase not in ("measure", "warmup", "profile"):
                phase = "unknown"
            available = self.capture.capacity - self.identities - self.pending
            for request in fields["requests"]:
                if self.config.selects(request):
                    if len(members) >= available:
                        self.capture.drop()
                        return None
                    members.append(
                        dict(request=_text(request), generation=None, parent_request=_text(parent), request_phase=phase)
                    )
            if not members:
                return None
            if parent in self.requests:
                raise ValueError("duplicate Proxy parent identity")
            self.requests[parent] = (members, len(fields["requests"]))
            self.identities += len(members)
            begin = fields.get("entered_ns")
            if type(begin) is not int or not 0 < begin <= stamp:
                begin = stamp
            self._point("PROXY_HANDLER_ENTER", members, begin)
            return None
        if event == "release":
            members, _ = self.requests.pop(parent, ([], 0))
            self.identities -= len(members)
            return None
        if event == "end":
            token = fields.get("token")
            if token:
                self.pending -= 1
                self.capture.emit(dict(token, kind="batch_span", end_ns=stamp, success=fields["success"]))
            return None
        members, size = self.requests.get(parent, ([], 0))
        if not members:
            return None
        if fields.get("request") is not None:
            members = [member for member in members if member["request"] == fields["request"]]
            size = len(members)
            if not members:
                return None
        if event == "generation":
            for member in members:
                if member["request"] == fields["request"]:
                    generation = fields["generation"]
                    member["generation"] = _text(generation) if generation is not None else None
                    self._point("PROXY_METADATA_RECEIVED", [member], stamp)
            return None
        if event == "point":
            self._point(fields["name"], members, stamp)
            return None
        if event == "failed":
            self._point(fields["name"], members, stamp, success=False)
            return None
        if event == "begin":
            if self.identities + self.pending + 1 > self.capture.capacity:
                self.capture.drop()
                return None
            self.pending += 1
            identity = f"{self.capture.identity}:proxy:{next(self.sequence)}"
            for member in members:
                self.capture.emit(dict(member, kind="membership", batch_id=identity))
            token = dict(name=fields["name"], batch_id=identity, batch_size=size, start_ns=stamp, parent_request=parent)
            self.capture.emit(dict(token, kind="batch_begin"))
            return token
        raise ValueError("unknown Proxy observation")

    def _point(self, name, members, stamp, *, success=True):
        for member in members:
            self.capture.emit(dict(member, kind="milestone", name=name, time_ns=stamp, success=success))
