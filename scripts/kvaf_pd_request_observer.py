# SPDX-License-Identifier: Apache-2.0
"""Request-only passive host milestones. Holds no SDK/Ready/Tensor objects."""

from __future__ import annotations

import itertools
import threading
import time

from kvaf_pd_performance import BatchObserver, Capture, SenderObserver, _text


class RequestEvents:
    DETAIL_PHASES = frozenset(
        (
            "D.runtime_lock_wait",
            "D.compute_dependency_wait",
            "D.cpu_prepare",
            "D.cpu_submit",
            "D.cpu_wait",
            "D.cpu_commit",
            "D.cpu_publish",
            "D.npu_read",
            "D.npu_recovery",
        )
    )

    def __init__(self, config, capture, external_id, role):
        self.config, self.capture, self.external_id, self.role = config, capture, external_id, role
        self.lock = threading.Lock()
        self.requests, self.jobs = {}, {}
        self.limit = capture.capacity // 2 if role == "P" else capture.capacity
        self.pending = 0
        self.sequence = itertools.count()

    def snapshot(self):
        with self.lock:
            return dict(
                observed_requests=len(self.requests),
                observed_control_jobs=len(self.jobs),
                pending_phase_records=self.pending,
            )

    def _reserve(self, count):
        if len(self.requests) + len(self.jobs) + self.pending + count > self.limit:
            self.capture.drop()
            return False
        self.pending += count
        return True

    def _id(self):
        return f"{self.capture.identity}:{next(self.sequence)}"

    def _context(self, raw):
        if not raw:
            return None
        if "job" in raw:
            context = self.jobs.get(raw["job"])
            if context is None:
                return None
            context = dict(context)
        elif "request_id" in raw:
            alias = raw["request_id"]
            context = dict(self.requests.get(alias, {}))
            context.setdefault("request", self.external_id(alias))
        elif "request" in raw:
            context = dict(request=raw["request"])
        else:
            return None
        for key in ("generation", "layer", "first", "tokens", "final", "member", "ratio", "path", "batch"):
            if key in raw:
                context[key] = raw[key]
        if not self.config.selects(_text(context["request"])) or not context.get("generation"):
            return None
        _text(context["generation"])
        if "layer" in context:
            _text(context["layer"])
        return context

    def _point(self, name, context, stamp):
        self.capture.emit(
            dict(
                context,
                kind="milestone",
                name=name,
                time_ns=stamp,
                success=True,
                observation=context.get("observation", "host_confirmed"),
            )
        )

    def _members(self, aliases, *, require_ready=False):
        return [
            dict(self.requests[alias])
            for alias in aliases
            if alias in self.requests and (not require_ready or self.requests[alias].get("gate_ready"))
        ]

    def _task_members(self, layer, requests):
        result = []
        for alias, request, generation in requests:
            context = self._context(
                dict(
                    request_id=alias,
                    generation=generation,
                    layer=layer,
                    first=request.local_transed_tokens // 128,
                    tokens=min(request.local_computed_tokens, request.prompt_len),
                    final=request.chunk_finish,
                )
            )
            if context:
                result.append(context)
        return result

    def record(self, event, **fields):
        if self.capture.stop.is_set():
            return None
        if event == "end" and fields.get("token") is None:
            return None
        if self.config.detail_level == "stage":
            if event in ("p_task", "p_eligible", "point"):
                return None
            if event == "begin" and (
                fields["name"] in ("P.production_wait", "D.seed_wait") or fields["name"].startswith("D.control.")
            ):
                return None
        if event == "begin" and fields["name"] == "D.control.status" and self.config.detail_level != "detail":
            return None
        if event == "begin" and fields["name"] in self.DETAIL_PHASES and self.config.detail_level != "detail":
            return None
        if not self.lock.acquire(blocking=False):
            self.capture.drop()
            return None
        try:
            stamp = time.perf_counter_ns()
            if fields.get("at_ns") is not None:
                actual = fields["at_ns"]
                if type(actual) is not int or not 0 < actual <= stamp:
                    raise ValueError("invalid locally measured observation timestamp")
                stamp = actual
            return self._record(event, fields, stamp)
        finally:
            self.lock.release()

    def _record(self, event, fields, stamp):
        if event == "request_released":
            old = self.requests.pop(fields["request_id"], None)
            if old:
                self.jobs = {
                    job: context
                    for job, context in self.jobs.items()
                    if (context["request"], context["generation"]) != (old["request"], old["generation"])
                }
            return None
        if event == "request_bound":
            alias = fields["request_id"]
            external = _text(fields.get("external_id") or self.external_id(alias))
            if not self.config.selects(external):
                return None
            if alias not in self.requests and len(self.requests) + len(self.jobs) + self.pending >= self.limit:
                self.capture.drop()
                return None
            context = dict(request=external, generation=_text(fields["generation"]), tokens=fields["tokens"])
            old = self.requests.get(alias)
            if old and old["generation"] == context["generation"]:
                context["gate_ready"] = old.get("gate_ready", False)
                context["decode_started"] = old.get("decode_started", False)
            self.requests[alias] = context
            self._point(fields["name"], context, stamp)
            return None
        if event == "gate_ready":
            context = self.requests.get(fields["request_id"])
            if context and context["generation"] == fields["generation"]:
                context["gate_ready"] = True
                self._point("RECEIVE_GATE_READY", context, stamp)
            return None
        if event == "p_task":
            for context in self._task_members(fields["layer"], fields["requests"]):
                self._point(fields["name"], context, stamp)
            return None
        if event == "p_eligible":
            request = fields["request"]
            context = self._context(
                dict(
                    request_id=fields["request_id"],
                    generation=fields["generation"],
                    layer=fields["layer"],
                    first=request.local_transed_tokens // 128,
                    tokens=min(request.local_computed_tokens, request.prompt_len),
                    final=request.chunk_finish,
                    member=fields["member"],
                    ratio=fields["ratio"],
                )
            )
            if context and self.config.detail_level != "stage":
                context["peer"] = _text(fields["peer"])
                self._point("TRANSFER_ELIGIBLE", context, stamp)
            return None
        if event == "point":
            context = self._context(fields["context"])
            if context:
                self._point(fields["name"], context, stamp)
            return None
        if event == "scheduled":
            return self._batch(
                "D_SCHEDULED", self._members(fields["request_ids"]), len(fields["request_ids"]), stamp, instant=True
            )
        if event == "begin":
            name, raw = fields["name"], fields.get("context") or {}
            if "request_ids" in raw:
                members = self._members(raw["request_ids"], require_ready=self.role == "D")
                if not members:
                    return None
                token = self._batch(name, members, len(raw["request_ids"]), stamp)
                if name == "D.target_host_forward":
                    for alias in raw["request_ids"]:
                        member = self.requests.get(alias)
                        if member and member.get("gate_ready") and not member.get("decode_started"):
                            member["decode_started"] = True
                            self._point("D_DECODE_STARTED", dict(member, observation="host_model_forward_enter"), stamp)
                return token
            if "requests" in raw:
                members = self._task_members(raw["layer"], raw["requests"])
                token = self._batch(name, members, len(raw["requests"]), stamp)
                if token:
                    token["production"] = tuple(members)
                return token
            context = self._context(raw)
            if context and self._reserve(1):
                token = dict(
                    context=context, name=name, start_ns=stamp, span_id=self._id(), job=raw.get("job"), charge=1
                )
                self.capture.emit(
                    dict(
                        context,
                        kind="phase_begin",
                        name=name,
                        start_ns=stamp,
                        span_id=token["span_id"],
                        job=raw.get("job"),
                    )
                )
                return token
            return None
        if event == "end":
            token = fields.get("token")
            if token is None:
                return None
            self.pending -= token.get("charge", 0)
            success = fields["success"]
            result = fields.get("result")
            if isinstance(result, dict) and result.get("error"):
                success = False
            if token.get("batch_id"):
                self.capture.emit(
                    dict(
                        kind="batch_span",
                        name=token["name"],
                        batch_id=token["batch_id"],
                        batch_size=token["batch_size"],
                        start_ns=token["start_ns"],
                        end_ns=stamp,
                        success=success,
                        **({"layer": token["layer"]} if "layer" in token else {}),
                    )
                )
                if success and token.get("production"):
                    for context in token["production"]:
                        self._point("P_SOURCE_READY_OBSERVED", context, stamp)
                return None
            context = token["context"]
            phase_job = (
                result.get("job")
                if token["name"] == "D.control.plan" and isinstance(result, dict)
                else token.get("job")
            )
            self.capture.emit(
                dict(
                    context,
                    kind="phase",
                    name=token["name"],
                    span_id=token["span_id"],
                    job=phase_job,
                    start_ns=token["start_ns"],
                    end_ns=stamp,
                    success=success,
                )
            )
            if token["name"] == "D.control.plan" and success and isinstance(result, dict):
                if len(self.requests) + len(self.jobs) + self.pending >= self.limit:
                    self.capture.drop()
                else:
                    self.jobs[_text(result["job"])] = dict(context)
            elif token["name"] in ("D.control.complete", "D.control.status") and success:
                if token["name"] == "D.control.complete" or isinstance(result, dict) and result.get("done"):
                    job = token.get("job")
                    if job:
                        self.jobs.pop(job, None)
            return None
        raise ValueError("unknown passive request observation")

    def _batch(self, name, members, size, stamp, *, instant=False):
        if not members:
            return None
        if not self._reserve(len(members)):
            return None
        identity = self._id()
        for member in members:
            self.capture.emit(dict(member, kind="membership", batch_id=identity))
        token = dict(name=name, batch_id=identity, batch_size=size, start_ns=stamp, charge=len(members))
        if members[0].get("layer") and all(member.get("layer") == members[0]["layer"] for member in members):
            token["layer"] = members[0]["layer"]
        if instant:
            self.capture.emit(dict(token, kind="batch_span", end_ns=stamp, instant=True, success=True))
            self.pending -= token["charge"]
            return None
        self.capture.emit(dict(token, kind="batch_begin"))
        return token


class ProducerObserver(SenderObserver):
    def __init__(self, sender, config, header):
        self.events = None
        roles = {name: "target" if memory.offload_main else "mtp" for name, memory in sender.memories.items()}
        super().__init__(sender, config, dict(header, layer_roles=roles, eligible_observation=True))
        self.job_limit = self.capture.capacity // 2
        self.events = RequestEvents(config, self.capture, sender.external_id, "P")

    def record(self, event, **fields):
        return self.events.record(event, **fields)

    def snapshot(self):
        return dict(super().snapshot(), **(self.events.snapshot() if self.events else {}))


class WorkerObserver:
    def __init__(self, config, header, external_id):
        self.events = None
        self.capture = Capture(config, dict(header, role="D_worker"), self.snapshot)
        self.events = RequestEvents(config, self.capture, external_id, "D")

    def record(self, event, **fields):
        return self.events.record(event, **fields)

    def snapshot(self):
        return self.events.snapshot() if self.events else {}


class SchedulerObserver(BatchObserver):
    def __init__(self, config, header, external_id):
        self.events = None
        super().__init__(config, header)
        self.events = RequestEvents(config, self.capture, external_id, "D_scheduler")

    def record(self, event, **fields):
        return self.events.record(event, **fields)

    def snapshot(self):
        return dict(super().snapshot(), **(self.events.snapshot() if self.events else {}))

    def observe(self, output):
        super().observe(output)
        try:
            self.record("scheduled", request_ids=tuple(output.num_scheduled_tokens))
        except Exception:
            self.capture.fail("request scheduling observation failed")
