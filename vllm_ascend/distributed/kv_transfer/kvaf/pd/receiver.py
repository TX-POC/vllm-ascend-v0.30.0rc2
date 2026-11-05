# SPDX-License-Identifier: Apache-2.0
"""D-side address planning, dual Pull, directory publication and request gate."""

import time
from dataclasses import dataclass, field
from threading import Condition, RLock
from uuid import uuid4

from ..state import RequestCacheState, RequestGroupState
from .control import ControlServer
from .layout import BLOCK_SIZE, MAX_SEGMENTS, Segments, plan_layer
from .state import CompletionGate, Contribution
from .transfer import CpuPullTransfer, NpuPullTransfer


@dataclass
class PullJob:
    request_id: str
    generation: str
    producer: str
    layer: str
    first: int
    tokens: int
    final: bool
    member: int
    ratio: int
    plan: object
    count: dict
    completed: dict = field(default_factory=dict)
    transfers: list = field(default_factory=list)
    finished: bool = False
    error: Exception | None = None
    acknowledged: bool = False
    diagnostic: bool = False
    replicas_checked: bool = False


class Receiver:
    def __init__(
        self,
        runtime,
        writer,
        pool,
        memories,
        *,
        host,
        port,
        local_eid="",
        timeout=120,
        activate=lambda: None,
        stream_factory=lambda: None,
        raw_transfer=NpuPullTransfer,
        start_immediately=True,
        draft_limit=None,
        num_spec_tokens=3,
        diagnostics=None,
    ):
        self.runtime, self.writer, self.pool, self.memories = runtime, writer, pool, memories
        self.identity = uuid4().hex
        self.timeout, self.local_eid = timeout, local_eid
        self.draft_limit, self.num_spec_tokens = draft_limit, num_spec_tokens
        self.diagnostics = diagnostics
        self.activate, self.stream_factory, self.raw_transfer = activate, stream_factory, raw_transfer
        self.condition = Condition(RLock())
        self.sources, self.requests, self.jobs = {}, {}, {}
        self.terminal_jobs = {}
        self.finished, self.errors = set(), set()
        self.native_finished = set()
        self.closed = False
        self._listen = host, port
        self.server = None
        if start_immediately:
            self.start()

    def start(self):
        with self.runtime.lock:
            if self.closed or self.server is not None:
                raise RuntimeError("PD control can start only once")
            self.runtime.wait_compute()
            self.server = ControlServer(*self._listen, self.dispatch, timeout=self.timeout)

    def seed(self, request_id, external_id, seed):
        with self.condition:
            if external_id in self.requests:
                if self.requests[external_id]["seed"] != seed:
                    raise ValueError("live request identity reused with another generation")
                return
            tail_blocks = {
                name: seed.blocks[memory.groups["main"]][seed.tokens // BLOCK_SIZE]
                for name, memory in self.memories.items()
                if memory.offload_main and seed.tokens % BLOCK_SIZE
            }
            self.pool.policy.reserve_receive(request_id, tail_blocks)
            self.writer.protect_receive(request_id)
            self.writer.requests[request_id] = [
                RequestGroupState(RequestCacheState(group.layout.block_size)) for group in self.writer.groups
            ]
            self.requests[external_id] = {
                "id": request_id,
                "seed": seed,
                "cohort": None,
                "final": {},
                "paths": {},
                "published": False,
                "cancelled": False,
                "failed": False,
                "cancel_votes": set(),
            }
            if self.diagnostics and self.diagnostics.select(external_id, seed.generation, alias=request_id):
                self.diagnostics.emit(
                    "RECEIVE_SEED",
                    (external_id, seed.generation),
                    request=request_id,
                    external=external_id,
                    generation=seed.generation,
                    tokens=seed.tokens,
                    full_blocks=seed.tokens // BLOCK_SIZE,
                    tail_tokens=seed.tokens % BLOCK_SIZE,
                    groups=[dict(count=len(group), first=list(group[:4])) for group in seed.blocks],
                )
            self.condition.notify_all()

    def _request(self, external_id, generation):
        deadline = time.monotonic() + self.timeout
        with self.condition:
            while external_id not in self.requests and not self.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("D scheduler allocation metadata has not arrived")
                self.condition.wait(remaining)
            if self.closed:
                raise RuntimeError("PD receiver is closing")
            request = self.requests[external_id]
            if request["seed"].generation != generation:
                raise ValueError("stale request generation")
            return request

    def dispatch(self, message):
        if self.closed:
            raise RuntimeError("PD receiver is closed")
        self.activate()
        operation = message["op"]
        if operation == "hello":
            ticket = self.runtime.offload_region()
            if self.diagnostics:
                self.diagnostics.emit(
                    "DDR_REGION",
                    "startup",
                    peer=self.identity,
                    region_id=getattr(ticket, "region_id", None),
                    region_epoch=getattr(ticket, "region_epoch", None),
                    control_host=self._listen[0],
                    control_port=self._listen[1],
                )
            return {
                "peer": self.identity,
                "ticket": ticket,
                "draft_limit": self.draft_limit,
                "num_spec_tokens": self.num_spec_tokens,
                "verify_kv": bool(self.diagnostics and self.diagnostics.verify),
            }
        if message.get("peer") != self.identity:
            raise ValueError("stale D worker/DDR region")
        if operation == "register":
            metadata = message["metadata"]
            memory = self.memories[metadata.layout.name]
            if message["offload"] != memory.offload_main:
                raise ValueError("P/D disagree about native Target versus MTP ownership")
            key = (message["producer"], metadata.layout.name)
            with self.runtime.lock:
                if self.closed:
                    raise RuntimeError("PD receiver is closed")
                if key in self.sources:
                    raise ValueError("source already registered")
                self.runtime.wait_compute()
                metadata.attach(self.runtime, peer=self.identity, local_eid=self.local_eid)
                self.sources[key] = metadata
            return True
        if operation == "detach":
            with self.runtime.lock:
                producer = message["producer"]
                if any(job.producer == producer and not job.finished for job in self.jobs.values()):
                    raise RuntimeError("drain original source jobs before detaching")
                self.runtime.wait_compute()
                for key, source in tuple(self.sources.items()):
                    if key[0] == producer:
                        self.runtime.detach_hbm(source.npu_ticket)
                        self.sources.pop(key)
                self.terminal_jobs = {key: value for key, value in self.terminal_jobs.items() if value[0] != producer}
            return True
        if operation == "plan":
            return self._plan(message)
        if operation in ("pull", "complete", "status", "ack_complete"):
            return self._job_operation(operation, message)
        if operation == "cancel":
            request = self._request(message["request"], message["generation"])
            with self.runtime.lock:
                member, ratio = message["member"], message["ratio"]
                if type(member) is not int or type(ratio) is not int or not 0 <= member < ratio:
                    raise ValueError("invalid cancellation contributor")
                cohort = (message["cohort"], ratio)
                if request["cohort"] not in (None, cohort):
                    raise ValueError("another producer group attempted to cancel this request")
                request["cohort"] = cohort
                request["cancel_votes"].add(member)
                if len(request["cancel_votes"]) == ratio:
                    request["cancelled"] = True
                    self._fail_request(request)
            return True
        raise ValueError("unknown PD control operation")

    def _job_operation(self, operation, message):
        with self.runtime.lock:
            key = message["job"]
            identity = message["producer"], message["generation"]
            job = self.jobs.get(key)
            if job is None:
                terminal = self.terminal_jobs.get(key)
                if terminal is None:
                    if operation == "ack_complete":
                        return True  # Duplicate cleanup carries no resource ownership.
                    raise KeyError(key)
                if identity != terminal[:2]:
                    raise ValueError("stale source/job identity")
                if operation == "ack_complete":
                    self.terminal_jobs.pop(key)
                    return True
                if operation == "complete":
                    return True
                if operation == "pull":
                    raise RuntimeError("Pull job is terminal")
                return {"done": True, "error": terminal[2], "completed": {}}
            if identity != (job.producer, job.generation):
                raise ValueError("stale source/job identity")
            if operation == "pull":
                return self._pull(job, message)
            if operation == "complete":
                if job.diagnostic and self.diagnostics.verify and not job.replicas_checked and job.error is None:
                    self._check_replicas(job, message.get("replicas"))
                return self._complete(job)
            if operation == "ack_complete":
                if not job.finished:
                    raise RuntimeError("layer completion must precede its acknowledgement")
                job.acknowledged = True
                return True
            return {
                "done": job.finished,
                "error": str(job.error) if job.error else None,
                "completed": dict(job.completed),
            }

    def _forget_request_jobs(self, request_id):
        for key, job in tuple(self.jobs.items()):
            if job.request_id == request_id:
                if job.finished and not job.acknowledged:
                    # No tensors or SDK handles: recover a lost complete reply
                    # even when native Decode teardown already won the race.
                    self.terminal_jobs[key] = (job.producer, job.generation, str(job.error) if job.error else None)
                self.jobs.pop(key)

    def _plan(self, message):
        request = self._request(message["request"], message["generation"])
        with self.runtime.lock:
            if request["cancelled"] or request["published"]:
                raise ValueError("request no longer accepts Pull plans")
            producer, layer = message["producer"], message["layer"]
            source = self.sources[producer, layer].layout
            memory = self.memories[layer]
            member, ratio = message["member"], message["ratio"]
            if type(member) is not int or type(ratio) is not int or not 0 <= member < ratio:
                raise ValueError("invalid Pull contributor")
            if member in request["cancel_votes"]:
                raise ValueError("source participant already cancelled this generation")
            cohort = (message["cohort"], ratio)
            if request["cohort"] not in (None, cohort):
                raise ValueError("request routed to multiple P contributor groups")
            request["cohort"] = cohort
            tokens, first, final = message["tokens"], message["first"], message["final"]
            if type(tokens) is not int or type(first) is not int or first < 0 or type(final) is not bool:
                raise ValueError("invalid Pull progress")
            if (
                not 0 <= tokens <= request["seed"].tokens
                or first > tokens // BLOCK_SIZE
                or final
                and tokens != request["seed"].tokens
            ):
                raise ValueError("P/D disagree about prompt length")
            slot = self.pool.policy.receive_slots[request["id"]] if memory.offload_main else None
            if memory.offload_main and request["seed"].tokens % BLOCK_SIZE:
                native_block = request["seed"].blocks[memory.groups["main"]][request["seed"].tokens // BLOCK_SIZE]
                slot = self.pool.policy.receive_slot_for(request["id"], layer, native_block)
            tables = memory.tables(request["seed"].blocks, tail_slot=slot, tokens=tokens)
            skipped = message.get("draft_skipped", False)
            if type(skipped) is not bool:
                raise ValueError("draft skip must be an explicit boolean")
            if skipped and (
                memory.offload_main
                or self.draft_limit is None
                or request["seed"].tokens + self.num_spec_tokens <= self.draft_limit
            ):
                raise ValueError("draft skip would leave KV required by native Decode uninitialized")
            # A skipped MTP frame carries progress, never dummy-forward bytes.
            # At this prompt length the same native limit also skips every
            # later D draft, so no MTP source or destination is required.
            plan = (
                None
                if skipped
                else plan_layer(
                    source,
                    memory.describe(layer),
                    message["tables"],
                    tables,
                    tokens,
                    first_block=first,
                    final=final,
                    member=member,
                    ratio=ratio,
                    offload_main=memory.offload_main,
                )
            )
            owner = (layer, member)
            previous = [
                job
                for job in self.jobs.values()
                if job.request_id == request["id"] and (job.layer, job.member) == owner
            ]
            expected_first = max((job.tokens // BLOCK_SIZE for job in previous), default=0)
            if first != expected_first or any(not job.finished for job in previous) or owner in request["final"]:
                raise ValueError("duplicate, missing or out-of-order layer chunk")
            if previous and any(job.producer != producer for job in previous):
                raise ValueError("P contributor identity changed mid-request")
            count = (
                {"cpu": 0, "npu": 0}
                if skipped
                else {"cpu": (len(plan.full_blocks) + MAX_SEGMENTS - 1) // MAX_SEGMENTS, "npu": len(plan.hbm.batches())}
            )
            job_id = uuid4().hex
            self.jobs[job_id] = PullJob(
                request["id"],
                message["generation"],
                producer,
                layer,
                first,
                tokens,
                final,
                member,
                ratio,
                plan,
                count,
            )
            if request["failed"]:
                # Continue draining participant notifications without issuing
                # new DMA for a receive that will enter native error handling.
                count.update(cpu=0, npu=0)
                self.jobs[job_id].error = RuntimeError("an earlier request contribution failed")
            result = {"job": job_id, "count": count}
            if self.diagnostics and self.diagnostics.active(request["id"]):
                self.jobs[job_id].diagnostic = True
                result["trace"] = True
                self.diagnostics.emit(
                    "PULL_PLAN",
                    job_id,
                    request=request["id"],
                    generation=message["generation"],
                    layer=layer,
                    member=member,
                    ratio=ratio,
                    first=first,
                    tokens=tokens,
                    final=final,
                    draft_skipped=skipped,
                    count=count,
                    source_tables={k: dict(count=len(v), first=list(v[:4])) for k, v in message["tables"].items()},
                    destination_tables={k: dict(count=len(v), first=list(v[:4])) for k, v in tables.items()},
                )
                if self.diagnostics.verify and plan is not None and not request["failed"]:
                    expected_destinations = self.diagnostics.spans(memory, tables, tokens, first, final, member, ratio)[
                        "npu"
                    ]
                    destinations = tuple((b.destinations, b.lengths) for b in plan.hbm.batches())
                    self.diagnostics.address_check(
                        expected_destinations,
                        destinations,
                        job_id,
                        request=request["id"],
                        generation=message["generation"],
                        layer=layer,
                        path="D_HBM_destination",
                    )
                    result["verify"] = {
                        "npu": tuple((b.sources, b.lengths) for b in plan.hbm.batches()),
                        "cpu": tuple(
                            (
                                plan.main_sources[i : i + MAX_SEGMENTS],
                                (BLOCK_SIZE * 656,) * len(plan.full_blocks[i : i + MAX_SEGMENTS]),
                            )
                            for i in range(0, len(plan.full_blocks), MAX_SEGMENTS)
                        ),
                    }
            return result

    def _pull(self, job, message):
        path, batch = message["path"], message["batch"]
        ready = message["ready"]
        with self.runtime.lock:
            if job.error is not None or job.finished:
                raise RuntimeError("Pull job is terminal")
            if path not in job.count or not 0 <= batch < job.count[path]:
                raise ValueError("unexpected Pull batch")
            self.sources[job.producer, job.layer].check_ready(path, ready)
            key = (path, batch)
            if key in job.completed:
                if job.completed[key] != ready.sequence:
                    raise ValueError("another Ready reused a completed batch")
                return True
            if batch and (path, batch - 1) not in job.completed:
                raise ValueError("Pull batches must remain ordered")
            try:
                self.runtime.wait_compute()
                if path == "npu":
                    transfer = self.raw_transfer(
                        self.runtime,
                        ready,
                        job.plan.hbm.batches()[batch],
                        self.memories[job.layer].tensor,
                        self.stream_factory(),
                    )
                    job.transfers.append(transfer)
                    try:
                        transfer.run()
                    except Exception:
                        if not self.runtime.pending_raw_addr:
                            raise
                        transfer.recover()
                else:
                    transfer = self._cpu_pull(job, batch, ready)
                if job.diagnostic:
                    self._diagnose_pull(job, path, batch, ready, transfer, message.get("digests"))
                job.completed[key] = ready.sequence
                return True
            except Exception as exc:
                if job.diagnostic:
                    self.diagnostics.emit(
                        "DIAGNOSTIC_OR_PULL_ERROR",
                        (job.request_id, job.layer, job.member, path, batch),
                        request=job.request_id,
                        generation=job.generation,
                        layer=job.layer,
                        path=path,
                        batch=batch,
                        status="FAIL",
                        error=str(exc),
                    )
                if self.diagnostics and self.diagnostics.retained is not None:
                    self.runtime.failure = exc
                    raise
                job.error = exc
                transfer = job.transfers[-1] if job.transfers else None
                if self._resolve_failure(transfer, job):
                    job.completed[key] = ready.sequence
                    # P must prove release/cancellation with the SDK before
                    # sending complete; no failed receive is exposed earlier.
                    return {"error": str(exc)}
                # No fresh Put/Ready and no replay. Retain original handles and
                # storage; native Decode Get must not cross unresolved DMA.
                self.runtime.failure = exc
                raise

    def _resolve_failure(self, transfer, job):
        if isinstance(transfer, CpuPullTransfer):
            if transfer.committed:
                return True
            try:
                transfer.abort()
            except Exception:
                return False
            group = self.writer.groups[self.pool.indices[job.layer]]
            pool = self.writer.slots[group.geometry]
            pool.free.extend(pool.release([entry.key for entry in transfer.put.entries]))
            return True
        return isinstance(transfer, NpuPullTransfer) and not self.runtime.pending_raw_addr

    def _cpu_pull(self, job, batch, ready):
        group_index = self.pool.indices[job.layer]
        group = self.writer.groups[group_index]
        state = self.writer.requests[job.request_id][group_index]
        begin = batch * MAX_SEGMENTS
        blocks = job.plan.full_blocks[begin : begin + MAX_SEGMENTS]
        if not blocks or blocks[0] != len(state.keys):
            raise ValueError("DDR main prefix has a gap or a duplicate")
        keys = self.writer.slots[group.geometry].take(len(blocks))
        # Retain reserved keys even if Prepare raises after a remote decision.
        reservation = {"keys": keys, "put": None}
        job.transfers.append(reservation)
        put = self.runtime.prepare_put(keys, [BLOCK_SIZE * 656] * len(keys))
        reservation["put"] = put
        partial = type(job.plan)(tuple(blocks), job.plan.main_sources[begin : begin + len(blocks)], Segments())
        transfer = CpuPullTransfer(self.runtime, ready, put, partial.ddr_segments(put.entries, keys))
        job.transfers[-1] = transfer
        try:
            transfer.submit()
        except Exception:
            if not put.submitted:
                raise
        try:
            transfer.finish(int(self.timeout * 1000))
        except Exception:
            # Resolve the original Wait/Commit only. A sticky Commit decision
            # is never replaced with an Abort or a newly prepared transaction.
            transfer.finish(int(self.timeout * 1000))
        transfer.publish(self.writer, group, state)
        return transfer

    def _diagnose_pull(self, job, path, batch, ready, transfer, expected):
        diag = self.diagnostics
        fields = dict(
            request=job.request_id,
            generation=job.generation,
            layer=job.layer,
            member=job.member,
            path=path,
            batch=batch,
            sequence=ready.sequence,
        )
        key = (job.request_id, job.generation, job.layer, job.member, job.first, job.tokens, path, batch)
        diag.emit(
            "PULL_DONE",
            key,
            **fields,
            state="COMMITTED_PUBLISHED" if path == "cpu" else "DONE",
            api="submit_cpu_pull/wait_cpu_pull/commit_put" if path == "cpu" else "batch_get_raw_addr",
            bytes=sum(transfer.segments.lengths),
            segments=len(transfer.segments.lengths),
            source_offsets=list(transfer.segments.sources[:4]),
            destinations=list(transfer.segments.destinations[:4]),
        )
        if diag.verify:
            if path == "npu":
                actual = diag.hashes(transfer.output, transfer.segments.destinations, transfer.segments.lengths)
            else:
                actual = diag.ddr_hashes(self.runtime, transfer, self.memories[job.layer].tensor.device)
            diag.compare(expected, actual, key, **fields)

    def _check_replicas(self, job, evidence):
        request = next(value for value in self.requests.values() if value["id"] == job.request_id)
        end = job.tokens // BLOCK_SIZE + int(job.final and job.tokens % BLOCK_SIZE > 0)
        blocks = () if job.plan is None else tuple(range(job.first, end))
        reference = request.setdefault("diagnostic_replicas", {})
        mismatches = []
        if evidence is None or len(evidence) != len(blocks):
            mismatches.append("missing_evidence")
        else:
            for expected_block, item in zip(blocks, evidence):
                block, size, digest = item
                wanted = min(BLOCK_SIZE, job.tokens - expected_block * BLOCK_SIZE) * 656
                if block != expected_block or size != wanted or not isinstance(digest, str) or len(digest) != 64:
                    mismatches.append(expected_block)
                    continue
                key = (job.layer, block, size)
                if reference.setdefault(key, digest) != digest:
                    mismatches.append(block)
        job.replicas_checked = True
        self.diagnostics.emit(
            "MAIN_REPLICA_CHECK",
            (job.request_id, job.generation, job.layer, job.member, job.first, job.tokens),
            request=job.request_id,
            generation=job.generation,
            layer=job.layer,
            member=job.member,
            ratio=job.ratio,
            blocks=len(blocks),
            status="FAIL" if mismatches else "PASS",
            mismatches=mismatches[:4],
        )
        if mismatches:
            job.error = RuntimeError("P TP main KV replicas differ or their evidence is incomplete")

    def _complete(self, job):
        with self.runtime.lock:
            if job.finished:
                return True
            if self.runtime.failure is not None or job.error is None and len(job.completed) != sum(job.count.values()):
                raise RuntimeError("layer cannot complete before all Pull batches")
            job.finished = True
            # SDK completion and P release are established. Keep small replay/
            # chunk records, without retaining per-batch NPU descriptor arrays
            # for the remainder of a long Decode request.
            job.transfers.clear()
            job.plan = None
            request = next(value for value in self.requests.values() if value["id"] == job.request_id)
            owner = (job.layer, job.member)
            request["failed"] |= job.error is not None
            request["paths"].setdefault(owner, set()).update(path for path, count in job.count.items() if count)
            if job.final:
                request["final"][owner] = job
            try:
                self._maybe_publish(request)
            except Exception as exc:
                # All required reads have already drained at this boundary.
                # Catalog/tail handoff failure must reach native load-error
                # handling rather than silently leaving D waiting forever.
                request["failed"] = True
                job.error = exc
                self._fail_request(request)
            return True

    def _maybe_publish(self, request):
        ratio = request["cohort"][1]
        if len(request["final"]) != len(self.memories) * ratio or request["published"]:
            return
        seed, req_id = request["seed"], request["id"]
        if request["failed"]:
            self._fail_request(request)
            return
        needs = {
            Contribution(layer, str(member), 0): frozenset(
                "ddr" if path == "cpu" else "hbm" for path in request["paths"].get((layer, member), ())
            )
            for layer in self.memories
            for member in range(ratio)
        }
        gate = CompletionGate(req_id, int(seed.generation, 16), needs)
        for key, paths in needs.items():
            if "hbm" in paths:
                gate.record(req_id, gate.epoch, key, "hbm_done")
            if "ddr" in paths:
                gate.record(req_id, gate.epoch, key, "ddr_committed")
                gate.record(req_id, gate.epoch, key, "ddr_published")
            gate.record(req_id, gate.epoch, key, "ack")
        for layer, index in self.pool.indices.items():
            state = self.writer.requests[req_id][index]
            if len(state.keys) != seed.tokens // BLOCK_SIZE or len(state.addresses) != len(state.keys):
                raise RuntimeError("incomplete committed Target DDR directory")
            if seed.tokens % BLOCK_SIZE:
                slot = self.pool.policy.receive_slots[req_id]
                self.pool.policy.bind_received_tail(
                    req_id, layer, self.memories[layer].planes["main"][slot], seed.tokens
                )
        if not gate.decode_ready:
            raise RuntimeError("request completion gate is not satisfied")
        request["published"] = True
        if self.diagnostics and self.diagnostics.active(req_id):
            self.diagnostics.emit(
                "DECODE_READY",
                (req_id, seed.generation),
                request=req_id,
                generation=seed.generation,
                contributions=len(request["final"]),
                expected=len(self.memories) * ratio,
                tokens=seed.tokens,
                full_blocks=seed.tokens // BLOCK_SIZE,
                tail_tokens=seed.tokens % BLOCK_SIZE,
                verify_kv=self.diagnostics.verify,
                verified_batches=sum(len(job.completed) for job in self.jobs.values() if job.request_id == req_id)
                if self.diagnostics.verify
                else 0,
                replica_checks=sum(job.replicas_checked for job in self.jobs.values() if job.request_id == req_id),
            )
        self.writer.complete_receive(req_id)
        self.finished.add(req_id)

    def _fail_request(self, request):
        if any(not job.finished for job in self.jobs.values() if job.request_id == request["id"]):
            raise RuntimeError("retain cancelled request until original Pull jobs drain")
        self.errors.update(block for group in request["seed"].blocks for block in group)
        self.writer.complete_receive(request["id"])
        self.finished.add(request["id"])

    def get_finished(self):
        with self.runtime.lock:
            if self.runtime.failure is not None:
                raise RuntimeError(
                    "PD transfer unresolved; retain source and destination memory"
                ) from self.runtime.failure
            result, self.finished = self.finished, set()
            return result

    def get_errors(self):
        with self.runtime.lock:
            result, self.errors = self.errors, set()
            return result

    def release(self, request_ids):
        with self.runtime.lock:
            self.native_finished.update(request_ids & {request["id"] for request in self.requests.values()})
            for external_id, request in tuple(self.requests.items()):
                if request["id"] in self.native_finished:
                    if not (request["published"] or request["cancelled"] or request["failed"]):
                        continue
                    self.writer.release_requests((request["id"],))
                    self.native_finished.discard(request["id"])
                    self.requests.pop(external_id)
                    self._forget_request_jobs(request["id"])

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            if any(not job.finished for job in self.jobs.values()):
                raise RuntimeError("PD receiver retains unfinished transfers; resolve before shutdown")
            self.closed = True
            self.runtime.wait_compute()
            for source in self.sources.values():
                self.runtime.detach_hbm(source.npu_ticket)
            self.sources.clear()
        with self.condition:
            self.condition.notify_all()
        if self.server is not None:
            self.server.close()
