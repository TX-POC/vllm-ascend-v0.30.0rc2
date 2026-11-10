# SPDX-License-Identifier: Apache-2.0
"""P-side per-layer publication and physical-storage reuse gates."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from copy import deepcopy
from threading import Event
from types import SimpleNamespace

from .control import call
from .registration import PULL_API_VERSION, PeerRegistration

INITIALIZATION_CLEANUP_TIMEOUT = 5.0


class _ProductionEvent:
    """Wait once for one layer dispatch's freshly recorded event.

    Owned by a single send task, never shared across tasks or re-recorded.
    Registration.ready keeps its normal wait contract; repeated batches use
    this task's completed wait without another native synchronization.
    """

    def __init__(self, event):
        self._event = event
        self._complete = False

    def synchronize(self):
        if not self._complete:
            self._event.synchronize()
            self._complete = True


class Sender:
    def __init__(
        self,
        sdk,
        memories,
        *,
        config,
        device_id,
        producer,
        host,
        activate,
        stream_factory,
        event_factory,
        external_id,
        tp_rank,
        tp_size,
        dp_rank,
        cpu_eid="",
        npu_eid="",
        draft_limit=None,
        num_spec_tokens=3,
        diagnostics=None,
    ):
        self.sdk, self.memories, self.config = sdk, memories, config
        self.producer, self.host, self.device_id = producer, host, device_id
        self.cpu_eid, self.npu_eid = cpu_eid, npu_eid
        self.draft_limit, self.num_spec_tokens = draft_limit, num_spec_tokens
        self.diagnostics = diagnostics
        self.activate, self.stream_factory, self.event_factory = activate, stream_factory, event_factory
        self.external_id, self.tp_rank, self.tp_size, self.dp_rank = external_id, tp_rank, tp_size, dp_rank
        self.active = not config.source_tp0_only or tp_rank == 0
        self.registrations, self.peers, self.futures = {}, {}, {}
        self.source_groups = {}
        self.executor = None
        self.metadata = None
        self.dispatched, self.draft_written = set(), set()
        self.draft_skipped = self.draft_completed = False
        self.native_finished, self.final_submitted, self.active_requests = set(), set(), {}
        self.forward_pending = set()
        self.closed = False
        self.bootstrap = None
        self._initialization_attempted = False
        self._initialization_failed = False
        self._registered = set()
        self._cleanup_barriers = set()

    def initialize(self):
        # The connector owns this object before any SDK/remote side effects.
        # A failed initialization can therefore retain unresolved sources for
        # shutdown, rather than losing their owner in a failed constructor.
        if self._initialization_attempted or self.closed:
            raise RuntimeError("PD sender initialization cannot repeat; resolve/close the original sources")
        self._initialization_attempted = True
        if not self.active:
            return  # Keep native per-rank completion bookkeeping, without SDK or communication resources.
        logging.getLogger("vllm").info(
            "KVAF PD source registration budget: producer=%s device=%s D_peers=%s layers=%s "
            "mrs_per_path=%s source_groups_per_path=%s source_tp0_only=%s",
            self.producer,
            self.device_id,
            len(self.config.decode_endpoints),
            len(self.memories),
            len(self.config.decode_endpoints) * len(self.memories),
            len(self.config.decode_endpoints),
            self.config.source_tp0_only,
        )
        try:
            self.executor = ThreadPoolExecutor(
                max_workers=min(len(self.memories), 8), thread_name_prefix="kvaf-pd-send"
            )
            self.bootstrap = self.sdk.NpuRuntime(device_id=self.device_id, enable_cpu_offload=True)
            self.bootstrap.initialize()
            for endpoint in self.config.decode_endpoints:
                self._register_peer(endpoint)
        except Exception as exc:
            self._initialization_failed = True
            logging.getLogger("vllm").error(
                "KVAF PD initialization failed: producer=%s device=%s completed=%s attempted=%s; "
                "rolling back confirmed registrations; error=%s",
                self.producer,
                self.device_id,
                len(self._registered),
                len(self.registrations),
                exc,
            )
            try:
                self._close_sources(timeout=min(self.config.timeout, INITIALIZATION_CLEANUP_TIMEOUT))
            except Exception:
                # Preserve the original registration error and this object's
                # remaining source/Tensor owners; never turn cleanup into a
                # retry of registration or an unconfirmed remote detach.
                logging.getLogger("vllm").exception(
                    "KVAF PD initialization cleanup incomplete: producer=%s retained_registrations=%s "
                    "retained_peers=%s; retain source memory and quiesce peers before recovery",
                    self.producer,
                    len(self.registrations),
                    len(self.peers),
                )
            else:
                logging.getLogger("vllm").warning(
                    "KVAF PD initialization cleanup complete: producer=%s; sender closed, registration not retried",
                    self.producer,
                )
            raise
        if self.config.source_tp0_only:
            logging.getLogger("vllm").info(
                "KVAF PD source mode: mode=tp0_only dp_rank=%s tp_rank=%s compute_tp_size=%s "
                "D_peers=%s layers=%s mrs_per_path=%s source_groups_per_path=%s",
                self.dp_rank,
                self.tp_rank,
                self.tp_size,
                len(self.config.decode_endpoints),
                len(self.memories),
                len(self.registrations),
                len(self.source_groups),
            )

    def _register_peer(self, endpoint):
        deadline = time.monotonic() + self.config.timeout
        while True:
            try:
                hello = call(endpoint, {"op": "hello"}, timeout=min(2, self.config.timeout))
                break
            except (OSError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"D worker is not ready at {endpoint}") from None
                Event().wait(0.1)
        if (hello.get("draft_limit"), hello.get("num_spec_tokens", 3)) != (
            self.draft_limit,
            self.num_spec_tokens,
        ):
            raise ValueError("P/D native drafter limits and speculative widths must match")
        if hello.get("verify_kv", False) != bool(self.diagnostics and self.diagnostics.verify):
            raise ValueError("P/D verify_kv modes must match; restart both with the same diagnostic configuration")
        if hello.get("source_tp0_only", False) is not self.config.source_tp0_only:
            raise ValueError("P/D source_tp0_only modes must match before source registration")
        self.peers[endpoint] = hello["peer"]
        if hello.get("source_cleanup_v1", False) is True:
            self._cleanup_barriers.add(endpoint)
        if hello.get("pull_api_version") != PULL_API_VERSION:
            raise ValueError("P/D require KVAF multi-MR Pull API v2; update both framework and SDK")
        group = PeerRegistration(
            self.sdk,
            hello["peer"],
            self.memories,
            hello["ticket"],
            device_id=self.device_id,
            cpu_eid=self.cpu_eid,
            npu_eid=self.npu_eid,
        )
        self.source_groups[endpoint] = group
        self.registrations.update(((endpoint, name), view) for name, view in group.layers.items())
        layer = "<group>"
        try:
            group.register(hello["receiver"], self.stream_factory())
            group.stage = "D_register"
            for layer, registration in group.layers.items():
                self._call(
                    endpoint,
                    {"op": "register", "metadata": registration.metadata, "offload": self.memories[layer].offload_main},
                )
                self._registered.add((endpoint, layer))
        except Exception:
            logging.getLogger("vllm").exception(
                "KVAF PD source registration failed: producer=%s device=%s endpoint=%s layer=%s "
                "stage=%s completed=%s attempted_mrs=%s source_groups=%s cpu_registered=%s npu_registered=%s",
                self.producer,
                self.device_id,
                endpoint,
                layer,
                group.stage,
                len(self._registered),
                len(self.registrations),
                len(self.source_groups),
                group.cpu_registered,
                group.registered,
            )
            raise
        if self.diagnostics:
            self.diagnostics.emit(
                "REGISTERED",
                endpoint,
                peer=hello["peer"],
                endpoint=endpoint,
                producer=self.producer,
                layers=len(self.memories),
            )

    def _call(self, endpoint, message, *, timeout=None):
        if self.config.source_tp0_only:
            message = dict(message, source_tp0_only=True, source_tp_rank=self.tp_rank)
        return call(
            endpoint,
            dict(message, peer=self.peers[endpoint], producer=self.producer),
            timeout=self.config.timeout if timeout is None else timeout,
        )

    def bind(self, metadata):
        self.metadata = deepcopy(metadata)
        self.dispatched, self.draft_written = set(), set()
        self.draft_skipped = self.draft_completed = False
        self.forward_pending.update(metadata.requests)
        for req_id, request in metadata.requests.items():
            if self.diagnostics:
                self.diagnostics.select(self.external_id(req_id), metadata.generations[req_id], alias=req_id)
            self.active_requests[req_id] = (request, metadata.generations[req_id])
            if request.chunk_finish:
                self.final_submitted.add(req_id)
        for req_id, params in getattr(metadata, "cancellations", {}).items():
            self.active_requests[req_id] = (SimpleNamespace(**params), params["kvaf_generation"])

    def wait_layer(self, layer):
        future = self.futures.get(layer)
        if future is not None:
            # Failed ACKs retain Ready and layer memory; propagating the failure
            # stops subsequent scatter rather than reusing uncertain storage.
            future.result(timeout=self.config.timeout)

    def wait_all(self):
        # Graph replay omits Python layer-entry hooks. Drain the previous
        # frame before any native cache zeroing or graph scatter can reuse it.
        for layer in self.futures:
            self.wait_layer(layer)

    def written(self, layer):
        if self.metadata is None or not self.metadata.requests or layer not in self.memories:
            return
        if not self.memories[layer].offload_main:
            # Native MTP may write this cache more than once. Dispatch its
            # final prompt snapshot at deferred native connector finalization.
            if not self.draft_skipped:
                self.draft_written.add(layer)
            return
        self._dispatch(layer)

    def skip_drafting(self, limit):
        if self.metadata is None or not self.metadata.requests:
            return
        if limit != self.draft_limit or any(
            request.prompt_len + self.num_spec_tokens <= limit for request in self.metadata.requests.values()
        ):
            raise RuntimeError("native draft skip cannot omit KV required by a future D draft")
        # Native DP alignment may execute dummy MTP forwards. Their hooks do
        # not describe request KV and must never trigger a real source Pull.
        self.draft_skipped = True
        self.draft_written.clear()

    def complete_drafting(self):
        if self.draft_skipped:
            raise RuntimeError("one native frame cannot both skip and complete drafting")
        self.draft_completed = True

    def _dispatch(self, layer, *, skipped=False):
        if layer in self.dispatched:
            return
        self.wait_layer(layer)
        self.dispatched.add(layer)
        if not self.active:
            return  # Native writes/graph/MTP still finish before local send-complete is reported.
        event = self.event_factory()
        # Snapshot host metadata: the runner may bind the next scheduler step
        # while this layer is still being read remotely.
        requests = tuple(
            (req_id, request, self.metadata.generations[req_id]) for req_id, request in self.metadata.requests.items()
        )
        self.futures[layer] = self.executor.submit(self._send_layer, layer, requests, event, skipped)

    def _send_layer(self, layer, requests, event, skipped=False):
        self.activate()
        event = _ProductionEvent(event)
        event.synchronize()
        memory = self.memories[layer]
        for req_id, request, generation in requests:
            if request.remote_tp_size is None or request.remote_tp_size < 1 or self.tp_size % request.remote_tp_size:
                raise ValueError("P TP must be a multiple of D TP")
            ratio, member, d_rank = self._contributors(request)
            endpoint = f"{request.remote_host}:{request.remote_port + d_rank}"
            if endpoint not in self.peers:
                raise ValueError(f"D worker {endpoint} is outside the configured full mesh")
            registration = self.registrations[endpoint, layer]
            with registration.owner.operation(self.config.timeout):
                message = {
                    "op": "plan",
                    "request": self.external_id(req_id),
                    "generation": generation,
                    "cohort": f"{self.producer.rsplit(':', 1)[0]}:{d_rank}",
                    "layer": layer,
                    "first": request.local_transed_tokens // 128,
                    "tokens": min(request.local_computed_tokens, request.prompt_len),
                    "final": request.chunk_finish,
                    "member": member,
                    "ratio": ratio,
                    "tables": memory.tables(request.local_block_ids),
                    "draft_skipped": skipped,
                }
                planned = self._call(endpoint, message)
                if "verify" in planned:
                    expected = self.diagnostics.spans(
                        memory,
                        message["tables"],
                        message["tokens"],
                        message["first"],
                        message["final"],
                        member,
                        ratio,
                        skipped,
                    )
                    self.diagnostics.address_check(
                        expected,
                        planned["verify"],
                        planned["job"],
                        request=message["request"],
                        generation=generation,
                        layer=layer,
                        path="P_source",
                    )
                replicas = None
                if self.diagnostics and self.diagnostics.verify and planned.get("trace"):
                    layout = memory.describe(layer)
                    end = message["tokens"] // 128 + int(message["final"] and message["tokens"] % 128 > 0)
                    replicas = []
                    for block in () if skipped else range(message["first"], end):
                        size = min(128, message["tokens"] - block * 128) * 656
                        offset = layout.offset("main", message["tables"]["main"], block)
                        replicas.append((block, size, self.diagnostics.hashes(memory.tensor, (offset,), (size,))[0]))
                if self.diagnostics and planned.get("trace"):
                    self.diagnostics.emit(
                        "SEND_PLAN",
                        planned["job"],
                        request=message["request"],
                        generation=generation,
                        layer=layer,
                        member=member,
                        ratio=ratio,
                        first=message["first"],
                        tokens=message["tokens"],
                        final=message["final"],
                        draft_skipped=skipped,
                        count=planned["count"],
                        endpoint=endpoint,
                    )
                failed = False
                for path in ("npu", "cpu"):
                    for batch in range(planned["count"][path]):
                        digests = None
                        if "verify" in planned:
                            sources, lengths = planned["verify"][path][batch]
                            digests = self.diagnostics.hashes(memory.tensor, sources, lengths)
                        ready = registration.ready(path, event)
                        operation = dict(
                            op="pull", job=planned["job"], generation=generation, path=path, batch=batch, ready=ready
                        )
                        if digests is not None:
                            operation["digests"] = digests
                        if self.diagnostics and planned.get("trace"):
                            self.diagnostics.emit(
                                "SOURCE_READY",
                                (planned["job"], path, batch),
                                request=message["request"],
                                layer=layer,
                                path=path,
                                batch=batch,
                                sequence=ready.sequence,
                                source=ready.source.source_id,
                                source_bytes=memory.tensor.numel() * memory.tensor.element_size(),
                                mr_id=ready.mr_id,
                            )
                        try:
                            result = self._call(endpoint, operation)
                        except (ConnectionError, OSError, TimeoutError):
                            # Query only: never replay READ after a lost reply.
                            status = self._call(endpoint, dict(op="status", job=planned["job"], generation=generation))
                            if status["completed"].get((path, batch)) != ready.sequence:
                                raise RuntimeError(
                                    "Pull ACK lost with unresolved original Ready; retain source"
                                ) from None
                            result = {"error": status["error"]} if status["error"] else True
                        failed = isinstance(result, dict) and "error" in result
                        if failed:
                            try:
                                registration.release(path, ready)
                            except Exception:
                                registration.cancel(path, ready)
                                registration.release(path, ready)
                            break
                        registration.release(path, ready)
                        if self.diagnostics and planned.get("trace"):
                            self.diagnostics.emit(
                                "SOURCE_RELEASED",
                                (planned["job"], path, batch),
                                request=message["request"],
                                layer=layer,
                                path=path,
                                sequence=ready.sequence,
                            )
                    if failed:
                        break
                self._complete_job(endpoint, planned["job"], generation, replicas=replicas)

    def _contributors(self, request):
        if self.config.source_tp0_only:
            if not self.active or request.remote_tp_size != 1:
                raise ValueError("TP0-only transport requires P TP0 and D TP=1")
            # This is the transfer cohort, never a replacement for model TP.
            return 1, 0, 0
        ratio = self.tp_size // request.remote_tp_size
        return ratio, self.tp_rank % ratio, self.tp_rank // ratio

    def _complete_job(self, endpoint, job_id, generation, *, replicas=None):
        message = dict(job=job_id, generation=generation)
        if replicas is not None:
            message["replicas"] = replicas
        for attempt in range(2):
            try:
                self._call(endpoint, dict(message, op="complete"))
                break
            except (ConnectionError, OSError, TimeoutError):
                status = self._call(endpoint, dict(message, op="status"))
                if status["done"]:
                    break
                if attempt:
                    raise RuntimeError("layer completion remains unconfirmed; retain source") from None
                # complete is idempotent metadata; never replay its Pulls.
        # Completion is already proven. A lost cleanup ACK must not prevent P
        # storage reuse; D retains only a small terminal record until detach.
        with suppress(ConnectionError, OSError, TimeoutError, RuntimeError):
            self._call(endpoint, dict(message, op="ack_complete"))

    def finalize(self):
        if self.metadata is None or not self.metadata.requests:
            return
        for layer, memory in self.memories.items():
            if not memory.offload_main and (self.draft_skipped or self.draft_completed or layer in self.draft_written):
                self._dispatch(layer, skipped=self.draft_skipped)

    def complete_forward(self):
        if self.metadata is None or not self.metadata.requests:
            return
        self.finalize()
        # Called only after successful native sampling/finalization. Captured
        # target graphs may omit Python write hooks; all their KV is now valid.
        for layer, memory in self.memories.items():
            if memory.offload_main and layer not in self.dispatched:
                self._dispatch(layer)
        missing = self.memories.keys() - self.dispatched
        if missing:
            raise RuntimeError(f"native KV write hooks did not produce the required PD layers: {sorted(missing)}")
        self.forward_pending.difference_update(self.metadata.requests)

    def get_finished(self, finished_req_ids):
        self.native_finished.update(finished_req_ids & self.active_requests.keys())
        for future in self.futures.values():
            if future.done():
                future.result()  # Surface failures even when no further forward runs.
        if any(not future.done() for future in self.futures.values()):
            return set()
        # If native scheduling cancelled a partially-prefilled request, drain
        # its original layer work before letting D discard its allocated blocks.
        eligible = self.native_finished - self.forward_pending
        cancelled = eligible - self.final_submitted
        for req_id in cancelled & self.active_requests.keys() if self.active else ():
            request, generation = self.active_requests[req_id]
            ratio, member, d_rank = self._contributors(request)
            endpoint = f"{request.remote_host}:{request.remote_port + d_rank}"
            self._call(
                endpoint,
                dict(
                    op="cancel",
                    request=self.external_id(req_id),
                    generation=generation,
                    member=member,
                    ratio=ratio,
                    cohort=f"{self.producer.rsplit(':', 1)[0]}:{d_rank}",
                ),
            )
        done = eligible & self.active_requests.keys()
        self.native_finished.difference_update(done)
        for req_id in done:
            self.active_requests.pop(req_id)
            self.final_submitted.discard(req_id)
        return done

    def close(self):
        if self.closed:
            return
        if not self.active:
            self.closed = True
            return
        self._close_sources()

    def _close_sources(self, *, timeout=None):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        for future in self.futures.values():
            future.result()
        first_error = None
        for endpoint in tuple(self.peers):
            try:
                if self._initialization_failed and endpoint not in self._cleanup_barriers:
                    raise RuntimeError(
                        "D lacks source_cleanup_v1; initialization rollback cannot fence late register RPCs"
                    )
                # An ACK is required even after a register reply was lost.
                # D may still have imported a source or be finishing attach.
                self._call(endpoint, {"op": "detach"}, timeout=timeout)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                logging.getLogger("vllm").exception(
                    "KVAF PD source detach unconfirmed: producer=%s endpoint=%s; retain this peer's sources",
                    self.producer,
                    endpoint,
                )
                continue
            group = self.source_groups.get(endpoint)
            try:
                if group is not None:
                    group.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                logging.getLogger("vllm").exception(
                    "KVAF PD source shutdown unresolved: producer=%s endpoint=%s; retain all group MRs",
                    self.producer,
                    endpoint,
                )
                continue
            self.source_groups.pop(endpoint, None)
            for key in tuple(self.registrations):
                if key[0] == endpoint:
                    del self.registrations[key]
                    self._registered.discard(key)
            del self.peers[endpoint]
            self._cleanup_barriers.discard(endpoint)
        if first_error is not None:
            raise RuntimeError(
                "PD source cleanup incomplete; unresolved sources and bootstrap retained"
            ) from first_error
        if self.bootstrap is not None:
            self.bootstrap.shutdown()
        self.closed = True
