# SPDX-License-Identifier: Apache-2.0
"""P-side per-layer publication and physical-storage reuse gates."""

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from copy import deepcopy
from threading import Event
from types import SimpleNamespace

from .control import call
from .registration import LayerRegistration


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
    ):
        self.sdk, self.memories, self.config = sdk, memories, config
        self.producer, self.host, self.device_id = producer, host, device_id
        self.cpu_eid, self.npu_eid = cpu_eid, npu_eid
        self.draft_limit, self.num_spec_tokens = draft_limit, num_spec_tokens
        self.activate, self.stream_factory, self.event_factory = activate, stream_factory, event_factory
        self.external_id, self.tp_rank, self.tp_size, self.dp_rank = external_id, tp_rank, tp_size, dp_rank
        self.registrations, self.peers, self.futures = {}, {}, {}
        self.executor = ThreadPoolExecutor(max_workers=min(len(memories), 8), thread_name_prefix="kvaf-pd-send")
        self.metadata = None
        self.dispatched, self.draft_written = set(), set()
        self.draft_skipped = self.draft_completed = False
        self.native_finished, self.final_submitted, self.active_requests = set(), set(), {}
        self.forward_pending = set()
        self.closed = False
        self.bootstrap = sdk.NpuRuntime(device_id=device_id, enable_cpu_offload=True)
        self.bootstrap.initialize()
        for endpoint in config.decode_endpoints:
            self._register_peer(endpoint)

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
        self.peers[endpoint] = hello["peer"]
        stream = self.stream_factory()
        for layer, memory in self.memories.items():
            registration = LayerRegistration(
                self.sdk,
                hello["peer"],
                layer,
                memory.tensor,
                memory.planes,
                device_id=self.device_id,
                cpu_eid=self.cpu_eid,
                npu_eid=self.npu_eid,
            )
            self.registrations[endpoint, layer] = registration
            metadata = registration.register(
                hello["ticket"], stream, listen_address="0.0.0.0:0", advertised_address=f"{self.host}:0"
            )
            self._call(endpoint, {"op": "register", "metadata": metadata, "offload": memory.offload_main})

    def _call(self, endpoint, message):
        return call(
            endpoint, dict(message, peer=self.peers[endpoint], producer=self.producer), timeout=self.config.timeout
        )

    def bind(self, metadata):
        self.metadata = deepcopy(metadata)
        self.dispatched, self.draft_written = set(), set()
        self.draft_skipped = self.draft_completed = False
        self.forward_pending.update(metadata.requests)
        for req_id, request in metadata.requests.items():
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
        event = self.event_factory()
        # Snapshot host metadata: the runner may bind the next scheduler step
        # while this layer is still being read remotely.
        requests = tuple(
            (req_id, request, self.metadata.generations[req_id]) for req_id, request in self.metadata.requests.items()
        )
        self.futures[layer] = self.executor.submit(self._send_layer, layer, requests, event, skipped)

    def _send_layer(self, layer, requests, event, skipped=False):
        self.activate()
        event.synchronize()
        memory = self.memories[layer]
        for req_id, request, generation in requests:
            if request.remote_tp_size is None or request.remote_tp_size < 1 or self.tp_size % request.remote_tp_size:
                raise ValueError("P TP must be a multiple of D TP")
            ratio = self.tp_size // request.remote_tp_size
            member = self.tp_rank % ratio
            d_rank = self.tp_rank // ratio
            endpoint = f"{request.remote_host}:{request.remote_port + d_rank}"
            if endpoint not in self.peers:
                raise ValueError(f"D worker {endpoint} is outside the configured full mesh")
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
            registration = self.registrations[endpoint, layer]
            failed = False
            for path in ("npu", "cpu"):
                for batch in range(planned["count"][path]):
                    ready = registration.ready(path, event)
                    operation = dict(
                        op="pull", job=planned["job"], generation=generation, path=path, batch=batch, ready=ready
                    )
                    try:
                        result = self._call(endpoint, operation)
                    except (ConnectionError, OSError, TimeoutError):
                        # Query only: never replay READ after a lost reply.
                        status = self._call(endpoint, dict(op="status", job=planned["job"], generation=generation))
                        if status["completed"].get((path, batch)) != ready.sequence:
                            raise RuntimeError("Pull ACK lost with unresolved original Ready; retain source") from None
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
                if failed:
                    break
            self._complete_job(endpoint, planned["job"], generation)

    def _complete_job(self, endpoint, job_id, generation):
        message = dict(job=job_id, generation=generation)
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
        for req_id in cancelled & self.active_requests.keys():
            request, generation = self.active_requests[req_id]
            ratio = self.tp_size // request.remote_tp_size
            endpoint = f"{request.remote_host}:{request.remote_port + self.tp_rank // ratio}"
            self._call(
                endpoint,
                dict(
                    op="cancel",
                    request=self.external_id(req_id),
                    generation=generation,
                    member=self.tp_rank % ratio,
                    ratio=ratio,
                    cohort=f"{self.producer.rsplit(':', 1)[0]}:{self.tp_rank // ratio}",
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
        self.executor.shutdown(wait=True)
        for future in self.futures.values():
            future.result()
        for endpoint in self.peers:
            with suppress(OSError):
                self._call(endpoint, {"op": "detach"})
        for registration in self.registrations.values():
            registration.close()
        self.bootstrap.shutdown()
        self.closed = True
