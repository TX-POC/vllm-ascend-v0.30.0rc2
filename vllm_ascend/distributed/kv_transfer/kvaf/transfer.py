# SPDX-License-Identifier: Apache-2.0
"""Put/Get execution and address publication; cache policy owns transfer selection."""

import logging
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .layout import BLOCK_SIZE, TOKEN_BYTES, KVCacheLayout
from .state import CacheGroup, DDRSlotPool, RequestCacheState, RequestGroupState

logger = logging.getLogger("vllm")


class KVTransfer:
    def __init__(
        self,
        runtimes: dict[tuple[int, int], Any],
        slot_counts: dict[tuple[int, int], int],
        layouts: tuple[KVCacheLayout, ...],
        kv_caches: dict[str, Any],
        *,
        verify=None,
    ):
        self.runtimes = runtimes
        self.slots = {geometry: DDRSlotPool(count) for geometry, count in slot_counts.items()}
        self.layouts = layouts
        self._regions_ready = False
        self.groups = []
        for layout in layouts:
            if "target" not in layout.roles or layout.cache_kind != "main":
                continue
            plane = layout.planes[0]
            if len(layout.planes) != 1 or not plane.direct_token_read or plane.token_bytes is None:
                raise ValueError("KVAF BatchPut requires one token-contiguous Target cache plane")
            cache = kv_caches[layout.layer_names[0]]
            cache = cache[0] if isinstance(cache, (tuple, list)) else cache
            geometry = (plane.token_bytes, layout.block_size)
            # Include page gaps in a storage view, without allocating or copying.
            # Keep this address/size stable so KVAF reuses its HBM registration.
            span_bytes = (plane.shape[0] - 1) * plane.page_stride_bytes + plane.page_bytes
            source = cache.as_strided((span_bytes // cache.element_size(),), (1,))
            self.groups.append(CacheGroup(layout, geometry, source))
        self.requests: dict[str, list[RequestGroupState]] = {}
        self.feedback = None
        self.policy = None
        self._logged: dict[str, set[str]] = {}
        self.verify = verify
        self._verified: set[str] = set()
        self.put_pages = self.put_bytes = self.released_pages = 0
        self.phase_pages = {"prefill": 0, "decode": 0}
        self.put_executor = self.put_stream = self.pending_put = None
        self.put_batches = self.put_waits = 0
        self.put_seconds = self.put_wait_seconds = 0.0
        self.put_last_log = time.monotonic()

    def configure_slot_regions(self):
        if self._regions_ready:
            return
        weights = {geometry: {} for geometry in self.slots}
        for i, layer in enumerate(self.groups):
            weights[layer.geometry][i] = 1
        geometry = self.groups[0].geometry
        # Match runtime_weights: old auxiliary snapshots and their active
        # borrowers coexist; Target prefixes themselves share references.
        aux = sum(
            2 * sum(p.page_bytes for p in layout.planes) / layout.block_size
            for layout in self.layouts
            if "target" not in layout.roles or layout.cache_kind != "main"
        )
        if aux:
            weights[geometry][None] = aux / geometry[0]
        for geometry, pool in self.slots.items():
            if weights[geometry]:
                pool.partition(weights[geometry])
        self._regions_ready = True
        logger.info(
            "KVAF DDR layout: layer-group regions; contiguous prompt reservations; shared prefix pages retained"
        )

    def release_requests(self, req_ids: Iterable[str]) -> None:
        """Free native DDR entries before making their slot IDs reusable."""
        req_ids = tuple(req_ids)
        if self.pending_put is not None and any(req in self.pending_put[1] for req in req_ids):
            self.poll_put(wait=True)  # No owner/key reuse while Put reads its ABC.
        for req_id in req_ids:
            if self.feedback is not None:
                self.feedback.release(req_id)
            states = self.requests.get(req_id)
            if states is None:
                continue
            pages = byte_count = freed_pages = 0
            for layer, state in zip(self.groups, states):
                pool = self.slots[layer.geometry]
                pool.recycle(pool.release(state.reserved))
                state.reserved.clear()
                if not state.keys:
                    state.progress.clear()
                    continue
                retired = self.slots[layer.geometry].release(state.keys)
                freed_pages += len(retired)
                if retired:
                    self.runtimes[layer.geometry].batch_free(retired)
                    self.slots[layer.geometry].recycle(retired)
                pages += len(state.keys)
                byte_count += len(state.keys) * layer.geometry[0] * layer.geometry[1]
                self.released_pages += len(state.keys)
                state.keys.clear()
                state.block_ids.clear()
                state.addresses.clear()
                state.progress.clear()
            del self.requests[req_id]
            self._logged.pop(req_id, None)
            if pages:
                logger.info(
                    "KVAF request released: request=%s bindings=%s bytes=%s freed_pages=%s",
                    req_id,
                    pages,
                    byte_count,
                    freed_pages,
                )

    def prepare_requests(self, requests, scheduled_tokens):
        """Prepare request metadata before forward; KVCachePolicy owns Target storage."""
        self.poll_put()
        self.configure_slot_regions()
        if self.feedback is not None:
            self.feedback.poll_feedback()
        if self.policy is not None:
            self.policy.requests = requests
        for req_id, scheduled in scheduled_tokens.items():
            if scheduled <= 0:
                continue
            if req_id not in self.requests:
                self.requests[req_id] = [
                    RequestGroupState(RequestCacheState(layer.layout.block_size)) for layer in self.groups
                ]
                logger.info("KVAF slot maps: request=%s layers=%s; host logical metadata", req_id, len(self.groups))
            self._update_hbm(self.requests[req_id], requests[req_id], scheduled)

    def _update_hbm(self, states, request, scheduled):
        # Compare block tables once per shared layout, not once per layer.
        # Immutable page tuples can be shared; the two mutable maps cannot.
        plans = {}
        for layer, state in zip(self.groups, states):
            layout = layer.layout
            key = (layout.group_ids, layout.block_size, layout.planes[0].shape[0])
            if key not in plans:
                pages = (request.num_computed_tokens + scheduled + layout.block_size - 1) // layout.block_size
                blocks = tuple(request.block_ids[layout.group_ids[0]][:pages])
                if blocks != state.progress.block_ids:
                    if any(tuple(request.block_ids[group][:pages]) != blocks for group in layout.group_ids[1:]):
                        raise ValueError("KVAF shared HBM cache groups have different block IDs")
                else:
                    blocks = state.progress.block_ids
                plans[key] = blocks
            state.progress.prepare(plans[key], request.num_computed_tokens, scheduled)

    def _read_addresses(self, layer, keys):
        """Snapshot native addresses only after Put succeeds; never derive them."""
        runtime = self.runtimes[layer.geometry]
        handle = runtime.prepare_access_table(keys)
        try:
            table = runtime.access_table_view(handle)
            addresses = list(table["addresses"])
            page_bytes = layer.geometry[0] * layer.geometry[1]
            if (
                len(addresses) != len(keys)
                or len(table["valid"]) != len(keys)
                or len(table["lengths"]) != len(keys)
                or any(address == 0 for address in addresses)
                or any(value != 1 for value in table["valid"])
                or any(length != page_bytes for length in table["lengths"])
            ):
                raise RuntimeError("KVAF Put succeeded but its DDR address table is invalid")
        finally:
            runtime.release_access_table(handle)
        return addresses

    @staticmethod
    def _publish_result(addresses, jobs):
        offset = 0
        for state, count in jobs:
            state.addresses.extend(addresses[offset : offset + count])
            offset += count

    def put_requests(self, requests: dict[str, Any], scheduled_tokens: dict[str, int]) -> None:
        """Publish stable staging/ABC pages through the serving cache policy."""
        self.policy.after_forward(requests, scheduled_tokens)

    def start_decode_put(self, device):
        import torch

        if self.verify is not None:
            logger.info("KVAF Put verification enabled: decode Put remains synchronous for safe Get readback")
            return
        if any(not getattr(runtime, "supports_background_put", False) for runtime in self.runtimes.values()):
            raise RuntimeError("Rebuild KVAF with scripts/install_kvip.sh for GIL-free background Put")
        self.put_stream = torch.npu.Stream(device=device)
        self.put_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kvaf-put")

    def _prepare_put(self, policy, targets):
        batches = []
        for i, layer in enumerate(self.groups):
            keys, offsets, jobs = [], [], []
            for req_id, tokens in targets.items():
                state = self.requests[req_id][i]
                start = len(state.keys)
                count = tokens // BLOCK_SIZE - start
                if count <= 0:
                    continue
                pool = self.slots[layer.geometry]
                if len(state.reserved) < count:
                    prompt = policy.requests[req_id].num_prompt_tokens
                    desired = max(count, (prompt + BLOCK_SIZE - 1) // BLOCK_SIZE - start)
                    previous = state.reserved[-1] if state.reserved else state.keys[-1] if state.keys else None
                    state.reserved.extend(
                        pool.take(
                            desired - len(state.reserved),
                            region=i,
                            after=None if previous is None else previous + 1,
                        )
                    )
                selected = [state.reserved.popleft() for _ in range(count)]
                # Reserved/in-flight keys are owned, but addresses stay unpublished.
                state.keys.extend(selected)
                keys.extend(selected)
                offsets.extend(
                    policy._source_offset(req_id, page) * policy.group_sizes[i] for page in range(start, start + count)
                )
                jobs.append((state, count))
            if keys:
                batches.append((i, keys, offsets, jobs))
        return batches

    def _execute_put(self, policy, batches, phase):
        results = []
        for i, keys, offsets, jobs in batches:
            layer = self.groups[i]
            runtime = self.runtimes[layer.geometry]
            source = policy.bytes_outputs[i].view(-1)
            runtime.batch_put(
                keys=keys, source=source, input_offsets=offsets, lengths=[BLOCK_SIZE * layer.geometry[0]] * len(keys)
            )
            addresses = self._read_addresses(layer, keys)
            if self.verify is not None and phase not in self._verified:
                self.verify(runtime, source, keys[0], offsets[0], layer.layout)
                self._verified.add(phase)
                logger.info(
                    "KVAF BatchPut verified: phase=%s layer=%s; DDR matches HBM", phase, layer.layout.layer_names[0]
                )
            results.append((i, addresses, jobs))
        return results

    def _finish_put(self, results, targets, phase):
        published = [req for req in targets if any(len(s.keys) > len(s.addresses) for s in self.requests[req])]
        written = 0
        for i, addresses, jobs in results:
            self._publish_result(addresses, jobs)
            written += len(addresses) * len(self.groups[i].layout.layer_names)
        self.put_pages += written
        self.put_bytes += written * BLOCK_SIZE * TOKEN_BYTES
        self.phase_pages[phase] += written
        for req_id in published:
            if written and phase not in self._logged.setdefault(req_id, set()):
                logger.info("KVAF first BatchPut: request=%s phase=%s; DDR addresses published", req_id, phase)
                self._logged[req_id].add(phase)

    def publish(self, policy, req_id, tokens):
        # Prefill and archive retain synchronous semantics and cannot race the
        # runtime's reusable Put descriptors or request-key cleanup.
        phase = "decode" if policy.plans.get(req_id) is None or policy.plans[req_id].decode else "prefill"
        self.publish_batch(policy, {req_id: tokens}, phase)

    def publish_batch(self, policy, targets, phase):
        self.poll_put(wait=True)
        batches = self._prepare_put(policy, targets)
        self._finish_put(self._execute_put(policy, batches, phase), targets, phase)

    def submit_decode(self, policy, targets):
        self.poll_put()
        if self.pending_put is not None:
            return  # Bounded queue; pressure is handled before ABC overwrite.
        batches = self._prepare_put(policy, targets)
        if not batches:
            return
        targets = {
            req: tokens
            for req, tokens in targets.items()
            if any(len(s.keys) > len(s.addresses) for s in self.requests[req])
        }
        if self.put_executor is None:  # CPU reference tests only.
            self._finish_put(self._execute_put(policy, batches, "decode"), targets, "decode")
            return
        import torch

        produced = torch.npu.Event()
        produced.record(torch.npu.current_stream(self.put_stream.device))
        future = self.put_executor.submit(self._background_put, policy, batches, produced)
        self.pending_put = (future, targets)

    def _background_put(self, policy, batches, produced):
        import torch

        torch.npu.set_device(self.put_stream.device)
        started = time.monotonic()
        with torch.npu.stream(self.put_stream):
            self.put_stream.wait_event(produced)
            results = self._execute_put(policy, batches, "decode")
        return results, time.monotonic() - started

    def poll_put(self, *, wait=False):
        if self.pending_put is None:
            return
        future, targets = self.pending_put
        if not wait and not future.done():
            return
        before = time.monotonic()
        stalled = not future.done()
        results, elapsed = future.result()  # Failure is fatal; never publish/recycle uncertain keys.
        self._finish_put(results, targets, "decode")
        self.pending_put = None
        now = time.monotonic()
        self.put_batches += 1
        self.put_seconds += elapsed
        self.put_waits += int(stalled)
        self.put_wait_seconds += now - before if stalled else 0
        if now - self.put_last_log >= 5:
            logger.info(
                "KVAF decode Put: batches=%s background_ms=%.3f waits=%s wait_ms=%.3f; "
                "ABC source retained until completion; no KV snapshot copy",
                self.put_batches,
                1000 * self.put_seconds / self.put_batches,
                self.put_waits,
                1000 * self.put_wait_seconds,
            )
            self.put_batches = self.put_waits = 0
            self.put_seconds = self.put_wait_seconds = 0.0
            self.put_last_log = now

    def close_put(self):
        try:
            self.poll_put(wait=True)
        finally:
            if self.put_executor is not None:
                self.put_executor.shutdown(wait=True)
                self.put_executor = None


def verify_put_block(runtime, source, key, byte_offset, layout) -> None:
    # Worker-only import. Verification deliberately copies one block to CPU;
    # normal writes do not copy KV through host memory or read it back.
    import torch

    tokens, token_bytes = layout.block_size, layout.planes[0].token_bytes
    handle = runtime.prepare_access_table([key])
    try:
        metadata = runtime.access_table_view(handle)
        if list(metadata["valid"]) != [1] or list(metadata["lengths"]) != [tokens * token_bytes]:
            raise RuntimeError("KVAF BatchPut verification: committed entry is missing or has the wrong length")
        table = runtime.access_table_view(handle, device="npu")
        indices = torch.zeros(tokens, dtype=torch.int32, device=source.device)
        destinations = torch.arange(tokens, dtype=torch.int32, device=source.device)
        offsets = destinations * token_bytes
        count = torch.tensor([tokens], dtype=torch.int32, device=source.device)
        output = torch.empty((tokens, token_bytes), dtype=torch.uint8, device=source.device)
        runtime.batch_get(table, indices, offsets, count, destinations, output, synchronize=True)
        expected = source.view(torch.uint8)[byte_offset : byte_offset + tokens * token_bytes]
        if not torch.equal(output.cpu().reshape(-1), expected.cpu()):
            raise RuntimeError("KVAF BatchPut verification: DDR bytes differ from the source HBM block")
    finally:
        runtime.release_access_table(handle)


def get_tokens(runtime, *args, **kwargs):
    """Execute the caller's descriptors on the current stream without extra waits."""
    return runtime.batch_get(*args, **kwargs)
