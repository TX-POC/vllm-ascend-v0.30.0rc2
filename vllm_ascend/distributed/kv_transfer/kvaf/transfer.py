# SPDX-License-Identifier: Apache-2.0
"""Put/Get execution and address publication; cache policy owns transfer selection."""

import logging
from collections.abc import Iterable
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

    def release_requests(self, req_ids: Iterable[str]) -> None:
        """Free native DDR entries before making their slot IDs reusable."""
        for req_id in req_ids:
            if self.feedback is not None:
                self.feedback.release(req_id)
            states = self.requests.get(req_id)
            if states is None:
                continue
            pages = byte_count = freed_pages = 0
            for layer, state in zip(self.groups, states):
                if not state.keys:
                    state.progress.clear()
                    continue
                retired = self.slots[layer.geometry].release(state.keys)
                freed_pages += len(retired)
                if retired:
                    self.runtimes[layer.geometry].batch_free(retired)
                    self.slots[layer.geometry].free.extend(retired)
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

    def _publish_addresses(self, layer, keys, jobs):
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
        offset = 0
        for state, count in jobs:
            state.addresses.extend(addresses[offset : offset + count])
            offset += count

    def put_requests(self, requests: dict[str, Any], scheduled_tokens: dict[str, int]) -> None:
        """Publish stable staging/ABC pages through the serving cache policy."""
        self.policy.after_forward(requests, scheduled_tokens)

    def publish(self, policy, req_id, tokens):
        phase = "decode" if policy.plans.get(req_id) is None or policy.plans[req_id].decode else "prefill"
        written = 0
        for i, (layer, state) in enumerate(zip(self.groups, self.requests[req_id])):
            count = tokens // BLOCK_SIZE - len(state.keys)
            if count <= 0:
                continue
            keys = self.slots[layer.geometry].take(count)
            offsets = [
                policy._source_offset(req_id, p) * policy.group_sizes[i]
                for p in range(len(state.keys), len(state.keys) + count)
            ]
            self.runtimes[layer.geometry].batch_put(
                keys=keys,
                source=policy.bytes_outputs[i].view(-1),
                input_offsets=offsets,
                lengths=[BLOCK_SIZE * TOKEN_BYTES * policy.group_sizes[i]] * count,
            )
            state.keys.extend(keys)
            self._publish_addresses(layer, keys, [(state, count)])
            written += count * policy.group_sizes[i]
            if self.verify is not None and phase not in self._verified:
                self.verify(
                    self.runtimes[layer.geometry],
                    policy.bytes_outputs[i].view(-1),
                    keys[0],
                    offsets[0],
                    layer.layout,
                )
                self._verified.add(phase)
                logger.info(
                    "KVAF BatchPut verified: phase=%s layer=%s; DDR matches HBM", phase, layer.layout.layer_names[0]
                )
        self.put_pages += written
        self.put_bytes += written * BLOCK_SIZE * TOKEN_BYTES
        self.phase_pages[phase] += written
        if written and phase not in self._logged.setdefault(req_id, set()):
            logger.info(
                "KVAF first BatchPut: request=%s phase=%s pages=%s; DDR addresses published", req_id, phase, written
            )
            self._logged[req_id].add(phase)


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
