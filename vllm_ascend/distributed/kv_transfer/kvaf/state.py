# SPDX-License-Identifier: Apache-2.0
"""Request progress, reference-counted DDR slots, and graph-stable device maps."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from .layout import KVCacheLayout


@dataclass
class RequestCacheState:
    block_size: int
    block_ids: tuple[int, ...] = ()
    computed_tokens: int = 0
    write_end: int = 0

    def prepare(self, blocks, start, scheduled):
        pages = (start + scheduled + self.block_size - 1) // self.block_size
        if pages > len(blocks):
            raise ValueError(f"KVAF HBM block table too short: required={pages} available={len(blocks)}")
        self.block_ids = blocks if isinstance(blocks, tuple) and len(blocks) == pages else tuple(blocks[:pages])
        self.computed_tokens = start
        self.write_end = start + scheduled

    def clear(self):
        self.block_ids = ()
        self.computed_tokens = self.write_end = 0


@dataclass
class DeviceSlotMaps:
    token_to_slot: torch.Tensor
    slot_to_token: torch.Tensor


class DeviceCacheState:
    """Allocate maps before profiling; never mirror device-owned maps on CPU."""

    def __init__(self, layer_names, max_requests, max_tokens, device):
        import torch

        self.names = tuple(layer_names)
        self.indices = {name: index for index, name in enumerate(self.names)}
        self.max_requests, self.max_tokens = max_requests, max_tokens
        self.device = torch.device(device)
        self.on_npu = self.device.type == "npu"
        shape = (len(self.names), max_requests, max_tokens)
        self.storage = {
            name: torch.full(shape, -1, dtype=torch.int32, device=device) for name in ("token_to_slot", "slot_to_token")
        }
        self.views = {
            name: DeviceSlotMaps(**{key: tensor[index] for key, tensor in self.storage.items()})
            for index, name in enumerate(self.names)
        }

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.storage.values()) + self.policy.nbytes


class DDRSlotPool:
    """Reference-counted slots with soft layer regions and coalescing extents.

    A region is an allocation preference, not a capacity limit. Native KVAF
    keys are slot IDs; only returned access-table addresses are used by Get.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.references = {}
        self.extents = [(0, capacity)] if capacity else []
        self.regions = {}

    def partition(self, weights):
        total = sum(weights.values())
        start = cumulative = 0
        for name, weight in weights.items():
            cumulative += weight
            end = min(self.capacity, int(self.capacity * cumulative / total))
            self.regions[name] = (start, end)
            start = end

    def take(self, count, *, region=None, after=None):
        if count < 0 or count > sum(end - start for start, end in self.extents):
            raise RuntimeError("KVAF DDR slots exhausted; increase kvaf_ddr_size_gb or reduce concurrent KV length")
        if not count:
            return []
        low, high = self.regions.get(region, (0, self.capacity))
        selected = []
        # Extend decode at its previous segment's end whenever possible.
        if after is not None:
            for start, end in self.extents:
                if start <= after and after + count <= end and low <= after < after + count <= high:
                    selected = [(after, after + count)]
                    break
        # Prefer one whole segment within the layer region, then anywhere.
        for left, right in ((low, high), (0, self.capacity)):
            if selected:
                break
            for start, end in self.extents:
                begin, stop = max(start, left), min(end, right)
                if stop - begin >= count:
                    selected = [(begin, begin + count)]
                    break
        if not selected:
            # Fragmentation must not turn available capacity into admission
            # failure. Fill the preferred region first, then borrow outside it.
            remaining = count
            for left, right in ((low, high), (0, low), (high, self.capacity)):
                for start, end in self.extents:
                    begin, stop = max(start, left), min(end, right)
                    size = min(remaining, max(0, stop - begin))
                    if size:
                        selected.append((begin, begin + size))
                        remaining -= size
                    if not remaining:
                        break
                if not remaining:
                    break
        for begin, stop in sorted(selected):
            for i, (start, end) in enumerate(self.extents):
                if start <= begin and stop <= end:
                    self.extents[i : i + 1] = ([(start, begin)] if start < begin else []) + (
                        [(stop, end)] if stop < end else []
                    )
                    break
        keys = [key for start, end in selected for key in range(start, end)]
        self.references.update((key, 1) for key in keys)
        return keys

    def retain(self, keys):
        for key in keys:
            self.references[key] += 1

    def release(self, keys):
        retired = []
        for key in keys:
            self.references[key] -= 1
            if self.references[key] == 0:
                del self.references[key]
                retired.append(key)
        return retired

    def recycle(self, keys):
        """Called after native batch_free, or for never-published reservations."""
        runs = []
        for key in sorted(keys):
            if runs and runs[-1][1] == key:
                runs[-1] = (runs[-1][0], key + 1)
            else:
                runs.append((key, key + 1))
        merged = []
        for start, end in sorted([*self.extents, *runs]):
            if merged and merged[-1][1] == start:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        self.extents = merged


@dataclass
class RequestGroupState:
    progress: RequestCacheState
    addresses: list[int] = field(default_factory=list)
    keys: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    reserved: deque[int] = field(default_factory=deque)


@dataclass
class CacheGroup:
    layout: KVCacheLayout
    geometry: tuple[int, int]
    source: Any
