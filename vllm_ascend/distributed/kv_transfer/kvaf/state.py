# SPDX-License-Identifier: Apache-2.0
"""Request progress, reference-counted DDR slots, and graph-stable device maps."""

from __future__ import annotations

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


@dataclass
class DDRSlotPool:
    capacity: int
    next_key: int = 0
    free: list[int] = field(default_factory=list)
    references: dict[int, int] = field(default_factory=dict)

    def take(self, count: int) -> list[int]:
        if count > len(self.free) + self.capacity - self.next_key:
            raise RuntimeError("KVAF DDR slots exhausted; increase kvaf_ddr_size_gb or reduce concurrent KV length")
        keys = [self.free.pop() for _ in range(min(count, len(self.free)))]
        remaining = count - len(keys)
        keys.extend(range(self.next_key, self.next_key + remaining))
        self.next_key += remaining
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


@dataclass
class RequestGroupState:
    progress: RequestCacheState
    addresses: list[int] = field(default_factory=list)
    keys: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)


@dataclass
class CacheGroup:
    layout: KVCacheLayout
    geometry: tuple[int, int]
    source: Any
