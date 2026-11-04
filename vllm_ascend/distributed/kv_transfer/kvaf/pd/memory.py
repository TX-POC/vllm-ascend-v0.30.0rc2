# SPDX-License-Identifier: Apache-2.0
"""Real per-layer allocations; no address spans across independent tensors."""

from dataclasses import dataclass
from typing import Any

from .layout import BLOCK_SIZE, PLANE_BYTES, LayerLayout


@dataclass
class LayerMemory:
    tensor: Any
    planes: dict[str, Any]
    groups: dict[str, int]
    offload_main: bool

    def describe(self, name):
        return LayerLayout.from_tensors(name, self.tensor, self.planes)

    def tables(self, block_ids, *, tail_slot=None, tokens=0):
        result = {plane: tuple(block_ids[group]) for plane, group in self.groups.items()}
        if tail_slot is not None:
            # Scheduler's virtual main block IDs remain untouched. Only the
            # ordinary receive pool uses a compact Backend-owned physical slot.
            result["main"] = (tail_slot,) * (tokens // BLOCK_SIZE + 1)
        return result


def allocate_layer(main_blocks, indexer_blocks, *, device, groups, offload_main, allocator=None):
    import torch

    counts = {"main": main_blocks}
    if indexer_blocks:
        counts.update(indexer=indexer_blocks, scale=indexer_blocks)
    if set(counts) != set(groups) or any(type(count) is not int or count <= 0 for count in counts.values()):
        raise ValueError("invalid C8 layer capacity/group ownership")
    size = sum(counts[name] * BLOCK_SIZE * width for name, width in PLANE_BYTES if name in counts)
    tensor = torch.empty(size, dtype=torch.int8, device=device) if allocator is None else allocator(size)
    planes, offset = {}, 0
    for name, width in PLANE_BYTES:
        if name in counts:
            length = counts[name] * BLOCK_SIZE * width
            planes[name] = tensor[offset : offset + length].view(counts[name], BLOCK_SIZE, width)
            offset += length
    return LayerMemory(tensor, planes, dict(groups), offload_main)
