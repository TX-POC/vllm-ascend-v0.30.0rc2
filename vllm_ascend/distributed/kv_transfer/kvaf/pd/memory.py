# SPDX-License-Identifier: Apache-2.0
"""Real layer allocations or views in explicitly enabled contiguous P arenas."""

from dataclasses import dataclass
from typing import Any

from .layout import BLOCK_SIZE, PLANE_BYTES, LayerLayout


@dataclass
class LayerMemory:
    tensor: Any
    planes: dict[str, Any]
    groups: dict[str, int]
    offload_main: bool
    arena: Any = None

    def describe(self, name):
        if self.arena is not None and (
            not self.arena.is_contiguous()
            or self.arena.untyped_storage().data_ptr() != self.tensor.untyped_storage().data_ptr()
            or self.tensor.data_ptr() < self.arena.data_ptr()
            or self.tensor.data_ptr() + self.tensor.numel() * self.tensor.element_size()
            > self.arena.data_ptr() + self.arena.numel() * self.arena.element_size()
        ):
            raise ValueError("layer must be an actual view contained in its registered arena")
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


def allocate_layers(plans, *, device, allocator, arena_layers=1, alignment=2 * 1024 * 1024):
    """Allocate once per group; never concatenate live layer tensors.

    plans maps each layer to (main_blocks, indexer_blocks, groups, offload_main).
    Target and MTP are split because MTP can rewrite its KV repeatedly.
    """
    groups = []
    for name, plan in plans.items():
        if not groups or len(groups[-1]) == arena_layers or plans[groups[-1][0]][3] != plan[3]:
            groups.append([])
        groups[-1].append(name)
    result = {}
    for names in groups:
        offsets, size = {}, 0
        for name in names:
            main, indexer, _, _ = plans[name]
            size = ((size + alignment - 1) // alignment) * alignment
            length = BLOCK_SIZE * (main * 656 + indexer * (128 + 4))
            offsets[name] = (size, length)
            size += length
        arena = allocator(size)
        for name in names:
            main, indexer, plane_groups, offload = plans[name]
            offset, length = offsets[name]
            memory = allocate_layer(
                main,
                indexer,
                device=device,
                groups=plane_groups,
                offload_main=offload,
                allocator=lambda requested, view=arena[offset : offset + length]: view,
            )
            if arena_layers > 1:
                memory.arena = arena
            memory.describe(name)
            result[name] = memory
    return result
