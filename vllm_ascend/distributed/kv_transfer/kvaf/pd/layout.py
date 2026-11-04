# SPDX-License-Identifier: Apache-2.0
"""PA layout and byte descriptors for one registered C8 layer.

The SDK adds the registered UBA base. This module only subtracts PA bases
belonging to the same layer allocation; it never constructs an absolute UBA.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

BLOCK_SIZE = 128
MAX_SEGMENTS = 4096
MAX_READ_BYTES = (1 << 32) - 1
MAX_INT64 = (1 << 63) - 1
PLANE_BYTES = (("main", 656), ("indexer", 128), ("scale", 4))


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= MAX_INT64:
        raise ValueError(f"{name} must be an integer in [{minimum}, {MAX_INT64}]")


@dataclass(frozen=True)
class Plane:
    name: str
    pa_base: int
    blocks: int
    token_bytes: int

    def __post_init__(self):
        if (self.name, self.token_bytes) not in PLANE_BYTES:
            raise ValueError("PD requires main/indexer/scale token widths 656/128/4")
        _integer(self.pa_base, "plane PA base", 1)
        _integer(self.blocks, "block capacity", 1)

    @property
    def block_bytes(self) -> int:
        return BLOCK_SIZE * self.token_bytes


@dataclass(frozen=True)
class LayerLayout:
    name: str
    pa_base: int
    nbytes: int
    planes: tuple[Plane, ...]

    def __post_init__(self):
        object.__setattr__(self, "planes", tuple(self.planes))
        _integer(self.pa_base, "layer PA base", 1)
        _integer(self.nbytes, "layer bytes", 1)
        if not self.name or self.pa_base + self.nbytes > MAX_INT64:
            raise ValueError("invalid layer name or PA range")
        names = sorted(p.name for p in self.planes)
        if names not in (["main"], sorted(name for name, _ in PLANE_BYTES)):
            raise ValueError("a layer requires main, with both indexer and scale when owned")
        end = self.pa_base
        for plane in sorted(self.planes, key=lambda p: p.pa_base):
            if plane.pa_base < end or plane.pa_base + plane.blocks * plane.block_bytes > self.pa_base + self.nbytes:
                raise ValueError("planes overlap or fall outside the registered layer")
            end = plane.pa_base + plane.blocks * plane.block_bytes

    @classmethod
    def from_tensors(cls, name: str, layer_tensor: Any, tensors: Mapping[str, Any]) -> "LayerLayout":
        """Inspect actual views. Independent allocations cannot form one MR.

        layer_tensor is the *actual* contiguous Tensor passed to Register (or
        raw READ as output), not a synthetic span between unrelated pointers.
        No tensors are allocated, read, concatenated, or retained here.
        """
        if not layer_tensor.is_contiguous() or set(tensors) not in ({"main"}, {name for name, _ in PLANE_BYTES}):
            raise ValueError("require a contiguous layer Tensor and the owned planes")
        storage = layer_tensor.untyped_storage().data_ptr()
        planes = []
        for plane_name, width in PLANE_BYTES:
            if plane_name not in tensors:
                continue
            tensor = tensors[plane_name]
            if (
                tensor.device != layer_tensor.device
                or tensor.untyped_storage().data_ptr() != storage
                or not tensor.is_contiguous()
                or tensor.ndim < 2
                or tensor.shape[1] != BLOCK_SIZE
                or tensor.stride(0) * tensor.element_size() != BLOCK_SIZE * width
                or tensor[0].numel() * tensor.element_size() != BLOCK_SIZE * width
            ):
                raise ValueError(f"{plane_name} must be a packed block-first view of the registered layer storage")
            planes.append(Plane(plane_name, tensor.data_ptr(), tensor.shape[0], width))
        return cls(name, layer_tensor.data_ptr(), layer_tensor.numel() * layer_tensor.element_size(), tuple(planes))

    def plane(self, name: str) -> Plane:
        return next(p for p in self.planes if p.name == name)

    def offset(self, plane_name: str, table: Sequence[int], logical_block: int) -> int:
        _integer(logical_block, "logical block")
        plane = self.plane(plane_name)
        if logical_block >= len(table):
            raise ValueError(f"{plane_name} block table is too short")
        physical = table[logical_block]
        _integer(physical, "physical block")
        if physical >= plane.blocks:
            raise ValueError(f"{plane_name} physical block exceeds its plane capacity")
        return plane.pa_base - self.pa_base + physical * plane.block_bytes


@dataclass(frozen=True)
class Segments:
    sources: tuple[int, ...] = ()
    destinations: tuple[int, ...] = ()
    lengths: tuple[int, ...] = ()

    def __post_init__(self):
        for name in ("sources", "destinations", "lengths"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not len(self.sources) == len(self.destinations) == len(self.lengths):
            raise ValueError("Pull arrays must have equal lengths")
        for src, dst, length in zip(self.sources, self.destinations, self.lengths):
            _integer(src, "source offset")
            _integer(dst, "destination")
            _integer(length, "length", 1)
            if length > MAX_READ_BYTES or max(src, dst) + length > MAX_INT64:
                raise ValueError("Pull segment exceeds address/READ limits")
        end = 0
        for dst, length in sorted(zip(self.destinations, self.lengths)):
            if dst < end:
                raise ValueError("Pull destinations overlap")
            end = dst + length

    def validate_call(self, source_bytes: int, output_bytes: int | None = None) -> None:
        if not 1 <= len(self.sources) <= MAX_SEGMENTS:
            raise ValueError("one Ready consumes one batch of 1..4096 segments")
        if any(src + size > source_bytes for src, size in zip(self.sources, self.lengths)):
            raise ValueError("Pull exceeds registered source")
        if output_bytes is not None and any(
            dst + size > output_bytes for dst, size in zip(self.destinations, self.lengths)
        ):
            raise ValueError("Pull exceeds output")

    def batches(self) -> tuple["Segments", ...]:
        """Each returned batch requires its own Ready; never replay one Ready."""
        return tuple(
            Segments(*(values[i : i + MAX_SEGMENTS] for values in (self.sources, self.destinations, self.lengths)))
            for i in range(0, len(self.sources), MAX_SEGMENTS)
        )


def contributor_range(count: int, member: int, ratio: int, first_block: int = 0) -> range:
    """Native SFA _tp_block_range rule, without importing its hardware worker."""
    for value, name in ((count, "count"), (member, "member"), (first_block, "first block")):
        _integer(value, name)
    _integer(ratio, "TP ratio", 1)
    if member >= ratio:
        raise ValueError("contributor is outside the P/D TP group")
    logical_rank = (member - first_block % ratio) % ratio
    width, remainder = divmod(count, ratio)
    start = logical_rank * width + min(logical_rank, remainder)
    return range(first_block + start, first_block + start + width + int(logical_rank < remainder))


@dataclass(frozen=True)
class LayerPlan:
    full_blocks: tuple[int, ...]
    main_sources: tuple[int, ...]
    hbm: Segments

    def ddr_segments(self, entries: Sequence[Any], keys: Sequence[int]) -> Segments:
        """Use this Put's authoritative addresses; reject grouped DDR payloads."""
        page_bytes = BLOCK_SIZE * dict(PLANE_BYTES)["main"]
        if len(entries) != len(self.full_blocks) or len(keys) != len(entries) or len(set(keys)) != len(keys):
            raise ValueError("Put keys must match the complete main blocks")
        if any(e.key != key or e.length != page_bytes or e.address <= 0 for e, key in zip(entries, keys)):
            raise ValueError("Put entries must describe complete single-layer C8 blocks; grouped DDR needs adaptation")
        return Segments(self.main_sources, tuple(e.address for e in entries), (page_bytes,) * len(entries))


def plan_layer(
    source: LayerLayout,
    output: LayerLayout,
    source_tables: Mapping[str, Sequence[int]],
    output_tables: Mapping[str, Sequence[int]],
    tokens: int,
    *,
    first_block: int = 0,
    final: bool = True,
    member: int = 0,
    ratio: int = 1,
    offload_main: bool = True,
) -> LayerPlan:
    """Plan one contributor's newly complete blocks and optional final tail.

    first_block is the first *untransferred* logical block, not ceil(previous
    chunk tokens / 128). The caller advances it only after completion. Main KV
    is replicated (member 0); indexer/scale follow the native TP partition.
    GLM C8 replication and the real runner allocation remain integration gates.
    """
    _integer(tokens, "tokens")
    _integer(first_block, "first block")
    full, tail = divmod(tokens, BLOCK_SIZE)
    end = full + int(final and tail > 0)
    if source.name != output.name or first_block > end:
        raise ValueError("layer identity or chunk block interval mismatch")
    owned = contributor_range(end - first_block, member, ratio, first_block)
    if {p.name for p in source.planes} != {p.name for p in output.planes}:
        raise ValueError("P/D disagree about the indexer owner")
    blocks = tuple(range(first_block, full)) if member == 0 and offload_main else ()
    main = tuple(source.offset("main", source_tables["main"], q) for q in blocks)
    src, dst, sizes = [], [], []
    for plane_name, width in PLANE_BYTES:
        if plane_name not in {p.name for p in source.planes}:
            continue
        if plane_name == "main":
            if offload_main:
                selected = (full,) if member == 0 and final and tail and full >= first_block else ()
            else:
                selected = range(first_block, end) if member == 0 else ()
        else:
            selected = owned
        for q in selected:
            length = min(BLOCK_SIZE, tokens - q * BLOCK_SIZE) * width
            src.append(source.offset(plane_name, source_tables[plane_name], q))
            dst.append(output.offset(plane_name, output_tables[plane_name], q))
            sizes.append(length)
    return LayerPlan(blocks, main, Segments(tuple(src), tuple(dst), tuple(sizes)))
