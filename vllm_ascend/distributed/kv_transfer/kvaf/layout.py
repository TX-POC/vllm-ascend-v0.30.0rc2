# SPDX-License-Identifier: Apache-2.0
"""Describe bound cache views without importing torch or changing allocations."""

from collections import Counter
from dataclasses import dataclass, replace
from math import prod
from typing import Any


@dataclass(frozen=True)
class KVCachePlane:
    """One tensor view; page_bytes is payload, page_stride_bytes includes gaps.

    direct_token_read means one token is contiguous within this plane. It does
    not mean separate planes (such as indexer K and scale) can be read together.
    """

    shape: tuple[int, ...]
    dtype: str
    stride: tuple[int, ...]
    device: str
    data_ptr: int
    page_bytes: int
    page_stride_bytes: int
    token_bytes: int | None
    direct_token_read: bool


@dataclass(frozen=True)
class KVCacheLayout:
    """One physical cache with all layer aliases; page_bytes comes from its spec."""

    layer_names: tuple[str, ...]
    group_ids: tuple[int, ...]
    roles: tuple[str, ...]
    cache_kind: str  # "main" or "indexer", independently of draft ownership.
    block_size: int
    page_bytes: int
    packed_c8: bool
    planes: tuple[KVCachePlane, ...]


def _describe_plane(tensor: Any, block_size: int) -> KVCachePlane:
    shape = tuple(int(size) for size in tensor.shape)
    stride = tuple(int(size) for size in tensor.stride())
    if len(shape) < 2 or any(size <= 0 for size in shape):
        raise ValueError(f"KVAF requires nonempty block-first cache views, got {shape}")
    item_bytes = int(tensor.element_size())
    page_bytes = prod(shape[1:]) * item_bytes
    # Current SFA caches use (blocks, tokens, heads, features). Other tensor
    # layouts remain described, but are not advertised as directly readable.
    direct = len(shape) >= 3 and shape[1] == block_size
    expected_stride = 1
    for size, actual_stride in reversed(tuple(zip(shape[2:], stride[2:]))):
        direct = direct and (size == 1 or actual_stride == expected_stride)
        expected_stride *= size
    token_bytes = page_bytes // block_size if direct else None
    direct = direct and stride[1] == expected_stride and stride[0] >= block_size * stride[1]
    return KVCachePlane(
        shape,
        str(tensor.dtype),
        stride,
        str(tensor.device),
        int(tensor.data_ptr()),
        page_bytes,
        stride[0] * item_bytes,
        token_bytes if direct else None,
        direct,
    )


def describe_cache_layouts(
    kv_caches: dict[str, Any],
    kv_cache_config: Any,
    draft_layer_names: tuple[str, ...] = (),
    *,
    layer_specs: dict[str, Any] | None = None,
) -> tuple[KVCacheLayout, ...]:
    """Describe actual bound tensors and deduplicate identical shared views.

    Pass the runner's resolved layer_specs to retain live indexer overrides.
    Uniform groups retain per-layer specs. Draft names must come from the
    runner's actual draft ownership; layer-name substrings are not sufficient.
    Distinct views of one backing allocation remain distinct planes/layouts.
    No tensors are retained and no device data is read.
    """
    specs = {}
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        group_spec = group.kv_cache_spec
        group_layer_specs = getattr(group_spec, "kv_cache_specs", {})
        for name in group.layer_names:
            spec = (layer_specs or {}).get(name, group_layer_specs.get(name, group_spec))
            specs[name] = (group_id, spec)
    drafts = frozenset(draft_layer_names)
    layouts: dict[tuple, KVCacheLayout] = {}
    for name, cache in kv_caches.items():
        if name not in specs:
            raise ValueError(f"No KV cache spec for layer {name}")
        group_id, spec = specs[name]
        block_size, page_bytes = int(spec.block_size), int(spec.page_size_bytes)
        if block_size <= 0 or page_bytes <= 0:
            raise ValueError(f"Invalid KV cache spec for layer {name}")
        tensors = cache if isinstance(cache, (tuple, list)) else (cache,)
        if not tensors:
            raise ValueError(f"No cache planes for layer {name}")
        planes = tuple(_describe_plane(tensor, block_size) for tensor in tensors)
        identity = tuple((plane.device, plane.data_ptr, plane.shape, plane.stride, plane.dtype) for plane in planes)
        is_indexer = hasattr(spec, "sfa_dcp_replicated_indexer_size")
        cache_kind = "indexer" if is_indexer else "main"
        role = "draft" if name in drafts else "indexer" if is_indexer else "target"
        packed_c8 = bool(getattr(spec, "cache_sparse_sfa_c8", False))
        previous = layouts.get(identity)
        if previous is None:
            layouts[identity] = KVCacheLayout(
                (name,), (group_id,), (role,), cache_kind, block_size, page_bytes, packed_c8, planes
            )
        else:
            if (previous.block_size, previous.page_bytes, previous.packed_c8, previous.cache_kind) != (
                block_size,
                page_bytes,
                packed_c8,
                cache_kind,
            ):
                raise ValueError(f"Shared KV cache has incompatible specs: {previous.layer_names}, {name}")
            layouts[identity] = replace(
                previous,
                layer_names=(*previous.layer_names, name),
                group_ids=tuple(sorted(set((*previous.group_ids, group_id)))),
                roles=tuple(sorted(set((*previous.roles, role)))),
            )
    return tuple(layouts.values())


TOPK = 2048
MAX_QUERIES = 4
FUSED_HISTORY_ALIGNMENT = 256  # Original LIFetch LI tile-pair requirement.
NATIVE_SLOT_CAPACITY = (1 << 17) + 20 * 128
NATIVE_BITMAP_WORDS = 4176
BLOCK_SIZE = 128

GROUP_WIDTH = 4


def index_groups(names, forward_context):
    runs = []
    for name in names:
        impl = forward_context[name].impl
        if not runs or not impl.skip_topk:
            runs.append([])
        runs[-1].append(name)
    groups = []
    for run in runs:
        for start in range(0, len(run), GROUP_WIDTH):
            chunk = run[start : start + GROUP_WIDTH]
            groups.extend([tuple(chunk)] if len(chunk) == GROUP_WIDTH else [(name,) for name in chunk])
    return tuple(groups)


def grouped_layouts(layouts, groups):
    targets = {name: layout for layout in layouts for name in layout.layer_names}
    result = []
    for names in groups:
        first = targets[names[0]]
        width = len(names)
        for name in names:
            other = targets[name]
            if (other.block_size, other.group_ids, other.planes[0].token_bytes) != (
                first.block_size,
                first.group_ids,
                first.planes[0].token_bytes,
            ):
                raise ValueError("Grouped KV layers require identical token geometry and scheduler block tables")
        plane = first.planes[0]
        packed = replace(
            plane,
            token_bytes=plane.token_bytes * width,
            page_bytes=plane.page_bytes * width,
            page_stride_bytes=plane.page_bytes * width,
        )
        result.append(replace(first, layer_names=names, page_bytes=first.page_bytes * width, planes=(packed,)))
    return tuple(result)


def runtime_weights(layouts, target_layouts, slot_alignment=4096):
    """Physical DDR demand for a warmed prefix and its active borrower.

    Target pages are shared. Auxiliary snapshots are copied, so admission
    reserves the new snapshot while the old prefix still owns its snapshot.
    Separate runtimes cannot borrow each other's unused slots.
    """
    weights = Counter()
    for layout in target_layouts:
        weights[layout.planes[0].token_bytes, layout.block_size] += layout.page_bytes / layout.block_size
    auxiliary_geometry = (target_layouts[0].planes[0].token_bytes, target_layouts[0].block_size)
    for layout in layouts:
        if "target" not in layout.roles or layout.cache_kind != "main":
            weights[auxiliary_geometry] += 2 * sum(p.page_bytes for p in layout.planes) / layout.block_size
    for geometry in weights:
        payload = geometry[0] * geometry[1]
        allocated = (payload + slot_alignment - 1) // slot_alignment * slot_alignment
        weights[geometry] *= allocated / payload
    return weights


HISTORY = MAX_QUERIES * TOPK
TAIL = 3 * BLOCK_SIZE
SLOTS = HISTORY + TAIL
TOKEN_BYTES = 656
