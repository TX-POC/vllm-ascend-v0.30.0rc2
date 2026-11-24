# SPDX-License-Identifier: Apache-2.0
"""Cross-request prefix reuse: scheduler catalog, worker snapshots, and vLLM protocol."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.logger import init_logger

from .state import RequestCacheState, RequestGroupState
from .transfer import get_tokens

logger = init_logger(__name__)


def _ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class Capacity:
    block_size: int
    capacities: dict[tuple[int, int], int]
    slots_per_block: dict[tuple[int, int], int]
    # (runtime geometry, physical plane block size, plane bytes per block).
    # These planes are packed together within each runtime's slots.
    aux_planes: tuple[tuple[tuple[int, int], int, int], ...] = ()

    def cost(self, num_tokens):
        blocks = _ceildiv(max(0, num_tokens), self.block_size)
        result = {geometry: blocks * count for geometry, count in self.slots_per_block.items()}
        packed_bytes = {}
        for geometry, block_size, page_bytes in self.aux_planes:
            packed_bytes[geometry] = packed_bytes.get(geometry, 0) + _ceildiv(num_tokens, block_size) * page_bytes
        for geometry, size in packed_bytes.items():
            result[geometry] = result.get(geometry, 0) + _ceildiv(size, geometry[0] * geometry[1])
        return result


@dataclass(frozen=True)
class SaveJob:
    entry_id: int
    request_id: str
    block_ids: tuple[tuple[int, ...], ...]
    num_tokens: int
    drop_block_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class LoadJob:
    entry_id: int
    request_id: str
    block_ids: tuple[tuple[int, ...], ...]
    num_tokens: int


@dataclass(frozen=True)
class EvictJob:
    entry_id: int
    drop_block_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class DropJob:
    request_id: str
    drop_block_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class CleanupJob:
    drop_block_ids: tuple[int, ...] = ()


@dataclass
class PrefixEntry:
    entry_id: int
    request_id: str
    hashes: tuple
    num_tokens: int
    cost: dict
    block_ids: tuple[tuple[int, ...], ...]
    # Snapshot of scheduler block identity, not just its recyclable numeric ID.
    block_hashes: dict[int, object] = field(default_factory=dict)
    state: str = "saving"
    leases: set[str] = field(default_factory=set)
    completed_ranks: set[int] = field(default_factory=set)
    touched: int = 0


class PrefixCatalog:
    """One DP scheduler's catalog. IDs increase and never alias older ACKs."""

    def __init__(self, capacity, world_size, replay_blocks=1):
        self.capacity = capacity
        self.world_size = world_size
        self.replay_blocks = replay_blocks
        self.entries = {}
        self.reservations = {}
        self.shared_targets = {}  # request -> (leased archive ID, already-funded Target slots)
        self._prefixes = {}
        self._next_id = 0
        self._clock = 0
        self.pending_evictions = set()

    def _touch(self, entry):
        self._clock += 1
        entry.touched = self._clock

    def _usage(self):
        costs = [*self.reservations.values(), *(entry.cost for entry in self.entries.values())]
        credits = [cost for _, cost in self.shared_targets.values()]
        return {
            geometry: sum(cost.get(geometry, 0) for cost in costs) - sum(cost.get(geometry, 0) for cost in credits)
            for geometry in self.capacity.capacities
        }

    def reserve(self, request_id, num_tokens, *, source=None, matched_tokens=0):
        cost = self.capacity.cost(num_tokens)
        shared = {}
        if source is not None:
            entry = self.entries[source]
            if entry.state != "ready" or not 0 <= matched_tokens <= min(entry.num_tokens, num_tokens):
                raise ValueError("KVAF reservation requires a ready prefix within the request horizon")
            shared = {
                g: matched_tokens // self.capacity.block_size * layers
                for g, layers in self.capacity.slots_per_block.items()
            }
        # Target keys are retained by load(), not copied. Auxiliary snapshots
        # are still copied, so their full future save remains in the budget.
        additional = {g: count - shared.get(g, 0) for g, count in cost.items()}
        if any(count > self.capacity.capacities.get(geometry, 0) for geometry, count in cost.items()):
            raise ValueError(
                f"KVAF DDR cannot hold request {request_id!r} through its configured maximum length: "
                f"required_slots={cost}, capacity={self.capacity.capacities}; reduce max_tokens or increase DDR"
            )
        usage = self._usage()
        previous = self.reservations.get(request_id, {})
        previous_source, previous_shared = self.shared_targets.get(request_id, (None, {}))
        for geometry in usage:
            usage[geometry] -= previous.get(geometry, 0) - previous_shared.get(geometry, 0)
        if all(usage[g] + additional.get(g, 0) <= limit for g, limit in self.capacity.capacities.items()):
            if previous_source is not None:
                self.entries[previous_source].leases.discard(request_id)
            self.reservations[request_id] = cost
            self.shared_targets.pop(request_id, None)
            if source is not None:
                self.lease(source, request_id)
                self.shared_targets[request_id] = (source, shared)
            return True
        # Free is a separate worker transaction. Its slots remain charged until
        # every TP rank confirms completion, so no new Put can race old storage.
        prospective = usage.copy()
        for entry in self.entries.values():
            if entry.state == "evicting":
                for geometry, count in entry.cost.items():
                    prospective[geometry] -= count
        for entry in sorted(self.entries.values(), key=lambda item: item.touched):
            if all(prospective[g] + additional.get(g, 0) <= limit for g, limit in self.capacity.capacities.items()):
                break
            if entry.state != "ready" or entry.leases or entry.entry_id == source:
                continue
            self.evict(entry.entry_id)
            for geometry, count in entry.cost.items():
                prospective[geometry] -= count
        return False

    def lookup(self, hashes, max_tokens):
        # Replay the final common block for MTP's volatile suffix. Partial
        # saved tails stay in the archive but are not advertised as computed.
        limit = max(0, max_tokens // self.capacity.block_size) + self.replay_blocks
        for index in range(min(len(hashes), limit) - 1, -1, -1):
            matched = max(0, index + 1 - self.replay_blocks) * self.capacity.block_size
            if not matched:
                break
            for entry_id in self._prefixes.get(hashes[index], ()):
                entry = self.entries[entry_id]
                if entry.state == "ready":
                    self._touch(entry)
                    return entry, matched
        return None, 0

    def has_pending_match(self, hashes, max_tokens, num_computed_tokens):
        """A replay of a just-finished request waits for its all-worker save."""
        limit = max(0, max_tokens // self.capacity.block_size) + self.replay_blocks
        first = num_computed_tokens // self.capacity.block_size + self.replay_blocks
        for entry in self.entries.values():
            if entry.state != "saving":
                continue
            usable = entry.num_tokens // self.capacity.block_size
            stop = min(len(hashes), len(entry.hashes), limit, usable)
            if any(hashes[index] == entry.hashes[index] for index in range(stop - 1, first - 1, -1)):
                return True
        return False

    def lease(self, entry_id, request_id):
        entry = self.entries[entry_id]
        if entry.state != "ready":
            raise RuntimeError("KVAF attempted to load an unpublished or evicting archive")
        entry.leases.add(request_id)
        self._touch(entry)

    def release(self, request_id):
        self.reservations.pop(request_id, None)
        self.shared_targets.pop(request_id, None)
        for entry in self.entries.values():
            entry.leases.discard(request_id)

    def save(self, request_id, hashes, num_tokens, block_ids, block_hashes, drop_block_ids=()):
        reservation = self.reservations.pop(request_id)
        # Once saved, charge the new archive in full. Shared pages may be
        # conservatively counted twice until the unleased older copy is evicted;
        # never let freeing an older archive underfund another active borrower.
        self.shared_targets.pop(request_id, None)
        cost = self.capacity.cost(num_tokens)
        if any(count > reservation.get(geometry, 0) for geometry, count in cost.items()):
            raise RuntimeError("KVAF archive exceeds its request's DDR reservation")
        entry_id = self._next_id
        self._next_id += 1
        # Keep the full lifetime reservation until workers finish saving AND
        # dropping speculative/old logical bindings. An in-flight save cannot
        # prematurely fund another admission.
        entry = PrefixEntry(entry_id, request_id, tuple(hashes), num_tokens, reservation, block_ids, block_hashes)
        self.entries[entry_id] = entry
        self._touch(entry)
        return SaveJob(entry_id, request_id, block_ids, num_tokens, drop_block_ids)

    def acknowledge_save(self, entry_id, ranks):
        entry = self.entries[entry_id]
        if entry.state != "saving":
            return
        entry.completed_ranks.update(ranks)
        if len(entry.completed_ranks) != self.world_size:
            return
        entry.state = "ready"
        entry.cost = self.capacity.cost(entry.num_tokens)
        entry.completed_ranks.clear()
        usable = entry.num_tokens // self.capacity.block_size
        for block_hash in entry.hashes[:usable]:
            self._prefixes.setdefault(block_hash, set()).add(entry_id)

    def evict(self, entry_id):
        entry = self.entries[entry_id]
        if entry.leases:
            raise RuntimeError("Cannot evict a leased KVAF archive")
        entry.state = "evicting"
        entry.completed_ranks.clear()
        for block_hash in entry.hashes:
            matches = self._prefixes.get(block_hash)
            if matches is not None:
                matches.discard(entry_id)
                if not matches:
                    del self._prefixes[block_hash]
        self.pending_evictions.add(entry_id)

    def retire_redundant(self):
        # A completed replay may retain the same Target keys in a newer,
        # longer archive. Drop the covered old copy before LRU pressure can
        # evict an unrelated warmed prefix to fund that duplicate charge.
        for entry in self.entries.values():
            usable = min(len(entry.hashes), entry.num_tokens // self.capacity.block_size)
            if entry.state != "ready" or entry.leases or not usable:
                continue
            candidates = (self.entries[i] for i in self._prefixes.get(entry.hashes[usable - 1], ()))
            newer = max(
                (e for e in candidates if e.state == "ready"),
                key=lambda e: (e.num_tokens // self.capacity.block_size, e.entry_id),
                default=entry,
            )
            if newer.entry_id != entry.entry_id:
                self.evict(entry.entry_id)

    def acknowledge_eviction(self, entry_id, ranks):
        entry = self.entries[entry_id]
        if entry.state != "evicting":
            raise RuntimeError("Unexpected KVAF eviction acknowledgement")
        entry.completed_ranks.update(ranks)
        if len(entry.completed_ranks) == self.world_size:
            del self.entries[entry_id]


@dataclass(frozen=True)
class AuxPlaneRange:
    layer_name: str
    plane_index: int
    offset: int
    pages: int


@dataclass(frozen=True)
class AuxArchive:
    keys: tuple[int, ...]
    num_tokens: int
    payload_bytes: int
    planes: tuple[AuxPlaneRange, ...]


@dataclass(frozen=True)
class _Plane:
    layer_name: str
    index: int
    group_ids: tuple[int, ...]
    block_size: int
    page_bytes: int
    cache: Any


class AuxiliarySnapshotStore:
    """Pack all auxiliary planes into shared slots without adding KVAF sessions.

    ``slots.take`` returns keys with one owned reference. The caller releases
    successful archives' keys. A native Put failure is fatal: its reservations
    are deliberately not reused because native completion may be uncertain.
    A final partial token page is copied whole; ``num_tokens`` limits reuse,
    and the caller remains responsible for MTP's matched-token/replay boundary.
    """

    def __init__(self, runtime, slots, layouts, kv_caches, *, token_bytes, tokens_per_block, scratch_bytes=8 << 20):
        import torch

        self.runtime, self.slots = runtime, slots
        self.token_bytes = token_bytes
        self.slot_payload_bytes = token_bytes * tokens_per_block
        self._planes = []
        for layout in layouts:
            if "target" in layout.roles and layout.cache_kind == "main":
                continue
            name = layout.layer_names[0]
            tensors = kv_caches[name]
            tensors = tensors if isinstance(tensors, (tuple, list)) else (tensors,)
            for index, (cache, plane) in enumerate(zip(tensors, layout.planes)):
                if not cache[0].is_contiguous():
                    raise ValueError(f"Auxiliary archive requires contiguous bytes within each page: {name}")
                # Byte views avoid dtype-dependent gather/scatter support for
                # FP8 keys/scales; page padding stays outside the payload.
                byte_pages = cache.view(torch.uint8).as_strided(
                    (cache.shape[0], plane.page_bytes), (plane.page_stride_bytes, 1)
                )
                self._planes.append(
                    _Plane(name, index, layout.group_ids, layout.block_size, plane.page_bytes, byte_pages)
                )
        self.aux_planes = tuple((p.block_size, p.page_bytes) for p in self._planes)
        self._by_name = {(p.layer_name, p.index): p for p in self._planes}
        if not self._planes:
            return
        capacity = scratch_bytes // self.slot_payload_bytes * self.slot_payload_bytes
        if capacity < max(p.page_bytes for p in self._planes) + 2 * token_bytes:
            raise ValueError("Auxiliary archive scratch must fit a full page and Get alignment padding")
        self._device = self._planes[0].cache.device
        self._scratch = torch.empty((capacity // token_bytes, token_bytes), dtype=torch.uint8, device=self._device)
        self._indices = torch.empty(self._scratch.shape[0], dtype=torch.int32, device=self._device)
        self._offsets = torch.empty_like(self._indices)
        self._output_indices = torch.arange(self._indices.numel(), dtype=torch.int32, device=self._device)
        self._count = torch.empty(1, dtype=torch.int32, device=self._device)

    def required_slots(self, num_tokens):
        size = sum(((num_tokens + p.block_size - 1) // p.block_size) * p.page_bytes for p in self._planes)
        return (size + self.slot_payload_bytes - 1) // self.slot_payload_bytes

    @staticmethod
    def _blocks(plane, block_ids, pages):
        ids = tuple(block_ids[plane.group_ids[0]][:pages])
        if len(ids) != pages or any(tuple(block_ids[g][:pages]) != ids for g in plane.group_ids[1:]):
            raise ValueError(f"Auxiliary archive block tables disagree or are too short: {plane.layer_name}")
        return ids

    def save(self, block_ids_by_group, num_tokens):
        import torch

        if num_tokens < 0:
            raise ValueError("Auxiliary archive token length cannot be negative")
        records, blocks, total = [], [], 0
        for plane in self._planes:
            pages = (num_tokens + plane.block_size - 1) // plane.block_size
            blocks.append(self._blocks(plane, block_ids_by_group, pages))
            records.append(AuxPlaneRange(plane.layer_name, plane.index, total, pages))
            total += pages * plane.page_bytes
        keys = tuple(self.slots.take(self.required_slots(num_tokens)))
        archive = AuxArchive(keys, num_tokens, total, tuple(records))
        if not keys:
            return archive
        flat = self._scratch.view(-1)
        filled = key_index = 0
        for plane, ids in zip(self._planes, blocks):
            batch_pages = max(1, flat.numel() // plane.page_bytes)
            for start in range(0, len(ids), batch_pages):
                selected = torch.tensor(ids[start : start + batch_pages], dtype=torch.int64, device=self._device)
                payload = plane.cache.index_select(0, selected).contiguous().view(torch.uint8).reshape(-1)
                offset = 0
                while offset < payload.numel():
                    size = min(flat.numel() - filled, payload.numel() - offset)
                    flat[filled : filled + size].copy_(payload[offset : offset + size])
                    filled += size
                    offset += size
                    if filled == flat.numel():
                        key_index += self._put(keys, key_index, filled)
                        filled = 0
                # Release this gather before allocating the next chunk; the
                # memory budget reserves one gathered chunk plus the scratch.
                del payload
        if filled:
            # Get's fixed row width may cover the end of the serialized payload.
            padded = (filled + self.token_bytes - 1) // self.token_bytes * self.token_bytes
            flat[filled:padded].zero_()
            self._put(keys, key_index, padded)
        return archive

    def _put(self, keys, start, length):
        count = (length + self.slot_payload_bytes - 1) // self.slot_payload_bytes
        offsets = [i * self.slot_payload_bytes for i in range(count)]
        self.runtime.batch_put(
            keys=keys[start : start + count],
            source=self._scratch.view(-1),
            input_offsets=offsets,
            lengths=[min(self.slot_payload_bytes, length - offset) for offset in offsets],
        )
        return count

    def load(self, archive, block_ids_by_group, num_tokens):
        import torch

        if not 0 <= num_tokens <= archive.num_tokens:
            raise ValueError("Auxiliary restore exceeds the saved token prefix")
        if not num_tokens or not archive.keys:
            return
        handle = self.runtime.prepare_access_table(archive.keys)
        try:
            table = self.runtime.access_table_view(handle)
            addresses = torch.tensor(
                [v if v < (1 << 63) else v - (1 << 64) for v in table["addresses"]],
                dtype=torch.int64,
                device=self._device,
            )
            valid = torch.tensor(list(table["valid"]), dtype=torch.uint8, device=self._device)
            for record in archive.planes:
                plane = self._by_name[record.layer_name, record.plane_index]
                pages = (num_tokens + plane.block_size - 1) // plane.block_size
                ids = self._blocks(plane, block_ids_by_group, pages)
                batch_pages = (self._scratch.numel() - 2 * self.token_bytes) // plane.page_bytes
                for start in range(0, pages, batch_pages):
                    batch = ids[start : start + batch_pages]
                    begin = record.offset + start * plane.page_bytes
                    size = len(batch) * plane.page_bytes
                    first = begin // self.token_bytes
                    end = (begin + size + self.token_bytes - 1) // self.token_bytes
                    rows = torch.arange(first, end, dtype=torch.int64, device=self._device)
                    self._indices[: len(rows)].copy_(rows // (self.slot_payload_bytes // self.token_bytes))
                    self._offsets[: len(rows)].copy_(rows * self.token_bytes % self.slot_payload_bytes)
                    self._count.fill_(len(rows))
                    get_tokens(
                        self.runtime,
                        addresses,
                        valid,
                        self._indices,
                        self._offsets,
                        self._count,
                        self._output_indices,
                        self._scratch,
                        synchronize=False,
                    )
                    shift = begin % self.token_bytes
                    payload = self._scratch.view(-1)[shift : shift + size]
                    restored = payload.view(len(batch), plane.page_bytes)
                    selected = torch.tensor(batch, dtype=torch.int64, device=self._device)
                    plane.cache.index_copy_(0, selected, restored)
        finally:
            # Protect the table's DDR entries and the final index_copy until all
            # reads finish. These operations run only at request boundaries.
            if self._device.type != "cpu":
                torch.npu.current_stream().synchronize()
            self.runtime.release_access_table(handle)


@dataclass
class SnapshotEntry:
    targets: tuple
    auxiliary: object
    num_tokens: int


class RequestSnapshotStore:
    def __init__(self, writer, auxiliary):
        self.writer, self.auxiliary = writer, auxiliary
        self.entries = {}
        geometry = writer.groups[0].geometry
        self.capacity = Capacity(
            geometry[1],
            {g: slots.capacity for g, slots in writer.slots.items()},
            dict(Counter(layer.geometry for layer in writer.groups)),
            tuple((geometry, size, payload) for size, payload in auxiliary.aux_planes),
        )

    def save(self, job):
        if job.entry_id in self.entries:
            return
        # Only complete computed blocks are advertised. MTP's final replay
        # block is additionally excluded by the scheduler catalog.
        tokens = job.num_tokens // self.capacity.block_size * self.capacity.block_size
        self.writer.policy.publish(job.request_id, tokens)
        states = self.writer.requests[job.request_id]
        targets = []
        for layer, state in zip(self.writer.groups, states):
            count = tokens // layer.geometry[1]
            keys = tuple(state.keys[:count])
            self.writer.slots[layer.geometry].retain(keys)
            targets.append((keys, tuple(state.addresses[:count])))
        aux = self.auxiliary.save(job.block_ids, tokens)
        self.entries[job.entry_id] = SnapshotEntry(tuple(targets), aux, tokens)
        self.writer.release_requests((job.request_id,))

    def load(self, job):
        entry = self.entries[job.entry_id]
        self.writer.release_requests((job.request_id,))
        states = []
        for layer, (keys, addresses) in zip(self.writer.groups, entry.targets):
            count = job.num_tokens // layer.geometry[1]
            selected = list(keys[:count])
            self.writer.slots[layer.geometry].retain(selected)
            states.append(RequestGroupState(RequestCacheState(layer.geometry[1]), list(addresses[:count]), selected))
        self.writer.requests[job.request_id] = states
        started = time.perf_counter()
        self.auxiliary.load(entry.auxiliary, job.block_ids, job.num_tokens)
        logger.info(
            "KVAF archive RESTORED: request=%s tokens=%s auxiliary_ms=%.3f; Target keys shared",
            job.request_id,
            job.num_tokens,
            (time.perf_counter() - started) * 1000,
        )

    def evict(self, job):
        entry = self.entries.pop(job.entry_id)
        for layer, (keys, _) in zip(self.writer.groups, entry.targets):
            self._free(layer.geometry, keys)
        self._free(self.writer.groups[0].geometry, entry.auxiliary.keys)

    def _free(self, geometry, keys):
        retired = self.writer.slots[geometry].release(keys)
        if retired:
            self.writer.runtimes[geometry].batch_free(retired)
            self.writer.slots[geometry].recycle(retired)

    def drop(self, job):
        request = getattr(job, "request_id", None)
        if request is not None:
            self.writer.release_requests((request,))


@dataclass
class ArchiveMetadata(KVConnectorMetadata):
    saves: tuple[SaveJob, ...] = ()
    loads: tuple[LoadJob, ...] = ()
    evictions: tuple[EvictJob, ...] = ()
    drops: tuple[DropJob, ...] = ()
    cleanups: tuple[CleanupJob, ...] = ()


@dataclass
class ArchiveHandshake(KVConnectorHandshakeMetadata):
    capacity: Capacity


@dataclass
class ArchiveWorkerMetadata(KVConnectorWorkerMetadata):
    saved: dict[int, set[int]] = field(default_factory=dict)
    evicted: dict[int, set[int]] = field(default_factory=dict)

    def aggregate(self, other):
        for name in ("saved", "evicted"):
            target = getattr(self, name)
            for entry_id, ranks in getattr(other, name).items():
                target.setdefault(entry_id, set()).update(ranks)
        return self


class KVAFConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.world_size = vllm_config.parallel_config.tensor_parallel_size
        self.max_model_len = vllm_config.model_config.max_model_len
        self.spec_tokens = getattr(vllm_config.speculative_config, "num_speculative_tokens", 0)
        self.prefix_caching = vllm_config.cache_config.enable_prefix_caching
        self.catalog = None
        self.archive = None
        self.rank = 0
        self._saves = []
        self._loads = []
        self._drops = []
        self._lookups = {}
        self._capacity_waiting = set()
        self._request_blocks = {}
        self._receiving = set()
        self._cancelled_receives = {}
        self._pending_sends = set()
        self._processed_metadata = None
        self._loaded_metadata = None
        self._worker_meta = ArchiveWorkerMetadata()
        self._saved_requests = set()
        self._loaded_requests = set()
        self._reported_finished = set()
        self._known_requests = set()

    def bind_kv_cache_manager(self, manager):
        super().bind_kv_cache_manager(manager)
        # Target pages are transient. Prefix hashes remain enabled on requests,
        # but only the DDR catalog may advertise a reusable computed prefix.
        manager.enable_caching = False

    @property
    def requires_kv_delivery(self):
        return False

    def bind_archive(self, archive, rank=0):
        self.archive = archive
        archive.managed = True
        self.rank = rank

    def get_handshake_metadata(self):
        if self.archive is None:
            raise RuntimeError("KVAF archive must be bound before the worker handshake")
        return ArchiveHandshake(self.archive.capacity)

    def set_xfer_handshake_metadata(self, metadata):
        if len(metadata) != self.world_size:
            raise ValueError("KVAF capacity handshake must include every TP worker")
        capacities = [value.capacity for value in metadata.values()]
        first = capacities[0]
        if any(
            (value.block_size, value.slots_per_block, value.aux_planes)
            != (first.block_size, first.slots_per_block, first.aux_planes)
            or value.capacities.keys() != first.capacities.keys()
            for value in capacities
        ):
            raise ValueError("KVAF workers disagree on archive cache geometry")
        capacity = Capacity(
            first.block_size,
            {geometry: min(value.capacities[geometry] for value in capacities) for geometry in first.capacities},
            first.slots_per_block,
            first.aux_planes,
        )
        self.catalog = PrefixCatalog(capacity, self.world_size, replay_blocks=1 if self.spec_tokens else 0)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if self.catalog is None:
            raise RuntimeError("KVAF capacity handshake was not completed before scheduling")
        horizon = (
            min(request.num_prompt_tokens + request.max_tokens, self.max_model_len)
            + self.spec_tokens
            + 2 * self.catalog.capacity.block_size
        )
        entry, matched = None, 0
        if self.prefix_caching and not getattr(request, "skip_reading_prefix_cache", False):
            hashes = self._request_hashes(request)
            previous = self._lookups.get(request.request_id)
            if previous is not None:
                entry, matched = self.catalog.entries[previous[0]], previous[1]
            else:
                entry, matched = self.catalog.lookup(hashes, request.num_tokens - 1)
            if self.catalog.has_pending_match(hashes, request.num_tokens - 1, max(num_computed_tokens, matched)):
                return None, False
            if matched <= num_computed_tokens:
                entry, matched = None, 0
        first_admission = request.request_id not in self.catalog.reservations
        if not self.catalog.reserve(
            request.request_id, horizon, source=entry.entry_id if entry else None, matched_tokens=matched
        ):
            if request.request_id not in self._capacity_waiting:
                logger.info(
                    "KVAF DDR admission WAIT: request=%s matched_tokens=%s usage_slots=%s capacity_slots=%s",
                    request.request_id,
                    matched,
                    self.catalog._usage(),
                    self.catalog.capacity.capacities,
                )
                self._capacity_waiting.add(request.request_id)
            return None, False
        self._capacity_waiting.discard(request.request_id)
        if first_admission:
            shared = self.catalog.shared_targets.get(request.request_id, (None, {}))[1]
            required = {g: n - shared.get(g, 0) for g, n in self.catalog.reservations[request.request_id].items()}
            logger.info(
                "KVAF DDR admission: request=%s matched_tokens=%s additional_slots=%s shared_target_slots=%s",
                request.request_id,
                matched,
                required,
                shared,
            )
        if entry is None:
            self._lookups.pop(request.request_id, None)
            return 0, False
        self._lookups[request.request_id] = (entry.entry_id, matched)
        return matched - num_computed_tokens, True

    def _request_hashes(self, request):
        # vLLM can hash more finely than the physical Target block. A hash at
        # each Target block end still identifies its entire preceding chain.
        block_size = self.catalog.capacity.block_size
        hash_size = self._kv_cache_manager.block_pool.hash_block_size
        if block_size % hash_size:
            raise ValueError("KVAF Target block size must be divisible by vLLM's prefix hash block size")
        stride = block_size // hash_size
        return tuple(request.block_hashes[stride - 1 :: stride])

    def _block_hashes(self, block_ids):
        pool = self._kv_cache_manager.block_pool
        return {block_id: pool.blocks[block_id].block_hash for group in block_ids for block_id in group if block_id}

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        block_ids = tuple(tuple(group) for group in blocks.get_block_ids())
        self._request_blocks[request.request_id] = block_ids
        # Local HBM prefix hits can still depend on a catalog's Target DDR.
        # They require the same lease even though no external load was needed.
        identities = self._block_hashes(block_ids)
        for entry in self.catalog.entries.values():
            if entry.state != "evicting" and any(
                block_hash is not None and block_id in identities and identities[block_id] == block_hash
                for block_id, block_hash in entry.block_hashes.items()
            ):
                entry.leases.add(request.request_id)
                self.catalog._touch(entry)
        if num_external_tokens:
            entry_id, matched = self._lookups.pop(request.request_id)
            self.catalog.lease(entry_id, request.request_id)
            self._loads.append(LoadJob(entry_id, request.request_id, block_ids, matched))
            self._receiving.add(request.request_id)
            logger.info("KVAF archive HIT: entry=%s request=%s tokens=%s", entry_id, request.request_id, matched)

    def request_finished_all_groups(self, request, block_ids):
        request_id = request.request_id
        self._capacity_waiting.discard(request_id)
        if self.catalog is None or request_id not in self.catalog.reservations:
            return False, None
        # The standard finish hook clips to computed blocks. Include allocated
        # lookahead blocks too, so rejected MTP tails cannot retain orphan DDR.
        block_ids = tuple(tuple(group) for group in self._kv_cache_manager.get_block_ids(request_id))
        identities = self._block_hashes(block_ids)
        for entry in self.catalog.entries.values():
            if request_id in entry.leases:
                # Restored logical IDs may differ from the source request's.
                # Remember their final hash before releasing the active lease.
                entry.block_hashes.update(identities)
        computed = max(0, request.num_computed_tokens - getattr(request, "num_in_flight_tokens", 0))
        valid_tokens = min(computed, max(request.num_prompt_tokens, request.num_tokens - 1))
        pool = self._kv_cache_manager.block_pool
        safe = tuple(block_id for block_id in identities if pool.blocks[block_id].ref_cnt == 1)
        self._kv_cache_manager.evict_blocks(set(safe))
        self._lookups.pop(request_id, None)
        self._request_blocks.pop(request_id, None)
        self._pending_sends.add(request_id)
        if request_id in self._receiving:
            # Abort during receive: intercept its receive ACK, then drop the
            # restored binding before the one final send ACK frees HBM. Letting
            # both receive and save ACKs reach vLLM would free this request twice.
            self._cancelled_receives[request_id] = DropJob(request_id, safe)
            return True, None
        if valid_tokens <= 0 or not self.prefix_caching:
            self._drops.append(DropJob(request_id, safe))
            return True, None
        job = self.catalog.save(request_id, self._request_hashes(request), valid_tokens, block_ids, identities, safe)
        # A shared native prefix can already be in use by another request
        # before this first archive exists. Protect those owners as well.
        entry = self.catalog.entries[job.entry_id]
        shared = set(identities) - set(safe)
        for owner in self._request_blocks:
            # Running requests can have grown beyond their admission snapshot.
            owner_blocks = self._kv_cache_manager.get_block_ids(owner)
            if any(shared.intersection(group) for group in owner_blocks):
                entry.leases.add(owner)
        self._saves.append(job)
        return True, None

    def request_finished(self, request, block_ids):
        return self.request_finished_all_groups(request, (block_ids,))

    def _eviction_jobs(self):
        jobs = []
        pool = self._kv_cache_manager.block_pool
        for entry_id in sorted(self.catalog.pending_evictions):
            entry = self.catalog.entries[entry_id]
            safe = tuple(
                block_id
                for block_id, block_hash in entry.block_hashes.items()
                if pool.blocks[block_id].ref_cnt == 0 and pool.blocks[block_id].block_hash == block_hash
            )
            # Remove local prefix hashes before freeing the DDR on which these
            # apparent HBM hits depend. The worker additionally checks slot ID.
            self._kv_cache_manager.evict_blocks(set(safe))
            jobs.append(EvictJob(entry_id, safe))
        self.catalog.pending_evictions.clear()
        return tuple(jobs)

    def build_connector_meta(self, scheduler_output):
        cleanups = ()
        if getattr(scheduler_output, "preempted_req_ids", None):
            pool = self._kv_cache_manager.block_pool
            safe = tuple(block.block_id for block in pool.blocks if block.ref_cnt == 0 and not block.is_null)
            self._kv_cache_manager.evict_blocks(set(safe))
            cleanups = (CleanupJob(safe),)
        metadata = ArchiveMetadata(
            tuple(self._saves),
            tuple(self._loads),
            self._eviction_jobs(),
            tuple(self._drops),
            cleanups,
        )
        self._saves.clear()
        self._loads.clear()
        self._drops.clear()
        return metadata

    def update_connector_output(self, connector_output):
        metadata = connector_output.kv_connector_worker_meta
        if metadata is not None:
            for entry_id, ranks in metadata.saved.items():
                previous = self.catalog.entries[entry_id].state
                self.catalog.acknowledge_save(entry_id, ranks)
                entry = self.catalog.entries[entry_id]
                if previous != "ready" and entry.state == "ready":
                    logger.info(
                        "KVAF archive READY: entry=%s request=%s tokens=%s TP=%s",
                        entry_id,
                        entry.request_id,
                        entry.num_tokens,
                        self.world_size,
                    )
            for entry_id, ranks in metadata.evicted.items():
                self.catalog.acknowledge_eviction(entry_id, ranks)
                if entry_id not in self.catalog.entries:
                    logger.info("KVAF archive EVICTED: entry=%s TP=%s", entry_id, self.world_size)
        # Standard KVOutputAggregator only reports this after ALL TP workers.
        for request_id in connector_output.finished_sending or ():
            self.catalog.release(request_id)
            self._pending_sends.discard(request_id)
        if (metadata is not None and metadata.saved) or connector_output.finished_sending:
            self.catalog.retire_redundant()
        received = set(connector_output.finished_recving or ())
        self._receiving.difference_update(received)
        cancelled = received & self._cancelled_receives.keys()
        # The source archive funds the shared Target reservation. Keep its
        # lease until this borrower finishes (not merely until restore ACK).
        for request_id in cancelled:
            self._drops.append(self._cancelled_receives.pop(request_id))
        if cancelled:
            connector_output.finished_recving = received - cancelled

    def has_pending_push_work(self):
        return bool(
            self._saves
            or self._loads
            or self._drops
            or self._receiving
            or self._cancelled_receives
            or self._pending_sends
            or (self.catalog is not None and any(entry.state != "ready" for entry in self.catalog.entries.values()))
        )

    def has_pending_block_frees(self):
        return bool(self._saves or self._drops) or self.has_pending_push_work()

    def reset_cache(self):
        if self.catalog is None:
            return False
        if self.catalog.reservations or any(
            entry.leases or entry.state == "saving" for entry in self.catalog.entries.values()
        ):
            return False
        for entry in self.catalog.entries.values():
            if entry.state == "ready":
                self.catalog.evict(entry.entry_id)
        return True

    def handle_preemptions(self, kv_connector_metadata):
        # Runner calls this before deleting old request state / preparing new
        # logical mappings. Loads MUST wait until _update_states has zeroed new
        # blocks; async receive requests do not run forward in their load step.
        self.process_jobs(kv_connector_metadata)

    def process_jobs(self, metadata):
        if metadata is self._processed_metadata:
            return
        if self.archive is None:
            raise RuntimeError("KVAF worker archive is not bound")
        for job in metadata.evictions:
            self.archive.evict(job)
            self._worker_meta.evicted[job.entry_id] = {self.rank}
        for job in metadata.saves:
            self._known_requests.add(job.request_id)
            self.archive.save(job)
            self._worker_meta.saved[job.entry_id] = {self.rank}
            self._saved_requests.add(job.request_id)
        for job in metadata.drops:
            self._known_requests.add(job.request_id)
            self.archive.drop(job)
            self._saved_requests.add(job.request_id)
        for job in metadata.cleanups:
            self.archive.drop(job)
        self._processed_metadata = metadata

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        self.process_jobs(metadata)
        if metadata is self._loaded_metadata:
            return
        for job in metadata.loads:
            self._known_requests.add(job.request_id)
            self.archive.load(job)
            self._loaded_requests.add(job.request_id)
        self._loaded_metadata = metadata

    def get_finished(self, finished_req_ids):
        self._reported_finished.update(finished_req_ids & self._known_requests)
        saved = self._saved_requests & self._reported_finished
        self._saved_requests.difference_update(saved)
        self._reported_finished.difference_update(saved)
        self._known_requests.difference_update(saved)
        loaded, self._loaded_requests = self._loaded_requests, set()
        return saved, loaded

    def build_connector_worker_meta(self):
        metadata, self._worker_meta = self._worker_meta, ArchiveWorkerMetadata()
        return metadata if metadata.saved or metadata.evicted else None

    def register_kv_caches(self, kv_caches):
        # The runner binds the finalized Target pools and auxiliary views later.
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        # Archive tasks finish before forward. No per-layer save or decode wait.
        pass
