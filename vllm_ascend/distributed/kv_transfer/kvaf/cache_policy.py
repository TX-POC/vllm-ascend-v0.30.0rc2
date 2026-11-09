# SPDX-License-Identifier: Apache-2.0
"""Prefill/decode residency, ABC mapping, and asynchronous LRU feedback."""

import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np
import torch

from .adapter import NativeCacheAdapter
from .layout import BLOCK_SIZE, HISTORY, MAX_QUERIES, NATIVE_BITMAP_WORDS, SLOTS, TAIL, TOKEN_BYTES, TOPK
from .metrics import HotMissStats
from .transfer import get_tokens


class DecodeHotCache(NativeCacheAdapter):
    slot_capacity = SLOTS
    history_capacity = HISTORY

    def __init__(
        self,
        pool,
        native=None,
        staging_tokens=0,
        group_sizes=None,
        hot_tokens=HISTORY,
        tail_tokens=TAIL,
    ):
        self.history_capacity = hot_tokens
        self.tail_tokens = tail_tokens
        self.tail_start = hot_tokens
        self.slot_capacity = self.tail_start + self.tail_tokens
        if self.slot_capacity > pool.max_tokens:
            raise ValueError("KVAF hot slots exceed the token/slot map capacity")
        super().__init__(pool, native=native)
        layers, batch = len(pool.names), pool.max_requests
        self.group_sizes = tuple(group_sizes or [1] * layers)
        self.hot = [
            torch.empty(
                ((batch * self.slot_capacity + staging_tokens) * width // BLOCK_SIZE, BLOCK_SIZE, 1, TOKEN_BYTES),
                dtype=torch.int8,
                device=pool.device,
            )
            for width in self.group_sizes
        ]
        # Capture warmup has query lengths but no request plans or valid maps.
        # Reserve row zero and the first physical block for dummy/padded queries.
        # A base of one also maps LI's -1 sentinel inside the guard row; it must
        # not point at a real owner's mapping when a padded graph is replayed.
        pool.storage["token_to_slot"][:, 0].zero_()
        # Real owner 1 + LI(-1) still addresses this sentinel.
        pool.storage["token_to_slot"][:, 0, -1].fill_(-1)
        for cache in self.hot:
            cache[0].zero_()
        self.mapped = {}
        self.addresses = torch.zeros((batch, layers, self.qcap), dtype=torch.int64, device=pool.device)
        self.valid = torch.zeros_like(self.addresses, dtype=torch.uint8)
        self.write_slots = torch.full((self.max_queries,), -1, dtype=torch.int64, device=pool.device)
        self.query_bases = torch.ones((self.max_queries, 1, 1), dtype=torch.int64, device=pool.device)
        self.attention_lengths = torch.full((batch,), self.slot_capacity, dtype=torch.int32, device=pool.device)
        self.spare_slots = torch.zeros((layers, batch, TOPK), dtype=torch.int32, device=pool.device)
        self.spare_bitmap = torch.zeros((layers, batch, NATIVE_BITMAP_WORDS), dtype=torch.int32, device=pool.device)
        self.resident_tables = torch.zeros(
            (batch, self.slot_capacity // BLOCK_SIZE), dtype=torch.int32, device=pool.device
        )
        self.owners = {}
        self.free = list(range(batch - 1, 0, -1))  # Row zero guards LI's -1 entries.
        self.snapshots = {}
        self.published_pages = {}
        self.page_clear_rows = set()
        self.writer = None
        self.put_targets = {}
        self.bytes_outputs = []
        self.attention_caches = []
        self.abc_descriptors = torch.empty((layers, batch, 4), dtype=torch.int32, device=pool.device)
        self.uploaded_table_owners = None
        self.transfer_host = [{}, {}]
        self.current_ids = ()
        self.prepared = False

    @property
    def nbytes(self):
        return super().nbytes + sum(t.numel() * t.element_size() for t in (*self.hot, self.abc_descriptors))

    def bind(self, manager):
        super().bind(manager)
        self.writer = manager.writer
        if self.group_count != len(self.pool.names):
            raise ValueError("Hot Get requires one Target layout per allocated layer")
        for i, layer in enumerate(self.writer.groups):
            if (
                layer.geometry != (TOKEN_BYTES * self.group_sizes[i], BLOCK_SIZE)
                or not layer.layout.packed_c8
                or layer.layout.planes[0].page_stride_bytes != BLOCK_SIZE * TOKEN_BYTES * self.group_sizes[i]
            ):
                raise ValueError("Original LIDU/Get requires block-128 packed C8 with 656 bytes/token")
        # Prolog/SFA require the original dtype (e.g. FP8), not the raw int8 storage.
        self.attention_caches = [self.hot[i].view(layer.source.dtype) for i, layer in enumerate(self.writer.groups)]
        # BatchGet addresses output rows by token; each row must be token_bytes.
        self.bytes_outputs = [
            self.hot[i].view(torch.uint8).view(-1, TOKEN_BYTES * self.group_sizes[i]) for i in range(self.group_count)
        ]
        zero = torch.zeros(1, dtype=torch.int32, device=self.pool.device)
        for i, layer in enumerate(self.writer.groups):
            get_tokens(
                self.writer.runtimes[layer.geometry],
                self.addresses.view(-1),
                self.valid.view(-1),
                self.indices,
                self.offsets,
                zero,
                self.output_indices,
                self.bytes_outputs[i],
                synchronize=True,
            )
        logging.getLogger("vllm").info(
            "KVAF Get enabled: independent Target hot slots=%s+%s per request",
            self.history_capacity,
            self.tail_tokens,
        )

    def release(self, req_id):
        self.put_targets.pop(req_id, None)
        owner = self.owners.pop(req_id, None)
        if owner is not None:
            self.free.append(owner)
        for key in tuple(self.snapshots):
            if key[0] == req_id:
                del self.snapshots[key]
        for key in tuple(self.published_pages):
            if key[0] == req_id:
                del self.published_pages[key]

    def prepare(self, manager, *, decode, padded_requests):
        frame = manager.groups[0]
        ids = frame.request_ids
        self.enabled, self.batch, self.current_ids = decode, padded_requests, ids
        self.prepared = True
        ring = self.turn % 2
        event = self.done[ring]
        if event is not None and not event.query():
            event.synchronize()
        host = self.transfer_host[ring]

        def staging(key, target):
            source = host.get(key)
            if source is None or source.shape != target.shape or source.dtype != target.dtype:
                source = host[key] = torch.empty(target.shape, dtype=target.dtype, pin_memory=self.pool.on_npu)
            return source

        def upload(key, target, values):
            source = staging(key, target)
            source.numpy()[...] = values
            target.copy_(source, non_blocking=self.pool.on_npu)

        batch = self.pool.max_requests
        entries = np.full(batch, batch, np.int32)
        budgets = np.zeros(batch, np.int32)
        lengths = np.zeros(batch, np.int32)
        logical = np.ones(batch, np.int32)
        ends = np.zeros(batch, np.int32)
        writes = np.full(self.max_queries, -1, np.int64)
        bases = np.ones((self.max_queries, 1, 1), np.int64)
        if decode:
            candidates_host = staging("victims_batch", self.spare_slots)
            bitmap_host = staging("bitmap_batch", self.spare_bitmap)
            descriptors_host = staging("abc_batch", self.abc_descriptors)
            candidate_values = candidates_host.numpy()
            candidate_values.fill(0)
            reserved = getattr(self, "fused_reserved_hot", None)
            if reserved is not None:
                from kvip.adapters.vllm.inputs import committed_candidates

                reserved_host = staging("reserved_hot_batch", reserved)
                reserved_values = reserved_host.numpy()
                reserved_values.fill(-1)
                ordinals = np.arange(TOPK, dtype=np.int32)
            bitmap_values = bitmap_host.numpy()
            bitmap_values.fill(0)
            descriptors = descriptors_host.numpy()
            descriptors.fill(0)
            tables = []
            ends.fill(frame.query_offsets[-1])
            ends[: len(ids)] = frame.query_offsets[1:]
            for row_index, req_id in enumerate(ids):
                state = frame.rows[row_index].state
                progress = state.progress
                if req_id not in self.owners:
                    self.owners[req_id] = self.free.pop()
                    self.addresses[self.owners[req_id]].zero_()
                    self.valid[self.owners[req_id]].zero_()
                pool_row = self.owners[req_id]
                begin, end = frame.query_offsets[row_index : row_index + 2]
                if not 1 <= end - begin <= MAX_QUERIES:
                    raise ValueError("Hot Get supports 1..4 decode queries per request")
                tokens = np.arange(progress.computed_tokens, progress.write_end)
                writes[begin:end] = pool_row * self.slot_capacity + self.tail_start + tokens % self.tail_tokens
                bases[begin:end] = pool_row * self.pool.max_tokens
                table = np.arange(
                    pool_row * (self.slot_capacity // BLOCK_SIZE),
                    (pool_row + 1) * (self.slot_capacity // BLOCK_SIZE),
                    dtype=np.int32,
                )
                tables.append(table)
                # Native block table has a larger fixed stride; only hot pages
                # are addressable by the resident budget.
                entries[row_index] = pool_row if progress.computed_tokens >= TOPK else batch
                budgets[row_index] = self.slot_capacity if entries[row_index] != batch else 0
                lengths[row_index] = progress.write_end if budgets[row_index] else 0
                logical[row_index] = progress.write_end if budgets[row_index] else 1
                for i, inputs in enumerate(manager.groups):
                    item = inputs.rows[row_index]
                    source_state = item.state
                    maps = inputs.device
                    key = (req_id, i)
                    old = self.snapshots.get(key)
                    committed = len(source_state.addresses)
                    tail_begin = committed * BLOCK_SIZE
                    if progress.write_end - tail_begin > self.tail_tokens:
                        raise RuntimeError(
                            "ABC tail exceeds configured capacity; "
                            f"request={req_id} layer={i} first_hot_round={old is None} "
                            f"computed={progress.computed_tokens} write_end={progress.write_end} "
                            f"scheduled={end - begin} committed_pages={committed} "
                            f"tail_begin={tail_begin} pending_tokens={progress.write_end - tail_begin} "
                            f"previous_snapshot={old}; Put must complete before tail reuse"
                        )
                    if old is None:
                        maps.token_to_slot[pool_row].fill_(-1)
                        maps.slot_to_token[pool_row].fill_(-1)
                        # Short contexts cannot enter original fixed-2048 LIDU.
                        # Their committed prefix is copied once; SFA uses hot.
                        dense = min(tail_begin, progress.computed_tokens) if progress.computed_tokens < TOPK else 0
                        if dense:
                            self._seed(i, pool_row, source_state, 0, dense, False)
                            upload(("dense_t", i, pool_row), maps.token_to_slot[pool_row, :dense], np.arange(dense))
                            upload(("dense_s", i, pool_row), maps.slot_to_token[pool_row, :dense], np.arange(dense))
                        self._seed(i, pool_row, source_state, tail_begin, progress.computed_tokens, True)
                    elif tail_begin > old[0]:
                        if progress.computed_tokens < TOPK:
                            # Short contexts bypass LIDU: keep committed pages
                            # densely mapped when they leave the ABC ring.
                            self._promote_abc(i, pool_row, old[0], tail_begin)
                            dense_tokens = np.arange(old[0], tail_begin)
                            upload(
                                ("dense_t", i, pool_row),
                                maps.token_to_slot[pool_row, old[0] : tail_begin],
                                dense_tokens,
                            )
                            upload(
                                ("dense_s", i, pool_row),
                                maps.slot_to_token[pool_row, old[0] : tail_begin],
                                dense_tokens,
                            )
                        else:
                            self.page_clear_rows.add(key)
                    descriptors[i, row_index] = (self.map_groups[i], pool_row, tail_begin, progress.write_end)
                    candidates, bitmap = getattr(item.lru, "hot_snapshot", (None, None))
                    if candidates is None:
                        candidates = np.arange(TOPK, dtype=np.int32)
                        bitmap = np.zeros(NATIVE_BITMAP_WORDS, np.int32)
                        bitmap[: TOPK // 32] = -1
                    if reserved is not None:
                        candidates = committed_candidates(candidates, tail_begin, self.history_capacity)
                    candidate_values[i, row_index] = candidates
                    if reserved is not None:
                        reserved_values[i, row_index, candidates] = ordinals
                    bitmap_values[i, row_index, : len(bitmap)] = bitmap
                    self.snapshots[key] = (tail_begin, committed)
            # Whole [group, batch, ...] transfers, not one H2D per row.
            self.spare_slots.copy_(candidates_host, non_blocking=self.pool.on_npu)
            if reserved is not None:
                self.fused_reserved_hot.copy_(reserved_host, non_blocking=self.pool.on_npu)
            self.spare_bitmap.copy_(bitmap_host, non_blocking=self.pool.on_npu)
            self.abc_descriptors.copy_(descriptors_host, non_blocking=self.pool.on_npu)
            from kvip.cache.abc import update_abc

            update_abc(
                self.pool.storage["token_to_slot"],
                self.pool.storage["slot_to_token"],
                self.abc_descriptors,
                len(ids),
                self.tail_start,
                self.tail_tokens,
            )
            table_owners = tuple(self.owners[req] for req in ids)
            if self.uploaded_table_owners != table_owners:
                resident = np.zeros(self.resident_tables.shape, dtype=np.int32)
                resident[: len(ids)] = tables
                blocks = np.zeros(self.inputs["blocks"].shape, dtype=np.int32)
                blocks[:, : len(ids), : resident.shape[1]] = resident[None, : len(ids)]
                upload("tables_batch", self.resident_tables, resident)
                upload("blocks_batch", self.inputs["blocks"], blocks)
                self.uploaded_table_owners = table_owners
        for name, values in (
            ("entries", entries),
            ("query_ends", ends),
            ("budgets", budgets),
            ("lengths", lengths),
            ("logical", logical),
            ("miss_counts", -1),
        ):
            upload(name, self.inputs[name], values)
        upload("writes", self.write_slots, writes)
        upload("bases", self.query_bases, bases)
        # Original extra-victim scan starts in history. At most 8192 selected
        # tokens exist, so history has enough victims before reaching ABC slots.
        self.cursors.zero_()
        if self.pool.on_npu:
            if self.done[ring] is None:
                self.done[ring] = torch.npu.Event()
            self.done[ring].record(torch.npu.current_stream(self.pool.device))
        self.turn += 1

    def run(self, layer, topk, metadata, local_layer=0):
        if metadata.num_prefills:
            return topk
        width = self.group_sizes[layer]
        if local_layer:
            # Preserve the SFA invalid-index sentinel after interleaving layers.
            return (self.mapped[layer] * width + local_layer).clamp_min(-1)
        selector = self.indexer_selectors.get(layer)
        if selector is not None:
            # Selection already fetched these physical slots. Do not launch
            # native LIDU/Get again or look up the pre-update logical mapping.
            mapped = selector.slots(topk.shape[0]).view_as(topk)
            self.mapped[layer] = mapped
            self.destinations[layer, : topk.shape[0]].copy_(mapped.view(-1, 1, TOPK))
            return mapped if width == 1 else (mapped * width).clamp_min(-1)
        super().run(layer, topk, metadata)
        batch = metadata.cum_query_lens.shape[0]
        runtime = self.writer.runtimes[self.writer.groups[layer].geometry]
        if width != 1:
            # The unchanged LIDU emits offsets in 656-byte rows.
            self.offsets[: batch * TOPK * MAX_QUERIES].mul_(width)
        get_tokens(
            runtime,
            self.addresses.view(-1),
            self.valid.view(-1),
            self.indices[: batch * TOPK * MAX_QUERIES],
            self.offsets[: batch * TOPK * MAX_QUERIES],
            self.inputs["miss_counts"][layer, batch : batch + 1],
            self.output_indices[: batch * TOPK * MAX_QUERIES],
            self.bytes_outputs[layer],
            synchronize=False,
        )
        # Keep original LI order. No external sorting/compaction.
        shape = topk.shape
        queries = shape[0]
        indices = topk.view(queries, 1, TOPK) + self.query_bases[:queries]
        maps = self.pool.views[self.pool.names[self.map_groups[layer]]].token_to_slot
        mapped = torch.gather(maps.view(-1), 0, indices.reshape(-1)).view(shape)
        self.mapped[layer] = mapped
        return mapped if width == 1 else (mapped * width).clamp_min(-1)


logger = logging.getLogger("vllm")


class KVCachePolicy(DecodeHotCache):
    def __init__(
        self,
        pool,
        max_tokens,
        max_model_len,
        native=None,
        group_sizes=None,
        hot_tokens=HISTORY,
        fused_history=0,
        tail_tokens=TAIL,
        put_interval_steps=1,
    ):
        staging = ((max_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + pool.max_requests) * BLOCK_SIZE
        super().__init__(
            pool,
            native,
            staging_tokens=staging,
            group_sizes=group_sizes,
            hot_tokens=hot_tokens,
            tail_tokens=tail_tokens,
        )
        self.put_interval_steps = put_interval_steps
        self.put_step = 0
        self.stage_start = pool.max_requests * self.slot_capacity
        self.decode_writes = self.write_slots
        self.all_writes = torch.full((max(max_tokens, self.max_queries),), -1, dtype=torch.int64, device=pool.device)
        self.write_slots = self.all_writes  # The same address is captured and updated at runtime.
        self.write_hosts = [torch.empty_like(self.all_writes, device="cpu", pin_memory=pool.on_npu) for _ in range(2)]
        length = (max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
        self.context = torch.empty(
            (length // BLOCK_SIZE, BLOCK_SIZE, 1, TOKEN_BYTES), dtype=torch.int8, device=pool.device
        )
        tokens = torch.arange(length, dtype=torch.int32, device=pool.device)
        self.context_indices = tokens // BLOCK_SIZE
        self.context_offsets = tokens % BLOCK_SIZE * TOKEN_BYTES
        self.context_destinations = tokens
        self.context_count = torch.zeros(1, dtype=torch.int32, device=pool.device)
        self.context_table = torch.arange(length // BLOCK_SIZE, dtype=torch.int32, device=pool.device).view(1, -1)
        self.group_context = None
        self.group_writes = {}
        self.group_write_hosts = {}
        self.group_tables = {}
        self.group_lengths = {}
        self.write_factors = {}
        self.local_offsets = {}
        self.table_ids = None
        for width in set(self.group_sizes):
            if width == 1:
                continue
            self.group_writes[width] = torch.full(
                (width, self.all_writes.numel()), -1, dtype=torch.int64, device=pool.device
            )
            self.group_write_hosts[width] = [
                torch.empty_like(self.group_writes[width], device="cpu", pin_memory=pool.on_npu) for _ in range(2)
            ]
            self.group_tables[width] = torch.zeros(
                (pool.max_requests, self.slot_capacity * width // BLOCK_SIZE), dtype=torch.int32, device=pool.device
            )
            self.group_lengths[width] = torch.full_like(self.attention_lengths, self.slot_capacity * width)
            self.write_factors[width] = torch.full((width, 1), width, dtype=torch.int64, device=pool.device)
            self.local_offsets[width] = torch.arange(width, dtype=torch.int64, device=pool.device)[:, None]
        if max(self.group_sizes) > 1:
            width = max(self.group_sizes)
            # KVAF's scheduler admits one prompt computation per step. Keep
            # its four-layer history until the group's last layer consumes it;
            # subsequent steps reuse this address on the ordered compute stream.
            self.group_context = torch.empty((1, length, width, TOKEN_BYTES), dtype=torch.int8, device=pool.device)
        self.context_offsets_by_width = {w: self.context_offsets * w for w in set(self.group_sizes)}
        self.plans = {}
        self.decode_frames = ()
        self.only_decode = True
        self.indexer_selectors = {}
        self.fused_query_inputs = ()
        self.fused_index_storage = ()
        self.fused_workspace = None
        self.fused_setup = None
        self.fused_outputs = None
        self.fused_fetch_rows = None
        self.fused_errors = None
        self.fused_reserved_hot = None
        if fused_history:
            from kvip.adapters.vllm.index_import import VllmIndexCacheStorage

            self.fused_index_storage = tuple(
                VllmIndexCacheStorage(torch, pool.max_requests, fused_history, hot_tokens, pool.device)
                for _ in self.group_sizes
            )
            # Per-group storage: all groups are prepared before forward.
            # Use the same compact decode-row order as spare_slots.
            self.fused_reserved_hot = torch.full(
                (len(self.group_sizes), pool.max_requests, hot_tokens),
                -1,
                dtype=torch.int32,
                device=pool.device,
            )
            # Stable TND outputs survive subsequent shared-layer consumers and
            # are budgeted before HBM profiling, not allocated during capture.
            self.fused_outputs = torch.zeros(
                (len(self.group_sizes), 2, (pool.max_requests - 1) * MAX_QUERIES, 1, TOPK),
                dtype=torch.int32,
                device=pool.device,
            )
            # Original v9 row ABI, retained per group for one asynchronous upload.
            self.fused_fetch_rows = torch.zeros(
                (len(self.group_sizes), pool.max_requests - 1, 256),
                dtype=torch.int32,
                device=pool.device,
            )
            self.fused_errors = torch.zeros(
                (len(self.group_sizes), pool.max_requests - 1),
                dtype=torch.int32,
                device=pool.device,
            )

        from kvip.cache.page_updates import PageUpdates

        self.page_updates = PageUpdates(
            torch,
            self.addresses,
            self.valid,
            self.pool.storage["token_to_slot"],
            fused=tuple(storage.addresses for storage in self.fused_index_storage),
            widths=self.group_sizes,
            token_bytes=TOKEN_BYTES,
            block_size=BLOCK_SIZE,
        )

    def _publish_page_updates(self, ring):
        rows = []
        counts = []
        for plan in self.plans.values():
            for group, state in enumerate(self.writer.requests[plan.request]):
                key = (plan.request, group)
                previous = self.published_pages.get(key, 0)
                count = len(state.addresses)
                base = (plan.owner * len(self.group_sizes) + group) * self.qcap
                map_base = (self.map_groups[group] * self.pool.max_requests + plan.owner) * self.pool.max_tokens
                for page in range(previous, count):
                    clear = map_base + page * BLOCK_SIZE if key in self.page_clear_rows else -1
                    rows.append((base + page, state.addresses[page], clear))
                counts.append((key, count, plan.owner, group))
        # Existing done[ring] protects this pinned buffer; compute-stream order
        # publishes addresses and clears retired ABC mappings before forward.
        self.page_updates.submit(rows, ring)
        for key, count, owner, group in counts:
            self.published_pages[key] = count
            if self.fused_index_storage:
                self.fused_index_storage[group].published[owner] = count

    @property
    def nbytes(self):
        # context/group_context are counted by NativeCacheAdapter; dictionaries are not.
        tensors = [
            t
            for d in (
                self.group_writes,
                self.group_tables,
                self.group_lengths,
                self.context_offsets_by_width,
                self.write_factors,
                self.local_offsets,
            )
            for t in d.values()
        ]
        return (
            super().nbytes
            + sum(t.numel() * t.element_size() for t in tensors)
            + self.page_updates.nbytes
            + sum(storage.nbytes for storage in self.fused_index_storage)
            + (0 if self.fused_workspace is None else self.fused_workspace.nbytes)
        )

    def initialize_fused_workspace(self, config, device_id, soc, index_heads=64):
        """Allocate shared LI work memory before the runner profiles HBM."""
        if config.transport == "ubmem":
            from kvip.adapters.vllm.binding import prepare_workspace

            prepare_workspace(torch, self, config, device_id, soc, index_heads=index_heads)
            return
        from kvip.operators.lifetch import geometry
        from kvip.operators.lifetch.workspace import LIFetchWorkspace
        from kvip.runtime.build import kvaf_build_paths
        from kvip.runtime.device import query_device_resources

        if not self.fused_index_storage or self.fused_workspace is not None:
            raise RuntimeError("Fused Index storage must be initialized exactly once")
        cann, headers = kvaf_build_paths(config.build_dir)
        resources = query_device_resources(cann, device_id)
        aics = resources["effective_aic"]
        batch = self.pool.max_requests - 1  # Owner zero is the padding guard.
        shape = geometry.li_geometry(batch, "auto", config.intermediate_fetches, aics)
        if shape["logical_aic_tasks"] > aics:
            raise ValueError("LIFetch maximum request batch exceeds concurrently resident LI tasks")
        final_workers = geometry.final_aivs(batch, "auto", aics)
        workers = (batch * (final_workers + 1) + 1) // 2
        self.fused_setup = SimpleNamespace(
            cann=cann,
            headers=headers,
            soc=soc,
            aics=aics,
            batch=batch,
            windows=config.intermediate_fetches,
            workers=max(config.npu_core_count, workers),
        )
        self.fused_workspace = LIFetchWorkspace(
            torch,
            batch=batch,
            history=self.fused_index_storage[0].history,
            aics=aics,
            windows=config.intermediate_fetches,
            device=self.pool.device,
            index_heads=index_heads,
        )
        self.fused_workspace.allocate_rotation(
            torch,
            batch=batch,
            history=self.fused_index_storage[0].history,
            resident=self.history_capacity,
            aics=aics,
            windows=config.intermediate_fetches,
        )

    def prepare_indexer_metadata(self, layer, local_layer, metadata):
        # Indexer metadata can be reused across layers/steps. Always overwrite
        # the selector, including native/prefill steps, rather than leaving a
        # stale callback from a previous group attached to the shared object.
        metadata.topk_selector = self.indexer_selectors.get(layer) if local_layer == 0 else None

    def write_slots_for(self, layer, local_layer):
        width = self.group_sizes[layer]
        return self.all_writes if width == 1 else self.group_writes[width][local_layer]

    def _copy_slots(self, source, destination, count):
        for i, width in enumerate(self.group_sizes):
            flat = self.hot[i].view(-1, width * TOKEN_BYTES)
            flat[destination : destination + count].copy_(flat[source : source + count])

    def bind(self, manager):
        super().bind(manager)
        self.writer.policy = self
        if 1 in self.group_sizes:
            runtime = self.writer.runtimes[TOKEN_BYTES, BLOCK_SIZE]
            get_tokens(
                runtime,
                self.addresses.view(-1),
                self.valid.view(-1),
                self.context_indices,
                self.context_offsets,
                self.context_count,
                self.context_destinations,
                self.context.view(torch.uint8).view(-1, TOKEN_BYTES),
                synchronize=True,
            )
        if self.group_context is not None:
            width = self.group_context.shape[2]
            runtime = self.writer.runtimes[TOKEN_BYTES * width, BLOCK_SIZE]
            for context in self.group_context:
                get_tokens(
                    runtime,
                    self.addresses.view(-1),
                    self.valid.view(-1),
                    self.context_indices,
                    self.context_offsets_by_width[width],
                    self.context_count,
                    self.context_destinations,
                    context.view(torch.uint8).view(-1, width * TOKEN_BYTES),
                    synchronize=True,
                )
        logger.info(
            "KVAF grouped Get: groups=%s widths=%s; one LIDU/Get per group; "
            "max_prefill_requests=1; prefill history workspace=%.2f MiB",
            len(self.group_sizes),
            self.group_sizes,
            (self.context.numel() + (0 if self.group_context is None else self.group_context.numel())) / (1 << 20),
        )
        if self.pool.on_npu:
            self.writer.start_decode_put(self.pool.device)
        logger.info(
            "KVAF decode Put schedule: interval_steps=%s ABC_tokens=%s", self.put_interval_steps, self.tail_tokens
        )
        logger.info(
            "KVAF Target HBM: original full-length pages removed; transient prefill tokens=%s; "
            "shared history tokens=%s; hot slots=%s per request/layer",
            self.hot[0].shape[0] * BLOCK_SIZE // self.group_sizes[0] - self.stage_start,
            self.context.shape[0] * BLOCK_SIZE,
            self.slot_capacity,
        )

    def prepare(self, manager, *, decode, padded_requests):
        # Phase is request-local. Never stall/reject a normal mixed batch.
        self.writer.poll_put()
        self.page_clear_rows.clear()
        frame = manager.groups[0]
        self.plans = {}
        selected = []
        stage = self.stage_start
        for index, req_id in enumerate(frame.request_ids):
            state = frame.rows[index].state
            start, end = frame.query_offsets[index : index + 2]
            progress = state.progress
            is_decode = progress.computed_tokens >= self.requests[req_id].num_prompt_tokens
            if req_id not in self.owners:
                self.owners[req_id] = self.free.pop()
                owner = self.owners[req_id]
                self.addresses[owner].zero_()
                self.valid[owner].zero_()
            owner = self.owners[req_id]
            plan = SimpleNamespace(
                request=req_id,
                owner=owner,
                start=progress.computed_tokens,
                end=progress.write_end,
                begin=int(start),
                stop=int(end),
                decode=is_decode,
                stage=stage,
            )
            self.plans[req_id] = plan
            if is_decode:
                # The high-water end also covers asynchronous MTP correction:
                # positions may move backwards, but never beyond this scheduled end.
                committed = min(len(states.addresses) for states in self.writer.requests[req_id]) * BLOCK_SIZE
                if progress.write_end - committed > self.tail_tokens:
                    self.writer.poll_put(wait=True)
                    committed = min(len(s.addresses) for s in self.writer.requests[req_id]) * BLOCK_SIZE
                if (
                    progress.write_end - committed > self.tail_tokens
                    and progress.computed_tokens - committed <= self.tail_tokens
                ):
                    logger.warning(
                        "KVAF ABC backpressure: request=%s committed=%s computed=%s; "
                        "publishing stable tail before the next forward",
                        req_id,
                        committed,
                        progress.computed_tokens,
                    )
                    # Use only lengths confirmed after prior MTP corrections;
                    # prepare-time computed_tokens may still be optimistic.
                    targets = {req: tokens for req, tokens in self.put_targets.items() if req in frame.request_ids}
                    self.writer.publish_batch(self, targets, "decode")
                selected.append(index)
            else:
                stage += (
                    (progress.computed_tokens % BLOCK_SIZE + end - start + BLOCK_SIZE - 1) // BLOCK_SIZE
                ) * BLOCK_SIZE
                # Carry only the unfinished prefill page, never full history.
                prefix = progress.computed_tokens % BLOCK_SIZE
                if prefix:
                    source = (
                        owner * self.slot_capacity
                        + self.tail_start
                        + (progress.computed_tokens - prefix) % self.tail_tokens
                    )
                    self._copy_slots(source, plan.stage, prefix)
        ids = tuple(frame.request_ids[i] for i in selected)
        offsets = np.zeros(len(ids) + 1, np.int32)
        np.cumsum([frame.query_offsets[i + 1] - frame.query_offsets[i] for i in selected], out=offsets[1:])
        self.decode_frames = tuple(
            replace(f, request_ids=ids, rows=tuple(f.rows[i] for i in selected), query_offsets=offsets)
            for f in manager.groups
        )
        self.only_decode = len(ids) == len(frame.request_ids)
        if not self.only_decode:
            for plan in self.plans.values():
                plan.query_end = torch.tensor([plan.stop - plan.begin], dtype=torch.int32, device=self.pool.device)
                plan.key_end = torch.tensor([plan.end], dtype=torch.int32, device=self.pool.device)
        self.write_slots = self.decode_writes
        # The inherited path is unchanged for a pure decode batch.
        super().prepare(
            SimpleNamespace(groups=self.decode_frames),
            decode=bool(ids),
            padded_requests=padded_requests if self.only_decode else len(ids),
        )
        ring = (self.turn - 1) % 2
        host = self.write_hosts[ring]
        host.fill_(-1)
        values = host.numpy()
        for plan in self.plans.values():
            if plan.decode:
                values[plan.begin : plan.stop] = (
                    plan.owner * self.slot_capacity
                    + self.tail_start
                    + np.arange(plan.start, plan.end) % self.tail_tokens
                )
            else:
                values[plan.begin : plan.stop] = (
                    plan.stage + plan.start % BLOCK_SIZE + np.arange(plan.stop - plan.begin)
                )
        self.all_writes.copy_(host, non_blocking=self.pool.on_npu)
        self.write_slots = self.all_writes
        table_ids = tuple((req, self.owners[req]) for req in ids)
        for width, target in self.group_writes.items():
            source = self.group_write_hosts[width][ring]
            array = source.numpy()
            array.fill(-1)
            active = values >= 0
            array[:, active] = values[active][None, :] * width + np.arange(width)[:, None]
            target.copy_(source, non_blocking=self.pool.on_npu)
            tables = [
                np.arange(
                    self.owners[req] * self.slot_capacity * width // BLOCK_SIZE,
                    (self.owners[req] + 1) * self.slot_capacity * width // BLOCK_SIZE,
                    dtype=np.int32,
                )
                for req in ids
            ]
            if tables and self.table_ids != table_ids:
                self.group_tables[width][: len(ids)].copy_(torch.tensor(np.stack(tables), device=self.pool.device))
        self.table_ids = table_ids
        shared_inputs = {}
        for layer, selector in self.indexer_selectors.items():
            identity = id(selector.inputs)
            selector.prepare(self.plans.values(), prepare_inputs=identity not in shared_inputs)
            shared_inputs[identity] = selector.inputs
        self._publish_page_updates(ring)
        self.fused_query_inputs = tuple(shared_inputs.values())
        if self.pool.on_npu:
            self.done[ring].record(torch.npu.current_stream(self.pool.device))

    def align_decode_writes(self, positions):
        # Async MTP corrects positions on device before CPU bookkeeping catches
        # up. Use those positions for the ring writes without waiting on CPU.
        for inputs in self.fused_query_inputs:
            inputs.bind_positions(positions)
        for selector in self.indexer_selectors.values():
            selector.positions = positions
        for plan in self.plans.values():
            if plan.decode:
                slots = self.all_writes[plan.begin : plan.stop]
                torch.remainder(positions[plan.begin : plan.stop], self.tail_tokens, out=slots)
                slots.add_(plan.owner * self.slot_capacity + self.tail_start)
                for width, writes in self.group_writes.items():
                    torch.mul(slots[None, :], self.write_factors[width], out=writes[:, plan.begin : plan.stop])
                    writes[:, plan.begin : plan.stop].add_(self.local_offsets[width])

    def _promote_abc(self, layer, owner, start, end):
        """Copy committed ABC bytes before this forward can reuse their ring slots."""
        if not 0 <= end - start <= self.tail_tokens or end > self.history_capacity:
            raise ValueError("Retired ABC range exceeds retained ring or hot capacity")
        flat = self.bytes_outputs[layer]
        base = owner * self.slot_capacity
        while start < end:
            offset = start % self.tail_tokens
            count = min(end - start, self.tail_tokens - offset)
            source = base + self.tail_start + offset
            flat[base + start : base + start + count].copy_(flat[source : source + count])
            start += count

    def _seed(self, layer, owner, state, start, end, tail):
        if tail or start == end:
            return  # Prefill already left its partial page in the ABC ring.
        runtime = self.writer.runtimes[self.writer.groups[layer].geometry]
        handle = runtime.prepare_access_table(state.keys)
        try:
            table = runtime.access_table_view(handle, device="npu")
            count = torch.tensor([end - start], dtype=torch.int32, device=self.pool.device)
            destinations = self.context_destinations[start:end] + owner * self.slot_capacity
            get_tokens(
                runtime,
                table,
                self.context_indices[start:end],
                self.context_offsets_by_width[self.group_sizes[layer]][start:end],
                count,
                destinations,
                self.bytes_outputs[layer],
                synchronize=True,
            )
        finally:
            runtime.release_access_table(handle)

    def finish(self, manager):
        if not self.enabled:
            return
        original = manager.groups
        manager.groups = self.decode_frames
        try:
            if self.indexer_selectors:
                queries = int(manager.groups[0].query_offsets[-1])
                manager.submit_feedback(
                    self.destinations[: self.group_count, :queries],
                    generation=manager.generation,
                    device_errors=self.fused_errors,
                    fetch_rows=self.fused_fetch_rows,
                )
            else:
                super().finish(manager)
        finally:
            manager.groups = original

    def _source_offset(self, req_id, page):
        plan = self.plans.get(req_id)
        if plan is not None and not plan.decode and page >= plan.start // BLOCK_SIZE:
            return (plan.stage + (page - plan.start // BLOCK_SIZE) * BLOCK_SIZE) * TOKEN_BYTES
        return (
            self.owners[req_id] * self.slot_capacity + self.tail_start + page * BLOCK_SIZE % self.tail_tokens
        ) * TOKEN_BYTES

    def publish(self, req_id, tokens):
        self.writer.publish(self, req_id, tokens)

    def after_forward(self, requests, scheduled):
        self.put_step += 1
        decode = {}
        for req_id, plan in self.plans.items():
            request = requests[req_id]
            if plan.decode:
                safe = max(request.num_prompt_tokens // BLOCK_SIZE, request.num_computed_tokens // BLOCK_SIZE - 1)
                decode[req_id] = max(0, safe) * BLOCK_SIZE
            else:
                self.publish(req_id, plan.end // BLOCK_SIZE * BLOCK_SIZE)
            if not plan.decode and plan.end % BLOCK_SIZE:
                prefix = plan.end % BLOCK_SIZE
                page = plan.end // BLOCK_SIZE
                source = self._source_offset(req_id, page) // TOKEN_BYTES
                destination = plan.owner * self.slot_capacity + self.tail_start + page * BLOCK_SIZE % self.tail_tokens
                self._copy_slots(source, destination, prefix)
        self.put_targets.update(decode)
        if decode and self.put_step % self.put_interval_steps == 0:
            self.writer.submit_decode(self, decode)

    def attention(self, impl, layer, q, rope, topk, metadata, query_ends, *, full_graph=False):
        from vllm_ascend.device.device_op import DeviceOperator

        cache = self.attention_caches[layer]
        width = self.group_sizes[layer]
        local = getattr(impl, "kvaf_group_offset", 0)
        tables = self.resident_tables if width == 1 else self.group_tables[width]
        # Mode 3 bounds selected entries by each query's causal length.
        # Physical hot-slot addresses come from indices and block tables;
        # interleaving layers must not multiply the logical KV lengths.
        lengths = metadata.seq_lens
        if self.only_decode:
            output_rows = q.shape[0]
            batch = metadata.cum_query_lens.shape[0]
            # FULL replay must retain capture-time shapes. Eager attention
            # (including piecewise execution) uses this round's local plans.
            # Capture warmup also runs eagerly, but has synthetic inputs and
            # no request plans. Preserve those inputs rather than trim to zero.
            trim_local = not full_graph and bool(self.plans)
            queries = max(p.stop for p in self.plans.values()) if trim_local else topk.shape[0]
            dp_padding = trim_local and (topk.shape[0] != queries or q.shape[0] != queries or batch != len(self.plans))
            if dp_padding:
                # Another DP rank can still be prefilling: MoE pads this rank
                # to its token count. LIDU only owns local decode queries.
                # Use host plans and existing device query ends; never read a
                # device scalar or resize graph-stable decode buffers here.
                batch = len(self.plans)
                topk, q, rope = topk[:queries], q[:queries], rope[:queries]
                query_ends = self.inputs["query_ends"][:batch]
                metadata = SimpleNamespace(num_prefills=0, cum_query_lens=query_ends)
            mapped = self.run(layer, topk, metadata, local)
            result = DeviceOperator.execute_sparse_flash_attention_process(
                impl,
                q,
                rope,
                (cache,),
                mapped,
                metadata,
                query_ends,
                lengths[:batch],
                block_table=tables[:batch],
                sparse_mode=3,
            )
            if dp_padding and result.shape[0] < output_rows:
                result = torch.cat((result, result.new_zeros((output_rows - result.shape[0], *result.shape[1:]))))
            return result
        decodes = [p for p in self.plans.values() if p.decode]
        mapped = None
        if decodes:
            selected = torch.cat([topk[p.begin : p.stop] for p in decodes])
            native_meta = SimpleNamespace(num_prefills=0, cum_query_lens=self.inputs["query_ends"][: len(decodes)])
            mapped = self.run(layer, selected, native_meta, local)
        results, offset, decode_row = [], 0, 0
        for request_index, p in enumerate(self.plans.values()):
            count = p.stop - p.begin
            if p.decode:
                indices = mapped[offset : offset + count]
                offset += count
                storage, table, key_end, mode = (
                    cache,
                    tables[decode_row : decode_row + 1],
                    lengths[request_index : request_index + 1],
                    3,
                )
                decode_row += 1
            else:
                history = p.start // BLOCK_SIZE * BLOCK_SIZE
                context = self.context.view(-1, 1, TOKEN_BYTES) if width == 1 else self.group_context[0]
                if history and local == 0:
                    self.context_count.fill_(history)
                    runtime = self.writer.runtimes[self.writer.groups[layer].geometry]
                    get_tokens(
                        runtime,
                        self.addresses[p.owner, layer],
                        self.valid[p.owner, layer],
                        self.context_indices,
                        self.context_offsets_by_width[width],
                        self.context_count,
                        self.context_destinations,
                        context.view(torch.uint8).view(-1, width * TOKEN_BYTES),
                        synchronize=False,
                    )
                if width > 1 and history:
                    # Preserve original causal SFA/token numbering. Only HBM
                    # extraction repeats; DDR is fetched once for all four layers.
                    self.context.view(-1, TOKEN_BYTES)[:history].copy_(context[:history, local])
                size = p.end - history
                self.context.view(-1, TOKEN_BYTES)[history : p.end].copy_(
                    cache.view(torch.int8).view(-1, width, TOKEN_BYTES)[p.stage : p.stage + size, local]
                )
                storage, table, key_end, mode = self.context.view(cache.dtype), self.context_table, p.key_end, 3
                indices = topk[p.begin : p.stop]
            results.append(
                DeviceOperator.execute_sparse_flash_attention_process(
                    impl,
                    q[p.begin : p.stop],
                    rope[p.begin : p.stop],
                    (storage,),
                    indices,
                    metadata,
                    p.query_end,
                    key_end,
                    block_table=table,
                    sparse_mode=mode,
                )
            )
        result = torch.cat(results)
        # TP sequence padding has no real request, but O-proj still expects its
        # rows. Keep it out of DDR descriptors and return zero attention there.
        if result.shape[0] < q.shape[0]:
            result = torch.cat((result, result.new_zeros((q.shape[0] - result.shape[0], *result.shape[1:]))))
        return result


@dataclass
class RequestFeedback:
    state: object
    lru: object


@dataclass
class GroupFeedback:
    """Stable holder attached to SFA.impl, with host rows and device tensors.

    Rows follow the runner's final request order, including prefill rows.
    query_offsets defines each request's real queries; graph padding is absent.
    ``device`` exposes fixed-address NPU arrays staged before forward.
    """

    request_ids: tuple = ()
    query_offsets: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32))
    rows: tuple = ()
    generation: int = 0
    device: object = None


class CacheFeedbackManager:
    def __init__(self, writer, *, log_context=""):
        # KVAF-disabled startup must not import/compile Numba or require KVIP.
        from kvip.cache.lru import SlotLRU, warmup

        warmup()
        self.row_factory = SlotLRU
        self.writer = writer
        self.miss_stats = HotMissStats(
            len(writer.groups), log_context, weights=[len(layer.layout.layer_names) for layer in writer.groups]
        )
        self.groups = tuple(GroupFeedback() for _ in writer.groups)
        self.owners = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kvaf-lru")
        self.generation = 0
        self.submitted_generation = -1
        self.device_pool = None
        self.policy = None
        self.logged_feedback = False
        self.logged_get = False
        self.inflight = deque()
        self.feedback_buffers = [{}, {}]
        self.feedback_turn = 0
        self.copy_stream = None
        self.feedback_read_done = None

    def bind(self, forward_context):
        for index, (layer, inputs) in enumerate(zip(self.writer.groups, self.groups)):
            for local_layer, name in enumerate(layer.layout.layer_names):
                forward_context[name].impl.kvaf_hot = (self.policy, index)
                forward_context[name].impl.kvaf_group_offset = local_layer

    def bind_device_pool(self, pool):
        self.device_pool = pool
        self.policy = pool.policy
        self.policy.bind(self)
        for layer, inputs in zip(self.writer.groups, self.groups):
            inputs.device = pool.views[layer.layout.layer_names[0]]

    def poll_feedback(self):
        while self.inflight and self.inflight[0].done():
            self._finish_oldest()

    def _finish_oldest(self):
        accesses = self.inflight[0].result()
        self.inflight.popleft()
        if isinstance(accesses, tuple):
            accesses, misses = accesses
            if misses and not self.logged_get:
                logging.getLogger("vllm").info(
                    "KVAF DDR Get completed: fetched_tokens=%s across Target cache groups; "
                    "SFA uses independent hot slots",
                    misses,
                )
                self.logged_get = True
        if accesses and not self.logged_feedback:
            logging.getLogger("vllm").info(
                "KVAF cache feedback committed: accessed_slots=%s; original LRU updated",
                accesses,
            )
            self.logged_feedback = True

    def wait_pending(self):
        while self.inflight:
            self._finish_oldest()

    def prepare(self, request_ids, scheduled_tokens, *, decode=False, padded_requests=None):
        self.poll_feedback()
        if self.feedback_read_done is not None:
            import torch

            # Sampling/draft work already queued on the producer can overlap
            # D2H. Wait only before resetting/reusing the next round's outputs.
            torch.npu.current_stream(self.copy_stream.device).wait_event(self.feedback_read_done)
            self.feedback_read_done = None
        ids = tuple(req_id for req_id in request_ids if scheduled_tokens.get(req_id, 0) > 0)
        offsets = np.zeros(len(ids) + 1, dtype=np.int32)
        np.cumsum([scheduled_tokens[req_id] for req_id in ids], out=offsets[1:])
        self.generation += 1
        for index, inputs in enumerate(self.groups):
            rows = []
            for req_id in ids:
                state = self.writer.requests[req_id][index]
                key = (req_id, index)
                row = self.owners.get(key)
                if row is None or row.state is not state:
                    row = self.owners[key] = RequestFeedback(state, self.row_factory())
                rows.append(row)
            inputs.request_ids, inputs.query_offsets, inputs.rows = ids, offsets, tuple(rows)
            inputs.generation = self.generation
        self.policy.prepare(self, decode=decode, padded_requests=padded_requests or len(ids))

    def after_forward(self):
        """Enqueue access feedback from the completed serving forward."""
        self.policy.finish(self)

    def submit_feedback(self, destinations, *, generation, miss_counts=None, device_errors=None, fetch_rows=None):
        """Accept real LIDU feedback [layers, queries, (1), topk].

        NPU feedback copies on a separate stream; the worker waits for its
        event before touching host arrays. Before reusing graph-stable outputs,
        prepare enqueues a device-side wait, not a host synchronization.
        Native-mode prepare only polls CPU work;
        release and bounded queue exhaustion join outstanding work.
        No synthetic feedback is generated by the current service path.
        """
        if generation != self.generation or generation == self.submitted_generation:
            raise RuntimeError("Stale or duplicate LIDU feedback")
        if destinations.ndim == 4 and destinations.shape[2] == 1:
            destinations = destinations.squeeze(2)  # Metadata view, no device kernel.
        frames = tuple(
            (
                tuple(
                    (row.lru, self.policy.slot_capacity, self.policy.history_capacity, self.policy.history_capacity)
                    for row in inputs.rows
                ),
                inputs.query_offsets.copy(),
            )
            for inputs in self.groups
        )
        queries = int(frames[0][1][-1]) if frames else 0
        if destinations.ndim != 3 or destinations.shape[:2] != (len(frames), queries):
            raise ValueError("LIDU feedback must be [layers, real queries, topk]")
        # Bounded double buffering; only queue exhaustion applies backpressure.
        self.poll_feedback()
        if len(self.inflight) == len(self.feedback_buffers):
            self._finish_oldest()
        buffer = self.feedback_buffers[self.feedback_turn % len(self.feedback_buffers)]
        traffic = fetch_rows is not None
        if traffic:
            miss_counts = fetch_rows
        ready = None
        errors = None
        if isinstance(destinations, np.ndarray):
            if destinations.dtype != np.int32:
                raise TypeError("LIDU feedback must use int32")
            host = destinations.copy()
            counts = None if miss_counts is None else np.array(miss_counts, copy=True)
            errors = None if device_errors is None else np.array(device_errors, copy=True)
        else:
            import torch

            if destinations.dtype != torch.int32 or destinations.device.type != "npu":
                raise TypeError("LIDU feedback must be an int32 NPU tensor or NumPy array")
            if self.copy_stream is None:
                self.copy_stream = torch.npu.Stream(device=destinations.device)
            # Reuse each ring slot's events only after its CPU consumer exits.
            if "produced" not in buffer:
                buffer["produced"], buffer["copied"] = torch.npu.Event(), torch.npu.Event()
            produced, ready = buffer["produced"], buffer["copied"]
            produced.record(torch.npu.current_stream(destinations.device))
            elements = destinations.numel()
            if "slots" not in buffer or buffer["slots"].numel() < elements:
                buffer["slots"] = torch.empty(elements, dtype=torch.int32, device="cpu", pin_memory=True)
            staging = buffer["slots"][:elements].view(destinations.shape)
            counts = None
            if miss_counts is not None:
                elements = miss_counts.numel()
                if "counts" not in buffer or buffer["counts"].numel() < elements:
                    buffer["counts"] = torch.empty(elements, dtype=torch.int32, device="cpu", pin_memory=True)
                count_staging = buffer["counts"][:elements].view(miss_counts.shape)
                counts = count_staging.numpy()
            self.copy_stream.wait_event(produced)
            with torch.npu.stream(self.copy_stream):
                if device_errors is not None:
                    elements = device_errors.numel()
                    if "errors" not in buffer or buffer["errors"].numel() < elements:
                        buffer["errors"] = torch.empty(elements, dtype=torch.int32, device="cpu", pin_memory=True)
                    error_staging = buffer["errors"][:elements].view(device_errors.shape)
                    error_staging.copy_(device_errors, non_blocking=True)
                    device_errors.record_stream(self.copy_stream)
                    errors = error_staging.numpy()
                for layer in range(destinations.shape[0]):
                    staging[layer].copy_(destinations[layer], non_blocking=True)
                if miss_counts is not None:
                    count_staging.copy_(miss_counts, non_blocking=True)
                destinations.record_stream(self.copy_stream)
                if miss_counts is not None:
                    miss_counts.record_stream(self.copy_stream)
                ready.record(self.copy_stream)
            self.feedback_read_done = ready
            host = staging.numpy()  # Keeps the pinned tensor alive through NumPy.base.
        self.submitted_generation = generation
        future = self.executor.submit(self._commit, frames, host, ready, counts, self.miss_stats, errors, traffic)
        self.inflight.append(future)
        self.feedback_turn += 1

    @staticmethod
    def _commit(frames, host, ready, miss_counts=None, stats=None, device_errors=None, traffic=False):
        if ready is not None:
            ready.synchronize()  # Background thread only, not the model launch thread.
        if device_errors is not None and np.any(device_errors):
            failures = np.argwhere(device_errors != 0)
            details = []
            for group, request in failures[:8]:
                status = int(device_errors[group, request])
                item = {"group": int(group), "row": int(request), "status": status}
                item["active"] = request < len(frames[group][0])
                if traffic and miss_counts is not None:
                    row = miss_counts[group, request]
                    item.update(
                        prefetch_error=int(row[11]),
                        prefetch_reads=int(row[9]),
                        prefetch_pending=int(row[211]),
                        final_reads=int(row[247]),
                        final_pending=int(row[248]),
                    )
                details.append(item)
            raise RuntimeError(
                f"LIFetch Index update failed: failed_rows={len(failures)} details={details}; "
                "status bits: 1=invalid Index change records, 2=prefetch state, "
                "4=resolver/transport failure. Counts are from the same feedback copy."
            )
        if traffic:
            # Original v9 ABI: word 9 counts prefetch requests, 247 final reads.
            # Ignore graph padding; sum on CPU after the existing copy event.
            requests = len(frames[0][0])
            reads = miss_counts[:, :requests, 9].astype(np.int64) + miss_counts[:, :requests, 247]
            miss_counts = np.concatenate((reads, reads.sum(axis=1, keepdims=True)), axis=1)
        if miss_counts is not None and np.any(miss_counts < 0):
            raise RuntimeError(
                f"LIDU expected nonnegative counts; got {miss_counts.tolist()}. "
                "Negative counts indicate native errors or missing layer execution."
            )
        jobs = []
        for layer, (rows, offsets) in enumerate(frames):
            for index, row in enumerate(rows):
                lru, capacity = row[:2]
                values = host[layer, offsets[index] : offsets[index + 1]].reshape(-1)
                if np.any(values < -1) or np.any(values >= capacity):
                    raise ValueError("LIDU feedback points outside request HBM slots")
                jobs.append((lru, values[values >= 0], row[2:]))
        # Validate the complete frame before changing any LRU chain.
        references = (
            np.zeros((len(frames), len(frames[0][0])), dtype=np.int64)
            if stats is not None and miss_counts is not None
            else None
        )
        for job_index, (lru, values, resize) in enumerate(jobs):
            if resize:
                lru.resize(*resize)  # FIFO worker exclusively owns native-mode LRU.
            if references is not None:
                lru.unique[0] = 0  # commit() may early-return for an ABC-only row.
            lru.commit(values)
            if references is not None:
                references.flat[job_index] = lru.unique[0]
            lru.hot_snapshot = (lru.slots[0].copy(), lru.bitmap[0].copy())
        if references is not None:
            stats.record(
                miss_counts[:, : references.shape[1]],
                references,
                get_counts=miss_counts[:, -1],
                ddr_fetch=traffic,
            )
        accesses = sum(values.size for _, values, _ in jobs)
        return accesses if miss_counts is None else (accesses, int(miss_counts[:, -1].sum()))

    def release(self, req_id):
        self.wait_pending()
        self.policy.release(req_id)
        self.generation += 1  # Reject feedback captured before the release.
        for index in range(len(self.groups)):
            self.owners.pop((req_id, index), None)
        # Drop borrowed map references so finished requests cannot be consumed.
        for inputs in self.groups:
            inputs.request_ids, inputs.rows = (), ()
            inputs.query_offsets = np.zeros(1, dtype=np.int32)

    def close(self):
        try:
            self.wait_pending()
        finally:
            self.executor.shutdown(wait=True)
            self.miss_stats.flush()
