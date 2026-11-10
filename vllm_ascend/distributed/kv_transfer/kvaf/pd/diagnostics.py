# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded PD evidence. Never imported by the feature-off path.

SHA256 covers every byte in selected transfer segments, not sampled values.
These checks synchronize and read back device data; use only for diagnostics.
No prompt, generated text, raw KV payload, or UBA base is logged.
"""

import hashlib
import json
import logging
from functools import wraps
from threading import RLock

import torch

from .layout import BLOCK_SIZE, MAX_SEGMENTS, PLANE_BYTES, contributor_range


class Diagnostics:
    def __init__(self, requests, verify, role, worker=None):
        self.limit, self.verify, self.role = requests, verify, role
        self.worker = worker or role
        self.selected, self.aliases, self.seen = set(), {}, set()
        self.lock = RLock()
        self.logger = logging.getLogger("vllm")
        self.retained = None  # Keep uncertain diagnostic Get buffers/handle alive.
        self.context = ()
        self.draft_skipped = False
        self.indexer_owners = set()

    def select(self, request, generation, alias=None):
        key = (request, generation)
        with self.lock:
            if key not in self.selected:
                if len(self.selected) >= self.limit:
                    return False
                self.selected.add(key)
            self.aliases[request] = key
            if alias is not None:
                self.aliases[alias] = key
            return True

    def active(self, request):
        return request in self.aliases

    def fresh(self, event, key):
        with self.lock:
            identity = (event, key)
            if identity in self.seen:
                return False
            self.seen.add(identity)
            return True

    def emit(self, event, key, **fields):
        if self.fresh(event, key):
            self.logger.info(
                "KVAF_PD_DIAG %s",
                json.dumps(dict(event=event, role=self.role, worker=self.worker, **fields), sort_keys=True),
            )

    @staticmethod
    def hashes(tensor, offsets, lengths):
        if len(offsets) != len(lengths) or not 1 <= len(offsets) <= MAX_SEGMENTS:
            raise ValueError("diagnostic segments must have equal nonempty lengths within SDK limits")
        raw = tensor.detach().view(torch.uint8).reshape(-1)
        result = []
        for offset, size in zip(offsets, lengths):
            if (
                type(offset) is not int
                or type(size) is not int
                or offset < 0
                or size <= 0
                or offset + size > raw.numel()
            ):
                raise ValueError("diagnostic segment exceeds its actual tensor storage")
            result.append(hashlib.sha256(raw[offset : offset + size].cpu().numpy().tobytes()).hexdigest())
        return tuple(result)

    def compare(self, expected, actual, key, **fields):
        if not expected or len(expected) != len(actual):
            raise ValueError("missing or incomplete source SHA256 evidence")
        mismatches = [i for i, (source, dest) in enumerate(zip(expected, actual)) if source != dest]
        self.emit(
            "KV_CHECK",
            key,
            status="FAIL" if mismatches else "PASS",
            segments=len(actual),
            mismatches=mismatches[:4],
            source_sha256=list(expected[:2]),
            destination_sha256=list(actual[:2]),
            **fields,
        )
        if mismatches:
            raise RuntimeError(f"PD KV byte mismatch: {fields}, segments={mismatches[:4]}")

    @staticmethod
    def spans(memory, tables, tokens, first, final, member, ratio, skipped=False):
        """Independent PA formula oracle against the real tensor views.

        Does not call LayerLayout.offset/plan_layer; checks those calculations
        against plane_PA - arena_PA + physical_block * 128 * plane_token_bytes.
        """
        result = {"npu": [], "cpu": []}
        full, tail = divmod(tokens, BLOCK_SIZE)
        end = full + int(final and tail > 0)
        owned = contributor_range(end - first, member, ratio, first)
        for plane, width in () if skipped else PLANE_BYTES:
            if plane not in memory.planes:
                continue
            if plane == "main":
                if member:
                    continue
                blocks = range(first, full) if memory.offload_main else ()
                selected = ((full,) if final and tail else ()) if memory.offload_main else range(first, end)
            else:
                blocks, selected = (), owned
            for path, chosen in (("cpu", blocks), ("npu", selected)):
                for q in chosen:
                    offset = (
                        memory.planes[plane].data_ptr()
                        - memory.tensor.data_ptr()
                        + tables[plane][q] * BLOCK_SIZE * width
                    )
                    result[path].append((offset, min(BLOCK_SIZE, tokens - q * BLOCK_SIZE) * width))
        return {
            path: tuple(
                (tuple(o for o, _ in items[i : i + MAX_SEGMENTS]), tuple(n for _, n in items[i : i + MAX_SEGMENTS]))
                for i in range(0, len(items), MAX_SEGMENTS)
            )
            for path, items in result.items()
        }

    def address_check(self, expected, actual, key, **fields):
        valid = expected == actual
        self.emit("ADDRESS_CHECK", key, status="PASS" if valid else "FAIL", **fields)
        if not valid:
            raise RuntimeError(f"PD segment addresses disagree with actual Tensor PA/block formula: {fields}")

    def ddr_hashes(self, runtime, transfer, device):
        """Read committed pages through native Get, with one-page scratch HBM.

        Keep metadata/output/access-table references if diagnostic Get raises.
        The receiver marks Runtime failed in that case; no subsequent Get or
        forced resource reuse is allowed across an uncertain diagnostic read.
        """
        entries = transfer.put.entries
        output = torch.empty((BLOCK_SIZE, 656), dtype=torch.uint8, device=device)
        offsets = torch.arange(BLOCK_SIZE, dtype=torch.int32, device=device) * 656
        destinations = torch.arange(BLOCK_SIZE, dtype=torch.int32, device=device)
        count = torch.tensor([BLOCK_SIZE], dtype=torch.int32, device=device)
        indices = torch.empty(BLOCK_SIZE, dtype=torch.int32, device=device)
        handle = runtime.prepare_access_table([entry.key for entry in entries])
        self.retained = (handle, output, offsets, destinations, count, indices)
        table = runtime.access_table_view(handle, device="npu")
        self.retained = (*self.retained, table)
        result = []
        for index, entry in enumerate(entries):
            if entry.length != BLOCK_SIZE * 656:
                raise ValueError("diagnostic DDR read requires complete single-layer C8 pages")
            indices.fill_(index)
            runtime.batch_get(table, indices, offsets, count, destinations, output, synchronize=True)
            result.extend(self.hashes(output, (0,), (entry.length,)))
        runtime.release_access_table(handle)
        self.retained = None
        return tuple(result)

    def layout(self, name, memory):
        layout = memory.describe(name)
        self.emit(
            "LAYOUT",
            name,
            layer=name,
            layer_pa=layout.pa_base,
            bytes=memory.tensor.numel() * memory.tensor.element_size(),
            offload_main=memory.offload_main,
            planes={
                plane: dict(
                    pa=t.data_ptr(),
                    dtype=str(t.dtype),
                    shape=list(t.shape),
                    stride=list(t.stride()),
                    offset=t.data_ptr() - layout.pa_base,
                )
                for plane, t in memory.planes.items()
            },
        )

    def decorate_indexer(self, indexer, layer, *, draft=False):
        # Native IndexerWrapper dispatches compute/cache writes to its impl.
        if not callable(getattr(indexer, "write_cache", None)):
            indexer = getattr(indexer, "impl", None)
        if not callable(getattr(indexer, "write_cache", None)):
            self.emit("INDEXER_UNSUPPORTED", layer, layer=layer)
            return
        if id(indexer) in self.indexer_owners:
            return
        native = indexer.write_cache

        @wraps(native)
        def write(k, scale, *args, **kwargs):
            requests = tuple(req for req in self.context if self.active(req))
            dummy = bool(draft and self.draft_skipped)
            for request in requests:
                key = (request, layer, dummy)
                self.emit(
                    "INDEXER_WRITE",
                    key,
                    request=request,
                    layer=layer,
                    operator="store_kv_block" if indexer._use_c8_reshape_optim() else "scatter_cache",
                    dtype=str(k.dtype),
                    scale_dtype=str(scale.dtype) if scale is not None else None,
                    cache_dtype=str(indexer.k_cache.kv_cache[0].dtype),
                    dummy=dummy,
                )
            result = native(k, scale, *args, **kwargs)
            for request in requests:
                self.emit("INDEXER_WRITE_DONE", (request, layer, dummy), request=request, layer=layer, dummy=dummy)
            return result

        # Instance-local decorator, installed only in this opt-in PD Runner.
        indexer.write_cache = write
        self.indexer_owners.add(id(indexer))
