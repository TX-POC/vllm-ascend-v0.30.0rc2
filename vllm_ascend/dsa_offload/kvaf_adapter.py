# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin adapter for the external KVAF Python package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vllm_ascend import envs
from vllm_ascend.dsa_offload.dsa_row import (
    DDRBlockTable,
    DsaRowIndexGeometry,
    PackedKvLayout,
)


@dataclass
class _ActiveBatchPut:
    pool_rows: np.ndarray
    block_qs: np.ndarray
    source_block_ids: np.ndarray
    geometry: DsaRowIndexGeometry
    written_layers: set[int] = field(default_factory=set)


class KvafAdapter:
    """Import and dispatch the external KVAF runtime without duplicating it."""

    def __init__(
        self,
        *,
        device_id: int = 0,
    ) -> None:
        try:
            import kvaf
        except ImportError as error:
            raise RuntimeError(
                "The external kvaf package is required for DSA offload"
            ) from error

        npu_runtime = getattr(kvaf, "NpuRuntime", None)
        if npu_runtime is None:
            raise RuntimeError(
                "The external kvaf package does not provide NpuRuntime; "
                "build KVAF with KVAF_ENABLE_AIV_BATCH_GET=ON"
            )
        batch_get_kernel = envs.VLLM_ASCEND_KVAF_BATCH_GET_KERNEL
        if not batch_get_kernel:
            raise RuntimeError(
                "VLLM_ASCEND_KVAF_BATCH_GET_KERNEL must point to the "
                "external KVAF BatchGet kernel object"
            )
        batch_put_kernel = envs.VLLM_ASCEND_KVAF_BATCH_PUT_KERNEL
        if not batch_put_kernel:
            raise RuntimeError(
                "VLLM_ASCEND_KVAF_BATCH_PUT_KERNEL must point to the "
                "external KVAF BatchPut kernel object"
            )
        device_id = int(device_id)
        torch.npu.set_device(device_id)
        self.runtime = npu_runtime(
            batch_get_kernel,
            device_id,
            envs.VLLM_ASCEND_KVAF_NPU_CORE_COUNT,
            batch_put_kernel,
            envs.VLLM_ASCEND_KVAF_JETTYS_PER_CORE,
        )
        self._layout: PackedKvLayout | None = None
        self._geometry: DsaRowIndexGeometry | None = None
        self._committed_block_counts: dict[int, int] = {}
        self._active_put: _ActiveBatchPut | None = None

    @staticmethod
    def slot_size(layout: PackedKvLayout) -> int:
        return _align_up(
            layout.tokens_per_logical_block * layout.packed_token_bytes,
            layout.ddr_layer_alignment_bytes,
        )

    @property
    def active_batch_put(self) -> bool:
        return self._active_put is not None

    def initialize_dsa_row_index(
        self,
        *,
        layout: PackedKvLayout,
        ddr_bytes: int,
    ) -> None:
        """Register KVAF's region using DSA's packed layer slot size."""

        ddr_bytes = int(ddr_bytes)
        if ddr_bytes <= 0:
            raise ValueError(f"ddr_bytes must be positive, got {ddr_bytes}")
        slot_size = self.slot_size(layout)
        initialized = getattr(self.runtime, "initialized", None)
        if initialized is not True:
            initialize = getattr(self.runtime, "initialize", None)
            if not callable(initialize):
                raise RuntimeError(
                    "The external kvaf.Runtime does not expose initialize()"
                )
            initialize(ddr_bytes, slot_size)
        self.validate_region(layout=layout)
        self._layout = layout

    def validate_region(self, *, layout: PackedKvLayout) -> None:
        """Check that KVAF's registered fixed slot matches DSA's layer block."""

        region = getattr(self.runtime, "remote_region", None)
        if not isinstance(region, dict):
            raise RuntimeError(
                "The external kvaf.Runtime does not expose remote_region"
            )
        expected_slot_size = self.slot_size(layout)
        actual_slot_size = int(region.get("slot_size", 0))
        if actual_slot_size != expected_slot_size:
            raise RuntimeError(
                "KVAF region slot_size does not match DSA packed layout: "
                f"kvaf={actual_slot_size}, expected={expected_slot_size}"
            )

    def committed_block_count(self, pool_row: int) -> int:
        geometry = self._require_geometry()
        pool_row = int(pool_row)
        if not 0 <= pool_row < geometry.raw_row_capacity:
            raise IndexError("KVAF pool row is outside DSA_ROW")
        return self._committed_block_counts.get(pool_row, 0)

    def begin_batch_put(
        self,
        *,
        pool_rows: np.ndarray,
        block_qs: np.ndarray,
        source_block_ids: np.ndarray,
        geometry: DsaRowIndexGeometry,
    ) -> None:
        """Freeze one append before any layer is sent to KVAF."""

        if self._active_put is not None:
            raise RuntimeError("KVAF BatchPut already has active jobs")
        pool_rows = np.asarray(pool_rows, dtype=np.int64)
        block_qs = np.asarray(block_qs, dtype=np.int64)
        source_block_ids = np.asarray(source_block_ids, dtype=np.int64)
        if (
            pool_rows.ndim != 1
            or block_qs.ndim != 1
            or source_block_ids.ndim != 1
            or pool_rows.size == 0
            or not (pool_rows.shape == block_qs.shape == source_block_ids.shape)
        ):
            raise ValueError("KVAF BatchPut arrays must be non-empty 1D arrays")
        if (
            np.any(pool_rows < 0)
            or np.any(pool_rows >= geometry.raw_row_capacity)
            or np.any(block_qs < 0)
            or np.any(block_qs >= geometry.q_capacity)
        ):
            raise IndexError("KVAF BatchPut coordinates are outside DSA_ROW")
        if self._geometry is None:
            self._geometry = geometry
        elif self._geometry != geometry:
            raise RuntimeError("KVAF DSA_ROW geometry changed after first Put")
        self._active_put = _ActiveBatchPut(
            pool_rows=pool_rows,
            block_qs=block_qs,
            source_block_ids=source_block_ids,
            geometry=geometry,
        )

    def write_batch_put_layer(
        self,
        *,
        local_layer_ordinal: int,
        packed_cache: torch.Tensor,
        stream: Any = None,
    ) -> None:
        """Write one layer's complete packed blocks through KVAF BatchPut."""

        active = self._require_active_put()
        geometry = active.geometry
        layer_id = int(local_layer_ordinal)
        if not 0 <= layer_id < geometry.n_layers:
            raise IndexError("KVAF layer is outside DSA_ROW")
        if layer_id in active.written_layers:
            raise RuntimeError(f"KVAF layer {layer_id} was written twice")
        if packed_cache.ndim < 2 or not packed_cache.is_contiguous():
            raise ValueError(
                "KVAF BatchPut requires a contiguous packed block Tensor"
            )
        max_source = int(active.source_block_ids.max())
        if max_source >= int(packed_cache.shape[0]):
            raise IndexError("KVAF source block is outside the packed cache")

        layout = self._require_layout()
        block_bytes = int(
            packed_cache[0].numel() * packed_cache.element_size()
        )
        expected_block_bytes = (
            layout.tokens_per_logical_block * layout.packed_token_bytes
        )
        if block_bytes != expected_block_bytes:
            raise ValueError(
                "KVAF packed block size does not match DSA layout: "
                f"cache={block_bytes}, expected={expected_block_bytes}"
            )
        block_stride_bytes = int(
            packed_cache.stride(0) * packed_cache.element_size()
        )
        if block_stride_bytes < block_bytes:
            raise ValueError("KVAF packed cache has an invalid block stride")

        keys = self._build_layer_keys(
            geometry,
            active.pool_rows,
            active.block_qs,
            layer_id,
        )
        input_offsets = (
            active.source_block_ids.astype(np.uint64, copy=False)
            * np.uint64(block_stride_bytes)
        )
        lengths = np.full(keys.size, block_bytes, dtype=np.uint64)
        self.runtime.batch_put(
            keys.tolist(),
            packed_cache,
            input_offsets.tolist(),
            lengths.tolist(),
            stream,
        )
        active.written_layers.add(layer_id)

    def finalize_batch_put(
        self,
        *,
        block_table: DDRBlockTable,
    ) -> dict[int, int]:
        """Publish all new addresses only after every layer was written."""

        active = self._require_active_put()
        geometry = active.geometry
        device_addresses = self._validate_block_table(
            block_table,
            geometry=geometry,
        )
        missing = sorted(set(range(geometry.n_layers)) - active.written_layers)
        if missing:
            raise RuntimeError(
                "KVAF BatchPut cannot publish before all layers are written: "
                f"missing={missing}"
            )

        all_keys = self._build_all_layer_keys(
            geometry,
            active.pool_rows,
            active.block_qs,
        )
        cpu_keys = torch.from_numpy(all_keys.copy())
        handle = self.runtime.prepare_access_table(cpu_keys)
        try:
            expected_size = all_keys.size
            device_view = self.runtime.access_table_view(
                handle,
                device="npu",
                capacity=int(expected_size),
            )
            addresses = getattr(device_view, "addresses", None)
            valid = getattr(device_view, "valid", None)
            self._validate_device_view(
                addresses=addresses,
                valid=valid,
                expected_size=expected_size,
                expected_device=device_addresses.device,
            )
            device_keys = torch.as_tensor(
                all_keys,
                dtype=torch.int64,
                device=device_addresses.device,
            )
            self._index_copy_compat(
                device_addresses,
                device_keys,
                addresses.to(dtype=device_addresses.dtype),
            )
            expected_block_bytes = (
                self._require_layout().tokens_per_logical_block
                * self._require_layout().packed_token_bytes
            )
            self._index_fill_compat(
                block_table.lengths,
                device_keys,
                expected_block_bytes,
            )
            self._index_copy_compat(block_table.valid, device_keys, valid)
            self._synchronize_device(device_addresses.device)
        finally:
            self.runtime.release_access_table(handle)

        updated: dict[int, int] = {}
        for pool_row in np.unique(active.pool_rows):
            row = int(pool_row)
            appended = int(np.count_nonzero(active.pool_rows == row))
            committed = self.committed_block_count(row) + appended
            self._committed_block_counts[row] = committed
            updated[row] = committed
        self._active_put = None
        return updated

    def abort_batch_put(
        self,
        *,
        block_table: DDRBlockTable | None = None,
    ) -> None:
        """Free layer keys committed before a group-level failure."""

        active = self._active_put
        if active is None:
            return
        try:
            written_keys = self._build_keys_for_layers(
                active.geometry,
                active.pool_rows,
                active.block_qs,
                active.written_layers,
            )
            if block_table is not None:
                self._clear_table(block_table, written_keys)
            if written_keys.size:
                self.runtime.batch_free(torch.from_numpy(written_keys.copy()))
        finally:
            self._active_put = None

    def release_pool_row(
        self,
        pool_row: int,
        *,
        block_table: DDRBlockTable,
    ) -> None:
        """Free every committed KVAF key before a pool row is reused."""

        pool_row = int(pool_row)
        if self._active_put is not None and np.any(
            self._active_put.pool_rows == pool_row
        ):
            raise RuntimeError("Cannot release a row with an active KVAF BatchPut")
        committed = self._committed_block_counts.get(pool_row, 0)
        if committed == 0:
            return
        geometry = self._require_geometry()
        rows = np.full(committed, pool_row, dtype=np.int64)
        block_qs = np.arange(committed, dtype=np.int64)
        keys = self._build_all_layer_keys(geometry, rows, block_qs)
        self._clear_table(block_table, keys)
        self.runtime.batch_free(torch.from_numpy(keys.copy()))
        self._committed_block_counts.pop(pool_row, None)

    def shutdown(self, *, block_table: DDRBlockTable) -> None:
        """Release DSA keys and close the KVAF runtime."""

        self.abort_batch_put(block_table=block_table)
        for pool_row in tuple(self._committed_block_counts):
            self.release_pool_row(pool_row, block_table=block_table)
        shutdown = getattr(self.runtime, "shutdown", None)
        if callable(shutdown) and getattr(self.runtime, "initialized", False):
            shutdown()

    @staticmethod
    def _build_layer_keys(
        geometry: DsaRowIndexGeometry,
        pool_rows: np.ndarray,
        block_qs: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        return (
            pool_rows * np.int64(geometry.row_stride_cells)
            + np.int64(layer_id * geometry.q_capacity)
            + block_qs
        )

    @classmethod
    def _build_all_layer_keys(
        cls,
        geometry: DsaRowIndexGeometry,
        pool_rows: np.ndarray,
        block_qs: np.ndarray,
    ) -> np.ndarray:
        return cls._build_keys_for_layers(
            geometry,
            pool_rows,
            block_qs,
            range(geometry.n_layers),
        )

    @classmethod
    def _build_keys_for_layers(
        cls,
        geometry: DsaRowIndexGeometry,
        pool_rows: np.ndarray,
        block_qs: np.ndarray,
        layer_ids: Any,
    ) -> np.ndarray:
        layer_keys = [
            cls._build_layer_keys(
                geometry,
                pool_rows,
                block_qs,
                layer_id,
            )
            for layer_id in sorted(layer_ids)
        ]
        return (
            np.concatenate(layer_keys)
            if layer_keys
            else np.empty(0, dtype=np.int64)
        )

    @staticmethod
    def _validate_device_view(
        *,
        addresses: Any,
        valid: Any,
        expected_size: int,
        expected_device: torch.device,
    ) -> None:
        if (
            not isinstance(addresses, torch.Tensor)
            or addresses.dtype != torch.int64
            or addresses.ndim != 1
            or addresses.numel() != expected_size
        ):
            raise TypeError("KVAF NPU AccessTable returned invalid addresses")
        if (
            not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.uint8
            or valid.ndim != 1
            or valid.numel() != expected_size
        ):
            raise TypeError("KVAF NPU AccessTable returned invalid valid bits")
        if addresses.device != expected_device or valid.device != expected_device:
            raise ValueError(
                "KVAF NPU AccessTable and fixed DSA table must share a device"
            )

    def _clear_table(
        self,
        block_table: DDRBlockTable,
        keys: np.ndarray,
    ) -> None:
        geometry = self._require_geometry()
        device_addresses = self._validate_block_table(
            block_table,
            geometry=geometry,
        )
        keys = np.asarray(keys, dtype=np.int64)
        device_keys = torch.as_tensor(
            keys,
            dtype=torch.int64,
            device=device_addresses.device,
        )
        self._index_fill_compat(device_addresses, device_keys, 0)
        self._index_fill_compat(block_table.lengths, device_keys, 0)
        self._index_fill_compat(block_table.valid, device_keys, 0)
        self._synchronize_device(device_addresses.device)

    @staticmethod
    def _index_copy_compat(
        target: torch.Tensor,
        indices: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Update fixed metadata tensors across CPU and NPU backends.

        CPU PyTorch does not implement ``index_copy`` for ``uint64``.  The
        HBM path keeps the regular indexed kernel; the CPU fallback is used
        by unit tests and host-side validation only.
        """

        if target.device.type == "cpu" and target.dtype == torch.uint64:
            # CPU PyTorch has no UInt64 index_copy/index_put kernel.  The
            # tensor is a host metadata buffer here, so update its NumPy view
            # without allocating a second table.
            target.detach().numpy()[
                indices.detach().numpy()
            ] = values.detach().numpy()
            return
        target.index_copy_(0, indices, values)

    @staticmethod
    def _index_fill_compat(
        target: torch.Tensor,
        indices: torch.Tensor,
        value: int,
    ) -> None:
        if target.device.type == "cpu" and target.dtype == torch.uint64:
            target.detach().numpy()[indices.detach().numpy()] = value
            return
        target.index_fill_(0, indices, value)

    def _require_layout(self) -> PackedKvLayout:
        if self._layout is None:
            raise RuntimeError("KVAF DSA_ROW layout has not been initialized")
        return self._layout

    def _require_geometry(self) -> DsaRowIndexGeometry:
        if self._geometry is None:
            raise RuntimeError("KVAF BatchPut geometry has not been prepared")
        return self._geometry

    @staticmethod
    def _validate_block_table(
        block_table: DDRBlockTable,
        *,
        geometry: DsaRowIndexGeometry,
    ) -> torch.Tensor:
        addresses = block_table.addresses
        if (
            not isinstance(addresses, torch.Tensor)
            or addresses.dtype != torch.uint64
            or addresses.numel() != geometry.directory_cell_count
            or not addresses.is_contiguous()
        ):
            raise ValueError(
                "DDRBlockTable addresses must be a contiguous uint64 Tensor "
                "with DSA_ROW capacity"
            )
        lengths = block_table.lengths
        valid = block_table.valid
        if (
            not isinstance(lengths, torch.Tensor)
            or lengths.dtype != torch.uint64
            or lengths.shape != addresses.shape
            or not lengths.is_contiguous()
            or lengths.device != addresses.device
        ):
            raise ValueError(
                "DDRBlockTable lengths must be a contiguous uint64 Tensor "
                "on the address-table device"
            )
        if (
            not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.uint8
            or valid.shape != addresses.shape
            or not valid.is_contiguous()
            or valid.device != addresses.device
        ):
            raise ValueError(
                "DDRBlockTable valid must be a contiguous uint8 Tensor "
                "on the address-table device"
            )
        return addresses.view(-1)

    def _require_active_put(self) -> _ActiveBatchPut:
        if self._active_put is None:
            raise RuntimeError("KVAF BatchPut has no active jobs")
        return self._active_put

    @staticmethod
    def _synchronize_device(device: torch.device) -> None:
        if device.type == "npu":
            torch.npu.synchronize(device)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


__all__ = ["KvafAdapter"]
