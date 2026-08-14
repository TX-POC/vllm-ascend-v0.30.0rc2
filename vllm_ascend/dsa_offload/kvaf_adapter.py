# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin adapter for the external KVAF Python package."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm_ascend import envs
from vllm_ascend.dsa_offload.dsa_row import (
    PackedKvLayout,
)


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
        device_id = int(device_id)
        torch.npu.set_device(device_id)
        self.runtime = npu_runtime(
            batch_get_kernel,
            device_id,
        )

    @staticmethod
    def slot_size(layout: PackedKvLayout) -> int:
        return _align_up(
            layout.tokens_per_logical_block * layout.packed_token_bytes,
            layout.ddr_layer_alignment_bytes,
        )

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

    def batch_free(
        self,
        *,
        flat_indices: Sequence[int],
    ) -> None:
        """Free DSA_ROW flat indices through KVAF's existing API."""

        keys: list[int] = []
        for flat_index in flat_indices:
            flat_index = int(flat_index)
            if flat_index < 0:
                raise ValueError(
                    f"flat_index must be non-negative, got {flat_index}"
                )
            keys.append(flat_index)
        if not keys:
            return

        free = getattr(self.runtime, "batch_free", None)
        if not callable(free):
            raise RuntimeError(
                "The external kvaf.Runtime does not expose batch_free()"
            )
        free(keys)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


__all__ = ["KvafAdapter"]
