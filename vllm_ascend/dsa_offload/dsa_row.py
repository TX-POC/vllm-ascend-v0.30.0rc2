# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal DSA_ROW layout, directory table, and flat-index geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch

_INT32_MAX = (1 << 31) - 1

M0_LAYOUT_ID = "A5_C8_W4A4C8_V1"
M0_DDR_LAYER_ALIGNMENT_BYTES = 4096


def _positive(value: int, *, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class PackedKvLayout:
    """Packed KV parameters frozen at DSA_ROW initialization."""

    layout_id: str
    tokens_per_logical_block: int
    packed_token_bytes: int
    ddr_layer_alignment_bytes: int

    def __post_init__(self) -> None:
        if not self.layout_id:
            raise ValueError("layout_id must not be empty")
        _positive(
            self.tokens_per_logical_block,
            name="tokens_per_logical_block",
        )
        _positive(self.packed_token_bytes, name="packed_token_bytes")
        _positive(
            self.ddr_layer_alignment_bytes,
            name="ddr_layer_alignment_bytes",
        )


@dataclass(frozen=True)
class DsaRowIndexGeometry:
    """Row-major geometry shared by DSA and external KVAF."""

    raw_row_capacity: int
    n_layers: int
    q_capacity: int
    row_stride_cells: int
    directory_cell_count: int

    @classmethod
    def from_runtime(
        cls,
        *,
        raw_row_capacity: int,
        n_layers: int,
        max_sequence_tokens: int,
        tokens_per_logical_block: int,
    ) -> DsaRowIndexGeometry:
        raw_row_capacity = _positive(
            raw_row_capacity,
            name="raw_row_capacity",
        )
        n_layers = _positive(n_layers, name="n_layers")
        max_sequence_tokens = _positive(
            max_sequence_tokens,
            name="max_sequence_tokens",
        )
        tokens_per_logical_block = _positive(
            tokens_per_logical_block,
            name="tokens_per_logical_block",
        )
        q_capacity = (
            max_sequence_tokens + tokens_per_logical_block - 1
        ) // tokens_per_logical_block
        row_stride_cells = n_layers * q_capacity
        directory_cell_count = raw_row_capacity * row_stride_cells
        if directory_cell_count > _INT32_MAX:
            raise ValueError(
                "directory_cell_count exceeds int32 range: "
                f"value={directory_cell_count}"
            )
        return cls(
            raw_row_capacity=raw_row_capacity,
            n_layers=n_layers,
            q_capacity=q_capacity,
            row_stride_cells=row_stride_cells,
            directory_cell_count=directory_cell_count,
        )

    def flat_index(
        self,
        *,
        pool_row: int,
        local_layer_ordinal: int,
        block_q: int,
    ) -> int:
        pool_row = int(pool_row)
        local_layer_ordinal = int(local_layer_ordinal)
        block_q = int(block_q)
        if not 0 <= pool_row < self.raw_row_capacity:
            raise IndexError(
                f"pool_row={pool_row} is outside "
                f"[0, {self.raw_row_capacity})"
            )
        if not 0 <= local_layer_ordinal < self.n_layers:
            raise IndexError(
                f"local_layer_ordinal={local_layer_ordinal} is outside "
                f"[0, {self.n_layers})"
            )
        if not 0 <= block_q < self.q_capacity:
            raise IndexError(
                f"block_q={block_q} is outside [0, {self.q_capacity})"
            )
        return (
            pool_row * self.row_stride_cells
            + local_layer_ordinal * self.q_capacity
            + block_q
        )


class DDRBlockTable:
    """Fixed-size tensor table containing address, length, and valid arrays."""

    def __init__(
        self,
        cell_count: int,
        *,
        device: torch.device | str,
    ) -> None:
        cell_count = int(cell_count)
        if cell_count <= 0:
            raise ValueError(f"cell_count must be positive, got {cell_count}")
        self.addresses = torch.zeros(
            cell_count,
            dtype=torch.uint64,
            device=device,
        )
        self.lengths = torch.zeros(
            cell_count,
            dtype=torch.uint64,
            device=device,
        )
        self.valid = torch.zeros(
            cell_count,
            dtype=torch.uint8,
            device=device,
        )


__all__ = [
    "DDRBlockTable",
    "DsaRowIndexGeometry",
    "M0_DDR_LAYER_ALIGNMENT_BYTES",
    "M0_LAYOUT_ID",
    "PackedKvLayout",
]
