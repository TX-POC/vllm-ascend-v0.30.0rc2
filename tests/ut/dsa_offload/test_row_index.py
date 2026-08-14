# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_ascend.dsa_offload.dsa_row import (
    DsaRowIndexGeometry,
    PackedKvLayout,
)


def test_row_geometry_matches_design_flat_index_oracle() -> None:
    geometry = DsaRowIndexGeometry.from_runtime(
        raw_row_capacity=4,
        n_layers=3,
        max_sequence_tokens=1024,
        tokens_per_logical_block=128,
    )

    assert geometry.q_capacity == 8
    assert geometry.row_stride_cells == 24
    assert geometry.directory_cell_count == 96
    assert geometry.flat_index(
        pool_row=2,
        local_layer_ordinal=1,
        block_q=3,
    ) == 59


def test_flat_index_rejects_padding_row() -> None:
    geometry = DsaRowIndexGeometry.from_runtime(
        raw_row_capacity=2,
        n_layers=2,
        max_sequence_tokens=256,
        tokens_per_logical_block=128,
    )

    assert geometry.flat_index(
        pool_row=0,
        local_layer_ordinal=1,
        block_q=0,
    ) == 2
    assert geometry.flat_index(
        pool_row=1,
        local_layer_ordinal=1,
        block_q=1,
    ) == 7

    with pytest.raises(IndexError, match="outside"):
        geometry.flat_index(
            pool_row=2,
            local_layer_ordinal=0,
            block_q=0,
        )


def test_packed_layout_is_checked() -> None:
    layout = PackedKvLayout(
        layout_id="A5_C8_W4A4C8_V1",
        tokens_per_logical_block=128,
        packed_token_bytes=656,
        ddr_layer_alignment_bytes=4096,
    )
    assert layout.tokens_per_logical_block == 128
    assert layout.packed_token_bytes == 656


def test_directory_cell_count_is_limited_to_int32() -> None:
    with pytest.raises(ValueError, match="directory_cell_count"):
        DsaRowIndexGeometry.from_runtime(
            raw_row_capacity=1 << 20,
            n_layers=256,
            max_sequence_tokens=1 << 20,
            tokens_per_logical_block=128,
        )
