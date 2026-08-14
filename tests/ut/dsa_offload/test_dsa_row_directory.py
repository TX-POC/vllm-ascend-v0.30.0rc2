# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.dsa_offload.dsa_row import (
    DDRBlockTable,
    DsaRowIndexGeometry,
)


def test_directory_initializes_fixed_address_length_valid_tables() -> None:
    geometry = DsaRowIndexGeometry.from_runtime(
        raw_row_capacity=2,
        n_layers=2,
        max_sequence_tokens=256,
        tokens_per_logical_block=128,
    )
    table = DDRBlockTable(
        geometry.directory_cell_count,
        device=torch.device("cpu"),
    )

    assert isinstance(table.addresses, torch.Tensor)
    assert table.addresses.numel() == 8
    assert table.addresses.device == torch.device("cpu")
    assert table.addresses.dtype == torch.uint64
    assert table.lengths.dtype == torch.uint64
    assert table.valid.dtype == torch.uint8
    assert table.addresses.tolist() == [0] * 8
    assert table.lengths.tolist() == [0] * 8
    assert table.valid.tolist() == [0] * 8
