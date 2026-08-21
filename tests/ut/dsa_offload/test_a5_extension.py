# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend A5 C8 custom-op ABI regressions."""

from __future__ import annotations

import pytest


def test_c8_qsfa_uses_repository_resident_kernel(monkeypatch) -> None:
    """Resident C8 attention must use the repository-pinned QSFA operator."""

    pytest.importorskip("torch_npu")
    import torch

    from vllm_ascend.dsa_offload import a5_extension

    captured: dict[str, object] = {}

    def fake_qsfa(
        query: torch.Tensor,
        packed_kv: torch.Tensor,
        slots: torch.Tensor,
        scale_value: float,
        block_table: torch.Tensor,
        actual_query_lengths: torch.Tensor,
        resident_seq_lengths: torch.Tensor,
    ) -> torch.Tensor:
        captured.update(
            query=query,
            packed_kv=packed_kv,
            slots=slots,
            scale_value=scale_value,
            block_table=block_table,
            actual_query_lengths=actual_query_lengths,
            resident_seq_lengths=resident_seq_lengths,
        )
        return torch.empty(
            (*query.shape[:-1], 512),
            dtype=query.dtype,
            device=query.device,
        )

    monkeypatch.setattr(a5_extension, "_resident_c8_qsfa", lambda: fake_qsfa)
    device = torch.device("npu")
    query_storage = torch.zeros(
        (1, 4, 1152), dtype=torch.bfloat16, device=device
    )
    query = query_storage[..., ::2]
    assert not query.is_contiguous()
    packed_kv = torch.zeros(
        (1, 128, 1, 656), dtype=torch.int8, device=device
    )
    slots = torch.zeros((1, 1, 2048), dtype=torch.int32, device=device)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    actual_query_lengths = torch.ones((1,), dtype=torch.int32, device=device)
    resident_seq_lengths = torch.ones((1,), dtype=torch.int32, device=device)

    output = a5_extension._sparse_and_tail_attention_c8(
        query,
        packed_kv,
        slots,
        block_table,
        actual_query_lengths,
        resident_seq_lengths,
        1.0,
    )

    assert output.shape == (1, 4, 512)
    captured_query = captured["query"]
    assert isinstance(captured_query, torch.Tensor)
    assert captured_query.is_contiguous()
    assert captured["packed_kv"] is packed_kv
    assert captured["slots"] is slots
    assert captured["scale_value"] == 1.0
    assert captured["block_table"] is block_table
    assert captured["actual_query_lengths"] is actual_query_lengths
    assert captured["resident_seq_lengths"] is resident_seq_lengths
