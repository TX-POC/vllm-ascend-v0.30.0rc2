# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.dsa_offload.dsa_row import (
    DDRBlockTable,
    DsaRowIndexGeometry,
    PackedKvLayout,
)
from vllm_ascend.dsa_offload.kvaf_adapter import KvafAdapter


def _layout() -> PackedKvLayout:
    return PackedKvLayout(
        layout_id="A5_C8_W4A4C8_V1",
        tokens_per_logical_block=128,
        packed_token_bytes=656,
        ddr_layer_alignment_bytes=4096,
    )


def _geometry() -> DsaRowIndexGeometry:
    return DsaRowIndexGeometry.from_runtime(
        raw_row_capacity=2,
        n_layers=3,
        max_sequence_tokens=512,
        tokens_per_logical_block=128,
    )


def _make_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[KvafAdapter, SimpleNamespace]:
    runtime = SimpleNamespace(
        initialized=False,
        remote_region={},
        batch_get=Mock(),
    )

    def initialize(ddr_bytes: int, slot_size: int) -> None:
        runtime.initialized = True
        runtime.remote_region = {
            "length": ddr_bytes,
            "slot_size": slot_size,
        }

    runtime.initialize = initialize
    runtime.link_to = Mock()
    monkeypatch.setitem(
        sys.modules,
        "kvaf",
        SimpleNamespace(NpuRuntime=lambda *_args, **_kwargs: runtime),
    )
    monkeypatch.setenv(
        "VLLM_ASCEND_KVAF_BATCH_GET_KERNEL",
        "/tmp/kvaf_batch_get.o",
    )
    monkeypatch.setenv(
        "VLLM_ASCEND_KVAF_BATCH_PUT_KERNEL",
        "/tmp/kvaf_batch_put.o",
    )
    monkeypatch.setenv("VLLM_ASCEND_KVAF_NPU_CORE_COUNT", "1")
    monkeypatch.setenv("VLLM_ASCEND_KVAF_JETTYS_PER_CORE", "2")
    monkeypatch.setenv("VLLM_ASCEND_KVAF_TRANSPORT", "remote")
    monkeypatch.setenv(
        "VLLM_ASCEND_KVAF_CPU_SERVICE_ADDR",
        "192.0.2.1:28999",
    )
    monkeypatch.setattr(torch.npu, "set_device", lambda _device_id: None)
    adapter = KvafAdapter(device_id=0)
    runtime.link_to.assert_called_once_with("192.0.2.1:28999")
    adapter.initialize_dsa_row_index(
        layout=_layout(),
        ddr_bytes=1 << 20,
        geometry=_geometry(),
        device=torch.device("cpu"),
    )
    return adapter, runtime


def test_kvaf_adapter_allocates_fixed_abi_buffers_and_flattens_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = _make_adapter(monkeypatch)
    buffers = adapter.batch_get_buffers
    assert buffers is not None
    assert buffers.indices.dtype == torch.int32
    assert buffers.offsets.dtype == torch.int32
    assert buffers.mask.dtype == torch.uint8
    assert buffers.output_indices.dtype == torch.int32

    packed = torch.empty((7, 128, 1, 656), dtype=torch.int8)
    rows = adapter._packed_output_rows(packed)
    assert rows.shape == (7 * 128, 656)
    assert rows.data_ptr() == packed.data_ptr()


def test_kvaf_adapter_rejects_noncanonical_output() -> None:
    bad = torch.empty((7, 128, 656), dtype=torch.int8)
    try:
        KvafAdapter._packed_output_rows(bad)
    except ValueError as error:
        assert "[physical_blocks,128,1,656]" in str(error)
    else:
        raise AssertionError("noncanonical packed cache was accepted")


def test_kvaf_adapter_uses_runtime_directory_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, kvaf_runtime = _make_adapter(monkeypatch)
    directory = DDRBlockTable(24, device=torch.device("cpu"))
    prepare = Mock()
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.dsa_offload.ops_nanovllm",
        SimpleNamespace(prepare_kvaf_batch_get_inputs=prepare),
    )
    stream = object()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda: stream,
            stream=lambda _: nullcontext(),
        ),
        raising=False,
    )

    topk = torch.zeros((1, 1, 2048), dtype=torch.int32)
    row = torch.zeros(1, dtype=torch.int32)
    adapter.execute_batch_get(
        layer_id=1,
        source_token_ids=topk,
        destination_slots=topk,
        miss_count=row,
        resident_pool_indices=row,
        resident_block_table=torch.zeros((1, 16), dtype=torch.int32),
        cache_tokens=row,
        candidate_lens=row,
        logical_seq_lengths=row,
        resident_seq_lengths=row,
        address_table=directory.addresses,
        valid_table=directory.valid,
        output=torch.empty((1, 128, 1, 656), dtype=torch.int8),
    )

    prepare.assert_called_once()
    assert kvaf_runtime.batch_get.call_args.args[0] is directory.addresses
    assert kvaf_runtime.batch_get.call_args.args[1] is directory.valid
