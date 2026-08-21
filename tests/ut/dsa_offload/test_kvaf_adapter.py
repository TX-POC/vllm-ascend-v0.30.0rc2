# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.dsa_offload.dsa_row import (
    DDRBlockTable,
    DsaRowIndexGeometry,
    PackedKvLayout,
)
from vllm_ascend.dsa_offload.kvaf_adapter import (
    KvafAdapter,
)


class _FakeKvafRuntime:
    def __init__(self, *, fail_batch_put_call: int | None = None) -> None:
        self.initialized = False
        self.remote_region: dict[str, int] = {}
        self.fail_batch_put_call = fail_batch_put_call
        self.batch_put_calls: list[dict[str, object]] = []
        self.batch_get_calls: list[tuple[object, ...]] = []
        self.batch_free_calls: list[list[int]] = []
        self.released_handles: list[tuple[int, ...]] = []
        self.committed: dict[int, tuple[int, int]] = {}

    def initialize(self, ddr_bytes: int, slot_size: int) -> None:
        self.initialized = True
        self.remote_region = {
            "base": 4096,
            "length": ddr_bytes,
            "slot_size": slot_size,
        }

    def batch_put(
        self,
        keys: list[int],
        source: torch.Tensor,
        input_offsets: list[int],
        lengths: list[int],
        stream: object,
    ) -> None:
        call_index = len(self.batch_put_calls) + 1
        if self.fail_batch_put_call == call_index:
            raise RuntimeError("configured BatchPut failure")
        self.batch_put_calls.append(
            {
                "keys": keys,
                "source": source,
                "input_offsets": input_offsets,
                "lengths": lengths,
                "stream": stream,
            }
        )
        for key, length in zip(keys, lengths):
            self.committed[key] = (0x100000 + key * 0x1000, length)

    def batch_get(self, *args: object) -> None:
        self.batch_get_calls.append(args)

    def prepare_access_table(
        self,
        keys: torch.Tensor,
    ) -> tuple[int, ...]:
        handle = tuple(int(key) for key in keys.tolist())
        if any(key not in self.committed for key in handle):
            raise KeyError("missing key")
        return handle

    def access_table_view(
        self,
        handle: tuple[int, ...],
        *,
        device: str,
        capacity: int = 0,
    ) -> object:
        addresses = [self.committed[key][0] for key in handle]
        lengths = [self.committed[key][1] for key in handle]
        valid = [1] * len(handle)
        if device == "cpu":
            return {
                "addresses": np.asarray(addresses, dtype=np.uint64),
                "lengths": np.asarray(lengths, dtype=np.uint64),
                "valid": np.asarray(valid, dtype=np.uint8),
            }
        assert device == "npu"
        assert capacity == len(handle)
        return SimpleNamespace(
            addresses=torch.tensor(addresses, dtype=torch.int64),
            valid=torch.tensor(valid, dtype=torch.uint8),
        )

    def release_access_table(self, handle: tuple[int, ...]) -> None:
        self.released_handles.append(handle)

    def batch_free(self, keys: torch.Tensor) -> None:
        copied = [int(key) for key in keys.tolist()]
        self.batch_free_calls.append(copied)
        for key in copied:
            self.committed.pop(key, None)

    def shutdown(self) -> None:
        self.initialized = False


def _make_adapter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_batch_put_call: int | None = None,
) -> tuple[
    KvafAdapter,
    _FakeKvafRuntime,
    DDRBlockTable,
    DsaRowIndexGeometry,
]:
    runtime = _FakeKvafRuntime(
        fail_batch_put_call=fail_batch_put_call,
    )
    constructor_args = []

    def npu_runtime(*args):
        constructor_args.append(args)
        return runtime

    monkeypatch.setitem(
        __import__("sys").modules,
        "kvaf",
        SimpleNamespace(NpuRuntime=npu_runtime),
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
    monkeypatch.setattr(torch.npu, "set_device", lambda _device_id: None)
    adapter = KvafAdapter(device_id=3)
    assert constructor_args == [
        (
            "/tmp/kvaf_batch_get.o",
            3,
            1,
            "/tmp/kvaf_batch_put.o",
            2,
        )
    ]
    layout = PackedKvLayout(
        layout_id="test",
        tokens_per_logical_block=2,
        packed_token_bytes=4,
        ddr_layer_alignment_bytes=8,
    )
    geometry = DsaRowIndexGeometry.from_runtime(
        raw_row_capacity=2,
        n_layers=2,
        max_sequence_tokens=8,
        tokens_per_logical_block=2,
    )
    block_table = DDRBlockTable(
        geometry.directory_cell_count,
        device=torch.device("cpu"),
    )
    adapter.initialize_dsa_row_index(
        layout=layout,
        ddr_bytes=256,
        geometry=geometry,
        device=torch.device("cpu"),
    )
    return (
        adapter,
        runtime,
        block_table,
        geometry,
    )


def _jobs() -> dict[str, np.ndarray]:
    return {
        "pool_rows": np.asarray([0, 0]),
        "block_qs": np.asarray([0, 1]),
        "source_block_ids": np.asarray([2, 3]),
    }


def test_batch_put_publishes_all_layers_into_fixed_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, runtime, block_table, geometry = _make_adapter(monkeypatch)
    packed_cache = torch.arange(64, dtype=torch.uint8).view(8, 2, 4)
    synchronized_devices: list[torch.device] = []
    monkeypatch.setattr(
        adapter,
        "_synchronize_device",
        synchronized_devices.append,
    )

    adapter.begin_batch_put(**_jobs(), geometry=geometry)
    adapter.write_batch_put_layer(
        local_layer_ordinal=0,
        packed_cache=packed_cache,
    )
    adapter.write_batch_put_layer(
        local_layer_ordinal=1,
        packed_cache=packed_cache,
    )
    updated = adapter.finalize_batch_put(block_table=block_table)

    assert [call["keys"] for call in runtime.batch_put_calls] == [
        [0, 1],
        [4, 5],
    ]
    assert runtime.batch_put_calls[0]["input_offsets"] == [16, 24]
    assert runtime.batch_put_calls[0]["lengths"] == [8, 8]
    source = runtime.batch_put_calls[0]["source"]
    assert isinstance(source, torch.Tensor)
    assert source.shape == (4, 2, 4)
    assert source.data_ptr() == packed_cache.data_ptr()
    assert (
        source.untyped_storage().data_ptr()
        == packed_cache.untyped_storage().data_ptr()
    )
    assert updated == {0: 2}
    assert adapter.committed_block_count(0) == 2
    assert block_table.valid[[0, 1, 4, 5]].tolist() == [1, 1, 1, 1]
    assert block_table.lengths[[0, 1, 4, 5]].tolist() == [8, 8, 8, 8]
    assert block_table.addresses[0].item() == 0x100000
    assert runtime.released_handles == [(0, 1, 4, 5)]
    assert synchronized_devices == []


def test_failed_layer_is_freed_without_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, runtime, block_table, geometry = _make_adapter(
        monkeypatch,
        fail_batch_put_call=2,
    )
    packed_cache = torch.zeros((4, 2, 4), dtype=torch.uint8)

    adapter.begin_batch_put(**_jobs(), geometry=geometry)
    adapter.write_batch_put_layer(
        local_layer_ordinal=0,
        packed_cache=packed_cache,
    )
    with pytest.raises(RuntimeError, match="configured"):
        adapter.write_batch_put_layer(
            local_layer_ordinal=1,
            packed_cache=packed_cache,
        )
    adapter.abort_batch_put(block_table=block_table)

    assert runtime.batch_free_calls == [[0, 1]]
    assert not runtime.committed
    assert not block_table.valid.any()
    assert not block_table.addresses.any()
    assert adapter.committed_block_count(0) == 0


def test_release_pool_row_frees_every_layer_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, runtime, block_table, geometry = _make_adapter(monkeypatch)
    packed_cache = torch.zeros((4, 2, 4), dtype=torch.uint8)
    adapter.begin_batch_put(**_jobs(), geometry=geometry)
    for layer_id in range(2):
        adapter.write_batch_put_layer(
            local_layer_ordinal=layer_id,
            packed_cache=packed_cache,
        )
    adapter.finalize_batch_put(block_table=block_table)

    adapter.release_pool_row(0, block_table=block_table)

    assert runtime.batch_free_calls == [[0, 1, 4, 5]]
    assert adapter.committed_block_count(0) == 0
    assert not block_table.valid.any()
    assert not block_table.addresses.any()
