# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_ascend.dsa_offload.kvaf_adapter import KvafAdapter


class _FakeKvafRuntime:
    initialized = True
    remote_region = {"slot_size": 4096}

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def batch_free(self, keys):
        self.calls.append(("free", list(keys)))


def test_batch_free_forwards_flat_indices_without_row_state() -> None:
    runtime = _FakeKvafRuntime()
    adapter = KvafAdapter(runtime=runtime)

    adapter.batch_free(flat_indices=[8, 7, 8])

    assert runtime.calls == [("free", [8, 7, 8])]


def test_batch_free_skips_empty_rows() -> None:
    runtime = _FakeKvafRuntime()
    adapter = KvafAdapter(runtime=runtime)

    adapter.batch_free(flat_indices=[])

    assert runtime.calls == []


def test_batch_free_requires_kvaf_api() -> None:
    class _Runtime:
        initialized = True
        remote_region = {"slot_size": 4096}

    adapter = KvafAdapter(runtime=_Runtime())

    with pytest.raises(RuntimeError, match="batch_free"):
        adapter.batch_free(flat_indices=[1])
