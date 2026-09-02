# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.dsa_offload.contracts import (
    DSA_OP_BACKEND_NANOVLLM_A5,
    DSA_SFA_COMPUTE_TOPK,
)
from vllm_ascend.dsa_offload.resident_pool import DSAResidentTokenPool


def _make_pool() -> DSAResidentTokenPool:
    return DSAResidentTokenPool(
        max_num_reqs=2,
        num_layers=3,
        max_model_len=8192,
        max_resident_budget_tokens=4096,
        device=torch.device("cpu"),
    )


def _make_a5_pool(
    *,
    reserve_speculative_resident_slots: bool = False,
    max_resident_budget_tokens: int = 6144,
) -> DSAResidentTokenPool:
    return DSAResidentTokenPool(
        max_num_reqs=2,
        num_layers=3,
        max_model_len=8192,
        max_resident_budget_tokens=max_resident_budget_tokens,
        device=torch.device("cpu"),
        op_backend=DSA_OP_BACKEND_NANOVLLM_A5,
        reserve_speculative_resident_slots=reserve_speculative_resident_slots,
    )


def test_pool_row_is_stable_until_request_release() -> None:
    pool = _make_pool()

    first = pool.acquire("req-a")
    assert pool.acquire("req-a") == first
    second = pool.acquire("req-b")
    assert second != first

    pool.release("req-a")
    assert pool.get_index("req-a") is None
    # release 不再重复清大状态行；复用前 acquire 会完成唯一一次清理。
    pool.get_cache_slots(0)[first, 0] = 123
    assert pool.acquire("req-c") == first
    assert pool.get_cache_slots(0)[first, 0].item() == -1


def test_enter_marks_every_layer_for_first_fill_once() -> None:
    pool = _make_pool()
    row = pool.acquire("req-a")

    pool.prepare_sparse_request(
        "req-a",
        target_budget_tokens=4096,
    )

    for layer_id in range(pool.num_layers):
        cache_slots = pool.get_cache_slots(layer_id)
        assert cache_slots[row, pool.cache_metadata_index].item() == -4096

    # ENTER 重复投影只做幂等核验，不覆盖 LIDU 已经原址刷新的正预算。
    pool.get_cache_slots(0)[row, pool.cache_metadata_index] = 4096
    pool.prepare_sparse_request(
        "req-a",
        target_budget_tokens=4096,
    )
    assert (
        pool.get_cache_slots(0)[row, pool.cache_metadata_index].item()
        == 4096
    )


def test_padding_row_is_not_allocatable() -> None:
    pool = _make_pool()

    assert pool.padding_pool_index == pool.max_num_reqs
    assert pool.acquire("req-a") != pool.padding_pool_index
    assert pool.acquire("req-b") != pool.padding_pool_index


def test_graph_capture_first_fill_is_cleared_after_dummy_run() -> None:
    pool = _make_pool()

    pool.prepare_graph_capture(
        row_count=2,
        target_budget_tokens=4096,
    )

    for layer_id in range(pool.num_layers):
        cache_slots = pool.get_cache_slots(layer_id)
        assert cache_slots[
            :2,
            pool.cache_metadata_index,
        ].tolist() == [-4096, -4096]

    pool.restore_after_graph_capture()

    for layer_id in range(pool.num_layers):
        cache_slots = pool.get_cache_slots(layer_id)
        assert cache_slots[
            :2,
            :pool.cache_metadata_index,
        ].eq(-1).all()
        assert cache_slots[
            :2,
            pool.cache_metadata_index,
        ].tolist() == [0, 0]


def test_graph_capture_reuses_one_resident_owner_across_sizes() -> None:
    pool = _make_pool()
    cache_slots_ptr = pool.get_cache_slots(0).data_ptr()

    for row_count in (1, 2, 1):
        pool.prepare_graph_capture(
            row_count=row_count,
            target_budget_tokens=4096,
        )
        assert pool.get_cache_slots(0).data_ptr() == cache_slots_ptr
        for layer_id in range(pool.num_layers):
            assert pool.get_cache_slots(layer_id)[
                :row_count,
                pool.cache_metadata_index,
            ].tolist() == [-4096] * row_count

        pool.restore_after_graph_capture()
        assert pool.graph_capture_row_count == 0
        for layer_id in range(pool.num_layers):
            assert pool.get_cache_slots(layer_id)[
                :,
                pool.cache_metadata_index,
            ].eq(0).all()


def test_a5_speculative_resident_range_is_disabled_by_default() -> None:
    pool = _make_a5_pool()
    row = pool.acquire("req-a")
    pool.prepare_sparse_request("req-a", target_budget_tokens=6144)

    assert pool.speculative_resident_slot_count == 0
    for layer_id in range(pool.num_layers):
        assert pool.get_speculative_resident_slots(layer_id) is None
        assert pool.get_cache_slots(layer_id)[row, :6144].tolist() == list(
            range(6144)
        )


def test_a5_speculative_resident_range_uses_only_high_budget_slots() -> None:
    pool = _make_a5_pool(reserve_speculative_resident_slots=True)
    row = pool.acquire("req-a")
    budget = 6144
    normal_budget = budget - DSA_SFA_COMPUTE_TOPK

    pool.prepare_sparse_request("req-a", target_budget_tokens=budget)

    assert pool.speculative_resident_slot_count == DSA_SFA_COMPUTE_TOPK
    expected_range = list(range(normal_budget, budget))
    for layer_id in range(pool.num_layers):
        cache_slots = pool.get_cache_slots(layer_id)
        speculative_slots = pool.get_speculative_resident_slots(layer_id)
        assert speculative_slots is not None
        assert cache_slots[row, :normal_budget].tolist() == list(
            range(normal_budget)
        )
        # The canonical speculative range is deliberately absent from the
        # normal source-to-slot mapping until the fused operator publishes a
        # final winner into it.
        assert cache_slots[row, normal_budget:budget].eq(-1).all()
        assert speculative_slots[row].tolist() == expected_range


def test_a5_speculative_resident_range_is_recreated_for_graph_capture() -> None:
    pool = _make_a5_pool(reserve_speculative_resident_slots=True)
    budget = 6144
    normal_budget = budget - DSA_SFA_COMPUTE_TOPK

    pool.prepare_graph_capture(row_count=2, target_budget_tokens=budget)

    for layer_id in range(pool.num_layers):
        cache_slots = pool.get_cache_slots(layer_id)
        speculative_slots = pool.get_speculative_resident_slots(layer_id)
        assert speculative_slots is not None
        assert cache_slots[:2, :normal_budget].eq(
            torch.arange(normal_budget, dtype=torch.int32)
        ).all()
        assert cache_slots[:2, normal_budget:budget].eq(-1).all()
        assert speculative_slots[:2].eq(
            torch.arange(normal_budget, budget, dtype=torch.int32)
        ).all()

    pool.restore_after_graph_capture()

    for layer_id in range(pool.num_layers):
        speculative_slots = pool.get_speculative_resident_slots(layer_id)
        assert speculative_slots is not None
        assert speculative_slots[:2].eq(-1).all()


def test_speculative_resident_range_requires_a5_and_a_full_topk() -> None:
    with pytest.raises(ValueError, match="nanovllm_a5"):
        DSAResidentTokenPool(
            max_num_reqs=1,
            num_layers=1,
            max_model_len=4096,
            max_resident_budget_tokens=4096,
            device=torch.device("cpu"),
            reserve_speculative_resident_slots=True,
        )

    with pytest.raises(ValueError, match="capacity cannot reserve"):
        _make_a5_pool(
            reserve_speculative_resident_slots=True,
            max_resident_budget_tokens=2 * DSA_SFA_COMPUTE_TOPK - 1,
        )
