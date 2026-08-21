# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest

from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCachePlanner,
    DSARequestCacheStage,
)


@dataclass
class _Request:
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int
    num_output_tokens: int
    num_tokens: int


def _make_planner() -> DSARequestCachePlanner:
    return DSARequestCachePlanner(
        block_size=128,
        sparse_activation_tokens=2048,
        prompt_budget_thresholds=(4096, 8192),
        resident_budget_tokens=(2048, 4096, 6144),
    )


def test_budget_is_selected_from_prompt_and_frozen_on_first_commit() -> None:
    planner = _make_planner()
    request = _Request("req", 5000, 0, 0, 5000)

    prefill = planner.plan(
        request,
        num_new_tokens=5000,
        max_model_len=16384,
    )
    assert prefill.stage == DSARequestCacheStage.PREFILL
    assert prefill.target_resident_budget_tokens == 4096
    planner.commit(prefill)

    request.num_prompt_tokens = 9000
    request.num_computed_tokens = 9000
    request.num_output_tokens = 1
    request.num_tokens = 9001
    decode = planner.plan(
        request,
        num_new_tokens=1,
        max_model_len=16384,
    )
    assert decode.target_resident_budget_tokens == 4096


def test_long_prompt_transitions_prefill_enter_then_sparse() -> None:
    planner = _make_planner()
    request = _Request("req", 5000, 0, 0, 5000)
    prefill = planner.plan(
        request,
        num_new_tokens=5000,
        max_model_len=16384,
    )
    planner.commit(prefill)

    request.num_computed_tokens = 5000
    request.num_output_tokens = 1
    request.num_tokens = 5001
    enter = planner.plan(
        request,
        num_new_tokens=1,
        max_model_len=16384,
    )
    assert enter.stage == DSARequestCacheStage.ENTER_SPARSE_DECODE
    assert enter.sparse_budget_tokens == 4096
    assert enter.tail_tokens == 9
    assert enter.resident_valid_tokens == 4105
    assert enter.replace_resident_blocks
    assert enter.preserve_resident_tail_block
    planner.commit(enter)

    request.num_computed_tokens = 5001
    request.num_output_tokens = 2
    request.num_tokens = 5002
    sparse = planner.plan(
        request,
        num_new_tokens=1,
        max_model_len=16384,
    )
    assert sparse.stage == DSARequestCacheStage.SPARSE_DECODE
    assert sparse.resident_valid_tokens == 4106
    assert not sparse.replace_resident_blocks


def test_chunked_prefill_stays_prefill_until_the_next_decode_step() -> None:
    planner = _make_planner()
    request = _Request("req", 5000, 0, 0, 5000)

    for computed, chunk_size in (
        (0, 2048),
        (2048, 2048),
        (4096, 904),
    ):
        request.num_computed_tokens = computed
        chunk = planner.plan(
            request,
            num_new_tokens=chunk_size,
            max_model_len=16384,
        )
        assert chunk.stage == DSARequestCacheStage.PREFILL
        assert chunk.indexer_tokens_need_slot == computed + chunk_size
        planner.commit(chunk)
        request.num_computed_tokens = computed + chunk_size
        assert (
            planner.should_release_resident_after_prefill(request)
            == (request.num_computed_tokens == request.num_prompt_tokens)
        )

    request.num_computed_tokens = request.num_prompt_tokens
    request.num_output_tokens = 1
    request.num_tokens = request.num_prompt_tokens + 1
    first_decode = planner.plan(
        request,
        num_new_tokens=1,
        max_model_len=16384,
    )
    assert first_decode.stage == DSARequestCacheStage.ENTER_SPARSE_DECODE


def test_short_prompt_bypasses_dsa_after_crossing_sparse_boundary() -> None:
    planner = _make_planner()
    request = _Request("req", 1000, 0, 0, 1000)
    prefill = planner.plan(
        request,
        num_new_tokens=1000,
        max_model_len=16384,
    )
    planner.commit(prefill)

    request.num_computed_tokens = 2047
    request.num_output_tokens = 1048
    request.num_tokens = 2048
    bypass = planner.plan(
        request,
        num_new_tokens=1,
        max_model_len=16384,
    )
    assert bypass.stage == DSARequestCacheStage.BYPASS
    planner.commit(bypass)

    request.num_computed_tokens = 2048
    request.num_output_tokens = 1049
    request.num_tokens = 2049
    still_bypass = planner.plan(
        request,
        num_new_tokens=1,
        max_model_len=16384,
    )
    assert still_bypass.stage == DSARequestCacheStage.BYPASS
    assert still_bypass.tail_tokens == 0
    assert still_bypass.resident_valid_tokens == -1


def test_plan_does_not_mutate_state_until_commit() -> None:
    planner = _make_planner()
    request = _Request("req", 3000, 0, 0, 3000)

    plan = planner.plan(
        request,
        num_new_tokens=3000,
        max_model_len=16384,
    )
    assert planner.get_state(request.request_id) is None

    planner.commit(plan)
    state = planner.get_state(request.request_id)
    assert state is not None
    assert state.stage == plan.stage
    assert state.target_resident_budget_tokens == plan.target_resident_budget_tokens


def test_commit_reuses_the_slotted_state_object_across_decode_steps() -> None:
    planner = _make_planner()
    request = _Request("req", 3000, 0, 0, 3000)
    planner.commit(
        planner.plan(
            request,
            num_new_tokens=3000,
            max_model_len=16384,
        )
    )
    state = planner.get_state(request.request_id)
    assert state is not None
    assert not hasattr(state, "__dict__")

    request.num_computed_tokens = 3000
    request.num_output_tokens = 1
    request.num_tokens = 3001
    planner.commit(
        planner.plan(
            request,
            num_new_tokens=1,
            max_model_len=16384,
        )
    )
    assert planner.get_state(request.request_id) is state


def test_prefill_release_is_idempotent_and_free_clears_state() -> None:
    planner = _make_planner()
    request = _Request("req", 3000, 0, 0, 3000)
    prefill = planner.plan(
        request,
        num_new_tokens=3000,
        max_model_len=16384,
    )
    planner.commit(prefill)

    request.num_computed_tokens = request.num_prompt_tokens
    assert planner.should_release_resident_after_prefill(request)
    planner.mark_prefill_resident_released(request.request_id)
    assert not planner.should_release_resident_after_prefill(request)

    planner.free(request.request_id)
    assert planner.get_state(request.request_id) is None


def test_sparse_request_cannot_move_back_to_dense() -> None:
    planner = _make_planner()
    request = _Request("req", 3000, 3000, 1, 3001)
    enter = planner.plan(
        request,
        num_new_tokens=1,
        max_model_len=16384,
    )
    planner.commit(enter)

    request.num_computed_tokens = 1500
    request.num_output_tokens = 2
    request.num_tokens = 1501
    with pytest.raises(RuntimeError, match="cannot move"):
        planner.plan(
            request,
            num_new_tokens=1,
            max_model_len=16384,
        )


def test_keep_decode_kv_in_hbm_freezes_candidate_at_prompt_suffix() -> None:
    planner = DSARequestCachePlanner(
        block_size=128,
        sparse_activation_tokens=6144,
        prompt_budget_thresholds=(32768, 65536),
        resident_budget_tokens=(6144, 10240, 12288),
        keep_decode_kv_in_hbm=True,
    )
    request = _Request("req", 6145, 0, 0, 6145)
    planner.commit(
        planner.plan(
            request,
            num_new_tokens=request.num_prompt_tokens,
            max_model_len=16384,
        )
    )

    for logical_tokens in (6146, 6272, 6273, 6400, 8192):
        request.num_computed_tokens = logical_tokens - 1
        request.num_output_tokens = logical_tokens - request.num_prompt_tokens
        request.num_tokens = logical_tokens
        plan = planner.plan(
            request,
            num_new_tokens=1,
            max_model_len=16384,
        )

        assert plan.stage.is_sparse_decode
        assert plan.sparse_budget_tokens == 6144
        assert plan.persistent_suffix_start_tokens == 6144
        assert plan.persistent_suffix_valid_tokens == logical_tokens - 6144
        assert plan.resident_valid_tokens == logical_tokens
        planner.commit(plan)


def test_keep_decode_kv_in_hbm_requires_prefix_to_cover_sparse_budget() -> None:
    planner = DSARequestCachePlanner(
        block_size=128,
        sparse_activation_tokens=2048,
        prompt_budget_thresholds=(4096,),
        resident_budget_tokens=(4096, 4096),
        keep_decode_kv_in_hbm=True,
    )
    request = _Request("req", 2050, 2050, 1, 4097)

    plan = planner.plan(
        request,
        num_new_tokens=1,
        max_model_len=16384,
    )

    assert plan.stage == DSARequestCacheStage.DENSE_DECODE
    assert plan.sparse_budget_tokens == 0
    assert plan.resident_valid_tokens == -1


def test_keep_decode_kv_in_hbm_sizes_suffix_for_multi_token_forward() -> None:
    planner = DSARequestCachePlanner(
        block_size=128,
        sparse_activation_tokens=2048,
        prompt_budget_thresholds=(4096, 8192),
        resident_budget_tokens=(2048, 4096, 6144),
        keep_decode_kv_in_hbm=True,
    )
    request = _Request("req", 5000, 5000, 1, 5001)

    plan = planner.plan(
        request,
        num_new_tokens=4,
        max_model_len=16384,
    )

    assert plan.stage == DSARequestCacheStage.ENTER_SPARSE_DECODE
    assert plan.persistent_suffix_start_tokens == 4992
    assert plan.persistent_suffix_valid_tokens == 12
    assert plan.resident_valid_tokens == 4108
    assert plan.preserve_resident_tail_block


def test_keep_decode_kv_in_hbm_preserves_prompt_tail_owner() -> None:
    planner = DSARequestCachePlanner(
        block_size=128,
        sparse_activation_tokens=2048,
        prompt_budget_thresholds=(),
        resident_budget_tokens=(4096,),
        keep_decode_kv_in_hbm=True,
    )
    request = _Request("req", 5001, 5120, 120, 5121)

    plan = planner.plan(
        request,
        num_new_tokens=1,
        max_model_len=16384,
    )

    assert plan.stage == DSARequestCacheStage.ENTER_SPARSE_DECODE
    assert request.num_computed_tokens % 128 == 0
    assert request.num_prompt_tokens % 128 != 0
    assert plan.preserve_resident_tail_block
