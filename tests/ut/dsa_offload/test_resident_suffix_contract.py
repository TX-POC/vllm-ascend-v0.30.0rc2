# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only mathematical contract for DSA's resident decode suffix."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCachePlanner,
    DSARequestCacheStage,
)


BLOCK_SIZE = 128
SPARSE_ACTIVATION_TOKENS = 6144


@dataclass
class _Request:
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int
    num_output_tokens: int
    num_tokens: int


@dataclass(frozen=True)
class _ResidentSuffixOracle:
    candidate_tokens: int
    logical_tokens: int
    write_horizon_tokens: int
    suffix_tokens: int
    resident_tokens: int


def _oracle(
    *,
    prompt_tokens: int,
    logical_tokens: int,
    write_horizon_tokens: int,
    resident_budget_tokens: int,
    max_model_len: int,
) -> _ResidentSuffixOracle | None:
    """Compute the reference contract without using the production planner."""

    prompt_prefix = (prompt_tokens // BLOCK_SIZE) * BLOCK_SIZE
    logical_tokens = min(logical_tokens, max_model_len)
    write_horizon_tokens = min(write_horizon_tokens, max_model_len)
    if (
        prompt_tokens <= SPARSE_ACTIVATION_TOKENS
        or logical_tokens <= SPARSE_ACTIVATION_TOKENS
        or prompt_prefix < resident_budget_tokens
    ):
        return None

    suffix_tokens = max(
        0,
        max(logical_tokens, write_horizon_tokens) - prompt_prefix,
    )
    return _ResidentSuffixOracle(
        candidate_tokens=prompt_prefix,
        logical_tokens=logical_tokens,
        write_horizon_tokens=write_horizon_tokens,
        suffix_tokens=suffix_tokens,
        resident_tokens=resident_budget_tokens + suffix_tokens,
    )


def _planner_for_budget(resident_budget_tokens: int) -> DSARequestCachePlanner:
    return DSARequestCachePlanner(
        block_size=BLOCK_SIZE,
        sparse_activation_tokens=SPARSE_ACTIVATION_TOKENS,
        prompt_budget_thresholds=(),
        resident_budget_tokens=(resident_budget_tokens,),
        keep_decode_kv_in_hbm=True,
    )


def _decode_plan(
    *,
    prompt_tokens: int,
    logical_tokens: int,
    write_horizon_tokens: int,
    resident_budget_tokens: int,
    max_model_len: int,
):
    planner = _planner_for_budget(resident_budget_tokens)
    request = _Request("request", prompt_tokens, 0, 0, prompt_tokens)
    planner.commit(
        planner.plan(
            request,
            num_new_tokens=prompt_tokens,
            max_model_len=max_model_len,
        )
    )
    request.num_computed_tokens = write_horizon_tokens - 1
    request.num_output_tokens = max(1, logical_tokens - prompt_tokens)
    request.num_tokens = logical_tokens
    return planner.plan(
        request,
        num_new_tokens=1,
        max_model_len=max_model_len,
    )


@pytest.mark.parametrize(
    ("prompt_tokens", "logical_tokens", "write_horizon_tokens", "budget"),
    [
        (6144, 6145, 6145, 6144),
        (6145, 6146, 6146, 6144),
        (6271, 6272, 6272, 6144),
        (6272, 6273, 6273, 6144),
        (32767, 32768, 32768, 10240),
        (32768, 32769, 32769, 10240),
        (32769, 32770, 32770, 10240),
        (65535, 65536, 65536, 12288),
        (65536, 65537, 65537, 12288),
        (65537, 65538, 65538, 12288),
        (7000, 7001, 7004, 4096),
    ],
)
def test_resident_suffix_oracle_boundaries(
    prompt_tokens: int,
    logical_tokens: int,
    write_horizon_tokens: int,
    budget: int,
) -> None:
    contract = _oracle(
        prompt_tokens=prompt_tokens,
        logical_tokens=logical_tokens,
        write_horizon_tokens=write_horizon_tokens,
        resident_budget_tokens=budget,
        max_model_len=131072,
    )

    if contract is None:
        assert prompt_tokens <= SPARSE_ACTIVATION_TOKENS
        return

    assert contract.candidate_tokens % BLOCK_SIZE == 0
    assert contract.candidate_tokens >= budget
    assert contract.resident_tokens == budget + contract.suffix_tokens
    assert contract.candidate_tokens + contract.suffix_tokens == max(
        contract.logical_tokens,
        contract.write_horizon_tokens,
    )


@pytest.mark.parametrize(
    (
        "prompt_tokens",
        "logical_tokens",
        "write_horizon_tokens",
        "budget",
        "max_model_len",
    ),
    [
        (6145, 6146, 6146, 6144, 16384),
        (6271, 6400, 6400, 6144, 16384),
        (7000, 7100, 7001, 4096, 16384),
        (7000, 7001, 7004, 4096, 16384),
        (7000, 16383, 16390, 4096, 16384),
    ],
)
def test_planner_matches_resident_suffix_oracle(
    prompt_tokens: int,
    logical_tokens: int,
    write_horizon_tokens: int,
    budget: int,
    max_model_len: int,
) -> None:
    contract = _oracle(
        prompt_tokens=prompt_tokens,
        logical_tokens=logical_tokens,
        write_horizon_tokens=write_horizon_tokens,
        resident_budget_tokens=budget,
        max_model_len=max_model_len,
    )
    assert contract is not None

    plan = _decode_plan(
        prompt_tokens=prompt_tokens,
        logical_tokens=logical_tokens,
        write_horizon_tokens=write_horizon_tokens,
        resident_budget_tokens=budget,
        max_model_len=max_model_len,
    )

    assert plan.stage == DSARequestCacheStage.ENTER_SPARSE_DECODE
    assert plan.sparse_budget_tokens == budget
    assert plan.persistent_suffix_start_tokens == contract.candidate_tokens
    assert plan.persistent_suffix_valid_tokens == contract.suffix_tokens
    assert plan.resident_valid_tokens == contract.resident_tokens


def test_prefix_smaller_than_budget_cannot_enter_sparse_decode() -> None:
    plan = _decode_plan(
        prompt_tokens=6145,
        logical_tokens=7000,
        write_horizon_tokens=7000,
        resident_budget_tokens=10240,
        max_model_len=16384,
    )

    assert plan.stage == DSARequestCacheStage.DENSE_DECODE
    assert plan.sparse_budget_tokens == 0
    assert plan.resident_valid_tokens == -1


def test_sparse_decode_never_regresses_to_dense() -> None:
    planner = _planner_for_budget(6144)
    request = _Request("request", 6145, 0, 0, 6145)
    planner.commit(
        planner.plan(request, num_new_tokens=6145, max_model_len=16384)
    )
    request.num_computed_tokens = 6145
    request.num_output_tokens = 1
    request.num_tokens = 6146
    planner.commit(planner.plan(request, num_new_tokens=1, max_model_len=16384))

    request.num_computed_tokens = 1
    request.num_output_tokens = 2
    request.num_tokens = 2
    with pytest.raises(RuntimeError, match="cannot move"):
        planner.plan(request, num_new_tokens=1, max_model_len=16384)
