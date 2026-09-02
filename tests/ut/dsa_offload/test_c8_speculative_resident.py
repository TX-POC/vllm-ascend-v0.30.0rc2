# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.dsa_offload.c8_speculative_resident import (
    plan_c8_canonical_resident_slots,
)


def test_exact_winners_keep_hits_use_prefetch_then_recall() -> None:
    plan = plan_c8_canonical_resident_slots(
        topk_source_ids=torch.tensor([[6, 2, 3, 5]], dtype=torch.int32),
        cache_slots=torch.tensor(
            [[-1, -1, 11, -1, -1, -1, -1, -1]], dtype=torch.int32
        ),
        speculative_source_ids=torch.tensor([[6, 7, -1, -1]], dtype=torch.int32),
        speculative_resident_slots=torch.tensor(
            [[20, 21, 22, 23]], dtype=torch.int32
        ),
    )

    # 6 uses the slot to which it was posted while C8 continued scanning; 2
    # was a persistent hit; 3/5 get the remaining canonical slots in exact
    # Top-K order through Guaranteed Recall.
    assert plan.attention_slots.tolist() == [[20, 11, 21, 22]]
    assert plan.speculative_winner_mask.tolist() == [[True, False, False, False]]
    assert plan.recall_source_ids.tolist() == [[3, 5, -1, -1]]
    assert plan.recall_destination_slots.tolist() == [[21, 22, -1, -1]]
    assert plan.recall_count.tolist() == [2]


def test_fixed_range_covers_all_final_misses() -> None:
    plan = plan_c8_canonical_resident_slots(
        topk_source_ids=torch.tensor([[4, 3, 2, 1]], dtype=torch.int32),
        cache_slots=torch.full((1, 8), -1, dtype=torch.int32),
        speculative_source_ids=torch.tensor([[4, 3, 2, 1]], dtype=torch.int32),
        speculative_resident_slots=torch.tensor(
            [[40, 41, 42, 43]], dtype=torch.int32
        ),
    )

    assert plan.attention_slots.tolist() == [[40, 41, 42, 43]]
    assert plan.recall_count.tolist() == [0]


def test_planner_rejects_overlap_with_persistent_cache() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        plan_c8_canonical_resident_slots(
            topk_source_ids=torch.tensor([[1, 2]], dtype=torch.int32),
            cache_slots=torch.tensor([[7, -1, -1]], dtype=torch.int32),
            speculative_source_ids=torch.tensor([[1, -1]], dtype=torch.int32),
            speculative_resident_slots=torch.tensor([[7, 8]], dtype=torch.int32),
        )


def test_planner_rejects_speculative_hit() -> None:
    with pytest.raises(ValueError, match="already resident"):
        plan_c8_canonical_resident_slots(
            topk_source_ids=torch.tensor([[1, 2]], dtype=torch.int32),
            cache_slots=torch.tensor([[-1, 3, -1]], dtype=torch.int32),
            speculative_source_ids=torch.tensor([[1, -1]], dtype=torch.int32),
            speculative_resident_slots=torch.tensor([[7, 8]], dtype=torch.int32),
        )
