# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA target/draft allocation with on-demand resident suffix growth.

The stable allocation helpers remain in ``kv_cache_manager_base``. This owner
module overrides the allocation paths that need the keep-in-HBM policy:
resident blocks grow only for currently valid decode tokens, while MTP draft
groups continue to allocate target plus lookahead tokens independently.

MTP speculative verification currently uses a multi-token target forward. The
A5 sparse LIDU/KSC contract is single-token/request-row oriented, so MTP keeps
the target resident plane dense as a correctness-first fallback. This prevents
SpecDecoding SFA from addressing prompt blocks that were released after Prefill,
while target and draft decode KV both remain HBM-resident.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.request import Request

from vllm_ascend.dsa_offload import kv_cache_manager_base as _base
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCachePlan,
    DSARequestCacheStage,
)

if TYPE_CHECKING:
    from vllm_ascend.dsa_offload.kv_cache_coordinator import (
        DSAKVCacheCoordinator,
    )


DSAIndexerKVCacheManager = _base.DSAIndexerKVCacheManager
DSAResidentMLAKVCacheManager = _base.DSAResidentMLAKVCacheManager


def _force_mtp_dense_target_plan(
    coordinator: DSAKVCacheCoordinator,
    plan: DSARequestCachePlan,
) -> DSARequestCachePlan:
    """Keep target resident dense for multi-token MTP verification.

    The MTP draft groups remain independent and still allocate
    ``target + lookahead`` tokens. Only the target resident layout transition is
    suppressed. Decode dumping is separately disabled by
    ``keep_decode_kv_in_hbm`` in the runtime dump planner.
    """

    if (
        not coordinator.mtp_group_ids
        or not coordinator.keep_decode_kv_in_hbm
        or not plan.stage.is_sparse_decode
    ):
        return plan

    logger.info_once(
        "[DSA_MTP_KV] event=FORCE_DENSE_TARGET_LAYOUT "
        "reason=multi_token_spec_verification target_dump=false "
        "draft_dump=false"
    )
    return replace(
        plan,
        stage=DSARequestCacheStage.DENSE_DECODE,
        sparse_budget_tokens=0,
        resident_valid_tokens=-1,
        persistent_suffix_start_tokens=0,
        persistent_suffix_valid_tokens=0,
        preserve_resident_tail_block=False,
    )


def _log_mtp_allocate(
    coordinator: DSAKVCacheCoordinator,
    request: Request,
    plan: DSARequestCachePlan,
    *,
    target_tokens: int,
    num_lookahead_tokens: int,
    new_blocks: list[list[KVCacheBlock]],
) -> None:
    """Log MTP only when a group/layout allocation actually changes."""

    if not coordinator.mtp_group_ids or not logger.isEnabledFor(logging.INFO):
        return
    draft_new_blocks = {
        group_id: len(new_blocks[group_id])
        for group_id in sorted(coordinator.mtp_group_ids)
    }
    if not any(draft_new_blocks.values()) and not plan.replace_resident_blocks:
        return
    logger.info(
        "[DSA_MTP_KV] event=ALLOCATE request_id=%s "
        "target_tokens=%d lookahead_tokens=%d cumulative_output_tokens=%d "
        "resident_valid_tokens=%d draft_new_blocks=%s",
        request.request_id,
        int(target_tokens),
        int(num_lookahead_tokens),
        int(request.num_output_tokens),
        int(plan.resident_valid_tokens),
        draft_new_blocks,
    )


def _allocate_dense_plan(
    manager: KVCacheManager,
    coordinator: DSAKVCacheCoordinator,
    request: Request,
    plan: DSARequestCachePlan,
    empty_blocks: tuple[Sequence[KVCacheBlock], ...],
    *,
    num_lookahead_tokens: int,
) -> KVCacheBlocks | None:
    target_tokens = int(plan.indexer_tokens_need_slot)
    total_tokens = min(
        target_tokens + int(num_lookahead_tokens),
        int(manager.max_model_len),
    )
    requirements = _base._requirements_for_groups(
        coordinator,
        request_id=request.request_id,
        target_tokens=target_tokens,
        total_tokens=total_tokens,
        total_computed_tokens=int(request.num_computed_tokens),
        empty_blocks=empty_blocks,
    )
    if not coordinator.can_allocate(requirements):
        return None

    new_blocks = _base._allocate_group_blocks(
        coordinator,
        request_id=request.request_id,
        target_tokens=target_tokens,
        total_tokens=total_tokens,
    )
    coordinator.request_cache_layout.commit(plan)
    _log_mtp_allocate(
        coordinator,
        request,
        plan,
        target_tokens=target_tokens,
        num_lookahead_tokens=num_lookahead_tokens,
        new_blocks=new_blocks,
    )
    return manager.create_kv_cache_blocks(tuple(new_blocks))


def _prepare_keep_resident_layout(
    coordinator: DSAKVCacheCoordinator,
    request: Request,
    plan: DSARequestCachePlan,
    *,
    num_lookahead_tokens: int,
) -> tuple[list[KVCacheBlock], list[KVCacheBlock], int, int]:
    """Return the compact current resident layout without future reservation."""

    del request, num_lookahead_tokens
    resident_manager = coordinator.resident_manager
    request_id = plan.request_id
    block_size = int(resident_manager.block_size)
    budget_blocks = cdiv(int(plan.sparse_budget_tokens), block_size)
    required_blocks = cdiv(int(plan.resident_valid_tokens), block_size)
    existing = list(resident_manager.req_to_blocks.get(request_id, ()))
    previous_state = coordinator.request_cache_layout.get_state(request_id)

    if previous_state is not None and previous_state.prefill_resident_released:
        if len(existing) < budget_blocks:
            raise RuntimeError(
                "DSA persistent resident layout lost its budget prefix: "
                f"request_id={request_id!r}, actual={len(existing)}, "
                f"budget_blocks={budget_blocks}"
            )
        preserved_layout = existing
        releasable: list[KVCacheBlock] = []
    else:
        suffix_block_index = (
            int(plan.persistent_suffix_start_tokens) // block_size
        )
        prompt_prefix = existing[:suffix_block_index]
        persistent_suffix = existing[suffix_block_index:]
        if len(prompt_prefix) < budget_blocks:
            raise RuntimeError(
                "DSA ENTER cannot reuse enough dumped prompt blocks for "
                f"its budget: request_id={request_id!r}, "
                f"prompt_prefix={len(prompt_prefix)}, "
                f"budget_blocks={budget_blocks}"
            )
        preserved_layout = prompt_prefix[:budget_blocks] + persistent_suffix
        releasable = prompt_prefix[budget_blocks:]

    if len(preserved_layout) > required_blocks:
        releasable.extend(preserved_layout[required_blocks:])
        preserved_layout = preserved_layout[:required_blocks]

    resident_suffix_blocks = max(0, required_blocks - budget_blocks)
    return (
        preserved_layout,
        releasable,
        required_blocks,
        resident_suffix_blocks,
    )


def _allocate_keep_sparse_plan(
    manager: KVCacheManager,
    coordinator: DSAKVCacheCoordinator,
    request: Request,
    plan: DSARequestCachePlan,
    *,
    num_lookahead_tokens: int,
    empty_blocks: tuple[Sequence[KVCacheBlock], ...],
) -> KVCacheBlocks | None:
    """Allocate target resident blocks only for currently valid suffix tokens."""

    request_id = request.request_id
    target_tokens = int(plan.indexer_tokens_need_slot)
    total_tokens = min(
        target_tokens + int(num_lookahead_tokens),
        int(manager.max_model_len),
    )
    requirements = _base._requirements_for_groups(
        coordinator,
        request_id=request_id,
        target_tokens=target_tokens,
        total_tokens=total_tokens,
        total_computed_tokens=int(request.num_computed_tokens),
        empty_blocks=empty_blocks,
        skip_resident=True,
    )

    resident_manager = coordinator.resident_manager
    resident_group_id = coordinator.group_ids.resident_mla
    block_size = int(resident_manager.block_size)
    budget_block_count = cdiv(int(plan.sparse_budget_tokens), block_size)
    preserved_layout: list[KVCacheBlock] = []
    releasable: list[KVCacheBlock] = []
    target_resident_blocks = len(
        resident_manager.req_to_blocks.get(request_id, ())
    )
    resident_suffix_blocks = max(
        0,
        target_resident_blocks - budget_block_count,
    )

    if plan.replace_resident_blocks:
        (
            preserved_layout,
            releasable,
            target_resident_blocks,
            resident_suffix_blocks,
        ) = _prepare_keep_resident_layout(
            coordinator,
            request,
            plan,
            num_lookahead_tokens=num_lookahead_tokens,
        )
        resident_new_blocks_needed = (
            target_resident_blocks - len(preserved_layout)
        )
        if resident_new_blocks_needed < 0:
            raise RuntimeError(
                "DSA persistent suffix layout exceeds its current plan: "
                f"request_id={request_id!r}, "
                f"preserved={len(preserved_layout)}, "
                f"target={target_resident_blocks}"
            )
        requirements[resident_group_id] = max(
            0,
            resident_new_blocks_needed - len(releasable),
        )
    else:
        current_blocks = len(
            resident_manager.req_to_blocks.get(request_id, ())
        )
        required_valid_blocks = cdiv(
            int(plan.resident_valid_tokens),
            block_size,
        )
        resident_new_blocks_needed = max(
            0,
            required_valid_blocks - current_blocks,
        )
        requirements[resident_group_id] = resident_new_blocks_needed
        target_resident_blocks = max(current_blocks, required_valid_blocks)
        resident_suffix_blocks = max(
            0,
            target_resident_blocks - budget_block_count,
        )

    if not coordinator.can_allocate(requirements):
        return None

    new_blocks = _base._allocate_group_blocks(
        coordinator,
        request_id=request_id,
        target_tokens=target_tokens,
        total_tokens=total_tokens,
        skip_resident=True,
    )

    if plan.replace_resident_blocks:
        resident_manager.req_to_blocks[request_id] = list(preserved_layout)
        if releasable:
            resident_manager.block_pool.free_blocks(reversed(releasable))
        resident_manager.num_cached_block.pop(request_id, None)

        resident_capacity_tokens = target_resident_blocks * block_size
        resident_manager.allocate_new_blocks(
            request_id,
            resident_capacity_tokens,
            resident_capacity_tokens,
        )
        new_blocks[resident_group_id] = list(
            resident_manager.req_to_blocks[request_id][:budget_block_count]
        )
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "[DSA_RESIDENT_LAYOUT] event=ENTER_PERSISTENT_SUFFIX "
                "request_id=%s sparse_budget_tokens=%d suffix_start=%d "
                "persistent_suffix_valid_tokens=%d "
                "resident_suffix_blocks=%d total_resident_blocks=%d",
                request_id,
                int(plan.sparse_budget_tokens),
                int(plan.persistent_suffix_start_tokens),
                int(plan.persistent_suffix_valid_tokens),
                resident_suffix_blocks,
                len(resident_manager.req_to_blocks[request_id]),
            )
    elif requirements[resident_group_id]:
        resident_capacity_tokens = target_resident_blocks * block_size
        new_blocks[resident_group_id] = resident_manager.allocate_new_blocks(
            request_id,
            resident_capacity_tokens,
            resident_capacity_tokens,
        )

    coordinator.request_cache_layout.commit(plan)
    _log_mtp_allocate(
        coordinator,
        request,
        plan,
        target_tokens=target_tokens,
        num_lookahead_tokens=num_lookahead_tokens,
        new_blocks=new_blocks,
    )
    return manager.create_kv_cache_blocks(tuple(new_blocks))


def allocate_dsa_slots(
    manager: KVCacheManager,
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: KVCacheBlocks | None = None,
    num_lookahead_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
    full_sequence_must_fit: bool = False,
    reserved_blocks: int = 0,
) -> KVCacheBlocks | None:
    """Dispatch DSA allocation while keeping target/draft ownership separate."""

    from vllm_ascend.dsa_offload.kv_cache_coordinator import (
        DSAKVCacheCoordinator,
    )

    coordinator = manager.coordinator
    if not isinstance(coordinator, DSAKVCacheCoordinator):
        raise RuntimeError(
            "allocate_dsa_slots requires DSAKVCacheCoordinator, got "
            f"{type(coordinator).__name__}"
        )
    if num_new_tokens <= 0:
        raise ValueError("DSA num_new_tokens must be greater than 0")

    empty_blocks = _base._require_runtime_contract(
        manager=manager,
        coordinator=coordinator,
        new_computed_blocks=new_computed_blocks,
        num_new_computed_tokens=num_new_computed_tokens,
        num_lookahead_tokens=num_lookahead_tokens,
        num_external_computed_tokens=num_external_computed_tokens,
        delay_cache_blocks=delay_cache_blocks,
        num_encoder_tokens=num_encoder_tokens,
        reserved_blocks=reserved_blocks,
    )

    if full_sequence_must_fit:
        full_target_tokens = min(
            int(request.num_tokens),
            int(manager.max_model_len),
        )
        full_total_tokens = min(
            full_target_tokens + int(num_lookahead_tokens),
            int(manager.max_model_len),
        )
        full_requirements = _base._requirements_for_groups(
            coordinator,
            request_id=request.request_id,
            target_tokens=full_target_tokens,
            total_tokens=full_total_tokens,
            total_computed_tokens=int(request.num_computed_tokens),
            empty_blocks=empty_blocks,
        )
        if not coordinator.can_allocate(full_requirements):
            return None

    plan = coordinator.request_cache_layout.plan(
        request,
        num_new_tokens=num_new_tokens,
        max_model_len=manager.max_model_len,
    )
    plan = _force_mtp_dense_target_plan(coordinator, plan)
    if plan.stage.uses_dense_layout:
        return _allocate_dense_plan(
            manager,
            coordinator,
            request,
            plan,
            empty_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
        )
    if not coordinator.keep_decode_kv_in_hbm:
        return _base._allocate_legacy_sparse_plan(
            manager,
            coordinator,
            request,
            plan,
        )
    return _allocate_keep_sparse_plan(
        manager,
        coordinator,
        request,
        plan,
        num_lookahead_tokens=num_lookahead_tokens,
        empty_blocks=empty_blocks,
    )


__all__ = [
    "DSAIndexerKVCacheManager",
    "DSAResidentMLAKVCacheManager",
    "allocate_dsa_slots",
]
