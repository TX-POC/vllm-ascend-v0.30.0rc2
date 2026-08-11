# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA target planes 与 MTP draft groups 的生命周期分配。

请求阶段由 ``request_cache_layout`` 以 plan/commit 协议统一规划。本文件把
规划结果翻译为独立物理池的实际 block 分配：

* target Indexer 按主模型 token 数增长；
* target resident 在 keep 模式使用 ``budget + persistent suffix``；
* 每个 MTP draft group 按 ``target + lookahead`` 分配；
* 关闭 keep 开关时保持原来的单-tail resident 合同并拒绝 lookahead。

所有容量不足都必须在物理表修改前返回，不能让失败重试提前推进阶段。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.request import Request

from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCachePlan,
    DSARequestCacheStage,
)

if TYPE_CHECKING:
    from vllm_ascend.dsa_offload.kv_cache_coordinator import (
        DSAKVCacheCoordinator,
    )


class DSAIndexerKVCacheManager(FullAttentionManager):
    """管理完整上下文 target Indexer dense plane。"""


class DSAResidentMLAKVCacheManager(FullAttentionManager):
    """管理 dense-prefill / sparse-decode 共用的 target resident plane。"""


def _require_runtime_contract(
    *,
    manager: KVCacheManager,
    coordinator: DSAKVCacheCoordinator,
    new_computed_blocks: KVCacheBlocks | None,
    num_new_computed_tokens: int,
    num_lookahead_tokens: int,
    num_external_computed_tokens: int,
    delay_cache_blocks: bool,
    num_encoder_tokens: int,
    reserved_blocks: int,
) -> tuple[Sequence[KVCacheBlock], ...]:
    unsupported = {
        "num_new_computed_tokens": num_new_computed_tokens,
        "num_external_computed_tokens": num_external_computed_tokens,
        "delay_cache_blocks": delay_cache_blocks,
        "num_encoder_tokens": num_encoder_tokens,
        "reserved_blocks": reserved_blocks,
    }
    if not coordinator.keep_decode_kv_in_hbm:
        unsupported["num_lookahead_tokens"] = num_lookahead_tokens
    active = {name: value for name, value in unsupported.items() if bool(value)}
    if active:
        raise RuntimeError(
            "DSA request cache layout received unsupported allocation "
            f"features: {active}"
        )

    blocks = (
        manager.empty_kv_cache_blocks.blocks
        if new_computed_blocks is None
        else new_computed_blocks.blocks
    )
    if any(blocks):
        raise RuntimeError("DSA request cache layout does not yet support prefix-cache hits")
    return blocks


def _group_desired_tokens(
    coordinator: DSAKVCacheCoordinator,
    group_id: int,
    *,
    target_tokens: int,
    total_tokens: int,
) -> int:
    if group_id in (
        coordinator.group_ids.indexer,
        coordinator.group_ids.resident_mla,
    ):
        return target_tokens
    return total_tokens


def _requirements_for_groups(
    coordinator: DSAKVCacheCoordinator,
    *,
    request_id: str,
    target_tokens: int,
    total_tokens: int,
    total_computed_tokens: int,
    empty_blocks: tuple[Sequence[KVCacheBlock], ...],
    skip_resident: bool = False,
) -> list[int]:
    requirements: list[int] = []
    for group_id, single_manager in enumerate(
        coordinator.single_type_managers
    ):
        if skip_resident and group_id == coordinator.group_ids.resident_mla:
            requirements.append(0)
            continue
        desired_tokens = _group_desired_tokens(
            coordinator,
            group_id,
            target_tokens=target_tokens,
            total_tokens=total_tokens,
        )
        requirements.append(
            single_manager.get_num_blocks_to_allocate(
                request_id=request_id,
                num_tokens=desired_tokens,
                new_computed_blocks=empty_blocks[group_id],
                total_computed_tokens=total_computed_tokens,
                num_tokens_main_model=target_tokens,
            )
        )
    return requirements


def _allocate_group_blocks(
    coordinator: DSAKVCacheCoordinator,
    *,
    request_id: str,
    target_tokens: int,
    total_tokens: int,
    skip_resident: bool = False,
) -> list[list[KVCacheBlock]]:
    new_blocks: list[list[KVCacheBlock]] = [
        [] for _ in range(coordinator.num_single_type_manager)
    ]
    for group_id, single_manager in enumerate(
        coordinator.single_type_managers
    ):
        if skip_resident and group_id == coordinator.group_ids.resident_mla:
            continue
        desired_tokens = _group_desired_tokens(
            coordinator,
            group_id,
            target_tokens=target_tokens,
            total_tokens=total_tokens,
        )
        new_blocks[group_id] = single_manager.allocate_new_blocks(
            request_id,
            desired_tokens,
            target_tokens,
        )
    return new_blocks


def _log_mtp_allocate(
    coordinator: DSAKVCacheCoordinator,
    request: Request,
    plan: DSARequestCachePlan,
    *,
    target_tokens: int,
    num_lookahead_tokens: int,
    new_blocks: list[list[KVCacheBlock]],
) -> None:
    if not coordinator.mtp_group_ids or not logger.isEnabledFor(logging.INFO):
        return
    logger.info(
        "[DSA_MTP_KV] event=ALLOCATE request_id=%s "
        "target_tokens=%d lookahead_tokens=%d accepted_tokens=%d "
        "resident_valid_tokens=%d draft_new_blocks=%s",
        request.request_id,
        target_tokens,
        int(num_lookahead_tokens),
        int(request.num_output_tokens),
        int(plan.resident_valid_tokens),
        {
            group_id: len(new_blocks[group_id])
            for group_id in sorted(coordinator.mtp_group_ids)
        },
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
    requirements = _requirements_for_groups(
        coordinator,
        request_id=request.request_id,
        target_tokens=target_tokens,
        total_tokens=total_tokens,
        total_computed_tokens=int(request.num_computed_tokens),
        empty_blocks=empty_blocks,
    )
    if not coordinator.can_allocate(requirements):
        return None

    new_blocks = _allocate_group_blocks(
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


def _release_legacy_resident_for_enter(
    coordinator: DSAKVCacheCoordinator,
    request_id: str,
    *,
    preserve_tail_block: bool,
) -> KVCacheBlock | None:
    resident_manager = coordinator.resident_manager
    req_blocks = resident_manager.req_to_blocks.get(request_id, [])
    preserved_tail = req_blocks[-1] if preserve_tail_block and req_blocks else None
    releasable = req_blocks[:-1] if preserved_tail is not None else req_blocks
    resident_manager.req_to_blocks[request_id] = []
    if releasable:
        resident_manager.block_pool.free_blocks(reversed(releasable))
    resident_manager.num_cached_block.pop(request_id, None)
    return preserved_tail


def _allocate_legacy_sparse_plan(
    manager: KVCacheManager,
    coordinator: DSAKVCacheCoordinator,
    request: Request,
    plan: DSARequestCachePlan,
) -> KVCacheBlocks | None:
    indexer_manager = coordinator.indexer_manager
    resident_manager = coordinator.resident_manager
    request_id = request.request_id
    empty: tuple[KVCacheBlock, ...] = ()

    indexer_blocks_needed = indexer_manager.get_num_blocks_to_allocate(
        request_id=request_id,
        num_tokens=plan.indexer_tokens_need_slot,
        new_computed_blocks=empty,
        total_computed_tokens=request.num_computed_tokens,
        num_tokens_main_model=plan.indexer_tokens_need_slot,
    )
    if indexer_blocks_needed > indexer_manager.block_pool.get_num_free_blocks():
        return None

    resident_new_blocks_needed = 0
    preserve_tail = False
    releasable_resident_blocks = 0
    if plan.replace_resident_blocks:
        existing_resident_blocks = resident_manager.req_to_blocks.get(
            request_id,
            [],
        )
        preserve_tail = plan.preserve_resident_tail_block and bool(existing_resident_blocks)
        releasable_resident_blocks = len(existing_resident_blocks) - int(preserve_tail)
        target_resident_blocks = cdiv(
            plan.resident_valid_tokens,
            resident_manager.block_size,
        )
        resident_new_blocks_needed = target_resident_blocks - int(preserve_tail)
        resident_available_after_release = (
            resident_manager.block_pool.get_num_free_blocks()
            + releasable_resident_blocks
        )
        if resident_new_blocks_needed > resident_available_after_release:
            return None
    else:
        expected_resident_blocks = cdiv(
            plan.resident_valid_tokens,
            resident_manager.block_size,
        )
        actual_resident_blocks = len(
            resident_manager.req_to_blocks.get(request_id, ())
        )
        if actual_resident_blocks != expected_resident_blocks:
            raise RuntimeError(
                "DSA steady sparse resident block table changed "
                f"unexpectedly: req_id={request_id}, "
                f"actual={actual_resident_blocks}, "
                f"expected={expected_resident_blocks}"
            )

    new_indexer_blocks = indexer_manager.allocate_new_blocks(
        request_id,
        plan.indexer_tokens_need_slot,
        plan.indexer_tokens_need_slot,
    )
    new_resident_blocks: list[KVCacheBlock] = []
    if plan.replace_resident_blocks:
        preserved_tail = _release_legacy_resident_for_enter(
            coordinator,
            request_id,
            preserve_tail_block=preserve_tail,
        )
        resident_slots_to_allocate = (
            plan.sparse_budget_tokens
            if preserved_tail is not None
            else plan.resident_valid_tokens
        )
        new_resident_blocks = resident_manager.allocate_new_blocks(
            request_id,
            resident_slots_to_allocate,
            resident_slots_to_allocate,
        )
        if preserved_tail is not None:
            resident_manager.req_to_blocks[request_id].append(preserved_tail)
        if len(new_resident_blocks) != resident_new_blocks_needed:
            raise RuntimeError(
                "DSA ENTER resident allocation disagreed with its "
                f"capacity precheck: req_id={request_id}, "
                f"allocated={len(new_resident_blocks)}, "
                f"expected={resident_new_blocks_needed}"
            )

    coordinator.request_cache_layout.commit(plan)
    grouped: list[list[KVCacheBlock]] = [
        [] for _ in range(coordinator.num_single_type_manager)
    ]
    grouped[coordinator.group_ids.indexer] = new_indexer_blocks
    grouped[coordinator.group_ids.resident_mla] = new_resident_blocks
    return manager.create_kv_cache_blocks(tuple(grouped))


def _prepare_keep_resident_layout(
    coordinator: DSAKVCacheCoordinator,
    request: Request,
    plan: DSARequestCachePlan,
    *,
    num_lookahead_tokens: int,
) -> tuple[list[KVCacheBlock], list[KVCacheBlock], int, int]:
    """返回 ENTER 的已保留布局、可释放块、目标容量和 suffix block 数。"""

    resident_manager = coordinator.resident_manager
    request_id = request.request_id
    block_size = int(resident_manager.block_size)
    budget_blocks = cdiv(int(plan.sparse_budget_tokens), block_size)
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
        suffix_block_index = int(plan.persistent_suffix_start_tokens) // block_size
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

    speculative_slack = max(
        int(num_lookahead_tokens),
        len(request.spec_token_ids),
    )
    max_target_tokens = min(
        int(request.num_prompt_tokens)
        + int(request.max_tokens)
        + speculative_slack,
        int(coordinator.max_model_len),
    )
    max_target_tokens = max(
        max_target_tokens,
        int(plan.indexer_tokens_need_slot),
    )
    suffix_capacity_tokens = max(
        0,
        max_target_tokens - int(plan.persistent_suffix_start_tokens),
    )
    resident_capacity_tokens = (
        int(plan.sparse_budget_tokens) + suffix_capacity_tokens
    )
    target_resident_blocks = cdiv(resident_capacity_tokens, block_size)
    return (
        preserved_layout,
        releasable,
        target_resident_blocks,
        cdiv(suffix_capacity_tokens, block_size),
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
    request_id = request.request_id
    target_tokens = int(plan.indexer_tokens_need_slot)
    total_tokens = min(
        target_tokens + int(num_lookahead_tokens),
        int(manager.max_model_len),
    )
    requirements = _requirements_for_groups(
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
    pseudo_resident_delta: list[KVCacheBlock] = []
    preserved_layout: list[KVCacheBlock] = []
    releasable: list[KVCacheBlock] = []
    target_resident_blocks = len(
        resident_manager.req_to_blocks.get(request_id, ())
    )
    suffix_capacity_blocks = max(
        0,
        target_resident_blocks
        - cdiv(int(plan.sparse_budget_tokens), resident_manager.block_size),
    )

    if plan.replace_resident_blocks:
        (
            preserved_layout,
            releasable,
            target_resident_blocks,
            suffix_capacity_blocks,
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
                "DSA persistent suffix layout exceeds its capacity plan: "
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
            int(resident_manager.block_size),
        )
        resident_new_blocks_needed = max(
            0,
            required_valid_blocks - current_blocks,
        )
        requirements[resident_group_id] = resident_new_blocks_needed
        target_resident_blocks = max(current_blocks, required_valid_blocks)

    if not coordinator.can_allocate(requirements):
        return None

    new_blocks = _allocate_group_blocks(
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
        resident_capacity_tokens = (
            target_resident_blocks * int(resident_manager.block_size)
        )
        resident_manager.allocate_new_blocks(
            request_id,
            resident_capacity_tokens,
            resident_capacity_tokens,
        )
        budget_block_count = cdiv(
            int(plan.sparse_budget_tokens),
            int(resident_manager.block_size),
        )
        pseudo_resident_delta = list(
            resident_manager.req_to_blocks[request_id][:budget_block_count]
        )
        new_blocks[resident_group_id] = pseudo_resident_delta
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "[DSA_RESIDENT_LAYOUT] event=ENTER_PERSISTENT_SUFFIX "
                "request_id=%s sparse_budget_tokens=%d suffix_start=%d "
                "persistent_suffix_valid_tokens=%d "
                "reserved_decode_blocks=%d total_resident_blocks=%d",
                request_id,
                int(plan.sparse_budget_tokens),
                int(plan.persistent_suffix_start_tokens),
                int(plan.persistent_suffix_valid_tokens),
                suffix_capacity_blocks,
                len(resident_manager.req_to_blocks[request_id]),
            )
    elif requirements[resident_group_id]:
        resident_capacity_tokens = (
            target_resident_blocks * int(resident_manager.block_size)
        )
        new_blocks[resident_group_id] = (
            resident_manager.allocate_new_blocks(
                request_id,
                resident_capacity_tokens,
                resident_capacity_tokens,
            )
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
    """执行 target/draft 分组分配和 resident persistent suffix 转换。"""

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

    empty_blocks = _require_runtime_contract(
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
        full_requirements = _requirements_for_groups(
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
    if plan.stage in (
        DSARequestCacheStage.PREFILL,
        DSARequestCacheStage.DENSE_DECODE,
    ):
        return _allocate_dense_plan(
            manager,
            coordinator,
            request,
            plan,
            empty_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
        )
    if not coordinator.keep_decode_kv_in_hbm:
        return _allocate_legacy_sparse_plan(
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
