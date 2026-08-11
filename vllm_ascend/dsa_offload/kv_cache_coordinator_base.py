# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA target 双物理池与 MTP draft pools coordinator。

每个 finalized KV-cache group 使用独立 ``BlockPool``。target Indexer、target
resident MLA 和每个 MTP draft group 的 block id 都只索引各自 tensor；
admission 因此必须逐 component 检查，不能把 free blocks 汇总后作标量判断。

本模块同时持有请求 cache-layout planner，并在最终 prefill dump 完成后把
resident dense 表转换为 ``budget + persistent suffix``。关闭 keep 开关时
保持原来的“释放满块、只留 tail”语义。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

from vllm_ascend.dsa_offload.kv_cache import (
    get_dsa_group_num_blocks,
    get_dsa_kv_cache_group_ids,
    validate_dsa_kv_cache_config,
)
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCachePlanner,
    DSARequestCacheState,
)


class DSABlockPoolView:
    """向 vLLM 顶层暴露多个独立物理池的只读聚合视图。"""

    def __init__(self, block_pools: Sequence[BlockPool]) -> None:
        if not block_pools:
            raise ValueError("DSA requires at least one physical block pool")
        self.block_pools = tuple(block_pools)
        self.num_gpu_blocks = sum(pool.num_gpu_blocks for pool in self.block_pools)
        self.enable_caching = False

    def get_num_free_blocks(self) -> int:
        return sum(pool.get_num_free_blocks() for pool in self.block_pools)

    def get_usage(self) -> float:
        return max(pool.get_usage() for pool in self.block_pools)

    def reset_prefix_cache(self) -> bool:
        results = [pool.reset_prefix_cache() for pool in self.block_pools]
        return all(results)

    def evict_blocks(self, block_ids: set[int]) -> None:
        if block_ids:
            raise RuntimeError("DSA split KV-cache eviction requires group-qualified block ids")

    def take_events(self) -> list:
        events = []
        for pool in self.block_pools:
            events.extend(pool.take_events())
        return events


class DSAKVCacheCoordinator(KVCacheCoordinator):
    """为 target pair 和 MTP draft groups 构造独立 BlockPool/manager。"""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        if enable_caching:
            raise ValueError("DSA split KV-cache does not yet support prefix caching")
        if enable_kv_cache_events:
            raise ValueError(
                "DSA split KV-cache does not yet support KV-cache events: "
                "the independent pools currently reuse block IDs"
            )
        if metrics_collector is not None:
            raise ValueError(
                "DSA split KV-cache does not yet support KV-cache metrics: "
                "the independent pools currently reuse block IDs"
            )
        if scheduler_block_size % hash_block_size != 0 or any(
            scheduler_block_size % group.kv_cache_spec.block_size != 0
            for group in kv_cache_config.kv_cache_groups
        ):
            raise ValueError(
                "DSA scheduler block size must be divisible by the hash block size and every KV-cache group block size"
            )

        validate_dsa_kv_cache_config(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.max_model_len = int(max_model_len)
        self.enable_caching = False
        self.scheduler_block_size = int(scheduler_block_size)
        self.retention_interval = None
        self.group_ids = get_dsa_kv_cache_group_ids(kv_cache_config)

        from vllm_ascend.ascend_config import get_ascend_config

        dsa_config = get_ascend_config().dsa_offload_config
        self.keep_decode_kv_in_hbm = bool(
            dsa_config.keep_decode_kv_in_hbm
        )
        self.eagle_group_ids = set(self.group_ids.draft)
        self.mtp_group_ids = set(self.group_ids.draft)
        if use_eagle and not self.keep_decode_kv_in_hbm:
            raise ValueError(
                "DSA speculative decoding requires keep_decode_kv_in_hbm=True"
            )
        if use_eagle and not self.mtp_group_ids:
            raise ValueError(
                "DSA MTP is enabled but no independent draft KV-cache group was retained"
            )
        if self.mtp_group_ids and not use_eagle:
            raise ValueError(
                "DSA draft KV-cache groups exist without an enabled MTP/eagle runtime"
            )

        physical_pools = tuple(
            BlockPool(
                num_gpu_blocks=get_dsa_group_num_blocks(
                    kv_cache_config,
                    group,
                ),
                enable_caching=False,
                hash_block_size=hash_block_size,
                enable_kv_cache_events=enable_kv_cache_events,
                metrics_collector=metrics_collector,
            )
            for group in kv_cache_config.kv_cache_groups
        )
        self.physical_block_pools = physical_pools
        self.block_pool = DSABlockPoolView(physical_pools)

        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=group.kv_cache_spec,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
                block_pool=physical_pools[group_id],
                enable_caching=False,
                kv_cache_group_id=group_id,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                scheduler_block_size=scheduler_block_size,
            )
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        )
        self.num_single_type_manager = len(self.single_type_managers)
        for group_id, manager in enumerate(self.single_type_managers):
            manager.use_eagle = group_id in self.eagle_group_ids

        from vllm_ascend.dsa_offload.kv_cache_manager import (
            DSAIndexerKVCacheManager,
            DSAResidentMLAKVCacheManager,
        )

        indexer_manager = self.single_type_managers[self.group_ids.indexer]
        resident_manager = self.single_type_managers[self.group_ids.resident_mla]
        if not isinstance(indexer_manager, DSAIndexerKVCacheManager):
            raise RuntimeError(
                f"DSA Indexer group did not resolve to DSAIndexerKVCacheManager: {type(indexer_manager).__name__}"
            )
        if not isinstance(resident_manager, DSAResidentMLAKVCacheManager):
            raise RuntimeError(
                "DSA resident MLA group did not resolve to "
                "DSAResidentMLAKVCacheManager: "
                f"{type(resident_manager).__name__}"
            )
        self.indexer_manager = indexer_manager
        self.resident_manager = resident_manager

        self.request_cache_layout = DSARequestCachePlanner(
            block_size=scheduler_block_size,
            sparse_activation_tokens=dsa_config.sparse_activation_tokens,
            prompt_budget_thresholds=dsa_config.prompt_budget_thresholds,
            resident_budget_tokens=dsa_config.resident_budget_tokens,
            keep_decode_kv_in_hbm=self.keep_decode_kv_in_hbm,
        )

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "[DSA_MTP_KV] event=GROUPS_READY "
                "target_group_ids=%s draft_group_ids=%s capacities=%s",
                [self.group_ids.indexer, self.group_ids.resident_mla],
                sorted(self.mtp_group_ids),
                {
                    group_id: pool.num_gpu_blocks
                    for group_id, pool in enumerate(
                        self.physical_block_pools
                    )
                },
            )

    def get_num_blocks_to_allocate_by_group(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> tuple[int, ...]:
        requirements: list[int] = []
        for group_id, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                requirement = manager.get_num_blocks_to_allocate(
                    request_id,
                    num_encoder_tokens,
                    (),
                    0,
                    num_encoder_tokens,
                    apply_admission_cap=apply_admission_cap,
                )
            else:
                requirement = manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[group_id],
                    total_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
            requirements.append(requirement)
        return tuple(requirements)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        return sum(
            self.get_num_blocks_to_allocate_by_group(
                request_id=request_id,
                num_tokens=num_tokens,
                new_computed_blocks=new_computed_blocks,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_tokens_main_model=num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
        )

    def can_allocate(
        self,
        requirements: Sequence[int],
        *,
        reserved_blocks: int = 0,
    ) -> bool:
        if len(requirements) != len(self.physical_block_pools):
            raise ValueError(
                "DSA component requirement count does not match physical "
                f"pools: requirements={len(requirements)}, "
                f"pools={len(self.physical_block_pools)}"
            )
        if reserved_blocks:
            raise RuntimeError("DSA split KV-cache does not support connector block reservations")
        return all(
            required <= pool.get_num_free_blocks()
            for required, pool in zip(
                requirements,
                self.physical_block_pools,
                strict=True,
            )
        )

    def can_admit_dense_request(
        self,
        *,
        request_id: str,
        num_tokens: int,
        total_computed_tokens: int = 0,
    ) -> bool:
        """检查一个 prefill 请求能否同时装入 target 和 draft planes。"""

        empty_blocks = tuple(() for _ in range(self.num_single_type_manager))
        target_tokens = min(int(num_tokens), self.max_model_len)
        requirements = self.get_num_blocks_to_allocate_by_group(
            request_id=request_id,
            num_tokens=target_tokens,
            new_computed_blocks=empty_blocks,
            num_encoder_tokens=0,
            total_computed_tokens=int(total_computed_tokens),
            num_tokens_main_model=target_tokens,
            apply_admission_cap=True,
        )
        return self.can_allocate(requirements)

    def get_request_cache_state(
        self,
        request_id: str,
    ) -> DSARequestCacheState | None:
        return self.request_cache_layout.get_state(request_id)

    def release_prefill_resident_blocks(
        self,
        request: Request,
        *,
        preserve_tail_block: bool,
    ) -> bool:
        """最终 prefill dump 后转换 resident dense 表。"""

        request_id = request.request_id
        req_blocks = self.resident_manager.req_to_blocks.get(request_id)
        if not req_blocks:
            return False

        before_blocks = list(req_blocks)
        reserved_decode_blocks = 0
        if not self.keep_decode_kv_in_hbm:
            preserved_tail = before_blocks[-1] if preserve_tail_block else None
            releasable = (
                before_blocks[:-1]
                if preserved_tail is not None
                else before_blocks
            )
            self.resident_manager.req_to_blocks[request_id] = (
                [preserved_tail] if preserved_tail is not None else []
            )
            if releasable:
                self.resident_manager.block_pool.free_blocks(
                    reversed(releasable)
                )
            released_prompt_blocks = len(releasable)
            reused_budget_blocks = 0
            kept_tail_blocks = int(preserved_tail is not None)
        else:
            state = self.request_cache_layout.get_state(request_id)
            if state is None:
                raise RuntimeError(
                    "DSA prefill release has no request layout state: "
                    f"request_id={request_id!r}"
                )
            block_size = int(self.resident_manager.block_size)
            budget_block_count = cdiv(
                int(state.target_resident_budget_tokens),
                block_size,
            )
            tail_block = before_blocks[-1] if preserve_tail_block else None
            full_prompt_blocks = (
                before_blocks[:-1] if tail_block is not None else before_blocks
            )
            if len(full_prompt_blocks) < budget_block_count:
                raise RuntimeError(
                    "DSA prefill release cannot reuse enough dumped prompt "
                    "blocks for the sparse resident budget: "
                    f"request_id={request_id!r}, "
                    f"required_budget_blocks={budget_block_count}, "
                    f"full_prompt_blocks={len(full_prompt_blocks)}"
                )

            suffix_start_tokens = (
                int(request.num_prompt_tokens) // block_size
            ) * block_size
            speculative_slack = len(request.spec_token_ids)
            max_target_tokens = min(
                int(request.num_prompt_tokens)
                + int(request.max_tokens)
                + speculative_slack,
                self.max_model_len,
            )
            suffix_capacity_blocks = cdiv(
                max(0, max_target_tokens - suffix_start_tokens),
                block_size,
            )

            budget_blocks = full_prompt_blocks[:budget_block_count]
            remaining_dumped_blocks = full_prompt_blocks[budget_block_count:]
            kept_tail_blocks = int(tail_block is not None)
            reserve_needed = max(
                0,
                suffix_capacity_blocks - kept_tail_blocks,
            )
            decode_reserve = remaining_dumped_blocks[:reserve_needed]
            releasable = remaining_dumped_blocks[len(decode_reserve):]
            persistent_layout = list(budget_blocks)
            if tail_block is not None:
                persistent_layout.append(tail_block)
            persistent_layout.extend(decode_reserve)
            self.resident_manager.req_to_blocks[request_id] = persistent_layout
            if releasable:
                self.resident_manager.block_pool.free_blocks(
                    reversed(releasable)
                )
            released_prompt_blocks = len(releasable)
            reused_budget_blocks = len(budget_blocks)
            reserved_decode_blocks = len(decode_reserve)

        self.resident_manager.num_cached_block.pop(request_id, None)
        self.request_cache_layout.mark_prefill_resident_released(request_id)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "[DSA_RESIDENT_LAYOUT] event=PREFILL_RELEASE "
                "request_id=%s released_prompt_blocks=%d "
                "reused_budget_blocks=%d kept_tail_blocks=%d "
                "reserved_decode_blocks=%d resident_blocks=%d",
                request_id,
                released_prompt_blocks,
                reused_budget_blocks,
                kept_tail_blocks,
                reserved_decode_blocks,
                len(self.resident_manager.req_to_blocks[request_id]),
            )
        return True

    def free(self, request_id: str) -> None:
        super().free(request_id)
        self.request_cache_layout.free(request_id)

    def get_num_common_prefix_blocks(
        self,
        running_request_id: str,
    ) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        del block_hashes, max_cache_hit_length
        return (
            tuple([] for _ in range(self.num_single_type_manager)),
            0,
        )
