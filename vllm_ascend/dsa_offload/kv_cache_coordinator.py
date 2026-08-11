# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA KV-cache coordinator with on-demand persistent-suffix retention.

The stable multi-pool coordinator remains in ``kv_cache_coordinator_base``.
This owner module narrows the keep-in-HBM optimization to final-prefill
resident release: retain only the sparse budget and an optional prompt tail;
decode blocks are appended on demand by ``kv_cache_manager``.

MTP speculative verification is a correctness-first exception. The current A5
LIDU/KSC sparse data plane is single-token/request-row oriented, while MTP
verifies multiple target tokens in one forward. Until a multi-token offload
operator contract is available, MTP keeps the target resident prompt dense in
HBM after the optional Prefill dump so generic SpecDecoding SFA never addresses
an already-released prompt block. Draft groups remain independent HBM pools.
"""

from __future__ import annotations

import logging

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.request import Request

from vllm_ascend.dsa_offload.kv_cache_coordinator_base import (
    DSABlockPoolView,
    DSAKVCacheCoordinator as _BaseDSAKVCacheCoordinator,
)


class DSAKVCacheCoordinator(_BaseDSAKVCacheCoordinator):
    """Keep only currently valid target resident KV after final prefill."""

    def release_prefill_resident_blocks(
        self,
        request: Request,
        *,
        preserve_tail_block: bool,
    ) -> bool:
        """Convert dense prompt resident blocks after final-prefill dump.

        With ``keep_decode_kv_in_hbm`` enabled, this normally keeps only:

        * the blocks reused as the sparse resident budget; and
        * the prompt tail block when the prompt is not block aligned.

        Future decode blocks are not reserved from ``request.max_tokens``.
        They are allocated only when the persistent suffix crosses a block
        boundary, preserving HBM capacity for concurrent requests.

        MTP is intentionally different: its multi-token verification path uses
        the generic SFA interface and therefore still requires the complete
        target resident block table. Keep the dense prompt resident in HBM for
        correctness; Prefill dumping, when enabled, has already completed before
        this scheduler-side release hook runs.
        """

        request_id = request.request_id
        req_blocks = self.resident_manager.req_to_blocks.get(request_id)
        if not req_blocks:
            return False

        if self.keep_decode_kv_in_hbm and self.mtp_group_ids:
            logger.info_once(
                "[DSA_MTP_KV] event=KEEP_DENSE_TARGET_RESIDENT "
                "reason=multi_token_spec_verification "
                "prefill_resident_release=false decode_dump=false"
            )
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

            budget_blocks = full_prompt_blocks[:budget_block_count]
            releasable = full_prompt_blocks[budget_block_count:]
            persistent_layout = list(budget_blocks)
            if tail_block is not None:
                persistent_layout.append(tail_block)
            self.resident_manager.req_to_blocks[request_id] = persistent_layout
            reused_budget_blocks = len(budget_blocks)
            kept_tail_blocks = int(tail_block is not None)

        if releasable:
            self.resident_manager.block_pool.free_blocks(reversed(releasable))
        released_prompt_blocks = len(releasable)
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


__all__ = ["DSABlockPoolView", "DSAKVCacheCoordinator"]
