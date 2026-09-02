# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 请求行与逐层 token-position→resident-slot 持久状态池。

A5 backend 使用纯 token-position→resident-slot 映射，不再占用最后一列保存
legacy first-fill metadata；请求首次进入 sparse 时一次性建立 identity 映射。
legacy backend 继续保留原有 metadata-column 协议。

pool row 独立于 ``InputBatch`` 行号，因此基线对请求行做 condense/reorder 时
不需要搬运一整行 ``max_model_len`` 状态；每轮只需把最终 batch row 映射为
一个稳定 pool index。最后额外保留一行给图模式 PAD 使用。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from contextlib import suppress

import torch

from vllm.utils.math_utils import cdiv

from vllm_ascend.dsa_offload.contracts import (
    DSA_LIDU_CACHE_ROW_ALIGNMENT,
    DSA_LIDU_TOKEN_CAPACITY,
    DSA_OP_BACKEND_NANOVLLM_A5,
    DSA_REQUIRED_CACHE_BLOCK_SIZE,
    DSA_SFA_COMPUTE_TOPK,
)


class DSAResidentTokenPool:
    """管理 worker 内活跃请求的稳定 resident 行与逐层 LIDU 状态。"""

    def __init__(
        self,
        *,
        max_num_reqs: int,
        num_layers: int,
        max_model_len: int,
        max_resident_budget_tokens: int,
        device: torch.device,
        op_backend: str = "default",
        block_size: int = DSA_REQUIRED_CACHE_BLOCK_SIZE,
        reserve_speculative_resident_slots: bool = False,
    ) -> None:
        if max_num_reqs <= 0:
            raise ValueError("DSA resident pool capacity must be positive")
        if num_layers <= 0:
            raise ValueError("DSA resident pool requires at least one layer")
        if max_model_len <= 0:
            raise ValueError("DSA resident pool max_model_len must be positive")
        if max_model_len > DSA_LIDU_TOKEN_CAPACITY:
            raise ValueError(
                "DSA max_model_len exceeds LIDU token-position capacity: "
                f"max_model_len={max_model_len}, "
                f"capacity={DSA_LIDU_TOKEN_CAPACITY}"
            )

        self.max_num_reqs = int(max_num_reqs)
        self.padding_pool_index = self.max_num_reqs
        self.storage_rows = self.max_num_reqs + 1
        self.num_layers = int(num_layers)
        self.max_model_len = int(max_model_len)
        self.max_resident_budget_tokens = int(max_resident_budget_tokens)
        self.device = torch.device(device)
        self.op_backend = str(op_backend)
        self.block_size = int(block_size)
        self._uses_nanovllm_a5_pool = (
            self.op_backend == DSA_OP_BACKEND_NANOVLLM_A5
        )
        self._reserves_speculative_resident_slots = bool(
            reserve_speculative_resident_slots
        )
        if (
            self._reserves_speculative_resident_slots
            and not self._uses_nanovllm_a5_pool
        ):
            raise ValueError(
                "Speculative resident slots require the nanovllm_a5 "
                "DSA backend"
            )
        if (
            self._reserves_speculative_resident_slots
            and self.max_resident_budget_tokens < 2 * DSA_SFA_COMPUTE_TOPK
        ):
            raise ValueError(
                "DSA resident pool capacity cannot reserve the fixed "
                "speculative range: "
                f"capacity={self.max_resident_budget_tokens}, "
                f"required={2 * DSA_SFA_COMPUTE_TOPK}"
            )

        if self._uses_nanovllm_a5_pool:
            max_logical_blocks = cdiv(self.max_model_len, self.block_size)
            self.cache_row_width = max_logical_blocks * self.block_size
        else:
            raw_width = self.max_model_len + 1
            aligned_width = (
                (
                    raw_width
                    + DSA_LIDU_CACHE_ROW_ALIGNMENT
                    - 1
                )
                // DSA_LIDU_CACHE_ROW_ALIGNMENT
                * DSA_LIDU_CACHE_ROW_ALIGNMENT
            )
            self.cache_row_width = (
                aligned_width
                if aligned_width - 1 <= DSA_LIDU_TOKEN_CAPACITY
                else raw_width
            )
        self.cache_metadata_index: int | None = (
            None
            if self._uses_nanovllm_a5_pool
            else self.cache_row_width - 1
        )
        self._identity_slots = (
            torch.arange(
                self.max_resident_budget_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            if self._uses_nanovllm_a5_pool
            else None
        )

        self._free_indices = deque(range(self.max_num_reqs))
        self._request_to_index: dict[Hashable, int] = {}
        self._request_target_budgets: dict[Hashable, int] = {}
        self._cache_slots = torch.full(
            (
                self.num_layers,
                self.storage_rows,
                self.cache_row_width,
            ),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._speculative_resident_slots: torch.Tensor | None
        if self._reserves_speculative_resident_slots:
            self._speculative_resident_slots = torch.full(
                (
                    self.num_layers,
                    self.storage_rows,
                    DSA_SFA_COMPUTE_TOPK,
                ),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self._speculative_resident_slots = None
        if self.cache_metadata_index is not None:
            self._cache_slots[
                :,
                :,
                self.cache_metadata_index,
            ].zero_()
        self._graph_capture_row_count = 0

    def acquire(self, request_id: Hashable) -> int:
        """返回请求稳定行；首次出现时分配并清空一行。"""

        current = self._request_to_index.get(request_id)
        if current is not None:
            return current
        if not self._free_indices:
            raise RuntimeError(
                "No free DSA resident metadata row is available"
            )
        pool_index = self._free_indices.popleft()
        self._request_to_index[request_id] = pool_index
        self._clear_index(pool_index)
        return pool_index

    def prepare_sparse_request(
        self,
        request_id: Hashable,
        *,
        target_budget_tokens: int,
    ) -> None:
        """在请求首次进入 sparse 时安装 backend 对应的 resident 初始状态。"""

        pool_index = self._require_index(request_id)
        target_budget_tokens = int(target_budget_tokens)
        if not (
            0
            < target_budget_tokens
            <= self.max_resident_budget_tokens
        ):
            raise ValueError(
                "DSA resident budget is outside pool capacity: "
                f"budget={target_budget_tokens}, "
                f"capacity={self.max_resident_budget_tokens}"
            )
        existing = self._request_target_budgets.get(request_id)
        if existing is not None:
            if existing != target_budget_tokens:
                raise RuntimeError(
                    "DSA request resident budget changed after binding: "
                    f"request={request_id!r}, old={existing}, "
                    f"new={target_budget_tokens}"
                )
            return

        if self._uses_nanovllm_a5_pool:
            assert self._identity_slots is not None
            normal_budget_tokens = self._normal_budget_tokens(
                target_budget_tokens
            )
            self._cache_slots[
                :, pool_index, :normal_budget_tokens
            ].copy_(
                self._identity_slots[:normal_budget_tokens]
                .view(1, -1)
                .expand(self.num_layers, -1)
            )
            self._install_speculative_resident_slots(
                pool_index,
                target_budget_tokens,
            )
        else:
            assert self.cache_metadata_index is not None
            self._cache_slots[
                :,
                pool_index,
                self.cache_metadata_index,
            ].fill_(-target_budget_tokens)
        self._request_target_budgets[request_id] = target_budget_tokens

    def release(self, request_id: Hashable) -> int | None:
        """释放请求行；下一次分配该行时统一清除所有层状态。"""

        pool_index = self._request_to_index.pop(request_id, None)
        if pool_index is None:
            return None
        self._request_target_budgets.pop(request_id, None)
        self._free_indices.appendleft(pool_index)
        return pool_index

    def get_index(self, request_id: Hashable) -> int | None:
        return self._request_to_index.get(request_id)

    def get_cache_slots(self, layer_id: int) -> torch.Tensor:
        layer_id = int(layer_id)
        if not 0 <= layer_id < self.num_layers:
            raise IndexError(
                f"DSA layer_id={layer_id} is outside "
                f"[0, {self.num_layers})"
            )
        return self._cache_slots[layer_id]

    def get_speculative_resident_slots(
        self,
        layer_id: int,
    ) -> torch.Tensor | None:
        """Return each pool row's fixed speculative resident range.

        The range exists only when explicitly enabled for the nanovllm A5
        backend.  It is initialized at sparse-request admission and occupies
        exactly the high ``DSA_SFA_COMPUTE_TOPK`` slots of that request's
        resident budget.  Normal source-to-slot mappings never use it.
        """

        layer_id = int(layer_id)
        if not 0 <= layer_id < self.num_layers:
            raise IndexError(
                f"DSA layer_id={layer_id} is outside "
                f"[0, {self.num_layers})"
            )
        if self._speculative_resident_slots is None:
            return None
        return self._speculative_resident_slots[layer_id]

    @property
    def speculative_resident_slot_count(self) -> int:
        """Number of fixed high-end slots reserved per admitted A5 row."""

        return (
            DSA_SFA_COMPUTE_TOPK
            if self._speculative_resident_slots is not None
            else 0
        )

    @property
    def graph_capture_row_count(self) -> int:
        return self._graph_capture_row_count

    def prepare_graph_capture(
        self,
        *,
        row_count: int,
        target_budget_tokens: int,
    ) -> None:
        """为原生 FULL-graph dummy-run 安装合法 resident 状态。"""

        row_count = int(row_count)
        target_budget_tokens = int(target_budget_tokens)
        if self._graph_capture_row_count:
            raise RuntimeError(
                "DSA resident graph-capture rows were installed twice"
            )
        if self._request_to_index:
            raise RuntimeError(
                "DSA graph capture must complete before serving requests"
            )
        if not 0 < row_count <= self.max_num_reqs:
            raise ValueError(
                "DSA resident graph-capture row count is outside pool "
                f"capacity: rows={row_count}, capacity={self.max_num_reqs}"
            )
        if not (
            0
            < target_budget_tokens
            <= self.max_resident_budget_tokens
        ):
            raise ValueError(
                "DSA graph-capture budget is outside resident capacity: "
                f"budget={target_budget_tokens}, "
                f"capacity={self.max_resident_budget_tokens}"
            )

        self._graph_capture_row_count = row_count
        try:
            self._cache_slots[:, :row_count].fill_(-1)
            if self._uses_nanovllm_a5_pool:
                assert self._identity_slots is not None
                normal_budget_tokens = self._normal_budget_tokens(
                    target_budget_tokens
                )
                self._cache_slots[
                    :, :row_count, :normal_budget_tokens
                ].copy_(
                    self._identity_slots[:normal_budget_tokens]
                    .view(1, 1, -1)
                    .expand(self.num_layers, row_count, -1)
                )
                for pool_index in range(row_count):
                    self._install_speculative_resident_slots(
                        pool_index,
                        target_budget_tokens,
                    )
            else:
                assert self.cache_metadata_index is not None
                self._cache_slots[
                    :,
                    :row_count,
                    self.cache_metadata_index,
                ].fill_(-target_budget_tokens)
        except Exception:
            with suppress(Exception):
                self._cache_slots[:, :row_count].fill_(-1)
                if self._speculative_resident_slots is not None:
                    self._speculative_resident_slots[
                        :, :row_count
                    ].fill_(-1)
                if self.cache_metadata_index is not None:
                    self._cache_slots[
                        :,
                        :row_count,
                        self.cache_metadata_index,
                    ].zero_()
            self._graph_capture_row_count = 0
            raise

    def restore_after_graph_capture(self) -> None:
        """清除 dummy 对逐层 tokenwise 状态的原址修改。"""

        row_count = self._graph_capture_row_count
        if not row_count:
            return
        self._cache_slots[:, :row_count].fill_(-1)
        if self._speculative_resident_slots is not None:
            self._speculative_resident_slots[:, :row_count].fill_(-1)
        if self.cache_metadata_index is not None:
            self._cache_slots[
                :,
                :row_count,
                self.cache_metadata_index,
            ].zero_()
        self._graph_capture_row_count = 0

    def _require_index(self, request_id: Hashable) -> int:
        pool_index = self._request_to_index.get(request_id)
        if pool_index is None:
            raise KeyError(
                f"DSA request {request_id!r} has no resident pool row"
            )
        return pool_index

    def _clear_index(self, pool_index: int) -> None:
        self._cache_slots[:, pool_index].fill_(-1)
        if self._speculative_resident_slots is not None:
            self._speculative_resident_slots[:, pool_index].fill_(-1)
        if self.cache_metadata_index is not None:
            self._cache_slots[
                :,
                pool_index,
                self.cache_metadata_index,
            ].zero_()

    def _normal_budget_tokens(self, target_budget_tokens: int) -> int:
        """Return the portion of an A5 resident budget usable by source maps."""

        if not self._reserves_speculative_resident_slots:
            return target_budget_tokens
        if target_budget_tokens < 2 * DSA_SFA_COMPUTE_TOPK:
            raise ValueError(
                "DSA resident budget cannot reserve the fixed speculative "
                "range: "
                f"budget={target_budget_tokens}, "
                f"required={2 * DSA_SFA_COMPUTE_TOPK}"
            )
        return target_budget_tokens - DSA_SFA_COMPUTE_TOPK

    def _install_speculative_resident_slots(
        self,
        pool_index: int,
        target_budget_tokens: int,
    ) -> None:
        """Install the canonical high-end range for one A5 pool row."""

        if self._speculative_resident_slots is None:
            return
        start = self._normal_budget_tokens(target_budget_tokens)
        slots = torch.arange(
            start,
            target_budget_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        assert slots.numel() == DSA_SFA_COMPUTE_TOPK
        self._speculative_resident_slots[:, pool_index].copy_(
            slots.view(1, -1).expand(self.num_layers, -1)
        )
