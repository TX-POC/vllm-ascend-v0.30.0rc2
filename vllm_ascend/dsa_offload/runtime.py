# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 数据面的 worker-lifetime tensor owner 与逐层执行上下文。

``DSAInputBatchCacheLayout`` 仍是请求行状态的唯一 worker 投影；本模块只把
其 active-prefix 转成设备算子需要的固定地址 view：

* resident slot-mapping position；
* active DRAM block table；
* 所有层串行复用的 caller-owned LIDU scratch 输出；
* 本轮满块 dump 的紧凑 src/dst 列。

所有大 tensor 在 model runner 初始化期预分配。steady step 只刷新 active
prefix；DRAM 逻辑表未变化时不会重复 H2D。图模式复用这些 owner 的
captured-prefix + PAD view，而不是另建一套 graph-only 语义。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.dsa_offload.contracts import (
    DSA_DRAM_NULL_BLOCK_ID,
    DSA_DUMP_NOOP_DST_BLOCK_ID,
    DSA_LIDU_OUTPUT_CAPACITY,
    DSA_OP_BACKEND_NANOVLLM_A5,
    DSA_PACKED_KV_DIM,
    DSA_REQUIRED_CACHE_BLOCK_SIZE,
    DSA_ROW_MODE_SPARSE,
    DSA_SFA_COMPUTE_TOPK,
)
from vllm_ascend.dsa_offload.dram_store import DSAHotDRAMStore
from vllm_ascend.dsa_offload.dsa_row import (
    DDRBlockTable,
    DsaRowIndexGeometry,
    PackedKvLayout,
)
from vllm_ascend.dsa_offload.kvaf_adapter import KvafAdapter
from vllm_ascend.dsa_offload.ops import (
    DSALightningIndexerOutputs,
    DSAOffloadSelectionOutput,
    dump_full_kv_cache_blocks,
    kvcache_scatter_copy,
    lightning_indexer_decode_update,
    sparse_flash_attention_for_offload,
)
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)
from vllm_ascend.dsa_offload.resident_pool import DSAResidentTokenPool

if TYPE_CHECKING:
    from vllm_ascend.dsa_offload.input_batch import (
        DSAInputBatchCacheLayout,
    )
    from vllm_ascend.worker.npu_input_batch import NPUInputBatch


class DSAOffloadRuntime:
    """持有 eager 与 graph 共用的固定地址设备元数据。"""

    def __init__(
        self,
        *,
        max_num_reqs: int,
        max_num_tokens: int,
        num_layers: int,
        max_model_len: int,
        block_size: int,
        resident_token_pool: DSAResidentTokenPool,
        device: torch.device,
        pin_memory: bool,
        enable_prefill_full_kv_offload: bool = False,
        keep_decode_kv_in_hbm: bool = False,
        op_backend: str = "default",
        is_c8: bool = False,
    ) -> None:
        self.max_num_reqs = int(max_num_reqs)
        self.max_num_tokens = int(max_num_tokens)
        self.num_layers = int(num_layers)
        self.max_model_len = int(max_model_len)
        self.block_size = int(block_size)
        self.device = torch.device(device)
        self.resident_token_pool = resident_token_pool
        self.enable_prefill_full_kv_offload = bool(
            enable_prefill_full_kv_offload
        )
        self.keep_decode_kv_in_hbm = bool(keep_decode_kv_in_hbm)
        self.op_backend = str(op_backend)
        self.is_c8 = bool(is_c8)
        self.kvaf_adapter = KvafAdapter()
        if self.op_backend == DSA_OP_BACKEND_NANOVLLM_A5:
            from vllm_ascend.dsa_offload.ops_nanovllm import (
                require_nanovllm_dsa_ops,
            )
            require_nanovllm_dsa_ops()
        self.max_logical_blocks = cdiv(
            self.max_model_len,
            self.block_size,
        )
        self.row_index_geometry = DsaRowIndexGeometry.from_runtime(
            raw_row_capacity=self.max_num_reqs,
            n_layers=self.num_layers,
            max_sequence_tokens=self.max_model_len,
            tokens_per_logical_block=self.block_size,
        )

        output_shape = (
            self.max_num_reqs,
            1,
            DSA_LIDU_OUTPUT_CAPACITY,
        )
        self._lidu_topk_index = torch.empty(
            output_shape,
            dtype=torch.int32,
            device=self.device,
        )
        self._lidu_topk_slots = torch.empty_like(self._lidu_topk_index)
        self._lidu_miss_count = torch.empty(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self._lidu_tail_info = torch.empty(
            (self.max_num_reqs, 2),
            dtype=torch.int32,
            device=self.device,
        )
        self._lidu_output_views: dict[
            int,
            DSALightningIndexerOutputs,
        ] = {}

        self.resident_positions = torch.empty(
            self.max_num_tokens,
            dtype=torch.int64,
            device=self.device,
        )
        self._token_row_modes = torch.empty(
            self.max_num_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self._token_sparse_budgets = torch.empty_like(
            self._token_row_modes
        )
        self._token_persistent_suffix_starts = torch.empty_like(
            self._token_row_modes
        )
        self._sparse_token_mask = torch.empty(
            self.max_num_tokens,
            dtype=torch.bool,
            device=self.device,
        )

        self.active_dram_block_table = CpuGpuBuffer(
            self.max_num_reqs,
            self.max_logical_blocks,
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.active_dram_block_table.np.fill(
            DSA_DRAM_NULL_BLOCK_ID
        )
        self.active_dram_block_table.gpu.zero_()
        max_dump_jobs = (
            cdiv(self.max_num_tokens, self.block_size)
            + self.max_num_reqs
        )
        if self.enable_prefill_full_kv_offload:
            max_dump_jobs = max(
                max_dump_jobs,
                self.max_num_reqs * self.max_logical_blocks,
            )
        self.dump_src_block_ids = CpuGpuBuffer(
            max_dump_jobs,
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.dump_dst_block_ids = CpuGpuBuffer(
            max_dump_jobs,
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.dump_src_block_ids.np.fill(0)
        self.dump_dst_block_ids.np.fill(
            DSA_DUMP_NOOP_DST_BLOCK_ID
        )
        self.dump_src_block_ids.gpu.zero_()
        self.dump_dst_block_ids.gpu.fill_(
            DSA_DUMP_NOOP_DST_BLOCK_ID
        )
        self._dump_pool_indices = np.empty(max_dump_jobs, dtype=np.intp)
        self._dump_logical_indices = np.empty(max_dump_jobs, dtype=np.intp)
        self._dump_source_blocks = np.empty(max_dump_jobs, dtype=np.int32)
        self._logical_block_indices = np.arange(
            self.max_logical_blocks,
            dtype=np.intp,
        )
        self._scheduled_tokens_i64 = np.empty(
            self.max_num_reqs,
            dtype=np.int64,
        )
        self._computed_tokens_i64 = np.empty_like(
            self._scheduled_tokens_i64
        )
        self._tokens_after_schedule = np.empty_like(
            self._scheduled_tokens_i64
        )
        self._first_logical_blocks = np.empty_like(
            self._scheduled_tokens_i64
        )
        self._completed_after_blocks = np.empty_like(
            self._scheduled_tokens_i64
        )
        self._completed_block_counts = np.empty_like(
            self._scheduled_tokens_i64
        )
        self._dump_boundary_mask = np.empty(
            self.max_num_reqs,
            dtype=np.bool_,
        )
        self._prefill_mask = np.empty_like(self._dump_boundary_mask)
        self._final_prefill_mask = np.empty_like(
            self._dump_boundary_mask
        )
        self._decode_row_mask = np.empty_like(self._dump_boundary_mask)
        self._decode_boundary_mask = np.empty_like(
            self._dump_boundary_mask
        )

        self.dram_store: DSAHotDRAMStore | None = None
        self.dsa_row_directory: DDRBlockTable | None = None
        self.packed_kv_layout: PackedKvLayout | None = None
        self.active_num_reqs = 0
        self.execution_num_reqs = 0
        self.dump_job_count = 0
        self.dump_launch_count = 0
        self._graph_capture_row_count = 0
        self._dram_table_row_count = 0
        self._dram_table_signature: tuple[int, int, int] | None = None
        self._defer_prefill_layer_dump = False
        self._resident_layer_caches: dict[
            int,
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._req_cache_tokens: torch.Tensor | None = None
        self._c8_key_dequant_scales: dict[int, torch.Tensor] = {}
        self._c8_attention_slots: dict[int, torch.Tensor] = {}
        self._c8_resident_seq_lengths: dict[int, torch.Tensor] = {}

    def bind_dram_store(self, store: DSAHotDRAMStore) -> None:
        if self.dram_store is not None:
            raise RuntimeError("DSA offload runtime DRAM store was bound twice")
        if store.storage_rows != self.resident_token_pool.storage_rows:
            raise RuntimeError(
                "DSA DRAM/resident pool row capacities differ: "
                f"dram={store.storage_rows}, "
                f"resident={self.resident_token_pool.storage_rows}"
            )
        if store.max_logical_blocks != self.max_logical_blocks:
            raise RuntimeError(
                "DSA DRAM/runtime logical widths differ: "
                f"dram={store.max_logical_blocks}, "
                f"runtime={self.max_logical_blocks}"
            )
        self.dram_store = store

    def register_packed_kv_layout(self, layout: PackedKvLayout) -> None:
        """Freeze the worker-lifetime packed layout used by KVAF."""

        if layout.tokens_per_logical_block != self.block_size:
            raise ValueError(
                "KVAF layout block size does not match DSA runtime: "
                f"layout={layout.tokens_per_logical_block}, "
                f"runtime={self.block_size}"
            )
        if self.packed_kv_layout is not None:
            if self.packed_kv_layout != layout:
                raise RuntimeError(
                    "DSA packed KV layout changed after it was frozen: "
                    f"old={self.packed_kv_layout}, new={layout}"
                )
            return
        self.packed_kv_layout = layout

    def initialize_dsa_row_index(
        self,
        *,
        layout: PackedKvLayout,
        ddr_block_count: int,
    ) -> DDRBlockTable:
        """Initialize KVAF and create its DSA_ROW directory view."""

        if layout.tokens_per_logical_block != self.block_size:
            raise ValueError(
                "KVAF layout block size does not match DSA runtime: "
                f"layout={layout.tokens_per_logical_block}, "
                f"runtime={self.block_size}"
            )
        self.register_packed_kv_layout(layout)
        if self.dsa_row_directory is not None:
            raise RuntimeError("DSA_ROW directory was initialized twice")
        ddr_block_count = int(ddr_block_count)
        if ddr_block_count <= 0:
            raise ValueError(
                f"ddr_block_count must be positive, got {ddr_block_count}"
            )
        self.kvaf_adapter.initialize_dsa_row_index(
            layout=layout,
            ddr_bytes=(
                ddr_block_count
                * self.kvaf_adapter.slot_size(layout)
            ),
        )
        self.dsa_row_directory = DDRBlockTable(
            self.row_index_geometry.directory_cell_count,
            device=self.device,
        )
        return self.dsa_row_directory

    def bind_resident_layer_cache(
        self,
        *,
        layer_id: int,
        resident_nope_cache: torch.Tensor,
        resident_rope_cache: torch.Tensor,
    ) -> None:
        layer_id = int(layer_id)
        if layer_id in self._resident_layer_caches:
            raise RuntimeError(
                f"DSA resident layer cache {layer_id} was bound twice"
            )
        self._resident_layer_caches[layer_id] = (
            resident_nope_cache,
            resident_rope_cache,
        )
        if self.is_c8:
            max_tail_tokens = DSA_REQUIRED_CACHE_BLOCK_SIZE
            attn_slot_width = DSA_SFA_COMPUTE_TOPK + max_tail_tokens
            self._c8_attention_slots[layer_id] = torch.empty(
                (self.max_num_reqs, 1, attn_slot_width),
                dtype=torch.int32,
                device=self.device,
            )
            self._c8_resident_seq_lengths[layer_id] = torch.empty(
                self.max_num_reqs,
                dtype=torch.int32,
                device=self.device,
            )

    def bind_c8_key_dequant_scale(
        self,
        *,
        layer_id: int,
        key_dequant_scale: torch.Tensor,
    ) -> None:
        layer_id = int(layer_id)
        self._c8_key_dequant_scales[layer_id] = key_dequant_scale

    @property
    def defer_prefill_layer_dump(self) -> bool:
        return self._defer_prefill_layer_dump

    def dump_prefill_full_blocks(self) -> None:
        """在完整 prefill model forward 结束后统一卸载所有层满块。"""

        if not self._defer_prefill_layer_dump:
            return
        try:
            job_count = self.dump_launch_count
            if job_count == 0:
                logger.info(
                    "[DSA_PREFILL_FULL_DUMP] SKIP_EMPTY_FINAL_PREFILL "
                    "reason=no_complete_block"
                )
                return
            logger.info(
                "[DSA_PREFILL_FULL_DUMP] BEGIN phase=after_model_forward "
                "job_count=%d bound_layers=%d expected_layers=%d",
                job_count,
                len(self._resident_layer_caches),
                self.num_layers,
            )
            store = self.dram_store
            if store is None:
                raise RuntimeError("DSA runtime has no bound DRAM store")
            if len(self._resident_layer_caches) != self.num_layers:
                raise RuntimeError(
                    "DSA prefill full-KV offload cache binding is incomplete: "
                    f"bound={len(self._resident_layer_caches)}, "
                    f"expected={self.num_layers}"
                )
            src_block_ids = self.dump_src_block_ids.gpu[:job_count]
            dst_block_ids = self.dump_dst_block_ids.gpu[:job_count]
            for layer_id in sorted(self._resident_layer_caches):
                resident_nope_cache, resident_rope_cache = (
                    self._resident_layer_caches[layer_id]
                )
                arenas = store.get_layer_arenas(layer_id)
                dump_full_kv_cache_blocks(
                    resident_nope_cache=resident_nope_cache,
                    resident_rope_cache=resident_rope_cache,
                    dram_nope_arena=arenas.nope,
                    dram_rope_arena=arenas.rope,
                    src_block_ids=src_block_ids,
                    dst_block_ids=dst_block_ids,
                )
            logger.info(
                "[DSA_PREFILL_FULL_DUMP] END phase=after_model_forward "
                "job_count=%d dumped_layers=%d",
                job_count,
                len(self._resident_layer_caches),
            )
        except Exception:
            logger.exception(
                "[DSA_PREFILL_FULL_DUMP] FAILED phase=after_model_forward"
            )
            raise
        finally:
            self._defer_prefill_layer_dump = False

    def release_pool_index(self, pool_index: int) -> None:
        store = self.dram_store
        if store is not None:
            store.release_pool_index(pool_index)

    def prepare_forward(
        self,
        *,
        input_batch: NPUInputBatch,
        state: DSAInputBatchCacheLayout,
        num_scheduled_tokens: np.ndarray,
        req_indices: torch.Tensor,
        positions: torch.Tensor,
        num_reqs: int,
        num_tokens: int,
        resident_group_id: int,
    ) -> torch.Tensor:
        """刷新本轮统一元数据并返回 resident slot-mapping positions。"""

        if self.dram_store is None:
            raise RuntimeError("DSA offload runtime has no DRAM store")
        if not state.valid or state.row_count != num_reqs:
            raise RuntimeError(
                "DSA InputBatch state is not ready for eager execution: "
                f"valid={state.valid}, rows={state.row_count}, "
                f"num_reqs={num_reqs}"
            )
        if num_tokens > self.max_num_tokens:
            raise RuntimeError(
                f"DSA token capacity exceeded: {num_tokens} > "
                f"{self.max_num_tokens}"
            )

        # 七列共用一个 owner，仅一次 H2D。
        state.copy_to_device(num_reqs)
        self.active_num_reqs = int(num_reqs)
        self.execution_num_reqs = 0
        self.dump_launch_count = 0
        self._defer_prefill_layer_dump = False
        self._req_cache_tokens = state.sparse_budget_tokens

        self._prepare_dump_plan(
            input_batch=input_batch,
            state=state,
            num_scheduled_tokens=num_scheduled_tokens,
            resident_group_id=resident_group_id,
        )
        dram_table_refreshed = self._refresh_active_dram_table(state)
        self._validate_sparse_dram_rows(
            input_batch=input_batch,
            state=state,
            validate_all_sparse=dram_table_refreshed,
        )

        active_modes = state.row_modes_cpu[:num_reqs]
        if not np.any(active_modes == DSA_ROW_MODE_SPARSE):
            return positions[:num_tokens]

        token_modes = self._token_row_modes[:num_tokens]
        token_budgets = self._token_sparse_budgets[:num_tokens]
        token_suffix_starts = self._token_persistent_suffix_starts[:num_tokens]
        torch.index_select(
            state.row_modes,
            0,
            req_indices[:num_tokens],
            out=token_modes,
        )
        torch.index_select(
            state.sparse_budget_tokens,
            0,
            req_indices[:num_tokens],
            out=token_budgets,
        )
        torch.index_select(
            state.persistent_suffix_start_tokens,
            0,
            req_indices[:num_tokens],
            out=token_suffix_starts,
        )
        resident_positions = self.resident_positions[:num_tokens]
        torch.sub(
            positions[:num_tokens],
            token_suffix_starts,
            out=resident_positions,
        )
        resident_positions.add_(token_budgets)
        torch.eq(
            token_modes,
            DSA_ROW_MODE_SPARSE,
            out=self._sparse_token_mask[:num_tokens],
        )
        torch.where(
            self._sparse_token_mask[:num_tokens],
            resident_positions,
            positions[:num_tokens],
            out=resident_positions,
        )
        return resident_positions

    @property
    def graph_capture_row_count(self) -> int:
        return self._graph_capture_row_count

    def prepare_execution_view(
        self,
        *,
        active_num_reqs: int,
        graph_row_count: int | None,
    ) -> int:
        if self._graph_capture_row_count:
            raise RuntimeError(
                "DSA real execution cannot reuse an active capture dummy"
            )
        active_num_reqs = int(active_num_reqs)
        if active_num_reqs != self.active_num_reqs:
            raise RuntimeError(
                "DSA runtime active rows changed before execution view "
                f"was finalized: prepared={self.active_num_reqs}, "
                f"requested={active_num_reqs}"
            )

        if graph_row_count is None:
            execution_num_reqs = active_num_reqs
            launch_count = self.dump_job_count
        else:
            execution_num_reqs = int(graph_row_count)
            if not (
                active_num_reqs
                <= execution_num_reqs
                <= self.max_num_reqs
            ):
                raise RuntimeError(
                    "DSA graph rows must cover active rows and fit runtime "
                    f"capacity: active={active_num_reqs}, "
                    f"graph={execution_num_reqs}, "
                    f"capacity={self.max_num_reqs}"
                )
            if self.dump_job_count > active_num_reqs:
                raise RuntimeError(
                    "DSA single-token graph step produced more dump jobs "
                    f"than active rows: jobs={self.dump_job_count}, "
                    f"rows={active_num_reqs}"
                )
            launch_count = execution_num_reqs
            if self.dump_job_count < launch_count:
                tail = slice(self.dump_job_count, launch_count)
                self.dump_src_block_ids.np[tail] = 0
                self.dump_dst_block_ids.np[tail] = (
                    DSA_DUMP_NOOP_DST_BLOCK_ID
                )

        if launch_count:
            self.dump_src_block_ids.gpu[:launch_count].copy_(
                self.dump_src_block_ids.cpu[:launch_count],
                non_blocking=True,
            )
            self.dump_dst_block_ids.gpu[:launch_count].copy_(
                self.dump_dst_block_ids.cpu[:launch_count],
                non_blocking=True,
            )
        self.execution_num_reqs = execution_num_reqs
        self.dump_launch_count = int(launch_count)
        return execution_num_reqs

    def prepare_graph_capture(
        self,
        *,
        row_count: int,
        cache_tokens: torch.Tensor | None = None,
    ) -> None:
        if self.dram_store is None:
            raise RuntimeError(
                "DSA graph capture requires an initialized DRAM store"
            )
        row_count = int(row_count)
        if self._graph_capture_row_count:
            raise RuntimeError(
                "DSA runtime graph-capture state was installed twice"
            )
        if not 0 < row_count <= self.max_num_reqs:
            raise ValueError(
                "DSA runtime graph-capture row count is outside capacity: "
                f"rows={row_count}, capacity={self.max_num_reqs}"
            )

        self._graph_capture_row_count = row_count
        self._req_cache_tokens = cache_tokens
        try:
            self.active_dram_block_table.np[:row_count].fill(
                DSA_DRAM_NULL_BLOCK_ID
            )
            self.active_dram_block_table.gpu[:row_count].zero_()
            self.dump_src_block_ids.np[:row_count] = 0
            self.dump_dst_block_ids.np[:row_count] = (
                DSA_DUMP_NOOP_DST_BLOCK_ID
            )
            self.dump_src_block_ids.gpu[:row_count].zero_()
            self.dump_dst_block_ids.gpu[:row_count].fill_(
                DSA_DUMP_NOOP_DST_BLOCK_ID
            )
            self.active_num_reqs = row_count
            self.execution_num_reqs = row_count
            self.dump_job_count = 0
            self.dump_launch_count = row_count
        except Exception:
            try:
                self.restore_after_graph_capture()
            except Exception:
                self.active_num_reqs = 0
                self.execution_num_reqs = 0
                self.dump_job_count = 0
                self.dump_launch_count = 0
                self._graph_capture_row_count = 0
            raise

    def restore_after_graph_capture(self) -> None:
        if not self._graph_capture_row_count:
            return
        self.active_num_reqs = 0
        self.execution_num_reqs = 0
        self.dump_job_count = 0
        self.dump_launch_count = 0
        self._graph_capture_row_count = 0
        self._dram_table_row_count = 0
        self._dram_table_signature = None
        self._req_cache_tokens = None

    def _prepare_dump_plan(
        self,
        *,
        input_batch: NPUInputBatch,
        state: DSAInputBatchCacheLayout,
        num_scheduled_tokens: np.ndarray,
        resident_group_id: int,
    ) -> None:
        """构造紧凑 dump jobs；keep 模式仅抑制 decode dump。"""

        store = self.dram_store
        assert store is not None
        num_reqs = self.active_num_reqs
        scheduled = self._scheduled_tokens_i64[:num_reqs]
        computed = self._computed_tokens_i64[:num_reqs]
        tokens_after = self._tokens_after_schedule[:num_reqs]
        first_logical = self._first_logical_blocks[:num_reqs]
        completed_after = self._completed_after_blocks[:num_reqs]
        completed_counts = self._completed_block_counts[:num_reqs]
        boundary_mask = self._dump_boundary_mask[:num_reqs]
        prefill_mask = self._prefill_mask[:num_reqs]
        final_prefill_mask = self._final_prefill_mask[:num_reqs]
        decode_mask = self._decode_row_mask[:num_reqs]
        decode_boundary_mask = self._decode_boundary_mask[:num_reqs]
        has_prefill = False

        np.copyto(
            scheduled,
            num_scheduled_tokens[:num_reqs],
            casting="unsafe",
        )
        np.copyto(
            computed,
            input_batch.num_computed_tokens_cpu[:num_reqs],
            casting="unsafe",
        )
        np.floor_divide(
            computed,
            self.block_size,
            out=first_logical,
        )
        np.add(computed, scheduled, out=tokens_after)
        np.floor_divide(
            tokens_after,
            self.block_size,
            out=completed_after,
        )
        np.subtract(
            completed_after,
            first_logical,
            out=completed_counts,
        )
        if self.enable_prefill_full_kv_offload:
            np.equal(
                state.stages_cpu[:num_reqs],
                int(DSARequestCacheStage.PREFILL),
                out=prefill_mask,
            )
            np.greater_equal(
                tokens_after,
                input_batch.num_prompt_tokens[:num_reqs],
                out=final_prefill_mask,
            )
            np.logical_and(
                final_prefill_mask,
                prefill_mask,
                out=final_prefill_mask,
            )
            completed_counts[prefill_mask] = 0
            first_logical[final_prefill_mask] = 0
            completed_counts[final_prefill_mask] = completed_after[
                final_prefill_mask
            ]
            self._defer_prefill_layer_dump = bool(
                np.any(final_prefill_mask)
            )
            has_prefill = bool(np.any(prefill_mask))
            if has_prefill and logger.isEnabledFor(logging.INFO):
                logger.info(
                    "[DSA_PREFILL_FULL_DUMP] PREFILL_PLAN_DECISION "
                    "prefill_rows=%s final_prefill_rows=%s defer=%s "
                    "computed_tokens=%s scheduled_tokens=%s "
                    "prompt_tokens=%s tokens_after=%s",
                    np.flatnonzero(prefill_mask).tolist(),
                    np.flatnonzero(final_prefill_mask).tolist(),
                    self._defer_prefill_layer_dump,
                    computed[prefill_mask].tolist(),
                    scheduled[prefill_mask].tolist(),
                    input_batch.num_prompt_tokens[:num_reqs][
                        prefill_mask
                    ].tolist(),
                    tokens_after[prefill_mask].tolist(),
                )

        np.greater_equal(
            state.stages_cpu[:num_reqs],
            int(DSARequestCacheStage.DENSE_DECODE),
            out=decode_mask,
        )
        np.greater(completed_counts, 0, out=decode_boundary_mask)
        np.logical_and(
            decode_boundary_mask,
            decode_mask,
            out=decode_boundary_mask,
        )
        if self.keep_decode_kv_in_hbm and np.any(decode_boundary_mask):
            boundary_rows = np.flatnonzero(decode_boundary_mask)
            completed_counts[decode_mask] = 0
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "[DSA_DECODE_KV_POLICY] action=KEEP_HBM rows=%s "
                    "dump_job_count=0 computed_tokens=%s "
                    "scheduled_tokens=%s resident_valid_tokens=%s",
                    boundary_rows.tolist(),
                    computed[boundary_rows].tolist(),
                    scheduled[boundary_rows].tolist(),
                    state.resident_valid_tokens_cpu[
                        boundary_rows
                    ].tolist(),
                )

        np.greater(completed_counts, 0, out=boundary_mask)
        if not np.any(boundary_mask):
            self.dump_job_count = 0
            if self.enable_prefill_full_kv_offload and has_prefill:
                reason = (
                    "final_prefill_without_complete_block"
                    if self._defer_prefill_layer_dump
                    else "intermediate_prefill"
                )
                logger.info(
                    "[DSA_PREFILL_FULL_DUMP] NO_DUMP_PLAN reason=%s",
                    reason,
                )
            return
        boundary_rows = np.flatnonzero(boundary_mask)

        resident_table = input_batch.block_table[int(resident_group_id)]
        resident_blocks = resident_table.get_numpy_array()
        resident_row_widths = resident_table.num_blocks_per_row
        job_count = 0
        for row_value in boundary_rows:
            row = int(row_value)
            count = int(completed_counts[row])
            logical_start = int(first_logical[row])
            logical_end = logical_start + count
            next_job_count = job_count + count
            if next_job_count > self._dump_pool_indices.size:
                raise RuntimeError(
                    "DSA full-block dump job capacity exceeded: "
                    f"required={next_job_count}, "
                    f"capacity={self._dump_pool_indices.size}"
                )

            jobs = slice(job_count, next_job_count)
            self._dump_pool_indices[jobs] = (
                state.resident_pool_indices_cpu[row]
            )
            self._dump_logical_indices[jobs] = self._logical_block_indices[
                logical_start:logical_end
            ]
            if state.row_modes_cpu[row] == DSA_ROW_MODE_SPARSE:
                if count != 1:
                    raise RuntimeError(
                        "DSA sparse decode completed more than one full "
                        f"block in one step: row={row}, count={count}"
                    )
                tail_column = int(resident_row_widths[row]) - 1
                if tail_column < 0:
                    raise RuntimeError(
                        f"DSA sparse row {row} has no resident tail block"
                    )
                self._dump_source_blocks[jobs] = resident_blocks[
                    row,
                    tail_column,
                ]
            else:
                if logical_end > int(resident_row_widths[row]):
                    raise RuntimeError(
                        "DSA dense dump references an unallocated resident "
                        f"block: row={row}, logical_end={logical_end}, "
                        f"resident_blocks={int(resident_row_widths[row])}"
                    )
                self._dump_source_blocks[jobs] = resident_blocks[
                    row,
                    logical_start:logical_end,
                ]
            job_count = next_job_count

        if job_count == 0:
            self.dump_job_count = 0
            return

        reservation = store.reserve_blocks(
            pool_indices=self._dump_pool_indices[:job_count],
            logical_block_indices=self._dump_logical_indices[:job_count],
        )
        new_rows = np.flatnonzero(reservation.new_mask)
        copy_count = int(new_rows.size)
        if copy_count:
            np.take(
                self._dump_source_blocks[:job_count],
                new_rows,
                out=self.dump_src_block_ids.np[:copy_count],
            )
            np.take(
                reservation.physical_block_ids,
                new_rows,
                out=self.dump_dst_block_ids.np[:copy_count],
            )
        self.dump_job_count = copy_count
        if (
            not self.keep_decode_kv_in_hbm
            and np.any(decode_boundary_mask)
            and logger.isEnabledFor(logging.INFO)
        ):
            logger.info(
                "[DSA_DECODE_KV_POLICY] action=DUMP_DRAM rows=%s "
                "dump_job_count=%d computed_tokens=%s "
                "scheduled_tokens=%s",
                np.flatnonzero(decode_boundary_mask).tolist(),
                copy_count,
                computed[decode_boundary_mask].tolist(),
                scheduled[decode_boundary_mask].tolist(),
            )
        if (
            self._defer_prefill_layer_dump
            and logger.isEnabledFor(logging.INFO)
        ):
            logger.info(
                "[DSA_PREFILL_FULL_DUMP] FINAL_PREFILL_PLAN_READY "
                "planned_jobs=%d new_dump_jobs=%d final_prefill_rows=%s",
                job_count,
                copy_count,
                np.flatnonzero(final_prefill_mask).tolist(),
            )

    def _refresh_active_dram_table(
        self,
        state: DSAInputBatchCacheLayout,
    ) -> bool:
        store = self.dram_store
        assert store is not None
        signature = (
            int(state.mapping_version),
            int(store.table_version),
            self.active_num_reqs,
        )
        if signature == self._dram_table_signature:
            return False
        previous_row_count = self._dram_table_row_count
        active_cpu = self.active_dram_block_table.np[
            : self.active_num_reqs
        ]
        store.gather_rows(
            pool_indices=state.resident_pool_indices_cpu[
                : self.active_num_reqs
            ],
            output=active_cpu,
        )
        if self.active_num_reqs < previous_row_count:
            self.active_dram_block_table.np[
                self.active_num_reqs:previous_row_count
            ].fill(DSA_DRAM_NULL_BLOCK_ID)
        copy_row_count = max(
            self.active_num_reqs,
            previous_row_count,
        )
        self.active_dram_block_table.gpu[
            :copy_row_count
        ].copy_(
            self.active_dram_block_table.cpu[
                :copy_row_count
            ],
            non_blocking=True,
        )
        self._dram_table_row_count = self.active_num_reqs
        self._dram_table_signature = signature
        return True

    def _validate_sparse_dram_rows(
        self,
        *,
        input_batch: NPUInputBatch,
        state: DSAInputBatchCacheLayout,
        validate_all_sparse: bool,
    ) -> None:
        num_reqs = self.active_num_reqs
        stages = state.stages_cpu[:num_reqs]
        if validate_all_sparse:
            rows = np.flatnonzero(
                stages >= int(
                    DSARequestCacheStage.ENTER_SPARSE_DECODE
                )
            )
        else:
            rows = np.flatnonzero(
                stages == int(
                    DSARequestCacheStage.ENTER_SPARSE_DECODE
                )
            )
        if rows.size == 0:
            return

        dram_table = self.active_dram_block_table.np
        for row_value in rows:
            row = int(row_value)
            if self.keep_decode_kv_in_hbm:
                required_blocks = (
                    int(
                        state.persistent_suffix_start_tokens_cpu[row]
                    )
                    // self.block_size
                )
            else:
                required_blocks = (
                    max(0, int(self._tokens_after_schedule[row]) - 1)
                    // self.block_size
                )
            if required_blocks == 0:
                continue
            missing = np.flatnonzero(
                dram_table[row, :required_blocks]
                == DSA_DRAM_NULL_BLOCK_ID
            )
            if missing.size:
                request_id = input_batch.req_ids[row]
                table_kind = (
                    "prompt"
                    if self.keep_decode_kv_in_hbm
                    else "decode-history"
                )
                raise RuntimeError(
                    "DSA sparse decode has an incomplete DRAM block table: "
                    f"request_id={request_id!r}, batch_row={row}, "
                    f"table_kind={table_kind}, "
                    "resident_pool_row="
                    f"{int(state.resident_pool_indices_cpu[row])}, "
                    f"required_blocks={required_blocks}, "
                    f"first_missing_logical_block={int(missing[0])}. "
                    "Refusing to let KSC read the null DRAM block."
                )

    def get_lidu_outputs(
        self,
        *,
        num_reqs: int,
    ) -> DSALightningIndexerOutputs:
        num_reqs = int(num_reqs)
        view = self._lidu_output_views.get(num_reqs)
        if view is None:
            if not 0 < num_reqs <= self.max_num_reqs:
                raise RuntimeError(
                    f"DSA LIDU row count {num_reqs} is outside capacity"
                )
            view = DSALightningIndexerOutputs(
                topk_index=self._lidu_topk_index[:num_reqs],
                topk_slots=self._lidu_topk_slots[:num_reqs],
                miss_count=self._lidu_miss_count[:num_reqs],
                tail_info=self._lidu_tail_info[:num_reqs],
            )
            self._lidu_output_views[num_reqs] = view
        return view


@dataclass(frozen=True)
class DSALayerOffloadContext:
    """绑定到一个 ``AscendSFAImpl`` 的逐层稳定资源。"""

    layer_id: int
    indexer_cache: torch.Tensor
    runtime: DSAOffloadRuntime

    def execute_decode_selection(
        self,
        *,
        query: torch.Tensor,
        weights: torch.Tensor,
        row_modes: torch.Tensor,
        resident_pool_indices: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        indexer_block_table: torch.Tensor,
        resident_nope_cache: torch.Tensor,
        resident_rope_cache: torch.Tensor,
        resident_block_table: torch.Tensor,
        dram_block_table: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor | None = None,
        query_dequant_scale: torch.Tensor | None = None,
    ) -> DSAOffloadSelectionOutput:
        num_reqs = int(query.shape[0])
        outputs = self.runtime.get_lidu_outputs(
            num_reqs=num_reqs,
        )
        cache_slots = self.runtime.resident_token_pool.get_cache_slots(
            self.layer_id
        )
        if (
            self.runtime.op_backend == DSA_OP_BACKEND_NANOVLLM_A5
            and self.runtime.is_c8
        ):
            from vllm_ascend.dsa_offload.ops_nanovllm import (
                lightning_indexer_decode_update_c8 as lidu_c8,
                packed_kvcache_scatter_copy as ksc_c8,
            )
            cache_tokens = self.runtime._req_cache_tokens[:num_reqs]
            key_dequant_scale = self.runtime._c8_key_dequant_scales.get(
                self.layer_id
            )
            if key_dequant_scale is None:
                raise RuntimeError(
                    f"DSA C8 key dequant scale not bound for layer "
                    f"{self.layer_id}"
                )
            lidu_c8(
                query=query,
                key=self.indexer_cache,
                weights=weights,
                query_dequant_scale=query_dequant_scale,
                key_dequant_scale=key_dequant_scale,
                actual_seq_lengths_query=actual_seq_lengths_query,
                req_pool_entries=resident_pool_indices,
                cache_slots=cache_slots,
                cache_tokens=cache_tokens,
                candidate_lens=actual_seq_lengths_key,
                block_table=indexer_block_table,
                outputs=outputs,
            )
            store = self.runtime.dram_store
            if store is None:
                raise RuntimeError("DSA layer has no bound DRAM store")
            arenas = store.get_layer_arenas(self.layer_id)
            attention_slots = self.runtime._c8_attention_slots[
                self.layer_id
            ][:num_reqs]
            resident_seq_lengths = (
                self.runtime._c8_resident_seq_lengths[self.layer_id][
                    :num_reqs
                ]
            )
            ksc_c8(
                hbm_packed_kv=resident_nope_cache,
                dram_packed_kv=arenas.nope,
                hbm_block_table=resident_block_table,
                dram_block_table=dram_block_table,
                source_token_ids=outputs.topk_index,
                destination_slots=outputs.topk_slots,
                copy_counts=outputs.miss_count,
                cache_tokens=cache_tokens,
                candidate_lens=actual_seq_lengths_key,
                actual_seq_lengths_kv=actual_seq_lengths_key,
                max_tail_tokens=DSA_REQUIRED_CACHE_BLOCK_SIZE,
                attention_slots=attention_slots,
                resident_seq_lengths=resident_seq_lengths,
            )
            return DSAOffloadSelectionOutput(
                sparse_indices=attention_slots,
                tail_info=resident_seq_lengths,
            )
        if self.runtime.op_backend == DSA_OP_BACKEND_NANOVLLM_A5:
            from vllm_ascend.dsa_offload.ops_nanovllm import (
                kvcache_scatter_copy as ksc_nv,
                lightning_indexer_decode_update as lidu_nv,
            )
            cache_tokens = self.runtime._req_cache_tokens[:num_reqs]
            lidu_nv(
                query=query,
                key=self.indexer_cache,
                weights=weights,
                req_pool_entries=resident_pool_indices,
                cache_slots=cache_slots,
                row_modes=row_modes,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=indexer_block_table,
                outputs=outputs,
                cache_tokens=cache_tokens,
            )
        else:
            lightning_indexer_decode_update(
                query=query,
                key=self.indexer_cache,
                weights=weights,
                req_pool_entries=resident_pool_indices,
                cache_slots=cache_slots,
                row_modes=row_modes,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=indexer_block_table,
                outputs=outputs,
            )
        store = self.runtime.dram_store
        if store is None:
            raise RuntimeError("DSA layer has no bound DRAM store")
        arenas = store.get_layer_arenas(self.layer_id)
        if self.runtime.op_backend == DSA_OP_BACKEND_NANOVLLM_A5:
            ksc_nv(
                resident_nope_cache=resident_nope_cache,
                resident_rope_cache=resident_rope_cache,
                dram_nope_arena=arenas.nope,
                dram_rope_arena=arenas.rope,
                resident_block_table=resident_block_table,
                dram_block_table=dram_block_table,
                src_token_ids=outputs.topk_index,
                dst_slots=outputs.topk_slots,
                copy_counts=outputs.miss_count,
            )
        else:
            kvcache_scatter_copy(
                resident_nope_cache=resident_nope_cache,
                resident_rope_cache=resident_rope_cache,
                dram_nope_arena=arenas.nope,
                dram_rope_arena=arenas.rope,
                resident_block_table=resident_block_table,
                dram_block_table=dram_block_table,
                src_token_ids=outputs.topk_index,
                dst_slots=outputs.topk_slots,
                copy_counts=outputs.miss_count,
            )
        return DSAOffloadSelectionOutput(
            sparse_indices=outputs.topk_slots,
            tail_info=outputs.tail_info,
        )

    def execute_sparse_attention(
        self,
        *,
        query: torch.Tensor,
        resident_nope_cache: torch.Tensor,
        selection: DSAOffloadSelectionOutput,
        scale_value: float,
        resident_block_table: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        query_rope: torch.Tensor,
        resident_rope_cache: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.runtime.op_backend == DSA_OP_BACKEND_NANOVLLM_A5
            and self.runtime.is_c8
        ):
            from vllm_ascend.dsa_offload.ops_nanovllm import (
                sparse_flash_attention_for_offload_c8 as sfa_c8,
            )
            c8_query = torch.cat([query, query_rope], dim=-1)
            return sfa_c8(
                query=c8_query,
                packed_kv=resident_nope_cache,
                attention_slots=selection.sparse_indices,
                block_table=resident_block_table,
                actual_seq_lengths_query=actual_seq_lengths_query,
                resident_seq_lengths=selection.tail_info,
                scale_value=scale_value,
            )
        if self.runtime.op_backend == DSA_OP_BACKEND_NANOVLLM_A5:
            from vllm_ascend.dsa_offload.ops_nanovllm import (
                sparse_flash_attention_for_offload as sfa_nv,
            )
            num_reqs = int(query.shape[0])
            cache_tokens = self.runtime._req_cache_tokens[:num_reqs]
            return sfa_nv(
                query=query,
                key=resident_nope_cache,
                sparse_indices=selection.sparse_indices,
                tail_info=selection.tail_info,
                scale_value=scale_value,
                block_table=resident_block_table,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                query_rope=query_rope,
                key_rope=resident_rope_cache,
                cache_tokens=cache_tokens,
            )
        return sparse_flash_attention_for_offload(
            query=query,
            key=resident_nope_cache,
            sparse_indices=selection.sparse_indices,
            tail_info=selection.tail_info,
            scale_value=scale_value,
            block_table=resident_block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=query_rope,
            key_rope=resident_rope_cache,
        )

    def dump_full_blocks(
        self,
        *,
        resident_nope_cache: torch.Tensor,
        resident_rope_cache: torch.Tensor,
    ) -> None:
        if self.runtime.defer_prefill_layer_dump:
            return
        job_count = self.runtime.dump_launch_count
        if job_count == 0:
            return
        store = self.runtime.dram_store
        if store is None:
            raise RuntimeError("DSA layer has no bound DRAM store")
        arenas = store.get_layer_arenas(self.layer_id)
        dump_full_kv_cache_blocks(
            resident_nope_cache=resident_nope_cache,
            resident_rope_cache=resident_rope_cache,
            dram_nope_arena=arenas.nope,
            dram_rope_arena=arenas.rope,
            src_block_ids=self.runtime.dump_src_block_ids.gpu[:job_count],
            dst_block_ids=self.runtime.dump_dst_block_ids.gpu[:job_count],
        )
