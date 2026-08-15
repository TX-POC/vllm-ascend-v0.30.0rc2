# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 W4A4C8 DSA runtime policy installed on top of the shared DSA lifecycle.

The scheduler/request lifecycle remains the single source of truth.  This
module only specializes the A5 C8 data plane:

* Decode keep-HBM means the sparse candidate range stops at the persistent
  suffix start, so newly generated target KV never becomes a DRAM candidate.
* MTP draft KV remains outside the DSA target-resident dump planner.
* candidate/logical/resident lengths use worker-lifetime fixed-address buffers
  shared by eager and FULL decode graph replay.
* C8 sparse+tail metadata uses one shared graph-stable scratch instead of one
  max-model-length allocation per transformer layer.
* DSA resident MLA keeps the legacy 512+64 physical backing but exposes a
  zero-copy packed 656-byte FP8 view to every A5 C8 consumer.
* all A5 C8 full-block dump paths use the packed 656-byte dump operator.

This is deliberately an adapter rather than a second scheduler/runtime.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from vllm.logger import logger
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.dsa_offload.contracts import (
    DSA_OP_BACKEND_NANOVLLM_A5,
    DSA_PACKED_KV_DIM,
    DSA_REQUIRED_CACHE_BLOCK_SIZE,
    DSA_SFA_COMPUTE_TOPK,
)
from vllm_ascend.dsa_offload.request_cache_layout import DSARequestCacheStage

_INSTALLED = False


def _is_a5_c8(runtime: Any) -> bool:
    return (
        runtime.op_backend == DSA_OP_BACKEND_NANOVLLM_A5
        and bool(runtime.is_c8)
    )


def _runner_is_a5_c8(runner: Any) -> bool:
    runtime = getattr(runner, "dsa_offload_runtime", None)
    return runtime is not None and _is_a5_c8(runtime)


def _runtime_pin_memory(runtime: Any) -> bool:
    cpu = getattr(runtime.active_dram_block_table, "cpu", None)
    return bool(cpu is not None and hasattr(cpu, "is_pinned") and cpu.is_pinned())


def _make_packed_resident_view(
    resident_cache: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose the A5 packed C8 ABI without allocating another resident cache.

    The DSA resident spec intentionally keeps the legacy BF16 split backing so
    scheduler/page accounting and the existing two-plane binder stay unchanged.
    The NOPE plane alone has 1024 bytes/token (512 BF16 values), which is more
    than the 656 bytes/token required by the A5 packed C8 ABI.  Reinterpret the
    backing once during KV-cache initialization and use that packed view for
    MLAPO/KSC/SFA/dump.  The ROPE plane remains only as a compatibility plane for
    the shared DSA binder/DRAM-store contract and is ignored by packed kernels.
    """

    if len(resident_cache) < 2:
        raise RuntimeError(
            "A5 C8 DSA resident cache requires the legacy two-plane backing "
            f"before packed reinterpretation, got len={len(resident_cache)}"
        )
    nope, rope = resident_cache[0], resident_cache[1]
    if nope.ndim != 4:
        raise RuntimeError(
            "A5 C8 DSA resident NOPE backing must be 4D: "
            f"shape={tuple(nope.shape)}"
        )
    blocks, block_size, heads, _ = map(int, nope.shape)
    if block_size != DSA_REQUIRED_CACHE_BLOCK_SIZE or heads != 1:
        raise RuntimeError(
            "A5 C8 DSA resident backing requires [blocks,128,1,*]: "
            f"shape={tuple(nope.shape)}"
        )
    if not nope.is_contiguous():
        raise RuntimeError(
            "A5 C8 DSA resident NOPE backing must be contiguous before "
            "packed reinterpretation"
        )

    if (
        nope.dtype == torch.float8_e4m3fn
        and int(nope.shape[-1]) == DSA_PACKED_KV_DIM
    ):
        return nope, rope

    if nope.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            "A5 C8 DSA resident backing must be BF16/FP16 or already packed "
            f"FP8, got dtype={nope.dtype}"
        )

    raw = nope.view(torch.int8).view(-1)
    required_bytes = (
        blocks
        * block_size
        * heads
        * DSA_PACKED_KV_DIM
    )
    if raw.numel() < required_bytes:
        raise RuntimeError(
            "Legacy DSA resident backing is too small for the A5 packed ABI: "
            f"available={raw.numel()} bytes, required={required_bytes} bytes"
        )

    packed = (
        raw[:required_bytes]
        .view(torch.float8_e4m3fn)
        .view(
            blocks,
            block_size,
            heads,
            DSA_PACKED_KV_DIM,
        )
    )
    if not packed.is_contiguous():
        raise RuntimeError("A5 C8 packed resident KV view must be contiguous")
    return packed, rope


def _ensure_fixed_metadata(runtime: Any) -> None:
    """Allocate graph-stable A5 C8 metadata once per worker runtime."""

    if hasattr(runtime, "_a5_candidate_lens"):
        return

    pin_memory = _runtime_pin_memory(runtime)
    runtime._a5_candidate_lens = CpuGpuBuffer(
        runtime.max_num_reqs,
        dtype=torch.int32,
        device=runtime.device,
        pin_memory=pin_memory,
    )
    runtime._a5_logical_seq_lengths = CpuGpuBuffer(
        runtime.max_num_reqs,
        dtype=torch.int32,
        device=runtime.device,
        pin_memory=pin_memory,
    )
    runtime._a5_resident_seq_lengths = CpuGpuBuffer(
        runtime.max_num_reqs,
        dtype=torch.int32,
        device=runtime.device,
        pin_memory=pin_memory,
    )
    for buffer in (
        runtime._a5_candidate_lens,
        runtime._a5_logical_seq_lengths,
        runtime._a5_resident_seq_lengths,
    ):
        buffer.np.fill(0)
        buffer.gpu.zero_()

    # Correctness-first graph contract: keep one long sparse+tail scratch for
    # the whole model.  Layers execute serially on one captured stream, so this
    # avoids N_layers * max_model_len metadata while supporting a Decode suffix
    # that grows beyond one 128-token block.
    runtime._a5_c8_max_tail_tokens = int(runtime.max_model_len)
    runtime._a5_c8_attention_slots = torch.empty(
        (
            runtime.max_num_reqs,
            1,
            DSA_SFA_COMPUTE_TOPK + runtime._a5_c8_max_tail_tokens,
        ),
        dtype=torch.int32,
        device=runtime.device,
    )
    runtime._a5_c8_last_range_log: tuple[Any, ...] | None = None

    logger.info_once(
        "[DSA_A5_C8_POLICY] installed=true backend=nanovllm_a5 "
        "fixed_graph_metadata=true sparse_topk=%d max_tail_tokens=%d "
        "packed_resident_dim=%d zero_copy_resident=true "
        "mtp_draft_dump=false target_dump_policy=keep_decode_kv_in_hbm",
        DSA_SFA_COMPUTE_TOPK,
        runtime._a5_c8_max_tail_tokens,
        DSA_PACKED_KV_DIM,
    )


def _refresh_fixed_metadata(
    runtime: Any,
    *,
    state: Any,
    num_reqs: int,
) -> None:
    _ensure_fixed_metadata(runtime)
    num_reqs = int(num_reqs)
    logical = runtime._a5_logical_seq_lengths.np[:num_reqs]
    candidate = runtime._a5_candidate_lens.np[:num_reqs]
    resident = runtime._a5_resident_seq_lengths.np[:num_reqs]

    # _prepare_dump_plan() has already populated _tokens_after_schedule when
    # prepare_forward() returns, so no second request-length calculation is
    # needed here.
    np.copyto(
        logical,
        runtime._tokens_after_schedule[:num_reqs],
        casting="unsafe",
    )
    np.copyto(candidate, logical)
    np.copyto(resident, logical)

    stages = state.stages_cpu[:num_reqs]
    sparse_rows = np.flatnonzero(
        stages >= int(DSARequestCacheStage.ENTER_SPARSE_DECODE)
    )
    if sparse_rows.size:
        candidate[sparse_rows] = (
            state.persistent_suffix_start_tokens_cpu[sparse_rows]
        )
        resident[sparse_rows] = state.resident_valid_tokens_cpu[sparse_rows]

    runtime._a5_candidate_lens.gpu[:num_reqs].copy_(
        runtime._a5_candidate_lens.cpu[:num_reqs],
        non_blocking=True,
    )
    runtime._a5_logical_seq_lengths.gpu[:num_reqs].copy_(
        runtime._a5_logical_seq_lengths.cpu[:num_reqs],
        non_blocking=True,
    )
    runtime._a5_resident_seq_lengths.gpu[:num_reqs].copy_(
        runtime._a5_resident_seq_lengths.cpu[:num_reqs],
        non_blocking=True,
    )

    if sparse_rows.size and logger.isEnabledFor(logging.INFO):
        # Log ENTER_SPARSE and full-block boundaries only.  Avoid one info log
        # per generated token in the steady Decode hot path.
        interesting = sparse_rows[
            np.logical_or(
                stages[sparse_rows]
                == int(DSARequestCacheStage.ENTER_SPARSE_DECODE),
                np.remainder(logical[sparse_rows], runtime.block_size) == 0,
            )
        ]
        if interesting.size:
            signature = (
                tuple(interesting.tolist()),
                tuple(candidate[interesting].tolist()),
                tuple(logical[interesting].tolist()),
                tuple(resident[interesting].tolist()),
            )
            if signature != runtime._a5_c8_last_range_log:
                logger.info(
                    "[DSA_A5_C8_RANGE] rows=%s candidate_lens=%s "
                    "logical_seq_lens=%s resident_seq_lens=%s "
                    "keep_decode_kv_in_hbm=%s",
                    interesting.tolist(),
                    candidate[interesting].tolist(),
                    logical[interesting].tolist(),
                    resident[interesting].tolist(),
                    runtime.keep_decode_kv_in_hbm,
                )
                runtime._a5_c8_last_range_log = signature


def install_a5_c8_runtime_policy() -> None:
    """Patch only A5 C8 data-plane methods; preserve the shared lifecycle."""

    global _INSTALLED
    if _INSTALLED:
        return

    import vllm_ascend.worker.model_runner_v1 as runner_mod
    from vllm_ascend.dsa_offload.kv_cache import DSAResidentMLAAttentionSpec
    from vllm_ascend.dsa_offload.ops import DSAOffloadSelectionOutput
    from vllm_ascend.dsa_offload.runtime import (
        DSALayerOffloadContext,
        DSAOffloadRuntime,
    )

    NPUModelRunner = runner_mod.NPUModelRunner
    original_reshape = NPUModelRunner._reshape_kv_cache_tensors
    original_bind_resident = DSAOffloadRuntime.bind_resident_layer_cache
    original_prepare_forward = DSAOffloadRuntime.prepare_forward
    original_prepare_graph_capture = DSAOffloadRuntime.prepare_graph_capture
    original_dump_prefill = DSAOffloadRuntime.dump_prefill_full_blocks
    original_execute_decode = DSALayerOffloadContext.execute_decode_selection
    original_dump_full_blocks = DSALayerOffloadContext.dump_full_blocks

    def reshape_kv_cache_tensors(
        self: Any,
        kv_cache_config: Any,
        kv_cache_raw_tensors: Any,
    ) -> dict[str, Any]:
        caches = original_reshape(
            self,
            kv_cache_config,
            kv_cache_raw_tensors,
        )
        if not _runner_is_a5_c8(self):
            return caches

        specs = self._get_layer_kv_cache_specs(kv_cache_config)
        for layer_name, spec in specs.items():
            if not isinstance(spec, DSAResidentMLAAttentionSpec):
                continue
            cache = caches.get(layer_name)
            if not isinstance(cache, tuple):
                raise RuntimeError(
                    "A5 C8 DSA resident cache must be a tuple before packed "
                    f"reinterpretation: layer={layer_name}, "
                    f"type={type(cache).__name__}"
                )
            packed, unused_rope = _make_packed_resident_view(cache)
            caches[layer_name] = (packed, unused_rope)
            logger.info(
                "[DSA_C8_RESIDENT_LAYOUT] layer=%s packed_shape=%s "
                "packed_dtype=%s compatibility_rope_shape=%s "
                "zero_copy_backing=true",
                layer_name,
                tuple(packed.shape),
                packed.dtype,
                tuple(unused_rope.shape),
            )
        return caches

    def bind_resident_layer_cache(self: Any, **kwargs: Any) -> None:
        if _is_a5_c8(self):
            packed = kwargs.get("resident_nope_cache")
            if (
                not isinstance(packed, torch.Tensor)
                or packed.dtype != torch.float8_e4m3fn
                or packed.ndim != 4
                or int(packed.shape[-1]) != DSA_PACKED_KV_DIM
            ):
                raise RuntimeError(
                    "A5 C8 DSA resident cache must be packed FP8 before "
                    "runtime binding: "
                    f"shape={getattr(packed, 'shape', None)}, "
                    f"dtype={getattr(packed, 'dtype', None)}"
                )
        original_bind_resident(self, **kwargs)
        if not _is_a5_c8(self):
            return
        _ensure_fixed_metadata(self)
        layer_id = int(kwargs["layer_id"])
        # Replace the old per-layer 2048+128 scratch with one shared long-tail
        # fixed-address scratch.  This is graph-stable and keeps Decode suffix
        # metadata valid when output grows beyond one block.
        self._c8_attention_slots[layer_id] = self._a5_c8_attention_slots
        self._c8_resident_seq_lengths[layer_id] = (
            self._a5_resident_seq_lengths.gpu
        )

    def prepare_forward(self: Any, **kwargs: Any) -> torch.Tensor:
        result = original_prepare_forward(self, **kwargs)
        if _is_a5_c8(self):
            _refresh_fixed_metadata(
                self,
                state=kwargs["state"],
                num_reqs=int(kwargs["num_reqs"]),
            )
        return result

    def prepare_graph_capture(self: Any, **kwargs: Any) -> None:
        if _is_a5_c8(self):
            _ensure_fixed_metadata(self)
        original_prepare_graph_capture(self, **kwargs)
        if not _is_a5_c8(self):
            return
        row_count = int(kwargs["row_count"])
        cache_tokens = kwargs.get("cache_tokens")
        if cache_tokens is None:
            raise RuntimeError(
                "A5 C8 graph capture requires fixed cache_tokens metadata"
            )
        # InputBatch dummy uses resident_valid=budget+1 and suffix_start=budget.
        # Preserve those exact values in the additional A5 C8 fixed buffers.
        self._a5_candidate_lens.gpu[:row_count].copy_(
            cache_tokens[:row_count]
        )
        self._a5_logical_seq_lengths.gpu[:row_count].copy_(
            cache_tokens[:row_count]
        )
        self._a5_logical_seq_lengths.gpu[:row_count].add_(1)
        self._a5_resident_seq_lengths.gpu[:row_count].copy_(
            cache_tokens[:row_count]
        )
        self._a5_resident_seq_lengths.gpu[:row_count].add_(1)
        logger.info(
            "[DSA_A5_C8_GRAPH] event=CAPTURE_METADATA_READY rows=%d "
            "candidate=budget logical=budget+1 resident=budget+1 "
            "dump_padding_dst=-1",
            row_count,
        )

    def dump_prefill_full_blocks(self: Any) -> None:
        if not _is_a5_c8(self):
            original_dump_prefill(self)
            return
        if not self._defer_prefill_layer_dump:
            return

        try:
            job_count = int(self.dump_launch_count)
            if job_count == 0:
                logger.info(
                    "[DSA_PREFILL_FULL_DUMP] SKIP_EMPTY_FINAL_PREFILL "
                    "backend=nanovllm_a5 precision=W4A4C8"
                )
                return
            store = self.dram_store
            if store is None:
                raise RuntimeError("DSA runtime has no bound DRAM store")
            if len(self._resident_layer_caches) != self.num_layers:
                raise RuntimeError(
                    "DSA C8 prefill full-KV offload cache binding is "
                    f"incomplete: bound={len(self._resident_layer_caches)}, "
                    f"expected={self.num_layers}"
                )

            from vllm_ascend.dsa_offload.ops_nanovllm import (
                packed_full_kv_cache_blocks,
            )

            src_block_ids = self.dump_src_block_ids.gpu[:job_count]
            dst_block_ids = self.dump_dst_block_ids.gpu[:job_count]
            src_block_ids_cpu = self.dump_src_block_ids.np[:job_count]
            dst_block_ids_cpu = self.dump_dst_block_ids.np[:job_count]
            if np.any(src_block_ids_cpu < 0):
                raise RuntimeError(
                    "A5 C8 prefill dump contains a negative resident source "
                    "block id"
                )
            if np.any(dst_block_ids_cpu <= 0):
                raise RuntimeError(
                    "A5 C8 prefill dump contains an invalid DRAM destination "
                    "block id; block 0 is reserved for the null mapping"
                )

            # _model_forward() is a host-side boundary. NPU work that writes
            # the final Prefill KV may still be outstanding on another stream.
            # Complete those writes before any dump kernel is allowed to read
            # the resident cache. This is one barrier per final Prefill, not a
            # per-layer or Decode-hot-path synchronization.
            torch.npu.synchronize()

            logger.info(
                "[DSA_PREFILL_FULL_DUMP] BEGIN phase=after_model_forward "
                "backend=nanovllm_a5 precision=W4A4C8 packed_bytes=%d "
                "job_count=%d layers=%d",
                DSA_PACKED_KV_DIM,
                job_count,
                self.num_layers,
            )
            max_src_block_id = int(src_block_ids_cpu.max())
            max_dst_block_id = int(dst_block_ids_cpu.max())
            for layer_id in sorted(self._resident_layer_caches):
                resident_packed_cache, _unused_second_plane = (
                    self._resident_layer_caches[layer_id]
                )
                arenas = store.get_layer_arenas(layer_id)
                if max_src_block_id >= int(resident_packed_cache.shape[0]):
                    raise RuntimeError(
                        "A5 C8 prefill dump source block is outside the "
                        "resident cache: "
                        f"layer={layer_id}, max_src={max_src_block_id}, "
                        f"resident_blocks={int(resident_packed_cache.shape[0])}"
                    )
                if max_dst_block_id >= int(arenas.nope.shape[0]):
                    raise RuntimeError(
                        "A5 C8 prefill dump destination block is outside the "
                        "DRAM arena: "
                        f"layer={layer_id}, max_dst={max_dst_block_id}, "
                        f"dram_blocks={int(arenas.nope.shape[0])}"
                    )
                packed_full_kv_cache_blocks(
                    hbm_packed_kv=resident_packed_cache,
                    dram_packed_kv=arenas.nope,
                    src_block_ids=src_block_ids,
                    dst_block_ids=dst_block_ids,
                )

            # packed_full_block_dump is an asynchronous ACLNN launch. Commit
            # every swapped-memory write before this worker returns; only after
            # that return may the scheduler release full Prompt HBM blocks and
            # sparse Decode/KSC start consuming the DRAM table.
            torch.npu.synchronize()
            logger.info(
                "[DSA_PREFILL_FULL_DUMP] END phase=after_model_forward "
                "backend=nanovllm_a5 precision=W4A4C8 job_count=%d "
                "dumped_layers=%d",
                job_count,
                self.num_layers,
            )
        except Exception:
            logger.exception(
                "[DSA_PREFILL_FULL_DUMP] FAILED phase=after_model_forward "
                "backend=nanovllm_a5 precision=W4A4C8"
            )
            raise
        finally:
            self._defer_prefill_layer_dump = False

    def dump_full_blocks(
        self: Any,
        *,
        resident_nope_cache: torch.Tensor,
        resident_rope_cache: torch.Tensor,
    ) -> None:
        if not _is_a5_c8(self.runtime):
            original_dump_full_blocks(
                self,
                resident_nope_cache=resident_nope_cache,
                resident_rope_cache=resident_rope_cache,
            )
            return
        if self.runtime.defer_prefill_layer_dump:
            return
        job_count = int(self.runtime.dump_launch_count)
        if job_count == 0:
            return
        if (
            resident_nope_cache.dtype != torch.float8_e4m3fn
            or resident_nope_cache.ndim != 4
            or int(resident_nope_cache.shape[-1]) != DSA_PACKED_KV_DIM
        ):
            raise RuntimeError(
                "A5 C8 incremental dump requires packed resident KV: "
                f"shape={tuple(resident_nope_cache.shape)}, "
                f"dtype={resident_nope_cache.dtype}"
            )
        store = self.runtime.dram_store
        if store is None:
            raise RuntimeError("DSA layer has no bound DRAM store")
        arenas = store.get_layer_arenas(self.layer_id)
        from vllm_ascend.dsa_offload.ops_nanovllm import (
            packed_full_kv_cache_blocks,
        )

        packed_full_kv_cache_blocks(
            hbm_packed_kv=resident_nope_cache,
            dram_packed_kv=arenas.nope,
            src_block_ids=self.runtime.dump_src_block_ids.gpu[:job_count],
            dst_block_ids=self.runtime.dump_dst_block_ids.gpu[:job_count],
        )

    def execute_decode_selection(self: Any, **kwargs: Any) -> Any:
        runtime = self.runtime
        if not _is_a5_c8(runtime):
            return original_execute_decode(self, **kwargs)

        _ensure_fixed_metadata(runtime)
        query = kwargs["query"]
        weights = kwargs["weights"]
        resident_pool_indices = kwargs["resident_pool_indices"]

        # q_li is dynamically quantized after [B,N,D] is flattened to
        # [B*N,D].  query.shape[0] is therefore the number of Indexer heads,
        # not the request-row count.  The DSA request metadata remains [B], so
        # use it as the authoritative batch dimension and validate the LI ABI.
        if resident_pool_indices.ndim != 1:
            raise RuntimeError(
                "A5 C8 DSA resident_pool_indices must be a 1D request-row "
                f"tensor, got shape={tuple(resident_pool_indices.shape)}"
            )
        num_reqs = int(resident_pool_indices.shape[0])
        if not 0 < num_reqs <= runtime.max_num_reqs:
            raise RuntimeError(
                "A5 C8 DSA request-row count is outside runtime capacity: "
                f"rows={num_reqs}, capacity={runtime.max_num_reqs}"
            )
        if weights.ndim != 2 or int(weights.shape[0]) != num_reqs:
            raise RuntimeError(
                "A5 C8 LIDU weights must be [request_rows,index_heads]: "
                f"weights_shape={tuple(weights.shape)}, request_rows={num_reqs}"
            )
        num_heads = int(weights.shape[1])
        if query.ndim == 2:
            expected_query_rows = num_reqs * num_heads
            if int(query.shape[0]) != expected_query_rows:
                raise RuntimeError(
                    "Flattened A5 C8 LIDU query does not match request/head "
                    f"shape: query_shape={tuple(query.shape)}, "
                    f"request_rows={num_reqs}, index_heads={num_heads}, "
                    f"expected_rows={expected_query_rows}"
                )
        elif query.ndim == 3:
            if (
                int(query.shape[0]) != num_reqs
                or int(query.shape[1]) != num_heads
            ):
                raise RuntimeError(
                    "A5 C8 LIDU query must be [request_rows,index_heads,D]: "
                    f"query_shape={tuple(query.shape)}, "
                    f"request_rows={num_reqs}, index_heads={num_heads}"
                )
        else:
            raise RuntimeError(
                "A5 C8 LIDU query must be flattened [B*N,D] or [B,N,D], "
                f"got shape={tuple(query.shape)}"
            )
        if (
            runtime.execution_num_reqs
            and int(runtime.execution_num_reqs) != num_reqs
        ):
            raise RuntimeError(
                "A5 C8 DSA decode request rows disagree with the prepared "
                f"execution view: li_rows={num_reqs}, "
                f"execution_rows={runtime.execution_num_reqs}, "
                f"active_rows={runtime.active_num_reqs}"
            )
        if not getattr(runtime, "_a5_c8_lidu_shape_logged", False):
            logger.info(
                "[DSA_A5_C8_LIDU_SHAPE] request_rows=%d index_heads=%d "
                "query_shape=%s weights_shape=%s execution_rows=%d "
                "active_rows=%d",
                num_reqs,
                num_heads,
                tuple(query.shape),
                tuple(weights.shape),
                int(runtime.execution_num_reqs),
                int(runtime.active_num_reqs),
            )
            runtime._a5_c8_lidu_shape_logged = True

        outputs = runtime.get_lidu_outputs(num_reqs=num_reqs)
        cache_slots = runtime.resident_token_pool.get_cache_slots(self.layer_id)
        cache_tokens = runtime._req_cache_tokens[:num_reqs]
        candidate_lens = runtime._a5_candidate_lens.gpu[:num_reqs]
        logical_seq_lengths = runtime._a5_logical_seq_lengths.gpu[:num_reqs]

        key_dequant_scale = runtime._c8_key_dequant_scales.get(self.layer_id)
        if key_dequant_scale is None:
            raise RuntimeError(
                f"DSA C8 key dequant scale not bound for layer {self.layer_id}"
            )
        actual_seq_lengths_query = kwargs.get("actual_seq_lengths_query")
        query_dequant_scale = kwargs.get("query_dequant_scale")
        if actual_seq_lengths_query is None or query_dequant_scale is None:
            raise RuntimeError(
                "A5 C8 DSA requires query lengths and query dequant scales"
            )

        from vllm_ascend.dsa_offload.ops_nanovllm import (
            build_c8_attention_metadata,
        )
        from vllm_ascend.dsa_offload.ops_nanovllm import (
            lightning_indexer_decode_update_c8 as lidu_c8,
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
            candidate_lens=candidate_lens,
            block_table=kwargs["indexer_block_table"],
            outputs=outputs,
        )

        resident_seq_lengths = runtime._a5_resident_seq_lengths.gpu[:num_reqs]
        kvaf_adapter = runtime.kvaf_adapter
        if kvaf_adapter is not None:
            directory = runtime.dsa_row_directory
            if directory is None:
                raise RuntimeError("DSA_ROW directory was not initialized")
            kvaf_adapter.execute_batch_get(
                layer_id=self.layer_id,
                source_token_ids=outputs.topk_index,
                destination_slots=outputs.topk_slots,
                miss_count=outputs.miss_count,
                resident_pool_indices=resident_pool_indices,
                resident_block_table=kwargs["resident_block_table"],
                cache_tokens=cache_tokens,
                candidate_lens=candidate_lens,
                logical_seq_lengths=logical_seq_lengths,
                resident_seq_lengths=resident_seq_lengths,
                address_table=directory.addresses,
                output=kwargs["resident_nope_cache"],
            )
            # Subgraph C hands the raw LIDU selection and Resident length to
            # subgraph D. C must not construct sparse+tail attention slots.
            return DSAOffloadSelectionOutput(
                sparse_indices=outputs.topk_slots,
                tail_info=resident_seq_lengths,
            )

        from vllm_ascend.dsa_offload.ops_nanovllm import (
            packed_kvcache_scatter_copy as ksc_c8,
        )

        store = runtime.dram_store
        if store is None:
            raise RuntimeError("DSA layer has no bound DRAM store")
        arenas = store.get_layer_arenas(self.layer_id)
        attention_slots = runtime._a5_c8_attention_slots[:num_reqs]
        ksc_c8(
            hbm_packed_kv=kwargs["resident_nope_cache"],
            dram_packed_kv=arenas.nope,
            hbm_block_table=kwargs["resident_block_table"],
            dram_block_table=kwargs["dram_block_table"],
            source_token_ids=outputs.topk_index,
            destination_slots=outputs.topk_slots,
            copy_counts=outputs.miss_count,
            cache_tokens=cache_tokens,
            candidate_lens=candidate_lens,
            actual_seq_lengths_kv=logical_seq_lengths,
            max_tail_tokens=runtime._a5_c8_max_tail_tokens,
            attention_slots=attention_slots,
            resident_seq_lengths=resident_seq_lengths,
        )
        build_c8_attention_metadata(
            destination_slots=outputs.topk_slots,
            cache_tokens=cache_tokens,
            candidate_lens=candidate_lens,
            actual_seq_lengths_kv=logical_seq_lengths,
            max_tail_tokens=runtime._a5_c8_max_tail_tokens,
            resident_seq_lengths=resident_seq_lengths,
            attention_slots=attention_slots,
        )
        return DSAOffloadSelectionOutput(
            attention_slots=attention_slots,
            resident_seq_lengths=resident_seq_lengths,
        )

    NPUModelRunner._reshape_kv_cache_tensors = reshape_kv_cache_tensors
    DSAOffloadRuntime.bind_resident_layer_cache = bind_resident_layer_cache
    DSAOffloadRuntime.prepare_forward = prepare_forward
    DSAOffloadRuntime.prepare_graph_capture = prepare_graph_capture
    DSAOffloadRuntime.dump_prefill_full_blocks = dump_prefill_full_blocks
    DSALayerOffloadContext.execute_decode_selection = execute_decode_selection
    DSALayerOffloadContext.dump_full_blocks = dump_full_blocks
    _INSTALLED = True
