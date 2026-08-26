# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 C8 split-Indexer cache adapter.

The initial DSA split cache allocated the Indexer group using the legacy dense
page size. For W4A4C8 that backing is large enough to host the real A5 ABI
without another allocation:

    FP8 key    [blocks,128,1,128]
    FP32 scale [blocks,128,1]

This module creates zero-copy views into that existing backing, binds key and
scale to their real consumers, writes both planes together, routes Prefill TopK
through the A5 quant LightningIndexer, and sanitizes padded FULL-graph rows.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch_npu
from vllm.logger import logger
from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend.dsa_offload.kv_cache import DSAIndexerKVSpec

_INSTALLED = False


def _runner_is_a5_c8(runner: Any) -> bool:
    cfg = runner.ascend_config
    dsa = cfg.dsa_offload_config
    return bool(
        runner.dsa_offload_enabled
        and dsa.op_backend == "nanovllm_a5"
        and cfg.enable_sparse_sfa_c8
        and cfg.enable_sparse_li_c8
    )


def _runtime_is_a5_c8(runtime: Any) -> bool:
    return bool(runtime.op_backend == "nanovllm_a5" and runtime.is_c8)


def _impl_is_a5_c8(impl: Any) -> bool:
    context = getattr(impl, "dsa_offload_context", None)
    return context is not None and _runtime_is_a5_c8(context.runtime)


def _make_split_indexer_views(
    dense_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dense_cache.ndim != 4:
        raise RuntimeError(
            "DSA split Indexer backing must be 4D before C8 reinterpretation: "
            f"shape={tuple(dense_cache.shape)}"
        )
    blocks, block_size, heads, head_dim = map(int, dense_cache.shape)
    if block_size != 128 or heads != 1 or head_dim != 128:
        raise RuntimeError(
            "A5 C8 split Indexer requires [blocks,128,1,128] logical shape, "
            f"got {tuple(dense_cache.shape)}"
        )
    if not dense_cache.is_contiguous():
        raise RuntimeError("DSA split Indexer backing must be contiguous")

    raw = dense_cache.view(torch.int8).view(-1)
    key_elements = blocks * block_size * heads * head_dim
    scale_elements = blocks * block_size * heads
    scale_bytes = scale_elements * torch.empty(
        (), dtype=torch.float32
    ).element_size()
    required_bytes = key_elements + scale_bytes
    if raw.numel() < required_bytes:
        raise RuntimeError(
            "Legacy DSA Indexer backing is too small for A5 C8 key+scale: "
            f"available={raw.numel()} bytes, required={required_bytes} bytes"
        )
    scale_offset = key_elements
    if (raw.data_ptr() + scale_offset) % 4:
        raise RuntimeError(
            "A5 C8 Indexer scale view is not FP32 aligned: "
            f"offset={scale_offset}"
        )

    key = raw[:key_elements].view(torch.float8_e4m3fn).view(
        blocks,
        block_size,
        heads,
        head_dim,
    )
    scale = raw[
        scale_offset : scale_offset + scale_bytes
    ].view(torch.float32).view(
        blocks,
        block_size,
        heads,
    )
    scale.zero_()
    return key, scale


def install_a5_c8_split_cache_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import vllm_ascend.attention.sfa_v1 as sfa_mod
    import vllm_ascend.worker.model_runner_v1 as runner_mod
    from vllm_ascend.dsa_offload.runtime import DSAOffloadRuntime

    NPUModelRunner = runner_mod.NPUModelRunner
    AscendSFAImpl = sfa_mod.AscendSFAImpl

    original_reshape = NPUModelRunner._reshape_kv_cache_tensors
    original_bind_target = NPUModelRunner._bind_dsa_target_kv_caches
    original_indexer_pre = AscendSFAImpl.indexer_select_pre_process
    original_indexer_post = AscendSFAImpl.indexer_select_post_process
    original_prepare_execution_view = DSAOffloadRuntime.prepare_execution_view

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
            if not isinstance(spec, DSAIndexerKVSpec):
                continue
            backing = caches[layer_name]
            if isinstance(backing, tuple):
                if len(backing) != 2:
                    raise RuntimeError(
                        "A5 C8 split Indexer tuple must contain key+scale: "
                        f"layer={layer_name}, len={len(backing)}"
                    )
                continue
            if not isinstance(backing, torch.Tensor):
                raise RuntimeError(
                    "A5 C8 split Indexer backing must be a Tensor before "
                    f"reinterpretation: layer={layer_name}"
                )
            key, scale = _make_split_indexer_views(backing)
            caches[layer_name] = (key, scale)
            logger.info(
                "[DSA_C8_INDEXER_LAYOUT] layer=%s key_shape=%s "
                "key_dtype=%s scale_shape=%s scale_dtype=%s "
                "zero_copy_backing=true",
                layer_name,
                tuple(key.shape),
                key.dtype,
                tuple(scale.shape),
                scale.dtype,
            )
        return caches

    def bind_dsa_target_kv_caches(
        self: Any,
        kv_caches: dict[str, Any],
        kv_cache_config: Any,
    ) -> None:
        if not _runner_is_a5_c8(self):
            original_bind_target(self, kv_caches, kv_cache_config)
            return

        specs = self._get_layer_kv_cache_specs(kv_cache_config)
        split_indexers: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        proxy = dict(kv_caches)
        for layer_name, spec in specs.items():
            if not isinstance(spec, DSAIndexerKVSpec):
                continue
            cache = kv_caches[layer_name]
            if not isinstance(cache, tuple) or len(cache) != 2:
                raise RuntimeError(
                    "A5 C8 split Indexer must expose (key, scale): "
                    f"layer={layer_name}, type={type(cache).__name__}"
                )
            key, scale = cache
            split_indexers[layer_name] = (key, scale)
            # The legacy binder only understands one dense Indexer tensor. Feed
            # it the key view, then overwrite its temporary scale binding below.
            proxy[layer_name] = key

        original_bind_target(self, proxy, kv_cache_config)

        runtime = self.dsa_offload_runtime
        if runtime is None:
            raise RuntimeError("DSA runtime missing after C8 target binding")
        ordered = sorted(
            (
                extract_layer_index(layer_name),
                layer_name,
                key,
                scale,
            )
            for layer_name, (key, scale) in split_indexers.items()
        )
        if len(ordered) != runtime.num_layers:
            raise RuntimeError(
                "A5 C8 split Indexer layer count does not match runtime: "
                f"indexer={len(ordered)}, runtime={runtime.num_layers}"
            )
        for runtime_layer_id, (
            layer_index,
            layer_name,
            key,
            scale,
        ) in enumerate(ordered):
            runtime.bind_c8_key_dequant_scale(
                layer_id=runtime_layer_id,
                key_dequant_scale=scale,
            )
            logger.info(
                "[DSA_C8_INDEXER_BIND] runtime_layer=%d model_layer=%d "
                "layer=%s key_shape=%s scale_shape=%s",
                runtime_layer_id,
                layer_index,
                layer_name,
                tuple(key.shape),
                tuple(scale.shape),
            )

    def indexer_select_pre_process(
        self: Any,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Any:
        result = original_indexer_pre(self, x, cos, sin)
        if _impl_is_a5_c8(self):
            k_li, k_li_scale = result
            if k_li_scale is None:
                raise RuntimeError(
                    "A5 C8 split Indexer preprocessing produced no key scale"
                )
            self._dsa_a5_pending_k_li_scale = k_li_scale
        return result

    def _write_split_scale(
        self: Any,
        attn_metadata: Any,
    ) -> torch.Tensor:
        context = self.dsa_offload_context
        assert context is not None
        scale_cache = context.runtime._c8_key_dequant_scales.get(
            context.layer_id
        )
        if scale_cache is None:
            raise RuntimeError(
                f"A5 C8 Indexer scale is not bound for layer {context.layer_id}"
            )
        pending = getattr(self, "_dsa_a5_pending_k_li_scale", None)
        if pending is None:
            raise RuntimeError(
                "A5 C8 split Indexer key scale was not produced before cache write"
            )
        slot_mapping = attn_metadata.dsa_indexer_slot_mapping
        if slot_mapping is None:
            raise RuntimeError(
                "A5 C8 split Indexer scale write is missing slot mapping"
            )
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        if pending.shape[0] < num_actual_tokens:
            raise RuntimeError(
                "A5 C8 pending Indexer scale is shorter than active tokens: "
                f"scale_rows={pending.shape[0]}, tokens={num_actual_tokens}"
            )
        torch_npu.npu_scatter_nd_update_(
            scale_cache.view(-1, 1),
            slot_mapping[:num_actual_tokens].view(-1, 1),
            pending[:num_actual_tokens].view(-1, 1),
        )
        return scale_cache

    def indexer_select_post_process(
        self: Any,
        x: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: Any,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ) -> Any:
        if not _impl_is_a5_c8(self):
            return original_indexer_post(
                self,
                x,
                q_c,
                kv_cache,
                attn_metadata,
                cos,
                sin,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
            )

        scale_cache = _write_split_scale(self, attn_metadata)

        # Single-token Decode reuses the existing q preparation; the patched
        # DSALayerOffloadContext consumes the correct key-scale binding.
        if attn_metadata.attn_state == sfa_mod.AscendAttentionState.DecodeOnly:
            return original_indexer_post(
                self,
                x,
                q_c,
                kv_cache,
                attn_metadata,
                cos,
                sin,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
            )

        # Prefill/mixed/spec-verification uses the A5 quant LightningIndexer
        # rather than the legacy BF16 npu_lightning_indexer.
        assert self.wk_weights_proj is not None
        assert self.wq_b is not None
        kw, _ = self.wk_weights_proj(x)
        weights = kw[:, self.head_dim :]

        if isinstance(q_c, tuple):
            q_c_tensor, q_c_scale = q_c
            q_c_tensor = q_c_tensor.view(-1, q_c_tensor.shape[-1])
            quant_matmul_kwargs: dict[str, Any] = dict(
                bias=None,
                output_dtype=x.dtype,
            )
            if q_c_tensor.dtype == torch.float8_e4m3fn:
                if q_c_scale.dim() == 2:
                    q_c_scale = q_c_scale.view(q_c_scale.shape[0], -1, 2)
                quant_matmul_kwargs.update(
                    scale_dtype=sfa_mod.FLOAT8_E8M0FNU_DTYPE,
                    pertoken_scale_dtype=sfa_mod.FLOAT8_E8M0FNU_DTYPE,
                    group_sizes=[
                        1,
                        1,
                        getattr(
                            self.wq_b.quant_method.quant_method,
                            "group_size",
                            32,
                        ),
                    ],
                )
            elif q_c_scale.dim() > 1 and q_c_scale.shape[-1] == 1:
                q_c_scale = q_c_scale.squeeze(dim=-1)
            q_li = torch_npu.npu_quant_matmul(
                q_c_tensor,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=q_c_scale,
                **quant_matmul_kwargs,
            )
        else:
            q_li, _ = self.wq_b(q_c)

        q_li = q_li.view(-1, self.n_head, self.head_dim)
        if sfa_mod.HAS_TRITON:
            q_li = sfa_mod.rope_forward_triton_siso(
                q_li,
                cos,
                sin,
                rope_dim=self.qk_rope_head_dim,
                is_neox_style=self.is_rope_neox_style,
            )
        else:
            q_li_pe, q_li_nope = torch.split(
                q_li,
                [
                    self.qk_rope_head_dim,
                    self.head_dim - self.qk_rope_head_dim,
                ],
                dim=-1,
            )
            q_li_pe = q_li_pe.unsqueeze(2)
            q_li_pe = torch_npu.npu_rotary_mul(q_li_pe, cos, sin)
            q_li_pe = q_li_pe.squeeze(2)
            q_li = torch.cat([q_li_pe, q_li_nope], dim=-1)

        if AscendSFAImpl.q_hadamard is None:
            raise RuntimeError("A5 C8 q_hadamard was not initialized")
        q_li = q_li @ AscendSFAImpl.q_hadamard
        q_li, q_li_scale = torch_npu.npu_dynamic_quant(
            q_li.view(-1, self.head_dim),
            dst_type=self.c8_k_cache_dtype,
        )
        q_li_scale = q_li_scale.to(self.c8_k_scale_cache_dtype)
        sfa_mod.record_attention_compute_start()

        context = self.dsa_offload_context
        assert context is not None
        indexer_block_table = attn_metadata.dsa_indexer_block_table
        if indexer_block_table is None:
            raise RuntimeError(
                "A5 C8 Prefill is missing the independent Indexer block table"
            )
        from vllm_ascend.dsa_offload.ops_nanovllm import (
            quant_lightning_indexer_c8,
        )

        topk = quant_lightning_indexer_c8(
            query=q_li,
            key=context.indexer_cache,
            weights=weights,
            query_dequant_scale=q_li_scale,
            key_dequant_scale=scale_cache,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=indexer_block_table,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[DSA_C8_PREFILL_INDEXER] topk_shape=%s key_shape=%s "
                "scale_shape=%s",
                tuple(topk.shape),
                tuple(context.indexer_cache.shape),
                tuple(scale_cache.shape),
            )
        return topk

    def prepare_execution_view(
        self: Any,
        *,
        active_num_reqs: int,
        graph_row_count: int | None,
    ) -> int:
        execution_num_reqs = original_prepare_execution_view(
            self,
            active_num_reqs=active_num_reqs,
            graph_row_count=graph_row_count,
        )
        if (
            not _runtime_is_a5_c8(self)
            or graph_row_count is None
            or execution_num_reqs <= int(active_num_reqs)
        ):
            return execution_num_reqs
        if not all(
            hasattr(self, name)
            for name in (
                "_a5_candidate_lens",
                "_a5_logical_seq_lengths",
                "_a5_resident_seq_lengths",
                "_a5_c8_selected_slot_lengths",
            )
        ):
            raise RuntimeError(
                "A5 C8 graph replay metadata was not initialized before padding"
            )

        start = int(active_num_reqs)
        end = int(execution_num_reqs)
        # Padding rows have cache_tokens=0 / padding resident-pool entries in
        # the shared DSA state.  Zero lengths make LI/KSC strict no-op rows and
        # prevent stale data from a previous larger graph replay.
        for buffer in (
            self._a5_candidate_lens,
            self._a5_logical_seq_lengths,
            self._a5_resident_seq_lengths,
            self._a5_c8_selected_slot_lengths,
        ):
            buffer.np[start:end] = 0
            buffer.gpu[start:end].copy_(
                buffer.cpu[start:end],
                non_blocking=True,
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[DSA_A5_C8_GRAPH] event=PAD_METADATA active_rows=%d "
                "graph_rows=%d padded_rows=%d",
                start,
                end,
                end - start,
            )
        return execution_num_reqs

    NPUModelRunner._reshape_kv_cache_tensors = reshape_kv_cache_tensors
    NPUModelRunner._bind_dsa_target_kv_caches = bind_dsa_target_kv_caches
    AscendSFAImpl.indexer_select_pre_process = indexer_select_pre_process
    AscendSFAImpl.indexer_select_post_process = indexer_select_post_process
    DSAOffloadRuntime.prepare_execution_view = prepare_execution_view
    _INSTALLED = True
