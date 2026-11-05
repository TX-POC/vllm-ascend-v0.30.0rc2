# SPDX-License-Identifier: Apache-2.0
"""Opt-in Runner extension; native Runner and attention code remain untouched."""

from copy import copy
from dataclasses import replace

import torch
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.platforms import current_platform

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import get_layerwise_reuse_config
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

from ..layout import describe_cache_layouts
from .backend import create_decode_cache
from .config import PDConfig
from .memory import allocate_layer
from .runtime import PDAdapter, load_sdk


class PDKVModelRunner(NPUModelRunner):
    def __init__(self, vllm_config, device):
        self.pd_config = PDConfig.from_vllm(vllm_config)
        self.pd_memories = {}
        self.pd_diagnostics = None
        self._pd_indexer_names = {}
        super().__init__(vllm_config, device)
        producer = vllm_config.kv_transfer_config.is_kv_producer
        if self.ascend_config.kvaf_config.enabled == producer:
            raise ValueError("P requires kvaf_config.enabled=false; D requires enabled=true")
        if self.ascend_config.sparse_kv_offload_config.enabled or self.ascend_config.kvpp_config.size != 1:
            raise ValueError("PD KVAF cannot combine local-CPU offload or KVPP with remote DDR")
        # Own the temporary config view; never change the process-wide native
        # AscendConfig used by attention or another Connector.
        self.ascend_config = copy(self.ascend_config)

    def _initialize_kvaf_cache_state(self):
        if not self.ascend_config.kvaf_config.enabled or self.kvaf_cache_state is not None:
            return
        specs = getattr(self, "kv_cache_spec", None) or self.get_kv_cache_spec()
        drafts = frozenset(getattr(self.drafter, "attn_layer_names", ()))
        targets = tuple(
            name
            for name, spec in specs.items()
            if name not in drafts and not hasattr(spec, "sfa_dcp_replicated_indexer_size")
        )
        self.kvaf_cache_state = create_decode_cache(
            targets,
            self.max_num_reqs,
            self.max_num_tokens,
            self.model_config.max_model_len,
            self.device,
            num_spec_tokens=self.num_spec_tokens,
            hot_tokens=self.ascend_config.kvaf_config.hot_tokens,
        )
        self.kvaf_target_layer_names = set(targets)

    def _allocate_kv_cache_tensors(self, kv_cache_config):
        specs = self._get_layer_kv_cache_specs(kv_cache_config)
        context = self.compilation_config.static_forward_context
        groups = {
            name: index for index, group in enumerate(kv_cache_config.kv_cache_groups) for name in group.layer_names
        }
        if any(spec.block_size != 128 for spec in specs.values()):
            raise ValueError("PD KVAF requires 128-token blocks")
        # Cache aliasing/layerwise reuse requires storage-slot arbitration. This
        # PD layout deliberately registers each real layer independently.
        transfer = self.vllm_config.kv_transfer_config
        if get_layerwise_reuse_config(transfer) is not None:
            raise ValueError("disable layerwise cache aliasing for PD KVAF")
        if self.kvaf_cache_state is not None:
            # Profiling already charged this capacity. Drop its placeholder
            # before allocating the final arenas, avoiding a second full pool
            # at the peak of a production-sized initialization.
            self.kvaf_cache_state.storage.pop("pd_receive_reservation", None)
        raw, covered = {}, set()
        drafts = frozenset(getattr(self.drafter, "attn_layer_names", ()))
        for name, spec in specs.items():
            if hasattr(spec, "sfa_dcp_replicated_indexer_size"):
                continue
            if spec.page_size_bytes != 128 * 656:
                raise ValueError(f"{name}: PD requires packed 656-byte C8 main KV")
            owner = context[name].impl
            # Ownership follows the allocated k_cache, including a checkpoint
            # indexer whose top-k work is currently skipped at runtime.
            indexer = getattr(owner, "indexer", None)
            indexer_name = indexer.k_cache.prefix if indexer is not None else None
            self._pd_indexer_names[name] = indexer_name
            plane_groups = {"main": groups[name]}
            if indexer_name is not None:
                indexer_spec = specs[indexer_name]
                if indexer_spec.page_size_bytes != 128 * (128 + 4) or indexer.num_cache_tensors != 2:
                    raise ValueError("PD requires C8 indexer with 128-byte keys and four-byte scale")
                plane_groups.update(indexer=groups[indexer_name], scale=groups[indexer_name])
            offload = name in self.kvaf_target_layer_names
            memory = allocate_layer(
                self.max_num_reqs if offload else kv_cache_config.num_blocks,
                kv_cache_config.num_blocks if indexer_name else 0,
                device=self.device,
                groups=plane_groups,
                offload_main=name not in drafts,
                allocator=lambda size: self._allocate_int8_cache_tensor(size, 2 * 1024 * 1024),
            )
            memory.describe(name)  # Check the actual storage before registration.
            self.pd_memories[name] = memory
            main = memory.planes["main"].view(-1)
            if offload:
                main = self.kvaf_cache_state.policy.hot[self.kvaf_cache_state.indices[name]].view(-1)
            raw[name] = (main,)
            covered.add(name)
            if indexer_name is not None:
                if indexer_name in covered:
                    raise ValueError("one indexer allocation cannot have multiple PD layer owners")
                raw[indexer_name] = (memory.planes["indexer"].view(-1), memory.planes["scale"].view(-1))
                covered.add(indexer_name)
        if covered != specs.keys():
            raise ValueError(f"unowned PD cache specs: {specs.keys() - covered}")
        self.hybrid_with_attn_and_mamba = False
        return raw

    def _reshape_kv_cache_tensors(self, *args, **kwargs):
        caches = super()._reshape_kv_cache_tensors(*args, **kwargs)
        for name, memory in self.pd_memories.items():
            if name not in self.kvaf_target_layer_names:
                if len(caches[name]) != 1:
                    raise ValueError("PD main cache must remain one packed C8 view")
                memory.planes["main"] = caches[name][0]
            indexer_name = self._pd_indexer_names[name]
            if indexer_name is not None:
                memory.planes["indexer"], memory.planes["scale"] = caches[indexer_name]
            memory.describe(name)  # Validate final native views, including typed scale strides.
        self._pd_bound_caches = caches
        return caches

    def initialize_kv_cache(self, kv_cache_config, kv_cache_allocation_context=None):
        config = self.ascend_config.kvaf_config
        # Keep native allocation/binding/attention setup, replacing only its
        # hard-coded non-PD adapter creation with our external Backend below.
        self.ascend_config.kvaf_config = replace(config, enabled=False)
        try:
            super().initialize_kv_cache(kv_cache_config, kv_cache_allocation_context)
        finally:
            self.ascend_config.kvaf_config = config
        connector = get_kv_transfer_group()
        if not hasattr(connector, "bind_pd_backend"):
            raise ValueError("PD Worker requires the independent PDKVoffloadConnector")
        runtime = None
        if config.enabled:
            caches = self._pd_bound_caches
            layouts = describe_cache_layouts(
                caches,
                self.kv_cache_config,
                getattr(self.drafter, "attn_layer_names", ()),
                layer_specs=self._get_layer_kv_cache_specs(self.kv_cache_config),
            )
            self.kvaf_aux_caches = [cache for name, cache in caches.items() if name not in self.kvaf_target_layer_names]
            self.kvaf_adapter = PDAdapter(
                config,
                torch.npu.current_device(),
                layouts,
                groups=self.kvaf_cache_state.groups,
                log_dir=self.ascend_config.ascend_log_path,
                tp_rank=self.tp_rank,
                dp_rank=self.dp_rank,
                cpu_eid=self.pd_config.eids(
                    current_platform.visible_device_id_to_physical_device_id(torch.npu.current_device())
                )[0],
            )
            self.kvaf_adapter.connect()
            self.kvaf_adapter.bind_caches(caches)
            feedback = self.kvaf_adapter.writer.feedback
            feedback.bind_device_pool(self.kvaf_cache_state)
            feedback.bind(self.compilation_config.static_forward_context)
            runtime = self.kvaf_adapter.runtimes[656, 128]
            runtime.compute_done = torch.npu.Event()
            runtime.compute_done.record()
        sdk = load_sdk(config.build_dir)
        connector.bind_pd_backend(
            sdk=sdk,
            runtime=runtime,
            memories=self.pd_memories,
            writer=self.kvaf_adapter.writer if self.kvaf_adapter is not None else None,
            pool=self.kvaf_cache_state,
            draft_limit=self.effective_drafter_max_model_len,
        )
        self.pd_diagnostics = connector.diagnostics
        if self.pd_diagnostics:
            wrapped = set()
            for name in self.pd_memories:
                indexer = getattr(self.compilation_config.static_forward_context[name].impl, "indexer", None)
                if indexer is not None and id(indexer) not in wrapped:
                    self.pd_diagnostics.decorate_indexer(indexer, name, draft=not self.pd_memories[name].offload_main)
                    wrapped.add(id(indexer))

    def _run_pd_compute(self, operation, *args, **kwargs):
        diagnostics = getattr(self, "pd_diagnostics", None)
        if diagnostics:
            if args and hasattr(args[0], "num_scheduled_tokens"):
                diagnostics.context = tuple(args[0].num_scheduled_tokens)
                diagnostics.draft_skipped = False
            else:
                diagnostics.context = tuple(getattr(getattr(self, "input_batch", None), "req_ids", ()))
        if self.kvaf_adapter is None:
            return operation(*args, **kwargs)
        runtime = self.kvaf_adapter.runtimes[656, 128]
        with runtime.lock:
            if runtime.failure is not None:
                raise RuntimeError("PD Runtime has an unresolved transfer") from runtime.failure
            try:
                return operation(*args, **kwargs)
            finally:
                # Covers asynchronous graph replay Get kernels as well as
                # ordinary SDK calls. A PD receiver waits this event while
                # holding the same lock; normal Decode adds no host sync.
                runtime.compute_done.record()

    def execute_model(self, *args, **kwargs):
        if self.is_kv_producer:
            get_kv_transfer_group().wait_for_pd_sources()
        return self._run_pd_compute(super().execute_model, *args, **kwargs)

    def sample_tokens(self, *args, **kwargs):
        result = self._run_pd_compute(super().sample_tokens, *args, **kwargs)
        if self.is_kv_producer:
            get_kv_transfer_group().finish_pd_forward()
        return result

    def _model_forward(self, *args, **kwargs):
        diagnostics = getattr(self, "pd_diagnostics", None)
        if diagnostics:
            from vllm.forward_context import get_forward_context

            context = get_forward_context()
            mode = getattr(context.cudagraph_runtime_mode, "name", str(context.cudagraph_runtime_mode))
            for req in diagnostics.context:
                if diagnostics.active(req):
                    diagnostics.emit(
                        "MODEL_FORWARD",
                        (req, mode),
                        request=req,
                        graph_mode=mode,
                        num_tokens_padded=args[0] if args else kwargs.get("num_tokens_padded"),
                    )
        result = super()._model_forward(*args, **kwargs)
        if diagnostics:
            for req in diagnostics.context:
                if diagnostics.active(req):
                    diagnostics.emit("MODEL_FORWARD_DONE", (req, mode), request=req, graph_mode=mode)
        return result

    def _skip_drafting(self, *args, **kwargs):
        if getattr(self, "pd_diagnostics", None):
            self.pd_diagnostics.draft_skipped = True
        self._diagnose_drafting("MTP_SKIP")
        if self.is_kv_producer:
            get_kv_transfer_group().skip_pd_drafting(self.effective_drafter_max_model_len)
        return super()._skip_drafting(*args, **kwargs)

    def propose_draft_token_ids(self, *args, **kwargs):
        result = super().propose_draft_token_ids(*args, **kwargs)
        self._diagnose_drafting("MTP_PROPOSE_DONE")
        if self.is_kv_producer:
            # Captured MTP graphs may omit Python write hooks, just as target
            # graphs do. Only a successful real native proposal proves validity.
            get_kv_transfer_group().complete_pd_drafting()
        return result

    def _diagnose_drafting(self, event):
        diagnostics = getattr(self, "pd_diagnostics", None)
        if diagnostics:
            for req in diagnostics.context:
                if diagnostics.active(req):
                    diagnostics.emit(
                        event, req, request=req, tokens=self.num_spec_tokens, limit=self.effective_drafter_max_model_len
                    )

    def shutdown(self):
        if has_kv_transfer_group():
            get_kv_transfer_group().shutdown()
        super().shutdown()
