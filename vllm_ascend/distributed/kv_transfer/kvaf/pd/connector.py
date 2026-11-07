# SPDX-License-Identifier: Apache-2.0
"""Independent Scheduler/Worker entry for DSA PD and remote DDR offload."""

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorRole, SupportsHMA

from .config import PDConfig


class PDKVoffloadConnector(KVConnectorBase_V1, SupportsHMA):
    supports_layerwise_buffer_reuse = True

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.config = PDConfig.from_vllm(vllm_config)
        self.vllm_config = vllm_config
        self.is_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.requires_full_blocks_on_update_after_alloc = self.is_producer and role == KVConnectorRole.SCHEDULER
        self.scheduler = self.worker = None
        self._caches = None
        self._layer_names = ()
        if role == KVConnectorRole.SCHEDULER:
            from .scheduler import ConsumerScheduler, ProducerScheduler

            self.scheduler = (
                ProducerScheduler(vllm_config, kv_cache_config, str(vllm_config.kv_transfer_config.engine_id))
                if self.is_producer
                else ConsumerScheduler(vllm_config, True, kv_cache_config)
            )

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return self.scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return self.scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output):
        return self.scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request, block_ids):
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(self, request, block_ids):
        return self.scheduler.request_finished_all_groups(request, block_ids)

    def update_connector_output(self, connector_output):
        if not self.is_producer:
            self.scheduler.update_connector_output(connector_output)

    def register_kv_caches(self, kv_caches):
        # Native Runner invokes this before its adapter is created. The external
        # Runner finishes binding once both actual layer arenas and Runtime exist.
        if self._caches is not None:
            raise RuntimeError("PD cache registration cannot repeat")
        self._caches = kv_caches

    def bind_pd_backend(self, *, sdk, runtime, memories, writer, pool, draft_limit):
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.platforms import current_platform
        from vllm.utils.network_utils import get_ip

        from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import get_external_request_id

        from .receiver import Receiver
        from .sender import Sender

        if self._caches is None or self.worker is not None:
            raise RuntimeError("bind PD Backend once, after native cache registration")
        device_id = torch.npu.current_device()
        physical_id = current_platform.visible_device_id_to_physical_device_id(device_id)
        cpu_eid, npu_eid = self.config.eids(physical_id)
        pc = self.vllm_config.parallel_config
        tp_rank = get_tensor_model_parallel_rank()
        host = get_ip()
        self.diagnostics = None
        if self.config.diagnostics_requests:
            from .diagnostics import Diagnostics

            self.diagnostics = Diagnostics(
                self.config.diagnostics_requests,
                self.config.verify_kv,
                "P" if self.is_producer else "D",
                worker=f"{host}:DP{pc.data_parallel_rank}:TP{tp_rank}",
            )
            self.diagnostics.emit(
                "WORKER",
                "startup",
                host=host,
                physical_device=physical_id,
                tp_rank=tp_rank,
                tp_size=pc.tensor_parallel_size,
                dp_rank=pc.data_parallel_rank,
                draft_limit=draft_limit,
                verify_kv=self.config.verify_kv,
                cpu_eid=cpu_eid,
                npu_eid=npu_eid,
                model=self.vllm_config.model_config.model,
                model_dtype=str(self.vllm_config.model_config.dtype),
                quantization=getattr(self.vllm_config.model_config, "quantization", None),
                kv_dtype=self.vllm_config.cache_config.cache_dtype,
                indexer_dtype=self.vllm_config.attention_config.indexer_kv_dtype,
                target_layers=sum(memory.offload_main for memory in memories.values()),
                source_tp0_only=self.config.source_tp0_only,
                source_participant=not self.is_producer or not self.config.source_tp0_only or tp_rank == 0,
            )
            for name, memory in memories.items():
                self.diagnostics.layout(name, memory)
        self._layer_names = tuple(memories)
        self._external_id = get_external_request_id

        def activate():
            torch.npu.set_device(device_id)

        def stream():
            return torch.npu.current_stream(device_id)

        def event():
            result = torch.npu.Event()
            result.record(stream())
            return result

        if self.is_producer:
            producer = f"{self.vllm_config.kv_transfer_config.engine_id}:{pc.data_parallel_rank}:{tp_rank}"
            self.worker = Sender(
                sdk,
                memories,
                config=self.config,
                device_id=device_id,
                producer=producer,
                host=host,
                activate=activate,
                stream_factory=stream,
                event_factory=event,
                external_id=get_external_request_id,
                tp_rank=tp_rank,
                tp_size=pc.tensor_parallel_size,
                dp_rank=pc.data_parallel_rank,
                cpu_eid=cpu_eid,
                npu_eid=npu_eid,
                draft_limit=draft_limit,
                num_spec_tokens=self.vllm_config.speculative_config.num_speculative_tokens,
                diagnostics=self.diagnostics,
            )
        else:
            pool.policy.diagnostics = self.diagnostics
            port = (
                self.vllm_config.kv_transfer_config.kv_port + pc.data_parallel_rank * pc.tensor_parallel_size + tp_rank
            )
            self.worker = Receiver(
                runtime,
                writer,
                pool,
                memories,
                host=host,
                port=port,
                local_eid=npu_eid,
                timeout=self.config.timeout,
                activate=activate,
                stream_factory=stream,
                start_immediately=False,
                source_tp0_only=self.config.source_tp0_only,
                draft_limit=draft_limit,
                num_spec_tokens=self.vllm_config.speculative_config.num_speculative_tokens,
                diagnostics=self.diagnostics,
            )

    def start_pd_control(self):
        if not self.is_producer and self.worker is not None:
            # Called only after native warmup/capture. No attach/probe can race
            # those initialization kernels when P and D start concurrently.
            self.worker.runtime.compute_done.record()
            self.worker.start()

    def bind_connector_metadata(self, connector_metadata):
        super().bind_connector_metadata(connector_metadata)
        if self.worker is not None and self.is_producer:
            self.worker.bind(connector_metadata)

    def start_load_kv(self, forward_context, **kwargs):
        if self.worker is not None and not self.is_producer:
            metadata = self._get_connector_metadata()
            for req_id, seed in metadata.requests.items():
                self.worker.seed(req_id, self._external_id(req_id), seed)

    def wait_for_layer_load(self, layer_name):
        if self.is_producer and self.worker is not None:
            self.worker.wait_layer(layer_name)

    def wait_for_layer_reuse(self, layer_idx):
        if self.is_producer and self.worker is not None:
            self.worker.wait_layer(self._layer_names[layer_idx])

    def on_kv_cache_written(self, layer_name=""):
        if self.is_producer and self.worker is not None and self.has_connector_metadata():
            self.worker.written(layer_name)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        self.on_kv_cache_written(layer_name)

    def wait_for_save(self):
        if self.is_producer and self.worker is not None:
            self.worker.finalize()

    def wait_for_pd_sources(self):
        if self.is_producer and self.worker is not None:
            self.worker.wait_all()

    def finish_pd_forward(self):
        if self.is_producer and self.worker is not None:
            self.worker.complete_forward()

    def skip_pd_drafting(self, limit):
        if self.is_producer and self.worker is not None:
            self.worker.skip_drafting(limit)

    def complete_pd_drafting(self):
        if self.is_producer and self.worker is not None:
            self.worker.complete_drafting()

    def get_finished(self, finished_req_ids):
        if self.worker is None:
            return set(), set()
        if self.is_producer:
            return self.worker.get_finished(finished_req_ids), set()
        finished = self.worker.get_finished()
        # Original native request teardown already releases successful Decode
        # directories. This also covers a request cancelled before first Decode.
        self.worker.release(set(finished_req_ids))
        return set(), finished

    def get_block_ids_with_load_errors(self):
        if self.worker is None or self.is_producer:
            return set()
        return self.worker.get_errors()

    def shutdown(self):
        if self.worker is not None:
            self.worker.close()
        shutdown = getattr(self.scheduler, "shutdown", None)
        if shutdown is not None:
            shutdown()

    def close(self):
        self.shutdown()
