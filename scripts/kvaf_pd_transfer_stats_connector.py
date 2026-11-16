# SPDX-License-Identifier: Apache-2.0
"""Opt-in request measurement through passive KVAF-PD observation points."""

import time
import weakref
from pathlib import Path

from kvaf_pd_performance import load_config
from vllm.logger import init_logger

from vllm_ascend.distributed.kv_transfer.kvaf.pd.connector import PDKVoffloadConnector as NativeConnector

logger = init_logger(__name__)


class PDKVoffloadConnector(NativeConnector):
    def __init__(self, *args, **kwargs):
        self._transfer_observer = self._batch_observer = None
        self._performance_config = None
        self._decode_batch_settings = self._decode_batch_gate = None
        super().__init__(*args, **kwargs)
        from vllm_ascend.distributed.kv_transfer.kvaf.pd.decode_batch import CONFIG_FILE_NAME, load_settings

        decode = load_settings(Path(__file__).with_name(CONFIG_FILE_NAME))
        if decode.enabled and not self.is_producer:
            pc = self.vllm_config.parallel_config
            if pc.tensor_parallel_size != 1 or pc.data_parallel_size != decode.dp_size:
                raise ValueError("decode batch requires D TP1 and matching DP size")
            if self.vllm_config.scheduler_config.max_num_seqs < decode.max_batch_size:
                raise ValueError("D --max-num-seqs must cover decode batch max_batch_size")
            required_tokens = decode.max_batch_size * (self.vllm_config.speculative_config.num_speculative_tokens + 1)
            if self.vllm_config.scheduler_config.max_num_batched_tokens < required_tokens:
                raise ValueError("D max-num-batched-tokens cannot fit the configured speculative decode batch")
            self._decode_batch_settings = decode
            if self.scheduler is not None:
                from kvaf_pd_decode_batch_metadata import BatchReceiveRequest

                self.scheduler.decode_batch_settings = decode
                self.scheduler.decode_seed_factory = BatchReceiveRequest
        try:
            config = load_config()
            if not config.active:
                return
            if self.config.diagnostics_requests or self.config.verify_kv:
                raise ValueError("request capture requires diagnostics_requests=0 and verify_kv=false")
            self._performance_config = config
            if not self.is_producer and self.scheduler is not None:
                from kvaf_pd_request_observer import SchedulerObserver

                from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import get_external_request_id
                from vllm_ascend.distributed.kv_transfer.kvaf.pd.observation import observed_call

                self._batch_observer = SchedulerObserver(
                    config, self._capture_header("D_scheduler"), get_external_request_id
                )
                self.scheduler.request_observer = self._batch_observer
                original = weakref.WeakMethod(self.scheduler._access_metaserver)
                observer = self._batch_observer

                def metaserver(url, message):
                    operation = original()
                    if operation is None:
                        raise RuntimeError("native scheduler owner is no longer alive")
                    return observed_call(
                        observer,
                        "D.metaserver_transport",
                        operation,
                        url,
                        message,
                        context=dict(request=message["request_id"], generation=message.get("kvaf_generation")),
                    )

                self.scheduler._access_metaserver = metaserver
        except Exception as error:
            self._performance_config = None
            logger.warning("KVAF performance capture disabled; native inference retained: %s", error)

    def bind_pd_backend(self, **kwargs):
        super().bind_pd_backend(**kwargs)
        if self._performance_config is not None:
            try:
                from kvaf_pd_request_observer import ProducerObserver, WorkerObserver

                if self.is_producer:
                    self._transfer_observer = ProducerObserver(
                        self.worker, self._performance_config, self._capture_header("P")
                    )
                    self.worker._call = self._transfer_observer
                    self.request_observer = self._transfer_observer
                else:
                    header = self._capture_header("D_worker")
                    header["layer_roles"] = {
                        name: "target" if memory.offload_main else "mtp" for name, memory in kwargs["memories"].items()
                    }
                    self.request_observer = WorkerObserver(self._performance_config, header, self._external_id)
                self.worker.request_observer = self.request_observer
            except Exception as error:
                logger.warning("KVAF performance observer disabled; native calls retained: %s", error)
        if self._decode_batch_settings is not None:
            from vllm_ascend.distributed.kv_transfer.kvaf.pd.decode_batch import Gate

            callback = weakref.WeakMethod(self._observe_decode_batch)

            def observe(*values):
                operation = callback()
                return operation(*values) if operation is not None else None

            self._decode_batch_gate = Gate(
                self._decode_batch_settings,
                self.vllm_config.parallel_config.data_parallel_rank,
                observe if self.request_observer is not None else None,
                logger=logger,
            )

    def _capture_header(self, role):
        from kvaf_pd_transfer_stats import capture_header

        header = capture_header(self.vllm_config, role)
        if self._decode_batch_settings is not None:
            settings = self._decode_batch_settings
            header["decode_batch_mode"] = dict(
                enabled=True,
                dp_size=settings.dp_size,
                max_batch_size=settings.max_batch_size,
                timeout_seconds=settings.timeout_seconds,
            )
        return header

    def _observe_decode_batch(self, event, request, generation, token):
        from vllm_ascend.distributed.kv_transfer.kvaf.pd.observation import notify

        if event == "begin":
            return notify(
                self.request_observer,
                "begin",
                name="D.decode_batch_wait",
                context=dict(request_id=request, generation=generation),
            )
        return notify(self.request_observer, "end", token=token, success=True)

    def start_load_kv(self, forward_context, **kwargs):
        result = super().start_load_kv(forward_context, **kwargs)
        if self._decode_batch_gate is not None:
            for req_id, seed in self._get_connector_metadata().requests.items():
                self._decode_batch_gate.bind(req_id, self._external_id(req_id), seed)
        return result

    def get_finished(self, finished_req_ids):
        if self._decode_batch_gate is None:
            return super().get_finished(finished_req_ids)
        finished = self.worker.get_finished()
        failed = (
            {
                request["id"]
                for request in self.worker.requests.values()
                if request["id"] in finished and (request["failed"] or request["cancelled"])
            }
            if finished
            else set()
        )
        released = self._decode_batch_gate.poll(finished, finished_req_ids, failed)
        self.worker.release(set(finished_req_ids))
        return set(), released

    def build_connector_meta(self, scheduler_output):
        metadata = super().build_connector_meta(scheduler_output)
        if self._batch_observer is not None:
            self._batch_observer.observe(scheduler_output)
        return metadata

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if self._batch_observer is None:
            return super().update_state_after_alloc(request, blocks, num_external_tokens)
        begin = time.perf_counter_ns()
        result = super().update_state_after_alloc(request, blocks, num_external_tokens)
        end = time.perf_counter_ns()
        try:
            seed = self.scheduler._pd_requests.get(request.request_id)
            if seed:
                token = self._batch_observer.record(
                    "begin",
                    name="D.block_bind_and_advertise",
                    context=dict(request_id=request.request_id, generation=seed.generation),
                    at_ns=begin,
                )
                if token:
                    self._batch_observer.record("end", token=token, success=True, at_ns=end)
        except Exception:
            self._batch_observer.capture.fail("allocation callback observation failed")
        return result

    def shutdown(self):
        try:
            if self._decode_batch_gate is not None:
                self._decode_batch_gate.close()
            super().shutdown()
        finally:
            for observer in (self._transfer_observer, self._batch_observer, self.request_observer):
                if observer is not None:
                    try:
                        observer.capture.close()
                    except Exception as error:
                        logger.warning("KVAF capture close failed; native shutdown retained: %s", error)
