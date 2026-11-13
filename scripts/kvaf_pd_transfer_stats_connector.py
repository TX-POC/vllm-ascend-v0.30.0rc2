# SPDX-License-Identifier: Apache-2.0
"""Opt-in request measurement through passive KVAF-PD observation points."""

import time
import weakref

from kvaf_pd_performance import load_config
from vllm.logger import init_logger

from vllm_ascend.distributed.kv_transfer.kvaf.pd.connector import PDKVoffloadConnector as NativeConnector

logger = init_logger(__name__)


class PDKVoffloadConnector(NativeConnector):
    def __init__(self, *args, **kwargs):
        self._transfer_observer = self._batch_observer = None
        self._performance_config = None
        super().__init__(*args, **kwargs)
        try:
            config = load_config()
            if not config.active:
                return
            if self.config.diagnostics_requests or self.config.verify_kv:
                raise ValueError("request capture requires diagnostics_requests=0 and verify_kv=false")
            self._performance_config = config
            if not self.is_producer and self.scheduler is not None:
                from kvaf_pd_request_observer import SchedulerObserver
                from kvaf_pd_transfer_stats import capture_header

                from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import get_external_request_id
                from vllm_ascend.distributed.kv_transfer.kvaf.pd.observation import observed_call

                self._batch_observer = SchedulerObserver(
                    config, capture_header(self.vllm_config, "D_scheduler"), get_external_request_id
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
                from kvaf_pd_transfer_stats import capture_header

                if self.is_producer:
                    self._transfer_observer = ProducerObserver(
                        self.worker, self._performance_config, capture_header(self.vllm_config, "P")
                    )
                    self.worker._call = self._transfer_observer
                    self.request_observer = self._transfer_observer
                else:
                    header = capture_header(self.vllm_config, "D_worker")
                    header["layer_roles"] = {
                        name: "target" if memory.offload_main else "mtp" for name, memory in kwargs["memories"].items()
                    }
                    self.request_observer = WorkerObserver(self._performance_config, header, self._external_id)
                self.worker.request_observer = self.request_observer
            except Exception as error:
                logger.warning("KVAF performance observer disabled; native calls retained: %s", error)

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
            super().shutdown()
        finally:
            for observer in (self._transfer_observer, self._batch_observer, self.request_observer):
                if observer is not None:
                    try:
                        observer.capture.close()
                    except Exception as error:
                        logger.warning("KVAF capture close failed; native shutdown retained: %s", error)
