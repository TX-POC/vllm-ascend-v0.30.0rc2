# SPDX-License-Identifier: Apache-2.0
"""Opt-in measurement Connector. Keep the original Worker and SDK unchanged."""

from kvaf_pd_transfer_stats import BatchObserver, PullObserver, capture_header

from vllm_ascend.distributed.kv_transfer.kvaf.pd.connector import PDKVoffloadConnector as NativeConnector


class PDKVoffloadConnector(NativeConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.diagnostics_requests or self.config.verify_kv:
            raise ValueError("latency capture requires diagnostics_requests=0 and verify_kv=false on P/D")
        self._transfer_observer = None
        self._batch_observer = None
        if not self.is_producer and self.scheduler is not None:
            self._batch_observer = BatchObserver(
                capture_header(self.vllm_config, "D_scheduler"),
                (self.vllm_config.additional_config or {}).get("kvaf_config", {}),
            )

    def bind_pd_backend(self, **kwargs):
        super().bind_pd_backend(**kwargs)
        if self.is_producer and self.worker.active:
            self._transfer_observer = PullObserver(self.worker, capture_header(self.vllm_config, "P"))
            # Decorate this instance only. The original native call still owns
            # all arguments, returns, retries, exceptions and DMA completion.
            self.worker._call = self._transfer_observer

    def build_connector_meta(self, scheduler_output):
        metadata = super().build_connector_meta(scheduler_output)
        if self._batch_observer is not None:
            self._batch_observer.observe(scheduler_output)
        return metadata
