# SPDX-License-Identifier: Apache-2.0
"""Select with native --worker-cls; no replacement of native module globals."""

from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_ascend.worker.worker import NPUWorker

from .config import PDConfig
from .runner import PDKVModelRunner


class PDKVWorker(NPUWorker):
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        get_kv_transfer_group().start_pd_control()
        return result

    def init_device(self):
        PDConfig.from_vllm(self.vllm_config)
        if self.use_v2_model_runner:
            raise ValueError("PD KVAF currently requires the native v1 Runner")
        self.device = self._init_device()
        init_workspace_manager(self.device, 1)
        self.model_runner = PDKVModelRunner(self.vllm_config, self.device)
        if self.rank == 0:
            report_usage_stats(self.vllm_config)
