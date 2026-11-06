# SPDX-License-Identifier: Apache-2.0
"""Explicit native Connector selection is the PD feature switch."""

from dataclasses import dataclass, field

from ..config import KVAFConfig
from .initialization import validate_endpoint

WORKER_CLASS = "vllm_ascend.distributed.kv_transfer.kvaf.pd.worker.PDKVWorker"


@dataclass(frozen=True)
class PDConfig:
    decode_endpoints: tuple[str, ...] = ()
    cpu_eids: dict[str, str] = field(default_factory=dict)
    npu_eids: dict[str, str] = field(default_factory=dict)
    timeout: float = 120.0
    initialization_coordinator: str = ""  # Empty disables cross-host initialization admission.
    diagnostics_requests: int = 0
    verify_kv: bool = False

    def eids(self, physical_device_id):
        key = str(physical_device_id)
        return self.cpu_eids.get(key, ""), self.npu_eids.get(key, "")

    @classmethod
    def from_vllm(cls, config):
        transfer = config.kv_transfer_config
        if transfer is None or transfer.kv_connector != "PDKVoffloadConnector":
            raise ValueError("select PDKVoffloadConnector explicitly before loading its Worker")
        values = dict(transfer.kv_connector_extra_config)
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"unknown PDKVoffloadConnector settings: {sorted(unknown)}")
        if "decode_endpoints" in values:
            values["decode_endpoints"] = tuple(values["decode_endpoints"])
        result = cls(**values)
        if not isinstance(result.initialization_coordinator, str):
            raise ValueError("initialization_coordinator must be an empty string or host:port")
        if result.initialization_coordinator:
            validate_endpoint(result.initialization_coordinator)
            if transfer.is_kv_producer:
                raise ValueError("initialization_coordinator belongs on D consumers only")
        if type(result.diagnostics_requests) is not int or not 0 <= result.diagnostics_requests <= 16:
            raise ValueError("diagnostics_requests must be an integer in 0..16 (0 disables diagnostics)")
        if type(result.verify_kv) is not bool or result.verify_kv and not result.diagnostics_requests:
            raise ValueError("verify_kv requires diagnostics_requests > 0 and a boolean value")
        for routes in (result.cpu_eids, result.npu_eids):
            if not isinstance(routes, dict) or any(
                not isinstance(key, str)
                or not key.isdecimal()
                or str(int(key)) != key
                or not isinstance(value, str)
                or len(value) != 32
                or any(char not in "0123456789abcdefABCDEF" for char in value)
                for key, value in routes.items()
            ):
                raise ValueError("CPU/NPU EID maps require physical NPU ID keys and 32 hex digits per EID")
        if type(result.timeout) not in (int, float) or not 1 <= result.timeout <= 3600:
            raise ValueError("PD timeout must be in 1..3600 seconds")
        if len(set(result.decode_endpoints)) != len(result.decode_endpoints):
            raise ValueError("duplicate D control endpoints")
        for address in result.decode_endpoints:
            host, separator, port = address.rpartition(":")
            if not separator or not host or not port.isdecimal() or not 0 < int(port) < 65536:
                raise ValueError("D endpoints must be reachable host:port strings")
        if transfer.is_kv_producer and not result.decode_endpoints:
            raise ValueError("P must configure all D worker control endpoints for full-mesh registration")
        pc = config.parallel_config
        if pc.worker_cls != WORKER_CLASS:
            raise ValueError(f"PD KVAF requires --worker-cls {WORKER_CLASS}")
        backend = KVAFConfig.from_dict((config.additional_config or {}).get("kvaf_config", {}))
        if not backend.build_dir.strip() or backend.enabled == transfer.is_kv_producer:
            raise ValueError("configure kvaf_config.build_dir on P/D, enabled=false on P and enabled=true on D")
        if any(
            getattr(pc, name) != 1
            for name in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
        ):
            raise ValueError("PD KVAF requires PP=PCP=DCP=1")
        speculative = config.speculative_config
        if (
            speculative is None
            or speculative.num_speculative_tokens != 3
            or speculative.method not in ("mtp", "deepseek_mtp")
        ):
            raise ValueError("PD KVAF requires native MTP with three speculative tokens")
        if config.cache_config.enable_prefix_caching:
            # Native prefix hashes do not track whether a near-limit MTP
            # forward was skipped. Reusing those pages can expose stale KV.
            raise ValueError("disable native P/D prefix caching for PD KVAF")
        if transfer.kv_role not in ("kv_producer", "kv_consumer"):
            raise ValueError("PD KVAF requires a distinct producer or consumer role")
        return result
