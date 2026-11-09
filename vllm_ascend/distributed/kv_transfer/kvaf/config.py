# SPDX-License-Identifier: Apache-2.0
"""Connection settings; DDR budget is per vLLM worker, not per TP group."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class KVAFConfig:
    enabled: bool = False
    cpu_address: str = "127.0.0.1:28999"
    cpu_addresses: dict[str, str] = field(default_factory=dict)  # Physical NPU ID -> host:port; overrides cpu_address.
    build_dir: str = ""
    ddr_size_gb: float = 1.0
    hot_tokens: int = 8192  # Per request/layer; excludes the computed ABC tail.
    put_interval_steps: int = 32
    put_overlap_steps: int = 8
    npu_core_count: int = 2
    jettys_per_core: int = 2
    transport: str = "urma"  # KVAF transport, fixed for the worker lifetime.
    decode_backend: str = "batch_get"  # batch_get or li_fetch; fixed for the worker lifetime.
    intermediate_fetches: int = 8  # LIFetch windows, including the initial hot window; 1..32.
    prefetch_tokens: int = 512  # Per request/round; divided equally across LIFetch windows.
    verify_put: bool = False  # Read back one prefill and one decode block per worker for byte comparison.

    def __post_init__(self) -> None:
        if self.transport not in ("urma", "ubmem"):
            raise ValueError("kvaf_config.transport must be urma or ubmem")
        if self.decode_backend not in ("batch_get", "li_fetch"):
            raise ValueError("kvaf_config.decode_backend must be batch_get or li_fetch")
        if type(self.intermediate_fetches) is not int or not 1 <= self.intermediate_fetches <= 32:
            raise ValueError("kvaf_config.intermediate_fetches must be an integer in 1..32")
        if type(self.prefetch_tokens) is not int or not 0 <= self.prefetch_tokens <= 2048:
            raise ValueError("kvaf_config.prefetch_tokens must be an integer in 0..2048")
        if self.decode_backend == "li_fetch" and (
            self.prefetch_tokens % self.intermediate_fetches
            or self.prefetch_tokens // self.intermediate_fetches > 480
        ):
            raise ValueError("prefetch_tokens must divide evenly into windows, with at most 480 tokens/window")
        if type(self.enabled) is not bool:
            raise ValueError("kvaf_config.enabled must be true or false")
        if type(self.verify_put) is not bool:
            raise ValueError("kvaf_config.verify_put must be true or false")
        if not isinstance(self.build_dir, str) or (self.enabled and not self.build_dir.strip()):
            raise ValueError("kvaf_config.build_dir must name the KVAF build-npu directory")
        if not isinstance(self.cpu_addresses, dict) or any(
            not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key for key in self.cpu_addresses
        ):
            raise ValueError("kvaf_config.cpu_addresses must map physical NPU IDs (e.g. '4') to host:port")
        for address in (self.cpu_address, *self.cpu_addresses.values()):
            if not isinstance(address, str):
                raise ValueError("KVAF CPU addresses must be host:port strings")
            host, separator, port = address.rpartition(":")
            if not separator or not host or not port.isdigit() or not 0 < int(port) <= 65535:
                raise ValueError("KVAF CPU addresses must be host:port with port in 1..65535")
        if type(self.ddr_size_gb) not in (int, float) or not isfinite(self.ddr_size_gb) or self.ddr_size_gb <= 0:
            raise ValueError("kvaf_config.ddr_size_gb must be finite and positive (per worker)")
        if type(self.hot_tokens) is not int or self.hot_tokens < 8192 or self.hot_tokens % 128:
            raise ValueError("kvaf_config.hot_tokens must be an integer >= 8192 and a multiple of 128")
        if self.decode_backend == "li_fetch" and self.hot_tokens % 256:
            raise ValueError("li_fetch requires hot_tokens aligned to the original kernel's 256-token tiles")
        for name in ("put_interval_steps", "put_overlap_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"kvaf_config.{name} must be a positive integer")
        # Limits of the pinned KVAF native runtime.
        for name, maximum in (("npu_core_count", 48), ("jettys_per_core", 64)):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= maximum:
                raise ValueError(f"kvaf_config.{name} must be an integer in 1..{maximum}")

    def abc_tokens(self, max_queries):
        from .layout import BLOCK_SIZE, TAIL

        tokens = TAIL + max_queries * (self.put_interval_steps + self.put_overlap_steps)
        return (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE

    @classmethod
    def from_dict(cls, values: Any) -> "KVAFConfig":
        if not isinstance(values, dict):
            raise ValueError("additional_config.kvaf_config must be an object")
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown kvaf_config settings: {sorted(unknown)}")
        return cls(**values)
