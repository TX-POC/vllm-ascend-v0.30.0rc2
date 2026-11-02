# SPDX-License-Identifier: Apache-2.0
"""Own native KVAF sessions and synchronous Target KV writes to DDR."""

import atexit
import importlib
import logging
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import KVAFConfig
from .layout import BLOCK_SIZE, MAX_QUERIES, NATIVE_SLOT_CAPACITY, TOPK, KVCacheLayout, grouped_layouts, runtime_weights
from .native_log import NativeLog
from .transfer import KVTransfer, verify_put_block

# vLLM installs its handler on this namespace; avoid a vLLM dependency in
# the small connection layer so its ownership checks also run on a host CPU.
logger = logging.getLogger("vllm")
SLOT_ALIGNMENT = 4096
BYTES_PER_GIB = 1 << 30


class KVAFAdapter:
    def __init__(
        self,
        config: KVAFConfig,
        device_id: int,
        layouts: tuple[KVCacheLayout, ...],
        *,
        log_dir: str | Path,
        tp_rank: int,
        dp_rank: int,
        reuse_device_ra: bool = False,
        groups=None,
    ):
        self.config = config
        self.device_id = device_id  # Logical NPU ID; KVAF resolves the physical device itself.
        self.reuse_device_ra = reuse_device_ra
        self.cpu_address = config.cpu_address
        if config.cpu_addresses:
            # Resolve after device binding, and only when per-card routing is requested.
            from vllm.platforms import current_platform

            physical_id = current_platform.visible_device_id_to_physical_device_id(device_id)
            self.cpu_address = config.cpu_addresses.get(str(physical_id), config.cpu_address)
            logger.info(
                "KVAF CPU route: logical_device=%s physical_device=%s cpu=%s", device_id, physical_id, self.cpu_address
            )
        self.layouts = layouts
        self.groups = groups
        self.target_layouts = ()
        self.runtimes: dict[tuple[int, int], Any] = {}
        self.slot_counts: dict[tuple[int, int], int] = {}
        self.writer: KVTransfer | None = None
        self.log_path = (
            Path(log_dir).expanduser().resolve()
            / "kvaf"
            / f"kvaf_DP{dp_rank}_TP{tp_rank}_device{device_id}_pid{os.getpid()}.log"
        )
        self._log_context = f"DP{dp_rank} TP{tp_rank} NPU{device_id} pid={os.getpid()}"

        self._native_logs = NativeLog(self.log_path)

    @contextmanager
    def _native_log(self, operation: str):
        """Annotate startup/shutdown; the persistent relay also captures Put."""
        self._native_logs.start()
        with self.log_path.open("a", encoding="utf-8", buffering=1) as output:
            output.write(f"\nDEBUG {time.strftime('%m-%d %H:%M:%S')} [{self._log_context}] {operation}\n")
            try:
                yield
            except BaseException:
                traceback.print_exc(file=output)
                raise

    def connect(self) -> None:
        """Initialize one session per distinct (token bytes, tokens per block)."""
        if self.runtimes:
            raise RuntimeError("KVAF adapter is already connected")
        self.target_layouts = (
            grouped_layouts(self.layouts, self.groups)
            if self.groups
            else tuple(layout for layout in self.layouts if "target" in layout.roles and layout.cache_kind == "main")
        )
        read_sizes = set()
        for layout in self.target_layouts:
            if "target" not in layout.roles or layout.cache_kind != "main":
                continue
            if len(layout.planes) != 1 or not layout.planes[0].direct_token_read:
                raise ValueError(f"KVAF requires a single token-contiguous Target cache tensor: {layout.layer_names}")
            read_sizes.add((layout.planes[0].token_bytes, layout.block_size))
        if not read_sizes:
            raise ValueError("KVAF found no Target cache")

        build_dir = Path(self.config.build_dir).expanduser().resolve()
        kernels = {kind: build_dir / "bin" / f"kvaf_execute_batch_{kind}_kernels.o" for kind in ("get", "put")}
        for path in kernels.values():
            if not path.is_file():
                raise FileNotFoundError(f"Missing KVAF kernel: {path}; run bash scripts/install_kvaf.sh")
        # Worker-isolated imports: the ordinary vLLM startup needs no KVAF SDK.
        importlib.import_module("torch_npu")
        sdk = importlib.import_module("kvaf")
        if not hasattr(sdk, "NpuRuntime"):
            raise RuntimeError("Installed kvaf has no NpuRuntime; install the NPU build with Get and Put enabled")
        if not Path(sdk.__file__).resolve().is_relative_to(build_dir / "python" / "kvaf"):
            raise RuntimeError(f"Imported kvaf from {sdk.__file__}, outside {build_dir}; rerun scripts/install_kvaf.sh")

        weights = runtime_weights(self.layouts, self.target_layouts, SLOT_ALIGNMENT)
        total_budget = int(self.config.ddr_size_gb * BYTES_PER_GIB)
        logger.info(
            "KVAF connecting: device=%s cpu=%s; native log: %s",
            self.device_id,
            self.cpu_address,
            self.log_path,
        )
        try:
            for token_bytes, block_size in sorted(read_sizes):
                # Slot allocation padding is not payload length. Later Put/Get
                # calls must use token_bytes * block_size, not slot_size.
                slot_size = ((token_bytes * block_size + SLOT_ALIGNMENT - 1) // SLOT_ALIGNMENT) * SLOT_ALIGNMENT
                budget = int(total_budget * weights[token_bytes, block_size] / sum(weights.values()))
                ddr_bytes = budget // slot_size * slot_size
                if ddr_bytes < slot_size:
                    raise ValueError("KVAF DDR budget is too small for one slot per Target layout")
                logger.debug(
                    "KVAF layout: device=%s token_bytes=%s block_size=%s DDR_bytes=%s",
                    self.device_id,
                    token_bytes,
                    block_size,
                    ddr_bytes,
                )
                with self._native_log(
                    f"initialize token_bytes={token_bytes} block_size={block_size} DDR_bytes={ddr_bytes}"
                ):
                    runtime = sdk.NpuRuntime(
                        batch_get_kernel_object_path=str(kernels["get"]),
                        batch_get_token_bytes=token_bytes,
                        batch_get_tokens_per_block=block_size,
                        device_id=self.device_id,
                        npu_core_count=self.config.npu_core_count,
                        batch_put_kernel_object_path=str(kernels["put"]),
                        jettys_per_core=self.config.jettys_per_core,
                        reuse_device_ra=self.reuse_device_ra,
                    )
                    self.runtimes[token_bytes, block_size] = runtime
                    runtime.link_to(self.cpu_address)
                    runtime.initialize(ddr_bytes, slot_size)
                    self.slot_counts[token_bytes, block_size] = ddr_bytes // slot_size
        except Exception as exc:
            self.close()
            logger.error(
                "KVAF initialization failed: device=%s: %s; native log: %s", self.device_id, exc, self.log_path
            )
            raise RuntimeError(
                f"KVAF initialization failed on logical NPU {self.device_id} to {self.cpu_address}: {exc}; "
                f"native log: {self.log_path}"
            ) from exc
        atexit.register(self.close)
        logger.info(
            "KVAF connected: device=%s sessions=%s layouts=%s; Target uses transient prefill and hot decode storage; "
            "native startup/Put logs redirected: %s",
            self.device_id,
            len(self.runtimes),
            len(self.layouts),
            self.log_path,
        )

    def bind_caches(self, kv_caches: dict[str, Any]) -> None:
        from .cache_policy import CacheFeedbackManager

        self.writer = KVTransfer(
            self.runtimes,
            self.slot_counts,
            self.layouts,
            kv_caches,
            verify=verify_put_block if self.config.verify_put else None,
        )
        if self.groups:
            sources = {name: layer for layer in self.writer.groups for name in layer.layout.layer_names}
            self.writer.groups = [
                replace(
                    sources[layout.layer_names[0]],
                    layout=layout,
                    geometry=(layout.planes[0].token_bytes, layout.block_size),
                )
                for layout in self.target_layouts
            ]
        self.writer.feedback = CacheFeedbackManager(self.writer, log_context=self._log_context)
        logger.info(
            "KVAF BatchPut enabled: stable Target blocks only; independent hot/staging storage; verify_put=%s",
            self.config.verify_put,
        )
        logger.info("KVAF Target workspace and DDR prefix archive will bind before graph capture")

    def close(self) -> None:
        """Release every session on normal shutdown, including partial startup."""
        if self.writer is not None and self.writer.feedback is not None:
            try:
                self.writer.feedback.close()
            except Exception:
                logger.exception("KVAF LRU feedback failed during shutdown")
            finally:
                self.writer.feedback = None
        session_count = len(self.runtimes)
        for geometry, runtime in list(self.runtimes.items()):
            try:
                with self._native_log(f"shutdown token_bytes={geometry[0]} block_size={geometry[1]}"):
                    runtime.shutdown()
            except Exception:
                logger.exception(
                    "KVAF shutdown failed for device=%s layout=%s; native log: %s",
                    self.device_id,
                    geometry,
                    self.log_path,
                )
            else:
                del self.runtimes[geometry]
                self.slot_counts.pop(geometry, None)
        if not self.runtimes:
            self._native_logs.close()
            self.writer = None
            atexit.unregister(self.close)
            if session_count:
                logger.info(
                    "KVAF released: device=%s sessions=%s; native log: %s", self.device_id, session_count, self.log_path
                )


class NativeCacheAdapter:
    def __init__(self, pool, *, native=None):
        import torch

        if pool.max_tokens > NATIVE_SLOT_CAPACITY or pool.max_tokens < 2 * TOPK:
            raise ValueError(f"LIDU requires {2 * TOPK}..{NATIVE_SLOT_CAPACITY} map entries including spare slots")
        if native is None:
            from nanovllm_dsa_lidu_fused_batch_get_prepare_mtp import lidu_fused_batch_get_prepare_mtp_out

            native = lidu_fused_batch_get_prepare_mtp_out
        self.prepare_op, self.pool = native, pool
        layers, batch = len(pool.names), pool.max_requests
        self.max_queries = batch * MAX_QUERIES
        self.qcap = (pool.max_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        self.enabled = False
        self.batch = 0
        self.map_groups = list(range(layers))
        self.group_count = layers
        self.inputs = {}
        self.done = [None, None]
        self.turn = 0

        def tensor(shape, fill=0, dtype=torch.int32):
            import torch

            return torch.full(shape, fill, dtype=dtype, device=pool.device)

        self.cursors = tensor((layers, batch))
        self.profile = tensor((1,))
        for name, shape, fill in (
            ("entries", (batch,), batch),
            ("query_ends", (batch,), 0),
            ("budgets", (batch,), 0),
            ("lengths", (batch,), 0),
            ("logical", (batch,), 1),
            ("blocks", (layers, batch, self.qcap), 0),
            ("miss_counts", (layers, batch + 1), -1),
        ):
            self.inputs[name] = tensor(shape, fill)
        self.destinations = tensor((layers, self.max_queries, 1, TOPK), -1)
        # Get subclasses consume these descriptors. Reuse scratch across
        # sequential layer calls; keep destinations/counts independently.
        self.indices = tensor((batch * TOPK * MAX_QUERIES,))
        self.offsets = torch.empty_like(self.indices)
        self.output_indices = torch.empty_like(self.indices)
        self.resident_lengths = tensor((batch,))
        self.profiling = tensor((batch, 8, 32), dtype=torch.int64)

    def bind(self, manager):
        self.map_groups = [self.pool.indices[layer.layout.layer_names[0]] for layer in manager.writer.groups]
        self.group_count = len(self.map_groups)

    @property
    def nbytes(self):
        import torch

        tensors = [value for value in vars(self).values() if isinstance(value, torch.Tensor)]
        return sum(t.numel() * t.element_size() for t in (*tensors, *self.inputs.values()))

    def run(self, layer, topk, metadata):
        """Called immediately after LI; all indexing below creates views only."""
        if metadata.num_prefills:
            return
        batch = metadata.cum_query_lens.shape[0]
        queries = topk.shape[0]
        if topk.dim() == 2:
            topk = topk.unsqueeze(1)
        if queries > self.max_queries or batch > self.pool.max_requests:
            raise ValueError("LIDU decode shape exceeds preallocated capacity")
        maps = self.pool.views[self.pool.names[self.map_groups[layer]]]
        descriptor_count = batch * TOPK * MAX_QUERIES
        self.prepare_op(
            topk,
            self.inputs["query_ends"][:batch],
            self.inputs["entries"][:batch],
            maps.token_to_slot,
            maps.slot_to_token,
            self.cursors[layer],
            self.inputs["budgets"][:batch],
            self.inputs["lengths"][:batch],
            (self.spare_slots[layer] if self.spare_slots.dim() == 3 else self.spare_slots)[:batch],
            (self.spare_bitmap[layer] if self.spare_bitmap.dim() == 3 else self.spare_bitmap)[:batch],
            self.inputs["blocks"][layer, :batch],
            self.inputs["logical"][:batch],
            self.profile,
            self.destinations[layer, :queries],
            self.inputs["miss_counts"][layer, : batch + 1],
            self.indices[:descriptor_count],
            self.offsets[:descriptor_count],
            self.output_indices[:descriptor_count],
            self.resident_lengths[:batch],
            self.profiling[:batch],
            layer,
            self.group_count,
            self.pool.max_requests,
            self.qcap,
        )

    def finish(self, manager):
        if not self.enabled:
            return
        queries = int(manager.groups[0].query_offsets[-1])
        manager.submit_feedback(
            self.destinations[: self.group_count, :queries],
            generation=manager.generation,
            miss_counts=self.inputs["miss_counts"][: self.group_count, : self.batch + 1],
        )
