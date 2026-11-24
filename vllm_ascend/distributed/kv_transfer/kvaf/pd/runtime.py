# SPDX-License-Identifier: Apache-2.0
"""Worker-only SDK setup and serialization shared with native Decode Get."""

import importlib
from pathlib import Path
from threading import RLock

from ..adapter import BYTES_PER_GIB, SLOT_ALIGNMENT, KVAFAdapter
from ..layout import grouped_layouts
from ..transfer import KVTransfer
from .initialization import initialization_slot


class PDTransfer(KVTransfer):
    """Prevent native cancellation/preemption from freeing in-flight receives."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.receiving, self.deferred_release = set(), set()

    def start_decode_put(self, device):
        # The receive path shares a Runtime and its lock with ordinary Get/Put.
        # A background Put cannot drain while the runner holds that lock.
        pass

    @property
    def lock(self):
        return self.runtimes[656, 128].lock

    def protect_receive(self, req_id):
        with self.lock:
            self.receiving.add(req_id)

    def complete_receive(self, req_id):
        with self.lock:
            self.receiving.discard(req_id)
            if req_id in self.deferred_release:
                self.deferred_release.discard(req_id)
                super().release_requests((req_id,))

    def release_requests(self, req_ids):
        with self.lock:
            ids = set(req_ids)
            if ids:
                self.runtimes[656, 128].wait_compute()
            self.deferred_release.update(ids & self.receiving)
            super().release_requests(ids - self.receiving)

    def prepare_requests(self, requests, scheduled_tokens):
        with self.lock:
            super().prepare_requests(requests, scheduled_tokens)

    def put_requests(self, requests, scheduled_tokens):
        with self.lock:
            super().put_requests(requests, scheduled_tokens)


class SerializedRuntime:
    def __init__(self, native):
        self.native = native
        self.lock = RLock()
        self.failure = None
        self.compute_done = None

    def wait_compute(self):
        if self.compute_done is not None:
            self.compute_done.synchronize()

    def __getattr__(self, name):
        value = getattr(self.native, name)
        if not callable(value):
            return value

        def invoke(*args, **kwargs):
            with self.lock:
                if self.failure is not None and name not in (
                    "query_cpu_pull",
                    "wait_cpu_pull",
                    "commit_put",
                    "abort_put",
                    "cancel_cpu_pull",
                    "resolve_raw_addr",
                    "cancel_raw_addr",
                    "shutdown",
                ):
                    raise RuntimeError(
                        "PD Runtime retains an unresolved transfer; restart after recovery"
                    ) from self.failure
                return value(*args, **kwargs)

        return invoke


def load_sdk(build_dir):
    build_dir = Path(build_dir).expanduser().resolve()
    sdk = importlib.import_module("kvaf")
    if not Path(sdk.__file__).resolve().is_relative_to(build_dir / "python" / "kvaf"):
        raise RuntimeError(f"KVAF imported outside configured build directory {build_dir}")
    for name in (
        "NpuRuntime",
        "CpuPullHbmSource",
        "NpuPullHbmSource",
        "CpuPullOptions",
        "NpuPullOptions",
        "ExecutionResources",
        "PutExecutionConfig",
        "GetExecutionConfig",
    ):
        if not hasattr(sdk, name):
            raise RuntimeError(f"PD requires unified KVAF main binding {name}")
    for name in (
        "listen_hbm",
        "cancel_raw_addr",
        "cancel_cpu_pull",
        "submit_cpu_pull",
        "batch_get_raw_addr",
        "graph_resources",
    ):
        if not hasattr(sdk.NpuRuntime, name):
            raise RuntimeError(f"KVAF main multi-MR Pull requires NpuRuntime.{name}; rebuild SDK and K together")
    if not hasattr(sdk.NpuPullOptions, "verify_completion"):
        raise RuntimeError("rebuild PD KVAF with scripts/install_kvaf_pd.sh for checked synchronous raw READ")
    for kind, field in (("ReadyToken", "mr_id"), ("CpuPullSourceDesc", "mr_length")):
        if not hasattr(sdk, kind) or not hasattr(getattr(sdk, kind), field):
            raise RuntimeError(f"KVAF main multi-MR Pull requires {kind}.{field}; rebuild SDK and K together")
    return sdk


class PDAdapter(KVAFAdapter):
    def __init__(
        self, *args, cpu_eid="", initialization_coordinator="", initialization_timeout=120.0, receive_layers, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cpu_eid = cpu_eid
        self.receive_layers = receive_layers
        self.initialization_coordinator = initialization_coordinator
        self.initialization_timeout = initialization_timeout

    def bind_caches(self, kv_caches):
        self._bind_caches(kv_caches, PDTransfer)

    def _create_runtime(self, sdk, kernels, token_bytes, block_size, **options):
        return super()._create_runtime(
            sdk,
            kernels,
            token_bytes,
            block_size,
            cpu_pull_options=sdk.CpuPullOptions(timeout_ms=int(self.initialization_timeout * 1000)),
            npu_pull_options=sdk.NpuPullOptions(
                kernel_path=str(kernels["get_raw_addr"]),
                cores=self.config.npu_core_count,
                jettys=self.config.jettys_per_core,
                # Each connection can write into one stable arena per model layer.
                max_output_bindings=self.receive_layers,
                verify_completion=True,
            ),
            npu_eids=[self.cpu_eid] if self.cpu_eid else [],
            **options,
        )

    def connect(self):
        if self.runtimes:
            raise RuntimeError("PD Runtime already connected")
        self.target_layouts = grouped_layouts(self.layouts, self.groups)
        if not self.target_layouts or any(
            len(layout.layer_names) != 1 or layout.block_size != 128 or layout.planes[0].token_bytes != 656
            for layout in self.target_layouts
        ):
            raise ValueError("PD requires singleton C8 Target DDR groups")
        sdk = load_sdk(self.config.build_dir)
        build_dir = Path(self.config.build_dir).expanduser().resolve()
        kernels = {
            kind: build_dir / "bin" / f"kvaf_execute_batch_{kind}_kernels.o" for kind in ("get", "put", "get_raw_addr")
        }
        for path in kernels.values():
            if not path.is_file():
                raise FileNotFoundError(f"missing KVAF kernel {path}")
        payload = 128 * 656
        slot_size = (payload + SLOT_ALIGNMENT - 1) // SLOT_ALIGNMENT * SLOT_ALIGNMENT
        slots = int(self.config.ddr_size_gb * BYTES_PER_GIB) // slot_size
        if slots < len(self.target_layouts):
            raise ValueError("DDR budget cannot hold one page per Target layer")
        native = self._create_runtime(sdk, kernels, 656, 128)
        runtime = SerializedRuntime(native)
        self.runtimes[656, 128] = runtime
        self.slot_counts[656, 128] = slots
        try:
            with (
                initialization_slot(
                    self.initialization_coordinator, self.cpu_address, timeout=self.initialization_timeout
                ),
                self._native_log("PD initialize DDR and dual Pull"),
            ):
                runtime.link_to(self.cpu_address)
                runtime.initialize()
                runtime.initialize_ddr(slots * slot_size, slot_size)
                # Main allocates CPU Pull resources lazily at ticket export.
                runtime.offload_region()
        except BaseException:
            self.close()
            raise
