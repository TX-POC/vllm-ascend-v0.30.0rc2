# SPDX-License-Identifier: Apache-2.0
"""Worker-only SDK setup and serialization shared with native Decode Get."""

import importlib
import os
from dataclasses import replace
from pathlib import Path
from threading import RLock

from ..adapter import BYTES_PER_GIB, SLOT_ALIGNMENT, KVAFAdapter
from ..cache_policy import CacheFeedbackManager
from ..layout import grouped_layouts
from ..transfer import KVTransfer, verify_put_block


class PDTransfer(KVTransfer):
    """Prevent native cancellation/preemption from freeing in-flight receives."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.receiving, self.deferred_release = set(), set()

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
                    "resolve_raw_addr",
                    "shutdown",
                ):
                    raise RuntimeError(
                        "PD Runtime retains an unresolved transfer; restart after recovery"
                    ) from self.failure
                return value(*args, **kwargs)

        return invoke


def load_sdk(build_dir):
    build_dir = Path(build_dir).expanduser().resolve()
    # HCCL owns device RA after native worker initialization. The separately
    # shipped SDK patch implements explicit borrowing before the first RaInit.
    # This affects only the explicitly selected external Worker process.
    os.environ["KVAF_REUSE_DEVICE_RA"] = "1"
    sdk = importlib.import_module("kvaf")
    if not getattr(getattr(sdk, "_native", None), "reuse_device_ra_supported", False):
        raise RuntimeError("build PD SDK with scripts/install_kvaf_pd.sh to borrow HCCL device RA explicitly")
    if not Path(sdk.__file__).resolve().is_relative_to(build_dir / "python" / "kvaf"):
        raise RuntimeError(f"KVAF imported outside configured build directory {build_dir}")
    for name in ("NpuRuntime", "CpuPullHbmSource", "NpuPullHbmSource"):
        if not hasattr(sdk, name):
            raise RuntimeError(f"build the cpu-pull-hbm-reference SDK with {name}")
    return sdk


class PDAdapter(KVAFAdapter):
    def __init__(self, *args, cpu_eid="", **kwargs):
        super().__init__(*args, **kwargs)
        self.cpu_eid = cpu_eid

    def bind_caches(self, kv_caches):
        self.writer = PDTransfer(
            self.runtimes,
            self.slot_counts,
            self.layouts,
            kv_caches,
            verify=verify_put_block if self.config.verify_put else None,
        )
        sources = {name: group for group in self.writer.groups for name in group.layout.layer_names}
        self.writer.groups = [
            replace(sources[layout.layer_names[0]], layout=layout, geometry=(656, 128))
            for layout in self.target_layouts
        ]
        self.writer.feedback = CacheFeedbackManager(self.writer, log_context=self._log_context)

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
        native = sdk.NpuRuntime(
            batch_get_kernel_object_path=str(kernels["get"]),
            batch_put_kernel_object_path=str(kernels["put"]),
            batch_get_raw_addr_kernel_object_path=str(kernels["get_raw_addr"]),
            batch_get_token_bytes=656,
            batch_get_tokens_per_block=128,
            device_id=self.device_id,
            npu_core_count=self.config.npu_core_count,
            jettys_per_core=self.config.jettys_per_core,
            enable_cpu_offload=True,
            npu_eids=[self.cpu_eid] if self.cpu_eid else [],
        )
        runtime = SerializedRuntime(native)
        self.runtimes[656, 128] = runtime
        self.slot_counts[656, 128] = slots
        try:
            with self._native_log("PD initialize DDR and dual Pull"):
                runtime.link_to(self.cpu_address)
                runtime.initialize()
                runtime.initialize_ddr(slots * slot_size, slot_size)
        except BaseException:
            self.close()
            raise
