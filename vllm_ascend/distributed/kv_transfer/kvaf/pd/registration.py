# SPDX-License-Identifier: Apache-2.0
"""Independent layer MRs sharing one CPU/NPU source pair per D peer."""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Any
from uuid import uuid4

from .layout import LayerLayout

PULL_API_VERSION = 2
MAX_SOURCE_MRS = 4096


def batch_identity(path, ready):
    source = ready.source
    if path == "cpu":
        identity = source.source_id, source.region_id, source.region_epoch
    elif path == "npu":
        identity = source.source_id, source.source_epoch
    else:
        raise ValueError("source path must be cpu or npu")
    return identity, ready.sequence, ready.mr_id


@dataclass(frozen=True)
class SourceMetadata:
    peer: str
    layout: LayerLayout
    group: str
    mr_id: int
    region_id: int
    region_epoch: int

    def check_ready(self, path: str, ready: Any):
        identity, sequence, mr_id = batch_identity(path, ready)
        if mr_id != self.mr_id or sequence <= 0 or any(value <= 0 for value in identity):
            raise ValueError("Ready does not belong to the registered P layer/MR")
        if path == "cpu" and (
            (ready.source.region_id, ready.source.region_epoch) != (self.region_id, self.region_epoch)
            or ready.source.mr_length != self.layout.nbytes
        ):
            raise ValueError("Ready does not belong to the registered DDR region/MR length")
        return identity


class LayerRegistration:
    """Business layer view; hardware and batch ownership belong to its group."""

    def __init__(self, owner, name, memory, mr_id):
        self.owner, self.tensor = owner, memory.tensor
        self.layout = memory.describe(name)
        self.metadata = SourceMetadata(
            owner.peer, self.layout, owner.identity, mr_id, owner.ticket["region_id"], owner.ticket["epoch"]
        )

    def ready(self, path, production_event):
        owner = self.owner
        if not owner.registered or owner.closed or path in owner.batches:
            raise RuntimeError("source unavailable or previous shared-source batch unresolved")
        production_event.synchronize()
        # Pin even a Ready whose reply is lost. No other MR may replace it.
        owner.batches[path] = None
        ready = owner.source(path).ready(self.tensor)
        identity = self.metadata.check_ready(path, ready)
        if owner.identities.setdefault(path, identity) != identity:
            raise ValueError("SDK source identity changed inside a shared MR group")
        owner.batches[path] = ready
        return ready

    def release(self, path, ready):
        self._check_batch(path, ready)
        self.owner.source(path).release_batch(ready)
        del self.owner.batches[path]

    def cancel(self, path, ready):
        self._check_batch(path, ready)
        self.owner.source(path).cancel_batch(ready)

    def _check_batch(self, path, ready):
        current = self.owner.batches.get(path)
        self.metadata.check_ready(path, ready)
        if current is None or batch_identity(path, current) != batch_identity(path, ready):
            raise ValueError("completion does not match the active source/sequence/MR")


class PeerRegistration:
    def __init__(self, sdk, peer, memories, ticket, *, device_id, cpu_eid, npu_eid):
        if not peer or not 1 <= len(memories) <= MAX_SOURCE_MRS or ticket.get("version") != 2:
            raise ValueError("multi-MR sources require a peer, 1..4096 layers and DDR ticket v2")
        self.peer, self.ticket, self.identity = peer, ticket, uuid4().hex
        self.layers = {
            name: LayerRegistration(self, name, memory, i) for i, (name, memory) in enumerate(memories.items(), 1)
        }
        ranges = sorted(
            (view.layout.pa_base, view.layout.pa_base + view.layout.nbytes) for view in self.layers.values()
        )
        if any(end > following for (_, end), (following, _) in zip(ranges, ranges[1:])):
            raise ValueError("registered layer tensors must not overlap")
        self.cpu = sdk.CpuPullHbmSource(device_id=device_id, npu_eid=cpu_eid)
        self.npu = sdk.NpuPullHbmSource(device_id=device_id, npu_eid=npu_eid)
        self.batches = {}
        self.identities = {}
        self.lock = Lock()
        self.failure = None
        self.stage = "created"
        self.attempted = self.cpu_registered = self.registered = self.closed = False
        self.npu_closed = self.cpu_closed = False

    def register(self, receiver, stream):
        if self.attempted or self.closed:
            raise RuntimeError("source registration cannot repeat; resolve/close the original group")
        self.attempted = True
        tensors = [view.tensor for view in self.layers.values()]
        self.stage = "cpu_register"
        self.cpu.register(self.ticket, tensors, stream)
        self.cpu_registered = True
        self.stage = "npu_register"
        self.npu.register(tensors, receiver=receiver, stream=stream)
        self.registered = True
        self.stage = "registered"

    @contextmanager
    def operation(self, timeout):
        if not self.lock.acquire(timeout=timeout):
            raise TimeoutError("shared source group is still occupied; retain the waiting layer")
        try:
            if self.failure is not None or self.closed:
                raise RuntimeError("shared source group retains an unresolved operation") from self.failure
            try:
                yield
            except Exception as exc:
                self.failure = exc
                raise
        finally:
            self.lock.release()

    def source(self, path):
        if path not in ("cpu", "npu"):
            raise ValueError("source path must be cpu or npu")
        return self.cpu if path == "cpu" else self.npu

    def close(self):
        if self.batches:
            raise RuntimeError("shared source still has active or unresolved Ready batches")
        if not self.npu_closed:
            self.npu.shutdown()  # SDK coordinates reader retirement, including partial register failures.
            self.npu_closed = True
        if not self.cpu_closed:
            self.cpu.shutdown()
            self.cpu_closed = True
        self.closed = True
