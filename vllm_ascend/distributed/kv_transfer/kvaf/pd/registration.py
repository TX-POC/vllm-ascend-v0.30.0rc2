# SPDX-License-Identifier: Apache-2.0
"""Per-layer, per-D-peer source ownership for the KVAF Pull SDK.

The control transport forwards SourceMetadata as business PA layout plus opaque
SDK descriptors. UBA export/import and remote synchronization stay inside KVAF.
Create a distinct LayerRegistration for each D peer; NPU sources are single-reader.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .layout import MAX_INT64, LayerLayout


@dataclass(frozen=True)
class SourceMetadata:
    peer: str
    layout: LayerLayout
    cpu_descriptor: Any
    npu_ticket: Any
    npu_address: str
    registered_range: tuple[int, int] | None = None  # PA base/bytes of the actual source arena.

    @property
    def source_bias(self):
        if self.registered_range is None:
            return 0
        base, size = self.registered_range
        if (
            type(base) is not int
            or type(size) is not int
            or min(base, size) <= 0
            or base + size > MAX_INT64
            or self.layout.pa_base < base
            or self.layout.pa_base + self.layout.nbytes > base + size
            or self.cpu_descriptor.source_bytes != size
            or self.npu_ticket["bytes"] != size
        ):
            raise ValueError("layer must be contained in the actual CPU/NPU registered source range")
        return self.layout.pa_base - base

    @property
    def binding_key(self):
        return (
            self.peer,
            self.cpu_descriptor.region_id,
            self.cpu_descriptor.region_epoch,
            self.cpu_descriptor.source_id,
            self.npu_ticket["source_id"],
            self.npu_ticket["epoch"],
        )

    def check_ready(self, path: str, ready: Any) -> None:
        source = ready.source
        if path == "cpu":
            fields = ("source_id", "source_bytes", "region_id", "region_epoch")
            valid = all(getattr(source, name) == getattr(self.cpu_descriptor, name) for name in fields)
        elif path == "npu":
            valid = (source.source_id, source.source_epoch, source.source_bytes) == (
                self.npu_ticket["source_id"],
                self.npu_ticket["epoch"],
                self.npu_ticket["bytes"],
            )
        else:
            raise ValueError("unknown source path")
        if not valid:
            raise ValueError("Ready does not belong to the registered P layer/peer")

    def attach(self, runtime: Any, *, peer: str, local_eid: str) -> None:
        if peer != self.peer:
            raise ValueError("source registration belongs to another D peer")
        _ = self.source_bias  # Validate the registered range before creating any binding.
        # The SDK verifies CPU region/epoch and the NPU ticket. Never infer
        # UBA from PA, and retain this metadata if attach/detach is uncertain.
        runtime.attach_cpu_source(self.cpu_descriptor)
        runtime.attach_hbm(self.npu_address, self.npu_ticket, local_eid=local_eid)


class LayerRegistration:
    def __init__(
        self,
        sdk: Any,
        peer: str,
        layer_name: str,
        layer_tensor: Any,
        planes: Mapping[str, Any],
        *,
        device_id: int,
        cpu_eid: str,
        npu_eid: str,
    ):
        if not peer:
            raise ValueError("D peer identity is required")
        self.layout = LayerLayout.from_tensors(layer_name, layer_tensor, planes)
        self.peer, self.tensor = peer, layer_tensor
        self.cpu = sdk.CpuPullHbmSource(device_id=device_id, npu_eid=cpu_eid)
        self.npu = sdk.NpuPullHbmSource(device_id=device_id, npu_eid=npu_eid)
        self.metadata: SourceMetadata | None = None
        self._attempted = False
        self._batches: dict[str, Any] = {}
        self._cpu_closed = self._npu_closed = False
        # Preserve partial-registration descriptors until explicit shutdown.
        self.cpu_descriptor = self.npu_ticket = None

    def register(self, ddr_ticket: Any, stream: Any, *, listen_address: str, advertised_address: str) -> SourceMetadata:
        if self._attempted or self._cpu_closed or self._npu_closed:
            raise RuntimeError(
                "registration already attempted; retain resources and resolve/close the original sources"
            )
        if not listen_address or not advertised_address or advertised_address.startswith(("0.0.0.0:", "[::]:")):
            raise ValueError("advertise a reachable NPU control address")
        self._attempted = True
        self.cpu_descriptor = self.cpu.register(ddr_ticket, self.tensor, stream)
        self.npu_ticket = self.npu.register(self.tensor, stream)
        port = self.npu.listen(listen_address)
        host, _, requested_port = advertised_address.rpartition(":")
        if requested_port == "0":
            if type(port) is not int or not 0 < port <= 65535:
                raise RuntimeError("SDK did not return the bound source port")
            advertised_address = f"{host}:{port}"
        self.metadata = SourceMetadata(self.peer, self.layout, self.cpu_descriptor, self.npu_ticket, advertised_address)
        return self.metadata

    def ready(self, path: str, production_event: Any) -> Any:
        if self.metadata is None or self._cpu_closed or self._npu_closed:
            raise RuntimeError("registration is not available")
        source = self._source(path)
        if path in self._batches:
            raise RuntimeError("release the existing source batch before publishing another Ready")
        # Per-layer event recorded *after* scatter. Never publish Ready merely
        # because the host has enqueued the KV write.
        production_event.synchronize()
        ready = source.ready()
        self._batches[path] = ready
        return ready

    def release(self, path: str, ready: Any) -> None:
        source = self._source(path)
        self._check_batch(path, ready)
        # The SDK validates source identity + sequence, including deserialized
        # duplicate/late notifications, before any ownership is changed here.
        source.release_batch(ready)
        del self._batches[path]

    def cancel(self, path: str, ready: Any) -> None:
        """Only the SDK can establish that this batch has not been acquired."""
        source = self._source(path)
        self._check_batch(path, ready)
        source.cancel_batch(ready)
        # Cancellation is not release; keep the owner until release succeeds.

    def _check_batch(self, path: str, ready: Any) -> None:
        if path not in self._batches:
            raise RuntimeError("no active source batch")
        fields = ("source_id", "source_bytes") + (("region_id", "region_epoch") if path == "cpu" else ("source_epoch",))
        current = self._batches[path]
        if ready.sequence != current.sequence or any(
            getattr(ready.source, name) != getattr(current.source, name) for name in fields
        ):
            raise ValueError("completion does not match the active source/sequence")

    def _source(self, path: str) -> Any:
        if path not in ("cpu", "npu"):
            raise ValueError("source path must be cpu or npu")
        return self.cpu if path == "cpu" else self.npu

    def close(self) -> None:
        """Caller first drains batches and detaches D; failure retains buffers."""
        if self._batches:
            raise RuntimeError("source still has active batches")
        if not self._npu_closed:
            self.npu.shutdown()
            self._npu_closed = True
        if not self._cpu_closed:
            self.cpu.shutdown()
            self._cpu_closed = True


class ArenaRegistration:
    """One SDK source pair per arena/D peer, with explicit layer references.

    Sender schedules one arena future at a time. A failed/lost ACK retains the
    original Ready and this owner; no layer can start a replacement batch.
    """

    def __init__(self, sdk, peer, name, memory, **kwargs):
        memory.describe(name)
        self.registration = LayerRegistration(sdk, peer, name, memory.arena, memory.planes, **kwargs)
        self.members = set()
        self.active_layers = {}

    def register(self, *args, **kwargs):
        return self.registration.register(*args, **kwargs)

    def layer(self, name, memory):
        if name in self.members:
            raise ValueError("arena layer already referenced")
        reference = ArenaLayerRegistration(self, name, memory)
        self.members.add(name)
        return reference


class ArenaLayerRegistration:
    def __init__(self, owner, name, memory):
        self.owner, self.name = owner, name
        self.tensor = owner.registration.tensor
        self.metadata = replace(
            owner.registration.metadata,
            layout=memory.describe(name),
            registered_range=(self.tensor.data_ptr(), self.tensor.numel() * self.tensor.element_size()),
        )
        _ = self.metadata.source_bias
        self.closed = False

    def ready(self, path, production_event):
        if self.closed or path in self.owner.active_layers:
            raise RuntimeError("arena source has an active batch or a closed layer")
        ready = self.owner.registration.ready(path, production_event)
        self.owner.active_layers[path] = self.name
        return ready

    def _check_layer(self, path):
        if self.closed or self.owner.active_layers.get(path) != self.name:
            raise ValueError("arena Ready belongs to another layer")

    def release(self, path, ready):
        self._check_layer(path)
        self.owner.registration.release(path, ready)
        del self.owner.active_layers[path]

    def cancel(self, path, ready):
        self._check_layer(path)
        self.owner.registration.cancel(path, ready)

    def close(self):
        if self.closed:
            return
        if self.owner.active_layers:
            raise RuntimeError("arena retains active layer batches")
        if len(self.owner.members) == 1:
            self.owner.registration.close()
        self.owner.members.remove(self.name)
        self.closed = True
