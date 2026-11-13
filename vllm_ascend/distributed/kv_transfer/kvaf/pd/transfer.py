# SPDX-License-Identifier: Apache-2.0
"""Explicit SDK transactions. No task replay, key pool, or Hot addressing.

Callers serialize access to a worker's Runtime, including ordinary Decode Get.
They retain these objects until completion/recovery and own request cancellation
and key lifetime. SDK Put/Ready handles remain the authority for DMA state.
"""

from typing import Any

from .layout import Segments
from .observation import notify, observed_call


def _exact_cover(segments: Segments, entries) -> None:
    spans = iter(sorted(zip(segments.destinations, segments.lengths)))
    current = next(spans, None)
    previous_end = 0
    for entry in sorted(entries, key=lambda e: e.address):
        cursor, end = entry.address, entry.address + entry.length
        if cursor <= 0 or cursor < previous_end or end <= cursor:
            raise ValueError("invalid Put reservation")
        while current is not None and cursor < end:
            address, size = current
            if address != cursor or address + size > end:
                raise ValueError("CPU Pull must exactly cover each Put entry without crossing slots")
            cursor += size
            current = next(spans, None)
        if cursor != end:
            raise ValueError("CPU Pull leaves unwritten Put bytes")
        previous_end = end
    if current is not None:
        raise ValueError("CPU Pull targets bytes outside this Put")


class CpuPullTransfer:
    """One existing Put and Ready. Prepare only when source data is available.

    The owner allocates keys through the existing writer.slots and calls
    prepare_put; this class never creates another Put during recovery. Failed
    calls retain the handle and all metadata, including an uncertain Commit.
    """

    def __init__(self, runtime: Any, ready: Any, put: Any, segments: Segments):
        segments.validate_call(ready.source.mr_length)
        _exact_cover(segments, put.entries)
        self.runtime, self.ready, self.put, self.segments = runtime, ready, put, segments
        self.request_observer = self.observer_context = None
        base = runtime.remote_region["base"]
        self.dst_offsets = tuple(address - base for address in segments.destinations)
        if base <= 0 or any(offset < 0 for offset in self.dst_offsets):
            raise ValueError("CPU Pull destination is below the DDR region base")
        self.attempted = False
        self.decision: str | None = None
        self.committed = False
        self.aborted = False
        self.published = False
        self._publication = None

    def submit(self) -> None:
        if self.attempted or self.decision is not None:
            raise RuntimeError("recover the original Put; do not resubmit READ")
        self.attempted = True
        if self.request_observer is None:
            self.runtime.submit_cpu_pull(
                self.ready, self.put, self.segments.sources, self.dst_offsets, self.segments.lengths
            )
        else:
            observed_call(
                self.request_observer,
                "D.cpu_submit",
                self.runtime.submit_cpu_pull,
                self.ready,
                self.put,
                self.segments.sources,
                self.dst_offsets,
                self.segments.lengths,
                context=self.observer_context,
            )

    def finish(self, timeout_ms: int = 30000) -> None:
        if self.decision == "abort":
            raise RuntimeError("retry the original Abort decision")
        if self.committed:
            return
        if self.decision is None:
            if not self.attempted or not self.put.submitted:
                raise RuntimeError("CPU Pull was not submitted")
            # Wait can raise on timeout or IN_DOUBT; no Abort/Prepare in except.
            if self.request_observer is None:
                state = self.runtime.wait_cpu_pull(self.put, timeout_ms)
            else:
                state = observed_call(
                    self.request_observer,
                    "D.cpu_wait",
                    self.runtime.wait_cpu_pull,
                    self.put,
                    timeout_ms,
                    context=self.observer_context,
                )
            if state.name not in ("DONE", "COMMITTED"):
                raise RuntimeError("CPU Pull has not completed")
            self.decision = "commit"
            if self.request_observer is not None:
                notify(self.request_observer, "point", name="CPU_PULL_DONE", context=self.observer_context)
        if self.request_observer is None:
            result = self.runtime.commit_put(self.put)
        else:
            result = observed_call(
                self.request_observer, "D.cpu_commit", self.runtime.commit_put, self.put, context=self.observer_context
            )
        if result.name != "COMMITTED":
            raise RuntimeError("CPU Pull Commit has not completed")
        self.committed = True
        if self.request_observer is not None:
            notify(self.request_observer, "point", name="DDR_COMMITTED", context=self.observer_context)

    def abort(self) -> None:
        if self.decision == "commit":
            raise RuntimeError("retry the original Commit; cannot reverse it")
        if self.aborted:
            return
        if self.decision is None and self.put.submitted:
            state = self.runtime.query_cpu_pull(self.put)
            if state.name not in ("DONE", "FAILED", "ABORTED"):
                raise RuntimeError("DMA may still be active; retain Put, source and destinations")
        self.decision = "abort"
        result = self.runtime.abort_put(self.put)
        if result.name != "ABORTED":
            raise RuntimeError("CPU Pull Abort has not completed")
        self.aborted = True

    def publish(self, writer: Any, group: Any, state: Any) -> None:
        """Reuse KVTransfer's checked catalog publication after Commit.

        state is the existing writer.requests[request_id][group_index]. The
        caller must hold its request/runtime serialization until this returns.
        Keys must already be owned by writer.slots; no second allocator exists.
        Native policy.prepare uploads these addresses to its device directory.
        """
        if not self.committed:
            raise RuntimeError("Commit must complete before publishing DDR addresses")
        if writer.runtimes[group.geometry] is not self.runtime:
            raise ValueError("writer group belongs to another Runtime/region")
        keys = tuple(entry.key for entry in self.put.entries)
        if any(entry.length != group.geometry[0] * group.geometry[1] for entry in self.put.entries):
            raise ValueError("Put payload does not match the existing DDR group geometry")
        if self._publication is None:
            if len(state.keys) != len(state.addresses) or set(keys).intersection(state.keys):
                raise ValueError("request directory is pending or already contains these keys")
            self._publication = (writer, group, state, len(state.keys))
            # Keep committed keys reclaimable even if catalog validation fails.
            state.keys.extend(keys)
        old_writer, old_group, old_state, start = self._publication
        if old_writer is not writer or old_group is not group or old_state is not state:
            raise ValueError("a Put cannot be published into another request/group")
        if tuple(state.keys[start:]) != keys:
            raise ValueError("request keys changed during publication")
        if not self.published:
            if len(state.addresses) != start:
                raise ValueError("request addresses changed during publication")
            if self.request_observer is None:
                writer._publish_addresses(group, list(keys), [(state, len(keys))])
            else:
                observed_call(
                    self.request_observer,
                    "D.cpu_publish",
                    writer._publish_addresses,
                    group,
                    list(keys),
                    [(state, len(keys))],
                    context=self.observer_context,
                )
            self.published = True
            if self.request_observer is not None:
                notify(self.request_observer, "point", name="DDR_PUBLISHED", context=self.observer_context)


class NpuPullTransfer:
    """One raw READ with stable output/metadata references and explicit recovery."""

    def __init__(self, runtime: Any, ready: Any, segments: Segments, output: Any, stream: Any, *, source_bytes: int):
        if not output.is_contiguous():
            raise ValueError("raw READ output must be contiguous")
        segments.validate_call(source_bytes, output.numel() * output.element_size())
        self.runtime, self.ready, self.segments = runtime, ready, segments
        self.request_observer = self.observer_context = None
        self.output, self.stream = output, stream
        self.arrays = None
        self.attempted = False
        self.done = False

    def _make_arrays(self):
        # Worker-only dependency; the layout, control and CPU paths import no
        # Ascend API. All three arrays become ready on the READ stream.
        import torch

        if self.output.device.type != "npu":
            raise ValueError("raw READ requires an NPU output")
        with torch.npu.stream(self.stream):
            return tuple(
                torch.tensor(values, dtype=torch.int64, device=self.output.device)
                for values in (self.segments.sources, self.segments.destinations, self.segments.lengths)
            )

    def run(self) -> None:
        if self.attempted:
            raise RuntimeError("recover this Ready; never replay raw READ")
        self.arrays = self._make_arrays()
        self.attempted = True
        if self.request_observer is None:
            result = self.runtime.batch_get_raw_addr(self.ready, *self.arrays, self.output, self.stream)
        else:
            result = observed_call(
                self.request_observer,
                "D.npu_read",
                self.runtime.batch_get_raw_addr,
                self.ready,
                *self.arrays,
                self.output,
                self.stream,
                context=self.observer_context,
            )
        if result != {"state": "done", "sequence": self.ready.sequence}:
            raise RuntimeError("unexpected raw READ completion")
        self.done = True
        if self.request_observer is not None:
            notify(self.request_observer, "point", name="HBM_PULL_DONE", context=self.observer_context)

    def recover(self) -> None:
        if self.done:
            return
        if not self.attempted or not self.runtime.pending_raw_addr:
            raise RuntimeError("no recoverable raw READ; do not report output ready")
        # This SDK returns None on successful resolve, and raises if recovery
        # merely cancels/cleans up a failed operation. None is not a Tensor.
        if self.request_observer is None:
            self.runtime.resolve_raw_addr(self.ready)
        else:
            observed_call(
                self.request_observer,
                "D.npu_recovery",
                self.runtime.resolve_raw_addr,
                self.ready,
                context=self.observer_context,
            )
        if self.runtime.pending_raw_addr:
            raise RuntimeError("raw READ is still pending")
        self.done = True
        if self.request_observer is not None:
            notify(self.request_observer, "point", name="HBM_PULL_DONE", context=self.observer_context)
