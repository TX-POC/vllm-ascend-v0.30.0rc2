"""P10's W4A4C8 producer/executor launch and ACLGraph replay policy.

This module deliberately separates two meanings of "graph":

* :func:`launch_p10_forward` builds P10's *device dependency graph* on normal
  NPU streams.  That is the eager product route.
* :class:`P10ForwardGraph` captures the replay-safe preparation and native
  W4A4C8 score launches as raw ``aclmdlRI`` ACLGraph segments.  It relaunches
  the Direct-HCCP executor on its persistent companion stream for every
  replay, after the captured preparation event.

The split is necessary because P10 submits raw ACLNN calls directly on its
caller-selected stream.  ``torch.npu.NPUGraph`` is not the capture owner for
such calls and can produce a syntactically valid replay that omits those raw
nodes.  Raw ``aclmdlRI`` is the correct owner for the replay-safe ACLNN
compute nodes, but it is not a transport replay contract for Direct-HCCP:
a captured Stage-B invocation can leave terminal-looking control aliases
while omitting the external HCCP payload write on a later replay.  Stage B
therefore remains a fixed device executor launch, not a captured graph node.
There is one normal ACLNN submission per forward, but no host candidate loop,
no per-candidate host Get, and no host completion wait.

The graph-mode route is consequently:

``captured prepare -> {direct resident executor, captured score producer} -> QSFA``.

The braces are intentional.  After the common prepare event, the host enqueues
the resident executor on its companion stream *before* it enqueues native C8
on the producer stream.  This is a submission-order admission rule, not a
Stage-B-to-Stage-A dependency: Stage B polls the device-owned receipt protocol
while C8 continues producing, and C8 never waits for Stage B.  It matters for
short decode forwards, where submitting C8 first can let it seal before the
companion launch reaches an AIV.  No candidate is copied to Python, and no
host KVAF request is issued here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from . import _C


@dataclass(frozen=True)
class P10ForwardInputs:
    """Cold-bound aliases for exactly one W4A4C8 P10 forward/layer call.

    ``resident_row_to_qsfa_slot`` is a required translation table.  Resident
    payload rows are private forward rows; QSFA consumes logical slot IDs, so
    identity mapping is never inferred by the bridge.
    """

    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    query_dequant_scale: torch.Tensor
    key_dequant_scale: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    candidate_lens: torch.Tensor
    block_table: torch.Tensor
    canonical_sparse_indices: torch.Tensor
    canonical_score_source_lens: torch.Tensor
    score_surface: torch.Tensor
    receipt_ring: torch.Tensor
    receipt_state: torch.Tensor
    forward_control: torch.Tensor
    active_row_pool_entries: torch.Tensor
    kv_object_map: torch.Tensor
    kv_object_slots_snapshot: torch.Tensor
    kv_object_to_forward_slot: torch.Tensor
    source_to_forward_slot: torch.Tensor
    speculative_slot_band: torch.Tensor
    speculative_slot_cursor: torch.Tensor
    recall_slot_band: torch.Tensor
    kvaf_forward_resident_control: torch.Tensor
    queue_leases: torch.Tensor
    resident_payload: torch.Tensor
    resident_ready: torch.Tensor
    resident_row_to_qsfa_slot: torch.Tensor
    attention_slots_buffer: torch.Tensor
    selected_slot_lengths_buffer: torch.Tensor
    num_layers: int
    layer_id: int
    executor_worker_count: int
    executor_worker_limit: int
    receipt_poll_quantum: int
    bounded_drain_polls: int


@dataclass(frozen=True)
class P10QsfaAliases:
    """The only P10 values that cross into existing QSFA attention."""

    attention_slots: torch.Tensor
    resident_seq_lengths: torch.Tensor


@dataclass(frozen=True)
class P10OverlapWitness:
    """A post-forward diagnostic proving device-side overlap occurred.

    This is intentionally not consulted by the normal inference path.  Stage
    A writes ``producer_seal_sequence`` immediately before it seals receipt
    production; Stage B CAS-records ``first_preseal_ready_claim_sequence``
    only after it claims a READY receipt while the producer is still OPEN.
    """

    producer_seal_sequence: int
    first_preseal_ready_claim_sequence: int

    @property
    def overlapped(self) -> bool:
        return (
            self.producer_seal_sequence > 0
            and self.first_preseal_ready_claim_sequence > 0
            and self.first_preseal_ready_claim_sequence
            < self.producer_seal_sequence
        )


@dataclass(frozen=True)
class P10ForwardLaunch:
    """A completed graph construction, not a host-completed forward.

    Calling :meth:`bind_qsfa` inserts one device event dependency.  It never
    synchronizes the host and therefore preserves the subsequent forward graph
    boundary.  Stage B itself reaches ``COMPLETE`` only after its final recall
    and resident-slot drain, so that event is the sole QSFA dependency.
    """

    attention_slots: torch.Tensor
    resident_seq_lengths: torch.Tensor
    stage_b_complete: Any
    producer_stream: Any
    executor_stream: Any
    forward_control: torch.Tensor

    def bind_qsfa(self, qsfa_stream: Any | None = None) -> P10QsfaAliases:
        """Insert the required Stage-B-COMPLETE -> QSFA device edge."""

        if qsfa_stream is None:
            qsfa_stream = torch.npu.current_stream(
                device=self.attention_slots.device
            )
        qsfa_stream.wait_event(self.stage_b_complete)
        return P10QsfaAliases(
            attention_slots=self.attention_slots,
            resident_seq_lengths=self.resident_seq_lengths,
        )

    def read_overlap_witness_after_completion(self) -> P10OverlapWitness:
        """Synchronize only an explicit validation probe after Stage-B ends.

        Product forward code must use :meth:`bind_qsfa` and never call this
        helper.  It exists for the P10 NPU acceptance harness, where a host
        read after the complete event proves this was not merely a serial
        Stage-A-then-Stage-B launch.
        """

        self.stage_b_complete.synchronize()
        probe = self.forward_control.narrow(0, 8, 2).to(device="cpu")
        return P10OverlapWitness(
            producer_seal_sequence=int(probe[0].item()),
            first_preseal_ready_claim_sequence=int(probe[1].item()),
        )


# Every tensor below participates in one of the three ACLNN graph nodes.  A
# replay must retain its original storage address, shape and stride.  Captured
# graph inputs are deliberately cold-bound: P10's graph ABI contains no host
# copies or dynamic candidate submissions that could repair an alias mismatch.
_GRAPH_TENSOR_FIELDS = (
    "query",
    "key",
    "weights",
    "query_dequant_scale",
    "key_dequant_scale",
    "actual_seq_lengths_query",
    "candidate_lens",
    "block_table",
    "canonical_sparse_indices",
    "canonical_score_source_lens",
    "score_surface",
    "receipt_ring",
    "receipt_state",
    "forward_control",
    "active_row_pool_entries",
    "kv_object_map",
    "kv_object_slots_snapshot",
    "kv_object_to_forward_slot",
    "source_to_forward_slot",
    "speculative_slot_band",
    "speculative_slot_cursor",
    "recall_slot_band",
    "kvaf_forward_resident_control",
    "queue_leases",
    "resident_payload",
    "resident_ready",
    "resident_row_to_qsfa_slot",
    "attention_slots_buffer",
    "selected_slot_lengths_buffer",
)

_GRAPH_SCALAR_FIELDS = (
    "num_layers",
    "layer_id",
    "executor_worker_count",
    "executor_worker_limit",
    "receipt_poll_quantum",
    "bounded_drain_polls",
)


def _require_npu_inputs(inputs: P10ForwardInputs) -> None:
    if not inputs.query.device.type.startswith("privateuseone") and \
            inputs.query.device.type != "npu":
        raise RuntimeError("P10 forward graph requires an NPU query tensor")


def _tensor_graph_identity(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Return the ABI-relevant static identity of one captured tensor.

    ``data_ptr`` alone is insufficient for an as-strided view.  The captured
    ACLNN descriptor also depends on the logical shape and stride, so replay
    rejects either kind of drift before it can execute a stale graph.
    """

    return (
        tensor.data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


@dataclass(frozen=True)
class _P10GraphInputSignature:
    tensors: tuple[tuple[str, tuple[Any, ...]], ...]
    scalars: tuple[tuple[str, int], ...]

    @classmethod
    def from_inputs(cls, inputs: P10ForwardInputs) -> "_P10GraphInputSignature":
        return cls(
            tensors=tuple(
                (name, _tensor_graph_identity(getattr(inputs, name)))
                for name in _GRAPH_TENSOR_FIELDS
            ),
            scalars=tuple(
                (name, int(getattr(inputs, name))) for name in _GRAPH_SCALAR_FIELDS
            ),
        )

    def mismatch_reason(self, inputs: P10ForwardInputs) -> str | None:
        replay = _P10GraphInputSignature.from_inputs(inputs)
        for (name, expected), (_, actual) in zip(self.tensors, replay.tensors):
            if expected != actual:
                return f"tensor_static_alias_mismatch:{name}"
        for (name, expected), (_, actual) in zip(self.scalars, replay.scalars):
            if expected != actual:
                return f"scalar_tiling_mismatch:{name}"
        return None


def _require_raw_acl_graph_type() -> Any:
    """Return KVAF's raw ACL graph owner or fail before capture.

    P10's bridge deliberately calls ACLNN directly on the current CANN stream.
    Capturing those calls through ``torch.npu.NPUGraph`` is not equivalent to
    capturing the raw stream: the wrapper can retain no graph nodes even when
    it returns a graph object successfully. KVAF's ``AclGraph`` is the
    already-validated ``aclmdlRI`` owner for KVAF's Direct-HCCP data plane and
    is built from the same exact product KVAF source as P10's device ABI.
    """

    try:
        import kvaf
    except Exception as error:  # noqa: BLE001 -- retain the useful import cause.
        raise RuntimeError(
            "P10_GRAPH_CAPTURE_BLOCKED:kvaf_aclgraph_import_failed:"
            f"{type(error).__name__}:{error}"
        ) from error
    graph_type = getattr(kvaf, "AclGraph", None)
    if graph_type is None:
        raise RuntimeError(
            "P10_GRAPH_CAPTURE_BLOCKED:kvaf_aclgraph_missing; rebuild the "
            "exact P10 KVAF source with the AIV graph binding enabled"
        )
    return graph_type


def _capture_raw_acl_segment(
    *,
    graph_type: Any,
    stream: Any,
    submit: Any,
    label: str,
) -> tuple[Any, int, int]:
    """Capture one direct ACLNN segment with a graph-owned workspace scope.

    ``aclnn*`` records the workspace address in raw ``aclmdlRI`` capture. The
    C++ bridge retains each workspace under the returned token until this graph
    segment is reset. Without that retention, a normal local ``at::Tensor``
    destructor may recycle the address before replay and make an apparently
    terminal Direct-HCCP forward leave its resident payload untouched.
    """

    raw_graph = graph_type()
    workspace_token = int(_C.begin_graph_workspace_capture())
    capture_active = False
    workspace_sealed = False
    try:
        with torch.npu.stream(stream):
            raw_graph.capture_begin(stream)
            capture_active = True
            submit()
            raw_graph.capture_end()
            capture_active = False
        workspace_count = int(_C.end_graph_workspace_capture(workspace_token))
        workspace_sealed = True
        return raw_graph, workspace_token, workspace_count
    except Exception as error:  # noqa: BLE001 -- preserve exact CANN failure.
        if capture_active:
            try:
                raw_graph.capture_end()
            except Exception:  # noqa: BLE001 -- original error is authoritative.
                pass
        try:
            raw_graph.reset()
        except Exception:  # noqa: BLE001 -- original error is authoritative.
            pass
        if not workspace_sealed:
            _C.release_graph_workspace_capture(workspace_token)
        raise RuntimeError(
            "P10_GRAPH_CAPTURE_FAILED: "
            f"segment={label} type={type(error).__name__} text={error}"
        ) from error


class P10ForwardGraph:
    """Capture P10's replay-safe nodes and relaunch its transport executor.

    P10's C++ bridge bypasses torch_npu's task queue and submits ACLNN directly
    on the current CANN stream. It consequently uses KVAF's raw ``aclmdlRI``
    graph owner for two independently captured segments:

    * ``prepare_graph`` on the producer stream;
    * ``stage_a_graph`` on that same producer stream.

    During replay, the prepare graph records one device event.  Stage B is
    submitted directly first after that event, then the native W4A4C8 graph is
    replayed on the producer stream.  This preserves the eager route's
    admission order and its absence of a Stage-A-complete -> Stage-B edge.
    Capturing Stage B is deliberately forbidden: its Direct-HCCP doorbell is
    an external transport side effect rather than a replay-safe ACL graph
    node.  Capture is intentionally one-time and may synchronize.  ``replay``
    never synchronizes the host, allocates a new stream, or launches a Python
    candidate/Get loop.
    """

    def __init__(
        self,
        *,
        signature: _P10GraphInputSignature,
        execution_mode: str,
        prepare_graph: Any,
        stage_a_graph: Any,
        producer_stream: Any,
        executor_stream: Any,
        forward_control: torch.Tensor,
        attention_slots: torch.Tensor,
        resident_seq_lengths: torch.Tensor,
        workspace_tokens: tuple[int, int],
        workspace_counts: tuple[int, int],
    ) -> None:
        self._signature = signature
        self._execution_mode = execution_mode
        self._prepare_graph = prepare_graph
        self._stage_a_graph = stage_a_graph
        self._producer_stream = producer_stream
        self._executor_stream = executor_stream
        self._forward_control = forward_control
        self._attention_slots = attention_slots
        self._resident_seq_lengths = resident_seq_lengths
        self._workspace_tokens = workspace_tokens
        self._workspace_counts = workspace_counts
        self._closed = False

    @property
    def execution_mode(self) -> str:
        return self._execution_mode

    @property
    def workspace_counts(self) -> tuple[int, int]:
        """Captured workspace buffers for the prepare and score segments."""

        return self._workspace_counts

    @property
    def captured_segment_count(self) -> int:
        """The exact number of raw ACLGraph segments retained by this object."""

        return len(self._workspace_tokens)

    @classmethod
    def capture(
        cls,
        inputs: P10ForwardInputs,
        *,
        producer_stream: Any | None = None,
        executor_stream: Any | None = None,
        graph_pool: Any | None = None,
        execution_mode: str = "overlap",
    ) -> "P10ForwardGraph":
        """Perform a one-time capture of the exact P10 product operators.

        The warm-up sequence is deliberately serial only while *constructing*
        the two raw graphs.  It initializes the full product closure,
        including the non-captured Direct-HCCP executor, outside capture.
        This has no bearing on replay, where the overlap topology below is
        restored.  The initial serial warm-up also forces lazy ACLNN/OPP
        setup outside capture, so a capture failure cannot silently become an
        eager fallback.
        """

        _require_npu_inputs(inputs)
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError(
                "P10_GRAPH_CAPTURE_BLOCKED:nested_aclgraph_capture_forbidden"
            )
        if execution_mode not in ("overlap", "serial"):
            raise ValueError(
                "P10 graph execution_mode must be 'overlap' or benchmark-only "
                f"'serial', got {execution_mode!r}"
            )

        # NPU graphs cannot be captured on the default stream.  The dedicated
        # streams remain owned by this graph object and are reused on every
        # replay; they are logical streams, never a physical-AIV partition.
        if producer_stream is None:
            producer_stream = _make_companion_stream(inputs.query)
        if executor_stream is None:
            executor_stream = _make_companion_stream(inputs.query)
        if execution_mode == "overlap" and producer_stream == executor_stream:
            raise RuntimeError(
                "P10_GRAPH_CAPTURE_BLOCKED:overlap_requires_distinct_logical_streams"
            )
        if graph_pool is not None:
            raise RuntimeError(
                "P10_GRAPH_CAPTURE_BLOCKED:torch_graph_pool_is_not_valid_for_"
                "raw_direct_aclnn_capture"
            )
        raw_graph_type = _require_raw_acl_graph_type()

        # Warm the exact three-product-operator closure before capture.  The
        # completion event is synchronized only here, never on graph replay.
        warmup = launch_p10_forward(
            inputs,
            producer_stream=producer_stream,
            executor_stream=executor_stream,
            execution_mode="serial",
        )
        warmup.stage_b_complete.synchronize()

        prepare_graph = None
        stage_a_graph = None
        workspace_tokens: list[int] = []
        workspace_counts: list[int] = []
        try:
            prepare_graph, token, count = _capture_raw_acl_segment(
                graph_type=raw_graph_type,
                stream=producer_stream,
                submit=lambda: _launch_prepare(inputs),
                label="prepare",
            )
            workspace_tokens.append(token)
            workspace_counts.append(count)

            stage_a_graph, token, count = _capture_raw_acl_segment(
                graph_type=raw_graph_type,
                stream=producer_stream,
                submit=lambda: _launch_stage_a(inputs),
                label="stage_a",
            )
            workspace_tokens.append(token)
            workspace_counts.append(count)

        except Exception:
            for graph in (stage_a_graph, prepare_graph):
                if graph is not None:
                    try:
                        graph.reset()
                    except Exception:  # noqa: BLE001 -- capture error wins.
                        pass
            for token in workspace_tokens:
                _C.release_graph_workspace_capture(token)
            raise

        if len(workspace_tokens) != 2 or len(workspace_counts) != 2:
            raise RuntimeError("P10_GRAPH_CAPTURE_FAILED:incomplete_raw_segments")

        return cls(
            signature=_P10GraphInputSignature.from_inputs(inputs),
            execution_mode=execution_mode,
            prepare_graph=prepare_graph,
            stage_a_graph=stage_a_graph,
            producer_stream=producer_stream,
            executor_stream=executor_stream,
            forward_control=inputs.forward_control,
            attention_slots=inputs.attention_slots_buffer,
            resident_seq_lengths=inputs.selected_slot_lengths_buffer,
            workspace_tokens=tuple(workspace_tokens),
            workspace_counts=tuple(workspace_counts),
        )

    def replay(self, inputs: P10ForwardInputs) -> P10ForwardLaunch:
        """Replay the captured product graph without a host completion wait."""

        if self._closed:
            raise RuntimeError("P10_GRAPH_REPLAY_BLOCKED:graph_is_closed")
        mismatch = self._signature.mismatch_reason(inputs)
        if mismatch is not None:
            raise RuntimeError(f"P10_GRAPH_REPLAY_BLOCKED:{mismatch}")

        with torch.npu.stream(self._producer_stream):
            self._prepare_graph.replay(self._producer_stream)
            prepared = _record_event(self._producer_stream)

        if self._execution_mode == "overlap":
            # Preserve device-executor admission before the producer, without
            # introducing the forbidden Stage-A completion dependency.  The
            # Direct-HCCP doorbell must be issued for every replay; raw
            # aclmdlRI capture does not provide a transport replay guarantee.
            with torch.npu.stream(self._executor_stream):
                self._executor_stream.wait_event(prepared)
                _launch_stage_b(inputs)
                stage_b_complete = _record_event(self._executor_stream)
            with torch.npu.stream(self._producer_stream):
                self._producer_stream.wait_event(prepared)
                self._stage_a_graph.replay(self._producer_stream)
        else:
            # The graph-mode baseline differs by exactly the same one device
            # edge as eager serial: Stage A completion gates Stage B.
            with torch.npu.stream(self._producer_stream):
                self._producer_stream.wait_event(prepared)
                self._stage_a_graph.replay(self._producer_stream)
                stage_a_complete = _record_event(self._producer_stream)
            with torch.npu.stream(self._executor_stream):
                self._executor_stream.wait_event(stage_a_complete)
                _launch_stage_b(inputs)
                stage_b_complete = _record_event(self._executor_stream)

        return P10ForwardLaunch(
            attention_slots=self._attention_slots,
            resident_seq_lengths=self._resident_seq_lengths,
            stage_b_complete=stage_b_complete,
            producer_stream=self._producer_stream,
            executor_stream=self._executor_stream,
            forward_control=self._forward_control,
        )

    def close(self) -> None:
        """Destroy raw graphs before releasing their captured workspaces.

        Call this only after the last ``stage_b_complete`` event has reached
        terminal completion. ``_run_graph_benchmark`` does so in its finalizer;
        normal inference keeps the object alive for its graph-cache lifetime.
        """

        if self._closed:
            return
        errors: list[str] = []
        for label, graph in (
            ("stage_a", self._stage_a_graph),
            ("prepare", self._prepare_graph),
        ):
            try:
                graph.reset()
            except Exception as error:  # noqa: BLE001 -- release all scopes.
                errors.append(f"{label}:{type(error).__name__}:{error}")
        for token in self._workspace_tokens:
            try:
                _C.release_graph_workspace_capture(token)
            except Exception as error:  # noqa: BLE001 -- report after cleanup.
                errors.append(f"workspace:{type(error).__name__}:{error}")
        self._closed = True
        if errors:
            raise RuntimeError("P10_GRAPH_CLOSE_FAILED:" + ";".join(errors))

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _record_event(stream: Any) -> Any:
    event = torch.npu.Event(enable_timing=False)
    event.record(stream)
    return event


def _make_companion_stream(anchor: torch.Tensor) -> Any:
    # This is a logical companion stream.  It is deliberately not a request
    # for a physical AIV core or a fixed C8/Get core partition.
    return torch.npu.Stream(device=anchor.device)


def _launch_prepare(inputs: P10ForwardInputs) -> None:
    _C.prepare_forward(
        inputs.receipt_state,
        inputs.forward_control,
        inputs.kv_object_to_forward_slot,
        inputs.source_to_forward_slot,
        inputs.speculative_slot_cursor,
        inputs.queue_leases,
        inputs.resident_ready,
    )


def _launch_stage_a(inputs: P10ForwardInputs) -> None:
    _C.launch_stage_a(
        inputs.query,
        inputs.key,
        inputs.weights,
        inputs.query_dequant_scale,
        inputs.key_dequant_scale,
        inputs.actual_seq_lengths_query,
        inputs.candidate_lens,
        inputs.block_table,
        inputs.canonical_sparse_indices,
        inputs.canonical_score_source_lens,
        inputs.score_surface,
        inputs.receipt_ring,
        inputs.receipt_state,
        inputs.forward_control,
    )


def _launch_stage_b(inputs: P10ForwardInputs) -> None:
    # These are cold-bound aliases.  In particular, this call does not consume
    # a return value or completion event from _launch_stage_a.
    _C.launch_stage_b(
        inputs.score_surface,
        inputs.receipt_ring,
        inputs.receipt_state,
        inputs.forward_control,
        inputs.active_row_pool_entries,
        inputs.kv_object_map,
        inputs.kv_object_slots_snapshot,
        inputs.kv_object_to_forward_slot,
        inputs.source_to_forward_slot,
        inputs.speculative_slot_band,
        inputs.speculative_slot_cursor,
        inputs.recall_slot_band,
        inputs.canonical_sparse_indices,
        inputs.canonical_score_source_lens,
        inputs.kvaf_forward_resident_control,
        inputs.queue_leases,
        inputs.resident_payload,
        inputs.resident_ready,
        inputs.resident_row_to_qsfa_slot,
        inputs.attention_slots_buffer,
        inputs.selected_slot_lengths_buffer,
        inputs.num_layers,
        inputs.layer_id,
        inputs.executor_worker_count,
        inputs.executor_worker_limit,
        inputs.receipt_poll_quantum,
        inputs.bounded_drain_polls,
    )


def launch_p10_forward(
    inputs: P10ForwardInputs,
    *,
    producer_stream: Any | None = None,
    executor_stream: Any | None = None,
    execution_mode: str = "overlap",
) -> P10ForwardLaunch:
    """Build the P10 product graph without a host candidate submission path.

    The caller may supply streams already managed by the inference graph.  If
    omitted, the current NPU stream is the native W4A4C8 producer and a fresh
    logical companion stream is allocated for the resident executor.  Both
    wait only for the graph-visible preparation event.  Stage B's device-side
    receipt protocol is the only producer/executor synchronization mechanism.

    ``execution_mode="overlap"`` is the only product path.  The explicit
    ``"serial"`` mode exists solely for the P10 acceptance benchmark: it uses
    the same three product operators, inputs, Direct-HCCP control, and QSFA
    boundary, but adds exactly one Stage-A-complete event before Stage B.  It
    is therefore a valid same-workload reference, rather than a host-Get or
    separately computed C8 baseline.  Inference callers must leave the
    default unchanged.
    """

    _require_npu_inputs(inputs)

    if producer_stream is None:
        producer_stream = torch.npu.current_stream(device=inputs.query.device)
    if executor_stream is None:
        executor_stream = _make_companion_stream(inputs.query)
    if execution_mode not in ("overlap", "serial"):
        raise ValueError(
            "P10 execution_mode must be 'overlap' or benchmark-only 'serial', "
            f"got {execution_mode!r}"
        )

    # Prepare executes as a real custom operator in the graph.  The event is
    # the only precondition shared by the two concurrent successors.
    with torch.npu.stream(producer_stream):
        _launch_prepare(inputs)
        prepared = _record_event(producer_stream)

    if execution_mode == "overlap":
        # Enqueue the persistent consumer first.  This does not make Stage A
        # wait for Stage B and it does not reserve a physical AIV: it only
        # ensures that the device-side receipt poller has been admitted before
        # the first native score tile can become READY.  In particular, do not
        # reverse these two independent submissions for a short forward, or
        # C8 may seal before the companion is ever scheduled and eliminate all
        # useful overlap.
        with torch.npu.stream(executor_stream):
            executor_stream.wait_event(prepared)
            _launch_stage_b(inputs)
            stage_b_complete = _record_event(executor_stream)

        with torch.npu.stream(producer_stream):
            producer_stream.wait_event(prepared)
            _launch_stage_a(inputs)
    else:
        # Benchmark-only serial reference.  No host synchronization occurs:
        # the sole added edge is native C8 completion -> the unchanged Stage-B
        # Direct-HCCP executor.  This preserves every input/output alias and
        # makes a serial timing comparison meaningful.
        with torch.npu.stream(producer_stream):
            producer_stream.wait_event(prepared)
            _launch_stage_a(inputs)
            serial_stage_a_complete = _record_event(producer_stream)

        with torch.npu.stream(executor_stream):
            executor_stream.wait_event(serial_stage_a_complete)
            _launch_stage_b(inputs)
            stage_b_complete = _record_event(executor_stream)

    return P10ForwardLaunch(
        attention_slots=inputs.attention_slots_buffer,
        resident_seq_lengths=inputs.selected_slot_lengths_buffer,
        stage_b_complete=stage_b_complete,
        producer_stream=producer_stream,
        executor_stream=executor_stream,
        forward_control=inputs.forward_control,
    )


def capture_p10_forward_graph(
    inputs: P10ForwardInputs,
    *,
    producer_stream: Any | None = None,
    executor_stream: Any | None = None,
    graph_pool: Any | None = None,
    execution_mode: str = "overlap",
) -> P10ForwardGraph:
    """Capture the P10 product route for later ACLGraph replay.

    This explicit factory keeps graph capture out of the eager product path.
    Callers that are already inside vLLM's outer ACLGraph must construct this
    object at graph-setup time, then call :meth:`P10ForwardGraph.replay` at
    replay time; nested capture is rejected rather than falling back to eager.
    """

    return P10ForwardGraph.capture(
        inputs,
        producer_stream=producer_stream,
        executor_stream=executor_stream,
        graph_pool=graph_pool,
        execution_mode=execution_mode,
    )
