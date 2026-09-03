"""P10's graph-level W4A4C8 producer/executor launch policy.

This is deliberately a graph adapter, not a candidate scheduler.  It enqueues
three device operators only:

``prepare -> {native score producer, resident executor} -> QSFA``.

The braces are intentional.  Stage B is launched on a companion device stream
after the common prepare event; it does *not* wait for native C8 to finish.
It polls the device-owned receipt protocol while C8 continues producing.  No
candidate is copied to Python, and no host KVAF request is issued here.
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
) -> P10ForwardLaunch:
    """Build the P10 product graph without a host candidate submission path.

    The caller may supply streams already managed by the inference graph.  If
    omitted, the current NPU stream is the native W4A4C8 producer and a fresh
    logical companion stream is allocated for the resident executor.  Both
    wait only for the graph-visible preparation event.  Stage B's device-side
    receipt protocol is the only producer/executor synchronization mechanism.
    """

    if not inputs.query.device.type.startswith("privateuseone") and \
            inputs.query.device.type != "npu":
        raise RuntimeError("P10 forward graph requires an NPU query tensor")

    if producer_stream is None:
        producer_stream = torch.npu.current_stream(device=inputs.query.device)
    if executor_stream is None:
        executor_stream = _make_companion_stream(inputs.query)

    # Prepare executes as a real custom operator in the graph.  The event is
    # the only precondition shared by the two concurrent successors.
    with torch.npu.stream(producer_stream):
        _launch_prepare(inputs)
        prepared = _record_event(producer_stream)

    with torch.npu.stream(producer_stream):
        producer_stream.wait_event(prepared)
        _launch_stage_a(inputs)

    with torch.npu.stream(executor_stream):
        executor_stream.wait_event(prepared)
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
