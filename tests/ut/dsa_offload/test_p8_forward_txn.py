# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure-Python contract tests for the P8 forward-owned async transaction."""

from __future__ import annotations

from dataclasses import replace

import pytest

from vllm_ascend.dsa_offload.p8_forward_txn import (
    CandidateState,
    ForwardPhase,
    ForwardState,
    P8FixedResources,
    P8LifecycleError,
    P8QuarantineError,
    PrivateSlotState,
    QueueState,
)


def _post_one_successful_wave():
    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=2,
        private_slot_capacity=2,
    )
    txn = resources.begin_forward(
        epoch=11,
        graph_generation=3,
        expected_layers=2,
    )
    txn.retain_host_lease("kvaf-wave-0")
    txn.begin_admission()
    txn.publish_candidate(0, producer=7, layer_control_index=0)
    txn.reserve_private_slot(0, owner=7)
    lease = txn.lease_queue(0, owner=7, layer_control_index=0)
    txn.prepare_post(lease, candidate_slots=[0])
    wave = txn.post_only(lease)
    assert txn.try_claim_drain(wave)
    txn.observe_cq(wave, succeeded=True)
    txn.finalize_wave(wave)
    txn.publish_private(wave)
    txn.release_queue(wave)
    return resources, txn, wave


def test_p8_normal_forward_releases_only_after_qsfa_terminal() -> None:
    resources, txn, _ = _post_one_successful_wave()

    assert resources.queues[0].state is QueueState.FREE
    assert resources.candidates[0].state is CandidateState.CONSUMED
    assert resources.private_slots[0].state is PrivateSlotState.READY
    assert resources.private_slots[0].layer_control_index == 0
    assert resources.private_slots[0].queue_generation == 1
    assert resources.private_slots[0].post_sequence == 1
    assert resources.private_source_to_slot == {0: 0}
    assert resources.private_object_to_slot == {0: 0}

    txn.seal_admission()
    txn.begin_draining()
    txn.begin_reconcile()
    txn.mark_exact_winners_confirmed(0)
    txn.mark_exact_winners_confirmed(1)
    txn.mark_guaranteed_recall_terminal(0)
    txn.mark_guaranteed_recall_terminal(1)
    txn.begin_qsfa_reading()

    with pytest.raises(P8LifecycleError, match="FORWARD_TERMINAL"):
        txn.release()

    txn.mark_qsfa_terminal(0)
    # QSFA is the last reader for layer 0, so its global private HBM slot is
    # reusable before the unrelated model-terminal event.  Host/native
    # transaction ownership still remains until model terminal is observed.
    assert resources.private_slots[0].state is PrivateSlotState.FREE
    assert resources.private_source_to_slot == {}
    assert resources.private_object_to_slot == {}
    with pytest.raises(P8LifecycleError, match="every layer"):
        txn.mark_model_terminal_enqueued()

    txn.mark_qsfa_terminal(1)
    txn.mark_model_terminal_enqueued()
    with pytest.raises(P8LifecycleError, match="observed"):
        txn.enter_forward_terminal()
    txn.mark_model_terminal_observed()
    txn.enter_forward_terminal()
    assert txn.phase is ForwardPhase.FORWARD_TERMINAL
    txn.release()

    assert txn.state is ForwardState.RELEASED
    assert txn.host_leases == set()
    assert resources.active_txn is None
    assert resources.private_slots[0].state is PrivateSlotState.FREE


def test_p8_cq_failure_defers_candidate_and_requires_exact_recall() -> None:
    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=12, graph_generation=0)
    txn.begin_admission()
    txn.publish_candidate(0, producer=1, layer_control_index=0)
    txn.reserve_private_slot(0, owner=1)
    lease = txn.lease_queue(0, owner=1, layer_control_index=0)
    txn.prepare_post(lease, candidate_slots=[0])
    wave = txn.post_only(lease)
    assert txn.try_claim_drain(wave)
    txn.observe_cq(wave, succeeded=False)
    txn.finalize_wave(wave)

    assert resources.candidates[0].state is CandidateState.DEFERRED
    assert resources.private_slots[0].state is PrivateSlotState.FAILED
    with pytest.raises(P8LifecycleError, match="failed CQ"):
        txn.publish_private(wave)

    txn.release_queue(wave)
    txn.seal_admission()
    txn.begin_draining()
    txn.begin_reconcile()


def test_p8_qsfa_requires_per_layer_exact_winner_and_guaranteed_recall() -> None:
    """Speculation never bypasses the exact winner/recall correctness gate."""

    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=121, graph_generation=0, expected_layers=2)
    txn.begin_admission()
    txn.seal_admission()
    txn.begin_draining()
    txn.begin_reconcile()

    txn.mark_exact_winners_confirmed(0)
    with pytest.raises(P8LifecycleError, match="GuaranteedRecall"):
        txn.mark_guaranteed_recall_terminal(1)
    with pytest.raises(P8LifecycleError, match="exact_missing"):
        txn.begin_qsfa_reading()

    txn.mark_exact_winners_confirmed(1)
    txn.mark_guaranteed_recall_terminal(0)
    with pytest.raises(P8LifecycleError, match="recall_missing"):
        txn.begin_qsfa_reading()
    txn.mark_guaranteed_recall_terminal(1)
    txn.begin_qsfa_reading()


def test_p8_timeout_quarantines_and_blocks_reuse() -> None:
    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=13, graph_generation=0)
    txn.retain_host_lease("native-wave")
    txn.begin_admission()
    txn.publish_candidate(0, producer=2, layer_control_index=0)
    txn.reserve_private_slot(0, owner=2)
    lease = txn.lease_queue(0, owner=2, layer_control_index=0)
    txn.prepare_post(lease, candidate_slots=[0])
    wave = txn.post_only(lease)
    assert txn.try_claim_drain(wave)

    with pytest.raises(P8QuarantineError, match="timeout"):
        txn.timeout_queue(wave, reason="CQ did not reach terminal")

    assert txn.state is ForwardState.ABORTING
    assert resources.queues[0].state is QueueState.QUARANTINED
    assert resources.private_slots[0].state is PrivateSlotState.QUARANTINED
    assert txn.host_leases == set()
    assert resources.quarantine_keepalives == {"native-wave"}

    txn.enter_forward_terminal()
    txn.mark_fallback_ready()
    assert txn.state is ForwardState.FALLBACK_READY
    assert resources.active_txn is None
    with pytest.raises(P8QuarantineError, match="quarantined"):
        resources.begin_forward(epoch=14, graph_generation=0)


def test_p8_release_cas_failure_quarantines_instead_of_silent_reuse() -> None:
    resources, txn, wave = _post_one_successful_wave()

    forged = replace(wave, generation=wave.generation + 1)
    with pytest.raises(P8QuarantineError, match="release failure"):
        txn.release_queue(forged)

    assert txn.state is ForwardState.ABORTING
    assert resources.queues[0].state is QueueState.QUARANTINED
    assert resources.private_slots[0].state is PrivateSlotState.QUARANTINED


def test_p8_post_intent_ambiguity_quarantines_without_cq_consumption() -> None:
    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=15, graph_generation=1)
    txn.begin_admission()
    txn.publish_candidate(0, producer=4, layer_control_index=0)
    txn.reserve_private_slot(0, owner=4)
    lease = txn.lease_queue(0, owner=4, layer_control_index=0)
    txn.prepare_post(lease, candidate_slots=[0])

    with pytest.raises(P8QuarantineError, match="timeout"):
        txn.timeout_queue(lease, reason="doorbell outcome is ambiguous")

    assert resources.queues[0].state is QueueState.QUARANTINED
    assert resources.queues[0].cq_success is None
    assert resources.private_slots[0].state is PrivateSlotState.QUARANTINED


def test_p8_post_rejected_drain_exactly_cleans_pending_private_rows() -> None:
    """A proved-no-doorbell rejection is the one safe pre-CQ cleanup path."""

    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=151, graph_generation=1)
    txn.begin_admission()
    txn.publish_candidate(
        0,
        producer=4,
        layer_control_index=0,
        source_offset=701,
        object_offset=702,
    )
    txn.reserve_private_slot(0, owner=4)
    lease = txn.lease_queue(0, owner=4, layer_control_index=0)
    txn.prepare_post(lease, candidate_slots=[0])

    rejected = txn.record_post_rejected(lease)
    queue = resources.queues[0]
    assert queue.state is QueueState.FINALIZED
    assert not queue.post_committed
    assert queue.post_rejected
    assert queue.cq_success is None
    assert not queue.private_ready_published
    assert resources.private_slots[0].state is PrivateSlotState.PENDING
    assert resources.private_source_to_slot == {701: 0}
    assert resources.private_object_to_slot == {702: 0}

    # A competing dynamic drain worker loses the FINALIZED -> DRAIN_CLAIMED
    # claim and must not clear mappings or release the private row.
    txn.seal_admission()
    txn.begin_draining()
    assert txn.try_claim_post_rejected_drain(rejected) is True
    assert txn.try_claim_post_rejected_drain(rejected) is False
    assert resources.private_slots[0].state is PrivateSlotState.PENDING

    txn.cleanup_post_rejected_wave(rejected)
    assert resources.queues[0].state is QueueState.FREE
    assert resources.candidates[0].state is CandidateState.DEFERRED
    assert resources.candidates[0].private_slot is None
    assert resources.private_slots[0].state is PrivateSlotState.FREE
    assert resources.private_ready[0] is PrivateSlotState.FREE
    assert resources.private_source_to_slot == {}
    assert resources.private_object_to_slot == {}
    assert not txn.abort_latched


def test_p8_post_rejected_stale_mapping_quarantines_instead_of_freeing() -> None:
    """A stale source/object map invalidates exact no-post cleanup proof."""

    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=152, graph_generation=1)
    txn.begin_admission()
    txn.publish_candidate(
        0,
        producer=4,
        layer_control_index=0,
        source_offset=703,
        object_offset=704,
    )
    txn.reserve_private_slot(0, owner=4)
    lease = txn.lease_queue(0, owner=4, layer_control_index=0)
    txn.prepare_post(lease, candidate_slots=[0])
    rejected = txn.record_post_rejected(lease)
    txn.seal_admission()
    txn.begin_draining()
    assert txn.try_claim_post_rejected_drain(rejected)

    resources.private_object_to_slot[704] = 99
    with pytest.raises(P8QuarantineError, match="POST_REJECTED cleanup failure"):
        txn.cleanup_post_rejected_wave(rejected)

    assert txn.abort_latched
    assert resources.queues[0].state is QueueState.QUARANTINED
    assert resources.private_slots[0].state is PrivateSlotState.QUARANTINED
    # Ordered source-then-object clear intentionally does not compensate a
    # partial clear; quarantine prevents the remaining stale map from reuse.
    assert resources.private_source_to_slot == {}
    assert resources.private_object_to_slot == {704: 99}


def test_p8_post_rejected_proof_mismatch_quarantines() -> None:
    """A FINALIZED record that also claims a doorbell is never early-freed."""

    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=153, graph_generation=1)
    txn.begin_admission()
    txn.publish_candidate(0, producer=4, layer_control_index=0)
    txn.reserve_private_slot(0, owner=4)
    lease = txn.lease_queue(0, owner=4, layer_control_index=0)
    txn.prepare_post(lease, candidate_slots=[0])
    rejected = txn.record_post_rejected(lease)
    txn.seal_admission()
    txn.begin_draining()

    # Model a corrupted durable control row: POST_REJECTED is only safe when
    # it *also* proves that no doorbell was committed.
    resources.queues[0].post_committed = True
    with pytest.raises(P8QuarantineError, match="POST_REJECTED proof mismatch"):
        txn.try_claim_post_rejected_drain(rejected)

    assert txn.abort_latched
    assert resources.queues[0].state is QueueState.QUARANTINED
    assert resources.private_slots[0].state is PrivateSlotState.QUARANTINED


def test_p8_cq_error_is_releasable_only_after_cq_observed() -> None:
    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=16, graph_generation=0)
    txn.begin_admission()
    txn.publish_candidate(0, producer=5, layer_control_index=0)
    txn.reserve_private_slot(0, owner=5)
    lease = txn.lease_queue(0, owner=5, layer_control_index=0)
    txn.prepare_post(lease, candidate_slots=[0])
    wave = txn.post_only(lease)
    assert txn.try_claim_drain(wave)

    # A caller that tries to release before CQ observation cannot prove that
    # the device writer stopped, so the queue is quarantined rather than freed.
    with pytest.raises(P8QuarantineError, match="release failure"):
        txn.release_queue(wave)
    assert resources.queues[0].state is QueueState.QUARANTINED


def test_p8_candidate_claim_has_single_owner_and_cannot_be_reclaimed() -> None:
    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=17, graph_generation=0)
    txn.begin_admission()
    txn.publish_candidate(0, producer=8, layer_control_index=0)
    txn.claim_candidate(0, owner=8)

    assert resources.candidates[0].state is CandidateState.CLAIMED
    assert resources.candidates[0].owner == 8
    with pytest.raises(P8LifecycleError, match="not published"):
        txn.claim_candidate(0, owner=9)


def test_p8_stale_graph_replay_identity_quarantines_queue() -> None:
    """A late CQ from the same queue but a prior graph replay is never trusted."""

    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=18, graph_generation=9, expected_layers=2)
    txn.begin_admission()
    txn.publish_candidate(0, producer=3, layer_control_index=1)
    txn.reserve_private_slot(0, owner=3)
    lease = txn.lease_queue(0, owner=3, layer_control_index=1)
    txn.prepare_post(lease, candidate_slots=[0])
    wave = txn.post_only(lease)
    assert txn.try_claim_drain(wave)

    stale_replay = replace(wave, graph_generation=wave.graph_generation - 1)
    with pytest.raises(P8QuarantineError, match="CQ identity failure"):
        txn.observe_cq(stale_replay, succeeded=True)

    assert txn.state is ForwardState.ABORTING
    assert resources.queues[0].state is QueueState.QUARANTINED
    assert resources.private_slots[0].state is PrivateSlotState.QUARANTINED


def test_p8_private_slot_rejects_stale_post_sequence() -> None:
    """Private publication also validates queue generation and post sequence."""

    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=19, graph_generation=4)
    txn.begin_admission()
    txn.publish_candidate(0, producer=1, layer_control_index=0)
    txn.reserve_private_slot(0, owner=1)
    lease = txn.lease_queue(0, owner=1, layer_control_index=0)
    txn.prepare_post(lease, candidate_slots=[0])
    wave = txn.post_only(lease)
    assert txn.try_claim_drain(wave)
    txn.observe_cq(wave, succeeded=True)
    txn.finalize_wave(wave)

    resources.private_slots[0].post_sequence += 1
    with pytest.raises(P8QuarantineError, match="private slot identity"):
        txn.publish_private(wave)

    assert resources.queues[0].state is QueueState.QUARANTINED
    assert resources.private_slots[0].state is PrivateSlotState.QUARANTINED


def test_p8_queue_reuse_keeps_monotonic_post_sequence() -> None:
    """Queue reuse cannot overwrite a prior READY global private slot.

    This is the central P8 distinction from the rejected fixed ``queue ->
    private-range`` scheme.  Queue zero is reused for a second PostOnly wave
    before QSFA, but the first payload remains in absolute private slot zero;
    the global allocator gives the second wave slot one instead.
    """

    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=2,
        private_slot_capacity=2,
    )
    txn = resources.begin_forward(epoch=20, graph_generation=5)
    txn.begin_admission()

    txn.publish_candidate(0, producer=1, layer_control_index=0)
    txn.reserve_private_slot(0, owner=1)
    lease0 = txn.lease_queue(0, owner=1, layer_control_index=0)
    txn.prepare_post(lease0, candidate_slots=[0])
    wave0 = txn.post_only(lease0)
    assert txn.try_claim_drain(wave0)
    txn.observe_cq(wave0, succeeded=True)
    txn.finalize_wave(wave0)
    txn.publish_private(wave0)
    txn.release_queue(wave0)

    first_private = resources.private_slots[0]
    assert first_private.state is PrivateSlotState.READY
    assert first_private.queue_id == 0
    assert first_private.queue_generation == wave0.generation
    assert first_private.post_sequence == wave0.post_sequence
    assert first_private.allocation_generation == 1

    txn.publish_candidate(1, producer=1, layer_control_index=0)
    second_private_slot = txn.reserve_private_slot(1, owner=1)
    assert second_private_slot == 1
    assert second_private_slot != first_private.slot_id
    lease1 = txn.lease_queue(0, owner=1, layer_control_index=0)
    assert lease1.generation == wave0.generation + 1
    assert lease1.post_sequence == wave0.post_sequence + 1
    txn.prepare_post(lease1, candidate_slots=[1])
    wave1 = txn.post_only(lease1)
    assert wave1.post_sequence == lease1.post_sequence

    # The second queue lease has no authority over slot zero.  It may only
    # use its separately claimed global private destination.
    assert first_private.state is PrivateSlotState.READY
    assert first_private.queue_generation == wave0.generation
    assert first_private.post_sequence == wave0.post_sequence
    assert resources.private_slots[1].state is PrivateSlotState.PENDING
    assert resources.private_slots[1].queue_id == 0
    assert resources.private_slots[1].queue_generation == wave1.generation


def test_p8_global_private_allocator_exhaustion_defers_without_reusing_ready() -> None:
    """A full private arena defers speculation rather than overwriting QSFA data."""

    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=2,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=21, graph_generation=5)
    txn.begin_admission()

    txn.publish_candidate(0, producer=2, layer_control_index=0)
    txn.reserve_private_slot(0, owner=2)
    lease0 = txn.lease_queue(0, owner=2, layer_control_index=0)
    txn.prepare_post(lease0, candidate_slots=[0])
    wave0 = txn.post_only(lease0)
    assert txn.try_claim_drain(wave0)
    txn.observe_cq(wave0, succeeded=True)
    txn.finalize_wave(wave0)
    txn.publish_private(wave0)
    txn.release_queue(wave0)

    txn.publish_candidate(1, producer=2, layer_control_index=0)
    with pytest.raises(P8LifecycleError, match="private arena is full"):
        txn.reserve_private_slot(1, owner=2)

    assert resources.private_slots[0].state is PrivateSlotState.READY
    assert resources.candidates[1].state is CandidateState.DEFERRED
    assert resources.candidates[1].private_slot is None


def test_p8_fallback_cannot_detach_unresolved_private_ownership() -> None:
    """Abort never silently detaches a PENDING private slot from its forward."""

    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=22, graph_generation=0)
    txn.begin_admission()
    txn.publish_candidate(0, producer=3, layer_control_index=0)
    txn.reserve_private_slot(0, owner=3)
    txn.retain_host_lease("unposted-native-resource")
    txn.abort("runtime selected exact fallback before PostOnly")

    with pytest.raises(P8LifecycleError, match="FORWARD_TERMINAL"):
        txn.mark_fallback_ready()

    # This reservation never reached PostOnly, so it is safe to abandon into
    # the exact-recall path; only then can the forward detach for fallback.
    txn.abandon_unposted_candidate(0, owner=3)
    txn.enter_forward_terminal()
    txn.mark_fallback_ready()
    assert txn.state is ForwardState.FALLBACK_READY
    assert resources.active_txn is None
    assert resources.private_slots[0].state is PrivateSlotState.FREE
    assert resources.quarantine_keepalives == {"unposted-native-resource"}


def test_p8_bound_descriptor_cannot_be_deferred_or_freed() -> None:
    """POST_INTENT is ambiguous, so only quarantine can end its ownership."""

    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=23, graph_generation=0)
    txn.begin_admission()
    txn.publish_candidate(0, producer=4, layer_control_index=0)
    txn.reserve_private_slot(0, owner=4)
    lease = txn.lease_queue(0, owner=4, layer_control_index=0)
    txn.prepare_post(lease, candidate_slots=[0])

    with pytest.raises(P8LifecycleError, match="already bound to queue"):
        txn.defer_candidate(0, owner=4)
    assert resources.private_slots[0].state is PrivateSlotState.PENDING

    with pytest.raises(P8QuarantineError, match="timeout"):
        txn.timeout_queue(lease, reason="post intent has no definitive doorbell")
    assert resources.private_slots[0].state is PrivateSlotState.QUARANTINED


def test_p8_dynamic_drain_claim_has_one_owner_and_can_return_to_posted() -> None:
    """Only one AIV may poll a durable wave; an empty poll restores POSTED."""

    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=24, graph_generation=1)
    txn.begin_admission()
    txn.publish_candidate(0, producer=6, layer_control_index=0)
    txn.reserve_private_slot(0, owner=6)
    lease = txn.lease_queue(0, owner=6, layer_control_index=0)
    txn.prepare_post(lease, candidate_slots=[0])
    wave = txn.post_only(lease)

    assert txn.try_claim_drain(wave) is True
    assert resources.queues[0].state is QueueState.DRAIN_CLAIMED
    # A competing dynamic worker loses the CAS and must not touch CQ state.
    assert txn.try_claim_drain(wave) is False
    assert resources.queues[0].cq_success is None

    txn.leave_drain_pending(wave)
    assert resources.queues[0].state is QueueState.POSTED
    assert txn.try_claim_drain(wave) is True


def test_p8_qsfa_release_requires_exact_mapping_cas_and_quarantines_partial_clear() -> None:
    """Never reuse a payload if either source/object directory clear is stale."""

    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=25, graph_generation=2)
    txn.begin_admission()
    txn.publish_candidate(
        0,
        producer=7,
        layer_control_index=0,
        source_offset=101,
        object_offset=202,
    )
    txn.reserve_private_slot(0, owner=7)
    lease = txn.lease_queue(0, owner=7, layer_control_index=0)
    txn.prepare_post(lease, candidate_slots=[0])
    wave = txn.post_only(lease)
    assert txn.try_claim_drain(wave)
    txn.observe_cq(wave, succeeded=True)
    txn.finalize_wave(wave)
    txn.publish_private(wave)
    txn.release_queue(wave)
    txn.seal_admission()
    txn.begin_draining()
    txn.begin_reconcile()
    txn.mark_exact_winners_confirmed(0)
    txn.mark_guaranteed_recall_terminal(0)
    txn.begin_qsfa_reading()

    # Simulate an ABA/stale owner in just one directory.  Native P8 clears
    # source first, then object; a partial clear is intentionally terminally
    # non-reusable rather than compensated.
    resources.private_object_to_slot[202] = 99
    with pytest.raises(P8QuarantineError, match="mapping-clear mismatch"):
        txn.mark_qsfa_terminal(0)

    assert txn.abort_latched
    assert txn.state is ForwardState.ABORTING
    assert resources.private_slots[0].state is PrivateSlotState.QUARANTINED
    assert resources.private_ready[0] is PrivateSlotState.QUARANTINED
    assert resources.private_source_to_slot == {}
    assert resources.private_object_to_slot == {202: 99}


def test_p8_forward_terminal_quarantines_unresolved_wave_and_preserves_abort_reason() -> None:
    """Forward terminal is fail-closed; it never guesses an in-flight CQ safe."""

    resources = P8FixedResources(
        queue_count=1,
        candidate_capacity=1,
        private_slot_capacity=1,
    )
    txn = resources.begin_forward(epoch=26, graph_generation=3)
    txn.retain_host_lease("p8-native-keepalive")
    txn.begin_admission()
    txn.publish_candidate(
        0,
        producer=8,
        layer_control_index=0,
        source_offset=303,
        object_offset=404,
    )
    txn.reserve_private_slot(0, owner=8)
    lease = txn.lease_queue(0, owner=8, layer_control_index=0)
    txn.prepare_post(lease, candidate_slots=[0])
    wave = txn.post_only(lease)
    assert txn.try_claim_drain(wave)

    txn.abort("first worker observed control corruption")
    txn.abort("later worker must not overwrite first reason")
    txn.enter_forward_terminal()

    assert txn.phase is ForwardPhase.FORWARD_TERMINAL
    assert txn.abort_reason == "first worker observed control corruption"
    assert resources.queues[0].state is QueueState.QUARANTINED
    assert resources.private_slots[0].state is PrivateSlotState.QUARANTINED
    # Terminal quarantine deliberately leaves both maps installed.  A late
    # DMA writer can never be mistaken for a reusable private payload.
    assert resources.private_source_to_slot == {303: 0}
    assert resources.private_object_to_slot == {404: 0}
    assert resources.quarantine_keepalives == {"p8-native-keepalive"}

    txn.mark_fallback_ready()
    assert txn.state is ForwardState.FALLBACK_READY
    assert resources.active_txn is None
