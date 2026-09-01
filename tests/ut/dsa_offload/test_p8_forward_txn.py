# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure-Python contract tests for the P8 forward-owned async transaction."""

from __future__ import annotations

from dataclasses import replace

import pytest

from vllm_ascend.dsa_offload.p8_forward_txn import (
    CandidateState,
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
    txn.observe_cq(wave, succeeded=True)
    txn.finalize_wave(wave)
    txn.publish_private(wave)
    txn.release_queue(wave)
    return resources, txn, wave


def test_p8_normal_forward_releases_only_after_qsfa_terminal() -> None:
    resources, txn, _ = _post_one_successful_wave()

    assert resources.queues[0].state is QueueState.RELEASED
    assert resources.candidates[0].state is CandidateState.CONSUMED
    assert resources.private_slots[0].state is PrivateSlotState.READY
    assert resources.private_slots[0].layer_control_index == 0
    assert resources.private_slots[0].queue_generation == 1
    assert resources.private_slots[0].post_sequence == 1

    txn.seal_admission()
    txn.begin_draining()
    txn.begin_reconcile()
    txn.begin_qsfa_reading()

    with pytest.raises(P8LifecycleError, match="model terminal"):
        txn.release()

    txn.mark_qsfa_terminal(0)
    with pytest.raises(P8LifecycleError, match="every layer"):
        txn.mark_model_terminal_enqueued()

    txn.mark_qsfa_terminal(1)
    txn.mark_model_terminal_enqueued()
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

    with pytest.raises(P8QuarantineError, match="timeout"):
        txn.timeout_queue(wave, reason="CQ did not reach terminal")

    assert txn.state is ForwardState.ABORTING
    assert resources.queues[0].state is QueueState.QUARANTINED
    assert resources.private_slots[0].state is PrivateSlotState.QUARANTINED
    assert txn.host_leases == {"native-wave"}

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
    assert lease1.post_sequence == wave0.post_sequence
    txn.prepare_post(lease1, candidate_slots=[1])
    wave1 = txn.post_only(lease1)
    assert wave1.post_sequence == wave0.post_sequence + 1

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
    txn.abort("runtime selected exact fallback before PostOnly")

    with pytest.raises(P8LifecycleError, match="candidate/private ownership"):
        txn.mark_fallback_ready()

    # This reservation never reached PostOnly, so it is safe to abandon into
    # the exact-recall path; only then can the forward detach for fallback.
    txn.abandon_unposted_candidate(0, owner=3)
    txn.mark_fallback_ready()
    assert txn.state is ForwardState.FALLBACK_READY
    assert resources.active_txn is None
    assert resources.private_slots[0].state is PrivateSlotState.FREE


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
