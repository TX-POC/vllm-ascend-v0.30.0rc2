# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward-owned lifecycle model for P8 streaming speculative Get.

The production P8 path will keep these records in fixed-address device
buffers.  This module deliberately contains no device or KVAF imports: it is
the small, testable ownership contract shared by the future runtime adapter,
the graph binding, and the native PostOnly/CQ implementation.

There are two different lifetimes in the design:

* :class:`P8FixedResources` lives with ``DSAOffloadRuntime``.  It represents
  the bounded queue, candidate-ring and private-slot capacity whose addresses
  must stay stable while a graph is captured and replayed.
* :class:`P8ForwardAsyncTxn` owns one forward's identity, host keepalive
  leases, and the legal transition through admission, DMA completion,
  reconcile, QSFA reading, and release.

The model is intentionally strict.  A stale CQ identity, timeout, or failed
queue-release compare-and-swap is not treated as an ordinary retry.  The
affected queue and private destination become ``QUARANTINED`` and the forward
can only move to ``FALLBACK_READY``.  That is the safe behaviour when a
device-side writer may still exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable


class P8LifecycleError(RuntimeError):
    """Raised when a P8 lifecycle transition would violate its contract."""


class P8QuarantineError(P8LifecycleError):
    """Raised after a resource is quarantined instead of being reused."""


class ForwardState(Enum):
    """Forward transaction states in their only legal high-level order."""

    OPEN = auto()
    ADMITTING = auto()
    SEALED = auto()
    DRAINING = auto()
    RECONCILING = auto()
    QSFA_READING = auto()
    RELEASED = auto()
    ABORTING = auto()
    FALLBACK_READY = auto()


class QueueState(Enum):
    """State of one fixed Direct-HCCP queue/wave record."""

    FREE = auto()
    FILLING = auto()
    POST_INTENT = auto()
    POSTED = auto()
    CQ_OBSERVED = auto()
    FINALIZED = auto()
    PRIVATE_PUBLISHED = auto()
    RELEASED = auto()
    QUARANTINED = auto()


class CandidateState(Enum):
    """State of one bounded CandidateSeed ring entry."""

    EMPTY = auto()
    PUBLISHED = auto()
    CLAIMED = auto()
    CONSUMED = auto()
    DEFERRED = auto()


class PrivateSlotState(Enum):
    """Readiness state of a fixed private HBM staging slot."""

    FREE = auto()
    PENDING = auto()
    READY = auto()
    FAILED = auto()
    QUARANTINED = auto()


@dataclass(frozen=True)
class P8WaveIdentity:
    """Immutable identity echoed by PostOnly and validated by CQ handling.

    ``epoch`` and ``graph_generation`` reject completions from a prior
    forward/replay. ``layer_control_index`` makes a completion layer-local,
    while ``generation`` and ``post_sequence`` reject reuse of the same
    physical queue. It is an internal state-machine identity, not a
    user-visible ticket API.
    """

    queue_id: int
    epoch: int
    graph_generation: int
    layer_control_index: int
    generation: int
    post_sequence: int
    owner: int


@dataclass
class CandidateRecord:
    """One fixed CandidateSeed ring cell, without copying KV payload bytes."""

    slot_id: int
    state: CandidateState = CandidateState.EMPTY
    epoch: int = 0
    graph_generation: int = 0
    layer_control_index: int = 0
    producer: int | None = None
    owner: int | None = None
    private_slot: int | None = None
    # A private slot comes from one global bounded arena, not from a queue's
    # fixed slice.  Its allocation generation prevents a stale candidate
    # record from being rebound merely because the same numerical slot was
    # recycled later.
    private_allocation_generation: int = 0


@dataclass
class PrivateSlotRecord:
    """One global HBM private destination and its transaction identity.

    ``slot_id`` is an absolute index in the graph-stable private arena.  It
    deliberately has no relationship to ``queue_id``: a queue is released as
    soon as its CQ is terminal and the private record is published, while the
    private slot remains owned until QSFA's last reader.  The bounded global
    allocator below therefore skips ``READY`` slots instead of allowing a
    newly leased queue to overwrite them.
    """

    slot_id: int
    state: PrivateSlotState = PrivateSlotState.FREE
    epoch: int = 0
    graph_generation: int = 0
    layer_control_index: int = 0
    owner: int | None = None
    allocation_generation: int = 0
    queue_id: int | None = None
    queue_generation: int = 0
    post_sequence: int = 0


@dataclass
class P8WaveRecord:
    """One fixed queue lease plus the candidates and private slots it owns."""

    queue_id: int
    state: QueueState = QueueState.FREE
    epoch: int = 0
    graph_generation: int = 0
    layer_control_index: int = 0
    generation: int = 0
    post_sequence: int = 0
    owner: int | None = None
    candidate_slots: tuple[int, ...] = ()
    private_slots: tuple[int, ...] = ()
    cq_success: bool | None = None
    quarantine_reason: str | None = None


@dataclass
class P8FixedResources:
    """Bounded P8 resources that persist across graph replays.

    The real implementation will back these records with graph-stable NPU
    tensors and native queue storage.  Queue descriptors and private HBM
    destinations have intentionally separate ownership: queues are a short
    CQ-scoped pool, whereas private slots form one global, bounded allocator
    whose leases survive queue reuse until QSFA terminal.  A new forward
    reuses only cleanly released records; a quarantined record makes P8
    unavailable until runtime teardown explicitly repairs or recreates it.
    """

    queue_count: int
    candidate_capacity: int
    private_slot_capacity: int
    queues: list[P8WaveRecord] = field(init=False)
    candidates: list[CandidateRecord] = field(init=False)
    private_slots: list[PrivateSlotRecord] = field(init=False)
    _next_private_slot_hint: int = field(init=False, default=0, repr=False)
    active_txn: P8ForwardAsyncTxn | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.queue_count <= 0:
            raise ValueError("queue_count must be positive")
        if self.candidate_capacity <= 0:
            raise ValueError("candidate_capacity must be positive")
        if self.private_slot_capacity <= 0:
            raise ValueError("private_slot_capacity must be positive")
        self.queues = [P8WaveRecord(queue_id=index) for index in range(self.queue_count)]
        self.candidates = [
            CandidateRecord(slot_id=index)
            for index in range(self.candidate_capacity)
        ]
        self.private_slots = [
            PrivateSlotRecord(slot_id=index)
            for index in range(self.private_slot_capacity)
        ]

    def claim_global_private_slot(
        self,
        *,
        epoch: int,
        graph_generation: int,
        layer_control_index: int,
        owner: int,
    ) -> PrivateSlotRecord | None:
        """Claim one free absolute slot from the global private HBM arena.

        The native equivalent is a bounded CAS/bitmap probe.  The model uses
        a deterministic round-robin scan so tests can prove the key
        invariant: this allocation is independent of the Direct-HCCP queue
        selected for the wave, and no READY slot is reissued before QSFA
        terminal releases it.
        """

        for offset in range(self.private_slot_capacity):
            slot_id = (self._next_private_slot_hint + offset) % self.private_slot_capacity
            private_slot = self.private_slots[slot_id]
            if private_slot.state is not PrivateSlotState.FREE:
                continue
            private_slot.state = PrivateSlotState.PENDING
            private_slot.epoch = epoch
            private_slot.graph_generation = graph_generation
            private_slot.layer_control_index = layer_control_index
            private_slot.owner = owner
            private_slot.allocation_generation += 1
            private_slot.queue_id = None
            private_slot.queue_generation = 0
            private_slot.post_sequence = 0
            self._next_private_slot_hint = (slot_id + 1) % self.private_slot_capacity
            return private_slot
        return None

    def release_global_private_slot(self, private_slot: PrivateSlotRecord) -> None:
        """Return one proven-safe private slot to the global allocator.

        Callers may use this only before PostOnly, after failed CQ handling,
        or after the forward's QSFA terminal gate.  ``allocation_generation``
        intentionally remains monotonic across reuse.
        """

        if private_slot.state is PrivateSlotState.QUARANTINED:
            raise P8QuarantineError(
                f"private slot {private_slot.slot_id} is quarantined"
            )
        private_slot.state = PrivateSlotState.FREE
        private_slot.epoch = 0
        private_slot.graph_generation = 0
        private_slot.layer_control_index = 0
        private_slot.owner = None
        private_slot.queue_id = None
        private_slot.queue_generation = 0
        private_slot.post_sequence = 0

    @property
    def has_quarantine(self) -> bool:
        """Whether any fixed resource is unsafe to reuse."""

        return any(
            queue.state is QueueState.QUARANTINED for queue in self.queues
        ) or any(
            slot.state is PrivateSlotState.QUARANTINED
            for slot in self.private_slots
        )

    def begin_forward(
        self,
        *,
        epoch: int,
        graph_generation: int,
        expected_layers: int = 1,
    ) -> P8ForwardAsyncTxn:
        """Open one forward transaction over the preallocated resources."""

        if epoch <= 0:
            raise ValueError("epoch must be positive")
        if graph_generation < 0:
            raise ValueError("graph_generation must be non-negative")
        if expected_layers <= 0:
            raise ValueError("expected_layers must be positive")
        if self.active_txn is not None:
            raise P8LifecycleError("previous P8 forward is still active")
        if self.has_quarantine:
            raise P8QuarantineError(
                "P8 fixed resources contain quarantined queue/private slots; "
                "route this forward to the exact fallback until teardown"
            )
        self._reset_clean_records()
        txn = P8ForwardAsyncTxn(
            resources=self,
            epoch=epoch,
            graph_generation=graph_generation,
            expected_layers=expected_layers,
        )
        self.active_txn = txn
        return txn

    def _reset_clean_records(self) -> None:
        """Reuse only records that were terminally released by a prior txn."""

        for queue in self.queues:
            if queue.state not in {QueueState.FREE, QueueState.RELEASED}:
                raise P8LifecycleError(
                    "cannot reset non-terminal queue "
                    f"{queue.queue_id}: state={queue.state.name}"
                )
            queue.state = QueueState.FREE
            queue.epoch = 0
            queue.graph_generation = 0
            queue.layer_control_index = 0
            queue.owner = None
            queue.candidate_slots = ()
            queue.private_slots = ()
            queue.cq_success = None
            queue.quarantine_reason = None

        for candidate in self.candidates:
            if candidate.state not in {
                CandidateState.EMPTY,
                CandidateState.CONSUMED,
                CandidateState.DEFERRED,
            }:
                raise P8LifecycleError(
                    "cannot reset unfinished candidate "
                    f"{candidate.slot_id}: state={candidate.state.name}"
                )
            candidate.state = CandidateState.EMPTY
            candidate.epoch = 0
            candidate.graph_generation = 0
            candidate.layer_control_index = 0
            candidate.producer = None
            candidate.owner = None
            candidate.private_slot = None
            candidate.private_allocation_generation = 0

        for private_slot in self.private_slots:
            if private_slot.state is not PrivateSlotState.FREE:
                raise P8LifecycleError(
                    "cannot reset non-free private slot "
                    f"{private_slot.slot_id}: state={private_slot.state.name}"
                )
            private_slot.epoch = 0
            private_slot.graph_generation = 0
            private_slot.layer_control_index = 0
            private_slot.owner = None
            private_slot.queue_id = None
            private_slot.queue_generation = 0
            private_slot.post_sequence = 0

    def _detach(self, txn: P8ForwardAsyncTxn) -> None:
        if self.active_txn is not txn:
            raise P8LifecycleError("transaction is not this resource owner's active txn")
        self.active_txn = None


@dataclass
class P8ForwardAsyncTxn:
    """One forward-owned asynchronous Get lifecycle.

    A transaction never changes the persistent resident mapping.  It owns
    only speculative private slots and host keepalive leases.  The caller may
    release those leases only after all queue waves are terminal, all required
    QSFA readers have passed, and the model-terminal operation was enqueued.
    """

    resources: P8FixedResources
    epoch: int
    graph_generation: int
    expected_layers: int
    state: ForwardState = ForwardState.OPEN
    host_leases: set[str] = field(default_factory=set)
    qsfa_terminal_layers: set[int] = field(default_factory=set)
    model_terminal_enqueued: bool = False
    abort_reason: str | None = None

    def retain_host_lease(self, lease_name: str) -> None:
        """Keep a host/native resource alive until this txn reaches terminal."""

        self._require_not_terminal()
        if not lease_name:
            raise ValueError("lease_name must be non-empty")
        self.host_leases.add(lease_name)

    def begin_admission(self) -> None:
        """Allow score-tile producers to publish CandidateSeed entries."""

        self._transition(ForwardState.OPEN, ForwardState.ADMITTING)

    def publish_candidate(
        self,
        candidate_slot: int,
        *,
        producer: int,
        layer_control_index: int,
    ) -> None:
        """Publish one already-compacted candidate into its fixed ring cell."""

        self._require_state(ForwardState.ADMITTING)
        self._validate_layer_control_index(layer_control_index)
        candidate = self._candidate(candidate_slot)
        if candidate.state is not CandidateState.EMPTY:
            raise P8LifecycleError(
                f"candidate {candidate_slot} is not empty: {candidate.state.name}"
            )
        candidate.state = CandidateState.PUBLISHED
        candidate.epoch = self.epoch
        candidate.graph_generation = self.graph_generation
        candidate.layer_control_index = layer_control_index
        candidate.producer = producer

    def claim_candidate(self, candidate_slot: int, *, owner: int) -> None:
        """Claim a published seed before it can be put into a Get wave."""

        self._require_state(
            ForwardState.ADMITTING,
            ForwardState.SEALED,
            ForwardState.DRAINING,
        )
        candidate = self._candidate(candidate_slot)
        self._require_candidate_identity(candidate)
        if candidate.state is not CandidateState.PUBLISHED:
            raise P8LifecycleError(
                f"candidate {candidate_slot} is not published: {candidate.state.name}"
            )
        candidate.state = CandidateState.CLAIMED
        candidate.owner = owner

    def defer_candidate(self, candidate_slot: int, *, owner: int) -> None:
        """Give up only the speculative opportunity; exact recall remains valid."""

        self._require_state(
            ForwardState.ADMITTING,
            ForwardState.SEALED,
            ForwardState.DRAINING,
        )
        candidate = self._candidate(candidate_slot)
        self._require_candidate_identity(candidate)
        if candidate.state not in {CandidateState.PUBLISHED, CandidateState.CLAIMED}:
            raise P8LifecycleError(
                f"candidate {candidate_slot} cannot be deferred from "
                f"{candidate.state.name}"
            )
        if candidate.owner is not None and candidate.owner != owner:
            raise P8LifecycleError(
                f"candidate {candidate_slot} belongs to owner {candidate.owner}, "
                f"not {owner}"
            )
        if candidate.private_slot is not None:
            private_slot = self._private_slot(candidate.private_slot)
            if private_slot.queue_id is not None:
                raise P8LifecycleError(
                    f"candidate {candidate_slot} is already bound to queue "
                    f"{private_slot.queue_id}; a descriptor may be postable, "
                    "so use timeout/quarantine instead of deferring"
                )
        self._free_candidate_private_slot(candidate)
        candidate.state = CandidateState.DEFERRED
        candidate.owner = owner

    def abandon_unposted_candidate(self, candidate_slot: int, *, owner: int) -> None:
        """Safely defer a seed that provably never entered a PostOnly wave.

        This is deliberately narrower than a generic abort cleanup.  Once a
        descriptor has been associated with a queue, ``POST_INTENT`` is an
        ambiguous state and must be quarantined rather than freed.  A seed
        whose private slot has no queue binding has not reached that boundary
        and can be returned to exact recall without retaining the forward.
        """

        self._require_state(
            ForwardState.ADMITTING,
            ForwardState.SEALED,
            ForwardState.DRAINING,
            ForwardState.ABORTING,
        )
        candidate = self._candidate(candidate_slot)
        self._require_candidate_identity(candidate)
        if candidate.state not in {CandidateState.PUBLISHED, CandidateState.CLAIMED}:
            raise P8LifecycleError(
                f"candidate {candidate_slot} cannot be abandoned from "
                f"{candidate.state.name}"
            )
        if candidate.owner is not None and candidate.owner != owner:
            raise P8LifecycleError(
                f"candidate {candidate_slot} belongs to owner {candidate.owner}, "
                f"not {owner}"
            )
        if candidate.private_slot is not None:
            private_slot = self._private_slot(candidate.private_slot)
            if private_slot.state is not PrivateSlotState.PENDING:
                raise P8LifecycleError(
                    f"candidate {candidate_slot} private slot "
                    f"{private_slot.slot_id} is {private_slot.state.name}, "
                    "not an unposted pending reservation"
                )
            if private_slot.queue_id is not None:
                raise P8LifecycleError(
                    f"candidate {candidate_slot} is already bound to queue "
                    f"{private_slot.queue_id}; use timeout/quarantine instead"
                )
        self._free_candidate_private_slot(candidate)
        candidate.state = CandidateState.DEFERRED
        candidate.owner = owner

    def reserve_private_slot(self, candidate_slot: int, *, owner: int) -> int:
        """Reserve a global private HBM slot before a descriptor is filled.

        This operation is intentionally independent of ``queue_id``.  A
        candidate receives an absolute output slot from the shared bounded
        arena first; only ``prepare_post`` later binds that slot to the
        short-lived queue wave that performs its DMA.
        """

        self._require_state(
            ForwardState.ADMITTING,
            ForwardState.SEALED,
            ForwardState.DRAINING,
        )
        candidate = self._candidate(candidate_slot)
        self._require_candidate_identity(candidate)
        if candidate.state is not CandidateState.PUBLISHED:
            raise P8LifecycleError(
                "private-slot reservation requires an unclaimed published "
                f"candidate, got {candidate.state.name}"
            )
        self.claim_candidate(candidate_slot, owner=owner)
        private_slot = self.resources.claim_global_private_slot(
            epoch=self.epoch,
            graph_generation=self.graph_generation,
            layer_control_index=candidate.layer_control_index,
            owner=owner,
        )
        if private_slot is not None:
            candidate.private_slot = private_slot.slot_id
            candidate.private_allocation_generation = private_slot.allocation_generation
            return private_slot.slot_id
        self.defer_candidate(candidate_slot, owner=owner)
        raise P8LifecycleError("private arena is full; candidate was deferred")

    def lease_queue(
        self,
        queue_id: int,
        *,
        owner: int,
        layer_control_index: int,
    ) -> P8WaveIdentity:
        """CAS-lease one fixed queue for a short device-side PostOnly wave."""

        self._require_state(
            ForwardState.ADMITTING,
            ForwardState.SEALED,
            ForwardState.DRAINING,
        )
        self._validate_layer_control_index(layer_control_index)
        queue = self._queue(queue_id)
        if queue.state is QueueState.QUARANTINED:
            raise P8QuarantineError(f"queue {queue_id} is quarantined")
        if queue.state not in {QueueState.FREE, QueueState.RELEASED}:
            raise P8LifecycleError(
                f"queue {queue_id} cannot be leased from {queue.state.name}"
            )
        queue.generation += 1
        queue.epoch = self.epoch
        queue.graph_generation = self.graph_generation
        queue.layer_control_index = layer_control_index
        queue.owner = owner
        queue.candidate_slots = ()
        queue.private_slots = ()
        queue.cq_success = None
        queue.quarantine_reason = None
        queue.state = QueueState.FILLING
        return P8WaveIdentity(
            queue_id=queue_id,
            epoch=self.epoch,
            graph_generation=self.graph_generation,
            layer_control_index=layer_control_index,
            generation=queue.generation,
            # ``post_sequence`` is monotonic for the physical queue.  A new
            # lease starts from the prior value and PostOnly advances it,
            # rather than resetting to zero when the queue is reused.
            post_sequence=queue.post_sequence,
            owner=owner,
        )

    def prepare_post(
        self,
        lease: P8WaveIdentity,
        *,
        candidate_slots: Iterable[int],
    ) -> None:
        """Fill a wave's fixed descriptors, but do not ring the doorbell yet."""

        self._require_state(
            ForwardState.ADMITTING,
            ForwardState.SEALED,
            ForwardState.DRAINING,
        )
        queue = self._validate_identity(lease, expected_state=QueueState.FILLING)
        slots = tuple(candidate_slots)
        if not slots:
            raise ValueError("a PostOnly wave must contain at least one candidate")
        if len(set(slots)) != len(slots):
            raise ValueError("candidate_slots must not contain duplicates")
        private_slots: list[int] = []
        for candidate_slot in slots:
            candidate = self._candidate(candidate_slot)
            self._require_candidate_identity(candidate)
            if candidate.layer_control_index != lease.layer_control_index:
                raise P8LifecycleError(
                    f"candidate {candidate_slot} layer mismatch: "
                    f"{candidate.layer_control_index} != "
                    f"{lease.layer_control_index}"
                )
            if candidate.state is not CandidateState.CLAIMED:
                raise P8LifecycleError(
                    f"candidate {candidate_slot} is not claimed: "
                    f"{candidate.state.name}"
                )
            if candidate.owner != lease.owner:
                raise P8LifecycleError(
                    f"candidate {candidate_slot} owner mismatch: "
                    f"{candidate.owner} != {lease.owner}"
                )
            if candidate.private_slot is None:
                raise P8LifecycleError(
                    f"candidate {candidate_slot} has no reserved private slot"
                )
            private_slot = self._private_slot(candidate.private_slot)
            if private_slot.state is not PrivateSlotState.PENDING:
                raise P8LifecycleError(
                    f"private slot {private_slot.slot_id} is not pending: "
                    f"{private_slot.state.name}"
                )
            if private_slot.layer_control_index != lease.layer_control_index:
                raise P8LifecycleError(
                    f"private slot {private_slot.slot_id} layer mismatch: "
                    f"{private_slot.layer_control_index} != "
                    f"{lease.layer_control_index}"
                )
            if private_slot.owner != lease.owner:
                raise P8LifecycleError(
                    f"private slot {private_slot.slot_id} owner mismatch: "
                    f"{private_slot.owner} != {lease.owner}"
                )
            if (
                candidate.private_allocation_generation
                != private_slot.allocation_generation
            ):
                raise P8LifecycleError(
                    f"private slot {private_slot.slot_id} allocation-generation "
                    "mismatch"
                )
            private_slots.append(private_slot.slot_id)
        queue.candidate_slots = slots
        queue.private_slots = tuple(private_slots)
        for private_slot in private_slots:
            private_record = self._private_slot(private_slot)
            private_record.queue_id = queue.queue_id
            private_record.queue_generation = queue.generation
        queue.state = QueueState.POST_INTENT

    def post_only(self, lease: P8WaveIdentity) -> P8WaveIdentity:
        """Record the non-blocking doorbell transition and return CQ identity."""

        queue = self._validate_identity(lease, expected_state=QueueState.POST_INTENT)
        queue.post_sequence += 1
        for private_slot in queue.private_slots:
            self._private_slot(private_slot).post_sequence = queue.post_sequence
        queue.state = QueueState.POSTED
        return P8WaveIdentity(
            queue_id=queue.queue_id,
            epoch=self.epoch,
            graph_generation=self.graph_generation,
            layer_control_index=queue.layer_control_index,
            generation=queue.generation,
            post_sequence=queue.post_sequence,
            owner=lease.owner,
        )

    def observe_cq(self, wave: P8WaveIdentity, *, succeeded: bool) -> None:
        """Observe one terminal CQE without publishing the private view yet."""

        try:
            queue = self._validate_identity(
                wave,
                expected_state=QueueState.POSTED,
            )
        except P8LifecycleError as error:
            queue = self._queue(wave.queue_id)
            self._quarantine_queue(queue, f"CQ identity failure: {error}")
            raise P8QuarantineError(
                f"queue {wave.queue_id} quarantined after CQ identity failure"
            ) from error
        queue.cq_success = bool(succeeded)
        queue.state = QueueState.CQ_OBSERVED

    def finalize_wave(self, wave: P8WaveIdentity) -> None:
        """Finish CQ handling; success still needs explicit private publication."""

        queue = self._validate_identity(wave, expected_state=QueueState.CQ_OBSERVED)
        if queue.cq_success is None:
            raise P8LifecycleError("CQ_OBSERVED wave has no success result")
        if not queue.cq_success:
            for candidate_slot in queue.candidate_slots:
                candidate = self._candidate(candidate_slot)
                candidate.state = CandidateState.DEFERRED
                self._free_candidate_private_slot(candidate, failed=True)
        queue.state = QueueState.FINALIZED

    def publish_private(self, wave: P8WaveIdentity) -> None:
        """Make successful H2D payloads visible to private QSFA reconciliation."""

        queue = self._validate_identity(wave, expected_state=QueueState.FINALIZED)
        if not queue.cq_success:
            raise P8LifecycleError(
                "failed CQ cannot publish private payload; use exact recall"
            )
        for candidate_slot in queue.candidate_slots:
            candidate = self._candidate(candidate_slot)
            private_slot = self._private_slot(candidate.private_slot)
            if private_slot.state is not PrivateSlotState.PENDING:
                raise P8LifecycleError(
                    f"private slot {private_slot.slot_id} cannot publish from "
                    f"{private_slot.state.name}"
                )
            try:
                self._validate_private_slot_identity(
                    candidate,
                    private_slot,
                    queue,
                    wave,
                )
            except P8LifecycleError as error:
                self._quarantine_queue(
                    queue,
                    f"private-slot identity failure: {error}",
                )
                raise P8QuarantineError(
                    "private slot identity mismatch prevents publication"
                ) from error
            private_slot.state = PrivateSlotState.READY
            candidate.state = CandidateState.CONSUMED
        queue.state = QueueState.PRIVATE_PUBLISHED

    def release_queue(self, wave: P8WaveIdentity) -> None:
        """Release a terminal queue only if its epoch/generation CAS matches.

        Any mismatch is intentionally escalated to quarantine: treating it as
        a harmless release failure could reuse descriptor memory while an old
        HCCP writer still owns it.
        """

        try:
            queue = self._validate_identity(
                wave,
                expected_state=(QueueState.FINALIZED, QueueState.PRIVATE_PUBLISHED),
            )
        except P8LifecycleError as error:
            queue = self._queue(wave.queue_id)
            self._quarantine_queue(queue, f"release CAS/identity failure: {error}")
            raise P8QuarantineError(
                f"queue {wave.queue_id} quarantined after release failure"
            ) from error
        if queue.state is QueueState.FINALIZED and queue.cq_success:
            self._quarantine_queue(
                queue,
                "successful CQ finalized without private READY publication",
            )
            raise P8QuarantineError(
                f"queue {queue.queue_id} quarantined: successful CQ not published"
            )
        queue.state = QueueState.RELEASED

    def timeout_queue(self, wave: P8WaveIdentity, *, reason: str) -> None:
        """Fail closed when a PostOnly wave cannot prove terminal CQ state.

        ``POST_INTENT`` is deliberately included.  Once descriptors have been
        prepared, a caller cannot prove whether a doorbell raced with an
        exception.  Releasing that queue would be an unsafe ambiguity, so it
        follows the same quarantine path as a timed-out posted wave.
        """

        queue = self._queue(wave.queue_id)
        try:
            self._validate_identity(
                wave,
                expected_state=(
                    QueueState.FILLING,
                    QueueState.POST_INTENT,
                    QueueState.POSTED,
                    QueueState.CQ_OBSERVED,
                ),
            )
        except P8LifecycleError as error:
            self._quarantine_queue(queue, f"timeout identity failure: {error}")
            raise P8QuarantineError(
                f"queue {wave.queue_id} quarantined after timeout identity failure"
            ) from error
        self._quarantine_queue(queue, f"timeout: {reason}")
        raise P8QuarantineError(
            f"queue {wave.queue_id} quarantined after timeout: {reason}"
        )

    def seal_admission(self) -> None:
        """Close producers; already published candidates remain drainable."""

        self._transition(ForwardState.ADMITTING, ForwardState.SEALED)

    def begin_draining(self) -> None:
        """Drain all published/posted work without accepting new seeds."""

        self._transition(ForwardState.SEALED, ForwardState.DRAINING)

    def begin_reconcile(self) -> None:
        """Enter exact Top-K reconcile only after every speculative wave ends."""

        self._require_state(ForwardState.DRAINING)
        if not self._all_candidates_terminal():
            raise P8LifecycleError("cannot reconcile while candidates are pending")
        if not self._all_queues_released():
            raise P8LifecycleError("cannot reconcile while queue waves are not released")
        self.state = ForwardState.RECONCILING

    def begin_qsfa_reading(self) -> None:
        """Declare that QSFA may consume the reconciled private view."""

        self._transition(ForwardState.RECONCILING, ForwardState.QSFA_READING)

    def mark_qsfa_terminal(self, layer_id: int) -> None:
        """Record one same-stream QSFA last-reader terminal operation."""

        self._require_state(ForwardState.QSFA_READING)
        if not 0 <= layer_id < self.expected_layers:
            raise ValueError(
                f"layer_id {layer_id} is outside [0,{self.expected_layers})"
            )
        self.qsfa_terminal_layers.add(layer_id)

    def mark_model_terminal_enqueued(self) -> None:
        """Record an event after all graph-side QSFA terminal ops were queued."""

        self._require_state(ForwardState.QSFA_READING)
        if len(self.qsfa_terminal_layers) != self.expected_layers:
            raise P8LifecycleError(
                "model terminal cannot be enqueued before every layer's "
                "QSFA last-reader terminal"
            )
        self.model_terminal_enqueued = True

    def release(self) -> None:
        """Release host leases and private slots after the forward terminal event."""

        self._require_state(ForwardState.QSFA_READING)
        if not self.model_terminal_enqueued:
            raise P8LifecycleError(
                "cannot release P8 transaction before model terminal is enqueued"
            )
        if len(self.qsfa_terminal_layers) != self.expected_layers:
            raise P8LifecycleError("cannot release before every QSFA reader is terminal")
        if not self._all_queues_released():
            raise P8LifecycleError("cannot release while a queue is still active")
        for private_slot in self.resources.private_slots:
            if private_slot.state in {PrivateSlotState.PENDING, PrivateSlotState.READY, PrivateSlotState.FAILED}:
                self.resources.release_global_private_slot(private_slot)
        self.host_leases.clear()
        self.state = ForwardState.RELEASED
        self.resources._detach(self)

    def abort(self, reason: str) -> None:
        """Enter fail-closed abort; callers must then drain or quarantine waves."""

        self._require_not_terminal()
        if not reason:
            raise ValueError("abort reason must be non-empty")
        self.abort_reason = reason
        self.state = ForwardState.ABORTING

    def mark_fallback_ready(self) -> None:
        """Allow an exact fallback after all unsafe P8 records are quarantined.

        Host leases deliberately remain retained.  A future runtime teardown
        owns their recovery, because this method cannot prove that a late
        device writer has stopped.
        """

        self._require_state(ForwardState.ABORTING)
        active = [
            queue.queue_id
            for queue in self.resources.queues
            if queue.state not in {QueueState.FREE, QueueState.RELEASED, QueueState.QUARANTINED}
        ]
        if active:
            raise P8LifecycleError(
                "cannot enter fallback while queues are neither released nor "
                f"quarantined: {active}"
            )
        unresolved_candidates = [
            candidate.slot_id
            for candidate in self.resources.candidates
            if candidate.state
            not in {
                CandidateState.EMPTY,
                CandidateState.CONSUMED,
                CandidateState.DEFERRED,
            }
        ]
        unresolved_private = [
            private_slot.slot_id
            for private_slot in self.resources.private_slots
            if private_slot.state
            not in {PrivateSlotState.FREE, PrivateSlotState.QUARANTINED}
        ]
        if unresolved_candidates or unresolved_private:
            raise P8LifecycleError(
                "cannot enter fallback while candidate/private ownership is "
                "unresolved: "
                f"candidates={unresolved_candidates} private_slots="
                f"{unresolved_private}"
            )
        self.state = ForwardState.FALLBACK_READY
        self.resources._detach(self)

    def _transition(self, expected: ForwardState, target: ForwardState) -> None:
        self._require_state(expected)
        self.state = target

    def _require_state(self, *allowed: ForwardState) -> None:
        if self.state not in allowed:
            wanted = ", ".join(item.name for item in allowed)
            raise P8LifecycleError(
                f"P8 txn state {self.state.name} is not one of [{wanted}]"
            )

    def _require_not_terminal(self) -> None:
        self._require_state(
            ForwardState.OPEN,
            ForwardState.ADMITTING,
            ForwardState.SEALED,
            ForwardState.DRAINING,
            ForwardState.RECONCILING,
            ForwardState.QSFA_READING,
            ForwardState.ABORTING,
        )

    def _candidate(self, candidate_slot: int) -> CandidateRecord:
        try:
            return self.resources.candidates[candidate_slot]
        except IndexError as error:
            raise ValueError(f"candidate slot {candidate_slot} is out of range") from error

    def _queue(self, queue_id: int) -> P8WaveRecord:
        try:
            return self.resources.queues[queue_id]
        except IndexError as error:
            raise ValueError(f"queue {queue_id} is out of range") from error

    def _private_slot(self, private_slot: int | None) -> PrivateSlotRecord:
        if private_slot is None:
            raise P8LifecycleError("candidate has no private slot")
        try:
            return self.resources.private_slots[private_slot]
        except IndexError as error:
            raise ValueError(f"private slot {private_slot} is out of range") from error

    def _require_candidate_identity(self, candidate: CandidateRecord) -> None:
        if (
            candidate.epoch != self.epoch
            or candidate.graph_generation != self.graph_generation
        ):
            raise P8LifecycleError(
                f"candidate {candidate.slot_id} identity mismatch: "
                f"epoch={candidate.epoch}, graph={candidate.graph_generation}"
            )

    def _validate_layer_control_index(self, layer_control_index: int) -> None:
        """Reject a layer index outside this forward's static graph shape."""

        if not 0 <= layer_control_index < self.expected_layers:
            raise ValueError(
                "layer_control_index "
                f"{layer_control_index} is outside [0,{self.expected_layers})"
            )

    def _validate_identity(
        self,
        wave: P8WaveIdentity,
        *,
        expected_state: QueueState | tuple[QueueState, ...],
    ) -> P8WaveRecord:
        queue = self._queue(wave.queue_id)
        states = (
            expected_state
            if isinstance(expected_state, tuple)
            else (expected_state,)
        )
        if queue.state not in states:
            wanted = ", ".join(state.name for state in states)
            raise P8LifecycleError(
                f"queue {queue.queue_id} state {queue.state.name} is not one of [{wanted}]"
            )
        if (
            wave.epoch != self.epoch
            or wave.graph_generation != self.graph_generation
            or wave.graph_generation != queue.graph_generation
            or wave.layer_control_index != queue.layer_control_index
            or wave.generation != queue.generation
            or wave.owner != queue.owner
            or wave.post_sequence != queue.post_sequence
        ):
            raise P8LifecycleError(
                "wave identity mismatch: "
                f"queue={queue.queue_id} expected=(epoch={self.epoch}, "
                f"graph={queue.graph_generation}, layer="
                f"{queue.layer_control_index}, generation={queue.generation}, "
                f"post={queue.post_sequence}, owner={queue.owner}) "
                f"got=(epoch={wave.epoch}, graph={wave.graph_generation}, "
                f"layer={wave.layer_control_index}, generation={wave.generation}, "
                f"post={wave.post_sequence}, owner={wave.owner})"
            )
        return queue

    def _free_candidate_private_slot(
        self,
        candidate: CandidateRecord,
        *,
        failed: bool = False,
    ) -> None:
        if candidate.private_slot is None:
            return
        private_slot = self._private_slot(candidate.private_slot)
        if private_slot.state is PrivateSlotState.PENDING:
            if failed:
                private_slot.state = PrivateSlotState.FAILED
                private_slot.queue_id = None
                private_slot.queue_generation = 0
                private_slot.post_sequence = 0
            else:
                self.resources.release_global_private_slot(private_slot)
        candidate.private_slot = None
        candidate.private_allocation_generation = 0

    def _validate_private_slot_identity(
        self,
        candidate: CandidateRecord,
        private_slot: PrivateSlotRecord,
        queue: P8WaveRecord,
        wave: P8WaveIdentity,
    ) -> None:
        """Prove a private payload belongs to this exact CQ wave.

        A graph replay may use the same physical queue and staging slot.  The
        tuple below prevents a late prior completion from becoming visible to
        the new forward merely because its slot number happens to match.
        """

        if (
            private_slot.epoch != self.epoch
            or private_slot.graph_generation != self.graph_generation
            or private_slot.layer_control_index != queue.layer_control_index
            or private_slot.owner != wave.owner
            or candidate.owner != wave.owner
            or candidate.private_slot != private_slot.slot_id
            or candidate.private_allocation_generation
            != private_slot.allocation_generation
            or private_slot.queue_id != queue.queue_id
            or private_slot.queue_generation != queue.generation
            or private_slot.post_sequence != wave.post_sequence
        ):
            raise P8LifecycleError(
                "private slot identity mismatch: "
                f"slot={private_slot.slot_id} epoch={private_slot.epoch} "
                f"graph={private_slot.graph_generation} layer="
                f"{private_slot.layer_control_index} queue={private_slot.queue_id} "
                f"queue_generation={private_slot.queue_generation} post="
                f"{private_slot.post_sequence}"
            )

    def _quarantine_queue(self, queue: P8WaveRecord, reason: str) -> None:
        queue.state = QueueState.QUARANTINED
        queue.quarantine_reason = reason
        for candidate_slot in queue.candidate_slots:
            candidate = self._candidate(candidate_slot)
            candidate.state = CandidateState.DEFERRED
            if candidate.private_slot is not None:
                private_slot = self._private_slot(candidate.private_slot)
                private_slot.state = PrivateSlotState.QUARANTINED
        self.abort_reason = reason
        self.state = ForwardState.ABORTING

    def _all_candidates_terminal(self) -> bool:
        return all(
            candidate.state
            in {CandidateState.EMPTY, CandidateState.CONSUMED, CandidateState.DEFERRED}
            for candidate in self.resources.candidates
        )

    def _all_queues_released(self) -> bool:
        return all(
            queue.state in {QueueState.FREE, QueueState.RELEASED}
            for queue in self.resources.queues
        )


__all__ = [
    "CandidateRecord",
    "CandidateState",
    "ForwardState",
    "P8FixedResources",
    "P8ForwardAsyncTxn",
    "P8LifecycleError",
    "P8QuarantineError",
    "P8WaveIdentity",
    "P8WaveRecord",
    "PrivateSlotRecord",
    "PrivateSlotState",
    "QueueState",
]
