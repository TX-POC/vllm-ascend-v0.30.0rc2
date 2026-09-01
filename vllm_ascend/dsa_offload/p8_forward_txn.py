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
  reconcile, QSFA reading, forward-terminal sweep, and release.

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
    FORWARD_TERMINAL = auto()
    RELEASED = auto()
    ABORTING = auto()
    FALLBACK_READY = auto()


class ForwardPhase(Enum):
    """Graph-visible control phase carried by the fixed P8 control row."""

    ADMIT_POST = auto()
    DRAIN = auto()
    QSFA_TERMINAL = auto()
    FORWARD_TERMINAL = auto()


class ArenaLeaseState(Enum):
    """The one forward-owned lease over P8's shared fixed arena.

    This deliberately stays smaller than a forward identity.  Strict P8 does
    not allow a queue/private arena to cross a forward boundary: all posted
    work must reach CQ terminal (or quarantine), and every private reader
    must finish, before the arena can become reusable.  ``generation`` is
    therefore only a defensive lease token that makes a delayed terminal
    harmless if a future bug were to violate that rule.
    """

    IDLE = auto()
    LEASED = auto()
    # Exactly one terminal worker owns the final LEASED -> IDLE decision.
    # It is a short state-machine edge, not a new data buffer or scheduler.
    TERMINAL_CLAIMED = auto()
    QUARANTINED = auto()


class QueueState(Enum):
    """State of one fixed Direct-HCCP queue/wave record."""

    FREE = auto()
    # Queue owner-table acquisition succeeded, but a durable full wave
    # identity has not been published yet.  This non-consumable edge prevents
    # a terminal sweep from mistaking an interrupted admission for FREE.
    FILLING_CLAIMED = auto()
    FILLING = auto()
    # A proved-empty FILLING wave has one owner that is performing the final
    # owner-table release.  ForwardTerminal must quarantine this transient
    # state if it races the second CAS; it may never guess that the release
    # completed just because the original worker returned.
    FILLING_RELEASE_CLAIMED = auto()
    POST_INTENT = auto()
    POSTED = auto()
    DRAIN_CLAIMED = auto()
    FINALIZED = auto()
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
    # The physical row is exclusive and carries an owner identity, but its
    # source/object maps are not yet visible.  It is never QSFA-readable.
    ALLOCATING = auto()
    PENDING = auto()
    READY = auto()
    FAILED = auto()
    QUARANTINED = auto()
    RELEASE_CLAIMED = auto()


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
    # These are the source/object-directory positions that point to the
    # private output row.  They are installed while the row is PENDING and
    # cleared by exact CAS only after QSFA's terminal reader.
    source_offset: int | None = None
    object_offset: int | None = None
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
    source_offset: int | None = None
    object_offset: int | None = None
    output_index: int = -1


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
    private_ready_published: bool = False
    # A descriptor can leave POST_INTENT through exactly one durable outcome:
    # either the doorbell was committed, or PostOnly proved that it was not.
    # ``post_rejected`` is deliberately not inferred from ``cq_success``:
    # a CQ result proves a committed post, while a rejected post has no CQ at
    # all.  Keeping both facts makes the only safe early-cleanup path explicit.
    post_committed: bool = False
    post_rejected: bool = False
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
    # The native executor owns two graph-stable int32 directories.  A model
    # dictionary preserves the same crucial compare-and-swap contract without
    # pretending that a Python object has device-memory concurrency.
    private_source_to_slot: dict[int, int] = field(init=False, default_factory=dict)
    private_object_to_slot: dict[int, int] = field(init=False, default_factory=dict)
    private_ready: list[PrivateSlotState] = field(init=False)
    # Forward-owned host/native handles are transferred here when a forward
    # takes the quarantine/fallback exit.  Runtime teardown, not a detached
    # transaction object, is then the explicit owner responsible for proving
    # that it is safe to drop them.
    quarantine_keepalives: set[str] = field(init=False, default_factory=set)
    # P8 has one fixed arena for queues, candidate records and private slots.
    # Under the strict no-cross-forward contract it needs only one atomically
    # comparable lease token: ``(state, generation)``.  Do not turn
    # epoch/replay/layer into arena ownership fields: those remain useful
    # diagnostics on individual wave/private records, but do not make the
    # arena safer than terminal-before-reuse already does.
    _arena_lease_state: ArenaLeaseState = field(
        init=False,
        default=ArenaLeaseState.IDLE,
        repr=False,
    )
    _arena_lease_generation: int = field(init=False, default=0, repr=False)
    _arena_fault_reason: str | None = field(init=False, default=None, repr=False)
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
        self.private_ready = [
            PrivateSlotState.FREE for _ in range(self.private_slot_capacity)
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
            # Model the native `FREE -> ALLOCATING` reservation.  The record
            # exists before either directory gets its short `-2` claim, so a
            # forward-terminal sweep always has an owner to quarantine.
            private_slot.state = PrivateSlotState.ALLOCATING
            private_slot.epoch = epoch
            private_slot.graph_generation = graph_generation
            private_slot.layer_control_index = layer_control_index
            private_slot.owner = owner
            private_slot.allocation_generation += 1
            private_slot.queue_id = None
            private_slot.queue_generation = 0
            private_slot.post_sequence = 0
            private_slot.source_offset = None
            private_slot.object_offset = None
            private_slot.output_index = private_slot.slot_id
            self.private_ready[slot_id] = PrivateSlotState.FREE
            self._next_private_slot_hint = (slot_id + 1) % self.private_slot_capacity
            return private_slot
        return None

    def _commit_private_free(self, private_slot: PrivateSlotRecord) -> None:
        """Commit an already-proven-safe private row back to ``FREE``.

        This is deliberately private to the fixed-resource owner.  Normal
        callers must first pass through either the unposted rollback or the
        QSFA-terminal ``RELEASE_CLAIMED -> map CAS clear -> FREE`` protocol.
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
        private_slot.source_offset = None
        private_slot.object_offset = None
        private_slot.output_index = -1
        self.private_ready[private_slot.slot_id] = PrivateSlotState.FREE

    def adopt_quarantine_keepalives(self, leases: set[str]) -> None:
        """Transfer forward-owned keepalives to explicit teardown ownership."""

        self.quarantine_keepalives.update(leases)
        leases.clear()

    @property
    def arena_lease_state(self) -> ArenaLeaseState:
        """Current shared-arena state, exposed for lifecycle diagnostics."""

        return self._arena_lease_state

    @property
    def arena_lease_generation(self) -> int:
        """Monotonic token of the currently held or last released lease."""

        return self._arena_lease_generation

    @property
    def arena_healthy(self) -> bool:
        """Compatibility view: only a quarantined arena is unhealthy."""

        return self._arena_lease_state is not ArenaLeaseState.QUARANTINED

    @property
    def arena_fault_reason(self) -> str | None:
        """First sticky quarantine reason, retained for runtime teardown."""

        return self._arena_fault_reason

    def _acquire_arena_lease(self) -> int:
        """Model the one atomic ``IDLE(g) -> LEASED(g + 1)`` transition.

        The caller has already proved every fixed queue/private row clean.
        This token is intentionally unrelated to forward epoch/replay/layer:
        it identifies only this physical-arena acquisition.
        """

        if self._arena_lease_state is ArenaLeaseState.QUARANTINED:
            raise P8QuarantineError(
                "P8 fixed arena is unhealthy/quarantined; recreate runtime "
                f"resources before reuse: {self._arena_fault_reason}"
            )
        if self._arena_lease_state is not ArenaLeaseState.IDLE:
            raise P8LifecycleError("P8 fixed arena is still leased by a prior forward")
        self._arena_lease_generation += 1
        if self._arena_lease_generation <= 0:
            raise P8LifecycleError("P8 arena lease generation must be nonzero")
        self._arena_lease_state = ArenaLeaseState.LEASED
        return self._arena_lease_generation

    def claim_arena_terminal(self, lease_generation: int) -> bool:
        """Conditionally perform ``LEASED(g) -> TERMINAL_CLAIMED(g)``.

        The normal transaction invokes this only after its forward-terminal
        checks prove that no Post/CQ/private reader remains.  A second or
        stale terminal loses this one-owner claim and must not touch the
        current arena lease.
        """

        if (
            self._arena_lease_state is not ArenaLeaseState.LEASED
            or self._arena_lease_generation != lease_generation
        ):
            return False
        self._arena_lease_state = ArenaLeaseState.TERMINAL_CLAIMED
        return True

    def complete_arena_terminal(self, lease_generation: int) -> bool:
        """Conditionally perform ``TERMINAL_CLAIMED(g) -> IDLE(g)``.

        Keeping this second, exact transition makes the terminal leader
        explicit.  It prevents a duplicate terminal path from silently
        releasing a lease after a different worker already owns the terminal
        protocol.
        """

        if (
            self._arena_lease_state is not ArenaLeaseState.TERMINAL_CLAIMED
            or self._arena_lease_generation != lease_generation
        ):
            return False
        self._arena_lease_state = ArenaLeaseState.IDLE
        return True

    def quarantine_arena_lease(self, lease_generation: int, reason: str) -> bool:
        """Conditionally perform ``LEASED(g) -> QUARANTINED(g)``.

        A delayed fault from an older lease returns ``False`` instead of
        poisoning a newly acquired forward.  This is a defense-in-depth
        guard; the primary rule remains terminal/quarantine before reuse.
        """

        if not reason:
            raise ValueError("arena unhealthy reason must be non-empty")
        if (
            self._arena_lease_state
            not in {ArenaLeaseState.LEASED, ArenaLeaseState.TERMINAL_CLAIMED}
            or self._arena_lease_generation != lease_generation
        ):
            return False
        self._arena_lease_state = ArenaLeaseState.QUARANTINED
        if self._arena_fault_reason is None:
            self._arena_fault_reason = reason
        return True

    def mark_arena_unhealthy(
        self,
        reason: str,
        *,
        lease_generation: int | None = None,
    ) -> bool:
        """Latch the *matching* current arena lease unavailable.

        The optional argument preserves a narrow compatibility path for
        model-internal callers, but it never falls back to a non-atomic
        "mark whatever is current unhealthy" operation.  If no active
        transaction owns a lease, there is no safe arena target to mutate.
        """

        if lease_generation is None:
            if self.active_txn is None:
                return False
            lease_generation = self.active_txn.arena_lease_generation
        return self.quarantine_arena_lease(lease_generation, reason)

    @property
    def has_quarantine(self) -> bool:
        """Whether any fixed resource is unsafe to reuse."""

        return (self._arena_lease_state is ArenaLeaseState.QUARANTINED) or any(
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
        if self._arena_lease_state is ArenaLeaseState.QUARANTINED:
            raise P8QuarantineError(
                "P8 fixed arena is unhealthy/quarantined; recreate runtime "
                f"resources before reuse: {self.arena_fault_reason}"
            )
        if self._arena_lease_state is not ArenaLeaseState.IDLE:
            raise P8LifecycleError("previous P8 arena lease is still active")
        if self.has_quarantine:
            raise P8QuarantineError(
                "P8 fixed resources contain quarantined queue/private slots; "
                "route this forward to the exact fallback until teardown"
            )
        self._reset_clean_records()
        arena_lease_generation = self._acquire_arena_lease()
        txn = P8ForwardAsyncTxn(
            resources=self,
            epoch=epoch,
            graph_generation=graph_generation,
            expected_layers=expected_layers,
            arena_lease_generation=arena_lease_generation,
        )
        self.active_txn = txn
        return txn

    def _reset_clean_records(self) -> None:
        """Reuse only records that were terminally released by a prior txn."""

        for queue in self.queues:
            if queue.state is not QueueState.FREE:
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
            queue.private_ready_published = False
            queue.post_committed = False
            queue.post_rejected = False
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
            candidate.source_offset = None
            candidate.object_offset = None

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
            private_slot.source_offset = None
            private_slot.object_offset = None
            private_slot.output_index = -1
            self.private_ready[private_slot.slot_id] = PrivateSlotState.FREE

        if self.private_source_to_slot or self.private_object_to_slot:
            raise P8LifecycleError(
                "cannot reset P8 resources while private directories still "
                "point to a prior forward"
            )

    def _detach(self, txn: P8ForwardAsyncTxn) -> None:
        if self.active_txn is not txn:
            raise P8LifecycleError("transaction is not this resource owner's active txn")
        if (
            self._arena_lease_state
            in {ArenaLeaseState.LEASED, ArenaLeaseState.TERMINAL_CLAIMED}
            and self._arena_lease_generation == txn.arena_lease_generation
        ):
            raise P8LifecycleError(
                "cannot detach forward while its arena lease is still active"
            )
        self.active_txn = None


@dataclass
class P8ForwardAsyncTxn:
    """One forward-owned asynchronous Get lifecycle.

    A transaction never changes the persistent resident mapping.  It owns
    only speculative private slots and host keepalive leases.  The caller may
    release those leases only after all queue waves are terminal, all required
    QSFA readers have passed, the model-terminal operation has been observed,
    and the explicit forward-terminal sweep found no unresolved resource.

    P8 uses one sticky abort domain for a whole forward even though each
    native graph invocation carries a layer-local control row.  When any
    layer cannot prove queue/private ownership, it latches the shared arena
    unhealthy and disables speculative admission for the remaining layers;
    forward terminal then quarantines every unresolved record with the same
    ``(epoch, graph_generation)``.  This deliberately conservative rule
    prevents a layer-local failure from being hidden by another layer leasing
    the same global queue/private arena.
    """

    resources: P8FixedResources
    epoch: int
    graph_generation: int
    expected_layers: int
    # The only arena-ownership identity.  Forward epoch/replay/layer remain
    # wave/private diagnostics, not conditions for arena reuse.
    arena_lease_generation: int
    state: ForwardState = ForwardState.OPEN
    phase: ForwardPhase = ForwardPhase.ADMIT_POST
    host_leases: set[str] = field(default_factory=set)
    exact_winner_layers: set[int] = field(default_factory=set)
    guaranteed_recall_layers: set[int] = field(default_factory=set)
    qsfa_terminal_layers: set[int] = field(default_factory=set)
    model_terminal_enqueued: bool = False
    model_terminal_observed: bool = False
    exact_input_ready: bool = False
    qsfa_terminal_observed: bool = False
    abort_latched: bool = False
    forward_terminal_swept: bool = False
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
        source_offset: int | None = None,
        object_offset: int | None = None,
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
        # The production CandidateSeed contains both offsets.  Keeping the
        # default deterministic lets small lifecycle tests omit irrelevant
        # score metadata while still exercising two independent directories.
        candidate.source_offset = (
            candidate_slot if source_offset is None else source_offset
        )
        candidate.object_offset = (
            candidate_slot if object_offset is None else object_offset
        )
        if candidate.source_offset < 0 or candidate.object_offset < 0:
            raise ValueError("CandidateSeed source/object offsets must be non-negative")

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
            if candidate.source_offset is None or candidate.object_offset is None:
                raise P8LifecycleError(
                    f"candidate {candidate_slot} has no source/object offsets"
                )
            private_slot.source_offset = candidate.source_offset
            private_slot.object_offset = candidate.object_offset
            self._publish_pending_private_mappings(private_slot)
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
        if not self.resources.arena_healthy:
            raise P8QuarantineError(
                "P8 fixed arena is unhealthy; speculative admission is closed "
                "until runtime teardown recreates it"
            )
        if (
            self.resources.arena_lease_state is not ArenaLeaseState.LEASED
            or self.resources.arena_lease_generation
            != self.arena_lease_generation
        ):
            raise P8LifecycleError(
                "P8 forward no longer owns the shared arena lease"
            )
        queue = self._queue(queue_id)
        if queue.state is QueueState.QUARANTINED:
            raise P8QuarantineError(f"queue {queue_id} is quarantined")
        if queue.state is not QueueState.FREE:
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
        queue.private_ready_published = False
        queue.post_committed = False
        queue.post_rejected = False
        queue.quarantine_reason = None
        # Reserve a nonzero post sequence *before* descriptor/private-record
        # binding.  PostOnly consumes this pre-reserved identity; it must not
        # manufacture a new sequence after descriptor publication.
        queue.post_sequence += 1
        if queue.post_sequence <= 0:
            raise P8LifecycleError("queue post sequence must be nonzero")
        # The native record first reaches a non-consumable claimed state;
        # identity/payload publication then commits it to FILLING.  Keeping
        # this separate lets tests model a terminal race in that gap.
        queue.state = QueueState.FILLING_CLAIMED
        return P8WaveIdentity(
            queue_id=queue_id,
            epoch=self.epoch,
            graph_generation=self.graph_generation,
            layer_control_index=layer_control_index,
            generation=queue.generation,
            # ``post_sequence`` is monotonic for the physical queue and is
            # reserved before private-record binding, not after the doorbell.
            post_sequence=queue.post_sequence,
            owner=owner,
        )

    def _publish_filling_identity(self, lease: P8WaveIdentity) -> P8WaveRecord:
        """Commit ``FILLING_CLAIMED -> FILLING`` after identity publication.

        In native code the identity fields are written and DDR-published
        before the expected-state CAS.  A ForwardTerminal winner remains
        quarantined; this helper never writes a state back over it.
        """

        queue = self._validate_identity(
            lease, expected_state=QueueState.FILLING_CLAIMED
        )
        queue.state = QueueState.FILLING
        return queue

    def claim_unposted_filling_release(self, lease: P8WaveIdentity) -> None:
        """CAS-claim a proved-empty ``FILLING`` wave for its final release.

        ``FILLING`` is a durable ownership record immediately after queue
        acquisition.  Its owner cannot jump straight to ``FREE``: the native
        implementation first CASes to ``FILLING_RELEASE_CLAIMED`` so a racing
        ForwardTerminal can turn the lease into ``QUARANTINED`` instead of
        allowing a late full-record write to resurrect it.

        The split is public in this small model solely to test the real
        terminal race.  Production P8 performs both CAS operations in the
        same dynamic task; it does not introduce a host callback.
        """

        try:
            # The explicit test-facing release claim starts after the same
            # identity publication that a native no-work task performs.  It
            # is still impossible to jump directly from FILLING_CLAIMED to
            # FREE, and a forged identity is quarantined by this same guard.
            if self._queue(lease.queue_id).state is QueueState.FILLING_CLAIMED:
                self._publish_filling_identity(lease)
            queue = self._validate_identity(
                lease,
                expected_state=QueueState.FILLING,
            )
            if (
                queue.candidate_slots
                or queue.private_slots
                or queue.post_committed
                or queue.post_rejected
                or queue.cq_success is not None
            ):
                raise P8LifecycleError(
                    "unposted lease has descriptor/CQ state and cannot be "
                    "safely released"
                )
        except P8LifecycleError as error:
            queue = self._queue(lease.queue_id)
            self._quarantine_queue(
                queue,
                f"unposted lease release proof/identity failure: {error}",
            )
            raise P8QuarantineError(
                f"queue {lease.queue_id} quarantined after unposted lease release failure"
            ) from error
        queue.state = QueueState.FILLING_RELEASE_CLAIMED

    def finalize_unposted_filling_release(self, lease: P8WaveIdentity) -> None:
        """Commit the release-owner CAS only after exact identity revalidation."""

        try:
            queue = self._validate_identity(
                lease,
                expected_state=QueueState.FILLING_RELEASE_CLAIMED,
            )
            if (
                queue.candidate_slots
                or queue.private_slots
                or queue.post_committed
                or queue.post_rejected
                or queue.cq_success is not None
            ):
                raise P8LifecycleError(
                    "filling-release claimant has descriptor/CQ state"
                )
        except P8LifecycleError as error:
            queue = self._queue(lease.queue_id)
            self._quarantine_queue(
                queue,
                f"filling release final-CAS/identity failure: {error}",
            )
            raise P8QuarantineError(
                f"queue {lease.queue_id} quarantined after filling release final-CAS failure"
            ) from error
        queue.state = QueueState.FREE

    def release_unposted_lease(self, lease: P8WaveIdentity) -> None:
        """Return a lease only while it is provably before descriptor binding.

        This convenience endpoint is the ordinary no-work path.  It preserves
        both native state-machine edges (``FILLING`` →
        ``FILLING_RELEASE_CLAIMED`` → ``FREE``), rather than modelling them
        as a misleading direct assignment.

        This is the model counterpart of native P8's pre-doorbell release
        CAS.  It prevents an error path from silently dropping a queue lease
        merely because a PostOnly call was never reached.
        """

        self.claim_unposted_filling_release(lease)
        self.finalize_unposted_filling_release(lease)

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
        if self._queue(lease.queue_id).state is QueueState.FILLING_CLAIMED:
            self._publish_filling_identity(lease)
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
            private_record.post_sequence = lease.post_sequence
        queue.state = QueueState.POST_INTENT

    def post_only(self, lease: P8WaveIdentity) -> P8WaveIdentity:
        """Record the non-blocking doorbell transition and return CQ identity."""

        queue = self._validate_wave_or_quarantine(
            lease,
            expected_state=QueueState.POST_INTENT,
            operation="PostOnly",
        )
        if queue.post_sequence <= 0:
            raise P8LifecycleError("PostOnly requires a reserved post sequence")
        queue.post_committed = True
        queue.post_rejected = False
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

    def record_post_rejected(self, lease: P8WaveIdentity) -> P8WaveIdentity:
        """Record a deterministic PostOnly rejection with no doorbell.

        This is intentionally the *only* exception to the normal rule that a
        queue-bound PENDING private row cannot be released before CQ.  The
        native PostOnly implementation reaches this result only after it has
        proved that the descriptor was rejected before a doorbell could be
        issued.  ``POST_INTENT`` itself is still ambiguous and must use the
        timeout/quarantine path.
        """

        queue = self._validate_wave_or_quarantine(
            lease,
            expected_state=QueueState.POST_INTENT,
            operation="POST_REJECTED recording",
        )
        if queue.post_sequence <= 0:
            raise P8LifecycleError("POST_REJECTED requires a reserved post sequence")
        queue.post_committed = False
        queue.post_rejected = True
        # No CQ exists for this outcome.  FINALIZED stores the durable
        # terminal outcome until a dynamic DRAIN worker claims exact cleanup.
        queue.cq_success = None
        queue.state = QueueState.FINALIZED
        return P8WaveIdentity(
            queue_id=queue.queue_id,
            epoch=self.epoch,
            graph_generation=self.graph_generation,
            layer_control_index=queue.layer_control_index,
            generation=queue.generation,
            post_sequence=queue.post_sequence,
            owner=lease.owner,
        )

    def try_claim_drain(self, wave: P8WaveIdentity) -> bool:
        """Dynamically claim one durable POSTED wave without spinning.

        This models the AIV CAS ``POSTED -> DRAIN_CLAIMED``.  A losing worker
        simply returns ``False`` and continues score work; it must not probe,
        finish or release a wave claimed by another worker.
        """

        queue = self._validate_wave_or_quarantine(
            wave,
            expected_state=(QueueState.POSTED, QueueState.DRAIN_CLAIMED),
            operation="DRAIN claim",
        )
        if not queue.post_committed or queue.post_rejected:
            self._quarantine_queue(
                queue,
                "POSTED wave lacks an unambiguous committed-doorbell proof",
            )
            raise P8QuarantineError(
                f"queue {queue.queue_id} quarantined after post-proof mismatch"
            )
        if queue.state is QueueState.DRAIN_CLAIMED:
            return False
        queue.state = QueueState.DRAIN_CLAIMED
        return True

    def try_claim_post_rejected_drain(self, wave: P8WaveIdentity) -> bool:
        """CAS-claim a proved-no-doorbell wave for PENDING-private cleanup.

        ``FINALIZED -> DRAIN_CLAIMED`` is a distinct dynamic DRAIN claim from
        ``POSTED -> DRAIN_CLAIMED``: it never polls CQ.  It is legal only if
        the durable record proves that no doorbell was committed.  Any stale
        identity or contradictory record is fail-closed rather than guessed
        free.
        """

        try:
            queue = self._validate_identity(
                wave,
                expected_state=(QueueState.FINALIZED, QueueState.DRAIN_CLAIMED),
            )
            if (
                not queue.post_rejected
                or queue.post_committed
                or queue.cq_success is not None
                or queue.private_ready_published
            ):
                raise P8LifecycleError(
                    "POST_REJECTED no-doorbell proof is missing or contradictory"
                )
        except P8LifecycleError as error:
            queue = self._queue(wave.queue_id)
            self._quarantine_queue(
                queue,
                f"POST_REJECTED drain proof/identity failure: {error}",
            )
            raise P8QuarantineError(
                f"queue {wave.queue_id} quarantined after POST_REJECTED proof mismatch"
            ) from error
        if queue.state is QueueState.DRAIN_CLAIMED:
            return False
        queue.state = QueueState.DRAIN_CLAIMED
        return True

    def leave_drain_pending(self, wave: P8WaveIdentity) -> None:
        """Return a claimed wave to POSTED when a nonblocking CQ poll is empty."""

        queue = self._validate_wave_or_quarantine(
            wave,
            expected_state=QueueState.DRAIN_CLAIMED,
            operation="DRAIN pending return",
        )
        if queue.cq_success is not None:
            raise P8LifecycleError(
                "cannot return a CQ-terminal wave to POSTED"
            )
        if not queue.post_committed or queue.post_rejected:
            self._quarantine_queue(
                queue,
                "only a committed doorbell may return from DRAIN to POSTED",
            )
            raise P8QuarantineError(
                f"queue {queue.queue_id} quarantined after invalid DRAIN return"
            )
        queue.state = QueueState.POSTED

    def observe_cq(self, wave: P8WaveIdentity, *, succeeded: bool) -> None:
        """Observe one terminal CQE without publishing the private view yet."""

        try:
            queue = self._validate_identity(
                wave,
                expected_state=QueueState.DRAIN_CLAIMED,
            )
            if not queue.post_committed or queue.post_rejected:
                raise P8LifecycleError(
                    "CQ observation requires a committed PostOnly doorbell"
                )
        except P8LifecycleError as error:
            queue = self._queue(wave.queue_id)
            self._quarantine_queue(queue, f"CQ identity failure: {error}")
            raise P8QuarantineError(
                f"queue {wave.queue_id} quarantined after CQ identity failure"
            ) from error
        queue.cq_success = bool(succeeded)

    def finalize_wave(self, wave: P8WaveIdentity) -> None:
        """Finish CQ handling; success still needs explicit private publication."""

        queue = self._validate_wave_or_quarantine(
            wave,
            expected_state=QueueState.DRAIN_CLAIMED,
            operation="CQ finalization",
        )
        if queue.cq_success is None:
            raise P8LifecycleError("DRAIN_CLAIMED wave has no CQ terminal result")
        queue.state = QueueState.FINALIZED

        # ``PublishKvafP8PrivateTerminalRecords`` runs for both CQ success
        # and observed CQ error.  A failed row remains mapped and marked
        # FAILED until QSFA terminal clears both directories; exact recall,
        # not early reuse, supplies the mathematical fallback.
        if not queue.cq_success:
            self._publish_private_terminal_records(queue, payload_ready=False)

    def publish_private(self, wave: P8WaveIdentity) -> None:
        """Make successful H2D payloads visible to private QSFA reconciliation."""

        queue = self._validate_wave_or_quarantine(
            wave,
            expected_state=QueueState.FINALIZED,
            operation="private terminal publication",
        )
        if not queue.cq_success:
            raise P8LifecycleError(
                "failed CQ cannot publish private payload; use exact recall"
            )
        try:
            self._publish_private_terminal_records(queue, payload_ready=True)
        except P8LifecycleError as error:
            self._quarantine_queue(
                queue,
                f"private-slot identity failure: {error}",
            )
            raise P8QuarantineError(
                "private slot identity mismatch prevents publication"
            ) from error

    def release_queue(self, wave: P8WaveIdentity) -> None:
        """Release a terminal queue only if its epoch/generation CAS matches.

        Any mismatch is intentionally escalated to quarantine: treating it as
        a harmless release failure could reuse descriptor memory while an old
        HCCP writer still owns it.
        """

        try:
            queue = self._validate_identity(
                wave,
                expected_state=QueueState.FINALIZED,
            )
        except P8LifecycleError as error:
            queue = self._queue(wave.queue_id)
            self._quarantine_queue(queue, f"release CAS/identity failure: {error}")
            raise P8QuarantineError(
                f"queue {wave.queue_id} quarantined after release failure"
            ) from error
        if not queue.post_committed or queue.post_rejected:
            self._quarantine_queue(
                queue,
                "normal queue release received a non-committed PostOnly outcome",
            )
            raise P8QuarantineError(
                f"queue {queue.queue_id} quarantined: POST_REJECTED needs DRAIN cleanup"
            )
        if not queue.private_ready_published:
            self._quarantine_queue(
                queue,
                "CQ-finalized wave lacks private READY/FAILED publication",
            )
            raise P8QuarantineError(
                f"queue {queue.queue_id} quarantined: private terminal not published"
            )
        queue.state = QueueState.FREE

    def cleanup_post_rejected_wave(self, wave: P8WaveIdentity) -> None:
        """Return a proved-unposted wave after exact PENDING-map cleanup.

        The state sequence is ``FINALIZED -> DRAIN_CLAIMED -> FREE``.  Each
        private row follows ``PENDING -> map CAS clear -> FREE``; it never
        becomes QSFA-visible.  The first stale identity/mapping invalidates
        the no-post proof for safe reuse, so the remaining durable resources
        are quarantined and the forward aborts.
        """

        try:
            queue = self._validate_identity(
                wave,
                expected_state=QueueState.DRAIN_CLAIMED,
            )
            if (
                not queue.post_rejected
                or queue.post_committed
                or queue.cq_success is not None
                or queue.private_ready_published
            ):
                raise P8LifecycleError(
                    "POST_REJECTED cleanup lacks an exact no-doorbell proof"
                )
            if len(queue.candidate_slots) != len(queue.private_slots):
                raise P8LifecycleError(
                    "POST_REJECTED wave candidate/private cardinality mismatch"
                )
            for candidate_slot, private_slot_id in zip(
                queue.candidate_slots,
                queue.private_slots,
            ):
                candidate = self._candidate(candidate_slot)
                private_slot = self._private_slot(private_slot_id)
                self._require_candidate_identity(candidate)
                self._validate_private_slot_identity(
                    candidate,
                    private_slot,
                    queue,
                    wave,
                )
                if candidate.state is not CandidateState.CLAIMED:
                    raise P8LifecycleError(
                        f"POST_REJECTED candidate {candidate.slot_id} is "
                        f"{candidate.state.name}, not CLAIMED"
                    )
                if private_slot.state is not PrivateSlotState.PENDING:
                    raise P8LifecycleError(
                        f"POST_REJECTED private slot {private_slot.slot_id} is "
                        f"{private_slot.state.name}, not PENDING"
                    )
                if (
                    self.resources.private_ready[private_slot.slot_id]
                    is not PrivateSlotState.PENDING
                ):
                    raise P8LifecycleError(
                        "POST_REJECTED private slot "
                        f"{private_slot.slot_id} readiness mismatch"
                    )
                if not self._clear_private_mappings_exact(private_slot):
                    raise P8LifecycleError(
                        f"POST_REJECTED private slot {private_slot.slot_id} "
                        "mapping-clear mismatch"
                    )
                # The map clear is the ownership handoff.  Only after it
                # succeeds is it safe to return the absolute private slot.
                self.resources._commit_private_free(private_slot)
                candidate.private_slot = None
                candidate.private_allocation_generation = 0
                candidate.state = CandidateState.DEFERRED
            queue.state = QueueState.FREE
        except P8LifecycleError as error:
            queue = self._queue(wave.queue_id)
            self._quarantine_queue(
                queue,
                f"POST_REJECTED pending-private cleanup failure: {error}",
            )
            raise P8QuarantineError(
                f"queue {wave.queue_id} quarantined after POST_REJECTED cleanup failure"
            ) from error

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
                    QueueState.FILLING_CLAIMED,
                    QueueState.FILLING,
                    QueueState.FILLING_RELEASE_CLAIMED,
                    QueueState.POST_INTENT,
                    QueueState.POSTED,
                    QueueState.DRAIN_CLAIMED,
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
        self.phase = ForwardPhase.DRAIN

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

        self._require_state(ForwardState.RECONCILING)
        missing_exact = self._missing_layers(self.exact_winner_layers)
        missing_recall = self._missing_layers(self.guaranteed_recall_layers)
        if missing_exact or missing_recall:
            raise P8LifecycleError(
                "cannot begin QSFA before exact winner and GuaranteedRecall "
                f"are terminal: exact_missing={missing_exact} "
                f"recall_missing={missing_recall}"
            )
        self.exact_input_ready = True
        self.phase = ForwardPhase.QSFA_TERMINAL
        self.state = ForwardState.QSFA_READING

    def mark_exact_winners_confirmed(self, layer_id: int) -> None:
        """Record exact Top-K completion for one layer after all score tiles."""

        self._require_state(ForwardState.RECONCILING)
        self._validate_layer_control_index(layer_id)
        self.exact_winner_layers.add(layer_id)

    def mark_guaranteed_recall_terminal(self, layer_id: int) -> None:
        """Record the final exact miss-fill/readiness gate for one layer."""

        self._require_state(ForwardState.RECONCILING)
        self._validate_layer_control_index(layer_id)
        if layer_id not in self.exact_winner_layers:
            raise P8LifecycleError(
                "GuaranteedRecall cannot be terminal before exact winner "
                f"confirmation for layer {layer_id}"
            )
        self.guaranteed_recall_layers.add(layer_id)

    def mark_qsfa_terminal(self, layer_id: int) -> None:
        """Record one same-stream QSFA last-reader terminal operation."""

        self._require_state(ForwardState.QSFA_READING)
        if not 0 <= layer_id < self.expected_layers:
            raise ValueError(
                f"layer_id {layer_id} is outside [0,{self.expected_layers})"
            )
        if layer_id in self.qsfa_terminal_layers:
            raise P8LifecycleError(
                f"QSFA terminal was already recorded for layer {layer_id}"
            )
        self.qsfa_terminal_layers.add(layer_id)
        self.qsfa_terminal_observed = True
        # QSFA terminal is the last reader of this layer's private physical
        # slots.  Release those slots now, rather than artificially holding
        # the global private arena until an unrelated later layer/model event.
        # The forward's model-terminal event still owns host/native handles
        # and transaction destruction; it is not a substitute for a layer's
        # payload-reader proof.
        self._release_layer_private_slots_after_qsfa_terminal(layer_id)

    def mark_model_terminal_enqueued(self) -> None:
        """Record an event after all graph-side QSFA terminal ops were queued."""

        self._require_state(ForwardState.QSFA_READING)
        if len(self.qsfa_terminal_layers) != self.expected_layers:
            raise P8LifecycleError(
                "model terminal cannot be enqueued before every layer's "
                "QSFA last-reader terminal"
            )
        self.model_terminal_enqueued = True

    def mark_model_terminal_observed(self) -> None:
        """Record completion of the model-terminal event before reuse/free."""

        self._require_state(ForwardState.QSFA_READING)
        if not self.model_terminal_enqueued:
            raise P8LifecycleError(
                "model terminal cannot be observed before it is enqueued"
            )
        self.model_terminal_observed = True

    def enter_forward_terminal(self) -> None:
        """Close P8 admission and sweep unresolved durable resources.

        This is a *fail-closed* phase, not a second drain loop.  A normal
        forward may enter only after every QSFA reader and the model terminal
        have completed.  An aborting forward may enter earlier, but every
        non-free queue/private row belonging to the forward is quarantined;
        it is never guessed free merely because a Python-side callback ended.
        """

        if self.state is ForwardState.QSFA_READING:
            if len(self.qsfa_terminal_layers) != self.expected_layers:
                raise P8LifecycleError(
                    "forward terminal requires every QSFA last-reader terminal"
                )
            if not self.model_terminal_observed:
                raise P8LifecycleError(
                    "forward terminal requires observed model terminal"
                )
        elif self.state is not ForwardState.ABORTING:
            raise P8LifecycleError(
                "forward terminal is legal only after normal QSFA/model terminal "
                "or from an aborting forward"
            )

        self.phase = ForwardPhase.FORWARD_TERMINAL
        self._forward_terminal_sweep()
        self.forward_terminal_swept = True
        self.state = ForwardState.FORWARD_TERMINAL

    def release(self) -> None:
        """Release host/native ownership after the forward terminal event."""

        self._require_state(ForwardState.FORWARD_TERMINAL)
        if self.abort_latched:
            raise P8LifecycleError(
                "aborting forward must route to fallback, not normal release"
            )
        if not self.model_terminal_enqueued:
            raise P8LifecycleError(
                "cannot release P8 transaction before model terminal is enqueued"
            )
        if not self.model_terminal_observed:
            raise P8LifecycleError(
                "cannot release P8 transaction before model terminal is observed"
            )
        if len(self.qsfa_terminal_layers) != self.expected_layers:
            raise P8LifecycleError("cannot release before every QSFA reader is terminal")
        if not self.forward_terminal_swept:
            raise P8LifecycleError("cannot release before forward-terminal sweep")
        if not self._all_queues_released():
            raise P8LifecycleError("cannot release while a queue is still active")
        unresolved_private = [
            slot.slot_id
            for slot in self.resources.private_slots
            if slot.state in {
                PrivateSlotState.PENDING,
                PrivateSlotState.READY,
                PrivateSlotState.FAILED,
            }
        ]
        if unresolved_private:
            raise P8LifecycleError(
                "cannot release forward while a layer has not reported its "
                f"QSFA terminal: private_slots={unresolved_private}"
            )
        if not self.resources.claim_arena_terminal(self.arena_lease_generation):
            raise P8LifecycleError(
                "cannot claim P8 terminal after arena-lease generation mismatch"
            )
        if not self.resources.complete_arena_terminal(self.arena_lease_generation):
            self.resources.quarantine_arena_lease(
                self.arena_lease_generation,
                "arena terminal completion failed after unique terminal claim",
            )
            raise P8QuarantineError(
                "cannot release P8 transaction after arena-terminal completion failure"
            )
        self.host_leases.clear()
        self.state = ForwardState.RELEASED
        self.resources._detach(self)

    def abort(self, reason: str) -> None:
        """Latch a monotonic fail-closed abort bit for this forward.

        Multiple dynamic workers may discover faults.  The first reason is
        retained for diagnosis; later failures cannot clear, overwrite or
        reopen the forward.  Resource recovery remains the explicit
        forward-terminal sweep.
        """

        self._require_not_terminal()
        if not reason:
            raise ValueError("abort reason must be non-empty")
        if self.abort_latched:
            return
        self.abort_latched = True
        self.abort_reason = reason
        self.state = ForwardState.ABORTING

    def mark_fallback_ready(self) -> None:
        """Allow an exact fallback after all unsafe P8 records are quarantined.

        Host leases deliberately remain retained.  A future runtime teardown
        owns their recovery, because this method cannot prove that a late
        device writer has stopped.
        """

        self._require_state(ForwardState.FORWARD_TERMINAL)
        if not self.abort_latched:
            raise P8LifecycleError(
                "normal forward must release instead of entering fallback"
            )
        if not self.forward_terminal_swept:
            raise P8LifecycleError("fallback requires forward-terminal sweep")
        active = [
            queue.queue_id
            for queue in self.resources.queues
            if queue.state not in {QueueState.FREE, QueueState.QUARANTINED}
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
        # An early exact-fallback that proved every P8 resource clean may
        # return the arena normally.  Any actual ambiguity has already
        # transitioned the matching lease to QUARANTINED, which intentionally
        # remains unavailable until runtime teardown.
        if self.resources.arena_lease_state is ArenaLeaseState.LEASED:
            if not self.resources.claim_arena_terminal(self.arena_lease_generation):
                raise P8LifecycleError(
                    "cannot claim fallback terminal after arena-lease generation mismatch"
                )
            if not self.resources.complete_arena_terminal(
                self.arena_lease_generation
            ):
                self.resources.quarantine_arena_lease(
                    self.arena_lease_generation,
                    "fallback arena terminal completion failed after unique claim",
                )
                raise P8QuarantineError(
                    "cannot enter fallback after arena-terminal completion failure"
                )
        elif self.resources.arena_lease_state is not ArenaLeaseState.QUARANTINED:
            raise P8LifecycleError(
                "cannot enter fallback after arena lease was released by another owner"
            )
        self.state = ForwardState.FALLBACK_READY
        self.resources.adopt_quarantine_keepalives(self.host_leases)
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
            or wave.epoch != queue.epoch
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

    def _validate_wave_or_quarantine(
        self,
        wave: P8WaveIdentity,
        *,
        expected_state: QueueState | tuple[QueueState, ...],
        operation: str,
    ) -> P8WaveRecord:
        """Validate a potentially device-visible wave or fail closed.

        Every transition after ``POST_INTENT`` operates on a descriptor that
        may be observed by a device-side writer.  Therefore an epoch,
        generation, owner or state mismatch is not a recoverable Python
        control-flow error: the queue/private rows are quarantined and the
        shared arena admission gate is latched unhealthy.  Keeping this rule
        in one helper prevents individual lifecycle methods from accidentally
        reintroducing a raw, non-durable identity failure.
        """

        try:
            return self._validate_identity(wave, expected_state=expected_state)
        except P8LifecycleError as error:
            queue = self._queue(wave.queue_id)
            self._quarantine_queue(queue, f"{operation} identity failure: {error}")
            raise P8QuarantineError(
                f"queue {wave.queue_id} quarantined after {operation} identity failure"
            ) from error

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
                self.resources.private_ready[private_slot.slot_id] = (
                    PrivateSlotState.FAILED
                )
            else:
                self._rollback_unposted_private_claim(private_slot)
        if not failed:
            candidate.private_slot = None
            candidate.private_allocation_generation = 0

    def _publish_pending_private_mappings(
        self, private_slot: PrivateSlotRecord
    ) -> None:
        """Publish both PENDING directory entries before descriptor creation.

        The native claim does this with a ``-2`` transient marker and two
        CASes.  In this deterministic model the all-or-nothing precondition
        captures the same ownership rule: a candidate never adopts a stale
        source/object entry, and a partial publication is quarantined.
        """

        if (
            private_slot.state is not PrivateSlotState.ALLOCATING
            or private_slot.source_offset is None
            or private_slot.object_offset is None
            or private_slot.output_index < 0
        ):
            self._quarantine_private(
                private_slot,
                "invalid allocating private row before source/object publication",
            )
            raise P8QuarantineError(
                f"private slot {private_slot.slot_id} cannot publish mappings"
            )
        source = private_slot.source_offset
        obj = private_slot.object_offset
        if (
            source in self.resources.private_source_to_slot
            or obj in self.resources.private_object_to_slot
        ):
            self._quarantine_private(
                private_slot,
                "source/object mapping was already owned",
            )
            raise P8QuarantineError(
                f"private slot {private_slot.slot_id} mapping claim failed"
            )
        self.resources.private_source_to_slot[source] = private_slot.output_index
        self.resources.private_object_to_slot[obj] = private_slot.output_index
        # Publication is state-last: no downstream consumer may observe a
        # PENDING private row until both maps name this exact output slot.
        private_slot.state = PrivateSlotState.PENDING
        self.resources.private_ready[private_slot.slot_id] = (
            PrivateSlotState.PENDING
        )

    def _clear_private_mappings_exact(
        self, private_slot: PrivateSlotRecord
    ) -> bool:
        """Model the ordered exact-CAS mapping clear used at QSFA terminal."""

        source = private_slot.source_offset
        obj = private_slot.object_offset
        output_index = private_slot.output_index
        if source is None or obj is None or output_index < 0:
            return False
        if self.resources.private_source_to_slot.get(source) != output_index:
            return False
        # Preserve the native failure semantics: if object CAS fails after a
        # successful source CAS, do not try to compensate.  The row is later
        # quarantined, never reused.
        del self.resources.private_source_to_slot[source]
        if self.resources.private_object_to_slot.get(obj) != output_index:
            return False
        del self.resources.private_object_to_slot[obj]
        return True

    def _rollback_unposted_private_claim(
        self, private_slot: PrivateSlotRecord
    ) -> None:
        """Rollback only a PENDING row that never acquired a queue binding."""

        if (
            private_slot.state is not PrivateSlotState.PENDING
            or private_slot.queue_id is not None
        ):
            raise P8LifecycleError(
                "only an unposted PENDING private claim may be rolled back"
            )
        if not self._clear_private_mappings_exact(private_slot):
            self._quarantine_private(
                private_slot,
                "unposted private mapping rollback CAS failed",
            )
            raise P8QuarantineError(
                f"private slot {private_slot.slot_id} rollback mapping mismatch"
            )
        self.resources._commit_private_free(private_slot)

    def _publish_private_terminal_records(
        self,
        queue: P8WaveRecord,
        *,
        payload_ready: bool,
    ) -> None:
        """Publish READY/FAILED for every CQ-terminal row of one wave."""

        if queue.state is not QueueState.FINALIZED:
            raise P8LifecycleError("private terminal publication requires FINALIZED")
        if queue.private_ready_published:
            raise P8LifecycleError("private terminal records were already published")
        for candidate_slot in queue.candidate_slots:
            candidate = self._candidate(candidate_slot)
            private_slot = self._private_slot(candidate.private_slot)
            self._validate_private_slot_identity(
                candidate,
                private_slot,
                queue,
                P8WaveIdentity(
                    queue_id=queue.queue_id,
                    epoch=self.epoch,
                    graph_generation=self.graph_generation,
                    layer_control_index=queue.layer_control_index,
                    generation=queue.generation,
                    post_sequence=queue.post_sequence,
                    owner=queue.owner if queue.owner is not None else -1,
                ),
            )
            if private_slot.state is not PrivateSlotState.PENDING:
                raise P8LifecycleError(
                    f"private slot {private_slot.slot_id} cannot publish from "
                    f"{private_slot.state.name}"
                )
            private_slot.state = (
                PrivateSlotState.READY if payload_ready else PrivateSlotState.FAILED
            )
            self.resources.private_ready[private_slot.slot_id] = private_slot.state
            candidate.state = (
                CandidateState.CONSUMED if payload_ready else CandidateState.DEFERRED
            )
        queue.private_ready_published = True

    def _release_layer_private_slots_after_qsfa_terminal(
        self, layer_id: int
    ) -> None:
        """Return only this layer's proven last-read private payloads.

        Queue reuse is already allowed at CQ terminal.  This is the separate
        global-private-arena reuse boundary: exactly the matching layer's
        READY/FAILED rows become free after its QSFA terminal, while another
        layer's payload remains protected.
        """

        private_slots = [
            private_slot
            for private_slot in self.resources.private_slots
            if private_slot.epoch == self.epoch
            and private_slot.graph_generation == self.graph_generation
            and private_slot.layer_control_index == layer_id
        ]
        for private_slot in private_slots:
            if private_slot.state in {
                PrivateSlotState.PENDING,
                PrivateSlotState.QUARANTINED,
                PrivateSlotState.RELEASE_CLAIMED,
            }:
                raise P8LifecycleError(
                    "QSFA terminal cannot release non-terminal private slot "
                    f"{private_slot.slot_id}: state={private_slot.state.name}"
                )
            if private_slot.state in {
                PrivateSlotState.READY,
                PrivateSlotState.FAILED,
            }:
                self._release_private_after_qsfa_terminal(private_slot)

        for candidate in self.resources.candidates:
            if (
                candidate.epoch == self.epoch
                and candidate.graph_generation == self.graph_generation
                and candidate.layer_control_index == layer_id
            ):
                candidate.private_slot = None
                candidate.private_allocation_generation = 0

    def _release_private_after_qsfa_terminal(
        self, private_slot: PrivateSlotRecord
    ) -> None:
        """Perform ``READY/FAILED -> CLAIMED -> map clear -> FREE`` exactly."""

        if (
            self.phase is not ForwardPhase.QSFA_TERMINAL
            or not self.exact_input_ready
            or not self.qsfa_terminal_observed
            or private_slot.state
            not in {PrivateSlotState.READY, PrivateSlotState.FAILED}
        ):
            self._quarantine_private(
                private_slot,
                "QSFA terminal private-release control validation failed",
            )
            raise P8QuarantineError(
                f"private slot {private_slot.slot_id} cannot enter release claim"
            )
        original_state = private_slot.state
        if self.resources.private_ready[private_slot.slot_id] is not original_state:
            self._quarantine_private(
                private_slot,
                "private-ready table differs from private record",
            )
            raise P8QuarantineError(
                f"private slot {private_slot.slot_id} readiness mismatch"
            )
        private_slot.state = PrivateSlotState.RELEASE_CLAIMED
        if not self._clear_private_mappings_exact(private_slot):
            self._quarantine_private(
                private_slot,
                "private source/object mapping CAS clear failed",
            )
            raise P8QuarantineError(
                f"private slot {private_slot.slot_id} mapping-clear mismatch"
            )
        # The mapping disappearance becomes visible before the final free.
        self.resources.private_ready[private_slot.slot_id] = PrivateSlotState.FREE
        self.resources._commit_private_free(private_slot)

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
        self.resources.mark_arena_unhealthy(
            reason,
            lease_generation=self.arena_lease_generation,
        )
        for candidate_slot in queue.candidate_slots:
            candidate = self._candidate(candidate_slot)
            candidate.state = CandidateState.DEFERRED
            if candidate.private_slot is not None:
                private_slot = self._private_slot(candidate.private_slot)
                self._quarantine_private(private_slot, reason, latch_abort=False)
        self._latch_abort(reason)
        self.resources.adopt_quarantine_keepalives(self.host_leases)

    def _quarantine_private(
        self,
        private_slot: PrivateSlotRecord,
        reason: str,
        *,
        latch_abort: bool = True,
    ) -> None:
        """Make a private row permanently unavailable to this runtime arena."""

        private_slot.state = PrivateSlotState.QUARANTINED
        self.resources.private_ready[private_slot.slot_id] = (
            PrivateSlotState.QUARANTINED
        )
        self.resources.mark_arena_unhealthy(
            reason,
            lease_generation=self.arena_lease_generation,
        )
        if latch_abort:
            self._latch_abort(reason)

    def _latch_abort(self, reason: str) -> None:
        """Model the native sticky ABORT|PRIVATE_VIEW_DISABLED CAS."""

        if not self.abort_latched:
            self.abort_latched = True
            self.abort_reason = reason
        # A later worker may retain a more precise local diagnostic, but it
        # must never re-open the forward or replace the first durable reason.
        if self.state not in {
            ForwardState.FORWARD_TERMINAL,
            ForwardState.RELEASED,
            ForwardState.FALLBACK_READY,
        }:
            self.state = ForwardState.ABORTING

    def _forward_terminal_sweep(self) -> None:
        """Quarantine unresolved records owned by this exact forward only."""

        for queue in self.resources.queues:
            if (
                queue.epoch != self.epoch
                or queue.graph_generation != self.graph_generation
            ):
                continue
            if queue.state is not QueueState.FREE:
                self._quarantine_queue(
                    queue,
                    "forward-terminal found an unresolved queue wave",
                )

        for private_slot in self.resources.private_slots:
            if (
                private_slot.epoch != self.epoch
                or private_slot.graph_generation != self.graph_generation
            ):
                continue
            if private_slot.state is not PrivateSlotState.FREE:
                self._quarantine_private(
                    private_slot,
                    "forward-terminal found an unresolved private slot",
                )

        for candidate in self.resources.candidates:
            if (
                candidate.epoch == self.epoch
                and candidate.graph_generation == self.graph_generation
                and candidate.state in {CandidateState.PUBLISHED, CandidateState.CLAIMED}
            ):
                candidate.state = CandidateState.DEFERRED

    def _all_queues_released(self) -> bool:
        return all(queue.state is QueueState.FREE for queue in self.resources.queues)

    def _missing_layers(self, completed_layers: set[int]) -> list[int]:
        return [
            layer_id
            for layer_id in range(self.expected_layers)
            if layer_id not in completed_layers
        ]

    def _all_candidates_terminal(self) -> bool:
        return all(
            candidate.state
            in {CandidateState.EMPTY, CandidateState.CONSUMED, CandidateState.DEFERRED}
            for candidate in self.resources.candidates
        )

__all__ = [
    "ArenaLeaseState",
    "CandidateRecord",
    "CandidateState",
    "ForwardPhase",
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
