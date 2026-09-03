"""Device-independent P10 score-receipt and resident-slot protocol.

This is a mathematical/lifecycle model of the intended forward-owned device
pipeline, not a CPU transport fallback:

* Stage A publishes only descriptors for completed, native-C8 uint16 score
  windows.  It has no threshold, S->O mapping, resident state, CandidateSeed,
  queue, or HCCP knowledge.
* Stage B claims a descriptor, reads the referenced score surface, performs
  ExactScoreScan and CandidateCompact, reserves a forward-owned slot, and
  models PostOnly/CQ completion.
* Stage C consumes unchanged exact C8 Top-K source IDs and guaranteed-recalls
  every unresolved winner before QSFA may read resident slots.

Receipt-ring entries use the explicit lifecycle
``FREE -> FILLING -> READY -> CLAIMED -> DONE -> FREE``.  ``DONE`` is
deliberate: Stage B must first make a durable local copy (or terminally reject
an invalid descriptor) before an explicit recycle makes the GM ring slot
available again.  A claimed record is therefore never silently made free.

Resident slots are monotonic per-forward allocations, so a false-positive
speculative payload cannot displace a final winner and no cross-forward ABA
protocol is needed.  The bounded HCCP queue is shared by speculative and
guaranteed-recall posts; final exact winners are admitted while that executor
is still alive, not after it has drained to terminal.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence


class ProtocolViolation(RuntimeError):
    """Raised when an invalid stage ordering or bounded-resource rule is used."""


class ReceiptState(str, Enum):
    """A logical GM receipt-slot state machine."""

    FREE = "free"
    FILLING = "filling"
    READY = "ready"
    CLAIMED = "claimed"
    DONE = "done"


@dataclass(frozen=True)
class ScoreObservation:
    """One post-reduction score in C8's native uint16 score-source domain."""

    row: int
    source: int
    score: int


@dataclass(frozen=True)
class ScoreReceiptDescriptor:
    """The exact Stage-A ring payload, excluding ring ownership metadata.

    ``score_surface_offset_bytes`` is relative to the formal graph-visible
    score surface of the streaming sibling.  It is explicitly *not* an address
    into CANN's opaque per-AIC workspace.  The reference model retains
    ``ScoreReceipt.score_surface_values`` only to model that separately owned
    surface; those values are not CandidateSeed fields in the descriptor.
    """

    row: int
    source_begin: int
    valid_count: int
    score_surface_offset_bytes: int
    score_element_bytes: int
    sequence: int


@dataclass(frozen=True)
class CandidateSeed:
    """Stage-B output: ``[row, source, kv_object, slot_ordinal]``."""

    row: int
    source: int
    kv_object: int
    slot_ordinal: int


@dataclass(frozen=True)
class ScoreReceipt:
    """A published raw-score descriptor plus its modeled score-surface view."""

    worker: int
    ring_slot: int
    tile: int
    descriptor: ScoreReceiptDescriptor
    score_surface_values: tuple[int, ...]


@dataclass(frozen=True)
class PendingPost:
    """Stage-B-owned descriptor copy accepted by PostOnly but not CQ-drained."""

    queue_id: int
    seeds: tuple[CandidateSeed, ...]
    path: str
    receipt: ScoreReceipt | None = None


@dataclass(frozen=True)
class WinnerPlacement:
    """One exact winner and its final resident location."""

    row: int
    source: int
    kv_object: int
    slot_id: int
    path: str


@dataclass(frozen=True)
class FinalizedForward:
    """QSFA-facing result after every final winner has a resident slot."""

    winner_sources: tuple[tuple[int, ...], ...]
    winner_objects: tuple[tuple[int, ...], ...]
    attention_slots: tuple[tuple[int, ...], ...]
    placements: tuple[WinnerPlacement, ...]

    def assert_object_equivalent(
        self, object_directory: Sequence[Sequence[int]]
    ) -> None:
        """Prove speculation did not change exact winner object order."""

        expected: list[tuple[int, ...]] = []
        for row, sources in enumerate(self.winner_sources):
            row_objects: list[int] = []
            for source in sources:
                if source == -1:
                    row_objects.append(-1)
                else:
                    row_objects.append(int(object_directory[row][source]))
            expected.append(tuple(row_objects))
        if tuple(expected) != self.winner_objects:
            raise ProtocolViolation(
                "final winner objects differ from the exact source-domain "
                "Top-K reference"
            )


@dataclass
class ProtocolStats:
    """Backpressure diagnostics; none affects exact Top-K semantics."""

    score_observations: int = 0
    score_receipts_published: int = 0
    candidates_compacted: int = 0
    candidates_posted: int = 0
    guaranteed_recall_posted: int = 0
    cq_terminal: int = 0
    receipts_completed: int = 0
    receipts_rejected: int = 0
    deferred_ring_full: int = 0
    deferred_seed_capacity: int = 0
    deferred_arena_capacity: int = 0
    deferred_queue_busy: int = 0
    skipped_persistent: int = 0
    skipped_duplicate: int = 0
    skipped_invalid_object: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass
class _ReceiptRecord:
    state: ReceiptState = ReceiptState.FREE
    receipt: ScoreReceipt | None = None


@dataclass
class ForwardOwnedArena:
    """Per-forward fixed ``S + K`` resident layout without slot release.

    Score-time speculation consumes only the prefix.  Final guaranteed recall
    consumes only the suffix.  This is a memory-capacity partition, not an
    AIV/core partition and not an arena-reuse lifecycle.
    """

    rows: int
    speculative_capacity: int
    recall_capacity: int
    _speculative_cursor: list[int] = field(init=False, repr=False)
    _recall_cursor: list[int] = field(init=False, repr=False)
    _allocated: set[tuple[int, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError("rows must be positive")
        if self.speculative_capacity < 0 or self.recall_capacity < 0:
            raise ValueError("arena capacities cannot be negative")
        self._speculative_cursor = [0] * self.rows
        self._recall_cursor = [0] * self.rows
        self._allocated = set()

    @property
    def slots_per_row(self) -> int:
        return self.speculative_capacity + self.recall_capacity

    def slot_id(self, row: int, ordinal: int) -> int:
        self._check_row(row)
        if ordinal < 0 or ordinal >= self.slots_per_row:
            raise ProtocolViolation("slot ordinal is outside the forward arena")
        return row * self.slots_per_row + ordinal

    def reserve_speculative(self, row: int) -> int | None:
        self._check_row(row)
        cursor = self._speculative_cursor[row]
        if cursor >= self.speculative_capacity:
            return None
        self._speculative_cursor[row] = cursor + 1
        self._mark_allocated(row, cursor)
        return cursor

    def reserve_recall(self, row: int) -> int | None:
        self._check_row(row)
        cursor = self._recall_cursor[row]
        if cursor >= self.recall_capacity:
            return None
        self._recall_cursor[row] = cursor + 1
        ordinal = self.speculative_capacity + cursor
        self._mark_allocated(row, ordinal)
        return ordinal

    def remaining_recall(self, row: int) -> int:
        self._check_row(row)
        return self.recall_capacity - self._recall_cursor[row]

    def allocated_slot_ids(self) -> tuple[int, ...]:
        return tuple(
            self.slot_id(row, ordinal) for row, ordinal in sorted(self._allocated)
        )

    def _mark_allocated(self, row: int, ordinal: int) -> None:
        key = (row, ordinal)
        if key in self._allocated:
            raise ProtocolViolation("forward-owned arena attempted slot reuse")
        self._allocated.add(key)

    def _check_row(self, row: int) -> None:
        if not 0 <= row < self.rows:
            raise ProtocolViolation("row is outside the forward arena")


class ScoreReceiptResidentProtocol:
    """Reference state machine for one forward-owned streaming invocation."""

    def __init__(
        self,
        *,
        object_directory: Sequence[Sequence[int]],
        active_row_count: int | None = None,
        workers: int,
        ring_slots_per_worker: int = 2,
        seed_capacity: int = 64,
        hccp_queue_count: int = 2,
        speculative_slots_per_row: int,
        recall_slots_per_row: int,
    ) -> None:
        if not object_directory:
            raise ValueError("object_directory must contain at least one row")
        if (
            workers <= 0
            or ring_slots_per_worker <= 0
            or seed_capacity <= 0
            or hccp_queue_count <= 0
        ):
            raise ValueError(
                "workers, ring slots, seed capacity, and HCCP queue count must be positive"
            )
        self._object_directory = tuple(
            tuple(int(value) for value in row) for row in object_directory
        )
        if any(not row for row in self._object_directory):
            raise ValueError("every object-directory row must be non-empty")
        self.rows = len(self._object_directory)
        self.active_row_count = (
            self.rows if active_row_count is None else int(active_row_count)
        )
        if not 0 <= self.active_row_count <= self.rows:
            raise ValueError("active_row_count is outside object-directory rows")
        self.workers = int(workers)
        self.ring_slots_per_worker = int(ring_slots_per_worker)
        # This is Stage B's bounded CandidateCompact microbatch, not Stage A's
        # ring payload capacity.
        self.seed_capacity = int(seed_capacity)
        # This is a fixed transport-resource pool, not a physical-AIV split.
        self.hccp_queue_count = int(hccp_queue_count)
        self.arena = ForwardOwnedArena(
            rows=self.rows,
            speculative_capacity=int(speculative_slots_per_row),
            recall_capacity=int(recall_slots_per_row),
        )
        self.stats = ProtocolStats()
        self._records = [
            [_ReceiptRecord() for _ in range(self.ring_slots_per_worker)]
            for _ in range(self.workers)
        ]
        self._producer_done = [False] * self.workers
        self._next_receipt_sequence = 1
        self._compacted: dict[tuple[int, int, int], tuple[CandidateSeed, ...]] = {}
        # The resident payload is object O, while C8 scores source occurrence S.
        self._reserved_object_to_ordinal: dict[tuple[int, int], int] = {}
        self._terminal_object_to_ordinal: dict[tuple[int, int], int] = {}
        self._terminal_object_path: dict[tuple[int, int], str] = {}
        self._free_queue_ids = set(range(self.hccp_queue_count))
        self._pending: list[PendingPost] = []
        self._recall_backlog: list[CandidateSeed] = []
        self._admitted_winners: tuple[tuple[int, ...], ...] | None = None
        self._admitted_persistent: dict[tuple[int, int], int] | None = None
        self._finalized: FinalizedForward | None = None
        self._executor_terminal = False

    @property
    def finalized(self) -> FinalizedForward | None:
        return self._finalized

    @property
    def executor_terminal(self) -> bool:
        return self._executor_terminal

    @property
    def pending_post_count(self) -> int:
        """Number of durable PostOnly waves awaiting CQ terminal."""

        return len(self._pending)

    @property
    def free_queue_count(self) -> int:
        """Number of HCCP queues currently available to either post path."""

        return len(self._free_queue_ids)

    @property
    def recall_backlog_count(self) -> int:
        """Exact-winner Get seeds admitted but awaiting shared-queue capacity."""

        return len(self._recall_backlog)

    def receipt_states(self) -> tuple[tuple[ReceiptState, ...], ...]:
        return tuple(
            tuple(record.state for record in worker_records)
            for worker_records in self._records
        )

    def publish_exact_score_window(
        self,
        *,
        worker: int,
        tile: int,
        observations: Iterable[ScoreObservation],
        score_surface_offset_bytes: int,
        score_element_bytes: int = 2,
    ) -> ScoreReceipt | None:
        """Stage A: publish a raw completed score-window descriptor.

        Crucially, this method accepts no threshold, object map, resident map,
        or arena.  The model's observations are the values addressed by the
        formal score surface; production Stage A publishes only the descriptor
        after its independent score-DMA completion edge.
        """

        self._ensure_open()
        self._check_worker(worker)
        if self._producer_done[worker]:
            raise ProtocolViolation("a completed producer cannot publish another tile")
        values = tuple(observations)
        if not values:
            return None
        row, source_begin = self._validate_exact_score_window(
            values, score_surface_offset_bytes, score_element_bytes
        )
        self.stats.score_observations += len(values)
        if row >= self.active_row_count:
            # Graph-padded rows have no Stage-A receipt and cannot later make
            # a QSFA mapping.  No candidate policy was evaluated here.
            return None
        free_index = self._find_free_ring_slot(worker)
        if free_index is None:
            self.stats.deferred_ring_full += len(values)
            return None
        sequence = self._next_receipt_sequence
        self._next_receipt_sequence += 1
        descriptor = ScoreReceiptDescriptor(
            row=row,
            source_begin=source_begin,
            valid_count=len(values),
            score_surface_offset_bytes=int(score_surface_offset_bytes),
            score_element_bytes=int(score_element_bytes),
            sequence=sequence,
        )
        record = self._records[worker][free_index]
        record.state = ReceiptState.FILLING
        receipt = ScoreReceipt(
            worker=worker,
            ring_slot=free_index,
            tile=int(tile),
            descriptor=descriptor,
            score_surface_values=tuple(int(value.score) for value in values),
        )
        record.receipt = receipt
        # State-last publish models: descriptor/score surface visible first,
        # then a GM release store transitions the ring slot to READY.
        record.state = ReceiptState.READY
        self.stats.score_receipts_published += 1
        return receipt

    def claim_ready_receipt(self) -> ScoreReceipt | None:
        """Stage B's READY -> CLAIMED CAS, followed by descriptor acquire."""

        self._ensure_open()
        for worker_records in self._records:
            for record in worker_records:
                if record.state is ReceiptState.READY:
                    record.state = ReceiptState.CLAIMED
                    receipt = record.receipt
                    if receipt is None or not self._receipt_payload_is_valid(receipt):
                        # Fail closed.  The descriptor cannot become reusable
                        # until an explicit DONE -> FREE recycle, and it is
                        # visible in diagnostics rather than being a no-op.
                        record.receipt = None
                        record.state = ReceiptState.DONE
                        self.stats.receipts_rejected += 1
                        continue
                    return receipt
        return None

    def compact_claimed_receipt(
        self,
        receipt: ScoreReceipt,
        *,
        threshold_by_row: Mapping[int, int],
        persistent_object_slots: Mapping[tuple[int, int], int] | None = None,
    ) -> tuple[CandidateSeed, ...]:
        """Stage B: ExactScoreScan + CandidateCompact + resident admission.

        This is deliberately the first point that accesses threshold, S->O,
        resident/persistent state, or forward arena capacity.  It never changes
        the descriptor or any original C8 Top-K state.
        """

        self._ensure_open()
        record = self._record_for(receipt)
        if record.state is not ReceiptState.CLAIMED or record.receipt != receipt:
            raise ProtocolViolation("only the claimed receipt can be compacted")
        key = self._receipt_key(receipt)
        if key in self._compacted:
            raise ProtocolViolation("a claimed score receipt was compacted twice")
        persistent = persistent_object_slots or {}
        seeds: list[CandidateSeed] = []
        row = receipt.descriptor.row
        for offset, score in enumerate(receipt.score_surface_values):
            observation = ScoreObservation(
                row=row,
                source=receipt.descriptor.source_begin + offset,
                score=int(score),
            )
            candidate = self._candidate_from_observation(
                observation, threshold_by_row, persistent
            )
            if candidate is None:
                continue
            if len(seeds) >= self.seed_capacity:
                self.stats.deferred_seed_capacity += 1
                continue
            ordinal = self.arena.reserve_speculative(candidate.row)
            if ordinal is None:
                self.stats.deferred_arena_capacity += 1
                continue
            seed = CandidateSeed(
                row=candidate.row,
                source=candidate.source,
                kv_object=candidate.kv_object,
                slot_ordinal=ordinal,
            )
            self._reserved_object_to_ordinal[(seed.row, seed.kv_object)] = ordinal
            seeds.append(seed)
            self.stats.candidates_compacted += 1
        compacted = tuple(seeds)
        self._compacted[key] = compacted
        return compacted

    def post_compacted_receipt(
        self, receipt: ScoreReceipt, seeds: Sequence[CandidateSeed]
    ) -> bool:
        """Try Stage-B PostOnly without bypassing bounded queue ownership.

        ``False`` is bounded-resource backpressure, not a dropped seed.  The
        receipt remains ``CLAIMED`` and its compacted list remains durable for
        a later retry after CQ frees any queue.  A successful copy moves the
        receipt to ``DONE``; only :meth:`recycle_done_receipt` can make it
        ``FREE``.
        """

        self._ensure_open()
        record = self._record_for(receipt)
        if record.state is not ReceiptState.CLAIMED or record.receipt != receipt:
            raise ProtocolViolation("only the claimed receipt can be posted")
        key = self._receipt_key(receipt)
        expected = self._compacted.get(key)
        actual = tuple(seeds)
        if expected is None:
            raise ProtocolViolation("PostOnly requires Stage-B CandidateCompact first")
        if actual != expected:
            raise ProtocolViolation("PostOnly seed list differs from compacted receipt")
        if actual and not self._try_post(
            actual, path="speculative_terminal", receipt=receipt
        ):
            return False
        self._compacted.pop(key)
        record.state = ReceiptState.DONE
        self.stats.receipts_completed += 1
        return True

    def recycle_done_receipt(self, receipt: ScoreReceipt) -> None:
        """Explicitly recycle a completed Stage-B receipt slot.

        PostOnly owns a private seed copy before this operation.  CQ may still
        be pending, but ring reuse cannot invalidate that copy.  This split is
        what makes the ``DONE -> FREE`` transition observable and prevents a
        CLAIMED record from being silently reused.
        """

        self._ensure_open()
        record = self._record_for(receipt)
        if record.state is not ReceiptState.DONE or record.receipt != receipt:
            raise ProtocolViolation("only the matching DONE receipt can be recycled")
        record.receipt = None
        record.state = ReceiptState.FREE

    def recycle_done_receipts(self) -> int:
        """Recycle all completed/rejected ring entries after their terminal copy."""

        self._ensure_open()
        recycled = 0
        for worker_records in self._records:
            for record in worker_records:
                if record.state is ReceiptState.DONE:
                    record.receipt = None
                    record.state = ReceiptState.FREE
                    recycled += 1
        return recycled

    def drain_cq(self, *, limit: int | None = None) -> int:
        """Make accepted posts terminal under the no-transport-failure model."""

        self._ensure_open()
        count = None if limit is None else int(limit)
        if count is not None and count < 0:
            raise ValueError("CQ drain limit cannot be negative")
        drained = 0
        while self._pending and (count is None or drained < count):
            pending = self._pending.pop(0)
            if pending.queue_id in self._free_queue_ids:
                raise ProtocolViolation("CQ terminal observed a queue that was already free")
            for seed in pending.seeds:
                key = (seed.row, seed.kv_object)
                expected = self._reserved_object_to_ordinal.get(key)
                if expected != seed.slot_ordinal:
                    raise ProtocolViolation("terminal post lost its arena reservation")
                self._terminal_object_to_ordinal[key] = seed.slot_ordinal
                self._terminal_object_path[key] = pending.path
                self.stats.cq_terminal += 1
            self._free_queue_ids.add(pending.queue_id)
            drained += 1
            # Exact final recall uses precisely this same queue pool.  A CQ
            # terminal immediately gives it a chance to make progress without
            # waiting for speculative executor terminal.
            self._pump_recall_posts()
        return drained

    def mark_producer_done(self, worker: int) -> None:
        self._ensure_open()
        self._check_worker(worker)
        if any(
            record.state in (ReceiptState.FILLING, ReceiptState.CLAIMED)
            for record in self._records[worker]
        ):
            raise ProtocolViolation("producer completion raced an active receipt")
        self._producer_done[worker] = True

    def mark_executor_terminal(self) -> None:
        """Close Stage B only after all producer/ring/CQ work is terminal."""

        self._ensure_open()
        if not all(self._producer_done):
            raise ProtocolViolation("executor terminal before every producer completed")
        if self._pending:
            raise ProtocolViolation("executor terminal before CQ drain")
        if self._compacted:
            raise ProtocolViolation("executor terminal before compacted receipt PostOnly")
        if self._recall_backlog:
            raise ProtocolViolation("executor terminal before final recall PostOnly")
        if any(
            record.state is not ReceiptState.FREE
            for worker_records in self._records
            for record in worker_records
        ):
            raise ProtocolViolation("executor terminal while a receipt is not free")
        if self._admitted_winners is None:
            raise ProtocolViolation("executor terminal before exact winner admission")
        if not self._all_admitted_winners_terminal():
            raise ProtocolViolation("executor terminal before final winner CQ drain")
        self._executor_terminal = True

    def admit_final_winners(
        self,
        exact_winner_sources: Sequence[Sequence[int]],
        *,
        persistent_object_slots: Mapping[tuple[int, int], int] | None = None,
    ) -> int:
        """Admit exact winners into the still-live shared PostOnly executor.

        C8's exact Top-K is authoritative.  Objects already reserved by
        speculation keep their fixed prefix slot and simply wait for their
        existing PostOnly/CQ path.  Every otherwise-unresolved winner receives
        a fixed suffix slot and is submitted through the *same* bounded queue
        pool.  This must happen before executor terminal; ``finalize`` only
        exposes terminal resident slots to QSFA.
        """

        self._ensure_open()
        if self._executor_terminal:
            raise ProtocolViolation("exact winners cannot be admitted after executor terminal")
        winners = self._validate_exact_winners(exact_winner_sources)
        persistent = self._normalize_persistent(persistent_object_slots)
        if self._admitted_winners is not None:
            if winners != self._admitted_winners or persistent != self._admitted_persistent:
                raise ProtocolViolation("exact winner admission differs from prior admission")
            return 0

        missing: list[tuple[int, int, int]] = []
        missing_by_row = [0] * self.rows
        seen_missing: set[tuple[int, int]] = set()
        for row, row_winners in enumerate(winners):
            for source in row_winners:
                if source == -1:
                    continue
                obj = self._object_at(row, source)
                key = (row, obj)
                if key in persistent or key in self._reserved_object_to_ordinal:
                    continue
                if key in seen_missing:
                    continue
                seen_missing.add(key)
                missing.append((row, source, obj))
                missing_by_row[row] += 1
        for row, needed in enumerate(missing_by_row):
            if needed > self.arena.remaining_recall(row):
                raise ProtocolViolation(
                    "recall arena is too small for exact winners; no slots were "
                    f"allocated for row={row} needed={needed} "
                    f"remaining={self.arena.remaining_recall(row)}"
                )

        recall_seeds: list[CandidateSeed] = []
        for row, source, obj in missing:
            ordinal = self.arena.reserve_recall(row)
            if ordinal is None:
                raise ProtocolViolation("recall reservation unexpectedly failed")
            key = (row, obj)
            self._reserved_object_to_ordinal[key] = ordinal
            recall_seeds.append(
                CandidateSeed(
                    row=row, source=source, kv_object=obj, slot_ordinal=ordinal
                )
            )
        self._admitted_winners = winners
        self._admitted_persistent = persistent
        self._recall_backlog.extend(recall_seeds)
        self._pump_recall_posts()
        return len(recall_seeds)

    def finalize(
        self,
        exact_winner_sources: Sequence[Sequence[int]],
        *,
        persistent_object_slots: Mapping[tuple[int, int], int] | None = None,
    ) -> FinalizedForward:
        """Expose terminal resident slots after prior exact-winner admission."""

        self._ensure_open()
        if not self._executor_terminal:
            raise ProtocolViolation("finalization requires terminal executor drain")
        persistent = self._normalize_persistent(persistent_object_slots)
        winners = self._validate_exact_winners(exact_winner_sources)
        if self._admitted_winners is None:
            raise ProtocolViolation("finalization requires exact winner admission")
        if winners != self._admitted_winners or persistent != self._admitted_persistent:
            raise ProtocolViolation("finalization differs from exact winner admission")
        if not self._all_admitted_winners_terminal():
            raise ProtocolViolation("finalization requires final winner CQ drain")

        placements: list[WinnerPlacement] = []
        attention_slots: list[tuple[int, ...]] = []
        winner_objects: list[tuple[int, ...]] = []
        for row, row_winners in enumerate(winners):
            row_slots: list[int] = []
            row_objects: list[int] = []
            for source in row_winners:
                if source == -1:
                    row_slots.append(-1)
                    row_objects.append(-1)
                    continue
                obj = self._object_at(row, source)
                persistent_slot = persistent.get((row, obj))
                if persistent_slot is not None:
                    slot_id = int(persistent_slot)
                    path = "persistent"
                else:
                    object_key = (row, obj)
                    ordinal = self._terminal_object_to_ordinal.get(object_key)
                    if ordinal is None:
                        raise ProtocolViolation(
                            "final winner has no terminal forward-owned resident slot"
                        )
                    path = self._terminal_object_path[object_key]
                    slot_id = self.arena.slot_id(row, ordinal)
                row_slots.append(slot_id)
                row_objects.append(obj)
                placements.append(
                    WinnerPlacement(
                        row=row,
                        source=source,
                        kv_object=obj,
                        slot_id=slot_id,
                        path=path,
                    )
                )
            attention_slots.append(tuple(row_slots))
            winner_objects.append(tuple(row_objects))

        result = FinalizedForward(
            winner_sources=winners,
            winner_objects=tuple(winner_objects),
            attention_slots=tuple(attention_slots),
            placements=tuple(placements),
        )
        result.assert_object_equivalent(self._object_directory)
        self._finalized = result
        return result

    def _validate_exact_score_window(
        self,
        observations: Sequence[ScoreObservation],
        score_surface_offset_bytes: int,
        score_element_bytes: int,
    ) -> tuple[int, int]:
        if int(score_element_bytes) != 2:
            raise ProtocolViolation("raw C8 score receipt requires uint16 score elements")
        if int(score_surface_offset_bytes) < 0 or int(score_surface_offset_bytes) % 2:
            raise ProtocolViolation("score surface offset must be uint16 aligned")
        row = int(observations[0].row)
        source_begin = int(observations[0].source)
        if not 0 <= row < self.rows:
            raise ProtocolViolation("score observation row is outside the object directory")
        for offset, observation in enumerate(observations):
            if int(observation.row) != row:
                raise ProtocolViolation("one score receipt must describe one logical row")
            if int(observation.source) != source_begin + offset:
                raise ProtocolViolation("score receipt sources must be one contiguous range")
            if not 0 <= int(observation.score) <= 0xFFFF:
                raise ProtocolViolation("score must be a uint16 value in the C8 score domain")
            if not 0 <= int(observation.source) < len(self._object_directory[row]):
                raise ProtocolViolation("score observation source is outside its row")
        return row, source_begin

    def _candidate_from_observation(
        self,
        observation: ScoreObservation,
        threshold_by_row: Mapping[int, int],
        persistent: Mapping[tuple[int, int], int],
    ) -> CandidateSeed | None:
        row = int(observation.row)
        source = int(observation.source)
        score = int(observation.score)
        if not 0 <= row < self.active_row_count:
            return None
        if row not in threshold_by_row:
            raise ProtocolViolation("every active score row requires a threshold")
        threshold = int(threshold_by_row[row])
        if not 0 <= threshold <= 0xFFFF:
            raise ProtocolViolation("threshold must use the same uint16 score domain")
        if score < threshold:
            return None
        obj = self._object_at(row, source)
        if obj < 0:
            self.stats.skipped_invalid_object += 1
            return None
        if (row, obj) in persistent:
            self.stats.skipped_persistent += 1
            return None
        key = (row, obj)
        if key in self._reserved_object_to_ordinal:
            self.stats.skipped_duplicate += 1
            return None
        return CandidateSeed(row=row, source=source, kv_object=obj, slot_ordinal=-1)

    def _try_post(
        self,
        seeds: Sequence[CandidateSeed],
        *,
        path: str,
        receipt: ScoreReceipt | None = None,
    ) -> bool:
        """Own a fixed queue, then make one PostOnly wave durable.

        It intentionally never waits for CQ.  Queue exhaustion retains the
        caller's seed ownership and reports bounded backpressure.
        """

        if not seeds:
            return True
        if not self._free_queue_ids:
            self.stats.deferred_queue_busy += len(seeds)
            return False
        queue_id = min(self._free_queue_ids)
        self._free_queue_ids.remove(queue_id)
        self._pending.append(
            PendingPost(
                queue_id=queue_id,
                seeds=tuple(seeds),
                path=path,
                receipt=receipt,
            )
        )
        if path == "speculative_terminal":
            self.stats.candidates_posted += len(seeds)
        elif path == "guaranteed_recall":
            self.stats.guaranteed_recall_posted += len(seeds)
        else:
            raise ProtocolViolation("unknown PostOnly path")
        return True

    def _pump_recall_posts(self) -> int:
        """Move admitted exact winners through the same bounded queue pool."""

        posted = 0
        while self._recall_backlog:
            batch = tuple(self._recall_backlog[: self.seed_capacity])
            if not self._try_post(batch, path="guaranteed_recall"):
                break
            del self._recall_backlog[: len(batch)]
            posted += len(batch)
        return posted

    def _all_admitted_winners_terminal(self) -> bool:
        if self._admitted_winners is None or self._admitted_persistent is None:
            return False
        for row, row_winners in enumerate(self._admitted_winners):
            for source in row_winners:
                if source == -1:
                    continue
                obj = self._object_at(row, source)
                key = (row, obj)
                if key not in self._admitted_persistent and key not in self._terminal_object_to_ordinal:
                    return False
        return True

    def _normalize_persistent(
        self, mapping: Mapping[tuple[int, int], int] | None
    ) -> dict[tuple[int, int], int]:
        result: dict[tuple[int, int], int] = {}
        for (row, obj), slot_id in (mapping or {}).items():
            if not 0 <= int(row) < self.rows or int(obj) < 0 or int(slot_id) < 0:
                raise ProtocolViolation("persistent resident mapping is invalid")
            result[(int(row), int(obj))] = int(slot_id)
        return result

    def _receipt_payload_is_valid(self, receipt: ScoreReceipt) -> bool:
        """Validate a claimed descriptor before Stage B trusts its score window."""

        descriptor = receipt.descriptor
        row = int(descriptor.row)
        if (
            receipt.worker < 0
            or receipt.worker >= self.workers
            or receipt.ring_slot < 0
            or receipt.ring_slot >= self.ring_slots_per_worker
            or descriptor.sequence <= 0
            or descriptor.score_element_bytes != 2
            or descriptor.score_surface_offset_bytes < 0
            or descriptor.score_surface_offset_bytes % 2
            or descriptor.valid_count != len(receipt.score_surface_values)
            or not 0 <= row < self.rows
            or descriptor.source_begin < 0
            or descriptor.source_begin + descriptor.valid_count > len(self._object_directory[row])
        ):
            return False
        return all(0 <= int(score) <= 0xFFFF for score in receipt.score_surface_values)

    def _validate_exact_winners(
        self, exact_winner_sources: Sequence[Sequence[int]]
    ) -> tuple[tuple[int, ...], ...]:
        if len(exact_winner_sources) != self.rows:
            raise ProtocolViolation("exact Top-K rows do not match the forward arena")
        result: list[tuple[int, ...]] = []
        for row, row_winners in enumerate(exact_winner_sources):
            values = tuple(int(source) for source in row_winners)
            if row >= self.active_row_count and any(source != -1 for source in values):
                raise ProtocolViolation(
                    "an inactive graph-padded row cannot have an exact Top-K winner"
                )
            seen: set[int] = set()
            saw_padding = False
            for source in values:
                if source == -1:
                    saw_padding = True
                    continue
                if saw_padding:
                    raise ProtocolViolation("exact Top-K padding must be a suffix")
                if not 0 <= source < len(self._object_directory[row]):
                    raise ProtocolViolation("exact Top-K source is outside the object directory")
                if self._object_at(row, source) < 0:
                    raise ProtocolViolation(
                        "exact Top-K source maps to an invalid KV object"
                    )
                if source in seen:
                    raise ProtocolViolation("exact Top-K cannot contain a duplicate source")
                seen.add(source)
            result.append(values)
        return tuple(result)

    def _find_free_ring_slot(self, worker: int) -> int | None:
        for index, record in enumerate(self._records[worker]):
            if record.state is ReceiptState.FREE:
                return index
        return None

    def _record_for(self, receipt: ScoreReceipt) -> _ReceiptRecord:
        self._check_worker(receipt.worker)
        if not 0 <= receipt.ring_slot < self.ring_slots_per_worker:
            raise ProtocolViolation("receipt ring slot is invalid")
        return self._records[receipt.worker][receipt.ring_slot]

    @staticmethod
    def _receipt_key(receipt: ScoreReceipt) -> tuple[int, int, int]:
        return (receipt.worker, receipt.ring_slot, receipt.descriptor.sequence)

    def _object_at(self, row: int, source: int) -> int:
        return int(self._object_directory[row][source])

    def _check_worker(self, worker: int) -> None:
        if not 0 <= int(worker) < self.workers:
            raise ProtocolViolation("worker is outside the receipt ring")

    def _ensure_open(self) -> None:
        if self._finalized is not None:
            raise ProtocolViolation("a finalized forward cannot accept more work")


def winner_objects_from_exact_sources(
    object_directory: Sequence[Sequence[int]],
    exact_winner_sources: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Pure baseline: map unchanged exact C8 source IDs to KV object IDs."""

    if len(object_directory) != len(exact_winner_sources):
        raise ProtocolViolation("baseline row count mismatch")
    result: list[tuple[int, ...]] = []
    for row, sources in enumerate(exact_winner_sources):
        values: list[int] = []
        for source in sources:
            source = int(source)
            if source == -1:
                values.append(-1)
            elif 0 <= source < len(object_directory[row]):
                values.append(int(object_directory[row][source]))
            else:
                raise ProtocolViolation("baseline exact source is outside the directory")
        result.append(tuple(values))
    return tuple(result)


def summarize_paths(result: FinalizedForward) -> Counter[str]:
    """Return stable terminal-path counts for a compact test receipt."""

    return Counter(placement.path for placement in result.placements)
