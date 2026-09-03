# P10 Stage B: score-receipt to forward-resident Get

`A5P10ScoreReceiptForwardResident` is the device-side companion of the P10
W4A4C8 score producer. It launches from the forward's cold-bound mutable
aliases on a companion stream; it is *not* connected to Stage A by a normal
Stage-A-to-Stage-B completion edge. The native C8 numerical loop retains
ownership of exact score and Top-K, while this executor polls only published,
graph-visible score receipts. It is not a host callback and it must not call a
Python KVAF API.

The only required completion edge is from this executor's `forward_state` to
QSFA: QSFA may read `attention_slots` only after `forward_state=COMPLETE`.
That preserves overlap without allowing QSFA to see an unfinished resident
slot map.

The companion's `executorWorkerCount` is a tiler-chosen logical worker count
bounded by `executorWorkerLimit`; it is not a physical-AIV selection and must
not mean "launch on every C8 AIV". Each worker reaches a bounded
`receiptPollQuantum` checkpoint between scan turns. This CANN source domain
has no portable instruction that releases a running AIV to a different
kernel, so real occupancy control belongs to the product graph/task scheduler,
not a fictitious in-kernel physical-core policy.

## Data flow

1. Before either C8 or the companion begins, the graph allocates and
   initializes all mutable Stage-B aliases: every `receipt_state` entry is
   `FREE`, `producer_sealed=OPEN`, `source_to_forward_slot`,
   `kv_object_to_forward_slot`, fixed speculative/recall bands, queue leases,
   resident rows, `producer_seal_sequence=0`, and
   `first_preseal_ready_claim_sequence=0`. The companion executor is launched
   at this point and polls the graph-static `receiptCapacity` range; it does
   not wait for C8 completion. Stage A must not clear or reinitialize receipt state once this concurrent phase begins.
2. Stage A writes a completed `uint16` score window to `score_surface`, writes
   its five-word receipt descriptor, and makes `receipt_state=READY` last.
   After all possible slots in its static receipt geometry have either been
   published or left unused, it writes its final static receipt sequence, then
   writes `producer_sealed=SEALED` last. The first Stage-B `READY` claim that
   atomically observes OPEN latches its nonzero receipt sequence in the second
   diagnostic word. Only
   after that seal is `FREE` terminal (an unused slot); `FILLING` or `CLAIMED`
   still requires bounded progress to `DONE`.
3. A Stage-B worker atomically claims `READY`, performs `ExactScoreScan` and
   `CandidateCompact`, then resolves `row -> pool -> kv_object -> resident
   slot` on device. `kv_object_to_forward_slot[P,S]` is authoritative, so
   duplicate references to the same pool/source token share one Post and one
   fixed forward slot. A queue lease is dynamic; it is not an AIV-core
   assignment.
4. The worker uses the verified KVAF forward-resident helpers in this order:
   `BeginKvafForwardResidentBatchGet` (which publishes the descriptor and
   rings PostOnly), `TryRetireKvafForwardResidentBatchGet`, then
   `FinishKvafForwardResidentBatchGet`.  The queue is returned only after the
   terminal finish.  All of those calls are device-side.
5. Once the producer seal is observed and every receipt slot is either `DONE`
   or unused `FREE`, Stage B resolves every source in
   `canonical_sparse_indices`.  The cold-bound `kv_object_map[P,S]` is not an
   arbitrary remap: Stage B verifies
   `O_token=((P*num_layers+layer_id)*qcap*128)+S`; it keys its P-local resident
   directories by `S`, then gives KVAF `O_token` so its address table uses
   `P*num_layers*qcap+layer_id*qcap+(S>>7)` and byte offset `(S&127)*656`.
   Winners that do not already have a persistent
   or speculative placement use their deterministic row in `recall_slot_band`.
   `resident_row_to_qsfa_slot[W]` translates the fixed resident row to the
   QSFA logical slot understood by the resident block table; the caller-owned
   full `[B,1,2048+tail]` `attention_slots_buffer` is updated only in its
   `[0,2048)` prefix and preserves the native tail verbatim. It never writes a
   physical resident row number into a QSFA logical-slot field.
   `selected_slot_lengths` is retained verbatim. `attention_slots` and
   `resident_seq_lengths` are produced only after all final winner rows are
   ready. QSFA therefore consumes its usual slot map;
   it does not learn about receipts or queue leases.

The graph ABI has explicit tensors for every mutable forward-local resource:
receipt state, source-to-forward-slot map, object-to-forward-slot de-dup map,
speculative cursor, queue leases, resident readiness, and forward control.
The cold-bound KVAF control record is passed as an opaque `int64` tensor, but
the graph still receives the queue lease tensor separately so side effects are
graph-visible.

## Scope and build guard

The source below is a real ABI and device-wrapper boundary, not a simulated
transport implementation. `build_stage_b_opp.sh` is the only supported OPP
route: it creates a fresh `msopgen` package, stages this source plus the exact
matched KVAF device closure, replaces the generated kernel stub with the
distinct P10 MIX entry, verifies one `OP_ADD`, and emits `P10_STAGE_PROVENANCE`.
It never accepts an installed/ambient KVAF header, a pre-existing package, or a
generic/P8/P9 artifact. `CMakeLists.txt` is manifest-friendly for a CANN
superproject; it is not an alternative ambient-artifact build route.

The only supported transport interface is:

```text
kvaf_forward_resident_get_device.h
  BeginKvafForwardResidentBatchGet
  TryRetireKvafForwardResidentBatchGet
  FinishKvafForwardResidentBatchGet
```

No P8/P9 source, host-side `execute_batch_get`, or fixed Get-core partition is
permitted in this directory.
