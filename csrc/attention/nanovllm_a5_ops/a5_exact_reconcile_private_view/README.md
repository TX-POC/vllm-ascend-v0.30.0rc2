# A5ExactReconcilePrivateView (isolated prototype)

This operator is the **post-Top-K transaction planner** for the streaming C8
experiment.  It deliberately begins only after the source-derived Lightning
Indexer has emitted its unchanged exact Top-2048 indices.

It has one narrow job: reproduce the selection order and victim-slot choice of
`A5LiduCacheUpdate` against a read-only `cache_slots_snapshot`, then classify
each exact miss as either already present in the transaction's private HBM
arena or requiring a guaranteed final Get.

It does **not** issue a Get, copy a private packed-KV row into the resident
cache, call QSFA, mutate `cache_slots`, publish a mapping, or recycle a private
slot.  Those steps require a terminal-CQ / QSFA-reader lifetime contract and
are intentionally outside this component.

## Input contract

All tensors are contiguous NPU tensors.

| Input | Shape | Meaning |
| --- | --- | --- |
| `exact_topk_indices` | `[B, 1, 2048]` int32 | Exact official LI candidates.  These are the same source indices consumed by legacy LIDU. |
| `req_pool_entries` | `[B]` int32 | Pool row for each request. |
| `cache_slots_snapshot` | `[P, S]` int32 | Frozen logical-source to resident-slot directory.  It is never written. |
| `cache_tokens`, `candidate_lens` | `[B]` int32 | Same budget and candidate-length inputs used by legacy LIDU. |
| `logical_token_map` | `[P, S]` int32 | Maps a LI source index to the logical KV token required by Direct-HCCP Get. |
| `private_source_to_slot` | `[B, S]` int32 | `-1` for no speculative copy; otherwise global private slot `queue * private_rows_per_queue + row`. |
| `private_ready` | `[Q, R]` int32 | Must be `1` only after the corresponding speculative Get's terminal CQ succeeds. |

`logical_token_map` does not change legacy source ordering.  `source_ids` use
the original LI source indices so they can be checked mechanically against
legacy LIDU; only the Get request list is translated to logical token IDs.

## Outputs

`source_ids`, `destination_slots`, and `miss_counts` exactly mirror legacy
`A5LiduCacheUpdate` for the same snapshot.  The miss prefix and hit suffix are
preserved.

`placement_delta` has shape `[B, 3, 2048]`.  Each miss-prefix column contains
`[victim_source, incoming_source, destination_slot]`; all other entries are
`-1`.  A later transaction terminal may apply this delta **after** QSFA's last
reader.  This operator never applies it.

`private_view_descriptors` has shape `[B, 4, 2048]`, with planes
`[logical_token, destination_slot, disposition, private_global_slot]`.
Disposition is `0 = resident hit`, `1 = private CQ-ready`, `2 = guaranteed
fallback Get required`, or `3 = invalid transaction descriptor`.

The explicit plan split makes the next two device stages unambiguous:

- `ready_private_source_ids`, `ready_private_slots`,
  `ready_private_destination_slots`, and `ready_private_counts` compact the
  disposition-1 exact misses in original miss-prefix order.  Source IDs remain
  the exact LI source IDs; the materializer reads the already-ready private
  row at `ready_private_slots` and writes it to the given final destination.
- `fallback_logical_token_ids`, `fallback_destination_slots`, and
  `fallback_counts` compact only disposition-2 exact misses in the same order.
  These source IDs are **logical token IDs**, because Direct-HCCP Get indexes
  the layer directory by logical token rather than LI's possibly compressed
  score index.

Either plan count being `-1` means a malformed logical mapping or private-slot
reference; the caller must fail closed rather than let QSFA read an incomplete
view.

## Correct future composition

```text
unchanged exact LI Top-K
          |
          v
A5ExactReconcilePrivateView
  |                 |
  | private-ready   +--> guaranteed final Get (only exact unresolved misses)
  v
materialize both paths into a QSFA-readable temporary physical-slot view
          |
          v
QSFA terminal reader
          |
          v
apply placement_delta / release private arena
```

The operator is buildable only through
`NANOVLLM_A5_OPS_ONLY=A5ExactReconcilePrivateView`; it is not added to the
normal product package or policy route.
