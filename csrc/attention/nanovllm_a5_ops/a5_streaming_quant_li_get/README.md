# A5StreamingQuantLiGet — isolated streaming C8 → speculative-Get prototype

This operator is an experiment only. It copies the official Ascend-950 C8
Lightning Indexer (LI), leaves its score calculation and exact `ProcessTopK`
algorithm unchanged, and adds a side effect after each completed score tile.
The side effect may start a private H2D Get while LI continues scanning later
score tiles. It never changes a C8 score, sort order, Top-K index, resident
directory, or persistent placement mapping.

The important point is that C8 and QSFA do **not** necessarily name a KV by
the same number. Treating every C8 output index as a raw token ID is correct
only in one restricted configuration. This ABI makes the missing distinction
explicit so that a compressed C8 path cannot silently issue a Get for the
wrong data.

## The three address domains

For one flattened C8 output row, the operator uses the following domains.

| Name | Meaning | Who produces/consumes it |
| --- | --- | --- |
| `S` — score source | The index emitted by official C8 Top-K. It names an entry in the LI score stream, not automatically a cache address. | Official LI produces it as `sparse_indices`. |
| `O` — KV object | The unit addressed by the resident directory, speculative/guaranteed Get, private view, and final QSFA. It must identify the exact packed/compressed KV representation that QSFA will read. | `kv_object_map` resolves `S -> O`. |
| raw token | A legacy token position. It is an `O` only when the graph declares `RAW_TOKEN`. | Legacy LIDU/QSFA-compatible case. |

`kv_object_domain` is a **required graph-static attribute** that tells both
this operator and the later reconcile operator what `O` means:

| Value | Name | Allowed `cmp_ratio` | Meaning |
| ---: | --- | --- | --- |
| `0` | `RAW_TOKEN` | `1` only | `O` is the raw token ID. This is the only domain in which existing raw-token LIDU output can be reused directly. |
| `1` | `C8_COMPRESSED_OBJECT` | `1`, `4`, or `128` | `O` identifies the exact C8/QSFA compressed KV object. It is not converted to a token by multiplication or division. |

The graph fails closed if the attribute is omitted, has another value, or
uses `RAW_TOKEN` with `cmp_ratio != 1`. In particular, neither
`O = S * cmp_ratio` nor `O = floor(S / cmp_ratio)` is a valid fallback.

## What happens during one layer

```text
official C8 computes a tile of scores in score-source domain S
    |
    +-- tile-ready hook observes a promising S
    |       |
    |       +-- validate S < score_source_lens[row]
    |       +-- O = kv_object_map[req_pool_entries[row], S]
    |       +-- validate O < kv_object_lens[row]
    |       +-- inspect kv_object_slots_snapshot[pool, O]
    |       +-- if not resident, reserve a private slot and post Get(O)
    |
    +-- official C8 continues scanning and finally produces exact Top-K S
            |
            +-- A5ExactReconcilePrivateView resolves the same S -> O map
            +-- it keeps resident/ready private O values and plans only the
                still-missing guaranteed Gets
            +-- QSFA reads the completed private object view
```

The hook does not wait for the complete 128K scan. The final C8 Top-K remains
the authority; a speculative transfer only reduces the number of later
guaranteed transfers. A failed, pending, duplicate, unavailable, or
unallocated speculative entry is treated as a miss by reconcile and therefore
cannot change attention math.

## ABI: 21 inputs and 9 outputs

Inputs `0..8` are the original official C8 LI ABI. The experimental inputs
are appended, so their addition does not alter official C8 input positions.

| Input | Shape / type | Purpose |
| ---: | --- | --- |
| 9 | `resident_threshold:int32[R]` | Resident-only C8 threshold in the official sortable-score representation. |
| 10 | `resident_threshold_valid:int32[R]` | Zero disables speculation for that row; zero threshold alone is never interpreted as valid. |
| 11 | `kv_object_slots_snapshot:int32[P,O_capacity]` | Read-only resident directory, indexed by KV object `O`. |
| 12 | `req_pool_entries:int32[R]` | Pool row for every flattened C8 output row. |
| 13 | `kv_object_map:int32[P,S_capacity]` | Explicit mapping from C8 score source `S` to QSFA/Get object `O`. |
| 14 | `private_source_to_slot:int32[R,S_capacity]` | Per-source speculative private-slot receipt; `-1` means no speculative copy. |
| 15 | `private_ready_table:int32[Q,private_rows]` | `0=free`, `1=pending`, `2=terminal-CQ ready`, `3=failed`. |
| 16 | `private_arena_head:int32[Q]` | Per-forward private-slot allocator head. |
| 17 | `private_packed_output:uint8[Q*private_rows,656]` | Explicit registered graph-visible H2D output. |
| 18 | `queue_control:int64[16]` | Bound KVAF streaming control and descriptor storage. |
| 19 | `queue_leases:int32[Q]` | Direct-HCCP queue ownership words. |
| 20 | `kv_object_lens:int32[R]` | Active KV-object prefix for the resident directory. |

`R` is the flattened C8 output-row count. In the copied official A5 C8
implementation, `N2` is required to be `1`; a BSND output therefore flattens
as `row = b * S1 + q`, while a TND output flattens as
`row = prefixQ(b) + q`. The well-known C8 `C=4` is an internal score tile,
not an output-head (`N2`) dimension.

The nine outputs are the unchanged official `sparse_indices` and
`sparse_values`, five explicit alias receipts for mutable transaction state,
`telemetry`, and `score_source_lens:int32[R]`. The latter is produced from
the same official `ProcessTopK` valid-source calculation as the exact C8
indices. It is an `S` bound, not an `O` bound.

The required attributes are the official C8 attributes plus
`kvaf_rows_per_entry` and `kv_object_domain`. A Get descriptor must address
the KVAF object directory using `O`; it may use
`entry = O / kvaf_rows_per_entry` and
`offset = (O % kvaf_rows_per_entry) * 656` only when the registered KVAF
directory is itself indexed by that same `O` domain.

## Exactness and duplicate rules

For valid C8 positions, official Top-K chooses distinct source indices `S`.
Padding values such as `-1` may repeat only outside the valid prefix and are
rejected by reconciliation. A user-supplied `kv_object_map` can still map
two valid sources to the same `O`. That is unsafe: it could make two exact
Top-K elements alias one KV object and invalidate LIDU-style placement.
`A5ExactReconcilePrivateView` therefore requires the selected `S -> O` map to
be injective and fails closed otherwise. The tile hook may suppress or
coalesce a speculative duplicate, but it may never make correctness depend on
that speculation.

## Conditions for mathematical equivalence

The strongest reference is the current serial path: official C8 determines
the exact selection, then LIDU/recall prepares the KV representation consumed
by official QSFA. This prototype is equivalent only when all of the following
hold:

1. The copied LI's exact indices are bitwise equal to the official C8 result.
   The hook is observational and cannot modify score or Top-K state.
2. Every selected `S` maps to the same object that the reference QSFA path
   would read. `RAW_TOKEN` gives this directly only for `cmp_ratio=1`.
3. In `C8_COMPRESSED_OBJECT` mode, the object directory, local/DDR payload
   layout, speculative Get, guaranteed Get, and QSFA private view all use the
   same compressed object representation. It is **not** enough for the
   object to cover a similar range of raw tokens.
4. Reconcile routes every nonresident object that is not terminal-CQ ready to
   a guaranteed Get before QSFA. Speculation only changes when bytes arrive,
   never which object QSFA reads.
5. A mapping/length/domain violation disables this route. It must not fall
   back inside the same graph to a guessed raw-token interpretation.

Under domain `0`, this reduces to the existing raw-token LIDU/QSFA semantics.
Under domain `1`, the proper reference is an object-domain LIDU/QSFA path;
claiming equivalence to a raw-token baseline without that representation proof
would be incorrect.

## Transaction and graph boundary

`private_packed_output_out` is a receipt, not permission to assume CANN made
an in-place alias. Capture must bind each mutable receipt output to the same
physical allocation as its input and verify device addresses before graph
capture. The later reconcile/QSFA path consumes that verified allocation.
Persistent mapping updates and private-arena release happen only after the
QSFA terminal reader has finished.

This OPP is isolated from the product policy route. It is validated first by
the static source-contract probe:

```bash
python3 csrc/attention/nanovllm_a5_ops/tests/test_streaming_quant_li_get_contract.py
```

Target validation must additionally compare this operator and official C8 on
identical inputs and require bitwise-equal exact `sparse_indices` before any
product gate can open. A performance result without that comparator only
measures scheduling, not mathematical correctness.
