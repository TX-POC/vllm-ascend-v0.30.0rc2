# A5ExactReconcilePrivateView — exact object-domain planner

This isolated operator runs after the streaming C8 Lightning Indexer has
finished its unchanged exact Top-K. Its job is deliberately narrow: it turns
the selected C8 score sources into a transaction-private plan that tells the
next stage which KV objects are already resident, which speculative H2D rows
are ready, and which objects still need a guaranteed Get.

It does not copy payload, start a Get, call QSFA, publish a persistent
mapping, mutate the resident directory, or release a private slot. Keeping
those actions outside the planner is what lets the transaction protect the
current QSFA read set until its terminal reader finishes.

## Why the ABI says “KV object”, not “token”

Official C8 Top-K indices live in score-source domain `S`. When C8 is
compressed, an `S` index is not automatically a raw token address. The
planner therefore resolves each selected source through an explicit map to a
KV object `O`, the address unit understood by the resident directory, H2D
descriptors, private view, and final QSFA.

`kv_object_domain` is required for every graph:

| Value | Name | Valid `cmp_ratio` | Object `O` means |
| ---: | --- | --- | --- |
| `0` | `RAW_TOKEN` | `1` only | A legacy raw token ID. |
| `1` | `C8_COMPRESSED_OBJECT` | `1`, `4`, `128` | The exact compressed KV object that final QSFA consumes. |

The bridge and host tiling reject every other combination. In particular,
the planner never derives an object ID from `S * cmp_ratio`. That inference
is not valid for a paged or causal C8 score stream.

## Inputs

All nine inputs are contiguous `int32` NPU tensors. Let
`R = ProductExceptLast(exact_topk_indices)`; one independent plan is produced
for every flattened exact C8 Top-K row.

| Input | Shape | Meaning |
| --- | --- | --- |
| `exact_topk_indices` | `[...,2048]` | Unchanged official C8 score-source indices `S`. |
| `req_pool_entries` | `[R]` | Pool row for each C8 output row. |
| `kv_object_slots_snapshot` | `[P,O_capacity]` | Frozen resident object-to-slot directory. It is read only. |
| `cache_tokens` | `[R]` | Resident-slot budget used by the reference placement algorithm. |
| `kv_object_lens` | `[R]` | Active `O` prefix for each row. |
| `score_source_lens` | `[R]` | Exact active `S` prefix emitted by C8; it is not inferred from `O`. |
| `kv_object_map` | `[P,S_capacity]` | Explicit `S -> O` map. |
| `private_source_to_slot` | `[R,S_capacity]` | Speculative private row per C8 source, or `-1`. |
| `private_ready` | `[Q,private_rows]` | `0=free`, `1=pending`, `2=terminal-CQ ready`, `3=failed`. |

For every exact index, the planner performs this order of checks:

```text
S is inside score_source_lens[row]
    -> O = kv_object_map[pool, S]
    -> O is inside kv_object_lens[row]
    -> inspect kv_object_slots_snapshot[pool, O]
```

Only then may it classify the object as a resident hit, a ready private row,
or a guaranteed fallback. A padded C8 source, invalid map entry, invalid
private row, or duplicate selected object fails closed; it is not repaired by
guessing a raw token number.

## Outputs

All fields named `*_kv_object_ids` contain `O` values. They contain raw token
IDs only when `kv_object_domain=RAW_TOKEN` (which is already limited to
`cmp_ratio=1`).

| Output | Meaning |
| --- | --- |
| `kv_object_ids`, `destination_slots`, `miss_counts` | Exact LIDU-style miss-prefix/hit-suffix plan in object domain `O`. |
| `placement_delta[R,3,2048]` | `[victim_O, incoming_O, destination_slot]` for miss-prefix positions. It is a proposal, not a mutation. |
| `private_view_descriptors[R,4,2048]` | `[O, destination_slot, disposition, private_global_slot]`; disposition is resident hit, ready private, fallback, or invalid. |
| `ready_private_kv_object_ids`, `ready_private_slots`, `ready_private_destination_slots`, `ready_private_counts` | Compact ready-private subset in original miss order. |
| `fallback_kv_object_ids`, `fallback_destination_slots`, `fallback_counts` | Compact exact unresolved subset that must be fetched before QSFA. |

The selected `S -> O` map must be injective. Official C8 normally makes valid
selected `S` values unique, but a user map can still cause two selected
sources to name the same object. Allowing that would make a placement delta
ambiguous, so the kernel rejects it.

## Equivalence boundary

The planner preserves the serial selection/placement order only in the
address domain declared by `kv_object_domain`:

- For `RAW_TOKEN`, this is the legacy LIDU contract and is valid only when
  `cmp_ratio=1`.
- For `C8_COMPRESSED_OBJECT`, the comparison baseline must also operate on
  those exact compressed objects. The DDR/HBM source, private payload,
  object directory, and QSFA view must all describe the same representation.

After this plan, a materializer combines resident hits, terminal-CQ-ready
private rows, and the small exact fallback Get set into the private QSFA
view. QSFA is the final reader. Only after it finishes may a later stage apply
`placement_delta`, publish persistent mapping, and recycle private resources.

The operator is built only through
`NANOVLLM_A5_OPS_ONLY=A5ExactReconcilePrivateView`; it is not part of the
normal product policy path. The static contract test is:

```bash
python3 -m pytest tests/ut/dsa_offload/test_exact_reconcile_private_view_contract.py
```
