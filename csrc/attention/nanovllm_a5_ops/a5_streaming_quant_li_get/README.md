# A5StreamingQuantLiGet — isolated P3 source-derived prototype

This directory is deliberately isolated from the product LI route. It derives
the Ascend-950 `VllmQuantLightningIndexer` C8 kernel and keeps the official
score calculation and `ProcessTopK` unchanged.

**Mathematical contract.** `resident_threshold` is read only by the
speculative side effect. It cannot select, mask, reorder, or write an official
score, score-GM entry, metadata field, or `sparse_indices` entry. The derived
op's exact Top-K path is source-identical to official C8; only the extra
transaction-private Get/ready state can differ. Target hardware still must
prove the resulting official sparse indices are bitwise equal before any
product gate opens.

The only added seam is inside `ProcessVec1`:

```text
official BatchMulWeightAndReduceSum  (uint16 sortable BF16 scores)
    -> PipeBarrier<PIPE_V>
    -> StreamingKvafTileReadyHook::Emit
    -> official DataCopyPad(scoreGm, ...)
    -> later official ProcessTopK
```

`Emit` has a bounded, lossless-speculation contract:

1. It computes the exact output row as
   `info.indiceOutOffset / sparse_count + s1_index`; this works for both
   product `TND` and `BSND` output layouts.
2. It observes the official score tile. For a score at or above the resident
   threshold, it resolves
   `pool = req_pool_entries[row]`, then
   `logical_token = logical_token_map[pool, source_score_index]`, then checks
   `cache_slots_snapshot[pool, logical_token]`. It does **not** look up cache
   residency by compressed S2 index.
3. A nonresident candidate atomically claims a bounded transaction-private
   arena slot. `private_source_to_slot[row, source]` moves `-1 -> -2` while
   claiming, then to encoded `(queue, local_slot)` only after a real post.
4. The producer leases one `queue_leases:int32[queue_count]` entry, fills the
   already host-bound per-queue descriptor arrays, publishes the descriptor,
   and calls KVAF `BeginKvafStreamingBatchGet`. It does not invent a WQE.
   Addressing is explicit: `indices=logical_token/kvaf_rows_per_entry`,
   `offsets=(logical_token%kvaf_rows_per_entry)*656`, and `output_indices`
   are absolute rows in the queue-private range of the one registered output.
5. Later score tiles call `TryRetire`.  The **final** in-flight wave stays
   posted while unchanged official `ProcessTopK` reads `scoreGm` and writes
   exact sparse indices.  Only after that exact output copy has completed
   does `ProcessTopK` call `Finish`.  A terminal CQ then changes private
   readiness from `PENDING` to `READY`/`FAILED`, and only then is the queue
   lease released.  Thus the final Get can overlap exact Top-K; neither Top-K
   nor its math depends on the private arena.

If no queue or private slot is available, the hook simply suppresses that
prefetch. It never changes official scores, sparse indices, cache slots,
persistent mapping, or managed HBM placement. Thus final exact Top-K plus a
later guaranteed fallback remains mathematically equivalent to the serial
baseline.

## Current pipeline boundary

The hook deliberately remains immediately after
`BatchMulWeightAndReduceSum` and **before** the official `V_MTE2`, `V_MTE3`,
and cross-core release sequence. At that point `outBase` is known complete and
is still owned by this ping-pong vector iteration. Moving the cross-core flag
or allowing the next reuse before the hook consumes that tile could race the
same UB score buffer or its `DataCopyPad` read; this prototype does not make
that unsafe rearrangement merely to claim more overlap.

Consequently P3 first proves a real scan-time Direct-HCCP post/CQ lifecycle
with the exact official score representation, including final-H2D versus
official-TopK overlap. It does **not** yet claim an optimized AIC-versus-vector
overlap gain. A later performance pass must first add a separately synchronized
score-tile handoff (or prove a safe multi-reader UB/event schedule), then
compare the original and rescheduled AIC/vector dependencies on target
hardware.

## Resident-threshold prerequisite

`resident_threshold:uint16[row_count]` is intentionally an input, not a value
derived from the final full-sequence Top-K.  Its producer must be a separate,
graph-stable `A5ResidentC8Threshold`/ResidentSeed node that runs the same C8
score representation and Top-K selection on the **resident-only LI index
view**, then emits the per-row 2048th sortable `uint16` score.  It must also
produce the resident source-to-logical-token relation used to construct this
op's `logical_token_map`.

The existing packed 656-byte QSFA KV payload is not a Lightning-Indexer key
view, and `cache_slots_snapshot` merely tells this op whether a logical token
is resident.  Neither can be used to score resident candidates.  The seed
therefore needs a graph-static gather/compact view of the LI key and scale
tensors (the C8 FP8 key/scales), plus the inverse resident-slot-to-logical-token
mapping.  Existing LIDU/cache-update code receives final Top-K indices and
cannot recover a resident-only 2048th score; reusing its full-stream kth value
would be the wrong threshold.  Until that upstream node and its exact runtime
comparison exist, this P3 op remains fail-closed rather than inventing a
threshold.

## Exact transaction ABI

The root `ops.json` and host definition use exactly 19 inputs and 8 outputs:

| Inputs 0–8 | Unchanged official C8 LI tensors, including mandatory metadata |
| --- | --- |
| 9 | `resident_threshold:uint16[row_count]` |
| 10 | `cache_slots_snapshot:int32[pool_rows,max_logical_tokens]` |
| 11 | `req_pool_entries:int32[row_count]` |
| 12 | `logical_token_map:int32[pool_rows,source_score_count]` |
| 13 | `private_source_to_slot:int32[row_count,source_score_count]`, initially `-1` |
| 14 | `private_ready_table:int32[queue_count,private_arena_capacity]` |
| 15 | `private_arena_head:int32[queue_count]`, zeroed per forward |
| 16 | `private_packed_output:uint8[total_private_rows,656]`, the explicit graph-visible owner of the registered KVAF output |
| 17 | `queue_control:int64[16]` (`KvafStreamingBatchGetControl`) |
| 18 | `queue_leases:int32[queue_count]`, zero means free |

Outputs are exact official `sparse_indices`, official `sparse_values`, and
formal alias receipts for every mutable transaction resource:

```text
private_source_to_slot_out  -> input 13
private_ready_table_out     -> input 14
private_arena_head_out      -> input 15
private_packed_output_out   -> input 16
queue_leases_out            -> input 18
telemetry                   -> int64[queue_count,13]
```

The alias outputs must bind to the same transaction allocations as their
inputs; the LI kernel deliberately does not copy a 128K source map or a
private HBM payload.  `StreamingKvafTileReadyHook::Init` additionally rejects
the graph if `queue_control.output` is not exactly the address of explicit
input 16, or if `queue_control.output_rows` differs from its row count.

### Graph physical-buffer rule (fail closed)

`private_packed_output_out` is **not** permission to assume that CANN made an
in-place alias automatically.  The capture bridge must explicitly bind each
receipt output to the same allocation as its mutable input and verify the
device addresses before graph capture:

```text
13 -> private_source_to_slot_out
14 -> private_ready_table_out
15 -> private_arena_head_out
16 -> private_packed_output_out
18 -> queue_leases_out
```

The later reconcile/QSFA stage reads the *same physical buffer supplied as
input 16*, using `private_source_to_slot_out` and `private_ready_table_out` as
the ordering/visibility receipts. It must either consume the verified alias
output 5 as that buffer or consume input 16 plus a receipt output that creates
the graph dependency. If any input/output address pair differs, or the bridge
cannot express that dependency, the streaming route is disabled; it may not
silently read a separately allocated `private_packed_output_out`.

The cold KVAF binder owns graph-static per-queue request templates
(`address_table`, `valid_table`, `indices`, `offsets`, `mask`,
`output_indices`). Each queue gets non-overlapping candidate-array slices and
an absolute private row range inside input 16. The device hook loads that
template only after acquiring the lease, fills only candidate arrays plus
`active_request_count`, then publishes into the active shared scratch. The
control itself represents one registered layer output whose rows divide evenly
by queue count; the device helper validates this before a doorbell.

## What this operator does not do

It does not implement post-TopK reconciliation, private QSFA view creation,
or persistent cache publication. A later `ExactReconcile/PrivateView` stage
must consume:

```text
exact_sparse_indices + req_pool_entries + logical_token_map
    + private_source_to_slot + private_ready_table
```

For each exact index it selects resident payload, READY private payload, or a
guaranteed fallback Get. `FAILED`, `PENDING`, no source-map entry, or arena
exhaustion all route to fallback. No speculative failure changes math.

## Metadata companion

The derived main LI still unconditionally consumes official C8 metadata. The
standalone graph must enqueue the already registered
`VllmQuantLightningIndexerMetadata` companion immediately before this op on
the same stream, producing `int32[1024]`. The source closure is recorded in
`metadata_companion.md`; no hidden `torch_npu` metadata is accepted.

## Validation boundary

Run the portable source-contract probe before target compilation:

```bash
python3 csrc/attention/nanovllm_a5_ops/tests/test_streaming_quant_li_get_contract.py
```

It reports only a source-level score non-mutation guard and doorbell/CQ
ordering. It deliberately does not report H2D bandwidth, end-to-end payload
success, or target-runtime Top-K equivalence. Before any product gate, a CANN
9.1 Ascend-950 runtime comparator must run official C8 and this derived op on
the same inputs and require bitwise equality of `sparse_indices` and (when
the official reference can expose them) `sparse_values`. The fail-closed
target harness is
`tests/test_streaming_quant_li_get_target_equivalence.py --adapter
module:function`; its adapter is intentionally not supplied until a generated
ACLNN bridge can launch both real target ops. Current official A5 C8 tiling
rejects `return_values=true`, so its first real case compares exact indices
only rather than fabricating a score-value comparison. The isolated OPP build
is additionally required to prove that the embedded Direct-HCCP helper's
UB/SIMT resources coexist with the official LI vector path.
