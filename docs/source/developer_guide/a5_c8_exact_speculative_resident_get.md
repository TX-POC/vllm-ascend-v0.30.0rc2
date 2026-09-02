# A5 C8 ExactScan + SpeculativeGet：canonical resident slot 目标态

## Scope

This document defines the target implementation for one decode layer.  It
replaces the older P7/P8 private-arena experiment with a layer-local,
single-operator boundary:

```text
C8 score scan → score-time CandidateCompact / Direct-HCCP PostOnly
              → exact C8 Top-2048
              → Guaranteed Recall for omitted final misses
              → drain every posted CQ
              → publish final attention_slots
              → QSFA reads resident slots
```

The operator has not completed until every final Top-2048 KV has been written
to the `attention_slots` it returns.  QSFA never consumes a pending transfer.
The ordinary framework stream dependency from this operator to QSFA also
orders the next reuse of the same layer's fixed range.  P9 does not support
two independent forwards concurrently writing one range; that would be a
different scheduling contract, not a missing per-slot ABA field.

## Fixed canonical resident range

The current LIDU update assigns `topk_slots` only after exact Top-K, so it
cannot supply a safe destination for an early DMA.  P9 therefore reserves a
fixed `TopK=2048` slot range per request row and layer before the fused
operator starts:

```text
persistent resident range: [0, resident_budget - TopK)
fixed speculative range:  [resident_budget - TopK, resident_budget)
```

The normal persistent source-to-slot table intentionally excludes the fixed
range.  A score-time candidate obtains a unique slot in that range.  It writes
there directly through Direct-HCCP, but does not mutate the persistent
source-to-slot map early.

After exact Top-K:

1. Existing persistent hits keep their existing slots.
2. A final winner whose speculative Get was posted keeps its predetermined
   fixed slot.
3. Every remaining final miss takes an unused slot in the same fixed range and
   is posted by Guaranteed Recall.

Before step 3 reuses an entry that held a non-winner speculative request, the
operator drains the initial speculative waves.  That is the only required
reuse barrier: the old DMA must be terminal before the same canonical slot is
overwritten by Guaranteed Recall.  It is wholly inside one operator call, not
a cross-forward slot-release protocol.

The fixed range is sufficient: there are at most TopK final misses, and every
speculative winner consumes exactly one of its TopK slots.  Candidate overflow
is deferred to Guaranteed Recall; it cannot overwrite a slot or affect the
exact C8 result.

This deliberately trades `TopK` entries of persistent history for an
unambiguous direct landing zone.  The gate requires
`resident_budget >= 2 * TopK`; it is an explicit capacity requirement, never a
silent change to the baseline cache policy.

## Intra-operator queue protocol

Queue ownership remains dynamic, not hard-partitioned by AIV:

```text
any C8 worker: compact seed → CAS lease idle HCCP queue
             → fill descriptor → DDR publish barrier → PostOnly
             → continue score tiles → bounded TryRetire
             → Finish all remaining waves before operator return
```

Only one wave occupies a hardware queue at a time because its descriptor
scratch and CQ are shared.  This is a transport constraint, not a permanent
Get core assignment.  A queue becomes reusable only after its CQ is terminal,
but that reuse is entirely inside the invocation.

For this first direct-resident route, a successful Post is assumed to complete
through URMA as requested.  Completion is still mandatory: `Finish`/drain
must occur before the operator publishes `attention_slots` and returns.

## What is intentionally absent

- no private payload arena;
- no private-ready table;
- no cross-forward/layer/replay ownership;
- no arena release, ABA tag, quarantine, or terminal ownership protocol;
- no fixed `31 + 1` AIV partition;
- no host/GIL submission in the hot path.

The only temporary state is invocation-local candidate metadata and queue
descriptor scratch.  It is discarded after all CQ drains complete.

## Correctness gate

The source-derived C8 kernel keeps its score arithmetic and Top-K output
unchanged.  The required comparisons are:

1. official C8 `sparse_indices` and `score_source_lens` are bitwise equal;
2. P9 `attention_slots` point only to existing hits or finished direct-resident
   copies;
3. the exact winner source order is unchanged;
4. no `-1`, duplicate destination, pending queue, or unfinished recall remains
   when QSFA starts.

`vllm_ascend.dsa_offload.c8_speculative_resident` is the reference slot planner
used to test the device resolver independently of CANN compilation.
