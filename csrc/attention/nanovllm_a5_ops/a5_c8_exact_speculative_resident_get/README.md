# A5C8ExactSpeculativeResidentGet — P9 source-derived target scaffold

This directory is the clean P9 replacement for the retired P7/P8 private
arena experiments. It deliberately does **not** register an OPP or alter a
product route yet. Its purpose is to freeze the source boundary and ABI that
the next CANN implementation must satisfy before any C8 source is copied or
compiled.

## One invocation, one canonical resident range

For each flattened C8 row, the caller supplies a fixed range of 2048 normal
resident slots. The range belongs to this layer invocation and is already
registered in the normal resident KV output allocation. It is not a private
view and is not an extension of the persistent source-to-slot map.

During the original C8 score scan, Stage A may assign a promising cold object
to one unique entry in that row's fixed range and issue an asynchronous
Direct-HCCP post. The original C8 score calculation and Top-K algorithm stay
unchanged. After exact Top-K completes, Stage B maps every final winner to
one of:

1. its existing persistent resident slot;
2. the fixed slot already posted for that winner; or
3. a fixed slot reused after the initial posts have reached terminal CQ, then
   filled by exact guaranteed recall.

Stage B returns only after all initial and recall posts for the final winner
set have reached terminal CQ. Its `attention_slots` then directly name the
normal resident KV allocation that QSFA consumes. Thus there is no
cross-forward private arena, deferred cleanup, or later remapping step.

The range cannot reuse an entry while an earlier post may still write it.
Therefore Stage B drains the initial candidate posts before reusing a
nonwinner entry for exact recall. This is an invocation-local ordering rule,
not an ABA protocol.

## Source boundary

The real target will be generated from the official C8 source at:

`csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch35/quant_lightning_indexer_service_vector.h`

The only score-time insertion point is after
`vector1::BatchMulWeightAndReduceSum(...)` and before the unchanged
`DataCopyPad(scoreGm, ...)`. It may observe exact per-tile scores and compact
candidate seeds, but may not change score UB, score GM, or `ProcessTopK`.

The exact Stage-A/Stage-B ABI and completion condition are in
[`stage_a_b_contract.json`](stage_a_b_contract.json). The constexpr layout
used by the eventual AICore/SIMT source is in
[`a5_c8_exact_speculative_resident_contract.h`](op_kernel/arch35/a5_c8_exact_speculative_resident_contract.h).

## Why this is not build-selectable yet

The existing `A5StreamingQuantLiGet` tree has old private-slot and lifecycle
closure. Copying it into P9 would reintroduce the failure mode we are trying
to eliminate. P9 remains a non-buildable source/ABI gate until the separate
KVAF direct-resident control ABI is available; then one isolated generated
OPP will be added from official C8 sources plus this contract. It will not
reuse an existing OPP, bridge, or CMake cache.
