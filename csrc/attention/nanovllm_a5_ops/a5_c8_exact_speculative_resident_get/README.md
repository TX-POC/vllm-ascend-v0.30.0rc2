# A5C8ExactSpeculativeResidentGet — P9 source-derived target scaffold

This directory is the clean P9 replacement for the retired P7/P8 private
arena experiments. It deliberately does **not** register an OPP or alter a
product route yet. Its purpose is to freeze the source boundary and ABI that
the next CANN implementation must satisfy before any C8 source is copied or
compiled.

## One invocation, one canonical resident range

For each flattened C8 row, the caller supplies one `int32[4]` canonical
layout: `[first_slot, slots_per_row, row_stride, row_count]`.  It declares a
fixed range of 2048 normal resident slots per row; `row_stride` may be larger
than 2048 when the layer allocation contains padding.  The ranges belong to
this layer invocation and are already registered in the normal resident KV
output allocation. They are not a private view and are not an extension of
the persistent source-to-slot map.

During the original C8 score scan, Stage A may assign a promising cold object
to one unique entry in that row's fixed range and issue an asynchronous
Direct-HCCP post. Only the invocation-local `source -> slot` map is written,
and only after the descriptor is fully published and the PostOnly call has
accepted it. It is not a persistent cache directory or a reusable arena state
machine. The original C8 score calculation and Top-K algorithm stay unchanged.
After exact Top-K completes, Stage B maps every final winner to
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
Therefore Stage B drains *all* initial candidate posts before reusing a
nonwinner entry for exact recall. The only retained map is `source -> slot`.
Stage B derives a local 2048-bit used-slot bitmap from exact winners, then
chooses remaining fixed entries for recall. There is no inverse map, slot
release, arena reacquire, or cross-forward identity. This is an
invocation-local ordering rule, not an ABA protocol.

## Source boundary

The real target will be generated from the official C8 source at:

`csrc/attention/vllm_quant_lightning_indexer/op_kernel/arch35/quant_lightning_indexer_service_vector.h`

The only score-time insertion point is after
`vector1::BatchMulWeightAndReduceSum(...)` and before the unchanged
`DataCopyPad(scoreGm, ...)`. It may observe exact per-tile scores and compact
candidate seeds, but may not change score UB, score GM, or `ProcessTopK`.

The exact Stage-A/Stage-B *target ABI* and completion condition are in
[`stage_a_b_contract.json`](stage_a_b_contract.json). The constexpr layout
used by the eventual AICore/SIMT source is in
[`a5_c8_exact_speculative_resident_contract.h`](op_kernel/arch35/a5_c8_exact_speculative_resident_contract.h).
The checked-in `ops.json` intentionally remains the 9-input/2-output official
C8 closure until the kernel, host tiler, and torch bridge can be switched in
one atomic ABI change; it must not be mistaken for a live P9 transport route.

The existing official `outInvalidBuf_` is 8 KiB at Top-2048. Its P9 layout is
phase-exclusive. During score scan, the first 2080 bytes hold two 64-seed
batches plus one 16-byte queue-lease receipt and one 16-byte source-slot
reservation receipt. After every initial wave is terminal, Stage B reuses the
same bytes as one 256-seed exact-recall wave (4096 bytes), a 256-byte
winner-slot bitmap, and two new 16-byte receipts: 4384 bytes total. It adds no
P9 `TBuf`. Before Stage B reuses a nonwinner slot, all initial CQ waves must
be terminal; before the operator returns, all recall CQ waves must also be
terminal.

## Build boundary

`build_p9_exact_resident_opp.sh` is the only P9 OPP builder. It stages the
exact official C8 source and the exact KVAF direct-resident ABI into a fresh
directory keyed by both Git SHAs. It never uses an existing OPP, bridge, or
CMake cache, and never routes through the retired private-slot trees.

The builder deliberately refuses a normal CANN build until the source overlay
manifest exists. It parses the real score-hook, Post/CQ executor, and
terminalizer headers while both P9 macros are disabled, catching CANN-domain
and KVAF ABI errors without changing official C8 behavior. This closure build
is not an asynchronous-Get claim; active transport arrives only with the one
atomic graph ABI change, not through a substitute P7/P8-style probe.
