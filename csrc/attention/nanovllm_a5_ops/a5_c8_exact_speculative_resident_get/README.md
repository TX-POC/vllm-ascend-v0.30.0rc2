# A5C8ExactSpeculativeResidentGet — P9 source-derived target scaffold

This directory is the clean P9 replacement for the retired P7/P8 private
arena experiments. It deliberately does **not** register an OPP or alter a
product route yet. Its purpose is to freeze the source boundary and ABI that
the next CANN implementation must satisfy before any C8 source is copied or
compiled.

## One invocation, fixed resident slots

The P9 graph ABI keeps the two meanings of a resident destination separate.
`canonical_resident_slots` is an `int32[pool_size, 2048]` table of stable
*logical* slots: it is what QSFA receives in `attention_slots`. The execution
row can be permuted, so a row first resolves `pool = req_pool_entries[row]`
and then reads `canonical_resident_slots[pool, ordinal]`. Direct-HCCP needs a
physical packed-output row instead; it derives that independently from the
execution-row `resident_block_table` as
`resident_block_table[row, logical_slot / 128] * 128 + logical_slot % 128`.

`active_row_count` is a graph-visible `int32[1]` padding gate. Only rows in
`[0, active_row_count)` may inspect P9 pool/table/control state. Captured
graph padding can reuse a pool row, so a zero candidate count is not a safe
substitute for this gate. Admission validates the active prefix's mapping and
that fixed logical slots are disjoint from persistent resident slots; the
device does not perform a candidate-by-object collision scan.

The fixed slots belong to this layer invocation and are already registered in
the normal resident KV output allocation. They are not a private view and are
not an extension of the persistent source-to-slot map.

During the original C8 score scan, Stage A may assign a promising cold object
to one unique fixed-table *ordinal* and issue an asynchronous Direct-HCCP
post. Only the invocation-local `source -> ordinal` map is written,
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

The fixed table cannot reuse an entry while an earlier post may still write it.
Therefore Stage B drains *all* initial candidate posts before reusing a
nonwinner entry for exact recall. The only retained map is `source -> ordinal`.
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
The checked-in `ops.json` is the atomic active P9 schema: official inputs
`0..8`, fourteen P9 inputs (`9..22`), and six outputs. It is intentionally
not compatible with the retired 21-input affine-layout variant. The source
overlay, host tiler, and Torch bridge must all move together; a partial ABI is
rejected before CANN compilation.

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
