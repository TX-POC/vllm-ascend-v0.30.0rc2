# No ambient build; one explicit isolated CANN route

The historical filename is retained so old scripts fail closed rather than
selecting this source through a recursive CMake search. It does **not** mean
that Stage-A is a source-only placeholder.

Build the real isolated operator only through:

```bash
bash build_stage_a_w4a4c8_score_receipt_opp.sh
```

That script requires an exact clean source SHA, a fresh package root, and an
explicit CANN third-party cache. It invokes `msopgen`, retains the generated
CANN package/CMake machinery, removes the one generated generic host stub,
and verifies that the checked-in W4A4C8 `OP_ADD` is the unique registration.
It never reuses a CMake cache, OPP, bridge, or normal vLLM-Ascend build tree.

The registered entry checks all of the following before CANN is invoked:

1. one graph-owned full-native `uint16 [B, ceil(256/H), S]` score surface;
2. canonical `int32 [B, 1, 2048]` Top-K storage, retained byte-for-byte by
   the imported `ProcessTopK` path;
3. receipt ring/state with exactly `B * (S / 128)` records and five `int64`
   descriptor words each;
4. graph-preinitialized `receipt_state=FREE` and
   `forward_control[0]=OPEN` before Stage-A/Stage-B concurrent launch; and
5. a dedicated producer seal in `forward_control[0]`, written only after the
   mixed AIC/AIV terminal barrier.

The trailing alias outputs are graph-visible mutation handles, not a normal
Stage-A-to-Stage-B completion edge. The product graph must launch Stage-B from
the same cold-bound input aliases and let it poll `READY` records and the
producer seal.
