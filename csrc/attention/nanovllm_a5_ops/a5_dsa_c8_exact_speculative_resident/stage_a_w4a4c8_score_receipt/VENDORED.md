# P10 W4A4C8 score-receipt Stage-A source base

This directory is the source-owned Stage-A base for the P10 product route. It
contains only the Ascend 950 W4A4C8 numerical Lightning Indexer closure needed
to compute the native score tiles and exact Top-K. It deliberately contains no
request-pool manager, cache mutation, resident-slot mapping, Direct-HCCP call,
or CQ logic.

## Provenance

- imported repository: `vLLM-ascend-DSA-01`
- imported repository commit: `603d96030f47deefc3d21408c5d4418c79251b70`
- imported source directory:
  `csrc/attention/nano/fused_li_manage_c8`
- upstream source provenance recorded by that directory: repository
  `nano-vllm-ascend-DeepseekV32`, branch `ops_a5`, commit
  `e00beba99c98b74f7522ed45c2383835a4ac8c80`

The imported source is a W4A4C8 numerical sibling, not a claim that the
installed `torch_npu.npu_quant_lightning_indexer` binary was compiled from this
directory. It must pass ordered parity on the same FP8 query/key, BF16 weights,
and FP32 scale metadata before P10 can be enabled.

## Exact boundary

`QuantLiPhase` is copied from the C8 closure and remains responsible for the
native score reduction and `ProcessTopK`. The Stage-A formal outputs are:

1. `canonical_sparse_indices`: the unmodified ordered native Top-2048;
2. `canonical_score_source_lens`: the valid source length for every row;
3. `score_surface`: a separately owned `uint16` score-tile surface;
4. `receipt_ring` and `receipt_state`: the descriptor and ownership tables
   consumed later by the device executor.

The entry rebinds the imported score-GM pointer once per logical row to the
caller-owned full-native score surface. `ProcessVec1` therefore writes its
original `uint16` scores directly to that surface and `ProcessTopK` reads the
same bytes with its original offsets. The only seam addition is immediately
after that original score DMA: a separate MTE3-to-Scalar fence, descriptor
write, DDR publication barrier, and `FILLING -> READY` receipt transition.
It does not copy/recompute score values, alter the original Top-K routine, or
reuse its event IDs.

`receipt_state` and `forward_control` are initialized by graph forward
preparation before Stage-A and Stage-B launch concurrently. Stage-A never
clears either table during `Process`; it only publishes its disjoint records
and writes `forward_control[0]=SEALED` after the final mixed AIC/AIV barrier.
The trailing output aliases are graph-visible mutation handles, not a normal
completion edge that would serialize Stage-B after C8.

The actual isolated CANN package route is
`build_stage_a_w4a4c8_score_receipt_opp.sh`; it uses a fresh msopgen project
and does not add this directory to any ambient vLLM CMake package.

## Deliberate exclusions

- No `A5FusedLiManageC8` request-pool logic.
- No cache input or cache output.
- No slot allocation, reuse, release, or ABA protocol.
- No Direct-HCCP, no host fallback, and no CQ wait.
- No P8/P9 closure code.

Subsequent changes may only tighten the proven direct-surface publication
contract or the independent Stage-B integration. They must not reintroduce a
manager, a second score DMA, a fixed-core partition, or an inline transport
closure into this numerical Stage-A directory.
