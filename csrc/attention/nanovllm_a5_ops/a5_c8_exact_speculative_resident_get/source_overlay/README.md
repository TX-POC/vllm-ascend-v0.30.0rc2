# P9 source overlay boundary

This directory intentionally contains no copied C8 source and no generated
CANN files. The build helper added with the real implementation must:

1. begin from the exact checked-out
   `csrc/attention/vllm_quant_lightning_indexer` source tree;
2. overlay only the P9 hook and direct-resident contract header;
3. reject a source root whose Git head differs from the requested head; and
4. create a new output directory keyed by both the vLLM and KVAF heads.

`apply_p9_overlay.py` is the checked-in, anchor-checked implementation of
this boundary.  It runs only after a fresh copy is created and refuses to
proceed if an upstream source anchor has changed.  Its first closure is
intentionally compile-disabled: it inserts a no-op score-seam macro after the
completed `BatchMulWeightAndReduceSum` and before the normal score-GM
`DataCopyPad`.  It calculates the complete `ProcessTopK` function digest
before and after transformation and fails if those byte-for-byte contents
differ.

The active target will replace that no-op with CandidateCompact and the
invocation-local PostOnly executor in one ABI-complete change.  It must not
modify score UB, score GM, Top-K working storage, or final C8 index ordering.

The direct-resident target is non-buildable until it has its own KVAF ABI and
is not allowed to include the retired streaming, private-view, P7, or P8
trees. Static checks enforce that source-closure rule before an OPP build is
added.
