# P9 source overlay boundary

This directory intentionally contains no copied C8 source and no generated
CANN files. The build helper added with the real implementation must:

1. begin from the exact checked-out
   `csrc/attention/vllm_quant_lightning_indexer` source tree;
2. overlay only the P9 hook and direct-resident contract header;
3. reject a source root whose Git head differs from the requested head; and
4. create a new output directory keyed by both the vLLM and KVAF heads.

The copied target must preserve `ProcessVec1` and `ProcessTopK` byte-for-byte
except for the one observation hook immediately after the completed score
reduction and before score GM publication. The hook is allowed to populate a
CandidateSeed and issue an invocation-local PostOnly. It must not modify
score UB, score GM, Top-K working storage, or final C8 index ordering.

The direct-resident target is non-buildable until it has its own KVAF ABI and
is not allowed to include the retired streaming, private-view, P7, or P8
trees. Static checks enforce that source-closure rule before an OPP build is
added.
