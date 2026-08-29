# Source-matched official C8 reference bundle

This directory is a validation-only companion for
`A5StreamingQuantLiGet`.  It builds the **unmodified** current-checkout
sources:

* `csrc/attention/vllm_quant_lightning_indexer`;
* `csrc/attention/vllm_quant_lightning_indexer_metadata`.

The resulting OPP is installed into a caller-selected, isolated directory and
is launched through `nanovllm_official_c8_reference`.  It deliberately does
not import `vllm_ascend`, `vllm_ascend_C`, or an editable vLLM-Ascend package.
That matters for P3: an installed package from a different worktree is not a
valid mathematical reference for an operator built from the current source.

## What is and is not built

`build_reference_opp.sh` calls the upstream CANN operator builder with exactly
the two official C8 op names.  It does **not** run `pip install -e`, does not
compile `vllm_ascend_C`, and does not select a product route.

The bridge is a small typed ACLNN caller copied in behaviour from the two C8
functions in `csrc/torch_binding.cpp`.  It exposes the same input ABI under a
private namespace:

* `torch.ops.a5_official_c8_reference.npu_vllm_quant_lightning_indexer_metadata`
* `torch.ops.a5_official_c8_reference.npu_vllm_quant_lightning_indexer`

The reference bridge resolves `NANOVLLM_A5_REFERENCE_C8_OPAPI_LIB`; the P3
target bridge keeps `NANOVLLM_CUST_OPAPI_LIB`.  The two DSOs therefore have
separate resolver domains even though they run in the same Python process.
The adapter puts both vendor paths in `ASCEND_CUSTOM_OPP_PATH` before the
first ACLNN call, launches the reference with its DSO, then launches P3 with
its own DSO.

## Provenance gate

The OPP builder writes `P3_REFERENCE_PROVENANCE` in its vendor directory.  It
contains the exact source checkout and Git revision used for the two official
operators.  At runtime the target adapter rejects the bundle unless this
revision matches `P3_REFERENCE_EXPECTED_OPP_SOURCE_HEAD` when that variable
is set.  The paired helper
`tests/resolve_streaming_li_get_artifacts.sh` reads this value directly from
the selected artifact manifests and sets it with the target artifact's source
head.  It also records the Python runner revision separately as
`P3_RUNNER_EXPECTED_HEAD`.  A test-only commit therefore cannot masquerade as
a stale compiled OPP, while selecting the wrong artifact still fails closed.

This component is only for a bitwise C8 reference test.  It does not report
H2D bandwidth and it does not make any product execution-path change.
