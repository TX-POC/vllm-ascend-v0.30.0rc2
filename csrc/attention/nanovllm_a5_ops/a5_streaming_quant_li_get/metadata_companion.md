# Metadata companion closure

`A5StreamingQuantLiGet` is a source-derived main C8 LI kernel. Just like the
official main kernel, it calls `SplitCoreByAICPU` unconditionally and therefore
requires a real `int32[1024]` metadata input.

The standalone graph must run the existing official AICPU companion
`VllmQuantLightningIndexerMetadata` on the same stream immediately before the
derived main op. Its source closure is:

```text
csrc/attention/vllm_quant_lightning_indexer_metadata/
  op_host/vllm_quant_lightning_indexer_metadata_infershape.cpp
  op_kernel_aicpu/vllm_quant_lightning_indexer_metadata_aicpu.cpp
  op_kernel_aicpu/vllm_quant_lightning_indexer_metadata_aicpu.json
```

The derived op does not fabricate metadata and does not accept a hidden
`torch_npu` side channel. Reuse of this official companion preserves the
metadata ABI and avoids registering a conflicting second AICPU op merely to
rename it. Isolated OPP staging must assert that its metadata op is registered
before it enables `A5StreamingQuantLiGet`; otherwise the prototype path is
fail-closed.
