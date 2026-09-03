# P10: A5 W4A4C8 native streaming-C8 boundary

P10 targets exactly one product route: the A5 DSA W4A4C8 Lightning Indexer
called as `torch_npu.npu_quant_lightning_indexer`.

Its mixed tensor ABI is part of the one W4A4C8 product contract:

| field | required representation |
| --- | --- |
| query / key | `float8_e4m3fn` |
| indexer weights | `bfloat16` |
| query / key dequant scales | `float32` scale metadata |
| sparse result | `int32` Top-K source indices |

The scale tensors being FP32 does **not** create an FP32 product route. P10
does not widen BF16 weights to FP32 and does not compare against an FP32
operator as a mathematical oracle.

## What is deliberately rejected

The checked-in `VllmQuantLightningIndexer` declares an Ascend-950 ABI with
FP32 weights and scales. `LightningIndexerQuant` declares INT8 Q/K with FP16
weights/scales and has no Ascend-950 configuration. Neither matches the
product W4A4C8 runtime call, so neither may be copied, modified, compiled, or
used as the source-matched oracle for Stage A.

The runner fails closed until all of the following identify the same
implementation:

1. the runtime primary callable is present (`torch_npu.npu_quant_lightning_indexer`);
2. the runtime input ABI matches the W4A4C8 contract above;
3. the provider OPP/library is identified; and
4. an explicit source root and Git revision are proven to build that provider.

Until then, a local generic source is diagnostic evidence only. There is no
Stage-A OPP build and no FP32 comparison to mislabel as product correctness.

## Target pipeline after source provenance is resolved

```text
W4A4C8 native C8: exact score tiles + original exact Top-K
        │
        ├── unchanged Top-K result: the only mathematical truth
        │
        └── independent score-window DMA → raw multi-row receipt ring
                                              │
                                              ▼
Stage B device executor: ExactScoreScan → CandidateCompact
    → dynamic HCCP queue lease → PostOnly → nonblocking CQ drain
                                              │
                                              ▼
exact final Top-K → recall missing winners into fixed resident slots
                                              │
                                              ▼
all final slots ready → QSFA
```

Stage A publishes no HCCP descriptor and does not alter original Top-K. It
uses a separate score-window surface, not the native operator's opaque,
reused score workspace. Stage B owns the bounded queue and candidate
backpressure. There is no 31+1 physical-core partition: queues are fixed
resources leased dynamically by executor work items.

For one forward, resident slots are monotonic and fixed. False-positive
speculation may move extra data but may not occupy a final winner's canonical
slot. The forward reaches QSFA only after all exact final winners are ready.
