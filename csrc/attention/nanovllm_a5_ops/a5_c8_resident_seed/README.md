# A5C8ResidentSeed — isolated resident threshold probe

`A5C8ResidentSeed` is the first, deliberately isolated node of the resident-threshold prefetch path.  It is an Ascend 950 source-derived copy of the official C8 Lightning Indexer closure.  It computes the normal C8 `sparse_indices` and, after the same final C8 Top-2048 reduction, emits one resident score threshold per selection row.

It is **not** a product route, does not register itself in the normal `ALL_OP_NAMES` collection, and does not alter HBM placement, `HBMBlockTable`, cache slots, or KVAF Get behavior.

## Why this is a real threshold rather than a proxy

The copied AIV code calls the official `BatchMulWeightAndReduceSum` score path.  The resulting `uint16` sortable scores are reduced by the same official `LiTopKVF` used to produce `sparse_indices`.  `LiTopKVF` leaves its final K-th histogram boundary in `nkValueLocal[0]`; this prototype writes its low 16 bits into `resident_tau` only after the final reduction completes.

Therefore, for a valid row, `resident_tau[r] & 0xffff` is exactly the C8-sortable score of that row's 2048th selected resident key.  It is not a dequantized score and is not recomputed from Top-K indices.

Equal-score ties can make `score >= tau` contain more than 2048 speculative candidates.  That is intentional: later full C8 Top-K remains authoritative, while an early prefetch may over-fetch tied candidates but must not miss a score strictly above the boundary.

## Explicit resident-index ABI

The operator intentionally requires a **separate resident index view**:

| ABI tensor | Required contents for `layout_key=PA_BSND` |
| --- | --- |
| `resident_index_key` | C8 FP8 index key, `[physical_blocks, block_size, 1, 128]` |
| `resident_index_key_dequant_scale` | C8 FP32 key scale aligned to that index key, `[physical_blocks, block_size, 1]` |
| `resident_index_block_table` | `int32 [row_count, max_blocks]`, mapping each logical resident block to a physical **index-key** block |
| `resident_seq_lengths` | `int32 [row_count]`, the valid resident-index length for each selection row |
| `metadata` | existing official C8 split-core metadata (`[1024]`) |

The 656-byte packed KV representation is a QSFA payload format, **not** an LI key.  It must never be passed as `resident_index_key`.  Current runtime state does not yet maintain this resident index view or an inverse token-to-resident-block directory; that maintenance is a prerequisite for product wiring and is purposely outside this probe.

`query`, `weights`, and `query_dequant_scale` retain the official C8 ABI.  The default is the decode-relevant `layout_query=TND`, `layout_key=PA_BSND`.  The copied tiling closure still validates official compatible layout variants, but this probe is Ascend-950-only.

### Why the JSON says `int8` although the operator consumes FP8

The CANN 9.1 `msopgen` parser accepts `int8` in its generic JSON `TypeRange`,
but rejects the JSON spelling `float8_e4m3fn`.  For that bootstrap-only
metadata, `ops.json` therefore declares `query` and `resident_index_key` as
`int8`; the generic `OpDef` uses the same generator-facing type.  This is
exactly the official/P3 C8 pattern.  It is **not** a cast or a conversion: the
Ascend-950 `AICoreConfig` in
`op_host/a5_c8_resident_seed_def.cpp` requires `DT_FLOAT8_E4M3FN` for both
tensors, and the independent Torch bridge rejects every dtype except
`torch.float8_e4m3fn`.  The generated JSON can therefore be understood as a
CANN-generator compatibility envelope; the compiled target ABI and the bytes
seen by the C8 score kernel remain FP8 E4M3FN.

## Outputs and graph contract

`resident_tau` and `tau_valid` are flat `int32 [selection_row_count]` tensors.  Row `r` corresponds to `sparse_indices.reshape(-1, 2048)[r]`.

* `resident_tau[r]`: low 16 bits contain the unsigned sortable C8 score threshold; upper bits are zero.
* `tau_valid[r] == 1`: the row had at least 2048 valid resident keys and `resident_tau[r]` is exact.
* `tau_valid[r] == 0`: the row is short or otherwise invalid; `resident_tau[r] == 0` and callers must not speculate from it.

All pointers and shapes must be stable before graph capture.  During replay the kernel writes only the per-invocation output tensors.  It does not publish a persistent mapping or mutate managed HBM slots.

## Isolated build only

Run `build_isolated_a5_c8_resident_seed.sh` after sourcing the CANN 9.1 environment.  It creates a fresh, caller-selected staging directory, invokes `msopgen` for only `A5C8ResidentSeed`, and installs the resulting OPP below that same staging directory.  It neither calls the normal `nanovllm_a5_ops` build nor builds vLLM-Ascend.

The isolated project also stages a local diagnostics-only compatibility header
as both `err/ops_err.h` and `log/log.h`.  Those are normally supplied by the
repository-wide vLLM-Ascend include graph, which an isolated `msopgen` project
does not inherit.  The compatibility layer preserves all source-derived host
validation return paths; it does not change C8 math or device code.

The static contract check can run without an NPU:

```bash
python3 csrc/attention/nanovllm_a5_ops/tests/test_a5_c8_resident_seed_contract.py
```
