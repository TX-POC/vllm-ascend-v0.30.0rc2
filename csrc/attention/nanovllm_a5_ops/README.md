# Ascend 950 nano-vLLM offload_split 算子

本仓库是独立的 Ascend 950 / CANN 9.1 decode KV-cache 卸载算子工程。W4A8 路径包含 BF16 `lightning_indexer_decode_update`、`kvcache_scatter_copy` 和 `sparse_and_tail_attention`；W4A4C8 路径使用官方 A5 C8 LightningIndexer、request-pool update、packed-C8 SCATTER 和原生 C8 QSFA。构建与运行不读取任何参考仓源码，也不包含 nano-vLLM 引擎、融合算子或 MTP。

## 来源

- LIDU：基于 `ops_li_update_a5@0362e7e` 的生产 P5 内核，增加 request pool、逐请求 C、动态 pool 行跨度、mutable alias 和 caller-owned out。
- SCATTER：基于 `ops_dsa_offload_a5@01f2065` 已验证的 Ascend 950 swapped-memory DRAM→HBM 路径。
- SFA：基于 `vllm-ascend-v0.23.0-custom@6af99b372` 的官方 Arch35 SFA叠加 sparse+tail 语义。官方接口说明见 [aclnnSparseFlashAttention](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/attention/sparse_flash_attention/docs/aclnnSparseFlashAttention.md)。
- W4A4C8 Indexer：使用 A5 官方 `torch_npu.npu_quant_lightning_indexer`，采用 FP8 E4M3 query/key、BF16 weights、FP32 query/key scale、`TND/PA_BSND`、`sparse_count=2048` 和 `sparse_mode=3`；C8 LIDU 接续完成 request-pool hit/miss、淘汰和更新。
- W4A4C8 Attention：packed KV ABI 与 A5 原生 `npu_kv_quant_sparse_flash_attention` 对齐。ModelSlim 的 GLM-5.1 W4A4C8 量化说明见 [GLM-5 量化 README](https://gitcode.com/Ascend/msmodelslim/blob/26.1.0/example/GLM-5/README.md)。

## W4A8 算子一览表

| 公开入口 | 实际执行 | 作用 | 定位 |
| --- | --- | --- | --- |
| `lidu_decode_update` / `lidu_decode_update_out` | `LightningIndexerDecodeUpdateA5` | BF16/FP16 LightningIndexer top-2048 与 request-pool 更新；`_out` 写入 caller-owned buffer | eager / 稳定图主链 |
| `scatter_copy` | `A5KvcacheScatterCopy` | 按 miss-prefix 将分离的 BF16/FP16 CKV、KPE 从 swapped-memory DRAM 搬到 HBM | 稳定图主链 |
| `sparse_and_tail_attention` | `A5SparseAndTailAttention` | 对 top-2048 sparse slots 与 tail 执行 BF16/FP16 MLA | 稳定图主链 |

## W4A4C8 算子一览表

| 公开入口 | 实际执行 | 作用 | 定位 |
| --- | --- | --- | --- |
| `lidu_decode_update_c8` / `lidu_decode_update_c8_out` | 官方 `npu_quant_lightning_indexer` + request-pool update | C8 top-2048 与 request-pool 更新；`_out` 写入 caller-owned buffer | eager / 稳定图主链 |
| `packed_scatter_copy` / `packed_scatter_copy_out` | `A5PackedKvcacheScatterCopy` | 整行搬运 656-byte packed KV，并生成 sparse+tail slots 和 resident length；`_out` 写入 caller-owned metadata | eager / 稳定图主链 |
| `sparse_and_tail_attention_c8` | Python custom op → 原生 `npu_kv_quant_sparse_flash_attention` | 对 packed C8 KV 的 top-2048 sparse slots 与 tail 执行量化 MLA | 稳定图主链 |

## 接口

共同语义：`B` 是 decode batch，`C=cache_tokens[b]` 是请求固定的 HBM token 预算，`candidate_lens[b]` 是参与稀疏选择的 prefill 满块 token 数。`cache_slots_pool[req_pool_entries[b], token_id]` 保存 source token 到 HBM slot 的映射，`-1` 表示未缓存。LIDU 输出中前 `miss_counts[b]` 项为 miss，SCATTER 只搬运这段；全部 2048 个 `destination_slots` 都供 Attention 使用。稳定图使用 caller-owned `_out` 接口。

### W4A8：BF16/FP16 KV

链路：`lidu_decode_update_out → scatter_copy → sparse_and_tail_attention`。

```python
torch.ops.nanovllm_dsa.lidu_decode_update(
    query,                  # bf16/fp16[B, INDEX_HEADS, 128]，INDEX_HEADS=32|64
    key,                    # bf16/fp16[INDEX_BLOCKS, 128, 1, 128]
    weights,                # bf16/fp16[B, INDEX_HEADS]
    req_pool_entries,       # int32[B]，batch row -> request-pool row
    cache_slots_pool,       # int32[POOL_SIZE, SOURCE_CAPACITY]，in/out
    cache_tokens,           # int32[B]，逐请求 C；C=0 时该请求 no-op
    candidate_lens,         # int32[B]
    block_table,            # int32[B, SOURCE_CAPACITY/128]
) -> (
    source_ids,             # int32[B,1,2048]，miss-prefix + hit-suffix
    destination_slots,      # int32[B,1,2048]，与 source_ids 对齐
    miss_counts,            # int32[B]
    cache_slots_alias,      # cache_slots_pool alias
)

torch.ops.nanovllm_dsa.lidu_decode_update_out(
    query, key, weights, req_pool_entries, cache_slots_pool,
    cache_tokens, candidate_lens, block_table,
    source_ids,             # caller-owned int32[B,1,2048]，in/out
    destination_slots,      # caller-owned int32[B,1,2048]，in/out
    miss_counts,            # caller-owned int32[B]，in/out
) -> (source_ids_alias, destination_slots_alias, miss_counts_alias, cache_slots_alias)

torch.ops.nanovllm_dsa.scatter_copy(
    hbm_kpe,                # bf16/fp16[HBM_BLOCKS,128,64]，in/out
    hbm_ckv,                # bf16/fp16[HBM_BLOCKS,128,512]，in/out
    dram_kpe,               # swapped-memory bf16/fp16[DRAM_BLOCKS,128,64]
    dram_ckv,               # swapped-memory bf16/fp16[DRAM_BLOCKS,128,512]
    hbm_block_table,        # int32[B,HBM_MAX_BLOCKS]
    dram_block_table,       # int32[B,DRAM_MAX_BLOCKS]
    source_token_ids,       # int32[B,COPY_CAP]，取 LIDU miss-prefix
    destination_slots,      # int32[B,COPY_CAP]
    copy_counts,            # int32[B]，即 miss_counts
) -> (hbm_kpe_alias, hbm_ckv_alias)

torch.ops.nanovllm_dsa.sparse_and_tail_attention(
    query,                  # bf16/fp16[B,Q_HEAD,512]，1<=Q_HEAD<=64
    key,                    # bf16/fp16[HBM_BLOCKS,128,1,512]
    value,                  # 必须与 key alias
    sparse_slots,           # int32[B,1,2048]，即 LIDU destination_slots
    cache_tokens,           # int32[B]
    block_table,            # int32[B,HBM_MAX_BLOCKS]
    actual_seq_lengths_query, # cumulative int32[B]，decode 为 [1,...,B]
    actual_seq_lengths_kv,  # int32[B]，HBM resident 长度；C>0 时为 C+tail
    query_rope,             # bf16/fp16[B,Q_HEAD,64]
    key_rope,               # bf16/fp16[HBM_BLOCKS,128,1,64]
    scale_value,            # float
) -> attention_out          # bf16/fp16[B,Q_HEAD,512]
```

### W4A4C8：packed C8 KV

链路：`lidu_decode_update_c8_out → packed_scatter_copy_out → sparse_and_tail_attention_c8`。每个 packed KV token 固定为 656 bytes：`512 FP8 E4M3 + 64 BF16 RoPE + 4 FP32 scales`。

```python
torch.ops.nanovllm_dsa.lidu_decode_update_c8(
    query,                  # float8_e4m3fn[B,INDEX_HEADS,128]，INDEX_HEADS=32|64
    key,                    # float8_e4m3fn[INDEX_BLOCKS,128,1,128]
    weights,                # bf16[B,INDEX_HEADS]
    query_dequant_scale,    # fp32[B,INDEX_HEADS]
    key_dequant_scale,      # fp32[INDEX_BLOCKS,128,1]
    actual_seq_lengths_query, # cumulative int32[B]，decode 为 [1,...,B]
    req_pool_entries,       # int32[B]
    cache_slots_pool,       # int32[POOL_SIZE,SOURCE_CAPACITY]，in/out
    cache_tokens,           # int32[B]
    candidate_lens,         # int32[B]
    block_table,            # int32[B,SOURCE_CAPACITY/128]
) -> (
    source_ids,             # int32[B,1,2048]，miss-prefix + hit-suffix
    destination_slots,      # int32[B,1,2048]
    miss_counts,            # int32[B]
    cache_slots_alias,
)

torch.ops.nanovllm_dsa.lidu_decode_update_c8_out(
    query, key, weights, query_dequant_scale, key_dequant_scale,
    actual_seq_lengths_query, req_pool_entries, cache_slots_pool,
    cache_tokens, candidate_lens, block_table,
    source_ids,             # caller-owned int32[B,1,2048]，in/out
    destination_slots,      # caller-owned int32[B,1,2048]，in/out
    miss_counts,            # caller-owned int32[B]，in/out
) -> (source_ids_alias, destination_slots_alias, miss_counts_alias, cache_slots_alias)

torch.ops.nanovllm_dsa.packed_scatter_copy_out(
    hbm_packed_kv_bytes,    # int8 view[HBM_BLOCKS,128,1,656]，in/out
    dram_packed_kv_bytes,   # swapped-memory int8 view[DRAM_BLOCKS,128,1,656]
    hbm_block_table,        # int32[B,HBM_MAX_BLOCKS]
    dram_block_table,       # int32[B,DRAM_MAX_BLOCKS]
    source_token_ids,       # int32[B,1,2048] 或 int32[B,2048]
    destination_slots,      # 同 source_token_ids
    copy_counts,            # int32[B]，即 miss_counts
    cache_tokens,           # int32[B]
    candidate_lens,         # int32[B]
    actual_seq_lengths_kv,  # int32[B]，原始逻辑 KV 长度
    max_tail_tokens,        # int，固定图的 tail 容量
    attention_slots,        # caller-owned int32[B,1,2048+max_tail_tokens]
    resident_seq_lengths,   # caller-owned int32[B]
) -> (
    hbm_packed_kv_alias,
    attention_slots_alias,  # C>0: top2048 slots + [C,C+tail)；C=0: [0,actual_len)
    resident_lengths_alias, # C>0: C+tail；C=0: actual_len
)

torch.ops.nanovllm_dsa.sparse_and_tail_attention_c8(
    query,                  # bf16/fp16[T,Q_HEAD,576]，1<=Q_HEAD<=64
    packed_kv,              # float8_e4m3fn view[HBM_BLOCKS,128,1,656]
    attention_slots,        # int32[T,1,2048+max_tail_tokens]
    hbm_block_table,        # int32[B,HBM_MAX_BLOCKS]
    actual_seq_lengths_query, # cumulative int32[B]
    resident_seq_lengths,   # int32[B]，直接使用 packed SCATTER 输出
    scale_value,            # float
) -> attention_out          # bf16/fp16[T,Q_HEAD,512]
```

非 `_out` 的 `packed_scatter_copy` 参数相同，但由算子分配 `attention_slots` 和 `resident_seq_lengths`。C8 LIDU 的组合接口内部完成官方 A5 Quant LightningIndexer 和 request-pool 更新。全部框架接口提供 Fake/Meta；LIDU、SCATTER 的 schema 显式声明 mutable alias。

## 约束

- 仅面向 GLM-5.1 W4A8/W4A4C8 decode，`q_seq_len=1`，block size 为 128。
- LIDU 支持 `q_head=32|64`；SFA 仅支持 `q_head=1..64`，门禁使用 8。
- `C=0` 时 LIDU no-op，SFA 计算全部有效 KV；`C>0` 时 SFA 计算 2048 个 sparse slots 与 `[C, actual_kv_len)` tail。
- source token ID 为 18 bit，`SOURCE_CAPACITY <= 262144`；LIDU slot 为 14 bit，block-aligned `C <= 16256`。当前预算 `3072/6144/8192/12288` 均受支持。
- `req_pool_entries` 在活跃 batch 内必须唯一且位于 pool 范围内；非零 C 的 pool 行必须恰有 C 个唯一 slots。
- C8 Indexer 使用独立的 `float8_e4m3fn[blocks,128,1,128]` key cache 和 `float32[blocks,128,1]` scale cache，不再复用 BF16 Indexer。C8 Attention cache 每个 token 的 656 bytes 固定为 `512 FP8 E4M3 + 64 BF16 RoPE + 4 FP32 scales`，SCATTER 必须整行搬运。
- C8 LIDU 的 query/key 接口位于官方 GLM 预处理之后：调用方须先完成 RoPE、归一化 128×128 Hadamard，再执行 FP8 E4M3 动态量化并传入对应 FP32 scale；C8 UT 也按该顺序构造输入。
- `max_tail_tokens` 是 full-decode-only 的静态 capture 容量；实际 tail 为 `actual_seq_lengths_kv-candidate_lens`。`C>0` 时 Attention 索引为 2048 个 LIDU slots 加 `[C,C+tail)`，`C=0` 时为 `[0,actual_len)`。

## 环境与构建

从仓库根目录执行：

```bash
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_A5_INSTALL_OPP_PATH
unset NANOVLLM_CUST_OPAPI_LIB
unset A5_SOC_VERSION
unset SOC_VERSION
unset CANN_INSTALL_PATH
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export CANN_INSTALL_PATH=$ASCEND_HOME_PATH
source "$ASCEND_HOME_PATH/set_env.sh"
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH
export SOC_VERSION=ascend950
export NANOVLLM_A5_OPS_PYTHON=python3
export NANOVLLM_A5_OPS_BUILD_JOBS=64
```

```bash
bash build.sh
```

## 正确性与图回放

```bash
python3 tests/test_api_meta.py
```

```bash
python3 tests/test_lidu.py --device npu:0 --mode check --heads 32,64 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --miss-ranges 0:300 --seed 7
```

该 LIDU 命令还会强制覆盖不同 candidate length、`C=0/2048/3072/6144/8192/12288/16256`、零 miss、2048 miss、乱序 pool entries、重复更新、inactive pool guard 和 caller-owned out。

```bash
for count in 0 1 100 300 2048; do python3 tests/test_scatter_copy.py --device npu:0 --batch-size 24 --source-len 65536 --hbm-slots 4096 --copy-cap 2048 --copy-min "$count" --copy-max "$count" --warmup 3 --iters 10 --seed 7; done
```

SCATTER 使用 `empty_with_swapped_memory` 创建真实 DRAM tensor；每次正确性调用前 poison HBM 目标，并验证 CKV、KPE、随机 block tables 和未触碰 guard。

```bash
python3 tests/test_sparse_and_tail_attention.py --device npu:0 --mode check --heads 8 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --tail-tokens 64 --seed 7
```

SFA check 会先验证 2048-token smoke、dense `C=0`、四档 C 和 tail `0/1/64/127/257`，并与独立 CPU FP32 golden 比较。

```bash
python3 tests/test_offload_split_graph.py --device npu:0 --case pure-long --replays 4 --seed 7
```

```bash
python3 tests/test_offload_split_graph.py --device npu:0 --case mixed --replays 4 --seed 7
```

图门禁捕获 `lidu_decode_update_out → scatter_copy → sparse_and_tail_attention`；capture 为零 miss，replay 交替产生非零 miss，并校验输出地址、pool 更新、真实 DRAM→HBM 搬运和 attention golden。

## W4A4C8 门禁

C8 LIDU 直接用官方 A5 C8 LightningIndexer 作为 top-2048 基线，并覆盖 mixed C、乱序 request-pool、零/随机/2048 miss、重复更新和 caller-owned out：

```bash
python3 tests/test_lidu_c8.py --device npu:0 --mode check --heads 32,64 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --miss-ranges 0:300 --seed 7
```

packed SCATTER 使用真实 swapped memory；分配式与 caller-owned 两条路径都重新 poison HBM，并校验完整 656 bytes、guard、动态 tail metadata、`resident_seq_lengths` 和输出地址：

```bash
python3 tests/test_packed_scatter_copy_c8.py --device npu:0 --batch-size 24 --source-len 20096 --cache-tokens 6144 --tail-tokens 257 --max-tail-tokens 512 --copy-min 0 --copy-max 300 --warmup 10 --iters 100 --seed 7
```

C8 QSFA 使用独立 CPU FP32 golden 校验 FP8 latent、BF16 RoPE 和 FP32 scales 的 packed layout。首轮门禁固定 `q_head=8`；本仓库不支持 `q_head>64`：

```bash
for C in 3072 6144 8192 12288; do for tail in 0 1 64 127 257; do python3 tests/test_sparse_and_tail_attention_c8.py --device npu:0 --mode check --batch-size 24 --heads 8 --cache-tokens "$C" --tail-tokens "$tail" --max-tail-tokens 512 --seed 7; done; done
```

```bash
python3 tests/test_sparse_and_tail_attention_c8.py --device npu:0 --mode check --batch-size 1 --heads 8 --cache-tokens 0 --tail-tokens 2048 --max-tail-tokens 2048 --seed 7
```

图门禁捕获框架链路 `lidu_decode_update_c8_out → packed_scatter_copy_out → sparse_and_tail_attention_c8`；不依赖 capture 阶段执行，先用一次 replay 验证零 miss 和完整输出写回，再恢复初始 pool 进行多次非零 miss replay：

```bash
python3 tests/test_offload_split_c8_graph.py --device npu:0 --case pure-long --batch-size 2 --heads 8 --index-heads 32 --source-len 4096 --cache-tokens 3072 --tail-tokens 64 --max-tail-tokens 256 --miss-min 256 --miss-max 512 --replays 4 --seed 7
```

```bash
python3 tests/test_offload_split_c8_graph.py --device npu:0 --case mixed --batch-size 2 --heads 8 --index-heads 32 --source-len 4096 --cache-tokens 3072 --tail-tokens 64 --max-tail-tokens 256 --miss-min 256 --miss-max 512 --replays 4 --seed 7
```

## 性能矩阵

`q_head=8` 的三算子总时延需与相同输入下的来源版本对照，目标是不超过参考时延的 `1.10x`。

LIDU：

```bash
python3 tests/test_lidu.py --device npu:0 --mode bench --heads 32 --batch-sizes 1,4,8,12,16,24,32 --source-lens 12288,20096,65536,131072 --cache-tokens 6144 --miss-ranges 0:0,0:300,300:300 --warmup 10 --iters 100 --seed 7
```

C8 LIDU：

```bash
python3 tests/test_lidu_c8.py --device npu:0 --mode bench --heads 32 --batch-sizes 1,4,8,12,16,24,32 --source-lens 12288,20096,65536,131072 --cache-tokens 6144 --miss-ranges 0:0,0:300,300:300 --warmup 10 --iters 100 --seed 7
```

SCATTER：

```bash
for bs in 1 4 8 12 16 24 32; do for len in 12288 20096 65536 131072; do python3 tests/test_scatter_copy.py --device npu:0 --batch-size "$bs" --source-len "$len" --hbm-slots 8192 --copy-cap 2048 --copy-min 0 --copy-max 0 --warmup 10 --iters 100 --seed 7; python3 tests/test_scatter_copy.py --device npu:0 --batch-size "$bs" --source-len "$len" --hbm-slots 8192 --copy-cap 2048 --copy-min 0 --copy-max 300 --warmup 10 --iters 100 --seed 7; python3 tests/test_scatter_copy.py --device npu:0 --batch-size "$bs" --source-len "$len" --hbm-slots 8192 --copy-cap 2048 --copy-min 300 --copy-max 300 --warmup 10 --iters 100 --seed 7; done; done
```

SFA：

```bash
python3 tests/test_sparse_and_tail_attention.py --device npu:0 --mode bench --heads 8 --batch-sizes 1,4,8,12,16,24,32 --source-lens 12288,20096,65536,131072 --cache-tokens 6144 --tail-tokens 64 --warmup 10 --iters 100 --seed 7
```

## Profile

```bash
msprof --application="python3 tests/test_lidu.py --device npu:0 --mode profile --heads 32 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --miss-ranges 0:300 --profile-replays 4 --seed 7" --output=./profile_lidu
```

```bash
msprof --application="python3 tests/test_lidu_c8.py --device npu:0 --mode profile --heads 32 --batch-sizes 24 --source-lens 20096 --cache-tokens 6144 --miss-ranges 0:300 --profile-replays 4 --seed 7" --output=./profile_lidu_c8
```

```bash
msprof --application="python3 tests/test_scatter_copy.py --device npu:0 --batch-size 24 --source-len 65536 --hbm-slots 8192 --copy-cap 2048 --copy-min 0 --copy-max 300 --warmup 3 --iters 4 --seed 7" --output=./profile_scatter
```

```bash
msprof --application="python3 tests/test_sparse_and_tail_attention.py --device npu:0 --mode profile --heads 8 --batch-sizes 24 --source-lens 65536 --cache-tokens 6144 --tail-tokens 64 --profile-replays 4 --seed 7" --output=./profile_sfa_h8
```
