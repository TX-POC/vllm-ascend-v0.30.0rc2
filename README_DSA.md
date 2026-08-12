# vLLM-Ascend DSA KVC Offload for Ascend 950

基于 vLLM-Ascend v0.23.0，为 Ascend 950 (A5) 平台添加 DSA (Decode Sparse Attention) KV Cache Offload 能力。支持 W4A8 (BF16) 和 W4A4C8 (FP8 量化) 两种精度路径。

## 核心能力

### DSA KVC Offload

将 MLA decode 阶段的 KV Cache 从 HBM 卸载到 DRAM (swapped-memory)，通过三算子流水线实现稀疏注意力：

- **LIDU** (Lightning Indexer Decode Update)：每层每 decode step 选出 top-2048 重要 token，维护 request-pool hit/miss 状态
- **KSC** (KV Cache Scatter Copy)：将 miss token 的 KV 从 DRAM 搬回 HBM（单向 DRAM→HBM）
- **SFA** (Sparse Flash Attention)：对 top-2048 sparse slots + tail 区间执行 MLA attention

### 三项需求

| 需求 | 开关 | 说明 |
|------|------|------|
| Prefill 计算完后统一 dump 所有层 KVC 到 DDR | `enable_prefill_full_kv_offload: true` | 中间 chunk 跳过 dump，最后 chunk 完成后统一卸载 |
| Decode + MTP 时新增 token/draft/tail 不卸载到 DDR | `keep_decode_kv_in_hbm: true` | decode 行 zero dump，MTP draft 组不参与 dump |
| 关闭 MTP 时新增 token 也不卸载到 DDR | `keep_decode_kv_in_hbm: true` | 同上，不依赖 speculative_config |
| 图模式 + DP + EP 部署 | — | 支持 CUDA Graph、Data Parallel、Expert Parallel |

### W4A8 / W4A4C8 双精度

| 精度 | LIDU | KSC | SFA | Dump |
|------|------|-----|-----|------|
| W4A8 (BF16) | BF16 query/key | 分离 KPE(64)+CKV(512) | BF16 MLA | `full_block_dump` |
| W4A4C8 (FP8) | FP8 E4M3 query/key + FP32 scale | Packed int8 656B/token | C8 量化 MLA | `packed_full_block_dump` |

## 配置

```python
additional_config = {
    "dsa_sparse_config": {
        "enabled": True,
        "enable_prefill_full_kv_offload": True,
        "keep_decode_kv_in_hbm": True,
        "op_backend": "nanovllm_a5",        # W4A4C8 必须设为 nanovllm_a5
        "sparse_activation_tokens": 6144,
        "resident_budget_tokens": [6144, 10240, 12288],
    },
    "enable_sparse_sfa_c8": True,            # W4A4C8 开关（须与 li_c8 一致）
    "enable_sparse_li_c8": True,
}
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enabled` | `False` | DSA offload 总开关 |
| `enable_prefill_full_kv_offload` | `False` | Prefill 后统一 dump 所有层 |
| `keep_decode_kv_in_hbm` | `False` | Decode KV 不卸载到 DDR |
| `op_backend` | `"default"` | `"default"`=框架原有算子；`"nanovllm_a5"`=A5 自定义算子 |
| `sparse_activation_tokens` | `6144` | 进入 sparse decode 的序列长度阈值 |
| `resident_budget_tokens` | `(6144, 10240, 12288)` | HBM resident 预算档位 |
| `enable_sparse_sfa_c8` | `False` | C8 packed KV cache 开关 |
| `enable_sparse_li_c8` | `False` | C8 量化 Indexer 开关 |

### 按请求上限定容（阶段二）

默认 `capacity_mode="auto"` 保持 vLLM 的原有行为：根据可用 HBM 尽可能
多地分配 KV block。对于 DSA，这会间接放大 swapped-DRAM arena，因此长上下文
低并发服务应使用 `request_limit`：

```python
"dsa_sparse_config": {
    "capacity_mode": "request_limit",
    "capacity_guard_blocks": 0,
    "indexer_mla_block_ratio": 1,
}
```

启动时按以下规则固定容量（`B` 是 `--block-size`）：

```text
request_blocks = MAX_NUM_SEQS * ceil(MAX_MODEL_LEN / B) + capacity_guard_blocks
resident HBM base blocks = request_blocks + 1  # BlockPool block 0 是空映射
Indexer HBM blocks       = resident_base_blocks * indexer_mla_block_ratio
DDR usable blocks        = request_blocks
DDR arena blocks         = request_blocks + 1  # block 0 是空映射
```

因此 `MAX_NUM_SEQS=1`、`MAX_MODEL_LEN=131072`、`B=128`、guard 为 `0` 时，
请求/DDR usable 均为 `1024`，而 resident base 和 Indexer（ratio=1）均为
`1025`（其中 block 0 不可分配）；无需设置
`--num-gpu-blocks-override`。如果可用 HBM 无法容纳请求上限，服务会在启动期
明确失败，而不是分配过小的 cache。`--num-gpu-blocks-override` 仍可作为显式
运维覆盖，但在 `request_limit` 模式不得小于计算出的 base 下限。
`capacity_guard_blocks` 是额外的完整 block 余量；严格按上限做容量验收可设为
`0`，在线服务若希望保留调度余量可设为 `128`。

`enable_prefill_full_kv_offload=true` 会在最后一个 prefill chunk 后 dump 所有
已经完整的 block；`keep_decode_kv_in_hbm=true` 则让 decode 新生成 KV 和不足一个
block 的尾部继续留在 HBM。这两个策略不能表示“decode 全量 KV 都落 DDR”。

## 算子安装

### 前置条件

- Ascend 950 (A5) 服务器
- CANN 9.1
- `msopgen` 工具可用

### 方式 1：通过 vLLM-Ascend 根构建（推荐）

A5 DSA 算子已嵌入主扩展 `vllm_ascend_C`，CANN OPP 包自动构建到 `vllm_ascend/_cann_ops_custom/nanovllm_a5/`：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export SOC_VERSION=ascend950
COMPILE_CUSTOM_KERNELS=1 pip install -e . --no-build-isolation
```

构建流程：
1. `csrc/build_aclnn.sh` 构建 CANN 自定义算子 → 安装到 `vllm_ascend/_cann_ops_custom/`
2. A5 DSA OPP 由 `build_nanovllm_ops.sh` 构建 → 安装到 `_cann_ops_custom/nanovllm_a5/`
3. C++ 扩展 `vllm_ascend_C` 编译时嵌入 A5 DSA torch 算子注册

### 方式 2：独立构建

```bash
cd csrc/attention/nanovllm_a5_ops
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export SOC_VERSION=ascend950
bash install_nanovllm_a5.sh
```

### 验证

```bash
python3 -c "
import torch
import vllm_ascend.dsa_offload.a5_extension  # noqa
required = (
    'lidu_decode_update_out', 'scatter_copy', 'sparse_and_tail_attention',
    'lidu_cache_update_out', 'packed_scatter_copy_out',
    'sparse_and_tail_attention_c8', 'full_block_dump', 'packed_full_block_dump',
)
for name in required:
    assert hasattr(torch.ops.nanovllm_dsa, name), f'Missing: {name}'
print('A5 DSA operators: OK')
"
```

## 架构

### 算子嵌入

A5 DSA 算子注册嵌入主扩展 `vllm_ascend_C`：

```
vllm_ascend_C.so
├── csrc/torch_binding.cpp                          ← _C_ascend 命名空间
├── csrc/nanovllm_a5_ops_binding.cpp                ← nanovllm_dsa 命名空间（LIDU/KSC/SFA）
├── csrc/nanovllm_a5_full_block_dump_binding.cpp    ← W4A8 dump
├── csrc/nanovllm_a5_packed_full_block_dump_binding.cpp  ← W4A4C8 packed dump
├── csrc/nanovllm_a5_vllm_op_api_bridge.h           ← ACLNN 桥接（EXEC_NPU_CMD_ORDERED → EXEC_NPU_CMD）
└── 通过 ASCEND_CUSTOM_OPP_PATH 解析 libcust_opapi.so
```

### C8 适配器

`ops_nanovllm.py` 模块加载时自动安装两个 patch 模块：

**`a5_c8_policy.py`** — C8 数据面 runtime policy（8 个 patch）：

| Patch | 作用 |
|-------|------|
| `reshape_kv_cache_tensors` | resident cache 零拷贝重解释为 packed int8 [656] |
| `bind_resident_layer_cache` | 共享 C8 attention_slots scratch |
| `prepare_forward` | 刷新 candidate/logical/resident lengths |
| `prepare_graph_capture` | 初始化 C8 图元数据 |
| `dump_prefill_full_blocks` | C8 packed full-block dump |
| `dump_full_blocks` | C8 增量 packed dump |
| `execute_decode_selection` | C8 LIDU + packed KSC |

**`a5_c8_split_cache.py`** — C8 Indexer split-cache adapter（5 个 patch）：

| Patch | 作用 |
|-------|------|
| `reshape_kv_cache_tensors` | Indexer backing 零拷贝拆分为 FP8 key + FP32 scale |
| `bind_dsa_target_kv_caches` | 绑定 key_dequant_scale |
| `indexer_select_pre_process` | 保留 k_li_scale |
| `indexer_select_post_process` | Prefill 用 A5 quant LightningIndexer |
| `prepare_execution_view` | 图 replay padding 清零 |

**`a5_extension.py`** — A5 DSA Python 包入口：
- 从 `vllm_ascend_C` 加载 C++ 算子注册
- 设置 `ASCEND_CUSTOM_OPP_PATH` 指向 `_cann_ops_custom/nanovllm_a5/`
- 注册 C8 Python composite ops（`lidu_decode_update_c8`、`sparse_and_tail_attention_c8`）

### 数据流

```
W4A8 Decode:
  LIDU(BF16) → KSC(分离KPE/CKV, DRAM→HBM) → SFA(BF16, cache_tokens定义tail)

W4A4C8 Decode:
  LIDU(FP8+scale) → packed_KSC(int8 packed 656B, DRAM→HBM)
    → SFA_C8(packed KV, attention_slots)

Prefill Dump:
  W4A8:  dump_prefill_full_blocks → full_block_dump (分离 NOPE/ROPE)
  W4A4C8: dump_prefill_full_blocks → packed_full_block_dump (packed 656B)
```

## 算子接口

### W4A8

```python
torch.ops.nanovllm_dsa.lidu_decode_update_out(
    query, key, weights, req_pool_entries, cache_slots_pool,
    cache_tokens, candidate_lens, block_table,
    source_ids, destination_slots, miss_counts
) -> (source_ids, destination_slots, miss_counts, cache_slots_alias)

torch.ops.nanovllm_dsa.scatter_copy(
    hbm_kpe, hbm_ckv, dram_kpe, dram_ckv,
    hbm_block_table, dram_block_table,
    source_token_ids, destination_slots, copy_counts
) -> (hbm_kpe_alias, hbm_ckv_alias)

torch.ops.nanovllm_dsa.sparse_and_tail_attention(
    query, key, value, sparse_slots, cache_tokens, block_table,
    actual_seq_lengths_query, actual_seq_lengths_kv,
    query_rope, key_rope, scale_value
) -> attention_out
```

### W4A4C8

```python
torch.ops.nanovllm_dsa.lidu_decode_update_c8_out(
    query, key, weights, query_dequant_scale, key_dequant_scale,
    actual_seq_lengths_query, req_pool_entries, cache_slots_pool,
    cache_tokens, candidate_lens, block_table,
    source_ids, destination_slots, miss_counts
) -> (source_ids, destination_slots, miss_counts, cache_slots_alias)

torch.ops.nanovllm_dsa.packed_scatter_copy_out(
    hbm_kv_bytes, dram_kv_bytes,
    hbm_block_table, dram_block_table,
    source_token_ids, destination_slots, copy_counts,
    cache_tokens, candidate_lens, actual_seq_lengths_kv,
    max_tail_tokens, attention_slots, resident_seq_lengths
) -> (hbm_kv_alias, attention_slots, resident_lengths)

torch.ops.nanovllm_dsa.sparse_and_tail_attention_c8(
    query, packed_kv, attention_slots, block_table,
    actual_seq_lengths_query, resident_seq_lengths, scale_value
) -> attention_out
```

### Dump

```python
torch.ops.nanovllm_dsa.full_block_dump(
    hbm_kpe, hbm_ckv, dram_kpe, dram_ckv, src_block_ids, dst_block_ids
)

torch.ops.nanovllm_dsa.packed_full_block_dump(
    hbm_packed_kv, dram_packed_kv, src_block_ids, dst_block_ids
)
```

## CANN 自定义算子

```
csrc/attention/nanovllm_a5_ops/
├── lightning_indexer_decode_update_a5/   ← LIDU (W4A8)
├── a5_kvcache_scatter_copy/              ← KSC (W4A8)
├── a5_sparse_and_tail_attention/         ← SFA (W4A8)
├── a5_lidu_cache_update/                 ← C8 request-pool update
├── a5_packed_kvcache_scatter_copy/       ← C8 packed KSC
├── a5_full_block_dump/                   ← W4A8 满块 dump
├── a5_packed_full_block_dump/            ← W4A4C8 packed 满块 dump
├── torch_extension/                      ← 独立 Python 扩展（可选）
└── build_nanovllm_ops.sh                 ← CANN 算子构建脚本
```

## 约束

- 仅面向 GLM-5.1 decode，`q_seq_len=1`，block size=128
- LIDU 支持 `index_heads=32|64`；SFA 支持 `q_head=1..64`
- `cache_tokens` 预算：3072/6144/8192/12288
- `enable_sparse_sfa_c8` 和 `enable_sparse_li_c8` 必须同时开/关
- W4A4C8 需要 `op_backend="nanovllm_a5"`
- MTP 需要 `keep_decode_kv_in_hbm=true`，且 `method="mtp"`
- 不支持 CP (Context Parallel)、PP (Pipeline Parallel)、prefix caching、async scheduling
- 支持 DP (Data Parallel)、EP (Expert Parallel)
