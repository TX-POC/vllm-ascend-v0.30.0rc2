# P10：产品原生 C8 streaming resident 设计

## 结论

P10 不再修补 P9，也不把 `runtime_enable=0` 的 P9 二进制当作 C8
基线。P9 把闭包注入一个已有 C8 类；即使运行时关闭闭包，编译器仍会看到
不同的对象布局、UB（Unified Buffer）分配、寄存器压力和事件依赖。因此，
“关闭 P9 后输出不等价”只能说明这种对照方式无效，不能说明真正的
streaming C8 不可行。

P10 的产品路径是一个**同一 forward 内的设备侧图组合**：一个正式派生的
C8 Stage-A 负责真实 score 和原始精确 Top-K，同时发布已完成 score tile 的
**原始 score receipt**；一个 AICore Direct-HCCP Stage-B 消费 receipt，执行
`ExactScoreScan → CandidateCompact → Post/CQ`；一个 Stage-C 在最终 Top-K 已知后
补齐遗漏 winner。QSFA 只在所有最终 winner 已经到达指定 resident slot 后读取
它们。SIMT 在此设计中只承担原子 lease 的小型 helper/handoff，不执行 HCCP
事务。

```text
main stream
  Stage-A: 真实 C8 score tile → raw score-receipt publish → 原始精确 Top-K
       │                                  │
       │                                  └── event / GM cursor
       │                                              │
auxiliary device stream                              ▼
  Stage-B: ExactScoreScan → CandidateCompact → AICore PostOnly → 非阻塞 CQ drain
            ▲
            └── SIMT atomic lease helper / event handoff
                                                     │
main stream                                          ▼
  Stage-C: 精确 Top-K → Guaranteed Recall → all winner READY
       │
       └── QSFA 只读取最终 resident slots
```

这不是“31+1 分核”或 CPU/Python sidecar。queue 是固定数量的共享硬件资源，
任意可用 executor worker 均可动态租用；C8 worker 也没有永久放弃自己的
AIV。Stage-A 和 Stage-B 分开，是因为 Stage-A 的 score/Top-K 热循环和
Stage-B 的 Direct-HCCP transaction 都必须有各自正式预算；把二者拆成图中的
两个节点是硬件与生命周期边界，不是退化为主机串行调度。Stage-B 的 PostOnly、
TryRetire 与 Finish 由 AICore 执行；AICore 通过短 SIMT helper 完成 CANN 要求
的原子 CAS/lease handoff，不能把 raw SIMT callee 直接当作 AICore 函数调用。

## 先解析、再派生：C8 source provenance

P10 不能把目录名或 P9 的 reference 当作“唯一产品 C8 真相”。已知 P9
reference 使用的是 `VllmQuantLightningIndexer` 家族；仓库中也存在
`csrc/attention/lightning_indexer_quant/` 的 `LightningIndexerQuant` 家族。
但是 DSA runtime 的 Python dispatch（例如 `torch_npu.npu_quant_lightning_indexer`
与相应 `torch.ops` route）最终绑定哪一个已注册 OPP，必须由本次构建和运行
显式证明。没有这份证明，就不能从任一目录派生 P10，也不能声称其为产品
数学 oracle。

每次 P10 build 必须先生成并校验一个 machine-readable provenance manifest：

```text
Python dispatch symbol + resolved torch op schema
registered OPP/operator name + vendor/opapi path
source family/root + source Git SHA
score seam file/anchor + final Top-K file/anchor
compiled sibling source SHA + target device/CANN version
```

manifest 必须由实际 import/registration 和 OPP 构建输出生成；禁止由 P9 测试、
目录名称或旧 artifact 猜测。P10 sibling 的 source family 必须与 manifest 中
解析出的产品 family 完全一致；否则构建应 fail closed。

### 前端 ABI gate：不能把 BF16 DSA-offload 与 FP32 main DSA 混为一个产品调用

当前仓库存在两条实际产品 frontend，它们的 C8 weights ABI 不相同，P10 的
数学 gate 必须由 `P10_PRODUCT_C8_FRONTEND` 显式选择，禁止根据某个恰好可见的
torch op 自动猜测：

| frontend | 产品调用 | 产品 weights | 官方 Vllm reference weights | 合法比较方式 |
| --- | --- | --- | --- | --- |
| `dsa_offload_bf16` | `torch_npu.npu_quant_lightning_indexer` | BF16 | FP32 | 仅传入同一 BF16 tensor 的无损 `.float()` widening |
| `main_dsa_v1_fp32` | `torch.ops._C_ascend.npu_vllm_quant_lightning_indexer` | FP32 | FP32 | 两端传入同一 FP32 tensor |

这不是一个可省略的格式细节：若把 BF16 frontend 直接喂给 FP32 reference，或把
main DSA 的 FP32 route 误当成 BF16 native route，输出差异无法归因于 streaming
C8。`run_p10_product_c8_abi_gate_npu.sh` 在选择设备、导入 bridge 或启动任何
target OPP 前，先检查 frontend 的源码 anchor、reference source SHA、OPP
`P3_REFERENCE_PROVENANCE` 和 bridge 的 write-once
`P10_REFERENCE_BRIDGE_PROVENANCE`。缺任一项即阻断，不会使用旧 bridge 或隐式
fallback callable。

还必须显式选择产品前端，不能把不同 ABI 混为一个“C8 产品路径”。当前至少有：

* `dsa_offload_bf16`：经 `torch_npu.npu_quant_lightning_indexer`，权重是 BF16；
  若参考 OPP 的权重 ABI 是 FP32，参考只能接收同一 BF16 值的无损
  `BF16 → FP32` widening；
* `main_dsa_v1_fp32`：经
  `torch.ops._C_ascend.npu_vllm_quant_lightning_indexer`，产品与参考必须接收
  同一份 FP32 权重。

因此 provenance 的“frontend + callable + tensor ABI”与 source family 同等重要。
禁止根据某个 torch op 恰好存在而自动猜测路径，也禁止用一种前端的输入证明另一种
前端的 bitwise 等价。

| 层次 | 需要解析的事实 | P10 的处理 |
| --- | --- | --- |
| C8 OPP | 被 DSA runtime 实际注册/调度的 operator family；候选包括 `VllmQuantLightningIndexer` 与 `LightningIndexerQuant` | 保持解析出的 family 为不可修改的数学 oracle；从它建立 sibling 源包 |
| C8 AICore 热路径 | manifest 指定的 score tile 与 final Top-K anchor | 只在 sibling 中增加正式 score-receipt ABI |
| DSA 产品策略 | `vllm_ascend/dsa_offload/a5_c8_policy.py::execute_decode_selection` | OFF 保持原顺序；ON 调用 P10 graph composite |
| C8 Python bridge | `vllm_ascend/dsa_offload/ops_nanovllm.py::lightning_indexer_decode_update_c8` | 仅保留 legacy；P10 添加独立 bridge |
| slot/runtime 状态 | `vllm_ascend/dsa_offload/runtime.py` 与 `resident_pool.py` | 预分配 graph-visible P10 控制/slot tensor，禁止热路径主机分配 |

当前 legacy 顺序为：native C8 → `lidu_cache_update_out` →
`KvafAdapter.execute_batch_get` → attention metadata。其中
`lightning_indexer_decode_update_c8` 的 `torch.npu.synchronize()` 也只属于
legacy 路径；P10 active 路径不得调用它。

## P10 的源码边界

### 1. 不修改、也不 overlay 已解析的产品 C8

P10 build 的第一步必须先解析 product provenance；解析出的 source root 永远
保持原样，并作为该构建唯一的 C8 数学 oracle。例如，manifest 可能指向：

```text
csrc/attention/vllm_quant_lightning_indexer/
或 csrc/attention/lightning_indexer_quant/
```

尤其不改 manifest 所列 operator 定义、tiling 或 score/Top-K 类。P10 不采用
运行时开关关闭功能，也不采用 build-time 文本 overlay。这样 OFF 产品路径
继续调用原始 OPP，ACTIVE 路径调用不同的、明确命名的 OPP；两条路径的二进制
身份和 provenance 都能独立验证。

### 2. 新建产品派生的 Stage-A OPP

新增目录：

```text
csrc/attention/nanovllm_a5_ops/a5_c8_streaming_resident/
  op_host/a5_c8_streaming_resident_def.cpp
  op_host/a5_c8_streaming_resident_inferrs.cpp
  op_host/a5_c8_streaming_resident_tiling.cpp
  op_kernel/a5_c8_streaming_resident.cpp
  op_kernel/a5_c8_streaming_resident_kernel.h
  op_kernel/a5_c8_streaming_resident_service_vector.h
```

该目录从 provenance manifest 解析出的产品 C8 family **复制一次并长期显式
维护**，而不是每次构建时修改源文件。建议 operator 名为
`A5C8StreamingResidentStageA`。实际文件名、类名和 score seam 以 manifest
为准；本文不将 `LIQVector::ProcessVec1` 或任何 `Vllm*` 类名硬编码为跨版本
不变的产品事实。

Stage-A 保留解析出的原始 C8 输入和原始 `sparse_indices`/`sparse_values` 输出；
正式增加以下 graph-visible ABI：

```text
score_surface
score_receipt_ring
receipt_producer, receipt_done, stage_a_stats
```

第一版 receipt 是一个轻量 descriptor：
`[row, source_begin, valid_count, score_surface_offset_bytes,
score_element_bytes, sequence]`；`READY` 是 ring state，不是 payload 字段。
`score_surface` 是一个有界的、forward-owned **score-window pool**：每个 receipt
slot 对应一段与原 score 相同的 `uint16` 表示和对齐。它不是原 C8 每 AIC 的
可覆写 `scoreGm` workspace，也不是 Stage-B 偷读 opaque workspace。否则下一
个 `gS1` block 覆写旧 `scoreGm` 时，会与异步 executor 产生读写竞争。

在 manifest 指定的真实 score tile seam，先保留原本 `outBase → scoreGm` DMA；
只有成功 reserve 一个 receipt slot 时，才额外将同一已完成 score window DMA 到
该 slot 的 score-window pool。该 DMA 使用新的、独立预算的 MTE3/event edge，
然后才发布 descriptor：

```text
normal `outBase → scoreGm` DMA → reserve receipt slot
  → streaming `outBase → score-window pool` DMA → 等待新 MTE3 completion
  → 写完整 receipt descriptor → GM publish barrier → 更新 producer cursor
  → 继续原始 Sort/Top-K
```

这个 seam 不读 resident 映射、不做阈值判断、不 claim resident slot，也不构造
HCCP descriptor。receipt ring 已满时，它跳过第二次 DMA 并将该 window 标为
`DEFERRED`；最终 Guaranteed Recall 仍覆盖真正 winner。它绝不改写被原始
排序/归并读取的 score、index 或输出。score-window pool、receipt ring 和其新
event/GM 可见性均由 sibling 的 tiling 显式预算；禁止在原 C8 layout 中暗插
`TBuf` 或偷用 Top-K event。

把 `ExactScoreScan → CandidateCompact` 放入 Stage-B 不是退化或多一次 Q/K 计算：
Stage-B 只读 Stage-A 已产生的真实 `uint16` score surface，不重新计算 Q/K。这样
先把 C8 热循环与原始 Top-K 的二进制等价风险控制在一个正式、可比较的 score-surface
ABI 内。只有当 sibling 与原始 C8 已经通过跨二进制 bitwise gate，且 profile 证明
score-surface reread 是瓶颈时，才可把 compact 提升到同一 tile seam；这仍是同一个
图架构，而不是另一条产品路线。

### 3. 新建 Stage-B AICore Direct-HCCP executor

新增目录：

```text
csrc/attention/nanovllm_a5_ops/a5_c8_resident_executor/
```

operator 名建议为 `A5C8ResidentExecutor`。它只接触 score receipt、score surface、
固定 queue descriptor/CQ 控制和 resident readiness，不接触 C8 的 UB 或 Top-K
workspace。它的 Direct-HCCP transaction 由 AICore 执行；SIMT 只执行
lease/CAS helper 并通过 event 将结果交回 AICore。其真实 hot path 是：

```text
AICore claim score receipt → ExactScoreScan → CandidateCompact
                         → SIMT CAS lease helper → AICore fill descriptor
                         → publish barrier → AICore PostOnly
                         → continue next candidate → AICore bounded TryRetire
```

`PostOnly`、`TryRetire` 和 `Finish` 直接复用 KVAF 已有拆分 ABI；P10 不允许
在 Python 线程或 CPU/GIL 下逐 candidate 提交。Stage-B 以 NPU auxiliary
stream/事件与 Stage-A 同属一个 forward graph，而非孤立 benchmark sidecar。
SIMT 既不独立拥有 HCCP queue 生命周期，也不替 AICore 写 descriptor、doorbell
或轮询 CQ。

### 4. Stage-C 是产品完整性的必要部分

新增 `A5C8ResidentFinalize`（可与 Stage-B 同包）消费 exact `sparse_indices`。
它执行以下确定流程：

1. final Top-K 已经为数学唯一真相；
2. 已 resident 的 winner 保留 persistent slot；
3. 已 speculative-ready 的 winner 保留其先前 claim 的 arena slot；
4. 未 ready 的 winner 从同一 arena 的单调 cursor 新 claim slot，并做
   Guaranteed Recall；
5. drain 所有已 admission 的 post，确认每个 winner `READY`；
6. 输出 `attention_slots`、`resident_seq_lengths` 和 terminal-ready marker。

只有第 6 步以后，主 stream 才等待 ready event 并进入 QSFA。这样一个
forward 内没有 slot 的早释放、跨 forward reuse 或 ABA 语义；P10 不引入
P8 的 arena/replay/quarantine 协议。

## forward-owned resident arena 的硬前提

现有 `runtime.py` 仅预分配了 `[B,1,2048]` 的 `topk_slots`，而
`lidu_cache_update_out` 在 exact Top-K **之后**才分配 destination slot。因此
它不能作为评分阶段 candidate 的直接 DMA 落点。

P10 为每个 row/layer 建立一个**只属于当前 forward 的 resident arena**。它不
复用 persistent hit 的已有 slot，并具有一个仅递增的 `arena_slot_claim_cursor`：

```text
arena_capacity = TOPK + speculative_budget
score-time admitted candidate: claim one new arena slot
final Top-K non-ready winner: claim one further new arena slot for recall
```

这里的 `speculative_budget` 是允许在 score scan 期间被 admission 的最大候选数，
不是固定 Get core 数，也不是 Top-K rank 的别名。容量不变式是：

```text
admitted_speculative_count <= speculative_budget
final_recall_count <= TOPK
total_arena_claims <= speculative_budget + TOPK == arena_capacity
```

因此早期 candidate 不需要知道自己的最终排名，也不会与另一个 candidate 或
recall 覆盖同一地址。final winner 若已经 `READY`，Stage-C 直接采用其
`arena_slot`；若未 ready，Stage-C 只从 cursor 的新位置领取 slot。false-positive
candidate 仍保有自己的唯一 slot，直到 forward terminal，绝不被 recall 覆盖。

candidate ring 或 arena admission 达到 `speculative_budget` 后，后续候选必须被
标记为 `DEFERRED`，不得 Post、不得重用 slot；它们仅在 final Top-K 出现时通过
Guaranteed Recall 获得新的 arena slot。这降低可能的 overlap，不影响精确
Top-K、最终 resident placement 或 QSFA 数学结果。

`forward terminal` 在这里是存储生命周期边界：只有所有被 admission 的 post
（包括 false-positive）都已 terminal/drained，或其底层存储被明确永久保留而
绝不复用时，arena 才能离开该 forward。这样不需要 ABA tag、slot release 或
跨-forward owner 协议；关键不变量只是 **claim 后直到 terminal 不复用**。

## 产品集成边界

在 `a5_c8_policy.py::execute_decode_selection` 增加一个独立 active branch：

```text
OFF: 原有 lidu_c8 → execute_batch_get → metadata，字节级保持
ON : streaming_c8_resident_select → wait resident-ready → metadata/QSFA
```

`ops_nanovllm.py` 新增 `streaming_c8_resident_select(...)`，不改写
`quant_lightning_indexer_c8` 或 `lightning_indexer_decode_update_c8` 的旧 ABI。
`runtime.py` 在 graph capture 前准备 ring、forward-owned arena、monotonic
cursor、event 和
统计 tensor；forward 内不得新建 Python list、动态 device allocation 或调用
`torch.npu.synchronize()`。

仓库已有 `vllm_ascend/attention/dsa_v1.py::dsv4_dsa_overlap_stream` 和
`_mla_prolog_multistream` 的 `torch.npu.Stream/Event` 用法，P10 应沿用同类
stream/event 依赖，而不是发明 host callback 协议。

## 正确性与性能 gate

P10 的 C8 数学 gate 是跨二进制的：同一输入下，provenance manifest 解析出的
未修改产品 C8 与从同一 family 派生的 `A5C8StreamingResidentStageA` 的
`sparse_indices` 必须 bitwise 相同。禁止将 active binary 的 runtime-off
分支当作 reference，也禁止将 P9 reference family 自动当成产品 family。

完整 forward gate 还必须证明：

1. Stage-A 发布的 raw score receipt 指向已完成的真实 score surface；
2. Stage-B 只更新 forward-owned arena slot，不触碰 C8 Top-K state；
3. Stage-C 对所有 exact winner 都给出 ready resident slot；
4. QSFA 前没有 pending CQ 或未完成 recall；
5. OFF 路径仍是原始 legacy 调用链；
6. ACTIVE 路径没有 legacy C8 + legacy `execute_batch_get` 的重复串行调用。

性能只在真实 Direct-HCCP payload validation 通过后才报告。若 auxiliary
executor 竞争 AIV、stream 实际被 CANN 串行化、或 candidate overflow 高，P10
可以没有收益；这些都是应测量并可能否定的假设，不是可预先宣称的优化。

## 明确排除的路线

- 不再修 P9 的 source overlay 或其 `runtime_enable` 数学 gate；
- 不再向已解析产品 C8 原类追加跨 AICore/SIMT 的 Get/CQ 闭包；
- 不采用永久 `31+1` 或其他物理核心分区；
- 不通过两个 Python 线程规避 GIL；
- 不将未终态 false-positive copy 误标为 QSFA-ready，或将其 slot 交给 recall
  覆盖；
- 不以已知失败的 payload fixture 报告端到端带宽或收益。

P10 的第一份代码提交应同时包含 sibling Stage-A 的正式 ABI、Stage-B/Stage-C
的 graph-visible contract、以及上述跨二进制数学 gate；否则它仍只是 P9 风格的
中间原型，而不是产品原生实现。
