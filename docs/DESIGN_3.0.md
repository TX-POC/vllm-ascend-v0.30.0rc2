# CacheOS 1.0 Stage 2：基于 custom-v1 的 W4A4C8 KVAF 集成设计 2.0

## 0. 文档定位与冻结结论

| 项目 | 内容 |
|---|---|
| vLLM Ascend / DSA 基线 | `vllm-ascend-v0.23.0-custom-v1 @ 17adad28c3604353c9442a1c5bd4147da68ebc37` |
| KVAF 基线 | `main @ 5d310319a2134d5e0d868a7b99f053f308c5ce33` |
| 目标硬件 | A5（昇腾 950DT） |
| M0 主场景 | W4A4C8 Decode KV offload，Eager，同步 Put facade |
| 数据布局 | 每卡持有完整 MLA KV；每卡独立维护 KVAF 目录和生命周期 |

M0 冻结以下主线：

- DSA 控制面用稳定的 `pool_row` 表示请求在本 Worker 内的 resident row；
- 每个本地 MLA layer、每个 logical block 在目录中占一个 cell；
- KVAF 内部目录是地址稳定的一维 row-major 表；
- KVAF Get 的单条 DDR source 坐标只有 `{flat_index, offset}`；
- KVAF Put 按完整 layer block 写入，单条目标坐标是 `{flat_index}`；
- Dense、Sparse、PAD 使用同一个固定形状执行 envelope，以 `row_mode`、`row_valid` 和计数控制实际工作量；
- pool row、DDR block、buffer 和 plan 采用严格串行复用；旧 owner 完全结算后才允许重新绑定，不设置版本字段。

---

# 第一部分：必须实现的主业务流程

## 1. 目标、范围与数据主线

### 1.1 数据主线

模型形成完整 logical block 后，KVAF 把该 block 的每个本地 MLA layer 从 packed Resident HBM 写入 DDR。Sparse Attention 需要历史 token 时，LIDU 给出 source token；Adapter 将其转换成 `{flat_index, offset}`，KVAF 读取 656B token payload并直接写入 Attention 最终读取的 packed Resident HBM。

~~~text
完整layer block产生：packed Resident HBM -- Block Put --> KVAF DDR
Sparse miss出现：        KVAF DDR -- Token Get --> packed Resident HBM
Attention：              只读取packed Resident HBM
~~~

Attention 不感知 DDR UBA、DDR block id、Host/Device 目录或 Put/Get plan。

### 1.2 M0 packed layout

~~~cpp
struct PackedKvLayout {
  LayoutId layout_id;
  uint32_t tokens_per_logical_block;   // M0 = 128
  uint32_t packed_token_bytes;         // W4A4C8 = 656
  uint32_t ddr_block_alignment_bytes;
};

layer_payload_bytes =
    tokens_per_logical_block * packed_token_bytes;
ddr_block_bytes =
    align_up(layer_payload_bytes, ddr_block_alignment_bytes);
ddr_block_padding_bytes =
    ddr_block_bytes - layer_payload_bytes;
~~~

一个 DDR 物理块保存一个 logical block 的一个本地 MLA layer。Put 写入 `layer_payload_bytes`；Get 每次读取 `packed_token_bytes`。对齐 padding 不属于 KV payload。

### 1.3 M0 范围

M0 支持：

- final-prefill 多 block Group Put；
- Decode boundary 单 block Group Put；
- LIDU 驱动的 Direct Get；
- Dense/Sparse/PAD mixed batch；
- C8 metadata 与 Dense/Sparse Attention；
- continuous batching reorder、临时 unschedule 和 Finish；
- 多卡 MLA 全量副本部署，但 KVAF 不新增跨卡可靠性协议。

M0 不支持 prefix sharing、可恢复 preemption/resume、P/D migration、KV connector、异步 Put writer、多 lane 和生产 Graph。

## 2. 基线能力与最小改造范围

### 2.1 直接复用

custom-v1 继续提供 Scheduler、`prepare_forward`、resident pool、LIDU、C8 Attention、逐层回调和 Finish 接缝。KVAF 继续提供：

- Runtime、DDR allocator 和 region registration；
- 现有固定粒度DDR分配器（源码当前以slot命名）；
- Direct-HCCP Put/Get transport；
- AIV queue、CQ、status/workspace；
- caller stream 提交；
- local HBM registration cache；
- Put 的 source-offset/length/destination-address 数据面；
- Get 的一维 `address_table[index]` 搬运主体。

### 2.2 必须替换的生产路径

| custom-v1 当前实现 | Stage 2 目标 |
|---|---|
| reserve 后提前发布 DDR 映射 | 所有本地 layer 写完、Device 目录可见后再发布 committed count |
| legacy two-plane reinterpret | allocator 直接提供 canonical packed cache |
| KSC 同时搬 KV 和生成 metadata | KVAF Direct Get 搬 KV；独立 metadata op 只写索引 |
| import-time monkey patch | 显式 `DsaKvafAdapter` / C8 strategy |
| DSA 自建 DRAM payload/table | KVAF 独占 DDR block、目录和生命周期 |
| Finish 先归还 pool row | KVAF Close 成功后再归还 pool row |

### 2.3 KVAF 最小增量

新增能力只有：

1. 固定 row-major Host/Device DDRBlockTable；
2. resident row 的 Bind、count 和 Close；
3. Group Block Put plan；
4. dirty range Device publication；
5. forward-level Row Access；
6. `{flat_index, offset}` Get source ABI；
7. 本卡批内全量校验后统一放行 Get；
8. quarantine、Shutdown 和统计。

现有 flat `KeyId` API 原样保留给 `FLAT_COMPAT`。DSA_ROW 生产路径不生成 KeyId、不进入 Key hash、不为每个 forward 重建完整 prefix AccessTable。

## 3. 扁平目录、坐标和生命周期

### 3.1 四类坐标

| 坐标 | 含义 | 使用位置 |
|---|---|---|
| `batch_row` | 当前 forward 的 full-batch 行 | tensor 排布与输出 scatter |
| `pool_row` | 请求在 resident pool 中的稳定 row | DSA 生命周期、目录 row base |
| `block_q` | token 所属 logical block，范围 `[0,Qcap)` | P/count、boundary Put |
| `local_layer_ordinal` | 本卡 MLA layer 序号，范围 `[0,n_layers)` | 选择目录中的 layer 段 |

### 3.2 `q` 与 `flat_index`

初始化冻结：

~~~text
RawRowCapacity = DSAResidentTokenPool.max_num_reqs
n_layers = local_layer_count
Qcap = ceil(runtime_max_sequence_tokens / tokens_per_logical_block)
RowStrideCells = n_layers * Qcap
~~~

本文在目录公式中使用的 `q` 是 layer 与 logical block 的线性化坐标：

~~~text
block_q = token_id / tokens_per_logical_block
offset  = token_id % tokens_per_logical_block       // 0..127
q       = local_layer_ordinal * Qcap + block_q

flat_index = pool_row * n_layers * Qcap + q
           = pool_row * RowStrideCells
             + local_layer_ordinal * Qcap
             + block_q
~~~

边界：

~~~text
0 <= pool_row < RawRowCapacity
0 <= local_layer_ordinal < n_layers
0 <= block_q < Qcap
0 <= q < n_layers * Qcap
0 <= flat_index < RawRowCapacity * n_layers * Qcap
0 <= offset < tokens_per_logical_block
~~~

`padding_pool_index == RawRowCapacity`，不能 Bind、建 Access 或计算 `flat_index`。

### 3.3 Host 与 Device 目录

~~~cpp
struct RowMeta {
  RowState state;                         // FREE/BINDING/OPEN/CLOSING/QUARANTINED
  uint32_t committed_full_block_count;   // 每层共同的连续block前缀长度
  uint32_t active_access;
  optional<ActivePutOwner> active_put;   // plan与本次追加范围
};

struct DdrBlockState {
  DdrBlockId block_id;
  DdrBlockLifecycle state;               // FREE/RESERVED/OWNED/QUARANTINED
  optional<uint32_t> owner_flat_index;
  uint32_t in_flight_io;
};

class DdrBlockAllocator {
 public:
  Status Reserve(uint32_t count, Span<DdrBlockId>* blocks);
  Status Commit(Span<const DdrBlockId> blocks);
  Status Release(Span<const DdrBlockId> blocks);
  Status Quarantine(Span<const DdrBlockId> blocks);
};

struct HostBlockCell {
  DdrBlockId block_id;
  HostCellState state;                    // PLAN_PRIVATE/COMMITTED/RETIRING
};

struct DeviceBlockCell {
  uint64_t block_base_uba;
};
~~~

物理存储：

~~~text
RowMeta[RawRowCapacity]
HostDDRBlockTable[RawRowCapacity * n_layers * Qcap]
DeviceDDRBlockTable[RawRowCapacity * n_layers * Qcap]
DdrBlockStateTable[ddr_block_count]
~~~

职责：

- `RowMeta`保存 row 状态和已发布 block count；
- `HostDDRBlockTable[flat_index]`是 DDR block 绑定的 Host 权威值；
- `DeviceDDRBlockTable[flat_index]`是 Get 热路使用的只读地址投影；
- `DdrBlockStateTable`管理 DDR 物理块的分配、I/O 和回收。

`DdrBlockAllocator`由现有MetaManager的free-slot队列抽取而来，是FLAT_COMPAT与DSA_ROW共同使用的唯一DDR块分配器。DSA_ROW按`DdrBlockId`调用它，不经过KeyId，也不建立第二套allocator。

### 3.4 Get 地址链

Adapter 负责把业务坐标转换成 KVAF Get source：

~~~text
source_token_id
  → block_q = source_token_id / tokens_per_logical_block
  → offset = source_token_id % tokens_per_logical_block
  → q = local_layer_ordinal * Qcap + block_q
  → flat_index = pool_row * n_layers * Qcap + q
  → KvafGetSource{flat_index, offset}
~~~

KVAF Get 内部只执行：

~~~text
cell = DeviceDDRBlockTable[flat_index]
remote_address = cell.block_base_uba + offset * packed_token_bytes
~~~

Get的逐item source不接收 `pool_row`、`local_layer_ordinal`、`block_q` 或 `Qcap`；layer只作为本次批调用的目标cache绑定传一次。

### 3.5 Put 地址链

Put 按完整 layer block 工作，不需要 token offset：

~~~text
pool_row + local_layer_ordinal + block_q
  → q = local_layer_ordinal * Qcap + block_q
  → flat_index = pool_row * n_layers * Qcap + q
  → KvafPutBlockItem{flat_index, source_block}
~~~

KVAF Put 为 `flat_index` 预留一个 DDR 物理块，把 `source_block` 的 `layer_payload_bytes` 写入该块，再把 block base 发布到 Host/Device DDRBlockTable。

### 3.6 可读前缀

`RowMeta.committed_full_block_count = N` 表示：该 row 的所有本地 layer 在 `block_q ∈ [0,N)` 上均已完整写入并发布。任何 layer 缺块时都不能推进 count。

本 forward 的 Row Access 冻结 `snapshot_count`。Get source 必须满足：

~~~text
block_q < snapshot_count
flat_index属于该Row Access绑定的row和当前layer
Host/Device cell已发布
publication dependency已完成
~~~

### 3.7 容量预算

~~~text
DirectoryCellCount = RawRowCapacity * n_layers * Qcap
HostTableBytes = DirectoryCellCount * sizeof(HostBlockCell)
DeviceTableBytes = DirectoryCellCount * sizeof(DeviceBlockCell)
RowMetaBytes = RawRowCapacity * sizeof(RowMeta)
DdrBlockStateBytes = ddr_block_count * sizeof(DdrBlockState)
~~~

若现有 AIV 索引继续使用 `int32`，必须满足：

~~~text
DirectoryCellCount <= INT32_MAX
~~~

Initialize 对乘法、字节数、地址和索引做 checked arithmetic；任一预算超限时 fail-fast。

## 4. A～E 主业务总览

### 4.1 统一逻辑执行模板

该模板把一次请求从进入 Worker 到 Finish 分成四个调用层级：Worker 初始化一次、请求建立一次、每个 forward 一次、每个本地 MLA layer 一次。Dense、Sparse、PAD 的数量可以逐 forward 变化，但逐层执行的节点、buffer 容量和调用顺序保持不变。

~~~mermaid
flowchart TD
    A["A Initialize / Bind / 固定DDRBlockTable"] --> PF["Prefill生成canonical packed KV"]
    PF --> B0["B Final-prefill Group Block Put"]
    B0 --> BQ{"B0已COMMITTED?"}
    BQ -- 是 --> E0["E 发布初始P"]
    BQ -- 否 --> BF["Abort或Quarantine并终止请求"]
    E0 --> F0["E 创建本forward上下文"]
    F0 --> FQ{"上下文及可选Put Prepare成功?"}
    FQ -- 否 --> EF["E Failure Finalize"]
    FQ -- 是 --> ENQ{"enter_mask是否有有效row?"}
    ENQ -- 是 --> EN["E BatchEnterSparseResident"]
    ENQ -- 否 --> U0["D 构造固定形状UnifiedRowExecution"]
    EN --> ER{"Enter结果可继续?"}
    ER -- 是 --> U0
    ER -- 否 --> EF
    U0 --> UQ{"Unified descriptor成功?"}
    UQ -- 否 --> EF
    UQ -- 是 --> C0["C PrepareUnifiedAccess"]
    C0 --> CQ{"Access成功?"}
    CQ -- 否 --> EF
    CQ -- 是 --> L0["进入当前local_layer_ordinal"]
    L0 --> C1["C Unified Select / Resolve / Gated Get"]
    C1 --> OK1{"C1成功?"}
    OK1 -- 是 --> D0["D Unified Metadata"]
    OK1 -- 否 --> EF["E Failure Finalize"]
    D0 --> OK2{"D0成功?"}
    OK2 -- 是 --> D1["D AttentionDispatch"]
    OK2 -- 否 --> EF
    D1 --> OK3{"D1成功?"}
    OK3 -- 是 --> B1["B 本层Block Put Write"]
    OK3 -- 否 --> EF
    B1 --> OK4{"B1成功或no-op?"}
    OK4 -- 否 --> EF
    OK4 -- 是 --> LN{"还有下一本地layer?"}
    LN -- 是 --> L0
    LN -- 否 --> E1["E Forward Finalize / P-tail / ReleaseAccess"]
    EF --> FT["Settle / Abort或Quarantine / ReleaseAccess / P不推进"]
    FT --> FR{"失败终态"}
    FR -- FAILED_SAFE --> FS["本请求按既有错误策略终止或显式重试"]
    FR -- UNKNOWN_QUARANTINED --> FU["本卡Runtime terminal / 禁止新admission与普通释放"]
    E1 --> EQ{"Finalize成功?"}
    EQ -- 否 --> FT
    EQ -- 是 --> Q{"请求Finish?"}
    Q -- 否 --> F0
    Q -- 是 --> E2["E Close KVAF row / 释放HBM与pool row"]
~~~

各流程的含义如下。

图中的判定节点遵守以下规则：

| 判定 | 含义 | 否定/失败结果 |
|---|---|---|
| `BQ` | Final-prefill Put是否完整COMMITTED | Abort或Quarantine，不发布P |
| `FQ` | forward上下文及可选Group Put Prepare是否成功 | 终结已创建plan，P不推进 |
| `ENQ` | 本batch的`enter_mask`是否至少有一个有效row | 无有效row时直接进入U0 |
| `ER` | 所有enter row是否得到可继续的确定结果 | UNKNOWN进入Failure Finalize；安全失败row保持Dense |
| `UQ/CQ` | row descriptor与Access是否完整建立 | 不进入逐layer执行，回收已建owner |
| `OK1～OK4` | 当前layer的Get、Metadata、Attention和Put Writer是否成功 | 立即停止后续业务节点，进入Failure Finalize |
| `LN` | 是否还有未执行的本地MLA layer | 有则进入下一layer，无则执行E1 |
| `EQ` | Seal/Commit或无Put收口是否成功 | Abort/Quarantine，P不推进 |
| `Q` | Scheduler是否确认请求最终Finish | 否则进入下一forward；临时unschedule不算Finish |

Failure Finalize必须区分两类终态。`FAILED_SAFE`表示已证明不再有晚到I/O，相关plan与Access已安全终结；请求只能按上层既有错误策略终止，或由明确允许的重试入口重新执行，不能从图中的正常边直接进入下一forward。`UNKNOWN_QUARANTINED`表示仍可能存在设备访问或Meta结果不确定，本卡Runtime进入terminal，停止新admission并保留row、DDR block、HBM、registration和目录owner，直到reset/quiescence与审计完成。

#### 4.1.1 A：Initialize、Bind 与固定目录

**触发时机：** Runtime 启动时初始化一次；新请求进入 DSA runtime 时 Bind 一次。

**输入：** packed layout、`RawRowCapacity`、`n_layers`、`Qcap`、DDR 容量、各本地 layer 的 canonical packed cache，以及请求获得的 `pool_row`。

**动作：** 创建地址稳定的 Host/Device DDRBlockTable，冻结 flat-index 几何；随后把请求的 `pool_row` 置为 OPEN，并把 committed block count 初始化为 0。该步骤只建立目录和资源关系，不搬运 KV payload。

**输出：** 可供后续 Put/Get 使用的 Runtime、目录、row 状态和 layer cache binding。

#### 4.1.2 PF：Prefill 生成 canonical packed KV

**触发时机：** 请求执行 Prefill 时。

**输入：** 模型输入 token、各层 Attention 计算结果和 A 阶段绑定的 packed cache。

**动作：** 模型按现有 Prefill 路径生成 KV，并直接写入 canonical packed Resident HBM。此时 KV 仍是模型本地数据，KVAF 尚未为这些完整 block 建立可读 DDR 目录。

**输出：** 完整前缀 block 和不足一个 block 的 tail。完整 block 进入 B0；tail 继续保留在 HBM。

#### 4.1.3 B0：Final-prefill Group Block Put

**触发时机：** Final-prefill 完成，能够确定需要 offload 的完整 logical blocks 时。

**输入：** 每个请求的 `pool_row`、完整 block 范围、所有本地 MLA layer 的 packed source block，以及由 Adapter 计算的目标 `flat_index`。

**动作：** 为每个 `(row, layer, block_q)` 预留一个 DDR block；逐 layer 写入完整 `layer_payload_bytes`；所有 required layers 写完后，增量发布对应 DeviceDDRBlockTable cells，最后推进该 row 的 committed block count。

**输出：** 每个 row 的`RowCommitResult`。只有整个B0 plan为COMMITTED时才进入E0；ABORTED_SAFE走Abort并释放预留资源，UNKNOWN走Quarantine并终止该请求，二者都不发布初始P。

#### 4.1.4 E0：发布初始 P

**触发时机：** B0 返回 COMMITTED 后。

**输入：** 每个 row 的 `new_committed_block_count`、`tokens_per_logical_block` 和当前序列长度 `S`。

**动作：** 计算 `P = new_committed_block_count * tokens_per_logical_block`，并计算 `tail = S - P`。`P` 表示已经完整写入 DDR、允许成为 Sparse 历史候选的前缀长度。

**输出：** 请求级 `P/S/tail` 账本。该账本进入后续每个 forward 的上下文。

#### 4.1.5 F0：创建本 forward 上下文

**触发时机：** 每个 Decode forward 开始时一次。

**输入：** 当前 full batch、`batch_row → pool_row` 投影、各请求的 `P/S/R/K`、是否到达 block boundary，以及预分配的固定容量 buffers。

**动作：** 冻结本forward使用的请求顺序、业务长度和boundary决策。`UnifiedBlockPutBatch.put_mask`只标记本轮确实形成完整block的row；`put_count>0`时，仅为这些row构造targets/row_commits并在进入layer循环前Prepare唯一的Group Put plan；`put_count=0`时不调用KVAF Prepare。后续逐层流程只读取这份上下文，不在中途重新读取可能变化的Scheduler状态。

**输出：** 本 forward 唯一的 immutable 执行上下文。

#### 4.1.6 EN：逐 row 首次进入 Sparse Resident

**触发时机：** mixed batch中任一请求从Dense切换到Sparse的第一个forward。F0生成固定容量`enter_mask[B_total]`，每个请求只成功进入一次。

**输入：** `enter_mask`中有效row各自的`P/S/R/K`、KVAF committed block count、所有本地layer的Dense cache/block table、目标Resident cache/block table及容量。

**动作：** 逐有效row核对`K <= R <= P <= S`、`P`按logical block对齐、`tail = S-P`以及`R+tail`不超过Resident容量；再把`[0,R)`建立为初始Sparse working set，把tail映射到`[R,R+tail)`，并确认所有本地layer使用一致布局。只有某row完整Resident mapping发布成功后，才允许释放该row在Sparse模式不再需要的Dense HBM blocks。

**输出：** `enter_result[B_total]`。成功row可由U0标记为`SPARSE_REMOTE`；安全失败row保持原Dense mapping，且仅在完整Dense KV仍有效时继续Dense；mapping结果未知的row进入terminal/quarantine，整个forward走Failure Finalize。

#### 4.1.7 U0：构造 UnifiedRowExecution

**触发时机：** F0 完成后，每个 forward 一次。

**输入：** full-batch row 状态、Dense KV 是否完整驻留、Sparse 状态和 PAD 信息。

**动作：** 为 `[B_total]` 每一行写入 `row_mode`、`row_valid`、Attention variant 和 full-batch row；同时构造 `dense_to_batch`、`sparse_to_batch`、`pad_to_batch` 与输出 scatter map。Dense/Sparse/PAD 数量可以变化，但 descriptor 和 map 的容量固定为 `B_total`。

**输出：** C 和 D 共用的 `UnifiedRowExecution + UnifiedExecutionMap`。两条分支不得各自重新划分 row。

#### 4.1.8 C0：PrepareUnifiedAccess

**触发时机：** U0 完成后，每个 forward 一次；不随 layer 数重复执行。

**输入：** `[B_total]` Unified access rows、每个 Sparse row 的 `pool_row` 和期望 committed block count。

**动作：** 对Sparse row核对OPEN状态和committed count，准备row read pins并冻结`row_flat_base + snapshot_count`；Dense/PAD保留同形descriptor，但不建立DDR read access。全batch校验成功后才一次安装全部pins并返回handle；任一row失败时自动回滚本次已准备状态，不留下partial pin或partial handle。该步骤不执行LIDU，也不发Get WQE。

**输出：** `PrepareUnifiedAccessResult`，其中包含本 forward 唯一的 `RowAccessHandle` 和 `[B_total]` snapshots。

#### 4.1.9 L0/C1：逐 layer Unified Select、Resolve 与 Gated Get

**触发时机：** C0 之后，按 Initialize 冻结的 `local_layer_ordinal = 0..n_layers-1` 重复。

**输入：** 当前 layer 的 query/indexer 输入、U0 的 row map、C0 的 snapshots、LIDU buffers，以及当前 layer 的 `BoundLayerCacheView`。

**动作：** Identity 子节点处理 Dense rows；LIDU 子节点处理 Sparse rows；Empty 子节点处理 PAD rows。对 Sparse miss，Adapter把 source token转换成 `{flat_index, offset}`。KVAF先解析并校验本批全部 source/destination；全部通过后才统一放行 URMA Get，直接写入当前 layer 的最终 packed Resident HBM。任一批内校验失败时，本批 `post_count=0`。

**输出：** 当前 layer 已补齐的 Resident KV、selection、destination slots、`resident_seq_lengths` 和本地 Get 结果。count 为 0 的子节点仍保持固定 launch，但不读写业务数据。

#### 4.1.10 D0：Unified Metadata

**触发时机：** 当前 layer 的 C1 成功后。

**输入：** U0 的统一 row map、C1 已验证的 destination slots、`P/S/R`、tail 和 `resident_seq_lengths`。

**动作：** 为 Dense rows 绑定完整本地 KV view；为 Sparse rows把 LIDU destination slots 与 tail slots 组合成 `attention_slots`；为 PAD rows写入确定的空 metadata。该步骤只生成 Attention 索引和长度，不搬运 KV payload。

**输出：** 当前 layer 可供 AttentionDispatch 消费的固定容量 metadata buffers。

#### 4.1.11 D1：AttentionDispatch

**触发时机：** 当前 layer 的 KV 和 metadata 都就绪后。

**输入：** Dense/Sparse row maps、query、canonical packed HBM、`attention_slots`、lengths 和输出 scatter map。

**动作：** 固定保留 Dense Attention 与 Sparse Attention 两个子节点。Dense 子节点只处理 `dense_count` rows；Sparse 子节点只处理 `sparse_count` rows；PAD 不参与计算。任一 count 为 0 时，对应子节点仍按固定模板完成，但不得访问无效业务输入。两个子节点的结果最终 scatter 回 full-batch 顺序。

**输出：** 当前layer的full-batch Attention output，以及记录在Attention stream上的completion event。D1不在Host侧等待该event。B1 source blocks若来自本层Attention刚产生或可能与其读写backing重叠，producer stream必须wait该event；只有合同能够证明source与Attention完全不重叠时才可免wait。E1/Failure Finalize在ReleaseAccess前必须等待相关event完成。

#### 4.1.12 B1：本 layer Block Put Write

**触发时机：** 当前 layer Attention 完成后。

**输入：** F0在boundary判定后、进入layer循环前创建的Group Put plan，`UnifiedBlockPutBatch`、当前 layer 的完整 source blocks 和 `BoundLayerCacheView`。

**动作：** boundary row 的 mask 为 true 时，将当前 layer 的完整 block同步写入 plan 已预留的 DDR block；非boundary forward 的`put_count=0`，Adapter的B1 wrapper直接no-op，不调用KVAF Write，也不需要PutPlanHandle。B1只完成当前 layer writer，不发布目录、不推进P。

**输出：** 当前 layer writer 完成状态。还有下一 layer 时回到 L0；最后一层完成后进入 E1。

#### 4.1.13 E1：Forward Finalize、P/tail 与 ReleaseAccess

**触发时机：** 所有本地 layers 完成后，每个 forward 一次。

**输入：** 各 layer Get/Attention完成状态、Group Put plan和C0创建的`RowAccessHandle`。

**动作：** 先结算Get/Attention。只有全部required layer writers成功时，boundary plan才允许Seal并Commit；任一Get、Attention或writer失败时，必须对plan执行Abort，无法证明安全终结时转Quarantine，并等待plan离开active_put。Commit成功后，完整验证全部`row_commits`并以一个账本事务更新各请求`P`和`tail`；无Put时保持P不变。最后在所有读者完成后释放Row Access read pins。

**输出：** 下一 forward 可以读取的新 `P/S/tail`，以及已经结算的本 forward 资源。

任一C1、D0、D1或B1失败时，不再执行该layer的后续业务节点，也不进入正常E1 Commit路径。Failure Finalize必须Settle已提交的Get和Attention event；若已Prepare Put plan，则执行Abort，无法证明安全终结时转Quarantine；随后Release Access并保持P不变。

#### 4.1.14 Q：判断请求是否 Finish

**触发时机：** E1 完成后。

**动作：** 未Finish的请求进入下一次F0；临时unschedule只保留row和DDR目录，不执行Close；只有Scheduler明确给出最终Finish时才进入E2。

#### 4.1.15 E2：Close KVAF row并释放请求资源

**触发时机：** 请求最终Finish，且Put/Get/Attention/Access均已结算后。

**输入：** 请求的`pool_row`和Runtime内部row/DDR block状态。

**动作：** `CloseResidentPoolRow`先验证无active Put、Access或I/O，再令 committed count 归零、退役Host cells并释放DDR blocks。Close SUCCESS后，Adapter才释放Resident HBM并把`pool_row`归还resident pool；BUSY保持OPEN，UNKNOWN进入QUARANTINED。

**输出：** SUCCESS时该请求的KVAF、HBM和pool row生命周期全部结束。

#### 4.1.16 与后续图模式的关系

M0按上述固定shape和固定launch顺序执行，但不要求ACL Graph capture。后续启用图模式时：

- 图内固定单layer数据面节点为Identity/LIDU/Empty、Resolve/Gated Get、Metadata、Dense Attention、Sparse Attention和scatter；
- 每次replay只更新count、mask、row map、Get items和业务tensor内容，图可见buffer的基址与容量保持不变；
- A、B0目录Commit、E0、EN、C0 read pin、E1 Host账本Commit、Q和E2仍属于图外控制面；
- 当前同步B1保留在图外，因此M1首先采用“replay当前layer数据面子图 → 按B1 source依赖等待producer/Attention event → 图外B1 → replay下一layer”的边界，不捕获整个`n_layers`循环；把B1纳入整forward图前，必须先拆成图内enqueue和图外Settle/目录Commit。

### 4.2 Unified row 语义

~~~cpp
enum class RowMode : uint8_t {
  DENSE_LOCAL,
  SPARSE_REMOTE,
  PAD,
};

struct UnifiedRowExecution {
  RowMode mode;
  bool row_valid;
  uint32_t remote_get_count;
  AttentionVariant attention_variant;
  uint32_t full_batch_row;
};

struct UnifiedExecutionMap {
  uint32_t dense_count;
  uint32_t sparse_count;
  uint32_t pad_count;
  uint32_t output_count;  // dense_count + sparse_count
  uint32_t dense_to_batch[B_total];
  uint32_t sparse_to_batch[B_total];
  uint32_t pad_to_batch[B_total];
  uint32_t output_scatter_to_batch[B_total]; // [Dense outputs, Sparse outputs] → batch
};

struct UnifiedBlockPutBatch {
  uint32_t put_count;
  bool put_mask[B_total];
};
~~~

所有 row 进入同一个固定形状算子序列：

| mode | Select/Resolve/Get | Metadata | AttentionDispatch |
|---|---|---|---|
| `DENSE_LOCAL` | `remote_get_count=0`，不发 DDR Get | 生成 Dense 所需 view | Dense 子节点处理该行 |
| `SPARSE_REMOTE` | LIDU 产生 selection；只对 miss 构造 Get item | 生成 `attention_slots` | Sparse 子节点处理该行 |
| `PAD` | `row_valid=false`，所有计数为0 | 写确定的空输出 | no-op |

逻辑执行模板不按 row mode 建三条控制流分支。`row_mode`、`row_valid`、`remote_get_count`和mask决定每个阶段的实际工作量。`UnifiedSelect`固定包含Identity、LIDU和Empty三个子节点；`AttentionDispatch`固定包含Dense与Sparse两个子节点。任一子集为空时，对应子节点仍以`count=0`执行。

Decode Block Put是forward级副作用，不由Dense/Sparse/PAD决定。它使用独立的固定形状`UnifiedBlockPutBatch`；非boundary forward令`put_count=0`、全mask为false，boundary forward只打开需要写完整block的row。

### 4.3 A～E 职责

| 分支 | 正常动作 | 输出 |
|---|---|---|
| A | 冻结layout和索引几何；分配目录；Bind pool row | Runtime、row、cache binding |
| B | 为每层完整block构造flat index；写DDR；发布目录和count | `PutLocalResult` |
| C | 建统一Access；LIDU/identity select；生成`{flat_index,offset}`；Direct Get | KV、length、Get result |
| D | 构造统一row descriptor；metadata；Attention dispatch；scatter | Attention output |
| E | ENTER、P/tail、forward收口、Close、Shutdown | 下一forward或safe finish |

## 5. A～E 主流程详细设计

### 5.1 A：Initialize、Bind 与统一 row contract

~~~cpp
struct DsaRowIndexGeometry {
  uint32_t raw_row_capacity;
  uint32_t n_layers;
  uint32_t q_capacity;
  uint32_t row_stride_cells;      // n_layers * q_capacity
  uint32_t directory_cell_count;  // raw_row_capacity * row_stride_cells
};

struct DsaRowInitRequest {
  PackedKvLayout layout;
  DsaRowIndexGeometry geometry;
  uint32_t ddr_block_count;
};

struct RowBindRequest {
  uint32_t pool_row;
  uint32_t max_sequence_tokens;
  uint32_t resident_budget_tokens;
};

struct BoundLayerCacheView {
  PackedCacheView cache;
  LayoutId layout_id;
  uint32_t local_layer_ordinal;
};

// PackedCacheView至少包含base、bytes、row_capacity；所有handle均为Runtime
// registry中的opaque owner，必须通过对应Commit/Abort/Settle/Release终结。
~~~

正常步骤：

~~~text
ProbeCapabilities
→ Runtime::Initialize
→ Runtime::InitializeDsaRowIndex
→ BindResidentCaches
→ resident pool acquire pool_row
→ Runtime::BindResidentPoolRow
~~~

关键接口：

| 接口 | 关键输入 | 成功输出 |
|---|---|---|
| `DsaKvafAdapter::ProbeCapabilities` | layout候选、kernel/transport/device配置 | capability和layout fingerprint |
| `Runtime::InitializeDsaRowIndex` | `RawRowCapacity/n_layers/Qcap`、cell ABI、capacity | 固定Host/Device DDRBlockTable |
| `DsaKvafAdapter::BindResidentCaches` | 各本地layer packed cache backing/capacity | Worker-lifetime cache binding |
| `DSAResidentTokenPool.acquire` | request id | `pool_row` |
| `Runtime::BindResidentPoolRow` | `pool_row`、最大序列长度、quota | RowMeta OPEN，count=0 |

reorder 只更新 `batch_row → pool_row` 小投影，不移动 DDRBlockTable row。PAD row不绑定pool row。

`BuildUnifiedRowExecution`只在`[0,S)`完整KV仍驻留本地HBM时选择`DENSE_LOCAL`；否则必须选择`SPARSE_REMOTE`。M0一旦进入Sparse，不允许直接切回Dense，除非先由显式全量restore流程恢复完整Dense KV。

### 5.2 B：Group Block Put

#### 5.2.1 Adapter 控制输入

~~~cpp
struct ResidentSourceBlockRef {
  int32_t physical_block_id;
};

struct PackedSourceBlockView {
  uint64_t source_byte_offset;  // 相对本层PackedCacheView.base
  uint32_t byte_length;         // 必须等于layer_payload_bytes
};

struct RowAppendIntent {
  uint32_t pool_row;
  uint32_t expected_committed_block_count;
  uint32_t block_count;
  Span<ResidentSourceBlockRef> source_blocks;
};

struct FullBlockGroupPutRequest {
  Span<RowAppendIntent> appends;
  PutReason reason;  // FINAL_PREFILL / DECODE_BOUNDARY
};
~~~

`source_blocks[j]`对应 `block_q = expected_committed_block_count + j`。Adapter 为每个 append、每个本地 layer 生成一个 resolved target：

~~~cpp
struct KvafPutBlockTarget {
  uint32_t flat_index;
};

struct KvafPutBlockItem {
  uint32_t flat_index;
  PackedSourceBlockView source_block;  // 完整layer block
};

struct RowCountCommitSpec {
  uint32_t row_flat_base;
  uint32_t expected_committed_block_count;
  uint32_t appended_block_count;
};

struct RowCommitResult {
  uint32_t row_flat_base;
  uint32_t old_committed_block_count;
  uint32_t new_committed_block_count;
};

enum class PutLocalState {
  COMMITTED,
  ABORTED_SAFE,
  UNKNOWN_QUARANTINED,
};

enum class SealResult {
  SEALED_READY,
  ABORTED_SAFE,
  UNKNOWN_QUARANTINED,
};

struct PutLocalResult {
  PutLocalState state;
  vector<RowCommitResult> row_commits;
};
~~~

公式：

~~~text
block_q = expected_committed_block_count + j
q = local_layer_ordinal * Qcap + block_q
flat_index = pool_row * n_layers * Qcap + q
~~~

#### 5.2.2 KVAF Put 接口

~~~cpp
Status Runtime::PrepareFullBlockGroupPut(
    Span<const KvafPutBlockTarget> all_targets,
    Span<const RowCountCommitSpec> row_commits,
    PutPlanHandle* plan);

Status Runtime::WriteFullBlockGroupLayer(
    PutPlanHandle& plan,
    Span<const KvafPutBlockItem> layer_blocks,
    BoundLayerCacheView layer_cache,
    void* producer_stream);

Status Runtime::SealFullBlockGroupPut(
    PutPlanHandle& plan,
    SealResult* result);

Status Runtime::CommitFullBlockGroupPut(
    PutPlanHandle& plan,
    PutLocalResult* result);
~~~

`all_targets`包含本plan全部 `(row,layer,block)` 的flat index。`WriteFullBlockGroupLayer`只接收本层的`{flat_index,source_block}`，不接收pool row、layer ordinal、block_q或Qcap。KVAF要求本次Write的全部flat index解码为`layer_cache.local_layer_ordinal`。每个item写完整 `layer_payload_bytes`，因此没有token offset参数。

`row_commits`也不传pool row。Adapter先计算`row_flat_base = pool_row * n_layers * Qcap`；KVAF使用Initialize时冻结的几何检查该row的全部layer区间和连续block范围。

Put plan要求：

- 每个flat index唯一；
- 每个row的每个required layer都覆盖同一连续block_q范围；
- `all_targets.size == sum(row.block_count) * n_layers`；
- layer Write只能提交属于该层段的flat index；
- 所有required layer完成后才能Seal；
- Commit成功后才推进每个row的committed count。

对每个`RowCountCommitSpec`，KVAF先检查`row_flat_base % RowStrideCells == 0`、`row_flat_base < DirectoryCellCount`、对应row处于OPEN，并用checked-add验证`expected + appended <= Qcap`。其合法目标集合由冻结几何唯一计算为：

~~~text
{ row_flat_base + layer * Qcap + block_q |
  0 <= layer < n_layers,
  expected <= block_q < expected + appended }
~~~

`all_targets`必须与该集合完全相等。Put core随后只按这些flat index解析或分配DDR目标地址。

每个`PackedSourceBlockView`还必须满足：`byte_length == layer_payload_bytes`，且`source_byte_offset + byte_length`用checked-add计算后落在`layer_cache.cache.bytes`内。

`COMMITTED`结果必须覆盖plan内全部且仅有的row：`row_commits`不得缺项、重复或包含额外row，且每项满足`new = old + appended_block_count <= Qcap`。非`COMMITTED`结果的`row_commits`为空。

M0继续复用同步Put数据面。source只在单次`WriteFullBlockGroupLayer`调用期间借用；调用返回SUCCESS或确认stream静止后，本次source访问结束。

#### 5.2.3 增量目录发布

一个row、一个layer、一个连续block范围形成一个Device patch：

~~~cpp
struct DeviceBlockRangePatch {
  uint32_t flat_index_begin;
  Span<const DeviceBlockCell> cells;
};
~~~

发布顺序：

~~~text
全部layer block payload写入完成
→ Host cells写入plan-private状态
→ 按layer形成连续flat-index patches
→ patches局部H2D到固定DeviceDDRBlockTable
→ publication event完成
→ Host事务安装全部cells并推进各row committed count
→ plan COMMITTED
~~~

final-prefill通常是每层一个长range；Decode boundary通常是每层一个cell。不得重建或上传完整目录。

### 5.3 C：Unified Access、Select 与 Direct Get

#### 5.3.1 Unified Access

~~~cpp
struct UnifiedAccessRow {
  RowMode mode;
  bool row_valid;
  optional<uint32_t> pool_row;
  uint32_t expected_committed_block_count;
};

struct RowAccessSnapshot {
  RowMode mode;
  bool row_valid;
  bool ddr_access_valid;
  uint32_t row_flat_base;     // 仅ddr_access_valid时有效
  uint32_t snapshot_count;
};

struct PrepareUnifiedAccessResult {
  RowAccessHandle handle;
  Span<const RowAccessSnapshot> snapshots;  // [B_total]，由handle拥有
};
~~~

`PrepareUnifiedAccess`每forward调用一次，输入固定`[B_total]` rows：

- Sparse row：检查pool row、OPEN和expected count，增加read pin；
- Dense row：保留相同descriptor形状，`remote_get_count=0`；
- PAD row：`row_valid=false`，所有计数为0；
- 不枚举prefix block，不复制DDRBlockTable。

`RowAccessHandle`保存full-batch快照、Sparse row的`row_flat_base`集合和read pins。KVAF可由`flat_index / RowStrideCells`得到目录row，并只允许item命中该handle内已绑定的Sparse row。Dense/PAD快照的`ddr_access_valid=false`、`snapshot_count=0`。

`Runtime::PrepareUnifiedAccess`返回`PrepareUnifiedAccessResult`。Adapter只能读取该结果中的`snapshots[batch_row]`构造Get item，不得重新依据当前batch投影计算另一份row base或count。

#### 5.3.2 Get source ABI

~~~cpp
struct KvafGetSource {
  uint32_t flat_index;
  uint16_t offset;            // 0..127
};

struct KvafGetItem {
  KvafGetSource source;
  int32_t destination_index;  // 最终packed Resident HBM线性row
};

struct DirectGetRequest {
  Span<const KvafGetItem> items;
  BoundLayerCacheView output_cache;
  void* stream;
};
~~~

Adapter在调用KVAF前完成：

~~~text
block_q = source_token_id / tokens_per_logical_block
offset = source_token_id % tokens_per_logical_block
q = local_layer_ordinal * Qcap + block_q
flat_index = snapshot.row_flat_base + q
~~~

Adapter同时按`source_token_id < P`、`P % tokens_per_logical_block == 0`和`P / tokens_per_logical_block == snapshot_count`校验本row业务前缀；KVAF按handle中的snapshot和flat index再次验证目录边界。

`Runtime::PrepareDirectBatchGet`与`ExecutePreparedDirectBatchGet`接收的每个DDR source只有`flat_index`和`offset`。逐item source不接收pool row、layer ordinal、block_q、Qcap、P或S。

`output_cache.local_layer_ordinal`是批次级目标cache绑定，不是单条source坐标。KVAF要求所有item的flat index都解码为该layer，并核对layout和cache容量。

#### 5.3.3 Unified Select/Get 步骤

~~~text
PrepareUnifiedAccess[B_total]
→ UnifiedSelect
→ BuildKvafGetItems
→ ResolveValidate全部items
→ AggregateBatchValidation
→ allow时ExecutePreparedDirectBatchGet
→ SettleDirectBatchGet
~~~

三种row mode使用同一固定shape调用：

- Identity子节点：只处理`dense_count`行，输出本地Dense view；
- LIDU子节点：只处理`sparse_count`行，只为miss构造Get items；
- Empty子节点：只处理`pad_count`行，输出确定的空值；
- 三个子节点的buffer容量均为`B_total`，每次调用都覆盖本轮有效范围；
- 任一count为0时仍launch对应子节点，且必须零访存业务输入、零写业务输出；
- 全batch无Get item时，Aggregate返回empty-success，不发WQE。

#### 5.3.4 KVAF Get 内部

KVAF按item执行：

~~~text
validate flat_index < DirectoryCellCount
validate offset < tokens_per_logical_block
row_flat_base = floor(flat_index / RowStrideCells) * RowStrideCells
q = flat_index - row_flat_base
local_layer_ordinal = q / Qcap
block_q = q % Qcap
validate row_flat_base属于当前RowAccessHandle
validate block_q < 该row的snapshot_count
cell = DeviceDDRBlockTable[flat_index]
remote_address = cell.block_base_uba + offset * packed_token_bytes
validate 0 <= destination_index < output_cache.cache.row_capacity
validate destination_index * packed_token_bytes不溢出且落在output_cache.cache.bytes内
output_address = output_cache.cache.base + destination_index * packed_token_bytes
~~~

全部item校验完成后统一写`global_allow`。任一item错误时全批`post_count=0`。allow后复用现有BatchGet compact/post/wait、URMA transport、CQ与656B搬运主体。

~~~cpp
enum class GetLocalState {
  SUCCESS,
  FAILED_SAFE,
  UNKNOWN_QUARANTINED,
};

struct GetCompletion {
  StreamEvent event;
  StatusBufferHandle status;
};

struct GetLocalResult {
  GetLocalState state;
  uint32_t post_count;
};
~~~

status、workspace和completion由plan独占，launch前全部覆盖初始化，Settle后统一归还。

### 5.4 D：Unified Metadata 与 AttentionDispatch

`DsaKvafAdapter::BuildUnifiedRowExecution`由D分支每forward调用一次，输入full-batch row kind与query顺序，输出固定`[B_total]`的`UnifiedRowExecution`和`UnifiedExecutionMap`。C的Identity/LIDU/Empty子节点与D的Metadata/Attention子节点必须消费同一份map：分别使用`dense_to_batch`、`sparse_to_batch`和`pad_to_batch`，不得各自重新分区。

~~~cpp
struct UnifiedAttentionRow {
  RowMode mode;
  bool row_valid;
  AttentionVariant variant;
  uint32_t attention_length;
};
~~~

`BuildUnifiedAttentionMetadata`同时接收`UnifiedExecutionMap`：Dense/Sparse子节点分别按`dense_to_batch`和`sparse_to_batch`读取full-batch输入，最终按`output_scatter_to_batch`写回原batch顺序。

固定执行序列：

~~~text
BuildUnifiedAttentionMetadata[B_total]
→ AttentionDispatch {
     DenseAttention[dense_count]
     SparseAttention[sparse_count]
   }
→ ScatterToFullBatch
~~~

mode语义：

- Dense：metadata引用完整本地KV，进入Dense子节点；
- Sparse：metadata把LIDU destination slots与tail slots拼成`attention_slots`，进入Sparse子节点；
- PAD：metadata写空值，dispatch no-op；
- 两个Attention子节点每个forward都保留在执行图中，行数可以为0；
- count为0时仍launch对应子节点，且必须零访存业务输入、零写业务输出；
- `resident_seq_lengths`由C统一写，D只读；
- D不访问DDR、不发URMA、不搬KV。

### 5.5 E：ENTER、P/tail、Forward Finalize 与 Finish

| 符号 | 含义 |
|---|---|
| `S` | 当前总token数 |
| `P` | 本卡已完整Commit到DDR的前缀token数 |
| `tail=S-P` | 尚未Commit、只在HBM的后缀 |
| `R` | Sparse Resident HBM预算 |
| `K` | LIDU从`[0,P)`选择的token数 |

Boundary forward：

~~~text
forward前Prepare Group Put plan
→ 每层Unified Select/Get
→ Unified Metadata
→ AttentionDispatch
→ 本层WriteFullBlockGroupLayer
→ forward末Seal/Commit
→ 按row commit结果更新P与tail
→ ReleaseUnifiedAccess
~~~

每个forward都构造固定形状`UnifiedBlockPutBatch[B_total]`。非boundary时`put_count=0`；boundary时Adapter只把mask为true的row转换成`RowAppendIntent`，B分支仍按完整layer block写入。

Finish：

~~~text
settle Put/Get/Attention
→ ReleaseUnifiedAccess
→ Runtime::CloseResidentPoolRow
→ 释放Resident HBM
→ resident pool release
~~~

临时unschedule不Close。每卡依据本地Commit结果更新P，不新增跨卡确认。

---

# 第二部分：按 A～E 组织的可靠性与鲁棒性增量

## 6. A：索引、容量、Bind 与复用约束

### 6.1 索引不变量

- `n_layers`、`Qcap`、`RowStrideCells`在Initialize后不可改变；
- `flat_index`使用checked乘加；
- `q = local_layer_ordinal * Qcap + block_q`；
- `DirectoryCellCount = RawRowCapacity * n_layers * Qcap`；
- padding row不能进入目录；
- Host/Device目录基址、shape和stride在Runtime生命周期内稳定；
- Adapter与KVAF使用同一个CPU oracle测试flat index。

### 6.2 Bind 与 row 复用

- Bind只允许FREE row；
- OPEN row不能再次Bind；
- Bind成功时count=0，不读取旧cell；
- 新owner的每个cell必须先完整覆盖并完成publication，随后才能推进count；
- Close完成前pool row不得回到free list；
- 同step Finish+Add必须完成旧row Close后才能复用该raw row；
- M0假设Close后不存在旧plan、旧Access、旧buffer或晚到I/O；不能证明时row进入QUARANTINED。

### 6.3 A 验收

- layer 0、layer `n_layers-1`、block 0、block `Qcap-1` oracle正确；
- layer、block、row越界均在目录访问前拒绝；
- 容量乘法和`INT32_MAX`边界正确；
- Bind/Close/reuse过程中不清整张Device目录；
- DSA_ROW热路KeyId构造和Key hash查询数为0。

## 7. B：Group Put 与目录原子发布

### 7.1 Put plan状态

~~~text
PREPARED → WRITING → SEALED_READY → COMMITTED
         ↘ ABORTED_SAFE
         ↘ UNKNOWN / QUARANTINED
~~~

plan维护required-layer bitmap和目标flat-index集合。缺层、重复层、坏flat index、同一层缺block或payload写失败都不能Commit。

### 7.2 `active_put`

Prepare在Meta锁内校验：

- row为OPEN；
- `expected_count == committed_count`；
- 追加range不越过Qcap；
- row没有另一个active Put；
- 全部层的target flat index集合完整且互异。

随后安装`active_put`并预留DDR blocks。Commit、Abort或Quarantine终结前，同row新的append和Close均被拒绝；旧snapshot Access仍可读取已提交prefix。

### 7.3 Publication顺序

~~~text
全部payload writer完成
→ Host plan-private cells
→ 每层dirty flat-index range H2D
→ publication event完成
→ Host事务安装全部cells
→ 推进row committed count
~~~

Device suffix提前写入但count未推进时不可读。任一publication结果未知时，plan涉及的row、DDR blocks、staging和Device ranges全部隔离。

### 7.4 同步source借用

`WriteFullBlockGroupLayer`只在本次同步调用期间借用source。调用kernel launch API之前的失败表示未访问source；调用过launch API后必须证明stream静止，才能归还本次payload view和LocalBuffer调用引用。无法证明时，本卡Runtime terminal，并保留相关registration与allocator owner直到reset/quiescence。

### 7.5 B验收

- final-prefill多block和boundary单block覆盖所有layer；
- 每层target flat indices正确且不重叠；
- dirty H2D bytes随新增block数和layer数增长，不随旧prefix增长；
- count-before-cell、缺层、partial patch和late patch不产生可读prefix；
- Put data plane只接收flat index和完整source block。

## 8. C：Unified Access、LIDU 与 Direct Get

### 8.1 Row Access read pin

Sparse row的Access在forward期间持有read pin，保护：

- `[0,snapshot_count)`对应的所有layer cells；
- 对应DDR blocks；
- Device目录内容；
- Get完成前的output binding。

suffix append允许并发，但不能覆盖旧prefix。Release必须等全部Get、CQ/fence和Attention读取完成。

### 8.2 LIDU mutation

LIDU必须先Validate/Reserve，后一次发布mapping；不能留下可被Attention消费的部分mapping。selection scratch固定为`[B_total,1,K]`，Dense/PAD行也必须覆盖本轮确定值，禁止读取上一轮残留。

### 8.3 Get全批校验

在任何WQE之前校验：

- `0 <= miss_count <= K`；
- 每个active item的`flat_index < DirectoryCellCount`；
- `0 <= offset < tokens_per_logical_block`；
- `flat_index`可由冻结几何解码成合法row、layer和block；
- 解码后的row属于当前RowAccessHandle，且`block_q < snapshot_count`；
- Device cell已发布且UBA范围合法；
- destination index合法、唯一、不重叠；
- Dense/PAD的remote get count为0；
- 所有status buffer在本次launch前清零并由本plan独占到Settle。

任一错误要求全批`post_count=0`。

### 8.4 Completion与poison

SUCCESS要求全部accepted CQ完成、device status成功、HBM visibility fence完成。LIDU发布后发生Get失败时，不得进入Attention；已经写入的Resident mapping/HBM进入poison，由本卡terminal owner持有到reset、审计和Close。

### 8.5 C验收

- KVAF Get接口逐item只有`flat_index`、`offset`和destination；
- layer 0、最后一层、block边界、offset 0/127正确；
- Dense-only与PAD-only batch执行相同算子序列且WQE为0；
- 0/1/K miss正确；
- 任一validation错误全批0 WQE；
- `PrepareUnifiedAccess`复杂度为`O(B_total)`，不随prefix长度增长。

## 9. D：Unified Metadata 与 Attention

### 9.1 固定shape与mask

- `UnifiedRowExecution`、selection、length和metadata buffer按`B_total`固定shape；
- Dense/Sparse子节点的输入buffer容量固定为`B_total`，运行时只更新count和mask；
- Dense/Sparse/PAD只改变mode、valid、count和mask；
- batch shrink和PAD必须覆盖所有有效范围；
- metadata输出在Attention stream完成前不可复用。

### 9.2 AttentionDispatch

`AttentionDispatch`是统一图节点，不要求Dense与Sparse使用同一个数学kernel。Dense是统一执行envelope中`remote_get_count=0`的特例，不是Sparse Attention数学的特例：

- Dense调用Dense Attention variant；
- Sparse调用C8 Sparse Attention variant；
- PAD no-op；
- dispatch前统一核对本row Get/metadata result；
- 输出统一scatter回full-batch顺序。

### 9.3 D验收

- mixed、Dense-only、Sparse-only、PAD-only均不改变逻辑节点和launch序列；
- row mode与mask组合正确；
- Dense不发DDR Get；
- PAD不读selection、不写业务输出；
- KSC、KV staging、full-cache clone和全设备同步不可达。

## 10. E：本地编排、Close、Quarantine 与 Shutdown

### 10.1 本地forward收口

- 每层Get/metadata成功后才允许AttentionDispatch消费对应row；
- forward末Seal成功后执行Commit；
- Commit SUCCESS后先完整验证`row_commits`的row集合、old/new count和Qcap边界，再用checked-mul计算`P = new_committed_block_count * tokens_per_logical_block`，最后以一个账本事务更新所有请求P；
- Commit失败不推进P；
- 没有active Put/Get/Attention/Access时才允许Finish。

### 10.2 Close

~~~cpp
enum class CloseLocalState {
  SUCCESS,
  BUSY_NO_STATE_CHANGE,
  UNKNOWN_QUARANTINED,
};

struct CloseLocalResult {
  CloseLocalState state;
};
~~~

Adapter先结算外部owner并Release Access，再调用Runtime Close。Runtime在Meta临界区检查：

~~~text
RowMeta.state == OPEN
active_put == null
active_access == 0
该row全部DDR block.in_flight_io == 0
~~~

条件不满足时返回`BUSY_NO_STATE_CHANGE`且row保持OPEN。满足后执行：

~~~text
OPEN → CLOSING
→ committed count = 0，拒绝任何新Access
→ retire Host cells
→ release DDR blocks与quota
→ FREE
~~~

Device cells不在Close时逐项清零；在没有旧plan、旧Access或晚到I/O后，它们只是不可达旧值。下一次Bind仍从count=0开始，并在推进count前覆盖将要发布的全部cell。

进入CLOSING后的任何未知结果都转为`UNKNOWN_QUARANTINED`，不得释放HBM或pool row。

### 10.3 Shutdown

~~~text
stop admission/new I/O
→ settle已知plan
→ drain/reset transport与device
→ 审计UNKNOWN Put/Get/publication
→ Release Access
→ Close rows
→ assert owner/pin/quarantine均为0
→ unregister/free DDR与Device目录
~~~

无法证明设备停止访问时Shutdown失败，并继续持有相关registration和内存。

reset/device-wide quiescence成功后，`ReconcileQuarantinedResources`审计Host cells、Device cells、DDR块owner和I/O状态：能够证明无悬空访问的资源转为可Close/可Release；不能证明的资源保持QUARANTINED，Runtime保持terminal且Shutdown返回失败。

### 10.4 E验收

- final-prefill、ENTER、连续boundary、unschedule与Finish通过；
- P只在本卡Commit成功后推进；
- Busy Close保持OPEN；
- UNKNOWN Close进入QUARANTINED；
- same-step Finish+Add只在safe Close后复用row；
- Shutdown和析构不提前free。

---

# 第三部分：KVAF接口、开发分工与验收

## 11. KVAF接口最小改动

### 11.1 保留接口

- `Runtime::Initialize/Shutdown`；
- DDR allocator与region registration；
- 现有固定粒度DDR分配器（源码当前以slot命名）；
- `FLAT_COMPAT`的`BatchPut/BatchGet/BatchFree/PrepareAccessTable`；
- Put/Get transport、queue、CQ、status/workspace；
- LocalBuffer registration cache；
- 现有Put kernel主体与Get 656B URMA搬运主体。

### 11.2 新增Runtime verbs

~~~cpp
Status InitializeDsaRowIndex(const DsaRowInitRequest&);
Status BindResidentPoolRow(const RowBindRequest&);
Status AbortBindResidentPoolRow(uint32_t pool_row);

Status PrepareFullBlockGroupPut(
    Span<const KvafPutBlockTarget>,
    Span<const RowCountCommitSpec>,
    PutPlanHandle*);
Status WriteFullBlockGroupLayer(
    PutPlanHandle&,
    Span<const KvafPutBlockItem>,
    BoundLayerCacheView,
    void* stream);
Status SealFullBlockGroupPut(PutPlanHandle&, SealResult*);
Status CommitFullBlockGroupPut(PutPlanHandle&, PutLocalResult*);
Status AbortFullBlockGroupPut(PutPlanHandle&, PutLocalResult*);

Status PrepareUnifiedAccess(
    Span<const UnifiedAccessRow>,
    PrepareUnifiedAccessResult*);
Status PrepareDirectBatchGet(
    RowAccessHandle&,
    const DirectGetRequest&,
    DirectGetPlanHandle*);
Status ExecutePreparedDirectBatchGet(
    DirectGetPlanHandle&,
    GetCompletion*);
Status SettleDirectBatchGet(
    DirectGetPlanHandle&,
    GetCompletion&&,
    GetLocalResult*);
Status ReleaseUnifiedAccess(RowAccessHandle&&);

Status CloseResidentPoolRow(
    uint32_t pool_row,
    CloseLocalResult*);
~~~

### 11.3 数据面薄入口

~~~cpp
Status AccessOperator::ExecuteBatchPutToAddresses(
    Span<const KvafPutBlockItem>,
    Span<const uint64_t> destination_addresses,
    void* stream);

Status AccessOperator::ExecuteBatchGetByFlatIndex(
    Span<const KvafGetItem>,
    const DeviceDDRBlockTableView&,
    BoundLayerCacheView output,
    void* stream,
    GetCompletion*);

Status Runtime::PublishDirectoryRanges(
    Span<const DeviceBlockRangePatch>,
    void* stream,
    PublicationToken*);

Status Runtime::ReconcileQuarantinedResources(
    ReconcileResult*);
~~~

`PublicationToken`是Runtime持有的opaque publication completion owner；`ReconcileResult`汇总已安全回收和仍需隔离的row、DDR block及目录range。

`ExecuteBatchGetByFlatIndex`内部解析`flat_index`和`offset`后，复用现有Get compact/post/wait与transport主体。`ExecuteBatchPutToAddresses`复用现有同步Put主体。

### 11.4 不新增

- 不新增SequenceHandle或第二请求身份；
- 不新增RecordDirectory或第二DDR块分配器；
- 不新增DSA KeyId或Key hash cache；
- 不新增per-forward完整prefix地址表；
- 不复制Put/Get transport主体。

## 12. A～E开发分工与文件边界

### 12.1 建议目录

~~~text
vllm_ascend/dsa_offload/kvaf_adapter/
  contracts.py  layout.py  bind.py                     # A
  append_plan.py  put.py                               # B
  access.py  lidu.py  get.py                           # C
  row_execution.py  metadata.py  attention_dispatch.py # D
  lifecycle.py  forward_orchestrator.py                # E
  adapter.py  __init__.py                              # INT

KVAF/runtime/src/
  dsa_row_index.cc                                     # A
  dsa_row_put.cc                                       # B
  dsa_row_get.cc                                       # C
  dsa_row_lifecycle.cc                                 # E

KVAF/meta_manager/
  include/kvaf/meta_manager/ddr_block_allocator.h      # A
  src/ddr_block_allocator.cc                           # A

KVAF/include/kvaf/dsa_row/
  types.h  requests.h                                  # A单写公共ABI
  put.h                                                # B
  get.h                                                # C
  lifecycle.h                                          # E

KVAF/access_operator/src/
  dsa_row_put_adapter.cc                               # B
  dsa_row_get_adapter.cc                               # C

KVAF/access_operator/demo/aiv_batch_get/
  include/dsa_row_batch_get_abi.h                      # C
  kernel/resolve_flat_index_get_aiv.asc                # C
  kernel/execute_dsa_row_get_gated_aiv.asc             # C
~~~

### 12.2 工作包

| WP | 独占职责 | 完成标准 |
|---|---|---|
| A | layout、flat-index oracle、目录、Bind | `row/layer/block → flat_index`全边界正确 |
| B | Group Block Put、dirty publication | Put core只消费flat index和完整block source |
| C | Unified Access、LIDU、Get | Get core逐item只消费flat index、offset和destination |
| D | Unified row、metadata、AttentionDispatch | 三种mode使用同一图节点序列 |
| E | P/tail、Finalize、Close、Shutdown | 本卡生命周期无提前释放 |
| INT | 旧大文件薄接缝、bindings、构建、CI | 业务实现不回填旧巨型文件 |

共享接缝`vllm_ascend/dsa_offload/ops_nanovllm.py`和`vllm_ascend/attention/sfa_v1.py`只由WP-INT修改；WP-D交付参数组装、row map和dispatch合同，由WP-INT把薄调用接入现有Attention路径。

`KVAF/runtime/include/kvaf/runtime/runtime.h`、`KVAF/access_operator/include/**/access_operator.h`、Python bindings、CMake和公共测试入口只由WP-INT合并。WP-A拥有公共DSA_ROW types/requests与共享`DdrBlockAllocator`；B/C/E只修改各自子图header和实现文件。

### 12.3 依赖

~~~text
WP-A冻结contract与ABI
  → WP-B / WP-C / WP-D并行
  → WP-E集成生命周期
  → WP-INT构建、E2E与性能
~~~

## 13. M0总体验收

### 13.1 索引与接口

- CPU oracle、Host lookup和Device resolver的flat index完全一致；
- 覆盖所有row、layer、block边界；
- Get source ABI没有pool row、layer、block_q或Qcap字段；
- Put block ABI没有pool row、layer、block_q或Qcap字段；
- Put/Get必须拒绝flat index解码层与`BoundLayerCacheView.local_layer_ordinal`不一致的批次；
- Get offset只允许`0..127`；
- Put一次写完整`layer_payload_bytes`；
- 普通forward没有Key hash和完整prefix表materialize。

### 13.2 Unified row执行

- Dense/Sparse/PAD使用相同固定shape算子序列；
- Dense的remote get count为0并进入Dense Attention；
- Sparse只为miss构造Get items并进入Sparse Attention；
- PAD全程确定性no-op；
- mixed batch输出scatter正确；
- 逻辑节点和launch数量不随row mode组合变化。

### 13.3 Put/Get正确性

- final-prefill与boundary覆盖所有本地layer；
- 每层Put target flat index正确；
- Get覆盖layer 0/最后一层、block 0/最后一块、offset 0/127；
- 0/1/K miss和全batch零Get正确；
- validation失败全批0 WQE；
- Get目标cache跨层绑定错误时全批0 WQE；Put source cache跨层绑定错误时不启动writer；
- 真实656B payload逐字节一致；
- payload→Device cells→publication→Host count顺序不可打乱。

### 13.4 生命周期与故障

- Access pin期间旧prefix不迁移、不释放；
- suffix append不影响旧snapshot；
- Close前结算Put/Get/Attention/Access；
- Busy Close保持OPEN；
- UNKNOWN资源在reset/quiescence前不复用；
- same-step Finish+Add安全；
- Shutdown和析构不提前释放registration、DDR或HBM。

### 13.5 性能门禁

至少比较：

1. `KeyId hash + full PrepareAccessTable`；
2. row-major目录 + per-layer range patch；
3. row-major目录 + batch memcpy；
4. row-major目录 + packed patch/scatter。

记录：

- Host prepare p50/p99；
- dirty cells、H2D bytes/calls和publication p50/p99；
- Unified Access bytes/calls；
- flat-index resolver时间、GM bytes和L2 hit；
- 真实Get时间；
- Host/NPU目录内存；
- TPOT p50/p95/p99、tokens/s和NPU空洞；
- count=0 Identity/LIDU/Empty/Dense/Sparse子节点的launch开销与E2E回归。

## 14. M1

BF16、prefix sharing、preemption/resume、跨Worker迁移、异步Put、多lane、生产Graph和chunked directory属于M1，不进入本版接口。

## 附录：主要接缝

| 位置 | Stage 2 动作 |
|---|---|
| `vllm_ascend/dsa_offload/resident_pool.py` | 分配和安全释放pool row |
| `vllm_ascend/dsa_offload/input_batch.py` | 维护batch row到pool row投影 |
| `vllm_ascend/dsa_offload/runtime.py` | 改为薄Adapter调用 |
| `vllm_ascend/dsa_offload/a5_c8_policy.py` | 删除生产monkey patch，接Unified执行计划 |
| `vllm_ascend/dsa_offload/ops_nanovllm.py` | LIDU、metadata、Attention薄wrapper |
| `vllm_ascend/attention/sfa_v1.py` | canonical packed cache与AttentionDispatch接缝 |
| `vllm_ascend/worker/model_runner_v1.py` | forward/Finish薄接缝 |
| `KVAF/runtime/src/runtime.cc` | DSA_ROW mode dispatch |
| `KVAF/meta_manager/src/in_memory_meta_manager.cc` | 复用现有DDR block allocator |
| `KVAF/access_operator/src/access_operator.cc` | Put/Get resolved-input薄overload |
