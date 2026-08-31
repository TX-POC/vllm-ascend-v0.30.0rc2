# P7.1：独立 CandidateSeed -> SpeculativeGet 设备执行器 ABI

`A5ScoreReceiptSpeculativeExecutor` 是一个**不编入 C8** 的 Ascend **AIV-only**
算子。它消费已经由上游 ExactScoreScan / CandidateCompact 写入 GM 的
`CandidateSeed`，并调用 KVAF 现有的 Direct-HCCP `PostOnly`、非阻塞 CQ
`TryRetire` 和 terminal `Finish`。它不是 Python 线程，也不预留物理 AIV
作为 Get 核；任一 AIV worker 都可以动态 CAS 获取任一空闲 queue。

## 资源边界

每个 worker 自己拥有两个 `64 x 4 x int32` 的 UB CandidateSeed 暂存区：
一组正在装载/等待 queue，另一组可以在上一个 HCCP wave 未完成时继续装载。
这 2 KiB 属于新执行器的 UB，不复用也不改变 C8 的 UB。每个 queue 仍只有
一个 in-flight wave；这个限制来自现有 KVAF SQ/CQ ABI，而不是 AIV 分核。P7.1
采用 AIV-only：KVAF 的 lease/CQ helper 使用同一 AIV 的 SIMT/VF 调用，并不需要
保留一个闲置 AIC lane。首次硬件验证必须实际观察 `acquire -> PostOnly -> Finish ->
Release`，不能把 AIV-only 当成已经证明的性能结论。

host tiling 的 worker 数是 `min(available_AIV, receipt_count, queue_count)`。这不是
把 AIV 物理切成 queue 对应的硬分区：任意被调度的 worker 都能 CAS 租到任意 queue，
并按 stride 继续处理多个 receipt。该上限只是避免已压缩的候选只有 `Q` 个 in-flight
queue 时仍启动几十个 worker 空转抢 lease，造成无收益的 CAS/queue-busy 抖动。

## 31 个 ACLNN 张量 / 18 个原地 Ref 输出

Torch 侧 schema 仍是 `torch_schema=31in_18aliasout`，这样图能看到 18 个可变状态的
别名输出；但 **ACLNN 调用 ABI** 是 `abi=31tensor_18refout`。也就是说，18 对
可变 input/output 在 `ops.json` 中使用同一个端口名，CANN 自动生成一个
可写的 `...Ref` 参数，而不是再附加 18 个 `_out` descriptor。这样
`GetWorkspaceSize` 只接收 31 个 tensor（CANN 的单算子输入上限是 32），
不会走 49 descriptor 的未支持路径。前 7 个输入定义 receipt 与地址域：

| index | tensor | shape | 含义 |
|---:|---|---|---|
| 0..3 | `candidate_seeds/counts/ready_epoch/status` | `N×128×4`, `N` | 上游已发布的 CandidateSeed receipt |
| 4 | `streaming_epoch` | `int64[1]` | 当前 forward/layer transaction epoch |
| 5 | `source_lens` | `int32[R]` | 每行当前 forward 有效 score-source 长度；拒绝图静态 padded tail |
| 6 | `kv_object_address_domain` | `int32[1]` | 当前 P7.1 **只接受** `1=MAPPED_OBJECT`：`O` 必须等于 `kv_object_map[P,S]`，并作为 KVAF descriptor object index。`0=RAW_TOKEN` 尚无显式转换/table ABI，会 fail-closed，绝不被悄悄当成 object index |

其余输入是 `S -> O` 映射、resident snapshot、KVAF control/queue descriptor
ledger、private mapping/ready tag 与 telemetry。逻辑输出严格别名输入
11..15、17..29；Torch 返回同一 Tensor，而 ACLNN 只拿对应的可写 Ref
descriptor。不能用 copy 或额外 `_out` descriptor 代替 Ref。

## 图可见状态

KVAF v1 control 只含 `queue_leases` 和 descriptor scratch，不能表达 replay
身份或 quarantine。因此 P7.1 把以下资源作为独立的图输入/输出，且它们必须
在同一 forward 的 graph resource 生命周期中保持不变：

| tensor | shape | 用途 |
|---|---:|---|
| `queue_epoch` | `int64[Q]` | 当前 wave 的 forward/layer epoch |
| `queue_generation` | `int64[Q]` | queue 每次 lease 后递增 |
| `queue_post_sequence` | `int64[Q]` | 本 generation 的 post 序号 |
| `queue_state` | `int32[Q]` | `FREE/FILLING/POSTED/CQ_TERMINAL/QUARANTINED` |
| `queue_quarantine` | `int32[Q]` | release CAS 或身份校验失败后置 1，禁止复用 |
| `queue_request_*` | `Q x 64` | 当前 wave 的 source/object/output ledger；CQ terminal 只读此表 |
| `private_ready_epoch/generation` | `Q x A` | QSFA/reconcile 可与 `private_ready==READY` 一起校验同一 wave 身份 |

`queue_leases` 仍是 KVAF 的唯一互斥源。P7.1 不用 `queue_state` 替代 CAS。
反过来，CAS owner tag 只证明当前互斥，不证明 completion 属于哪一轮；所以 CQ
terminal 必须同时比较 epoch/generation/post sequence。

## 状态机

```
queue:   FREE -> FILLING -> POSTED -> CQ_TERMINAL -> FREE
                                      \-> QUARANTINED
private: FREE -> PENDING -> READY | FAILED
```

`descriptor` 与 request ledger 填完后，先 `DataSyncBarrier<DDR>()`，再调用
KVAF descriptor publish；KVAF publish 在 PostOnly 前再执行一次 DDR barrier。
CQ terminal 前绝不发布 READY，也绝不释放 queue。release CAS 失败时，执行器
保留 nonzero KVAF owner，设置 `queue_quarantine=1`，不把局部 wave 当成释放；
KVAF runtime terminal drain/teardown 负责后续恢复。

## 正确性边界

推测路径只多搬数据。它不写 C8 Top-K，不发布持久 resident mapping；CQ 成功仅
写 transaction-private `private_ready==READY`，失败写 `FAILED`。最终 Exact
Top-K + Guaranteed Recall 仍负责所有真正 winner。QSFA 不得仅凭
`source_to_slot >= 0` 读取，而要校验 private ready 以及 epoch/generation。

该初版以 completed-C8 CandidateSeed 为输入，因此可验证真实 HCCP 生命周期，
但不能把它宣传成与尚在运行的 C8 tile 重叠。要实现 tile-level overlap，必须先
建立受 CANN/graph 证明的 publisher -> executor release/acquire 事件边；不能以
GM 轮询代替该合同。

### mapping 身份与失败清理

新 private claim 先写 `PENDING`，同时写入 `private_ready_epoch=current_epoch` 与
`private_ready_generation=current_queue_generation`。已有 `source/object -> slot`
只能在这两个 tag 仍属于当前 epoch、且 private state 非 `FREE` 时共享；旧 mapping
会被 CAS 清回 `-1`，由 Guaranteed Recall 处理，绝不跨 forward 复用。

若 descriptor 已 PostOnly、随后 source/object mapping publish 出现局部失败，P7.1
只将该 queue quarantine，仍调用 `Finish` 等待 CQ terminal。只有 terminal 后才将
private state 标记 `FAILED` 并用 CAS 清理该 wave 的 `-2/output_index` mapping；不能
在 HCCP 仍可能写 private payload 时回收它。
