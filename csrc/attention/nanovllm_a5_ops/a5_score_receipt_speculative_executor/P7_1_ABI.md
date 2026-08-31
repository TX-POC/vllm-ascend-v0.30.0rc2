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

## 宿主提交边界

P7.1 的宿主侧只做一次**直接 ACLNN 入流**：创建 31 个 tensor descriptor，调用
`GetWorkspaceSize`，随后在调用者当前 NPU stream 直接调用
`aclnnA5ScoreReceiptSpeculativeExecutor`。这个调用只把 AIV-only kernel 排入当前
stream，**不等待** DMA 或 CQ，也不通过 `torch_npu::OpCommand` 再包一层 host
task queue。原因不是绕开异步，而是 P7.1 的真实异步工作已经在设备端：
`PostOnly -> DMA -> CQ`。再添加一个私有 host callback 会增加资源所有权边界，且曾在
kernel fail-closed 后造成 post-submit 进程崩溃。

host 返回后，CANN stream 继续执行 P7.1 kernel；kernel 内部才动态租 queue、发布
descriptor、PostOnly，并在后续 receipt 中非阻塞 drain CQ。因此这里没有回到 CPU/GIL
逐候选提交，也没有把 Get 固定给某些 AIV。

当前 P7.1 tiling 明确声明 `workspace=0`，bridge 也对此做 fail-closed 检查。因而
direct launch 没有一个在 host 返回后还要单独保活的临时 workspace Tensor；31 个输入和
18 个 Ref 状态本来就是 graph `RuntimeResources`，其生命周期至少覆盖本 forward 的
stream terminal。未来若算子需要非零 workspace，必须先补上可验证的 stream-owned
workspace 生命周期，不能悄悄复用这里的零 workspace 路径。

但 `workspace=0` **不等于**不需要区分两种不同寿命的 host 对象。P7.1 按 CANN
标准 adapter 的顺序处理它们：

1. `aclTensor` descriptor 只属于 ACLNN 的 phase-1/phase-2 host 调用。`launch`
   成功返回后，设备已经拿到了所需地址；bridge 必须立刻销毁 31 个 descriptor，**再**
   执行 `ReleaseHugeMem -> UnInitHugeMem -> UnInitPTA`。
2. 31 个 ATen Tensor handle 才是设备流仍可能读取的 graph resource。它们被放进一个
   有上限的 `direct-launch lease`，直到同一 forward 的 stream terminal 才释放。

这个 lease 不执行 DMA、也不轮询 CQ；它只保活真实 Tensor，绝不把短命的
`aclTensor` descriptor 跨过 PTA/HugeMem TLS 边界保存。

每次直接入流成功时，bridge 同时生成一个单调的 `launch token`，并把 token 绑定到
**这一次 ACLNN 实际入流的 `aclrtStream`**。dispatch return 后，Runtime 立即调用
`capture_forward_terminal_lease()`，将这个 token 放进本 forward 的 `RuntimeResources`；
这一步不等待 NPU。等到本 forward 的最终 stream terminal（`forward stream terminal`，实际产品中是 QSFA terminal）已经
确认后，Runtime 在任意 host 线程调用 `release_at_forward_terminal(lease)`。registry 只会释放
“同一 stream 且 token 不晚于该 terminal”的 **Tensor keepalives**：一个 stream 的 terminal
绝不会把并发 stream 的 lease 一起清掉，也不要求 terminal 回调与入流回调在同一 host thread。

这里的 token 是 host-side Tensor-keepalive 生命周期身份，不替代设备端的
`streaming_epoch + queue_generation + post_sequence`。后者仍由 P7.1 kernel/CQ 用于拒绝旧 wave
completion；token 只回答“哪些 Tensor handle 在该 stream terminal 后可以释放”。若 forward
没有走 terminal，lease 池会在固定上限前 fail-closed，而不是无限积压。

这仍然没有重新引入 `OpCommand`、CPU/GIL 逐候选提交或固定 Get 核：设备端的
`PostOnly -> DMA -> CQ` 从 P7.1 kernel 入流后持续异步执行；terminal hook 只处理已经
不再被设备读取的 Tensor keepalive 生命周期。当前 standalone P7.1 测试已走同一
`capture -> stream terminal -> release` API；产品图路径接入时必须把 capture 的返回值放进
transaction `RuntimeResources`，并由现有 QSFA terminal 取出释放，不能直接调用底层 pybind
接口或按进程全局清空 lease。

### 测试进程退出与产品路径要分开

P7.1 的正确性 gate 是“ACLNN 已入同一 stream、该 stream 已 terminal、lease 已释放、真实
fixture 已完成 `Finish/Release`”。这几个条件完成后，测试进程再退出时，CPython、Torch、CANN
和动态库还会依次做各自的全局析构；这不是一次 forward 的执行步骤。

因此 scoped runner 使用三个**独立子进程**：第一个以无效 control 运行到
`P7_1_FINAL_RECEIPT` 后用测试专用 `os._exit(0)` 结束，证明直接 ACLNN 入流、stream terminal 和
lease 生命周期；第二个用真实 cold-bound KVAF control，在 `Finish/Release/cleanup` 全部结束后才
发出同类 receipt，证明真正的 Direct-HCCP transaction；第三个保持正常 Python 退出，仅记录
`P7_1_PROCESS_EXIT`，用于定位 CPython/ELF/CANN teardown。`os._exit` 绝不出现在产品 runtime
路径，也不跳过 fixture 的 `Finish/Release/cleanup`：真实 HCCP receipt 只能在这些都结束后发出。

桥接内部唯一含 mutex/vector 的 lease registry 采用进程生命周期分配。它在运行期仍严格限制
最多 16 个 pending lease，forward terminal 仍必须使其清零；改变的只是让这个 bridge-owned
对象不参与 Python/Torch/CANN 的退出析构排序，避免没有业务价值的 shutdown-order 风险。

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
