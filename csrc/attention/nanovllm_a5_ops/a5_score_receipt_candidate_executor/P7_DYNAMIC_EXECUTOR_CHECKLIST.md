# P7 动态设备执行器：安全与验证清单

## 不可改变的边界

P7 只能消费 C8 已发布的 `score_workspace + tile_receipt`；不得在 C8
vector 类中新增 `TBuf`、SIMT 调用、HCCP 状态或事件。C8 的精确 Top-K
仍是唯一权威结果。P7 只允许多搬数据，最终由 Exact Top-K + Guaranteed
Recall 补齐真正 winner。

`2 x 64 x 4 x int32 = 2 KiB` 是**执行器自己的**双 microbatch 上限，
不是可以侵入 C8 UB 的理由。P7 独立算子有独立 UB layout；未来若做真正
单核融合，必须从官方 C8 的 tiling/UB 预算整体重排，并先通过 bitwise C8
gate，不能把 P6 闭包重新编进去。

## 最小但完整的状态机

两套状态不能混用：

```
Candidate: FREE -> FILLING -> READY -> CLAIMED -> FREE | DEFERRED
Queue:     FREE -> OWNED_FILLING -> POSTED -> CQ_TERMINAL -> FREE | QUARANTINED
Private:   FREE -> RESERVED -> IN_FLIGHT -> READY | FAILED -> FREE
```

- queue 的 CAS lease 是唯一所有权裁决；没有固定 Get 核，也没有 31+1 分区。
- descriptor 的 candidate arrays、count、output indices 全部写完后，执行一次
  `DataSyncBarrier<DDR>()`，再写门铃/Post；禁止逐字段 barrier。
- 只有 CQ terminal 后，才能将 private 标为 `READY/FAILED`，并释放 queue。
- release CAS 失败绝不能清本地 `hasWave` 或把 queue 当作 free：将 queue 标成
  `QUARANTINED`，保留波次诊断，private 标为 `FAILED`；仅 forward terminal
  drain/runtime teardown 可统一回收。
- `source_to_slot >= 0` 不是可读条件；QSFA 只接受同一 epoch/generation 的
  `private_ready == READY`。

## 生命周期字段：必须与可后置项

P7 必须为每个 queue 保存 `{forward_layer_epoch, queue_generation,
post_sequence, owner_tag, state}`。`owner_tag = producer_id + 1` 只用于本次
CAS 抢占，不可单独作为 CQ 身份。

不要求现在引入通用异常恢复/重试框架，也不要求把上述字段打包成 64-bit
原子。现有 int32 CAS 可继续负责互斥；获得 lease 后由 owner 写入 epoch、
generation、post sequence，并由 CQ terminal 校验。这足以避免 graph replay/
跨 forward 的陈旧 completion 被误认；更复杂的恢复策略可后置。

## P7 分两步实现，不能混称为“已重叠”

### P7.0：receipt/candidate 数学 oracle

保留当前 `A5ScoreReceiptCandidateExecutor` 为**完成 C8 后**的独立 score
consumer：验证 receipt、ExactScoreScan 和 CandidateCompact 与 CPU oracle
一致。它没有 Post/CQ，也不能声称 tile-level overlap。此步的价值是把 C8
数学、score-GM 格式和 candidate 判定从 HCCP 故障中分离出来。

它的 NPU gate 是 `tests/test_score_receipt_candidate_executor.py`：测试自己构造
已完成的 packed-score/receipt，不导入或启动 C8/KVAF，并逐元素比对
`candidate_seeds`、`candidate_counts`、`candidate_ready_epoch` 和
`receipt_status`。通过标志是 `P7_ORACLE_PASS`。

### P7.1：真正动态 Post/CQ executor

新建独立的 `A5ScoreReceiptSpeculativeExecutor`，而不是向 P7.0 或 P6 追加
闭包：

1. 保持 C8 为原始 publisher：只发布 score receipt，不连接 P6 closure。
2. 所有 executor AIV worker 动态 claim receipt，读取 receipt，ExactScoreScan
   + CandidateCompact，并在自身 `2 x 64` UB 中形成 seed。
3. 同一 worker 动态租约 queue，填 descriptor、release barrier、`PostOnly`；
   后续 receipt/tile 间非阻塞 `TryRetire`，CQ terminal 才发布 private ready。
4. terminal op 调用 `Finish`，处理 quarantine，并将 deferred/failed 项交给
   Guaranteed Recall。不要把候选 seed 持久塞回 C8。

P7.1 只有在 CANN 支持的 graph/device 生产者-消费者可见性边已经建立后，
才能与仍在运行的 C8 tile 并发。单纯轮询 GM epoch 不足以自行制造这个合同；
在此前只能作为串行正确性测试。

## 每次构建前必须验证

- C8 publisher OPP 的 source manifest 与 runner 分开验证；runner 新提交不要求
  重建无关 C8 OPP，但 publisher OPP 变更必须重建。
- 先跑原始 C8 vs publisher C8 bitwise Top-K/score-source-lens gate；失败即停止
  P7，不跑 Post/CQ。
- P7 单独验证 receipt epoch、候选 CPU oracle、barrier 之后的 descriptor
  可见性、queue lease 唯一性、CQ ready/failed、release-failure quarantine。
- 同负载报告：candidate compacted/deferred、queue busy/lease 成功率、Post->CQ
  延迟、CQ pending 时完成的后续 receipt 数、Guaranteed Recall 补量。没有真实
  payload correctness 时，不报告 H2D 带宽或端到端收益。
