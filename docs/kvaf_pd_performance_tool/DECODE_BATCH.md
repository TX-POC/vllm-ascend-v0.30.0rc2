# 同步 Decode 组批压测开关

此功能按用户指定的实验策略，等待整组真实请求的远端 KV 就绪，再把各 D rank 的接收完成集合一次性交给原调度器。
它主动改变 Decode 放行时机，与被动性能采集独立；默认关闭。开启后的结果是同步组批条件下的 TPOT，不能冒充普通在线调度性能。
KV 传输、数据内容、原 SDK 完成判定、模型/图计算和 MTP 算法沿用原实现，不在设备上新增同步算子。

## 1. 可配置 batch 与放行条件

- 压测端 `--levels 16`：DP8，每卡目标 batch 16，总计 128 个请求；`--levels 8,16` 分别执行 64、128 请求两轮。
- 每轮有唯一 group ID、每个 rank/slot 有唯一请求；预热轮使用自己的 group 和实际预热 batch，不与正式请求混合。
- D 只有在原 Receiver 确认 KV 已接收并发布后，才将该请求计为就绪。每个 rank 凑齐本轮全部就绪请求后，由后台控制线程向协调器报告。
- 协调器收到全部 DP rank 的完整集合才放行；D 主循环不阻塞等待网络、不持有 Runtime 业务锁等待屏障。
- “统一”表示所有 KV 就绪之后才允许各 rank 调度，不是设备在同一纳秒开始执行。原调度预算、内存及抢占仍决定实际 batch，工具用 D 调度器记录的首步请求数核验，并在 result.json 标记未达到目标的轮次。
- 放行前请求失败、取消、错误路由、重复 slot、配置冲突或超时不降级放行部分 batch，明确报错；排查并结束失败轮次后，重新启动受影响 D 实例再测试，不复用旧 group。放行后的取消仍由原推理生命周期处理。

支持单个 D API 入口覆盖全部 D Rank，拓扑为 TP1/PP1。

## 2. 配置开关

仅编辑 **Proxy 节点**的 `scripts/kvaf_pd_decode_batch.json`，启动 D 后再启动或重启 Proxy：

```json
{
  "enabled": true
}
```

Proxy 启动时读取 D 的实际 DP 数和调度容量。每 Rank 的容量上限取 `max_num_seqs`、`max_num_batched_tokens // (1 + num_speculative_tokens)` 与工具上限 512 三者的最小值。实际测试档位取 benchmark 的 `--levels`，同步模式省略时使用 1。通常也无需填写 DP、TP 或屏障超时。屏障默认 3600 秒，兼容旧配置中的 `timeout_seconds`、`dp_size`、`max_batch_size` 和 `proxy_url`，其中 DP 必须与 D 一致，batch 上限可进一步收紧。

## 3. 复用现有 Proxy 汇总与放行

请求就绪状态由现有 Proxy 进程汇总，D 通过其原 HTTP 端口 8000 上报。无需额外进程、监听端口或 K 侧脚本；原初始化协调器 28998 保持原用途。

D 从请求已有的 metaserver 地址获取回调地址。Proxy 的 `GET /v1/kvaf-decode-batch` 提供开关、D 拓扑、容量和 metadata 地址，默认由 benchmark 自动读取。
`POST /v1/kvaf-decode-batch/ready` 汇总每个 Rank 的完整 KV 就绪集合，`POST /v1/kvaf-decode-batch/scheduled` 接收首步调度计数。benchmark 结束时查询本轮结果。
Proxy 在原事件循环等待所有 Rank，仅交换请求身份和计数。

当前 Proxy 原入口为单进程，同一轮所有 D 必须上报同一个 Proxy 进程。它有 1024 个 group 的固定预算，成功/失败身份保持终态，不自动复用；实验间可重启清理，不能在活动轮次中重启。
客户端在放行前断开、请求取消或 Proxy 退出时，未放行组失败并唤醒等待者，不拼凑部分 batch。已完成的放行结果不能被迟到重试或 abort 撤销。

## 4. D 与 Proxy 接入

仓库的 `scripts/d_start.sh` 和 `scripts/start_vllm_pd.sh d ...` 已带 D 容量查询入口。自定义 D 启动命令首次升级时加入：

```bash
--middleware vllm_ascend.distributed.kv_transfer.kvaf.pd.decode_batch.DecoderCapacityMiddleware
```

同步放行由原生 `PDKVoffloadConnector` 处理。原有性能采集版本的 Connector 也继承此能力。D 不读取同步开关文件。NPU 内存和 KV 块仍按原机制分配，长上下文能承载的实际 batch 以运行结果为准。

Proxy 可沿用原启动入口和参数：

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3 bash scripts/kvaf_pd_proxy.sh \
  --host 141.62.38.12 --port 8000 \
  --prefiller-hosts 141.62.38.12 --prefiller-ports 8100 \
  --decoder-hosts 141.62.38.24 --decoder-ports 8200
```

若还需要采集 Proxy 自身的阶段，可将上述入口替换为 `python3 scripts/kvaf_pd_performance_proxy.py`，两者复用同一应用、路由和汇总实现，不重复启动第二个 Proxy。
开启模式要求每个 HTTP 生成请求只带一个 prompt 和显式 cohort 标签，使用专用压测流量。
Proxy 将请求的目标 rank 通过原生 `X-data-parallel-rank` 转给 D；D Scheduler 再核对实际 rank，路由不支持或错误时明确失败，不把负载均衡结果假定为每卡满 batch。
P 保留原启动和处理逻辑，无需配置屏障。D、Proxy 和 benchmark 代码需更新到同一版本。

## 5. 发送与 TPOT 统计

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
/usr/local/python3.12.13/bin/python3 scripts/benchmark.py \
  --pd-proxy \
  --base-url http://141.62.38.12:8000 \
  --model glm --ignore-env-proxy \
  --input-tokens 130947 --output-tokens 125 \
  --warmup-rounds 2 --warmup-batch-size 1 \
  --levels 16 --request-timeout 7200 \
  --result-dir /home/c00612283/pd-benchmark/sync-run01
```

发送工具先校验 Proxy 已集成汇总、开关、DP 与容量，失败时不发送本轮请求。请求超时必须大于屏障超时，并给后续 Decode 留出时间；长上下文大批量 P 阶段可能很久，按实际实验同时调整两者。
输入加输出仍不得超过服务 `max-model-len`。Warmup 使用两个独立的 8 请求轮次，然后正式执行一个 128 请求轮次。
`result.json` 的 `levels[].decode_batch_result` 记录目标 batch、每 Rank 完整报告的 KV 就绪数和首步调度请求数。`matched` 表示所有 Rank 的首步均恰好调度目标数的本轮请求。`batch_mismatch` 或 `unavailable` 会标记该轮失败，并保留客户端结果。首步记录来自 Host 调度器。

TPOT/TPS 仍基于真实客户端 token IDs 和到达时间，不拿屏障放行时间当 token、不伪造有效公共窗口。
整组 overlap 窗口从所有请求最晚首 token 到最早末 token；同步放行有助于形成窗口，但不保证一定存在。无有效窗口仍报告不可用，可延长输出并相应减少输入。
逐请求 TTFT/E2E 保留主动组批等待。终端输出 Decode TPOT 的 Mean/P50/P90/P95/P99、每轮公共窗口 TPS 和各档位的平均 TPS。
`--measure-rounds` 设置每档正式轮数。目标与实际 Decode batch 按 DP Rank 计，TP1 时对应单卡。

添加 `--hot-kv-stats` 可采集本轮 Hot KV 命中率分位数。默认关闭。D 复用已有的异步 CPU 反馈，仅在正式轮结束后按 Rank 上报直方图。
统计缺失显示 `N/A`，不影响原推理结果。[指标口径](../../scripts/README_benchmark.md#decode-指标与-hot-kv-命中率)

开启性能采集时，额外记录 `D.decode_batch_wait`（该请求 KV 已就绪至本 rank 获得全组放行），以及实际 D 调度 batch；Trace 来源标注同步模式。
日志 `KVAF_DECODE_BATCH_RELEASE` 表示该 rank 获得全组就绪回执，不是设备计算起点。完整组批耗时和 Decode 后客户端 TPOT 分开解释。

## 6. 关闭与验证范围

在 Proxy 上将 `enabled` 改回 `false` 并重启 Proxy。当 benchmark 用普通 PD 模式时，仍通过 `--metadata-base-url` 提供模型和分词接口地址。
关闭路径不创建 Gate、屏障线程或控制连接；原 ReceiveRequest 格式不变。原 KV 释放及异常处理仍由原实现负责。

已用本地 HTTP/ASGI 验证原 Proxy 及采集入口共用原端口，DP8 的 batch 1/8/16 在最后一个就绪请求到达前不放行、完整集合和日志只放行/打印一次。
等待期间健康检查及 metaserver 可响应；另检查 64 个异步 rank、真实 HTTP 断开、超时、取消、abort、Proxy 生命周期关闭、旧配置拒绝和终态重试。
原生 Connector/Scheduler 边界用替身 SDK 检查 8×16 的完成集合和释放生命周期；真实 ASGI 检查 Proxy 标签、DP header、P 元数据及缺失标签拒绝。
以上不代表 NPU EP/图模式/MTP 精度与实际 batch 验收。实机先用每卡 1，再提高至 16，核对成功率、实际 batch、TPOT 和输出。
