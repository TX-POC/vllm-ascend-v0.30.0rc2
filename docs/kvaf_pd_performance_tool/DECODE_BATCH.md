# 同步 Decode 组批压测开关

此功能按用户指定的实验策略，等待整组真实请求的远端 KV 就绪，再把各 D rank 的接收完成集合一次性交给原调度器。
它主动改变 Decode 放行时机，与被动性能采集独立；默认关闭。开启后的结果是同步组批条件下的 TPOT，不能冒充普通在线调度性能。
KV 传输、数据内容、原 SDK 完成判定、模型/图计算和 MTP 算法沿用原实现，不在设备上新增同步算子。

## 1. 可配置 batch 与放行条件

- 压测端 `--levels 16`：DP8，每卡目标 batch 16，总计 128 个请求；`--levels 8,16` 分别执行 64、128 请求两轮。
- 每轮有唯一 group ID、每个 rank/slot 有唯一请求；预热轮使用自己的 group 和实际预热 batch，不与正式请求混合。
- D 只有在原 Receiver 确认 KV 已接收并发布后，才将该请求计为就绪。每个 rank 凑齐本轮全部就绪请求后，由后台控制线程向协调器报告。
- 协调器收到全部 DP rank 的完整集合才放行；D 主循环不阻塞等待网络、不持有 Runtime 业务锁等待屏障。
- “统一”表示所有 KV 就绪之后才允许各 rank 调度，不是设备在同一纳秒开始执行。原调度预算、内存及抢占仍决定实际 batch，必须结合 D_SCHEDULED 和实际请求结果核对。
- 放行前请求失败、取消、错误路由、重复 slot、配置冲突或超时不降级放行部分 batch，明确报错；排查并结束失败轮次后，重新启动受影响 D 实例再测试，不复用旧 group。放行后的取消仍由原推理生命周期处理。

支持当前 D TP1/DP8 拓扑；batch 可调整，但不得超过配置的 `max_batch_size`。本地检查也覆盖小 batch；NPU 图模式/EP/MTP 实机批量与精度仍待验证。

## 2. 配置开关

在 **D 和运行 Proxy 的节点**，编辑仓库的 `scripts/kvaf_pd_decode_batch.json`：

```json
{
  "enabled": true,
  "proxy_url": "http://141.62.38.12:8000",
  "dp_size": 8,
  "max_batch_size": 16,
  "timeout_seconds": 3600
}
```

`max_batch_size` 是容量上限，实际本轮 batch 来自 `--levels` 或 `--warmup-batch-size`；测试更大档位时同步提高 D 的容量及该上限。
配置按进程启动读取，没有热切换。`enabled=false` 为关闭；与 `kvaf_pd_performance.json` 的采集开关互不替代。
采集关闭仍可做同步组批压测；组批关闭仍可正常采集原调度行为。

## 3. 复用现有 Proxy 汇总与放行

请求就绪状态由现有 Proxy 进程汇总，D 通过其原 HTTP 端口 8000 上报。无需额外进程、监听端口或 K 侧脚本；原初始化协调器 28998 保持原用途。

`proxy_url` 指向 D 可访问的现有 Proxy 基础地址。旧版 `coordinator` 字段需替换为 `proxy_url`，不能继续使用独立 29000 地址，也不同时保留两个字段。
原独立服务入口已删除；此前启动的旧 29000 进程可在结束旧轮次后自行停用，新版不会连接它。

`GET /v1/kvaf-decode-batch` 返回开关、DP 和容量，发送工具用它预检查；`POST /v1/kvaf-decode-batch/ready` 承载就绪/abort。
Proxy 在同一事件循环异步等待全组状态，不占用阻塞线程等待、不持有原 metaserver 锁等待；普通转发、metaserver 和健康检查可以继续处理。
仅传请求身份及状态，不中转 KV。默认关闭时不创建汇总 Registry，也不建立 D 控制连接。

当前 Proxy 原入口为单进程，同一轮所有 D 必须上报同一个 Proxy 进程。它有 1024 个 group 的固定预算，成功/失败身份保持终态，不自动复用；实验间可重启清理，不能在活动轮次中重启。
客户端在放行前断开、请求取消或 Proxy 退出时，未放行组失败并唤醒等待者，不拼凑部分 batch。已完成的放行结果不能被迟到重试或 abort 撤销。

## 4. D 与 Proxy 接入

D 沿用性能工具入口：

```bash
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

`--kv-transfer-config` 保留原 Connector 名，module path 选择：

```json
"kv_connector_module_path": "kvaf_pd_transfer_stats_connector"
```

D 启动脚本至少配置 `--max-num-seqs 16`；`max-num-batched-tokens` 至少能容纳上限 batch 的一轮 MTP Decode，当前每请求 3 个 speculative token，工具要求至少 `max_batch_size × 4`。
原 4096 token 预算可覆盖 16 的该项要求。启动检查只验证调度配置，不保证 KV/HBM/DDR 容量足够。
同时保留 16 个长上下文会显著增加 KV 容量需求，需要按实际模型和池容量配置；容量不足时不能依靠屏障获得 batch 16。

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
D 未开启该模式却收到带组批标签的请求时也明确拒绝，避免只启用 Proxy 就误以为 D 已同步。
P 保留原启动和处理逻辑，无需配置屏障。D/Proxy 源码和设置需更新到同一版本。

## 5. 发送与 TPOT 统计

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
/usr/local/python3.12.13/bin/python3 scripts/benchmark.py \
  --pd-proxy --sync-decode-batch \
  --base-url http://141.62.38.12:8000 \
  --metadata-base-url http://141.62.38.24:8200 \
  --model glm --ignore-env-proxy \
  --input-tokens 130947 --output-tokens 125 \
  --data-parallel-size 8 --tensor-parallel-size 1 \
  --warmup-rounds 2 --warmup-batch-size 1 \
  --levels 16 --request-timeout 7200 \
  --result-dir /home/c00612283/pd-benchmark/sync-run01
```

发送工具先校验 Proxy 已集成汇总、开关、DP 与容量，失败时不发送本轮请求。请求超时必须大于屏障超时，并给后续 Decode 留出时间；长上下文大批量 P 阶段可能很久，按实际实验同时调整两者。
输入加输出仍不得超过服务 `max-model-len`。Warmup 使用两个独立的 8 请求轮次，然后正式执行一个 128 请求轮次。
只有同时启用 D/Proxy 配置和 `--sync-decode-batch` 才是同步模式；不带发送端开关的请求会被开启模式的 Proxy 拒绝。

TPOT/TPS 仍基于真实客户端 token IDs 和到达时间，不拿屏障放行时间当 token、不伪造有效公共窗口。
整组 overlap 窗口从所有请求最晚首 token 到最早末 token；同步放行有助于形成窗口，但不保证一定存在。无有效窗口仍报告不可用，可延长输出并相应减少输入。
逐请求 TTFT/E2E/TPOT 及其 mean/P90/P99 保留；TTFT/E2E 包含主动组批等待，不能与普通在线模式不加区分地比较。

开启性能采集时，额外记录 `D.decode_batch_wait`（该请求 KV 已就绪至本 rank 获得全组放行），以及实际 D 调度 batch；Trace 来源标注同步模式。
日志 `KVAF_DECODE_BATCH_RELEASE` 表示该 rank 获得全组就绪回执，不是设备计算起点。完整组批耗时和 Decode 后客户端 TPOT 分开解释。

## 6. 关闭与验证范围

将 D/Proxy 的 `enabled` 改回 `false` 后重新启动，并去掉发送端 `--sync-decode-batch`，即恢复原完成上报和路由路径。
关闭路径不创建 Gate、屏障线程或控制连接；原 ReceiveRequest 格式不变。原 KV 释放及异常处理仍由原实现负责。

已用本地 HTTP/ASGI 验证原 Proxy 及采集入口共用原端口，DP8 的 batch 1/8/16 在最后一个就绪请求到达前不放行、完整集合和日志只放行/打印一次。
等待期间健康检查及 metaserver 可响应；另检查 64 个异步 rank、真实 HTTP 断开、超时、取消、abort、Proxy 生命周期关闭、旧配置拒绝和终态重试。
原生 Connector/Scheduler 边界用替身 SDK 检查 8×16 的完成集合和释放生命周期；真实 ASGI 检查 Proxy 标签、DP header、P 元数据及缺失标签拒绝。
以上不代表 NPU EP/图模式/MTP 精度与实际 batch 验收。实机先用每卡 1，再提高至 16，核对成功率、实际 batch、TPOT 和输出。
