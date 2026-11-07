# P 端多层 source arena

分支：`v0.30.0rc2-develop2`，基于 develop 的 `1e7d334f8`。
仅修改外置 PD Worker/Backend，不修改原生框架核心、KVAF SDK 或 K 服务。
CPU 回归不代表 NPU 硬件或模型精度验收完成。

## 原因与方案

原实现逐层、逐 D peer 创建源，出现过 D 的 `DIRECT_JETTY_CREATE_FAILED ret=328100`，
以及 K 的 `CPU Pull: urma_create_jetty / urma_create_jfc`。
初始化协调器控制并发，不能减少长期存活的源数量。

新实现为若干层分配一个真实连续 arena，各层 main、indexer、scale 保持独立且不重叠的 Tensor 视图。
每个 arena 针对每个 D peer 创建一个 CPU source 和一个 NPU source，不跨 D 复用单读者源。
D 仍逐层接收布局元数据，但同一 arena 只执行一次实际 SDK attach/detach。
这改变了“每层分别调用 SDK Register”的物理注册粒度，保留逐层业务布局和 DDR 目录。
没有把不相邻的 Tensor 伪装成连续地址，也没有初始化后复制整份 KV。

## 开关与数量

P 的原生 `kv_connector_extra_config.source_arena_layers` 默认值为 **1**：

| 值 | 行为 |
| --- | --- |
| 1 或省略 | 原逐层独立分配、注册和发布 |
| 2..128 的整数 | 每组最多该数量的层，共用真实源 arena |

仅在 P 配置大于 1 的值，D 保持默认 1。P/D 均须更新到 develop2；
P 会检查 D 的 source_arena_v1 能力，旧 D 在创建 arena 源之前被明确拒绝。
Target 与 MTP 不混组，遇到角色切换会分组，尾组可少于配置层数。

示例：78 个连续 Target 层和 1 个 MTP 层、分组 16，实际为 5+1=6 个 arena。
TP8 对每个 D 的真实源数从约 8×79=632 降为 8×6=48，逻辑层元数据仍为 632 份。
实际层数、组数以模型和启动日志为准；每个源内部的队列数仍由 SDK 决定。

## 地址及生命周期

每个 arena 只调用一次原生缓存 allocator，各层起点按 2 MiB 对齐。
层内 128-token block、656/128/4 token 字节数和三种 plane 布局不变。
arena 只用于 P 源缓存；D 的目的缓存、逐层 DDR 目录和 Hot/Get 保持原实现。

元数据增加 registered_range=(Arena_PA_base, Arena_bytes)，仍交换 PA，不传绝对 UBA。

```text
LayerOffset = (plane_PA_base - Layer_PA_base) + physical_block * block_bytes
SourceOffset = (Layer_PA_base - Arena_PA_base) + LayerOffset
```

D 按原逐层 blockTable 计算后，将 CPU/NPU 两条 Pull 的源偏移换算为 SourceOffset。
HBM 目的偏移和 DDR 地址不加此偏移。SDK 根据 arena 注册、票据和偏移解析实际 UBA。
边界检查确保层范围落在真实 arena 内；对齐 Padding 不作为业务 KV 传输。

每层、每批仍使用新的 Ready，按 Pull→Wait/Commit→ACK→release 完成后才处理共享源的下一层，
不能把一份 Ready 重复用于多个 Put。一个 arena 由一个发送 future 顺序处理，不同 arena 可以并行。
Target 在同组全部层写好后记录 event 并发布；MTP 单独分组，直到 native 采样/最终 proposal 完成才发布，
避免重复写入与读取重叠。图回放遗漏 Python 钩子时在成功 forward 完成边界补发。
下一次模型执行仍先等待上一帧所有 source future。MTP skip 不读取 dummy-forward KV。

层引用显式持有源，最后一个引用关闭前不执行 SDK shutdown。
失败或丢 ACK 保留原 Ready 和引用，停止同组后续层，不重新提交不确定的 DMA。
D 在 peer 全部任务结束后按实际绑定 detach 一次，P 再关闭源。
部分 attach 的描述符也保留以便清理，不能把未确认绑定当成已成功。

分组会推迟早期层发送，可能减少计算与通信的重叠。
大 MR 注册能力、连续分配能力、HBM 峰值、启动耗时和吞吐均需硬件验证；16 只是本次验证起点。

## 更新与配置

先停止本轮 Proxy/P/D 并正常清理旧源，避免残留资源混入新计数。
若清理失败，保留日志和资源，确认传输静止后处理，不要反复只启动 P。
在 **P、D 源码容器**分别执行：

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
git fetch origin v0.30.0rc2-develop2:refs/remotes/origin/v0.30.0rc2-develop2
git switch v0.30.0rc2-develop2
git branch --set-upstream-to=origin/v0.30.0rc2-develop2
git pull --ff-only origin v0.30.0rc2-develop2
```

保持现有 editable 安装，不需重编译 vLLM、Ascend、KVOP、KVAF 或 K。
在 **P 的 p_dp1tp8start.sh** 中合并以下 JSON，不重复参数或键：

```json
{
  "kv_connector": "PDKVoffloadConnector",
  "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kvaf.pd.connector",
  "kv_role": "kv_producer",
  "kv_port": 31000,
  "kv_connector_extra_config": {
    "timeout": 600,
    "source_arena_layers": 16,
    "decode_endpoints": [
      "141.62.38.24:31000", "141.62.38.24:31001",
      "141.62.38.24:31002", "141.62.38.24:31003",
      "141.62.38.24:31004", "141.62.38.24:31005",
      "141.62.38.24:31006", "141.62.38.24:31007"
    ]
  }
}
```

上例对应当前 P TP8/DP1、D TP1/DP8。原始 P TP8/DP2、D TP1/DP16 仍按原生端口规则配置全量 endpoint。
IP、权重、TP/DP、K 地址和 DDR 预算保持对应部署的原值。已有诊断字段保留，双方 verify_kv 模式必须相同。
D 继续使用 [初始化协调配置](kvaf_pd_initialization.md)，不要在 D 设置 source_arena_layers=16。

健康 K 28997 与协调器 28998 保持运行。启动 D、等待 DDR 初始化完成，再启动 P，最后启动 Proxy。
首次对照保持现有 core/jetty 等其余参数一致，不同时改变多个变量。

## 验收与回退

P 每个 worker 只打印一次原生日志摘要：

```text
KVAF PD source arenas: device=... layers=... arenas=... D_peers=... registrations_per_path=...
```

registrations_per_path=arenas×D_peers，分别对应 CPU 和 NPU 源数，不是请求数。
启用有界诊断时，REGISTERED 事件还包含 source_arenas。
在 D 检查本轮独立 SDK 日志（替换实际 PID）：

```bash
grep -c 'HBM_READ_READY ' /root/ascend/log/vllm_ascend/kvaf/kvaf_DP0_TP0_device0_pid实际PID.log
grep -n 'DIRECT_JETTY_CREATE_FAILED' /root/ascend/log/vllm_ascend/kvaf/kvaf_DP0_TP0_device0_pid实际PID.log
```

成功源数应随 P worker数×arena数增长；同源 EID 候选回退需结合最终 READY 判断。
K 同期不应再出现创建 Jetty/JFC 失败。这只验收资源与建链，不能代替 KV 或模型精度验证。
随后验证短请求、128 边界、带 tail 的长请求、chunked Prefill、正常/跳过 MTP，
双方 diagnostics_requests=2、verify_kv=true 做字节检查，再与同模型基线比较输出。
诊断有同步/读回开销，完成后恢复 0/false。

回退时先排空 P/D，将 P 的 source_arena_layers 删除或设为 1，重新启动两端；不能热切换现有注册。
CPU UT 覆盖实际 storage 视图、非零 arena 偏移、双 Pull 字节、逐层目录、MTP/graph、
丢 ACK、引用与释放、一次 attach/detach、版本拒绝和原始 TP8/DP2→TP1/DP16 八 K 目录注册矩阵。
这些模拟 SDK 的测试尚未验证真实 UB/NPU 上的大 arena 注册和最终输出精度。

控制端口时序、并发连接和失败 RPC 定位见 [PD 控制连接排障](kvaf_pd_control_troubleshooting.md)。
