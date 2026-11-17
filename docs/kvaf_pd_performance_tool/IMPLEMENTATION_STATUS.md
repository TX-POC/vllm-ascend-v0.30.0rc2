# KVAF PD 性能工具：本地实现与使用记录

更新日期：2026-11-17。依据：[v1.9 设计](DESIGN.md)，沿用请求用途筛选，增加请求 ID 联动双行时序、按层传输核验、负载与 TPS，并精简重复视图。新增同部署 P 多 DP 分析、固定结果覆盖和本轮客户端集合隔离。本文记录实现、使用方法及验证边界。

## 1. 当前交付边界（第四批）

在有界采集、终端分析和请求观察基础上，已具备可选 Proxy 观察和较大 Trace 的资源检查；本轮将展示统一为概览、请求 ID 联动时序及按层传输核验。
第四批增加真实请求用途标签：明确区分请求预热、正式测试、profiling 与未知用途，分析默认只统计正式请求。
仅修改仓库自有的 KVAF-PD 外部组件及示例 Proxy，未修改上游 vLLM、KVAF SDK、K 服务或原部署脚本。
原始 KV 读写、事务、完成判定、异常、恢复和资源释放仍由原业务代码执行。
采集不增加设备同步、SDK 调用、READ、控制 RPC、KV 内容检查或路由操作。

**只统计真实请求阶段。** 没有实例启动、模型加载、通用内存池、MR 注册/导入、建链/探测、图捕获或预热的计时与报表。
hello/register/detach 直接透传；没有真实请求绑定的 dummy forward 不生成计算事件。
身份、层类型和时钟标签只是关联元数据。

同步 Decode 组批是另一个默认关闭的压测功能，会主动推迟完成放行；其配置、容量约束与验证边界见 [同步 Decode 组批说明](DECODE_BATCH.md)。它不改变被动采集开关的含义。
请求汇总已并入原 Proxy 的 HTTP 端口，原 `kvaf_pd_proxy.sh` 和可选采集入口共用一份实现；不再启动独立请求协调器。D 配置使用 `proxy_url` 指向现有 Proxy，K 侧初始化服务保持原样。

| 已接入观察点               | 实际计时依据与边界                                                                                                                |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| D 请求 block 绑定与通告    | 原 `update_state_after_alloc` 回调、绑定完成点及原 metaserver 调用；回调在物理分配之后，不能称作完整内存分配耗时                  |
| P 请求、发送排队和生产等待 | 原请求绑定、executor 提交/执行、原 production event 的一次同步调用；不新增 event 或同步                                           |
| `TRANSFER_ELIGIBLE`        | 原生产等待已完成、请求/源/路由信息具备，进入源组 operation 之前；是 Host 已观察到的编排起点，不声称是设备最早就绪时刻             |
| P 控制交互                 | 原 plan/pull/complete/status/cancel 调用，保留同一 P Host 时钟的原完成回执                                                        |
| D 请求内部阶段             | 原 seed 等待、Runtime 锁等待、compute dependency、CPU Prepare/Submit/Wait/Commit/Publish、NPU READ/恢复调用                       |
| 完成里程碑                 | 原 SDK 返回及原状态检查成功后才记录 HBM/CPU Pull done、DDR committed/published；CPU READ 完成不等于 DDR 已发布可读                |
| D 接收门控与 Decode        | 原请求接收发布成功后记录 gate；首轮包含该请求的 target Host forward 入口记录 `D_DECODE_STARTED`，不冒充设备开始时间               |
| Target/MTP 与实际 batch    | 层类型取自原 `offload_main`；计算/生产等待 span 只存一份，以 membership 关联采样请求；实际逻辑 batch 包含未采样成员，不除以 batch |
| 客户端证据                 | 导入已有发送工具的发送、首输出、完成和实际输出数；按精确 ID/代次关联，不按时间接近或 prompt 猜测                                  |
| Proxy 阶段                 | 原 handler 入口、D 选择/派发、metaserver 汇合、P task 排队/执行、原 P/D HTTP 调用及响应流关闭                                     |

`D.metaserver_transport` 包含原 HTTP 处理和等待；远端可能完成 P 计算后才返回，不能当作纯信息编码或网络耗时。
原接口不暴露 HTTP 状态时，只能证明原方法返回，不能据此宣称 HTTP 200。
P/D Host forward 返回跨度也不等于设备 kernel 执行耗时。

原 Sender 控制包装不能观察 Proxy 内部边界，因此仅在示例 Proxy 增加默认 None 的对象级入口。
可选启动器复用原 FastAPI app、lifespan、参数、路由及重试，不复制处理流程或修改模块全局方法。
普通 Proxy 不导入采集模块或创建采集资源。Proxy 同样不采集任何实例阶段。

当前主表在新版 `layer/detail` 采集下采用 `transfer_eligible_to_ack`，同时保存 `pull_rpc_compat` 供对照。
前者按路径从同一 eligible 点到该路径最后一个成功 Pull 回执，包含后续 owner/plan、控制及原串行编排等待；不是纯 DMA。
该路径原本无任务时保持无样本，不补零。完整请求缺少 eligible 或完成证据时不以 RPC 数值替代主指标。
旧采集或 `stage` 档明确标示兼容口径，缺少逐层证据时四组有效 N 为零。

核心四组为 t1/t2 的请求层和请求累计 mean/P90/P99/有效 N。
同请求同层先合并并行贡献者的区间包络，再累加 chunk；请求累计为所有层之和，不是请求墙钟时间。
各阶段分布使用每请求同名成功区间的并集；不同阶段可能重叠，不能相加得到 E2E 或把各阶段 P99 拼成请求。

## 2. 文件与可插拔接入

| 文件                                                       | 作用                                                                        |
| ---------------------------------------------------------- | --------------------------------------------------------------------------- |
| `scripts/kvaf_pd_performance.py`                           | 标准库配置、请求采样、有界队列/写盘、P 控制观察、D 全窗口 batch 直方图      |
| `scripts/kvaf_pd_performance.json`                         | 唯一采集配置文件，随仓默认 `enabled=false`                                  |
| `scripts/kvaf_pd_request_observer.py`                      | 请求身份、真实里程碑、共享 batch、Target/MTP 及阶段观察，状态只保存有界标量 |
| `scripts/kvaf_pd_client_records.py`                        | 导入已有 JSON/JSONL/benchmark 结果，精确关联客户端证据                      |
| `scripts/kvaf_pd_request_phase.py`                         | 在长度识别和聚合前按显式用途及精确请求/代次筛选，拒绝冲突标签               |
| `scripts/kvaf_pd_proxy_observer.py`                        | 有界 Proxy 身份、共享请求批与 Host 阶段，不保存请求正文或重新解析 token     |
| `scripts/kvaf_pd_performance_proxy.py`                     | 可选 Proxy 入口，复用原应用和原命令行参数                                   |
| `scripts/kvaf_pd_transfer_stats_connector.py`              | 可选 Connector，给选中的对象安装 observer；关闭配置不创建采集资源           |
| `scripts/kvaf_pd_transfer_stats.py`                        | 保留原 v1 分析/compare，增加 summary/export-trace 入口                      |
| `scripts/kvaf_pd_performance_report.py`                    | schema 1/2/3 快照、四组统计、请求阶段、关联与单文件 Trace                   |
| `scripts/kvaf_pd_stage_breakdown.py`                       | 离线逐请求阶段矩阵、业务阶段主值及各列 mean/P90/P99/有效 N                  |
| `scripts/kvaf_pd_trace_links.py`                           | 精确控制身份、已知程序顺序和元数据交接；歧义不连线，不估计跨机网络耗时      |
| `scripts/kvaf_pd_layer_observer.py`                        | 可选、默认关闭的 eager P 逐层 Host 接口观察，弱引用、无设备同步             |
| `scripts/kvaf_pd_trace_viewer.html`                        | 单 Trace 概览、请求 ID 双行时序、按层统计和 SVG/PNG 导出                    |
| `vllm_ascend/.../kvaf/pd/observation.py`                   | 被动观察辅助函数，原调用执行一次，原返回值/异常保持                         |
| 同目录 Connector/Scheduler/Sender/Receiver/Runner/Transfer | 最小请求观察入口，observer 默认 None；不修改通用框架或 SDK                  |
| `scripts/benchmark.py`                                     | 额外保存已有 SSE 的服务端响应 ID；不改变请求负载、流式 token 统计或路由     |

P/D 及需要观察的 Proxy 使用同一版本。只复制 scripts 不能获得新增请求观察点，需要同步本批外部组件和示例 Proxy。
沿用 [既有接入说明](../kvaf_pd_transfer_stats.md) 的 PYTHONPATH 和可选 Connector module_path：

```bash
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

在原 `--kv-transfer-config` 中仅将 module path 选择为 `kvaf_pd_transfer_stats_connector`，Connector 名仍是 `PDKVoffloadConnector`。
Worker、图模式、MTP、`source_tp0_only`、并行与原启动参数不变；`diagnostics_requests=0`、`verify_kv=false`。
恢复原 module path 前须先关闭 D 同步组批策略；否则带组批标签的请求不受支持。恢复原 module path 即退出工具接入；插件的 `enabled=false` 或 `sample_rate=0` 也关闭队列、线程与文件。
配置缺失时关闭采集；配置无效时报告停用原因，保留原推理路径。调整配置需要新的采集轮次，不提供运行中热更新。

## 3. 配置与预算

配置路径为 `kvaf_pd_performance.py` 顶部 `CONFIG_PATH`，默认与该模块同目录。
不读取业务 additional_config，不新增环境变量或原生 CLI。完整字段、默认值与约束见 [设计 3.3 节](DESIGN.md#33-可配置采集契约)。

128K 长输入的短窗口采集起始配置如下。保存到 **P 的 `scripts/kvaf_pd_performance.json`**；D 使用同一配置，但将 `layer_compute` 改成 `false`。这是实验建议值，不更改随仓默认配置，也不保证任意负载无丢样。

```json
{
  "enabled": true,
  "sample_rate": 1.0,
  "sample_seed": 0,
  "detail_level": "layer",
  "layer_compute": true,
  "buffer": {
    "max_events": 4096,
    "max_bytes": 67108864,
    "overflow_policy": "drop_new"
  },
  "writer": {
    "enabled": true,
    "output_dir": "/home/c00612283/pd-perf/run02",
    "batch_events": 256,
    "flush_interval_ms": 1000,
    "file_buffer_bytes": 1048576,
    "max_file_bytes": 67108864,
    "max_total_bytes": 524288000
  }
}
```

| 参数                             | 本例含义                                                                 |
| -------------------------------- | ------------------------------------------------------------------------ |
| `enabled`                        | 采集总开关，与同步 Decode 开关独立                                       |
| `sample_rate`                    | 1.0 为全采样，0.1 为约 10%；P/D 保持一致                                 |
| `sample_seed`                    | 确定性采样种子，不是预热次数；两侧保持一致                               |
| `detail_level`                   | `layer` 包含逐层传输和请求阶段                                           |
| `layer_compute`                  | P eager 下观察逐层 Host forward；D 保持 false；不是设备计时              |
| `buffer.max_events/max_bytes`    | 每进程事件数上限 4096、缓冲预算 64 MiB，实际同时受身份预算约束           |
| `overflow_policy`                | `drop_new` 丢弃新样本，不在这里填写路径                                  |
| `writer.output_dir`              | 使用绝对路径；可固定复用，两侧分别写本机文件；分析必须关联本轮客户端结果 |
| `batch_events/flush_interval_ms` | 后台批量写盘 256 条、刷新间隔 1000 ms                                    |
| `file_buffer_bytes`              | 文件缓冲 1 MiB                                                           |
| `max_file_bytes/max_total_bytes` | 单文件 64 MiB、每进程日志总上限 500 MiB；8 个写盘进程约 3.91 GiB         |

修改配置后重启对应服务；没有热更新。服务不重启时，后续 benchmark 可复用同一采集目录，分析按本轮客户端结果隔离请求，不删除或截断服务正在写入的文件。采集预算仍按进程累计，不会因客户端新一轮开始而重置；日志预算耗尽后需结束实验并重新启动采集，不能假定新请求已被记录。先小批采集，检查有效 N、丢样和预算耗尽，再扩大负载。采集关闭不等于同步组批关闭。

P/D/Proxy 使用一致的采样比例、seed 和档位。采样键只使用原外部请求 ID，不能用 TP 本地 alias 或独立随机 generation。
长期观测调低 sample_rate；默认 10% 是配置建议，未经实机 A/B，不保证足够尾延迟样本或可接受开销。

| 档位     | 本批采集内容                                                                                                                                      |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stage`  | 请求绑定、D 门控/首 Host Decode、共享计算 batch、metaserver/绑定回调及 P 控制外层；不生成逐层 eligible、生产等待和 D SDK 细节，不输出四组核心样本 |
| `layer`  | 增加层/chunk/贡献者、发送排队、生产等待、eligible、D plan/seed、Pull 与原完成里程碑；足够归集四组指标                                             |
| `detail` | 再增加 Runtime 锁、compute dependency、SDK Prepare/Submit/Wait/Commit/Publish、READ/恢复及非终态 status 调用                                      |

Host 阶段均有成本；高档不承诺能观察 SDK/设备内部 kernel。
固定槽按每条 16 KiB 保守计费，预算一半供待处理事件、一半供请求/job/在途阶段身份；同时受数量限制。
P 在途身份预算再由原 RPC job 和请求阶段状态分配，因此较小预算可能无法覆盖高并发；超限丢样或停采，绝不扩大预算或反压推理。
`buffer.max_bytes` 至少 32768；单文件至少 131072 字节；总容量不小于单文件。
预算不是整个 Python RSS 上限，后台在途记录、I/O 缓冲和静态元数据另有有界开销。

按 79 层、多 chunk、多 TP 和路径采集会产生大量事件。建议先短窗口/低采样率运行，再看有效 N、丢样和文件量调整预算。
采集预算是每进程预算，整机还要乘实际进程数。全采样 `detail` 不适合作为未经测量的长期默认值。
`writer.enabled=false` 仅保留有界内存和退出计数，不隐式落盘，独立分析脚本无文件可读。

### 3.1 同步 Decode 的容量与启动参数

本例 P/Proxy=`141.62.38.12`、D=`141.62.38.24`，D 为 DP8/TP1/EP8。P 拓扑以实际启动参数为准，不由 benchmark 的 D 元数据推断。
P/D 保留 `PDKVWorker`、原生传输角色和其余已验证参数；按第 2 节设置 `PYTHONPATH` 与 `kvaf_pd_transfer_stats_connector`。
P 的 `kvaf_config.enabled=false`、`kv_role=kv_producer`；D 的 `kvaf_config.enabled=true`、`kv_role=kv_consumer`。
P 为获得逐层计算证据保留已验证 eager 配置，D 可沿用已验证的 `FULL_DECODE_ONLY`；不要为查看器改变业务图模式。

在 **D 和 Proxy 节点**分别编辑现有 `scripts/kvaf_pd_decode_batch.json`：

```json
{
  "enabled": true,
  "proxy_url": "http://141.62.38.12:8000",
  "dp_size": 8,
  "max_batch_size": 4,
  "timeout_seconds": 3600
}
```

D 启动脚本对应参数：

```bash
--max-model-len 131072 \
--max-num-seqs 4 \
--max-num-batched-tokens 4096 \
--gpu-memory-utilization 0.90 \
```

| 参数                            | 作用                                                                         |
| ------------------------------- | ---------------------------------------------------------------------------- |
| `max_batch_size=4`              | 同步组批每个 D rank 的目标 batch 上限，不是全机总数                          |
| `--max-num-seqs 4`              | 每个 D DP 实例的调度请求数上限；不会自动增加 KV 容量                         |
| `--max-num-batched-tokens 4096` | 每个 DP 实例单轮所有请求合计的 token 预算，不是单请求长度；影响试运行/工作区 |
| `--gpu-memory-utilization 0.90` | 引擎 HBM 使用预算比例，不是 KV 占比，也不是内存释放开关                      |
| `--levels 4`                    | 本轮每个 D rank 目标 4 请求，DP8 总计 32；TP1 时等价于每卡 4                 |

D 的 token 预算至少覆盖 `max_batch_size × (num_speculative_tokens + 1)`；本例 MTP 3、batch 4 的该项需求为 16，但不能据此断言所有必要运行路径只需 16 token。
必须检查启动日志的 `logical_blocks`、`Maximum concurrency` 及运行时实际 batch；准入容量不足时，全组等待可能形成循环等待。
K DDR 承载 Target 主 KV，HBM 中仍有 Hot/尾块、Indexer/MTP 和运行工作区；不能用 DDR 大小代替本地准入容量。
P 不需要开启组批开关，也不必跟随 D 把 `max-num-seqs` 改成 4。P 的每个 DP rank 实际分配由 P 路由/调度决定。
完整协议与限制见 [同步 Decode 组批说明](DECODE_BATCH.md)。

### 3.2 启动 Proxy、发送请求与核验

K 服务及原初始化协调器就绪后启动 D、P，再启动一个 Proxy。D/Proxy 配置均按进程启动读取。
在 P/Proxy 容器执行，日志写文件，避免后续无法取得终端历史：

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
mkdir -p pd-logs/proxy

KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3 \
bash scripts/kvaf_pd_proxy.sh \
  --host 141.62.38.12 --port 8000 \
  --prefiller-hosts 141.62.38.12 --prefiller-ports 8100 \
  --decoder-hosts 141.62.38.24 --decoder-ports 8200 \
  2>&1 | tee "pd-logs/proxy/$(date +%Y%m%d_%H%M%S).log"
```

P/D 采集不要求额外采集 Proxy。若需要 Proxy 阶段事件，使用同一解释器启动 `scripts/kvaf_pd_performance_proxy.py` 替代上述 shell 入口，参数相同；不要额外启动第二个 Proxy。

另开终端确认 P/D 健康及 Proxy **实际运行**的配置：

```bash
curl --fail --noproxy '*' --max-time 5 http://141.62.38.12:8100/health
curl --fail --noproxy '*' --max-time 5 http://141.62.38.24:8200/health
curl --fail --noproxy '*' --max-time 5 http://141.62.38.12:8000/v1/kvaf-decode-batch
```

最后一项应包含 `enabled=true`、`dp_size=8`、`max_batch_size=4`。磁盘 JSON 正确不代表旧进程已重新加载。
benchmark 是标准库 HTTP 客户端，建议在 P 容器新终端执行，也可在能访问两个地址且具备同版本脚本的客户端执行，不需要 NPU。
首次将下面的 `--levels 4` 改为 `--levels 1`，8 请求全部完成后，再测试每卡目标 4、共 32 请求：

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2

python3 scripts/benchmark.py \
  --pd-proxy --sync-decode-batch \
  --levels 4 \
  --base-url http://141.62.38.12:8000 \
  --metadata-base-url http://141.62.38.24:8200 \
  --model glm --ignore-env-proxy \
  --input-tokens 130947 --output-tokens 125 \
  --data-parallel-size 8 --tensor-parallel-size 1 \
  --warmup-rounds 0 \
  --request-timeout 7200 \
  --result-dir /home/c00612283/pd-benchmark/sync-run04
```

| 压测参数                                      | 含义                                                                   |
| --------------------------------------------- | ---------------------------------------------------------------------- |
| `--pd-proxy`                                  | 全部生成请求经 Proxy，metadata 后端只提供模型/分词信息                 |
| `--sync-decode-batch`                         | 启用本轮组身份、定向 D rank 及全组 KV 就绪后放行                       |
| `--data-parallel-size/--tensor-parallel-size` | 描述 D 拓扑，不能当作 P 拓扑                                           |
| `--input-tokens/--output-tokens`              | 本例输入加目标输出为 131072，不可超过服务上下文上限                    |
| `--warmup-rounds 0`                           | 不额外预热，适合先定位和比较同条件基线                                 |
| `--warmup-rounds 2 --warmup-batch-size 1`     | 可替换上项，先发两轮各 8 请求，分析默认排除预热                        |
| `--request-timeout 7200`                      | 大于本例组批超时 3600 秒，并为后续计算留时间                           |
| `--result-dir`                                | 固定写在客户端执行位置；同目录下 `live.json` 和 `result.json` 每轮覆盖 |

同一个 `--result-dir` 不再自动建立时间戳子目录，直接覆盖 `live.json` 和 `result.json`。启动新一轮时两者先写入本轮 `run_id`、`started_at_ns` 和 running 状态，结束后写最终状态；预检失败也不会展示旧轮成功结果。结果原子替换，写盘失败保留上一次完整文件，不产生半截 JSON。目录内其他文件和旧版时间戳归档不自动删除；要保留某轮结果，请提前复制结果文件或换一个结果目录。
同一结果目录使用 `.benchmark.lock` 防止两个 benchmark 并发覆盖；进程退出后锁释放，锁文件保留是正常现象，不要手工删除活动锁。若要并行实验，分别指定结果目录。

不需要修改 benchmark 源文件。`--levels 4`、无预热时产生 32 个 P Prefill 和 32 个 D Decode 请求；两轮各 8 请求预热则累计 48 个请求走完整 PD 流程，正式统计仍是 32 个。
同步模式会延迟 D 的放行，TTFT/E2E 包含该等待；TPOT/TPS 仍来自真实 token ID/到达时间。
公共窗口 TPS 为整组 token/s，不是每卡吞吐或全程吞吐；单轮 TPS 样本不提供 P90/P99。

`Measuring`/`Warmup` 的 `0` 和 `live.json` 空结果是实时状态，不能直接称为失败。128K Prefill 可能耗时数分钟；比较客户端 token/完成数、P 队列、D 的 capacity/deferred 和 Proxy 记录，不能只看 HTTP 200 或 NPU 100%。
`KVAF_DECODE_BATCH_RELEASE` 表示 rank 获得放行，不是设备同时开始的证明；`request released` 可能是完成、取消或失败清理，不是成功判据。
实验反馈中，128K 总长、每卡目标 1 的 8 请求已报告 8/8 成功；这不代表每卡 4/16 的容量、精度或采集开销也已验收。

活动轮次中关闭 Proxy 会丢失组状态，重启不能接续旧组。应结束旧 benchmark、清理受影响的 D 状态并确认 P 旧请求结束，再开始新轮次；不要重复叠加发送。
关闭同步策略时，将 D/Proxy 的组批 `enabled=false` 后重启，并去掉客户端 `--sync-decode-batch`；关闭被动采集则在两侧性能 JSON 设置 `enabled=false` 后重启。

## 4. 服务环境内分析与导出

在服务容器/服务器另开终端运行独立短期分析进程，不导入 NPU/SDK，不给推理添加分析回调或分析 API。
完成本轮请求后，在 P 容器集中 P/D 文件。采集路径可沿用上轮，无需为每个 benchmark 改服务配置：传入本轮 `--client-records` 后，按精确请求 ID/代次限定分析集合，自动排除没有本轮选中请求的旧实例；当前部署的 `_dp0` 等 P 分片可共同统计。相同请求身份跨实例或代次有歧义时仍报错，不按文件时间猜测。
下例采集路径 `run02` 必须替换成两侧配置中实际的 `writer.output_dir`，不能因为 benchmark 结果目录名称变了就假定服务端路径也改变。复制过程应在本轮请求结束并等待后台刷新后进行；原始历史仍计入输入扫描预算。

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
mkdir -p /home/c00612283/capture-run02/{p,d}

cp -a /home/c00612283/pd-perf/run02/. /home/c00612283/capture-run02/p/
scp -r root@141.62.38.24:/home/c00612283/pd-perf/run02/. \
  /home/c00612283/capture-run02/d/

# 替换为本轮 benchmark 末尾打印的真实结果路径。
CLIENT_RESULT=/home/c00612283/pd-benchmark/sync-run04/result.json
test -f "$CLIENT_RESULT"

python3 scripts/kvaf_pd_transfer_stats.py summary \
  /home/c00612283/capture-run02/p /home/c00612283/capture-run02/d \
  --client-records "$CLIENT_RESULT" \
  --max-records 10000000 --max-input-bytes 4294967296

# 整组轻量概览：保留请求矩阵及统计，不携带原始时序。
python3 scripts/kvaf_pd_transfer_stats.py export-trace \
  /home/c00612283/capture-run02/p /home/c00612283/capture-run02/d \
  --client-records "$CLIENT_RESULT" \
  --max-records 10000000 --max-input-bytes 4294967296 \
  --trace-detail stages --overwrite \
  --output /home/c00612283/capture-run02/overview.trace.json

# 从概览请求表复制完整 ID；保留整组统计，仅携带该请求时序。
python3 scripts/kvaf_pd_transfer_stats.py export-trace \
  /home/c00612283/capture-run02/p /home/c00612283/capture-run02/d \
  --client-records "$CLIENT_RESULT" \
  --max-records 10000000 --max-input-bytes 4294967296 \
  --trace-detail full --timeline-request-id '替换为完整请求ID' --overwrite \
  --output /home/c00612283/capture-run02/request.trace.json
```

仅开启 Proxy 阶段采集时，额外集中其文件并作为第三个输入目录；原 Proxy 的访问日志不能替代事件文件。
`scp` 首次主机指纹应从目标主机核验，不关闭主机身份校验。P 单侧文件也能分析局部阶段，但缺少的 D/客户端证据保留未知。
将新 Trace 与同版本 `scripts/kvaf_pd_trace_viewer.html` 复制到 Windows，在 Chrome/Edge 打开 HTML 后选择 Trace，无需网络或服务端绘图。
首页查看 TTFT/TPOT/TPS、负载与阶段统计，请求表点击 ID 联动计算/传输双行时序；轻量文件缺少原始时序时使用上面的定向 full 文件。详细操作及历史 run01 重导示例见第 5 节。

输入不能混合不同实验的 P engine。允许同 host/boot、同部署标识下的 `_dp0`、`_dp1` 等兄弟 engine：后缀必须与 `dp_rank` 一致，DP/TP 拓扑和层集合一致，同一请求/代次只能属于一个 P engine。不能为了通过校验删除后缀或伪造采集头；筛选来源应以共同部署标识分组，不把正常 DP 分片当成多次重启。读 schema 1 旧文件、schema 2 和本批 schema 3；原无子命令分析与 compare 仍保持旧用法。
提供 `--client-records` 时，分析仅包含该文件关联的请求（再应用正式/预热、输入长度与请求 ID 筛选），不把其他已标记 measure 的旧请求计入。显式空结果或没有匹配时不会回退统计历史请求；缺少有效证据仍显示 INCOMPLETE。缺少客户端文件时没有自动选择最新实验的依据，混合旧部署仍拒绝。
正式性能分析使用 `summary` / `export-trace`；旧版无子命令流程仅兼容历史原始统计，不提供请求用途筛选。
多个输入长度需 `--input-tokens`，可重复 `--request-id` 精确选原生外部请求 ID。终端显示四组、阶段分布、客户端可得 E2E/TTFT 和完整性。
未导入客户端证据就显示 E2E 未知；不以第一个服务事件为请求发送，也不把传输累计当作 E2E。

默认输入字节上限 256 MiB、记录及展开事件上限各 100000、Trace 上限 256 MiB，可用
`--max-input-bytes`、`--max-records`、`--max-trace-bytes` 显式调整；不是分析进程 RSS 限额。
客户端文件也计入输入预算。超预算明确拒绝，不静默截断或拆成多个 Trace。Trace 默认不覆盖已有输出；传入 `--overwrite` 后，成功导出才原子替换，失败保留旧 Trace，且拒绝覆盖客户端记录或原始采集文件。
快照只读最后完整 checkpoint，不强制业务线程 flush；相同副本去重，缺失分片、冲突副本或损坏完整记录明确报错。

原 begin 已提交而 end 未提交的事件显示为“未结束”的起点，持续时间未知，不用当前时间补造结束。
已知丢样/故障来源的核心样本排除，失败/取消/未完成另计。原路径成功恢复的错误调用保留为诊断事件，不等同于请求推理失败。
`LOCAL_ONLY`/返回 0 表示局部分析成功，不代表完整 E2E、精度或性能验收；已知不完整证据显示 `INCOMPLETE`/返回 2，仍可导出诊断 Trace。

### 4.1 现有客户端记录

可直接导入新版 benchmark 的 `result.json`。新增字段 `response_id`、`response_id_consistent`、`endpoint` 和
`completion_prompt_index` 来自实际响应及现有单 prompt 请求结构。
Proxy 会生成自己的请求 ID；只按仓库实际 `/v1/completions` 子请求规则 `response_id-prompt_index`，与观察到的原生 ID 精确匹配。
若多个候选都匹配、服务 ID 不一致、generation 有歧义或 ID 不存在，就不产生匹配样本；不剥离任意前后缀、不按时间接近配对。
已有旧压测结果缺少服务端 ID 时，只有其原始 request_id 恰好等于真实原生 ID 才能关联，不假定一定可用。

其他发送工具可导出以下 JSONL（示意，数值不代表实测）：

```json
{
  "external_request_id": "实际原生外部ID",
  "generation": "实际代次",
  "request_phase": "measure",
  "clock_domain": "client-run-1",
  "sent_ns": 1000000,
  "first_output_ns": 2000000,
  "completed_ns": 4000000,
  "status": "ok",
  "output_tokens": 125
}
```

也接受 `request`/`request_id`、`start_ns`/`first_ns`/`end_ns`、`completion_tokens` 别名。
generation 省略时，只有这个精确原生 ID 在输入中存在唯一代次才补齐；重复请求/代次拒绝，重试需先由发送工具明确归并。
时间必须来自同一客户端单调时钟，成功状态才进入 E2E/TTFT 分布，输出数未知保持 null。
同一 benchmark 文件共享其客户端时间域；通用记录未声明 clock_domain 时每条隔离，绝不假定可共轴。
客户端时间不与 P/D Host 时间裸减，即使已经精确关联到同一请求。

### 4.1.1 只统计正式请求

发送端通过结果中的 `request_phase` 明确标记 `measure`、`warmup` 或 `profile`。
默认 HTTP 请求格式保持原状；显式开启 `--tag-request-phase` 才新增 `X-KVAF-Request-Phase` 头。
缺失标签为 `unknown`，不能根据第几条请求、名称里的 warm/probe、输入长度或耗时推断。
新版 benchmark 自动给 APC 预热和探针标记 `warmup`，profiling 标记 `profile`；一般阶梯默认 `measure`。
另提供 `--warmup-rounds`、`--warmup-batch-size` 请求预热，默认 0 轮；全部预热返回后才进入正式波次。
用 `--request-phase warmup` 可把整次普通阶梯声明为请求预热。具体命令见 [压测说明](../../scripts/README_benchmark.md#请求预热与正式测试)。

分析端默认 `--request-phase measure`，先按精确 ID/代次关联用途，再识别输入长度并计算四组 t1/t2、阶段分布、
客户端 E2E/TTFT、实际 batch 与请求矩阵。Trace 的事件、membership、统计和图表使用同一筛选集合。
预热的错误、取消和不同输入长度不会作为正式性能样本；采集来源本身的丢样/故障仍保留全局完整性约束。
Proxy 会保存入站头的标签，不改变生成请求 ID 的规则，也不向 P/D 的业务 KV 协议加入未知字段。
仅有 P/D 文件时需提供带标签的客户端结果；Proxy 和客户端对同一 ID/代次的标签冲突会明确报错，不自动选一方。

终端和 Trace 保存用途计数、正式选择数、排除数及无法关联的客户端记录数。浏览器标明“仅正式请求”。
只有预热或未知请求时，正式有效 N 为 0、状态为 INCOMPLETE/返回 2，不自动把剩余请求当作正式请求。
诊断用途可显式选择 `--request-phase warmup|profile|unknown|all`，页面会标为诊断；此时不宣称结果为正式测试统计。
旧 benchmark 文件即使存放在 levels 中，没有明确用途标签也不自动认定正式。需重新发送带标签请求，或提供经核实的用途清单。

其他发送工具可在 Proxy 请求头设置 `X-KVAF-Request-Phase: measure` / `warmup`，或通过 `--client-records` 提供 JSONL：

```json
{"external_request_id":"实际正式请求ID","generation":"实际代次","request_phase":"measure"}
{"external_request_id":"实际预热请求ID","generation":"实际代次","request_phase":"warmup"}
```

仅用途清单不需要伪造时间戳，仍可筛选服务端统计；缺少客户端计时就保持 E2E/TTFT 未知。
需要 E2E 时补入第 4.1 节的真实时间字段。每个请求/代次只提供一条已核实客户端记录。
原始采集仍可能保留预热事件，受相同缓冲/写盘预算约束；默认统计和导出排除它们。
正式请求与预热共用的 batch 只保留正式 membership，但保留实际 batch 大小与完整共享跨度，不按人数拆分计算耗时。
Scheduler 的用途筛选 batch 统计来自所选请求关联的 D_SCHEDULED 事件，不拿整个窗口的直方图冒充正式统计。

这里的请求预热是真实 HTTP 生成请求，与实例初始化/dummy forward 不同；没有恢复任何实例阶段计时需求。

### 4.1.2 关闭采集的路径

使用原 Connector/Proxy 入口，或采集 JSON 的 `enabled=false`、`sample_rate=0` 时，不创建 observer、采集队列、采集后台线程和文件。
请求/逐层路径的 observer 为 None 时直接执行原操作，跳过采集计时、序列化、元数据复制和所有采集回调。
Proxy 关闭时不读取用途头，也不增加任何请求、重试、同步或 SDK 操作。
benchmark 的客户端 HTTP 压测指标是原有能力，与服务采集开关独立；新增用途标签只写客户端结果，默认不改变请求头。
仍有少量关闭条件判断，不宣称未经实机 A/B 证明的绝对零开销；原逻辑、设备操作与性能影响分别验证。
`writer.enabled=false` 仅关闭写盘，仍有内存采集，不能作为关闭采集使用；配置按轮固定，修改 JSON 不会热切换已运行的 observer。
独立的同步 Decode 组批开关若仍开启，仍会执行其主动等待及控制通信；要恢复普通调度性能，应同时关闭组批开关。采集关闭不等于撤销显式选择的压测策略。

### 4.2 可选 Proxy 采集

原 `scripts/kvaf_pd_proxy.sh` 保持不变。需要 Proxy 细分阶段时，选择下列工具入口，参数沿用原脚本：

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
/usr/local/python3.12.13/bin/python3 scripts/kvaf_pd_performance_proxy.py \
  --host 141.62.38.12 --port 8000 \
  --prefiller-hosts 141.62.38.12 --prefiller-ports 8100 \
  --decoder-hosts 141.62.38.24 --decoder-ports 8200
```

路径/IP 换为实际值；同一个 JSON 中 `enabled=true` 才采集。入口与原 shell 一样移除 HTTP(S)/ALL 环境代理，防止后端连接被转发。
恢复原 Proxy 入口或关闭 JSON 即停用。没有新增 Proxy 原生参数或环境变量。

| Proxy 指标                  | 本批语义                                                                                                    |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `Proxy.route_and_prepare`   | 原 handler 入口至 D 选择完成，包含原正文处理、UUID 锁和路由准备；不包含 socket/ASGI 进入 handler 之前的等待 |
| `Proxy.dispatch_wait`       | D 选择完成至首次 D HTTP 派发；重试导致起点/终点不唯一时不补造单一跨度                                       |
| `Proxy.metaserver_gather`   | 原 metaserver 锁/同批元数据汇合等待，到获得原 P task 为止；不将随后等待 P 返回的时间混在该值中              |
| `Proxy.prefill_task_queue`  | 原 create_task 至任务实际开始；重复/重算无法唯一配对时不推断                                                |
| `Proxy.prefill_http`        | 原 P HTTP 调用及处理，失败尝试还可能包含原重试退避；不是纯网络或纯 P 计算                                   |
| `Proxy.decoder_http_stream` | 原 D HTTP 响应流；包含后端等待、网络和原消费者背压，不能当作纯 D 计算                                       |

`PROXY_FIRST_BODY_CHUNK` 是原 HTTP 响应首个 body chunk，可能是控制内容，不标为首 token/TTFT。
`PROXY_STREAM_CLOSED` 表示原流及请求清理已走到对应位置，不证明客户端收到完整输出。
客户端完成仍使用发送工具证据；连接池内部排队和 handler 之前的服务器等待保持未知。

Proxy 早期尚未获得 KVAF generation 时，只保存原 API 子请求 ID。离线仅在该 ID 对应唯一已观察代次时补齐。
代次缺失/有歧义时显示 `unresolved`，保留 Proxy 本地证据，不加入跨端确定关系或作为客户端自动匹配依据。
批量 prompt 的公共 HTTP span 仅存一份，membership 只关联原来属于该批的采样请求。
Proxy 的 batch_size 是原 API 子请求数，不代表 D 每卡实际 batch。

## 5. 单 Trace 与 Windows 查看

当前界面只保留概览、请求矩阵/联动时序、按层传输核验三层。删除独立的阶段分布柱状图、旧多行时间轴及关联图，不重复展示同一请求。内部 Trace 字段保留 t1/t2 兼容旧数据，主界面只用 P→D KV 传输（D2D）、P→K DDR KV 传输（D2H）。

### 5.1 先看已有数据：重新导出

更新整套同版本工具后，可以从旧采集重新导出部署已记录的字段、逐层传输统计、请求矩阵和 TPS。旧采集没有的 EP、实际 batch、逐层计算不会被补造；新源码并不改变旧文件中的事实。
下面沿用本次实验的路径；替换客户端结果为同轮文件。原有事件约 152 万条，因此显式设置输入预算。

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
CLIENT_RESULT=/home/c00612283/pd-benchmark/run01/20261116_110224_7642a1e5/result.json

# 轻量概览：整组请求、逐请求矩阵、层统计都保留，不携带原始时间轴。
python3 scripts/kvaf_pd_transfer_stats.py export-trace \
  /home/c00612283/capture-run01/p /home/c00612283/capture-run01/d \
  --client-records "$CLIENT_RESULT" \
  --max-records 3000000 --max-input-bytes 1073741824 \
  --trace-detail stages \
  --output /home/c00612283/capture-run01/pd-overview-v19.trace.json
```

需要点击某个请求看真实时间轴时，用同一批输入另行导出，输出文件名需是未存在的新文件：

```bash
python3 scripts/kvaf_pd_transfer_stats.py export-trace \
  /home/c00612283/capture-run01/p /home/c00612283/capture-run01/d \
  --client-records "$CLIENT_RESULT" \
  --max-records 3000000 --max-input-bytes 1073741824 \
  --trace-detail full --timeline-request-id '替换为请求表中的完整请求ID' \
  --output /home/c00612283/capture-run01/pd-request-v19.trace.json
```

`--timeline-request-id` 可重复，只限制携带的事件，整组统计和矩阵仍保留。未携带的请求点击后明确显示缺少原始证据；不能由汇总值恢复时间轴。`--request-id` 则筛选整个分析集合并重新计算统计，不可混淆。
仍有采集损坏/预算耗尽时，命令会标记 INCOMPLETE 并返回 2，即使成功写出文件；局部阶段不等于完整传输证据。

将新 Trace 和新版 `scripts/kvaf_pd_trace_viewer.html` 复制到 Windows，Chrome/Edge 打开 HTML 后选择 Trace。无需服务端绘图页面、Python、SDK 或网络连接。

### 5.2 新采集：可选的 P 逐层计算接口观察

要看 P 下一层计算与 KV 传输的时间关系，需要重新采集逐层计算。**默认不开启**，已可跑通的原业务配置不必为了查看旧数据而修改。
在 P 的 `scripts/kvaf_pd_performance.json` 中修改对应字段，缓冲/写盘等其他字段保持原值：

```json
{
  "enabled": true,
  "sample_rate": 1.0,
  "detail_level": "layer",
  "layer_compute": true
}
```

这段只说明需要修改的字段，不是完整配置替换。写盘使用已确认的绝对路径，例如 `/home/c00612283/pd-perf/run01`；可固定复用；提供本轮客户端结果来隔离分析。
D 可保持 `layer_compute=false`，但仍需开启原请求/接收观察。逐层 hook 按当前 GLM 的 `.self_attn.` 所属 Target block 匹配，未匹配时 `hooked_layers=0`，不假定任意模型都具备逐层证据。
P 保留已经验证的 eager 启动配置，不为了观察强制改变原图模式；非 eager P 不安装逐层 hook，checkpoint 明确报告 `hooked_layers=0`。
重启 P 后，在正常真实请求中记录各 Target block 的 Host `forward`；初始化/dummy、未绑定请求、未采样请求不产生逐层计算样本。原模型只调用一次，hook 不改参数、输出、异常，不同步设备、不读取 Tensor，不持有 KV/SDK 所有权。
新开关关闭或零采样率时不安装 hook；关闭路径沿用原模型执行。开启会增加 Host 回调和标量事件量，受既有缓冲/文件预算限制；超限丢样/停采不反压业务。长输入、高 batch 先小规模采集检查质量，再按实际磁盘/内存设置预算。

这是 **Host 调用时序，不是 NPU kernel/DMA Profiler**。如果目标是量化真正设备掩盖率，还需可靠的设备时间证据；本次不会增加采样专用设备同步来补齐它。

### 5.3 页面操作与统计口径

- 首行：TTFT、请求 TPOT、E2E 的 Mean/P90/P99/有效样本数，以及 TPS（token/s）。TPS 来自整组公共输出窗口，每个同负载完整有效轮次为样本；单轮不显示 P90/P99，混合负载不平均，缺公共窗口时保留原因。
- 测试背景：P/D DP/TP/EP、模型、输入/目标及实际输出、目标并发、已关联客户端请求峰值、D 目标/实际 batch。EP 按配置 TP×DP 记录，旧字段未知不推断；实际 batch 取 D 调度记录，不能用并发÷DP 代替。
- 四阶段概要：准备/交互、P 计算/发送准备、Connector/KV 传输、D 接收/开始解码。始终显示四段，短区间最小宽度且注明不完全按比例，缺失占位。主值分别是 D 绑定回调、P 计算/等待并集、P 控制调用并集、D 门控至首轮 Host Decode 等待，均不是纯设备计时，不能相加成 E2E。
- 请求矩阵：每行一个精确 ID/代次，各列为阶段并集；表尾纵向求整组 Mean/P90/P99/有效样本数。筛选和分页只改变显示，不重算汇总；每页最多 32 行。
- 点击请求 ID：高亮所选行，展开唯一的“请求阶段耗时”双行图。切换请求同步更新图与详情，整组汇总不变。共享 batch 保留完整原跨度，不平摊，也不解释成请求独占。
- 计算行：逐层 `forward`；传输行：队列、生产等待、控制及 P→D/P→K 两条路径。并行事件在行内错层；按实际时间保留间隙/重叠。色块“作用＋接口”，短色块优先接口名；悬停说明作用，点击给出边界、层/chunk 和关联高亮，原始字段折叠。
- 缩放/拖动/Rank/时间域/路径筛选只影响所选请求时序。聚焦传输层时保留后续计算。每窗口最多绘制 2000 个事件，超限明确要求缩小窗口，不修改或截断统计；SVG/PNG 保留请求、时间域、窗口和观察口径。
- 传输核验：Target/MTP 和具体模型层独立选择，从原完整性校验后的层样本重算分布。指定层累计标为“每请求所选层累计”；恢复所有层后回到全部层累计。首页和请求矩阵不随此选择改变。

P→D 内容为 Indexer KV＋尾部主 KV，P→K DDR 内容为主 KV。完整主表仍要求 eligible→ACK 的原证据，兼容旧数据时明确是 Pull RPC 口径，不能把显示出来的 RPC 当作完整主表有效样本。
Host 重叠只匹配同请求/代次、同进程/时间域、同 Target 层类型、同 chunk，分别与 L+1 和后续层计算取交集并集。失败、采集不完整、chunk/层号缺失或下一层未观察时不输出比例。Host 重叠不是设备覆盖率，未覆盖时长不是推理阻塞；实际阻塞只解释已有 Runtime/生产/依赖等待观察点。

### 5.4 兼容性、资源与回退

继续兼容 Trace format 1/2 和 capture schema 1/2/3。旧 Trace 可查看已有统计，缺少矩阵/层摘要时提示从现有采集重导；缺少逐层计算需要新采集，不能把整个 forward 拆造成逐层时间。
查看器限制 256 MiB、100 万原始记录、20 万可绘制事件、1 万阶段/层摘要；矩阵限制 1 万请求、20 万单元。超原始事件预算且有摘要时进入整组统计视图，不建立旧关系图索引。大实验建议定向 full 或轻量导出，不要求把整组统计改成单请求。
关闭新采集只需 P 的 `layer_compute=false` 后重启；`enabled=false` 或 `sample_rate=0` 仍关闭全部采集。同步 Decode 屏障是独立实验策略，关闭步骤见 [组批说明](DECODE_BATCH.md)。

## 6. 验证和未完成项

2026-11-17 多 DP 与重复运行修复验证（CPU/本地 HTTP/TCP，非 NPU 实测）：

- 91 项原有 PD、传输统计及 benchmark 测试通过，15 个关闭/零采样率检查通过；没有更改推理、调度、KV 传输或采集热路径。
- 合成 P DP4/TP2 的请求/层统计、同部署多 DP 来源校验通过；拒绝跨主机/boot/重启、拓扑变化、rank 后缀不一致，以及同请求代次跨 DP 合并。
- 原始文件含旧实例、旧 measure 请求、错误来源时，显式客户端集合的样本数、阶段值、状态与干净输入一致；空客户端结果不回退历史数据。
- 真实本地 HTTP 连续两轮覆盖相同目录，预检失败清除旧成功结果；线程外进程竞争也被拒绝。覆盖写盘错误及 Trace 超预算均保留旧完整文件，原始/客户端文件覆盖被拒绝。
- 请求用途、warmup-only、预热失败终止、同步组批 rank/slot 与原 token 载荷回归通过；复用临时检查并适配新的固定结果路径，未新增仓库测试文件。
- `format.sh ci` 本轮仍在 Markdownlint 钩子的 npm Git 安装策略处失败（EALLOWGIT），相关文件另行执行 Ruff、Markdownlint、Prettier 等检查；不宣称全量 CI 或 NPU 性能/精度已验收。

以下为 v1.9 前一批验证（CPU/本地 TCP/Chromium，非 NPU 实测）：

- 91 项原有 PD flow/recovery/integration、传输统计及 benchmark 用例通过。关闭/零采样率另复用 15 个原生用例，禁止调用采集工厂、计时器、回调及元数据复制，原字节、同步次数、恢复和返回/异常保持。
- 可选逐层 hook 用真实 CPU Torch 模块验证一次调用、输出及原异常对象身份不变、dummy 排除、精确层/chunk、采集故障隔离；图模式和新开关关闭均不安装 hook。真实设备开销与精度仍待原环境 A/B。
- 16 请求、2 TP 贡献者、3 层/2 chunk 合成检查原核心统计不变，层筛选有效 N 从 48 到 16；两轮同负载公共窗口 TPS、单轮分位数不可用、混合负载不平均、不同客户端时钟不混算。
- Chromium 验证四阶段短/缺失占位、请求 ID 联动、高亮、双行真实时序、已知 30 ms/60% Host 重叠、不同 chunk/Rank/域/代次拒绝配对、共享 batch 不平摊、拖动/缩放、SVG/PNG、轻量/定向 full 整组统计一致、旧/坏/超预算文件及无网络请求。
- 本轮 Python 语法、Ruff、Prettier、Markdownlint 及相关本地规则单独检查。完整 `format.sh ci` 未通过：钩子引导遇到 npm Git 安装限制、Go 工具不支持 install；其余全量扫描还发现既有 prefix_cache/deepseek_mtp 日志和拼写、KVOP/测试导入规则问题，以及 Gitleaks 下载受限、ShellCheck 并行解压冲突。未将这些视为本轮通过，也不夹带无关业务修复；自动格式化无关文件已撤回。

CPU/替身 SDK 加真实本地 TCP 检查已覆盖：KV 字节一致、原生产同步次数、双 Pull 完成、四组聚合、eligible 与兼容口径区别、
D Commit 丢 ACK 的原恢复次数、采集异常保持原返回/原异常、三档粒度、未结束记录、客户端精确 ID/歧义和独立时间域。
另检查配置 0/1、缓冲丢样/停采、文件分片/总预算、弱引用所有权、半写快照、去重及原子不覆盖导出。
复用现有 PD flow/recovery/integration/统计和 benchmark 测试，未新增仓库 UT 文件；临时验证数据不是 NPU 性能数据。
此前各批 75 项现有 PD/统计测试、16 项 benchmark 测试通过，Python 语法、Ruff、差异空白和 Markdown 检查通过。
关闭路径另复用 15 次原 Connector/传输检查：对关闭配置和零采样率禁止调用采集工厂、队列与计时器；对原 Sender/Receiver/Transfer 禁止调用观察回调及上下文复制函数，分段传输字节、原同步次数、完成回执丢失恢复及 DMA 失败行为均通过。
Runner 检查禁止读取采集用 req_ids、禁止调用观察包装，确认 target/MTP 原操作只执行一次并保留原返回/异常对象。
真实 ASGI Proxy 在两种关闭配置下不创建采集资源，不读取用途头、不调用观察回调；流式字节、请求数和原重试行为通过。
真实本地 HTTP 同时验证默认不带用途头、结果仍标明用途，显式开启才发送标签头；上述检查没有新增仓库 UT 文件，不代表 NPU 实机性能/精度验收。
第四批验证同长度、误导性请求名的预热/正式/未知/profiling 混合数据、预热失败剔除、标签冲突、代次歧义、
混合 batch、所选 batch 直方图及全部统计/Trace 集合一致；真实本地 HTTP 验证 8 个预热请求结束后再发送 2 个正式请求。

Proxy 检查复用原 FastAPI/httpx 处理流程及替身后端，核对启用前后的响应字节、请求数、原重试次数和批量 P 派发一次。
执行仓库已有批量 metaserver 汇合回归函数，验证采集异常及默认 None 对象入口保持原行为。
此处只替换框架导入边界与后端服务，不代表实机 Proxy→NPU 全链路验证。

Linux Chromium 已验证本地选文件、四组/阶段/Target/MTP 表、共享成员、里程碑、客户端表、缩放/筛选/时钟切换、坏文件和 SVG/PNG，页面无网络请求。
云策略禁止直接 file:// 页面导航，检查使用相同 HTML 内容和本地文件选择；Windows 实机和通用 Perfetto 导入尚未验证。

第三批使用 10 万条合成事件检查单文件导出、浏览器加载/筛选、DOM 分页、超预算拒绝与缓存清理。
本环境最后一次结果：38,297,674 字节（约 36.5 MiB），Linux Chromium 加载约 0.71 秒，切换请求约 3.5 毫秒，
全事件视图处理约 199 毫秒，观测 JS 堆约 62.1 MiB。堆读数不是峰值/RSS 或内存限额，不构成 Windows 性能保证，更不是 NPU 推理数据。

仍需逐步补齐或实测：

- Proxy handler 之前的等待、连接池内部排队与设备计算证据仍未知；物理分配时间不可用。SDK/K 内部执行不由 Host 外层跨度代填。
- 图模式/MTP 实机输出、取消/异步边界、长期资源增长和同负载 A/B 性能/精度尚未完成；不能宣称零开销或精度验收通过。
- 跨主机公共时轴及全局关键路径需要同步/完成证据，目前保留独立域和有据关系；Windows、通用查看器及真实长期负载仍待验证。

## 7. 阶段耗时统计原理与打点位置

### 7.1 计时与范围

采集在原调用前后被动读取 `time.perf_counter_ns()`，持续时间为结束减开始；里程碑只记录时间点，不生成虚构持续时间。
统一调用包装见 [observation.py](../../vllm_ascend/distributed/kv_transfer/kvaf/pd/observation.py) 的 `observed_call`、`ObservedLock`。
原操作只执行一次，保留原返回值及异常；不增加设备同步、READ 或控制 RPC。

Host 调用跨度不等于 NPU kernel 执行时间。异步提交或图执行时，函数返回不一定代表设备计算完成。
只统计真实请求；不包含实例启动、内存池初始化、MR 注册/导入、建链、探测或图捕获耗时。
不同主机未对齐的时钟不直接相减，各阶段可能嵌套或重叠，不能将全部阶段相加当成 E2E。

### 7.2 请求准备与元数据交互

| 阶段/里程碑                  | 起止与含义                                                               | 源码位置                                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `D.block_bind_and_advertise` | 原 `update_state_after_alloc()` 回调开始至返回；不包含此前的物理内存分配 | [可选 Connector](../../scripts/kvaf_pd_transfer_stats_connector.py)，`update_state_after_alloc`            |
| `D.metaserver_transport`     | 原 `_access_metaserver()` 调用前至返回，包含通信及等待                   | 同上，`__init__` 内对象级包装                                                                              |
| `D_REQUEST_BLOCKS_READY`     | D 请求 block 绑定完成的时间点                                            | [scheduler.py](../../vllm_ascend/distributed/kv_transfer/kvaf/pd/scheduler.py)，`update_state_after_alloc` |
| `D_RECEIVE_SEEDED`           | Receiver 登记真实请求和 generation 的时间点                              | [receiver.py](../../vllm_ascend/distributed/kv_transfer/kvaf/pd/receiver.py)，`seed`                       |
| `P_REQUEST_BOUND`            | Sender 绑定请求和 generation 的时间点                                    | [sender.py](../../vllm_ascend/distributed/kv_transfer/kvaf/pd/sender.py)，`bind`                           |

### 7.3 P 计算与发送准备

| 阶段/里程碑             | 起止与含义                                                           | 源码位置                                                                                             |
| ----------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `P.target_host_forward` | 原 `_model_forward()` 调用前至返回；Host 观察跨度                    | [runner.py](../../vllm_ascend/distributed/kv_transfer/kvaf/pd/runner.py)，`_model_forward`           |
| `P.mtp_host_propose`    | 原 MTP proposal 调用前至返回；Host 观察跨度                          | 同上，`propose_draft_token_ids`                                                                      |
| `P.send_queue_wait`     | `P_SEND_QUEUED` 至 `P_SEND_STARTED`，即提交 executor 至任务开始      | [sender.py](../../vllm_ascend/distributed/kv_transfer/kvaf/pd/sender.py)，`_dispatch`、`_send_layer` |
| `P.production_wait`     | 原 production event 的 `synchronize()` 调用前至返回；不新增同步      | 同上，`_send_layer`                                                                                  |
| `TRANSFER_ELIGIBLE`     | 原生产等待完成，请求、源、路由信息具备，进入后续源组操作之前的时间点 | 同上，`_send_layer`                                                                                  |

### 7.4 t1/t2 路径与 Connector 内部细分

| 指标            | 开始                                     | 结束                                 | 统计含义                                              |
| --------------- | ---------------------------------------- | ------------------------------------ | ----------------------------------------------------- |
| t1：P HBM→D HBM | 对应请求、层、分段的 `TRANSFER_ELIGIBLE` | 该段 NPU Pull 最后有效完成回执到达 P | Indexer KV 与 tail 主 KV 路径的允许发起至确认完成跨度 |
| t2：P HBM→K DDR | 同上                                     | 该段 CPU Pull 最后有效完成回执到达 P | 主 KV 到远端 DDR 路径的允许发起至确认完成跨度         |

两者包含控制交互、源组/排队等待、传输及回执等待，不是纯 DMA 时间。
归集见 [performance_report.py](../../scripts/kvaf_pd_performance_report.py) 的 `_normalize`；还要求任务具备有效完成确认，缺失/重复/失败回执和分段不完整的样本不作为成功样本。

| 内部阶段                    | 原操作边界                                                   | 观察端/源码                                                                                        |
| --------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| P 控制交互                  | plan/pull/complete/status 等原调用前后，含通信及对端处理等待 | P；[ProducerObserver](../../scripts/kvaf_pd_request_observer.py) 及其原 Sender 调用链              |
| `D.control.*`               | 原控制请求处理入口至返回                                     | D；[receiver.py](../../vllm_ascend/distributed/kv_transfer/kvaf/pd/receiver.py)，`dispatch`        |
| `D.seed_wait`               | 等待请求接收信息的原调用前后                                 | D；同上，`_plan`                                                                                   |
| `D.runtime_lock_wait`       | 开始获取原 Runtime 锁至获得锁                                | D；同上，`_job_operation` 与 `ObservedLock`                                                        |
| `D.compute_dependency_wait` | 原 `wait_compute()` 调用前后                                 | D；同上，接收处理路径                                                                              |
| `D.cpu_prepare`             | 原 `prepare_put()` 调用前后                                  | D；同上，`_cpu_pull`                                                                               |
| `D.cpu_submit`              | 原 `submit_cpu_pull()` 调用前后；提交不等于完成              | D；[transfer.py](../../vllm_ascend/distributed/kv_transfer/kvaf/pd/transfer.py)，`CpuPullTransfer` |
| `D.cpu_wait`                | 原 `wait_cpu_pull()` 调用前后                                | D；同上                                                                                            |
| `D.cpu_commit`              | 原 `commit_put()` 调用前后                                   | D；同上                                                                                            |
| `D.cpu_publish`             | 原地址发布操作前后                                           | D；同上                                                                                            |
| `D.npu_read`                | 原 `batch_get_raw_addr()` 调用前后并检查返回结果             | D；同上，`NpuPullTransfer`                                                                         |
| `D.npu_recovery`            | 原恢复操作调用前后                                           | D；同上                                                                                            |

完成里程碑分别为 `CPU_PULL_DONE`、`DDR_COMMITTED`、`DDR_PUBLISHED` 和 `HBM_PULL_DONE`，由原返回状态检查成功后记录。
CPU Pull 完成、事务提交、地址发布可读是不同条件，不能互相替代。
K 端没有独立内部打点，当前显示的是 D/P 调用方观察；不声称测得 K 内部执行或纯网络耗时。

### 7.5 D 接收门控与 Decode

| 阶段/里程碑             | 起止与含义                                                                                            |
| ----------------------- | ----------------------------------------------------------------------------------------------------- |
| `RECEIVE_GATE_READY`    | D 请求接收及发布条件满足的时间点                                                                      |
| `D.remote_kv_wait`      | `D_REQUEST_BLOCKS_READY` 至 `RECEIVE_GATE_READY`，表示 D 等待远端 KV 就绪的整体跨度                   |
| `D.gate_to_decode_host` | `RECEIVE_GATE_READY` 至 `D_DECODE_STARTED`，表示 KV 就绪后至首次进入该请求 target Host forward 的等待 |
| `D_DECODE_STARTED`      | 首次包含该就绪请求的 target Host forward 入口；不是设备精确执行起点                                   |
| `D.target_host_forward` | 原 D `_model_forward()` 调用前至返回                                                                  |
| `D.mtp_host_propose`    | 原 D MTP proposal 调用前至返回                                                                        |

接收门控位于 [receiver.py](../../vllm_ascend/distributed/kv_transfer/kvaf/pd/receiver.py)，Decode 标记由 [request_observer.py](../../scripts/kvaf_pd_request_observer.py) 关联真实请求生成。
计算边界位于 [runner.py](../../vllm_ascend/distributed/kv_transfer/kvaf/pd/runner.py)。派生跨度由分析脚本按唯一且同时间域的里程碑配对，证据缺失或有歧义时不补造。

### 7.6 Proxy 与客户端

需启用第 4.2 节可选入口才能采集 Proxy 内部阶段。打点位于 [原 Proxy 示例](../../examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py)，由 [proxy_observer.py](../../scripts/kvaf_pd_proxy_observer.py) 保存。

| 阶段                        | 起止                                                    |
| --------------------------- | ------------------------------------------------------- |
| `Proxy.route_and_prepare`   | handler 入口至 D 选择完成                               |
| `Proxy.dispatch_wait`       | D 选择完成至首次 D HTTP 派发                            |
| `Proxy.metaserver_gather`   | 原 metaserver 汇合处理跨度，不混入随后等待 P 返回的时间 |
| `Proxy.prefill_task_queue`  | 原 P task 创建至任务实际开始                            |
| `Proxy.prefill_http`        | 原 P HTTP 调用及响应处理；失败尝试可能包含原重试退避    |
| `Proxy.decoder_http_stream` | 原 D HTTP 流调用及消费，包含等待、网络和消费者背压      |

首个 HTTP body chunk 可能是控制内容，不作为首 token。客户端 TTFT/E2E 来自已有发送工具真实发送、首 token 和完成时间，使用独立客户端时间域。
客户端记录的精确关联见第 4.1 节；无客户端证据时 E2E/TTFT 保持未知。

### 7.7 样本归集、分位数与详细级别

- 请求×层：按请求、generation、层及分段关联；并行贡献者按该段整体起止跨度合并，再累计各段，形成一个请求层样本。
- 请求累计：累加该请求各层有效传输耗时，形成请求样本；层间可能重叠，不等于 E2E 墙钟耗时。
- 普通阶段：使用对应有效 span 的实际持续时间。共享 batch 计算跨度只保存一次，不按成员复制或除以 batch；显示实际 batch。
- mean 为样本算术平均；P90/P99 对排序样本在 `(N-1)×分位数` 位置做线性插值，见 [transfer_stats.py](../../scripts/kvaf_pd_transfer_stats.py) 的 `percentile`。
- 有效 N 是满足该指标完整性要求的实际样本数；默认仅统计显式 `measure` 正式请求。失败、取消、未完成及丢样另行标记。

`detail_level=layer` 适合 t1/t2 与主要请求阶段；锁等待、SDK 调用、Commit/Publish 等细分使用 `detail` 短时采集。
`stage` 缺少逐层传输证据时，对应指标显示不可用；不以外层跨度替代缺失指标。
