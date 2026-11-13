# KVAF PD 性能工具：本地实现与使用记录

日期：2026-11-13。依据：[v1.7 设计](DESIGN.md)，在已确认 v1.6 基准上按用户指示增补请求用途筛选。本文记录实现、使用方法及验证边界。

## 1. 当前交付边界（第四批）

在有界采集、终端分析和请求观察基础上，本批增加可选 Proxy 观察、请求关联图、阶段分布图和较大 Trace 的资源检查。
第四批增加真实请求用途标签：明确区分请求预热、正式测试、profiling 与未知用途，分析默认只统计正式请求。
仅修改仓库自有的 KVAF-PD 外部组件及示例 Proxy，未修改上游 vLLM、KVAF SDK、K 服务或原部署脚本。
原始 KV 读写、事务、完成判定、异常、恢复和资源释放仍由原业务代码执行。
采集不增加设备同步、SDK 调用、READ、控制 RPC、KV 内容检查或路由操作。

**只统计真实请求阶段。** 没有实例启动、模型加载、通用内存池、MR 注册/导入、建链/探测、图捕获或预热的计时与报表。
hello/register/detach 直接透传；没有真实请求绑定的 dummy forward 不生成计算事件。
身份、层类型和时钟标签只是关联元数据。

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
| `scripts/kvaf_pd_trace_links.py`                           | 精确控制身份、已知程序顺序和元数据交接；歧义不连线，不估计跨机网络耗时      |
| `scripts/kvaf_pd_trace_viewer.html`                        | 单 Trace 离线阶段耗时、统计表、时钟域/请求/层筛选、详情与 SVG/PNG 导出      |
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
恢复原 module path 即退出工具接入；插件的 `enabled=false` 或 `sample_rate=0` 也关闭队列、线程与文件。
配置缺失时关闭采集；配置无效时报告停用原因，保留原推理路径。调整配置需要新的采集轮次，不提供运行中热更新。

## 3. 配置与预算

配置路径为 `kvaf_pd_performance.py` 顶部 `CONFIG_PATH`，默认与该模块同目录。
不读取业务 additional_config，不新增环境变量或原生 CLI。完整字段、默认值与约束见 [设计 3.3 节](DESIGN.md#33-可配置采集契约)。

短窗口全请求采集示例，其余字段沿用设计默认值：

```json
{
  "enabled": true,
  "sample_rate": 1.0,
  "sample_seed": 0,
  "detail_level": "layer",
  "writer": {
    "output_dir": "/home/c00612283/pd-perf"
  }
}
```

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

## 4. 服务环境内分析与导出

在服务容器/服务器另开终端运行独立短期进程，不导入 NPU/SDK，不给推理添加分析回调或分析 API：

```bash
# 只校验配置，不创建采集资源。
PYTHONPATH=scripts python3 -c 'from kvaf_pd_performance import load_config; print(load_config())'

# 本地 P 文件即可先看 P/传输统计；缺少的 D/客户端阶段保留未知。
python3 scripts/kvaf_pd_transfer_stats.py summary /home/c00612283/pd-perf \
  --client-records /home/c00612283/capture-run/result.json

# 多源文件已集中到分析侧时，直接终端汇总，并导入已有压测结果。
python3 scripts/kvaf_pd_transfer_stats.py summary \
  /home/c00612283/capture-run/p \
  /home/c00612283/capture-run/d \
  --client-records /home/c00612283/capture-run/result.json

# 同一分析集合汇总成一个 Trace，之后 Windows 不需要这些原始目录。
python3 scripts/kvaf_pd_transfer_stats.py export-trace \
  /home/c00612283/capture-run/p \
  /home/c00612283/capture-run/d \
  /home/c00612283/capture-run/proxy \
  --client-records /home/c00612283/capture-run/result.json \
  --output /home/c00612283/capture-run/kvaf-pd.trace.json
```

输入不能混合不同实验的 P engine。读 schema 1 旧文件、schema 2 和本批 schema 3；原无子命令分析与 compare 仍保持旧用法。
正式性能分析使用 `summary` / `export-trace`；旧版无子命令流程仅兼容历史原始统计，不提供请求用途筛选。
多个输入长度需 `--input-tokens`，可重复 `--request-id` 精确选原生外部请求 ID。终端显示四组、阶段分布、客户端可得 E2E/TTFT 和完整性。
未导入客户端证据就显示 E2E 未知；不以第一个服务事件为请求发送，也不把传输累计当作 E2E。

默认输入字节上限 256 MiB、记录及展开事件上限各 100000、Trace 上限 256 MiB，可用
`--max-input-bytes`、`--max-records`、`--max-trace-bytes` 显式调整；不是分析进程 RSS 限额。
客户端文件也计入输入预算。超预算明确拒绝，不静默截断或拆成多个 Trace；已存在的输出不覆盖。
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
客户端 E2E/TTFT、实际 batch 与代表请求。Trace 的事件、membership、统计和图表使用同一筛选集合。
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

使用原 Connector/Proxy 入口，或工具 JSON 的 `enabled=false`、`sample_rate=0` 时，不创建 observer、采集队列、后台线程和文件。
请求/逐层路径的 observer 为 None 时直接执行原操作，跳过采集计时、序列化、元数据复制和所有采集回调。
Proxy 关闭时不读取用途头，也不增加任何请求、重试、同步或 SDK 操作。
benchmark 的客户端 HTTP 压测指标是原有能力，与服务采集开关独立；新增用途标签只写客户端结果，默认不改变请求头。
仍有少量关闭条件判断，不宣称未经实机 A/B 证明的绝对零开销；原逻辑、设备操作与性能影响分别验证。
`writer.enabled=false` 仅关闭写盘，仍有内存采集，不能作为关闭采集使用；配置按轮固定，修改 JSON 不会热切换已运行的 observer。

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

将 `kvaf_pd_trace_viewer.html` 复制到 Windows 一次，每轮只复制一个 `.trace.json`。
Chrome/Edge 打开 HTML 后选择/拖入文件；页面无上传、CDN 或服务连接，不需要 Python/vLLM/SDK。
Trace 自包含事件、共享 batch membership、关联、统计、配置、完成质量和时钟信息。

查看器支持真实事件阶段耗时、请求/层/路径筛选、缩放平移、分页、点击详情和 SVG/PNG。
默认按阶段折叠跨层阶段耗时，可切换原始逐层视图；只合并行，不改变任何原始跨度或统计值。
里程碑用竖线而非持续条；共享计算事件仅画一次；主表可选全部/Target/MTP，阶段表同时显示 mean/P90/P99/有效 N。
统计对应整个导出集合，下方视图筛选不重算分位数；需要新统计集合时用分析命令重新筛选导出。
点击详情可检查共享成员、实际 batch、同类阶段统计及精确控制 job/op/path/batch 的 P/D 对应事件。
控制重试无法唯一关联时不画确定关系；跨域关系只表示身份对应，不表示单向网络耗时。

本批增加独立于时间轴的请求关联图：已知同域程序顺序为实线，跨域/控制/客户端身份关系为虚线。
没有校准或设备证据时，不计算全局关键路径；缺失阶段不补零、不以几何距离表示时延。
按请求、代次和关系类型筛选，每页最多 30 条关系/60 个节点，支持点击详情及 SVG 导出。
分位数选择器跳到 t1/t2 请求累计 P50/P90/P99 附近的真实请求，并显示该请求实际值，不生成平均请求轨迹。

阶段表和三组并列条形图显示当前时间域的导出集合，每页 12 个阶段；图表支持 SVG/PNG。
请求/文本筛选不重算导出集合的分位数。浏览器为请求、事件 ID 和关系建立索引，避免点击时反复展开全部事件。
查看器上限为 256 MiB、20 万个可绘制事件、100 万条总记录和 20 万条关系；超限明确拒绝。
导出关系也受 20 万条上限约束，不静默截断。分页只限制当前绘制，单 Trace 中仍保留全部所选证据。

Trace format 2 使用标准 `X` 持续事件和 `i` 里程碑，查看器兼容 format 1。
参考 P Host 的事件置于标准 traceEvents；未对齐的 D/客户端域仍在同一文件的 `kvaf_pd.unaligned_domains` 中。
页面逐域显示独立本地轴，不把分别归零当作全局先后。同域有可靠程序/控制配对时附标准 flow。
跨域及仅客户端身份对应不投影为全局 flow。
通用 Perfetto/Chrome Trace 的标准区是兼容目标，完整面板及独立域使用配套查看器。

## 6. 验证和未完成项

CPU/替身 SDK 加真实本地 TCP 检查已覆盖：KV 字节一致、原生产同步次数、双 Pull 完成、四组聚合、eligible 与兼容口径区别、
D Commit 丢 ACK 的原恢复次数、采集异常保持原返回/原异常、三档粒度、未结束记录、客户端精确 ID/歧义和独立时间域。
另检查配置 0/1、缓冲丢样/停采、文件分片/总预算、弱引用所有权、半写快照、去重及原子不覆盖导出。
复用现有 PD flow/recovery/integration/统计和 benchmark 测试，未新增仓库 UT 文件；临时验证数据不是 NPU 性能数据。
本批 75 项现有 PD/统计测试、16 项 benchmark 测试通过，Python 语法、Ruff、差异空白和 Markdown 检查通过。
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
