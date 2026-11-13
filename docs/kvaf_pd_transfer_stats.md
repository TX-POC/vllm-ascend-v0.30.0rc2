# KVAF PD 传输时延临时统计工具

当前已开始逐步实现 v1.6 性能工具：可配置采集、服务环境内 `summary`、单 `export-trace` 和离线查看器已在本地落地。
请优先按 [本批配置、命令与边界](kvaf_pd_performance_tool/IMPLEMENTATION_STATUS.md) 使用。
新版采样 Connector 需要同目录 `kvaf_pd_performance.json` 中显式 `enabled=true`，随仓默认关闭。
下文的原无子命令统计流程保留用于旧 schema 1 文件；新版 schema 3 采用 summary/export-trace，两种新命令均可读 schema 1/2/3。
正式测试请使用 summary/export-trace：默认仅选择显式 request_phase=measure，预热、profiling 和未知用途均排除。
仅 P/D 文件需同时提供带标签的 --client-records；历史无子命令分析不提供用途识别。

本工具仅采集和离线统计，不发送推理请求、不修改 KVAF SDK 或框架核心，也不自动修改现有启动脚本。
文件为 [统计脚本](../scripts/kvaf_pd_transfer_stats.py) 和
[采样 Connector](../scripts/kvaf_pd_transfer_stats_connector.py)。通过原生 Connector 配置显式启用，
默认部署不加载采样工具；启用和关闭均需重启 P/D。

## 1. 场景与结果

场景为两台 A 节点：P TP8/DP1/EP8，D TP1/DP8/EP8；两台鲲鹏共四个 K server，
每个 CPU/socket 服务对应两个 D NPU。沿用已有部署，采样不改变 NPU→K 映射。

请求输入 131072 token、输出 125 token，D 每卡目标逻辑 batch=16。
请求数量、并发、预热和输出长度校验均由已有发送工具负责。

| 指标 | 路径 | 内容 | 粒度 |
| --- | --- | --- | --- |
| t1 | P HBM → D HBM | 原路径的 Indexer/Scale 与 tail 主 KV | 请求×层、请求累计 |
| t2 | P HBM → K DDR | CPU Pull 的主 KV，方向通常称 D2H | 请求×层、请求累计 |

四组分别输出平均值、P90、P99 和有效样本数，单位 ms。
请求累计值为该请求各层耗时之和，不是首段开始到末段完成的墙钟时间。

## 2. 计时与开销边界

在 P Sender 的 `op=pull` 控制调用进入前读取单调时钟，调用成功返回完成回执后再次读时钟。
原生 D 仅在 NPU READ 完成，或 CPU Pull 完成并 Commit/Publish 后返回成功。
这段时间包含 Pull RPC 的参数编码、连接、TCP/服务端排队、Runtime/K 等待、SDK 调用及回执返回。
此前的 P 计算、发送线程池等待、plan RPC、建链注册不在该区间内；它不是纯设备 kernel 时间。

所有 t1/t2 时钟都来自同一 P 主机，跨 TP 进程使用同一 monotonic 时基；不做 P/D/K 跨主机时钟相减。
同一请求、同一层、同一 chunk/path 的并行 TP 分支，取最早 Pull 发起到最晚回执返回的跨度。
一个分支多次 Pull 时也取首个发起到末次完成的跨度，包含其间原有控制准备。
再累加不同 chunk 的耗时，排除两段之间的 P 计算间隔，最后累加各层得到请求累计值。

采样只包装显式选择的 Connector 内的 Sender 实例，原调用参数、返回、异常、恢复、释放顺序保持原样。
不增加设备同步、Tensor 回读、哈希校验或额外控制请求；不启用 SDK profile，不替换业务 Pull 接口。
每次 Pull 只读两次主机时钟并更新内存状态。每个请求×层仅完成时入队一份分段汇总，
后台线程负责 JSON 编码、缓冲文件写入及每五秒 checkpoint；不向业务日志逐调用打印。

有界队列满时丢弃测量数据，不阻塞业务；丢样或采样异常会使离线结果标记 INCOMPLETE。
这仍有少量 CPU、锁、内存和后台 I/O 成本，不能承诺零影响，必须通过真实负载的 A/B 对照验收。
D 仅在原生 Scheduler build_connector_meta 后累计 `num_scheduled_tokens` 的请求数直方图，
记录的是实际调度逻辑 batch，不是配置上限、图 padding 大小，也不保证整个测试始终 batch=16。
直方图覆盖整个 D 采集窗口，包括预热；--skip-first 和 --request-ids 仅过滤传输样本。
验收目标 batch 时应使用独立测试窗口，例如低并发预热后仅运行目标负载，避免混入其他批量实验。

## 3. 测试前核对

- 131072 输入加 125 输出至少需要 131197 上下文长度。四舍五入留出 MTP 余量时可考虑
  `--max-model-len 132096`，前提是模型及当前框架允许；不能继续沿用 131072 上限。
- D 使用 `--max-num-seqs 16`，最终以每个 DP 的调度直方图验证是否达到目标。
  P 使用原生调度，max-num-seqs、token budget 和外部并发由实际负载配置，不由本工具更改。
- 重新核算 HBM/DDR 容量。按 79 层、656 bytes/token、16 个 131072-token 请求估算，
  每 NPU 的主 KV 裸数据约 101 GiB；按 86016-byte slot 对齐约 104 GiB。
  实际取决于目标层、保留 Hot/Tail 和卸载规则，不能直接沿用旧的每 worker 16 GiB 配置。
  两个 NPU 共用一个 K 服务时，CPU/socket 容量须覆盖两个独立 region 及开销。
- P/D 同时设置 diagnostics_requests=0、verify_kv=false；不要同时进行额外 Profiler/字节验证。
- 保留原有图模式、量化、MTP、TP/DP/EP、source_tp0_only 及网络配置，避免改变对照变量。
- 本工具不会验证实际生成了 125 token，请由发送工具核对成功请求和实际输出长度。

两个 NPU 对应一个 CPU 的映射示例，IP 使用实际机器替换：

| D 物理 NPU | K endpoint |
| --- | --- |
| 0、1 | K0:28997 |
| 2、3 | K0:28999 |
| 4、5 | K1:28997 |
| 6、7 | K1:28999 |

## 4. 接入已有启动脚本

将两个 Python 文件放入 P/D 框架仓库的 scripts 目录。离线统计所在机器只需标准库，无需安装 vLLM 或 SDK。
采样端使用现有服务 Python，先确认目录可写。每次实验使用全新目录，不混入历史启动文件。
可以编辑统计脚本顶部 CAPTURE_DIR；默认路径为仓库 `scripts/result/kvaf_pd_transfer_stats`。

在 P/D 启动脚本加载 KVOP 后，将 scripts 目录加入 PYTHONPATH，保留已有 SDK 路径：

```bash
export PYTHONPATH="/home/c00612283/vllm-ascend-v0.30.0rc2/scripts:/home/c00612283/KVAF-Devlop/build-pd-py312/python${PYTHONPATH:+:$PYTHONPATH}"
```

两侧的 --kv-transfer-config 中，只替换 module_path，并关闭原有诊断：

```json
{
  "kv_connector": "PDKVoffloadConnector",
  "kv_connector_module_path": "kvaf_pd_transfer_stats_connector",
  "kv_role": "kv_producer",
  "kv_port": 31000,
  "kv_connector_extra_config": {
    "decode_endpoints": [
      "141.62.38.24:31000", "141.62.38.24:31001",
      "141.62.38.24:31002", "141.62.38.24:31003",
      "141.62.38.24:31004", "141.62.38.24:31005",
      "141.62.38.24:31006", "141.62.38.24:31007"
    ],
    "timeout": 600,
    "source_tp0_only": false,
    "diagnostics_requests": 0,
    "verify_kv": false
  }
}
```

以上为 P 示例；D 保留原本的 kv_consumer、协调器、control_backlog 等字段，
同样把 kv_connector_module_path 改为 kvaf_pd_transfer_stats_connector、诊断改为 0/false。
两侧 --worker-cls 继续使用原来的 vllm_ascend.distributed.kv_transfer.kvaf.pd.worker.PDKVWorker。
source_tp0_only 使用实际部署的原值，不能因为复制示例而单侧切换模式。
这里使用现有原生 vllm serve 启动命令；scripts/start_vllm_pd.sh 仅接受原始 Connector，
不能用于启动本次采样 Connector。

排空旧请求后重启 P/D 即生效，不需要重编译。K 不加载此工具，健康 K 和协调器可保留。
按 D → P → Proxy 就绪后，使用原有发送工具发起负载。

## 5. 收集与离线统计

发送工具完成后停止新请求，等待 P 的所有分段传输结束，再等至少六秒让后台写入 checkpoint。
无需先杀 P/D。P 产生 P-*.jsonl，D 各 Scheduler 产生 D_scheduler-*.jsonl。
checkpoint 应满足 enqueued=written、dropped=errors=0；P pending_jobs=pending_layers=0。
不要用 Ctrl+Z 暂停服务或采样线程。

将 P、D 的采样文件汇总到分析机器同一实验目录，例如 capture-run/p、capture-run/d。
挂载代码目录时可直接在宿主机复制文件；容器文件未挂载时先导出。
仅复制本次实验文件，不能混入 P 重启前的 engine 或重复 D DP rank。

```bash
python3 scripts/kvaf_pd_transfer_stats.py \
  /home/c00612283/capture-run/p \
  /home/c00612283/capture-run/d \
  --output-dir /home/c00612283/capture-run/report \
  --expected-requests 128 \
  --input-tokens 131072 \
  --expected-output-tokens 125 \
  --expected-dp 8 \
  --batch-target 16
```

128 是示例，替换成发送工具实际用于统计的请求数。工具不发送这 128 个请求。
短输入预热会按长度过滤；同长度预热可加 --skip-first N，按 P 侧首次传输时刻排除最早 N 个请求。
也可以加 --request-ids successful-request-ids.txt，文件每行一个精确的外部请求 ID，
使用发送工具确认成功的 ID 选择样本。工具只证明传输完成，不能代替 HTTP/模型精度验证。

输出：

- summary.csv：t1/t2 × 请求层/请求累计四行统计。
- by_layer.csv：每个层在所有有效请求上的分位数。
- request_layers.csv：逐请求、逐层、D endpoint、chunk 数和 t1/t2。
- requests.csv：各请求累计 t1/t2，以及实际参与两条路径的层数。
- summary.json：统计口径、覆盖率、拒绝原因、D batch 直方图和原文件列表。

分位数采用线性插值，直接对归集后的样本计算。
无实际传输的层/path 不伪造零时延样本；请求累计值只累加实际传输层。
任何层缺失、TP contributor 缺失、分段不连续或 Pull 完成回执丢失时，整个请求不参与有效分布。
原生失 ACK 恢复仍照常执行，但无法精确恢复原始完成时延，因此测量明确作废，不重放 READ。

返回码 0 表示采集完整、请求数匹配、全部 D DP 都观察到目标 batch；返回码 2 表示 INCOMPLETE。
部分有效数据仍写入报告供排查，不能当作“所有请求已完整统计”。

## 6. 性能对照验收

使用相同发送工具、请求形状、并发、请求数和预热条件，分别测原始 Connector 与采样 Connector。
建议交替重复至少三轮，避免一次抖动影响结论；两个版本都关闭原字节诊断。
对照原始 Connector 时将 module_path 恢复为
vllm_ascend.distributed.kv_transfer.kvaf.pd.connector 并重启 P/D，采样不会被导入或创建资源。
比较的是同一业务负载下吞吐、TTFT、TPOT，而不是用主机微基准推断 NPU 性能无影响。

把已有发送工具导出的性能指标整理成以下 JSON，两份文件使用相同字段：

```json
{
  "output_throughput_tps": 1000,
  "mean_ttft_ms": 100,
  "p99_ttft_ms": 150,
  "mean_tpot_ms": 10,
  "p99_tpot_ms": 15
}
```

还支持 request_throughput_rps；至少提供一个吞吐和一个时延指标。
容许的退化阈值由你指定，例如 1%：

```bash
python3 scripts/kvaf_pd_transfer_stats.py compare \
  --baseline baseline-performance.json \
  --sampled sampled-performance.json \
  --max-regression-percent 1 \
  --output performance-comparison.json
```

该命令只比较已有指标，不执行压测。通过返回 0；任一指标退化超过阈值返回 2。
它不能代替重复实测或证明严格零开销。当前云环境没有对应 NPU 拓扑，实机性能验收须在测试环境完成。
