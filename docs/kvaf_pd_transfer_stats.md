# KVAF PD 传输统计历史兼容说明

当前采集和 Trace 导出见 [性能工具使用手册](kvaf_pd_performance_tool/IMPLEMENTATION_STATUS.md)。当前由 `kvaf_pd_performance.json` 管理采集开关和 `writer.output_dir`。

## 读取历史 schema 1 数据

`summary` 和 `export-trace` 可读 schema 1/2/3，正式请求的分析需要精确的客户端请求关联。若要复现 schema 1 历史 CSV，继续使用以下兼容命令：

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

## 旧性能 JSON 对照

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
