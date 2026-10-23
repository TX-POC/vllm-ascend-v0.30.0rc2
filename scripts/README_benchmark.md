# HTTP benchmark

纯客户端压测，只需 Python 3.10+ 标准库。通过 HTTP 发送推理请求和读取性能数据，
不修改推理服务，不导入 vLLM，不读取服务日志或服务端文件。

## 使用

常改参数集中在 `scripts/benchmark.py` 顶部的 `Config`，也可用命令行覆盖：

```bash
python3 scripts/benchmark.py \
  --base-url http://127.0.0.1:8080 \
  --input-tokens 8192 --output-tokens 512 --levels 1,2,4,8,16 \
  --data-parallel-size 1 --tensor-parallel-size 8
```

默认连接本机 8080，TP=8、DP=1；输入 130816、输出 256，总计 131072 tokens。
压测前会用 `/v1/models` 返回的 `max_model_len` 校验实际请求总长；接口未提供该字段时跳过此检查。
本机 `localhost`、`127.0.0.1`、`::1` 请求显式绕过环境代理，远端保留环境代理，HTTPS 仍校验证书。

客户端 DP/TP 应与服务配置一致。每档 batch 是每个 DP 的请求数：
`TP=8, DP=1, BS/DP=16` 共 16 个请求；`TP=1, DP=8, BS/DP=16` 共 128 个请求。
TP 只用于说明卡数，不会把相同输出乘以 TP。可见八张卡不代表服务已使用全部八张卡。

可将 `benchmark.py` 和 `benchmark_metrics.py` 一起复制到任意客户端机器运行。
服务需要支持流式 `token_ids`、最终 usage；多 DP 压测需要支持 DP 定向请求。
鉴权使用 `--api-key`，无需安装服务端扩展；此前运行过带自定义 benchmark 扩展的版本，
应先用更新后的启动脚本重启服务。

`Connection refused` 表示目标连接被拒绝，先核对报错中的地址、端口及服务是否已监听。
在运行服务的容器内检查：

```bash
curl --noproxy '*' http://127.0.0.1:8080/v1/models
python3 scripts/benchmark.py
```

宿主机若输出 `ask: session error: ...`，这不是本脚本的错误前缀；用 `type -a python python3`
确认解释器是否被别名或其他程序替代，建议先在容器内使用 `python3`。

## 功能

| 功能 | 用法/行为 |
| --- | --- |
| 合成定长输入 | `/tokenize` 获取合法 token IDs；不同请求使用不同前缀 |
| 相同输入 | `--repeat-prompt`，每个 DP 内重复使用输入 |
| JSONL | `--prompt-jsonl input.jsonl --prompt-field question`，读取首个非空记录 |
| 长度控制 | `--dataset-length-policy truncate` 截断过长输入；`strict` 要求完全相等；过短均报错 |
| 并发阶梯 | `--levels 1,2,4,8,16`，每档同步发起一波请求，使用 `X-data-parallel-rank` 路由 |
| 性能指标 | 全请求重叠区间 TPS/TPOT，以及 HTTP TTFT、E2E、请求 TPOT 和分位数 |
| 调度观测 | `/metrics` 的 running/waiting；缺失明确标注 |
| 预热/APC 校验 | `--prefix-warmup`，分档补充尚未预热的输入并检查 cached_tokens |
| Profiling | 通过服务已有的 `/start_profile`、`/stop_profile` 接口控制 |
| 展示/保存 | 实时终端表格、`live.json` 和 `result.json`，保存原始到达事件及错误 |

APC 校验要求服务已开启 prefix caching 和 prompt token 统计。`--cache-block-size`
需与服务一致，默认 128；校验阈值为输入长度减一个 cache block。
缓存被驱逐或缺少 cached_tokens 时会报错。结果默认保存到客户端
`scripts/result/<时间戳_标识>/`，API key 会隐藏。

## TPOT / TPS 口径

主表统计**客户端全请求重叠区间**：所有请求均已返回首个 token，且尚无请求返回最后一个 token。
所有 DP 的请求统一使用客户端的单调时钟，即使服务分布在多台机器，也不需要校准服务端时钟。

```text
t0 = 所有请求中最晚的首个 token 到达时刻
t1 = 所有请求中最早的最后一个 token 到达时刻
T = t1 - t0（秒）
N = 严格落在 (t0, t1) 内的真实输出 token 总数
B = BS/DP × DP
TPS = N / T
Effective TPOT (ms) = 1000 × B × T / N
TPS × Effective TPOT / 1000 = B
```

按 SSE 返回的真实 token IDs 计数，MTP 一次返回多个 token 就按实际数量累加。
边界上的整个返回事件均排除，不按比例切分；不会把一个 SSE 包当成一个 token。
各 DP 使用同一时间窗，TP 卡不重复计数。短窗口的边界误差更明显，建议增加输出长度。
这是客户端可观察的并发区间，包含网络传输和合包影响，不表示设备 kernel 利用率或实际调度 batch。

HTTP 请求 TPOT 单独统计：首个到最后一个返回事件的时间，除以首个事件之后的 token 数。
它与主表的 Effective TPOT 口径不同；不使用分位数倒数计算吞吐。
请求失败、长度不符、缺少真实 token 数据或没有有效公共区间时，保存原因并报错。

## Profiling

仅在服务已启用原生 profiling 接口时可用；脚本不会添加服务端配置或扩展：

```bash
python3 scripts/benchmark.py --profile-prefill --input-tokens 131071
python3 scripts/benchmark.py --profile-decode --input-tokens 130560 --levels 16 \
  --profile-decode-batch 16 --profile-decode-tokens 32 --output-tokens 512
```

`--profile-prefill` 只执行独立采集：开始后发送一个 DP0、输出 1 token 的请求，完成后停止。
Decode 在正常压测之后另发一波请求，不计入主表；所有请求已返回至少 8 个 token 后开始，以开始确认后的
各请求 token 数为基线，每个请求再返回指定数量的 token 后停止，提前结束则标记不完整。
`--profile-decode-tokens` 表示输出 token，不表示服务端 forward 步数。
HTTP 成功只说明控制请求已被接受，客户端不检查 trace 文件，也不声称已验证 kernel 采集。
