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

默认连接本机 8080，TP=8、DP=1；输入 130560、输出 512，总计 131072 tokens。
压测前会用 `/v1/models` 返回的 `max_model_len` 校验实际请求总长；接口未提供该字段时跳过此检查。
本机 `localhost`、`127.0.0.1`、`::1` 请求显式绕过环境代理，远端默认保留环境代理；
`--ignore-env-proxy` 可让所有请求绕过环境 HTTP(S) 代理，HTTPS 仍校验证书。

客户端 DP/TP 应与服务配置一致。每档 batch 是每个 DP 的请求数：
`TP=8, DP=1, BS/DP=16` 共 16 个请求；`TP=1, DP=8, BS/DP=16` 共 128 个请求。
TP 只用于说明卡数，不会把相同输出乘以 TP。可见八张卡不代表服务已使用全部八张卡。

可将 `benchmark.py` 和 `benchmark_metrics.py` 一起复制到任意客户端机器运行。
服务需要支持流式 `token_ids`、最终 usage；直连模式的多 DP 压测需要支持 DP 定向请求。
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

## PD Proxy 压测

适用于仓库 `scripts/kvaf_pd_proxy.sh` 启动的 PD Proxy。推理请求发送到 Proxy 的
`/v1/completions`；Proxy 不提供 `/v1/models`、`/tokenize`，因此通过
`--metadata-base-url` 指定使用相同模型、分词器的 P 或 D 后端，仅获取模型信息和 token IDs。
所有压测生成请求均经过 Proxy，不会直接发送给这个元数据后端。

以下示例对应 P/Proxy=`141.62.38.12`、D=`141.62.38.24`，D 为 TP1/DP8：

```bash
python3 scripts/benchmark.py \
  --pd-proxy \
  --base-url http://141.62.38.12:8000 \
  --metadata-base-url http://141.62.38.24:8200 \
  --model glm --ignore-env-proxy \
  --input-tokens 130947 --output-tokens 125 \
  --data-parallel-size 8 --tensor-parallel-size 1 \
  --levels 1,2,4,8,16 \
  --result-dir scripts/result/pd-proxy
```

`input_tokens + output_tokens` 必须不超过模型上下文上限；上例为 131072。
输入 131072 再输出 125，需要服务、模型及容量都支持更大的上下文。

Proxy 模式下 `--levels` 仍按 `level × data_parallel_size` 计算每波总并发，上例为
8、16、32、64、128 个请求。终端显示 `Concurrency`，结果保存 `concurrency` 和
`routing=proxy_balanced`。客户端不发送 `X-data-parallel-rank`，后端 rank 记录为 `null`，
`overlap.per_dp` 为空；由 Proxy 和 D 服务调度请求，不能据此声称每卡实际 batch 为 16。
服务的 `--max-num-seqs`、token 预算和内存容量会约束实际 batch；特别是
`--max-num-seqs 1` 不会因为客户端发送 128 并发就变成每卡 batch 16。

Proxy 模式默认关闭 APC 预热与命中校验，适配现有 `--no-enable-prefix-caching` 部署；
显式指定 `--prefix-warmup` 或 profiling 时，在发送请求前报错。
不请求 Proxy 的 `/metrics`、`/start_profile`、`/stop_profile`，调度观测明确标记不可用。
推理响应仍必须包含真实流式 `token_ids`、usage 和终止标记；缺失、传输中断或长度不符均报错，
不会把 SSE 包数当成 token 数。鉴权参数 `--api-key` 同时用于 Proxy 和元数据后端。
去掉 `--pd-proxy` 和 `--metadata-base-url` 即恢复直接压测 vLLM 的方式。

每条原始请求还保存实际 SSE `response_id`、`response_id_consistent`、`endpoint` 和单 prompt 的
`completion_prompt_index=0`，供 [请求性能工具](../docs/kvaf_pd_performance_tool/IMPLEMENTATION_STATUS.md) 导入 `result.json`。
Proxy 会生成自己的 ID，客户端发送的 `request_id` 不一定等于 P/D 原生请求 ID；分析只按实际服务端 ID 和仓库的子请求规则精确匹配。
响应未提供 ID、ID 改变或匹配有歧义时不补造关联，也不影响原有压测结果的成功判定。

## 功能

### 请求预热与正式测试

`--warmup-rounds` 默认 `0`，不会额外发送预热请求。需要请求级预热时，例如先预热两轮再正式压测：

```bash
python3 scripts/benchmark.py \
  --pd-proxy --base-url http://141.62.38.12:8000 \
  --metadata-base-url http://141.62.38.24:8200 --model glm --ignore-env-proxy \
  --input-tokens 130947 --output-tokens 125 \
  --data-parallel-size 8 --tensor-parallel-size 1 \
  --warmup-rounds 2 --warmup-batch-size 1 --levels 16 \
  --result-dir scripts/result/pd-measure
```

每轮请求预热总并发为 `warmup_batch_size × DP`，上例先发两轮各 8 个请求，全部结束后再发正式的 128 并发。
请求预热使用同样的输入/输出长度，与 APC 前缀预热独立；Proxy 模式也支持，不要求开启 prefix caching。
预热失败时停止本次运行，不继续将尚未就绪的结果当作正式压测。
若只执行预热，可指定 `--request-phase warmup`；普通阶梯请求默认 `--request-phase measure`。

每条结果明确标记用途：普通正式请求为 `measure`，
请求预热、APC 前缀预热和 prefix 探针为 `warmup`，profiling 请求为 `profile`。
默认不增加 HTTP 请求头；需要采样 Proxy 直接保存用途时显式加 `--tag-request-phase`，才发送 `X-KVAF-Request-Phase`。
标签不进入模型请求正文，不改变推理参数。预热保存在 `request_warmups`、`prefix_warmups`、`prefix_probe`；
`levels` 与终端正式性能表只含 `measure`，profiling 仍单独保存在 `profiles`。

性能工具 `summary` / `export-trace` 默认只分析明确标记的 `measure`。
仅有 P/D 采集文件时，同时传入该次 `result.json`，按服务端响应 ID 和原生代次关联标签；
采样 Proxy 可直接保存 HTTP 标签，供 P/D 事件离线关联。不会按请求先后、名称前缀或耗时猜测预热。
旧文件没有标签时视为 `unknown`，默认排除；诊断全部原始请求须显式选择分析端 `--request-phase all`。

### 其他能力

| 功能          | 用法/行为                                                                          |
| ------------- | ---------------------------------------------------------------------------------- |
| 合成定长输入  | `/tokenize` 获取合法 token IDs；不同请求使用不同前缀                               |
| 相同输入      | `--repeat-prompt`，每个 DP 内重复使用输入                                          |
| JSONL         | `--prompt-jsonl input.jsonl --prompt-field question`，读取首个非空记录             |
| 长度控制      | `--dataset-length-policy truncate` 截断过长输入；`strict` 要求完全相等；过短均报错 |
| 并发阶梯      | `--levels 1,2,4,8,16`，每档同步发起一波请求；直连按 DP 路由，Proxy 模式由后端调度  |
| 性能指标      | 全请求重叠区间 TPS/TPOT，以及 HTTP TTFT、E2E、请求 TPOT 和分位数                   |
| 调度观测      | 直连读取 `/metrics` 的 running/waiting；Proxy 模式明确标记不可用                   |
| 预热/APC 校验 | `--prefix-warmup`，分档补充尚未预热的输入并检查 cached_tokens                      |
| Profiling     | 通过服务已有的 `/start_profile`、`/stop_profile` 接口控制                          |
| 展示/保存     | 实时终端表格、`live.json` 和 `result.json`，保存原始到达事件及错误                 |

APC 校验要求服务已开启 prefix caching 和 prompt token 统计。`--cache-block-size`
需与服务一致，默认 128。探针通过 HTTP 回放识别每个 DP 是否额外重算一个尾块
（兼容 MTP），并将结果记录在 `prefix_extra_recompute_blocks` 中，按 DP rank 排列。
正式请求的最低命中数为 `floor((输入长度 - 1) / 块大小) × 块大小 - 额外重算块数 × 块大小`；
例如块大小 128、额外重算一块时，4096-token 探针要求命中 3840，130000-token 输入要求命中 129792。
探针输入必须大于两个块；零命中、低于校准阈值或缺少 cached*tokens 时仍会报错。
校准只允许额外回退一块，不代表识别出了服务的 MTP 配置。结果默认保存到客户端
`scripts/result/<时间戳*标识>/`，API key 会隐藏。

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
Proxy 模式中 B 为该波总并发 `level × DP`，仅统计全局与逐请求指标，不推断实际 DP 分布。
TTFT、E2E 和请求 TPOT 均提供 mean、P90、P99 和有效 count，时间包含 Proxy、排队、
P 计算、KV 传输及 D 解码对客户端响应造成的影响，不能据此单独拆出 Connector 或 KV 传输耗时。

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
