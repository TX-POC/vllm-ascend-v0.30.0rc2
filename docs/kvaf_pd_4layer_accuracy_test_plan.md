# 单机四层模型：混部与 PD 分离输出一致性测试方案

## 1. 目标与结论边界

使用当前同一份 /home/models/glm5-mtp-4l 权重，在单台 8 卡 950DT 上比较原生混部和 PDKVoffloadConnector 分离推理，判断卸载及 PD 改造是否引入额外数值差异。本测试不要求四层截断模型具备完整 GLM5.3 的问答能力，也不能替代全模型准确率或原始多机拓扑验收。

即使混部输出乱码，也可以比较分离是否产生额外差异；两者同样乱码不代表正常模型精度通过。HTTP 200、KV 字节 PASS、文本相同分别是不同证据，不能互相替代。

## 2. 对照组与执行顺序

| 组别 | 部署 | 卡号与并行 | KVAF | 请求入口 |
| --- | --- | --- | --- | --- |
| A | 原生混部基线 | 4,5；TP2/DP1/EP2 | 关闭 | 本机 8200 |
| B | 原生混部卸载 | 4,5；TP2/DP1/EP2 | 开启已有原生卸载路径 | 本机 8200 |
| C | PD 分离，同 TP | P 卡 2,3 TP2/DP1；D 卡 4,5 TP2/DP1，各 EP2 | 开启 PD 远端 DDR | Proxy 8000 |
| D | PD 分离，异构 TP | P 卡 2,3 TP2/DP1/EP2；D 卡 4,5 TP1/DP2/EP2 | 开启 PD 远端 DDR | Proxy 8000 |
| E | 原生混部，异构阶段基线 | 4,5；TP1/DP2/EP2 | 开启已有原生卸载路径 | 本机 8200 |

执行 A → B → C；通过后再执行 E → D。各组顺序运行，正常停止旧引擎后再启动下一组，避免卡号和端口冲突。K 机头 CPU 服务可保持运行，每轮 P/D 都重新初始化 Runtime/region，不复用上一轮请求目录或正在使用的 DDR 空间。

A/B 隔离原生 KVAF 卸载影响，B/C 检查相同 TP 下的 PD 改造影响，E/D 检查原始 P2 → D1 比例。B 和 C 的 DDR 布局实现不同，因此 B/C 仍包含 PD 专用 Backend 布局变化，不能单凭这两组输出差异归因于网络传输。异构 TP 阶段应同时参考 P TP2 的 B 基线与 D TP1 的 E 基线。

## 3. 固定测试条件

- 同一模型文件及 tokenizer；保存权重、config、tokenizer 配置校验值，不修改层数或专家数。
- 同一 Ascend/KVAF SHA、Python、torch/torch_npu/CANN、KVOP 构建与设备环境。
- W4A4C8，主 KV 和 indexer fp8，block-size=128，MTP3，PP=PCP=DCP=1。
- 全部 eager、关闭 prefix caching、c8_enable_reshape_optim=false；相同 seed=1024。
- 初始 max-model-len=8192、max-num-batched-tokens=4096、max-num-seqs=1；每次只发送一个请求。
- 请求 temperature=0、top_p=1、seed=1024，固定输出上限与 stop 参数，stream=false。不混用不同 chat template、工具调用或 reasoning parser。
- 每条请求每组重复 3 次，先确认组内稳定性，再做组间比较。保存实际启动脚本和有效参数，而不只保存计划参数。

不要直接使用 start_vllm_server.sh 的全部默认值作为基线：它默认开启 prefix、async scheduling 和图模式，还设置了解析器，与本轮 PD 控制条件不同。

## 4. 启动准备

沿用现有已编译容器和 KVAF，不为这次测试重新安装 vLLM。确认已有源码版本至少包含 de01cdf55 的分块副本日志修复。源码安装和 K 构建见[已有 vLLM 容器教程](kvaf_pd_existing_vllm_tutorial.md)。

保存独立实验目录，记录 git log -1、环境版本、模型校验值、配置及每轮 server.log。建议命名 accuracy-4layer/A、B、C、E、D，响应文件为 case_id.repeat.json。每次启动日志使用脚本打印的精确路径，不用“最新日志”自动推断跨轮归属。

### A / B / E：原生混部脚本

复制当前四层 d_start.sh 为单独的 baseline_start.sh，保留环境初始化、日志、量化和模型参数，再按以下字段修改：

1. 移除 --worker-cls PDKVWorker 整项，使用框架原生 Worker。
2. 移除完整 --kv-transfer-config JSON，确保不再加载 PD Connector。
3. A 的 kvaf_config.enabled=false；B/E 为 true，保留实际 build_dir、cpu_address=141.62.38.12:28997 和合适的 DDR 预算。
4. A/B 使用 TP2、DP1、data-parallel-size-local=1、卡 4,5；E 使用 TP1、DP2、data-parallel-size-local=2、卡 4,5。
5. 按第 3 节统一 eager、prefix、MTP、并发、上下文等配置，保存到各组独立日志目录。

B/E 使用此前已经适配的原生 KVAF Backend，不使用 PD 专用 Worker。若 B/E 不支持本轮配置或启动失败，应记录为“该对照组不可用”，不能悄悄将其改为关闭 KVAF 后继续使用原组名。

### C / D：PD 脚本

使用仓库根目录现有四层 p_start.sh、d_start.sh，或已保存的现场版本；不要使用双机全模型模板的模型路径。

C：P TP2/DP1，D TP2/DP1。D 控制端口为 31000、31001；P decode_endpoints 保留这两个地址。

D：P TP2/DP1，D TP1/DP2。控制端口仍是 31000、31001，D 的 DP RPC 端口与 P 分开。两组均使用卡 2,3 与 4,5。

启动顺序 K → D → P → Proxy。Proxy 指向本机 P:8100、D:8200，业务统一向 141.62.38.12:8000 发 POST。/v1/models 仅查模型，不触发推理；直接向 PD D:8200 发普通请求不能替代分离测试。

## 5. 测试用例

| 用例 | 输入覆盖 | 主要目的 |
| --- | --- | --- |
| S01 | 小于 128 实际 prompt tokens | tail、indexer/scale、末 token 重算 |
| B01～B03 | 127、128、129 实际 prompt tokens | 满 block、无 tail、有 tail 边界 |
| B04～B06 | 255、256、257 实际 prompt tokens | 多 block 边界与寻址 |
| C01～C03 | 4095、4096、4097 实际 prompt tokens | Prefill 分块附近的通知/拼接 |
| L01 | 约 6K tokens，非 128 整倍数 | 多 chunk、DDR 与 tail 同时覆盖 |
| L02 | 现有 41510 tokens 的固定请求 | 长上下文多轮传输，单独配置轮次 |
| R01 | 同一条短请求重复 3 次 | 组内确定性 |
| R02 | 长 → 短 → 长，顺序且每条完成后再发 | 请求隔离和缓存复用 |

这里的长度以实际 tokenizer 及响应 usage 为准。可以先使用 /v1/completions 原始 prompt 做边界测试，避免 chat template 的额外 tokens 干扰；之后单独验证 /v1/chat/completions。不要把汉字数或字符串长度当 token 数。精确边界用容器中匹配的 tokenizer.encode 检查，并用服务返回的 prompt_tokens 复核。

L02 在全部组统一 max-model-len=65536 后重测，单独保存该配置下的基线。重新核算 HBM/DDR 预算，输入加输出不得超过 65536。不能用 8192 配置下另一条短请求作长请求的基线。

## 6. 发请求与保存结果

所有组使用完全相同的请求文件。下面的短请求用于先打通对照，不代表长上下文验收：

```bash
mkdir -p /home/c00612283/accuracy-4layer/requests
cat > /home/c00612283/accuracy-4layer/requests/S01.json <<'JSON'
{
  "model": "glm",
  "prompt": "中国的首都是",
  "temperature": 0,
  "top_p": 1,
  "seed": 1024,
  "max_tokens": 64,
  "logprobs": 5,
  "stream": false
}
JSON
```

原生混部 A/B/E：

```bash
# 每次修改为实际组名和重复编号
mkdir -p /home/c00612283/accuracy-4layer/A
curl --fail --noproxy '*' --max-time 600 \
  http://141.62.38.12:8200/v1/completions \
  -H 'Content-Type: application/json' \
  --data-binary @/home/c00612283/accuracy-4layer/requests/S01.json \
  -o /home/c00612283/accuracy-4layer/A/S01.1.json
```

PD 分离 C/D：

```bash
mkdir -p /home/c00612283/accuracy-4layer/C
curl --fail --noproxy '*' --max-time 600 \
  http://141.62.38.12:8000/v1/completions \
  -H 'Content-Type: application/json' \
  --data-binary @/home/c00612283/accuracy-4layer/requests/S01.json \
  -o /home/c00612283/accuracy-4layer/C/S01.1.json
```

对每条用例分别保存 .1、.2、.3，不覆盖已有响应。发生 HTTP 错误、超时或非 JSON 响应记录为失败，不计入有效输出比较。若当前 Proxy 未完整保留 logprobs，先记录这一限制，不能用缺失的概率数据作一致性结论。

## 7. 对比指标与判定

| 指标 | 判定方式 |
| --- | --- |
| 输入一致性 | 请求文件相同，prompt_tokens 相同，实际权重和配置一致 |
| 组内稳定性 | 同组同请求 3 次生成序列一致；不一致先归类为非确定性 |
| 输出序列 | 优先比较原生返回的 token IDs（若接口支持）；否则比较 logprobs.tokens 和文本，文本一致不等于已证明 token IDs 一致 |
| 首个分歧 | 记录首个不同的生成 token 序号、双方 token/概率与 finish_reason |
| 概率差异 | 在相同生成前缀上比较对应 token logprob；前缀分歧后的整段概率不能直接逐位置比较 |
| 请求完整性 | completion_tokens、finish_reason、截断/停止条件一致或解释差异 |
| 数据完整性 | PD 本轮已选请求的地址、HBM/DDR 字节、副本、tail 交接证据无 FAIL |
| 隔离性 | R02 的短请求与独立运行短请求结果一致，重复长请求无额外变化 |

首轮同 TP 的严格目标为：组内稳定，所有规定用例生成 token 序列与参考组一致，且 PD 数据校验无 FAIL。即使文本一致，只要数据校验 FAIL，也不能通过。

如 token 不一致，标记“严格一致性未通过，待定位”，不直接宣布网络损坏，也不以“浮点误差”自动放行。量化、batch 形状、Prefill/Decode 算子路径可能引入数值差异，需要根据首个分歧及相同前缀的概率差进一步判断。概率差的容差应在看到实验结果前约定，本方案不虚构适用于 W4A4 的统一阈值。

## 8. PD 证据采样

诊断请求数按请求代次计数，每 worker 上限 16。初次 C/D 启动时两侧设 diagnostics_requests=2、verify_kv=true，优先采集 S01 和 L01；边界问题可重启后针对特定用例采样。进行全套重复比较时可以关闭诊断再跑一轮，确认诊断开/关不会改变输出；两轮结果分开记录。

把本轮准确的 P/D 日志路径写入变量后执行：

```bash
python3 /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/check_kvaf_pd_diagnostics.py \
  "$P_LOG" "$D_LOG" --require-ddr --require-compute
```

该命令应覆盖已采集的长请求；仅短请求时不要强制要求 DDR。已有 generation 隔离、异步计算完成与 DDR ticket 打印问题尚未全部修复，当前审计 PASS 仅作辅助证据。保留原始日志与已知限制，不能将它作为端到端准确度证书。SFA_DONE/MODEL_FORWARD_DONE 当前是调用返回证据，HOT_GET_SUBMITTED 不能证明实际 miss 数非零。

## 9. 结果表与定位顺序

每个用例填写：组别、配置/SHA、输入 tokens、输出 tokens、三次组内是否一致、与参考组首个分歧、KV 审计结果、是否存在设备错误、结论。

- A 自身不稳定：先固定原生配置与运行环境。
- A/B 首次不同：优先检查原生卸载与计算输入，暂不归因于 PD。
- B/C 不同且 PD KV_CHECK FAIL：按具体层/路径定位传输或接收布局。
- B/C 不同但 KV 字节一致：比较正确的逻辑 token/block 对应关系、D 的 tail 重算、Hot Get/TopK/indexer 以及 SFA 输入，字节相同不能排除选错源数据。
- C 通过而 E/D 不同：关注 TP 比例变化、indexer block 贡献方划分和量化并行数值差异。
- 所有组同样乱码且稳定：该四层模型可用于改造一致性验证，无法据此宣称完整 GLM5.3 的语义准确度正常。

本方案不改动原生核心，不引入新的推理功能或默认日志；未在云环境执行 NPU 端准确度测试。最终交付应包含请求集、各组启动脚本、原始响应、分歧表和本轮日志。
