# 双机全模型 PD 部署与准确度对照

适用：两台各有 8 张可用 Ascend NPU 的机器，已有匹配 CANN/torch/torch_npu/vLLM 容器。以下按 950DT 环境编排；其他设备需重新验证容量与软件版本。P_NODE_IP、D_NODE_IP 是待填写的地址，不是有效默认 IP。两台机器的实际 IP、网卡、完整模型路径尚需现场填写。

这是临时准确度验证，不替代原始 P TP8/DP2/EP16、D TP1/DP16/EP16 和八台 K 的生产部署。两个独立引擎不能合并计算 EP 大小；下面每个引擎 EP8，而非跨 P/D 的 EP16。双机测试不能覆盖全部生产拓扑。

## 1. 拓扑与测试顺序

| 机器 | 服务 | 首轮并行 | 端口 |
| --- | --- | --- | --- |
| P | Prefill | TP8 / DP1 / EP8，卡 0～7 | HTTP 8100，DP RPC 13389 |
| D | Decode | TP8 / DP1 / EP8，卡 0～7 | HTTP 8200，DP RPC 13489，Connector 31000～31007 |
| D 机头 CPU | KVAF K | 单服务，八个 D worker 各自 Runtime/region | 28997 |
| P 机头 CPU | 原生 PD Proxy | 路由到 P:8100、D:8200 | 8000 |

首轮 P/D 保持同 TP，减少并行变化对数值对照的影响。通过后，在 D 改为 TP1/DP8/EP8，再测原始方案中的 P8 → D1 比例。P、D 是两个独立服务，不传 --nnodes 2；两侧 data-parallel-address 分别填写本机 IP。

建议顺序：D 原生非 PD 基线 → 停止基线 → K → D → P → Proxy → 同输入结果对照。基线与 PD 顺序运行，避免同卡抢占内存。先用 max-model-len=8192、max-num-seqs=1 验证全模型，再增加上下文和并发。

## 2. 新启动模板

本仓提供：

- examples/disaggregated_prefill_v1/kvaf_two_node/p_start.sh：P 全模型 TP8。
- examples/disaggregated_prefill_v1/kvaf_two_node/d_start.sh：D 全模型 TP8。
- examples/disaggregated_prefill_v1/kvaf_two_node/d_baseline.sh：D 同 TP/DP 的原生非 PD、无 KVAF 卸载基线。

直接编辑原生参数和 JSON 字段，不通过额外配置生成器。三个脚本均默认关闭诊断；IP 占位未修改时会退出。模板完整模型路径取自现有 start_vllm_server.sh：/home/models/GLM-5.3-w4a4c8-mxfp4，必须确认是实际完整权重。不要沿用 glm5-mtp-4l，不设置减层或裁剪专家的 hf-overrides。

在 P 的模板中修改：local_ip=P 的可路由 IP，nic_name=本机通信网卡；decode_endpoints 的八个 D_NODE_IP 全部改为 D 的地址。

在 D 的两个模板中修改：local_ip=D 的 IP，nic_name=本机通信网卡。D PD 模板 cpu_address 改为 D_IP:28997。三份脚本同步填写 Python、KVOP env.sh、PYTHONPATH、kvaf_config.build_dir、模型路径。两台路径可不同，但权重与软件版本必须相同。

代码仓和 SDK 目录示例沿用 /home/c00612283/vllm-ascend-v0.30.0rc2、/home/c00612283/KVAF-Devlop；SDK build 使用 build-pd-py312。现场 Python 路径应以实际导入 torch/torch_npu/vllm 的解释器为准，不因示例写了 Python3.12 就直接复制不存在的路径。

## 3. 两端版本、权重与构建

两端记录相同 Ascend 开发分支 SHA、KVAF 分支 SHA；至少包含 de01cdf55 的 chunk 日志身份修复。保存现场手改脚本，再更新工作区。前面发现的诊断 generation/异步完成证据缺口尚未全部修复，不能将审计 PASS 当作准确度证书。

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
git status --short
git log -1 --oneline
git -C /home/c00612283/KVAF-Devlop log -1 --oneline

export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
"$KVAF_PD_PYTHON" -c 'import sys, torch, torch_npu, vllm; print(sys.executable); print(torch.__version__, torch_npu.__version__, vllm.__version__); print(vllm.__file__)'
```

两端读取完整模型 config.json，核对实际层数、专家数、模型类型、量化配置；若是复合 config，主模型字段可能在 text_config。对比 tokenizer/chat template 以及权重文件校验值。初次部署可以在各自模型目录执行并比较清单：

```bash
cd /home/models/GLM-5.3-w4a4c8-mxfp4
sha256sum config.json tokenizer_config.json
find . -maxdepth 1 -type f -name '*.safetensors' -print0 \
  | sort -z | xargs -0 sha256sum > /tmp/full-model-weights.sha256
```

权重清单应非空；其他权重格式或子目录需按实际目录生成清单。不要修改原始模型文件以适应测试。

保留容器已安装 vLLM，在两台各自构建 Ascend/KVOP/SDK：

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2
export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
bash scripts/install_kvaf_pd_source.sh - \
  /home/c00612283/KVAF-Devlop \
  /home/c00612283/KVAF-Devlop/build-pd-py312 \
  2>&1 | tee /home/c00612283/full-model-pd-build.log
)
```

应看到 PD_SOURCE_INSTALL_OK。两端均需 KVAF binding 和 RA 借用补丁：P 负责注册并提供源内存，D 负责接收与 DDR 访问。首次或 C++/ABI 改动才重编；仅更新外部 Python 代码时，editable 安装生效后重启服务即可。详细编译排障见[已有 vLLM 容器教程](kvaf_pd_existing_vllm_tutorial.md)。

K 只在 D 构建。下面枚举兼容参数适用于此前已确认的 UMDK 26.06 头文件，其他版本先核对枚举：

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2
bash scripts/kvaf_pd_k.sh build \
  /home/c00612283/KVAF-Devlop \
  /home/c00612283/KVAF-Devlop/build-hostcpu-k \
  -DKVAF_URMA_INCLUDE_DIR=/usr/include/ub/umdk/urma \
  -DKVAF_URMA_LIBRARY=/usr/lib64/liburma.so \
  '-DCMAKE_CXX_FLAGS=-DURMA_ACCESS_REMOTE_READ=URMA_ACCESS_READ -DURMA_ACCESS_REMOTE_WRITE=URMA_ACCESS_WRITE -DURMA_ACCESS_REMOTE_ATOMIC=URMA_ACCESS_ATOMIC' \
  2>&1 | tee /home/c00612283/full-model-k-build.log
test -x /home/c00612283/KVAF-Devlop/build-hostcpu-k/runtime/kvaf_cpu_service
)
```

## 4. 跨机网络与容量检查

容器须能访问对应 NPU、UB 设备及匹配的宿主驱动。沿用现场已验证的设备映射与容器网络方式；使用 host 网络时，填写宿主机真实可路由地址。不能填写容器内部不可达地址或用 127.0.0.1 代表另一台机器。

除 HTTP 外，还要验证：

- P 能访问 D:31000～31007 和 D 机头 K:28997。
- D 与 K 能访问 P 注册源提供的动态 TCP 控制端口；SDK 使用 0 端口分配监听端口，因此仅放行上述固定端口不够。按现场网络策略检查实际监听和回连，不假定固定端口范围。
- P NPU ↔ D NPU、P NPU ↔ D CPU 的跨机 UB 链路/EID 必须可达。业务网卡 IP 可通不代表 UB 数据面可用。可选 EID maps 必须按实际物理卡和拓扑配置，不能复制旧机器 EID。
- 各引擎本机 HCCL 八卡集合通信正常。此前单机双卡 SDK 测试成功不等于跨机 P/D 传输通过。

模板 D 的 ddr_size_gb=4 是每 worker 预算，八个 worker 合计约 32 GiB，另留 Runtime/元数据空间；不是整个 K 仅 4 GiB。主 KV 的基础预算可按：主模型层数 × floor(上下文/128) × 128 × 656 × 每 worker 并发，另加对齐与空闲余量。MTP 主 KV 仍驻 HBM，不计入 Target DDR 公式。

全模型权重、量化 scale、热 KV、indexer、MTP 和通信 buffer 都占 HBM；没有现场 profiling 不能保证可装载。先单并发、短上下文。不要通过裁剪层数或专家数解决“全模型”测试的容量问题。

## 5. 先建立全模型原生基线（D）

在 D 上编辑 d_baseline.sh，确认路径和 D 本机 IP。基线关闭 KVAF，移除了外部 Worker 与 Connector，保持 TP8/DP1/EP8、W4A4C8、MTP3、eager、相同 tokenizer 与 seed。若基线已乱码，先处理原生全模型环境和配置，不能将后续乱码归因于 PD。

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2/examples/disaggregated_prefill_v1/kvaf_two_node
./d_baseline.sh
```

以下客户端命令需把 D_NODE_IP 替换为实际地址。在可以访问 D 的客户端准备唯一请求文件：

```bash
cat > /tmp/accuracy-request.json <<'JSON'
{
  "model": "glm",
  "messages": [{"role": "user", "content": "请用一句话回答：中国的首都是哪座城市？"}],
  "temperature": 0,
  "top_p": 1,
  "seed": 1024,
  "max_tokens": 128,
  "stream": false
}
JSON

curl --fail --noproxy '*' http://D_NODE_IP:8200/health
curl --fail --noproxy '*' http://D_NODE_IP:8200/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @/tmp/accuracy-request.json \
  -o /tmp/baseline-response.json
```

保存响应与基线日志，再用 Ctrl+C 正常停止基线并确认卡已释放。不要同时运行基线和 D PD；不要使用 Ctrl+Z。单问首都只是部署检查，不代表通用准确率通过。

## 6. 启动 K、D、P、Proxy

D 终端一，修改下面的 D_NODE_IP：

```bash
bash /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/kvaf_pd_k.sh start \
  /home/c00612283/KVAF-Devlop/build-hostcpu-k D_NODE_IP --port 28997
```

首次路径验收，在 P/D 两份 PD 模板中同时改 diagnostics_requests=2、verify_kv=true；默认仍为 0/false。这会同步、哈希、回读，速度明显降低；本轮不测吞吐。超时字段默认 600 秒用于全模型诊断，不代表能解决实际链路故障。

D 终端二：

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2/examples/disaggregated_prefill_v1/kvaf_two_node
./d_start.sh
```

P 终端一：

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2/examples/disaggregated_prefill_v1/kvaf_two_node
./p_start.sh
```

等待两端健康与布局注册完成。完整模型逐层全建链可能较慢；诊断开启时，P 的八个 worker 各向八个 D worker 注册，预期 64 条 REGISTERED。无注册异常并不等价于业务 KV 已传输。

在 P 的另一个终端启动 Proxy；将两个 IP 占位替换为现场地址：

```bash
curl --fail --noproxy '*' http://P_NODE_IP:8100/health
curl --fail --noproxy '*' http://D_NODE_IP:8200/health

KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3 \
bash /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/kvaf_pd_proxy.sh \
  --host P_NODE_IP --port 8000 \
  --prefiller-hosts P_NODE_IP --prefiller-ports 8100 \
  --decoder-hosts D_NODE_IP --decoder-ports 8200
```

## 7. 同请求对照与长请求

客户端复用先前原样保存的 accuracy-request.json，向 P 的 Proxy 发 POST：

```bash
curl --fail --noproxy '*' http://P_NODE_IP:8000/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @/tmp/accuracy-request.json \
  -o /tmp/pd-response.json

python3 - <<'PY'
import json
for name in ('baseline', 'pd'):
    with open(f'/tmp/{name}-response.json') as source:
        r = json.load(source)
    choice = r['choices'][0]
    print(name, 'usage=', r.get('usage'), 'finish_reason=', choice.get('finish_reason'))
    print(json.dumps(choice['message'], ensure_ascii=False))
PY
```

通过 Proxy 8000 才是完整 PD 请求；直接向 PD consumer 的 8200 发普通 POST 不能替代 PD 路径验证。/v1/models 是查询接口，不触发推理。

至少准备短请求和超过 128 实际 tokens 的长请求，对两种模式发送相同 JSON。长请求要覆盖满 block 与非零 tail，输入加 max_tokens 不超过两端 max-model-len。用响应 usage 与实际 DECODE_READY.tokens 核对，不能把汉字数当 token 数。最初两条采样请求应分别用于短/长覆盖。

若要复用此前 41510 tokens 的长请求，应先在基线、P、D 三份脚本统一改 max-model-len=65536，重新评估 HBM 与 DDR；D ddr_size_gb 可先改为 8（八 worker 共约 64 GiB 预算），按实际层数计算是否充足。保存这组新配置的独立基线，不能拿 8192 配置下另一条请求的结果作对照。

## 8. 日志与准确度判定

模板日志位于各自启动工作目录 pd-logs/p、pd-logs/d、pd-logs/baseline 下。P 与 D 在不同机器，需分别收集后放到同一台审计机；不要在一台上 find 出旧单机日志误当新一轮结果。使用本次脚本启动时打印的精确 server.log 路径。

```bash
python3 /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/check_kvaf_pd_diagnostics.py \
  /tmp/two-node-p.log /tmp/two-node-d.log --require-ddr --require-compute
```

其中两个 /tmp 文件由现场从两机本轮 server.log 复制得到，不会自动生成。核对：

- P/D 完整权重、tokenizer、实际参数与软件 SHA 一致；原生基线能正常回答。
- P/D 业务 POST 都出现；请求身份对应；满 block 主 KV 的 CPU Pull/Commit/回读校验 PASS，indexer/scale 与 tail 的 raw READ 校验 PASS，所有目标层贡献齐全。
- tail → ABC、末 token 重算、SFA、MTP 有对应证据；发生任何 FAIL 都不能用 HTTP 200 覆盖。
- 已知审计缺口仍需修复：部分计算事件未携带 generation，调用返回不代表设备完成，DDR_REGION 的旧字段读取可能打印 null。因此原始日志、请求响应和实际设备错误日志必须保留，不能单靠审计 PASS 宣称完整准确度通过。
- 同一批题目比较有效回答、任务得分、重复/乱码、finish_reason、首个分歧位置；量化并行与硬件可能产生数值差异，不强制所有自然语言输出逐字相同。需要严格定位时用相同输入做逐 token/logit 对照。

一次首都问答只能检查连通与明显乱码。正式准确度建议固定 20～50 条起步样本，覆盖知识、算术、长上下文检索及 block 边界，再扩展到现有标准评测集；记录基线与 PD 的得分差及预先约定的容差。诊断采样最多 16 个请求代次，批量评分时关闭诊断并保留错误日志；性能测试另行进行。

## 9. 再验证 P8 → D1/DP8

完成上述同 TP 对照后，在 D 的 d_start.sh 和 d_baseline.sh 同时改：

```bash
--tensor-parallel-size 1 \
--data-parallel-size 8 --data-parallel-size-local 8
```

其余模型、量化、MTP、上下文、seed 保持一致，仍使用全部八张卡。P 保持 TP8/DP1；decode_endpoints 仍为八个端口，端口公式 kv_port + dp_rank*tp_size + tp_rank 在两种拓扑均覆盖 31000～31007。EP 仍为每个引擎 EP8。

先重跑新的 D TP1/DP8 非 PD 基线，再停止并启动新 D PD，重启 P/Proxy 重新注册，重复相同请求与评测。TP1 下稠密权重和部分工作空间复制方式不同，需要重新验证 HBM，不能据 TP8 能加载就保证 TP1 一定可加载。若 TP1 超容量，可以先以 D TP2/DP4 做中间验证，但它不代表 TP1 或原始 EP16 已通过。

本教程模板仅完成静态参数与 Shell 语法校验，云环境无 NPU/跨机 UB，未执行全模型加载、跨机传输或准确度实测。
