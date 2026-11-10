# KVAF PD 分离部署操作指南

本文替代此前分散的安装、单机四层、双机、TP0、多 MR、初始化和诊断操作说明。
配置含义见 [PD 配置说明](kvaf_pd_configuration.md)。命令中的路径、IP 和网卡按实际环境修改。

## 1. 验收拓扑与版本

| 角色 | 执行位置 | 本次验收配置 | 地址 |
| --- | --- | --- | --- |
| P | 12 号容器 | TP8 / DP1 / EP8，8 张 950DT | 141.62.38.12:8100 |
| D | 24 号容器 | TP1 / DP8 / EP8，8 张 950DT | 141.62.38.24:8200 |
| K | 24 号宿主机 | KVAF CPU DDR 服务 | 141.62.38.24:28997 |
| 初始化协调器 | 24 号宿主机 | 同一 K endpoint 串行初始化 | 141.62.38.24:28998 |
| Proxy | 12 号容器 | 原生 layerwise PD Proxy | 141.62.38.12:8000 |

- 权重：`/home/models/GLM-5.3-w4a4c8-mxfp4`；P/D 使用同一模型、tokenizer 和量化配置。
- 框架：`/home/c00612283/vllm-ascend-v0.30.0rc2`，使用本指南所在的合并提交。
- SDK：`/home/c00612283/KVAF-Devlop`，分支 `codex/cpu-pull-hbm-reference`，
  适配基准 `74c3fc491a257bae73356da54eafb08e1e9e4e45`。
- P/D SDK build：`/home/c00612283/KVAF-Devlop/build-pd-py312`。
- 容器 Python：`/usr/local/python3.12.13/bin/python3`；CANN：`/usr/local/Ascend/cann-9.2.0`。
- 容器已有匹配的 vLLM 0.30.0、torch、torch_npu；不要为本部署替换已有 vLLM。
- 用户已反馈本次双机推理和精度测试通过；这不代表所有模型、并发量和图模式均已验收。

P、D 框架版本必须一致；P/D binding 与 K 必须来自同一 SDK 版本。
新 SDK 使用多 MR Pull，不能混用旧 CPU/NPU 协议或旧 ticket。
本分支不包含 develop2 的跨层 arena，不能配置 `source_arena_layers`。

网络需允许 P/Proxy 到 D 的 8200、31000～31007，以及 SDK receiver ticket 中的动态 HBM
监听端口；D 到 K 的 28997、协调器的 28998 必须可达。
模型计算仍需正常的 HCCL 设备通信；HTTP 可达不能代替 UB 数据面验证。

### 原始全建链的配置选择

下面的 P/D 启动示例保留已验收的 `source_tp0_only=true` 临时 TP0 单源配置。
按原始全建链需求部署时，P、D 两侧都删除此字段，或同时设为 false；代码默认值为 false。
所有 P TP 恢复参与向全部 D 的多 MR 注册与原传输分工，P 的完整 decode_endpoints 清单仍须保留。
不要只修改一侧，也不要把关闭 TP0-only 当作开启 DSA CP 的兼容性证明。
切换前排空并停止 P/D，更新两侧配置后按 D → P → Proxy 启动；
重新验证通信资源占用和精度，详见 [两种模式及切换说明](kvaf_pd_configuration.md#原始全建链与临时-tp0-单源模式)。

## 2. 更新前保存现场

停止新请求和 Proxy，排空并停止 P，再停止 D。确认旧初始化和 DMA 已停止后才更换 SDK/K。
保存 `git status --short`、本地启动脚本、请求/响应和 P/D/K 日志。
不要使用 `Ctrl+Z` 停服务；暂停进程仍会占用端口、HBM 或初始化许可。

在 P/D 源码目录检查后更新；本地分叉或修改需先备份，不能直接 reset 或覆盖：

```bash
git status --short
git branch --show-current
git log -1 --oneline
```

本次历史已压缩，旧部署分支与新分支不能再用 `git pull --ff-only` 直接合并。
保存本地修改后，给旧 HEAD 建立备份分支，再采用远端新提交；不要把旧历史 merge 回新分支。

## 3. P/D 容器编译与检查

以下操作在两侧各执行一次。CANN 脚本不一定兼容 nounset，加载时暂时关闭 `set -u`。

```bash
(
set -Eeuo pipefail
trap 'rc=$?; printf "失败：行=%s 命令=%s 退出码=%s\n" "$LINENO" "$BASH_COMMAND" "$rc" >&2' ERR
cd /home/c00612283/vllm-ascend-v0.30.0rc2
export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
export SOC_VERSION=ascend950dt_9582
set +u
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -u
test -x "$KVAF_PD_PYTHON"
"$KVAF_PD_PYTHON" -c 'import sys, torch, torch_npu, vllm; print(sys.executable, torch.__version__, torch_npu.__version__, vllm.__version__)'
command -v cmake
command -v ninja
command -v g++
)
```

首次安装，或 Ascend 扩展、KVOP 尚未成功构建时，在同一匹配 CANN 环境执行：

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2
export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
export SOC_VERSION=ascend950dt_9582
export COMPILE_CUSTOM_KERNELS=1 MAX_JOBS=8
bash scripts/install_kvaf_pd_source.sh - \
  /home/c00612283/KVAF-Devlop \
  /home/c00612283/KVAF-Devlop/build-pd-py312 \
  2>&1 | tee /home/c00612283/pd-source-build.log
)
```

`-` 表示保留已有 vLLM，仍构建 Ascend 原生扩展、KVOP 和 KVAF。
若扩展/KVOP 已存在，仅更新本次 SDK 接口适配，在已加载 CANN 的终端执行：

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2
export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
bash scripts/install_kvaf_pd.sh \
  /home/c00612283/KVAF-Devlop \
  /home/c00612283/KVAF-Devlop/build-pd-py312
)
```

安装脚本保留已有的 HCCL device RA 借用补丁并校验接口，不下载或切换 SDK 分支。
它会修改匹配 SDK 中的补丁目标文件；这不是本次新增的 SDK 修改。
框架采用 editable 安装时，纯 Python 更新无需重新编译 vLLM、Ascend 扩展或 KVOP。

检查原生扩展及 SDK，不能只以 binding 链接成功作为整体安装成功的依据：

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2
source third_party/KVOP/env.sh
export PYTHONPATH="/home/c00612283/KVAF-Devlop/build-pd-py312/python${PYTHONPATH:+:$PYTHONPATH}"
/usr/local/python3.12.13/bin/python3 - <<'CHECK'
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C as extension
from vllm_ascend.distributed.kv_transfer.kvaf.pd.runtime import load_sdk
sdk = load_sdk('/home/c00612283/KVAF-Devlop/build-pd-py312')
print('ASCEND_EXTENSION_OK', extension.__file__)
print('PD_SDK_IMPORT_OK', sdk.__file__)
CHECK
)
```

## 4. K 编译与后台启动：24 号宿主机

宿主机需有同版本 SDK 源码、UMDK 头文件及 liburma；不是在 P/D 容器内运行 K。

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2
test -f /usr/include/ub/umdk/urma/urma_api.h
test -f /usr/lib64/liburma.so
bash scripts/kvaf_pd_k.sh build \
  /home/c00612283/KVAF-Devlop \
  /home/c00612283/KVAF-Devlop/build-hostcpu-k \
  -DKVAF_URMA_INCLUDE_DIR=/usr/include/ub/umdk/urma \
  -DKVAF_URMA_LIBRARY=/usr/lib64/liburma.so \
  '-DCMAKE_CXX_FLAGS=-DURMA_ACCESS_REMOTE_READ=URMA_ACCESS_READ -DURMA_ACCESS_REMOTE_WRITE=URMA_ACCESS_WRITE -DURMA_ACCESS_REMOTE_ATOMIC=URMA_ACCESS_ATOMIC'
)
```

先确认旧监听者已退出，再启动 K；直接写文件，避免 tee 被暂停后阻塞 K 的日志输出：

```bash
K_LOG="/home/c00612283/kvaf-k-$(date +%Y%m%d_%H%M%S).log"
nohup /home/c00612283/KVAF-Devlop/build-hostcpu-k/runtime/kvaf_cpu_service \
  --listen 141.62.38.24 --port 28997 --memory-page-mode default \
  > "$K_LOG" 2>&1 < /dev/null &
echo "K_PID=$! K_LOG=$K_LOG"
```

协调器只依赖 Python 标准库，用宿主机实际 Python，不使用容器内解释器路径：

```bash
COORD_LOG="/home/c00612283/kvaf-coordinator-$(date +%Y%m%d_%H%M%S).log"
nohup "$(command -v python3)" -u \
  /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/kvaf_pd_init_coordinator.py \
  --host 141.62.38.24 --port 28998 \
  > "$COORD_LOG" 2>&1 < /dev/null &
echo "COORD_PID=$! COORD_LOG=$COORD_LOG"
```

```bash
netstat -lntp 2>/dev/null | grep -E ':28997[[:space:]]|:28998[[:space:]]'
tail -n 10 "$K_LOG"
tail -n 10 "$COORD_LOG"
```

必须看到 K 的 `LISTEN` 和协调器的 `KVAF_INIT_COORDINATOR_LISTEN`，不能只看后台 PID。
已有健康服务可复用，不重复启动。同一部署只运行一个初始化协调器。

## 5. 启动 D：24 号容器

以下为用户在 2026-11-10 确认可跑通的 D 配置，直接调用原生 CLI，不经过会默认添加
`--enforce-eager` 的启动 helper。保留异步调度、FULL_DECODE_ONLY、MLAPO 和 profiling 配置。
这条命令在当前终端前台运行，另开终端检查就绪；不要使用 Ctrl+Z 暂停服务或 tee。
先在实际启动 shell 检查 `ulimit -Sn`、`ulimit -Hn`，硬限制不足时先调整该 shell 或容器配置。
profiling 目录 `/home/d00843668/profiling_no_pooling` 须存在或可由运行用户创建、写入。

```bash
(
set -euo pipefail

cd /home/c00612283/vllm-ascend-v0.30.0rc2
unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY
unset http_proxy HTTP_PROXY all_proxy ALL_PROXY
source third_party/KVOP/env.sh
export PYTHONPATH="/home/c00612283/KVAF-Devlop/build-pd-py312/python${PYTHONPATH:+:$PYTHONPATH}"

local_ip=141.62.38.24
nic_name=enp34s0f1
export VLLM_HOST_IP="$local_ip"
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export HCCL_DFS_CONFIG="task_exception:off,inconsistent_check:off"
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_BUFFSIZE=400
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export TORCH_HCCL_ASYNC_ERROR_HANDLING=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

dir="$PWD/pd-logs/d/$(date +%Y%m%d_%H%M%S)_$$"
export ASCEND_PROCESS_LOG_PATH="$dir/plog"
export PROMETHEUS_MULTIPROC_DIR="/dev/shm/vllm_metrics_d_$$"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$PROMETHEUS_MULTIPROC_DIR"
printf 'D server log: %s/server.log\n' "$dir"
exec > >(tee "$dir/server.log") 2>&1

exec /usr/local/python3.12.13/bin/python3 \
  -m vllm.entrypoints.cli.main serve \
  /home/models/GLM-5.3-w4a4c8-mxfp4 \
  --host 0.0.0.0 \
  --port 8200 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --data-parallel-size-local 8 \
  --data-parallel-address "$local_ip" \
  --data-parallel-rpc-port 13489 \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --served-model-name glm \
  --seed 1024 \
  --max-model-len 131072 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --quantization ascend \
  --no-enable-prefix-caching \
  --worker-cls vllm_ascend.distributed.kv_transfer.kvaf.pd.worker.PDKVWorker \
  --block-size 128 \
  --kv-cache-dtype fp8 \
  --attention_config.indexer_kv_dtype fp8 \
  --async-scheduling \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/d00843668/profiling_no_pooling","torch_profiler_with_stack":false}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{
    "num_speculative_tokens": 3,
    "method": "deepseek_mtp",
    "enforce_eager": true
  }' \
  --kv-transfer-config '{
    "kv_connector": "PDKVoffloadConnector",
    "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kvaf.pd.connector",
    "kv_role": "kv_consumer",
    "kv_port": 31000,
    "kv_connector_extra_config": {
      "timeout": 600,
      "initialization_coordinator": "141.62.38.24:28998",
      "source_tp0_only": true,
      "diagnostics_requests": 1,
      "verify_kv": true
    }
  }' \
  --additional-config '{
    "enable_cpu_binding": true,
    "multistream_overlap_shared_expert": true,
    "ascend_compilation_config": {
      "enable_npugraph_ex": true,
      "enable_static_kernel": false,
      "enable_super_kernel": false
    },
    "recompute_scheduler_enable": "True",
    "finegrained_tp_config": {"lmhead_tensor_parallel_size": 8},
    "enable_mlapo": true,
    "c8_enable_reshape_optim": false,
    "kvaf_config": {
      "npu_core_count": 1,
      "jettys_per_core": 1,
      "enabled": true,
      "build_dir": "/home/c00612283/KVAF-Devlop/build-pd-py312",
      "cpu_address": "141.62.38.24:28997",
      "ddr_size_gb": 16
    }
  }'
)
```

D 主模型不传 `--enforce-eager`；MTP 子配置中的 `enforce_eager=true` 仅作用于 draft，
不能据此将 D 主模型解释为 eager。未显式配置的 `enable_dsa_cp` 使用当前默认 false。
`recompute_scheduler_enable` 保留用户成功命令中的字符串 "True" 和兼容字段位置，
不在本文档更新中转换为其他类型或配置结构。

等待初始化、warmup 和控制监听全部完成：

```bash
curl --fail --noproxy '*' --max-time 10 -i http://141.62.38.24:8200/health
netstat -lntp 2>/dev/null | grep -E ':3100[0-7][[:space:]]'
```

应为 HTTP 200 且八个控制端口均监听。`/health` 响应体为空正常。

## 6. 启动 P：12 号容器

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2
export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=10
LOG_DIR="$PWD/pd-logs/p/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LOG_DIR"
export ASCEND_PROCESS_LOG_PATH="$LOG_DIR/plog"
export PROMETHEUS_MULTIPROC_DIR="/dev/shm/vllm_metrics_p_$$"
nohup bash scripts/start_vllm_pd.sh p 141.62.38.12 enp34s0f1 -- \
  /home/models/GLM-5.3-w4a4c8-mxfp4 \
  --port 8100 --tensor-parallel-size 8 \
  --data-parallel-size 1 --data-parallel-size-local 1 \
  --data-parallel-address 141.62.38.12 --data-parallel-rpc-port 13389 \
  --pipeline-parallel-size 1 --max-model-len 131072 \
  --max-num-batched-tokens 4096 --max-num-seqs 1 --gpu-memory-utilization 0.85 \
  --kv-transfer-config '{
    "kv_connector": "PDKVoffloadConnector",
    "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kvaf.pd.connector",
    "kv_role": "kv_producer",
    "kv_port": 31000,
    "kv_connector_extra_config": {
      "decode_endpoints": [
        "141.62.38.24:31000", "141.62.38.24:31001",
        "141.62.38.24:31002", "141.62.38.24:31003",
        "141.62.38.24:31004", "141.62.38.24:31005",
        "141.62.38.24:31006", "141.62.38.24:31007"
      ],
      "source_tp0_only": true,
      "timeout": 600,
      "diagnostics_requests": 1,
      "verify_kv": true
    }
  }' \
  --additional-config '{
    "enable_cpu_binding": true,
    "enable_dsa_cp": false,
    "multistream_overlap_shared_expert": true,
    "c8_enable_reshape_optim": false,
    "kvaf_config": {
      "enabled": false,
      "build_dir": "/home/c00612283/KVAF-Devlop/build-pd-py312"
    }
  }' > "$LOG_DIR/server.log" 2>&1 < /dev/null &
echo "P_PID=$! P_LOG=$LOG_DIR/server.log"
)
```

```bash
curl --fail --noproxy '*' --max-time 10 -i http://141.62.38.12:8100/health
curl --fail --noproxy '*' --max-time 10 -i http://141.62.38.24:8200/health
```

P 的 `enabled=false` 仅关闭原混部卸载入口，不关闭显式选中的 PD Connector。
本节 P 使用 helper 默认的 FP8、block128、MTP3、seed1024、eager、禁用 prefix 和 EP 等参数，
完整默认值见配置说明。P 的 diagnostics_requests=1、verify_kv=true 与上述 D 对齐。
已有自定义启动脚本可以保留，但须逐项对齐这些配置；不要把 D 的图模式参数直接套用到 P。
仓库根目录四层脚本、`examples/.../kvaf_two_node` 的历史模板仍保留兼容用途；
其默认 TP/DP、8192 长度和占位 IP 不等同于本次验收配置，不直接套用于全模型。

## 7. Proxy 与请求：12 号容器

确认 8000 没有旧 Proxy。先核对监听进程参数，再结束明确属于旧部署的进程。

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
PROXY_LOG="/home/c00612283/kvaf-proxy-$(date +%Y%m%d_%H%M%S).log"
KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3 \
nohup bash scripts/kvaf_pd_proxy.sh \
  --host 141.62.38.12 --port 8000 \
  --prefiller-hosts 141.62.38.12 --prefiller-ports 8100 \
  --decoder-hosts 141.62.38.24 --decoder-ports 8200 \
  > "$PROXY_LOG" 2>&1 < /dev/null &
echo "PROXY_PID=$! PROXY_LOG=$PROXY_LOG"
```

```bash
curl --noproxy '*' --connect-timeout 3 --max-time 180 \
  http://141.62.38.12:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm","messages":[{"role":"user","content":"请回复：测试成功"}],"temperature":0,"max_tokens":32,"stream":false}' \
  -w '\nHTTP_CODE=%{http_code} TOTAL_TIME=%{time_total}s\n'
```

响应应包含 choices 和 usage。流式请求的访问日志 200 可能只表示响应头已发送；
以客户端完整收到内容为准。`finish_reason=length` 表示达到输出 token 上限。
短请求不覆盖满块 DDR 路径，继续使用已保存的 41510-token 请求做长请求回归，
输入 token 加输出上限必须小于等于 131072，不能重新拼写 prompt 后假定仍是同一用例。

## 8. 诊断与精度验收

上述 P/D 示例已经配置 `diagnostics_requests=1, verify_kv=true`，重启 P/D 后
发送一份固定采样长请求。完成后恢复 `0/false`，校验模式不用于性能测量。
使用本轮明确的日志路径，不把不同启动代的日志混合；P/D 分机时先把本次日志复制到同一机器。

```bash
python3 scripts/check_kvaf_pd_diagnostics.py \
  /home/c00612283/pd-diag-p.log /home/c00612283/pd-diag-d.log \
  --require-ddr --require-compute
```

这两个文件必须先从本次 P/D server.log 提取 `KVAF_PD_DIAG` 行，或直接传入实际日志路径。
预期 `PD_DIAGNOSTICS_PASS`，事件含 CPU/NPU `PULL_DONE`、`KV_CHECK=PASS`、
`DECODE_READY` 及原生计算完成证据。K 的 `REMOTE_PULL_DONE` 仅能证明对应 CPU Pull 完成，
单独不能证明 NPU 路径、解码消费或模型精度。

精度用同一模型/权重、tokenizer、完整请求、采样参数、MTP 设置和尽可能一致的计算拓扑做非 PD 基线，
再比较 PD 响应的 token IDs（可获得时）或原始输出文本、finish_reason、usage。
忽略 id/created 等请求元数据。每种模式重复请求确认自身稳定性，保存首个分歧位置。
TP 拓扑不同可能引入数值差异，不能仅凭文本差异断定传输错误。
四层模型可以比较两种路径的一致性，但其自然语言输出不代表全模型能力。

覆盖短请求、128 token 边界、长输入分块、MTP 正常/跳过、取消及多 DP 路由。
本次用户反馈精度通过；新设备或新 SDK 版本仍需重做相同回归。

## 9. 失败定位与恢复

| 现象 | 处理 |
| --- | --- |
| Could not resolve hostname / 证书链错误 | 先验证安装网络、代理和可信 CA；不是 C++ 编译错误 |
| makeself / abseil / protobuf 下载失败 | 看首个 CMake 错误；下载完整包并核对仓库固定哈希，不能用自行计算的哈希冒充可信来源 |
| patch 询问 File to patch | 停止交互，核对依赖版本、解包目录和补丁工作目录，不反复套补丁 |
| 缺少 `_build_info` 或 `vllm_ascend_C` | 完整安装对应 Ascend 原生扩展，再验证 import；KVAF binding 成功不能替代它 |
| Build KVOP first | 在同一容器、Python/CANN 环境运行 third_party/KVOP/build.sh |
| Available KV cache memory: 0.00 GiB | 查实际 NPU 占用、遗留进程及 profiling 预算；不是物理 HBM 必然为零，不能盲目提高利用率 |
| shared memory broadcast 等待 | 找 worker 的首个异常或长耗时操作，不能据此直接认定共享内存不足 |
| CPU control response unavailable | 同时检查 K 日志、服务状态、初始化并发和协议版本；TCP 可连不等于协议处理正常 |
| previous initializer failed/disconnected | 协调器封锁了该 K endpoint，按下面顺序排空恢复 |
| Too many open files | 检查实际 worker `/proc/PID/limits` 和 fd 数；新 shell 不一定继承旧 shell 的提升 |
| source CTP query failed / create jetty 失败 | 记录源组数和底层错误；FD 高不必然触及 nofile，更不能仅凭记录数推断 Jetty 硬件上限 |
| Address already in use | 查 netstat 和 ps；STAT=T 表示暂停，Z 表示待父进程回收，不能直接重复起同端口 |
| Cannot assign requested address | listen IP 必须属于当前宿主机/网络命名空间 |

安装用可信 CA 的示例：在宿主机将 CA 内容经 `docker exec -i 容器名 sh -c 'cat > /tmp/host-ca-bundle.pem'`
写入容器，比较两侧 SHA256 后使用 `curl --cacert`。CMake/Python/pip 在安装终端分别配置
`SSL_CERT_FILE`、`CURL_CA_BUNDLE`、`REQUESTS_CA_BUNDLE`、`PIP_CERT` 指向该文件。
不要持久关闭系统 TLS 校验。CMake 未固定 URL_HASH 时，即使 pkg 中有压缩包也可能重新下载。

协调器失败恢复顺序：停止新请求及 Proxy，停止并确认所有相关 P/D Worker 已静止，保存最早异常；
确认 K 健康，状态不明时在所有客户端停止后重启 K；最后重启唯一协调器，再 D → P → Proxy。
协调器封锁状态在内存中，不会因 D 重启或延长 timeout 自动清除。K 重启使旧 ticket 全部失效，
不能继续使用原 D/P 实例。不要在旧 initializer 存活时重启协调器放行新一批初始化。

K/协调器日志直接重定向到文件；本节 D 的前台 tee 需保持运行。`STAT=T` 使用明确 PID 恢复或正常结束，避免暂停 tee
造成 `pipe_write`。只清理确认属于本次部署的进程，不使用全局 pkill python 或盲目重置设备。

## 10. 辅助工具与归档

`scripts/kvaf_pd_ddr_residency.py --help` 提供无模型串行/并行 DDR 驻留测试；
它检验初始化与共驻留，不证明 KV 字节或模型精度。
`scripts/kvaf_pd_smoke.py --help` 提供固定文本的 completions 连通测试。
它们保留用于回归，不参与正常请求热路径。

归档本次框架/SDK 提交号、实际构建路径、P/D/K/协调器命令、请求、基线响应、PD 响应和对应日志。
升级后复用同一回归用例；不要将未执行的模式写成已通过。

四智算节点与八鲲鹏节点的扩容模板及脚本见 [池化部署指南](kvaf_pd_pooled_deployment.md)。
