# KVAF 1P1D 双机部署指南（P 八卡 / D 八卡）

本文将用户实际验证过的编译命令、P/D 启动配置、KVAF CPU 服务、初始化协调器和 Proxy 命令汇总到一个文件。
P 使用 TP8/DP1，D 使用 TP1/DP8；K 数据服务和协调器运行在 D 所在的 24 号宿主机，不另需第三台服务器。
所有需要保存的启动脚本均在本文完整给出；安装和服务入口使用仓库已有程序，不依赖其他部署说明中的启动模板。

地址按用户确认统一：P/Proxy 为 `141.62.38.12`，D/K/协调器为 `141.62.38.24`。
原补充命令中的 K `--listen 141.62.38.12`、Proxy 的 `.16` 已据此统一，避免与 D/P 配置不一致。
编译内容去除重复粘贴并补齐末尾 `)`；D 的共享专家开关重复键只保留最终生效的布尔值 `true`。
其余 P/D 运行参数沿用用户提供的实测配置，包括 `recompute_scheduler_enable` 的字符串 `"True"`。

## 1. 拓扑、目录与版本

| 角色 | 执行位置 | 配置 | 服务地址 |
| --- | --- | --- | --- |
| P | 12 号服务器容器 | 8 卡，TP8/DP1/EP8，eager | `141.62.38.12:8100` |
| D | 24 号服务器容器 | 8 卡，TP1/DP8/EP8，Target Decode 图模式 | `141.62.38.24:8200` |
| K 数据服务 | 24 号服务器宿主机 | CPU DDR 服务，8 个 D worker 共用 endpoint | `141.62.38.24:28997` |
| 初始化协调器 | 24 号服务器宿主机 | 单实例，串行化同一 K endpoint 的 D 初始化 | `141.62.38.24:28998` |
| Proxy | 12 号服务器容器 | P HTTP → D HTTP | `141.62.38.12:8000` |
| D 控制服务 | 24 号服务器容器 | 每个 DP worker 一个端口 | `141.62.38.24:31000`～`31007` |

P 和 D 的 EP 组各自包含本机 8 卡，不组成跨 P/D 的 EP16。
P/D 均使用物理 NPU `0,1,2,3,4,5,6,7`，网卡为 `enp34s0f1`，PP=1。

| 项目 | 固定路径 / 版本 |
| --- | --- |
| 框架源码 | `/home/c00612283/vllm-ascend-v0.30.0rc2`，`v0.30.0rc2-develop` 分支 |
| KVAF SDK 源码 | `/home/c00612283/KVAF-Devlop`，`codex/cpu-pull-hbm-reference` 分支 |
| SDK 适配基准 | `74c3fc491a257bae73356da54eafb08e1e9e4e45` |
| P/D SDK 构建目录 | `/home/c00612283/KVAF-Devlop/build-pd-py312` |
| K CPU 构建目录 | `/home/c00612283/KVAF-Devlop/build-hostcpu-k` |
| 模型 | `/home/models/GLM-5.3-w4a4c8-mxfp4` |
| 容器 Python | `/usr/local/python3.12.13/bin/python3` |
| CANN / SoC | `/usr/local/Ascend/cann-9.2.0` / `ascend950dt_9582` |

P/D 使用相同框架提交、模型、tokenizer、量化配置及配套 torch/torch_npu/CANN/ATB 环境。
P/D Python binding 与 K 服务必须由同一 KVAF SDK 提交构建；不能混用旧 Pull 协议或旧 ticket。
本文按框架 `3927ee424` 所含的 PD 入口整理，不使用 develop2 的跨层 arena 参数。

### 1.1 容量与配置边界

- D 的 `ddr_size_gb=16` 是每个 worker 的预算，8 个 worker 合计约 128 GiB，宿主机还需预留 SDK、系统及其他进程内存。
- `max-num-seqs=1` 是每个 D DP 引擎的调度上限，不是把单卡 batch 设置为 8 或 16。
- `max-model-len=131072` 限制输入与输出总长度；长请求测试必须为输出预留 token。
- P/D 均设置 `source_tp0_only=true`：每个 P DP 实例由逻辑 TP0 注册并提供所需源 KV；P 计算仍然是 TP8。
- P 的 `kvaf_config.enabled=false` 不会关闭 PD producer；P 仍通过 PDKVoffloadConnector 向 D 注册源。
- P 保留 `--enforce-eager`；D 不传该参数，使用 `FULL_DECODE_ONLY` 和 `enable_npugraph_ex=true`。
- MTP 的 `enforce_eager=true` 只控制 Draft；D Target Decode 仍按图模式运行。保留已验证的 LMHead TP8、MLAPO 和异步调度。
- 两端 `diagnostics_requests=0`、`verify_kv=false`；本基线不增加 profiler 配置。

### 1.2 启动前检查

在实际运行服务的终端检查本机 IP、网卡、NPU 和文件描述符限制：

```bash
ip -br addr
ip link show enp34s0f1
npu-smi info
ulimit -Sn
ulimit -Hn
```

在 P/D 容器各检查源码与 SDK 版本，保存输出：

```bash
git -C /home/c00612283/vllm-ascend-v0.30.0rc2 status --short
git -C /home/c00612283/vllm-ascend-v0.30.0rc2 log -1 --oneline
git -C /home/c00612283/KVAF-Devlop status --short
git -C /home/c00612283/KVAF-Devlop log -1 --oneline
```

K 宿主机也检查 SDK 提交，必须与 P/D 一致。已有源码及依赖按当前成功部署保留，不覆盖本地修改。
源码尚未准备时，在需要源码的环境执行以下克隆命令；目录已存在时不要重复克隆或强制重置：

```bash
cd /home/c00612283
git clone --branch v0.30.0rc2-develop \
  https://github.com/TX-POC/vllm-ascend-v0.30.0rc2.git \
  vllm-ascend-v0.30.0rc2
git clone --branch codex/cpu-pull-hbm-reference \
  https://github.com/TX-POC/KVAF-Devlop.git KVAF-Devlop
```

分支头可能前进，克隆后核对版本再编译；同一轮部署保存并统一实际使用的提交。
K 运行协调器仅需框架脚本文件，无需在宿主机 pip 安装或编译 vLLM、vLLM-Ascend、KVOP。
K 仍需单独编译下一节所述的 KVAF CPU 服务。

P/D/Proxy 使用容器中可达的业务 IP。P 到全部 D 控制端口和 SDK ticket 中的动态 HBM 端口、
P/D 到 K `28997`、D 到协调器 `28998`、Proxy 到 P `8100` 与 D `8200` 必须可达；
同时保持原生 HCCL/UB 数据面通信正常。HTTP 或 TCP 连通不能替代远程内存探测。

## 2. 编译安装：P/D 容器各执行一次

以下是用户提供的实际编译脚本，去除重复粘贴并补齐子 shell。
`-` 表示保留现有 vLLM，仍会构建安装 vLLM-Ascend、KVOP 和 KVAF；不要替换已配套的 vLLM、torch 或 torch_npu。
K 宿主机不要执行本节的 NPU 编译。

```bash
(
set -euo pipefail

export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
export ASCEND_REPO=/home/c00612283/vllm-ascend-v0.30.0rc2
export KVAF_PD_SRC=/home/c00612283/KVAF-Devlop
export KVAF_PD_BUILD=/home/c00612283/KVAF-Devlop/build-pd-py312
export MODEL=/home/models/GLM-5.3-w4a4c8-mxfp4
export SOC_VERSION=ascend950dt_9582

CANN_ROOT=/usr/local/Ascend/cann-9.2.0

# 加载本次日志中使用的 CANN 环境。
set +u
if [ -f "$CANN_ROOT/set_env.sh" ]; then
    source "$CANN_ROOT/set_env.sh"
elif [ -f "$CANN_ROOT/bin/setenv.bash" ]; then
    source "$CANN_ROOT/bin/setenv.bash"
else
    echo "未找到 CANN 环境脚本：$CANN_ROOT"
    exit 1
fi
set -u

cd "$ASCEND_REPO"
# 检查解释器、源码和依赖。
test -x "$KVAF_PD_PYTHON"
test -f "$KVAF_PD_SRC/CMakeLists.txt"
test -f "$KVAF_PD_SRC/docs/connector_pull_api_design.md"

"$KVAF_PD_PYTHON" -c \
'import sys, torch, torch_npu, vllm; print(sys.executable); print(torch.__version__, torch_npu.__version__); print(vllm.__version__, vllm.__file__)'
command -v cmake
command -v ninja
command -v g++

# “-”表示保留现有 vLLM，仍编译安装 vLLM-Ascend。
if bash scripts/install_kvaf_pd_source.sh - \
    "$KVAF_PD_SRC" "$KVAF_PD_BUILD" \
    2>&1 | tee /home/c00612283/full-model-pd-build.log
then
    echo "ASCEND_KVOP_KVAF_BUILD_OK"
else
    echo "构建失败，提取具体错误："
    grep -n -B 15 -A 40 -m 1 \
      -E 'CMake Error|fatal error:|Traceback|ModuleNotFoundError|No such file|not found' \
      /home/c00612283/full-model-pd-build.log \
      || tail -n 150 /home/c00612283/full-model-pd-build.log
    exit 1
fi
)
```

完整安装子 shell 中加载的 CANN 环境不会传回父终端。
启动 P/D 前，在各自运行终端加载同一套 CANN，沿用原有 ATB 环境：

```bash
CANN_ROOT=/usr/local/Ascend/cann-9.2.0
if [ -f "$CANN_ROOT/set_env.sh" ]; then
    source "$CANN_ROOT/set_env.sh"
elif [ -f "$CANN_ROOT/bin/setenv.bash" ]; then
    source "$CANN_ROOT/bin/setenv.bash"
else
    echo "未找到 CANN 环境脚本：$CANN_ROOT"
fi
```

加载失败时先修正环境，再执行启动脚本。脚本会各自加载 KVOP，并优先导入指定 SDK binding。
可在两个容器各做一次最终导入检查：

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

`connector_pull_api_design.md` 缺失表示 SDK 路径或分支需核对，不要创建空文件绕过检查。
安装入口会检查/应用已有 device RA 借用补丁；补丁不匹配时先核对源码版本。

## 3. K CPU 服务：24 号宿主机

### 3.1 编译 K CPU 服务

宿主机需有 CMake、C++ 编译器、匹配的 UMDK 头文件及 liburma。
本节采用当前部署指南中的 CPU-only 构建参数，不使用容器 Python，不构建 NPU kernel 或 Python binding。
已完成同版本构建时可直接进入启动步骤。

```bash
(
set -euo pipefail

SRC=/home/c00612283/KVAF-Devlop
BUILD="$SRC/build-hostcpu-k"

if [ ! -f "$SRC/CMakeLists.txt" ] ||
   [ ! -f "$SRC/docs/connector_pull_api_design.md" ]; then
    echo "KVAF 源码不完整或分支不匹配，请检查 cpu-pull-hbm-reference 分支。"
    exit 1
fi

test -f /usr/include/ub/umdk/urma/urma_api.h
test -f /usr/lib64/liburma.so

cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON \
  -DKVAF_BUILD_PYTHON=OFF \
  -DKVAF_ENABLE_CPU_OFFLOAD=OFF \
  -DKVAF_ENABLE_AIV_BATCH_GET=OFF \
  -DKVAF_ENABLE_AIV_BATCH_PUT=OFF \
  -DBUILD_TESTING=OFF \
  -DKVAF_URMA_INCLUDE_DIR=/usr/include/ub/umdk/urma \
  -DKVAF_URMA_LIBRARY=/usr/lib64/liburma.so \
  '-DCMAKE_CXX_FLAGS=-DURMA_ACCESS_REMOTE_READ=URMA_ACCESS_READ -DURMA_ACCESS_REMOTE_WRITE=URMA_ACCESS_WRITE -DURMA_ACCESS_REMOTE_ATOMIC=URMA_ACCESS_ATOMIC' \
  2>&1 | tee /home/c00612283/kvaf-k-configure.log

cmake --build "$BUILD" --parallel 8 \
  2>&1 | tee /home/c00612283/kvaf-k-build.log

test -x "$BUILD/runtime/kvaf_cpu_service"
ldd "$BUILD/runtime/kvaf_cpu_service"
echo "K_SERVICE_BUILD_OK"
)
```

`KVAF_ENABLE_CPU_OFFLOAD=OFF` 关闭的是 NPU 侧 offload Runtime 构建；远程 CPU 服务由
`KVAF_BUILD_REMOTE_CPU_SERVICE=ON` 启用。`ldd` 不应出现 `not found`。

### 3.2 检查端口并启动 K

在 24 号宿主机检查端口。没有输出表示当前命令未找到监听者，不代表服务已启动：

```bash
netstat -lntp 2>/dev/null | grep ':28997[[:space:]]'
```

确认没有冲突后，在专用终端执行用户提供的前台服务命令；监听地址按本次确认改为 `.24`：

```bash
(
set -euo pipefail

K_BIN=/home/c00612283/KVAF-Devlop/build-hostcpu-k/runtime/kvaf_cpu_service
test -x "$K_BIN"

stdbuf -oL -eL "$K_BIN" \
  --listen 141.62.38.24 \
  --port 28997 \
  --memory-page-mode default \
  2>&1 | tee "/home/c00612283/kvaf-k-$(date +%Y%m%d_%H%M%S).log"
)
```

保留该终端和 tee 正常运行，另开终端确认出现 `[KVAF_CPU] LISTEN address=141.62.38.24 port=28997`。
此处日志直接写入 `/home/c00612283/kvaf-k-时间.log`，不是 `pd-pool-logs`。

## 4. 初始化协调器：24 号宿主机

协调器只使用 Python 标准库；宿主机无需安装 torch、torch_npu 或 vLLM。
按当前源码入口，以下两个文件必须存在且保留相对目录，完整克隆框架仓库也可以：

```text
/home/c00612283/vllm-ascend-v0.30.0rc2/scripts/kvaf_pd_init_coordinator.py
/home/c00612283/vllm-ascend-v0.30.0rc2/vllm_ascend/distributed/kv_transfer/kvaf/pd/initialization.py
```

先检查 `28998` 是否已有健康协调器；只启动一个实例。以下保留用户提供的后台启动方式：

```bash
(
set -euo pipefail

REPO=/home/c00612283/vllm-ascend-v0.30.0rc2
PYTHON=$(command -v python3)
COORD_LOG="/home/c00612283/kvaf-coordinator-$(date +%Y%m%d_%H%M%S).log"

test -f "$REPO/scripts/kvaf_pd_init_coordinator.py"

nohup "$PYTHON" "$REPO/scripts/kvaf_pd_init_coordinator.py" \
  --host 141.62.38.24 \
  --port 28998 \
  > "$COORD_LOG" 2>&1 < /dev/null &

echo "COORD_PID=$!"
echo "COORD_LOG=$COORD_LOG"
)
```

打印 PID 仅表示已创建后台进程；在打印出的 `COORD_LOG` 中检查
`KVAF_INIT_COORDINATOR_LISTEN address=141.62.38.24:28998`，并核对实际监听进程：

```bash
netstat -lntp 2>/dev/null \
  | grep -E ':28997[[:space:]]|:28998[[:space:]]'
ps -ww -eo pid,ppid,stat,etime,args \
  | grep -E '[k]vaf_cpu_service|[k]vaf_pd_init_coordinator.py'
grep -H 'KVAF_INIT_COORDINATOR_LISTEN' /home/c00612283/kvaf-coordinator-*.log
```

只以本次启动打印的日志和 PID 为准，历史日志中的 LISTEN 不能代替本次就绪检查。
K 和协调器均就绪后再启动 D。

## 5. D 启动：24 号容器，八卡 TP1/DP8

将下面完整内容保存为 `/home/c00612283/d_dp8tp1start.sh`。
D 参数按用户实测脚本保留，重复的 `multistream_overlap_shared_expert` 只留一处 `true`；
`recompute_scheduler_enable` 仍为字符串 `"True"`。

```bash
cat > /home/c00612283/d_dp8tp1start.sh <<'D_SCRIPT'
#!/usr/bin/env bash
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
      "diagnostics_requests": 0,
      "verify_kv": false
    }
  }' \
  --additional-config '{
    "ascend_compilation_config": {
      "enable_npugraph_ex": true,
      "enable_static_kernel": false,
      "enable_super_kernel": false
    },
    "recompute_scheduler_enable": "True",
    "finegrained_tp_config": {"lmhead_tensor_parallel_size": 8},
    "enable_mlapo": true,
    "enable_cpu_binding": true,
    "enable_dsa_cp": false,
    "multistream_overlap_shared_expert": true,
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
D_SCRIPT
bash -n /home/c00612283/d_dp8tp1start.sh
```

在已加载 CANN/ATB 的 24 号容器终端启动：

```bash
bash /home/c00612283/d_dp8tp1start.sh
```

该终端前台运行，日志打印在 `pd-logs/d/本次时间_PID/server.log`，另有 plog。
等待全部 worker 完成初始化及图 warmup，另开终端检查：

```bash
curl --fail --noproxy '*' --connect-timeout 3 --max-time 10 \
  -i http://141.62.38.24:8200/health
netstat -lntp 2>/dev/null | grep -E ':3100[0-7][[:space:]]'
```

HTTP 200 且响应体为空正常；八个 D 控制端口应全部监听。
D 的 `control_backlog` 未在用户脚本显式填写，本分支默认 128，不为本次文档额外补参。

## 6. P 启动：12 号容器，八卡 TP8/DP1

将下面完整内容保存为 `/home/c00612283/dp1_tp8_chuban_start.sh`，保留用户提供的 P 配置，
包括共享专家重叠开关 `true`、主模型 eager 和八个 D endpoint：

```bash
cat > /home/c00612283/dp1_tp8_chuban_start.sh <<'P_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

cd /home/c00612283/vllm-ascend-v0.30.0rc2

# 沿用容器已有 CANN / ATB 环境。
unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY
unset http_proxy HTTP_PROXY all_proxy ALL_PROXY

source third_party/KVOP/env.sh
export PYTHONPATH="/home/c00612283/KVAF-Devlop/build-pd-py312/python${PYTHONPATH:+:$PYTHONPATH}"

local_ip=141.62.38.12
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

dir="$PWD/pd-logs/p/$(date +%Y%m%d_%H%M%S)_$$"
export ASCEND_PROCESS_LOG_PATH="$dir/plog"
export PROMETHEUS_MULTIPROC_DIR="/dev/shm/vllm_metrics_p_$$"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$PROMETHEUS_MULTIPROC_DIR"
printf 'P server log: %s/server.log\n' "$dir"
exec > >(tee "$dir/server.log") 2>&1

exec /usr/local/python3.12.13/bin/python3 \
  -m vllm.entrypoints.cli.main serve \
  /home/models/GLM-5.3-w4a4c8-mxfp4 \
  --host 0.0.0.0 \
  --port 8100 \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --data-parallel-address "$local_ip" \
  --data-parallel-rpc-port 13389 \
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
  --enforce-eager \
  --quantization ascend \
  --no-enable-prefix-caching \
  --worker-cls vllm_ascend.distributed.kv_transfer.kvaf.pd.worker.PDKVWorker \
  --block-size 128 \
  --kv-cache-dtype fp8 \
  --attention_config.indexer_kv_dtype fp8 \
  --speculative-config '{
    "num_speculative_tokens": 3,
    "method": "deepseek_mtp",
    "enforce_eager": true
  }' \
  --kv-transfer-config '{
    "kv_connector": "PDKVoffloadConnector",
    "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kvaf.pd.connector",
    "kv_role": "kv_producer",
    "kv_port": 31000,
    "kv_connector_extra_config": {
      "decode_endpoints": [
        "141.62.38.24:31000",
        "141.62.38.24:31001",
        "141.62.38.24:31002",
        "141.62.38.24:31003",
        "141.62.38.24:31004",
        "141.62.38.24:31005",
        "141.62.38.24:31006",
        "141.62.38.24:31007"
      ],
      "source_tp0_only": true,
      "timeout": 600,
      "diagnostics_requests": 0,
      "verify_kv": false
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
  }'
P_SCRIPT
bash -n /home/c00612283/dp1_tp8_chuban_start.sh
```

在已加载 CANN/ATB 的 12 号容器终端启动：

```bash
bash /home/c00612283/dp1_tp8_chuban_start.sh
```

日志打印在 `pd-logs/p/本次时间_PID/server.log`，另开终端检查：

```bash
curl --fail --noproxy '*' --connect-timeout 3 --max-time 10 \
  -i http://141.62.38.12:8100/health
```

P/D 在同一轮部署均 ready 后再启动 Proxy。

## 7. Proxy 启动与推理请求：12 号容器

以下沿用用户的健康检查和 Proxy 命令，地址统一为已确认的 P `.12`、D `.24`：

```bash
curl --fail --noproxy '*' http://141.62.38.12:8100/health
curl --fail --noproxy '*' http://141.62.38.24:8200/health

KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3 \
bash /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/kvaf_pd_proxy.sh \
  --host 141.62.38.12 --port 8000 \
  --prefiller-hosts 141.62.38.12 --prefiller-ports 8100 \
  --decoder-hosts 141.62.38.24 --decoder-ports 8200
```

如果健康检查失败，先处理 P/D，再执行 Proxy 命令。Proxy 前台运行，保持终端正常。
另开终端检查 Proxy，并发送一个真实 PD 请求：

```bash
curl --fail --noproxy '*' --max-time 5 \
  http://141.62.38.12:8000/healthcheck
curl --fail --noproxy '*' --connect-timeout 3 --max-time 180 \
  http://141.62.38.12:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm","messages":[{"role":"user","content":"请回复：测试成功"}],"temperature":0,"max_tokens":32,"stream":false}' \
  -w '\nHTTP_CODE=%{http_code} TOTAL_TIME=%{time_total}s\n'
```

Proxy 的 `/healthcheck` 仅证明自身服务状态；以 P/D 的 `/health` 和实际请求完整返回作为连通检查。
响应应包含 `choices` 和 `usage`，`finish_reason=length` 表示达到输出 token 上限。
短请求无法覆盖满 128-token block 的 DDR 路径，应继续使用保存的同一长请求做回归。

## 8. 启动顺序、验收与日志

按 `K → 协调器 → D → P → Proxy → 请求` 顺序执行，每一步确认对应服务就绪。
P/D 与 K 使用前台终端时保持 tee 正常，不使用 `Ctrl+Z` 暂停服务。

| 组件 | 本文日志位置 |
| --- | --- |
| K | `/home/c00612283/kvaf-k-时间.log` |
| 协调器 | `/home/c00612283/kvaf-coordinator-时间.log`，由启动命令打印 |
| P | 框架仓库 `pd-logs/p/时间_PID/server.log` 与 `plog/` |
| D | 框架仓库 `pd-logs/d/时间_PID/server.log` 与 `plog/` |
| Proxy | 当前运行终端 |

只关联同一轮启动日志，保存 P/D/K 的提交号、实际启动命令、日志、固定请求和完整响应。
本次为实测命令的文档归档，新增文档不代表重新完成一轮硬件精度或性能测试。

需要字节验收时，先排空并停止 P/D，将两份脚本同时改为
`diagnostics_requests=1`、`verify_kv=true`，再按 D → P → Proxy 启动并发送固定长请求。
将本轮两台服务器的 P/D 日志复制到同一机器后执行：

```bash
cd /home/c00612283/vllm-ascend-v0.30.0rc2
python3 scripts/check_kvaf_pd_diagnostics.py \
  /home/c00612283/pd-diag-p.log /home/c00612283/pd-diag-d.log \
  --require-ddr --require-compute
```

这两个路径需替换为已复制的实际日志；可传完整 server.log 或提取的 `KVAF_PD_DIAG` 行。
预期 `PD_DIAGNOSTICS_PASS`，核对 CPU/NPU `PULL_DONE`、`KV_CHECK=PASS`、`DECODE_READY` 及计算完成证据。
性能测量前将两端恢复为本文的 `0/false` 并重启；字节校验产生的时延不当作稳态性能。
模型精度继续使用同模型、tokenizer、采样参数和 MTP 设置的非 PD 基线比较完整输出。

## 9. 常见问题与恢复

| 现象 | 检查方式 |
| --- | --- |
| `--additional-config` 无法转换为 JSON | 检查逗号、引号和重复键；本文代码块已规范为合法 JSON，保留实测字段类型 |
| `Address already in use` | 在对应网络命名空间核对监听进程，不重复启动同端口 |
| `Cannot assign requested address` | 绑定 IP 必须属于实际运行宿主机或容器；本文 K/协调器均为 `.24` |
| `REMOTE_SOURCE_REJECTED` / `Probe rejected; result=4 state=5` | 保存 P 原始异常与同轮 K probe 上下文，核对实际 K 地址、SDK 版本及 CPU→源 HBM 数据面；不能只凭状态数字断定原因 |
| 协调器 `previous initializer failed/disconnected` | 初始化门控已封锁，不会因 D 重启或延长 timeout 自动清除；按下述顺序恢复 |
| 共享专家路径异常 | 保留 traceback 最后的异常及设备错误码；不能只凭函数名归因于图模式或重叠开关 |
| `Too many open files` | 检查实际 worker 的 `/proc/PID/limits` 与 fd 数；另开 shell 不会改变旧 worker 限制 |
| 日志显示通道 READY 但请求失败 | 建链成功不等于 MR 探测、Pull、解码消费和精度均通过，继续检查完整调用链 |

端口与进程检查，在相应主机/容器执行：

```bash
netstat -lntp 2>/dev/null \
  | grep -E ':28997[[:space:]]|:28998[[:space:]]|:3100[0-7][[:space:]]|:8100[[:space:]]|:8200[[:space:]]|:8000[[:space:]]'
ps -ww -eo pid,ppid,stat,etime,args \
  | grep -E '[k]vaf_cpu_service|[k]vaf_pd_init_coordinator.py|[v]llm.entrypoints.cli.main|[l]oad_balance_proxy'
```

本部署 K 日志不在 `pd-pool-logs`，在 24 号宿主机用精确关键字检查：

```bash
grep -n -i -B 10 -A 5 \
  -E 'probe|remote_source|rejected|retired' \
  /home/c00612283/kvaf-k-*.log
```

按启动命令打印的具体文件与时间定位本轮记录。不要只搜 `error`，正常日志里的
`error_suspend=1` 会产生大量误匹配。无输出只代表这些文件未匹配到关键字，应继续核对监听进程、
日志重定向和 D 的实际 `cpu_address`，不能认定探测已通过。

恢复时停止新请求和 Proxy，排空并正常停止 P，再停止 D，确认所有相关初始化和 DMA 已停止。
保存最早异常后检查 K；需要重启 K 时，旧 P/D ticket 全部失效，随后重启唯一协调器，
按 D → P → Proxy 重新启动。不要在旧 initializer 存活时重启协调器放行新初始化。
只处理确认属于本次部署的 PID，不使用全局 `pkill python` 或盲目重置设备。
