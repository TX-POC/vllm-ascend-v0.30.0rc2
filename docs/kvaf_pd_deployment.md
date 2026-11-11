# KVAF PD 分离部署操作指南

本文替代此前分散的安装、单机四层、双机、TP0、多 MR、初始化和诊断操作说明。
编译和 P/D/K/协调器/Proxy 命令以用户在 2026-11-11 提供的已验证环境为准；
K 服务地址经用户再次确认，统一为 24 号宿主机 141.62.38.24:28997。
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

本节按用户在 2026-11-11 提供的已验证命令整理。P、D 容器各执行一次，
使用容器内 `/usr/local/python3.12.13/bin/python3`、CANN 9.2.0 和 ascend950dt_9582。
K 侧的宿主机编译另见第 4 节，不使用容器 Python 路径。

### 3.1 完整安装：Ascend、KVOP、KVAF

完整安装会依次构建 Ascend 原生扩展、KVOP 和 KVAF SDK，成功后无需立即再执行单独重建。
`-` 表示保留现有 vLLM；不是跳过 vLLM-Ascend 编译。

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
test -x "$KVAF_PD_PYTHON"
test -f "$KVAF_PD_SRC/CMakeLists.txt"
test -f "$KVAF_PD_SRC/docs/connector_pull_api_design.md"

"$KVAF_PD_PYTHON" -c \
'import sys, torch, torch_npu, vllm; print(sys.executable); print(torch.__version__, torch_npu.__version__); print(vllm.__version__, vllm.__file__)'
command -v cmake
command -v ninja
command -v g++

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

`connector_pull_api_design.md` 是这组安装入口检查的分支标识；缺失时核对 SDK 版本和路径，
不要创建空文件绕过。安装脚本不下载或切换 SDK 分支；保留现有 HCCL device RA 借用补丁，
会检查/应用匹配 SDK 中的既有补丁目标，这不是本次文档更新新增的 SDK 修改。

### 3.2 单独重建的环境边界

完整安装括号内 source 的环境不会保留到父终端。下面每个单独重建和导入检查块都在
自己的子 shell 加载同一套 CANN，不依赖上一个括号的环境；同时沿用容器已有 ATB 配套环境。

### 3.3 单独重建 vLLM-Ascend 原生扩展

仅在完整安装未完成或原生扩展缺失时执行。保留现有 vLLM、torch 和 torch_npu，
使用原生 editable 安装流程，并通过实际扩展导入确认结果：

```bash
(
set -euo pipefail

CANN_ROOT=/usr/local/Ascend/cann-9.2.0
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

cd /home/c00612283/vllm-ascend-v0.30.0rc2

export SOC_VERSION=ascend950dt_9582
export COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS=8

/usr/local/python3.12.13/bin/python3 -m pip install \
  --no-deps --no-build-isolation -v -e . \
  2>&1 | tee /home/c00612283/ascend-native-extension-build.log

/usr/local/python3.12.13/bin/python3 -c \
'import torch, torch_npu; import vllm_ascend.vllm_ascend_C as m; print("ASCEND_EXTENSION_OK", m.__file__)'
)
```

### 3.4 单独重建 KVOP

KVOP 与 KVAF 是两个不同组件。这里使用用户验证过的 KVOP 原生 build.sh 参数，
并发 8、Python 3.12.13；原记录中的 KVOF 指这里的 KVOP。

```bash
(
set -euo pipefail

CANN_ROOT=/usr/local/Ascend/cann-9.2.0
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

cd /home/c00612283/vllm-ascend-v0.30.0rc2

bash third_party/KVOP/build.sh \
  8 /usr/local/python3.12.13/bin/python3 \
  2>&1 | tee /home/c00612283/kvop-build.log

source third_party/KVOP/env.sh
echo "KVOP_ENV_OK"
)
```

括号内加载的 KVOP 环境也不会传回父终端；P/D 启动脚本仍须各自 source third_party/KVOP/env.sh。

### 3.5 仅更新 SDK 及最终导入检查

若 Ascend 原生扩展和 KVOP 已正确构建，仅更新 SDK 时可继续使用 SDK-only 安装入口：

```bash
(
set -euo pipefail

CANN_ROOT=/usr/local/Ascend/cann-9.2.0
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

cd /home/c00612283/vllm-ascend-v0.30.0rc2
export KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3
bash scripts/install_kvaf_pd.sh \
  /home/c00612283/KVAF-Devlop \
  /home/c00612283/KVAF-Devlop/build-pd-py312
)
```

检查原生扩展及 SDK，不能只以 binding 链接成功作为整体安装成功的依据：

```bash
(
set -euo pipefail

CANN_ROOT=/usr/local/Ascend/cann-9.2.0
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

框架采用 editable 安装时，纯 Python 或文档更新无需重编译 vLLM、Ascend 扩展、KVOP 或 SDK。

## 4. K 编译与启动：24 号宿主机

K 在宿主机编译、运行，需有同版本 SDK 源码、CMake、C++ 编译器、UMDK 头文件及 liburma。
这里直接使用 SDK 的 CMake 入口，不依赖宿主机存在完整 vLLM/Ascend 容器环境或容器 Python。
协调器另需框架目录中的 Python 脚本和宿主机 Python3。

### 4.1 K 侧机头编译

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

K 的 KVAF_ENABLE_CPU_OFFLOAD=OFF 关闭的是构建 NPU 侧 offload Runtime，不是关闭远程 CPU 服务；
远程服务由 KVAF_BUILD_REMOTE_CPU_SERVICE=ON 单独启用。K 不构建 Python binding 或 AIV Get/Put kernel。
ldd 输出须没有 not found；K_SERVICE_BUILD_OK 还不能代替后续实际启动检查。

### 4.2 K 服务：24 号宿主机前台运行

用户已确认 K 在 24 号，因此统一监听 141.62.38.24:28997，D 的 cpu_address 保持同一个地址。
早先命令中的 141.62.38.12 不能在 24 号上直接绑定。先确认旧监听者已退出，另开专用终端执行：

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

这是前台服务，保持终端及 tee 正常运行，不使用 Ctrl+Z 暂停。
如需后台驻留，使用直接重定向到文件的 nohup 方式，避免后台 tee 被暂停后造成 pipe_write。

### 4.3 K 初始化协调器：24 号宿主机

只运行一个协调器。使用宿主机实际 command -v python3，不使用容器解释器路径。
先检查 28998；已有健康实例可保留，暂停或残留实例按第 9 节处理。

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

COORD_PID=$!
echo "COORD_PID=$COORD_PID"
echo "COORD_LOG=$COORD_LOG"

ready=0
for attempt in {1..20}; do
    if ! kill -0 "$COORD_PID" 2>/dev/null; then
        cat "$COORD_LOG" >&2
        exit 1
    fi
    if [ -s "$COORD_LOG" ] &&
       grep -Fq 'KVAF_INIT_COORDINATOR_LISTEN address=141.62.38.24:28998' "$COORD_LOG" &&
       kill -0 "$COORD_PID" 2>/dev/null; then
        ready=1
        break
    fi
    sleep 1
done
if [ "$ready" -ne 1 ]; then
    tail -n 30 "$COORD_LOG" >&2
    echo "协调器未在20秒内就绪，请先检查已打印的PID和日志，不要重复启动。" >&2
    exit 1
fi
)
```

启动命令保留原有参数，补充新进程存活和就绪日志检查；超时不自动杀进程，先查状态。
后台 PID 本身不是成功依据。然后在同一宿主机核对实际监听者和命令：

```bash
netstat -lntp 2>/dev/null \
  | grep -E ':28997[[:space:]]|:28998[[:space:]]'
ps -ww -eo pid,ppid,stat,etime,args \
  | grep -E '[k]vaf_cpu_service|[k]vaf_pd_init_coordinator.py'
```

必须看到 K 的 LISTEN、协调器的 KVAF_INIT_COORDINATOR_LISTEN 和正确进程持有的监听端口。
启动顺序为 K → 协调器 → D → P → Proxy。

## 5. 启动 D：24 号容器

以下为用户在 2026-11-11 再次提供的已验证 D 配置，直接调用原生 CLI，不经过会默认添加
`--enforce-eager` 的启动 helper。保留异步调度、FULL_DECODE_ONLY、MLAPO 和 profiling 配置。
这条命令在当前终端前台运行，另开终端检查就绪；不要使用 Ctrl+Z 暂停服务或 tee。
先在实际启动 shell 检查 `ulimit -Sn`、`ulimit -Hn`，硬限制不足时先调整该 shell 或容器配置。
profiling 目录 `/home/d00843668/profiling_no_pooling` 须存在或可由运行用户创建、写入。
按用户本次命令保留 control_backlog=128；省略也使用同一默认值。它位于 kv_connector_extra_config，
不能放进 additional_config.kvaf_config。
该参数仅作用于 D 框架控制端口，不影响 KVAF/模型协议，详见 [监听队列说明](kvaf_pd_configuration.md#d-控制连接突发与监听队列)。

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
      "control_backlog": 128,
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

### 监听队列更新与复测

仅此 Python 优化无需重编译 vLLM、Ascend、KVOP 或 KVAF；健康 K 和协调器可保留。
先停止 Proxy 新请求，排空并正常停止 P/D，更新源码和 D JSON，再按 D → P → Proxy 重启。
只有已更新代码的 D 才接受 control_backlog；旧 D 会因未知字段拒绝启动，不能只改 JSON 不更新代码。
不需要在 P 添加该字段。当前代码的 D 即使不添加字段，也默认使用 128。

在 D 所在的网络命名空间读取系统上限，检查全部控制端口，并保留同一轮连接突发测试的时间和日志：

```bash
cat /proc/sys/net/core/somaxconn
netstat -lntp 2>/dev/null | grep -E ':3100[0-7][[:space:]]'
```

比较请求前后的 ListenOverflows/ListenDrops 增量及控制 RPC 失败率，不能仅凭历史累计值判断优化有效。
队列只吸收短时连接突发，不替代硬件精度验证，也不保证所有超时均能消除。

## 6. 启动 P：12 号容器

以下按用户本次提供的 P 命令展开原生 CLI，保留 TP8/DP1/EP8、eager、MTP3、
8 个 decode_endpoints、TP0 单源、1 个诊断请求和 KV 字节校验。
同样前台运行，另开终端检查健康；不经过启动 helper 自动补参。

```bash
(
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
      "diagnostics_requests": 1,
      "verify_kv": true
    }
  }' \
  --additional-config '{
    "enable_cpu_binding": true,
    "enable_dsa_cp": false,
    "c8_enable_reshape_optim": false,
    "kvaf_config": {
      "enabled": false,
      "build_dir": "/home/c00612283/KVAF-Devlop/build-pd-py312"
    }
  }'
)
```

```bash
curl --fail --noproxy '*' --max-time 10 -i http://141.62.38.12:8100/health
```

本次 P/D 命令记录的是已验证部署基线：max-model-len=131072、max-num-seqs=1、
source_tp0_only=true、diagnostics_requests=1、verify_kv=true。
它不是 131072 输入/125 输出、D 单卡 batch=16 的性能采样配置；切换负载时须另核算总长度、
HBM/DDR 预算及诊断成本，不能把两套配置混用。没有在本次安装文档中自动启用临时采样 Connector。

已有自定义启动脚本可继续使用，但须逐项对齐上述已验证参数。
仓库根目录四层脚本和 examples/.../kvaf_two_node 的历史模板保留兼容用途，
其默认 TP/DP、8192 长度和占位 IP 不等同于本次全模型基线，不直接套用。

## 7. Proxy 与请求：12 号容器

确认 8000 没有旧 Proxy，先核对监听者和进程参数。
按用户已验证方式，在 12 号容器先检查 P/D 健康，再以前台方式启动 Proxy：

```bash
(
set -euo pipefail
cd /home/c00612283/vllm-ascend-v0.30.0rc2

curl --fail --noproxy '*' http://141.62.38.12:8100/health
curl --fail --noproxy '*' http://141.62.38.24:8200/health

KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3 \
bash /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/kvaf_pd_proxy.sh \
  --host 141.62.38.12 --port 8000 \
  --prefiller-hosts 141.62.38.12 --prefiller-ports 8100 \
  --decoder-hosts 141.62.38.24 --decoder-ports 8200
)
```

本节 Proxy 前台运行，保持终端正常，另开终端发送请求。
/health 返回 HTTP 200 且 body 为空是正常行为；端口占用时不要暂停旧服务后直接起第二个。

```bash
curl --noproxy '*' --connect-timeout 3 --max-time 180 \
  http://141.62.38.12:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm","messages":[{"role":"user","content":"请回复：测试成功"}],"temperature":0,"max_tokens":32,"stream":false}' \
  -w '\nHTTP_CODE=%{http_code} TOTAL_TIME=%{time_total}s\n'
```

响应应包含 choices 和 usage。流式请求的访问日志 200 可能只表示响应头已发送；
以客户端完整收到内容为准。finish_reason=length 表示达到输出 token 上限。
短请求不覆盖满块 DDR 路径，继续使用已保存的 41510-token 请求做长请求回归，
输入 token 加输出上限必须小于等于本基线的 131072，不能重新拼写 prompt 后假定仍是同一用例。

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
| Free memory on device … less than desired GPU memory utilization | 启动前 HBM 空闲不足；在宿主机用 npu-smi info 和 ps 查旧实例/其他容器占用，确认归属后释放；只剩 1.11 GiB 时把 0.85 降成 0.8 也无法解决 |
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

本次已验证命令中 K/P/D 使用前台 tee，协调器后台直接重定向文件，Proxy 前台运行。
采用后台运行替代方案时，K/协调器优先直接写文件；前台 tee 需保持运行。`STAT=T` 使用明确 PID 恢复或正常结束，避免暂停 tee
造成 `pipe_write`。只清理确认属于本次部署的进程，不使用全局 pkill python 或盲目重置设备。

## 10. 辅助工具与归档

`scripts/kvaf_pd_ddr_residency.py --help` 提供无模型串行/并行 DDR 驻留测试；
它检验初始化与共驻留，不证明 KV 字节或模型精度。
`scripts/kvaf_pd_smoke.py --help` 提供固定文本的 completions 连通测试。
它们保留用于回归，不参与正常请求热路径。

归档本次框架/SDK 提交号、实际构建路径、P/D/K/协调器命令、请求、基线响应、PD 响应和对应日志。
升级后复用同一回归用例；不要将未执行的模式写成已通过。

四智算节点与八鲲鹏节点的扩容模板及脚本见 [池化部署指南](kvaf_pd_pooled_deployment.md)。
其中 P0/P1、D0/D1、K0～K7、协调器和 Proxy 均提供独立启动脚本，直接编辑本机参数及原生 JSON。
后台脚本检查新进程持有的监听和就绪状态，失败返回非零；K 部分成功时保留服务，需人工核对后恢复。
