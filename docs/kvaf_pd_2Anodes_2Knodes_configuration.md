# KVAF 两台 A 服务器与两台 K 服务器配置说明

本文按用户提供的 K1、K2、协调器及 P/D 启动脚本整理，包含完整可保存脚本、关键参数解释、
D 设备到 K 的路由、启动顺序和日志核对命令。两台 A 是八卡智算服务器，分别承担 P 与 D；
两台 K 是独立鲲鹏 CPU DDR 服务器。本方案不将两台 A 配成两个 P 或两个 D。

参数沿用用户提供的脚本；仅清理粘贴转义和 D 中重复的 `multistream_overlap_shared_expert`，
保留最终生效的布尔值 `true`，`recompute_scheduler_enable` 的字符串 `"True"` 保持不变。
K1 的 `k4`、K2 的 `k0` 日志目录及脚本中的原注释也保留；它们是历史模板标签，不改变实际主机角色。

## 1. 主机与服务拓扑

| 角色 | 地址 / 执行环境 | 本次配置 | 服务 |
| --- | --- | --- | --- |
| A-P | `141.62.38.12`，容器 | 八卡 TP8 / DP1 / EP8，eager | HTTP `8100`，DP RPC 基础端口 `13389` |
| A-D | `141.62.38.24`，容器 | 八卡 TP1 / DP8 / EP8，Target Decode 图模式 | HTTP `8200`，DP RPC 基础端口 `13489`，PD 控制 `31000`～`31007` |
| K1 | `141.62.38.71`，宿主机 | 两个 NUMA 绑定 CPU 服务 | `28997`、`28999` |
| K2 | `141.62.38.73`，宿主机 | 两个 NUMA 绑定 CPU 服务 | `28997`、`28999` |
| 初始化协调器 | K2 `141.62.38.73`，宿主机 | 唯一实例，Python 标准库 | `28998` |
| Proxy | A-P `141.62.38.12`，容器 | P HTTP → D HTTP | `8000` |

P/D 各自为本机 EP8，不组成跨 A 的 EP16。协调器也管理 K1 endpoint 的初始化，不要求 K1 再起一个协调器。
K1/K2 上相同的端口号属于不同主机，不冲突；协调器端口不同于 K CPU 数据服务端口。

### 1.1 源码与编译准备

- P/D 框架使用同一 `v0.30.0rc2-develop` 提交，目录为 `/home/c00612283/vllm-ascend-v0.30.0rc2`。
- P/D 和两个 K 的 SDK 使用同一 `codex/cpu-pull-hbm-reference` 提交；当前适配基准为 `74c3fc491a257bae73356da54eafb08e1e9e4e45`。
- P/D 的 Python binding 为 `/home/c00612283/KVAF-Devlop/build-pd-py312`；K 的 CPU 二进制来自 `build-hostcpu-k`。
- 模型为 `/home/models/GLM-5.3-w4a4c8-mxfp4`；容器使用 Python 3.12.13、配套 CANN/ATB 和 KVOP。
- 两个 K 各安装匹配 UMDK/liburma、C++ 编译器、CMake、numactl 和宿主机 Python3；不需要编译完整 vLLM/Ascend/KVOP。

完整 P/D 编译和 K CPU-only CMake 命令见 [双机部署指南](kvaf_pd_2nodes_deployment.md) 的编译章节。
P/D 容器各编译一次，K1/K2 宿主机各编译一次；该指南原单 K 的 `.24:28997` 启动地址不用于本拓扑，
启动和路由以本文 `.71/.73` 的脚本为准。协调器还需框架中的入口脚本及其相对目录下的 `initialization.py`。

## 2. D 八卡到四个 K endpoint 的映射

`cpu_addresses` 的键是物理 NPU ID 字符串，不是 DP rank 或 socket 号。
当前 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，完整映射如下：

| D 物理 NPU | K 主机 | CPU 服务端口 | K 脚本索引 / NUMA 集合 | 每 D worker DDR |
| --- | --- | --- | --- | --- |
| 0、1 | K1 `.71` | `28997` | `socket=0` / `numa_nodes[0]=0` | 各 16 GiB |
| 2、3 | K1 `.71` | `28999` | `socket=1` / `numa_nodes[1]=1` | 各 16 GiB |
| 4、5 | K2 `.73` | `28997` | `socket=0` / `numa_nodes[0]=0` | 各 16 GiB |
| 6、7 | K2 `.73` | `28999` | `socket=1` / `numa_nodes[1]=1` | 各 16 GiB |

两个 D worker 共用一个 CPU 服务 endpoint，但分别拥有自己的 Runtime/session 和 DDR region。
`ddr_size_gb=16` 是每 worker 的预算：每服务约 32 GiB、每 K 主机约 64 GiB、两个 K 合计约 128 GiB，
另需预留 SDK、系统和其他进程的内存。不能把四个 CPU 服务各当作只分配 16 GiB，也不能把 16 当作每层预算。
这是固定分区路由，不是全局均衡池；某个 region 用满不会自动借用另一个 K 的空闲容量。

本次 D 不填写 `cpu_address`，由覆盖 0～7 的完整 `cpu_addresses` 决定全部实际 K 路由。
源码对未命中的物理设备会回退到 `cpu_address`，缺省为 `127.0.0.1:28999`；它不是本拓扑的备用 K。
更换可见卡或物理卡序后必须检查映射覆盖，不能让缺失项意外连接本机回环地址。

### 2.1 NUMA 与启动参数

| 参数 | K1 / K2 取值 | 作用 |
| --- | --- | --- |
| `local_ip` | `.71 / .73` | 本机 CPU 服务监听地址，必须属于实际宿主机 |
| `numa_nodes` | 两端 `(0 1)` | 两个服务分别执行 CPU 与内存 NUMA 绑定；先核对真实拓扑 |
| `ports` | 两端 `(28997 28999)` | 与数组索引一一对应的监听端口，必须和 D 映射一致 |
| `K_BIN` | SDK `build-hostcpu-k/runtime/kvaf_cpu_service` | 匹配版本的 CPU 服务程序 |
| `python_bin` | `command -v python3` | 就绪检查解释器，不使用 NPU 容器 Python 路径 |
| `startup_timeout` | K 为 60，协调器为 20 | 单次就绪检查的秒数，脚本允许大于 0 且不超过 3600；不是 PD timeout 或 DMA 超时 |
| `--memory-page-mode` | `default` | 沿用服务默认内存页模式，不等于已经启用大页 |

在两台 K 各执行以下检查。NUMA node 号不必等于 CPU socket 号：

```bash
lscpu -e=CPU,SOCKET,NODE
numactl --hardware
netstat -lntp 2>/dev/null | grep -E ':28997[[:space:]]|:28999[[:space:]]'
```

若每个 socket 包含多个 node，应按本机实际集合调整，例如 `numa_nodes=("0-3" "4-7")`。
`socket=0/1` 在脚本中只是服务索引；numactl 绑定不等于固定 URMA/EID 候选，也不能单凭绑定参数证明远程 Pull 数据面可用。

## 3. K1 完整脚本：141.62.38.71

在 K1 保存为 `/home/c00612283/k_start.sh`，不传参数执行。
保留用户原脚本的 `K4` 注释和 `pd-pool-logs/k4` 目录，实际监听以 `.71` 为准。

```bash
cat > /home/c00612283/k_start.sh <<'K1_SCRIPT'
#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 编辑本文件中的 IP、网卡、路径和原生 JSON，然后不带参数执行。
set -euo pipefail
[[ $# == 0 ]] || { echo "Edit this script, then run it without arguments" >&2; exit 2; }

# 在 K4 鲲鹏宿主机执行；先核对两个 CPU socket 的实际 NUMA 节点集合。
local_ip=141.62.38.71
numa_nodes=(0 1)
ports=(28997 28999)
K_BIN=/home/c00612283/KVAF-Devlop/build-hostcpu-k/runtime/kvaf_cpu_service
python_bin=$(command -v python3)
startup_timeout=60
command -v numactl >/dev/null
test -x "$K_BIN"
dir="/home/c00612283/pd-pool-logs/k4/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$dir"

# 只观察本脚本新建进程的日志和监听；失败时不杀已有服务，也不自动回收 K 会话。
wait_service() {
    if "$python_bin" - "$1" "$2" "$3" "$4" "$5" "$startup_timeout" "${6:-}" <<'READY'
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time
from urllib.request import ProxyHandler, build_opener

pid, host, port, log, marker, timeout, health = sys.argv[1:]
pid, port, timeout = int(pid), int(port), float(timeout)
if not 0 < timeout <= 3600:
    raise SystemExit("startup_timeout must be in (0, 3600]")
process = Path(f"/proc/{pid}")
deadline = time.monotonic() + timeout
identity = None
announced = False
opener = build_opener(ProxyHandler({}))
last = "waiting for listener and readiness log"


def owns_listener():
    inodes = set()
    for fd in (process / "fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith("socket:["):
            inodes.add(target[8:-1])
    with open("/proc/net/tcp") as table:
        for line in table:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A" or fields[9] not in inodes:
                continue
            address, service_port = fields[1].split(":")
            address = socket.inet_ntoa(struct.pack("=I", int(address, 16)))
            if int(service_port, 16) == port and address in (host, "0.0.0.0"):
                return True
    return False


try:
    while time.monotonic() < deadline:
        state = (process / "stat").read_text().rsplit(")", 1)[1].split()
        if state[0] in ("Z", "X", "T", "t"):
            raise RuntimeError(f"process exited or stopped: state={state[0]}")
        if identity is not None and identity != state[19]:
            raise RuntimeError("PID was reused by another process")
        identity = state[19]
        if not announced:
            try:
                announced = marker in Path(log).read_text(errors="replace")
            except FileNotFoundError:
                pass
        if announced and owns_listener():
            if health:
                try:
                    with opener.open(health, timeout=min(1, max(0.01, deadline - time.monotonic()))) as response:
                        payload = json.load(response)
                        if response.status != 200 or not isinstance(payload, dict) or payload.get("status") != "ok":
                            raise ValueError("healthcheck did not return status=ok")
                except (OSError, ValueError) as error:
                    last = f"healthcheck: {error}"
                    time.sleep(0.1)
                    continue
            # Recheck the same process after observing its socket/HTTP response.
            current = (process / "stat").read_text().rsplit(")", 1)[1].split()
            if current[19] != identity or current[0] in ("Z", "X", "T", "t"):
                raise RuntimeError("service exited or stopped during readiness check")
            print(f"SERVICE_READY pid={pid} address={host}:{port} log={log}", flush=True)
            break
        time.sleep(0.1)
    else:
        raise TimeoutError(last)
except (OSError, ValueError, RuntimeError, TimeoutError) as error:
    raise SystemExit(f"SERVICE_START_FAILED pid={pid} address={host}:{port} log={log}: {error}")
READY
    then
        return 0
    else
        tail -n 20 "$4" >&2 || true
        return 1
    fi
}

pids=()
for socket in 0 1; do
    log="$dir/socket-$socket.log"
    nohup numactl --cpunodebind="${numa_nodes[socket]}" --membind="${numa_nodes[socket]}" \
      "$K_BIN" --listen "$local_ip" --port "${ports[socket]}" --memory-page-mode default \
      > "$log" 2>&1 < /dev/null &
    pids+=("$!")
    printf '%s\n' "$!" > "$dir/socket-$socket.pid"
    printf 'K_LAUNCHED socket=%s pid=%s log=%s\n' "$socket" "$!" "$log"
done
result=0
for socket in 0 1; do
    if ! wait_service "${pids[socket]}" "$local_ip" "${ports[socket]}" "$dir/socket-$socket.log" \
      "[KVAF_CPU] LISTEN address=$local_ip port=${ports[socket]}"; then
        result=1
    fi
done
if [[ $result != 0 ]]; then
    echo "K startup incomplete; inspect the printed PIDs before retrying. Running services were retained." >&2
fi
exit "$result"
K1_SCRIPT
bash -n /home/c00612283/k_start.sh
bash /home/c00612283/k_start.sh
```

## 4. K2 完整脚本：141.62.38.73

在 K2 保存为同一路径 `/home/c00612283/k_start.sh`。这是另一台机器，不覆盖 K1 文件。
保留用户原脚本的 `K0` 注释和 `pd-pool-logs/k0` 目录，实际监听以 `.73` 为准。

```bash
cat > /home/c00612283/k_start.sh <<'K2_SCRIPT'
#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 编辑本文件中的 IP、网卡、路径和原生 JSON，然后不带参数执行。
set -euo pipefail
[[ $# == 0 ]] || { echo "Edit this script, then run it without arguments" >&2; exit 2; }

# 在 K0 鲲鹏宿主机执行；先核对两个 CPU socket 的实际 NUMA 节点集合。
local_ip=141.62.38.73
numa_nodes=(0 1)
ports=(28997 28999)
K_BIN=/home/c00612283/KVAF-Devlop/build-hostcpu-k/runtime/kvaf_cpu_service
python_bin=$(command -v python3)
startup_timeout=60
command -v numactl >/dev/null
test -x "$K_BIN"
dir="/home/c00612283/pd-pool-logs/k0/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$dir"

# 只观察本脚本新建进程的日志和监听；失败时不杀已有服务，也不自动回收 K 会话。
wait_service() {
    if "$python_bin" - "$1" "$2" "$3" "$4" "$5" "$startup_timeout" "${6:-}" <<'READY'
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time
from urllib.request import ProxyHandler, build_opener

pid, host, port, log, marker, timeout, health = sys.argv[1:]
pid, port, timeout = int(pid), int(port), float(timeout)
if not 0 < timeout <= 3600:
    raise SystemExit("startup_timeout must be in (0, 3600]")
process = Path(f"/proc/{pid}")
deadline = time.monotonic() + timeout
identity = None
announced = False
opener = build_opener(ProxyHandler({}))
last = "waiting for listener and readiness log"


def owns_listener():
    inodes = set()
    for fd in (process / "fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith("socket:["):
            inodes.add(target[8:-1])
    with open("/proc/net/tcp") as table:
        for line in table:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A" or fields[9] not in inodes:
                continue
            address, service_port = fields[1].split(":")
            address = socket.inet_ntoa(struct.pack("=I", int(address, 16)))
            if int(service_port, 16) == port and address in (host, "0.0.0.0"):
                return True
    return False


try:
    while time.monotonic() < deadline:
        state = (process / "stat").read_text().rsplit(")", 1)[1].split()
        if state[0] in ("Z", "X", "T", "t"):
            raise RuntimeError(f"process exited or stopped: state={state[0]}")
        if identity is not None and identity != state[19]:
            raise RuntimeError("PID was reused by another process")
        identity = state[19]
        if not announced:
            try:
                announced = marker in Path(log).read_text(errors="replace")
            except FileNotFoundError:
                pass
        if announced and owns_listener():
            if health:
                try:
                    with opener.open(health, timeout=min(1, max(0.01, deadline - time.monotonic()))) as response:
                        payload = json.load(response)
                        if response.status != 200 or not isinstance(payload, dict) or payload.get("status") != "ok":
                            raise ValueError("healthcheck did not return status=ok")
                except (OSError, ValueError) as error:
                    last = f"healthcheck: {error}"
                    time.sleep(0.1)
                    continue
            # Recheck the same process after observing its socket/HTTP response.
            current = (process / "stat").read_text().rsplit(")", 1)[1].split()
            if current[19] != identity or current[0] in ("Z", "X", "T", "t"):
                raise RuntimeError("service exited or stopped during readiness check")
            print(f"SERVICE_READY pid={pid} address={host}:{port} log={log}", flush=True)
            break
        time.sleep(0.1)
    else:
        raise TimeoutError(last)
except (OSError, ValueError, RuntimeError, TimeoutError) as error:
    raise SystemExit(f"SERVICE_START_FAILED pid={pid} address={host}:{port} log={log}: {error}")
READY
    then
        return 0
    else
        tail -n 20 "$4" >&2 || true
        return 1
    fi
}

pids=()
for socket in 0 1; do
    log="$dir/socket-$socket.log"
    nohup numactl --cpunodebind="${numa_nodes[socket]}" --membind="${numa_nodes[socket]}" \
      "$K_BIN" --listen "$local_ip" --port "${ports[socket]}" --memory-page-mode default \
      > "$log" 2>&1 < /dev/null &
    pids+=("$!")
    printf '%s\n' "$!" > "$dir/socket-$socket.pid"
    printf 'K_LAUNCHED socket=%s pid=%s log=%s\n' "$socket" "$!" "$log"
done
result=0
for socket in 0 1; do
    if ! wait_service "${pids[socket]}" "$local_ip" "${ports[socket]}" "$dir/socket-$socket.log" \
      "[KVAF_CPU] LISTEN address=$local_ip port=${ports[socket]}"; then
        result=1
    fi
done
if [[ $result != 0 ]]; then
    echo "K startup incomplete; inspect the printed PIDs before retrying. Running services were retained." >&2
fi
exit "$result"
K2_SCRIPT
bash -n /home/c00612283/k_start.sh
bash /home/c00612283/k_start.sh
```

### 4.1 两份 K 脚本的就绪判定

脚本先在后台启动两个服务，再依次检查；每个都需要满足以下条件：

1. 本次新 PID 仍存活，且不处于停止或僵尸状态，观察期间 PID 未被其他进程复用。
2. 本次日志已出现对应 `[KVAF_CPU] LISTEN` 标记。
3. 从 `/proc/PID/fd` 和 `/proc/net/tcp` 确认目标 IPv4 listener 属于这个新进程，不能以旧服务占端口冒充成功。

两服务都打印 `SERVICE_READY`，且脚本返回 0，才表示本机启动检查通过。
任一失败则整体返回非零，保留已运行服务、PID 和日志，不自动杀进程、重试或回收会话。
每项最多等待 60 秒，依次检查最坏可能约 120 秒；超时后服务可能稍后就绪，先查 PID 再决定是否重试。
函数带有可选 HTTP health 分支，但 K 和协调器调用都未传 health URL；本次不会额外发控制探测请求。

## 5. 唯一协调器：K2 141.62.38.73:28998

仅在 K2 保存并执行 `/home/c00612283/coordinator_start.sh`，不在 K1 再启动第二个协调器。
原注释中的 K0 是历史标签；D 的 `initialization_coordinator` 指向 `.73:28998`。

```bash
cat > /home/c00612283/coordinator_start.sh <<'COORDINATOR_SCRIPT'
#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 编辑本文件中的 IP、网卡、路径和原生 JSON，然后不带参数执行。
set -euo pipefail
[[ $# == 0 ]] || { echo "Edit this script, then run it without arguments" >&2; exit 2; }

# 仅在 K0 宿主机运行一个协调器，使用宿主机实际 Python。
local_ip=141.62.38.73
port=28998
python_bin=$(command -v python3)
startup_timeout=20
COORDINATOR=/home/c00612283/vllm-ascend-v0.30.0rc2/scripts/kvaf_pd_init_coordinator.py
test -f "$COORDINATOR"
dir="/home/c00612283/pd-pool-logs/coordinator/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$dir"

# 只观察本脚本新建进程的日志和监听；失败时不杀已有服务，也不自动回收 K 会话。
wait_service() {
    if "$python_bin" - "$1" "$2" "$3" "$4" "$5" "$startup_timeout" "${6:-}" <<'READY'
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time
from urllib.request import ProxyHandler, build_opener

pid, host, port, log, marker, timeout, health = sys.argv[1:]
pid, port, timeout = int(pid), int(port), float(timeout)
if not 0 < timeout <= 3600:
    raise SystemExit("startup_timeout must be in (0, 3600]")
process = Path(f"/proc/{pid}")
deadline = time.monotonic() + timeout
identity = None
announced = False
opener = build_opener(ProxyHandler({}))
last = "waiting for listener and readiness log"


def owns_listener():
    inodes = set()
    for fd in (process / "fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith("socket:["):
            inodes.add(target[8:-1])
    with open("/proc/net/tcp") as table:
        for line in table:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A" or fields[9] not in inodes:
                continue
            address, service_port = fields[1].split(":")
            address = socket.inet_ntoa(struct.pack("=I", int(address, 16)))
            if int(service_port, 16) == port and address in (host, "0.0.0.0"):
                return True
    return False


try:
    while time.monotonic() < deadline:
        state = (process / "stat").read_text().rsplit(")", 1)[1].split()
        if state[0] in ("Z", "X", "T", "t"):
            raise RuntimeError(f"process exited or stopped: state={state[0]}")
        if identity is not None and identity != state[19]:
            raise RuntimeError("PID was reused by another process")
        identity = state[19]
        if not announced:
            try:
                announced = marker in Path(log).read_text(errors="replace")
            except FileNotFoundError:
                pass
        if announced and owns_listener():
            if health:
                try:
                    with opener.open(health, timeout=min(1, max(0.01, deadline - time.monotonic()))) as response:
                        payload = json.load(response)
                        if response.status != 200 or not isinstance(payload, dict) or payload.get("status") != "ok":
                            raise ValueError("healthcheck did not return status=ok")
                except (OSError, ValueError) as error:
                    last = f"healthcheck: {error}"
                    time.sleep(0.1)
                    continue
            # Recheck the same process after observing its socket/HTTP response.
            current = (process / "stat").read_text().rsplit(")", 1)[1].split()
            if current[19] != identity or current[0] in ("Z", "X", "T", "t"):
                raise RuntimeError("service exited or stopped during readiness check")
            print(f"SERVICE_READY pid={pid} address={host}:{port} log={log}", flush=True)
            break
        time.sleep(0.1)
    else:
        raise TimeoutError(last)
except (OSError, ValueError, RuntimeError, TimeoutError) as error:
    raise SystemExit(f"SERVICE_START_FAILED pid={pid} address={host}:{port} log={log}: {error}")
READY
    then
        return 0
    else
        tail -n 20 "$4" >&2 || true
        return 1
    fi
}

log="$dir/server.log"
nohup "$python_bin" -u "$COORDINATOR" --host "$local_ip" --port "$port" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$dir/server.pid"
printf 'COORDINATOR_LAUNCHED pid=%s log=%s\n' "$pid" "$log"
wait_service "$pid" "$local_ip" "$port" "$log" "KVAF_INIT_COORDINATOR_LISTEN address=$local_ip:$port"
COORDINATOR_SCRIPT
bash -n /home/c00612283/coordinator_start.sh
bash /home/c00612283/coordinator_start.sh
```

同一个协调器按实际 K endpoint 字符串分别门控初始化：本次有 `.71:28997`、`.71:28999`、
`.73:28997`、`.73:28999` 四个键，同一键下的两个 D worker 初始化被串行准入。
不同 endpoint 独立门控；协调器不把所有 D 永久串成一条执行队列，也不承载请求热路径 KV 数据。
所有 D 都应填写同一个协调器地址，且引用同一 K 时使用一致的 endpoint 字符串。

协调器使用宿主机 Python 标准库，入口依赖以下两个文件保持相对目录：

```text
/home/c00612283/vllm-ascend-v0.30.0rc2/scripts/kvaf_pd_init_coordinator.py
/home/c00612283/vllm-ascend-v0.30.0rc2/vllm_ascend/distributed/kv_transfer/kvaf/pd/initialization.py
```

## 6. P 完整脚本：A-P 141.62.38.12

P 沿用八卡 TP8/DP1、eager、MTP3、TP0-only 和 D 的八个控制地址；不向 P 添加 K 路由。
在 12 号容器已加载的 CANN/ATB 环境保存并执行：

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
bash /home/c00612283/dp1_tp8_chuban_start.sh
```

P 的 `kvaf_config.enabled=false` 表示不启用 D 侧 KVAF Runtime，不是关闭 PD producer。
P 仍通过 Connector 注册源 KV；CPU Pull 的 K endpoint 由目标 D 的 receiver 信息决定。
`source_tp0_only=true` 时仅逻辑 TP0 参与源注册/传输，其他 TP rank 仍参与模型计算。

## 7. D 完整脚本：A-D 141.62.38.24

D 沿用 TP1/DP8、异步调度、Target `FULL_DECODE_ONLY`、npugraph_ex、LMHead TP8、MLAPO 和 MTP eager。
与单 K 基线的关键差异是协调器迁至 `.73`，并使用第 2 节的完整 `cpu_addresses`。
在 24 号容器已加载的 CANN/ATB 环境保存并执行：

```bash
cat > /home/c00612283/d_k2_dp8tp1start.sh <<'D_SCRIPT'
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
      "initialization_coordinator": "141.62.38.73:28998",
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
  "finegrained_tp_config": {
    "lmhead_tensor_parallel_size": 8
  },
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
    "ddr_size_gb": 16,
    "cpu_addresses": {
      "0": "141.62.38.71:28997",
      "1": "141.62.38.71:28997",
      "2": "141.62.38.71:28999",
      "3": "141.62.38.71:28999",
      "4": "141.62.38.73:28997",
      "5": "141.62.38.73:28997",
      "6": "141.62.38.73:28999",
      "7": "141.62.38.73:28999"
    }
  }
}'
D_SCRIPT
bash -n /home/c00612283/d_k2_dp8tp1start.sh
bash /home/c00612283/d_k2_dp8tp1start.sh
```

### 7.1 P/D 关键配置解释

| 字段 | 本次值 | 作用及修改约束 |
| --- | --- | --- |
| TP / DP | P `8/1`，D `1/8` | 模型计算拓扑；本机 EP8 由原生专家并行组成 |
| `max-model-len` | `131072` | 输入与输出总 token 上限，长请求要预留输出额度 |
| `max-num-batched-tokens` | `4096` | 每引擎单步调度 token 预算，不是上下文总长 |
| `max-num-seqs` | `1` | 每引擎并发序列上限；不是 D 单卡 batch=8 |
| `gpu-memory-utilization` | `0.85` | NPU HBM 预算比例，不是 K DDR 比例 |
| `source_tp0_only` | P/D 均 `true` | 传输源参与模式；改为全建链时两端同步改为 false 并排空重启 |
| `initialization_coordinator` | D `.73:28998` | 初始化门控；不是 K 数据服务地址 |
| `cpu_addresses` | D 完整映射 0～7 | 物理 NPU → K endpoint，命中时优先于 cpu_address |
| `ddr_size_gb` | D `16` | 每 worker 的 GiB 预算；两个 worker 共 endpoint 时每服务合计约 32 GiB |
| `npu_core_count / jettys_per_core` | D `1/1` | SDK AIV 核和每核队列参数，不是 K NUMA node、DP 或整机 Jetty 总数 |
| `timeout` | `600` | 秒，PD 控制操作与初始化等待预算；不替代 SDK 内部超时或启动脚本 startup_timeout |
| `diagnostics_requests / verify_kv` | 两端 `0/false` | 当前关闭诊断；字节验收时两端一起改为大于 0 / true，性能测量前恢复 |
| `multistream_overlap_shared_expert` | 两端 `true` | 共享专家多流重叠，默认 false；不控制 PD 或 Target 图开关 |
| `recompute_scheduler_enable` | D 字符串 `"True"` | 保留成功脚本的 D 调度兼容入口，不泛化为其他布尔字段都接受字符串 |
| `finegrained_tp_config.lmhead_tensor_parallel_size` | D `8` | 沿 DP rank 建立独立 LMHead 分组，不把 D 主模型 TP1 改成 TP8 |
| `enable_mlapo` | D `true` | 允许受支持的融合注意力预处理；实际使用仍检查模型和量化条件 |
| `enable_dsa_cp / c8_enable_reshape_optim` | 两端 `false/false` | 当前 TP0-only 与缓存布局约束，不能随意省略 C8 项或改为 true |
| `kv-cache-dtype / indexer_kv_dtype` | 两端 `fp8/fp8` | 主 KV 与 Indexer 的独立类型配置，当前 TP0-only 要求匹配 |
| `block-size / num_speculative_tokens` | 两端 `128/3` | 当前 PD 布局和原生 MTP 约束，不仅是性能调节参数 |

`control_backlog` 未在用户 D 脚本显式填写，本分支默认 128，作用于 D 控制监听队列，不是 K 队列数。
所有 P/D CLI、环境变量、图参数默认值及单位的逐项解释见 [双机参数说明](kvaf_pd_2nodes_configuration.md)。
该文档的参数语义可复用，但旧单 K 地址和路由不用于本方案。

## 8. 启动、网络及 Proxy

启动顺序为：**K1 两服务、K2 两服务 → K2 唯一协调器 → D → P → Proxy → 实际请求**。
前两台 K 可以分别准备；先确认四个 endpoint 和协调器都 ready，再启动 D。

| 发起端 | 必须可达的目标 |
| --- | --- |
| P 所有参与源注册的 worker | D `.24:31000`～`31007` 及 SDK ticket 返回的动态 HBM 端口 |
| P/D 的对应 SDK 通路 | K1 `.71:28997/28999`、K2 `.73:28997/28999` 的实际路由及 UB 数据面 |
| 全部 D worker | 协调器 `.73:28998` |
| Proxy | P `.12:8100`、D `.24:8200` |
| 原生模型进程 | 相应 HCCL/DP 通信端口与数据面，不只上述业务端口 |

D 的控制端口仍由 `31000 + DP rank × TP size + TP rank` 计算，本次为 31000～31007。
迁移 K 不改变这些 P→D 控制地址；迁移协调器也不改变 Proxy 的 P/D HTTP 地址。

P/D ready 后，在 12 号容器检查健康并运行原有 Proxy：

```bash
curl --fail --noproxy '*' http://141.62.38.12:8100/health
curl --fail --noproxy '*' http://141.62.38.24:8200/health

KVAF_PD_PYTHON=/usr/local/python3.12.13/bin/python3 \
bash /home/c00612283/vllm-ascend-v0.30.0rc2/scripts/kvaf_pd_proxy.sh \
  --host 141.62.38.12 --port 8000 \
  --prefiller-hosts 141.62.38.12 --prefiller-ports 8100 \
  --decoder-hosts 141.62.38.24 --decoder-ports 8200
```

健康检查失败时先处理 P/D，不继续启动 Proxy。Proxy 前台运行，另开终端发送请求：

```bash
curl --fail --noproxy '*' --connect-timeout 3 --max-time 180 \
  http://141.62.38.12:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm","messages":[{"role":"user","content":"请回复测试成功"}],"temperature":0,"max_tokens":32,"stream":false}'
```

HTTP 短请求只证明对应路径连通，不能覆盖全部 D worker 和四个 K endpoint。
继续用固定长请求覆盖满 block DDR、Indexer/scale、tail、MTP 及全部 D 路由，并和非 PD 基线比较精度。

## 9. 日志、验收与恢复

| 主机 / 组件 | 日志及 PID |
| --- | --- |
| K1 `.71` | `/home/c00612283/pd-pool-logs/k4/时间_PID/socket-0.log`、`socket-1.log` 及对应 `.pid` |
| K2 `.73` | `/home/c00612283/pd-pool-logs/k0/时间_PID/socket-0.log`、`socket-1.log` 及对应 `.pid` |
| 协调器 K2 `.73` | `/home/c00612283/pd-pool-logs/coordinator/时间_PID/server.log`、`server.pid` |
| P / D | 各自框架仓库 `pd-logs/p或d/时间_PID/server.log`、plog 和 KVAF 原生日志 |

两台 K 各检查监听者；K1 应有两个 CPU 服务，K2 应另有唯一协调器：

```bash
netstat -lntp 2>/dev/null \
  | grep -E ':28997[[:space:]]|:28998[[:space:]]|:28999[[:space:]]'
ps -ww -eo pid,ppid,stat,etime,args \
  | grep -E '[k]vaf_cpu_service|[k]vaf_pd_init_coordinator.py'
```

确认 D 本轮日志中八个 `KVAF CPU route` 与第 2 节一致，再按物理卡与 endpoint 关联日志。
P 只报 `Remote Offload: Probe rejected` 时，应查实际接收该源注册的 K 服务，不只搜协调器。
在 K1、K2 分别执行并根据本轮目录过滤，避免把历史记录混入：

```bash
grep -R -n -i -B 10 -A 5 \
  -E 'probe|remote_source|rejected|retired' \
  /home/c00612283/pd-pool-logs
```

正常日志的 `error_suspend=1` 会匹配宽泛的 error 搜索，不代表通道失败。
没有 probe 关键字也不代表探测通过，应核对对应进程 stdout/stderr、本次日志路径和 SDK 版本。
`CHANNEL_CREATED` 或 `SERVICE_READY` 不能替代源 MR 探测、Pull 完成、Decode 消费及字节校验。

如需字节验收，排空并重启 P/D，将两端 `diagnostics_requests=1`、`verify_kv=true` 后发送固定长请求，
用现有 `scripts/check_kvaf_pd_diagnostics.py --require-ddr --require-compute` 审计同轮 P/D 日志；
性能测量前两端恢复 `0/false`。NUMA 参数和同端口多 region 的硬件效果按实际机器验证，不由文档推定。

恢复时停止新请求和 Proxy，排空并停止 P、D，确认相关初始化和 DMA 静止，再处理属于本次部署的 K PID。
K 重启使对应旧 ticket 失效，不能让旧 P/D 继续运行；协调器失败门控也不会因延长 timeout 自动清除。
必要时在所有旧 initializer 已退出后重启唯一协调器，再按 D → P → Proxy 恢复。
不要在部分 K 服务超时后盲目重复运行脚本，也不使用全局 pkill 清理其他部署。

## 10. 源码与参数依据

- [双机参数逐项解释](kvaf_pd_2nodes_configuration.md)：CLI、图模式、MTP、Runtime 和环境变量。
- [PD 配置校验](../vllm_ascend/distributed/kv_transfer/kvaf/pd/config.py)：Connector 字段、TP0-only 及类型约束。
- [物理卡路由](../vllm_ascend/distributed/kv_transfer/kvaf/adapter.py)：cpu_addresses 优先级及未命中回退。
- [PD Runtime](../vllm_ascend/distributed/kv_transfer/kvaf/pd/runtime.py)：按实际 K endpoint 初始化与 DDR 容量。
- [协调器](../vllm_ascend/distributed/kv_transfer/kvaf/pd/initialization.py)：独立 endpoint 门控及失败处理。

本次只归档提供的配置并核对脚本结构，不将未运行的四机硬件验收写为已通过。
