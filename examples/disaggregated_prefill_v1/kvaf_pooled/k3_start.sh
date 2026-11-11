#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 编辑本文件中的 IP、网卡、路径和原生 JSON，然后不带参数执行。
set -euo pipefail
[[ $# == 0 ]] || { echo "Edit this script, then run it without arguments" >&2; exit 2; }

# 在 K3 鲲鹏宿主机执行；先核对两个 CPU socket 的实际 NUMA 节点集合。
local_ip=141.62.38.29
numa_nodes=(0 1)
ports=(28997 28999)
K_BIN=/home/c00612283/KVAF-Devlop/build-hostcpu-k/runtime/kvaf_cpu_service
python_bin=$(command -v python3)
startup_timeout=60
command -v numactl >/dev/null
test -x "$K_BIN"
dir="/home/c00612283/pd-pool-logs/k3/$(date +%Y%m%d_%H%M%S)_$$"
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
