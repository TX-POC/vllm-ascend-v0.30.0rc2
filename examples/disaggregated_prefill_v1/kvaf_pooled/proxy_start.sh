#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 编辑本文件中的 IP、网卡、路径和原生 JSON，然后不带参数执行。
set -euo pipefail
[[ $# == 0 ]] || { echo "Edit this script, then run it without arguments" >&2; exit 2; }

# 在 P0 容器执行；Proxy 连接两个集群的主入口，不配置 headless 节点。
cd /home/c00612283/vllm-ascend-v0.30.0rc2
unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY all_proxy ALL_PROXY
local_ip=141.62.38.12
port=8000
python_bin=/usr/local/python3.12.13/bin/python3
startup_timeout=180
PROXY=/home/c00612283/vllm-ascend-v0.30.0rc2/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py
test -x "$python_bin"
test -f "$PROXY"
dir="$PWD/pd-logs/proxy/$(date +%Y%m%d_%H%M%S)_$$"
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
nohup "$python_bin" "$PROXY" \
  --host "$local_ip" --port "$port" \
  --prefiller-hosts 141.62.38.12 --prefiller-ports 8100 \
  --decoder-hosts 141.62.38.24 --decoder-ports 8200 > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$dir/server.pid"
printf 'PROXY_LAUNCHED pid=%s log=%s\n' "$pid" "$log"
wait_service "$pid" "$local_ip" "$port" "$log" "Uvicorn running on http://$local_ip:$port" \
  "http://$local_ip:$port/healthcheck"
