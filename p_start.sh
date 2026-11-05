#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Single-host P: edit this section; no exported variables or arguments required.
set -euo pipefail
role=p
python_bin=/usr/local/python3.12.13/bin/python3
model=/home/models/glm5-mtp-4l
kvaf_build=/home/c00612283/KVAF-Devlop/build-pd-py312
devices=2,3
tp=2
dp=1
port=8100
rpc_port=13389
decode_workers=2  # Match d_start.sh's dp (D TP=1).
kv_port=31000
preferred_nic=enp34s0f1
local_ip=""  # Empty: detect IPv4 on preferred_nic, then the default route.
env_file=""  # Optional vendor environment script; CANN/ATB may already be loaded.
diagnostics_requests=0  # Set to 2 on P/D for bounded diagnostic logs.
verify_kv=false         # Set true on BOTH P/D for byte checks (affects performance).
extra_args=()           # Additional native vllm serve arguments, as a Bash array.

# ---- Automatic environment and native launch ----
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -gt 1 || ( $# == 1 && $1 != --dry-run ) ]]; then
    echo "Usage: ./${role}_start.sh [--dry-run]" >&2
    exit 2
fi
if [[ -n $env_file ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$env_file"
    set -u
fi
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY ftp_proxy FTP_PROXY all_proxy ALL_PROXY
export KVAF_PD_PYTHON="$python_bin"
test -x "$python_bin" || { echo "Python not executable: $python_bin" >&2; exit 1; }

# stdlib-only IPv4 discovery: works even without ip/ss in the container.
network=$("$python_bin" - "$preferred_nic" "$local_ip" <<'PY'
import fcntl
import socket
import struct
import sys
from pathlib import Path

preferred, requested = sys.argv[1:]
addresses = {}
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    for _, name in socket.if_nameindex():
        if name == "lo":
            continue
        try:
            address = socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", name.encode()[:15]))[20:24])
        except OSError:
            continue
        addresses[name] = address
if requested:
    matches = [name for name, ip in addresses.items() if ip == requested]
    nic = preferred if preferred in matches else (matches[0] if len(matches) == 1 else None)
else:
    routes = []
    for row in Path("/proc/net/route").read_text().splitlines()[1:]:
        fields = row.split()
        if fields[1] == "00000000" and int(fields[3], 16) & 1 and fields[0] in addresses:
            routes.append((int(fields[6]), fields[0]))
    nic = preferred if preferred in addresses else (min(routes)[1] if routes else None)
if not nic:
    raise SystemExit("Cannot select local IPv4/NIC; set preferred_nic/local_ip in the entry script")
print(nic, addresses[nic])
PY
)
read -r nic local_ip <<< "$network"

configs=$("$python_bin" - "$role" "$local_ip" "$kvaf_build" "$kv_port" "$decode_workers" \
    "${k_host:-$local_ip}" "${k_port:-28997}" "${ddr_size_gb:-1}" "$diagnostics_requests" "$verify_kv" <<'PY'
import json
import sys

role, ip, build, port, workers, k_host, k_port, ddr, count, verify = sys.argv[1:]
if verify not in ("true", "false"):
    raise SystemExit("verify_kv must be true or false")
extra = {"diagnostics_requests": int(count), "verify_kv": verify == "true"}
if role == "p":
    extra["decode_endpoints"] = [f"{ip}:{int(port) + i}" for i in range(int(workers))]
transfer = {"kv_connector": "PDKVoffloadConnector",
            "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kvaf.pd.connector",
            "kv_role": "kv_producer" if role == "p" else "kv_consumer",
            "kv_port": int(port), "kv_connector_extra_config": extra}
backend = {"enabled": role == "d", "build_dir": build}
if role == "d":
    backend.update(cpu_address=f"{k_host or ip}:{k_port}", ddr_size_gb=float(ddr))
print(json.dumps(transfer))
print(json.dumps({"enable_cpu_binding": True, "multistream_overlap_shared_expert": True,
                  "c8_enable_reshape_optim": False, "kvaf_config": backend}))
PY
)
mapfile -t json_configs <<< "$configs"
args=("$model" --port "$port" --tensor-parallel-size "$tp"
    --data-parallel-size "$dp" --data-parallel-size-local "$dp"
    --data-parallel-address "$local_ip" --data-parallel-rpc-port "$rpc_port"
    --max-num-seqs 2 --max-model-len 8192 --max-num-batched-tokens 4096
    --gpu-memory-utilization 0.85
    --kv-transfer-config "${json_configs[0]}" --additional-config "${json_configs[1]}"
    "${extra_args[@]}")
if [[ ${1:-} == --dry-run ]]; then
    printf 'Role=%s IP=%s NIC=%s devices=%s TP=%s DP=%s\n' "$role" "$local_ip" "$nic" "$devices" "$tp" "$dp"
    exec bash "$root/scripts/start_vllm_pd.sh" --dry-run "$role" "$local_ip" "$nic" -- "${args[@]}"
fi
test -d "$model" || { echo "Model directory missing: $model" >&2; exit 1; }
test -d "$kvaf_build/python/kvaf" || { echo "Build KVAF Python binding first: $kvaf_build/python/kvaf" >&2; exit 1; }
stamp=$(date +%Y%m%d_%H%M%S)_$$
log_dir="$root/pd-logs/$role/$stamp"
mkdir -p "$log_dir/plog"
export ASCEND_PROCESS_LOG_PATH="$log_dir/plog"
export PROMETHEUS_MULTIPROC_DIR="/dev/shm/vllm_metrics_pd_${role}_$stamp"
export ASCEND_RT_VISIBLE_DEVICES="$devices"
export OMP_NUM_THREADS=10
printf 'Starting %s: IP=%s NIC=%s devices=%s TP=%s DP=%s HTTP=%s\nLog: %s/server.log\n' \
    "$role" "$local_ip" "$nic" "$devices" "$tp" "$dp" "$port" "$log_dir"
exec > >(tee "$log_dir/server.log") 2>&1
exec bash "$root/scripts/start_vllm_pd.sh" "$role" "$local_ip" "$nic" -- "${args[@]}"
