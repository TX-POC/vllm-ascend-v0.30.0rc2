#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"
kvaf_dir="$repo_dir/third_party/KVAF"
binary="$kvaf_dir/build-cpu/runtime/kvaf_cpu_service"
probe="$script_dir/cpu_page_probe.py"
mode="${1:-}"
cpu_nodes=(0 0 0 0 0 0 0 0)
memory_nodes=(2 2 3 3 0 0 1 1)
ub_devices=(udmac1d1e2 udmac1d1e3 udmac1d1e4 udmac1d1e5 udmac0d1e2 udmac0d1e3 udmac0d1e4 udmac0d1e5)

if [[ "$mode" == stop ]]; then
  log_dir="${2:?Usage: bash $0 stop /tmp/kvaf-pages-base-latest}"
  python3 - "$log_dir" "$binary" <<'PY_STOP'
import os
import signal
import sys
from pathlib import Path

logs = Path(sys.argv[1]).resolve()
binary = Path(sys.argv[2]).resolve()
for filename in ("pids", "monitor.pid"):
    if not (logs / filename).exists():
        continue
    for value in (logs / filename).read_text().split():
        pid = int(value)
        try:
            proc = Path("/proc") / str(pid)
            cmd = (proc / "cmdline").read_bytes().split(b"\0")
            owned = (proc / "exe").resolve() == binary if filename == "pids" else (
                str(logs).encode() in cmd and any(b"cpu_page_probe.py" in part for part in cmd)
            )
            if owned:
                os.kill(pid, signal.SIGTERM)
                print(f"Stopped PID {pid}")
            else:
                print(f"Skipped PID {pid}: no longer this run's process")
        except (FileNotFoundError, ProcessLookupError):
            pass
PY_STOP
  exit 0
fi

if [[ "$mode" != base && "$mode" != thp ]]; then
  echo "Usage: bash $0 base|thp" >&2
  exit 2
fi
for command in python3 numactl ss; do
  command -v "$command" >/dev/null
done
test -x "$binary"
test -f "$probe"
# Never stop an unrelated service occupying an existing port.
for i in "${!ub_devices[@]}"; do
  port=$((28880 + i))
  if [[ -n "$(ss -H -ltn "sport = :$port")" ]]; then
    echo "Port $port is occupied. Stop the previous CPU services before starting this run." >&2
    exit 1
  fi
done
if [[ -n "${3:-}" ]]; then
  IFS=',' read -r -a cpu_nodes <<< "$3"
  [[ "${#cpu_nodes[@]}" == 8 ]] || { echo 'Expected eight CPU nodes'; exit 2; }
  for node in "${cpu_nodes[@]}"; do
    [[ "$node" =~ ^[0-9]+$ && -d "/sys/devices/system/node/node$node" ]] || exit 2
  done
fi
if [[ -n "${2:-}" ]]; then
  mkdir -p "$(dirname "$2")"
  mkdir "$2"
  log_dir="$(cd "$2" && pwd)"
else
  log_dir="$(mktemp -d "/tmp/kvaf-pages-${mode}-$(date +%Y%m%d_%H%M%S)-XXXXXX")"
fi
ln -sfn "$log_dir" "/tmp/kvaf-pages-${mode}-latest"
pids=()
cleanup_failed_start() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  echo "Startup failed; this run's processes stopped. Logs: $log_dir" >&2
}
trap cleanup_failed_start ERR
printf 'NPU\tPORT\tUB_DEVICE\tCPU_NODE\tMEMORY_NODE\tPID\n' > "$log_dir/routes.tsv"
cd "$kvaf_dir"
for i in "${!ub_devices[@]}"; do
  port=$((28880 + i))
  nohup python3 "$probe" run --mode "$mode" -- \
    numactl --cpunodebind="${cpu_nodes[$i]}" --membind="${memory_nodes[$i]}" \
    "$binary" --listen 0.0.0.0 --port "$port" \
    --ub-device "${ub_devices[$i]}" --memory-page-mode thp \
    > "$log_dir/$port.log" 2>&1 < /dev/null &
  pid=$!
  pids+=("$pid")
  echo "$pid" >> "$log_dir/pids"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$i" "$port" "${ub_devices[$i]}" "${cpu_nodes[$i]}" "${memory_nodes[$i]}" "$pid" \
    | tee -a "$log_dir/routes.tsv"
done
for i in "${!pids[@]}"; do
  port=$((28880 + i))
  ready=0
  for ((attempt=0; attempt<30; attempt++)); do
    kill -0 "${pids[$i]}"
    if [[ -n "$(ss -H -ltn "sport = :$port")" ]]; then ready=1; break; fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    echo "Port $port did not become ready; inspect $log_dir/$port.log" >&2
    false
  fi
done
nohup python3 "$probe" watch --pids "${pids[@]}" --seconds 86400 --output "$log_dir" \
  > "$log_dir/monitor.log" 2>&1 < /dev/null &
echo "$!" > "$log_dir/monitor.pid"
trap - ERR
numactl --hardware > "$log_dir/numa.txt"
if command -v urma_admin >/dev/null; then
  urma_admin show > "$log_dir/urma.txt" 2>&1 || true
fi
printf 'All eight services ready: mode=%s\nLogs/PIDs: %s\nMemory summary: %s/summary.log\n' "$mode" "$log_dir" "$log_dir"
echo 'Memory summary updates every 10 seconds while the vLLM service uses DDR.'
