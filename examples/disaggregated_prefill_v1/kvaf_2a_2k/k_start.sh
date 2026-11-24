#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

case "${1:-}" in
    k1) local_ip=141.62.38.71 ;;
    k2) local_ip=141.62.38.73 ;;
    *) echo "Usage: $0 k1|k2" >&2; exit 2 ;;
esac
[[ $# == 1 ]] || { echo "Usage: $0 k1|k2" >&2; exit 2; }

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../../.." && pwd)
kvaf_source=${KVAF_PD_SDK_ROOT:-/home/c00612283/pd-main-deps/KVAF}
build=${KVAF_PD_K_BUILD_DIR:-$kvaf_source/build-hostcpu-k}
python_bin=${KVAF_PD_K_PYTHON:-python3}
log_root=${KVAF_PD_K_LOG_DIR:-/home/c00612283/pd-pool-logs}
numa_nodes=("${KVAF_PD_NUMA0:-0}" "${KVAF_PD_NUMA1:-1}")
ports=(28997 28999)

command -v numactl >/dev/null
test -x "$build/runtime/kvaf_cpu_service"
for node in "${numa_nodes[@]}"; do
    numactl --cpunodebind="$node" --membind="$node" true
done
dir="$log_root/$1/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$dir"

pids=()
for socket in 0 1; do
    log="$dir/socket-$socket.log"
    nohup numactl --cpunodebind="${numa_nodes[socket]}" --membind="${numa_nodes[socket]}" \
        bash "$root/scripts/kvaf_pd_k.sh" start "$build" "$local_ip" --port "${ports[socket]}" \
        > "$log" 2>&1 < /dev/null &
    pids+=("$!")
    printf '%s\n' "$!" > "$dir/socket-$socket.pid"
    printf 'K_LAUNCHED pid=%s log=%s\n' "$!" "$log"
done

result=0
for socket in 0 1; do
    log="$dir/socket-$socket.log"
    port=${ports[socket]}
    if ! "$python_bin" "$here/wait_listener.py" "${pids[socket]}" "$local_ip" "$port" "$log" \
        "[KVAF_CPU] LISTEN address=$local_ip port=$port" 60; then
        tail -n 20 "$log" >&2
        result=1
    fi
done
exit "$result"
