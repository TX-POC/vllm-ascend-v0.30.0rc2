#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/config.sh"
[[ $# == 1 && $1 =~ ^[0-7]$ ]] || { echo "Usage: $0 K_NODE_INDEX(0..7)" >&2; exit 2; }
node=$1
command -v numactl >/dev/null
test -x "$K_BUILD/runtime/kvaf_cpu_service"
mkdir -p "$LOG_ROOT/k$node"
for socket in 0 1; do
    log="$LOG_ROOT/k$node/socket$socket-$(date +%Y%m%d_%H%M%S)-$$.log"
    nohup numactl --cpunodebind="${K_NUMA_NODES[socket]}" --membind="${K_NUMA_NODES[socket]}" \
      "$K_BUILD/runtime/kvaf_cpu_service" --listen "${K_HOSTS[node]}" \
      --port "${K_PORTS[socket]}" --memory-page-mode default > "$log" 2>&1 < /dev/null &
    echo "K_NODE=$node SOCKET=$socket PID=$! LOG=$log"
done
