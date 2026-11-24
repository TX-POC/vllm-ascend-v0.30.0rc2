#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

[[ $# == 0 ]] || { echo "Usage: $0" >&2; exit 2; }
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../../.." && pwd)
python_bin=${KVAF_PD_K_PYTHON:-python3}
log_root=${KVAF_PD_K_LOG_DIR:-/home/c00612283/pd-pool-logs}
local_ip=141.62.38.73
port=28998
dir="$log_root/coordinator/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$dir"
log="$dir/server.log"

nohup "$python_bin" -u "$root/scripts/kvaf_pd_init_coordinator.py" --host "$local_ip" --port "$port" \
    > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$dir/server.pid"
printf 'COORDINATOR_LAUNCHED pid=%s log=%s\n' "$pid" "$log"
if ! "$python_bin" "$here/wait_listener.py" "$pid" "$local_ip" "$port" "$log" \
    "KVAF_INIT_COORDINATOR_LISTEN address=$local_ip:$port" 20; then
    tail -n 20 "$log" >&2
    exit 1
fi
