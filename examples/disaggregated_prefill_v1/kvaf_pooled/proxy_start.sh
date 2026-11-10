#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/config.sh"
mkdir -p "$LOG_ROOT/proxy"
log="$LOG_ROOT/proxy/$(date +%Y%m%d_%H%M%S)-$$.log"
KVAF_PD_PYTHON="$PYTHON" nohup bash "$REPO/scripts/kvaf_pd_proxy.sh" \
  --host "${P_HOSTS[0]}" --port 8000 \
  --prefiller-hosts "${P_HOSTS[0]}" --prefiller-ports 8100 \
  --decoder-hosts "${D_HOSTS[0]}" --decoder-ports 8200 > "$log" 2>&1 < /dev/null &
echo "PROXY_PID=$! LOG=$log"
