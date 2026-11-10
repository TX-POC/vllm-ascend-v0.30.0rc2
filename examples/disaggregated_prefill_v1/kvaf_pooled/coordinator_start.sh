#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/config.sh"
mkdir -p "$LOG_ROOT/coordinator"
log="$LOG_ROOT/coordinator/$(date +%Y%m%d_%H%M%S)-$$.log"
nohup "$HOST_PYTHON" -u "$REPO/scripts/kvaf_pd_init_coordinator.py" \
  --host "${K_HOSTS[0]}" --port 28998 > "$log" 2>&1 < /dev/null &
echo "COORD_PID=$! LOG=$log"
