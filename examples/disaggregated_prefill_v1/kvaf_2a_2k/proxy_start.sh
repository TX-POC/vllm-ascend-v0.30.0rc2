#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

[[ $# == 0 ]] || { echo "Usage: $0" >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export KVAF_PD_PYTHON=${KVAF_PD_PYTHON:-/usr/local/python3.12.13/bin/python3}
for backend in 141.62.38.12:8100 141.62.38.24:8200; do
    curl --fail --silent --show-error --noproxy '*' --max-time 5 "http://$backend/health"
done

dir="$root/pd-logs/proxy/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$dir"
printf 'Proxy log: %s/server.log\n' "$dir"
exec > >(tee "$dir/server.log") 2>&1
exec bash "$root/scripts/kvaf_pd_proxy.sh" --host 141.62.38.12 --port 8000 \
    --prefiller-hosts 141.62.38.12 --prefiller-ports 8100 \
    --decoder-hosts 141.62.38.24 --decoder-ports 8200
