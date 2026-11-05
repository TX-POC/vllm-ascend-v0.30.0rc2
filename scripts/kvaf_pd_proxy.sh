#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
exec "${KVAF_PD_PYTHON:-python3}" "$root/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py" "$@"
