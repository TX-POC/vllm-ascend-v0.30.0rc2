#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "${KVAF_PD_PYTHON:-python3}" "$root/scripts/kvaf_pd_performance_proxy.py" --no-capture "$@"
