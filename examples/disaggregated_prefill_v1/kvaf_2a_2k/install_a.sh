#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

[[ $# == 0 ]] || { echo "Usage: $0" >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export KVAF_PD_PYTHON=${KVAF_PD_PYTHON:-/usr/local/python3.12.13/bin/python3}
kvaf_source=${KVAF_PD_SDK_ROOT:-/home/c00612283/pd-main-deps/KVAF}
kvip_source=${KVAF_PD_KVIP_ROOT:-/home/c00612283/pd-main-deps/KVIP}
test -d "$kvaf_source"
test -d "$kvip_source"
exec bash "$root/scripts/install_kvaf_pd_source.sh" - "$kvaf_source" "$kvip_source"
