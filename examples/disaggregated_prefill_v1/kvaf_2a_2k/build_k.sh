#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
kvaf_source=${KVAF_PD_SDK_ROOT:-/home/c00612283/pd-main-deps/KVAF}
build=${KVAF_PD_K_BUILD_DIR:-$kvaf_source/build-hostcpu-k}
kvaf_pin=95a14f1cfaa6d7c76666f5d692df4c4f182360ec
if [[ "$(git -C "$kvaf_source" rev-parse HEAD)" != "$kvaf_pin" ]]; then
    echo "Expected KVAF main checkout at $kvaf_pin" >&2
    exit 1
fi
exec bash "$root/scripts/kvaf_pd_k.sh" build "$kvaf_source" "$build" "$@"
