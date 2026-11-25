#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build checked PD bindings with the release's pinned external KVIP and KVAF.
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "Usage: $0 KVAF_CHECKOUT KVIP_CHECKOUT" >&2
    exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
test -d "$1"
KVAF_PD_SDK_ROOT=$(realpath "$1")
export KVAF_PD_SDK_ROOT
test -d "$2"
KVAF_PD_KVIP_ROOT=$(realpath "$2")
export KVAF_PD_KVIP_ROOT
export KVAF_PD_PYTHON=${KVAF_PD_PYTHON:-python3}
exec bash "$root/scripts/build_kvaf_pd_a.sh" --sdk-only
