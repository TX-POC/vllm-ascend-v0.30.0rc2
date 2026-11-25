#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
kvaf_source=${KVAF_PD_SDK_ROOT:-$(dirname "$root")/KVAF}
build=${KVAF_PD_K_BUILD_DIR:-$kvaf_source/build-hostcpu-k}
# shellcheck source=scripts/kvaf_dependency_versions.sh
source "$root/scripts/kvaf_dependency_versions.sh"
if [[ "$(git -C "$kvaf_source" rev-parse HEAD)" != "$kvaf_revision" ]]; then
    echo "Expected KVAF main checkout at $kvaf_revision" >&2
    exit 1
fi
exec bash "$root/scripts/kvaf_pd_k.sh" build "$kvaf_source" "$build" "$@"
