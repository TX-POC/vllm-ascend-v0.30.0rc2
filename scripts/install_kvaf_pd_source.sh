#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run in the existing matched torch/torch_npu/CANN container.
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "Usage: $0 VLLM_SOURCE_OR_- KVAF_CHECKOUT KVIP_CHECKOUT" >&2
    exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
test -d "$2"
KVAF_PD_SDK_ROOT=$(realpath "$2")
export KVAF_PD_SDK_ROOT
test -d "$3"
KVAF_PD_KVIP_ROOT=$(realpath "$3")
export KVAF_PD_KVIP_ROOT
export KVAF_PD_PYTHON=${KVAF_PD_PYTHON:-python3}
args=(--full)
if [[ "$1" != - ]]; then
    args+=(--vllm-source "$1")
fi
exec bash "$root/scripts/build_kvaf_pd_a.sh" "${args[@]}"
