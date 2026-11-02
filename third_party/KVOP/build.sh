#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${2:-python3}"
NANOVLLM_A5_OPS_PYTHON="$python_bin" NANOVLLM_A5_OPS_BUILD_JOBS="${1:-$(nproc)}" \
    bash "$root/original/build_lidu_fused_batch_get_prepare_mtp.sh"
"$python_bin" -m pip install --no-deps --no-build-isolation -e "$root"
echo "Source $root/env.sh before starting the service. KVAF does not need rebuilding."
