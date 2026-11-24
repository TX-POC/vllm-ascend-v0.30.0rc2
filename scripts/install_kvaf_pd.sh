#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build checked PD bindings with the release's pinned external KVIP and KVAF.
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "Usage: $0 KVAF_CHECKOUT KVIP_CHECKOUT" >&2
    exit 2
fi

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
kvaf_source=$(realpath "$1")
kvip_source=$(realpath "$2")
python_bin=${KVAF_PD_PYTHON:-python3}
patch="$root/scripts/patches/kvaf-pd-checked-raw-read.patch"
build_dir="$kvaf_source/build-npu"

# Prepare checks the matching KVAF revision and applies KVIP's existing patches.
PYTHON="$python_bin" bash "$root/scripts/install_kvip.sh" --main-kvaf "$kvip_source" "$kvaf_source" --source-only
if git -C "$kvaf_source" apply --check "$patch" 2>/dev/null; then
    git -C "$kvaf_source" apply "$patch"
elif ! git -C "$kvaf_source" apply --reverse --check "$patch" 2>/dev/null; then
    echo "KVAF checkout does not match the checked raw READ patch" >&2
    exit 1
fi
PYTHON="$python_bin" bash "$root/scripts/install_kvip.sh" --main-kvaf "$kvip_source" "$kvaf_source"

test -f "$build_dir/bin/kvaf_execute_batch_get_raw_addr_kernels.o"
PYTHONPATH="$build_dir/python${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" \
    "$kvaf_source/bindings/verify_python_binding.py" --require-npu --require-offload --require-raw
PYTHONPATH="$build_dir/python${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -c \
    'import kvaf; assert kvaf.NpuPullOptions(verify_completion=True).verify_completion; print("PD_SDK_IMPORT_OK", kvaf.__file__)'
echo "Use $build_dir as kvaf_config.build_dir and source $kvip_source/scripts/env.sh on P/D"
