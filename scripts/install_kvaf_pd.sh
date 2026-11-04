#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the user's existing cpu-pull-hbm-reference checkout; no repository download.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash scripts/install_kvaf_pd.sh KVAF_SOURCE [BUILD_DIR]" >&2
    exit 2
fi

KVAF_PD_SOURCE=$(realpath "$1")
KVAF_PD_BUILD=$(realpath -m "${2:-$KVAF_PD_SOURCE/build-pd}")
KVAF_PD_PYTHON=${KVAF_PD_PYTHON:-python3}
KVAF_PD_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
KVAF_PD_PATCH="$KVAF_PD_SCRIPT_DIR/patches/kvaf-pd-borrow-device-ra.patch"

test -f "$KVAF_PD_SOURCE/CMakeLists.txt"
test -f "$KVAF_PD_SOURCE/docs/connector_pull_api_design.md"
if git -C "$KVAF_PD_SOURCE" apply --check "$KVAF_PD_PATCH" 2>/dev/null; then
    git -C "$KVAF_PD_SOURCE" apply "$KVAF_PD_PATCH"
elif git -C "$KVAF_PD_SOURCE" apply --reverse --check "$KVAF_PD_PATCH" 2>/dev/null; then
    echo "PD device-RA borrowing patch already applied"
else
    echo "SDK source does not match the cpu-pull-hbm-reference PD patch; inspect before building" >&2
    exit 1
fi

cmake -S "$KVAF_PD_SOURCE" -B "$KVAF_PD_BUILD" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DKVAF_ENABLE_CPU_OFFLOAD=ON \
    -DKVAF_ENABLE_AIV_BATCH_GET=ON \
    -DKVAF_ENABLE_AIV_BATCH_PUT=ON \
    -DKVAF_BUILD_PYTHON=ON \
    -DKVAF_BUILD_REMOTE_CPU_SERVICE=OFF \
    -DBUILD_TESTING=OFF \
    -DPython_EXECUTABLE="$(command -v "$KVAF_PD_PYTHON")"
cmake --build "$KVAF_PD_BUILD" -j "${KVAF_PD_BUILD_JOBS:-8}"
PYTHONPATH="$KVAF_PD_BUILD/python${PYTHONPATH:+:$PYTHONPATH}" "$KVAF_PD_PYTHON" -c \
    'import kvaf; assert kvaf._native.reuse_device_ra_supported; assert hasattr(kvaf.NpuRuntime, "submit_cpu_pull"); assert hasattr(kvaf.NpuRuntime, "batch_get_raw_addr"); print("PD_SDK_IMPORT_OK", kvaf.__file__)'
echo "Set additional_config.kvaf_config.build_dir to $KVAF_PD_BUILD and prepend $KVAF_PD_BUILD/python to PYTHONPATH"
