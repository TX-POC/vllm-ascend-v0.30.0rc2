#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail

usage() {
    echo "Usage: bash scripts/build_kvaf_pd_a.sh (--full|--sdk-only) [--remove-stale-headers]"
    echo "Requires the matching vLLM, torch, torch_npu and ATB environment inside the A-node container."
    echo "Applies the pinned KVAF/KVIP compatibility patches to the selected checkouts."
}

fail() { printf '%s\n' "$*" >&2; exit 1; }

mode=${1:-}
case "$mode" in
    --full|--sdk-only) shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
remove_stale=false
if [[ ${1:-} == --remove-stale-headers ]]; then
    remove_stale=true
    shift
fi
[[ $# == 0 ]] || { usage >&2; exit 2; }

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
base=$(dirname "$root")
kvaf=${KVAF_PD_SDK_ROOT:-$base/KVAF}
kvip=${KVAF_PD_KVIP_ROOT:-$base/KVIP}
python_bin=${KVAF_PD_PYTHON:-/usr/local/python3.12.13/bin/python3}
jobs=${KVAF_PD_BUILD_JOBS:-2}
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || fail "KVAF_PD_BUILD_JOBS must be a positive integer"
export ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.2.0}
if [[ -f "$ASCEND_CANN_PACKAGE_PATH/set_env.sh" ]]; then
    source "$ASCEND_CANN_PACKAGE_PATH/set_env.sh"
elif [[ -f "$ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash" ]]; then
    source "$ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash"
fi
test -r "$ASCEND_CANN_PACKAGE_PATH/include/acl/acl.h" || fail "Missing CANN ACL headers under $ASCEND_CANN_PACKAGE_PATH"
cd "$root"
"$python_bin" -c 'import torch, torch_npu, vllm'

# The official archive is optional when the system or an explicit include directory provides Boost.
boost_header=boost/property_tree/json_parser.hpp
if [[ -n ${BOOST_INCLUDEDIR:-} ]]; then
    include="$BOOST_INCLUDEDIR"
elif [[ -n ${BOOST_ROOT:-} ]]; then
    include="$BOOST_ROOT/include"
elif [[ -r "/usr/include/$boost_header" ]]; then
    include=/usr/include
else
    include="$base/build-deps/boost-1.83/include"
    if [[ ! -r "$include/$boost_header" ]]; then
        archive=${KVAF_PD_BOOST_ARCHIVE:-$base/boost_1_83_0.tar.gz}
        [[ -r "$archive" ]] || fail "Set BOOST_INCLUDEDIR or provide the official Boost 1.83.0 archive at $archive"
        printf '%s  %s\n' c0685b68dd44cc46574cce86c4e17c0f611b15e195be9848dfd0769a0a207628 "$archive" | sha256sum -c -
        mkdir -p "$include"
        tar -xzf "$archive" -C "$include" --strip-components=1 boost_1_83_0/boost
    fi
fi
[[ -r "$include/$boost_header" && -r "$include/boost/version.hpp" ]] || fail "Boost headers missing under $include"

checkout() {
    local path=$1 url=$2 expected=$3 actual
    if [[ ! -e "$path" ]]; then
        git clone --no-checkout --branch main "$url" "$path"
        git -C "$path" switch --detach "$expected"
    fi
    actual=$(git -C "$path" rev-parse HEAD)
    [[ "$actual" == "$expected" ]] || fail "$path has $actual; expected $expected"
}

checkout "$kvaf" https://github.com/TX-POC/KVAF-Develop.git 95a14f1cfaa6d7c76666f5d692df4c4f182360ec
checkout "$kvip" https://github.com/TX-POC/KVAF-KVIP.git d6cfd392f722dd3f96ef6e05175f503aac511dbb
kvaf=$(realpath "$kvaf")
kvip=$(realpath "$kvip")
build="$kvaf/build-npu"
export PYTHON="$python_bin"

if [[ "$remove_stale" == true ]]; then
    remove_untracked_header() {
        local file=$1 patch=$2
        shift 2
        [[ -f "$kvaf/$file" ]] || return 0
        if git -C "$kvaf" apply "$@" --reverse --check "$patch" 2>/dev/null; then
            return 0
        fi
        [[ "$(git -C "$kvaf" status --porcelain --untracked-files=all -- "$file")" == "?? $file" ]] ||
            fail "Refusing to remove tracked file $file"
        rm -- "$kvaf/$file"
        printf 'Removed stale patch header %s\n' "$file"
    }
    remove_untracked_header access_operator/algorithms/get/direct/urma_aiv/include/kvaf_batch_get_device.h \
        "$kvip/scripts/patches/kvaf-main-fused-device.patch"
    remove_untracked_header access_operator/include/ubmem/fused_batch_get_device.h \
        "$kvip/patches/kvaf-ubmem-device-get.patch" '--include=access_operator/include/ubmem/*'
fi

bash scripts/install_kvip.sh --main-kvaf "$kvip" "$kvaf" --source-only
patch="$root/scripts/patches/kvaf-pd-checked-raw-read.patch"
if ! git -C "$kvaf" apply --reverse --check "$patch" 2>/dev/null; then
    git -C "$kvaf" apply --check "$patch"
    git -C "$kvaf" apply "$patch"
fi

if [[ "$mode" == --full ]]; then
    "$python_bin" -m pip install 'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8' 'setuptools-rust>=1.9.0' 'pybind11>=2.13'
    "$python_bin" -m pip install --no-deps --no-build-isolation -e "$root"
else
    "$python_bin" -I -c 'import importlib.util, pathlib, sys
path = pathlib.Path(importlib.util.find_spec("vllm_ascend").origin).resolve()
assert path.is_relative_to(pathlib.Path(sys.argv[1]).resolve()), f"Wrong vLLM-Ascend install: {path}; use --full"' "$root"
fi

# Match the unified NPU build without AICPU; omit the incompatible Host test targets.
pybind_dir=$("$python_bin" -m pybind11 --cmakedir)
cmake -S "$kvaf" -B "$build" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF \
    -DKVAF_BUILD_PYTHON=ON -DKVAF_BUILD_REMOTE_CPU_SERVICE=OFF \
    -DKVAF_ENABLE_AIV_BATCH_GET=ON -DKVAF_ENABLE_AIV_BATCH_PUT=ON \
    -DKVAF_ENABLE_AICPU_URMA=OFF -DKVAF_ENABLE_AICPU_CPU_PUSH=OFF \
    -DKVAF_ENABLE_CPU_OFFLOAD=ON -DKVAF_ENABLE_NPU_PULL=ON \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_CANN_PACKAGE_PATH" -DBoost_INCLUDE_DIR="$include" \
    -DPython_EXECUTABLE="$python_bin" -DPython3_EXECUTABLE="$python_bin" -Dpybind11_DIR="$pybind_dir"
cmake --build "$build" --config RelWithDebInfo --parallel "$jobs"

bash scripts/install_kvip.sh --main-kvaf "$kvip" "$kvaf" --skip-kvaf-build
test -f "$build/bin/kvaf_execute_batch_get_raw_addr_kernels.o"
export PYTHONPATH="$build/python${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" "$kvaf/bindings/verify_python_binding.py" --require-npu --require-offload --require-raw
"$python_bin" -c 'from pathlib import Path
import sys
import kvaf
from vllm_ascend.distributed.kv_transfer.kvaf.pd.connector import PDKVoffloadConnector
assert Path(kvaf.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert kvaf.NpuPullOptions(verify_completion=True).verify_completion
print("PD_SOURCE_INSTALL_OK", kvaf.__file__)' "$build/python"
