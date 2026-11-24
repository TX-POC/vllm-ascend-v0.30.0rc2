#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run in the existing matched torch/torch_npu/CANN container.
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "Usage: $0 VLLM_SOURCE_OR_- KVAF_CHECKOUT KVIP_CHECKOUT" >&2
    exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
vllm_source=$1
kvaf_source=$(realpath "$2")
kvip_source=$(realpath "$3")
kvaf_build="$kvaf_source/build-npu"
python_bin=${KVAF_PD_PYTHON:-python3}
if [[ $vllm_source != - ]]; then
    vllm_source=$(realpath "$vllm_source")
    test -f "$vllm_source/pyproject.toml"
fi
test -f "$kvaf_source/docs/connector_pull_api_design.md"
"$python_bin" -c 'import sys, torch, torch_npu; print(sys.executable, torch.__version__, torch_npu.__version__)'
"$python_bin" -m pip install 'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8' 'setuptools-rust>=1.9.0' 'pybind11>=2.13'
if [[ $vllm_source != - ]]; then
    VLLM_TARGET_DEVICE=empty "$python_bin" -m pip install --no-deps --no-build-isolation -e "$vllm_source"
else
    "$python_bin" -c 'import vllm; print("Using installed vLLM", vllm.__version__)'
fi
"$python_bin" -m pip install --no-deps --no-build-isolation -e "$root"
KVAF_PD_PYTHON="$python_bin" bash "$root/scripts/install_kvaf_pd.sh" "$kvaf_source" "$kvip_source"
PYTHONPATH="$kvaf_build/python${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -c \
    'import vllm, vllm_ascend, kvaf; from vllm_ascend.distributed.kv_transfer.kvaf.pd.connector import PDKVoffloadConnector; print("PD_SOURCE_INSTALL_OK", vllm.__version__, kvaf.__file__)'
