#!/usr/bin/env bash
# Run inside the container created by start_vllm_container.sh.
# Usage: bash /vllm-workspace/vllm-ascend/scripts/install_vllm_in_container.sh

# Keep nounset off while sourcing CANN's environment scripts.
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 -m pip install 'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8' 'setuptools-rust>=1.9.0' 'pybind11>=2.13'
# Keep the image's matched torch/torch-npu/CANN dependencies.
VLLM_TARGET_DEVICE=empty python3 -m pip install --no-deps --no-build-isolation -e /vllm-workspace/vllm
python3 -m pip install --no-deps --no-build-isolation -e "$repo_root"

echo 'Editable vLLM and vLLM-Ascend installation complete.'
