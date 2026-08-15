#!/usr/bin/env bash
set -euo pipefail

OPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${OPS_ROOT}/../../.." && pwd)"
PYTHON_BIN="${NANOVLLM_A5_OPS_PYTHON:-python3}"

# Compatibility entry point only. A5 DSA no longer installs a standalone
# nanovllm_dsa_a5._C extension. The repository root setup.py/build_aclnn.sh
# builds the A5 OPP package and vllm_ascend_C in one installation flow.
cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m pip install -e . --no-build-isolation

cd /tmp
"${PYTHON_BIN}" - <<'PY'
import importlib

import torch
from vllm_ascend.a5_dsa_registration import require_a5_dsa_registration

native_required = (
    "kvaf_batch_get_prepare_out",
    "packed_scatter_copy_out",
    "full_block_dump",
    "packed_full_block_dump",
)
require_a5_dsa_registration(native_required)

# C8 LIDU implementations and sparse attention are registered by the Python
# extension layer after the schemas embedded in vllm_ascend_C are available.
# Keep them out of the native-only check above, then validate the complete
# runtime registration surface below.
importlib.import_module("vllm_ascend.dsa_offload.a5_extension")

python_registered = (
    "lidu_decode_update_c8_out",
    "sparse_and_tail_attention_c8",
)
required = (*native_required, *python_registered)
for name in required:
    ok = hasattr(torch.ops.nanovllm_dsa, name)
    print(name, ok)
    if not ok:
        raise RuntimeError(f"A5 DSA operator is missing: {name}")

print("A5 DSA root vllm_ascend_C installation verification: PASS")
PY

echo "[nanovllm_a5_ops] root vLLM-Ascend install complete"
