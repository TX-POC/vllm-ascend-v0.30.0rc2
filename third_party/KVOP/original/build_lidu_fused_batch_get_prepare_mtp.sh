#!/usr/bin/env bash
# Build only A5LiduFusedBatchGetPrepareMtp and its standalone Torch registration.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${ROOT}/build"
OP_DEV="${BUILD_ROOT}/op_dev_staging"
GENERATED="${BUILD_ROOT}/custom_op"
LOCAL_OPP="${ROOT}/_custom_opp"
TORCH_EXTENSION="${ROOT}/torch_extension"

PYTHON_BIN="${NANOVLLM_A5_OPS_PYTHON:-python3}"
BUILD_JOBS="${NANOVLLM_A5_OPS_BUILD_JOBS:-$(nproc)}"
A5_SOC_VERSION_RAW="${A5_SOC_VERSION:-${SOC_VERSION:-ascend950}}"
case "${A5_SOC_VERSION_RAW,,}" in
    ascend950 | ascend950pr* | ascend950dt*) A5_SOC_VERSION="ascend950" ;;
    *)
        echo "[a5_lidu_fused_batch_get_prepare_mtp] ERROR: only Ascend 950 is supported; got ${A5_SOC_VERSION_RAW}." >&2
        exit 2
        ;;
esac

command -v msopgen >/dev/null 2>&1 || {
    echo "[a5_lidu_fused_batch_get_prepare_mtp] ERROR: msopgen is unavailable; source the CANN 9.1 environment first." >&2
    exit 2
}
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "[a5_lidu_fused_batch_get_prepare_mtp] ERROR: Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
}

echo "[a5_lidu_fused_batch_get_prepare_mtp] building A5LiduFusedBatchGetPrepareMtp only"
echo "[a5_lidu_fused_batch_get_prepare_mtp] target=${A5_SOC_VERSION} jobs=${BUILD_JOBS}"

rm -rf "${GENERATED}" "${OP_DEV}" "${LOCAL_OPP}"
mkdir -p "${OP_DEV}/op_host" "${OP_DEV}/op_kernel"
cp -f "${ROOT}/ops.json" "${OP_DEV}/ops.json"
cp -f "${ROOT}/op_host/a5_lidu_fused_batch_get_prepare_mtp.cpp" "${OP_DEV}/op_host/"
cp -f "${ROOT}/op_kernel/a5_lidu_fused_batch_get_prepare_mtp.cpp" "${OP_DEV}/op_kernel/"
cp -f "${ROOT}/op_kernel/a5_lidu_fused_batch_get_prepare_mtp_tiling.h" "${OP_DEV}/op_kernel/"
cp -f "${ROOT}/op_kernel/a5_lidu_fused_batch_get_prepare_mtp_vector.h" "${OP_DEV}/op_kernel/"

msopgen gen \
    -i "${OP_DEV}/ops.json" \
    -f aclnn \
    -c "ai_core-${A5_SOC_VERSION}" \
    -lan cpp \
    -op A5LiduFusedBatchGetPrepareMtp \
    -out "${GENERATED}"

cp -a "${OP_DEV}/op_host/." "${GENERATED}/op_host/"
cp -a "${OP_DEV}/op_kernel/." "${GENERATED}/op_kernel/"

pushd "${GENERATED}" >/dev/null
OPS_CPU_NUMBER="${BUILD_JOBS}" bash build.sh
popd >/dev/null

RUN_PKG="$(find "${GENERATED}/build_out" -maxdepth 1 -type f -name '*.run' | head -n 1)"
[[ -n "${RUN_PKG}" ]] || {
    echo "[a5_lidu_fused_batch_get_prepare_mtp] ERROR: msopgen did not produce a .run package." >&2
    exit 1
}
mkdir -p "${LOCAL_OPP}"
chmod +x "${RUN_PKG}"
"${RUN_PKG}" --quiet --install-path="${LOCAL_OPP}"

mapfile -t OPAPI_LIBS < <(
    find "${LOCAL_OPP}/vendors" -type f -path '*/op_api/lib/libcust_opapi.so'
)
[[ "${#OPAPI_LIBS[@]}" -eq 1 ]] || {
    echo "[a5_lidu_fused_batch_get_prepare_mtp] ERROR: expected one local libcust_opapi.so, found ${#OPAPI_LIBS[@]}." >&2
    exit 1
}
VENDOR_DIR="$(cd "$(dirname "${OPAPI_LIBS[0]}")/../.." && pwd)"
find "${VENDOR_DIR}" -type f -name binary_info_config.json -print0 | \
    xargs -0 -r grep -q 'A5LiduFusedBatchGetPrepareMtp' || {
    echo "[a5_lidu_fused_batch_get_prepare_mtp] ERROR: generated metadata lacks A5LiduFusedBatchGetPrepareMtp." >&2
    exit 1
}

pushd "${TORCH_EXTENSION}" >/dev/null
rm -rf build
rm -f nanovllm_dsa_lidu_fused_batch_get_prepare_mtp/_C*.so
MAX_JOBS="${BUILD_JOBS}" "${PYTHON_BIN}" setup.py build_ext --inplace

PTH_DIRECTORY="$("${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path
import site
import sysconfig

candidates: list[str] = []
for key in ("purelib", "platlib"):
    directory = sysconfig.get_path(key)
    if directory and directory not in candidates:
        candidates.append(directory)
for directory in getattr(site, "getsitepackages", lambda: [])():
    if directory and directory not in candidates:
        candidates.append(directory)
user_site = site.getusersitepackages()
if site.ENABLE_USER_SITE and user_site and user_site not in candidates:
    candidates.append(user_site)

for directory in candidates:
    path = Path(directory)
    if not path.exists() and directory == user_site and site.ENABLE_USER_SITE:
        path.mkdir(parents=True, exist_ok=True)
    if path.is_dir() and os.access(path, os.W_OK):
        print(path.resolve())
        break
else:
    raise SystemExit("ERROR: no writable site-packages directory")
PY
)"
PTH_FILE="${PTH_DIRECTORY}/nanovllm_dsa_lidu_fused_batch_get_prepare_mtp.pth"
printf '%s\n' "${TORCH_EXTENSION}" > "${PTH_FILE}"
echo "[a5_lidu_fused_batch_get_prepare_mtp] path registration: ${PTH_FILE} -> ${TORCH_EXTENSION}"
popd >/dev/null

"${PYTHON_BIN}" - <<'PY'
import nanovllm_dsa_lidu_fused_batch_get_prepare_mtp as lidu

if not hasattr(lidu, "lidu_fused_batch_get_prepare_mtp_out"):
    raise SystemExit("ERROR: standalone fused registration is absent")
print("[a5_lidu_fused_batch_get_prepare_mtp] standalone torch registration: OK")
PY

echo "[a5_lidu_fused_batch_get_prepare_mtp] build complete"
echo "[a5_lidu_fused_batch_get_prepare_mtp] local vendor: ${VENDOR_DIR}"
