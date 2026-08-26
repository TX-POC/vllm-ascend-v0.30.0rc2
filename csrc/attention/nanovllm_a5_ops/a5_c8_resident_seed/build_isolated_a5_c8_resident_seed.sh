#!/usr/bin/env bash
# Build only the resident-seed probe. This script never modifies the normal
# custom OPP package and never invokes the vLLM-Ascend package build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${A5_C8_RESIDENT_SEED_BUILD_ROOT:-${ROOT}/build-isolated}"
TARGET_SOC="${A5_C8_RESIDENT_SEED_SOC:-ascend950}"
BUILD_JOBS="${A5_C8_RESIDENT_SEED_BUILD_JOBS:-4}"

case "${TARGET_SOC}" in
  ascend950) ;;
  *)
    echo "[a5_c8_resident_seed] ERROR: only ascend950 is supported; got ${TARGET_SOC}." >&2
    exit 2
    ;;
esac
case "${BUILD_ROOT}" in
  "${ROOT}"/build-isolated|"${ROOT}"/build-isolated/*) ;;
  *)
    echo "[a5_c8_resident_seed] ERROR: build root must stay below ${ROOT}/build-isolated." >&2
    exit 2
    ;;
esac
if ! command -v msopgen >/dev/null 2>&1; then
  echo "[a5_c8_resident_seed] ERROR: msopgen is unavailable; source the CANN 9.1 environment first." >&2
  exit 2
fi
if [[ -e "${BUILD_ROOT}/op_dev" || -e "${BUILD_ROOT}/custom_op" || -e "${BUILD_ROOT}/opp" ]]; then
  echo "[a5_c8_resident_seed] ERROR: ${BUILD_ROOT} already contains an isolated build; choose a fresh timestamped A5_C8_RESIDENT_SEED_BUILD_ROOT." >&2
  exit 2
fi

OP_DEV="${BUILD_ROOT}/op_dev"
GENERATED="${BUILD_ROOT}/custom_op"
LOCAL_OPP="${BUILD_ROOT}/opp"
mkdir -p "${OP_DEV}/op_host" "${OP_DEV}/op_kernel"

cp "${ROOT}/ops.json" "${OP_DEV}/ops.json"
cp -a "${ROOT}/op_host/." "${OP_DEV}/op_host/"
cp -a "${ROOT}/op_kernel/." "${OP_DEV}/op_kernel/"

echo "[a5_c8_resident_seed] isolated root=${BUILD_ROOT} target=${TARGET_SOC} jobs=${BUILD_JOBS}"
echo "[a5_c8_resident_seed] source-derived C8 closure only; no normal OPP or vLLM build"
msopgen gen -i "${OP_DEV}/ops.json" -f aclnn -c "ai_core-${TARGET_SOC}" \
  -lan cpp -op A5C8ResidentSeed -out "${GENERATED}"

cp -a "${OP_DEV}/op_host/." "${GENERATED}/op_host/"
cp -a "${OP_DEV}/op_kernel/." "${GENERATED}/op_kernel/"
if ! grep -Rqs 'GetLastKthValue' "${GENERATED}/op_kernel" || \
   ! grep -Rqs 'DataCopyPad(residentTauGm' "${GENERATED}/op_kernel"; then
  echo "[a5_c8_resident_seed] ERROR: generated tree lost the real resident threshold write." >&2
  exit 1
fi

(
  cd "${GENERATED}"
  OPS_CPU_NUMBER="${BUILD_JOBS}" bash build.sh
)

RUN_PKG="$(find "${GENERATED}/build_out" -maxdepth 1 -type f -name '*.run' -print -quit)"
if [[ -z "${RUN_PKG}" ]]; then
  echo "[a5_c8_resident_seed] ERROR: msopgen build did not produce a .run package." >&2
  exit 1
fi
chmod +x "${RUN_PKG}"
"${RUN_PKG}" --quiet --install-path="${LOCAL_OPP}"

echo "[a5_c8_resident_seed] PASS opp=${LOCAL_OPP}"
