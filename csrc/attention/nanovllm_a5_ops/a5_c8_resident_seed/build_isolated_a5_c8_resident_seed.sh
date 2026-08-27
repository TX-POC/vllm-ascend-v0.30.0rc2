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
SPEC="${ROOT}/ops.json"
OP_DEF="${ROOT}/op_host/a5_c8_resident_seed_def.cpp"
BRIDGE="${ROOT}/torch_extension/csrc/a5_c8_resident_seed.cpp"
HOST_COMPAT="${ROOT}/op_host/a5_c8_resident_seed_host_compat.h"
VECTOR_KERNEL="${ROOT}/op_kernel/arch35/quant_lightning_indexer_service_vector.h"
# CANN 9.1's generic msopgen TypeRange parser does not accept the JSON
# spelling `float8_e4m3fn`.  Match the official/P3 two-layer contract:
# bootstrap JSON says int8; the target-only OpDef and typed bridge remain the
# fail-closed FP8 E4M3FN ABI guards.  Check all three before invoking
# msopgen, so a wrong schema is diagnosed in one short, actionable message.
if ! grep -Eq '"name": "query".*"type": \["int8"\]' "${SPEC}" || \
   ! grep -Eq '"name": "resident_index_key".*"type": \["int8"\]' "${SPEC}" || \
   ! grep -A3 'this->Input("query")' "${OP_DEF}" | grep -q 'ge::DT_INT8' || \
   ! grep -A3 'this->Input("resident_index_key")' "${OP_DEF}" | grep -q 'ge::DT_INT8' || \
   ! grep -q 'aicore_config_950.Input("query")' "${OP_DEF}" || \
   ! grep -q 'aicore_config_950.Input("resident_index_key")' "${OP_DEF}" || \
   ! grep -q 'ge::DT_FLOAT8_E4M3FN' "${OP_DEF}" || \
   ! grep -q 'at::kFloat8_e4m3fn' "${BRIDGE}" || \
   [[ ! -s "${HOST_COMPAT}" ]] || \
   ! grep -q 'OP_CHECK_NULL_WITH_CONTEXT' "${HOST_COMPAT}" || \
   ! grep -q 'OP_CHECK_IF' "${HOST_COMPAT}" || \
   ! grep -q 'OP_LOGE' "${HOST_COMPAT}" || \
   ! grep -q 'REPORT_INNER_ERR_MSG' "${HOST_COMPAT}"; then
  echo "[a5_c8_resident_seed] ERROR: CANN-bootstrap/FP8 ABI contract is incomplete." >&2
  echo "[a5_c8_resident_seed] expected: JSON/generic OpDef query/index-key=int8; Ascend950 OpDef/bridge=FP8 E4M3FN; standalone err/ops_err.h + log/log.h compatibility source." >&2
  exit 2
fi
# CANN 9.1 declares InitGlobalMemory with a writable GlobalTensor lvalue.
# Indexing a GlobalTensor produces a temporary view; a direct indexed GM view
# therefore fails only late in the multi-variant kernel build.  Reject the
# two resident-seed instances before msopgen and require the address-identical
# named views used by the source closure.
if grep -qF 'InitGlobalMemory(residentTauGm[rowOffset]' "${VECTOR_KERNEL}" || \
   grep -qF 'InitGlobalMemory(tauValidGm[rowOffset]' "${VECTOR_KERNEL}" || \
   grep -qF 'residentSeedLocal_[1]' "${VECTOR_KERNEL}" || \
   ! grep -qF 'InitBuffer(residentSeedBuf_, 16 * sizeof(int32_t));' "${VECTOR_KERNEL}" || \
   ! grep -qF 'residentSeedLocal_.SetValue(8, tauValid);' "${VECTOR_KERNEL}" || \
   ! grep -qF 'DataCopyPad(tauValidGm[outputRow], residentSeedLocal_[8], scalarCopy);' "${VECTOR_KERNEL}" || \
   ! grep -qF 'GetTPipePtr()->FetchEventID(HardEvent::V_S)' "${VECTOR_KERNEL}" || \
   ! grep -qF 'SetFlag<HardEvent::V_S>(topkVectorToScalar);' "${VECTOR_KERNEL}" || \
   ! grep -qF 'WaitFlag<HardEvent::V_S>(topkVectorToScalar);' "${VECTOR_KERNEL}" || \
   ! grep -qF 'GetTPipePtr()->FetchEventID(HardEvent::S_MTE3)' "${VECTOR_KERNEL}" || \
   ! grep -qF 'GetTPipePtr()->FetchEventID(HardEvent::MTE3_S)' "${VECTOR_KERNEL}" || \
   ! grep -qF 'GlobalTensor<int32_t> residentTauOutput = residentTauGm[rowOffset];' "${VECTOR_KERNEL}" || \
   ! grep -qF 'GlobalTensor<int32_t> tauValidOutput = tauValidGm[rowOffset];' "${VECTOR_KERNEL}" || \
   ! grep -qF 'InitGlobalMemory(residentTauOutput, rowCount, 0);' "${VECTOR_KERNEL}" || \
   ! grep -qF 'InitGlobalMemory(tauValidOutput, rowCount, 0);' "${VECTOR_KERNEL}"; then
  echo "[a5_c8_resident_seed] ERROR: CANN 9.1 InitGlobalMemory lvalue contract or resident receipt device ordering is incomplete (GM lvalue, V-to-S, or Scalar/MTE3)." >&2
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

cp "${SPEC}" "${OP_DEV}/ops.json"
cp -a "${ROOT}/op_host/." "${OP_DEV}/op_host/"
cp -a "${ROOT}/op_kernel/." "${OP_DEV}/op_kernel/"
# msopgen builds a fresh, isolated CMake project.  Unlike normal
# vLLM-Ascend, it has no repository include graph that supplies these two
# private headers.  Stage the local, diagnostics-only compatibility layer
# under both quoted paths before generation; the later op_host copy carries
# them into the generated build tree as well.
mkdir -p "${OP_DEV}/op_host/err" "${OP_DEV}/op_host/log"
cp -f "${HOST_COMPAT}" "${OP_DEV}/op_host/err/ops_err.h"
cp -f "${HOST_COMPAT}" "${OP_DEV}/op_host/log/log.h"
if [[ ! -s "${OP_DEV}/op_host/err/ops_err.h" ]] || \
   ! grep -q 'OP_CHECK_NULL_WITH_CONTEXT' "${OP_DEV}/op_host/err/ops_err.h" || \
   ! grep -q 'OP_CHECK_IF' "${OP_DEV}/op_host/err/ops_err.h" || \
   ! grep -q 'OP_LOGE' "${OP_DEV}/op_host/log/log.h" || \
   ! grep -q 'REPORT_INNER_ERR_MSG' "${OP_DEV}/op_host/err/ops_err.h"; then
  echo "[a5_c8_resident_seed] ERROR: staged host closure is incomplete (err/ops_err.h + log/log.h)." >&2
  exit 2
fi

echo "[a5_c8_resident_seed] isolated root=${BUILD_ROOT} target=${TARGET_SOC} jobs=${BUILD_JOBS}"
echo "[a5_c8_resident_seed] source-derived C8 closure only; no normal OPP or vLLM build"
echo "[a5_c8_resident_seed] host-header closure: staged standalone err/ops_err.h + log/log.h; remaining host SDK includes=CANN"
msopgen gen -i "${OP_DEV}/ops.json" -f aclnn -c "ai_core-${TARGET_SOC}" \
  -lan cpp -op A5C8ResidentSeed -out "${GENERATED}"

cp -a "${OP_DEV}/op_host/." "${GENERATED}/op_host/"
cp -a "${OP_DEV}/op_kernel/." "${GENERATED}/op_kernel/"
# Check the generated CMake input, not just the pre-msopgen staging tree.
# This is the exact closure that failed on 179: source files include these
# paths with quotes, so both headers must sit below generated/op_host before
# CANN invokes the host compiler.
if [[ ! -s "${GENERATED}/op_host/err/ops_err.h" ]] || \
   [[ ! -s "${GENERATED}/op_host/log/log.h" ]] || \
   ! grep -q 'OP_CHECK_NULL_WITH_CONTEXT' "${GENERATED}/op_host/err/ops_err.h" || \
   ! grep -q 'OP_LOGE' "${GENERATED}/op_host/log/log.h"; then
  echo "[a5_c8_resident_seed] ERROR: generated host closure is incomplete (expected custom_op/op_host/{err/ops_err.h,log/log.h})." >&2
  exit 2
fi
# `msopgen` creates an all-in-one host source named
# `a5_c8_resident_seed.cpp`, while the source-derived C8 closure supplies its
# real OpDef in `a5_c8_resident_seed_def.cpp`.  Those filenames differ, so a
# plain overlay leaves two `OP_ADD(A5C8ResidentSeed)` registrations.  CANN
# then emits two same-named sections in aic-*-ops-info.ini and fails before
# kernel compilation.  This is the exact P3 failure mode: retain the generated
# CMake scaffold, remove only its generated host stub, and prove that the
# source-derived OpDef is now the one and only registration.
P4_GENERATED_HOST_STUB="${GENERATED}/op_host/a5_c8_resident_seed.cpp"
P4_SOURCE_OPDEF="${GENERATED}/op_host/a5_c8_resident_seed_def.cpp"
P4_OPADD_REGEX='OP_ADD[[:space:]]*\([[:space:]]*A5C8ResidentSeed[[:space:]]*\)'
if [[ ! -f "${P4_GENERATED_HOST_STUB}" ]]; then
  echo "[a5_c8_resident_seed] ERROR: expected msopgen host stub is missing: ${P4_GENERATED_HOST_STUB}" >&2
  exit 1
fi
if [[ ! -f "${P4_SOURCE_OPDEF}" ]] || ! cmp -s "${OP_DEF}" "${P4_SOURCE_OPDEF}"; then
  echo "[a5_c8_resident_seed] ERROR: generated source OpDef is missing or differs from the staged source: ${P4_SOURCE_OPDEF}" >&2
  exit 1
fi
P4_STUB_REGISTRATIONS="$(grep -Ec "${P4_OPADD_REGEX}" "${P4_GENERATED_HOST_STUB}" || true)"
if [[ "${P4_STUB_REGISTRATIONS}" -ne 1 ]]; then
  echo "[a5_c8_resident_seed] ERROR: generated host stub must contain exactly one A5C8ResidentSeed OP_ADD; found ${P4_STUB_REGISTRATIONS}." >&2
  exit 1
fi
P4_SOURCE_REGISTRATIONS="$(grep -Ec "${P4_OPADD_REGEX}" "${P4_SOURCE_OPDEF}" || true)"
if [[ "${P4_SOURCE_REGISTRATIONS}" -ne 1 ]]; then
  echo "[a5_c8_resident_seed] ERROR: source-derived OpDef must contain exactly one A5C8ResidentSeed OP_ADD; found ${P4_SOURCE_REGISTRATIONS}." >&2
  exit 1
fi
rm -f "${P4_GENERATED_HOST_STUB}"
if [[ -e "${P4_GENERATED_HOST_STUB}" ]]; then
  echo "[a5_c8_resident_seed] ERROR: failed to remove generated host stub: ${P4_GENERATED_HOST_STUB}" >&2
  exit 1
fi

P4_OPDEF_SOURCES=()
P4_OPDEF_COUNTS=()
P4_TOTAL_REGISTRATIONS=0
while IFS= read -r -d '' P4_HOST_SOURCE; do
  P4_MATCH_COUNT="$(grep -Ec "${P4_OPADD_REGEX}" "${P4_HOST_SOURCE}" || true)"
  if [[ "${P4_MATCH_COUNT}" -gt 0 ]]; then
    P4_OPDEF_SOURCES+=("${P4_HOST_SOURCE}")
    P4_OPDEF_COUNTS+=("${P4_MATCH_COUNT}")
    P4_TOTAL_REGISTRATIONS=$((P4_TOTAL_REGISTRATIONS + P4_MATCH_COUNT))
  fi
done < <(find "${GENERATED}/op_host" -type f \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print0)

if [[ "${P4_TOTAL_REGISTRATIONS}" -ne 1 || \
      "${#P4_OPDEF_SOURCES[@]}" -ne 1 || \
      "${P4_OPDEF_SOURCES[0]:-}" != "${P4_SOURCE_OPDEF}" ]]; then
  echo "[a5_c8_resident_seed] ERROR: expected exactly one source-derived A5C8ResidentSeed OP_ADD registration; total=${P4_TOTAL_REGISTRATIONS}, files=${#P4_OPDEF_SOURCES[@]}." >&2
  if [[ "${#P4_OPDEF_SOURCES[@]}" -eq 0 ]]; then
    echo "[a5_c8_resident_seed] OP_ADD source: <none>" >&2
  else
    for P4_OPDEF_INDEX in "${!P4_OPDEF_SOURCES[@]}"; do
      printf '[a5_c8_resident_seed] OP_ADD source: count=%s path=%s\n' \
        "${P4_OPDEF_COUNTS[P4_OPDEF_INDEX]}" "${P4_OPDEF_SOURCES[P4_OPDEF_INDEX]}" >&2
    done
  fi
  exit 1
fi
echo "[a5_c8_resident_seed] host registration closure: total=${P4_TOTAL_REGISTRATIONS} source=${P4_OPDEF_SOURCES[0]}"

if ! grep -Rqs 'GetLastKthValue' "${GENERATED}/op_kernel" || \
   ! grep -Rqs 'DataCopyPad(residentTauGm' "${GENERATED}/op_kernel"; then
  echo "[a5_c8_resident_seed] ERROR: generated tree lost the real resident threshold write." >&2
  exit 1
fi

if ! (
  cd "${GENERATED}"
  OPS_CPU_NUMBER="${BUILD_JOBS}" bash build.sh
); then
  # Do not edit this generated INI to hide a registration bug.  If CANN ever
  # reports the same class of failure again, print its section count once so
  # the retained build log immediately identifies the broken closure.
  P4_AUTOGEN_INI="${GENERATED}/build_out/autogen/aic-${TARGET_SOC}-ops-info.ini"
  if [[ -f "${P4_AUTOGEN_INI}" ]]; then
    P4_INI_SECTIONS="$(grep -Ec '^\[A5C8ResidentSeed\]$' "${P4_AUTOGEN_INI}" || true)"
  else
    P4_INI_SECTIONS=0
  fi
  echo "[a5_c8_resident_seed] build failure diagnostic: generated_ini=${P4_AUTOGEN_INI} resident_seed_sections=${P4_INI_SECTIONS} (INI is diagnostic-only and was not modified)." >&2
  P4_DIRECT_INDEXED_INIT="$(grep -RnhE 'InitGlobalMemory\([^,]*Gm\[[^]]+\]' "${GENERATED}/op_kernel" 2>/dev/null || true)"
  if [[ -n "${P4_DIRECT_INDEXED_INIT}" ]]; then
    printf '[a5_c8_resident_seed] build failure diagnostic: direct_indexed_InitGlobalMemory_calls=\n%s\n' "${P4_DIRECT_INDEXED_INIT}" >&2
  else
    echo "[a5_c8_resident_seed] build failure diagnostic: direct_indexed_InitGlobalMemory_calls=<none>." >&2
  fi
  exit 1
fi

RUN_PKG="$(find "${GENERATED}/build_out" -maxdepth 1 -type f -name '*.run' -print -quit)"
if [[ -z "${RUN_PKG}" ]]; then
  echo "[a5_c8_resident_seed] ERROR: msopgen build did not produce a .run package." >&2
  exit 1
fi
chmod +x "${RUN_PKG}"
"${RUN_PKG}" --quiet --install-path="${LOCAL_OPP}"

echo "[a5_c8_resident_seed] PASS opp=${LOCAL_OPP}"
