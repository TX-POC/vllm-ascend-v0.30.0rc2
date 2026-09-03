#!/usr/bin/env bash
# Build the *isolated* P10 Stage-A W4A4C8 score-receipt OPP.
#
# This is intentionally not an ambient recursive CMake build and never
# touches vLLM-Ascend's installed/custom OPP.  A fresh msopgen project is
# created for one exact source revision; its generated host stub is replaced
# by the checked-in, source-derived OpDef so the package has exactly one
# registration.  The package is a real CANN entrypoint, not a marker OPP.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VLLM_ROOT="${A5_P10_STAGE_A_SOURCE_ROOT:-$(cd "${ROOT}/../../../../.." && pwd -P)}"
EXPECTED_HEAD="${A5_P10_STAGE_A_EXPECTED_SOURCE_HEAD:-}"
PACKAGE_PARENT="${A5_P10_STAGE_A_PACKAGE_PARENT:-${ROOT}/build-stage-a}"
CANN_3RD_LIB_PATH="${A5_P10_STAGE_A_CANN_3RD_LIB_PATH:-}"
BUILD_JOBS="${A5_P10_STAGE_A_BUILD_JOBS:-4}"
TARGET_SOC_RAW="${A5_P10_STAGE_A_SOC:-ascend950}"

fail() {
  printf 'P10_STAGE_A_BUILD_FAIL reason=%s\n' "$1" >&2
  exit 2
}

case "${TARGET_SOC_RAW,,}" in
  ascend950|ascend950pr*|ascend950dt*) TARGET_SOC=ascend950 ;;
  *) fail "unsupported_soc:value=${TARGET_SOC_RAW}" ;;
esac
[[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || fail "invalid_jobs:value=${BUILD_JOBS}"
[[ -z "${SOC_VERSION:-}" ]] || fail "legacy_SOC_VERSION_set"
[[ -d "${VLLM_ROOT}/.git" ]] || fail "vllm_git_root_missing:path=${VLLM_ROOT}"
[[ -n "${EXPECTED_HEAD}" ]] || fail "expected_source_head_missing"

SOURCE_HEAD="$(git -C "${VLLM_ROOT}" rev-parse HEAD 2>/dev/null || true)"
[[ "${SOURCE_HEAD}" == "${EXPECTED_HEAD}" ]] ||
  fail "source_head_mismatch:actual=${SOURCE_HEAD}:expected=${EXPECTED_HEAD}"
SOURCE_DIRTY="$(git -C "${VLLM_ROOT}" status --porcelain --untracked-files=no)"
[[ -z "${SOURCE_DIRTY}" ]] || fail "source_tracked_changes_present"

[[ -n "${CANN_3RD_LIB_PATH}" ]] || fail "cann_third_party_cache_path_missing"
for cache_file in \
  "${CANN_3RD_LIB_PATH}/makeself/makeself.sh" \
  "${CANN_3RD_LIB_PATH}/makeself/makeself-header.sh" \
  "${CANN_3RD_LIB_PATH}/json/include/nlohmann/json.hpp"; do
  [[ -r "${cache_file}" ]] || fail "cann_third_party_cache_incomplete:path=${cache_file}"
done
if [[ ! -r "${CANN_3RD_LIB_PATH}/abseil-cpp/abseil-cpp-20230802.1.tar.gz" &&
      ! -r "${CANN_3RD_LIB_PATH}/pkg/abseil-cpp-20230802.1.tar.gz" ]]; then
  fail "cann_third_party_cache_incomplete:need=abseil"
fi
if [[ ! -r "${CANN_3RD_LIB_PATH}/protobuf/protobuf-all-25.1.tar.gz" &&
      ! -r "${CANN_3RD_LIB_PATH}/pkg/protobuf-25.1.tar.gz" ]]; then
  fail "cann_third_party_cache_incomplete:need=protobuf"
fi
CANN_3RD_LIB_PATH="$(cd "${CANN_3RD_LIB_PATH}" && pwd -P)"
command -v msopgen >/dev/null 2>&1 || fail "msopgen_missing"

SPEC="${ROOT}/ops.json"
HOST_DEF="${ROOT}/op_host/a5_p10_w4a4c8_score_receipt_def.cpp"
KERNEL_ROOT="${ROOT}/op_kernel"
for file in \
  "${SPEC}" "${HOST_DEF}" \
  "${KERNEL_ROOT}/a5_p10_w4a4c8_score_receipt.cpp" \
  "${KERNEL_ROOT}/a5_p10_w4a4c8_score_receipt_qli.h" \
  "${KERNEL_ROOT}/a5_p10_w4a4c8_score_receipt_tiling.h" \
  "${KERNEL_ROOT}/a5_p10_w4a4c8_score_receipt_abi.h"; do
  [[ -f "${file}" ]] || fail "source_closure_file_missing:path=${file}"
done

# CANN 9.1's generic JSON parser cannot spell FP8 E4M3FN.  The generated
# schema therefore uses int8 only to bootstrap msopgen; the checked-in OpDef
# is the sole compiled registration and requires FP8/BF16/FP32 exactly.
grep -Eq '"name": "query".*"type": \["int8"\]' "${SPEC}" ||
  fail "bootstrap_json_query_not_int8"
grep -Eq '"name": "key".*"type": \["int8"\]' "${SPEC}" ||
  fail "bootstrap_json_key_not_int8"
grep -Fq 'ge::DT_FLOAT8_E4M3FN' "${HOST_DEF}" || fail "host_fp8_guard_missing"
grep -Fq 'ge::DT_BF16' "${HOST_DEF}" || fail "host_bf16_guard_missing"
grep -Fq 'scoreSurface.GetDim(1) != s1Base' "${HOST_DEF}" ||
  fail "full_native_score_stride_gate_missing"
grep -Fq 'receipt_state' "${KERNEL_ROOT}/a5_p10_w4a4c8_score_receipt_abi.h" ||
  fail "forward_preparation_contract_missing"
grep -Fq 'FREE -> FILLING -> READY' "${KERNEL_ROOT}/a5_p10_w4a4c8_score_receipt_qli.h" ||
  fail "stage_a_receipt_transition_contract_missing"
if grep -Fq 'InitializeReceiptStateShard' "${KERNEL_ROOT}/a5_p10_w4a4c8_score_receipt_qli.h" ||
   grep -Fq 'InitGlobalMemory' "${KERNEL_ROOT}/arch35/quant_lightning_indexer_service_vector.h"; then
  fail "stage_a_must_not_reset_concurrent_receipt_state"
fi

mkdir -p "${PACKAGE_PARENT}"
PACKAGE_ROOT="${A5_P10_STAGE_A_PACKAGE_ROOT:-${PACKAGE_PARENT}/p10_stage_a_${SOURCE_HEAD:0:12}_$(date +%Y%m%d_%H%M%S)}"
case "${PACKAGE_ROOT}" in
  "${PACKAGE_PARENT}"/*) ;;
  *) fail "unsafe_package_root:path=${PACKAGE_ROOT}:parent=${PACKAGE_PARENT}" ;;
esac
[[ ! -e "${PACKAGE_ROOT}" ]] || fail "fresh_package_root_exists:path=${PACKAGE_ROOT}"

STAGE_ROOT="${PACKAGE_ROOT}/source"
OP_DEV="${STAGE_ROOT}/op_dev"
GENERATED="${PACKAGE_ROOT}/custom_op"
INSTALL_ROOT="${PACKAGE_ROOT}/opp"
mkdir -p "${OP_DEV}/op_host" "${OP_DEV}/op_kernel"
cp "${SPEC}" "${OP_DEV}/ops.json"
cp -a "${ROOT}/op_host/." "${OP_DEV}/op_host/"
cp -a "${KERNEL_ROOT}/." "${OP_DEV}/op_kernel/"

printf 'P10_STAGE_A_BUILD_SOURCE root=%s head=%s package=%s target=%s jobs=%s\n' \
  "${VLLM_ROOT}" "${SOURCE_HEAD}" "${PACKAGE_ROOT}" "${TARGET_SOC}" "${BUILD_JOBS}"

msopgen gen -i "${OP_DEV}/ops.json" -f aclnn -c "ai_core-${TARGET_SOC}" \
  -lan cpp -op A5P10W4A4C8ScoreReceipt -out "${GENERATED}"

# Keep msopgen's CMake/package machinery but replace its generic registration
# with the one that owns the real W4A4C8 tiler and graph-alias ABI.  Removing
# the generated stub is safe because GENERATED is freshly created above.
cp -a "${OP_DEV}/op_host/." "${GENERATED}/op_host/"
cp -a "${OP_DEV}/op_kernel/." "${GENERATED}/op_kernel/"
# CANN normalizes W4/A4/C8 runs in the CamelCase op name when naming generated
# host and kernel files.  The source closure intentionally retains its ABI
# spelling (without inserted separators), so explicitly remap it to CANN's
# generated canonical filename and delete the source-only spelling.  Keeping
# both names would compile a placeholder or duplicate the real kernel.
GENERATED_KERNEL="${GENERATED}/op_kernel/a5_p10_w4_a4_c8_score_receipt.cpp"
SOURCE_KERNEL="${GENERATED}/op_kernel/a5_p10_w4a4c8_score_receipt.cpp"
[[ -f "${GENERATED_KERNEL}" ]] || fail "generated_kernel_stub_missing"
[[ -f "${SOURCE_KERNEL}" ]] || fail "source_kernel_missing_after_stage"
cp -p "${SOURCE_KERNEL}" "${GENERATED_KERNEL}"
rm -f "${SOURCE_KERNEL}"

GENERATED_STUB="${GENERATED}/op_host/a5_p10_w4_a4_c8_score_receipt.cpp"
SOURCE_OPDEF="${GENERATED}/op_host/a5_p10_w4a4c8_score_receipt_def.cpp"
[[ -f "${GENERATED_STUB}" ]] || fail "generated_host_stub_missing"
[[ -f "${SOURCE_OPDEF}" ]] || fail "source_opdef_missing_after_stage"
REGEX='OP_ADD[[:space:]]*\([[:space:]]*A5P10W4A4C8ScoreReceipt[[:space:]]*\)'
[[ "$(grep -Ec "${REGEX}" "${GENERATED_STUB}" || true)" -eq 1 ]] ||
  fail "generated_host_stub_registration_unexpected"
[[ "$(grep -Ec "${REGEX}" "${SOURCE_OPDEF}" || true)" -eq 1 ]] ||
  fail "source_opdef_registration_unexpected"
rm -f "${GENERATED_STUB}"
TOTAL_REGISTRATIONS="$(grep -REc "${REGEX}" "${GENERATED}/op_host" --include='*.cpp' --include='*.cc' --include='*.cxx' 2>/dev/null | awk -F: '{sum += $NF} END {print sum + 0}')"
[[ "${TOTAL_REGISTRATIONS}" -eq 1 ]] ||
  fail "duplicate_or_missing_op_registration:count=${TOTAL_REGISTRATIONS}"

if ! (
  cd "${GENERATED}"
  CANN_3RD_LIB_PATH="${CANN_3RD_LIB_PATH}" OPS_CPU_NUMBER="${BUILD_JOBS}" bash build.sh
); then
  INI="${GENERATED}/build_out/autogen/aic-${TARGET_SOC}-ops-info.ini"
  SECTION_COUNT=0
  [[ -f "${INI}" ]] && SECTION_COUNT="$(grep -Ec '^\[A5P10W4A4C8ScoreReceipt\]$' "${INI}" || true)"
  printf 'P10_STAGE_A_BUILD_DIAG generated_ini=%s registrations=%s\n' \
    "${INI}" "${SECTION_COUNT}" >&2
  exit 1
fi

RUN_PKG="$(find "${GENERATED}/build_out" -maxdepth 1 -type f -name '*.run' -print -quit)"
[[ -n "${RUN_PKG}" ]] || fail "run_package_missing"
chmod +x "${RUN_PKG}"
"${RUN_PKG}" --quiet --install-path="${INSTALL_ROOT}"

OPAPI="$(find "${INSTALL_ROOT}" -type f -path '*/op_api/lib/libcust_opapi.so' -print -quit)"
[[ -s "${OPAPI}" ]] || fail "installed_opapi_missing"
printf 'P10_STAGE_A_BUILD_PASS package=%s opp=%s opapi=%s source=%s\n' \
  "${PACKAGE_ROOT}" "${INSTALL_ROOT}" "${OPAPI}" "${SOURCE_HEAD}"
