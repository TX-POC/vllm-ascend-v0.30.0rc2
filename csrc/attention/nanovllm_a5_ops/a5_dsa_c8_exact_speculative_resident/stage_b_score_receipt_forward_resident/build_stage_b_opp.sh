#!/usr/bin/env bash
# Build the isolated P10 Stage-B score-receipt/forward-resident OPP.
#
# This is the only Stage-B package route. It stages checked-in Stage-B source
# and the four-file KVAF device closure from one matched, clean KVAF worktree
# into a fresh msopgen project. It never reads an installed KVAF header or
# reuses a CMake cache, OPP, bridge, or product build directory.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_ROOT="${P10_STAGE_SOURCE_ROOT:-${ROOT}}"
VLLM_ROOT="${P10_STAGE_VLLM_SOURCE_ROOT:-$(cd "${ROOT}/../../../../.." && pwd -P)}"
KVAF_ROOT="${P10_STAGE_KVAF_SOURCE_ROOT:-${KVAF_FUSED:-}}"
EXPECTED_VLLM_HEAD="${P10_STAGE_EXPECTED_VLLM_SHA:-}"
EXPECTED_KVAF_HEAD="${P10_STAGE_EXPECTED_KVAF_SHA:-}"
EXPECTED_SOURCE_SHA="${P10_STAGE_SOURCE_SHA256:-}"
STAGE_NAME="${P10_STAGE_NAME:-stage_b}"
BUILD_JOBS="${P10_STAGE_BUILD_JOBS:-4}"
TARGET_SOC_RAW="${P10_STAGE_SOC:-ascend950}"
CANN_3RD_LIB_PATH="${P10_STAGE_CANN_3RD_LIB_PATH:-${A5_P10_STAGE_B_CANN_3RD_LIB_PATH:-}}"
OUTPUT_ROOT="${P10_STAGE_OUTPUT_ROOT:-}"
PROVENANCE_FILE="${P10_STAGE_PROVENANCE_FILE:-}"

fail() {
  printf 'P10_STAGE_B_BUILD_FAIL reason=%s\n' "$1" >&2
  exit 2
}

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# Match the source-only digest used by build_p10_target_assembly.sh. A stale
# staged source tree must fail before msopgen can spend time compiling it.
tree_digest() {
  local root="$1"
  local list file rel file_hash
  list="$(mktemp "${TMPDIR:-/tmp}/p10-stage-b-tree.XXXXXX")"
  while IFS= read -r -d '' file; do
    rel="${file#${root}/}"
    case "${rel}" in
      .git/*|build/*|build-*/*|*/build/*|*/build-*/*|op_dev/*|custom_op/*|opp/*|packages/*|*/packages/*|__pycache__/*|*/__pycache__/*|*.pyc|*.o|*.so|*.run|CMakeCache.txt)
        continue
        ;;
    esac
    printf '%s\0' "${rel}" >> "${list}"
  done < <(find "${root}" -type f -print0)
  [[ -s "${list}" ]] || {
    rm -f "${list}"
    fail "stage_source_empty:path=${root}"
  }
  while IFS= read -r -d '' rel; do
    file="${root}/${rel}"
    file_hash="$(hash_file "${file}")"
    printf '%s  %s\n' "${file_hash}" "${rel}"
  done < <(LC_ALL=C sort -z "${list}") | {
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum | awk '{print $1}'
    else
      shasum -a 256 | awk '{print $1}'
    fi
  }
  rm -f "${list}"
}

require_clean_exact_git_root() {
  local label="$1"
  local root="$2"
  local expected="$3"
  local head dirty
  [[ -e "${root}/.git" ]] || fail "${label}_git_root_missing:path=${root}"
  [[ -n "${expected}" ]] || fail "${label}_expected_head_missing"
  head="$(git -C "${root}" rev-parse HEAD 2>/dev/null || true)"
  [[ "${head}" == "${expected}" ]] ||
    fail "${label}_head_mismatch:actual=${head}:expected=${expected}"
  dirty="$(git -C "${root}" status --porcelain --untracked-files=no 2>/dev/null || true)"
  [[ -z "${dirty}" ]] || fail "${label}_tracked_changes_present:path=${root}"
  printf 'P10_STAGE_B_SOURCE label=%s root=%s head=%s expected=%s dirty=0\n' \
    "${label}" "${root}" "${head}" "${expected}"
}

[[ "${STAGE_NAME}" == "stage_b" ]] || fail "invalid_stage_name:value=${STAGE_NAME}"
[[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || fail "invalid_jobs:value=${BUILD_JOBS}"
[[ -z "${SOC_VERSION:-}" ]] || fail "legacy_SOC_VERSION_set"
case "${TARGET_SOC_RAW,,}" in
  ascend950|ascend950pr*|ascend950dt*) TARGET_SOC=ascend950 ;;
  *) fail "unsupported_soc:value=${TARGET_SOC_RAW}" ;;
esac
[[ -d "${SOURCE_ROOT}" ]] || fail "stage_source_root_missing:path=${SOURCE_ROOT}"
[[ -n "${KVAF_ROOT}" ]] || fail "kvaf_source_root_missing"
[[ -n "${CANN_3RD_LIB_PATH}" ]] || fail "cann_third_party_cache_path_missing"
[[ -n "${PROVENANCE_FILE}" ]] || fail "stage_provenance_file_missing"

require_clean_exact_git_root vllm "${VLLM_ROOT}" "${EXPECTED_VLLM_HEAD}"
require_clean_exact_git_root kvaf "${KVAF_ROOT}" "${EXPECTED_KVAF_HEAD}"
VLLM_HEAD="$(git -C "${VLLM_ROOT}" rev-parse HEAD)"
KVAF_HEAD="$(git -C "${KVAF_ROOT}" rev-parse HEAD)"
SOURCE_SHA="$(tree_digest "${SOURCE_ROOT}")"
if [[ -n "${EXPECTED_SOURCE_SHA}" && "${SOURCE_SHA}" != "${EXPECTED_SOURCE_SHA}" ]]; then
  fail "stage_source_digest_mismatch:actual=${SOURCE_SHA}:expected=${EXPECTED_SOURCE_SHA}"
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${ROOT}/build-stage-b/opp_v${VLLM_HEAD:0:12}_k${KVAF_HEAD:0:12}_${SOURCE_SHA:0:12}"
fi
OUTPUT_PARENT="$(dirname "${OUTPUT_ROOT}")"
OUTPUT_LEAF="$(basename "${OUTPUT_ROOT}")"
mkdir -p "${OUTPUT_PARENT}"
OUTPUT_ROOT="$(cd "${OUTPUT_PARENT}" && pwd -P)/${OUTPUT_LEAF}"
mkdir -p "${OUTPUT_ROOT}"
if [[ -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  fail "output_root_not_empty:path=${OUTPUT_ROOT}"
fi

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

SPEC="${SOURCE_ROOT}/ops.json"
HOST_DEF="${SOURCE_ROOT}/op_host/a5_p10_score_receipt_forward_resident_def.cpp"
KERNEL_ROOT="${SOURCE_ROOT}/op_kernel"
STAGE_A_ABI="${VLLM_ROOT}/csrc/attention/nanovllm_a5_ops/a5_dsa_c8_exact_speculative_resident/stage_a_w4a4c8_score_receipt/op_kernel/a5_p10_w4a4c8_score_receipt_abi.h"
KVAF_DEMO_ROOT="${KVAF_ROOT}/access_operator/demo/aiv_batch_get"
KVAF_HEADER="${KVAF_DEMO_ROOT}/kernel/kvaf_forward_resident_get_device.h"
KVAF_DENSE_SOURCE="${KVAF_DEMO_ROOT}/kernel/execute_batch_get_aiv.asc"
KVAF_ABI="${KVAF_DEMO_ROOT}/include/batch_get_abi.h"
KVAF_DIRECT_ABI="${KVAF_DEMO_ROOT}/include/direct_hccp_abi.h"
for file in \
  "${SPEC}" "${HOST_DEF}" \
  "${KERNEL_ROOT}/a5_p10_score_receipt_forward_resident_kernel.cpp" \
  "${KERNEL_ROOT}/a5_p10_score_receipt_forward_resident_abi.h" \
  "${KERNEL_ROOT}/a5_p10_score_receipt_forward_resident_tiling.h" \
  "${KERNEL_ROOT}/a5_p10_score_receipt_forward_resident_kvaf_include.h" \
  "${STAGE_A_ABI}" \
  "${KVAF_HEADER}" "${KVAF_DENSE_SOURCE}" "${KVAF_ABI}" "${KVAF_DIRECT_ABI}"; do
  [[ -f "${file}" ]] || fail "source_closure_file_missing:path=${file}"
done
grep -Fq 'OP_ADD(A5P10ScoreReceiptForwardResident)' "${HOST_DEF}" ||
  fail "stage_b_registration_missing"
grep -Fq 'P10_STAGE_B_KVAF_FORWARD_RESIDENT_DEVICE_HEADER' \
  "${KERNEL_ROOT}/a5_p10_score_receipt_forward_resident_kvaf_include.h" ||
  fail "staged_kvaf_header_guard_missing"
grep -Fq 'kvaf_forward_resident/kvaf_forward_resident_get_device.h' \
  "${KERNEL_ROOT}/a5_p10_score_receipt_forward_resident_kvaf_include.h" ||
  fail "staged_kvaf_header_path_missing"

PACKAGE_PARENT="${P10_STAGE_PACKAGE_PARENT:-${OUTPUT_PARENT}}"
PACKAGE_PARENT="$(mkdir -p "${PACKAGE_PARENT}" && cd "${PACKAGE_PARENT}" && pwd -P)"
PACKAGE_ROOT="${P10_STAGE_PACKAGE_ROOT:-${PACKAGE_PARENT}/.p10_stage_b_${VLLM_HEAD:0:12}_${KVAF_HEAD:0:12}_${SOURCE_SHA:0:12}_$(date +%Y%m%d_%H%M%S)}"
case "${PACKAGE_ROOT}" in
  "${PACKAGE_PARENT}"/*) ;;
  *) fail "unsafe_package_root:path=${PACKAGE_ROOT}:parent=${PACKAGE_PARENT}" ;;
esac
[[ ! -e "${PACKAGE_ROOT}" ]] || fail "fresh_package_root_exists:path=${PACKAGE_ROOT}"

STAGE_ROOT="${PACKAGE_ROOT}/source"
OP_DEV="${STAGE_ROOT}/op_dev"
GENERATED="${PACKAGE_ROOT}/custom_op"
mkdir -p "${OP_DEV}/op_host" "${OP_DEV}/op_kernel/kvaf_forward_resident" \
  "${PACKAGE_ROOT}/stage_a_w4a4c8_score_receipt/op_kernel"
cp "${SPEC}" "${OP_DEV}/ops.json"
cp -a "${SOURCE_ROOT}/op_host/." "${OP_DEV}/op_host/"
cp -a "${KERNEL_ROOT}/." "${OP_DEV}/op_kernel/"
# Stage-B's source ABI imports only the graph receipt definitions from the
# matched native W4A4C8 Stage-A sibling. Its checked-in relative include is
# retained in the fresh package, rather than silently resolving an ambient
# source checkout during CANN compilation.
cp -p "${STAGE_A_ABI}" \
  "${PACKAGE_ROOT}/stage_a_w4a4c8_score_receipt/op_kernel/a5_p10_w4a4c8_score_receipt_abi.h"
cp -p "${KVAF_HEADER}" \
  "${OP_DEV}/op_kernel/kvaf_forward_resident/kvaf_forward_resident_get_device.h"
cp -p "${KVAF_DENSE_SOURCE}" \
  "${OP_DEV}/op_kernel/kvaf_forward_resident/execute_batch_get_aiv.asc"
cp -p "${KVAF_ABI}" \
  "${OP_DEV}/op_kernel/kvaf_forward_resident/batch_get_abi.h"
cp -p "${KVAF_DIRECT_ABI}" \
  "${OP_DEV}/op_kernel/kvaf_forward_resident/direct_hccp_abi.h"

KVAF_HEADER_SHA="$(hash_file "${KVAF_HEADER}")"
STAGE_A_ABI_SHA="$(hash_file "${STAGE_A_ABI}")"
[[ "$(hash_file "${OP_DEV}/op_kernel/kvaf_forward_resident/kvaf_forward_resident_get_device.h")" == "${KVAF_HEADER_SHA}" ]] ||
  fail "staged_kvaf_header_digest_mismatch"
[[ "$(hash_file "${PACKAGE_ROOT}/stage_a_w4a4c8_score_receipt/op_kernel/a5_p10_w4a4c8_score_receipt_abi.h")" == "${STAGE_A_ABI_SHA}" ]] ||
  fail "staged_stage_a_abi_digest_mismatch"

printf 'P10_STAGE_B_BUILD_SOURCE stage=%s source_sha256=%s vllm=%s kvaf=%s package=%s output=%s target=%s jobs=%s\n' \
  "${STAGE_NAME}" "${SOURCE_SHA}" "${VLLM_HEAD}" "${KVAF_HEAD}" \
  "${PACKAGE_ROOT}" "${OUTPUT_ROOT}" "${TARGET_SOC}" "${BUILD_JOBS}"

msopgen gen -i "${OP_DEV}/ops.json" -f aclnn -c "ai_core-${TARGET_SOC}" \
  -lan cpp -op A5P10ScoreReceiptForwardResident -out "${GENERATED}"

# Retain only the checked-in OpDef. The generic msopgen source would add a
# second OP_ADD for the same identity and has no knowledge of the graph alias
# ABI. This generated directory is fresh, so removing that one stub cannot
# affect another package.
cp -a "${OP_DEV}/op_host/." "${GENERATED}/op_host/"
cp -a "${OP_DEV}/op_kernel/." "${GENERATED}/op_kernel/"
# msopgen creates the canonical kernel filename. Replace that generated
# wrapper with the checked-in MIX kernel, whose ABI exactly matches ops.json.
GENERATED_KERNEL="${GENERATED}/op_kernel/a5_p10_score_receipt_forward_resident.cpp"
SOURCE_KERNEL="${GENERATED}/op_kernel/a5_p10_score_receipt_forward_resident_kernel.cpp"
[[ -f "${GENERATED_KERNEL}" ]] || fail "generated_kernel_stub_missing"
[[ -f "${SOURCE_KERNEL}" ]] || fail "source_kernel_missing_after_stage"
cp -p "${SOURCE_KERNEL}" "${GENERATED_KERNEL}"
rm -f "${SOURCE_KERNEL}"
GENERATED_STUB="${GENERATED}/op_host/a5_p10_score_receipt_forward_resident.cpp"
SOURCE_OPDEF="${GENERATED}/op_host/a5_p10_score_receipt_forward_resident_def.cpp"
[[ -f "${GENERATED_STUB}" ]] || fail "generated_host_stub_missing"
[[ -f "${SOURCE_OPDEF}" ]] || fail "source_opdef_missing_after_stage"
REGEX='OP_ADD[[:space:]]*\([[:space:]]*A5P10ScoreReceiptForwardResident[[:space:]]*\)'
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
  CANN_3RD_LIB_PATH="${CANN_3RD_LIB_PATH}" OPS_CPU_NUMBER="${BUILD_JOBS}" \
    bash build.sh
); then
  INI="${GENERATED}/build_out/autogen/aic-${TARGET_SOC}-ops-info.ini"
  SECTION_COUNT=0
  [[ -f "${INI}" ]] && SECTION_COUNT="$(grep -Ec '^\[A5P10ScoreReceiptForwardResident\]$' "${INI}" || true)"
  printf 'P10_STAGE_B_BUILD_DIAG generated_ini=%s registrations=%s staged_kvaf_header_sha256=%s\n' \
    "${INI}" "${SECTION_COUNT}" "${KVAF_HEADER_SHA}" >&2
  exit 1
fi

RUN_PKG="$(find "${GENERATED}/build_out" -maxdepth 1 -type f -name '*.run' -print -quit)"
[[ -n "${RUN_PKG}" ]] || fail "run_package_missing"
chmod +x "${RUN_PKG}"
"${RUN_PKG}" --quiet --install-path="${OUTPUT_ROOT}"

OPAPI="$(find "${OUTPUT_ROOT}" -type f -path '*/op_api/lib/libcust_opapi.so' -print -quit)"
[[ -s "${OPAPI}" ]] || fail "installed_opapi_missing"
mkdir -p "$(dirname "${PROVENANCE_FILE}")"
cat > "${PROVENANCE_FILE}" <<EOF
format=1
stage=stage_b
kind=p10_score_receipt_forward_resident_isolated_opp
operator=A5P10ScoreReceiptForwardResident
vllm_source_root=${VLLM_ROOT}
vllm_source_head=${VLLM_HEAD}
kvaf_source_root=${KVAF_ROOT}
kvaf_source_head=${KVAF_HEAD}
source_sha256=${SOURCE_SHA}
kvaf_forward_resident_header_sha256=${KVAF_HEADER_SHA}
stage_a_receipt_abi_sha256=${STAGE_A_ABI_SHA}
staged_kvaf_closure=${OP_DEV}/op_kernel/kvaf_forward_resident
staged_stage_a_receipt_abi=${PACKAGE_ROOT}/stage_a_w4a4c8_score_receipt/op_kernel/a5_p10_w4a4c8_score_receipt_abi.h
target_soc=${TARGET_SOC}
output_root=${OUTPUT_ROOT}
opapi=${OPAPI}
EOF
printf 'P10_STAGE_PROVENANCE stage=stage_b file=%s vllm=%s kvaf=%s source_sha256=%s kvaf_header_sha256=%s\n' \
  "${PROVENANCE_FILE}" "${VLLM_HEAD}" "${KVAF_HEAD}" "${SOURCE_SHA}" "${KVAF_HEADER_SHA}"
printf 'P10_STAGE_B_BUILD_PASS package=%s opp=%s opapi=%s provenance=%s\n' \
  "${PACKAGE_ROOT}" "${OUTPUT_ROOT}" "${OPAPI}" "${PROVENANCE_FILE}"
