#!/usr/bin/env bash
# Build the isolated P9 source-derived C8 + canonical-resident Get OPP.
#
# This is deliberately independent from build_nanovllm_ops.sh and from every
# retired P7/P8/streaming OPP tree.  It creates a fresh package directory
# keyed by the *exact* vLLM and KVAF commits, stages the official C8 source
# closure, then overlays only manifest-listed P9 files.  It never reuses a
# CMake cache, an OPP, or a bridge from another source pair.
#
# The currently committed P9 tree intentionally contains only the source/ABI
# contract, not the final host/kernel overlay.  In that state --stage-only is
# useful to prove the source closure; a normal build stops with one explicit
# P9_OVERLAY_MANIFEST diagnostic before invoking CANN.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_VLLM_ROOT="$(cd "${THIS_DIR}/../../../.." && pwd -P)"

VLLM_ROOT="${A5_C8_EXACT_RESIDENT_SOURCE_ROOT:-${DEFAULT_VLLM_ROOT}}"
KVAF_ROOT="${A5_C8_EXACT_RESIDENT_KVAF_SOURCE_ROOT:-${KVAF_FUSED:-}}"
EXPECTED_VLLM_HEAD="${A5_C8_EXACT_RESIDENT_EXPECTED_SOURCE_HEAD:-}"
EXPECTED_KVAF_HEAD="${A5_C8_EXACT_RESIDENT_EXPECTED_KVAF_HEAD:-}"
PACKAGE_PARENT="${A5_C8_EXACT_RESIDENT_PACKAGE_PARENT:-${THIS_DIR}/build-p9-exact-resident}"
CANN_3RD_LIB_PATH="${A5_C8_EXACT_RESIDENT_CANN_3RD_LIB_PATH:-}"
BUILD_JOBS="${A5_C8_EXACT_RESIDENT_BUILD_JOBS:-4}"
TARGET_SOC_RAW="${A5_C8_EXACT_RESIDENT_SOC:-ascend950}"
TARGET_OPERATOR="${A5_C8_EXACT_RESIDENT_OPERATOR:-A5C8ExactSpeculativeResidentGet}"

MODE="build"
if [[ "${1:-}" == "--stage-only" ]]; then
  MODE="stage"
elif [[ -n "${1:-}" ]]; then
  printf 'P9_OPP_BUILD_FAIL reason=invalid_argument value=%s\n' "$1" >&2
  exit 2
fi

fail() {
  printf 'P9_OPP_BUILD_FAIL reason=%s\n' "$1" >&2
  exit 2
}

require_clean_git_root() {
  local label="$1"
  local root="$2"
  local expected="$3"
  local head dirty

  [[ -e "${root}/.git" ]] || fail "${label}_git_root_missing:path=${root}"
  head="$(git -C "${root}" rev-parse HEAD 2>/dev/null || true)"
  [[ -n "${head}" ]] || fail "${label}_head_unavailable:path=${root}"
  [[ -n "${expected}" ]] || fail "${label}_expected_head_missing"
  [[ "${head}" == "${expected}" ]] ||
    fail "${label}_head_mismatch:actual=${head}:expected=${expected}"
  dirty="$(git -C "${root}" status --porcelain --untracked-files=no 2>/dev/null || true)"
  [[ -z "${dirty}" ]] || fail "${label}_tracked_changes_present:path=${root}"
  printf 'P9_OPP_SOURCE label=%s root=%s head=%s expected=%s dirty=0\n' \
    "${label}" "${root}" "${head}" "${expected}"
}

TARGET_SOC_NORMALIZED="$(printf '%s' "${TARGET_SOC_RAW}" | tr '[:upper:]' '[:lower:]')"
case "${TARGET_SOC_NORMALIZED}" in
  ascend950|ascend950pr*|ascend950dt*) TARGET_SOC="ascend950" ;;
  *) fail "unsupported_soc:value=${TARGET_SOC_RAW}" ;;
esac
[[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || fail "invalid_jobs:value=${BUILD_JOBS}"
[[ -n "${KVAF_ROOT}" ]] || fail "kvaf_source_root_missing"

# Do not inherit the historical product selector: it repeatedly selected a
# different CANN product for isolated A5 OPPs.
[[ -z "${SOC_VERSION:-}" ]] || fail "legacy_SOC_VERSION_set"

require_clean_git_root vllm "${VLLM_ROOT}" "${EXPECTED_VLLM_HEAD}"
require_clean_git_root kvaf "${KVAF_ROOT}" "${EXPECTED_KVAF_HEAD}"

VLLM_HEAD="$(git -C "${VLLM_ROOT}" rev-parse HEAD)"
KVAF_HEAD="$(git -C "${KVAF_ROOT}" rev-parse HEAD)"
VLLM_SHA12="${VLLM_HEAD:0:12}"
KVAF_SHA12="${KVAF_HEAD:0:12}"
PACKAGE_TAG="p9_exact_resident_v${VLLM_SHA12}_k${KVAF_SHA12}_$(date +%Y%m%d_%H%M%S)"
PACKAGE_ROOT="${A5_C8_EXACT_RESIDENT_PACKAGE_ROOT:-${PACKAGE_PARENT}/${PACKAGE_TAG}}"

case "${PACKAGE_ROOT}" in
  "${PACKAGE_PARENT}"/*) ;;
  *) fail "unsafe_package_root:path=${PACKAGE_ROOT}:parent=${PACKAGE_PARENT}" ;;
esac
[[ ! -e "${PACKAGE_ROOT}" ]] || fail "fresh_package_root_exists:path=${PACKAGE_ROOT}"

OFFICIAL_C8_ROOT="${VLLM_ROOT}/csrc/attention/vllm_quant_lightning_indexer"
P9_ROOT="${VLLM_ROOT}/csrc/attention/nanovllm_a5_ops/a5_c8_exact_speculative_resident_get"
OVERLAY_ROOT="${P9_ROOT}/source_overlay"
P9_CONTRACT_HEADER="${P9_ROOT}/op_kernel/arch35/a5_c8_exact_speculative_resident_contract.h"
KVAF_DEMO_ROOT="${KVAF_ROOT}/access_operator/demo/aiv_batch_get"
KVAF_RESIDENT_HEADER="${KVAF_DEMO_ROOT}/kernel/kvaf_layer_resident_get_device.h"
KVAF_DENSE_SOURCE="${KVAF_DEMO_ROOT}/kernel/execute_batch_get_aiv.asc"
KVAF_ABI_HEADER="${KVAF_DEMO_ROOT}/include/batch_get_abi.h"
KVAF_DIRECT_ABI_HEADER="${KVAF_DEMO_ROOT}/include/direct_hccp_abi.h"

for required in \
  "${OFFICIAL_C8_ROOT}/CMakeLists.txt" \
  "${OFFICIAL_C8_ROOT}/op_kernel/vllm_quant_lightning_indexer.cpp" \
  "${OFFICIAL_C8_ROOT}/op_kernel/arch35/quant_lightning_indexer_service_vector.h" \
  "${OFFICIAL_C8_ROOT}/op_host/vllm_quant_lightning_indexer_def.cpp" \
  "${P9_CONTRACT_HEADER}" \
  "${KVAF_RESIDENT_HEADER}" \
  "${KVAF_DENSE_SOURCE}" \
  "${KVAF_ABI_HEADER}" \
  "${KVAF_DIRECT_ABI_HEADER}"; do
  [[ -f "${required}" ]] || fail "source_closure_file_missing:path=${required}"
done
[[ -d "${OVERLAY_ROOT}" ]] || fail "p9_overlay_root_missing:path=${OVERLAY_ROOT}"

# The source-derived C8 target must not silently import the retired private
# arena/score-receipt implementation.  These names are forbidden in the P9
# overlay's actual include graph; comments in the KVAF ABI are intentionally
# not scanned because that shared ABI still documents historical structures.
if grep -RInE '^#[[:space:]]*include[[:space:]]*[<"][^>"]*(a5_streaming_quant_li_get|a5_p7_|a5_p8_|a5_score_receipt)' \
    "${OVERLAY_ROOT}" "${P9_CONTRACT_HEADER}" 2>/dev/null; then
  fail "legacy_p7_p8_streaming_include_in_p9_overlay"
fi

if [[ "${MODE}" == "build" ]]; then
  [[ -n "${CANN_3RD_LIB_PATH}" ]] || fail "cann_third_party_cache_path_missing"
  CACHE_MAKESELF="${CANN_3RD_LIB_PATH}/makeself/makeself.sh"
  CACHE_MAKESELF_HEADER="${CANN_3RD_LIB_PATH}/makeself/makeself-header.sh"
  CACHE_JSON_HEADER="${CANN_3RD_LIB_PATH}/json/include/nlohmann/json.hpp"
  CACHE_ABSEIL_ARCHIVE="${CANN_3RD_LIB_PATH}/abseil-cpp/abseil-cpp-20230802.1.tar.gz"
  CACHE_ABSEIL_PKG_ARCHIVE="${CANN_3RD_LIB_PATH}/pkg/abseil-cpp-20230802.1.tar.gz"
  CACHE_PROTOBUF_ARCHIVE="${CANN_3RD_LIB_PATH}/protobuf/protobuf-all-25.1.tar.gz"
  CACHE_PROTOBUF_PKG_ARCHIVE="${CANN_3RD_LIB_PATH}/pkg/protobuf-25.1.tar.gz"
  if [[ ! -r "${CACHE_MAKESELF}" || ! -r "${CACHE_MAKESELF_HEADER}" || \
        ! -r "${CACHE_JSON_HEADER}" ]] || \
     { [[ ! -r "${CACHE_ABSEIL_ARCHIVE}" ]] && [[ ! -r "${CACHE_ABSEIL_PKG_ARCHIVE}" ]]; } || \
     { [[ ! -r "${CACHE_PROTOBUF_ARCHIVE}" ]] && [[ ! -r "${CACHE_PROTOBUF_PKG_ARCHIVE}" ]]; }; then
    fail "cann_third_party_cache_incomplete:path=${CANN_3RD_LIB_PATH}:need=makeself_json_abseil_protobuf"
  fi
  CANN_3RD_LIB_PATH="$(cd "${CANN_3RD_LIB_PATH}" && pwd -P)"
  command -v msopgen >/dev/null 2>&1 || fail "msopgen_missing"
fi

mkdir -p "${PACKAGE_PARENT}"
mkdir "${PACKAGE_ROOT}"
STAGE_ROOT="${PACKAGE_ROOT}/source"
STAGED_C8_ROOT="${STAGE_ROOT}/vllm_quant_lightning_indexer"
STAGED_OVERLAY_ROOT="${STAGE_ROOT}/p9_overlay_input"
STAGED_KVAF_ROOT="${STAGED_C8_ROOT}/op_kernel/kvaf_layer_resident"
GENERATED_ROOT="${PACKAGE_ROOT}/generated"
INSTALL_ROOT="${PACKAGE_ROOT}/opp"

# `cp -a` is intentionally source-only.  No generated project, local OPP,
# CMake cache, or torch bridge is copied into this fresh SHA-pair package.
mkdir -p "${STAGED_C8_ROOT}" "${STAGED_OVERLAY_ROOT}" "${STAGED_KVAF_ROOT}"
cp -a "${OFFICIAL_C8_ROOT}/." "${STAGED_C8_ROOT}/"
cp -a "${OVERLAY_ROOT}/." "${STAGED_OVERLAY_ROOT}/"
install -m 0644 "${P9_CONTRACT_HEADER}" \
  "${STAGED_C8_ROOT}/op_kernel/arch35/a5_c8_exact_speculative_resident_contract.h"
install -m 0644 "${KVAF_RESIDENT_HEADER}" \
  "${STAGED_KVAF_ROOT}/kvaf_layer_resident_get_device.h"
install -m 0644 "${KVAF_DENSE_SOURCE}" \
  "${STAGED_KVAF_ROOT}/execute_batch_get_aiv.asc"
install -m 0644 "${KVAF_ABI_HEADER}" "${STAGED_KVAF_ROOT}/batch_get_abi.h"
install -m 0644 "${KVAF_DIRECT_ABI_HEADER}" \
  "${STAGED_KVAF_ROOT}/direct_hccp_abi.h"

cat > "${PACKAGE_ROOT}/P9_PROVENANCE" <<EOF
format=1
kind=source_derived_c8_exact_speculative_resident
operator=${TARGET_OPERATOR}
mode=${MODE}
vllm_source_root=${VLLM_ROOT}
vllm_source_head=${VLLM_HEAD}
kvaf_source_root=${KVAF_ROOT}
kvaf_source_head=${KVAF_HEAD}
official_c8_source_root=${OFFICIAL_C8_ROOT}
p9_overlay_root=${OVERLAY_ROOT}
staged_c8_root=${STAGED_C8_ROOT}
staged_kvaf_root=${STAGED_KVAF_ROOT}
target_soc=${TARGET_SOC}
cann_3rd_lib_path=${CANN_3RD_LIB_PATH:-<stage-only>}
EOF

printf 'P9_OPP_STAGE source_vllm=%s source_kvaf=%s package=%s staged_c8=%s\n' \
  "${VLLM_HEAD}" "${KVAF_HEAD}" "${PACKAGE_ROOT}" "${STAGED_C8_ROOT}"

if [[ "${MODE}" == "stage" ]]; then
  printf 'P9_OPP_STAGE_PASS provenance=%s\n' "${PACKAGE_ROOT}/P9_PROVENANCE"
  exit 0
fi

# The manifest keeps the actual P9 host/kernel API out of this generic source
# closure script.  It is a line-oriented list of regular files relative to
# source_overlay/, copied over the staged official C8 tree.  This avoids
# inventing an ACLNN signature before the target source exists.
OVERLAY_MANIFEST="${OVERLAY_ROOT}/P9_OVERLAY_MANIFEST"
[[ -f "${OVERLAY_MANIFEST}" ]] ||
  fail "p9_overlay_manifest_missing:path=${OVERLAY_MANIFEST}:use_stage_only_until_target_sources_exist"

OVERLAY_FILES=()
while IFS= read -r relative_path || [[ -n "${relative_path}" ]]; do
  [[ -z "${relative_path}" || "${relative_path}" == \#* ]] && continue
  [[ "${relative_path}" != /* && "${relative_path}" != *".."* ]] ||
    fail "p9_overlay_manifest_unsafe_path:value=${relative_path}"
  [[ -f "${OVERLAY_ROOT}/${relative_path}" ]] ||
    fail "p9_overlay_file_missing:path=${OVERLAY_ROOT}/${relative_path}"
  OVERLAY_FILES+=("${relative_path}")
done < "${OVERLAY_MANIFEST}"
[[ "${#OVERLAY_FILES[@]}" -gt 0 ]] || fail "p9_overlay_manifest_empty"

for relative_path in "${OVERLAY_FILES[@]}"; do
  destination="${STAGED_C8_ROOT}/${relative_path}"
  mkdir -p "$(dirname "${destination}")"
  install -m 0644 "${OVERLAY_ROOT}/${relative_path}" "${destination}"
done

[[ -f "${STAGED_C8_ROOT}/ops.json" ]] || fail "p9_overlay_missing_ops_json"
[[ -d "${STAGED_C8_ROOT}/op_host" && -d "${STAGED_C8_ROOT}/op_kernel" ]] ||
  fail "p9_overlay_missing_host_or_kernel_tree"
if grep -RInE '^#[[:space:]]*include[[:space:]]*[<"][^>"]*(a5_streaming_quant_li_get|a5_p7_|a5_p8_|a5_score_receipt)' \
    "${STAGED_C8_ROOT}/op_host" "${STAGED_C8_ROOT}/op_kernel" 2>/dev/null; then
  fail "legacy_p7_p8_streaming_include_after_overlay"
fi

# Generate a fresh CANN project.  The generated generic host stub is removed
# only when it registers the same target; the overlay must leave exactly one
# source-derived OP_ADD registration.  This prevents the repeated duplicate
# OP_ADD/INI-section failure without editing generated metadata.
msopgen gen -i "${STAGED_C8_ROOT}/ops.json" -f aclnn \
  -c "ai_core-${TARGET_SOC}" -lan cpp -op "${TARGET_OPERATOR}" \
  -out "${GENERATED_ROOT}"

mapfile -t GENERATED_HOST_SOURCES < <(
  find "${GENERATED_ROOT}/op_host" -type f \
    \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print | sort
)
cp -a "${STAGED_C8_ROOT}/op_host/." "${GENERATED_ROOT}/op_host/"
cp -a "${STAGED_C8_ROOT}/op_kernel/." "${GENERATED_ROOT}/op_kernel/"

OP_ADD_REGEX="OP_ADD[[:space:]]*\\([[:space:]]*${TARGET_OPERATOR}[[:space:]]*\\)"
for generated_source in "${GENERATED_HOST_SOURCES[@]}"; do
  if [[ -f "${generated_source}" ]] && grep -Eq "${OP_ADD_REGEX}" "${generated_source}"; then
    rm -f "${generated_source}"
  fi
done

mapfile -t OP_ADD_SOURCES < <(
  while IFS= read -r source_file; do
    if grep -Eq "${OP_ADD_REGEX}" "${source_file}"; then
      printf '%s\n' "${source_file}"
    fi
  done < <(find "${GENERATED_ROOT}/op_host" -type f \
    \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print | sort)
)
[[ "${#OP_ADD_SOURCES[@]}" -eq 1 ]] ||
  fail "p9_op_add_registration_count:count=${#OP_ADD_SOURCES[@]}:operator=${TARGET_OPERATOR}"

STAGED_MAKESELF_DIR="${GENERATED_ROOT}/third_party/makeself"
mkdir -p "${STAGED_MAKESELF_DIR}"
install -m 0755 "${CANN_3RD_LIB_PATH}/makeself/makeself.sh" \
  "${STAGED_MAKESELF_DIR}/makeself.sh"
install -m 0644 "${CANN_3RD_LIB_PATH}/makeself/makeself-header.sh" \
  "${STAGED_MAKESELF_DIR}/makeself-header.sh"

(
  cd "${GENERATED_ROOT}"
  OPS_CPU_NUMBER="${BUILD_JOBS}" bash build.sh \
    --cann_3rd_lib_path="${CANN_3RD_LIB_PATH}"
)

mapfile -t RUN_PACKAGES < <(
  find "${GENERATED_ROOT}/build_out" -maxdepth 1 -type f -name '*.run' -print | sort
)
[[ "${#RUN_PACKAGES[@]}" -eq 1 ]] ||
  fail "p9_run_package_ambiguous:count=${#RUN_PACKAGES[@]}:root=${GENERATED_ROOT}/build_out"
chmod +x "${RUN_PACKAGES[0]}"
"${RUN_PACKAGES[0]}" --quiet --install-path="${INSTALL_ROOT}"

mapfile -t OPAPI_LIBS < <(
  find "${INSTALL_ROOT}/vendors" -type f -path '*/op_api/lib/libcust_opapi.so' -print | sort
)
[[ "${#OPAPI_LIBS[@]}" -eq 1 ]] ||
  fail "p9_opapi_ambiguous:count=${#OPAPI_LIBS[@]}:root=${INSTALL_ROOT}"
VENDOR_ROOT="$(cd "$(dirname "${OPAPI_LIBS[0]}")/../.." && pwd -P)"
cp "${PACKAGE_ROOT}/P9_PROVENANCE" "${VENDOR_ROOT}/P9_PROVENANCE"

printf 'P9_OPP_BUILD_PASS vllm=%s kvaf=%s package=%s vendor=%s opapi=%s provenance=%s\n' \
  "${VLLM_HEAD}" "${KVAF_HEAD}" "${PACKAGE_ROOT}" "${VENDOR_ROOT}" \
  "${OPAPI_LIBS[0]}" "${VENDOR_ROOT}/P9_PROVENANCE"
