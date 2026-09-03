#!/usr/bin/env bash
# Assemble the P10 product OPPs from one exact, clean vLLM/KVAF source pair.
#
# P10 has three independently-owned source components. They intentionally do
# not share a generated CMake directory or an ambient custom-OPP path:
#
#   ForwardPrepare: graph-visible forward-local reset node
#   Stage A: native W4A4C8 score surface + receipt publisher
#   Stage B: receipt consumer + forward-resident Direct-HCCP executor
#
# A build is admitted only from a digest-keyed package directory.  The package
# owns source-only copies of all three directories and records their digests
# with the exact vLLM/KVAF revisions. A package can be reused only when its
# complete manifest and exactly one combined OPP API library still match;
# existence alone is never evidence that it was built from the current pair.
#
# In `stage` mode this script performs all source/provenance/staging checks but
# does not require a CANN toolchain. In `build` mode one combined recipe
# receives the three staged closures and emits one vendor containing all three
# symbols. This prevents a bridge from resolving a mixed set of separately
# installed packages.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_VLLM_ROOT="$(cd "${THIS_DIR}/../../../.." && pwd -P)"

VLLM_ROOT="${P10_BUILD_VLLM_ROOT:-${DEFAULT_VLLM_ROOT}}"
KVAF_ROOT="${P10_BUILD_KVAF_ROOT:-${KVAF_FUSED:-}}"
EXPECTED_VLLM_SHA="${P10_BUILD_EXPECTED_VLLM_SHA:-}"
EXPECTED_KVAF_SHA="${P10_BUILD_EXPECTED_KVAF_SHA:-}"
MODE="${P10_BUILD_MODE:-build}"
BUILD_JOBS="${P10_BUILD_JOBS:-4}"
PACKAGE_PARENT="${P10_BUILD_PACKAGE_PARENT:-${THIS_DIR}/build-p10-target}"
PACKAGE_ROOT="${P10_BUILD_PACKAGE_ROOT:-}"

STAGE_A_REL="csrc/attention/nanovllm_a5_ops/a5_dsa_c8_exact_speculative_resident/stage_a_w4a4c8_score_receipt"
STAGE_B_REL="csrc/attention/nanovllm_a5_ops/a5_dsa_c8_exact_speculative_resident/stage_b_score_receipt_forward_resident"
FORWARD_REL="csrc/attention/nanovllm_a5_ops/a5_dsa_c8_exact_speculative_resident/p10_forward_bridge"
PRODUCT_RECIPE_REL="csrc/attention/nanovllm_a5_ops/a5_dsa_c8_exact_speculative_resident/build_p10_product_opp.sh"
CANN_3RD_LIB_PATH="${P10_BUILD_CANN_3RD_LIB_PATH:-}"
PRODUCT_PYTHON="${P10_BUILD_PYTHON:-$(command -v python3 2>/dev/null || true)}"

fail() {
    printf 'P10_ASSEMBLY_FAIL reason=%s\n' "$1" >&2
    exit 2
}

case "${MODE}" in
    stage|build) ;;
    *) fail "invalid_mode:value=${MODE}" ;;
esac
[[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || fail "invalid_jobs:value=${BUILD_JOBS}"
[[ -n "${EXPECTED_VLLM_SHA}" ]] || fail "expected_vllm_sha_missing"
[[ -n "${EXPECTED_KVAF_SHA}" ]] || fail "expected_kvaf_sha_missing"
[[ -n "${KVAF_ROOT}" ]] || fail "kvaf_root_missing"
[[ -z "${SOC_VERSION:-}" ]] || fail "legacy_SOC_VERSION_set"

require_clean_exact_git_root() {
    local label="$1"
    local root="$2"
    local expected="$3"
    local head dirty

    [[ -e "${root}/.git" ]] || fail "${label}_git_root_missing:path=${root}"
    head="$(git -C "${root}" rev-parse HEAD 2>/dev/null || true)"
    [[ -n "${head}" ]] || fail "${label}_head_unavailable:path=${root}"
    [[ "${head}" == "${expected}" ]] ||
        fail "${label}_head_mismatch:actual=${head}:expected=${expected}"
    dirty="$(git -C "${root}" status --porcelain --untracked-files=no 2>/dev/null || true)"
    [[ -z "${dirty}" ]] || fail "${label}_tracked_changes_present:path=${root}"
    printf 'P10_ASSEMBLY_SOURCE label=%s root=%s head=%s expected=%s dirty=0\n' \
        "${label}" "${root}" "${head}" "${expected}"
}

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

# Hash only regular source files.  Generated CANN projects, Python bytecode,
# and prior OPP packages are never copied or fingerprinted as P10 source.
tree_digest() {
    local root="$1"
    local list file rel file_hash
    list="$(mktemp "${TMPDIR:-/tmp}/p10-tree.XXXXXX")"
    while IFS= read -r -d '' file; do
        rel="${file#${root}/}"
        case "${rel}" in
            .git/*|build/*|build-*/*|*/build/*|*/build-*/*|op_dev/*|custom_op/*|opp/*|_custom_opp*/*|*/_custom_opp*/*|packages/*|*/packages/*|__pycache__/*|*/__pycache__/*|*.pyc|*.o|*.so|*.run|CMakeCache.txt)
                continue
                ;;
        esac
        printf '%s\0' "${rel}" >> "${list}"
    done < <(find "${root}" -type f -print0)

    # A source directory without regular files is never a valid stage.
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

copy_source_tree() {
    local source="$1"
    local destination="$2"
    local file rel target
    mkdir -p "${destination}"
    while IFS= read -r -d '' file; do
        rel="${file#${source}/}"
        case "${rel}" in
            .git/*|build/*|build-*/*|*/build/*|*/build-*/*|op_dev/*|custom_op/*|opp/*|_custom_opp*/*|*/_custom_opp*/*|packages/*|*/packages/*|__pycache__/*|*/__pycache__/*|*.pyc|*.o|*.so|*.run|CMakeCache.txt)
                continue
                ;;
        esac
        target="${destination}/${rel}"
        mkdir -p "$(dirname "${target}")"
        cp -p "${file}" "${target}"
    done < <(find "${source}" -type f -print0)
}

require_stage_source() {
    local label="$1"
    local source="$2"
    [[ -d "${source}" ]] || fail "${label}_source_missing:path=${source}"
    [[ -d "${source}/op_host" ]] || fail "${label}_op_host_missing:path=${source}/op_host"
    [[ -d "${source}/op_kernel" ]] || fail "${label}_op_kernel_missing:path=${source}/op_kernel"
    [[ -f "${source}/VENDORED.md" ]] || fail "${label}_vendored_provenance_missing:path=${source}/VENDORED.md"
    # The source closure may describe legacy paths in Markdown, but compiled
    # source must not include or shell out to them.
    if grep -RInE '^#[[:space:]]*include[[:space:]]*[<"][^>"]*(a5_p8_|a5_c8_exact_speculative_resident_get|a5_streaming_quant_li_get|a5_score_receipt_speculative_executor)' \
        "${source}/op_host" "${source}/op_kernel" 2>/dev/null; then
        fail "${label}_legacy_include_detected:path=${source}"
    fi
}

require_forward_preinit_boundary() {
    # Receipt ownership is established by the graph's forward-prep node before
    # Stage A and its concurrent Stage-B companion launch.  Stage A may only
    # transition a caller-preinitialized record through FILLING -> READY; it
    # must never clear receipt_state during its score loop.  This cheap source
    # gate prevents a build from reintroducing the race as a hidden init hook.
    local stage_a="$1"
    local stage_b="$2"
    local forward="$3"
    local qli="${stage_a}/op_kernel/a5_p10_w4a4c8_score_receipt_qli.h"
    local vector="${stage_a}/op_kernel/arch35/quant_lightning_indexer_service_vector.h"
    local abi="${stage_a}/op_kernel/a5_p10_w4a4c8_score_receipt_abi.h"
    for required in "${qli}" "${vector}" "${abi}"; do
        [[ -f "${required}" ]] ||
            fail "stage_a_forward_preinit_source_missing:path=${required}"
    done
    grep -Fq 'forward_control' "${abi}" ||
        fail "stage_a_forward_control_abi_missing"
    grep -Rqs 'Input("forward_control")' "${stage_b}/op_host" ||
        fail "stage_b_forward_control_input_missing"
    grep -Fq 'OP_ADD(A5P10ForwardPrepare)' \
        "${forward}/op_host/a5_p10_forward_prepare_def.cpp" ||
        fail "forward_prepare_registration_missing"
    grep -Fq 'must_run_before' "${forward}/P10_FORWARD_BRIDGE_CONTRACT.json" ||
        fail "forward_prepare_graph_contract_missing"
    if grep -Fq 'InitializeReceiptStateShard' "${qli}" || \
       grep -Fq 'InitializeReceiptStateShard' "${vector}"; then
        fail "stage_a_illegal_receipt_init_hook_present"
    fi
}

require_forward_source() {
    local source="$1"
    for required in \
        "${source}/ops.json" \
        "${source}/P10_FORWARD_BRIDGE_CONTRACT.json" \
        "${source}/build_p10_forward_prepare_opp.sh" \
        "${source}/op_host/a5_p10_forward_prepare_def.cpp" \
        "${source}/op_kernel/a5_p10_forward_prepare.cpp" \
        "${source}/torch_extension/p10_forward_bridge_setup.py"; do
        [[ -f "${required}" ]] ||
            fail "forward_prepare_source_missing:path=${required}"
    done
}

require_clean_exact_git_root vllm "${VLLM_ROOT}" "${EXPECTED_VLLM_SHA}"
require_clean_exact_git_root kvaf "${KVAF_ROOT}" "${EXPECTED_KVAF_SHA}"

VLLM_HEAD="$(git -C "${VLLM_ROOT}" rev-parse HEAD)"
KVAF_HEAD="$(git -C "${KVAF_ROOT}" rev-parse HEAD)"
STAGE_A_SOURCE="${VLLM_ROOT}/${STAGE_A_REL}"
STAGE_B_SOURCE="${VLLM_ROOT}/${STAGE_B_REL}"
FORWARD_SOURCE="${VLLM_ROOT}/${FORWARD_REL}"
PRODUCT_RECIPE="${VLLM_ROOT}/${PRODUCT_RECIPE_REL}"
require_stage_source stage_a "${STAGE_A_SOURCE}"
require_stage_source stage_b "${STAGE_B_SOURCE}"
require_forward_source "${FORWARD_SOURCE}"
require_forward_preinit_boundary "${STAGE_A_SOURCE}" "${STAGE_B_SOURCE}" "${FORWARD_SOURCE}"
[[ -r "${PRODUCT_RECIPE}" ]] || fail "product_recipe_missing:path=${PRODUCT_RECIPE}"

STAGE_A_SHA="$(tree_digest "${STAGE_A_SOURCE}")"
STAGE_B_SHA="$(tree_digest "${STAGE_B_SOURCE}")"
FORWARD_SHA="$(tree_digest "${FORWARD_SOURCE}")"

PACKAGE_PARENT="$(mkdir -p "${PACKAGE_PARENT}" && cd "${PACKAGE_PARENT}" && pwd -P)"
if [[ -z "${PACKAGE_ROOT}" ]]; then
    # Stage-only and build roots are separate so a deliberately source-only
    # audit can never be mistaken for a complete OPP package on the next run.
    PACKAGE_ROOT="${PACKAGE_PARENT}/p10_${MODE}_v${VLLM_HEAD:0:12}_k${KVAF_HEAD:0:12}_f${FORWARD_SHA:0:12}_a${STAGE_A_SHA:0:12}_b${STAGE_B_SHA:0:12}"
fi
case "${PACKAGE_ROOT}" in
    "${PACKAGE_PARENT}"/*) ;;
    *) fail "unsafe_package_root:path=${PACKAGE_ROOT}:parent=${PACKAGE_PARENT}" ;;
esac
INPUT_MANIFEST_NAME="P10_ASSEMBLY_INPUT_PROVENANCE"
FINAL_MANIFEST_NAME="P10_ASSEMBLY_PROVENANCE"

manifest_has_exact_input() {
    local manifest="$1"
    [[ -s "${manifest}" ]] || return 1
    grep -Fqx 'format=1' "${manifest}" &&
    grep -Fqx 'kind=p10_target_three_component_combined_assembly' "${manifest}" &&
    grep -Fqx "mode=${MODE}" "${manifest}" &&
    grep -Fqx "vllm_source_root=${VLLM_ROOT}" "${manifest}" &&
    grep -Fqx "vllm_source_head=${VLLM_HEAD}" "${manifest}" &&
    grep -Fqx "kvaf_source_root=${KVAF_ROOT}" "${manifest}" &&
    grep -Fqx "kvaf_source_head=${KVAF_HEAD}" "${manifest}" &&
    grep -Fqx "stage_a_source_relative=${STAGE_A_REL}" "${manifest}" &&
    grep -Fqx "stage_a_source_sha256=${STAGE_A_SHA}" "${manifest}" &&
    grep -Fqx "stage_b_source_relative=${STAGE_B_REL}" "${manifest}" &&
    grep -Fqx "stage_b_source_sha256=${STAGE_B_SHA}" "${manifest}" &&
    grep -Fqx "forward_prepare_source_relative=${FORWARD_REL}" "${manifest}" &&
    grep -Fqx "forward_prepare_source_sha256=${FORWARD_SHA}" "${manifest}" &&
    grep -Fqx "staged_stage_a=${PACKAGE_ROOT}/source/stage_a" "${manifest}" &&
    grep -Fqx "staged_stage_b=${PACKAGE_ROOT}/source/stage_b" "${manifest}" &&
    grep -Fqx "staged_forward_prepare=${PACKAGE_ROOT}/source/forward_prepare" "${manifest}" &&
    grep -Fqx "build_jobs=${BUILD_JOBS}" "${manifest}" &&
    grep -Fqx 'legacy_overlay_reuse=forbidden' "${manifest}" &&
    grep -Fqx 'artifact_reuse=manifest_and_opapi_hash_only' "${manifest}"
}

validate_reusable_package() {
    local manifest="${PACKAGE_ROOT}/${FINAL_MANIFEST_NAME}"
    local opapi_count opapi recorded_hash actual_hash product_manifest
    manifest_has_exact_input "${manifest}" || return 1
    if [[ "${MODE}" == "stage" ]]; then
        grep -Fqx 'assembly_state=staged_only' "${manifest}" || return 1
        return 0
    fi
    grep -Fqx 'assembly_state=complete' "${manifest}" || return 1
    product_manifest="${PACKAGE_ROOT}/product/P10_PRODUCT_PROVENANCE"
    [[ -s "${product_manifest}" ]] || return 1
    grep -Fqx 'kind=p10_combined_prepare_stage_a_stage_b_product_opp' "${product_manifest}" || return 1
    grep -Fqx "vllm_source_head=${VLLM_HEAD}" "${product_manifest}" || return 1
    grep -Fqx "kvaf_source_head=${KVAF_HEAD}" "${product_manifest}" || return 1
    grep -Fqx "stage_a_source_sha256=${STAGE_A_SHA}" "${product_manifest}" || return 1
    grep -Fqx "stage_b_source_sha256=${STAGE_B_SHA}" "${product_manifest}" || return 1
    grep -Fqx "forward_prepare_source_sha256=${FORWARD_SHA}" "${product_manifest}" || return 1
    grep -Fqx 'operators=A5P10ForwardPrepare,A5P10W4A4C8ScoreReceipt,A5P10ScoreReceiptForwardResident' "${product_manifest}" || return 1
    opapi_count="$(find "${PACKAGE_ROOT}/product/opp" -type f -path '*/op_api/lib/libcust_opapi.so' -print 2>/dev/null | wc -l | tr -d ' ')"
    [[ "${opapi_count}" == 1 ]] || return 1
    opapi="$(find "${PACKAGE_ROOT}/product/opp" -type f -path '*/op_api/lib/libcust_opapi.so' -print -quit)"
    actual_hash="$(hash_file "${opapi}")"
    recorded_hash="$(sed -n 's/^product_opapi_sha256=//p' "${manifest}" | head -n 1)"
    [[ -n "${recorded_hash}" && "${recorded_hash}" == "${actual_hash}" ]] || return 1
    return 0
}

if [[ -e "${PACKAGE_ROOT}" ]]; then
    if validate_reusable_package; then
        printf 'P10_ASSEMBLY_REUSE_PASS package=%s manifest=%s\n' \
            "${PACKAGE_ROOT}" "${PACKAGE_ROOT}/${FINAL_MANIFEST_NAME}"
        exit 0
    fi
    fail "existing_package_manifest_or_artifact_mismatch:path=${PACKAGE_ROOT}"
fi

mkdir "${PACKAGE_ROOT}"
STAGED_ROOT="${PACKAGE_ROOT}/source"
STAGED_A="${STAGED_ROOT}/stage_a"
STAGED_B="${STAGED_ROOT}/stage_b"
STAGED_FORWARD="${STAGED_ROOT}/forward_prepare"
copy_source_tree "${STAGE_A_SOURCE}" "${STAGED_A}"
copy_source_tree "${STAGE_B_SOURCE}" "${STAGED_B}"
copy_source_tree "${FORWARD_SOURCE}" "${STAGED_FORWARD}"

# Re-digest staged input, so an unexpectedly omitted file cannot become a
# different build closure with a valid-looking provenance stamp.
[[ "$(tree_digest "${STAGED_A}")" == "${STAGE_A_SHA}" ]] ||
    fail "stage_a_copy_digest_mismatch"
[[ "$(tree_digest "${STAGED_B}")" == "${STAGE_B_SHA}" ]] ||
    fail "stage_b_copy_digest_mismatch"
[[ "$(tree_digest "${STAGED_FORWARD}")" == "${FORWARD_SHA}" ]] ||
    fail "forward_prepare_copy_digest_mismatch"

INPUT_MANIFEST="${PACKAGE_ROOT}/${INPUT_MANIFEST_NAME}"
cat > "${INPUT_MANIFEST}" <<EOF
format=1
kind=p10_target_three_component_combined_assembly
mode=${MODE}
vllm_source_root=${VLLM_ROOT}
vllm_source_head=${VLLM_HEAD}
kvaf_source_root=${KVAF_ROOT}
kvaf_source_head=${KVAF_HEAD}
stage_a_source_relative=${STAGE_A_REL}
stage_a_source_sha256=${STAGE_A_SHA}
stage_b_source_relative=${STAGE_B_REL}
stage_b_source_sha256=${STAGE_B_SHA}
forward_prepare_source_relative=${FORWARD_REL}
forward_prepare_source_sha256=${FORWARD_SHA}
staged_stage_a=${STAGED_A}
staged_stage_b=${STAGED_B}
staged_forward_prepare=${STAGED_FORWARD}
build_jobs=${BUILD_JOBS}
legacy_overlay_reuse=forbidden
artifact_reuse=manifest_and_opapi_hash_only
EOF

printf 'P10_ASSEMBLY_STAGE_PASS package=%s forward_sha256=%s stage_a_sha256=%s stage_b_sha256=%s manifest=%s\n' \
    "${PACKAGE_ROOT}" "${FORWARD_SHA}" "${STAGE_A_SHA}" "${STAGE_B_SHA}" "${INPUT_MANIFEST}"

if [[ "${MODE}" == "stage" ]]; then
    cat "${INPUT_MANIFEST}" > "${PACKAGE_ROOT}/${FINAL_MANIFEST_NAME}"
    printf 'assembly_state=staged_only\n' >> "${PACKAGE_ROOT}/${FINAL_MANIFEST_NAME}"
    printf 'P10_ASSEMBLY_STAGE_ONLY_PASS package=%s manifest=%s\n' \
        "${PACKAGE_ROOT}" "${PACKAGE_ROOT}/${FINAL_MANIFEST_NAME}"
    exit 0
fi

[[ -n "${CANN_3RD_LIB_PATH}" ]] || fail "cann_third_party_cache_path_missing"
[[ -x "${PRODUCT_PYTHON}" ]] || fail "python_not_executable:path=${PRODUCT_PYTHON:-<unset>}"
printf 'P10_ASSEMBLY_BUILD_START product=combined source=%s output=%s\n' \
    "${PACKAGE_ROOT}/source" "${PACKAGE_ROOT}/product/opp"
P10_PRODUCT_VLLM_SOURCE_ROOT="${VLLM_ROOT}" \
P10_PRODUCT_KVAF_SOURCE_ROOT="${KVAF_ROOT}" \
P10_PRODUCT_EXPECTED_VLLM_SHA="${VLLM_HEAD}" \
P10_PRODUCT_EXPECTED_KVAF_SHA="${KVAF_HEAD}" \
P10_PRODUCT_STAGE_A_SOURCE_ROOT="${STAGED_A}" \
P10_PRODUCT_STAGE_B_SOURCE_ROOT="${STAGED_B}" \
P10_PRODUCT_FORWARD_SOURCE_ROOT="${STAGED_FORWARD}" \
P10_PRODUCT_STAGE_A_SHA256="${STAGE_A_SHA}" \
P10_PRODUCT_STAGE_B_SHA256="${STAGE_B_SHA}" \
P10_PRODUCT_FORWARD_SHA256="${FORWARD_SHA}" \
P10_PRODUCT_OUTPUT_ROOT="${PACKAGE_ROOT}/product/opp" \
P10_PRODUCT_PROVENANCE_FILE="${PACKAGE_ROOT}/product/P10_PRODUCT_PROVENANCE" \
P10_PRODUCT_PACKAGE_PARENT="${PACKAGE_ROOT}/product" \
P10_PRODUCT_PACKAGE_ROOT="${PACKAGE_ROOT}/product/build" \
P10_PRODUCT_CANN_3RD_LIB_PATH="${CANN_3RD_LIB_PATH}" \
P10_PRODUCT_BUILD_JOBS="${BUILD_JOBS}" \
P10_PRODUCT_PYTHON="${PRODUCT_PYTHON}" \
bash "${PRODUCT_RECIPE}"

FINAL_MANIFEST="${PACKAGE_ROOT}/${FINAL_MANIFEST_NAME}"
cat "${INPUT_MANIFEST}" > "${FINAL_MANIFEST}"
printf 'assembly_state=complete\n' >> "${FINAL_MANIFEST}"
product_opapi="$(find "${PACKAGE_ROOT}/product/opp" -type f -path '*/op_api/lib/libcust_opapi.so' -print -quit)"
[[ -s "${product_opapi}" ]] || fail "combined_product_opapi_missing"
printf 'product_opapi_sha256=%s\n' "$(hash_file "${product_opapi}")" >> "${FINAL_MANIFEST}"
printf 'product_operators=A5P10ForwardPrepare,A5P10W4A4C8ScoreReceipt,A5P10ScoreReceiptForwardResident\n' >> "${FINAL_MANIFEST}"

printf 'P10_ASSEMBLY_PASS package=%s manifest=%s\n' "${PACKAGE_ROOT}" "${FINAL_MANIFEST}"
