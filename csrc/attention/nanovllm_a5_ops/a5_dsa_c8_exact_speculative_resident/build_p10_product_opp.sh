#!/usr/bin/env bash
# Build P10 as one fresh custom-OP vendor package.
#
# The P10 graph bridge resolves Prepare, Stage A, and Stage B from one OpAPI
# DSO.  Building three independently installed vendors and hoping dlopen
# ordering finds the matching trio is not a safe product route: an old vendor
# can satisfy one symbol while another comes from the new build.  This script
# instead consumes three *source-only* staged directories, generates one
# multi-op CANN project, and installs exactly one libcust_opapi.so.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VLLM_ROOT="${P10_PRODUCT_VLLM_SOURCE_ROOT:-}"
KVAF_ROOT="${P10_PRODUCT_KVAF_SOURCE_ROOT:-}"
EXPECTED_VLLM_SHA="${P10_PRODUCT_EXPECTED_VLLM_SHA:-}"
EXPECTED_KVAF_SHA="${P10_PRODUCT_EXPECTED_KVAF_SHA:-}"
STAGE_A_ROOT="${P10_PRODUCT_STAGE_A_SOURCE_ROOT:-}"
STAGE_B_ROOT="${P10_PRODUCT_STAGE_B_SOURCE_ROOT:-}"
FORWARD_ROOT="${P10_PRODUCT_FORWARD_SOURCE_ROOT:-}"
EXPECTED_STAGE_A_SHA="${P10_PRODUCT_STAGE_A_SHA256:-}"
EXPECTED_STAGE_B_SHA="${P10_PRODUCT_STAGE_B_SHA256:-}"
EXPECTED_FORWARD_SHA="${P10_PRODUCT_FORWARD_SHA256:-}"
OUTPUT_ROOT="${P10_PRODUCT_OUTPUT_ROOT:-}"
PROVENANCE_FILE="${P10_PRODUCT_PROVENANCE_FILE:-}"
PACKAGE_PARENT="${P10_PRODUCT_PACKAGE_PARENT:-}"
PACKAGE_ROOT="${P10_PRODUCT_PACKAGE_ROOT:-}"
CANN_3RD_LIB_PATH="${P10_PRODUCT_CANN_3RD_LIB_PATH:-}"
BUILD_JOBS="${P10_PRODUCT_BUILD_JOBS:-4}"
TARGET_SOC_RAW="${P10_PRODUCT_SOC:-ascend950}"
PYTHON_BIN="${P10_PRODUCT_PYTHON:-$(command -v python3 2>/dev/null || true)}"

fail() {
    printf 'P10_PRODUCT_BUILD_FAIL reason=%s\n' "$1" >&2
    exit 2
}

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

# Keep this in lockstep with build_p10_target_assembly.sh.  Generated files
# must never affect the source identity selected by the assembly manifest.
tree_digest() {
    local source="$1"
    local list file rel file_hash
    list="$(mktemp "${TMPDIR:-/tmp}/p10-product-tree.XXXXXX")"
    while IFS= read -r -d '' file; do
        rel="${file#${source}/}"
        case "${rel}" in
            .git/*|build/*|build-*/*|*/build/*|*/build-*/*|op_dev/*|custom_op/*|opp/*|_custom_opp*/*|*/_custom_opp*/*|packages/*|*/packages/*|__pycache__/*|*/__pycache__/*|*.pyc|*.o|*.so|*.run|CMakeCache.txt)
                continue
                ;;
        esac
        printf '%s\0' "${rel}" >> "${list}"
    done < <(find "${source}" -type f -print0)
    [[ -s "${list}" ]] || {
        rm -f "${list}"
        fail "source_tree_empty:path=${source}"
    }
    while IFS= read -r -d '' rel; do
        file="${source}/${rel}"
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
    local source="$2"
    local expected="$3"
    local head dirty
    [[ -e "${source}/.git" ]] || fail "${label}_git_root_missing:path=${source}"
    [[ -n "${expected}" ]] || fail "${label}_expected_sha_missing"
    head="$(git -C "${source}" rev-parse HEAD 2>/dev/null || true)"
    [[ "${head}" == "${expected}" ]] ||
        fail "${label}_head_mismatch:actual=${head}:expected=${expected}"
    dirty="$(git -C "${source}" status --porcelain --untracked-files=no 2>/dev/null || true)"
    [[ -z "${dirty}" ]] || fail "${label}_tracked_changes_present:path=${source}"
}

require_stage() {
    local label="$1"
    local source="$2"
    local expected_digest="$3"
    local actual_digest
    [[ -d "${source}/op_host" && -d "${source}/op_kernel" ]] ||
        fail "${label}_source_layout_missing:path=${source}"
    [[ -s "${source}/ops.json" ]] || fail "${label}_ops_json_missing"
    [[ -n "${expected_digest}" ]] || fail "${label}_source_digest_missing"
    actual_digest="$(tree_digest "${source}")"
    [[ "${actual_digest}" == "${expected_digest}" ]] ||
        fail "${label}_source_digest_mismatch:actual=${actual_digest}:expected=${expected_digest}"
}

case "${TARGET_SOC_RAW,,}" in
    ascend950|ascend950pr*|ascend950dt*) TARGET_SOC=ascend950 ;;
    *) fail "unsupported_soc:value=${TARGET_SOC_RAW}" ;;
esac
[[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || fail "invalid_jobs:value=${BUILD_JOBS}"
[[ -z "${SOC_VERSION:-}" ]] || fail "legacy_SOC_VERSION_set"
[[ -x "${PYTHON_BIN}" ]] || fail "python_not_executable:path=${PYTHON_BIN:-<unset>}"
[[ -n "${OUTPUT_ROOT}" ]] || fail "output_root_missing"
[[ -n "${PROVENANCE_FILE}" ]] || fail "provenance_file_missing"
[[ -n "${PACKAGE_PARENT}" ]] || fail "package_parent_missing"
[[ -n "${PACKAGE_ROOT}" ]] || fail "package_root_missing"
[[ -n "${CANN_3RD_LIB_PATH}" ]] || fail "cann_third_party_cache_path_missing"
command -v msopgen >/dev/null 2>&1 || fail "msopgen_missing"

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

require_clean_exact_git_root vllm "${VLLM_ROOT}" "${EXPECTED_VLLM_SHA}"
require_clean_exact_git_root kvaf "${KVAF_ROOT}" "${EXPECTED_KVAF_SHA}"
require_stage stage_a "${STAGE_A_ROOT}" "${EXPECTED_STAGE_A_SHA}"
require_stage stage_b "${STAGE_B_ROOT}" "${EXPECTED_STAGE_B_SHA}"
require_stage forward_prepare "${FORWARD_ROOT}" "${EXPECTED_FORWARD_SHA}"

VLLM_HEAD="$(git -C "${VLLM_ROOT}" rev-parse HEAD)"
KVAF_HEAD="$(git -C "${KVAF_ROOT}" rev-parse HEAD)"
KVAF_DEMO_ROOT="${KVAF_ROOT}/access_operator/demo/aiv_batch_get"
for file in \
    "${KVAF_DEMO_ROOT}/kernel/kvaf_forward_resident_get_device.h" \
    "${KVAF_DEMO_ROOT}/kernel/execute_batch_get_aiv.asc" \
    "${KVAF_DEMO_ROOT}/include/batch_get_abi.h" \
    "${KVAF_DEMO_ROOT}/include/direct_hccp_abi.h"; do
    [[ -f "${file}" ]] || fail "kvaf_device_closure_missing:path=${file}"
done

PACKAGE_PARENT="$(mkdir -p "${PACKAGE_PARENT}" && cd "${PACKAGE_PARENT}" && pwd -P)"
case "${PACKAGE_ROOT}" in
    "${PACKAGE_PARENT}"/*) ;;
    *) fail "unsafe_package_root:path=${PACKAGE_ROOT}:parent=${PACKAGE_PARENT}" ;;
esac
OUTPUT_PARENT="$(dirname "${OUTPUT_ROOT}")"
OUTPUT_PARENT="$(mkdir -p "${OUTPUT_PARENT}" && cd "${OUTPUT_PARENT}" && pwd -P)"
OUTPUT_ROOT="${OUTPUT_PARENT}/$(basename "${OUTPUT_ROOT}")"
[[ ! -e "${PACKAGE_ROOT}" ]] || fail "fresh_package_root_exists:path=${PACKAGE_ROOT}"
[[ ! -e "${OUTPUT_ROOT}" ]] || fail "fresh_output_root_exists:path=${OUTPUT_ROOT}"

OP_DEV="${PACKAGE_ROOT}/source/op_dev"
GENERATED="${PACKAGE_ROOT}/custom_op"
mkdir -p "${OP_DEV}/op_host" "${OP_DEV}/op_kernel/kvaf_forward_resident"

# Stage B's checked-in ABI deliberately imports Stage A's receipt protocol by
# a sibling-relative include.  In a standalone Stage-B package that sibling is
# staged at ``${PACKAGE_ROOT}/stage_a_w4a4c8_score_receipt``.  Preserve the
# same source layout in the combined package as well.  Merely copying the
# header into ``GENERATED/op_kernel`` is insufficient: the preprocessor first
# resolves the checked-in relative path from the Stage-B ABI's own directory.
# Keeping this explicit compatibility sibling avoids an ambient source-tree
# include and turns a would-be remote CANN compile failure into a deterministic
# source-closed package.
STAGE_A_COMPAT_KERNEL_DIR="${PACKAGE_ROOT}/stage_a_w4a4c8_score_receipt/op_kernel"
mkdir -p "${STAGE_A_COMPAT_KERNEL_DIR}"
cp -p "${STAGE_A_ROOT}/op_kernel/a5_p10_w4a4c8_score_receipt_abi.h" \
    "${STAGE_A_COMPAT_KERNEL_DIR}/a5_p10_w4a4c8_score_receipt_abi.h"
[[ -s "${STAGE_A_COMPAT_KERNEL_DIR}/a5_p10_w4a4c8_score_receipt_abi.h" ]] ||
    fail "stage_a_compat_abi_stage_missing"

# msopgen's documented schema is an array of operator definitions.  Preserve
# each source's one-object array and concatenate them atomically rather than
# maintaining a hand-copied third ABI declaration.
"${PYTHON_BIN}" - "${STAGE_A_ROOT}/ops.json" "${STAGE_B_ROOT}/ops.json" \
    "${FORWARD_ROOT}/ops.json" "${OP_DEV}/ops.json" <<'PY'
import json
import pathlib
import sys

input_paths = [pathlib.Path(item) for item in sys.argv[1:4]]
output_path = pathlib.Path(sys.argv[4])
operators = []
for input_path in input_paths:
    payload = json.loads(input_path.read_text())
    if not isinstance(payload, list) or len(payload) != 1:
        raise SystemExit(f"expected one operator declaration: {input_path}")
    operators.extend(payload)
names = [entry.get("op") for entry in operators]
expected = [
    "A5P10W4A4C8ScoreReceipt",
    "A5P10ScoreReceiptForwardResident",
    "A5P10ForwardPrepare",
]
if names != expected or len(set(names)) != len(names):
    raise SystemExit(f"unexpected combined operator set: {names!r}")
output_path.write_text(json.dumps(operators, indent=2) + "\n")
PY

# The CMakeLists.txt files in the checked-in stages are monorepo fragments.
# A fresh ``msopgen`` project owns its standalone CMake scaffold; copying a
# stage fragment over it either drops the other two operators or calls macros
# unavailable in that scaffold.  Copy only source closure files and retain the
# generated CMake project.  Non-CMake filename collisions remain fatal: they
# would make the combined source identity ambiguous.
copy_unique_tree_without_repository_cmake() {
    local source="$1"
    local destination="$2"
    local file rel target
    while IFS= read -r -d '' file; do
        rel="${file#${source}/}"
        case "${rel}" in
            CMakeLists.txt|*/CMakeLists.txt)
                continue
                ;;
        esac
        target="${destination}/${rel}"
        [[ ! -e "${target}" ]] || fail "combined_source_filename_collision:path=${rel}"
        mkdir -p "$(dirname "${target}")"
        cp -p "${file}" "${target}"
    done < <(find "${source}" -type f -print0)
}

copy_unique_tree_without_repository_cmake "${STAGE_A_ROOT}/op_host" "${OP_DEV}/op_host"
copy_unique_tree_without_repository_cmake "${STAGE_B_ROOT}/op_host" "${OP_DEV}/op_host"
copy_unique_tree_without_repository_cmake "${FORWARD_ROOT}/op_host" "${OP_DEV}/op_host"
copy_unique_tree_without_repository_cmake "${STAGE_A_ROOT}/op_kernel" "${OP_DEV}/op_kernel"
copy_unique_tree_without_repository_cmake "${STAGE_B_ROOT}/op_kernel" "${OP_DEV}/op_kernel"
copy_unique_tree_without_repository_cmake "${FORWARD_ROOT}/op_kernel" "${OP_DEV}/op_kernel"
cp -p "${KVAF_DEMO_ROOT}/kernel/kvaf_forward_resident_get_device.h" \
    "${OP_DEV}/op_kernel/kvaf_forward_resident/kvaf_forward_resident_get_device.h"
cp -p "${KVAF_DEMO_ROOT}/kernel/execute_batch_get_aiv.asc" \
    "${OP_DEV}/op_kernel/kvaf_forward_resident/execute_batch_get_aiv.asc"
cp -p "${KVAF_DEMO_ROOT}/include/batch_get_abi.h" \
    "${OP_DEV}/op_kernel/kvaf_forward_resident/batch_get_abi.h"
cp -p "${KVAF_DEMO_ROOT}/include/direct_hccp_abi.h" \
    "${OP_DEV}/op_kernel/kvaf_forward_resident/direct_hccp_abi.h"

printf 'P10_PRODUCT_BUILD_SOURCE vllm=%s kvaf=%s package=%s output=%s target=%s jobs=%s\n' \
    "${VLLM_HEAD}" "${KVAF_HEAD}" "${PACKAGE_ROOT}" "${OUTPUT_ROOT}" "${TARGET_SOC}" "${BUILD_JOBS}"

# ``ops.json`` intentionally contains all three graph nodes.  Omitting
# ``-op`` makes msopgen enter an interactive selector whenever it sees that
# array, which is invalid for a non-interactive remote build and can spin on
# EOF under a piped log.  Generate the first node as a fresh project, then
# append each remaining node to that same project.  CANN documents ``-m 1``
# as append-to-existing-project mode; together with ``-op`` this preserves
# one CMake scaffold and one vendor while making the selection deterministic.
declare -a OP_NAMES=(
    A5P10W4A4C8ScoreReceipt
    A5P10ScoreReceiptForwardResident
    A5P10ForwardPrepare
)
for index in "${!OP_NAMES[@]}"; do
    op="${OP_NAMES[${index}]}"
    msopgen_args=(
        gen
        -i "${OP_DEV}/ops.json"
        -f aclnn
        -c "ai_core-${TARGET_SOC}"
        -lan cpp
        -op "${op}"
        -out "${GENERATED}"
    )
    merge=0
    if (( index > 0 )); then
        msopgen_args+=(-m 1)
        merge=1
    fi
    printf 'P10_PRODUCT_MSOPGEN op=%s merge=%s interactive=0\n' \
        "${op}" "${merge}"
    msopgen "${msopgen_args[@]}"
done

generated_host_cmake="${GENERATED}/op_host/CMakeLists.txt"
[[ -f "${generated_host_cmake}" ]] || fail "generated_host_cmake_missing"
if grep -Fq 'add_op_to_compiled_list' "${generated_host_cmake}"; then
    fail "repository_host_cmake_leaked_into_msopgen_project"
fi
cp -a "${OP_DEV}/op_host/." "${GENERATED}/op_host/"
cp -a "${OP_DEV}/op_kernel/." "${GENERATED}/op_kernel/"

# Source filenames are not necessarily msopgen filenames.  In particular,
# CANN normalizes the W4/A4/C8 runs in Stage A's CamelCase operator name into
# ``a5_p10_w4_a4_c8_score_receipt.cpp``.  The checked-in implementation keeps
# the human-facing, ABI-oriented spelling without separators.  Do not leave
# both files in the merged tree: CMake would compile the generated placeholder
# and/or compile the real kernel twice.  Stage B has the analogous deliberately
# suffixed source filename.  Replace each generated canonical wrapper with its
# source implementation, then remove the source-only spelling.
stage_a_generated_kernel="${GENERATED}/op_kernel/a5_p10_w4_a4_c8_score_receipt.cpp"
stage_a_source_kernel="${GENERATED}/op_kernel/a5_p10_w4a4c8_score_receipt.cpp"
[[ -f "${stage_a_generated_kernel}" ]] ||
    fail "generated_stage_a_kernel_stub_missing"
[[ -f "${stage_a_source_kernel}" ]] ||
    fail "stage_a_source_kernel_missing_after_copy"
cp -p "${stage_a_source_kernel}" "${stage_a_generated_kernel}"
rm -f "${stage_a_source_kernel}"

stage_b_generated_kernel="${GENERATED}/op_kernel/a5_p10_score_receipt_forward_resident.cpp"
stage_b_source_kernel="${GENERATED}/op_kernel/a5_p10_score_receipt_forward_resident_kernel.cpp"
[[ -f "${stage_b_generated_kernel}" ]] ||
    fail "generated_stage_b_kernel_stub_missing"
[[ -f "${stage_b_source_kernel}" ]] ||
    fail "stage_b_source_kernel_missing_after_copy"
cp -p "${stage_b_source_kernel}" "${stage_b_generated_kernel}"
rm -f "${stage_b_source_kernel}"

declare -a STUBS=(
    a5_p10_w4_a4_c8_score_receipt.cpp
    a5_p10_score_receipt_forward_resident.cpp
    a5_p10_forward_prepare.cpp
)
declare -a DEFS=(
    a5_p10_w4a4c8_score_receipt_def.cpp
    a5_p10_score_receipt_forward_resident_def.cpp
    a5_p10_forward_prepare_def.cpp
)

for index in 0 1 2; do
    stub="${GENERATED}/op_host/${STUBS[${index}]}"
    definition="${GENERATED}/op_host/${DEFS[${index}]}"
    op="${OP_NAMES[${index}]}"
    [[ -f "${stub}" ]] || fail "generated_host_stub_missing:operator=${op}"
    [[ -f "${definition}" ]] || fail "source_opdef_missing:operator=${op}"
    regex="OP_ADD[[:space:]]*\\([[:space:]]*${op}[[:space:]]*\\)"
    [[ "$(grep -Ec "${regex}" "${stub}" || true)" -eq 1 ]] ||
        fail "generated_host_stub_registration_unexpected:operator=${op}"
    [[ "$(grep -Ec "${regex}" "${definition}" || true)" -eq 1 ]] ||
        fail "source_opdef_registration_unexpected:operator=${op}"
    rm -f "${stub}"
done
for op in "${OP_NAMES[@]}"; do
    regex="OP_ADD[[:space:]]*\\([[:space:]]*${op}[[:space:]]*\\)"
    count="$(grep -REc "${regex}" "${GENERATED}/op_host" --include='*.cpp' --include='*.cc' --include='*.cxx' 2>/dev/null | awk -F: '{sum += $NF} END {print sum + 0}')"
    [[ "${count}" -eq 1 ]] || fail "combined_registration_invalid:operator=${op}:count=${count}"
done
total_registrations="$(grep -REh 'OP_ADD[[:space:]]*\\(' "${GENERATED}/op_host" --include='*.cpp' --include='*.cc' --include='*.cxx' 2>/dev/null | wc -l | tr -d ' ')"
[[ "${total_registrations}" -eq 3 ]] ||
    fail "combined_registration_total_invalid:count=${total_registrations}"

if ! (
    cd "${GENERATED}"
    CANN_3RD_LIB_PATH="${CANN_3RD_LIB_PATH}" OPS_CPU_NUMBER="${BUILD_JOBS}" bash build.sh
); then
    ini="${GENERATED}/build_out/autogen/aic-${TARGET_SOC}-ops-info.ini"
    printf 'P10_PRODUCT_BUILD_DIAG generated_ini=%s\n' "${ini}" >&2
    [[ -f "${ini}" ]] && grep -E '^\[(A5P10W4A4C8ScoreReceipt|A5P10ScoreReceiptForwardResident|A5P10ForwardPrepare)\]$' "${ini}" >&2 || true
    exit 1
fi

run_package="$(find "${GENERATED}/build_out" -maxdepth 1 -type f -name '*.run' -print -quit)"
[[ -n "${run_package}" ]] || fail "run_package_missing"
chmod +x "${run_package}"
"${run_package}" --quiet --install-path="${OUTPUT_ROOT}"

opapi_count="$(find "${OUTPUT_ROOT}" -type f -path '*/op_api/lib/libcust_opapi.so' -print | wc -l | tr -d ' ')"
[[ "${opapi_count}" -eq 1 ]] || fail "combined_opapi_ambiguous:count=${opapi_count}"
opapi="$(find "${OUTPUT_ROOT}" -type f -path '*/op_api/lib/libcust_opapi.so' -print -quit)"
[[ -s "${opapi}" ]] || fail "combined_opapi_missing"
vendor="$(cd "$(dirname "$(dirname "$(dirname "${opapi}")")")" && pwd -P)"
mkdir -p "$(dirname "${PROVENANCE_FILE}")"
cat > "${PROVENANCE_FILE}" <<EOF
format=1
kind=p10_combined_prepare_stage_a_stage_b_product_opp
vllm_source_root=${VLLM_ROOT}
vllm_source_head=${VLLM_HEAD}
kvaf_source_root=${KVAF_ROOT}
kvaf_source_head=${KVAF_HEAD}
stage_a_source_sha256=${EXPECTED_STAGE_A_SHA}
stage_b_source_sha256=${EXPECTED_STAGE_B_SHA}
forward_prepare_source_sha256=${EXPECTED_FORWARD_SHA}
operators=A5P10ForwardPrepare,A5P10W4A4C8ScoreReceipt,A5P10ScoreReceiptForwardResident
target_soc=${TARGET_SOC}
vendor=${vendor}
opapi=${opapi}
opapi_sha256=$(hash_file "${opapi}")
EOF
printf 'P10_PRODUCT_BUILD_PASS opp=%s vendor=%s opapi=%s provenance=%s\n' \
    "${OUTPUT_ROOT}" "${vendor}" "${opapi}" "${PROVENANCE_FILE}"
