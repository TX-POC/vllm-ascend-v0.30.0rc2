#!/usr/bin/env bash
# Stage a fresh, source-only VllmQuant C8 sibling for P10 Stage-A derivation.
#
# This script does not patch an existing C8 tree and does not build an OPP.
# It deliberately stops after provenance-locked staging, because the formal
# seam must be reviewed with its own OpDef, tiling budget, and graph ABI.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
STAGE_A_ROOT="$(cd "${THIS_DIR}/.." && pwd -P)"
TARGET_ROOT="$(cd "${STAGE_A_ROOT}/.." && pwd -P)"
PROBE="${TARGET_ROOT}/tools/probe_product_c8_provenance.py"

SOURCE_ROOT="${P10_PRODUCT_C8_SOURCE_ROOT:-}"
EXPECTED_SOURCE_HEAD="${P10_PRODUCT_C8_EXPECTED_SOURCE_HEAD:-}"
PACKAGE_PARENT="${P10_STAGE_A_PACKAGE_PARENT:-${STAGE_A_ROOT}/build-stage-a}"
PACKAGE_ROOT="${P10_STAGE_A_PACKAGE_ROOT:-}"
PYTHON_BIN="${P10_STAGE_A_PYTHON:-python3}"

fail() {
  printf 'P10_STAGE_A_DERIVATION_FAIL reason=%s\n' "$1" >&2
  exit 2
}

if [[ "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  P10_PRODUCT_C8_SOURCE_ROOT=/absolute/exact/vllm_quant_lightning_indexer \
  P10_PRODUCT_C8_EXPECTED_SOURCE_HEAD=<git-sha> \
  stage_product_c8_sibling.sh

The command creates a fresh source-only package after a fail-closed provenance
check.  It does not modify the source root and does not build an operator.
USAGE
  exit 0
fi
[[ -z "${1:-}" ]] || fail "invalid_argument:value=$1"
[[ -n "${SOURCE_ROOT}" ]] || fail "source_root_missing"
[[ -d "${SOURCE_ROOT}" ]] || fail "source_root_not_directory:path=${SOURCE_ROOT}"
[[ -x "$(command -v "${PYTHON_BIN}" 2>/dev/null || true)" ]] ||
  fail "python_missing:value=${PYTHON_BIN}"
[[ -f "${PROBE}" ]] || fail "provenance_probe_missing:path=${PROBE}"

SOURCE_ROOT="$(cd "${SOURCE_ROOT}" && pwd -P)"
PROVENANCE="$(${PYTHON_BIN} "${PROBE}" --source-root "${SOURCE_ROOT}")" || {
  printf '%s\n' "${PROVENANCE:-P10_PRODUCT_C8_PROVENANCE status=fail reason=probe_failed}" >&2
  fail "provenance_probe_failed"
}
printf '%s\n' "${PROVENANCE}"
[[ "${PROVENANCE}" == *"status=pass"* ]] || fail "provenance_not_pass"
[[ "${PROVENANCE}" == *"family=vllm_quant_lightning_indexer"* ]] ||
  fail "unsupported_source_family"
[[ "${PROVENANCE}" == *"supported=1"* ]] || fail "source_family_not_enabled"

SOURCE_HEAD="$(git -C "${SOURCE_ROOT}" rev-parse HEAD 2>/dev/null || true)"
[[ -n "${SOURCE_HEAD}" ]] || fail "source_git_head_unavailable:path=${SOURCE_ROOT}"
[[ -n "${EXPECTED_SOURCE_HEAD}" ]] || fail "expected_source_head_missing"
[[ "${SOURCE_HEAD}" == "${EXPECTED_SOURCE_HEAD}" ]] ||
  fail "source_head_mismatch:actual=${SOURCE_HEAD}:expected=${EXPECTED_SOURCE_HEAD}"
SOURCE_DIRTY="$(git -C "${SOURCE_ROOT}" status --porcelain --untracked-files=no 2>/dev/null || true)"
[[ -z "${SOURCE_DIRTY}" ]] || fail "source_tracked_changes_present:path=${SOURCE_ROOT}"

[[ -d "${PACKAGE_PARENT}" || ! -e "${PACKAGE_PARENT}" ]] ||
  fail "package_parent_not_directory:path=${PACKAGE_PARENT}"
mkdir -p "${PACKAGE_PARENT}"
PACKAGE_PARENT="$(cd "${PACKAGE_PARENT}" && pwd -P)"
if [[ -z "${PACKAGE_ROOT}" ]]; then
  PACKAGE_ROOT="${PACKAGE_PARENT}/vllm_quant_stage_a_${SOURCE_HEAD:0:12}_$(date +%Y%m%d_%H%M%S)"
fi
case "${PACKAGE_ROOT}" in
  "${PACKAGE_PARENT}"/*) ;;
  *) fail "unsafe_package_root:path=${PACKAGE_ROOT}:parent=${PACKAGE_PARENT}" ;;
esac
[[ ! -e "${PACKAGE_ROOT}" ]] || fail "fresh_package_root_exists:path=${PACKAGE_ROOT}"

for required in \
  CMakeLists.txt \
  op_host/vllm_quant_lightning_indexer_def.cpp \
  op_host/vllm_quant_lightning_indexer_tiling.cpp \
  op_kernel/vllm_quant_lightning_indexer.cpp \
  op_kernel/arch35/quant_lightning_indexer_kernel.h \
  op_kernel/arch35/quant_lightning_indexer_service_vector.h; do
  [[ -f "${SOURCE_ROOT}/${required}" ]] ||
    fail "source_closure_missing:path=${SOURCE_ROOT}/${required}"
done

mkdir "${PACKAGE_ROOT}"
STAGED_C8_ROOT="${PACKAGE_ROOT}/source/vllm_quant_lightning_indexer"
mkdir -p "${STAGED_C8_ROOT}"

# Do not carry previous CMake/OPP products into the sibling.  `tar` preserves
# source permissions while its exclusion list makes the staged tree auditable.
tar -C "${SOURCE_ROOT}" \
  --exclude='./build' \
  --exclude='./build-*' \
  --exclude='./_custom_opp*' \
  --exclude='./CMakeCache.txt' \
  -cf - . | tar -C "${STAGED_C8_ROOT}" -xf -

for header in \
  a5_c8_receipt_ring.h \
  a5_c8_score_receipt_seam.h; do
  install -m 0644 "${STAGE_A_ROOT}/op_kernel/arch35/${header}" \
    "${STAGED_C8_ROOT}/op_kernel/arch35/${header}"
done

if find "${STAGED_C8_ROOT}" -type f \( -name CMakeCache.txt -o -name '*.o' -o -name '*.so' \) -print -quit | grep -q .; then
  fail "staged_tree_contains_build_product"
fi

HEADER_SHA="$(sha256sum "${STAGE_A_ROOT}/op_kernel/arch35/a5_c8_receipt_ring.h" | awk '{print $1}')"
SEAM_SHA="$(sha256sum "${STAGE_A_ROOT}/op_kernel/arch35/a5_c8_score_receipt_seam.h" | awk '{print $1}')"
cat > "${PACKAGE_ROOT}/P10_STAGE_A_DERIVATION_RECEIPT" <<EOF
format=1
stage=exact_score_receipt
source_root=${SOURCE_ROOT}
source_head=${SOURCE_HEAD}
source_dirty=0
family=vllm_quant_lightning_indexer
operator=VllmQuantLightningIndexer
staged_c8_root=${STAGED_C8_ROOT}
original_source_mutated=0
source_only_clone=1
automatic_text_transform=0
formal_seam_anchor=BatchMulWeightAndReduceSum
original_score_copy_preserved=required
original_sparse_output_preserved=required
receipt_ring_header_sha256=${HEADER_SHA}
score_seam_header_sha256=${SEAM_SHA}
derivation_state=MANUAL_SEAM_DERIVATION_REQUIRED
opp_build_started=0
EOF

printf 'P10_STAGE_A_DERIVATION_READY source_head=%s package=%s staged_c8=%s state=manual_seam_derivation_required\n' \
  "${SOURCE_HEAD}" "${PACKAGE_ROOT}" "${STAGED_C8_ROOT}"
