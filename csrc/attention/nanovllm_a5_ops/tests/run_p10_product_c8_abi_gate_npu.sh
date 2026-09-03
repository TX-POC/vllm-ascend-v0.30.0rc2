#!/usr/bin/env bash
# Compare exactly one selected product C8 frontend with the separately built
# source-matched official C8 reference.  This runner does not stage or build
# an operator; all reference artifacts must already carry exact provenance.

set -o pipefail

this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ops_root="$(cd "${this_dir}/.." && pwd -P)"
target_root="${ops_root}/a5_dsa_c8_exact_speculative_resident"
source_probe="${target_root}/tools/probe_product_bf16_abi.py"
runtime_fixture="${target_root}/tools/run_product_bf16_reference_equivalence.py"

blocked() {
  printf 'P10_ABI_GATE_BLOCKED reason=%s\n' "$1"
  exit 2
}

require_env() {
  local name="$1"
  local value="${!name:-}"
  [[ -n "${value}" ]] || blocked "environment_missing:${name}"
}

for required in \
  P10_PRODUCT_C8_FRONTEND \
  P10_PRODUCT_VLLM_ROOT \
  P10_PRODUCT_VLLM_EXPECTED_SHA \
  P10_REFERENCE_VLLM_ROOT \
  P10_REFERENCE_SOURCE_SHA \
  P10_REFERENCE_OPAPI \
  P10_REFERENCE_BRIDGE_ROOT \
  P10_REFERENCE_BRIDGE_MODULE \
  P10_PHYSICAL_NPU_ID; do
  require_env "${required}"
done

P10_PYTHON="${P10_PYTHON:-$(command -v python3 2>/dev/null || true)}"
[[ -x "${P10_PYTHON}" ]] || blocked "python_not_executable:${P10_PYTHON:-<unset>}"
[[ -f "${source_probe}" && -f "${runtime_fixture}" ]] ||
  blocked "p10_abi_gate_files_missing"
[[ -z "${SOC_VERSION:-}" ]] || blocked "legacy_SOC_VERSION_must_be_unset"

P10_LOGICAL_NPU_ID="${P10_LOGICAL_NPU_ID:-0}"
[[ "${P10_LOGICAL_NPU_ID}" == "0" ]] ||
  blocked "logical_npu_must_be_zero:actual=${P10_LOGICAL_NPU_ID}"
[[ "${ASCEND_RT_VISIBLE_DEVICES:-}" == "${P10_PHYSICAL_NPU_ID}" ]] ||
  blocked "physical_logical_mapping_invalid:physical=${P10_PHYSICAL_NPU_ID}:visible=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"

for root_name in P10_PRODUCT_VLLM_ROOT P10_REFERENCE_VLLM_ROOT; do
  root="${!root_name}"
  [[ -e "${root}/.git" ]] || blocked "git_worktree_missing:${root_name}:${root}"
  dirty="$(git -C "${root}" status --porcelain --untracked-files=no 2>/dev/null || true)"
  [[ -z "${dirty}" ]] || blocked "tracked_changes_present:${root_name}"
done

product_head="$(git -C "${P10_PRODUCT_VLLM_ROOT}" rev-parse HEAD 2>/dev/null || true)"
reference_head="$(git -C "${P10_REFERENCE_VLLM_ROOT}" rev-parse HEAD 2>/dev/null || true)"
[[ "${product_head}" == "${P10_PRODUCT_VLLM_EXPECTED_SHA}" ]] ||
  blocked "product_source_sha_mismatch:actual=${product_head}:expected=${P10_PRODUCT_VLLM_EXPECTED_SHA}"
[[ "${reference_head}" == "${P10_REFERENCE_SOURCE_SHA}" ]] ||
  blocked "reference_source_sha_mismatch:actual=${reference_head}:expected=${P10_REFERENCE_SOURCE_SHA}"

printf 'P10_ABI_GATE_CONFIG frontend=%s product_head=%s reference_head=%s physical=%s logical=0 visible=%s\n' \
  "${P10_PRODUCT_C8_FRONTEND}" "${product_head}" "${reference_head}" \
  "${P10_PHYSICAL_NPU_ID}" "${ASCEND_RT_VISIBLE_DEVICES}"

"${P10_PYTHON}" "${source_probe}" \
  --frontend "${P10_PRODUCT_C8_FRONTEND}" \
  --product-vllm-root "${P10_PRODUCT_VLLM_ROOT}" \
  --reference-vllm-root "${P10_REFERENCE_VLLM_ROOT}" \
  --reference-opapi "${P10_REFERENCE_OPAPI}" \
  --reference-bridge-root "${P10_REFERENCE_BRIDGE_ROOT}" \
  --reference-bridge-module "${P10_REFERENCE_BRIDGE_MODULE}" \
  --reference-source-sha "${P10_REFERENCE_SOURCE_SHA}"
preflight_status=$?
printf 'P10_ABI_GATE_STEP label=source_and_artifact_provenance status=%s\n' \
  "${preflight_status}"
[[ "${preflight_status}" -eq 0 ]] || exit "${preflight_status}"

export P10_LOGICAL_NPU_ID
"${P10_PYTHON}" "${runtime_fixture}"
runtime_status=$?
printf 'P10_ABI_GATE_STEP label=product_vs_reference status=%s\n' \
  "${runtime_status}"
if [[ "${runtime_status}" -eq 0 ]]; then
  printf 'P10_ABI_GATE_PASS frontend=%s physical=%s logical=0\n' \
    "${P10_PRODUCT_C8_FRONTEND}" "${P10_PHYSICAL_NPU_ID}"
fi
exit "${runtime_status}"
