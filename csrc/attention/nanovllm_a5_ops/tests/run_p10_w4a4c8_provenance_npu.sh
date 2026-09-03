#!/usr/bin/env bash
# Verify the real A5 W4A4C8 runtime binding before any Stage-A OPP build.
# Invoke this file with `bash`; it intentionally has no execute-bit requirement.

set -o pipefail

this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ops_root="$(cd "${this_dir}/.." && pwd -P)"
target_root="${ops_root}/a5_dsa_c8_exact_speculative_resident"
probe="${target_root}/tools/probe_w4a4c8_runtime.py"
contract_test="${this_dir}/test_p10_w4a4c8_source_contract.py"

blocked() {
  printf 'P10_W4A4C8_RUNNER_BLOCKED reason=%s\n' "$1"
  exit 2
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || blocked "environment_missing:${name}"
}

for required in P10_PRODUCT_VLLM_ROOT P10_PRODUCT_VLLM_EXPECTED_SHA P10_PHYSICAL_NPU_ID; do
  require_env "${required}"
done

P10_PYTHON="${P10_PYTHON:-$(command -v python3 2>/dev/null || true)}"
[[ -x "${P10_PYTHON}" ]] || blocked "python_not_executable:${P10_PYTHON:-<unset>}"
[[ -f "${probe}" ]] || blocked "w4a4c8_probe_missing"
[[ -f "${contract_test}" ]] || blocked "w4a4c8_contract_test_missing"
[[ -e "${P10_PRODUCT_VLLM_ROOT}/.git" ]] ||
  blocked "git_worktree_missing:${P10_PRODUCT_VLLM_ROOT}"
[[ -z "${SOC_VERSION:-}" ]] || blocked "legacy_SOC_VERSION_must_be_unset"

P10_LOGICAL_NPU_ID="${P10_LOGICAL_NPU_ID:-0}"
[[ "${P10_LOGICAL_NPU_ID}" == "0" ]] ||
  blocked "logical_npu_must_be_zero:${P10_LOGICAL_NPU_ID}"
[[ "${ASCEND_RT_VISIBLE_DEVICES:-}" == "${P10_PHYSICAL_NPU_ID}" ]] ||
  blocked "physical_logical_mapping_invalid:visible=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"

dirty="$(git -C "${P10_PRODUCT_VLLM_ROOT}" status --porcelain --untracked-files=no 2>/dev/null || true)"
[[ -z "${dirty}" ]] || blocked "tracked_changes_present:product_vllm"
head="$(git -C "${P10_PRODUCT_VLLM_ROOT}" rev-parse HEAD 2>/dev/null || true)"
[[ "${head}" == "${P10_PRODUCT_VLLM_EXPECTED_SHA}" ]] ||
  blocked "product_source_sha_mismatch:actual=${head}:expected=${P10_PRODUCT_VLLM_EXPECTED_SHA}"

printf 'P10_W4A4C8_RUNNER_CONFIG head=%s physical=%s logical=0 visible=%s stage_a_opp=buildable_not_installed product_route=runtime_fail_closed_until_parity\n' \
  "${head}" "${P10_PHYSICAL_NPU_ID}" "${ASCEND_RT_VISIBLE_DEVICES}"

"${P10_PYTHON}" "${contract_test}"
contract_status=$?
printf 'P10_W4A4C8_RUNNER_STEP label=stage_a_contract status=%s\n' "${contract_status}"
[[ "${contract_status}" -eq 0 ]] ||
  blocked "stage_a_canonical_topk_contract_failed"

"${P10_PYTHON}" "${probe}" --product-vllm-root "${P10_PRODUCT_VLLM_ROOT}"
probe_status=$?
printf 'P10_W4A4C8_RUNNER_STEP label=runtime_provenance status=%s\n' "${probe_status}"
exit "${probe_status}"
