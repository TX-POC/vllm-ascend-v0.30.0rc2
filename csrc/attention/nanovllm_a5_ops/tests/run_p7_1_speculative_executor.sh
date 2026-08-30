#!/usr/bin/env bash
# Build and run the standalone P7.1 Direct-HCCP executor on one physical A5.
#
# Execute this file with `bash .../run_p7_1_speculative_executor.sh`; do not
# source it.  It builds only A5ScoreReceiptSpeculativeExecutor and its own
# bridge, never C8/P7.0/P7.2a nor the full vLLM-Ascend extension.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf 'P7_1_FINAL status=2 reason=runner_must_be_executed_with_bash_not_sourced\n' >&2
    return 2
fi

set +e
set +u
set -o pipefail

fail() {
    local reason="$1"
    printf 'P7_1_FINAL status=1 reason=%s\n' "${reason}"
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
OPS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
VLLM_ROOT_DEFAULT="$(cd "${OPS_ROOT}/../../.." && pwd -P)"
P7_VLLM_ROOT="${P7_VLLM_ROOT:-${VLLM_ROOT_DEFAULT}}"
P7_KVAF_SOURCE="${P7_KVAF_SOURCE:-${KVAF_FUSED_GET_SOURCE:-}}"
P7_EXPECTED_VLLM_SHA="${P7_EXPECTED_VLLM_SHA:-}"
P7_EXPECTED_KVAF_SHA="${P7_EXPECTED_KVAF_SHA:-}"
P7_KVAF_BUILD="${P7_KVAF_BUILD:-}"
P7_PHYSICAL_NPU_ID="${P7_PHYSICAL_NPU_ID:-7}"
P7_PYTHON="${P7_PYTHON:-$(command -v python3 2>/dev/null)}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.1.0}}"
P7_BUILD_JOBS="${P7_BUILD_JOBS:-16}"
P7_STAMP="$(date +%Y%m%d_%H%M%S)"
P7_RESULT="${P7_RESULT:-$(dirname "${P7_VLLM_ROOT}")/results/p7_1_executor_npu${P7_PHYSICAL_NPU_ID}_${P7_STAMP}}"
P7_BUILD_ROOT="${P7_BUILD_ROOT:-${OPS_ROOT}/build/p7_1_executor_${P7_STAMP}}"
P7_OPP_ROOT="${P7_OPP_ROOT:-${OPS_ROOT}/_custom_opp_p7_1_executor_${P7_STAMP}}"
P7_BRIDGE_ROOT="${OPS_ROOT}/a5_score_receipt_speculative_executor/torch_extension"
P7_TEST="${OPS_ROOT}/tests/test_score_receipt_speculative_executor.py"
P7_CONTRACT_TEST="${OPS_ROOT}/tests/test_score_receipt_speculative_executor_contract.py"
P7_BUILD_SCRIPT="${OPS_ROOT}/build_nanovllm_ops.sh"

# A P7.1 run is provenance-sensitive.  Never reuse an arbitrary caller
# directory: an old manifest/opapi can otherwise survive a source update and
# produce exactly the stale-artifact mismatch this runner is meant to avoid.
for P7_FRESH_PATH in "${P7_RESULT}" "${P7_BUILD_ROOT}" "${P7_OPP_ROOT}"; do
    if [[ -e "${P7_FRESH_PATH}" ]]; then
        printf 'P7_1_SOURCE ready=0 reason=fresh_path_already_exists path=%s\n' \
            "${P7_FRESH_PATH}"
        fail "fresh_path_already_exists"
    fi
done
mkdir -p "${P7_RESULT}"

if [[ -z "${P7_EXPECTED_VLLM_SHA}" || -z "${P7_EXPECTED_KVAF_SHA}" || \
      -z "${P7_KVAF_SOURCE}" || -z "${P7_KVAF_BUILD}" || -z "${P7_PYTHON}" ]]; then
    printf 'P7_1_SOURCE ready=0 reason=missing_required_env vllm_expected=%s kvaf_expected=%s kvaf_source=%s kvaf_build=%s python=%s\n' \
        "${P7_EXPECTED_VLLM_SHA:-missing}" "${P7_EXPECTED_KVAF_SHA:-missing}" \
        "${P7_KVAF_SOURCE:-missing}" "${P7_KVAF_BUILD:-missing}" "${P7_PYTHON:-missing}"
    fail "missing_required_env"
fi

if ! git -C "${P7_VLLM_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
   ! git -C "${P7_KVAF_SOURCE}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'P7_1_SOURCE ready=0 reason=invalid_git_worktree vllm=%s kvaf=%s\n' \
        "${P7_VLLM_ROOT}" "${P7_KVAF_SOURCE}"
    fail "invalid_git_worktree"
fi

P7_VLLM_DIRTY="$(git -C "${P7_VLLM_ROOT}" status --porcelain --untracked-files=no)"
P7_KVAF_DIRTY="$(git -C "${P7_KVAF_SOURCE}" status --porcelain --untracked-files=no)"
P7_VLLM_HEAD="$(git -C "${P7_VLLM_ROOT}" rev-parse HEAD 2>/dev/null)"
P7_KVAF_HEAD="$(git -C "${P7_KVAF_SOURCE}" rev-parse HEAD 2>/dev/null)"
P7_SOURCE_READY=1
if [[ -n "${P7_VLLM_DIRTY}" || -n "${P7_KVAF_DIRTY}" || \
      "${P7_VLLM_HEAD}" != "${P7_EXPECTED_VLLM_SHA}" || \
      "${P7_KVAF_HEAD}" != "${P7_EXPECTED_KVAF_SHA}" ]]; then
    P7_SOURCE_READY=0
fi
printf 'P7_1_SOURCE vllm_head=%s vllm_expected=%s vllm_dirty=%s kvaf_head=%s kvaf_expected=%s kvaf_dirty=%s ready=%s\n' \
    "${P7_VLLM_HEAD}" "${P7_EXPECTED_VLLM_SHA}" \
    "$([[ -n "${P7_VLLM_DIRTY}" ]] && echo 1 || echo 0)" \
    "${P7_KVAF_HEAD}" "${P7_EXPECTED_KVAF_SHA}" \
    "$([[ -n "${P7_KVAF_DIRTY}" ]] && echo 1 || echo 0)" "${P7_SOURCE_READY}"
if [[ "${P7_SOURCE_READY}" != "1" ]]; then
    fail "source_not_exact_or_clean"
fi

if [[ ! -x "${P7_PYTHON}" || ! -f "${P7_BUILD_SCRIPT}" || \
      ! -f "${P7_TEST}" || ! -f "${P7_CONTRACT_TEST}" || \
      ! -f "${P7_BRIDGE_ROOT}/score_receipt_speculative_executor_setup.py" ]]; then
    fail "missing_p7_1_source_or_python"
fi
if [[ ! -f "${ASCEND_HOME_PATH}/bin/setenv.bash" || \
      ! -f "${ASCEND_HOME_PATH}/aarch64-linux/asc/include/simt_api/device_atomic_functions.h" ]]; then
    printf 'P7_1_SOURCE ready=0 reason=missing_cann asc=%s\n' "${ASCEND_HOME_PATH}"
    fail "missing_cann"
fi

P7_KVAF_SOURCE_ABS="$(cd "${P7_KVAF_SOURCE}" && pwd -P)"
if [[ ! -f "${P7_KVAF_BUILD}/CMakeCache.txt" ]]; then
    printf 'P7_1_KVAF_BUILD ready=0 reason=missing_cmake_cache build=%s\n' "${P7_KVAF_BUILD}"
    fail "missing_kvaf_configure"
fi
P7_CACHE_SOURCE="$(sed -n 's|^CMAKE_HOME_DIRECTORY:INTERNAL=||p' "${P7_KVAF_BUILD}/CMakeCache.txt" | sed -n '1p')"
if [[ -z "${P7_CACHE_SOURCE}" || ! -d "${P7_CACHE_SOURCE}" || \
      "$(cd "${P7_CACHE_SOURCE}" && pwd -P)" != "${P7_KVAF_SOURCE_ABS}" ]]; then
    printf 'P7_1_KVAF_BUILD ready=0 reason=cmake_source_mismatch build=%s cache_source=%s expected=%s\n' \
        "${P7_KVAF_BUILD}" "${P7_CACHE_SOURCE:-missing}" "${P7_KVAF_SOURCE_ABS}"
    fail "kvaf_build_source_mismatch"
fi

# Product selection is deliberately explicit.  Never inherit the historic
# SOC_VERSION variable; build_nanovllm_ops.sh rejects it too.
unset SOC_VERSION
export A5_SOC_VERSION=ascend950
export ASCEND_HOME_PATH
export ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
export ASCEND_SIMT_API_DIR="${ASCEND_HOME_PATH}/aarch64-linux/asc/include"
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
unset SOC_VERSION
export A5_SOC_VERSION=ascend950

# The target is physical NPU 7 by default.  Every test uses its logical npu:0
# after this process-level mapping; never call it on implicit NPU 0.  CANN's
# runtime variable is authoritative here; deliberately remove the legacy
# alternate variable so a stale inherited mapping cannot win.
unset ASCEND_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES="${P7_PHYSICAL_NPU_ID}"
export NPU_ID=0
export DEVICE_ID=0
printf 'P7_1_NPU physical=%s logical=0 runtime_visible=%s legacy_visible=unset\n' \
    "${P7_PHYSICAL_NPU_ID}" "${ASCEND_RT_VISIBLE_DEVICES}"

"${P7_PYTHON}" -c 'import torch, torch_npu, pybind11; torch.npu.set_device(0); print("P7_1_PYTHON exe=%s torch=%s torch_npu=%s device=%s" % (__import__("sys").executable, torch.__version__, torch_npu.__version__, torch.npu.get_device_name(0)))' \
    2>&1 | tee "${P7_RESULT}/python_npu.log"
P7_PYTHON_RC=${PIPESTATUS[0]}
if [[ "${P7_PYTHON_RC}" != "0" ]]; then
    fail "npu7_python_context_failed"
fi

"${P7_PYTHON}" "${P7_CONTRACT_TEST}" --ops-root "${OPS_ROOT}" \
    2>&1 | tee "${P7_RESULT}/contract.log"
P7_CONTRACT_RC=${PIPESTATUS[0]}
if [[ "${P7_CONTRACT_RC}" != "0" ]]; then
    fail "p7_1_static_contract_failed"
fi

# Rebuild the selected KVAF checkout before staging it into P7.1.  This is an
# incremental KVAF build only; it never invokes the full vLLM-Ascend build.
cmake --build "${P7_KVAF_BUILD}" -j "${P7_BUILD_JOBS}" \
    2>&1 | tee "${P7_RESULT}/kvaf_incremental_build.log"
P7_KVAF_BUILD_RC=${PIPESTATUS[0]}
P7_KVAF_GET="${P7_KVAF_BUILD}/bin/kvaf_execute_batch_get_kernels.o"
P7_KVAF_PUT="${P7_KVAF_BUILD}/bin/kvaf_execute_batch_put_kernels.o"
P7_KVAF_PYEXT="$(find "${P7_KVAF_BUILD}/python/kvaf" -maxdepth 1 -type f -name '_kvaf*.so' -print -quit 2>/dev/null)"
if [[ "${P7_KVAF_BUILD_RC}" != "0" || ! -s "${P7_KVAF_GET}" || ! -s "${P7_KVAF_PUT}" || \
      ! -d "${P7_KVAF_BUILD}/python" || -z "${P7_KVAF_PYEXT}" ]]; then
    printf 'P7_1_KVAF_BUILD status=%s get=%s put=%s python=%s pyext=%s\n' \
        "${P7_KVAF_BUILD_RC}" "$([[ -s "${P7_KVAF_GET}" ]] && echo 1 || echo 0)" \
        "$([[ -s "${P7_KVAF_PUT}" ]] && echo 1 || echo 0)" \
        "$([[ -d "${P7_KVAF_BUILD}/python" ]] && echo 1 || echo 0)" \
        "${P7_KVAF_PYEXT:-missing}"
    fail "kvaf_incremental_build_failed"
fi
printf 'P7_1_KVAF_BUILD status=0 source_head=%s get=1 put=1\n' "${P7_KVAF_HEAD}"

# Fresh timestamped OPP/build roots prevent an old CANN package from being
# selected after a source update.  The P7.1 build script itself verifies the
# generated closure, exactly one OP_ADD, KVAF helper symbols, and provenance.
NANOVLLM_A5_OPS_ONLY=A5ScoreReceiptSpeculativeExecutor \
NANOVLLM_A5_SKIP_TORCH_EXTENSION=1 \
NANOVLLM_A5_OPS_PYTHON="${P7_PYTHON}" \
NANOVLLM_A5_OPS_BUILD_JOBS="${P7_BUILD_JOBS}" \
NANOVLLM_A5_OPS_BUILD_ROOT="${P7_BUILD_ROOT}" \
NANOVLLM_A5_LOCAL_OPP="${P7_OPP_ROOT}" \
KVAF_FUSED_GET_SOURCE="${P7_KVAF_SOURCE}" \
ASCEND_SIMT_API_DIR="${ASCEND_SIMT_API_DIR}" \
A5_SOC_VERSION=ascend950 \
bash "${P7_BUILD_SCRIPT}" 2>&1 | tee "${P7_RESULT}/p7_1_opp_build.log"
P7_OPP_RC=${PIPESTATUS[0]}
P7_MANIFEST_COUNT="$(find "${P7_OPP_ROOT}/vendors" -type f -name P7_1_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_PROVENANCE -print 2>/dev/null | wc -l | tr -d ' ')"
P7_MANIFEST="$(find "${P7_OPP_ROOT}/vendors" -type f -name P7_1_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_PROVENANCE -print -quit 2>/dev/null)"
if [[ "${P7_OPP_RC}" != "0" || "${P7_MANIFEST_COUNT}" != "1" || -z "${P7_MANIFEST}" || \
      ! -f "${P7_OPP_ROOT}/vendors/customize/bin/set_env.bash" ]]; then
    printf 'P7_1_OPP status=%s root=%s manifest_count=%s manifest=%s\n' \
        "${P7_OPP_RC}" "${P7_OPP_ROOT}" "${P7_MANIFEST_COUNT}" "${P7_MANIFEST:-missing}"
    fail "p7_1_fresh_opp_build_failed"
fi
if ! grep -Fqx "source_head=${P7_VLLM_HEAD}" "${P7_MANIFEST}" || \
   ! grep -Fqx "kvaf_source_head=${P7_KVAF_HEAD}" "${P7_MANIFEST}" || \
   ! grep -Fqx 'abi=31in_18aliasout' "${P7_MANIFEST}" || \
   ! grep -Fqx 'post_cq=real_kvaf_streaming_postonly_tryretire_finish' "${P7_MANIFEST}" || \
   ! grep -Fqx 'c8_source_closure=absent' "${P7_MANIFEST}" || \
   ! grep -Fqx 'fixed_get_core=absent' "${P7_MANIFEST}"; then
    printf 'P7_1_MANIFEST ready=0 manifest=%s\n' "${P7_MANIFEST}"
    fail "p7_1_manifest_mismatch"
fi
printf 'P7_1_OPP status=0 root=%s\n' "${P7_OPP_ROOT}"
printf 'P7_1_MANIFEST source=%s kvaf=%s abi=31in_18aliasout c8_closure=absent\n' \
    "${P7_VLLM_HEAD}" "${P7_KVAF_HEAD}"

# Source only the new package.  The bridge then replaces, rather than appends,
# custom OPP/opapi environment variables with the matching fresh vendor.
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_CUST_OPAPI_LIB
source "${P7_OPP_ROOT}/vendors/customize/bin/set_env.bash"
unset SOC_VERSION
export A5_SOC_VERSION=ascend950
unset ASCEND_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES="${P7_PHYSICAL_NPU_ID}"
export NANOVLLM_A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_OPP="${P7_OPP_ROOT}"
export P7_EXPECTED_VLLM_SHA="${P7_VLLM_HEAD}"
export P7_EXPECTED_KVAF_SHA="${P7_KVAF_HEAD}"
export P7_KVAF_BATCH_GET_KERNEL="${P7_KVAF_GET}"
export P7_KVAF_BATCH_PUT_KERNEL="${P7_KVAF_PUT}"
export PYTHONPATH="${P7_BRIDGE_ROOT}:${P7_KVAF_BUILD}/python${PYTHONPATH:+:${PYTHONPATH}}"

(
    cd "${P7_BRIDGE_ROOT}" || exit 1
    USE_NINJA=0 \
    NANOVLLM_A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_BUILD_JOBS="${P7_BUILD_JOBS}" \
    "${P7_PYTHON}" score_receipt_speculative_executor_setup.py build_ext --inplace --force
) 2>&1 | tee "${P7_RESULT}/bridge_build.log"
P7_BRIDGE_RC=${PIPESTATUS[0]}
P7_BRIDGE_SO="$(find "${P7_BRIDGE_ROOT}/nanovllm_score_receipt_speculative_executor" -maxdepth 1 -type f -name '_C*.so' -print -quit 2>/dev/null)"
if [[ "${P7_BRIDGE_RC}" != "0" || -z "${P7_BRIDGE_SO}" ]]; then
    printf 'P7_1_BRIDGE status=%s artifact=%s\n' \
        "${P7_BRIDGE_RC}" "${P7_BRIDGE_SO:-missing}"
    fail "p7_1_bridge_build_failed"
fi
printf 'P7_1_BRIDGE status=0 artifact=%s aliases=18\n' "${P7_BRIDGE_SO}"

# First prove invalid KVAF control fails before acquiring a queue.  It is a
# guard only, never reported as real HCCP success.
"${P7_PYTHON}" "${P7_TEST}" --device npu:0 \
    2>&1 | tee "${P7_RESULT}/fail_closed.log"
P7_FAIL_CLOSED_RC=${PIPESTATUS[0]}
if [[ "${P7_FAIL_CLOSED_RC}" != "0" ]]; then
    fail "p7_1_fail_closed_guard_failed"
fi

# Default final test is a real local KVAF transaction: cold bind -> dynamic
# device lease -> descriptor publish/PostOnly -> CQ terminal Finish -> release.
# It must not report PASS if the concrete fixture cannot initialize.
"${P7_PYTHON}" "${P7_TEST}" --device npu:0 \
    --fixture-module p7_1_real_direct_hccp_fixture \
    2>&1 | tee "${P7_RESULT}/real_direct_hccp.log"
P7_REAL_RC=${PIPESTATUS[0]}
if [[ "${P7_REAL_RC}" != "0" ]]; then
    fail "p7_1_real_direct_hccp_failed"
fi

grep -hE '^P7_1_(SOURCE|NPU|KVAF_BUILD|OPP|MANIFEST|BRIDGE|CONTROL|RECEIPTS|QUEUE|TRANSPORT|FAILURE|REAL_BIND|REAL_LIFECYCLE|TEST|FINAL)' \
    "${P7_RESULT}"/*.log 2>/dev/null | tail -n 80
printf 'P7_1_FINAL status=0 physical_npu=%s logical_npu=0 result=%s real_direct_hccp=1\n' \
    "${P7_PHYSICAL_NPU_ID}" "${P7_RESULT}"
