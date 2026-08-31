#!/usr/bin/env bash
# Re-run P7.1 Python tests without recompiling an OPP only when the compiled
# OPP closure is unchanged. The small torch bridge is deliberately rebuilt:
# it owns ACLNN descriptor/workspace lifetime and is not part of the OPP
# kernel/op-host artifact. This is intentionally for bridge/test retries,
# not for kernel/host/KVAF development.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf 'P7_REUSE_FINAL status=2 reason=runner_must_be_executed_with_bash_not_sourced\n' >&2
    return 2
fi

set +e
set +u
set -o pipefail

fail() {
    printf 'P7_REUSE_FINAL status=1 reason=%s result=%s\n' "$1" "${P7_RESULT:-missing}"
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
OPS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
VLLM_ROOT_DEFAULT="$(cd "${OPS_ROOT}/../../.." && pwd -P)"
P7_VLLM_ROOT="${P7_VLLM_ROOT:-${VLLM_ROOT_DEFAULT}}"
P7_KVAF_SOURCE="${P7_KVAF_SOURCE:-${KVAF_FUSED_GET_SOURCE:-}}"
P7_REUSE_OPP_ROOT="${P7_REUSE_OPP_ROOT:-}"
P7_KVAF_BUILD="${P7_KVAF_BUILD:-}"
P7_REQUIRED_RUNTIME_SHA="${P7_REQUIRED_RUNTIME_SHA:-}"
P7_REQUIRED_KVAF_SHA="${P7_REQUIRED_KVAF_SHA:-}"
P7_PHYSICAL_NPU_ID="${P7_PHYSICAL_NPU_ID:-7}"
P7_PYTHON="${P7_PYTHON:-$(command -v python3 2>/dev/null)}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.1.0}}"
P7_BUILD_JOBS="${P7_BUILD_JOBS:-16}"
P7_STAMP="$(date +%Y%m%d_%H%M%S)"
P7_RESULT="${P7_RESULT:-$(dirname "${P7_VLLM_ROOT}")/results/p7_1_test_reuse_npu${P7_PHYSICAL_NPU_ID}_${P7_STAMP}}"
P7_TEST="${OPS_ROOT}/tests/test_score_receipt_speculative_executor.py"
P7_CONTRACT_TEST="${OPS_ROOT}/tests/test_score_receipt_speculative_executor_contract.py"
P7_BRIDGE_ROOT="${OPS_ROOT}/a5_score_receipt_speculative_executor/torch_extension"
P7_OPS_REL="${OPS_ROOT#"${P7_VLLM_ROOT}"/}"

# The product gate is one fresh-process direct-current-stream fail-closed
# launch, followed by the real Direct-HCCP fixture. The old cleanup-bisect
# variables are explicitly removed so a stale shell cannot re-enable a
# torch_npu task-queue path that P7.1 no longer uses.
unset NANOVLLM_A5_OPAPI_CLEANUP_BISECT NANOVLLM_A5_OPAPI_CLEANUP_MODE

if [[ -e "${P7_RESULT}" ]]; then
    fail "result_path_already_exists"
fi
mkdir -p "${P7_RESULT}"

if [[ -z "${P7_KVAF_SOURCE}" || -z "${P7_REUSE_OPP_ROOT}" || \
      ! -x "${P7_PYTHON}" || ! -f "${P7_TEST}" || ! -f "${P7_CONTRACT_TEST}" || \
      ! -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
    fail "missing_required_env_or_source"
fi
if ! git -C "${P7_VLLM_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
   ! git -C "${P7_KVAF_SOURCE}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    fail "invalid_git_worktree"
fi

P7_MANIFEST_COUNT="$(find "${P7_REUSE_OPP_ROOT}/vendors" -type f -name P7_1_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_PROVENANCE -print 2>/dev/null | wc -l | tr -d ' ')"
P7_MANIFEST="$(find "${P7_REUSE_OPP_ROOT}/vendors" -type f -name P7_1_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_PROVENANCE -print -quit 2>/dev/null)"
P7_ARTIFACT_VLLM_SHA="$(sed -n 's/^source_head=//p' "${P7_MANIFEST}" 2>/dev/null | sed -n '1p')"
P7_ARTIFACT_KVAF_SHA="$(sed -n 's/^kvaf_source_head=//p' "${P7_MANIFEST}" 2>/dev/null | sed -n '1p')"
P7_VLLM_HEAD="$(git -C "${P7_VLLM_ROOT}" rev-parse HEAD 2>/dev/null || true)"
P7_KVAF_HEAD="$(git -C "${P7_KVAF_SOURCE}" rev-parse HEAD 2>/dev/null || true)"
P7_VLLM_DIRTY="$(git -C "${P7_VLLM_ROOT}" status --porcelain --untracked-files=no)"
P7_KVAF_DIRTY="$(git -C "${P7_KVAF_SOURCE}" status --porcelain --untracked-files=no)"

# An artifact may legitimately come from an older ancestor when only the
# bridge changes, but the *checkout that builds that bridge* must be exactly
# the revision requested by the caller. This prevents a fetch race from being
# silently diagnosed as an OPP/bridge mismatch.
P7_RUNTIME_SHA_MATCH=1
P7_KVAF_SHA_MATCH=1
if [[ -n "${P7_REQUIRED_RUNTIME_SHA}" && \
      "${P7_VLLM_HEAD}" != "${P7_REQUIRED_RUNTIME_SHA}" ]]; then
    P7_RUNTIME_SHA_MATCH=0
fi
if [[ -n "${P7_REQUIRED_KVAF_SHA}" && \
      "${P7_KVAF_HEAD}" != "${P7_REQUIRED_KVAF_SHA}" ]]; then
    P7_KVAF_SHA_MATCH=0
fi

P7_MANIFEST_VALID=0
P7_OPAPI="$(dirname "${P7_MANIFEST}")/op_api/lib/libcust_opapi.so"
if [[ "${P7_MANIFEST_COUNT}" == "1" && -f "${P7_MANIFEST}" && -f "${P7_OPAPI}" ]]; then
    P7_MANIFEST_VALID=1
    for required_line in \
        'kind=standalone_score_receipt_speculative_executor' \
        'abi=31tensor_18refout' \
        'torch_schema=31in_18aliasout' \
        'local_microbatches=2x64' \
        'queue_lifecycle=epoch,generation,post_sequence,state,quarantine' \
        'post_cq=real_kvaf_streaming_postonly_tryretire_finish' \
        'c8_source_closure=absent' \
        'fixed_get_core=absent' \
        'operators=A5ScoreReceiptSpeculativeExecutor'; do
        if ! grep -Fqx "${required_line}" "${P7_MANIFEST}"; then
            P7_MANIFEST_VALID=0
        fi
    done
fi

P7_ARTIFACT_COMMIT_VALID=0
if [[ "${P7_ARTIFACT_VLLM_SHA}" =~ ^[0-9a-f]{40}$ ]] && \
   git -C "${P7_VLLM_ROOT}" cat-file -e "${P7_ARTIFACT_VLLM_SHA}^{commit}" 2>/dev/null; then
    P7_ARTIFACT_COMMIT_VALID=1
fi
P7_ARTIFACT_ANCESTOR=0
if [[ "${P7_ARTIFACT_COMMIT_VALID}" == "1" ]] && \
   git -C "${P7_VLLM_ROOT}" merge-base --is-ancestor "${P7_ARTIFACT_VLLM_SHA}" "${P7_VLLM_HEAD}"; then
    P7_ARTIFACT_ANCESTOR=1
fi

# This reuse gate intentionally names the *compiled* OPP closure instead of
# the whole P7 directory.  The bridge and its Python fixture are host-only and
# rebuilt below; treating them as OPP sources caused needless rebuilds and, in
# practice, stale-artifact/checkout confusion.  Any change to schema, OpDef,
# device kernel, tiling header, or staging script still blocks reuse.
P7_CLOSURE_UNCHANGED=0
if git -C "${P7_VLLM_ROOT}" diff --quiet "${P7_ARTIFACT_VLLM_SHA}" "${P7_VLLM_HEAD}" -- \
    "${P7_OPS_REL}/a5_score_receipt_speculative_executor/ops.json" \
    "${P7_OPS_REL}/a5_score_receipt_speculative_executor/op_host" \
    "${P7_OPS_REL}/a5_score_receipt_speculative_executor/op_kernel" \
    "${P7_OPS_REL}/build_nanovllm_ops.sh"; then
    P7_CLOSURE_UNCHANGED=1
fi

P7_CHANGED_PATHS="$(git -C "${P7_VLLM_ROOT}" diff --name-only "${P7_ARTIFACT_VLLM_SHA}" "${P7_VLLM_HEAD}" 2>/dev/null || true)"
P7_UNEXPECTED_PATHS=""
while IFS= read -r changed_path; do
    [[ -z "${changed_path}" ]] && continue
    case "${changed_path}" in
        "${P7_OPS_REL}/tests/run_p7_1_speculative_executor.sh" | \
        "${P7_OPS_REL}/tests/run_p7_1_test_only_artifact_reuse.sh" | \
        "${P7_OPS_REL}/tests/run_p7_1_fail_closed_dispatch_diag.sh" | \
        "${P7_OPS_REL}/tests/test_score_receipt_speculative_executor.py" | \
        "${P7_OPS_REL}/tests/test_score_receipt_speculative_executor_contract.py" | \
        "${P7_OPS_REL}/torch_extension/csrc/op_api_common.h" | \
        "${P7_OPS_REL}/a5_score_receipt_speculative_executor/torch_extension/"* | \
        "${P7_OPS_REL}/a5_score_receipt_speculative_executor/P7_1_ABI.md") ;;
        *) P7_UNEXPECTED_PATHS="${P7_UNEXPECTED_PATHS}${changed_path}," ;;
    esac
done <<< "${P7_CHANGED_PATHS}"

P7_KVAF_SOURCE_REAL="$(cd "${P7_KVAF_SOURCE}" && pwd -P)"
if [[ -z "${P7_KVAF_BUILD}" ]]; then
    for candidate in "${P7_KVAF_SOURCE}"/build-*; do
        cache_source="$(sed -n 's|^CMAKE_HOME_DIRECTORY:INTERNAL=||p' "${candidate}/CMakeCache.txt" 2>/dev/null | sed -n '1p')"
        if [[ -d "${cache_source}" && \
              "$(cd "${cache_source}" && pwd -P)" == "${P7_KVAF_SOURCE_REAL}" && \
              -s "${candidate}/bin/kvaf_execute_batch_get_kernels.o" && \
              -s "${candidate}/bin/kvaf_execute_batch_put_kernels.o" && \
              -n "$(find "${candidate}/python/kvaf" -maxdepth 1 -type f -name '_kvaf*.so' -print -quit 2>/dev/null)" ]]; then
            P7_KVAF_BUILD="${candidate}"
            break
        fi
    done
fi
P7_KVAF_BUILD_VALID=0
P7_KVAF_BUILD_SOURCE="$(sed -n 's|^CMAKE_HOME_DIRECTORY:INTERNAL=||p' "${P7_KVAF_BUILD}/CMakeCache.txt" 2>/dev/null | sed -n '1p')"
if [[ -d "${P7_KVAF_BUILD_SOURCE}" && \
      "$(cd "${P7_KVAF_BUILD_SOURCE}" && pwd -P)" == "${P7_KVAF_SOURCE_REAL}" && \
      -s "${P7_KVAF_BUILD}/bin/kvaf_execute_batch_get_kernels.o" && \
      -s "${P7_KVAF_BUILD}/bin/kvaf_execute_batch_put_kernels.o" && \
      -n "$(find "${P7_KVAF_BUILD}/python/kvaf" -maxdepth 1 -type f -name '_kvaf*.so' -print -quit 2>/dev/null)" ]]; then
    P7_KVAF_BUILD_VALID=1
fi
P7_KVAF_BUILD_STAMP="${P7_KVAF_BUILD}/.p7_kvaf_source_head"
P7_KVAF_BUILD_HEAD="$(sed -n '1p' "${P7_KVAF_BUILD_STAMP}" 2>/dev/null | tr -d '[:space:]')"
P7_KVAF_BUILD_HEAD_MATCH=0
if [[ "${P7_KVAF_BUILD_HEAD}" == "${P7_KVAF_HEAD}" ]]; then
    P7_KVAF_BUILD_HEAD_MATCH=1
fi

P7_REUSE_READY=0
if [[ "${P7_MANIFEST_VALID}" == "1" && "${P7_ARTIFACT_COMMIT_VALID}" == "1" && \
      "${P7_ARTIFACT_ANCESTOR}" == "1" && \
      "${P7_ARTIFACT_KVAF_SHA}" == "${P7_KVAF_HEAD}" && \
      -z "${P7_VLLM_DIRTY}" && -z "${P7_KVAF_DIRTY}" && \
      "${P7_RUNTIME_SHA_MATCH}" == "1" && \
      "${P7_KVAF_SHA_MATCH}" == "1" && \
      "${P7_CLOSURE_UNCHANGED}" == "1" && -z "${P7_UNEXPECTED_PATHS}" && \
      "${P7_KVAF_BUILD_VALID}" == "1" && \
      "${P7_KVAF_BUILD_HEAD_MATCH}" == "1" && \
      -f "${P7_BRIDGE_ROOT}/score_receipt_speculative_executor_setup.py" && \
      -f "${P7_REUSE_OPP_ROOT}/vendors/customize/bin/set_env.bash" ]]; then
    P7_REUSE_READY=1
fi

printf 'P7_REUSE_GATE ready=%s runtime_vllm=%s required_runtime=%s runtime_match=%s artifact_vllm=%s artifact_ancestor=%s runtime_kvaf=%s required_kvaf=%s kvaf_match=%s artifact_kvaf=%s manifest_valid=%s closure_unchanged=%s kvaf_build_valid=%s kvaf_build_head=%s kvaf_build_head_match=%s unexpected_paths=<%s>\n' \
    "${P7_REUSE_READY}" "${P7_VLLM_HEAD}" "${P7_REQUIRED_RUNTIME_SHA:-none}" \
    "${P7_RUNTIME_SHA_MATCH}" \
    "${P7_ARTIFACT_VLLM_SHA:-missing}" "${P7_ARTIFACT_ANCESTOR}" \
    "${P7_KVAF_HEAD}" "${P7_REQUIRED_KVAF_SHA:-none}" "${P7_KVAF_SHA_MATCH}" \
    "${P7_ARTIFACT_KVAF_SHA:-missing}" \
    "${P7_MANIFEST_VALID}" "${P7_CLOSURE_UNCHANGED}" "${P7_KVAF_BUILD_VALID}" \
    "${P7_KVAF_BUILD_HEAD:-missing}" "${P7_KVAF_BUILD_HEAD_MATCH}" \
    "${P7_UNEXPECTED_PATHS:-none}"
if [[ "${P7_REUSE_READY}" != "1" ]]; then
    fail "closure_or_provenance_gate_failed"
fi

printf 'P7_REUSE_BRIDGE_PROVENANCE bridge_source=%s opp_source=%s bridge_only_rebuild=1 opp_closure_unchanged=%s\n' \
    "${P7_VLLM_HEAD}" "${P7_ARTIFACT_VLLM_SHA}" "${P7_CLOSURE_UNCHANGED}"

# The artifact source SHA is intentionally exported here: the package checks
# the binary that is actually loaded. The gate above proves the compiled OPP
# closure is unchanged; the bridge itself is rebuilt from P7_VLLM_HEAD.
unset SOC_VERSION ASCEND_VISIBLE_DEVICES
export A5_SOC_VERSION=ascend950
export ASCEND_HOME_PATH
export ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
unset SOC_VERSION ASCEND_VISIBLE_DEVICES
export A5_SOC_VERSION=ascend950
export ASCEND_RT_VISIBLE_DEVICES="${P7_PHYSICAL_NPU_ID}"
export NPU_ID=0 DEVICE_ID=0
unset ASCEND_CUSTOM_OPP_PATH NANOVLLM_CUST_OPAPI_LIB
source "${P7_REUSE_OPP_ROOT}/vendors/customize/bin/set_env.bash"
unset ASCEND_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES="${P7_PHYSICAL_NPU_ID}"
export NPU_ID=0 DEVICE_ID=0
unset P7_EXPECTED_VLLM_SHA P7_EXPECTED_KVAF_SHA
export NANOVLLM_A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_OPP="${P7_REUSE_OPP_ROOT}"
export P7_EXPECTED_VLLM_SHA="${P7_ARTIFACT_VLLM_SHA}"
export P7_EXPECTED_KVAF_SHA="${P7_ARTIFACT_KVAF_SHA}"
export P7_KVAF_BATCH_GET_KERNEL="${P7_KVAF_BUILD}/bin/kvaf_execute_batch_get_kernels.o"
export P7_KVAF_BATCH_PUT_KERNEL="${P7_KVAF_BUILD}/bin/kvaf_execute_batch_put_kernels.o"
export PYTHONPATH="${P7_BRIDGE_ROOT}:${P7_KVAF_BUILD}/python${PYTHONPATH:+:${PYTHONPATH}}"
# This is intentionally on for the scoped P7.1 retry. The marker proves that
# the bridge directly enqueued on the caller's current NPU stream.
export NANOVLLM_A5_OPAPI_DEBUG=1

"${P7_PYTHON}" -c 'import os, torch, torch_npu; torch.npu.set_device(0); print("P7_REUSE_NPU physical=%s logical=0 device=%s" % (os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), torch.npu.get_device_name(0)))' \
    2>&1 | tee "${P7_RESULT}/npu.log"
P7_NPU_RC=${PIPESTATUS[0]}
if [[ "${P7_NPU_RC}" != "0" ]]; then
    fail "npu_context_failed"
fi

(
    cd "${P7_BRIDGE_ROOT}" || exit 1
    USE_NINJA=0 \
    NANOVLLM_A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_BUILD_JOBS="${P7_BUILD_JOBS}" \
    "${P7_PYTHON}" score_receipt_speculative_executor_setup.py build_ext --inplace --force
) 2>&1 | tee "${P7_RESULT}/bridge_build.log"
P7_BRIDGE_RC=${PIPESTATUS[0]}
P7_BRIDGE_SO="$(find "${P7_BRIDGE_ROOT}/nanovllm_score_receipt_speculative_executor" -maxdepth 1 -type f -name '_C*.so' -print -quit 2>/dev/null)"
if [[ "${P7_BRIDGE_RC}" != "0" || ! -s "${P7_BRIDGE_SO}" ]]; then
    fail "bridge_rebuild_failed"
fi
printf 'P7_REUSE_BRIDGE status=0 artifact=%s\n' "${P7_BRIDGE_SO}"

"${P7_PYTHON}" "${P7_CONTRACT_TEST}" --ops-root "${OPS_ROOT}" \
    2>&1 | tee "${P7_RESULT}/contract.log"
P7_CONTRACT_RC=${PIPESTATUS[0]}
if [[ "${P7_CONTRACT_RC}" == "0" ]]; then
    timeout 120s "${P7_PYTHON}" "${P7_TEST}" --device npu:0 --fixture-only \
        2>&1 | tee "${P7_RESULT}/fixture_preflight.log"
    P7_PREFLIGHT_RC=${PIPESTATUS[0]}
else
    P7_PREFLIGHT_RC=98
fi
run_direct_async_fail_closed_case() {
    local case_log="${P7_RESULT}/fail_closed_direct_async.log"
    local case_rc=0
    local descriptor_ledger=0
    local workspace=0
    local launch=0
    local descriptor_lease=0
    local pta_ready=0
    local pta_uninit=0
    local submit=0
    local task_queue_handoff=0
    local terminal_lease=0
    local test_pass=0
    local teardown=0

    # The invalid control rejects in the device kernel before lease/PostOnly/CQ.
    # It still proves the product host path is direct-current-stream and does
    # not introduce the old torch_npu task-queue ownership boundary.
    timeout 90s "${P7_PYTHON}" "${P7_TEST}" --device npu:0 \
        2>&1 | tee "${case_log}"
    case_rc=${PIPESTATUS[0]}

    grep -Fq 'P7_1_DESCRIPTOR_LEDGER api=aclnnA5ScoreReceiptSpeculativeExecutor tuple_args=33 acl_tensors=31 unique_acl_tensors=31' "${case_log}" && descriptor_ledger=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=direct_async_workspace_call_begin' "${case_log}" && workspace=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=direct_async_launch_done' "${case_log}" && launch=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=direct_async_descriptor_lease_held' "${case_log}" && descriptor_lease=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=direct_async_pta_cache_ready' "${case_log}" && pta_ready=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=direct_async_pta_cache_uninit' "${case_log}" && pta_uninit=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=direct_async_submit_done' "${case_log}" && submit=1
    grep -Fq 'handoff_runopapi_by_value' "${case_log}" && task_queue_handoff=1
    grep -Fq 'P7_1_TERMINAL_LEASE released=1 pending=0 terminal=stream_synchronized' "${case_log}" && terminal_lease=1
    grep -Fq 'P7_1_TEST mode=fail_closed pass=1' "${case_log}" && test_pass=1
    grep -Fq 'P7_1_TEARDOWN mode=fail_closed stage=python_refs_collected sync=1' "${case_log}" && teardown=1
    printf 'P7_DIRECT_ASYNC_FAIL_CLOSED rc=%s descriptor_ledger=%s workspace=%s launch=%s descriptor_lease=%s terminal_lease=%s pta_ready=%s pta_uninit=%s submit=%s task_queue_handoff=%s test_pass=%s teardown=%s real_hccp=scheduled_after_pass log=%s\n' \
        "${case_rc}" "${descriptor_ledger}" "${workspace}" "${launch}" \
        "${descriptor_lease}" "${terminal_lease}" "${pta_ready}" "${pta_uninit}" "${submit}" \
        "${task_queue_handoff}" "${test_pass}" "${teardown}" "${case_log}"
    if [[ "${case_rc}" != "0" || "${descriptor_ledger}" != "1" || \
          "${workspace}" != "1" || "${launch}" != "1" || \
          "${descriptor_lease}" != "1" || "${terminal_lease}" != "1" || \
          "${pta_ready}" != "1" || \
          "${pta_uninit}" != "1" || "${submit}" != "1" || \
          "${task_queue_handoff}" != "0" || "${test_pass}" != "1" || \
          "${teardown}" != "1" ]]; then
        return 1
    fi
    return 0
}

P7_DIRECT_CURRENT_STREAM=0
P7_REAL_RC=98
P7_DIRECT_ASYNC_RC=98
P7_FAIL_CLOSED_RC=98
if [[ "${P7_PREFLIGHT_RC}" == "0" ]]; then
    run_direct_async_fail_closed_case
    P7_DIRECT_ASYNC_RC=$?
    P7_FAIL_CLOSED_RC="${P7_DIRECT_ASYNC_RC}"
    if [[ "${P7_DIRECT_ASYNC_RC}" == "0" ]]; then
        P7_DIRECT_CURRENT_STREAM=1
        timeout 180s "${P7_PYTHON}" "${P7_TEST}" --device npu:0 \
            --fixture-module p7_1_real_direct_hccp_fixture \
            2>&1 | tee "${P7_RESULT}/real_direct_hccp.log"
        P7_REAL_RC=${PIPESTATUS[0]}
        if [[ "${P7_REAL_RC}" != "0" ]]; then
            P7_FAIL_CLOSED_RC=1
        fi
    else
        printf 'P7_REAL_HCCP_SKIPPED reason=direct_async_fail_closed_gate_failed\n'
    fi
    printf 'P7_REAL_FLOW_FINAL status=%s direct_async=%s real_hccp=%s execution=direct_current_stream_then_real_hccp\n' \
        "${P7_FAIL_CLOSED_RC}" "${P7_DIRECT_ASYNC_RC}" "${P7_REAL_RC}"
else
    printf 'P7_REAL_HCCP_SKIPPED reason=fixture_preflight_failed\n'
fi
printf 'P7_DIRECT_CURRENT_STREAM observed=%s api=A5ScoreReceiptSpeculativeExecutor\n' \
    "${P7_DIRECT_CURRENT_STREAM}"

grep -hE '^(P7_1_(FIXTURE|SOURCE_ABI|CONTROL|RECEIPTS|QUEUE|TRANSPORT|FAILURE|REAL_BIND|REAL_LIFECYCLE|TEST|TEST_FAIL|CONTRACT_PASS|ACL_DSO|HOST_DSO|DESCRIPTOR_LEDGER|TERMINAL_LEASE)|P7_(DIRECT_ASYNC_FAIL_CLOSED|DIRECT_CURRENT_STREAM|REAL_FLOW_FINAL|REAL_HCCP_SKIPPED)|NANOVLLM_A5_OPAPI_LIFECYCLE .*stage=(direct_async_enqueue_ready|direct_async_pta_cache_ready|direct_async_pta_cache_unavailable|direct_async_pta_cache_uninit|direct_async_pta_cache_not_active|direct_async_workspace_call_begin|direct_async_workspace_ready|direct_async_launch_begin|direct_async_launch_done|direct_async_descriptor_lease_held|direct_async_submit_done))' \
    "${P7_RESULT}"/*.log 2>/dev/null | tail -n 120
if [[ "${P7_CONTRACT_RC}" == "0" && "${P7_PREFLIGHT_RC}" == "0" && \
        "${P7_FAIL_CLOSED_RC}" == "0" && "${P7_REAL_RC}" == "0" ]]; then
    printf 'P7_REUSE_FINAL status=0 contract=0 preflight=0 direct_async=0 real=0 result=%s\n' "${P7_RESULT}"
else
    printf 'P7_REUSE_FINAL status=1 contract=%s preflight=%s direct_async=%s real=%s result=%s\n' \
        "${P7_CONTRACT_RC}" "${P7_PREFLIGHT_RC}" "${P7_FAIL_CLOSED_RC}" "${P7_REAL_RC}" "${P7_RESULT}"
    exit 1
fi
