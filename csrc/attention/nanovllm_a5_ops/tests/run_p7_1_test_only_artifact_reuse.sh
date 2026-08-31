#!/usr/bin/env bash
# Re-run P7.1 Python tests without recompiling an OPP only when the compiled
# OPP manifest names the exact runtime/KVAF revisions. The small torch bridge
# is deliberately rebuilt because it owns ACLNN descriptor/workspace lifetime.
# This is intentionally for bridge/test retries, not kernel/host/KVAF changes.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf 'P7_REUSE_FINAL status=2 reason=runner_must_be_executed_with_bash_not_sourced\n' >&2
    return 2
fi

set +e
set +u
set -o pipefail

emit_failure_excerpt() {
    local result_dir="${P7_RESULT:-}"
    [[ -d "${result_dir}" ]] || return 0
    printf 'P7_REUSE_FAILURE_EXCERPT result=%s\n' "${result_dir}"
    grep -hE '^(P7_1_(SOURCE|NPU|CONTROL|ORDERING_|BOUNDARY|FAILURE|TEST_FAIL|FINAL_RECEIPT|TERMINAL_LEASE|FAIL_CLOSED_NOOP)|P7_(ORDERED|REAL|REUSE)|NANOVLLM_A5_OPAPI_LIFECYCLE)' \
        "${result_dir}"/*.log 2>/dev/null | tail -n 96
}

fail() {
    printf 'P7_REUSE_FINAL status=1 reason=%s result=%s\n' "$1" "${P7_RESULT:-missing}"
    emit_failure_excerpt
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

# The product gate is one fresh-process framework-ordered fail-closed launch,
# followed by the real Direct-HCCP fixture. The old cleanup-bisect variables
# are explicitly removed: P7.1 must use the normal torch_npu producer/consumer
# hand-off, never its test-only raw direct-stream bisect route.
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

# The bridge and OPP are one provenance unit.  Even a source ancestor with an
# unchanged-looking kernel is not reusable here: it may contain the former raw
# direct submission path or a different generated ACLNN closure.  Refuse it
# rather than creating another stale-artifact/bridge combination.
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
P7_ARTIFACT_RUNTIME_EXACT=0
if [[ "${P7_ARTIFACT_COMMIT_VALID}" == "1" && \
      "${P7_ARTIFACT_VLLM_SHA}" == "${P7_VLLM_HEAD}" ]]; then
    P7_ARTIFACT_RUNTIME_EXACT=1
fi

# The closure diff remains a diagnostic receipt, but exact manifest provenance
# above is the non-negotiable execution gate.  It prevents a host-only bridge
# rebuild from pairing with a stale OPP whose generated ACLNN ABI predates it.
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
      "${P7_ARTIFACT_RUNTIME_EXACT}" == "1" && \
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

printf 'P7_REUSE_GATE ready=%s runtime_vllm=%s required_runtime=%s runtime_match=%s artifact_vllm=%s artifact_exact=%s artifact_ancestor=%s runtime_kvaf=%s required_kvaf=%s kvaf_match=%s artifact_kvaf=%s manifest_valid=%s closure_unchanged=%s kvaf_build_valid=%s kvaf_build_head=%s kvaf_build_head_match=%s unexpected_paths=<%s>\n' \
    "${P7_REUSE_READY}" "${P7_VLLM_HEAD}" "${P7_REQUIRED_RUNTIME_SHA:-none}" \
    "${P7_RUNTIME_SHA_MATCH}" \
    "${P7_ARTIFACT_VLLM_SHA:-missing}" "${P7_ARTIFACT_RUNTIME_EXACT}" "${P7_ARTIFACT_ANCESTOR}" \
    "${P7_KVAF_HEAD}" "${P7_REQUIRED_KVAF_SHA:-none}" "${P7_KVAF_SHA_MATCH}" \
    "${P7_ARTIFACT_KVAF_SHA:-missing}" \
    "${P7_MANIFEST_VALID}" "${P7_CLOSURE_UNCHANGED}" "${P7_KVAF_BUILD_VALID}" \
    "${P7_KVAF_BUILD_HEAD:-missing}" "${P7_KVAF_BUILD_HEAD_MATCH}" \
    "${P7_UNEXPECTED_PATHS:-none}"
if [[ "${P7_REUSE_READY}" != "1" ]]; then
    fail "exact_opp_provenance_gate_failed"
fi

printf 'P7_REUSE_BRIDGE_PROVENANCE bridge_source=%s opp_source=%s exact_opp_source=1 bridge_rebuild=1\n' \
    "${P7_VLLM_HEAD}" "${P7_ARTIFACT_VLLM_SHA}"

# The package verifies the exact binary it loads.  The gate above already
# proved that its OPP and the bridge checkout have one shared source revision.
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
unset NANOVLLM_A5_OPAPI_CLEANUP_BISECT NANOVLLM_A5_OPAPI_CLEANUP_MODE
unset P7_EXPECTED_VLLM_SHA P7_EXPECTED_KVAF_SHA
export NANOVLLM_A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_OPP="${P7_REUSE_OPP_ROOT}"
export P7_EXPECTED_VLLM_SHA="${P7_ARTIFACT_VLLM_SHA}"
export P7_EXPECTED_KVAF_SHA="${P7_ARTIFACT_KVAF_SHA}"
export P7_KVAF_BATCH_GET_KERNEL="${P7_KVAF_BUILD}/bin/kvaf_execute_batch_get_kernels.o"
export P7_KVAF_BATCH_PUT_KERNEL="${P7_KVAF_BUILD}/bin/kvaf_execute_batch_put_kernels.o"
export PYTHONPATH="${P7_BRIDGE_ROOT}:${P7_KVAF_BUILD}/python${PYTHONPATH:+:${PYTHONPATH}}"
# This is intentionally on for the scoped P7.1 retry. The marker proves that
# the bridge handed an executor closure to torch_npu's ordered task queue.
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
run_ordered_async_fail_closed_case() {
    local case_log="${P7_RESULT}/fail_closed_ordered_async.log"
    local case_rc=0
    local descriptor_ledger=0
    local workspace=0
    local launch=0
    local descriptor_release=0
    local tensor_keepalive=0
    local pta_ready=0
    local pta_uninit=0
    local submit=0
    local task_queue_handoff=0
    local terminal_lease=0
    local torch_return=0
    local stream_terminal=0
    local test_pass=0
    local teardown=0
    local final_receipt=0
    local noop=0
    local ordering_receipt=0

    # The fixture first establishes a valid zero-work control, then queues
    # control.zero_ without synchronizing.  P7 must consume that exact queued
    # invalidation through torch_npu's ordered hand-off; it returns before
    # queue/PostOnly/CQ only if the consumer observes the predecessor.
    P7_1_TEST_EXIT_AFTER_RECEIPT=1 \
        timeout 90s "${P7_PYTHON}" "${P7_TEST}" --device npu:0 \
        2>&1 | tee "${case_log}"
    case_rc=${PIPESTATUS[0]}

    grep -Fq 'P7_1_DESCRIPTOR_LEDGER api=aclnnA5ScoreReceiptSpeculativeExecutor tuple_args=33 acl_tensors=31 unique_acl_tensors=31' "${case_log}" && descriptor_ledger=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=workspace_call_begin' "${case_log}" && workspace=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=launch_done' "${case_log}" && launch=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=descriptor_release_done' "${case_log}" && descriptor_release=1
    grep -Fq 'P7_1_ORDERED_SUBMISSION token=' "${case_log}" && tensor_keepalive=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=pta_cache_ready' "${case_log}" && pta_ready=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=pta_cache_uninit' "${case_log}" && pta_uninit=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=submit_done' "${case_log}" && submit=1
    grep -Fq 'handoff_runopapi_by_value' "${case_log}" && task_queue_handoff=1
    grep -Fq 'P7_1_FAIL_CLOSED_NOOP mutable_refs_unchanged=18 queue_or_private_write=0 control_stat_write=0 ordered_control_predecessor=1' "${case_log}" && noop=1
    grep -Fq 'P7_1_ORDERING_RECEIPT baseline_valid=1 predecessor_invalid_observed=1 control_rejected=0 mutable_refs_unchanged=18' "${case_log}" && ordering_receipt=1
    grep -Fq 'P7_1_TERMINAL_LEASE released=1 pending=0 terminal=stream_synchronized' "${case_log}" && terminal_lease=1
    grep -Fq 'P7_1_BOUNDARY stage=torch_op_return' "${case_log}" && torch_return=1
    grep -Fq 'P7_1_BOUNDARY stage=stream_terminal_done' "${case_log}" && stream_terminal=1
    grep -Fq 'P7_1_TEST mode=fail_closed pass=1' "${case_log}" && test_pass=1
    grep -Fq 'P7_1_TEARDOWN mode=fail_closed stage=python_refs_collected sync=1' "${case_log}" && teardown=1
    grep -Fq 'P7_1_FINAL_RECEIPT mode=fail_closed pass=1 terminal=stream_synchronized teardown=complete exit_strategy=os__exit' "${case_log}" && final_receipt=1
    printf 'P7_ORDERED_ASYNC_FAIL_CLOSED rc=%s descriptor_ledger=%s workspace=%s launch=%s descriptor_release=%s terminal_lease_published=%s torch_return=%s stream_terminal=%s terminal_lease=%s pta_ready=%s pta_uninit=%s submit=%s task_queue_handoff=%s test_pass=%s teardown=%s noop=%s ordering_receipt=%s real_hccp=scheduled_after_pass log=%s\n' \
        "${case_rc}" "${descriptor_ledger}" "${workspace}" "${launch}" \
        "${descriptor_release}" "${tensor_keepalive}" "${torch_return}" "${stream_terminal}" "${terminal_lease}" "${pta_ready}" "${pta_uninit}" "${submit}" \
        "${task_queue_handoff}" "${test_pass}" "${teardown}" "${noop}" \
        "${ordering_receipt}" "${case_log}"
    if [[ "${case_rc}" != "0" || "${descriptor_ledger}" != "1" || \
          "${workspace}" != "1" || "${launch}" != "1" || \
          "${descriptor_release}" != "1" || "${tensor_keepalive}" != "1" || \
          "${torch_return}" != "1" || "${stream_terminal}" != "1" || \
          "${terminal_lease}" != "1" || \
          "${pta_ready}" != "1" || \
          "${pta_uninit}" != "1" || "${submit}" != "1" || \
          "${task_queue_handoff}" != "1" || "${test_pass}" != "1" || \
          "${teardown}" != "1" || "${final_receipt}" != "1" || \
          "${noop}" != "1" || "${ordering_receipt}" != "1" ]]; then
        return 1
    fi
    return 0
}

# The receipt-complete child above validates functionality.  This normal-exit child is a
# *diagnostic only*: it must not suppress a fresh-process real HCCP attempt
# after a fully completed stream terminal.  Its compact receipts say whether a
# remaining crash is before Python atexit, or after it in loader/runtime exit.
P7_TEARDOWN_PROBE_CHILD_RC=98
P7_TEARDOWN_PROBE_CLASS=not_run
run_ordered_async_teardown_probe() {
    local probe_log="${P7_RESULT}/fail_closed_process_teardown_probe.log"
    local probe_rc=0
    local test_pass=0
    local teardown=0
    local final_receipt=0
    local python_atexit=0

    P7_1_TEST_PROCESS_EXIT_DIAG=1 \
        timeout 90s "${P7_PYTHON}" "${P7_TEST}" --device npu:0 \
        2>&1 | tee "${probe_log}"
    probe_rc=${PIPESTATUS[0]}
    P7_TEARDOWN_PROBE_CHILD_RC="${probe_rc}"

    grep -Fq 'P7_1_TEST mode=fail_closed pass=1' "${probe_log}" && test_pass=1
    grep -Fq 'P7_1_TEARDOWN mode=fail_closed stage=python_refs_collected sync=1' "${probe_log}" && teardown=1
    grep -Fq 'P7_1_FINAL_RECEIPT mode=fail_closed pass=1 terminal=stream_synchronized teardown=complete exit_strategy=normal_python_exit' "${probe_log}" && final_receipt=1
    grep -Fq 'P7_1_PROCESS_EXIT stage=python_atexit' "${probe_log}" && python_atexit=1

    if [[ "${probe_rc}" == "0" && "${test_pass}" == "1" && \
          "${teardown}" == "1" && "${final_receipt}" == "1" && \
          "${python_atexit}" == "1" ]]; then
        P7_TEARDOWN_PROBE_CLASS=clean_normal_exit
    elif [[ "${probe_rc}" == "139" && "${test_pass}" == "1" && \
            "${teardown}" == "1" && "${final_receipt}" == "1" && \
            "${python_atexit}" == "1" ]]; then
        P7_TEARDOWN_PROBE_CLASS=after_python_atexit
    elif [[ "${probe_rc}" == "139" && "${test_pass}" == "1" && \
            "${teardown}" == "1" && "${final_receipt}" == "1" ]]; then
        P7_TEARDOWN_PROBE_CLASS=after_test_before_python_atexit
    else
        P7_TEARDOWN_PROBE_CLASS=unclassified
    fi

    printf 'P7_ORDERED_ASYNC_TEARDOWN_PROBE child_rc=%s test_pass=%s teardown=%s final_receipt=%s python_atexit=%s classification=%s gating=diagnostic_only log=%s\n' \
        "${probe_rc}" "${test_pass}" "${teardown}" "${final_receipt}" \
        "${python_atexit}" "${P7_TEARDOWN_PROBE_CLASS}" "${probe_log}"

    # The caller records this diagnosis but gates execution on the two
    # receipt-complete children, not on CPython process shutdown.
    if [[ "${test_pass}" == "1" && "${teardown}" == "1" && \
          "${final_receipt}" == "1" ]]; then
        return 0
    fi
    return 1
}

run_real_direct_hccp_case() {
    local case_log="${P7_RESULT}/real_direct_hccp.log"
    local case_rc=0
    local test_pass=0
    local teardown=0
    local final_receipt=0

    # This is the actual P7 target path: a real cold-bound KVAF control is
    # consumed in a new Python process.  The test fixture performs its own
    # Finish/release/cleanup before the receipt allows this child to exit.
    P7_1_TEST_EXIT_AFTER_RECEIPT=1 \
        timeout 180s "${P7_PYTHON}" "${P7_TEST}" --device npu:0 \
            --fixture-module p7_1_real_direct_hccp_fixture \
        2>&1 | tee "${case_log}"
    case_rc=${PIPESTATUS[0]}
    grep -Fq 'P7_1_TEST mode=real_direct_hccp pass=1 real_hccp=acquire_post_retire_finish_release' "${case_log}" && test_pass=1
    grep -Fq 'P7_1_TEARDOWN mode=real_direct_hccp stage=python_refs_collected sync=1' "${case_log}" && teardown=1
    grep -Fq 'P7_1_FINAL_RECEIPT mode=real_direct_hccp pass=1 terminal=stream_synchronized teardown=complete exit_strategy=os__exit' "${case_log}" && final_receipt=1
    printf 'P7_REAL_DIRECT_HCCP rc=%s test_pass=%s teardown=%s final_receipt=%s log=%s\n' \
        "${case_rc}" "${test_pass}" "${teardown}" "${final_receipt}" "${case_log}"
    if [[ "${case_rc}" != "0" || "${test_pass}" != "1" || \
          "${teardown}" != "1" || "${final_receipt}" != "1" ]]; then
        return 1
    fi
    return 0
}

P7_ORDERED_HOST_SUBMISSION=0
P7_REAL_RC=98
P7_ORDERED_ASYNC_RC=98
P7_FAIL_CLOSED_RC=98
if [[ "${P7_PREFLIGHT_RC}" == "0" ]]; then
    run_ordered_async_fail_closed_case
    P7_ORDERED_ASYNC_RC=$?
    P7_FAIL_CLOSED_RC="${P7_ORDERED_ASYNC_RC}"
    if [[ "${P7_ORDERED_ASYNC_RC}" == "0" ]]; then
        P7_ORDERED_HOST_SUBMISSION=1
        run_real_direct_hccp_case
        P7_REAL_RC=$?
        if [[ "${P7_REAL_RC}" != "0" ]]; then
            P7_FAIL_CLOSED_RC=1
        fi
        run_ordered_async_teardown_probe
        P7_TEARDOWN_PROBE_RC=$?
    else
        printf 'P7_REAL_HCCP_SKIPPED reason=ordered_async_fail_closed_gate_failed\n'
    fi
    printf 'P7_REAL_FLOW_FINAL status=%s ordered_async=%s real_hccp=%s teardown_probe=%s teardown_probe_child=%s teardown_class=%s execution=framework_ordered_submit_then_device_post_cq\n' \
        "${P7_FAIL_CLOSED_RC}" "${P7_ORDERED_ASYNC_RC}" "${P7_REAL_RC}" \
        "${P7_TEARDOWN_PROBE_RC:-98}" "${P7_TEARDOWN_PROBE_CHILD_RC}" \
        "${P7_TEARDOWN_PROBE_CLASS}"
else
    printf 'P7_REAL_HCCP_SKIPPED reason=fixture_preflight_failed\n'
fi
printf 'P7_ORDERED_HOST_SUBMISSION observed=%s api=A5ScoreReceiptSpeculativeExecutor\n' \
    "${P7_ORDERED_HOST_SUBMISSION}"

grep -hE '^(P7_1_(FIXTURE|SOURCE_ABI|CONTROL|RECEIPTS|QUEUE|TRANSPORT|FAILURE|REAL_BIND|REAL_LIFECYCLE|TEST|TEST_FAIL|FINAL_RECEIPT|PROCESS_EXIT|CONTRACT_PASS|ACL_DSO|HOST_DSO|DESCRIPTOR_LEDGER|TERMINAL_LEASE|BOUNDARY|ORDERED_SUBMISSION|FAIL_CLOSED_NOOP)|P7_(ORDERED_ASYNC_FAIL_CLOSED|ORDERED_ASYNC_TEARDOWN_PROBE|ORDERED_HOST_SUBMISSION|REAL_DIRECT_HCCP|REAL_FLOW_FINAL|REAL_HCCP_SKIPPED)|NANOVLLM_A5_OPAPI_LIFECYCLE .*stage=(enqueue_ready|pta_cache_ready|pta_cache_unavailable|pta_cache_uninit|pta_cache_not_active|workspace_call_begin|workspace_ready|launch_begin|launch_done|descriptor_release_done|submit_done))' \
    "${P7_RESULT}"/*.log 2>/dev/null | tail -n 120
if [[ "${P7_CONTRACT_RC}" == "0" && "${P7_PREFLIGHT_RC}" == "0" && \
        "${P7_FAIL_CLOSED_RC}" == "0" && "${P7_REAL_RC}" == "0" ]]; then
    printf 'P7_REUSE_FINAL status=0 contract=0 preflight=0 ordered_async=0 real=0 teardown_probe_child=%s teardown_class=%s result=%s\n' \
        "${P7_TEARDOWN_PROBE_CHILD_RC}" "${P7_TEARDOWN_PROBE_CLASS}" "${P7_RESULT}"
else
    printf 'P7_REUSE_FINAL status=1 contract=%s preflight=%s ordered_async=%s real=%s result=%s\n' \
        "${P7_CONTRACT_RC}" "${P7_PREFLIGHT_RC}" "${P7_FAIL_CLOSED_RC}" "${P7_REAL_RC}" "${P7_RESULT}"
    exit 1
fi
