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
P7_CLEANUP_BISECT="${P7_CLEANUP_BISECT:-0}"
P7_STAMP="$(date +%Y%m%d_%H%M%S)"
P7_RESULT="${P7_RESULT:-$(dirname "${P7_VLLM_ROOT}")/results/p7_1_test_reuse_npu${P7_PHYSICAL_NPU_ID}_${P7_STAMP}}"
P7_TEST="${OPS_ROOT}/tests/test_score_receipt_speculative_executor.py"
P7_CONTRACT_TEST="${OPS_ROOT}/tests/test_score_receipt_speculative_executor_contract.py"
P7_BRIDGE_ROOT="${OPS_ROOT}/a5_score_receipt_speculative_executor/torch_extension"
P7_OPS_REL="${OPS_ROOT#"${P7_VLLM_ROOT}"/}"

case "${P7_CLEANUP_BISECT}" in
    0|1) ;;
    *) fail "invalid_P7_CLEANUP_BISECT" ;;
esac

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
        'abi=31in_18aliasout' \
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

P7_REUSE_READY=0
if [[ "${P7_MANIFEST_VALID}" == "1" && "${P7_ARTIFACT_COMMIT_VALID}" == "1" && \
      "${P7_ARTIFACT_ANCESTOR}" == "1" && \
      "${P7_ARTIFACT_KVAF_SHA}" == "${P7_KVAF_HEAD}" && \
      -z "${P7_VLLM_DIRTY}" && -z "${P7_KVAF_DIRTY}" && \
      "${P7_RUNTIME_SHA_MATCH}" == "1" && \
      "${P7_KVAF_SHA_MATCH}" == "1" && \
      "${P7_CLOSURE_UNCHANGED}" == "1" && -z "${P7_UNEXPECTED_PATHS}" && \
      "${P7_KVAF_BUILD_VALID}" == "1" && \
      -f "${P7_BRIDGE_ROOT}/score_receipt_speculative_executor_setup.py" && \
      -f "${P7_REUSE_OPP_ROOT}/vendors/customize/bin/set_env.bash" ]]; then
    P7_REUSE_READY=1
fi

printf 'P7_REUSE_GATE ready=%s runtime_vllm=%s required_runtime=%s runtime_match=%s artifact_vllm=%s artifact_ancestor=%s runtime_kvaf=%s required_kvaf=%s kvaf_match=%s artifact_kvaf=%s manifest_valid=%s closure_unchanged=%s kvaf_build_valid=%s unexpected_paths=<%s>\n' \
    "${P7_REUSE_READY}" "${P7_VLLM_HEAD}" "${P7_REQUIRED_RUNTIME_SHA:-none}" \
    "${P7_RUNTIME_SHA_MATCH}" \
    "${P7_ARTIFACT_VLLM_SHA:-missing}" "${P7_ARTIFACT_ANCESTOR}" \
    "${P7_KVAF_HEAD}" "${P7_REQUIRED_KVAF_SHA:-none}" "${P7_KVAF_SHA_MATCH}" \
    "${P7_ARTIFACT_KVAF_SHA:-missing}" \
    "${P7_MANIFEST_VALID}" "${P7_CLOSURE_UNCHANGED}" "${P7_KVAF_BUILD_VALID}" \
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
# This is intentionally on for the scoped P7.1 retry.  The marker proves the
# bridge used the owning RunOpApi hand-off before a result is interpreted.
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
run_fail_closed_case() {
    local mode="$1"
    local topology="${2:-aliases}"
    local case_log="${P7_RESULT}/fail_closed_${mode}_${topology}.log"
    local case_rc=0
    local config=0
    local completion=0
    local test_pass=0
    local teardown=0
    local descriptor_ledger=0
    local workspace_call=0
    local pta_ready=0
    local workspace_executor_destroyed=0
    local workspace_cleanup=0

    # Each mode gets a fresh interpreter and a fresh torch bridge load.  That
    # is essential: launch_only intentionally defers cleanup until process
    # exit, so sharing an interpreter would contaminate the next checkpoint.
    NANOVLLM_A5_OPAPI_CLEANUP_BISECT=1 \
    NANOVLLM_A5_OPAPI_CLEANUP_MODE="${mode}" \
    NANOVLLM_A5_P7_OUTPUT_TOPOLOGY="${topology}" \
    timeout 90s "${P7_PYTHON}" "${P7_TEST}" --device npu:0 \
        2>&1 | tee "${case_log}"
    case_rc=${PIPESTATUS[0]}

    grep -Fq "P7_1_CLEANUP api=aclnnA5ScoreReceiptSpeculativeExecutor mode=${mode} stage=config bisect=1" "${case_log}" && config=1
    # Search the explicit field sequence rather than an exact physical line:
    # an external CANN logger may append a suffix to the same stderr record.
    grep -Fq 'P7_1_DESCRIPTOR_LEDGER api=aclnnA5ScoreReceiptSpeculativeExecutor tuple_args=51 acl_tensors=49 unique_acl_tensors=49' "${case_log}" && descriptor_ledger=1
    grep -Fq "P7_1_CLEANUP api=aclnnA5ScoreReceiptSpeculativeExecutor mode=${mode} stage=workspace_call_begin bisect=1" "${case_log}" && workspace_call=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=pta_cache_ready' "${case_log}" && pta_ready=1
    if [[ "${mode}" == "workspace_only" ]]; then
        # A workspace query has no launch, so it must explicitly destroy the
        # generated executor before releasing its descriptors and TLS state.
        grep -Fq "P7_1_CLEANUP api=aclnnA5ScoreReceiptSpeculativeExecutor mode=${mode} stage=workspace_executor_destroy_done bisect=1" "${case_log}" && workspace_executor_destroyed=1
        grep -Fq "P7_1_CLEANUP api=aclnnA5ScoreReceiptSpeculativeExecutor mode=${mode} stage=workspace_query_done bisect=1" "${case_log}" && completion=1
        grep -Fq "P7_1_CLEANUP api=aclnnA5ScoreReceiptSpeculativeExecutor mode=${mode} stage=workspace_cleanup_done bisect=1" "${case_log}" && workspace_cleanup=1
        grep -Fq 'P7_1_TEST mode=workspace_only pass=1' "${case_log}" && test_pass=1
        grep -Fq 'P7_1_TEARDOWN mode=workspace_only stage=python_refs_collected sync=1' "${case_log}" && teardown=1
    else
        grep -Fq "P7_1_CLEANUP api=aclnnA5ScoreReceiptSpeculativeExecutor mode=${mode} stage=direct_stream_sync_done bisect=1" "${case_log}" && completion=1
        grep -Fq 'P7_1_TEST mode=fail_closed pass=1' "${case_log}" && test_pass=1
        grep -Fq 'P7_1_TEARDOWN mode=fail_closed stage=python_refs_collected sync=1' "${case_log}" && teardown=1
    fi

    printf 'P7_CLEANUP_CASE mode=%s topology=%s rc=%s config=%s pta_ready=%s workspace_executor_destroyed=%s workspace_cleanup=%s descriptor_ledger=%s workspace_call=%s completion=%s test_pass=%s teardown=%s execution=direct_synchronized log=%s\n' \
        "${mode}" "${topology}" "${case_rc}" "${config}" \
        "${pta_ready}" "${workspace_executor_destroyed}" "${workspace_cleanup}" "${descriptor_ledger}" "${workspace_call}" \
        "${completion}" "${test_pass}" \
        "${teardown}" "${case_log}"

    if [[ "${case_rc}" != "0" || "${config}" != "1" || \
          "${pta_ready}" != "1" || "${descriptor_ledger}" != "1" || \
          "${workspace_call}" != "1" || "${completion}" != "1" || \
          "${test_pass}" != "1" || \
          "${teardown}" != "1" ]]; then
        return 1
    fi
    if [[ "${mode}" == "workspace_only" && \
          ( "${workspace_executor_destroyed}" != "1" || "${workspace_cleanup}" != "1" ) ]]; then
        return 1
    fi
    return 0
}

run_normal_fail_closed_case() {
    local case_log="${P7_RESULT}/fail_closed_normal_async.log"
    local case_rc=0
    local descriptor_ledger=0
    local handoff=0
    local launch=0
    local descriptor_release=0
    local pta_ready=0
    local pta_uninit=0
    local test_pass=0
    local teardown=0

    # This final fresh interpreter is intentionally *not* the real-HCCP
    # fixture.  It proves the normal OpCommand handoff/lifetime path after the
    # direct cleanup matrix has passed, while the invalid P7 control still
    # rejects before any queue lease, DMA PostOnly, or CQ operation.
    env -u NANOVLLM_A5_OPAPI_CLEANUP_BISECT \
        -u NANOVLLM_A5_OPAPI_CLEANUP_MODE \
        -u NANOVLLM_A5_P7_OUTPUT_TOPOLOGY \
        timeout 90s "${P7_PYTHON}" "${P7_TEST}" --device npu:0 \
        2>&1 | tee "${case_log}"
    case_rc=${PIPESTATUS[0]}

    grep -Fq 'P7_1_DESCRIPTOR_LEDGER api=aclnnA5ScoreReceiptSpeculativeExecutor tuple_args=51 acl_tensors=49 unique_acl_tensors=49' "${case_log}" && descriptor_ledger=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=handoff_runopapi_by_value' "${case_log}" && handoff=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=launch_done' "${case_log}" && launch=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=descriptor_release_done' "${case_log}" && descriptor_release=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=pta_cache_ready' "${case_log}" && pta_ready=1
    grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=pta_cache_uninit' "${case_log}" && pta_uninit=1
    grep -Fq 'P7_1_TEST mode=fail_closed pass=1' "${case_log}" && test_pass=1
    grep -Fq 'P7_1_TEARDOWN mode=fail_closed stage=python_refs_collected sync=1' "${case_log}" && teardown=1
    printf 'P7_CLEANUP_NORMAL_ASYNC rc=%s descriptor_ledger=%s handoff=%s launch=%s descriptor_release=%s pta_ready=%s pta_uninit=%s test_pass=%s teardown=%s real_hccp=not_run log=%s\n' \
        "${case_rc}" "${descriptor_ledger}" "${handoff}" "${launch}" \
        "${descriptor_release}" "${pta_ready}" "${pta_uninit}" \
        "${test_pass}" "${teardown}" "${case_log}"
    if [[ "${case_rc}" != "0" || "${descriptor_ledger}" != "1" || \
          "${handoff}" != "1" || "${launch}" != "1" || \
          "${descriptor_release}" != "1" || "${pta_ready}" != "1" || \
          "${pta_uninit}" != "1" || "${test_pass}" != "1" || \
          "${teardown}" != "1" ]]; then
        return 1
    fi
    return 0
}

P7_OWNING_HANDOFF=0
P7_REAL_RC=98
P7_CLEANUP_BISECT_RC=98
P7_CLEANUP_PTA_RC=98
P7_CLEANUP_WORKSPACE_RC=98
P7_CLEANUP_LAUNCH_ALIAS_RC=98
P7_CLEANUP_LAUNCH_DISTINCT_RC=98
P7_CLEANUP_RELEASE_CONVERTED_RC=98
P7_CLEANUP_RELEASE_HUGE_RC=98
P7_CLEANUP_UNINIT_HUGE_RC=98
P7_CLEANUP_NORMAL_ASYNC_RC=98
if [[ "${P7_PREFLIGHT_RC}" == "0" && "${P7_CLEANUP_BISECT}" == "1" ]]; then
    run_fail_closed_case workspace_only aliases
    P7_CLEANUP_WORKSPACE_RC=$?
    if [[ "${P7_CLEANUP_WORKSPACE_RC}" == "0" ]]; then
        run_fail_closed_case launch_only aliases
        P7_CLEANUP_LAUNCH_ALIAS_RC=$?
    else
        # Do not run an output-topology A/B after the earliest safe
        # workspace query has failed.  That would classify an invalid
        # executor/allocator teardown as an alias-ABI failure.
        printf 'P7_CLEANUP_WORKSPACE_BLOCKED reason=workspace_query_or_process_teardown_failed\n'
    fi

    if [[ "${P7_CLEANUP_LAUNCH_ALIAS_RC}" != "0" && \
          "${P7_CLEANUP_WORKSPACE_RC}" == "0" ]]; then
        # The kernel ignores output-address arguments and mutates the matching
        # input state tensors.  Re-run only the first failing launch boundary
        # with distinct output descriptors: this cleanly answers whether 18
        # physical graph aliases are the host-stub corruption source.
        run_fail_closed_case launch_only distinct
        P7_CLEANUP_LAUNCH_DISTINCT_RC=$?
        printf 'P7_CLEANUP_ALIAS_AB mode=launch_only aliases=%s distinct=%s interpretation=%s\n' \
            "${P7_CLEANUP_LAUNCH_ALIAS_RC}" "${P7_CLEANUP_LAUNCH_DISTINCT_RC}" \
            "$([[ "${P7_CLEANUP_LAUNCH_DISTINCT_RC}" == "0" ]] && echo alias_descriptor_topology_suspect || echo launch_or_loader_not_alias_only)"
    elif [[ "${P7_CLEANUP_LAUNCH_ALIAS_RC}" == "0" ]]; then
        run_fail_closed_case release_converted aliases
        P7_CLEANUP_RELEASE_CONVERTED_RC=$?
        if [[ "${P7_CLEANUP_RELEASE_CONVERTED_RC}" == "0" ]]; then
            run_fail_closed_case release_huge_mem aliases
            P7_CLEANUP_RELEASE_HUGE_RC=$?
        fi
        if [[ "${P7_CLEANUP_RELEASE_HUGE_RC}" == "0" ]]; then
            run_fail_closed_case uninit_huge_mem aliases
            P7_CLEANUP_UNINIT_HUGE_RC=$?
        fi
    fi

    P7_CLEANUP_BISECT_RC=1
    if [[ "${P7_CLEANUP_WORKSPACE_RC}" == "0" && \
          "${P7_CLEANUP_LAUNCH_ALIAS_RC}" == "0" && \
          "${P7_CLEANUP_RELEASE_CONVERTED_RC}" == "0" && \
          "${P7_CLEANUP_RELEASE_HUGE_RC}" == "0" && \
          "${P7_CLEANUP_UNINIT_HUGE_RC}" == "0" ]]; then
        P7_CLEANUP_BISECT_RC=0
    fi
    if [[ "${P7_CLEANUP_BISECT_RC}" == "0" ]]; then
        # Every query initializes PTA because it is a GetWorkspaceSize
        # precondition. This final process adds the explicit PTA *uninit*
        # boundary after the prior descriptor and HugeMem boundaries passed.
        run_fail_closed_case pta_cache
        P7_CLEANUP_PTA_RC=$?
    else
        printf 'P7_CLEANUP_PTA_SKIPPED reason=pre_pta_boundary_failed\n'
    fi
    if [[ "${P7_CLEANUP_BISECT_RC}" == "0" && \
          "${P7_CLEANUP_PTA_RC}" == "0" ]]; then
        run_normal_fail_closed_case
        P7_CLEANUP_NORMAL_ASYNC_RC=$?
    else
        printf 'P7_CLEANUP_NORMAL_ASYNC_SKIPPED reason=direct_cleanup_boundary_failed\n'
    fi
    P7_FAIL_CLOSED_RC="${P7_CLEANUP_BISECT_RC}"
    if [[ "${P7_CLEANUP_PTA_RC}" != "0" ]]; then
        P7_FAIL_CLOSED_RC=1
    fi
    if [[ "${P7_CLEANUP_NORMAL_ASYNC_RC}" != "0" ]]; then
        P7_FAIL_CLOSED_RC=1
    fi
    printf 'P7_CLEANUP_BISECT_FINAL status=%s workspace_alias=%s launch_alias=%s launch_distinct=%s release_converted=%s release_huge=%s uninit_huge=%s pta_cache=%s normal_async=%s real_hccp=not_run execution=direct_synchronized_then_normal_async\n' \
        "${P7_FAIL_CLOSED_RC}" "${P7_CLEANUP_WORKSPACE_RC}" \
        "${P7_CLEANUP_LAUNCH_ALIAS_RC}" "${P7_CLEANUP_LAUNCH_DISTINCT_RC}" \
        "${P7_CLEANUP_RELEASE_CONVERTED_RC}" "${P7_CLEANUP_RELEASE_HUGE_RC}" \
        "${P7_CLEANUP_UNINIT_HUGE_RC}" "${P7_CLEANUP_PTA_RC}" \
        "${P7_CLEANUP_NORMAL_ASYNC_RC}"
elif [[ "${P7_PREFLIGHT_RC}" == "0" ]]; then
    timeout 180s "${P7_PYTHON}" "${P7_TEST}" --device npu:0 \
        2>&1 | tee "${P7_RESULT}/fail_closed.log"
    P7_FAIL_CLOSED_RC=${PIPESTATUS[0]}
    if [[ "${P7_FAIL_CLOSED_RC}" == "0" ]] && \
       grep -Fq 'NANOVLLM_A5_OPAPI_LIFECYCLE api=aclnnA5ScoreReceiptSpeculativeExecutor stage=handoff_runopapi_by_value' \
           "${P7_RESULT}/fail_closed.log"; then
        P7_OWNING_HANDOFF=1
    fi
    if [[ "${P7_FAIL_CLOSED_RC}" == "0" && "${P7_OWNING_HANDOFF}" != "1" ]]; then
        # A passing tensor result from another registered bridge is not evidence
        # that the rebuilt bridge was dispatched.  Fail before attempting HCCP.
        P7_FAIL_CLOSED_RC=97
    fi
    if [[ "${P7_FAIL_CLOSED_RC}" == "0" ]]; then
        timeout 180s "${P7_PYTHON}" "${P7_TEST}" --device npu:0 \
            --fixture-module p7_1_real_direct_hccp_fixture \
            2>&1 | tee "${P7_RESULT}/real_direct_hccp.log"
        P7_REAL_RC=${PIPESTATUS[0]}
    fi
else
    P7_FAIL_CLOSED_RC=98
fi
printf 'P7_REUSE_OWNING_HANDOFF observed=%s api=A5ScoreReceiptSpeculativeExecutor\n' \
    "${P7_OWNING_HANDOFF}"
if [[ "${P7_CLEANUP_BISECT}" == "1" ]]; then
    printf 'P7_CLEANUP_EXECUTION model=direct_synchronized_task_queue=not_used\n'
fi

grep -hE '^(P7_1_(FIXTURE|SOURCE_ABI|CONTROL|RECEIPTS|QUEUE|TRANSPORT|FAILURE|REAL_BIND|REAL_LIFECYCLE|TEST|TEST_FAIL|CONTRACT_PASS|CLEANUP|ACL_DSO|HOST_DSO|DESCRIPTOR_LEDGER)|P7_CLEANUP_(CASE|NORMAL_ASYNC|ALIAS_AB|PTA_SKIPPED|BISECT_FINAL)|NANOVLLM_A5_OPAPI_LIFECYCLE .*stage=(handoff_runopapi_by_value|pta_cache_ready|pta_cache_unavailable|pta_cache_uninit|pta_cache_not_active|launch_begin|launch_done|descriptor_release_done|submit_done))' \
    "${P7_RESULT}"/*.log 2>/dev/null | tail -n 120
if [[ "${P7_CLEANUP_BISECT}" == "1" ]]; then
    if [[ "${P7_CONTRACT_RC}" == "0" && "${P7_PREFLIGHT_RC}" == "0" && \
          "${P7_FAIL_CLOSED_RC}" == "0" ]]; then
        printf 'P7_REUSE_FINAL status=0 contract=0 preflight=0 cleanup_bisect=0 real=not_run result=%s\n' "${P7_RESULT}"
    else
        printf 'P7_REUSE_FINAL status=1 contract=%s preflight=%s cleanup_bisect=%s real=not_run result=%s\n' \
            "${P7_CONTRACT_RC}" "${P7_PREFLIGHT_RC}" "${P7_FAIL_CLOSED_RC}" "${P7_RESULT}"
        exit 1
    fi
elif [[ "${P7_CONTRACT_RC}" == "0" && "${P7_PREFLIGHT_RC}" == "0" && \
        "${P7_FAIL_CLOSED_RC}" == "0" && "${P7_REAL_RC}" == "0" ]]; then
    printf 'P7_REUSE_FINAL status=0 contract=0 preflight=0 fail_closed=0 real=0 result=%s\n' "${P7_RESULT}"
else
    printf 'P7_REUSE_FINAL status=1 contract=%s preflight=%s fail_closed=%s real=%s result=%s\n' \
        "${P7_CONTRACT_RC}" "${P7_PREFLIGHT_RC}" "${P7_FAIL_CLOSED_RC}" "${P7_REAL_RC}" "${P7_RESULT}"
    exit 1
fi
