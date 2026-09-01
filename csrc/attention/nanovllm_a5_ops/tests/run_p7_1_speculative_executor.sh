#!/usr/bin/env bash
# Build and run one fresh, self-contained P7.1 assembly.
#
# This intentionally has no artifact reuse mode.  It obtains the two exact
# remote branch commits, creates detached worktrees, then builds KVAF, the P7
# OPP, and the bridge beneath one new result directory.  A later source change
# therefore cannot accidentally select an old .so or OPP from another run.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf 'P7_1_FINAL status=2 reason=runner_must_be_executed_with_bash_not_sourced\n' >&2
    return 2
fi

set +e
set +u
set -o pipefail

fail() {
    local reason="$1"
    printf 'P7_1_FINAL status=1 reason=%s result=%s\n' "$reason" "${P7_RUN_ROOT:-missing}"
    if [[ -d "${P7_RUN_ROOT:-}" ]]; then
        grep -hE '^(P7_1_(SOURCE|NPU|ASSEMBLY|PACKAGE_IMPORT|CONTROL|ORDERING_|TILING|FAIL_CLOSED_NOOP|FINAL_RECEIPT|TERMINAL_LEASE|FINAL)|NANOVLLM_A5_OPAPI_LIFECYCLE)' \
            "${P7_RUN_ROOT}"/*.log 2>/dev/null | tail -n 80
    fi
    exit 1
}

git_branch_commit() {
    local repo="$1"
    local branch="$2"
    git -C "$repo" fetch origin \
        "refs/heads/${branch}:refs/remotes/origin/${branch}" || return 1
    git -C "$repo" rev-parse "origin/${branch}^{commit}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
P7_CALLER_VLLM_REPO="${P7_VLLM_REPO:-$(cd "${SCRIPT_DIR}/../../../.." && pwd -P)}"
P7_KVAF_REPO="${P7_KVAF_REPO:-${KVAF_FUSED_GET_SOURCE:-}}"
P7_VLLM_BRANCH="${P7_VLLM_BRANCH:-kip_demo_cs}"
P7_KVAF_BRANCH="${P7_KVAF_BRANCH:-kip_demo_cs_native_enqueue}"
P7_ARTIFACT_ROOT="${P7_ARTIFACT_ROOT:-$(dirname "$P7_CALLER_VLLM_REPO")/results/p7_1_fresh}"
P7_PYTHON="${P7_PYTHON:-$(command -v python3 2>/dev/null)}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.1.0}}"
P7_PHYSICAL_NPU_ID="${P7_PHYSICAL_NPU_ID:-7}"
P7_BUILD_JOBS="${P7_BUILD_JOBS:-16}"
P7_RUNTIME_TIMEOUT="${P7_RUNTIME_TIMEOUT:-120s}"
P7_TASK_QUEUE_ENABLE="${P7_TASK_QUEUE_ENABLE:-1}"
P7_ASCEND_LAUNCH_BLOCKING="${P7_ASCEND_LAUNCH_BLOCKING:-0}"

if [[ -z "$P7_KVAF_REPO" || ! -x "$P7_PYTHON" || "$P7_ARTIFACT_ROOT" != /* || \
      "$P7_ARTIFACT_ROOT" == "/" || ! "$P7_PHYSICAL_NPU_ID" =~ ^[0-9]+$ ]]; then
    printf 'P7_1_SOURCE ready=0 reason=missing_input vllm_repo=%s kvaf_repo=%s artifact_root=%s python=%s\n' \
        "$P7_CALLER_VLLM_REPO" "${P7_KVAF_REPO:-missing}" "$P7_ARTIFACT_ROOT" "${P7_PYTHON:-missing}"
    fail "missing_or_invalid_input"
fi
if [[ "$P7_TASK_QUEUE_ENABLE" != "1" || "$P7_ASCEND_LAUNCH_BLOCKING" != "0" ]]; then
    printf 'P7_1_ENV_GATE pass=0 task_queue_enable=%s launch_blocking=%s required_task_queue_enable=1 required_launch_blocking=0\n' \
        "$P7_TASK_QUEUE_ENABLE" "$P7_ASCEND_LAUNCH_BLOCKING"
    fail "unsafe_ordered_launch_environment"
fi

# No prior P7/P3 assembly is allowed to influence this fresh run.
unset P7_ASSEMBLY_MANIFEST P7_ASSEMBLY_VLLM_SHA P7_ASSEMBLY_KVAF_SHA
unset P7_REUSE_OPP_ROOT P7_OPP_ROOT P7_KVAF_BUILD
unset P7_EXPECTED_VLLM_SHA P7_EXPECTED_KVAF_SHA
if ! git -C "$P7_CALLER_VLLM_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
   ! git -C "$P7_KVAF_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    fail "input_repository_is_not_git"
fi
if ! command -v cmake >/dev/null 2>&1 || ! command -v timeout >/dev/null 2>&1 || \
   ! command -v mktemp >/dev/null 2>&1; then
    fail "missing_required_host_command"
fi
if [[ ! -f "${ASCEND_HOME_PATH}/bin/setenv.bash" || \
      ! -f "${ASCEND_HOME_PATH}/aarch64-linux/asc/include/simt_api/device_atomic_functions.h" ]]; then
    fail "missing_cann"
fi

# The caller worktrees are only Git object stores.  Do not build from them and
# do not merge/reset them; a developer's local edits cannot affect this run.
P7_VLLM_SHA="$(git_branch_commit "$P7_CALLER_VLLM_REPO" "$P7_VLLM_BRANCH")"
P7_KVAF_SHA="$(git_branch_commit "$P7_KVAF_REPO" "$P7_KVAF_BRANCH")"
if [[ ! "$P7_VLLM_SHA" =~ ^[0-9a-f]{40}$ || ! "$P7_KVAF_SHA" =~ ^[0-9a-f]{40}$ ]]; then
    fail "cannot_resolve_remote_branch_commit"
fi

P7_PYEXT_SUFFIX="$($P7_PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")' 2>/dev/null)"
P7_PYBIND11_DIR="$($P7_PYTHON -m pybind11 --cmakedir 2>/dev/null)"
if [[ -z "$P7_PYEXT_SUFFIX" || -z "$P7_PYBIND11_DIR" ]]; then
    fail "python_or_pybind11_environment_incomplete"
fi

mkdir -p "$P7_ARTIFACT_ROOT" || fail "cannot_create_artifact_root"
P7_RUN_ROOT="$(mktemp -d "${P7_ARTIFACT_ROOT}/p7_1_${P7_VLLM_SHA:0:12}_${P7_KVAF_SHA:0:12}_XXXXXX")" || \
    fail "cannot_create_fresh_assembly_directory"
P7_VLLM_ROOT="${P7_RUN_ROOT}/vllm_source"
P7_KVAF_ROOT="${P7_RUN_ROOT}/kvaf_source"
P7_KVAF_BUILD="${P7_RUN_ROOT}/kvaf_build"

printf 'P7_1_SOURCE vllm=%s kvaf=%s vllm_branch=%s kvaf_branch=%s fresh_root=%s\n' \
    "$P7_VLLM_SHA" "$P7_KVAF_SHA" "$P7_VLLM_BRANCH" "$P7_KVAF_BRANCH" "$P7_RUN_ROOT"

git -C "$P7_CALLER_VLLM_REPO" worktree add --detach "$P7_VLLM_ROOT" "$P7_VLLM_SHA" \
    >"${P7_RUN_ROOT}/vllm_worktree.log" 2>&1 || fail "cannot_create_vllm_worktree"
git -C "$P7_KVAF_REPO" worktree add --detach "$P7_KVAF_ROOT" "$P7_KVAF_SHA" \
    >"${P7_RUN_ROOT}/kvaf_worktree.log" 2>&1 || fail "cannot_create_kvaf_worktree"

if [[ "$(git -C "$P7_VLLM_ROOT" rev-parse HEAD 2>/dev/null)" != "$P7_VLLM_SHA" || \
      "$(git -C "$P7_KVAF_ROOT" rev-parse HEAD 2>/dev/null)" != "$P7_KVAF_SHA" || \
      -n "$(git -C "$P7_VLLM_ROOT" status --porcelain --untracked-files=all)" || \
      -n "$(git -C "$P7_KVAF_ROOT" status --porcelain --untracked-files=all)" ]]; then
    fail "canonical_worktree_is_not_exact_and_clean"
fi

P7_OPS_ROOT="${P7_VLLM_ROOT}/csrc/attention/nanovllm_a5_ops"
P7_BUILD_SCRIPT="${P7_OPS_ROOT}/build_nanovllm_ops.sh"
P7_TEST="${P7_OPS_ROOT}/tests/test_score_receipt_speculative_executor.py"
P7_CONTRACT_TEST="${P7_OPS_ROOT}/tests/test_score_receipt_speculative_executor_contract.py"
P7_BRIDGE_SOURCE="${P7_OPS_ROOT}/a5_score_receipt_speculative_executor/torch_extension"
# build_nanovllm_ops.sh deliberately permits build/OPP paths only below its
# source root.  That source root is this run's detached worktree under
# P7_RUN_ROOT, never the caller's checkout, so these generated files remain
# isolated without weakening the build script's path-safety checks.
P7_BUILD_ROOT="${P7_OPS_ROOT}/build/p7_1_fresh_$(basename "$P7_RUN_ROOT")"
P7_OPP_ROOT="${P7_OPS_ROOT}/_custom_opp_p7_1_fresh_$(basename "$P7_RUN_ROOT")"
P7_OPP_VENDOR="${P7_OPP_ROOT}/vendors/customize"
P7_OPP_SET_ENV="${P7_OPP_VENDOR}/bin/set_env.bash"
P7_OPAPI="${P7_OPP_VENDOR}/op_api/lib/libcust_opapi.so"
P7_BRIDGE_PYTHON_ROOT="${P7_RUN_ROOT}/bridge/lib"
P7_BRIDGE_TEMP="${P7_RUN_ROOT}/bridge/temp"
P7_BRIDGE_SO="${P7_BRIDGE_PYTHON_ROOT}/nanovllm_score_receipt_speculative_executor/_C${P7_PYEXT_SUFFIX}"
P7_MANIFEST="${P7_RUN_ROOT}/assembly.env"

if [[ ! -f "$P7_BUILD_SCRIPT" || ! -f "$P7_TEST" || ! -f "$P7_CONTRACT_TEST" || \
      ! -f "${P7_BRIDGE_SOURCE}/score_receipt_speculative_executor_setup.py" ]]; then
    fail "canonical_source_is_missing_p7_files"
fi

unset SOC_VERSION ASCEND_VISIBLE_DEVICES ASCEND_CUSTOM_OPP_PATH NANOVLLM_CUST_OPAPI_LIB
export A5_SOC_VERSION=ascend950
export ASCEND_HOME_PATH ASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
export ASCEND_SIMT_API_DIR="${ASCEND_HOME_PATH}/aarch64-linux/asc/include"
set +u
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
unset SOC_VERSION ASCEND_VISIBLE_DEVICES ASCEND_CUSTOM_OPP_PATH NANOVLLM_CUST_OPAPI_LIB
export A5_SOC_VERSION=ascend950
export ASCEND_RT_VISIBLE_DEVICES="$P7_PHYSICAL_NPU_ID"
export NPU_ID=0 DEVICE_ID=0
# P7.1 has no safe inline/synchronous fallback.  Make the process-start
# torch_npu task-queue contract explicit before the first Python import, then
# re-export it after any OPP environment script below.
export TASK_QUEUE_ENABLE="$P7_TASK_QUEUE_ENABLE"
export ASCEND_LAUNCH_BLOCKING="$P7_ASCEND_LAUNCH_BLOCKING"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${P7_OPS_ROOT}/tests${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
ulimit -c 0 2>/dev/null || true

printf 'P7_1_NPU physical=%s logical=0 runtime_visible=%s task_queue_enable=%s launch_blocking=%s legacy_visible=unset\n' \
    "$P7_PHYSICAL_NPU_ID" "$ASCEND_RT_VISIBLE_DEVICES" "$TASK_QUEUE_ENABLE" "$ASCEND_LAUNCH_BLOCKING"
printf 'P7_1_ENV_GATE pass=1 task_queue_enable=%s launch_blocking=%s\n' \
    "$TASK_QUEUE_ENABLE" "$ASCEND_LAUNCH_BLOCKING"
"$P7_PYTHON" -c 'import torch, torch_npu; torch.npu.set_device(0); print("P7_1_PYTHON exe=%s torch=%s torch_npu=%s device=%s" % (__import__("sys").executable, torch.__version__, torch_npu.__version__, torch.npu.get_device_name(0)))' \
    2>&1 | tee "${P7_RUN_ROOT}/python_npu.log"
P7_PYTHON_RC=${PIPESTATUS[0]}
[[ "$P7_PYTHON_RC" == "0" ]] || fail "physical_npu_context_failed"

"$P7_PYTHON" "$P7_CONTRACT_TEST" --ops-root "$P7_OPS_ROOT" \
    2>&1 | tee "${P7_RUN_ROOT}/contract.log"
P7_CONTRACT_RC=${PIPESTATUS[0]}
[[ "$P7_CONTRACT_RC" == "0" ]] || fail "p7_static_contract_failed"

# Cheap NPU/fixture validation happens before any scoped compilation.
"$P7_PYTHON" "$P7_TEST" --device npu:0 --fixture-only \
    2>&1 | tee "${P7_RUN_ROOT}/fixture_preflight.log"
P7_FIXTURE_RC=${PIPESTATUS[0]}
[[ "$P7_FIXTURE_RC" == "0" ]] || fail "fixture_preflight_failed"

(
    cmake -S "$P7_KVAF_ROOT" -B "$P7_KVAF_BUILD" \
        -DBUILD_TESTING=ON \
        -DKVAF_BUILD_PYTHON=ON \
        -DKVAF_ENABLE_AIV_BATCH_GET=ON \
        -DKVAF_ENABLE_AIV_BATCH_PUT=ON \
        -DPython_EXECUTABLE="$P7_PYTHON" \
        -Dpybind11_DIR="$P7_PYBIND11_DIR" \
        -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" && \
    cmake --build "$P7_KVAF_BUILD" -j "$P7_BUILD_JOBS"
) 2>&1 | tee "${P7_RUN_ROOT}/kvaf_build.log"
P7_KVAF_BUILD_RC=${PIPESTATUS[0]}
P7_KVAF_GET="${P7_KVAF_BUILD}/bin/kvaf_execute_batch_get_kernels.o"
P7_KVAF_PUT="${P7_KVAF_BUILD}/bin/kvaf_execute_batch_put_kernels.o"
P7_KVAF_PYEXT="${P7_KVAF_BUILD}/python/kvaf/_kvaf${P7_PYEXT_SUFFIX}"
if [[ "$P7_KVAF_BUILD_RC" != "0" || ! -s "$P7_KVAF_GET" || ! -s "$P7_KVAF_PUT" || \
      ! -s "$P7_KVAF_PYEXT" ]]; then
    fail "kvaf_fresh_build_failed"
fi

NANOVLLM_A5_OPS_ONLY=A5ScoreReceiptSpeculativeExecutor \
NANOVLLM_A5_SKIP_TORCH_EXTENSION=1 \
NANOVLLM_A5_OPS_PYTHON="$P7_PYTHON" \
NANOVLLM_A5_OPS_BUILD_JOBS="$P7_BUILD_JOBS" \
NANOVLLM_A5_OPS_BUILD_ROOT="$P7_BUILD_ROOT" \
NANOVLLM_A5_LOCAL_OPP="$P7_OPP_ROOT" \
KVAF_FUSED_GET_SOURCE="$P7_KVAF_ROOT" \
ASCEND_SIMT_API_DIR="$ASCEND_SIMT_API_DIR" \
A5_SOC_VERSION=ascend950 \
bash "$P7_BUILD_SCRIPT" 2>&1 | tee "${P7_RUN_ROOT}/p7_opp_build.log"
P7_OPP_RC=${PIPESTATUS[0]}
if [[ "$P7_OPP_RC" != "0" || ! -f "$P7_OPP_SET_ENV" || ! -s "$P7_OPAPI" ]]; then
    fail "p7_opp_fresh_build_failed"
fi

# The bridge goes outside the source worktree, so neither a previous in-place
# .so nor an editable install can be discovered by this run.
(
    cd "$P7_BRIDGE_SOURCE" || exit 1
    USE_NINJA=0 \
    NANOVLLM_A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_BUILD_JOBS="$P7_BUILD_JOBS" \
    "$P7_PYTHON" score_receipt_speculative_executor_setup.py build_py \
        --build-lib "$P7_BRIDGE_PYTHON_ROOT" && \
    USE_NINJA=0 \
    NANOVLLM_A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_BUILD_JOBS="$P7_BUILD_JOBS" \
    "$P7_PYTHON" score_receipt_speculative_executor_setup.py build_ext \
        --build-lib "$P7_BRIDGE_PYTHON_ROOT" --build-temp "$P7_BRIDGE_TEMP" --force
) 2>&1 | tee "${P7_RUN_ROOT}/bridge_build.log"
P7_BRIDGE_RC=${PIPESTATUS[0]}
if [[ "$P7_BRIDGE_RC" != "0" || ! -s "$P7_BRIDGE_SO" ]]; then
    fail "p7_bridge_fresh_build_failed"
fi

# This is the only assembly record.  It is created only after all scoped
# artifacts exist; P7 package import consumes it and never searches directories.
P7_MANIFEST_TMP="${P7_MANIFEST}.tmp.$$"
{
    printf 'schema=1\nassembly_root=%s\n' "$P7_RUN_ROOT"
    printf 'vllm_root=%s\nvllm_sha=%s\n' "$P7_VLLM_ROOT" "$P7_VLLM_SHA"
    printf 'kvaf_root=%s\nkvaf_sha=%s\nkvaf_build=%s\n' \
        "$P7_KVAF_ROOT" "$P7_KVAF_SHA" "$P7_KVAF_BUILD"
    printf 'batch_get_kernel=%s\nbatch_put_kernel=%s\n' "$P7_KVAF_GET" "$P7_KVAF_PUT"
    printf 'opp_vendor=%s\nopapi=%s\n' "$P7_OPP_VENDOR" "$P7_OPAPI"
    printf 'bridge_python_root=%s\nbridge_so=%s\n' "$P7_BRIDGE_PYTHON_ROOT" "$P7_BRIDGE_SO"
    printf 'p7_abi=31tensor_18refout\ntask_queue_enable=%s\nlaunch_blocking=%s\nready=1\n' \
        "$TASK_QUEUE_ENABLE" "$ASCEND_LAUNCH_BLOCKING"
} >"$P7_MANIFEST_TMP" || fail "cannot_write_assembly_manifest"
mv "$P7_MANIFEST_TMP" "$P7_MANIFEST" || fail "cannot_publish_assembly_manifest"

source "$P7_OPP_SET_ENV"
unset SOC_VERSION ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES
export A5_SOC_VERSION=ascend950
export ASCEND_RT_VISIBLE_DEVICES="$P7_PHYSICAL_NPU_ID"
export NPU_ID=0 DEVICE_ID=0
export TASK_QUEUE_ENABLE="$P7_TASK_QUEUE_ENABLE"
export ASCEND_LAUNCH_BLOCKING="$P7_ASCEND_LAUNCH_BLOCKING"
export P7_ASSEMBLY_MANIFEST="$P7_MANIFEST"
export P7_ASSEMBLY_VLLM_SHA="$P7_VLLM_SHA"
export P7_ASSEMBLY_KVAF_SHA="$P7_KVAF_SHA"
export P7_KVAF_BATCH_GET_KERNEL="$P7_KVAF_GET"
export P7_KVAF_BATCH_PUT_KERNEL="$P7_KVAF_PUT"
export PYTHONPATH="${P7_OPS_ROOT}/tests:${P7_BRIDGE_PYTHON_ROOT}:${P7_BRIDGE_SOURCE}:${P7_KVAF_BUILD}/python${PYTHONPATH:+:${PYTHONPATH}}"
unset NANOVLLM_A5_OPAPI_CLEANUP_BISECT NANOVLLM_A5_OPAPI_CLEANUP_MODE
export NANOVLLM_A5_OPAPI_DEBUG=1

printf 'P7_1_ASSEMBLY ready=1 manifest=%s vllm=%s kvaf=%s opp=%s bridge=%s\n' \
    "$P7_MANIFEST" "$P7_VLLM_SHA" "$P7_KVAF_SHA" "$P7_OPP_ROOT" "$P7_BRIDGE_SO"

"$P7_PYTHON" - <<'PY' 2>&1 | tee "${P7_RUN_ROOT}/package_import.log"
import nanovllm_score_receipt_speculative_executor._C as bridge
print(
    "P7_1_PACKAGE_IMPORT pass=1 abi=%s aliases=%s" %
    (bridge.p7_1_abi_contract()["abi"], bridge.p7_1_abi_contract()["mutable_ref_outputs"]),
    flush=True,
)
PY
P7_IMPORT_RC=${PIPESTATUS[0]}
[[ "$P7_IMPORT_RC" == "0" ]] || fail "manifest_bound_package_import_failed"

P7_1_TEST_EXIT_AFTER_RECEIPT=1 \
timeout -k 15s "$P7_RUNTIME_TIMEOUT" "$P7_PYTHON" "$P7_TEST" --device npu:0 \
    2>&1 | tee "${P7_RUN_ROOT}/fail_closed.log"
P7_FAIL_CLOSED_RC=${PIPESTATUS[0]}
P7_FAIL_CLOSED_NOOP=0
P7_FAIL_CLOSED_ORDERED=0
P7_FAIL_CLOSED_RECEIPT=0
grep -Fq 'P7_1_FAIL_CLOSED_NOOP mutable_refs_unchanged=18 queue_or_private_write=0 control_stat_write=0' \
    "${P7_RUN_ROOT}/fail_closed.log" && P7_FAIL_CLOSED_NOOP=1
grep -Fq 'P7_1_ORDERED_SUBMISSION token=' "${P7_RUN_ROOT}/fail_closed.log" && P7_FAIL_CLOSED_ORDERED=1
grep -Fq 'P7_1_ORDERING_RECEIPT baseline_valid=1 predecessor_invalid_observed=1 control_rejected=0 mutable_refs_unchanged=18' \
    "${P7_RUN_ROOT}/fail_closed.log" && P7_FAIL_CLOSED_RECEIPT=1
if [[ "$P7_FAIL_CLOSED_RC" != "0" || "$P7_FAIL_CLOSED_NOOP" != "1" || \
      "$P7_FAIL_CLOSED_ORDERED" != "1" || "$P7_FAIL_CLOSED_RECEIPT" != "1" ]]; then
    fail "fail_closed_gate_failed"
fi

P7_1_TEST_EXIT_AFTER_RECEIPT=1 \
timeout -k 15s "$P7_RUNTIME_TIMEOUT" "$P7_PYTHON" "$P7_TEST" --device npu:0 \
    --fixture-module p7_1_real_direct_hccp_fixture \
    2>&1 | tee "${P7_RUN_ROOT}/real_direct_hccp.log"
P7_REAL_RC=${PIPESTATUS[0]}
P7_REAL_RECEIPT=0
grep -Fq 'P7_1_FINAL_RECEIPT mode=real_direct_hccp pass=1 terminal=stream_synchronized teardown=complete exit_strategy=os__exit' \
    "${P7_RUN_ROOT}/real_direct_hccp.log" && P7_REAL_RECEIPT=1
if [[ "$P7_REAL_RC" != "0" || "$P7_REAL_RECEIPT" != "1" ]]; then
    fail "real_direct_hccp_failed"
fi

grep -hE '^P7_1_(SOURCE|NPU|ASSEMBLY|PACKAGE_IMPORT|CONTROL|ORDERING_|TILING|RECEIPTS|QUEUE|TRANSPORT|REAL_BIND|REAL_LIFECYCLE|ORDERED_SUBMISSION|FAIL_CLOSED_NOOP|TERMINAL_LEASE|FINAL)' \
    "${P7_RUN_ROOT}"/*.log 2>/dev/null | tail -n 96
printf 'P7_1_FINAL status=0 physical_npu=%s logical_npu=0 result=%s real_direct_hccp=1\n' \
    "$P7_PHYSICAL_NPU_ID" "$P7_RUN_ROOT"
