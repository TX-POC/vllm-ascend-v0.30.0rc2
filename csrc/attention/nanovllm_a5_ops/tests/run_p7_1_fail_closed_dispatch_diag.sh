#!/usr/bin/env bash
# One bounded, C8-free diagnostic for the P7.1 ACLNN boundary.
#
# This runner deliberately dispatches only the built-in fail-closed fixture:
# candidate_count=0 and an all-zero KVAF control block.  Therefore no queue
# lease, Direct-HCCP PostOnly, CQ poll, or DMA may occur.  It is safe to use
# against an already-provenanced P7.1 OPP and rebuilds only the small bridge.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf 'P7_DIAG_FINAL status=2 reason=runner_must_be_executed_with_bash_not_sourced\n' >&2
    return 2
fi

set +e
set +u
set -o pipefail

# A cleanup-bisect changes host descriptor/PTA lifetime and is not a valid
# diagnostic of P7's normal ordered hand-off.  Start from the same clean
# lifecycle as the full and artifact-reuse runners; an inherited shell option
# must not choose a different execution contract.
unset NANOVLLM_A5_OPAPI_CLEANUP_BISECT NANOVLLM_A5_OPAPI_CLEANUP_MODE

finish() {
    local status="$1"
    local reason="$2"
    printf 'P7_DIAG_FINAL status=%s reason=%s result=%s\n' \
        "${status}" "${reason}" "${P7_RESULT:-missing}"
    return "${status}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
OPS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
VLLM_ROOT_DEFAULT="$(cd "${OPS_ROOT}/../../.." && pwd -P)"
P7_VLLM_ROOT="${P7_VLLM_ROOT:-${VLLM_ROOT_DEFAULT}}"
P7_KVAF_SOURCE="${P7_KVAF_SOURCE:-${KVAF_FUSED_GET_SOURCE:-}}"
P7_REUSE_OPP_ROOT="${P7_REUSE_OPP_ROOT:-}"
P7_KVAF_BUILD="${P7_KVAF_BUILD:-}"
P7_PHYSICAL_NPU_ID="${P7_PHYSICAL_NPU_ID:-7}"
P7_PYTHON="${P7_PYTHON:-$(command -v python3 2>/dev/null)}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.1.0}}"
P7_BUILD_JOBS="${P7_BUILD_JOBS:-16}"
P7_STAMP="$(date +%Y%m%d_%H%M%S)"
P7_RESULT="${P7_RESULT:-$(dirname "${P7_VLLM_ROOT}")/results/p7_1_dispatch_diag_npu${P7_PHYSICAL_NPU_ID}_${P7_STAMP}}"
P7_TEST="${OPS_ROOT}/tests/test_score_receipt_speculative_executor.py"
P7_CONTRACT_TEST="${OPS_ROOT}/tests/test_score_receipt_speculative_executor_contract.py"
P7_BRIDGE_ROOT="${OPS_ROOT}/a5_score_receipt_speculative_executor/torch_extension"

if [[ -e "${P7_RESULT}" ]]; then
    finish 1 "result_path_already_exists"
    exit $?
fi
mkdir -p "${P7_RESULT}"

if [[ -z "${P7_KVAF_SOURCE}" || -z "${P7_REUSE_OPP_ROOT}" || \
      ! -x "${P7_PYTHON}" || ! -f "${P7_TEST}" || \
      ! -f "${P7_CONTRACT_TEST}" || \
      ! -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
    finish 1 "missing_required_env_or_source"
    exit $?
fi
if ! git -C "${P7_VLLM_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
   ! git -C "${P7_KVAF_SOURCE}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    finish 1 "invalid_git_worktree"
    exit $?
fi

P7_MANIFEST_COUNT="$(find "${P7_REUSE_OPP_ROOT}/vendors" -type f \
    -name P7_1_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_PROVENANCE -print 2>/dev/null \
    | wc -l | tr -d ' ')"
P7_MANIFEST="$(find "${P7_REUSE_OPP_ROOT}/vendors" -type f \
    -name P7_1_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_PROVENANCE -print -quit 2>/dev/null)"
P7_OPAPI="$(dirname "${P7_MANIFEST}")/op_api/lib/libcust_opapi.so"
P7_ARTIFACT_VLLM_SHA="$(sed -n 's/^source_head=//p' "${P7_MANIFEST}" 2>/dev/null | sed -n '1p')"
P7_ARTIFACT_KVAF_SHA="$(sed -n 's/^kvaf_source_head=//p' "${P7_MANIFEST}" 2>/dev/null | sed -n '1p')"
P7_RUNTIME_VLLM_SHA="$(git -C "${P7_VLLM_ROOT}" rev-parse HEAD 2>/dev/null || true)"
P7_RUNTIME_KVAF_SHA="$(git -C "${P7_KVAF_SOURCE}" rev-parse HEAD 2>/dev/null || true)"

# An OPP package and its bridge are one ABI/provenance unit.  Do not turn an
# ancestor package into an apparent valid diagnostic merely because a later
# source checkout can rebuild the bridge: that was precisely how an old raw
# ACLNN path was mixed with a newer test harness.  Keep the ancestor bit only
# as a diagnostic clue; exact source equality is the admission gate.
P7_ARTIFACT_ANCESTOR=0
if [[ "${P7_ARTIFACT_VLLM_SHA}" =~ ^[0-9a-f]{40}$ ]] && \
   git -C "${P7_VLLM_ROOT}" merge-base --is-ancestor \
       "${P7_ARTIFACT_VLLM_SHA}" "${P7_RUNTIME_VLLM_SHA}"; then
    P7_ARTIFACT_ANCESTOR=1
fi
P7_ARTIFACT_RUNTIME_EXACT=0
if [[ "${P7_ARTIFACT_VLLM_SHA}" =~ ^[0-9a-f]{40}$ && \
      "${P7_ARTIFACT_VLLM_SHA}" == "${P7_RUNTIME_VLLM_SHA}" ]]; then
    P7_ARTIFACT_RUNTIME_EXACT=1
fi

P7_KVAF_SOURCE_REAL="$(cd "${P7_KVAF_SOURCE}" && pwd -P)"
if [[ -z "${P7_KVAF_BUILD}" ]]; then
    for candidate in "${P7_KVAF_SOURCE}"/build-*; do
        cache_source="$(sed -n 's|^CMAKE_HOME_DIRECTORY:INTERNAL=||p' \
            "${candidate}/CMakeCache.txt" 2>/dev/null | sed -n '1p')"
        if [[ -d "${cache_source}" && \
              "$(cd "${cache_source}" && pwd -P)" == "${P7_KVAF_SOURCE_REAL}" && \
              -s "${candidate}/bin/kvaf_execute_batch_get_kernels.o" && \
              -s "${candidate}/bin/kvaf_execute_batch_put_kernels.o" && \
              -n "$(find "${candidate}/python/kvaf" -maxdepth 1 -type f \
                    -name '_kvaf*.so' -print -quit 2>/dev/null)" ]]; then
            P7_KVAF_BUILD="${candidate}"
            break
        fi
    done
fi

# The selected kernel/python build also has to belong to the exact KVAF
# checkout.  A compatible-looking object file is not sufficient when the
# P7 descriptor/CQ ABI evolves.
P7_KVAF_BUILD_HEAD="$(sed -n '1p' "${P7_KVAF_BUILD}/.p7_kvaf_source_head" 2>/dev/null || true)"
P7_KVAF_BUILD_HEAD_MATCH=0
if [[ "${P7_KVAF_BUILD_HEAD}" == "${P7_RUNTIME_KVAF_SHA}" ]]; then
    P7_KVAF_BUILD_HEAD_MATCH=1
fi

P7_READY=0
if [[ "${P7_MANIFEST_COUNT}" == "1" && -f "${P7_MANIFEST}" && \
      -s "${P7_OPAPI}" && "${P7_ARTIFACT_RUNTIME_EXACT}" == "1" && \
      "${P7_ARTIFACT_KVAF_SHA}" == "${P7_RUNTIME_KVAF_SHA}" && \
      "${P7_KVAF_BUILD_HEAD_MATCH}" == "1" && \
      -s "${P7_KVAF_BUILD}/bin/kvaf_execute_batch_get_kernels.o" && \
      -s "${P7_KVAF_BUILD}/bin/kvaf_execute_batch_put_kernels.o" && \
      -d "${P7_KVAF_BUILD}/python" && \
      -f "${P7_REUSE_OPP_ROOT}/vendors/customize/bin/set_env.bash" ]]; then
    P7_READY=1
fi
printf 'P7_DIAG_GATE ready=%s runtime_vllm=%s artifact_vllm=%s artifact_exact=%s artifact_ancestor=%s runtime_kvaf=%s artifact_kvaf=%s kvaf_build=%s kvaf_build_head=%s kvaf_build_head_match=%s\n' \
    "${P7_READY}" "${P7_RUNTIME_VLLM_SHA}" "${P7_ARTIFACT_VLLM_SHA:-missing}" \
    "${P7_ARTIFACT_RUNTIME_EXACT}" "${P7_ARTIFACT_ANCESTOR}" \
    "${P7_RUNTIME_KVAF_SHA}" "${P7_ARTIFACT_KVAF_SHA:-missing}" \
    "${P7_KVAF_BUILD:-missing}" "${P7_KVAF_BUILD_HEAD:-missing}" \
    "${P7_KVAF_BUILD_HEAD_MATCH}"
if [[ "${P7_READY}" != "1" ]]; then
    finish 1 "provenance_or_runtime_gate_failed"
    exit $?
fi

# The package must validate the binary it actually loads, not the later
# test-only checkout revision.  Clear every ambient package first so that one
# P7.1 vendor is the only possible ACLNN symbol provider.
unset SOC_VERSION ASCEND_VISIBLE_DEVICES ASCEND_CUSTOM_OPP_PATH NANOVLLM_CUST_OPAPI_LIB
export A5_SOC_VERSION=ascend950
export ASCEND_HOME_PATH
export ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
unset SOC_VERSION ASCEND_VISIBLE_DEVICES ASCEND_CUSTOM_OPP_PATH NANOVLLM_CUST_OPAPI_LIB
export A5_SOC_VERSION=ascend950
export ASCEND_RT_VISIBLE_DEVICES="${P7_PHYSICAL_NPU_ID}"
export NPU_ID=0 DEVICE_ID=0
source "${P7_REUSE_OPP_ROOT}/vendors/customize/bin/set_env.bash"
unset ASCEND_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES="${P7_PHYSICAL_NPU_ID}"
export NPU_ID=0 DEVICE_ID=0
# Vendor setup is allowed to set package paths, but not to resurrect a
# test-only lifecycle bisect from the invoking shell.
unset NANOVLLM_A5_OPAPI_CLEANUP_BISECT NANOVLLM_A5_OPAPI_CLEANUP_MODE
export NANOVLLM_A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_OPP="${P7_REUSE_OPP_ROOT}"
export P7_EXPECTED_VLLM_SHA="${P7_ARTIFACT_VLLM_SHA}"
export P7_EXPECTED_KVAF_SHA="${P7_ARTIFACT_KVAF_SHA}"
export P7_KVAF_BATCH_GET_KERNEL="${P7_KVAF_BUILD}/bin/kvaf_execute_batch_get_kernels.o"
export P7_KVAF_BATCH_PUT_KERNEL="${P7_KVAF_BUILD}/bin/kvaf_execute_batch_put_kernels.o"
export PYTHONPATH="${OPS_ROOT}/tests:${P7_BRIDGE_ROOT}:${P7_KVAF_BUILD}/python${PYTHONPATH:+:${PYTHONPATH}}"
export NANOVLLM_A5_OPAPI_DEBUG=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
ulimit -c 0 2>/dev/null || true

(
    cd "${P7_BRIDGE_ROOT}" || exit 1
    USE_NINJA=0 \
    NANOVLLM_A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_BUILD_JOBS="${P7_BUILD_JOBS}" \
    "${P7_PYTHON}" score_receipt_speculative_executor_setup.py build_ext --inplace --force
) 2>&1 | tee "${P7_RESULT}/bridge_build.log"
P7_BRIDGE_RC=${PIPESTATUS[0]}
P7_BRIDGE_SO="$(find "${P7_BRIDGE_ROOT}/nanovllm_score_receipt_speculative_executor" \
    -maxdepth 1 -type f -name '_C*.so' -print -quit 2>/dev/null)"
if [[ "${P7_BRIDGE_RC}" != "0" || ! -s "${P7_BRIDGE_SO}" ]]; then
    finish 1 "bridge_rebuild_failed"
    exit $?
fi

"${P7_PYTHON}" "${P7_CONTRACT_TEST}" --ops-root "${OPS_ROOT}" \
    2>&1 | tee "${P7_RESULT}/contract.log"
P7_CONTRACT_RC=${PIPESTATUS[0]}
if [[ "${P7_CONTRACT_RC}" != "0" ]]; then
    finish 1 "static_contract_failed"
    exit $?
fi

# This process intentionally stops before ACLNN dispatch.  It proves package
# loading/registration, all 31 NPU fixtures, and the target NPU mapping.
timeout -k 5s 60s "${P7_PYTHON}" - <<'PY' 2>&1 | tee "${P7_RESULT}/import_registration.log"
import os
import faulthandler
faulthandler.enable(all_threads=True)

def mark(stage, **fields):
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"P7_DIAG_STAGE stage={stage}" + (f" {suffix}" if suffix else ""), flush=True)

mark("python_start", physical=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), logical=os.environ.get("NPU_ID"))
import torch
mark("torch_imported", version=torch.__version__)
import torch_npu
mark("torch_npu_imported", version=torch_npu.__version__)
torch.npu.set_device(0)
mark("device_ready", name=torch.npu.get_device_name(0))
from test_score_receipt_speculative_executor import _fail_closed_fixture, INPUT_COUNT
inputs = _fail_closed_fixture(torch.device("npu:0"))
assert len(inputs) == INPUT_COUNT
assert int(inputs[1].cpu().item()) == 0
assert int(inputs[16].cpu().reshape(-1)[0].item()) == 0
mark("fixture_ready", inputs=len(inputs), candidate_count=int(inputs[1].cpu().item()), control_magic=int(inputs[16].cpu().reshape(-1)[0].item()))
import nanovllm_score_receipt_speculative_executor as package
mark("bridge_imported", module=package.__name__, opapi=os.environ.get("NANOVLLM_CUST_OPAPI_LIB"))
import nanovllm_score_receipt_speculative_executor._C as bridge
contract = bridge.p7_1_abi_contract()
mark("bridge_contract", abi=contract["abi"], aliases=contract["mutable_ref_outputs"])
op = torch.ops.a5_score_receipt_speculative_executor.score_receipt_speculative_executor
mark("torch_registration_ready", op=str(op).split(".")[-1])
torch.npu.synchronize()
mark("import_registration_pass")
PY
P7_IMPORT_RC=${PIPESTATUS[0]}

P7_DISPATCH_RC=98
P7_DISPATCH_MODE=blocked
if [[ "${P7_IMPORT_RC}" == "0" ]]; then
    if command -v gdb >/dev/null 2>&1; then
        P7_DISPATCH_MODE=gdb
        timeout -k 10s 120s gdb -q -batch \
            -ex 'set pagination off' \
            -ex 'set confirm off' \
            -ex 'set breakpoint pending on' \
            -ex 'break aclnnA5ScoreReceiptSpeculativeExecutorGetWorkspaceSize' \
            -ex 'commands' \
            -ex 'silent' \
            -ex 'printf "P7_GDB_HIT stage=workspace\\n"' \
            -ex 'bt 8' \
            -ex 'continue' \
            -ex 'end' \
            -ex 'break aclnnA5ScoreReceiptSpeculativeExecutor' \
            -ex 'commands' \
            -ex 'silent' \
            -ex 'printf "P7_GDB_HIT stage=launch\\n"' \
            -ex 'bt 8' \
            -ex 'continue' \
            -ex 'end' \
            -ex 'run' \
            -ex 'echo P7_GDB_AFTER_RUN\\n' \
            -ex 'info program' \
            -ex 'bt 32' \
            -ex 'thread apply all bt 12' \
            --args "${P7_PYTHON}" "${P7_TEST}" --device npu:0 \
            2>&1 | tee "${P7_RESULT}/dispatch_gdb.log"
        P7_DISPATCH_RC=${PIPESTATUS[0]}
    else
        P7_DISPATCH_MODE=faulthandler_no_gdb
        timeout -k 10s 120s "${P7_PYTHON}" "${P7_TEST}" --device npu:0 \
            2>&1 | tee "${P7_RESULT}/dispatch_fallback.log"
        P7_DISPATCH_RC=${PIPESTATUS[0]}
    fi
fi

grep -hE '^(P7_DIAG_|P7_GDB_|P7_1_|NANOVLLM_A5_OPAPI_RESOLVE|Program received signal|#0 |#1 |#2 |#3 )' \
    "${P7_RESULT}"/*.log 2>/dev/null | tail -n 240
printf 'P7_DIAG_FINAL import_registration=%s dispatch=%s mode=%s result=%s\n' \
    "${P7_IMPORT_RC}" "${P7_DISPATCH_RC}" "${P7_DISPATCH_MODE}" "${P7_RESULT}"

if [[ "${P7_IMPORT_RC}" == "0" && "${P7_DISPATCH_RC}" == "0" ]]; then
    exit 0
fi
exit 1
