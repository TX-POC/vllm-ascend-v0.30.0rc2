#!/usr/bin/env bash
# Build and run one source-closed P8 Stage-A/B path on an Ascend 950 NPU.
#
# Required environment (the caller must have already fetched these exact,
# clean revisions from the canonical repositories):
#   P8_VLLM_ROOT=/path/to/vLLM-DSA-kip_demo_cs
#   P8_REFERENCE_VLLM_ROOT=/path/to/stable-vllm-reference-worktree (optional)
#   P8_KVAF_ROOT=/path/to/KVAF-kip_demo_cs_native_enqueue
#   P8_VLLM_EXPECTED_SHA=<exact HEAD>
#   P8_KVAF_EXPECTED_SHA=<exact clean HEAD of the paired canonical KVAF branch>
#
# Do not copy a historical SHA into this file or a launch recipe.  The caller
# must obtain the paired value from the same clean KVAF worktree after its
# explicit fetch/fast-forward, then pass that exact value to this runner.
#
# Default mode is `npu`, designed for the 179 Docker environment with physical
# NPU 7 exposed as logical NPU 0.  `P8_RUNNER_MODE=static` is a useful host
# check: it validates the source/ABI contracts without loading CANN or NPU.
# No git fetch/worktree/artifact discovery occurs here.  P8 target artifacts
# are fresh per invocation.  The optional reference root may persist only its
# CMake object cache, and is accepted only when it is clean and at the same
# exact source SHA as the target.

set +u
set -o pipefail

P8_RUNNER_MODE="${P8_RUNNER_MODE:-npu}"
P8_VLLM_ROOT="${P8_VLLM_ROOT:-}"
P8_REFERENCE_VLLM_ROOT="${P8_REFERENCE_VLLM_ROOT:-$P8_VLLM_ROOT}"
P8_KVAF_ROOT="${P8_KVAF_ROOT:-}"
P8_VLLM_EXPECTED_SHA="${P8_VLLM_EXPECTED_SHA:-}"
P8_KVAF_EXPECTED_SHA="${P8_KVAF_EXPECTED_SHA:-}"
P8_PHYSICAL_NPU_ID="${P8_PHYSICAL_NPU_ID:-7}"
P8_PYTHON="${P8_PYTHON:-$(command -v python3 2>/dev/null || true)}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}"
P8_BUILD_JOBS="${P8_BUILD_JOBS:-16}"
P8_TEST_TIMEOUT="${P8_TEST_TIMEOUT:-300s}"
# P8's public-ABI direct acceptance path keeps the host task queue disabled.
# One caller submits C8, Stage-A and P8 directly to its current raw stream in
# FIFO order; device work remains asynchronous.  TQ1 has no public
# queue-drain/raw-launch ABI and is a separate future product path.
P8_TASK_QUEUE_ENABLE="${P8_TASK_QUEUE_ENABLE:-0}"
P8_ASCEND_LAUNCH_BLOCKING="${P8_ASCEND_LAUNCH_BLOCKING:-0}"
# CANN's downloaded source dependencies belong to the stable, exact reference
# checkout.  ``csrc/third_party`` is ignored by Git: it is neither a compiled
# P8 artifact nor a CMake object cache.  Let upstream CANN populate it on the
# first build and reuse it on later runs of this same source root.  This keeps
# the source/artifact provenance rule simple without turning an absent local
# download cache into a false P8 readiness failure.
P8_CANN_3RD_LIB_PATH="${P8_REFERENCE_VLLM_ROOT}/csrc/third_party"
# P8 is a real Direct-HCCP validation runner, not a stale-artifact smoke test.
# The only permitted runtime fixture is built from this run root's fresh KVAF
# output.  Set P8_REQUIRE_REAL_HCCP=0 only for a deliberately fail-closed
# source/ABI diagnosis; the default requires real payload + C8 equivalence.
P8_REAL_FIXTURE_MODULE="${P8_REAL_FIXTURE_MODULE:-p8_real_direct_hccp_fixture}"
P8_REQUIRE_REAL_HCCP="${P8_REQUIRE_REAL_HCCP:-1}"
# `prepost_control_reject` preserves the full pre-POST graph composition but
# intentionally corrupts the already-bound control in place.  It therefore
# cannot claim a validated payload and must not inherit the full-mode gate.
# `prepost_then_full` runs that cut in its own process and, only after it
# passes, runs the real Post/CQ path using the same fresh assembly.  It is the
# shortest valid failure-domain split: no second compilation and no reuse of
# a prior source/artifact assembly.
P8_RUNTIME_CASE="${P8_RUNTIME_CASE:-full}"

runner_fail() {
    printf 'P8_RUNNER_BLOCKED reason=%s\n' "$1"
    return 1
}

run_logged() {
    local label="$1"
    local log="$2"
    local rc=0
    shift 2
    "$@" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    printf 'P8_RUNNER_STEP label=%s status=%s log=%s\n' "$label" "$rc" "$log"
    return "$rc"
}

# A successful C++ shared-library link does not prove that the deployed
# torch_npu runtime exports every symbol the bridge imported.  In particular,
# a default NPUStream::stream() call on the pinned 179 runtime imports the
# private UseStreamResInCurrentThread symbol and fails only at Python import.
# Check the actual DSO before any runtime Python process can load it.
#
# Do not require ldd -r itself to return zero: normal Python extension symbols
# can be supplied by the embedding interpreter rather than an ELF DT_NEEDED
# library.  The contract here is narrower and exact: no P8 bridge may import a
# private torch_npu queue/resource symbol, and no c10_npu/at_npu symbol may be
# unresolved by the deployed dynamic loader.  ldd -r remains recorded so a new
# deployment-side loader problem is visible in the bounded log.
p8_stage_b_direct_abi_gate() {
    local bridge_so="$1"
    local log="$2"
    local ldd_rc=0
    local nm_rc=0

    if [[ ! -s "$bridge_so" ]]; then
        printf 'P8_BRIDGE_ABI_FAIL reason=bridge_missing bridge=%s log=%s\n' \
            "$bridge_so" "$log"
        return 1
    fi
    if ! command -v ldd >/dev/null 2>&1 ||
       ! command -v nm >/dev/null 2>&1 ||
       ! command -v c++filt >/dev/null 2>&1; then
        printf 'P8_BRIDGE_ABI_FAIL reason=elf_tool_missing bridge=%s log=%s\n' \
            "$bridge_so" "$log"
        return 1
    fi

    {
        printf 'P8_BRIDGE_ABI_SCAN bridge=%s phase=ldd_r\n' "$bridge_so"
        ldd -r "$bridge_so" 2>&1 | c++filt
    } >"$log" 2>&1
    ldd_rc=$?

    {
        printf 'P8_BRIDGE_ABI_SCAN bridge=%s phase=undefined_dynamic_symbols\n' \
            "$bridge_so"
        nm -D --undefined-only "$bridge_so" | c++filt
    } >>"$log" 2>&1
    nm_rc=$?
    if [[ "$nm_rc" -ne 0 ]]; then
        printf 'P8_BRIDGE_ABI_FAIL reason=nm_scan_failed bridge=%s log=%s\n' \
            "$bridge_so" "$log"
        return 1
    fi

    if grep -Eq \
        'UseStreamResInCurrentThread|is_core_control_enabled|check_enqueue_need_use|check_dequeue_need_use|OptionsManager::|NPUStream::stream\(\)|NPUStream::isSyncLaunchStream|NPUStream::getRepoStopFlag|OpCommand::RunOpApi' \
        "$log"; then
        printf 'P8_BRIDGE_ABI_FAIL reason=private_torch_npu_symbol bridge=%s ldd_r=%s log=%s\n' \
            "$bridge_so" "$ldd_rc" "$log"
        grep -E \
            'UseStreamResInCurrentThread|is_core_control_enabled|check_enqueue_need_use|check_dequeue_need_use|OptionsManager::|NPUStream::stream\(\)|NPUStream::isSyncLaunchStream|NPUStream::getRepoStopFlag|OpCommand::RunOpApi' \
            "$log" | sed -n '1,32p'
        return 1
    fi
    if grep -Eq 'undefined symbol:.*(c10_npu|at_npu)::' "$log"; then
        printf 'P8_BRIDGE_ABI_FAIL reason=unresolved_torch_npu_symbol bridge=%s ldd_r=%s log=%s\n' \
            "$bridge_so" "$ldd_rc" "$log"
        grep -E 'undefined symbol:.*(c10_npu|at_npu)::' "$log" | sed -n '1,32p'
        return 1
    fi

    printf 'P8_BRIDGE_ABI_C10_UNDEFINED bridge=%s symbols_begin\n' "$bridge_so"
    grep -E '(c10_npu|at_npu)::' "$log" | sed -n '1,48p' || true
    printf 'P8_BRIDGE_ABI_C10_UNDEFINED bridge=%s symbols_end\n' "$bridge_so"
    printf 'P8_BRIDGE_ABI_PASS bridge=%s ldd_r=%s log=%s\n' \
        "$bridge_so" "$ldd_rc" "$log"
}

is_clean_exact_repo() {
    local label="$1"
    local repo="$2"
    local expected="$3"
    # A normal checkout has a .git directory; a linked worktree has a .git
    # *file*.  P8 validation intentionally supports both, because all target
    # servers use linked worktrees for the canonical branch.
    if [[ -z "$repo" || -z "$expected" || ! -e "$repo/.git" ]] ||
       ! git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        printf 'P8_SOURCE label=%s ready=0 reason=missing_git_repo path=%s\n' "$label" "$repo"
        return 1
    fi
    local dirty head
    dirty="$(git -C "$repo" status --porcelain --untracked-files=all 2>/dev/null)"
    head="$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)"
    printf 'P8_SOURCE label=%s head=%s expected=%s dirty=%s\n' \
        "$label" "$head" "$expected" "$([[ -n "$dirty" ]] && echo 1 || echo 0)"
    # Fail closed on every non-ignored source-tree change, but print a bounded
    # receipt once so a remote build never turns into a blind retry loop.
    if [[ -n "$dirty" ]]; then
        printf 'P8_SOURCE_DIRTY label=%s entries_begin\n' "$label"
        printf '%s\n' "$dirty" | sed -n '1,80p'
        printf 'P8_SOURCE_DIRTY label=%s entries_end\n' "$label"
    fi
    [[ -z "$dirty" && "$head" == "$expected" ]]
}

if [[ "$P8_RUNNER_MODE" != "static" && "$P8_RUNNER_MODE" != "npu" ]]; then
    runner_fail "invalid_mode:${P8_RUNNER_MODE}"
    exit 2
fi
if [[ "$P8_RUNTIME_CASE" != "full" &&
      "$P8_RUNTIME_CASE" != "prepost_control_reject" &&
      "$P8_RUNTIME_CASE" != "prepost_then_full" ]]; then
    runner_fail "invalid_runtime_case:${P8_RUNTIME_CASE}"
    exit 2
fi
if [[ "$P8_TASK_QUEUE_ENABLE" != "0" ||
      "$P8_ASCEND_LAUNCH_BLOCKING" != "0" ]]; then
    printf 'P8_TQ0_DIRECT_HOST_ENV pass=0 task_queue_enable=%s launch_blocking=%s required_task_queue_enable=0 required_launch_blocking=0\n' \
        "$P8_TASK_QUEUE_ENABLE" "$P8_ASCEND_LAUNCH_BLOCKING"
    runner_fail "unsafe_tq0_direct_launch_environment"
    exit 2
fi

if [[ -z "$P8_PYTHON" || ! -x "$P8_PYTHON" ]]; then
    runner_fail "python_missing"
    exit 2
fi
if ! is_clean_exact_repo vllm_target "$P8_VLLM_ROOT" "$P8_VLLM_EXPECTED_SHA" ||
   ! is_clean_exact_repo vllm_reference "$P8_REFERENCE_VLLM_ROOT" "$P8_VLLM_EXPECTED_SHA" ||
   ! is_clean_exact_repo kvaf "$P8_KVAF_ROOT" "$P8_KVAF_EXPECTED_SHA"; then
    runner_fail "source_not_clean_or_exact"
    exit 2
fi

P8_OPS_ROOT="$P8_VLLM_ROOT/csrc/attention/nanovllm_a5_ops"
P8_REFERENCE_OPS_ROOT="$P8_REFERENCE_VLLM_ROOT/csrc/attention/nanovllm_a5_ops"
P8_BUILD_SCRIPT="$P8_OPS_ROOT/build_nanovllm_ops.sh"
P8_RUNTIME_TEST="$P8_OPS_ROOT/tests/test_p8_forward_owned_executor_runtime.py"
P8_STAGE_A_SETUP="$P8_OPS_ROOT/a5_score_receipt_candidate_executor/torch_extension/score_receipt_candidate_executor_setup.py"
P8_STAGE_B_SETUP="$P8_OPS_ROOT/a5_p8_forward_owned_executor/torch_extension/p8_forward_owned_executor_setup.py"
P8_REFERENCE_BUILD_SCRIPT="$P8_REFERENCE_OPS_ROOT/a5_official_c8_reference/build_reference_opp.sh"
P8_REFERENCE_SETUP="$P8_REFERENCE_OPS_ROOT/a5_official_c8_reference/torch_extension/official_c8_reference_setup.py"
P8_REFERENCE_CONTRACT_TEST="$P8_OPS_ROOT/tests/test_official_c8_reference_contract.py"
# Both shell build helpers are deliberately invoked below as
# ``bash "$script"``.  They therefore need to be regular readable files, not
# executable files.  Requiring ``-x`` here rejects a normal Git checkout when
# a helper is tracked as mode 100644, before P8 gets a chance to build.
if [[ ! -f "$P8_BUILD_SCRIPT" || ! -r "$P8_BUILD_SCRIPT" ||
      ! -f "$P8_RUNTIME_TEST" || ! -r "$P8_RUNTIME_TEST" ||
      ! -f "$P8_STAGE_A_SETUP" || ! -r "$P8_STAGE_A_SETUP" ||
      ! -f "$P8_STAGE_B_SETUP" || ! -r "$P8_STAGE_B_SETUP" ||
      ! -f "$P8_REFERENCE_BUILD_SCRIPT" || ! -r "$P8_REFERENCE_BUILD_SCRIPT" ||
      ! -f "$P8_REFERENCE_SETUP" || ! -r "$P8_REFERENCE_SETUP" ||
      ! -f "$P8_REFERENCE_CONTRACT_TEST" || ! -r "$P8_REFERENCE_CONTRACT_TEST" ]]; then
    runner_fail "p8_sources_missing_or_unreadable"
    exit 2
fi

if [[ "$P8_RUNNER_MODE" == "static" ]]; then
    P8_STATIC_LOG="${P8_STATIC_LOG:-$P8_OPS_ROOT/build/p8_static_${P8_VLLM_EXPECTED_SHA:0:12}_${P8_KVAF_EXPECTED_SHA:0:12}.log}"
    mkdir -p "$(dirname "$P8_STATIC_LOG")"
    if run_logged static_contracts "$P8_STATIC_LOG" \
        bash -c 'cd "$1" && "$2" -m unittest -v \
          csrc.attention.nanovllm_a5_ops.tests.test_p8_forward_owned_executor_contract \
          csrc.attention.nanovllm_a5_ops.tests.test_p8_stage_a_b_graph_contract && \
          "$2" "$3"' \
        _ "$P8_VLLM_ROOT" "$P8_PYTHON" "$P8_REFERENCE_CONTRACT_TEST"; then
        if bash -n "$P8_BUILD_SCRIPT"; then
            printf 'P8_STATIC_PASS vllm=%s kvaf=%s\n' \
                "$P8_VLLM_EXPECTED_SHA" "$P8_KVAF_EXPECTED_SHA"
            exit 0
        fi
    fi
    printf 'P8_STATIC_FAIL log=%s\n' "$P8_STATIC_LOG"
    exit 1
fi

if [[ "$P8_PHYSICAL_NPU_ID" != "7" ]]; then
    runner_fail "requires_physical_npu_7_got:${P8_PHYSICAL_NPU_ID}"
    exit 2
fi
if [[ ! -f "$ASCEND_HOME_PATH/bin/setenv.bash" ]]; then
    runner_fail "cann_setenv_missing:${ASCEND_HOME_PATH}"
    exit 2
fi

# Do not inherit a logical NPU mapping or the legacy product selector.  On
# 179, this deliberately makes physical NPU 7 appear as logical NPU 0.
unset SOC_VERSION ASCEND_VISIBLE_DEVICES
export A5_SOC_VERSION=ascend950
export ASCEND_RT_VISIBLE_DEVICES="$P8_PHYSICAL_NPU_ID"
export P8_PHYSICAL_NPU_ID
export NPU_ID=0
export DEVICE_ID=0
source "$ASCEND_HOME_PATH/bin/setenv.bash"
# Python bridge builds inherit only exported variables. The generated P8
# ACLNN header includes CANN's aclnn/acl_meta.h, so export this exact root.
export ASCEND_HOME_PATH
export ASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
# This must happen before the NPU probe, which is the first Python process to
# import torch_npu.  Re-export after CANN's environment script so a shell
# default cannot silently override the TQ0 direct ownership contract.
export TASK_QUEUE_ENABLE="$P8_TASK_QUEUE_ENABLE"
export ASCEND_LAUNCH_BLOCKING="$P8_ASCEND_LAUNCH_BLOCKING"
printf 'P8_TQ0_DIRECT_HOST_ENV pass=1 task_queue_enable=%s launch_blocking=%s physical=7 logical=0 queue_order=caller_submission_tq0\n' \
    "$TASK_QUEUE_ENABLE" "$ASCEND_LAUNCH_BLOCKING"
# The runner owns a three-OPP process (unchanged C8, Stage A, Stage B).  Do
# not inherit vendor paths or opapi handles from an earlier P3/P7/P8 shell;
# each bridge below contributes only its freshly built vendor.  This is the
# assembly boundary that prevents a file-existing-but-old OPP from winning an
# ACLNN resolution.
unset ASCEND_CUSTOM_OPP_PATH NANOVLLM_CUST_OPAPI_LIB
unset NANOVLLM_A5_P8_FORWARD_OWNED_OPAPI NANOVLLM_A5_REFERENCE_C8_OPAPI_LIB
export PYTHONNOUSERSITE=1

P8_PROBE_LOG="${P8_OPS_ROOT}/build/p8_probe_${P8_VLLM_EXPECTED_SHA:0:12}_${P8_KVAF_EXPECTED_SHA:0:12}.log"
mkdir -p "$(dirname "$P8_PROBE_LOG")"
if ! run_logged npu7_probe "$P8_PROBE_LOG" "$P8_PYTHON" -c '
import os, torch, torch_npu
requested_queue = int(os.environ["TASK_QUEUE_ENABLE"])
launch_blocking = os.environ.get("ASCEND_LAUNCH_BLOCKING", "")
if requested_queue != 0 or launch_blocking not in ("", "0"):
    raise RuntimeError(
        "unsafe P8 process-start environment: "
        f"TASK_QUEUE_ENABLE={requested_queue} "
        "ASCEND_LAUNCH_BLOCKING=%s" % (launch_blocking or "<unset>"))
torch.npu.set_device(0)
value = torch.ones(1, device="npu:0").cpu().item()
print("P8_NPU_PROBE physical=%s logical=0 visible=%r device=%s value=%s task_queue_requested=%s task_queue_runtime_guard=fresh_process_tq0" % (
    os.environ.get("P8_PHYSICAL_NPU_ID"),
    os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    torch.npu.get_device_name(0), value, requested_queue))
'; then
    runner_fail "npu7_probe_failed"
    exit 1
fi

# Do not preflight a four-package offline cache.  The first upstream CANN
# build may download only the dependencies it actually needs; later builds
# reuse this source-owned, Git-ignored directory.  The compiled OPPs, bridges
# and KVAF build still remain freshly rooted under P8_RUN_ROOT below.
mkdir -p "$P8_CANN_3RD_LIB_PATH"
P8_CANN_3RD_LIB_PATH="$(cd "$P8_CANN_3RD_LIB_PATH" && pwd -P)"
printf 'P8_TOOLCHAIN_CACHE mode=source_owned_lazy_fetch root=%s\n' \
    "$P8_CANN_3RD_LIB_PATH"

mkdir -p "$P8_OPS_ROOT/build"
P8_RUN_ROOT="$(mktemp -d "$P8_OPS_ROOT/build/p8_npu7_${P8_VLLM_EXPECTED_SHA:0:12}_${P8_KVAF_EXPECTED_SHA:0:12}_XXXXXX")"
P8_STAGE_A_BUILD="$P8_RUN_ROOT/stage_a_build"
P8_STAGE_A_OPP="$P8_RUN_ROOT/stage_a_opp"
P8_STAGE_B_BUILD="$P8_RUN_ROOT/stage_b_build"
P8_STAGE_B_OPP="$P8_RUN_ROOT/stage_b_opp"
P8_STAGE_A_BRIDGE="$P8_RUN_ROOT/stage_a_bridge"
P8_STAGE_B_BRIDGE="$P8_RUN_ROOT/stage_b_bridge"
P8_STAGE_A_TEMP="$P8_RUN_ROOT/stage_a_temp"
P8_STAGE_B_TEMP="$P8_RUN_ROOT/stage_b_temp"
P8_KVAF_BUILD="$P8_RUN_ROOT/kvaf_build"
P8_REFERENCE_OPP="$P8_RUN_ROOT/reference_c8_opp"
P8_REFERENCE_BRIDGE="$P8_RUN_ROOT/reference_c8_bridge"
P8_REFERENCE_TEMP="$P8_RUN_ROOT/reference_c8_temp"
P8_SUFFIX="$($P8_PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')"

printf 'P8_RUNNER_CONFIG mode=npu physical=7 logical=0 run_root=%s vllm_target=%s vllm_reference=%s kvaf=%s\n' \
    "$P8_RUN_ROOT" "$P8_VLLM_ROOT" "$P8_REFERENCE_VLLM_ROOT" "$P8_KVAF_ROOT"

# Run the small source-level lifecycle suite once per real run.  The happy
# Direct-HCCP fixture below validates the only hardware transaction it can
# honestly inject; these targeted native contracts cover deterministic no-post,
# failed CQ/private-miss, and forward-terminal quarantine without claiming
# that a happy HCCP run exercised fault injection.
if ! run_logged p8_lifecycle_contracts "$P8_RUN_ROOT/p8_lifecycle_contracts.log" \
    bash -c 'cd "$1" && "$2" -m unittest -v \
      csrc.attention.nanovllm_a5_ops.tests.test_p8_forward_owned_executor_contract \
      csrc.attention.nanovllm_a5_ops.tests.test_p8_stage_a_b_graph_contract && \
      "$2" "$3"' \
    _ "$P8_VLLM_ROOT" "$P8_PYTHON" "$P8_REFERENCE_CONTRACT_TEST"; then
    printf 'P8_RUNNER_FAIL stage=p8_lifecycle_contracts result=%s\n' "$P8_RUN_ROOT"
    exit 1
fi
printf 'P8_TERMINAL_CONTRACT source_cases=happy_post_cq,no_post_drain,failed_cq_private_miss,forward_terminal_quarantine result=pass\n'

# Every real P8 run gets a fresh KVAF build below its unique run root.  Do
# not discover a prior build directory or import an installed ``kvaf``: those
# two shortcuts caused the historical source/artifact mismatch failures.
P8_PYBIND11_DIR="$($P8_PYTHON -m pybind11 --cmakedir 2>/dev/null || true)"
if [[ -z "$P8_PYBIND11_DIR" ]]; then
    runner_fail "pybind11_cmakedir_missing"
    exit 2
fi
if ! run_logged kvaf_configure "$P8_RUN_ROOT/kvaf_configure.log" \
    cmake -S "$P8_KVAF_ROOT" -B "$P8_KVAF_BUILD" \
      -DBUILD_TESTING=ON \
      -DKVAF_BUILD_PYTHON=ON \
      -DKVAF_ENABLE_AIV_BATCH_GET=ON \
      -DKVAF_ENABLE_AIV_BATCH_PUT=ON \
      -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
      -DPython_EXECUTABLE="$P8_PYTHON" \
      -DPython3_EXECUTABLE="$P8_PYTHON" \
      -Dpybind11_DIR="$P8_PYBIND11_DIR"; then
    printf 'P8_RUNNER_FAIL stage=kvaf_configure result=%s\n' "$P8_RUN_ROOT"
    exit 1
fi
if ! run_logged kvaf_build "$P8_RUN_ROOT/kvaf_build.log" \
    cmake --build "$P8_KVAF_BUILD" --parallel "$P8_BUILD_JOBS"; then
    printf 'P8_RUNNER_FAIL stage=kvaf_build result=%s\n' "$P8_RUN_ROOT"
    exit 1
fi
P8_KVAF_BATCH_GET_KERNEL="$P8_KVAF_BUILD/bin/kvaf_execute_batch_get_kernels.o"
P8_KVAF_BATCH_PUT_KERNEL="$P8_KVAF_BUILD/bin/kvaf_execute_batch_put_kernels.o"
if [[ ! -s "$P8_KVAF_BATCH_GET_KERNEL" || ! -s "$P8_KVAF_BATCH_PUT_KERNEL" ||
      ! -d "$P8_KVAF_BUILD/python" ]]; then
    runner_fail "fresh_kvaf_artifacts_missing"
    exit 1
fi
export P8_KVAF_BUILD P8_KVAF_BATCH_GET_KERNEL P8_KVAF_BATCH_PUT_KERNEL
export PYTHONPATH="$P8_KVAF_BUILD/python${PYTHONPATH:+:$PYTHONPATH}"
if ! run_logged kvaf_import "$P8_RUN_ROOT/kvaf_import.log" "$P8_PYTHON" -c '
import os, pathlib, kvaf
root = pathlib.Path(os.environ["P8_KVAF_BUILD"]).resolve()
loaded = pathlib.Path(kvaf.__file__).resolve()
print("P8_KVAF_IMPORT module=%s expected_prefix=%s exact=%d" % (
    loaded, root, int(str(loaded).startswith(str(root)))))
if not str(loaded).startswith(str(root)):
    raise RuntimeError("stale kvaf import")
'; then
    printf 'P8_RUNNER_FAIL stage=kvaf_import result=%s\n' "$P8_RUN_ROOT"
    exit 1
fi

# C8 itself remains unmodified.  Its reference root is source-matched to the
# target root, while its CMake objects may remain at this one stable path.
# CMake therefore recompiles the official tiling source only when it or one
# of its declared dependencies changed; the packaged OPP remains fresh below
# P8_RUN_ROOT on every invocation.
export A5_OFFICIAL_C8_SOURCE_ROOT="$P8_REFERENCE_VLLM_ROOT"
export A5_OFFICIAL_C8_REFERENCE_ROOT="$P8_REFERENCE_OPP"
export A5_OFFICIAL_C8_EXPECTED_SOURCE_HEAD="$P8_VLLM_EXPECTED_SHA"
export A5_OFFICIAL_C8_REFERENCE_BUILD_JOBS="$P8_BUILD_JOBS"
export A5_OFFICIAL_C8_CANN_3RD_LIB_PATH="$P8_CANN_3RD_LIB_PATH"
if ! run_logged reference_c8_opp "$P8_RUN_ROOT/reference_c8_opp.log" \
    bash "$P8_REFERENCE_BUILD_SCRIPT"; then
    printf 'P8_RUNNER_FAIL stage=reference_c8_opp result=%s\n' "$P8_RUN_ROOT"
    exit 1
fi
P8_REFERENCE_OPAPI="$P8_REFERENCE_OPP/vendors/official_c8_reference_transformer/op_api/lib/libcust_opapi.so"
if [[ ! -s "$P8_REFERENCE_OPAPI" ]]; then
    runner_fail "reference_c8_opapi_missing"
    exit 1
fi
export USE_NINJA=0
export A5_OFFICIAL_C8_REFERENCE_BRIDGE_BUILD_JOBS="$P8_BUILD_JOBS"
if ! run_logged reference_c8_bridge "$P8_RUN_ROOT/reference_c8_bridge.log" \
    "$P8_PYTHON" "$P8_REFERENCE_SETUP" \
      build_py --force --build-lib "$P8_REFERENCE_BRIDGE" \
      build_ext --force --build-lib "$P8_REFERENCE_BRIDGE" \
        --build-temp "$P8_REFERENCE_TEMP"; then
    printf 'P8_RUNNER_FAIL stage=reference_c8_bridge result=%s\n' "$P8_RUN_ROOT"
    exit 1
fi
if [[ ! -f "$P8_REFERENCE_BRIDGE/nanovllm_official_c8_reference/__init__.py" ]]; then
    runner_fail "reference_c8_bridge_package_missing"
    exit 1
fi
if [[ ! -s "$P8_REFERENCE_BRIDGE/nanovllm_official_c8_reference/_C$P8_SUFFIX" ]]; then
    runner_fail "reference_c8_bridge_missing"
    exit 1
fi

export NANOVLLM_A5_OPS_PYTHON="$P8_PYTHON"
export NANOVLLM_A5_OPS_BUILD_JOBS="$P8_BUILD_JOBS"
export NANOVLLM_A5_SKIP_TORCH_EXTENSION=1
# Stage-A/B are fresh msopgen projects.  Do not impose the reference C8
# dependency directory on them: their builder already handles its own CANN
# dependency setup, and importing it here would recreate the old cache gate.
unset NANOVLLM_A5_CANN_3RD_LIB_PATH
export NANOVLLM_A5_OPS_ONLY=A5ScoreReceiptCandidateExecutor
export NANOVLLM_A5_OPS_BUILD_ROOT="$P8_STAGE_A_BUILD"
export NANOVLLM_A5_LOCAL_OPP="$P8_STAGE_A_OPP"
if ! run_logged stage_a_opp "$P8_RUN_ROOT/stage_a_opp.log" bash "$P8_BUILD_SCRIPT"; then
    printf 'P8_RUNNER_FAIL stage=stage_a_opp result=%s\n' "$P8_RUN_ROOT"
    exit 1
fi
if [[ ! -s "$P8_STAGE_A_OPP/vendors/customize/op_api/lib/libcust_opapi.so" ]]; then
    runner_fail "stage_a_opapi_missing"
    exit 1
fi

# P8 is one source-closed graph closure: admission validates a clean shared
# arena once, then dynamic workers own POST/DRAIN/QSFA/FORWARD-terminal.  It
# is deliberately not a ``31+1`` core partition.
export NANOVLLM_A5_OPS_ONLY=A5P8ForwardOwnedExecutor,A5P8ForwardControlAdvance,A5P8ArenaAdmissionGate
export NANOVLLM_A5_OPS_BUILD_ROOT="$P8_STAGE_B_BUILD"
export NANOVLLM_A5_LOCAL_OPP="$P8_STAGE_B_OPP"
export KVAF_FUSED_GET_SOURCE="$P8_KVAF_ROOT"
export NANOVLLM_A5_P8_KVAF_EXPECTED_SHA="$P8_KVAF_EXPECTED_SHA"
if ! run_logged stage_b_opp "$P8_RUN_ROOT/stage_b_opp.log" bash "$P8_BUILD_SCRIPT"; then
    printf 'P8_RUNNER_FAIL stage=stage_b_opp result=%s\n' "$P8_RUN_ROOT"
    exit 1
fi
if [[ ! -s "$P8_STAGE_B_OPP/vendors/customize/op_api/lib/libcust_opapi.so" ]]; then
    runner_fail "stage_b_opapi_missing"
    exit 1
fi
# The Stage-B bridge includes the header generated by this exact run-local
# OPP.  Export it before *any* bridge compilation so setup.py cannot resolve an
# installed or prior-run package by accident.
export NANOVLLM_A5_P8_FORWARD_OWNED_OPP="$P8_STAGE_B_OPP"

# The P8 Stage-B bridge has a 28-ref ACLNN ABI: its 19 Python returns are
# same-name mutable references, not an additional C ABI output suffix.  Check
# the freshly generated header and the installed tiling library before Python
# imports any custom operator.  This is a deliberately small, run-local
# provenance guard: it catches a stale or incorrectly assembled host tiler
# without turning artifact provenance into a persistent database.
P8_STAGE_B_VENDOR="$P8_STAGE_B_OPP/vendors/customize"
P8_STAGE_B_GENERATED_HOST="$P8_STAGE_B_BUILD/custom_op/op_host/a5_p8_forward_owned_executor.cpp"
P8_STAGE_B_INSTALLED_HOST="$P8_STAGE_B_VENDOR/op_impl/ai_core/tbe/customize_impl/dynamic/a5_p8_forward_owned_executor.cpp"
# `liboptiling.so` is the framework-facing shim.  The Stage-B host tiler
# compiled by this package lives in the vendor runtime library below.  Checking
# the shim for our source literal would reject a correct package every time.
P8_STAGE_B_TILING_RUNTIME_LIB="$P8_STAGE_B_VENDOR/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
P8_STAGE_B_ACLNN_HEADER="$P8_STAGE_B_VENDOR/op_api/include/aclnn_a5_p8_forward_owned_executor.h"
P8_STAGE_B_SOURCE_OVERLAY=0
P8_STAGE_B_VENDOR_OVERLAY=0
P8_STAGE_B_RUNTIME_MARKER=0
if cmp -s "$P8_OPS_ROOT/a5_p8_forward_owned_executor/op_host/a5_p8_forward_owned_executor.cpp" \
          "$P8_STAGE_B_GENERATED_HOST"; then
    P8_STAGE_B_SOURCE_OVERLAY=1
fi
if cmp -s "$P8_STAGE_B_GENERATED_HOST" "$P8_STAGE_B_INSTALLED_HOST"; then
    P8_STAGE_B_VENDOR_OVERLAY=1
fi
if grep -aFq 'P8_TILING_DIAG_REVISION=p8_tiling_diag_v1' \
        "$P8_STAGE_B_TILING_RUNTIME_LIB" 2>/dev/null; then
    P8_STAGE_B_RUNTIME_MARKER=1
fi
P8_STAGE_B_ACLNN_SIGNATURE=""
P8_STAGE_B_ACLNN_TENSOR_REFS=missing
if [[ -r "$P8_STAGE_B_ACLNN_HEADER" ]]; then
    P8_STAGE_B_ACLNN_SIGNATURE="$(sed -n \
        '/aclnnA5P8ForwardOwnedExecutorGetWorkspaceSize(/,/);/p' \
        "$P8_STAGE_B_ACLNN_HEADER")"
    P8_STAGE_B_ACLNN_TENSOR_REFS="$(printf '%s\n' "$P8_STAGE_B_ACLNN_SIGNATURE" | \
        grep -Eo 'aclTensor[[:space:]]*\*' | wc -l | tr -d '[:space:]')"
fi
printf 'P8_STAGE_B_TILING_ARTIFACT_CHECK source_overlay=%s vendor_overlay=%s runtime_marker=%s aclnn_tensor_refs=%s generated_host=%s installed_host=%s tiling_runtime_lib=%s header=%s\n' \
    "$P8_STAGE_B_SOURCE_OVERLAY" "$P8_STAGE_B_VENDOR_OVERLAY" \
    "$P8_STAGE_B_RUNTIME_MARKER" "$P8_STAGE_B_ACLNN_TENSOR_REFS" \
    "$P8_STAGE_B_GENERATED_HOST" "$P8_STAGE_B_INSTALLED_HOST" \
    "$P8_STAGE_B_TILING_RUNTIME_LIB" "$P8_STAGE_B_ACLNN_HEADER"
if [[ "$P8_STAGE_B_SOURCE_OVERLAY" != "1" ||
      "$P8_STAGE_B_RUNTIME_MARKER" != "1" ||
      "$P8_STAGE_B_ACLNN_TENSOR_REFS" != "28" ]]; then
    runner_fail "stage_b_tiling_artifact_provenance_failed"
    exit 1
fi
# CANN's package installer may rewrite the installed .cpp manifest (for
# example, generated include/path normalization).  It is not the compiled
# runtime source of truth.  The source overlay plus a marker in the actual
# `libcust_opmaster_rt2.0.so` is the meaningful, run-local provenance proof.
printf 'P8_STAGE_B_TILING_ARTIFACT marker=1 source_overlay=1 vendor_overlay=%s aclnn_tensor_refs=%s header=%s tiling_runtime_lib=%s\n' \
    "$P8_STAGE_B_VENDOR_OVERLAY" "$P8_STAGE_B_ACLNN_TENSOR_REFS" \
    "$P8_STAGE_B_ACLNN_HEADER" "$P8_STAGE_B_TILING_RUNTIME_LIB"

export USE_NINJA=0
export NANOVLLM_A5_SCORE_RECEIPT_EXECUTOR_BUILD_JOBS="$P8_BUILD_JOBS"
if ! run_logged stage_a_bridge "$P8_RUN_ROOT/stage_a_bridge.log" \
    "$P8_PYTHON" "$P8_STAGE_A_SETUP" \
      build_py --force --build-lib "$P8_STAGE_A_BRIDGE" \
      build_ext --force --build-lib "$P8_STAGE_A_BRIDGE" \
        --build-temp "$P8_STAGE_A_TEMP"; then
    printf 'P8_RUNNER_FAIL stage=stage_a_bridge result=%s\n' "$P8_RUN_ROOT"
    exit 1
fi
if [[ ! -f "$P8_STAGE_A_BRIDGE/nanovllm_score_receipt_executor/__init__.py" ]]; then
    runner_fail "stage_a_bridge_package_missing"
    exit 1
fi
if [[ ! -s "$P8_STAGE_A_BRIDGE/nanovllm_score_receipt_executor/_C$P8_SUFFIX" ]]; then
    runner_fail "stage_a_bridge_missing"
    exit 1
fi

export NANOVLLM_A5_P8_FORWARD_OWNED_EXECUTOR_BUILD_JOBS="$P8_BUILD_JOBS"
if ! run_logged stage_b_bridge "$P8_RUN_ROOT/stage_b_bridge.log" \
    "$P8_PYTHON" "$P8_STAGE_B_SETUP" \
      build_py --force --build-lib "$P8_STAGE_B_BRIDGE" \
      build_ext --force --build-lib "$P8_STAGE_B_BRIDGE" \
        --build-temp "$P8_STAGE_B_TEMP"; then
    printf 'P8_RUNNER_FAIL stage=stage_b_bridge result=%s\n' "$P8_RUN_ROOT"
    exit 1
fi
if [[ ! -f "$P8_STAGE_B_BRIDGE/nanovllm_p8_forward_owned_executor/__init__.py" ]]; then
    runner_fail "stage_b_bridge_package_missing"
    exit 1
fi
P8_STAGE_B_BRIDGE_SO="$P8_STAGE_B_BRIDGE/nanovllm_p8_forward_owned_executor/_C$P8_SUFFIX"
if [[ ! -s "$P8_STAGE_B_BRIDGE_SO" ]]; then
    runner_fail "stage_b_bridge_missing"
    exit 1
fi
if ! p8_stage_b_direct_abi_gate \
    "$P8_STAGE_B_BRIDGE_SO" "$P8_RUN_ROOT/stage_b_bridge_abi.log"; then
    runner_fail "stage_b_bridge_private_torch_npu_abi"
    exit 1
fi

export NANOVLLM_A5_SCORE_RECEIPT_EXECUTOR_OPP="$P8_STAGE_A_OPP"
export NANOVLLM_A5_REFERENCE_C8_OPP="$P8_REFERENCE_OPP"
export NANOVLLM_A5_REFERENCE_C8_OPAPI_LIB="$P8_REFERENCE_OPAPI"
# The runtime verifies these exact paths after import.  Existence of a .so is
# not provenance: all three packages must come from this runner's unique root.
export P8_REFERENCE_BRIDGE_ROOT="$P8_REFERENCE_BRIDGE"
export P8_STAGE_A_BRIDGE_ROOT="$P8_STAGE_A_BRIDGE"
export P8_STAGE_B_BRIDGE_ROOT="$P8_STAGE_B_BRIDGE"
export P8_VLLM_EXPECTED_SHA P8_KVAF_EXPECTED_SHA
export P3_REFERENCE_EXPECTED_OPP_SOURCE_HEAD="$P8_VLLM_EXPECTED_SHA"
export P3_RUNNER_EXPECTED_HEAD="$P8_VLLM_EXPECTED_SHA"
export P3_NPU_ID=0
export P3_PHYSICAL_NPU_ID="$P8_PHYSICAL_NPU_ID"
# The test module is deliberately first after the freshly built KVAF package,
# so Python cannot discover an editable/installed historical bridge or KVAF.
export PYTHONPATH="$P8_KVAF_BUILD/python:$P8_REFERENCE_BRIDGE:$P8_STAGE_A_BRIDGE:$P8_STAGE_B_BRIDGE:$P8_OPS_ROOT/tests${PYTHONPATH:+:$PYTHONPATH}"
# Building never authorizes a stale source revision to run.  This second
# cheap check catches an accidental edit or revision change during the scoped
# builds without reintroducing a global artifact-fingerprint framework.
if ! is_clean_exact_repo vllm_target "$P8_VLLM_ROOT" "$P8_VLLM_EXPECTED_SHA" ||
   ! is_clean_exact_repo vllm_reference "$P8_REFERENCE_VLLM_ROOT" "$P8_VLLM_EXPECTED_SHA" ||
   ! is_clean_exact_repo kvaf "$P8_KVAF_ROOT" "$P8_KVAF_EXPECTED_SHA"; then
    runner_fail "source_changed_during_scoped_build"
    exit 1
fi
run_p8_runtime_case() {
    local label="$1"
    local case_name="$2"
    local -a args=("$P8_RUNTIME_TEST" --device npu:0 --case "$case_name")
    args+=(--real-fixture-module "$P8_REAL_FIXTURE_MODULE")
    if [[ "$case_name" == "full" && "$P8_REQUIRE_REAL_HCCP" == "1" ]]; then
        args+=(--require-real-hccp)
    fi
    run_logged "$label" "$P8_RUN_ROOT/${label}.log" \
        timeout -k 30s "$P8_TEST_TIMEOUT" "$P8_PYTHON" "${args[@]}"
}

if [[ "$P8_RUNTIME_CASE" == "prepost_then_full" ]]; then
    if ! run_p8_runtime_case runtime_prepost prepost_control_reject; then
        printf 'P8_RUNNER_FAIL stage=runtime_prepost result=%s\n' "$P8_RUN_ROOT"
        exit 1
    fi
    if ! run_p8_runtime_case runtime_full full; then
        printf 'P8_RUNNER_FAIL stage=runtime_full result=%s\n' "$P8_RUN_ROOT"
        exit 1
    fi
else
    if ! run_p8_runtime_case runtime "$P8_RUNTIME_CASE"; then
        printf 'P8_RUNNER_FAIL stage=runtime result=%s\n' "$P8_RUN_ROOT"
        exit 1
    fi
fi

printf 'P8_RUNNER_PASS physical=7 logical=0 kvaf_build=%s stage_a_opp=%s stage_b_opp=%s reference_opp=%s result=%s\n' \
    "$P8_KVAF_BUILD" "$P8_STAGE_A_OPP" "$P8_STAGE_B_OPP" "$P8_REFERENCE_OPP" "$P8_RUN_ROOT"
