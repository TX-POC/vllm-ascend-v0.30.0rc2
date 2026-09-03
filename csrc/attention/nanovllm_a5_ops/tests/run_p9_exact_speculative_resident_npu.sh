#!/usr/bin/env bash
# Build and validate one fresh, source-closed P9 atomic C8 + resident-Get run.
#
# This runner intentionally does not accept a historical OPP, bridge or KVAF
# build directory.  Each NPU invocation creates a unique run root containing
# a fresh KVAF build, source-matched official C8 OPP/bridge, P9 OPP/bridge and
# one fresh Python runtime process.  That is the small provenance rule needed
# to prevent a source change from accidentally running old device code.
#
# Required caller inputs:
#   P9_VLLM_ROOT, P9_KVAF_ROOT
#   P9_VLLM_EXPECTED_SHA, P9_KVAF_EXPECTED_SHA
#   P9_CANN_3RD_LIB_PATH       (an explicit existing CANN dependency cache)
#
# The NPU mode is deliberately strict for the shared 179 environment:
# physical NPU 7 must be exposed as logical npu:0.  It never sets SOC_VERSION;
# CANN selection is A5_SOC_VERSION=ascend950 only.

set -uo pipefail

P9_RUNNER_MODE="${P9_RUNNER_MODE:-npu}"
P9_VLLM_ROOT="${P9_VLLM_ROOT:-}"
P9_REFERENCE_VLLM_ROOT="${P9_REFERENCE_VLLM_ROOT:-$P9_VLLM_ROOT}"
P9_KVAF_ROOT="${P9_KVAF_ROOT:-}"
P9_VLLM_EXPECTED_SHA="${P9_VLLM_EXPECTED_SHA:-}"
P9_KVAF_EXPECTED_SHA="${P9_KVAF_EXPECTED_SHA:-}"
P9_PHYSICAL_NPU_ID="${P9_PHYSICAL_NPU_ID:-7}"
P9_PYTHON="${P9_PYTHON:-$(command -v python3 2>/dev/null || true)}"
P9_BUILD_JOBS="${P9_BUILD_JOBS:-16}"
P9_TEST_TIMEOUT="${P9_TEST_TIMEOUT:-300s}"
P9_CANN_3RD_LIB_PATH="${P9_CANN_3RD_LIB_PATH:-}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}"

runner_blocked() {
    printf 'P9_RUNNER_BLOCKED reason=%s\n' "$1"
    exit 2
}

run_logged() {
    local label="$1"
    local log="$2"
    shift 2
    "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    printf 'P9_RUNNER_STEP label=%s status=%s log=%s\n' "$label" "$rc" "$log"
    return "$rc"
}

is_clean_exact_repo() {
    local label="$1" repo="$2" expected="$3"
    if [[ -z "$repo" || -z "$expected" || ! -e "$repo/.git" ]] || \
       ! git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        printf 'P9_SOURCE label=%s ready=0 reason=missing_git_repo path=%s\n' "$label" "$repo"
        return 1
    fi
    local head dirty
    head="$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)"
    dirty="$(git -C "$repo" status --porcelain --untracked-files=no 2>/dev/null || true)"
    printf 'P9_SOURCE label=%s head=%s expected=%s dirty=%s\n' \
        "$label" "$head" "$expected" "$([[ -n "$dirty" ]] && echo 1 || echo 0)"
    [[ -z "$dirty" && "$head" == "$expected" ]]
}

if [[ "$P9_RUNNER_MODE" != "static" && "$P9_RUNNER_MODE" != "npu" ]]; then
    runner_blocked "invalid_mode:${P9_RUNNER_MODE}"
fi
if [[ -z "$P9_PYTHON" || ! -x "$P9_PYTHON" ]]; then
    runner_blocked "python_missing"
fi
if ! is_clean_exact_repo vllm_target "$P9_VLLM_ROOT" "$P9_VLLM_EXPECTED_SHA" || \
   ! is_clean_exact_repo vllm_reference "$P9_REFERENCE_VLLM_ROOT" "$P9_VLLM_EXPECTED_SHA" || \
   ! is_clean_exact_repo kvaf "$P9_KVAF_ROOT" "$P9_KVAF_EXPECTED_SHA"; then
    runner_blocked "source_not_clean_or_exact"
fi

P9_OPS_ROOT="$P9_VLLM_ROOT/csrc/attention/nanovllm_a5_ops"
P9_REFERENCE_OPS_ROOT="$P9_REFERENCE_VLLM_ROOT/csrc/attention/nanovllm_a5_ops"
P9_RUNTIME_TEST="$P9_OPS_ROOT/tests/test_p9_exact_speculative_resident_runtime.py"
P9_RUNNER_CONTRACT_TEST="$P9_OPS_ROOT/tests/test_p9_exact_speculative_resident_runner_contract.py"
P9_BUILD_SCRIPT="$P9_OPS_ROOT/a5_c8_exact_speculative_resident_get/build_p9_exact_resident_opp.sh"
P9_TARGET_SETUP="$P9_OPS_ROOT/a5_c8_exact_speculative_resident_get/torch_extension/a5_c8_exact_speculative_resident_get_setup.py"
P9_TARGET_TORCH_ROOT="$P9_OPS_ROOT/a5_c8_exact_speculative_resident_get/torch_extension"
P9_REFERENCE_BUILD_SCRIPT="$P9_REFERENCE_OPS_ROOT/a5_official_c8_reference/build_reference_opp.sh"
P9_REFERENCE_SETUP="$P9_REFERENCE_OPS_ROOT/a5_official_c8_reference/torch_extension/official_c8_reference_setup.py"
P9_REFERENCE_TORCH_ROOT="$P9_REFERENCE_OPS_ROOT/a5_official_c8_reference/torch_extension"

for path in \
    "$P9_RUNTIME_TEST" "$P9_RUNNER_CONTRACT_TEST" "$P9_BUILD_SCRIPT" "$P9_TARGET_SETUP" "$P9_REFERENCE_BUILD_SCRIPT" "$P9_REFERENCE_SETUP"; do
    [[ -f "$path" && -r "$path" ]] || runner_blocked "source_file_missing:${path}"
done

if [[ "$P9_RUNNER_MODE" == "static" ]]; then
    P9_STATIC_LOG="${P9_STATIC_LOG:-$P9_OPS_ROOT/build/p9_static_${P9_VLLM_EXPECTED_SHA:0:12}_${P9_KVAF_EXPECTED_SHA:0:12}.log}"
    mkdir -p "$(dirname "$P9_STATIC_LOG")"
    if run_logged static_contracts "$P9_STATIC_LOG" bash -c '
        cd "$1" && "$2" -m unittest -v \
          csrc.attention.nanovllm_a5_ops.tests.test_c8_exact_speculative_resident_overlay \
          csrc.attention.nanovllm_a5_ops.tests.test_c8_exact_speculative_resident_stage_contract \
          csrc.attention.nanovllm_a5_ops.tests.test_a5_c8_exact_speculative_resident_bridge_contract \
          csrc.attention.nanovllm_a5_ops.tests.test_a5_c8_exact_speculative_resident_direct_launch_contract \
          csrc.attention.nanovllm_a5_ops.tests.test_p9_exact_speculative_resident_runner_contract && \
          "$2" -m py_compile "$3" && bash -n "$4" && bash -n "$5"' \
        _ "$P9_VLLM_ROOT" "$P9_PYTHON" "$P9_RUNTIME_TEST" "$P9_BUILD_SCRIPT" "$0"; then
        printf 'P9_STATIC_PASS vllm=%s kvaf=%s\n' "$P9_VLLM_EXPECTED_SHA" "$P9_KVAF_EXPECTED_SHA"
        exit 0
    fi
    printf 'P9_STATIC_FAIL log=%s\n' "$P9_STATIC_LOG"
    exit 1
fi

[[ "$P9_PHYSICAL_NPU_ID" == "7" ]] || runner_blocked "requires_physical_npu_7_got:${P9_PHYSICAL_NPU_ID}"
[[ -f "$ASCEND_HOME_PATH/bin/setenv.bash" ]] || runner_blocked "cann_setenv_missing:${ASCEND_HOME_PATH}"
[[ -n "$P9_CANN_3RD_LIB_PATH" && -d "$P9_CANN_3RD_LIB_PATH" ]] || \
    runner_blocked "cann_third_party_cache_path_missing"
command -v timeout >/dev/null 2>&1 || runner_blocked "timeout_command_missing"

# Do this before importing torch_npu in any child.  Do not inherit the legacy
# product selector or a multi-card mask from an interactive shell.
set +u
# shellcheck disable=SC1090
source "$ASCEND_HOME_PATH/bin/setenv.bash"
unset SOC_VERSION
unset ASCEND_VISIBLE_DEVICES
unset NANOVLLM_A5_REFERENCE_C8_OPAPI_LIB
unset NANOVLLM_A5_EXACT_SPECULATIVE_RESIDENT_OPAPI_LIB
unset ASCEND_CUSTOM_OPP_PATH
export A5_SOC_VERSION=ascend950
export ASCEND_RT_VISIBLE_DEVICES=7
export NPU_ID=0
export DEVICE_ID=0
export TASK_QUEUE_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0

if ! "$P9_PYTHON" - <<'PY'
import os
import torch
import torch_npu
if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "7":
    raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES must be 7")
if int(torch.npu.device_count()) != 1:
    raise RuntimeError("expected exactly one visible NPU")
torch.npu.set_device(0)
value = torch.ones(1, device="npu:0").cpu().item()
print("P9_NPU_PROBE physical=7 logical=0 visible=%r device=%s value=%s" % (
    os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), torch.npu.get_device_name(0), value))
PY
then
    runner_blocked "npu7_probe_failed"
fi

P9_PYBIND11_DIR="$($P9_PYTHON -m pybind11 --cmakedir 2>/dev/null || true)"
[[ -n "$P9_PYBIND11_DIR" ]] || runner_blocked "pybind11_cmakedir_missing"
mkdir -p "$P9_OPS_ROOT/build"
P9_RUN_ROOT="$(mktemp -d "$P9_OPS_ROOT/build/p9_npu7_${P9_VLLM_EXPECTED_SHA:0:12}_${P9_KVAF_EXPECTED_SHA:0:12}_XXXXXX")"
P9_KVAF_BUILD="$P9_RUN_ROOT/kvaf_build"
P9_REFERENCE_OPP="$P9_RUN_ROOT/reference_c8_opp"
P9_REFERENCE_BRIDGE="$P9_RUN_ROOT/reference_c8_bridge"
P9_REFERENCE_TEMP="$P9_RUN_ROOT/reference_c8_temp"
P9_TARGET_PACKAGE_PARENT="$P9_RUN_ROOT/p9_packages"
P9_TARGET_BRIDGE="$P9_RUN_ROOT/p9_target_bridge"
P9_TARGET_TEMP="$P9_RUN_ROOT/p9_target_temp"
P9_SUFFIX="$($P9_PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or ".so")')"
printf 'P9_RUNNER_CONFIG physical=7 logical=0 run_root=%s vllm=%s kvaf=%s cache=%s\n' \
    "$P9_RUN_ROOT" "$P9_VLLM_ROOT" "$P9_KVAF_ROOT" "$P9_CANN_3RD_LIB_PATH"

if ! run_logged kvaf_configure "$P9_RUN_ROOT/kvaf_configure.log" \
    cmake -S "$P9_KVAF_ROOT" -B "$P9_KVAF_BUILD" \
      -DBUILD_TESTING=ON \
      -DKVAF_BUILD_PYTHON=ON \
      -DKVAF_ENABLE_AIV_BATCH_GET=ON \
      -DKVAF_ENABLE_AIV_BATCH_PUT=ON \
      -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
      -DPython_EXECUTABLE="$P9_PYTHON" \
      -DPython3_EXECUTABLE="$P9_PYTHON" \
      -Dpybind11_DIR="$P9_PYBIND11_DIR"; then
    printf 'P9_RUNNER_FAIL stage=kvaf_configure result=%s\n' "$P9_RUN_ROOT"
    exit 1
fi
if ! run_logged kvaf_build "$P9_RUN_ROOT/kvaf_build.log" \
    cmake --build "$P9_KVAF_BUILD" --parallel "$P9_BUILD_JOBS"; then
    printf 'P9_RUNNER_FAIL stage=kvaf_build result=%s\n' "$P9_RUN_ROOT"
    exit 1
fi
P9_KVAF_BATCH_GET_KERNEL="$P9_KVAF_BUILD/bin/kvaf_execute_batch_get_kernels.o"
P9_KVAF_BATCH_PUT_KERNEL="$P9_KVAF_BUILD/bin/kvaf_execute_batch_put_kernels.o"
[[ -s "$P9_KVAF_BATCH_GET_KERNEL" && -s "$P9_KVAF_BATCH_PUT_KERNEL" && -d "$P9_KVAF_BUILD/python" ]] || \
    runner_blocked "fresh_kvaf_artifacts_missing"
# The import verifier below reads this path from its clean child environment.
# Export it before launching Python; exporting it after that verifier would
# make a successful fresh KVAF build look like an import failure.
export P9_KVAF_BUILD
export PYTHONPATH="$P9_KVAF_BUILD/python${PYTHONPATH:+:$PYTHONPATH}"
if ! run_logged kvaf_import "$P9_RUN_ROOT/kvaf_import.log" "$P9_PYTHON" -c '
import os, pathlib, kvaf
root = pathlib.Path(os.environ["P9_KVAF_BUILD"]).resolve()
loaded = pathlib.Path(kvaf.__file__).resolve()
print("P9_KVAF_IMPORT module=%s expected_prefix=%s exact=%d" % (loaded, root, int(str(loaded).startswith(str(root)))))
if not str(loaded).startswith(str(root)):
    raise RuntimeError("stale kvaf import")
'; then
    printf 'P9_RUNNER_FAIL stage=kvaf_import result=%s\n' "$P9_RUN_ROOT"
    exit 1
fi

export A5_OFFICIAL_C8_SOURCE_ROOT="$P9_REFERENCE_VLLM_ROOT"
export A5_OFFICIAL_C8_REFERENCE_ROOT="$P9_REFERENCE_OPP"
export A5_OFFICIAL_C8_EXPECTED_SOURCE_HEAD="$P9_VLLM_EXPECTED_SHA"
export A5_OFFICIAL_C8_REFERENCE_BUILD_JOBS="$P9_BUILD_JOBS"
export A5_OFFICIAL_C8_CANN_3RD_LIB_PATH="$P9_CANN_3RD_LIB_PATH"
if ! run_logged reference_c8_opp "$P9_RUN_ROOT/reference_c8_opp.log" bash "$P9_REFERENCE_BUILD_SCRIPT"; then
    printf 'P9_RUNNER_FAIL stage=reference_c8_opp result=%s\n' "$P9_RUN_ROOT"
    exit 1
fi
P9_REFERENCE_OPAPI="$P9_REFERENCE_OPP/vendors/official_c8_reference_transformer/op_api/lib/libcust_opapi.so"
[[ -s "$P9_REFERENCE_OPAPI" ]] || runner_blocked "reference_c8_opapi_missing"

if ! run_logged reference_c8_bridge "$P9_RUN_ROOT/reference_c8_bridge.log" bash -c '
    cd "$1" && USE_NINJA=0 A5_OFFICIAL_C8_REFERENCE_BRIDGE_BUILD_JOBS="$4" "$2" "$3" \
      build_py --force --build-lib "$5" build_ext --force --build-lib "$5" --build-temp "$6"' \
    _ "$P9_REFERENCE_TORCH_ROOT" "$P9_PYTHON" "$P9_REFERENCE_SETUP" "$P9_BUILD_JOBS" "$P9_REFERENCE_BRIDGE" "$P9_REFERENCE_TEMP"; then
    printf 'P9_RUNNER_FAIL stage=reference_c8_bridge result=%s\n' "$P9_RUN_ROOT"
    exit 1
fi
[[ -f "$P9_REFERENCE_BRIDGE/nanovllm_official_c8_reference/__init__.py" && \
   -s "$P9_REFERENCE_BRIDGE/nanovllm_official_c8_reference/_C$P9_SUFFIX" ]] || \
    runner_blocked "reference_c8_bridge_artifact_missing"

export A5_C8_EXACT_RESIDENT_SOURCE_ROOT="$P9_VLLM_ROOT"
export A5_C8_EXACT_RESIDENT_KVAF_SOURCE_ROOT="$P9_KVAF_ROOT"
export A5_C8_EXACT_RESIDENT_EXPECTED_SOURCE_HEAD="$P9_VLLM_EXPECTED_SHA"
export A5_C8_EXACT_RESIDENT_EXPECTED_KVAF_HEAD="$P9_KVAF_EXPECTED_SHA"
export A5_C8_EXACT_RESIDENT_PACKAGE_PARENT="$P9_TARGET_PACKAGE_PARENT"
export A5_C8_EXACT_RESIDENT_CANN_3RD_LIB_PATH="$P9_CANN_3RD_LIB_PATH"
export A5_C8_EXACT_RESIDENT_BUILD_JOBS="$P9_BUILD_JOBS"
export A5_C8_EXACT_RESIDENT_SOC=ascend950
export A5_C8_EXACT_RESIDENT_SCORE_SEAM=1
if ! run_logged p9_target_opp "$P9_RUN_ROOT/p9_target_opp.log" bash "$P9_BUILD_SCRIPT"; then
    printf 'P9_RUNNER_FAIL stage=p9_target_opp result=%s\n' "$P9_RUN_ROOT"
    exit 1
fi
# The source package deliberately retains its build receipt at
# <package>/P9_PROVENANCE, then copies the same receipt beside the installed
# vendor OpAPI.  Select the one direct fresh package directory by its exact
# SHA-pair tag, then prove the vendor receipt next to the loaded OpAPI is an
# unmodified copy of that package receipt.
P9_PACKAGE_TAG="p9_exact_resident_v${P9_VLLM_EXPECTED_SHA:0:12}_k${P9_KVAF_EXPECTED_SHA:0:12}_*"
mapfile -t P9_TARGET_PACKAGES < <(
    find "$P9_TARGET_PACKAGE_PARENT" -mindepth 1 -maxdepth 1 -type d \
        -name "$P9_PACKAGE_TAG" -print | sort
)
[[ "${#P9_TARGET_PACKAGES[@]}" -eq 1 ]] || \
    runner_blocked "p9_package_root_ambiguous"
P9_TARGET_PACKAGE="${P9_TARGET_PACKAGES[0]}"
P9_PACKAGE_PROVENANCE="$P9_TARGET_PACKAGE/P9_PROVENANCE"
P9_VENDOR_ROOT="$P9_TARGET_PACKAGE/opp/vendors/customize"
P9_VENDOR_PROVENANCE="$P9_VENDOR_ROOT/P9_PROVENANCE"
[[ -f "$P9_PACKAGE_PROVENANCE" && -f "$P9_VENDOR_PROVENANCE" ]] || \
    runner_blocked "p9_provenance_missing"
cmp -s "$P9_PACKAGE_PROVENANCE" "$P9_VENDOR_PROVENANCE" || \
    runner_blocked "p9_provenance_copy_mismatch"
P9_TARGET_OPAPI="$P9_VENDOR_ROOT/op_api/lib/libcust_opapi.so"
[[ -s "$P9_TARGET_OPAPI" ]] || runner_blocked "p9_target_opapi_missing"
grep -Fxq "vllm_source_head=$P9_VLLM_EXPECTED_SHA" "$P9_VENDOR_PROVENANCE" || runner_blocked "p9_vllm_provenance_mismatch"
grep -Fxq "kvaf_source_head=$P9_KVAF_EXPECTED_SHA" "$P9_VENDOR_PROVENANCE" || runner_blocked "p9_kvaf_provenance_mismatch"
# A pre-23-input package predates the separate logical-slot / execution-row
# physical-block mapping and is not a valid P9 target artifact.  Do not let a
# merely existing libcust_opapi.so silently exercise an old ABI.
grep -Fxq 'p9_abi=official9_plus14_inputs_6_outputs' "$P9_VENDOR_PROVENANCE" || runner_blocked "p9_abi_provenance_mismatch"
grep -Fxq 'score_seam=active' "$P9_VENDOR_PROVENANCE" || runner_blocked "p9_score_seam_not_active"
grep -Fxq 'stage_b_finalizer=active' "$P9_VENDOR_PROVENANCE" || runner_blocked "p9_terminal_finalizer_not_active"
if grep -Eq '(^|=)(compile_disabled|disabled)(:|$)|score_seam=(0|disabled)' "$P9_VENDOR_PROVENANCE"; then
    runner_blocked "p9_legacy_compile_disabled_provenance"
fi
printf 'P9_OPP_PROVENANCE package=%s opapi=%s score_seam=active\n' "$P9_TARGET_PACKAGE" "$P9_TARGET_OPAPI"

export NANOVLLM_A5_EXACT_SPECULATIVE_RESIDENT_OPAPI_LIB="$P9_TARGET_OPAPI"
if ! run_logged p9_target_bridge "$P9_RUN_ROOT/p9_target_bridge.log" bash -c '
    cd "$1" && USE_NINJA=0 A5_C8_EXACT_SPECULATIVE_RESIDENT_BRIDGE_BUILD_JOBS="$4" "$2" "$3" \
      build_py --force --build-lib "$5" build_ext --force --build-lib "$5" --build-temp "$6"' \
    _ "$P9_TARGET_TORCH_ROOT" "$P9_PYTHON" "$P9_TARGET_SETUP" "$P9_BUILD_JOBS" "$P9_TARGET_BRIDGE" "$P9_TARGET_TEMP"; then
    printf 'P9_RUNNER_FAIL stage=p9_target_bridge result=%s\n' "$P9_RUN_ROOT"
    exit 1
fi
[[ -f "$P9_TARGET_BRIDGE/nanovllm_a5_c8_exact_speculative_resident_get/__init__.py" && \
   -s "$P9_TARGET_BRIDGE/nanovllm_a5_c8_exact_speculative_resident_get/_C$P9_SUFFIX" ]] || \
    runner_blocked "p9_target_bridge_artifact_missing"

# Recheck the exact clean source pair after all scoped builds.  There is no
# artifact discovery fallback: a source edit during compilation makes the
# whole run invalid rather than merely stale.
if ! is_clean_exact_repo vllm_target "$P9_VLLM_ROOT" "$P9_VLLM_EXPECTED_SHA" || \
   ! is_clean_exact_repo vllm_reference "$P9_REFERENCE_VLLM_ROOT" "$P9_VLLM_EXPECTED_SHA" || \
   ! is_clean_exact_repo kvaf "$P9_KVAF_ROOT" "$P9_KVAF_EXPECTED_SHA"; then
    runner_blocked "source_changed_during_build"
fi

export P9_REFERENCE_OPAPI P9_TARGET_OPAPI
export P9_REFERENCE_BRIDGE_ROOT="$P9_REFERENCE_BRIDGE"
export P9_TARGET_BRIDGE_ROOT="$P9_TARGET_BRIDGE"
export P9_KVAF_BATCH_GET_KERNEL P9_KVAF_BATCH_PUT_KERNEL
export P9_REQUIRE_REAL_HCCP=1
export P9_PHYSICAL_NPU_ID=7
export PYTHONPATH="$P9_KVAF_BUILD/python:$P9_REFERENCE_BRIDGE:$P9_TARGET_BRIDGE:$P9_OPS_ROOT/tests${PYTHONPATH:+:$PYTHONPATH}"
if ! run_logged runtime "$P9_RUN_ROOT/runtime.log" \
    timeout -k 30s "$P9_TEST_TIMEOUT" "$P9_PYTHON" "$P9_RUNTIME_TEST" --device npu:0 --case full; then
    printf 'P9_RUNNER_FAIL stage=runtime result=%s\n' "$P9_RUN_ROOT"
    exit 1
fi
printf 'P9_RUNNER_PASS physical=7 logical=0 kvaf_build=%s reference_opp=%s target_opapi=%s result=%s\n' \
    "$P9_KVAF_BUILD" "$P9_REFERENCE_OPP" "$P9_TARGET_OPAPI" "$P9_RUN_ROOT"
