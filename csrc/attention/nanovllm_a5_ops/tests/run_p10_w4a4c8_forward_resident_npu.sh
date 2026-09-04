#!/usr/bin/env bash
# Build and prove the one P10 product route on one explicitly mapped A5 NPU.
#
# P10 is deliberately a single assembly: ForwardPrepare, native W4A4C8
# score/receipt production, and the forward-resident Direct-HCCP executor are
# generated into one custom-OP vendor and resolved by one bridge.  Do not use
# a pre-existing custom OPP, a separately installed Stage-A/Stage-B DSO, or a
# whole vLLM-Ascend rebuild as a substitute for this runner.
#
# Invoke with `bash`, not `source`.  This matters operationally: every failure
# exits only this runner, never the user's interactive shell.

set -euo pipefail

this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

mode="${P10_RUNNER_MODE:-npu}"
runtime_case="${P10_RUNTIME_CASE:-acceptance}"
build_jobs="${P10_BUILD_JOBS:-8}"

fail() {
    printf 'P10_RUNNER_BLOCKED reason=%s\n' "$1" >&2
    exit 2
}

require_env() {
    local name="$1"
    [[ -n "${!name:-}" ]] || fail "environment_missing:${name}"
}

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

require_exact_clean_source() {
    local label="$1"
    local root="$2"
    local expected="$3"
    local head dirty
    [[ -e "${root}/.git" ]] || fail "${label}_git_worktree_missing:path=${root}"
    head="$(git -C "${root}" rev-parse HEAD 2>/dev/null || true)"
    [[ "${head}" == "${expected}" ]] ||
        fail "${label}_sha_mismatch:actual=${head}:expected=${expected}"
    dirty="$(git -C "${root}" status --porcelain --untracked-files=no 2>/dev/null || true)"
    [[ -z "${dirty}" ]] || fail "${label}_tracked_changes_present:path=${root}"
    printf 'P10_SOURCE label=%s head=%s expected=%s dirty=0\n' \
        "${label}" "${head}" "${expected}"
}

case "${mode}" in
    static|npu) ;;
    *) fail "invalid_runner_mode:${mode}" ;;
esac
case "${runtime_case}" in
    acceptance|benchmark|graph_benchmark|graph_compatible_benchmark) ;;
    *) fail "invalid_runtime_case:${runtime_case}" ;;
esac
[[ "${build_jobs}" =~ ^[1-9][0-9]*$ ]] || fail "invalid_build_jobs:${build_jobs}"
[[ -z "${SOC_VERSION:-}" ]] || fail "legacy_SOC_VERSION_must_be_unset"

for required in \
    P10_VLLM_ROOT \
    P10_KVAF_ROOT \
    P10_VLLM_EXPECTED_SHA \
    P10_KVAF_EXPECTED_SHA \
    P10_PYTHON; do
    require_env "${required}"
done

vllm_root="${P10_VLLM_ROOT}"
kvaf_root="${P10_KVAF_ROOT}"
python_bin="${P10_PYTHON}"
expected_test_dir="${vllm_root}/csrc/attention/nanovllm_a5_ops/tests"
[[ "${this_dir}" == "${expected_test_dir}" ]] ||
    fail "runner_not_from_exact_vllm_source:path=${this_dir}:expected=${expected_test_dir}"
ops_root="$(cd "${expected_test_dir}/.." && pwd -P)"
target_root="${ops_root}/a5_dsa_c8_exact_speculative_resident"
assembly="${target_root}/build_p10_target_assembly.sh"
bridge_root="${target_root}/p10_forward_bridge"
bridge_setup="${bridge_root}/torch_extension/p10_forward_bridge_setup.py"
runtime_fixture="${expected_test_dir}/test_p10_w4a4c8_forward_resident_runtime.py"
[[ -x "${python_bin}" ]] || fail "python_not_executable:path=${python_bin}"
[[ -r "${assembly}" ]] || fail "assembly_script_missing:path=${assembly}"
[[ -r "${bridge_setup}" ]] || fail "bridge_setup_missing:path=${bridge_setup}"
[[ -r "${runtime_fixture}" ]] || fail "runtime_fixture_missing:path=${runtime_fixture}"

require_exact_clean_source vllm "${vllm_root}" "${P10_VLLM_EXPECTED_SHA}"
require_exact_clean_source kvaf "${kvaf_root}" "${P10_KVAF_EXPECTED_SHA}"

static_tests=(
    "${this_dir}/test_p10_w4a4c8_source_contract.py"
    "${this_dir}/test_p10_w4a4c8_score_receipt_source_contract.py"
    "${this_dir}/test_p10_stage_b_source_contract.py"
    "${this_dir}/test_p10_forward_bridge_contract.py"
    "${this_dir}/test_p10_build_assembly_contract.py"
    "${this_dir}/test_p10_product_runner_contract.py"
)
for test_file in "${static_tests[@]}"; do
    [[ -r "${test_file}" ]] || fail "static_contract_missing:path=${test_file}"
done

printf 'P10_RUNNER_CONFIG mode=%s runtime_case=%s product=combined_forward_prepare_stage_a_stage_b physical=%s logical=%s build_jobs=%s\n' \
    "${mode}" "${runtime_case}" \
    "${P10_PHYSICAL_NPU_ID:-<not-required-in-static>}" \
    "${P10_LOGICAL_NPU_ID:-0}" "${build_jobs}"

( cd "${vllm_root}" && "${python_bin}" -m unittest \
    csrc.attention.nanovllm_a5_ops.tests.test_p10_w4a4c8_source_contract \
    csrc.attention.nanovllm_a5_ops.tests.test_p10_w4a4c8_score_receipt_source_contract \
    csrc.attention.nanovllm_a5_ops.tests.test_p10_stage_b_source_contract \
    csrc.attention.nanovllm_a5_ops.tests.test_p10_forward_bridge_contract \
    csrc.attention.nanovllm_a5_ops.tests.test_p10_build_assembly_contract \
    csrc.attention.nanovllm_a5_ops.tests.test_p10_product_runner_contract )
printf 'P10_RUNNER_STEP label=static_contracts status=0\n'

if [[ "${mode}" == "static" ]]; then
    printf 'P10_RUNNER_PASS mode=static product_route=combined_source_contracts\n'
    exit 0
fi

require_env P10_PHYSICAL_NPU_ID
physical_npu="${P10_PHYSICAL_NPU_ID}"
logical_npu="${P10_LOGICAL_NPU_ID:-0}"
[[ "${logical_npu}" == "0" ]] || fail "logical_npu_must_be_zero:value=${logical_npu}"
[[ "${physical_npu}" =~ ^[0-9]+$ ]] || fail "invalid_physical_npu:${physical_npu}"

ascend_home="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}"
[[ -r "${ascend_home}/bin/setenv.bash" ]] ||
    fail "cann_setenv_missing:path=${ascend_home}/bin/setenv.bash"
set +u
. "${ascend_home}/bin/setenv.bash"
set -u
# Never assign SOC_VERSION for A5.  Shell `unset` is deliberately written
# with literal variable names, so it cannot accidentally receive a path.
unset SOC_VERSION
unset ASCEND_VISIBLE_DEVICES
export A5_SOC_VERSION=ascend950
export ASCEND_RT_VISIBLE_DEVICES="${physical_npu}"
export NPU_ID=0
export DEVICE_ID=0
# P10 is a direct caller-stream device pipeline.  These are asserted before
# the first torch_npu import and again by the bridge; neither a task-queue
# fallback nor host launch blocking is a valid product execution mode.
export TASK_QUEUE_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0
printf 'P10_DIRECT_DISPATCH_CONFIG pass=1 task_queue_enable=%s launch_blocking=%s physical=%s logical=0 phase=pre_torch_npu_import\n' \
    "${TASK_QUEUE_ENABLE}" "${ASCEND_LAUNCH_BLOCKING}" "${physical_npu}"

"${python_bin}" - <<'PY'
import os
import torch
import torch_npu

torch.npu.set_device(0)
value = torch.ones(1, device="npu:0").cpu().item()
print(
    "P10_NPU_PROBE "
    f"physical={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} logical=0 "
    f"device={torch.npu.get_device_name(0)!r} value={value}"
)
PY
printf 'P10_RUNNER_STEP label=npu_probe status=0\n'

# The benchmark's immutable W4A4C8-input gate hashes raw FP8 storage.  NumPy
# cannot materialize Float8_e4m3fn directly, so prove its uint8 dtype view
# works before compiling any P10 artifact.  This is a capability preflight,
# not a numerical fallback or a conversion of the product Q/K tensors.
"${python_bin}" - <<'PY'
import hashlib
import torch

value = torch.tensor([1.0, -2.0], dtype=torch.float8_e4m3fn).contiguous()
raw = value.reshape(-1).view(torch.uint8)
if raw.dtype != torch.uint8 or raw.numel() != value.numel():
    raise RuntimeError(
        f"invalid_fp8_raw_view:dtype={raw.dtype}:raw_numel={raw.numel()}:"
        f"fp8_numel={value.numel()}"
    )
digest = hashlib.sha256(raw.numpy().tobytes()).hexdigest()[:16]
print(
    "P10_FLOAT8_RAW_HASH_PROBE "
    f"pass=1 fp8_dtype={value.dtype} raw_dtype={raw.dtype} bytes={raw.numel()} "
    f"sha16={digest}"
)
PY
printf 'P10_RUNNER_STEP label=fp8_raw_hash_probe status=0\n'

# CANN's third-party inputs are Git-ignored build dependencies.  A fresh
# checkout therefore must not be required to contain them under
# ``${vllm_root}/csrc/third_party``.  Keep the reusable cache beside the
# checkout by default, bootstrap only missing required inputs before any P10
# build, and never allow it to dirty or alter the exact VLLM worktree used for
# provenance.
cann_cache_default="$(dirname "${vllm_root}")/.cann-third-party-$(basename "${ascend_home}")"
cann_cache="${P10_CANN_3RD_LIB_PATH:-${cann_cache_default}}"
case "${cann_cache}" in
    "${vllm_root}"|"${vllm_root}"/*)
        fail "cann_third_party_cache_must_be_external:path=${cann_cache}"
        ;;
esac
mkdir -p "${cann_cache}" || fail "cann_third_party_cache_create_failed:path=${cann_cache}"
cann_cache="$(cd "${cann_cache}" && pwd -P)"
cache_bootstrap="${ops_root}/tools/bootstrap_cann_third_party_cache.sh"
[[ -r "${cache_bootstrap}" ]] || fail "cann_cache_bootstrap_missing:path=${cache_bootstrap}"
cache_complete() {
    [[ -r "${cann_cache}/makeself/makeself.sh" ]] &&
        [[ -r "${cann_cache}/makeself/makeself-header.sh" ]] &&
        [[ -r "${cann_cache}/json/include/nlohmann/json.hpp" ]] &&
        { [[ -r "${cann_cache}/abseil-cpp/abseil-cpp-20230802.1.tar.gz" ]] ||
          [[ -r "${cann_cache}/pkg/abseil-cpp-20230802.1.tar.gz" ]]; } &&
        { [[ -r "${cann_cache}/protobuf/protobuf-all-25.1.tar.gz" ]] ||
          [[ -r "${cann_cache}/pkg/protobuf-25.1.tar.gz" ]]; }
}
if ! cache_complete; then
    case "${P10_CANN_3RD_BOOTSTRAP:-1}" in
        1|true|TRUE|yes|YES)
            P10_CANN_3RD_LIB_PATH="${cann_cache}" bash "${cache_bootstrap}" \
                "${cann_cache}"
            ;;
        *)
            fail "cann_third_party_cache_incomplete:path=${cann_cache}:set=P10_CANN_3RD_BOOTSTRAP=1"
            ;;
    esac
fi
cache_complete || fail "cann_third_party_cache_incomplete_after_bootstrap:path=${cann_cache}"
printf 'P10_CANN_CACHE path=%s source_tree=0 complete=1 bootstrap=%s\n' \
    "${cann_cache}" "${P10_CANN_3RD_BOOTSTRAP:-1}"

# Keep all generated P10 artifacts under the exact source tree and a
# source-SHA keyed package root.  Do not permit an inherited interactive-shell
# override to redirect this run into a package made for a prior revision: that
# would either reuse stale code or waste a full OPP build before failing its
# manifest check.
package_parent="${ops_root}/build-p10-product"
package_root="${package_parent}/p10_v${P10_VLLM_EXPECTED_SHA:0:12}_k${P10_KVAF_EXPECTED_SHA:0:12}"
if [[ -n "${P10_PACKAGE_PARENT:-}" && "${P10_PACKAGE_PARENT}" != "${package_parent}" ]]; then
    printf 'P10_ASSEMBLY_OVERRIDE_IGNORED name=P10_PACKAGE_PARENT supplied=%s canonical=%s\n' \
        "${P10_PACKAGE_PARENT}" "${package_parent}"
fi
if [[ -n "${P10_PRODUCT_PACKAGE_ROOT:-}" && "${P10_PRODUCT_PACKAGE_ROOT}" != "${package_root}" ]]; then
    printf 'P10_ASSEMBLY_OVERRIDE_IGNORED name=P10_PRODUCT_PACKAGE_ROOT supplied=%s canonical=%s\n' \
        "${P10_PRODUCT_PACKAGE_ROOT}" "${package_root}"
fi
mkdir -p "${package_parent}"
case "${package_root}" in
    "${package_parent}"/*) ;;
    *) fail "unsafe_product_package_root:path=${package_root}:parent=${package_parent}" ;;
esac

P10_BUILD_MODE=build \
P10_BUILD_VLLM_ROOT="${vllm_root}" \
P10_BUILD_KVAF_ROOT="${kvaf_root}" \
P10_BUILD_EXPECTED_VLLM_SHA="${P10_VLLM_EXPECTED_SHA}" \
P10_BUILD_EXPECTED_KVAF_SHA="${P10_KVAF_EXPECTED_SHA}" \
P10_BUILD_CANN_3RD_LIB_PATH="${cann_cache}" \
P10_BUILD_PYTHON="${python_bin}" \
P10_BUILD_PACKAGE_PARENT="${package_parent}" \
P10_BUILD_PACKAGE_ROOT="${package_root}" \
P10_BUILD_JOBS="${build_jobs}" \
P10_ASSEMBLY_INVALID_PACKAGE_POLICY=quarantine \
bash "${assembly}"
printf 'P10_RUNNER_STEP label=combined_product_opp status=0 package=%s\n' "${package_root}"

assembly_manifest="${package_root}/P10_ASSEMBLY_PROVENANCE"
product_manifest="${package_root}/product/P10_PRODUCT_PROVENANCE"
[[ -s "${assembly_manifest}" && -s "${product_manifest}" ]] ||
    fail "combined_product_provenance_missing"
product_opapi_count="$(find "${package_root}/product/opp" -type f -path '*/op_api/lib/libcust_opapi.so' -print 2>/dev/null | wc -l | tr -d ' ')"
[[ "${product_opapi_count}" == "1" ]] ||
    fail "combined_product_opapi_count_invalid:count=${product_opapi_count}"
product_opapi="$(find "${package_root}/product/opp" -type f -path '*/op_api/lib/libcust_opapi.so' -print -quit)"
product_vendor="$(cd "$(dirname "$(dirname "$(dirname "${product_opapi}")")")" && pwd -P)"
for line in \
    'kind=p10_combined_prepare_stage_a_stage_b_product_opp' \
    "vllm_source_head=${P10_VLLM_EXPECTED_SHA}" \
    "kvaf_source_head=${P10_KVAF_EXPECTED_SHA}" \
    'operators=A5P10ForwardPrepare,A5P10W4A4C8ScoreReceipt,A5P10ScoreReceiptForwardResident' \
    "opapi=${product_opapi}"; do
    grep -Fqx "${line}" "${product_manifest}" ||
        fail "combined_product_provenance_mismatch:line=${line}"
done
printf 'P10_OPP_PROVENANCE vendor=%s opapi=%s sha256=%s\n' \
    "${product_vendor}" "${product_opapi}" "$(hash_file "${product_opapi}")"

# Build the bridge into this exact product package.  It is forced deliberately:
# the forward bridge source is part of the assembly digest but Python extension
# timestamps are not trusted as provenance.
bridge_python="${package_root}/bridge/python"
bridge_temp="${package_root}/bridge/temp"
mkdir -p "${bridge_python}" "${bridge_temp}"
(
    cd "${bridge_root}/torch_extension"
    # ``build_ext`` only places the native module in --build-lib.  The P10
    # package itself also owns the graph adapter and its __init__.py; without
    # this explicit step Python sees a namespace package (__file__ is None)
    # and provenance cannot establish that the imported bridge is this exact
    # SHA-keyed product package.
    "${python_bin}" p10_forward_bridge_setup.py build_py \
        --build-lib "${bridge_python}"
    USE_NINJA=0 \
    NANOVLLM_A5_P10_FORWARD_BRIDGE_BUILD_JOBS="${build_jobs}" \
    "${python_bin}" p10_forward_bridge_setup.py build_ext --force \
        --build-lib "${bridge_python}" --build-temp "${bridge_temp}"
)
bridge_package="${bridge_python}/nanovllm_p10_forward_bridge"
bridge_init="${bridge_package}/__init__.py"
bridge_graph="${bridge_package}/p10_forward_graph.py"
[[ -s "${bridge_init}" && -s "${bridge_graph}" ]] ||
    fail "bridge_python_package_incomplete:init=$([[ -s "${bridge_init}" ]] && echo 1 || echo 0):graph=$([[ -s "${bridge_graph}" ]] && echo 1 || echo 0)"
bridge_count="$(find "${bridge_package}" -maxdepth 1 -type f -name '_C*.so' -print 2>/dev/null | wc -l | tr -d ' ')"
[[ "${bridge_count}" == "1" ]] || fail "bridge_artifact_count_invalid:count=${bridge_count}"
bridge_so="$(find "${bridge_package}" -maxdepth 1 -type f -name '_C*.so' -print -quit)"
printf 'P10_BRIDGE_PROVENANCE package=%s init=%s graph=%s bridge=%s sha256=%s\n' \
    "${bridge_package}" "${bridge_init}" "${bridge_graph}" \
    "${bridge_so}" "$(hash_file "${bridge_so}")"
printf 'P10_RUNNER_STEP label=forward_bridge status=0\n'

# KVAF is built from its own source tree and has no VLLM header dependency.
# Key its reusable build directory only by the inputs that can affect its
# binary: KVAF SHA, CANN distribution, Python ABI, and the fixed A5 CMake
# flags below.  The combined P10 OPP/bridge remains keyed by the full source
# pair.  This avoids rebuilding KVAF solely because a P10 C++ source edit
# requires a fresh OPP package, without opening a stale-artifact path.
python_abi="$("${python_bin}" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
[[ "${python_abi}" =~ ^[0-9]+$ ]] || fail "python_abi_probe_invalid:value=${python_abi}"
cann_id="$(basename "${ascend_home}")"
kvaf_build="${kvaf_root}/build-p10-k${P10_KVAF_EXPECTED_SHA:0:12}-c${cann_id}-py${python_abi}-a5"
if [[ -n "${P10_KVAF_BUILD:-}" && "${P10_KVAF_BUILD}" != "${kvaf_build}" ]]; then
    printf 'P10_ASSEMBLY_OVERRIDE_IGNORED name=P10_KVAF_BUILD supplied=%s canonical=%s\n' \
        "${P10_KVAF_BUILD}" "${kvaf_build}"
fi
pybind_dir="$("${python_bin}" -m pybind11 --cmakedir 2>/dev/null || true)"
[[ -n "${pybind_dir}" && -d "${pybind_dir}" ]] || fail "pybind11_cmakedir_unavailable"
cmake -S "${kvaf_root}" -B "${kvaf_build}" \
    -DBUILD_TESTING=OFF \
    -DKVAF_BUILD_PYTHON=ON \
    -DKVAF_ENABLE_AIV_BATCH_GET=ON \
    -DKVAF_ENABLE_AIV_BATCH_PUT=ON \
    -DPython_EXECUTABLE="${python_bin}" \
    -Dpybind11_DIR="${pybind_dir}" \
    -DASCEND_CANN_PACKAGE_PATH="${ascend_home}"
cmake --build "${kvaf_build}" -j "${build_jobs}"
get_kernel="${kvaf_build}/bin/kvaf_execute_batch_get_kernels.o"
put_kernel="${kvaf_build}/bin/kvaf_execute_batch_put_kernels.o"
[[ -s "${get_kernel}" && -s "${put_kernel}" ]] ||
    fail "kvaf_kernel_artifacts_missing:build=${kvaf_build}"
kvaf_extension_count="$(find "${kvaf_build}/python/kvaf" -maxdepth 1 -type f -name '_kvaf*.so' -print 2>/dev/null | wc -l | tr -d ' ')"
[[ "${kvaf_extension_count}" == "1" ]] ||
    fail "kvaf_python_artifact_count_invalid:count=${kvaf_extension_count}"
printf 'P10_KVAF_PROVENANCE build=%s get=%s put=%s extension_count=%s\n' \
    "${kvaf_build}" "${get_kernel}" "${put_kernel}" "${kvaf_extension_count}"
printf 'P10_RUNNER_STEP label=kvaf_incremental status=0\n'

runtime_provenance="${package_root}/P10_RUNTIME_PROVENANCE"
cat > "${runtime_provenance}" <<EOF
format=1
kind=p10_w4a4c8_forward_resident_runtime
vllm_source_root=${vllm_root}
vllm_source_head=${P10_VLLM_EXPECTED_SHA}
kvaf_source_root=${kvaf_root}
kvaf_source_head=${P10_KVAF_EXPECTED_SHA}
product_vendor=${product_vendor}
product_opapi=${product_opapi}
product_opapi_sha256=$(hash_file "${product_opapi}")
forward_bridge=${bridge_so}
forward_bridge_sha256=$(hash_file "${bridge_so}")
kvaf_build=${kvaf_build}
kvaf_get_kernel=${get_kernel}
kvaf_put_kernel=${put_kernel}
physical_npu=${physical_npu}
logical_npu=0
EOF

export NANOVLLM_A5_P10_PRODUCT_OPAPI_LIB="${product_opapi}"
export NANOVLLM_A5_P10_PRODUCT_OPP_VENDOR="${product_vendor}"
export ASCEND_CUSTOM_OPP_PATH="${product_vendor}"
export PYTHONPATH="${bridge_python}:${kvaf_build}/python:${vllm_root}${PYTHONPATH:+:${PYTHONPATH}}"
export P10_KVAF_BATCH_GET_KERNEL="${get_kernel}"
export P10_KVAF_BATCH_PUT_KERNEL="${put_kernel}"
export P10_KVAF_BUILD="${kvaf_build}"
export P10_PRODUCT_BRIDGE_PYTHON="${bridge_python}"
export P10_PRODUCT_RUNTIME_PROVENANCE="${runtime_provenance}"
export P10_REQUIRE_REAL_HCCP=1
export P10_PHYSICAL_NPU_ID="${physical_npu}"
export P10_LOGICAL_NPU_ID=0
export P10_VLLM_EXPECTED_SHA
export P10_KVAF_EXPECTED_SHA

P10_RUNTIME_CASE="${runtime_case}" "${python_bin}" "${runtime_fixture}"
printf 'P10_RUNNER_STEP label=forward_resident_runtime case=%s status=0\n' "${runtime_case}"
printf 'P10_RUNNER_PASS mode=npu runtime_case=%s product_route=combined_w4a4c8_prepare_score_receipt_resident_qsfa\n' \
    "${runtime_case}"
