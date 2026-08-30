#!/usr/bin/env bash
# Build and verify P7.2a: a clean C8 score-window publisher, with no P6/P7
# executor closure. Run this *inside the A5 Docker container*. It never pulls
# source, modifies the selected worktree, runs KVAF, or starts HCCP.

set +u
set -o pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_ROOT="$(cd "${THIS_DIR}/.." && pwd)"
VLLM_KIP="${VLLM_KIP:-$(cd "${OPS_ROOT}/../../.." && pwd)}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}"
P7_PYTHON="${P7_PYTHON:-$(command -v python3 2>/dev/null || true)}"
P7_BUILD_JOBS="${P7_BUILD_JOBS:-16}"
P7_PHYSICAL_NPU_ID="${P7_PHYSICAL_NPU_ID:-7}"
P7_RESULT="${P7_RESULT:-${OPS_ROOT}/build/p7_clean_publisher_$(date +%Y%m%d_%H%M%S)}"

P7_REFERENCE_OPP_ROOT="${P7_RESULT}/official_c8_opp"
P7_TARGET_BUILD_ROOT="${P7_RESULT}/derived_publisher_generated"
P7_TARGET_OPP_ROOT="${OPS_ROOT}/_custom_opp_p7_clean_publisher_$(basename "${P7_RESULT}")"
P7_REFERENCE_BRIDGE_ROOT="${OPS_ROOT}/a5_official_c8_reference/torch_extension"
P7_TARGET_BRIDGE_ROOT="${OPS_ROOT}/a5_streaming_quant_li_get/torch_extension"

P7_REFERENCE_OPP_RC=99
P7_TARGET_OPP_RC=99
P7_REFERENCE_BRIDGE_RC=99
P7_TARGET_BRIDGE_RC=99
P7_TEST_RC=99

P7_SOURCE_HEAD="$(git -C "${VLLM_KIP}" rev-parse HEAD 2>/dev/null || true)"
P7_EXPECTED_SOURCE_HEAD="${P7_EXPECTED_SOURCE_HEAD:-${P7_SOURCE_HEAD}}"
P7_DIRTY="$(git -C "${VLLM_KIP}" status --porcelain --untracked-files=no 2>/dev/null || true)"

case "${P7_RESULT}" in
    "${OPS_ROOT}"/build/*) ;;
    *)
        printf 'P7_CLEAN_PUBLISHER_BLOCKED reason=result_must_be_under_ops_build result=%s ops_root=%s\n' \
            "${P7_RESULT}" "${OPS_ROOT}" >&2
        exit 2
        ;;
esac

if [[ ! -e "${VLLM_KIP}/.git" || "${P7_SOURCE_HEAD}" != "${P7_EXPECTED_SOURCE_HEAD}" || -n "${P7_DIRTY}" || ! -x "${P7_PYTHON}" || ! -f "${ASCEND_HOME_PATH}/bin/setenv.bash" || ! -f "${OPS_ROOT}/build_nanovllm_ops.sh" || ! -f "${OPS_ROOT}/a5_official_c8_reference/build_reference_opp.sh" || -e "${P7_RESULT}" || ! "${P7_BUILD_JOBS}" =~ ^[1-9][0-9]*$ || ! "${P7_PHYSICAL_NPU_ID}" =~ ^[0-9]+$ ]]; then
    printf 'P7_CLEAN_PUBLISHER_BLOCKED root=%s source=%s expected=%s dirty=%s python=%s result_exists=%s jobs=%s physical_npu=%s\n' "${VLLM_KIP}" "${P7_SOURCE_HEAD:-missing}" "${P7_EXPECTED_SOURCE_HEAD:-missing}" "$([[ -n "${P7_DIRTY}" ]] && echo 1 || echo 0)" "${P7_PYTHON:-missing}" "$([[ -e "${P7_RESULT}" ]] && echo 1 || echo 0)" "${P7_BUILD_JOBS}" "${P7_PHYSICAL_NPU_ID}" >&2
    exit 2
fi

mkdir -p "${P7_RESULT}"
printf 'P7_CLEAN_PUBLISHER_PROFILE source=%s expected=%s physical_npu=%s logical_npu=0 publication=publisher inline=off kvaf_post_cq=absent result=%s\n' "${P7_SOURCE_HEAD}" "${P7_EXPECTED_SOURCE_HEAD}" "${P7_PHYSICAL_NPU_ID}" "${P7_RESULT}"

source "${ASCEND_HOME_PATH}/bin/setenv.bash"
set +u

(
    export A5_OFFICIAL_C8_SOURCE_ROOT="${VLLM_KIP}"
    export A5_OFFICIAL_C8_REFERENCE_ROOT="${P7_REFERENCE_OPP_ROOT}"
    export A5_OFFICIAL_C8_EXPECTED_SOURCE_HEAD="${P7_EXPECTED_SOURCE_HEAD}"
    export A5_OFFICIAL_C8_REFERENCE_BUILD_JOBS="${P7_BUILD_JOBS}"
    bash "${OPS_ROOT}/a5_official_c8_reference/build_reference_opp.sh"
) 2>&1 | tee "${P7_RESULT}/reference_opp.log"
P7_REFERENCE_OPP_RC=${PIPESTATUS[0]}

if [[ "${P7_REFERENCE_OPP_RC}" -eq 0 ]]; then
    (
        unset NANOVLLM_A5_STREAMING_LI_GET_HOOK_MODE
        unset NANOVLLM_A5_STREAMING_LI_GET_POST_CLOSURE_MODE
        unset NANOVLLM_A5_STREAMING_LI_GET_POST_SCHEDULER
        export NANOVLLM_A5_OPS_ONLY=A5StreamingQuantLiGet
        export NANOVLLM_A5_OPS_BUILD_ROOT="${P7_TARGET_BUILD_ROOT}"
        export NANOVLLM_A5_LOCAL_OPP="${P7_TARGET_OPP_ROOT}"
        export NANOVLLM_A5_OPS_PYTHON="${P7_PYTHON}"
        export NANOVLLM_A5_OPS_BUILD_JOBS="${P7_BUILD_JOBS}"
        export NANOVLLM_A5_SKIP_TORCH_EXTENSION=1
        export NANOVLLM_A5_STREAMING_LI_GET_PUBLISH_MODE=publisher
        export NANOVLLM_A5_STREAMING_LI_GET_INLINE_MODE=off
        export A5_SOC_VERSION=ascend950
        bash "${OPS_ROOT}/build_nanovllm_ops.sh"
    ) 2>&1 | tee "${P7_RESULT}/target_opp.log"
    P7_TARGET_OPP_RC=${PIPESTATUS[0]}
fi

if [[ "${P7_REFERENCE_OPP_RC}" -eq 0 && "${P7_TARGET_OPP_RC}" -eq 0 ]]; then
    (
        cd "${P7_REFERENCE_BRIDGE_ROOT}" || exit 2
        export USE_NINJA=0
        export A5_OFFICIAL_C8_REFERENCE_BRIDGE_BUILD_JOBS="${P7_BUILD_JOBS}"
        "${P7_PYTHON}" official_c8_reference_setup.py build_ext --inplace --force
    ) 2>&1 | tee "${P7_RESULT}/reference_bridge.log"
    P7_REFERENCE_BRIDGE_RC=${PIPESTATUS[0]}

    if [[ "${P7_REFERENCE_BRIDGE_RC}" -eq 0 ]]; then
        (
            cd "${P7_TARGET_BRIDGE_ROOT}" || exit 2
            export USE_NINJA=0
            export NANOVLLM_A5_STREAMING_LI_BUILD_JOBS="${P7_BUILD_JOBS}"
            "${P7_PYTHON}" streaming_quant_li_get_setup.py build_ext --inplace --force
        ) 2>&1 | tee "${P7_RESULT}/target_bridge.log"
        P7_TARGET_BRIDGE_RC=${PIPESTATUS[0]}
    fi
fi

if [[ "${P7_REFERENCE_OPP_RC}" -eq 0 && "${P7_TARGET_OPP_RC}" -eq 0 && "${P7_REFERENCE_BRIDGE_RC}" -eq 0 && "${P7_TARGET_BRIDGE_RC}" -eq 0 ]]; then
    (
        export ASCEND_RT_VISIBLE_DEVICES="${P7_PHYSICAL_NPU_ID}"
        unset ASCEND_VISIBLE_DEVICES
        export P3_NPU_ID=0
        export P3_PHYSICAL_NPU_ID="${P7_PHYSICAL_NPU_ID}"
        export P3_PUBLICATION_MODE=publisher
        export P3_INLINE_MODE=off
        unset P3_POST_CLOSURE_MODE
        unset P3_POST_SCHEDULER_MODE
        export NANOVLLM_A5_REFERENCE_C8_OPP="${P7_REFERENCE_OPP_ROOT}"
        export NANOVLLM_A5_STREAMING_LI_OPP="${P7_TARGET_OPP_ROOT}"
        export PYTHONPATH="${P7_REFERENCE_BRIDGE_ROOT}:${P7_TARGET_BRIDGE_ROOT}:${THIS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
        source "${THIS_DIR}/resolve_streaming_li_get_artifacts.sh" "${P7_REFERENCE_OPP_ROOT}" "${P7_TARGET_OPP_ROOT}" "${VLLM_KIP}"
        P7_ARTIFACT_ENV_RC=$?
        if [[ "${P7_ARTIFACT_ENV_RC}" -ne 0 ]]; then
            exit "${P7_ARTIFACT_ENV_RC}"
        fi
        timeout 240s "${P7_PYTHON}" "${THIS_DIR}/test_p7_clean_publisher_equivalence.py"
    ) 2>&1 | tee "${P7_RESULT}/equivalence.log"
    P7_TEST_RC=${PIPESTATUS[0]}
fi

printf 'P7_CLEAN_PUBLISHER_SUMMARY reference_opp=%s target_opp=%s reference_bridge=%s target_bridge=%s test=%s result=%s target_opp_root=%s\n' "${P7_REFERENCE_OPP_RC}" "${P7_TARGET_OPP_RC}" "${P7_REFERENCE_BRIDGE_RC}" "${P7_TARGET_BRIDGE_RC}" "${P7_TEST_RC}" "${P7_RESULT}" "${P7_TARGET_OPP_ROOT}"
if [[ "${P7_TEST_RC}" -eq 0 ]]; then
    grep -E '^(P7_CLEAN_PUBLISHER_(CONFIG|RESULT|PASS|FAIL|SUMMARY)|P3_(REFERENCE_PROVENANCE|TARGET_PROVENANCE|PUBLICATION_PROOF))' "${P7_RESULT}/equivalence.log" || true
fi
exit "${P7_TEST_RC}"
