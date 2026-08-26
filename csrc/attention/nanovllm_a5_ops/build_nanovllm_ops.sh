#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${NANOVLLM_A5_OPS_BUILD_ROOT:-${ROOT}/build}"
OP_DEV="${BUILD_ROOT}/op_dev_staging"
GENERATED="${BUILD_ROOT}/custom_op"
LOCAL_OPP="${NANOVLLM_A5_LOCAL_OPP:-${ROOT}/_custom_opp}"
TORCH_EXTENSION="${ROOT}/torch_extension"

PYTHON_BIN="${NANOVLLM_A5_OPS_PYTHON:-python3}"
BUILD_JOBS="${NANOVLLM_A5_OPS_BUILD_JOBS:-$(nproc)}"
SKIP_TORCH_EXTENSION="${NANOVLLM_A5_SKIP_TORCH_EXTENSION:-0}"
A5_SOC_VERSION_RAW="${A5_SOC_VERSION:-${SOC_VERSION:-ascend950}}"
case "${A5_SOC_VERSION_RAW,,}" in
    ascend950 | ascend950pr* | ascend950dt*)
        A5_SOC_VERSION="ascend950"
        ;;
    *)
        echo "[nanovllm_a5_ops] ERROR: only Ascend 950 is supported; got A5_SOC_VERSION=${A5_SOC_VERSION_RAW}." >&2
        exit 2
        ;;
esac

if ! command -v msopgen >/dev/null 2>&1; then
    echo "[nanovllm_a5_ops] ERROR: msopgen is unavailable; source the CANN 9.1 environment first." >&2
    exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[nanovllm_a5_ops] ERROR: Python is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi

case "${GENERATED}" in
    "${ROOT}"/*) ;;
    *)
        echo "[nanovllm_a5_ops] ERROR: unsafe generated-project path: ${GENERATED}" >&2
        exit 2
        ;;
esac
case "${BUILD_ROOT}" in
    "${ROOT}"/build|"${ROOT}"/build/*) ;;
    *)
        echo "[nanovllm_a5_ops] ERROR: build root must stay below ${ROOT}/build: ${BUILD_ROOT}." >&2
        exit 2
        ;;
esac
case "${LOCAL_OPP}" in
    "${ROOT}"/*) ;;
    *)
        echo "[nanovllm_a5_ops] ERROR: unsafe local OPP path: ${LOCAL_OPP}" >&2
        exit 2
        ;;
esac

echo "[nanovllm_a5_ops] root: ${ROOT}"
echo "[nanovllm_a5_ops] Python: ${PYTHON_BIN}"
echo "[nanovllm_a5_ops] A5 target: ${A5_SOC_VERSION}"
echo "[nanovllm_a5_ops] skip torch extension: ${SKIP_TORCH_EXTENSION}"
if [[ "${A5_SOC_VERSION_RAW,,}" != "${A5_SOC_VERSION}" ]]; then
    echo "[nanovllm_a5_ops] normalized product name ${A5_SOC_VERSION_RAW} -> ${A5_SOC_VERSION}"
fi
echo "[nanovllm_a5_ops] jobs: ${BUILD_JOBS}"

ALL_OP_NAMES=(
    LightningIndexerDecodeUpdateA5
    A5LiduCacheUpdate
    A5KvcacheScatterCopy
    A5PackedKvcacheScatterCopy
    A5KvafBatchGetPrepare
    A5SparseAndTailAttention
    A5FullBlockDump
    A5PackedFullBlockDump
    A5ExactScanSpeculativePostProbe
    A5ExactScanSpeculativeGet
)
# The source-derived streaming LI and reconcile op are deliberately not in
# ALL_OP_NAMES: both are standalone P3 probes, never default product routes.
SUPPORTED_OP_NAMES=(
    "${ALL_OP_NAMES[@]}"
    A5StreamingQuantLiGet
    A5ExactReconcilePrivateView
)

# A prototype operator must be buildable without replacing the normal custom
# OPP package or compiling the full vLLM-Ascend extension.  A comma-separated
# allow-list keeps generated artifacts in caller-selected, isolated paths.
OP_NAMES=("${ALL_OP_NAMES[@]}")
if [[ -n "${NANOVLLM_A5_OPS_ONLY:-}" ]]; then
    IFS=',' read -r -a REQUESTED_OP_NAMES <<< "${NANOVLLM_A5_OPS_ONLY}"
    OP_NAMES=()
    for requested_name in "${REQUESTED_OP_NAMES[@]}"; do
        found=0
        for known_name in "${SUPPORTED_OP_NAMES[@]}"; do
            if [[ "${requested_name}" == "${known_name}" ]]; then
                found=1
                break
            fi
        done
        if [[ "${found}" != "1" ]]; then
            echo "[nanovllm_a5_ops] ERROR: unknown operator in NANOVLLM_A5_OPS_ONLY: ${requested_name}." >&2
            exit 2
        fi
        OP_NAMES+=("${requested_name}")
    done
    if [[ "${#OP_NAMES[@]}" == "0" ]]; then
        echo "[nanovllm_a5_ops] ERROR: NANOVLLM_A5_OPS_ONLY selected no operators." >&2
        exit 2
    fi
fi
echo "[nanovllm_a5_ops] selected operators: ${OP_NAMES[*]}"

# A single ExactScanSpeculativeGet build is a self-contained P1 component:
# staging the complete repository here would make msopgen package (and often
# compile) unrelated SFA/LI implementations.  Besides wasting minutes, that
# turns an unrelated legacy compile failure into a false P1 blocker.  Keep
# its generated project strictly to this operator plus the KVAF device helper
# injected below.  Multi-op/normal builds retain the repository-wide staging
# behavior because those existing operators have cross-directory headers.
STAGE_EXACT_SCAN_SPECULATIVE_GET_ONLY=0
STAGE_STREAMING_QUANT_LI_GET_ONLY=0
STAGE_EXACT_RECONCILE_PRIVATE_VIEW_ONLY=0
if [[ "${#OP_NAMES[@]}" == "1" && \
      "${OP_NAMES[0]}" == "A5ExactScanSpeculativeGet" ]]; then
    STAGE_EXACT_SCAN_SPECULATIVE_GET_ONLY=1
fi
if [[ "${#OP_NAMES[@]}" == "1" && \
      "${OP_NAMES[0]}" == "A5StreamingQuantLiGet" ]]; then
    STAGE_STREAMING_QUANT_LI_GET_ONLY=1
fi
if [[ "${#OP_NAMES[@]}" == "1" && \
      "${OP_NAMES[0]}" == "A5ExactReconcilePrivateView" ]]; then
    STAGE_EXACT_RECONCILE_PRIVATE_VIEW_ONLY=1
fi
if [[ " ${OP_NAMES[*]} " == *" A5StreamingQuantLiGet "* && \
      "${STAGE_STREAMING_QUANT_LI_GET_ONLY}" != "1" ]]; then
    echo "[nanovllm_a5_ops] ERROR: A5StreamingQuantLiGet is intentionally buildable only as NANOVLLM_A5_OPS_ONLY=A5StreamingQuantLiGet." >&2
    echo "[nanovllm_a5_ops] Mixing it with legacy operators would stage unrelated sources and invalidate the isolated C8 closure check." >&2
    exit 2
fi
if [[ " ${OP_NAMES[*]} " == *" A5ExactReconcilePrivateView "* && \
      "${STAGE_EXACT_RECONCILE_PRIVATE_VIEW_ONLY}" != "1" ]]; then
    echo "[nanovllm_a5_ops] ERROR: A5ExactReconcilePrivateView is intentionally buildable only as NANOVLLM_A5_OPS_ONLY=A5ExactReconcilePrivateView." >&2
    echo "[nanovllm_a5_ops] It is a transaction-local P3 probe and must not enter the normal product OPP collection." >&2
    exit 2
fi

# The repository stores each operator in its own directory. Build a temporary
# flat *operator collection* for msopgen, but preserve the relative directory
# structure below every op_host/op_kernel directory.  In particular, do not
# copy repository-local CMakeLists.txt files: msopgen owns the generated CMake
# scaffold and CANN 9.1 does not define legacy macros such as
# add_op_to_compiled_list().
rm -rf "${GENERATED}" "${OP_DEV}"
mkdir -p "${BUILD_ROOT}" "${OP_DEV}/op_host" "${OP_DEV}/op_kernel"
if [[ "${STAGE_EXACT_SCAN_SPECULATIVE_GET_ONLY}" == "1" ]]; then
    cp "${ROOT}/a5_exact_scan_speculative_get/ops.json" \
        "${OP_DEV}/ops.json"
    cp -a "${ROOT}/a5_exact_scan_speculative_get/op_host/." \
        "${OP_DEV}/op_host/"
    cp -a "${ROOT}/a5_exact_scan_speculative_get/op_kernel/." \
        "${OP_DEV}/op_kernel/"
    echo "[nanovllm_a5_ops] strict P1 staging: A5ExactScanSpeculativeGet only"
elif [[ "${STAGE_STREAMING_QUANT_LI_GET_ONLY}" == "1" ]]; then
    # A5StreamingQuantLiGet is a source-derived copy of the official C8 LI
    # closure.  Keep it isolated from every legacy custom op: generated OPP
    # code must see precisely this operator's host/kernel tree, plus the KVAF
    # device helper staged below.  In particular, do not use the old
    # descriptor_stage_only prototype: its ABI starts from externally supplied
    # score tiles and cannot prove score-time prefetch.
    STREAMING_LI_DIR="${ROOT}/a5_streaming_quant_li_get"
    STREAMING_LI_SPEC="${STREAMING_LI_DIR}/ops.json"
    if [[ ! -f "${STREAMING_LI_SPEC}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: A5StreamingQuantLiGet requires ${STREAMING_LI_SPEC}." >&2
        echo "[nanovllm_a5_ops] The descriptor_stage_only/ spec is intentionally incompatible and will not be staged." >&2
        exit 2
    fi
    cp "${STREAMING_LI_SPEC}" "${OP_DEV}/ops.json"
    cp -a "${STREAMING_LI_DIR}/op_host/." "${OP_DEV}/op_host/"
    cp -a "${STREAMING_LI_DIR}/op_kernel/." "${OP_DEV}/op_kernel/"
    echo "[nanovllm_a5_ops] strict P3 staging: A5StreamingQuantLiGet official C8 closure only"
    echo "[nanovllm_a5_ops] metadata companion: VllmQuantLightningIndexerMetadata must already be registered at runtime; this isolated OPP consumes its [1024] output and does not package a duplicate AICPU producer"
    echo "[nanovllm_a5_ops] layout contract: source-derived tiling preserves official layout_query=TND and BSND behavior; production decode uses TND with layout_key=PA_BSND"
elif [[ "${STAGE_EXACT_RECONCILE_PRIVATE_VIEW_ONLY}" == "1" ]]; then
    RECONCILE_DIR="${ROOT}/a5_exact_reconcile_private_view"
    RECONCILE_SPEC="${RECONCILE_DIR}/ops.json"
    if [[ ! -f "${RECONCILE_SPEC}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: A5ExactReconcilePrivateView requires ${RECONCILE_SPEC}." >&2
        exit 2
    fi
    cp "${RECONCILE_SPEC}" "${OP_DEV}/ops.json"
    cp -a "${RECONCILE_DIR}/op_host/." "${OP_DEV}/op_host/"
    cp -a "${RECONCILE_DIR}/op_kernel/." "${OP_DEV}/op_kernel/"
    echo "[nanovllm_a5_ops] strict P3 staging: A5ExactReconcilePrivateView only"
    echo "[nanovllm_a5_ops] transaction boundary: plan-only; no Get, QSFA, or persistent cache-slot publication"
else
    cp "${ROOT}/ops.json" "${OP_DEV}/ops.json"

    while IFS= read -r -d '' file; do
        rel="${file#*/op_host/}"
        if [[ "${rel}" == "CMakeLists.txt" || "${rel}" == */CMakeLists.txt ]]; then
            continue
        fi
        dst="${OP_DEV}/op_host/${rel}"
        mkdir -p "$(dirname "${dst}")"
        cp -f "${file}" "${dst}"
    done < <(
        find "${ROOT}" -type f -path '*/op_host/*' \
            ! -path "${ROOT}/a5_exact_reconcile_private_view/*" \
            ! -path "${BUILD_ROOT}/*" -print0
    )

    while IFS= read -r -d '' file; do
        rel="${file#*/op_kernel/}"
        if [[ "${rel}" == "CMakeLists.txt" || "${rel}" == */CMakeLists.txt ]]; then
            continue
        fi
        dst="${OP_DEV}/op_kernel/${rel}"
        mkdir -p "$(dirname "${dst}")"
        cp -f "${file}" "${dst}"
    done < <(
        find "${ROOT}" -type f -path '*/op_kernel/*' \
            ! -path "${ROOT}/a5_exact_reconcile_private_view/*" \
            ! -path "${BUILD_ROOT}/*" -print0
    )
fi

# ExactScanSpeculativeGet compiles the current KVAF device-side full Get
# protocol into its two dedicated AIV lanes.  Keep KVAF as the source of
# truth: stage the helper and ABI from the exact checkout selected for this
# build instead of maintaining a divergent WQE/CQ copy under vLLM.
if [[ " ${OP_NAMES[*]} " == *" A5ExactScanSpeculativeGet "* || \
      " ${OP_NAMES[*]} " == *" A5StreamingQuantLiGet "* ]]; then
    KVAF_CONSUMER_NAMES=()
    if [[ " ${OP_NAMES[*]} " == *" A5ExactScanSpeculativeGet "* ]]; then
        KVAF_CONSUMER_NAMES+=("A5ExactScanSpeculativeGet")
    fi
    if [[ " ${OP_NAMES[*]} " == *" A5StreamingQuantLiGet "* ]]; then
        KVAF_CONSUMER_NAMES+=("A5StreamingQuantLiGet")
    fi
    if [[ -z "${KVAF_FUSED_GET_SOURCE:-}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: ${KVAF_CONSUMER_NAMES[*]} requires KVAF_FUSED_GET_SOURCE=<exact KVAF checkout>." >&2
        exit 2
    fi
    KVAF_FUSED_HEADER="${KVAF_FUSED_GET_SOURCE}/access_operator/demo/aiv_batch_get/kernel/kvaf_fused_full_get_device.h"
    KVAF_FUSED_SOURCE="${KVAF_FUSED_GET_SOURCE}/access_operator/demo/aiv_batch_get/kernel/execute_batch_get_aiv.asc"
    KVAF_ABI_HEADER="${KVAF_FUSED_GET_SOURCE}/access_operator/demo/aiv_batch_get/include/batch_get_abi.h"
    if [[ ! -f "${KVAF_FUSED_HEADER}" || ! -f "${KVAF_FUSED_SOURCE}" || ! -f "${KVAF_ABI_HEADER}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: KVAF_FUSED_GET_SOURCE does not expose the P1 device helper/ABI." >&2
        echo "[nanovllm_a5_ops] helper=${KVAF_FUSED_HEADER}" >&2
        echo "[nanovllm_a5_ops] abi=${KVAF_ABI_HEADER}" >&2
        exit 2
    fi
    # P2's parallel Get lane relies on the split transaction API.  Rejecting
    # an older KVAF checkout here is much clearer than compiling the custom
    # operator against a blocking-only helper and silently losing the intended
    # descriptor/H2D overlap.
    if ! grep -q 'BeginKvafDenseBatchGetPostOnly' "${KVAF_FUSED_SOURCE}" || \
       ! grep -q 'TryRetireKvafDenseBatchGet' "${KVAF_FUSED_SOURCE}" || \
       ! grep -q 'FinishKvafDenseBatchGet' "${KVAF_FUSED_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: ${KVAF_CONSUMER_NAMES[*]} requires KVAF split PostOnly/TryRetire/Finish helpers." >&2
        echo "[nanovllm_a5_ops] source=${KVAF_FUSED_SOURCE}" >&2
        exit 2
    fi
    cp -f "${KVAF_FUSED_HEADER}" "${OP_DEV}/op_kernel/kvaf_fused_full_get_device.h"
    cp -f "${KVAF_FUSED_SOURCE}" "${OP_DEV}/op_kernel/execute_batch_get_aiv.asc"
    cp -f "${KVAF_ABI_HEADER}" "${OP_DEV}/op_kernel/batch_get_abi.h"
    # The embedded KVAF implementation uses SIMT atomics.  msopgen's
    # generated CCEC include list does not inherit KVAF's CMake discovery, so
    # stage the exact CANN SIMT headers next to the fused kernel instead of
    # relying on an ambient include path.
    KVAF_SIMT_ROOT=""
    for candidate in \
        "${ASCEND_SIMT_API_DIR:-}" \
        "${ASCEND_CANN_PACKAGE_PATH:-}" \
        "${ASCEND_HOME_PATH:-}" \
        "${CANN_INSTALL_PATH:-}"; do
        if [[ -z "${candidate}" ]]; then
            continue
        fi
        # CANN packages use more than one toolkit layout.  In particular,
        # the A5 9.1 image exposes SIMT under
        # <cann>/aarch64-linux/asc/include/simt_api, while development
        # toolkits commonly use either <cann>/include or
        # <cann>/include/ascendc.  Every item below is the directory directly
        # above simt_api, so the subsequent copy is layout-independent.
        for simt_root in \
            "${candidate}" \
            "${candidate}/include" \
            "${candidate}/include/ascendc" \
            "${candidate}/aarch64-linux/asc/include" \
            "${candidate}/aarch64-linux/asc/include/ascendc" \
            "${candidate}/tools/tikcpp/tikcfw/lib"; do
            if [[ -f "${simt_root}/simt_api/device_atomic_functions.h" ]]; then
                KVAF_SIMT_ROOT="${simt_root}"
                break
            fi
        done
        if [[ -n "${KVAF_SIMT_ROOT}" ]]; then
            break
        fi
    done
    if [[ -z "${KVAF_SIMT_ROOT}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: cannot find simt_api/device_atomic_functions.h." >&2
        echo "[nanovllm_a5_ops] Set ASCEND_SIMT_API_DIR to its parent directory." >&2
        exit 2
    fi
    rm -rf "${OP_DEV}/op_kernel/simt_api"
    cp -a "${KVAF_SIMT_ROOT}/simt_api" "${OP_DEV}/op_kernel/simt_api"
    echo "[nanovllm_a5_ops] ${KVAF_CONSUMER_NAMES[*]} KVAF source: ${KVAF_FUSED_GET_SOURCE}"
    echo "[nanovllm_a5_ops] ${KVAF_CONSUMER_NAMES[*]} SIMT root: ${KVAF_SIMT_ROOT}"
fi

if find "${OP_DEV}/op_host" "${OP_DEV}/op_kernel" \
        -type f -name 'CMakeLists.txt' -print -quit | grep -q .; then
    echo "[nanovllm_a5_ops] ERROR: repository CMakeLists.txt leaked into staging." >&2
    exit 1
fi

for index in "${!OP_NAMES[@]}"; do
    op_name="${OP_NAMES[index]}"
    op_spec="${ROOT}/ops.json"
    if [[ "${op_name}" == "A5FullBlockDump" ]]; then
        op_spec="${ROOT}/a5_full_block_dump/ops.json"
    elif [[ "${op_name}" == "A5PackedFullBlockDump" ]]; then
        op_spec="${ROOT}/a5_packed_full_block_dump/ops.json"
    elif [[ "${op_name}" == "A5ExactScanSpeculativePostProbe" ]]; then
        op_spec="${ROOT}/a5_exact_scan_speculative_post_probe/ops.json"
    elif [[ "${op_name}" == "A5ExactScanSpeculativeGet" ]]; then
        op_spec="${ROOT}/a5_exact_scan_speculative_get/ops.json"
    elif [[ "${op_name}" == "A5StreamingQuantLiGet" ]]; then
        op_spec="${ROOT}/a5_streaming_quant_li_get/ops.json"
    elif [[ "${op_name}" == "A5ExactReconcilePrivateView" ]]; then
        op_spec="${ROOT}/a5_exact_reconcile_private_view/ops.json"
    fi
    echo "[nanovllm_a5_ops] generate operator: ${op_name}"
    msopgen_args=(
        gen
        -i "${op_spec}"
        -f aclnn
        -c "ai_core-${A5_SOC_VERSION}"
        -lan cpp
        -op "${op_name}"
        -out "${GENERATED}"
    )
    if (( index > 0 )); then
        msopgen_args+=(-m 1)
    fi
    msopgen "${msopgen_args[@]}"
done

# msopgen owns the CMake scaffold. Checked-in Host/AscendC implementation
# sources overwrite only generated source stubs; directory structure is kept.
cp -a "${OP_DEV}/op_host/." "${GENERATED}/op_host/"
cp -a "${OP_DEV}/op_kernel/." "${GENERATED}/op_kernel/"
rm -f "${GENERATED}/op_host/lightning_indexer_decode_update_a5_def.cpp"

if grep -Rqs --include='CMakeLists.txt' 'add_op_to_compiled_list' \
        "${GENERATED}/op_host" "${GENERATED}/op_kernel"; then
    echo "[nanovllm_a5_ops] ERROR: legacy add_op_to_compiled_list leaked into the CANN 9.1 generated project." >&2
    exit 1
fi

pushd "${GENERATED}" >/dev/null
OPS_CPU_NUMBER="${BUILD_JOBS}" bash build.sh
popd >/dev/null

RUN_PKG="$(find "${GENERATED}/build_out" -maxdepth 1 -type f -name '*.run' | head -n 1)"
if [[ -z "${RUN_PKG}" ]]; then
    echo "[nanovllm_a5_ops] ERROR: msopgen project did not produce a .run package." >&2
    exit 1
fi

rm -rf "${LOCAL_OPP}"
mkdir -p "${LOCAL_OPP}"
chmod +x "${RUN_PKG}"
"${RUN_PKG}" --quiet --install-path="${LOCAL_OPP}"

mapfile -t OPAPI_LIBS < <(find "${LOCAL_OPP}/vendors" -type f -path '*/op_api/lib/libcust_opapi.so')
if [[ "${#OPAPI_LIBS[@]}" -ne 1 ]]; then
    echo "[nanovllm_a5_ops] ERROR: expected one local libcust_opapi.so, found ${#OPAPI_LIBS[@]}." >&2
    exit 1
fi
VENDOR_DIR="$(cd "$(dirname "${OPAPI_LIBS[0]}")/../.." && pwd)"

for op_name in "${OP_NAMES[@]}"; do
    if ! find "${VENDOR_DIR}" -type f -name 'binary_info_config.json' -print0 |
        xargs -0 -r grep -q "${op_name}"; then
        echo "[nanovllm_a5_ops] ERROR: ${op_name} is absent from generated kernel metadata." >&2
        exit 1
    fi
done

if [[ "${SKIP_TORCH_EXTENSION}" != "1" ]]; then
    # Legacy standalone mode retained for local operator development only.
    # The normal vLLM-Ascend package build sets SKIP_TORCH_EXTENSION=1 and
    # compiles the same registrations into vllm_ascend_C instead.
    pushd "${TORCH_EXTENSION}" >/dev/null
    rm -rf build
    rm -f nanovllm_dsa_a5/_C*.so
    MAX_JOBS="${BUILD_JOBS}" "${PYTHON_BIN}" setup.py build_ext --inplace
    popd >/dev/null

    export ASCEND_CUSTOM_OPP_PATH="${VENDOR_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
    export NANOVLLM_A5_INSTALL_OPP_PATH="${LOCAL_OPP}"
    PYTHONPATH="${TORCH_EXTENSION}:${PYTHONPATH:-}" "${PYTHON_BIN}" -c \
        "import nanovllm_dsa_a5; print('[nanovllm_a5_ops] standalone torch extension import: OK')"
else
    echo "[nanovllm_a5_ops] standalone nanovllm_dsa_a5._C build skipped; root vllm_ascend_C will own torch registration"
fi

echo "[nanovllm_a5_ops] build complete"
echo "[nanovllm_a5_ops] local vendor: ${VENDOR_DIR}"
