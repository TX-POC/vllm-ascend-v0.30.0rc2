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
# An isolated P8 build may use a machine-local CANN dependency cache.  This
# holds downloaded third-party packaging/build inputs only; it is not a reuse
# of an older OPP, bridge, or CMakeCache.  Keep legacy callers unchanged when
# the variable is unset.
CANN_3RD_LIB_PATH="${NANOVLLM_A5_CANN_3RD_LIB_PATH:-}"
# `SOC_VERSION` is a legacy/product-global setting and has repeatedly
# selected the wrong CANN product in isolated A5 builds.  Do not silently
# inherit it.  Callers must set only A5_SOC_VERSION (or accept ascend950).
if [[ -n "${SOC_VERSION:-}" ]]; then
    echo "[nanovllm_a5_ops] ERROR: legacy SOC_VERSION is set; unset it and use A5_SOC_VERSION=ascend950." >&2
    exit 2
fi
A5_SOC_VERSION_RAW="${A5_SOC_VERSION:-ascend950}"
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
if [[ -n "${CANN_3RD_LIB_PATH}" ]]; then
    CACHE_MAKESELF="${CANN_3RD_LIB_PATH}/makeself/makeself.sh"
    CACHE_MAKESELF_HEADER="${CANN_3RD_LIB_PATH}/makeself/makeself-header.sh"
    CACHE_JSON_HEADER="${CANN_3RD_LIB_PATH}/json/include/nlohmann/json.hpp"
    CACHE_ABSEIL_ARCHIVE="${CANN_3RD_LIB_PATH}/abseil-cpp/abseil-cpp-20230802.1.tar.gz"
    CACHE_ABSEIL_PKG_ARCHIVE="${CANN_3RD_LIB_PATH}/pkg/abseil-cpp-20230802.1.tar.gz"
    CACHE_PROTOBUF_ARCHIVE="${CANN_3RD_LIB_PATH}/protobuf/protobuf-all-25.1.tar.gz"
    CACHE_PROTOBUF_PKG_ARCHIVE="${CANN_3RD_LIB_PATH}/pkg/protobuf-25.1.tar.gz"
    if [[ ! -r "${CACHE_MAKESELF}" || ! -r "${CACHE_MAKESELF_HEADER}" || \
          ! -r "${CACHE_JSON_HEADER}" ]] || \
       { [[ ! -r "${CACHE_ABSEIL_ARCHIVE}" ]] && [[ ! -r "${CACHE_ABSEIL_PKG_ARCHIVE}" ]]; } || \
       { [[ ! -r "${CACHE_PROTOBUF_ARCHIVE}" ]] && [[ ! -r "${CACHE_PROTOBUF_PKG_ARCHIVE}" ]]; }; then
        echo "[nanovllm_a5_ops] ERROR: NANOVLLM_A5_CANN_3RD_LIB_PATH is incomplete: ${CANN_3RD_LIB_PATH}." >&2
        exit 2
    fi
    CANN_3RD_LIB_PATH="$(cd "${CANN_3RD_LIB_PATH}" && pwd -P)"
    echo "[nanovllm_a5_ops] CANN third-party cache: ${CANN_3RD_LIB_PATH} (makeself/json/abseil/protobuf ready)"
fi

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
# The source-derived streaming LI, reconcile planner, and P7.0 receipt oracle
# are deliberately not in ALL_OP_NAMES.  They are isolated probes, never
# default product routes.
SUPPORTED_OP_NAMES=(
    "${ALL_OP_NAMES[@]}"
    A5StreamingQuantLiGet
    A5ExactReconcilePrivateView
    A5ScoreReceiptCandidateExecutor
    A5ScoreReceiptSpeculativeExecutor
    A5P8ForwardOwnedExecutor
    A5P8ForwardControlAdvance
    A5P8ArenaAdmissionGate
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
# P7.0 is intentionally only a completed-C8 score receipt -> candidate oracle.
# It is not the P7 dynamic Post/CQ executor and must therefore be built as one
# operator in a fresh OPP package with no C8/KVAF/SIMT closure.
STAGE_SCORE_RECEIPT_CANDIDATE_EXECUTOR_ONLY=0
# P7.1 is the standalone candidate receipt -> Direct-HCCP Post/CQ executor.
# It deliberately consumes only explicit graph-visible CandidateSeed and
# lifecycle tensors; C8 source must never enter this OPP closure.
STAGE_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_ONLY=0
# P8 is the forward-owned product-path executor.  It is an isolated OPP on
# purpose: C8 remains a separately built score-receipt publisher, and this
# package contains only P8's CandidateSeed/PostOnly/CQ/QSFA-terminal state
# machine plus the exact KVAF Direct-HCCP helper closure.
STAGE_P8_FORWARD_OWNED_EXECUTOR_ONLY=0
# P3's source-derived C8 operator has two independent compile-time axes:
# publication (score workspace/receipts) and inline KVAF.  Default values
# are declared before staging because this script is nounset-safe and the
# later KVAF closure gate is shared with P1.
STREAMING_LI_PUBLISH_MODE=0
STREAMING_LI_PUBLISH_MODE_NAME=off
STREAMING_LI_INLINE_MODE=0
STREAMING_LI_INLINE_MODE_NAME=off
STREAMING_LI_POST_CLOSURE_MODE=0
STREAMING_LI_POST_CLOSURE_MODE_NAME=not_applicable
STREAMING_LI_POST_SCHEDULER_MODE=0
STREAMING_LI_POST_SCHEDULER_MODE_NAME=immediate
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
if [[ "${#OP_NAMES[@]}" == "1" && \
      "${OP_NAMES[0]}" == "A5ScoreReceiptCandidateExecutor" ]]; then
    STAGE_SCORE_RECEIPT_CANDIDATE_EXECUTOR_ONLY=1
fi
if [[ "${#OP_NAMES[@]}" == "1" && \
      "${OP_NAMES[0]}" == "A5ScoreReceiptSpeculativeExecutor" ]]; then
    STAGE_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_ONLY=1
fi
if [[ "${#OP_NAMES[@]}" == "3" && \
      "${OP_NAMES[0]}" == "A5P8ForwardOwnedExecutor" && \
      "${OP_NAMES[1]}" == "A5P8ForwardControlAdvance" && \
      "${OP_NAMES[2]}" == "A5P8ArenaAdmissionGate" ]]; then
    STAGE_P8_FORWARD_OWNED_EXECUTOR_ONLY=1
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
if [[ " ${OP_NAMES[*]} " == *" A5ScoreReceiptCandidateExecutor "* && \
      "${STAGE_SCORE_RECEIPT_CANDIDATE_EXECUTOR_ONLY}" != "1" ]]; then
    echo "[nanovllm_a5_ops] ERROR: A5ScoreReceiptCandidateExecutor is intentionally buildable only as NANOVLLM_A5_OPS_ONLY=A5ScoreReceiptCandidateExecutor." >&2
    echo "[nanovllm_a5_ops] It is P7.0's isolated completed-C8 candidate oracle; mixing it with C8 or KVAF sources would invalidate its source-closure gate." >&2
    exit 2
fi
if [[ " ${OP_NAMES[*]} " == *" A5ScoreReceiptSpeculativeExecutor "* && \
      "${STAGE_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_ONLY}" != "1" ]]; then
    echo "[nanovllm_a5_ops] ERROR: A5ScoreReceiptSpeculativeExecutor is buildable only as NANOVLLM_A5_OPS_ONLY=A5ScoreReceiptSpeculativeExecutor." >&2
    echo "[nanovllm_a5_ops] It is a standalone P7.1 PostOnly/CQ executor; mixing it with C8 or legacy source would invalidate the closure gate." >&2
    exit 2
fi
if [[ " ${OP_NAMES[*]} " == *" A5P8ForwardOwnedExecutor "* || \
      " ${OP_NAMES[*]} " == *" A5P8ForwardControlAdvance "* || \
      " ${OP_NAMES[*]} " == *" A5P8ArenaAdmissionGate "* ]]; then
    if [[ "${STAGE_P8_FORWARD_OWNED_EXECUTOR_ONLY}" != "1" ]]; then
        echo "[nanovllm_a5_ops] ERROR: P8 is buildable only as NANOVLLM_A5_OPS_ONLY=A5P8ForwardOwnedExecutor,A5P8ForwardControlAdvance,A5P8ArenaAdmissionGate." >&2
        echo "[nanovllm_a5_ops] Mixing P8 with C8, P7, or legacy operators would destroy the forward-owned source closure." >&2
        exit 2
    fi
fi

# The repository stores each operator in its own directory. Build a temporary
# flat *operator collection* for msopgen, but preserve the relative directory
# structure below every op_host/op_kernel directory.  In particular, do not
# copy repository-local CMakeLists.txt files: msopgen owns the generated CMake
# scaffold and CANN 9.1 does not define legacy macros such as
# add_op_to_compiled_list().
rm -rf "${GENERATED}" "${OP_DEV}"
mkdir -p "${BUILD_ROOT}" "${OP_DEV}/op_host" "${OP_DEV}/op_kernel"

# msopgen generates the only CMake scaffold that is valid for this CANN
# version.  Strict single-op staging used to copy every file verbatim, which
# accidentally reintroduced a checked-in operator CMakeLists.txt and made the
# later leak guard reject the staging area before compilation.  Keep the
# source closure complete while deliberately excluding repository build files.
copy_tree_without_repository_cmake() {
    local source_dir="$1"
    local destination_dir="$2"
    local file
    local rel
    local dst

    while IFS= read -r -d '' file; do
        rel="${file#${source_dir}/}"
        if [[ "${rel}" == "CMakeLists.txt" || "${rel}" == */CMakeLists.txt ]]; then
            continue
        fi
        dst="${destination_dir}/${rel}"
        mkdir -p "$(dirname "${dst}")"
        cp -f "${file}" "${dst}"
    done < <(find "${source_dir}" -type f -print0)
}

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
    # closure.  The control/publisher/audit variants deliberately contain no
    # KVAF transport code.  Only the separately selected inline=post variant
    # stages the exact KVAF Direct-HCCP/SIMT closure below.  In particular, do
    # not use the old descriptor_stage_only prototype: its ABI starts from
    # externally supplied score tiles and cannot prove score-time prefetch.
    STREAMING_LI_DIR="${ROOT}/a5_streaming_quant_li_get"
    STREAMING_LI_SPEC="${STREAMING_LI_DIR}/ops.json"
    STREAMING_LI_DEF="${STREAMING_LI_DIR}/op_host/a5_streaming_quant_li_get_def.cpp"
    STREAMING_LI_BRIDGE="${STREAMING_LI_DIR}/torch_extension/csrc/streaming_quant_li_get.cpp"
    # The normal vLLM-Ascend build supplies err/ops_err.h and log/log.h via
    # its repository-wide include graph.  An isolated msopgen project must
    # not import that whole graph: the common err header itself depends on an
    # unavailable monorepo log/log.h.  Stage one P3-local compatibility header
    # beneath both quoted include paths instead.  Its guards preserve every
    # host validation early return while keeping LI/Top-K math untouched.
    STREAMING_LI_HOST_COMPAT="${STREAMING_LI_DIR}/op_host/a5_streaming_host_compat.h"
    if [[ ! -f "${STREAMING_LI_SPEC}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: A5StreamingQuantLiGet requires ${STREAMING_LI_SPEC}." >&2
        echo "[nanovllm_a5_ops] The descriptor_stage_only/ spec is intentionally incompatible and will not be staged." >&2
        exit 2
    fi
    # CANN 9.1 only accepts int8 in the generic msopgen TypeRange.  This is
    # also how the official C8 JSON/bootstrap schema is expressed.  It must
    # never be mistaken for the actual target ABI: the checked-in Ascend950
    # OpAICoreConfig and typed bridge below remain the two fail-closed FP8
    # guards.  Check all three layers before spending time in msopgen.
    if [[ ! -f "${STREAMING_LI_HOST_COMPAT}" ]] || \
       ! grep -Eq '"name": "query".*"type": \["int8"\]' "${STREAMING_LI_SPEC}" || \
       ! grep -Eq '"name": "key".*"type": \["int8"\]' "${STREAMING_LI_SPEC}" || \
       ! grep -q 'aicore_config_950.Input("query")' "${STREAMING_LI_DEF}" || \
       ! grep -q 'aicore_config_950.Input("key")' "${STREAMING_LI_DEF}" || \
       ! grep -q 'ge::DT_FLOAT8_E4M3FN' "${STREAMING_LI_DEF}" || \
       ! grep -q 'at::kFloat8_e4m3fn' "${STREAMING_LI_BRIDGE}"; then
        echo "[nanovllm_a5_ops] ERROR: P3 CANN-bootstrap/FP8 ABI or host-header contract is incomplete." >&2
        echo "[nanovllm_a5_ops] expected: ops.json query/key=int8; Ascend950 OpDef and bridge=FP8 E4M3FN; standalone P3 host compatibility header." >&2
        exit 2
    fi
    cp "${STREAMING_LI_SPEC}" "${OP_DEV}/ops.json"
    copy_tree_without_repository_cmake "${STREAMING_LI_DIR}/op_host" \
        "${OP_DEV}/op_host"
    copy_tree_without_repository_cmake "${STREAMING_LI_DIR}/op_kernel" \
        "${OP_DEV}/op_kernel"
    # These decisions are compile-time because they change the exact AIV
    # closure. The appended ABI remains 24 inputs / 11 outputs in every
    # variant. True tile-inline KVAF is exclusive from score publication. The
    # single narrow exception, inline=audit + publisher, is a score-provenance
    # diagnostic binary: it has no KVAF/SIMT/HCCP closure and exposes the
    # score-GM consumed by C8 Top-K for comparison against audit receipts.
    STREAMING_LI_PUBLISH_MODE_RAW="${NANOVLLM_A5_STREAMING_LI_GET_PUBLISH_MODE:-}"
    STREAMING_LI_HOOK_MODE_LEGACY_RAW="${NANOVLLM_A5_STREAMING_LI_GET_HOOK_MODE:-}"
    if [[ -n "${STREAMING_LI_PUBLISH_MODE_RAW}" ]]; then
        case "${STREAMING_LI_PUBLISH_MODE_RAW,,}" in
            off)
                STREAMING_LI_PUBLISH_MODE=0
                STREAMING_LI_PUBLISH_MODE_NAME=off
                ;;
            score_workspace)
                STREAMING_LI_PUBLISH_MODE=1
                STREAMING_LI_PUBLISH_MODE_NAME=score_workspace
                ;;
            publisher)
                STREAMING_LI_PUBLISH_MODE=2
                STREAMING_LI_PUBLISH_MODE_NAME=publisher
                ;;
            *)
                echo "[nanovllm_a5_ops] ERROR: NANOVLLM_A5_STREAMING_LI_GET_PUBLISH_MODE must be off, score_workspace, or publisher; got ${STREAMING_LI_PUBLISH_MODE_RAW}." >&2
                exit 2
                ;;
        esac
        if [[ -n "${STREAMING_LI_HOOK_MODE_LEGACY_RAW}" ]]; then
            echo "[nanovllm_a5_ops] WARNING: NANOVLLM_A5_STREAMING_LI_GET_HOOK_MODE is ignored because NANOVLLM_A5_STREAMING_LI_GET_PUBLISH_MODE is set." >&2
        fi
    else
        # Compatibility only: historical hook=on meant the score publisher
        # was enabled; it now maps to publication mode 2. No old value maps
        # to score_workspace because that mode did not exist in the hook ABI.
        STREAMING_LI_HOOK_MODE_LEGACY_EFFECTIVE="${STREAMING_LI_HOOK_MODE_LEGACY_RAW:-on}"
        case "${STREAMING_LI_HOOK_MODE_LEGACY_EFFECTIVE,,}" in
            on)
                STREAMING_LI_PUBLISH_MODE=2
                STREAMING_LI_PUBLISH_MODE_NAME=publisher
                ;;
            off)
                STREAMING_LI_PUBLISH_MODE=0
                STREAMING_LI_PUBLISH_MODE_NAME=off
                ;;
            *)
                echo "[nanovllm_a5_ops] ERROR: legacy NANOVLLM_A5_STREAMING_LI_GET_HOOK_MODE must be on or off; got ${STREAMING_LI_HOOK_MODE_LEGACY_EFFECTIVE}. Use NANOVLLM_A5_STREAMING_LI_GET_PUBLISH_MODE=off|score_workspace|publisher." >&2
                exit 2
                ;;
        esac
        if [[ -n "${STREAMING_LI_HOOK_MODE_LEGACY_RAW}" ]]; then
            echo "[nanovllm_a5_ops] WARNING: legacy NANOVLLM_A5_STREAMING_LI_GET_HOOK_MODE=${STREAMING_LI_HOOK_MODE_LEGACY_RAW} maps to publication_mode=${STREAMING_LI_PUBLISH_MODE_NAME}." >&2
        fi
    fi
    STREAMING_LI_INLINE_MODE_RAW="${NANOVLLM_A5_STREAMING_LI_GET_INLINE_MODE:-off}"
    case "${STREAMING_LI_INLINE_MODE_RAW,,}" in
        off)
            STREAMING_LI_INLINE_MODE=0
            STREAMING_LI_INLINE_MODE_NAME=off
            ;;
        audit)
            STREAMING_LI_INLINE_MODE=1
            STREAMING_LI_INLINE_MODE_NAME=audit
            ;;
        post)
            STREAMING_LI_INLINE_MODE=2
            STREAMING_LI_INLINE_MODE_NAME=post
            ;;
        *)
            echo "[nanovllm_a5_ops] ERROR: NANOVLLM_A5_STREAMING_LI_GET_INLINE_MODE must be off, audit, or post; got ${STREAMING_LI_INLINE_MODE_RAW}." >&2
            exit 2
            ;;
    esac
    if [[ "${STREAMING_LI_INLINE_MODE}" != "0" && \
          "${STREAMING_LI_PUBLISH_MODE}" != "0" ]]; then
        if [[ "${STREAMING_LI_INLINE_MODE}" != "1" || \
              "${STREAMING_LI_PUBLISH_MODE}" != "2" ]]; then
            echo "[nanovllm_a5_ops] ERROR: inline mode requires publication=off, except diagnostic inline=audit + publication=publisher." >&2
            echo "[nanovllm_a5_ops] post/HCCP and score publication remain separate isolated OPP artifacts." >&2
            exit 2
        fi
    fi
    # A post artifact has two distinct device closures.  ``full`` is the
    # Direct-HCCP/SIMT implementation; ``empty_shell`` preserves its 24/11
    # ABI but compiles none of that closure so it can be used as a strict C8
    # math ablation.  Do not accept this selector for off/audit: that would
    # make a stale shell variable silently change an unrelated artifact.
    if [[ "${STREAMING_LI_INLINE_MODE}" == "2" ]]; then
        STREAMING_LI_POST_CLOSURE_MODE_RAW="${NANOVLLM_A5_STREAMING_LI_GET_POST_CLOSURE_MODE:-full}"
        case "${STREAMING_LI_POST_CLOSURE_MODE_RAW,,}" in
            empty_shell)
                STREAMING_LI_POST_CLOSURE_MODE=0
                STREAMING_LI_POST_CLOSURE_MODE_NAME=empty_shell
                ;;
            full)
                STREAMING_LI_POST_CLOSURE_MODE=1
                STREAMING_LI_POST_CLOSURE_MODE_NAME=full
                ;;
            *)
                echo "[nanovllm_a5_ops] ERROR: NANOVLLM_A5_STREAMING_LI_GET_POST_CLOSURE_MODE must be empty_shell or full for inline=post; got ${STREAMING_LI_POST_CLOSURE_MODE_RAW}." >&2
                exit 2
                ;;
        esac
    elif [[ -n "${NANOVLLM_A5_STREAMING_LI_GET_POST_CLOSURE_MODE:-}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: NANOVLLM_A5_STREAMING_LI_GET_POST_CLOSURE_MODE is valid only with NANOVLLM_A5_STREAMING_LI_GET_INLINE_MODE=post." >&2
        exit 2
    fi

    # P6 changes *when* a normal C8 worker posts a candidate, never which
    # worker is allowed to do it.  Keep the legacy immediate closure as the
    # default P3 oracle; candidate_seed is an explicit, provenance-recorded
    # build that first compacts exact scores and posts only at a later seam.
    if [[ "${STREAMING_LI_INLINE_MODE}" == "2" && \
          "${STREAMING_LI_POST_CLOSURE_MODE}" == "1" ]]; then
        STREAMING_LI_POST_SCHEDULER_MODE_RAW="${NANOVLLM_A5_STREAMING_LI_GET_POST_SCHEDULER:-immediate}"
        case "${STREAMING_LI_POST_SCHEDULER_MODE_RAW,,}" in
            immediate)
                STREAMING_LI_POST_SCHEDULER_MODE=0
                STREAMING_LI_POST_SCHEDULER_MODE_NAME=immediate
                ;;
            candidate_seed)
                STREAMING_LI_POST_SCHEDULER_MODE=1
                STREAMING_LI_POST_SCHEDULER_MODE_NAME=candidate_seed
                ;;
            *)
                echo "[nanovllm_a5_ops] ERROR: NANOVLLM_A5_STREAMING_LI_GET_POST_SCHEDULER must be immediate or candidate_seed for inline=post/full; got ${STREAMING_LI_POST_SCHEDULER_MODE_RAW}." >&2
                exit 2
                ;;
        esac
    elif [[ -n "${NANOVLLM_A5_STREAMING_LI_GET_POST_SCHEDULER:-}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: NANOVLLM_A5_STREAMING_LI_GET_POST_SCHEDULER is valid only with inline=post and post_closure=full." >&2
        exit 2
    fi

    # Start from a clean transport closure regardless of mode.  Only post is
    # populated from KVAF_FUSED_GET_SOURCE below; audit must never inherit a
    # stale helper checked into the vLLM tree.
    rm -rf "${OP_DEV}/op_kernel/kvaf_device" \
           "${OP_DEV}/op_kernel/simt_api"
    rm -f "${OP_DEV}/op_kernel/kvaf_fused_full_get_device.h" \
          "${OP_DEV}/op_kernel/execute_batch_get_aiv.asc" \
          "${OP_DEV}/op_kernel/batch_get_abi.h"
    if [[ "${STREAMING_LI_INLINE_MODE}" == "0" ]]; then
        rm -f "${OP_DEV}/op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h" \
              "${OP_DEV}/op_kernel/arch35/a5_streaming_quant_li_get_tile_audit.h"
        if [[ -e "${OP_DEV}/op_kernel/kvaf_device" || \
              -e "${OP_DEV}/op_kernel/simt_api" || \
              -e "${OP_DEV}/op_kernel/kvaf_fused_full_get_device.h" || \
              -e "${OP_DEV}/op_kernel/execute_batch_get_aiv.asc" || \
              -e "${OP_DEV}/op_kernel/batch_get_abi.h" || \
              -e "${OP_DEV}/op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h" || \
              -e "${OP_DEV}/op_kernel/arch35/a5_streaming_quant_li_get_tile_audit.h" ]]; then
            echo "[nanovllm_a5_ops] ERROR: P3 inline=off contains a KVAF/SIMT closure." >&2
            exit 2
        fi
    elif [[ "${STREAMING_LI_INLINE_MODE}" == "1" ]]; then
        # Audit is intentionally a score-only binary.  Delete all transport
        # sources from its staging tree, including the post hook itself, and
        # retain only the small no-SIMT audit observer.
        rm -f "${OP_DEV}/op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h"
        if [[ ! -f "${OP_DEV}/op_kernel/arch35/a5_streaming_quant_li_get_tile_audit.h" || \
              -e "${OP_DEV}/op_kernel/kvaf_device" || \
              -e "${OP_DEV}/op_kernel/simt_api" || \
              -e "${OP_DEV}/op_kernel/kvaf_fused_full_get_device.h" || \
              -e "${OP_DEV}/op_kernel/execute_batch_get_aiv.asc" || \
              -e "${OP_DEV}/op_kernel/batch_get_abi.h" || \
              -e "${OP_DEV}/op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h" ]]; then
            echo "[nanovllm_a5_ops] ERROR: P3 inline=audit closure must contain only the audit observer." >&2
            exit 2
        fi
    else
        rm -f "${OP_DEV}/op_kernel/arch35/a5_streaming_quant_li_get_tile_audit.h"
        if [[ "${STREAMING_LI_POST_CLOSURE_MODE}" == "0" ]]; then
            rm -f "${OP_DEV}/op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h"
            if [[ -e "${OP_DEV}/op_kernel/kvaf_device" || \
                  -e "${OP_DEV}/op_kernel/simt_api" || \
                  -e "${OP_DEV}/op_kernel/kvaf_fused_full_get_device.h" || \
                  -e "${OP_DEV}/op_kernel/execute_batch_get_aiv.asc" || \
                  -e "${OP_DEV}/op_kernel/batch_get_abi.h" || \
                  -e "${OP_DEV}/op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h" ]]; then
                echo "[nanovllm_a5_ops] ERROR: P3 inline=post empty_shell contains a KVAF/SIMT/post-hook closure." >&2
                exit 2
            fi
        elif [[ ! -f "${OP_DEV}/op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h" ]]; then
            echo "[nanovllm_a5_ops] ERROR: P3 inline=post full requires the score-tile post hook header." >&2
            exit 2
        fi
    fi
    # The mode header is not just four raw values: it derives the closure
    # macros consumed by the C8 source (`HAS_FULL_POST_CLOSURE`,
    # `HAS_CANDIDATE_SEED_PIPELINE`, and the score-source-lens alias).
    # Earlier staging replaced that source header with the raw definitions,
    # which silently compiled P6 as the no-post branch.  Prepend the selected
    # raw values, then append the canonical derivation body.  Its `#ifndef`
    # guards preserve these chosen values.
    STAGED_FULL_POST_EXPECTED=0
    if [[ "${STREAMING_LI_INLINE_MODE}" == "2" && \
          "${STREAMING_LI_POST_CLOSURE_MODE}" == "1" ]]; then
        STAGED_FULL_POST_EXPECTED=1
    fi
    STAGED_CANDIDATE_SEED_EXPECTED=0
    if [[ "${STREAMING_LI_POST_SCHEDULER_MODE}" == "1" ]]; then
        STAGED_CANDIDATE_SEED_EXPECTED=1
    fi
    {
        printf '%s\n' \
            '// Generated only in the isolated P3 staging tree.' \
            "#define A5_STREAMING_LI_GET_PUBLISH_MODE ${STREAMING_LI_PUBLISH_MODE}" \
            "#define A5_STREAMING_LI_GET_INLINE_MODE ${STREAMING_LI_INLINE_MODE}" \
            "#define A5_STREAMING_LI_GET_POST_CLOSURE_MODE ${STREAMING_LI_POST_CLOSURE_MODE}" \
            "#define A5_STREAMING_LI_GET_POST_SCHEDULER_MODE ${STREAMING_LI_POST_SCHEDULER_MODE}"
        cat "${STREAMING_LI_DIR}/op_kernel/a5_streaming_quant_li_get_hook_mode.h"
        printf '\n#if A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE != %s\n' \
            "${STAGED_FULL_POST_EXPECTED}"
        printf '%s\n' '#error "streaming LI staged full-post closure derivation mismatch"' '#endif'
        printf '#if A5_STREAMING_LI_GET_HAS_CANDIDATE_SEED_PIPELINE != %s\n' \
            "${STAGED_CANDIDATE_SEED_EXPECTED}"
        printf '%s\n' '#error "streaming LI staged CandidateSeed derivation mismatch"' '#endif'
    } > "${OP_DEV}/op_kernel/a5_streaming_quant_li_get_hook_mode.h"
    if ! grep -Fqx "#define A5_STREAMING_LI_GET_PUBLISH_MODE ${STREAMING_LI_PUBLISH_MODE}" \
            "${OP_DEV}/op_kernel/a5_streaming_quant_li_get_hook_mode.h" || \
       ! grep -Fqx "#define A5_STREAMING_LI_GET_INLINE_MODE ${STREAMING_LI_INLINE_MODE}" \
            "${OP_DEV}/op_kernel/a5_streaming_quant_li_get_hook_mode.h" || \
       ! grep -Fqx "#define A5_STREAMING_LI_GET_POST_CLOSURE_MODE ${STREAMING_LI_POST_CLOSURE_MODE}" \
            "${OP_DEV}/op_kernel/a5_streaming_quant_li_get_hook_mode.h" || \
       ! grep -Fqx "#define A5_STREAMING_LI_GET_POST_SCHEDULER_MODE ${STREAMING_LI_POST_SCHEDULER_MODE}" \
            "${OP_DEV}/op_kernel/a5_streaming_quant_li_get_hook_mode.h" || \
       ! grep -Fq 'streaming LI staged full-post closure derivation mismatch' \
            "${OP_DEV}/op_kernel/a5_streaming_quant_li_get_hook_mode.h" || \
       ! grep -Fq 'streaming LI staged CandidateSeed derivation mismatch' \
            "${OP_DEV}/op_kernel/a5_streaming_quant_li_get_hook_mode.h"; then
        echo "[nanovllm_a5_ops] ERROR: P3 mode staging header is incomplete." >&2
        exit 2
    fi
    mkdir -p "${OP_DEV}/op_host/err" "${OP_DEV}/op_host/log"
    cp -f "${STREAMING_LI_HOST_COMPAT}" "${OP_DEV}/op_host/err/ops_err.h"
    cp -f "${STREAMING_LI_HOST_COMPAT}" "${OP_DEV}/op_host/log/log.h"
    if [[ ! -s "${OP_DEV}/op_host/err/ops_err.h" ]] || \
       ! grep -q 'OP_CHECK_NULL_WITH_CONTEXT' "${OP_DEV}/op_host/err/ops_err.h" || \
       ! grep -q 'OP_CHECK_IF' "${OP_DEV}/op_host/err/ops_err.h" || \
       ! grep -q 'OP_LOGE' "${OP_DEV}/op_host/log/log.h" || \
       ! grep -q 'REPORT_INNER_ERR_MSG' "${OP_DEV}/op_host/err/ops_err.h"; then
        echo "[nanovllm_a5_ops] ERROR: P3 staged host closure is incomplete (err/ops_err.h + log/log.h)." >&2
        exit 2
    fi
    echo "[nanovllm_a5_ops] strict P3 staging: A5StreamingQuantLiGet source-derived official C8 closure"
    echo "[nanovllm_a5_ops] P3 publication mode: ${STREAMING_LI_PUBLISH_MODE_NAME} (compiled=${STREAMING_LI_PUBLISH_MODE})"
    echo "[nanovllm_a5_ops] P3 inline KVAF mode: ${STREAMING_LI_INLINE_MODE_NAME} (compiled=${STREAMING_LI_INLINE_MODE})"
    echo "[nanovllm_a5_ops] P3 post closure mode: ${STREAMING_LI_POST_CLOSURE_MODE_NAME} (compiled=${STREAMING_LI_POST_CLOSURE_MODE})"
    echo "[nanovllm_a5_ops] P6 post scheduler: ${STREAMING_LI_POST_SCHEDULER_MODE_NAME} (compiled=${STREAMING_LI_POST_SCHEDULER_MODE}); no fixed Get core=1"
    case "${STREAMING_LI_INLINE_MODE}" in
        0)
            echo "[nanovllm_a5_ops] P3 KVAF/SIMT closure: absent (inline=off) verified=1"
            ;;
        1)
            echo "[nanovllm_a5_ops] P3 KVAF/SIMT closure: absent (inline=audit score-only) verified=1"
            if [[ "${STREAMING_LI_PUBLISH_MODE}" == "2" ]]; then
                echo "[nanovllm_a5_ops] P3 audit score provenance: publisher score-GM mirror enabled; no H2D/KVAF closure=1"
            fi
            ;;
        2)
            if [[ "${STREAMING_LI_POST_CLOSURE_MODE}" == "0" ]]; then
                echo "[nanovllm_a5_ops] P3 KVAF/SIMT/post-hook closure: absent (inline=post empty_shell) verified=1"
            else
                echo "[nanovllm_a5_ops] P3 KVAF/SIMT closure: post-only required from KVAF_FUSED_GET_SOURCE"
            fi
            ;;
    esac
    echo "[nanovllm_a5_ops] strict P3 staging: repository CMakeLists.txt excluded"
    echo "[nanovllm_a5_ops] P3 JSON TypeRange=int8 bootstrap; target OpDef/bridge=FP8 E4M3FN"
    echo "[nanovllm_a5_ops] P3 host-header closure: staged standalone err/ops_err.h + log/log.h; remaining host SDK includes=CANN"
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
elif [[ "${STAGE_SCORE_RECEIPT_CANDIDATE_EXECUTOR_ONLY}" == "1" ]]; then
    P7_DIR="${ROOT}/a5_score_receipt_candidate_executor"
    P7_SPEC="${P7_DIR}/ops.json"
    P7_HOST="${P7_DIR}/op_host/a5_score_receipt_candidate_executor.cpp"
    P7_KERNEL="${P7_DIR}/op_kernel/a5_score_receipt_candidate_executor.cpp"
    P7_TILING="${P7_DIR}/op_kernel/a5_score_receipt_candidate_executor_tiling.h"
    for P7_REQUIRED_FILE in "${P7_SPEC}" "${P7_HOST}" "${P7_KERNEL}" \
                            "${P7_TILING}"; do
        if [[ ! -s "${P7_REQUIRED_FILE}" ]]; then
            echo "[nanovllm_a5_ops] ERROR: P7.0 candidate oracle requires ${P7_REQUIRED_FILE}." >&2
            exit 2
        fi
    done
    cp "${P7_SPEC}" "${OP_DEV}/ops.json"
    cp -a "${P7_DIR}/op_host/." "${OP_DEV}/op_host/"
    cp -a "${P7_DIR}/op_kernel/." "${OP_DEV}/op_kernel/"
    echo "[nanovllm_a5_ops] strict P7.0 staging: A5ScoreReceiptCandidateExecutor only"
    echo "[nanovllm_a5_ops] P7.0 contract: completed-C8 score receipt oracle; C8/KVAF/SIMT/Post/CQ closure=absent"
elif [[ "${STAGE_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_ONLY}" == "1" ]]; then
    P7_1_DIR="${ROOT}/a5_score_receipt_speculative_executor"
    P7_1_SPEC="${P7_1_DIR}/ops.json"
    P7_1_HOST="${P7_1_DIR}/op_host/a5_score_receipt_speculative_executor.cpp"
    P7_1_KERNEL="${P7_1_DIR}/op_kernel/a5_score_receipt_speculative_executor.cpp"
    P7_1_TILING="${P7_1_DIR}/op_kernel/a5_score_receipt_speculative_executor_tiling.h"
    for P7_1_REQUIRED_FILE in "${P7_1_SPEC}" "${P7_1_HOST}" "${P7_1_KERNEL}" \
                              "${P7_1_TILING}"; do
        if [[ ! -s "${P7_1_REQUIRED_FILE}" ]]; then
            echo "[nanovllm_a5_ops] ERROR: P7.1 speculative executor requires ${P7_1_REQUIRED_FILE}." >&2
            exit 2
        fi
    done
    cp "${P7_1_SPEC}" "${OP_DEV}/ops.json"
    cp -a "${P7_1_DIR}/op_host/." "${OP_DEV}/op_host/"
    cp -a "${P7_1_DIR}/op_kernel/." "${OP_DEV}/op_kernel/"
    echo "[nanovllm_a5_ops] strict P7.1 staging: A5ScoreReceiptSpeculativeExecutor only"
    echo "[nanovllm_a5_ops] P7.1 contract: CandidateSeed -> dynamic queue lease -> Direct-HCCP PostOnly/CQ; C8 closure=absent"
elif [[ "${STAGE_P8_FORWARD_OWNED_EXECUTOR_ONLY}" == "1" ]]; then
    # P8 deliberately has a narrow, forward-owned closure.  Do not copy a
    # whole neighbouring probe directory here: a stale P7/C8 source in this
    # package would be a materially different execution model, not a harmless
    # build dependency.
    P8_DIR="${ROOT}/a5_p8_forward_owned_executor"
    P8_SPEC="${P8_DIR}/ops.json"
    P8_HOST="${P8_DIR}/op_host/a5_p8_forward_owned_executor.cpp"
    P8_KERNEL="${P8_DIR}/op_kernel/a5_p8_forward_owned_executor.cpp"
    P8_TILING="${P8_DIR}/op_kernel/a5_p8_forward_owned_executor_tiling.h"
    P8_ARENA_TAG="${P8_DIR}/op_kernel/a5_p8_arena_tag.h"
    P8_ADVANCE_HOST="${P8_DIR}/op_host/a5_p8_forward_control_advance.cpp"
    P8_ADVANCE_KERNEL="${P8_DIR}/op_kernel/a5_p8_forward_control_advance.cpp"
    P8_ADVANCE_TILING="${P8_DIR}/op_kernel/a5_p8_forward_control_advance_tiling.h"
    P8_GATE_HOST="${P8_DIR}/op_host/a5_p8_arena_admission_gate.cpp"
    P8_GATE_KERNEL="${P8_DIR}/op_kernel/a5_p8_arena_admission_gate.cpp"
    P8_GATE_TILING="${P8_DIR}/op_kernel/a5_p8_arena_admission_gate_tiling.h"
    for P8_REQUIRED_FILE in "${P8_SPEC}" "${P8_HOST}" "${P8_KERNEL}" \
                            "${P8_TILING}" "${P8_ARENA_TAG}" "${P8_ADVANCE_HOST}" \
                            "${P8_ADVANCE_KERNEL}" "${P8_ADVANCE_TILING}" \
                            "${P8_GATE_HOST}" "${P8_GATE_KERNEL}" \
                            "${P8_GATE_TILING}"; do
        if [[ ! -s "${P8_REQUIRED_FILE}" ]]; then
            echo "[nanovllm_a5_ops] ERROR: P8 forward-owned executor requires ${P8_REQUIRED_FILE}." >&2
            exit 2
        fi
    done
    cp "${P8_SPEC}" "${OP_DEV}/ops.json"
    cp "${P8_HOST}" "${P8_ADVANCE_HOST}" "${P8_GATE_HOST}" \
       "${OP_DEV}/op_host/"
    cp "${P8_KERNEL}" "${P8_TILING}" "${P8_ARENA_TAG}" "${P8_ADVANCE_KERNEL}" \
       "${P8_ADVANCE_TILING}" "${P8_GATE_KERNEL}" "${P8_GATE_TILING}" \
       "${OP_DEV}/op_kernel/"
    echo "[nanovllm_a5_ops] strict P8 staging: arena-admission gate + forward-owned executor + graph-visible control advance only"
    echo "[nanovllm_a5_ops] P8 contract: one-time arena admission -> receipt -> dynamic queue lease -> durable FILLING/PostOnly/CQ -> QSFA-terminal verified private release -> ForwardTerminal quarantine; captured POST->DRAIN->QSFA->FORWARD_TERMINAL advance; C8/P7 closure=absent"
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

# Keep KVAF as the source of truth: stage the helper and ABI from the exact
# checkout selected for this build instead of maintaining a divergent WQE/CQ
# copy under vLLM.  P3 publication-only variants need no such closure, while
# the real score-tile inline post variant does.  The audit variant is
# deliberately score-only and therefore must not stage this transport closure.
NEEDS_P1_DENSE_KVAF=0
NEEDS_P3_POST_KVAF=0
NEEDS_P7_1_STREAMING_KVAF=0
NEEDS_P8_FORWARD_OWNED_KVAF=0
# Never inherit an ambient shell's staging variables.  An audit-only build
# must not accidentally validate or package a closure left by an earlier
# post/P1 invocation in the same terminal.
KVAF_STAGE_DIR=""
KVAF_FUSED_HEADER=""
KVAF_FUSED_SOURCE=""
KVAF_ABI_HEADER=""
KVAF_FUSED_HEAD="not_applicable"
if [[ " ${OP_NAMES[*]} " == *" A5ExactScanSpeculativeGet "* ]]; then
    NEEDS_P1_DENSE_KVAF=1
fi
if [[ "${STAGE_STREAMING_QUANT_LI_GET_ONLY}" == "1" && \
      "${STREAMING_LI_INLINE_MODE}" == "2" && \
      "${STREAMING_LI_POST_CLOSURE_MODE}" == "1" ]]; then
    NEEDS_P3_POST_KVAF=1
fi
if [[ "${STAGE_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_ONLY}" == "1" ]]; then
    NEEDS_P7_1_STREAMING_KVAF=1
fi
if [[ "${STAGE_P8_FORWARD_OWNED_EXECUTOR_ONLY}" == "1" ]]; then
    NEEDS_P8_FORWARD_OWNED_KVAF=1
fi
if [[ "${NEEDS_P1_DENSE_KVAF}" == "1" || \
      "${NEEDS_P3_POST_KVAF}" == "1" || \
      "${NEEDS_P7_1_STREAMING_KVAF}" == "1" || \
      "${NEEDS_P8_FORWARD_OWNED_KVAF}" == "1" ]]; then
    KVAF_CONSUMER_NAMES=()
    if [[ "${NEEDS_P1_DENSE_KVAF}" == "1" ]]; then
        KVAF_CONSUMER_NAMES+=("A5ExactScanSpeculativeGet")
    fi
    if [[ "${NEEDS_P3_POST_KVAF}" == "1" ]]; then
        KVAF_CONSUMER_NAMES+=("A5StreamingQuantLiGet(inline=post)")
    fi
    if [[ "${NEEDS_P7_1_STREAMING_KVAF}" == "1" ]]; then
        KVAF_CONSUMER_NAMES+=("A5ScoreReceiptSpeculativeExecutor")
    fi
    if [[ "${NEEDS_P8_FORWARD_OWNED_KVAF}" == "1" ]]; then
        KVAF_CONSUMER_NAMES+=("A5P8ForwardOwnedExecutor")
    fi
    if [[ -z "${KVAF_FUSED_GET_SOURCE:-}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: ${KVAF_CONSUMER_NAMES[*]} requires KVAF_FUSED_GET_SOURCE=<exact KVAF checkout>." >&2
        exit 2
    fi
    KVAF_FUSED_HEADER="${KVAF_FUSED_GET_SOURCE}/access_operator/demo/aiv_batch_get/kernel/kvaf_fused_full_get_device.h"
    KVAF_FUSED_SOURCE="${KVAF_FUSED_GET_SOURCE}/access_operator/demo/aiv_batch_get/kernel/execute_batch_get_aiv.asc"
    KVAF_ABI_HEADER="${KVAF_FUSED_GET_SOURCE}/access_operator/demo/aiv_batch_get/include/batch_get_abi.h"
    if [[ ! -f "${KVAF_FUSED_HEADER}" || ! -f "${KVAF_FUSED_SOURCE}" || ! -f "${KVAF_ABI_HEADER}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: KVAF_FUSED_GET_SOURCE does not expose the required device helper/ABI." >&2
        echo "[nanovllm_a5_ops] helper=${KVAF_FUSED_HEADER}" >&2
        echo "[nanovllm_a5_ops] abi=${KVAF_ABI_HEADER}" >&2
        exit 2
    fi
    # ExactScan's separate Get lane needs the dense split lifecycle.  Inline
    # P3 instead uses the streaming lease/CQ lifecycle checked below.
    if [[ "${NEEDS_P1_DENSE_KVAF}" == "1" ]] && \
       { ! grep -q 'BeginKvafDenseBatchGetPostOnly' "${KVAF_FUSED_SOURCE}" || \
         ! grep -q 'TryRetireKvafDenseBatchGet' "${KVAF_FUSED_SOURCE}" || \
         ! grep -q 'FinishKvafDenseBatchGet' "${KVAF_FUSED_SOURCE}"; }; then
        echo "[nanovllm_a5_ops] ERROR: A5ExactScanSpeculativeGet requires KVAF split PostOnly/TryRetire/Finish helpers." >&2
        echo "[nanovllm_a5_ops] source=${KVAF_FUSED_SOURCE}" >&2
        exit 2
    fi
    # ExactScan reads the queue control from GM while the embedded Get helper
    # also materializes a descriptor in GM. AscendC does not allow an
    # implicit aggregate copy across that address-space boundary.  The KVAF
    # helper therefore has an explicit, field-wise ABI bridge.  Earlier KVAF
    # revisions already exposed the split Get lifecycle above, so checking
    # only that lifecycle would still allow a stale header to reach CANN and
    # fail minutes later with opaque __gm__/Local-Memory diagnostics.
    #
    # Fail before msopgen/CANN and print the exact checkout used for staging.
    # This is a build-closure guard only: it neither changes LI score/Top-K
    # math nor the Direct-HCCP command/CQ ordering.
    KVAF_FUSED_HEAD="unavailable"
    if ! git -C "${KVAF_FUSED_GET_SOURCE}" rev-parse --is-inside-work-tree \
            >/dev/null 2>&1; then
        echo "[nanovllm_a5_ops] ERROR: KVAF_FUSED_GET_SOURCE must be a git worktree; refuse an unverifiable transport closure." >&2
        echo "[nanovllm_a5_ops] source=${KVAF_FUSED_GET_SOURCE}" >&2
        exit 2
    fi
    KVAF_FUSED_HEAD="$(git -C "${KVAF_FUSED_GET_SOURCE}" rev-parse HEAD 2>/dev/null || echo unavailable)"
    KVAF_FUSED_DIRTY="$(git -C "${KVAF_FUSED_GET_SOURCE}" status --porcelain --untracked-files=no 2>/dev/null || true)"
    if [[ -n "${KVAF_FUSED_DIRTY}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: refuse KVAF transport closure from a tracked-dirty worktree." >&2
        echo "[nanovllm_a5_ops] source=${KVAF_FUSED_GET_SOURCE} head=${KVAF_FUSED_HEAD}" >&2
        printf '%s\n' "${KVAF_FUSED_DIRTY}" >&2
        exit 2
    fi
    for required_token in \
        'using ::KvafStreamingBatchGetControl;' \
        'using ::KvafStreamingBatchGetDescriptorScratch;' \
        'LoadKvafStreamingBatchGetControlFromGmForAicore' \
        'LoadKvafStreamingBatchGetControlFromGmForSimt' \
        'LoadKvafStreamingBatchGetDescriptorScratchFromGm' \
        'StoreKvafStreamingBatchGetDescriptorScratchToGm'; do
        if ! grep -Fq "${required_token}" "${KVAF_FUSED_HEADER}"; then
            echo "[nanovllm_a5_ops] ERROR: ${KVAF_CONSUMER_NAMES[*]} requires KVAF GM/local ABI bridge." >&2
            echo "[nanovllm_a5_ops] kvaf_head=${KVAF_FUSED_HEAD} helper=${KVAF_FUSED_HEADER}" >&2
            echo "[nanovllm_a5_ops] missing_token=${required_token}" >&2
            exit 2
        fi
    done
    for forbidden_token in \
        'const KvafStreamingBatchGetControl control = *control_gm;' \
        '*descriptor = *source;' \
        '*scratch = descriptor;'; do
        if grep -Fq "${forbidden_token}" "${KVAF_FUSED_HEADER}"; then
            echo "[nanovllm_a5_ops] ERROR: stale implicit GM/local aggregate copy in KVAF helper." >&2
            echo "[nanovllm_a5_ops] kvaf_head=${KVAF_FUSED_HEAD} helper=${KVAF_FUSED_HEADER}" >&2
            echo "[nanovllm_a5_ops] forbidden_token=${forbidden_token}" >&2
            exit 2
        fi
    done
    if [[ "${NEEDS_P3_POST_KVAF}" == "1" || \
          "${NEEDS_P7_1_STREAMING_KVAF}" == "1" ]]; then
        for required_token in \
            'TryAcquireKvafStreamingBatchGetQueueFromAicore' \
            'LoadKvafStreamingBatchGetQueueTemplate' \
            'PublishKvafStreamingBatchGetDescriptorScratch' \
            'BeginKvafStreamingBatchGet' \
            'TryRetireKvafStreamingBatchGet' \
            'FinishKvafStreamingBatchGet' \
            'GetKvafStreamingBatchGetReady' \
            'ReleaseKvafStreamingBatchGetQueueFromAicore'; do
            if ! grep -Fq "${required_token}" "${KVAF_FUSED_HEADER}"; then
                echo "[nanovllm_a5_ops] ERROR: streaming KVAF closure misses ${required_token}." >&2
                echo "[nanovllm_a5_ops] kvaf_head=${KVAF_FUSED_HEAD} helper=${KVAF_FUSED_HEADER}" >&2
                exit 2
            fi
        done
    fi
    # P8 must use the v2 forward-owned lifecycle—not the older P7.1
    # queue-local callback protocol.  Check the exact KVAF helper and ABI
    # before staging anything so an old checkout cannot spend minutes in CANN
    # and then fail with a mismatched record layout or a missing drain claim.
    if [[ "${NEEDS_P8_FORWARD_OWNED_KVAF}" == "1" ]]; then
        # P8 has a durable graph ABI shared by Post, Drain, QSFA terminal,
        # and ForwardTerminal.  A merely "new enough" helper is unsafe:
        # staging a different checkout can compile while silently changing a
        # graph-stable record protocol.  The small P8 runner supplies this
        # exact SHA; require it here before CANN work begins.
        P8_EXPECTED_KVAF_SHA="${NANOVLLM_A5_P8_KVAF_EXPECTED_SHA:-}"
        if [[ -z "${P8_EXPECTED_KVAF_SHA}" ]]; then
            echo "[nanovllm_a5_ops] ERROR: P8 requires NANOVLLM_A5_P8_KVAF_EXPECTED_SHA=<exact clean KVAF HEAD>." >&2
            echo "[nanovllm_a5_ops] selected_kvaf_head=${KVAF_FUSED_HEAD}" >&2
            exit 2
        fi
        if [[ "${KVAF_FUSED_HEAD}" != "${P8_EXPECTED_KVAF_SHA}" ]]; then
            echo "[nanovllm_a5_ops] ERROR: P8 refuses a KVAF source/provenance mismatch." >&2
            echo "[nanovllm_a5_ops] expected_kvaf_head=${P8_EXPECTED_KVAF_SHA} actual_kvaf_head=${KVAF_FUSED_HEAD}" >&2
            exit 2
        fi
        for required_token in \
            'KvafP8ForwardControl' \
            'LoadKvafP8ForwardControlFromGm' \
            'StoreKvafP8ForwardControlToGm' \
            'MarkKvafP8ForwardAbortFromAicore' \
            'KvafP8ForwardAbortResult' \
            'KvafP8PrivateRecord' \
            'KvafP8PrivateReleaseResult' \
            'LoadKvafP8WaveRecordFromGm' \
            'StoreKvafP8WaveRecordToGm' \
            'TryClaimKvafP8WaveDrainFromAicore' \
            'TryClaimKvafP8UnpostedWaveDrainFromAicore' \
            'BeginAndPublishKvafP8WaveFilling' \
            'ArmKvafP8WaveFillingForExternalWrites' \
            'ReleaseKvafP8FillingLeaseAfterProvenNoWork' \
            'PromoteKvafP8WaveFillingToPostIntent' \
            'kKvafP8WaveFilling' \
            'PostKvafP8Wave' \
            'TryRetireKvafP8Wave' \
            'FinishKvafP8Wave' \
            'PublishKvafP8PrivateTerminalRecords' \
            'ReleaseKvafP8WaveAfterPrivateReady' \
            'ReleaseKvafP8PendingPrivateRecordAfterNoPost' \
            'ReleaseKvafP8ClaimedUnpostedWaveAfterPrivateCleanup' \
            'QuarantineKvafP8PrivateReleaseClaimedFromAicore' \
            'ReleaseKvafP8PrivateRecordAfterQsfaTerminal' \
            'BeginKvafP8PrivateAllocatingPayloadFromSimt' \
            'QuarantineKvafP8WaveAtForwardTerminal' \
            'QuarantineKvafP8PrivateAtForwardTerminal' \
            'KvafP8ForwardTerminalResult'; do
            if ! grep -Fq "${required_token}" "${KVAF_FUSED_HEADER}"; then
                echo "[nanovllm_a5_ops] ERROR: P8 forward-owned KVAF v2 closure misses ${required_token}." >&2
                echo "[nanovllm_a5_ops] kvaf_head=${KVAF_FUSED_HEAD} helper=${KVAF_FUSED_HEADER}" >&2
                exit 2
            fi
        done
        for required_token in \
            'KVAF_P8_WAVE_RECORD_BYTES' \
            'KVAF_P8_PRIVATE_RECORD_BYTES' \
            'KVAF_P8_FORWARD_CONTROL_BYTES'; do
            if ! grep -Fq "${required_token}" "${KVAF_ABI_HEADER}"; then
                echo "[nanovllm_a5_ops] ERROR: P8 forward-owned KVAF v2 ABI misses ${required_token}." >&2
                echo "[nanovllm_a5_ops] kvaf_head=${KVAF_FUSED_HEAD} abi=${KVAF_ABI_HEADER}" >&2
                exit 2
            fi
        done
    fi
    # ExactScan includes the helper through its relative kvaf_device path.
    # Keep the flat copies for the P1 closure, but overwrite the relative
    # closure as well. Otherwise a checked-in, older helper could silently
    # win and the CANN build would mix new ExactScan code with an old
    # GM/local ABI implementation.
    KVAF_STAGE_DIR="${OP_DEV}/op_kernel/kvaf_device"
    mkdir -p "${KVAF_STAGE_DIR}"
    cp -f "${KVAF_FUSED_HEADER}" "${OP_DEV}/op_kernel/kvaf_fused_full_get_device.h"
    cp -f "${KVAF_FUSED_SOURCE}" "${OP_DEV}/op_kernel/execute_batch_get_aiv.asc"
    cp -f "${KVAF_ABI_HEADER}" "${OP_DEV}/op_kernel/batch_get_abi.h"
    cp -f "${KVAF_FUSED_HEADER}" "${KVAF_STAGE_DIR}/kvaf_fused_full_get_device.h"
    cp -f "${KVAF_FUSED_SOURCE}" "${KVAF_STAGE_DIR}/execute_batch_get_aiv.asc"
    cp -f "${KVAF_ABI_HEADER}" "${KVAF_STAGE_DIR}/batch_get_abi.h"
    for staged_pair in \
        "${KVAF_FUSED_HEADER}:${KVAF_STAGE_DIR}/kvaf_fused_full_get_device.h" \
        "${KVAF_FUSED_SOURCE}:${KVAF_STAGE_DIR}/execute_batch_get_aiv.asc" \
        "${KVAF_ABI_HEADER}:${KVAF_STAGE_DIR}/batch_get_abi.h"; do
        KVAF_SOURCE_FILE="${staged_pair%%:*}"
        KVAF_STAGED_FILE="${staged_pair#*:}"
        if ! cmp -s "${KVAF_SOURCE_FILE}" "${KVAF_STAGED_FILE}"; then
            echo "[nanovllm_a5_ops] ERROR: staged KVAF relative include closure differs from selected source." >&2
            echo "[nanovllm_a5_ops] source=${KVAF_SOURCE_FILE} staged=${KVAF_STAGED_FILE}" >&2
            exit 2
        fi
    done
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
    echo "[nanovllm_a5_ops] ${KVAF_CONSUMER_NAMES[*]} KVAF GM/local ABI bridge: head=${KVAF_FUSED_HEAD} verified=1"
    echo "[nanovllm_a5_ops] ${KVAF_CONSUMER_NAMES[*]} KVAF relative include closure: ${KVAF_STAGE_DIR} verified=1"
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
    elif [[ "${op_name}" == "A5ScoreReceiptCandidateExecutor" ]]; then
        op_spec="${ROOT}/a5_score_receipt_candidate_executor/ops.json"
    elif [[ "${op_name}" == "A5ScoreReceiptSpeculativeExecutor" ]]; then
        op_spec="${ROOT}/a5_score_receipt_speculative_executor/ops.json"
    elif [[ "${op_name}" == "A5P8ForwardOwnedExecutor" || \
            "${op_name}" == "A5P8ForwardControlAdvance" || \
            "${op_name}" == "A5P8ArenaAdmissionGate" ]]; then
        op_spec="${ROOT}/a5_p8_forward_owned_executor/ops.json"
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

# In the strict P1 build, msopgen first writes its own host OpDef stub and
# the checked-in ExactScan host implementation is overlaid immediately after
# this block.  Depending on the CANN generator revision, that stub may use a
# different filename from our implementation.  Leaving it in place then
# emits two OP_ADD(A5ExactScanSpeculativeGet) registrations and produces two
# same-name sections in aic-*-ops-info.ini.  Remove only the generated
# registration source while the project contains *only* msopgen output; the
# checked-in host source is copied in the next block.  This deliberately does
# not run for a normal/multi-op build, where the generated sources can be
# shared by legacy operators.
if [[ "${STAGE_EXACT_SCAN_SPECULATIVE_GET_ONLY}" == "1" ]]; then
    P1_GENERATED_OPDEF_STUBS=()
    while IFS= read -r -d '' P1_GENERATED_HOST_SOURCE; do
        if grep -qE 'OP_ADD[[:space:]]*\([[:space:]]*A5ExactScanSpeculativeGet[[:space:]]*\)' \
                "${P1_GENERATED_HOST_SOURCE}"; then
            P1_GENERATED_OPDEF_STUBS+=("${P1_GENERATED_HOST_SOURCE}")
        fi
    done < <(find "${GENERATED}/op_host" -type f \
        \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print0)

    if [[ "${#P1_GENERATED_OPDEF_STUBS[@]}" -ne 1 ]]; then
        echo "[nanovllm_a5_ops] ERROR: expected exactly one msopgen A5ExactScanSpeculativeGet host stub; found ${#P1_GENERATED_OPDEF_STUBS[@]}." >&2
        printf '[nanovllm_a5_ops] P1 generated OpDef stub: %s\n' \
            "${P1_GENERATED_OPDEF_STUBS[@]}" >&2
        exit 1
    fi
    rm -f "${P1_GENERATED_OPDEF_STUBS[0]}"
    echo "[nanovllm_a5_ops] P1 removed generated host OpDef stub: ${P1_GENERATED_OPDEF_STUBS[0]}"
fi

# P7.0 also overlays one checked-in host OpDef onto msopgen's generated
# project.  Remove exactly its generated stub before the overlay.  Keeping
# both sources would register the same operator twice and let CANN emit two
# same-name INI sections, which is a build-closure error rather than an INI
# formatting issue.  This guard is deliberately separate from P3/P1 so the
# P7.0 source closure stays independent of their implementation choices.
if [[ "${STAGE_SCORE_RECEIPT_CANDIDATE_EXECUTOR_ONLY}" == "1" ]]; then
    P7_GENERATED_OPDEF_STUBS=()
    while IFS= read -r -d '' P7_GENERATED_HOST_SOURCE; do
        if grep -qE 'OP_ADD[[:space:]]*\([[:space:]]*A5ScoreReceiptCandidateExecutor[[:space:]]*\)' \
                "${P7_GENERATED_HOST_SOURCE}"; then
            P7_GENERATED_OPDEF_STUBS+=("${P7_GENERATED_HOST_SOURCE}")
        fi
    done < <(find "${GENERATED}/op_host" -type f \
        \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print0)

    if [[ "${#P7_GENERATED_OPDEF_STUBS[@]}" -ne 1 ]]; then
        echo "[nanovllm_a5_ops] ERROR: expected exactly one msopgen A5ScoreReceiptCandidateExecutor host stub; found ${#P7_GENERATED_OPDEF_STUBS[@]}." >&2
        printf '[nanovllm_a5_ops] P7 generated OpDef stub: %s\n' \
            "${P7_GENERATED_OPDEF_STUBS[@]}" >&2
        exit 1
    fi
    rm -f "${P7_GENERATED_OPDEF_STUBS[0]}"
    echo "[nanovllm_a5_ops] P7 removed generated host OpDef stub: ${P7_GENERATED_OPDEF_STUBS[0]}"
fi

# P7.1 overlays the checked-in host contract after msopgen creates its
# scaffold.  Remove the one generated registration first; otherwise CANN
# sees two OP_ADD registrations and emits duplicate INI sections.
if [[ "${STAGE_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_ONLY}" == "1" ]]; then
    P7_1_GENERATED_OPDEF_STUBS=()
    while IFS= read -r -d '' P7_1_GENERATED_HOST_SOURCE; do
        if grep -qE 'OP_ADD[[:space:]]*\([[:space:]]*A5ScoreReceiptSpeculativeExecutor[[:space:]]*\)' \
                "${P7_1_GENERATED_HOST_SOURCE}"; then
            P7_1_GENERATED_OPDEF_STUBS+=("${P7_1_GENERATED_HOST_SOURCE}")
        fi
    done < <(find "${GENERATED}/op_host" -type f \
        \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print0)
    if [[ "${#P7_1_GENERATED_OPDEF_STUBS[@]}" -ne 1 ]]; then
        echo "[nanovllm_a5_ops] ERROR: expected exactly one msopgen A5ScoreReceiptSpeculativeExecutor host stub; found ${#P7_1_GENERATED_OPDEF_STUBS[@]}." >&2
        printf '[nanovllm_a5_ops] P7.1 generated OpDef stub: %s\n' \
            "${P7_1_GENERATED_OPDEF_STUBS[@]}" >&2
        exit 1
    fi
    rm -f "${P7_1_GENERATED_OPDEF_STUBS[0]}"
    echo "[nanovllm_a5_ops] P7.1 removed generated host OpDef stub: ${P7_1_GENERATED_OPDEF_STUBS[0]}"
fi

# P8 overlays three checked-in graph nodes: the one-time arena-admission
# control node, forward-owned transport executor, and graph-visible
# phase-control advance node.  Remove exactly the generated stubs
# generated registration stubs while the project still holds msopgen output;
# leaving either one would create duplicate OP_ADD registrations and same-name
# CANN INI sections.  This is a source-closure failure, never an INI issue to
# patch after packaging.
if [[ "${STAGE_P8_FORWARD_OWNED_EXECUTOR_ONLY}" == "1" ]]; then
    for P8_OVERLAY_OP in A5P8ForwardOwnedExecutor A5P8ForwardControlAdvance A5P8ArenaAdmissionGate; do
        P8_GENERATED_OPDEF_STUBS=()
        while IFS= read -r -d '' P8_GENERATED_HOST_SOURCE; do
            if grep -qE "OP_ADD[[:space:]]*\\([[:space:]]*${P8_OVERLAY_OP}[[:space:]]*\\)" \
                    "${P8_GENERATED_HOST_SOURCE}"; then
                P8_GENERATED_OPDEF_STUBS+=("${P8_GENERATED_HOST_SOURCE}")
            fi
        done < <(find "${GENERATED}/op_host" -type f \
            \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print0)
        if [[ "${#P8_GENERATED_OPDEF_STUBS[@]}" -ne 1 ]]; then
            echo "[nanovllm_a5_ops] ERROR: expected exactly one msopgen ${P8_OVERLAY_OP} host stub; found ${#P8_GENERATED_OPDEF_STUBS[@]}." >&2
            printf '[nanovllm_a5_ops] P8 generated OpDef stub: %s\n' \
                "${P8_GENERATED_OPDEF_STUBS[@]}" >&2
            exit 1
        fi
        rm -f "${P8_GENERATED_OPDEF_STUBS[0]}"
        echo "[nanovllm_a5_ops] P8 removed generated ${P8_OVERLAY_OP} host stub: ${P8_GENERATED_OPDEF_STUBS[0]}"
    done
fi

# msopgen owns the CMake scaffold. Checked-in Host/AscendC implementation
# sources overwrite only generated source stubs; directory structure is kept.
cp -a "${OP_DEV}/op_host/." "${GENERATED}/op_host/"
cp -a "${OP_DEV}/op_kernel/." "${GENERATED}/op_kernel/"
rm -f "${GENERATED}/op_host/lightning_indexer_decode_update_a5_def.cpp"

# `build.sh` compiles from GENERATED, rather than from OP_DEV.  Prove that
# the exact KVAF header selected above survived the final overlay at the
# relative path used by the source-derived LI hook.  This makes a future
# staging/order regression fail before CANN starts compiling kernels.
if [[ "${NEEDS_P1_DENSE_KVAF}" == "1" || \
      "${NEEDS_P3_POST_KVAF}" == "1" || \
      "${NEEDS_P7_1_STREAMING_KVAF}" == "1" || \
      "${NEEDS_P8_FORWARD_OWNED_KVAF}" == "1" ]]; then
    KVAF_GENERATED_STAGE_DIR="${GENERATED}/op_kernel/kvaf_device"
    for generated_pair in \
        "${KVAF_FUSED_HEADER}:${KVAF_GENERATED_STAGE_DIR}/kvaf_fused_full_get_device.h" \
        "${KVAF_FUSED_SOURCE}:${KVAF_GENERATED_STAGE_DIR}/execute_batch_get_aiv.asc" \
        "${KVAF_ABI_HEADER}:${KVAF_GENERATED_STAGE_DIR}/batch_get_abi.h"; do
        KVAF_SOURCE_FILE="${generated_pair%%:*}"
        KVAF_GENERATED_FILE="${generated_pair#*:}"
        if ! cmp -s "${KVAF_SOURCE_FILE}" "${KVAF_GENERATED_FILE}"; then
            echo "[nanovllm_a5_ops] ERROR: generated KVAF relative include closure differs from selected source." >&2
            echo "[nanovllm_a5_ops] source=${KVAF_SOURCE_FILE} generated=${KVAF_GENERATED_FILE}" >&2
            exit 2
        fi
    done
    echo "[nanovllm_a5_ops] ${KVAF_CONSUMER_NAMES[*]} generated KVAF relative include closure: ${KVAF_GENERATED_STAGE_DIR} verified=1"
fi

# The P3 build-mode header is generated after the source-derived closure has
# been staged, then overlaid onto msopgen's generated project.
# The source-derived vector header reaches it through `../` from arch35, so
# validate that exact final path before invoking CANN. This turns a missing
# staging overlay into an immediate, concise build error rather than an opaque
# device-compiler include failure several minutes later.
if [[ "${STAGE_STREAMING_QUANT_LI_GET_ONLY}" == "1" ]]; then
    P3_GENERATED_MODE_HEADER="${GENERATED}/op_kernel/a5_streaming_quant_li_get_hook_mode.h"
    if [[ ! -f "${P3_GENERATED_MODE_HEADER}" ]] || \
       ! grep -Fqx "#define A5_STREAMING_LI_GET_PUBLISH_MODE ${STREAMING_LI_PUBLISH_MODE}" \
            "${P3_GENERATED_MODE_HEADER}" || \
       ! grep -Fqx "#define A5_STREAMING_LI_GET_INLINE_MODE ${STREAMING_LI_INLINE_MODE}" \
            "${P3_GENERATED_MODE_HEADER}" || \
       ! grep -Fqx "#define A5_STREAMING_LI_GET_POST_CLOSURE_MODE ${STREAMING_LI_POST_CLOSURE_MODE}" \
            "${P3_GENERATED_MODE_HEADER}" || \
       ! grep -Fqx "#define A5_STREAMING_LI_GET_POST_SCHEDULER_MODE ${STREAMING_LI_POST_SCHEDULER_MODE}" \
            "${P3_GENERATED_MODE_HEADER}" || \
       ! grep -Fq 'streaming LI staged full-post closure derivation mismatch' \
            "${P3_GENERATED_MODE_HEADER}" || \
       ! grep -Fq 'streaming LI staged CandidateSeed derivation mismatch' \
            "${P3_GENERATED_MODE_HEADER}" || \
       grep -q 'A5_STREAMING_LI_GET_ENABLE_HOOK' \
            "${P3_GENERATED_MODE_HEADER}"; then
        echo "[nanovllm_a5_ops] ERROR: generated P3 mode header is missing or invalid." >&2
        echo "[nanovllm_a5_ops] expected=${P3_GENERATED_MODE_HEADER}" >&2
        exit 2
    fi
    if [[ "${STREAMING_LI_INLINE_MODE}" == "0" ]]; then
        if [[ -e "${GENERATED}/op_kernel/kvaf_device" || \
              -e "${GENERATED}/op_kernel/simt_api" || \
              -e "${GENERATED}/op_kernel/kvaf_fused_full_get_device.h" || \
              -e "${GENERATED}/op_kernel/execute_batch_get_aiv.asc" || \
              -e "${GENERATED}/op_kernel/batch_get_abi.h" || \
              -e "${GENERATED}/op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h" || \
              -e "${GENERATED}/op_kernel/arch35/a5_streaming_quant_li_get_tile_audit.h" ]]; then
            echo "[nanovllm_a5_ops] ERROR: generated P3 inline=off project contains a KVAF/SIMT closure." >&2
            exit 2
        fi
        echo "[nanovllm_a5_ops] P3 generated KVAF/SIMT closure: absent verified=1"
    elif [[ "${STREAMING_LI_INLINE_MODE}" == "1" ]]; then
        if [[ ! -s "${GENERATED}/op_kernel/arch35/a5_streaming_quant_li_get_tile_audit.h" || \
              -e "${GENERATED}/op_kernel/kvaf_device" || \
              -e "${GENERATED}/op_kernel/simt_api" || \
              -e "${GENERATED}/op_kernel/kvaf_fused_full_get_device.h" || \
              -e "${GENERATED}/op_kernel/execute_batch_get_aiv.asc" || \
              -e "${GENERATED}/op_kernel/batch_get_abi.h" || \
              -e "${GENERATED}/op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h" ]]; then
            echo "[nanovllm_a5_ops] ERROR: generated P3 inline=audit contains a transport closure." >&2
            exit 2
        fi
        # Match actual transport symbols/includes, not prose comments.  The
        # audit header is allowed to document why it has no transport path.
        if grep -Eq 'TryAcquireKvafStreamingBatchGet|BeginKvafStreamingBatchGet|PublishKvafStreamingBatchGetDescriptorScratch|TryRetireKvafStreamingBatchGet|FinishKvafStreamingBatchGet|kvaf_fused_get::|kvaf_fused_full_get_device|execute_batch_get_aiv|batch_get_abi|#include[[:space:]]*[<"].*(kvaf_device|simt_api)' \
                "${GENERATED}/op_kernel/arch35/a5_streaming_quant_li_get_tile_audit.h"; then
            echo "[nanovllm_a5_ops] ERROR: generated P3 audit observer references a transport symbol." >&2
            exit 2
        fi
        echo "[nanovllm_a5_ops] P3 generated audit closure: score-only verified=1"
    else
        if [[ "${STREAMING_LI_POST_CLOSURE_MODE}" == "0" ]]; then
            if [[ -e "${GENERATED}/op_kernel/kvaf_device" || \
                  -e "${GENERATED}/op_kernel/simt_api" || \
                  -e "${GENERATED}/op_kernel/kvaf_fused_full_get_device.h" || \
                  -e "${GENERATED}/op_kernel/execute_batch_get_aiv.asc" || \
                  -e "${GENERATED}/op_kernel/batch_get_abi.h" || \
                  -e "${GENERATED}/op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h" || \
                  -e "${GENERATED}/op_kernel/arch35/a5_streaming_quant_li_get_tile_audit.h" ]]; then
                echo "[nanovllm_a5_ops] ERROR: generated P3 inline=post empty_shell contains a transport/post-hook closure." >&2
                exit 2
            fi
            echo "[nanovllm_a5_ops] P3 generated KVAF/SIMT/post-hook closure: absent (inline=post empty_shell) verified=1"
        else
        for required_file in \
            "${GENERATED}/op_kernel/kvaf_device/kvaf_fused_full_get_device.h" \
            "${GENERATED}/op_kernel/simt_api/device_atomic_functions.h" \
            "${GENERATED}/op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h"; do
            if [[ ! -s "${required_file}" ]]; then
                echo "[nanovllm_a5_ops] ERROR: generated P3 inline closure is incomplete: ${required_file}" >&2
                exit 2
            fi
        done
        if [[ -e "${GENERATED}/op_kernel/arch35/a5_streaming_quant_li_get_tile_audit.h" ]]; then
            echo "[nanovllm_a5_ops] ERROR: generated P3 inline=post contains the audit observer." >&2
            exit 2
        fi
        echo "[nanovllm_a5_ops] P3 generated KVAF/SIMT closure: post-only verified=1"
        fi
    fi
    echo "[nanovllm_a5_ops] P3 generated mode header: ${P3_GENERATED_MODE_HEADER} publication=${STREAMING_LI_PUBLISH_MODE_NAME}/${STREAMING_LI_PUBLISH_MODE} inline=${STREAMING_LI_INLINE_MODE_NAME}/${STREAMING_LI_INLINE_MODE} post_closure=${STREAMING_LI_POST_CLOSURE_MODE_NAME}/${STREAMING_LI_POST_CLOSURE_MODE} post_scheduler=${STREAMING_LI_POST_SCHEDULER_MODE_NAME}/${STREAMING_LI_POST_SCHEDULER_MODE} no_fixed_get_core=1 verified=1"
fi

# `msopgen` creates an all-in-one host source named
# `a5_streaming_quant_li_get.cpp`.  P3 deliberately replaces that generated
# implementation with the source-derived C8 host closure, whose OpDef lives
# in `a5_streaming_quant_li_get_def.cpp`.  Unlike the device entry point,
# those filenames differ, so a normal overlay leaves both OP_ADD registrations
# in the generated project.  CANN's host-source discovery then emits two
# sections with the same name in aic-*-ops-info.ini.
#
# Keep the generated CMake scaffold and remove only the generated P3 host
# stub.  The checked-in _def/_infershape/_tiling sources are the one coherent
# P3 host implementation.  Fail before invoking CANN if staging ever drifts
# back to zero or multiple registrations; changing the generated INI itself
# would merely hide an ABI/build-closure error.
if [[ "${STAGE_STREAMING_QUANT_LI_GET_ONLY}" == "1" ]]; then
    P3_GENERATED_HOST_STUB="${GENERATED}/op_host/a5_streaming_quant_li_get.cpp"
    if [[ ! -f "${P3_GENERATED_HOST_STUB}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: expected msopgen P3 host stub is missing: ${P3_GENERATED_HOST_STUB}" >&2
        exit 1
    fi
    rm -f "${P3_GENERATED_HOST_STUB}"

    P3_OPDEF_SOURCES=()
    while IFS= read -r -d '' P3_HOST_SOURCE; do
        if grep -qE 'OP_ADD[[:space:]]*\([[:space:]]*A5StreamingQuantLiGet[[:space:]]*\)' \
                "${P3_HOST_SOURCE}"; then
            P3_OPDEF_SOURCES+=("${P3_HOST_SOURCE}")
        fi
    done < <(find "${GENERATED}/op_host" -type f -name '*.cpp' -print0)

    if [[ "${#P3_OPDEF_SOURCES[@]}" -ne 1 ]]; then
        echo "[nanovllm_a5_ops] ERROR: P3 requires exactly one A5StreamingQuantLiGet OpDef; found ${#P3_OPDEF_SOURCES[@]}." >&2
        printf '[nanovllm_a5_ops] P3 OpDef source: %s\n' "${P3_OPDEF_SOURCES[@]}" >&2
        exit 1
    fi
    echo "[nanovllm_a5_ops] P3 host registration closure: ${P3_OPDEF_SOURCES[0]}"
fi

# P1 has one authoritative host implementation.  This is deliberately a
# post-overlay check: it proves both that the generated registration stub was
# removed above and that the checked-in implementation is the sole OP_ADD
# source seen by CANN.  Do not "fix" the generated INI after the fact: a
# duplicate there would otherwise hide a source/ABI closure error until a
# later build or runtime failure.
if [[ "${STAGE_EXACT_SCAN_SPECULATIVE_GET_ONLY}" == "1" ]]; then
    P1_EXPECTED_OPDEF_SOURCE="${GENERATED}/op_host/a5_exact_scan_speculative_get.cpp"
    if [[ ! -f "${P1_EXPECTED_OPDEF_SOURCE}" ]] || \
       ! cmp -s "${ROOT}/a5_exact_scan_speculative_get/op_host/a5_exact_scan_speculative_get.cpp" \
                 "${P1_EXPECTED_OPDEF_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P1 checked-in ExactScan host implementation did not survive the generated-project overlay." >&2
        echo "[nanovllm_a5_ops] expected=${P1_EXPECTED_OPDEF_SOURCE}" >&2
        exit 1
    fi

    P1_OPDEF_SOURCES=()
    while IFS= read -r -d '' P1_HOST_SOURCE; do
        if grep -qE 'OP_ADD[[:space:]]*\([[:space:]]*A5ExactScanSpeculativeGet[[:space:]]*\)' \
                "${P1_HOST_SOURCE}"; then
            P1_OPDEF_SOURCES+=("${P1_HOST_SOURCE}")
        fi
    done < <(find "${GENERATED}/op_host" -type f \
        \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print0)

    if [[ "${#P1_OPDEF_SOURCES[@]}" -ne 1 || \
          "${P1_OPDEF_SOURCES[0]:-}" != "${P1_EXPECTED_OPDEF_SOURCE}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: P1 requires exactly one A5ExactScanSpeculativeGet OpDef from the checked-in host source; found ${#P1_OPDEF_SOURCES[@]}." >&2
        printf '[nanovllm_a5_ops] P1 OpDef source: %s\n' \
            "${P1_OPDEF_SOURCES[@]}" >&2
        exit 1
    fi
    # P5 follows CANN's standard C++ POD tiling contract.  The exact same
    # type is registered by the device entry point and acquired by the host
    # through GetTilingData<T>().  Do not accept the legacy macro/raw-buffer
    # route here: direct ACLNN would then enter the host callback with a
    # zero-capacity payload even though the device kernel uses
    # REGISTER_TILING_DEFAULT.
    P1_DEVICE_TILING_HEADER="${GENERATED}/op_kernel/a5_exact_scan_speculative_get_tiling.h"
    P1_DEVICE_TILING_SOURCE="${GENERATED}/op_kernel/a5_exact_scan_speculative_get.cpp"
    if ! grep -q 'GetTilingData<' "${P1_EXPECTED_OPDEF_SOURCE}" || \
       ! grep -q 'A5ExactScanSpeculativeGetTilingData' "${P1_EXPECTED_OPDEF_SOURCE}" || \
       grep -qE 'GetRawTilingData|SaveToBuffer|REGISTER_TILING_DATA_CLASS|BEGIN_TILING_DATA_DEF' \
               "${P1_EXPECTED_OPDEF_SOURCE}" || \
       ! grep -q 'struct A5ExactScanSpeculativeGetTilingData' \
               "${P1_DEVICE_TILING_HEADER}" || \
       ! grep -q 'REGISTER_TILING_DEFAULT(A5ExactScanSpeculativeGetTilingData)' \
               "${P1_DEVICE_TILING_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P1 standard POD tiling closure did not survive the generated-project overlay." >&2
        echo "[nanovllm_a5_ops] host=${P1_EXPECTED_OPDEF_SOURCE}" >&2
        echo "[nanovllm_a5_ops] kernel_header=${P1_DEVICE_TILING_HEADER}" >&2
        echo "[nanovllm_a5_ops] kernel_source=${P1_DEVICE_TILING_SOURCE}" >&2
        exit 1
    fi
    echo "[nanovllm_a5_ops] P1 host registration closure: ${P1_OPDEF_SOURCES[0]} verified=1"
    echo "[nanovllm_a5_ops] P1 standard POD tiling closure: host=${P1_EXPECTED_OPDEF_SOURCE} kernel=${P1_DEVICE_TILING_SOURCE} verified=1"
fi

# P7.0 is deliberately a source-closed oracle: it consumes a C8-produced
# score workspace only after the publisher's established completion edge.  It
# must not pull C8 source, KVAF transport, SIMT helpers, or Post/CQ code into
# this OPP.  Enforce that property on the generated project, not merely the
# checked-in tree, so an accidental staging/overlay regression fails before a
# CANN build starts.
if [[ "${STAGE_SCORE_RECEIPT_CANDIDATE_EXECUTOR_ONLY}" == "1" ]]; then
    P7_EXPECTED_OPDEF_SOURCE="${GENERATED}/op_host/a5_score_receipt_candidate_executor.cpp"
    P7_DEVICE_TILING_HEADER="${GENERATED}/op_kernel/a5_score_receipt_candidate_executor_tiling.h"
    P7_DEVICE_TILING_SOURCE="${GENERATED}/op_kernel/a5_score_receipt_candidate_executor.cpp"
    if [[ ! -f "${P7_EXPECTED_OPDEF_SOURCE}" ]] || \
       ! cmp -s "${ROOT}/a5_score_receipt_candidate_executor/op_host/a5_score_receipt_candidate_executor.cpp" \
                 "${P7_EXPECTED_OPDEF_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P7 checked-in candidate-oracle host implementation did not survive the generated-project overlay." >&2
        echo "[nanovllm_a5_ops] expected=${P7_EXPECTED_OPDEF_SOURCE}" >&2
        exit 1
    fi

    P7_OPDEF_SOURCES=()
    while IFS= read -r -d '' P7_HOST_SOURCE; do
        if grep -qE 'OP_ADD[[:space:]]*\([[:space:]]*A5ScoreReceiptCandidateExecutor[[:space:]]*\)' \
                "${P7_HOST_SOURCE}"; then
            P7_OPDEF_SOURCES+=("${P7_HOST_SOURCE}")
        fi
    done < <(find "${GENERATED}/op_host" -type f \
        \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print0)

    if [[ "${#P7_OPDEF_SOURCES[@]}" -ne 1 || \
          "${P7_OPDEF_SOURCES[0]:-}" != "${P7_EXPECTED_OPDEF_SOURCE}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: P7 requires exactly one A5ScoreReceiptCandidateExecutor OpDef from the checked-in host source; found ${#P7_OPDEF_SOURCES[@]}." >&2
        printf '[nanovllm_a5_ops] P7 OpDef source: %s\n' \
            "${P7_OPDEF_SOURCES[@]}" >&2
        exit 1
    fi
    if ! grep -q 'GetTilingData<' "${P7_EXPECTED_OPDEF_SOURCE}" || \
       ! grep -q 'A5ScoreReceiptCandidateExecutorTilingData' \
               "${P7_EXPECTED_OPDEF_SOURCE}" || \
       ! grep -q 'struct A5ScoreReceiptCandidateExecutorTilingData' \
               "${P7_DEVICE_TILING_HEADER}" || \
       ! grep -q 'REGISTER_TILING_DEFAULT(A5ScoreReceiptCandidateExecutorTilingData)' \
               "${P7_DEVICE_TILING_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P7 standard POD tiling closure did not survive the generated-project overlay." >&2
        echo "[nanovllm_a5_ops] host=${P7_EXPECTED_OPDEF_SOURCE}" >&2
        echo "[nanovllm_a5_ops] kernel_header=${P7_DEVICE_TILING_HEADER}" >&2
        echo "[nanovllm_a5_ops] kernel_source=${P7_DEVICE_TILING_SOURCE}" >&2
        exit 1
    fi
    if grep -REn \
        '#[[:space:]]*include[[:space:]]*[<"][^>"]*(kvaf|simt_api|quant_lightning|a5_streaming_quant)|kvaf_fused_get::|\b(TryAcquire|Begin|TryRetire|Finish)KvafStreamingBatchGet\b|\b(DirectHccp|DirectHCCP|URMA|urma_)\b' \
        "${P7_EXPECTED_OPDEF_SOURCE}" "${P7_DEVICE_TILING_HEADER}" \
        "${P7_DEVICE_TILING_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P7.0 generated source closure unexpectedly contains C8/KVAF/SIMT/Post/CQ code." >&2
        exit 1
    fi
    echo "[nanovllm_a5_ops] P7 host registration closure: ${P7_OPDEF_SOURCES[0]} verified=1"
    echo "[nanovllm_a5_ops] P7 standard POD tiling closure: host=${P7_EXPECTED_OPDEF_SOURCE} kernel=${P7_DEVICE_TILING_SOURCE} verified=1"
    echo "[nanovllm_a5_ops] P7 source closure: c8=absent kvaf=absent simt=absent post_cq=absent verified=1"
fi

# P7.1 deliberately stages KVAF Direct-HCCP/SIMT, but it must remain entirely
# independent of C8 source and of a fixed physical Get core.  Verify the
# final generated project—not only the checked-in source—before invoking the
# device compiler so stale overlay artifacts cannot masquerade as this build.
if [[ "${STAGE_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_ONLY}" == "1" ]]; then
    P7_1_EXPECTED_OPDEF_SOURCE="${GENERATED}/op_host/a5_score_receipt_speculative_executor.cpp"
    P7_1_DEVICE_TILING_HEADER="${GENERATED}/op_kernel/a5_score_receipt_speculative_executor_tiling.h"
    P7_1_DEVICE_TILING_SOURCE="${GENERATED}/op_kernel/a5_score_receipt_speculative_executor.cpp"
    if [[ ! -f "${P7_1_EXPECTED_OPDEF_SOURCE}" ]] || \
       ! cmp -s "${ROOT}/a5_score_receipt_speculative_executor/op_host/a5_score_receipt_speculative_executor.cpp" \
                 "${P7_1_EXPECTED_OPDEF_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P7.1 checked-in speculative-executor host implementation did not survive the generated-project overlay." >&2
        echo "[nanovllm_a5_ops] expected=${P7_1_EXPECTED_OPDEF_SOURCE}" >&2
        exit 1
    fi
    P7_1_OPDEF_SOURCES=()
    while IFS= read -r -d '' P7_1_HOST_SOURCE; do
        if grep -qE 'OP_ADD[[:space:]]*\([[:space:]]*A5ScoreReceiptSpeculativeExecutor[[:space:]]*\)' \
                "${P7_1_HOST_SOURCE}"; then
            P7_1_OPDEF_SOURCES+=("${P7_1_HOST_SOURCE}")
        fi
    done < <(find "${GENERATED}/op_host" -type f \
        \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print0)
    if [[ "${#P7_1_OPDEF_SOURCES[@]}" -ne 1 || \
          "${P7_1_OPDEF_SOURCES[0]:-}" != "${P7_1_EXPECTED_OPDEF_SOURCE}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: P7.1 requires exactly one A5ScoreReceiptSpeculativeExecutor OpDef from the checked-in host source; found ${#P7_1_OPDEF_SOURCES[@]}." >&2
        printf '[nanovllm_a5_ops] P7.1 OpDef source: %s\n' \
            "${P7_1_OPDEF_SOURCES[@]}" >&2
        exit 1
    fi
    if ! grep -q 'GetTilingData<' "${P7_1_EXPECTED_OPDEF_SOURCE}" || \
       ! grep -q 'A5ScoreReceiptSpeculativeExecutorTilingData' \
               "${P7_1_EXPECTED_OPDEF_SOURCE}" || \
       ! grep -q 'struct A5ScoreReceiptSpeculativeExecutorTilingData' \
               "${P7_1_DEVICE_TILING_HEADER}" || \
       ! grep -q 'REGISTER_TILING_DEFAULT(A5ScoreReceiptSpeculativeExecutorTilingData)' \
               "${P7_1_DEVICE_TILING_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P7.1 standard POD tiling closure did not survive the generated-project overlay." >&2
        exit 1
    fi
    for P7_1_REQUIRED_TOKEN in \
        'TryAcquireKvafStreamingBatchGetQueueFromAicore' \
        'LoadKvafStreamingBatchGetQueueTemplate' \
        'PublishKvafStreamingBatchGetDescriptorScratch' \
        'BeginKvafStreamingBatchGet' \
        'TryRetireKvafStreamingBatchGet' \
        'FinishKvafStreamingBatchGet' \
        'GetKvafStreamingBatchGetReady' \
        'ReleaseKvafStreamingBatchGetQueueFromAicore' \
        'DataSyncBarrier<MemDsbT::DDR>' \
        'KERNEL_TYPE_AIV_ONLY'; do
        if ! grep -Fq "${P7_1_REQUIRED_TOKEN}" "${P7_1_DEVICE_TILING_SOURCE}"; then
            echo "[nanovllm_a5_ops] ERROR: P7.1 generated source closure misses ${P7_1_REQUIRED_TOKEN}." >&2
            exit 1
        fi
    done
    # The two bounded microbatch constants intentionally live in the shared
    # tiling POD header, not in the device .cpp.  Check their actual source
    # location so this gate cannot reject a correct generated overlay.
    for P7_1_TILING_TOKEN in \
        'kA5ScoreReceiptExecutorMicrobatchCapacity' \
        'kA5ScoreReceiptExecutorMicrobatchSlots'; do
        if ! grep -Fq "${P7_1_TILING_TOKEN}" "${P7_1_DEVICE_TILING_HEADER}"; then
            echo "[nanovllm_a5_ops] ERROR: P7.1 generated tiling closure misses ${P7_1_TILING_TOKEN}." >&2
            exit 1
        fi
    done
    if grep -Fq 'KERNEL_TYPE_MIX_AIC_1_2' "${P7_1_DEVICE_TILING_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P7.1 must remain AIV_ONLY; MIX would reserve an unnecessary AIC lane." >&2
        exit 1
    fi
    if grep -REn \
        '#[[:space:]]*include[[:space:]]*[<"][^>"]*(quant_lightning|a5_streaming_quant|a5_official_c8)|\b(kGetLane|LI_CORE|GET_CORE|physical.*core)\b' \
        "${P7_1_EXPECTED_OPDEF_SOURCE}" "${P7_1_DEVICE_TILING_HEADER}" \
        "${P7_1_DEVICE_TILING_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P7.1 generated source closure unexpectedly contains C8 source or a fixed Get-core role." >&2
        exit 1
    fi
    echo "[nanovllm_a5_ops] P7.1 host registration closure: ${P7_1_OPDEF_SOURCES[0]} verified=1"
    echo "[nanovllm_a5_ops] P7.1 source closure: c8=absent fixed_get_core=absent kvaf=staged simt=staged post_cq=present verified=1"
fi

# P8 is the forward-owned lifecycle executor, not a modification of C8 and
# not a second P7.1 package.  The generated project is the compiler input, so
# prove its exact host/kernel/tiling/KVAF closure here.  In particular, a
# durable wave drain has to claim the GM record before rehydrating it; a build
# that silently falls back to the old queue-local lifecycle is rejected before
# any device compiler work begins.
if [[ "${STAGE_P8_FORWARD_OWNED_EXECUTOR_ONLY}" == "1" ]]; then
    P8_EXPECTED_OPDEF_SOURCE="${GENERATED}/op_host/a5_p8_forward_owned_executor.cpp"
    P8_DEVICE_TILING_HEADER="${GENERATED}/op_kernel/a5_p8_forward_owned_executor_tiling.h"
    P8_ARENA_TAG_HEADER="${GENERATED}/op_kernel/a5_p8_arena_tag.h"
    P8_DEVICE_TILING_SOURCE="${GENERATED}/op_kernel/a5_p8_forward_owned_executor.cpp"
    P8_ADVANCE_EXPECTED_OPDEF_SOURCE="${GENERATED}/op_host/a5_p8_forward_control_advance.cpp"
    P8_ADVANCE_DEVICE_TILING_HEADER="${GENERATED}/op_kernel/a5_p8_forward_control_advance_tiling.h"
    P8_ADVANCE_DEVICE_TILING_SOURCE="${GENERATED}/op_kernel/a5_p8_forward_control_advance.cpp"
    P8_GATE_EXPECTED_OPDEF_SOURCE="${GENERATED}/op_host/a5_p8_arena_admission_gate.cpp"
    P8_GATE_DEVICE_TILING_HEADER="${GENERATED}/op_kernel/a5_p8_arena_admission_gate_tiling.h"
    P8_GATE_DEVICE_TILING_SOURCE="${GENERATED}/op_kernel/a5_p8_arena_admission_gate.cpp"
    if [[ ! -f "${P8_EXPECTED_OPDEF_SOURCE}" ]] || \
       [[ ! -f "${P8_ARENA_TAG_HEADER}" ]] || \
       [[ ! -f "${P8_ADVANCE_EXPECTED_OPDEF_SOURCE}" ]] || \
       [[ ! -f "${P8_GATE_EXPECTED_OPDEF_SOURCE}" ]] || \
       ! cmp -s "${ROOT}/a5_p8_forward_owned_executor/op_host/a5_p8_forward_owned_executor.cpp" \
                 "${P8_EXPECTED_OPDEF_SOURCE}" || \
       ! cmp -s "${ROOT}/a5_p8_forward_owned_executor/op_kernel/a5_p8_forward_owned_executor.cpp" \
                 "${P8_DEVICE_TILING_SOURCE}" || \
       ! cmp -s "${ROOT}/a5_p8_forward_owned_executor/op_kernel/a5_p8_forward_owned_executor_tiling.h" \
                 "${P8_DEVICE_TILING_HEADER}" || \
       ! cmp -s "${ROOT}/a5_p8_forward_owned_executor/op_kernel/a5_p8_arena_tag.h" \
                 "${P8_ARENA_TAG_HEADER}" || \
       ! cmp -s "${ROOT}/a5_p8_forward_owned_executor/op_host/a5_p8_forward_control_advance.cpp" \
                 "${P8_ADVANCE_EXPECTED_OPDEF_SOURCE}" || \
       ! cmp -s "${ROOT}/a5_p8_forward_owned_executor/op_kernel/a5_p8_forward_control_advance.cpp" \
                 "${P8_ADVANCE_DEVICE_TILING_SOURCE}" || \
       ! cmp -s "${ROOT}/a5_p8_forward_owned_executor/op_kernel/a5_p8_forward_control_advance_tiling.h" \
                 "${P8_ADVANCE_DEVICE_TILING_HEADER}" || \
       ! cmp -s "${ROOT}/a5_p8_forward_owned_executor/op_host/a5_p8_arena_admission_gate.cpp" \
                 "${P8_GATE_EXPECTED_OPDEF_SOURCE}" || \
       ! cmp -s "${ROOT}/a5_p8_forward_owned_executor/op_kernel/a5_p8_arena_admission_gate.cpp" \
                 "${P8_GATE_DEVICE_TILING_SOURCE}" || \
       ! cmp -s "${ROOT}/a5_p8_forward_owned_executor/op_kernel/a5_p8_arena_admission_gate_tiling.h" \
                 "${P8_GATE_DEVICE_TILING_HEADER}"; then
        echo "[nanovllm_a5_ops] ERROR: P8 checked-in forward-owned source did not survive the generated-project overlay." >&2
        echo "[nanovllm_a5_ops] executor_host=${P8_EXPECTED_OPDEF_SOURCE} advance_host=${P8_ADVANCE_EXPECTED_OPDEF_SOURCE} gate_host=${P8_GATE_EXPECTED_OPDEF_SOURCE}" >&2
        exit 1
    fi

    P8_OPDEF_SOURCES=()
    while IFS= read -r -d '' P8_HOST_SOURCE; do
        if grep -qE 'OP_ADD[[:space:]]*\([[:space:]]*A5P8ForwardOwnedExecutor[[:space:]]*\)' \
                "${P8_HOST_SOURCE}"; then
            P8_OPDEF_SOURCES+=("${P8_HOST_SOURCE}")
        fi
    done < <(find "${GENERATED}/op_host" -type f \
        \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print0)
    if [[ "${#P8_OPDEF_SOURCES[@]}" -ne 1 || \
          "${P8_OPDEF_SOURCES[0]:-}" != "${P8_EXPECTED_OPDEF_SOURCE}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: P8 requires exactly one A5P8ForwardOwnedExecutor OpDef from the checked-in host source; found ${#P8_OPDEF_SOURCES[@]}." >&2
        printf '[nanovllm_a5_ops] P8 OpDef source: %s\n' \
            "${P8_OPDEF_SOURCES[@]}" >&2
        exit 1
    fi
    P8_ADVANCE_OPDEF_SOURCES=()
    while IFS= read -r -d '' P8_ADVANCE_HOST_SOURCE; do
        if grep -qE 'OP_ADD[[:space:]]*\([[:space:]]*A5P8ForwardControlAdvance[[:space:]]*\)' \
                "${P8_ADVANCE_HOST_SOURCE}"; then
            P8_ADVANCE_OPDEF_SOURCES+=("${P8_ADVANCE_HOST_SOURCE}")
        fi
    done < <(find "${GENERATED}/op_host" -type f \
        \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print0)
    if [[ "${#P8_ADVANCE_OPDEF_SOURCES[@]}" -ne 1 || \
          "${P8_ADVANCE_OPDEF_SOURCES[0]:-}" != "${P8_ADVANCE_EXPECTED_OPDEF_SOURCE}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: P8 requires exactly one A5P8ForwardControlAdvance OpDef from the checked-in host source; found ${#P8_ADVANCE_OPDEF_SOURCES[@]}." >&2
        printf '[nanovllm_a5_ops] P8 advance OpDef source: %s\n' \
            "${P8_ADVANCE_OPDEF_SOURCES[@]}" >&2
        exit 1
    fi
    P8_GATE_OPDEF_SOURCES=()
    while IFS= read -r -d '' P8_GATE_HOST_SOURCE; do
        if grep -qE 'OP_ADD[[:space:]]*\([[:space:]]*A5P8ArenaAdmissionGate[[:space:]]*\)' \
                "${P8_GATE_HOST_SOURCE}"; then
            P8_GATE_OPDEF_SOURCES+=("${P8_GATE_HOST_SOURCE}")
        fi
    done < <(find "${GENERATED}/op_host" -type f \
        \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) -print0)
    if [[ "${#P8_GATE_OPDEF_SOURCES[@]}" -ne 1 || \
          "${P8_GATE_OPDEF_SOURCES[0]:-}" != "${P8_GATE_EXPECTED_OPDEF_SOURCE}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: P8 requires exactly one A5P8ArenaAdmissionGate OpDef from the checked-in host source; found ${#P8_GATE_OPDEF_SOURCES[@]}." >&2
        printf '[nanovllm_a5_ops] P8 admission OpDef source: %s\n' \
            "${P8_GATE_OPDEF_SOURCES[@]}" >&2
        exit 1
    fi
    if ! grep -q 'GetTilingData<' "${P8_EXPECTED_OPDEF_SOURCE}" || \
       ! grep -q 'A5P8ForwardOwnedExecutorTilingData' \
               "${P8_EXPECTED_OPDEF_SOURCE}" || \
       ! grep -q 'struct A5P8ForwardOwnedExecutorTilingData' \
               "${P8_DEVICE_TILING_HEADER}" || \
       ! grep -q 'REGISTER_TILING_DEFAULT(A5P8ForwardOwnedExecutorTilingData)' \
               "${P8_DEVICE_TILING_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P8 standard POD tiling closure did not survive the generated-project overlay." >&2
        exit 1
    fi
    if ! grep -q 'GetTilingData<' "${P8_ADVANCE_EXPECTED_OPDEF_SOURCE}" || \
       ! grep -q 'A5P8ForwardControlAdvanceTilingData' \
               "${P8_ADVANCE_EXPECTED_OPDEF_SOURCE}" || \
       ! grep -q 'struct A5P8ForwardControlAdvanceTilingData' \
               "${P8_ADVANCE_DEVICE_TILING_HEADER}" || \
       ! grep -q 'REGISTER_TILING_DEFAULT(A5P8ForwardControlAdvanceTilingData)' \
               "${P8_ADVANCE_DEVICE_TILING_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P8 graph-control advance tiling closure did not survive the generated-project overlay." >&2
        exit 1
    fi
    if ! grep -q 'GetTilingData<' "${P8_GATE_EXPECTED_OPDEF_SOURCE}" || \
       ! grep -q 'A5P8ArenaAdmissionGateTilingData' \
               "${P8_GATE_EXPECTED_OPDEF_SOURCE}" || \
       ! grep -q 'struct A5P8ArenaAdmissionGateTilingData' \
               "${P8_GATE_DEVICE_TILING_HEADER}" || \
       ! grep -q 'REGISTER_TILING_DEFAULT(A5P8ArenaAdmissionGateTilingData)' \
               "${P8_GATE_DEVICE_TILING_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P8 one-time arena-admission gate tiling closure did not survive the generated-project overlay." >&2
        exit 1
    fi
    for P8_REQUIRED_TOKEN in \
        'LoadKvafP8ForwardControlFromGm' \
        'TryAcquireKvafStreamingBatchGetQueueFromAicore' \
        'BeginAndPublishKvafP8WaveFilling' \
        'ArmKvafP8WaveFillingForExternalWrites' \
        'ReleaseKvafP8FillingLeaseAfterProvenNoWork' \
        'PromoteKvafP8WaveFillingToPostIntent' \
        'PostKvafP8Wave' \
        'TryClaimKvafP8WaveDrainFromAicore' \
        'TryClaimKvafP8UnpostedWaveDrainFromAicore' \
        'TryRetireKvafP8Wave' \
        'FinishKvafP8Wave' \
        'PublishKvafP8PrivateTerminalRecords' \
        'ReleaseKvafP8WaveAfterPrivateReady' \
        'ReleaseKvafP8PendingPrivateRecordAfterNoPost' \
        'ReleaseKvafP8ClaimedUnpostedWaveAfterPrivateCleanup' \
        'ReleaseKvafP8PrivateRecordAfterQsfaTerminal' \
        'MarkKvafP8ForwardAbortFromAicore' \
        'QuarantineKvafP8WaveAtForwardTerminal' \
        'QuarantineKvafP8PrivateAtForwardTerminal' \
        'kA5P8PhaseForwardTerminal' \
        'DataSyncBarrier<MemDsbT::DDR>' \
        'KERNEL_TYPE_AIV_ONLY'; do
        if ! grep -Fq "${P8_REQUIRED_TOKEN}" "${P8_DEVICE_TILING_SOURCE}"; then
            echo "[nanovllm_a5_ops] ERROR: P8 generated forward-owned source closure misses ${P8_REQUIRED_TOKEN}." >&2
            exit 1
        fi
    done
    for P8_ADVANCE_REQUIRED_TOKEN in \
        'P8ForwardControlAdvanceSimtService' \
        'MarkKvafP8ForwardAbortFromAicore' \
        'kKvafP8ForwardPhaseQsfaTerminal' \
        'kKvafP8ForwardPhaseForwardTerminal' \
        'KERNEL_TYPE_AIV_ONLY'; do
        if ! grep -Fq "${P8_ADVANCE_REQUIRED_TOKEN}" "${P8_ADVANCE_DEVICE_TILING_SOURCE}"; then
            echo "[nanovllm_a5_ops] ERROR: P8 graph-control advance source closure misses ${P8_ADVANCE_REQUIRED_TOKEN}." >&2
            exit 1
        fi
    done
    for P8_GATE_REQUIRED_TOKEN in \
        'A5P8ArenaAdmissionGateTilingData' \
        'MarkKvafP8ForwardAbortFromAicore' \
        'IsKvafP8WaveRecordAllZero' \
        'kKvafP8WaveFree' \
        'KERNEL_TYPE_AIV_ONLY'; do
        if ! grep -Fq "${P8_GATE_REQUIRED_TOKEN}" "${P8_GATE_DEVICE_TILING_SOURCE}"; then
            echo "[nanovllm_a5_ops] ERROR: P8 arena-admission gate source closure misses ${P8_GATE_REQUIRED_TOKEN}." >&2
            exit 1
        fi
    done
    if grep -Fq 'ReleaseKvafP8UnpostedWave' "${P8_DEVICE_TILING_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P8 may not use the unposted-wave release path after durable PostOnly intent." >&2
        exit 1
    fi
    if grep -Fq 'ClearP8PrivateMappingSimt' "${P8_DEVICE_TILING_SOURCE}" || \
       grep -Fq 'MarkKvafP8ForwardAbort(' "${P8_DEVICE_TILING_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P8 generated source retained an unsafe local map-clear or non-atomic abort path." >&2
        exit 1
    fi
    if grep -Fq 'KERNEL_TYPE_MIX_AIC_1_2' "${P8_DEVICE_TILING_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P8 must remain AIV_ONLY; MIX would reserve an unnecessary AIC lane." >&2
        exit 1
    fi
    if grep -REn \
        '#[[:space:]]*include[[:space:]]*[<"][^>"]*(quant_lightning|a5_streaming_quant|a5_official_c8|a5_score_receipt_speculative_executor)|\b(kGetLane|LI_CORE|GET_CORE|physical.*core)\b' \
        "${P8_EXPECTED_OPDEF_SOURCE}" "${P8_DEVICE_TILING_HEADER}" \
        "${P8_DEVICE_TILING_SOURCE}" "${P8_ADVANCE_EXPECTED_OPDEF_SOURCE}" \
        "${P8_ADVANCE_DEVICE_TILING_HEADER}" "${P8_ADVANCE_DEVICE_TILING_SOURCE}" \
        "${P8_GATE_EXPECTED_OPDEF_SOURCE}" "${P8_GATE_DEVICE_TILING_HEADER}" \
        "${P8_GATE_DEVICE_TILING_SOURCE}"; then
        echo "[nanovllm_a5_ops] ERROR: P8 generated source closure unexpectedly contains C8/P7 code or a fixed physical Get-core role." >&2
        exit 1
    fi
    echo "[nanovllm_a5_ops] P8 host registration closure: executor=${P8_OPDEF_SOURCES[0]} advance=${P8_ADVANCE_EXPECTED_OPDEF_SOURCE} admission=${P8_GATE_EXPECTED_OPDEF_SOURCE} verified=1"
    echo "[nanovllm_a5_ops] P8 source closure: c8=absent p7=absent fixed_get_core=absent kvaf_v2=staged filling_durable=present atomic_abort=present arena_admission=present qsfa_verified_release=present forward_terminal=present graph_phase_advance=present simt=staged dynamic_queue_lease=present verified=1"
fi

if grep -Rqs --include='CMakeLists.txt' 'add_op_to_compiled_list' \
        "${GENERATED}/op_host" "${GENERATED}/op_kernel"; then
    echo "[nanovllm_a5_ops] ERROR: legacy add_op_to_compiled_list leaked into the CANN 9.1 generated project." >&2
    exit 1
fi

# CANN's packaging fallback is inconsistent with its dependency option: it
# receives CANN_3RD_LIB_PATH for FetchContent, then CPack later hard-codes
# <generated-project>/third_party/makeself.  Stage just these two scripts in
# the ignored generated project before build.sh runs.  Do not copy a build
# directory, OPP, or bridge: all compiled outputs below remain unique to this
# invocation.
if [[ -n "${CANN_3RD_LIB_PATH}" ]]; then
    STAGED_MAKESELF_DIR="${GENERATED}/third_party/makeself"
    mkdir -p "${STAGED_MAKESELF_DIR}"
    install -m 0755 "${CANN_3RD_LIB_PATH}/makeself/makeself.sh" \
        "${STAGED_MAKESELF_DIR}/makeself.sh"
    install -m 0644 "${CANN_3RD_LIB_PATH}/makeself/makeself-header.sh" \
        "${STAGED_MAKESELF_DIR}/makeself-header.sh"
    if [[ ! -x "${STAGED_MAKESELF_DIR}/makeself.sh" || \
          ! -r "${STAGED_MAKESELF_DIR}/makeself-header.sh" ]]; then
        echo "[nanovllm_a5_ops] ERROR: CPack makeself staging failed: ${STAGED_MAKESELF_DIR}." >&2
        exit 2
    fi
    echo "[nanovllm_a5_ops] staged CPack makeself: ${STAGED_MAKESELF_DIR}"
fi

pushd "${GENERATED}" >/dev/null
if [[ -n "${CANN_3RD_LIB_PATH}" ]]; then
    OPS_CPU_NUMBER="${BUILD_JOBS}" bash build.sh \
        --cann_3rd_lib_path="${CANN_3RD_LIB_PATH}"
else
    OPS_CPU_NUMBER="${BUILD_JOBS}" bash build.sh
fi
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

# ``opParaSize`` is the preallocated payload size observed by a direct ACLNN
# tiling callback.  The P5 kernel's shared POD is 84 bytes and CANN reserves
# an additional eight-byte runtime header, so a P5 package with opParaSize=8
# would deterministically recreate the zero-capacity failure before any HCCP
# or score-publisher code runs.  Inspect the installed artifact, not merely
# the staged source, so this catches a generator/packaging regression before
# a target test spends time initializing Direct-HCCP.
if [[ "${STAGE_EXACT_SCAN_SPECULATIVE_GET_ONLY}" == "1" ]]; then
    P1_METADATA_FILES=()
    while IFS= read -r -d '' P1_METADATA; do
        P1_METADATA_FILES+=("${P1_METADATA}")
    done < <(find "${VENDOR_DIR}" -type f -name '*.json' -print0)
    if [[ "${#P1_METADATA_FILES[@]}" -eq 0 ]]; then
        echo "[nanovllm_a5_ops] ERROR: P1 installed OPP has no JSON metadata." >&2
        echo "[nanovllm_a5_ops] vendor=${VENDOR_DIR}" >&2
        exit 1
    fi
    P1_OP_PARA_SIZE="$("${PYTHON_BIN}" - "${P1_METADATA_FILES[@]}" <<'PY'
import json
import os
import sys

# `binary_info_config.json` may carry the op identity in its filename, in a
# parent object, or beside the per-kernel `opParaSize` node.  Accept all three
# standard layouts; otherwise a healthy package could be rejected merely
# because CANN moved common metadata out of the leaf dictionary.
needles = (
    "a5_exact_scan_speculative_get",
    "a5exactscanspeculativeget",
)
matches = []

def walk(value, path, filename_matches):
    if isinstance(value, dict):
        encoded = json.dumps(value, sort_keys=True).lower()
        if "opParaSize" in value and (
                filename_matches or any(needle in encoded for needle in needles)):
            matches.append((path, value["opParaSize"]))
        for key, child in value.items():
            walk(child, f"{path}.{key}", filename_matches)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk(child, f"{path}[{index}]", filename_matches)

for filename in sys.argv[1:]:
    try:
        with open(filename, "r", encoding="utf-8") as handle:
            filename_matches = any(
                needle in os.path.basename(filename).lower()
                for needle in needles)
            walk(json.load(handle), filename, filename_matches)
    except (OSError, json.JSONDecodeError):
        pass

for path, value in matches:
    print(f"{path}={value}")
PY
)"
    if [[ -z "${P1_OP_PARA_SIZE}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: P1 installed OPP metadata has no A5ExactScanSpeculativeGet opParaSize." >&2
        printf '[nanovllm_a5_ops] metadata=%s\n' "${P1_METADATA_FILES[@]}" >&2
        exit 1
    fi
    if ! printf '%s\n' "${P1_OP_PARA_SIZE}" | \
            awk -F= '$NF + 0 >= 92 { found = 1 } END { exit found ? 0 : 1 }'; then
        echo "[nanovllm_a5_ops] ERROR: P1 installed OPP has insufficient tiling payload; expected opParaSize>=92." >&2
        printf '%s\n' "${P1_OP_PARA_SIZE}" >&2
        exit 1
    fi
    printf '[nanovllm_a5_ops] P1 installed tiling metadata: %s\n' "${P1_OP_PARA_SIZE}"
fi

for op_name in "${OP_NAMES[@]}"; do
    if ! find "${VENDOR_DIR}" -type f -name 'binary_info_config.json' -print0 |
        xargs -0 -r grep -q "${op_name}"; then
        echo "[nanovllm_a5_ops] ERROR: ${op_name} is absent from generated kernel metadata." >&2
        exit 1
    fi
done

# The source-derived P3 OPP is deliberately built in a fresh, isolated
# directory for every equivalence/audit run.  Its ACLNN bridge keeps a static
# resolver handle, so accepting an arbitrary old libcust_opapi.so here would
# make a later test appear to exercise the current source while actually
# executing an older binary. Stamp the installed vendor with the exact
# worktree revision, both compile-time modes, and fixed P3 ABI.
# The Python adapters read this before importing the P3 bridge and fail closed
# on any mismatch.
if [[ "${STAGE_STREAMING_QUANT_LI_GET_ONLY}" == "1" ]]; then
    P3_PROVENANCE_HEAD="$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || true)"
    if [[ ! "${P3_PROVENANCE_HEAD}" =~ ^[0-9a-f]{40}$ ]]; then
        echo "[nanovllm_a5_ops] ERROR: cannot record P3 OPP provenance; source HEAD is unavailable." >&2
        exit 1
    fi
    # A commit ID alone is insufficient evidence when a remote worktree has
    # locally modified tracked sources: CANN would compile those bytes while
    # the manifest still claimed the committed HEAD.  Untracked experiment
    # directories are deliberately ignored, but a tracked diff is a hard
    # build error for this mathematical-equivalence artifact.
    P3_PROVENANCE_DIRTY="$(git -C "${ROOT}" status --porcelain --untracked-files=no 2>/dev/null || true)"
    if [[ -n "${P3_PROVENANCE_DIRTY}" ]]; then
        echo "[nanovllm_a5_ops] ERROR: refuse P3 OPP build from a tracked-dirty source worktree." >&2
        echo "[nanovllm_a5_ops] source_root=${ROOT} source_head=${P3_PROVENANCE_HEAD}" >&2
        printf '%s\n' "${P3_PROVENANCE_DIRTY}" >&2
        exit 1
    fi
    if [[ "${STREAMING_LI_INLINE_MODE}" == "2" && \
          "${STREAMING_LI_POST_CLOSURE_MODE}" == "1" ]]; then
        P3_KVAF_CLOSURE=staged
        P3_KVAF_SOURCE_HEAD="${KVAF_FUSED_HEAD:-unavailable}"
    else
        P3_KVAF_CLOSURE=absent
        P3_KVAF_SOURCE_HEAD=not_applicable
    fi
    cat > "${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE" <<EOF
format=1
kind=source_derived_streaming_quant_li_get
source_root=${ROOT}
source_head=${P3_PROVENANCE_HEAD}
source_clean=1
publication_mode=${STREAMING_LI_PUBLISH_MODE_NAME}
publication_mode_value=${STREAMING_LI_PUBLISH_MODE}
inline_mode=${STREAMING_LI_INLINE_MODE_NAME}
inline_mode_value=${STREAMING_LI_INLINE_MODE}
post_closure_mode=${STREAMING_LI_POST_CLOSURE_MODE_NAME}
post_closure_mode_value=${STREAMING_LI_POST_CLOSURE_MODE}
post_scheduler_mode=${STREAMING_LI_POST_SCHEDULER_MODE_NAME}
post_scheduler_mode_value=${STREAMING_LI_POST_SCHEDULER_MODE}
kvaf_closure=${P3_KVAF_CLOSURE}
kvaf_source_head=${P3_KVAF_SOURCE_HEAD}
abi=24in_11out
operators=${OP_NAMES[*]}
EOF
    if ! grep -Fqx 'kind=source_derived_streaming_quant_li_get' \
            "${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE" || \
       ! grep -Fqx "source_head=${P3_PROVENANCE_HEAD}" \
            "${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE" || \
       ! grep -Fqx 'source_clean=1' \
            "${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE" || \
       ! grep -Fqx "publication_mode=${STREAMING_LI_PUBLISH_MODE_NAME}" \
            "${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE" || \
       ! grep -Fqx "publication_mode_value=${STREAMING_LI_PUBLISH_MODE}" \
            "${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE" || \
       ! grep -Fqx "inline_mode=${STREAMING_LI_INLINE_MODE_NAME}" \
            "${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE" || \
       ! grep -Fqx "inline_mode_value=${STREAMING_LI_INLINE_MODE}" \
            "${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE" || \
       ! grep -Fqx "post_closure_mode=${STREAMING_LI_POST_CLOSURE_MODE_NAME}" \
            "${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE" || \
       ! grep -Fqx "post_closure_mode_value=${STREAMING_LI_POST_CLOSURE_MODE}" \
            "${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE" || \
       ! grep -Fqx "post_scheduler_mode=${STREAMING_LI_POST_SCHEDULER_MODE_NAME}" \
            "${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE" || \
       ! grep -Fqx "post_scheduler_mode_value=${STREAMING_LI_POST_SCHEDULER_MODE}" \
            "${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE" || \
       ! grep -Fqx "kvaf_closure=${P3_KVAF_CLOSURE}" \
            "${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE" || \
       ! grep -Fqx "kvaf_source_head=${P3_KVAF_SOURCE_HEAD}" \
            "${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE" || \
       ! grep -Fqx 'abi=24in_11out' \
            "${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE"; then
        echo "[nanovllm_a5_ops] ERROR: P3 OPP provenance manifest is incomplete." >&2
        exit 1
    fi
    echo "[nanovllm_a5_ops] P3 target provenance: ${VENDOR_DIR}/P3_STREAMING_LI_PROVENANCE"
fi

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
