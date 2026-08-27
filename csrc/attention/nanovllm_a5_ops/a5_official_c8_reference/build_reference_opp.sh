#!/usr/bin/env bash
# Build an isolated source-matched official C8 OPP reference.
#
# This intentionally invokes only the two upstream operators.  It never runs
# a vLLM-Ascend Python/package build and never overwrites a normal installed
# OPP.  Give A5_OFFICIAL_C8_REFERENCE_ROOT a fresh timestamped directory.

set -o pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOURCE_ROOT="$(cd "${THIS_DIR}/../../../.." && pwd)"
SOURCE_ROOT="${A5_OFFICIAL_C8_SOURCE_ROOT:-${DEFAULT_SOURCE_ROOT}}"
INSTALL_ROOT="${A5_OFFICIAL_C8_REFERENCE_ROOT:-${THIS_DIR}/build-reference-opp}"
EXPECTED_HEAD="${A5_OFFICIAL_C8_EXPECTED_SOURCE_HEAD:-}"
BUILD_JOBS="${A5_OFFICIAL_C8_REFERENCE_BUILD_JOBS:-4}"
VENDOR_NAME="official_c8_reference"
OP_LIST="vllm_quant_lightning_indexer,vllm_quant_lightning_indexer_metadata"

printf 'P3_REFERENCE_BUILD source_root=%s install_root=%s ops=%s jobs=%s\n' \
  "${SOURCE_ROOT}" "${INSTALL_ROOT}" "${OP_LIST}" "${BUILD_JOBS}"

# A linked Git worktree stores .git as a file pointing back to the primary
# repository, while a normal checkout stores it as a directory.  Both are
# valid source roots for the provenance gate.
if [[ ! -e "${SOURCE_ROOT}/.git" || ! -f "${SOURCE_ROOT}/csrc/build.sh" ]]; then
  printf 'P3_REFERENCE_BUILD_FAIL reason=invalid_source_root path=%s\n' \
    "${SOURCE_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_ROOT}/csrc/attention/vllm_quant_lightning_indexer/op_kernel/vllm_quant_lightning_indexer.cpp" || \
      ! -f "${SOURCE_ROOT}/csrc/attention/vllm_quant_lightning_indexer_metadata/op_kernel_aicpu/vllm_quant_lightning_indexer_metadata_aicpu.cpp" ]]; then
  printf 'P3_REFERENCE_BUILD_FAIL reason=official_c8_source_closure_missing\n' >&2
  exit 2
fi
if ! command -v msopgen >/dev/null 2>&1; then
  printf 'P3_REFERENCE_BUILD_FAIL reason=msopgen_missing\n' >&2
  exit 2
fi
if [[ ! "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'P3_REFERENCE_BUILD_FAIL reason=invalid_jobs value=%s\n' "${BUILD_JOBS}" >&2
  exit 2
fi

SOURCE_HEAD="$(git -C "${SOURCE_ROOT}" rev-parse HEAD 2>/dev/null || true)"
if [[ -z "${SOURCE_HEAD}" ]]; then
  printf 'P3_REFERENCE_BUILD_FAIL reason=source_head_unavailable\n' >&2
  exit 2
fi
if [[ -n "${EXPECTED_HEAD}" && "${SOURCE_HEAD}" != "${EXPECTED_HEAD}" ]]; then
  printf 'P3_REFERENCE_BUILD_FAIL reason=unexpected_source_head got=%s expected=%s\n' \
    "${SOURCE_HEAD}" "${EXPECTED_HEAD}" >&2
  exit 2
fi

# Do not delete or overlay a prior reference bundle.  A fresh path makes its
# source provenance unambiguous and keeps a failed build recoverable.
if [[ -e "${INSTALL_ROOT}" ]]; then
  printf 'P3_REFERENCE_BUILD_FAIL reason=install_root_exists path=%s\n' \
    "${INSTALL_ROOT}" >&2
  exit 2
fi
# Do not place a marker inside INSTALL_ROOT: CANN's .run installer owns that
# directory and some releases require it to be empty.  The sidecar marker
# makes the package-selection rule below deterministic even when csrc/build
# still contains an older .run from another build.
INSTALL_PARENT="$(dirname "${INSTALL_ROOT}")"
INSTALL_LEAF="$(basename "${INSTALL_ROOT}")"
mkdir -p "${INSTALL_PARENT}"
BUILD_MARKER="${INSTALL_PARENT}/.${INSTALL_LEAF}.p3_reference_build_started"
: > "${BUILD_MARKER}"

# The upstream CANN builder owns csrc/build and csrc/output.  It is scoped to
# the two requested operators via --ops; no vllm_ascend_C target is involved.
(
  cd "${SOURCE_ROOT}/csrc" || exit 2
  MAX_JOBS="${BUILD_JOBS}" \
    bash ./build.sh --pkg --soc=ascend950 --ops="${OP_LIST}" \
      --vendor_name="${VENDOR_NAME}" -j"${BUILD_JOBS}"
)
BUILD_STATUS=$?
if [[ "${BUILD_STATUS}" -ne 0 ]]; then
  printf 'P3_REFERENCE_BUILD_FAIL reason=upstream_opp_build status=%s\n' \
    "${BUILD_STATUS}" >&2
  exit "${BUILD_STATUS}"
fi

RUN_CANDIDATES=()
while IFS= read -r candidate; do
  RUN_CANDIDATES+=("${candidate}")
done < <(find "${SOURCE_ROOT}/csrc/build" -maxdepth 1 -type f \
  -name 'cann-ops-transformer*.run' -newer "${BUILD_MARKER}" -print \
  2>/dev/null | sort)
if [[ "${#RUN_CANDIDATES[@]}" -ne 1 ]]; then
  printf 'P3_REFERENCE_BUILD_FAIL reason=run_package_ambiguous count=%s build_dir=%s\n' \
    "${#RUN_CANDIDATES[@]}" "${SOURCE_ROOT}/csrc/build" >&2
  printf 'P3_REFERENCE_BUILD_RUN_CANDIDATES=%s\n' \
    "${RUN_CANDIDATES[*]:-<none>}" >&2
  exit 2
fi
RUN_PACKAGE="${RUN_CANDIDATES[0]}"
chmod +x "${RUN_PACKAGE}"
"${RUN_PACKAGE}" --quiet --install-path="${INSTALL_ROOT}"
INSTALL_STATUS=$?
if [[ "${INSTALL_STATUS}" -ne 0 ]]; then
  printf 'P3_REFERENCE_BUILD_FAIL reason=install status=%s\n' \
    "${INSTALL_STATUS}" >&2
  exit "${INSTALL_STATUS}"
fi

VENDOR="${INSTALL_ROOT}/vendors/${VENDOR_NAME}_transformer"
OPAPI="${VENDOR}/op_api/lib/libcust_opapi.so"
if [[ ! -s "${OPAPI}" ]]; then
  printf 'P3_REFERENCE_BUILD_FAIL reason=opapi_missing path=%s\n' "${OPAPI}" >&2
  exit 2
fi
if ! grep -Rqs 'VllmQuantLightningIndexer' "${VENDOR}" || \
   ! grep -Rqs 'VllmQuantLightningIndexerMetadata' "${VENDOR}"; then
  printf 'P3_REFERENCE_BUILD_FAIL reason=operator_manifest_missing vendor=%s\n' \
    "${VENDOR}" >&2
  exit 2
fi

cat > "${VENDOR}/P3_REFERENCE_PROVENANCE" <<EOF
format=1
kind=source_matched_official_c8
source_root=${SOURCE_ROOT}
source_head=${SOURCE_HEAD}
operators=${OP_LIST}
vendor=${VENDOR_NAME}_transformer
EOF

printf 'P3_REFERENCE_BUILD_PASS source_head=%s vendor=%s opapi=%s provenance=%s\n' \
  "${SOURCE_HEAD}" "${VENDOR}" "${OPAPI}" \
  "${VENDOR}/P3_REFERENCE_PROVENANCE"
