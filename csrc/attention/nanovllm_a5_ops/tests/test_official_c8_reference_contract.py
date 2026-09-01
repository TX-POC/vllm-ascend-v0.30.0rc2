#!/usr/bin/env python3
"""Static gate for the source-matched official C8 reference bundle.

The P3 equivalence test must not silently load an editable ``vllm_ascend``
package from an unrelated checkout.  This probe deliberately has no NPU or
CANN dependency; it protects the source/OPP/bridge boundaries that make the
target test a valid mathematical comparison.
"""

from __future__ import annotations

from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = OPS_ROOT / "a5_official_c8_reference"
BUILD_SCRIPT = REFERENCE_ROOT / "build_reference_opp.sh"
REFERENCE_BRIDGE = (
    REFERENCE_ROOT / "torch_extension/csrc/official_c8_reference.cpp"
)
TARGET_ADAPTER = OPS_ROOT / "tests/streaming_quant_li_get_target_adapter.py"
COMMON_OPAPI = OPS_ROOT / "torch_extension/csrc/op_api_common.h"
UPSTREAM_BUILD_SCRIPT = OPS_ROOT.parents[1] / "build.sh"
GITIGNORE = OPS_ROOT.parents[2] / ".gitignore"


def require_source_matched_reference_contract() -> None:
    build_script = BUILD_SCRIPT.read_text(encoding="utf-8")
    upstream_build_script = UPSTREAM_BUILD_SCRIPT.read_text(encoding="utf-8")
    gitignore = GITIGNORE.read_text(encoding="utf-8")
    reference_bridge = REFERENCE_BRIDGE.read_text(encoding="utf-8")
    adapter = TARGET_ADAPTER.read_text(encoding="utf-8")
    common_opapi = COMMON_OPAPI.read_text(encoding="utf-8")

    # The only upstream build inputs are the two official C8 operators.  This
    # prevents a full vLLM build or a stale product package from becoming the
    # hidden reference implementation.
    assert "vllm_quant_lightning_indexer,vllm_quant_lightning_indexer_metadata" in build_script
    assert "--soc=ascend950" in build_script
    assert "pip install" not in build_script
    assert "setup.py" not in build_script
    assert "cmake --build" not in build_script
    assert "P3_REFERENCE_PROVENANCE" in build_script
    assert "source_head=${SOURCE_HEAD}" in build_script
    assert "-newer \"${BUILD_MARKER}\"" in build_script
    assert '! -e "${SOURCE_ROOT}/.git"' in build_script
    # A filtered two-op reference build must not reuse the normal product
    # target graph.  The upstream builder supports independent sibling
    # directories, and the reference cache is keyed by exact source revision
    # so rebuilds of the same revision remain incremental.
    assert "build_official_c8_reference_${SOURCE_HEAD_SHORT}" in build_script
    assert 'VLLM_ASCEND_CMAKE_BUILD_DIR="${UPSTREAM_BUILD_DIR}"' in build_script
    assert 'VLLM_ASCEND_CMAKE_OUTPUT_DIR="${UPSTREAM_OUTPUT_DIR}"' in build_script
    assert 'VLLM_ASCEND_CMAKE_BUILD_OUT_DIR="${UPSTREAM_BUILD_OUT_DIR}"' in build_script
    assert 'find "${UPSTREAM_BUILD_DIR}"' in build_script
    assert 'BUILD_DIR=${VLLM_ASCEND_CMAKE_BUILD_DIR:-${CURRENT_DIR}/build}' in upstream_build_script
    assert 'OUTPUT_DIR=${VLLM_ASCEND_CMAKE_OUTPUT_DIR:-${CURRENT_DIR}/output}' in upstream_build_script
    assert 'BUILD_OUT_DIR=${VLLM_ASCEND_CMAKE_BUILD_OUT_DIR:-${CURRENT_DIR}/build_out}' in upstream_build_script
    # These revision-keyed CMake directories are deliberate generated output.
    # Ignoring exactly these paths preserves the runner's strict check for any
    # real source modification while avoiding a false "dirty" result after a
    # successful source-matched reference build.
    assert "/csrc/build_official_c8_reference_*/" in gitignore
    assert "/csrc/output_official_c8_reference_*/" in gitignore
    assert "/csrc/build_out_official_c8_reference_*/" in gitignore

    # Both real ACLNN calls retain the canonical names from torch_binding.cpp.
    # The reference bridge gets a private resolver slot so changing the P3
    # target DSO later in the same process cannot redirect it.
    assert '#define NANOVLLM_A5_OPAPI_ENV "NANOVLLM_A5_REFERENCE_C8_OPAPI_LIB"' in reference_bridge
    assert "aclnnVllmQuantLightningIndexerMetadata" in reference_bridge
    assert "aclnnVllmQuantLightningIndexer" in reference_bridge
    assert "TORCH_LIBRARY(a5_official_c8_reference" in reference_bridge
    assert "#ifndef NANOVLLM_A5_OPAPI_ENV" in common_opapi
    # aclDataType values are C++ enum identifiers, rather than preprocessor
    # macros.  Do not silently turn valid FP8 E4M3FN tensors into
    # ACL_DT_UNDEFINED before the source-matched reference reaches its tiler.
    assert "kAclFloat8E4M3Fn = ACL_FLOAT8_E4M3FN" in common_opapi
    assert "static_assert(kAclFloat8E4M3Fn != ACL_DT_UNDEFINED" in common_opapi
    assert "#define ACL_FLOAT8_E4M3FN ACL_DT_UNDEFINED" not in common_opapi
    assert "NANOVLLM_A5_ACL_TENSOR" in common_opapi

    # The test adapter must not import the normal editable package at all.
    # It validates the source revision from the reference OPP manifest before
    # the first ACLNN call and makes both vendors visible at process startup.
    assert "import vllm_ascend" not in adapter
    assert "a5_dsa_registration" not in adapter
    assert 'REFERENCE_OPP_ENV = "NANOVLLM_A5_REFERENCE_C8_OPP"' in adapter
    assert 'REFERENCE_EXPECTED_OPP_SOURCE_HEAD_ENV = "P3_REFERENCE_EXPECTED_OPP_SOURCE_HEAD"' in adapter
    assert 'RUNNER_EXPECTED_HEAD_ENV = "P3_RUNNER_EXPECTED_HEAD"' in adapter
    artifact_env = (OPS_ROOT / "tests" / "resolve_streaming_li_get_artifacts.sh").read_text()
    assert "P3_REFERENCE_EXPECTED_OPP_SOURCE_HEAD" in artifact_env
    assert "P3_TARGET_EXPECTED_OPP_SOURCE_HEAD" in artifact_env
    assert "P3_RUNNER_EXPECTED_HEAD" in artifact_env
    assert "unset P3_REFERENCE_EXPECTED_SOURCE_HEAD P3_TARGET_EXPECTED_SOURCE_HEAD" in artifact_env
    assert "P3_REFERENCE_PROVENANCE" in adapter
    assert "ASCEND_CUSTOM_OPP_PATH" in adapter
    assert "nanovllm_official_c8_reference" in adapter
    # Target hardware selection must be explicit.  A single-card
    # ASCEND_RT_VISIBLE_DEVICES deliberately maps physical card 7 to logical
    # npu:0, and the adapter must print both identities in the validation log.
    assert 'TARGET_NPU_ENV = "P3_NPU_ID"' in adapter
    assert 'torch.device(f"npu:{requested_id}")' in adapter
    assert "ASCEND_RT_VISIBLE_DEVICES" in adapter
    assert "P3_TARGET_NPU" in adapter
    # Keep target-call boundaries visible in remote timeout logs.  A derived
    # OPP launch may be asynchronous, so call-return and synchronize markers
    # distinguish host-side ACLNN setup from an AICore task timeout.
    assert "P3_DERIVED_STAGE stage=call_begin" in adapter
    assert "P3_DERIVED_STAGE stage=call_return" in adapter
    assert "P3_DERIVED_STAGE stage=synchronize_begin" in adapter
    assert "P3_DERIVED_STAGE stage=synchronize_end" in adapter

    print(
        "P3_OFFICIAL_C8_REFERENCE_CONTRACT_PASS "
        "source_matched_opp_only=1 standalone_bridge=1 "
        "editable_vllm_reference_forbidden=1 dual_dso_resolver=1"
    )


if __name__ == "__main__":
    require_source_matched_reference_contract()
