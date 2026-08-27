#!/usr/bin/env python3
"""Static contract for the isolated source-derived resident-threshold probe."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OP = ROOT / "a5_c8_resident_seed"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    spec = json.loads((OP / "ops.json").read_text())
    require(len(spec) == 1 and spec[0]["op"] == "A5C8ResidentSeed", "isolated op name")
    item = spec[0]
    inputs = {entry["name"]: entry for entry in item["input_desc"]}
    outputs = {entry["name"]: entry for entry in item["output_desc"]}
    for name in (
        "resident_index_key",
        "resident_index_key_dequant_scale",
        "resident_seq_lengths",
        "resident_index_block_table",
    ):
        require(name in inputs, f"explicit resident index ABI missing {name}")
    require("packed_kv" not in inputs and "packed_output" not in inputs, "packed QSFA payload must not be LI input")
    require(inputs["query"]["type"] == ["int8"],
            "generic CANN JSON must use the supported int8 bootstrap TypeRange")
    require(inputs["resident_index_key"]["type"] == ["int8"],
            "generic CANN JSON must use the supported int8 bootstrap TypeRange")
    require(outputs["resident_tau"]["type"] == ["int32"], "tau must be int32 graph ABI")
    require(outputs["tau_valid"]["type"] == ["int32"], "tau_valid must be int32 graph ABI")

    attrs = {entry["name"]: entry["default_value"] for entry in item["attr"]}
    require(attrs["sparse_count"] == "2048", "seed requires Top-2048")
    require(attrs["layout_query"] == "TND", "decode-oriented TND default")
    require(attrs["layout_key"] == "PA_BSND", "paged resident index default")

    root_kernel = (OP / "op_kernel" / "a5_c8_resident_seed.cpp").read_text()
    vector = (OP / "op_kernel" / "arch35" / "quant_lightning_indexer_service_vector.h").read_text()
    topk = (OP / "op_kernel" / "arch35" / "vf" / "quant_lightning_indexer_topk.h").read_text()
    op_def = (OP / "op_host" / "a5_c8_resident_seed_def.cpp").read_text()
    tiling = (OP / "op_host" / "a5_c8_resident_seed_tiling.cpp").read_text()
    op_add_pattern = r"OP_ADD\s*\(\s*A5C8ResidentSeed\s*\)"
    host_registration_sources = []
    for source in sorted((OP / "op_host").glob("*.cpp")):
        match_count = len(re.findall(op_add_pattern, source.read_text()))
        if match_count:
            host_registration_sources.append((source.name, match_count))
    require(host_registration_sources == [("a5_c8_resident_seed_def.cpp", 1)],
            "checked-in host closure must have exactly one ResidentSeed OP_ADD")

    require("residentTau" in root_kernel and "tauValid" in root_kernel, "tau must be real kernel outputs")
    require("BatchMulWeightAndReduceSum" in vector, "must retain official C8 score closure")
    require("GetLastKthValue" in topk, "must expose official final TopK boundary")
    require("DataCopyPad(residentTauGm" in vector, "tau must be AIV DMA output, not host shadow")
    require("DataCopyPad(tauValidGm" in vector, "tau validity must be AIV DMA output")
    # CANN 9.1's InitGlobalMemory takes a non-const GlobalTensor lvalue.
    # Keep the GM slice's address and row offset unchanged, but materialize
    # the slice as a named view before the fail-closed zero fill.
    require("GlobalTensor<int32_t> residentTauOutput = residentTauGm[rowOffset];" in vector,
            "resident tau zero-fill must materialize its GM slice as an lvalue")
    require("GlobalTensor<int32_t> tauValidOutput = tauValidGm[rowOffset];" in vector,
            "tau-valid zero-fill must materialize its GM slice as an lvalue")
    require("AscendC::InitGlobalMemory(residentTauOutput, rowCount, 0);" in vector,
            "resident tau zero-fill must use the named lvalue view")
    require("AscendC::InitGlobalMemory(tauValidOutput, rowCount, 0);" in vector,
            "tau-valid zero-fill must use the named lvalue view")
    require("InitGlobalMemory(residentTauGm[rowOffset]" not in vector and
            "InitGlobalMemory(tauValidGm[rowOffset]" not in vector,
            "CANN 9.1 rejects temporary indexed GM views in InitGlobalMemory")
    require("WriteResidentSeed(outputRow, kthTau, hasExactTau)" in vector, "final TopK must feed tau write")
    require(vector.rindex("topkOp_(mrgValueLocal_") < vector.index("WriteResidentSeed(outputRow, kthTau, hasExactTau)"),
            "tau write must follow the final official TopK reduction")
    require("sparse_count=%d so resident_tau is the exact 2048th C8 score" in tiling,
            "tiling must reject non-2048 threshold semantics")
    require("AddConfig(\"ascend950\"" in op_def and "AddConfig(\"ascend910b\"" not in op_def,
            "probe must be A5-only")
    require("aicore_config_950.Input(\"query\")" in op_def and
            "aicore_config_950.Input(\"resident_index_key\")" in op_def and
            "ge::DT_FLOAT8_E4M3FN" in op_def,
            "Ascend950 OpDef must retain the real FP8 E4M3FN ABI")
    require('this->Input("query")\n            .ParamType(REQUIRED)\n            .DataType({ge::DT_INT8})' in op_def and
            'this->Input("resident_index_key")\n            .ParamType(REQUIRED)\n            .DataType({ge::DT_INT8})' in op_def,
            "generic OpDef must agree with the CANN JSON bootstrap TypeRange")

    bridge = (OP / "torch_extension" / "csrc" / "a5_c8_resident_seed.cpp").read_text()
    require("at::kFloat8_e4m3fn" in bridge,
            "typed bridge must reject a non-FP8 input despite generic JSON int8")

    host_compat = (OP / "op_host" / "a5_c8_resident_seed_host_compat.h").read_text()
    for macro in ("OP_CHECK_NULL_WITH_CONTEXT", "OP_CHECK_IF", "OP_LOGE", "REPORT_INNER_ERR_MSG"):
        require(macro in host_compat, f"isolated host compatibility header misses {macro}")

    normal_build = (ROOT / "build_nanovllm_ops.sh").read_text()
    all_ops = normal_build.split("ALL_OP_NAMES=(", 1)[1].split(")", 1)[0]
    require("A5C8ResidentSeed" not in all_ops, "probe must not enter ALL_OP_NAMES")
    isolated = (OP / "build_isolated_a5_c8_resident_seed.sh").read_text()
    require("A5C8ResidentSeed" in isolated and "build_nanovllm_ops.sh" not in isolated,
            "only independent isolated build is allowed")
    require("CANN-bootstrap/FP8 ABI contract" in isolated,
            "isolated builder must fail early with the three-layer ABI diagnostic")
    require('op_host/err/ops_err.h' in isolated and 'op_host/log/log.h' in isolated,
            "isolated builder must stage the full host-header closure before msopgen")
    require('generated host closure is incomplete' in isolated,
            "isolated builder must verify the generated CMake input also has the header closure")
    # `msopgen` creates an all-in-one host stub with an OP_ADD registration.
    # The source-derived C8 OpDef has a different filename, so merely
    # overlaying it leaves duplicate INI sections.  ResidentSeed must use the
    # same remove-and-count closure that made P3's isolated build reliable.
    require("P4_GENERATED_HOST_STUB" in isolated,
            "isolated builder must identify the msopgen ResidentSeed host stub")
    require('rm -f "${P4_GENERATED_HOST_STUB}"' in isolated,
            "isolated builder must remove the generated duplicate OP_ADD stub")
    require("P4_STUB_REGISTRATIONS" in isolated and
            "failed to remove generated host stub" in isolated,
            "isolated builder must prove the removed stub was the expected single registration")
    require("P4_OPDEF_SOURCES" in isolated and
            "host registration closure:" in isolated,
            "isolated builder must prove exactly one source-derived OP_ADD remains")
    require("P4_TOTAL_REGISTRATIONS" in isolated and
            "P4_SOURCE_REGISTRATIONS" in isolated,
            "isolated builder must count OP_ADD occurrences, not only matching files")
    build_sequence = (
        isolated.index("msopgen gen"),
        isolated.index('cp -a "${OP_DEV}/op_host/." "${GENERATED}/op_host/"'),
        isolated.index("P4_GENERATED_HOST_STUB"),
        isolated.index('rm -f "${P4_GENERATED_HOST_STUB}"'),
        isolated.index("P4_OPDEF_SOURCES=()"),
        isolated.index('OPS_CPU_NUMBER="${BUILD_JOBS}" bash build.sh'),
    )
    require(build_sequence == tuple(sorted(build_sequence)),
            "builder order must be msopgen, overlay, stub removal, unique OP_ADD check, then build")
    require("INI is diagnostic-only and was not modified" in isolated,
            "duplicate registration diagnostics must never patch CANN's generated INI")
    require("CANN 9.1 InitGlobalMemory lvalue contract" in isolated and
            "direct indexed GM view" in isolated,
            "isolated builder must reject the known temporary-GM InitGlobalMemory form before msopgen")

    target_equivalence = (
        ROOT / "tests" / "test_a5_c8_resident_seed_target_equivalence.py"
    ).read_text()
    require("A5_C8_RESIDENT_SEED_TARGET_PASS" in target_equivalence,
            "ResidentSeed needs an isolated target equivalence gate before P4")
    require("_call_official_resident_reference" in target_equivalence and
            "_launch_resident_seed" in target_equivalence,
            "target gate must use one resident FP8 view for official and seed calls")
    require("official_tau=not_exported" in target_equivalence and
            "threshold_functional_proof=next_p4_candidate_audit" in target_equivalence,
            "target gate must not overclaim a public official tau comparator")

    audit_adapter = (ROOT / "tests" / "fused_streaming_quant_li_get_audit_adapter.py").read_text()
    require('block_table = inputs["block_table"]' in audit_adapter and
            "source_blocks = int(block_table.shape[1])" in audit_adapter,
            "resident fixture must derive source capacity from the concrete input view")
    require("SOURCE_TOKENS // BLOCK_SIZE <= RESIDENT_BLOCKS" not in audit_adapter,
            "resident fixture must not depend on an unimported source-length global")
    require("P4_RESIDENT_SEED_BRIDGE" in audit_adapter and
            "sys.path.insert(0, bridge_root_text)" in audit_adapter,
            "target adapter must resolve the source-matched in-place Seed bridge")
    require("import nanovllm_a5_c8_resident_seed" in audit_adapter and
            "import nanovllm_a5_resident_seed" not in audit_adapter,
            "target adapter must import the exact ResidentSeed bridge package")
    require("A5_C8_RESIDENT_SEED_TARGET_STAGE stage=seed_launch" in target_equivalence and
            "A5_C8_RESIDENT_SEED_TARGET_STAGE stage=official_launch" in target_equivalence,
            "target gate must identify which NPU launch failed")
    require("sparse_values=bitwise_equal_empty_fp32_return_values_false" in target_equivalence,
            "target gate must compare the official and Seed empty-value outputs")

    print("A5_C8_RESIDENT_SEED_CONTRACT_PASS")


if __name__ == "__main__":
    main()
