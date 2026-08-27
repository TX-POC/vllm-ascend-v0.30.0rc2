#!/usr/bin/env python3
"""Static contract for the isolated source-derived resident-threshold probe."""

from __future__ import annotations

import json
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

    require("residentTau" in root_kernel and "tauValid" in root_kernel, "tau must be real kernel outputs")
    require("BatchMulWeightAndReduceSum" in vector, "must retain official C8 score closure")
    require("GetLastKthValue" in topk, "must expose official final TopK boundary")
    require("DataCopyPad(residentTauGm" in vector, "tau must be AIV DMA output, not host shadow")
    require("DataCopyPad(tauValidGm" in vector, "tau validity must be AIV DMA output")
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

    bridge = (OP / "torch_extension" / "csrc" / "a5_c8_resident_seed.cpp").read_text()
    require("at::kFloat8_e4m3fn" in bridge,
            "typed bridge must reject a non-FP8 input despite generic JSON int8")

    normal_build = (ROOT / "build_nanovllm_ops.sh").read_text()
    all_ops = normal_build.split("ALL_OP_NAMES=(", 1)[1].split(")", 1)[0]
    require("A5C8ResidentSeed" not in all_ops, "probe must not enter ALL_OP_NAMES")
    isolated = (OP / "build_isolated_a5_c8_resident_seed.sh").read_text()
    require("A5C8ResidentSeed" in isolated and "build_nanovllm_ops.sh" not in isolated,
            "only independent isolated build is allowed")
    require("CANN-bootstrap/FP8 ABI contract" in isolated,
            "isolated builder must fail early with the three-layer ABI diagnostic")

    print("A5_C8_RESIDENT_SEED_CONTRACT_PASS")


if __name__ == "__main__":
    main()
