#!/usr/bin/env python3
"""Report, but never guess, the C8 source family selected for P10 staging.

The resolved input must be an exact operator source root. Both known C8
families are meaningful provenance results: only the VllmQuant family has a
P10 executor today, while LightningIndexerQuant is reported as recognised but
not yet implemented. No NPU work is performed here.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


_FAMILIES = {
    "vllm_quant_lightning_indexer": {
        "operator": "VllmQuantLightningIndexer",
        "definition": "op_host/vllm_quant_lightning_indexer_def.cpp",
        "required_files": (
            "op_host/vllm_quant_lightning_indexer_tiling.cpp",
            "op_kernel/vllm_quant_lightning_indexer.cpp",
            "op_kernel/arch35/quant_lightning_indexer_kernel.h",
            "op_kernel/arch35/quant_lightning_indexer_service_vector.h",
        ),
        "anchors": "BatchMulWeightAndReduceSum,ProcessTopK",
        # This probe is intentionally source-only.  The product frontend may
        # present BF16 or FP32 weights; that choice is proved separately by
        # product_bf16_abi_gate.py and must never be inferred from this root.
        "abi": "source:fp8_qk_fp32_weights_fp32_scales",
        "supported": True,
    },
    "lightning_indexer_quant": {
        "operator": "LightningIndexerQuant",
        "definition": "op_host/lightning_indexer_quant_def.cpp",
        "required_files": (
            "op_host/lightning_indexer_quant_tiling.cpp",
            "op_kernel/lightning_indexer_quant.cpp",
            "op_kernel/lightning_indexer_quant_kernel.h",
            "op_kernel/lightning_indexer_quant_service_vector.h",
        ),
        "anchors": "lightning_indexer_quant_kernel",
        "abi": "source:int8_qk_fp16_weights_and_scales",
        "supported": False,
    },
}


def _matches_family(root: Path, family: str) -> bool:
    spec = _FAMILIES[family]
    definition = root / spec["definition"]
    if not definition.is_file():
        return False
    text = definition.read_text(errors="replace")
    operator = spec["operator"]
    return f"class {operator}" in text and f"OP_ADD({operator})" in text


def _source_sha(root: Path) -> str:
    for candidate in (root, *root.parents):
        if not (candidate / ".git").exists():
            continue
        result = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"],
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    return "<unavailable>"


def probe(root: Path) -> tuple[bool, str, str | None, tuple[str, ...], tuple[str, ...]]:
    """Return ok/reason/family/candidates/missing without path discovery."""
    if not root.is_dir():
        return False, "source_root_missing", None, (), ()

    families = tuple(family for family in _FAMILIES if _matches_family(root, family))
    if len(families) == 0:
        return False, "source_family_unsupported", None, (), ()
    if len(families) > 1:
        return False, "source_family_ambiguous", None, families, ()

    family = families[0]
    missing = tuple(
        relative
        for relative in _FAMILIES[family]["required_files"]
        if not (root / relative).is_file()
    )
    if missing:
        return False, "source_family_incomplete", family, families, missing
    return True, "ok", family, families, ()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default=os.environ.get("P10_PRODUCT_C8_SOURCE_ROOT"),
        help="resolved exact operator source root; defaults only to P10_PRODUCT_C8_SOURCE_ROOT",
    )
    args = parser.parse_args()
    if not args.source_root:
        parser.error("--source-root or P10_PRODUCT_C8_SOURCE_ROOT is required")

    root = Path(args.source_root).resolve()
    ok, reason, family, candidates, missing = probe(root)
    fields = [
        f"status={'pass' if ok else 'fail'}",
        f"reason={reason}",
        f"source_root={root}",
        f"source_sha={_source_sha(root)}",
        f"family={family if family else '<none>'}",
        f"candidates={','.join(candidates) if candidates else '<none>'}",
        f"missing={','.join(missing) if missing else '<none>'}",
    ]
    if family is not None:
        spec = _FAMILIES[family]
        fields.extend(
            [
                f"operator={spec['operator']}",
                f"supported={int(spec['supported'])}",
                f"anchors={spec['anchors']}",
                f"abi={spec['abi']}",
            ]
        )
        if not spec["supported"]:
            fields[1] = "reason=family_not_implemented"
    print("P10_PRODUCT_C8_PROVENANCE " + " ".join(fields))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
