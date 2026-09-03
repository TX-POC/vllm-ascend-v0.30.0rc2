#!/usr/bin/env python3
"""Seal a freshly built official C8 bridge with exact, local provenance.

The caller must use a new timestamped bridge output directory and invoke this
immediately after its successful ``build_ext``.  The file is deliberately
write-once: a later run cannot overwrite provenance for an existing bridge.
No CANN, torch, or NPU API is imported here.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


TARGET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_ROOT))

from product_bf16_abi_gate import (  # noqa: E402
    GateFailure,
    REFERENCE_BRIDGE_MARKER,
    REFERENCE_BRIDGE_SOURCE,
    _git_head,
    _sha256,
    _require_bridge_module,
    _require_directory,
    _require_file,
)


def _required(value: str | None, label: str) -> str:
    if not value:
        raise GateFailure(f"argument_missing:{label}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-vllm-root", default=os.environ.get("P10_REFERENCE_VLLM_ROOT")
    )
    parser.add_argument(
        "--reference-opapi", default=os.environ.get("P10_REFERENCE_OPAPI")
    )
    parser.add_argument(
        "--reference-bridge-root",
        default=os.environ.get("P10_REFERENCE_BRIDGE_ROOT"),
    )
    parser.add_argument(
        "--reference-bridge-module",
        default=os.environ.get("P10_REFERENCE_BRIDGE_MODULE"),
    )
    args = parser.parse_args()
    try:
        root = _require_directory(
            Path(_required(args.reference_vllm_root, "reference_vllm_root")),
            "reference_vllm_root_missing",
        )
        opapi = _require_file(
            Path(_required(args.reference_opapi, "reference_opapi")),
            "reference_opapi_missing",
        )
        bridge_root = _require_directory(
            Path(_required(args.reference_bridge_root, "reference_bridge_root")),
            "reference_bridge_root_missing",
        )
        marker = bridge_root / REFERENCE_BRIDGE_MARKER
        if marker.exists():
            raise GateFailure(f"bridge_provenance_already_exists:{marker}")
        bridge_module = _require_bridge_module(
            bridge_root,
            Path(_required(args.reference_bridge_module, "reference_bridge_module")),
        )
        bridge_source = _require_file(
            root / REFERENCE_BRIDGE_SOURCE, "reference_bridge_source_missing"
        )
        content = "\n".join(
            (
                "format=1",
                "kind=source_matched_official_c8_bridge",
                f"source_root={root}",
                f"source_head={_git_head(root)}",
                f"opapi={opapi}",
                f"opapi_sha256={_sha256(opapi)}",
                f"bridge_source={bridge_source}",
                f"bridge_source_sha256={_sha256(bridge_source)}",
                f"bridge_module={bridge_module}",
                f"bridge_module_sha256={_sha256(bridge_module)}",
                "",
            )
        )
        marker.write_text(content, encoding="utf-8")
    except GateFailure as error:
        print(f"P10_REFERENCE_BRIDGE_PROVENANCE status=fail reason={error}")
        return 2
    print(
        "P10_REFERENCE_BRIDGE_PROVENANCE "
        f"status=pass marker={marker} module={bridge_module}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
