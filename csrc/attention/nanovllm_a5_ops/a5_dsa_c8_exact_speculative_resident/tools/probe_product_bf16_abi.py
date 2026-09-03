#!/usr/bin/env python3
"""Host-safe P10 product-C8/reference ABI and provenance preflight.

It checks files, Git revisions, and artifact receipts only.  In particular it
does not import torch, torch_npu, CANN, or an NPU bridge, so it is safe before
the target NPU is selected.
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
    provenance_fields,
    verify_product_c8_frontend_source,
    verify_reference_artifacts,
)


def _argument(
    parser: argparse.ArgumentParser, name: str, environment: str, help_text: str
) -> None:
    parser.add_argument(
        name,
        default=os.environ.get(environment),
        help=f"{help_text}; defaults to ${environment}",
    )


def _required(value: str | None, label: str) -> str:
    if not value:
        raise GateFailure(f"argument_missing:{label}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    _argument(
        parser,
        "--frontend",
        "P10_PRODUCT_C8_FRONTEND",
        "explicit product frontend: dsa_offload_bf16 or main_dsa_v1_fp32",
    )
    _argument(
        parser,
        "--product-vllm-root",
        "P10_PRODUCT_VLLM_ROOT",
        "exact product vLLM source root",
    )
    _argument(
        parser,
        "--reference-vllm-root",
        "P10_REFERENCE_VLLM_ROOT",
        "exact source root used to build the official C8 reference",
    )
    _argument(
        parser,
        "--reference-opapi",
        "P10_REFERENCE_OPAPI",
        "official reference libcust_opapi.so",
    )
    _argument(
        parser,
        "--reference-bridge-root",
        "P10_REFERENCE_BRIDGE_ROOT",
        "fresh official reference bridge output root",
    )
    _argument(
        parser,
        "--reference-bridge-module",
        "P10_REFERENCE_BRIDGE_MODULE",
        "exact official reference bridge _C*.so selected for Python import",
    )
    _argument(
        parser,
        "--reference-source-sha",
        "P10_REFERENCE_SOURCE_SHA",
        "optional exact source revision expected by the caller",
    )
    args = parser.parse_args()

    try:
        product = verify_product_c8_frontend_source(
            Path(_required(args.product_vllm_root, "product_vllm_root")),
            _required(args.frontend, "frontend"),
        )
        reference = verify_reference_artifacts(
            reference_vllm_root=Path(
                _required(args.reference_vllm_root, "reference_vllm_root")
            ),
            reference_opapi=Path(
                _required(args.reference_opapi, "reference_opapi")
            ),
            reference_bridge_root=Path(
                _required(args.reference_bridge_root, "reference_bridge_root")
            ),
            reference_bridge_module=Path(
                _required(args.reference_bridge_module, "reference_bridge_module")
            ),
            expected_source_sha=args.reference_source_sha or None,
        )
    except GateFailure as error:
        print(f"P10_PRODUCT_C8_ABI_GATE status=fail reason={error}")
        return 2
    print(provenance_fields(product, reference))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
