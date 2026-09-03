#!/usr/bin/env python3
"""Print compact, fail-closed provenance for the actual A5 W4A4C8 frontend.

This deliberately does *not* allocate a tensor, select a device, launch a
kernel, build an OPP, or infer an operator source directory.  It is the one
cheap remote check needed before we derive a Stage-A sibling from the actual
``torch_npu.npu_quant_lightning_indexer`` implementation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
TARGET_ROOT = THIS_DIR.parent
OPS_ROOT = TARGET_ROOT.parent
sys.path.insert(0, str(OPS_ROOT))

from a5_dsa_c8_exact_speculative_resident.product_w4a4c8_abi_gate import (  # noqa: E402
    FALLBACK_CALLABLE,
    GateFailure,
    PRIMARY_CALLABLE,
    verify_product_w4a4c8_source,
)


def _one_line(value: object | None) -> str:
    if value is None:
        return "<unset>"
    return str(value).replace("\n", " ").replace("\r", " ")


def _callable_info(owner: object | None, attribute: str) -> tuple[int, str]:
    value = getattr(owner, attribute, None) if owner is not None else None
    if not callable(value):
        return 0, "<absent>"
    module = getattr(value, "__module__", "<native>")
    name = getattr(value, "__qualname__", getattr(value, "__name__", "<native>"))
    return 1, _one_line(f"{module}.{name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--product-vllm-root",
        default=os.environ.get("P10_PRODUCT_VLLM_ROOT"),
        help="vLLM worktree containing the W4A4C8 DSA runtime source",
    )
    args = parser.parse_args()
    if not args.product_vllm_root:
        print("P10_W4A4C8_PROBE_BLOCKED reason=product_vllm_root_missing")
        return 2

    try:
        product = verify_product_w4a4c8_source(Path(args.product_vllm_root))
    except GateFailure as error:
        print(f"P10_W4A4C8_SOURCE ready=0 reason={error}")
        return 2

    print(
        "P10_W4A4C8_SOURCE "
        f"ready=1 root={product.product_vllm_root} head={product.product_source_sha} "
        f"layout={product.architecture} callable={product.product_callable} "
        f"qk={product.query_key_dtype} weights={product.indexer_weight_dtype} "
        f"scales={product.scale_dtype}"
    )

    try:
        import torch  # type: ignore[import-not-found]
        import torch_npu  # type: ignore[import-not-found]
    except Exception as error:
        print(
            "P10_W4A4C8_RUNTIME ready=0 "
            f"reason=torch_or_torch_npu_import_failed:{type(error).__name__}"
        )
        return 2

    primary, primary_origin = _callable_info(
        torch_npu, "npu_quant_lightning_indexer"
    )
    ascend = getattr(torch.ops, "_C_ascend", None)
    fallback, fallback_origin = _callable_info(
        ascend, "npu_lightning_indexer_quant"
    )
    print(
        "P10_W4A4C8_RUNTIME "
        f"primary={primary} primary_name={PRIMARY_CALLABLE} "
        f"primary_origin={primary_origin} fallback={fallback} "
        f"fallback_name={FALLBACK_CALLABLE} fallback_origin={fallback_origin} "
        f"torch={_one_line(getattr(torch, '__version__', None))} "
        f"torch_npu={_one_line(getattr(torch_npu, '__version__', None))} "
        f"torch_npu_file={_one_line(getattr(torch_npu, '__file__', None))}"
    )
    print(
        "P10_W4A4C8_OPP_ENV "
        f"ASCEND_OPP_PATH={_one_line(os.environ.get('ASCEND_OPP_PATH'))} "
        f"ASCEND_CUSTOM_OPP_PATH={_one_line(os.environ.get('ASCEND_CUSTOM_OPP_PATH'))} "
        f"ASCEND_HOME_PATH={_one_line(os.environ.get('ASCEND_HOME_PATH'))}"
    )
    if primary != 1:
        print(
            "P10_W4A4C8_STAGE_A ready=0 "
            "reason=primary_runtime_callable_unavailable"
        )
        return 2
    print(
        "P10_W4A4C8_STAGE_A ready=0 "
        "reason=exact_w4a4c8_operator_source_and_opp_provenance_unresolved"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
