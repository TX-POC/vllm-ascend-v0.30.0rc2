#!/usr/bin/env python3
"""Print compact, fail-closed provenance for the actual A5 W4A4C8 frontend.

This deliberately does *not* allocate a tensor, select a device, launch a
kernel, or build an OPP.  It verifies the actual provider ABI and reports
whether the checked-in W4A4C8 Stage-A numerical source base is present.  A
native binary is never treated as source code.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
TARGET_ROOT = THIS_DIR.parent
OPS_ROOT = TARGET_ROOT.parent
STAGE_A_SOURCE_ROOT = TARGET_ROOT / "stage_a_w4a4c8_score_receipt"
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


def _file_sha256(path_value: object | None) -> str:
    """Return binary provenance without treating a binary as source code."""

    if not path_value:
        return "<unavailable>"
    path = Path(str(path_value))
    if not path.is_file():
        return "<unavailable>"
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return "<unavailable>"
    return digest.hexdigest()


def _stage_a_source_state() -> tuple[int, str]:
    """Report only a checked-in source base, never infer one from a binary.

    The checked-in Stage-A source has a real isolated build recipe.  It is
    still fail-closed for runtime selection until that recipe, strict ordered
    W4A4C8 parity, and the full Stage-B graph proof have passed.  Thus
    ``source=1`` never claims that an OPP is installed or runnable.
    """

    required = (
        STAGE_A_SOURCE_ROOT / "VENDORED.md",
        STAGE_A_SOURCE_ROOT / "ops.json",
        STAGE_A_SOURCE_ROOT / "build_stage_a_w4a4c8_score_receipt_opp.sh",
        STAGE_A_SOURCE_ROOT / "op_host" / "a5_p10_w4a4c8_score_receipt_def.cpp",
        STAGE_A_SOURCE_ROOT / "op_kernel" / "a5_p10_w4a4c8_score_receipt.cpp",
        STAGE_A_SOURCE_ROOT / "op_kernel" / "a5_p10_w4a4c8_score_receipt_qli.h",
        STAGE_A_SOURCE_ROOT / "op_kernel" / "a5_p10_w4a4c8_score_receipt_abi.h",
    )
    if not all(path.is_file() for path in required):
        return 0, "checked_in_score_receipt_source_missing"
    return 1, "buildable_score_receipt_source_runtime_parity_pending"


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
    native_module = getattr(torch_npu, "_C", None)
    native_module_file = getattr(native_module, "__file__", None)
    print(
        "P10_W4A4C8_PROVIDER_BINARY "
        f"module_file={_one_line(native_module_file)} "
        f"sha256={_file_sha256(native_module_file)} "
        "source_derivation_permitted=0"
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
    stage_source, stage_reason = _stage_a_source_state()
    print(
        "P10_W4A4C8_STAGE_A_SOURCE "
        f"source={stage_source} root={_one_line(STAGE_A_SOURCE_ROOT)} "
        f"state={stage_reason}"
    )
    print(
        "P10_W4A4C8_STAGE_A ready=0 "
        f"reason={stage_reason}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
