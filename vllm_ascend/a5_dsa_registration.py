# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight loader for A5 DSA torch registrations.

This module deliberately does not import ``dsa_offload.ops_nanovllm`` or
install runtime precision policies.  Standalone operator tools only need the
custom OPP paths and the registrations embedded in ``vllm_ascend_C``; loading
the complete model/runtime stack would unnecessarily require an exactly
matching upstream vLLM installation.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import torch
import torch_npu  # noqa: F401


_PACKAGE_ROOT = Path(__file__).resolve().parent
_CUSTOM_OP_ROOT = _PACKAGE_ROOT / "_cann_ops_custom"
_A5_OP_ROOT = _CUSTOM_OP_ROOT / "nanovllm_a5"


def _prepend_env_path(name: str, path: Path) -> None:
    value = str(path)
    entries = [entry for entry in os.getenv(name, "").split(":") if entry]
    if value not in entries:
        os.environ[name] = ":".join([value, *entries])


def _bootstrap_a5_opp() -> Path:
    vendor_dirs: list[Path] = []
    standard_vendors = _CUSTOM_OP_ROOT / "vendors"
    if standard_vendors.is_dir():
        vendor_dirs.extend(
            path for path in sorted(standard_vendors.iterdir()) if path.is_dir()
        )

    a5_vendors = _A5_OP_ROOT / "vendors"
    if a5_vendors.is_dir():
        vendor_dirs.extend(
            path for path in sorted(a5_vendors.iterdir()) if path.is_dir()
        )

    a5_opapi = [
        vendor / "op_api" / "lib" / "libcust_opapi.so"
        for vendor in vendor_dirs
        if vendor.parent == a5_vendors
        and (vendor / "op_api" / "lib" / "libcust_opapi.so").is_file()
    ]
    if len(a5_opapi) != 1:
        raise RuntimeError(
            "A5 DSA requires exactly one packaged libcust_opapi.so under "
            f"{a5_vendors}; found={len(a5_opapi)}. Reinstall vLLM-Ascend "
            "with custom kernels enabled on Ascend 950."
        )

    for vendor in reversed(vendor_dirs):
        _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", vendor)
    os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(a5_opapi[0])
    os.environ["NANOVLLM_A5_INSTALL_OPP_PATH"] = str(_A5_OP_ROOT)
    return a5_opapi[0]


_OPAPI = _bootstrap_a5_opp()

# Import only the native registration owner.  Runtime/model policy modules are
# intentionally left to the normal vLLM-Ascend initialization path.
import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: E402,F401


def require_a5_dsa_registration(required_ops: Iterable[str] = ()) -> None:
    missing = [
        name for name in required_ops if not hasattr(torch.ops.nanovllm_dsa, name)
    ]
    if missing:
        raise RuntimeError(
            "A5 DSA registrations embedded in vllm_ascend_C are incomplete: "
            f"missing={tuple(missing)}. Rebuild vLLM-Ascend from the current "
            "source tree."
        )


def local_opapi_path() -> str:
    return str(_OPAPI)


__all__ = ["local_opapi_path", "require_a5_dsa_registration"]
