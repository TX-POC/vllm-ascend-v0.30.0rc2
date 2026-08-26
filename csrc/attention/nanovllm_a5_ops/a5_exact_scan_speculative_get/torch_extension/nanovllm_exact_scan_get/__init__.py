"""Prototype-only registration for A5ExactScanSpeculativeGet.

This bridge is intentionally built against an isolated P1 OPP package.  It
must select that package *before* loading ``_C``: the C++ ACLNN launcher
resolves the generated API symbols through ``NANOVLLM_CUST_OPAPI_LIB``.  An
ambient normal/custom OPP with an older operator ABI can export the same
symbol names but accept a different argument list, which otherwise surfaces
as a misleading generic dtype/format error at GetWorkspaceSize time.
"""

from __future__ import annotations

import os
from pathlib import Path


_OPS_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OPP = _OPS_ROOT / "_custom_opp_exact_scan_speculative_get"
_LOCAL_OPP = Path(
    os.getenv("NANOVLLM_A5_EXACT_SCAN_OPP", str(_DEFAULT_OPP))
).resolve()
_OPAPI_LIBS = tuple(
    (_LOCAL_OPP / "vendors").glob("*/op_api/lib/libcust_opapi.so")
)
if len(_OPAPI_LIBS) != 1:
    raise RuntimeError(
        "nanovllm_exact_scan_get requires exactly one libcust_opapi.so in "
        f"{_LOCAL_OPP}; build the isolated P1 OPP first, or set "
        "NANOVLLM_A5_EXACT_SCAN_OPP to that OPP root."
    )

_OPAPI = _OPAPI_LIBS[0].resolve()
_VENDOR = _OPAPI.parents[2]
_existing = [
    value for value in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(":")
    if value
]
if str(_VENDOR) not in _existing:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(
        [str(_VENDOR), *_existing]
    )
os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(_OPAPI)

if os.getenv("NANOVLLM_A5_OPAPI_DEBUG") not in (None, "", "0"):
    print(
        "P1_OPAPI_BIND "
        f"opp_root={_LOCAL_OPP} vendor={_VENDOR} opapi={_OPAPI} "
        f"ascend_custom_opp_path={os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}",
        flush=True,
    )

from . import _C  # noqa: E402,F401

__all__ = []
