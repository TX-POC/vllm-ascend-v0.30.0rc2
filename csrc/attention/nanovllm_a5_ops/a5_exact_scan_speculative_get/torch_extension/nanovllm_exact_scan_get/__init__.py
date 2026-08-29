"""Typed bridge for the P5 ``A5ExactScanSpeculativeGet`` custom OPP.

The live P5 runner makes this package visible before its first ACLNN call so
CANN can discover the P5 binary metadata alongside the official C8 and P3
packages.  This bridge still selects the P5 generated OPAPI DSO before
loading ``_C``: the C++ ACLNN launcher resolves its symbols through
``NANOVLLM_CUST_OPAPI_LIB``.  Selection of an OPAPI DSO and discovery of every
custom OPP package are distinct concerns; the former must not erase the
latter's search path.
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
# Keep every package that was registered before the first ACLNN call.  P5 is
# put first for clarity, but CANN has already discovered the official C8, P3,
# and P5 package metadata by the time this bridge switches the OPAPI resolver.
_VENDOR_TEXT = str(_VENDOR)
_EXISTING = [
    value
    for value in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(":")
    if value and value != _VENDOR_TEXT
]
os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join([_VENDOR_TEXT, *_EXISTING])
os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(_OPAPI)

if os.getenv("NANOVLLM_A5_OPAPI_DEBUG") not in (None, "", "0"):
    print(
        "P5_OPAPI_BIND "
        f"opp_root={_LOCAL_OPP} vendor={_VENDOR} opapi={_OPAPI} "
        f"ascend_custom_opp_path={os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')} "
        "preserved_multi_opp_path=1",
        flush=True,
    )

from . import _C  # noqa: E402,F401

__all__ = []
