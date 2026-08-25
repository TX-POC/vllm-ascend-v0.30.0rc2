"""Prototype-only registration for A5ExactScanSpeculativePostProbe."""

from __future__ import annotations

import os
from pathlib import Path


# This prototype is built into an OPP directory separate from the normal
# nanovllm_dsa_a5 package.  Its C++ bridge resolves ACLNN entry points through
# NANOVLLM_CUST_OPAPI_LIB, so choose this probe's libcust_opapi before loading
# the bridge.  The P0 test is deliberately standalone; mixed use with the
# normal custom-op package requires a future multi-library resolver.
_OPS_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OPP = _OPS_ROOT / "_custom_opp_exact_scan_speculative_post_probe"
_LOCAL_OPP = Path(os.getenv("NANOVLLM_A5_PROBE_OPP", str(_DEFAULT_OPP)))
_OPAPI_LIBS = tuple(
    (_LOCAL_OPP / "vendors").glob("*/op_api/lib/libcust_opapi.so")
)
if len(_OPAPI_LIBS) != 1:
    raise RuntimeError(
        "nanovllm_exact_scan_probe requires exactly one libcust_opapi.so in "
        f"{_LOCAL_OPP}; build the isolated P0 OPP first, or set "
        "NANOVLLM_A5_PROBE_OPP."
    )

_OPAPI = _OPAPI_LIBS[0]
_VENDOR = _OPAPI.parents[2]
_existing = [
    value for value in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(":") if value
]
if str(_VENDOR) not in _existing:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join([str(_VENDOR), *_existing])
os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(_OPAPI)

from . import _C  # noqa: E402,F401
