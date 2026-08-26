"""Prototype-only transaction-local ExactReconcilePrivateView registration.

The package must bind exactly one isolated custom OPP before importing the
bridge.  It does not alter the normal product OPP path or select the prototype
from ``a5_c8_policy.py``.
"""

from __future__ import annotations

import os
from pathlib import Path


_OPS_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OPP = _OPS_ROOT / "_custom_opp_exact_reconcile_private_view"
_LOCAL_OPP = Path(
    os.getenv("NANOVLLM_A5_EXACT_RECONCILE_OPP", str(_DEFAULT_OPP))
).resolve()
_OPAPI_LIBS = tuple(
    (_LOCAL_OPP / "vendors").glob("*/op_api/lib/libcust_opapi.so")
)
if len(_OPAPI_LIBS) != 1:
    raise RuntimeError(
        "nanovllm_exact_reconcile_private_view requires exactly one "
        "libcust_opapi.so in "
        f"{_LOCAL_OPP}; build the isolated P3 OPP first, or set "
        "NANOVLLM_A5_EXACT_RECONCILE_OPP to that OPP root."
    )

_OPAPI = _OPAPI_LIBS[0].resolve()
_VENDOR = _OPAPI.parents[2]
os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(_VENDOR)
os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(_OPAPI)

if os.getenv("NANOVLLM_A5_OPAPI_DEBUG") not in (None, "", "0"):
    print(
        "P3_RECONCILE_OPAPI_BIND "
        f"opp_root={_LOCAL_OPP} vendor={_VENDOR} opapi={_OPAPI} "
        "isolated=1",
        flush=True,
    )

from . import _C  # noqa: E402,F401

__all__ = []
