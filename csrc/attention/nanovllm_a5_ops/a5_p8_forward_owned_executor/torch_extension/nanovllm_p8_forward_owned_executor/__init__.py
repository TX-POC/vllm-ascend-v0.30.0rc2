"""P8 Stage-B forward-owned Direct-HCCP executor registration.

The Stage-A score-receipt scanner is intentionally a separately built source
closure.  This package owns only CandidateSeed -> PostOnly/CQ -> QSFA-terminal
private release and therefore never loads a C8 or P7 transport OPP.
"""

from __future__ import annotations

import os
from pathlib import Path


_OPS_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OPP = _OPS_ROOT / "_custom_opp_p8_forward_owned_executor"
_LOCAL_OPP = Path(
    os.getenv("NANOVLLM_A5_P8_FORWARD_OWNED_OPP", str(_DEFAULT_OPP))
).resolve()
_OPAPI_LIBS = tuple(
    (_LOCAL_OPP / "vendors").glob("*/op_api/lib/libcust_opapi.so")
)
if len(_OPAPI_LIBS) != 1:
    raise RuntimeError(
        "nanovllm_p8_forward_owned_executor requires exactly one "
        "libcust_opapi.so in "
        f"{_LOCAL_OPP}; build the isolated P8 OPP first, or set "
        "NANOVLLM_A5_P8_FORWARD_OWNED_OPP."
    )

_OPAPI = _OPAPI_LIBS[0].resolve()
_VENDOR = _OPAPI.parents[2]
# Preserve Stage-A/reference vendors.  P8 uses three independently built
# OPPs in one Python process; overwriting this path makes correctness depend
# on incidental import order rather than the source-closed runner contract.
_existing_vendors = [item for item in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(":") if item]
os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(
    [str(_VENDOR), *(item for item in _existing_vendors if item != str(_VENDOR))]
)
os.environ["NANOVLLM_A5_P8_FORWARD_OWNED_OPAPI"] = str(_OPAPI)

if os.getenv("NANOVLLM_A5_OPAPI_DEBUG") not in (None, "", "0"):
    print(
        "P8_STAGE_B_OPAPI_BIND "
        f"opp_root={_LOCAL_OPP} vendor={_VENDOR} opapi={_OPAPI} "
        "post_cq=forward_owned qsfa_terminal=required "
        "c8_dependency=0 fixed_get_core=0",
        flush=True,
    )

from . import _C  # noqa: E402,F401

__all__ = []
