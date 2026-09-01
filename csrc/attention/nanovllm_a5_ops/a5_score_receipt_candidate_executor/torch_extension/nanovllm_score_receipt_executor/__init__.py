"""Isolated P7.0 score-receipt/candidate oracle registration.

This package consumes a completed C8 score publisher receipt.  It deliberately
contains no KVAF, HCCP, queue lease, private-slot, or QSFA implementation and
therefore must not be described as the P7 asynchronous Get executor.
"""

from __future__ import annotations

import os
from pathlib import Path


_OPS_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_OPP = _OPS_ROOT / "_custom_opp_score_receipt_candidate_executor"
_LOCAL_OPP = Path(
    os.getenv("NANOVLLM_A5_SCORE_RECEIPT_EXECUTOR_OPP", str(_DEFAULT_OPP))
).resolve()
_OPAPI_LIBS = tuple(
    (_LOCAL_OPP / "vendors").glob("*/op_api/lib/libcust_opapi.so")
)
if len(_OPAPI_LIBS) != 1:
    raise RuntimeError(
        "nanovllm_score_receipt_executor requires exactly one "
        "libcust_opapi.so in "
        f"{_LOCAL_OPP}; build the isolated P7.0 OPP first, or set "
        "NANOVLLM_A5_SCORE_RECEIPT_EXECUTOR_OPP."
    )

_OPAPI = _OPAPI_LIBS[0].resolve()
_VENDOR = _OPAPI.parents[2]
# Several independently built P8 closures may coexist in one process
# (unchanged C8 reference, Stage A, and Stage B).  CANN accepts a colon
# separated vendor list.  Preserve vendors registered by earlier closures
# instead of replacing the whole list, otherwise import order silently picks
# a different op package at the first ACLNN execution.
_existing_vendors = [item for item in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(":") if item]
os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(
    [str(_VENDOR), *(item for item in _existing_vendors if item != str(_VENDOR))]
)
os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(_OPAPI)

if os.getenv("NANOVLLM_A5_OPAPI_DEBUG") not in (None, "", "0"):
    print(
        "P7_SCORE_RECEIPT_ORACLE_OPAPI_BIND "
        f"opp_root={_LOCAL_OPP} vendor={_VENDOR} opapi={_OPAPI} "
        "post_cq=0 c8_dependency=0 kvaf_dependency=0",
        flush=True,
    )

from . import _C  # noqa: E402,F401

__all__ = []
