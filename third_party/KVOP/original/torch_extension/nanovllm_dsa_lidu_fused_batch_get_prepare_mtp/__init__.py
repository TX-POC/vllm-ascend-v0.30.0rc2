"""Standalone loader for A5LiduFusedBatchGetPrepareMtp."""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_OPAPI_LIBS = tuple(
    (_ROOT / "_custom_opp" / "vendors").glob("*/op_api/lib/libcust_opapi.so")
)
if len(_OPAPI_LIBS) != 1:
    raise RuntimeError(
        "nanovllm_dsa_lidu_fused_batch_get_prepare_mtp requires one local "
        "libcust_opapi.so. Run bash csrc/attention/nanovllm_a5_ops/"
        "a5_lidu_fused_batch_get_prepare_mtp/build_lidu_fused_batch_get_prepare_mtp.sh"
    )
_OPAPI = _OPAPI_LIBS[0]
_VENDOR = _OPAPI.parents[2]

_vendors = [item for item in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(":") if item]
if str(_VENDOR) not in _vendors:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join([str(_VENDOR), *_vendors])

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

from . import _C  # noqa: E402

_C.set_opapi_path(str(_OPAPI))

lidu_fused_batch_get_prepare_mtp = (
    torch.ops.nanovllm_dsa.lidu_fused_batch_get_prepare_mtp
)
lidu_fused_batch_get_prepare_mtp_out = (
    torch.ops.nanovllm_dsa.lidu_fused_batch_get_prepare_mtp_out
)

__all__ = ["lidu_fused_batch_get_prepare_mtp", "lidu_fused_batch_get_prepare_mtp_out"]
