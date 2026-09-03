"""Runtime binding for the one P10 forward product assembly.

This module only loads a single OPP that contains prepare, score receipt, and
resident executor entry points from the same source pair.  It does not search
for independently built packages or offer an alternate host submission path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _require_p10_direct_dispatch_contract() -> None:
    """Reject an unsafe TQ1 process before importing the NPU runtime.

    P10's product bridge issues three caller-stream ACLNN graph nodes.  It
    does not hand a closure to torch_npu's task queue, so using TQ1 would make
    framework ordering and host ownership ambiguous.  The product runner must
    set these variables before starting Python/torch_npu; this import-time
    gate also catches accidental embedded use with a conflicting environment.
    """

    task_queue = os.environ.get("TASK_QUEUE_ENABLE")
    launch_blocking = os.environ.get("ASCEND_LAUNCH_BLOCKING", "0")
    if task_queue != "0" or launch_blocking != "0":
        raise RuntimeError(
            "P10_DIRECT_DISPATCH_BLOCKED requires TASK_QUEUE_ENABLE=0 and "
            "ASCEND_LAUNCH_BLOCKING=0 before torch_npu initialization; "
            f"TASK_QUEUE_ENABLE={task_queue!r} "
            f"ASCEND_LAUNCH_BLOCKING={launch_blocking!r}"
        )
    sys.stderr.write(
        "P10_DIRECT_DISPATCH_CONFIG task_queue_enable=0 "
        f"launch_blocking={launch_blocking} owner=caller_stream "
        "task_queue_fallback=forbidden phase=pre_torch_npu_import\\n"
    )


_require_p10_direct_dispatch_contract()

_OPAPI = Path(os.environ.get("NANOVLLM_A5_P10_PRODUCT_OPAPI_LIB", "")).expanduser()
_VENDOR = Path(os.environ.get("NANOVLLM_A5_P10_PRODUCT_OPP_VENDOR", "")).expanduser()
if not _OPAPI.is_file() or _OPAPI.name != "libcust_opapi.so":
    raise RuntimeError(
        "P10 forward bridge requires NANOVLLM_A5_P10_PRODUCT_OPAPI_LIB to name "
        f"the exact combined product libcust_opapi.so; got {_OPAPI!s}"
    )
if not _VENDOR.is_dir():
    raise RuntimeError(
        "P10 forward bridge requires NANOVLLM_A5_P10_PRODUCT_OPP_VENDOR to name "
        f"the matching combined product vendor directory; got {_VENDOR!s}"
    )

# One vendor owns all three P10 entry points.  Do not retain incidental
# vendors from older prototype packages: same-name ACLNN symbols would make
# the bridge's exact source/ABI provenance ambiguous.
os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(_VENDOR.resolve())
os.environ["NANOVLLM_A5_P10_PRODUCT_OPAPI_LIB"] = str(_OPAPI.resolve())

import torch  # noqa: E402,F401
import torch_npu  # noqa: E402,F401

from . import _C  # noqa: E402,F401
from .p10_forward_graph import P10ForwardLaunch, launch_p10_forward  # noqa: E402,F401

__all__ = ["P10ForwardLaunch", "launch_p10_forward"]
