"""Bind exactly one fresh P7.1 OPP before loading the isolated bridge.

P7.1 is C8-free by construction.  Importing it must never resolve a P3/P6
OPP left in ``ASCEND_CUSTOM_OPP_PATH``; the runner supplies a newly built OPP
root and this module resets the custom-op resolver to that single vendor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OPP = OPS_ROOT / "_custom_opp_score_receipt_speculative_executor"
OPP_ROOT = Path(
    os.getenv(
        "NANOVLLM_A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_OPP", str(DEFAULT_OPP)
    )
).resolve()
MANIFESTS = tuple(
    (OPP_ROOT / "vendors").glob(
        "*/P7_1_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_PROVENANCE"
    )
)
if len(MANIFESTS) != 1:
    raise RuntimeError(
        "P7.1 requires exactly one fresh provenance manifest under "
        f"{OPP_ROOT}/vendors; set "
        "NANOVLLM_A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_OPP to the current build."
    )

MANIFEST = MANIFESTS[0].resolve()
FIELDS: dict[str, str] = {}
for line in MANIFEST.read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        FIELDS[key] = value

REQUIRED = {
    "kind": "standalone_score_receipt_speculative_executor",
    "abi": "31tensor_18refout",
    "torch_schema": "31in_18aliasout",
    "local_microbatches": "2x64",
    "queue_lifecycle": "epoch,generation,post_sequence,state,quarantine",
    "post_cq": "real_kvaf_streaming_postonly_tryretire_finish",
    "c8_source_closure": "absent",
    "fixed_get_core": "absent",
}
for key, expected in REQUIRED.items():
    if FIELDS.get(key) != expected:
        raise RuntimeError(
            f"P7.1 OPP manifest mismatch for {key}: "
            f"actual={FIELDS.get(key)!r} expected={expected!r}; manifest={MANIFEST}"
        )

expected_source = os.getenv("P7_EXPECTED_VLLM_SHA")
if expected_source and FIELDS.get("source_head") != expected_source:
    raise RuntimeError(
        "P7.1 OPP source revision mismatch: "
        f"actual={FIELDS.get('source_head')} expected={expected_source}"
    )
expected_kvaf = os.getenv("P7_EXPECTED_KVAF_SHA")
if expected_kvaf and FIELDS.get("kvaf_source_head") != expected_kvaf:
    raise RuntimeError(
        "P7.1 OPP KVAF revision mismatch: "
        f"actual={FIELDS.get('kvaf_source_head')} expected={expected_kvaf}"
    )

OPAPI = MANIFEST.parent / "op_api" / "lib" / "libcust_opapi.so"
if not OPAPI.is_file():
    raise RuntimeError(f"P7.1 OPP has no op_api library: {OPAPI}")

# Do not append: an older P3/P6 vendor can otherwise win symbol resolution.
os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(MANIFEST.parent)
os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(OPAPI)

if os.getenv("NANOVLLM_A5_OPAPI_DEBUG") not in (None, "", "0"):
    print(
        "P7_1_OPAPI_BIND "
        f"opp_root={OPP_ROOT} vendor={MANIFEST.parent} opapi={OPAPI} "
        f"source_head={FIELDS.get('source_head')} "
        f"kvaf_head={FIELDS.get('kvaf_source_head')} abi=31tensor_18refout",
        flush=True,
    )

from . import _C  # noqa: E402,F401


@dataclass(frozen=True)
class ForwardTerminalLease:
    """Host ownership for one direct P7 launch until its forward terminal.

    The token is created by the C++ bridge together with the exact NPU stream
    that received the ACLNN launch.  It is captured synchronously on the
    dispatch thread, then may be released from any graph/runtime thread once
    that *same forward's* QSFA/stream terminal has completed.
    """

    token: int


def capture_forward_terminal_lease() -> ForwardTerminalLease:
    """Move the latest direct-launch token into forward RuntimeResources.

    Call this immediately after the P7 torch op returns, before scheduling a
    later forward on the same runtime worker.  This does not synchronize the
    NPU or alter P7's device-side PostOnly/CQ execution.
    """

    token = int(_C.p7_1_take_launch_token())
    if token <= 0:
        raise RuntimeError(f"P7.1 returned invalid direct-launch token {token}")
    return ForwardTerminalLease(token=token)


def release_at_forward_terminal(lease: ForwardTerminalLease) -> int:
    """Release only descriptors protected by ``lease``'s stream/token boundary.

    The caller must establish the forward terminal first (in production, after
    the final QSFA reader; in the scoped test, after stream synchronization).
    The underlying registry deliberately releases no work from other streams.
    """

    if not isinstance(lease, ForwardTerminalLease):
        raise TypeError("lease must be a P7.1 ForwardTerminalLease")
    if lease.token <= 0:
        raise ValueError(f"P7.1 terminal token must be positive, got {lease.token}")
    return int(_C.p7_1_release_after_forward_terminal(int(lease.token)))


def pending_launch_leases() -> int:
    """Return the bounded host-descriptor leases not yet terminally released."""

    return int(_C.p7_1_pending_launch_leases())


__all__ = [
    "ForwardTerminalLease",
    "capture_forward_terminal_lease",
    "pending_launch_leases",
    "release_at_forward_terminal",
]
