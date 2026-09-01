"""Bind the P7.1 bridge to the one fresh assembly selected by its runner.

P7 does not search for an OPP, a bridge, or a KVAF build. The runner creates
one short ``assembly.env`` after all three have been built from the same two
Git commits, then this module only accepts the paths recorded in that file.
"""

from __future__ import annotations

import os
import sysconfig
from dataclasses import dataclass
from pathlib import Path


def _load_assembly() -> dict[str, str]:
    manifest_text = os.getenv("P7_ASSEMBLY_MANIFEST", "")
    if not manifest_text:
        raise RuntimeError(
            "P7.1 requires P7_ASSEMBLY_MANIFEST from "
            "run_p7_1_speculative_executor.sh; direct artifact selection is disabled."
        )
    manifest = Path(manifest_text).resolve()
    if not manifest.is_file():
        raise RuntimeError(f"P7.1 assembly manifest is missing: {manifest}")

    fields: dict[str, str] = {}
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RuntimeError(f"P7.1 invalid manifest line {number}: {line!r}")
        key, value = line.split("=", 1)
        if not key or not value or key in fields:
            raise RuntimeError(f"P7.1 invalid manifest field line {number}: {line!r}")
        fields[key] = value

    required = (
        "schema",
        "ready",
        "assembly_root",
        "vllm_sha",
        "kvaf_sha",
        "kvaf_build",
        "batch_get_kernel",
        "batch_put_kernel",
        "opp_vendor",
        "opapi",
        "bridge_python_root",
        "bridge_so",
        "p7_abi",
    )
    missing = [key for key in required if key not in fields]
    if missing or fields.get("schema") != "1" or fields.get("ready") != "1" or \
       fields.get("p7_abi") != "31tensor_18refout":
        raise RuntimeError(
            "P7.1 assembly manifest is incomplete: "
            f"missing={','.join(missing) or 'none'} ready={fields.get('ready')!r}"
        )

    root = Path(fields["assembly_root"]).resolve()
    if root != manifest.parent:
        raise RuntimeError(
            f"P7.1 manifest root mismatch: root={root} manifest={manifest}"
        )

    def within_root(name: str) -> Path:
        candidate = Path(fields[name]).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"P7.1 manifest path escapes its assembly: {name}={candidate}"
            ) from error
        return candidate

    paths = {
        name: within_root(name)
        for name in (
            "kvaf_build",
            "batch_get_kernel",
            "batch_put_kernel",
            "opp_vendor",
            "opapi",
            "bridge_python_root",
            "bridge_so",
        )
    }
    if not paths["kvaf_build"].is_dir() or not paths["opp_vendor"].is_dir():
        raise RuntimeError("P7.1 manifest names a missing build directory")
    for name in ("batch_get_kernel", "batch_put_kernel", "opapi", "bridge_so"):
        if not paths[name].is_file():
            raise RuntimeError(f"P7.1 manifest names a missing artifact: {name}={paths[name]}")

    package = Path(__file__).resolve().parent
    expected_package = paths["bridge_python_root"] / package.name
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    if package != expected_package or paths["bridge_so"] != package / f"_C{suffix}":
        raise RuntimeError(
            "P7.1 bridge package is not the one recorded by the assembly: "
            f"loaded={package} expected={expected_package}"
        )

    for name, env_name in (
        ("vllm_sha", "P7_ASSEMBLY_VLLM_SHA"),
        ("kvaf_sha", "P7_ASSEMBLY_KVAF_SHA"),
    ):
        expected = os.getenv(env_name)
        if expected and expected != fields[name]:
            raise RuntimeError(
                f"P7.1 {name} differs from selected assembly: "
                f"runtime={expected} assembly={fields[name]}"
            )
    fields["_opp_vendor"] = str(paths["opp_vendor"])
    fields["_opapi"] = str(paths["opapi"])
    return fields


FIELDS = _load_assembly()
OPP_VENDOR = Path(FIELDS["_opp_vendor"])
OPAPI = Path(FIELDS["_opapi"])

# Do not append: an old P3/P6 vendor must not win symbol resolution.
os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(OPP_VENDOR)
os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(OPAPI)

if os.getenv("NANOVLLM_A5_OPAPI_DEBUG") not in (None, "", "0"):
    print(
        "P7_1_OPAPI_BIND "
        f"manifest={os.environ['P7_ASSEMBLY_MANIFEST']} vendor={OPP_VENDOR} "
        f"opapi={OPAPI} vllm={FIELDS['vllm_sha']} kvaf={FIELDS['kvaf_sha']} "
        "abi=31tensor_18refout",
        flush=True,
    )

# The bridge links against libc10/libtorch/torch_npu.  A normal vLLM process
# already imports these modules, but the fresh-assembly runner deliberately
# verifies this package in a clean Python child.  Import them before `_C` so
# their loader-visible DSOs are present in both cases; do not encode a
# machine-specific torch library path or silently fall back to another bridge.
import torch  # noqa: E402,F401
import torch_npu  # noqa: E402,F401

from . import _C  # noqa: E402,F401


@dataclass(frozen=True)
class ForwardTerminalLease:
    """Tensor keepalive ownership for one ordered P7 submission.

    The token is created by the C++ bridge together with the exact NPU stream
    that will receive the ordered ACLNN launch. It is captured synchronously
    on the dispatch thread, then may be released from any graph/runtime thread
    once that *same forward's* QSFA/stream terminal has completed.
    """

    token: int


def capture_forward_terminal_lease() -> ForwardTerminalLease:
    """Move the latest ordered-submission token into RuntimeResources.

    Call this immediately after the P7 torch op returns, before scheduling a
    later forward on the same runtime worker. This does not synchronize the
    NPU or alter P7's device-side PostOnly/CQ execution.
    """

    token = int(_C.p7_1_take_launch_token())
    if token <= 0:
        raise RuntimeError(f"P7.1 returned invalid ordered-submission token {token}")
    return ForwardTerminalLease(token=token)


def release_at_forward_terminal(lease: ForwardTerminalLease) -> int:
    """Release Tensor keepalives protected by ``lease``'s stream/token boundary.

    The caller must establish the forward terminal first (in production, after
    the final QSFA reader; in the scoped test, after stream synchronization).
    The underlying registry deliberately releases no Tensor handles from other
    streams. ``aclTensor`` descriptors have already been released immediately
    after ACLNN launch, before HugeMem/PTA TLS cleanup.
    """

    if not isinstance(lease, ForwardTerminalLease):
        raise TypeError("lease must be a P7.1 ForwardTerminalLease")
    if lease.token <= 0:
        raise ValueError(f"P7.1 terminal token must be positive, got {lease.token}")
    return int(_C.p7_1_release_after_forward_terminal(int(lease.token)))


def pending_launch_leases() -> int:
    """Return Tensor-keepalive leases not yet terminally released."""

    return int(_C.p7_1_pending_launch_leases())


__all__ = [
    "ForwardTerminalLease",
    "capture_forward_terminal_lease",
    "pending_launch_leases",
    "release_at_forward_terminal",
]
