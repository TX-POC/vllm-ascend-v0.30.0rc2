#!/usr/bin/env python3
"""Fresh-process P8 Stage-B ACLNN fail-closed probe.

This is a diagnostic *boundary* probe, not an alternate P8 execution path.
It imports only the freshly staged P8 Stage-B bridge and its matching OPP,
then invokes the existing 28-input ``forward_owned_executor`` with:

* one geometrically valid receipt whose CandidateSeed count is zero; and
* an all-zero Direct-HCCP control tensor.

The device kernel must therefore reject before queue leasing, descriptor
publication, PostOnly, CQ polling, or private-slot mutation.  The probe is
deliberately independent of C8, Stage-A and KVAF so a failure at
``GetWorkspaceSize`` or task-queue dispatch can be attributed to the Stage-B
bridge/OPP boundary instead of a preceding graph node.

Run it in a *new* Python process after the P8 Stage-B OPP and bridge have
been built.  TASK_QUEUE_ENABLE=0 and ASCEND_LAUNCH_BLOCKING=0 must be set
before Python starts.  This is the P8 TQ0 direct acceptance path: the caller
submits its graph nodes directly to one current NPU stream, so the Stage-B
launch keeps FIFO stream order without relying on torch_npu private queue
internals.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Iterable

import torch
import torch_npu  # noqa: F401 -- registers PrivateUse1/NPU.


INPUT_COUNT = 28
MUTABLE_INPUTS = (
    4,
    5,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    2,
    3,
)
CONTROL_REJECTED_STAT_COLUMN = 10
PACKED_KV_ROW_BYTES = 656
CANDIDATE_CAPACITY = 128
CANDIDATE_SEED_WORDS = 4
CANDIDATE_IDENTITY_WORDS = 4
FORWARD_CONTROL_WORDS = 4
STREAMING_CONTROL_WORDS = 16
WAVE_RECORD_WORDS = 32
PRIVATE_RECORD_WORDS = 10
REQUEST_CAPACITY = 64
WORKER_STAT_COLUMNS = 16


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--stage-b-bridge-root",
        default=os.environ.get("P8_STAGE_B_BRIDGE_ROOT", ""),
        help="Fresh run-root containing nanovllm_p8_forward_owned_executor/.",
    )
    parser.add_argument(
        "--stage-b-opp-root",
        default=os.environ.get("NANOVLLM_A5_P8_FORWARD_OWNED_OPP", ""),
        help="Fresh run-root containing vendors/customize/op_api/lib/libcust_opapi.so.",
    )
    return parser.parse_args()


def _require_directory(label: str, raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not raw or not path.is_dir():
        raise RuntimeError(f"P8_STAGE_B_PROBE_BLOCKED reason=missing_{label} path={raw!r}")
    return path


def _under_root(label: str, actual: str | os.PathLike[str], root: Path) -> Path:
    resolved = Path(actual).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            "P8_STAGE_B_PROBE_BLOCKED "
            f"reason=stale_{label} actual={resolved} expected_root={root}"
        ) from exc
    return resolved


def _require_tq0_direct_environment() -> int:
    task_queue = os.environ.get("TASK_QUEUE_ENABLE", "")
    launch_blocking = os.environ.get("ASCEND_LAUNCH_BLOCKING", "")
    if task_queue != "0" or launch_blocking not in ("", "0"):
        raise RuntimeError(
            "P8_STAGE_B_PROBE_BLOCKED "
            "reason=unsafe_tq0_direct_environment "
            f"TASK_QUEUE_ENABLE={task_queue or '<unset>'} "
            f"ASCEND_LAUNCH_BLOCKING={launch_blocking or '<unset>'}"
        )
    # torch_npu has no stable public Python getter for the latched task-queue
    # state.  Do not replace this fresh-process gate with best-effort runtime
    # introspection: that would turn a compatibility difference into a false
    # pass or an AttributeError before Stage-B is exercised.
    return 0


def _npu(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device).contiguous()


def _make_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Return the exact 28-input Stage-B ABI with no candidate/Post work.

    The one-receipt/one-queue geometry is enough for the host tiler to create
    one AIV worker.  The all-zero ``fused_get_control`` is intentionally the
    first kernel validation failure; no other value below is relied upon by
    the fail-closed branch.  Shapes and dtypes nevertheless satisfy the full
    production bridge and OPP ABI, so this remains a useful GetWorkspace
    boundary test rather than a malformed-input test.
    """

    receipts = 1
    rows = 1
    sources = 1
    pools = 1
    objects = 1
    queues = 1
    private_rows = 1
    workers = 1
    forward_control = torch.tensor(
        [37, 0, 1 << 32, 0], dtype=torch.int64
    )
    inputs: tuple[torch.Tensor, ...] = (
        torch.zeros((receipts, CANDIDATE_CAPACITY, CANDIDATE_SEED_WORDS), dtype=torch.int32),
        torch.zeros((receipts,), dtype=torch.int32),
        torch.zeros((receipts, CANDIDATE_IDENTITY_WORDS), dtype=torch.int64),
        torch.zeros((receipts,), dtype=torch.int32),
        forward_control,
        torch.zeros((1,), dtype=torch.int32),
        torch.ones((rows,), dtype=torch.int32),
        torch.ones((1,), dtype=torch.int32),
        torch.zeros((rows,), dtype=torch.int32),
        torch.zeros((pools, sources), dtype=torch.int32),
        torch.ones((rows,), dtype=torch.int32),
        torch.full((pools, objects), -1, dtype=torch.int32),
        torch.full((rows, sources), -1, dtype=torch.int32),
        torch.full((pools, objects), -1, dtype=torch.int32),
        torch.zeros((private_rows,), dtype=torch.int32),
        torch.zeros((1,), dtype=torch.int32),
        torch.zeros((private_rows, PACKED_KV_ROW_BYTES), dtype=torch.uint8),
        torch.zeros((STREAMING_CONTROL_WORDS,), dtype=torch.int64),
        torch.zeros((queues, WAVE_RECORD_WORDS), dtype=torch.int64),
        torch.zeros((private_rows, PRIVATE_RECORD_WORDS), dtype=torch.int64),
        torch.zeros((queues, REQUEST_CAPACITY), dtype=torch.int64),
        torch.zeros((queues, REQUEST_CAPACITY), dtype=torch.int64),
        torch.full((queues, REQUEST_CAPACITY), -1, dtype=torch.int32),
        torch.zeros((queues,), dtype=torch.int32),
        torch.zeros((receipts,), dtype=torch.int32),
        torch.zeros((receipts, CANDIDATE_IDENTITY_WORDS), dtype=torch.int64),
        torch.ones((1,), dtype=torch.int32),
        torch.zeros((workers, WORKER_STAT_COLUMNS), dtype=torch.int64),
    )
    if len(inputs) != INPUT_COUNT:
        raise AssertionError(f"P8 Stage-B probe built {len(inputs)} inputs, expected {INPUT_COUNT}")
    return tuple(_npu(tensor, device) for tensor in inputs)


def _assert_aliases(outputs: Iterable[torch.Tensor], inputs: tuple[torch.Tensor, ...]) -> None:
    outputs = tuple(outputs)
    if len(outputs) != len(MUTABLE_INPUTS):
        raise AssertionError(
            f"P8 Stage-B probe returned {len(outputs)} aliases, expected {len(MUTABLE_INPUTS)}"
        )
    for output, index in zip(outputs, MUTABLE_INPUTS, strict=True):
        if output.data_ptr() != inputs[index].data_ptr():
            raise AssertionError(
                f"P8 Stage-B probe alias mismatch output_input_index={index}"
            )


def _assert_fail_closed(inputs: tuple[torch.Tensor, ...], before: tuple[torch.Tensor, ...]) -> int:
    for index in MUTABLE_INPUTS:
        if index == 27:
            continue
        if not torch.equal(inputs[index].detach().cpu(), before[index].detach().cpu()):
            raise AssertionError(f"P8 Stage-B probe unexpected_mutation input={index}")
    delta = inputs[27].detach().cpu().to(torch.int64) - before[27].detach().cpu().to(torch.int64)
    allowed = torch.zeros_like(delta)
    allowed[:, CONTROL_REJECTED_STAT_COLUMN] = delta[:, CONTROL_REJECTED_STAT_COLUMN]
    if not torch.equal(delta, allowed):
        raise AssertionError("P8 Stage-B probe worker_stats changed outside control_rejected")
    rejected = int(delta[:, CONTROL_REJECTED_STAT_COLUMN].sum())
    if rejected <= 0:
        raise AssertionError("P8 Stage-B probe did not report control_rejected")
    return rejected


def main() -> None:
    args = _parse_args()
    requested_task_queue = _require_tq0_direct_environment()
    bridge_root = _require_directory("stage_b_bridge_root", args.stage_b_bridge_root)
    opp_root = _require_directory("stage_b_opp_root", args.stage_b_opp_root)
    opapi = opp_root / "vendors" / "customize" / "op_api" / "lib" / "libcust_opapi.so"
    package_init = bridge_root / "nanovllm_p8_forward_owned_executor" / "__init__.py"
    if not opapi.is_file() or not package_init.is_file():
        raise RuntimeError(
            "P8_STAGE_B_PROBE_BLOCKED reason=missing_stage_b_artifact "
            f"opapi={opapi} package={package_init}"
        )
    forbidden = (
        "nanovllm_official_c8_reference",
        "nanovllm_score_receipt_executor",
        "kvaf",
    )
    present = [name for name in forbidden if name in sys.modules]
    if present:
        raise RuntimeError(
            "P8_STAGE_B_PROBE_BLOCKED reason=forbidden_module_preloaded "
            f"modules={','.join(present)}"
        )
    # Start with a Stage-B-only OPP search path.  This new Python process must
    # not inherit a C8 or Stage-A vendor from a preceding full-P8 invocation.
    os.environ.pop("ASCEND_CUSTOM_OPP_PATH", None)
    os.environ.pop("NANOVLLM_A5_REFERENCE_C8_OPAPI_LIB", None)
    os.environ.pop("NANOVLLM_A5_SCORE_RECEIPT_OPAPI", None)
    os.environ["NANOVLLM_A5_P8_FORWARD_OWNED_OPP"] = str(opp_root)
    sys.path.insert(0, str(bridge_root))

    import nanovllm_p8_forward_owned_executor as stage_b  # noqa: PLC0415

    package_path = _under_root("stage_b_package", stage_b.__file__ or "", bridge_root)
    native_path = _under_root("stage_b_extension", stage_b._C.__file__, bridge_root)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    inputs = _make_inputs(device)
    candidate_count = int(inputs[1].detach().cpu().sum())
    fused_control_nonzero = int(torch.count_nonzero(inputs[17].detach().cpu()))
    print(
        "P8_STAGE_B_PROBE_CONFIG "
        f"device={device} physical={os.environ.get('P8_PHYSICAL_NPU_ID', 'unspecified')} "
        f"visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'unset')!r} "
        f"task_queue_requested={requested_task_queue} "
        "task_queue_runtime_guard=fresh_process_tq0 "
        f"inputs={len(inputs)} candidate_count={candidate_count} "
        f"fused_control_nonzero={fused_control_nonzero} "
        f"bridge={package_path.parent} native={native_path} opapi={opapi}",
        flush=True,
    )
    before = tuple(tensor.detach().clone() for tensor in inputs)
    print("P8_STAGE_B_PROBE_STEP label=submit_begin", flush=True)
    outputs = torch.ops.a5_p8_forward_owned_executor.forward_owned_executor(*inputs)
    print("P8_STAGE_B_PROBE_STEP label=submit_return", flush=True)
    torch.npu.synchronize()
    print("P8_STAGE_B_PROBE_STEP label=synchronize_return", flush=True)
    _assert_aliases(outputs, inputs)
    rejected = _assert_fail_closed(inputs, before)
    print(
        "P8_STAGE_B_PROBE_PASS "
        f"aliases=19 control_rejected={rejected} post=0 cq=0 "
        "c8_import=0 stage_a_import=0 kvaf_import=0",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            "P8_STAGE_B_PROBE_FAIL "
            f"type={type(error).__name__} text={error}",
            flush=True,
        )
        raise
