#!/usr/bin/env python3
"""P7.1 bridge/lifecycle diagnostic with compact, copyable receipts.

Default ``fail_closed`` mode launches the real P7.1 ACLNN op with a malformed
zero KVAF control block.  It proves 31-in/18-alias dispatch and that the
kernel refuses to touch queues before a valid Direct-HCCP binding exists.

``--fixture-module`` is the real transport mode.  The module must define
``build_p7_1_fixture(device)`` and return ``{"inputs": tuple_of_31_tensors}``
whose `fused_get_control` has already been cold-bound with KVAF
``BindStreamingBatchGetControl`` + ``ConfigureStreamingBatchGetQueueTemplates``.
This test deliberately does not fabricate that control: a fake pointer table
would make PostOnly/CQ evidence meaningless.  An optional ``validate(outputs)``
callback verifies payload/transaction-specific expectations.
"""

from __future__ import annotations

import argparse
import gc
import importlib
from typing import Any, Iterable

import torch
import torch_npu  # noqa: F401


INPUT_COUNT = 31
OUTPUT_COUNT = 18
MUTABLE_INPUTS = (11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29)
STAT_LEASE = 3
STAT_POST = 7
STAT_RETIRE = 8
STAT_READY = 9
STAT_FAILED = 10
STAT_RELEASE = 11
STAT_QUARANTINE = 12
STAT_CONTROL_REJECTED = 13
STAT_FINISH = 15
QUEUE_FREE = 0
QUEUE_QUARANTINED = 4


def _fail_closed_fixture(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Valid tensor geometry + invalid control: assert fail-closed before HCCP."""

    receipts, rows, sources, pools, objects, queues, arena, workers = 1, 1, 128, 1, 128, 1, 64, 1
    npu = lambda tensor: tensor.to(device).contiguous()
    # This is intentionally a tuple literal, not ``tuple(a, b, ...)``:
    # Python's tuple constructor accepts exactly one iterable argument.  Keep
    # the 31-input ABI visible here and assert it before ACLNN dispatch so a
    # test-harness typo cannot be mistaken for an operator/transport failure.
    inputs = (
        npu(torch.full((receipts, 128, 4), -1, dtype=torch.int32)),  # candidate_seeds
        npu(torch.zeros(receipts, dtype=torch.int32)),
        npu(torch.tensor([41], dtype=torch.int64)),
        npu(torch.ones(receipts, dtype=torch.int32)),
        npu(torch.tensor([41], dtype=torch.int64)),
        npu(torch.tensor([sources], dtype=torch.int32)),
        npu(torch.tensor([1], dtype=torch.int32)),  # explicit MAPPED_OBJECT
        npu(torch.zeros(rows, dtype=torch.int32)),
        npu(torch.arange(sources, dtype=torch.int32).reshape(pools, sources)),
        npu(torch.tensor([objects], dtype=torch.int32)),
        npu(torch.full((pools, objects), -1, dtype=torch.int32)),
        npu(torch.full((rows, sources), -1, dtype=torch.int32)),
        npu(torch.full((pools, objects), -1, dtype=torch.int32)),
        npu(torch.zeros((queues, arena), dtype=torch.int32)),
        npu(torch.zeros(queues, dtype=torch.int32)),
        npu(torch.zeros((queues * arena, 656), dtype=torch.uint8)),
        npu(torch.zeros(16, dtype=torch.int64)),  # invalid KVAF magic/version
        npu(torch.zeros(queues, dtype=torch.int32)),
        npu(torch.zeros(queues, dtype=torch.int64)),
        npu(torch.zeros(queues, dtype=torch.int64)),
        npu(torch.zeros(queues, dtype=torch.int64)),
        npu(torch.zeros(queues, dtype=torch.int32)),
        npu(torch.zeros(queues, dtype=torch.int32)),
        npu(torch.zeros((queues, 64), dtype=torch.int64)),
        npu(torch.zeros((queues, 64), dtype=torch.int64)),
        npu(torch.full((queues, 64), -1, dtype=torch.int32)),
        npu(torch.zeros(queues, dtype=torch.int32)),
        npu(torch.zeros((queues, arena), dtype=torch.int64)),
        npu(torch.zeros((queues, arena), dtype=torch.int64)),
        npu(torch.zeros((workers, 16), dtype=torch.int64)),
        npu(torch.ones(1, dtype=torch.int32)),
    )
    if len(inputs) != INPUT_COUNT:
        raise AssertionError(
            f"fail-closed fixture must provide {INPUT_COUNT} inputs, got {len(inputs)}"
        )
    return inputs


def _stat_sum(stats: torch.Tensor, column: int) -> int:
    return int(stats[:, column].detach().cpu().sum().item())


def _first_queue(queue_state: torch.Tensor, queue_quarantine: torch.Tensor) -> int:
    state = queue_state.detach().cpu().reshape(-1)
    quarantine = queue_quarantine.detach().cpu().reshape(-1)
    for index, (state_value, quarantine_value) in enumerate(zip(state, quarantine, strict=True)):
        if int(state_value) == QUEUE_QUARANTINED or int(quarantine_value) != 0:
            return index
    return -1


def _check_aliases(inputs: tuple[torch.Tensor, ...], outputs: Iterable[torch.Tensor]) -> None:
    values = tuple(outputs)
    if len(values) != OUTPUT_COUNT:
        raise AssertionError(f"expected {OUTPUT_COUNT} outputs, got {len(values)}")
    for output_index, input_index in enumerate(MUTABLE_INPUTS):
        if values[output_index].data_ptr() != inputs[input_index].data_ptr():
            raise AssertionError(
                f"output {output_index} is not a physical alias of input {input_index}"
            )


def _run(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != INPUT_COUNT:
        raise ValueError(f"fixture must provide exactly {INPUT_COUNT} inputs")
    # Keep the fail-closed fixture preflight independent of an OPP package.
    # The runner executes that preflight before any expensive KVAF/OPP build;
    # importing the bridge belongs only to the actual ACLNN dispatch path.
    import nanovllm_score_receipt_speculative_executor  # noqa: F401

    outputs = torch.ops.a5_score_receipt_speculative_executor.score_receipt_speculative_executor(*inputs)
    torch.npu.synchronize()
    _check_aliases(inputs, outputs)
    return tuple(outputs)


def _report(mode: str, inputs: tuple[torch.Tensor, ...], outputs: tuple[torch.Tensor, ...]) -> dict[str, int]:
    stats = outputs[-1]
    queue_state = outputs[9]
    queue_quarantine = outputs[10]
    control = inputs[16].detach().cpu().reshape(-1)
    epoch = int(inputs[4].detach().cpu().item())
    queue_count = int(inputs[17].numel())
    receipt_count = int(inputs[0].size(0))
    candidate_total = int(inputs[1].detach().cpu().sum().item())
    values = {
        "lease": _stat_sum(stats, STAT_LEASE),
        "post": _stat_sum(stats, STAT_POST),
        "retire": _stat_sum(stats, STAT_RETIRE),
        "finish": _stat_sum(stats, STAT_FINISH),
        "release": _stat_sum(stats, STAT_RELEASE),
        "ready": _stat_sum(stats, STAT_READY),
        "failed": _stat_sum(stats, STAT_FAILED),
        "quarantine": _stat_sum(stats, STAT_QUARANTINE),
        "control_rejected": _stat_sum(stats, STAT_CONTROL_REJECTED),
    }
    state_cpu = queue_state.detach().cpu()
    quarantine_cpu = queue_quarantine.detach().cpu()
    queue_free = int((state_cpu == QUEUE_FREE).sum().item())
    queue_quarantined = int(
        ((state_cpu == QUEUE_QUARANTINED) | (quarantine_cpu != 0)).sum().item()
    )
    first_queue = _first_queue(queue_state, queue_quarantine)
    diagnostic_queue = first_queue if first_queue >= 0 else 0
    queue_epoch = int(outputs[6].detach().cpu().reshape(-1)[diagnostic_queue])
    queue_generation = int(outputs[7].detach().cpu().reshape(-1)[diagnostic_queue])
    queue_post_sequence = int(outputs[8].detach().cpu().reshape(-1)[diagnostic_queue])
    print(
        "P7_1_SOURCE_ABI "
        f"mode={mode} inputs=31 outputs=18 aliases=18 "
        f"candidate_seed_shape={list(inputs[0].shape)}",
        flush=True,
    )
    print(
        "P7_1_CONTROL "
        f"magic={int(control[0])} version={int(control[1])} words={control.numel()} "
        f"epoch={epoch}",
        flush=True,
    )
    print(
        "P7_1_RECEIPTS "
        f"receipts={receipt_count} candidate_total={candidate_total} capacity=128",
        flush=True,
    )
    print(
        "P7_1_QUEUE "
        f"count={queue_count} free={queue_free} quarantined={queue_quarantined} "
        f"first_quarantine_queue={first_queue}",
        flush=True,
    )
    print(
        "P7_1_TRANSPORT "
        "lease={lease} post={post} retire={retire} finish={finish} "
        "release={release} ready={ready} failed={failed} quarantine={quarantine}".format(
            **values
        ),
        flush=True,
    )
    first_code = "none"
    if values["control_rejected"]:
        first_code = "control_rejected"
    elif values["quarantine"]:
        first_code = "queue_quarantined"
    elif values["failed"]:
        first_code = "cq_failed"
    lifecycle_state = int(state_cpu.reshape(-1)[diagnostic_queue])
    lifecycle_quarantine = int(quarantine_cpu.reshape(-1)[diagnostic_queue])
    print(
        "P7_1_FAILURE "
        f"first_code={first_code} queue={diagnostic_queue} epoch={queue_epoch or epoch} "
        f"generation={queue_generation} post_sequence={queue_post_sequence} "
        f"control_rejected={values['control_rejected']} "
        f"lifecycle_state={lifecycle_state} quarantine={lifecycle_quarantine}",
        flush=True,
    )
    return values


def _teardown_checkpoint(mode: str) -> None:
    """Make Python-level ownership release visible before interpreter teardown.

    `_run` has already synchronized the NPU stream.  This checkpoint therefore
    does not change the transaction; it only makes a future bridge destructor
    failure appear in a compact, attributable position rather than as a bare
    process-exit abort.
    """

    gc.collect()
    torch.npu.synchronize()
    print(
        f"P7_1_TEARDOWN mode={mode} stage=python_refs_collected sync=1",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--fixture-module",
        default="",
        help="Python module exposing build_p7_1_fixture(device) for real Direct-HCCP mode",
    )
    parser.add_argument(
        "--fixture-only",
        action="store_true",
        help="construct and validate only the built-in fail-closed fixture; never import an OPP or dispatch ACLNN",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must be npu:<logical-id>")
    torch.npu.set_device(device)
    if args.fixture_only:
        if args.fixture_module:
            raise ValueError("--fixture-only cannot be combined with --fixture-module")
        inputs = _fail_closed_fixture(device)
        if len(inputs) != INPUT_COUNT:
            raise AssertionError(f"expected {INPUT_COUNT} fixture inputs, got {len(inputs)}")
        if any(value.device != device for value in inputs):
            raise AssertionError("fixture tensors must all be allocated on --device")
        print(
            "P7_1_FIXTURE_PREFLIGHT pass=1 inputs=31 "
            "device=%s opp_imported=0 aclnn_dispatched=0" % device,
            flush=True,
        )
        return
    if not args.fixture_module:
        inputs = _fail_closed_fixture(device)
        outputs = _run(inputs)
        values = _report("fail_closed", inputs, outputs)
        if values["control_rejected"] <= 0 or any(
            values[key] != 0 for key in ("lease", "post", "retire", "finish", "release")
        ):
            raise AssertionError("invalid control must reject before lease/PostOnly/CQ")
        print(
            "P7_1_TEST mode=fail_closed pass=1 aliases=18 "
            "real_hccp=not_exercised reason=invalid_control_expected",
            flush=True,
        )
        del outputs
        del inputs
        _teardown_checkpoint("fail_closed")
        return

    module = importlib.import_module(args.fixture_module)
    factory = getattr(module, "build_p7_1_fixture", None)
    if factory is None:
        raise RuntimeError(
            f"fixture module {args.fixture_module!r} lacks build_p7_1_fixture(device)"
        )
    fixture: dict[str, Any] = factory(device)
    cleanup = fixture.get("cleanup")
    try:
        inputs = tuple(fixture["inputs"])
        # A fixture is a transport transaction, not a pure function: running
        # the op twice would issue two device-side PostOnly waves against the
        # same graph-visible queue/private state.  Dispatch exactly once.
        outputs = _run(inputs)
        values = _report("real_direct_hccp", inputs, outputs)
        minimum = fixture.get("minimum", {})
        for key in ("lease", "post", "finish", "release"):
            wanted = int(minimum.get(key, 1))
            if values[key] < wanted:
                raise AssertionError(
                    f"real P7.1 fixture requires {key}>={wanted}, got {values[key]}"
                )
        validator = fixture.get("validate")
        if validator is not None:
            validator(outputs)
        print(
            "P7_1_TEST mode=real_direct_hccp pass=1 "
            "real_hccp=acquire_post_retire_finish_release",
            flush=True,
        )
        del outputs
        del inputs
        _teardown_checkpoint("real_direct_hccp")
    finally:
        if cleanup is not None:
            cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"P7_1_TEST_FAIL type={type(exc).__name__} text={exc}", flush=True)
        raise
