#!/usr/bin/env python3
"""Focused P8 Stage-A -> Stage-B graph-order NPU test.

This is deliberately the *only* P8 runtime seam test.  It launches the
existing score-receipt CandidateSeed producer (Stage A), then drives the
forward-owned P8 pair through its four graph-visible phases:

``POST -> DRAIN -> QSFA_TERMINAL -> FORWARD_TERMINAL``.

The default fixture is intentionally fail-closed: it proves that an invalid
Direct-HCCP control cannot mutate P8 transport state while the real Stage-A
CandidateSeed math and the device-side phase-control node execute.  A real
runtime can supply ``--real-fixture-module`` to bind registered KVAF Direct-
HCCP resources.  Only that module may report a payload-validated H2D result;
this test never infers it from an ACLNN submission returning successfully.

Stage A is reused as the existing score-receipt consumer.  C8 is neither
modified nor imported by this test.  A real fixture optionally supplies C8
baseline tensors captured from its unchanged official C8 path so the runner
can prove that P8 did not alter final sparse Top-K or score source lengths.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Any, Callable, Mapping, Sequence

import torch
import torch_npu  # type: ignore  # noqa: F401

# This is the completed Stage-A score-receipt producer test fixture.  Reusing
# it prevents a second candidate predicate from quietly diverging from the
# published CandidateSeed ABI.  Its public-facing P8 markers below deliberately
# name the stage, not its historical prototype number.
import test_score_receipt_candidate_executor as _stage_a_fixture

import nanovllm_p8_forward_owned_executor  # noqa: F401


PHASE_POST = 1
PHASE_DRAIN = 2
PHASE_QSFA_TERMINAL = 3
PHASE_FORWARD_TERMINAL = 4
FLAG_EXACT_INPUT_READY = 1 << 2
FLAG_QSFA_TERMINAL = 1 << 3
FLAG_MODEL_TERMINAL = 1 << 4
# The executor returns aliases for all mutable P8 tensors.  The invalid-control
# fixture intentionally takes the Init-time fail-closed branch, which records
# only `kP8StatControlRejected` in worker_stats.  It is a diagnostic mutation,
# not a transport-state mutation, so test it explicitly rather than treating
# every alias as byte-for-byte immutable.
WORKER_STATS_INPUT = 26
CONTROL_REJECTED_STAT_COLUMN = 10
MUTABLE_INPUTS = (4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26)
FAIL_CLOSED_TRANSPORT_INPUTS = tuple(
    index for index in MUTABLE_INPUTS if index not in {4, WORKER_STATS_INPUT}
)
# A real local HCCP fixture can honestly force and byte-validate only the
# successful hardware transaction.  The error/quarantine branches are proven
# by the P8 source/state-machine contract suite; labelling a happy hardware
# run as though it injected those failures would be a false claim.
REQUIRED_REAL_CASES = {"happy_post_cq_qsfa_release"}


def _digest(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().reshape(-1).to(torch.int64)
    value = 0xCBF29CE484222325
    for item in values.tolist():
        value ^= int(item) & ((1 << 64) - 1)
        value = (value * 0x100000001B3) & ((1 << 64) - 1)
    return f"{value:016x}"


def _assert_equal(label: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_cpu = actual.detach().cpu().contiguous()
    expected_cpu = expected.detach().cpu().contiguous()
    if torch.equal(actual_cpu, expected_cpu):
        return
    mismatches = (actual_cpu != expected_cpu).reshape(-1).nonzero(as_tuple=False)
    first = int(mismatches[0].item()) if mismatches.numel() else -1
    raise AssertionError(
        f"{label} mismatch count={int(mismatches.numel())} first_flat={first} "
        f"actual_sha16={_digest(actual_cpu)} expected_sha16={_digest(expected_cpu)}"
    )


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device).contiguous()


def _run_stage_a(device: torch.device) -> tuple[Any, tuple[torch.Tensor, ...]]:
    fixture = _stage_a_fixture._fixture()
    expected = _stage_a_fixture._reference(fixture)
    inputs = [
        _to_device(tensor, device)
        for tensor in (
            fixture.score_workspace,
            fixture.tile_receipts,
            torch.tensor([fixture.epoch], dtype=torch.int64),
            fixture.score_source_lens,
            fixture.resident_threshold,
            fixture.resident_threshold_valid,
            fixture.kv_object_slots_snapshot,
            fixture.req_pool_entries,
            fixture.kv_object_map,
            fixture.kv_object_lens,
        )
    ]
    outputs = torch.ops.a5_score_receipt_executor.score_receipt_candidate_executor(
        *inputs
    )
    torch.npu.synchronize()
    candidate_seeds, candidate_counts, ready_epoch, candidate_status, _owners = outputs
    for label, actual, wanted in zip(
        ("candidate_seeds", "candidate_counts", "candidate_ready_epoch", "candidate_status"),
        (candidate_seeds, candidate_counts, ready_epoch, candidate_status),
        expected,
        strict=True,
    ):
        _assert_equal(f"P8_STAGE_A_{label}", actual, wanted)
    counts_cpu = candidate_counts.cpu()
    print(
        "P8_STAGE_A_EXACT_RECEIPT "
        f"candidate_total={int(counts_cpu.sum())} "
        f"nonempty_receipts={int((counts_cpu > 0).sum())} "
        f"seeds_sha16={_digest(candidate_seeds)} counts_sha16={_digest(candidate_counts)} "
        "candidate_seed_abi=int32[N,128,4] ready_epoch_barrier=1",
        flush=True,
    )
    return fixture, (candidate_seeds, candidate_counts, ready_epoch, candidate_status)


def _pack_control(epoch: int, replay: int, layer: int, phase: int, flags: int) -> list[int]:
    if epoch <= 0 or replay < 0 or not 0 <= layer <= 0xFFFFFFFF:
        raise ValueError("invalid P8 forward identity")
    packed_layer_phase = (int(phase) << 32) | int(layer)
    return [int(epoch), int(replay), packed_layer_phase, int(flags)]


def _phase(control: torch.Tensor) -> tuple[int, int]:
    values = control.detach().cpu().to(torch.int64).tolist()
    return int((values[2] >> 32) & 0xFFFFFFFF), int(values[3])


def _fail_closed_inputs(
    fixture: Any,
    stage_a: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    candidate_seeds, candidate_counts, ready_epoch, candidate_status = stage_a
    receipts = int(candidate_seeds.size(0))
    rows = int(fixture.rows)
    sources = int(fixture.source_capacity)
    pools = int(fixture.pool_size)
    objects = int(fixture.object_capacity)
    queues = 1
    private_rows = 64
    workers = max(receipts, queues, private_rows)
    forward_control = _to_device(
        torch.tensor(_pack_control(fixture.epoch, 0, 0, PHASE_POST, 0), dtype=torch.int64),
        device,
    )
    expected_identity = _to_device(
        torch.tensor([fixture.epoch, 0, 0], dtype=torch.int64), device
    )
    # `fused_get_control` is intentionally all-zero.  The P8 kernel must
    # reject it before any irreversible PostOnly action; phase transitions
    # still execute through their independent P8 graph node.
    inputs: tuple[torch.Tensor, ...] = (
        candidate_seeds,
        candidate_counts,
        ready_epoch,
        candidate_status,
        forward_control,
        _to_device(fixture.score_source_lens, device),
        _to_device(torch.tensor([1], dtype=torch.int32), device),
        _to_device(fixture.req_pool_entries, device),
        _to_device(fixture.kv_object_map, device),
        _to_device(fixture.kv_object_lens, device),
        _to_device(fixture.kv_object_slots_snapshot, device),
        _to_device(torch.full((rows, sources), -1, dtype=torch.int32), device),
        _to_device(torch.full((pools, objects), -1, dtype=torch.int32), device),
        _to_device(torch.zeros((private_rows,), dtype=torch.int32), device),
        _to_device(torch.zeros((1,), dtype=torch.int32), device),
        _to_device(torch.zeros((private_rows, 656), dtype=torch.uint8), device),
        _to_device(torch.zeros((16,), dtype=torch.int64), device),
        _to_device(torch.zeros((queues, 32), dtype=torch.int64), device),
        _to_device(torch.zeros((private_rows, 10), dtype=torch.int64), device),
        _to_device(torch.zeros((queues, 64), dtype=torch.int64), device),
        _to_device(torch.zeros((queues, 64), dtype=torch.int64), device),
        _to_device(torch.full((queues, 64), -1, dtype=torch.int32), device),
        _to_device(torch.zeros((queues,), dtype=torch.int32), device),
        _to_device(torch.zeros((receipts,), dtype=torch.int32), device),
        _to_device(torch.zeros((receipts,), dtype=torch.int64), device),
        _to_device(torch.ones((1,), dtype=torch.int32), device),
        _to_device(torch.zeros((workers, 16), dtype=torch.int64), device),
    )
    return inputs, expected_identity


def _validate_inputs(inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 27:
        raise ValueError(f"P8 fixture must provide exactly 27 executor inputs, got {len(inputs)}")
    forward_control = inputs[4]
    if forward_control.dtype != torch.int64 or forward_control.numel() != 4:
        raise ValueError("P8 forward_control must be int64[4]")
    # This is validation-only code, not a graph node.  Decode on CPU so the
    # NPU test never depends on a backend-specific implementation of scalar
    # bitwise-and before the actual P8 graph begins.
    words = forward_control.detach().cpu().reshape(-1).to(torch.int64).tolist()
    expected_identity = torch.tensor(
        [int(words[0]), int(words[1]), int(words[2]) & 0xFFFFFFFF],
        dtype=torch.int64,
        device=forward_control.device,
    )
    return forward_control, expected_identity


def _run_executor(inputs: tuple[torch.Tensor, ...], label: str) -> tuple[torch.Tensor, ...]:
    outputs = torch.ops.a5_p8_forward_owned_executor.forward_owned_executor(*inputs)
    torch.npu.synchronize()
    if len(outputs) != len(MUTABLE_INPUTS):
        raise AssertionError(f"P8 {label} returned {len(outputs)} aliases, expected 16")
    for output, input_index in zip(outputs, MUTABLE_INPUTS, strict=True):
        if output.data_ptr() != inputs[input_index].data_ptr():
            raise AssertionError(f"P8 {label} alias mismatch output for input {input_index}")
    phase, flags = _phase(inputs[4])
    print(
        f"P8_EXECUTOR_PHASE phase={label} control_phase={phase} flags=0x{flags:x} "
        "submit=1 host_gil_mutation=0",
        flush=True,
    )
    return outputs


def _advance(
    control: torch.Tensor,
    identity: torch.Tensor,
    expected_phase: int,
    next_phase: int,
    required_flags: int,
    set_flags: int,
    label: str,
) -> None:
    program = torch.tensor(
        [expected_phase, next_phase, required_flags, set_flags],
        dtype=torch.int64,
        device=control.device,
    )
    result = torch.zeros((4,), dtype=torch.int64, device=control.device)
    outputs = torch.ops.a5_p8_forward_owned_executor.advance_forward_control(
        control, identity, program, result
    )
    torch.npu.synchronize()
    if outputs[0].data_ptr() != control.data_ptr() or outputs[1].data_ptr() != result.data_ptr():
        raise AssertionError(f"P8 advance {label} alias contract failed")
    success, observed_phase, observed_flags, error = result.cpu().tolist()
    if success != 1 or observed_phase != next_phase or error != 0:
        raise AssertionError(
            f"P8 advance {label} failed success={success} observed_phase={observed_phase} "
            f"flags=0x{int(observed_flags):x} error={error}"
        )
    phase, flags = _phase(control)
    if phase != next_phase or (flags & set_flags) != set_flags:
        raise AssertionError(
            f"P8 advance {label} control mismatch phase={phase} flags=0x{flags:x}"
        )
    print(
        f"P8_CONTROL_ADVANCE label={label} phase={phase} flags=0x{flags:x} result=pass",
        flush=True,
    )


def _load_real_fixture(
    module_name: str | None,
    device: torch.device,
    stage_a: tuple[torch.Tensor, ...],
    fallback_fixture: Any,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, Mapping[str, Any], bool]:
    if not module_name:
        inputs, identity = _fail_closed_inputs(fallback_fixture, stage_a, device)
        return inputs, identity, {}, False
    module = importlib.import_module(module_name)
    factory: Callable[..., Mapping[str, Any]] | None = getattr(
        module, "build_p8_real_fixture", None
    )
    if factory is None:
        raise RuntimeError(
            f"{module_name} must export build_p8_real_fixture(device, stage_a_outputs)"
        )
    fixture = factory(device, stage_a)
    inputs_value = fixture.get("executor_inputs")
    if not isinstance(inputs_value, (tuple, list)):
        raise RuntimeError("real P8 fixture must provide executor_inputs tuple[27]")
    inputs = tuple(inputs_value)
    control, derived_identity = _validate_inputs(inputs)
    identity = fixture.get("expected_identity", derived_identity)
    if not isinstance(identity, torch.Tensor):
        raise RuntimeError("real P8 fixture expected_identity must be an int64[3] tensor")
    if identity.device != control.device:
        identity = identity.to(control.device).contiguous()
    return inputs, identity, fixture, True


def _check_c8_baseline(real_fixture: Mapping[str, Any]) -> str:
    before = real_fixture.get("c8_baseline_before")
    after = real_fixture.get("c8_baseline_after")
    if before is None and after is None:
        print(
            "P8_C8_BASELINE status=not_exercised "
            "reason=real_fixture_did_not_supply_unchanged_official_c8_outputs",
            flush=True,
        )
        return "not_exercised"
    if not (isinstance(before, (tuple, list)) and isinstance(after, (tuple, list)) and
            len(before) == 2 and len(after) == 2):
        raise RuntimeError(
            "real P8 fixture c8_baseline_before/after must each be "
            "(sparse_indices, score_source_lens)"
        )
    _assert_equal("P8_C8_sparse_indices", after[0], before[0])
    _assert_equal("P8_C8_score_source_lens", after[1], before[1])
    print(
        "P8_C8_BASELINE sparse_indices=pass score_source_lens=pass "
        f"sparse_sha16={_digest(after[0])} lens_sha16={_digest(after[1])}",
        flush=True,
    )
    return "pass"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--real-fixture-module",
        default="",
        help=(
            "Optional module exporting build_p8_real_fixture(device, stage_a_outputs). "
            "Its fixture owns Direct-HCCP registration and may supply unchanged "
            "official-C8 baseline tensors."
        ),
    )
    parser.add_argument(
        "--require-real-hccp",
        action="store_true",
        help="Fail unless a real fixture validates terminal Direct-HCCP payload semantics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    print(
        "P8_TARGET_NPU "
        f"device={device} logical={device.index} "
        f"physical={__import__('os').environ.get('P8_PHYSICAL_NPU_ID', 'unspecified')} "
        f"visible_mapping={__import__('os').environ.get('ASCEND_RT_VISIBLE_DEVICES', 'unset')!r}",
        flush=True,
    )

    stage_a_fixture, stage_a = _run_stage_a(device)
    inputs, identity, real_fixture, real_mode = _load_real_fixture(
        args.real_fixture_module or None, device, stage_a, stage_a_fixture
    )
    control, derived_identity = _validate_inputs(inputs)
    if identity.dtype != torch.int64 or identity.numel() != 3:
        raise ValueError("P8 expected_identity must be int64[3]")
    identity = identity.reshape(3).contiguous()
    if not torch.equal(identity.cpu(), derived_identity.cpu()):
        raise AssertionError("P8 expected_identity does not match forward_control")

    mutable_before = {
        index: inputs[index].detach().clone()
        for index in FAIL_CLOSED_TRANSPORT_INPUTS
    }
    worker_stats_before = inputs[WORKER_STATS_INPUT].detach().clone()
    _run_executor(inputs, "post")
    _advance(control, identity, PHASE_POST, PHASE_DRAIN, 0, 0, "post_to_drain")
    # Direct-HCCP CQ is intentionally nonblocking inside the P8 DRAIN node.
    # A validation harness therefore submits a bounded number of *separate
    # graph-visible DRAIN nodes*, rather than turning one node into a device
    # spin loop or adding a host-side Get wait.  A real fixture requests this
    # only when it owns the cold HCCP queue; the fail-closed probe remains one
    # launch.
    drain_attempts = int(real_fixture.get("drain_attempts", 1)) if real_mode else 1
    if drain_attempts < 1 or drain_attempts > 32:
        raise ValueError(f"P8 drain_attempts must be in [1,32], got {drain_attempts}")
    for attempt in range(drain_attempts):
        _run_executor(inputs, f"drain_{attempt + 1}")
    # This advance is the graph-visible RuntimeResources proof that exact Top-K
    # plus Guaranteed Recall have completed; it is not a speculative executor
    # write and it is not a host/GIL per-candidate mutation.
    _advance(
        control, identity, PHASE_DRAIN, PHASE_DRAIN, 0,
        FLAG_EXACT_INPUT_READY, "exact_input_ready",
    )
    _advance(
        control, identity, PHASE_DRAIN, PHASE_QSFA_TERMINAL,
        FLAG_EXACT_INPUT_READY, FLAG_QSFA_TERMINAL, "qsfa_terminal",
    )
    _run_executor(inputs, "qsfa_terminal")
    _advance(
        control, identity, PHASE_QSFA_TERMINAL, PHASE_FORWARD_TERMINAL,
        FLAG_EXACT_INPUT_READY | FLAG_QSFA_TERMINAL,
        FLAG_MODEL_TERMINAL, "forward_terminal",
    )
    _run_executor(inputs, "forward_terminal")

    validated_cases: set[str] = set()
    if not real_mode:
        for index, before in mutable_before.items():
            _assert_equal(f"P8_fail_closed_input_{index}", inputs[index], before)
        worker_delta = (
            inputs[WORKER_STATS_INPUT].detach().cpu().to(torch.int64)
            - worker_stats_before.detach().cpu().to(torch.int64)
        )
        allowed = torch.zeros_like(worker_delta)
        allowed[:, CONTROL_REJECTED_STAT_COLUMN] = worker_delta[
            :, CONTROL_REJECTED_STAT_COLUMN
        ]
        if not torch.equal(worker_delta, allowed):
            raise AssertionError(
                "P8 fail-closed invalid Direct-HCCP control changed non-diagnostic "
                f"worker_stats delta_sha16={_digest(worker_delta)}"
            )
        rejected = int(worker_delta[:, CONTROL_REJECTED_STAT_COLUMN].sum().item())
        if rejected <= 0:
            raise AssertionError(
                "P8 fail-closed invalid Direct-HCCP control did not publish "
                "a control-rejected diagnostic"
            )
        print(
            "P8_POST_CQ outcome=not_exercised reason=fail_closed_invalid_direct_hccp_control "
            "post_only_blocking=0 cq_nonblocking=1",
            flush=True,
        )
        print(
            "P8_FAIL_CLOSED_DIAGNOSTIC "
            f"control_rejected={rejected} non_transport_mutation=0",
            flush=True,
        )
        print(
            "P8_LIFECYCLE_TERMINAL state=fail_closed_no_transport_mutation "
            "exact_recall_fallback=required",
            flush=True,
        )
    else:
        terminal_validator = real_fixture.get("validate_terminal")
        if not callable(terminal_validator):
            raise RuntimeError(
                "real P8 fixture must export validate_terminal(executor_inputs) "
                "to validate CQ/private/QSFA terminal state"
            )
        terminal_validator(inputs)
        payload_validated = bool(real_fixture.get("payload_validated", False))
        validated_cases = set(real_fixture.get("validated_cases", ()))
        print(
            "P8_POST_CQ "
            f"outcome=real_fixture_terminal payload_validated={int(payload_validated)} "
            "post_only_blocking=0 cq_nonblocking=1",
            flush=True,
        )
        print(
            "P8_LIFECYCLE_TERMINAL state=validated "
            "private_release=qsfa_terminal forward_sweep=1",
            flush=True,
        )
        if args.require_real_hccp and not payload_validated:
            raise AssertionError("P8 real fixture did not validate Direct-HCCP payload")
        if args.require_real_hccp:
            missing_cases = sorted(REQUIRED_REAL_CASES - validated_cases)
            if missing_cases:
                raise AssertionError(
                    "P8 real fixture did not validate lifecycle cases="
                    + ",".join(missing_cases)
                )
    print(
        "P8_LIFECYCLE_CASES "
        f"mode={'real' if real_mode else 'fail_closed'} "
        f"validated={','.join(sorted(validated_cases)) or 'none'}",
        flush=True,
    )

    c8_gate = _check_c8_baseline(real_fixture)
    # A successful Post/CQ lifecycle alone is not an acceptance result for
    # P8.  In the real Direct-HCCP mode the fixture must also show that the
    # unchanged official C8 output is bitwise identical before/after the P8
    # sidecar path.  Keeping this requirement here prevents a lifecycle-only
    # fixture from being mistaken for a mathematical-equivalence validation.
    if args.require_real_hccp and c8_gate != "pass":
        raise AssertionError(
            "P8 real Direct-HCCP validation requires unchanged official-C8 "
            "sparse_indices and score_source_lens baselines"
        )
    print(
        "P8_RUNTIME_PASS "
        f"stage_a_exact=1 phase_graph=1 real_hccp={int(real_mode)} "
        f"c8_math_gate={c8_gate}",
        flush=True,
    )
    # Normal release is legal only after real terminal validation and the C8
    # math gate.  On an exception the fresh runner process exits instead of
    # releasing a potentially in-flight Direct-HCCP registration.
    if real_mode:
        cleanup = real_fixture.get("cleanup")
        if callable(cleanup):
            cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"P8_RUNTIME_FAIL type={type(exc).__name__} text={exc}", flush=True)
        raise
