#!/usr/bin/env python3
"""Focused P8 forward-owned graph-order NPU test.

This is deliberately the *only* P8 runtime seam test.  It first prepares and
admits the fixed P8 arena, then launches the existing score-receipt
CandidateSeed producer (Stage A), and finally drives the forward-owned P8
pair through its four graph-visible phases:

``ADMISSION -> STAGE_A -> POST -> DRAIN -> QSFA_TERMINAL ->
FORWARD_TERMINAL -> REUSE_FENCE``.

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

The fixture boundary is intentionally two-step.  ``prepare`` creates only
the graph-stable arena/control tensors needed by admission; ``attach`` takes
the immutable Stage-A receipt outputs afterwards.  The runtime proves that
the attached Stage-B input aliases the *already admitted* shell.  It therefore
cannot accidentally reset ``forward_control``, queue/private records, or the
arena tag between admission and POST.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
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
WORKER_STATS_INPUT = 27
CONTROL_REJECTED_STAT_COLUMN = 10
# Exact alias order from the 28-input/19-output P8 OPP.  Receipt identity and
# status are aliases too, but ForwardTerminal intentionally tombstones them;
# exclude those terminal-domain mutations from the fail-closed transport
# immutability comparison below.
MUTABLE_INPUTS = (
    4, 5, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 27, 2, 3,
)
FAIL_CLOSED_TRANSPORT_INPUTS = tuple(
    index
    for index in MUTABLE_INPUTS
    if index not in {2, 3, 4, WORKER_STATS_INPUT}
)
# A real local HCCP fixture can honestly force and byte-validate only the
# successful hardware transaction.  The error/quarantine branches are proven
# by the P8 source/state-machine contract suite; labelling a happy hardware
# run as though it injected those failures would be a false claim.
REQUIRED_REAL_CASES = {"happy_post_cq_qsfa_release"}
ADMISSION_RESULT_WORDS = 4
ADMISSION_ADMITTED = 0
ADMISSION_CONTROL_INVALID = 2


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


def _run_stage_a(
    device: torch.device, fixture: Any
) -> tuple[torch.Tensor, ...]:
    # The P8 forward-control fixture below owns (epoch=37, replay=0,
    # layer=0).  Stage A must publish that exact triple plus a unique
    # per-receipt generation, rather than the old epoch-only marker.
    receipt_identity = _stage_a_fixture._receipt_identity(
        fixture, graph_replay_generation=0, layer_control_index=0
    )
    expected = _stage_a_fixture._reference(fixture, receipt_identity)
    inputs = [
        _to_device(tensor, device)
        for tensor in (
            fixture.score_workspace,
            fixture.tile_receipts,
            receipt_identity,
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
    candidate_seeds, candidate_counts, candidate_receipt_identity, candidate_status, _owners = outputs
    for label, actual, wanted in zip(
        ("candidate_seeds", "candidate_counts", "candidate_receipt_identity", "candidate_status"),
        (candidate_seeds, candidate_counts, candidate_receipt_identity, candidate_status),
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
        "candidate_seed_abi=int32[N,128,4] receipt_identity_release=1",
        flush=True,
    )
    return (
        candidate_seeds,
        candidate_counts,
        candidate_receipt_identity,
        candidate_status,
    )


def _pack_control(epoch: int, replay: int, layer: int, phase: int, flags: int) -> list[int]:
    if epoch <= 0 or replay < 0 or not 0 <= layer <= 0xFFFFFFFF:
        raise ValueError("invalid P8 forward identity")
    packed_layer_phase = (int(phase) << 32) | int(layer)
    return [int(epoch), int(replay), packed_layer_phase, int(flags)]


def _phase(control: torch.Tensor) -> tuple[int, int]:
    values = control.detach().cpu().to(torch.int64).tolist()
    return int((values[2] >> 32) & 0xFFFFFFFF), int(values[3])


def _prepare_fail_closed_fixture(
    fixture: Any, device: torch.device
) -> dict[str, Any]:
    """Allocate only the stable P8 arena before Stage-A publishes receipts.

    The invalid Direct-HCCP control deliberately exercises the documented
    fail-closed admission branch.  It still has the same topology as the real
    fixture: no candidate/receipt tensor is allocated or written until after
    the admission node has completed.
    """

    rows = int(fixture.rows)
    sources = int(fixture.source_capacity)
    pools = int(fixture.pool_size)
    objects = int(fixture.object_capacity)
    queues = 1
    private_rows = 64
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
    return {
        "fixture": fixture,
        "arena_shell": {
            "forward_control": forward_control,
            # Sticky shared queue/private-arena health is graph-visible state.
            "p8_arena_health": _to_device(torch.zeros((1,), dtype=torch.int32), device),
            "fused_get_control": _to_device(torch.zeros((16,), dtype=torch.int64), device),
            "p8_wave_records": _to_device(torch.zeros((queues, 32), dtype=torch.int64), device),
            "p8_private_records": _to_device(
                torch.zeros((private_rows, 10), dtype=torch.int64), device
            ),
            "expected_identity": expected_identity,
        },
        "static_inputs": {
            "score_source_lens": _to_device(fixture.score_source_lens, device),
            "active_rows": _to_device(torch.tensor([1], dtype=torch.int32), device),
            "req_pool_entries": _to_device(fixture.req_pool_entries, device),
            "kv_object_map": _to_device(fixture.kv_object_map, device),
            "kv_object_lens": _to_device(fixture.kv_object_lens, device),
            "kv_object_slots_snapshot": _to_device(
                fixture.kv_object_slots_snapshot, device
            ),
            "private_source_to_slot": _to_device(
                torch.full((rows, sources), -1, dtype=torch.int32), device
            ),
            "private_object_to_slot": _to_device(
                torch.full((pools, objects), -1, dtype=torch.int32), device
            ),
            "private_ready": _to_device(torch.zeros((private_rows,), dtype=torch.int32), device),
            "private_slot_cursor": _to_device(torch.zeros((1,), dtype=torch.int32), device),
            "private_output": _to_device(
                torch.zeros((private_rows, 656), dtype=torch.uint8), device
            ),
            "request_source_offsets": _to_device(
                torch.zeros((queues, 64), dtype=torch.int64), device
            ),
            "request_object_offsets": _to_device(
                torch.zeros((queues, 64), dtype=torch.int64), device
            ),
            "request_output_indices": _to_device(
                torch.full((queues, 64), -1, dtype=torch.int32), device
            ),
            "request_count": _to_device(torch.zeros((queues,), dtype=torch.int32), device),
            "rows_per_entry": _to_device(torch.ones((1,), dtype=torch.int32), device),
            "private_rows": private_rows,
            "queues": queues,
        },
    }


def _attach_fail_closed_stage_a_receipts(
    prepared: Mapping[str, Any], stage_a: tuple[torch.Tensor, ...]
) -> dict[str, Any]:
    """Attach Stage-A outputs without replacing any admitted arena tensor."""

    candidate_seeds, candidate_counts, candidate_receipt_identity, candidate_status = stage_a
    receipts = int(candidate_seeds.size(0))
    static = prepared["static_inputs"]
    shell = prepared["arena_shell"]
    workers = max(receipts, int(static["queues"]), int(static["private_rows"]))
    inputs: tuple[torch.Tensor, ...] = (
        candidate_seeds,
        candidate_counts,
        candidate_receipt_identity,
        candidate_status,
        shell["forward_control"],
        shell["p8_arena_health"],
        static["score_source_lens"],
        static["active_rows"],
        static["req_pool_entries"],
        static["kv_object_map"],
        static["kv_object_lens"],
        static["kv_object_slots_snapshot"],
        static["private_source_to_slot"],
        static["private_object_to_slot"],
        static["private_ready"],
        static["private_slot_cursor"],
        static["private_output"],
        shell["fused_get_control"],
        shell["p8_wave_records"],
        shell["p8_private_records"],
        static["request_source_offsets"],
        static["request_object_offsets"],
        static["request_output_indices"],
        static["request_count"],
        _to_device(torch.zeros((receipts,), dtype=torch.int32), candidate_seeds.device),
        _to_device(torch.zeros((receipts, 4), dtype=torch.int64), candidate_seeds.device),
        static["rows_per_entry"],
        _to_device(torch.zeros((workers, 16), dtype=torch.int64), candidate_seeds.device),
    )
    return {
        "executor_inputs": inputs,
        "expected_identity": shell["expected_identity"],
        "payload_validated": False,
        "validated_cases": (),
    }


def _identity_from_forward_control(
    forward_control: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _validate_inputs(inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 28:
        raise ValueError(f"P8 fixture must provide exactly 28 executor inputs, got {len(inputs)}")
    forward_control, expected_identity = _identity_from_forward_control(inputs[4])
    if inputs[5].dtype != torch.int32 or inputs[5].numel() != 1:
        raise ValueError("P8 p8_arena_health must be int32[1]")
    return forward_control, expected_identity


ARENA_SHELL_KEYS = (
    "forward_control",
    "p8_arena_health",
    "fused_get_control",
    "p8_wave_records",
    "p8_private_records",
    "expected_identity",
)
ARENA_SHELL_INPUT_INDEX = {
    "forward_control": 4,
    "p8_arena_health": 5,
    "fused_get_control": 17,
    "p8_wave_records": 18,
    "p8_private_records": 19,
}


def _require_arena_shell(prepared: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Validate the fixed P8 state allocated before Stage-A publication."""

    shell = prepared.get("arena_shell")
    if not isinstance(shell, Mapping):
        raise RuntimeError("P8 prepared fixture must provide arena_shell")
    missing = [key for key in ARENA_SHELL_KEYS if not isinstance(shell.get(key), torch.Tensor)]
    if missing:
        raise RuntimeError("P8 arena_shell missing tensors=" + ",".join(missing))
    control, derived_identity = _identity_from_forward_control(shell["forward_control"])
    health = shell["p8_arena_health"]
    identity = shell["expected_identity"]
    if health.dtype != torch.int32 or health.numel() != 1:
        raise RuntimeError("P8 arena_shell p8_arena_health must be int32[1]")
    if identity.dtype != torch.int64 or identity.numel() != 3:
        raise RuntimeError("P8 arena_shell expected_identity must be int64[3]")
    if identity.device != control.device or health.device != control.device:
        raise RuntimeError("P8 arena_shell tensors must be on one device")
    if not torch.equal(identity.detach().cpu(), derived_identity.detach().cpu()):
        raise RuntimeError("P8 arena_shell expected_identity does not match forward_control")
    return shell  # type: ignore[return-value]


def _assert_attached_to_admitted_shell(
    inputs: tuple[torch.Tensor, ...], shell: Mapping[str, torch.Tensor]
) -> None:
    """Receipt attachment may add candidate tensors, never reset P8 arena GM.

    Admission changed ``forward_control`` and ``p8_arena_health`` in place.
    Pointer identity is the compact, exact runtime proof that the subsequent
    Stage-B node sees those same objects rather than a freshly initialized
    copy with identical-looking contents.
    """

    for key, input_index in ARENA_SHELL_INPUT_INDEX.items():
        if inputs[input_index].data_ptr() != shell[key].data_ptr():
            raise RuntimeError(
                "P8 receipt attachment replaced admitted arena tensor "
                f"key={key} input={input_index}"
            )
    print(
        "P8_ARENA_ATTACHMENT admitted_shell_alias=1 "
        "stage_a_after_admission=1 arena_reinitialized=0",
        flush=True,
    )


def _require_under_root(label: str, path: str | os.PathLike[str], root: str) -> str:
    """Return a resolved imported path only when it belongs to this P8 run.

    The runner creates a unique bridge directory on every invocation.  This
    small boundary check is intentionally simpler than a persistent artifact
    fingerprint database: if any Python package or its native ``_C`` extension
    resolves outside that directory, this process stops before it can submit a
    graph against a stale registration.
    """

    resolved = Path(path).resolve()
    expected_root = Path(root).resolve()
    try:
        resolved.relative_to(expected_root)
    except ValueError as exc:
        raise RuntimeError(
            f"P8 stale {label} import: actual={resolved} expected_root={expected_root}"
        ) from exc
    return str(resolved)


def _assert_fresh_bridge_provenance() -> None:
    roots = {
        "reference": os.environ.get("P8_REFERENCE_BRIDGE_ROOT", ""),
        "stage_a": os.environ.get("P8_STAGE_A_BRIDGE_ROOT", ""),
        "stage_b": os.environ.get("P8_STAGE_B_BRIDGE_ROOT", ""),
    }
    if any(not root for root in roots.values()):
        raise RuntimeError("P8 bridge provenance roots are missing from the runner")

    modules = {
        "reference": importlib.import_module("nanovllm_official_c8_reference"),
        "stage_a": importlib.import_module("nanovllm_score_receipt_executor"),
        "stage_b": importlib.import_module("nanovllm_p8_forward_owned_executor"),
    }
    rendered: list[str] = []
    for label, module in modules.items():
        package_path = _require_under_root(
            f"{label}_package", str(module.__file__), roots[label]
        )
        extension = getattr(module, "_C", None)
        extension_path = getattr(extension, "__file__", None)
        if not extension_path:
            raise RuntimeError(f"P8 {label} bridge has no loaded _C extension")
        native_path = _require_under_root(
            f"{label}_extension", str(extension_path), roots[label]
        )
        rendered.append(f"{label}_package={package_path} {label}_extension={native_path}")
    print(
        "P8_ARTIFACT_PROVENANCE "
        f"vllm={os.environ.get('P8_VLLM_EXPECTED_SHA', 'missing')} "
        f"kvaf={os.environ.get('P8_KVAF_EXPECTED_SHA', 'missing')} "
        + " ".join(rendered)
        + " exact=1",
        flush=True,
    )


def _run_arena_admission_gate(
    shell: Mapping[str, torch.Tensor],
    *,
    expect_admitted: bool,
) -> tuple[int, int, int, int]:
    """Run P8's one-time shared-arena gate before any POST attempt.

    This node validates fixed queue/private records and the sticky health word;
    it is not a permanently reserved Get core.  The normal POST workers remain
    dynamically scheduled AIV work after this bounded control node completes.
    """

    result = torch.full(
        (ADMISSION_RESULT_WORDS,), -99, dtype=torch.int64,
        device=shell["forward_control"].device,
    )
    outputs = torch.ops.a5_p8_forward_owned_executor.arena_admission_gate(
        shell["forward_control"],
        shell["p8_arena_health"],
        shell["fused_get_control"],
        shell["p8_wave_records"],
        shell["p8_private_records"],
        shell["expected_identity"],
        result,
    )
    torch.npu.synchronize()
    expected_aliases = ("forward_control", "p8_arena_health")
    if len(outputs) != 3:
        raise AssertionError(f"P8 admission returned {len(outputs)} aliases, expected 3")
    for output, key in zip(outputs[:2], expected_aliases, strict=True):
        if output.data_ptr() != shell[key].data_ptr():
            raise AssertionError(f"P8 admission alias mismatch key={key}")
    if outputs[2].data_ptr() != result.data_ptr():
        raise AssertionError("P8 admission alias mismatch result")
    values = tuple(int(value) for value in result.detach().cpu().tolist())
    expected = (1, ADMISSION_ADMITTED, -1, 0) if expect_admitted else (
        0,
        ADMISSION_CONTROL_INVALID,
        -1,
        -1,
    )
    if values != expected:
        raise AssertionError(
            "P8 arena admission result mismatch "
            f"actual={values} expected={expected} real_mode={int(expect_admitted)}"
        )
    print(
        "P8_ARENA_ADMISSION "
        f"admitted={values[0]} reason={values[1]} index={values[2]} "
        f"observed={values[3]} dynamic_get_workers=1 result=pass",
        flush=True,
    )
    return values


def _run_executor(inputs: tuple[torch.Tensor, ...], label: str) -> tuple[torch.Tensor, ...]:
    outputs = torch.ops.a5_p8_forward_owned_executor.forward_owned_executor(*inputs)
    torch.npu.synchronize()
    if len(outputs) != len(MUTABLE_INPUTS):
        raise AssertionError(
            f"P8 {label} returned {len(outputs)} aliases, expected {len(MUTABLE_INPUTS)}"
        )
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
    arena_health: torch.Tensor,
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
        control, arena_health, identity, program, result
    )
    torch.npu.synchronize()
    if len(outputs) != 3:
        raise AssertionError(
            f"P8 advance {label} returned {len(outputs)} aliases, expected 3"
        )
    if (
        outputs[0].data_ptr() != control.data_ptr()
        or outputs[1].data_ptr() != arena_health.data_ptr()
        or outputs[2].data_ptr() != result.data_ptr()
    ):
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


def _prepare_fixture(
    module_name: str | None,
    device: torch.device,
    fallback_fixture: Any,
) -> tuple[Mapping[str, Any], Callable[..., Mapping[str, Any]], bool]:
    """Prepare the fixed arena before Stage-A writes CandidateSeed GM."""

    if not module_name:
        return (
            _prepare_fail_closed_fixture(fallback_fixture, device),
            _attach_fail_closed_stage_a_receipts,
            False,
        )
    module = importlib.import_module(module_name)
    factory: Callable[..., Mapping[str, Any]] | None = getattr(
        module, "prepare_p8_real_fixture", None
    )
    attach: Callable[..., Mapping[str, Any]] | None = getattr(
        module, "attach_p8_stage_a_receipts", None
    )
    if factory is None or attach is None:
        raise RuntimeError(
            f"{module_name} must export prepare_p8_real_fixture(device) and "
            "attach_p8_stage_a_receipts(prepared, stage_a_outputs)"
        )
    prepared = factory(device)
    if not isinstance(prepared, Mapping):
        raise RuntimeError("real P8 prepare fixture must return a mapping")
    _require_arena_shell(prepared)
    return prepared, attach, True


def _attach_stage_a_receipts(
    prepared: Mapping[str, Any],
    attach: Callable[..., Mapping[str, Any]],
    stage_a: tuple[torch.Tensor, ...],
    shell: Mapping[str, torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, Mapping[str, Any]]:
    """Attach receipts after admission and prove it preserved arena aliases."""

    fixture = attach(prepared, stage_a)
    if not isinstance(fixture, Mapping):
        raise RuntimeError("P8 receipt attachment must return a mapping")
    inputs_value = fixture.get("executor_inputs")
    if not isinstance(inputs_value, (tuple, list)):
        raise RuntimeError("real P8 fixture must provide executor_inputs tuple[28]")
    inputs = tuple(inputs_value)
    _assert_attached_to_admitted_shell(inputs, shell)
    control, derived_identity = _validate_inputs(inputs)
    identity = fixture.get("expected_identity", shell["expected_identity"])
    if not isinstance(identity, torch.Tensor):
        raise RuntimeError("real P8 fixture expected_identity must be an int64[3] tensor")
    if identity.device != control.device:
        identity = identity.to(control.device).contiguous()
    if identity.data_ptr() != shell["expected_identity"].data_ptr():
        raise RuntimeError("P8 receipt attachment replaced admitted expected_identity")
    if not torch.equal(identity.detach().cpu(), derived_identity.detach().cpu()):
        raise AssertionError("P8 attached expected_identity does not match forward_control")
    return inputs, identity, fixture


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


def _validate_terminal_reuse_fence(
    inputs: tuple[torch.Tensor, ...],
    real_fixture: Mapping[str, Any],
    *,
    real_mode: bool,
) -> None:
    """Verify the only legal point at which a next forward may reinitialize.

    This is intentionally a *fence check*, not a synthetic second forward.
    Constructing another forward here would add a second lifecycle surface and
    obscure the actual invariant: ForwardTerminal must run after QSFA's last
    reader and must release/quarantine the arena before any F1 initialization.
    """

    phase, flags = _phase(inputs[4])
    required = FLAG_EXACT_INPUT_READY | FLAG_QSFA_TERMINAL | FLAG_MODEL_TERMINAL
    if phase != PHASE_FORWARD_TERMINAL or (flags & required) != required:
        raise AssertionError(
            "P8 reuse fence reached before a complete forward terminal "
            f"phase={phase} flags=0x{flags:x} required=0x{required:x}"
        )
    if real_mode:
        validator = real_fixture.get("validate_reuse_fence")
        if not callable(validator):
            raise RuntimeError(
                "real P8 fixture must export validate_reuse_fence(executor_inputs)"
            )
        validator(inputs)
        print(
            "P8_REUSE_FENCE terminal_after_qsfa=1 arena_reuse_permitted=1 "
            "next_forward_reset_before_fence=0",
            flush=True,
        )
    else:
        # Admission rejected this fixture before any transport lease.  The
        # no-transport terminal path is allowed to leave the untouched idle
        # tag at zero; it still cannot pretend it completed a reusable Get.
        tag = int(inputs[5].detach().cpu().reshape(-1)[0]) & 0xFFFFFFFF
        if tag != 0:
            raise AssertionError(
                "P8 fail-closed path changed the unadmitted arena tag "
                f"tag=0x{tag:08x}"
            )
        print(
            "P8_REUSE_FENCE terminal_after_qsfa=1 arena_reuse_permitted=0 "
            "reason=admission_rejected",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--real-fixture-module",
        default="",
        help=(
            "Optional module exporting prepare_p8_real_fixture(device) then "
            "attach_p8_stage_a_receipts(prepared, stage_a_outputs).  It owns "
            "Direct-HCCP registration and may supply unchanged official-C8 "
            "baseline tensors."
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
    _assert_fresh_bridge_provenance()

    # The forward's fixed queue/private arena exists before receipt
    # publication.  This is deliberately separate from Stage-A output
    # allocation: a later attachment must alias it, never rebuild/reset it.
    stage_a_fixture = _stage_a_fixture._fixture()
    prepared, attach, real_mode = _prepare_fixture(
        args.real_fixture_module or None, device, stage_a_fixture
    )
    shell = _require_arena_shell(prepared)
    admission = _run_arena_admission_gate(shell, expect_admitted=real_mode)
    if admission[0] != 0:
        marker = prepared.get("mark_admitted")
        if real_mode and not callable(marker):
            raise RuntimeError(
                "real P8 prepared fixture must expose mark_admitted() after "
                "a successful graph admission"
            )
        if callable(marker):
            marker()

    # P8_LAYER_REUSE_CONTRACT: this admission is once for the whole forward,
    # not once per layer.  A later layer may reuse a queue/private record only
    # after the preceding layer's bounded CQ drain *and* QSFA terminal have
    # returned that record to FREE.  This focused one-layer harness makes
    # layer 0 the terminal layer, so it proceeds directly to FORWARD_TERMINAL
    # after the QSFA release below.

    # Stage A may now publish CandidateSeed GM.  The exact Stage-A oracle is
    # unchanged; only its position in the forward graph changed.
    stage_a = _run_stage_a(device, stage_a_fixture)
    inputs, identity, real_fixture = _attach_stage_a_receipts(
        prepared, attach, stage_a, shell
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
    _advance(
        control, inputs[5], identity, PHASE_POST, PHASE_DRAIN, 0, 0,
        "post_to_drain",
    )
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
        control, inputs[5], identity, PHASE_DRAIN, PHASE_DRAIN, 0,
        FLAG_EXACT_INPUT_READY, "exact_input_ready",
    )
    _advance(
        control, inputs[5], identity, PHASE_DRAIN, PHASE_QSFA_TERMINAL,
        FLAG_EXACT_INPUT_READY, FLAG_QSFA_TERMINAL, "qsfa_terminal",
    )
    _run_executor(inputs, "qsfa_terminal")
    _advance(
        control, inputs[5], identity, PHASE_QSFA_TERMINAL, PHASE_FORWARD_TERMINAL,
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
        # ForwardTerminal closes the receipt ABA domain even when the invalid
        # control prevented every PostOnly.  Generation is the sole release
        # marker, so no candidate or cursor identity may remain live.
        receipt_identity_cpu = inputs[2].detach().cpu().reshape(-1, 4)
        cursor_identity_cpu = inputs[25].detach().cpu().reshape(-1, 4)
        status_cpu = inputs[3].detach().cpu()
        if int(torch.count_nonzero(receipt_identity_cpu[:, 3])) != 0:
            raise AssertionError("P8 forward terminal left a live candidate receipt generation")
        if int(torch.count_nonzero(cursor_identity_cpu[:, 3])) != 0:
            raise AssertionError("P8 forward terminal left a live cursor receipt generation")
        if not bool(torch.all(status_cpu == -6).item()):
            raise AssertionError(
                "P8 forward terminal did not tombstone all current candidate statuses"
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
            "exact_recall_fallback=required receipt_tombstone=1 cursor_tombstone=1",
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
    # No test node is permitted to allocate/reset F1 state above this line.
    # The real fixture proves the terminal tag is clean here; only then could
    # a subsequent graph forward initialize the same fixed P8 arena.
    _validate_terminal_reuse_fence(
        inputs, real_fixture, real_mode=real_mode
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
