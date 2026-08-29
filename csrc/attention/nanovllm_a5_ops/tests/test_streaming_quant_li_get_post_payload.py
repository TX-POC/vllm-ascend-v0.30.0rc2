#!/usr/bin/env python3
"""Strict ``inline=post`` Direct-HCCP payload proof for P3.

This is deliberately a component-level transport proof, not a product route
and not a performance benchmark.  P4 already proves the official C8 score,
resident-threshold, and exact-TopK mathematics.  This test proves the next
link in the same forward:

``official score tile -> thresholded cold source -> KVAF Direct-HCCP ->
private packed-KV slot``.

It uses the real local native-enqueue KVAF runtime.  The test first cold-binds
one graph-visible streaming queue, then launches the source-derived C8 LI with
``inline=post``.  After its terminal CQ, it reconstructs every returned byte
through the real directory chain:

``row -> req_pool_entries -> kv_object_map -> table -> key -> DDR entry``.

No identity shortcut is used: request pools are swapped and every active
source maps to a reversed KV-object index.  Therefore a wrong row/pool/object
mapping cannot accidentally pass merely because the source allocation is
regular.  The test makes no bandwidth or end-to-end performance claim.

Run in a fresh process with exactly one visible physical card, for example
``ASCEND_RT_VISIBLE_DEVICES=7 ... --device npu:0 --physical-npu-id 7``.  This
is the Ascend runtime device-control variable; ``ASCEND_VISIBLE_DEVICES`` does
not remap ``torch_npu`` devices in the validation container.  In that process
logical ``npu:0`` is physical NPU 7; the explicit mapping check below rejects
any ambiguous launch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import sys
from typing import Any

import torch
import torch_npu  # noqa: F401  # Registers the NPU backend.

import kvaf
import streaming_quant_li_get_target_adapter as target


ROWS_PER_ENTRY = target.BLOCK_SIZE
PACKED_KV_ROW_BYTES = target.PACKED_KV_ROW_BYTES
ENTRY_BYTES = ROWS_PER_ENTRY * PACKED_KV_ROW_BYTES
STREAMING_CONTROL_WORDS = target.STREAMING_CONTROL_WORDS
STREAMING_CONTROL_MAGIC = target.STREAMING_CONTROL_MAGIC
STREAMING_CONTROL_VERSION = target.STREAMING_CONTROL_VERSION
QUEUE_TELEMETRY_COLUMNS = 14
TELEMETRY_DOORBELL_POSTED = 8
TELEMETRY_CQ_RETIRED_READY = 9
TELEMETRY_CQ_RETIRED_FAILED = 10
TELEMETRY_LEASE_RELEASED = 11
TELEMETRY_CONTROL_REJECTED = 12
TELEMETRY_SCORE_TILES_WITH_CQ_PENDING = 13
DEFAULT_QUEUE_COUNT = 1
DEFAULT_PRIVATE_ARENA_CAPACITY = 256
MAX_DENSE_REQUESTS = 256
MAX_DIAGNOSTIC_SAMPLES = 8
POST_EXERCISE_MODES = ("all_busy", "threshold_off", "full")


# The target ACLNN call is asynchronous.  Keep one compact host-side marker so
# a device-side failure reported by the following synchronize can still be
# tied to the last completed lifecycle step without relying on a huge log.
_LAST_STAGE = "startup"


def _stage(name: str, **fields: object) -> None:
    global _LAST_STAGE
    _LAST_STAGE = name
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"P3_POST_STAGE name={name}" + (f" {suffix}" if suffix else ""), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--physical-npu-id", type=int, required=True)
    parser.add_argument("--batch-get-kernel", required=True)
    parser.add_argument("--batch-put-kernel", required=True)
    parser.add_argument(
        "--table-capacity",
        type=int,
        default=target.SOURCE_TOKENS // ROWS_PER_ENTRY,
    )
    parser.add_argument("--queue-count", type=int, default=DEFAULT_QUEUE_COUNT)
    parser.add_argument(
        "--private-arena-capacity",
        type=int,
        default=DEFAULT_PRIVATE_ARENA_CAPACITY,
    )
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument(
        "--post-exercise",
        choices=POST_EXERCISE_MODES,
        default="full",
        help=(
            "all_busy: prove the P3 C8 base path while skipping every post action; "
            "threshold_off: exercise score observation plus SIMT lease acquire/release "
            "without a candidate or H2D post; full: exercise the real Post/CQ payload path"
        ),
    )
    return parser.parse_args()


def _u64(value: int) -> int:
    return int(value) & ((1 << 64) - 1)


def _sha16(value: torch.Tensor) -> str:
    payload = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def _first_difference(expected: torch.Tensor, actual: torch.Tensor) -> tuple[int, int, int]:
    unequal = torch.nonzero(expected != actual, as_tuple=False)
    if unequal.numel() == 0:
        return -1, -1, -1
    index = int(unequal[0].item())
    return index, int(expected[index].item()), int(actual[index].item())


def _shape_text(value: torch.Tensor) -> str:
    return "[" + ",".join(str(int(dim)) for dim in value.shape) + "]"


def _flat_coordinate(flat_index: int, shape: tuple[int, ...]) -> str:
    """Render a compact logical coordinate without relying on a device helper."""
    if flat_index < 0 or not shape:
        return "[]"
    coordinate: list[int] = []
    remaining = flat_index
    for dimension in reversed(shape):
        if dimension <= 0:
            return "[]"
        coordinate.append(remaining % dimension)
        remaining //= dimension
    return "[" + ",".join(str(value) for value in reversed(coordinate)) + "]"


def _strict_tensor_diagnostic(
    kind: str, expected: torch.Tensor, actual: torch.Tensor
) -> tuple[bool, str]:
    """Return a one-line, deterministic strict-equality diagnostic.

    The target and reference execute in separate ACLNN launches.  On a failure,
    a plain ``torch.equal`` says only that the final gate failed, which leaves
    shape/dtype drift, a complete corruption, and one unstable Top-K rank
    indistinguishable.  Move both small result tensors to CPU once and emit the
    exact compact evidence needed to classify the failure.
    """
    expected_cpu = expected.detach().cpu().contiguous()
    actual_cpu = actual.detach().cpu().contiguous()
    same_shape = tuple(expected_cpu.shape) == tuple(actual_cpu.shape)
    same_dtype = expected_cpu.dtype == actual_cpu.dtype
    mismatch_count = -1
    first_flat = -1
    first_coord = "[]"
    expected_value: object = "<none>"
    actual_value: object = "<none>"
    if same_shape and same_dtype:
        unequal = expected_cpu != actual_cpu
        mismatch_count = int(torch.count_nonzero(unequal).item())
        if mismatch_count:
            first_flat = int(torch.nonzero(unequal.reshape(-1), as_tuple=False)[0, 0].item())
            first_coord = _flat_coordinate(first_flat, tuple(expected_cpu.shape))
            expected_value = expected_cpu.reshape(-1)[first_flat].item()
            actual_value = actual_cpu.reshape(-1)[first_flat].item()
    equal = same_shape and same_dtype and mismatch_count == 0
    values = ""
    if kind == "score_source_lens":
        values = (
            f" expected_values={expected_cpu.reshape(-1).tolist()}"
            f" actual_values={actual_cpu.reshape(-1).tolist()}"
        )
    return equal, (
        f"kind={kind} equal={int(equal)} "
        f"expected_shape={_shape_text(expected_cpu)} actual_shape={_shape_text(actual_cpu)} "
        f"expected_dtype={expected_cpu.dtype} actual_dtype={actual_cpu.dtype} "
        f"expected_numel={expected_cpu.numel()} actual_numel={actual_cpu.numel()} "
        f"expected_sha16={_sha16(expected_cpu)} actual_sha16={_sha16(actual_cpu)} "
        f"mismatch_count={mismatch_count} first_flat={first_flat} "
        f"first_coord={first_coord} expected_value={expected_value} actual_value={actual_value}"
        f"{values}"
    )


def _require_test_args(args: argparse.Namespace) -> torch.device:
    if args.device != "npu:0":
        raise ValueError(
            "post payload proof requires logical npu:0; choose the physical card "
            "only with ASCEND_RT_VISIBLE_DEVICES before the process starts"
        )
    if args.physical_npu_id < 0:
        raise ValueError("--physical-npu-id must be non-negative")
    visible = os.getenv("ASCEND_RT_VISIBLE_DEVICES")
    expected_visible = str(args.physical_npu_id)
    if visible != expected_visible:
        raise RuntimeError(
            "strict physical/logical mapping requires exactly one visible physical "
            "card selected through Ascend runtime visibility: expected "
            f"ASCEND_RT_VISIBLE_DEVICES={expected_visible!r}, got {visible!r}; "
            f"ASCEND_VISIBLE_DEVICES={os.getenv('ASCEND_VISIBLE_DEVICES')!r} is not "
            "a substitute"
        )
    if args.table_capacity < target.SOURCE_TOKENS // ROWS_PER_ENTRY:
        raise ValueError(
            "--table-capacity must cover every raw KV object in the C8 fixture: "
            f"need at least {target.SOURCE_TOKENS // ROWS_PER_ENTRY}"
        )
    if args.queue_count <= 0 or args.queue_count > 48:
        raise ValueError("--queue-count must be in [1, 48]")
    if not 0 < args.private_arena_capacity <= MAX_DENSE_REQUESTS:
        raise ValueError(
            f"--private-arena-capacity must be in [1, {MAX_DENSE_REQUESTS}]"
        )
    if int(torch.npu.device_count()) != 1:
        raise RuntimeError(
            "strict physical/logical mapping requires one visible NPU: "
            f"device_count={torch.npu.device_count()} visible={visible!r}"
        )
    torch.npu.set_device(0)
    current = int(torch.npu.current_device())
    if current != 0:
        raise RuntimeError(f"failed to select logical npu:0: current={current}")
    name_getter = getattr(torch.npu, "get_device_name", torch_npu.npu.get_device_name)
    device_name = str(name_getter(0))
    if "950" not in device_name.lower():
        raise RuntimeError(
            "post payload proof requires an Ascend 950 target, "
            f"got {device_name!r}"
        )
    device = torch.device("npu:0")
    print(
        "P3_POST_TARGET_NPU "
        f"physical_npu={args.physical_npu_id} logical_npu={current} "
        f"visible_mapping={visible!r} device={device} device_name={device_name!r}",
        flush=True,
    )
    return device


def _create_runtime(args: argparse.Namespace) -> Any:
    runtime = kvaf.NpuRuntime(
        args.batch_get_kernel,
        0,
        args.queue_count,
        args.batch_put_kernel,
        1,
    )
    if callable(getattr(runtime, "link_to", None)):
        raise RuntimeError(
            "the imported KVAF module exposes a remote CPU-service ABI; select the "
            "matching local native-enqueue build"
        )
    if int(runtime.device_id) != 0:
        raise RuntimeError(
            "masked physical-card process requires KVAF logical device_id=0, "
            f"got {runtime.device_id}"
        )
    return runtime


def _require_streaming_runtime_api(runtime: Any) -> None:
    missing = [
        name
        for name in (
            "bind_streaming_batch_get_control",
            "configure_streaming_batch_get_queue_templates",
            "release_streaming_batch_get_control",
            "streaming_batch_get_control_diagnostic",
            "streaming_pool_snapshot",
        )
        if not callable(getattr(runtime, name, None))
    ]
    if missing:
        raise RuntimeError(
            "KVAF build lacks the streaming native-enqueue API: "
            + ", ".join(missing)
            + ". Rebuild the matching kip_demo_cs_native_enqueue revision; "
            "do not use an older _kvaf extension."
        )


def _build_local_ddr_payload(table_capacity: int) -> torch.Tensor:
    """Create deterministic full KVAF entries, one unique byte pattern per key."""
    entry = torch.arange(table_capacity, dtype=torch.int64).view(-1, 1)
    byte = torch.arange(ENTRY_BYTES, dtype=torch.int64).view(1, -1)
    return ((entry * 29 + byte * 7 + (entry >> 2)) & 0xFF).to(torch.uint8)


def _make_post_transaction(
    device: torch.device,
    queue_count: int,
    private_arena_capacity: int,
) -> dict[str, torch.Tensor]:
    """Allocate private forward state with a deliberately non-identity map."""
    row_count = target.BATCH_SIZE
    source_capacity = target.SOURCE_TOKENS
    source_lens = (target.SOURCE_TOKENS, target.SOURCE_TOKENS - ROWS_PER_ENTRY)
    pool_rows = row_count
    req_pool_entries = torch.tensor((1, 0), dtype=torch.int32, device=device)
    kv_object_map = torch.full(
        (pool_rows, source_capacity), -1, dtype=torch.int32, device=device
    )
    for row, object_len in enumerate(source_lens):
        pool = int(req_pool_entries[row].item())
        source = torch.arange(object_len, dtype=torch.int32, device=device)
        kv_object_map[pool, :object_len] = object_len - 1 - source
    return {
        # A zero sortable threshold intentionally creates a dense candidate
        # frontier. The arena limits the actual Direct-HCCP wave to 256 items;
        # P4 already establishes that this does not alter official Top-K.
        "resident_threshold": torch.zeros(row_count, dtype=torch.int32, device=device),
        "resident_threshold_valid": torch.ones(
            row_count, dtype=torch.int32, device=device
        ),
        "kv_object_slots_snapshot": torch.full(
            (pool_rows, source_capacity), -1, dtype=torch.int32, device=device
        ),
        "req_pool_entries": req_pool_entries,
        "kv_object_map": kv_object_map,
        "private_source_to_slot": torch.full(
            (row_count, source_capacity), -1, dtype=torch.int32, device=device
        ),
        "private_ready_table": torch.zeros(
            (queue_count, private_arena_capacity),
            dtype=torch.int32,
            device=device,
        ),
        "private_arena_head": torch.zeros(queue_count, dtype=torch.int32, device=device),
        # A non-zero allocation makes a missing copy distinguishable from a
        # legitimate all-zero source row.
        "private_packed_output": torch.full(
            (queue_count * private_arena_capacity, PACKED_KV_ROW_BYTES),
            0xA5,
            dtype=torch.uint8,
            device=device,
        ),
        "queue_control": torch.zeros(
            STREAMING_CONTROL_WORDS, dtype=torch.int64, device=device
        ),
        "queue_leases": torch.zeros(queue_count, dtype=torch.int32, device=device),
        "kv_object_lens": torch.tensor(source_lens, dtype=torch.int32, device=device),
    }


def _make_descriptor_templates(
    device: torch.device, queue_count: int
) -> dict[str, list[torch.Tensor]]:
    def alloc(dtype: torch.dtype) -> list[torch.Tensor]:
        return [
            torch.zeros(MAX_DENSE_REQUESTS, dtype=dtype, device=device)
            for _ in range(queue_count)
        ]

    return {
        "indices": alloc(torch.int32),
        "offsets": alloc(torch.int32),
        "masks": alloc(torch.uint8),
        "output_indices": alloc(torch.int32),
    }


def _require_logical_npu0_tensors(named: dict[str, torch.Tensor]) -> None:
    """Close KVAF's device-ordinal validation gap before a raw ACL bind."""
    bad = [
        f"{name}={tensor.device}"
        for name, tensor in named.items()
        if not isinstance(tensor, torch.Tensor) or str(tensor.device) != "npu:0"
    ]
    if bad:
        raise RuntimeError(
            "P3_POST_PAYLOAD_DIAGNOSTIC tensor_device_mismatch=" + ",".join(bad)
        )


def _control_words(control: torch.Tensor) -> list[int]:
    torch.npu.synchronize()
    return [_u64(value) for value in control.detach().cpu().tolist()]


def _assert_bound_control(
    transaction: dict[str, torch.Tensor], queue_count: int, private_arena_capacity: int
) -> list[int]:
    words = _control_words(transaction["queue_control"])
    expected_stride = private_arena_capacity * PACKED_KV_ROW_BYTES
    checks = {
        "magic": words[0] == STREAMING_CONTROL_MAGIC,
        "version": words[1] == STREAMING_CONTROL_VERSION,
        "channel_entity": words[2] != 0,
        "queue_state": words[3] != 0,
        "core_status": words[4] != 0,
        "output_alias": words[5] == _u64(transaction["private_packed_output"].data_ptr()),
        "queue_leases_alias": words[6] == _u64(transaction["queue_leases"].data_ptr()),
        "descriptor_scratch": words[7] != 0,
        "queue_count": words[8] == queue_count,
        "output_rows": words[9] == queue_count * private_arena_capacity,
        "private_stride": words[10] == expected_stride,
        "output_offset": words[11] == 0,
        "jettys_per_core": words[12] == 1,
        "block_bytes": words[13] == PACKED_KV_ROW_BYTES,
        "descriptor_templates": words[14] != 0,
        "template_count": words[15] == queue_count,
    }
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        raise AssertionError(
            "invalid KVAF streaming control: "
            f"bad={bad} words={words} "
            f"output_ptr=0x{transaction['private_packed_output'].data_ptr():x} "
            f"lease_ptr=0x{transaction['queue_leases'].data_ptr():x}"
        )
    return words


def _require_output_aliases(
    transaction: dict[str, torch.Tensor],
    publication: dict[str, torch.Tensor],
    outputs: tuple[torch.Tensor, ...],
) -> None:
    for name, output_index in (
        ("private_source_to_slot", 2),
        ("private_ready_table", 3),
        ("private_arena_head", 4),
        ("private_packed_output", 5),
        ("queue_leases", 6),
    ):
        if transaction[name].data_ptr() != outputs[output_index].data_ptr():
            raise AssertionError(
                f"{name} lost its graph-visible alias: "
                f"input=0x{transaction[name].data_ptr():x} "
                f"output=0x{outputs[output_index].data_ptr():x}"
            )
    for name, output_index in (("score_workspace", 9), ("tile_receipts", 10)):
        if publication[name].data_ptr() != outputs[output_index].data_ptr():
            raise AssertionError(
                f"{name} lost its graph-visible alias: "
                f"input=0x{publication[name].data_ptr():x} "
                f"output=0x{outputs[output_index].data_ptr():x}"
            )


def _call_post_op(
    inputs: dict[str, torch.Tensor],
    transaction: dict[str, torch.Tensor],
    publication: dict[str, torch.Tensor],
    target_opapi: Any,
) -> tuple[torch.Tensor, ...]:
    if "nanovllm_streaming_quant_li_get" in sys.modules:
        raise RuntimeError(
            "run in a fresh Python process: the isolated target bridge was "
            "already imported before the source-matched official reference"
        )
    _stage("target_bridge_bind_begin")
    target._bind_isolated_target_opp(target_opapi)
    _stage("target_bridge_bind_done")
    _stage("target_bridge_import_begin")
    importlib.import_module("nanovllm_streaming_quant_li_get")
    _stage("target_bridge_import_done")
    namespace = getattr(torch.ops, "a5_streaming_quant_li_get", None)
    op = getattr(namespace, "streaming_quant_li_get", None) if namespace else None
    if op is None:
        raise RuntimeError(
            "isolated target bridge did not register "
            "torch.ops.a5_streaming_quant_li_get.streaming_quant_li_get"
        )
    _stage("target_op_submit_begin")
    result = op(
        inputs["query"],
        inputs["key"],
        inputs["weights"],
        inputs["query_dequant_scale"],
        inputs["key_dequant_scale"],
        inputs["actual_seq_lengths_query"],
        inputs["actual_seq_lengths_key"],
        inputs["block_table"],
        inputs["metadata"],
        transaction["resident_threshold"],
        transaction["resident_threshold_valid"],
        transaction["kv_object_slots_snapshot"],
        transaction["req_pool_entries"],
        transaction["kv_object_map"],
        transaction["private_source_to_slot"],
        transaction["private_ready_table"],
        transaction["private_arena_head"],
        transaction["private_packed_output"],
        transaction["queue_control"],
        transaction["queue_leases"],
        transaction["kv_object_lens"],
        publication["score_workspace"],
        publication["tile_receipts"],
        publication["streaming_epoch"],
        kv_object_domain=0,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=target.SPARSE_COUNT,
        sparse_mode=3,
        pre_tokens=target.INT64_MAX,
        next_tokens=target.INT64_MAX,
        cmp_ratio=1,
        return_values=False,
        stride=int(inputs["key"].stride(0)),
        scale_stride=int(inputs["key_dequant_scale"].stride(0)),
        kvaf_rows_per_entry=ROWS_PER_ENTRY,
    )
    if not isinstance(result, tuple) or len(result) != 11:
        raise AssertionError(
            "derived bridge did not return the required 24-input/11-output ABI"
        )
    _stage("target_op_submitted", outputs=len(result))
    return result


def _terminal_summary(
    transaction: dict[str, torch.Tensor], outputs: tuple[torch.Tensor, ...]
) -> dict[str, int]:
    telemetry = outputs[7].detach().cpu().contiguous()
    leases = transaction["queue_leases"].detach().cpu().contiguous()
    ready = transaction["private_ready_table"].detach().cpu().contiguous()
    source_to_slot = transaction["private_source_to_slot"].detach().cpu().contiguous()
    expected_shape = (transaction["queue_leases"].numel(), QUEUE_TELEMETRY_COLUMNS)
    if tuple(telemetry.shape) != expected_shape:
        raise AssertionError(
            "unexpected P3 telemetry shape: "
            f"got={tuple(telemetry.shape)} expected={expected_shape}"
        )
    summary = {
        "doorbell_posted": int(telemetry[:, TELEMETRY_DOORBELL_POSTED].sum().item()),
        "cq_retired_ready": int(telemetry[:, TELEMETRY_CQ_RETIRED_READY].sum().item()),
        "cq_retired_failed": int(telemetry[:, TELEMETRY_CQ_RETIRED_FAILED].sum().item()),
        "lease_released": int(telemetry[:, TELEMETRY_LEASE_RELEASED].sum().item()),
        "control_rejected": int(telemetry[:, TELEMETRY_CONTROL_REJECTED].sum().item()),
        "score_tiles_with_cq_pending": int(
            telemetry[:, TELEMETRY_SCORE_TILES_WITH_CQ_PENDING].sum().item()
        ),
        "private_claims": int(telemetry[:, 5].sum().item()),
        "mapped_sources": int(torch.count_nonzero(source_to_slot >= 0).item()),
        "ready_slots": int(torch.count_nonzero(ready == 2).item()),
        "failed_slots": int(torch.count_nonzero(ready == 3).item()),
    }
    required = {
        "control_rejected": summary["control_rejected"] == 0,
        "doorbell_posted": summary["doorbell_posted"] > 0,
        "private_claims": summary["private_claims"] > 0,
        "mapped_sources": summary["mapped_sources"] > 0,
        "cq_retired_ready": summary["cq_retired_ready"] > 0,
        "ready_slots": summary["ready_slots"] > 0,
        "cq_retired_failed": summary["cq_retired_failed"] == 0,
        "failed_slots": summary["failed_slots"] == 0,
        "lease_released": summary["lease_released"] > 0,
        # This is the strict online-pipeline gate.  The current score tile
        # reaches the hook only after its real C8 reduction; a nonzero count
        # means a preceding Direct-HCCP wave was still nonterminal when C8
        # had already produced a later score tile.  Doorbell/CQ counts alone
        # would only prove a serial post followed by a terminal drain.
        "score_tiles_with_cq_pending": summary["score_tiles_with_cq_pending"] > 0,
        "queue_leases_zero": bool(torch.equal(leases, torch.zeros_like(leases))),
    }
    bad = [name for name, ok in required.items() if not ok]
    if bad:
        raise AssertionError(
            "P3_POST_PAYLOAD_DIAGNOSTIC terminal="
            f"bad={bad} summary={summary} leases={leases.tolist()} "
            f"telemetry={telemetry.tolist()}"
        )
    return summary


def _probe_summary(
    transaction: dict[str, torch.Tensor], outputs: tuple[torch.Tensor, ...]
) -> dict[str, int]:
    """Read the same terminal state without asserting that a Get was posted."""
    telemetry = outputs[7].detach().cpu().contiguous()
    leases = transaction["queue_leases"].detach().cpu().contiguous()
    ready = transaction["private_ready_table"].detach().cpu().contiguous()
    source_to_slot = transaction["private_source_to_slot"].detach().cpu().contiguous()
    expected_shape = (transaction["queue_leases"].numel(), QUEUE_TELEMETRY_COLUMNS)
    if tuple(telemetry.shape) != expected_shape:
        raise AssertionError(
            "unexpected P3 telemetry shape: "
            f"got={tuple(telemetry.shape)} expected={expected_shape}"
        )
    return {
        "tile_ready_calls": int(telemetry[:, 0].sum().item()),
        "doorbell_posted": int(telemetry[:, TELEMETRY_DOORBELL_POSTED].sum().item()),
        "cq_retired_ready": int(telemetry[:, TELEMETRY_CQ_RETIRED_READY].sum().item()),
        "cq_retired_failed": int(telemetry[:, TELEMETRY_CQ_RETIRED_FAILED].sum().item()),
        "lease_released": int(telemetry[:, TELEMETRY_LEASE_RELEASED].sum().item()),
        "control_rejected": int(telemetry[:, TELEMETRY_CONTROL_REJECTED].sum().item()),
        "score_tiles_with_cq_pending": int(
            telemetry[:, TELEMETRY_SCORE_TILES_WITH_CQ_PENDING].sum().item()
        ),
        "private_claims": int(telemetry[:, 5].sum().item()),
        "mapped_sources": int(torch.count_nonzero(source_to_slot >= 0).item()),
        "ready_slots": int(torch.count_nonzero(ready == 2).item()),
        "failed_slots": int(torch.count_nonzero(ready == 3).item()),
        "leases_nonzero": int(torch.count_nonzero(leases).item()),
    }


def _apply_post_exercise(
    transaction: dict[str, torch.Tensor], mode: str
) -> None:
    """Select one device-side closure without changing C8 inputs or output ABI.

    ``all_busy`` makes ``MayNeedScoreObservation`` false, so the same target
    OPP runs its normal C8 path but bypasses the P3 hook altogether.
    ``threshold_off`` leaves a queue free; it reaches the hook and its SIMT
    acquire/release path, but rejects every score before private claim/Begin.
    ``full`` is the real thresholded cold-candidate Post/CQ path.
    """
    if mode == "all_busy":
        transaction["queue_leases"].fill_(1)
    elif mode == "threshold_off":
        transaction["resident_threshold_valid"].zero_()
    elif mode != "full":
        raise ValueError(f"unsupported post exercise mode: {mode}")
    torch.npu.synchronize()
    leases = transaction["queue_leases"].detach().cpu().tolist()
    threshold_valid = transaction["resident_threshold_valid"].detach().cpu().tolist()
    _stage(
        "post_exercise_applied",
        mode=mode,
        leases=leases,
        threshold_valid=threshold_valid,
    )


def _assert_non_payload_mode(
    mode: str,
    transaction: dict[str, torch.Tensor],
    outputs: tuple[torch.Tensor, ...],
) -> dict[str, int]:
    """Assert exactly what the two isolation controls are meant to prove."""
    summary = _probe_summary(transaction, outputs)
    common = {
        "control_rejected": summary["control_rejected"] == 0,
        "doorbell_posted": summary["doorbell_posted"] == 0,
        "private_claims": summary["private_claims"] == 0,
        "mapped_sources": summary["mapped_sources"] == 0,
        "ready_slots": summary["ready_slots"] == 0,
        "failed_slots": summary["failed_slots"] == 0,
        "cq_retired_ready": summary["cq_retired_ready"] == 0,
        "cq_retired_failed": summary["cq_retired_failed"] == 0,
        "no_pending_cq_score_tiles": summary["score_tiles_with_cq_pending"] == 0,
    }
    if mode == "all_busy":
        common["all_busy_kept_leases"] = summary["leases_nonzero"] == transaction[
            "queue_leases"
        ].numel()
        common["all_busy_skipped_hook"] = summary["tile_ready_calls"] == 0
    elif mode == "threshold_off":
        common["threshold_off_entered_hook"] = summary["tile_ready_calls"] > 0
        common["threshold_off_released_lease"] = summary["leases_nonzero"] == 0
        common["threshold_off_no_cq"] = summary["lease_released"] == 0
    else:
        raise ValueError(f"not an isolation mode: {mode}")
    bad = [name for name, ok in common.items() if not ok]
    if bad:
        raise AssertionError(
            "P3_POST_PAYLOAD_DIAGNOSTIC isolation="
            f"mode={mode} bad={bad} summary={summary}"
        )
    return summary


def _validate_payload_chain(
    transaction: dict[str, torch.Tensor],
    ddr_payload: torch.Tensor,
    keys: list[int],
    access: Any,
    private_arena_capacity: int,
) -> dict[str, int]:
    """Check every completed private slot against the exact runtime directory chain."""
    source_to_slot = transaction["private_source_to_slot"].detach().cpu().contiguous()
    ready = transaction["private_ready_table"].detach().cpu().contiguous()
    private_payload = transaction["private_packed_output"].detach().cpu().contiguous()
    req_pool_entries = transaction["req_pool_entries"].detach().cpu().contiguous()
    kv_object_map = transaction["kv_object_map"].detach().cpu().contiguous()
    kv_object_lens = transaction["kv_object_lens"].detach().cpu().contiguous()
    access_addresses = access.addresses.detach().cpu().contiguous()
    access_valid = access.valid.detach().cpu().contiguous()
    mapped = torch.nonzero(source_to_slot >= 0, as_tuple=False)
    ready_slots = int(torch.count_nonzero(ready == 2).item())
    unique_slots: set[int] = set()
    samples: list[str] = []
    error_count = 0

    def add_error(text: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(samples) < MAX_DIAGNOSTIC_SAMPLES:
            samples.append(text)

    if mapped.numel() == 0:
        add_error("no_private_source_mapping")
    for coordinate in mapped.tolist():
        row, source = (int(coordinate[0]), int(coordinate[1]))
        pool = int(req_pool_entries[row].item()) if 0 <= row < req_pool_entries.numel() else -1
        source_len = int(kv_object_lens[row].item()) if 0 <= row < kv_object_lens.numel() else -1
        slot = int(source_to_slot[row, source].item())
        object_id = -1
        table_index = -1
        byte_offset = -1
        key = -1
        queue = -1
        local_slot = -1
        ready_state = -1
        valid = -1
        address = 0
        if not (0 <= pool < kv_object_map.size(0)):
            add_error(f"row={row} source={source} bad_pool={pool}")
            continue
        if not (0 <= source < source_len <= kv_object_map.size(1)):
            add_error(
                f"row={row} source={source} pool={pool} bad_source_len={source_len}"
            )
            continue
        object_id = int(kv_object_map[pool, source].item())
        if not (0 <= object_id < source_len):
            add_error(
                f"row={row} source={source} pool={pool} object={object_id} "
                f"bad_object_len={source_len}"
            )
            continue
        table_index = object_id // ROWS_PER_ENTRY
        byte_offset = (object_id % ROWS_PER_ENTRY) * PACKED_KV_ROW_BYTES
        if not (0 <= table_index < len(keys) and table_index < access_valid.numel()):
            add_error(
                f"row={row} source={source} pool={pool} object={object_id} "
                f"bad_table={table_index} key_count={len(keys)} valid_count={access_valid.numel()}"
            )
            continue
        key = int(keys[table_index])
        valid = int(access_valid[table_index].item())
        address = int(access_addresses[table_index].item())
        queue = slot // private_arena_capacity
        local_slot = slot % private_arena_capacity
        if not (0 <= slot < private_payload.size(0) and 0 <= queue < ready.size(0)):
            add_error(
                f"row={row} source={source} pool={pool} object={object_id} table={table_index} "
                f"slot={slot} bad_slot payload_rows={private_payload.size(0)} queues={ready.size(0)}"
            )
            continue
        ready_state = int(ready[queue, local_slot].item())
        if slot in unique_slots:
            add_error(
                f"row={row} source={source} pool={pool} object={object_id} table={table_index} "
                f"slot={slot} duplicate_slot=1"
            )
            continue
        unique_slots.add(slot)
        if valid != 1 or address == 0 or ready_state != 2:
            add_error(
                f"row={row} source={source} pool={pool} object={object_id} table={table_index} "
                f"offset={byte_offset} slot={slot} queue={queue} local={local_slot} "
                f"ready={ready_state} valid={valid} addr=0x{address:x}"
            )
            continue
        if not (1 <= key <= ddr_payload.size(0)):
            add_error(
                f"row={row} source={source} pool={pool} object={object_id} table={table_index} "
                f"invalid_key={key} payload_entries={ddr_payload.size(0)}"
            )
            continue
        expected = ddr_payload[key - 1, byte_offset : byte_offset + PACKED_KV_ROW_BYTES]
        actual = private_payload[slot]
        if not torch.equal(expected, actual):
            first_index, expected_byte, actual_byte = _first_difference(expected, actual)
            add_error(
                f"row={row} source={source} pool={pool} object={object_id} table={table_index} "
                f"offset={byte_offset} key={key} slot={slot} queue={queue} local={local_slot} "
                f"ready={ready_state} valid={valid} addr=0x{address:x} "
                f"expected_sha16={_sha16(expected)} actual_sha16={_sha16(actual)} "
                f"first_diff={first_index} expected={expected_byte} actual={actual_byte}"
            )

    mapped_sources = int(mapped.size(0))
    if mapped_sources != len(unique_slots) or mapped_sources != ready_slots:
        add_error(
            "cardinality "
            f"mapped_sources={mapped_sources} unique_slots={len(unique_slots)} "
            f"ready_slots={ready_slots}"
        )
    if error_count:
        raise AssertionError(
            "P3_POST_PAYLOAD_DIAGNOSTIC "
            f"errors={error_count} mapped_sources={mapped_sources} "
            f"unique_slots={len(unique_slots)} ready_slots={ready_slots} "
            f"samples={' || '.join(samples)}"
        )
    return {
        "mapped_sources": mapped_sources,
        "unique_slots": len(unique_slots),
        "ready_slots": ready_slots,
        "payload_bytes": mapped_sources * PACKED_KV_ROW_BYTES,
    }


def main() -> int:
    args = parse_args()
    device = _require_test_args(args)
    if os.getenv(target.PUBLICATION_MODE_ENV) != "off":
        raise RuntimeError(f"{target.PUBLICATION_MODE_ENV}=off is required for post payload proof")
    if os.getenv(target.INLINE_MODE_ENV) != "post":
        raise RuntimeError(f"{target.INLINE_MODE_ENV}=post is required for post payload proof")

    reference_opapi, reference_vendor = target._resolve_single_opp(
        os.getenv(target.REFERENCE_OPP_ENV), "reference"
    )
    target_opapi, target_vendor = target._resolve_single_opp(
        os.getenv(target.TARGET_OPP_ENV), "target"
    )
    target_provenance = target._read_target_provenance(target_vendor, "off", "post")
    target._prepend_unique_path("ASCEND_CUSTOM_OPP_PATH", (reference_vendor, target_vendor))
    official_op, metadata_op, reference_provenance = target._require_source_matched_official_c8_ops(
        reference_opapi, reference_vendor
    )
    print(
        "P3_POST_PAYLOAD_CONFIG "
        f"reference_head={reference_provenance['source_head']} "
        f"target_head={target_provenance['source_head']} "
        f"kvaf_head={target_provenance['kvaf_source_head']} "
        f"inline={target_provenance['inline_mode']} "
        f"publication={target_provenance['publication_mode']} "
        f"exercise={args.post_exercise} "
        f"kvaf_module={getattr(kvaf, '__file__', '<unknown>')}",
        flush=True,
    )

    # The official C8 call occurs before the target bridge resolver is bound.
    inputs = target._build_official_inputs(device)
    inputs["metadata"] = target._official_metadata(metadata_op, inputs)
    _stage("official_c8_submit_begin")
    official_indices = target._call_official(official_op, inputs)
    _stage("official_c8_sync_begin")
    torch.npu.synchronize()
    _stage("official_c8_sync_done")

    runtime = _create_runtime(args)
    _require_streaming_runtime_api(runtime)
    table: Any | None = None
    control_bound = False
    try:
        ddr_payload = _build_local_ddr_payload(args.table_capacity)
        source_payload = ddr_payload.to(device).contiguous()
        keys = list(range(1, args.table_capacity + 1))
        runtime.initialize(args.table_capacity * ENTRY_BYTES, ENTRY_BYTES)
        runtime.batch_put(
            keys,
            source_payload.reshape(-1),
            [entry * ENTRY_BYTES for entry in range(args.table_capacity)],
            [ENTRY_BYTES] * args.table_capacity,
        )
        _stage("kvaf_batch_put_sync_begin")
        torch.npu.synchronize()
        _stage("kvaf_batch_put_sync_done")
        table = runtime.prepare_access_table(keys)
        access = runtime.access_table_view(table, device="npu", capacity=args.table_capacity)
        _stage("access_table_sync_begin")
        torch.npu.synchronize()
        _stage("access_table_sync_done")

        transaction = _make_post_transaction(
            device, args.queue_count, args.private_arena_capacity
        )
        _require_logical_npu0_tensors(
            {
                **inputs,
                **transaction,
                "source_payload": source_payload,
                "access.addresses": access.addresses,
                "access.valid": access.valid,
            }
        )
        descriptor = _make_descriptor_templates(device, args.queue_count)
        _require_logical_npu0_tensors(
            {
                f"descriptor.{name}[{index}]": tensor
                for name, tensors in descriptor.items()
                for index, tensor in enumerate(tensors)
            }
        )
        producer_stream = torch.npu.current_stream()
        runtime.bind_streaming_batch_get_control(
            [args.layer_id],
            [transaction["private_packed_output"]],
            [transaction["queue_control"]],
            transaction["queue_leases"],
            args.queue_count,
            producer_stream,
        )
        control_bound = True
        runtime.configure_streaming_batch_get_queue_templates(
            [args.layer_id] * args.queue_count,
            list(range(args.queue_count)),
            [access.addresses] * args.queue_count,
            [access.valid] * args.queue_count,
            descriptor["indices"],
            descriptor["offsets"],
            descriptor["masks"],
            descriptor["output_indices"],
            producer_stream,
        )
        control_words = _assert_bound_control(
            transaction, args.queue_count, args.private_arena_capacity
        )
        snapshot = runtime.streaming_pool_snapshot()
        print(
            "P3_POST_PAYLOAD_BIND "
            f"queue_count={args.queue_count} private_arena_capacity={args.private_arena_capacity} "
            f"output_ptr=0x{transaction['private_packed_output'].data_ptr():x} "
            f"control_output_ptr=0x{control_words[5]:x} "
            f"lease_ptr=0x{transaction['queue_leases'].data_ptr():x} "
            f"control_lease_ptr=0x{control_words[6]:x} "
            f"templates_configured={snapshot.get('template_configure_count')}",
            flush=True,
        )
        _stage("streaming_control_bound")

        _apply_post_exercise(transaction, args.post_exercise)
        persistent_snapshot = transaction["kv_object_slots_snapshot"].detach().cpu().clone()

        publication = target._make_publication_buffers(device)
        _require_logical_npu0_tensors(publication)
        _stage(
            "publication_allocated",
            score_workspace=publication["score_workspace"].numel(),
            tile_receipts=publication["tile_receipts"].numel(),
        )
        outputs = _call_post_op(inputs, transaction, publication, target_opapi)
        _stage("target_op_sync_begin")
        torch.npu.synchronize()
        _stage("target_op_sync_done")
        _require_logical_npu0_tensors(
            {f"output[{index}]": value for index, value in enumerate(outputs)}
        )
        _require_output_aliases(transaction, publication, outputs)
        expected_score_source_lens = target._expected_score_source_lens(device)
        topk_equal, topk_diagnostic = _strict_tensor_diagnostic(
            "sparse_indices", official_indices, outputs[0]
        )
        source_lens_equal, source_lens_diagnostic = _strict_tensor_diagnostic(
            "score_source_lens", expected_score_source_lens, outputs[8]
        )
        print("P3_POST_MATH_DIAG " + topk_diagnostic, flush=True)
        print("P3_POST_MATH_DIAG " + source_lens_diagnostic, flush=True)
        if not topk_equal or not source_lens_equal:
            raise AssertionError(
                "derived inline=post strict math mismatch: "
                f"sparse_indices_equal={int(topk_equal)} "
                f"score_source_lens_equal={int(source_lens_equal)}"
            )
        target._assert_publication_contract("off", publication)
        if not torch.equal(transaction["kv_object_slots_snapshot"].detach().cpu(), persistent_snapshot):
            raise AssertionError("post speculation mutated persistent kv_object_slots_snapshot")
        if args.post_exercise == "full":
            terminal = _terminal_summary(transaction, outputs)
            payload = _validate_payload_chain(
                transaction,
                ddr_payload,
                keys,
                access,
                args.private_arena_capacity,
            )
            print(
                "P3_POST_ONLINE_OVERLAP_PROOF "
                f"score_tiles_with_cq_pending={terminal['score_tiles_with_cq_pending']} "
                f"doorbell_posted={terminal['doorbell_posted']} "
                f"cq_ready={terminal['cq_retired_ready']} "
                "meaning=later_exact_score_tile_observed_while_prior_get_cq_nonterminal",
                flush=True,
            )
            print(
                "P3_POST_PAYLOAD_STRICT_PASS "
                "topk=bitwise score_source_lens=exact publication=off "
                "persistent_mapping=unchanged mapping=permuted_pool "
                f"mapped_sources={payload['mapped_sources']} "
                f"unique_slots={payload['unique_slots']} ready_slots={payload['ready_slots']} "
                f"payload_bytes={payload['payload_bytes']} "
                f"doorbell_posted={terminal['doorbell_posted']} "
                f"cq_ready={terminal['cq_retired_ready']} "
                f"score_tiles_with_cq_pending={terminal['score_tiles_with_cq_pending']} "
                f"physical_npu={args.physical_npu_id} logical_npu=0",
                flush=True,
            )
        else:
            summary = _assert_non_payload_mode(args.post_exercise, transaction, outputs)
            print(
                "P3_POST_ISOLATION_PASS "
                f"mode={args.post_exercise} topk=bitwise "
                "score_source_lens=exact persistent_mapping=unchanged "
                f"summary={summary} "
                f"physical_npu={args.physical_npu_id} logical_npu=0",
                flush=True,
            )
        return 0
    finally:
        if control_bound:
            runtime.release_streaming_batch_get_control()
        if table is not None:
            runtime.release_access_table(table)
        runtime.shutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            "P3_POST_PAYLOAD_FAIL "
            f"stage={_LAST_STAGE} type={type(error).__name__} text={error}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
