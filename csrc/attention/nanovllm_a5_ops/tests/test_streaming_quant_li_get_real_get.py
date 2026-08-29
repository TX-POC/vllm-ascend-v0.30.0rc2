#!/usr/bin/env python3
"""Target-only real-local-Get probe for ``A5StreamingQuantLiGet``.

This is deliberately a component test, not a vLLM product route and not a
bandwidth benchmark.  It uses the real local KVAF runtime to cold-bind one
graph-visible streaming queue lease table and one layer-private output, then
publishes real Direct-HCCP descriptor templates.  The derived C8 operator is
run on the same nine official-C8 input allocations as the reference operator.

The test proves only the following narrow contract:

* official and derived C8 ``sparse_indices`` are bitwise equal;
* the derived launch uses the exact queue lease allocation that KVAF bound;
* a device-side speculative wave rings a real Direct-HCCP doorbell, reaches a
  terminal CQ state, and releases its graph-visible lease; and
* no persistent resident mapping is changed by the speculative path.

Current local Direct-HCCP payload bytes are a known failed validation area.
This test intentionally labels them ``known_failed`` and neither copies the
payload back nor reports bandwidth.  It must not be used as a payload or
end-to-end-performance result.

Run it in a fresh Python process after building the isolated OPP bridge and
setting ``NANOVLLM_A5_STREAMING_LI_OPP``.  It does not import or select any
vLLM product execution route.
"""

from __future__ import annotations

import argparse
import importlib
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
# Keep this target-only legacy probe compatible with the graph-visible
# telemetry ABI of the source-derived operator.  P6 appended four scheduling
# counters; the existing lifecycle columns below retain their original
# indices.
QUEUE_TELEMETRY_COLUMNS = 18
TELEMETRY_DOORBELL_POSTED = 8
TELEMETRY_CQ_RETIRED_READY = 9
TELEMETRY_CQ_RETIRED_FAILED = 10
TELEMETRY_LEASE_RELEASED = 11
TELEMETRY_CONTROL_REJECTED = 12
DEFAULT_QUEUE_COUNT = 1
DEFAULT_PRIVATE_ARENA_CAPACITY = 256
MAX_DENSE_REQUESTS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # The target adapter deliberately fixes logical npu:0.  A remote caller
    # that wants a different physical card must set ASCEND_VISIBLE_DEVICES
    # before starting this fresh process; do not remap it after a CANN module
    # has been imported.
    parser.add_argument("--device", default="npu:0")
    # This target probe validates the native-enqueue ABI that is currently
    # bound by KVAF's local NpuRuntime.  It deliberately does not pretend
    # that this runtime also exposes the older CPU-service constructor.
    parser.add_argument("--transport", choices=("local",), default="local")
    parser.add_argument("--batch-get-kernel", required=True)
    parser.add_argument("--batch-put-kernel", required=True)
    parser.add_argument("--table-capacity", type=int, default=target.SOURCE_TOKENS // ROWS_PER_ENTRY)
    parser.add_argument("--queue-count", type=int, default=DEFAULT_QUEUE_COUNT)
    parser.add_argument(
        "--private-arena-capacity",
        type=int,
        default=DEFAULT_PRIVATE_ARENA_CAPACITY,
    )
    parser.add_argument("--layer-id", type=int, default=0)
    return parser.parse_args()


def _u64(value: int) -> int:
    return int(value) & ((1 << 64) - 1)


def _require_test_args(args: argparse.Namespace) -> None:
    if args.device != "npu:0":
        raise ValueError(
            "this isolated adapter is logical-npu:0 only; select the target "
            "physical card with ASCEND_VISIBLE_DEVICES before process start"
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
def _create_runtime(args: argparse.Namespace, device_id: int) -> Any:
    """Match the currently bound local NpuRuntime constructor exactly."""
    runtime = kvaf.NpuRuntime(
        args.batch_get_kernel,
        device_id,
        args.queue_count,
        args.batch_put_kernel,
        1,
    )
    if callable(getattr(runtime, "link_to", None)):
        raise RuntimeError(
            "this test is local-only, but the imported KVAF module exposes "
            "a remote CPU-service ABI; select the matching local native-enqueue build"
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
            + ". Build the matching KVAF kip_demo_cs_native_enqueue revision; "
            "do not substitute a legacy BatchGet runtime."
        )


def _build_local_ddr_payload(table_capacity: int) -> torch.Tensor:
    """Create deterministic full KVAF entries without using them as a metric."""
    entry = torch.arange(table_capacity, dtype=torch.int64).view(-1, 1)
    byte = torch.arange(ENTRY_BYTES, dtype=torch.int64).view(1, -1)
    return ((entry * 29 + byte * 7 + (entry >> 2)) & 0xFF).to(torch.uint8)


def _make_graph_transaction(
    device: torch.device,
    queue_count: int,
    private_arena_capacity: int,
) -> dict[str, torch.Tensor]:
    """Allocate one layer's graph-visible private transaction state.

    The output and queue_leases allocations become the exact objects passed to
    ``bind_streaming_batch_get_control``.  No control pointer is fabricated in
    Python: KVAF publishes the real HCCP entities/templates into queue_control.
    """
    row_count = target.BATCH_SIZE
    source_capacity = target.SOURCE_TOKENS
    pool_rows = target.BATCH_SIZE
    return {
        "resident_threshold": torch.zeros(row_count, dtype=torch.int32, device=device),
        "resident_threshold_valid": torch.ones(
            row_count, dtype=torch.int32, device=device
        ),
        "kv_object_slots_snapshot": torch.full(
            (pool_rows, source_capacity), -1, dtype=torch.int32, device=device
        ),
        "req_pool_entries": torch.arange(row_count, dtype=torch.int32, device=device),
        # Raw-token object domain: source score S is object id L because
        # cmp_ratio=1.  This is the only mapping used by the probe.
        "kv_object_map": torch.arange(
            source_capacity, dtype=torch.int32, device=device
        ).repeat(pool_rows, 1).contiguous(),
        "private_source_to_slot": torch.full(
            (row_count, source_capacity), -1, dtype=torch.int32, device=device
        ),
        "private_ready_table": torch.zeros(
            (queue_count, private_arena_capacity),
            dtype=torch.int32,
            device=device,
        ),
        "private_arena_head": torch.zeros(queue_count, dtype=torch.int32, device=device),
        "private_packed_output": torch.empty(
            (queue_count * private_arena_capacity, PACKED_KV_ROW_BYTES),
            dtype=torch.uint8,
            device=device,
        ),
        "queue_control": torch.zeros(
            STREAMING_CONTROL_WORDS, dtype=torch.int64, device=device
        ),
        # KVAF zeros this during cold bind.  The operator must receive this
        # exact allocation, not an internal/decorative copy.
        "queue_leases": torch.empty(queue_count, dtype=torch.int32, device=device),
        "kv_object_lens": torch.tensor(
            (target.SOURCE_TOKENS, target.SOURCE_TOKENS - target.BLOCK_SIZE),
            dtype=torch.int32,
            device=device,
        ),
    }


def _make_descriptor_templates(
    device: torch.device,
    queue_count: int,
) -> dict[str, list[torch.Tensor]]:
    """Allocate disjoint mutable descriptor vectors for every queue."""
    def alloc(dtype: torch.dtype, count: int) -> list[torch.Tensor]:
        return [torch.zeros(count, dtype=dtype, device=device) for _ in range(queue_count)]

    return {
        "indices": alloc(torch.int32, MAX_DENSE_REQUESTS),
        "offsets": alloc(torch.int32, MAX_DENSE_REQUESTS),
        "masks": alloc(torch.uint8, MAX_DENSE_REQUESTS),
        "output_indices": alloc(torch.int32, MAX_DENSE_REQUESTS),
    }


def _control_summary(control: torch.Tensor) -> list[int]:
    torch.npu.synchronize()
    return [_u64(value) for value in control.cpu().tolist()]


def _assert_bound_control(
    control: torch.Tensor,
    private_packed_output: torch.Tensor,
    queue_leases: torch.Tensor,
    queue_count: int,
    private_arena_capacity: int,
) -> list[int]:
    words = _control_summary(control)
    expected_stride = private_arena_capacity * PACKED_KV_ROW_BYTES
    checks = {
        "magic": words[0] == STREAMING_CONTROL_MAGIC,
        "version": words[1] == STREAMING_CONTROL_VERSION,
        "channel_entity": words[2] != 0,
        "queue_state": words[3] != 0,
        "core_status": words[4] != 0,
        "output_alias": words[5] == _u64(private_packed_output.data_ptr()),
        "queue_leases_alias": words[6] == _u64(queue_leases.data_ptr()),
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
            "KVAF did not publish a valid streaming control: "
            f"bad={bad} words={words} output_ptr=0x{private_packed_output.data_ptr():x} "
            f"lease_ptr=0x{queue_leases.data_ptr():x}"
        )
    return words


def _require_mutable_aliases(
    transaction: dict[str, torch.Tensor], outputs: tuple[torch.Tensor, ...]
) -> None:
    for name, output_index in (
        ("private_source_to_slot", 2),
        ("private_ready_table", 3),
        ("private_arena_head", 4),
        ("private_packed_output", 5),
        ("queue_leases", 6),
    ):
        input_ptr = transaction[name].data_ptr()
        output_ptr = outputs[output_index].data_ptr()
        if input_ptr != output_ptr:
            raise AssertionError(
                f"{name} lost its graph-visible alias: "
                f"input=0x{input_ptr:x} output=0x{output_ptr:x}"
            )


def _call_derived(
    inputs: dict[str, torch.Tensor], transaction: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    """Call the public 21-input/9-output bridge with exact C8 input views."""
    if "nanovllm_streaming_quant_li_get" in sys.modules:
        raise RuntimeError(
            "run in a fresh Python process: the isolated target bridge was "
            "already imported before the normal C8 reference"
        )
    target._bind_isolated_target_opp()
    importlib.import_module("nanovllm_streaming_quant_li_get")
    namespace = getattr(torch.ops, "a5_streaming_quant_li_get", None)
    op = getattr(namespace, "streaming_quant_li_get", None) if namespace else None
    if op is None:
        raise RuntimeError(
            "isolated bridge did not register "
            "torch.ops.a5_streaming_quant_li_get.streaming_quant_li_get"
        )
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
        # Public PyTorch schema places this required semantic choice first;
        # the typed C++ bridge reorders it to CANN's generated ABI.
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
    if not isinstance(result, tuple) or len(result) != 9:
        raise AssertionError("derived bridge did not return the required 9-output ABI")
    return result


def _validate_terminal_state(
    transaction: dict[str, torch.Tensor], outputs: tuple[torch.Tensor, ...]
) -> dict[str, int]:
    telemetry = outputs[7].cpu()
    leases = transaction["queue_leases"].cpu()
    ready = transaction["private_ready_table"].cpu()
    private_slots = transaction["private_source_to_slot"].cpu()
    if tuple(telemetry.shape) != (transaction["queue_leases"].numel(), QUEUE_TELEMETRY_COLUMNS):
        raise AssertionError(
            "unexpected telemetry shape: "
            f"got={tuple(telemetry.shape)} expected="
            f"({transaction['queue_leases'].numel()},{QUEUE_TELEMETRY_COLUMNS})"
        )
    summary = {
        "doorbell_posted": int(telemetry[:, TELEMETRY_DOORBELL_POSTED].sum()),
        "cq_retired_ready": int(telemetry[:, TELEMETRY_CQ_RETIRED_READY].sum()),
        "cq_retired_failed": int(telemetry[:, TELEMETRY_CQ_RETIRED_FAILED].sum()),
        "lease_released": int(telemetry[:, TELEMETRY_LEASE_RELEASED].sum()),
        "control_rejected": int(telemetry[:, TELEMETRY_CONTROL_REJECTED].sum()),
        "private_claims": int(telemetry[:, 5].sum()),
        "mapped_sources": int(torch.count_nonzero(private_slots >= 0)),
        "ready_slots": int(torch.count_nonzero(ready == 2)),
        "failed_slots": int(torch.count_nonzero(ready == 3)),
    }
    if summary["control_rejected"] != 0:
        raise AssertionError(f"streaming control rejected: telemetry={telemetry.tolist()}")
    if summary["doorbell_posted"] <= 0:
        raise AssertionError(
            "no real Direct-HCCP doorbell was posted; "
            f"telemetry={telemetry.tolist()}"
        )
    if summary["private_claims"] <= 0 or summary["mapped_sources"] <= 0:
        raise AssertionError(
            "no score candidate was claimed into the transaction-private arena; "
            f"telemetry={telemetry.tolist()}"
        )
    if summary["cq_retired_ready"] <= 0 or summary["ready_slots"] <= 0:
        raise AssertionError(
            "the posted Direct-HCCP wave did not reach a ready terminal CQ state; "
            f"summary={summary} telemetry={telemetry.tolist()}"
        )
    if summary["cq_retired_failed"] != 0 or summary["failed_slots"] != 0:
        raise AssertionError(
            "the streaming Direct-HCCP CQ reported failure; "
            f"summary={summary} telemetry={telemetry.tolist()}"
        )
    if not bool(torch.equal(leases, torch.zeros_like(leases))):
        raise AssertionError(
            "graph-visible queue leases were not released after terminal CQ: "
            f"leases={leases.tolist()} telemetry={telemetry.tolist()}"
        )
    if summary["lease_released"] <= 0:
        raise AssertionError(
            "terminal CQ was observed but device telemetry recorded no lease release; "
            f"telemetry={telemetry.tolist()}"
        )
    return summary


def main() -> int:
    args = parse_args()
    _require_test_args(args)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    device_id = device.index if device.index is not None else torch.npu.current_device()

    # Official C8 runs first while its normal packaged OPP remains selected.
    # The target bridge's process-static resolver is selected only afterwards.
    official_op, metadata_op = target._require_official_c8_ops()
    inputs = target._build_official_inputs(device)
    inputs["metadata"] = target._official_metadata(metadata_op, inputs)
    official_indices = target._call_official(official_op, inputs)
    torch.npu.synchronize()

    runtime = _create_runtime(args, device_id)
    _require_streaming_runtime_api(runtime)
    table = None
    control_bound = False
    try:
        ddr_bytes = args.table_capacity * ENTRY_BYTES
        ddr_payload = _build_local_ddr_payload(args.table_capacity)
        # Keep the source allocation alive through the explicit cold-path
        # completion fence.  The real KVAF submission is asynchronous.
        source_payload = ddr_payload.to(device).contiguous()
        keys = list(range(1, args.table_capacity + 1))
        runtime.initialize(ddr_bytes, ENTRY_BYTES)
        runtime.batch_put(
            keys,
            source_payload.reshape(-1),
            [index * ENTRY_BYTES for index in range(args.table_capacity)],
            [ENTRY_BYTES] * args.table_capacity,
        )
        torch.npu.synchronize()
        table = runtime.prepare_access_table(keys)
        access = runtime.access_table_view(
            table, device="npu", capacity=args.table_capacity
        )
        torch.npu.synchronize()

        transaction = _make_graph_transaction(
            device, args.queue_count, args.private_arena_capacity
        )
        descriptor = _make_descriptor_templates(device, args.queue_count)
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
            transaction["queue_control"],
            transaction["private_packed_output"],
            transaction["queue_leases"],
            args.queue_count,
            args.private_arena_capacity,
        )
        snapshot = runtime.streaming_pool_snapshot()
        diagnostic = runtime.streaming_batch_get_control_diagnostic()
        print(
            "P3_STREAMING_REAL_GET_BIND "
            f"transport={args.transport} queue_count={args.queue_count} "
            f"private_arena_capacity={args.private_arena_capacity} "
            f"lease_ptr=0x{transaction['queue_leases'].data_ptr():x} "
            f"control_lease_ptr=0x{control_words[6]:x} "
            f"templates_configured={snapshot.get('template_configure_count')} "
            f"diagnostic={diagnostic}",
            flush=True,
        )

        outputs = _call_derived(inputs, transaction)
        torch.npu.synchronize()
        _require_mutable_aliases(transaction, outputs)
        torch.testing.assert_close(outputs[0], official_indices)
        torch.testing.assert_close(
            outputs[8], target._expected_score_source_lens(device)
        )
        # The source directory is a snapshot read by the hook.  Its all--1
        # state proves the speculative execution did not publish a persistent
        # resident mapping during this forward.
        if int(torch.count_nonzero(transaction["kv_object_slots_snapshot"] >= 0)) != 0:
            raise AssertionError("speculative path mutated persistent slot snapshot")
        terminal = _validate_terminal_state(transaction, outputs)
        print(
            "P3_STREAMING_REAL_GET_CORRECTNESS "
            "topk=bitwise_official_c8 score_source_lens=exact "
            "queue_leases=graph_visible_released persistent_mapping=unchanged "
            f"doorbell_posted={terminal['doorbell_posted']} "
            f"cq_ready={terminal['cq_retired_ready']} "
            f"private_claims={terminal['private_claims']} "
            f"ready_slots={terminal['ready_slots']}",
            flush=True,
        )
        # Do not inspect/copy payload bytes and do not calculate BW.  This is
        # explicit because existing zero-payload behavior is known failed.
        print(
            "P3_STREAMING_REAL_GET_PAYLOAD status=known_failed "
            "bytes_checked=0 bandwidth_reported=0",
            flush=True,
        )
        print("P3_STREAMING_REAL_GET_PASS", flush=True)
        return 0
    finally:
        # Terminal CQ and `torch.npu.synchronize()` were already required and
        # observed before this cold release.  The release is intentionally not
        # an implicit replay synchronization.
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
            "P3_STREAMING_REAL_GET_FAIL "
            f"type={type(error).__name__} text={error}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
