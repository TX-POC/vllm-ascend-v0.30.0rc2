"""Real local Direct-HCCP cold fixture for P8.

This file is deliberately small and owns only the cold resources which the
P8 device executor consumes.  It is *not* a second transport executor:

* P8's AIV Post phase performs the real ``PostOnly``;
* P8's AIV Drain phase performs ``TryRetire`` and ``Finish``;
* the fixture verifies the terminal private/queue state and byte-compares the
  completed local Direct-HCCP payload against the original DDR entries.

The fixture is local-only.  A remote CPU-service ``link_to`` ABI is rejected
before it allocates or binds anything.  Its C8 proof uses the separately built
source-matched official C8 bridge supplied by the P8 runner; P8 itself never
loads or modifies C8.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch_npu  # noqa: F401 -- registers PrivateUse1/NPU.

import kvaf


PACKED_KV_ROW_BYTES = 656
ROWS_PER_ENTRY = 128
ENTRY_BYTES = PACKED_KV_ROW_BYTES * ROWS_PER_ENTRY
QUEUE_COUNT = 1
PRIVATE_ROWS = 64
REQUEST_CAPACITY = 64
CONTROL_WORDS = 16
FORWARD_EPOCH = 37
PHASE_POST = 1
FORWARD_ABORT_MASK = 0x3

# P8's graph-visible worker telemetry columns.  Keep these local rather than
# importing a generated OPP header: the fixture only validates the stable ABI.
STAT_POST_COMMITTED = 5
STAT_CQ_POLL = 6
STAT_CQ_TERMINAL = 7
STAT_QUEUE_RELEASED = 8
STAT_PRIVATE_RELEASED = 9
STAT_CONTROL_REJECTED = 10
STAT_QUARANTINED = 11
STAT_ABORT = 14


def _required_file(name: str) -> str:
    value = os.environ.get(name, "")
    if not value or not Path(value).is_file():
        raise RuntimeError(
            f"P8 real Direct-HCCP fixture requires {name} to name a built file; "
            f"got {value!r}"
        )
    return value


def _sha16(value: torch.Tensor) -> str:
    # Hash the underlying tensor bytes, not a numeric conversion to uint8.
    # The latter would silently truncate C8 int32 indices and P8 int64 control
    # words, which makes a diagnostic hash much less useful precisely when a
    # state/identity mismatch must be located.  `view(torch.uint8)` preserves
    # every byte for the contiguous CPU representation used by the ABI.
    flat = value.detach().to("cpu").contiguous().view(torch.uint8).reshape(-1)
    # A bounded payload proof never needs an external package, and bytes() is
    # explicit about exactly which ABI bytes are digested.
    return hashlib.sha256(bytes(flat.tolist())).hexdigest()[:16]


def _u64(value: int) -> int:
    return int(value) & ((1 << 64) - 1)


def _npu(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device).contiguous()


def _require_local_streaming_runtime(runtime: Any) -> None:
    required = (
        "bind_streaming_batch_get_control",
        "configure_streaming_batch_get_queue_templates",
        "release_streaming_batch_get_control",
        "streaming_batch_get_control_diagnostic",
        "streaming_pool_snapshot",
        "release_access_table",
        "shutdown",
    )
    missing = [item for item in required if not callable(getattr(runtime, item, None))]
    if missing:
        raise RuntimeError(
            "P8 requires the local native-enqueue streaming KVAF ABI; missing="
            + ",".join(missing)
        )
    if callable(getattr(runtime, "link_to", None)):
        raise RuntimeError(
            "P8 local Direct-HCCP fixture rejected remote CPU-service KVAF ABI; "
            "select the native-enqueue local KVAF build"
        )


def _assert_cold_control(
    control: torch.Tensor, output: torch.Tensor, queue_leases: torch.Tensor
) -> list[int]:
    torch.npu.synchronize()
    words = [_u64(item) for item in control.cpu().tolist()]
    checks = {
        "magic": words[0] != 0,
        "version": words[1] != 0,
        "channel_entity": words[2] != 0,
        "queue_state": words[3] != 0,
        "core_status": words[4] != 0,
        "output_alias": words[5] == _u64(output.data_ptr()),
        "leases_alias": words[6] == _u64(queue_leases.data_ptr()),
        "descriptor_scratch": words[7] != 0,
        "queue_count": words[8] == QUEUE_COUNT,
        "output_rows": words[9] == PRIVATE_ROWS,
        "output_stride": words[10] == PRIVATE_ROWS * PACKED_KV_ROW_BYTES,
        "block_bytes": words[13] == PACKED_KV_ROW_BYTES,
        "templates": words[14] != 0,
        "template_count": words[15] == QUEUE_COUNT,
    }
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        raise RuntimeError(
            "P8 cold Direct-HCCP control is not graph-usable: "
            f"bad={bad} words={words} output_ptr=0x{output.data_ptr():x} "
            f"lease_ptr=0x{queue_leases.data_ptr():x}"
        )
    return words


def _payload(entries: int) -> torch.Tensor:
    """A deterministic, non-zero source for byte-exact Direct-HCCP checking."""
    table = torch.arange(entries, dtype=torch.int64).reshape(entries, 1)
    byte = torch.arange(ENTRY_BYTES, dtype=torch.int64).reshape(1, ENTRY_BYTES)
    return ((table * 29 + byte * 7 + (table >> 2) + 17) & 0xFF).to(torch.uint8)


def _capture_official_c8(device: torch.device) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
    """Run the untouched standalone official C8 bridge once.

    The runner builds and exposes this bridge/OPP afresh.  Keeping its resolver
    separate from P8 means an after-P8 repeat is a real C8 mathematical gate,
    not a comparison against a copied result.
    """
    # The existing source-matched adapter is only used for its public-ish
    # reference helpers.  It does not import vllm_ascend or a P3 target OPP.
    import streaming_quant_li_get_target_adapter as c8

    root = os.environ.get("NANOVLLM_A5_REFERENCE_C8_OPP", "")
    opapi, vendor = c8._resolve_single_opp(root, "reference")
    # The P8 runner imports the Stage-A and Stage-B bridges before it builds
    # this cold fixture.  Those bridges each add their own vendor directory.
    # Preserve that multi-vendor discovery set and add the untouched C8
    # vendor before the *first* official ACLNN call.  Replacing the path here
    # would make later Stage-A/B calls resolve a different OPP, which is the
    # exact artifact-assembly failure P8 must rule out.
    c8._prepend_unique_path("ASCEND_CUSTOM_OPP_PATH", (vendor,))
    reference_op, metadata_op, provenance = c8._require_source_matched_official_c8_ops(
        opapi, vendor
    )
    inputs = c8._build_official_inputs(device)
    inputs["metadata"] = c8._official_metadata(metadata_op, inputs)
    indices = c8._call_official(reference_op, inputs)
    torch.npu.synchronize()
    # `score_source_lens` is the exact static source-length companion for this
    # unchanged C8 case.  It is intentionally retained separately because C8
    # only returns Top-K indices at this public ABI.
    source_lens = inputs["actual_seq_lengths_key"].detach().clone()
    print(
        "P8_REAL_C8_BEFORE "
        f"sparse_sha16={_sha16(indices)} lens_sha16={_sha16(source_lens)} "
        f"source_head={provenance.get('source_head', '<missing>')}",
        flush=True,
    )
    return (indices.detach().clone(), source_lens), {
        "op": reference_op,
        "inputs": inputs,
        "provenance": provenance,
    }


def _repeat_official_c8(context: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    # Reuse the reference adapter's exact registered call ABI.  This avoids
    # accidentally turning the before/after proof into two slightly different
    # official C8 invocations as new optional attributes are added upstream.
    import streaming_quant_li_get_target_adapter as c8

    indices = c8._call_official(context["op"], context["inputs"])
    torch.npu.synchronize()
    source_lens = context["inputs"]["actual_seq_lengths_key"].detach().clone()
    print(
        "P8_REAL_C8_AFTER "
        f"sparse_sha16={_sha16(indices)} lens_sha16={_sha16(source_lens)}",
        flush=True,
    )
    return indices.detach().clone(), source_lens


def _assert_stage_a_shape(stage_a_outputs: tuple[torch.Tensor, ...]) -> Any:
    # The P8 runner already verifies Stage A byte-for-byte against its CPU
    # oracle.  Reusing the same source fixture only supplies its mapping/view
    # tensors; it must match the live receipt tensors exactly.
    import test_score_receipt_candidate_executor as stage_a

    fixture = stage_a._fixture()
    seeds, counts, ready_epoch, status = stage_a_outputs
    expected_receipts = fixture.rows * ((fixture.source_capacity + 127) // 128)
    expected = (
        (tuple(seeds.shape), (expected_receipts, 128, 4)),
        (tuple(counts.shape), (expected_receipts,)),
        (tuple(ready_epoch.shape), (expected_receipts,)),
        (tuple(status.shape), (expected_receipts,)),
    )
    if any(got != wanted for got, wanted in expected):
        raise RuntimeError(f"P8 Stage-A/live receipt shape mismatch: {expected}")
    if int(fixture.epoch) != FORWARD_EPOCH:
        raise RuntimeError(
            f"P8 fixture epoch drift: stage_a={fixture.epoch} expected={FORWARD_EPOCH}"
        )
    return fixture


def build_p8_real_fixture(
    device: torch.device, stage_a_outputs: tuple[torch.Tensor, ...]
) -> dict[str, Any]:
    """Build real local Direct-HCCP resources for the P8 graph lifecycle.

    This function only cold-binds runtime state.  It never calls BatchGet
    itself after bind: P8 device phases own PostOnly, CQ poll/Finish, private
    release, and forward terminal handling.
    """
    if device.type != "npu" or device.index not in (None, 0):
        raise RuntimeError(
            "P8 fixture requires physical NPU 7 to be mapped to logical npu:0 "
            "before Python starts"
        )
    device_id = 0
    get_kernel = _required_file("P8_KVAF_BATCH_GET_KERNEL")
    put_kernel = _required_file("P8_KVAF_BATCH_PUT_KERNEL")
    fixture = _assert_stage_a_shape(stage_a_outputs)
    # Stage-A's map spans objects [0,306], requiring three 128-row local DDR
    # entries.  This is exact physical address coverage, not a fake one-entry
    # table that would silently reject candidates above object 127.
    entries = (int(fixture.object_capacity) + ROWS_PER_ENTRY - 1) // ROWS_PER_ENTRY
    runtime = kvaf.NpuRuntime(get_kernel, device_id, QUEUE_COUNT, put_kernel, 1)
    _require_local_streaming_runtime(runtime)

    state: dict[str, Any] = {
        "runtime": runtime,
        "table": None,
        "control_bound": False,
        "handed_to_executor": False,
        "terminal_safe": False,
        "cleaned": False,
        "ddr_payload": _payload(entries),
        "c8_after": None,
        "payload_validated": False,
    }

    def cleanup() -> None:
        """Release only after terminal proof; otherwise preserve diagnostics.

        A failed P8 graph may have a late CQ.  Releasing its registered HBM
        binding would be exactly the lifecycle bug P8 is designed to avoid.
        The per-runner Python process then exits, making process teardown the
        explicit quarantine boundary instead of pretending the queue is safe.
        """
        if state["cleaned"]:
            return
        state["cleaned"] = True
        if state["handed_to_executor"] and not state["terminal_safe"]:
            print(
                "P8_REAL_CLEANUP safe=0 action=process_exit_quarantine "
                "reason=forward_terminal_not_validated",
                flush=True,
            )
            return
        try:
            torch.npu.synchronize()
        except Exception:
            pass
        try:
            if state["control_bound"]:
                runtime.release_streaming_batch_get_control()
        finally:
            state["control_bound"] = False
            try:
                if state["table"] is not None:
                    runtime.release_access_table(state["table"])
            finally:
                state["table"] = None
                if getattr(runtime, "initialized", False):
                    runtime.shutdown()
        print("P8_REAL_CLEANUP safe=1 action=released_after_forward_terminal", flush=True)

    try:
        torch.npu.set_device(device_id)
        c8_before, c8_context = _capture_official_c8(device)

        runtime.initialize(entries * ENTRY_BYTES, ENTRY_BYTES)
        source_payload = state["ddr_payload"].to(device).reshape(-1).contiguous()
        keys = list(range(1, entries + 1))
        runtime.batch_put(
            keys,
            source_payload,
            [index * ENTRY_BYTES for index in range(entries)],
            [ENTRY_BYTES] * entries,
        )
        torch.npu.synchronize()
        state["table"] = runtime.prepare_access_table(keys)
        access = runtime.access_table_view(state["table"], device="npu", capacity=entries)
        torch.npu.synchronize()

        seeds, candidate_counts, candidate_ready_epoch, candidate_status = stage_a_outputs
        forward_control = _npu(
            torch.tensor([FORWARD_EPOCH, 0, PHASE_POST << 32, 0], dtype=torch.int64),
            device,
        )
        expected_identity = _npu(torch.tensor([FORWARD_EPOCH, 0, 0], dtype=torch.int64), device)
        private_source_to_slot = _npu(
            torch.full((fixture.rows, fixture.source_capacity), -1, dtype=torch.int32), device
        )
        private_object_to_slot = _npu(
            torch.full((fixture.pool_size, fixture.object_capacity), -1, dtype=torch.int32), device
        )
        private_ready = _npu(torch.zeros(PRIVATE_ROWS, dtype=torch.int32), device)
        private_slot_cursor = _npu(torch.zeros(1, dtype=torch.int32), device)
        private_output = _npu(
            torch.zeros((PRIVATE_ROWS, PACKED_KV_ROW_BYTES), dtype=torch.uint8), device
        )
        fused_get_control = _npu(torch.zeros(CONTROL_WORDS, dtype=torch.int64), device)
        queue_leases = _npu(torch.zeros(QUEUE_COUNT, dtype=torch.int32), device)
        wave_records = _npu(torch.zeros((QUEUE_COUNT, 32), dtype=torch.int64), device)
        private_records = _npu(torch.zeros((PRIVATE_ROWS, 10), dtype=torch.int64), device)
        request_source_offsets = _npu(torch.zeros((QUEUE_COUNT, REQUEST_CAPACITY), dtype=torch.int64), device)
        request_object_offsets = _npu(torch.zeros((QUEUE_COUNT, REQUEST_CAPACITY), dtype=torch.int64), device)
        request_output_indices = _npu(torch.full((QUEUE_COUNT, REQUEST_CAPACITY), -1, dtype=torch.int32), device)
        request_count = _npu(torch.zeros(QUEUE_COUNT, dtype=torch.int32), device)
        candidate_cursor = _npu(torch.zeros(seeds.size(0), dtype=torch.int32), device)
        candidate_cursor_epoch = _npu(torch.zeros(seeds.size(0), dtype=torch.int64), device)
        # The P8 tiler runs up to min(available AIV, max(receipts, private rows)).
        # 64 rows covers a 48/72-AIV device without a silent stats OOB.
        worker_stats = _npu(torch.zeros((PRIVATE_ROWS, 16), dtype=torch.int64), device)
        rows_per_entry = _npu(torch.tensor([ROWS_PER_ENTRY], dtype=torch.int32), device)

        descriptors = {
            "indices": [torch.zeros(REQUEST_CAPACITY, dtype=torch.int32, device=device)],
            "offsets": [torch.zeros(REQUEST_CAPACITY, dtype=torch.int32, device=device)],
            "masks": [torch.zeros(REQUEST_CAPACITY, dtype=torch.uint8, device=device)],
            "output_indices": [torch.zeros(REQUEST_CAPACITY, dtype=torch.int32, device=device)],
        }
        stream = torch.npu.current_stream()
        runtime.bind_streaming_batch_get_control(
            [0], [private_output], [fused_get_control], queue_leases, QUEUE_COUNT, stream
        )
        state["control_bound"] = True
        runtime.configure_streaming_batch_get_queue_templates(
            [0], [0], [access.addresses], [access.valid],
            descriptors["indices"], descriptors["offsets"], descriptors["masks"],
            descriptors["output_indices"], stream,
        )
        words = _assert_cold_control(fused_get_control, private_output, queue_leases)
        print(
            "P8_REAL_BIND "
            f"entries={entries} queue_count={QUEUE_COUNT} private_rows={PRIVATE_ROWS} "
            f"control_magic={words[0]} templates={runtime.streaming_pool_snapshot().get('template_configure_count')} "
            f"diagnostic={runtime.streaming_batch_get_control_diagnostic()}",
            flush=True,
        )

        executor_inputs: tuple[torch.Tensor, ...] = (
            seeds,
            candidate_counts,
            candidate_ready_epoch,
            candidate_status,
            forward_control,
            _npu(fixture.score_source_lens, device),
            _npu(torch.tensor([1], dtype=torch.int32), device),
            _npu(fixture.req_pool_entries, device),
            _npu(fixture.kv_object_map, device),
            _npu(fixture.kv_object_lens, device),
            _npu(fixture.kv_object_slots_snapshot, device),
            private_source_to_slot,
            private_object_to_slot,
            private_ready,
            private_slot_cursor,
            private_output,
            fused_get_control,
            wave_records,
            private_records,
            request_source_offsets,
            request_object_offsets,
            request_output_indices,
            request_count,
            candidate_cursor,
            candidate_cursor_epoch,
            rows_per_entry,
            worker_stats,
        )
        state["handed_to_executor"] = True

        result: dict[str, Any] = {
            "executor_inputs": executor_inputs,
            "expected_identity": expected_identity,
            "payload_validated": False,
            "validated_cases": (),
            # Bounded graph-node CQ polls: no host Get wait and no device spin.
            "drain_attempts": 16,
            "c8_baseline_before": c8_before,
            "cleanup": cleanup,
        }

        def validate_terminal(inputs: tuple[torch.Tensor, ...]) -> None:
            torch.npu.synchronize()
            stats = inputs[26].detach().cpu()
            stat_sums = [int(stats[:, column].sum()) for column in range(stats.size(1))]
            leases_cpu = queue_leases.detach().cpu()
            source_map = inputs[11].detach().cpu()
            object_map = inputs[12].detach().cpu()
            ready_cpu = inputs[13].detach().cpu()
            control_cpu = inputs[4].detach().cpu()
            wave_cpu = inputs[17].detach().cpu()
            private_records_cpu = inputs[18].detach().cpu()
            ledger_source = inputs[19].detach().cpu()
            ledger_object = inputs[20].detach().cpu()
            ledger_output = inputs[21].detach().cpu()
            output_cpu = inputs[15].detach().cpu()

            required = {
                "post": stat_sums[STAT_POST_COMMITTED],
                "cq_poll": stat_sums[STAT_CQ_POLL],
                "cq_terminal": stat_sums[STAT_CQ_TERMINAL],
                "queue_release": stat_sums[STAT_QUEUE_RELEASED],
                "private_release": stat_sums[STAT_PRIVATE_RELEASED],
            }
            if any(value <= 0 for value in required.values()):
                raise AssertionError(
                    "P8 real local Direct-HCCP did not close PostOnly->TryRetire->Finish "
                    f"required={required} stats={stat_sums} leases={leases_cpu.tolist()}"
                )
            if any(stat_sums[column] != 0 for column in (
                STAT_CONTROL_REJECTED, STAT_QUARANTINED, STAT_ABORT
            )):
                raise AssertionError(
                    "P8 real Direct-HCCP entered fail-closed transport path: "
                    f"stats={stat_sums} control={control_cpu.tolist()}"
                )
            if int(control_cpu[3]) & FORWARD_ABORT_MASK:
                raise AssertionError(
                    "P8 real Direct-HCCP published abort/private-disable flag: "
                    f"control={control_cpu.tolist()} stats={stat_sums}"
                )
            if int(torch.count_nonzero(leases_cpu)) != 0:
                raise AssertionError(f"P8 queue lease leaked after terminal CQ: {leases_cpu.tolist()}")
            if int(torch.count_nonzero(source_map >= 0)) != 0 or int(torch.count_nonzero(object_map >= 0)) != 0:
                raise AssertionError(
                    "P8 QSFA-terminal did not clear transaction-private maps: "
                    f"source_mapped={int(torch.count_nonzero(source_map >= 0))} "
                    f"object_mapped={int(torch.count_nonzero(object_map >= 0))}"
                )
            if int(torch.count_nonzero(ready_cpu)) != 0:
                raise AssertionError(f"P8 QSFA-terminal did not free private readiness: {ready_cpu.tolist()}")
            # `state` is int32 lane 3 of the fixed 80-byte private record
            # (magic:int64, version:uint32, state:uint32), not int64 word 3.
            # We do not require every record's padding to be zero after DMA.
            private_state = private_records_cpu.view(torch.int32).reshape(PRIVATE_ROWS, 20)[:, 3]
            if int(torch.count_nonzero(private_state)) != 0:
                raise AssertionError("P8 private record remained non-free after QSFA terminal")
            wave_state = wave_cpu.view(torch.int32).reshape(QUEUE_COUNT, 64)[:, 3]
            if int(torch.count_nonzero(wave_state)) != 0:
                raise AssertionError(f"P8 wave remained non-free after terminal: {wave_state.tolist()}")

            observed: list[tuple[int, int, int, int]] = []
            for request in range(REQUEST_CAPACITY):
                output_index = int(ledger_output[0, request])
                if output_index < 0:
                    continue
                source_offset = int(ledger_source[0, request])
                object_offset = int(ledger_object[0, request])
                if not (0 <= output_index < PRIVATE_ROWS and source_offset >= 0 and object_offset >= 0):
                    raise AssertionError(
                        "P8 descriptor ledger malformed after terminal: "
                        f"request={request} source={source_offset} object={object_offset} output={output_index}"
                    )
                row, source = divmod(source_offset, int(fixture.source_capacity))
                pool, obj = divmod(object_offset, int(fixture.object_capacity))
                if (
                    not (0 <= row < fixture.rows and 0 <= pool < fixture.pool_size) or
                    int(fixture.kv_object_map[pool, source]) != obj
                ):
                    raise AssertionError(
                        "P8 descriptor ledger does not preserve Stage-A S->O mapping: "
                        f"request={request} row={row} source={source} pool={pool} object={obj}"
                    )
                observed.append((request, output_index, source_offset, object_offset))
            if not observed:
                raise AssertionError(
                    "P8 no persisted Direct-HCCP descriptor ledger after post; "
                    f"stats={stat_sums} wave_digest={_sha16(wave_cpu)}"
                )

            mismatch: tuple[int, int, int, int, int, int] | None = None
            for request, output_index, source_offset, object_offset in observed:
                obj = object_offset % int(fixture.object_capacity)
                expected = state["ddr_payload"][obj // ROWS_PER_ENTRY].reshape(
                    ROWS_PER_ENTRY, PACKED_KV_ROW_BYTES
                )[obj % ROWS_PER_ENTRY]
                actual = output_cpu[output_index]
                if not torch.equal(actual, expected):
                    nonzero = (actual != expected).nonzero(as_tuple=False)
                    first = int(nonzero[0].item()) if nonzero.numel() else -1
                    mismatch = (
                        request, output_index, source_offset, object_offset,
                        first, int(expected[first]) if first >= 0 else -1,
                    )
                    actual_value = int(actual[first]) if first >= 0 else -1
                    break
            if mismatch is not None:
                request, output_index, source_offset, object_offset, first, expected_byte = mismatch
                print(
                    "P8_REAL_PAYLOAD status=mismatch bandwidth_reported=0 "
                    f"request={request} output={output_index} source_offset={source_offset} "
                    f"object_offset={object_offset} first_byte={first} expected={expected_byte} "
                    f"actual={actual_value} output_sha16={_sha16(output_cpu[output_index])} "
                    f"ddr_sha16={_sha16(state['ddr_payload'])} control_sha16={_sha16(inputs[16])}",
                    flush=True,
                )
                raise AssertionError("P8 local Direct-HCCP payload byte comparison failed")
            result["payload_validated"] = True
            state["payload_validated"] = True
            result["validated_cases"] = ("happy_post_cq_qsfa_release",)
            print(
                "P8_REAL_PAYLOAD status=validated bandwidth_reported=0 "
                f"requests={len(observed)} ddr_sha16={_sha16(state['ddr_payload'])} "
                f"output_sha16={_sha16(output_cpu)}",
                flush=True,
            )
            result["c8_baseline_after"] = _repeat_official_c8(c8_context)
            state["c8_after"] = result["c8_baseline_after"]
            state["terminal_safe"] = True

        result["validate_terminal"] = validate_terminal
        return result
    except Exception:
        cleanup()
        raise
