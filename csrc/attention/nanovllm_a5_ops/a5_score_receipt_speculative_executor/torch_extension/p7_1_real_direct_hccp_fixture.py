"""Real local-KVAF fixture for the standalone P7.1 executor.

This module owns only a cold Direct-HCCP transaction setup.  It does **not**
import C8, P7.0, a vLLM product path, or a host submit worker.  The returned
CandidateSeed receipt is consumed exactly once by
``test_score_receipt_speculative_executor.py``.

The fixture deliberately verifies the device-side queue lifecycle, not raw
payload bytes.  Existing local Direct-HCCP payload validation remains a
separate known-failed area; reporting bandwidth or payload correctness here
would overclaim what this vertical slice proves.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch_npu  # noqa: F401  # Register ``npu`` device.

import kvaf


PACKED_KV_ROW_BYTES = 656
ROWS_PER_ENTRY = 128
ENTRY_BYTES = PACKED_KV_ROW_BYTES * ROWS_PER_ENTRY
QUEUE_COUNT = 1
PRIVATE_ARENA_CAPACITY = 64
RECEIPT_CAPACITY = 128
STREAMING_EPOCH = 41
CANDIDATE_STATUS_READY = 1
PRIVATE_READY = 2
QUEUE_FREE = 0


def _required_path(env_name: str) -> str:
    value = os.environ.get(env_name, "")
    if not value or not Path(value).is_file():
        raise RuntimeError(
            f"P7.1 real fixture requires {env_name} to name a built KVAF object; "
            f"got {value!r}"
        )
    return value


def _require_streaming_api(runtime: Any) -> None:
    required = (
        "bind_streaming_batch_get_control",
        "configure_streaming_batch_get_queue_templates",
        "release_streaming_batch_get_control",
        "streaming_batch_get_control_diagnostic",
        "streaming_pool_snapshot",
        "release_access_table",
        "shutdown",
    )
    missing = [name for name in required if not callable(getattr(runtime, name, None))]
    if missing:
        raise RuntimeError(
            "P7.1 requires the local native-enqueue KVAF streaming ABI; "
            f"missing={','.join(missing)}. Do not run this with a remote CPU-service "
            "or legacy BatchGet build."
        )
    if callable(getattr(runtime, "link_to", None)):
        raise RuntimeError(
            "P7.1 real fixture is local Direct-HCCP only, but imported KVAF exposes "
            "the remote CPU-service ABI. Select the native-enqueue KVAF build."
        )


def _u64(value: int) -> int:
    return int(value) & ((1 << 64) - 1)


def _assert_cold_control(
    control: torch.Tensor,
    output: torch.Tensor,
    leases: torch.Tensor,
) -> list[int]:
    torch.npu.synchronize()
    words = [_u64(value) for value in control.cpu().tolist()]
    # This is intentionally structural rather than importing KVAF C++
    # constants into Python: it proves pointers/counts/stride came from the
    # real cold binding, not from a fabricated tensor.
    expected_stride = PRIVATE_ARENA_CAPACITY * PACKED_KV_ROW_BYTES
    checks = {
        "magic": words[0] != 0,
        "version": words[1] != 0,
        "channel": words[2] != 0,
        "queue_state": words[3] != 0,
        "core_status": words[4] != 0,
        "output_alias": words[5] == _u64(output.data_ptr()),
        "lease_alias": words[6] == _u64(leases.data_ptr()),
        "scratch": words[7] != 0,
        "queue_count": words[8] == QUEUE_COUNT,
        "output_rows": words[9] == QUEUE_COUNT * PRIVATE_ARENA_CAPACITY,
        "stride": words[10] == expected_stride,
        "block_bytes": words[13] == PACKED_KV_ROW_BYTES,
        "templates": words[14] != 0,
        "template_count": words[15] == QUEUE_COUNT,
    }
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        raise RuntimeError(
            "P7.1 KVAF cold binding did not produce a usable streaming control: "
            f"bad={bad} words={words} output_ptr=0x{output.data_ptr():x} "
            f"lease_ptr=0x{leases.data_ptr():x}"
        )
    return words


def _build_inputs(device: torch.device, control: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Build the exact 31-input P7.1 ABI around one real candidate wave."""

    npu = lambda value: value.to(device).contiguous()
    seeds = torch.full(
        (1, RECEIPT_CAPACITY, 4), -1, dtype=torch.int32, device=device
    )
    # CandidateSeed ABI is [row, S, O, score_bits].  P7.1 accepts only
    # MAPPED_OBJECT domain, therefore O=kv_object_map[P,S]=0 here.
    seeds[0, 0] = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device=device)
    candidate_counts = npu(torch.tensor([1], dtype=torch.int32))
    candidate_ready_epoch = npu(torch.tensor([STREAMING_EPOCH], dtype=torch.int64))
    candidate_status = npu(torch.tensor([CANDIDATE_STATUS_READY], dtype=torch.int32))
    streaming_epoch = npu(torch.tensor([STREAMING_EPOCH], dtype=torch.int64))
    source_lens = npu(torch.tensor([ROWS_PER_ENTRY], dtype=torch.int32))
    kv_object_address_domain = npu(torch.tensor([1], dtype=torch.int32))
    req_pool_entries = npu(torch.tensor([0], dtype=torch.int32))
    kv_object_map = npu(torch.arange(ROWS_PER_ENTRY, dtype=torch.int32).reshape(1, -1))
    kv_object_lens = npu(torch.tensor([ROWS_PER_ENTRY], dtype=torch.int32))
    kv_object_slots_snapshot = npu(torch.full((1, ROWS_PER_ENTRY), -1, dtype=torch.int32))
    private_source_to_slot = npu(torch.full((1, ROWS_PER_ENTRY), -1, dtype=torch.int32))
    private_object_to_slot = npu(torch.full((1, ROWS_PER_ENTRY), -1, dtype=torch.int32))
    private_ready_table = npu(torch.zeros((QUEUE_COUNT, PRIVATE_ARENA_CAPACITY), dtype=torch.int32))
    private_arena_head = npu(torch.zeros(QUEUE_COUNT, dtype=torch.int32))
    private_packed_output = npu(
        torch.zeros(
            (QUEUE_COUNT * PRIVATE_ARENA_CAPACITY, PACKED_KV_ROW_BYTES),
            dtype=torch.uint8,
        )
    )
    queue_leases = npu(torch.zeros(QUEUE_COUNT, dtype=torch.int32))
    queue_epoch = npu(torch.zeros(QUEUE_COUNT, dtype=torch.int64))
    queue_generation = npu(torch.zeros(QUEUE_COUNT, dtype=torch.int64))
    queue_post_sequence = npu(torch.zeros(QUEUE_COUNT, dtype=torch.int64))
    queue_state = npu(torch.zeros(QUEUE_COUNT, dtype=torch.int32))
    queue_quarantine = npu(torch.zeros(QUEUE_COUNT, dtype=torch.int32))
    queue_request_source_offsets = npu(torch.zeros((QUEUE_COUNT, 64), dtype=torch.int64))
    queue_request_object_offsets = npu(torch.zeros((QUEUE_COUNT, 64), dtype=torch.int64))
    queue_request_output_indices = npu(torch.full((QUEUE_COUNT, 64), -1, dtype=torch.int32))
    queue_request_count = npu(torch.zeros(QUEUE_COUNT, dtype=torch.int32))
    private_ready_epoch = npu(torch.zeros((QUEUE_COUNT, PRIVATE_ARENA_CAPACITY), dtype=torch.int64))
    private_ready_generation = npu(torch.zeros((QUEUE_COUNT, PRIVATE_ARENA_CAPACITY), dtype=torch.int64))
    worker_stats = npu(torch.zeros((1, 16), dtype=torch.int64))
    rows_per_entry = npu(torch.tensor([ROWS_PER_ENTRY], dtype=torch.int32))
    # The caller replaces the output/control/lease allocations below with the
    # exact graph-visible tensors passed to KVAF during cold bind.
    return (
        seeds,
        candidate_counts,
        candidate_ready_epoch,
        candidate_status,
        streaming_epoch,
        source_lens,
        kv_object_address_domain,
        req_pool_entries,
        kv_object_map,
        kv_object_lens,
        kv_object_slots_snapshot,
        private_source_to_slot,
        private_object_to_slot,
        private_ready_table,
        private_arena_head,
        private_packed_output,
        control,
        queue_leases,
        queue_epoch,
        queue_generation,
        queue_post_sequence,
        queue_state,
        queue_quarantine,
        queue_request_source_offsets,
        queue_request_object_offsets,
        queue_request_output_indices,
        queue_request_count,
        private_ready_epoch,
        private_ready_generation,
        worker_stats,
        rows_per_entry,
    )


def build_p7_1_fixture(device: torch.device) -> dict[str, Any]:
    """Cold-bind a real local KVAF Direct-HCCP queue then return P7.1 inputs."""

    if device.type != "npu" or device.index not in (0, None):
        raise RuntimeError(
            "P7.1 runner must map the requested physical NPU to logical npu:0 "
            "with ASCEND_RT_VISIBLE_DEVICES before Python starts."
        )
    device_id = 0 if device.index is None else device.index
    get_kernel = _required_path("P7_KVAF_BATCH_GET_KERNEL")
    put_kernel = _required_path("P7_KVAF_BATCH_PUT_KERNEL")
    runtime = kvaf.NpuRuntime(get_kernel, device_id, QUEUE_COUNT, put_kernel, 1)
    _require_streaming_api(runtime)

    table = None
    control_bound = False
    cleaned = False

    def cleanup() -> None:
        nonlocal table, control_bound, cleaned
        if cleaned:
            return
        cleaned = True
        # The P7.1 test synchronizes the op before `validate`/cleanup.  This
        # is a cold terminal release, not a per-wave host synchronization.
        try:
            torch.npu.synchronize()
        except Exception:
            pass
        try:
            if control_bound:
                runtime.release_streaming_batch_get_control()
        finally:
            control_bound = False
            try:
                if table is not None:
                    runtime.release_access_table(table)
            finally:
                table = None
                if runtime.initialized:
                    runtime.shutdown()

    try:
        torch.npu.set_device(device_id)
        runtime.initialize(ENTRY_BYTES, ENTRY_BYTES)
        # One deterministic full 128-row KV object is sufficient to make
        # descriptor index=0/offset=0 address a real DDR entry.
        source = (
            torch.arange(ENTRY_BYTES, dtype=torch.int64)
            .remainder(251)
            .to(torch.uint8)
            .to(device)
            .contiguous()
        )
        runtime.batch_put([1], source, [0], [ENTRY_BYTES])
        torch.npu.synchronize()
        table = runtime.prepare_access_table([1])
        access = runtime.access_table_view(table, device="npu", capacity=1)
        torch.npu.synchronize()

        # Allocate the graph-visible P7 tensors first, then bind the exact
        # output/control/lease identities to KVAF.  They are never recreated.
        provisional_control = torch.zeros(16, dtype=torch.int64, device=device)
        inputs = list(_build_inputs(device, provisional_control))
        private_output = inputs[15]
        control = provisional_control
        queue_leases = inputs[17]

        descriptor_indices = [torch.zeros(64, dtype=torch.int32, device=device)]
        descriptor_offsets = [torch.zeros(64, dtype=torch.int32, device=device)]
        descriptor_masks = [torch.zeros(64, dtype=torch.uint8, device=device)]
        descriptor_output_indices = [torch.zeros(64, dtype=torch.int32, device=device)]
        producer_stream = torch.npu.current_stream()
        runtime.bind_streaming_batch_get_control(
            [0], [private_output], [control], queue_leases, QUEUE_COUNT, producer_stream
        )
        control_bound = True
        runtime.configure_streaming_batch_get_queue_templates(
            [0],
            [0],
            [access.addresses],
            [access.valid],
            descriptor_indices,
            descriptor_offsets,
            descriptor_masks,
            descriptor_output_indices,
            producer_stream,
        )
        words = _assert_cold_control(control, private_output, queue_leases)
        snapshot = runtime.streaming_pool_snapshot()
        diagnostic = runtime.streaming_batch_get_control_diagnostic()
        print(
            "P7_1_REAL_BIND "
            f"queue_count={QUEUE_COUNT} private_rows={private_output.size(0)} "
            f"control_magic={words[0]} control_version={words[1]} "
            f"templates={snapshot.get('template_configure_count')} "
            f"diagnostic={diagnostic}",
            flush=True,
        )

        def validate(outputs: tuple[torch.Tensor, ...]) -> None:
            # Lifecycle success is the P7.1 proof.  Do not inspect H2D payload
            # bytes or report bandwidth while payload validation remains known
            # failed in the local Direct-HCCP environment.
            source_map = outputs[0].detach().cpu()
            object_map = outputs[1].detach().cpu()
            ready = outputs[2].detach().cpu()
            leases = outputs[5].detach().cpu()
            state = outputs[9].detach().cpu()
            quarantine = outputs[10].detach().cpu()
            if int(source_map[0, 0]) != 0 or int(object_map[0, 0]) != 0:
                raise AssertionError(
                    "real P7.1 Direct-HCCP wave did not publish source/object private slot 0"
                )
            if int(ready[0, 0]) != PRIVATE_READY:
                raise AssertionError(
                    f"real P7.1 Direct-HCCP CQ did not mark private slot ready: {ready.tolist()}"
                )
            if int(leases[0]) != 0 or int(state[0]) != QUEUE_FREE or int(quarantine[0]) != 0:
                raise AssertionError(
                    "real P7.1 queue was not terminally released: "
                    f"leases={leases.tolist()} state={state.tolist()} "
                    f"quarantine={quarantine.tolist()}"
                )
            print(
                "P7_1_REAL_LIFECYCLE "
                "private_ready=1 queue_released=1 payload=known_failed_not_checked "
                "bandwidth_reported=0",
                flush=True,
            )

        return {
            "inputs": tuple(inputs),
            "minimum": {"lease": 1, "post": 1, "finish": 1, "release": 1},
            "validate": validate,
            "cleanup": cleanup,
        }
    except Exception:
        cleanup()
        raise
