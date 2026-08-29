#!/usr/bin/env python3
"""Strict live C8-publication -> P5 ExactScan/Post/CQ integration gate.

This is intentionally an integration *correctness* gate, not an overlap or
bandwidth benchmark.  It proves the real data edge required by the target
design without manufacturing a second score surface:

  official C8 inputs
    -> source-derived C8 score workspace + per-tile receipts
    -> ExactScoreScan / CandidateCompact / Direct-HCCP Post / CQ drain
    -> layer-private READY KV payloads.

The source-derived C8 launch is first compared bitwise with the unmodified
official C8.  Only then are the exact same ``score_workspace``,
``tile_receipts`` and ``streaming_epoch`` allocations passed to P5.  The
candidate threshold is zero for this gate, so it is intentionally a safe
overfetch superset: every valid C8 score is scanned and every final C8 Top-K
source is required to be covered by a private READY slot.  The test never
claims that this candidate superset *is* the final Top-K.

Because the current public P3 publisher returns only after its C8 launch has
completed, this gate is sequential C8 -> P5.  It validates the true ABI and
the real Post/CQ lifecycle before the next target step moves the consumer to
the C8 tile publication seam for measured overlap.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


P5_OPP_ENV = "NANOVLLM_A5_EXACT_SCAN_OPP"


def _preload_p5_opp_before_acl() -> tuple[Path, Path]:
    """Make P5's package discoverable before the process's first ACLNN call.

    The P3 adapter invokes the official C8 ACLNN operator before this test
    imports the P5 bridge.  CANN snapshots custom OPP package metadata at that
    first ACLNN call, so merely loading the P5 bridge later is too late for
    ``A5ExactScanSpeculativeGet``.  This function deliberately registers only
    the *package* now.  It does not set ``NANOVLLM_CUST_OPAPI_LIB``: each
    isolated bridge selects its own generated OPAPI DSO immediately before its
    own call.
    """
    root_value = os.getenv(P5_OPP_ENV)
    if not root_value:
        raise RuntimeError(
            f"{P5_OPP_ENV} must name the built P5 OPP root before this "
            "live C8 gate starts"
        )
    opp_root = Path(root_value).expanduser().resolve()
    candidates = {
        candidate.resolve()
        for candidate in (
            opp_root / "op_api/lib/libcust_opapi.so",
            *opp_root.glob("vendors/*/op_api/lib/libcust_opapi.so"),
        )
        if candidate.is_file()
    }
    if len(candidates) != 1:
        raise RuntimeError(
            "P5 live C8 gate requires exactly one P5 custom OPAPI in "
            f"{opp_root}; found={[str(path) for path in sorted(candidates)]}"
        )
    opapi = next(iter(candidates))
    vendor = opapi.parents[2]
    vendor_text = str(vendor)
    existing = [
        path
        for path in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(":")
        if path and path != vendor_text
    ]
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join([vendor_text, *existing])
    print(
        "P5_LIVE_C8_OPP_PRELOAD "
        f"p5_vendor={vendor} p5_opapi={opapi} "
        f"custom_opp_path={os.environ['ASCEND_CUSTOM_OPP_PATH']}",
        flush=True,
    )
    return opapi, vendor


# This must run before importing torch_npu or the P3 adapter.  The latter
# reaches the first ACLNN call, where CANN discovers all custom OPP packages.
_P5_OPAPI, _P5_VENDOR = _preload_p5_opp_before_acl()

import torch
import torch_npu  # noqa: F401  # Registers the NPU backend.

import streaming_quant_li_get_target_adapter as c8_target


GET_LANE_COUNT = 2
PRODUCER_COUNT = 62
ROWS_PER_ENTRY = c8_target.BLOCK_SIZE
PACKED_ENTRY_BYTES = c8_target.PACKED_KV_ROW_BYTES
ENTRY_BYTES = ROWS_PER_ENTRY * PACKED_ENTRY_BYTES
PIPELINE_MODE_PARALLEL = 0
PRIVATE_READY_FREE = 0
PRIVATE_READY_READY = 2
RECEIPT_WORDS = c8_target.TILE_RECEIPT_WORDS
FUSED_CONTROL_MAGIC = 0x4B56414646555345
FUSED_CONTROL_VERSION = 1
RING_SLOTS = 2
RECORDS_PER_RING_SLOT = 32
CACHE_LINE_WORDS = 8
RING_SLOT_RECORD_WORD = 3 * CACHE_LINE_WORDS
RING_WORDS_PER_SLOT = RING_SLOT_RECORD_WORD + RECORDS_PER_RING_SLOT * 3
PIPELINE_WORDS_PER_RING = RING_SLOTS * RING_WORDS_PER_SLOT + CACHE_LINE_WORDS
PIPELINE_RING_COUNT = PRODUCER_COUNT * GET_LANE_COUNT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="npu:0",
        help=(
            "logical device.  For physical NPU 7 use a fresh process with "
            "ASCEND_RT_VISIBLE_DEVICES=7 and keep this as npu:0"
        ),
    )
    parser.add_argument("--batch-get-kernel", required=True)
    parser.add_argument("--batch-put-kernel", required=True)
    return parser.parse_args()


def _require_mapped_npu0(args: argparse.Namespace) -> torch.device:
    if args.device != "npu:0":
        raise ValueError(
            "live P5 gate is logical-npu:0 only; map the requested physical "
            "card before process start with ASCEND_RT_VISIBLE_DEVICES"
        )
    if os.getenv("P3_NPU_ID") != "0" or os.getenv("P3_PHYSICAL_NPU_ID") is None:
        raise RuntimeError(
            "set P3_NPU_ID=0 and P3_PHYSICAL_NPU_ID=<physical card>; "
            "the P3 publisher and P5 consumer must share one mapped NPU"
        )
    physical = os.getenv("P3_PHYSICAL_NPU_ID")
    mapped = os.getenv("ASCEND_RT_VISIBLE_DEVICES")
    if mapped != physical:
        raise RuntimeError(
            "strict physical proof requires ASCEND_RT_VISIBLE_DEVICES to name "
            f"exactly P3_PHYSICAL_NPU_ID; got mapping={mapped!r} physical={physical!r}"
        )
    device = torch.device(args.device)
    torch.npu.set_device(device)
    if torch.npu.current_device() != 0:
        raise RuntimeError("failed to select the mapped logical npu:0")
    return device


def _ptr(tensor: torch.Tensor) -> int:
    return int(tensor.data_ptr())


def _require_c8_strict_gate(result: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Validate C8 math before allocating or posting any KVAF request."""
    torch.testing.assert_close(result["official_indices"], result["derived_indices"])
    torch.testing.assert_close(
        result["expected_score_source_lens"], result["derived_score_source_lens"]
    )
    publication = result.get("publication_tensors")
    aliases = result.get("publication_aliases")
    summary = result.get("publication_summary")
    if not isinstance(publication, dict) or not isinstance(aliases, dict) or not isinstance(summary, dict):
        raise AssertionError("P3 target adapter did not expose its publication ABI")
    if summary.get("mode") != "publisher":
        raise AssertionError(f"P3 publisher mode is required, got {summary.get('mode')!r}")
    score_workspace = publication.get("score_workspace")
    tile_receipts = publication.get("tile_receipts")
    streaming_epoch = publication.get("streaming_epoch")
    if not all(isinstance(value, torch.Tensor) for value in (
        score_workspace, tile_receipts, streaming_epoch
    )):
        raise AssertionError("P3 publisher did not return all three device tensors")
    assert isinstance(score_workspace, torch.Tensor)
    assert isinstance(tile_receipts, torch.Tensor)
    assert isinstance(streaming_epoch, torch.Tensor)
    if score_workspace.device != device or tile_receipts.device != device or streaming_epoch.device != device:
        raise AssertionError("P3 publisher tensors are not on the P5 target NPU")
    for name, tensor in (("score_workspace", score_workspace), ("tile_receipts", tile_receipts)):
        pair = aliases.get(name)
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise AssertionError(f"P3 publisher has no alias pair for {name}")
        if _ptr(pair[0]) != _ptr(pair[1]) or _ptr(pair[0]) != _ptr(tensor):
            raise AssertionError(f"P3 publisher lost the graph-visible {name} allocation")
    if score_workspace.dtype != torch.int32 or tile_receipts.dtype != torch.int64:
        raise AssertionError("P3 publication dtype ABI changed")
    if tile_receipts.ndim != 2 or tile_receipts.shape[1] != RECEIPT_WORDS:
        raise AssertionError(f"unexpected P3 receipt shape={tuple(tile_receipts.shape)}")
    expected_ready = int(summary.get("receipt_expected_active", -1))
    actual_ready = int(summary.get("receipt_ready_count", -1))
    if actual_ready != expected_ready or actual_ready <= 0:
        raise AssertionError(
            "P3 receipt publication is incomplete: "
            f"ready={actual_ready} expected={expected_ready}"
        )
    print(
        "P5_LIVE_C8_STRICT_GATE_PASS "
        "official_c8_topk_bitwise=1 score_source_lens_bitwise=1 "
        f"publication_score_ptr=0x{_ptr(score_workspace):x} "
        f"publication_receipt_ptr=0x{_ptr(tile_receipts):x} "
        f"publication_epoch_ptr=0x{_ptr(streaming_epoch):x} "
        f"receipt_ready={actual_ready}",
        flush=True,
    )
    return {
        "score_workspace": score_workspace,
        "tile_receipts": tile_receipts,
        "streaming_epoch": streaming_epoch,
        # This is scheduler-owned graph-static metadata.  The equality above
        # validates the P3 terminal output, but P5 must not depend on that
        # terminal output to discover an active score tile.
        "score_source_lens": result["expected_score_source_lens"],
    }


def _make_full_object_payload(source_capacity: int) -> torch.Tensor:
    """Return one deterministic packed KV row for every raw source object."""
    rows = torch.arange(source_capacity, dtype=torch.int64).reshape(-1, 1)
    bytes_ = torch.arange(PACKED_ENTRY_BYTES, dtype=torch.int64).reshape(1, -1)
    return ((rows * 29 + bytes_ * 17 + (rows >> 3)) & 0xFF).to(torch.uint8)


def _make_p5_inputs(
    publication: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Build only scheduler/private state; score/receipt tensors are borrowed."""
    source_lens = publication["score_source_lens"]
    if source_lens.ndim != 1:
        raise AssertionError("score_source_lens must be a vector")
    rows = int(source_lens.numel())
    source_capacity = c8_target.SOURCE_TOKENS
    if rows != c8_target.BATCH_SIZE:
        raise AssertionError(f"unexpected C8 row count={rows}")
    active_lens = [int(value) for value in source_lens.cpu().tolist()]
    if any(value < 0 or value > source_capacity for value in active_lens):
        raise AssertionError(f"invalid C8 source lengths={active_lens}")
    # Threshold 0 intentionally creates an overfetch superset.  It guarantees
    # every final C8 Top-K source is covered while leaving final Top-K authority
    # with the unmodified C8 result.
    total_posts = sum(active_lens)
    if total_posts % GET_LANE_COUNT:
        raise AssertionError("fixture must split evenly between the two Get lanes")
    private_arena_capacity = total_posts // GET_LANE_COUNT
    object_map = torch.arange(
        source_capacity, dtype=torch.int32, device=device
    ).repeat(rows, 1).contiguous()
    inputs = {
        "score_workspace": publication["score_workspace"],
        "tile_receipts": publication["tile_receipts"],
        "streaming_epoch": publication["streaming_epoch"],
        "score_source_lens": source_lens,
        "resident_threshold": torch.zeros(rows, dtype=torch.int32, device=device),
        "resident_threshold_valid": torch.ones(rows, dtype=torch.int32, device=device),
        "kv_object_slots_snapshot": torch.full(
            (rows, source_capacity), -1, dtype=torch.int32, device=device
        ),
        "req_pool_entries": torch.arange(rows, dtype=torch.int32, device=device),
        "kv_object_map": object_map,
        "kv_object_lens": source_lens.clone(),
        "private_source_to_slot": torch.full(
            (rows, source_capacity), -1, dtype=torch.int32, device=device
        ),
        "private_object_to_slot": torch.full(
            (rows, source_capacity), -1, dtype=torch.int32, device=device
        ),
        "private_ready_table": torch.full(
            (GET_LANE_COUNT, private_arena_capacity),
            PRIVATE_READY_FREE,
            dtype=torch.int32,
            device=device,
        ),
        "private_arena_head": torch.zeros(GET_LANE_COUNT, dtype=torch.int32, device=device),
        "private_packed_output": torch.empty(
            (GET_LANE_COUNT * private_arena_capacity, PACKED_ENTRY_BYTES),
            dtype=torch.uint8,
            device=device,
        ),
        "fused_get_control": torch.zeros(16, dtype=torch.int64, device=device),
        "pipeline_state": torch.zeros(
            (PIPELINE_RING_COUNT, PIPELINE_WORDS_PER_RING),
            dtype=torch.int64,
            device=device,
        ),
        "execution_mode": torch.tensor(
            [PIPELINE_MODE_PARALLEL], dtype=torch.int64, device=device
        ),
    }
    return inputs, source_capacity // ROWS_PER_ENTRY, private_arena_capacity


def _require_fused_control(control: torch.Tensor, output: torch.Tensor) -> None:
    words = [int(value) for value in control.cpu().tolist()]
    required = (
        len(words) == 16
        and words[0] == FUSED_CONTROL_MAGIC
        and words[1] == FUSED_CONTROL_VERSION
        and words[2] != 0
        and words[3] != 0
        and words[4] != 0
        and words[5] == output.data_ptr()
        and words[6] == output.shape[0]
        and words[7] == GET_LANE_COUNT
        and words[8] == 1
        and words[9] == PACKED_ENTRY_BYTES
    )
    if not required:
        raise AssertionError(f"invalid P5 fused Get control words={words}")


def _call_p5(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return torch.ops.a5_exact_scan_get.exact_scan_speculative_get(
        inputs["score_workspace"],
        inputs["tile_receipts"],
        inputs["streaming_epoch"],
        inputs["score_source_lens"],
        inputs["resident_threshold"],
        inputs["resident_threshold_valid"],
        inputs["kv_object_slots_snapshot"],
        inputs["req_pool_entries"],
        inputs["kv_object_map"],
        inputs["kv_object_lens"],
        inputs["private_source_to_slot"],
        inputs["private_object_to_slot"],
        inputs["private_ready_table"],
        inputs["private_arena_head"],
        inputs["private_packed_output"],
        inputs["address_table"],
        inputs["valid_table"],
        inputs["fused_get_control"],
        inputs["pipeline_state"],
        inputs["execution_mode"],
    )


def _assert_live_oracle(
    result: dict[str, Any],
    inputs: dict[str, torch.Tensor],
    outputs: tuple[torch.Tensor, ...],
    source_cpu: torch.Tensor,
    private_arena_capacity: int,
    publication_before: dict[str, torch.Tensor],
) -> None:
    if len(outputs) != 17:
        raise AssertionError(f"P5 returned {len(outputs)} outputs, expected 17")
    for name, output in zip(
        (
            "private_source_to_slot",
            "private_object_to_slot",
            "private_ready_table",
            "private_arena_head",
            "private_packed_output",
        ),
        outputs[:5],
        strict=True,
    ):
        if _ptr(inputs[name]) != _ptr(output):
            raise AssertionError(f"P5 lost mutable transaction alias {name}")
    if _ptr(inputs["pipeline_state"]) != _ptr(outputs[16]):
        raise AssertionError("P5 lost pipeline_state transaction alias")

    private_source = outputs[0].cpu()
    private_object = outputs[1].cpu()
    private_ready = outputs[2].cpu()
    private_head = outputs[3].cpu()
    private_output = outputs[4].cpu()
    candidate_counts = outputs[6].cpu()
    producer_stats = outputs[7].cpu()
    get_stats = outputs[8].cpu()

    source_lens = [int(value) for value in inputs["score_source_lens"].cpu().tolist()]
    expected_total = sum(source_lens)
    expected_per_lane = expected_total // GET_LANE_COUNT
    if int(candidate_counts.sum()) != expected_total:
        raise AssertionError(
            f"CandidateCompact count={int(candidate_counts.sum())} expected={expected_total}"
        )
    if int(producer_stats[:, 0].sum()) != expected_total:
        raise AssertionError("ExactScoreScan selected count does not match valid C8 scores")
    if int(producer_stats[:, 1].sum()) != 0 or int(producer_stats[:, 4].sum()) != 0:
        raise AssertionError("live fixture unexpectedly classified a candidate as resident/deferred")
    torch.testing.assert_close(
        get_stats[:, 0], torch.full((GET_LANE_COUNT,), expected_per_lane, dtype=torch.int64)
    )
    torch.testing.assert_close(
        get_stats[:, 2], torch.zeros(GET_LANE_COUNT, dtype=torch.int64)
    )
    torch.testing.assert_close(
        private_head, torch.full((GET_LANE_COUNT,), expected_per_lane, dtype=torch.int32)
    )
    if int((private_ready == PRIVATE_READY_READY).sum()) != expected_total:
        raise AssertionError("terminal CQ did not make every live private slot READY")

    # Every valid raw source is mapped exactly once for its row/pool.  Validate
    # the output payload by private object slot, independent of producer/CQ
    # arrival order.
    expected_slots: list[int] = []
    expected_objects: list[int] = []
    for pool, source_len in enumerate(source_lens):
        for source in range(source_len):
            slot = int(private_source[pool, source])
            object_slot = int(private_object[pool, source])
            if slot < 0 or object_slot != slot:
                raise AssertionError(
                    f"private mapping missing pool={pool} source={source} "
                    f"source_slot={slot} object_slot={object_slot}"
                )
            lane = slot // private_arena_capacity
            local_slot = slot % private_arena_capacity
            if lane != source % GET_LANE_COUNT or int(private_ready[lane, local_slot]) != PRIVATE_READY_READY:
                raise AssertionError(
                    f"private ready/lane mismatch pool={pool} source={source} slot={slot}"
                )
            expected_slots.append(slot)
            expected_objects.append(source)
        if source_len < private_source.shape[1] and int((private_source[pool, source_len:] >= 0).sum()) != 0:
            raise AssertionError(f"inactive C8 tail was speculatively mapped for pool={pool}")
    slot_tensor = torch.tensor(expected_slots, dtype=torch.long)
    object_tensor = torch.tensor(expected_objects, dtype=torch.long)
    torch.testing.assert_close(
        private_output.index_select(0, slot_tensor),
        source_cpu.index_select(0, object_tensor),
    )

    # The final C8 Top-K must be a subset of the speculative coverage.  C8
    # retains all authority over the final Top-K; this only proves no winner
    # remains absent from the private view before GuaranteedRecall.
    derived_indices = result["derived_indices"].cpu()
    for row, source_len in enumerate(source_lens):
        for source in derived_indices[row].reshape(-1).tolist():
            source = int(source)
            if source < 0 or source >= source_len or int(private_source[row, source]) < 0:
                raise AssertionError(
                    f"final C8 Top-K source is not prefetch-covered row={row} source={source}"
                )
    for name, before in publication_before.items():
        torch.testing.assert_close(inputs[name].cpu(), before)

    print(
        "P5_LIVE_C8_EXECUTOR_PASS "
        f"candidates={expected_total} lane0_requested={int(get_stats[0, 0])} "
        f"lane1_requested={int(get_stats[1, 0])} "
        f"cq_errors={int(get_stats[:, 2].sum())} "
        "c8_topk_prefetch_covered=1 payload_exact=1 post_cq_ready=1",
        flush=True,
    )


def _run(args: argparse.Namespace) -> None:
    device = _require_mapped_npu0(args)
    if os.getenv("P3_PUBLICATION_MODE") != "publisher" or os.getenv("P3_INLINE_MODE") != "off":
        raise RuntimeError(
            "build/run the strict C8 publisher with P3_PUBLICATION_MODE=publisher "
            "and P3_INLINE_MODE=off before this live P5 gate"
        )

    # P3 must run before importing the P5 bridge: each bridge has an isolated
    # generated OPAPI resolver.  The P3 launch is complete and strictly
    # validated before P5 changes the resolver to its own package.
    result = c8_target.run_target_equivalence()
    publication = _require_c8_strict_gate(result, device)
    torch.npu.synchronize()
    publication_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in publication.items()
        if name != "score_source_lens"
    }

    import kvaf
    import nanovllm_exact_scan_get  # noqa: F401  # Registers the P5 bridge.

    inputs, table_capacity, private_arena_capacity = _make_p5_inputs(publication, device)
    runtime = kvaf.NpuRuntime(
        args.batch_get_kernel,
        0,
        GET_LANE_COUNT,
        args.batch_put_kernel,
        1,
    )
    if callable(getattr(runtime, "link_to", None)):
        raise RuntimeError("live P5 gate requires the local Direct-HCCP KVAF runtime")

    table = None
    bound = False
    try:
        source_cpu = _make_full_object_payload(c8_target.SOURCE_TOKENS)
        keys = list(range(1, table_capacity + 1))
        offsets = [index * ENTRY_BYTES for index in range(table_capacity)]
        lengths = [ENTRY_BYTES] * table_capacity
        print(
            "P5_LIVE_C8_RUNTIME_INIT "
            f"physical_npu={os.getenv('P3_PHYSICAL_NPU_ID')} logical_npu=0 "
            f"entries={table_capacity} ddr_bytes={table_capacity * ENTRY_BYTES} "
            f"private_arena_per_lane={private_arena_capacity}",
            flush=True,
        )
        runtime.initialize(table_capacity * ENTRY_BYTES, ENTRY_BYTES)
        runtime.batch_put(keys, source_cpu.reshape(-1).to(device), offsets, lengths)
        torch.npu.synchronize()
        table = runtime.prepare_access_table(keys)
        access = runtime.access_table_view(table, device="npu", capacity=table_capacity)
        inputs["address_table"] = access.addresses
        inputs["valid_table"] = access.valid
        runtime.bind_fused_batch_get_control(
            inputs["private_packed_output"],
            inputs["fused_get_control"],
            torch.npu.current_stream(),
        )
        bound = True
        torch.npu.synchronize()
        _require_fused_control(inputs["fused_get_control"], inputs["private_packed_output"])
        outputs = _call_p5(inputs)
        torch.npu.synchronize()
        _assert_live_oracle(
            result,
            inputs,
            outputs,
            source_cpu,
            private_arena_capacity,
            publication_before,
        )
    finally:
        primary_error = sys.exc_info()[0] is not None
        cleanup_error: BaseException | None = None
        try:
            torch.npu.synchronize()
        except BaseException as error:
            cleanup_error = error
        if bound:
            try:
                runtime.release_fused_batch_get_control()
            except BaseException as error:
                cleanup_error = cleanup_error or error
        if table is not None:
            try:
                runtime.release_access_table(table)
            except BaseException as error:
                cleanup_error = cleanup_error or error
        try:
            runtime.shutdown()
        except BaseException as error:
            cleanup_error = cleanup_error or error
        if cleanup_error is not None and not primary_error:
            raise cleanup_error


def main() -> None:
    _run(_parse_args())


if __name__ == "__main__":
    main()
