#!/usr/bin/env python3
"""NPU acceptance fixture for the atomic P9 resident-Get operator.

This is deliberately a P9 fixture, rather than a reuse of a historical
streaming/P8 fixture.  It has exactly two duties:

* with ``runtime_enable=0``, prove that the source-derived P9 C8 operator
  produces the byte-identical official-C8 Top-K result; and
* with ``runtime_enable=1``, cold-bind a *local* KVAF layer-resident
  Direct-HCCP control, run the one P9 graph node, then prove that every final
  Top-K source has reached its canonical resident slot with the original DDR
  bytes.

The fixture intentionally has no private arena, cross-forward state, replay
identifier, or host-side BatchGet submission.  P9 owns the device-side
score-seam, post, drain, exact recall and final resident-slot mapping.  The
host only performs one-time Direct-HCCP registration before the graph node.

It is fail-closed: a remote CPU-service KVAF runtime, a missing fresh local
kernel, absent custom OPP, or an incomplete terminal result is a blocked/fail
result, never an implicit smoke-test pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch_npu  # noqa: F401 -- registers the NPU backend.


BLOCK_SIZE = 128
PACKED_KV_ROW_BYTES = 656
ENTRY_BYTES = BLOCK_SIZE * PACKED_KV_ROW_BYTES
CONTROL_WORDS = 16
TOPK = 2048
# P9's fixture reserves a low logical range for persistent resident entries
# and gives speculation a separate high range.  This is a construction-time
# admission property, not a device-side scan of persistent_object_slots.
PERSISTENT_LOGICAL_SLOT_LIMIT = TOPK
SPECULATIVE_LOGICAL_SLOT_BASE = PERSISTENT_LOGICAL_SLOT_LIMIT
RESIDENT_BLOCK_TABLE_COLUMNS = (
    SPECULATIVE_LOGICAL_SLOT_BASE + TOPK + BLOCK_SIZE - 1
) // BLOCK_SIZE
QUEUE_COUNT = 1
INDEX_HEADS = 64
KEY_HEADS = 1
HEAD_DIM = 128
BATCH_SIZE = 2
SOURCE_TOKENS = 4096
METADATA_WORDS = 1024
INT64_MAX = (1 << 63) - 1


def _blocked(reason: str) -> None:
    print(f"P9_RUNTIME_BLOCKED reason={reason}", flush=True)
    raise SystemExit(2)


def _require_file(name: str) -> Path:
    raw = os.environ.get(name, "")
    path = Path(raw).expanduser() if raw else None
    if path is None or not path.is_file():
        _blocked(f"missing_file:{name}")
    return path.resolve()


def _require_dir(name: str) -> Path:
    raw = os.environ.get(name, "")
    path = Path(raw).expanduser() if raw else None
    if path is None or not path.is_dir():
        _blocked(f"missing_directory:{name}")
    return path.resolve()


def _sha16(value: torch.Tensor) -> str:
    cpu = value.detach().to("cpu").contiguous().view(torch.uint8).reshape(-1)
    return hashlib.sha256(bytes(cpu.tolist())).hexdigest()[:16]


def _npu(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device).contiguous()


def _require_npu(device_text: str) -> torch.device:
    physical = os.environ.get("P9_PHYSICAL_NPU_ID", "")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if physical != "7" or visible != "7" or device_text != "npu:0":
        _blocked(
            "requires_physical_7_mapped_to_logical_0:"
            f"physical={physical or '<unset>'}:visible={visible or '<unset>'}:device={device_text}"
        )
    if int(torch.npu.device_count()) != 1:
        _blocked(f"expected_one_visible_npu:count={torch.npu.device_count()}")
    torch.npu.set_device(0)
    value = torch.ones(1, device="npu:0").cpu().item()
    name = str(getattr(torch.npu, "get_device_name")(0))
    if "950" not in name.lower() or value != 1.0:
        _blocked(f"npu_probe_failed:name={name!r}:value={value}")
    print(
        "P9_RUNTIME_NPU "
        "physical=7 logical=0 visible='7' "
        f"device=npu:0 device_name={name!r} value={value}",
        flush=True,
    )
    return torch.device("npu:0")


def _quantize_fp8(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quantized, scale = torch_npu.npu_dynamic_quant(value, dst_type=torch.float8_e4m3fn)
    return quantized.contiguous(), scale.view(value.shape[:-1]).to(torch.float32).contiguous()


def _build_official_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    """Create one deterministic, real TND/PA_BSND C8 input set."""
    blocks = SOURCE_TOKENS // BLOCK_SIZE
    torch.manual_seed(20260902)
    torch.npu.manual_seed_all(20260902)
    query_fp = torch.empty(
        (BATCH_SIZE, INDEX_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-1.0, 1.0)
    key_fp = torch.empty(
        (blocks, BLOCK_SIZE, KEY_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-1.0, 1.0)
    query, query_scale = _quantize_fp8(query_fp)
    key, key_scale = _quantize_fp8(key_fp)
    weights = torch.empty(
        (BATCH_SIZE, INDEX_HEADS), dtype=torch.float32, device=device
    ).uniform_(0.01, 1.0).contiguous()
    actual_q = torch.arange(1, BATCH_SIZE + 1, dtype=torch.int32, device=device)
    actual_k = torch.tensor(
        [SOURCE_TOKENS, SOURCE_TOKENS - BLOCK_SIZE], dtype=torch.int32, device=device
    )
    block_table = torch.stack(
        [torch.randperm(blocks, dtype=torch.int64).to(torch.int32) for _ in range(BATCH_SIZE)]
    ).to(device)
    return {
        "query": query,
        "key": key,
        "weights": weights,
        "query_dequant_scale": query_scale,
        "key_dequant_scale": key_scale,
        "actual_seq_lengths_query": actual_q,
        "actual_seq_lengths_key": actual_k,
        "block_table": block_table,
    }


def _module_under(module: Any, root: Path, label: str) -> None:
    file_name = Path(str(getattr(module, "__file__", ""))).resolve()
    try:
        file_name.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"{label} module came from a stale/imported path: module={file_name} root={root}"
        ) from error


def _load_ops() -> tuple[Any, Any, Any, Path, Path]:
    """Load two isolated bridge resolvers and their two distinct vendors."""
    reference_opapi = _require_file("P9_REFERENCE_OPAPI")
    target_opapi = _require_file("P9_TARGET_OPAPI")
    reference_root = _require_dir("P9_REFERENCE_BRIDGE_ROOT")
    target_root = _require_dir("P9_TARGET_BRIDGE_ROOT")
    reference_vendor = reference_opapi.parents[2]
    target_vendor = target_opapi.parents[2]
    if not (reference_vendor.is_dir() and target_vendor.is_dir()):
        _blocked("opapi_vendor_missing")
    if reference_opapi == target_opapi:
        _blocked("reference_and_target_opapi_are_identical")
    if "nanovllm_official_c8_reference" in sys.modules or (
        "nanovllm_a5_c8_exact_speculative_resident_get" in sys.modules
    ):
        _blocked("requires_fresh_python_process_before_bridge_import")

    # CANN discovers custom op definitions from this process-level list;
    # each bridge still dlopens only its own opapi via its dedicated env var.
    inherited = [entry for entry in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if entry]
    ordered = [str(reference_vendor), str(target_vendor)]
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(
        [*ordered, *(entry for entry in inherited if entry not in ordered)]
    )
    os.environ["NANOVLLM_A5_REFERENCE_C8_OPAPI_LIB"] = str(reference_opapi)
    os.environ["NANOVLLM_A5_EXACT_SPECULATIVE_RESIDENT_OPAPI_LIB"] = str(target_opapi)

    reference_module = importlib.import_module("nanovllm_official_c8_reference")
    target_module = importlib.import_module("nanovllm_a5_c8_exact_speculative_resident_get")
    _module_under(reference_module, reference_root, "reference")
    _module_under(target_module, target_root, "target")
    reference_namespace = getattr(torch.ops, "a5_official_c8_reference", None)
    target_namespace = getattr(torch.ops, "a5_c8_exact_speculative_resident_get", None)
    reference_op = getattr(reference_namespace, "npu_vllm_quant_lightning_indexer", None)
    metadata_op = getattr(reference_namespace, "npu_vllm_quant_lightning_indexer_metadata", None)
    target_op = getattr(target_namespace, "npu_a5_c8_exact_speculative_resident_get", None)
    if reference_op is None or metadata_op is None or target_op is None:
        _blocked("bridge_op_registration_missing")
    print(
        "P9_RUNTIME_ARTIFACTS "
        f"reference_opapi={reference_opapi} target_opapi={target_opapi} "
        f"reference_bridge={reference_root} target_bridge={target_root}",
        flush=True,
    )
    return reference_op, metadata_op, target_op, reference_opapi, target_opapi


def _metadata(metadata_op: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    result = metadata_op(
        num_heads_q=INDEX_HEADS,
        num_heads_k=KEY_HEADS,
        head_dim=HEAD_DIM,
        query_quant_mode=0,
        key_quant_mode=0,
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_key=inputs["actual_seq_lengths_key"],
        batch_size=BATCH_SIZE,
        max_seqlen_q=1,
        max_seqlen_k=SOURCE_TOKENS,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        cmp_ratio=1,
        device=str(inputs["query"].device),
    )
    if not isinstance(result, torch.Tensor) or result.dtype != torch.int32 or result.numel() != METADATA_WORDS:
        raise RuntimeError(f"official metadata ABI mismatch: {result!r}")
    return result.contiguous()


def _call_reference(op: Any, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    result = op(
        query=inputs["query"],
        key=inputs["key"],
        weights=inputs["weights"],
        query_dequant_scale=inputs["query_dequant_scale"],
        key_dequant_scale=inputs["key_dequant_scale"],
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_key=inputs["actual_seq_lengths_key"],
        block_table=inputs["block_table"],
        metadata=inputs["metadata"],
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        cmp_ratio=1,
        return_value=False,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("official reference output ABI mismatch")
    indices, values = result
    if tuple(indices.shape) != (BATCH_SIZE, KEY_HEADS, TOPK) or indices.dtype != torch.int32:
        raise RuntimeError(f"official sparse ABI mismatch: shape={tuple(indices.shape)} dtype={indices.dtype}")
    return indices, values


def _assert_fixture_resident_admission(
    *,
    req_pool_entries: torch.Tensor,
    persistent_object_slots: torch.Tensor,
    canonical_resident_slots: torch.Tensor,
    resident_block_table: torch.Tensor,
    active_row_count: int,
    output_rows: int,
    require_padded_alias: bool,
) -> None:
    """Validate the fixed P9 plan once while constructing this test fixture.

    The production device path deliberately never performs a
    candidate-by-object persistent-directory scan.  This small CPU-side
    fixture gate instead proves the construction invariant it relies on:
    active execution rows map to distinct pools and physical output rows;
    logical speculative slots are in the high range, structurally separate
    from the low persistent range; and a padded duplicate pool entry is
    harmless because it is outside ``active_row_count``.
    """

    rows = int(req_pool_entries.numel())
    if active_row_count < 0 or active_row_count > rows:
        raise AssertionError(
            f"fixture active-row range invalid: active={active_row_count} rows={rows}"
        )
    if tuple(canonical_resident_slots.shape[1:]) != (TOPK,):
        raise AssertionError(
            "fixture canonical logical table shape invalid: "
            f"shape={tuple(canonical_resident_slots.shape)}"
        )
    if tuple(resident_block_table.shape) != (
        rows,
        RESIDENT_BLOCK_TABLE_COLUMNS,
    ):
        raise AssertionError(
            "fixture execution-row block table shape invalid: "
            f"shape={tuple(resident_block_table.shape)}"
        )
    if output_rows <= 0:
        raise AssertionError(f"fixture output row count invalid: {output_rows}")

    persistent_nonnegative = persistent_object_slots[
        persistent_object_slots >= 0
    ]
    if persistent_nonnegative.numel() and bool(
        torch.any(persistent_nonnegative >= PERSISTENT_LOGICAL_SLOT_LIMIT)
    ):
        raise AssertionError(
            "fixture persistent slot escaped the reserved low logical range"
        )
    if bool(
        torch.any(canonical_resident_slots < SPECULATIVE_LOGICAL_SLOT_BASE)
    ):
        raise AssertionError(
            "fixture canonical slots overlap the persistent logical range"
        )

    active_pools = req_pool_entries[:active_row_count]
    if active_pools.numel() and int(torch.unique(active_pools).numel()) != active_row_count:
        raise AssertionError(
            "fixture active execution rows alias a pool: "
            f"pools={req_pool_entries.tolist()} active={active_row_count}"
        )
    if require_padded_alias:
        padded_pools = req_pool_entries[active_row_count:]
        if padded_pools.numel() == 0 or active_pools.numel() == 0:
            raise AssertionError("fixture requested a padded alias without active and padded rows")
        if not bool(torch.any(padded_pools == active_pools[0])):
            raise AssertionError(
                "fixture padded row does not intentionally alias an active pool"
            )

    physical_rows: list[torch.Tensor] = []
    for row in range(active_row_count):
        pool = int(req_pool_entries[row])
        if pool < 0 or pool >= canonical_resident_slots.size(0):
            raise AssertionError(f"fixture active pool out of range: row={row} pool={pool}")
        logical = canonical_resident_slots[pool]
        if int(torch.unique(logical).numel()) != TOPK:
            raise AssertionError(f"fixture logical slot duplicate: row={row} pool={pool}")
        block_indices = torch.div(logical, BLOCK_SIZE, rounding_mode="floor")
        if bool(
            torch.any(block_indices < 0)
            or torch.any(block_indices >= RESIDENT_BLOCK_TABLE_COLUMNS)
        ):
            raise AssertionError(f"fixture logical block range invalid: row={row}")
        physical = (
            resident_block_table[row, block_indices] * BLOCK_SIZE
            + torch.remainder(logical, BLOCK_SIZE)
        )
        if bool(torch.any(physical < 0) or torch.any(physical >= output_rows)):
            raise AssertionError(f"fixture physical destination range invalid: row={row}")
        if int(torch.unique(physical).numel()) != TOPK:
            raise AssertionError(f"fixture physical destination duplicate: row={row}")
        physical_rows.append(physical)

    if physical_rows:
        merged = torch.cat(physical_rows)
        if int(torch.unique(merged).numel()) != int(merged.numel()):
            raise AssertionError("fixture active physical destinations overlap")
    print(
        "P9_RUNTIME_ADMISSION "
        f"active_rows={active_row_count} pool_injective=1 "
        "logical_persistent_disjoint=1 physical_destinations_unique=1 "
        f"padded_alias_excluded={int(require_padded_alias)}",
        flush=True,
    )


def _p9_inputs(
    base: dict[str, torch.Tensor],
    device: torch.device,
    runtime_enabled: bool,
    control: torch.Tensor,
    *,
    active_rows: int | None = None,
    padded_alias: bool = False,
) -> dict[str, torch.Tensor]:
    rows = BATCH_SIZE * KEY_HEADS
    source = SOURCE_TOKENS
    if active_rows is None:
        active_rows = rows
    # P9 input source->object mapping is deterministic in this cold fixture:
    # source S maps to KV object S within each request pool.  The K/V payload
    # is one 656-byte object row and Direct-HCCP entry capacity is 128 rows.
    runtime = _npu(torch.tensor([1 if runtime_enabled else 0], dtype=torch.int32), device)
    threshold = _npu(torch.zeros(rows, dtype=torch.int32), device)
    threshold_valid = _npu(torch.ones(rows, dtype=torch.int32), device)
    # Pool 1 uses a one-entry rotation so the active proof validates both
    # execution-row -> pool lookup *and* pool-local S -> O mapping.  The
    # source C8 inputs remain identical; only the fixture's DDR object map
    # chooses which registered source object is fetched for that source id.
    object_map_cpu = torch.stack(
        [
            torch.arange(source, dtype=torch.int32),
            (torch.arange(source, dtype=torch.int32) + BLOCK_SIZE) % source,
        ]
    )
    object_map = _npu(object_map_cpu, device)
    object_lens = base["actual_seq_lengths_key"].detach().clone().to(torch.int32)
    # Use a non-identity request->pool mapping in the acceptance fixture.
    # This specifically proves that P9 does not derive a destination slot by
    # the old affine ``row * 2048 + rank`` rule: the device must look up the
    # canonical slot row selected by ``req_pool_entries[row]``.
    req_pool_cpu = torch.arange(rows - 1, -1, -1, dtype=torch.int32)
    if padded_alias:
        if active_rows <= 0 or active_rows >= rows:
            raise AssertionError(
                "padded_alias fixture requires a nonempty strict active prefix"
            )
        req_pool_cpu[active_rows] = req_pool_cpu[0]
    persistent_cpu = torch.full((rows, source), -1, dtype=torch.int32)
    # Logical QSFA slots and physical Direct-HCCP output rows are deliberately
    # separate inputs.  Logical slots are pool-owned and remain the values
    # returned to QSFA.  Physical rows are execution-row-owned, because two
    # active execution rows may point at different HBM blocks even when their
    # pool ordinals are arranged differently.
    #
    # Both tables are non-identity: a target that treats a logical slot as a
    # physical output row, or assumes ``row * 2048 + ordinal``, fails this
    # acceptance fixture.
    ordinal = torch.arange(TOPK, dtype=torch.int32)
    canonical_slots_cpu = torch.empty((rows, TOPK), dtype=torch.int32)
    for pool in range(rows):
        canonical_slots_cpu[pool] = (
            SPECULATIVE_LOGICAL_SLOT_BASE + ordinal
            if pool % 2 == 0
            else SPECULATIVE_LOGICAL_SLOT_BASE + TOPK - 1 - ordinal
        )

    blocks_per_row = TOPK // BLOCK_SIZE
    first_speculative_block = SPECULATIVE_LOGICAL_SLOT_BASE // BLOCK_SIZE
    resident_blocks_cpu = torch.full(
        (rows, RESIDENT_BLOCK_TABLE_COLUMNS), -1, dtype=torch.int32
    )
    for row in range(rows):
        # The high logical range is mapped to a unique, reverse physical HBM
        # interval per execution row.  Thus neither logical slot values nor
        # a row*2048 affine formula can substitute for this table.
        row_base = (rows - 1 - row) * blocks_per_row
        resident_blocks_cpu[
            row, first_speculative_block : first_speculative_block + blocks_per_row
        ] = row_base + torch.arange(
            blocks_per_row - 1, -1, -1, dtype=torch.int32
        )
    _assert_fixture_resident_admission(
        req_pool_entries=req_pool_cpu,
        persistent_object_slots=persistent_cpu,
        canonical_resident_slots=canonical_slots_cpu,
        resident_block_table=resident_blocks_cpu,
        active_row_count=active_rows,
        output_rows=rows * TOPK,
        require_padded_alias=padded_alias,
    )

    req_pool = _npu(req_pool_cpu, device)
    persistent = _npu(persistent_cpu, device)
    canonical_slots = _npu(canonical_slots_cpu, device)
    resident_block_table = _npu(resident_blocks_cpu, device)
    candidate_map = _npu(torch.full((rows, source), -1, dtype=torch.int32), device)
    candidate_cursor = _npu(torch.zeros(rows, dtype=torch.int32), device)
    rows_per_entry = _npu(torch.tensor([BLOCK_SIZE], dtype=torch.int32), device)
    active_row_count = _npu(torch.tensor([active_rows], dtype=torch.int32), device)
    return {
        **base,
        "runtime_enable": runtime,
        "resident_threshold": threshold,
        "resident_threshold_valid": threshold_valid,
        "req_pool_entries": req_pool,
        "kv_object_map": object_map,
        "kv_object_lens": object_lens,
        "persistent_object_slots_snapshot": persistent,
        "canonical_resident_slots": canonical_slots,
        "candidate_source_to_slot": candidate_map,
        "candidate_slot_cursor": candidate_cursor,
        "layer_resident_get_control": control,
        "rows_per_entry": rows_per_entry,
        "resident_block_table": resident_block_table,
        "active_row_count": active_row_count,
    }


def _call_p9(op: Any, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return op(
        query=inputs["query"],
        key=inputs["key"],
        weights=inputs["weights"],
        query_dequant_scale=inputs["query_dequant_scale"],
        key_dequant_scale=inputs["key_dequant_scale"],
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_key=inputs["actual_seq_lengths_key"],
        block_table=inputs["block_table"],
        metadata=inputs["metadata"],
        runtime_enable=inputs["runtime_enable"],
        resident_threshold=inputs["resident_threshold"],
        resident_threshold_valid=inputs["resident_threshold_valid"],
        req_pool_entries=inputs["req_pool_entries"],
        kv_object_map=inputs["kv_object_map"],
        kv_object_lens=inputs["kv_object_lens"],
        persistent_object_slots_snapshot=inputs["persistent_object_slots_snapshot"],
        canonical_resident_slots=inputs["canonical_resident_slots"],
        candidate_source_to_slot=inputs["candidate_source_to_slot"],
        candidate_slot_cursor=inputs["candidate_slot_cursor"],
        layer_resident_get_control=inputs["layer_resident_get_control"],
        rows_per_entry=inputs["rows_per_entry"],
        resident_block_table=inputs["resident_block_table"],
        active_row_count=inputs["active_row_count"],
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        cmp_ratio=1,
        return_values=False,
        stride=int(inputs["key"].stride(0)),
        scale_stride=int(inputs["key_dequant_scale"].stride(0)),
    )


def _assert_disabled_sparse_math(
    reference: tuple[torch.Tensor, torch.Tensor],
    p9_result: tuple[torch.Tensor, ...],
) -> None:
    """Compare only C8's public mathematical result.

    The ABI-only control operator deliberately preserves the C8 compute core
    while leaving P9-only auxiliary outputs unwritten.  This narrow gate lets
    the same fixture distinguish an ABI/build/launch effect from a score-seam
    effect without pretending that the control implements resident Get.
    """

    torch.npu.synchronize()
    indices, values = reference
    target_indices, target_values, _slots, _stats = p9_result
    if not torch.equal(indices.cpu(), target_indices.cpu()):
        raise AssertionError(
            "runtime_enable=0 changed exact C8 sparse_indices: "
            f"reference_sha16={_sha16(indices)} target_sha16={_sha16(target_indices)}"
        )
    if tuple(target_values.shape) != tuple(values.shape):
        raise AssertionError("runtime_enable=0 changed sparse_values ABI")
    print(
        "P9_RUNTIME_DISABLED_MATH runtime_enable=0 exact_sparse=1 "
        f"reference_sha16={_sha16(indices)} target_sha16={_sha16(target_indices)}",
        flush=True,
    )


def _assert_disabled_gate(
    reference: tuple[torch.Tensor, torch.Tensor], p9_result: tuple[torch.Tensor, ...], p9: dict[str, torch.Tensor]
) -> None:
    _assert_disabled_sparse_math(reference, p9_result)
    _target_indices, _target_values, slots, stats = p9_result
    if int(torch.count_nonzero(p9["candidate_source_to_slot"].cpu() >= 0)) != 0:
        raise AssertionError("runtime_enable=0 mutated candidate source slots")
    if int(torch.count_nonzero(p9["candidate_slot_cursor"].cpu())) != 0:
        raise AssertionError("runtime_enable=0 mutated candidate cursor")
    if int(torch.count_nonzero(slots.cpu() >= 0)) != 0 or int(torch.count_nonzero(stats.cpu())) != 0:
        raise AssertionError("runtime_enable=0 produced resident/Post state")
    print(
        "P9_RUNTIME_C8_GATE runtime_enable=0 exact_sparse=1 "
        f"reference_sha16={_sha16(indices)} target_sha16={_sha16(target_indices)}",
        flush=True,
    )


def _assert_all_padding_gate(
    reference: tuple[torch.Tensor, torch.Tensor],
    p9_result: tuple[torch.Tensor, ...],
    p9: dict[str, torch.Tensor],
) -> None:
    """Prove graph padding cannot read Direct-HCCP control or mutate P9 state."""

    torch.npu.synchronize()
    indices, values = reference
    target_indices, target_values, slots, stats = p9_result
    if not torch.equal(indices.cpu(), target_indices.cpu()):
        raise AssertionError("active_row_count=0 changed exact C8 sparse_indices")
    if tuple(target_values.shape) != tuple(values.shape):
        raise AssertionError("active_row_count=0 changed sparse_values ABI")
    if int(torch.count_nonzero(p9["candidate_source_to_slot"].cpu() >= 0)) != 0:
        raise AssertionError("active_row_count=0 mutated candidate source slots")
    if int(torch.count_nonzero(p9["candidate_slot_cursor"].cpu())) != 0:
        raise AssertionError("active_row_count=0 mutated candidate cursor")
    if int(torch.count_nonzero(slots.cpu() >= 0)) != 0 or int(torch.count_nonzero(stats.cpu())) != 0:
        raise AssertionError("active_row_count=0 touched terminal resident state")
    print("P9_RUNTIME_PADDING_GATE active_row_count=0 exact_sparse=1 p9_state_untouched=1", flush=True)


def _payload(entries: int) -> torch.Tensor:
    entry = torch.arange(entries, dtype=torch.int64).reshape(entries, 1)
    byte = torch.arange(ENTRY_BYTES, dtype=torch.int64).reshape(1, ENTRY_BYTES)
    return ((entry * 29 + byte * 7 + (entry >> 2) + 17) & 0xFF).to(torch.uint8)


def _require_local_layer_resident_runtime(runtime: Any) -> None:
    required = (
        "initialize",
        "batch_put",
        "prepare_access_table",
        "access_table_view",
        "bind_layer_resident_batch_get_control",
        "configure_layer_resident_batch_get_queue_templates",
        "release_layer_resident_batch_get_control",
        "layer_resident_batch_get_control_diagnostic",
        "release_access_table",
        "shutdown",
    )
    missing = [name for name in required if not callable(getattr(runtime, name, None))]
    if missing:
        _blocked("local_layer_resident_kvaf_api_missing:" + ",".join(missing))
    if callable(getattr(runtime, "link_to", None)):
        _blocked("remote_cpu_service_kvaf_is_not_p9_local_direct_hccp")


def _assert_control(control: torch.Tensor, output: torch.Tensor, leases: torch.Tensor, rows: int) -> None:
    torch.npu.synchronize()
    words = [int(item) & ((1 << 64) - 1) for item in control.cpu().tolist()]
    failures: list[str] = []
    if len(words) != CONTROL_WORDS or words[0] == 0:
        failures.append("magic")
    if words[5] != (output.data_ptr() & ((1 << 64) - 1)):
        failures.append("output_alias")
    if words[6] != (leases.data_ptr() & ((1 << 64) - 1)):
        failures.append("lease_alias")
    if words[8] != QUEUE_COUNT:
        failures.append("queue_count")
    if words[9] != rows * TOPK:
        failures.append("output_rows")
    if words[10] != 0:
        failures.append("resident_row_base")
    if words[11] != rows * TOPK:
        failures.append("resident_row_count")
    if words[13] != PACKED_KV_ROW_BYTES or words[14] == 0:
        failures.append("descriptor_abi")
    if failures:
        raise RuntimeError(f"layer-resident control is not graph-ready: bad={failures} words={words}")


class _LocalResidentFixture:
    """Cold local KVAF binding; P9 device code remains the only Get submitter."""

    def __init__(self, device: torch.device, rows: int) -> None:
        try:
            import kvaf
        except Exception as error:  # noqa: BLE001 -- report a strict runtime prerequisite.
            _blocked(f"fresh_kvaf_python_import_failed:{type(error).__name__}")
        get_kernel = _require_file("P9_KVAF_BATCH_GET_KERNEL")
        put_kernel = _require_file("P9_KVAF_BATCH_PUT_KERNEL")
        self.device = device
        self.rows = rows
        self.entries = SOURCE_TOKENS // BLOCK_SIZE
        self.runtime = kvaf.NpuRuntime(str(get_kernel), 0, QUEUE_COUNT, str(put_kernel), 1)
        _require_local_layer_resident_runtime(self.runtime)
        self.payload = _payload(self.entries)
        self.table: Any | None = None
        self.control_bound = False
        self.terminal_validated = False
        self.output: torch.Tensor | None = None
        self.control: torch.Tensor | None = None
        self.leases: torch.Tensor | None = None

    def bind(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.runtime.initialize(self.entries * ENTRY_BYTES, ENTRY_BYTES)
        source = self.payload.to(self.device).reshape(-1).contiguous()
        keys = list(range(1, self.entries + 1))
        self.runtime.batch_put(
            keys,
            source,
            [index * ENTRY_BYTES for index in range(self.entries)],
            [ENTRY_BYTES] * self.entries,
        )
        torch.npu.synchronize()
        self.table = self.runtime.prepare_access_table(keys)
        access = self.runtime.access_table_view(self.table, device="npu", capacity=self.entries)
        torch.npu.synchronize()

        # One output row per entry in the explicit canonical slot table.  It
        # is ordinary graph-visible NPU storage, not a private arena.  P9
        # maps execution row -> req_pool_entries[row] -> this table, so the
        # binding covers pools rather than assuming an affine execution-row
        # layout.
        output = torch.zeros(
            (self.rows * TOPK, PACKED_KV_ROW_BYTES), dtype=torch.uint8, device=self.device
        )
        control = torch.zeros(CONTROL_WORDS, dtype=torch.int64, device=self.device)
        leases = torch.zeros(QUEUE_COUNT, dtype=torch.int32, device=self.device)
        capacity = TOPK
        indices = torch.zeros(capacity, dtype=torch.int32, device=self.device)
        offsets = torch.zeros(capacity, dtype=torch.int32, device=self.device)
        masks = torch.zeros(capacity, dtype=torch.uint8, device=self.device)
        outputs = torch.zeros(capacity, dtype=torch.int32, device=self.device)
        stream = torch.npu.current_stream()
        self.runtime.bind_layer_resident_batch_get_control(
            0, output, control, 0, self.rows * TOPK, leases, QUEUE_COUNT, stream
        )
        self.control_bound = True
        self.runtime.configure_layer_resident_batch_get_queue_templates(
            [0], [access.addresses], [access.valid], [indices], [offsets], [masks], [outputs], stream
        )
        _assert_control(control, output, leases, self.rows)
        self.output, self.control, self.leases = output, control, leases
        print(
            "P9_RUNTIME_REAL_BIND "
            f"entries={self.entries} rows={self.rows} canonical_rows={self.rows * TOPK} "
            f"queue_count={QUEUE_COUNT} diagnostic={self.runtime.layer_resident_batch_get_control_diagnostic()}",
            flush=True,
        )
        return output, control, leases

    def close(self) -> None:
        # A failed P9 launch may have a DMA whose terminal state is unknown.
        # Do not turn that into a false clean release; the fresh process is the
        # quarantine boundary.  On success every queue has been drained and
        # canonical output remains valid until this verification ends.
        if not self.terminal_validated:
            print("P9_RUNTIME_CLEANUP safe=0 action=process_exit_quarantine", flush=True)
            return
        try:
            torch.npu.synchronize()
            if self.control_bound:
                self.runtime.release_layer_resident_batch_get_control()
                self.control_bound = False
            if self.table is not None:
                self.runtime.release_access_table(self.table)
                self.table = None
            if getattr(self.runtime, "initialized", False):
                self.runtime.shutdown()
            print("P9_RUNTIME_CLEANUP safe=1 action=released_after_terminal", flush=True)
        except Exception as error:  # noqa: BLE001
            print(f"P9_RUNTIME_CLEANUP safe=0 action=process_exit_quarantine error={type(error).__name__}", flush=True)


def _assert_active(
    reference: torch.Tensor,
    p9_result: tuple[torch.Tensor, ...],
    p9: dict[str, torch.Tensor],
    fixture: _LocalResidentFixture,
) -> None:
    torch.npu.synchronize()
    indices, _values, attention_slots, stats = p9_result
    if not torch.equal(reference.cpu(), indices.cpu()):
        raise AssertionError(
            "runtime_enable=1 changed exact C8 sparse_indices: "
            f"reference_sha16={_sha16(reference)} target_sha16={_sha16(indices)}"
        )
    stats_cpu = stats.detach().cpu().to(torch.int64)
    slots_cpu = attention_slots.detach().cpu().to(torch.int64)
    indices_cpu = indices.detach().cpu().to(torch.int64).reshape(BATCH_SIZE * KEY_HEADS, TOPK)
    output_cpu = fixture.output.detach().cpu() if fixture.output is not None else None
    leases_cpu = fixture.leases.detach().cpu() if fixture.leases is not None else None
    if output_cpu is None or leases_cpu is None:
        raise AssertionError("resident fixture lost canonical output/lease keepalive")
    if tuple(stats_cpu.shape) != (BATCH_SIZE * KEY_HEADS, 7):
        raise AssertionError(f"terminal stats ABI mismatch: {tuple(stats_cpu.shape)}")
    if int(torch.count_nonzero(stats_cpu[:, 6])) != 0:
        raise AssertionError(f"P9 terminal reported failure: stats={stats_cpu.tolist()}")
    if int(torch.count_nonzero(leases_cpu)) != 0:
        raise AssertionError(f"P9 terminal leaked a Direct-HCCP queue lease: {leases_cpu.tolist()}")

    mismatches: list[str] = []
    canonical_slots = p9["canonical_resident_slots"].detach().cpu().to(torch.int64)
    pools = p9["req_pool_entries"].detach().cpu().to(torch.int64)
    resident_blocks = p9["resident_block_table"].detach().cpu().to(torch.int64)
    active_rows = int(p9["active_row_count"].detach().cpu().item())
    expected_rows = BATCH_SIZE * KEY_HEADS
    if active_rows != expected_rows:
        raise AssertionError(
            f"fixture expected every row active: active={active_rows} rows={expected_rows}"
        )
    if int(torch.unique(pools[:active_rows]).numel()) != active_rows:
        raise AssertionError(f"active execution rows alias one pool: pools={pools.tolist()}")
    if tuple(resident_blocks.shape) != (
        expected_rows,
        RESIDENT_BLOCK_TABLE_COLUMNS,
    ):
        raise AssertionError(
            "resident_block_table ABI mismatch: "
            f"shape={tuple(resident_blocks.shape)}"
        )

    all_physical_rows: list[torch.Tensor] = []
    for row in range(active_rows):
        slots = slots_cpu[row]
        sources = indices_cpu[row]
        pool = int(pools[row])
        if pool < 0 or pool >= canonical_slots.size(0):
            mismatches.append(f"pool_range_row={row}:pool={pool}")
            continue
        if tuple(canonical_slots.shape[1:]) != (TOPK,):
            raise AssertionError(
                "canonical resident slot-table ABI mismatch: "
                f"shape={tuple(canonical_slots.shape)}"
            )
        logical_slots = canonical_slots[pool]
        if bool(torch.any(logical_slots < SPECULATIVE_LOGICAL_SLOT_BASE).item()):
            mismatches.append(f"logical_persistent_overlap={row}:pool={pool}")
            continue
        physical_rows = (
            resident_blocks[row, torch.div(logical_slots, BLOCK_SIZE, rounding_mode="floor")]
            * BLOCK_SIZE
            + torch.remainder(logical_slots, BLOCK_SIZE)
        )
        if not torch.equal(slots, logical_slots):
            mismatches.append(f"logical_slot_table_row={row}:pool={pool}")
            continue
        if bool(torch.any(physical_rows < 0).item()) or bool(torch.any(physical_rows >= output_cpu.size(0)).item()):
            mismatches.append(f"physical_row_table_range={row}:pool={pool}")
            continue
        if int(torch.unique(physical_rows).numel()) != TOPK:
            mismatches.append(f"physical_row_table_duplicate={row}:pool={pool}")
            continue
        if int(torch.unique(slots).numel()) != TOPK:
            mismatches.append(f"slot_duplicate_row={row}")
            continue
        all_physical_rows.append(physical_rows)
        logical_limit = int(p9["kv_object_lens"][row].detach().cpu().item())
        for col in range(TOPK):
            source = int(sources[col])
            physical_row = int(physical_rows[col])
            if source < 0 or source >= logical_limit:
                mismatches.append(f"source_range_row={row}:col={col}:source={source}")
                break
            obj = int(p9["kv_object_map"][pool, source].detach().cpu().item())
            if obj < 0 or obj >= SOURCE_TOKENS:
                mismatches.append(f"object_range_row={row}:col={col}:object={obj}")
                break
            expected = fixture.payload[obj // BLOCK_SIZE].reshape(BLOCK_SIZE, PACKED_KV_ROW_BYTES)[obj % BLOCK_SIZE]
            if not torch.equal(output_cpu[physical_row], expected):
                mismatches.append(
                    f"payload_row={row}:col={col}:source={source}:object={obj}:physical={physical_row}"
                )
                break
    if len(all_physical_rows) == active_rows:
        all_physical = torch.cat(all_physical_rows)
        if int(torch.unique(all_physical).numel()) != int(all_physical.numel()):
            mismatches.append("active_rows_physical_destination_overlap")
    if mismatches:
        raise AssertionError("P9 canonical resident payload mismatch: " + ",".join(mismatches[:8]))

    # With an all-cold persistent snapshot, every final winner must have been
    # transported by speculative post or exact guaranteed recall.  This is
    # topology-independent: it does not assume a permanent Get core.
    transported = stats_cpu[:, 1] + stats_cpu[:, 2]
    if bool(torch.any(transported != TOPK).item()):
        raise AssertionError(
            "all-cold P9 terminal transport accounting is incomplete: "
            f"speculative={stats_cpu[:, 1].tolist()} recall={stats_cpu[:, 2].tolist()}"
        )
    fixture.terminal_validated = True
    print(
        "P9_RUNTIME_ACTIVE "
        "exact_sparse=1 canonical_payload=1 physical_map_nonidentity=1 "
        "active_destinations_unique=1 queue_leases_zero=1 "
        f"speculative={stats_cpu[:, 1].tolist()} recall={stats_cpu[:, 2].tolist()} "
        f"recall_waves={stats_cpu[:, 3].tolist()} candidate_mapped="
        f"{int(torch.count_nonzero(p9['candidate_source_to_slot'].detach().cpu() >= 0))}",
        flush=True,
    )


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--case", choices=("disabled-math", "full"), default="full")
    args = parser.parse_args()
    if os.environ.get("P9_REQUIRE_REAL_HCCP", "1") != "1":
        _blocked("P9_REQUIRE_REAL_HCCP_must_be_1_for_acceptance")
    if os.environ.get("TASK_QUEUE_ENABLE") != "0" or os.environ.get("ASCEND_LAUNCH_BLOCKING", "0") != "0":
        _blocked("requires_direct_async_host_environment_task_queue_0_launch_blocking_0")
    device = _require_npu(args.device)
    reference_op, metadata_op, target_op, reference_opapi, target_opapi = _load_ops()
    base = _build_official_inputs(device)
    base["metadata"] = _metadata(metadata_op, base)
    reference = _call_reference(reference_op, base)
    torch.npu.synchronize()
    print(
        "P9_RUNTIME_REFERENCE "
        f"sparse_sha16={_sha16(reference[0])} reference_opapi={reference_opapi} target_opapi={target_opapi}",
        flush=True,
    )

    disabled_control = torch.zeros(CONTROL_WORDS, dtype=torch.int64, device=device)
    disabled = _p9_inputs(base, device, False, disabled_control)
    disabled_result = _call_p9(target_op, disabled)
    if args.case == "disabled-math":
        _assert_disabled_sparse_math(reference, disabled_result)
        print("P9_RUNTIME_PASS case=disabled-math runtime_enable_0=1", flush=True)
        return 0
    _assert_disabled_gate(reference, disabled_result, disabled)

    # A captured graph may retain padded execution rows whose pool indices
    # alias a live request.  This must be harmless even when runtime_enable
    # is on: the first P9 control/CQ read is gated by active_row_count, so a
    # zero active prefix cannot observe the deliberately invalid all-zero
    # Direct-HCCP control.
    all_padding = _p9_inputs(
        base,
        device,
        True,
        disabled_control,
        active_rows=1,
        padded_alias=True,
    )
    all_padding["active_row_count"].zero_()
    _assert_all_padding_gate(
        reference, _call_p9(target_op, all_padding), all_padding
    )

    fixture = _LocalResidentFixture(device, BATCH_SIZE * KEY_HEADS)
    try:
        _output, control, _leases = fixture.bind()
        active = _p9_inputs(base, device, True, control)
        _assert_active(reference[0], _call_p9(target_op, active), active, fixture)
    finally:
        fixture.close()
    print("P9_RUNTIME_PASS case=full runtime_enable_0=1 runtime_enable_1=1", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001 -- emit a compact remote diagnostic.
        print(
            f"P9_RUNTIME_FAIL type={type(error).__name__} text={error}",
            flush=True,
        )
        raise
