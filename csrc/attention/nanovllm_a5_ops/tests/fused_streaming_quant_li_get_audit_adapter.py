#!/usr/bin/env python3
"""Target runner for the P4 resident-threshold candidate-safety harness.

The device-side path exercised here has the same two scans the product path
needs: a resident-only C8 scan produces an exact sortable Top-2048 threshold,
then one full raw-Q/K scan records threshold-qualified cold sources in a fixed
``[R,S]`` candidate map.  It never posts H2D in this audit mode.

The full official C8 result remains the only final-Top-K authority.  The
threshold map is only a speculative-transport hint: final ExactTopK plus
GuaranteedRecall owns losslessness.  This audit additionally builds a
score-GM publisher variant so every Scalar-read receipt score can be compared
with the exact score-GM values C8 subsequently consumes for Top-K.  The
source-derived P3 clone's own Top-K stays diagnostic and cannot route QSFA.
"""

from __future__ import annotations

import os
import sys
from time import monotonic
from pathlib import Path
from typing import Any

import torch

from streaming_quant_li_get_target_adapter import (
    BLOCK_SIZE,
    BATCH_SIZE,
    HOOK_CONTROL_MODE_ALL_BUSY,
    INT64_MAX,
    INLINE_MODE_ENV,
    PUBLICATION_MODE_ENV,
    RUNNER_EXPECTED_HEAD_ENV,
    REFERENCE_OPP_ENV,
    SOURCE_TOKENS,
    SPARSE_COUNT,
    TARGET_OPP_ENV,
    TARGET_EXPECTED_OPP_SOURCE_HEAD_ENV,
    TILE_RECEIPT_CONTRACT_VERSION,
    TILE_RECEIPT_SENTINEL,
    TILE_SCORE_WIDTH,
    _bind_isolated_target_opp,
    _build_official_inputs,
    _call_official,
    _assert_publication_contract,
    _make_publication_buffers,
    _make_no_post_transaction,
    _official_metadata,
    _expected_score_source_lens,
    _expected_opp_source_head,
    _prepend_unique_path,
    _require_a5_npu,
    _require_source_matched_official_c8_ops,
    _read_target_provenance,
    _resolve_inline_mode,
    _resolve_publication_mode,
    _resolve_single_opp,
)


RESIDENT_SEED_OPP_ENV = "NANOVLLM_A5_RESIDENT_SEED_OPP"
RESIDENT_SEED_OPAPI_ENV = "NANOVLLM_A5_RESIDENT_SEED_OPAPI_LIB"
# 24 C8 pages provide 3072 resident keys: enough to define an exact Top-2048
# threshold while deliberately leaving a meaningful cold suffix in the 4096
# / 3968-token full source rows.  Keeping whole pages makes the resident
# block-table ABI identical to the future runtime representation.
RESIDENT_BLOCKS = 24


def _resolve_resident_seed_opp(root_value: str | None) -> tuple[Path, Path]:
    """Resolve exactly one independently built ResidentSeed OPP."""
    if not root_value:
        raise RuntimeError(
            f"set {RESIDENT_SEED_OPP_ENV} to the isolated A5C8ResidentSeed OPP root"
        )
    root = Path(root_value).expanduser().resolve()
    candidates = [root / "op_api" / "lib" / "libcust_opapi.so"]
    candidates.extend(root.glob("vendors/*/op_api/lib/libcust_opapi.so"))
    opapis = tuple(sorted({item.resolve() for item in candidates if item.is_file()}))
    if len(opapis) != 1:
        raise RuntimeError(
            "expected exactly one A5C8ResidentSeed libcust_opapi.so below "
            f"{root}, found={len(opapis)}"
        )
    opapi = opapis[0]
    return opapi, opapi.parents[2]


def _cpu_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(left.detach().cpu().contiguous(), right.detach().cpu().contiguous()))


def _audit_receipt(
    result: dict[str, Any],
    transaction: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Validate audit's side-effect boundary before decoding output 2."""
    marker = int(result["candidate_marker"])
    sentinel = int(result["audit_lease_sentinel"])
    receipt_kind = str(result["audit_receipt_kind"])
    source_map = result["candidate_source_to_slot"].detach().cpu().contiguous()
    queue_leases = result["queue_leases"].detach().cpu().contiguous()
    ready = result["private_ready_table"].detach().cpu().contiguous()
    arena_head = result["private_arena_head"].detach().cpu().contiguous()
    packed = result["private_packed_output"].detach().cpu().contiguous()
    telemetry = result["telemetry"].detach().cpu().contiguous()

    if not bool(torch.all(queue_leases == sentinel)):
        raise RuntimeError("P4 audit changed the all-sentinel queue lease table")
    if int(torch.count_nonzero(ready).item()) != 0:
        raise RuntimeError("P4 audit wrote the private ready table")
    if int(torch.count_nonzero(arena_head).item()) != 0:
        raise RuntimeError("P4 audit allocated a private HBM arena slot")
    if not bool(torch.all(packed == 0xA5)):
        raise RuntimeError("P4 audit wrote the private packed payload arena")
    if int(torch.count_nonzero(telemetry).item()) != 0:
        raise RuntimeError("P4 audit wrote queue-owned transport telemetry")
    if receipt_kind == "candidate":
        candidate_count = int(torch.count_nonzero(source_map == marker).item())
        if candidate_count <= 0:
            raise RuntimeError("P4 audit produced no score-time candidate markers")
        if bool(torch.any((source_map != -1) & (source_map != marker))):
            raise RuntimeError("P4 audit produced a non-candidate private-slot value")
    elif receipt_kind == "score":
        candidate_count = -1
    else:
        raise RuntimeError(f"P4 audit has an unknown receipt kind: {receipt_kind!r}")
    return {
        "candidate_count": candidate_count,
        "candidate_overflow": 0,
        "candidate_marker": marker,
        "lease_sentinel": sentinel,
        "receipt_kind": receipt_kind,
    }


def _assert_transaction_output_aliases(
    result: dict[str, Any], transaction: dict[str, torch.Tensor]
) -> None:
    """Require graph-visible receipts to reuse their transaction allocations."""
    output_names = {
        "private_source_to_slot": "candidate_source_to_slot",
        "private_ready_table": "private_ready_table",
        "private_arena_head": "private_arena_head",
        "private_packed_output": "private_packed_output",
        "queue_leases": "queue_leases",
    }
    for transaction_name, result_name in output_names.items():
        expected = transaction[transaction_name]
        actual = result[result_name]
        if int(expected.data_ptr()) != int(actual.data_ptr()):
            raise RuntimeError(
                "P4 audit returned a copied transaction receipt: "
                f"name={transaction_name} "
                f"input=0x{int(expected.data_ptr()):x} "
                f"output=0x{int(actual.data_ptr()):x}"
            )


def _assert_publication_output_aliases(
    result: dict[str, Any], publication: dict[str, torch.Tensor]
) -> None:
    """Require the dormant publication buffers to preserve physical aliases."""
    for name in ("score_workspace", "tile_receipts"):
        expected = publication[name]
        actual = result[name]
        if int(expected.data_ptr()) != int(actual.data_ptr()):
            raise RuntimeError(
                "P4 audit returned a copied publication buffer: "
                f"name={name} input=0x{int(expected.data_ptr()):x} "
                f"output=0x{int(actual.data_ptr()):x}"
            )


def _decode_published_score_mirror(
    publication: dict[str, torch.Tensor], score_source_lens: torch.Tensor
) -> dict[str, Any]:
    """Decode the exact uint16 score-GM tiles from publisher receipts.

    `A5StreamingQuantLiGet` writes the same score-GM that its later
    `ProcessTopK` consumes.  Publisher records describe each V->MTE3 tile
    copy after it has been enqueued, so this is an independent host-side view
    of the score domain; it never reuses the Scalar audit receipt values.
    """
    score_words = publication["score_workspace"].detach().cpu().contiguous()
    tile_receipts = publication["tile_receipts"].detach().cpu().contiguous()
    epoch = int(publication["streaming_epoch"].detach().cpu().item())
    if score_words.dtype != torch.int32:
        raise RuntimeError(
            f"P4 score workspace must be int32 carrier, got {score_words.dtype}"
        )
    if tile_receipts.ndim != 2 or tile_receipts.shape[1] != 16:
        raise RuntimeError(
            "P4 tile receipt ABI must be [slots,16], got "
            f"shape={tuple(tile_receipts.shape)}"
        )
    if score_source_lens.numel() != BATCH_SIZE:
        raise RuntimeError(
            "P4 score-source length ABI has unexpected shape: "
            f"{tuple(score_source_lens.shape)}"
        )

    # Every int32 carrier stores two exact sortable uint16 score values.  The
    # publisher's byte offsets are relative to score_workspace's GM base.
    score_u16 = score_words.view(torch.uint16).reshape(-1)
    mirror = torch.full(
        (BATCH_SIZE, int(score_source_lens.max().item())),
        -1,
        dtype=torch.int64,
    )
    written = torch.zeros_like(mirror, dtype=torch.bool)
    descriptor_errors: list[str] = []
    duplicate_count = 0
    interval_overlap_count = 0
    active_records = 0
    ready_records = 0
    zero_valid_records = 0
    # A descriptor proves both a logical coordinate and a physical score-GM
    # interval.  Keep the latter explicit: two distinct source tiles must not
    # accidentally read the same legal workspace range.
    intervals: list[tuple[int, int, int]] = []
    source_lens = score_source_lens.detach().cpu().contiguous()
    tiles_per_row = (SOURCE_TOKENS + TILE_SCORE_WIDTH - 1) // TILE_SCORE_WIDTH
    expected_active_records = sum(
        (int(source_lens[row].item()) + TILE_SCORE_WIDTH - 1) // TILE_SCORE_WIDTH
        for row in range(BATCH_SIZE)
    )

    def record_error(message: str) -> None:
        if len(descriptor_errors) < 8:
            descriptor_errors.append(message)

    for slot, record in enumerate(tile_receipts):
        if int(record[8].item()) != epoch:
            if not bool(torch.all(record == TILE_RECEIPT_SENTINEL).item()):
                record_error(f"slot={slot},unready_record_not_sentinel=1")
            continue
        ready_records += 1
        byte_offset = int(record[0].item())
        row = int(record[1].item())
        source_base = int(record[2].item())
        valid = int(record[3].item())
        source_stride = int(record[4].item())
        expected_valid = -1
        expected_slot = -1
        if 0 <= row < BATCH_SIZE and source_base >= 0 and \
                source_base % TILE_SCORE_WIDTH == 0 and \
                source_base < SOURCE_TOKENS:
            expected_valid = max(
                0,
                min(
                    TILE_SCORE_WIDTH,
                    int(source_lens[row].item()) - source_base,
                ),
            )
            expected_slot = row * tiles_per_row + source_base // TILE_SCORE_WIDTH
        descriptor_common_ok = (
            slot == expected_slot
            and expected_valid >= 0
            and valid == expected_valid
            and source_stride == SOURCE_TOKENS
            and int(record[7].item()) == TILE_RECEIPT_CONTRACT_VERSION
            and bool(torch.all(record[9:] == 0).item())
            and byte_offset >= 0
            and byte_offset % 2 == 0
            and (byte_offset // 2) + valid <= score_u16.numel()
        )
        # A shorter row can legally produce a *fixed-grid* ready record with
        # valid=0.  It carries no score data and is accepted only when its
        # complete descriptor is otherwise exact; it can never fill the
        # mirror or satisfy a missing score coordinate.
        if expected_valid == 0 and descriptor_common_ok:
            zero_valid_records += 1
            continue
        if not (expected_valid > 0 and descriptor_common_ok):
            record_error(
                "slot=%d,row=%d,source=%d,valid=%d,expected_valid=%d,"
                "expected_slot=%d,stride=%d,contract=%d,offset=%d" % (
                    slot,
                    row,
                    source_base,
                    valid,
                    expected_valid,
                    expected_slot,
                    source_stride,
                    int(record[7].item()),
                    byte_offset,
                )
            )
            continue
        active_records += 1
        target = written[row, source_base:source_base + valid]
        duplicate_count += int(torch.count_nonzero(target).item())
        offset_u16 = byte_offset // 2
        interval_begin = offset_u16
        interval_end = offset_u16 + valid
        for existing_begin, existing_end, existing_slot in intervals:
            if interval_begin < existing_end and existing_begin < interval_end:
                interval_overlap_count += 1
                record_error(
                    "score_interval_overlap slot=%d other_slot=%d range=[%d,%d) "
                    "other=[%d,%d)" % (
                        slot,
                        existing_slot,
                        interval_begin,
                        interval_end,
                        existing_begin,
                        existing_end,
                    )
                )
        intervals.append((interval_begin, interval_end, slot))
        mirror[row, source_base:source_base + valid] = score_u16[
            offset_u16:offset_u16 + valid
        ].to(torch.int64)
        written[row, source_base:source_base + valid] = True

    expected = torch.zeros_like(written)
    for row in range(BATCH_SIZE):
        expected[row, :int(score_source_lens[row].item())] = True
    missing_count = int(torch.count_nonzero(expected & ~written).item())
    outside_count = int(torch.count_nonzero(written & ~expected).item())
    return {
        "scores": mirror,
        "written": written,
        "active_records": active_records,
        "ready_records": ready_records,
        "expected_active_records": expected_active_records,
        "zero_valid_records": zero_valid_records,
        "missing_count": missing_count,
        "outside_count": outside_count,
        "duplicate_count": duplicate_count,
        "interval_overlap_count": interval_overlap_count,
        "descriptor_errors": descriptor_errors,
    }


def _assert_clone_source_domain(
    clone_indices: torch.Tensor, score_source_lens: torch.Tensor
) -> None:
    """Reject an apparently equal Top-K tensor with an invalid S-domain."""
    indices = clone_indices.detach().cpu().contiguous().reshape(BATCH_SIZE, SPARSE_COUNT)
    source_lens = score_source_lens.detach().cpu().contiguous().reshape(BATCH_SIZE)
    for row in range(BATCH_SIZE):
        source_len = int(source_lens[row].item())
        if source_len <= 0:
            raise RuntimeError(
                f"P4 audit has a nonpositive score-source length: row={row} length={source_len}"
            )
        invalid = (indices[row] != -1) & (
            (indices[row] < 0) | (indices[row] >= source_len)
        )
        if bool(torch.any(invalid)):
            first = int(torch.nonzero(invalid, as_tuple=False)[0].item())
            raise RuntimeError(
                "P4 audit clone Top-K contains an out-of-domain score source: "
                f"row={row} topk_offset={first} source={int(indices[row, first].item())} "
                f"length={source_len}"
            )


def _build_resident_index_view(
    inputs: dict[str, torch.Tensor],
    metadata_op: Any,
) -> dict[str, torch.Tensor]:
    """Build an independent resident-index view for this isolated test.

    Production will maintain this view incrementally with HBM residency.  The
    validation fixture instead clones the source FP8 index store and exposes
    its first 24 *logical* pages through a separate block table.  The clone is
    important: passing the packed QSFA payload or reusing a host reconstruction
    would test a different score domain.  The logical resident prefix leaves a
    cold suffix, so the result is not the old all-cold/zero-threshold case.
    """
    resident_tokens = RESIDENT_BLOCKS * BLOCK_SIZE
    if resident_tokens < SPARSE_COUNT:
        raise RuntimeError("resident fixture must contain at least Top-2048 keys")
    # Do not reach back to the source fixture's module-level length constant:
    # this helper receives the concrete FP8/key-table allocations that both
    # the official reference and ResidentSeed will consume.  Deriving capacity
    # here keeps the guard correct if a target test changes its source length.
    block_table = inputs["block_table"]
    if block_table.ndim != 2:
        raise RuntimeError(
            "resident fixture block_table must be [batch, source_blocks], "
            f"got shape={tuple(block_table.shape)}"
        )
    source_blocks = int(block_table.shape[1])
    if source_blocks <= RESIDENT_BLOCKS:
        raise RuntimeError("resident fixture must leave at least one cold page")
    key_blocks = int(inputs["key"].shape[0])
    if key_blocks < source_blocks:
        raise RuntimeError(
            "resident fixture key allocation has fewer blocks than its block table: "
            f"key_blocks={key_blocks} source_blocks={source_blocks}"
        )

    resident_lengths = torch.full(
        (BATCH_SIZE,), resident_tokens, dtype=torch.int32, device=inputs["query"].device
    )
    resident_key = inputs["key"].clone(memory_format=torch.preserve_format)
    resident_scale = inputs["key_dequant_scale"].clone(memory_format=torch.preserve_format)
    resident_block_table = block_table[:, :RESIDENT_BLOCKS].clone().contiguous()
    resident_metadata = metadata_op(
        num_heads_q=64,
        num_heads_k=1,
        head_dim=128,
        query_quant_mode=0,
        key_quant_mode=0,
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_key=resident_lengths,
        batch_size=BATCH_SIZE,
        max_seqlen_q=1,
        max_seqlen_k=resident_tokens,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        cmp_ratio=1,
        device=str(inputs["query"].device),
    )
    if resident_metadata.dtype != torch.int32 or not resident_metadata.is_contiguous():
        raise RuntimeError("resident C8 metadata has an invalid ABI")
    return {
        "resident_index_key": resident_key,
        "resident_index_key_dequant_scale": resident_scale,
        "resident_seq_lengths": resident_lengths,
        "resident_index_block_table": resident_block_table,
        "metadata": resident_metadata,
    }


def _bind_resident_seed_opp(opapi: Any) -> None:
    """Load the current-source ResidentSeed bridge without touching P3.

    This test adapter is run directly from the source tree.  Its bridge is an
    in-place extension below the sibling ``torch_extension`` directory, rather
    than a globally installed package.  Put that exact source-matched root
    first on ``sys.path`` so a shell's inherited ``PYTHONPATH`` cannot select
    a stale package or omit the bridge after the reference import.
    """
    os.environ[RESIDENT_SEED_OPAPI_ENV] = str(opapi)
    bridge_root = (
        Path(__file__).resolve().parents[1]
        / "a5_c8_resident_seed"
        / "torch_extension"
    )
    package_dir = bridge_root / "nanovllm_a5_c8_resident_seed"
    if not (package_dir / "__init__.py").is_file():
        raise RuntimeError(
            "ResidentSeed bridge package is missing from the current source tree: "
            f"{package_dir}"
        )
    bridge_root_text = str(bridge_root)
    if bridge_root_text in sys.path:
        sys.path.remove(bridge_root_text)
    sys.path.insert(0, bridge_root_text)
    import nanovllm_a5_c8_resident_seed  # noqa: F401, PLC0415

    print(
        "P4_RESIDENT_SEED_BRIDGE "
        f"source_root={bridge_root} path_rank={sys.path.index(bridge_root_text)} "
        f"opapi={opapi}",
        flush=True,
    )


def _launch_resident_seed(
    inputs: dict[str, torch.Tensor], resident: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    namespace = getattr(torch.ops, "a5_c8_resident_seed", None)
    seed_op = (
        getattr(namespace, "npu_a5_c8_resident_seed", None) if namespace else None
    )
    if seed_op is None:
        raise RuntimeError("isolated A5C8ResidentSeed bridge did not register")
    result = seed_op(
        query=inputs["query"],
        resident_index_key=resident["resident_index_key"],
        weights=inputs["weights"],
        query_dequant_scale=inputs["query_dequant_scale"],
        resident_index_key_dequant_scale=resident["resident_index_key_dequant_scale"],
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        resident_seq_lengths=resident["resident_seq_lengths"],
        resident_index_block_table=resident["resident_index_block_table"],
        metadata=resident["metadata"],
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        cmp_ratio=1,
        return_values=False,
        stride=int(resident["resident_index_key"].stride(0)),
        scale_stride=int(
            resident["resident_index_key_dequant_scale"].stride(0)
        ),
    )
    if not isinstance(result, tuple) or len(result) != 4:
        raise RuntimeError("A5C8ResidentSeed returned an unexpected ABI")
    sparse_indices, sparse_values, resident_tau, tau_valid = result
    row_count = BATCH_SIZE
    if (
        sparse_indices.dtype != torch.int32
        or tuple(sparse_indices.shape) != (BATCH_SIZE, 1, SPARSE_COUNT)
        or sparse_values.dtype != torch.float32
        or resident_tau.dtype != torch.int32
        or tuple(resident_tau.shape) != (row_count,)
        or tau_valid.dtype != torch.int32
        or tuple(tau_valid.shape) != (row_count,)
    ):
        raise RuntimeError(
            "A5C8ResidentSeed output ABI mismatch: "
            f"indices={tuple(sparse_indices.shape)}/{sparse_indices.dtype} "
            f"values={tuple(sparse_values.shape)}/{sparse_values.dtype} "
            f"tau={tuple(resident_tau.shape)}/{resident_tau.dtype} "
            f"valid={tuple(tau_valid.shape)}/{tau_valid.dtype}"
        )
    return sparse_indices, sparse_values, resident_tau, tau_valid


def _call_official_resident_reference(
    official: Any,
    inputs: dict[str, torch.Tensor],
    resident: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the copied seed's selection domain against official C8."""
    result = official(
        query=inputs["query"],
        key=resident["resident_index_key"],
        weights=inputs["weights"],
        query_dequant_scale=inputs["query_dequant_scale"],
        key_dequant_scale=resident["resident_index_key_dequant_scale"],
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_key=resident["resident_seq_lengths"],
        block_table=resident["resident_index_block_table"],
        metadata=resident["metadata"],
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        cmp_ratio=1,
        return_value=False,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("official resident C8 call returned an unexpected ABI")
    sparse_indices, sparse_values = result
    if not isinstance(sparse_indices, torch.Tensor) or not isinstance(
        sparse_values, torch.Tensor
    ):
        raise RuntimeError("official resident C8 returned non-tensor outputs")
    return sparse_indices, sparse_values


def _install_resident_snapshot(
    transaction: dict[str, torch.Tensor], resident: dict[str, torch.Tensor]
) -> None:
    """Mark exactly the seed's logical resident view as already in HBM.

    The slot snapshot is addressed in the runtime's physical object domain,
    not in the score-source domain.  Resolve every resident logical source
    through ``row -> req_pool_entries -> kv_object_map`` here, so the audit
    exercises the same indirection used by the device hook.
    """
    lengths = resident["resident_seq_lengths"].detach().cpu().tolist()
    slots = transaction["kv_object_slots_snapshot"].detach().cpu().contiguous()
    pools = transaction["req_pool_entries"].detach().cpu().contiguous()
    object_map = transaction["kv_object_map"].detach().cpu().contiguous()
    object_lens = transaction["kv_object_lens"].detach().cpu().contiguous()
    for row, length in enumerate(lengths):
        pool_row = int(pools[row].item())
        object_len = int(object_lens[row].item())
        resident_length = int(length)
        if pool_row < 0 or pool_row >= slots.shape[0]:
            raise RuntimeError(f"resident snapshot has invalid pool row {pool_row}")
        if resident_length > object_len:
            raise RuntimeError(
                "resident fixture exceeds the active KV-object prefix: "
                f"row={row} resident_length={resident_length} object_len={object_len}"
            )
        mapped_objects = object_map[pool_row, :resident_length]
        if bool(torch.any((mapped_objects < 0) | (mapped_objects >= object_len))):
            raise RuntimeError(
                "resident snapshot contains an invalid logical-source to KV-object map: "
                f"row={row} pool_row={pool_row}"
            )
        if int(torch.unique(mapped_objects).numel()) != resident_length:
            raise RuntimeError(
                "resident snapshot mapping is not one-to-one over its resident prefix: "
                f"row={row}"
            )
        slots[pool_row, mapped_objects] = 0
    transaction["kv_object_slots_snapshot"].copy_(
        slots.to(transaction["kv_object_slots_snapshot"].device)
    )


def _block_histogram(mask: torch.Tensor, *, block_size: int = BLOCK_SIZE) -> str:
    """Return a compact, deterministic ``block:count`` summary for a row mask."""
    parts: list[str] = []
    for block in range((int(mask.numel()) + block_size - 1) // block_size):
        start = block * block_size
        count = int(torch.count_nonzero(mask[start:start + block_size]).item())
        if count:
            parts.append(f"{block}:{count}")
    return ",".join(parts) if parts else "none"


def _source_receipt(
    sources: torch.Tensor,
    *,
    pool_row: int,
    object_map: torch.Tensor,
    slots: torch.Tensor,
    receipt_u32: torch.Tensor | None = None,
    receipt_encoding: dict[str, int] | None = None,
    limit: int = 12,
) -> str:
    """Return bounded mapping/score evidence that is practical to paste back."""
    selected = torch.nonzero(sources, as_tuple=False).flatten().tolist()
    entries: list[str] = []
    for source in selected[:limit]:
        kv_object = int(object_map[pool_row, source].item())
        slot = int(slots[pool_row, kv_object].item())
        prefix = f"{source}(t{source // BLOCK_SIZE}i{source % BLOCK_SIZE})>{kv_object}@{slot}"
        if receipt_u32 is not None and receipt_encoding is not None:
            raw = int(receipt_u32[source].item()) & 0xFFFFFFFF
            if (raw & receipt_encoding["tag_mask"]) == receipt_encoding["tag"]:
                entries.append(
                    f"{prefix}#s={raw & receipt_encoding['score_mask']}"
                    f"/tp={int(bool(raw & receipt_encoding['threshold_pass_bit']))}"
                    f"/mv={int(bool(raw & receipt_encoding['mapping_valid_bit']))}"
                    f"/cold={int(bool(raw & receipt_encoding['cold_bit']))}"
                    f"/unmapped={int(bool(raw & receipt_encoding['source_unmapped_bit']))}"
                    f"/cand={int(bool(raw & receipt_encoding['candidate_bit']))}"
                )
            else:
                entries.append(f"{prefix}#unwritten=1/raw=0x{raw:08x}")
        else:
            entries.append(prefix)
    suffix = "" if len(selected) <= limit else f",+{len(selected) - limit}"
    return "[" + ",".join(entries) + suffix + "]"


def _collect_authoritative_cold_coverage(
    official_indices: torch.Tensor,
    candidate_map: torch.Tensor,
    transaction: dict[str, torch.Tensor],
    score_source_lens: torch.Tensor,
    marker: int,
    resident_threshold: torch.Tensor,
    receipt_kind: str,
    receipt_encoding: dict[str, int] | None,
    score_mirror: dict[str, Any],
) -> dict[str, Any]:
    """Collect threshold safety and independent score-provenance evidence.

    Official C8 remains the unique selection authority.  The candidate map is
    a score-time transport decision, so this function reports its coverage
    separately from final Top-K equivalence.  Crucially, the scalar audit
    receipt is compared with an independently published score-GM mirror rather
    than only rechecking its own packed bits.  It returns every failure as a
    bounded fact so one device launch can distinguish a score-seam error from
    a threshold-policy miss.
    """
    official = official_indices.detach().cpu().contiguous().reshape(BATCH_SIZE, SPARSE_COUNT)
    candidates = candidate_map.detach().cpu().contiguous()
    slots = transaction["kv_object_slots_snapshot"].detach().cpu().contiguous()
    pools = transaction["req_pool_entries"].detach().cpu().contiguous()
    object_map = transaction["kv_object_map"].detach().cpu().contiguous()
    object_lens = transaction["kv_object_lens"].detach().cpu().contiguous()
    source_lens = score_source_lens.detach().cpu().contiguous()
    thresholds = resident_threshold.detach().cpu().contiguous()
    required_mask = torch.zeros_like(candidates, dtype=torch.bool)
    official_selected_mask = torch.zeros_like(candidates, dtype=torch.bool)
    eligible_cold_mask = torch.zeros_like(candidates, dtype=torch.bool)
    mapping_valid_mask = torch.zeros_like(candidates, dtype=torch.bool)
    mapping_difference_mask = torch.zeros_like(candidates, dtype=torch.bool)
    valid_domain_mask = torch.zeros_like(candidates, dtype=torch.bool)
    outside_domain_mask = torch.zeros_like(candidates, dtype=torch.bool)
    selected = 0
    mapping_discriminator_selected = 0
    mapping_discriminator_sources = 0
    for row in range(BATCH_SIZE):
        valid = int(source_lens[row].item())
        pool_row = int(pools[row].item())
        object_len = int(object_lens[row].item())
        if pool_row < 0 or pool_row >= slots.shape[0]:
            raise RuntimeError(f"invalid request pool entry: row={row} pool_row={pool_row}")
        if object_len <= 0 or object_len > slots.shape[1]:
            raise RuntimeError(
                f"invalid KV-object length: row={row} object_len={object_len}"
            )
        valid_domain_mask[row, :valid] = True
        mapped_active = object_map[pool_row, :valid]
        if bool(torch.any((mapped_active < 0) | (mapped_active >= object_len))):
            raise RuntimeError(
                "P4 audit has an invalid active logical-source to KV-object map: "
                f"row={row} pool_row={pool_row} source_len={valid}"
            )
        correct_cold = slots[pool_row, mapped_active] < 0
        legacy_identity_cold = slots[row, :valid] < 0
        eligible_cold_mask[row, :valid] = correct_cold
        mapping_valid_mask[row, :valid] = True
        mapping_difference_mask[row, :valid] = (
            correct_cold != legacy_identity_cold
        )
        mapping_discriminator_sources += int(
            torch.count_nonzero(correct_cold != legacy_identity_cold).item()
        )
        for source in official[row].tolist():
            if source < 0:
                continue
            if source >= valid:
                raise RuntimeError(
                    "official C8 Top-K contains an invalid source: "
                    f"row={row} source={source} valid={valid}"
                )
            selected += 1
            official_selected_mask[row, source] = True
            kv_object = int(object_map[pool_row, source].item())
            if kv_object < 0 or kv_object >= object_len:
                raise RuntimeError(
                    "P4 audit has no valid logical-source to KV-object mapping: "
                    f"row={row} pool_row={pool_row} source={source} "
                    f"kv_object={kv_object} object_len={object_len}"
                )
            is_cold = bool(correct_cold[source].item())
            if is_cold != bool(legacy_identity_cold[source].item()):
                mapping_discriminator_selected += 1
            if is_cold:
                required_mask[row, source] = True

    score_receipt = receipt_kind == "score"
    receipt_u32: torch.Tensor | None = None
    written_mask = torch.zeros_like(candidates, dtype=torch.bool)
    score_values = torch.zeros_like(candidates, dtype=torch.int64)
    threshold_bit_mask = torch.zeros_like(candidates, dtype=torch.bool)
    mapping_bit_mask = torch.zeros_like(candidates, dtype=torch.bool)
    cold_bit_mask = torch.zeros_like(candidates, dtype=torch.bool)
    source_unmapped_bit_mask = torch.zeros_like(candidates, dtype=torch.bool)
    candidate_bit_mask = torch.zeros_like(candidates, dtype=torch.bool)
    unwritten_mask = torch.zeros_like(candidates, dtype=torch.bool)
    malformed_mask = torch.zeros_like(candidates, dtype=torch.bool)
    threshold_bit_mismatch_mask = torch.zeros_like(candidates, dtype=torch.bool)
    mapping_bit_mismatch_mask = torch.zeros_like(candidates, dtype=torch.bool)
    cold_bit_mismatch_mask = torch.zeros_like(candidates, dtype=torch.bool)
    source_unmapped_bit_mismatch_mask = torch.zeros_like(candidates, dtype=torch.bool)
    candidate_bit_mismatch_mask = torch.zeros_like(candidates, dtype=torch.bool)
    mirror_scores = score_mirror["scores"].detach().cpu().contiguous()
    mirror_written = score_mirror["written"].detach().cpu().contiguous()
    if tuple(mirror_scores.shape) != tuple(candidates.shape) or \
            tuple(mirror_written.shape) != tuple(candidates.shape):
        raise RuntimeError(
            "P4 score-GM mirror shape differs from the score receipt domain: "
            f"mirror_scores={tuple(mirror_scores.shape)} "
            f"mirror_written={tuple(mirror_written.shape)} "
            f"receipt={tuple(candidates.shape)}"
        )
    mirror_score_mismatch_mask = torch.zeros_like(candidates, dtype=torch.bool)
    candidate_from_mirror_mismatch_mask = torch.zeros_like(
        candidates, dtype=torch.bool
    )
    selected_below_tau_mask = torch.zeros_like(candidates, dtype=torch.bool)
    resident_tau_invariant_by_row: list[dict[str, int]] = []
    tau_valid_by_row = transaction["resident_threshold_valid"].detach().cpu().contiguous()
    if score_receipt:
        if receipt_encoding is None:
            raise RuntimeError("score receipt has no encoding specification")
        receipt_u32 = candidates.to(torch.int64) & 0xFFFFFFFF
        written_mask = (
            receipt_u32 & receipt_encoding["tag_mask"]
        ) == receipt_encoding["tag"]
        score_values = receipt_u32 & receipt_encoding["score_mask"]
        threshold_bit_mask = written_mask & (
            (receipt_u32 & receipt_encoding["threshold_pass_bit"]) != 0
        )
        mapping_bit_mask = written_mask & (
            (receipt_u32 & receipt_encoding["mapping_valid_bit"]) != 0
        )
        cold_bit_mask = written_mask & (
            (receipt_u32 & receipt_encoding["cold_bit"]) != 0
        )
        source_unmapped_bit_mask = written_mask & (
            (receipt_u32 & receipt_encoding["source_unmapped_bit"]) != 0
        )
        candidate_bit_mask = written_mask & (
            (receipt_u32 & receipt_encoding["candidate_bit"]) != 0
        )
        unwritten_mask = valid_domain_mask & ~written_mask
        malformed_mask = valid_domain_mask & ~written_mask & (candidates != -1)
        outside_domain_mask = written_mask & ~valid_domain_mask
        expected_threshold_pass = torch.zeros_like(candidates, dtype=torch.bool)
        expected_threshold_from_mirror = torch.zeros_like(
            candidates, dtype=torch.bool
        )
        for row in range(BATCH_SIZE):
            valid = int(source_lens[row].item())
            tau = int(thresholds[row].item()) & 0xFFFF
            tau_valid = int(tau_valid_by_row[row].item()) == 1
            expected_threshold_pass[row, :valid] = (
                score_values[row, :valid] >= tau
            ) & tau_valid
            expected_threshold_from_mirror[row, :valid] = (
                mirror_scores[row, :valid] >= tau
            ) & tau_valid
            prefix_len = min(RESIDENT_BLOCKS * BLOCK_SIZE, valid)
            prefix_scores = mirror_scores[row, :prefix_len]
            gt_count = int(torch.count_nonzero(prefix_scores > tau).item())
            ge_count = int(torch.count_nonzero(prefix_scores >= tau).item())
            resident_tau_invariant_by_row.append(
                {
                    "row": row,
                    "tau": tau,
                    "prefix_len": prefix_len,
                    "gt_count": gt_count,
                    "ge_count": ge_count,
                    "threshold_valid": int(tau_valid),
                    "rank_valid": int(
                        tau_valid and gt_count < SPARSE_COUNT <= ge_count
                    ),
                }
            )
        expected_source_unmapped = valid_domain_mask
        expected_candidate = (
            expected_threshold_pass & mapping_valid_mask & eligible_cold_mask &
            expected_source_unmapped
        )
        expected_candidate_from_mirror = (
            expected_threshold_from_mirror & mapping_valid_mask &
            eligible_cold_mask & expected_source_unmapped
        )
        threshold_bit_mismatch_mask = (
            valid_domain_mask & written_mask &
            (threshold_bit_mask != expected_threshold_pass)
        )
        mapping_bit_mismatch_mask = (
            valid_domain_mask & written_mask &
            (mapping_bit_mask != mapping_valid_mask)
        )
        cold_bit_mismatch_mask = (
            valid_domain_mask & written_mask &
            (cold_bit_mask != eligible_cold_mask)
        )
        source_unmapped_bit_mismatch_mask = (
            valid_domain_mask & written_mask &
            (source_unmapped_bit_mask != expected_source_unmapped)
        )
        candidate_bit_mismatch_mask = (
            valid_domain_mask & written_mask &
            (candidate_bit_mask != expected_candidate)
        )
        mirror_score_mismatch_mask = (
            valid_domain_mask & written_mask & mirror_written &
            (score_values != mirror_scores)
        )
        candidate_from_mirror_mismatch_mask = (
            valid_domain_mask & written_mask & mirror_written &
            (candidate_bit_mask != expected_candidate_from_mirror)
        )
        for row in range(BATCH_SIZE):
            valid = int(source_lens[row].item())
            tau = int(thresholds[row].item()) & 0xFFFF
            tau_valid = int(tau_valid_by_row[row].item()) == 1
            selected_below_tau_mask[row, :valid] = (
                official_selected_mask[row, :valid] & mirror_written[row, :valid] &
                (mirror_scores[row, :valid] < tau) & tau_valid
            )
        candidate_mask = candidate_bit_mask
    else:
        candidate_mask = candidates == marker
        outside_domain_mask = candidate_mask & ~valid_domain_mask
    invalid_candidate_mask = candidate_mask & ~eligible_cold_mask
    invalid_candidate_count = int(torch.count_nonzero(invalid_candidate_mask).item())
    candidate_discriminator_count = int(
        torch.count_nonzero(candidate_mask & mapping_difference_mask).item()
    )
    covered_mask = candidate_mask & required_mask
    missing_mask = required_mask & ~candidate_mask
    covered = int(torch.count_nonzero(covered_mask).item())
    required = int(torch.count_nonzero(required_mask).item())
    missing = int(torch.count_nonzero(missing_mask).item())
    candidate_count = int(torch.count_nonzero(candidate_mask).item())
    false_positive = int(torch.count_nonzero(candidate_mask & ~required_mask).item())
    cold_capacity = 0
    for row in range(BATCH_SIZE):
        pool_row = int(pools[row].item())
        valid = int(source_lens[row].item())
        mapped_active = object_map[pool_row, :valid]
        cold_capacity += int(
            torch.count_nonzero(slots[pool_row, mapped_active] < 0).item()
        )
    outside_domain_count = int(torch.count_nonzero(outside_domain_mask).item())
    receipt_written_count = int(torch.count_nonzero(written_mask & valid_domain_mask).item())
    receipt_unwritten_count = int(torch.count_nonzero(unwritten_mask).item())
    receipt_malformed_count = int(torch.count_nonzero(malformed_mask).item())
    threshold_bit_mismatch_count = int(
        torch.count_nonzero(threshold_bit_mismatch_mask).item()
    )
    mapping_bit_mismatch_count = int(
        torch.count_nonzero(mapping_bit_mismatch_mask).item()
    )
    cold_bit_mismatch_count = int(torch.count_nonzero(cold_bit_mismatch_mask).item())
    source_unmapped_bit_mismatch_count = int(
        torch.count_nonzero(source_unmapped_bit_mismatch_mask).item()
    )
    candidate_bit_mismatch_count = int(
        torch.count_nonzero(candidate_bit_mismatch_mask).item()
    )
    mirror_score_mismatch_count = int(
        torch.count_nonzero(mirror_score_mismatch_mask).item()
    )
    candidate_from_mirror_mismatch_count = int(
        torch.count_nonzero(candidate_from_mirror_mismatch_mask).item()
    )
    selected_below_tau_count = int(
        torch.count_nonzero(selected_below_tau_mask).item()
    )
    resident_tau_invalid_count = sum(
        1 for item in resident_tau_invariant_by_row
        if item["threshold_valid"] == 0
    )
    resident_tau_rank_invariant_count = sum(
        1 for item in resident_tau_invariant_by_row if item["rank_valid"] == 0
    )
    score_provenance_mask = (
        mirror_score_mismatch_mask | candidate_from_mirror_mismatch_mask
    )
    threshold_domain_samples: list[str] = []
    for row, source in torch.nonzero(
        selected_below_tau_mask, as_tuple=False
    ).tolist()[:8]:
        raw = (
            int(receipt_u32[row, source].item()) & 0xFFFFFFFF
            if receipt_u32 is not None
            else 0
        )
        threshold_domain_samples.append(
            "row=%d source=%d score_gm=%d tau=%d receipt_tp=%d "
            "receipt_candidate=%d selected=%d reason=selected_below_tau" % (
                row,
                source,
                int(mirror_scores[row, source].item()),
                int(thresholds[row].item()) & 0xFFFF,
                int(bool(raw & receipt_encoding["threshold_pass_bit"]))
                if receipt_encoding
                else 0,
                int(bool(raw & receipt_encoding["candidate_bit"]))
                if receipt_encoding
                else 0,
                int(official_selected_mask[row, source].item()),
            )
        )
    score_provenance_samples: list[str] = []
    for row, source in torch.nonzero(
        score_provenance_mask, as_tuple=False
    ).tolist()[:8]:
        raw = int(receipt_u32[row, source].item()) & 0xFFFFFFFF if receipt_u32 is not None else 0
        score_provenance_samples.append(
            "row=%d source=%d receipt_score=%d score_gm=%d tau=%d receipt_tp=%d "
            "receipt_candidate=%d selected=%d" % (
                row,
                source,
                int(score_values[row, source].item()),
                int(mirror_scores[row, source].item()),
                int(thresholds[row].item()) & 0xFFFF,
                int(bool(raw & receipt_encoding["threshold_pass_bit"])) if receipt_encoding else 0,
                int(bool(raw & receipt_encoding["candidate_bit"])) if receipt_encoding else 0,
                int(official_selected_mask[row, source].item()),
            )
        )
    row_diagnostics: list[dict[str, Any]] = []
    for row in range(BATCH_SIZE):
        valid = int(source_lens[row].item())
        pool_row = int(pools[row].item())
        row_candidate = candidate_mask[row, :valid]
        row_required = required_mask[row, :valid]
        row_covered = covered_mask[row, :valid]
        row_missing = missing_mask[row, :valid]
        row_invalid = invalid_candidate_mask[row, :valid]
        row_outside = outside_domain_mask[row]
        tau_rank = (
            resident_tau_invariant_by_row[row]
            if row < len(resident_tau_invariant_by_row)
            else {
                "prefix_len": 0,
                "gt_count": 0,
                "ge_count": 0,
                "threshold_valid": 0,
                "rank_valid": 0,
            }
        )
        row_diagnostics.append(
            {
                "row": row,
                "source_len": valid,
                "pool_row": pool_row,
                "tau": int(thresholds[row].item()) & 0xFFFF,
                "candidate": int(torch.count_nonzero(row_candidate).item()),
                "required": int(torch.count_nonzero(row_required).item()),
                "covered": int(torch.count_nonzero(row_covered).item()),
                "missing": int(torch.count_nonzero(row_missing).item()),
                "invalid": int(torch.count_nonzero(row_invalid).item()),
                "outside_domain": int(torch.count_nonzero(row_outside).item()),
                "written": int(torch.count_nonzero(written_mask[row, :valid]).item()),
                "unwritten": int(torch.count_nonzero(unwritten_mask[row, :valid]).item()),
                "threshold_bit_mismatch": int(
                    torch.count_nonzero(threshold_bit_mismatch_mask[row, :valid]).item()
                ),
                "mapping_bit_mismatch": int(
                    torch.count_nonzero(mapping_bit_mismatch_mask[row, :valid]).item()
                ),
                "cold_bit_mismatch": int(
                    torch.count_nonzero(cold_bit_mismatch_mask[row, :valid]).item()
                ),
                "source_unmapped_bit_mismatch": int(
                    torch.count_nonzero(
                        source_unmapped_bit_mismatch_mask[row, :valid]
                    ).item()
                ),
                "candidate_bit_mismatch": int(
                    torch.count_nonzero(candidate_bit_mismatch_mask[row, :valid]).item()
                ),
                "score_mirror_mismatch": int(
                    torch.count_nonzero(mirror_score_mismatch_mask[row, :valid]).item()
                ),
                "candidate_from_mirror_mismatch": int(
                    torch.count_nonzero(
                        candidate_from_mirror_mismatch_mask[row, :valid]
                    ).item()
                ),
                "selected_below_tau": int(
                    torch.count_nonzero(selected_below_tau_mask[row, :valid]).item()
                ),
                "resident_prefix_len": int(tau_rank["prefix_len"]),
                "resident_gt_tau": int(tau_rank["gt_count"]),
                "resident_ge_tau": int(tau_rank["ge_count"]),
                "resident_tau_threshold_valid": int(tau_rank["threshold_valid"]),
                "resident_tau_rank_valid": int(tau_rank["rank_valid"]),
                "candidate_blocks": _block_histogram(row_candidate),
                "required_blocks": _block_histogram(row_required),
                "missing_blocks": _block_histogram(row_missing),
                "missing_sources": _source_receipt(
                    row_missing,
                    pool_row=pool_row,
                    object_map=object_map,
                    slots=slots,
                    receipt_u32=None if receipt_u32 is None else receipt_u32[row],
                    receipt_encoding=receipt_encoding,
                ),
                "invalid_sources": _source_receipt(
                    row_invalid,
                    pool_row=pool_row,
                    object_map=object_map,
                    slots=slots,
                    receipt_u32=None if receipt_u32 is None else receipt_u32[row],
                    receipt_encoding=receipt_encoding,
                ),
            }
        )
    return {
        "official_selected": selected,
        "required_cold": required,
        "covered_cold": covered,
        "missing_cold": missing,
        "candidate_count": candidate_count,
        "false_positive": false_positive,
        "cold_capacity": cold_capacity,
        "invalid_candidate_count": invalid_candidate_count,
        "mapping_discriminator_selected": mapping_discriminator_selected,
        "mapping_discriminator_sources": mapping_discriminator_sources,
        "candidate_discriminator_count": candidate_discriminator_count,
        "outside_domain_count": outside_domain_count,
        "coverage_complete": missing == 0,
        "candidate_selective": candidate_count < cold_capacity,
        "fixture_has_required_cold": required > 0,
        "mapping_fixture_discriminates": mapping_discriminator_sources > 0,
        "receipt_kind": receipt_kind,
        "receipt_written_count": receipt_written_count,
        "receipt_unwritten_count": receipt_unwritten_count,
        "receipt_malformed_count": receipt_malformed_count,
        "threshold_bit_mismatch_count": threshold_bit_mismatch_count,
        "mapping_bit_mismatch_count": mapping_bit_mismatch_count,
        "cold_bit_mismatch_count": cold_bit_mismatch_count,
        "source_unmapped_bit_mismatch_count": source_unmapped_bit_mismatch_count,
        "candidate_bit_mismatch_count": candidate_bit_mismatch_count,
        "score_mirror_active_records": int(score_mirror["active_records"]),
        "score_mirror_ready_records": int(score_mirror["ready_records"]),
        "score_mirror_expected_active_records": int(
            score_mirror["expected_active_records"]
        ),
        "score_mirror_zero_valid_records": int(score_mirror["zero_valid_records"]),
        "score_mirror_missing_count": int(score_mirror["missing_count"]),
        "score_mirror_outside_count": int(score_mirror["outside_count"]),
        "score_mirror_duplicate_count": int(score_mirror["duplicate_count"]),
        "score_mirror_interval_overlap_count": int(
            score_mirror["interval_overlap_count"]
        ),
        "score_mirror_descriptor_errors": list(score_mirror["descriptor_errors"]),
        "score_mirror_mismatch_count": mirror_score_mismatch_count,
        "candidate_from_mirror_mismatch_count": candidate_from_mirror_mismatch_count,
        "selected_below_tau_count": selected_below_tau_count,
        "resident_tau_invalid_count": resident_tau_invalid_count,
        "resident_tau_rank_invariant_count": resident_tau_rank_invariant_count,
        "resident_tau_invariant_by_row": resident_tau_invariant_by_row,
        "score_provenance_samples": score_provenance_samples,
        "threshold_domain_samples": threshold_domain_samples,
        "row_diagnostics": row_diagnostics,
    }


def run_p4_fused_audit() -> dict[str, Any]:
    """Run resident seed + one full score-time candidate scan.

    The extra resident-only scan is the planned Stage-5 threshold-estimation
    step, not a score dump and not a second full scan.  The following full
    scan is the single raw-Q/K traversal that creates candidates while C8
    computes its normal (currently diagnostic) clone output.
    """
    reference_opapi, reference_vendor = _resolve_single_opp(
        os.getenv(REFERENCE_OPP_ENV), "reference"
    )
    target_opapi, target_vendor = _resolve_single_opp(
        os.getenv(TARGET_OPP_ENV), "target"
    )
    publication_mode = _resolve_publication_mode()
    inline_mode = _resolve_inline_mode()
    if publication_mode != "publisher" or inline_mode != "audit":
        raise RuntimeError(
            "P4 resident-threshold provenance audit requires "
            f"{PUBLICATION_MODE_ENV}=publisher and {INLINE_MODE_ENV}=audit; got "
            f"publication={publication_mode!r} inline={inline_mode!r}"
        )
    target_provenance = _read_target_provenance(
        target_vendor, publication_mode, inline_mode
    )
    seed_opapi, seed_vendor = _resolve_resident_seed_opp(
        os.getenv(RESIDENT_SEED_OPP_ENV)
    )
    _prepend_unique_path(
        "ASCEND_CUSTOM_OPP_PATH", (reference_vendor, seed_vendor, target_vendor)
    )
    official, metadata, _ = _require_source_matched_official_c8_ops(
        reference_opapi, reference_vendor
    )
    print(
        "P3_TARGET_PROVENANCE "
        f"source_root={target_provenance['source_root']} "
        f"source_head={target_provenance['source_head']} "
        f"expected_opp_source_head={_expected_opp_source_head(TARGET_EXPECTED_OPP_SOURCE_HEAD_ENV, 'P3_TARGET_EXPECTED_SOURCE_HEAD', 'target')[0] or '<unset>'} "
        f"runner_head={os.getenv(RUNNER_EXPECTED_HEAD_ENV, '<unset>')} "
        f"source_clean={target_provenance['source_clean']} "
        f"publication_mode={target_provenance['publication_mode']} "
        f"inline_mode={target_provenance['inline_mode']} "
        f"kvaf_closure={target_provenance['kvaf_closure']} ready=1 "
        f"vendor={target_vendor} opapi={target_opapi}",
        flush=True,
    )
    device = _require_a5_npu()
    inputs = _build_official_inputs(device)
    inputs["metadata"] = _official_metadata(metadata, inputs)
    official_indices = _call_official(official, inputs)
    torch.npu.synchronize()

    # Build a separate C8 index-key view for the resident threshold scan.  It
    # contains only the first 24 logical pages per request, whereas the full
    # C8 scan below keeps all 32/31 pages.  This establishes a nonzero,
    # selective threshold and a real resident/cold split.
    resident = _build_resident_index_view(inputs, metadata)
    _bind_resident_seed_opp(seed_opapi)
    seed_indices, seed_values, resident_tau, tau_valid = _launch_resident_seed(
        inputs, resident
    )
    torch.npu.synchronize()
    (
        official_resident_indices,
        official_resident_values,
    ) = _call_official_resident_reference(official, inputs, resident)
    torch.npu.synchronize()
    if not _cpu_equal(seed_indices, official_resident_indices):
        raise RuntimeError(
            "A5C8ResidentSeed selection is not bitwise equal to source-matched "
            "official C8 on the resident index view"
        )
    if not _cpu_equal(seed_values, official_resident_values):
        raise RuntimeError(
            "A5C8ResidentSeed return_values=False output is not bitwise equal "
            "to source-matched official C8 on the resident index view"
        )
    if not bool(torch.all(tau_valid == 1)):
        raise RuntimeError(
            "resident fixture did not produce a valid Top-2048 threshold for every row"
        )

    # The P4 bridge loads the existing P3 typed ACLNN resolver after its
    # target opapi path has been set.  It does not load vllm_ascend.
    _bind_isolated_target_opp(target_opapi)
    from nanovllm_p4_fused_streaming_li_get import (  # noqa: PLC0415
        AUDIT_SCORE_RECEIPT_QUEUE_LEASE_SENTINEL,
        SCORE_RECEIPT_CANDIDATE_BIT,
        SCORE_RECEIPT_COLD_BIT,
        SCORE_RECEIPT_MAPPING_VALID_BIT,
        SCORE_RECEIPT_SCORE_MASK,
        SCORE_RECEIPT_SOURCE_UNMAPPED_BIT,
        SCORE_RECEIPT_TAG,
        SCORE_RECEIPT_TAG_MASK,
        SCORE_RECEIPT_THRESHOLD_PASS_BIT,
        launch,
    )
    score_receipt_encoding = {
        "tag": int(SCORE_RECEIPT_TAG),
        "tag_mask": int(SCORE_RECEIPT_TAG_MASK),
        "candidate_bit": int(SCORE_RECEIPT_CANDIDATE_BIT),
        "mapping_valid_bit": int(SCORE_RECEIPT_MAPPING_VALID_BIT),
        "cold_bit": int(SCORE_RECEIPT_COLD_BIT),
        "threshold_pass_bit": int(SCORE_RECEIPT_THRESHOLD_PASS_BIT),
        "source_unmapped_bit": int(SCORE_RECEIPT_SOURCE_UNMAPPED_BIT),
        "score_mask": int(SCORE_RECEIPT_SCORE_MASK),
    }

    transaction = _make_no_post_transaction(
        device,
        BATCH_SIZE,
        HOOK_CONTROL_MODE_ALL_BUSY,
        object_mapping="permuted_pool",
    )
    transaction["resident_threshold"].copy_(resident_tau)
    transaction["resident_threshold_valid"].copy_(tau_valid)
    _install_resident_snapshot(transaction, resident)
    if not bool(torch.any(transaction["kv_object_slots_snapshot"] >= 0)):
        raise RuntimeError("resident fixture did not mark any HBM-resident object")
    if not bool(torch.any(transaction["kv_object_slots_snapshot"] < 0)):
        raise RuntimeError("resident fixture did not leave any cold object")
    if not bool(torch.all(transaction["private_source_to_slot"] == -1)):
        raise RuntimeError(
            "score-receipt fixture requires every private_source_to_slot cell "
            "to begin as the normal unmapped sentinel"
        )
    # The sentinel is a graph input value, deliberately installed before the
    # one cloned LI launch.  There is no second scan or host-side candidate
    # reconstruction.
    transaction["queue_leases"].fill_(AUDIT_SCORE_RECEIPT_QUEUE_LEASE_SENTINEL)
    # Publisher is enabled only for this score-provenance binary. It exports
    # the score-GM that the target clone's later Top-K consumes; audit remains
    # score-only and cannot submit H2D because its transport closure is absent.
    publication = _make_publication_buffers(device)
    torch.npu.synchronize()

    print(
        "P4_FUSED_AUDIT_CONFIG "
        f"device={device} physical_npu={device.index} "
        f"reference_opapi={reference_opapi} resident_seed_opapi={seed_opapi} "
        f"target_opapi={target_opapi} "
        "raw_qk_scan_count=1 clone_topk_authority=0 "
        "score_receipt_output=private_source_to_slot[R,S] "
        "publication=publisher inline=audit "
        f"mode=audit_score_receipt resident_blocks={RESIDENT_BLOCKS} "
        "threshold_fixture=resident_seed_exact_sortable_uint16 "
        "h2d_result=not_exercised",
        flush=True,
    )
    started = monotonic()
    result = launch(
        inputs,
        transaction,
        publication,
        mode="audit_score_receipt",
        # The adversarial fixture deliberately uses a source -> compressed
        # KV-object permutation.  It is therefore C8_COMPRESSED_OBJECT,
        # rather than the raw-token identity domain used by legacy LIDU.
        kv_object_domain=1,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        cmp_ratio=1,
        return_values=False,
        stride=int(inputs["key"].stride(0)),
        scale_stride=int(inputs["key_dequant_scale"].stride(0)),
        kvaf_rows_per_entry=BLOCK_SIZE,
    )
    torch.npu.synchronize()
    elapsed_us = (monotonic() - started) * 1_000_000.0

    _assert_transaction_output_aliases(result, transaction)
    _assert_publication_output_aliases(result, publication)
    publication_summary = _assert_publication_contract(publication_mode, publication)
    receipt = _audit_receipt(result, transaction)
    score_source_lens = result["score_source_lens"].detach().cpu().contiguous()
    expected_score_source_lens = _expected_score_source_lens(device)
    if not _cpu_equal(expected_score_source_lens, result["score_source_lens"]):
        raise RuntimeError(
            "P4 audit score-source lengths differ from the independent C8 formula: "
            f"expected={expected_score_source_lens.detach().cpu().tolist()} "
            f"derived={score_source_lens.tolist()}"
        )
    score_mirror = _decode_published_score_mirror(publication, score_source_lens)
    _assert_clone_source_domain(
        result["non_authoritative_sparse_indices"], result["score_source_lens"]
    )
    tau_values = resident_tau.detach().cpu().contiguous().tolist()
    tau_valid_values = tau_valid.detach().cpu().contiguous().tolist()
    print(
        "P4_FUSED_AUDIT_RESIDENT_SEED "
        "resident_index_view=independent_fp8_clone "
        f"resident_seq_lengths={resident['resident_seq_lengths'].detach().cpu().tolist()} "
        f"resident_tau={tau_values} tau_valid={tau_valid_values} "
        "official_resident_topk=bitwise_equal "
        "threshold_domain=c8_sortable_uint16",
        flush=True,
    )
    print(
        "P4_FUSED_AUDIT_RECEIPT_CONFIG "
        f"kind={receipt['receipt_kind']} "
        f"queue_lease_sentinel={receipt['lease_sentinel']} "
        "encoding=tag31_21_score15_0_unmapped16_threshold17_cold18_mapping19_candidate20 "
        f"score_source_lens={score_source_lens.tolist()} "
        "score_source_lens_independent=bitwise_equal "
        "transaction_output_aliases=verified "
        f"publisher_score_workspace_changed={int(publication_summary['score_workspace_changed'])} "
        f"publisher_receipt_ready={int(publication_summary['receipt_ready_count'])} "
        "object_mapping=permuted_pool_reverse_compressed "
        "doorbell_posted=0 private_payload_unchanged=1 telemetry_unchanged=1 "
        f"elapsed_us={elapsed_us:.3f}",
        flush=True,
    )
    coverage = _collect_authoritative_cold_coverage(
        official_indices,
        result["candidate_source_to_slot"],
        transaction,
        score_source_lens,
        receipt["candidate_marker"],
        resident_tau,
        receipt["receipt_kind"],
        score_receipt_encoding,
        score_mirror,
    )
    for row_diag in coverage["row_diagnostics"]:
        print(
            "P4_FUSED_AUDIT_COVERAGE_DIAG "
            f"row={row_diag['row']} source_len={row_diag['source_len']} "
            f"pool_row={row_diag['pool_row']} tau={row_diag['tau']} "
            f"candidate={row_diag['candidate']} required={row_diag['required']} "
            f"covered={row_diag['covered']} missing={row_diag['missing']} "
            f"invalid={row_diag['invalid']} outside_domain={row_diag['outside_domain']} "
            f"written={row_diag['written']} unwritten={row_diag['unwritten']} "
            f"threshold_bit_mismatch={row_diag['threshold_bit_mismatch']} "
            f"mapping_bit_mismatch={row_diag['mapping_bit_mismatch']} "
            f"cold_bit_mismatch={row_diag['cold_bit_mismatch']} "
            f"source_unmapped_bit_mismatch={row_diag['source_unmapped_bit_mismatch']} "
            f"candidate_bit_mismatch={row_diag['candidate_bit_mismatch']} "
            f"score_mirror_mismatch={row_diag['score_mirror_mismatch']} "
            f"candidate_from_mirror_mismatch={row_diag['candidate_from_mirror_mismatch']} "
            f"selected_below_tau={row_diag['selected_below_tau']} "
            f"resident_prefix_len={row_diag['resident_prefix_len']} "
            f"resident_gt_tau={row_diag['resident_gt_tau']} "
            f"resident_ge_tau={row_diag['resident_ge_tau']} "
            f"resident_tau_threshold_valid={row_diag['resident_tau_threshold_valid']} "
            f"resident_tau_rank_valid={row_diag['resident_tau_rank_valid']} "
            f"candidate_blocks={row_diag['candidate_blocks']} "
            f"required_blocks={row_diag['required_blocks']} "
            f"missing_blocks={row_diag['missing_blocks']} "
            f"missing_sources={row_diag['missing_sources']} "
            f"invalid_sources={row_diag['invalid_sources']}",
            flush=True,
        )
    print(
        "P4_FUSED_AUDIT_SCORE_RECEIPT "
        f"kind={coverage['receipt_kind']} "
        f"written={coverage['receipt_written_count']} "
        f"unwritten={coverage['receipt_unwritten_count']} "
        f"malformed={coverage['receipt_malformed_count']} "
        f"threshold_bit_mismatch={coverage['threshold_bit_mismatch_count']} "
        f"mapping_bit_mismatch={coverage['mapping_bit_mismatch_count']} "
        f"cold_bit_mismatch={coverage['cold_bit_mismatch_count']} "
        f"source_unmapped_bit_mismatch={coverage['source_unmapped_bit_mismatch_count']} "
        f"candidate_bit_mismatch={coverage['candidate_bit_mismatch_count']}",
        flush=True,
    )
    print(
        "P4_SCORE_GM_MIRROR "
        f"active_records={coverage['score_mirror_active_records']} "
        f"ready_records={coverage['score_mirror_ready_records']} "
        f"expected_active_records={coverage['score_mirror_expected_active_records']} "
        f"zero_valid_records={coverage['score_mirror_zero_valid_records']} "
        f"missing={coverage['score_mirror_missing_count']} "
        f"outside={coverage['score_mirror_outside_count']} "
        f"duplicates={coverage['score_mirror_duplicate_count']} "
        f"interval_overlaps={coverage['score_mirror_interval_overlap_count']} "
        f"descriptor_errors={len(coverage['score_mirror_descriptor_errors'])}",
        flush=True,
    )
    print(
        "P4_SCORE_PROVENANCE "
        f"receipt_score_mismatch={coverage['score_mirror_mismatch_count']} "
        f"candidate_from_mirror_mismatch={coverage['candidate_from_mirror_mismatch_count']} "
        f"selected_below_tau={coverage['selected_below_tau_count']} "
        f"resident_tau_invalid={coverage['resident_tau_invalid_count']} "
        f"resident_tau_rank_invariant_failures="
        f"{coverage['resident_tau_rank_invariant_count']}",
        flush=True,
    )
    for sample in coverage["score_provenance_samples"]:
        print(f"P4_SCORE_PROVENANCE_DIFF {sample}", flush=True)
    for sample in coverage["threshold_domain_samples"]:
        print(f"P4_RESIDENT_THRESHOLD_DOMAIN_DIFF {sample}", flush=True)
    print(
        "P4_FUSED_AUDIT_OFFICIAL_COVERAGE "
        f"official_selected={coverage['official_selected']} "
        f"required_cold={coverage['required_cold']} "
        f"covered_cold={coverage['covered_cold']} "
        f"missing_cold={coverage['missing_cold']} "
        f"false_positive={coverage['false_positive']} "
        f"candidate_count={coverage['candidate_count']} "
        f"cold_capacity={coverage['cold_capacity']} "
        f"invalid_candidate_count={coverage['invalid_candidate_count']} "
        f"outside_domain_count={coverage['outside_domain_count']} "
        f"mapping_discriminator_selected={coverage['mapping_discriminator_selected']} "
        f"mapping_discriminator_sources={coverage['mapping_discriminator_sources']} "
        f"candidate_discriminator_count={coverage['candidate_discriminator_count']} "
        f"coverage_complete={int(coverage['coverage_complete'])} "
        f"candidate_selective={int(coverage['candidate_selective'])} "
        f"fixture_has_required_cold={int(coverage['fixture_has_required_cold'])} "
        f"mapping_fixture_discriminates={int(coverage['mapping_fixture_discriminates'])} "
        "candidate_recall=official_topk_cold_only "
        "h2d_result=not_exercised",
        flush=True,
    )
    return {
        "official_indices": official_indices,
        "clone_indices": result["non_authoritative_sparse_indices"],
        "candidate_count": coverage["candidate_count"],
        "candidate_marker": receipt["candidate_marker"],
        "required_cold": coverage["required_cold"],
        "covered_cold": coverage["covered_cold"],
        "missing_cold": coverage["missing_cold"],
        "mapping_discriminator_selected": coverage["mapping_discriminator_selected"],
        "mapping_discriminator_sources": coverage["mapping_discriminator_sources"],
        "candidate_discriminator_count": coverage["candidate_discriminator_count"],
        "invalid_candidate_count": coverage["invalid_candidate_count"],
        "outside_domain_count": coverage["outside_domain_count"],
        "coverage_complete": coverage["coverage_complete"],
        "candidate_selective": coverage["candidate_selective"],
        "fixture_has_required_cold": coverage["fixture_has_required_cold"],
        "mapping_fixture_discriminates": coverage["mapping_fixture_discriminates"],
        "receipt_kind": coverage["receipt_kind"],
        "receipt_written_count": coverage["receipt_written_count"],
        "receipt_unwritten_count": coverage["receipt_unwritten_count"],
        "receipt_malformed_count": coverage["receipt_malformed_count"],
        "threshold_bit_mismatch_count": coverage["threshold_bit_mismatch_count"],
        "mapping_bit_mismatch_count": coverage["mapping_bit_mismatch_count"],
        "cold_bit_mismatch_count": coverage["cold_bit_mismatch_count"],
        "source_unmapped_bit_mismatch_count": coverage[
            "source_unmapped_bit_mismatch_count"
        ],
        "candidate_bit_mismatch_count": coverage["candidate_bit_mismatch_count"],
        "score_mirror_active_records": coverage["score_mirror_active_records"],
        "score_mirror_ready_records": coverage["score_mirror_ready_records"],
        "score_mirror_expected_active_records": coverage[
            "score_mirror_expected_active_records"
        ],
        "score_mirror_zero_valid_records": coverage[
            "score_mirror_zero_valid_records"
        ],
        "score_mirror_missing_count": coverage["score_mirror_missing_count"],
        "score_mirror_outside_count": coverage["score_mirror_outside_count"],
        "score_mirror_duplicate_count": coverage["score_mirror_duplicate_count"],
        "score_mirror_interval_overlap_count": coverage[
            "score_mirror_interval_overlap_count"
        ],
        "score_mirror_descriptor_errors": coverage["score_mirror_descriptor_errors"],
        "score_mirror_mismatch_count": coverage["score_mirror_mismatch_count"],
        "candidate_from_mirror_mismatch_count": coverage[
            "candidate_from_mirror_mismatch_count"
        ],
        "selected_below_tau_count": coverage["selected_below_tau_count"],
        "resident_tau_invalid_count": coverage["resident_tau_invalid_count"],
        "resident_tau_rank_invariant_count": coverage[
            "resident_tau_rank_invariant_count"
        ],
        "resident_tau_invariant_by_row": coverage["resident_tau_invariant_by_row"],
        "score_provenance_samples": coverage["score_provenance_samples"],
        "threshold_domain_samples": coverage["threshold_domain_samples"],
        "expected_score_source_lens": expected_score_source_lens,
        "derived_score_source_lens": result["score_source_lens"],
    }


__all__ = ["run_p4_fused_audit"]
