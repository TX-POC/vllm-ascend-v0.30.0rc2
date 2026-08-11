# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""nanovllm-dsa A5 算子适配层。

本模块为 nanovllm-dsa Ascend 950 自定义算子提供 DSA runtime 薄适配。
所有 reshape/view 都只调整 tensor metadata；大 tensor 不在 decode 热路径做
隐式 contiguous/materialization。
"""

from __future__ import annotations

import logging
import zlib

import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend.dsa_offload.contracts import DSA_SFA_COMPUTE_TOPK
from vllm_ascend.dsa_offload.ops import (
    DSALightningIndexerOutputs,
    _normalize_lidu_weights_layout,
    _squeeze_cache_head_dim,
)

_SPARSE_COUNT = DSA_SFA_COMPUTE_TOPK
_PACKED_ROW_BYTES = 656
_BLOCK_SIZE = 128
_PACKED_KSC_ROUTE_DIAG_LOGGED = False
_PACKED_PAYLOAD_DIAG_CAPTURED = False
_PACKED_PAYLOAD_DIAG_CHECKED = False
_PACKED_PAYLOAD_DIAG = None
_PACKED_ATTENTION_META_DIAG_LOGGED = False
_PACKED_QSFA_DIAG_LOGGED = False


def _packed_row_summary(row: torch.Tensor) -> str:
    """Decode one CPU 656-byte packed row without changing model data."""

    row = row.contiguous().view(torch.int8)
    ckv = row[:512]
    fp8_tiles = ckv.view(4, 128)
    rope = row[512:640].contiguous().view(torch.bfloat16).to(torch.float32)
    scales = row[640:656].contiguous().view(torch.float32)
    ckv_checksum = int(ckv.to(torch.int64).sum().item())
    row_crc32 = f"{zlib.crc32(row.numpy().tobytes()):08x}"
    fp8_tile_crc32 = [
        f"{zlib.crc32(tile.numpy().tobytes()):08x}" for tile in fp8_tiles
    ]
    rope_crc32 = f"{zlib.crc32(row[512:640].numpy().tobytes()):08x}"
    scale_crc32 = f"{zlib.crc32(row[640:656].numpy().tobytes()):08x}"
    rope_head = [round(float(value), 6) for value in rope[:4].tolist()]
    scale_values = [round(float(value), 8) for value in scales.tolist()]
    return (
        f"row_crc32={row_crc32} ckv_checksum={ckv_checksum} "
        f"fp8_tile_crc32={fp8_tile_crc32} "
        f"rope_crc32={rope_crc32} scale_crc32={scale_crc32} "
        f"rope_head={rope_head} scales={scale_values} "
        f"rope_finite={int(torch.isfinite(rope).sum().item())}/64 "
        f"scale_finite={int(torch.isfinite(scales).sum().item())}/4"
    )


def _packed_block_uniqueness(block: torch.Tensor) -> str:
    """Summarize row-level payload diversity in one CPU packed block."""

    rows = block.contiguous().view(torch.int8).reshape(
        _BLOCK_SIZE,
        _PACKED_ROW_BYTES,
    )

    def component_stats(start: int, end: int) -> tuple[int, int, list[str]]:
        groups: dict[int, list[int]] = {}
        for row_index in range(_BLOCK_SIZE):
            payload = rows[row_index, start:end].numpy().tobytes()
            checksum = zlib.crc32(payload)
            groups.setdefault(checksum, []).append(row_index)
        repeats = sorted(
            (
                (checksum, offsets)
                for checksum, offsets in groups.items()
                if len(offsets) > 1
            ),
            key=lambda item: (-len(item[1]), item[0]),
        )
        repeat_desc = [
            f"{checksum:08x}:{len(offsets)}@{offsets[:12]}"
            for checksum, offsets in repeats[:4]
        ]
        max_repeat = max((len(offsets) for offsets in groups.values()), default=0)
        return len(groups), max_repeat, repeat_desc

    ckv_unique, ckv_max_repeat, ckv_repeats = component_stats(0, 512)
    rope_unique, rope_max_repeat, rope_repeats = component_stats(512, 640)
    scale_unique, scale_max_repeat, scale_repeats = component_stats(640, 656)
    return (
        f"unique_ckv_rows={ckv_unique}/128 "
        f"max_ckv_repeat={ckv_max_repeat} ckv_repeats={ckv_repeats} "
        f"unique_rope_rows={rope_unique}/128 "
        f"max_rope_repeat={rope_max_repeat} rope_repeats={rope_repeats} "
        f"unique_scale_rows={scale_unique}/128 "
        f"max_scale_repeat={scale_max_repeat} scale_repeats={scale_repeats}"
    )


def _capture_packed_dump_payload_once(
    *,
    hbm_packed_kv: torch.Tensor,
    src_block_ids: torch.Tensor,
    dst_block_ids: torch.Tensor,
) -> None:
    """Snapshot a few pre-dump HBM blocks for a later KSC round-trip check."""

    global _PACKED_PAYLOAD_DIAG_CAPTURED
    global _PACKED_PAYLOAD_DIAG
    if _PACKED_PAYLOAD_DIAG_CAPTURED:
        return
    if not logger.isEnabledFor(logging.INFO):
        return

    try:
        torch.npu.synchronize()
        if (
            hbm_packed_kv.ndim != 4
            or int(hbm_packed_kv.shape[1]) != _BLOCK_SIZE
            or int(hbm_packed_kv.shape[2]) != 1
            or int(hbm_packed_kv.shape[3]) != _PACKED_ROW_BYTES
            or not hbm_packed_kv.is_contiguous()
        ):
            logger.error(
                "[DSA_A5_C8_DUMP_INPUT] invalid_layout=true shape=%s dtype=%s "
                "stride=%s contiguous=%s",
                tuple(hbm_packed_kv.shape),
                hbm_packed_kv.dtype,
                tuple(hbm_packed_kv.stride()),
                hbm_packed_kv.is_contiguous(),
            )
            return

        src = [int(value) for value in src_block_ids.detach().cpu().tolist()]
        dst = [int(value) for value in dst_block_ids.detach().cpu().tolist()]
        job_count = min(len(src), len(dst))
        if job_count == 0:
            return

        # First evicted Prompt block is typically around logical block 80 for
        # the 10,240-token resident budget. Keep nearby blocks plus endpoints.
        sample_jobs = sorted(
            {
                index
                for index in (0, 1, 79, 80, 81, 82, job_count - 1)
                if 0 <= index < job_count
            }
        )
        snapshots = {}
        sample_desc: list[str] = []
        for job_index in sample_jobs:
            src_block = src[job_index]
            dst_block = dst[job_index]
            # Graph-capture padding uses negative IDs and DRAM block 0 is the
            # reserved null mapping. Do not let a dummy launch consume the
            # one-shot diagnostic.
            if (
                not (0 <= src_block < int(hbm_packed_kv.shape[0]))
                or dst_block <= 0
            ):
                continue
            block = (
                hbm_packed_kv[src_block]
                .view(torch.int8)
                .reshape(_BLOCK_SIZE, _PACKED_ROW_BYTES)
                .detach()
                .cpu()
                .clone()
            )
            snapshots[dst_block] = {
                "job": job_index,
                "src_block": src_block,
                "rows": block,
            }
            sample_desc.append(
                f"job={job_index}:srcP={src_block}->dramP={dst_block}:"
                f"{_packed_row_summary(block[0])} "
                f"{_packed_block_uniqueness(block)}"
            )

        if not snapshots:
            return
        _PACKED_PAYLOAD_DIAG = {
            "hbm_data_ptr": int(hbm_packed_kv.data_ptr()),
            "snapshots": snapshots,
        }
        _PACKED_PAYLOAD_DIAG_CAPTURED = True
        logger.info(
            "[DSA_A5_C8_DUMP_INPUT] shape=%s dtype=%s stride=%s "
            "contiguous=%s job_count=%d sampled_blocks=%d samples=%s",
            tuple(hbm_packed_kv.shape),
            hbm_packed_kv.dtype,
            tuple(hbm_packed_kv.stride()),
            hbm_packed_kv.is_contiguous(),
            job_count,
            len(snapshots),
            " | ".join(sample_desc),
        )
    except Exception:
        logger.exception("[DSA_A5_C8_DUMP_INPUT] diagnostic_failed=true")


def _log_packed_attention_metadata_once(
    *,
    hbm_block_count: int,
    hbm_block_table: torch.Tensor,
    attention_slots: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
) -> None:
    """Audit the exact logical-slot metadata consumed by packed C8 QSFA."""

    global _PACKED_ATTENTION_META_DIAG_LOGGED
    if _PACKED_ATTENTION_META_DIAG_LOGGED:
        return
    if not logger.isEnabledFor(logging.INFO):
        return
    _PACKED_ATTENTION_META_DIAG_LOGGED = True

    try:
        cache = [int(value) for value in cache_tokens.detach().cpu().tolist()]
        candidate = [
            int(value) for value in candidate_lens.detach().cpu().tolist()
        ]
        actual = [
            int(value)
            for value in actual_seq_lengths_kv.detach().cpu().tolist()
        ]
        resident = [
            int(value)
            for value in resident_seq_lengths.detach().cpu().tolist()
        ]
        slot_rows = attention_slots.detach().cpu()
        table_rows = hbm_block_table.detach().cpu()
        row_count = min(
            4,
            len(cache),
            len(candidate),
            len(actual),
            len(resident),
            int(slot_rows.shape[0]),
            int(table_rows.shape[0]),
        )
        row_desc: list[str] = []
        attention_capacity = int(slot_rows.shape[-1])
        tail_capacity = attention_capacity - _SPARSE_COUNT

        for row_index in range(row_count):
            cache_count = cache[row_index]
            candidate_len = candidate[row_index]
            actual_len = actual[row_index]
            resident_len = resident[row_index]
            metadata_valid = (
                cache_count >= _SPARSE_COUNT
                and candidate_len >= _SPARSE_COUNT
                and actual_len >= candidate_len
                and actual_len - candidate_len <= tail_capacity
            )
            tail_count = actual_len - candidate_len if metadata_valid else 0
            expected_resident = (
                cache_count + tail_count if metadata_valid else 0
            )
            slots = [
                int(value)
                for value in slot_rows[row_index, 0].tolist()
            ]
            topk = slots[:_SPARSE_COUNT]
            tail = slots[
                _SPARSE_COUNT : _SPARSE_COUNT + max(0, tail_count)
            ]
            valid_topk = [
                value for value in topk if 0 <= value < cache_count
            ]
            topk_invalid = len(topk) - len(valid_topk)
            topk_duplicates = len(valid_topk) - len(set(valid_topk))
            expected_tail = list(
                range(cache_count, cache_count + max(0, tail_count))
            )
            tail_mismatch = sum(
                left != right
                for left, right in zip(tail, expected_tail)
            ) + abs(len(tail) - len(expected_tail))

            required_cols = (
                (max(resident_len, 0) + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            )
            table = [
                int(value)
                for value in table_rows[
                    row_index,
                    : min(required_cols, int(table_rows.shape[1])),
                ].tolist()
            ]
            invalid_phys = sum(
                value < 0 or value >= hbm_block_count for value in table
            )
            unresolved_topk = 0
            for value in valid_topk:
                logical_block = value // _BLOCK_SIZE
                if (
                    logical_block >= len(table)
                    or table[logical_block] < 0
                    or table[logical_block] >= hbm_block_count
                ):
                    unresolved_topk += 1

            row_desc.append(
                f"row={row_index}:cache={cache_count}:"
                f"candidate={candidate_len}:actual={actual_len}:"
                f"resident={resident_len}:expected_resident="
                f"{expected_resident}:topk_invalid={topk_invalid}:"
                f"topk_duplicates={topk_duplicates}:"
                f"unresolved_topk={unresolved_topk}:"
                f"tail_count={tail_count}:tail_mismatch={tail_mismatch}:"
                f"required_table_cols={required_cols}:"
                f"table_cols_read={len(table)}:invalid_phys={invalid_phys}:"
                f"topk_head={topk[:16]}:tail_head={tail[:16]}"
            )

        logger.info(
            "[DSA_A5_C8_ATTN_META] hbm_blocks=%d attention_shape=%s "
            "attention_dtype=%s block_table_shape=%s "
            "resident_shape=%s rows=%s",
            hbm_block_count,
            tuple(attention_slots.shape),
            attention_slots.dtype,
            tuple(hbm_block_table.shape),
            tuple(resident_seq_lengths.shape),
            " | ".join(row_desc),
        )
    except Exception:
        logger.exception("[DSA_A5_C8_ATTN_META] diagnostic_failed=true")


def _check_packed_roundtrip_once(
    *,
    hbm_packed_kv: torch.Tensor,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    source_token_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    copy_counts: torch.Tensor,
) -> None:
    """Compare pre-dump rows with the same rows after DDR->HBM KSC."""

    global _PACKED_PAYLOAD_DIAG_CHECKED
    state = _PACKED_PAYLOAD_DIAG
    if _PACKED_PAYLOAD_DIAG_CHECKED or state is None:
        return
    # First packed dump call and the matching KSC layer share the same resident
    # tensor storage. Ignore earlier/later layers until the data_ptr matches.
    if int(hbm_packed_kv.data_ptr()) != int(state["hbm_data_ptr"]):
        return

    try:
        torch.npu.synchronize()
        counts = copy_counts.detach().cpu().tolist()
        count = max(0, min(int(counts[0]), _SPARSE_COUNT)) if counts else 0
        if count == 0:
            return

        src = source_token_ids[0, :count].detach().cpu().tolist()
        dst = destination_slots[0, :count].detach().cpu().tolist()
        max_src_col = max((int(value) >> 7 for value in src if int(value) >= 0), default=-1)
        max_dst_col = max((int(value) >> 7 for value in dst if int(value) >= 0), default=-1)
        dram_cols = min(max_src_col + 1, int(dram_block_table.shape[1]))
        hbm_cols = min(max_dst_col + 1, int(hbm_block_table.shape[1]))
        dram_row = dram_block_table[0, :dram_cols].detach().cpu().tolist() if dram_cols > 0 else []
        hbm_row = hbm_block_table[0, :hbm_cols].detach().cpu().tolist() if hbm_cols > 0 else []

        matched = 0
        equal_rows = 0
        mismatch_rows = 0
        mismatch_bytes = 0
        ckv_mismatch = 0
        fp8_tile_mismatch = [0, 0, 0, 0]
        rope_mismatch = 0
        scale_mismatch = 0
        first_mismatch = "none"
        route_desc: list[str] = []
        snapshots = state["snapshots"]

        for copy_index, (source_value, destination_value) in enumerate(zip(src, dst)):
            if matched >= 32:
                break
            source = int(source_value)
            destination = int(destination_value)
            if source < 0 or destination < 0:
                continue
            source_col = source >> 7
            source_off = source & 127
            destination_col = destination >> 7
            destination_off = destination & 127
            if source_col >= len(dram_row) or destination_col >= len(hbm_row):
                continue
            source_phys = int(dram_row[source_col])
            destination_phys = int(hbm_row[destination_col])
            snapshot = snapshots.get(source_phys)
            if snapshot is None or destination_phys < 0:
                continue

            expected = snapshot["rows"][source_off]
            actual = (
                hbm_packed_kv[destination_phys, destination_off, 0]
                .view(torch.int8)
                .detach()
                .cpu()
                .contiguous()
            )
            diff = expected != actual
            row_mismatch = int(diff.sum().item())
            matched += 1
            if row_mismatch == 0:
                equal_rows += 1
            else:
                mismatch_rows += 1
                mismatch_bytes += row_mismatch
                ckv_mismatch += int(diff[:512].sum().item())
                for tile_index in range(4):
                    tile_start = tile_index * 128
                    tile_end = tile_start + 128
                    fp8_tile_mismatch[tile_index] += int(
                        diff[tile_start:tile_end].sum().item()
                    )
                rope_mismatch += int(diff[512:640].sum().item())
                scale_mismatch += int(diff[640:656].sum().item())
                if first_mismatch == "none":
                    offsets = torch.nonzero(diff, as_tuple=False).view(-1)
                    first_offset = int(offsets[0].item()) if offsets.numel() else -1
                    first_mismatch = (
                        f"copy={copy_index}/src={source}/dramP={source_phys}/o={source_off}"
                        f"->slot={destination}/hbmP={destination_phys}/o={destination_off}"
                        f"/byte={first_offset} expected={int(expected[first_offset])}"
                        f" actual={int(actual[first_offset])}"
                    )
            if len(route_desc) < 4:
                route_desc.append(
                    f"copy={copy_index}:src={source}/dramP={source_phys}/o{source_off}"
                    f"->slot={destination}/hbmP={destination_phys}/o{destination_off}:"
                    f"diff={row_mismatch}:expected[{_packed_row_summary(expected)}]:"
                    f"actual[{_packed_row_summary(actual)}]"
                )

        if matched == 0:
            return
        _PACKED_PAYLOAD_DIAG_CHECKED = True
        logger.info(
            "[DSA_A5_C8_KV_ROUNDTRIP] copy_count=%d matched_routes=%d "
            "equal_rows=%d mismatch_rows=%d mismatch_bytes=%d "
            "ckv_mismatch_bytes=%d fp8_tile_mismatch_bytes=%s "
            "rope_mismatch_bytes=%d "
            "scale_mismatch_bytes=%d first_mismatch=%s routes=%s",
            count,
            matched,
            equal_rows,
            mismatch_rows,
            mismatch_bytes,
            ckv_mismatch,
            fp8_tile_mismatch,
            rope_mismatch,
            scale_mismatch,
            first_mismatch,
            " | ".join(route_desc),
        )
    except Exception:
        logger.exception("[DSA_A5_C8_KV_ROUNDTRIP] diagnostic_failed=true")


def _log_packed_ksc_route_once(
    *,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    source_token_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    copy_counts: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
) -> None:
    """一次性审计 A5 packed KSC 的 logical->physical 路由。

    这里只读取少量 int32 metadata，不读取/修改 KV payload，也不改变任何
    LIDU/KSC/QSFA 算子行为。诊断重点是判断长序列首次大量 miss 时：

    * source token 是否都落在 candidate 范围；
    * destination slot 是否都落在 resident sparse budget；
    * logical block 是否解析到有效 DDR/HBM physical block；
    * KSC 按 copy-index 多核调度时，是否出现大量相邻 656-byte rows 共享
      同一个 32-byte GM sector 的高风险写入模式。
    """

    global _PACKED_KSC_ROUTE_DIAG_LOGGED
    if _PACKED_KSC_ROUTE_DIAG_LOGGED:
        return
    _PACKED_KSC_ROUTE_DIAG_LOGGED = True
    if not logger.isEnabledFor(logging.INFO):
        return

    try:
        # LIDU C8 当前已经在返回前 synchronize；这里再同步一次只发生在
        # 每个 worker 的第一次 KSC，确保诊断读取的是最终 metadata。
        torch.npu.synchronize()
        counts = copy_counts.detach().cpu().tolist()
        if not counts:
            logger.info("[DSA_A5_C8_KSC_ROUTE] empty_copy_counts=true")
            return

        count = max(0, min(int(counts[0]), _SPARSE_COUNT))
        cache_token_count = int(cache_tokens[0].detach().cpu().item())
        candidate_len = int(candidate_lens[0].detach().cpu().item())
        if count == 0:
            logger.info(
                "[DSA_A5_C8_KSC_ROUTE] copy_count=0 cache_tokens=%d "
                "candidate_len=%d",
                cache_token_count,
                candidate_len,
            )
            return

        src = source_token_ids[0, :count].detach().cpu().tolist()
        dst = destination_slots[0, :count].detach().cpu().tolist()
        src_invalid = sum(
            int(value) < 0 or int(value) >= candidate_len for value in src
        )
        dst_invalid = sum(
            int(value) < 0 or int(value) >= cache_token_count for value in dst
        )
        src_duplicates = count - len(set(int(value) for value in src))
        dst_duplicates = count - len(set(int(value) for value in dst))

        max_src_col = max(
            (int(value) >> 7 for value in src if int(value) >= 0),
            default=-1,
        )
        max_dst_col = max(
            (int(value) >> 7 for value in dst if int(value) >= 0),
            default=-1,
        )
        dram_width = int(dram_block_table.shape[1])
        hbm_width = int(hbm_block_table.shape[1])
        dram_cols = min(max_src_col + 1, dram_width)
        hbm_cols = min(max_dst_col + 1, hbm_width)
        dram_row = (
            dram_block_table[0, :dram_cols].detach().cpu().tolist()
            if dram_cols > 0
            else []
        )
        hbm_row = (
            hbm_block_table[0, :hbm_cols].detach().cpu().tolist()
            if hbm_cols > 0
            else []
        )

        null_or_invalid_dram = 0
        invalid_hbm = 0
        shared_32b_boundary_pairs = 0
        for index, (source, destination) in enumerate(zip(src, dst)):
            source = int(source)
            destination = int(destination)
            source_col = source >> 7 if source >= 0 else -1
            destination_col = destination >> 7 if destination >= 0 else -1
            source_phys = (
                int(dram_row[source_col])
                if 0 <= source_col < len(dram_row)
                else -1
            )
            destination_phys = (
                int(hbm_row[destination_col])
                if 0 <= destination_col < len(hbm_row)
                else -1
            )
            if source_phys <= 0:
                null_or_invalid_dram += 1
            if destination_phys < 0:
                invalid_hbm += 1

            if index + 1 >= count:
                continue
            next_destination = int(dst[index + 1])
            if destination < 0 or next_destination < 0:
                continue
            low = min(destination, next_destination)
            high = max(destination, next_destination)
            # packed block base(128*656=83968) is 32B aligned. Two adjacent
            # token rows share a 32B sector exactly when the lower row index
            # inside the block is even. KSC assigns adjacent copy indices to
            # different AIV cores whenever usedCoreNum > 1 (A5 uses 64).
            if (
                high == low + 1
                and (low >> 7) == (high >> 7)
                and (low & 1) == 0
            ):
                shared_32b_boundary_pairs += 1

        route_sample: list[str] = []
        for index in range(min(count, 16)):
            source = int(src[index])
            destination = int(dst[index])
            source_col = source >> 7 if source >= 0 else -1
            source_off = source & 127 if source >= 0 else -1
            destination_col = destination >> 7 if destination >= 0 else -1
            destination_off = destination & 127 if destination >= 0 else -1
            source_phys = (
                int(dram_row[source_col])
                if 0 <= source_col < len(dram_row)
                else -1
            )
            destination_phys = (
                int(hbm_row[destination_col])
                if 0 <= destination_col < len(hbm_row)
                else -1
            )
            route_sample.append(
                f"{index}:src={source}/L{source_col}/P{source_phys}/o{source_off}"
                f"->slot={destination}/L{destination_col}/P{destination_phys}/o{destination_off}"
            )

        logger.info(
            "[DSA_A5_C8_KSC_ROUTE] copy_count=%d cache_tokens=%d "
            "candidate_len=%d src_invalid=%d dst_invalid=%d "
            "src_duplicates=%d dst_duplicates=%d "
            "null_or_invalid_dram=%d invalid_hbm=%d "
            "shared_32b_boundary_pairs=%d packed_row_bytes=%d "
            "routes=%s",
            count,
            cache_token_count,
            candidate_len,
            src_invalid,
            dst_invalid,
            src_duplicates,
            dst_duplicates,
            null_or_invalid_dram,
            invalid_hbm,
            shared_32b_boundary_pairs,
            _PACKED_ROW_BYTES,
            " | ".join(route_sample),
        )
    except Exception:
        # 诊断失败不能改变服务正确性；保留 traceback 方便继续定位。
        logger.exception("[DSA_A5_C8_KSC_ROUTE] diagnostic_failed=true")


def require_nanovllm_dsa_ops() -> None:
    """初始化期加载主 vLLM-Ascend 扩展并确认 A5 算子注册。"""

    try:
        import vllm_ascend.dsa_offload.a5_extension  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize A5 DSA operators from vllm_ascend_C; "
            "rebuild/reinstall vllm-ascend with custom kernels enabled on "
            "Ascend 950."
        ) from exc

    if not hasattr(torch.ops, "nanovllm_dsa"):
        raise RuntimeError(
            "A5 DSA operators are not registered after loading "
            "vllm_ascend_C."
        )
    missing = [
        op_name
        for op_name in (
            "lidu_decode_update_out",
            "lidu_decode_update_c8_out",
            "scatter_copy",
            "sparse_and_tail_attention",
            "lidu_cache_update_out",
            "packed_scatter_copy_out",
            "sparse_and_tail_attention_c8",
            "full_block_dump",
            "packed_full_block_dump",
        )
        if not hasattr(torch.ops.nanovllm_dsa, op_name)
    ]
    if missing:
        raise RuntimeError(
            "A5 DSA registrations embedded in vllm_ascend_C are incomplete: "
            f"missing={tuple(missing)}"
        )

    from vllm_ascend.ascend_config import get_ascend_config

    ascend_config = get_ascend_config()
    sfa_c8 = bool(ascend_config.enable_sparse_sfa_c8)
    li_c8 = bool(ascend_config.enable_sparse_li_c8)
    if sfa_c8 != li_c8:
        raise RuntimeError(
            "nanovllm_a5 DSA requires enable_sparse_sfa_c8 and "
            "enable_sparse_li_c8 to be enabled/disabled together: "
            f"sfa_c8={sfa_c8}, li_c8={li_c8}"
        )


def _ensure_4d_cache(cache: torch.Tensor, *, name: str) -> torch.Tensor:
    if cache.ndim == 4 and cache.shape[2] == 1:
        return cache
    if cache.ndim == 3:
        return cache.unsqueeze(2)
    raise ValueError(
        f"{name} must be [blocks, block, 1, dim] or "
        f"[blocks, block, dim], got {tuple(cache.shape)}"
    )


def _a5_topk_view(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    """零拷贝取得连续 [B,1,2048] A5 TopK scratch view。"""

    if tensor.ndim != 3 or tensor.shape[1] != 1:
        raise ValueError(
            f"{name} must be [B,1,capacity], got {tuple(tensor.shape)}"
        )
    batch = int(tensor.shape[0])
    required = batch * _SPARSE_COUNT
    if tensor.numel() < required:
        raise ValueError(
            f"{name} has insufficient storage for A5 TopK: "
            f"numel={tensor.numel()}, required={required}"
        )
    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} backing scratch must be contiguous, stride={tensor.stride()}"
        )
    return tensor.view(-1)[:required].view(batch, 1, _SPARSE_COUNT)


def _restore_c8_lidu_views(
    *,
    query: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """恢复 C8 LI 的 [B,N,128]/[B,N] ABI，仅做 view、不复制数据。"""

    if weights.ndim != 2:
        raise ValueError(
            "C8 LIDU weights must be [B,N] so query views can be restored, "
            f"got {tuple(weights.shape)}"
        )
    batch, heads = int(weights.shape[0]), int(weights.shape[1])
    if query.ndim == 2:
        if int(query.shape[0]) != batch * heads:
            raise ValueError(
                "Flattened C8 LIDU query rows do not match weights: "
                f"query={tuple(query.shape)}, weights={tuple(weights.shape)}"
            )
        query = query.view(batch, heads, query.shape[-1])
    if query.ndim != 3:
        raise ValueError(
            f"C8 LIDU query must be [B,N,D], got {tuple(query.shape)}"
        )
    if query_dequant_scale.ndim == 1:
        if int(query_dequant_scale.numel()) != batch * heads:
            raise ValueError(
                "Flattened C8 query scale does not match weights: "
                f"scale={tuple(query_dequant_scale.shape)}, "
                f"weights={tuple(weights.shape)}"
            )
        query_dequant_scale = query_dequant_scale.view(batch, heads)
    if query_dequant_scale.ndim != 2:
        raise ValueError(
            "C8 LIDU query_dequant_scale must be [B,N], got "
            f"{tuple(query_dequant_scale.shape)}"
        )
    return query, query_dequant_scale


def quant_lightning_indexer_c8(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    """A5 native quant LightningIndexer without resident-pool mutation。"""

    weights = _normalize_lidu_weights_layout(weights)
    query, query_dequant_scale = _restore_c8_lidu_views(
        query=query,
        query_dequant_scale=query_dequant_scale,
        weights=weights,
    )
    op = getattr(torch_npu, "npu_quant_lightning_indexer", None)
    if op is None:
        namespace = getattr(torch.ops, "_C_ascend", None)
        op = (
            getattr(namespace, "npu_lightning_indexer_quant", None)
            if namespace is not None
            else None
        )
    if op is None:
        raise RuntimeError("A5 Quant LightningIndexer is not registered")

    result = op(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_dequant_scale,
        key_dequant_scale=key_dequant_scale,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=_SPARSE_COUNT,
        sparse_mode=3,
    )
    topk = result[0] if isinstance(result, tuple) else result
    if not isinstance(topk, torch.Tensor) or topk.dtype != torch.int32:
        raise RuntimeError("A5 Quant LightningIndexer returned no int32 TopK")
    expected = int(query.shape[0]) * _SPARSE_COUNT
    if topk.numel() != expected:
        raise RuntimeError(
            "A5 Quant LightningIndexer returned unexpected TopK shape: "
            f"shape={tuple(topk.shape)}, expected_numel={expected}"
        )
    return topk.reshape(query.shape[0], 1, _SPARSE_COUNT)


def lightning_indexer_decode_update(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots: torch.Tensor,
    row_modes: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
    outputs: DSALightningIndexerOutputs,
    cache_tokens: torch.Tensor,
) -> None:
    """调用 nanovllm BF16 LIDU。"""

    del row_modes
    weights = _normalize_lidu_weights_layout(weights)
    source_ids = _a5_topk_view(outputs.topk_index, name="topk_index")
    destination_slots = _a5_topk_view(outputs.topk_slots, name="topk_slots")
    torch.ops.nanovllm_dsa.lidu_decode_update_out(
        query,
        key,
        weights,
        req_pool_entries,
        cache_slots,
        cache_tokens,
        actual_seq_lengths_key,
        block_table,
        source_ids,
        destination_slots,
        outputs.miss_count,
    )


def kvcache_scatter_copy(
    *,
    resident_nope_cache: torch.Tensor,
    resident_rope_cache: torch.Tensor,
    dram_nope_arena: torch.Tensor,
    dram_rope_arena: torch.Tensor,
    resident_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    src_token_ids: torch.Tensor,
    dst_slots: torch.Tensor,
    copy_counts: torch.Tensor,
) -> None:
    """调用 nanovllm KSC；TopK 只做零拷贝 view。"""

    hbm_kpe = _squeeze_cache_head_dim(
        resident_rope_cache,
        name="resident_rope_cache",
    )
    hbm_ckv = _squeeze_cache_head_dim(
        resident_nope_cache,
        name="resident_nope_cache",
    )
    dram_kpe = _squeeze_cache_head_dim(
        dram_rope_arena,
        name="dram_rope_arena",
    )
    dram_ckv = _squeeze_cache_head_dim(
        dram_nope_arena,
        name="dram_nope_arena",
    )
    source_token_ids = _a5_topk_view(
        src_token_ids, name="src_token_ids"
    ).view(src_token_ids.shape[0], _SPARSE_COUNT)
    destination_slots_2d = _a5_topk_view(
        dst_slots, name="dst_slots"
    ).view(dst_slots.shape[0], _SPARSE_COUNT)
    torch.ops.nanovllm_dsa.scatter_copy(
        hbm_kpe,
        hbm_ckv,
        dram_kpe,
        dram_ckv,
        resident_block_table,
        dram_block_table,
        source_token_ids,
        destination_slots_2d,
        copy_counts,
    )


def sparse_flash_attention_for_offload(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    sparse_indices: torch.Tensor,
    tail_info: torch.Tensor,
    scale_value: float,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    cache_tokens: torch.Tensor,
) -> torch.Tensor:
    """调用 nanovllm BF16 sparse+tail attention。"""

    del tail_info
    sparse_slots = _a5_topk_view(
        sparse_indices, name="sparse_indices"
    )
    key_4d = _ensure_4d_cache(key, name="key")
    key_rope_4d = _ensure_4d_cache(key_rope, name="key_rope")
    return torch.ops.nanovllm_dsa.sparse_and_tail_attention(
        query,
        key_4d,
        key_4d,
        sparse_slots,
        cache_tokens,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope_4d,
        float(scale_value),
    )


def lightning_indexer_decode_update_c8(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
    outputs: DSALightningIndexerOutputs,
) -> None:
    """C8 LIDU：量化 LI + request-pool update。"""

    weights = _normalize_lidu_weights_layout(weights)
    query, query_dequant_scale = _restore_c8_lidu_views(
        query=query,
        query_dequant_scale=query_dequant_scale,
        weights=weights,
    )
    source_ids = _a5_topk_view(outputs.topk_index, name="topk_index")
    destination_slots = _a5_topk_view(outputs.topk_slots, name="topk_slots")
    torch.ops.nanovllm_dsa.lidu_decode_update_c8_out(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        actual_seq_lengths_query,
        req_pool_entries,
        cache_slots,
        cache_tokens,
        candidate_lens,
        block_table,
        source_ids,
        destination_slots,
        outputs.miss_count,
    )
    torch.npu.synchronize()


def packed_kvcache_scatter_copy(
    *,
    hbm_packed_kv: torch.Tensor,
    dram_packed_kv: torch.Tensor,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    source_token_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    copy_counts: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    max_tail_tokens: int,
    attention_slots: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """C8 packed SCATTER：整行搬运 packed KV 并生成 attention metadata。"""

    hbm_bytes = (
        hbm_packed_kv
        if hbm_packed_kv.dtype == torch.int8
        else hbm_packed_kv.view(torch.int8)
    )
    dram_bytes = (
        dram_packed_kv
        if dram_packed_kv.dtype == torch.int8
        else dram_packed_kv.view(torch.int8)
    )
    src_ids_2d = _a5_topk_view(
        source_token_ids, name="source_token_ids"
    ).view(source_token_ids.shape[0], _SPARSE_COUNT)
    dst_slots_2d = _a5_topk_view(
        destination_slots, name="dst_slots"
    ).view(destination_slots.shape[0], _SPARSE_COUNT)
    _log_packed_ksc_route_once(
        hbm_block_table=hbm_block_table,
        dram_block_table=dram_block_table,
        source_token_ids=src_ids_2d,
        destination_slots=dst_slots_2d,
        copy_counts=copy_counts,
        cache_tokens=cache_tokens,
        candidate_lens=candidate_lens,
    )
    _, attn_slots, res_lens = torch.ops.nanovllm_dsa.packed_scatter_copy_out(
        hbm_bytes,
        dram_bytes,
        hbm_block_table,
        dram_block_table,
        src_ids_2d,
        dst_slots_2d,
        copy_counts,
        cache_tokens,
        candidate_lens,
        actual_seq_lengths_kv,
        int(max_tail_tokens),
        attention_slots,
        resident_seq_lengths,
    )
    torch.npu.synchronize()
    _log_packed_attention_metadata_once(
        hbm_block_count=int(hbm_packed_kv.shape[0]),
        hbm_block_table=hbm_block_table,
        attention_slots=attn_slots,
        resident_seq_lengths=res_lens,
        cache_tokens=cache_tokens,
        candidate_lens=candidate_lens,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
    )
    _check_packed_roundtrip_once(
        hbm_packed_kv=hbm_packed_kv,
        hbm_block_table=hbm_block_table,
        dram_block_table=dram_block_table,
        source_token_ids=src_ids_2d,
        destination_slots=dst_slots_2d,
        copy_counts=copy_counts,
    )
    return attn_slots, res_lens


def packed_full_kv_cache_blocks(
    *,
    hbm_packed_kv: torch.Tensor,
    dram_packed_kv: torch.Tensor,
    src_block_ids: torch.Tensor,
    dst_block_ids: torch.Tensor,
) -> None:
    """A5 W4A4C8 final-Prefill packed full-block HBM -> swapped-DRAM dump。"""

    _capture_packed_dump_payload_once(
        hbm_packed_kv=hbm_packed_kv,
        src_block_ids=src_block_ids,
        dst_block_ids=dst_block_ids,
    )
    hbm_bytes = (
        hbm_packed_kv
        if hbm_packed_kv.dtype == torch.int8
        else hbm_packed_kv.view(torch.int8)
    )
    dram_bytes = (
        dram_packed_kv
        if dram_packed_kv.dtype == torch.int8
        else dram_packed_kv.view(torch.int8)
    )
    torch.ops.nanovllm_dsa.packed_full_block_dump(
        hbm_bytes,
        dram_bytes,
        src_block_ids,
        dst_block_ids,
    )


def _tensor_numerics(tensor: torch.Tensor) -> str:
    raw = (
        tensor.detach()
        .contiguous()
        .view(torch.int8)
        .cpu()
        .numpy()
        .tobytes()
    )
    values = tensor.detach().to(torch.float32).cpu()
    finite = torch.isfinite(values)
    finite_count = int(finite.sum().item())
    finite_values = values[finite]
    if finite_values.numel() == 0:
        absmax = float("nan")
        mean = float("nan")
        l2 = float("nan")
    else:
        absmax = float(finite_values.abs().max().item())
        mean = float(finite_values.mean().item())
        l2 = float(torch.linalg.vector_norm(finite_values).item())
    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"finite={finite_count}/{tensor.numel()} "
        f"absmax={absmax:.6g} mean={mean:.6g} l2={l2:.6g} "
        f"crc32={zlib.crc32(raw):08x}"
    )


def _log_packed_qsfa_io_once(
    *,
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    attention_slots: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Trace the exact packed rows and numerics consumed/returned by QSFA."""

    global _PACKED_QSFA_DIAG_LOGGED
    if _PACKED_QSFA_DIAG_LOGGED:
        return
    if not logger.isEnabledFor(logging.INFO):
        return
    _PACKED_QSFA_DIAG_LOGGED = True

    try:
        slots = [
            int(value)
            for value in attention_slots[0, 0, :8].detach().cpu().tolist()
        ]
        max_logical_block = max(
            (
                value // _BLOCK_SIZE
                for value in slots
                if value >= 0
            ),
            default=-1,
        )
        table = (
            [
                int(value)
                for value in block_table[
                    0,
                    : max_logical_block + 1,
                ]
                .detach()
                .cpu()
                .tolist()
            ]
            if max_logical_block >= 0
            else []
        )
        routes: list[str] = []
        for slot in slots:
            if slot < 0:
                routes.append(f"slot={slot}:invalid")
                continue
            logical_block = slot // _BLOCK_SIZE
            offset = slot % _BLOCK_SIZE
            if logical_block >= len(table):
                routes.append(
                    f"slot={slot}:logical_block={logical_block}:"
                    "table_oob"
                )
                continue
            physical_block = table[logical_block]
            if not 0 <= physical_block < int(packed_kv.shape[0]):
                routes.append(
                    f"slot={slot}:logical_block={logical_block}:"
                    f"physical_block={physical_block}:invalid"
                )
                continue
            packed_row = (
                packed_kv[physical_block, offset, 0]
                .detach()
                .cpu()
                .contiguous()
            )
            routes.append(
                f"slot={slot}/L{logical_block}/P{physical_block}/"
                f"o{offset}:[{_packed_row_summary(packed_row)}]"
            )

        logger.info(
            "[DSA_A5_C8_QSFA_IO] modes=key_quant_mode:2,"
            "value_quant_mode:2,attention_mode:2,"
            "quant_scale_repo_mode:1,tile_size:128,rope_head_dim:64 "
            "query_nope=[%s] query_rope=[%s] output=[%s] "
            "packed_shape=%s packed_dtype=%s packed_stride=%s "
            "attention_shape=%s actual_query=%s resident=%s routes=%s",
            _tensor_numerics(query[..., :512]),
            _tensor_numerics(query[..., 512:576]),
            _tensor_numerics(output),
            tuple(packed_kv.shape),
            packed_kv.dtype,
            tuple(packed_kv.stride()),
            tuple(attention_slots.shape),
            actual_seq_lengths_query.detach().cpu().tolist(),
            resident_seq_lengths.detach().cpu().tolist(),
            " | ".join(routes),
        )
    except Exception:
        logger.exception("[DSA_A5_C8_QSFA_IO] diagnostic_failed=true")


def sparse_flash_attention_for_offload_c8(
    *,
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    attention_slots: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    resident_seq_lengths: torch.Tensor,
    scale_value: float,
) -> torch.Tensor:
    """C8 SFA：对 packed KV 执行量化 sparse+tail attention。"""

    packed_kv_input = packed_kv.clone(
        memory_format=torch.contiguous_format
    )
    result = torch.ops.nanovllm_dsa.sparse_and_tail_attention_c8(
        query,
        packed_kv_input,
        attention_slots,
        block_table,
        actual_seq_lengths_query,
        resident_seq_lengths,
        float(scale_value),
    )
    torch.npu.synchronize()
    _log_packed_qsfa_io_once(
        query=query,
        packed_kv=packed_kv_input,
        attention_slots=attention_slots,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        resident_seq_lengths=resident_seq_lengths,
        output=result,
    )
    return result


from vllm_ascend.dsa_offload.a5_c8_policy import (  # noqa: E402
    install_a5_c8_runtime_policy,
)
from vllm_ascend.dsa_offload.a5_c8_split_cache import (  # noqa: E402
    install_a5_c8_split_cache_policy,
)

install_a5_c8_runtime_policy()
install_a5_c8_split_cache_policy()
