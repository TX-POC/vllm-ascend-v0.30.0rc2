#!/usr/bin/env python3
"""Ascend 950 correctness/performance test for packed-C8 DRAM -> HBM copy."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch

import nanovllm_dsa_a5
import torch_npu  # type: ignore  # noqa: E402,F401


BLOCK_SIZE = 128
PACKED_DIM = 656
TOPK = 2048
POISON = -91


@dataclass
class Case:
    device: torch.device
    device_name: str
    max_tail: int
    dram_cpu: torch.Tensor
    dram: torch.Tensor
    hbm: torch.Tensor
    dram_table_cpu: torch.Tensor
    hbm_table_cpu: torch.Tensor
    dram_table: torch.Tensor
    hbm_table: torch.Tensor
    sources_cpu: torch.Tensor
    slots_cpu: torch.Tensor
    counts_cpu: torch.Tensor
    cache_tokens_cpu: torch.Tensor
    candidates_cpu: torch.Tensor
    actual_kv_cpu: torch.Tensor
    sources: torch.Tensor
    slots: torch.Tensor
    counts: torch.Tensor
    cache_tokens: torch.Tensor
    candidates: torch.Tensor
    actual_kv: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--source-len", type=int, default=20096)
    parser.add_argument("--cache-tokens", type=int, default=6144)
    parser.add_argument("--tail-tokens", type=int, default=257)
    parser.add_argument("--max-tail-tokens", type=int, default=512)
    parser.add_argument("--copy-min", type=int, default=0)
    parser.add_argument("--copy-max", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--sector-race-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.source_len < TOPK or args.source_len % BLOCK_SIZE:
        raise ValueError("source length must be block aligned and >= 2048")
    if args.cache_tokens < TOPK or args.cache_tokens % BLOCK_SIZE:
        raise ValueError("cache tokens must be block aligned and >= 2048")
    if not 0 <= args.tail_tokens <= args.max_tail_tokens:
        raise ValueError("tail tokens must be in [0,max_tail_tokens]")
    if not 0 <= args.copy_min <= args.copy_max <= TOPK:
        raise ValueError("copy range must be within [0,2048]")
    if (
        args.warmup < 0
        or args.iters <= 0
        or args.sector_race_iters <= 0
    ):
        raise ValueError(
            "warmup must be non-negative and all iteration counts positive"
        )


def require_a5(device: torch.device, allow_non_a5: bool) -> str:
    index = device.index if device.index is not None else torch.npu.current_device()
    getter = getattr(torch.npu, "get_device_name", torch_npu.npu.get_device_name)
    name = getter(index)
    if "950" not in name.lower() and not allow_non_a5:
        raise RuntimeError(
            f"expected Ascend 950, got {name!r}; "
            "use --allow-non-a5 only for debugging"
        )
    return name


def swapped_from_cpu(cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not hasattr(torch_npu, "empty_with_swapped_memory"):
        raise RuntimeError("torch_npu.empty_with_swapped_memory is required")
    tensor = torch_npu.empty_with_swapped_memory(
        cpu.shape, dtype=cpu.dtype, device=device
    )
    tensor.zero_()
    staging = cpu.to(device)
    tensor.add_(staging)
    torch.npu.synchronize()
    del staging
    torch.npu.empty_cache()
    return tensor


def random_private_table(
    batch: int,
    blocks_per_row: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    physical_blocks = batch * blocks_per_row
    table = torch.randperm(
        physical_blocks, generator=generator, dtype=torch.int64
    ).reshape(batch, blocks_per_row)
    return table.to(torch.int32).contiguous(), physical_blocks


def random_shared_table(
    batch: int,
    blocks_per_row: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    table = torch.stack(
        [
            torch.randperm(
                blocks_per_row, generator=generator, dtype=torch.int64
            ).to(torch.int32)
            for _ in range(batch)
        ]
    )
    return table.contiguous(), blocks_per_row


def make_case(args: argparse.Namespace) -> Case:
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)
    generator = torch.Generator().manual_seed(args.seed)
    source_blocks_per_row = args.source_len // BLOCK_SIZE
    resident_tokens = args.cache_tokens + args.max_tail_tokens
    resident_blocks_per_row = max(
        1, (resident_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    )
    dram_table_cpu, dram_blocks = random_shared_table(
        args.batch_size, source_blocks_per_row, generator
    )
    hbm_table_cpu, hbm_blocks = random_private_table(
        args.batch_size, resident_blocks_per_row, generator
    )
    dram_cpu = torch.randint(
        -128,
        128,
        (dram_blocks, BLOCK_SIZE, 1, PACKED_DIM),
        generator=generator,
        dtype=torch.int16,
    ).to(torch.int8)

    sources_cpu = torch.full(
        (args.batch_size, 1, TOPK), -1, dtype=torch.int32
    )
    slots_cpu = torch.empty_like(sources_cpu)
    counts = torch.randint(
        args.copy_min,
        args.copy_max + 1,
        (args.batch_size,),
        generator=generator,
        dtype=torch.int64,
    )
    if args.batch_size > 1:
        counts[0] = args.copy_min
        counts[1] = args.copy_max
    for row in range(args.batch_size):
        count = int(counts[row])
        sources_cpu[row, 0, :count] = torch.randperm(
            args.source_len, generator=generator
        )[:count].to(torch.int32)
        slots_cpu[row, 0] = torch.randperm(
            args.cache_tokens, generator=generator
        )[:TOPK].to(torch.int32)

    counts_cpu = counts.to(torch.int32)
    cache_tokens_cpu = torch.full(
        (args.batch_size,), args.cache_tokens, dtype=torch.int32
    )
    candidates_cpu = torch.full(
        (args.batch_size,), args.source_len, dtype=torch.int32
    )
    actual_kv_cpu = candidates_cpu + args.tail_tokens
    return Case(
        device=device,
        device_name=device_name,
        max_tail=args.max_tail_tokens,
        dram_cpu=dram_cpu,
        dram=swapped_from_cpu(dram_cpu, device),
        hbm=torch.empty(
            (hbm_blocks, BLOCK_SIZE, 1, PACKED_DIM),
            dtype=torch.int8,
            device=device,
        ),
        dram_table_cpu=dram_table_cpu,
        hbm_table_cpu=hbm_table_cpu,
        dram_table=dram_table_cpu.to(device),
        hbm_table=hbm_table_cpu.to(device),
        sources_cpu=sources_cpu,
        slots_cpu=slots_cpu,
        counts_cpu=counts_cpu,
        cache_tokens_cpu=cache_tokens_cpu,
        candidates_cpu=candidates_cpu,
        actual_kv_cpu=actual_kv_cpu,
        sources=sources_cpu.to(device),
        slots=slots_cpu.to(device),
        counts=counts_cpu.to(device),
        cache_tokens=cache_tokens_cpu.to(device),
        candidates=candidates_cpu.to(device),
        actual_kv=actual_kv_cpu.to(device),
    )


def physical_rows(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    source_rows: list[torch.Tensor] = []
    destination_rows: list[torch.Tensor] = []
    for row, count in enumerate(case.counts_cpu.tolist()):
        source = case.sources_cpu[row, 0, :count].to(torch.int64)
        destination = case.slots_cpu[row, 0, :count].to(torch.int64)
        source_rows.append(
            case.dram_table_cpu[row].to(torch.int64)[source // BLOCK_SIZE]
            * BLOCK_SIZE
            + source % BLOCK_SIZE
        )
        destination_rows.append(
            case.hbm_table_cpu[row].to(torch.int64)[destination // BLOCK_SIZE]
            * BLOCK_SIZE
            + destination % BLOCK_SIZE
        )
    return torch.cat(source_rows), torch.cat(destination_rows)


def launch(
    case: Case,
    counts: torch.Tensor | None = None,
    attention_slots: torch.Tensor | None = None,
    resident_lengths: torch.Tensor | None = None,
):
    args = (
        case.hbm,
        case.dram,
        case.hbm_table,
        case.dram_table,
        case.sources,
        case.slots,
        case.counts if counts is None else counts,
        case.cache_tokens,
        case.candidates,
        case.actual_kv,
        case.max_tail,
    )
    if attention_slots is None:
        return nanovllm_dsa_a5.packed_scatter_copy(*args)
    return nanovllm_dsa_a5.packed_scatter_copy_out(
        *args, attention_slots, resident_lengths
    )


def check_copy(case: Case) -> None:
    source_rows, destination_rows = physical_rows(case)
    case.hbm.fill_(POISON)
    torch.npu.synchronize()
    hbm_alias, attention_slots, resident_lengths = launch(case)
    torch.npu.synchronize()
    if hbm_alias.data_ptr() != case.hbm.data_ptr():
        raise AssertionError("packed HBM output does not alias the input")
    if destination_rows.numel():
        expected = case.dram_cpu.view(-1, PACKED_DIM)[source_rows]
        actual = case.hbm.view(-1, PACKED_DIM)[
            destination_rows.to(case.device)
        ].cpu()
        if not torch.equal(actual, expected):
            raise AssertionError("packed DRAM->HBM bytes differ")
    active = set(destination_rows.tolist())
    guard = next(
        index
        for index in range(case.hbm.numel() // PACKED_DIM)
        if index not in active
    )
    if not bool(
        torch.all(case.hbm.view(-1, PACKED_DIM)[guard] == POISON).item()
    ):
        raise AssertionError("inactive packed-HBM guard row was modified")

    actual_slots = attention_slots.cpu()
    if not torch.equal(actual_slots[:, :, :TOPK], case.slots_cpu):
        raise AssertionError("attention topK slots differ from LIDU slots")
    for row in range(case.counts_cpu.numel()):
        tail_len = int(case.actual_kv_cpu[row] - case.candidates_cpu[row])
        expected_tail = torch.arange(
            int(case.cache_tokens_cpu[row]),
            int(case.cache_tokens_cpu[row]) + tail_len,
            dtype=torch.int32,
        )
        if not torch.equal(
            actual_slots[row, 0, TOPK : TOPK + tail_len], expected_tail
        ):
            raise AssertionError("tail slots are not [C,C+tail)")
        if bool((actual_slots[row, 0, TOPK + tail_len :] != -1).any()):
            raise AssertionError("attention slot padding must be -1")
    expected_lengths = case.cache_tokens_cpu + (
        case.actual_kv_cpu - case.candidates_cpu
    )
    actual_lengths = resident_lengths.cpu()
    if not torch.equal(actual_lengths, expected_lengths):
        raise AssertionError(
            "resident lengths differ from C+tail: "
            f"actual={actual_lengths.tolist()} "
            f"expected={expected_lengths.tolist()}"
        )

    out_slots = torch.empty_like(attention_slots)
    out_lengths = torch.empty_like(resident_lengths)
    case.hbm.fill_(POISON)
    out_slots.fill_(-123456789)
    out_lengths.fill_(-123456789)
    torch.npu.synchronize()
    outputs = launch(
        case, attention_slots=out_slots, resident_lengths=out_lengths
    )
    torch.npu.synchronize()
    if tuple(tensor.data_ptr() for tensor in outputs) != (
        case.hbm.data_ptr(),
        out_slots.data_ptr(),
        out_lengths.data_ptr(),
    ):
        raise AssertionError("caller-owned packed SCATTER addresses changed")
    if destination_rows.numel():
        expected = case.dram_cpu.view(-1, PACKED_DIM)[source_rows]
        actual = case.hbm.view(-1, PACKED_DIM)[
            destination_rows.to(case.device)
        ].cpu()
        if not torch.equal(actual, expected):
            raise AssertionError(
                "caller-owned packed SCATTER DRAM->HBM bytes differ"
            )
    if not bool(
        torch.all(case.hbm.view(-1, PACKED_DIM)[guard] == POISON).item()
    ):
        raise AssertionError(
            "caller-owned packed SCATTER modified an inactive guard row"
        )
    if not torch.equal(out_slots.cpu(), actual_slots):
        raise AssertionError(
            "caller-owned packed SCATTER attention metadata differs"
        )
    if not torch.equal(out_lengths.cpu(), expected_lengths):
        raise AssertionError(
            "caller-owned packed SCATTER resident lengths differ: "
            f"actual={out_lengths.cpu().tolist()} "
            f"expected={expected_lengths.tolist()}"
        )
    print(
        "A5_PACKED_C8_SCATTER_CHECK "
        f"copied_tokens={int(case.counts_cpu.sum())} row_bytes={PACKED_DIM} "
        "allocator=empty_with_swapped_memory byte_exact=1 "
        "guard_unchanged=1 topk_tail_indices=1 out_alias=1 ok=1",
        flush=True,
    )


def check_dense_sector_pairs(case: Case, iters: int) -> None:
    """Stress each 32-byte sector shared by two adjacent 656-byte rows."""

    sources_cpu = case.sources_cpu.clone()
    slots_cpu = case.slots_cpu.clone()
    counts_cpu = torch.zeros_like(case.counts_cpu)
    sources_cpu[0, 0] = torch.arange(TOPK, dtype=torch.int32)
    slots_cpu[0, 0] = torch.arange(TOPK, dtype=torch.int32)
    counts_cpu[0] = TOPK

    sources = sources_cpu.to(case.device)
    slots = slots_cpu.to(case.device)
    counts = counts_cpu.to(case.device)
    attention_slots = torch.empty(
        (case.counts.size(0), 1, TOPK + case.max_tail),
        dtype=torch.int32,
        device=case.device,
    )
    resident_lengths = torch.empty_like(case.counts)

    logical_rows = torch.arange(TOPK, dtype=torch.int64)
    source_rows = (
        case.dram_table_cpu[0].to(torch.int64)[
            logical_rows // BLOCK_SIZE
        ]
        * BLOCK_SIZE
        + logical_rows % BLOCK_SIZE
    )
    destination_rows = (
        case.hbm_table_cpu[0].to(torch.int64)[
            logical_rows // BLOCK_SIZE
        ]
        * BLOCK_SIZE
        + logical_rows % BLOCK_SIZE
    )
    expected = case.dram_cpu.view(-1, PACKED_DIM)[source_rows]
    destination_rows_npu = destination_rows.to(case.device)

    for iteration in range(iters):
        case.hbm.fill_(POISON)
        nanovllm_dsa_a5.packed_scatter_copy_out(
            case.hbm,
            case.dram,
            case.hbm_table,
            case.dram_table,
            sources,
            slots,
            counts,
            case.cache_tokens,
            case.candidates,
            case.actual_kv,
            case.max_tail,
            attention_slots,
            resident_lengths,
        )
        torch.npu.synchronize()
        actual = case.hbm.view(-1, PACKED_DIM)[destination_rows_npu].cpu()
        if not torch.equal(actual, expected):
            diff = actual != expected
            raise AssertionError(
                "packed KSC dense sector-pair mismatch: "
                f"iteration={iteration} mismatch_rows="
                f"{int(diff.any(dim=1).sum())} mismatch_bytes={int(diff.sum())}"
            )

    print(
        "A5_PACKED_C8_SCATTER_SECTOR_PAIR_CHECK "
        f"copied_tokens={TOPK} shared_32b_pairs={TOPK // 2} "
        f"iters={iters} byte_exact=1 ok=1",
        flush=True,
    )


def check_zero_copy_and_dense_row(case: Case) -> None:
    zero_counts = torch.zeros_like(case.counts)
    case.hbm.fill_(POISON)
    _, slots, _ = launch(case, counts=zero_counts)
    torch.npu.synchronize()
    if not bool(torch.all(case.hbm == POISON).item()):
        raise AssertionError("copy_count=0 modified packed HBM")
    if not torch.equal(slots.cpu()[:, :, :TOPK], case.slots_cpu):
        raise AssertionError("zero-copy path did not publish topK slots")

    dense_cache = case.cache_tokens.clone()
    dense_candidates = case.candidates.clone()
    dense_actual = case.actual_kv.clone()
    dense_cache[0] = 0
    dense_candidates[0] = 0
    dense_actual[0] = min(1024, TOPK + case.max_tail)
    _, dense_slots, dense_lengths = nanovllm_dsa_a5.packed_scatter_copy(
        case.hbm,
        case.dram,
        case.hbm_table,
        case.dram_table,
        case.sources,
        case.slots,
        zero_counts,
        dense_cache,
        dense_candidates,
        dense_actual,
        case.max_tail,
    )
    torch.npu.synchronize()
    dense_len = int(dense_actual[0].cpu())
    if not torch.equal(
        dense_slots[0, 0, :dense_len].cpu(),
        torch.arange(dense_len, dtype=torch.int32),
    ):
        raise AssertionError("C=0 row did not publish dense resident slots")
    if bool((dense_slots[0, 0, dense_len:].cpu() != -1).any()):
        raise AssertionError("C=0 row padding must be -1")
    if int(dense_lengths[0].cpu()) != dense_len:
        raise AssertionError("C=0 resident length is wrong")
    print(
        "A5_PACKED_C8_SCATTER_ZERO_COPY_CHECK "
        "hbm_unchanged=1 metadata_published=1 mixed_dense_row=1 ok=1",
        flush=True,
    )


def benchmark(case: Case, warmup: int, iters: int) -> None:
    attention_slots = torch.empty(
        (case.counts.size(0), 1, TOPK + case.max_tail),
        dtype=torch.int32,
        device=case.device,
    )
    resident_lengths = torch.empty_like(case.counts)
    for _ in range(warmup):
        launch(
            case,
            attention_slots=attention_slots,
            resident_lengths=resident_lengths,
        )
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        launch(
            case,
            attention_slots=attention_slots,
            resident_lengths=resident_lengths,
        )
        end.record()
    ends[-1].synchronize()
    avg_us = statistics.mean(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    ) * 1000
    payload_bytes = int(case.counts_cpu.sum()) * PACKED_DIM
    payload_gbps = payload_bytes / (avg_us * 1000) if avg_us else 0.0
    print(
        "A5_PACKED_C8_SCATTER_RESULT "
        f"batch={case.counts.size(0)} "
        f"copied_tokens={int(case.counts_cpu.sum())} "
        f"avg_us={avg_us:.3f} payload_gbps={payload_gbps:.3f} "
        f"warmup={warmup} iters={iters}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    case = make_case(args)
    print(
        "A5_PACKED_C8_SCATTER_CONFIG "
        f"device={case.device} device_name={case.device_name!r} "
        f"batch={args.batch_size} source_len={args.source_len} "
        f"cache_tokens={args.cache_tokens} tail_tokens={args.tail_tokens} "
        f"max_tail_tokens={args.max_tail_tokens} "
        f"copy_range=[{args.copy_min},{args.copy_max}] "
        f"opapi={nanovllm_dsa_a5.local_opapi_path()}",
        flush=True,
    )
    check_copy(case)
    check_dense_sector_pairs(case, args.sector_race_iters)
    check_zero_copy_and_dense_row(case)
    benchmark(case, args.warmup, args.iters)
    print("A5_PACKED_C8_SCATTER_UT_OK", flush=True)


if __name__ == "__main__":
    main()
