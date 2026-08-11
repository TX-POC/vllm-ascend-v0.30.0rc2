#!/usr/bin/env python3
"""Correctness and performance tests for nano-vLLM's Ascend 950 sparse+tail MLA."""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass

import torch

import nanovllm_dsa_a5
import torch_npu  # type: ignore  # noqa: E402,F401


BLOCK_SIZE = 128
CKV_DIM = 512
KPE_DIM = 64
SPARSE_COUNT = 2048
MAX_SOURCE_CAPACITY = 1 << 18
MAX_CACHE_TOKENS = 16256


@dataclass
class Case:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    sparse_slots: torch.Tensor
    cache_tokens: torch.Tensor
    block_table: torch.Tensor
    actual_q: torch.Tensor
    actual_kv: torch.Tensor
    query_rope: torch.Tensor
    key_rope: torch.Tensor
    block_table_cpu: torch.Tensor
    sparse_slots_cpu: torch.Tensor
    scale: float
    source_len: int
    cache_budget: int
    tail_tokens: int


def csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--mode", choices=("all", "check", "bench", "profile"), default="all")
    parser.add_argument("--batch-sizes", type=csv_ints, default=csv_ints("24"))
    parser.add_argument("--source-lens", type=csv_ints, default=csv_ints("20096"))
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--cache-tokens", type=csv_ints, default=csv_ints("6144"))
    parser.add_argument("--tail-tokens", type=csv_ints, default=csv_ints("64"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--profile-replays", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("all batch sizes must be positive")
    if any(length <= 0 or length > MAX_SOURCE_CAPACITY for length in args.source_lens):
        raise ValueError("source lengths must be in [1,2^18]")
    valid_budget = lambda c: c == 0 or (
        SPARSE_COUNT <= c <= MAX_CACHE_TOKENS and c % BLOCK_SIZE == 0
    )
    if any(not valid_budget(tokens) for tokens in args.cache_tokens):
        raise ValueError("cache tokens must be 0 or block-aligned in [2048,16256]")
    if any(tail < 0 for tail in args.tail_tokens):
        raise ValueError("tail token counts must be non-negative")
    if not 1 <= args.heads <= 64:
        raise ValueError("heads must be in [1,64]")
    if args.warmup < 0 or args.iters <= 0 or args.profile_replays <= 0:
        raise ValueError("warmup must be non-negative; iters/profile-replays must be positive")


def require_a5(device: torch.device, allow_non_a5: bool) -> None:
    index = device.index if device.index is not None else torch.npu.current_device()
    getter = getattr(torch.npu, "get_device_name", torch_npu.npu.get_device_name)
    name = getter(index)
    if "950" not in name.lower() and not allow_non_a5:
        raise RuntimeError(f"expected Ascend 950, got {name!r}; use --allow-non-a5 only for debugging")


def make_case(
    device: torch.device,
    batch: int,
    heads: int,
    source_len: int,
    cache_budget: int,
    tail_tokens: int,
    seed: int,
) -> Case:
    actual_len = source_len if cache_budget == 0 else cache_budget + tail_tokens
    if actual_len > source_len:
        raise ValueError(
            f"C+tail={actual_len} exceeds source_len={source_len}; increase --source-lens"
        )
    if cache_budget > source_len:
        raise ValueError(f"C={cache_budget} exceeds source_len={source_len}")
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    blocks = math.ceil(source_len / BLOCK_SIZE)
    # KV is read-only here. Every request gets an independent random logical-to-
    # physical mapping over one shared physical pool, avoiding B copies of a
    # 131K-token cache while still exercising random paged addresses.
    generator = torch.Generator().manual_seed(seed + 1)
    block_table_cpu = torch.stack(
        [torch.randperm(blocks, generator=generator).to(torch.int32) for _ in range(batch)]
    )
    key = torch.empty((blocks, BLOCK_SIZE, 1, CKV_DIM), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    # GLM MLA uses the latent CKV tensor as both key and value.
    value = key
    key_rope = torch.empty((blocks, BLOCK_SIZE, 1, KPE_DIM), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    query = torch.empty((batch, heads, CKV_DIM), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    query_rope = torch.empty((batch, heads, KPE_DIM), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    if cache_budget:
        sparse_slots_cpu = torch.stack(
            [
                torch.randperm(cache_budget, generator=generator)[:SPARSE_COUNT].to(torch.int32)
                for _ in range(batch)
            ]
        )
    else:
        sparse_slots_cpu = torch.full((batch, SPARSE_COUNT), -1, dtype=torch.int32)
    return Case(
        query=query,
        key=key,
        value=value,
        sparse_slots=sparse_slots_cpu[:, None, :].to(device),
        cache_tokens=torch.full((batch,), cache_budget, dtype=torch.int32, device=device),
        block_table=block_table_cpu.to(device),
        actual_q=torch.arange(1, batch + 1, dtype=torch.int32, device=device),
        actual_kv=torch.full((batch,), actual_len, dtype=torch.int32, device=device),
        query_rope=query_rope,
        key_rope=key_rope,
        block_table_cpu=block_table_cpu,
        sparse_slots_cpu=sparse_slots_cpu,
        scale=1.0 / math.sqrt(CKV_DIM + KPE_DIM),
        source_len=source_len,
        cache_budget=cache_budget,
        tail_tokens=tail_tokens,
    )


def launch(case: Case) -> torch.Tensor:
    return torch.ops.nanovllm_dsa.sparse_and_tail_attention.default(
        case.query,
        case.key,
        case.value,
        case.sparse_slots,
        case.cache_tokens,
        case.block_table,
        case.actual_q,
        case.actual_kv,
        case.query_rope,
        case.key_rope,
        case.scale,
    )


def logical_tokens(case: Case, row: int) -> torch.Tensor:
    actual_len = int(case.actual_kv[row].cpu())
    if case.cache_budget == 0:
        return torch.arange(actual_len, dtype=torch.int64)
    tail = torch.arange(case.cache_budget, actual_len, dtype=torch.int64)
    return torch.cat((case.sparse_slots_cpu[row].to(torch.int64), tail))


def cpu_golden(case: Case) -> torch.Tensor:
    query = case.query.cpu().float()
    query_rope = case.query_rope.cpu().float()
    flat_key = case.key.cpu().float().view(-1, CKV_DIM)
    flat_value = case.value.cpu().float().view(-1, CKV_DIM)
    flat_rope = case.key_rope.cpu().float().view(-1, KPE_DIM)
    outputs: list[torch.Tensor] = []
    for row in range(case.query.size(0)):
        tokens = logical_tokens(case, row)
        physical = (
            case.block_table_cpu[row, tokens // BLOCK_SIZE].to(torch.int64) * BLOCK_SIZE
            + tokens.remainder(BLOCK_SIZE)
        )
        key = flat_key[physical]
        value = flat_value[physical]
        key_rope = flat_rope[physical]
        scores = (
            query[row] @ key.T + query_rope[row] @ key_rope.T
        ) * case.scale
        outputs.append(torch.softmax(scores, dim=-1) @ value)
    return torch.stack(outputs)


def check_case(case: Case) -> None:
    golden = cpu_golden(case)
    output = launch(case)
    torch.npu.synchronize()
    output_cpu = output.cpu().float()
    if not torch.isfinite(output_cpu).all():
        raise AssertionError("SFA produced NaN/Inf")
    torch.testing.assert_close(output_cpu, golden, rtol=0.02, atol=0.02)
    max_abs = float((output_cpu - golden).abs().max())
    cosine = float(torch.nn.functional.cosine_similarity(
        output_cpu.flatten(), golden.flatten(), dim=0
    ))
    if cosine < 0.999:
        raise AssertionError(f"SFA cosine similarity is too low: {cosine:.9f}")
    attended = int(logical_tokens(case, 0).numel())
    print(
        "A5_SPARSE_TAIL_CHECK "
        f"heads={case.query.size(1)} batch={case.query.size(0)} "
        f"source_len={case.source_len} C={case.cache_budget} tail={case.tail_tokens} "
        f"attended_tokens={attended} "
        f"max_abs={max_abs:.9f} cosine={cosine:.9f} finite=1 ok=1",
        flush=True,
    )


def benchmark(case: Case, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        launch(case)
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    retained: list[torch.Tensor] = []
    for start, end in zip(starts, ends):
        start.record()
        retained.append(launch(case))
        end.record()
    ends[-1].synchronize()
    if not retained:
        raise AssertionError("timed outputs were not retained")
    avg_us = statistics.mean(start.elapsed_time(end) for start, end in zip(starts, ends)) * 1000
    print(
        "A5_SPARSE_TAIL_RESULT "
        f"heads={case.query.size(1)} batch={case.query.size(0)} "
        f"source_len={case.source_len} C={case.cache_budget} tail={case.tail_tokens} "
        f"attended_tokens={logical_tokens(case, 0).numel()} avg_us={avg_us:.3f} "
        f"timer=npu_event warmup={warmup} iters={iters}",
        flush=True,
    )
    return avg_us


def check_meta(heads: int) -> None:
    query = torch.empty((3, heads, CKV_DIM), dtype=torch.bfloat16, device="meta")
    key = torch.empty((96, BLOCK_SIZE, 1, CKV_DIM), dtype=torch.bfloat16, device="meta")
    slots = torch.empty((3, 1, SPARSE_COUNT), dtype=torch.int32, device="meta")
    ints = torch.empty((3,), dtype=torch.int32, device="meta")
    table = torch.empty((3, 96), dtype=torch.int32, device="meta")
    q_rope = torch.empty((3, heads, KPE_DIM), dtype=torch.bfloat16, device="meta")
    k_rope = torch.empty((96, BLOCK_SIZE, 1, KPE_DIM), dtype=torch.bfloat16, device="meta")
    output = torch.ops.nanovllm_dsa.sparse_and_tail_attention.default(
        query, key, key, slots, ints, table, ints, ints, q_rope, k_rope, 1.0
    )
    if tuple(output.shape) != tuple(query.shape) or output.dtype != query.dtype:
        raise AssertionError("SFA Meta implementation returned wrong shape/dtype")
    print(f"A5_SPARSE_TAIL_META_CHECK heads={heads} ok=1", flush=True)


def main() -> None:
    args = parse_args()
    check_args(args)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    require_a5(device, args.allow_non_a5)
    check_meta(args.heads)
    print(
        "A5_SPARSE_TAIL_CONFIG "
        f"device={device} heads={args.heads} batch_sizes={args.batch_sizes} "
        f"source_lens={args.source_lens} cache_tokens={args.cache_tokens} "
        f"tail_tokens={args.tail_tokens} "
        f"opapi={nanovllm_dsa_a5.local_opapi_path()}",
        flush=True,
    )

    # Lightweight mandatory semantic coverage.
    if args.mode in ("all", "check"):
        smoke = make_case(device, 1, args.heads, 2048, 2048, 0, args.seed + 10)
        check_case(smoke)
        dense = make_case(device, 1, args.heads, 2048, 0, 0, args.seed + 11)
        check_case(dense)
        check_index = 0
        for budget in (3072, 6144, 8192, 12288):
            for tail in (0, 1, 64, 127, 257):
                source_len = budget + tail
                case = make_case(
                    device, 1, args.heads, source_len, budget, tail,
                    args.seed + 100 + check_index,
                )
                check_case(case)
                check_index += 1

    case_index = 0
    for batch in args.batch_sizes:
        for source_len in args.source_lens:
            for budget in args.cache_tokens:
                for tail in args.tail_tokens:
                    actual_len = source_len if budget == 0 else budget + tail
                    if budget > source_len or actual_len > source_len:
                        print(
                            "A5_SPARSE_TAIL_SKIP "
                            f"heads={args.heads} batch={batch} source_len={source_len} "
                            f"C={budget} tail={tail} reason=capacity",
                            flush=True,
                        )
                        continue
                    case = make_case(
                        device, batch, args.heads, source_len, budget, tail,
                        args.seed + 1000 + case_index,
                    )
                    if args.mode in ("all", "check"):
                        check_case(case)
                    if args.mode in ("all", "bench"):
                        benchmark(case, args.warmup, args.iters)
                    if args.mode == "profile":
                        for _ in range(args.profile_replays):
                            launch(case)
                        torch.npu.synchronize()
                        print(
                            "A5_SPARSE_TAIL_PROFILE_DONE "
                            f"heads={args.heads} batch={batch} source_len={source_len} "
                            f"C={budget} tail={tail} replays={args.profile_replays}",
                            flush=True,
                        )
                    case_index += 1
    print("A5_SPARSE_AND_TAIL_ATTENTION_UT_OK", flush=True)


if __name__ == "__main__":
    main()
