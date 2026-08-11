#!/usr/bin/env python3
"""Correctness and performance tests for nano-vLLM's Ascend 950 LIDU."""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass

import torch

import nanovllm_dsa_a5
import torch_npu  # type: ignore  # noqa: E402,F401


BLOCK_SIZE = 128
HEAD_DIM = 128
TOPK = 2048
MAX_SOURCE_CAPACITY = 1 << 18
MAX_CACHE_TOKENS = 16256


@dataclass
class Case:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    req_entries: torch.Tensor
    req_entries_cpu: torch.Tensor
    cache_tokens: torch.Tensor
    candidate_lens: torch.Tensor
    block_table: torch.Tensor
    native_topk: torch.Tensor
    initial_pool: torch.Tensor
    target_misses: list[int]
    source_capacity: int


def csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def miss_ranges(value: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for item in value.split(","):
        lo_text, separator, hi_text = item.strip().partition(":")
        if not separator:
            raise argparse.ArgumentTypeError("miss ranges use MIN:MAX")
        lo, hi = int(lo_text), int(hi_text)
        if not 0 <= lo <= hi <= TOPK:
            raise argparse.ArgumentTypeError(
                f"invalid miss range {item!r}; require 0 <= MIN <= MAX <= {TOPK}"
            )
        result.append((lo, hi))
    if not result:
        raise argparse.ArgumentTypeError("expected at least one miss range")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--mode", choices=("all", "check", "bench", "profile"), default="all")
    parser.add_argument("--batch-sizes", type=csv_ints, default=csv_ints("24"))
    parser.add_argument("--source-lens", type=csv_ints, default=csv_ints("20096"))
    parser.add_argument("--heads", type=csv_ints, default=csv_ints("32,64"))
    parser.add_argument("--cache-tokens", type=csv_ints, default=csv_ints("6144"))
    parser.add_argument("--miss-ranges", type=miss_ranges, default=miss_ranges("0:300"))
    parser.add_argument("--pool-extra", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--profile-replays", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("all batch sizes must be positive")
    if any(length <= 0 or length % BLOCK_SIZE for length in args.source_lens):
        raise ValueError("source lengths must be positive multiples of 128")
    if any(length > MAX_SOURCE_CAPACITY for length in args.source_lens):
        raise ValueError("source length exceeds the 18-bit token-ID capacity")
    if any(heads not in (32, 64) for heads in args.heads):
        raise ValueError("LIDU heads must be 32 or 64")
    valid_budgets = lambda c: c == 0 or (TOPK <= c <= MAX_CACHE_TOKENS and c % BLOCK_SIZE == 0)
    if any(not valid_budgets(tokens) for tokens in args.cache_tokens):
        raise ValueError("cache tokens must be 0 or a block-aligned value in [2048,16256]")
    if args.pool_extra < 0 or args.warmup < 0 or args.iters <= 0:
        raise ValueError("pool-extra/warmup must be non-negative and iters positive")


def require_a5(device: torch.device, allow_non_a5: bool) -> None:
    index = device.index if device.index is not None else torch.npu.current_device()
    getter = getattr(torch.npu, "get_device_name", torch_npu.npu.get_device_name)
    name = getter(index)
    if "950" not in name.lower() and not allow_non_a5:
        raise RuntimeError(f"expected Ascend 950, got {name!r}; use --allow-non-a5 only for debugging")


def native_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    op = getattr(torch_npu, "npu_lightning_indexer", None)
    if op is None:
        namespace = getattr(torch.ops, "_C_ascend", None)
        op = getattr(namespace, "npu_lightning_indexer", None) if namespace else None
    if op is None:
        raise RuntimeError("the native A5 npu_lightning_indexer baseline is not registered")
    output = op(
        query=query.unsqueeze(1),
        key=key,
        weights=weights.unsqueeze(1),
        actual_seq_lengths_query=None,
        actual_seq_lengths_key=candidate_lens,
        block_table=block_table,
        layout_query="BSND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=0,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        return_value=False,
    )
    if isinstance(output, torch.Tensor):
        return output.reshape(query.size(0), TOPK)
    for tensor in output:
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.int32:
            return tensor.reshape(query.size(0), TOPK)
    raise RuntimeError("native LightningIndexer returned no int32 top-k tensor")


def feasible_miss(candidate_len: int, cache_tokens: int, miss: int) -> bool:
    if cache_tokens == 0:
        return miss == 0
    hits = TOPK - miss
    victims = cache_tokens - hits
    return candidate_len >= cache_tokens and 0 <= victims <= candidate_len - TOPK


def build_pool(
    native_topk: torch.Tensor,
    source_capacity: int,
    candidate_lens: list[int],
    budgets: list[int],
    target_misses: list[int],
    req_entries_cpu: torch.Tensor,
    pool_size: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    capacity = source_capacity
    pool = torch.full((pool_size, capacity), -1, dtype=torch.int32)
    topk_cpu = native_topk.cpu().to(torch.int64)
    for batch_row, (candidate_len, budget, miss) in enumerate(
        zip(candidate_lens, budgets, target_misses)
    ):
        pool_row = int(req_entries_cpu[batch_row])
        if budget == 0:
            continue
        if not feasible_miss(candidate_len, budget, miss):
            raise ValueError(
                f"cannot construct C={budget}, miss={miss}, candidate_len={candidate_len}"
            )
        selected = topk_cpu[batch_row]
        hit_count = TOPK - miss
        hit_ids = selected[torch.randperm(TOPK, generator=generator)[:hit_count]]
        outside_mask = torch.ones(candidate_len, dtype=torch.bool)
        outside_mask[selected] = False
        outside = torch.arange(candidate_len, dtype=torch.int64)[outside_mask]
        victim_count = budget - hit_count
        victim_ids = outside[
            torch.randperm(outside.numel(), generator=generator)[:victim_count]
        ]
        token_ids = torch.cat((hit_ids, victim_ids))
        slots = torch.randperm(budget, generator=generator, dtype=torch.int64).to(torch.int32)
        pool[pool_row, token_ids] = slots
    return pool


def make_case(
    device: torch.device,
    batch: int,
    source_len: int,
    heads: int,
    budgets: list[int],
    miss_range: tuple[int, int],
    pool_extra: int,
    seed: int,
    candidate_lens_cpu: list[int] | None = None,
) -> Case:
    if len(budgets) != batch:
        raise ValueError("budget list must match batch")
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    blocks = source_len // BLOCK_SIZE
    block_table_cpu = torch.stack(
        [torch.randperm(blocks, dtype=torch.int64).to(torch.int32) for _ in range(batch)]
    )
    key = torch.empty((blocks, BLOCK_SIZE, 1, HEAD_DIM), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    query = torch.empty((batch, heads, HEAD_DIM), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    weights = torch.empty((batch, heads), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
    if candidate_lens_cpu is None:
        candidate_lens_cpu = [source_len] * batch
    if len(candidate_lens_cpu) != batch:
        raise ValueError("candidate length list must match batch")
    if any(length < TOPK or length > source_len for length in candidate_lens_cpu):
        raise ValueError("candidate lengths must be in [2048, source_capacity]")
    candidate_lens = torch.tensor(candidate_lens_cpu, dtype=torch.int32, device=device)
    block_table = block_table_cpu.to(device)
    native_topk = native_lightning_indexer(query, key, weights, candidate_lens, block_table)
    torch.npu.synchronize()

    rng = random.Random(seed + 1)
    target_misses: list[int] = []
    for candidate_len, budget in zip(candidate_lens_cpu, budgets):
        if budget == 0:
            target_misses.append(0)
            continue
        feasible = [
            miss for miss in range(miss_range[0], miss_range[1] + 1)
            if feasible_miss(candidate_len, budget, miss)
        ]
        if not feasible:
            raise ValueError(
                f"no feasible miss in {miss_range} for candidate_len={candidate_len}, C={budget}"
            )
        target_misses.append(rng.choice(feasible))
    # Multi-row cases pin both range endpoints for coverage.  A one-row
    # performance case keeps its seeded random sample, so 0:300 does not
    # silently degenerate into the zero-miss benchmark.
    if batch > 1:
        if budgets[0] > 0 and feasible_miss(
            candidate_lens_cpu[0], budgets[0], miss_range[0]
        ):
            target_misses[0] = miss_range[0]
        if budgets[1] > 0 and feasible_miss(
            candidate_lens_cpu[1], budgets[1], miss_range[1]
        ):
            target_misses[1] = miss_range[1]

    pool_size = batch + pool_extra
    req_entries_cpu = torch.randperm(pool_size, dtype=torch.int64)[:batch].to(torch.int32)
    initial_pool = build_pool(
        native_topk,
        source_len,
        candidate_lens_cpu,
        budgets,
        target_misses,
        req_entries_cpu,
        pool_size,
        seed + 2,
    ).to(device)
    return Case(
        query=query,
        key=key,
        weights=weights,
        req_entries=req_entries_cpu.to(device),
        req_entries_cpu=req_entries_cpu,
        cache_tokens=torch.tensor(budgets, dtype=torch.int32, device=device),
        candidate_lens=candidate_lens,
        block_table=block_table,
        native_topk=native_topk,
        initial_pool=initial_pool,
        target_misses=target_misses,
        source_capacity=source_len,
    )


def launch(case: Case, pool: torch.Tensor):
    return torch.ops.nanovllm_dsa.lidu_decode_update.default(
        case.query,
        case.key,
        case.weights,
        case.req_entries,
        pool,
        case.cache_tokens,
        case.candidate_lens,
        case.block_table,
    )


def launch_out(
    case: Case,
    pool: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
):
    return torch.ops.nanovllm_dsa.lidu_decode_update_out.default(
        case.query,
        case.key,
        case.weights,
        case.req_entries,
        pool,
        case.cache_tokens,
        case.candidate_lens,
        case.block_table,
        source_ids,
        destination_slots,
        miss_counts,
    )


def assert_pool_row(row: torch.Tensor, candidate_len: int, budget: int) -> None:
    valid_tokens = (row[:candidate_len] >= 0).nonzero().flatten()
    if budget == 0:
        if valid_tokens.numel() != 0:
            raise AssertionError("C=0 request mutated its cache row")
        return
    if valid_tokens.numel() != budget:
        raise AssertionError(f"cache cardinality={valid_tokens.numel()}, expected {budget}")
    slots = row[valid_tokens]
    if not torch.equal(torch.sort(slots).values, torch.arange(budget, dtype=torch.int32)):
        raise AssertionError("cache slots are not the exact unique range [0,C)")


def check_case(case: Case) -> None:
    pool = case.initial_pool.clone()
    old_pool = pool.cpu()
    source_ids, destination_slots, miss_counts, alias = launch(case, pool)
    torch.npu.synchronize()
    if alias.data_ptr() != pool.data_ptr():
        raise AssertionError("cache_slots_alias does not alias cache_slots_pool")
    sources = source_ids.reshape(case.query.size(0), TOPK).cpu()
    slots = destination_slots.reshape(case.query.size(0), TOPK).cpu()
    counts = miss_counts.cpu()
    updated = pool.cpu()
    native = case.native_topk.cpu().reshape(case.query.size(0), TOPK)
    active_rows = set(case.req_entries_cpu.tolist())
    for pool_row in range(updated.size(0)):
        if pool_row not in active_rows and not torch.equal(updated[pool_row], old_pool[pool_row]):
            raise AssertionError(f"inactive request-pool row {pool_row} was modified")
    for batch_row in range(case.query.size(0)):
        pool_row = int(case.req_entries_cpu[batch_row])
        budget = int(case.cache_tokens[batch_row].cpu())
        expected_miss = case.target_misses[batch_row]
        assert_pool_row(updated[pool_row], case.source_capacity, budget)
        if budget == 0:
            if int(counts[batch_row]) != 0 or bool((sources[batch_row] != -1).any()) or bool((slots[batch_row] != -1).any()):
                raise AssertionError("C=0 output must be miss_count=0 with -1 source/slot rows")
            continue
        if not torch.equal(torch.sort(sources[batch_row]).values, torch.sort(native[batch_row]).values):
            raise AssertionError(f"row {batch_row}: top-2048 set differs from native A5 LI")
        old_selected_slots = old_pool[pool_row].gather(0, sources[batch_row].long())
        new_selected_slots = updated[pool_row].gather(0, sources[batch_row].long())
        actual_miss = int(counts[batch_row])
        if actual_miss != expected_miss or actual_miss != int((old_selected_slots < 0).sum()):
            raise AssertionError(
                f"row {batch_row}: miss_count={actual_miss}, expected={expected_miss}"
            )
        if actual_miss and bool((old_selected_slots[:actual_miss] >= 0).any()):
            raise AssertionError("miss prefix contains a cache hit")
        if actual_miss < TOPK and bool((old_selected_slots[actual_miss:] < 0).any()):
            raise AssertionError("hit suffix contains a cache miss")
        if not torch.equal(slots[batch_row], new_selected_slots):
            raise AssertionError("published attention slots differ from updated cache state")

    # Repeating the same query must see a fully warm top-2048 cache.
    _, second_slots, second_counts, _ = launch(case, pool)
    torch.npu.synchronize()
    expected_second = torch.zeros_like(second_counts.cpu())
    if not torch.equal(second_counts.cpu(), expected_second):
        raise AssertionError(f"repeated update is not zero-miss: {second_counts.cpu().tolist()}")
    if bool((case.cache_tokens.cpu() > 0).any()) and bool((second_slots.cpu()[case.cache_tokens.cpu() > 0] < 0).any()):
        raise AssertionError("repeated update did not publish complete attention slots")

    # Caller-owned output path is the graph-capture contract.
    out_pool = case.initial_pool.clone()
    out_sources = torch.empty_like(source_ids)
    out_slots = torch.empty_like(destination_slots)
    out_counts = torch.empty_like(miss_counts)
    outputs = launch_out(case, out_pool, out_sources, out_slots, out_counts)
    torch.npu.synchronize()
    expected_ptrs = (out_sources.data_ptr(), out_slots.data_ptr(), out_counts.data_ptr(), out_pool.data_ptr())
    if tuple(tensor.data_ptr() for tensor in outputs) != expected_ptrs:
        raise AssertionError("lidu_decode_update_out did not preserve caller-owned addresses")
    if not torch.equal(outputs[0].cpu(), source_ids.cpu()) or not torch.equal(outputs[1].cpu(), destination_slots.cpu()) or not torch.equal(outputs[2].cpu(), miss_counts.cpu()):
        raise AssertionError("allocating and caller-owned LIDU paths disagree")

    print(
        "A5_LIDU_CHECK "
        f"heads={case.query.size(1)} batch={case.query.size(0)} "
        f"source_capacity={case.source_capacity} candidates={case.candidate_lens.cpu().tolist()} "
        f"budgets={case.cache_tokens.cpu().tolist()} "
        f"misses={counts.tolist()} unordered_unique_pool_entries=1 repeat_zero_miss=1 out_alias=1 ok=1",
        flush=True,
    )


def event_benchmark(case: Case, warmup: int, iters: int) -> tuple[float, float]:
    pool = case.initial_pool.clone()
    for _ in range(warmup):
        pool.copy_(case.initial_pool)
        launch(case, pool)
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    retained = []
    for start, end in zip(starts, ends):
        pool.copy_(case.initial_pool)
        start.record()
        retained.append(launch(case, pool))
        end.record()
    ends[-1].synchronize()
    lidu_us = statistics.mean(start.elapsed_time(end) for start, end in zip(starts, ends)) * 1000

    native_starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    native_ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(native_starts, native_ends):
        start.record()
        native_lightning_indexer(case.query, case.key, case.weights, case.candidate_lens, case.block_table)
        end.record()
    native_ends[-1].synchronize()
    native_us = statistics.mean(start.elapsed_time(end) for start, end in zip(native_starts, native_ends)) * 1000
    if not retained:
        raise AssertionError("timed outputs were not retained")
    return native_us, lidu_us


def check_meta() -> None:
    query = torch.empty((3, 32, HEAD_DIM), dtype=torch.bfloat16, device="meta")
    key = torch.empty((96, BLOCK_SIZE, 1, HEAD_DIM), dtype=torch.bfloat16, device="meta")
    weights = torch.empty((3, 32), dtype=torch.bfloat16, device="meta")
    req = torch.empty((3,), dtype=torch.int32, device="meta")
    pool = torch.empty((7, 12288), dtype=torch.int32, device="meta")
    budgets = torch.empty((3,), dtype=torch.int32, device="meta")
    lengths = torch.empty((3,), dtype=torch.int32, device="meta")
    table = torch.empty((3, 96), dtype=torch.int32, device="meta")
    outputs = torch.ops.nanovllm_dsa.lidu_decode_update.default(
        query, key, weights, req, pool, budgets, lengths, table
    )
    if [tuple(t.shape) for t in outputs] != [(3, 1, TOPK), (3, 1, TOPK), (3,), (7, 12288)]:
        raise AssertionError("LIDU Meta implementation returned wrong shapes")
    print("A5_LIDU_META_CHECK ok=1", flush=True)


def main() -> None:
    args = parse_args()
    check_args(args)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    require_a5(device, args.allow_non_a5)
    check_meta()

    # One mandatory mixed-C case proves C=0 no-op, arbitrary legal C, pool
    # indirection, and the maximum block-aligned 14-bit slot range.
    if args.mode in ("all", "check"):
        mixed = make_case(
            device,
            6,
            32768,
            32,
            [0, 3072, 6144, 8192, 12288, MAX_CACHE_TOKENS],
            (0, 300),
            args.pool_extra,
            args.seed + 100,
            [2048, 8192, 12288, 20096, 24576, 32768],
        )
        check_case(mixed)
        miss_edges = make_case(
            device,
            2,
            4096,
            32,
            [2048, 2048],
            (0, TOPK),
            args.pool_extra,
            args.seed + 200,
        )
        check_case(miss_edges)

    case_index = 0
    for heads in args.heads:
        for batch in args.batch_sizes:
            for source_len in args.source_lens:
                for budget in args.cache_tokens:
                    for miss_range in args.miss_ranges:
                        if not any(
                            feasible_miss(source_len, budget, miss)
                            for miss in range(miss_range[0], miss_range[1] + 1)
                        ):
                            print(
                                "A5_LIDU_SKIP "
                                f"heads={heads} batch={batch} source_len={source_len} "
                                f"C={budget} miss_range={miss_range} reason=infeasible",
                                flush=True,
                            )
                            continue
                        case = make_case(
                            device,
                            batch,
                            source_len,
                            heads,
                            [budget] * batch,
                            miss_range,
                            args.pool_extra,
                            args.seed + case_index,
                        )
                        case_index += 1
                        if args.mode in ("all", "check"):
                            check_case(case)
                        if args.mode in ("all", "bench"):
                            native_us, lidu_us = event_benchmark(case, args.warmup, args.iters)
                            print(
                                "A5_LIDU_RESULT "
                                f"heads={heads} batch={batch} source_len={source_len} C={budget} "
                                f"miss_range={miss_range[0]}:{miss_range[1]} "
                                f"actual_miss_mean={statistics.mean(case.target_misses):.3f} "
                                f"native_li_us={native_us:.3f} lidu_us={lidu_us:.3f} "
                                f"index_management_us={lidu_us - native_us:+.3f} "
                                f"warmup={args.warmup} iters={args.iters}",
                                flush=True,
                            )
                        if args.mode == "profile":
                            pool = case.initial_pool.clone()
                            for _ in range(args.profile_replays):
                                pool.copy_(case.initial_pool)
                                launch(case, pool)
                            torch.npu.synchronize()
    print("A5_LIDU_UT_OK", flush=True)


if __name__ == "__main__":
    main()
