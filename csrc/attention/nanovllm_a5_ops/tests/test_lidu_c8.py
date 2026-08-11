#!/usr/bin/env python3
"""Validate the A5 C8 LightningIndexer plus request-pool update path."""

from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass

import torch

import nanovllm_dsa_a5
import torch_npu  # type: ignore  # noqa: E402,F401
from test_lidu import (
    MAX_CACHE_TOKENS,
    MAX_SOURCE_CAPACITY,
    assert_pool_row,
    build_pool,
    csv_ints,
    feasible_miss,
    miss_ranges,
)


BLOCK_SIZE = 128
HEAD_DIM = 128
TOPK = 2048


@dataclass
class C8Case:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    query_scale: torch.Tensor
    key_scale: torch.Tensor
    actual_q: torch.Tensor
    req_entries: torch.Tensor
    req_entries_cpu: torch.Tensor
    cache_tokens: torch.Tensor
    candidate_lens: torch.Tensor
    block_table: torch.Tensor
    native_topk: torch.Tensor
    initial_pool: torch.Tensor
    target_misses: list[int]
    source_capacity: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--mode", choices=("all", "check", "bench", "profile"), default="all"
    )
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
        raise ValueError("C8 LightningIndexer heads must be 32 or 64")
    valid_budget = lambda value: value == 0 or (
        TOPK <= value <= MAX_CACHE_TOKENS and value % BLOCK_SIZE == 0
    )
    if any(not valid_budget(value) for value in args.cache_tokens):
        raise ValueError("cache tokens must be 0 or block aligned in [2048,16256]")
    if args.pool_extra < 0 or args.warmup < 0 or args.iters <= 0:
        raise ValueError("pool-extra/warmup must be non-negative and iters positive")


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


def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quantized, scale = torch_npu.npu_dynamic_quant(
        tensor, dst_type=torch.float8_e4m3fn
    )
    return (
        quantized.contiguous(),
        scale.view(tensor.shape[:-1]).to(torch.float32).contiguous(),
    )


def normalized_hadamard_128(
    *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Match the normalized 128x128 Hadamard used by official GLM C8 LI."""

    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.size(0) < HEAD_DIM:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return (matrix / math.sqrt(HEAD_DIM)).to(dtype=dtype, device=device)


def official_c8_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    actual_q: torch.Tensor,
    candidate_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    """Call the same official A5 C8 LI ABI used by vLLM-Ascend."""

    op = getattr(torch_npu, "npu_quant_lightning_indexer", None)
    if op is None:
        namespace = getattr(torch.ops, "_C_ascend", None)
        op = (
            getattr(namespace, "npu_lightning_indexer_quant", None)
            if namespace is not None
            else None
        )
    if op is None:
        raise RuntimeError("official A5 C8 LightningIndexer is not registered")
    output = op(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        actual_seq_lengths_query=actual_q,
        actual_seq_lengths_key=candidate_lens,
        block_table=block_table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    topk = output[0] if isinstance(output, tuple) else output
    if not isinstance(topk, torch.Tensor) or topk.dtype != torch.int32:
        raise RuntimeError("official A5 C8 LI returned no int32 top-k tensor")
    if topk.numel() != query.size(0) * TOPK:
        raise RuntimeError(
            f"official A5 C8 LI returned unexpected shape {tuple(topk.shape)}"
        )
    return topk.reshape(query.size(0), TOPK).contiguous()


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
) -> C8Case:
    if len(budgets) != batch:
        raise ValueError("budget list must match batch")
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    blocks = source_len // BLOCK_SIZE
    block_table_cpu = torch.stack(
        [
            torch.randperm(blocks, dtype=torch.int64).to(torch.int32)
            for _ in range(batch)
        ]
    )
    query_fp = torch.empty(
        (batch, heads, HEAD_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-1, 1)
    key_fp = torch.empty(
        (blocks, BLOCK_SIZE, 1, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-1, 1)
    hadamard = normalized_hadamard_128(
        dtype=query_fp.dtype, device=device
    )
    query_fp = torch.matmul(query_fp, hadamard)
    key_fp = torch.matmul(key_fp, hadamard)
    query, query_scale = quantize_fp8(query_fp)
    key, key_scale = quantize_fp8(key_fp)
    weights = torch.empty(
        (batch, heads), dtype=torch.bfloat16, device=device
    ).uniform_(0.01, 1.0).contiguous()

    if candidate_lens_cpu is None:
        candidate_lens_cpu = [source_len] * batch
    if len(candidate_lens_cpu) != batch:
        raise ValueError("candidate length list must match batch")
    if any(length < TOPK or length > source_len for length in candidate_lens_cpu):
        raise ValueError("candidate lengths must be in [2048,source_capacity]")
    candidate_lens = torch.tensor(
        candidate_lens_cpu, dtype=torch.int32, device=device
    )
    actual_q = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    block_table = block_table_cpu.to(device)
    native_topk = official_c8_lightning_indexer(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        actual_q,
        candidate_lens,
        block_table,
    )
    torch.npu.synchronize()

    rng = random.Random(seed + 1)
    target_misses: list[int] = []
    for candidate_len, budget in zip(candidate_lens_cpu, budgets):
        if budget == 0:
            target_misses.append(0)
            continue
        feasible = [
            miss
            for miss in range(miss_range[0], miss_range[1] + 1)
            if feasible_miss(candidate_len, budget, miss)
        ]
        if not feasible:
            raise ValueError(
                f"no feasible miss in {miss_range} for "
                f"candidate_len={candidate_len}, C={budget}"
            )
        target_misses.append(rng.choice(feasible))
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
    req_entries_cpu = torch.randperm(pool_size, dtype=torch.int64)[:batch].to(
        torch.int32
    )
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
    return C8Case(
        query=query,
        key=key,
        weights=weights,
        query_scale=query_scale,
        key_scale=key_scale,
        actual_q=actual_q,
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


def launch(case: C8Case, pool: torch.Tensor):
    return torch.ops.nanovllm_dsa.lidu_decode_update_c8.default(
        case.query,
        case.key,
        case.weights,
        case.query_scale,
        case.key_scale,
        case.actual_q,
        case.req_entries,
        pool,
        case.cache_tokens,
        case.candidate_lens,
        case.block_table,
    )


def launch_out(
    case: C8Case,
    pool: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
):
    return torch.ops.nanovllm_dsa.lidu_decode_update_c8_out.default(
        case.query,
        case.key,
        case.weights,
        case.query_scale,
        case.key_scale,
        case.actual_q,
        case.req_entries,
        pool,
        case.cache_tokens,
        case.candidate_lens,
        case.block_table,
        source_ids,
        destination_slots,
        miss_counts,
    )


def check_case(case: C8Case) -> None:
    pool = case.initial_pool.clone()
    old_pool = pool.cpu()
    source_ids, destination_slots, miss_counts, alias = launch(case, pool)
    torch.npu.synchronize()
    if alias.data_ptr() != pool.data_ptr():
        raise AssertionError("C8 LIDU cache alias does not alias request state")
    sources = source_ids.reshape(case.query.size(0), TOPK).cpu()
    slots = destination_slots.reshape(case.query.size(0), TOPK).cpu()
    counts = miss_counts.cpu()
    updated = pool.cpu()
    native = case.native_topk.cpu().reshape(case.query.size(0), TOPK)
    active_rows = set(case.req_entries_cpu.tolist())
    for pool_row in range(updated.size(0)):
        if pool_row not in active_rows and not torch.equal(
            updated[pool_row], old_pool[pool_row]
        ):
            raise AssertionError(f"inactive request-pool row {pool_row} changed")
    for batch_row in range(case.query.size(0)):
        pool_row = int(case.req_entries_cpu[batch_row])
        budget = int(case.cache_tokens[batch_row].cpu())
        assert_pool_row(updated[pool_row], case.source_capacity, budget)
        if budget == 0:
            if (
                int(counts[batch_row]) != 0
                or bool((sources[batch_row] != -1).any())
                or bool((slots[batch_row] != -1).any())
            ):
                raise AssertionError("C=0 C8 row must be a strict no-op")
            continue
        if not torch.equal(
            torch.sort(sources[batch_row]).values,
            torch.sort(native[batch_row]).values,
        ):
            raise AssertionError(
                f"row {batch_row}: C8 LIDU top-2048 differs from official A5 C8 LI"
            )
        old_selected_slots = old_pool[pool_row].gather(
            0, sources[batch_row].long()
        )
        new_selected_slots = updated[pool_row].gather(
            0, sources[batch_row].long()
        )
        actual_miss = int(counts[batch_row])
        if (
            actual_miss != case.target_misses[batch_row]
            or actual_miss != int((old_selected_slots < 0).sum())
        ):
            raise AssertionError(
                f"row {batch_row}: miss_count={actual_miss}, "
                f"expected={case.target_misses[batch_row]}"
            )
        if actual_miss and bool((old_selected_slots[:actual_miss] >= 0).any()):
            raise AssertionError("C8 LIDU miss prefix contains a cache hit")
        if actual_miss < TOPK and bool(
            (old_selected_slots[actual_miss:] < 0).any()
        ):
            raise AssertionError("C8 LIDU hit suffix contains a cache miss")
        if not torch.equal(slots[batch_row], new_selected_slots):
            raise AssertionError("C8 LIDU published slots differ from request state")

    # Isolate the repository-local update stage from the official C8 LI launch.
    low_pool = case.initial_pool.clone()
    low_outputs = torch.ops.nanovllm_dsa.lidu_cache_update.default(
        case.native_topk.reshape(case.query.size(0), 1, TOPK),
        case.req_entries,
        low_pool,
        case.cache_tokens,
        case.candidate_lens,
    )
    torch.npu.synchronize()
    if not all(
        torch.equal(left.cpu(), right.cpu())
        for left, right in zip(low_outputs[:3], (source_ids, destination_slots, miss_counts))
    ) or not torch.equal(low_pool.cpu(), pool.cpu()):
        raise AssertionError("C8 high-level LIDU and isolated update stage disagree")

    _, second_slots, second_counts, _ = launch(case, pool)
    torch.npu.synchronize()
    if bool((second_counts.cpu() != 0).any()):
        raise AssertionError(
            f"repeated C8 update is not zero-miss: {second_counts.cpu().tolist()}"
        )
    active = case.cache_tokens.cpu() > 0
    if bool(active.any()) and bool((second_slots.cpu()[active] < 0).any()):
        raise AssertionError("repeated C8 update did not publish all top-2048 slots")

    out_pool = case.initial_pool.clone()
    out_sources = torch.full_like(source_ids, -777777)
    out_slots = torch.full_like(destination_slots, -777777)
    out_counts = torch.full_like(miss_counts, -777777)
    outputs = launch_out(
        case, out_pool, out_sources, out_slots, out_counts
    )
    torch.npu.synchronize()
    expected_ptrs = (
        out_sources.data_ptr(),
        out_slots.data_ptr(),
        out_counts.data_ptr(),
        out_pool.data_ptr(),
    )
    if tuple(tensor.data_ptr() for tensor in outputs) != expected_ptrs:
        raise AssertionError("C8 LIDU out path changed caller-owned addresses")
    if not all(
        torch.equal(left.cpu(), right.cpu())
        for left, right in zip(outputs[:3], (source_ids, destination_slots, miss_counts))
    ):
        raise AssertionError("allocating and caller-owned C8 LIDU paths disagree")

    print(
        "A5_C8_LIDU_CHECK "
        f"heads={case.query.size(1)} batch={case.query.size(0)} "
        f"source_capacity={case.source_capacity} "
        f"candidates={case.candidate_lens.cpu().tolist()} "
        f"budgets={case.cache_tokens.cpu().tolist()} misses={counts.tolist()} "
        "official_c8_li_topk=1 unordered_unique_pool_entries=1 "
        "repeat_zero_miss=1 isolated_update_match=1 out_alias=1 ok=1",
        flush=True,
    )


def event_benchmark(case: C8Case, warmup: int, iters: int) -> tuple[float, float]:
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
    lidu_us = statistics.mean(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    ) * 1000

    native_starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    native_ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(native_starts, native_ends):
        start.record()
        official_c8_lightning_indexer(
            case.query,
            case.key,
            case.weights,
            case.query_scale,
            case.key_scale,
            case.actual_q,
            case.candidate_lens,
            case.block_table,
        )
        end.record()
    native_ends[-1].synchronize()
    native_us = statistics.mean(
        start.elapsed_time(end) for start, end in zip(native_starts, native_ends)
    ) * 1000
    if not retained:
        raise AssertionError("timed C8 outputs were not retained")
    return native_us, lidu_us


def main() -> None:
    args = parse_args()
    check_args(args)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)

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
                                "A5_C8_LIDU_SKIP "
                                f"heads={heads} batch={batch} "
                                f"source_len={source_len} C={budget} "
                                f"miss_range={miss_range} reason=infeasible",
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
                            native_us, lidu_us = event_benchmark(
                                case, args.warmup, args.iters
                            )
                            print(
                                "A5_C8_LIDU_RESULT "
                                f"device_name={device_name!r} heads={heads} "
                                f"batch={batch} source_len={source_len} C={budget} "
                                f"miss_range={miss_range[0]}:{miss_range[1]} "
                                f"actual_miss_mean={statistics.mean(case.target_misses):.3f} "
                                f"official_c8_li_us={native_us:.3f} "
                                f"c8_lidu_us={lidu_us:.3f} "
                                f"update_overhead_us={lidu_us - native_us:+.3f} "
                                f"warmup={args.warmup} iters={args.iters}",
                                flush=True,
                            )
                        if args.mode == "profile":
                            pool = case.initial_pool.clone()
                            for _ in range(args.profile_replays):
                                pool.copy_(case.initial_pool)
                                launch(case, pool)
                            torch.npu.synchronize()
    print("A5_C8_LIDU_UT_OK", flush=True)


if __name__ == "__main__":
    main()
