#!/usr/bin/env python3
"""Capture/replay A5 LIDU -> packed-C8 SCATTER -> native C8 QSFA."""

from __future__ import annotations

import argparse
import math

import torch

import nanovllm_dsa_a5
import torch_npu  # type: ignore  # noqa: E402,F401
from test_lidu_c8 import make_case


BLOCK_SIZE = 128
NOPE_DIM = 512
ROPE_DIM = 64
PACKED_DIM = 656
TOPK = 2048
OUTPUT_SENTINEL = -123456789


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--case", choices=("pure-long", "mixed"), default="pure-long")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--index-heads", type=int, choices=(32, 64), default=32)
    parser.add_argument("--source-len", type=int, default=4096)
    parser.add_argument("--cache-tokens", type=int, default=3072)
    parser.add_argument("--tail-tokens", type=int, default=64)
    parser.add_argument("--max-tail-tokens", type=int, default=256)
    parser.add_argument("--miss-min", type=int, default=256)
    parser.add_argument("--miss-max", type=int, default=512)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.case == "mixed" and args.batch_size < 2:
        raise ValueError("mixed case requires batch-size >= 2")
    if not 1 <= args.heads <= 64:
        raise ValueError("C8 QSFA supports Q_HEAD <= 64")
    if args.source_len < args.cache_tokens or args.source_len % BLOCK_SIZE:
        raise ValueError("source length must be block aligned and >= C")
    if args.cache_tokens < TOPK or args.cache_tokens % BLOCK_SIZE:
        raise ValueError("C must be block aligned and >= 2048")
    if not 0 <= args.tail_tokens <= args.max_tail_tokens:
        raise ValueError("tail tokens must be in [0,max_tail_tokens]")
    if not 0 <= args.miss_min <= args.miss_max <= TOPK:
        raise ValueError("miss range must be within [0,2048]")
    if args.replays < 2:
        raise ValueError("at least two graph replays are required")


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


def private_block_table(
    batch: int,
    blocks_per_row: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    physical_blocks = batch * blocks_per_row
    table = torch.empty((batch, blocks_per_row), dtype=torch.int32)
    for row in range(batch):
        first = row * blocks_per_row
        table[row] = first + torch.randperm(
            blocks_per_row, generator=generator, dtype=torch.int64
        ).to(torch.int32)
    return table, physical_blocks


def physical_rows(
    table: torch.Tensor,
    row: int,
    logical_tokens: torch.Tensor,
) -> torch.Tensor:
    logical = logical_tokens.to(torch.int64)
    return (
        table[row, logical // BLOCK_SIZE].to(torch.int64) * BLOCK_SIZE
        + logical.remainder(BLOCK_SIZE)
    )


def make_packed_bytes(
    blocks: int,
    generator: torch.Generator,
) -> torch.Tensor:
    nope = torch.randint(
        -3,
        4,
        (blocks, BLOCK_SIZE, 1, NOPE_DIM),
        generator=generator,
        dtype=torch.int16,
    ).float().to(torch.float8_e4m3fn)
    rope = torch.empty(
        (blocks, BLOCK_SIZE, 1, ROPE_DIM), dtype=torch.float32
    ).uniform_(-0.5, 0.5, generator=generator).to(torch.bfloat16)
    scales = torch.empty(
        (blocks, BLOCK_SIZE, 1, NOPE_DIM // BLOCK_SIZE),
        dtype=torch.float32,
    ).uniform_(0.02, 0.08, generator=generator)
    packed = torch.cat(
        (
            nope.contiguous().view(torch.uint8),
            rope.contiguous().view(torch.uint8),
            scales.contiguous().view(torch.uint8),
        ),
        dim=-1,
    )
    if packed.shape[-1] != PACKED_DIM:
        raise AssertionError(f"packed row has {packed.shape[-1]} bytes")
    return packed.view(torch.int8).contiguous()


def swapped_from_cpu(cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not hasattr(torch_npu, "empty_with_swapped_memory"):
        raise RuntimeError("torch_npu.empty_with_swapped_memory is required")
    swapped = torch_npu.empty_with_swapped_memory(
        cpu.shape, dtype=cpu.dtype, device=device
    )
    swapped.zero_()
    staging = cpu.to(device)
    swapped.add_(staging)
    torch.npu.synchronize()
    del staging
    torch.npu.empty_cache()
    return swapped


def initialize_hbm(
    *,
    dram_bytes: torch.Tensor,
    dram_table: torch.Tensor,
    hbm_table: torch.Tensor,
    pool: torch.Tensor,
    req_entries: torch.Tensor,
    cache_tokens: list[int],
    candidate_lens: list[int],
    actual_lens: list[int],
) -> torch.Tensor:
    total_blocks = int(hbm_table.max()) + 1
    hbm = torch.full(
        (total_blocks, BLOCK_SIZE, 1, PACKED_DIM),
        -91,
        dtype=torch.int8,
    )
    dram_rows = dram_bytes.view(-1, PACKED_DIM)
    hbm_rows = hbm.view(-1, PACKED_DIM)
    pool_cpu = pool.cpu()
    req_cpu = req_entries.cpu()
    for row in range(req_cpu.numel()):
        budget = cache_tokens[row]
        candidate_len = candidate_lens[row]
        actual_len = actual_lens[row]
        if budget == 0:
            source_tokens = torch.arange(actual_len, dtype=torch.int64)
            hbm_rows[
                physical_rows(hbm_table, row, source_tokens)
            ] = dram_rows[physical_rows(dram_table, row, source_tokens)]
            continue
        state = pool_cpu[int(req_cpu[row]), :candidate_len]
        source_tokens = (state >= 0).nonzero().flatten().to(torch.int64)
        destination_slots = state[source_tokens].to(torch.int64)
        hbm_rows[physical_rows(hbm_table, row, destination_slots)] = (
            dram_rows[physical_rows(dram_table, row, source_tokens)]
        )
        tail_tokens = actual_len - candidate_len
        if tail_tokens:
            tail_sources = torch.arange(
                candidate_len, actual_len, dtype=torch.int64
            )
            tail_slots = torch.arange(
                budget,
                budget + tail_tokens,
                dtype=torch.int64,
            )
            hbm_rows[physical_rows(hbm_table, row, tail_slots)] = dram_rows[
                physical_rows(dram_table, row, tail_sources)
            ]
    return hbm


def assert_scatter_bytes(
    *,
    hbm: torch.Tensor,
    dram_cpu: torch.Tensor,
    hbm_table: torch.Tensor,
    dram_table: torch.Tensor,
    source_ids: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> None:
    sources = source_ids.cpu().view(source_ids.size(0), TOPK)
    destinations = destination_slots.cpu().view(source_ids.size(0), TOPK)
    counts = miss_counts.cpu()
    hbm_rows = hbm.view(torch.int8).view(-1, PACKED_DIM)
    dram_rows = dram_cpu.view(-1, PACKED_DIM)
    for row, count_value in enumerate(counts.tolist()):
        count = int(count_value)
        source_physical = physical_rows(
            dram_table, row, sources[row, :count]
        )
        destination_physical = physical_rows(
            hbm_table, row, destinations[row, :count]
        )
        actual = hbm_rows[destination_physical.to(hbm.device)].cpu()
        expected = dram_rows[source_physical]
        if not torch.equal(actual, expected):
            raise AssertionError(
                f"row {row}: graph SCATTER did not copy packed DRAM bytes"
            )


def assert_attention_metadata(
    *,
    attention_slots: torch.Tensor,
    resident_lengths: torch.Tensor,
    destination_slots: torch.Tensor,
    cache_tokens: torch.Tensor,
    candidate_lens: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
) -> None:
    slots = attention_slots.cpu()
    destinations = destination_slots.cpu().view(destination_slots.size(0), TOPK)
    budgets = cache_tokens.cpu()
    candidates = candidate_lens.cpu()
    actual_lengths = actual_seq_lengths_kv.cpu()
    expected_lengths = torch.empty_like(actual_lengths)
    for row in range(actual_lengths.numel()):
        budget = int(budgets[row])
        candidate = int(candidates[row])
        actual = int(actual_lengths[row])
        if budget == 0:
            active = torch.arange(actual, dtype=torch.int32)
            resident = actual
        else:
            tail = actual - candidate
            active = torch.cat(
                (
                    destinations[row],
                    torch.arange(budget, budget + tail, dtype=torch.int32),
                )
            )
            resident = budget + tail
        expected_lengths[row] = resident
        if not torch.equal(slots[row, 0, : active.numel()], active):
            raise AssertionError(
                f"row {row}: topK+tail attention metadata differs"
            )
        if bool((slots[row, 0, active.numel() :] != -1).any()):
            raise AssertionError(
                f"row {row}: attention metadata padding must be -1"
            )
    actual_resident = resident_lengths.cpu()
    if not torch.equal(actual_resident, expected_lengths):
        raise AssertionError(
            "resident lengths differ from the independently computed values: "
            f"actual={actual_resident.tolist()} "
            f"expected={expected_lengths.tolist()}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)
    generator = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)

    if args.case == "mixed":
        budgets = [0] + [args.cache_tokens] * (args.batch_size - 1)
        candidate_lens_cpu = [TOPK] + [args.source_len] * (args.batch_size - 1)
    else:
        budgets = [args.cache_tokens] * args.batch_size
        candidate_lens_cpu = [args.source_len] * args.batch_size
    actual_lens_cpu = [
        length + args.tail_tokens for length in candidate_lens_cpu
    ]
    lidu_case = make_case(
        device=device,
        batch=args.batch_size,
        source_len=args.source_len,
        heads=args.index_heads,
        budgets=budgets,
        miss_range=(args.miss_min, args.miss_max),
        pool_extra=3,
        seed=args.seed,
        candidate_lens_cpu=candidate_lens_cpu,
    )
    packed_source_tokens = args.source_len + args.tail_tokens
    dram_blocks_per_row = math.ceil(packed_source_tokens / BLOCK_SIZE)
    hbm_tokens_per_row = max(
        max(budgets) + args.max_tail_tokens,
        max(
            actual
            for budget, actual in zip(budgets, actual_lens_cpu)
            if budget == 0
        )
        if 0 in budgets
        else 0,
    )
    hbm_blocks_per_row = math.ceil(hbm_tokens_per_row / BLOCK_SIZE)
    dram_table_cpu, dram_blocks = private_block_table(
        args.batch_size, dram_blocks_per_row, generator
    )
    hbm_table_cpu, _ = private_block_table(
        args.batch_size, hbm_blocks_per_row, generator
    )
    dram_cpu = make_packed_bytes(dram_blocks, generator)
    initial_hbm_cpu = initialize_hbm(
        dram_bytes=dram_cpu,
        dram_table=dram_table_cpu,
        hbm_table=hbm_table_cpu,
        pool=lidu_case.initial_pool,
        req_entries=lidu_case.req_entries,
        cache_tokens=budgets,
        candidate_lens=candidate_lens_cpu,
        actual_lens=actual_lens_cpu,
    )
    dram = swapped_from_cpu(dram_cpu, device)
    initial_hbm = initial_hbm_cpu.view(torch.float8_e4m3fn).to(device)
    dram_table = dram_table_cpu.to(device)
    hbm_table = hbm_table_cpu.to(device)
    actual_kv = torch.tensor(
        actual_lens_cpu, dtype=torch.int32, device=device
    )
    actual_q = torch.arange(
        1, args.batch_size + 1, dtype=torch.int32, device=device
    )
    attention_query = torch.empty(
        (args.batch_size, args.heads, NOPE_DIM + ROPE_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-0.5, 0.5)
    scale = (NOPE_DIM + ROPE_DIM) ** -0.5

    def make_state(
        *,
        hbm_seed: torch.Tensor | None = None,
        pool_seed: torch.Tensor | None = None,
    ):
        hbm = initial_hbm.clone() if hbm_seed is None else hbm_seed.clone()
        pool = (
            lidu_case.initial_pool.clone()
            if pool_seed is None
            else pool_seed.clone()
        )
        source_ids = torch.full(
            (args.batch_size, 1, TOPK),
            OUTPUT_SENTINEL,
            dtype=torch.int32,
            device=device,
        )
        destination_slots = torch.full_like(source_ids, OUTPUT_SENTINEL)
        miss_counts = torch.full(
            (args.batch_size,),
            OUTPUT_SENTINEL,
            dtype=torch.int32,
            device=device,
        )
        attention_slots = torch.full(
            (args.batch_size, 1, TOPK + args.max_tail_tokens),
            OUTPUT_SENTINEL,
            dtype=torch.int32,
            device=device,
        )
        resident_lengths = torch.full_like(miss_counts, OUTPUT_SENTINEL)
        return (
            hbm,
            pool,
            source_ids,
            destination_slots,
            miss_counts,
            attention_slots,
            resident_lengths,
        )

    def chain(state):
        (
            hbm,
            pool,
            source_ids,
            destination_slots,
            miss_counts,
            attention_slots,
            resident_lengths,
        ) = state
        lidu = torch.ops.nanovllm_dsa.lidu_decode_update_c8_out.default(
            lidu_case.query,
            lidu_case.key,
            lidu_case.weights,
            lidu_case.query_scale,
            lidu_case.key_scale,
            lidu_case.actual_q,
            lidu_case.req_entries,
            pool,
            lidu_case.cache_tokens,
            lidu_case.candidate_lens,
            lidu_case.block_table,
            source_ids,
            destination_slots,
            miss_counts,
        )
        scatter = torch.ops.nanovllm_dsa.packed_scatter_copy_out.default(
            hbm.view(torch.int8),
            dram,
            hbm_table,
            dram_table,
            lidu[0],
            lidu[1],
            lidu[2],
            lidu_case.cache_tokens,
            lidu_case.candidate_lens,
            actual_kv,
            args.max_tail_tokens,
            attention_slots,
            resident_lengths,
        )
        attention = nanovllm_dsa_a5.sparse_and_tail_attention_c8(
            attention_query,
            hbm,
            scatter[1],
            hbm_table,
            actual_q,
            scatter[2],
            scale,
        )
        return lidu, scatter, attention

    eager_state = make_state()
    eager_lidu, eager_scatter, eager_attention = chain(eager_state)
    torch.npu.synchronize()
    expected_sources = eager_lidu[0].cpu()
    expected_destinations = eager_lidu[1].cpu()
    expected_counts = eager_lidu[2].cpu()
    expected_slots = eager_scatter[1].cpu()
    expected_lengths = eager_scatter[2].cpu()
    expected_attention = eager_attention.cpu()
    long_rows = lidu_case.cache_tokens.cpu() > 0
    if bool((expected_counts[long_rows] <= 0).any()):
        raise AssertionError(
            "graph case requires nonzero misses on offloaded rows, "
            f"got {expected_counts.tolist()}"
        )
    assert_scatter_bytes(
        hbm=eager_state[0],
        dram_cpu=dram_cpu,
        hbm_table=hbm_table_cpu,
        dram_table=dram_table_cpu,
        source_ids=eager_lidu[0],
        destination_slots=eager_lidu[1],
        miss_counts=eager_lidu[2],
    )
    assert_attention_metadata(
        attention_slots=eager_scatter[1],
        resident_lengths=eager_scatter[2],
        destination_slots=eager_lidu[1],
        cache_tokens=lidu_case.cache_tokens,
        candidate_lens=lidu_case.candidate_lens,
        actual_seq_lengths_kv=actual_kv,
    )
    if not bool(torch.isfinite(eager_attention).all()):
        raise AssertionError("eager C8 QSFA output contains NaN or Inf")

    graph_state = make_state(
        hbm_seed=eager_state[0], pool_seed=eager_state[1]
    )
    graph = torch.npu.NPUGraph()
    graph_pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=graph_pool):
        graph_lidu, graph_scatter, graph_attention = chain(graph_state)
    torch.npu.synchronize()

    caller_ptrs = tuple(tensor.data_ptr() for tensor in graph_state[2:7])
    captured_ptrs = (
        graph_lidu[0].data_ptr(),
        graph_lidu[1].data_ptr(),
        graph_lidu[2].data_ptr(),
        graph_scatter[1].data_ptr(),
        graph_scatter[2].data_ptr(),
    )
    if caller_ptrs != captured_ptrs:
        raise AssertionError("captured outputs are not caller-owned")
    attention_ptr = graph_attention.data_ptr()

    # Graph capture records device work; capture-time output contents are not
    # a correctness signal. Poison every caller-owned output and prove that
    # the first replay writes the complete zero-miss result instead.
    for tensor in graph_state[2:7]:
        tensor.fill_(OUTPUT_SENTINEL)
    torch.npu.current_stream().synchronize()
    graph.replay()
    torch.npu.synchronize()
    zero_miss_counts = graph_lidu[2].cpu()
    if bool((zero_miss_counts != 0).any()):
        raise AssertionError(
            "first captured replay did not publish the zero-miss result: "
            f"actual={zero_miss_counts.tolist()}"
        )
    assert_attention_metadata(
        attention_slots=graph_scatter[1],
        resident_lengths=graph_scatter[2],
        destination_slots=graph_lidu[1],
        cache_tokens=lidu_case.cache_tokens,
        candidate_lens=lidu_case.candidate_lens,
        actual_seq_lengths_kv=actual_kv,
    )
    if not bool(torch.isfinite(graph_attention).all()):
        raise AssertionError(
            "first captured C8 QSFA replay contains NaN or Inf"
        )

    for replay in range(args.replays):
        graph_state[0].copy_(initial_hbm)
        graph_state[1].copy_(lidu_case.initial_pool)
        for tensor in graph_state[2:7]:
            tensor.fill_(OUTPUT_SENTINEL)
        torch.npu.current_stream().synchronize()
        graph.replay()
        torch.npu.synchronize()
        if not torch.equal(graph_lidu[0].cpu(), expected_sources):
            raise AssertionError(f"replay {replay}: source IDs changed")
        if not torch.equal(graph_lidu[1].cpu(), expected_destinations):
            raise AssertionError(f"replay {replay}: destination slots changed")
        if not torch.equal(graph_lidu[2].cpu(), expected_counts):
            raise AssertionError(f"replay {replay}: miss counts changed")
        if not torch.equal(graph_scatter[1].cpu(), expected_slots):
            raise AssertionError(f"replay {replay}: topK+tail metadata changed")
        if not torch.equal(graph_scatter[2].cpu(), expected_lengths):
            raise AssertionError(f"replay {replay}: resident lengths changed")
        assert_attention_metadata(
            attention_slots=graph_scatter[1],
            resident_lengths=graph_scatter[2],
            destination_slots=graph_lidu[1],
            cache_tokens=lidu_case.cache_tokens,
            candidate_lens=lidu_case.candidate_lens,
            actual_seq_lengths_kv=actual_kv,
        )
        assert_scatter_bytes(
            hbm=graph_state[0],
            dram_cpu=dram_cpu,
            hbm_table=hbm_table_cpu,
            dram_table=dram_table_cpu,
            source_ids=graph_lidu[0],
            destination_slots=graph_lidu[1],
            miss_counts=graph_lidu[2],
        )
        torch.testing.assert_close(
            graph_attention.cpu(), expected_attention, atol=0.001, rtol=0.001
        )
        if graph_attention.data_ptr() != attention_ptr:
            raise AssertionError("QSFA replay changed output address")

    print(
        "A5_OFFLOAD_SPLIT_C8_GRAPH_CHECK "
        f"device={device} device_name={device_name!r} "
        f"case={args.case} "
        f"batch={args.batch_size} index_heads={args.index_heads} "
        f"attention_heads={args.heads} source_len={args.source_len} "
        f"cache_tokens={args.cache_tokens} tail_tokens={args.tail_tokens} "
        f"max_tail_tokens={args.max_tail_tokens} "
        f"misses={expected_counts.tolist()} replays={args.replays} "
        "zero_miss_replay=1 replay_nonzero_miss=1 "
        "official_c8_indexer=1 caller_owned_outputs=1 "
        "dram_to_hbm=1 native_c8_qsfa=1 "
        "stable_addresses=1 ok=1",
        flush=True,
    )
    print("A5_OFFLOAD_SPLIT_C8_GRAPH_UT_OK", flush=True)


if __name__ == "__main__":
    main()
