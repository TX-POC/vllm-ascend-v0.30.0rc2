#!/usr/bin/env python3
"""Capture and replay the A5 LIDU -> SCATTER -> sparse+tail Attention chain."""

from __future__ import annotations

import argparse
import math

import torch

import nanovllm_dsa_a5
import torch_npu  # type: ignore  # noqa: E402,F401


BLOCK_SIZE = 128
INDEX_DIM = 128
TOPK = 2048
CKV_DIM = 512
KPE_DIM = 64
SOURCE_LEN = 4096
CACHE_BUDGET = 3072
TAIL_TOKENS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--case",
        choices=("mixed", "pure-long"),
        default="mixed",
        help="mixed uses C=0/C=3072; pure-long uses C=3072/C=3072.",
    )
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def swapped_from_cpu(cpu: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not hasattr(torch_npu, "empty_with_swapped_memory"):
        raise RuntimeError("torch_npu.empty_with_swapped_memory is required")
    tensor = torch_npu.empty_with_swapped_memory(cpu.shape, dtype=cpu.dtype, device=device)
    tensor.fill_(0)
    staging = cpu.to(device)
    tensor.add_(staging)
    torch.npu.synchronize()
    del staging
    torch.npu.empty_cache()
    return tensor


def physical_rows(
    table: torch.Tensor, row: int, logical_tokens: torch.Tensor
) -> torch.Tensor:
    logical = logical_tokens.to(torch.int64)
    return (
        table[row, logical // BLOCK_SIZE].to(torch.int64) * BLOCK_SIZE
        + logical.remainder(BLOCK_SIZE)
    )


def cpu_attention_golden(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    hbm_ckv: torch.Tensor,
    hbm_kpe: torch.Tensor,
    hbm_table_cpu: torch.Tensor,
    slots: torch.Tensor,
    cache_tokens: torch.Tensor,
    actual_kv: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    query_cpu = query.cpu().float()
    query_rope_cpu = query_rope.cpu().float()
    ckv_cpu = hbm_ckv.cpu().float().view(-1, CKV_DIM)
    kpe_cpu = hbm_kpe.cpu().float().view(-1, KPE_DIM)
    slots_cpu = slots.cpu().view(query.size(0), TOPK).to(torch.int64)
    cache_cpu = cache_tokens.cpu()
    actual_cpu = actual_kv.cpu()
    rows: list[torch.Tensor] = []
    for row in range(query.size(0)):
        c = int(cache_cpu[row])
        length = int(actual_cpu[row])
        if c == 0:
            logical = torch.arange(length, dtype=torch.int64)
        else:
            logical = torch.cat(
                (slots_cpu[row], torch.arange(c, length, dtype=torch.int64))
            )
        physical = physical_rows(hbm_table_cpu, row, logical)
        key = ckv_cpu[physical]
        key_rope = kpe_cpu[physical]
        scores = (
            query_cpu[row] @ key.T + query_rope_cpu[row] @ key_rope.T
        ) * scale
        rows.append(torch.softmax(scores, dim=-1) @ key)
    return torch.stack(rows)


def main() -> None:
    args = parse_args()
    if args.replays < 2:
        raise ValueError("--replays must be at least 2")
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    index = device.index if device.index is not None else torch.npu.current_device()
    getter = getattr(torch.npu, "get_device_name", torch_npu.npu.get_device_name)
    name = getter(index)
    if "950" not in name.lower() and not args.allow_non_a5:
        raise RuntimeError(f"expected Ascend 950, got {name!r}")

    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    batch = 2
    index_heads = 32
    attention_heads = 8
    source_blocks_per_row = SOURCE_LEN // BLOCK_SIZE
    source_blocks = batch * source_blocks_per_row

    index_table_cpu = torch.empty((batch, source_blocks_per_row), dtype=torch.int32)
    dram_table_cpu = torch.empty_like(index_table_cpu)
    for row in range(batch):
        base = row * source_blocks_per_row
        index_table_cpu[row] = base + torch.randperm(
            source_blocks_per_row, generator=generator
        ).to(torch.int32)
        dram_table_cpu[row] = base + torch.randperm(
            source_blocks_per_row, generator=generator
        ).to(torch.int32)

    index_cpu = torch.zeros(
        (source_blocks, BLOCK_SIZE, 1, INDEX_DIM), dtype=torch.bfloat16
    )
    logical_ids = torch.arange(SOURCE_LEN, dtype=torch.int64)
    for row in range(batch):
        physical = physical_rows(index_table_cpu, row, logical_ids)
        # Give the two 2048-token halves exactly representable scores.  The
        # selected set is therefore unambiguous in BF16: positive query picks
        # the high half and negative query picks the low half.  Encoding the
        # full token ID as a BF16 score would create ties around the boundary.
        index_cpu.view(-1, INDEX_DIM)[physical, 0] = torch.where(
            logical_ids < TOPK,
            torch.full_like(logical_ids, -1),
            torch.full_like(logical_ids, 1),
        ).to(torch.bfloat16)

    dram_ckv_cpu = torch.randn(
        source_blocks, BLOCK_SIZE, CKV_DIM,
        generator=generator, dtype=torch.float32,
    ).to(torch.bfloat16)
    dram_kpe_cpu = torch.randn(
        source_blocks, BLOCK_SIZE, KPE_DIM,
        generator=generator, dtype=torch.float32,
    ).to(torch.bfloat16)

    is_mixed = args.case == "mixed"
    cache_tokens_cpu = torch.tensor(
        [0 if is_mixed else CACHE_BUDGET, CACHE_BUDGET],
        dtype=torch.int32,
    )
    actual_kv_cpu = torch.tensor(
        [TOPK if is_mixed else CACHE_BUDGET + TAIL_TOKENS,
         CACHE_BUDGET + TAIL_TOKENS],
        dtype=torch.int32,
    )
    hbm_blocks_per_row = math.ceil(int(actual_kv_cpu.max()) / BLOCK_SIZE)
    hbm_blocks = batch * hbm_blocks_per_row
    hbm_table_cpu = torch.empty((batch, hbm_blocks_per_row), dtype=torch.int32)
    for row in range(batch):
        base = row * hbm_blocks_per_row
        hbm_table_cpu[row] = base + torch.randperm(
            hbm_blocks_per_row, generator=generator
        ).to(torch.int32)
    hbm_ckv_cpu = torch.randn(
        hbm_blocks, BLOCK_SIZE, CKV_DIM,
        generator=generator, dtype=torch.float32,
    ).to(torch.bfloat16)
    hbm_kpe_cpu = torch.randn(
        hbm_blocks, BLOCK_SIZE, KPE_DIM,
        generator=generator, dtype=torch.float32,
    ).to(torch.bfloat16)

    req_entries_cpu = torch.tensor([2, 0], dtype=torch.int32)
    pool_cpu = torch.full((4, SOURCE_LEN), -1, dtype=torch.int32)
    high = torch.arange(SOURCE_LEN - TOPK, SOURCE_LEN, dtype=torch.int64)
    low_victims = torch.arange(CACHE_BUDGET - TOPK, dtype=torch.int64)
    initial_sources = torch.cat((high, low_victims))
    initial_slots = torch.randperm(CACHE_BUDGET, generator=generator).to(torch.int64)
    for row, budget in enumerate(cache_tokens_cpu.tolist()):
        if budget:
            pool_cpu[int(req_entries_cpu[row]), initial_sources] = initial_slots.to(torch.int32)

    # C=0 owns a dense HBM cache. A long row owns C indexed cache slots.
    for row, budget in enumerate(cache_tokens_cpu.tolist()):
        if budget == 0:
            source_tokens = torch.arange(TOPK, dtype=torch.int64)
            destination_tokens = source_tokens
        else:
            source_tokens = initial_sources
            destination_tokens = initial_slots
        hbm_rows = physical_rows(hbm_table_cpu, row, destination_tokens)
        dram_rows = physical_rows(dram_table_cpu, row, source_tokens)
        hbm_ckv_cpu.view(-1, CKV_DIM)[hbm_rows] = dram_ckv_cpu.view(-1, CKV_DIM)[dram_rows]
        hbm_kpe_cpu.view(-1, KPE_DIM)[hbm_rows] = dram_kpe_cpu.view(-1, KPE_DIM)[dram_rows]

    index_query = torch.zeros((batch, index_heads, INDEX_DIM), dtype=torch.bfloat16, device=device)
    index_query[:, 0, 0] = 1
    weights = torch.zeros((batch, index_heads), dtype=torch.bfloat16, device=device)
    weights[:, 0] = 1
    index_key = index_cpu.to(device)
    req_entries = req_entries_cpu.to(device)
    cache_slots_pool = pool_cpu.to(device)
    cache_tokens = cache_tokens_cpu.to(device)
    candidate_lens = torch.full((batch,), SOURCE_LEN, dtype=torch.int32, device=device)
    index_table = index_table_cpu.to(device)
    dram_table = dram_table_cpu.to(device)
    hbm_table = hbm_table_cpu.to(device)
    dram_ckv = swapped_from_cpu(dram_ckv_cpu, device)
    dram_kpe = swapped_from_cpu(dram_kpe_cpu, device)
    hbm_ckv = hbm_ckv_cpu.to(device)
    hbm_kpe = hbm_kpe_cpu.to(device)
    source_ids = torch.empty((batch, 1, TOPK), dtype=torch.int32, device=device)
    destination_slots = torch.empty_like(source_ids)
    miss_counts = torch.empty((batch,), dtype=torch.int32, device=device)
    attention_query = torch.randn(
        batch, attention_heads, CKV_DIM, dtype=torch.bfloat16, device=device
    )
    attention_query_rope = torch.randn(
        batch, attention_heads, KPE_DIM, dtype=torch.bfloat16, device=device
    )
    actual_q = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    actual_kv = actual_kv_cpu.to(device)
    scale = 1.0 / math.sqrt(CKV_DIM + KPE_DIM)

    def chain():
        lidu_outputs = torch.ops.nanovllm_dsa.lidu_decode_update_out.default(
            index_query,
            index_key,
            weights,
            req_entries,
            cache_slots_pool,
            cache_tokens,
            candidate_lens,
            index_table,
            source_ids,
            destination_slots,
            miss_counts,
        )
        scatter_outputs = torch.ops.nanovllm_dsa.scatter_copy.default(
            hbm_kpe,
            hbm_ckv,
            dram_kpe,
            dram_ckv,
            hbm_table,
            dram_table,
            lidu_outputs[0].view(batch, TOPK),
            lidu_outputs[1].view(batch, TOPK),
            lidu_outputs[2],
        )
        attention_key = scatter_outputs[1].view(
            hbm_blocks, BLOCK_SIZE, 1, CKV_DIM,
        )
        attention = torch.ops.nanovllm_dsa.sparse_and_tail_attention.default(
            attention_query,
            attention_key,
            attention_key,
            lidu_outputs[1],
            cache_tokens,
            hbm_table,
            actual_q,
            actual_kv,
            attention_query_rope,
            scatter_outputs[0].view(hbm_blocks, BLOCK_SIZE, 1, KPE_DIM),
            scale,
        )
        return lidu_outputs, scatter_outputs, attention

    # Eager warmup and capture both use the initial high-token zero-miss state.
    chain()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    graph_pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=graph_pool):
        graph_lidu, graph_scatter, graph_attention = chain()
    torch.npu.synchronize()
    if graph_lidu[0].data_ptr() != source_ids.data_ptr():
        raise AssertionError("captured LIDU source buffer is not caller-owned")
    if graph_lidu[1].data_ptr() != destination_slots.data_ptr():
        raise AssertionError("captured LIDU slot buffer is not caller-owned")
    if graph_lidu[2].data_ptr() != miss_counts.data_ptr():
        raise AssertionError("captured LIDU miss-count buffer is not caller-owned")
    if graph_scatter[0].data_ptr() != hbm_kpe.data_ptr() or graph_scatter[1].data_ptr() != hbm_ckv.data_ptr():
        raise AssertionError("captured SCATTER outputs do not alias HBM cache")
    attention_address = graph_attention.data_ptr()
    if attention_address == 0:
        raise AssertionError("captured Attention output has no persistent address")
    if miss_counts.cpu().tolist() != [0, 0]:
        raise AssertionError("graph capture must be zero-miss")

    low = torch.arange(0, TOPK, dtype=torch.int64)
    expected_sets = (low, high)
    for replay in range(args.replays):
        select_low = replay % 2 == 0
        index_query.zero_()
        index_query[:, 0, 0] = -1 if select_low else 1
        torch.npu.current_stream().synchronize()
        graph.replay()
        torch.npu.synchronize()
        counts = miss_counts.cpu()
        for row, budget in enumerate(cache_tokens_cpu.tolist()):
            if budget == 0 and int(counts[row]) != 0:
                raise AssertionError(
                    f"replay {replay}: C=0 row {row} must be no-op, got {counts.tolist()}"
                )
            if budget > 0 and int(counts[row]) <= 0:
                raise AssertionError(
                    f"replay {replay}: long row {row} must have nonzero miss, got {counts.tolist()}"
                )
        expected = expected_sets[0 if select_low else 1]
        for row, budget in enumerate(cache_tokens_cpu.tolist()):
            if budget == 0:
                if bool((source_ids[row].cpu() != -1).any()) or bool((destination_slots[row].cpu() != -1).any()):
                    raise AssertionError(f"replay {replay}: C=0 row {row} published copy metadata")
                continue
            actual_sources = source_ids[row].cpu().view(-1).to(torch.int64)
            if not torch.equal(torch.sort(actual_sources).values, expected):
                raise AssertionError(f"replay {replay}: row {row} LIDU top-2048 set is wrong")
            pool_row = int(req_entries_cpu[row])
            state = cache_slots_pool[pool_row].cpu()
            slots = state[expected].to(torch.int64)
            if bool((slots < 0).any()) or torch.unique(slots).numel() != TOPK:
                raise AssertionError(f"replay {replay}: row {row} cache does not cover top-2048")
            actual_hbm = hbm_ckv.view(-1, CKV_DIM)[
                physical_rows(hbm_table_cpu, row, slots).to(device)
            ].cpu()
            expected_dram = dram_ckv_cpu.view(-1, CKV_DIM)[
                physical_rows(dram_table_cpu, row, expected)
            ]
            if not torch.equal(actual_hbm, expected_dram):
                raise AssertionError(f"replay {replay}: row {row} CKV copy is wrong")
            actual_hbm_rope = hbm_kpe.view(-1, KPE_DIM)[
                physical_rows(hbm_table_cpu, row, slots).to(device)
            ].cpu()
            expected_dram_rope = dram_kpe_cpu.view(-1, KPE_DIM)[
                physical_rows(dram_table_cpu, row, expected)
            ]
            if not torch.equal(actual_hbm_rope, expected_dram_rope):
                raise AssertionError(f"replay {replay}: row {row} KPE copy is wrong")
        golden = cpu_attention_golden(
            attention_query,
            attention_query_rope,
            hbm_ckv,
            hbm_kpe,
            hbm_table_cpu,
            destination_slots,
            cache_tokens,
            actual_kv,
            scale,
        )
        actual_attention = graph_attention.cpu().float()
        if graph_attention.data_ptr() != attention_address:
            raise AssertionError("Attention replay changed the captured output address")
        if not torch.isfinite(actual_attention).all():
            raise AssertionError(f"replay {replay}: Attention produced NaN/Inf")
        torch.testing.assert_close(actual_attention, golden, rtol=0.02, atol=0.02)

    print(
        "A5_OFFLOAD_SPLIT_GRAPH_CHECK "
        f"device={device} case={args.case} replays={args.replays} capture_zero_miss=1 "
        "replay_nonzero_miss=1 caller_owned_outputs=1 "
        "dram_to_hbm=1 attention_golden=1 ok=1",
        flush=True,
    )
    print("A5_OFFLOAD_SPLIT_GRAPH_UT_OK", flush=True)


if __name__ == "__main__":
    main()
