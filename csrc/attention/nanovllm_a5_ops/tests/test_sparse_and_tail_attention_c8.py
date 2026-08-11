#!/usr/bin/env python3
"""Semantic/performance test for native A5 packed-C8 topK+tail QSFA."""

from __future__ import annotations

import argparse
import statistics

import torch

import nanovllm_dsa_a5
import torch_npu  # type: ignore  # noqa: E402,F401


BLOCK_SIZE = 128
NOPE_DIM = 512
ROPE_DIM = 64
QUERY_DIM = NOPE_DIM + ROPE_DIM
TILE_SIZE = 128
SCALE_COUNT = NOPE_DIM // TILE_SIZE
PACKED_DIM = 656
TOPK = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--mode", choices=("all", "check", "bench"), default="all")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--cache-tokens", type=int, default=6144)
    parser.add_argument("--tail-tokens", type=int, default=257)
    parser.add_argument("--max-tail-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-non-a5", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not 1 <= args.heads <= 64:
        raise ValueError("this project intentionally supports Q_HEAD <= 64")
    if args.cache_tokens != 0 and (
        args.cache_tokens < TOPK or args.cache_tokens % BLOCK_SIZE
    ):
        raise ValueError("cache tokens must be 0 or block-aligned >= 2048")
    if not 0 <= args.tail_tokens <= args.max_tail_tokens:
        raise ValueError("tail tokens must be in [0,max_tail_tokens]")
    if args.cache_tokens == 0 and args.tail_tokens == 0:
        raise ValueError("dense C=0 test requires at least one resident token")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters positive")


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


def pack_cache(
    nope: torch.Tensor,
    rope: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    packed_bytes = torch.cat(
        (
            nope.contiguous().view(torch.uint8),
            rope.contiguous().view(torch.uint8),
            scales.contiguous().view(torch.uint8),
        ),
        dim=-1,
    )
    if packed_bytes.shape[-1] != PACKED_DIM:
        raise AssertionError(f"packed row has {packed_bytes.shape[-1]} bytes")
    return packed_bytes.view(torch.float8_e4m3fn)


def make_inputs(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)
    resident_len = args.cache_tokens + args.tail_tokens
    blocks_per_row = (resident_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    physical_blocks = args.batch_size * blocks_per_row
    block_table_cpu = torch.randperm(
        physical_blocks, generator=generator, dtype=torch.int64
    ).reshape(args.batch_size, blocks_per_row).to(torch.int32)

    nope_cpu = torch.randint(
        -3,
        4,
        (physical_blocks, BLOCK_SIZE, 1, NOPE_DIM),
        generator=generator,
        dtype=torch.int16,
    ).float().to(torch.float8_e4m3fn)
    rope_cpu = torch.empty(
        (physical_blocks, BLOCK_SIZE, 1, ROPE_DIM), dtype=torch.float32
    ).uniform_(-0.5, 0.5, generator=generator).to(torch.bfloat16)
    scales_cpu = torch.empty(
        (physical_blocks, BLOCK_SIZE, 1, SCALE_COUNT), dtype=torch.float32
    ).uniform_(0.02, 0.08, generator=generator)
    packed_cpu = pack_cache(nope_cpu, rope_cpu, scales_cpu)

    q_nope_cpu = torch.empty(
        (args.batch_size, args.heads, NOPE_DIM), dtype=torch.float32
    ).uniform_(-0.5, 0.5, generator=generator).to(torch.bfloat16)
    q_rope_cpu = torch.empty(
        (args.batch_size, args.heads, ROPE_DIM), dtype=torch.float32
    ).uniform_(-0.5, 0.5, generator=generator).to(torch.bfloat16)
    query_cpu = torch.cat((q_nope_cpu, q_rope_cpu), dim=-1).contiguous()

    slots_cpu = torch.full(
        (args.batch_size, 1, TOPK + args.max_tail_tokens),
        -1,
        dtype=torch.int32,
    )
    for row in range(args.batch_size):
        if args.cache_tokens == 0:
            slots_cpu[row, 0, : args.tail_tokens] = torch.arange(
                args.tail_tokens, dtype=torch.int32
            )
        else:
            slots_cpu[row, 0, :TOPK] = torch.randperm(
                args.cache_tokens, generator=generator
            )[:TOPK].to(torch.int32)
            slots_cpu[row, 0, TOPK : TOPK + args.tail_tokens] = torch.arange(
                args.cache_tokens, resident_len, dtype=torch.int32
            )

    return {
        "device": device,
        "device_name": device_name,
        "query_cpu": query_cpu,
        "nope_cpu": nope_cpu,
        "rope_cpu": rope_cpu,
        "scales_cpu": scales_cpu,
        "packed": packed_cpu.to(device),
        "query": query_cpu.to(device),
        "slots_cpu": slots_cpu,
        "slots": slots_cpu.to(device),
        "block_table_cpu": block_table_cpu,
        "block_table": block_table_cpu.to(device),
        "actual_q": torch.arange(
            1, args.batch_size + 1, dtype=torch.int32, device=device
        ),
        "resident_lengths": torch.full(
            (args.batch_size,), resident_len, dtype=torch.int32, device=device
        ),
        "scale": QUERY_DIM**-0.5,
    }


def cpu_reference(
    inputs: dict[str, object], args: argparse.Namespace
) -> torch.Tensor:
    output = torch.empty(
        (args.batch_size, args.heads, NOPE_DIM), dtype=torch.float32
    )
    scales = inputs["scales_cpu"].repeat_interleave(TILE_SIZE, dim=-1)
    nope = inputs["nope_cpu"].float() * scales
    value = nope.to(torch.bfloat16).float()
    key = torch.cat(
        (nope.to(torch.bfloat16), inputs["rope_cpu"]), dim=-1
    ).float()
    query = inputs["query_cpu"].float()
    block_table = inputs["block_table_cpu"].to(torch.int64)
    valid_count = (
        args.tail_tokens
        if args.cache_tokens == 0
        else TOPK + args.tail_tokens
    )
    for row in range(args.batch_size):
        indices = inputs["slots_cpu"][row, 0, :valid_count].to(torch.int64)
        physical = block_table[row, indices // BLOCK_SIZE]
        offsets = indices % BLOCK_SIZE
        selected_key = key[physical, offsets, 0]
        selected_value = value[physical, offsets, 0]
        scores = query[row] @ selected_key.T * inputs["scale"]
        probabilities = torch.softmax(scores, dim=-1)
        output[row] = (
            probabilities.to(torch.bfloat16).float() @ selected_value
        )
    return output


def launch(inputs: dict[str, object]) -> torch.Tensor:
    return nanovllm_dsa_a5.sparse_and_tail_attention_c8(
        inputs["query"],
        inputs["packed"],
        inputs["slots"],
        inputs["block_table"],
        inputs["actual_q"],
        inputs["resident_lengths"],
        inputs["scale"],
    )


def check(inputs: dict[str, object], args: argparse.Namespace) -> None:
    expected = cpu_reference(inputs, args)
    actual = launch(inputs)
    torch.npu.synchronize()
    actual_cpu = actual.cpu().float()
    if not bool(torch.isfinite(actual_cpu).all()):
        raise AssertionError("C8 QSFA produced NaN or Inf")
    torch.testing.assert_close(actual_cpu, expected, atol=0.08, rtol=0.03)
    max_abs = float((actual_cpu - expected).abs().max())
    attended = (
        args.tail_tokens
        if args.cache_tokens == 0
        else TOPK + args.tail_tokens
    )
    print(
        "A5_PACKED_C8_QSFA_CHECK "
        f"batch={args.batch_size} heads={args.heads} "
        f"cache_tokens={args.cache_tokens} tail_tokens={args.tail_tokens} "
        f"attended_tokens={attended} max_abs={max_abs:.9f} "
        "finite=1 ok=1",
        flush=True,
    )


def benchmark(inputs: dict[str, object], args: argparse.Namespace) -> None:
    for _ in range(args.warmup):
        launch(inputs)
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(args.iters)]
    retained = []
    for start, end in zip(starts, ends):
        start.record()
        retained.append(launch(inputs))
        end.record()
    ends[-1].synchronize()
    avg_us = statistics.mean(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    ) * 1000
    print(
        "A5_PACKED_C8_QSFA_RESULT "
        f"batch={args.batch_size} heads={args.heads} "
        f"cache_tokens={args.cache_tokens} tail_tokens={args.tail_tokens} "
        f"avg_us={avg_us:.3f} warmup={args.warmup} iters={args.iters}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    inputs = make_inputs(args)
    print(
        "A5_PACKED_C8_QSFA_CONFIG "
        f"device={inputs['device']} device_name={inputs['device_name']!r} "
        f"batch={args.batch_size} heads={args.heads} "
        f"cache_tokens={args.cache_tokens} tail_tokens={args.tail_tokens} "
        f"max_tail_tokens={args.max_tail_tokens}",
        flush=True,
    )
    if args.mode in ("all", "check"):
        check(inputs, args)
    if args.mode in ("all", "bench"):
        benchmark(inputs, args)
    print("A5_PACKED_C8_QSFA_UT_OK", flush=True)


if __name__ == "__main__":
    main()
