#!/usr/bin/env python3
"""Benchmark subgraph-C BatchGet Prepare and KVAF BatchGet independently.

One-time KVAF initialization, BatchPut population, AccessTable materialization,
input construction, and correctness checks are excluded from timed regions.
"""

from __future__ import annotations

import argparse
import importlib
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch_npu  # type: ignore  # noqa: F401

import kvaf


BLOCK_SIZE = 128
PACKED_TOKEN_BYTES = 656
ENTRY_BYTES = BLOCK_SIZE * PACKED_TOKEN_BYTES
TOPK = 2048
DESTINATION_BLOCKS_PER_ROW = TOPK // BLOCK_SIZE
SOURCE_FILL = 0x5A


def load_dsa_registration() -> str:
    """Load the Prepare registration from the normal or standalone build."""

    failures: list[str] = []
    for module_name, backend in (
        ("vllm_ascend.dsa_offload.a5_extension", "vllm_ascend_C"),
        ("nanovllm_dsa_a5", "nanovllm_dsa_a5._C"),
    ):
        try:
            importlib.import_module(module_name)
            getattr(torch.ops.nanovllm_dsa, "kvaf_batch_get_prepare_out")
        except Exception as error:
            failures.append(f"{backend}: {type(error).__name__}: {error}")
            continue
        return backend
    raise RuntimeError(
        "Unable to load kvaf_batch_get_prepare_out. For the normal repository "
        "build, reinstall vLLM-Ascend from this source tree with custom kernels "
        "enabled. For standalone operator development, run "
        "build_nanovllm_ops.sh before adding torch_extension to PYTHONPATH. "
        f"Load failures: {' | '.join(failures)}"
    )


@dataclass(frozen=True)
class PrepareInputs:
    source_token_ids: torch.Tensor
    destination_slots: torch.Tensor
    miss_count: torch.Tensor
    resident_pool_indices: torch.Tensor
    resident_block_table: torch.Tensor
    cache_tokens: torch.Tensor
    candidate_lens: torch.Tensor
    logical_seq_lengths: torch.Tensor


@dataclass(frozen=True)
class PrepareOutputs:
    indices: torch.Tensor
    offsets: torch.Tensor
    mask: torch.Tensor
    output_indices: torch.Tensor
    resident_seq_lengths: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--qcap", type=int, default=64)
    parser.add_argument("--miss-count", type=int, default=256)
    parser.add_argument(
        "--source-layout",
        choices=("random", "sequential"),
        default="random",
        help="LIDU-like source-token layout; random is the representative default",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--put-batch-size", type=int, default=128)
    parser.add_argument("--npu-cores", type=int, default=32)
    parser.add_argument("--jettys-per-core", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--batch-get-kernel",
        type=Path,
        required=True,
        help="Built kvaf_execute_batch_get_kernels.o",
    )
    parser.add_argument(
        "--batch-put-kernel",
        type=Path,
        default=None,
        help="Optional kvaf_execute_batch_put_kernels.o; empty uses KVAF's mock Put backend",
    )
    parser.add_argument(
        "--max-ddr-gib",
        type=float,
        default=4.0,
        help="Safety limit for the benchmark's KVAF DDR allocation",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.num_layers <= 0 or args.qcap <= 0:
        raise ValueError("batch-size, num-layers, and qcap must be positive")
    if not 0 <= args.layer_id < args.num_layers:
        raise ValueError("layer-id must be inside [0,num-layers)")
    if not 0 <= args.miss_count <= TOPK:
        raise ValueError(f"miss-count must be inside [0,{TOPK}]")
    if args.qcap * BLOCK_SIZE < TOPK:
        raise ValueError(
            f"qcap must cover at least {TOPK} distinct source tokens "
            f"({TOPK // BLOCK_SIZE} blocks)"
        )
    if args.warmup < 0 or args.iters <= 0 or args.put_batch_size <= 0:
        raise ValueError("warmup must be non-negative; iters/put-batch-size positive")
    if args.npu_cores <= 0 or not 1 <= args.jettys_per_core <= 2:
        raise ValueError(
            "npu-cores must be positive and jettys-per-core must be 1 or 2"
        )
    if args.max_ddr_gib <= 0:
        raise ValueError("max-ddr-gib must be positive")
    if not args.batch_get_kernel.is_file():
        raise FileNotFoundError(args.batch_get_kernel)
    if args.batch_put_kernel is not None and not args.batch_put_kernel.is_file():
        raise FileNotFoundError(args.batch_put_kernel)
    if not hasattr(kvaf, "NpuRuntime"):
        raise RuntimeError("KVAF was not built with KVAF_ENABLE_AIV_BATCH_GET=ON")


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def report(
    operation: str,
    samples_us: list[float],
    *,
    warmup: int,
    active_tokens: int,
    moves_payload: bool = False,
) -> None:
    mean_us = statistics.mean(samples_us)
    payload_gibps = (
        active_tokens * PACKED_TOKEN_BYTES / (1024.0**3) / (mean_us / 1_000_000.0)
        if mean_us > 0
        else 0.0
    )
    payload_metric = f" payload_gibps={payload_gibps:.3f}" if moves_payload else ""
    print(
        "PERF_RESULT "
        f"operation={operation} samples={len(samples_us)} "
        f"warmup_excluded={warmup} active_tokens={active_tokens} "
        f"mean_us={mean_us:.3f} "
        f"p50_us={percentile(samples_us, 0.50):.3f} "
        f"p90_us={percentile(samples_us, 0.90):.3f} "
        f"p99_us={percentile(samples_us, 0.99):.3f}"
        f"{payload_metric}",
        flush=True,
    )


def make_prepare_case(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[PrepareInputs, PrepareOutputs, torch.Tensor]:
    source_capacity = args.qcap * BLOCK_SIZE
    if args.source_layout == "random":
        source_token_ids = torch.stack(
            [
                torch.randperm(source_capacity, dtype=torch.int32)[:TOPK]
                for _ in range(args.batch_size)
            ]
        ).unsqueeze(1)
        destination_slots = torch.stack(
            [torch.randperm(TOPK, dtype=torch.int32) for _ in range(args.batch_size)]
        ).unsqueeze(1)
    else:
        row = torch.arange(TOPK, dtype=torch.int32).view(1, 1, TOPK)
        source_token_ids = row.repeat(args.batch_size, 1, 1)
        destination_slots = row.repeat(args.batch_size, 1, 1)
    miss_count = torch.full((args.batch_size,), args.miss_count, dtype=torch.int32)
    resident_pool_indices = torch.arange(args.batch_size, dtype=torch.int32)
    physical_blocks = args.batch_size * DESTINATION_BLOCKS_PER_ROW
    resident_block_table = torch.arange(physical_blocks, dtype=torch.int32).view(
        args.batch_size, DESTINATION_BLOCKS_PER_ROW
    )
    candidate_len = source_capacity
    cache_tokens = torch.full((args.batch_size,), TOPK, dtype=torch.int32)
    candidate_lens = torch.full((args.batch_size,), candidate_len, dtype=torch.int32)
    logical_seq_lengths = candidate_lens + 1
    request_count = args.batch_size * TOPK
    inputs = PrepareInputs(
        source_token_ids=source_token_ids.to(device),
        destination_slots=destination_slots.to(device),
        miss_count=miss_count.to(device),
        resident_pool_indices=resident_pool_indices.to(device),
        resident_block_table=resident_block_table.to(device),
        cache_tokens=cache_tokens.to(device),
        candidate_lens=candidate_lens.to(device),
        logical_seq_lengths=logical_seq_lengths.to(device),
    )
    outputs = PrepareOutputs(
        indices=torch.empty(request_count, dtype=torch.int32, device=device),
        offsets=torch.empty(request_count, dtype=torch.int32, device=device),
        mask=torch.empty(request_count, dtype=torch.uint8, device=device),
        output_indices=torch.empty(request_count, dtype=torch.int32, device=device),
        resident_seq_lengths=torch.empty(
            args.batch_size, dtype=torch.int32, device=device
        ),
    )
    packed_output = torch.zeros(
        (physical_blocks * BLOCK_SIZE, PACKED_TOKEN_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    return inputs, outputs, packed_output


def launch_prepare(
    args: argparse.Namespace,
    inputs: PrepareInputs,
    outputs: PrepareOutputs,
) -> None:
    torch.ops.nanovllm_dsa.kvaf_batch_get_prepare_out(
        inputs.source_token_ids,
        inputs.destination_slots,
        inputs.miss_count,
        inputs.resident_pool_indices,
        inputs.resident_block_table,
        inputs.cache_tokens,
        inputs.candidate_lens,
        inputs.logical_seq_lengths,
        outputs.indices,
        outputs.offsets,
        outputs.mask,
        outputs.output_indices,
        outputs.resident_seq_lengths,
        args.layer_id,
        args.num_layers,
        args.batch_size,
        args.qcap,
    )


def benchmark_npu_event(
    launch: Callable[[], None],
    *,
    warmup: int,
    iters: int,
) -> list[float]:
    for _ in range(warmup):
        launch()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        launch()
        end.record()
    ends[-1].synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def benchmark_host_to_stream_completion(
    launch: Callable[[], None],
    synchronize: Callable[[], None],
    *,
    warmup: int,
    iters: int,
) -> tuple[list[float], list[float], list[float]]:
    for _ in range(warmup):
        launch()
    synchronize()
    submit_us: list[float] = []
    wait_us: list[float] = []
    total_us: list[float] = []
    for _ in range(iters):
        begin = time.perf_counter()
        launch()
        submitted = time.perf_counter()
        synchronize()
        completed = time.perf_counter()
        submit_us.append((submitted - begin) * 1_000_000.0)
        wait_us.append((completed - submitted) * 1_000_000.0)
        total_us.append((completed - begin) * 1_000_000.0)
    return submit_us, wait_us, total_us


def populate_kvaf(
    args: argparse.Namespace,
    runtime: object,
    device: torch.device,
    table_capacity: int,
) -> tuple[object, object, list[int]]:
    keys = list(range(1, table_capacity + 1))
    packed_input = torch.full(
        (args.put_batch_size, ENTRY_BYTES),
        SOURCE_FILL,
        dtype=torch.uint8,
        device=device,
    )
    for start in range(0, table_capacity, args.put_batch_size):
        key_batch = keys[start : start + args.put_batch_size]
        batch = len(key_batch)
        offsets = [row * ENTRY_BYTES for row in range(batch)]
        lengths = [ENTRY_BYTES] * batch
        runtime.batch_put(
            key_batch,
            packed_input[:batch].reshape(-1),
            offsets,
            lengths,
        )
    torch.npu.synchronize()
    handle = runtime.prepare_access_table(
        torch.arange(1, table_capacity + 1, dtype=torch.int64)
    )
    device_table = runtime.access_table_view(
        handle,
        device="npu",
        capacity=table_capacity,
    )
    torch.npu.synchronize()
    return handle, device_table, keys


def validate_prepare(
    args: argparse.Namespace,
    inputs: PrepareInputs,
    outputs: PrepareOutputs,
) -> None:
    torch.npu.synchronize()
    source = inputs.source_token_ids.cpu().view(args.batch_size, TOPK)
    destination = inputs.destination_slots.cpu().view(args.batch_size, TOPK)
    resident_block_table = inputs.resident_block_table.cpu()
    indices = outputs.indices.cpu().view(args.batch_size, TOPK)
    offsets = outputs.offsets.cpu().view(args.batch_size, TOPK)
    mask = outputs.mask.cpu().view(args.batch_size, TOPK)
    output_indices = outputs.output_indices.cpu().view(args.batch_size, TOPK)
    expected_mask = torch.zeros_like(mask)
    expected_mask[:, : args.miss_count] = 1
    torch.testing.assert_close(mask, expected_mask)
    expected_resident_lengths = torch.full(
        (args.batch_size,), TOPK + 1, dtype=torch.int32
    )
    torch.testing.assert_close(
        outputs.resident_seq_lengths.cpu(), expected_resident_lengths
    )
    for row in range(args.batch_size):
        active_source = source[row, : args.miss_count]
        active_destination = destination[row, : args.miss_count]
        expected_indices = (
            row * args.num_layers + args.layer_id
        ) * args.qcap + active_source // BLOCK_SIZE
        expected_offsets = active_source.remainder(BLOCK_SIZE) * PACKED_TOKEN_BYTES
        destination_blocks = active_destination // BLOCK_SIZE
        expected_output_indices = resident_block_table[row].index_select(
            0, destination_blocks.to(torch.int64)
        ) * BLOCK_SIZE + active_destination.remainder(BLOCK_SIZE)
        torch.testing.assert_close(indices[row, : args.miss_count], expected_indices)
        torch.testing.assert_close(offsets[row, : args.miss_count], expected_offsets)
        torch.testing.assert_close(
            output_indices[row, : args.miss_count], expected_output_indices
        )
        torch.testing.assert_close(
            indices[row, args.miss_count :],
            torch.zeros_like(indices[row, args.miss_count :]),
        )
        torch.testing.assert_close(
            offsets[row, args.miss_count :],
            torch.zeros_like(offsets[row, args.miss_count :]),
        )
        torch.testing.assert_close(
            output_indices[row, args.miss_count :],
            torch.zeros_like(output_indices[row, args.miss_count :]),
        )


def validate_batch_get(
    args: argparse.Namespace,
    outputs: PrepareOutputs,
    packed_output: torch.Tensor,
) -> None:
    if args.miss_count == 0:
        return
    active = outputs.mask.bool()
    destination_rows = outputs.output_indices[active].to(torch.int64)
    actual = packed_output.index_select(0, destination_rows)
    torch.npu.synchronize()
    if not bool(torch.all(actual == SOURCE_FILL).cpu()):
        raise AssertionError("BatchGet output payload does not match BatchPut data")


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU, for example npu:0")
    torch.npu.set_device(device)
    dsa_registration_backend = load_dsa_registration()
    table_capacity = args.batch_size * args.num_layers * args.qcap
    ddr_bytes = table_capacity * ENTRY_BYTES
    ddr_gib = ddr_bytes / (1024.0**3)
    if ddr_gib > args.max_ddr_gib:
        raise ValueError(
            f"requested DDR {ddr_gib:.3f} GiB exceeds --max-ddr-gib "
            f"{args.max_ddr_gib:.3f}"
        )

    device_id = 0 if device.index is None else int(device.index)
    batch_put_kernel = (
        "" if args.batch_put_kernel is None else str(args.batch_put_kernel)
    )
    runtime = kvaf.NpuRuntime(
        str(args.batch_get_kernel),
        device_id,
        args.npu_cores,
        batch_put_kernel,
        args.jettys_per_core,
    )
    inputs, outputs, packed_output = make_prepare_case(args, device)
    handle = None
    device_table = None
    keys: list[int] = []
    stream = torch.npu.current_stream()
    active_tokens = args.batch_size * args.miss_count
    try:
        print(
            "KVAF_SUBGRAPH_C_BENCHMARK_CONFIG "
            f"device={device} batch={args.batch_size} "
            f"num_layers={args.num_layers} layer_id={args.layer_id} "
            f"qcap={args.qcap} table_capacity={table_capacity} "
            f"ddr_gib={ddr_gib:.3f} miss_count={args.miss_count} "
            f"source_layout={args.source_layout} "
            f"active_tokens={active_tokens} warmup={args.warmup} "
            f"iters={args.iters} npu_cores={args.npu_cores} "
            f"jettys_per_core={args.jettys_per_core} "
            f"dsa_registration_backend={dsa_registration_backend}",
            flush=True,
        )
        runtime.initialize(ddr_bytes, ENTRY_BYTES)
        handle, device_table, keys = populate_kvaf(
            args, runtime, device, table_capacity
        )

        prepare = lambda: launch_prepare(args, inputs, outputs)
        prepare()
        validate_prepare(args, inputs, outputs)

        def batch_get() -> None:
            runtime.batch_get(
                device_table,
                outputs.indices,
                outputs.offsets,
                outputs.mask,
                outputs.output_indices,
                packed_output,
                stream,
                False,
            )

        batch_get()
        torch.npu.synchronize()
        validate_batch_get(args, outputs, packed_output)
        print("KVAF_SUBGRAPH_C_BENCHMARK_CORRECTNESS ok=1", flush=True)

        prepare_samples = benchmark_npu_event(
            prepare, warmup=args.warmup, iters=args.iters
        )
        report(
            "batch_get_prepare_npu_event",
            prepare_samples,
            warmup=args.warmup,
            active_tokens=active_tokens,
        )

        (
            prepare_submit,
            prepare_wait,
            prepare_total,
        ) = benchmark_host_to_stream_completion(
            prepare,
            stream.synchronize,
            warmup=args.warmup,
            iters=args.iters,
        )
        report(
            "batch_get_prepare_submit",
            prepare_submit,
            warmup=args.warmup,
            active_tokens=active_tokens,
        )
        report(
            "batch_get_prepare_stream_wait",
            prepare_wait,
            warmup=args.warmup,
            active_tokens=active_tokens,
        )
        report(
            "batch_get_prepare_call_to_stream_completion",
            prepare_total,
            warmup=args.warmup,
            active_tokens=active_tokens,
        )

        batch_get_event_samples = benchmark_npu_event(
            batch_get, warmup=args.warmup, iters=args.iters
        )
        report(
            "batch_get_npu_event",
            batch_get_event_samples,
            warmup=args.warmup,
            active_tokens=active_tokens,
            moves_payload=True,
        )

        submit, wait, total = benchmark_host_to_stream_completion(
            batch_get,
            stream.synchronize,
            warmup=args.warmup,
            iters=args.iters,
        )
        report(
            "batch_get_submit",
            submit,
            warmup=args.warmup,
            active_tokens=active_tokens,
        )
        report(
            "batch_get_stream_wait",
            wait,
            warmup=args.warmup,
            active_tokens=active_tokens,
        )
        report(
            "batch_get_call_to_stream_completion",
            total,
            warmup=args.warmup,
            active_tokens=active_tokens,
            moves_payload=True,
        )

        def prepare_and_batch_get() -> None:
            prepare()
            batch_get()

        _, _, combined = benchmark_host_to_stream_completion(
            prepare_and_batch_get,
            stream.synchronize,
            warmup=args.warmup,
            iters=args.iters,
        )
        report(
            "batch_get_prepare_plus_batch_get",
            combined,
            warmup=args.warmup,
            active_tokens=active_tokens,
            moves_payload=True,
        )
    finally:
        try:
            if handle is not None:
                runtime.release_access_table(handle)
            for start in range(0, len(keys), args.put_batch_size):
                runtime.batch_free(keys[start : start + args.put_batch_size])
        finally:
            runtime.shutdown()


if __name__ == "__main__":
    main()
