#!/usr/bin/env python3
"""Measure the eager overlap envelope of official C8 LI and remote KVAF Get.

This is deliberately not a Stage-5 prefetch implementation.  It answers one
narrow prerequisite question without running vLLM end-to-end: after a valid
BatchGet descriptor already exists, can a real A5 C8 Lightning Indexer scan
and a real A5->K5 KVAF BatchGet make progress on different NPU streams?

The descriptor source is intentionally honest.  Before timing, the script
runs the official C8 Lightning Indexer over the same B=32 / 128K-shaped input
and uses a fixed prefix of its real Top-2048 IDs as known exact misses.  It
does not invent scores, estimate a threshold on the CPU, or claim to exercise
scan-time candidate production.  A later device operator must provide that
producer if this overlap envelope is favourable.

All one-time work (KVAF link/init, DDR population, access-table creation,
official Top-K prelude, descriptor lowering, and correctness checks) is
outside the timed regions.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch_npu  # type: ignore  # noqa: F401

import kvaf


BLOCK_SIZE = 128
HEAD_DIM = 128
PACKED_TOKEN_BYTES = 656
ENTRY_BYTES = BLOCK_SIZE * PACKED_TOKEN_BYTES
TOPK = 2048
DESTINATION_BLOCKS_PER_ROW = TOPK // BLOCK_SIZE
SOURCE_FILL = 0x5A
INDICES_POISON = -101
OFFSETS_POISON = -102
OUTPUT_INDICES_POISON = -103
MASK_POISON = 0xA5
MAX_SOURCE_CAPACITY = 1 << 18


@dataclass(frozen=True)
class C8Inputs:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    query_scale: torch.Tensor
    key_scale: torch.Tensor
    actual_q: torch.Tensor
    candidate_lens: torch.Tensor
    block_table: torch.Tensor


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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--source-len", type=int, default=131072)
    parser.add_argument("--heads", type=int, choices=(32, 64), default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--qcap", type=int, default=1024)
    parser.add_argument("--miss-count", type=int, default=102)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--put-batch-size", type=int, default=128)
    parser.add_argument("--npu-cores", type=int, default=32)
    parser.add_argument("--jettys-per-core", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--batch-get-kernel",
        type=Path,
        required=True,
        help="Built kvaf_execute_batch_get_kernels.o from KVAF 6abe076.",
    )
    parser.add_argument(
        "--batch-put-kernel",
        type=Path,
        required=True,
        help="Built kvaf_execute_batch_put_kernels.o from KVAF 6abe076.",
    )
    parser.add_argument(
        "--cpu-service-addr",
        default=os.getenv(
            "VLLM_ASCEND_KVAF_CPU_SERVICE_ADDR",
            os.getenv("KVAF_CPU_SERVICE_ADDR", ""),
        ),
        help="K5 KVAF CPU service, for example 141.61.92.77:28999.",
    )
    parser.add_argument(
        "--max-ddr-gib",
        type=float,
        default=4.0,
        help="Safety limit for the one-time remote DDR allocation.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.source_len < TOPK or args.source_len % BLOCK_SIZE:
        raise ValueError("--source-len must be block aligned and at least 2048")
    if args.source_len > MAX_SOURCE_CAPACITY:
        raise ValueError("--source-len exceeds the 18-bit C8 token-ID capacity")
    if args.qcap <= 0 or args.qcap * BLOCK_SIZE != args.source_len:
        raise ValueError("--qcap * 128 must exactly equal --source-len")
    if args.num_layers <= 0 or not 0 <= args.layer_id < args.num_layers:
        raise ValueError("--layer-id must be inside [0,--num-layers)")
    if not 1 <= args.miss_count <= TOPK:
        raise ValueError("--miss-count must be inside [1,2048]")
    if args.warmup < 0 or args.iters <= 0 or args.put_batch_size <= 0:
        raise ValueError("--warmup must be non-negative; --iters and --put-batch-size positive")
    if args.npu_cores <= 0 or not 1 <= args.jettys_per_core <= 2:
        raise ValueError("--npu-cores must be positive and --jettys-per-core must be 1 or 2")
    if args.max_ddr_gib <= 0:
        raise ValueError("--max-ddr-gib must be positive")
    if not args.cpu_service_addr:
        raise ValueError("--cpu-service-addr is required for the remote K5 data plane")
    if not args.batch_get_kernel.is_file():
        raise FileNotFoundError(args.batch_get_kernel)
    if not args.batch_put_kernel.is_file():
        raise FileNotFoundError(args.batch_put_kernel)
    if not hasattr(kvaf, "NpuRuntime"):
        raise RuntimeError("KVAF was not built with the NPU Runtime binding")


def load_prepare_registration() -> str:
    """Load the existing device-side BatchGet descriptor-lowering operator."""

    failures: list[str] = []
    for module_name, backend in (
        ("vllm_ascend.a5_dsa_registration", "vllm_ascend_C"),
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
        "Unable to load kvaf_batch_get_prepare_out. The normal source build "
        "must have A5 custom kernels enabled, or the previously built "
        "standalone nanovllm_dsa_a5 extension must be on PYTHONPATH. "
        f"Load failures: {' | '.join(failures)}"
    )


def require_a5(device: torch.device) -> str:
    index = device.index if device.index is not None else torch.npu.current_device()
    getter = getattr(torch.npu, "get_device_name", torch_npu.npu.get_device_name)
    name = getter(index)
    if "950" not in name.lower():
        raise RuntimeError(f"expected Ascend 950, got {name!r}")
    return name


def normalized_hadamard_128(device: torch.device) -> torch.Tensor:
    """Match the normalized 128x128 Hadamard used by the C8 LI path."""

    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.size(0) < HEAD_DIM:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return (matrix / math.sqrt(HEAD_DIM)).to(
        dtype=torch.bfloat16, device=device
    )


def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quantized, scale = torch_npu.npu_dynamic_quant(
        tensor, dst_type=torch.float8_e4m3fn
    )
    return (
        quantized.contiguous(),
        scale.view(tensor.shape[:-1]).to(torch.float32).contiguous(),
    )


def make_disjoint_c8_inputs(
    args: argparse.Namespace,
    device: torch.device,
) -> C8Inputs:
    """Allocate B physically disjoint, shuffled 128K C8 key regions."""

    blocks_per_row = args.source_len // BLOCK_SIZE
    total_blocks = args.batch_size * blocks_per_row
    cpu_generator = torch.Generator().manual_seed(args.seed)
    block_table_cpu = torch.stack(
        [
            torch.randperm(
                blocks_per_row, generator=cpu_generator, dtype=torch.int64
            ).to(torch.int32)
            + row * blocks_per_row
            for row in range(args.batch_size)
        ]
    ).contiguous()

    # The order is the same as the existing C8 test: BF16 construction,
    # normalized Hadamard, then E4M3 dynamic quantization with FP32 scales.
    query_fp = torch.empty(
        (args.batch_size, args.heads, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-1, 1)
    key_fp = torch.empty(
        (total_blocks, BLOCK_SIZE, 1, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-1, 1)
    hadamard = normalized_hadamard_128(device)
    query_fp = torch.matmul(query_fp, hadamard)
    key_fp = torch.matmul(key_fp, hadamard)
    query, query_scale = quantize_fp8(query_fp)
    key, key_scale = quantize_fp8(key_fp)
    weights = torch.empty(
        (args.batch_size, args.heads), dtype=torch.bfloat16, device=device
    ).uniform_(0.01, 1.0).contiguous()
    return C8Inputs(
        query=query,
        key=key,
        weights=weights,
        query_scale=query_scale,
        key_scale=key_scale,
        actual_q=torch.arange(
            1, args.batch_size + 1, dtype=torch.int32, device=device
        ),
        candidate_lens=torch.full(
            (args.batch_size,), args.source_len, dtype=torch.int32, device=device
        ),
        block_table=block_table_cpu.to(device),
    )


def official_c8_lightning_indexer(inputs: C8Inputs) -> torch.Tensor:
    """Run the exact official C8 Lightning Indexer ABI used by vLLM-Ascend."""

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
    result = op(
        query=inputs.query,
        key=inputs.key,
        weights=inputs.weights,
        query_dequant_scale=inputs.query_scale,
        key_dequant_scale=inputs.key_scale,
        actual_seq_lengths_query=inputs.actual_q,
        actual_seq_lengths_key=inputs.candidate_lens,
        block_table=inputs.block_table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    topk = result[0] if isinstance(result, tuple) else result
    expected_count = inputs.query.size(0) * TOPK
    if not isinstance(topk, torch.Tensor) or topk.dtype != torch.int32:
        raise RuntimeError("official C8 LightningIndexer returned no int32 TopK")
    if topk.numel() != expected_count:
        raise RuntimeError(
            "official C8 LightningIndexer returned unexpected TopK shape: "
            f"{tuple(topk.shape)}, expected elements={expected_count}"
        )
    return topk.reshape(inputs.query.size(0), TOPK).contiguous()


def validate_official_topk(topk: torch.Tensor, source_len: int) -> None:
    """Check that the real official result is usable as token-ID descriptors."""

    torch.npu.synchronize()
    host = topk.cpu()
    if bool((host < 0).any()) or bool((host >= source_len).any()):
        raise AssertionError("official C8 TopK contains a token outside source_len")
    for row in range(host.size(0)):
        if int(torch.unique(host[row]).numel()) != TOPK:
            raise AssertionError(f"official C8 TopK contains duplicates for row={row}")


def make_prepare_case(
    args: argparse.Namespace,
    device: torch.device,
    exact_topk: torch.Tensor,
) -> tuple[PrepareInputs, PrepareOutputs, torch.Tensor]:
    """Lower a fixed prefix of real Top-K IDs into the existing Get ABI."""

    if tuple(exact_topk.shape) != (args.batch_size, TOPK):
        raise ValueError("exact_topk shape does not match the benchmark batch")
    source_token_ids = exact_topk.unsqueeze(1).contiguous()
    destination_slots = torch.arange(
        TOPK, dtype=torch.int32, device=device
    ).view(1, 1, TOPK).expand(args.batch_size, 1, TOPK).clone()
    miss_count = torch.full(
        (args.batch_size,), args.miss_count, dtype=torch.int32, device=device
    )
    resident_pool_indices = torch.arange(
        args.batch_size, dtype=torch.int32, device=device
    )
    physical_blocks = args.batch_size * DESTINATION_BLOCKS_PER_ROW
    resident_block_table = torch.arange(
        physical_blocks, dtype=torch.int32, device=device
    ).view(args.batch_size, DESTINATION_BLOCKS_PER_ROW)
    cache_tokens = torch.full(
        (args.batch_size,), TOPK, dtype=torch.int32, device=device
    )
    candidate_lens = torch.full(
        (args.batch_size,), args.source_len, dtype=torch.int32, device=device
    )
    logical_seq_lengths = candidate_lens + 1
    request_count = args.batch_size * TOPK
    outputs = PrepareOutputs(
        indices=torch.full(
            (request_count,), INDICES_POISON, dtype=torch.int32, device=device
        ),
        offsets=torch.full(
            (request_count,), OFFSETS_POISON, dtype=torch.int32, device=device
        ),
        mask=torch.full(
            (request_count,), MASK_POISON, dtype=torch.uint8, device=device
        ),
        output_indices=torch.full(
            (request_count,),
            OUTPUT_INDICES_POISON,
            dtype=torch.int32,
            device=device,
        ),
        resident_seq_lengths=torch.empty(
            args.batch_size, dtype=torch.int32, device=device
        ),
    )
    packed_output = torch.zeros(
        (physical_blocks * BLOCK_SIZE, PACKED_TOKEN_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    return (
        PrepareInputs(
            source_token_ids=source_token_ids,
            destination_slots=destination_slots,
            miss_count=miss_count,
            resident_pool_indices=resident_pool_indices,
            resident_block_table=resident_block_table,
            cache_tokens=cache_tokens,
            candidate_lens=candidate_lens,
            logical_seq_lengths=logical_seq_lengths,
        ),
        outputs,
        packed_output,
    )


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
    torch.testing.assert_close(
        outputs.resident_seq_lengths.cpu(),
        torch.full((args.batch_size,), TOPK + 1, dtype=torch.int32),
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
            torch.full_like(indices[row, args.miss_count :], INDICES_POISON),
        )
        torch.testing.assert_close(
            offsets[row, args.miss_count :],
            torch.full_like(offsets[row, args.miss_count :], OFFSETS_POISON),
        )
        torch.testing.assert_close(
            output_indices[row, args.miss_count :],
            torch.full_like(
                output_indices[row, args.miss_count :], OUTPUT_INDICES_POISON
            ),
        )


def create_remote_runtime(
    args: argparse.Namespace,
    device_id: int,
) -> object:
    """Construct only the KVAF 6abe remote-runtime ABI; do not fall back."""

    try:
        runtime = kvaf.NpuRuntime(
            str(args.batch_get_kernel),
            PACKED_TOKEN_BYTES,
            BLOCK_SIZE,
            device_id,
            args.npu_cores,
            str(args.batch_put_kernel),
            args.jettys_per_core,
        )
    except TypeError as error:
        raise RuntimeError(
            "This benchmark requires KVAF 6abe076 remote NpuRuntime: "
            "NpuRuntime(get_kernel, 656, 128, ..., put_kernel, ...)"
        ) from error
    link_to = getattr(runtime, "link_to", None)
    if not callable(link_to):
        raise RuntimeError(
            "This benchmark requires KVAF 6abe076 NpuRuntime.link_to(); "
            "do not run it against the older local/mock Runtime."
        )
    link_to(args.cpu_service_addr)
    return runtime


def populate_kvaf(
    args: argparse.Namespace,
    runtime: object,
    device: torch.device,
    table_capacity: int,
) -> tuple[object, torch.Tensor, torch.Tensor, list[int]]:
    """Fill all source blocks over the real K5 link, then materialize its table."""

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
        runtime.batch_put(
            key_batch,
            packed_input[:batch].reshape(-1),
            [row * ENTRY_BYTES for row in range(batch)],
            [ENTRY_BYTES] * batch,
        )
    torch.npu.synchronize()
    handle = runtime.prepare_access_table(
        torch.arange(1, table_capacity + 1, dtype=torch.int64)
    )
    device_table = runtime.access_table_view(
        handle, device="npu", capacity=table_capacity
    )
    addresses = getattr(device_table, "addresses", None)
    valid = getattr(device_table, "valid", None)
    if not isinstance(addresses, torch.Tensor) or not isinstance(valid, torch.Tensor):
        raise RuntimeError(
            "KVAF 6abe access_table_view(device='npu') did not return "
            "a DeviceAccessTable with addresses and valid tensors"
        )
    if (
        addresses.device != device
        or valid.device != device
        or addresses.dtype != torch.int64
        or valid.dtype != torch.uint8
        or addresses.numel() != table_capacity
        or valid.numel() != table_capacity
    ):
        raise AssertionError("KVAF DeviceAccessTable layout is not the expected NPU ABI")
    torch.npu.synchronize()
    return handle, addresses, valid, keys


def validate_batch_get(
    outputs: PrepareOutputs,
    packed_output: torch.Tensor,
) -> None:
    active = outputs.mask.bool()
    destination_rows = outputs.output_indices[active].to(torch.int64)
    actual = packed_output.index_select(0, destination_rows)
    torch.npu.synchronize()
    if not bool(torch.all(actual == SOURCE_FILL).cpu()):
        raise AssertionError("remote BatchGet output payload does not match BatchPut data")


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
    moves_payload: bool,
) -> None:
    mean_us = statistics.mean(samples_us)
    payload = ""
    if moves_payload:
        gibps = (
            active_tokens
            * PACKED_TOKEN_BYTES
            / (1024.0**3)
            / (mean_us / 1_000_000.0)
            if mean_us > 0
            else 0.0
        )
        payload = f" payload_gibps={gibps:.3f}"
    print(
        "PERF_RESULT "
        f"operation={operation} samples={len(samples_us)} "
        f"warmup_excluded={warmup} active_tokens={active_tokens} "
        f"mean_us={mean_us:.3f} "
        f"p50_us={percentile(samples_us, 0.50):.3f} "
        f"p90_us={percentile(samples_us, 0.90):.3f} "
        f"p99_us={percentile(samples_us, 0.99):.3f}"
        f"{payload}",
        flush=True,
    )


def warmup(launch: Callable[[], object], count: int) -> None:
    retained: list[object] = []
    for _ in range(count):
        retained.append(launch())
    torch.npu.synchronize()


def time_li_only(
    launch_li: Callable[[], torch.Tensor],
    main_stream: object,
    *,
    warmup_count: int,
    iters: int,
) -> list[float]:
    warmup(launch_li, warmup_count)
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    retained: list[torch.Tensor] = []
    for start, end in zip(starts, ends):
        start.record(main_stream)
        retained.append(launch_li())
        end.record(main_stream)
    ends[-1].synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def time_get_only(
    launch_get: Callable[[object], None],
    main_stream: object,
    copy_stream: object,
    *,
    warmup_count: int,
    iters: int,
) -> list[float]:
    def launch_warmup() -> tuple[object, object]:
        gate = torch.npu.Event()
        done = torch.npu.Event()
        gate.record(main_stream)
        with torch.npu.stream(copy_stream):
            copy_stream.wait_event(gate)
            launch_get(copy_stream)
            done.record(copy_stream)
        main_stream.wait_event(done)
        return gate, done

    warmup(launch_warmup, warmup_count)
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    gates = [torch.npu.Event() for _ in range(iters)]
    dones = [torch.npu.Event() for _ in range(iters)]
    for start, end, gate, done in zip(starts, ends, gates, dones):
        start.record(main_stream)
        gate.record(main_stream)
        with torch.npu.stream(copy_stream):
            copy_stream.wait_event(gate)
            launch_get(copy_stream)
            done.record(copy_stream)
        main_stream.wait_event(done)
        end.record(main_stream)
    ends[-1].synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def time_serial(
    launch_li: Callable[[], torch.Tensor],
    launch_get: Callable[[object], None],
    main_stream: object,
    copy_stream: object,
    *,
    warmup_count: int,
    iters: int,
) -> list[float]:
    def launch_warmup() -> tuple[torch.Tensor, object, object]:
        li_done = torch.npu.Event()
        get_done = torch.npu.Event()
        result = launch_li()
        li_done.record(main_stream)
        with torch.npu.stream(copy_stream):
            copy_stream.wait_event(li_done)
            launch_get(copy_stream)
            get_done.record(copy_stream)
        main_stream.wait_event(get_done)
        return result, li_done, get_done

    warmup(launch_warmup, warmup_count)
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    li_dones = [torch.npu.Event() for _ in range(iters)]
    get_dones = [torch.npu.Event() for _ in range(iters)]
    retained: list[torch.Tensor] = []
    for start, end, li_done, get_done in zip(starts, ends, li_dones, get_dones):
        start.record(main_stream)
        retained.append(launch_li())
        li_done.record(main_stream)
        with torch.npu.stream(copy_stream):
            copy_stream.wait_event(li_done)
            launch_get(copy_stream)
            get_done.record(copy_stream)
        main_stream.wait_event(get_done)
        end.record(main_stream)
    ends[-1].synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def time_concurrent_join(
    launch_li: Callable[[], torch.Tensor],
    launch_get: Callable[[object], None],
    main_stream: object,
    copy_stream: object,
    *,
    warmup_count: int,
    iters: int,
) -> list[float]:
    def launch_warmup() -> tuple[torch.Tensor, object, object]:
        gate = torch.npu.Event()
        get_done = torch.npu.Event()
        gate.record(main_stream)
        with torch.npu.stream(copy_stream):
            copy_stream.wait_event(gate)
            launch_get(copy_stream)
            get_done.record(copy_stream)
        result = launch_li()
        main_stream.wait_event(get_done)
        return result, gate, get_done

    warmup(launch_warmup, warmup_count)
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    gates = [torch.npu.Event() for _ in range(iters)]
    get_dones = [torch.npu.Event() for _ in range(iters)]
    retained: list[torch.Tensor] = []
    for start, end, gate, get_done in zip(starts, ends, gates, get_dones):
        # The gate makes the timing interval include both streams without
        # creating a LI -> Get dependency.  It precedes only a marker.
        start.record(main_stream)
        gate.record(main_stream)
        with torch.npu.stream(copy_stream):
            copy_stream.wait_event(gate)
            launch_get(copy_stream)
            get_done.record(copy_stream)
        retained.append(launch_li())
        main_stream.wait_event(get_done)
        end.record(main_stream)
    ends[-1].synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("--device must select an NPU, for example npu:0")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    device_name = require_a5(device)
    dsa_registration_backend = load_prepare_registration()

    table_capacity = args.batch_size * args.num_layers * args.qcap
    ddr_bytes = table_capacity * ENTRY_BYTES
    ddr_gib = ddr_bytes / (1024.0**3)
    if ddr_gib > args.max_ddr_gib:
        raise ValueError(
            f"requested DDR {ddr_gib:.3f} GiB exceeds --max-ddr-gib "
            f"{args.max_ddr_gib:.3f}"
        )
    device_id = 0 if device.index is None else int(device.index)
    active_tokens = args.batch_size * args.miss_count
    c8_inputs = make_disjoint_c8_inputs(args, device)
    runtime = create_remote_runtime(args, device_id)
    handle = None
    keys: list[int] = []
    try:
        print(
            "C8_KVAF_OVERLAP_CONFIG "
            f"device={device} device_name={device_name!r} "
            f"batch={args.batch_size} source_len={args.source_len} "
            f"heads={args.heads} qcap={args.qcap} "
            "key_layout=disjoint_physical_blocks_per_row_shuffled "
            f"miss_count={args.miss_count} active_tokens={active_tokens} "
            "descriptor_source=official_c8_topk_known_miss_prefix "
            f"cpu_service_addr={args.cpu_service_addr} "
            f"ddr_gib={ddr_gib:.3f} warmup={args.warmup} iters={args.iters} "
            f"npu_cores={args.npu_cores} jettys_per_core={args.jettys_per_core} "
            f"dsa_registration_backend={dsa_registration_backend}",
            flush=True,
        )

        # This is deliberately outside timing: it makes a descriptor from
        # genuine C8 Top-K IDs, then gives the overlap experiment stable input.
        exact_topk = official_c8_lightning_indexer(c8_inputs)
        validate_official_topk(exact_topk, args.source_len)
        inputs, outputs, packed_output = make_prepare_case(args, device, exact_topk)
        launch_prepare(args, inputs, outputs)
        validate_prepare(args, inputs, outputs)

        runtime.initialize(ddr_bytes, ENTRY_BYTES)
        handle, addresses, valid, keys = populate_kvaf(
            args, runtime, device, table_capacity
        )
        main_stream = torch.npu.current_stream()
        copy_stream = torch.npu.Stream()

        def launch_get(stream: object) -> None:
            runtime.batch_get(
                addresses,
                valid,
                outputs.indices,
                outputs.offsets,
                outputs.mask,
                outputs.output_indices,
                packed_output,
                stream,
                False,
            )

        launch_get(main_stream)
        main_stream.synchronize()
        validate_batch_get(outputs, packed_output)
        print(
            "C8_KVAF_OVERLAP_CORRECTNESS "
            "official_c8_topk=1 prepare=1 remote_payload=1",
            flush=True,
        )

        launch_li = lambda: official_c8_lightning_indexer(c8_inputs)
        li_samples = time_li_only(
            launch_li,
            main_stream,
            warmup_count=args.warmup,
            iters=args.iters,
        )
        get_samples = time_get_only(
            launch_get,
            main_stream,
            copy_stream,
            warmup_count=args.warmup,
            iters=args.iters,
        )
        serial_samples = time_serial(
            launch_li,
            launch_get,
            main_stream,
            copy_stream,
            warmup_count=args.warmup,
            iters=args.iters,
        )
        concurrent_samples = time_concurrent_join(
            launch_li,
            launch_get,
            main_stream,
            copy_stream,
            warmup_count=args.warmup,
            iters=args.iters,
        )

        report(
            "c8_li_only",
            li_samples,
            warmup=args.warmup,
            active_tokens=active_tokens,
            moves_payload=False,
        )
        report(
            "remote_get_only",
            get_samples,
            warmup=args.warmup,
            active_tokens=active_tokens,
            moves_payload=True,
        )
        report(
            "c8_li_then_get_serial",
            serial_samples,
            warmup=args.warmup,
            active_tokens=active_tokens,
            moves_payload=True,
        )
        report(
            "c8_li_with_remote_get_concurrent_join",
            concurrent_samples,
            warmup=args.warmup,
            active_tokens=active_tokens,
            moves_payload=True,
        )
        serial_p90 = percentile(serial_samples, 0.90)
        concurrent_p90 = percentile(concurrent_samples, 0.90)
        per_stream_p90_reference = max(
            percentile(li_samples, 0.90), percentile(get_samples, 0.90)
        )
        print(
            "C8_KVAF_OVERLAP_RESULT "
            f"serial_p90_us={serial_p90:.3f} "
            f"concurrent_p90_us={concurrent_p90:.3f} "
            f"per_stream_p90_reference_us={per_stream_p90_reference:.3f} "
            f"overlap_saved_p90_us={serial_p90 - concurrent_p90:.3f} "
            "concurrent_over_per_stream_reference_p90_us="
            f"{concurrent_p90 - per_stream_p90_reference:.3f}",
            flush=True,
        )
        print("C8_KVAF_OVERLAP_PASS", flush=True)
    finally:
        if handle is not None:
            try:
                runtime.release_access_table(handle)
            except Exception as error:
                print(
                    f"C8_KVAF_OVERLAP_CLEANUP release_access_table_error={error}",
                    flush=True,
                )
        if keys:
            try:
                for start in range(0, len(keys), args.put_batch_size):
                    runtime.batch_free(keys[start : start + args.put_batch_size])
            except Exception as error:
                print(
                    f"C8_KVAF_OVERLAP_CLEANUP batch_free_error={error}",
                    flush=True,
                )
        try:
            runtime.shutdown()
        except Exception as error:
            print(
                f"C8_KVAF_OVERLAP_CLEANUP shutdown_error={error}", flush=True
            )


if __name__ == "__main__":
    main()
