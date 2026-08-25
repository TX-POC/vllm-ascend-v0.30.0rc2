#!/usr/bin/env python3
"""Measure the eager overlap envelope of official C8 LI and KVAF Get.

This is deliberately not a Stage-5 prefetch implementation.  It answers one
narrow prerequisite question without running vLLM end-to-end: after a valid
BatchGet descriptor already exists, can a real A5 C8 Lightning Indexer scan
and a real KVAF BatchGet make progress on different NPU streams?

The descriptor source is intentionally honest.  Before timing, the script
runs the official C8 Lightning Indexer over the same B=32 / 128K-shaped input
and uses a fixed prefix of its real Top-2048 IDs as known exact misses.  It
does not invent scores, estimate a threshold on the CPU, or claim to exercise
scan-time candidate production.  A later device operator must provide that
producer if this overlap envelope is favourable.

All one-time work (KVAF link/init, DDR population, access-table creation,
official Top-K prelude, descriptor lowering, and correctness checks) is
outside the timed regions.

``--enable-stream-resource-quota`` is a separate eager capability probe.  It
uses CANN's public per-stream Cube/Vector *count* limits.  The probe gives LI
and Get their own explicit streams, binds the LI quota once per measured
phase while LI's host tiler runs, and leaves KVAF Get as its native explicit
two-AIV-block launch.  This avoids putting resource-TLS driver work in each
timed submit path.  It
does not claim fixed physical core IDs; that requires a later device-side
core-ID diagnostic.

The selected KVAF data plane is explicit. ``--transport local`` uses A5 HBM
and the same A5 host DDR through the dsa-direct-get Direct-HCCP backend;
``--transport remote`` uses the existing A5<->K5 CPU-service backend. The
benchmark never retries the other backend after a failure.

This is a prebuilt-descriptor overlap upper bound: Get deliberately does not
depend on the LI output in the timed region. It therefore cannot establish a
full Stage-5 producer/consumer pipeline, but it can rule out an overlap plan
whose best-case dependency graph already fails to overlap.
"""

from __future__ import annotations

import argparse
import ctypes
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


@dataclass(frozen=True)
class StreamResourceState:
    """CANN's configured or thread-bound resource view for one stream."""

    label: str
    stream_handle: int
    cube_cores: int
    vector_cores: int


@dataclass(frozen=True)
class ConcurrentTimingSamples:
    """Per-stream spans and their joint completion for one overlap run.

    The spans make the source of a bad join observable: LI may slow down,
    Get may slow down, or both can contend for shared resources.  All values
    are NPU-event durations in microseconds.
    """

    join_us: list[float]
    li_span_us: list[float]
    get_span_us: list[float]


class AclStreamResourceApi:
    """Small benchmark-only binding for CANN's public resource APIs."""

    _CUBE_CORE = 0
    _VECTOR_CORE = 1

    def __init__(self) -> None:
        self.library_name = "libascendcl.so"
        try:
            self._library = ctypes.CDLL(self.library_name)
        except OSError as error:
            raise RuntimeError(
                "C8_KVAF_RESOURCE_QUOTA_UNAVAILABLE "
                f"cannot_load={self.library_name} error={error}"
            ) from error
        self._set_stream_limit = self._require_symbol(
            "aclrtSetStreamResLimit",
            (ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32),
        )
        self._get_stream_limit = self._require_symbol(
            "aclrtGetStreamResLimit",
            (ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)),
        )
        self._reset_stream_limit = self._require_symbol(
            "aclrtResetStreamResLimit",
            (ctypes.c_void_p,),
        )
        self._use_stream_in_current_thread = self._require_symbol(
            "aclrtUseStreamResInCurrentThread",
            (ctypes.c_void_p,),
        )
        self._unuse_stream_in_current_thread = self._require_symbol(
            "aclrtUnuseStreamResInCurrentThread",
            (ctypes.c_void_p,),
        )
        self._get_current_thread_limit = self._require_symbol(
            "aclrtGetResInCurrentThread",
            (ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)),
        )

    def _require_symbol(
        self,
        name: str,
        argtypes: tuple[object, ...],
    ) -> object:
        try:
            function = getattr(self._library, name)
        except AttributeError as error:
            raise RuntimeError(
                "C8_KVAF_RESOURCE_QUOTA_UNAVAILABLE "
                f"missing_symbol={name} library={self.library_name}"
            ) from error
        function.argtypes = list(argtypes)
        function.restype = ctypes.c_int
        return function

    @staticmethod
    def _stream_handle(stream: object) -> tuple[ctypes.c_void_p, int]:
        raw_handle = getattr(stream, "npu_stream", None)
        if raw_handle is None:
            raise RuntimeError(
                "C8_KVAF_RESOURCE_QUOTA_UNAVAILABLE stream has no npu_stream"
            )
        try:
            handle = int(getattr(raw_handle, "value", raw_handle))
        except (TypeError, ValueError):
            raise RuntimeError(
                "C8_KVAF_RESOURCE_QUOTA_UNAVAILABLE cannot_convert_stream_handle"
            ) from None
        return ctypes.c_void_p(handle), handle

    @staticmethod
    def _check_status(operation: str, status: int, *, label: str) -> None:
        if status != 0:
            raise RuntimeError(
                "C8_KVAF_RESOURCE_QUOTA_FAILED "
                f"operation={operation} label={label} acl_error={status}"
            )

    def _read_value(
        self,
        function: object,
        resource_type: int,
        *,
        label: str,
        stream_handle: ctypes.c_void_p | None = None,
    ) -> int:
        value = ctypes.c_uint32()
        if stream_handle is None:
            status = int(function(resource_type, ctypes.byref(value)))
            operation = "aclrtGetResInCurrentThread"
        else:
            status = int(function(stream_handle, resource_type, ctypes.byref(value)))
            operation = "aclrtGetStreamResLimit"
        self._check_status(operation, status, label=label)
        return int(value.value)

    def configure(
        self,
        stream: object,
        *,
        label: str,
        cube_cores: int | None,
        vector_cores: int | None,
    ) -> StreamResourceState:
        if cube_cores is None and vector_cores is None:
            raise ValueError("at least one stream resource limit is required")
        stream_handle, handle_value = self._stream_handle(stream)
        for resource_type, requested in (
            (self._CUBE_CORE, cube_cores),
            (self._VECTOR_CORE, vector_cores),
        ):
            if requested is None:
                continue
            status = int(
                self._set_stream_limit(
                    stream_handle,
                    resource_type,
                    ctypes.c_uint32(requested),
                )
            )
            self._check_status("aclrtSetStreamResLimit", status, label=label)
        state = StreamResourceState(
            label=label,
            stream_handle=handle_value,
            cube_cores=self._read_value(
                self._get_stream_limit,
                self._CUBE_CORE,
                label=label,
                stream_handle=stream_handle,
            ),
            vector_cores=self._read_value(
                self._get_stream_limit,
                self._VECTOR_CORE,
                label=label,
                stream_handle=stream_handle,
            ),
        )
        if cube_cores is not None and state.cube_cores != cube_cores:
            raise RuntimeError(
                "C8_KVAF_RESOURCE_QUOTA_FAILED "
                f"label={label} requested_cube={cube_cores} "
                f"observed_cube={state.cube_cores}"
            )
        if vector_cores is not None and state.vector_cores != vector_cores:
            raise RuntimeError(
                "C8_KVAF_RESOURCE_QUOTA_FAILED "
                f"label={label} requested_vector={vector_cores} "
                f"observed_vector={state.vector_cores}"
            )
        return state

    def bind_current_thread(
        self,
        stream: object,
        *,
        label: str,
        expected_cube_cores: int | None,
        expected_vector_cores: int | None,
    ) -> StreamResourceState:
        stream_handle, handle_value = self._stream_handle(stream)
        status = int(self._use_stream_in_current_thread(stream_handle))
        self._check_status("aclrtUseStreamResInCurrentThread", status, label=label)
        state = StreamResourceState(
            label=label,
            stream_handle=handle_value,
            cube_cores=self._read_value(
                self._get_current_thread_limit, self._CUBE_CORE, label=label
            ),
            vector_cores=self._read_value(
                self._get_current_thread_limit, self._VECTOR_CORE, label=label
            ),
        )
        if (
            expected_cube_cores is not None
            and state.cube_cores != expected_cube_cores
        ):
            raise RuntimeError(
                "C8_KVAF_RESOURCE_QUOTA_FAILED "
                f"label={label} thread_cube={state.cube_cores} "
                f"expected_cube={expected_cube_cores}"
            )
        if (
            expected_vector_cores is not None
            and state.vector_cores != expected_vector_cores
        ):
            raise RuntimeError(
                "C8_KVAF_RESOURCE_QUOTA_FAILED "
                f"label={label} thread_vector={state.vector_cores} "
                f"expected_vector={expected_vector_cores}"
            )
        return state

    def unbind_current_thread(self, stream: object, *, label: str) -> None:
        stream_handle, _ = self._stream_handle(stream)
        status = int(self._unuse_stream_in_current_thread(stream_handle))
        self._check_status(
            "aclrtUnuseStreamResInCurrentThread", status, label=label
        )

    def reset_stream_limit(self, stream: object, *, label: str) -> None:
        stream_handle, _ = self._stream_handle(stream)
        status = int(self._reset_stream_limit(stream_handle))
        self._check_status("aclrtResetStreamResLimit", status, label=label)


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
    parser.add_argument(
        "--transport",
        choices=("local", "remote"),
        default=os.getenv(
            "VLLM_ASCEND_KVAF_TRANSPORT",
            os.getenv("KVAF_TRANSPORT", "remote"),
        ).strip().lower(),
        help=(
            "KVAF data plane: local=A5 HBM<->A5 CPU DDR Direct-HCCP; "
            "remote=A5<->K5 CPU-service. No automatic fallback occurs."
        ),
    )
    parser.add_argument(
        "--enable-stream-resource-quota",
        "--enable-stream-resource-slice",
        dest="enable_stream_resource_quota",
        action="store_true",
        help=(
            "Enable the eager CANN per-stream resource-quota probe. "
            "This is not physical core-ID pinning; the old --enable-stream-"
            "resource-slice spelling remains an alias."
        ),
    )
    parser.add_argument(
        "--li-cube-cores",
        type=int,
        default=31,
        help="LI stream Cube-core limit when resource quota is enabled.",
    )
    parser.add_argument(
        "--li-vector-cores",
        type=int,
        default=62,
        help="LI stream Vector-core limit when resource quota is enabled.",
    )
    parser.add_argument(
        "--get-vector-cores",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility argument. KVAF Get uses --npu-cores "
            "as its direct AIV launch block count; no CANN Get quota is set."
        ),
    )
    parser.add_argument(
        "--enable-kvaf-get-kernel-profile",
        action="store_true",
        help=(
            "After timing, run one non-timed isolated Get and print the "
            "KVAF per-logical-AIV-block timeline. This reports no physical "
            "AIV ID or affinity."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--batch-get-kernel",
        type=Path,
        required=True,
        help="Built kvaf_execute_batch_get_kernels.o for the selected KVAF build.",
    )
    parser.add_argument(
        "--batch-put-kernel",
        type=Path,
        required=True,
        help="Built kvaf_execute_batch_put_kernels.o for the selected KVAF build.",
    )
    parser.add_argument(
        "--cpu-service-addr",
        default=os.getenv(
            "VLLM_ASCEND_KVAF_CPU_SERVICE_ADDR",
            os.getenv("KVAF_CPU_SERVICE_ADDR", ""),
        ),
        help="Required only by --transport remote, for example 141.61.92.77:28999.",
    )
    parser.add_argument(
        "--max-ddr-gib",
        type=float,
        default=4.0,
        help="Safety limit for the selected KVAF DDR allocation.",
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
    if args.enable_stream_resource_quota:
        if min(args.li_cube_cores, args.li_vector_cores) <= 0:
            raise ValueError("stream-resource limits must be positive")
        if (
            args.get_vector_cores is not None
            and args.npu_cores != args.get_vector_cores
        ):
            raise ValueError(
                "--npu-cores must equal --get-vector-cores when "
                "--get-vector-cores is supplied; KVAF Get has no separate "
                "CANN quota and uses --npu-cores as its launch block count"
            )
    if args.max_ddr_gib <= 0:
        raise ValueError("--max-ddr-gib must be positive")
    if args.transport == "remote" and not args.cpu_service_addr:
        raise ValueError(
            "--cpu-service-addr is required when --transport remote"
        )
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


def create_kvaf_runtime(
    args: argparse.Namespace,
    device_id: int,
) -> object:
    """Construct the one explicitly selected KVAF ABI; do not fall back."""

    if args.transport == "local":
        try:
            runtime = kvaf.NpuRuntime(
                str(args.batch_get_kernel),
                device_id,
                args.npu_cores,
                str(args.batch_put_kernel),
                args.jettys_per_core,
            )
        except TypeError as error:
            raise RuntimeError(
                "KVAF transport=local ABI mismatch: this benchmark requires "
                "the dsa-direct-get NpuRuntime "
                "(get_kernel, device_id, npu_cores, put_kernel, jettys)"
            ) from error
        if callable(getattr(runtime, "link_to", None)):
            raise RuntimeError(
                "KVAF transport=local ABI mismatch: the selected binding "
                "exposes remote link_to(); select the dsa-direct-get build"
            )
        return runtime

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
            "KVAF transport=remote ABI mismatch: this benchmark requires "
            "the CPU-service NpuRuntime "
            "(get_kernel, 656, 128, device_id, npu_cores, put_kernel, jettys)"
        ) from error
    link_to = getattr(runtime, "link_to", None)
    if not callable(link_to):
        raise RuntimeError(
            "KVAF transport=remote ABI mismatch: the selected binding does "
            "not provide link_to(); select the CPU-service build"
        )
    link_to(args.cpu_service_addr)
    return runtime


def populate_kvaf(
    args: argparse.Namespace,
    runtime: object,
    device: torch.device,
    table_capacity: int,
) -> tuple[object, torch.Tensor, torch.Tensor, list[int]]:
    """Fill selected DDR, then materialize the stable NPU address table."""

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
            "KVAF access_table_view(device='npu') did not return "
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
        raise AssertionError(
            "KVAF BatchGet output payload does not match BatchPut data"
        )


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


def capture_get_aiv_profile(
    runtime: object,
    launch_get: Callable[[object], None],
    copy_stream: object,
) -> list[dict[str, int]]:
    """Capture one non-timed Get profile without perturbing benchmark samples."""

    start_profile = getattr(runtime, "start_batch_get_kernel_profile", None)
    stop_profile = getattr(runtime, "stop_batch_get_kernel_profile", None)
    if not callable(start_profile) or not callable(stop_profile):
        raise RuntimeError(
            "KVAF binding has no BatchGet kernel profile API; rebuild the "
            "selected KVAF source with its current NPU Runtime binding"
        )

    start_profile()
    try:
        with torch.npu.stream(copy_stream):
            launch_get(copy_stream)
        copy_stream.synchronize()
    except Exception:
        # Best effort only: disable the optional profile before propagating the
        # original benchmark failure.
        try:
            stop_profile()
        except Exception:
            pass
        raise

    raw_profiles = stop_profile()
    profiles: list[dict[str, int]] = []
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, dict):
            raise RuntimeError("KVAF BatchGet profile item is not a dictionary")
        profiles.append({str(key): int(value) for key, value in raw_profile.items()})
    return profiles


def report_get_aiv_profiles(profiles: list[dict[str, int]]) -> None:
    """Emit logical Get-worker activity; the runtime exposes no physical ID."""

    for profile in profiles:
        start = profile.get("kernel_start_cycle", 0)
        finish = profile.get("kernel_finish_cycle", 0)
        print(
            "C8_KVAF_GET_AIV_PROFILE "
            "role=isolated "
            f"logical_aiv_block={profile.get('core', -1)} "
            "physical_core_id=not_available "
            f"active_requests={profile.get('active_requests', 0)} "
            f"total_posts={profile.get('total_posts', 0)} "
            f"kernel_cycles={max(0, finish - start)} "
            f"setup_done_cycle={profile.get('setup_done_cycle', 0)} "
            f"compact_done_cycle={profile.get('compact_done_cycle', 0)} "
            f"first_doorbell_cycle={profile.get('first_doorbell_cycle', 0)} "
            f"completion_done_cycle={profile.get('completion_done_cycle', 0)} "
            "jetty_completion_wait_cycles="
            f"{profile.get('jetty_completion_wait_cycles', 0)} "
            f"jetty_completion_polls={profile.get('jetty_completion_polls', 0)}",
            flush=True,
        )


def warmup(launch: Callable[[], object], count: int) -> None:
    retained: list[object] = []
    for _ in range(count):
        retained.append(launch())
    torch.npu.synchronize()


def time_li_only(
    launch_li: Callable[[], torch.Tensor],
    li_stream: object,
    *,
    warmup_count: int,
    iters: int,
) -> list[float]:
    warmup(launch_li, warmup_count)
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    retained: list[torch.Tensor] = []
    for start, end in zip(starts, ends):
        start.record(li_stream)
        retained.append(launch_li())
        end.record(li_stream)
    ends[-1].synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def time_get_only(
    launch_get: Callable[[object], None],
    li_stream: object,
    copy_stream: object,
    *,
    warmup_count: int,
    iters: int,
) -> list[float]:
    def launch_warmup() -> tuple[object, object]:
        gate = torch.npu.Event()
        done = torch.npu.Event()
        gate.record(li_stream)
        with torch.npu.stream(copy_stream):
            copy_stream.wait_event(gate)
            launch_get(copy_stream)
            done.record(copy_stream)
        li_stream.wait_event(done)
        return gate, done

    warmup(launch_warmup, warmup_count)
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    gates = [torch.npu.Event() for _ in range(iters)]
    dones = [torch.npu.Event() for _ in range(iters)]
    for start, end, gate, done in zip(starts, ends, gates, dones):
        start.record(li_stream)
        gate.record(li_stream)
        with torch.npu.stream(copy_stream):
            copy_stream.wait_event(gate)
            launch_get(copy_stream)
            done.record(copy_stream)
        li_stream.wait_event(done)
        end.record(li_stream)
    ends[-1].synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def time_serial(
    launch_li: Callable[[], torch.Tensor],
    launch_get: Callable[[object], None],
    li_stream: object,
    copy_stream: object,
    *,
    warmup_count: int,
    iters: int,
) -> list[float]:
    def launch_warmup() -> tuple[torch.Tensor, object, object]:
        li_done = torch.npu.Event()
        get_done = torch.npu.Event()
        result = launch_li()
        li_done.record(li_stream)
        with torch.npu.stream(copy_stream):
            copy_stream.wait_event(li_done)
            launch_get(copy_stream)
            get_done.record(copy_stream)
        li_stream.wait_event(get_done)
        return result, li_done, get_done

    warmup(launch_warmup, warmup_count)
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    li_dones = [torch.npu.Event() for _ in range(iters)]
    get_dones = [torch.npu.Event() for _ in range(iters)]
    retained: list[torch.Tensor] = []
    for start, end, li_done, get_done in zip(starts, ends, li_dones, get_dones):
        start.record(li_stream)
        retained.append(launch_li())
        li_done.record(li_stream)
        with torch.npu.stream(copy_stream):
            copy_stream.wait_event(li_done)
            launch_get(copy_stream)
            get_done.record(copy_stream)
        li_stream.wait_event(get_done)
        end.record(li_stream)
    ends[-1].synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def time_concurrent_join(
    launch_li: Callable[[], torch.Tensor],
    launch_get: Callable[[object], None],
    li_stream: object,
    copy_stream: object,
    *,
    warmup_count: int,
    iters: int,
) -> ConcurrentTimingSamples:
    def launch_warmup() -> tuple[torch.Tensor, object, object]:
        gate = torch.npu.Event()
        get_done = torch.npu.Event()
        gate.record(li_stream)
        with torch.npu.stream(copy_stream):
            copy_stream.wait_event(gate)
            launch_get(copy_stream)
            get_done.record(copy_stream)
        result = launch_li()
        # li_stream is also the LI execution stream. Its in-stream ordering
        # already includes LI; this wait makes the following join include Get.
        li_stream.wait_event(get_done)
        return result, gate, get_done

    warmup(launch_warmup, warmup_count)
    join_starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    join_ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    li_starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    li_ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    get_starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    get_ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    gates = [torch.npu.Event() for _ in range(iters)]
    get_dones = [torch.npu.Event() for _ in range(iters)]
    retained: list[torch.Tensor] = []
    for (
        join_start,
        join_end,
        li_start,
        li_end,
        get_start,
        get_end,
        gate,
        get_done,
    ) in zip(
        join_starts,
        join_ends,
        li_starts,
        li_ends,
        get_starts,
        get_ends,
        gates,
        get_dones,
    ):
        # The common gate makes both streams start after the same LI-stream
        # marker. LI and Get have no dependency on each other here.
        join_start.record(li_stream)
        gate.record(li_stream)
        with torch.npu.stream(copy_stream):
            copy_stream.wait_event(gate)
            get_start.record(copy_stream)
            launch_get(copy_stream)
            get_end.record(copy_stream)
            get_done.record(copy_stream)
        li_start.record(li_stream)
        retained.append(launch_li())
        li_end.record(li_stream)
        # join_end is recorded on the LI stream after LI's in-stream work and
        # after the Get completion dependency, so it is a true two-stream join.
        li_stream.wait_event(get_done)
        join_end.record(li_stream)
    join_ends[-1].synchronize()
    return ConcurrentTimingSamples(
        join_us=[
            start.elapsed_time(end) * 1000.0
            for start, end in zip(join_starts, join_ends)
        ],
        li_span_us=[
            start.elapsed_time(end) * 1000.0
            for start, end in zip(li_starts, li_ends)
        ],
        get_span_us=[
            start.elapsed_time(end) * 1000.0
            for start, end in zip(get_starts, get_ends)
        ],
    )


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
    runtime = create_kvaf_runtime(args, device_id)
    control_stream = torch.npu.current_stream()
    # Do not reuse the default/control stream for LI.  Its earlier input
    # construction is intentionally not subject to the experimental quota.
    # The two streams below receive their limits before their first task.
    li_stream = torch.npu.Stream()
    copy_stream = torch.npu.Stream()
    li_inputs_ready = torch.npu.Event()
    li_inputs_ready.record(control_stream)
    handle = None
    keys: list[int] = []
    resource_api: AclStreamResourceApi | None = None
    limited_streams: list[tuple[str, object]] = []
    try:
        if args.enable_stream_resource_quota:
            resource_api = AclStreamResourceApi()
            li_resource_state = resource_api.configure(
                li_stream,
                label="li",
                cube_cores=args.li_cube_cores,
                vector_cores=args.li_vector_cores,
            )
            limited_streams = [("li", li_stream)]
            print(
                "C8_KVAF_RESOURCE_QUOTA_CONFIG "
                f"library={resource_api.library_name} "
                "physical_core_id_proof=0 "
                "li_stream_explicit=1 get_stream_explicit=1 "
                f"li_stream=0x{li_resource_state.stream_handle:x} "
                f"li_cube_limit={li_resource_state.cube_cores} "
                f"li_vector_limit={li_resource_state.vector_cores} "
                f"get_stream=0x{AclStreamResourceApi._stream_handle(copy_stream)[1]:x} "
                f"get_aiv_launch_blocks={args.npu_cores}",
                flush=True,
            )
            print(
                "C8_KVAF_RESOURCE_QUOTA_GET "
                f"stream=0x{AclStreamResourceApi._stream_handle(copy_stream)[1]:x} "
                "submit_tls=not_used "
                "cann_quota=not_set "
                "reason=kvaf_direct_aiv_grid_is_explicit",
                flush=True,
            )

        li_binding_printed = False

        def run_with_li_quota(operation: Callable[[], object]) -> object:
            """Expose LI's configured quota only while its host tiler submits."""
            nonlocal li_binding_printed
            if resource_api is None:
                return operation()
            with torch.npu.stream(li_stream):
                li_thread_state = resource_api.bind_current_thread(
                    li_stream,
                    label="li",
                    expected_cube_cores=args.li_cube_cores,
                    expected_vector_cores=args.li_vector_cores,
                )
            if not li_binding_printed:
                li_binding_printed = True
                print(
                    "C8_KVAF_RESOURCE_QUOTA_BIND "
                    f"label=li stream=0x{li_thread_state.stream_handle:x} "
                    "submit_scope=one_measured_phase "
                    f"thread_cube_limit={li_thread_state.cube_cores} "
                    f"thread_vector_limit={li_thread_state.vector_cores}",
                    flush=True,
                )
            try:
                return operation()
            finally:
                with torch.npu.stream(li_stream):
                    resource_api.unbind_current_thread(li_stream, label="li")

        li_input_wait_enqueued = False
        def launch_li() -> torch.Tensor:
            nonlocal li_input_wait_enqueued
            with torch.npu.stream(li_stream):
                if not li_input_wait_enqueued:
                    li_stream.wait_event(li_inputs_ready)
                    li_input_wait_enqueued = True
                return official_c8_lightning_indexer(c8_inputs)

        print(
            "KVAF_TRANSPORT "
            f"mode={args.transport} "
            "backend="
            f"{'aclrtMallocHost_direct_hccp' if args.transport == 'local' else 'peer_cpu_service'} "
            "cpu_service_addr="
            f"{args.cpu_service_addr if args.transport == 'remote' else 'not_used'}",
            flush=True,
        )
        print(
            "C8_KVAF_OVERLAP_CONFIG "
            f"device={device} device_name={device_name!r} "
            f"batch={args.batch_size} source_len={args.source_len} "
            f"heads={args.heads} qcap={args.qcap} "
            "key_layout=disjoint_physical_blocks_per_row_shuffled "
            f"miss_count={args.miss_count} active_tokens={active_tokens} "
            "descriptor_source=official_c8_topk_known_miss_prefix "
            "overlap_scope=prebuilt_descriptor_upper_bound "
            f"transport={args.transport} "
            "cpu_service_addr="
            f"{args.cpu_service_addr if args.transport == 'remote' else 'not_used'} "
            f"ddr_gib={ddr_gib:.3f} warmup={args.warmup} iters={args.iters} "
            f"npu_cores={args.npu_cores} jettys_per_core={args.jettys_per_core} "
            "kvaf_get_kernel_profile="
            f"{int(args.enable_kvaf_get_kernel_profile)} "
            "execution_mode="
            f"{'stream_resource_quota' if args.enable_stream_resource_quota else 'baseline'} "
            f"dsa_registration_backend={dsa_registration_backend}",
            flush=True,
        )

        # This is deliberately outside timing: it makes a descriptor from
        # genuine C8 Top-K IDs, then gives the overlap experiment stable input.
        exact_topk = run_with_li_quota(launch_li)
        if not isinstance(exact_topk, torch.Tensor):
            raise AssertionError("LI prelude did not return an NPU Tensor")
        li_stream.synchronize()
        validate_official_topk(exact_topk, args.source_len)
        inputs, outputs, packed_output = make_prepare_case(args, device, exact_topk)
        launch_prepare(args, inputs, outputs)
        validate_prepare(args, inputs, outputs)

        runtime.initialize(ddr_bytes, ENTRY_BYTES)
        handle, addresses, valid, keys = populate_kvaf(
            args, runtime, device, table_capacity
        )
        get_inputs_ready = torch.npu.Event()
        get_inputs_ready.record(control_stream)

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

        with torch.npu.stream(copy_stream):
            copy_stream.wait_event(get_inputs_ready)
            launch_get(copy_stream)
        copy_stream.synchronize()
        validate_batch_get(outputs, packed_output)
        print(
            "C8_KVAF_OVERLAP_CORRECTNESS "
            "official_c8_topk=1 prepare=1 kvaf_payload=1",
            flush=True,
        )

        li_samples = run_with_li_quota(
            lambda: time_li_only(
                launch_li,
                li_stream,
                warmup_count=args.warmup,
                iters=args.iters,
            )
        )
        if not isinstance(li_samples, list):
            raise AssertionError("LI timing did not return samples")
        get_samples = time_get_only(
            launch_get,
            li_stream,
            copy_stream,
            warmup_count=args.warmup,
            iters=args.iters,
        )
        if args.enable_kvaf_get_kernel_profile:
            report_get_aiv_profiles(
                capture_get_aiv_profile(runtime, launch_get, copy_stream)
            )
        serial_samples = run_with_li_quota(
            lambda: time_serial(
                launch_li,
                launch_get,
                li_stream,
                copy_stream,
                warmup_count=args.warmup,
                iters=args.iters,
            )
        )
        if not isinstance(serial_samples, list):
            raise AssertionError("serial timing did not return samples")
        concurrent_timing = run_with_li_quota(
            lambda: time_concurrent_join(
                launch_li,
                launch_get,
                li_stream,
                copy_stream,
                warmup_count=args.warmup,
                iters=args.iters,
            )
        )
        if not isinstance(concurrent_timing, ConcurrentTimingSamples):
            raise AssertionError("concurrent timing did not return samples")
        concurrent_samples = concurrent_timing.join_us

        report(
            "c8_li_only",
            li_samples,
            warmup=args.warmup,
            active_tokens=active_tokens,
            moves_payload=False,
        )
        get_label = f"{args.transport}_get_only"
        serial_label = f"c8_li_then_{args.transport}_get_serial"
        concurrent_label = f"c8_li_with_{args.transport}_get_concurrent_join"
        concurrent_li_span_label = (
            f"c8_li_with_{args.transport}_get_concurrent_li_span"
        )
        concurrent_get_span_label = (
            f"c8_li_with_{args.transport}_get_concurrent_get_span"
        )
        report(
            get_label,
            get_samples,
            warmup=args.warmup,
            active_tokens=active_tokens,
            moves_payload=True,
        )
        report(
            serial_label,
            serial_samples,
            warmup=args.warmup,
            active_tokens=active_tokens,
            moves_payload=True,
        )
        report(
            concurrent_label,
            concurrent_samples,
            warmup=args.warmup,
            active_tokens=active_tokens,
            moves_payload=True,
        )
        report(
            concurrent_li_span_label,
            concurrent_timing.li_span_us,
            warmup=args.warmup,
            active_tokens=active_tokens,
            moves_payload=False,
        )
        report(
            concurrent_get_span_label,
            concurrent_timing.get_span_us,
            warmup=args.warmup,
            active_tokens=active_tokens,
            moves_payload=True,
        )
        serial_p90 = percentile(serial_samples, 0.90)
        concurrent_p90 = percentile(concurrent_samples, 0.90)
        li_p90 = percentile(li_samples, 0.90)
        get_p90 = percentile(get_samples, 0.90)
        concurrent_li_span_p90 = percentile(concurrent_timing.li_span_us, 0.90)
        concurrent_get_span_p90 = percentile(concurrent_timing.get_span_us, 0.90)
        per_stream_p90_reference = max(li_p90, get_p90)
        hide_budget_p90 = serial_p90 - per_stream_p90_reference
        hidden_p90 = serial_p90 - concurrent_p90
        overlap_efficiency = (
            hidden_p90 / hide_budget_p90 if hide_budget_p90 > 0 else math.nan
        )
        contention_penalty_p90 = concurrent_p90 - per_stream_p90_reference
        print(
            "C8_KVAF_OVERLAP_RESULT "
            f"li_p90_us={li_p90:.3f} "
            f"get_p90_us={get_p90:.3f} "
            f"serial_p90_us={serial_p90:.3f} "
            f"concurrent_p90_us={concurrent_p90:.3f} "
            f"ideal_lower_bound_p90_us={per_stream_p90_reference:.3f} "
            f"hide_budget_p90_us={hide_budget_p90:.3f} "
            f"hidden_p90_us={hidden_p90:.3f} "
            f"overlap_efficiency={overlap_efficiency:.3f} "
            f"contention_penalty_p90_us={contention_penalty_p90:.3f} "
            f"concurrent_li_span_p90_us={concurrent_li_span_p90:.3f} "
            f"concurrent_get_span_p90_us={concurrent_get_span_p90:.3f}",
            flush=True,
        )
        if args.enable_stream_resource_quota:
            print(
                "C8_KVAF_RESOURCE_QUOTA_RESULT "
                f"serial_p90_us={serial_p90:.3f} "
                f"concurrent_p90_us={concurrent_p90:.3f} "
                f"per_stream_p90_reference_us={per_stream_p90_reference:.3f} "
                f"overlap_saved_p90_us={serial_p90 - concurrent_p90:.3f}",
                flush=True,
            )
            print(
                "C8_KVAF_RESOURCE_QUOTA_CAPABILITY_PASS "
                "physical_core_id_proof=0",
                flush=True,
            )
        print("C8_KVAF_OVERLAP_CORRECTNESS_PASS", flush=True)
    finally:
        if resource_api is not None:
            try:
                torch.npu.synchronize()
            except Exception as error:
                print(
                    "C8_KVAF_RESOURCE_QUOTA_CLEANUP "
                    f"synchronize_error={error}",
                    flush=True,
                )
            for label, stream in limited_streams:
                try:
                    resource_api.reset_stream_limit(stream, label=label)
                    print(
                        "C8_KVAF_RESOURCE_QUOTA_CLEANUP "
                        f"label={label} stream_limit=reset",
                        flush=True,
                    )
                except Exception as error:
                    print(
                        "C8_KVAF_RESOURCE_QUOTA_CLEANUP "
                        f"label={label} stream_limit_reset_error={error}",
                        flush=True,
                    )
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
