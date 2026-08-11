"""Ascend 950 penetration test for swapped-memory DRAM -> HBM copy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

# Select the repository-local custom OPP before torch_npu is initialized.
import nanovllm_dsa_a5
import torch_npu  # type: ignore  # noqa: E402


BLOCK_SIZE = 128
KPE_DIM = 64
CKV_DIM = 512
CKV_POISON = 37.0
KPE_POISON = -29.0


@dataclass
class Case:
    device: torch.device
    device_name: str
    dram_kpe_cpu: torch.Tensor
    dram_ckv_cpu: torch.Tensor
    dram_kpe: torch.Tensor
    dram_ckv: torch.Tensor
    hbm_kpe: torch.Tensor
    hbm_ckv: torch.Tensor
    dram_table_cpu: torch.Tensor
    hbm_table_cpu: torch.Tensor
    dram_table: torch.Tensor
    hbm_table: torch.Tensor
    src_ids_cpu: torch.Tensor
    dst_slots_cpu: torch.Tensor
    copy_counts_cpu: torch.Tensor
    src_ids: torch.Tensor
    dst_slots: torch.Tensor
    copy_counts: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Require a real Ascend 950 and prove that the custom kernel "
            "copies CKV/KPE from empty_with_swapped_memory into HBM."
        )
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--source-len", type=int, default=4096)
    parser.add_argument("--hbm-slots", type=int, default=512)
    parser.add_argument("--copy-min", type=int, default=1)
    parser.add_argument("--copy-max", type=int, default=32)
    parser.add_argument("--copy-cap", type=int, default=2048)
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16"),
        default="bf16",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--allow-non-a5",
        action="store_true",
        help="Only for portability debugging; a passing run is not A5 proof.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.source_len <= 0 or args.hbm_slots <= 0:
        raise ValueError("--source-len and --hbm-slots must be positive.")
    if not 0 <= args.copy_min <= args.copy_max:
        raise ValueError(
            "--copy-min/--copy-max must satisfy 0 <= min <= max."
        )
    if args.copy_cap <= 0 or args.copy_max > args.copy_cap:
        raise ValueError(
            "--copy-cap must be positive and at least --copy-max."
        )
    if args.copy_max > args.source_len:
        raise ValueError(
            "--copy-max must not exceed --source-len."
        )
    if args.copy_max >= args.hbm_slots:
        raise ValueError(
            "--hbm-slots must exceed --copy-max so the test has an "
            "untouched guard row."
        )
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters positive.")


def random_block_table(
    batch_size: int,
    blocks_per_row: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    total_blocks = batch_size * blocks_per_row
    table = torch.randperm(
        total_blocks,
        generator=generator,
        dtype=torch.int64,
    ).reshape(batch_size, blocks_per_row)
    return table.to(torch.int32).contiguous(), total_blocks


def random_shared_source_table(
    batch_size: int,
    blocks_per_row: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """Random per-request views over one read-only physical DRAM pool."""

    table = torch.stack(
        [
            torch.randperm(
                blocks_per_row,
                generator=generator,
                dtype=torch.int64,
            ).to(torch.int32)
            for _ in range(batch_size)
        ]
    )
    return table.contiguous(), blocks_per_row


def swapped_from_cpu(
    cpu: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if not hasattr(torch_npu, "empty_with_swapped_memory"):
        raise RuntimeError(
            "torch_npu.empty_with_swapped_memory is unavailable. "
            "This test refuses to replace DRAM with an HBM tensor."
        )
    tensor = torch_npu.empty_with_swapped_memory(
        cpu.shape,
        dtype=cpu.dtype,
        device=device,
    )
    tensor.fill_(0)
    staging = cpu.to(device)
    tensor.add_(staging)
    torch.npu.synchronize()
    del staging
    torch.npu.empty_cache()
    return tensor


def make_cache_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randn(
        shape,
        generator=generator,
        dtype=torch.float32,
    ).to(dtype)


def make_case(args: argparse.Namespace) -> Case:
    device = torch.device(args.device)
    device_index = (
        device.index
        if device.index is not None
        else torch.npu.current_device()
    )
    get_device_name = getattr(torch.npu, "get_device_name", None)
    if get_device_name is None:
        get_device_name = torch_npu.npu.get_device_name
    device_name = get_device_name(device_index)
    if "950" not in device_name.lower() and not args.allow_non_a5:
        raise RuntimeError(
            f"Expected an Ascend 950 device, got {device_name!r}. "
            "Use --allow-non-a5 only for portability debugging."
        )

    generator = torch.Generator().manual_seed(args.seed)
    source_blocks_per_row = (
        args.source_len + BLOCK_SIZE - 1
    ) // BLOCK_SIZE
    hbm_blocks_per_row = (
        args.hbm_slots + BLOCK_SIZE - 1
    ) // BLOCK_SIZE
    # DRAM is read-only, so requests can use independent permutations of one
    # source pool. HBM destinations remain private to each request.
    dram_table_cpu, dram_blocks = random_shared_source_table(
        args.batch_size,
        source_blocks_per_row,
        generator,
    )
    hbm_table_cpu, hbm_blocks = random_block_table(
        args.batch_size,
        hbm_blocks_per_row,
        generator,
    )
    cache_dtype = (
        torch.bfloat16 if args.dtype == "bf16" else torch.float16
    )

    dram_kpe_cpu = make_cache_tensor(
        (dram_blocks, BLOCK_SIZE, KPE_DIM),
        cache_dtype,
        generator,
    )
    dram_ckv_cpu = make_cache_tensor(
        (dram_blocks, BLOCK_SIZE, CKV_DIM),
        cache_dtype,
        generator,
    )

    copy_cap = args.copy_cap
    src_ids_cpu = torch.full(
        (args.batch_size, copy_cap),
        -1,
        dtype=torch.int32,
    )
    dst_slots_cpu = torch.full_like(src_ids_cpu, -1)
    sampled_counts = torch.randint(
        args.copy_min,
        args.copy_max + 1,
        (args.batch_size,),
        generator=generator,
        dtype=torch.int64,
    )
    # Multi-row cases pin both endpoints.  Keep the seeded random sample for
    # batch=1 so a 0..300 benchmark does not become a zero-copy benchmark.
    if args.batch_size > 1:
        sampled_counts[0] = args.copy_min
        sampled_counts[1] = args.copy_max
    counts = sampled_counts.tolist()
    for row in range(args.batch_size):
        count = counts[row]
        src_ids_cpu[row, :count] = torch.randperm(
            args.source_len,
            generator=generator,
        )[:count].to(torch.int32)
        dst_slots_cpu[row, :count] = torch.randperm(
            args.hbm_slots,
            generator=generator,
        )[:count].to(torch.int32)
    copy_counts_cpu = torch.tensor(counts, dtype=torch.int32)

    return Case(
        device=device,
        device_name=device_name,
        dram_kpe_cpu=dram_kpe_cpu,
        dram_ckv_cpu=dram_ckv_cpu,
        dram_kpe=swapped_from_cpu(dram_kpe_cpu, device),
        dram_ckv=swapped_from_cpu(dram_ckv_cpu, device),
        hbm_kpe=torch.empty(
            hbm_blocks,
            BLOCK_SIZE,
            KPE_DIM,
            dtype=cache_dtype,
            device=device,
        ),
        hbm_ckv=torch.empty(
            hbm_blocks,
            BLOCK_SIZE,
            CKV_DIM,
            dtype=cache_dtype,
            device=device,
        ),
        dram_table_cpu=dram_table_cpu,
        hbm_table_cpu=hbm_table_cpu,
        dram_table=dram_table_cpu.to(device),
        hbm_table=hbm_table_cpu.to(device),
        src_ids_cpu=src_ids_cpu,
        dst_slots_cpu=dst_slots_cpu,
        copy_counts_cpu=copy_counts_cpu,
        src_ids=src_ids_cpu.to(device),
        dst_slots=dst_slots_cpu.to(device),
        copy_counts=copy_counts_cpu.to(device),
    )


def active_physical_rows(
    case: Case,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_rows: list[torch.Tensor] = []
    destination_rows: list[torch.Tensor] = []
    for row, count in enumerate(case.copy_counts_cpu.tolist()):
        source = case.src_ids_cpu[row, :count].to(torch.int64)
        destination = case.dst_slots_cpu[row, :count].to(torch.int64)
        source_rows.append(
            case.dram_table_cpu[row].to(torch.int64)[
                source // BLOCK_SIZE
            ]
            * BLOCK_SIZE
            + source % BLOCK_SIZE
        )
        destination_rows.append(
            case.hbm_table_cpu[row].to(torch.int64)[
                destination // BLOCK_SIZE
            ]
            * BLOCK_SIZE
            + destination % BLOCK_SIZE
        )
    return torch.cat(source_rows), torch.cat(destination_rows)


def poison_hbm(case: Case) -> None:
    case.hbm_ckv.fill_(CKV_POISON)
    case.hbm_kpe.fill_(KPE_POISON)


def call_scatter(
    case: Case,
    copy_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return nanovllm_dsa_a5.scatter_copy(
        case.hbm_kpe,
        case.hbm_ckv,
        case.dram_kpe,
        case.dram_ckv,
        case.hbm_table,
        case.dram_table,
        case.src_ids,
        case.dst_slots,
        case.copy_counts if copy_counts is None else copy_counts,
    )


def assert_poisoned(
    case: Case,
    destination_rows: torch.Tensor,
) -> None:
    destination_rows_npu = destination_rows.to(case.device)
    actual_ckv = (
        case.hbm_ckv.view(-1, CKV_DIM)[destination_rows_npu].cpu()
    )
    actual_kpe = (
        case.hbm_kpe.view(-1, KPE_DIM)[destination_rows_npu].cpu()
    )
    if not torch.equal(
        actual_ckv,
        torch.full_like(actual_ckv, CKV_POISON),
    ):
        raise AssertionError("Active CKV destinations were not poisoned.")
    if not torch.equal(
        actual_kpe,
        torch.full_like(actual_kpe, KPE_POISON),
    ):
        raise AssertionError("Active KPE destinations were not poisoned.")


def assert_all_hbm_poisoned(case: Case) -> None:
    ckv_ok = bool(torch.all(case.hbm_ckv == CKV_POISON).item())
    kpe_ok = bool(torch.all(case.hbm_kpe == KPE_POISON).item())
    if not ckv_ok or not kpe_ok:
        raise AssertionError(
            "copy_count=0 modified the HBM cache."
        )


def assert_copied(
    case: Case,
    source_rows: torch.Tensor,
    destination_rows: torch.Tensor,
) -> None:
    expected_ckv = case.dram_ckv_cpu.view(-1, CKV_DIM)[source_rows]
    expected_kpe = case.dram_kpe_cpu.view(-1, KPE_DIM)[source_rows]
    destination_rows_npu = destination_rows.to(case.device)
    actual_ckv = (
        case.hbm_ckv.view(-1, CKV_DIM)[destination_rows_npu].cpu()
    )
    actual_kpe = (
        case.hbm_kpe.view(-1, KPE_DIM)[destination_rows_npu].cpu()
    )
    if not torch.equal(actual_ckv, expected_ckv):
        max_abs = (actual_ckv.float() - expected_ckv.float()).abs().max()
        raise AssertionError(
            f"DRAM->HBM CKV mismatch, max_abs={float(max_abs):.9f}"
        )
    if not torch.equal(actual_kpe, expected_kpe):
        max_abs = (actual_kpe.float() - expected_kpe.float()).abs().max()
        raise AssertionError(
            f"DRAM->HBM KPE mismatch, max_abs={float(max_abs):.9f}"
        )

    active = set(destination_rows.tolist())
    total_rows = case.hbm_ckv.shape[0] * BLOCK_SIZE
    guard = next(index for index in range(total_rows) if index not in active)
    guard_ckv = case.hbm_ckv.view(-1, CKV_DIM)[guard].cpu()
    guard_kpe = case.hbm_kpe.view(-1, KPE_DIM)[guard].cpu()
    if not torch.equal(
        guard_ckv,
        torch.full_like(guard_ckv, CKV_POISON),
    ) or not torch.equal(
        guard_kpe,
        torch.full_like(guard_kpe, KPE_POISON),
    ):
        raise AssertionError("An inactive HBM guard row was modified.")


def run(args: argparse.Namespace) -> None:
    case = make_case(args)
    source_rows, destination_rows = active_physical_rows(case)
    copied_tokens = int(case.copy_counts_cpu.sum())
    payload_bytes = (
        copied_tokens
        * (CKV_DIM + KPE_DIM)
        * case.dram_ckv_cpu.element_size()
    )

    print(
        "A5_SCATTER_CONFIG "
        f"device={case.device} device_name={case.device_name!r} "
        f"dtype={args.dtype} batch={args.batch_size} "
        f"source_len={args.source_len} hbm_slots={args.hbm_slots} "
        f"copy_cap={args.copy_cap} "
        f"copy_range=[{args.copy_min},{args.copy_max}] "
        f"copy_counts={case.copy_counts_cpu.tolist()} "
        f"opapi={nanovllm_dsa_a5.local_opapi_path()}"
    )

    poison_hbm(case)
    torch.npu.synchronize()
    assert_poisoned(case, destination_rows)
    out_kpe, out_ckv = call_scatter(case)
    torch.npu.synchronize()
    if (
        out_kpe.data_ptr() != case.hbm_kpe.data_ptr()
        or out_ckv.data_ptr() != case.hbm_ckv.data_ptr()
    ):
        raise AssertionError("HBM outputs did not alias the input caches.")
    if copied_tokens:
        assert_copied(case, source_rows, destination_rows)
    else:
        assert_all_hbm_poisoned(case)
    print(
        "A5_SCATTER_DRAM_TO_HBM_CHECK "
        "allocator=empty_with_swapped_memory "
        f"poisoned_destinations={copied_tokens} "
        f"payload_bytes={payload_bytes} guard_unchanged=1 ok=1"
    )

    zero_counts = torch.zeros_like(case.copy_counts)
    poison_hbm(case)
    call_scatter(case, zero_counts)
    torch.npu.synchronize()
    assert_all_hbm_poisoned(case)
    print("A5_SCATTER_ZERO_COUNT_CHECK all_hbm_unchanged=1 ok=1")

    for _ in range(args.warmup):
        call_scatter(case)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        call_scatter(case)
    end.record()
    end.synchronize()
    avg_us = start.elapsed_time(end) * 1000 / args.iters
    if copied_tokens:
        assert_copied(case, source_rows, destination_rows)
    else:
        assert_all_hbm_poisoned(case)
    payload_gbps = (
        payload_bytes / (avg_us * 1000)
        if payload_bytes and avg_us
        else 0.0
    )
    print(
        "A5_SCATTER_RESULT "
        f"copy_min={args.copy_min} copy_max={args.copy_max} "
        f"copied_tokens={copied_tokens} avg_us={avg_us:.3f} "
        f"payload_gbps={payload_gbps:.3f} timer=npu_event "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print("A5_KVCACHE_SCATTER_COPY_UT_OK")


def main() -> None:
    args = parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
