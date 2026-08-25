#!/usr/bin/env python3
"""P0 validation for the one-kernel ExactScan + SpeculativePost layout.

This test validates logical MIX roles through the AIV half of a 1C2V launch;
it does not claim physical core affinity, an AIC-side execution trace, or
actual H2D. The next prototype replaces the two post markers with KVAF
Direct-HCCP PostOnly after KVAF exposes a device-control view.
"""

from __future__ import annotations

import argparse

import torch
import torch_npu  # noqa: F401

import nanovllm_exact_scan_probe  # noqa: F401


LI_AIV_COUNT = 62
POST_AIV_COUNT = 2
ROLE_TRACE_COUNT = 96
POST_MARKER_XOR = 0x5A5A5A5A


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:1")
    parser.add_argument("--scan-values", type=int, default=62 * 1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if args.scan_values < LI_AIV_COUNT:
        raise ValueError("--scan-values must be at least 62")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters positive")


def expected_checksums(values: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [
            int(values[lane::LI_AIV_COUNT].to(torch.int64).sum().item())
            for lane in range(LI_AIV_COUNT)
        ],
        dtype=torch.int64,
    )


def check_result(
    cpu_values: torch.Tensor,
    post_seed: torch.Tensor,
    li_checksums: torch.Tensor,
    post_markers: torch.Tensor,
    role_trace: torch.Tensor,
) -> None:
    torch.testing.assert_close(li_checksums.cpu(), expected_checksums(cpu_values))
    torch.testing.assert_close(
        post_markers.cpu(), post_seed ^ POST_MARKER_XOR
    )
    # Only AIV directly writes this trace. The fixed 1C2V tiling establishes
    # 31 LI groups plus one Post group; direct scalar AIC-to-GM stores are not
    # used as an execution proof because they are not a stable trace path.
    expected_trace = torch.zeros(ROLE_TRACE_COUNT, dtype=torch.int32)
    expected_trace[:LI_AIV_COUNT] = torch.arange(
        1000, 1000 + LI_AIV_COUNT, dtype=torch.int32
    )
    expected_trace[62:64] = torch.arange(2000, 2000 + POST_AIV_COUNT, dtype=torch.int32)
    torch.testing.assert_close(role_trace.cpu(), expected_trace)


def main() -> None:
    args = parse_args()
    check_args(args)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    device_index = device.index if device.index is not None else torch.npu.current_device()
    get_device_name = getattr(
        torch.npu, "get_device_name", torch_npu.npu.get_device_name
    )
    device_name = get_device_name(device_index)
    if "950" not in device_name.lower():
        raise RuntimeError(f"expected Ascend 950 for the MIX prototype, got {device_name!r}")

    cpu_values = (torch.arange(args.scan_values, dtype=torch.int32) % 97).contiguous()
    post_seed = torch.tensor([17, 29], dtype=torch.int32)
    scan_values = cpu_values.to(device)
    post_seed_npu = post_seed.to(device)

    print(
        "A5_EXACT_SCAN_SPECULATIVE_POST_PROBE_CONFIG "
        f"device={args.device} device_name={device_name!r} "
        f"scan_values={args.scan_values} li_groups=31 li_aiv=62 "
        "post_aiv=2 aic_idle_groups=1 "
        "logical_group_partition=1 physical_aiv_id_binding=0 "
        "direct_hccp_post=0"
    )

    for _ in range(args.warmup):
        torch.ops.a5_exact_scan_probe.exact_scan_speculative_post_probe(
            scan_values, post_seed_npu
        )
    torch.npu.synchronize()

    result = None
    for _ in range(args.iters):
        result = torch.ops.a5_exact_scan_probe.exact_scan_speculative_post_probe(
            scan_values, post_seed_npu
        )
    torch.npu.synchronize()
    assert result is not None
    check_result(cpu_values, post_seed, *result)
    print(
        "A5_EXACT_SCAN_SPECULATIVE_POST_PROBE_CORRECTNESS "
        "li_exact_partition=1 post_lanes=2 aiv_role_trace=1 "
        "aic_role=structural_only"
    )
    print("A5_EXACT_SCAN_SPECULATIVE_POST_PROBE_PASS")


if __name__ == "__main__":
    main()
