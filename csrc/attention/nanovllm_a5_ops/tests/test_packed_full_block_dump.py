from __future__ import annotations

import torch
import torch_npu

import nanovllm_dsa_a5  # noqa: F401


BLOCK_SIZE = 128
PACKED_KV_DIM = 656


def main() -> None:
    device = torch.device("npu:0")
    hbm = torch.randint(
        -128,
        127,
        (8, BLOCK_SIZE, 1, PACKED_KV_DIM),
        dtype=torch.int8,
        device=device,
    )
    dram = torch_npu.empty_with_swapped_memory(
        (16, BLOCK_SIZE, 1, PACKED_KV_DIM),
        dtype=torch.int8,
        device=device,
    )
    src = torch.tensor([1, 3], dtype=torch.int32, device=device)
    dst = torch.tensor([2, 5], dtype=torch.int32, device=device)

    torch.ops.nanovllm_dsa.packed_full_block_dump(
        hbm,
        dram,
        src,
        dst,
    )
    torch.npu.synchronize()

    expected = hbm.index_select(0, src.to(torch.int64)).cpu()
    actual = dram.index_select(0, dst.to(torch.int64)).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("A5 packed_full_block_dump smoke test: PASS")


if __name__ == "__main__":
    main()
