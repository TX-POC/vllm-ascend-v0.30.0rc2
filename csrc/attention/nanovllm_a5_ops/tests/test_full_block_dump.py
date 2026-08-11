from __future__ import annotations

import torch
import torch_npu

import nanovllm_dsa_a5  # noqa: F401


BLOCK = 128
CKV = 512
ROPE = 64


def main() -> None:
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    hbm_blocks = 8
    dram_blocks = 16

    torch.manual_seed(7)
    hbm_ckv = torch.randn(
        (hbm_blocks, BLOCK, CKV), dtype=dtype, device=device
    )
    hbm_rope = torch.randn(
        (hbm_blocks, BLOCK, ROPE), dtype=dtype, device=device
    )
    dram_ckv = torch_npu.empty_with_swapped_memory(
        (dram_blocks, BLOCK, CKV), dtype=dtype, device=device
    )
    dram_rope = torch_npu.empty_with_swapped_memory(
        (dram_blocks, BLOCK, ROPE), dtype=dtype, device=device
    )

    src = torch.tensor([1, 3], dtype=torch.int32, device=device)
    dst = torch.tensor([2, 5], dtype=torch.int32, device=device)

    torch.ops.nanovllm_dsa.full_block_dump(
        hbm_rope,
        hbm_ckv,
        dram_rope,
        dram_ckv,
        src,
        dst,
    )
    torch.npu.synchronize()

    expected_ckv = hbm_ckv.index_select(0, src.to(torch.int64)).cpu()
    expected_rope = hbm_rope.index_select(0, src.to(torch.int64)).cpu()
    actual_ckv = dram_ckv.index_select(0, dst.to(torch.int64)).cpu()
    actual_rope = dram_rope.index_select(0, dst.to(torch.int64)).cpu()

    torch.testing.assert_close(actual_ckv, expected_ckv, rtol=0, atol=0)
    torch.testing.assert_close(actual_rope, expected_rope, rtol=0, atol=0)
    print("A5 full_block_dump smoke test: PASS")


if __name__ == "__main__":
    main()
