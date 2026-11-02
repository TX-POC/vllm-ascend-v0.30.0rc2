# SPDX-License-Identifier: Apache-2.0
"""Compare the actual batched NPU update with its CPU reference; no CPU service needed."""

import torch
import torch_npu  # noqa: F401
from kvop.abc_update import update_abc


def main():
    torch.npu.set_device(0)
    shape = (3, 5, 2048)
    descriptors = torch.tensor(
        [[[2, 4, 350, 734], [2, 1, 768, 768]], [[0, 3, 383, 390], [0, 2, 128, 140]]], dtype=torch.int32
    )
    cpu_t = torch.full(shape, 99, dtype=torch.int32)
    cpu_s = cpu_t.clone()
    npu_t, npu_s = cpu_t.npu(), cpu_s.npu()
    device_desc = descriptors.npu()
    for shift in (0, 128):
        descriptors[:, :, 2:] += shift
        device_desc.copy_(descriptors)
        update_abc(cpu_t, cpu_s, descriptors, 2, 1024, 384)
        update_abc(npu_t, npu_s, device_desc, 2, 1024, 384)
        torch.testing.assert_close(npu_t.cpu(), cpu_t)
        torch.testing.assert_close(npu_s.cpu(), cpu_s)
    print("PASS: batched ABC forward/reverse maps, wrap, empty tail, and untouched history")


if __name__ == "__main__":
    main()
