# SPDX-License-Identifier: Apache-2.0
"""Apply batched ABC metadata without overwriting device-owned history maps."""


def update_abc(token_to_slot, slot_to_token, descriptors, requests, history, tail):
    if token_to_slot.device.type == "npu":
        from .abc_update_kernel import update_abc_kernel

        update_abc_kernel[(descriptors.shape[0], requests)](
            token_to_slot,
            slot_to_token,
            descriptors,
            descriptors.shape[1],
            token_to_slot.shape[1],
            token_to_slot.shape[2],
            history,
            tail,
            BLOCK=512,
        )
        return
    # CPU reference used by byte-level service-path tests.
    import torch

    for group in range(descriptors.shape[0]):
        for row in range(requests):
            layer, owner, begin, end = descriptors[group, row].tolist()
            tokens = torch.arange(begin, end, dtype=torch.int32)
            token_to_slot[layer, owner, begin:end] = history + tokens % tail
            reverse = slot_to_token[layer, owner, history : history + tail]
            reverse.fill_(-1)
            reverse[tokens.long() % tail] = tokens
