# SPDX-License-Identifier: Apache-2.0
"""Serving hot-cache prepare -> original LIDU -> D2H -> LRU, eager and graph.

Seeds valid hot data so no DDR server is needed. Get is a zero-task transport
stub; any nonzero miss fails the test. This does not verify real RMA bandwidth.
"""

from types import SimpleNamespace as NS

import numpy as np
import torch
import torch_npu  # noqa: F401
from kvop.lru import SlotLRU

from vllm_ascend.distributed.kv_transfer.kvaf.cache_policy import CacheFeedbackManager, KVCachePolicy
from vllm_ascend.distributed.kv_transfer.kvaf.state import DeviceCacheState, RequestCacheState


def main():
    torch.npu.set_device(0)
    pool = DeviceCacheState(("target",), 3, 16384, "npu:0")
    hot = pool.policy = KVCachePolicy(pool, 16, 8192)
    states, requests = {}, {}
    for name, queries in (("a", 1), ("b", 4)):
        progress = RequestCacheState(128)
        progress.prepare(tuple(range(33)), 4096, queries)
        states[name] = [
            NS(progress=progress, addresses=list(range(1, 33)), keys=list(range(32)), block_ids=list(range(32)))
        ]
        requests[name] = NS(num_prompt_tokens=4096)
    layout = NS(layer_names=("target",), packed_c8=True, planes=(NS(page_stride_bytes=128 * 656),))
    writer = NS(
        groups=[NS(layout=layout, geometry=(656, 128), source=hot.hot[0])],
        requests=states,
        runtimes={(656, 128): NS(batch_get=lambda *a, **kw: None)},
    )
    manager = CacheFeedbackManager(writer)
    manager.bind_device_pool(pool)
    hot.requests = requests
    topk = torch.arange(2048, dtype=torch.int32, device="npu:0").view(1, 1, -1).repeat(12, 1, 1)
    metadata = NS(num_prefills=0, cum_query_lens=torch.tensor([1, 5, 12], dtype=torch.int32, device="npu:0"))

    def prepare():
        manager.prepare(("a", "b"), {"a": 1, "b": 4}, decode=True, padded_requests=3)

    def check():
        manager.after_forward()
        manager.wait_pending()
        assert hot.inputs["miss_counts"].cpu().tolist() == [[0, 0, 0, 0]]
        for req in ("a", "b"):
            owner = hot.owners[req]
            assert pool.views["target"].token_to_slot[owner, :4096].cpu().tolist() == list(range(4096))

    try:
        prepare()
        for req in ("a", "b"):
            owner = hot.owners[req]
            for tensor in (pool.views["target"].token_to_slot, pool.views["target"].slot_to_token):
                tensor[owner, :4096].copy_(torch.arange(4096, dtype=torch.int32, device="npu:0"))
            # Keep initial victims outside the all-hit selection.
            lru = SlotLRU()
            lru.resize(hot.slot_capacity, hot.history_capacity)
            lru.commit(np.arange(4096, dtype=np.int32))
            lru.hot_snapshot = (lru.slots[0].copy(), lru.bitmap[0].copy())
            manager.owners[req, 0].lru = lru
        prepare()
        hot.run(0, topk, metadata)
        check()
        prepare()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            hot.run(0, topk, metadata)
        check()
        seen = []
        commit = manager._commit

        def observe(frames, host, ready, *args):
            ready.synchronize()
            seen.append(sorted(host[0, 0].tolist()))
            return commit(frames, host, ready, *args)

        manager._commit = observe
        for step in range(4):
            prepare()
            if step:
                topk.add_(1)
            graph.replay()
            manager.after_forward()
        manager.wait_pending()
        assert seen == [list(range(step, 2048 + step)) for step in range(4)]
        assert hot.inputs["miss_counts"].cpu().tolist() == [[0, 0, 0, 0]]
        print("PASS: serving hot maps, real LIDU, graph replay, 1/4-query padding and asynchronous LRU feedback")
    finally:
        manager.close()


if __name__ == "__main__":
    main()
