# SPDX-License-Identifier: Apache-2.0
"""Worker-only C8 Backend for the explicitly selected PD path.

One model layer owns one DDR group with geometry (656, 128). The existing
directory and Get implementation use that geometry without four-layer token
interleaving. Import this only from an enabled PD Worker/Runner extension.
"""

from threading import RLock

import torch

from ..cache_policy import KVCachePolicy
from ..layout import BLOCK_SIZE, HISTORY, TAIL, TOKEN_BYTES, TOPK
from ..state import DeviceCacheState
from ..transfer import get_tokens


class PDCachePolicy(KVCachePolicy):
    def __init__(self, pool, max_tokens, max_model_len, *, native=None, hot_tokens=HISTORY):
        super().__init__(
            pool, max_tokens, max_model_len, native=native, group_sizes=(1,) * len(pool.names), hot_tokens=hot_tokens
        )
        self._received_tails = {}
        self._abc_states = {}
        self.receive_slots = {}
        self.receive_blocks = {}
        self._receive_lock = RLock()
        self._receive_free = list(reversed(range(pool.max_requests - 1)))

    def reserve_receive(self, req_id, blocks=None):
        with self._receive_lock:
            bindings = tuple((layer, block) for layer, block in (blocks or {}).items())
            if any(key in self.receive_blocks and self.receive_blocks[key][0] != req_id for key in bindings):
                raise RuntimeError("native tail block is still owned by another PD receive")
            if req_id not in self.receive_slots:
                if not self._receive_free:
                    raise RuntimeError("PD ordinary tail receive capacity exhausted")
                self.receive_slots[req_id] = self._receive_free.pop()
            for key in bindings:
                self.receive_blocks[key] = (req_id, self.receive_slots[req_id])
            return self.receive_slots[req_id]

    def receive_slot_for(self, req_id, layer, native_block):
        with self._receive_lock:
            owner, slot = self.receive_blocks[layer, native_block]
            if owner != req_id:
                raise ValueError("ordinary receive block belongs to another request")
            return slot

    def bind(self, manager):
        names = tuple(layer.layout.layer_names for layer in manager.writer.groups)
        if names != tuple((name,) for name in self.pool.names):
            raise ValueError("PD requires one distinct Target layer per DDR group in pool order")
        super().bind(manager)

    def bind_received_tail(self, req_id, layer_name, tensor, tokens):
        """Retain a completed ordinary HBM receive; do not preload Hot here.

        The caller owns the DMA completion proof and request generation. The
        native compute preparation installs the tail before last-token replay
        or Decode. The source remains alive until request release, also
        allowing stream-ordered asynchronous copies on the worker stream.
        """
        if type(tokens) is not int or tokens < 0:
            raise ValueError("invalid received token length")
        tail = tokens % BLOCK_SIZE
        if not tail:
            raise ValueError("a complete-block prompt has no main KV tail")
        if layer_name not in self.pool.indices or req_id not in self.writer.requests:
            raise ValueError("tail has no existing request/layer owner")
        index = self.pool.indices[layer_name]
        state = self.writer.requests[req_id][index]
        if len(state.keys) != tokens // BLOCK_SIZE or len(state.addresses) != len(state.keys):
            raise ValueError("publish all complete main blocks before binding the tail")
        if (
            tensor.device != self.pool.device
            or not tensor.is_contiguous()
            or not tail * TOKEN_BYTES <= tensor.numel() * tensor.element_size() <= BLOCK_SIZE * TOKEN_BYTES
        ):
            raise ValueError("tail must be a contiguous receive on the Backend device with valid C8 bytes")
        key = (req_id, index)
        if key in self._received_tails:
            raise ValueError("tail already bound; do not replace live receive storage")
        self._received_tails[key] = (state, tensor, tokens)

    def prepare(self, manager, *, decode, padded_requests):
        frame = manager.groups[0]
        for row, req_id in enumerate(frame.request_ids):
            if req_id not in self.receive_slots:
                continue
            if req_id not in self.owners:
                self.owners[req_id] = self.free.pop()
                self.addresses[self.owners[req_id]].zero_()
                self.valid[self.owners[req_id]].zero_()
            owner = self.owners[req_id]
            for layer, inputs in enumerate(manager.groups):
                state = inputs.rows[row].state
                if self._abc_states.get((req_id, layer)) is state:
                    continue
                tokens = self.requests[req_id].num_prompt_tokens
                if tokens % BLOCK_SIZE:
                    self._seed(layer, owner, state, tokens // BLOCK_SIZE * BLOCK_SIZE, tokens, True)
                elif state.progress.computed_tokens < tokens:
                    # Native PD replays the final prompt token. For an aligned
                    # prompt its preceding partial page lives in committed DDR.
                    end = state.progress.computed_tokens
                    self._seed_replay_prefix(layer, owner, state, end // BLOCK_SIZE * BLOCK_SIZE, end)
                self._abc_states[req_id, layer] = state
        super().prepare(manager, decode=decode, padded_requests=padded_requests)
        # Native local recomputation after preemption owns its new ABC state.
        for req_id, plan in self.plans.items():
            if not plan.decode:
                for layer, state in enumerate(self.writer.requests[req_id]):
                    self._abc_states[req_id, layer] = state

    def _seed_replay_prefix(self, layer, owner, state, start, end):
        if start == end:
            return
        runtime = self.writer.runtimes[self.writer.groups[layer].geometry]
        handle = runtime.prepare_access_table(state.keys)
        try:
            table = runtime.access_table_view(handle, device="npu")
            destinations = self.context_destinations[start:end] - start
            destinations = destinations + owner * self.slot_capacity + self.history_capacity + start % TAIL
            get_tokens(
                runtime,
                table,
                self.context_indices[start:end],
                self.context_offsets[start:end],
                torch.tensor([end - start], dtype=torch.int32, device=self.pool.device),
                destinations,
                self.bytes_outputs[layer],
                synchronize=True,
            )
        finally:
            runtime.release_access_table(handle)

    def _seed(self, layer, owner, state, start, end, tail):
        if not tail:
            return super()._seed(layer, owner, state, start, end, tail)
        if start == end:
            return
        req_id = next(req for req, row in self.owners.items() if row == owner)
        if self._abc_states.get((req_id, layer)) is state:
            return  # Native replay/Decode has continued writing this ABC ring.
        received = self._received_tails.get((req_id, layer))
        if (
            received is None
            or received[0] is not state
            or received[2] != end
            or start != end // BLOCK_SIZE * BLOCK_SIZE
        ):
            raise RuntimeError("first Decode requires the matching completed PD tail")
        # Hot/ABC addressing is confined to this Backend, at computation time.
        # start is block aligned and TAIL is three blocks, so this partial page
        # cannot cross the ring end. Copy raw bytes without dtype conversion.
        first = owner * self.slot_capacity + self.history_capacity + start % TAIL
        source = received[1].view(torch.int8).reshape(-1)[: (end - start) * TOKEN_BYTES]
        self.hot[layer].view(-1, TOKEN_BYTES)[first : first + end - start].copy_(source.view(-1, TOKEN_BYTES))
        self._abc_states[req_id, layer] = state

    def release(self, req_id):
        # Worker must drain/resolve DMA before invoking the native release path.
        super().release(req_id)
        for key in tuple(self._abc_states):
            if key[0] == req_id:
                del self._abc_states[key]
        for key in tuple(self._received_tails):
            if key[0] == req_id:
                del self._received_tails[key]
        with self._receive_lock:
            slot = self.receive_slots.pop(req_id, None)
            if slot is not None:
                self._receive_free.append(slot)
            for key, (owner, _) in tuple(self.receive_blocks.items()):
                if owner == req_id:
                    self.receive_blocks.pop(key)


def create_decode_cache(
    layer_names, max_requests, max_tokens, max_model_len, device, *, num_spec_tokens=3, hot_tokens=HISTORY, native=None
):
    """Allocate before memory profiling, using the native state/shape budget.

    max_requests excludes the native dummy row. layer_names are actual Target
    owners from resolved cache specs, with draft/indexer-only layers excluded.
    """
    names = tuple(layer_names)
    if not names or len(names) != len(set(names)) or min(max_requests, max_tokens, max_model_len) <= 0:
        raise ValueError("PD requires distinct layer owners and positive cache capacities")
    if num_spec_tokens != 3:
        raise ValueError("this PD design requires the confirmed 3-token MTP configuration")
    tokens = (max_model_len + num_spec_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    tokens = max(max(TOPK, tokens) + TOPK, hot_tokens + TAIL)
    pool = DeviceCacheState(names, max_requests + 1, tokens, device)
    pool.groups = tuple((name,) for name in names)
    pool.policy = PDCachePolicy(pool, max_tokens, max_model_len, native=native, hot_tokens=hot_tokens)
    # Charge the compact ordinary receiver to native profiling, independently
    # of Hot. Runner replaces this reservation with the final layer arenas.
    pool.storage["pd_receive_reservation"] = torch.empty(
        (len(names), max_requests, BLOCK_SIZE, TOKEN_BYTES), dtype=torch.int8, device=device
    )
    return pool
