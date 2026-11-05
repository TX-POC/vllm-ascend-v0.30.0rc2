# SPDX-License-Identifier: Apache-2.0
"""Native PD rendezvous and allocation, with all-group MTP ownership retained."""

from dataclasses import dataclass, field
from uuid import uuid4

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler import (
    SFAPDRD2HProducerScheduler,
    SFAPDRD2HScheduler,
)


def native_draft_limit(vllm_config):
    """Match the native Runner's effective drafter context limit."""
    draft_config = vllm_config.speculative_config.draft_model_config
    if draft_config is not None and draft_config.max_model_len is not None:
        return draft_config.max_model_len
    return vllm_config.model_config.max_model_len


@dataclass(frozen=True)
class ReceiveRequest:
    generation: str
    tokens: int
    blocks: tuple[tuple[int, ...], ...]


@dataclass
class ReceiveMetadata(KVConnectorMetadata):
    requests: dict[str, ReceiveRequest] = field(default_factory=dict)


class ProducerScheduler(SFAPDRD2HProducerScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pd_owners = set()
        self._pd_cancellations = {}
        self._pd_draft_limit = native_draft_limit(self.vllm_config)
        self._pd_num_speculative_tokens = self.vllm_config.speculative_config.num_speculative_tokens
        self._pd_admitted_classes = {}
        self._pd_sent_tokens = {}

    def _requires_draft(self, request):
        if request.max_tokens != 1:
            raise ValueError("PD P requires native proxy prefill with max_tokens=1")
        # Use the complete prompt even during chunked prefill: admitting a
        # near-limit request beside a short one can make native batch-wide
        # skip_drafting leave the short request without its MTP prefix.
        return len(request.prompt_token_ids) + self._pd_num_speculative_tokens <= self._pd_draft_limit

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        requires_draft = self._requires_draft(request)
        if self._pd_admitted_classes and requires_draft != next(iter(self._pd_admitted_classes.values())):
            return None, False
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        requires_draft = self._requires_draft(request)
        if self._pd_admitted_classes and requires_draft != next(iter(self._pd_admitted_classes.values())):
            raise ValueError("PD P cannot admit mixed native MTP drafting classes")
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if request.request_id in self._reqs_need_send_layerwise:
            params = request.kv_transfer_params
            if not params.get("kvaf_generation") or params.get("remote_cached_tokens", 0):
                raise ValueError("PD KVAF requires a D generation and no D native cached prefix")
            previous = self._pd_sent_tokens.get(request.request_id)
            if previous is not None and params["kvaf_generation"] != previous[0]:
                raise ValueError("PD P cannot change the generation of an already transferred request")
            self._pd_sent_tokens.setdefault(request.request_id, (params["kvaf_generation"], 0))
            self._pd_owners.add(request.request_id)
        self._pd_admitted_classes[request.request_id] = requires_draft

    def build_connector_meta(self, scheduler_output):
        generations = {
            req_id: info.request.kv_transfer_params["kvaf_generation"]
            for req_id, info in self._reqs_need_send_layerwise.items()
        }
        metadata = super().build_connector_meta(scheduler_output)
        for req_id, request in list(metadata.requests.items()):
            generation, sent = self._pd_sent_tokens.get(req_id, (generations[req_id], 0))
            first = max(request.local_transed_tokens, sent)
            complete = request.local_computed_tokens // min(self.block_size) * min(self.block_size)
            if not request.chunk_finish and complete <= first:
                # Native preemption resets its tracker and P's physical block
                # table. Keep D's committed prefix while P recomputes it.
                del metadata.requests[req_id]
                continue
            request.local_transed_tokens = first
            self._pd_sent_tokens[req_id] = generation, max(sent, complete)
        metadata.generations = {req_id: generations[req_id] for req_id in metadata.requests}
        metadata.cancellations, self._pd_cancellations = self._pd_cancellations, {}
        return metadata

    def request_finished_all_groups(self, request, block_ids):
        if request.request_id in self._reqs_need_send_layerwise:
            self._pd_cancellations[request.request_id] = dict(request.kv_transfer_params)
        super().request_finished_all_groups(request, block_ids)
        # vLLM keeps P's physical blocks until every rank reports send complete.
        hold = request.request_id in self._pd_owners
        self._pd_owners.discard(request.request_id)
        self._pd_admitted_classes.pop(request.request_id, None)
        self._pd_sent_tokens.pop(request.request_id, None)
        return hold, None


class ConsumerScheduler(SFAPDRD2HScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pd_requests = {}

    @staticmethod
    def _try_create_copy_sfa_slot_allocator(vllm_config):
        return None

    @staticmethod
    def _find_dspark_context_groups():
        return ()

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if (
            request.request_id not in self._pd_requests
            and len(self._pd_requests) >= self.vllm_config.scheduler_config.max_num_seqs
        ):
            return None, False
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        params = request.kv_transfer_params
        if params and params.get("do_remote_prefill"):
            if params.get("do_virtual"):
                raise ValueError("PD KVAF requires the native proxy rendezvous")
            seed = ReceiveRequest(
                uuid4().hex,
                len(request.prompt_token_ids),
                tuple(tuple(group) for group in blocks.get_block_ids()),
            )
            self._pd_requests[request.request_id] = seed
            # Reuse the native deferred-cancellation lifecycle for *all* PD
            # receives, including ordinary MTP. No DSpark protocol is enabled.
            self._dspark_pending_recv.add(request.request_id)
        return super().update_state_after_alloc(request, blocks, num_external_tokens)

    def _submit_metaserver_request(self, *, request_id, url, message):
        if request_id not in self._pd_requests:
            return  # A cancelled native HTTP retry cannot recreate its allocation.
        message = dict(message, kvaf_generation=self._pd_requests[request_id].generation)
        return super()._submit_metaserver_request(request_id=request_id, url=url, message=message)

    def build_connector_meta(self, scheduler_output):
        metadata = ReceiveMetadata({req_id: self._pd_requests[req_id] for req_id in self._reqs_need_recv})
        self._reqs_need_recv.clear()
        return metadata

    def _cleanup_finished_request(self, request_id):
        self._pd_requests.pop(request_id, None)
        return super()._cleanup_finished_request(request_id)
