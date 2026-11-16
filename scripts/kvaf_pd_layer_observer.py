# SPDX-License-Identifier: Apache-2.0
"""Optional eager P layer Host calls; no device events, sync or tensor reads."""

import threading
import weakref

from vllm_ascend.distributed.kv_transfer.kvaf.pd.observation import notify

MAX_HOOKED_LAYERS = 1024


class LayerObserver:
    def __init__(self, events, sender):
        self.events, self.sender = weakref.ref(events), sender
        self.local = threading.local()
        self.handles = []

    def scope(self, event, fields, token):
        if event == "begin" and fields.get("name") == "P.target_host_forward":
            aliases = fields.get("context", {}).get("request_ids", ()) if token else ()
            if token:
                token["previous_compute_aliases"] = getattr(self.local, "aliases", ())
            self.local.aliases = aliases
        elif event == "end" and (fields.get("token") or {}).get("name") == "P.target_host_forward":
            self.local.aliases = fields["token"].get("previous_compute_aliases", ())

    def bind(self, model):
        sender = self.sender()
        layers = {
            name.split(".self_attn.", 1)[0]: name
            for name, memory in sender.memories.items()
            if memory.offload_main and ".self_attn." in name
        }
        if len(layers) > MAX_HOOKED_LAYERS:
            return
        try:
            for name, module in model.named_modules():
                if name not in layers:
                    continue
                layer = layers[name]

                def before(module, args, layer=layer):
                    self.begin(layer)

                self.handles.append(module.register_forward_pre_hook(before))
                self.handles.append(module.register_forward_hook(self.end, always_call=True))
        except Exception:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            events = self.events()
            if events:
                events.capture.fail("optional layer Host observation unavailable; native model retained")

    def begin(self, layer):
        token = None
        events = self.events()
        try:
            aliases = getattr(self.local, "aliases", ())
            sender = self.sender()
            if events and not events.capture.stop.is_set() and aliases and sender and sender.metadata:
                requests = tuple(
                    (alias, request, sender.metadata.generations[alias])
                    for alias, request in sender.metadata.requests.items()
                    if alias in aliases
                )
                token = notify(
                    events, "begin", name="P.layer_host_forward", context=dict(layer=layer, requests=requests)
                )
        except Exception:
            if events:
                events.capture.fail("layer Host observation failed; native model retained")
        self.local.token = token

    def end(self, module, args, output):
        events = self.events()
        token = getattr(self.local, "token", None)
        self.local.token = None
        if events and token is not None:
            notify(events, "end", token=token, success=output is not None)
