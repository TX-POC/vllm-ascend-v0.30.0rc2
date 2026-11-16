# SPDX-License-Identifier: Apache-2.0
"""Exact identities and known program order; never estimate cross-host latency."""

from collections import defaultdict


def members(event):
    return event.get("members", [event])


def build_links(events, limit=200000):
    controls = defaultdict(lambda: defaultdict(list))
    indexes = {}
    links, seen = [], set()

    def add(left, right, relation, kind):
        if left == right or (left, right, relation) in seen:
            return
        first, last = events[left], events[right]
        aligned = first["domain"] == last["domain"]
        if kind == "program_order" and (
            not aligned
            or first["start_ns"] > last["start_ns"]
            or first.get("success") is not True
            or last.get("success") is not True
        ):
            return
        if len(links) >= limit:
            raise ValueError("Trace relation budget exceeded; select fewer requests")
        seen.add((left, right, relation))
        links.append(dict(source=str(left), target=str(right), relation=relation, kind=kind, aligned=aligned))

    def index(name, fields):
        key = name, fields
        if key not in indexes:
            table = defaultdict(list)
            for number, event in enumerate(events):
                if event["name"] != name:
                    continue
                for member in members(event):
                    if member.get("association") != "unresolved":
                        values = dict(event, **member)
                        identity = tuple(values.get(field) for field in fields)
                        table[identity].append(number)
            indexes[key] = table
        return indexes[key]

    def pair(begin, end, fields, relation, kind="program_order"):
        first, last = index(begin, fields), index(end, fields)
        for identity, left in first.items():
            right = last.get(identity, ())
            if len(left) == len(right) == 1:
                add(left[0], right[0], relation, kind)

    for number, event in enumerate(events):
        if event.get("job") and event.get("control_op") and event.get("role") in ("P", "D_worker"):
            identity = tuple(
                event.get(field) for field in ("request", "generation", "job", "control_op", "path", "batch")
            )
            controls[identity][event["role"]].append(number)
    for sides in controls.values():
        if len(sides["P"]) == len(sides["D_worker"]) == 1:
            add(sides["P"][0], sides["D_worker"][0], "exact control job/op/path/batch", "control")

    request = ("request", "generation")
    local_request = (*request, "domain")
    proxy_request = (*local_request, "actor", "parent_request")
    chunk = (*local_request, "actor", "layer", "first", "tokens")
    transfer = (*chunk, "member", "path", "batch")
    pair("P_SEND_QUEUED", "P_SEND_STARTED", chunk, "native sender task dispatch")
    pair("TRANSFER_ELIGIBLE", "connector.plan", (*chunk, "member", "peer"), "native source eligibility to plan")
    pair("D_REQUEST_BLOCKS_READY", "RECEIVE_GATE_READY", local_request, "native request receive gate")
    pair("RECEIVE_GATE_READY", "D_DECODE_STARTED", local_request, "native gate to first host Decode entry")
    pair("D.control.pull.npu", "HBM_PULL_DONE", transfer, "native READ completion within Pull")
    pair("D.control.pull.cpu", "CPU_PULL_DONE", transfer, "native CPU READ completion within Pull")
    pair("CPU_PULL_DONE", "DDR_COMMITTED", transfer, "native CPU transaction commit")
    pair("DDR_COMMITTED", "DDR_PUBLISHED", transfer, "native readable DDR publication")
    for begin, end, relation in (
        ("PROXY_HANDLER_ENTER", "PROXY_DECODER_SELECTED", "native Proxy routing preparation"),
        ("PROXY_DECODER_SELECTED", "PROXY_DECODE_HTTP_DISPATCHED", "native decoder dispatch"),
        ("PROXY_PREFILL_TASK_QUEUED", "PROXY_PREFILL_TASK_STARTED", "native prefill task scheduling"),
        ("PROXY_PREFILL_TASK_STARTED", "PROXY_PREFILL_HTTP_DISPATCHED", "native prefiller dispatch"),
    ):
        pair(begin, end, proxy_request, relation)
    pair("D_REQUEST_BLOCKS_READY", "PROXY_METADATA_RECEIVED", request, "exact request metadata handoff", "handoff")
    pair("PROXY_DECODE_HTTP_DISPATCHED", "D_REQUEST_BLOCKS_READY", request, "exact decoder request dispatch", "handoff")
    pair("CLIENT_REQUEST_SENT", "PROXY_HANDLER_ENTER", request, "exact client/Proxy request association", "identity")
    pair("D_DECODE_STARTED", "CLIENT_FIRST_OUTPUT", request, "exact Decode/client request association", "identity")
    return links
