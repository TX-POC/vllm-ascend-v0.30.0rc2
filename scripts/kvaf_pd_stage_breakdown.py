# SPDX-License-Identifier: Apache-2.0
"""Offline request-by-stage matrix and across-request stage distributions."""

from collections import defaultdict

MAX_MATRIX_CELLS = 200000
MAX_MATRIX_REQUESTS = 10000
# Main values use one named observation scope per business stage. Related spans
# remain drill-down evidence; nested HTTP/remote waits must not be added twice.
PHASES = (
    (
        "prepare",
        "请求准备与信息交互",
        ("D.block_bind_and_advertise",),
        ("D.block_bind", "D.metaserver", "D.seed", "Proxy."),
        "主值为 D block 绑定/通告回调区间；metaserver/Proxy 的等待跨度另列，可能包含 P 及传输等待。",
    ),
    (
        "prefill",
        "P 计算与发送准备",
        ("P.target_host_forward", "P.mtp_host_propose", "P.production_wait", "P.send_queue_wait"),
        ("P.",),
        "P Host 计算、生产等待和发送排队区间取并集；不是纯设备计算。",
    ),
    (
        "transfer",
        "PD Connector 与两条传输路径",
        (
            "connector.plan",
            "connector.pull.cpu",
            "connector.pull.npu",
            "connector.pull_pair.cpu",
            "connector.pull_pair.npu",
            "connector.complete",
            "connector.status",
            "PullRPC.cpu.chunk_envelope",
            "PullRPC.npu.chunk_envelope",
        ),
        ("connector.", "PullRPC.", "D.control.", "D.cpu", "D.npu", "D.runtime_lock", "D.compute_dependency"),
        "P 端已观察控制调用区间取并集；完整 P→D/P→K DDR 传输仍单独校验，不能代替纯 DMA 或完整传输时延。",
    ),
    (
        "decode",
        "D 接收完成与开始解码",
        ("D.gate_to_decode_host", "D.decode_batch_wait"),
        ("D.remote_kv_wait", "D.gate_to", "D.decode_batch", "D.target", "D.mtp"),
        "D KV 门控就绪到首轮 Host Decode 的等待，包含已启用的组批等待；后续 Decode 计算另列。",
    ),
)


def build_breakdown(intervals, identities, distribution, union_ms, quality, valid_keys, cancelled):
    definitions = [
        dict(id=key, label=label, primary_stages=list(names), detail_prefixes=list(prefixes), basis=basis)
        for key, label, names, prefixes, basis in PHASES
    ]
    names_to_phase = {name: key for key, _, names, _, _ in PHASES for name in names}
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    micro = defaultdict(list)
    matrix_available = (
        len(intervals) + len(identities) * len(PHASES) <= MAX_MATRIX_CELLS and len(identities) <= MAX_MATRIX_REQUESTS
    )
    for (domain, name, request, generation), spans in sorted(intervals.items()):
        identity = request, generation
        phase = names_to_phase.get(name)
        if phase:
            groups[identity][phase][domain].extend(spans)
        if matrix_available:
            micro[identity].append(
                dict(name=name, clock_domain=domain, observed_ms=union_ms(spans), observed_intervals=len(spans))
            )
    values = defaultdict(list)
    rows = []
    for identity in sorted(identities):
        cells = {}
        for key, *_ in PHASES:
            domains = groups[identity][key]
            value = union_ms(next(iter(domains.values()))) if len(domains) == 1 else None
            cells[key] = dict(
                observed_ms=value,
                clock_domains=sorted(domains),
                reason="observed intervals only"
                if value is not None
                else "multiple unaligned clock domains"
                if domains
                else "not observed",
                observed_only=True,
            )
            if value is not None:
                values[key].append(value)
        if matrix_available:
            rows.append(
                dict(
                    request=identity[0],
                    generation=identity[1],
                    phases=cells,
                    stages=micro[identity],
                    quality=sorted(quality.get(identity, ())),
                    kv_status="confirmed"
                    if identity in valid_keys
                    else "cancelled"
                    if identity in cancelled
                    else "unverified",
                    client=None,
                )
            )
    return dict(
        schema=1,
        phases=definitions,
        aggregation="one same-clock interval union per request/stage, then distributions across requests",
        display_axis="business-order statistical concatenation; not a measured global timeline or E2E decomposition",
        observed_requests=len(identities),
        summary=[dict(id=key, label=label, **distribution(values[key])) for key, label, *_ in PHASES],
        requests=rows,
        matrix_available=matrix_available,
        matrix_unavailable=None
        if matrix_available
        else "request-stage matrix exceeds offline display budget; aggregate statistics remain complete",
    )


def attach_clients(breakdown, client):
    metrics = {(row["request"], row["generation"]): row for row in client.get("request_metrics", [])}
    for row in breakdown["requests"]:
        row["client"] = metrics.get((row["request"], row["generation"]))
        if row["client"] is not None and row["client"]["status"] not in ("ok", "success"):
            row["quality"].append("client request failed; stage values are observed intervals only")
