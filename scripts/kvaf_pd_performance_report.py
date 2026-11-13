# SPDX-License-Identifier: Apache-2.0
"""Read-only terminal analysis and single-file trace export, standard library only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

MAX_LINE_BYTES = 1048576
MAX_FILES = 1024
CONTEXT = ("request", "generation", "layer", "first", "tokens", "final", "member", "ratio", "peer")


def snapshots(inputs, max_bytes, max_records):
    """Freeze each file at its last complete checkpoint, without flushing writers."""
    paths = set()
    for value in inputs:
        value = Path(value)
        if not value.exists():
            raise ValueError(f"capture path does not exist: {value}")
        for path in value.rglob("*.jsonl") if value.is_dir() else (value,):
            paths.add(path.resolve())
            if len(paths) > MAX_FILES:
                raise ValueError("too many capture files; select one run")
    remaining, total_records = max_bytes, 0
    parts, fingerprints, notes = {}, set(), []
    for path in sorted(paths):
        header, checkpoint, records, committed, signature = None, None, [], 0, None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            boundary = os.fstat(stream.fileno()).st_size
            if boundary > remaining:
                raise ValueError("input byte budget exceeded; select a smaller run or increase --max-input-bytes")
            remaining -= boundary
            while stream.tell() < boundary:
                raw = stream.readline(min(MAX_LINE_BYTES + 1, boundary - stream.tell()))
                if len(raw) > MAX_LINE_BYTES:
                    raise ValueError(f"{path.name}: record exceeds 1 MiB")
                if not raw.endswith(b"\n"):
                    notes.append(f"{path.name}: uncommitted tail excluded")
                    break
                try:
                    row = json.loads(raw)
                except ValueError as error:
                    raise ValueError(f"{path.name}: malformed complete JSON record") from error
                if not isinstance(row, dict):
                    raise ValueError(f"{path.name}: record must be an object")
                digest.update(raw)
                kind = row.get("kind")
                if header is None:
                    if kind != "header" or row.get("schema") not in (1, 2, 3):
                        raise ValueError(f"{path.name}: unsupported/missing capture header")
                    header = row
                elif kind == "header":
                    raise ValueError(f"{path.name}: repeated header")
                elif kind == "checkpoint":
                    checkpoint = dict(row, committed_file_bytes=stream.tell())
                    committed, signature = len(records), digest.hexdigest()
                else:
                    total_records += 1
                    if total_records > max_records:
                        raise ValueError("record budget exceeded; select fewer requests/files")
                    records.append(row)
        if checkpoint is None:
            notes.append(f"{path.name}: no committed checkpoint yet; this file excluded")
            continue
        if committed != len(records):
            notes.append(f"{path.name}: records after last checkpoint excluded")
        if signature in fingerprints:
            continue  # A byte-identical copied snapshot is not another contributor.
        fingerprints.add(signature)
        identity = header.get("capture_id", signature)
        part = header.get("part", 0)
        if type(part) is not int or part < 0:
            raise ValueError("invalid capture part")
        key = identity, part
        if key in parts:
            raise ValueError("conflicting copies of the same capture part")
        parts[key] = (header, checkpoint, records[:committed], path.name)
    grouped = defaultdict(list)
    for (identity, part), capture in parts.items():
        grouped[identity].append((part, capture))
    result = []
    for identity, fragments in grouped.items():
        fragments.sort()
        if [part for part, _ in fragments] != list(range(len(fragments))):
            raise ValueError("missing capture shard; supply the entire run directory")
        headers = [capture[0] for _, capture in fragments]
        if any(
            {k: v for k, v in header.items() if k not in ("part", "created_ns")}
            != {k: v for k, v in headers[0].items() if k not in ("part", "created_ns")}
            for header in headers
        ):
            raise ValueError("capture shard identity/config changed")
        rows = [row for _, capture in fragments for row in capture[2]]
        checkpoint = dict(
            fragments[-1][1][1],
            committed_file_bytes=sum(capture[1]["committed_file_bytes"] for _, capture in fragments),
        )
        for field in ("written", "enqueued", "dropped", "errors"):
            if type(checkpoint.get(field)) is not int or checkpoint[field] < 0:
                raise ValueError("invalid capture checkpoint counters")
        if checkpoint["written"] != len(rows) or checkpoint["enqueued"] < checkpoint["written"]:
            raise ValueError("checkpoint/data mismatch or missing capture shards")
        result.append(
            (dict(headers[0], capture_id=identity), checkpoint, rows, ",".join(capture[3] for _, capture in fragments))
        )
    if not result:
        raise ValueError("no committed capture snapshots; check the path or wait for the configured flush interval")
    return result, notes


def _normalize(header, records, *, eligible=False):
    """Translate schema 2/3 jobs into the validated layer/chunk contract."""
    if header["schema"] == 1:
        return records
    milestones = defaultdict(list)
    if eligible:
        for row in records:
            if row.get("kind") == "milestone" and row.get("name") == "TRANSFER_ELIGIBLE":
                milestones[tuple(row.get(key) for key in CONTEXT)].append(row["time_ns"])
    layers, jobs, cancelled = {}, defaultdict(list), []
    for row in records:
        if row["kind"] == "cancel":
            cancelled.append(dict(kind="cancel", request=row["request"], generation=row["generation"]))
        elif row["kind"] in ("plan", "pull", "complete", "status") and row.get("job"):
            jobs[row["job"]].append(row)
    for events in jobs.values():
        plans = [row for row in events if row["kind"] == "plan" and row["success"] is True]
        if len(plans) != 1:
            raise ValueError("missing or duplicate job plan")
        plan = plans[0]
        key = plan["request"], plan["generation"], plan["layer"], plan["member"]
        layer = layers.setdefault(
            key,
            dict(
                kind="layer",
                **{key: plan[key] for key in ("request", "generation", "layer", "peer", "member", "ratio")},
                chunks=[],
                errors=[],
                input_tokens=0,
            ),
        )
        errors = layer["errors"]
        if any(any(row.get(key) != plan.get(key) for key in CONTEXT) for row in events):
            errors.append("job context changed")
        done = any(
            row["kind"] == "complete"
            and row["success"] is True
            or row["kind"] == "status"
            and row.get("done")
            and row["success"] is True
            for row in events
        )
        if not done:
            errors.append("job has no confirmed completion")
        spans = {}
        for path in ("npu", "cpu"):
            pulls = [row for row in events if row["kind"] == "pull" and row.get("path") == path]
            expected = plan["count"][path]
            seen = [row["batch"] for row in pulls]
            if (
                type(expected) is not int
                or not 0 <= expected <= len(records)
                or len(seen) != expected
                or any(type(batch) is not int for batch in seen)
                or len(set(seen)) != expected
                or (seen and (min(seen) != 0 or max(seen) != expected - 1))
                or any(row["success"] is not True for row in pulls)
            ):
                errors.append(f"{path}: failed, duplicate or missing Pull ACK")
            spans[path] = (
                [min(row["start_ns"] for row in pulls), max(row["end_ns"] for row in pulls)] if pulls else None
            )
            if eligible and pulls:
                starts = milestones.get(tuple(plan.get(key) for key in CONTEXT), [])
                if len(starts) != 1 or not 0 < starts[0] <= plan["start_ns"] <= spans[path][0]:
                    errors.append("missing/ambiguous or late TRANSFER_ELIGIBLE")
                else:
                    spans[path][0] = starts[0]
        layer["chunks"].append(dict(first=plan["first"], tokens=plan["tokens"], final=plan["final"], spans=spans))
        if plan["final"]:
            layer["input_tokens"] = plan["tokens"]
    return [*layers.values(), *cancelled]


def event_members(event):
    if event.get("members"):
        return [(member["request"], member["generation"]) for member in event["members"]]
    if event.get("request") is not None:
        return [(event["request"], event["generation"])]
    return []


def _resolve_proxy_generations(captures, notes):
    generations = defaultdict(set)
    for _, _, rows, _ in captures:
        for row in rows:
            if row.get("request") and isinstance(row.get("generation"), str):
                generations[row["request"]].add(row["generation"])
    result = []
    for header, checkpoint, rows, source in captures:
        if header["role"] == "Proxy":
            resolved, unknown = [], set()
            for row in rows:
                if row.get("request") and row.get("generation") is None:
                    choices = generations[row["request"]]
                    unique = len(choices) == 1
                    row = dict(
                        row,
                        generation=next(iter(choices)) if unique else "unresolved:" + header["capture_id"],
                        association="exact ID and unique observed generation" if unique else "unresolved",
                    )
                    if not unique:
                        unknown.add(row["request"])
                resolved.append(row)
            if unknown:
                notes.append(
                    f"{source}: {len(unknown)} Proxy IDs have unknown/ambiguous generations; local evidence only"
                )
            rows = resolved
        result.append((header, checkpoint, rows, source))
    return result


def _select_rows(rows, request_ids, length, known_lengths):
    selected = [
        row
        for row in rows
        if row.get("request") is not None
        and (not request_ids or row["request"] in request_ids)
        and (length is None or known_lengths.get((row["request"], row.get("generation"))) == length)
    ]
    batches = {row["batch_id"] for row in selected if row["kind"] == "membership"}
    return [
        *selected,
        *(row for row in rows if row["kind"] in ("batch_span", "batch_begin") and row["batch_id"] in batches),
    ]


def _events(header, rows, limit):
    identity = [header.get("host"), header.get("boot_id")]
    if not all(identity):
        identity = ["unknown host/boot", header["capture_id"]]
    domain = json.dumps([*identity, "perf_counter_ns"], separators=(",", ":"))
    actor = f"{header.get('host', '?')}/TP{header.get('tp_rank', '?')}/{header['capture_id']}"
    memberships = defaultdict(list)
    ended = {
        row.get("span_id") or row.get("batch_id")
        for row in rows
        if row["kind"] not in ("rpc_begin", "phase_begin", "batch_begin", "membership") and "end_ns" in row
    }
    for row in rows:
        if row["kind"] == "membership":
            memberships[row["batch_id"]].append(
                {
                    key: row[key]
                    for key in (
                        "request",
                        "generation",
                        "layer",
                        "first",
                        "tokens",
                        "association",
                        "parent_request",
                        "request_phase",
                    )
                    if key in row
                }
            )
    result = []
    for row in rows:
        if row["kind"] == "membership":
            continue
        if row["kind"] in ("rpc_begin", "phase_begin", "batch_begin"):
            if (row.get("span_id") or row.get("batch_id")) in ended:
                continue
            row = dict(
                row,
                kind="batch_span" if row["kind"] == "batch_begin" else "phase",
                end_ns=row["start_ns"],
                instant=True,
                unfinished=True,
                success=False,
            )
        if row["kind"] in ("milestone", "batch_span"):
            item = {
                key: row[key]
                for key in (
                    "request",
                    "generation",
                    "layer",
                    "first",
                    "tokens",
                    "member",
                    "ratio",
                    "path",
                    "batch",
                    "batch_id",
                    "batch_size",
                    "name",
                    "success",
                    "observation",
                    "unfinished",
                    "association",
                    "parent_request",
                    "request_phase",
                )
                if key in row
            }
            stamp = row["time_ns"] if row["kind"] == "milestone" else row["start_ns"]
            item.update(
                start_ns=stamp,
                end_ns=row.get("end_ns", stamp),
                domain=domain,
                actor=actor,
                instant=row["kind"] == "milestone" or row.get("instant", False),
            )
            if row["kind"] == "batch_span":
                item.update(
                    members=memberships[row["batch_id"]],
                    shared_batch=True,
                    selected_members=len(memberships[row["batch_id"]]),
                )
            result.append(item)
            if len(result) > limit:
                raise ValueError("event budget exceeded; reduce input scope")
            continue
        if row["kind"] == "layer":
            for chunk in row["chunks"]:
                for path, interval in chunk["spans"].items():
                    if interval:
                        result.append(
                            dict(
                                request=row["request"],
                                generation=row["generation"],
                                layer=row["layer"],
                                member=row["member"],
                                peer=row["peer"],
                                first=chunk["first"],
                                name=f"PullRPC.{path}.chunk_envelope",
                                request_phase=row.get("request_phase", "unknown"),
                                path=path,
                                success=not row["errors"],
                                start_ns=interval[0],
                                end_ns=interval[1],
                                domain=domain,
                                actor=actor,
                            )
                        )
                        if len(result) > limit:
                            raise ValueError("event budget exceeded; reduce input scope")
        elif "start_ns" in row:
            result.append(
                dict(
                    {
                        key: row[key]
                        for key in (
                            "request",
                            "generation",
                            "layer",
                            "member",
                            "peer",
                            "first",
                            "path",
                            "success",
                            "job",
                            "span_id",
                            "batch_size",
                            "batch",
                            "instant",
                            "unfinished",
                            "tokens",
                            "final",
                            "ratio",
                            "request_phase",
                        )
                        if key in row
                    },
                    name=row.get("name", row.get("stage", "connector." + row["kind"]))
                    + (
                        "." + row["path"]
                        if row.get("path")
                        and (row["kind"] == "pull" or row.get("name") in ("connector.pull", "D.control.pull"))
                        else ""
                    ),
                    start_ns=row["start_ns"],
                    end_ns=row["end_ns"],
                    domain=domain,
                    actor=actor,
                )
            )
            if len(result) > limit:
                raise ValueError("event budget exceeded; reduce input scope")
    for event in result:
        event["role"] = header["role"]
        if event["name"].startswith("connector."):
            event["control_op"] = event["name"].split(".")[1]
        elif event["name"].startswith("D.control."):
            event["control_op"] = event["name"].split(".")[2]
        if event.get("layer"):
            event["layer_role"] = header.get("layer_roles", {}).get(event["layer"], "unknown")
        if (
            type(event["start_ns"]) is not int
            or type(event["end_ns"]) is not int
            or not 0 < event["start_ns"] <= event["end_ns"] < 2**63
        ):
            raise ValueError("invalid host timing interval")
    return result


def _wait_intervals(events):
    points = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event.get("instant"):
            for request, generation in event_members(event):
                key = (
                    event["domain"],
                    event["actor"] if event["name"].startswith(("P_", "PROXY_")) else None,
                    request,
                    generation,
                    event.get("layer"),
                    event.get("first"),
                    event.get("tokens"),
                )
                points[key][event["name"]].append(event)
    derived = []
    for markers in points.values():
        for begin, end, name in (
            ("P_SEND_QUEUED", "P_SEND_STARTED", "P.send_queue_wait"),
            ("D_REQUEST_BLOCKS_READY", "RECEIVE_GATE_READY", "D.remote_kv_wait"),
            ("RECEIVE_GATE_READY", "D_DECODE_STARTED", "D.gate_to_decode_host"),
            ("PROXY_HANDLER_ENTER", "PROXY_DECODER_SELECTED", "Proxy.route_and_prepare"),
            ("PROXY_DECODER_SELECTED", "PROXY_DECODE_HTTP_DISPATCHED", "Proxy.dispatch_wait"),
            ("PROXY_PREFILL_TASK_QUEUED", "PROXY_PREFILL_TASK_STARTED", "Proxy.prefill_task_queue"),
        ):
            first, last = markers.get(begin, []), markers.get(end, [])
            if len(first) == len(last) == 1 and last[0]["start_ns"] >= first[0]["start_ns"]:
                derived.append(
                    dict(
                        first[0],
                        name=name,
                        instant=False,
                        end_ns=last[0]["start_ns"],
                        observation="matched milestones in same host/boot clock domain",
                    )
                )
    return derived


def _statistics(api, layers, requests, basis):
    return [
        dict(metric=metric, scope=scope, basis=basis, **api.distribution([row[f"{metric}_ms"] for row in samples]))
        for metric in ("t1", "t2")
        for scope, samples in (("request_layer", layers), ("request", requests))
    ]


def _representatives(api, requests):
    result = []
    for metric in ("t1", "t2"):
        rows = [row for row in requests if row[f"{metric}_ms"] is not None]
        values = sorted(row[f"{metric}_ms"] for row in rows)
        if not values:
            continue
        for quantile in (0.5, 0.9, 0.99):
            target = api.percentile(values, quantile)
            actual = min(rows, key=lambda row: (abs(row[f"{metric}_ms"] - target), row["request"], row["generation"]))
            result.append(
                dict(
                    metric=metric,
                    quantile=quantile,
                    target_ms=target,
                    actual_ms=actual[f"{metric}_ms"],
                    request=actual["request"],
                    generation=actual["generation"],
                )
            )
    return result


def _union_ms(intervals):
    end, elapsed = -1, 0
    for left, right in sorted(intervals):
        elapsed += max(0, right - max(left, end))
        end = max(end, right)
    return elapsed / 1e6


def _role_statistics(api, layers, roles, basis):
    result = []
    for role in sorted({roles.get(row["layer"], "unknown") for row in layers}):
        selected = [row for row in layers if roles.get(row["layer"], "unknown") == role]
        grouped = defaultdict(list)
        for row in selected:
            grouped[row["request"], row["generation"]].append(row)
        totals = []
        for rows in grouped.values():
            total = {}
            for metric in ("t1", "t2"):
                values = [row[f"{metric}_ms"] for row in rows if row[f"{metric}_ms"] is not None]
                total[f"{metric}_ms"] = sum(values) if values else None
            totals.append(total)
        result.append(dict(role=role, summary=_statistics(api, selected, totals, basis)))
    return result


def analyze(captures, notes, args, api, *, client_rows=()):
    from kvaf_pd_request_phase import select_request_phase

    captures = _resolve_proxy_generations(captures, notes)
    desired_phase = getattr(args, "request_phase", "measure")
    captures, phase_quality = select_request_phase(captures, client_rows, desired_phase, args.request_id)
    canonical, full_canonical, events, problems, batch = [], [], [], [], {}
    if not phase_quality["selected_requests"] and desired_phase == "measure":
        problems.append(
            "no explicitly identified measurement requests; provide tagged Proxy captures or --client-records; "
            "unlabelled requests are unknown, not measurements"
        )
    lengths_by_request = {}
    contracts = set()
    for header, _, rows, _ in captures:
        if header["role"] == "P" and rows and header.get("config"):
            config = header["config"]
            contracts.add((config["sample_rate"], config["sample_seed"], config["detail_level"]))
        for row in rows:
            if args.request_id and row.get("request") not in args.request_id:
                continue
            tokens = (
                row.get("input_tokens")
                if row["kind"] == "layer"
                else (
                    row.get("tokens")
                    if (row["kind"] == "plan" and row.get("final"))
                    or row.get("name") in ("P_REQUEST_BOUND", "D_REQUEST_BLOCKS_READY", "D_RECEIVE_SEEDED")
                    else None
                )
            )
            if type(tokens) is int and tokens > 0:
                key = row["request"], row["generation"]
                if key in lengths_by_request and lengths_by_request[key] != tokens:
                    raise ValueError("request final token length changed")
                lengths_by_request[key] = tokens
    if len(contracts) > 1:
        raise ValueError("P workers use inconsistent sampling/detail configuration")
    lengths = set(lengths_by_request.values())
    if args.input_tokens is None and len(lengths) > 1:
        raise ValueError("multiple prompt lengths; specify --input-tokens to compare one cohort")
    length = args.input_tokens if args.input_tokens is not None else next(iter(lengths), None)
    for header, checkpoint, rows, source in captures:
        selected = _select_rows(rows, args.request_id, args.input_tokens, lengths_by_request)
        damaged = checkpoint["dropped"] or checkpoint["errors"] or checkpoint.get("reason")
        if damaged:
            problems.append(f"{source}: dropped/error/stopped capture; its core samples excluded")
        if (
            checkpoint["enqueued"] > checkpoint["written"]
            or checkpoint.get("pending_jobs")
            or checkpoint.get("pending_layers")
            or checkpoint.get("pending_phase_records")
            or checkpoint.get("pending_requests")
        ):
            notes.append(f"{source}: pending records/jobs; completed requests only")
        events.extend(_events(header, selected, args.max_records - len(events)))
        if header["role"] == "P" and not damaged and selected:
            normalized = _normalize(header, selected)
            canonical.append(
                (
                    header,
                    dict(
                        checkpoint, enqueued=len(normalized), written=len(normalized), pending_jobs=0, pending_layers=0
                    ),
                    normalized,
                    source,
                )
            )
        elif header["role"] == "D_scheduler":
            histogram = defaultdict(int)
            for row in selected:
                if row["kind"] == "batch_span" and row["name"] == "D_SCHEDULED":
                    histogram[str(row["batch_size"])] += 1
            batch[header["capture_id"]] = dict(
                dp_rank=header.get("dp_rank"),
                histogram=dict(histogram),
                scope="batches containing selected requests; actual size includes all real batch members",
            )
        if header["role"] == "P" and not damaged and selected and header.get("eligible_observation"):
            full = _normalize(header, selected, eligible=True)
            full_canonical.append(
                (
                    header,
                    dict(checkpoint, enqueued=len(full), written=len(full), pending_jobs=0, pending_layers=0),
                    full,
                    source,
                )
            )
    layers, requests, rejected = [], [], []
    if canonical and length is not None:
        layers, requests, rejected, _, issues = api.collect_samples(canonical, length)
        problems.extend(issues)
    rpc_summary = _statistics(api, layers, requests, "pull_rpc_compat")
    full_layers, full_requests, full_rejected = [], [], []
    if full_canonical and length is not None:
        full_layers, full_requests, full_rejected, _, issues = api.collect_samples(full_canonical, length)
        problems.extend(issues)
    full_summary = _statistics(api, full_layers, full_requests, "transfer_eligible_to_ack")
    expects_eligible = any(
        header["role"] == "P"
        and bool(rows)
        and header.get("eligible_observation")
        and header.get("config", {}).get("detail_level") != "stage"
        for header, _, rows, _ in captures
    )
    summary = full_summary if expects_eligible else rpc_summary
    basis = "transfer_eligible_to_ack" if expects_eligible else "pull_rpc_compat"
    if expects_eligible:
        layers, requests, rejected = full_layers, full_requests, full_rejected
    roles = {}
    for header, _, _, _ in canonical:
        for layer, role in header.get("layer_roles", {}).items():
            if layer in roles and roles[layer] != role:
                raise ValueError("P workers disagree on Target/MTP layer ownership")
            roles[layer] = role
    waits = _wait_intervals(events)
    if len(events) + len(waits) > args.max_records:
        raise ValueError("derived event budget exceeded; reduce input scope")
    events.extend(waits)
    intervals = defaultdict(list)
    for event in events:
        if event.get("success") is True and not event.get("instant"):
            for request, generation in event_members(event):
                intervals[event["domain"], event["name"], request, generation].append(
                    (event["start_ns"], event["end_ns"])
                )
    stages = defaultdict(list)
    for (domain, name, _, _), spans in intervals.items():
        stages[domain, name].append(_union_ms(spans))
    observed_requests = {member for event in events for member in event_members(event)}
    valid_keys = {(request["request"], request["generation"]) for request in requests}
    cancelled = {
        (event["request"], event["generation"])
        for event in events
        if event["name"] == "connector.cancel" and event.get("success") is True
    }
    report = dict(
        schema=1,
        scope="request host phases and confirmed transfers; device timing/E2E require separate evidence",
        status="INCOMPLETE"
        if problems or rejected or cancelled or any(event.get("success") is False for event in events)
        else "LOCAL_ONLY",
        input_tokens=length,
        selection=dict(
            request_ids=args.request_id,
            request_phase=desired_phase,
            input_tokens_filter=args.input_tokens,
            event_scope="all selected observed calls, including incomplete/failed requests",
            core_sample_scope="complete, validated KV requests only",
        ),
        selected_events=len(events),
        request_phase_quality=phase_quality,
        failed_events=sum(event.get("success") is False and not event.get("unfinished") for event in events),
        unfinished_events=sum(bool(event.get("unfinished")) for event in events),
        valid_requests=len(requests),
        outcomes=dict(
            observed_requests=len(observed_requests),
            confirmed_cancellations=len(cancelled),
            incomplete_or_unverified_requests=len(observed_requests - valid_keys - cancelled),
        ),
        summary=summary,
        summary_basis=basis,
        layer_role_summary=_role_statistics(api, layers, roles, basis),
        pull_rpc_summary=rpc_summary,
        eligible_summary=full_summary,
        representative_requests=_representatives(api, requests),
        rejected=rejected,
        problems=problems,
        notes=sorted(set(notes)),
        stages=[
            dict(
                name=name,
                clock_domain=domain,
                aggregation="per-request union of observed intervals; shared batches are not divided",
                completeness="named host boundaries only; device execution and unobserved stages remain unknown",
                **api.distribution(values),
            )
            for (domain, name), values in sorted(stages.items())
        ],
        unavailable={
            "client_e2e": "no correlated client send/receive evidence",
            **(
                {}
                if any(row["samples"] for row in full_summary)
                else {"transfer_eligible_to_ack": "no complete request with verified eligible/ACK evidence"}
            ),
            "P_device_compute": "no existing device timing evidence imported",
            **(
                {}
                if any(event["name"] == "D_DECODE_STARTED" for event in events)
                else {"D_decode_started": "no sampled receive gate and host model-forward entry observed"}
            ),
            "D_device_decode_started": "host model-forward entry is not a device execution timestamp",
            "D_physical_block_allocation": "observed callback runs after native block allocation",
            **(
                {}
                if any(event["name"] == "Proxy.route_and_prepare" for event in events)
                else {"proxy_routing_queue": "no exact Proxy handler/dispatch observations provided"}
            ),
            "proxy_pre_handler_wait": "no socket ingress or ASGI server queue evidence",
            "proxy_connection_pool_wait": "included in HTTP calls; no separate transport-pool timing",
        },
        d_scheduled_batches=batch,
        sources=[
            dict(
                name=source,
                capture_id=header["capture_id"],
                role=header["role"],
                host=header.get("host"),
                boot_id=header.get("boot_id"),
                engine=header.get("engine"),
                config=header.get("config"),
                checkpoint=checkpoint,
            )
            for header, checkpoint, _, source in captures
        ],
    )
    return report, events


def attach_client_records(report, events, rows, digest, args, api):
    from kvaf_pd_client_records import client_events
    from kvaf_pd_request_phase import request_phase

    known = defaultdict(set)
    for event in events:
        for member in event.get("members", [event]):
            if member.get("request") and member.get("association") != "unresolved":
                known[member["request"]].add(member["generation"])
    desired = report["selection"].get("request_phase", "measure")
    filtered = [row for row in rows if desired == "all" or request_phase(row) in (desired, "unknown")]
    client, summary = client_events(filtered, digest, known, args.request_id, api.distribution)
    summary["phase_excluded_records"] = len(rows) - len(filtered)
    for event in client:
        if desired != "all":
            event["request_phase"] = desired
    if len(events) + len(client) > args.max_records:
        raise ValueError("client/derived event budget exceeded; reduce input scope")
    report["client"] = summary
    report["outcomes"]["client_failed_requests"] = summary["failed_records"]
    report["failed_events"] += sum(event.get("success") is False for event in client)
    if summary["failed_records"]:
        report["status"] = "INCOMPLETE"
    if summary["e2e"]["samples"]:
        report["unavailable"].pop("client_e2e", None)
    else:
        report["unavailable"]["client_e2e"] = (
            "no successful client record with an exact native request/generation association"
        )
    events.extend(client)
    report["selected_events"] = len(events)


def trace_document(report, events):
    from kvaf_pd_trace_links import build_links

    domains = sorted({event["domain"] for event in events})
    reference = next((event["domain"] for event in events if event.get("role") == "P"), domains[0] if domains else None)
    if reference:
        domains = [reference, *(domain for domain in domains if domain != reference)]
    origins = {}
    for event in events:
        domain = event["domain"]
        origins[domain] = min(origins.get(domain, event["start_ns"]), event["start_ns"])
    actors, lanes, standard, local = {}, {}, [], defaultdict(list)
    indexed = {}
    for index, row in enumerate(events):
        domain = row["domain"]
        pid = actors.setdefault(row["actor"], len(actors) + 1)
        lane = row["actor"], row.get("layer", "request"), row["name"]
        tid = lanes.setdefault(lane, len(lanes) + 1)
        event = dict(
            name=row["name"],
            cat="KVAF request",
            ph="i" if row.get("instant") else "X",
            pid=pid,
            tid=tid,
            ts=(row["start_ns"] - origins[domain]) / 1000,
            dur=(row["end_ns"] - row["start_ns"]) / 1000,
            args={key: value for key, value in row.items() if key not in ("start_ns", "end_ns")},
        )
        if row.get("instant"):
            event.pop("dur")
            event["s"] = "t"
        event["args"].update(raw_start_ns=str(row["start_ns"]), raw_end_ns=str(row["end_ns"]))
        event["args"]["event_id"] = str(index)
        indexed[str(index)] = event
        (standard if domain == reference else local[domain]).append(event)
    links = build_links(events)
    for number, link in enumerate(links):
        left, right = indexed[link["source"]], indexed[link["target"]]
        if (
            link["kind"] != "identity"
            and link["aligned"]
            and left["args"]["domain"] == reference
            and left["ts"] <= right["ts"]
        ):
            identity = f"flow-{number}"
            standard.extend(
                [
                    dict(ph="s", name=link["relation"], id=identity, pid=left["pid"], tid=left["tid"], ts=left["ts"]),
                    dict(
                        ph="f",
                        bp="e",
                        name=link["relation"],
                        id=identity,
                        pid=right["pid"],
                        tid=right["tid"],
                        ts=right["ts"],
                    ),
                ]
            )
    for actor, pid in actors.items():
        standard.append(dict(ph="M", name="process_name", pid=pid, tid=0, args=dict(name=actor)))
    for (actor, layer, name), tid in lanes.items():
        standard.append(
            dict(ph="M", name="thread_name", pid=actors[actor], tid=tid, args=dict(name=f"{layer} / {name}"))
        )
    return dict(
        traceEvents=standard,
        displayTimeUnit="ms",
        kvaf_pd=dict(
            format_version=2,
            summary=report,
            reference_clock=reference,
            clocks=[
                dict(
                    domain=domain,
                    origin_ns=str(origins[domain]),
                    alignment="reference"
                    if domain == reference
                    else "unaligned local clock; do not compare cross-host offsets",
                )
                for domain in domains
            ],
            unaligned_domains=[dict(domain=domain, events=rows) for domain, rows in local.items()],
            links=links,
        ),
    )


def export_trace(path, document, max_bytes):
    path = Path(path)
    if path.exists():
        raise ValueError("Trace output already exists; choose a new filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        size = 0
        with temporary.open("xb") as stream:
            for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":")).iterencode(
                document
            ):
                data = chunk.encode("utf-8")
                size += len(data)
                if size > max_bytes:
                    raise ValueError("Trace size budget exceeded; reduce the selected data or raise --max-trace-bytes")
                stream.write(data)
        os.link(temporary, path)  # Atomic publication, refusing even a racing overwrite.
    finally:
        temporary.unlink(missing_ok=True)


def main(argv, api):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("summary", "export-trace"))
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument("--input-tokens", type=int)
    parser.add_argument(
        "--request-phase",
        choices=("measure", "warmup", "profile", "unknown", "all"),
        default="measure",
        help="default: explicitly labelled measurements only; all is an unfiltered diagnostic view",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--client-records", type=Path, help="existing JSON/JSONL request-tool output; IDs must correlate exactly"
    )
    parser.add_argument("--max-input-bytes", type=int, default=268435456)
    parser.add_argument("--max-records", type=int, default=100000)
    parser.add_argument("--max-trace-bytes", type=int, default=268435456)
    args = parser.parse_args(argv)
    try:
        if min(args.max_input_bytes, args.max_records, args.max_trace_bytes) <= 0 or (
            args.input_tokens is not None and args.input_tokens <= 0
        ):
            raise ValueError("length and resource budgets must be positive")
        if args.command == "export-trace" and args.output is None:
            raise ValueError("export-trace requires --output")
        if args.command == "summary" and args.output is not None:
            raise ValueError("summary writes only to the terminal; use export-trace for a file")
        client_rows, client_digest, client_size = [], None, 0
        if args.client_records is not None:
            from kvaf_pd_client_records import read_client_records

            client_rows, client_digest, client_size = read_client_records(args.client_records, args.max_input_bytes)
            if client_size >= args.max_input_bytes or len(client_rows) >= args.max_records:
                raise ValueError("client records exhaust the input budget; increase budget or reduce selection")
        captures, notes = snapshots(
            args.captures, args.max_input_bytes - client_size, args.max_records - len(client_rows)
        )
        report, events = analyze(captures, notes, args, api, client_rows=client_rows)
        if args.client_records is not None:
            attach_client_records(report, events, client_rows, client_digest, args, api)
        print(f"{report['status']} | scope={report['scope']} | complete KV requests={report['valid_requests']}")
        print("Request outcomes: " + json.dumps(report["outcomes"], sort_keys=True))
        print("Request phase selection: " + json.dumps(report["request_phase_quality"], sort_keys=True))
        print("Main transfer basis: " + report["summary_basis"])
        print("metric / scope                    mean ms       P90 ms       P99 ms    valid N")
        for row in report["summary"]:
            values = [
                "UNAVAILABLE" if row[field] is None else f"{row[field]:.3f}"
                for field in ("mean_ms", "p90_ms", "p99_ms")
            ]
            print(
                f"{row['metric']} / {row['scope']:25} "
                f"{values[0]:>12} {values[1]:>12} {values[2]:>12} {row['samples']:8}"
            )
        for row in report["stages"]:
            print(
                f"stage {row['name']}: mean={row['mean_ms']:.3f} P90={row['p90_ms']:.3f} "
                f"P99={row['p99_ms']:.3f} N={row['samples']} (request interval union)"
            )
        if report["summary_basis"] == "transfer_eligible_to_ack":
            print(
                "Pull RPC compatibility (excludes earlier owner/plan waiting): "
                + json.dumps(report["pull_rpc_summary"])
            )
        if report.get("client"):
            print("Client E2E: " + json.dumps(report["client"]["e2e"]))
            print("Client TTFT: " + json.dumps(report["client"]["ttft"]))
            print(
                f"Client association: matched={len(report['client']['matched'])} "
                f"unmatched={len(report['client']['rejected'])}"
            )
        for key, reason in report["unavailable"].items():
            print(f"UNAVAILABLE {key}: {reason}")
        for problem in [*report["notes"], *report["problems"]]:
            print(f"NOTE: {problem}")
        if args.command == "export-trace":
            export_trace(args.output, trace_document(report, events), args.max_trace_bytes)
            print(f"Trace: {args.output}")
        return 2 if report["status"] == "INCOMPLETE" else 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"Cannot analyze request capture: {error}", file=sys.stderr)
        return 2
