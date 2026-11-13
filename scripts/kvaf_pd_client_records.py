# SPDX-License-Identifier: Apache-2.0
"""Import existing request-tool output; no requests, clock probes or guessed IDs."""

import hashlib
import json
import math
from pathlib import Path


def read_client_records(path, limit):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("client record input exceeds --max-input-bytes")
    digest = hashlib.sha256(data).hexdigest()
    try:
        document = json.loads(data)
    except ValueError:
        records = [json.loads(line) for line in data.splitlines() if line.strip()]
    else:
        if isinstance(document, list):
            records = document
        elif isinstance(document, dict) and isinstance(document.get("levels"), list):
            records = []
            for section in ("prefix_warmups", "request_warmups", "levels", "profiles"):
                for group in document.get(section, []):
                    rows = group.get("requests", [])
                    if "request" in group:
                        rows = [*rows, group["request"]]
                    for row in rows:
                        item = dict(row, clock_domain="benchmark_perf_counter_ns", benchmark_section=section)
                        if (
                            "request_phase" in item
                            and "request_phase" in group
                            and item["request_phase"] != group["request_phase"]
                        ):
                            raise ValueError("conflicting benchmark group/request_phase labels")
                        if "request_phase" not in item and "request_phase" in group:
                            item["request_phase"] = group["request_phase"]
                        records.append(item)
            records.extend(
                dict(row, clock_domain="benchmark_perf_counter_ns", benchmark_section="prefix_probe")
                for row in document.get("prefix_probe", [])
            )
        elif isinstance(document, dict) and isinstance(document.get("requests"), list):
            records = document["requests"]
        elif isinstance(document, dict):
            records = [document]
        else:
            raise ValueError("client records must be an object/list, JSONL, or existing benchmark result")
    if not all(isinstance(row, dict) for row in records):
        raise ValueError("client record must be an object")
    return records, digest, len(data)


def request_candidates(row):
    request = row.get("external_request_id", row.get("request", row.get("request_id")))
    candidates = {request} if isinstance(request, str) else set()
    if "external_request_id" not in row and row.get("response_id_consistent") is True:
        response = row.get("response_id")
        if isinstance(response, str) and response:
            candidates.add(response)
            if row.get("endpoint") == "/v1/completions" and type(row.get("completion_prompt_index")) is int:
                candidates.add(f"{response}-{row['completion_prompt_index']}")
    return candidates


def request_identity(row, known, selection=()):
    """Match real IDs only, with a unique observed generation or an explicit one."""
    generation = row.get("generation")
    if generation is not None and not isinstance(generation, str):
        raise ValueError("client generation must be a string or omitted")
    matches = {
        candidate
        for candidate in request_candidates(row)
        if candidate in known
        and (generation is None or generation in known[candidate])
        and (not selection or candidate in selection)
    }
    if len(matches) != 1:
        return None
    request = next(iter(matches))
    choices = known[request]
    if generation is None and len(choices) == 1:
        generation = next(iter(choices))
    return (request, generation) if isinstance(generation, str) and generation and generation in choices else None


def client_events(records, digest, known, selection, distribution):
    events, rejected, matched, latencies, ttft, seen = [], [], [], [], [], set()
    phase_only = 0
    for index, row in enumerate(records):
        # external_request_id is an explicit mapping provided by the existing
        # tool/exporter. Neither prompt content nor arrival proximity is used.
        if selection and not request_candidates(row).intersection(selection):
            continue
        identity = request_identity(row, known, selection)
        if identity is None:
            rejected.append(dict(record=index, reason="unmatched/ambiguous request ID or generation"))
            continue
        request, generation = identity
        key = request, generation
        if key in seen:
            raise ValueError("duplicate client request/generation; provide one verified attempt per request")
        seen.add(key)
        begin = row.get("sent_ns", row.get("start_ns"))
        end = row.get("completed_ns", row.get("end_ns"))
        first = row.get("first_output_ns", row.get("first_ns"))
        if (
            begin is None
            and end is None
            and first is None
            and row.get("request_phase") in ("measure", "warmup", "profile")
        ):
            phase_only += 1
            continue
        if type(begin) is not int or type(end) is not int or not 0 < begin <= end < 2**63:
            raise ValueError("client times must be positive ordered integers from one monotonic clock")
        if first is not None and (type(first) is not int or not begin <= first <= end):
            raise ValueError("client first output is outside send/complete interval")
        output = row.get("output_tokens", row.get("completion_tokens"))
        if output is not None and (type(output) is not int or output < 0):
            raise ValueError("client output token count must be nonnegative or unknown")
        successful = row.get("status") in ("ok", "success")
        # Imported file identity isolates the client clock unless a future
        # validated clock mapping is supplied. File wall times are never used.
        clock = row.get("clock_domain", f"unspecified-record-{index}")
        if not isinstance(clock, str) or not clock or len(clock) > 256:
            raise ValueError("client clock_domain must be a nonempty string of at most 256 characters")
        domain = json.dumps(["client", digest, clock], separators=(",", ":"))
        common = dict(
            request=request,
            generation=generation,
            domain=domain,
            actor="existing-client/" + digest,
            success=successful,
            observation="imported_client",
            output_tokens=output,
            request_phase=row.get("request_phase", "unknown"),
        )
        events.append(dict(common, name="client.request_e2e", start_ns=begin, end_ns=end))
        events.append(dict(common, name="CLIENT_REQUEST_SENT", start_ns=begin, end_ns=begin, instant=True))
        events.append(dict(common, name="CLIENT_RESPONSE_COMPLETED", start_ns=end, end_ns=end, instant=True))
        if first is not None:
            events.append(dict(common, name="CLIENT_FIRST_OUTPUT", start_ns=first, end_ns=first, instant=True))
        if successful:
            latencies.append((end - begin) / 1e6)
            if first is not None:
                ttft.append((first - begin) / 1e6)
        matched.append(dict(request=request, generation=generation, status=row.get("status"), output_tokens=output))
    summary = dict(
        e2e=distribution(latencies),
        ttft=distribution(ttft),
        matched=matched,
        rejected=rejected,
        failed_records=sum(item["status"] not in ("ok", "success") for item in matched),
        phase_only_records=phase_only,
        timing="same client monotonic clock; never subtracted from service clocks",
    )
    if any(not math.isfinite(item) for item in latencies + ttft):
        raise ValueError("client latency is outside supported finite range")
    return events, summary
