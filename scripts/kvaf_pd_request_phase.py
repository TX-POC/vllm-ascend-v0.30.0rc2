# SPDX-License-Identifier: Apache-2.0
"""Explicit request-purpose selection before any performance aggregation."""

from collections import Counter, defaultdict

from kvaf_pd_client_records import request_identity

PHASES = ("measure", "warmup", "profile", "unknown")


def request_phase(row):
    phase = row.get("request_phase", "unknown")
    if phase not in PHASES:
        raise ValueError("request_phase must be measure, warmup, profile or unknown")
    return phase


def select_request_phase(captures, client_rows, desired="measure", request_ids=()):
    if desired not in (*PHASES, "all"):
        raise ValueError("invalid request phase selection")
    labels, known, keys = defaultdict(set), defaultdict(set), set()
    for _, _, rows, _ in captures:
        for row in rows:
            if row.get("request") is not None:
                key = row["request"], row["generation"]
                keys.add(key)
                phase = request_phase(row)
                if phase != "unknown":
                    labels[key].add(phase)
                if row.get("association") != "unresolved":
                    known[key[0]].add(key[1])
    unmatched = 0
    for row in client_rows:
        phase = request_phase(row)
        identity = request_identity(row, known, request_ids)
        if identity is None:
            unmatched += 1
        elif phase != "unknown":
            labels[identity].add(phase)
    resolved = {}
    for key in keys:
        choices = labels[key]
        if len(choices) > 1:
            raise ValueError(f"conflicting request_phase labels for request {key[0]!r}, generation {key[1]!r}")
        resolved[key] = next(iter(choices)) if choices else "unknown"
    cohort = {key for key in keys if not request_ids or key[0] in request_ids}
    selected = {key for key in cohort if desired == "all" or resolved[key] == desired}
    result = []
    for header, checkpoint, rows, source in captures:
        chosen = [
            dict(row, request_phase=resolved[row["request"], row["generation"]])
            for row in rows
            if (row.get("request"), row.get("generation")) in selected
        ]
        batches = {row["batch_id"] for row in chosen if row["kind"] == "membership"}
        batch_phases = defaultdict(set)
        for row in chosen:
            if row["kind"] == "membership":
                batch_phases[row["batch_id"]].add(row["request_phase"])
        chosen.extend(
            dict(
                row,
                request_phase=next(iter(batch_phases[row["batch_id"]]))
                if len(batch_phases[row["batch_id"]]) == 1
                else "mixed",
            )
            for row in rows
            if row["kind"] in ("batch_span", "batch_begin") and row["batch_id"] in batches
        )
        result.append((header, checkpoint, chosen, source))
    counts = Counter(resolved[key] for key in cohort)
    quality = dict(
        selected_phase=desired,
        selected_requests=len(selected),
        excluded_requests=len(cohort - selected),
        observed_by_phase={phase: counts[phase] for phase in PHASES},
        unmatched_client_records=unmatched,
        classification="explicit request_phase and exact request/generation identity; never order, length or latency",
        count_scope="captured request generations before input-length filtering",
    )
    return result, quality
