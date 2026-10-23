# SPDX-License-Identifier: Apache-2.0
"""Client arrival metrics; no server instrumentation or server clock required."""

from collections import Counter

NS_PER_SECOND = 1_000_000_000
MS_PER_SECOND = 1000


def summarize_overlap(requests, batch, dp_size):
    """Count real token events strictly between latest first and earliest last.

    Whole SSE bursts at either boundary are excluded, including MTP tokens.
    Every DP uses the same window; TP copies are never added to output counts.
    Effective TPOT is the reciprocal of throughput per concurrent request,
    not an arithmetic mean or percentile of individual token latencies.
    """
    result = {
        "status": "unavailable",
        "reason": "",
        "source": "client_sse_arrival",
        "scope": "all_requests_overlap",
        "bs": batch * dp_size,
        "tokens": 0,
        "tps": None,
        "tpot_ms": None,
        "per_dp": {},
        "per_request": {},
        "window": None,
    }

    def unavailable(reason):
        return {**result, "reason": reason}

    if batch < 1 or dp_size < 1 or len(requests) != batch * dp_size:
        return unavailable("Expected exactly batch * DP requests")
    identities = [row.get("request_id") for row in requests]
    if any(not isinstance(key, str) or not key for key in identities) or len(set(identities)) != len(requests):
        return unavailable("Missing or duplicate request ID")
    ranks = [row.get("dp_rank") for row in requests]
    if any(type(rank) is not int for rank in ranks) or Counter(ranks) != dict.fromkeys(range(dp_size), batch):
        return unavailable("Expected exactly batch requests routed to each DP")
    for row in requests:
        if row.get("status") != "ok" or row.get("token_ids_complete") is not True:
            return unavailable(f"{row['request_id']}: unsuccessful stream or incomplete token IDs")
        events = row.get("token_events")
        if not events or any(
            not isinstance(event, (list, tuple))
            or len(event) != 2
            or any(type(value) is not int for value in event)
            or event[0] < 0
            or event[1] < 1
            for event in events
        ):
            return unavailable(f"{row['request_id']}: missing or invalid token events")
        if any(left[0] > right[0] for left, right in zip(events, events[1:])):
            return unavailable(f"{row['request_id']}: non-monotonic client timestamps")
        if row.get("first_ns") != events[0][0] or row.get("last_ns") != events[-1][0]:
            return unavailable(f"{row['request_id']}: first/last arrival does not match token events")
        if (
            type(row.get("completion_tokens")) is not int
            or sum(count for _, count in events) != row["completion_tokens"]
        ):
            return unavailable(f"{row['request_id']}: token IDs disagree with completion usage")

    start = max(row["first_ns"] for row in requests)
    end = min(row["last_ns"] for row in requests)
    seconds = (end - start) / NS_PER_SECOND
    result["window"] = {"start_ns": start, "end_ns": end, "seconds": max(0, seconds)}
    if seconds <= 0:
        return unavailable("No positive all-request overlap window; increase output length")
    for row in requests:
        counts = [count for timestamp, count in row["token_events"] if start < timestamp < end]
        result["per_request"][row["request_id"]] = {
            "dp_rank": row["dp_rank"],
            "tokens": sum(counts),
            "events": len(counts),
        }
    if any(not row["tokens"] for row in result["per_request"].values()):
        return unavailable("At least one request has no interior token event; increase output length")
    for rank in range(dp_size):
        rows = [row for row in result["per_request"].values() if row["dp_rank"] == rank]
        tokens = sum(row["tokens"] for row in rows)
        result["per_dp"][str(rank)] = {
            "batch": batch,
            "tokens": tokens,
            "events": sum(row["events"] for row in rows),
            "tps": tokens / seconds,
            "tpot_ms": MS_PER_SECOND * batch * seconds / tokens,
        }
    tokens = sum(row["tokens"] for row in result["per_dp"].values())
    result.update(
        status="ready",
        tokens=tokens,
        tps=tokens / seconds,
        tpot_ms=MS_PER_SECOND * result["bs"] * seconds / tokens,
    )
    return result
