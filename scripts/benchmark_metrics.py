# SPDX-License-Identifier: Apache-2.0
"""Client arrival metrics; no server instrumentation or server clock required."""

from collections import Counter
from statistics import fmean

NS_PER_SECOND = 1_000_000_000
MS_PER_SECOND = 1000


def linear_quantile(ordered, fraction):
    """Interpolate at (N - 1) * fraction in nonempty, sorted samples."""
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def token_event_error(row):
    """Validate the same real-token evidence for request and overlap TPOT."""
    if row.get("status") != "ok" or row.get("token_ids_complete") is not True:
        return "unsuccessful stream or incomplete token IDs"
    events = row.get("token_events")
    if (
        not isinstance(events, (list, tuple))
        or not events
        or any(
            not isinstance(event, (list, tuple))
            or len(event) != 2
            or any(type(value) is not int for value in event)
            or not 0 <= event[0] < 2**63
            or event[1] < 1
            for event in events
        )
    ):
        return "missing or invalid token events"
    if any(left[0] > right[0] for left, right in zip(events, events[1:])):
        return "non-monotonic client timestamps"
    if row.get("first_ns") != events[0][0] or row.get("last_ns") != events[-1][0]:
        return "first/last arrival does not match token events"
    if type(row.get("completion_tokens")) is not int or sum(count for _, count in events) != row["completion_tokens"]:
        return "token IDs disagree with completion usage"
    return None


def request_tpot_ms(row):
    if token_event_error(row) is not None:
        return None
    remaining = row["completion_tokens"] - row["token_events"][0][1]
    return (row["last_ns"] - row["first_ns"]) / 1e6 / remaining if remaining > 0 else None


def summarize_overlap(requests, batch, dp_size, *, dp_routing=True):
    """Count real token events strictly between latest first and earliest last.

    Whole SSE bursts at either boundary are excluded, including MTP tokens.
    Every DP uses the same window; TP copies are never added to output counts.
    Effective TPOT is the reciprocal of throughput per concurrent request,
    not an arithmetic mean or percentile of individual token latencies.
    Without DP routing, only global/client-request metrics are available.
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
    if dp_routing:
        if any(type(rank) is not int for rank in ranks) or Counter(ranks) != dict.fromkeys(range(dp_size), batch):
            return unavailable("Expected exactly batch requests routed to each DP")
    elif any(rank is not None for rank in ranks):
        return unavailable("Proxy routing must leave backend DP ranks unknown")
    for row in requests:
        error = token_event_error(row)
        if error is not None:
            return unavailable(f"{row['request_id']}: {error}")

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
    for rank in range(dp_size) if dp_routing else ():
        rows = [row for row in result["per_request"].values() if row["dp_rank"] == rank]
        tokens = sum(row["tokens"] for row in rows)
        result["per_dp"][str(rank)] = {
            "batch": batch,
            "tokens": tokens,
            "events": sum(row["events"] for row in rows),
            "tps": tokens / seconds,
            "tpot_ms": MS_PER_SECOND * batch * seconds / tokens,
        }
    tokens = sum(row["tokens"] for row in result["per_request"].values())
    result.update(
        status="ready",
        tokens=tokens,
        tps=tokens / seconds,
        tpot_ms=MS_PER_SECOND * result["bs"] * seconds / tokens,
    )
    return result


HOT_KV_BUCKETS = 1001
HOT_KV_QUANTILES = (50, 90, 95, 99)


def summarize_hot_kv(report, dp_size):
    """Merge evidence from one synchronized wave, without averaging rank percentiles."""
    rank_samples = report.get("rank_samples", {})
    histogram = report.get("histogram", [])
    missing = [rank for rank in range(dp_size) if str(rank) not in rank_samples]
    result = dict(
        status="unavailable",
        samples=0,
        missing_ranks=missing,
        rank_samples=rank_samples,
        bucket_width_pp=0.1,
        sample_unit="request_decode_step",
    )
    if missing:
        return {**result, "reason": "missing D rank reports"}
    if len(histogram) != HOT_KV_BUCKETS or any(type(count) is not int or count < 0 for count in histogram):
        return {**result, "reason": "invalid Hot KV histogram"}
    samples = sum(histogram)
    if samples == 0:
        return {**result, "reason": "D returned no valid history hit/miss samples"}
    if samples != sum(rank_samples.values()):
        return {**result, "reason": "Hot KV counts disagree with D rank reports"}
    thresholds = iter((p, (samples * p + 99) // 100) for p in HOT_KV_QUANTILES)
    current = next(thresholds, None)
    running = 0
    for index, count in enumerate(histogram):
        running += count
        while current is not None and running >= current[1]:
            result[f"p{current[0]}"] = index / 10
            current = next(thresholds, None)
        if current is None:
            break
    return {**result, "status": "ready", "samples": samples}


def summarize_decode_tps(rows):
    """Equal-weight wave mean at each target batch, excluding failed waves."""
    levels = {}
    for row in rows:
        if row["status"] == "running":
            continue
        valid = levels.setdefault(row["batch"], [])
        overlap = row.get("overlap", {})
        if row["status"] == "ok" and overlap.get("status") == "ready":
            valid.append(overlap["tps"])
    return {
        str(batch): {"mean_tps": fmean(valid) if valid else None, "valid_waves": len(valid)}
        for batch, valid in levels.items()
    }
