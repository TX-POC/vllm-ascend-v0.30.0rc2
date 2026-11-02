#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP-only capacity benchmark; edit Config defaults or pass --help for overrides.

No vLLM imports, service log parsing, or shared service filesystem are needed.
The primary window starts when every request has returned output and ends at
the earliest final-token arrival. All timestamps and counts come from clients.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import dataclasses
import json
import math
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from benchmark_metrics import summarize_overlap


# Frequently changed workload configuration (batch is PER DP replica, not card).
@dataclasses.dataclass
class Config:
    base_url: str = "http://127.0.0.1:8080"
    model: str = "glm"
    api_key: str = ""
    input_tokens: int = 130560  # Leave 512 output tokens within a 131072-token context.
    output_tokens: int = 512
    levels: tuple[int, ...] = (1, 2, 4, 8, 16)
    data_parallel_size: int = 1
    tensor_parallel_size: int = 8
    prompt_jsonl: str = ""  # First nonempty record; reused by all requests.
    prompt_field: str = "question"
    dataset_length_policy: str = "truncate"  # or strict; short inputs fail.
    repeat_prompt: bool = False
    prefix_warmup: bool = True  # Requires APC + cached_tokens telemetry.
    cache_block_size: int = 128  # For APC verification; match the service setting.
    request_timeout: int = 3000
    result_dir: str = "scripts/result"


CONTROL_TIMEOUT = 120
PROFILE_TIMEOUT = 300
REFRESH_SECONDS = 1.0
PROFILE_WARMUP_TOKENS = 8


class BenchmarkError(RuntimeError):
    pass


def sse_events(response):
    """Parse SSE framing, including comments and multiline data fields."""
    lines = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if lines:
                yield "\n".join(lines)
                lines.clear()
        elif line.startswith("data:"):
            lines.append(line[5:].removeprefix(" "))
    if lines:
        yield "\n".join(lines)


class Http:
    def __init__(self, config: Config):
        self.config = config
        hostname = urllib.parse.urlparse(config.base_url).hostname
        proxies = {} if hostname in ("localhost", "127.0.0.1", "::1") else None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))

    def request(self, path, payload=None, *, dp_rank=None, request_id=None, timeout=CONTROL_TIMEOUT):
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if dp_rank is not None:
            headers["X-data-parallel-rank"] = str(dp_rank)
        if request_id:
            headers["X-Request-Id"] = request_id
        body = json.dumps(payload, separators=(",", ":")).encode() if payload is not None else None
        request = urllib.request.Request(self.config.base_url.rstrip("/") + path, data=body, headers=headers)
        try:
            return self.opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", errors="replace")
            raise BenchmarkError(f"{request.full_url}: HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BenchmarkError(
                f"{request.full_url}: {exc}. Check --base-url and whether the service has finished starting."
            ) from exc

    def json(self, path, payload=None, **kwargs):
        with self.request(path, payload, **kwargs) as response:
            result = json.load(response)
        if not isinstance(result, dict) or result.get("error"):
            raise BenchmarkError(f"{path}: invalid response: {result!r}")
        return result

    def complete(self, prompt, max_tokens, request_id, dp_rank, observer=None):
        """Return raw HTTP times and counts; SSE packets are never tokens."""
        result = {
            "request_id": request_id,
            "dp_rank": dp_rank,
            "status": "error",
            "error": None,
            "start_ns": 0,
            "first_ns": None,
            "last_ns": None,
            "end_ns": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "cached_tokens": None,
            "token_events": [],
            "token_ids_complete": True,
        }
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "return_token_ids": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }
        try:
            done = False
            # Constructing the request body is included in HTTP latency consistently.
            result["start_ns"] = time.perf_counter_ns()
            with self.request(
                "/v1/completions", payload, dp_rank=dp_rank, request_id=request_id, timeout=self.config.request_timeout
            ) as response:
                for data in sse_events(response):
                    now = time.perf_counter_ns()
                    if data == "[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise BenchmarkError(str(event["error"]))
                    usage = event.get("usage") or {}
                    for key in ("prompt_tokens", "completion_tokens"):
                        if key in usage:
                            value = usage[key]
                            if type(value) is not int or value < 0:
                                raise BenchmarkError(f"invalid usage.{key}: {value!r}")
                            result[key] = value
                    details = usage.get("prompt_tokens_details") or {}
                    if "cached_tokens" in details:
                        cached = details["cached_tokens"]
                        if type(cached) is not int or cached < 0:
                            raise BenchmarkError("invalid usage.prompt_tokens_details.cached_tokens")
                        result["cached_tokens"] = cached
                    choices = event.get("choices") or []
                    if len(choices) > 1:
                        raise BenchmarkError("expected exactly one completion choice")
                    if not choices:
                        continue
                    choice = choices[0]
                    ids = choice.get("token_ids")
                    has_ids = isinstance(ids, list) and all(type(token) is int and token >= 0 for token in ids)
                    if ids is not None and not has_ids:
                        raise BenchmarkError("invalid streamed token_ids")
                    has_output = bool(ids) if has_ids else bool(choice.get("text"))
                    if not has_output:
                        continue
                    if result["first_ns"] is None:
                        result["first_ns"] = now
                    result["last_ns"] = now
                    if has_ids:
                        result["token_events"].append([now, len(ids)])
                        if observer:
                            observer(request_id, len(ids))
                    else:
                        result["token_ids_complete"] = False
            if not done or result["first_ns"] is None:
                raise BenchmarkError("stream ended without output or terminal [DONE]")
            if result["prompt_tokens"] != len(prompt) or result["completion_tokens"] != max_tokens:
                raise BenchmarkError(
                    f"usage length mismatch: expected input={len(prompt)}, output={max_tokens}; "
                    f"received {result['prompt_tokens']}, {result['completion_tokens']}"
                )
            if result["cached_tokens"] is not None and result["cached_tokens"] > len(prompt):
                raise BenchmarkError("cached_tokens exceeds prompt length")
            if not result["token_ids_complete"]:
                raise BenchmarkError(
                    "streamed token_ids are required; SSE packet counts cannot measure generated tokens"
                )
            counted = sum(event[1] for event in result["token_events"])
            if counted != max_tokens:
                raise BenchmarkError(f"stream token IDs ({counted}) disagree with completion usage ({max_tokens})")
            result["status"] = "ok"
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            result["end_ns"] = time.perf_counter_ns()
            if observer:
                observer(request_id, None)
        return result


def tokenize(http: Http, text: str):
    result = http.json("/tokenize", {"model": http.config.model, "prompt": text, "add_special_tokens": False})
    tokens = result.get("tokens")
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise BenchmarkError("/tokenize did not return a nonempty integer token list")
    return tokens


class PromptSource:
    def __init__(self, config: Config, http: Http):
        self.config = config
        self.seed = tokenize(http, " Ascend long context benchmark deterministic text payload.")
        self.symbols = list(dict.fromkeys(self.seed))[:2]
        if len(self.symbols) < 2:
            raise BenchmarkError("synthetic seed needs at least two distinct valid tokens")
        self.dataset = None
        if config.prompt_jsonl:
            with Path(config.prompt_jsonl).open(encoding="utf-8") as source:
                record = next((json.loads(line) for line in source if line.strip()), None)
            text = record.get(config.prompt_field) if isinstance(record, dict) else None
            if not isinstance(text, str) or not text:
                raise BenchmarkError(f"first JSONL record needs a nonempty {config.prompt_field!r} field")
            tokens = tokenize(http, text)
            if len(tokens) < config.input_tokens:
                raise BenchmarkError("JSONL input is too short; dataset mode never repeats or pads text")
            if config.dataset_length_policy == "strict" and len(tokens) != config.input_tokens:
                raise BenchmarkError("JSONL input length differs from input_tokens with strict policy")
            self.dataset = tokens[: config.input_tokens]

    def build(self, variant, length=None, *, synthetic=False):
        length = self.config.input_tokens if length is None else length
        if self.dataset is not None and not synthetic:
            return list(self.dataset[:length])
        if length < 32 or not 0 <= variant < 2**32:
            raise BenchmarkError("synthetic prompts need >=32 tokens and a 32-bit variant")
        # Encode variants immediately, avoiding accidental shared long prefixes.
        signature = [self.symbols[(variant >> bit) & 1] for bit in range(32)]
        rest = (self.seed * math.ceil((length - 32) / len(self.seed)))[: length - 32]
        return signature + rest

    def variant(self, rank, index):
        if self.config.repeat_prompt or self.dataset is not None:
            return rank * 65536
        return rank * 65536 + index


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)

    def quantile(q):
        index = (len(ordered) - 1) * q
        lo = int(index)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (index - lo)

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "p50": quantile(0.5),
        "p75": quantile(0.75),
        "p90": quantile(0.9),
        "p99": quantile(0.99),
        "max": ordered[-1],
    }


def summarize_http(requests):
    good = [r for r in requests if r["status"] == "ok"]
    result = {
        "success": len(good),
        "failed": len(requests) - len(good),
        "ttft_ms": distribution([(r["first_ns"] - r["start_ns"]) / 1e6 for r in good]),
        "e2e_ms": distribution([(r["end_ns"] - r["start_ns"]) / 1e6 for r in good]),
        "request_tpot_ms": distribution(
            [
                (r["last_ns"] - r["first_ns"]) / 1e6 / (r["completion_tokens"] - r["token_events"][0][1])
                for r in good
                if r["completion_tokens"] > r["token_events"][0][1]
            ]
        ),
        "request_tpot_basis": "(last-first output arrival)/(output tokens-first SSE burst tokens)",
        "e2e_tps": None,
        "source": "client_http_arrival",
    }
    if good and len(good) == len(requests):
        elapsed = (max(r["end_ns"] for r in good) - min(r["start_ns"] for r in good)) / 1e9
        if elapsed > 0:
            result["e2e_tps"] = sum(r["completion_tokens"] for r in good) / elapsed
    return result


def scheduler_metrics(http: Http):
    try:
        with http.request("/metrics", timeout=2) as response:
            lines = response.read().decode().splitlines()
        result = {"running": 0.0, "waiting": 0.0}
        found = set()
        for line in lines:
            if not line or line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split()[0]
            for key in result:
                if name == f"vllm:num_requests_{key}":
                    tail = line.rsplit("}", 1)[-1] if "}" in line else line[len(name) :]
                    result[key] += float(tail.split()[0])
                    found.add(key)
        return result if len(found) == 2 else {"unavailable": "running/waiting series absent"}
    except Exception as exc:
        return {"unavailable": str(exc)}


class Reporter:
    def __init__(self, config: Config):
        stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.directory = Path(config.result_dir) / stamp
        self.directory.mkdir(parents=True)
        public_config = dataclasses.asdict(config)
        public_config["api_key"] = "***" if config.api_key else ""
        self.report = {"config": public_config, "levels": [], "profiles": [], "status": "running"}
        self.last_lines = 0

    def save(self, *, final=False):
        destination = self.directory / ("result.json" if final else "live.json")
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def display(self, progress=None):
        lines = ["BS/DP | success | TTFT mean ms | E2E mean ms | overlap TPOT ms | global TPS"]

        def value(item, key=None):
            if key and isinstance(item, dict):
                item = item.get(key)
            return f"{item:.3f}" if isinstance(item, (float, int)) else "--"

        for row in self.report["levels"]:
            http, overlap = row.get("http", {}), row.get("overlap", {})
            lines.append(
                f"{row['batch']:5} | {http.get('success', 0):7} | "
                f"{value(http.get('ttft_ms'), 'mean'):>12} | {value(http.get('e2e_ms'), 'mean'):>11} | "
                f"{value(overlap.get('tpot_ms')):>18} | {value(overlap.get('tps')):>10}"
            )
            if overlap.get("reason"):
                lines.append(f"       overlap: {overlap['reason']}")
            if self.report["config"].get("prefix_warmup") and row.get("http"):
                cached = [r.get("cached_tokens") for r in row.get("requests", ())]
                lines.append(f"       prefix cached_tokens/request: {cached}; status={row.get('status')}")
            for error in dict.fromkeys(row.get("errors", ())):
                lines.append(f"       error: {error}")
        if progress:
            lines.append(progress)
        if sys.stdout.isatty() and self.last_lines:
            print(f"\033[{self.last_lines}A\033[J", end="")
        print("\n".join(lines), flush=True)
        self.last_lines = len(lines)


def validate_config(config):
    parsed = urllib.parse.urlparse(config.base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path not in ("", "/"):
        raise BenchmarkError("base_url must be an HTTP(S) origin without /v1 or other path")
    for name in (
        "input_tokens",
        "output_tokens",
        "data_parallel_size",
        "tensor_parallel_size",
        "cache_block_size",
        "request_timeout",
    ):
        if getattr(config, name) <= 0:
            raise BenchmarkError(f"{name} must be positive")
    if config.input_tokens < 32 or not config.levels or any(level <= 0 or level >= 65536 for level in config.levels):
        raise BenchmarkError("input_tokens must be >=32; levels must be nonempty and between 1 and 65535")
    if len(set(config.levels)) != len(config.levels):
        raise BenchmarkError("levels must not repeat")
    if config.dataset_length_policy not in ("truncate", "strict"):
        raise BenchmarkError("dataset_length_policy must be truncate or strict")


def preflight(http, config, *, output_tokens=None):
    models = http.json("/v1/models").get("data", [])
    model = next((row for row in models if isinstance(row, dict) and row.get("id") == config.model), None)
    if model is None:
        raise BenchmarkError(f"model {config.model!r} is absent from /v1/models")
    max_model_len = model.get("max_model_len")
    output_tokens = config.output_tokens if output_tokens is None else output_tokens
    if type(max_model_len) is int and max_model_len > 0 and config.input_tokens + output_tokens > max_model_len:
        raise BenchmarkError(
            f"input_tokens ({config.input_tokens}) + output_tokens ({output_tokens}) exceeds "
            f"model max_model_len ({max_model_len}); reduce the workload length."
        )


def verify_cached(result, input_tokens, block_size, extra_recompute_blocks=0):
    if result["status"] != "ok":
        raise BenchmarkError("prefix request failed: " + str(result["error"]))
    cached = result.get("cached_tokens")
    if type(cached) is not int:
        raise BenchmarkError("prefix verification needs integer cached_tokens; enable --enable-prompt-tokens-details")
    # The final prompt token must be recomputed; cache hits end on block boundaries.
    full_prefix = (input_tokens - 1) // block_size * block_size
    threshold = max(1, full_prefix - extra_recompute_blocks * block_size)
    if cached < threshold:
        raise BenchmarkError(
            f"prefix verification needs cached_tokens>={threshold}; received {cached!r}. "
            "Check warmed prefix residency and --cache-block-size."
        )
    return max(0, (full_prefix - cached + block_size - 1) // block_size)


def prefix_probe(config, http, source, block_size, results):
    length = min(config.input_tokens, max(4096, 3 * block_size))
    if length <= 2 * block_size:
        raise BenchmarkError("prefix capability probe needs input longer than two cache blocks to verify MTP replay")
    extra_recompute_blocks = []
    for rank in range(config.data_parallel_size):
        print(f"Checking native prefix-cache replay on DP{rank}...", flush=True)
        prompt = source.build(2**32 - 1 - rank, length, synthetic=True)
        pair = [http.complete(prompt, 1, f"probe-{uuid.uuid4().hex}-{phase}", rank) for phase in range(2)]
        results.extend(pair)
        if pair[0]["status"] != "ok":
            raise BenchmarkError("prefix capability warmup failed: " + str(pair[0]["error"]))
        # MTP may drop one additional tail block. Calibrate once per DP over HTTP;
        # measured requests must then meet this same bound, even if caches evict.
        extra = verify_cached(pair[1], length, block_size, extra_recompute_blocks=1)
        extra_recompute_blocks.append(extra)
        print(f"DP{rank}: cached_tokens={pair[1]['cached_tokens']}; extra replay blocks={extra}", flush=True)
    return extra_recompute_blocks


def warm_prefixes(config, http, source, batch, warmed, reporter):
    jobs = []
    for rank in range(config.data_parallel_size):
        for index in range(batch):
            variant = source.variant(rank, index)
            key = (rank, variant)
            if key not in warmed:
                jobs.append((rank, variant))
                warmed.add(key)
    started = time.perf_counter_ns()
    results = []
    report = {"batch": batch, "duration_ms": None, "requests": results}
    reporter.report.setdefault("prefix_warmups", []).append(report)
    reporter.display(f"Warming BS/DP={batch}: {len(jobs)} new prefixes")
    reporter.save()
    # Bound prefill concurrency independently of the measured simultaneous wave.
    with futures.ThreadPoolExecutor(max_workers=config.data_parallel_size) as pool:
        pending = [
            pool.submit(http.complete, source.build(variant), 1, f"warm-{uuid.uuid4().hex}", rank)
            for rank, variant in jobs
        ]
        for future in pending:
            result = future.result()
            results.append(result)
            report["duration_ms"] = (time.perf_counter_ns() - started) / 1e6
            reporter.display(f"Warming BS/DP={batch}: completed {len(results)}/{len(jobs)} prefixes")
            reporter.save()
            if result["status"] != "ok":
                raise BenchmarkError("prefix warmup failed: " + str(result["error"]))
    report["duration_ms"] = (time.perf_counter_ns() - started) / 1e6
    return report


class ProfileCoordinator:
    """Native HTTP profiler controls, with a client-observed token budget."""

    def __init__(self, http, mode, tokens=0, request_ids=()):
        self.http, self.mode, self.tokens = http, mode, tokens
        self.counts = dict.fromkeys(request_ids, 0)
        self.condition = threading.Condition()
        self.done, self.finished, self.thread = set(), False, None
        self.started = False
        self.result = {
            "mode": mode,
            "status": "error",
            "requested_tokens_per_request": tokens,
            "warning": "Separate profile wave; HTTP ACKs do not verify trace files or device steps.",
        }

    def control(self, action):
        with self.http.request(f"/{action}_profile", {}, timeout=PROFILE_TIMEOUT) as response:
            response.read()
            self.result[action + "_ack"] = {"http_status": response.status, "time_ns": time.perf_counter_ns()}

    def start(self):
        self.started = True  # Stop even after a partial/failed start request.
        self.control("start")

    def observe(self, request_id, count):
        with self.condition:
            if count is None:
                self.done.add(request_id)
            else:
                self.counts[request_id] += count
            if self.thread is None and not self.done and all(v >= PROFILE_WARMUP_TOKENS for v in self.counts.values()):
                self.thread = threading.Thread(target=self.capture, name="benchmark-profile")
                self.thread.start()
            self.condition.notify_all()

    def capture(self):
        try:
            self.start()
            with self.condition:
                baseline = self.counts.copy()
                self.result["start_token_counts"] = baseline
                self.condition.notify_all()

                def reached():
                    return all(self.counts[k] - v >= self.tokens for k, v in baseline.items())

                self.condition.wait_for(lambda: self.done or self.finished or reached())
                self.result["observed_tokens_per_request"] = {k: self.counts[k] - v for k, v in baseline.items()}
                self.result["status"] = "ok" if reached() else "error"
                self.result["stop_reason"] = "token_budget" if reached() else "request_finished"
                if not reached():
                    self.result["error"] = "a request ended before every stream reached the profile token budget"
        except Exception as exc:
            self.result["error"] = str(exc)
        finally:
            self.stop()

    def stop(self):
        if self.started:
            self.started = False
            try:
                self.control("stop")
            except Exception as exc:
                self.result["status"] = "error"
                self.result["stop_error"] = str(exc)

    def finish(self):
        with self.condition:
            self.finished = True
            self.condition.notify_all()
        if self.thread:
            self.thread.join()
        elif self.mode == "decode":
            self.result["error"] = "not every request reached the decode warmup before a request ended"
        self.stop()
        return self.result


def run_level(config, http, source, batch, reporter, *, profile_tokens=None, prefix_extra_recompute_blocks=None):
    """One synchronized wave; profiling waves are stored outside baseline results."""
    run_id = uuid.uuid4().hex
    jobs = [
        (f"bench-{run_id}-dp{rank}-r{index}", rank, source.build(source.variant(rank, index)))
        for rank in range(config.data_parallel_size)
        for index in range(batch)
    ]
    barrier = threading.Barrier(len(jobs) + 1)
    profile = ProfileCoordinator(http, "decode", profile_tokens, [job[0] for job in jobs]) if profile_tokens else None
    row = {"batch": batch, "run_id": run_id, "requests": [], "scheduler": [], "errors": [], "status": "running"}
    reporter.report["profiles" if profile else "levels"].append(row)
    progress_lock = threading.Lock()
    counts, completed = dict.fromkeys((job[0] for job in jobs), 0), set()

    def observe(request_id, count):
        with progress_lock:
            if count is None:
                completed.add(request_id)
            else:
                counts[request_id] += count
        if profile:
            profile.observe(request_id, count)

    def invoke(request_id, rank, prompt):
        barrier.wait(timeout=CONTROL_TIMEOUT)
        return http.complete(prompt, config.output_tokens, request_id, rank, observer=observe)

    try:
        with futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            pending = [pool.submit(invoke, *job) for job in jobs]
            barrier.wait(timeout=CONTROL_TIMEOUT)
            while not all(future.done() for future in pending):
                with progress_lock:
                    row["progress"] = {
                        "output_tokens": sum(counts.values()),
                        "first_token_requests": sum(v > 0 for v in counts.values()),
                        "completed_requests": len(completed),
                        "total_requests": len(jobs),
                    }
                row["scheduler"].append({"time_ns": time.perf_counter_ns(), **scheduler_metrics(http)})
                row["requests"] = [future.result() for future in pending if future.done()]
                reporter.display(f"{'Profiling' if profile else 'Running'} BS/DP={batch}: {row['progress']}")
                reporter.save()
                futures.wait(pending, timeout=REFRESH_SECONDS, return_when=futures.ALL_COMPLETED)
            row["requests"] = [future.result() for future in pending]
    except BaseException as exc:
        row["errors"].append(str(exc) or type(exc).__name__)
        row["status"] = "error"
        raise
    finally:
        if profile:
            row["profile"] = profile.finish()
            if row["profile"]["status"] != "ok":
                row["errors"].append(row["profile"].get("error") or row["profile"].get("stop_error"))
        reporter.save()
    row["http"] = summarize_http(row["requests"])
    row["errors"].extend(r["error"] for r in row["requests"] if r["status"] != "ok")
    if config.prefix_warmup:
        for result in row["requests"]:
            try:
                extra = prefix_extra_recompute_blocks[result["dp_rank"]] if prefix_extra_recompute_blocks else 0
                verify_cached(result, config.input_tokens, config.cache_block_size, extra)
            except BenchmarkError as exc:
                row["errors"].append(str(exc))
    if not profile:
        row["overlap"] = summarize_overlap(row["requests"], batch, config.data_parallel_size)
        if row["overlap"]["status"] != "ready":
            row["errors"].append("client full-batch window unavailable: " + str(row["overlap"].get("reason")))
    row["status"] = "error" if row["errors"] else "ok"
    reporter.save()
    reporter.display()
    return row


def run_prefill_profile(config, http, source, reporter):
    profile = ProfileCoordinator(http, "prefill")
    row = {"mode": "prefill", "status": "error"}
    reporter.report["profiles"].append(row)
    try:
        profile.start()
        variant = int(uuid.uuid4().hex[:8], 16)
        row["request"] = http.complete(source.build(variant), 1, f"prefill-{uuid.uuid4().hex}", 0)
        profile.result["status"] = row["request"]["status"]
        if row["request"].get("error"):
            profile.result["error"] = row["request"]["error"]
        row["prefix_note"] = "Cached prefixes may reduce prefill work; cached_tokens is recorded in the response."
    except Exception as exc:
        profile.result["error"] = str(exc)
    finally:
        row["profile"] = profile.finish()
        row["status"] = row["profile"]["status"]
        reporter.save()
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in dataclasses.fields(Config):
        options = {"default": field.default}
        if isinstance(field.default, bool):
            options["action"] = argparse.BooleanOptionalAction
        elif field.name == "levels":
            options.update(type=lambda text: tuple(map(int, text.split(","))), help="batch sizes per DP: 1,2,4,8,16")
        else:
            options["type"] = type(field.default)
        if field.name == "dataset_length_policy":
            options["choices"] = ("truncate", "strict")
        parser.add_argument("--" + field.name.replace("_", "-"), **options)
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument("--profile-prefill", action="store_true")
    profile_group.add_argument("--profile-decode", action="store_true")
    parser.add_argument("--profile-decode-batch", type=int)
    parser.add_argument("--profile-decode-tokens", type=int, default=8)
    args = parser.parse_args(argv)
    config = Config(**{field.name: getattr(args, field.name) for field in dataclasses.fields(Config)})
    validate_config(config)
    if args.profile_decode_batch is not None and not args.profile_decode:
        raise BenchmarkError("--profile-decode-batch requires --profile-decode")
    if args.profile_decode_batch is not None and args.profile_decode_batch not in config.levels:
        raise BenchmarkError("--profile-decode-batch must be included in --levels")
    if args.profile_decode_tokens <= 0:
        raise BenchmarkError("profile token budget must be positive")
    if args.profile_decode and config.output_tokens <= PROFILE_WARMUP_TOKENS + args.profile_decode_tokens:
        raise BenchmarkError("decode profiling needs output_tokens > warmup_tokens + profile_tokens")
    return config, args


def main(argv=None):
    reporter = None
    try:
        config, args = parse_args(argv)
        reporter = Reporter(config)
        print(
            f"Endpoint={config.base_url}; model={config.model}; input={config.input_tokens}; "
            f"output={config.output_tokens}; DP={config.data_parallel_size}; "
            f"TP={config.tensor_parallel_size} (metadata)",
            flush=True,
        )
        http = Http(config)
        preflight(http, config, output_tokens=1 if args.profile_prefill else config.output_tokens)
        source = PromptSource(config, http)
        reporter.report["input_source"] = "JSONL first record" if source.dataset is not None else "synthetic token IDs"
        print("Primary TPOT/TPS: real token arrivals after every request starts and before any request's final token.")
        if args.profile_prefill:
            run_prefill_profile(config, http, source, reporter)
        else:
            warmed = set()
            prefix_extra_recompute_blocks = None
            if config.prefix_warmup:
                reporter.report["prefix_probe"] = []
                prefix_extra_recompute_blocks = prefix_probe(
                    config, http, source, config.cache_block_size, reporter.report["prefix_probe"]
                )
                reporter.report["prefix_extra_recompute_blocks"] = prefix_extra_recompute_blocks
            for batch in config.levels:
                if config.prefix_warmup:
                    warm_prefixes(config, http, source, batch, warmed, reporter)
                run_level(
                    config, http, source, batch, reporter, prefix_extra_recompute_blocks=prefix_extra_recompute_blocks
                )
                if args.profile_decode and args.profile_decode_batch in (None, batch):
                    run_level(
                        config,
                        http,
                        source,
                        batch,
                        reporter,
                        profile_tokens=args.profile_decode_tokens,
                        prefix_extra_recompute_blocks=prefix_extra_recompute_blocks,
                    )
        failed = any(row["status"] != "ok" for row in reporter.report["levels"] + reporter.report["profiles"])
        reporter.report["status"] = "error" if failed else "ok"
        return 1 if failed else 0
    except (Exception, KeyboardInterrupt) as exc:
        if reporter:
            reporter.report["status"] = "error"
            reporter.report["error"] = str(exc) or type(exc).__name__
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if reporter:
            reporter.save(final=True)
            print(f"Results: {reporter.directory / 'result.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
