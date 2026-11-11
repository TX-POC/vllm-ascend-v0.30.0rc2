#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Buffered PD transfer measurements and offline request/layer statistics.

The CLI uses only the standard library and sends no inference requests.
Enable the companion Connector explicitly to capture measurements. No device
synchronization, Tensor reads or additional control calls are performed.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import os
import socket
import statistics
import sys
import threading
import time
import weakref
from collections import Counter, defaultdict
from pathlib import Path
from queue import Empty, Full, Queue
from types import MappingProxyType
from uuid import uuid4

# Edit before starting P/D. Copy both scripts to the same scripts/ directory.
# Use a fresh directory for each comparison run; never mix service restarts.
CAPTURE_DIR = Path(__file__).resolve().parent / "result/kvaf_pd_transfer_stats"
FLUSH_SECONDS = 5.0
QUEUE_RECORDS = 2048
FILE_BUFFER_BYTES = 1024 * 1024
SCHEMA_VERSION = 1
METRICS = MappingProxyType({"npu": "t1", "cpu": "t2"})


class BufferedCapture:
    """A bounded nonblocking queue; only the writer thread serializes samples.

    Full queues invalidate the measurement rather than block inference. The
    periodic checkpoint makes dropped records and unfinished work visible.
    """

    def __init__(self, header, snapshot, *, directory=CAPTURE_DIR):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{header['role']}-{os.getpid()}-{uuid4().hex}.jsonl"
        self.stream = self.path.open("x", buffering=FILE_BUFFER_BYTES)
        self.header, self.snapshot = header, snapshot
        self.queue = Queue(maxsize=QUEUE_RECORDS)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.enqueued = self.written = self.dropped = self.errors = 0
        self.failed = False
        self._write(dict(header, kind="header", schema=SCHEMA_VERSION, created_ns=time.time_ns()))
        self.stream.flush()
        self.thread = threading.Thread(target=self._run, name="kvaf-transfer-stats", daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def emit(self, record):
        with self.lock:
            try:
                self.queue.put_nowait(record)
                self.enqueued += 1
            except Full:
                self.dropped += 1

    def note_error(self):
        with self.lock:
            self.errors += 1

    def _write(self, record):
        self.stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _checkpoint(self, closed=False):
        snapshot = self.snapshot()
        with self.lock:
            counters = dict(enqueued=self.enqueued, written=self.written, dropped=self.dropped, errors=self.errors)
        self._write(dict(kind="checkpoint", closed=closed, wall_ns=time.time_ns(), **counters, **snapshot))
        self.stream.flush()

    def _run(self):
        deadline = time.monotonic() + FLUSH_SECONDS
        try:
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    record = self.queue.get(timeout=min(0.25, max(0.001, deadline - time.monotonic())))
                except Empty:
                    record = None
                if record is not None:
                    self._write(record)
                    with self.lock:
                        self.written += 1
                if time.monotonic() >= deadline:
                    self._checkpoint()
                    deadline = time.monotonic() + FLUSH_SECONDS
            self._checkpoint(closed=True)
        except Exception:
            self.failed = True
            self.note_error()
            print(f"KVAF transfer capture failed: {self.path}; measurements are incomplete", file=sys.stderr)
        finally:
            self.stream.close()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=FLUSH_SECONDS + 1)


def capture_header(config, role):
    """Startup-only metadata; no prompts, tensors or memory addresses."""
    pc = config.parallel_config
    return dict(
        role=role,
        host=socket.gethostname(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        engine=str(config.kv_transfer_config.engine_id),
        tp_rank=getattr(pc, "tensor_parallel_rank", None),
        tp_size=pc.tensor_parallel_size,
        dp_rank=pc.data_parallel_rank,
        dp_size=pc.data_parallel_size,
        max_num_seqs=config.scheduler_config.max_num_seqs,
        max_model_len=config.model_config.max_model_len,
        kv_port=config.kv_transfer_config.kv_port,
        source_tp0_only=config.kv_transfer_config.kv_connector_extra_config.get("source_tp0_only", False),
    )


class PullObserver:
    """Observe one Sender instance without replacing framework module globals.

    Original calls, exceptions and completion recovery pass through unchanged.
    A lost Pull ACK invalidates its measurement even if native recovery later
    succeeds: its original completion latency cannot be reconstructed exactly.
    """

    def __init__(self, sender, header, *, directory=CAPTURE_DIR):
        original = sender._call
        # The writer/atexit hook may outlive the Connector. Do not keep the
        # Sender, SDK owners or KV tensors alive through a bound-method cycle.
        self.original = (
            weakref.WeakMethod(original) if getattr(original, "__self__", None) is not None else lambda: original
        )
        self.lock = threading.Lock()
        self.jobs, self.layers = {}, {}
        header = dict(header, tp_rank=sender.tp_rank, layers=sorted(sender.memories))
        self.capture = BufferedCapture(header, self.snapshot, directory=directory)

    def snapshot(self):
        with self.lock:
            return dict(pending_jobs=len(self.jobs), pending_layers=len(self.layers))

    def __call__(self, endpoint, message, **kwargs):
        original = self.original()
        if original is None:
            raise RuntimeError("native Sender owner is no longer alive")
        operation = message.get("op")
        if operation not in ("plan", "pull", "complete", "status", "cancel"):
            return original(endpoint, message, **kwargs)
        is_pull = operation == "pull"
        begin = time.perf_counter_ns() if is_pull else 0
        try:
            result = original(endpoint, message, **kwargs)
        except BaseException:
            if is_pull:
                self._safe_observe(endpoint, message, None, begin, time.perf_counter_ns())
            raise
        end = time.perf_counter_ns() if is_pull else 0
        self._safe_observe(endpoint, message, result, begin, end)
        return result

    def _safe_observe(self, endpoint, message, result, begin, end):
        try:
            with self.lock:
                self._observe(endpoint, message, result, begin, end)
        except Exception:
            # Measurement failure must never change native DMA/error handling.
            self.capture.note_error()

    def _observe(self, endpoint, message, result, begin, end):
        operation = message.get("op")
        if operation == "plan":
            context = {
                key: message[key]
                for key in ("request", "generation", "layer", "first", "tokens", "final", "member", "ratio")
            }
            self.jobs[result["job"]] = dict(
                context,
                peer=endpoint,
                expected=dict(result["count"]),
                seen={"npu": set(), "cpu": set()},
                spans={"npu": None, "cpu": None},
                errors=[],
            )
        elif operation == "pull":
            job = self.jobs[message["job"]]
            path, batch = message["path"], message["batch"]
            if result is not True or batch in job["seen"][path]:
                job["errors"].append("failed_or_unobserved_pull_ack")
                return
            job["seen"][path].add(batch)
            old = job["spans"][path]
            job["spans"][path] = [min(begin, old[0]), max(end, old[1])] if old else [begin, end]
        elif operation == "complete" or operation == "status" and result.get("done"):
            self._finish(message["job"])
        elif operation == "cancel":
            request, generation = message["request"], message["generation"]
            self.capture.emit(dict(kind="cancel", request=request, generation=generation))
            self.jobs = {
                key: value
                for key, value in self.jobs.items()
                if (value["request"], value["generation"]) != (request, generation)
            }
            self.layers = {key: value for key, value in self.layers.items() if key[:2] != (request, generation)}

    def _finish(self, job_id):
        job = self.jobs.pop(job_id, None)
        if job is None:
            return  # Native completion/status acknowledgements are idempotent.
        for path in METRICS:
            if job["seen"][path] != set(range(job["expected"][path])):
                job["errors"].append("missing_pull_completion")
        key = job["request"], job["generation"], job["layer"]
        if key not in self.layers:
            self.layers[key] = dict(
                kind="layer",
                **{name: job[name] for name in ("request", "generation", "layer", "peer", "member", "ratio")},
                chunks=[],
                errors=[],
            )
        layer = self.layers[key]
        if any(layer[name] != job[name] for name in ("peer", "member", "ratio")):
            layer["errors"].append("changed_layer_contributor_or_destination")
        layer["chunks"].append(dict(first=job["first"], tokens=job["tokens"], final=job["final"], spans=job["spans"]))
        layer["errors"].extend(job["errors"])
        if job["final"]:
            layer["input_tokens"] = job["tokens"]
            self.capture.emit(layer)
            del self.layers[key]


class BatchObserver:
    """Count D scheduled logical request batches; never read device metadata."""

    def __init__(self, header, cpu_config, *, directory=CAPTURE_DIR):
        self.lock = threading.Lock()
        self.batches = Counter()
        self.capture = BufferedCapture(dict(header, cpu_config=cpu_config), self.snapshot, directory=directory)

    def observe(self, scheduler_output):
        size = len(scheduler_output.num_scheduled_tokens)
        with self.lock:
            self.batches[size] += 1

    def snapshot(self):
        with self.lock:
            return dict(batch_histogram=dict(self.batches))


def read_captures(paths):
    captures = []
    for path in sorted(set(Path(path).resolve() for path in paths)):
        header, checkpoint, rows = None, None, []
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except (ValueError, TypeError) as error:
                    raise ValueError(
                        f"{path}:{number}: incomplete/invalid JSON; wait for the background flush"
                    ) from error
                if record.get("kind") == "header":
                    if header is not None or record.get("schema") != SCHEMA_VERSION:
                        raise ValueError(f"{path}: duplicate header or unsupported capture schema")
                    header = record
                elif record.get("kind") == "checkpoint":
                    checkpoint = record
                else:
                    rows.append(record)
        if header is None or checkpoint is None:
            raise ValueError(f"{path}: missing header/checkpoint; wait at least {FLUSH_SECONDS:g}s after quiescence")
        captures.append((header, checkpoint, rows, str(path)))
    if not captures:
        raise ValueError("no capture files found")
    return captures


def percentile(values, quantile):
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution(values):
    values = sorted(value for value in values if value is not None)
    return dict(
        samples=len(values),
        mean_ms=statistics.fmean(values) if values else None,
        p90_ms=percentile(values, 0.90) if values else None,
        p99_ms=percentile(values, 0.99) if values else None,
    )


def merge_layer(records, input_tokens):
    """One chunk's TP branches share a clock: use their enclosing interval.

    Sum chunk intervals to exclude intervening P compute gaps. No transfer is
    represented as None, never as a fabricated zero-latency sample.
    """
    first = records[0]
    ratio = first["ratio"]
    if len(records) != ratio or {row["member"] for row in records} != set(range(ratio)):
        raise ValueError("missing/duplicate TP contributor")
    chunks = defaultdict(list)
    for row in records:
        if row["errors"] or row["ratio"] != ratio or row["peer"] != first["peer"]:
            raise ValueError("failed Pull, changed contributor count or changed D endpoint")
        boundaries = set()
        for chunk in row["chunks"]:
            boundary = chunk["first"], chunk["tokens"], chunk["final"]
            if boundary in boundaries:
                raise ValueError("duplicate chunk")
            boundaries.add(boundary)
            chunks[boundary].append(chunk)
    previous, final_count = 0, 0
    totals = {path: None for path in METRICS}
    for (begin, tokens, final), parts in sorted(chunks.items()):
        if len(parts) != ratio or begin != previous // 128 or not previous < tokens <= input_tokens:
            raise ValueError("missing, mismatched or out-of-order chunk")
        previous = tokens
        final_count += int(final)
        if final and tokens != input_tokens:
            raise ValueError("final chunk length does not match the workload")
        for path in METRICS:
            spans = [part["spans"][path] for part in parts if part["spans"][path] is not None]
            if spans:
                if any(type(start) is not int or type(end) is not int or not 0 < start <= end for start, end in spans):
                    raise ValueError("invalid monotonic interval")
                elapsed = max(end for _, end in spans) - min(start for start, _ in spans)
                totals[path] = (totals[path] or 0) + elapsed / 1_000_000
    if previous != input_tokens or final_count != 1:
        raise ValueError("missing final chunk")
    return dict(
        request=first["request"],
        generation=first["generation"],
        layer=first["layer"],
        peer=first["peer"],
        chunks=len(chunks),
        **{f"{metric}_ms": totals[path] for path, metric in METRICS.items()},
    )


def collect_samples(captures, input_tokens, request_ids=None, skip_first=0):
    problems, headers, raw = [], [], defaultdict(lambda: defaultdict(list))
    cancelled, observed_batches = set(), {}
    for header, checkpoint, rows, path in captures:
        if checkpoint["dropped"] or checkpoint["errors"]:
            problems.append(f"{path}: capture dropped records or had observer/writer errors")
        if checkpoint["enqueued"] != len(rows) or checkpoint["written"] != len(rows):
            problems.append(f"{path}: data/checkpoint mismatch; wait for quiescence and flush")
        if header["role"] == "P":
            headers.append(header)
            if checkpoint.get("pending_jobs") or checkpoint.get("pending_layers"):
                problems.append(f"{path}: unfinished layer/job measurements")
            for row in rows:
                key = row["request"], row["generation"]
                if row["kind"] == "cancel":
                    cancelled.add(key)
                elif row["kind"] == "layer":
                    raw[key][row["layer"]].append(row)
        elif header["role"] == "D_scheduler":
            rank = header["dp_rank"]
            if rank in observed_batches:
                problems.append(f"duplicate D scheduler rank {rank}; do not mix restarts")
            observed_batches[rank] = dict(header=header, histogram=checkpoint.get("batch_histogram", {}))
    if not headers:
        raise ValueError("no P worker captures")
    if len({(header["host"], header["boot_id"], header["engine"]) for header in headers}) != 1:
        raise ValueError("P samples must come from one host/boot/engine; cross-host clocks cannot be merged")
    layer_names = set(headers[0]["layers"])
    if any(set(header["layers"]) != layer_names for header in headers):
        raise ValueError("P workers disagree on the layer set")
    candidates = []
    for key, layers in raw.items():
        rows = [row for group in layers.values() for row in group]
        if request_ids is not None and key[0] not in request_ids:
            continue
        if any(row["input_tokens"] != input_tokens for row in rows):
            continue
        starts = [span[0] for row in rows for chunk in row["chunks"] for span in chunk["spans"].values() if span]
        candidates.append((min(starts, default=0), key, layers))
    layer_samples, request_samples, rejected = [], [], []
    for _, key, layers in sorted(candidates)[skip_first:]:
        try:
            if key in cancelled or set(layers) != layer_names:
                raise ValueError("cancelled request or missing layer")
            merged = [merge_layer(rows, input_tokens) for rows in layers.values()]
            peers = {row["peer"] for row in merged}
            if len(peers) != 1:
                raise ValueError("request changed D endpoint")
            request = dict(request=key[0], generation=key[1], peer=next(iter(peers)), layers=len(merged))
            for metric in METRICS.values():
                times = [row[f"{metric}_ms"] for row in merged if row[f"{metric}_ms"] is not None]
                request[f"{metric}_ms"] = sum(times) if times else None
                request[f"{metric}_layers"] = len(times)
            layer_samples.extend(merged)
            request_samples.append(request)
        except (KeyError, TypeError, ValueError) as error:
            rejected.append(dict(request=key[0], generation=key[1], reason=str(error)))
    return layer_samples, request_samples, rejected, observed_batches, problems


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare_performance(argv):
    parser = argparse.ArgumentParser(
        description="Compare metrics exported by your existing request tool; sends no requests"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sampled", type=Path, required=True)
    parser.add_argument("--max-regression-percent", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    directions = {
        "request_throughput_rps": True,
        "output_throughput_tps": True,
        "mean_ttft_ms": False,
        "p99_ttft_ms": False,
        "mean_tpot_ms": False,
        "p99_tpot_ms": False,
    }
    try:
        if not math.isfinite(args.max_regression_percent) or args.max_regression_percent < 0:
            raise ValueError("regression threshold must be finite and nonnegative")
        baseline, sampled = json.loads(args.baseline.read_text()), json.loads(args.sampled.read_text())
        if baseline.keys() != sampled.keys() or baseline.keys() - directions.keys():
            raise ValueError("baseline/sampled must contain the same supported metric names")
        if not any(directions[key] for key in baseline) or not any(not directions[key] for key in baseline):
            raise ValueError("export at least one throughput and one latency metric")
        comparisons = []
        for key, value in baseline.items():
            other = sampled[key]
            if any(
                type(number) not in (int, float) or not math.isfinite(number) or number <= 0
                for number in (value, other)
            ):
                raise ValueError("performance values must be finite and positive")
            regression = (value - other if directions[key] else other - value) / value * 100
            comparisons.append(dict(metric=key, baseline=value, sampled=other, regression_percent=regression))
        passed = all(row["regression_percent"] <= args.max_regression_percent for row in comparisons)
        report = dict(
            status="PASS" if passed else "FAIL", threshold_percent=args.max_regression_percent, metrics=comparisons
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0 if passed else 2
    except (OSError, AttributeError, TypeError, ValueError) as error:
        print(f"Cannot compare performance: {error}", file=sys.stderr)
        return 2


def main(argv=None):
    """Validate complete cohorts and workload coverage before publishing reports.

    Invalid runs still retain their partial CSVs for diagnosis, with an explicit
    INCOMPLETE status and nonzero exit code instead of a successful full-run claim.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        return compare_performance(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path, help="P/D capture directories or .jsonl files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-requests", type=int, required=True, help="measured request count from your load tool"
    )
    parser.add_argument("--input-tokens", type=int, default=131072)
    parser.add_argument(
        "--expected-output-tokens", type=int, default=125, help="declared workload; checked by your load tool"
    )
    parser.add_argument("--request-ids", type=Path, help="optional exact successful external request IDs, one per line")
    parser.add_argument(
        "--skip-first", type=int, default=0, help="exclude this many earliest matching requests as warmup"
    )
    parser.add_argument("--expected-dp", type=int, default=8)
    parser.add_argument("--batch-target", type=int, default=16)
    args = parser.parse_args(argv)
    if (
        min(args.expected_requests, args.input_tokens, args.expected_output_tokens, args.expected_dp, args.batch_target)
        < 1
    ):
        parser.error("counts, token lengths, DP and batch target must be positive")
    if args.skip_first < 0:
        parser.error("--skip-first must be nonnegative")
    try:
        files = [
            path
            for source in args.captures
            for path in (sorted(source.rglob("*.jsonl")) if source.is_dir() else [source])
        ]
        captures = read_captures(files)
        ids = (
            {line.strip() for line in args.request_ids.read_text().splitlines() if line.strip()}
            if args.request_ids
            else None
        )
        layers, requests, rejected, batches, problems = collect_samples(
            captures, args.input_tokens, ids, args.skip_first
        )
        if len(requests) != args.expected_requests:
            problems.append(f"expected {args.expected_requests} complete requests, observed {len(requests)}")
        if rejected:
            problems.append(f"{len(rejected)} matching requests have incomplete/invalid transfer measurements")
        if set(batches) != set(range(args.expected_dp)):
            problems.append("missing or unexpected D scheduler DP ranks")
        for rank, sample in batches.items():
            if sample["header"]["max_num_seqs"] != args.batch_target:
                problems.append(f"D DP{rank}: configured max_num_seqs differs from target {args.batch_target}")
            if not int(sample["histogram"].get(str(args.batch_target), 0)):
                problems.append(f"D DP{rank}: target scheduled batch {args.batch_target} was not observed")
        summary = []
        for metric in METRICS.values():
            for scope, samples in (("request_layer", layers), ("request", requests)):
                row = dict(metric=metric, scope=scope, **distribution([row[f"{metric}_ms"] for row in samples]))
                summary.append(row)
                if not row["samples"]:
                    problems.append(f"{metric}/{scope}: no valid transfer samples")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            args.output_dir / "summary.csv", summary, ["metric", "scope", "samples", "mean_ms", "p90_ms", "p99_ms"]
        )
        write_csv(
            args.output_dir / "request_layers.csv",
            layers,
            ["request", "generation", "layer", "peer", "chunks", "t1_ms", "t2_ms"],
        )
        write_csv(
            args.output_dir / "requests.csv",
            requests,
            ["request", "generation", "peer", "layers", "t1_ms", "t1_layers", "t2_ms", "t2_layers"],
        )
        by_layer = []
        for layer in sorted({row["layer"] for row in layers}):
            for metric in METRICS.values():
                values = [row[f"{metric}_ms"] for row in layers if row["layer"] == layer]
                by_layer.append(dict(layer=layer, metric=metric, **distribution(values)))
        write_csv(
            args.output_dir / "by_layer.csv", by_layer, ["layer", "metric", "samples", "mean_ms", "p90_ms", "p99_ms"]
        )
        report = dict(
            status="INCOMPLETE" if problems else "OK",
            workload=dict(
                input_tokens=args.input_tokens,
                output_tokens_declared=args.expected_output_tokens,
                output_verified=False,
                output_verification="check the external load tool's successful request results",
            ),
            timing="P monotonic Pull RPC start through successful completion ACK; includes control/queue/wait",
            aggregation="TP interval envelope per chunk/path; sum chunks per layer; sum layers per request",
            percentile="linear interpolation on aggregated samples",
            no_transfer="excluded from layer percentiles; no fabricated zero sample",
            batch_histogram_scope="entire D scheduler capture, including warmup and unselected requests",
            request_success_scope="complete KV transfers; HTTP/output success is checked externally",
            summary=summary,
            rejected=rejected,
            problems=problems,
            d_scheduled_batches=batches,
            capture_files=[path for _, _, _, path in captures],
        )
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        for row in summary:
            print(
                f"{row['metric']} {row['scope']}: n={row['samples']} "
                f"mean_ms={row['mean_ms']} p90_ms={row['p90_ms']} p99_ms={row['p99_ms']}"
            )
        for problem in problems:
            print(f"INCOMPLETE: {problem}", file=sys.stderr)
        return 2 if problems else 0
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"Cannot produce validated transfer statistics: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
