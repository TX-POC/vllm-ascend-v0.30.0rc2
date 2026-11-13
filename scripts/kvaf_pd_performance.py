# SPDX-License-Identifier: Apache-2.0
"""Opt-in, request-only host RPC capture. No vLLM, SDK or device imports."""

from __future__ import annotations

import atexit
import dataclasses
import hashlib
import itertools
import json
import logging
import math
import threading
import time
import weakref
from pathlib import Path
from queue import Empty, Full, Queue
from uuid import uuid4

CONFIG_PATH = Path(__file__).with_name("kvaf_pd_performance.json")
SCHEMA = 3
# Conservative fixed charges for bounded, scalar-only records and job contexts.
RECORD_BYTES = 16384
MAX_TEXT = 256
FOOTER_BYTES = 65536
MAX_BATCH_SIZE = 4096
MAX_HISTOGRAM_KEYS = 512


@dataclasses.dataclass(frozen=True)
class BufferConfig:
    max_events: int = 2048
    max_bytes: int = 16777216
    overflow_policy: str = "drop_new"


@dataclasses.dataclass(frozen=True)
class WriterConfig:
    enabled: bool = True
    output_dir: str = "./pd-perf"
    batch_events: int = 256
    flush_interval_ms: int = 5000
    file_buffer_bytes: int = 1048576
    max_file_bytes: int = 67108864
    max_total_bytes: int = 268435456


@dataclasses.dataclass(frozen=True)
class CaptureConfig:
    enabled: bool = False
    sample_rate: float = 0.1
    sample_seed: int = 0
    detail_level: str = "layer"
    buffer: BufferConfig = dataclasses.field(default_factory=BufferConfig)
    writer: WriterConfig = dataclasses.field(default_factory=WriterConfig)

    @property
    def active(self):
        return self.enabled and self.sample_rate > 0

    def selects(self, request):
        # Only the existing external request ID is shared across TP/P/D. Do not
        # hash local aliases, random generations, Python hash() or timestamps.
        if not self.active:
            return False
        if self.sample_rate == 1:
            return True
        digest = hashlib.blake2b(f"{self.sample_seed}:{request}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") < self.sample_rate * 2**64


def _object(cls, value):
    if not isinstance(value, dict) or value.keys() - {field.name for field in dataclasses.fields(cls)}:
        raise ValueError(f"invalid or unknown {cls.__name__} fields")
    return cls(**value)


def load_config(path=CONFIG_PATH):
    path = Path(path)
    if not path.exists():
        return CaptureConfig()
    if path.stat().st_size > FOOTER_BYTES:
        raise ValueError("capture config exceeds 64 KiB")
    with path.open(encoding="utf-8") as stream:
        content = stream.read(FOOTER_BYTES + 1)
        if len(content) > FOOTER_BYTES:
            raise ValueError("capture config exceeds 64 KiB")
        raw = json.loads(content)
    if not isinstance(raw, dict):
        raise ValueError("capture config must be an object")
    raw = dict(raw)
    raw["buffer"] = _object(BufferConfig, raw.get("buffer", {}))
    raw["writer"] = _object(WriterConfig, raw.get("writer", {}))
    config = _object(CaptureConfig, raw)
    if type(config.enabled) is not bool or type(config.writer.enabled) is not bool:
        raise ValueError("enabled flags must be boolean")
    if type(config.sample_rate) not in (int, float) or not math.isfinite(config.sample_rate):
        raise ValueError("sample_rate must be finite")
    if not 0 <= config.sample_rate <= 1:
        raise ValueError("sample_rate must be in [0,1]")
    if type(config.sample_seed) is not int or not 0 <= config.sample_seed < 2**32:
        raise ValueError("sample_seed must be a uint32")
    if config.detail_level not in ("stage", "layer", "detail"):
        raise ValueError("detail_level must be stage, layer or detail")
    if config.buffer.overflow_policy not in ("drop_new", "stop_capture"):
        raise ValueError("overflow_policy must be drop_new or stop_capture")
    for group in (config.buffer, config.writer):
        for field in dataclasses.fields(group):
            value = getattr(group, field.name)
            if field.name not in ("enabled", "output_dir", "overflow_policy"):
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{field.name} must be a positive integer")
    if not isinstance(config.writer.output_dir, str) or not config.writer.output_dir.strip():
        raise ValueError("writer.output_dir must be a nonempty path")
    if config.buffer.max_bytes < 2 * RECORD_BYTES:
        raise ValueError("buffer.max_bytes must allow at least two 16 KiB slots")
    if config.writer.batch_events > config.buffer.max_events:
        raise ValueError("writer.batch_events must not exceed buffer.max_events")
    if config.writer.file_buffer_bytes == 1:
        raise ValueError("binary file buffering must be at least two bytes")
    if not 2 * FOOTER_BYTES <= config.writer.max_file_bytes <= config.writer.max_total_bytes:
        raise ValueError("128 KiB <= max_file_bytes <= max_total_bytes is required")
    return config


def _text(value):
    if not isinstance(value, str) or not value or len(value) > MAX_TEXT:
        raise ValueError("capture identity must be a nonempty string of at most 256 characters")
    return value


class Capture:
    """Bounded scalar records; all serialization, snapshots and I/O are off-thread.

    Half the configured record budget is reserved for in-flight job contexts.
    Queue capacity is conservative, not a promise about Python process RSS.
    """

    def __init__(self, config, header, snapshot):
        self.config, self.header, self.snapshot = config, dict(header), snapshot
        self.capacity = min(config.buffer.max_events, config.buffer.max_bytes // (2 * RECORD_BYTES))
        self.queue = Queue(maxsize=self.capacity)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.closed = False
        self.enqueued = self.written = self.dropped = self.errors = self.bytes_written = 0
        self.reason = ""
        self.identity = uuid4().hex
        self.paths = []
        self.thread = None
        if config.active and config.writer.enabled:
            self.thread = threading.Thread(target=self._run, name="kvaf-pd-perf-writer", daemon=True)
            self.thread.start()
        atexit.register(self.close)

    def fail(self, reason):
        self.errors += 1
        if not self.reason:
            self.reason = reason
        self.stop.set()

    def drop(self):
        self.dropped += 1
        if self.config.buffer.overflow_policy == "stop_capture":
            self.fail("capture buffer/state budget exceeded")

    def emit(self, record):
        if self.stop.is_set():
            self.dropped += 1
            return
        if not self.lock.acquire(blocking=False):
            self.drop()
            return
        try:
            try:
                self.queue.put_nowait(record)
                self.enqueued += 1
            except Full:
                self.drop()
        finally:
            self.lock.release()

    def _run(self):
        writer = self.config.writer
        stream, part, file_bytes = None, 0, 0
        directory = Path(writer.output_dir) / self.identity

        def write(record):
            nonlocal file_bytes
            data = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
            if (
                file_bytes + len(data) > writer.max_file_bytes
                or self.bytes_written + len(data) > writer.max_total_bytes
            ):
                raise ValueError("capture file budget exhausted")
            stream.write(data)
            file_bytes += len(data)
            self.bytes_written += len(data)

        def checkpoint(closed=False):
            with self.lock:
                counts = dict(
                    enqueued=self.enqueued,
                    written=self.written,
                    dropped=self.dropped,
                    errors=self.errors,
                    queued_events=self.queue.qsize(),
                    queue_capacity=self.capacity,
                )
            state = self.snapshot()
            write(dict(kind="checkpoint", closed=closed, reason=self.reason, **counts, **state))
            stream.flush()

        def open_part():
            nonlocal stream, file_bytes
            file_bytes = 0
            path = directory / f"events-{part:05d}.jsonl"
            stream = path.open("xb", buffering=writer.file_buffer_bytes)
            self.paths.append(path)
            write(
                dict(
                    self.header,
                    kind="header",
                    schema=SCHEMA,
                    capture_id=self.identity,
                    part=part,
                    config=dataclasses.asdict(self.config),
                    clock="perf_counter_ns",
                    created_ns=time.time_ns(),
                )
            )

        try:
            directory.mkdir(parents=True, exist_ok=False)
            open_part()
            deadline = time.monotonic() + writer.flush_interval_ms / 1000
            batch = 0
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    record = self.queue.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
                except Empty:
                    record = None
                if record is not None:
                    # Scalar schema bounds the serialized record below one charge.
                    if file_bytes + RECORD_BYTES + FOOTER_BYTES > writer.max_file_bytes:
                        checkpoint(closed=True)
                        stream.close()
                        part += 1
                        if self.bytes_written + 2 * FOOTER_BYTES > writer.max_total_bytes:
                            self.fail("total capture file budget exhausted")
                            break
                        open_part()
                    if self.bytes_written + RECORD_BYTES + FOOTER_BYTES > writer.max_total_bytes:
                        self.fail("total capture file budget exhausted")
                        break
                    write(record)
                    with self.lock:
                        self.written += 1
                    batch += 1
                if batch >= writer.batch_events or time.monotonic() >= deadline:
                    checkpoint()
                    batch = 0
                    deadline = time.monotonic() + writer.flush_interval_ms / 1000
            checkpoint(closed=True)
        except Exception as error:
            self.fail(f"writer failure: {type(error).__name__}: {str(error)[:160]}")
        finally:
            if stream is not None:
                stream.close()
            if self.reason:
                logging.getLogger(__name__).warning("KVAF performance capture stopped: %s", self.reason)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=self.config.writer.flush_interval_ms / 1000 + 1)
            if self.thread.is_alive():
                self.fail("writer did not drain within close budget")
        elif self.config.active:
            logging.getLogger(__name__).warning(
                "KVAF memory-only capture: queued=%s dropped=%s; no trace files", self.enqueued, self.dropped
            )
        atexit.unregister(self.close)


class SenderObserver:
    """Decorate only this Sender's control calls; never retain its SDK or Tensors."""

    def __init__(self, sender, config, header):
        original = sender._call
        self.original = weakref.WeakMethod(original)
        self.config = config
        self.lock = threading.Lock()
        self.jobs = {}
        self.rpc_sequence = itertools.count()
        self.capture = Capture(
            config, dict(header, tp_rank=sender.tp_rank, layers=sorted(sender.memories)), self.snapshot
        )
        self.job_limit = self.capture.capacity

    def snapshot(self):
        with self.lock:
            return dict(pending_jobs=len(self.jobs))

    def __call__(self, endpoint, message, **kwargs):
        original = self.original()
        if original is None:
            raise RuntimeError("native Sender owner is no longer alive")
        operation = message.get("op")
        if operation not in ("plan", "pull", "complete", "status", "cancel"):
            return original(endpoint, message, **kwargs)
        context = None
        if not self.capture.stop.is_set():
            try:
                if operation in ("plan", "cancel") and self.config.selects(message["request"]):
                    context = {key: message[key] for key in ("request", "generation")}
                    if operation == "plan" and self.config.detail_level != "stage":
                        context.update(
                            {key: message[key] for key in ("layer", "first", "tokens", "final", "member", "ratio")}
                        )
                    for key in ("request", "generation", "layer"):
                        if key in context:
                            _text(context[key])
                    _text(endpoint)
                elif self.lock.acquire(blocking=False):
                    try:
                        context = self.jobs.get(message.get("job"))
                    finally:
                        self.lock.release()
                else:
                    self.capture.drop()
            except Exception:
                self.capture.fail("request identity/observer metadata invalid")
        if context is None:
            return original(endpoint, message, **kwargs)
        begin = time.perf_counter_ns()
        span_id = f"{self.capture.identity}:rpc:{next(self.rpc_sequence)}"
        try:
            # Nonterminal status polling is detail-only. A begin record without
            # its deliberately omitted end would falsely look like a stalled RPC.
            if operation != "status" or self.config.detail_level == "detail":
                identity = context
                if self.config.detail_level == "stage":
                    identity = {key: context[key] for key in ("request", "generation")}
                self.capture.emit(
                    dict(
                        identity,
                        kind="rpc_begin",
                        name=f"connector.{operation}",
                        path=message.get("path"),
                        batch=message.get("batch"),
                        job=message.get("job"),
                        peer=endpoint,
                        span_id=span_id,
                        start_ns=begin,
                    )
                )
        except Exception:
            self.capture.fail("request begin observation failed; native call preserved")
        try:
            result = original(endpoint, message, **kwargs)
        except BaseException:
            self._observe(endpoint, message, context, None, begin, time.perf_counter_ns(), span_id)
            raise
        self._observe(endpoint, message, context, result, begin, time.perf_counter_ns(), span_id)
        return result

    def _observe(self, endpoint, message, context, result, begin, end, span_id):
        try:
            operation = message["op"]
            success = result is True
            job_id = message.get("job")
            if operation == "plan" and isinstance(result, dict):
                job_id = _text(result["job"])
                success = True
            if not self.lock.acquire(blocking=False):
                self.capture.drop()
                return
            try:
                if operation == "plan" and success:
                    if len(self.jobs) >= self.job_limit:
                        self.capture.drop()
                        return
                    self.jobs[job_id] = dict(context)
                elif operation == "cancel":
                    self.jobs = {
                        job: value
                        for job, value in self.jobs.items()
                        if (value["request"], value["generation"]) != (context["request"], context["generation"])
                    }
                elif (
                    operation == "complete"
                    and success
                    or operation == "status"
                    and isinstance(result, dict)
                    and result.get("done")
                ):
                    self.jobs.pop(job_id, None)
            finally:
                self.lock.release()
            record = dict(
                context,
                kind=operation,
                peer=endpoint,
                job=job_id,
                start_ns=begin,
                end_ns=end,
                success=success,
                span_id=span_id,
            )
            if operation == "plan" and success and self.config.detail_level != "stage":
                record["count"] = {path: result["count"][path] for path in ("npu", "cpu")}
            elif operation == "pull":
                record.update(path=message["path"], batch=message["batch"])
            elif operation == "status":
                record["done"] = isinstance(result, dict) and result.get("done") is True
                record["success"] = isinstance(result, dict) and not result.get("error")
            if operation == "status" and not record.get("done") and self.config.detail_level != "detail":
                return
            if self.config.detail_level == "stage":
                record["kind"] = "stage"
                record["stage"] = f"connector.{operation}"
                for key in ("layer", "first", "tokens", "final", "member", "ratio", "count"):
                    record.pop(key, None)
            self.capture.emit(record)
        except Exception:
            self.capture.fail("request observer failed; native call preserved")


class BatchObserver:
    """Whole-window logical batch counts, never a selected-request batch claim."""

    def __init__(self, config, header):
        self.lock = threading.Lock()
        self.batches = {}
        self.capture = Capture(config, dict(header, role="D_scheduler"), self.snapshot)

    def observe(self, output):
        if self.capture.stop.is_set():
            return
        try:
            size = len(output.num_scheduled_tokens)
            if not 0 <= size <= MAX_BATCH_SIZE:
                self.capture.fail("batch histogram exceeds supported bound")
            elif self.lock.acquire(blocking=False):
                try:
                    if size not in self.batches and len(self.batches) >= MAX_HISTOGRAM_KEYS:
                        self.capture.fail("batch histogram key budget exceeded")
                    else:
                        self.batches[size] = self.batches.get(size, 0) + 1
                finally:
                    self.lock.release()
            else:
                self.capture.drop()
        except Exception:
            self.capture.fail("batch observation failed; native scheduling preserved")

    def snapshot(self):
        with self.lock:
            return dict(batch_histogram=dict(self.batches))
