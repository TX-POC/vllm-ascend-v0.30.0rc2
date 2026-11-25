# SPDX-License-Identifier: Apache-2.0
"""Opt-in benchmark cohort gate; CPU-only control, never transfers KV data."""

import asyncio
import dataclasses
import json
import logging
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import Future
from contextlib import suppress
from pathlib import Path

CONFIG_FILE_NAME = "kvaf_pd_decode_batch.json"
HEADER = "X-KVAF-Decode-Batch"
PARAM = "kvaf_decode_batch"
MAX_MESSAGE_BYTES = 1048576
MAX_GROUPS = 1024
MAX_DP = 64
MAX_BATCH = 512
MAX_TIMEOUT = 86400
SOCKET_GRACE = 5
CONTROL_PATH = "/v1/kvaf-decode-batch"
READY_PATH = CONTROL_PATH + "/ready"
SCHEDULED_PATH = CONTROL_PATH + "/scheduled"
HOT_KV_PATH = CONTROL_PATH + "/hot-kv"
HOT_KV_STATS = "hot_kv_stats"
HOT_KV_BUCKETS = 1001  # 0.0% through 100.0%, in 0.1 percentage-point increments.
HOT_KV_CAPABILITY = "hot_kv_reporting"
DEFAULT_TIMEOUT = 3600
METASERVER_PATH = "/v1/metaserver"


def decoder_capacity(config):
    parallel = config.parallel_config
    scheduler = config.scheduler_config
    speculative = config.speculative_config
    width = 1 if speculative is None else 1 + speculative.num_speculative_tokens
    capacity = min(MAX_BATCH, scheduler.max_num_seqs, scheduler.max_num_batched_tokens // width)
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("synchronized Decode requires D TP1 and PP1")
    if not 1 <= parallel.data_parallel_size <= MAX_DP or capacity < 1:
        raise ValueError("D scheduler has unsupported DP or no capacity for synchronized Decode")
    return dict(dp_size=parallel.data_parallel_size, max_batch_size=capacity)


class DecoderCapacityMiddleware:
    """Use the native serving middleware extension to expose the D scheduler limits."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "GET" or scope["path"] != CONTROL_PATH + "/capacity":
            await self.app(scope, receive, send)
            return

        from vllm import envs
        from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware

        args = scope["app"].state.args
        tokens = [key for key in (args.api_key or [envs.VLLM_API_KEY]) if key]
        # User middleware is outside the native authentication layer.
        handler = AuthenticationMiddleware(self._capacity, tokens=tokens) if tokens else self._capacity
        await handler(scope, receive, send)

    @staticmethod
    async def _capacity(scope, receive, send):
        from starlette.responses import JSONResponse

        config = scope["app"].state.vllm_config
        transfer = config.kv_transfer_config
        if transfer is None or transfer.kv_connector != "PDKVoffloadConnector" or transfer.kv_role != "kv_consumer":
            response = JSONResponse({"error": "synchronized Decode requires the KVAF PD consumer"}, status_code=409)
        else:
            try:
                response = JSONResponse({**decoder_capacity(config), HOT_KV_CAPABILITY: True})
            except ValueError as error:
                response = JSONResponse({"error": str(error)}, status_code=409)
        await response(scope, receive, send)


def validate_proxy_url(value):
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
        or any(char.isspace() for char in value)
    ):
        raise ValueError("decode batch proxy_url must be the existing Proxy base URL, e.g. http://141.62.38.12:8000")
    if parsed.port is not None and not 0 < parsed.port < 65536:
        raise ValueError("invalid decode batch Proxy port")
    return value.rstrip("/")


def proxy_from_metaserver(url):
    if not isinstance(url, str) or not url.endswith(METASERVER_PATH):
        raise ValueError("synchronized Decode requires the existing Proxy metaserver URL")
    return validate_proxy_url(url.removesuffix(METASERVER_PATH))


def validate_spec(spec):
    fields = {"group", "dp_size", "batch_size", "rank", "index", "phase"}
    if not isinstance(spec, dict) or spec.keys() - {HOT_KV_STATS} != fields:
        raise ValueError("decode batch requires exact group/dp_size/batch_size/rank/index/phase fields")
    group = spec["group"]
    if (
        not isinstance(group, str)
        or not 1 <= len(group) <= 128
        or not all(c.isascii() and (c.isalnum() or c in "-_") for c in group)
    ):
        raise ValueError("invalid decode batch group identity")
    for name, lower, upper in (("dp_size", 1, MAX_DP), ("batch_size", 1, MAX_BATCH)):
        if type(spec[name]) is not int or not lower <= spec[name] <= upper:
            raise ValueError(f"decode batch {name} must be in {lower}..{upper}")
    for name, upper in (("rank", spec["dp_size"]), ("index", spec["batch_size"])):
        if type(spec[name]) is not int or not 0 <= spec[name] < upper:
            raise ValueError(f"decode batch {name} out of range")
    if spec["phase"] not in ("measure", "warmup"):
        raise ValueError("decode batch supports separate measure/warmup waves only")
    hot = spec.get(HOT_KV_STATS, False)
    if type(hot) is not bool or (hot and spec["phase"] != "measure"):
        raise ValueError("Hot KV stats require a boolean flag on a measure wave")
    result = dict(spec)
    if not hot:
        result.pop(HOT_KV_STATS, None)
    return result


@dataclasses.dataclass(frozen=True)
class Settings:
    enabled: bool = False
    proxy_url: str = ""
    dp_size: int | None = None
    max_batch_size: int | None = None
    timeout_seconds: int = DEFAULT_TIMEOUT

    def with_capacity(self, capacity):
        dp = capacity.get("dp_size")
        batch = capacity.get("max_batch_size")
        if type(dp) is not int or not 1 <= dp <= MAX_DP or type(batch) is not int or not 1 <= batch <= MAX_BATCH:
            raise ValueError("invalid D capacity response")
        if self.dp_size is not None and self.dp_size != dp:
            raise ValueError(f"Proxy dp_size={self.dp_size} differs from D dp_size={dp}")
        if self.max_batch_size is not None and self.max_batch_size > batch:
            raise ValueError(f"Proxy max_batch_size={self.max_batch_size} exceeds D scheduler capacity={batch}")
        return dataclasses.replace(self, dp_size=dp, max_batch_size=self.max_batch_size or batch)

    def validate(self, spec, rank=None):
        result = validate_spec(spec)
        if result["dp_size"] != self.dp_size or result["batch_size"] > self.max_batch_size:
            raise ValueError("decode batch wave exceeds configured DP/batch capacity")
        if rank is not None and result["rank"] != rank:
            raise ValueError("decode batch reached the wrong D rank; verify native X-data-parallel-rank routing")
        return result

    def parse_header(self, value):
        if not value or len(value) > 4096:
            raise ValueError("decode batch mode requires a bounded X-KVAF-Decode-Batch header")
        return self.validate(json.loads(value))


def load_settings(path):
    path = Path(path)
    if not path.exists():
        return Settings()
    if path.stat().st_size > 65536:
        raise ValueError("decode batch config exceeds 64 KiB")
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "coordinator" in raw:
        raise ValueError("replace retired coordinator with proxy_url=http://<existing Proxy host>:<port>")
    if not isinstance(raw, dict) or raw.keys() - {f.name for f in dataclasses.fields(Settings)}:
        raise ValueError("unknown decode batch config fields")
    config = Settings(**raw)
    if type(config.enabled) is not bool or not isinstance(config.proxy_url, str):
        raise ValueError("decode batch enabled must be boolean and proxy_url must be a string")
    for name, upper in (("dp_size", MAX_DP), ("max_batch_size", MAX_BATCH), ("timeout_seconds", MAX_TIMEOUT)):
        value = getattr(config, name)
        if value is None and name != "timeout_seconds":
            continue
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"decode batch {name} must be in 1..{upper}")
    if config.proxy_url:
        validate_proxy_url(config.proxy_url)
    return config


class DirectControlRequest(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def exchange(proxy_url, message, timeout, *, path=READY_PATH):
    payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("decode batch message budget exceeded")
    request = urllib.request.Request(
        proxy_url + path,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    # Internal control traffic follows the existing direct metaserver route.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), DirectControlRequest())
    try:
        with opener.open(request, timeout=timeout + SOCKET_GRACE) as response:
            data = response.read(MAX_MESSAGE_BYTES + 1)
    except urllib.error.HTTPError as error:
        detail = error.read(MAX_MESSAGE_BYTES + 1)
        raise RuntimeError(f"Proxy decode batch HTTP {error.code}: {detail[:4096].decode(errors='replace')}") from error
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("decode batch response budget exceeded")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError("decode batch response must be an object")
    status = "released" if path == READY_PATH else "recorded"
    if result.get("status") != status or result.get("group") != message["spec"]["group"]:
        raise RuntimeError("decode batch was not released: " + str(result.get("error", result)))
    return result


class Registry:
    """Proxy-owned async group outcomes; no extra listener or worker-pool waits."""

    def __init__(self, settings, *, hot_kv_stats=False):
        self.settings = settings
        self.hot_kv_stats = hot_kv_stats
        self.condition = asyncio.Condition()
        self.groups = {}
        self.closed = False

    @staticmethod
    def _contract(spec):
        return spec["dp_size"], spec["batch_size"], spec["phase"], spec.get(HOT_KV_STATS, False)

    async def accept(self, spec, external_id):
        if spec.get(HOT_KV_STATS) and not self.hot_kv_stats:
            raise ValueError("D does not advertise cohort Hot KV stats support")
        identity = spec["group"]
        contract = self._contract(spec)
        async with self.condition:
            if self.closed:
                raise RuntimeError("Proxy decode batch registry is closed")
            if identity not in self.groups:
                if len(self.groups) >= MAX_GROUPS:
                    raise RuntimeError("decode batch group budget exhausted; restart Proxy between experiments")
                if any(not (group["released"] or group["error"]) for group in self.groups.values()):
                    raise RuntimeError("synchronized Decode waves must not overlap")
                self.groups[identity] = dict(
                    contract=contract,
                    slots={},
                    ranks={},
                    scheduled={},
                    deadline=time.monotonic() + self.settings.timeout_seconds,
                    error="",
                    released=False,
                )
            group = self.groups[identity]
            slot = spec["rank"], spec["index"]
            if group["contract"] != contract or slot in group["slots"]:
                raise ValueError("conflicting decode batch contract or duplicate slot")
            if group["released"] or group["error"] or time.monotonic() >= group["deadline"]:
                raise RuntimeError("decode batch group no longer accepts requests")
            group["slots"][slot] = external_id

    async def abort(self, spec, reason):
        async with self.condition:
            group = self.groups.get(spec["group"])
            if group is not None and not (group["released"] or group["error"]):
                group["error"] = reason + self._missing_ranks(group)
                self.condition.notify_all()

    @staticmethod
    def _missing_ranks(group):
        missing = sorted(set(range(group["contract"][0])) - group["ranks"].keys())
        return f"; D ranks without a complete KV-ready batch: {missing} (target {group['contract'][1]} per rank)"

    async def status(self, identity):
        async with self.condition:
            group = self.groups.get(identity)
            if group is None:
                raise KeyError(identity)
            if not (group["released"] or group["error"]) and time.monotonic() >= group["deadline"]:
                group["error"] = "decode batch deadline expired" + self._missing_ranks(group)
                self.condition.notify_all()
            dp, batch, _, hot = group["contract"]
            result = dict(
                released=group["released"],
                failure=group["error"],
                target_batch_per_rank=batch,
                ranks={
                    rank: dict(kv_ready=len(group["ranks"].get(rank, ())), first_step=group["scheduled"].get(rank))
                    for rank in range(dp)
                },
            )
            if hot:
                counts = group.get("hot_kv_samples", {})
                result["hot_kv"] = {
                    "histogram": list(group.get("hot_kv_histogram", ())),
                    "rank_samples": dict(counts),
                }
            return result

    async def hot_kv(self, message):
        spec = self.settings.validate(message["spec"])
        histogram = message["histogram"]
        if (
            not isinstance(histogram, list)
            or len(histogram) != HOT_KV_BUCKETS
            or any(type(count) is not int or not 0 <= count <= 2**31 - 1 for count in histogram)
        ):
            raise ValueError("invalid Hot KV histogram")
        async with self.condition:
            group = self.groups.get(spec["group"])
            if (
                group is None
                or not group["released"]
                or not group["contract"][-1]
                or group["contract"] != self._contract(spec)
            ):
                raise RuntimeError("Hot KV stats require a released wave with collection enabled")
            rank = spec["rank"]
            counts = group.setdefault("hot_kv_samples", {})
            if rank in counts:
                raise RuntimeError("duplicate Hot KV report from D rank")
            counts[rank] = sum(histogram)
            if "hot_kv_histogram" not in group:
                group["hot_kv_histogram"] = histogram.copy()
            else:
                group["hot_kv_histogram"] = [a + b for a, b in zip(group["hot_kv_histogram"], histogram)]
        return dict(status="recorded", group=spec["group"])

    async def scheduled(self, message):
        spec = self.settings.validate(message["spec"])
        cohort, total = message["cohort_requests"], message["scheduled_requests"]
        if type(cohort) is not int or type(total) is not int or not 1 <= cohort <= min(spec["batch_size"], total):
            raise ValueError("invalid Decode first-step counts")
        report = dict(cohort_requests=cohort, scheduled_requests=total)
        async with self.condition:
            group = self.groups.get(spec["group"])
            if group is None or not group["released"] or group["contract"] != self._contract(spec):
                raise RuntimeError("cannot report first-step counts for an unreleased or conflicting decode batch")
            previous = group["scheduled"].setdefault(spec["rank"], report)
            if previous != report:
                raise RuntimeError("conflicting Decode first-step counts")
        return dict(status="recorded", group=spec["group"])

    async def arrive(self, message):
        spec = self.settings.validate(message["spec"])
        timeout = message["timeout"]
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= MAX_TIMEOUT:
            raise ValueError("invalid decode batch timeout")
        identity = spec["group"]
        contract = self._contract(spec)
        async with self.condition:
            if self.closed:
                raise RuntimeError("Proxy decode batch registry is closed")
            group = self.groups.get(identity)
            if group is None:
                raise RuntimeError("unknown decode batch group")
            if group["error"]:
                raise RuntimeError(group["error"])
            if group["released"]:
                # A terminal release cannot be revoked by a late duplicate/abort.
                if group["contract"] != contract:
                    raise RuntimeError("conflicting decode batch contract after release")
                if message.get("op") == "ready":
                    members = message.get("members")
                    if not isinstance(members, list) or group["ranks"].get(spec["rank"]) != sorted(members):
                        raise RuntimeError("conflicting decode batch membership after release")
                if message.get("op") not in ("ready", "abort"):
                    raise RuntimeError("unknown decode batch operation")
                return self._release_result(spec)
            if group["contract"] != contract:
                group["error"] = "conflicting decode batch contract"
            if not group["released"] and time.monotonic() >= group["deadline"]:
                group["error"] = "decode batch deadline expired before final arrival" + self._missing_ranks(group)
            if message.get("op") == "abort":
                group["error"] = "a D participant aborted this decode batch" + self._missing_ranks(group)
            elif message.get("op") != "ready":
                group["error"] = "unknown decode batch operation"
            else:
                members = message.get("members")
                valid = isinstance(members, list) and len(members) == spec["batch_size"]
                valid = valid and all(
                    isinstance(m, list)
                    and len(m) == 3
                    and type(m[0]) is int
                    and isinstance(m[1], str)
                    and 0 < len(m[1]) <= 256
                    and isinstance(m[2], str)
                    and 0 < len(m[2]) <= 256
                    for m in members
                )
                if valid:
                    indices = {m[0] for m in members}
                    valid = len(indices) == spec["batch_size"] and all(
                        group["slots"].get((spec["rank"], index)) == request for index, request, _ in members
                    )
                if not valid:
                    group["error"] = "missing/invalid local ready membership"
                else:
                    rank = spec["rank"]
                    members = sorted(members)
                    previous = group["ranks"].get(rank)
                    if previous is not None and previous != members:
                        group["error"] = "D rank attempted to replace an existing batch"
                    group["ranks"][rank] = members
                    names = [m[1] for batch in group["ranks"].values() for m in batch]
                    if len(names) != len(set(names)):
                        group["error"] = "duplicate request across decode batch slots"
                    group["deadline"] = min(group["deadline"], time.monotonic() + timeout)
                    if len(group["ranks"]) == spec["dp_size"] and not group["error"]:
                        group["released"] = True
            self.condition.notify_all()
            while not group["released"] and not group["error"]:
                remaining = group["deadline"] - time.monotonic()
                if remaining <= 0:
                    group["error"] = "timed out waiting for KV-ready requests" + self._missing_ranks(group)
                    self.condition.notify_all()
                    break
                try:
                    await asyncio.wait_for(self.condition.wait(), remaining)
                except asyncio.TimeoutError:
                    if group["released"] or group["error"]:
                        break
                    group["error"] = "timed out waiting for KV-ready requests" + self._missing_ranks(group)
                    self.condition.notify_all()
                except asyncio.CancelledError:
                    if not group["released"]:
                        group["error"] = "D ready request disconnected or cancelled before release"
                        self.condition.notify_all()
                    raise
            if group["error"]:
                raise RuntimeError(group["error"])
            return self._release_result(spec)

    @staticmethod
    def _release_result(spec):
        return dict(
            status="released", group=spec["group"], ranks=spec["dp_size"], requests=spec["dp_size"] * spec["batch_size"]
        )

    async def close(self):
        async with self.condition:
            self.closed = True
            for group in self.groups.values():
                if not group["released"]:
                    group["error"] = "Proxy stopped before decode batch release"
            self.condition.notify_all()


@dataclasses.dataclass
class HotKVWave:
    spec: dict
    proxy_url: str
    remaining: int
    histogram: list[int] = dataclasses.field(default_factory=lambda: [0] * HOT_KV_BUCKETS)

    def record_miss_percent(self, miss_percent):
        bucket = int((100.0 - miss_percent) * 10 + 0.5)
        self.histogram[bucket] += 1

    def send(self):
        try:
            exchange(self.proxy_url, dict(spec=self.spec, histogram=self.histogram), 5, path=HOT_KV_PATH)
        except (OSError, RuntimeError, ValueError) as error:
            logging.getLogger("vllm").warning(
                "Cannot report Hot KV stats for group=%s rank=%s: %s", self.spec["group"], self.spec["rank"], error
            )


class HotKVWaves:
    """Resolve existing feedback rows to a bounded histogram for each opt-in wave."""

    def __init__(self):
        self.requests = {}
        self.waves = {}

    def bind(self, request_id, seed):
        if request_id in self.requests:
            return
        spec = seed.decode_batch
        group = spec["group"]
        if group not in self.waves:
            self.waves[group] = HotKVWave(spec, seed.batch_proxy_url, spec["batch_size"])
        self.requests[request_id] = self.waves[group]

    def finish(self, request_ids, executor):
        for request_id in request_ids:
            wave = self.requests.pop(request_id, None)
            if wave is None:
                continue
            wave.remaining -= 1
            if wave.remaining == 0:
                del self.waves[wave.spec["group"]]
                # The FIFO feedback executor has applied every earlier row before
                # handing off this immutable histogram to a separate HTTP thread.
                executor.submit(self._send_after_feedback, wave)

    @staticmethod
    def _send_after_feedback(wave):
        threading.Thread(target=wave.send, name="kvaf-hot-kv-report", daemon=True).start()


class Gate:
    """Hold only native completion IDs; all network waiting is off the worker loop."""

    def __init__(self, settings, rank, observe=None, logger=None):
        self.settings, self.rank, self.observe = settings, rank, observe
        self.logger = logger or logging.getLogger(__name__)
        self.groups, self.requests = {}, {}
        self.closed = False

    def bind(self, request, external, seed):
        if self.closed:
            raise RuntimeError("decode batch gate is closed")
        spec = self.settings.validate(seed.decode_batch, self.rank)
        identity = spec["group"]
        if request in self.requests:
            previous = self.requests[request]
            if previous != (identity, external, seed.generation, spec["index"]):
                raise RuntimeError("decode batch request identity changed")
            return
        if identity not in self.groups:
            if any(not group["released"] for group in self.groups.values()) or len(self.groups) >= 2:
                raise RuntimeError("decode batch waves must not overlap on a D rank")
            self.groups[identity] = dict(
                spec=spec,
                deadline=time.monotonic() + seed.batch_timeout,
                proxy_url=seed.batch_proxy_url,
                timeout=seed.batch_timeout,
                members={},
                ready=set(),
                future=None,
                released=False,
                tokens={},
            )
        group = self.groups[identity]
        if (
            group["proxy_url"] != seed.batch_proxy_url
            or group["timeout"] != seed.batch_timeout
            or any(
                group["spec"].get(name) != spec.get(name) for name in ("dp_size", "batch_size", "phase", HOT_KV_STATS)
            )
        ):
            raise RuntimeError("decode batch contract changed")
        if spec["index"] in group["members"]:
            raise RuntimeError("duplicate decode batch slot")
        group["members"][spec["index"]] = request
        self.requests[request] = identity, external, seed.generation, spec["index"]

    def _report(self, group, members, future):
        try:
            result = exchange(
                group["proxy_url"],
                dict(
                    op="ready",
                    spec=group["spec"],
                    members=members,
                    timeout=max(0.001, group["deadline"] - time.monotonic()),
                ),
                max(0.001, group["deadline"] - time.monotonic()),
            )
            future.set_result(result)
        except Exception as error:
            future.set_exception(error)

    def _abort(self, group):
        # Native failure remains primary; peers also have finite deadlines.
        with suppress(OSError, RuntimeError, ValueError):
            exchange(group["proxy_url"], dict(op="abort", spec=group["spec"], timeout=1), 1)

    def poll(self, completed, finished, failed):
        if self.closed:
            raise RuntimeError("decode batch gate is closed")
        for request in finished:
            if request not in self.requests:
                continue
            identity = self.requests[request][0]
            group = self.groups[identity]
            if not group["released"]:
                threading.Thread(target=self._abort, args=(group,), daemon=True).start()
                raise RuntimeError("decode batch cancelled before global release")
            del self.requests[request]
            if not any(value[0] == identity for value in self.requests.values()):
                del self.groups[identity]
        for request in completed:
            if request not in self.requests:
                continue
            group = self.groups[self.requests[request][0]]
            if request in failed:
                threading.Thread(target=self._abort, args=(group,), daemon=True).start()
                raise RuntimeError("KV receive failed/cancelled; decode batch will not release")
            if request in group["ready"]:
                continue
            group["ready"].add(request)
            if self.observe:
                group["tokens"][request] = self.observe("begin", request, self.requests[request][2], None)
        released = set()
        for group in self.groups.values():
            if group["released"]:
                continue
            if time.monotonic() >= group["deadline"]:
                threading.Thread(target=self._abort, args=(group,), daemon=True).start()
                raise TimeoutError("decode batch deadline exceeded; no partial release")
            if len(group["ready"]) == group["spec"]["batch_size"] and group["future"] is None:
                group["future"] = Future()
                members = [[slot, *self.requests[request][1:3]] for slot, request in group["members"].items()]
                threading.Thread(target=self._report, args=(group, members, group["future"]), daemon=True).start()
            future = group["future"]
            if future is not None and future.done():
                try:
                    future.result()  # Never fall back to partial Decode on control failure.
                except Exception:
                    threading.Thread(target=self._abort, args=(group,), daemon=True).start()
                    raise
                group["released"] = True
                released.update(group["ready"])
                if self.observe:
                    for request, token in group["tokens"].items():
                        self.observe("end", request, self.requests[request][2], token)
                self.logger.info(
                    "KVAF_DECODE_BATCH_RELEASE group=%s rank=%s batch=%s",
                    group["spec"]["group"],
                    self.rank,
                    len(group["ready"]),
                )
        return released | completed.difference(self.requests)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for group in self.groups.values():
            if not group["released"]:
                threading.Thread(target=self._abort, args=(group,), daemon=True).start()
