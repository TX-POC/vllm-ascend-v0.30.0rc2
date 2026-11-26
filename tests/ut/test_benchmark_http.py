# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small loopback checks for the HTTP-only benchmark; no NPU/vLLM imports."""

import contextlib
import io
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from benchmark import (  # noqa: E402
    BenchmarkError,
    Config,
    Http,
    ProfileCoordinator,
    PromptSource,
    Reporter,
    main,
    prefix_probe,
    preflight,
    run_level,
    summarize_http,
    verify_cached,
    warm_prefixes,
)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        capability = getattr(self.server, "decode_batch_capability", None)
        if self.path == "/v1/kvaf-decode-batch" and capability is not None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(capability).encode())
            return
        if self.path.startswith("/v1/kvaf-decode-batch/groups/"):
            UUID(self.path.rsplit("/", 1)[-1])
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(self.server.decode_batch_result).encode())
            return
        self.send_response(200)
        self.end_headers()
        if self.path == "/v1/models":
            self.wfile.write(json.dumps({"data": [self.server.model]}).encode())
            return
        self.wfile.write(b'vllm:num_requests_running{model_name="glm"} 4\nvllm:num_requests_waiting 0\n')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, dict(self.headers), body))
        self.send_response(200)
        self.end_headers()
        if self.path == "/tokenize":
            tokens = list(range(48)) if body["prompt"] == "dataset" else [11, 13, 17]
            self.wfile.write(json.dumps({"tokens": tokens}).encode())
            return
        if self.path in ("/start_profile", "/stop_profile"):
            return
        if self.server.gate is not None:
            self.server.gate.wait(timeout=3)
        self.wfile.write(b": keep-alive\n\n")
        for ids in ([1, 2], [3], [4], [5]):
            choice = {"text": "content"} if body["model"] == "no-ids" else {"token_ids": ids}
            event = {"choices": [choice]}
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
            self.wfile.flush()
            time.sleep(0.02)
        usage = {
            "prompt_tokens": len(body["prompt"]),
            "completion_tokens": body["max_tokens"],
            "prompt_tokens_details": {
                "cached_tokens": getattr(self.server, "cached_tokens", max(0, len(body["prompt"]) - 1))
            },
        }
        self.wfile.write(("data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n").encode())
        if body["model"] != "truncated":
            self.wfile.write(b"data: [DONE]\n\n")


class BenchmarkHttpTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.requests, self.server.gate = [], None
        self.server.model = {"id": "glm", "max_model_len": 131072}
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.config = Config(
            base_url=f"http://127.0.0.1:{self.server.server_port}",
            input_tokens=32,
            output_tokens=5,
            request_timeout=3,
        )

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_proxy_auto_discovers_decode_batch_and_verifies_first_step(self):
        self.server.decode_batch_capability = dict(
            enabled=True,
            coordination="proxy",
            dp_size=2,
            tensor_parallel_size=1,
            max_batch_size=2,
            timeout_seconds=1,
            metadata_base_url=self.config.base_url,
            first_step_reporting=True,
        )
        result = {
            "released": True,
            "target_batch_per_rank": 1,
            "ranks": {
                str(rank): {"kv_ready": 1, "first_step": {"cohort_requests": 1, "scheduled_requests": 1}}
                for rank in range(2)
            },
        }
        self.server.decode_batch_result = result
        args = [
            "--pd-proxy",
            "--base-url",
            self.config.base_url,
            "--input-tokens",
            "32",
            "--output-tokens",
            "5",
        ]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args.extend(("--result-dir", directory))
            self.assertEqual(main(args), 0)
            report = json.loads((Path(directory) / "result.json").read_text())
            self.assertEqual(report["config"]["data_parallel_size"], 2)
            self.assertEqual(report["config"]["levels"], [1])
            self.assertEqual(report["levels"][0]["decode_batch_result"]["verification"], "matched")
            headers = [
                json.loads(next(value for key, value in headers.items() if key.lower() == "x-kvaf-decode-batch"))
                for path, headers, _ in self.server.requests
                if path == "/v1/completions"
            ]
            self.assertEqual({(tag["rank"], tag["index"]) for tag in headers}, {(0, 0), (1, 0)})
            result["ranks"]["1"]["first_step"]["scheduled_requests"] = 2
            self.assertEqual(main(args), 1)
            report = json.loads((Path(directory) / "result.json").read_text())
            self.assertEqual(report["levels"][0]["decode_batch_result"]["verification"], "batch_mismatch")
            request_count = len(self.server.requests)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args + ["--levels", "3"]), 1)
            self.assertEqual(len(self.server.requests), request_count)

    def test_terminal_displays_ttft_percentiles_with_decode_batch_disabled(self):
        self.server.decode_batch_capability = {"enabled": False}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            args = [
                "--pd-proxy",
                "--base-url", self.config.base_url,
                "--metadata-base-url", self.config.base_url,
                "--data-parallel-size", "2",
                "--levels", "1",
                "--warmup-rounds", "0",
                "--input-tokens", "32",
                "--output-tokens", "5",
                "--result-dir", directory,
            ]
            self.assertEqual(main(args), 0)
            report = json.loads((Path(directory) / "result.json").read_text())
        self.assertFalse(report["config"]["sync_decode_batch"])
        ttft = report["levels"][0]["http"]["ttft_ms"]
        self.assertEqual(ttft["count"], 2)
        lines = output.getvalue().splitlines()
        ttft_column = "TTFT (Mean/P50/P99, ms)"
        header_index = max(i for i, line in enumerate(lines) if ttft_column in line)
        headers = [cell.strip() for cell in lines[header_index].split("|")]
        cells = [cell.strip() for cell in lines[header_index + 1].split("|")]
        columns = dict(zip(headers, cells))
        self.assertEqual(columns[ttft_column], "/".join(f"{ttft[key]:.3f}" for key in ("mean", "p50", "p99")))

    def test_decode_statistics_and_optional_hot_kv_are_wave_scoped(self):
        self.server.decode_batch_capability = dict(
            enabled=True,
            coordination="proxy",
            dp_size=1,
            tensor_parallel_size=1,
            max_batch_size=2,
            timeout_seconds=1,
            metadata_base_url=self.config.base_url,
            first_step_reporting=True,
            hot_kv_reporting=True,
        )
        histogram = [0] * 1001
        histogram[823] = 19
        histogram[1000] = 1
        self.server.decode_batch_result = {
            "released": True,
            "target_batch_per_rank": 1,
            "ranks": {"0": {"kv_ready": 1, "first_step": {"cohort_requests": 1, "scheduled_requests": 1}}},
            "hot_kv": {"histogram": histogram, "rank_samples": {"0": 20}},
        }
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(
                main(
                    [
                        "--pd-proxy",
                        "--base-url",
                        self.config.base_url,
                        "--input-tokens",
                        "32",
                        "--output-tokens",
                        "5",
                        "--warmup-rounds",
                        "1",
                        "--measure-rounds",
                        "2",
                        "--hot-kv-stats",
                        "--result-dir",
                        directory,
                    ]
                ),
                0,
            )
            report = json.loads((Path(directory) / "result.json").read_text())
            self.assertNotIn("hot_kv_hit_pct", report["request_warmups"][0])
            for row in report["levels"]:
                self.assertEqual(row["hot_kv_hit_pct"]["p95"], 82.3)
                self.assertEqual(row["hot_kv_hit_pct"]["p99"], 100.0)
                tpot = row["http"]["decode_tpot_ms"]
                self.assertEqual(tpot["count"], 1)
                self.assertEqual(tpot["p95"], tpot["mean"])
                self.assertAlmostEqual(tpot["mean"], row["http"]["request_tpot_ms"]["mean"] * 0.75)
                self.assertNotIn("histogram", row["decode_batch_result"])
            aggregate = report["decode_tps_by_level"]["1"]
            self.assertEqual(aggregate["valid_waves"], 2)
            self.assertAlmostEqual(aggregate["mean_tps"], sum(row["overlap"]["tps"] for row in report["levels"]) / 2)
            lines = output.getvalue().splitlines()
            header_index = max(i for i, line in enumerate(lines) if "Decode Batch" in line)
            header, *rows = lines[header_index : header_index + 3]
            separators = [i for i, char in enumerate(header) if char == "|"]
            self.assertEqual(len(separators), 7)
            self.assertEqual(len(rows), 2)
            for line, row in zip(rows, report["levels"]):
                self.assertEqual([i for i, char in enumerate(line) if char == "|"], separators)
                cells = [cell.strip() for cell in line.split("|")]
                tpot = row["http"]["decode_tpot_ms"]
                ttft = row["http"]["ttft_ms"]
                self.assertEqual(cells[3], "/".join(f"{ttft[key]:.3f}" for key in ("mean", "p50", "p99")))
                self.assertEqual(cells[5], "/".join(f"{tpot[key]:.3f}" for key in ("p50", "p90", "p95", "p99", "mean")))
                self.assertEqual(cells[6], f"{row['overlap']['tps']:.3f}/{aggregate['mean_tps']:.3f}")
                self.assertEqual(cells[7], "2")
            self.assertIn("Hot KV Hit (P50/P90/P95/P99, %)", output.getvalue())
            specs = [
                json.loads(value)
                for path, headers, _ in self.server.requests
                if path == "/v1/completions"
                for name, value in headers.items()
                if name.lower() == "x-kvaf-decode-batch"
            ]
            self.assertEqual([spec.get("hot_kv_stats") for spec in specs], [None, True, True])
            self.assertEqual(len({spec["group"] for spec in specs}), 3)

    def test_defaults_match_server_launch(self):
        script = Path(__file__).resolve().parents[2] / "scripts" / "start_vllm_server.sh"
        settings = dict(re.findall(r"^(\w+)=(\d+)$", script.read_text(), re.MULTILINE))
        config = Config()
        self.assertEqual(urlparse(config.base_url).port, int(settings["port"]))
        self.assertEqual(config.tensor_parallel_size, int(settings["tensor_parallel_size"]))
        self.assertEqual(config.data_parallel_size, int(settings["data_parallel_size"]))
        self.assertLessEqual(config.input_tokens + config.output_tokens, int(settings["max_model_len"]))

    def test_local_service_ignores_environment_proxy(self):
        with mock.patch.dict(
            os.environ,
            {
                "http_proxy": "http://proxy.invalid:9",
                "HTTP_PROXY": "http://proxy.invalid:9",
                "no_proxy": "",
                "NO_PROXY": "",
            },
        ):
            preflight(Http(self.config), self.config)

    def test_preflight_context_limit_and_optional_metadata(self):
        http = Http(self.config)
        self.config.input_tokens, self.config.output_tokens = 131072, 256
        with self.assertRaisesRegex(BenchmarkError, r"input_tokens.*output_tokens.*max_model_len"):
            preflight(http, self.config)
        self.config.input_tokens = 130816
        preflight(http, self.config)
        self.config.input_tokens = 131071
        preflight(http, self.config, output_tokens=1)
        self.config.input_tokens = 131072
        for model in ({"id": "glm"}, {"id": "glm", "max_model_len": None}):
            self.server.model = model
            preflight(http, self.config)

    def test_connection_error_identifies_endpoint(self):
        # Reserve a port without listening, so no other process can take it.
        with socket.socket() as unavailable:
            unavailable.bind(("127.0.0.1", 0))
            self.config.base_url = f"http://127.0.0.1:{unavailable.getsockname()[1]}"
            with self.assertRaises(BenchmarkError) as error:
                Http(self.config).json("/v1/models", timeout=1)
        self.assertIn(self.config.base_url + "/v1/models", str(error.exception))
        self.assertIn("--base-url", str(error.exception))

    def test_token_bursts_and_incomplete_streams(self):
        http = Http(self.config)
        result = http.complete([11] * 32, 5, "exact-count", 0)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([event[1] for event in result["token_events"]], [2, 1, 1, 1])
        self.assertEqual(result["cached_tokens"], 31)
        metrics = summarize_http([result])
        expected = (result["last_ns"] - result["first_ns"]) / 1e6 / 3
        self.assertAlmostEqual(metrics["request_tpot_ms"]["mean"], expected)
        self.assertGreater(metrics["e2e_tps"], 0)
        self.config.model = "no-ids"
        missing = http.complete([11] * 32, 5, "no-ids", 0)
        self.assertEqual(missing["status"], "error")
        self.assertFalse(missing["token_ids_complete"])
        self.assertNotIn("sse_overlap_estimate", summarize_http([missing]))
        self.config.model = "truncated"
        self.assertEqual(http.complete([11] * 32, 5, "truncated", 0)["status"], "error")
        self.config.model = "glm"
        self.assertEqual(http.complete([11] * 32, 4, "wrong-count", 0)["status"], "error")

    def test_jsonl_first_record_and_prefix_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text('\n{"question":"dataset"}\nnot-read\n')
            self.config.prompt_jsonl = str(path)
            source = PromptSource(self.config, Http(self.config))
            self.assertEqual(source.build(0), list(range(32)))
            self.assertEqual(source.build(1), source.build(0))
            self.config.dataset_length_policy = "strict"
            with self.assertRaises(BenchmarkError):
                PromptSource(self.config, Http(self.config))
        verify_cached({"status": "ok", "cached_tokens": 24}, 32, 8)
        with self.assertRaises(BenchmarkError):
            verify_cached({"status": "ok", "cached_tokens": None}, 32, 8)

    def test_native_profile_controls_and_client_token_budget(self):
        http = Http(self.config)
        for early_end in (False, True):
            profile = ProfileCoordinator(http, "decode", 3, ["a", "b"])
            profile.observe("a", 8)
            profile.observe("b", 8)
            with profile.condition:
                self.assertTrue(
                    profile.condition.wait_for(
                        lambda profile=profile: "start_token_counts" in profile.result, timeout=3
                    )
                )
            profile.observe("a", 3)
            profile.observe("b", None if early_end else 3)
            result = profile.finish()
            self.assertEqual(result["status"], "error" if early_end else "ok")
            self.assertEqual(result["start_ack"]["http_status"], 200)
            self.assertEqual(result["stop_ack"]["http_status"], 200)
            self.assertEqual(result["observed_tokens_per_request"], {"a": 3, "b": 0 if early_end else 3})
        self.assertEqual([request[0] for request in self.server.requests], ["/start_profile", "/stop_profile"] * 2)

    def test_terminal_displays_real_prefix_misses_not_just_successful_http(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(result_dir=directory, prefix_warmup=True)
            reporter = Reporter(config)
            reporter.report["levels"].append(
                {
                    "batch": 2,
                    "status": "error",
                    "http": {"success": 2},
                    "requests": [{"cached_tokens": 130432}, {"cached_tokens": 0}],
                    "errors": ["prefix verification needs cached_tokens>=130432; received 0"],
                }
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                reporter.display()
            self.assertIn("[130432, 0]", output.getvalue())
            self.assertIn("status=error", output.getvalue())
            self.assertIn("error: prefix verification", output.getvalue())

    def test_wave_routes_all_dp_requests_concurrently(self):
        self.config.data_parallel_size = 2
        self.config.prefix_warmup = True
        self.config.cache_block_size = 8
        self.server.cached_tokens = 16  # 32-token input with one additional MTP tail block.
        http = Http(self.config)
        source = PromptSource(self.config, http)
        self.assertNotEqual(source.build(0), source.build(1))
        self.assertEqual(len(source.build(1)), 32)
        self.server.gate = threading.Barrier(4)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.config.result_dir = directory
            reporter = Reporter(self.config)
            row = run_level(self.config, http, source, 2, reporter, prefix_extra_recompute_blocks=[1, 1])
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["http"]["success"], 4)
            self.assertEqual(row["overlap"]["status"], "ready")
            self.assertAlmostEqual(row["overlap"]["tps"] * row["overlap"]["tpot_ms"] / 1000, 4)
            self.assertEqual(row["overlap"]["bs"], 4)
            self.assertTrue((reporter.directory / "live.json").is_file())
        requests = [request for request in self.server.requests if request[0] == "/v1/completions"]
        self.assertEqual(sorted(int(headers["X-Data-Parallel-Rank"]) for _, headers, _ in requests), [0, 0, 1, 1])
        self.assertEqual(len({headers["X-Request-Id"] for _, headers, _ in requests}), 4)


class BenchmarkPrefixTests(unittest.TestCase):
    def test_aligned_and_unaligned_replay_thresholds(self):
        for length, normal, mtp in ((4096, 3968, 3840), (4097, 4096, 3968), (130000, 129920, 129792)):
            for extra, threshold in ((0, normal), (1, mtp)):
                with self.subTest(length=length, extra=extra):
                    self.assertEqual(
                        verify_cached({"status": "ok", "cached_tokens": threshold}, length, 128, extra), extra
                    )
                    with self.assertRaises(BenchmarkError):
                        verify_cached({"status": "ok", "cached_tokens": threshold - 1}, length, 128, extra)
            with self.assertRaises(BenchmarkError):
                verify_cached({"status": "ok", "cached_tokens": mtp}, length, 128)
        for cached in (None, 0, 3712):
            with self.subTest(cached=cached), self.assertRaises(BenchmarkError):
                verify_cached({"status": "ok", "cached_tokens": cached}, 4096, 128, 1)

    def test_probe_calibrates_each_dp_and_rejects_insufficient_hits(self):
        config = Config(input_tokens=130000, data_parallel_size=2)
        source = mock.Mock()
        source.build.return_value = [11] * 4096
        http = mock.Mock()
        http.complete.side_effect = [{"status": "ok", "cached_tokens": cached} for cached in (0, 3968, 0, 3840)]
        results = []
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prefix_probe(config, http, source, 128, results), [0, 1])
        self.assertEqual(len(results), 4)
        self.assertEqual([call.args[3] for call in http.complete.call_args_list], [0, 0, 1, 1])
        for cached in (None, 0, 3712):
            http.complete.side_effect = [
                {"status": "ok", "cached_tokens": 0},
                {"status": "ok", "cached_tokens": cached},
            ]
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(BenchmarkError):
                prefix_probe(config, http, source, 128, [])
        config.input_tokens = 256
        with self.assertRaisesRegex(BenchmarkError, "longer than two cache blocks"):
            prefix_probe(config, http, source, 128, [])


class PrefixWarmupSchedulingTests(unittest.TestCase):
    def test_each_dp_advances_without_waiting_for_slow_dp(self):
        config = Config(data_parallel_size=2)
        slow_started, fast_finished = threading.Event(), threading.Event()
        active, peak = [0, 0], [0, 0]
        calls = [[], []]
        lock = threading.Lock()

        def complete(prompt, max_tokens, request_id, rank, *, request_phase):
            self.assertEqual(request_phase, "warmup")
            with lock:
                active[rank] += 1
                peak[rank] = max(peak[rank], active[rank])
                calls[rank].append(prompt)
            try:
                if rank == 0 and prompt == (0, 0):
                    slow_started.set()
                    self.assertTrue(fast_finished.wait(3), "DP1 must finish while DP0 is blocked")
                elif rank == 1:
                    self.assertTrue(slow_started.wait(3))
                    if prompt == (1, 3):
                        fast_finished.set()
                return {"status": "ok", "dp_rank": rank}
            finally:
                with lock:
                    active[rank] -= 1

        source = mock.Mock()
        source.variant.side_effect = lambda rank, index: (rank, index)
        source.build.side_effect = lambda variant: variant
        reporter = mock.Mock(report={})
        warmed = set()
        report = warm_prefixes(config, mock.Mock(complete=complete), source, 4, warmed, reporter)
        self.assertEqual(peak, [1, 1])
        self.assertEqual(calls, [[(rank, i) for i in range(4)] for rank in range(2)])
        self.assertEqual(len(report["requests"]), 8)
        again = warm_prefixes(config, mock.Mock(complete=complete), source, 4, warmed, reporter)
        self.assertEqual(again["requests"], [])

    def test_failed_request_stops_its_dp_queue(self):
        http = mock.Mock()
        http.complete.return_value = {"status": "error", "error": "failed"}
        source = mock.Mock()
        source.variant.side_effect = lambda rank, index: index
        source.build.side_effect = lambda variant: variant
        with self.assertRaisesRegex(BenchmarkError, "prefix warmup failed"):
            warm_prefixes(Config(data_parallel_size=1), http, source, 4, set(), mock.Mock(report={}))
        self.assertEqual(http.complete.call_count, 1)


if __name__ == "__main__":
    unittest.main()
