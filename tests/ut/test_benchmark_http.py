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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from benchmark import (  # noqa: E402
    BenchmarkError,
    Config,
    Http,
    ProfileCoordinator,
    PromptSource,
    Reporter,
    preflight,
    run_level,
    summarize_http,
    verify_cached,
)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
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
            "prompt_tokens_details": {"cached_tokens": max(0, len(body["prompt"]) - 1)},
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

    def test_wave_routes_all_dp_requests_concurrently(self):
        self.config.data_parallel_size = 2
        http = Http(self.config)
        source = PromptSource(self.config, http)
        self.assertNotEqual(source.build(0), source.build(1))
        self.assertEqual(len(source.build(1)), 32)
        self.server.gate = threading.Barrier(4)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.config.result_dir = directory
            reporter = Reporter(self.config)
            row = run_level(self.config, http, source, 2, reporter)
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["http"]["success"], 4)
            self.assertEqual(row["overlap"]["status"], "ready")
            self.assertAlmostEqual(row["overlap"]["tps"] * row["overlap"]["tpot_ms"] / 1000, 4)
            self.assertEqual(row["overlap"]["bs"], 4)
            self.assertTrue((reporter.directory / "live.json").is_file())
        requests = [request for request in self.server.requests if request[0] == "/v1/completions"]
        self.assertEqual(sorted(int(headers["X-Data-Parallel-Rank"]) for _, headers, _ in requests), [0, 0, 1, 1])
        self.assertEqual(len({headers["X-Request-Id"] for _, headers, _ in requests}), 4)


if __name__ == "__main__":
    unittest.main()
