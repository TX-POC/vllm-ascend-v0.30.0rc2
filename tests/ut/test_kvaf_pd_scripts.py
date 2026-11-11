# SPDX-License-Identifier: Apache-2.0
"""PD launcher validation without importing or executing the NPU runtime."""

import gc
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import weakref
from pathlib import Path
from queue import Full
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import test_kvaf_pd_integration as helpers

ROOT = Path(__file__).resolve().parents[2]
SMOKE = helpers.load_file("pd_smoke_for_ut", "scripts/kvaf_pd_smoke.py")
TRANSFER_STATS = helpers.load_file("pd_transfer_stats_for_ut", "scripts/kvaf_pd_transfer_stats.py")


class TestPDLauncher(unittest.TestCase):
    def launch(self, role="p", *, transfer=None, additional=None, extra=()):
        transfer = transfer or dict(
            kv_connector="PDKVoffloadConnector",
            kv_connector_module_path="vllm_ascend.distributed.kv_transfer.kvaf.pd.connector",
            kv_role="kv_producer" if role == "p" else "kv_consumer",
            kv_connector_extra_config=dict(decode_endpoints=["10.0.0.1:31000"]),
        )
        if additional is None:
            additional = dict(kvaf_config=dict(enabled=role == "d", build_dir="/tmp/SDK build"))
        return subprocess.run(
            [
                "bash",
                str(ROOT / "scripts/start_vllm_pd.sh"),
                "--dry-run",
                role,
                "10.0.0.1",
                "eth0",
                "--",
                "/tmp/model with spaces",
                "--kv-transfer-config",
                json.dumps(transfer),
                "--additional-config=" + json.dumps(additional),
                *extra,
            ],
            env=dict(os.environ, KVAF_PD_PYTHON=sys.executable),
            text=True,
            capture_output=True,
            check=False,
        )

    def test_native_json_spaces_and_mtp_are_preserved_for_both_roles(self):
        for role in ("p", "d"):
            with self.subTest(role=role):
                result = self.launch(role)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = shlex.split(result.stdout)
                self.assertEqual(args[:4], [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve"])
                self.assertIn("/tmp/model with spaces", args)
                self.assertIn("--no-enable-prefix-caching", args)
                self.assertIn("--enforce-eager", args)
                self.assertEqual(json.loads(args[args.index("--speculative-config") + 1])["num_speculative_tokens"], 3)

    def test_explicit_graph_and_native_parallel_options_are_not_overwritten(self):
        result = self.launch(extra=("--no-enforce-eager", "--tensor-parallel-size", "8", "--data-parallel-size", "2"))
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split(result.stdout)
        self.assertNotIn("--enforce-eager", args)
        self.assertEqual(args.count("--tensor-parallel-size"), 1)
        self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "8")

    def test_unsafe_prefix_and_wrong_backend_are_rejected(self):
        result = self.launch(extra=("--enable-prefix-caching",))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prefix", result.stderr)
        result = self.launch(additional=dict(kvaf_config=dict(enabled=True, build_dir="/SDK")))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("enabled=false", result.stderr)

    def test_missing_native_config_or_invalid_invocation_exits_without_launch(self):
        for args in ([], ["--dry-run", "p", "10.0.0.1", "eth0", "--", "/model"]):
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/start_vllm_pd.sh"), *args],
                env=dict(os.environ, KVAF_PD_PYTHON=sys.executable),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("unbound variable", result.stderr)

    def test_two_node_templates_keep_baseline_and_pd_native_parameters_aligned(self):
        templates = ROOT / "examples/disaggregated_prefill_v1/kvaf_two_node"
        for name in ("p_start.sh", "d_start.sh", "d_baseline.sh"):
            with self.subTest(script=name):
                path = templates / name
                subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
                source = path.read_text()
                command = source[source.index("exec /usr/local/") :].replace("\\\n", " ")
                args = shlex.split(command)

                def value(flag, args=args):
                    self.assertEqual(args.count(flag), 1)
                    return args[args.index(flag) + 1]

                self.assertEqual(value("--tensor-parallel-size"), "8")
                self.assertEqual(value("--data-parallel-size"), "1")
                self.assertEqual(value("--max-model-len"), "8192")
                self.assertEqual(value("--max-num-seqs"), "1")
                self.assertEqual(value("--kv-cache-dtype"), "fp8")
                self.assertEqual(json.loads(value("--speculative-config"))["num_speculative_tokens"], 3)
                additional = json.loads(value("--additional-config"))
                self.assertFalse(additional["c8_enable_reshape_optim"])
                self.assertNotIn("glm5-mtp-4l", source)
                if name == "d_baseline.sh":
                    self.assertNotIn("--kv-transfer-config", args)
                    self.assertNotIn("--worker-cls", args)
                    self.assertFalse(additional["kvaf_config"]["enabled"])
                    continue
                transfer = json.loads(value("--kv-transfer-config"))
                self.assertEqual(transfer["kv_connector"], "PDKVoffloadConnector")
                self.assertEqual(transfer["kv_connector_extra_config"]["diagnostics_requests"], 0)
                self.assertFalse(transfer["kv_connector_extra_config"]["verify_kv"])
                self.assertEqual(additional["kvaf_config"]["enabled"], name == "d_start.sh")
                if name == "p_start.sh":
                    self.assertEqual(
                        transfer["kv_connector_extra_config"]["decode_endpoints"],
                        [f"D_NODE_IP:{31000 + i}" for i in range(8)],
                    )


class TestPooledBackgroundStart(unittest.TestCase):
    """Exercise real background processes/listeners without loading NPU libraries."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.target = Path(self.directory.name)
        self.addCleanup(self.stop_launched_services)
        self.bin = self.target / "bin"
        self.bin.mkdir()
        (self.bin / "python3").symlink_to(sys.executable)
        numactl = self.bin / "numactl"
        numactl.write_text('#!/usr/bin/env bash\nshift 2\nexec "$@"\n')
        numactl.chmod(0o755)
        self.service = self.target / "service.py"
        self.service.write_text(
            f"#!{sys.executable}\n"
            "import argparse,json,os,signal,socket,sys,time\n"
            "from http.server import BaseHTTPRequestHandler,HTTPServer\n"
            "parser=argparse.ArgumentParser()\n"
            "parser.add_argument('--listen','--host',dest='host')\n"
            "parser.add_argument('--port',type=int)\n"
            "args,_=parser.parse_known_args()\n"
            "mode=os.environ.get('TEST_SERVICE_MODE','ready')\n"
            "if mode=='exit' or str(args.port)==os.environ.get('TEST_FAIL_PORT'): sys.exit(17)\n"
            "if mode=='stopped': os.kill(os.getpid(),signal.SIGSTOP)\n"
            "if mode=='timeout': time.sleep(60); sys.exit(0)\n"
            "if mode=='borrow':\n"
            " print(f'[KVAF_CPU] LISTEN address={args.host} port={args.port}',flush=True)\n"
            " time.sleep(60); sys.exit(0)\n"
            "time.sleep(.2)\n"
            "if '--listen' in sys.argv:\n"
            " listener=socket.socket(); listener.bind((args.host,args.port)); listener.listen()\n"
            " if mode!='no_marker': print(f'[KVAF_CPU] LISTEN address={args.host} port={args.port}',flush=True)\n"
            " while True: time.sleep(1)\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            " def do_GET(self):\n"
            "  self.send_response(503 if mode=='unhealthy' else 200); self.end_headers()\n"
            "  self.wfile.write(json.dumps({'status':'ok'}).encode())\n"
            "server=HTTPServer((args.host,args.port),Handler)\n"
            "print(f'Uvicorn running on http://{args.host}:{args.port}',flush=True)\n"
            "server.serve_forever()\n"
        )
        self.service.chmod(0o755)

    def stop_launched_services(self):
        # Only PID files in this test's private log directory are eligible.
        for file in self.target.rglob("*.pid"):
            pid = int(file.read_text())
            try:
                os.kill(pid, signal.SIGCONT)
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def unused_ports(self, count):
        listeners = [socket.socket() for _ in range(count)]
        try:
            for listener in listeners:
                listener.bind(("127.0.0.1", 0))
            return [listener.getsockname()[1] for listener in listeners]
        finally:
            for listener in listeners:
                listener.close()

    def launch(self, name, *, mode="ready", ports=None, fail_port=""):
        source = (ROOT / "examples/disaggregated_prefill_v1/kvaf_pooled" / name).read_text()
        ports = ports or self.unused_ports(2 if name.startswith("k") else 1)
        for host in range(12, 34):
            source = source.replace(f"local_ip=141.62.38.{host}\n", "local_ip=127.0.0.1\n")
        source = (
            source.replace("/home/c00612283/vllm-ascend-v0.30.0rc2", str(ROOT))
            .replace("/home/c00612283/pd-pool-logs", str(self.target / "logs"))
            .replace('dir="$PWD/pd-logs/proxy/', f'dir="{self.target}/logs/proxy/')
            .replace("/usr/local/python3.12.13/bin/python3", sys.executable)
            .replace("startup_timeout=60", "startup_timeout=1")
            .replace("startup_timeout=20", "startup_timeout=2")
            .replace("startup_timeout=180", "startup_timeout=1")
            .replace("ports=(28997 28999)", f"ports=({' '.join(map(str, ports))})")
            .replace("port=28998", f"port={ports[0]}")
            .replace("port=8000", f"port={ports[0]}")
            .replace(
                "K_BIN=/home/c00612283/KVAF-Devlop/build-hostcpu-k/runtime/kvaf_cpu_service",
                f"K_BIN={self.service}",
            )
            .replace(
                f"PROXY={ROOT}/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py",
                f"PROXY={self.service}",
            )
        )
        script = self.target / name
        script.write_text(source)
        return subprocess.run(
            ["bash", str(script)],
            env=dict(
                os.environ,
                PATH=f"{self.bin}:{os.environ['PATH']}",
                TEST_SERVICE_MODE=mode,
                TEST_FAIL_PORT=str(fail_port),
            ),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_each_k_host_script_succeeds_only_after_both_listeners_are_ready(self):
        for node in range(8):
            with self.subTest(node=node):
                result = self.launch(f"k{node}_start.sh")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.count("SERVICE_READY"), 2)

    def test_exited_or_stopped_background_child_fails_the_script(self):
        for mode in ("exit", "stopped"):
            with self.subTest(mode=mode):
                result = self.launch("k0_start.sh", mode=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("SERVICE_START_FAILED", result.stderr)
                self.assertNotIn("SERVICE_READY", result.stdout)

    def test_timeout_or_missing_readiness_log_is_not_success(self):
        for mode in ("timeout", "no_marker"):
            with self.subTest(mode=mode):
                result = self.launch("k0_start.sh", mode=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("SERVICE_READY", result.stdout)

    def test_partial_k_failure_returns_nonzero_and_retains_the_successful_service(self):
        ports = self.unused_ports(2)
        result = self.launch("k0_start.sh", ports=ports, fail_port=ports[1])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout.count("SERVICE_READY"), 1)
        with socket.create_connection(("127.0.0.1", ports[0]), timeout=1):
            pass

    def test_occupied_port_does_not_claim_an_existing_service_as_its_own(self):
        with socket.socket() as existing:
            existing.bind(("127.0.0.1", 0))
            existing.listen()
            ports = [existing.getsockname()[1], *self.unused_ports(1)]
            result = self.launch("k0_start.sh", ports=ports)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout.count("SERVICE_READY"), 1)
            with socket.create_connection(existing.getsockname(), timeout=1):
                pass

    def test_real_coordinator_readiness_and_duplicate_start_failure(self):
        port = self.unused_ports(1)[0]
        result = self.launch("coordinator_start.sh", ports=[port])
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.launch("coordinator_start.sh", ports=[port])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Address already in use", result.stderr)

    def test_log_and_existing_listener_cannot_replace_new_process_socket_ownership(self):
        with socket.socket() as existing:
            existing.bind(("127.0.0.1", 0))
            existing.listen()
            ports = [existing.getsockname()[1], *self.unused_ports(1)]
            result = self.launch("k0_start.sh", ports=ports, mode="borrow")
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("SERVICE_READY", result.stdout)

    def test_proxy_http_healthcheck_must_succeed(self):
        for mode in ("ready", "unhealthy", "exit"):
            with self.subTest(mode=mode):
                started = time.monotonic()
                result = self.launch("proxy_start.sh", mode=mode)
                self.assertEqual(result.returncode == 0, mode == "ready", result.stderr)
                self.assertLess(time.monotonic() - started, 5)


class TestPDTransferStats(unittest.TestCase):
    def test_capture_writer_does_not_retain_native_sender_or_its_tensors(self):
        class Sender:
            tp_rank = 0
            memories = {}

            def _call(self, endpoint, message, **kwargs):
                return True

        with tempfile.TemporaryDirectory() as directory:
            sender = Sender()
            reference = weakref.ref(sender)
            observer = TRANSFER_STATS.PullObserver(sender, dict(role="P"), directory=directory)
            del sender
            gc.collect()
            try:
                self.assertIsNone(reference())
            finally:
                observer.capture.close()

    def row(self, *, member=0, factor=1, layer="layer0"):
        return dict(
            kind="layer",
            request="r",
            generation="g",
            layer=layer,
            peer="D:31000",
            member=member,
            ratio=2,
            errors=[],
            input_tokens=256,
            chunks=[
                dict(first=0, tokens=128, final=False, spans=dict(npu=[1_000_000, 2_000_000 * factor], cpu=None)),
                dict(first=1, tokens=256, final=True, spans=dict(npu=[10_000_000, 11_000_000 * factor], cpu=None)),
            ],
        )

    def test_parallel_envelopes_sum_chunks_without_compute_gaps(self):
        first, second = self.row(), self.row(member=1)
        second["chunks"][0]["spans"]["npu"] = [1_500_000, 3_000_000]
        second["chunks"][1]["spans"]["npu"] = [10_500_000, 12_000_000]
        result = TRANSFER_STATS.merge_layer([first, second], 256)
        self.assertEqual(result["t1_ms"], 4)
        self.assertIsNone(result["t2_ms"])
        self.assertAlmostEqual(TRANSFER_STATS.distribution([1, 3])["p90_ms"], 2.8)

    def test_missing_rank_chunk_and_invalid_pull_are_rejected(self):
        for case in ("rank", "chunk", "error", "time"):
            with self.subTest(case=case):
                rows = [self.row(), self.row(member=1)]
                if case == "rank":
                    rows.pop()
                elif case == "chunk":
                    rows[0]["chunks"].pop()
                elif case == "error":
                    rows[0]["errors"].append("lost_ack")
                else:
                    rows[0]["chunks"][0]["spans"]["npu"] = [10, 5]
                with self.assertRaises(ValueError):
                    TRANSFER_STATS.merge_layer(rows, 256)

    def test_all_layers_are_required_and_request_totals_are_not_layer_means(self):
        header = dict(role="P", host="p", boot_id="b", engine="e", layers=["layer0", "layer1"])
        rows = [self.row(member=member, layer=layer) for layer in header["layers"] for member in range(2)]
        checkpoint = dict(enqueued=4, written=4, dropped=0, errors=0, pending_jobs=0, pending_layers=0)
        layers, requests, rejected, _, problems = TRANSFER_STATS.collect_samples([(header, checkpoint, rows, "p")], 256)
        self.assertEqual(len(layers), 2)
        self.assertEqual(requests[0]["t1_ms"], 4)
        self.assertFalse(rejected or problems)
        _, requests, rejected, _, _ = TRANSFER_STATS.collect_samples([(header, checkpoint, rows[:2], "p")], 256)
        self.assertFalse(requests)
        self.assertTrue(rejected)

    def test_observer_preserves_return_exception_and_rejects_lost_ack_samples(self):
        state = {"lost": False}

        def original(endpoint, message, **kwargs):
            if message["op"] == "plan":
                return dict(job="j", count=dict(npu=1, cpu=0))
            if message["op"] == "pull" and state["lost"]:
                raise TimeoutError("original error")
            return True

        with tempfile.TemporaryDirectory() as directory:
            observer = TRANSFER_STATS.PullObserver(
                NS(_call=original, tp_rank=0, memories={"layer0": None}),
                dict(role="P", host="p", boot_id="b", engine="e"),
                directory=directory,
            )
            try:
                observer(
                    "D",
                    dict(
                        op="plan",
                        request="r",
                        generation="g",
                        layer="layer0",
                        first=0,
                        tokens=128,
                        final=True,
                        member=0,
                        ratio=1,
                    ),
                )
                state["lost"] = True
                with self.assertRaisesRegex(TimeoutError, "original error"):
                    observer("D", dict(op="pull", job="j", path="npu", batch=0))
                self.assertTrue(observer("D", dict(op="complete", job="j")))
            finally:
                observer.capture.close()
            captures = TRANSFER_STATS.read_captures(Path(directory).glob("*.jsonl"))
            _, requests, rejected, _, _ = TRANSFER_STATS.collect_samples(captures, 128)
            self.assertFalse(requests)
            self.assertTrue(rejected)

    def test_full_buffer_drops_measurements_without_waiting(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = TRANSFER_STATS.BufferedCapture(dict(role="P"), lambda: {}, directory=directory)
            try:
                with patch.object(capture.queue, "put_nowait", side_effect=Full):
                    capture.emit(dict(kind="layer"))
                self.assertEqual(capture.dropped, 1)
                self.assertEqual(capture.enqueued, 0)
            finally:
                capture.close()
            self.assertEqual(TRANSFER_STATS.read_captures([capture.path])[0][1]["dropped"], 1)

    def test_performance_comparison_reports_regressions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "baseline.json").write_text(json.dumps(dict(output_throughput_tps=100, mean_tpot_ms=10)))
            argv = [
                "compare",
                "--baseline",
                str(root / "baseline.json"),
                "--sampled",
                str(root / "sampled.json"),
                "--max-regression-percent",
                "1",
                "--output",
                str(root / "comparison.json"),
            ]
            with patch("builtins.print"):
                (root / "sampled.json").write_text(json.dumps(dict(output_throughput_tps=99.5, mean_tpot_ms=10.05)))
                self.assertEqual(TRANSFER_STATS.main(argv), 0)
                (root / "sampled.json").write_text(json.dumps(dict(output_throughput_tps=95, mean_tpot_ms=10.05)))
                self.assertEqual(TRANSFER_STATS.main(argv), 2)

    def test_offline_cli_outputs_four_aggregated_distributions_and_checks_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for member in range(2):
                header = dict(
                    kind="header", schema=1, role="P", host="p", boot_id="b", engine="e", layers=["layer0", "layer1"]
                )
                rows = [self.row(member=member, layer=layer) for layer in header["layers"]]
                if member == 0:
                    for row in rows:
                        row["chunks"][0]["spans"]["cpu"] = [2_000_000, 3_000_000]
                        row["chunks"][1]["spans"]["cpu"] = [12_000_000, 14_000_000]
                checkpoint = dict(
                    kind="checkpoint", enqueued=2, written=2, dropped=0, errors=0, pending_jobs=0, pending_layers=0
                )
                (root / f"P{member}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in [header, *rows, checkpoint])
                )
                d_header = dict(kind="header", schema=1, role="D_scheduler", dp_rank=member, max_num_seqs=16)
                d_checkpoint = dict(
                    kind="checkpoint", enqueued=0, written=0, dropped=0, errors=0, batch_histogram={"16": 5}
                )
                (root / f"D{member}.jsonl").write_text(json.dumps(d_header) + "\n" + json.dumps(d_checkpoint) + "\n")
            argv = [
                str(root),
                "--output-dir",
                str(root / "report"),
                "--expected-requests",
                "1",
                "--input-tokens",
                "256",
                "--expected-dp",
                "2",
            ]
            with patch("builtins.print"):
                self.assertEqual(TRANSFER_STATS.main(argv), 0)
                report = json.loads((root / "report/summary.json").read_text())
                means = {(row["metric"], row["scope"]): row["mean_ms"] for row in report["summary"]}
                self.assertEqual(
                    means,
                    {
                        ("t1", "request_layer"): 2,
                        ("t1", "request"): 4,
                        ("t2", "request_layer"): 3,
                        ("t2", "request"): 6,
                    },
                )
                (root / "P1.jsonl").unlink()
                self.assertEqual(TRANSFER_STATS.main(argv), 2)
            self.assertEqual(json.loads((root / "report/summary.json").read_text())["status"], "INCOMPLETE")


class TestPDSmoke(unittest.TestCase):
    def test_http_payload_and_failure_are_checked(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"choices":[{"text":"hello"}]}'
        opener = MagicMock()
        opener.open.return_value = response
        args = ["smoke", "--prompt-file", "/prompt", "--rounds", "1"]
        with (
            patch.object(sys, "argv", args),
            patch.object(Path, "read_text", return_value="hello"),
            patch.object(SMOKE, "build_opener", return_value=opener),
            patch("builtins.print"),
        ):
            SMOKE.main()
            request = opener.open.call_args.args[0]
            self.assertTrue(request.full_url.endswith("/v1/completions"))
            self.assertEqual(json.loads(request.data)["temperature"], 0)
            response.__enter__.return_value.read.return_value = b'{"error":"failed"}'
            with self.assertRaisesRegex(RuntimeError, "Invalid completion"):
                SMOKE.main()


if __name__ == "__main__":
    unittest.main()
