# SPDX-License-Identifier: Apache-2.0
"""PD launcher validation without importing or executing the NPU runtime."""

import json
import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import test_kvaf_pd_integration as helpers

ROOT = Path(__file__).resolve().parents[2]
SMOKE = helpers.load_file("pd_smoke_for_ut", "scripts/kvaf_pd_smoke.py")


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
