# SPDX-License-Identifier: Apache-2.0
"""Validate the actual launch script's backend configuration serialization."""

import json
import subprocess
import sys
import unittest
from pathlib import Path


class LaunchConfigTest(unittest.TestCase):
    def test_backend_switch_reaches_additional_config(self):
        script = (Path(__file__).resolve().parents[3] / "scripts/start_vllm_server.sh").read_text()
        body = script.split('additional_config="$(python3 - ', 1)[1].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        for backend in ("batch_get", "li_fetch"):
            result = subprocess.run(
                [
                    sys.executable,
                    "-",
                    "false",
                    "true",
                    "127.0.0.1:28999",
                    "/tmp/build",
                    "16",
                    "{}",
                    "false",
                    "16384",
                    backend,
                    "8",
                    "32",
                    "8",
                    "512",
                ],
                input=body,
                text=True,
                capture_output=True,
                check=True,
            )
            config = json.loads(result.stdout)
            self.assertEqual(config["kvaf_config"]["decode_backend"], backend)
            self.assertEqual(config["kvaf_config"]["intermediate_fetches"], 8)
            self.assertEqual(config["kvaf_config"]["hot_tokens"], 16384)
            self.assertEqual(config["kvaf_config"]["prefetch_tokens"], 512)
            self.assertTrue(config["enable_cpu_binding"])
            self.assertEqual(config["kvaf_config"]["put_interval_steps"], 32)
            self.assertEqual(config["kvaf_config"]["put_overlap_steps"], 8)


class PrefetchBudgetTest(unittest.TestCase):
    def test_equal_window_budget_and_validation(self):
        import importlib.util

        path = Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/kv_transfer/kvaf/config.py"
        spec = importlib.util.spec_from_file_location("prefetch_config", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            for total in (0, 512, 1024, 2048):
                config = module.KVAFConfig(decode_backend="li_fetch", prefetch_tokens=total)
                self.assertEqual(config.prefetch_tokens // config.intermediate_fetches, total // 8)
            for total, windows in ((513, 8), (512, 1), (4096, 8), (-1, 8), (True, 8)):
                with self.assertRaises(ValueError):
                    module.KVAFConfig(decode_backend="li_fetch", prefetch_tokens=total, intermediate_fetches=windows)
        finally:
            sys.modules.pop(spec.name, None)
