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
            self.assertTrue(config["enable_cpu_binding"])
