# SPDX-License-Identifier: Apache-2.0
"""No-argument entry scripts preserve native PD configuration and roles."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestDirectPDStart(unittest.TestCase):
    def test_both_entries_autodetect_and_build_native_arguments(self):
        root = Path(__file__).resolve().parents[2]
        for role, tp, dp in (("p", "2", "1"), ("d", "1", "2")):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp)
                (target / "scripts").mkdir()
                shutil.copy2(root / "scripts/start_vllm_pd.sh", target / "scripts/start_vllm_pd.sh")
                script = target / f"{role}_start.sh"
                text = (
                    (root / script.name)
                    .read_text()
                    .replace(
                        "python_bin=/usr/local/python3.12.13/bin/python3", f"python_bin={shlex.quote(sys.executable)}"
                    )
                )
                script.write_text(text)
                result = subprocess.run(
                    ["bash", str(script), "--dry-run"],
                    cwd="/tmp",
                    env=os.environ.copy(),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                args = shlex.split(result.stdout.splitlines()[-1])
                self.assertEqual(args[args.index("--tensor-parallel-size") + 1], tp)
                self.assertEqual(args[args.index("--data-parallel-size") + 1], dp)
                self.assertIn("--no-enable-prefix-caching", args)
                additional = json.loads(args[args.index("--additional-config") + 1])
                self.assertFalse(additional["c8_enable_reshape_optim"])
                transfer = json.loads(args[args.index("--kv-transfer-config") + 1])
                self.assertEqual(transfer["kv_connector_extra_config"]["diagnostics_requests"], 0)
                self.assertEqual(transfer["kv_role"], "kv_producer" if role == "p" else "kv_consumer")
                if role == "p":
                    self.assertEqual(len(transfer["kv_connector_extra_config"]["decode_endpoints"]), 2)
                else:
                    self.assertTrue(additional["kvaf_config"]["cpu_address"].endswith(":28997"))


if __name__ == "__main__":
    unittest.main()
