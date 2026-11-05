# SPDX-License-Identifier: Apache-2.0
"""Manual deployment scripts forward native arguments to the chosen Python."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestDirectPDStart(unittest.TestCase):
    def test_both_manual_entries_preserve_native_config(self):
        root = Path(__file__).resolve().parents[2]
        for role, tp, dp, devices in (("p", "2", "1", "2,3"), ("d", "1", "2", "4,5")):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                rejected = subprocess.run(
                    ["bash", str(root / f"{role}_start.sh"), "--dry-run"], text=True, capture_output=True, check=False
                )
                self.assertEqual(rejected.returncode, 2)
                target = Path(tmp)
                model = target / "model"
                model.mkdir()
                sdk = target / "sdk"
                (sdk / "python/kvaf").mkdir(parents=True)
                (target / "third_party/KVOP").mkdir(parents=True)
                (target / "third_party/KVOP/env.sh").write_text("# Device environment substituted in CPU UT\n")
                mock_python = target / "python"
                mock_python.write_text(
                    f"#!{sys.executable}\nimport json,sys,os\n"
                    "print(json.dumps({'args':sys.argv[1:],'devices':os.environ['ASCEND_RT_VISIBLE_DEVICES'],"
                    "'ip':os.environ['HCCL_IF_IP'],'nic':os.environ['HCCL_SOCKET_IFNAME']}))\n"
                )
                mock_python.chmod(0o755)
                script = target / f"{role}_start.sh"
                text = (
                    (root / script.name)
                    .read_text()
                    .replace("/usr/local/python3.12.13/bin/python3", shlex.quote(str(mock_python)))
                    .replace(
                        "source /home/c00612283/vllm-ascend-v0.30.0rc2/third_party/KVOP/env.sh",
                        "source " + shlex.quote(str(target / "third_party/KVOP/env.sh")),
                    )
                )
                text = text.replace(
                    'export PROMETHEUS_MULTIPROC_DIR="/dev/shm/', f'export PROMETHEUS_MULTIPROC_DIR="{target}/'
                )
                script.write_text(text)
                result = subprocess.run(
                    ["bash", str(script)],
                    cwd=target,
                    env=os.environ.copy(),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(result.stdout.splitlines()[-1])
                args = data["args"]
                self.assertEqual(args[:3], ["-m", "vllm.entrypoints.cli.main", "serve"])
                self.assertEqual(data["devices"], devices)
                self.assertEqual(data["ip"], "141.62.38.12")
                self.assertEqual(data["nic"], "enp34s0f1")
                self.assertEqual(args[args.index("--tensor-parallel-size") + 1], tp)
                self.assertEqual(args[args.index("--data-parallel-size") + 1], dp)
                self.assertIn("--no-enable-prefix-caching", args)
                self.assertEqual(args[args.index("--block-size") + 1], "128")
                additional = json.loads(args[args.index("--additional-config") + 1])
                self.assertFalse(additional["c8_enable_reshape_optim"])
                transfer = json.loads(args[args.index("--kv-transfer-config") + 1])
                self.assertEqual(transfer["kv_connector_extra_config"]["diagnostics_requests"], 0)
                self.assertEqual(transfer["kv_role"], "kv_producer" if role == "p" else "kv_consumer")
                self.assertEqual(json.loads(args[args.index("--speculative-config") + 1])["num_speculative_tokens"], 3)
                if role == "p":
                    self.assertEqual(len(transfer["kv_connector_extra_config"]["decode_endpoints"]), 2)
                else:
                    self.assertEqual(additional["kvaf_config"]["cpu_address"], "141.62.38.12:28997")


if __name__ == "__main__":
    unittest.main()
