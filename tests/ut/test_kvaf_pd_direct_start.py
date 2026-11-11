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
    def test_pooled_entries_are_standalone_and_preserve_global_topology(self):
        root = Path(__file__).resolve().parents[2]
        routes = []
        for role, node in (("p", 0), ("p", 1), ("d", 0), ("d", 1)):
            with self.subTest(role=role, node=node), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp)
                environment = target / "third_party/KVOP/env.sh"
                environment.parent.mkdir(parents=True)
                environment.write_text("# CPU-only command capture\n")
                interpreter = target / "python"
                interpreter.write_text(
                    f"#!{sys.executable}\nimport json,sys,os\n"
                    "print(json.dumps({'args':sys.argv[1:],'ip':os.environ['VLLM_HOST_IP'],"
                    "'devices':os.environ['ASCEND_RT_VISIBLE_DEVICES']}))\n"
                )
                interpreter.chmod(0o755)
                source = (root / f"examples/disaggregated_prefill_v1/kvaf_pooled/{role}{node}_start.sh").read_text()
                script = target / "standalone.sh"
                script.write_text(
                    source.replace("/usr/local/python3.12.13/bin/python3", str(interpreter))
                    .replace("/home/c00612283/vllm-ascend-v0.30.0rc2", str(target))
                    .replace("/home/d00843668/profiling_no_pooling", str(target / "profiler"))
                    .replace("/dev/shm/vllm_metrics_", str(target / "metrics_"))
                )
                result = subprocess.run(["bash", str(script)], text=True, capture_output=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(result.stdout.splitlines()[-1])
                args = data["args"]

                def value(flag, args=args):
                    self.assertEqual(args.count(flag), 1)
                    return args[args.index(flag) + 1]

                producer = role == "p"
                self.assertEqual(data["ip"], f"141.62.38.{(12 if producer else 24) + node}")
                self.assertEqual(data["devices"], "0,1,2,3,4,5,6,7")
                self.assertEqual(value("--tensor-parallel-size"), "8" if producer else "1")
                self.assertEqual(value("--data-parallel-size"), "2" if producer else "16")
                self.assertEqual(value("--data-parallel-size-local"), "1" if producer else "8")
                self.assertEqual(value("--data-parallel-start-rank"), str(node * (1 if producer else 8)))
                self.assertEqual(value("--data-parallel-address"), f"141.62.38.{12 if producer else 24}")
                self.assertEqual("--headless" in args, node == 1)
                self.assertIn("--enable-expert-parallel", args)
                self.assertIn("--no-enable-prefix-caching", args)
                self.assertEqual(value("--block-size"), "128")
                self.assertEqual(value("--kv-cache-dtype"), "fp8")
                self.assertEqual(value("--attention_config.indexer_kv_dtype"), "fp8")
                self.assertEqual(value("--max-model-len"), "131072")
                self.assertEqual(json.loads(value("--speculative-config"))["num_speculative_tokens"], 3)
                transfer = json.loads(value("--kv-transfer-config"))
                extra = transfer["kv_connector_extra_config"]
                self.assertFalse(extra["source_tp0_only"])
                self.assertEqual(extra["diagnostics_requests"], 1)
                self.assertTrue(extra["verify_kv"])
                additional = json.loads(value("--additional-config"))
                self.assertFalse(additional["enable_dsa_cp"])
                self.assertFalse(additional["c8_enable_reshape_optim"])
                self.assertEqual(additional["kvaf_config"]["enabled"], not producer)
                if producer:
                    self.assertIn("--enforce-eager", args)
                    self.assertEqual(
                        extra["decode_endpoints"],
                        [f"141.62.38.{24 + host}:{31000 + host * 8 + rank}" for host in range(2) for rank in range(8)],
                    )
                    self.assertNotIn("control_backlog", extra)
                else:
                    self.assertNotIn("--enforce-eager", args)
                    self.assertIn("--async-scheduling", args)
                    self.assertEqual(json.loads(value("--compilation-config"))["cudagraph_mode"], "FULL_DECODE_ONLY")
                    self.assertEqual(extra["control_backlog"], 128)
                    self.assertEqual(extra["initialization_coordinator"], "141.62.38.26:28998")
                    self.assertTrue(additional["enable_mlapo"])
                    self.assertEqual(additional["finegrained_tp_config"]["lmhead_tensor_parallel_size"], 8)
                    mapping = additional["kvaf_config"]["cpu_addresses"]
                    self.assertEqual(
                        mapping,
                        {str(i): f"141.62.38.{26 + node * 4 + i // 2}:{28997 + (i % 2) * 2}" for i in range(8)},
                    )
                    routes.extend(mapping.values())
                rejected = subprocess.run(["bash", str(script), "--dry-run"], capture_output=True, check=False)
                self.assertEqual(rejected.returncode, 2)
        self.assertEqual(len(set(routes)), 16)

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
