import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestStartVllmServer(unittest.TestCase):
    def test_launch_arguments_environment_and_exit_status(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory).resolve()
            model = directory / "model with spaces"
            model.mkdir()
            metrics = directory / "metrics"
            source = Path(__file__).resolve().parents[2] / "scripts" / "start_vllm_server.sh"
            content = source.read_text()
            for name, path in (("model_path", model), ("export PROMETHEUS_MULTIPROC_DIR", metrics)):
                content, count = re.subn(
                    rf"^{name}=.*$", f"{name}={shlex.quote(str(path))}", content, flags=re.MULTILINE
                )
                self.assertEqual(count, 1, name)
            script = directory / source.name
            script.write_text(content)
            capture = directory / "launch.json"
            executable = directory / "vllm"
            executable.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                f"with open({str(capture)!r}, 'w') as output:\n"
                "    json.dump({'arguments': sys.argv[1:], 'environment': dict(os.environ)}, output)\n"
                "sys.exit(23)\n"
            )
            executable.chmod(0o755)
            proxy_names = ("http_proxy", "https_proxy", "ftp_proxy", "all_proxy")
            environment = dict(os.environ, PATH=f"{directory}{os.pathsep}{os.environ['PATH']}")
            environment.pop("BASH_ENV", None)
            for name in proxy_names:
                environment[name] = environment[name.upper()] = "http://proxy.invalid:8080"
            result = subprocess.run(
                ["bash", str(script)], cwd=directory, env=environment, capture_output=True, text=True, timeout=10
            )
            self.assertEqual(result.returncode, 23, result.stderr)
            launch = json.loads(capture.read_text())
            arguments = launch["arguments"]
            self.assertEqual(arguments[:2], ["serve", str(model)])
            self.assertEqual(arguments.count(str(model)), 1)
            for flag, expected in (
                ("--tensor-parallel-size", "8"),
                ("--data-parallel-size", "1"),
                ("--data-parallel-size-local", "1"),
                ("--port", "8080"),
                ("--max-model-len", "131072"),
                ("--max-num-batched-tokens", "4096"),
            ):
                self.assertEqual(arguments[arguments.index(flag) + 1], expected, flag)
            configs = {
                flag: json.loads(arguments[arguments.index(flag) + 1])
                for flag in ("--compilation-config", "--speculative-config", "--additional-config")
            }
            self.assertEqual(configs["--compilation-config"]["cudagraph_mode"], "FULL_DECODE_ONLY")
            self.assertEqual(configs["--speculative-config"]["num_speculative_tokens"], 3)
            self.assertIs(configs["--speculative-config"]["enforce_eager"], True)
            self.assertEqual(configs["--additional-config"], {"enable_dsa_cp": False})
            exported = launch["environment"]
            for name in proxy_names:
                self.assertNotIn(name, exported)
                self.assertNotIn(name.upper(), exported)
            self.assertEqual(exported["VLLM_ASCEND_ENABLE_FLASHCOMM1"], "0")
            self.assertEqual(exported["HCCL_CONNECT_TIMEOUT"], "1800")
            self.assertEqual(exported["HCCL_EXEC_TIMEOUT"], "1800")
            self.assertEqual(exported["HCCL_IF_IP"], "141.62.38.12")
            for name in ("GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME"):
                self.assertEqual(exported[name], "enp34s0f1")
            self.assertEqual(exported["ASCEND_RT_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
            plog = Path(exported["ASCEND_PROCESS_LOG_PATH"])
            self.assertEqual(plog.name, "plog")
            self.assertTrue(plog.is_relative_to(directory))
            self.assertTrue(plog.is_dir())
            self.assertEqual(exported["PROMETHEUS_MULTIPROC_DIR"], str(metrics))
            self.assertTrue(metrics.is_dir())

            # The editable switch must reach vLLM as a JSON boolean in both states.
            script.write_text(content.replace("enable_dsa_cp=false", "enable_dsa_cp=true", 1))
            result = subprocess.run(
                ["bash", str(script)], cwd=directory, env=environment, capture_output=True, text=True, timeout=10
            )
            self.assertEqual(result.returncode, 23, result.stderr)
            arguments = json.loads(capture.read_text())["arguments"]
            additional_config = json.loads(arguments[arguments.index("--additional-config") + 1])
            self.assertIs(additional_config["enable_dsa_cp"], True)


if __name__ == "__main__":
    unittest.main()
