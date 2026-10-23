import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestStartVllmContainer(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name).resolve()
        self.repository = self.directory / "checkout with spaces"
        self.script = self.repository / "scripts" / "start_vllm_container.sh"
        self.script.parent.mkdir(parents=True)
        source = Path(__file__).resolve().parents[2] / "scripts" / self.script.name
        shutil.copyfile(source, self.script)
        self.log = self.directory / "docker.jsonl"
        self.bin_directory = self.directory / "bin"
        self.bin_directory.mkdir()

    def run_script(self, *arguments, fail_command="", exit_code=1, proxies=None, device_ids="1,3"):
        docker = self.bin_directory / "docker"
        docker.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "arguments = sys.argv[1:]\n"
            "environment = {arguments[i + 1]: os.environ.get(arguments[i + 1])\n"
            "               for i, argument in enumerate(arguments[:-1]) if argument == '--env'}\n"
            f"with open({str(self.log)!r}, 'a') as log:\n"
            "    log.write(json.dumps({'arguments': arguments, 'environment': environment}) + '\\n')\n"
            f"sys.exit({exit_code} if arguments[0] == {fail_command!r} else 0)\n"
        )
        docker.chmod(0o755)
        environment = os.environ.copy()
        for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
            environment.pop(name, None)
            environment.pop(name.upper(), None)
        environment.update(proxies or {})
        environment["ASCEND_VISIBLE_DEVICES"] = device_ids
        environment["PATH"] = f"{self.bin_directory}{os.pathsep}{environment['PATH']}"
        result = subprocess.run(
            ["bash", str(self.script), *arguments],
            cwd=self.directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, calls

    def test_missing_image_does_not_start_docker(self):
        result, calls = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])

    def test_run_failure_is_propagated(self):
        result, calls = self.run_script("registry/image:tag", fail_command="run", exit_code=23)
        self.assertEqual(result.returncode, 23)
        self.assertEqual([call["arguments"][0] for call in calls], ["run"])

    def test_invalid_device_selection_does_not_start_docker(self):
        result, calls = self.run_script("registry/image:tag", device_ids="1,invalid")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])

    def test_host_proxies_are_forwarded_without_putting_values_in_arguments(self):
        proxies = {
            "http_proxy": "http://user:p@ss!$word&x=1@localhost:3128",
            "https_proxy": "http://localhost:3128",
            "HTTP_PROXY": "http://localhost:8080",
            "HTTPS_PROXY": "http://localhost:8080",
            "ALL_PROXY": "socks5://localhost:1080",
            "no_proxy": "localhost,127.0.0.1,.internal,10.0.0.0/8",
        }
        result, calls = self.run_script("registry/image:tag", proxies={**proxies, "NO_PROXY": ""})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[0]["environment"], proxies)
        for value in proxies.values():
            self.assertNotIn(value, calls[0]["arguments"])

    def test_success_preserves_arguments_without_running_installation(self):
        image = "registry/image:tag with 'quotes'"
        name = 'container with "quotes"'
        result, calls = self.run_script(image, name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call["arguments"][0] for call in calls], ["run"])
        run_arguments = calls[0]["arguments"]
        self.assertEqual(run_arguments[run_arguments.index("--name") + 1], name)
        self.assertIn(image, run_arguments)
        self.assertIn(f"type=bind,src={self.repository},dst=/vllm-workspace/vllm-ascend", run_arguments)
        for path in (
            "/usr/local/Ascend/driver",
            "/usr/local/dcmi",
            "/usr/local/sbin",
            "/usr/local/bin/npu-smi",
            "/etc/ascend_install.info",
            "/etc/hccl_rootinfo.json",
            "/etc/hixlep",
            "/root/host",
            "/home",
            "/mnt",
            "/data",
        ):
            self.assertIn(f"type=bind,src={path},dst={path}", run_arguments)
        self.assertIn("type=bind,src=/var/log/npu,dst=/usr/slog", run_arguments)
        self.assertIn("--shm-size=512g", run_arguments)
        self.assertIn("--net=host", run_arguments)
        self.assertIn("--privileged=true", run_arguments)
        self.assertIn("--runtime=runc", run_arguments)
        self.assertNotIn("--ipc=host", run_arguments)
        self.assertIn("-it", run_arguments)
        self.assertEqual(run_arguments[-2:], [image, "/usr/sbin/init"])
        self.assertEqual(
            [arg for arg in run_arguments if arg.startswith("--device=")],
            [
                f"--device=/dev/{name}"
                for name in ("davinci_manager", "hisi_hdc", "ummu", "uburma", "davinci1", "davinci3")
            ],
        )
        self.assertFalse(
            any(arg.startswith(("ASCEND_VISIBLE_DEVICES=", "ASCEND_RT_VISIBLE_DEVICES=")) for arg in run_arguments)
        )
        self.assertFalse(any("route.conf" in arg or "urma_admin" in arg or ",readonly" in arg for arg in run_arguments))


if __name__ == "__main__":
    unittest.main()
