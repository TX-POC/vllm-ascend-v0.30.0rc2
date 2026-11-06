# SPDX-License-Identifier: Apache-2.0
"""CPU-only coordinator checks; these do not emulate UB or NPU correctness."""

import importlib.util
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/kvaf_pd_ddr_residency.py"
SPEC = importlib.util.spec_from_file_location("ddr_residency_script", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def ready(device):
    return {
        "event": "READY",
        "device": device,
        "ticket": {
            "address": "k:1",
            "epoch": 7,
            "region_id": device + 1,
            "base": device * 16,
            "bytes": 16,
        },
    }


class TestDDRResidencyScript(unittest.TestCase):
    def test_spawn_with_fake_sdk_keeps_all_regions_until_release(self):
        # Exercise real spawn/pipes/exit, without importing NPU libraries.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "torch.py").write_text("from unittest.mock import Mock\nnpu = Mock()\n")
            (root / "torch_npu.py").write_text("")
            (root / "kvaf.py").write_text(
                "from pathlib import Path\n"
                "import time\n"
                "ROOT = Path(__file__).parent\n"
                "class NpuRuntime:\n"
                "    def __init__(self, **kw): self.device = kw['device_id']\n"
                "    def link_to(self, address): self.address = address\n"
                "    def initialize(self): pass\n"
                "    def initialize_ddr(self, size, slot):\n"
                "        active = ROOT / 'init-active'\n"
                "        active.mkdir()\n"
                "        try:\n"
                "            time.sleep(0.02)\n"
                "            self.size = size\n"
                "            (ROOT / f'ready-{self.device}').touch()\n"
                "        finally: active.rmdir()\n"
                "    def offload_region(self):\n"
                "        return dict(address=self.address, epoch=1, region_id=self.device + 1,\n"
                "                    base=self.device * self.size, bytes=self.size)\n"
                "    def shutdown(self):\n"
                "        assert len(list(ROOT.glob('ready-*'))) == 8\n"
            )
            (root / "bin").mkdir()
            for kind in ("get", "put", "get_raw_addr"):
                (root / "bin" / f"kvaf_execute_batch_{kind}_kernels.o").touch()
            env = dict(os.environ, PYTHONPATH=str(root))
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--cpu", "fake:1", "--build-dir", str(root)],
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.count('"event": "READY"'), 8)
            self.assertEqual(result.stdout.count('"event": "SHUTDOWN_PASS"'), 8)
            self.assertIn('"event": "SERIAL_RESIDENCY_PASS"', result.stdout)
            for path in root.glob("ready-*"):
                path.unlink()
            init_spec = importlib.util.spec_from_file_location(
                "init_coordinator_for_spawn_test",
                SCRIPT.parent.parent / "vllm_ascend/distributed/kv_transfer/kvaf/pd/initialization.py",
            )
            init_module = importlib.util.module_from_spec(init_spec)
            init_spec.loader.exec_module(init_module)
            server = init_module.InitializationServer(("127.0.0.1", 0))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--cpu",
                        "fake:1",
                        "--build-dir",
                        str(root),
                        "--parallel",
                        "--initialization-coordinator",
                        f"127.0.0.1:{server.server_address[1]}",
                    ],
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.count('"event": "READY"'), 8)
            self.assertIn('"event": "PARALLEL_RESIDENCY_PASS"', result.stdout)

    def test_regions_must_be_distinct_and_disjoint(self):
        MODULE.validate_tickets([ready(0), ready(1)], 16, "k:1")
        for field, value in (("region_id", 1), ("base", 8), ("address", "other:1"), ("bytes", 32)):
            records = [ready(0), ready(1)]
            records[1]["ticket"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                MODULE.validate_tickets(records, 16, "k:1")

    def test_worker_holds_runtime_until_release(self):
        runtime = Mock()
        runtime.offload_region.return_value = ready(0)["ticket"]
        pipe = Mock()

        def release():
            runtime.shutdown.assert_not_called()
            self.assertEqual(pipe.send.call_args.args[0]["event"], "READY")
            return "RELEASE"

        pipe.recv.side_effect = release
        args = NS(cpu="k:1", ddr_bytes=16, slot_size=4, initialization_coordinator="")
        with (
            patch.object(MODULE, "create_runtime", return_value=runtime),
            patch.object(MODULE.signal, "signal"),
            patch.object(MODULE, "emit"),
        ):
            MODULE.worker(args, 0, pipe)
        runtime.initialize_ddr.assert_called_once_with(16, 4)
        runtime.shutdown.assert_called_once()

    def test_initialization_failure_reports_stage_and_cleans_up(self):
        runtime = Mock()
        runtime.initialize_ddr.side_effect = RuntimeError("control timeout")
        pipe = Mock()
        args = NS(cpu="k:1", ddr_bytes=16, slot_size=4, initialization_coordinator="")
        with (
            patch.object(MODULE, "create_runtime", return_value=runtime),
            patch.object(MODULE.signal, "signal"),
            patch.object(MODULE, "emit"),
            patch.object(MODULE.traceback, "print_exc"),
        ):
            MODULE.worker(args, 0, pipe)
        self.assertEqual(
            pipe.send.call_args.args[0],
            {
                "event": "FAIL",
                "device": 0,
                "stage": "initialize_ddr",
            },
        )
        pipe.recv.assert_not_called()
        runtime.shutdown.assert_called_once()

    def test_shutdown_failure_retains_worker_instead_of_claiming_pass(self):
        runtime = Mock()
        runtime.offload_region.return_value = ready(0)["ticket"]
        runtime.shutdown.side_effect = RuntimeError("unresolved transport")
        pipe = Mock()
        pipe.recv.return_value = "RELEASE"
        args = NS(cpu="k:1", ddr_bytes=16, slot_size=4, initialization_coordinator="")
        # Stop the intentional retain loop only inside this unit test.
        with (
            patch.object(MODULE, "create_runtime", return_value=runtime),
            patch.object(MODULE.signal, "signal"),
            patch.object(MODULE, "emit") as emit,
            patch.object(MODULE.traceback, "print_exc"),
            patch.object(MODULE.time, "sleep", side_effect=SystemExit("test retain loop")),
            self.assertRaises(SystemExit),
        ):
            MODULE.worker(args, 0, pipe)
        self.assertEqual([call.args[0] for call in emit.call_args_list], ["RETAINED"])
        pipe.close.assert_not_called()

    def coordinator(self, failure=None):
        timeline = []
        context = Mock()

        def process_factory(target, args):
            device = args[1]
            process = Mock(pid=100 + device, exitcode=0)
            process.start.side_effect = lambda: timeline.append(("start", device))
            process.is_alive.side_effect = lambda: ("release", device) not in timeline
            return process

        def pipe_factory():
            device = context.Pipe.call_count - 1
            parent, child = Mock(), Mock()
            parent.poll.return_value = True
            parent.recv.return_value = {"event": "FAIL", "device": device} if device == failure else ready(device)
            parent.send.side_effect = lambda command: timeline.append(("release", device))
            return parent, child

        context.Process.side_effect = process_factory
        context.Pipe.side_effect = pipe_factory
        args = NS(devices=list(range(8)), init_timeout=1, cleanup_timeout=1, ddr_bytes=16, cpu="k:1", parallel=False)
        with patch.object(MODULE, "emit") as emit:
            result = MODULE.run(args, context)
        return result, timeline, emit, context

    def test_all_eight_ready_before_first_release(self):
        result, timeline, emit, context = self.coordinator()
        self.assertEqual(result, 0)
        self.assertEqual(timeline[:8], [("start", device) for device in range(8)])
        self.assertEqual(timeline[8:], [("release", device) for device in range(8)])
        self.assertIn("ALL_RESIDENT_PASS", [call.args[0] for call in emit.call_args_list])
        for call in context.Process.call_args_list:
            self.assertEqual(call.kwargs["target"], MODULE.worker)

    def test_failure_stops_starting_and_releases_previous_workers(self):
        result, timeline, emit, _ = self.coordinator(failure=2)
        self.assertEqual(result, 1)
        self.assertEqual(
            timeline, [("start", 0), ("start", 1), ("start", 2), ("release", 0), ("release", 1), ("release", 2)]
        )
        self.assertNotIn("ALL_RESIDENT_PASS", [call.args[0] for call in emit.call_args_list])


if __name__ == "__main__":
    unittest.main()
