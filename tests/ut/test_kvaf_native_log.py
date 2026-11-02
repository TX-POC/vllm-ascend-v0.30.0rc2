# SPDX-License-Identifier: Apache-2.0
"""Real subprocess/stdio checks; no NPU, model, or KVAF SDK required."""

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "vllm_ascend/distributed/kv_transfer/kvaf/native_log.py"


class TestKVAFNativeLog(unittest.TestCase):
    def run_child(self, code):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code), str(MODULE)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_gil_held_stdio_burst_and_non_native_output(self):
        result = self.run_child("""
            import ctypes, logging, os, runpy, sys, tempfile
            from pathlib import Path

            NativeLog = runpy.run_path(sys.argv[1])["NativeLog"]
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "native.log"
                relay = NativeLog(path)
                relay.start()
                process = relay._process
                relay.start()
                assert relay._process is process
                logging.basicConfig(level=logging.INFO, format="INFO [Worker_TP0] %(message)s")
                logging.info("normal vllm log")
                os.write(1, b"stdout unchanged\\n")
                os.write(2, b"[KVAF_")
                os.write(2, b"AIV] split native line\\n")
                # PyDLL deliberately holds the GIL. This single C stdio call
                # exceeds pipe capacity and would deadlock a Python reader thread.
                libc = ctypes.PyDLL(None)
                name = "__stderrp" if sys.platform == "darwin" else "stderr"
                stderr = ctypes.c_void_p.in_dll(libc, name)
                libc.fwrite.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
                libc.fwrite.restype = ctypes.c_size_t
                payload = b"[KVAF_AIV] " + b"x" * (1024 * 1024) + b"\\n"
                assert libc.fwrite(payload, 1, len(payload), stderr) == len(payload)
                os.write(2, b"native-free diagnostic\\n[KVAF_AIV] final without newline")
                relay.close()
                relay.close()
                assert process.poll() == 0
                expected = b"[KVAF_AIV] split native line\\n" + payload + b"[KVAF_AIV] final without newline"
                assert path.read_bytes() == expected
                os.write(2, b"restored stderr\\n")
            """)
        self.assertEqual(result.stdout, "stdout unchanged\n")
        self.assertEqual(result.stderr, "INFO [Worker_TP0] normal vllm log\nnative-free diagnostic\nrestored stderr\n")

    def test_failed_start_leaves_stderr_and_descriptors_intact(self):
        result = self.run_child("""
            import os, runpy, subprocess, sys, tempfile
            from pathlib import Path
            from unittest.mock import patch

            NativeLog = runpy.run_path(sys.argv[1])["NativeLog"]
            with tempfile.TemporaryDirectory() as directory:
                relay = NativeLog(Path(directory) / "native.log")
                before = set(os.listdir("/dev/fd"))
                with patch.object(subprocess, "Popen", side_effect=OSError("forced spawn failure")):
                    try:
                        relay.start()
                    except OSError:
                        pass
                    else:
                        raise AssertionError("spawn should fail")
                assert relay._process is None
                assert set(os.listdir("/dev/fd")) == before
                relay.close()
                os.write(2, b"stderr intact\\n")
            """)
        self.assertEqual(result.stderr, "stderr intact\n")


if __name__ == "__main__":
    unittest.main()
