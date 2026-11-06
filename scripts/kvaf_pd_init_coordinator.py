# SPDX-License-Identifier: Apache-2.0
"""CPU-only entry point; loading by path avoids importing vLLM/torch on K."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "vllm_ascend/distributed/kv_transfer/kvaf/pd/initialization.py"),
        run_name="__main__",
    )
