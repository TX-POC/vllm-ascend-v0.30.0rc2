"""Build the fresh P10 forward graph bridge.

The bridge deliberately has no generated ACLNN-header dependency: all three
entry points are resolved from the single exact product OPP selected at runtime.
It therefore cannot accidentally compile against a header from a previous
package while resolving another package at execution time.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

from setuptools import find_packages, setup

try:
    import torch_npu
    from torch.utils.cpp_extension import BuildExtension
    from torch_npu.utils.cpp_extension import NpuExtension
except ImportError as exc:
    raise RuntimeError(
        "P10 forward bridge requires the active Ascend torch_npu environment."
    ) from exc


ROOT = Path(__file__).resolve().parent
# ``ROOT`` is ``.../nanovllm_a5_ops/a5_dsa.../p10_forward_bridge/torch_extension``.
# The shared OpAPI bridge headers live directly under ``nanovllm_a5_ops``;
# one more parent would instead point at ``csrc/attention/torch_extension``,
# which does not exist in a clean remote worktree.
OPS_ROOT = ROOT.parents[2]
COMMON_CSRC = OPS_ROOT / "torch_extension" / "csrc"
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent
MAX_JOBS = int(
    os.getenv("NANOVLLM_A5_P10_FORWARD_BRIDGE_BUILD_JOBS", multiprocessing.cpu_count())
)


setup(
    name="nanovllm_p10_forward_bridge",
    version="0.1.0",
    packages=find_packages(where=str(ROOT)),
    package_dir={"": str(ROOT)},
    ext_modules=[
        NpuExtension(
            name="nanovllm_p10_forward_bridge._C",
            sources=[str(ROOT / "csrc" / "p10_forward_bridge.cpp")],
            extra_compile_args=[
                f"-I{TORCH_NPU_ROOT / 'include' / 'third_party' / 'acl' / 'inc'}",
                f"-I{COMMON_CSRC}",
                "-D_GNU_SOURCE",
                "-O3",
                "-std=c++17",
                "-fvisibility=hidden",
            ],
            extra_link_args=["-ldl"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=os.getenv("USE_NINJA") == "1",
            parallel=MAX_JOBS,
        )
    },
)
