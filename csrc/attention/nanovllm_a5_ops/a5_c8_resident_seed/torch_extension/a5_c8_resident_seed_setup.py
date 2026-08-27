"""Build the isolated A5C8ResidentSeed ACLNN bridge only.

The matching OPP must first be built by
``build_isolated_a5_c8_resident_seed.sh``.  At runtime this extension resolves
only ``NANOVLLM_A5_RESIDENT_SEED_OPAPI_LIB``; it never consumes the official
C8 or P3/P4 resolver variables.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

from setuptools import find_packages, setup

try:
    import torch
    import torch_npu
    from torch.utils.cpp_extension import BuildExtension
    from torch_npu.utils.cpp_extension import NpuExtension
except ImportError as exc:
    raise RuntimeError(
        "The A5C8ResidentSeed bridge requires torch and torch_npu."
    ) from exc


ROOT = Path(__file__).resolve().parent
OPS_ROOT = ROOT.parents[1]
COMMON_CSRC = OPS_ROOT / "torch_extension" / "csrc"
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent
MAX_JOBS = int(
    os.getenv("A5_C8_RESIDENT_SEED_BRIDGE_BUILD_JOBS", multiprocessing.cpu_count())
)


setup(
    name="nanovllm_a5_c8_resident_seed",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        NpuExtension(
            name="nanovllm_a5_c8_resident_seed._C",
            sources=[str(ROOT / "csrc" / "a5_c8_resident_seed.cpp")],
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
