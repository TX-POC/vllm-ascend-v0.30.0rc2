"""Build the dedicated P9 C8 + canonical-resident Get torch bridge.

The P9 OPP is built separately by ``build_p9_exact_resident_opp.sh``.  This
extension resolves only ``NANOVLLM_A5_EXACT_SPECULATIVE_RESIDENT_OPAPI_LIB``;
it never shares an ACLNN resolver domain with the official C8, ResidentSeed,
or retired P7/P8 bridges.
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
        "The P9 exact speculative resident bridge requires torch and torch_npu."
    ) from exc


ROOT = Path(__file__).resolve().parent
OPS_ROOT = ROOT.parents[1]
COMMON_CSRC = OPS_ROOT / "torch_extension" / "csrc"
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent
MAX_JOBS = int(
    os.getenv(
        "A5_C8_EXACT_SPECULATIVE_RESIDENT_BRIDGE_BUILD_JOBS",
        multiprocessing.cpu_count(),
    )
)


setup(
    name="nanovllm_a5_c8_exact_speculative_resident_get",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        NpuExtension(
            name="nanovllm_a5_c8_exact_speculative_resident_get._C",
            sources=[str(ROOT / "csrc" / "a5_c8_exact_speculative_resident_get.cpp")],
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
