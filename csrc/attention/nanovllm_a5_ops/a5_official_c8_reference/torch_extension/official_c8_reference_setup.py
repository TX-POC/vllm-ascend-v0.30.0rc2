"""Build the isolated source-matched official C8 ACLNN bridge.

This is purposefully independent from vLLM-Ascend's root extension.  The
matching OPP must be built first by ``build_reference_opp.sh``; the bridge
uses ``NANOVLLM_CUST_OPAPI_LIB`` at runtime to resolve that OPP's ACLNN APIs.
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
        "The official C8 reference bridge requires torch and torch_npu."
    ) from exc


ROOT = Path(__file__).resolve().parent
OPS_ROOT = ROOT.parents[1]
COMMON_CSRC = OPS_ROOT / "torch_extension" / "csrc"
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent
MAX_JOBS = int(
    os.getenv("A5_OFFICIAL_C8_REFERENCE_BRIDGE_BUILD_JOBS", multiprocessing.cpu_count())
)


setup(
    name="nanovllm_official_c8_reference",
    version="0.1.0",
    # The P8 runner invokes this setup script from its own work directory and
    # places the result under a unique ``--build-lib`` path.  Discovering
    # packages from the caller's CWD silently omits ``__init__.py`` there,
    # turning this bridge into a namespace package (``__file__ is None``).
    # Anchor both discovery and source mapping at this setup file instead.
    packages=find_packages(where=str(ROOT)),
    package_dir={"": str(ROOT)},
    ext_modules=[
        NpuExtension(
            name="nanovllm_official_c8_reference._C",
            sources=[str(ROOT / "csrc" / "official_c8_reference.cpp")],
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
