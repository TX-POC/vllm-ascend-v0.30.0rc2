"""Build only the P0 ExactScanSpeculativePost bridge, not vLLM-Ascend."""

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
        "The probe build requires the active Ascend Python environment with "
        "torch and torch_npu."
    ) from exc


ROOT = Path(__file__).resolve().parent
OPS_ROOT = ROOT.parents[1]
COMMON_CSRC = OPS_ROOT / "torch_extension" / "csrc"
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent
MAX_JOBS = int(os.getenv("NANOVLLM_A5_PROBE_BUILD_JOBS", multiprocessing.cpu_count()))

setup(
    name="nanovllm_exact_scan_probe",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        NpuExtension(
            name="nanovllm_exact_scan_probe._C",
            sources=[str(ROOT / "csrc" / "exact_scan_speculative_post_probe.cpp")],
            extra_compile_args=[
                f"-I{TORCH_NPU_ROOT / 'include' / 'third_party' / 'acl' / 'inc'}",
                f"-I{COMMON_CSRC}",
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
