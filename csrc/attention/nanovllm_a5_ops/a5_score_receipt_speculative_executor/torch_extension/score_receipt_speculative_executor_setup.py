"""Build only the isolated P7.1 CandidateSeed/PostOnly/CQ bridge."""

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
        "P7.1 requires the active Ascend torch_npu environment; do not build "
        "this bridge with a host-only Python."
    ) from exc


ROOT = Path(__file__).resolve().parent
OPS_ROOT = ROOT.parents[1]
COMMON_CSRC = OPS_ROOT / "torch_extension" / "csrc"
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent
MAX_JOBS = int(
    os.getenv(
        "NANOVLLM_A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_BUILD_JOBS",
        multiprocessing.cpu_count(),
    )
)


setup(
    name="nanovllm_score_receipt_speculative_executor",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        NpuExtension(
            name="nanovllm_score_receipt_speculative_executor._C",
            sources=[str(ROOT / "csrc" / "score_receipt_speculative_executor.cpp")],
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
