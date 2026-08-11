import multiprocessing
import os
from pathlib import Path

from setuptools import find_packages, setup

try:
    import torch
    import torch_npu
    from torch_npu.utils.cpp_extension import NpuExtension
    from torch.utils.cpp_extension import (
        BuildExtension,
        verify_ninja_availability,
    )
except ImportError as exc:
    raise RuntimeError(
        "nanovllm_dsa_a5 build requires torch and torch_npu in the current "
        "Python environment. Activate the same vLLM-Ascend environment used "
        "to run EngineCore, then install with `python -m pip install -e . "
        "--no-build-isolation`."
    ) from exc


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent
USE_NINJA = os.getenv("USE_NINJA") == "1"
MAX_JOBS = int(os.getenv("MAX_JOBS", multiprocessing.cpu_count()))

print(f"[nanovllm_dsa_a5] python torch: {torch.__version__}")
print(f"[nanovllm_dsa_a5] torch_npu: {torch_npu.__file__}")
print(
    "[nanovllm_dsa_a5] build isolation must be disabled because torch_npu "
    "is supplied by the active Ascend environment"
)

if USE_NINJA:
    verify_ninja_availability()

setup(
    name="nanovllm_dsa_a5",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        NpuExtension(
            name="nanovllm_dsa_a5._C",
            sources=[
                str(CSRC / "ops.cpp"),
                str(CSRC / "full_block_dump.cpp"),
                str(CSRC / "packed_full_block_dump.cpp"),
                str(CSRC / "op_api_common.cpp"),
            ],
            extra_compile_args=[
                f"-I{TORCH_NPU_ROOT / 'include' / 'third_party' / 'acl' / 'inc'}",
                f"-I{CSRC}",
                "-O3",
                "-std=c++17",
                "-fvisibility=hidden",
            ],
            extra_link_args=["-ldl"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=USE_NINJA,
            parallel=MAX_JOBS,
        )
    },
)
