import multiprocessing
import os
from pathlib import Path

from setuptools import find_packages, setup

try:
    import torch  # noqa: F401
    import torch_npu
    from torch.utils.cpp_extension import BuildExtension, verify_ninja_availability
    from torch_npu.utils.cpp_extension import NpuExtension
except ImportError as exc:
    raise RuntimeError(
        "nanovllm_dsa_lidu_fused_batch_get_prepare_mtp requires torch and torch_npu "
        "from the active Ascend runtime"
    ) from exc


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent
USE_NINJA = os.getenv("USE_NINJA") == "1"
MAX_JOBS = int(os.getenv("MAX_JOBS", multiprocessing.cpu_count()))

if USE_NINJA:
    verify_ninja_availability()

setup(
    name="nanovllm_dsa_lidu_fused_batch_get_prepare_mtp",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        NpuExtension(
            name="nanovllm_dsa_lidu_fused_batch_get_prepare_mtp._C",
            sources=[str(CSRC / "lidu_fused_batch_get_prepare_mtp_ops.cpp")],
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
