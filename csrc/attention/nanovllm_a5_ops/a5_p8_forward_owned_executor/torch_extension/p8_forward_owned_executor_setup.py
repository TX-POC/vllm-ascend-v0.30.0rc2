"""Build only the independent P8 forward-owned ACLNN bridge."""

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
        "P8 requires the active Ascend torch_npu environment; do not build "
        "this bridge with a host-only Python."
    ) from exc


ROOT = Path(__file__).resolve().parent
OPS_ROOT = ROOT.parents[1]
COMMON_CSRC = OPS_ROOT / "torch_extension" / "csrc"
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent
P8_OPP_ROOT = Path(
    os.environ.get("NANOVLLM_A5_P8_FORWARD_OWNED_OPP", "")
).expanduser()
P8_ACLNN_INCLUDE = P8_OPP_ROOT / "vendors" / "customize" / "op_api" / "include"
P8_ACLNN_HEADER = P8_ACLNN_INCLUDE / "aclnn_a5_p8_forward_owned_executor.h"

# P8's Stage-B bridge is never allowed to infer a workspace ABI from a stale
# DSO or an ad-hoc tuple.  The runner constructs this OPP in its unique run
# directory and exports the exact root before building the bridge.  Refuse to
# build if its generated public header is absent.
if not P8_ACLNN_HEADER.is_file():
    raise RuntimeError(
        "P8 requires a freshly generated Stage-B ACLNN header at "
        f"{P8_ACLNN_HEADER}; build the exact P8 Stage-B OPP first and export "
        "NANOVLLM_A5_P8_FORWARD_OWNED_OPP."
    )

MAX_JOBS = int(
    os.getenv(
        "NANOVLLM_A5_P8_FORWARD_OWNED_EXECUTOR_BUILD_JOBS",
        multiprocessing.cpu_count(),
    )
)


setup(
    name="nanovllm_p8_forward_owned_executor",
    version="0.1.0",
    # The source root must not depend on the shell CWD.  The P8 runner uses a
    # new --build-lib for every attempt; explicitly copy this package's
    # __init__.py beside _C so provenance and OPP environment binding cannot
    # be bypassed by namespace-package resolution.
    packages=find_packages(where=str(ROOT)),
    package_dir={"": str(ROOT)},
    ext_modules=[
        NpuExtension(
            name="nanovllm_p8_forward_owned_executor._C",
            sources=[str(ROOT / "csrc" / "p8_forward_owned_executor.cpp")],
            extra_compile_args=[
                f"-I{TORCH_NPU_ROOT / 'include' / 'third_party' / 'acl' / 'inc'}",
                f"-I{P8_ACLNN_INCLUDE}",
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
