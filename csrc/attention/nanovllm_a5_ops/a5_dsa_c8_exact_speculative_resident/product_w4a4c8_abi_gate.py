"""Fail-closed source gate for the A5 W4A4C8 Lightning Indexer path.

This module deliberately distinguishes the *actual* product frontend from
similarly named CANN sources that have a different declared ABI.  The DSA
product path is the W4A4C8 route exposed as
``torch_npu.npu_quant_lightning_indexer``.  Its query/key tensors are FP8,
its indexer-weight tensor is BF16, and its scale tensors are FP32 metadata.

The FP32 scale tensors do not make this an FP32 product route.  In particular,
this gate never widens the BF16 weights, never selects the main-DSA FP32
callable, and never treats the generic checked-in VllmQuant source as a
derivation target.  A Stage-A OPP build must wait for a source/OPP provenance
record for the exact runtime operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


PRODUCT_EXTENSION = Path("vllm_ascend/dsa_offload/a5_extension.py")
PRODUCT_LAYOUT = Path("vllm_ascend/dsa_offload/dsa_row.py")
W4A4C8_LAYOUT_ID = "A5_C8_W4A4C8_V1"
PRIMARY_CALLABLE = "torch_npu.npu_quant_lightning_indexer"
FALLBACK_CALLABLE = "torch.ops._C_ascend.npu_lightning_indexer_quant"


class GateFailure(RuntimeError):
    """A missing proof that must block an incompatible Stage-A build."""


@dataclass(frozen=True)
class ProductW4A4C8Abi:
    """The product callable ABI, without a synthetic FP32 reference path."""

    product_vllm_root: Path
    product_source_sha: str
    architecture: str
    product_callable: str
    fallback_callable: str
    query_key_dtype: str
    indexer_weight_dtype: str
    scale_dtype: str


def _require_file(path: Path, reason: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise GateFailure(f"{reason}:{resolved}")
    return resolved


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GateFailure(f"git_head_unavailable:{root}")
    head = result.stdout.strip()
    if len(head) != 40:
        raise GateFailure(f"invalid_git_head:{root}:{head}")
    return head


def _require_anchors(path: Path, reason: str, anchors: tuple[str, ...]) -> None:
    text = _require_file(path, reason).read_text(encoding="utf-8")
    missing = [anchor for anchor in anchors if anchor not in text]
    if missing:
        raise GateFailure(f"{reason}_anchor_missing:{','.join(missing)}")


def verify_product_w4a4c8_source(product_vllm_root: Path) -> ProductW4A4C8Abi:
    """Prove the checked-in DSA frontend is the W4A4C8 product route.

    This is intentionally a source-only check: it verifies the product
    frontend before importing Torch/CANN or allocating an NPU tensor.  It does
    not claim that the underlying CANN OPP source is available; that is a
    separate required provenance step before Stage-A derivation.
    """

    root = product_vllm_root.expanduser().resolve()
    if not root.is_dir():
        raise GateFailure(f"product_vllm_root_missing:{root}")
    _require_anchors(
        root / PRODUCT_EXTENSION,
        "w4a4c8_runtime_source",
        (
            "def _check_c8_lidu_inputs(",
            "query.dtype != fp8 or key.dtype != fp8",
            "weights.dtype != torch.bfloat16",
            "query_dequant_scale.dtype != torch.float32",
            "key_dequant_scale.dtype != torch.float32",
            'getattr(torch_npu, "npu_quant_lightning_indexer", None)',
            "query_quant_mode=0",
            "key_quant_mode=0",
            "sparse_count=SPARSE_COUNT",
        ),
    )
    _require_anchors(
        root / PRODUCT_LAYOUT,
        "w4a4c8_layout_source",
        (f'M0_LAYOUT_ID = "{W4A4C8_LAYOUT_ID}"',),
    )
    return ProductW4A4C8Abi(
        product_vllm_root=root,
        product_source_sha=_git_head(root),
        architecture=W4A4C8_LAYOUT_ID,
        product_callable=PRIMARY_CALLABLE,
        fallback_callable=FALLBACK_CALLABLE,
        query_key_dtype="float8_e4m3fn",
        indexer_weight_dtype="bfloat16",
        scale_dtype="float32_scale_metadata",
    )
