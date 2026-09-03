"""Fail-closed provenance and ABI gate for the P10 product C8 comparison.

There are two legitimate product frontends and they do *not* share a raw
weight ABI:

* the DSA-offload frontend calls ``torch_npu`` with BF16 weights; its official
  C8 reference receives the lossless BF16-to-FP32 widening of that tensor;
* the main DSA frontend calls the Vllm C8 operator directly with FP32 weights;
  its reference receives the exact same FP32 tensor.

The caller must select one frontend explicitly.  This module never guesses
from a callable name, which prevents an accidental comparison of different
product paths.  It verifies the independently built reference artifacts
before an NPU fixture imports either bridge.

There is intentionally no CANN, torch, or NPU import here.  It is safe to run
as a source/artifact preflight on a host without an NPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
from typing import Mapping


PRODUCT_BF16_EXTENSION = Path("vllm_ascend/dsa_offload/a5_extension.py")
PRODUCT_FP32_DSA = Path("vllm_ascend/attention/dsa_v1.py")
PRODUCT_FP32_DEVICE_OP = Path("vllm_ascend/device/device_op.py")
REFERENCE_BRIDGE_SOURCE = Path(
    "csrc/attention/nanovllm_a5_ops/a5_official_c8_reference/"
    "torch_extension/csrc/official_c8_reference.cpp"
)
REFERENCE_OPS = (
    "vllm_quant_lightning_indexer,"
    "vllm_quant_lightning_indexer_metadata"
)
REFERENCE_OPAPI_MARKER = "P3_REFERENCE_PROVENANCE"
REFERENCE_BRIDGE_MARKER = "P10_REFERENCE_BRIDGE_PROVENANCE"


class GateFailure(RuntimeError):
    """A missing or ambiguous proof that must block the NPU comparison."""


@dataclass(frozen=True)
class ProductC8Abi:
    product_vllm_root: Path
    product_source_sha: str
    frontend: str
    product_callable: str
    product_metadata_callable: str | None
    product_weight_dtype: str
    reference_weight_dtype: str
    weight_conversion: str


@dataclass(frozen=True)
class ReferenceArtifacts:
    reference_vllm_root: Path
    reference_source_sha: str
    opapi: Path
    bridge_root: Path
    bridge_module: Path
    reference_vendor: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_key_value(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise GateFailure(f"cannot_read:{path}:{type(error).__name__}") from error
    values: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise GateFailure(f"invalid_marker:{path}")
        values[key] = value
    return values


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


def _require_file(path: Path, reason: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise GateFailure(f"{reason}:{resolved}")
    return resolved


def _require_directory(path: Path, reason: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise GateFailure(f"{reason}:{resolved}")
    return resolved


def _require_under(path: Path, root: Path, reason: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise GateFailure(f"{reason}:path={path}:root={root}") from error


def _require_source_anchors(path: Path, reason: str, required: tuple[str, ...]) -> None:
    text = _require_file(path, reason).read_text(encoding="utf-8")
    missing = [token for token in required if token not in text]
    if missing:
        raise GateFailure(f"{reason}_anchor_missing:{','.join(missing)}")


def verify_product_c8_frontend_source(
    product_vllm_root: Path, frontend: str
) -> ProductC8Abi:
    """Prove the selected product frontend before any device interaction."""

    root = _require_directory(product_vllm_root, "product_vllm_root_missing")
    if frontend == "dsa_offload_bf16":
        _require_source_anchors(
            root / PRODUCT_BF16_EXTENSION,
            "product_bf16_abi_source",
            (
                "def _check_c8_lidu_inputs(",
                "weights.dtype != torch.bfloat16",
                "C8 LIDU weights must be bfloat16 [B,N]",
                'getattr(torch_npu, "npu_quant_lightning_indexer", None)',
                "def _native_quant_lightning_indexer(",
                "weights=weights,",
                "query_dequant_scale=query_dequant_scale,",
                "key_dequant_scale=key_dequant_scale,",
            ),
        )
        return ProductC8Abi(
            product_vllm_root=root,
            product_source_sha=_git_head(root),
            frontend=frontend,
            product_callable="torch_npu.npu_quant_lightning_indexer",
            product_metadata_callable=None,
            product_weight_dtype="bfloat16",
            reference_weight_dtype="float32",
            weight_conversion="lossless_bfloat16_to_float32_widen",
        )
    if frontend == "main_dsa_v1_fp32":
        _require_source_anchors(
            root / PRODUCT_FP32_DSA,
            "product_fp32_dsa_source",
            (
                "torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(",
                "DeviceOperator.prepare_dsa_indexer_weights(weights)",
                "query_dequant_scale=DeviceOperator.prepare_dsa_indexer_query_scale",
                "key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale",
            ),
        )
        _require_source_anchors(
            root / PRODUCT_FP32_DEVICE_OP,
            "product_fp32_device_op_source",
            (
                "def prepare_dsa_indexer_weights(weights):",
                "A5: cast indexer weights to float32",
                "return weights.float()",
                "A5: cast query dequant scale to float32",
                "return q_scale.float()",
                "A5: cast key dequant scale to float32",
                "return indexer_scale_cache.squeeze(-2).float()",
            ),
        )
        return ProductC8Abi(
            product_vllm_root=root,
            product_source_sha=_git_head(root),
            frontend=frontend,
            product_callable="torch.ops._C_ascend.npu_vllm_quant_lightning_indexer",
            product_metadata_callable=(
                "torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata"
            ),
            product_weight_dtype="float32",
            reference_weight_dtype="float32",
            weight_conversion="identity_float32",
        )
    raise GateFailure(f"product_frontend_unsupported_or_missing:{frontend or '<empty>'}")


def verify_product_bf16_source(product_vllm_root: Path) -> ProductC8Abi:
    """Backward-compatible explicit BF16 frontend entry point."""

    return verify_product_c8_frontend_source(product_vllm_root, "dsa_offload_bf16")


def _require_bridge_module(bridge_root: Path, bridge_module: Path) -> Path:
    """Require the caller to identify the exact bridge binary.

    ``setuptools build_ext --inplace`` conventionally leaves both an in-place
    binary and a copy under ``build/lib``.  A recursive ``find`` therefore
    cannot safely infer which binary Python will import.  Treating the module
    path as an explicit artifact is shorter *and* safer than guessing.
    """

    module = _require_file(bridge_module, "reference_bridge_module_missing")
    _require_under(module, bridge_root, "reference_bridge_module_outside_root")
    if not module.name.startswith("_C") or module.suffix != ".so":
        raise GateFailure(f"reference_bridge_module_invalid_name:{module}")
    if module.parent.name != "nanovllm_official_c8_reference":
        raise GateFailure(f"reference_bridge_module_invalid_package:{module}")
    return module


def _require_marker_value(
    marker: Mapping[str, str], key: str, expected: str, path: Path
) -> None:
    actual = marker.get(key)
    if actual != expected:
        raise GateFailure(
            f"marker_value_mismatch:marker={path}:key={key}:"
            f"actual={actual or '<missing>'}:expected={expected}"
        )


def verify_reference_artifacts(
    *,
    reference_vllm_root: Path,
    reference_opapi: Path,
    reference_bridge_root: Path,
    reference_bridge_module: Path,
    expected_source_sha: str | None = None,
) -> ReferenceArtifacts:
    """Verify source, OPP, and bridge form one explicit reference closure."""

    root = _require_directory(reference_vllm_root, "reference_vllm_root_missing")
    source_sha = _git_head(root)
    if expected_source_sha and source_sha != expected_source_sha:
        raise GateFailure(
            "reference_source_sha_mismatch:"
            f"actual={source_sha}:expected={expected_source_sha}"
        )
    bridge_source = _require_file(
        root / REFERENCE_BRIDGE_SOURCE, "reference_bridge_source_missing"
    )
    opapi = _require_file(reference_opapi, "reference_opapi_missing")
    bridge_root = _require_directory(
        reference_bridge_root, "reference_bridge_root_missing"
    )
    bridge_module = _require_bridge_module(bridge_root, reference_bridge_module)
    vendor = opapi.parents[2]
    marker_path = _require_file(
        vendor / REFERENCE_OPAPI_MARKER, "reference_opapi_provenance_missing"
    )
    marker = _read_key_value(marker_path)
    _require_marker_value(marker, "format", "1", marker_path)
    _require_marker_value(marker, "kind", "source_matched_official_c8", marker_path)
    _require_marker_value(marker, "source_root", str(root), marker_path)
    _require_marker_value(marker, "source_head", source_sha, marker_path)
    _require_marker_value(marker, "operators", REFERENCE_OPS, marker_path)

    bridge_marker_path = _require_file(
        bridge_root / REFERENCE_BRIDGE_MARKER,
        "reference_bridge_provenance_missing",
    )
    bridge_marker = _read_key_value(bridge_marker_path)
    _require_marker_value(bridge_marker, "format", "1", bridge_marker_path)
    _require_marker_value(
        bridge_marker,
        "kind",
        "source_matched_official_c8_bridge",
        bridge_marker_path,
    )
    _require_marker_value(
        bridge_marker, "source_root", str(root), bridge_marker_path
    )
    _require_marker_value(
        bridge_marker, "source_head", source_sha, bridge_marker_path
    )
    _require_marker_value(
        bridge_marker,
        "opapi",
        str(opapi),
        bridge_marker_path,
    )
    _require_marker_value(
        bridge_marker,
        "opapi_sha256",
        _sha256(opapi),
        bridge_marker_path,
    )
    _require_marker_value(
        bridge_marker,
        "bridge_source",
        str(bridge_source),
        bridge_marker_path,
    )
    _require_marker_value(
        bridge_marker,
        "bridge_source_sha256",
        _sha256(bridge_source),
        bridge_marker_path,
    )
    _require_marker_value(
        bridge_marker,
        "bridge_module",
        str(bridge_module),
        bridge_marker_path,
    )
    _require_marker_value(
        bridge_marker,
        "bridge_module_sha256",
        _sha256(bridge_module),
        bridge_marker_path,
    )
    _require_under(bridge_module, bridge_root, "reference_bridge_module_outside_root")
    return ReferenceArtifacts(
        reference_vllm_root=root,
        reference_source_sha=source_sha,
        opapi=opapi,
        bridge_root=bridge_root,
        bridge_module=bridge_module,
        reference_vendor=vendor,
    )


def provenance_fields(product: ProductC8Abi, reference: ReferenceArtifacts) -> str:
    """One compact line suitable for a remote log or CI artifact."""

    return " ".join(
        (
            "P10_PRODUCT_C8_ABI_GATE status=pass",
            f"product_root={product.product_vllm_root}",
            f"product_sha={product.product_source_sha}",
            f"frontend={product.frontend}",
            f"product_callable={product.product_callable}",
            "product_metadata_callable=" + (
                product.product_metadata_callable or "none"
            ),
            f"product_weights={product.product_weight_dtype}",
            f"reference_root={reference.reference_vllm_root}",
            f"reference_sha={reference.reference_source_sha}",
            "reference_weights=float32",
            f"weight_conversion={product.weight_conversion}",
            f"reference_opapi={reference.opapi}",
            f"reference_bridge={reference.bridge_module}",
        )
    )
