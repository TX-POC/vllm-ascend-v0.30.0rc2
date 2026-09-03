#!/usr/bin/env python3
"""NPU-only P10 baseline for one explicit product C8 frontend.

This is *not* a streaming-C8 build and it does not load a target P10 operator.
The caller must choose exactly one frontend:

* ``dsa_offload_bf16``: product BF16 weights are losslessly widened to FP32
  for the source-matched official reference;
* ``main_dsa_v1_fp32``: product and reference receive the same FP32 weights.

No fallback route, policy mutation, or transport operation participates.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import sys
from typing import Any


TARGET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TARGET_ROOT))

from product_bf16_abi_gate import (  # noqa: E402
    GateFailure,
    provenance_fields,
    verify_product_c8_frontend_source,
    verify_reference_artifacts,
)


TOPK = 2048
BLOCK_SIZE = 128
SOURCE_TOKENS = 4096
BATCH_SIZE = 2
INDEX_HEADS = 64
KEY_HEADS = 1
HEAD_DIM = 128
METADATA_WORDS = 1024
INT64_MAX = (1 << 63) - 1


def _fail(stage: str, error: BaseException) -> int:
    text = str(error).replace("\n", " ")[:900]
    print(
        "P10_PRODUCT_C8_REFERENCE_FAIL "
        f"stage={stage} type={type(error).__name__} text={text}",
        flush=True,
    )
    return 2


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise GateFailure(f"environment_missing:{name}")
    return value


def _sha16(value: Any) -> str:
    cpu = value.detach().to("cpu").contiguous()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()[:16]


def _require_npu(torch: Any) -> Any:
    physical = _required_environment("P10_PHYSICAL_NPU_ID")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    logical = os.environ.get("P10_LOGICAL_NPU_ID", "0")
    if logical != "0" or visible != physical:
        raise GateFailure(
            "npu_mapping_mismatch:"
            f"physical={physical}:visible={visible or '<unset>'}:logical={logical}"
        )
    if int(torch.npu.device_count()) != 1:
        raise GateFailure(f"expected_one_visible_npu:count={torch.npu.device_count()}")
    torch.npu.set_device(0)
    probe = torch.ones(1, device="npu:0").cpu().item()
    name = str(torch.npu.get_device_name(0))
    if probe != 1.0 or "950" not in name.lower():
        raise GateFailure(f"npu_probe_failed:name={name!r}:value={probe}")
    print(
        "P10_PRODUCT_C8_REFERENCE_NPU "
        f"physical={physical} logical=0 visible={visible!r} "
        f"device=npu:0 device_name={name!r}",
        flush=True,
    )
    return torch.device("npu:0")


def _module_under(module: Any, root: Path) -> None:
    module_file = Path(str(getattr(module, "__file__", ""))).resolve()
    try:
        module_file.relative_to(root)
    except ValueError as error:
        raise GateFailure(
            f"reference_bridge_imported_from_wrong_path:{module_file}:expected_root={root}"
        ) from error


def _load_reference(torch: Any, reference: Any) -> tuple[Any, Any]:
    module_name = "nanovllm_official_c8_reference"
    if module_name in sys.modules:
        raise GateFailure("reference_bridge_requires_fresh_python_process")
    inherited = [
        entry
        for entry in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")
        if entry
    ]
    vendor = str(reference.reference_vendor)
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(
        [vendor, *(entry for entry in inherited if entry != vendor)]
    )
    os.environ["NANOVLLM_A5_REFERENCE_C8_OPAPI_LIB"] = str(reference.opapi)
    # Build trees commonly contain both an in-place binary and a build/lib
    # copy.  Import from the parent of the receipt-selected module only;
    # broad source-tree insertion could silently choose a stale binary.
    bridge_package_root = reference.bridge_module.parent.parent
    sys.path.insert(0, str(bridge_package_root))
    module = importlib.import_module(module_name)
    _module_under(module, reference.bridge_root)
    extension = importlib.import_module(f"{module_name}._C")
    extension_file = Path(str(getattr(extension, "__file__", ""))).resolve()
    if extension_file != reference.bridge_module:
        raise GateFailure(
            "reference_bridge_extension_path_mismatch:"
            f"actual={extension_file}:expected={reference.bridge_module}"
        )
    namespace = getattr(torch.ops, "a5_official_c8_reference", None)
    indexer = getattr(namespace, "npu_vllm_quant_lightning_indexer", None)
    metadata = getattr(namespace, "npu_vllm_quant_lightning_indexer_metadata", None)
    if not callable(indexer) or not callable(metadata):
        raise GateFailure("reference_bridge_op_registration_missing")
    return indexer, metadata


def _quantize_fp8(torch: Any, torch_npu: Any, value: Any) -> tuple[Any, Any]:
    quantized, scale = torch_npu.npu_dynamic_quant(
        value, dst_type=torch.float8_e4m3fn
    )
    return (
        quantized.contiguous(),
        scale.view(value.shape[:-1]).to(torch.float32).contiguous(),
    )


def _inputs(
    torch: Any, torch_npu: Any, device: Any, product_weight_dtype: str
) -> dict[str, Any]:
    blocks = SOURCE_TOKENS // BLOCK_SIZE
    torch.manual_seed(20260910)
    torch.npu.manual_seed_all(20260910)
    query_bf16 = torch.empty(
        (BATCH_SIZE, INDEX_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-1.0, 1.0)
    key_bf16 = torch.empty(
        (blocks, BLOCK_SIZE, KEY_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-1.0, 1.0)
    query, query_scale = _quantize_fp8(torch, torch_npu, query_bf16)
    key, key_scale = _quantize_fp8(torch, torch_npu, key_bf16)
    weight_dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(product_weight_dtype)
    if weight_dtype is None:
        raise GateFailure(f"unsupported_product_weight_dtype:{product_weight_dtype}")
    product_weights = torch.empty(
        (BATCH_SIZE, INDEX_HEADS), dtype=weight_dtype, device=device
    ).uniform_(0.01, 1.0).contiguous()
    actual_query_lengths = torch.arange(
        1, BATCH_SIZE + 1, dtype=torch.int32, device=device
    )
    actual_key_lengths = torch.tensor(
        [SOURCE_TOKENS, SOURCE_TOKENS - BLOCK_SIZE], dtype=torch.int32, device=device
    )
    block_table = torch.stack(
        [torch.randperm(blocks, dtype=torch.int64).to(torch.int32) for _ in range(BATCH_SIZE)]
    ).to(device).contiguous()
    return {
        "query": query,
        "key": key,
        "product_weights": product_weights,
        "query_dequant_scale": query_scale,
        "key_dequant_scale": key_scale,
        "actual_seq_lengths_query": actual_query_lengths,
        "actual_seq_lengths_key": actual_key_lengths,
        "block_table": block_table,
    }


def _metadata(torch: Any, metadata_op: Any, inputs: dict[str, Any], label: str) -> Any:
    metadata = metadata_op(
        num_heads_q=INDEX_HEADS,
        num_heads_k=KEY_HEADS,
        head_dim=HEAD_DIM,
        query_quant_mode=0,
        key_quant_mode=0,
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_key=inputs["actual_seq_lengths_key"],
        batch_size=BATCH_SIZE,
        max_seqlen_q=1,
        max_seqlen_k=SOURCE_TOKENS,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        cmp_ratio=1,
        device="npu:0",
    )
    if not (
        getattr(metadata, "dtype", None) == torch.int32
        and int(metadata.numel()) == METADATA_WORDS
    ):
        raise GateFailure(f"{label}_metadata_abi_mismatch:{metadata!r}")
    return metadata.contiguous()


def _normalize_product_indices(torch: Any, value: Any) -> Any:
    indices = value[0] if isinstance(value, tuple) else value
    if not isinstance(indices, torch.Tensor) or indices.dtype != torch.int32:
        raise GateFailure("product_output_not_int32_tensor")
    expected = BATCH_SIZE * KEY_HEADS * TOPK
    if int(indices.numel()) != expected:
        raise GateFailure(
            f"product_output_numel_mismatch:actual={indices.numel()}:expected={expected}"
        )
    return indices.reshape(BATCH_SIZE, KEY_HEADS, TOPK).contiguous()


def _first_difference(torch: Any, expected: Any, actual: Any) -> str:
    mismatch = torch.nonzero(expected != actual, as_tuple=False)
    if mismatch.numel() == 0:
        return "<none>"
    coord = mismatch[0].to("cpu").tolist()
    index = tuple(int(item) for item in coord)
    return (
        f"coord={coord}:product={int(actual[index].item())}:"
        f"reference={int(expected[index].item())}:count={int(mismatch.shape[0])}"
    )


def _product_ops(torch: Any, torch_npu: Any, product: Any) -> tuple[Any, Any | None]:
    if product.frontend == "dsa_offload_bf16":
        indexer = getattr(torch_npu, "npu_quant_lightning_indexer", None)
        if not callable(indexer):
            raise GateFailure("product_primary_callable_unavailable")
        return indexer, None
    if product.frontend == "main_dsa_v1_fp32":
        namespace = getattr(torch.ops, "_C_ascend", None)
        indexer = getattr(namespace, "npu_vllm_quant_lightning_indexer", None)
        metadata = getattr(
            namespace, "npu_vllm_quant_lightning_indexer_metadata", None
        )
        if not callable(indexer) or not callable(metadata):
            raise GateFailure("product_main_dsa_v1_callable_unavailable")
        return indexer, metadata
    raise GateFailure(f"product_frontend_not_implemented:{product.frontend}")


def _call_product(
    torch: Any,
    product: Any,
    product_op: Any,
    product_metadata_op: Any | None,
    inputs: dict[str, Any],
) -> Any:
    common = {
        "query": inputs["query"],
        "key": inputs["key"],
        "weights": inputs["product_weights"],
        "query_dequant_scale": inputs["query_dequant_scale"],
        "key_dequant_scale": inputs["key_dequant_scale"],
        "actual_seq_lengths_query": inputs["actual_seq_lengths_query"],
        "actual_seq_lengths_key": inputs["actual_seq_lengths_key"],
        "block_table": inputs["block_table"],
        "query_quant_mode": 0,
        "key_quant_mode": 0,
        "layout_query": "TND",
        "layout_key": "PA_BSND",
        "sparse_count": TOPK,
        "sparse_mode": 3,
    }
    if product.frontend == "dsa_offload_bf16":
        return _normalize_product_indices(torch, product_op(**common))
    if product.frontend == "main_dsa_v1_fp32":
        if product_metadata_op is None:
            raise GateFailure("product_main_dsa_v1_metadata_missing")
        product_metadata = _metadata(
            torch, product_metadata_op, inputs, "product"
        )
        return _normalize_product_indices(
            torch,
            product_op(
                **common,
                metadata=product_metadata,
                pre_tokens=INT64_MAX,
                next_tokens=INT64_MAX,
                cmp_ratio=1,
                return_value=False,
            ),
        )
    raise GateFailure(f"product_frontend_not_implemented:{product.frontend}")


def main() -> int:
    try:
        product = verify_product_c8_frontend_source(
            Path(_required_environment("P10_PRODUCT_VLLM_ROOT")),
            _required_environment("P10_PRODUCT_C8_FRONTEND"),
        )
        reference = verify_reference_artifacts(
            reference_vllm_root=Path(
                _required_environment("P10_REFERENCE_VLLM_ROOT")
            ),
            reference_opapi=Path(_required_environment("P10_REFERENCE_OPAPI")),
            reference_bridge_root=Path(
                _required_environment("P10_REFERENCE_BRIDGE_ROOT")
            ),
            reference_bridge_module=Path(
                _required_environment("P10_REFERENCE_BRIDGE_MODULE")
            ),
            expected_source_sha=os.environ.get("P10_REFERENCE_SOURCE_SHA") or None,
        )
    except BaseException as error:
        return _fail("preflight", error)

    print(provenance_fields(product, reference), flush=True)
    try:
        import torch
        import torch_npu

        device = _require_npu(torch)
        product_op, product_metadata_op = _product_ops(torch, torch_npu, product)
        reference_op, metadata_op = _load_reference(torch, reference)
        inputs = _inputs(torch, torch_npu, device, product.product_weight_dtype)
        product_weights = inputs["product_weights"]
        if product.product_weight_dtype == "bfloat16":
            if product_weights.dtype != torch.bfloat16:
                raise GateFailure(
                    f"product_weights_not_bfloat16:{product_weights.dtype}"
                )
            reference_weights = product_weights.to(torch.float32).contiguous()
            if not torch.equal(reference_weights.to(torch.bfloat16), product_weights):
                raise GateFailure("reference_weight_widen_not_lossless")
        elif product.product_weight_dtype == "float32":
            if product_weights.dtype != torch.float32:
                raise GateFailure(
                    f"product_weights_not_float32:{product_weights.dtype}"
                )
            reference_weights = product_weights
        else:
            raise GateFailure(
                f"unsupported_product_weight_dtype:{product.product_weight_dtype}"
            )
        if reference_weights.dtype != torch.float32:
            raise GateFailure(
                f"reference_weights_not_float32:{reference_weights.dtype}"
            )
        metadata = _metadata(torch, metadata_op, inputs, "reference")
        product_indices = _call_product(
            torch, product, product_op, product_metadata_op, inputs
        )
        reference_result = reference_op(
            query=inputs["query"],
            key=inputs["key"],
            weights=reference_weights,
            query_dequant_scale=inputs["query_dequant_scale"],
            key_dequant_scale=inputs["key_dequant_scale"],
            actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
            actual_seq_lengths_key=inputs["actual_seq_lengths_key"],
            block_table=inputs["block_table"],
            metadata=metadata,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=TOPK,
            sparse_mode=3,
            pre_tokens=INT64_MAX,
            next_tokens=INT64_MAX,
            cmp_ratio=1,
            return_value=False,
        )
        reference_indices = _normalize_product_indices(torch, reference_result)
        torch.npu.synchronize()
        if not torch.equal(product_indices, reference_indices):
            raise AssertionError(
                "sparse_indices_mismatch:"
                + _first_difference(torch, reference_indices, product_indices)
            )
        print(
            "P10_PRODUCT_C8_REFERENCE_EQ status=pass "
            f"frontend={product.frontend} "
            f"product_weight_dtype={product_weights.dtype} "
            f"reference_weight_dtype={reference_weights.dtype} "
            f"weight_conversion={product.weight_conversion} "
            f"product_sparse_sha16={_sha16(product_indices)} "
            f"reference_sparse_sha16={_sha16(reference_indices)}",
            flush=True,
        )
        return 0
    except BaseException as error:
        return _fail("runtime", error)


if __name__ == "__main__":
    raise SystemExit(main())
