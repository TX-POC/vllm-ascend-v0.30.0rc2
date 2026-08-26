"""Static contract guard for the isolated P3 reconcile prototype.

This test intentionally does not import torch_npu or require an A5.  It stops
future edits from silently turning the transaction-plan op into a persistent
mapping writer or wiring it into the normal policy path.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OP_ROOT = (
    ROOT
    / "csrc"
    / "attention"
    / "nanovllm_a5_ops"
    / "a5_exact_reconcile_private_view"
)


def test_reconcile_schema_and_kernel_are_plan_only() -> None:
    schema = (OP_ROOT / "ops.json").read_text()
    kernel = (OP_ROOT / "op_kernel" / "a5_exact_reconcile_private_view.cpp").read_text()
    bridge = (
        OP_ROOT
        / "torch_extension"
        / "csrc"
        / "exact_reconcile_private_view.cpp"
    ).read_text()
    policy = (ROOT / "vllm_ascend" / "dsa_offload" / "a5_c8_policy.py").read_text()
    build_script = (
        ROOT / "csrc" / "attention" / "nanovllm_a5_ops" / "build_nanovllm_ops.sh"
    ).read_text()

    assert schema.count('"name":') == 23  # 9 inputs + 12 outputs + 2 attrs
    for name in (
        "kv_object_slots_snapshot",
        "kv_object_lens",
        "kv_object_map",
        "kv_object_ids",
        "ready_private_kv_object_ids",
        "ready_private_slots",
        "ready_private_destination_slots",
        "fallback_kv_object_ids",
        "fallback_destination_slots",
        "placement_delta",
        "score_source_lens",
    ):
        assert f'"name": "{name}"' in schema
    assert "kvObjectSlotsSnapshotGm_.SetValue" not in kernel
    assert "scoreSourceLensGm_" in kernel
    assert "scoreSourceCapacity" in kernel
    assert "kvObjectCapacity" in kernel
    assert '"kv_object_domain"' in schema
    assert '"cmp_ratio"' in schema
    assert "kv_object_domain=RAW_TOKEN requires cmp_ratio=1" in bridge
    host = (OP_ROOT / "op_host" / "a5_exact_reconcile_private_view.cpp").read_text()
    assert "kC8CompressedObjectDomain" in host
    assert "cmpRatio == 1 || cmpRatio == 4 || cmpRatio == 128" in host
    assert "IsValidKvObjectDomain" in host
    assert "tiling_->rowCount" in kernel
    assert "GetExactTopkRowCount" in (OP_ROOT / "op_host" / "a5_exact_reconcile_private_view.cpp").read_text()
    assert "ExactTopkRowCount" in bridge
    assert "ProductExceptLast(exact_topk_indices)" in bridge
    assert "aclnnA5ExactReconcilePrivateView" in bridge
    assert "exact_reconcile_private_view" not in policy
    assert "A5SparseAndTailAttention" not in kernel
    assert "GetKvaf" not in kernel
    assert "NANOVLLM_A5_OPS_ONLY=A5ExactReconcilePrivateView" in build_script
    assert '"${ROOT}/a5_exact_reconcile_private_view/*"' in build_script
    all_names = build_script.split("ALL_OP_NAMES=(", 1)[1].split(")", 1)[0]
    assert "A5ExactReconcilePrivateView" not in all_names
