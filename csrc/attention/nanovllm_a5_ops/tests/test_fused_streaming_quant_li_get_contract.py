#!/usr/bin/env python3
"""Static P4.1 contract: one raw-Q/K scan, audit candidate map, guarded post.

This test intentionally performs no target launch.  It guards the source
closure before a scoped OPP build: the clone is non-authoritative, audit has
no Direct-HCCP side effect, and candidates originate from score tiles rather
than an exported full score tensor or a post-TopK reconstruction.
"""

from __future__ import annotations

from pathlib import Path


OPS = Path(__file__).resolve().parents[1]
AUDIT_HOOK = OPS / "a5_streaming_quant_li_get/op_kernel/arch35/a5_streaming_quant_li_get_tile_audit.h"
POST_HOOK = OPS / "a5_streaming_quant_li_get/op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h"
VECTOR = OPS / "a5_streaming_quant_li_get/op_kernel/arch35/quant_lightning_indexer_service_vector.h"
BRIDGE = (
    OPS
    / "a5_fused_streaming_quant_li_get/torch_extension"
    / "nanovllm_p4_fused_streaming_li_get/bridge.py"
)
AUDIT_ADAPTER = OPS / "tests/fused_streaming_quant_li_get_audit_adapter.py"
RESIDENT_SEED_BRIDGE = (
    OPS
    / "a5_c8_resident_seed/torch_extension/csrc/a5_c8_resident_seed.cpp"
)
README = OPS / "a5_fused_streaming_quant_li_get/README.md"
BUILD = OPS / "build_nanovllm_ops.sh"


def main() -> int:
    audit = AUDIT_HOOK.read_text(encoding="utf-8")
    post = POST_HOOK.read_text(encoding="utf-8")
    vector = VECTOR.read_text(encoding="utf-8")
    bridge = BRIDGE.read_text(encoding="utf-8")
    audit_adapter = AUDIT_ADAPTER.read_text(encoding="utf-8")
    resident_seed_bridge = RESIDENT_SEED_BRIDGE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    build = BUILD.read_text(encoding="utf-8")

    assert "kAuditScoreReceiptQueueLeaseSentinel = -2147483647" in audit
    assert "kAuditScoreReceiptTag = 0x80000000U" in audit
    assert "A5_STREAMING_LI_GET_INLINE_MODE == 1" in audit
    assert "StreamingLiTileAuditHook" in audit
    assert "HasAuditReceiptLeaseTable" in audit
    assert "MayNeedScoreObservation" in audit
    assert "AuditScoreDecision Evaluate(" in audit
    assert "static_cast<int64_t>(receipt) - (1LL << 32)" in audit
    # Audit is its own compile-time score-only translation-unit closure, not
    # a runtime branch inside the Direct-HCCP hook.  No descriptor/queue/CQ
    # helper may be referenced by this header.
    for forbidden in (
        "TryAcquireKvafStreamingBatchGet",
        "BeginKvafStreamingBatchGet",
        "PublishKvafStreamingBatchGetDescriptorScratch",
        "TryRetireKvafStreamingBatchGet",
        "FinishKvafStreamingBatchGet",
        "kvaf_fused_get::",
        "privateReady_",
        "privateArenaHead_",
        "AddTelemetry",
    ):
        assert forbidden not in audit, forbidden
    assert "kAuditScoreReceipt" not in post
    assert "AuditScoreDecision" not in post
    # The post header is included through the generated mode header. It must
    # depend on the derived full-closure guard, not repeat the raw inline-mode
    # expression (which is intentionally absent from this translation unit).
    assert "A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE" in post
    assert "TryAcquireKvafStreamingBatchGetQueueFromAicore" in post
    assert "BeginKvafStreamingBatchGet" in post

    reduction = vector.index("BatchMulWeightAndReduceSum")
    emit = vector.index("streamingAuditHook_.Emit")
    score_dma = vector.index("DataCopyPad(scoreGm")
    assert reduction < emit < score_dma
    assert "ProcessTopK" in vector
    assert "li_scores" not in bridge
    assert "non_authoritative_sparse_indices" in bridge
    assert "topk_authority\": \"non_authoritative_pending_strict_equivalence" in bridge
    assert "P4_FUSED_ALLOW_POST" in bridge
    assert "marker-only audit was retired" in bridge
    assert "P4.1 mode/build mismatch" in bridge
    assert "P3_INLINE_MODE" in bridge
    assert 'publication_mode in ("off", "publisher")' in bridge
    assert 'post_publication_ok = publication_mode == "off"' in bridge
    assert "A5StreamingQuantLiGet" in build
    assert "NANOVLLM_A5_OPS_ONLY=A5StreamingQuantLiGet" in bridge
    assert "_AUDIT_SCORE_RECEIPT = \"audit_score_receipt\"" in bridge
    assert "SCORE_RECEIPT_CANDIDATE_BIT" in bridge
    assert "_PUBLICATION_KEYS" in bridge
    assert "publication: Mapping[str, Any]" in bridge
    assert "len(outputs) != 11" in bridge
    assert '"score_workspace": outputs[9]' in bridge
    assert '"tile_receipts": outputs[10]' in bridge
    assert "official C8" in readme
    assert "non-authoritative" in readme
    assert "no ring\noverflow" in readme
    # The threshold must be a real C8 sortable boundary generated from the
    # separate resident index view, rather than the unsupported float
    # sparse_values output or a host-derived proxy.
    assert "NANOVLLM_A5_RESIDENT_SEED_OPP" in audit_adapter
    assert "resident_seed_exact_sortable_uint16" in audit_adapter
    assert "_collect_authoritative_cold_coverage" in audit_adapter
    assert "object_mapping=\"permuted_pool\"" in audit_adapter
    assert "row -> req_pool_entries -> kv_object_map" in audit_adapter
    assert "_assert_transaction_output_aliases" in audit_adapter
    assert "_assert_publication_output_aliases" in audit_adapter
    assert "_make_publication_buffers" in audit_adapter
    assert "INLINE_MODE_ENV" in audit_adapter
    assert "PUBLICATION_MODE_ENV" in audit_adapter
    assert 'publication_mode != "publisher" or inline_mode != "audit"' in audit_adapter
    assert "_decode_published_score_mirror" in audit_adapter
    assert "P4_SCORE_GM_MIRROR" in audit_adapter
    assert "P4_SCORE_PROVENANCE" in audit_adapter
    assert "score_mirror_mismatch_count" in audit_adapter
    assert "_assert_clone_source_domain" in audit_adapter
    assert "kv_object_domain=1" in audit_adapter
    assert "mapping_discriminator_selected" in audit_adapter
    assert "candidate_discriminator_count" in audit_adapter
    assert "invalid_candidate_mask" in audit_adapter
    assert "P4_FUSED_AUDIT_COVERAGE_DIAG" in audit_adapter
    assert "P4_FUSED_AUDIT_SCORE_RECEIPT" in audit_adapter
    assert "mode=\"audit_score_receipt\"" in audit_adapter
    assert "_read_target_provenance" in audit_adapter
    assert "official_selected_mask" in audit_adapter
    assert "threshold safety" in audit_adapter
    assert "npu_a5_c8_resident_seed" in resident_seed_bridge
    assert "aclnnA5C8ResidentSeed" in resident_seed_bridge
    assert "NANOVLLM_A5_RESIDENT_SEED_OPAPI_LIB" in resident_seed_bridge
    assert "AllowStridedAclTensor(resident_index_key)" in resident_seed_bridge
    assert "A5C8ResidentSeed does not expose sparse_values" in resident_seed_bridge
    assert "cmp_ratio=1" in resident_seed_bridge
    strict_math_test = (
        OPS / "tests/test_fused_streaming_quant_li_get_strict_math_equivalence.py"
    ).read_text(encoding="utf-8")
    assert "P4_STRICT_MATH_GATE_PASS" in strict_math_test
    assert "P4_STRICT_MATH_GATE_FAIL" in strict_math_test
    assert "P4_SCORE_PROVENANCE_GATE" in strict_math_test
    assert "P4_RESIDENT_THRESHOLD_DOMAIN_GATE" in strict_math_test
    assert "P4_THRESHOLD_SAFETY_GATE" in strict_math_test
    assert "_assert_bitwise_equal" in strict_math_test
    target_adapter = (OPS / "tests/streaming_quant_li_get_target_adapter.py").read_text(
        encoding="utf-8"
    )
    assert 'TARGET_PROVENANCE_NAME = "P3_STREAMING_LI_PROVENANCE"' in target_adapter
    assert 'TARGET_EXPECTED_OPP_SOURCE_HEAD_ENV = "P3_TARGET_EXPECTED_OPP_SOURCE_HEAD"' in target_adapter
    assert 'RUNNER_EXPECTED_HEAD_ENV = "P3_RUNNER_EXPECTED_HEAD"' in target_adapter
    artifact_env = (OPS / "tests/resolve_streaming_li_get_artifacts.sh").read_text(encoding="utf-8")
    assert "P3_ARTIFACT_ENV" in artifact_env
    assert "P3_TARGET_EXPECTED_OPP_SOURCE_HEAD" in artifact_env
    assert "P3_RUNNER_EXPECTED_HEAD" in artifact_env
    assert '"source_clean"' in target_adapter
    assert "_read_target_provenance" in target_adapter
    assert 'INLINE_MODE_ENV = "P3_INLINE_MODE"' in target_adapter
    assert 'INLINE_MODES = ("off", "audit", "post")' in target_adapter
    assert '"inline_mode"' in target_adapter
    assert "this strict P3 publication adapter requires P3_INLINE_MODE=off" in target_adapter
    assert "P3 target provenance" in build
    assert "NANOVLLM_A5_STREAMING_LI_GET_INLINE_MODE" in build
    assert "NEEDS_P3_POST_KVAF=0" in build
    assert "P3 generated audit closure: score-only verified=1" in build
    assert "kvaf_closure=${P3_KVAF_CLOSURE}" in build
    assert "kvaf_source_head=${P3_KVAF_SOURCE_HEAD}" in build
    assert "P3 inline=audit closure must contain only the audit observer" in build

    print(
        "P4_FUSED_SINGLE_SCAN_CONTRACT_PASS "
        "raw_qk_score_tile=1 final_clone_topk=1 candidate_map=fixed_RxS "
        "audit_no_post=1 post_runtime_guard=1 "
        "official_c8_authority=required_until_bitwise_equivalence "
        "resident_threshold=source_derived_sortable_c8",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
