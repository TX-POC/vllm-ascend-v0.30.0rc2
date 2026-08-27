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
HOOK = OPS / "a5_streaming_quant_li_get/op_kernel/arch35/a5_streaming_quant_li_get_tile_ready.h"
VECTOR = OPS / "a5_streaming_quant_li_get/op_kernel/arch35/quant_lightning_indexer_service_vector.h"
BRIDGE = (
    OPS
    / "a5_fused_streaming_quant_li_get/torch_extension"
    / "nanovllm_p4_fused_streaming_li_get/bridge.py"
)
README = OPS / "a5_fused_streaming_quant_li_get/README.md"
BUILD = OPS / "build_nanovllm_ops.sh"


def _body(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[begin:finish]


def main() -> int:
    hook = HOOK.read_text(encoding="utf-8")
    vector = VECTOR.read_text(encoding="utf-8")
    bridge = BRIDGE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    build = BUILD.read_text(encoding="utf-8")

    assert "kAuditQueueLeaseSentinel = (-2147483647 - 1)" in hook
    assert "kSourceAuditCandidate = -3" in hook
    assert "auditMode_ = IsAuditLeaseTable();" in hook
    assert "if (auditMode_) {\n            return true;" in hook
    assert "EmitAuditCandidates(out_base" in hook

    audit = _body(hook, "void EmitAuditCandidates(", "void TryStageCandidate(")
    assert "TryRecordAuditCandidate(" in audit
    assert "sourceToSlot_.SetValue(sourceOffset, kSourceAuditCandidate)" in hook
    # Audit must stop before every transport/arena method.  It may read the
    # normal control ABI for shape validation, but it cannot claim/post/poll.
    assert "TryAcquireKvafStreamingBatchGetQueueFromAicore" not in audit
    assert "BeginKvafStreamingBatchGet" not in audit
    assert "PublishKvafStreamingBatchGetDescriptorScratch" not in audit
    assert "TryRetireKvafStreamingBatchGet" not in audit
    assert "FinishKvafStreamingBatchGet" not in audit
    assert "privateReady_.SetValue" not in audit
    assert "privateArenaHead_.SetValue" not in audit
    assert "AddTelemetry" not in audit

    reduction = vector.index("BatchMulWeightAndReduceSum")
    emit = vector.index("streamingKvafHook_.Emit")
    score_dma = vector.index("DataCopyPad(scoreGm")
    assert reduction < emit < score_dma
    assert "ProcessTopK" in vector
    assert "li_scores" not in bridge
    assert "non_authoritative_sparse_indices" in bridge
    assert "topk_authority\": \"non_authoritative_pending_strict_equivalence" in bridge
    assert "P4_FUSED_ALLOW_POST" in bridge
    assert "AUDIT_QUEUE_LEASE_SENTINEL" in bridge
    assert "A5StreamingQuantLiGet" in build
    assert "NANOVLLM_A5_OPS_ONLY=A5StreamingQuantLiGet" in bridge
    assert "official C8" in readme
    assert "non-authoritative" in readme
    assert "no ring\noverflow" in readme

    print(
        "P4_FUSED_SINGLE_SCAN_CONTRACT_PASS "
        "raw_qk_score_tile=1 final_clone_topk=1 candidate_map=fixed_RxS "
        "audit_no_post=1 post_runtime_guard=1 "
        "official_c8_authority=required_until_bitwise_equivalence",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
