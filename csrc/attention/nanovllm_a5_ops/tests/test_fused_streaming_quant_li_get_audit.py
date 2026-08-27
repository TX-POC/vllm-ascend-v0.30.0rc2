#!/usr/bin/env python3
"""P4 resident-threshold candidate gate and separate final-Top-K route gate.

The candidate gate answers one narrow but essential question: did the scan-time
resident-threshold predicate cover every cold source that authoritative C8
ultimately selected?  The clone's own Top-K remains a separate route gate.  A
clone mismatch must block using it for QSFA, but it must not hide the result of
the candidate-safety audit, because official C8 remains the active final-Top-K
producer throughout this bring-up.
"""

from __future__ import annotations

import hashlib

from fused_streaming_quant_li_get_audit_adapter import run_p4_fused_audit


def _bytes(value: object) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return tensor.numpy().tobytes()


def main() -> int:
    result = run_p4_fused_audit()
    official = _bytes(result["official_indices"])
    clone = _bytes(result["clone_indices"])
    coverage_ok = (
        int(result["required_cold"]) > 0
        and int(result["covered_cold"]) == int(result["required_cold"])
        and int(result["missing_cold"]) == 0
        and int(result["invalid_candidate_count"]) == 0
        and int(result["outside_domain_count"]) == 0
        and str(result["receipt_kind"]) == "score"
        and int(result["receipt_unwritten_count"]) == 0
        and int(result["receipt_malformed_count"]) == 0
        and int(result["threshold_bit_mismatch_count"]) == 0
        and int(result["mapping_bit_mismatch_count"]) == 0
        and int(result["cold_bit_mismatch_count"]) == 0
        and int(result["source_unmapped_bit_mismatch_count"]) == 0
        and int(result["candidate_bit_mismatch_count"]) == 0
        and bool(result["candidate_selective"])
        and bool(result["mapping_fixture_discriminates"])
    )
    if not coverage_ok:
        raise RuntimeError(
            "P4_FUSED_AUDIT_CANDIDATE_GATE status=blocked "
            f"required_cold={result['required_cold']} "
            f"covered_cold={result['covered_cold']} "
            f"missing_cold={result['missing_cold']} "
            f"invalid_candidate_count={result['invalid_candidate_count']} "
            f"outside_domain_count={result['outside_domain_count']} "
            f"receipt_kind={result['receipt_kind']} "
            f"receipt_unwritten_count={result['receipt_unwritten_count']} "
            f"receipt_malformed_count={result['receipt_malformed_count']} "
            f"threshold_bit_mismatch_count={result['threshold_bit_mismatch_count']} "
            f"mapping_bit_mismatch_count={result['mapping_bit_mismatch_count']} "
            f"cold_bit_mismatch_count={result['cold_bit_mismatch_count']} "
            f"source_unmapped_bit_mismatch_count={result['source_unmapped_bit_mismatch_count']} "
            f"candidate_bit_mismatch_count={result['candidate_bit_mismatch_count']} "
            f"candidate_selective={int(bool(result['candidate_selective']))} "
            f"mapping_fixture_discriminates={int(bool(result['mapping_fixture_discriminates']))}"
        )
    print(
        "P4_FUSED_AUDIT_CANDIDATE_GATE status=pass "
        "authority=official_c8 "
        f"required_cold={result['required_cold']} "
        f"covered_cold={result['covered_cold']} "
        f"missing_cold={result['missing_cold']} "
        f"candidate_count={result['candidate_count']} "
        "h2d_result=not_exercised",
        flush=True,
    )
    if official == clone:
        print(
            "P4_FUSED_AUDIT_TOPK_ROUTE_GATE status=pass "
            "official_c8_vs_clone=bitwise_equal "
            "clone_topk_route=eligible_after_independent_h2d_payload_proof",
            flush=True,
        )
        return 0
    first = next(
        index
        for index, (left, right) in enumerate(zip(official, clone, strict=True))
        if left != right
    )
    print(
        "P4_FUSED_AUDIT_TOPK_ROUTE_GATE status=blocked "
        "official_c8_vs_clone=bitwise_mismatch "
        f"total_bytes={len(official)} first_byte={first} "
        f"official_sha256={hashlib.sha256(official).hexdigest()[:16]} "
        f"clone_sha256={hashlib.sha256(clone).hexdigest()[:16]} "
        "action=official_c8_remains_qsfa_authority "
        "h2d_result=not_exercised",
        # This is deliberately stdout: an audited candidate pass plus a route
        # block is an expected, copyable validation outcome rather than a
        # hidden test failure.
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
