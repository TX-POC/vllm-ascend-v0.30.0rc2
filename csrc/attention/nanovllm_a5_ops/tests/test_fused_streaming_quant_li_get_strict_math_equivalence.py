#!/usr/bin/env python3
"""One-process, fail-closed math gate for the P4 prefetch predicate.

This wrapper deliberately shares the normal audit adapter, so it launches one
full raw-Q/K scan on one FP8 fixture.  It keeps two facts separate:

* the final C8 score/Top-K path and the audit receipt must agree exactly; and
* the resident threshold must be safe enough to cover final cold Top-K items.

The second condition remains a hard pre-H2D gate in this bring-up.  It is not
silently waived merely because a later GuaranteedRecall stage could recover a
miss in the product design.
"""

from __future__ import annotations

import hashlib
import sys

from fused_streaming_quant_li_get_audit_adapter import run_p4_fused_audit


def _canonical(value: object) -> tuple[tuple[int, ...], str, bytes]:
    tensor = value.detach().cpu().contiguous()
    return tuple(tensor.shape), str(tensor.dtype), tensor.numpy().tobytes()


def _assert_bitwise_equal(official_value: object, clone_value: object, label: str) -> None:
    official_shape, official_dtype, official = _canonical(official_value)
    clone_shape, clone_dtype, clone = _canonical(clone_value)
    if official_shape != clone_shape or official_dtype != clone_dtype:
        raise AssertionError(
            "P4_STRICT_MATH_GATE_FAIL "
            f"check={label}_abi official_shape={official_shape} clone_shape={clone_shape} "
            f"official_dtype={official_dtype} clone_dtype={clone_dtype}"
        )
    if official == clone:
        return
    first = next(
        index
        for index, (left, right) in enumerate(zip(official, clone, strict=True))
        if left != right
    )
    raise AssertionError(
        "P4_STRICT_MATH_GATE_FAIL "
        f"check={label} total_bytes={len(official)} first_byte={first} "
        f"official_sha256={hashlib.sha256(official).hexdigest()[:16]} "
        f"clone_sha256={hashlib.sha256(clone).hexdigest()[:16]}"
    )


def main() -> int:
    result = run_p4_fused_audit()
    _assert_bitwise_equal(
        result["official_indices"], result["clone_indices"], "full_topk"
    )
    print(
        "P4_STRICT_MATH_GATE_PROGRESS check=full_topk status=bitwise_equal",
        flush=True,
    )
    _assert_bitwise_equal(
        result["expected_score_source_lens"],
        result["derived_score_source_lens"],
        "score_source_lens",
    )
    print(
        "P4_STRICT_MATH_GATE_PROGRESS check=score_source_lens status=bitwise_equal",
        flush=True,
    )
    receipt_kind = str(result["receipt_kind"])
    receipt_unwritten_count = int(result["receipt_unwritten_count"])
    receipt_malformed_count = int(result["receipt_malformed_count"])
    threshold_bit_mismatch_count = int(result["threshold_bit_mismatch_count"])
    mapping_bit_mismatch_count = int(result["mapping_bit_mismatch_count"])
    cold_bit_mismatch_count = int(result["cold_bit_mismatch_count"])
    source_unmapped_bit_mismatch_count = int(
        result["source_unmapped_bit_mismatch_count"]
    )
    candidate_bit_mismatch_count = int(result["candidate_bit_mismatch_count"])
    if (
        receipt_kind != "score"
        or receipt_unwritten_count != 0
        or receipt_malformed_count != 0
        or threshold_bit_mismatch_count != 0
        or mapping_bit_mismatch_count != 0
        or cold_bit_mismatch_count != 0
        or source_unmapped_bit_mismatch_count != 0
        or candidate_bit_mismatch_count != 0
    ):
        raise AssertionError(
            "P4_STRICT_MATH_GATE_FAIL "
            "check=score_receipt "
            f"kind={receipt_kind} unwritten={receipt_unwritten_count} "
            f"malformed={receipt_malformed_count} "
            f"threshold_bit_mismatch={threshold_bit_mismatch_count} "
            f"mapping_bit_mismatch={mapping_bit_mismatch_count} "
            f"cold_bit_mismatch={cold_bit_mismatch_count} "
            f"source_unmapped_bit_mismatch={source_unmapped_bit_mismatch_count} "
            f"candidate_bit_mismatch={candidate_bit_mismatch_count}"
        )
    print(
        "P4_STRICT_MATH_GATE_PROGRESS check=score_receipt status=internal_consistent",
        flush=True,
    )
    score_mirror_missing_count = int(result["score_mirror_missing_count"])
    score_mirror_outside_count = int(result["score_mirror_outside_count"])
    score_mirror_duplicate_count = int(result["score_mirror_duplicate_count"])
    score_mirror_interval_overlap_count = int(
        result["score_mirror_interval_overlap_count"]
    )
    score_mirror_active_records = int(result["score_mirror_active_records"])
    score_mirror_expected_active_records = int(
        result["score_mirror_expected_active_records"]
    )
    score_mirror_ready_records = int(result["score_mirror_ready_records"])
    score_mirror_zero_valid_records = int(result["score_mirror_zero_valid_records"])
    score_mirror_descriptor_errors = tuple(result["score_mirror_descriptor_errors"])
    score_mirror_mismatch_count = int(result["score_mirror_mismatch_count"])
    candidate_from_mirror_mismatch_count = int(
        result["candidate_from_mirror_mismatch_count"]
    )
    score_provenance_ok = (
        score_mirror_missing_count == 0
        and score_mirror_outside_count == 0
        and score_mirror_duplicate_count == 0
        and score_mirror_interval_overlap_count == 0
        and score_mirror_active_records == score_mirror_expected_active_records
        and not score_mirror_descriptor_errors
        and score_mirror_mismatch_count == 0
        and candidate_from_mirror_mismatch_count == 0
    )
    print(
        "P4_SCORE_PROVENANCE_GATE "
        f"status={'pass' if score_provenance_ok else 'fail'} "
        f"mirror_missing={score_mirror_missing_count} "
        f"mirror_outside={score_mirror_outside_count} "
        f"mirror_duplicates={score_mirror_duplicate_count} "
        f"mirror_interval_overlaps={score_mirror_interval_overlap_count} "
        f"mirror_active_records={score_mirror_active_records} "
        f"mirror_expected_active_records={score_mirror_expected_active_records} "
        f"mirror_ready_records={score_mirror_ready_records} "
        f"mirror_zero_valid_records={score_mirror_zero_valid_records} "
        f"mirror_descriptor_errors={len(score_mirror_descriptor_errors)} "
        f"receipt_score_mismatch={score_mirror_mismatch_count} "
        f"candidate_from_mirror_mismatch={candidate_from_mirror_mismatch_count}",
        flush=True,
    )
    selected_below_tau_count = int(result["selected_below_tau_count"])
    resident_tau_invalid_count = int(result["resident_tau_invalid_count"])
    resident_tau_rank_invariant_count = int(
        result["resident_tau_rank_invariant_count"]
    )
    threshold_domain_ok = (
        selected_below_tau_count == 0
        and resident_tau_invalid_count == 0
        and resident_tau_rank_invariant_count == 0
    )
    print(
        "P4_RESIDENT_THRESHOLD_DOMAIN_GATE "
        f"status={'pass' if threshold_domain_ok else 'fail'} "
        f"selected_below_tau={selected_below_tau_count} "
        f"resident_tau_invalid={resident_tau_invalid_count} "
        f"resident_tau_rank_invariant_failures={resident_tau_rank_invariant_count}",
        flush=True,
    )
    required_cold = int(result["required_cold"])
    covered_cold = int(result["covered_cold"])
    missing_cold = int(result["missing_cold"])
    candidate_count = int(result["candidate_count"])
    invalid_candidate_count = int(result["invalid_candidate_count"])
    outside_domain_count = int(result["outside_domain_count"])
    mapping_discriminator_selected = int(result["mapping_discriminator_selected"])
    candidate_discriminator_count = int(result["candidate_discriminator_count"])
    threshold_safety_ok = (
        required_cold > 0
        and covered_cold == required_cold
        and missing_cold == 0
        and candidate_count >= required_cold
        and invalid_candidate_count == 0
        and outside_domain_count == 0
        and mapping_discriminator_selected > 0
        and candidate_discriminator_count > 0
    )
    if not threshold_safety_ok:
        print(
            "P4_THRESHOLD_SAFETY_GATE_FAIL "
            f"required_cold={required_cold} covered_cold={covered_cold} "
            f"missing_cold={missing_cold} candidate_count={candidate_count} "
            f"invalid_candidate_count={invalid_candidate_count} "
            f"outside_domain_count={outside_domain_count} "
            f"mapping_discriminator_selected={mapping_discriminator_selected} "
            f"candidate_discriminator_count={candidate_discriminator_count}",
            flush=True,
        )
    else:
        print(
            "P4_THRESHOLD_SAFETY_GATE_PASS "
            "official_topk_cold_subset_of_speculative_candidates=1",
            flush=True,
        )
    if int(result["mapping_discriminator_sources"]) <= 0:
        raise AssertionError(
            "P4_STRICT_MATH_GATE_FAIL "
            "check=mapping_fixture_discriminator"
        )
    if not score_provenance_ok:
        raise AssertionError(
            "P4_STRICT_MATH_GATE_FAIL "
            "check=score_provenance "
            f"mirror_missing={score_mirror_missing_count} "
            f"mirror_outside={score_mirror_outside_count} "
            f"mirror_duplicates={score_mirror_duplicate_count} "
            f"mirror_interval_overlaps={score_mirror_interval_overlap_count} "
            f"mirror_active_records={score_mirror_active_records} "
            f"mirror_expected_active_records={score_mirror_expected_active_records} "
            f"mirror_ready_records={score_mirror_ready_records} "
            f"mirror_zero_valid_records={score_mirror_zero_valid_records} "
            f"mirror_descriptor_errors={len(score_mirror_descriptor_errors)} "
            f"receipt_score_mismatch={score_mirror_mismatch_count} "
            f"candidate_from_mirror_mismatch={candidate_from_mirror_mismatch_count}"
        )
    if not threshold_domain_ok:
        raise AssertionError(
            "P4_STRICT_MATH_GATE_FAIL "
            "check=resident_threshold_domain "
            f"selected_below_tau={selected_below_tau_count} "
            f"resident_tau_invalid={resident_tau_invalid_count} "
            f"resident_tau_rank_invariant_failures={resident_tau_rank_invariant_count}"
        )
    if not threshold_safety_ok:
        raise AssertionError(
            "P4_STRICT_MATH_GATE_FAIL "
            "check=threshold_safety "
            f"required_cold={required_cold} covered_cold={covered_cold} "
            f"missing_cold={missing_cold} candidate_count={candidate_count} "
            f"invalid_candidate_count={invalid_candidate_count} "
            f"outside_domain_count={outside_domain_count} "
            f"mapping_discriminator_selected={mapping_discriminator_selected} "
            f"candidate_discriminator_count={candidate_discriminator_count}"
        )
    print(
        "P4_STRICT_MATH_GATE_PASS "
        "resident_indices=bitwise "
        "clone_indices=bitwise "
        "score_source_lens=bitwise "
        "cold_coverage=complete "
        "mapping=permuted_compressed "
        "aliases=verified "
        "h2d_result=not_exercised",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            "P4_STRICT_MATH_GATE_FAIL "
            f"type={type(error).__name__} text={error}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
