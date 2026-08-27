#!/usr/bin/env python3
"""Strict gate for P4.1 audit: diagnostic receipt first, equivalence second."""

from __future__ import annotations

import hashlib
import sys

from fused_streaming_quant_li_get_audit_adapter import run_p4_fused_audit


def _bytes(value: object) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return tensor.numpy().tobytes()


def main() -> int:
    result = run_p4_fused_audit()
    official = _bytes(result["official_indices"])
    clone = _bytes(result["clone_indices"])
    if official == clone:
        print(
            "P4_FUSED_AUDIT_STRICT_GATE status=pass "
            "official_c8_vs_clone=bitwise_equal "
            f"candidate_count={result['candidate_count']} "
            "h2d_result=not_exercised",
            flush=True,
        )
        return 0
    first = next(
        index
        for index, (left, right) in enumerate(zip(official, clone, strict=True))
        if left != right
    )
    print(
        "P4_FUSED_AUDIT_STRICT_GATE status=blocked "
        "official_c8_vs_clone=bitwise_mismatch "
        f"total_bytes={len(official)} first_byte={first} "
        f"official_sha256={hashlib.sha256(official).hexdigest()[:16]} "
        f"clone_sha256={hashlib.sha256(clone).hexdigest()[:16]} "
        f"candidate_count={result['candidate_count']} "
        "action=do_not_route_clone_topk_to_qsfa h2d_result=not_exercised",
        file=sys.stderr,
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
