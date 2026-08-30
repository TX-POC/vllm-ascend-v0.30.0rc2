#!/usr/bin/env python3
"""Strict P7.2a gate for the clean C8 score-window publisher.

P7.2a deliberately does *not* run a speculative executor.  It proves the
smallest safe C8 change first: the normal C8 score DMA writes a caller-owned
score workspace and then emits its fixed 16-word receipt.  The compiled
target must therefore be exactly:

* ``publication=publisher`` (C8 score workspace + receipt);
* ``inline=off`` (no audit, no KVAF/Post hook); and
* no post-closure or scheduler selector.

The target adapter launches the source-matched official C8 reference and the
derived clean publisher on the same physical input tensors.  It additionally
validates the receipt's immutable descriptor/epoch ABI after NPU completion.
No H2D, queue lease, private slot, CQ, P7.0, or P7.1 code is imported here.
"""

from __future__ import annotations

import os
import traceback

from streaming_quant_li_get_target_adapter import run_target_equivalence
from test_streaming_quant_li_get_target_equivalence import (
    _assert_actual_c8_output_domain,
    _assert_bitwise_equal,
    _assert_official_input_aliases,
    _assert_publication_contract,
    _assert_source_index_domain,
    _assert_transaction_aliases,
)


def _require_clean_publisher_environment() -> None:
    """Reject a stale P6/Post shell before any NPU work begins."""

    required = {
        "P3_PUBLICATION_MODE": "publisher",
        "P3_INLINE_MODE": "off",
    }
    for key, expected in required.items():
        actual = os.getenv(key)
        if actual != expected:
            raise RuntimeError(
                f"P7.2a requires {key}={expected!r}; got {actual!r}"
            )
    for key in ("P3_POST_CLOSURE_MODE", "P3_POST_SCHEDULER_MODE"):
        actual = os.getenv(key)
        if actual not in (None, "", "not_applicable"):
            raise RuntimeError(
                f"P7.2a is clean publisher only; {key} must be unset or "
                f"'not_applicable', got {actual!r}"
            )


def main() -> int:
    try:
        _require_clean_publisher_environment()
        result = run_target_equivalence()

        # Do not rely solely on the adapter's own diagnostics: these are the
        # same fail-closed byte, input-alias, output-domain, and publication
        # checks used by the generic target-equivalence harness.
        _assert_official_input_aliases(result)
        _assert_transaction_aliases(result)
        publication = _assert_publication_contract(result)
        _assert_bitwise_equal(
            result["official_indices"], result["derived_indices"], "indices"
        )
        _assert_actual_c8_output_domain(result)
        _assert_source_index_domain(result)

        if publication["mode"] != "publisher":
            raise AssertionError(f"unexpected publication mode: {publication}")
        if publication["receipt_ready_count"] != publication["receipt_expected_active"]:
            raise AssertionError(
                "P7.2a requires every active score tile to publish exactly one "
                f"receipt: {publication}"
            )
    except Exception as error:
        print(
            "P7_CLEAN_PUBLISHER_FAIL "
            f"type={type(error).__name__} text={error}",
            flush=True,
        )
        traceback.print_exc()
        return 1

    print(
        "P7_CLEAN_PUBLISHER_CONFIG "
        "publication=publisher inline=off post_closure=not_applicable "
        "post_scheduler=not_applicable kvaf_closure=absent executor=not_loaded",
        flush=True,
    )
    print(
        "P7_CLEAN_PUBLISHER_RESULT "
        "topk_bitwise=1 score_source_lens_bitwise=1 "
        f"receipt_ready={publication['receipt_ready_count']} "
        f"receipt_expected_active={publication['receipt_expected_active']}",
        flush=True,
    )
    print(
        "P7_CLEAN_PUBLISHER_PASS "
        "official_c8=bitwise score_window_receipt=layout_epoch_verified "
        "kvaf_post_cq=not_compiled",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
