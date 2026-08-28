#!/usr/bin/env python3
"""Run the strict mathematical control for P3 ``inline=post`` empty shell.

The target adapter owns the real A5 launches and provenance check.  This
small entry point intentionally owns only the two mathematical gates needed
for the closure ablation: official C8 sparse indices and the decode source-S
lengths returned through output #8.  It does *not* import the post-payload
test, because an empty shell deliberately stages no HCCP/KVAF closure.
"""

from __future__ import annotations

import os
import traceback

from streaming_quant_li_get_target_adapter import run_target_equivalence
from test_streaming_quant_li_get_target_equivalence import _assert_bitwise_equal


def _require_empty_shell_environment() -> None:
    if os.getenv("P3_INLINE_MODE") != "post":
        raise RuntimeError("P3_EMPTY_SHELL requires P3_INLINE_MODE=post")
    if os.getenv("P3_POST_CLOSURE_MODE") != "empty_shell":
        raise RuntimeError(
            "P3_EMPTY_SHELL requires P3_POST_CLOSURE_MODE=empty_shell"
        )
    if os.getenv("P3_PUBLICATION_MODE") != "off":
        raise RuntimeError("P3_EMPTY_SHELL requires P3_PUBLICATION_MODE=off")


def main() -> int:
    try:
        _require_empty_shell_environment()
        result = run_target_equivalence()
        _assert_bitwise_equal(
            result["official_indices"], result["derived_indices"], "indices"
        )
        _assert_bitwise_equal(
            result["expected_score_source_lens"],
            result["derived_score_source_lens"],
            "score_source_lens",
        )
    except Exception as error:  # Preserve the original traceback for remote triage.
        print(
            "P3_EMPTY_SHELL_STRICT_FAIL "
            f"type={type(error).__name__} text={error}",
            flush=True,
        )
        traceback.print_exc()
        return 1
    print(
        "P3_EMPTY_SHELL_STRICT_PASS "
        "topk=bitwise score_source_lens=bitwise "
        "inline=post post_closure=empty_shell hccp=not_compiled",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
