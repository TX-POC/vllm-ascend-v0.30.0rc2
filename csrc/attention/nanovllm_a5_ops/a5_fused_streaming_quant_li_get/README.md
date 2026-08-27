# P4 resident-threshold fused raw-Q/K scan bridge

`A5StreamingQuantLiGet` remains an isolated, source-derived C8 clone. This
directory does **not** replace or register the official C8 operator. It adds
the prefetch sidecar around the one full raw-Q/K scan:

```text
resident-index C8 scan -> exact resident Top-2048 sortable threshold

full raw Q/K tile -> C8 score tile -> threshold + resident-miss decision
                                      -> fixed [R,S] candidate map
                                      -> audit: stop
                                      -> post: existing Direct-HCCP queue path
                                -> clone Top-K (diagnostic only)
```

The first scan is `A5C8ResidentSeed`. It reads the C8 FP8 index-key view, not
QSFA's packed KV payload, and produces the low-16-bit sortable value of the
resident row's 2048th score. The full scan compares every score in that same
sortable domain. Therefore a cold KV is marked as soon as the current score
crosses the resident threshold; scanning then continues without waiting for
final Top-K.

The clone Top-K is explicitly non-authoritative. QSFA continues consuming the
source-matched official C8 output. A clone/official bitwise mismatch blocks a
future clone-Top-K route; it does not change model mathematics during this
candidate audit.

## Runtime modes

- `audit`: caller initializes every `queue_leases` value to `INT32_MIN`.
  The device hook records `-3` (`AUDIT_CANDIDATE_MARKER`) in the fixed
  `private_source_to_slot[R,S]` map for a threshold-qualified, nonresident
  score source.  It performs no queue claim, descriptor write, doorbell, CQ
  poll, private-payload write, or cache mapping publication.
- `post`: caller supplies the normal valid queue lease table.  The preexisting
  Direct-HCCP path is available only when `P4_FUSED_ALLOW_POST=1`; it remains
  experimental while payload validation is known failed.

The full `[R,S]` candidate map is intentional for audit: it has no ring
overflow. Its exact count is `count(marker == -3)` after the target stream
completes. It is a validation record, not a persistent HBM mapping.

## What the audit proves

The validation fixture creates a separate resident FP8 index-key view with 24
logical pages (3072 tokens). The full source still has 4096/3968 tokens, so a
real cold suffix remains. It checks the exact lossless-prefetch condition:

```text
official C8 final Top-K ∩ cold sources ⊆ score-time candidate map
```

`score >= threshold` intentionally permits tie-driven false positives. Missing
even one official cold source fails the audit. The test also verifies that the
ResidentSeed selected indices are bitwise equal to official C8 on the same
resident index view. It does not claim H2D bandwidth or payload correctness.

## Scoped build / test closure

Build the isolated `A5C8ResidentSeed` OPP and its typed bridge. Build the
existing `A5StreamingQuantLiGet` OPP/bridge only if its matching cached build
is unavailable. Put the isolated reference, resident-seed, P3, and P4 bridge
directories on `PYTHONPATH`, then run
`tests/test_fused_streaming_quant_li_get_audit.py`. No vLLM-Ascend package
build is involved.
