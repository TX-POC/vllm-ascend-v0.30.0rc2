# P4.1 fused raw-Q/K scan bridge

`A5StreamingQuantLiGet` remains an isolated, source-derived C8 clone.  This
directory does **not** replace or register the official C8 operator.  It adds
the product-shaped P4.1 bridge around that one raw-Q/K scan:

```text
raw Q/K tile -> C8 score tile -> threshold + resident-miss decision
                               -> fixed [R,S] candidate map
                               -> audit: stop
                               -> post: existing Direct-HCCP queue path
                         -> same clone's exact C8 Top-K output
```

The clone Top-K is explicitly non-authoritative until a strict target test
proves it bitwise equal to the source-matched official C8 result.  QSFA must
continue consuming the official C8 output in every mode before that proof.

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
overflow.  Its exact audit count is `count(marker == -3)` after the target
stream completes; audit deliberately does not write the normal queue-owned
telemetry rows from multiple AIV producers.  It is a validation record, not
a persistent HBM mapping.

## Scoped build / test closure

P4.1 adds only Python bridge/test code.  Build the existing isolated backing
OPP with `NANOVLLM_A5_OPS_ONLY=A5StreamingQuantLiGet`, then build its existing
typed bridge.  Put both its `torch_extension` directory and this directory's
`torch_extension` directory on `PYTHONPATH`, and run
`tests/test_fused_streaming_quant_li_get_audit.py`.  A Top-K mismatch is an
intentional nonzero strict-gate result, not a pass or a performance result.
