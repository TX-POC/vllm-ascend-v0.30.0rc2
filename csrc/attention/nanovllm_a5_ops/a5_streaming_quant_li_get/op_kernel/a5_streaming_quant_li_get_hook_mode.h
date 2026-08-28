// Build-time controls for the isolated streaming-LI experiment.
//
// 0: official C8 score workspace only.  This is the bitwise control.
// 1: score workspace is caller-owned/graph-visible, but no descriptor is
//    published.  It isolates a changed score-GM address from publication.
// 2: after the existing score DMA, publish a fixed-slot score-tile receipt.
//    The receipt contains no KVAF, SIMT, queue, lease, or Get state by
//    itself.
//
// The inline mode is independent of publication mode:
// 0: score producer only (the strict mathematical/publisher baseline).
// 1: at the exact post-reduction, pre-score-DMA seam, run the audit-only
//    threshold/mapping predicate.  It requires an audit lease sentinel and
//    is compiled so that it cannot submit Direct-HCCP.
// 2: use the same seam to submit bounded native-enqueue Direct-HCCP waves.
//    This remains disabled until the audit mode has passed.
//
// Both controls are compile-time on purpose.  A runtime flag would still
// compile the extra vector/queue state into the strict C8 baseline and would
// make it impossible to attribute a Top-K change to a particular closure.
#ifndef A5_STREAMING_LI_GET_PUBLISH_MODE
#define A5_STREAMING_LI_GET_PUBLISH_MODE 0
#endif

static_assert(A5_STREAMING_LI_GET_PUBLISH_MODE >= 0 &&
              A5_STREAMING_LI_GET_PUBLISH_MODE <= 2,
              "A5 streaming LI publication mode must be 0, 1, or 2.");

#ifndef A5_STREAMING_LI_GET_INLINE_MODE
#define A5_STREAMING_LI_GET_INLINE_MODE 0
#endif

static_assert(A5_STREAMING_LI_GET_INLINE_MODE >= 0 &&
              A5_STREAMING_LI_GET_INLINE_MODE <= 2,
              "A5 streaming LI inline mode must be 0, 1, or 2.");

// The two outputs serve different validation/product boundaries.  The
// publisher owns an externally visible score-workspace/receipt protocol;
// inline mode owns the transaction-local scan-time KVAF protocol.  Combining
// them in one binary would make a failed proof ambiguous (and can make two
// independent MTE3 publications contend for the same score-tile lifetime),
// so the isolated build contract deliberately keeps them mutually exclusive.
static_assert(A5_STREAMING_LI_GET_INLINE_MODE == 0 ||
              A5_STREAMING_LI_GET_PUBLISH_MODE == 0,
              "inline mode requires publication mode off.");
