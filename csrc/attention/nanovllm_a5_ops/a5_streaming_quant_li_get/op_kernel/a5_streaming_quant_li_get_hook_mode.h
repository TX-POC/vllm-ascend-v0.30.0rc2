// Build-time controls for the isolated streaming-LI experiment.
//
// 0: official C8 score workspace only.  This is the bitwise control.
// 1: score workspace is caller-owned/graph-visible, but no descriptor is
//    published.  It isolates a changed score-GM address from publication.
// 2: after the existing score DMA, publish a fixed-slot score-tile receipt.
//    The receipt contains no KVAF, SIMT, queue, lease, or Get state by
//    itself.
//
// These are two compile-time axes. Legal pairs are constrained below:
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

// ``inline=post`` has one further compile-time split.  ``full`` retains the
// native-enqueue Direct-HCCP closure; ``empty_shell`` deliberately keeps the
// *same* 24-input/11-output ABI while compiling that closure out completely.
// The latter is a mathematical-control binary, not a runtime if-statement:
// it lets us prove whether merely linking/initialising the post path perturbs
// source-derived C8 before investigating any transport behaviour.
//
// A source tree compiled outside the isolated staging script must default to
// no post closure.  The build script chooses ``full`` only when it is
// explicitly building a real ``inline=post`` artifact.
#ifndef A5_STREAMING_LI_GET_POST_CLOSURE_MODE
#define A5_STREAMING_LI_GET_POST_CLOSURE_MODE 0
#endif

static_assert(A5_STREAMING_LI_GET_POST_CLOSURE_MODE >= 0 &&
              A5_STREAMING_LI_GET_POST_CLOSURE_MODE <= 1,
              "A5 streaming LI post closure mode must be 0 (empty_shell) or 1 (full).");
static_assert(A5_STREAMING_LI_GET_INLINE_MODE == 2 ||
              A5_STREAMING_LI_GET_POST_CLOSURE_MODE == 0,
              "A5 streaming LI post closure is valid only with inline=post.");

#if A5_STREAMING_LI_GET_INLINE_MODE == 2 && \
    A5_STREAMING_LI_GET_POST_CLOSURE_MODE == 1
#define A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE 1
#else
#define A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE 0
#endif

// Full post has two scheduling implementations.  Mode 0 is the historical
// bring-up path: score scan, private-slot claim, descriptor materialisation
// and doorbell all happen in one score-UB hand-off.  Mode 1 is the P6
// CandidateSeed pipeline: a normal C8 worker first compacts its exact score
// tile into a bounded *worker-local* seed microbatch, releases the score UB,
// and only at a later tile seam opportunistically borrows a Direct-HCCP
// queue to post it.  This is deliberately a scheduler choice, not a core
// assignment: no AIV is reserved as a Get worker.
//
// Keep the default at 0 for the existing P3 mathematical/payload oracle.  A
// P6 artifact must opt in at build time, which makes its provenance and test
// evidence unambiguous.
#ifndef A5_STREAMING_LI_GET_POST_SCHEDULER_MODE
#define A5_STREAMING_LI_GET_POST_SCHEDULER_MODE 0
#endif

static_assert(A5_STREAMING_LI_GET_POST_SCHEDULER_MODE >= 0 &&
              A5_STREAMING_LI_GET_POST_SCHEDULER_MODE <= 1,
              "A5 streaming LI post scheduler must be 0 (immediate) or 1 (candidate seed).");
static_assert(A5_STREAMING_LI_GET_POST_SCHEDULER_MODE == 0 ||
              A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE == 1,
              "candidate-seed scheduling requires inline=post with the full Direct-HCCP closure.");

#if A5_STREAMING_LI_GET_POST_SCHEDULER_MODE == 1
#define A5_STREAMING_LI_GET_HAS_CANDIDATE_SEED_PIPELINE 1
#else
#define A5_STREAMING_LI_GET_HAS_CANDIDATE_SEED_PIPELINE 0
#endif

// In the decode-only empty shell, output #8 is a bridge alias of
// ``actual_seq_lengths_key``.  Do not allocate a side-output UB buffer or
// issue MTE3 writes from C8 for data that already exists as an immutable C8
// input.  All other artifacts retain their current independent side output.
#if A5_STREAMING_LI_GET_INLINE_MODE == 2 && \
    A5_STREAMING_LI_GET_POST_CLOSURE_MODE == 0
#define A5_STREAMING_LI_GET_SCORE_SOURCE_LENS_IS_INPUT_ALIAS 1
#else
#define A5_STREAMING_LI_GET_SCORE_SOURCE_LENS_IS_INPUT_ALIAS 0
#endif

// The publisher owns an externally visible score-workspace/receipt protocol;
// inline mode owns the transaction-local scan-time observer/transport path.
// Product post/HCCP therefore remains mutually exclusive with publication.
// The sole exception is audit + publisher: it is a score-provenance diagnostic
// that still has no KVAF/SIMT/HCCP closure, but exposes the exact score-GM that
// C8 Top-K consumes so the audit Scalar receipt can be checked independently.
static_assert(A5_STREAMING_LI_GET_INLINE_MODE == 0 ||
              A5_STREAMING_LI_GET_PUBLISH_MODE == 0 ||
              (A5_STREAMING_LI_GET_INLINE_MODE == 1 &&
               A5_STREAMING_LI_GET_PUBLISH_MODE == 2),
              "inline mode requires publication mode off, except audit + publisher provenance.");
