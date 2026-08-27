// Build-time publication mode for the isolated streaming-LI experiment.
//
// 0: official C8 score workspace only.  This is the bitwise control.
// 1: score workspace is caller-owned/graph-visible, but no descriptor is
//    published.  It isolates a changed score-GM address from publication.
// 2: after the existing score DMA, publish a fixed-slot score-tile receipt.
//    The receipt contains no KVAF, SIMT, queue, lease, or Get state.  A later
//    independent sidecar is the only component allowed to consume it and
//    submit a speculative Get.
//
// Keeping this compile-time is intentional.  A runtime flag would still
// compile the publication path into the C8 clone and could not diagnose an
// instruction/register-layout perturbation.
#ifndef A5_STREAMING_LI_GET_PUBLISH_MODE
#define A5_STREAMING_LI_GET_PUBLISH_MODE 0
#endif

static_assert(A5_STREAMING_LI_GET_PUBLISH_MODE >= 0 &&
              A5_STREAMING_LI_GET_PUBLISH_MODE <= 2,
              "A5 streaming LI publication mode must be 0, 1, or 2.");
