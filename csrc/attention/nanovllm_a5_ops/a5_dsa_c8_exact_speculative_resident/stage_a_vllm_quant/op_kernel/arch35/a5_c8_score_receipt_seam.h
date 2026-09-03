// Copyright (c) 2026.
//
// P10 Stage-A exact-score receipt seam.
//
// This seam is intentionally a raw score-surface descriptor publisher.  It
// neither selects candidates nor sees S->O mapping, resident state, private
// slots, transport, or completion state.  Those operations belong to the next
// stage after the descriptor is claimed.  Keeping the seam that narrow lets the
// sibling preserve the original C8 Top-K dataflow independently.

#ifndef VLLM_ASCEND_A5_DSA_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_A_SCORE_RECEIPT_SEAM_H_
#define VLLM_ASCEND_A5_DSA_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_A_SCORE_RECEIPT_SEAM_H_

#include <cstdint>

#include "a5_c8_receipt_ring.h"

namespace vllm_ascend {
namespace a5_dsa_c8_exact_speculative_resident {
namespace stage_a {

// Plan one logical C8 row segment before the producer reserves its ring slot.
// `score_surface_base_bytes` is the beginning of the derived sibling's
// explicit graph-visible uint16 score-window pool.  `score_window_bytes` is
// the fixed per-ring-slot stride.  Neither value may describe the current C8
// opaque score workspace.
struct ExactScoreWindowPlan {
  uint32_t row;
  uint32_t source_begin;
  uint32_t valid_count;
  uint64_t score_surface_base_bytes;
  uint32_t score_window_bytes;
  uint32_t score_element_bytes;
  uint32_t receipt_sequence;
};

// A formal score-surface descriptor for one completed logical C8 row segment.
// Its offset exists only after a receipt-slot reservation: slot `i` owns
// `[score_surface_base + i * score_window_bytes, ...)` for the duration of
// FILLING -> CLAIMED -> DONE.  That makes the extra DMA target stable before
// it starts and prevents a producer from copying into an unleased window.
struct ExactScoreWindow {
  uint32_t row;
  uint32_t source_begin;
  uint32_t valid_count;
  uint64_t score_surface_offset_bytes;
  uint32_t score_element_bytes;
  uint32_t receipt_sequence;
};

struct ReservedExactScoreWindow {
  ReceiptWriteLease receipt_lease;
  ExactScoreWindow window;
};

template <typename AtomicOps>
class ExactScoreReceiptSeam {
 public:
  // This is step 1 of the only valid Stage-A publication sequence:
  // reserve -> score-window DMA -> wait score-DMA completion -> publish.
  // A false return is bounded backpressure: Stage-A skips its extra copy and
  // leaves the original C8 score/Top-K path untouched for Guaranteed Recall.
  bool TryReserveScoreWindow(
      const ExactScoreWindowPlan &plan,
      const ScoreReceiptRingPublisher<AtomicOps> &publisher,
      ScoreReceiptRing *ring, ReservedExactScoreWindow *reservation) const {
    const uint64_t required_window_bytes =
        static_cast<uint64_t>(plan.valid_count) * sizeof(uint16_t);
    if (ring == nullptr || reservation == nullptr ||
        plan.valid_count == 0U || plan.receipt_sequence == 0U ||
        plan.score_element_bytes != sizeof(uint16_t) ||
        required_window_bytes > plan.score_window_bytes) {
      return false;
    }

    ReceiptWriteLease lease{};
    if (!publisher.TryReserve(ring, plan.receipt_sequence, &lease)) {
      return false;
    }
    const uint64_t slot_offset =
        static_cast<uint64_t>(lease.slot_index) * plan.score_window_bytes;
    if (plan.score_surface_base_bytes > UINT64_MAX - slot_offset) {
      publisher.AbortBeforePublish(lease);
      return false;
    }
    reservation->receipt_lease = lease;
    reservation->window = ExactScoreWindow{
        plan.row,
        plan.source_begin,
        plan.valid_count,
        plan.score_surface_base_bytes + slot_offset,
        plan.score_element_bytes,
        plan.receipt_sequence,
    };
    return true;
  }

  // This is step 3.  The caller may invoke it only after the independent
  // score-DMA completion edge (an MTE3 completion edge) has fired.  The DDR
  // publish barrier in ScoreReceiptRingPublisher makes the completed payload
  // visible to Stage-B; it does not complete an asynchronous score DMA.
  bool PublishReservedScoreWindowAfterDma(
      const ReservedExactScoreWindow &reservation,
      const ScoreReceiptRingPublisher<AtomicOps> &publisher) const {
    const ExactScoreWindow &window = reservation.window;
    ScoreReceiptHeader header{};
    header.sequence = window.receipt_sequence;
    header.row = window.row;
    header.source_begin = window.source_begin;
    header.valid_count = window.valid_count;
    header.score_surface_offset_bytes = window.score_surface_offset_bytes;
    header.score_element_bytes = window.score_element_bytes;
    header.flags = 0U;
    return publisher.Publish(reservation.receipt_lease, header);
  }

  // If the extra DMA cannot be issued or its completion edge reports an
  // error, only this pre-publication path may return the ring slot.
  void AbortReservedScoreWindowBeforePublish(
      const ReservedExactScoreWindow &reservation,
      const ScoreReceiptRingPublisher<AtomicOps> &publisher) const {
    publisher.AbortBeforePublish(reservation.receipt_lease);
  }
};

}  // namespace stage_a
}  // namespace a5_dsa_c8_exact_speculative_resident
}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_A5_DSA_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_A_SCORE_RECEIPT_SEAM_H_
