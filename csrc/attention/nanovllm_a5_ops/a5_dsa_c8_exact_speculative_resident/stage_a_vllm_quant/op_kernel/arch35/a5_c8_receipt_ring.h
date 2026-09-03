// Copyright (c) 2026.
//
// Formal P10 Stage-A GM score-receipt descriptor ring.
//
// Stage A copies exact C8 scores into a separately declared, bounded
// graph-visible score-window pool, then publishes one descriptor for a
// completed logical row segment.  It must never expose C8's reusable per-AIC
// score workspace: a later gS1 block may overwrite that workspace while Stage
// B still reads an earlier receipt.
// The ring deliberately contains no selection, object mapping, resident-slot,
// transport, or completion state.  The following stage owns score scanning,
// compaction, object/slot admission, posting, and completion drain.

#ifndef VLLM_ASCEND_A5_DSA_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_A_RECEIPT_RING_H_
#define VLLM_ASCEND_A5_DSA_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_A_RECEIPT_RING_H_

#include <cstdint>

namespace vllm_ascend {
namespace a5_dsa_c8_exact_speculative_resident {
namespace stage_a {

enum class ReceiptState : uint32_t {
  kEmpty = 0U,
  kFilling = 1U,
  kPublished = 2U,
  kClaimed = 3U,
  kDone = 4U,
};

// One descriptor describes a completed native-C8 uint16 score window.  The
// offset is relative to the formal score-surface tensor/resource supplied to
// the streaming sibling, never to CANN's opaque per-AIC workspace.
struct ScoreReceiptHeader {
  uint32_t sequence;
  uint32_t row;
  uint32_t source_begin;
  uint32_t valid_count;
  uint64_t score_surface_offset_bytes;
  uint32_t score_element_bytes;
  uint32_t flags;
};
static_assert(sizeof(ScoreReceiptHeader) == 32U,
              "P10 score receipt ABI must remain 32 bytes");

struct alignas(8) ScoreReceiptSlot {
  uint32_t state;
  uint32_t sequence;
  ScoreReceiptHeader header;
};
static_assert(sizeof(ScoreReceiptSlot) == 40U,
              "P10 receipt slot must be descriptor-only");

struct ScoreReceiptRing {
  ScoreReceiptSlot *slots;
  uint32_t slot_count;
};

struct ReceiptWriteLease {
  uint32_t slot_index;
  uint32_t sequence;
  ScoreReceiptSlot *slot;
};

// AtomicOps is injected so this header remains parseable on the host.  On
// device it is backed by the GM atomics and DDR visibility primitives chosen
// by the streaming sibling.  The caller MUST reserve the receipt slot before
// its extra score-window DMA, and establish that DMA's independent MTE3
// completion edge before Publish; DdrPublishBarrier makes completed writes
// visible, it does not complete an in-flight score DMA.
template <typename AtomicOps>
class ScoreReceiptRingPublisher {
 public:
  bool TryReserve(ScoreReceiptRing *ring, uint32_t sequence,
                  ReceiptWriteLease *lease) const {
    if (ring == nullptr || lease == nullptr || ring->slots == nullptr ||
        ring->slot_count == 0U || sequence == 0U) {
      return false;
    }
    for (uint32_t index = 0U; index < ring->slot_count; ++index) {
      ScoreReceiptSlot *slot = &ring->slots[index];
      const uint32_t before = AtomicOps::CompareExchange(
          &slot->state, static_cast<uint32_t>(ReceiptState::kEmpty),
          static_cast<uint32_t>(ReceiptState::kFilling));
      if (before == static_cast<uint32_t>(ReceiptState::kEmpty)) {
        lease->slot_index = index;
        lease->sequence = sequence;
        lease->slot = slot;
        return true;
      }
    }
    return false;
  }

  bool Publish(const ReceiptWriteLease &lease,
               const ScoreReceiptHeader &published_header) const {
    if (lease.slot == nullptr || lease.sequence == 0U ||
        published_header.sequence != lease.sequence ||
        published_header.valid_count == 0U ||
        published_header.score_element_bytes != sizeof(uint16_t)) {
      return false;
    }
    if (lease.slot->state != static_cast<uint32_t>(ReceiptState::kFilling)) {
      return false;
    }
    lease.slot->header = published_header;
    lease.slot->sequence = lease.sequence;
    AtomicOps::DdrPublishBarrier();
    AtomicOps::StoreRelease(&lease.slot->state,
                            static_cast<uint32_t>(ReceiptState::kPublished));
    return true;
  }

  void AbortBeforePublish(const ReceiptWriteLease &lease) const {
    if (lease.slot == nullptr) {
      return;
    }
    const uint32_t before = AtomicOps::CompareExchange(
        &lease.slot->state, static_cast<uint32_t>(ReceiptState::kFilling),
        static_cast<uint32_t>(ReceiptState::kEmpty));
    (void)before;
  }
};

template <typename AtomicOps>
class ScoreReceiptRingConsumer {
 public:
  bool TryClaim(ScoreReceiptRing *ring, uint32_t slot_index,
                uint32_t expected_sequence, ScoreReceiptHeader *header) const {
    if (ring == nullptr || header == nullptr || ring->slots == nullptr ||
        slot_index >= ring->slot_count || expected_sequence == 0U) {
      return false;
    }
    ScoreReceiptSlot *slot = &ring->slots[slot_index];
    const uint32_t before = AtomicOps::CompareExchange(
        &slot->state, static_cast<uint32_t>(ReceiptState::kPublished),
        static_cast<uint32_t>(ReceiptState::kClaimed));
    if (before != static_cast<uint32_t>(ReceiptState::kPublished)) {
      return false;
    }
    AtomicOps::DdrAcquireBarrier();
    if (slot->sequence != expected_sequence ||
        slot->header.sequence != expected_sequence ||
        slot->header.valid_count == 0U ||
        slot->header.score_element_bytes != sizeof(uint16_t)) {
      return false;
    }
    *header = slot->header;
    return true;
  }

  bool MarkDone(ScoreReceiptSlot *slot, uint32_t expected_sequence) const {
    if (slot == nullptr || expected_sequence == 0U ||
        slot->sequence != expected_sequence) {
      return false;
    }
    const uint32_t before = AtomicOps::CompareExchange(
        &slot->state, static_cast<uint32_t>(ReceiptState::kClaimed),
        static_cast<uint32_t>(ReceiptState::kDone));
    return before == static_cast<uint32_t>(ReceiptState::kClaimed);
  }

  bool RecycleDone(ScoreReceiptSlot *slot, uint32_t expected_sequence) const {
    if (slot == nullptr || expected_sequence == 0U ||
        slot->sequence != expected_sequence) {
      return false;
    }
    const uint32_t before = AtomicOps::CompareExchange(
        &slot->state, static_cast<uint32_t>(ReceiptState::kDone),
        static_cast<uint32_t>(ReceiptState::kEmpty));
    return before == static_cast<uint32_t>(ReceiptState::kDone);
  }
};

}  // namespace stage_a
}  // namespace a5_dsa_c8_exact_speculative_resident
}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_A5_DSA_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_A_RECEIPT_RING_H_
