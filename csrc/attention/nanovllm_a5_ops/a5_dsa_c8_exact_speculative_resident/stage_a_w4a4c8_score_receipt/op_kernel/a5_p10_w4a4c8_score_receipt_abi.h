#ifndef A5_P10_W4A4C8_SCORE_RECEIPT_ABI_H
#define A5_P10_W4A4C8_SCORE_RECEIPT_ABI_H

#include <cstdint>

// This is a graph ABI, not an address into the native C8 workspace.  Stage A
// owns score_surface; Stage B may consume only records made READY after the
// corresponding score bytes are visible in GM.
constexpr uint32_t A5_P10_SPARSE_COUNT = 2048;
constexpr uint32_t A5_P10_SCORE_ELEMENT_BYTES = sizeof(uint16_t);
constexpr uint32_t A5_P10_RECEIPT_WORDS64 = 5;
constexpr uint32_t A5_P10_RECEIPT_BYTES =
    A5_P10_RECEIPT_WORDS64 * sizeof(uint64_t);
// The descriptor is logically five words (40 B).  The producer must issue an
// exact 40-B DataCopyPad transfer; it must never round the GM destination up
// to a cache line, because adjacent graph-owned records are only 40 B apart.
// The existing invalid-output scratch supplies a 64-B *local* staging view,
// so this does not add or resize a C8 TBuf.
constexpr uint32_t A5_P10_RECEIPT_LOCAL_STAGING_BYTES = 64;

enum class A5P10ReceiptState : int32_t {
    FREE = 0,
    FILLING = 1,
    READY = 2,
    CLAIMED = 3,
    DONE = 4,
};

// `forward_control` is an explicitly graph-owned mutable alias shared with
// Stage B.  It is not an output dependency: Stage B starts independently,
// polls this word while it is OPEN, and may consume READY receipt records
// before C8 reaches its final Top-K completion.  Keep Stage-A's one control
// word separate from receipt_state, whose complete range is record state.
constexpr uint32_t A5_P10_FORWARD_CONTROL_PRODUCER_SEALED_WORD = 0U;
constexpr int32_t A5_P10_FORWARD_CONTROL_PRODUCER_OPEN = 0;
constexpr int32_t A5_P10_FORWARD_CONTROL_PRODUCER_SEALED = 1;
// Two diagnostic witnesses make overlap observable without adding an arena or
// lifetime protocol. Stage A stores the final static receipt sequence before
// it stores SEALED. Stage B atomically latches the first READY record it
// claimed while it observed OPEN. Both words are initialized to zero by
// forward preparation and are never used for scheduling decisions.
constexpr uint32_t A5_P10_FORWARD_CONTROL_PRODUCER_SEAL_SEQUENCE_WORD = 8U;
constexpr uint32_t A5_P10_FORWARD_CONTROL_FIRST_PRESEAL_READY_CLAIM_WORD = 9U;
constexpr uint32_t A5_P10_FORWARD_CONTROL_MIN_WORDS = 10U;

// Forward preparation owns reset, not either concurrent kernel:
//   receipt_state[0:receipt_capacity] = FREE
//   forward_control[PRODUCER_SEALED_WORD] = PRODUCER_OPEN
//   forward_control[PRODUCER_SEAL_SEQUENCE_WORD] = 0
//   forward_control[FIRST_PRESEAL_READY_CLAIM_WORD] = 0
// before Stage A and Stage B are launched from the same graph-visible input
// aliases.  During execution Stage A writes only FREE -> FILLING -> READY and
// finally PRODUCER_SEALED; Stage B may claim only READY.  There is no state
// sentinel after receipt_capacity and no Stage-A reset race with Stage B.

// Packed into receipt_ring[slot, 0:5] as five 64-bit words, in native
// little-endian field order.  Stage A writes this exact 40-B record before it
// changes receipt_state[slot] from FILLING to READY; Stage B must treat every
// non-READY state as unreadable.
struct A5P10ScoreReceiptDescriptor {
    uint64_t score_surface_offset_bytes;
    uint32_t row_begin;
    uint32_t row_count;
    uint32_t source_begin;
    uint32_t source_count;
    uint32_t row_stride_bytes;
    uint32_t score_element_bytes;
    uint32_t sequence;
    uint32_t reserved;
};

static_assert(sizeof(A5P10ScoreReceiptDescriptor) ==
                  A5_P10_RECEIPT_BYTES,
              "receipt ABI must remain five int64 words");
static_assert(sizeof(uint16_t) == A5_P10_SCORE_ELEMENT_BYTES,
              "P10 score surface is uint16");

#endif  // A5_P10_W4A4C8_SCORE_RECEIPT_ABI_H
