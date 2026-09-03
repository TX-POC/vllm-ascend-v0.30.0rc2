#ifndef A5_P10_FORWARD_PREPARE_ABI_H
#define A5_P10_FORWARD_PREPARE_ABI_H

#include <cstdint>

// P10 preparation is a distinct graph node.  It runs before both concurrent
// device stages and resets only forward-local mutable state.  It must never
// clear a score surface, receipt descriptor, canonical Top-K, or QSFA tail.
// Words 8 and 9 are the minimal overlap witness.  They are not a lifetime
// epoch: Stage A writes its seal sequence once, and Stage B records its first
// READY claim sequence only when that claim happened before the seal.
constexpr uint32_t A5_P10_FORWARD_PREPARE_CONTROL_WORDS = 10U;
constexpr int32_t A5_P10_FORWARD_PREPARE_RECEIPT_FREE = 0;
constexpr int32_t A5_P10_FORWARD_PREPARE_PRODUCER_OPEN = 0;
constexpr int32_t A5_P10_FORWARD_PREPARE_FORWARD_INCOMPLETE = 0;
constexpr int32_t A5_P10_FORWARD_PREPARE_UNMAPPED_SLOT = -1;
constexpr uint32_t A5_P10_FORWARD_PREPARE_PRODUCER_SEAL_SEQUENCE_WORD = 8U;
constexpr uint32_t A5_P10_FORWARD_PREPARE_FIRST_PRESEAL_READY_CLAIM_WORD = 9U;

enum A5P10ForwardPrepareInputIndex : uint32_t {
    A5_P10_PREP_RECEIPT_STATE = 0U,
    A5_P10_PREP_FORWARD_CONTROL,
    A5_P10_PREP_OBJECT_TO_FORWARD_SLOT,
    A5_P10_PREP_SOURCE_TO_FORWARD_SLOT,
    A5_P10_PREP_SPECULATIVE_SLOT_CURSOR,
    A5_P10_PREP_QUEUE_LEASES,
    A5_P10_PREP_RESIDENT_READY,
};

enum A5P10ForwardPrepareOutputIndex : uint32_t {
    A5_P10_PREP_RECEIPT_STATE_ALIAS = 0U,
    A5_P10_PREP_FORWARD_CONTROL_ALIAS,
    A5_P10_PREP_OBJECT_TO_FORWARD_SLOT_ALIAS,
    A5_P10_PREP_SOURCE_TO_FORWARD_SLOT_ALIAS,
    A5_P10_PREP_SPECULATIVE_SLOT_CURSOR_ALIAS,
    A5_P10_PREP_QUEUE_LEASES_ALIAS,
    A5_P10_PREP_RESIDENT_READY_ALIAS,
};

#endif  // A5_P10_FORWARD_PREPARE_ABI_H
