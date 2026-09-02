// P9 direct-resident ABI shared by the future source-derived C8 kernel and
// its Stage-B terminalizer. This header is intentionally standalone: it
// defines no transport implementation and imports no legacy P7/P8 closure.

#ifndef VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_CONTRACT_H_
#define VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_CONTRACT_H_

#include <cstdint>

namespace vllm_ascend::a5_c8_exact_speculative_resident {

constexpr uint32_t kCanonicalResidentSlotCount = 2048U;
constexpr uint32_t kCandidateSeedWords = 4U;
// The score seam uses two caller-provided microbatches in the *existing* C8
// invalid-index UB buffer: 2 x 64 x four int32 words.  These are local
// hand-off buffers, not permanently assigned Get workers or a new UB arena.
constexpr uint32_t kCandidateMicrobatchCapacity = 64U;
constexpr uint32_t kCandidateMicrobatchCount = 2U;

// Official C8 allocates outInvalidBuf_ as topkCount * sizeof(int32_t).  P9
// requires the same fixed Top-2048 contract, so that already-existing buffer
// is exactly 8192 bytes.  Do not describe the complete P9 hand-off as "2
// KiB": the two seed batches consume 2 KiB, and the dynamic queue lease plus
// source-slot reservation receipts need another 32 bytes.  Giving every byte
// a named offset makes the no-new-TBuf claim mechanically checkable.
constexpr uint32_t kExistingC8InvalidScratchBytes =
    kCanonicalResidentSlotCount * sizeof(int32_t);
constexpr uint32_t kCandidateSeedScratchOffset = 0U;
constexpr uint32_t kCandidateSeedScratchBytes =
    kCandidateMicrobatchCount * kCandidateMicrobatchCapacity *
    kCandidateSeedWords * sizeof(int32_t);
constexpr uint32_t kQueueLeaseReceiptScratchOffset =
    kCandidateSeedScratchOffset + kCandidateSeedScratchBytes;
constexpr uint32_t kQueueLeaseReceiptScratchBytes = 16U;
constexpr uint32_t kCandidateReservationScratchOffset =
    kQueueLeaseReceiptScratchOffset + kQueueLeaseReceiptScratchBytes;
constexpr uint32_t kCandidateReservationScratchBytes = 16U;
constexpr uint32_t kP9ScoreScratchBytes =
    kCandidateReservationScratchOffset + kCandidateReservationScratchBytes;

// Stage B takes exclusive ownership of the same existing buffer only after
// every initial score-time seed is either handed to KVAF or explicitly
// deferred *and* the last initial wave has reached terminal CQ.  It then
// needs a full KVAF dense-request wave, not the score seam's 64-seed batch:
// the Direct-HCCP ABI accepts at most 256 requests per descriptor.  The
// winner bitmap identifies final-winner direct slots; it is not a second
// source/slot directory.
constexpr uint32_t kExactRecallWaveCapacity = 256U;
constexpr uint32_t kExactRecallSeedScratchOffset = 0U;
constexpr uint32_t kExactRecallSeedScratchBytes =
    kExactRecallWaveCapacity * kCandidateSeedWords * sizeof(int32_t);
constexpr uint32_t kStageBUsedSlotBitmapScratchOffset =
    kExactRecallSeedScratchOffset + kExactRecallSeedScratchBytes;
constexpr uint32_t kStageBUsedSlotBitmapBytes =
    kCanonicalResidentSlotCount / 8U;
constexpr uint32_t kStageBQueueLeaseReceiptScratchOffset =
    kStageBUsedSlotBitmapScratchOffset + kStageBUsedSlotBitmapBytes;
constexpr uint32_t kStageBQueueLeaseReceiptScratchBytes = 16U;
constexpr uint32_t kStageBReservationScratchOffset =
    kStageBQueueLeaseReceiptScratchOffset +
    kStageBQueueLeaseReceiptScratchBytes;
constexpr uint32_t kStageBReservationScratchBytes = 16U;
constexpr uint32_t kP9ScratchBytes = kStageBReservationScratchOffset +
    kStageBReservationScratchBytes;

constexpr bool IsP9ExistingScratchLayoutValid(const uint32_t bytes)
{
    return bytes >= kP9ScratchBytes;
}

enum CandidateSeedWord : uint32_t {
    kCandidateRow = 0U,
    kCandidateSource = 1U,
    kCandidateKvObject = 2U,
    // This is a per-pool canonical-slot *ordinal*, never an affine physical
    // row.  The Direct-HCCP descriptor and final attention_slots resolve it
    // through canonical_resident_slots[req_pool_entries[row], ordinal].
    kCandidateResidentSlotOrdinal = 3U,
};

// A candidate owns one supplied normal-resident destination until the
// invocation's initial CQ drain. The same slot may later serve exact recall
// only after that drain proves no initial post can still write it.
struct CandidateSeed {
    int32_t row;
    int32_t source;
    int32_t kvObject;
    int32_t residentSlotOrdinal;
};

// `canonical_resident_slots` is a graph-visible int32[P, 2048] table. P is
// the resident-token-pool row count, not the current execution row count:
// execution row r must first resolve p = req_pool_entries[r].  Each table
// entry is a stable logical QSFA slot. The P9-only resident_block_table input
// is execution-row indexed and converts it to the packed-cache physical HCCP
// output row. This deliberately replaces the earlier affine [first_slot,
// slots_per_row, stride, rows] assumption: request rows may be a non-identity
// permutation of pool rows during a captured forward.
struct CanonicalResidentSlotTable {
    __gm__ int32_t* slots;
    uint32_t poolCount;
};

enum class CompletionPhase : uint32_t {
    kScoreScan = 0U,
    kInitialDrain = 1U,
    kExactRecall = 2U,
    kRecallDrain = 3U,
    kComplete = 4U,
};

__aicore__ inline bool IsCanonicalResidentSlotTableValid(
    const CanonicalResidentSlotTable table)
{
    return table.slots != nullptr && table.poolCount != 0U;
}

__aicore__ inline int32_t LoadCanonicalResidentLogicalSlot(
    const CanonicalResidentSlotTable table, const uint32_t pool_row,
    const uint32_t ordinal)
{
    if (!IsCanonicalResidentSlotTableValid(table) ||
        pool_row >= table.poolCount || ordinal >= kCanonicalResidentSlotCount) {
        return -1;
    }
    return table.slots[static_cast<uint64_t>(pool_row) *
                           kCanonicalResidentSlotCount + ordinal];
}

static_assert(sizeof(CandidateSeed) == kCandidateSeedWords * sizeof(int32_t),
              "CandidateSeed must remain four int32 words.");
static_assert(kCandidateSeedScratchBytes == 2048U,
              "P9 CandidateSeed hand-off must reuse exactly 2 KiB of C8 UB.");
static_assert(kQueueLeaseReceiptScratchOffset % 16U == 0U &&
                  kCandidateReservationScratchOffset % 16U == 0U,
              "P9 SIMT receipt hand-offs require 16-byte existing-UB alignment.");
static_assert(kExactRecallSeedScratchBytes == 4096U,
              "P9 Stage-B recall must carry one 256-entry KVAF wave.");
static_assert(kStageBQueueLeaseReceiptScratchOffset % 16U == 0U &&
                  kStageBReservationScratchOffset % 16U == 0U,
              "P9 Stage-B SIMT receipts require 16-byte existing-UB alignment.");
static_assert(kP9ScratchBytes == 4384U,
              "P9 existing-UB layout changed: update the source layout gate.");
static_assert(kP9ScratchBytes <= kExistingC8InvalidScratchBytes,
              "P9 may not allocate a new TBuf beyond official C8 invalid scratch.");
static_assert(kCanonicalResidentSlotCount == 2048U,
              "P9 exact C8 output count is fixed at Top-2048.");

}  // namespace vllm_ascend::a5_c8_exact_speculative_resident

#endif  // VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_CONTRACT_H_
