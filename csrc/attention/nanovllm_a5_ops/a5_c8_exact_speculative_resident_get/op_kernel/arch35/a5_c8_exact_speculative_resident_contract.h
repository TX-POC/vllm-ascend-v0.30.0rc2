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
    kCandidateResidentSlot = 3U,
};

// A candidate owns one supplied normal-resident destination until the
// invocation's initial CQ drain. The same slot may later serve exact recall
// only after that drain proves no initial post can still write it.
struct CandidateSeed {
    int32_t row;
    int32_t source;
    int32_t kvObject;
    int32_t residentSlot;
};

struct CanonicalResidentRange {
    int32_t firstSlot;
    int32_t slotCount;
};

// One logical C8 output row owns one fixed canonical Top-2048 interval.  The
// intervals may live inside a larger layer allocation, hence rowStride can be
// larger than 2048.  CandidateCompact assigns only a row's own interval; it
// never partitions that interval by AIV, Direct-HCCP queue, or worker id.
struct CanonicalResidentLayout {
    int32_t firstSlot;
    int32_t slotsPerRow;
    int32_t rowStride;
    uint32_t rowCount;
};

enum class CompletionPhase : uint32_t {
    kScoreScan = 0U,
    kInitialDrain = 1U,
    kExactRecall = 2U,
    kRecallDrain = 3U,
    kComplete = 4U,
};

constexpr bool IsCanonicalRange(const CanonicalResidentRange range)
{
    return range.firstSlot >= 0 &&
           range.slotCount == static_cast<int32_t>(kCanonicalResidentSlotCount);
}

constexpr bool IsSlotInRange(const CanonicalResidentRange range,
                             const int32_t residentSlot)
{
    return IsCanonicalRange(range) && residentSlot >= range.firstSlot &&
           residentSlot < range.firstSlot + range.slotCount;
}

constexpr bool IsCanonicalLayout(const CanonicalResidentLayout layout)
{
    return layout.firstSlot >= 0 &&
           layout.slotsPerRow ==
               static_cast<int32_t>(kCanonicalResidentSlotCount) &&
           layout.rowStride >= layout.slotsPerRow && layout.rowCount != 0U;
}

constexpr int64_t CanonicalRowFirstSlot(const CanonicalResidentLayout layout,
                                        const uint32_t row)
{
    return static_cast<int64_t>(layout.firstSlot) +
           static_cast<int64_t>(row) * layout.rowStride;
}

constexpr bool IsSlotInCanonicalRow(const CanonicalResidentLayout layout,
                                    const uint32_t row,
                                    const int32_t residentSlot)
{
    const int64_t first = CanonicalRowFirstSlot(layout, row);
    return IsCanonicalLayout(layout) && row < layout.rowCount &&
           static_cast<int64_t>(residentSlot) >= first &&
           static_cast<int64_t>(residentSlot) < first + layout.slotsPerRow;
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
