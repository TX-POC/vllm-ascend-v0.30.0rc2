// P9 direct-resident ABI shared by the future source-derived C8 kernel and
// its Stage-B terminalizer. This header is intentionally standalone: it
// defines no transport implementation and imports no legacy P7/P8 closure.

#ifndef VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_CONTRACT_H_
#define VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_CONTRACT_H_

#include <cstdint>

namespace vllm_ascend::a5_c8_exact_speculative_resident {

constexpr uint32_t kCanonicalResidentSlotCount = 2048U;
constexpr uint32_t kCandidateSeedWords = 4U;

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

static_assert(sizeof(CandidateSeed) == kCandidateSeedWords * sizeof(int32_t),
              "CandidateSeed must remain four int32 words.");
static_assert(kCanonicalResidentSlotCount == 2048U,
              "P9 exact C8 output count is fixed at Top-2048.");

}  // namespace vllm_ascend::a5_c8_exact_speculative_resident

#endif  // VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_CONTRACT_H_
