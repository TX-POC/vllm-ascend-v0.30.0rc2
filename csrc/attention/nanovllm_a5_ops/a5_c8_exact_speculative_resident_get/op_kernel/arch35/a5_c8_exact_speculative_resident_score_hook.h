// P9 score-seam hook for the source-derived A5 C8 operator.
//
// This header is intentionally narrow.  It observes the exact uint16 score
// tile produced by the official LI reduction, compacts cold candidates into
// caller-owned 2 KiB scratch, and hands a batch to the *dynamic* PostOnly
// executor after C8 has released that score tile.  It does not own a Direct-
// HCCP queue, a resident slot lifetime, or a physical AIV role.
//
// Required source-derived insertion point in QLIVector::ProcessVec1:
//
//   BatchMulWeightAndReduceSum(outBase, ...);
//   // P9 V->S / S->V lifetime envelope, then CompactExactScoreTile(...)
//   DataCopyPad(scoreGm[...], outBase, ...);
//
// The caller may obtain a ready CandidatePostView only after the S->V
// acknowledgement and the normal C8 score publication hand-off.  The Post
// executor dynamically leases a KVAF queue, fills its descriptor, publishes
// it, rings PostOnly, and later drains CQ.  This keeps score-UB ownership and
// transport ownership separate without creating a sidecar score recompute.

#ifndef VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_HOOK_H_
#define VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_HOOK_H_

#include <cstdint>

#include "kernel_operator.h"
#include "quant_lightning_indexer_common.h"

#include "a5_c8_exact_speculative_resident_contract.h"

// The strict reference binary keeps this at zero.  A P9 target binary must
// explicitly select one, then still pass a graph-visible runtime enable word
// before this hook observes a score.  This two-stage gate makes the no-hook
// C8 bitwise control mechanically distinct from the resident-get closure.
#ifndef A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM
#define A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM 0
#endif

static_assert(A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM == 0 ||
                  A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM == 1,
              "P9 score seam must be disabled (0) or enabled (1).");

namespace QLIKernel {
namespace exact_speculative_resident {

using namespace AscendC;
using vllm_ascend::a5_c8_exact_speculative_resident::CandidateSeed;
using vllm_ascend::a5_c8_exact_speculative_resident::CanonicalResidentLayout;
using vllm_ascend::a5_c8_exact_speculative_resident::IsCanonicalLayout;
using vllm_ascend::a5_c8_exact_speculative_resident::kCandidateMicrobatchCapacity;
using vllm_ascend::a5_c8_exact_speculative_resident::kCandidateMicrobatchCount;
using vllm_ascend::a5_c8_exact_speculative_resident::kCanonicalResidentSlotCount;

constexpr int32_t kCandidateSourceUnmapped = -1;
constexpr int32_t kCandidateResidentSlotUnassigned = -1;

enum class CandidateMicrobatchState : uint32_t {
    kFree = 0U,
    kReady = 1U,
    // Ownership changes to the dynamic Post/CQ executor only after TakeReady
    // succeeds.  The score seam must not reuse the batch until the executor
    // has copied it into the queue descriptor and accepted PostOnly.
    kHandedToExecutor = 2U,
};

// The hook exposes precisely the identities required by a Direct-HCCP
// PostOnly executor.  It deliberately contains no queue id or AIV id: queue
// acquisition is a dynamic operation over the control's shared lease pool.
struct CandidatePostView {
    __ubuf__ CandidateSeed* seeds;
    uint32_t count;
    uint32_t producerId;

    __gm__ int32_t* candidateSourceToSlot;
    __gm__ int32_t* candidateSlotCursor;
    __gm__ uint8_t* layerResidentGetControl;

    CanonicalResidentLayout residentLayout;
    // Carry the graph-visible row count separately from the layout so the
    // executor can reject a stale/cross-op layout before it touches KVAF.
    uint32_t rowCount;
    uint32_t sourceCapacity;
    uint32_t rowsPerEntry;
};

static_assert(sizeof(CandidateSeed) == 4U * sizeof(int32_t),
              "P9 score seam expects [row, source, object, resident_slot].");

// Graph-visible inputs owned by the P9 operator.  All pointers are HBM
// tensors; no pointer refers to an implicit host callback or a transient
// slot-pool protocol.  `runtimeEnable` is int32[1] and must be exactly one to enable a
// compiled P9 hook for an invocation.
struct C8ExactSpeculativeResidentScoreHookConfig {
    __gm__ uint8_t* runtimeEnable;
    __gm__ uint8_t* residentThreshold;
    __gm__ uint8_t* residentThresholdValid;
    __gm__ uint8_t* reqPoolEntries;
    __gm__ uint8_t* kvObjectMap;
    __gm__ uint8_t* kvObjectLens;
    __gm__ uint8_t* persistentObjectSlotsSnapshot;

    // `candidateSourceToSlot` has shape [rowCount, sourceCapacity], starts
    // at -1, and is committed by the Post executor only after PostOnly has
    // accepted the fully published descriptor. `candidateSlotCursor` has one
    // atomic cursor per row. No inverse slot->source map is needed: after the
    // initial CQ drain, Stage B derives a 2048-bit used-slot bitmap from the
    // exact Top-K and reuses every nonwinner fixed slot for recall.
    __gm__ uint8_t* candidateSourceToSlot;
    __gm__ uint8_t* candidateSlotCursor;
    __gm__ uint8_t* layerResidentGetControl;

    uint32_t rowCount;
    uint32_t sourceCapacity;
    uint32_t poolSize;
    uint32_t kvObjectCapacity;
    uint32_t rowsPerEntry;
    uint32_t producerId;
    CanonicalResidentLayout residentLayout;
};

// Two fixed microbatches are supplied by the source-derived vector service.
// They must alias the already-reserved 2 KiB C8 scratch, not a newly allocated
// TBuf.  This is the bounded backpressure point: if both are busy, the hook
// drops only a speculative opportunity; final Exact Top-K recall remains
// authoritative.
class C8ExactSpeculativeResidentScoreHook {
public:
    __aicore__ inline void Init(
        const C8ExactSpeculativeResidentScoreHookConfig& config,
        const QLICommon::ConstInfo& constInfo,
        __ubuf__ CandidateSeed* candidateScratch)
    {
        config_ = config;
        constInfo_ = constInfo;
        enabled_ = false;
        deferredCandidates_ = 0U;
        for (uint32_t slot = 0U; slot < kCandidateMicrobatchCount; ++slot) {
            batches_[slot].seeds = candidateScratch == nullptr
                ? nullptr
                : candidateScratch + static_cast<uint64_t>(slot) *
                    kCandidateMicrobatchCapacity;
            batches_[slot].count = 0U;
            batches_[slot].state = CandidateMicrobatchState::kFree;
        }

#if A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM == 1
        if (candidateScratch == nullptr || config.runtimeEnable == nullptr ||
            config.residentThreshold == nullptr ||
            config.residentThresholdValid == nullptr ||
            config.reqPoolEntries == nullptr || config.kvObjectMap == nullptr ||
            config.kvObjectLens == nullptr ||
            config.persistentObjectSlotsSnapshot == nullptr ||
            config.candidateSourceToSlot == nullptr ||
            config.candidateSlotCursor == nullptr ||
            config.layerResidentGetControl == nullptr || config.rowCount == 0U ||
            config.sourceCapacity == 0U || config.poolSize == 0U ||
            config.kvObjectCapacity == 0U || config.rowsPerEntry == 0U ||
            !IsCanonicalLayout(config.residentLayout) ||
            config.residentLayout.rowCount != config.rowCount ||
            constInfo_.sparseCount != kCanonicalResidentSlotCount) {
            return;
        }

        runtimeEnable_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.runtimeEnable));
        // A disabled invocation must be indistinguishable from the official
        // C8 path: it does not read score metadata or enter the V->S seam.
        if (runtimeEnable_.GetValue(0U) != 1) {
            return;
        }
        residentThreshold_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.residentThreshold));
        residentThresholdValid_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.residentThresholdValid));
        reqPoolEntries_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.reqPoolEntries));
        kvObjectMap_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.kvObjectMap));
        kvObjectLens_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.kvObjectLens));
        persistentObjectSlots_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(
                config.persistentObjectSlotsSnapshot));
        candidateSourceToSlot_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.candidateSourceToSlot));
        candidateSlotCursor_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.candidateSlotCursor));
        enabled_ = true;
#endif
    }

    __aicore__ inline bool Enabled() const
    {
        return enabled_;
    }

    __aicore__ inline uint32_t DeferredCandidates() const
    {
        return deferredCandidates_;
    }

    // This function is the only score-UB consumer.  It must run under the
    // explicit V->S / S->V lifetime envelope at the exact source-derived
    // seam.  It neither acquires a queue nor writes any HCCP descriptor.
    __aicore__ inline void CompactExactScoreTile(
        const LocalTensor<uint16_t>& outBase,
        const QLICommon::RunInfo& info,
        int64_t curAivS1Idx,
        int64_t curAivS1ProcNum,
        int64_t curS2Idx)
    {
#if A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM == 1
        if (!enabled_) {
            return;
        }
        const uint32_t batchIndex = FindFreeBatch();
        if (batchIndex == kInvalidBatch) {
            CountDeferredCandidates(outBase, info, curAivS1Idx,
                                    curAivS1ProcNum, curS2Idx);
            return;
        }

        CandidateMicrobatch& batch = batches_[batchIndex];
        uint32_t count = 0U;
        for (int64_t localRow = 0; localRow < curAivS1ProcNum; ++localRow) {
            const uint64_t row = OutputRow(info, curAivS1Idx + localRow);
            if (row >= config_.rowCount) {
                continue;
            }
            const int32_t validInTile = ValidInTile(
                info, curAivS1Idx, localRow, curS2Idx);
            const uint64_t outRowOffset = static_cast<uint64_t>(localRow) *
                (UB_BANK_DEPTH_STRIDE / sizeof(uint16_t));
            for (int32_t inTile = 0; inTile < validInTile; ++inTile) {
                const uint32_t source =
                    static_cast<uint32_t>(curS2Idx + inTile);
                const uint16_t score = outBase.GetValue(outRowOffset + inTile);
                CandidateSeed seed{};
                if (!ResolveColdCandidate(score, static_cast<uint32_t>(row),
                                          source, &seed)) {
                    continue;
                }
                if (count == kCandidateMicrobatchCapacity) {
                    ++deferredCandidates_;
                    continue;
                }
                batch.seeds[count++] = seed;
            }
        }
        batch.count = count;
        batch.state = count == 0U ? CandidateMicrobatchState::kFree
                                  : CandidateMicrobatchState::kReady;
#else
        (void)outBase;
        (void)info;
        (void)curAivS1Idx;
        (void)curAivS1ProcNum;
        (void)curS2Idx;
#endif
    }

    // Returns one local seed batch to a later, dynamic PostOnly executor. It
    // intentionally says nothing about a queue id: every normal LI worker can
    // submit it if it wins a queue lease, and no worker is reserved for Get.
    __aicore__ inline bool TakeReadyPostView(CandidatePostView* view,
                                             uint32_t* batchToken)
    {
#if A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM == 1
        if (!enabled_ || view == nullptr || batchToken == nullptr) {
            return false;
        }
        const uint32_t batchIndex = FindReadyBatch();
        if (batchIndex == kInvalidBatch) {
            return false;
        }
        CandidateMicrobatch& batch = batches_[batchIndex];
        batch.state = CandidateMicrobatchState::kHandedToExecutor;
        view->seeds = batch.seeds;
        view->count = batch.count;
        view->producerId = config_.producerId;
        view->candidateSourceToSlot = reinterpret_cast<__gm__ int32_t*>(
            config_.candidateSourceToSlot);
        view->candidateSlotCursor = reinterpret_cast<__gm__ int32_t*>(
            config_.candidateSlotCursor);
        view->layerResidentGetControl = config_.layerResidentGetControl;
        view->residentLayout = config_.residentLayout;
        view->rowCount = config_.rowCount;
        view->sourceCapacity = config_.sourceCapacity;
        view->rowsPerEntry = config_.rowsPerEntry;
        *batchToken = batchIndex;
        return true;
#else
        (void)view;
        (void)batchToken;
        return false;
#endif
    }

    // The executor calls this after it has either rolled back a pre-doorbell
    // claim or accepted PostOnly and committed every source->fixed-slot map.
    // It does not wait for CQ: after PostOnly the queue descriptor, not this
    // UB seed array, owns the request.  Releasing a local microbatch has no
    // cache-slot lifecycle meaning; it only lets this worker compact a later
    // score tile while HCCP DMA runs.
    __aicore__ inline void CompletePostView(uint32_t batchToken)
    {
#if A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM == 1
        if (batchToken >= kCandidateMicrobatchCount) {
            return;
        }
        CandidateMicrobatch& batch = batches_[batchToken];
        if (batch.state == CandidateMicrobatchState::kHandedToExecutor) {
            batch.count = 0U;
            batch.state = CandidateMicrobatchState::kFree;
        }
#else
        (void)batchToken;
#endif
    }

    // If the dynamic pool is busy, this drops only a speculative batch.  No
    // resident destination was claimed by this hook, so Stage-B exact recall
    // can later assign all genuine Top-2048 misses without rollback work.
    __aicore__ inline void DeferPostView(uint32_t batchToken)
    {
        CompletePostView(batchToken);
    }

    // Stage B invokes this before it drains the last initial wave and
    // repurposes the official outInvalidBuf_ for its 256-entry exact-recall
    // descriptor.  A ready batch has never reached a queue descriptor, so
    // dropping it is semantically safe: final Exact Top-K recall will cover
    // every genuine winner.  A handed batch is deliberately left alone;
    // only its owning Post executor may make its CQ terminal.
    __aicore__ inline void DeferAllReadyPostViews()
    {
#if A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM == 1
        for (uint32_t slot = 0U; slot < kCandidateMicrobatchCount; ++slot) {
            if (batches_[slot].state == CandidateMicrobatchState::kReady) {
                batches_[slot].count = 0U;
                batches_[slot].state = CandidateMicrobatchState::kFree;
            }
        }
#endif
    }

private:
    static constexpr uint32_t kInvalidBatch = kCandidateMicrobatchCount;

    struct CandidateMicrobatch {
        __ubuf__ CandidateSeed* seeds = nullptr;
        uint32_t count = 0U;
        CandidateMicrobatchState state = CandidateMicrobatchState::kFree;
    };

    __aicore__ inline uint32_t FindFreeBatch() const
    {
        for (uint32_t slot = 0U; slot < kCandidateMicrobatchCount; ++slot) {
            if (batches_[slot].state == CandidateMicrobatchState::kFree) {
                return slot;
            }
        }
        return kInvalidBatch;
    }

    __aicore__ inline uint32_t FindReadyBatch() const
    {
        for (uint32_t slot = 0U; slot < kCandidateMicrobatchCount; ++slot) {
            if (batches_[slot].state == CandidateMicrobatchState::kReady &&
                batches_[slot].count != 0U) {
                return slot;
            }
        }
        return kInvalidBatch;
    }

    __aicore__ inline uint64_t OutputRow(const QLICommon::RunInfo& info,
                                          int64_t s1Index) const
    {
        if (constInfo_.sparseCount == 0U || s1Index < 0) {
            return config_.rowCount;
        }
        return info.indiceOutOffset / constInfo_.sparseCount +
               static_cast<uint64_t>(s1Index);
    }

    __aicore__ inline int32_t ValidInTile(const QLICommon::RunInfo& info,
                                           int64_t curAivS1Idx,
                                           int64_t localRow,
                                           int64_t curS2Idx) const
    {
        int64_t validS2 = info.actS2Size;
        if (constInfo_.attenMaskFlag) {
            const int64_t base = static_cast<int64_t>(info.actS2SizeOrig) -
                static_cast<int64_t>(info.actS1Size) + curAivS1Idx + 1;
            validS2 = (base + localRow) / constInfo_.cmpRatio;
        }
        const int64_t remain = validS2 - curS2Idx;
        const int64_t sourceRemain =
            static_cast<int64_t>(config_.sourceCapacity) - curS2Idx;
        if (remain <= 0 || sourceRemain <= 0) {
            return 0;
        }
        const int64_t tile = remain < info.actualSingleProcessSInnerSize
            ? remain
            : info.actualSingleProcessSInnerSize;
        const int64_t bounded = tile < sourceRemain ? tile : sourceRemain;
        return static_cast<int32_t>(bounded);
    }

    // This is the mathematical score-time predicate only.  It reads the
    // persistent snapshot but does not modify it, does not touch C8 Top-K,
    // and does not reserve a canonical slot.  Reservation is deliberately
    // deferred until a dynamic queue lease has been won by the executor.
    __aicore__ inline bool ResolveColdCandidate(uint16_t score, uint32_t row,
                                                 uint32_t source,
                                                 CandidateSeed* seed) const
    {
        if (seed == nullptr || row >= config_.rowCount ||
            source >= config_.sourceCapacity ||
            residentThresholdValid_.GetValue(row) != 1) {
            return false;
        }
        const uint16_t threshold = static_cast<uint16_t>(
            static_cast<uint32_t>(residentThreshold_.GetValue(row)) & 0xffffU);
        if (score < threshold) {
            return false;
        }
        const int32_t poolRow = reqPoolEntries_.GetValue(row);
        if (poolRow < 0 || static_cast<uint32_t>(poolRow) >= config_.poolSize) {
            return false;
        }
        const int32_t object = kvObjectMap_.GetValue(
            static_cast<uint64_t>(poolRow) * config_.sourceCapacity + source);
        const int32_t objectLen = kvObjectLens_.GetValue(row);
        if (object < 0 || objectLen <= 0 ||
            static_cast<uint32_t>(object) >= config_.kvObjectCapacity ||
            static_cast<uint32_t>(object) >= static_cast<uint32_t>(objectLen)) {
            return false;
        }
        if (persistentObjectSlots_.GetValue(
                static_cast<uint64_t>(poolRow) * config_.kvObjectCapacity +
                static_cast<uint32_t>(object)) >= 0) {
            return false;
        }
        // A terminally committed candidate needs no second speculative post.
        // A currently claimed value is also skipped; the executor resolves it
        // before Stage-B exact recall starts.
        const uint64_t sourceOffset =
            static_cast<uint64_t>(row) * config_.sourceCapacity + source;
        if (candidateSourceToSlot_.GetValue(sourceOffset) !=
            kCandidateSourceUnmapped) {
            return false;
        }
        seed->row = static_cast<int32_t>(row);
        seed->source = static_cast<int32_t>(source);
        seed->kvObject = object;
        seed->residentSlot = kCandidateResidentSlotUnassigned;
        return true;
    }

    __aicore__ inline void CountDeferredCandidates(
        const LocalTensor<uint16_t>& outBase, const QLICommon::RunInfo& info,
        int64_t curAivS1Idx, int64_t curAivS1ProcNum, int64_t curS2Idx)
    {
        for (int64_t localRow = 0; localRow < curAivS1ProcNum; ++localRow) {
            const uint64_t row = OutputRow(info, curAivS1Idx + localRow);
            if (row >= config_.rowCount) {
                continue;
            }
            const int32_t validInTile = ValidInTile(
                info, curAivS1Idx, localRow, curS2Idx);
            const uint64_t outRowOffset = static_cast<uint64_t>(localRow) *
                (UB_BANK_DEPTH_STRIDE / sizeof(uint16_t));
            for (int32_t inTile = 0; inTile < validInTile; ++inTile) {
                CandidateSeed ignored{};
                if (ResolveColdCandidate(
                        outBase.GetValue(outRowOffset + inTile),
                        static_cast<uint32_t>(row),
                        static_cast<uint32_t>(curS2Idx + inTile), &ignored)) {
                    ++deferredCandidates_;
                }
            }
        }
    }

    C8ExactSpeculativeResidentScoreHookConfig config_{};
    QLICommon::ConstInfo constInfo_{};
    CandidateMicrobatch batches_[kCandidateMicrobatchCount]{};
    GlobalTensor<int32_t> runtimeEnable_;
    GlobalTensor<int32_t> residentThreshold_;
    GlobalTensor<int32_t> residentThresholdValid_;
    GlobalTensor<int32_t> reqPoolEntries_;
    GlobalTensor<int32_t> kvObjectMap_;
    GlobalTensor<int32_t> kvObjectLens_;
    GlobalTensor<int32_t> persistentObjectSlots_;
    GlobalTensor<int32_t> candidateSourceToSlot_;
    GlobalTensor<int32_t> candidateSlotCursor_;
    bool enabled_ = false;
    uint32_t deferredCandidates_ = 0U;
};

}  // namespace exact_speculative_resident
}  // namespace QLIKernel

#endif  // VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_HOOK_H_
