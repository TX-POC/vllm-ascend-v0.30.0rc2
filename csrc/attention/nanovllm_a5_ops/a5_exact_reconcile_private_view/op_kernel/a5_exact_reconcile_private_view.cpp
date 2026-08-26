/**
 * Transaction-local exact Top-K reconciliation for the A5 C8 prefetch path.
 *
 * This is intentionally the read-only counterpart of A5LiduCacheUpdate.  It
 * preserves that kernel's exact miss-prefix / hit-suffix ordering and its
 * first-unprotected-victim scan, but never writes cacheSlotsSnapshotGm_.
 * Instead it records the two mapping writes that legacy LIDU would have made
 * in placementDeltaGm_.  A later QSFA-terminal transaction stage owns any
 * actual mapping publication.
 */

#include "kernel_operator.h"

#include "a5_exact_reconcile_private_view_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kSparseCount = 2048U;
constexpr uint32_t kMaxCacheTokens = 16256U;
constexpr uint32_t kCacheChunk = 2048U;
constexpr uint32_t kPlacementDeltaPlanes = 3U;
constexpr uint32_t kPrivateViewPlanes = 4U;

constexpr int32_t kViewResidentHit = 0;
constexpr int32_t kViewPrivateReady = 1;
constexpr int32_t kViewFallbackRequired = 2;
constexpr int32_t kViewInvalid = 3;

class A5ExactReconcilePrivateViewKernel {
public:
    __aicore__ inline A5ExactReconcilePrivateViewKernel(
        TPipe* pipe, const A5ExactReconcilePrivateViewTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR exactTopkIndices, GM_ADDR reqPoolEntries,
        GM_ADDR cacheSlotsSnapshot, GM_ADDR cacheTokens,
        GM_ADDR candidateLens, GM_ADDR logicalTokenMap,
        GM_ADDR privateSourceToSlot, GM_ADDR privateReady,
        GM_ADDR sourceIds, GM_ADDR destinationSlots, GM_ADDR missCounts,
        GM_ADDR placementDelta, GM_ADDR privateViewDescriptors,
        GM_ADDR readyPrivateSourceIds, GM_ADDR readyPrivateSlots,
        GM_ADDR readyPrivateDestinationSlots, GM_ADDR readyPrivateCounts,
        GM_ADDR fallbackLogicalTokenIds, GM_ADDR fallbackDestinationSlots,
        GM_ADDR fallbackCounts)
    {
        coreIdx_ = GetBlockIdx();
        exactTopkIndicesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(exactTopkIndices));
        reqPoolEntriesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(reqPoolEntries));
        // cacheSlotsSnapshotGm_ is deliberately read-only.  There must be no
        // SetValue call on it in this file: that is the transaction boundary.
        cacheSlotsSnapshotGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(cacheSlotsSnapshot));
        cacheTokensGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(cacheTokens));
        candidateLensGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(candidateLens));
        logicalTokenMapGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(logicalTokenMap));
        privateSourceToSlotGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(privateSourceToSlot));
        privateReadyGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(privateReady));

        sourceIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sourceIds));
        destinationSlotsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(destinationSlots));
        missCountsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(missCounts));
        placementDeltaGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(placementDelta));
        privateViewDescriptorsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(privateViewDescriptors));
        readyPrivateSourceIdsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(readyPrivateSourceIds));
        readyPrivateSlotsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(readyPrivateSlots));
        readyPrivateDestinationSlotsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(readyPrivateDestinationSlots));
        readyPrivateCountsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(readyPrivateCounts));
        fallbackLogicalTokenIdsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(fallbackLogicalTokenIds));
        fallbackDestinationSlotsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(fallbackDestinationSlots));
        fallbackCountsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(fallbackCounts));

        pipe_->InitBuffer(outputTokensBuf_, kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(outputSlotsBuf_, kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(missTokensBuf_, kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(hitTokensBuf_, kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(hitSlotsBuf_, kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(cacheChunkBuf_, kCacheChunk * sizeof(int32_t));
        pipe_->InitBuffer(protectedSlotsBuf_,
                          kMaxCacheTokens * sizeof(uint8_t));
        pipe_->InitBuffer(deltaVictimBuf_, kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(deltaDestinationBuf_,
                          kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(viewLogicalBuf_, kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(viewKindBuf_, kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(viewPrivateSlotBuf_,
                          kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(readySourceBuf_, kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(readySlotBuf_, kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(readyDestinationBuf_,
                          kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(fallbackLogicalBuf_,
                          kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(fallbackDestinationBuf_,
                          kSparseCount * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t batch = coreIdx_; batch < tiling_->batchSize;
             batch += tiling_->usedCoreNum) {
            ProcessRow(batch);
        }
    }

private:
    __aicore__ inline uint32_t MinU32(uint32_t left, uint32_t right) const
    {
        return left < right ? left : right;
    }

    __aicore__ inline void WaitVectorToScalar()
    {
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    __aicore__ inline void LoadTopk(
        uint32_t batch, LocalTensor<int32_t> outputTokens)
    {
        DataCopyExtParams copy{1, kSparseCount * sizeof(int32_t), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad<int32_t, PaddingMode::Normal>(
            outputTokens,
            exactTopkIndicesGm_[static_cast<uint64_t>(batch) * kSparseCount],
            copy, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
    }

    __aicore__ inline void StoreVector(
        GlobalTensor<int32_t>& output, uint64_t base,
        LocalTensor<int32_t> input)
    {
        DataCopyExtParams copy{1, kSparseCount * sizeof(int32_t), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        DataCopyPad<int32_t, PaddingMode::Normal>(output[base], input, copy,
                                                   pad);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);
    }

    __aicore__ inline void StoreScalar(
        GlobalTensor<int32_t>& output, uint32_t batch,
        LocalTensor<int32_t> scratch, int32_t value)
    {
        scratch.SetValue(0, value);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        DataCopyParams copy{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
        DataCopyPad(output[batch], scratch, copy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);
    }

    __aicore__ inline void InitializePlans(
        LocalTensor<int32_t> outputTokens, LocalTensor<int32_t> outputSlots,
        LocalTensor<int32_t> missTokens,
        LocalTensor<int32_t> deltaVictim, LocalTensor<int32_t> viewLogical,
        LocalTensor<int32_t> deltaDestination,
        LocalTensor<int32_t> viewKind, LocalTensor<int32_t> viewPrivateSlot,
        LocalTensor<int32_t> readySource, LocalTensor<int32_t> readySlot,
        LocalTensor<int32_t> readyDestination,
        LocalTensor<int32_t> fallbackLogical,
        LocalTensor<int32_t> fallbackDestination)
    {
        Duplicate(outputTokens, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(outputSlots, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(missTokens, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(deltaVictim, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(deltaDestination, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(viewLogical, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(viewKind, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(viewPrivateSlot, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(readySource, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(readySlot, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(readyDestination, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(fallbackLogical, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(fallbackDestination, static_cast<int32_t>(-1),
                  kSparseCount);
        WaitVectorToScalar();
    }

    __aicore__ inline void StoreRow(
        uint32_t batch, LocalTensor<int32_t> outputTokens,
        LocalTensor<int32_t> outputSlots, int32_t missCount,
        LocalTensor<int32_t> deltaVictim,
        LocalTensor<int32_t> deltaIncoming,
        LocalTensor<int32_t> deltaDestination,
        LocalTensor<int32_t> viewLogical,
        LocalTensor<int32_t> viewKind, LocalTensor<int32_t> viewPrivateSlot,
        LocalTensor<int32_t> readySource, LocalTensor<int32_t> readySlot,
        LocalTensor<int32_t> readyDestination, int32_t readyCount,
        LocalTensor<int32_t> fallbackLogical,
        LocalTensor<int32_t> fallbackDestination, int32_t fallbackCount)
    {
        const uint64_t topkBase = static_cast<uint64_t>(batch) * kSparseCount;
        StoreVector(sourceIdsGm_, topkBase, outputTokens);
        StoreVector(destinationSlotsGm_, topkBase, outputSlots);

        const uint64_t deltaBase =
            static_cast<uint64_t>(batch) * kPlacementDeltaPlanes * kSparseCount;
        StoreVector(placementDeltaGm_, deltaBase, deltaVictim);
        StoreVector(placementDeltaGm_, deltaBase + kSparseCount,
                    deltaIncoming);
        StoreVector(placementDeltaGm_, deltaBase + 2U * kSparseCount,
                    deltaDestination);

        const uint64_t viewBase = static_cast<uint64_t>(batch) *
                                  kPrivateViewPlanes * kSparseCount;
        StoreVector(privateViewDescriptorsGm_, viewBase, viewLogical);
        StoreVector(privateViewDescriptorsGm_, viewBase + kSparseCount,
                    outputSlots);
        StoreVector(privateViewDescriptorsGm_, viewBase + 2U * kSparseCount,
                    viewKind);
        StoreVector(privateViewDescriptorsGm_, viewBase + 3U * kSparseCount,
                    viewPrivateSlot);

        StoreVector(readyPrivateSourceIdsGm_, topkBase, readySource);
        StoreVector(readyPrivateSlotsGm_, topkBase, readySlot);
        StoreVector(readyPrivateDestinationSlotsGm_, topkBase,
                    readyDestination);
        StoreVector(fallbackLogicalTokenIdsGm_, topkBase, fallbackLogical);
        StoreVector(fallbackDestinationSlotsGm_, topkBase,
                    fallbackDestination);

        // Scalar stores use MTE3 rather than direct scalar GM writes.  Adjacent
        // request rows can otherwise false-share one AIV DCache line.
        StoreScalar(missCountsGm_, batch, outputTokens, missCount);
        StoreScalar(readyPrivateCountsGm_, batch, outputTokens, readyCount);
        StoreScalar(fallbackCountsGm_, batch, outputTokens, fallbackCount);
    }

    __aicore__ inline void ProcessRow(uint32_t batch)
    {
        LocalTensor<int32_t> outputTokens = outputTokensBuf_.Get<int32_t>();
        LocalTensor<int32_t> outputSlots = outputSlotsBuf_.Get<int32_t>();
        LocalTensor<int32_t> missTokens = missTokensBuf_.Get<int32_t>();
        LocalTensor<int32_t> hitTokens = hitTokensBuf_.Get<int32_t>();
        LocalTensor<int32_t> hitSlots = hitSlotsBuf_.Get<int32_t>();
        LocalTensor<int32_t> cacheChunk = cacheChunkBuf_.Get<int32_t>();
        LocalTensor<uint8_t> protectedSlots =
            protectedSlotsBuf_.Get<uint8_t>();
        LocalTensor<int32_t> deltaVictim = deltaVictimBuf_.Get<int32_t>();
        LocalTensor<int32_t> deltaDestination =
            deltaDestinationBuf_.Get<int32_t>();
        LocalTensor<int32_t> viewLogical = viewLogicalBuf_.Get<int32_t>();
        LocalTensor<int32_t> viewKind = viewKindBuf_.Get<int32_t>();
        LocalTensor<int32_t> viewPrivateSlot =
            viewPrivateSlotBuf_.Get<int32_t>();
        LocalTensor<int32_t> readySource = readySourceBuf_.Get<int32_t>();
        LocalTensor<int32_t> readySlot = readySlotBuf_.Get<int32_t>();
        LocalTensor<int32_t> readyDestination =
            readyDestinationBuf_.Get<int32_t>();
        LocalTensor<int32_t> fallbackLogical =
            fallbackLogicalBuf_.Get<int32_t>();
        LocalTensor<int32_t> fallbackDestination =
            fallbackDestinationBuf_.Get<int32_t>();

        InitializePlans(outputTokens, outputSlots, missTokens, deltaVictim,
                        viewLogical, deltaDestination, viewKind,
                        viewPrivateSlot, readySource, readySlot,
                        readyDestination, fallbackLogical,
                        fallbackDestination);

        const int32_t budgetValue = cacheTokensGm_.GetValue(batch);
        const int32_t candidateValue = candidateLensGm_.GetValue(batch);
        const int32_t poolRowValue = reqPoolEntriesGm_.GetValue(batch);
        if (budgetValue == 0) {
            StoreRow(batch, outputTokens, outputSlots, 0, deltaVictim,
                     missTokens, deltaDestination, viewLogical, viewKind,
                     viewPrivateSlot, readySource, readySlot,
                     readyDestination, 0, fallbackLogical,
                     fallbackDestination, 0);
            return;
        }
        if (budgetValue < static_cast<int32_t>(kSparseCount) ||
            budgetValue > static_cast<int32_t>(kMaxCacheTokens) ||
            candidateValue < static_cast<int32_t>(kSparseCount) ||
            candidateValue > static_cast<int32_t>(tiling_->sourceCapacity) ||
            poolRowValue < 0 ||
            poolRowValue >= static_cast<int32_t>(tiling_->poolSize)) {
            StoreRow(batch, outputTokens, outputSlots, -1, deltaVictim,
                     missTokens, deltaDestination, viewLogical, viewKind,
                     viewPrivateSlot, readySource, readySlot,
                     readyDestination, -1, fallbackLogical,
                     fallbackDestination, -1);
            return;
        }

        const uint32_t budget = static_cast<uint32_t>(budgetValue);
        const uint32_t candidateLen = static_cast<uint32_t>(candidateValue);
        const uint64_t cacheBase = static_cast<uint64_t>(poolRowValue) *
                                   tiling_->sourceCapacity;
        const uint64_t sourceMapBase = static_cast<uint64_t>(batch) *
                                       tiling_->sourceCapacity;
        const uint64_t privateCapacity = static_cast<uint64_t>(
            tiling_->privateQueueCount) * tiling_->privateRowsPerQueue;

        Duplicate(protectedSlots, static_cast<uint8_t>(0), budget);
        WaitVectorToScalar();
        LoadTopk(batch, outputTokens);

        // This block is mechanically identical to A5LiduCacheUpdate's
        // classification phase, except it reads from a snapshot.
        uint32_t missCount = 0;
        uint32_t hitCount = 0;
        for (uint32_t index = 0; index < kSparseCount; ++index) {
            const int32_t source = outputTokens.GetValue(index);
            int32_t slot = -1;
            if (source >= 0 && source < candidateValue) {
                slot = cacheSlotsSnapshotGm_.GetValue(
                    cacheBase + static_cast<uint32_t>(source));
            }
            if (slot >= 0 && slot < budgetValue) {
                hitTokens.SetValue(hitCount, source);
                hitSlots.SetValue(hitCount, slot);
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 1);
                ++hitCount;
            } else {
                missTokens.SetValue(missCount, source);
                ++missCount;
            }
        }

        uint32_t assigned = 0;
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        for (uint32_t chunkBase = 0;
             chunkBase < candidateLen && assigned < missCount;
             chunkBase += kCacheChunk) {
            const uint32_t chunkLen = MinU32(kCacheChunk,
                                              candidateLen - chunkBase);
            DataCopyExtParams copy{
                1, static_cast<uint32_t>(chunkLen * sizeof(int32_t)), 0, 0,
                0};
            DataCopyPad<int32_t, PaddingMode::Normal>(
                cacheChunk, cacheSlotsSnapshotGm_[cacheBase + chunkBase],
                copy, pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            for (uint32_t offset = 0;
                 offset < chunkLen && assigned < missCount; ++offset) {
                const int32_t slot = cacheChunk.GetValue(offset);
                if (slot < 0 || slot >= budgetValue ||
                    protectedSlots.GetValue(static_cast<uint32_t>(slot)) != 0) {
                    continue;
                }
                const int32_t victimSource =
                    static_cast<int32_t>(chunkBase + offset);
                const int32_t missSource = missTokens.GetValue(assigned);
                if (missSource < 0 || missSource >= candidateValue) {
                    continue;
                }
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 2);
                // Legacy LIDU would clear victimSource and assign missSource
                // here.  This transaction-local stage records that exact pair
                // but leaves cacheSlotsSnapshotGm_ untouched.
                deltaVictim.SetValue(assigned, victimSource);
                deltaDestination.SetValue(assigned, slot);
                outputTokens.SetValue(assigned, missSource);
                outputSlots.SetValue(assigned, slot);
                ++assigned;
            }
        }

        if (assigned != missCount || missCount + hitCount != kSparseCount) {
            // Preserve the legacy kernel's diagnostic first-three-output
            // behavior for an infeasible/invalid exact list: it has already
            // loaded Top-K and may have populated a partial miss prefix before
            // returning miss_count=-1.  The private-only plans remain their
            // initial all--1 state with invalid counts.  Unlike legacy LIDU,
            // the equivalent partial mapping mutations remain only in delta.
            StoreRow(batch, outputTokens, outputSlots, -1, deltaVictim,
                     missTokens, deltaDestination, viewLogical, viewKind,
                     viewPrivateSlot, readySource, readySlot,
                     readyDestination, -1, fallbackLogical,
                     fallbackDestination, -1);
            return;
        }
        for (uint32_t index = 0; index < hitCount; ++index) {
            outputTokens.SetValue(missCount + index, hitTokens.GetValue(index));
            outputSlots.SetValue(missCount + index, hitSlots.GetValue(index));
        }

        // The exact Top-K order is now fixed.  Only its miss prefix is allowed
        // to consume speculative state.  A private READY bit is meaningful
        // only after the Direct-HCCP CQ terminal has been observed by the
        // producer/transaction that wrote it.
        uint32_t readyCount = 0;
        uint32_t fallbackCount = 0;
        bool malformed = false;
        for (uint32_t index = 0; index < kSparseCount; ++index) {
            const int32_t source = outputTokens.GetValue(index);
            const int32_t destination = outputSlots.GetValue(index);
            int32_t logicalToken = -1;
            if (source >= 0 && source < candidateValue) {
                logicalToken = logicalTokenMapGm_.GetValue(
                    cacheBase + static_cast<uint32_t>(source));
            }
            viewLogical.SetValue(index, logicalToken);
            if (index >= missCount) {
                viewKind.SetValue(index, kViewResidentHit);
                continue;
            }

            if (logicalToken < 0) {
                viewKind.SetValue(index, kViewInvalid);
                malformed = true;
                continue;
            }
            const int32_t privateSlot = privateSourceToSlotGm_.GetValue(
                sourceMapBase + static_cast<uint32_t>(source));
            if (privateSlot == -1) {
                viewKind.SetValue(index, kViewFallbackRequired);
                fallbackLogical.SetValue(fallbackCount, logicalToken);
                fallbackDestination.SetValue(fallbackCount, destination);
                ++fallbackCount;
                continue;
            }
            if (privateSlot < 0 ||
                static_cast<uint64_t>(privateSlot) >= privateCapacity) {
                viewKind.SetValue(index, kViewInvalid);
                malformed = true;
                continue;
            }
            if (privateReadyGm_.GetValue(
                    static_cast<uint64_t>(privateSlot)) == 1) {
                viewKind.SetValue(index, kViewPrivateReady);
                viewPrivateSlot.SetValue(index, privateSlot);
                readySource.SetValue(readyCount, source);
                readySlot.SetValue(readyCount, privateSlot);
                readyDestination.SetValue(readyCount, destination);
                ++readyCount;
            } else {
                viewKind.SetValue(index, kViewFallbackRequired);
                fallbackLogical.SetValue(fallbackCount, logicalToken);
                fallbackDestination.SetValue(fallbackCount, destination);
                ++fallbackCount;
            }
        }

        const int32_t finalReadyCount = malformed
                                            ? -1
                                            : static_cast<int32_t>(readyCount);
        const int32_t finalFallbackCount = malformed
                                               ? -1
                                               : static_cast<int32_t>(fallbackCount);
        StoreRow(batch, outputTokens, outputSlots,
                 static_cast<int32_t>(missCount), deltaVictim, missTokens,
                 deltaDestination, viewLogical, viewKind, viewPrivateSlot,
                 readySource, readySlot, readyDestination, finalReadyCount,
                 fallbackLogical, fallbackDestination, finalFallbackCount);
    }

private:
    TPipe* pipe_;
    const A5ExactReconcilePrivateViewTilingData* tiling_;
    uint32_t coreIdx_ = 0;

    GlobalTensor<int32_t> exactTopkIndicesGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> cacheSlotsSnapshotGm_;
    GlobalTensor<int32_t> cacheTokensGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> logicalTokenMapGm_;
    GlobalTensor<int32_t> privateSourceToSlotGm_;
    GlobalTensor<int32_t> privateReadyGm_;
    GlobalTensor<int32_t> sourceIdsGm_;
    GlobalTensor<int32_t> destinationSlotsGm_;
    GlobalTensor<int32_t> missCountsGm_;
    GlobalTensor<int32_t> placementDeltaGm_;
    GlobalTensor<int32_t> privateViewDescriptorsGm_;
    GlobalTensor<int32_t> readyPrivateSourceIdsGm_;
    GlobalTensor<int32_t> readyPrivateSlotsGm_;
    GlobalTensor<int32_t> readyPrivateDestinationSlotsGm_;
    GlobalTensor<int32_t> readyPrivateCountsGm_;
    GlobalTensor<int32_t> fallbackLogicalTokenIdsGm_;
    GlobalTensor<int32_t> fallbackDestinationSlotsGm_;
    GlobalTensor<int32_t> fallbackCountsGm_;

    TBuf<TPosition::VECCALC> outputTokensBuf_;
    TBuf<TPosition::VECCALC> outputSlotsBuf_;
    TBuf<TPosition::VECCALC> missTokensBuf_;
    TBuf<TPosition::VECCALC> hitTokensBuf_;
    TBuf<TPosition::VECCALC> hitSlotsBuf_;
    TBuf<TPosition::VECCALC> cacheChunkBuf_;
    TBuf<TPosition::VECCALC> protectedSlotsBuf_;
    TBuf<TPosition::VECCALC> deltaVictimBuf_;
    TBuf<TPosition::VECCALC> deltaDestinationBuf_;
    TBuf<TPosition::VECCALC> viewLogicalBuf_;
    TBuf<TPosition::VECCALC> viewKindBuf_;
    TBuf<TPosition::VECCALC> viewPrivateSlotBuf_;
    TBuf<TPosition::VECCALC> readySourceBuf_;
    TBuf<TPosition::VECCALC> readySlotBuf_;
    TBuf<TPosition::VECCALC> readyDestinationBuf_;
    TBuf<TPosition::VECCALC> fallbackLogicalBuf_;
    TBuf<TPosition::VECCALC> fallbackDestinationBuf_;
};
}  // namespace

extern "C" __global__ __aicore__ void a5_exact_reconcile_private_view(
    GM_ADDR exactTopkIndices, GM_ADDR reqPoolEntries,
    GM_ADDR cacheSlotsSnapshot, GM_ADDR cacheTokens, GM_ADDR candidateLens,
    GM_ADDR logicalTokenMap, GM_ADDR privateSourceToSlot,
    GM_ADDR privateReady, GM_ADDR sourceIds, GM_ADDR destinationSlots,
    GM_ADDR missCounts, GM_ADDR placementDelta, GM_ADDR privateViewDescriptors,
    GM_ADDR readyPrivateSourceIds, GM_ADDR readyPrivateSlots,
    GM_ADDR readyPrivateDestinationSlots, GM_ADDR readyPrivateCounts,
    GM_ADDR fallbackLogicalTokenIds, GM_ADDR fallbackDestinationSlots,
    GM_ADDR fallbackCounts, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5ExactReconcilePrivateViewTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    A5ExactReconcilePrivateViewKernel op(&pipe, &tilingData);
    op.Init(exactTopkIndices, reqPoolEntries, cacheSlotsSnapshot, cacheTokens,
            candidateLens, logicalTokenMap, privateSourceToSlot, privateReady,
            sourceIds, destinationSlots, missCounts, placementDelta,
            privateViewDescriptors, readyPrivateSourceIds, readyPrivateSlots,
            readyPrivateDestinationSlots, readyPrivateCounts,
            fallbackLogicalTokenIds, fallbackDestinationSlots, fallbackCounts);
    op.Process();
}
