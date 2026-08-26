/**
 * Transaction-local exact Top-K reconciliation for the A5 C8 prefetch path.
 *
 * This is intentionally the read-only counterpart of A5LiduCacheUpdate.  It
 * preserves that kernel's exact miss-prefix / hit-suffix ordering and its
 * first-unprotected-victim scan, but never writes kvObjectSlotsSnapshotGm_.
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
// There are exactly 2048 exact C8 selections.  A small open-addressed table
// detects a malformed many-to-one source->object map without a full object
// directory-sized UB allocation.
constexpr uint32_t kSeenKvObjectTable = 4096U;

constexpr int32_t kViewResidentHit = 0;
constexpr int32_t kViewPrivateReady = 1;
constexpr int32_t kViewFallbackRequired = 2;
constexpr int32_t kViewInvalid = 3;
// Shared transaction ABI with A5StreamingQuantLiGet: only terminal-CQ
// READY=2 may use a private payload. PENDING/FAILED stay on fallback.
constexpr int32_t kPrivateReadyReady = 2;

class A5ExactReconcilePrivateViewKernel {
public:
    __aicore__ inline A5ExactReconcilePrivateViewKernel(
        TPipe* pipe, const A5ExactReconcilePrivateViewTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR exactTopkIndices, GM_ADDR reqPoolEntries,
        GM_ADDR kvObjectSlotsSnapshot, GM_ADDR cacheTokens,
        GM_ADDR kvObjectLens, GM_ADDR scoreSourceLens, GM_ADDR kvObjectMap,
        GM_ADDR privateSourceToSlot, GM_ADDR privateReady,
        GM_ADDR kvObjectIds, GM_ADDR destinationSlots, GM_ADDR missCounts,
        GM_ADDR placementDelta, GM_ADDR privateViewDescriptors,
        GM_ADDR readyPrivateKvObjectIds, GM_ADDR readyPrivateSlots,
        GM_ADDR readyPrivateDestinationSlots, GM_ADDR readyPrivateCounts,
        GM_ADDR fallbackKvObjectIds, GM_ADDR fallbackDestinationSlots,
        GM_ADDR fallbackCounts)
    {
        coreIdx_ = GetBlockIdx();
        exactTopkIndicesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(exactTopkIndices));
        reqPoolEntriesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(reqPoolEntries));
        // kvObjectSlotsSnapshotGm_ is deliberately read-only.  There must be no
        // SetValue call on it in this file: that is the transaction boundary.
        kvObjectSlotsSnapshotGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(kvObjectSlotsSnapshot));
        cacheTokensGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(cacheTokens));
        kvObjectLensGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(kvObjectLens));
        scoreSourceLensGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(scoreSourceLens));
        kvObjectMapGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(kvObjectMap));
        privateSourceToSlotGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(privateSourceToSlot));
        privateReadyGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(privateReady));

        kvObjectIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(kvObjectIds));
        destinationSlotsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(destinationSlots));
        missCountsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(missCounts));
        placementDeltaGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(placementDelta));
        privateViewDescriptorsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(privateViewDescriptors));
        readyPrivateKvObjectIdsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(readyPrivateKvObjectIds));
        readyPrivateSlotsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(readyPrivateSlots));
        readyPrivateDestinationSlotsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(readyPrivateDestinationSlots));
        readyPrivateCountsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(readyPrivateCounts));
        fallbackKvObjectIdsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(fallbackKvObjectIds));
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
        pipe_->InitBuffer(viewKvObjectBuf_, kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(viewKindBuf_, kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(viewPrivateSlotBuf_,
                          kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(readySourceBuf_, kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(readySlotBuf_, kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(readyDestinationBuf_,
                          kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(fallbackKvObjectBuf_,
                          kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(fallbackDestinationBuf_,
                          kSparseCount * sizeof(int32_t));
        pipe_->InitBuffer(seenKvObjectBuf_,
                          kSeenKvObjectTable * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t batch = coreIdx_; batch < tiling_->rowCount;
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

    __aicore__ inline void LoadTopkSources(
        uint32_t batch, LocalTensor<int32_t> scoreSources)
    {
        DataCopyExtParams copy{1, kSparseCount * sizeof(int32_t), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad<int32_t, PaddingMode::Normal>(
            scoreSources,
            exactTopkIndicesGm_[static_cast<uint64_t>(batch) * kSparseCount],
            copy, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
    }

    __aicore__ inline bool ResolveUniqueKvObjectTopk(
        uint64_t kvObjectMapBase, uint32_t scoreSourceLen,
        uint32_t kvObjectLen,
        LocalTensor<int32_t> scoreSources,
        LocalTensor<int32_t> kvObjectIds,
        LocalTensor<int32_t> seenKvObjects)
    {
        Duplicate(seenKvObjects, static_cast<int32_t>(-1), kSeenKvObjectTable);
        WaitVectorToScalar();
        for (uint32_t index = 0; index < kSparseCount; ++index) {
            const int32_t source = scoreSources.GetValue(index);
            if (source < 0 ||
                static_cast<uint32_t>(source) >= scoreSourceLen) {
                return false;
            }
            const int32_t kvObject = kvObjectMapGm_.GetValue(
                kvObjectMapBase + static_cast<uint32_t>(source));
            if (kvObject < 0 ||
                static_cast<uint32_t>(kvObject) >= kvObjectLen) {
                return false;
            }

            // The source-to-object map must be injective over the selected
            // Top-K.  Without this guard, one real KV could consume multiple
            // destination slots and would no longer be equivalent to the
            // declared KV-object cache baseline.
            uint32_t probe =
                (static_cast<uint32_t>(kvObject) * 2654435761U) &
                (kSeenKvObjectTable - 1U);
            bool inserted = false;
            for (uint32_t attempt = 0; attempt < kSeenKvObjectTable;
                 ++attempt) {
                const int32_t previous = seenKvObjects.GetValue(probe);
                if (previous == kvObject) {
                    return false;
                }
                if (previous == -1) {
                    seenKvObjects.SetValue(probe, kvObject);
                    inserted = true;
                    break;
                }
                probe = (probe + 1U) & (kSeenKvObjectTable - 1U);
            }
            if (!inserted) {
                return false;
            }
            kvObjectIds.SetValue(index, kvObject);
        }
        return true;
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
        LocalTensor<int32_t> deltaVictim, LocalTensor<int32_t> viewKvObject,
        LocalTensor<int32_t> deltaDestination,
        LocalTensor<int32_t> viewKind, LocalTensor<int32_t> viewPrivateSlot,
        LocalTensor<int32_t> readySource, LocalTensor<int32_t> readySlot,
        LocalTensor<int32_t> readyDestination,
        LocalTensor<int32_t> fallbackKvObject,
        LocalTensor<int32_t> fallbackDestination)
    {
        Duplicate(outputTokens, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(outputSlots, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(missTokens, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(deltaVictim, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(deltaDestination, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(viewKvObject, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(viewKind, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(viewPrivateSlot, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(readySource, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(readySlot, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(readyDestination, static_cast<int32_t>(-1), kSparseCount);
        Duplicate(fallbackKvObject, static_cast<int32_t>(-1), kSparseCount);
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
        LocalTensor<int32_t> viewKvObject,
        LocalTensor<int32_t> viewKind, LocalTensor<int32_t> viewPrivateSlot,
        LocalTensor<int32_t> readySource, LocalTensor<int32_t> readySlot,
        LocalTensor<int32_t> readyDestination, int32_t readyCount,
        LocalTensor<int32_t> fallbackKvObject,
        LocalTensor<int32_t> fallbackDestination, int32_t fallbackCount)
    {
        const uint64_t topkBase = static_cast<uint64_t>(batch) * kSparseCount;
        StoreVector(kvObjectIdsGm_, topkBase, outputTokens);
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
        StoreVector(privateViewDescriptorsGm_, viewBase, viewKvObject);
        StoreVector(privateViewDescriptorsGm_, viewBase + kSparseCount,
                    outputSlots);
        StoreVector(privateViewDescriptorsGm_, viewBase + 2U * kSparseCount,
                    viewKind);
        StoreVector(privateViewDescriptorsGm_, viewBase + 3U * kSparseCount,
                    viewPrivateSlot);

        StoreVector(readyPrivateKvObjectIdsGm_, topkBase, readySource);
        StoreVector(readyPrivateSlotsGm_, topkBase, readySlot);
        StoreVector(readyPrivateDestinationSlotsGm_, topkBase,
                    readyDestination);
        StoreVector(fallbackKvObjectIdsGm_, topkBase, fallbackKvObject);
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
        LocalTensor<int32_t> viewKvObject = viewKvObjectBuf_.Get<int32_t>();
        LocalTensor<int32_t> viewKind = viewKindBuf_.Get<int32_t>();
        LocalTensor<int32_t> viewPrivateSlot =
            viewPrivateSlotBuf_.Get<int32_t>();
        LocalTensor<int32_t> readySource = readySourceBuf_.Get<int32_t>();
        LocalTensor<int32_t> readySlot = readySlotBuf_.Get<int32_t>();
        LocalTensor<int32_t> readyDestination =
            readyDestinationBuf_.Get<int32_t>();
        LocalTensor<int32_t> fallbackKvObject =
            fallbackKvObjectBuf_.Get<int32_t>();
        LocalTensor<int32_t> fallbackDestination =
            fallbackDestinationBuf_.Get<int32_t>();
        LocalTensor<int32_t> seenKvObject = seenKvObjectBuf_.Get<int32_t>();

        InitializePlans(outputTokens, outputSlots, missTokens, deltaVictim,
                        viewKvObject, deltaDestination, viewKind,
                        viewPrivateSlot, readySource, readySlot,
                        readyDestination, fallbackKvObject,
                        fallbackDestination);

        const int32_t budgetValue = cacheTokensGm_.GetValue(batch);
        const int32_t kvObjectLenValue = kvObjectLensGm_.GetValue(batch);
        const int32_t scoreSourceValue = scoreSourceLensGm_.GetValue(batch);
        const int32_t poolRowValue = reqPoolEntriesGm_.GetValue(batch);
        if (budgetValue == 0) {
            StoreRow(batch, outputTokens, outputSlots, 0, deltaVictim,
                     missTokens, deltaDestination, viewKvObject, viewKind,
                     viewPrivateSlot, readySource, readySlot,
                     readyDestination, 0, fallbackKvObject,
                     fallbackDestination, 0);
            return;
        }
        if (budgetValue < static_cast<int32_t>(kSparseCount) ||
            budgetValue > static_cast<int32_t>(kMaxCacheTokens) ||
            kvObjectLenValue < static_cast<int32_t>(kSparseCount) ||
            kvObjectLenValue >
                static_cast<int32_t>(tiling_->kvObjectCapacity) ||
            scoreSourceValue < static_cast<int32_t>(kSparseCount) ||
            scoreSourceValue >
                static_cast<int32_t>(tiling_->scoreSourceCapacity) ||
            poolRowValue < 0 ||
            poolRowValue >= static_cast<int32_t>(tiling_->poolSize)) {
            StoreRow(batch, outputTokens, outputSlots, -1, deltaVictim,
                     missTokens, deltaDestination, viewKvObject, viewKind,
                     viewPrivateSlot, readySource, readySlot,
                     readyDestination, -1, fallbackKvObject,
                     fallbackDestination, -1);
            return;
        }

        const uint32_t budget = static_cast<uint32_t>(budgetValue);
        const uint32_t kvObjectLen =
            static_cast<uint32_t>(kvObjectLenValue);
        const uint32_t scoreSourceLen =
            static_cast<uint32_t>(scoreSourceValue);
        const uint64_t kvObjectBase = static_cast<uint64_t>(poolRowValue) *
                                   tiling_->kvObjectCapacity;
        const uint64_t kvObjectMapBase = static_cast<uint64_t>(poolRowValue) *
                                        tiling_->scoreSourceCapacity;
        const uint64_t sourceMapBase = static_cast<uint64_t>(batch) *
                                       tiling_->scoreSourceCapacity;
        const uint64_t privateCapacity = static_cast<uint64_t>(
            tiling_->privateQueueCount) * tiling_->privateRowsPerQueue;

        Duplicate(protectedSlots, static_cast<uint8_t>(0), budget);
        WaitVectorToScalar();
        // Reuse deltaVictim as temporary Top-K source storage.  The raw source
        // is copied into readySource for each miss before deltaVictim becomes
        // the eventual victim-token output, so no extra 2x2048 UB buffers are
        // needed on the LI/Get critical path.
        LoadTopkSources(batch, deltaVictim);
        if (!ResolveUniqueKvObjectTopk(kvObjectMapBase, scoreSourceLen,
                                     kvObjectLen, deltaVictim,
                                     outputTokens,
                                     seenKvObject)) {
            // A source index that cannot resolve to one distinct KV object is
            // not a valid transaction input.  Fail closed rather than issue
            // a Get or publish a placement delta for a guessed address.
            InitializePlans(outputTokens, outputSlots, missTokens,
                            deltaVictim, viewKvObject, deltaDestination,
                            viewKind, viewPrivateSlot, readySource, readySlot,
                            readyDestination, fallbackKvObject,
                            fallbackDestination);
            StoreRow(batch, outputTokens, outputSlots, -1, deltaVictim,
                     missTokens, deltaDestination, viewKvObject, viewKind,
                     viewPrivateSlot, readySource, readySlot,
                     readyDestination, -1, fallbackKvObject,
                     fallbackDestination, -1);
            return;
        }

        // This is A5LiduCacheUpdate's classification phase in the real
        // declared KV-object domain. `outputTokens` already holds exact C8
        // choices translated through kv_object_map; the snapshot is
        // deliberately still read-only.
        uint32_t missCount = 0;
        uint32_t hitCount = 0;
        for (uint32_t index = 0; index < kSparseCount; ++index) {
            const int32_t kvObject = outputTokens.GetValue(index);
            int32_t slot = -1;
            if (kvObject >= 0 && kvObject < kvObjectLenValue) {
                slot = kvObjectSlotsSnapshotGm_.GetValue(
                    kvObjectBase + static_cast<uint32_t>(kvObject));
            }
            if (slot >= 0 && slot < budgetValue) {
                hitTokens.SetValue(hitCount, kvObject);
                hitSlots.SetValue(hitCount, slot);
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 1);
                ++hitCount;
            } else {
                missTokens.SetValue(missCount, kvObject);
                // Keep the original C8 source only until exact miss handling.
                // `readySource` is later overwritten in-place with object IDs
                // for the compact ready plan, after this raw source is read.
                readySource.SetValue(missCount, deltaVictim.GetValue(index));
                ++missCount;
            }
        }

        uint32_t assigned = 0;
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        for (uint32_t chunkBase = 0;
             chunkBase < kvObjectLen && assigned < missCount;
             chunkBase += kCacheChunk) {
            const uint32_t chunkLen = MinU32(kCacheChunk,
                                              kvObjectLen - chunkBase);
            DataCopyExtParams copy{
                1, static_cast<uint32_t>(chunkLen * sizeof(int32_t)), 0, 0,
                0};
            DataCopyPad<int32_t, PaddingMode::Normal>(
                cacheChunk, kvObjectSlotsSnapshotGm_[kvObjectBase + chunkBase],
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
                const int32_t victimKvObject =
                    static_cast<int32_t>(chunkBase + offset);
                const int32_t missKvObject =
                    missTokens.GetValue(assigned);
                if (missKvObject < 0 ||
                    missKvObject >= kvObjectLenValue) {
                    continue;
                }
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 2);
                // Object-domain LIDU would clear victimKvObject and
                // assign missKvObject here. This transaction-local stage
                // records that exact pair but leaves the persistent directory
                // untouched.
                deltaVictim.SetValue(assigned, victimKvObject);
                deltaDestination.SetValue(assigned, slot);
                outputTokens.SetValue(assigned, missKvObject);
                outputSlots.SetValue(assigned, slot);
                ++assigned;
            }
        }

        if (assigned != missCount || missCount + hitCount != kSparseCount) {
            // An infeasible placement plan must not expose a partial delta or
            // a partial private view. The caller sees invalid counts and takes
            // the standalone exact baseline rather than guessing a mapping.
            InitializePlans(outputTokens, outputSlots, missTokens,
                            deltaVictim, viewKvObject, deltaDestination,
                            viewKind, viewPrivateSlot, readySource, readySlot,
                            readyDestination, fallbackKvObject,
                            fallbackDestination);
            StoreRow(batch, outputTokens, outputSlots, -1, deltaVictim,
                     missTokens, deltaDestination, viewKvObject, viewKind,
                     viewPrivateSlot, readySource, readySlot,
                     readyDestination, -1, fallbackKvObject,
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
            const int32_t kvObject = outputTokens.GetValue(index);
            const int32_t destination = outputSlots.GetValue(index);
            viewKvObject.SetValue(index, kvObject);
            if (index >= missCount) {
                viewKind.SetValue(index, kViewResidentHit);
                continue;
            }

            if (kvObject < 0) {
                viewKind.SetValue(index, kViewInvalid);
                malformed = true;
                continue;
            }
            const int32_t scoreSource = readySource.GetValue(index);
            if (scoreSource < 0 ||
                static_cast<uint32_t>(scoreSource) >=
                    scoreSourceLen) {
                viewKind.SetValue(index, kViewInvalid);
                malformed = true;
                continue;
            }
            const int32_t privateSlot = privateSourceToSlotGm_.GetValue(
                sourceMapBase + static_cast<uint32_t>(scoreSource));
            if (privateSlot == -1) {
                viewKind.SetValue(index, kViewFallbackRequired);
                fallbackKvObject.SetValue(fallbackCount, kvObject);
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
                    static_cast<uint64_t>(privateSlot)) == kPrivateReadyReady) {
                viewKind.SetValue(index, kViewPrivateReady);
                viewPrivateSlot.SetValue(index, privateSlot);
                readySource.SetValue(readyCount, kvObject);
                readySlot.SetValue(readyCount, privateSlot);
                readyDestination.SetValue(readyCount, destination);
                ++readyCount;
            } else {
                viewKind.SetValue(index, kViewFallbackRequired);
                fallbackKvObject.SetValue(fallbackCount, kvObject);
                fallbackDestination.SetValue(fallbackCount, destination);
                ++fallbackCount;
            }
        }

        if (malformed) {
            // A bad private reference cannot be recovered by treating an
            // arbitrary source slot as payload. Clear every externally visible
            // plan and force the caller onto its exact standalone baseline.
            InitializePlans(outputTokens, outputSlots, missTokens,
                            deltaVictim, viewKvObject, deltaDestination,
                            viewKind, viewPrivateSlot, readySource, readySlot,
                            readyDestination, fallbackKvObject,
                            fallbackDestination);
            StoreRow(batch, outputTokens, outputSlots, -1, deltaVictim,
                     missTokens, deltaDestination, viewKvObject, viewKind,
                     viewPrivateSlot, readySource, readySlot,
                     readyDestination, -1, fallbackKvObject,
                     fallbackDestination, -1);
            return;
        }
        // readySource was temporary raw score-source storage for the miss
        // prefix. Only its compact [0, readyCount) KV-object-ID prefix is part
        // of the public plan; make the unused tail deterministic rather than
        // leaking raw C8 source IDs through an otherwise ignored output tail.
        if (readyCount < kSparseCount) {
            Duplicate(readySource[readyCount], static_cast<int32_t>(-1),
                      kSparseCount - readyCount);
            WaitVectorToScalar();
        }
        StoreRow(batch, outputTokens, outputSlots,
                 static_cast<int32_t>(missCount), deltaVictim, missTokens,
                 deltaDestination, viewKvObject, viewKind, viewPrivateSlot,
                 readySource, readySlot, readyDestination,
                 static_cast<int32_t>(readyCount), fallbackKvObject,
                 fallbackDestination, static_cast<int32_t>(fallbackCount));
    }

private:
    TPipe* pipe_;
    const A5ExactReconcilePrivateViewTilingData* tiling_;
    uint32_t coreIdx_ = 0;

    GlobalTensor<int32_t> exactTopkIndicesGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> kvObjectSlotsSnapshotGm_;
    GlobalTensor<int32_t> cacheTokensGm_;
    GlobalTensor<int32_t> kvObjectLensGm_;
    GlobalTensor<int32_t> scoreSourceLensGm_;
    GlobalTensor<int32_t> kvObjectMapGm_;
    GlobalTensor<int32_t> privateSourceToSlotGm_;
    GlobalTensor<int32_t> privateReadyGm_;
    GlobalTensor<int32_t> kvObjectIdsGm_;
    GlobalTensor<int32_t> destinationSlotsGm_;
    GlobalTensor<int32_t> missCountsGm_;
    GlobalTensor<int32_t> placementDeltaGm_;
    GlobalTensor<int32_t> privateViewDescriptorsGm_;
    GlobalTensor<int32_t> readyPrivateKvObjectIdsGm_;
    GlobalTensor<int32_t> readyPrivateSlotsGm_;
    GlobalTensor<int32_t> readyPrivateDestinationSlotsGm_;
    GlobalTensor<int32_t> readyPrivateCountsGm_;
    GlobalTensor<int32_t> fallbackKvObjectIdsGm_;
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
    TBuf<TPosition::VECCALC> viewKvObjectBuf_;
    TBuf<TPosition::VECCALC> viewKindBuf_;
    TBuf<TPosition::VECCALC> viewPrivateSlotBuf_;
    TBuf<TPosition::VECCALC> readySourceBuf_;
    TBuf<TPosition::VECCALC> readySlotBuf_;
    TBuf<TPosition::VECCALC> readyDestinationBuf_;
    TBuf<TPosition::VECCALC> fallbackKvObjectBuf_;
    TBuf<TPosition::VECCALC> fallbackDestinationBuf_;
    TBuf<TPosition::VECCALC> seenKvObjectBuf_;
};
}  // namespace

extern "C" __global__ __aicore__ void a5_exact_reconcile_private_view(
    GM_ADDR exactTopkIndices, GM_ADDR reqPoolEntries,
    GM_ADDR kvObjectSlotsSnapshot, GM_ADDR cacheTokens, GM_ADDR kvObjectLens,
    GM_ADDR scoreSourceLens, GM_ADDR kvObjectMap,
    GM_ADDR privateSourceToSlot,
    GM_ADDR privateReady, GM_ADDR kvObjectIds, GM_ADDR destinationSlots,
    GM_ADDR missCounts, GM_ADDR placementDelta, GM_ADDR privateViewDescriptors,
    GM_ADDR readyPrivateKvObjectIds, GM_ADDR readyPrivateSlots,
    GM_ADDR readyPrivateDestinationSlots, GM_ADDR readyPrivateCounts,
    GM_ADDR fallbackKvObjectIds, GM_ADDR fallbackDestinationSlots,
    GM_ADDR fallbackCounts, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5ExactReconcilePrivateViewTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    A5ExactReconcilePrivateViewKernel op(&pipe, &tilingData);
    op.Init(exactTopkIndices, reqPoolEntries, kvObjectSlotsSnapshot, cacheTokens,
            kvObjectLens, scoreSourceLens, kvObjectMap,
            privateSourceToSlot, privateReady,
            kvObjectIds, destinationSlots, missCounts, placementDelta,
            privateViewDescriptors, readyPrivateKvObjectIds, readyPrivateSlots,
            readyPrivateDestinationSlots, readyPrivateCounts,
            fallbackKvObjectIds, fallbackDestinationSlots, fallbackCounts);
    op.Process();
}
