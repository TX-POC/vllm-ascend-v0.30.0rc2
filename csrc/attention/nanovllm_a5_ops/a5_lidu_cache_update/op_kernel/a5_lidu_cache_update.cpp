/**
 * Request-pool update stage for the Ascend 950 C8 LightningIndexer path.
 *
 * Native A5 Quant LightningIndexer produces the exact C8 top-2048 set. This
 * AIV kernel classifies it into a miss prefix and hit suffix, selects one
 * unprotected resident slot per miss, updates token->slot request state, and
 * publishes the complete HBM slot row consumed by sparse attention.
 */

#include "kernel_operator.h"
#include "a5_lidu_cache_update_tiling.h"

namespace {
using namespace AscendC;

constexpr uint32_t SPARSE_COUNT = 2048;
constexpr uint32_t MAX_CACHE_TOKENS = 16256;
constexpr uint32_t CACHE_CHUNK = 2048;

class A5LiduCacheUpdateKernel {
public:
    __aicore__ inline A5LiduCacheUpdateKernel(
        TPipe *pipe,
        const A5LiduCacheUpdateTilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR topkIndices,
        GM_ADDR reqPoolEntries,
        GM_ADDR cacheSlotsPool,
        GM_ADDR cacheTokens,
        GM_ADDR candidateLens,
        GM_ADDR sourceIds,
        GM_ADDR destinationSlots,
        GM_ADDR missCounts)
    {
        coreIdx_ = GetBlockIdx();
        topkIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)topkIndices);
        reqPoolEntriesGm_.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntries);
        cacheSlotsPoolGm_.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsPool);
        cacheTokensGm_.SetGlobalBuffer((__gm__ int32_t *)cacheTokens);
        candidateLensGm_.SetGlobalBuffer((__gm__ int32_t *)candidateLens);
        sourceIdsGm_.SetGlobalBuffer((__gm__ int32_t *)sourceIds);
        destinationSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)destinationSlots);
        missCountsGm_.SetGlobalBuffer((__gm__ int32_t *)missCounts);

        pipe_->InitBuffer(outputTokensBuf_, SPARSE_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(outputSlotsBuf_, SPARSE_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(missTokensBuf_, SPARSE_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(hitTokensBuf_, SPARSE_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(hitSlotsBuf_, SPARSE_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(cacheChunkBuf_, CACHE_CHUNK * sizeof(int32_t));
        pipe_->InitBuffer(protectedSlotsBuf_, MAX_CACHE_TOKENS * sizeof(uint8_t));
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

    __aicore__ inline void ClearOutputs(
        LocalTensor<int32_t> outputTokens,
        LocalTensor<int32_t> outputSlots)
    {
        Duplicate(outputTokens, static_cast<int32_t>(-1), SPARSE_COUNT);
        Duplicate(outputSlots, static_cast<int32_t>(-1), SPARSE_COUNT);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    __aicore__ inline void LoadTopk(
        uint32_t batch,
        LocalTensor<int32_t> outputTokens)
    {
        DataCopyExtParams copy{
            1, SPARSE_COUNT * sizeof(int32_t), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad<int32_t, PaddingMode::Normal>(
            outputTokens,
            topkIndicesGm_[static_cast<uint64_t>(batch) * SPARSE_COUNT],
            copy,
            pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
    }

    __aicore__ inline void StoreOutputs(
        uint32_t batch,
        LocalTensor<int32_t> outputTokens,
        LocalTensor<int32_t> outputSlots,
        int32_t missCount)
    {
        DataCopyExtParams copy{
            1, SPARSE_COUNT * sizeof(int32_t), 0, 0, 0};
        const uint64_t outputBase =
            static_cast<uint64_t>(batch) * SPARSE_COUNT;
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        DataCopyPad<int32_t, PaddingMode::Normal>(
            sourceIdsGm_[outputBase], outputTokens, copy);
        DataCopyPad<int32_t, PaddingMode::Normal>(
            destinationSlotsGm_[outputBase], outputSlots, copy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);

        // On A5, scalar GM stores from multiple AIV cores can false-share the
        // compact miss_counts[B] cache line.  Match the official Lightning
        // Indexer path and publish the scalar through MTE3 instead.
        outputTokens.SetValue(0, missCount);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
        DataCopyParams scalarCopy{
            1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
        DataCopyPad(missCountsGm_[batch], outputTokens, scalarCopy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);
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

        ClearOutputs(outputTokens, outputSlots);
        const int32_t budgetValue = cacheTokensGm_.GetValue(batch);
        const int32_t candidateValue = candidateLensGm_.GetValue(batch);
        const int32_t poolRowValue = reqPoolEntriesGm_.GetValue(batch);
        if (budgetValue == 0) {
            StoreOutputs(batch, outputTokens, outputSlots, 0);
            return;
        }
        if (budgetValue < static_cast<int32_t>(SPARSE_COUNT) ||
            budgetValue > static_cast<int32_t>(MAX_CACHE_TOKENS) ||
            candidateValue < static_cast<int32_t>(SPARSE_COUNT) ||
            candidateValue > static_cast<int32_t>(tiling_->sourceCapacity) ||
            poolRowValue < 0 ||
            poolRowValue >= static_cast<int32_t>(tiling_->poolSize)) {
            StoreOutputs(batch, outputTokens, outputSlots, -1);
            return;
        }

        const uint32_t budget = static_cast<uint32_t>(budgetValue);
        const uint32_t candidateLen = static_cast<uint32_t>(candidateValue);
        const uint64_t cacheBase =
            static_cast<uint64_t>(poolRowValue) * tiling_->sourceCapacity;

        Duplicate(protectedSlots, static_cast<uint8_t>(0), budget);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        LoadTopk(batch, outputTokens);

        uint32_t missCount = 0;
        uint32_t hitCount = 0;
        for (uint32_t index = 0; index < SPARSE_COUNT; ++index) {
            const int32_t token = outputTokens.GetValue(index);
            int32_t slot = -1;
            if (token >= 0 && token < candidateValue) {
                slot = cacheSlotsPoolGm_.GetValue(
                    cacheBase + static_cast<uint32_t>(token));
            }
            if (slot >= 0 && slot < budgetValue) {
                hitTokens.SetValue(hitCount, token);
                hitSlots.SetValue(hitCount, slot);
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 1);
                ++hitCount;
            } else {
                missTokens.SetValue(missCount, token);
                ++missCount;
            }
        }

        uint32_t assigned = 0;
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        for (uint32_t chunkBase = 0;
             chunkBase < candidateLen && assigned < missCount;
             chunkBase += CACHE_CHUNK) {
            const uint32_t chunkLen = MinU32(
                CACHE_CHUNK, candidateLen - chunkBase);
            DataCopyExtParams copy{
                1,
                static_cast<uint32_t>(chunkLen * sizeof(int32_t)),
                0,
                0,
                0};
            DataCopyPad<int32_t, PaddingMode::Normal>(
                cacheChunk,
                cacheSlotsPoolGm_[cacheBase + chunkBase],
                copy,
                pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            for (uint32_t offset = 0;
                 offset < chunkLen && assigned < missCount;
                 ++offset) {
                const int32_t slot = cacheChunk.GetValue(offset);
                if (slot < 0 || slot >= budgetValue ||
                    protectedSlots.GetValue(static_cast<uint32_t>(slot)) != 0) {
                    continue;
                }
                const int32_t victimToken =
                    static_cast<int32_t>(chunkBase + offset);
                const int32_t missToken = missTokens.GetValue(assigned);
                if (missToken < 0 || missToken >= candidateValue) {
                    continue;
                }
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 2);
                cacheSlotsPoolGm_.SetValue(
                    cacheBase + static_cast<uint32_t>(victimToken), -1);
                cacheSlotsPoolGm_.SetValue(
                    cacheBase + static_cast<uint32_t>(missToken), slot);
                outputTokens.SetValue(assigned, missToken);
                outputSlots.SetValue(assigned, slot);
                ++assigned;
            }
        }

        if (assigned != missCount || missCount + hitCount != SPARSE_COUNT) {
            StoreOutputs(batch, outputTokens, outputSlots, -1);
            return;
        }
        for (uint32_t index = 0; index < hitCount; ++index) {
            outputTokens.SetValue(missCount + index, hitTokens.GetValue(index));
            outputSlots.SetValue(missCount + index, hitSlots.GetValue(index));
        }
        StoreOutputs(
            batch, outputTokens, outputSlots,
            static_cast<int32_t>(missCount));
    }

private:
    TPipe *pipe_;
    const A5LiduCacheUpdateTilingData *tiling_;
    uint32_t coreIdx_ = 0;
    GlobalTensor<int32_t> topkIndicesGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> cacheSlotsPoolGm_;
    GlobalTensor<int32_t> cacheTokensGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> sourceIdsGm_;
    GlobalTensor<int32_t> destinationSlotsGm_;
    GlobalTensor<int32_t> missCountsGm_;
    TBuf<TPosition::VECCALC> outputTokensBuf_;
    TBuf<TPosition::VECCALC> outputSlotsBuf_;
    TBuf<TPosition::VECCALC> missTokensBuf_;
    TBuf<TPosition::VECCALC> hitTokensBuf_;
    TBuf<TPosition::VECCALC> hitSlotsBuf_;
    TBuf<TPosition::VECCALC> cacheChunkBuf_;
    TBuf<TPosition::VECCALC> protectedSlotsBuf_;
};
} // namespace

extern "C" __global__ __aicore__ void a5_lidu_cache_update(
    GM_ADDR topkIndices,
    GM_ADDR reqPoolEntries,
    GM_ADDR cacheSlotsPool,
    GM_ADDR cacheTokens,
    GM_ADDR candidateLens,
    GM_ADDR sourceIds,
    GM_ADDR destinationSlots,
    GM_ADDR missCounts,
    GM_ADDR cacheSlotsAlias,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)cacheSlotsAlias;
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5LiduCacheUpdateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    A5LiduCacheUpdateKernel op(&pipe, &tilingData);
    op.Init(
        topkIndices, reqPoolEntries, cacheSlotsPool, cacheTokens,
        candidateLens, sourceIds, destinationSlots, missCounts);
    op.Process();
}
