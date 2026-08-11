/**
 * Ascend 950 token-granular copy for GLM-5.1 packed-C8 MLA cache rows.
 *
 * The row is copied byte-for-byte so FP8 latent values, BF16 RoPE values,
 * and FP32 per-tile scales always move together.  The same launch also emits
 * the resident topK+tail index rows consumed by native A5 C8 QSFA.
 */

#include "kernel_operator.h"
#include "a5_packed_kvcache_scatter_copy_tiling.h"

namespace {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 128;
constexpr uint32_t BLOCK_SHIFT = 7;
constexpr uint32_t BLOCK_MASK = BLOCK_SIZE - 1;
constexpr uint32_t SPARSE_COUNT = 2048;
constexpr uint32_t METADATA_CHUNK = 512;

class A5PackedKvcacheScatterCopyKernel {
public:
    __aicore__ inline A5PackedKvcacheScatterCopyKernel(
        TPipe *pipe,
        const A5PackedKvcacheScatterCopyTilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR hbmKv,
        GM_ADDR dramKv,
        GM_ADDR hbmBlockTable,
        GM_ADDR dramBlockTable,
        GM_ADDR sourceTokenIds,
        GM_ADDR destinationSlots,
        GM_ADDR copyCounts,
        GM_ADDR cacheTokens,
        GM_ADDR candidateLens,
        GM_ADDR actualSeqLengthsKv,
        GM_ADDR attentionSlots,
        GM_ADDR residentSeqLengths)
    {
        coreIdx_ = GetBlockIdx();
        const uint32_t packedBufferBytes =
            (tiling_->packedRowBytes + 31U) & ~31U;
        pipe_->InitBuffer(copyQueue_, 2, packedBufferBytes);
        pipe_->InitBuffer(
            metadataBuf_, METADATA_CHUNK * sizeof(int32_t));
        hbmKvGm_.SetGlobalBuffer((__gm__ uint8_t *)hbmKv);
        dramKvGm_.SetGlobalBuffer((__gm__ uint8_t *)dramKv);
        hbmBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)hbmBlockTable);
        dramBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)dramBlockTable);
        sourceTokenIdsGm_.SetGlobalBuffer((__gm__ int32_t *)sourceTokenIds);
        destinationSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)destinationSlots);
        copyCountsGm_.SetGlobalBuffer((__gm__ int32_t *)copyCounts);
        cacheTokensGm_.SetGlobalBuffer((__gm__ int32_t *)cacheTokens);
        candidateLensGm_.SetGlobalBuffer((__gm__ int32_t *)candidateLens);
        actualSeqLengthsKvGm_.SetGlobalBuffer(
            (__gm__ int32_t *)actualSeqLengthsKv);
        attentionSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)attentionSlots);
        residentSeqLengthsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)residentSeqLengths);
    }

    __aicore__ inline void Process()
    {
        BuildAttentionMetadata();
        cachedBatch_ = static_cast<uint32_t>(-1);
        cachedCount_ = 0;
        CopyAddress currentAddress;
        uint64_t current = FindNextOwned(0, currentAddress);
        if (current >= tiling_->totalPairSlots) {
            return;
        }

        CopyIn(currentAddress);
        while (true) {
            CopyAddress nextAddress;
            const uint64_t next = FindNextOwned(current + 1, nextAddress);
            const bool hasNext = next < tiling_->totalPairSlots;
            if (hasNext) {
                CopyIn(nextAddress);
            }
            CopyOut(currentAddress);
            if (!hasNext) {
                break;
            }
            current = next;
            currentAddress = nextAddress;
        }
    }

private:
    __aicore__ inline uint32_t MinU32(uint32_t left, uint32_t right) const
    {
        return left < right ? left : right;
    }

    __aicore__ inline void StoreChunk(
        uint64_t outputOffset,
        LocalTensor<int32_t> local,
        uint32_t count)
    {
        DataCopyExtParams params{
            1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad<int32_t, PaddingMode::Normal>(
            attentionSlotsGm_[outputOffset], local, params);
    }

    __aicore__ inline void ClearAttentionRow(
        uint64_t outputBase,
        LocalTensor<int32_t> local)
    {
        Duplicate(local, static_cast<int32_t>(-1), METADATA_CHUNK);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t offset = 0; offset < tiling_->attentionCapacity;
             offset += METADATA_CHUNK) {
            const uint32_t count = MinU32(
                METADATA_CHUNK, tiling_->attentionCapacity - offset);
            StoreChunk(outputBase + offset, local, count);
        }
    }

    __aicore__ inline void CopyTopkSlots(
        uint32_t batch,
        uint64_t outputBase,
        LocalTensor<int32_t> local)
    {
        const uint64_t inputBase =
            static_cast<uint64_t>(batch) * tiling_->copyCap;
        for (uint32_t offset = 0; offset < SPARSE_COUNT;
             offset += METADATA_CHUNK) {
            const uint32_t count = MinU32(
                METADATA_CHUNK, SPARSE_COUNT - offset);
            DataCopyExtParams inputCopy{
                1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad<int32_t, PaddingMode::Normal>(
                local, destinationSlotsGm_[inputBase + offset],
                inputCopy, pad);
            PipeBarrier<PIPE_ALL>();
            StoreChunk(outputBase + offset, local, count);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void StoreSequence(
        uint64_t outputBase,
        uint32_t outputOffset,
        int32_t firstValue,
        uint32_t totalCount,
        LocalTensor<int32_t> local)
    {
        for (uint32_t done = 0; done < totalCount;
             done += METADATA_CHUNK) {
            const uint32_t count = MinU32(
                METADATA_CHUNK, totalCount - done);
            for (uint32_t index = 0; index < count; ++index) {
                local.SetValue(
                    index,
                    firstValue + static_cast<int32_t>(done + index));
            }
            PipeBarrier<PIPE_ALL>();
            StoreChunk(outputBase + outputOffset + done, local, count);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void BuildAttentionMetadata()
    {
        for (uint32_t batch = coreIdx_; batch < tiling_->batchSize;
             batch += tiling_->usedCoreNum) {
            LocalTensor<int32_t> local = metadataBuf_.Get<int32_t>();
            const uint64_t outputBase =
                static_cast<uint64_t>(batch) * tiling_->attentionCapacity;
            ClearAttentionRow(outputBase, local);
            PipeBarrier<PIPE_ALL>();

            const int32_t cacheTokens = cacheTokensGm_.GetValue(batch);
            const int32_t candidateLen = candidateLensGm_.GetValue(batch);
            const int32_t actualLen = actualSeqLengthsKvGm_.GetValue(batch);
            const int32_t tailCapacity = static_cast<int32_t>(
                tiling_->attentionCapacity - SPARSE_COUNT);
            int32_t residentLen = 0;

            if (cacheTokens >= static_cast<int32_t>(SPARSE_COUNT) &&
                candidateLen >= static_cast<int32_t>(SPARSE_COUNT) &&
                actualLen >= candidateLen &&
                actualLen - candidateLen <= tailCapacity) {
                CopyTopkSlots(batch, outputBase, local);
                const uint32_t tailCount = static_cast<uint32_t>(
                    actualLen - candidateLen);
                StoreSequence(
                    outputBase, SPARSE_COUNT, cacheTokens, tailCount, local);
                residentLen = cacheTokens + static_cast<int32_t>(tailCount);
            } else if (cacheTokens == 0 && actualLen >= 0 &&
                       actualLen <=
                           static_cast<int32_t>(tiling_->attentionCapacity)) {
                StoreSequence(
                    outputBase, 0, 0,
                    static_cast<uint32_t>(actualLen), local);
                residentLen = actualLen;
            }

            // A5 cannot reliably publish adjacent int32 scalars from
            // different AIV cores with scalar GM stores: the compact
            // resident_seq_lengths[B] cache line can be false-shared.  The
            // LIDU miss-count path uses the same MTE3 publication pattern.
            local.SetValue(0, residentLen);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyParams scalarCopy{
                1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
            DataCopyPad(
                residentSeqLengthsGm_[batch], local, scalarCopy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }

    struct CopyAddress {
        uint64_t sourceByteOffset = 0;
        uint64_t destinationByteOffset = 0;
    };

    __aicore__ inline uint32_t DestinationOwner(
        int32_t destinationSlot) const
    {
        // A packed row is 656 = 20 * 32 + 16 bytes. Therefore rows 2n and
        // 2n+1 share one 32-byte GM sector. DataCopyPad can issue a
        // read-modify-write for that partial sector, so assigning the two
        // rows to different AIV cores can lose half of either row.
        //
        // BLOCK_SIZE is even and a complete packed block is 32-byte aligned,
        // hence a logical destination row pair is also the physical
        // destination sector-sharing pair after block-table translation.
        const uint32_t destinationRowPair =
            static_cast<uint32_t>(destinationSlot) >> 1;
        return destinationRowPair % tiling_->usedCoreNum;
    }

    __aicore__ inline uint64_t FindNextOwned(
        uint64_t flatPair,
        CopyAddress &address)
    {
        while (flatPair < tiling_->totalPairSlots) {
            const uint32_t batch = static_cast<uint32_t>(
                flatPair / tiling_->copyCap);
            const uint32_t copyIndex = static_cast<uint32_t>(
                flatPair - static_cast<uint64_t>(batch) * tiling_->copyCap);
            if (batch != cachedBatch_) {
                cachedCount_ = copyCountsGm_.GetValue(batch);
                if (cachedCount_ < 0) {
                    cachedCount_ = 0;
                } else if (cachedCount_ >
                           static_cast<int32_t>(tiling_->copyCap)) {
                    cachedCount_ = static_cast<int32_t>(tiling_->copyCap);
                }
                cachedBatch_ = batch;
            }
            if (copyIndex >= static_cast<uint32_t>(cachedCount_)) {
                flatPair =
                    (static_cast<uint64_t>(batch) + 1) * tiling_->copyCap;
                continue;
            }

            const uint64_t metadataOffset =
                static_cast<uint64_t>(batch) * tiling_->copyCap + copyIndex;
            const int32_t destinationSlot =
                destinationSlotsGm_.GetValue(metadataOffset);
            if (destinationSlot >= 0 &&
                DestinationOwner(destinationSlot) == coreIdx_ &&
                Resolve(flatPair, destinationSlot, address)) {
                return flatPair;
            }
            ++flatPair;
        }
        return tiling_->totalPairSlots;
    }

    __aicore__ inline bool Resolve(
        uint64_t flatPair,
        int32_t destinationSlot,
        CopyAddress &address)
    {
        const uint32_t batch = static_cast<uint32_t>(
            flatPair / tiling_->copyCap);
        const uint32_t copyIndex = static_cast<uint32_t>(
            flatPair - static_cast<uint64_t>(batch) * tiling_->copyCap);
        const uint64_t metadataOffset =
            static_cast<uint64_t>(batch) * tiling_->copyCap + copyIndex;
        const int32_t sourceToken =
            sourceTokenIdsGm_.GetValue(metadataOffset);
        if (sourceToken < 0 || destinationSlot < 0) {
            return false;
        }

        const uint32_t sourceBlockColumn =
            static_cast<uint32_t>(sourceToken) >> BLOCK_SHIFT;
        const uint32_t destinationBlockColumn =
            static_cast<uint32_t>(destinationSlot) >> BLOCK_SHIFT;
        if (sourceBlockColumn >= tiling_->dramMaxBlockNum ||
            destinationBlockColumn >= tiling_->hbmMaxBlockNum) {
            return false;
        }
        const int32_t sourceBlock = dramBlockTableGm_.GetValue(
            static_cast<uint64_t>(batch) * tiling_->dramMaxBlockNum +
            sourceBlockColumn);
        const int32_t destinationBlock = hbmBlockTableGm_.GetValue(
            static_cast<uint64_t>(batch) * tiling_->hbmMaxBlockNum +
            destinationBlockColumn);
        if (sourceBlock < 0 || destinationBlock < 0) {
            return false;
        }

        const uint64_t sourceRow =
            static_cast<uint64_t>(sourceBlock) * BLOCK_SIZE +
            (static_cast<uint32_t>(sourceToken) & BLOCK_MASK);
        const uint64_t destinationRow =
            static_cast<uint64_t>(destinationBlock) * BLOCK_SIZE +
            (static_cast<uint32_t>(destinationSlot) & BLOCK_MASK);
        address.sourceByteOffset = sourceRow * tiling_->packedRowBytes;
        address.destinationByteOffset =
            destinationRow * tiling_->packedRowBytes;
        return true;
    }

    __aicore__ inline void CopyIn(const CopyAddress &address)
    {
        LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
        DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
        DataCopyExtParams params{1, tiling_->packedRowBytes, 0, 0, 0};
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            local, dramKvGm_[address.sourceByteOffset], params, pad);
        copyQueue_.EnQue<uint8_t>(local);
    }

    __aicore__ inline void CopyOut(const CopyAddress &address)
    {
        LocalTensor<uint8_t> local = copyQueue_.DeQue<uint8_t>();
        DataCopyExtParams params{1, tiling_->packedRowBytes, 0, 0, 0};
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            hbmKvGm_[address.destinationByteOffset], local, params);
        copyQueue_.FreeTensor(local);
    }

private:
    TPipe *pipe_;
    const A5PackedKvcacheScatterCopyTilingData *tiling_;
    uint32_t coreIdx_ = 0;
    uint32_t cachedBatch_ = static_cast<uint32_t>(-1);
    int32_t cachedCount_ = 0;
    GlobalTensor<uint8_t> hbmKvGm_;
    GlobalTensor<uint8_t> dramKvGm_;
    GlobalTensor<int32_t> hbmBlockTableGm_;
    GlobalTensor<int32_t> dramBlockTableGm_;
    GlobalTensor<int32_t> sourceTokenIdsGm_;
    GlobalTensor<int32_t> destinationSlotsGm_;
    GlobalTensor<int32_t> copyCountsGm_;
    GlobalTensor<int32_t> cacheTokensGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> actualSeqLengthsKvGm_;
    GlobalTensor<int32_t> attentionSlotsGm_;
    GlobalTensor<int32_t> residentSeqLengthsGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
    TBuf<TPosition::VECCALC> metadataBuf_;
};
} // namespace

extern "C" __global__ __aicore__ void a5_packed_kvcache_scatter_copy(
    GM_ADDR hbmKv,
    GM_ADDR dramKv,
    GM_ADDR hbmBlockTable,
    GM_ADDR dramBlockTable,
    GM_ADDR sourceTokenIds,
    GM_ADDR destinationSlots,
    GM_ADDR copyCounts,
    GM_ADDR cacheTokens,
    GM_ADDR candidateLens,
    GM_ADDR actualSeqLengthsKv,
    GM_ADDR attentionSlotsBuffer,
    GM_ADDR residentSeqLengthsBuffer,
    GM_ADDR hbmKvOut,
    GM_ADDR attentionSlotsOut,
    GM_ADDR residentSeqLengthsOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)attentionSlotsBuffer;
    (void)residentSeqLengthsBuffer;
    (void)hbmKvOut;
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5PackedKvcacheScatterCopyTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    A5PackedKvcacheScatterCopyKernel op(&pipe, &tilingData);
    op.Init(
        hbmKv, dramKv, hbmBlockTable, dramBlockTable,
        sourceTokenIds, destinationSlots, copyCounts,
        cacheTokens, candidateLens, actualSeqLengthsKv,
        attentionSlotsOut, residentSeqLengthsOut);
    op.Process();
}
