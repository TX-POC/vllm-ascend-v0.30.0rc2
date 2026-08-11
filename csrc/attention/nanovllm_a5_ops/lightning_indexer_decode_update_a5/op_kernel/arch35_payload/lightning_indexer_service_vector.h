/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_service_vector.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_SERVICE_VECTOR_H
#define LIGHTNING_INDEXER_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_common.h"
#include "vf/lightning_indexer_vector1.h"
#include "payload/hist_topk_index_update_a5_topk.h"
#include "payload/hist_topk_index_update_a5_evict_vf.h"
#include "../lightning_indexer_decode_update_a5_classify_vf.h"

namespace LIKernel {
using namespace LICommon;
constexpr uint32_t TRUNK_LEN_16K = 16384;
constexpr uint32_t VICTIM_SCAN_CHUNK = 2048;

template<typename Q_T, typename W_T = void>
struct LightningIndexerTypeTraits {
    using weightsType = Q_T;   // 默认：weightsType绑定Q_T
};

template<typename Q_T>
struct LightningIndexerTypeTraits<Q_T, float> {
    using weightsType = float;  // W_T=float时，强制weightsType为float
};
template <typename LIT>
class LightningIndexerServiceVector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT LAYOUT_T = LIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = LIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using W_T = typename LightningIndexerTypeTraits<Q_T,
                                                typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;

    __aicore__ inline LightningIndexerServiceVector(){};
    __aicore__ inline void ProcessVec1(const LICommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const LICommon::RunInfo &info,
                                       bool allowFullSlotPrefetch);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct LICommon::ConstInfo &constInfo,
                                      const LIA5TilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<uint16_t> scoreGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<K_T> valueOutGm, GlobalTensor<int32_t> blockTableGm,
                                              GlobalTensor<int32_t> cacheSlotsGm,
                                              GlobalTensor<int32_t> topkSlotsGm,
                                              GlobalTensor<int32_t> missCountGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void FinalizePayloadUpdate(const LICommon::RunInfo &info,
                                                 uint32_t outputRow,
                                                 uint64_t scoreRowOffset,
                                                 uint32_t validS2Len);
    __aicore__ inline uint32_t FindVictimsOnDemand(
        const LICommon::RunInfo &info, uint64_t scoreRowOffset,
        uint32_t validS2Len, uint32_t requiredCount, uint16_t kthValue,
        const LocalTensor<uint32_t>& compactPayloadLocal);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

protected:
    GlobalTensor<uint16_t> scoreGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<K_T> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> topkSlotsGm;
    GlobalTensor<int32_t> missCountGm;
    // =================================常量区=================================
    static constexpr uint32_t VEC1_V_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;

    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;

private:
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<float> resMm1UB_;
    // tmp buff for weight
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<W_T> weightUB_;
    // tmp buff for weight cast float
    TBuf<TPosition::VECCALC> weightFloatBuf_;
    LocalTensor<float> weightFloatUB_;

    // tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<uint16_t> vec1OutUB_;

    // tmp buff for returnValue K_T
    TBuf<TPosition::VECCALC> valueOutBuf_;
    LocalTensor<K_T> valueOutLocal_;

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<uint16_t> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<uint16_t> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;

    // One 2048-entry cache_slots DMA stage shared by survivor payload
    // construction, non-final-request prefix prefetch, and victim scanning.
    TBuf<TPosition::VECCALC> slotStageBuf_;
    LocalTensor<int32_t> slotStageLocal_;

    TBuf<TPosition::VECCALC> candidatePayloadBuf_;
    LocalTensor<uint32_t> candidatePayloadLocal_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;
    uint32_t topkCount_ = 0;
    uint32_t topkCountAlign256_ = 0; // topkCount对齐到256(直方图需要)，支持topk泛化
    uint32_t trunkLen_ = 0;
    bool returnValueFlag = false;
    static constexpr uint32_t EVICT_CANDIDATE_CAP = 2048;

    struct LICommon::ConstInfo constInfo_;
    hist_topk_index_update_a5_payload::LITopk<uint16_t> topkOp_;
};

#if 0
template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::UpdateEvictCandidateCache(
    uint32_t tokenBase, uint32_t scoreLocalBase, uint32_t validLen,
    uint32_t loopIndex, uint32_t loopCount)
{
    (void)loopIndex;
    (void)loopCount;
    if (candidateCount_ >= EVICT_CANDIDATE_CAP) {
        return;
    }

    uint32_t alignedLen = LICommon::Align(
        validLen, LightningIndexerPayloadEvictVF::VF_B16_LANES);
    if (alignedLen > validLen) {
        for (uint32_t i = validLen; i < alignedLen; ++i) {
            currentSlotsLocal_.SetValue(i, static_cast<uint16_t>(0xffff));
        }
        SetFlag<HardEvent::S_V>(V_MTE2_EVENT3);
        WaitFlag<HardEvent::S_V>(V_MTE2_EVENT3);
    }

    // The kth boundary is monotonic as histogram stages are merged. A cached
    // token strictly below the current boundary can never become a final TopK
    // survivor, so retain its {slot, token} payload directly as an eviction
    // candidate. This avoids the former 96-KiB reservoir in resMm1UB_.
    LightningIndexerPayloadEvictVF::CompactEligiblePayloads(
        (__ubuf__ uint32_t *)compactPayloadLocal_.GetPhyAddr(),
        (__ubuf__ uint16_t *)mrgValueLocal_[scoreLocalBase].GetPhyAddr(),
        (__ubuf__ uint16_t *)currentSlotsLocal_.GetPhyAddr(),
        topkOp_.GetLastKthValue(), tokenBase,
        LICommon::Align(validLen, static_cast<uint32_t>(64)) / 64);
    uint32_t compactCount = static_cast<uint32_t>(
        AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>() / sizeof(uint32_t));
    PipeBarrier<PIPE_V>();
    uint32_t keepCount = compactCount;
    uint32_t remaining = EVICT_CANDIDATE_CAP - candidateCount_;
    if (keepCount > remaining) {
        keepCount = remaining;
    }
    SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    for (uint32_t i = 0; i < keepCount; ++i) {
        candidatePayloadLocal_.SetValue(candidateCount_ + i,
                                        compactPayloadLocal_.GetValue(i));
    }
    candidateCount_ += keepCount;
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::RunLargeMissEviction(
    const LICommon::RunInfo &info, uint32_t validS2Len,
    uint32_t currentMissCount,
    LocalTensor<int32_t> classifiedIndex,
    LocalTensor<int32_t> classifiedSlots,
    LocalTensor<uint32_t> selectedPayload)
{
    constexpr uint32_t CACHE_CHUNK_SHIFT = 11;
    constexpr uint32_t MAX_CACHE_CHUNKS = (1U << 18) >> CACHE_CHUNK_SHIFT;
    constexpr uint32_t CLEAR_SLOT =
        hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_SLOT_MASK;
    const uint64_t cacheBase =
        static_cast<uint64_t>(info.cacheRowIdx) * constInfo_.cacheSlotsSize;
    LocalTensor<uint32_t> updatePayload =
        topkSharedTmpLocal_.template ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> orderedUpdates =
        resMm1UB_.template ReinterpretCast<uint32_t>();
    LocalTensor<uint16_t> bucketCounts = scoreOutLocal_;
    LocalTensor<uint16_t> bucketCursor = scoreOutLocal_[MAX_CACHE_CHUNKS];
    LocalTensor<int32_t> cacheChunk =
        currentSlotsLocal_.template ReinterpretCast<int32_t>();

    // The victim payloads already came from the first histogram scan. This
    // method performs only A5-safe, cache-chunk-bucketed writeback.
    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> cachePadParams{false, 0, 0, 0};
    AscendC::DataCopyParams cacheWriteParams{1, 0, 0, 0};

    // updatePayload aliases classifiedIndex (topkSharedTmpLocal_), while
    // cacheChunk aliases classifiedSlots (currentSlotsLocal_). Preserve both
    // classified outputs before either scratch buffer is reused. The selected
    // victim payloads have already been compacted into compactPayloadLocal_, so
    // candidatePayloadLocal_ is free to hold the slot snapshot here.
    DataCopy(indicesOutLocal_, classifiedIndex.ReinterpretCast<uint32_t>(),
             topkCountAlign256_);
    DataCopy(candidatePayloadLocal_, classifiedSlots.ReinterpretCast<uint32_t>(),
             topkCountAlign256_);
    DataCopy(valueOutLocal_.template ReinterpretCast<uint16_t>(),
             scoreOutLocal_, topkCountAlign256_);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);

    uint32_t cacheChunkCount =
        (validS2Len + topkCount_ - 1) >> CACHE_CHUNK_SHIFT;
    for (uint32_t chunk = 0; chunk < cacheChunkCount; ++chunk) {
        bucketCounts.SetValue(chunk, static_cast<uint16_t>(0));
    }

    const uint32_t updateCount = currentMissCount * 2;
    for (uint32_t i = 0; i < currentMissCount; ++i) {
        uint32_t victimPayload = selectedPayload.GetValue(i);
        uint32_t slot =
            (victimPayload >> hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_SLOT_SHIFT) &
            hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_SLOT_MASK;
        uint32_t evictToken = victimPayload &
            hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_TOKEN_MASK;
        // classifiedIndex is being overwritten by updatePayload. Read the
        // immutable token snapshot instead of the aliased destination.
        uint32_t missToken = indicesOutLocal_.GetValue(i);
        uint32_t clearPayload =
            (CLEAR_SLOT << hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_SLOT_SHIFT) |
            evictToken;
        uint32_t assignPayload =
            (slot << hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_SLOT_SHIFT) |
            missToken;
        updatePayload.SetValue(i * 2, clearPayload);
        updatePayload.SetValue(i * 2 + 1, assignPayload);
        bucketCounts.SetValue(evictToken >> CACHE_CHUNK_SHIFT,
            static_cast<uint16_t>(bucketCounts.GetValue(evictToken >> CACHE_CHUNK_SHIFT) + 1));
        bucketCounts.SetValue(missToken >> CACHE_CHUNK_SHIFT,
            static_cast<uint16_t>(bucketCounts.GetValue(missToken >> CACHE_CHUNK_SHIFT) + 1));
        // classifiedSlots is reused below as the cache chunk DMA buffer.
        // Stage assigned slots until the classified output is restored.
        slotStageLocal_.SetValue(i, static_cast<int32_t>(slot));
    }

    uint32_t prefix = 0;
    for (uint32_t chunk = 0; chunk < cacheChunkCount; ++chunk) {
        bucketCursor.SetValue(chunk, static_cast<uint16_t>(prefix));
        prefix += bucketCounts.GetValue(chunk);
    }
    for (uint32_t i = 0; i < updateCount; ++i) {
        uint32_t payload = updatePayload.GetValue(i);
        uint32_t token = payload &
            hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_TOKEN_MASK;
        uint32_t chunk = token >> CACHE_CHUNK_SHIFT;
        uint32_t dst = bucketCursor.GetValue(chunk);
        orderedUpdates.SetValue(dst, payload);
        bucketCursor.SetValue(chunk, static_cast<uint16_t>(dst + 1));
    }

    uint32_t updateBase = 0;
    for (uint32_t chunk = 0; chunk < cacheChunkCount; ++chunk) {
        uint32_t count = bucketCounts.GetValue(chunk);
        if (count == 0) {
            continue;
        }
        uint32_t chunkBase = chunk << CACHE_CHUNK_SHIFT;
        uint32_t chunkLen = Min(topkCount_, validS2Len - chunkBase);
        copyParams.blockLen = chunkLen * sizeof(int32_t);
        DataCopyPad(cacheChunk, cacheSlotsGm[cacheBase + chunkBase],
                    copyParams, cachePadParams);
        SetFlag<HardEvent::MTE2_S>(V_MTE2_EVENT1);
        WaitFlag<HardEvent::MTE2_S>(V_MTE2_EVENT1);
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t payload = orderedUpdates.GetValue(updateBase + i);
            uint32_t token = payload &
                hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_TOKEN_MASK;
            uint32_t slot =
                (payload >> hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_SLOT_SHIFT) &
                hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_SLOT_MASK;
            cacheChunk.SetValue(token - chunkBase,
                slot == CLEAR_SLOT ? -1 : static_cast<int32_t>(slot));
        }
        updateBase += count;
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        cacheWriteParams.blockLen =
            static_cast<uint16_t>(chunkLen * sizeof(int32_t));
        DataCopyPad(cacheSlotsGm[cacheBase + chunkBase], cacheChunk,
                    cacheWriteParams);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
    }
    DataCopy(scoreOutLocal_,
             valueOutLocal_.template ReinterpretCast<uint16_t>(),
             topkCountAlign256_);
    DataCopy(classifiedIndex.ReinterpretCast<uint32_t>(), indicesOutLocal_,
             topkCountAlign256_);
    DataCopy(classifiedSlots.ReinterpretCast<uint32_t>(), candidatePayloadLocal_,
             topkCountAlign256_);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    for (uint32_t i = 0; i < currentMissCount; ++i) {
        classifiedSlots.SetValue(i, slotStageLocal_.GetValue(i));
    }
    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
}
#endif

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerServiceVector<LIT>::FindVictimsOnDemand(
    const LICommon::RunInfo &info, uint64_t scoreRowOffset,
    uint32_t validS2Len, uint32_t requiredCount, uint16_t kthValue,
    const LocalTensor<uint32_t>& compactPayloadLocal)
{
    const uint64_t cacheBase =
        static_cast<uint64_t>(info.cacheRowIdx) * constInfo_.cacheSlotsSize;
    LocalTensor<uint16_t> scanScoresLocal = mrgValueLocal_;
    LocalTensor<int16_t> scanSlotsLocal =
        mrgValueLocal_[VICTIM_SCAN_CHUNK].template ReinterpretCast<int16_t>();
    uint32_t candidateCount = 0;

    AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint16_t> scorePadParams{true, 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> slotPadParams{true, 0, 0, 0};

    for (uint32_t chunkBase = 0;
         chunkBase < validS2Len && candidateCount < requiredCount;
         chunkBase += VICTIM_SCAN_CHUNK) {
        uint32_t chunkLen =
            chunkBase + VICTIM_SCAN_CHUNK > validS2Len
                ? validS2Len - chunkBase
                : VICTIM_SCAN_CHUNK;
        uint32_t alignedLen = LICommon::Align(chunkLen, static_cast<uint32_t>(64));

        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        copyParams.blockLen = chunkLen * sizeof(uint16_t);
        AscendC::DataCopyPad(
            scanScoresLocal, scoreGm[scoreRowOffset + chunkBase],
            copyParams, scorePadParams);
        copyParams.blockLen = chunkLen * sizeof(int32_t);
        AscendC::DataCopyPad(
            slotStageLocal_, cacheSlotsGm[cacheBase + chunkBase],
            copyParams, slotPadParams);
        SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);

        Cast(scanSlotsLocal, slotStageLocal_, RoundMode::CAST_NONE, chunkLen);
        if (alignedLen > chunkLen) {
            Duplicate(scanSlotsLocal[chunkLen], static_cast<int16_t>(-1),
                      alignedLen - chunkLen);
        }
        PipeBarrier<PIPE_V>();

        LightningIndexerPayloadEvictVF::CompactEligiblePayloads(
            (__ubuf__ uint32_t *)compactPayloadLocal.GetPhyAddr(),
            (__ubuf__ uint16_t *)scanScoresLocal.GetPhyAddr(),
            (__ubuf__ uint16_t *)scanSlotsLocal.GetPhyAddr(),
            kthValue, chunkBase, alignedLen / 64);
        uint32_t compactCount = static_cast<uint32_t>(
            AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>() /
            sizeof(uint32_t));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
        WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);

        uint32_t keepCount = compactCount;
        uint32_t remaining = requiredCount - candidateCount;
        if (keepCount > remaining) {
            keepCount = remaining;
        }
        for (uint32_t i = 0; i < keepCount; ++i) {
            candidatePayloadLocal_.SetValue(
                candidateCount + i, compactPayloadLocal.GetValue(i));
        }
        candidateCount += keepCount;
    }

    if (candidateCount < requiredCount) {
        // Rare tie fallback. Build an exact open-addressed set of the final
        // TopK token IDs, then select any cached token outside that set. This
        // keeps the normal path bitmap-free and guarantees C - T semantics.
        constexpr uint32_t HASH_CAPACITY = 4096;
        constexpr uint32_t HASH_MASK = HASH_CAPACITY - 1;
        constexpr uint32_t TOKEN_MASK =
            hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_TOKEN_MASK;
        LocalTensor<uint32_t> topkHash =
            mrgValueLocal_.template ReinterpretCast<uint32_t>();
        Duplicate(topkHash.template ReinterpretCast<int32_t>(),
                  static_cast<int32_t>(-1), HASH_CAPACITY);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
        WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);

        for (uint32_t i = 0; i < topkCount_; ++i) {
            uint32_t token = indicesOutLocal_.GetValue(i) & TOKEN_MASK;
            uint32_t hashPos = (token * 2654435761U) & HASH_MASK;
            while (topkHash.GetValue(hashPos) != 0xffffffffU &&
                   topkHash.GetValue(hashPos) != token) {
                hashPos = (hashPos + 1) & HASH_MASK;
            }
            topkHash.SetValue(hashPos, token);
        }

        candidateCount = 0;
        for (uint32_t chunkBase = 0;
             chunkBase < validS2Len && candidateCount < requiredCount;
             chunkBase += VICTIM_SCAN_CHUNK) {
            uint32_t chunkLen =
                chunkBase + VICTIM_SCAN_CHUNK > validS2Len
                    ? validS2Len - chunkBase
                    : VICTIM_SCAN_CHUNK;
            copyParams.blockLen = chunkLen * sizeof(int32_t);
            AscendC::DataCopyPad(
                slotStageLocal_, cacheSlotsGm[cacheBase + chunkBase],
                copyParams, slotPadParams);
            SetFlag<HardEvent::MTE2_S>(V_MTE2_EVENT1);
            WaitFlag<HardEvent::MTE2_S>(V_MTE2_EVENT1);

            for (uint32_t i = 0;
                 i < chunkLen && candidateCount < requiredCount; ++i) {
                int32_t slot = slotStageLocal_.GetValue(i);
                if (slot < 0) {
                    continue;
                }
                uint32_t token = chunkBase + i;
                uint32_t hashPos = (token * 2654435761U) & HASH_MASK;
                while (topkHash.GetValue(hashPos) != 0xffffffffU &&
                       topkHash.GetValue(hashPos) != token) {
                    hashPos = (hashPos + 1) & HASH_MASK;
                }
                if (topkHash.GetValue(hashPos) == token) {
                    continue;
                }
                candidatePayloadLocal_.SetValue(
                    candidateCount,
                    (static_cast<uint32_t>(slot) <<
                     hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_SLOT_SHIFT) |
                    token);
                ++candidateCount;
            }
        }
    }
    return candidateCount;
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::FinalizePayloadUpdate(
    const LICommon::RunInfo &info, uint32_t outputRow,
    uint64_t scoreRowOffset, uint32_t validS2Len)
{
    constexpr uint32_t CLASSIFY_CHUNK = TopkIndexerClassifyVF::CHUNK_SIZE;
    LocalTensor<int32_t> classifiedIndex = topkSharedTmpLocal_.template ReinterpretCast<int32_t>();
    LocalTensor<uint32_t> compactPayloadLocal =
        topkSharedTmpLocal_[topkCountAlign256_];

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    if (info.cacheTokenCount == 0) {
        Duplicate(classifiedIndex, static_cast<int32_t>(-1), topkCount_);
        Duplicate(slotStageLocal_, static_cast<int32_t>(-1), topkCount_);
        LocalTensor<int32_t> missCountLocal =
            candidatePayloadLocal_.template ReinterpretCast<int32_t>();
        missCountLocal.SetValue(0, 0);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        AscendC::DataCopyParams scalarCopy{
            1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
        DataCopyPad(missCountGm[info.bIdx], missCountLocal, scalarCopy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyParams outputCopy{
            1, static_cast<uint16_t>(topkCount_ * sizeof(int32_t)), 0, 0};
        uint64_t outOffset = static_cast<uint64_t>(outputRow) * topkCount_;
        DataCopyPad(indiceOutGm[outOffset], classifiedIndex, outputCopy);
        DataCopyPad(topkSlotsGm[outOffset], slotStageLocal_, outputCopy);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        return;
    }
    SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    uint16_t kthValue = topkOp_.GetLastKthValue();
    TopkIndexerClassifyVF::SqueezeIndexerMissTokenIds(
        (__ubuf__ uint32_t *)classifiedIndex.GetPhyAddr(),
        (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(), topkCount_ / CLASSIFY_CHUNK);
    uint32_t currentMissCount = static_cast<uint32_t>(
        AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>() / sizeof(uint32_t));
    PipeBarrier<PIPE_V>();
    TopkIndexerClassifyVF::SqueezeIndexerHitTokenIds(
        (__ubuf__ uint32_t *)classifiedIndex[currentMissCount].GetPhyAddr(),
        (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(), topkCount_ / CLASSIFY_CHUNK);
    PipeBarrier<PIPE_V>();

    uint32_t candidateCount = 0;
#if LI_UPDATE_ABLATION_MODE >= 4
    if (currentMissCount > 0) {
        candidateCount = FindVictimsOnDemand(
            info, scoreRowOffset, validS2Len, currentMissCount,
            kthValue, compactPayloadLocal);
    }
#endif

    Duplicate(slotStageLocal_, static_cast<int32_t>(-1), topkCount_);
    PipeBarrier<PIPE_V>();
    TopkIndexerClassifyVF::SqueezeIndexerHitSlots(
        (__ubuf__ uint32_t *)slotStageLocal_[currentMissCount].GetPhyAddr(),
        (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(), topkCount_ / CLASSIFY_CHUNK);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);

    const uint64_t cacheBase =
        static_cast<uint64_t>(info.cacheRowIdx) * constInfo_.cacheSlotsSize;
    if (currentMissCount > 0 && candidateCount >= currentMissCount) {
        for (uint32_t i = 0; i < currentMissCount; ++i) {
            uint32_t payload = candidatePayloadLocal_.GetValue(i);
            uint32_t slot =
                (payload >> hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_SLOT_SHIFT) &
                hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_SLOT_MASK;
            uint32_t evictToken = payload &
                hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_TOKEN_MASK;
            uint32_t missToken =
                static_cast<uint32_t>(classifiedIndex.GetValue(i));
            slotStageLocal_.SetValue(i, static_cast<int32_t>(slot));
#if LI_UPDATE_ABLATION_MODE >= 5
            cacheSlotsGm.SetValue(cacheBase + evictToken, -1);
            cacheSlotsGm.SetValue(cacheBase + missToken,
                                  static_cast<int32_t>(slot));
#else
            (void)evictToken;
            (void)missToken;
#endif
        }
#if LI_UPDATE_ABLATION_MODE >= 5
        PipeBarrier<PIPE_ALL>();
#endif
    }

    LocalTensor<int32_t> missCountLocal =
        candidatePayloadLocal_.template ReinterpretCast<int32_t>();
    missCountLocal.SetValue(0, static_cast<int32_t>(currentMissCount));
    SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
    AscendC::DataCopyParams scalarCopy{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
    DataCopyPad(missCountGm[info.bIdx], missCountLocal, scalarCopy);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);

    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    AscendC::DataCopyParams outputCopy{
        1, static_cast<uint16_t>(topkCount_ * sizeof(int32_t)), 0, 0};
    uint64_t outOffset = static_cast<uint64_t>(outputRow) * topkCount_;
    DataCopyPad(indiceOutGm[outOffset], classifiedIndex, outputCopy);
    DataCopyPad(topkSlotsGm[outOffset], slotStageLocal_, outputCopy);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::InitBuffers(TPipe *pipe)
{
    uint64_t resMm1BufferSize =
        2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(float);
    pipe->InitBuffer(resMm1Buf_, resMm1BufferSize);
    resMm1UB_ = resMm1Buf_.Get<float>();

    pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightUB_ = weightBuf_.Get<W_T>();
    pipe->InitBuffer(weightFloatBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightFloatUB_ = weightFloatBuf_.Get<float>();
    pipe->InitBuffer(outBuf_,
                    2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(uint16_t));      // 大小：2(开dB) * 2 * 128 * 4 = 2KB
    vec1OutUB_ = outBuf_.Get<uint16_t>(); // out

    // Topk
    pipe->InitBuffer(mrgValueBuf_,
                    (topkCountAlign256_ + trunkLen_) * sizeof(uint16_t));
    mrgValueLocal_ = mrgValueBuf_.Get<uint16_t>();
    // returnvalue
    if (topkCount_ <= 2048) {
        pipe->InitBuffer(valueOutBuf_, topkCountAlign256_ * sizeof(K_T));
        valueOutLocal_ = valueOutBuf_.Get<K_T>();
    } else { // sparseCount > 2k时，复用return value相关UB
        valueOutLocal_ = mrgValueBuf_.Get<K_T>(); // returnValue float
    }

    // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    pipe->InitBuffer(indicesOutBuf_,
                    (topkCountAlign256_ + 64) * sizeof(uint32_t));
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    pipe->InitBuffer(scoreOutBuf_, topkCountAlign256_ * sizeof(uint16_t));
    scoreOutLocal_ = scoreOutBuf_.Get<uint16_t>();

    pipe->InitBuffer(slotStageBuf_, topkCount_ * sizeof(int32_t));
    slotStageLocal_ = slotStageBuf_.Get<int32_t>();

    pipe->InitBuffer(candidatePayloadBuf_, EVICT_CANDIDATE_CAP * sizeof(uint32_t));
    candidatePayloadLocal_ = candidatePayloadBuf_.Get<uint32_t>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_);

}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::InitParams(const struct LICommon::ConstInfo &constInfo,
                                                   const LIA5TilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;  // 4
    s2BaseSize_ = constInfo.s2BaseSize;  // 128
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    returnValueFlag = constInfo.returnValueFlag;
    blockId_ = GetBlockIdx();
    trunkLen_ = TRUNK_LEN_16K;
    topkCount_ = constInfo.sparseCount;
    topkOp_.Init(topkCount_, topkCount_, trunkLen_);
    topkCountAlign256_ = LICommon::Align(constInfo.sparseCount, (uint64_t)256); // topkCount对齐到256
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::InitVecInputTensor(GlobalTensor<W_T> weightsGm,
                                                                                GlobalTensor<int32_t> indiceOutGm,
                                                                                GlobalTensor<K_T> valueOutGm,
                                                                                GlobalTensor<int32_t> blockTableGm,
                                                                                GlobalTensor<int32_t> cacheSlotsGm,
                                                                                GlobalTensor<int32_t> topkSlotsGm,
                                                                                GlobalTensor<int32_t> missCountGm)
{
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
    this->blockTableGm = blockTableGm;
    this->cacheSlotsGm = cacheSlotsGm;
    this->topkSlotsGm = topkSlotsGm;
    this->missCountGm = missCountGm;
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::InitVecWorkspaceTensor(GlobalTensor<uint16_t> scoreGm)
{
    this->scoreGm = scoreGm; // resucesum*k
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // init -1 and copy to output
    uint64_t dealSize = constInfo_.sparseCount;
    GlobalTensor<int32_t> indexOutput = indiceOutGm[invalidS1Offset];
    AscendC::InitGlobalMemory(indexOutput, dealSize, constInfo_.INVALID_IDX);
    if (returnValueFlag) {
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        Duplicate(valueOutLocal_.template ReinterpretCast<uint16_t>(), constInfo_.INVALID_VAL, constInfo_.sparseCount);

        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

        AscendC::DataCopyParams copyOutValueParams;
        copyOutValueParams.blockCount = 1;
        copyOutValueParams.blockLen = constInfo_.sparseCount * sizeof(K_T);
        copyOutValueParams.srcStride = 0;
        copyOutValueParams.dstStride = 0;
        AscendC::DataCopyPad(valueOutGm[invalidS1Offset], valueOutLocal_, copyOutValueParams);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::ProcessVec1(const LICommon::RunInfo &info)
{
    auto pingpong = (info.loop % 2);
    auto s1BaseSizePerAIV = CeilDiv(s1BaseSize_, 2);
    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;
    if (curAivS1ProcNum == 0) {
        CrossCoreWaitFlag<LICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
            LICommon::ConstInfo::CROSS_CV_EVENT + pingpong
        );  // V核等C核计算完mm1，mm1Res已搬运到UB
        CrossCoreSetFlag<LICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
            LICommon::ConstInfo::CROSS_VC_EVENT + pingpong
        );   // V核处理完，通知C核可以把mm1Res搬运到UB
        return;
    }
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    // weightsGm --> weightUB_
    int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * kHeadNum_ * gSize_;
    DataCopyPadExtParams<W_T> padWeightsParams{false, 0, 0, 0};
    DataCopyExtParams wDataCopyExtParams;
    wDataCopyExtParams.blockCount = curAivS1ProcNum;
    wDataCopyExtParams.blockLen = gSize_ * sizeof(W_T);
    wDataCopyExtParams.srcStride = 0;
    wDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - wDataCopyExtParams.blockLen) / 32;
    DataCopyPad(weightUB_[pingpong * (UB_BANK_STRIDE / sizeof(W_T))],
                weightsGm[weightGmOffset], wDataCopyExtParams, padWeightsParams);

    SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);

    // CV同步
    CrossCoreWaitFlag<LICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
        LICommon::ConstInfo::CROSS_CV_EVENT + info.loop % 2
    );   // V核等C核计算完mm1，mm1Res已搬运到UB

    auto outBase = vec1OutUB_[pingpong * (UB_BANK_STRIDE / sizeof(uint16_t))];
    auto weightBase = weightUB_[pingpong * (UB_BANK_STRIDE / sizeof(W_T))];
    auto weightFloatBase = weightFloatUB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(float)) / 2 * constInfo_.mBaseSize;

    vector1::BatchMulWeightAndReduceSum(outBase, UB_BANK_DEPTH_STRIDE / sizeof(uint16_t),
                                        qkBase, qkVLstride, (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(float)),
                                        weightBase, UB_BANK_DEPTH_STRIDE / sizeof(W_T), weightFloatBase,
                                        gSize_, curAivS1ProcNum);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    // outUB_ --->  scoreGm
    int64_t vec1OutGmOffset = blockId_ % 2 == 0
                                            ? curS2Idx
                                            : s1BaseSizePerAIV * LICommon::Align(
                                                (uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_
                                                ) + curS2Idx;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(uint16_t);
    copyOutParams.srcStride = (UB_BANK_DEPTH_STRIDE - UB_BANK_STRIDE) / 32;
    copyOutParams.dstStride = (LICommon::Align(
                                        (uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_
                                        ) - s2BaseSize_) * sizeof(uint16_t);
    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, copyOutParams);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    CrossCoreSetFlag<LICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
        LICommon::ConstInfo::CROSS_VC_EVENT + pingpong
    );   // V核处理完，通知C核可以把mm1Res搬运到UB
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::ProcessTopK(
    const LICommon::RunInfo &info, bool allowFullSlotPrefetch)
{
    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);

    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    LocalTensor<int32_t> slotPrefetchLocal = slotStageLocal_;
    uint32_t slotPrefetchCap = topkCount_;
    if (allowFullSlotPrefetch) {
        slotPrefetchLocal = resMm1UB_.template ReinterpretCast<int32_t>();
        slotPrefetchCap = trunkLen_;
    }
    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    }

    int32_t validS2Len = cuRealAcSeq;
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;
        uint32_t vecOffset = blockId_ % 2 * CeilDiv(s1BaseSize_, 2) + i;

        uint16_t zero = 0;
        int32_t neg = -1;
        if (constInfo_.attenMaskFlag) {
            validS2Len = (int32_t)i + cuRealAcSeq;
        }
        if (validS2Len <= 0) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCount_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                                                          indicesOutLocal_.ReinterpretCast<int32_t>(),
                                                          copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            if (returnValueFlag) {
                WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                Duplicate(valueOutLocal_.template ReinterpretCast<uint16_t>(), constInfo_.INVALID_VAL, topkCount_);

                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

                AscendC::DataCopyParams copyOutValueParams;
                copyOutValueParams.blockCount = 1;
                copyOutValueParams.blockLen = topkCount_ * sizeof(K_T);
                copyOutValueParams.srcStride = 0;
                copyOutValueParams.dstStride = 0;
                AscendC::DataCopyPad(
                    valueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * topkCount_],
                    valueOutLocal_,
                    copyOutValueParams);
                SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            }
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        AscendC::DataCopyPadExtParams<uint16_t> padParams{true, 0, 0, 0};
        if (validS2Len >= topkCount_) {
            uint32_t s2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
            if (s2LoopNum == 1) {
                uint32_t validS2LenAlign = LICommon::Align(validS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[validS2Len / 256 * 256], zero, validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                copyInParams.blockLen = validS2Len * sizeof(uint16_t); // byte
                AscendC::DataCopyPadExtParams<uint16_t> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(
                    mrgValueLocal_,
                    scoreGm[vecOffset * LICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)],
                    copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                topkOp_.RunAblationStage(
                    mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                    slotPrefetchLocal, slotStageLocal_, cacheSlotsGm,
                    static_cast<uint64_t>(info.cacheRowIdx) * constInfo_.cacheSlotsSize,
                    0, topkCount_, validS2LenAlign,
                    static_cast<uint32_t>(validS2Len), 0, 1,
                    true, slotPrefetchCap);
            } else {
                for (uint32_t loopIdx = 0; loopIdx < s2LoopNum; loopIdx++) {
                    if (loopIdx == 0) {
                        copyInParams.blockLen = trunkLen_ * sizeof(uint16_t); // byte
                        AscendC::DataCopyPad(
                            mrgValueLocal_,
                            scoreGm[vecOffset * LICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)],
                            copyInParams, padParams);
                        SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        topkOp_.RunAblationStage(
                            mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                            slotPrefetchLocal, slotStageLocal_, cacheSlotsGm,
                            static_cast<uint64_t>(info.cacheRowIdx) * constInfo_.cacheSlotsSize,
                            0, topkCount_, trunkLen_, trunkLen_,
                            loopIdx, s2LoopNum, true, slotPrefetchCap);
                        continue;
                    }
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    uint32_t validTrunkLen = (loopIdx * trunkLen_ + trunkLen_) > validS2Len
                                                                               ? validS2Len % trunkLen_
                                                                               :trunkLen_;
                    uint32_t offset = vecOffset *
                                 LICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                                 loopIdx * trunkLen_;
                    AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkCountAlign256_);
                    // topk如果没有对齐到256，则把topkCountAlign256_ - topkCount_部分刷0
                    if (topkCountAlign256_ != topkCount_) {
                        uint64_t mask[1];
                        mask[0] = ~0;
                        mask[0] = mask[0] << (topkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        // 把topkCount_对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                        PipeBarrier<PIPE_V>();
                        // 把topk剩余对齐到256的部分刷0
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64 + 64], zero,
                                             topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    copyInParams.blockLen = validTrunkLen * sizeof(uint16_t); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if ((topkCountAlign256_ + validTrunkLen) % 256 != 0) {
                        Duplicate(mrgValueLocal_[topkCountAlign256_ + validTrunkLen / 256 * 256],
                                        zero, LICommon::Align(validTrunkLen,
                                        (uint32_t)256) - validTrunkLen / 256 * 256);
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                    }
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[offset], copyInParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    topkOp_.RunAblationStage(
                        mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                        slotPrefetchLocal, slotStageLocal_, cacheSlotsGm,
                        static_cast<uint64_t>(info.cacheRowIdx) * constInfo_.cacheSlotsSize,
                        loopIdx * trunkLen_, topkCount_,
                        LICommon::Align(topkCountAlign256_ + validTrunkLen,
                                        static_cast<uint32_t>(256)),
                        validTrunkLen, loopIdx, s2LoopNum,
                        true, slotPrefetchCap);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                }
            }
        } else {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero, validS2Len);
            if (returnValueFlag) {
                copyInParams.blockLen = LICommon::Align(validS2Len, (int32_t)32) * sizeof(uint16_t);
                AscendC::DataCopyPad(scoreOutLocal_,
                            scoreGm[vecOffset * LICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)],
                            copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            }
        }

        if (validS2Len < topkCount_) {
            uint64_t mask[1];
            mask[0] = ~0;
            mask[0] = mask[0] << (validS2Len % 8);
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, mask, 1, 1, 0);
        }

        if (validS2Len / 8 * 8 + 64 < topkCount_) {
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64],
                                        neg, topkCount_ - (validS2Len / 8 * 8 + 64));
        }

        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        uint32_t outputRow = static_cast<uint32_t>(info.indiceOutOffset / topkCount_) +
                             curS1Idx + rowIdx;
        uint64_t scoreRowOffset = vecOffset *
            LICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_);
#if LI_UPDATE_ABLATION_MODE < 2
        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyPad(
            indiceOutGm[static_cast<uint64_t>(outputRow) * topkCount_],
            indicesOutLocal_.template ReinterpretCast<int32_t>(),
            copyOutParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
#elif LI_UPDATE_ABLATION_MODE == 2
        // P2 deliberately exports the packed survivor payload. Performance
        // tests ignore the diagnostic output; avoiding an extra unpack keeps
        // P2-P1 focused on payload construction and chunk matching.
        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyPad(
            indiceOutGm[static_cast<uint64_t>(outputRow) * topkCount_],
            indicesOutLocal_.template ReinterpretCast<int32_t>(),
            copyOutParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
#else
        FinalizePayloadUpdate(info, outputRow, scoreRowOffset,
                              static_cast<uint32_t>(validS2Len));
#endif
        // 是否返回Value值
        if (returnValueFlag) {
            WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            // uint16_t -> bfloat16
            if (std::is_same_v<K_T, bfloat16_t>) {
                vector1::UIntToFloatReturnValue(valueOutLocal_.template ReinterpretCast<bfloat16_t>(),
                    scoreOutLocal_, topkCountAlign256_);
            } else {
                vector1::UIntToFloatReturnValue(valueOutLocal_.template ReinterpretCast<half>(),
                    scoreOutLocal_, topkCountAlign256_);
            }

            if (validS2Len < topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (validS2Len % 16);
                PipeBarrier<PIPE_V>();
                Duplicate(valueOutLocal_.template ReinterpretCast<uint16_t>()[validS2Len / 16 * 16],
                            constInfo_.INVALID_VAL, mask, 1, 1, 0);
            }
            if (validS2Len / 16 * 16 + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(valueOutLocal_.template ReinterpretCast<uint16_t>()[validS2Len / 16 * 16 + 64],
                            constInfo_.INVALID_VAL, topkCount_ - (validS2Len / 16 * 16 + 64));
            }
            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyParams copyOutValueParams;
            copyOutValueParams.blockCount = 1;
            copyOutValueParams.blockLen = topkCount_ * sizeof(K_T); // bytes
            copyOutValueParams.srcStride = 0;
            copyOutValueParams.dstStride = 0;
            // 搬运到GM
            AscendC::DataCopyPad(
                valueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * topkCount_],
                valueOutLocal_, copyOutValueParams);
        }
    }
}
}  // namespace LIKernel
#endif
