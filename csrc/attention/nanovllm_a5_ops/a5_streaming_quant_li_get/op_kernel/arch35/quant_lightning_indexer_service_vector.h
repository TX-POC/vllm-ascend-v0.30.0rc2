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
 * \file quant_lightning_indexer_service_vector.h
 * \brief
 */
#ifndef quant_lightning_indexer_SERVICE_VECTOR_H
#define quant_lightning_indexer_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_common.h"
#include "../a5_streaming_quant_li_get_hook_mode.h"
#if A5_STREAMING_LI_GET_INLINE_MODE == 1
#include "a5_streaming_quant_li_get_tile_audit.h"
#elif A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE
#include "a5_streaming_quant_li_get_tile_ready.h"
#endif
#include "../arch35/vf/quant_lightning_indexer_vector1.h"
#include "../arch35/vf/quant_lightning_indexer_topk.h"

namespace QLIKernel {
using namespace QLICommon;
constexpr uint32_t TRUNK_LEN_16K = 16384;
template <typename QLIT>
class QLIVector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = QLIT::pageAttention;

    using QK_T = typename QLIT::queryKeyType;
    using SCORE_T = typename QLIT::scoreType;

    __aicore__ inline QLIVector(){};
    __aicore__ inline void ProcessVec1(const QLICommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const QLICommon::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct QLICommon::ConstInfo &constInfo,
                                      const QLITilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm);
#if !A5_STREAMING_LI_GET_SCORE_SOURCE_LENS_IS_INPUT_ALIAS
    // Produced by this operator, not inferred by its host bridge.  Each
    // element is the exact score-source S length used by ProcessTopK for the
    // corresponding sparse-index output row.
    __aicore__ inline void InitScoreSourceLensOutput(
        GlobalTensor<int32_t> scoreSourceLensGm);
#endif
    // The caller-owned score workspace is a byte-identical replacement for
    // the private C8 scratch score-GM.  In mode 0 the original workspace is
    // still used; modes 1/2 use this explicit address so a sidecar can later
    // read the exact uint16 scores without reconstructing Q/K math.
    // `score_workspace` is declared as int32 only because that is the
    // graph/ACL tensor carrier.  C8 addresses it as uint16 sortable scores.
    // Keep the published base in *bytes* so the independent consumer cannot
    // accidentally interpret it as an int32 element index.
    __aicore__ inline void InitScoreWorkspaceBase(uint64_t baseBytes);
#if A5_STREAMING_LI_GET_PUBLISH_MODE >= 2
    // A receipt slot is two cache lines: line 0 is immutable descriptor data;
    // line 1 begins with epoch. Same-MTE3 ordering establishes their producer
    // issue order. A future concurrent sidecar must additionally use a
    // documented cross-kernel acquire/release edge before treating epoch as
    // ready; the current P3 proof reads receipts only after kernel completion.
    __aicore__ inline void InitScoreTileReceipts(
        GlobalTensor<int64_t> tileReceiptsGm,
        GlobalTensor<int64_t> streamingEpochGm);
    __aicore__ inline void PrepareScoreTileReceipts(
        const QLICommon::RunInfo &info, uint64_t vec1OutGmOffset,
        int64_t curAivS1Idx, int64_t curAivS1ProcNum,
        int64_t curS2Idx, uint32_t pingpong);
    __aicore__ inline void EmitScoreTileReceipts(
        const QLICommon::RunInfo &info, int64_t curAivS1Idx,
        int64_t curAivS1ProcNum, uint32_t pingpong);
#endif
#if A5_STREAMING_LI_GET_INLINE_MODE == 1
    // Audit has a deliberately smaller closure: it receives only the tensors
    // needed to reproduce the score-time predicate and fixed receipt.  It
    // has no Direct-HCCP/SIMT/descriptor/private-payload arguments.
    __aicore__ inline void InitStreamingAuditHook(
        __gm__ uint8_t *residentThreshold, __gm__ uint8_t *residentThresholdValid,
        __gm__ uint8_t *kvObjectSlotsSnapshot,
        __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *kvObjectMap,
        __gm__ uint8_t *privateSourceToSlot,
        __gm__ uint8_t *queueLeases, __gm__ uint8_t *kvObjectLens,
        const QLITilingData *__restrict tilingData);
#elif A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE
    // This hook observes the exact uint16 score tile after the official C8
    // reduction and before its existing score-GM DMA.  It owns only the
    // transaction-local prefetch state; sparse Top-K outputs remain wholly
    // owned by the unmodified C8 path below.
    __aicore__ inline void InitStreamingKvafHook(
        __gm__ uint8_t *residentThreshold, __gm__ uint8_t *residentThresholdValid,
        __gm__ uint8_t *kvObjectSlotsSnapshot,
        __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *kvObjectMap,
        __gm__ uint8_t *privateSourceToSlot, __gm__ uint8_t *privateReadyTable,
        __gm__ uint8_t *privateArenaHead, __gm__ uint8_t *privatePackedOutput,
        __gm__ uint8_t *queueControl,
        __gm__ uint8_t *queueLeases, __gm__ uint8_t *kvObjectLens,
        __gm__ uint8_t *telemetry,
        const QLITilingData *__restrict tilingData);
    // FreeEventID owns the final official C8 TOPK_MTE3_V_EVENT.  The optional
    // native-enqueue terminal poll/release is intentionally after that drain;
    // audit mode has no in-flight wave and returns immediately.
    __aicore__ inline void FinishStreamingKvafHook();
#endif
    __aicore__ inline void InitVecInputTensor(GlobalTensor<float> weightsGm, GlobalTensor<float> qScaleGm,
                                              GlobalTensor<float> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm);
#if !A5_STREAMING_LI_GET_SCORE_SOURCE_LENS_IS_INPUT_ALIAS
    // Used only by the all-invalid forward path. It clears non-alias output
    // #8 in disjoint AIV shards so graph replay cannot expose an old S length.
    __aicore__ inline void ClearScoreSourceLensForInvalidForward(
        uint32_t producerIndex, uint32_t producerCount);
#endif
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

protected:
    GlobalTensor<SCORE_T> scoreGm;
    GlobalTensor<float> weightsGm;
    GlobalTensor<float> qScaleGm;
    GlobalTensor<float> kScaleGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;
#if !A5_STREAMING_LI_GET_SCORE_SOURCE_LENS_IS_INPUT_ALIAS
    GlobalTensor<int32_t> scoreSourceLensGm;
#endif
    uint64_t scoreWorkspaceBaseBytes_ = 0;
#if A5_STREAMING_LI_GET_PUBLISH_MODE >= 2
    GlobalTensor<int64_t> tileReceiptsGm;
    int64_t streamingEpoch_ = 0;
#endif
    // =================================常量区=================================
    static constexpr uint32_t VEC1_V_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;

    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t KSCALE_S_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;

private:
    __aicore__ inline void GetKeyScale(const QLICommon::RunInfo &runInfo, LocalTensor<float> &kScaleUB,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
#if !A5_STREAMING_LI_GET_SCORE_SOURCE_LENS_IS_INPUT_ALIAS
    // `score_source_lens` is a P3-only side output.  It is intentionally
    // published through MTE3 rather than a scalar GM SetValue: paired AIVs
    // can own adjacent rows, which share a 32-byte GM sector on A5.
    __aicore__ inline void WriteScoreSourceLen(uint64_t outputRow, int32_t value);
#endif
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<QK_T> resMm1UB_;
    //tmp buff for weight
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<float> weightUB_;
    //tmp buff for kScale
    TBuf<TPosition::VECCALC> kScaleBuf_;
    LocalTensor<float> kScaleUB_;
    //tmp buff for qScale
    TBuf<TPosition::VECCALC> qScaleBuf_;
    LocalTensor<float> qScaleUB_;
    //tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;
    // tmp buff for LD

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

#if A5_STREAMING_LI_GET_PUBLISH_MODE >= 2
    // Receipt descriptors must not share the Top-K merge buffer. The final
    // score tile can still be in MTE3 when ProcessTopK starts consuming
    // mrgValueLocal_, so sharing that storage would allow the Top-K path to
    // overwrite an in-flight descriptor DMA source. The existing VEC1
    // MTE3->V ping-pong token already protects this dedicated buffer before
    // the next same-pingpong score tile reuses it; after the final tile,
    // FreeEventID drains that token before the kernel returns.
    TBuf<TPosition::VECCALC> tileReceiptBuf_;
    LocalTensor<int64_t> tileReceiptLocal_;
#endif

#if A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE
    // Small caller-owned UB hand-off records for KVAF's queue lease and
    // private-arena CAS.  Descriptor arrays remain graph-static in HBM and
    // are supplied by the cold-bound KVAF streaming control.
    TBuf<TPosition::VECCALC> streamingLeaseBuf_;
    LocalTensor<uint8_t> streamingLeaseUb_;
    TBuf<TPosition::VECCALC> streamingClaimBuf_;
    LocalTensor<uint8_t> streamingClaimUb_;
    // P3 uses this 2 KiB allocation as uint64 source offsets. P6 aliases the
    // same bytes as two 64x4 uint32 CandidateSeed microbatches. The two paths
    // are compile-time exclusive: P6 never executes legacy Emit(), so the
    // aliases cannot overlap in one invocation.
    TBuf<TPosition::VECCALC> streamingSourceOffsetsBuf_;
#if A5_STREAMING_LI_GET_HAS_CANDIDATE_SEED_PIPELINE
    LocalTensor<uint32_t> streamingCandidateSeedsUb_;
#else
    LocalTensor<uint64_t> streamingSourceOffsetsUb_;
#endif
#endif

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;

    TBuf<TPosition::VECCALC> outInvalidBuf_;
    LocalTensor<int32_t> outInvalidLocal_;

#if !A5_STREAMING_LI_GET_SCORE_SOURCE_LENS_IS_INPUT_ALIAS
    // One scalar is sufficient per publication; eight int32s reserve a full
    // 32-byte UB line for the MTE3 DMA source and mirror established A5
    // side-output publication patterns.
    TBuf<TPosition::VECCALC> scoreSourceLensBuf_;
    LocalTensor<int32_t> scoreSourceLensLocal_;
#endif

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
    uint32_t streamingRowCount_ = 0;

    struct QLICommon::ConstInfo constInfo_;
    topk::LITopk<SCORE_T> topkOp_;
#if A5_STREAMING_LI_GET_INLINE_MODE == 1
    streaming_li_get::StreamingLiTileAuditHook streamingAuditHook_;
#elif A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE
    streaming_li_get::StreamingKvafTileReadyHook streamingKvafHook_;
#endif
};

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(QK_T));   //大小：2(开dB) * 2 * 64 * 128 * 4 = 128KB
    resMm1UB_ = resMm1Buf_.Get<QK_T>();//qk
    pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, 2) * gSize_* sizeof(float));   // 大小：2(开dB) * 2 * 64 * 2 = 0.5KB
    weightUB_ = weightBuf_.Get<float>();//weight
    pipe->InitBuffer(kScaleBuf_, 2 * s2BaseSize_ * sizeof(float));                   // 大小：2(开dB) * 128 * 4 = 1KB
    kScaleUB_ = kScaleBuf_.Get<float>();//kScale
    pipe->InitBuffer(qScaleBuf_, 2 * CeilDiv(s1BaseSize_, 2) * gSize_* sizeof(float));      // 大小：2(开dB) * 2 * 64 * 4 = 1KB
    qScaleUB_ = qScaleBuf_.Get<float>();//qScale
    pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T));      // 大小：2(开dB) * 2 * 128 * 4 = 2KB
    vec1OutUB_ = outBuf_.Get<SCORE_T>();//out

    // Topk
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T));     // 大小：(topkCountAlign256_ + 每次排序长度) * sizeof(SCORE_T)
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();

#if A5_STREAMING_LI_GET_PUBLISH_MODE >= 2
    constexpr uint32_t kReceiptWordsPerTile = 16U;
    // One AIV owns at most ceil(S1_base / 2) rows for each ping-pong score
    // tile. Keep both ping-pong copies so a later MTE3 store never aliases a
    // descriptor being prepared on V.
    const uint32_t rowsPerAiv = CeilDiv(s1BaseSize_, 2);
    pipe->InitBuffer(tileReceiptBuf_,
                     2U * rowsPerAiv * kReceiptWordsPerTile * sizeof(int64_t));
    tileReceiptLocal_ = tileReceiptBuf_.Get<int64_t>();
#endif

#if A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE
    pipe->InitBuffer(streamingLeaseBuf_, 32U);
    streamingLeaseUb_ = streamingLeaseBuf_.Get<uint8_t>();
    pipe->InitBuffer(streamingClaimBuf_, 32U);
    streamingClaimUb_ = streamingClaimBuf_.Get<uint8_t>();
#if A5_STREAMING_LI_GET_HAS_CANDIDATE_SEED_PIPELINE
    // Keep P6 at P5's exact source-offset scratch footprint. Do not add a
    // separate CandidateSeed TBuf here: that shifts the official C8 buffers
    // beyond the Vector UB budget before any HCCP action can run.
    pipe->InitBuffer(streamingSourceOffsetsBuf_,
                     streaming_li_get::kCandidateRingSlots *
                     streaming_li_get::kCandidateMicrobatchCapacity *
                     streaming_li_get::kCandidateSeedWords * sizeof(uint32_t));
    streamingCandidateSeedsUb_ = streamingSourceOffsetsBuf_.Get<uint32_t>();
#else
    pipe->InitBuffer(streamingSourceOffsetsBuf_,
                     streaming_li_get::kMaxWaveRequests * sizeof(uint64_t));
    streamingSourceOffsetsUb_ = streamingSourceOffsetsBuf_.Get<uint64_t>();
#endif
#endif

    pipe->InitBuffer(indicesOutBuf_, (topkCountAlign256_ + 64) * sizeof(uint32_t));         // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    pipe->InitBuffer(scoreOutBuf_, topkCountAlign256_ * sizeof(SCORE_T));                   // 大小：topkCountAlign256_ * sizeof(SCORE_T)
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_);

    //刷-1
    pipe->InitBuffer(outInvalidBuf_, topkCount_ * sizeof(int32_t));
    outInvalidLocal_ = outInvalidBuf_.Get<int32_t>();
#if !A5_STREAMING_LI_GET_SCORE_SOURCE_LENS_IS_INPUT_ALIAS
    pipe->InitBuffer(scoreSourceLensBuf_, 8 * sizeof(int32_t));
    scoreSourceLensLocal_ = scoreSourceLensBuf_.Get<int32_t>();
#endif
    Duplicate(kScaleUB_, float(0), 2 * s2BaseSize_);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitParams(const struct QLICommon::ConstInfo &constInfo,
                                                   const QLITilingData *__restrict tilingData)
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
    blockId_ = GetBlockIdx();
    trunkLen_ = TRUNK_LEN_16K;
    topkCount_ = constInfo.sparseCount;
    topkCountAlign256_ = QLICommon::Align(constInfo.sparseCount, (uint64_t)256); // topkCount对齐到256
    streamingRowCount_ = tilingData->streamingRowCount;
    topkOp_.Init(topkCount_, trunkLen_);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecInputTensor(GlobalTensor<float> weightsGm, GlobalTensor<float> qScaleGm,
                                                           GlobalTensor<float> kScaleGm,
                                                           GlobalTensor<int32_t> indiceOutGm,
                                                           GlobalTensor<int32_t> blockTableGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm)
{
    this->scoreGm = scoreGm;//resucesum*k
}

#if !A5_STREAMING_LI_GET_SCORE_SOURCE_LENS_IS_INPUT_ALIAS
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitScoreSourceLensOutput(
    GlobalTensor<int32_t> scoreSourceLensGm)
{
    this->scoreSourceLensGm = scoreSourceLensGm;
}
#endif

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitScoreWorkspaceBase(
    uint64_t baseBytes)
{
    scoreWorkspaceBaseBytes_ = baseBytes;
}

#if A5_STREAMING_LI_GET_INLINE_MODE == 1
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitStreamingAuditHook(
    __gm__ uint8_t *residentThreshold, __gm__ uint8_t *residentThresholdValid,
    __gm__ uint8_t *kvObjectSlotsSnapshot,
    __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *kvObjectMap,
    __gm__ uint8_t *privateSourceToSlot,
    __gm__ uint8_t *queueLeases, __gm__ uint8_t *kvObjectLens,
    const QLITilingData *__restrict tilingData)
{
    streamingAuditHook_.Init(
        residentThreshold, residentThresholdValid, kvObjectSlotsSnapshot,
        reqPoolEntries, kvObjectLens, kvObjectMap, privateSourceToSlot,
        queueLeases, constInfo_, tilingData->streamingRowCount,
        tilingData->streamingSourceCapacity, tilingData->streamingPoolSize,
        tilingData->streamingKvObjectCapacity,
        tilingData->streamingQueueCount);
}
#elif A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitStreamingKvafHook(
    __gm__ uint8_t *residentThreshold, __gm__ uint8_t *residentThresholdValid,
    __gm__ uint8_t *kvObjectSlotsSnapshot,
    __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *kvObjectMap,
    __gm__ uint8_t *privateSourceToSlot, __gm__ uint8_t *privateReadyTable,
    __gm__ uint8_t *privateArenaHead, __gm__ uint8_t *privatePackedOutput,
    __gm__ uint8_t *queueControl,
    __gm__ uint8_t *queueLeases, __gm__ uint8_t *kvObjectLens,
    __gm__ uint8_t *telemetry,
    const QLITilingData *__restrict tilingData)
{
    streamingKvafHook_.Init(
        residentThreshold, residentThresholdValid, kvObjectSlotsSnapshot,
        reqPoolEntries, kvObjectLens, kvObjectMap,
        privateSourceToSlot, privateReadyTable, privateArenaHead,
        privatePackedOutput, queueControl,
        queueLeases, telemetry, constInfo_, tilingData->streamingRowCount,
        tilingData->streamingSourceCapacity, tilingData->streamingPoolSize,
        tilingData->streamingKvObjectCapacity,
        tilingData->streamingPrivatePackedRows,
        tilingData->streamingPrivateArenaCapacity,
        tilingData->streamingQueueCount,
        tilingData->streamingKvafRowsPerEntry,
        reinterpret_cast<__ubuf__ kvaf_fused_get::KvafStreamingBatchGetLeaseResult *>(
            streamingLeaseUb_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ streaming_li_get::PrivateArenaClaimResult *>(
            streamingClaimUb_.GetPhyAddr()),
#if A5_STREAMING_LI_GET_HAS_CANDIDATE_SEED_PIPELINE
        // The P6 CandidateSeed pipeline reuses P3's 2 KiB scratch. The
        // legacy source-offset path is compile-time exclusive from P6, so
        // both typed views deliberately name this same allocation.
        reinterpret_cast<__ubuf__ uint64_t *>(streamingCandidateSeedsUb_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t *>(streamingCandidateSeedsUb_.GetPhyAddr()),
#else
        reinterpret_cast<__ubuf__ uint64_t *>(streamingSourceOffsetsUb_.GetPhyAddr()),
        nullptr,
#endif
        static_cast<uint32_t>(blockId_));
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::FinishStreamingKvafHook()
{
    streamingKvafHook_.Finish();
}
#endif

#if A5_STREAMING_LI_GET_PUBLISH_MODE >= 2
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitScoreTileReceipts(
    GlobalTensor<int64_t> tileReceiptsGm,
    GlobalTensor<int64_t> streamingEpochGm)
{
    this->tileReceiptsGm = tileReceiptsGm;
    streamingEpoch_ = streamingEpochGm.GetValue(0);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::PrepareScoreTileReceipts(
    const QLICommon::RunInfo &info, uint64_t vec1OutGmOffset,
    int64_t curAivS1Idx, int64_t curAivS1ProcNum,
    int64_t curS2Idx, uint32_t pingpong)
{
    // Publication is deliberately descriptor-only.  It has no queue claim,
    // no HCCP call, no threshold comparison, no score observation, and no
    // extra C8 hard event.  The existing V->MTE3/MTE3->V event pair brackets
    // both the official score copy and these two fixed-line copies.
    constexpr uint64_t kWordsPerLine = 8;
    constexpr uint64_t kWordsPerReceipt = 16;
    constexpr uint64_t kContractVersion = 1;
    const uint64_t scoreRowStride = QLICommon::Align(
        static_cast<uint64_t>(constInfo_.kSeqSize),
        static_cast<uint64_t>(s2BaseSize_));
    const int64_t validBase = constInfo_.attenMaskFlag
        ? info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1
        : info.actS2Size;

    // This storage is intentionally disjoint from mrgValueLocal_: Top-K may
    // start before the final MTE3 descriptor transfer retires. The existing
    // VEC1 MTE3->V ping-pong handoff serializes reuse of tileReceiptLocal_
    // without adding an event ID or perturbing the official Top-K lifecycle.
    auto descriptorLocal = tileReceiptLocal_;
    for (int64_t localRow = 0; localRow < curAivS1ProcNum; ++localRow) {
        const uint64_t outputRow =
            static_cast<uint64_t>(info.indiceOutOffset) / topkCount_ +
            static_cast<uint64_t>(curAivS1Idx + localRow);
        const int64_t validS2 = constInfo_.attenMaskFlag
            ? (validBase + localRow) / static_cast<int64_t>(constInfo_.cmpRatio)
            : info.actS2Size;
        const int64_t remain = validS2 - curS2Idx;
        const uint64_t validInTile = remain <= 0 ? 0 :
            static_cast<uint64_t>(remain) <
                    static_cast<uint64_t>(info.actualSingleProcessSInnerSize)
                ? static_cast<uint64_t>(remain)
                : static_cast<uint64_t>(info.actualSingleProcessSInnerSize);
        const uint64_t localBase =
            static_cast<uint64_t>(pingpong) * 2U * kWordsPerReceipt +
            static_cast<uint64_t>(localRow) * kWordsPerReceipt;

        // Word 0 is a byte offset *relative to the beginning of the passed
        // score_workspace view*, not a device virtual address and not an
        // int32-/uint16-element index.  A consumer adds it to its own view
        // base, so graph replay cannot accidentally retain an old address or
        // introduce a hidden factor-of-two convention.
        descriptorLocal.SetValue(localBase + 0, static_cast<int64_t>(
            scoreWorkspaceBaseBytes_ +
            (vec1OutGmOffset + static_cast<uint64_t>(localRow) *
                scoreRowStride) * sizeof(SCORE_T)));
        descriptorLocal.SetValue(localBase + 1, static_cast<int64_t>(outputRow));
        descriptorLocal.SetValue(localBase + 2, curS2Idx);
        descriptorLocal.SetValue(localBase + 3, static_cast<int64_t>(validInTile));
        descriptorLocal.SetValue(localBase + 4, static_cast<int64_t>(scoreRowStride));
        descriptorLocal.SetValue(localBase + 5, static_cast<int64_t>(blockId_));
        descriptorLocal.SetValue(localBase + 6, static_cast<int64_t>(info.loop));
        descriptorLocal.SetValue(localBase + 7, static_cast<int64_t>(kContractVersion));

        // The ready cache line is separate from the immutable descriptor
        // line. A future consumer combines this epoch with its graph/device
        // acquire edge; it must not rely on a plain unsynchronized GM poll.
        for (uint64_t word = 0; word < kWordsPerLine; ++word) {
            descriptorLocal.SetValue(localBase + kWordsPerLine + word,
                word == 0 ? streamingEpoch_ : 0);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::EmitScoreTileReceipts(
    const QLICommon::RunInfo &info, int64_t curAivS1Idx,
    int64_t curAivS1ProcNum, uint32_t pingpong)
{
    constexpr uint64_t kWordsPerLine = 8;
    constexpr uint64_t kWordsPerReceipt = 16;
    const uint64_t tilesPerRow = CeilDiv(
        static_cast<uint64_t>(constInfo_.kSeqSize),
        static_cast<uint64_t>(s2BaseSize_));
    auto descriptorLocal = tileReceiptLocal_;
    AscendC::DataCopyParams lineCopy{1, 64, 0, 0};
    for (int64_t localRow = 0; localRow < curAivS1ProcNum; ++localRow) {
        const uint64_t outputRow =
            static_cast<uint64_t>(info.indiceOutOffset) / topkCount_ +
            static_cast<uint64_t>(curAivS1Idx + localRow);
        const uint64_t receiptBase =
            (outputRow * tilesPerRow + info.s2Idx) * kWordsPerReceipt;
        const uint64_t localBase =
            static_cast<uint64_t>(pingpong) * 2U * kWordsPerReceipt +
            static_cast<uint64_t>(localRow) * kWordsPerReceipt;
        // MTE3 is in-order: the score tile copy was enqueued first, this
        // immutable line second, and its epoch line last.
        AscendC::DataCopyPad(tileReceiptsGm[receiptBase],
                             descriptorLocal[localBase], lineCopy);
        AscendC::DataCopyPad(tileReceiptsGm[receiptBase + kWordsPerLine],
                             descriptorLocal[localBase + kWordsPerLine],
                             lineCopy);
    }
}
#endif

#if !A5_STREAMING_LI_GET_SCORE_SOURCE_LENS_IS_INPUT_ALIAS
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::WriteScoreSourceLen(
    uint64_t outputRow, int32_t value)
{
    scoreSourceLensLocal_.SetValue(0, value);
    PipeBarrier<PIPE_V>();

    // Do not reuse any official C8 TOPK event here.  The scalar receipt has
    // its own S<->MTE3 dependency and waits before reusing the local buffer.
    // This avoids A5 scalar-GM false sharing without changing LI/TopK math,
    // index ownership, or the canonical C8 output-DMA event lifecycle.
    AscendC::DataCopyParams scalarCopy{
        1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
    SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
    AscendC::DataCopyPad(scoreSourceLensGm[outputRow], scoreSourceLensLocal_,
                         scalarCopy);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ClearScoreSourceLensForInvalidForward(
    uint32_t producerIndex, uint32_t producerCount)
{
    if (producerCount == 0U || producerIndex >= producerCount ||
        streamingRowCount_ == 0U) {
        return;
    }
    const uint64_t begin =
        static_cast<uint64_t>(streamingRowCount_) * producerIndex / producerCount;
    const uint64_t end =
        static_cast<uint64_t>(streamingRowCount_) * (producerIndex + 1U) /
        producerCount;
    if (end > begin) {
        GlobalTensor<int32_t> output = scoreSourceLensGm[begin];
        AscendC::InitGlobalMemory(output, end - begin, 0);
    }
}
#endif

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // Keep the side output in exact lockstep with official invalid-index
    // cleanup.  `invalidS1Offset` is the flattened sparse-index offset, so
    // division by sparseCount yields exactly the graph-visible output row.
    // A downstream reconcile must treat zero as "no valid score source",
    // never as a host-inferred sequence length.
#if !A5_STREAMING_LI_GET_SCORE_SOURCE_LENS_IS_INPUT_ALIAS
    WriteScoreSourceLen(
        static_cast<uint64_t>(invalidS1Offset) / constInfo_.sparseCount, 0);
#endif
    // init -1 and copy to output
    Duplicate(outInvalidLocal_, constInfo_.INVALID_IDX, constInfo_.sparseCount);

    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

    AscendC::DataCopyParams dataCopyOutParams;
    dataCopyOutParams.blockCount = 1;
    dataCopyOutParams.blockLen = constInfo_.sparseCount * sizeof(int32_t);
    dataCopyOutParams.srcStride = 0;
    dataCopyOutParams.dstStride = 0;
    AscendC::DataCopyPad(indiceOutGm[invalidS1Offset], outInvalidLocal_, dataCopyOutParams);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::GetKeyScale(const QLICommon::RunInfo &runInfo, LocalTensor<float> &kScaleUB,
                                                    int64_t batchId, int64_t startS2, int64_t getLen)
{
    // startS2一定能整除kCacheBlockSize_
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    if constexpr (PAGE_ATTENTION) {
        int32_t startBlockTableIdx = startS2 / kCacheBlockSize_;
        int32_t startBlockTableOffset = startS2 % kCacheBlockSize_;
        int32_t blockTableBatchOffset = batchId * maxBlockNumPerBatch_;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        int32_t resUbBaseOffset = 0;
        if (startBlockTableOffset > 0) {
            int32_t firstPartLen =
                kCacheBlockSize_ - startBlockTableOffset > getLen ? getLen : kCacheBlockSize_ - startBlockTableOffset;
            copyInParams.blockLen = firstPartLen * sizeof(float);
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[(runInfo.loop % 2) * s2BaseSize_],
                                 kScaleGm[blockId * constInfo_.scaleStride + startBlockTableOffset],
                                 copyInParams, padParams);
            startBlockTableIdx++;
            getLen = getLen - firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        copyInParams.blockLen = kCacheBlockSize_ * sizeof(float);
        for (int32_t i = 0; i < getLoopNum; i++) {
            if (i == getLoopNum - 1) {
                copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(float);
            }
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[(runInfo.loop % 2) * s2BaseSize_ + resUbBaseOffset + i * kCacheBlockSize_],
                                 kScaleGm[blockId * constInfo_.scaleStride],
                                 copyInParams, padParams);
        }
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(kScaleUB[(runInfo.loop % 2) * s2BaseSize_], kScaleGm[runInfo.tensorKeyScaleOffset], copyInParams, padParams);
    }
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessVec1(const QLICommon::RunInfo &info)
{
    auto pingpong = (info.loop % 2);
    auto s1BaseSizePerAIV = CeilDiv(s1BaseSize_, 2);
    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    if (curAivS1ProcNum == 0) {
        CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_CV_EVENT + pingpong);  // V核等C核计算完mm1，mm1Res已搬运到UB
        CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + pingpong);   // V核处理完，通知C核可以把mm1Res搬运到UB
        return;
    }
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    //weightsGm --> weightUB_
    int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * kHeadNum_ * gSize_;
    DataCopyPadExtParams<float> padWeightsParams{false, 0, 0, 0};
    DataCopyExtParams qwDataCopyExtParams;
    qwDataCopyExtParams.blockCount = curAivS1ProcNum;
    qwDataCopyExtParams.blockLen = gSize_ * sizeof(float);
    qwDataCopyExtParams.srcStride = 0;
    qwDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - UB_BANK_STRIDE) / 32;
    DataCopyPad(weightUB_[pingpong * (UB_BANK_STRIDE / sizeof(float))],
                weightsGm[weightGmOffset], qwDataCopyExtParams, padWeightsParams);

    //qScaleGm  -->  qScaleUB_
    DataCopyPadExtParams<float> padQScaleParams{false, 0, 0, 0};
    DataCopyPad(qScaleUB_[pingpong * (UB_BANK_STRIDE / sizeof(float))],
                qScaleGm[weightGmOffset], qwDataCopyExtParams, padQScaleParams);

    //kScaleGm  -->  kScaleUB_
    GetKeyScale(info, kScaleUB_, info.bIdx, curS2Idx, info.actualSingleProcessSInnerSize);
    SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);

    //CV同步
    CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_CV_EVENT + info.loop % 2);   //V核等C核计算完mm1，mm1Res已搬运到UB

    static_assert(std::is_same_v<SCORE_T, uint16_t>);
    auto outBase = vec1OutUB_[pingpong * (UB_BANK_STRIDE / sizeof(SCORE_T))];
    auto weightBase = weightUB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qScaleBase = qScaleUB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto kScaleBase = kScaleUB_[pingpong * s2BaseSize_];

    auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(QK_T))];
    auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(QK_T)) / 2 * constInfo_.mBaseSize;
#if A5_STREAMING_LI_GET_INLINE_MODE == 1
    // Audit mode has no queue capacity or in-flight HCCP state.  Its only
    // reason to observe a score tile is to write the exact fixed receipt.
    const bool mayNeedScoreObservation =
        streamingAuditHook_.MayNeedScoreObservation();
#elif A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE
#if A5_STREAMING_LI_GET_HAS_CANDIDATE_SEED_PIPELINE
    // P6 opportunity seam.  It can only consume seeds from an earlier score
    // tile because the current tile has not been reduced yet.  It never reads
    // `outBase`, so a successful PostOnly doorbell can overlap the official
    // C8 reduction/DMA/Top-K that follows without holding score ping-pong UB.
    streamingKvafHook_.ProgressOwnPostAndCq();
#endif
    // Take the bounded queue-capacity snapshot before reduction.  The exact
    // score seam is left completely untouched in the immediate P3 path when
    // the producer has neither an audit obligation nor a locally
    // pending/available Get queue.  P6 deliberately returns true here: queue
    // scarcity suppresses only speculation, never ExactScoreScan.
    const bool mayNeedScoreObservation =
        streamingKvafHook_.MayNeedScoreObservation();
#endif
    vector1::BatchMulWeightAndReduceSum(outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T),
                                        qkBase, qkVLstride, (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(QK_T)),
                                        weightBase, UB_BANK_DEPTH_STRIDE / sizeof(float),
                                        kScaleBase, (uint32_t)0,
                                        qScaleBase, UB_BANK_DEPTH_STRIDE / sizeof(float),
                                        gSize_, curAivS1ProcNum);
#if A5_STREAMING_LI_GET_INLINE_MODE == 1
    if (mayNeedScoreObservation) {
        PipeBarrier<PIPE_V>();
        // BatchMulWeightAndReduceSum produces outBase on Vector, while the
        // audit observer consumes it with Scalar GetValue(). PIPE_V only
        // orders Vector instructions; it does not make a V-owned UB value
        // visible to Scalar. Use the same explicit V -> S dependency as the
        // ResidentSeed path uses before it reads a Vector-produced Top-K
        // boundary. This leaves the official score-GM and Top-K path intact.
        const event_t scoreVectorToScalar = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(scoreVectorToScalar);
        WaitFlag<HardEvent::V_S>(scoreVectorToScalar);
        streamingAuditHook_.Emit(outBase, info, curAivS1Idx,
                                 curAivS1ProcNum, curS2Idx);
        // The receipt hook has finished every Scalar GetValue() from this
        // ping-pong score tile.  Pair the V -> S readiness edge with S -> V
        // completion before Vector can hand the UB tile to MTE3 or recycle it
        // on the next ping-pong iteration.  Without this acknowledgement,
        // V_S alone proves only that Scalar did not start too early; it does
        // not protect Scalar's read lifetime against a later Vector reuse.
        const event_t scoreScalarToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(scoreScalarToVector);
        WaitFlag<HardEvent::S_V>(scoreScalarToVector);
    }
#elif A5_STREAMING_LI_GET_HAS_FULL_POST_CLOSURE
    // Exact inline seam: read only the uint16 values the official reduction
    // just produced, before C8 begins its normal score-GM copy.  The hook
    // never writes this tile or sparse Top-K state.
    //
    // `outBase` is Vector-owned.  A PIPE_V barrier alone cannot make it safe
    // for the Scalar hook to call GetValue(), nor can it prevent Vector/MTE3
    // from recycling the ping-pong buffer while Scalar is still scanning it.
    // The two cross-pipe edges below therefore form the minimum correctness
    // envelope for *any* inline ExactScoreScan/Post implementation:
    //
    //   Vector score complete -> Scalar reads/compacts/posts -> Vector reuse
    //
    // This is intentionally the same V->S / S->V lifetime contract used by
    // the audit path.  It fixes data correctness; it is not yet the final
    // P5 throughput schedule.  The target ring implementation will move the
    // scalar completion acknowledgement to the next reuse of this ping-pong
    // tile, after its bounded CandidateSeed hand-off is proven safe.
    if (mayNeedScoreObservation) {
        PipeBarrier<PIPE_V>();
        const event_t scoreVectorToScalar = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(scoreVectorToScalar);
        WaitFlag<HardEvent::V_S>(scoreVectorToScalar);
#if A5_STREAMING_LI_GET_HAS_CANDIDATE_SEED_PIPELINE
        // Only the cheap exact predicate and CandidateSeed compact live in
        // this score-UB lifetime.  Private-slot CAS, descriptor writes and
        // Direct-HCCP PostOnly run at a later opportunity seam instead.
        streamingKvafHook_.ScanAndCompactTile(outBase, info, curAivS1Idx,
                                               curAivS1ProcNum, curS2Idx);
#else
        streamingKvafHook_.Emit(outBase, info, curAivS1Idx,
                                curAivS1ProcNum, curS2Idx);
#endif
        const event_t scoreScalarToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(scoreScalarToVector);
        WaitFlag<HardEvent::S_V>(scoreScalarToVector);
#if A5_STREAMING_LI_GET_HAS_CANDIDATE_SEED_PIPELINE
        // The S->V acknowledgement returns score ping-pong UB to C8.  This
        // is the earliest safe opportunity to PostOnly the just-published
        // seed, so HCCP DMA can overlap this tile's official score DMA/Top-K
        // and all later tiles.  The routine never reads `outBase`.
        streamingKvafHook_.ProgressOwnPostAndCq();
        // Keep the causal CQ witness after the earliest Post seam.  It is
        // nonblocking and proves that subsequent exact C8 progress can occur
        // while an older HCCP CQ is still live.
        streamingKvafHook_.ObserveCqAfterScoreTile();
#endif
    }
#endif
    int64_t vec1OutGmOffset = blockId_ % 2 == 0 ? curS2Idx :
                            s1BaseSizePerAIV * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + curS2Idx;
#if A5_STREAMING_LI_GET_PUBLISH_MODE >= 2
    // Prepare on V before the existing V->MTE3 token. No MTE3 request is
    // issued yet, therefore the source is complete when the existing score
    // DMA begins below.
    PrepareScoreTileReceipts(info, vec1OutGmOffset, curAivS1Idx,
                             curAivS1ProcNum, curS2Idx, pingpong);
#endif
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    //outUB_ ---> scoreGm
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(SCORE_T);
    copyOutParams.srcStride = (UB_BANK_DEPTH_STRIDE - UB_BANK_STRIDE) / 32;
    copyOutParams.dstStride = (QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) - s2BaseSize_) * sizeof(SCORE_T);
    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, copyOutParams);
#if A5_STREAMING_LI_GET_PUBLISH_MODE >= 2
    EmitScoreTileReceipts(info, curAivS1Idx, curAivS1ProcNum, pingpong);
#endif
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + pingpong);   //V核处理完，通知C核可以把mm1Res搬运到UB
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessTopK(const QLICommon::RunInfo &info)
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

    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    }

    int32_t validS2Len = cuRealAcSeq;
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;
        uint32_t vecOffset = blockId_ % 2 * CeilDiv(s1BaseSize_, 2) + i;

        SCORE_T zero = 0;
        int32_t neg = -1;
        if (constInfo_.attenMaskFlag) {
            validS2Len = ((int32_t)i + cuRealAcSeq) / static_cast<int32_t>(constInfo_.cmpRatio);
        }
        // This is intentionally emitted from the unmodified official C8
        // valid-length calculation, before either the short-row or 16K
        // TopK branch.  `sparse_indices` live in score-source domain S; the
        // later exact-reconcile stage must use this S limit before mapping
        // any selected index to logical KV domain L.
#if !A5_STREAMING_LI_GET_SCORE_SOURCE_LENS_IS_INPUT_ALIAS
        const uint64_t outputRow =
            static_cast<uint64_t>(info.indiceOutOffset) / topkCount_ +
            static_cast<uint64_t>(curS1Idx + rowIdx);
        WriteScoreSourceLen(outputRow, validS2Len > 0 ? validS2Len : 0);
#endif
        if (validS2Len <= 0) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCount_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
        if (validS2Len >= topkCount_) {
            uint32_t s2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
            if (s2LoopNum == 1) {
                uint32_t validS2LenAlign = QLICommon::Align(validS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[validS2Len / 256 * 256], zero, validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                copyInParams.blockLen = validS2Len * sizeof(SCORE_T); // byte
                AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(mrgValueLocal_, scoreGm[vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)], copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, validS2LenAlign, 0, 1);
            } else {
                for (uint32_t loopIdx = 0; loopIdx < s2LoopNum; loopIdx++) {
                    if (loopIdx == 0) {
                        copyInParams.blockLen = trunkLen_ * sizeof(SCORE_T); // byte
                        AscendC::DataCopyPad(mrgValueLocal_, scoreGm[vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)], copyInParams, padParams);
                        SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, trunkLen_, loopIdx, s2LoopNum);
                        continue;
                    }
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    uint32_t validTrunkLen = (loopIdx * trunkLen_ + trunkLen_) > validS2Len ? validS2Len % trunkLen_ : trunkLen_;
                    uint32_t offset = vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + loopIdx * trunkLen_;
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
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64 + 64], zero, topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    copyInParams.blockLen = validTrunkLen * sizeof(SCORE_T); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if ((topkCountAlign256_ + validTrunkLen) % 256 != 0) {
                        Duplicate(mrgValueLocal_[topkCountAlign256_ + validTrunkLen / 256 * 256], zero, QLICommon::Align(validTrunkLen, (uint32_t)256) - validTrunkLen / 256 * 256);
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                    }
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[offset], copyInParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, QLICommon::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256), loopIdx, s2LoopNum);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                }
            }
        } else {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero, validS2Len);
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
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64], neg, topkCount_ - (validS2Len / 8 * 8 + 64));
        }

        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }

}
}  // namespace QLIKernel
#endif
