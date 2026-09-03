/**
 * Quantized LightningIndexer phase retained for the P10 score-receipt source
 * base.
 *
 * The implementation is derived from the Ascend 950 C8 Quant
 * LightningIndexer services used by vLLM-Ascend.  It writes the native
 * top-2048 token IDs directly into caller-owned canonical_sparse_indices.
 * The native score workspace is rebound to the caller-owned score surface;
 * receipt publication does not alter the original Top-K control path.
 */

#ifndef A5_P10_W4A4C8_SCORE_RECEIPT_QLI_H
#define A5_P10_W4A4C8_SCORE_RECEIPT_QLI_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "a5_p10_w4a4c8_score_receipt_tiling.h"
#include "arch35/quant_lightning_indexer_common.h"
#include "arch35/quant_lightning_indexer_service_cube.h"
#include "arch35/quant_lightning_indexer_service_vector.h"

namespace a5_p10_w4a4c8_score_receipt_impl {
using namespace AscendC;
using namespace QLICommon;
using namespace QLIKernel;

constexpr uint32_t QLI_BLOCK_SIZE = 128;
constexpr uint32_t QLI_HEAD_DIM = 128;
constexpr uint32_t QLI_TOPK = 2048;
constexpr uint32_t REQUEST_DONE_EVENT = 6;

using C8QliType = QLIType<
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, uint16_t, int32_t, true,
    LI_LAYOUT::TND, LI_LAYOUT::PA_BSND>;

class QuantLiPhase {
public:
    __aicore__ inline QuantLiPhase(
        TPipe *pipe, const A5P10W4A4C8ScoreReceiptTilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR query, GM_ADDR key, GM_ADDR weights,
        GM_ADDR queryDequantScale, GM_ADDR keyDequantScale,
        GM_ADDR candidateLens,
        GM_ADDR blockTable, GM_ADDR topkIndices,
        GM_ADDR canonicalScoreSourceLens, GM_ADDR scoreSurface,
        GM_ADDR receiptRing, GM_ADDR receiptState, GM_ADDR forwardControl,
        GM_ADDR userWorkspace)
    {
        // P10 deliberately has no opaque per-core score workspace.  Keep the
        // formal workspace argument for the CANN kernel ABI, but bind the
        // imported C8 layout directly to score_surface below.
        (void)userWorkspace;
        if ASCEND_IS_AIV {
            subBlockIdx_ = GetBlockIdx();
            aiCoreIdx_ = subBlockIdx_ / 2U;
        } else {
            subBlockIdx_ = GetBlockIdx();
            aiCoreIdx_ = subBlockIdx_;
        }

        candidateLensGm_.SetGlobalBuffer((__gm__ int32_t *)candidateLens);
        blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);

        constInfo_.batchSize = tiling_->batchSize;
        constInfo_.gSize = tiling_->indexHeads;
        constInfo_.qHeadNum = tiling_->indexHeads;
        constInfo_.kHeadNum = 1;
        constInfo_.headDim = QLI_HEAD_DIM;
        constInfo_.sparseCount = QLI_TOPK;
        constInfo_.kSeqSize = tiling_->maxCandidateLen;
        constInfo_.qSeqSize = 1;
        constInfo_.kCacheBlockSize = QLI_BLOCK_SIZE;
        constInfo_.maxBlockNumPerBatch = tiling_->maxBlockNumPerBatch;
        constInfo_.outputLayout = LI_LAYOUT::TND;
        constInfo_.attenMaskFlag = false;
        constInfo_.cmpRatio = 1;
        constInfo_.batchSupperFlag = false;
        constInfo_.stride = tiling_->keyStride;
        constInfo_.scaleStride = tiling_->scaleStride;
        constInfo_.mBaseSize = 256;
        constInfo_.s1BaseSize =
            (constInfo_.mBaseSize + constInfo_.gSize - 1U) /
            constInfo_.gSize;
        constInfo_.s2BaseSize = QLI_BLOCK_SIZE;

        if ASCEND_IS_AIV {
            weightsGm_.SetGlobalBuffer((__gm__ bfloat16_t *)weights);
            queryScaleGm_.SetGlobalBuffer(
                (__gm__ float *)queryDequantScale);
            keyScaleGm_.SetGlobalBuffer((__gm__ float *)keyDequantScale);
            topkIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)topkIndices);
            canonicalScoreSourceLensGm_.SetGlobalBuffer(
                (__gm__ int32_t *)canonicalScoreSourceLens);
            scoreSurfaceGm_.SetGlobalBuffer((__gm__ uint16_t *)scoreSurface);
            receiptRingGm_.SetGlobalBuffer((__gm__ int64_t *)receiptRing);
            receiptStateGm_.SetGlobalBuffer((__gm__ int32_t *)receiptState);
            forwardControlGm_.SetGlobalBuffer(
                (__gm__ int32_t *)forwardControl);
            vectorService_.InitParams(constInfo_);
            vectorService_.InitVecInputTensor(
                weightsGm_, queryScaleGm_, keyScaleGm_,
                topkIndicesGm_, blockTableGm_);
            vectorService_.InitVecWorkspaceTensor(scoreSurfaceGm_);
            vectorService_.InitScoreReceiptOutputs(
                canonicalScoreSourceLensGm_, receiptRingGm_, receiptStateGm_,
                forwardControlGm_,
                tiling_->receiptTilesPerRow, tiling_->receiptCapacity,
                tiling_->scoreSurfaceRowStrideBytes);
            vectorService_.InitBuffers(pipe_);
        } else {
            queryGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)query);
            keyGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)key);
            matmulService_.InitParams(constInfo_);
            matmulService_.InitMm1GlobalTensor(
                blockTableGm_, keyGm_, queryGm_);
            matmulService_.InitBuffers(pipe_);
        }
    }

    __aicore__ inline void Process()
    {
        // `receipt_state` and `forward_control` are cold-bound mutable graph
        // aliases initialized by forward preparation *before* C8 and Stage-B
        // start concurrently.  Stage-A must never clear receipt_state here:
        // doing so would race an already-running Stage-B consumer.  Its only
        // legal receipt transitions are FREE -> FILLING -> READY.

        bool hasActiveRow = false;
        for (uint32_t batch = aiCoreIdx_; batch < tiling_->batchSize;
             batch += tiling_->usedCoreNum) {
            if (IsActiveRow(batch)) {
                hasActiveRow = true;
                break;
            }
        }
        // ASCEND_IS_AIV expands to an `if constexpr` condition rather than a
        // Boolean expression.  Keep the runtime row predicate inside each
        // compile-time domain branch: AIV owns vector/event setup and AIC
        // owns Cube setup, exactly as the native C8 mixed-kernel protocol.
        if ASCEND_IS_AIV {
            if (hasActiveRow) {
                vectorService_.AllocEventID();
                CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                    ConstInfo::CROSS_VC_EVENT);
                CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                    ConstInfo::CROSS_VC_EVENT + 1U);
            }
        } else {
            if (hasActiveRow) {
                matmulService_.AllocEventID();
            }
        }

        uint32_t globalLoop = 0;
        for (uint32_t batch = aiCoreIdx_;
             hasActiveRow && batch < tiling_->batchSize;
             batch += tiling_->usedCoreNum) {
            if (!IsActiveRow(batch)) {
                continue;
            }
            const uint32_t candidate = static_cast<uint32_t>(
                candidateLensGm_.GetValue(batch));
            const uint32_t loopCount = candidate / QLI_BLOCK_SIZE;
            if ASCEND_IS_AIV {
                const uint64_t sliceOffsetBytes =
                    static_cast<uint64_t>(batch) *
                    tiling_->scoreSurfaceSliceStrideBytes;
                GlobalTensor<uint16_t> scoreSurfaceSlice =
                    scoreSurfaceGm_[sliceOffsetBytes / sizeof(uint16_t)];
                // This is a pointer rebind only: ProcessVec1 and ProcessTopK
                // retain their imported native row/tile offsets.
                vectorService_.BindGraphScoreSurfaceSlice(
                    scoreSurfaceSlice, sliceOffsetBytes);
            }
            for (uint32_t s2 = 0; s2 < loopCount; ++s2, ++globalLoop) {
                RunInfo run{};
                run.loop = globalLoop;
                run.bN2Idx = batch;
                run.bIdx = batch;
                run.n2Idx = 0;
                run.gS1Idx = 0;
                run.s2Idx = s2;
                run.actS1Size = 1;
                run.actS2Size = candidate;
                run.actS2SizeOrig = candidate;
                run.actMBaseSize = tiling_->indexHeads;
                run.actualSingleProcessSInnerSize = QLI_BLOCK_SIZE;
                run.actualSingleProcessSInnerSizeAlign = QLI_BLOCK_SIZE;
                run.tensorQueryOffset =
                    static_cast<uint64_t>(batch) * tiling_->indexHeads *
                    QLI_HEAD_DIM;
                run.tensorKeyOffset =
                    static_cast<uint64_t>(s2) * QLI_BLOCK_SIZE *
                    QLI_HEAD_DIM;
                run.tensorKeyScaleOffset =
                    static_cast<uint64_t>(s2) * QLI_BLOCK_SIZE;
                run.tensorWeightsOffset =
                    static_cast<uint64_t>(batch) * tiling_->indexHeads;
                run.indiceOutOffset =
                    static_cast<uint64_t>(batch) * QLI_TOPK;
                run.isFirstS2InnerLoop = s2 == 0;
                run.isLastS2InnerLoop = s2 + 1U == loopCount;
                run.isAllLoopEnd = false;
                run.isValid = true;

                if ASCEND_IS_AIC {
                    matmulService_.ComputeMm1(run);
                } else {
                    vectorService_.ProcessVec1(run);
                    if (run.isLastS2InnerLoop) {
                        vectorService_.ProcessTopK(run);
                        // With actS1Size=1 only the first AIV owns the
                        // logical row.  This scalar witness is independent
                        // of Top-K ordering and is written after the native
                        // Top-K routine returns.
                        if ((subBlockIdx_ & 1U) == 0U) {
                            vectorService_.WriteCanonicalScoreSourceLen(
                                batch, static_cast<int32_t>(candidate));
                        }
                    }
                }
            }

            // The Cube may reuse its score workspace only after the even AIV
            // has consumed the completed row in TopK.
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                    REQUEST_DONE_EVENT);
            } else {
                if ((subBlockIdx_ & 1U) == 0U) {
                    CrossCoreSetFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
                        REQUEST_DONE_EVENT);
                }
            }
        }

        if ASCEND_IS_AIV {
            if (hasActiveRow) {
                vectorService_.FreeEventID();
            }
        } else {
            if (hasActiveRow) {
                matmulService_.FreeEventID();
                CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                    ConstInfo::CROSS_VC_EVENT);
                CrossCoreWaitFlag<ConstInfo::QLI_SYNC_MODE4, PIPE_FIX>(
                    ConstInfo::CROSS_VC_EVENT + 1U);
            }
        }

        // This is a one-time terminal barrier, copied from the established
        // mixed AIC/AIV CANN pattern.  It is *not* a per-tile C8 barrier and
        // does not touch any native event ID.  The designated AIV then seals
        // the dedicated forward-control word so independently launched
        // Stage-B workers know that no further READY record can appear.
        SyncAll<false>();
        if ASCEND_IS_AIV {
            if (subBlockIdx_ == 0U) {
                vectorService_.WriteProducerSealed();
            }
        }
    }

private:
    __aicore__ inline bool IsActiveRow(uint32_t batch)
    {
        const int32_t candidate = candidateLensGm_.GetValue(batch);
        return candidate >= static_cast<int32_t>(QLI_TOPK) &&
            candidate <= static_cast<int32_t>(tiling_->maxCandidateLen) &&
            candidate % static_cast<int32_t>(QLI_BLOCK_SIZE) == 0;
    }

    TPipe *pipe_;
    const A5P10W4A4C8ScoreReceiptTilingData *tiling_;
    uint32_t subBlockIdx_ = 0;
    uint32_t aiCoreIdx_ = 0;
    ConstInfo constInfo_{};
    QLIMatmul<C8QliType> matmulService_;
    QLIVector<C8QliType> vectorService_;
    GlobalTensor<fp8_e4m3fn_t> queryGm_;
    GlobalTensor<fp8_e4m3fn_t> keyGm_;
    GlobalTensor<bfloat16_t> weightsGm_;
    GlobalTensor<float> queryScaleGm_;
    GlobalTensor<float> keyScaleGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> topkIndicesGm_;
    GlobalTensor<int32_t> canonicalScoreSourceLensGm_;
    GlobalTensor<uint16_t> scoreSurfaceGm_;
    GlobalTensor<int64_t> receiptRingGm_;
    GlobalTensor<int32_t> receiptStateGm_;
    GlobalTensor<int32_t> forwardControlGm_;
};
} // namespace a5_p10_w4a4c8_score_receipt_impl

#endif  // A5_P10_W4A4C8_SCORE_RECEIPT_QLI_H
