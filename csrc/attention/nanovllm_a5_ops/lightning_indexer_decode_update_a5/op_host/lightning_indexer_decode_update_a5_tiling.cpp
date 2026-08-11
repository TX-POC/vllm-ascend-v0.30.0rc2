/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "lightning_indexer_decode_update_a5_tiling.h"
#include <algorithm>
#include "../op_kernel/lightning_indexer_decode_update_a5_template_tiling_key.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

ge::graphStatus LightningIndexerDecodeUpdateA5Tiling::GetNpuInfo(LIA5TilingInfo &tilingInfo) const
{
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("LightningIndexerDecodeUpdateA5", "opName got from TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    tilingInfo.opName = context_->GetNodeName();
    tilingInfo.platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(tilingInfo.platformInfo == nullptr, OPS_LOG_E(tilingInfo.opName, "GetPlatformInfo is nullptr."),
               return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo.platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OPS_ERR_IF(aicNum == 0 || aivNum == 0, OPS_LOG_E(tilingInfo.opName, "num of core obtained is 0."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr,
               OPS_LOG_E(tilingInfo.opName, "workspace size buffer is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
               OPS_LOG_E(tilingInfo.opName, "raw tiling data is nullptr."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerDecodeUpdateA5Tiling::GetTensorInfo(LIA5TilingInfo &tilingInfo) const
{
    auto &op = tilingInfo.opParamInfo;
    op.query.desc = context_->GetInputDesc(QUERY_INDEX);
    op.query.shape = context_->GetInputShape(QUERY_INDEX);
    op.key.desc = context_->GetInputDesc(KEY_INDEX);
    op.key.shape = context_->GetInputShape(KEY_INDEX);
    op.weights.desc = context_->GetInputDesc(WEIGHTS_INDEX);
    op.weights.shape = context_->GetInputShape(WEIGHTS_INDEX);
    op.reqPoolEntries.desc = context_->GetInputDesc(REQ_POOL_ENTRIES_INDEX);
    op.reqPoolEntries.tensor = context_->GetInputTensor(REQ_POOL_ENTRIES_INDEX);
    op.cacheSlots.desc = context_->GetInputDesc(CACHE_SLOTS_INDEX);
    op.cacheSlots.shape = context_->GetInputShape(CACHE_SLOTS_INDEX);
    op.cacheTokens.desc = context_->GetInputDesc(CACHE_TOKENS_INDEX);
    op.cacheTokens.tensor = context_->GetInputTensor(CACHE_TOKENS_INDEX);
    op.actualSeqLengths.desc = context_->GetInputDesc(ACTUAL_SEQ_K_INDEX);
    op.actualSeqLengths.tensor = context_->GetInputTensor(ACTUAL_SEQ_K_INDEX);
    op.blockTable.desc = context_->GetInputDesc(BLOCK_TABLE_INDEX);
    op.blockTable.tensor = context_->GetInputTensor(BLOCK_TABLE_INDEX);
    op.topkIndexOut.desc = context_->GetOutputDesc(TOPK_INDEX);
    op.topkIndexOut.shape = context_->GetOutputShape(TOPK_INDEX);
    op.topkSlotsOut.desc = context_->GetOutputDesc(TOPK_SLOTS_INDEX);
    op.topkSlotsOut.shape = context_->GetOutputShape(TOPK_SLOTS_INDEX);
    op.missCountOut.desc = context_->GetOutputDesc(MISS_COUNT_INDEX);
    op.missCountOut.shape = context_->GetOutputShape(MISS_COUNT_INDEX);
    op.cacheSlotsOut.desc = context_->GetOutputDesc(CACHE_SLOTS_OUT_INDEX);
    op.cacheSlotsOut.shape = context_->GetOutputShape(CACHE_SLOTS_OUT_INDEX);

    OPS_ERR_IF(op.query.desc == nullptr || op.query.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "query desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.key.desc == nullptr || op.key.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "key desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.weights.desc == nullptr || op.weights.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "weights desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.reqPoolEntries.desc == nullptr || op.reqPoolEntries.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "req_pool_entries desc/tensor is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheSlots.desc == nullptr || op.cacheSlots.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "cache_slots_pool desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheTokens.desc == nullptr || op.cacheTokens.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "cache_tokens desc/tensor is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.actualSeqLengths.desc == nullptr || op.actualSeqLengths.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "actual_seq_lengths_key desc/tensor is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.blockTable.desc == nullptr || op.blockTable.tensor == nullptr,
               OPS_LOG_E(tilingInfo.opName, "block_table desc/tensor is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.topkIndexOut.desc == nullptr || op.topkIndexOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "topk_index desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.topkSlotsOut.desc == nullptr || op.topkSlotsOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "topk_slots desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.missCountOut.desc == nullptr || op.missCountOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "miss_count desc/shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheSlotsOut.desc == nullptr || op.cacheSlotsOut.shape == nullptr,
               OPS_LOG_E(tilingInfo.opName, "cache_slots_alias desc/shape is nullptr."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerDecodeUpdateA5Tiling::CheckDtype(const LIA5TilingInfo &tilingInfo) const
{
    const auto &op = tilingInfo.opParamInfo;
    ge::DataType qType = op.query.desc->GetDataType();
    ge::DataType kType = op.key.desc->GetDataType();
    ge::DataType wType = op.weights.desc->GetDataType();
    OPS_ERR_IF(qType != kType || qType != wType,
               OPS_LOG_E(tilingInfo.opName, "query/key/weights dtype must match."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(qType != ge::DT_FLOAT16 && qType != ge::DT_BF16,
               OPS_LOG_E(tilingInfo.opName, "query/key/weights dtype must be fp16 or bf16."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.cacheSlots.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "cache_slots_pool dtype must be int32."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.reqPoolEntries.desc->GetDataType() != ge::DT_INT32 ||
                   op.cacheTokens.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "req_pool_entries/cache_tokens dtype must be int32."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.actualSeqLengths.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "actual_seq_lengths_key dtype must be int32."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.blockTable.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName, "block_table dtype must be int32."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(op.topkIndexOut.desc->GetDataType() != ge::DT_INT32 ||
                   op.topkSlotsOut.desc->GetDataType() != ge::DT_INT32 ||
                   op.missCountOut.desc->GetDataType() != ge::DT_INT32 ||
                   op.cacheSlotsOut.desc->GetDataType() != ge::DT_INT32,
               OPS_LOG_E(tilingInfo.opName,
                         "source_ids/destination_slots/miss_counts/cache alias must be int32."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerDecodeUpdateA5Tiling::CheckShape(LIA5TilingInfo &tilingInfo) const
{
    const auto &op = tilingInfo.opParamInfo;
    const auto &qShape = op.query.shape->GetStorageShape();
    const auto &kShape = op.key.shape->GetStorageShape();
    const auto &wShape = op.weights.shape->GetStorageShape();
    const auto &reqPoolShape = op.reqPoolEntries.tensor->GetStorageShape();
    const auto &cacheShape = op.cacheSlots.shape->GetStorageShape();
    const auto &cacheTokensShape = op.cacheTokens.tensor->GetStorageShape();
    const auto &seqShape = op.actualSeqLengths.tensor->GetStorageShape();
    const auto &blockShape = op.blockTable.tensor->GetStorageShape();
    const auto &indexOutShape = op.topkIndexOut.shape->GetStorageShape();
    const auto &slotsOutShape = op.topkSlotsOut.shape->GetStorageShape();
    const auto &missCountOutShape = op.missCountOut.shape->GetStorageShape();
    const auto &cacheSlotsOutShape = op.cacheSlotsOut.shape->GetStorageShape();

    OPS_ERR_IF(qShape.GetDimNum() != DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "query must be TND [B, N1, 128], where N1 is 32 or 64."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDimNum() != DIM_NUM_FOUR,
               OPS_LOG_E(tilingInfo.opName, "key must be PA_BSND [num_blocks, block_size, 1, 128]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(wShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "weights must be [B, N1], where N1 is 32 or 64."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "cache_slots_pool must be [pool_size, source_capacity]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(reqPoolShape.GetDimNum() != DIM_NUM_ONE || cacheTokensShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "req_pool_entries/cache_tokens must be rank 1."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(seqShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "actual_seq_lengths_key must be rank 1."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(blockShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "block_table must be rank 2."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(indexOutShape.GetDimNum() != DIM_NUM_THREE || slotsOutShape.GetDimNum() != DIM_NUM_THREE,
               OPS_LOG_E(tilingInfo.opName, "topk_index/topk_slots must be [B, 1, 2048]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missCountOutShape.GetDimNum() != DIM_NUM_ONE,
               OPS_LOG_E(tilingInfo.opName, "miss_count must be [B]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheSlotsOutShape.GetDimNum() != DIM_NUM_TWO,
               OPS_LOG_E(tilingInfo.opName, "cache_slots_alias must be rank 2."),
               return ge::GRAPH_FAILED);

    tilingInfo.bSize = static_cast<uint32_t>(qShape.GetDim(0));
    tilingInfo.n1Size = static_cast<uint32_t>(qShape.GetDim(1));
    tilingInfo.n2Size = static_cast<uint32_t>(kShape.GetDim(DIM_IDX_TWO));
    tilingInfo.blockSize = static_cast<uint32_t>(kShape.GetDim(DIM_IDX_ONE));
    tilingInfo.maxBlockNumPerBatch = static_cast<uint32_t>(blockShape.GetDim(DIM_IDX_ONE));
    tilingInfo.s2Size = tilingInfo.blockSize * tilingInfo.maxBlockNumPerBatch;
    tilingInfo.poolSize = static_cast<uint32_t>(cacheShape.GetDim(0));
    tilingInfo.cacheSlotsSize = static_cast<uint32_t>(cacheShape.GetDim(1));

    OPS_ERR_IF(tilingInfo.bSize == 0, OPS_LOG_E(tilingInfo.opName, "batch size must be > 0."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(reqPoolShape.GetShapeSize() != tilingInfo.bSize ||
                   cacheTokensShape.GetShapeSize() != tilingInfo.bSize ||
                   seqShape.GetShapeSize() != tilingInfo.bSize ||
                   blockShape.GetDim(0) != tilingInfo.bSize,
               OPS_LOG_E(tilingInfo.opName,
                         "query and all per-request metadata batch dimensions must match."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.poolSize == 0 || tilingInfo.cacheSlotsSize == 0,
               OPS_LOG_E(tilingInfo.opName, "cache_slots_pool dimensions must be positive."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape.GetDim(0) == 0, OPS_LOG_E(tilingInfo.opName, "key num_blocks must be > 0."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.blockSize != 128,
               OPS_LOG_E(tilingInfo.opName, "key block_size must be 128."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.s2Size != tilingInfo.cacheSlotsSize ||
                   tilingInfo.cacheSlotsSize > MAX_CACHE_SLOTS_SIZE,
               OPS_LOG_E(tilingInfo.opName,
                         "source capacity must equal block-table capacity and be <= 2^18."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.n2Size != DECODE_N2,
               OPS_LOG_E(tilingInfo.opName, "key N2 must be 1."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(tilingInfo.n1Size != DECODE_G_SIZE_32 && tilingInfo.n1Size != DECODE_G_SIZE_64,
               OPS_LOG_E(tilingInfo.opName, "decode query N1 must be 32 or 64."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(qShape.GetDim(DIM_IDX_TWO) != DECODE_HEAD_DIM || kShape.GetDim(DIM_IDX_THREE) != DECODE_HEAD_DIM,
               OPS_LOG_E(tilingInfo.opName, "head_dim must be 128."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(wShape.GetDim(0) != tilingInfo.bSize || wShape.GetDim(1) != tilingInfo.n1Size,
               OPS_LOG_E(tilingInfo.opName, "weights must match query [B, N1]."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(indexOutShape.GetDim(0) != tilingInfo.bSize || indexOutShape.GetDim(1) != DECODE_N2 ||
                   indexOutShape.GetDim(2) != DECODE_SPARSE_COUNT ||
                   slotsOutShape.GetDim(0) != tilingInfo.bSize || slotsOutShape.GetDim(1) != DECODE_N2 ||
                   slotsOutShape.GetDim(2) != DECODE_SPARSE_COUNT,
               OPS_LOG_E(tilingInfo.opName, "topk_index/topk_slots must have shape [B, 1, 2048]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(missCountOutShape.GetDim(0) != tilingInfo.bSize,
               OPS_LOG_E(tilingInfo.opName, "miss_count must have shape [B]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(cacheSlotsOutShape.GetDim(0) != cacheShape.GetDim(0) ||
                   cacheSlotsOutShape.GetDim(1) != cacheShape.GetDim(1),
               OPS_LOG_E(tilingInfo.opName, "cache_slots_alias must match cache_slots_pool."),
               return ge::GRAPH_FAILED);

    tilingInfo.inputQType = op.query.desc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerDecodeUpdateA5Tiling::ParseAndCheck(LIA5TilingInfo &tilingInfo)
{
    if (GetNpuInfo(tilingInfo) != ge::GRAPH_SUCCESS || GetTensorInfo(tilingInfo) != ge::GRAPH_SUCCESS ||
        CheckDtype(tilingInfo) != ge::GRAPH_SUCCESS || CheckShape(tilingInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerDecodeUpdateA5Tiling::DoTiling(LIA5TilingInfo *tilingInfo)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    tilingInfo->usedCoreNum = aicNum;
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    constexpr uint32_t S1_BASE_SIZE = 4;
    constexpr uint32_t S2_BASE_SIZE = 128;
    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize += S1_BASE_SIZE *
        ((tilingInfo->s2Size + S2_BASE_SIZE - 1) / S2_BASE_SIZE) *
        S2_BASE_SIZE * sizeof(uint16_t) * aicNum;
    context_->GetWorkspaceSizes(1)[0] = workspaceSize;

    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_n2Size(DECODE_N2);
    tilingData_.set_gSize(tilingInfo->n1Size);
    tilingData_.set_s1Size(1);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_sparseCount(DECODE_SPARSE_COUNT);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_poolSize(tilingInfo->poolSize);
    tilingData_.set_cacheSlotsSize(tilingInfo->cacheSlotsSize);
    tilingData_.set_usedCoreNum(aicNum);
    tilingData_.set_sparseMode(0);
    tilingData_.set_preTokens(INT64_MAX);
    tilingData_.set_nextTokens(INT64_MAX);
    tilingData_.set_returnValue(0);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    uint32_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint32_t>(tilingInfo->inputQType));
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForLightningIndexerDecodeUpdateA5(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForLightningIndexerDecodeUpdateA5(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR("LightningIndexerDecodeUpdateA5", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    LIA5TilingInfo liInfo;
    LightningIndexerDecodeUpdateA5Tiling liTiling(context);
    if (liTiling.ParseAndCheck(liInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return liTiling.DoTiling(&liInfo);
}

IMPL_OP_OPTILING(LightningIndexerDecodeUpdateA5)
    .Tiling(TilingForLightningIndexerDecodeUpdateA5)
    .TilingParse<LIA5CompileInfo>(TilingPrepareForLightningIndexerDecodeUpdateA5);

} // namespace optiling
