/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../op_kernel/a5_kvcache_scatter_copy_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t HBM_K_ROPE = 0;
constexpr size_t HBM_KV_CACHE = 1;
constexpr size_t DRAM_K_ROPE = 2;
constexpr size_t DRAM_KV_CACHE = 3;
constexpr size_t HBM_BLOCK_TABLE = 4;
constexpr size_t DRAM_BLOCK_TABLE = 5;
constexpr size_t SRC_TOKEN_IDS = 6;
constexpr size_t DST_SLOTS = 7;
constexpr size_t COPY_COUNTS = 8;

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t K_ROPE_DIM = 64;
constexpr int64_t KV_CACHE_DIM = 512;
constexpr int64_t MAX_COPY_CAP = 65536;

bool HasShape(const gert::Shape& shape, int64_t dim0, int64_t dim1)
{
    return shape.GetDimNum() == 3 && shape.GetDim(1) == dim0 && shape.GetDim(2) == dim1;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i <= COPY_COUNTS; ++i) {
        if (context->GetInputShape(i) == nullptr ||
            context->GetInputDesc(i) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }

    const ge::DataType cacheDataType =
        context->GetInputDesc(HBM_K_ROPE)->GetDataType();
    if (cacheDataType != ge::DT_BF16 &&
        cacheDataType != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = HBM_K_ROPE; i <= DRAM_KV_CACHE; ++i) {
        if (context->GetInputDesc(i)->GetDataType() != cacheDataType) {
            return ge::GRAPH_FAILED;
        }
    }
    for (size_t i = HBM_BLOCK_TABLE; i <= COPY_COUNTS; ++i) {
        if (context->GetInputDesc(i)->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }
    constexpr uint32_t elementBytes = 2U;

    const gert::Shape hbmRope = context->GetInputShape(HBM_K_ROPE)->GetStorageShape();
    const gert::Shape hbmKv = context->GetInputShape(HBM_KV_CACHE)->GetStorageShape();
    const gert::Shape dramRope = context->GetInputShape(DRAM_K_ROPE)->GetStorageShape();
    const gert::Shape dramKv = context->GetInputShape(DRAM_KV_CACHE)->GetStorageShape();
    const gert::Shape hbmTable = context->GetInputShape(HBM_BLOCK_TABLE)->GetStorageShape();
    const gert::Shape dramTable = context->GetInputShape(DRAM_BLOCK_TABLE)->GetStorageShape();
    const gert::Shape srcIds = context->GetInputShape(SRC_TOKEN_IDS)->GetStorageShape();
    const gert::Shape dstSlots = context->GetInputShape(DST_SLOTS)->GetStorageShape();
    const gert::Shape copyCounts = context->GetInputShape(COPY_COUNTS)->GetStorageShape();

    if (!HasShape(hbmRope, BLOCK_SIZE, K_ROPE_DIM) ||
        !HasShape(dramRope, BLOCK_SIZE, K_ROPE_DIM) ||
        !HasShape(hbmKv, BLOCK_SIZE, KV_CACHE_DIM) ||
        !HasShape(dramKv, BLOCK_SIZE, KV_CACHE_DIM) ||
        hbmRope.GetDim(0) != hbmKv.GetDim(0) ||
        dramRope.GetDim(0) != dramKv.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }
    if (hbmTable.GetDimNum() != 2 || dramTable.GetDimNum() != 2 ||
        srcIds.GetDimNum() != 2 || dstSlots.GetDimNum() != 2 ||
        copyCounts.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batchSize = copyCounts.GetDim(0);
    const int64_t copyCap = srcIds.GetDim(1);
    if (batchSize <= 0 || copyCap <= 0 || copyCap > MAX_COPY_CAP ||
        srcIds.GetDim(0) != batchSize ||
        dstSlots.GetDim(0) != batchSize || dstSlots.GetDim(1) != copyCap ||
        hbmTable.GetDim(0) != batchSize || dramTable.GetDim(0) != batchSize ||
        hbmTable.GetDim(1) <= 0 || dramTable.GetDim(1) <= 0 ||
        dramTable.GetDim(1) * BLOCK_SIZE > (1 << 18)) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCoreNum = platform.GetCoreNumAiv();
    if (aivCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t totalPairSlots =
        static_cast<uint64_t>(batchSize) * static_cast<uint64_t>(copyCap);
    const uint32_t usedCoreNum =
        totalPairSlots < aivCoreNum ? static_cast<uint32_t>(totalPairSlots) : aivCoreNum;
    auto* tiling = context->GetTilingData<A5KvcacheScatterCopyTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->batchSize = static_cast<uint32_t>(batchSize);
    tiling->copyCap = static_cast<uint32_t>(copyCap);
    tiling->hbmMaxBlockNum = static_cast<uint32_t>(hbmTable.GetDim(1));
    tiling->dramMaxBlockNum = static_cast<uint32_t>(dramTable.GetDim(1));
    tiling->elementBytes = elementBytes;
    tiling->totalPairSlots = totalPairSlots;
    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
static ge::graphStatus InferShapeForA5KvcacheScatterCopy(
    gert::InferShapeContext* context)
{
    if (context == nullptr ||
        context->GetInputShape(HBM_K_ROPE) == nullptr ||
        context->GetInputShape(HBM_KV_CACHE) == nullptr ||
        context->GetOutputShape(0) == nullptr ||
        context->GetOutputShape(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) =
        *context->GetInputShape(HBM_K_ROPE);
    *context->GetOutputShape(1) =
        *context->GetInputShape(HBM_KV_CACHE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForA5KvcacheScatterCopy(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(
        0, context->GetInputDataType(HBM_K_ROPE));
    context->SetOutputDataType(
        1, context->GetInputDataType(HBM_KV_CACHE));
    return ge::GRAPH_SUCCESS;
}

class A5KvcacheScatterCopy : public OpDef {
public:
    explicit A5KvcacheScatterCopy(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> dataTypes = {
            ge::DT_BF16, ge::DT_FLOAT16};
        const std::vector<ge::DataType> intTypes = {
            ge::DT_INT32, ge::DT_INT32};
        const std::vector<ge::Format> dataFormats = {
            ge::FORMAT_ND, ge::FORMAT_ND};
        const std::vector<ge::Format> intFormats = {
            ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("hbm_k_rope").ParamType(REQUIRED).DataType(dataTypes).Format(dataFormats);
        this->Input("hbm_kv_cache").ParamType(REQUIRED).DataType(dataTypes).Format(dataFormats);
        this->Input("dram_k_rope").ParamType(REQUIRED).DataType(dataTypes).Format(dataFormats);
        this->Input("dram_kv_cache").ParamType(REQUIRED).DataType(dataTypes).Format(dataFormats);
        this->Input("hbm_block_table").ParamType(REQUIRED).DataType(intTypes).Format(intFormats);
        this->Input("dram_block_table").ParamType(REQUIRED).DataType(intTypes).Format(intFormats);
        this->Input("src_token_ids").ParamType(REQUIRED).DataType(intTypes).Format(intFormats);
        this->Input("dst_slots").ParamType(REQUIRED).DataType(intTypes).Format(intFormats);
        this->Input("copy_counts").ParamType(REQUIRED).DataType(intTypes).Format(intFormats);
        this->Output("hbm_k_rope_out").ParamType(REQUIRED).DataType(dataTypes).Format(dataFormats);
        this->Output("hbm_kv_cache_out").ParamType(REQUIRED).DataType(dataTypes).Format(dataFormats);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5KvcacheScatterCopy);

IMPL_OP_INFERSHAPE(A5KvcacheScatterCopy)
    .InferShape(InferShapeForA5KvcacheScatterCopy)
    .InferDataType(InferDataTypeForA5KvcacheScatterCopy);
} // namespace ops
