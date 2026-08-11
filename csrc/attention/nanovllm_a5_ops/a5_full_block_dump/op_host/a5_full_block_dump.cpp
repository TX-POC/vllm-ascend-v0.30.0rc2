/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "../op_kernel/a5_full_block_dump_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t HBM_K_ROPE = 0;
constexpr size_t HBM_KV_CACHE = 1;
constexpr size_t DRAM_K_ROPE = 2;
constexpr size_t DRAM_KV_CACHE = 3;
constexpr size_t SRC_BLOCK_IDS = 4;
constexpr size_t DST_BLOCK_IDS = 5;

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t K_ROPE_DIM = 64;
constexpr int64_t KV_CACHE_DIM = 512;

bool HasCacheShape(const gert::Shape& shape, int64_t dim)
{
    return shape.GetDimNum() == 3 &&
           shape.GetDim(0) > 0 &&
           shape.GetDim(1) == BLOCK_SIZE &&
           shape.GetDim(2) == dim;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i <= DST_BLOCK_IDS; ++i) {
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
    if (context->GetInputDesc(SRC_BLOCK_IDS)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(DST_BLOCK_IDS)->GetDataType() != ge::DT_INT32) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape hbmRope =
        context->GetInputShape(HBM_K_ROPE)->GetStorageShape();
    const gert::Shape hbmKv =
        context->GetInputShape(HBM_KV_CACHE)->GetStorageShape();
    const gert::Shape dramRope =
        context->GetInputShape(DRAM_K_ROPE)->GetStorageShape();
    const gert::Shape dramKv =
        context->GetInputShape(DRAM_KV_CACHE)->GetStorageShape();
    const gert::Shape srcIds =
        context->GetInputShape(SRC_BLOCK_IDS)->GetStorageShape();
    const gert::Shape dstIds =
        context->GetInputShape(DST_BLOCK_IDS)->GetStorageShape();

    if (!HasCacheShape(hbmRope, K_ROPE_DIM) ||
        !HasCacheShape(dramRope, K_ROPE_DIM) ||
        !HasCacheShape(hbmKv, KV_CACHE_DIM) ||
        !HasCacheShape(dramKv, KV_CACHE_DIM) ||
        hbmRope.GetDim(0) != hbmKv.GetDim(0) ||
        dramRope.GetDim(0) != dramKv.GetDim(0) ||
        srcIds.GetDimNum() != 1 || dstIds.GetDimNum() != 1 ||
        srcIds.GetDim(0) <= 0 || srcIds.GetDim(0) != dstIds.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t jobCount = srcIds.GetDim(0);
    if (jobCount > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        hbmRope.GetDim(0) > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        dramRope.GetDim(0) > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCoreNum = platform.GetCoreNumAiv();
    if (aivCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t totalTokenSlots =
        static_cast<uint64_t>(jobCount) * static_cast<uint64_t>(BLOCK_SIZE);
    const uint32_t usedCoreNum = totalTokenSlots < aivCoreNum
        ? static_cast<uint32_t>(totalTokenSlots)
        : aivCoreNum;
    if (usedCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<A5FullBlockDumpTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->jobCount = static_cast<uint32_t>(jobCount);
    tiling->hbmBlockCount = static_cast<uint32_t>(hbmRope.GetDim(0));
    tiling->dramBlockCount = static_cast<uint32_t>(dramRope.GetDim(0));
    tiling->elementBytes = 2U;
    tiling->totalTokenSlots = totalTokenSlots;
    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
static ge::graphStatus InferShapeForA5FullBlockDump(
    gert::InferShapeContext* context)
{
    if (context == nullptr ||
        context->GetInputShape(DRAM_K_ROPE) == nullptr ||
        context->GetInputShape(DRAM_KV_CACHE) == nullptr ||
        context->GetOutputShape(0) == nullptr ||
        context->GetOutputShape(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(DRAM_K_ROPE);
    *context->GetOutputShape(1) = *context->GetInputShape(DRAM_KV_CACHE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForA5FullBlockDump(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(
        0, context->GetInputDataType(DRAM_K_ROPE));
    context->SetOutputDataType(
        1, context->GetInputDataType(DRAM_KV_CACHE));
    return ge::GRAPH_SUCCESS;
}

class A5FullBlockDump : public OpDef {
public:
    explicit A5FullBlockDump(const char* name) : OpDef(name)
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
        this->Input("src_block_ids").ParamType(REQUIRED).DataType(intTypes).Format(intFormats);
        this->Input("dst_block_ids").ParamType(REQUIRED).DataType(intTypes).Format(intFormats);
        this->Output("dram_k_rope_out").ParamType(REQUIRED).DataType(dataTypes).Format(dataFormats);
        this->Output("dram_kv_cache_out").ParamType(REQUIRED).DataType(dataTypes).Format(dataFormats);

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
OP_ADD(A5FullBlockDump);

IMPL_OP_INFERSHAPE(A5FullBlockDump)
    .InferShape(InferShapeForA5FullBlockDump)
    .InferDataType(InferDataTypeForA5FullBlockDump);
} // namespace ops
