/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "../op_kernel/a5_packed_full_block_dump_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t HBM_PACKED_KV = 0;
constexpr size_t DRAM_PACKED_KV = 1;
constexpr size_t SRC_BLOCK_IDS = 2;
constexpr size_t DST_BLOCK_IDS = 3;

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t PACKED_KV_DIM = 656;
constexpr uint64_t BLOCK_BYTES =
    static_cast<uint64_t>(BLOCK_SIZE) * PACKED_KV_DIM;
constexpr uint64_t COPY_CHUNK_BYTES = 16 * 1024;
constexpr uint64_t CHUNKS_PER_BLOCK =
    (BLOCK_BYTES + COPY_CHUNK_BYTES - 1) / COPY_CHUNK_BYTES;

static_assert(BLOCK_BYTES % 32 == 0,
              "packed full-block byte size must be 32-byte aligned");
static_assert(COPY_CHUNK_BYTES % 32 == 0,
              "packed dump chunk byte size must be 32-byte aligned");

bool HasPackedCacheShape(const gert::Shape& shape)
{
    return shape.GetDimNum() == 4 &&
           shape.GetDim(0) > 0 &&
           shape.GetDim(1) == BLOCK_SIZE &&
           shape.GetDim(2) == 1 &&
           shape.GetDim(3) == PACKED_KV_DIM;
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

    if (context->GetInputDesc(HBM_PACKED_KV)->GetDataType() != ge::DT_INT8 ||
        context->GetInputDesc(DRAM_PACKED_KV)->GetDataType() != ge::DT_INT8 ||
        context->GetInputDesc(SRC_BLOCK_IDS)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(DST_BLOCK_IDS)->GetDataType() != ge::DT_INT32) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape hbm =
        context->GetInputShape(HBM_PACKED_KV)->GetStorageShape();
    const gert::Shape dram =
        context->GetInputShape(DRAM_PACKED_KV)->GetStorageShape();
    const gert::Shape srcIds =
        context->GetInputShape(SRC_BLOCK_IDS)->GetStorageShape();
    const gert::Shape dstIds =
        context->GetInputShape(DST_BLOCK_IDS)->GetStorageShape();

    if (!HasPackedCacheShape(hbm) || !HasPackedCacheShape(dram) ||
        srcIds.GetDimNum() != 1 || dstIds.GetDimNum() != 1 ||
        srcIds.GetDim(0) <= 0 || srcIds.GetDim(0) != dstIds.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t jobCount = srcIds.GetDim(0);
    if (jobCount > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        hbm.GetDim(0) > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        dram.GetDim(0) > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCoreNum = platform.GetCoreNumAiv();
    if (aivCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    // Assign complete packed blocks, not individual chunks, to AIV cores.
    // A block is 83,968 bytes and is internally copied as five 16-KiB aligned
    // chunks plus one 2-KiB aligned tail. One core remains the sole writer for
    // all chunks of a destination swapped-memory block.
    const uint32_t usedCoreNum =
        static_cast<uint64_t>(jobCount) < aivCoreNum
            ? static_cast<uint32_t>(jobCount)
            : aivCoreNum;
    if (usedCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t totalCopyChunks =
        static_cast<uint64_t>(jobCount) * CHUNKS_PER_BLOCK;

    auto* tiling = context->GetTilingData<A5PackedFullBlockDumpTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->jobCount = static_cast<uint32_t>(jobCount);
    tiling->hbmBlockCount = static_cast<uint32_t>(hbm.GetDim(0));
    tiling->dramBlockCount = static_cast<uint32_t>(dram.GetDim(0));
    // Retained as an internal diagnostic/accounting field; kernel ownership is
    // job/block based and no longer schedules individual flat chunks.
    tiling->totalCopyChunks = totalCopyChunks;
    context->SetBlockDim(usedCoreNum);

    // One process-local marker makes it obvious from service logs that the
    // rebuilt block-owner implementation is actually loaded.  This prints
    // only once per worker process and has no steady-state cost.
    static bool reportedImplementation = false;
    if (!reportedImplementation) {
        std::fprintf(
            stderr,
            "[A5_PACKED_FULL_BLOCK_DUMP] implementation=block_owner_v2 "
            "block_bytes=%llu chunks_per_block=%llu used_cores=%u\n",
            static_cast<unsigned long long>(BLOCK_BYTES),
            static_cast<unsigned long long>(CHUNKS_PER_BLOCK),
            usedCoreNum);
        reportedImplementation = true;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
static ge::graphStatus InferShapeForA5PackedFullBlockDump(
    gert::InferShapeContext* context)
{
    if (context == nullptr ||
        context->GetInputShape(DRAM_PACKED_KV) == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(DRAM_PACKED_KV);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForA5PackedFullBlockDump(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(
        0, context->GetInputDataType(DRAM_PACKED_KV));
    return ge::GRAPH_SUCCESS;
}

class A5PackedFullBlockDump : public OpDef {
public:
    explicit A5PackedFullBlockDump(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> byteTypes = {ge::DT_INT8};
        const std::vector<ge::DataType> intTypes = {ge::DT_INT32};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};

        this->Input("hbm_packed_kv").ParamType(REQUIRED).DataType(byteTypes).Format(formats);
        this->Input("dram_packed_kv").ParamType(REQUIRED).DataType(byteTypes).Format(formats);
        this->Input("src_block_ids").ParamType(REQUIRED).DataType(intTypes).Format(formats);
        this->Input("dst_block_ids").ParamType(REQUIRED).DataType(intTypes).Format(formats);
        this->Output("dram_packed_kv_out").ParamType(REQUIRED).DataType(byteTypes).Format(formats);

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
OP_ADD(A5PackedFullBlockDump);

IMPL_OP_INFERSHAPE(A5PackedFullBlockDump)
    .InferShape(InferShapeForA5PackedFullBlockDump)
    .InferDataType(InferDataTypeForA5PackedFullBlockDump);
} // namespace ops
