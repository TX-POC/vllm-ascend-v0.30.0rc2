/**
 * Host registration for Ascend 950 GLM-5.1 packed-C8 KV scatter copy.
 *
 * One opaque 656-byte row contains 512 FP8 latent bytes, 64 BF16 RoPE
 * elements (128 bytes), and four FP32 tile scales (16 bytes).  Attention
 * metadata is written into caller-owned buffers so its tail capacity can be
 * fixed by the engine capture shape instead of being hard-coded in the op.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../op_kernel/a5_packed_kvcache_scatter_copy_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t HBM_KV = 0;
constexpr size_t DRAM_KV = 1;
constexpr size_t HBM_BLOCK_TABLE = 2;
constexpr size_t DRAM_BLOCK_TABLE = 3;
constexpr size_t SOURCE_TOKEN_IDS = 4;
constexpr size_t DESTINATION_SLOTS = 5;
constexpr size_t COPY_COUNTS = 6;
constexpr size_t CACHE_TOKENS = 7;
constexpr size_t CANDIDATE_LENS = 8;
constexpr size_t ACTUAL_SEQ_LENGTHS_KV = 9;
constexpr size_t ATTENTION_SLOTS_BUFFER = 10;
constexpr size_t RESIDENT_SEQ_LENGTHS_BUFFER = 11;

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t KV_HEADS = 1;
constexpr int64_t PACKED_ROW_BYTES = 656;
constexpr int64_t COPY_CAP = 2048;
constexpr int64_t MAX_SOURCE_TOKENS = 1 << 18;

bool IsPackedCache(const gert::Shape &shape)
{
    return shape.GetDimNum() == 4 && shape.GetDim(0) > 0 &&
        shape.GetDim(1) == BLOCK_SIZE && shape.GetDim(2) == KV_HEADS &&
        shape.GetDim(3) == PACKED_ROW_BYTES;
}

bool GetCopyMetadataShape(
    const gert::Shape &shape,
    int64_t &batch,
    int64_t &capacity)
{
    if (shape.GetDimNum() == 2) {
        batch = shape.GetDim(0);
        capacity = shape.GetDim(1);
        return true;
    }
    if (shape.GetDimNum() == 3 && shape.GetDim(1) == 1) {
        batch = shape.GetDim(0);
        capacity = shape.GetDim(2);
        return true;
    }
    return false;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingA5PackedKvcacheScatterCopy(
    gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = HBM_KV;
         index <= RESIDENT_SEQ_LENGTHS_BUFFER; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }

    if (context->GetInputDesc(HBM_KV)->GetDataType() != ge::DT_INT8 ||
        context->GetInputDesc(DRAM_KV)->GetDataType() != ge::DT_INT8) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = HBM_BLOCK_TABLE;
         index <= RESIDENT_SEQ_LENGTHS_BUFFER; ++index) {
        if (context->GetInputDesc(index)->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape hbmKv =
        context->GetInputShape(HBM_KV)->GetStorageShape();
    const gert::Shape dramKv =
        context->GetInputShape(DRAM_KV)->GetStorageShape();
    const gert::Shape hbmTable =
        context->GetInputShape(HBM_BLOCK_TABLE)->GetStorageShape();
    const gert::Shape dramTable =
        context->GetInputShape(DRAM_BLOCK_TABLE)->GetStorageShape();
    const gert::Shape sourceIds =
        context->GetInputShape(SOURCE_TOKEN_IDS)->GetStorageShape();
    const gert::Shape destinationSlots =
        context->GetInputShape(DESTINATION_SLOTS)->GetStorageShape();
    const gert::Shape copyCounts =
        context->GetInputShape(COPY_COUNTS)->GetStorageShape();
    const gert::Shape cacheTokens =
        context->GetInputShape(CACHE_TOKENS)->GetStorageShape();
    const gert::Shape candidateLens =
        context->GetInputShape(CANDIDATE_LENS)->GetStorageShape();
    const gert::Shape actualKv =
        context->GetInputShape(ACTUAL_SEQ_LENGTHS_KV)->GetStorageShape();
    const gert::Shape attentionSlots =
        context->GetInputShape(ATTENTION_SLOTS_BUFFER)->GetStorageShape();
    const gert::Shape residentLengths =
        context->GetInputShape(RESIDENT_SEQ_LENGTHS_BUFFER)->GetStorageShape();

    int64_t sourceBatch = 0;
    int64_t sourceCapacity = 0;
    int64_t destinationBatch = 0;
    int64_t destinationCapacity = 0;
    if (!IsPackedCache(hbmKv) || !IsPackedCache(dramKv) ||
        hbmTable.GetDimNum() != 2 || dramTable.GetDimNum() != 2 ||
        copyCounts.GetDimNum() != 1 || cacheTokens.GetDimNum() != 1 ||
        candidateLens.GetDimNum() != 1 || actualKv.GetDimNum() != 1 ||
        attentionSlots.GetDimNum() != 3 ||
        attentionSlots.GetDim(1) != 1 ||
        residentLengths.GetDimNum() != 1 ||
        !GetCopyMetadataShape(sourceIds, sourceBatch, sourceCapacity) ||
        !GetCopyMetadataShape(
            destinationSlots, destinationBatch, destinationCapacity)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = copyCounts.GetDim(0);
    const int64_t attentionCapacity = attentionSlots.GetDim(2);
    if (batch <= 0 || sourceBatch != batch || destinationBatch != batch ||
        sourceCapacity != COPY_CAP || destinationCapacity != COPY_CAP ||
        cacheTokens.GetDim(0) != batch || candidateLens.GetDim(0) != batch ||
        actualKv.GetDim(0) != batch || attentionSlots.GetDim(0) != batch ||
        residentLengths.GetDim(0) != batch ||
        attentionCapacity < COPY_CAP ||
        attentionCapacity > MAX_SOURCE_TOKENS ||
        hbmTable.GetDim(0) != batch || dramTable.GetDim(0) != batch ||
        hbmTable.GetDim(1) <= 0 || dramTable.GetDim(1) <= 0 ||
        dramTable.GetDim(1) * BLOCK_SIZE > MAX_SOURCE_TOKENS) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aivCount == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t totalPairSlots =
        static_cast<uint64_t>(batch) * COPY_CAP;
    const uint32_t usedCoreNum = static_cast<uint32_t>(
        totalPairSlots < aivCount ? totalPairSlots : aivCount);
    auto *tiling = context->GetTilingData<
        A5PackedKvcacheScatterCopyTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->copyCap = COPY_CAP;
    tiling->hbmMaxBlockNum = static_cast<uint32_t>(hbmTable.GetDim(1));
    tiling->dramMaxBlockNum = static_cast<uint32_t>(dramTable.GetDim(1));
    tiling->packedRowBytes = PACKED_ROW_BYTES;
    tiling->attentionCapacity = static_cast<uint32_t>(attentionCapacity);
    tiling->totalPairSlots = totalPairSlots;
    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
static ge::graphStatus InferPackedScatterShape(
    gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(HBM_KV) == nullptr ||
        context->GetInputShape(ATTENTION_SLOTS_BUFFER) == nullptr ||
        context->GetInputShape(RESIDENT_SEQ_LENGTHS_BUFFER) == nullptr ||
        context->GetOutputShape(0) == nullptr ||
        context->GetOutputShape(1) == nullptr ||
        context->GetOutputShape(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(HBM_KV);
    *context->GetOutputShape(1) =
        *context->GetInputShape(ATTENTION_SLOTS_BUFFER);
    *context->GetOutputShape(2) =
        *context->GetInputShape(RESIDENT_SEQ_LENGTHS_BUFFER);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferPackedScatterDataType(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_INT8);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

class A5PackedKvcacheScatterCopy : public OpDef {
public:
    explicit A5PackedKvcacheScatterCopy(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> bytes = {ge::DT_INT8};
        const std::vector<ge::DataType> ints = {ge::DT_INT32};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};

        this->Input("hbm_kv").ParamType(REQUIRED).DataType(bytes).Format(formats);
        this->Input("dram_kv").ParamType(REQUIRED).DataType(bytes).Format(formats);
        this->Input("hbm_block_table").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("dram_block_table").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("source_token_ids").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("destination_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("copy_counts").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("cache_tokens").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("candidate_lens").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("actual_seq_lengths_kv").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("attention_slots_buffer").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("resident_seq_lengths_buffer").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("hbm_kv_out").ParamType(REQUIRED).DataType(bytes).Format(formats);
        this->Output("attention_slots_out").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("resident_seq_lengths_out").ParamType(REQUIRED).DataType(ints).Format(formats);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore()
            .SetTiling(optiling::TilingA5PackedKvcacheScatterCopy)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5PackedKvcacheScatterCopy);

IMPL_OP_INFERSHAPE(A5PackedKvcacheScatterCopy)
    .InferShape(InferPackedScatterShape)
    .InferDataType(InferPackedScatterDataType);
} // namespace ops
