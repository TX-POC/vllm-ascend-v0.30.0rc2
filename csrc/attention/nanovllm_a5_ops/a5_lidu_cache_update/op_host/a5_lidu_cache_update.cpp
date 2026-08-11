/** Host definition, shape inference, and tiling for A5 C8 LIDU state update. */

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../op_kernel/a5_lidu_cache_update_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t TOPK_INDICES = 0;
constexpr size_t REQ_POOL_ENTRIES = 1;
constexpr size_t CACHE_SLOTS_POOL = 2;
constexpr size_t CACHE_TOKENS = 3;
constexpr size_t CANDIDATE_LENS = 4;
constexpr int64_t SPARSE_COUNT = 2048;
constexpr int64_t MAX_SOURCE_CAPACITY = 1 << 18;
} // namespace

namespace optiling {
static ge::graphStatus TilingA5LiduCacheUpdate(gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = TOPK_INDICES; index <= CANDIDATE_LENS; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr ||
            context->GetInputDesc(index)->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape topk =
        context->GetInputShape(TOPK_INDICES)->GetStorageShape();
    const gert::Shape req =
        context->GetInputShape(REQ_POOL_ENTRIES)->GetStorageShape();
    const gert::Shape pool =
        context->GetInputShape(CACHE_SLOTS_POOL)->GetStorageShape();
    const gert::Shape budgets =
        context->GetInputShape(CACHE_TOKENS)->GetStorageShape();
    const gert::Shape lengths =
        context->GetInputShape(CANDIDATE_LENS)->GetStorageShape();
    if (topk.GetDimNum() != 3 || topk.GetDim(0) <= 0 ||
        topk.GetDim(1) != 1 || topk.GetDim(2) != SPARSE_COUNT ||
        req.GetDimNum() != 1 || budgets.GetDimNum() != 1 ||
        lengths.GetDimNum() != 1 || pool.GetDimNum() != 2 ||
        pool.GetDim(0) <= 0 || pool.GetDim(1) <= 0 ||
        pool.GetDim(1) > MAX_SOURCE_CAPACITY) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = topk.GetDim(0);
    if (req.GetDim(0) != batch || budgets.GetDim(0) != batch ||
        lengths.GetDim(0) != batch) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aivCount == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t usedCoreNum = static_cast<uint32_t>(
        batch < static_cast<int64_t>(aivCount) ? batch : aivCount);
    auto *tiling =
        context->GetTilingData<A5LiduCacheUpdateTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->poolSize = static_cast<uint32_t>(pool.GetDim(0));
    tiling->sourceCapacity = static_cast<uint32_t>(pool.GetDim(1));
    context->SetBlockDim(usedCoreNum);
    if (context->GetWorkspaceSizes(1) != nullptr) {
        context->GetWorkspaceSizes(1)[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
static ge::graphStatus InferA5LiduCacheUpdateShape(
    gert::InferShapeContext *context)
{
    if (context == nullptr ||
        context->GetInputShape(TOPK_INDICES) == nullptr ||
        context->GetInputShape(CACHE_SLOTS_POOL) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *topk = context->GetInputShape(TOPK_INDICES);
    const gert::Shape *pool = context->GetInputShape(CACHE_SLOTS_POOL);
    for (size_t index = 0; index < 4; ++index) {
        if (context->GetOutputShape(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    *context->GetOutputShape(0) = *topk;
    *context->GetOutputShape(1) = *topk;
    context->GetOutputShape(2)->SetDimNum(1);
    context->GetOutputShape(2)->SetDim(0, topk->GetDim(0));
    *context->GetOutputShape(3) = *pool;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferA5LiduCacheUpdateDataType(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < 4; ++index) {
        context->SetOutputDataType(index, ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}

class A5LiduCacheUpdate : public OpDef {
public:
    explicit A5LiduCacheUpdate(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> ints = {ge::DT_INT32};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        this->Input("topk_indices").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("req_pool_entries").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("cache_slots_pool").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("cache_tokens").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("candidate_lens").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("source_ids").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("destination_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("miss_counts").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("cache_slots_alias").ParamType(REQUIRED).DataType(ints).Format(formats);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore()
            .SetTiling(optiling::TilingA5LiduCacheUpdate)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5LiduCacheUpdate);

IMPL_OP_INFERSHAPE(A5LiduCacheUpdate)
    .InferShape(InferA5LiduCacheUpdateShape)
    .InferDataType(InferA5LiduCacheUpdateDataType);
} // namespace ops
