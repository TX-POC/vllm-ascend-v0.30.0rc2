/** Host registration, shape inference, and tiling for transaction-local P3 reconcile. */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "../op_kernel/a5_exact_reconcile_private_view_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t kExactTopkIndices = 0;
constexpr size_t kReqPoolEntries = 1;
constexpr size_t kCacheSlotsSnapshot = 2;
constexpr size_t kCacheTokens = 3;
constexpr size_t kCandidateLens = 4;
constexpr size_t kLogicalTokenMap = 5;
constexpr size_t kPrivateSourceToSlot = 6;
constexpr size_t kPrivateReady = 7;
constexpr size_t kInputCount = 8;

constexpr size_t kSourceIds = 0;
constexpr size_t kDestinationSlots = 1;
constexpr size_t kMissCounts = 2;
constexpr size_t kPlacementDelta = 3;
constexpr size_t kPrivateViewDescriptors = 4;
constexpr size_t kReadyPrivateSourceIds = 5;
constexpr size_t kReadyPrivateSlots = 6;
constexpr size_t kReadyPrivateDestinationSlots = 7;
constexpr size_t kReadyPrivateCounts = 8;
constexpr size_t kFallbackLogicalTokenIds = 9;
constexpr size_t kFallbackDestinationSlots = 10;
constexpr size_t kFallbackCounts = 11;
constexpr size_t kOutputCount = 12;

constexpr int64_t kSparseCount = 2048;
constexpr int64_t kPlacementDeltaPlanes = 3;
constexpr int64_t kPrivateViewPlanes = 4;
constexpr int64_t kMaxSourceCapacity = 1 << 18;

bool IsVectorOfLength(const gert::Shape& shape, int64_t length)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == length;
}

bool IsExactTopk(const gert::Shape& shape)
{
    return shape.GetDimNum() == 3 && shape.GetDim(0) > 0 &&
           shape.GetDim(1) == 1 && shape.GetDim(2) == kSparseCount;
}

bool IsPoolSourceMatrix(const gert::Shape& shape)
{
    return shape.GetDimNum() == 2 && shape.GetDim(0) > 0 &&
           shape.GetDim(1) > 0 && shape.GetDim(1) <= kMaxSourceCapacity;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingA5ExactReconcilePrivateView(
    gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < kInputCount; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr ||
            context->GetInputDesc(index)->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape topk =
        context->GetInputShape(kExactTopkIndices)->GetStorageShape();
    const gert::Shape req =
        context->GetInputShape(kReqPoolEntries)->GetStorageShape();
    const gert::Shape snapshot =
        context->GetInputShape(kCacheSlotsSnapshot)->GetStorageShape();
    const gert::Shape budgets =
        context->GetInputShape(kCacheTokens)->GetStorageShape();
    const gert::Shape candidateLens =
        context->GetInputShape(kCandidateLens)->GetStorageShape();
    const gert::Shape logicalMap =
        context->GetInputShape(kLogicalTokenMap)->GetStorageShape();
    const gert::Shape sourceToSlot =
        context->GetInputShape(kPrivateSourceToSlot)->GetStorageShape();
    const gert::Shape ready =
        context->GetInputShape(kPrivateReady)->GetStorageShape();

    if (!IsExactTopk(topk) || !IsPoolSourceMatrix(snapshot) ||
        snapshot.GetDim(0) >
            static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        logicalMap.GetDimNum() != 2 ||
        logicalMap.GetDim(0) != snapshot.GetDim(0) ||
        logicalMap.GetDim(1) != snapshot.GetDim(1) ||
        ready.GetDimNum() != 2 || ready.GetDim(0) <= 0 ||
        ready.GetDim(1) <= 0 ||
        ready.GetDim(0) >
            static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        ready.GetDim(1) >
            static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        static_cast<uint64_t>(ready.GetDim(0)) *
                static_cast<uint64_t>(ready.GetDim(1)) >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = topk.GetDim(0);
    if (!IsVectorOfLength(req, batch) || !IsVectorOfLength(budgets, batch) ||
        !IsVectorOfLength(candidateLens, batch) ||
        sourceToSlot.GetDimNum() != 2 || sourceToSlot.GetDim(0) != batch ||
        sourceToSlot.GetDim(1) != snapshot.GetDim(1)) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aivCount == 0 || batch >
        static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t usedCoreNum = static_cast<uint32_t>(
        batch < static_cast<int64_t>(aivCount) ? batch : aivCount);
    auto* tiling =
        context->GetTilingData<A5ExactReconcilePrivateViewTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->poolSize = static_cast<uint32_t>(snapshot.GetDim(0));
    tiling->sourceCapacity = static_cast<uint32_t>(snapshot.GetDim(1));
    tiling->privateQueueCount = static_cast<uint32_t>(ready.GetDim(0));
    tiling->privateRowsPerQueue = static_cast<uint32_t>(ready.GetDim(1));
    context->SetBlockDim(usedCoreNum);
    if (context->GetWorkspaceSizes(1) != nullptr) {
        context->GetWorkspaceSizes(1)[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferA5ExactReconcilePrivateViewShape(
    gert::InferShapeContext* context)
{
    if (context == nullptr ||
        context->GetInputShape(kExactTopkIndices) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* topk = context->GetInputShape(kExactTopkIndices);
    if (!IsExactTopk(*topk)) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < kOutputCount; ++index) {
        if (context->GetOutputShape(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    const int64_t batch = topk->GetDim(0);
    *context->GetOutputShape(kSourceIds) = *topk;
    *context->GetOutputShape(kDestinationSlots) = *topk;
    context->GetOutputShape(kMissCounts)->SetDimNum(1);
    context->GetOutputShape(kMissCounts)->SetDim(0, batch);
    context->GetOutputShape(kPlacementDelta)->SetDimNum(3);
    context->GetOutputShape(kPlacementDelta)->SetDim(0, batch);
    context->GetOutputShape(kPlacementDelta)->SetDim(1,
                                                      kPlacementDeltaPlanes);
    context->GetOutputShape(kPlacementDelta)->SetDim(2, kSparseCount);
    context->GetOutputShape(kPrivateViewDescriptors)->SetDimNum(3);
    context->GetOutputShape(kPrivateViewDescriptors)->SetDim(0, batch);
    context->GetOutputShape(kPrivateViewDescriptors)->SetDim(
        1, kPrivateViewPlanes);
    context->GetOutputShape(kPrivateViewDescriptors)->SetDim(2, kSparseCount);
    for (size_t output : {kReadyPrivateSourceIds, kReadyPrivateSlots,
                          kReadyPrivateDestinationSlots,
                          kFallbackLogicalTokenIds,
                          kFallbackDestinationSlots}) {
        *context->GetOutputShape(output) = *topk;
    }
    for (size_t output : {kReadyPrivateCounts, kFallbackCounts}) {
        context->GetOutputShape(output)->SetDimNum(1);
        context->GetOutputShape(output)->SetDim(0, batch);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferA5ExactReconcilePrivateViewDataType(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < kOutputCount; ++index) {
        context->SetOutputDataType(index, ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}

class A5ExactReconcilePrivateView : public OpDef {
public:
    explicit A5ExactReconcilePrivateView(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> ints = {ge::DT_INT32};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        this->Input("exact_topk_indices").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("req_pool_entries").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("cache_slots_snapshot").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("cache_tokens").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("candidate_lens").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("logical_token_map").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("private_source_to_slot").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("private_ready").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("source_ids").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("destination_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("miss_counts").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("placement_delta").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("private_view_descriptors").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("ready_private_source_ids").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("ready_private_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("ready_private_destination_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("ready_private_counts").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("fallback_logical_token_ids").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("fallback_destination_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("fallback_counts").ParamType(REQUIRED).DataType(ints).Format(formats);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore()
            .SetTiling(optiling::TilingA5ExactReconcilePrivateView)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5ExactReconcilePrivateView);

IMPL_OP_INFERSHAPE(A5ExactReconcilePrivateView)
    .InferShape(InferA5ExactReconcilePrivateViewShape)
    .InferDataType(InferA5ExactReconcilePrivateViewDataType);
}  // namespace ops
