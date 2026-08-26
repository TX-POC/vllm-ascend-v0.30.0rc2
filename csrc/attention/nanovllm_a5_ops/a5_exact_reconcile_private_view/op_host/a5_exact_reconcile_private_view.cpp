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
constexpr size_t kKvObjectSlotsSnapshot = 2;
constexpr size_t kCacheTokens = 3;
constexpr size_t kKvObjectLens = 4;
constexpr size_t kScoreSourceLens = 5;
constexpr size_t kKvObjectMap = 6;
constexpr size_t kPrivateSourceToSlot = 7;
constexpr size_t kPrivateReady = 8;
constexpr size_t kInputCount = 9;

constexpr size_t kKvObjectIds = 0;
constexpr size_t kDestinationSlots = 1;
constexpr size_t kMissCounts = 2;
constexpr size_t kPlacementDelta = 3;
constexpr size_t kPrivateViewDescriptors = 4;
constexpr size_t kReadyPrivateKvObjectIds = 5;
constexpr size_t kReadyPrivateSlots = 6;
constexpr size_t kReadyPrivateDestinationSlots = 7;
constexpr size_t kReadyPrivateCounts = 8;
constexpr size_t kFallbackKvObjectIds = 9;
constexpr size_t kFallbackDestinationSlots = 10;
constexpr size_t kFallbackCounts = 11;
constexpr size_t kOutputCount = 12;

constexpr int64_t kSparseCount = 2048;
constexpr int64_t kPlacementDeltaPlanes = 3;
constexpr int64_t kPrivateViewPlanes = 4;
constexpr int64_t kMaxSourceCapacity = 1 << 18;
constexpr size_t kAttrCmpRatio = 0;
constexpr size_t kAttrKvObjectDomain = 1;
constexpr int64_t kRawTokenDomain = 0;
constexpr int64_t kC8CompressedObjectDomain = 1;

bool IsValidKvObjectDomain(int64_t cmpRatio, int64_t kvObjectDomain)
{
    return (kvObjectDomain == kRawTokenDomain && cmpRatio == 1) ||
           (kvObjectDomain == kC8CompressedObjectDomain &&
            (cmpRatio == 1 || cmpRatio == 4 || cmpRatio == 128));
}

template <typename Context>
bool HasValidKvObjectDomain(Context* context)
{
    if (context == nullptr || context->GetAttrs() == nullptr) {
        return false;
    }
    auto* attrs = context->GetAttrs();
    const int64_t* cmpRatio = attrs->GetAttrPointer<int64_t>(kAttrCmpRatio);
    const int64_t* kvObjectDomain =
        attrs->GetAttrPointer<int64_t>(kAttrKvObjectDomain);
    return cmpRatio != nullptr && kvObjectDomain != nullptr &&
           IsValidKvObjectDomain(*cmpRatio, *kvObjectDomain);
}

bool IsVectorOfLength(const gert::Shape& shape, int64_t length)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == length;
}

// C8 emits one exact Top-K vector for every flattened leading output row.
// For example, decode commonly has [B, 1, 2048], while TND/MTP forms can
// have [T, N2, 2048] or [B, S1, N2, 2048].  The planner must keep that row
// cardinality rather than silently treating dim(0) as the request count.
bool GetExactTopkRowCount(const gert::Shape& shape, int64_t* rowCount)
{
    if (rowCount == nullptr || shape.GetDimNum() < 1 ||
        shape.GetDim(shape.GetDimNum() - 1) != kSparseCount) {
        return false;
    }
    int64_t rows = 1;
    for (size_t dim = 0; dim + 1 < shape.GetDimNum(); ++dim) {
        const int64_t value = shape.GetDim(dim);
        if (value <= 0 || rows > std::numeric_limits<int64_t>::max() / value) {
            return false;
        }
        rows *= value;
    }
    *rowCount = rows;
    return rows > 0;
}

bool IsPoolMatrix(const gert::Shape& shape)
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
    const gert::Shape kvObjectSlots =
        context->GetInputShape(kKvObjectSlotsSnapshot)->GetStorageShape();
    const gert::Shape budgets =
        context->GetInputShape(kCacheTokens)->GetStorageShape();
    const gert::Shape kvObjectLens =
        context->GetInputShape(kKvObjectLens)->GetStorageShape();
    const gert::Shape scoreSourceLens =
        context->GetInputShape(kScoreSourceLens)->GetStorageShape();
    const gert::Shape kvObjectMap =
        context->GetInputShape(kKvObjectMap)->GetStorageShape();
    const gert::Shape sourceToSlot =
        context->GetInputShape(kPrivateSourceToSlot)->GetStorageShape();
    const gert::Shape ready =
        context->GetInputShape(kPrivateReady)->GetStorageShape();

    // kv_object_slots_snapshot is the long-lived object directory. In
    // contrast, kv_object_map/private_source_to_slot are addressed by
    // C8's (possibly compressed) score-source index. The two address spaces
    // deliberately have independent capacities and per-row active lengths.
    int64_t rowCount = 0;
    if (!HasValidKvObjectDomain(context) ||
        !GetExactTopkRowCount(topk, &rowCount) || !IsPoolMatrix(kvObjectSlots) ||
        kvObjectSlots.GetDim(0) >
            static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        !IsPoolMatrix(kvObjectMap) ||
        kvObjectMap.GetDim(0) != kvObjectSlots.GetDim(0) ||
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
    if (!IsVectorOfLength(req, rowCount) ||
        !IsVectorOfLength(budgets, rowCount) ||
        !IsVectorOfLength(kvObjectLens, rowCount) ||
        !IsVectorOfLength(scoreSourceLens, rowCount) ||
        sourceToSlot.GetDimNum() != 2 || sourceToSlot.GetDim(0) != rowCount ||
        sourceToSlot.GetDim(1) != kvObjectMap.GetDim(1)) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aivCount == 0 || rowCount >
        static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t usedCoreNum = static_cast<uint32_t>(
        rowCount < static_cast<int64_t>(aivCount) ? rowCount : aivCount);
    auto* tiling =
        context->GetTilingData<A5ExactReconcilePrivateViewTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->rowCount = static_cast<uint32_t>(rowCount);
    tiling->poolSize = static_cast<uint32_t>(kvObjectSlots.GetDim(0));
    tiling->scoreSourceCapacity =
        static_cast<uint32_t>(kvObjectMap.GetDim(1));
    tiling->kvObjectCapacity =
        static_cast<uint32_t>(kvObjectSlots.GetDim(1));
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
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < kInputCount; ++index) {
        if (context->GetInputShape(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    const gert::Shape* topk = context->GetInputShape(kExactTopkIndices);
    const gert::Shape* req = context->GetInputShape(kReqPoolEntries);
    const gert::Shape* kvObjectSlots = context->GetInputShape(kKvObjectSlotsSnapshot);
    const gert::Shape* budgets = context->GetInputShape(kCacheTokens);
    const gert::Shape* kvObjectLens = context->GetInputShape(kKvObjectLens);
    const gert::Shape* scoreSourceLens =
        context->GetInputShape(kScoreSourceLens);
    const gert::Shape* kvObjectMap = context->GetInputShape(kKvObjectMap);
    const gert::Shape* sourceToSlot =
        context->GetInputShape(kPrivateSourceToSlot);
    const gert::Shape* ready = context->GetInputShape(kPrivateReady);
    int64_t rowCount = 0;
    if (!HasValidKvObjectDomain(context) ||
        !GetExactTopkRowCount(*topk, &rowCount) || !IsPoolMatrix(*kvObjectSlots) ||
        !IsPoolMatrix(*kvObjectMap) || kvObjectMap->GetDim(0) != kvObjectSlots->GetDim(0) ||
        ready->GetDimNum() != 2 || ready->GetDim(0) <= 0 ||
        ready->GetDim(1) <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (!IsVectorOfLength(*req, rowCount) ||
        !IsVectorOfLength(*budgets, rowCount) ||
        !IsVectorOfLength(*kvObjectLens, rowCount) ||
        !IsVectorOfLength(*scoreSourceLens, rowCount) ||
        sourceToSlot->GetDimNum() != 2 ||
        sourceToSlot->GetDim(0) != rowCount ||
        sourceToSlot->GetDim(1) != kvObjectMap->GetDim(1)) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < kOutputCount; ++index) {
        if (context->GetOutputShape(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    *context->GetOutputShape(kKvObjectIds) = *topk;
    *context->GetOutputShape(kDestinationSlots) = *topk;
    context->GetOutputShape(kMissCounts)->SetDimNum(1);
    context->GetOutputShape(kMissCounts)->SetDim(0, rowCount);
    context->GetOutputShape(kPlacementDelta)->SetDimNum(3);
    context->GetOutputShape(kPlacementDelta)->SetDim(0, rowCount);
    context->GetOutputShape(kPlacementDelta)->SetDim(1,
                                                      kPlacementDeltaPlanes);
    context->GetOutputShape(kPlacementDelta)->SetDim(2, kSparseCount);
    context->GetOutputShape(kPrivateViewDescriptors)->SetDimNum(3);
    context->GetOutputShape(kPrivateViewDescriptors)->SetDim(0, rowCount);
    context->GetOutputShape(kPrivateViewDescriptors)->SetDim(
        1, kPrivateViewPlanes);
    context->GetOutputShape(kPrivateViewDescriptors)->SetDim(2, kSparseCount);
    for (size_t output : {kReadyPrivateKvObjectIds, kReadyPrivateSlots,
                          kReadyPrivateDestinationSlots,
                          kFallbackKvObjectIds,
                          kFallbackDestinationSlots}) {
        *context->GetOutputShape(output) = *topk;
    }
    for (size_t output : {kReadyPrivateCounts, kFallbackCounts}) {
        context->GetOutputShape(output)->SetDimNum(1);
        context->GetOutputShape(output)->SetDim(0, rowCount);
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
        this->Input("kv_object_slots_snapshot").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("cache_tokens").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("kv_object_lens").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("score_source_lens").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("kv_object_map").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("private_source_to_slot").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("private_ready").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("kv_object_ids").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("destination_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("miss_counts").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("placement_delta").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("private_view_descriptors").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("ready_private_kv_object_ids").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("ready_private_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("ready_private_destination_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("ready_private_counts").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("fallback_kv_object_ids").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("fallback_destination_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("fallback_counts").ParamType(REQUIRED).DataType(ints).Format(formats);
        // Fail closed: the reconciler must know whether its directory object
        // IDs are raw tokens or C8 compressed QSFA objects.
        this->Attr("cmp_ratio").AttrType(REQUIRED).Int();
        this->Attr("kv_object_domain").AttrType(REQUIRED).Int();

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
