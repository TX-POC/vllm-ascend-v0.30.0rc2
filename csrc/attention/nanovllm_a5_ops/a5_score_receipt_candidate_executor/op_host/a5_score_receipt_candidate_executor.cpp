/**
 * P7 stage-1 host contract.
 *
 * This is deliberately an independent device executor.  It receives the
 * immutable score-GM / receipt publication from C8; it neither includes C8
 * source nor creates a new C8 UB/event dependency.  Every launched AIV is
 * an identical receipt worker.  There is no reserved Get group, no 31+1
 * partition, and no claim of physical-core affinity.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "../op_kernel/a5_score_receipt_candidate_executor_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t kScoreWorkspace = 0;
constexpr size_t kTileReceipts = 1;
constexpr size_t kStreamingEpoch = 2;
constexpr size_t kScoreSourceLens = 3;
constexpr size_t kResidentThreshold = 4;
constexpr size_t kResidentThresholdValid = 5;
constexpr size_t kKvObjectSlotsSnapshot = 6;
constexpr size_t kReqPoolEntries = 7;
constexpr size_t kKvObjectMap = 8;
constexpr size_t kKvObjectLens = 9;
constexpr size_t kInputCount = 10;

constexpr size_t kCandidateSeeds = 0;
constexpr size_t kCandidateCounts = 1;
constexpr size_t kCandidateReadyEpoch = 2;
constexpr size_t kReceiptStatus = 3;
constexpr size_t kReceiptOwner = 4;

constexpr uint32_t kReceiptWords = 16U;
constexpr uint32_t kScoreTileWidth = 128U;
constexpr uint32_t kCandidateSeedWords = 4U;

bool IsPositiveVector(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) > 0;
}

bool IsPositiveMatrix(const gert::Shape& shape)
{
    return shape.GetDimNum() == 2 && shape.GetDim(0) > 0 &&
           shape.GetDim(1) > 0;
}

bool GetGeometry(const gert::Shape& scoreWorkspace,
                 const gert::Shape& tileReceipts,
                 const gert::Shape& streamingEpoch,
                 const gert::Shape& scoreSourceLens,
                 const gert::Shape& residentThreshold,
                 const gert::Shape& residentThresholdValid,
                 const gert::Shape& kvObjectSlotsSnapshot,
                 const gert::Shape& reqPoolEntries,
                 const gert::Shape& kvObjectMap,
                 const gert::Shape& kvObjectLens,
                 uint32_t* rowCount, uint32_t* sourceCapacity,
                 uint32_t* poolSize, uint32_t* kvObjectCapacity,
                 uint32_t* tilesPerRow, uint32_t* tileReceiptCount,
                 uint32_t* scoreWorkspaceWords)
{
    if (rowCount == nullptr || sourceCapacity == nullptr || poolSize == nullptr ||
        kvObjectCapacity == nullptr || tilesPerRow == nullptr ||
        tileReceiptCount == nullptr || scoreWorkspaceWords == nullptr ||
        !IsPositiveVector(scoreWorkspace) || !IsPositiveVector(streamingEpoch) ||
        streamingEpoch.GetDim(0) != 1 || !IsPositiveVector(scoreSourceLens) ||
        !IsPositiveVector(residentThreshold) ||
        !IsPositiveVector(residentThresholdValid) ||
        !IsPositiveVector(reqPoolEntries) || !IsPositiveVector(kvObjectLens) ||
        !IsPositiveMatrix(kvObjectSlotsSnapshot) ||
        !IsPositiveMatrix(kvObjectMap) || tileReceipts.GetDimNum() != 2 ||
        tileReceipts.GetDim(1) != kReceiptWords ||
        scoreSourceLens.GetDim(0) != residentThreshold.GetDim(0) ||
        scoreSourceLens.GetDim(0) != residentThresholdValid.GetDim(0) ||
        scoreSourceLens.GetDim(0) != reqPoolEntries.GetDim(0) ||
        scoreSourceLens.GetDim(0) != kvObjectLens.GetDim(0) ||
        kvObjectSlotsSnapshot.GetDim(0) != kvObjectMap.GetDim(0)) {
        return false;
    }
    const int64_t rows = scoreSourceLens.GetDim(0);
    const int64_t sources = kvObjectMap.GetDim(1);
    const int64_t pools = kvObjectMap.GetDim(0);
    const int64_t objects = kvObjectSlotsSnapshot.GetDim(1);
    const int64_t workspaceWords = scoreWorkspace.GetDim(0);
    if (rows <= 0 || sources <= 0 || pools <= 0 || objects <= 0 ||
        workspaceWords <= 0 || rows > std::numeric_limits<uint32_t>::max() ||
        sources > std::numeric_limits<uint32_t>::max() ||
        pools > std::numeric_limits<uint32_t>::max() ||
        objects > std::numeric_limits<uint32_t>::max() ||
        workspaceWords > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const uint64_t tileCount =
        (static_cast<uint64_t>(sources) + kScoreTileWidth - 1U) /
        kScoreTileWidth;
    const uint64_t receiptCount = static_cast<uint64_t>(rows) * tileCount;
    const uint64_t scoreRowStride =
        tileCount * static_cast<uint64_t>(kScoreTileWidth);
    const uint64_t minimumWords =
        (static_cast<uint64_t>(rows) * scoreRowStride + 1U) / 2U;
    if (tileCount == 0U || receiptCount == 0U ||
        tileCount > std::numeric_limits<uint32_t>::max() ||
        receiptCount > std::numeric_limits<uint32_t>::max() ||
        tileReceipts.GetDim(0) != static_cast<int64_t>(receiptCount) ||
        static_cast<uint64_t>(workspaceWords) < minimumWords) {
        return false;
    }
    *rowCount = static_cast<uint32_t>(rows);
    *sourceCapacity = static_cast<uint32_t>(sources);
    *poolSize = static_cast<uint32_t>(pools);
    *kvObjectCapacity = static_cast<uint32_t>(objects);
    *tilesPerRow = static_cast<uint32_t>(tileCount);
    *tileReceiptCount = static_cast<uint32_t>(receiptCount);
    *scoreWorkspaceWords = static_cast<uint32_t>(workspaceWords);
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingA5ScoreReceiptCandidateExecutor(
    gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < kInputCount; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    const ge::DataType expectedTypes[kInputCount] = {
        ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32,
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        ge::DT_INT32, ge::DT_INT32};
    for (size_t index = 0; index < kInputCount; ++index) {
        if (context->GetInputDesc(index)->GetDataType() != expectedTypes[index]) {
            return ge::GRAPH_FAILED;
        }
    }
    const gert::Shape scoreWorkspace =
        context->GetInputShape(kScoreWorkspace)->GetStorageShape();
    const gert::Shape tileReceipts =
        context->GetInputShape(kTileReceipts)->GetStorageShape();
    const gert::Shape streamingEpoch =
        context->GetInputShape(kStreamingEpoch)->GetStorageShape();
    const gert::Shape scoreSourceLens =
        context->GetInputShape(kScoreSourceLens)->GetStorageShape();
    const gert::Shape residentThreshold =
        context->GetInputShape(kResidentThreshold)->GetStorageShape();
    const gert::Shape residentThresholdValid =
        context->GetInputShape(kResidentThresholdValid)->GetStorageShape();
    const gert::Shape kvObjectSlotsSnapshot =
        context->GetInputShape(kKvObjectSlotsSnapshot)->GetStorageShape();
    const gert::Shape reqPoolEntries =
        context->GetInputShape(kReqPoolEntries)->GetStorageShape();
    const gert::Shape kvObjectMap =
        context->GetInputShape(kKvObjectMap)->GetStorageShape();
    const gert::Shape kvObjectLens =
        context->GetInputShape(kKvObjectLens)->GetStorageShape();

    uint32_t rowCount = 0U;
    uint32_t sourceCapacity = 0U;
    uint32_t poolSize = 0U;
    uint32_t kvObjectCapacity = 0U;
    uint32_t tilesPerRow = 0U;
    uint32_t tileReceiptCount = 0U;
    uint32_t scoreWorkspaceWords = 0U;
    if (!GetGeometry(scoreWorkspace, tileReceipts, streamingEpoch,
                     scoreSourceLens, residentThreshold,
                     residentThresholdValid, kvObjectSlotsSnapshot,
                     reqPoolEntries, kvObjectMap, kvObjectLens, &rowCount,
                     &sourceCapacity, &poolSize, &kvObjectCapacity,
                     &tilesPerRow, &tileReceiptCount, &scoreWorkspaceWords)) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t availableAiv = platform.GetCoreNumAiv();
    if (availableAiv == 0U) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t workerCount =
        availableAiv < tileReceiptCount ? availableAiv : tileReceiptCount;
    auto* tiling = context->GetTilingData<
        A5ScoreReceiptCandidateExecutorTilingData>();
    if (tiling == nullptr || workerCount == 0U) {
        return ge::GRAPH_FAILED;
    }
    tiling->workerCount = workerCount;
    tiling->rowCount = rowCount;
    tiling->sourceCapacity = sourceCapacity;
    tiling->poolSize = poolSize;
    tiling->kvObjectCapacity = kvObjectCapacity;
    tiling->tilesPerRow = tilesPerRow;
    tiling->tileReceiptCount = tileReceiptCount;
    tiling->scoreWorkspaceWords = scoreWorkspaceWords;
    context->SetBlockDim(workerCount);
    if (context->GetWorkspaceSizes(1) != nullptr) {
        context->GetWorkspaceSizes(1)[0] = 0U;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferA5ScoreReceiptCandidateExecutorShape(
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
    const gert::Shape scoreWorkspace =
        *context->GetInputShape(kScoreWorkspace);
    const gert::Shape tileReceipts = *context->GetInputShape(kTileReceipts);
    const gert::Shape streamingEpoch =
        *context->GetInputShape(kStreamingEpoch);
    const gert::Shape scoreSourceLens =
        *context->GetInputShape(kScoreSourceLens);
    const gert::Shape residentThreshold =
        *context->GetInputShape(kResidentThreshold);
    const gert::Shape residentThresholdValid =
        *context->GetInputShape(kResidentThresholdValid);
    const gert::Shape kvObjectSlotsSnapshot =
        *context->GetInputShape(kKvObjectSlotsSnapshot);
    const gert::Shape reqPoolEntries =
        *context->GetInputShape(kReqPoolEntries);
    const gert::Shape kvObjectMap = *context->GetInputShape(kKvObjectMap);
    const gert::Shape kvObjectLens = *context->GetInputShape(kKvObjectLens);
    uint32_t rowCount = 0U;
    uint32_t sourceCapacity = 0U;
    uint32_t poolSize = 0U;
    uint32_t kvObjectCapacity = 0U;
    uint32_t tilesPerRow = 0U;
    uint32_t receiptCount = 0U;
    uint32_t scoreWorkspaceWords = 0U;
    if (!GetGeometry(scoreWorkspace, tileReceipts, streamingEpoch,
                     scoreSourceLens, residentThreshold,
                     residentThresholdValid, kvObjectSlotsSnapshot,
                     reqPoolEntries, kvObjectMap, kvObjectLens, &rowCount,
                     &sourceCapacity, &poolSize, &kvObjectCapacity,
                     &tilesPerRow, &receiptCount, &scoreWorkspaceWords)) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetOutputShape(kCandidateSeeds) == nullptr ||
        context->GetOutputShape(kCandidateCounts) == nullptr ||
        context->GetOutputShape(kCandidateReadyEpoch) == nullptr ||
        context->GetOutputShape(kReceiptStatus) == nullptr ||
        context->GetOutputShape(kReceiptOwner) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->GetOutputShape(kCandidateSeeds)->SetDimNum(3);
    context->GetOutputShape(kCandidateSeeds)->SetDim(0, receiptCount);
    context->GetOutputShape(kCandidateSeeds)->SetDim(1, kScoreTileWidth);
    context->GetOutputShape(kCandidateSeeds)->SetDim(2, kCandidateSeedWords);
    for (size_t index : {kCandidateCounts, kCandidateReadyEpoch,
                         kReceiptStatus, kReceiptOwner}) {
        context->GetOutputShape(index)->SetDimNum(1);
        context->GetOutputShape(index)->SetDim(0, receiptCount);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferA5ScoreReceiptCandidateExecutorDataType(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(kCandidateSeeds, ge::DT_INT32);
    context->SetOutputDataType(kCandidateCounts, ge::DT_INT32);
    context->SetOutputDataType(kCandidateReadyEpoch, ge::DT_INT64);
    context->SetOutputDataType(kReceiptStatus, ge::DT_INT32);
    context->SetOutputDataType(kReceiptOwner, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

class A5ScoreReceiptCandidateExecutor : public OpDef {
public:
    explicit A5ScoreReceiptCandidateExecutor(const char* name) : OpDef(name)
    {
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        const std::vector<ge::DataType> int32Types = {ge::DT_INT32};
        const std::vector<ge::DataType> int64Types = {ge::DT_INT64};
        this->Input("score_workspace").ParamType(REQUIRED)
            .DataType(int32Types).Format(formats);
        this->Input("tile_receipts").ParamType(REQUIRED)
            .DataType(int64Types).Format(formats);
        this->Input("streaming_epoch").ParamType(REQUIRED)
            .DataType(int64Types).Format(formats);
        for (const char* input : {"score_source_lens", "resident_threshold",
                                  "resident_threshold_valid",
                                  "kv_object_slots_snapshot",
                                  "req_pool_entries", "kv_object_map",
                                  "kv_object_lens"}) {
            this->Input(input).ParamType(REQUIRED)
                .DataType(int32Types).Format(formats);
        }
        this->Output("candidate_seeds").ParamType(REQUIRED)
            .DataType(int32Types).Format(formats);
        this->Output("candidate_counts").ParamType(REQUIRED)
            .DataType(int32Types).Format(formats);
        this->Output("candidate_ready_epoch").ParamType(REQUIRED)
            .DataType(int64Types).Format(formats);
        this->Output("receipt_status").ParamType(REQUIRED)
            .DataType(int32Types).Format(formats);
        this->Output("receipt_owner").ParamType(REQUIRED)
            .DataType(int32Types).Format(formats);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value", "a5_score_receipt_candidate_executor")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore()
            .SetTiling(optiling::TilingA5ScoreReceiptCandidateExecutor)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5ScoreReceiptCandidateExecutor);

IMPL_OP_INFERSHAPE(A5ScoreReceiptCandidateExecutor)
    .InferShape(InferA5ScoreReceiptCandidateExecutorShape)
    .InferDataType(InferA5ScoreReceiptCandidateExecutorDataType);
}  // namespace ops
