/**
 * Registration/tiling for the first real ExactScan + SpeculativeGet path.
 *
 * The P1 full-Get path contains no AIC work.  It therefore launches a single
 * 64-lane AIV task:
 *
 *   task lanes 0..1:  the two existing KVAF Direct-HCCP workers
 *   task lanes 2..63: ExactScan producers
 *
 * This preserves the task-lane positions used by the verified standalone
 * two-worker Get launch.  It is stronger than the old MIX logical layout but
 * is deliberately not described as a public physical-AIV affinity API.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "../op_kernel/a5_exact_scan_speculative_get_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t kLiScores = 0;
constexpr size_t kThreshold = 1;
constexpr size_t kResident = 2;
constexpr size_t kTableIndices = 3;
constexpr size_t kOffsets = 4;
constexpr size_t kOutputIndices = 5;
constexpr size_t kAddressTable = 6;
constexpr size_t kValidTable = 7;
constexpr size_t kPackedOutput = 8;
constexpr size_t kFusedGetControl = 9;
constexpr size_t kFrameEpoch = 10;
constexpr size_t kPipelineState = 11;

constexpr size_t kPackedOutputOut = 0;
constexpr size_t kCandidateIndices = 1;
constexpr size_t kCandidateKind = 2;
constexpr size_t kProducerStats = 3;
constexpr size_t kGetStats = 4;
constexpr size_t kRoleTrace = 5;
constexpr size_t kGetIndices = 6;
constexpr size_t kGetOffsets = 7;
constexpr size_t kGetOutputIndices = 8;
constexpr size_t kGetMask = 9;
constexpr size_t kPipelineStateOut = 10;

constexpr uint32_t kAivCount = 64;
constexpr uint32_t kGetLaneCount = 2;
constexpr uint32_t kProducerCount = kAivCount - kGetLaneCount;
constexpr uint32_t kCacheLineWords = 8;
constexpr uint32_t kRingSlots = 2;
constexpr uint32_t kRecordsPerRingSlot = 32;
constexpr uint32_t kRecordWords = 3;
constexpr uint32_t kRingSlotRecordWord = 2 * kCacheLineWords;
constexpr uint32_t kRingWordsPerSlot =
    kRingSlotRecordWord + kRecordsPerRingSlot * kRecordWords;
constexpr uint32_t kPipelineWordsPerRing =
    kRingSlots * kRingWordsPerSlot + kCacheLineWords;
constexpr uint32_t kGetBootstrapRecords = 64;
constexpr uint32_t kGetWaveRecords = 256;
constexpr uint32_t kFusedGetControlWords = 16;
constexpr uint32_t kPackedEntryBytes = 656;
constexpr uint32_t kProducerStatColumns = 7;
constexpr uint32_t kGetStatColumns = 8;

bool IsVectorWithPositiveLength(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) > 0;
}

bool IsVectorOfLength(const gert::Shape& shape, int64_t length)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == length;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingA5ExactScanSpeculativeGet(
    gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t input = 0; input <= kPipelineState; ++input) {
        if (context->GetInputShape(input) == nullptr ||
            context->GetInputDesc(input) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape scores =
        context->GetInputShape(kLiScores)->GetStorageShape();
    const gert::Shape threshold =
        context->GetInputShape(kThreshold)->GetStorageShape();
    const gert::Shape resident =
        context->GetInputShape(kResident)->GetStorageShape();
    const gert::Shape tableIndices =
        context->GetInputShape(kTableIndices)->GetStorageShape();
    const gert::Shape offsets =
        context->GetInputShape(kOffsets)->GetStorageShape();
    const gert::Shape outputIndices =
        context->GetInputShape(kOutputIndices)->GetStorageShape();
    const gert::Shape addressTable =
        context->GetInputShape(kAddressTable)->GetStorageShape();
    const gert::Shape validTable =
        context->GetInputShape(kValidTable)->GetStorageShape();
    const gert::Shape packedOutput =
        context->GetInputShape(kPackedOutput)->GetStorageShape();
    const gert::Shape control =
        context->GetInputShape(kFusedGetControl)->GetStorageShape();
    const gert::Shape epoch =
        context->GetInputShape(kFrameEpoch)->GetStorageShape();
    const gert::Shape pipelineState =
        context->GetInputShape(kPipelineState)->GetStorageShape();

    if (context->GetInputDesc(kLiScores)->GetDataType() != ge::DT_FLOAT ||
        context->GetInputDesc(kThreshold)->GetDataType() != ge::DT_FLOAT ||
        context->GetInputDesc(kResident)->GetDataType() != ge::DT_UINT8 ||
        context->GetInputDesc(kTableIndices)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(kOffsets)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(kOutputIndices)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(kAddressTable)->GetDataType() != ge::DT_INT64 ||
        context->GetInputDesc(kValidTable)->GetDataType() != ge::DT_UINT8 ||
        context->GetInputDesc(kPackedOutput)->GetDataType() != ge::DT_UINT8 ||
        context->GetInputDesc(kFusedGetControl)->GetDataType() != ge::DT_INT64 ||
        context->GetInputDesc(kFrameEpoch)->GetDataType() != ge::DT_INT64 ||
        context->GetInputDesc(kPipelineState)->GetDataType() != ge::DT_INT64) {
        return ge::GRAPH_FAILED;
    }

    if (!IsVectorWithPositiveLength(scores) ||
        scores.GetDim(0) < static_cast<int64_t>(kProducerCount) ||
        !IsVectorOfLength(threshold, 1) ||
        !IsVectorOfLength(resident, scores.GetDim(0)) ||
        !IsVectorOfLength(tableIndices, scores.GetDim(0)) ||
        !IsVectorOfLength(offsets, scores.GetDim(0)) ||
        !IsVectorOfLength(outputIndices, scores.GetDim(0)) ||
        !IsVectorWithPositiveLength(addressTable) ||
        !IsVectorOfLength(validTable, addressTable.GetDim(0)) ||
        packedOutput.GetDimNum() != 2 || packedOutput.GetDim(0) <= 0 ||
        packedOutput.GetDim(1) != static_cast<int64_t>(kPackedEntryBytes) ||
        !IsVectorOfLength(control, kFusedGetControlWords) ||
        !IsVectorOfLength(epoch, 1) ||
        pipelineState.GetDimNum() != 2 ||
        pipelineState.GetDim(0) !=
            static_cast<int64_t>(kProducerCount * kGetLaneCount) ||
        pipelineState.GetDim(1) !=
            static_cast<int64_t>(kPipelineWordsPerRing) ||
        scores.GetDim(0) >
            static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        addressTable.GetDim(0) >
            static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        packedOutput.GetDim(0) >
            static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aivCount != kAivCount) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t aivOnlyBlockDim =
        platform.CalcTschBlockDim(aivCount, 0U, aivCount);
    if (aivOnlyBlockDim != kAivCount) {
        return ge::GRAPH_FAILED;
    }
    // Task lanes 0/1 are persistent Get consumers while lanes 2..63 publish
    // their SPSC rings.  Batch mode is the launch contract that starts the
    // whole cooperative AIV set together; default wave scheduling can run the
    // consumers first and starve producers they are waiting for.
    if (context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<A5ExactScanSpeculativeGetTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->scoreCount = static_cast<uint32_t>(scores.GetDim(0));
    tiling->tableCapacity = static_cast<uint32_t>(addressTable.GetDim(0));
    tiling->outputRows = static_cast<uint32_t>(packedOutput.GetDim(0));
    tiling->controlWords = kFusedGetControlWords;
    tiling->producerCount = kProducerCount;
    tiling->getLaneCount = kGetLaneCount;
    tiling->ringSlots = kRingSlots;
    tiling->recordsPerRingSlot = kRecordsPerRingSlot;
    tiling->getBootstrapRecords = kGetBootstrapRecords;
    tiling->getWaveRecords = kGetWaveRecords;
    tiling->pipelineWordsPerRing = kPipelineWordsPerRing;
    tiling->reserved = 0;
    context->SetBlockDim(aivOnlyBlockDim);
    if (context->GetWorkspaceSizes(1) != nullptr) {
        context->GetWorkspaceSizes(1)[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferA5ExactScanSpeculativeGetShape(
    gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetInputShape(kLiScores) == nullptr ||
        context->GetInputShape(kPackedOutput) == nullptr ||
        context->GetInputShape(kPipelineState) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t output = 0; output <= kPipelineStateOut; ++output) {
        if (context->GetOutputShape(output) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    // InferShapeContext exposes gert::Shape directly on CANN 9.1.  The
    // GetStorageShape() wrapper exists on TilingContext's storage-shape view,
    // but not here; copying the Shape keeps this callback compatible with the
    // generated CANN 9.1 operator host ABI.
    const gert::Shape scores = *context->GetInputShape(kLiScores);
    const gert::Shape packedOutput = *context->GetInputShape(kPackedOutput);
    const gert::Shape pipelineState = *context->GetInputShape(kPipelineState);
    if (!IsVectorWithPositiveLength(scores)) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(kPackedOutputOut) = packedOutput;
    context->GetOutputShape(kCandidateIndices)->SetDimNum(1);
    context->GetOutputShape(kCandidateIndices)->SetDim(0, scores.GetDim(0));
    context->GetOutputShape(kCandidateKind)->SetDimNum(1);
    context->GetOutputShape(kCandidateKind)->SetDim(0, scores.GetDim(0));
    context->GetOutputShape(kProducerStats)->SetDimNum(2);
    context->GetOutputShape(kProducerStats)->SetDim(0, kProducerCount);
    context->GetOutputShape(kProducerStats)->SetDim(1, kProducerStatColumns);
    context->GetOutputShape(kGetStats)->SetDimNum(2);
    context->GetOutputShape(kGetStats)->SetDim(0, kGetLaneCount);
    context->GetOutputShape(kGetStats)->SetDim(1, kGetStatColumns);
    context->GetOutputShape(kRoleTrace)->SetDimNum(1);
    context->GetOutputShape(kRoleTrace)->SetDim(0, kAivCount);
    for (size_t output : {kGetIndices, kGetOffsets, kGetOutputIndices,
                          kGetMask}) {
        context->GetOutputShape(output)->SetDimNum(2);
        context->GetOutputShape(output)->SetDim(0, kGetLaneCount);
        context->GetOutputShape(output)->SetDim(1, kGetWaveRecords);
    }
    *context->GetOutputShape(kPipelineStateOut) = pipelineState;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferA5ExactScanSpeculativeGetDataType(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(kPackedOutputOut, ge::DT_UINT8);
    context->SetOutputDataType(kCandidateIndices, ge::DT_INT32);
    context->SetOutputDataType(kCandidateKind, ge::DT_UINT8);
    context->SetOutputDataType(kProducerStats, ge::DT_INT64);
    context->SetOutputDataType(kGetStats, ge::DT_INT64);
    context->SetOutputDataType(kRoleTrace, ge::DT_INT32);
    context->SetOutputDataType(kGetIndices, ge::DT_INT32);
    context->SetOutputDataType(kGetOffsets, ge::DT_INT32);
    context->SetOutputDataType(kGetOutputIndices, ge::DT_INT32);
    context->SetOutputDataType(kGetMask, ge::DT_UINT8);
    context->SetOutputDataType(kPipelineStateOut, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

class A5ExactScanSpeculativeGet : public OpDef {
public:
    explicit A5ExactScanSpeculativeGet(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> floatTypes = {ge::DT_FLOAT};
        const std::vector<ge::DataType> int32Types = {ge::DT_INT32};
        const std::vector<ge::DataType> int64Types = {ge::DT_INT64};
        const std::vector<ge::DataType> uint8Types = {ge::DT_UINT8};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        this->Input("li_scores").ParamType(REQUIRED).DataType(floatTypes).Format(formats);
        this->Input("threshold").ParamType(REQUIRED).DataType(floatTypes).Format(formats);
        this->Input("resident").ParamType(REQUIRED).DataType(uint8Types).Format(formats);
        this->Input("table_indices").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Input("offsets").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Input("output_indices").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Input("address_table").ParamType(REQUIRED).DataType(int64Types).Format(formats);
        this->Input("valid_table").ParamType(REQUIRED).DataType(uint8Types).Format(formats);
        this->Input("packed_output").ParamType(REQUIRED).DataType(uint8Types).Format(formats);
        this->Input("fused_get_control").ParamType(REQUIRED).DataType(int64Types).Format(formats);
        this->Input("frame_epoch").ParamType(REQUIRED).DataType(int64Types).Format(formats);
        this->Input("pipeline_state").ParamType(REQUIRED).DataType(int64Types).Format(formats);
        this->Output("packed_output_out").ParamType(REQUIRED).DataType(uint8Types).Format(formats);
        this->Output("candidate_indices").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("candidate_kind").ParamType(REQUIRED).DataType(uint8Types).Format(formats);
        this->Output("producer_stats").ParamType(REQUIRED).DataType(int64Types).Format(formats);
        this->Output("get_stats").ParamType(REQUIRED).DataType(int64Types).Format(formats);
        this->Output("role_trace").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("get_indices").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("get_offsets").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("get_output_indices").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("get_mask").ParamType(REQUIRED).DataType(uint8Types).Format(formats);
        this->Output("pipeline_state_out").ParamType(REQUIRED).DataType(int64Types).Format(formats);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore()
            .SetTiling(optiling::TilingA5ExactScanSpeculativeGet)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5ExactScanSpeculativeGet);

IMPL_OP_INFERSHAPE(A5ExactScanSpeculativeGet)
    .InferShape(InferA5ExactScanSpeculativeGetShape)
    .InferDataType(InferA5ExactScanSpeculativeGetDataType);
}  // namespace ops
