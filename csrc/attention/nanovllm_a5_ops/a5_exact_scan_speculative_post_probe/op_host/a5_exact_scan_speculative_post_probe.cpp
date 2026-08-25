/**
 * Host registration and tiling for the first ExactScan + SpeculativePost
 * prototype.
 *
 * The production operator will replace the synthetic scan and post markers
 * with Lightning score work and KVAF Direct-HCCP PostOnly.  This prototype
 * establishes the non-overlapping logical role layout in one MIX launch:
 * groups 0..30 run ExactScan roles, while the two AIV lanes in group 31 own
 * the future speculative-post role.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "../op_kernel/a5_exact_scan_speculative_post_probe_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t kScanValues = 0;
constexpr size_t kPostSeed = 1;
constexpr size_t kLiChecksums = 0;
constexpr size_t kPostMarkers = 1;
constexpr size_t kRoleTrace = 2;

constexpr uint32_t kAicCount = 32;
constexpr uint32_t kAivCount = 64;
constexpr uint32_t kAivPerMixedGroup = 2;
constexpr uint32_t kLiGroupCount = kAicCount - 1;
constexpr uint32_t kLiAivCount = kLiGroupCount * kAivPerMixedGroup;
constexpr uint32_t kPostAivCount = kAivPerMixedGroup;
constexpr int64_t kPostSeedCount = kPostAivCount;
// P0 exposes only the lanes that write a deterministic trace. The cube-side
// role is fixed by the 1C2V task layout, but it is not represented by a
// scalar AIC-to-GM trace output.
constexpr int64_t kRoleTraceCount = kAivCount;

bool IsPositiveVector(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) > 0;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingA5ExactScanSpeculativePostProbe(
    gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr ||
        context->GetInputShape(kScanValues) == nullptr ||
        context->GetInputShape(kPostSeed) == nullptr ||
        context->GetInputDesc(kScanValues) == nullptr ||
        context->GetInputDesc(kPostSeed) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputDesc(kScanValues)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(kPostSeed)->GetDataType() != ge::DT_INT32) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape scanValues =
        context->GetInputShape(kScanValues)->GetStorageShape();
    const gert::Shape postSeed =
        context->GetInputShape(kPostSeed)->GetStorageShape();
    if (!IsPositiveVector(scanValues) ||
        scanValues.GetDim(0) < static_cast<int64_t>(kLiAivCount) ||
        postSeed.GetDimNum() != 1 || postSeed.GetDim(0) != kPostSeedCount ||
        scanValues.GetDim(0) >
            static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aicCount = platform.GetCoreNumAic();
    const uint32_t aivCount = platform.GetCoreNumAiv();
    // This prototype intentionally refuses an unknown geometry.  The role
    // trace is useful only when it maps one-to-one to 32 MIX groups.
    if (aicCount != kAicCount || aivCount != kAivCount) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<
        A5ExactScanSpeculativePostProbeTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->liGroupCount = kLiGroupCount;
    tiling->liAivCount = kLiAivCount;
    tiling->postAivCount = kPostAivCount;
    tiling->scanValueCount = static_cast<uint32_t>(scanValues.GetDim(0));

    const uint32_t mixedBlockDim =
        platform.CalcTschBlockDim(aivCount, aicCount, aivCount);
    if (mixedBlockDim != kAicCount) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(mixedBlockDim);
    if (context->GetWorkspaceSizes(1) != nullptr) {
        context->GetWorkspaceSizes(1)[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferA5ExactScanSpeculativePostProbeShape(
    gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetOutputShape(kLiChecksums) == nullptr ||
        context->GetOutputShape(kPostMarkers) == nullptr ||
        context->GetOutputShape(kRoleTrace) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->GetOutputShape(kLiChecksums)->SetDimNum(1);
    context->GetOutputShape(kLiChecksums)->SetDim(0, kLiAivCount);
    context->GetOutputShape(kPostMarkers)->SetDimNum(1);
    context->GetOutputShape(kPostMarkers)->SetDim(0, kPostSeedCount);
    context->GetOutputShape(kRoleTrace)->SetDimNum(1);
    context->GetOutputShape(kRoleTrace)->SetDim(0, kRoleTraceCount);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferA5ExactScanSpeculativePostProbeDataType(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(kLiChecksums, ge::DT_INT64);
    context->SetOutputDataType(kPostMarkers, ge::DT_INT32);
    context->SetOutputDataType(kRoleTrace, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

class A5ExactScanSpeculativePostProbe : public OpDef {
public:
    explicit A5ExactScanSpeculativePostProbe(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> int32Types = {ge::DT_INT32};
        const std::vector<ge::DataType> int64Types = {ge::DT_INT64};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        this->Input("scan_values").ParamType(REQUIRED)
            .DataType(int32Types).Format(formats);
        this->Input("post_seed").ParamType(REQUIRED)
            .DataType(int32Types).Format(formats);
        this->Output("li_checksums").ParamType(REQUIRED)
            .DataType(int64Types).Format(formats);
        this->Output("post_markers").ParamType(REQUIRED)
            .DataType(int32Types).Format(formats);
        this->Output("role_trace").ParamType(REQUIRED)
            .DataType(int32Types).Format(formats);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore()
            .SetTiling(optiling::TilingA5ExactScanSpeculativePostProbe)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5ExactScanSpeculativePostProbe);

IMPL_OP_INFERSHAPE(A5ExactScanSpeculativePostProbe)
    .InferShape(InferA5ExactScanSpeculativePostProbeShape)
    .InferDataType(InferA5ExactScanSpeculativePostProbeDataType);
}  // namespace ops
