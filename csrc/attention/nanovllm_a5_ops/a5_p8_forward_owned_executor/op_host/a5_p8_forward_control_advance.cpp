// Host contract for the P8 graph-visible forward-control transition node.
//
// The op has no attributes and no host callback state.  `expected_identity`
// and `transition_program` are ordinary device tensors captured with the
// graph, which prevents Python/GIL mutation between POST, DRAIN, QSFA, and
// ForwardTerminal nodes.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../op_kernel/a5_p8_forward_control_advance_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
enum InputIndex : size_t {
    kForwardControl = 0,
    // Graph-stable sticky arena health.  It is intentionally an explicit
    // alias instead of a process-local error bit: every captured phase node
    // observes the same forward admission outcome.
    kP8ArenaHealth,
    kExpectedIdentity,
    kTransitionProgram,
    kTransitionResult,
    kInputCount,
};

enum OutputIndex : size_t {
    kForwardControlOut = 0,
    kP8ArenaHealthOut,
    kTransitionResultOut,
    kOutputCount,
};

constexpr size_t kAliasInputs[kOutputCount] = {
    kForwardControl,
    kP8ArenaHealth,
    kTransitionResult,
};

bool IsVectorOfLength(const gert::Shape& shape, int64_t length) {
    return shape.GetDimNum() == 1 && shape.GetDim(0) == length;
}

}  // namespace

namespace optiling {
static ge::graphStatus TilingA5P8ForwardControlAdvance(
    gert::TilingContext* context) {
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    constexpr ge::DataType kInputTypes[kInputCount] = {
        ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64,
        ge::DT_INT64,
    };
    for (size_t index = 0U; index < kInputCount; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr ||
            context->GetInputDesc(index)->GetDataType() != kInputTypes[index]) {
            return ge::GRAPH_FAILED;
        }
    }
    const gert::Shape& forward =
        context->GetInputShape(kForwardControl)->GetStorageShape();
    const gert::Shape& arena_health =
        context->GetInputShape(kP8ArenaHealth)->GetStorageShape();
    const gert::Shape& identity =
        context->GetInputShape(kExpectedIdentity)->GetStorageShape();
    const gert::Shape& program =
        context->GetInputShape(kTransitionProgram)->GetStorageShape();
    const gert::Shape& result =
        context->GetInputShape(kTransitionResult)->GetStorageShape();
    if (!IsVectorOfLength(forward, kA5P8ControlWords) ||
        !IsVectorOfLength(arena_health, 1) ||
        !IsVectorOfLength(identity, kA5P8IdentityWords) ||
        !IsVectorOfLength(program, kA5P8TransitionProgramWords) ||
        !IsVectorOfLength(result, kA5P8TransitionResultWords)) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    if (platform.GetCoreNumAiv() == 0U) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t block_dim = platform.CalcTschBlockDim(
        1U, 0U, platform.GetCoreNumAiv());
    if (block_dim != 1U) {
        return ge::GRAPH_FAILED;
    }
    auto* tiling = context->GetTilingData<A5P8ForwardControlAdvanceTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->reserved0 = 0U;
    tiling->reserved1 = 0U;
    context->SetScheduleMode(1U);
    context->SetBlockDim(block_dim);
    if (context->GetWorkspaceSizes(1) != nullptr) {
        context->GetWorkspaceSizes(1)[0] = 0U;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferA5P8ForwardControlAdvanceShape(
    gert::InferShapeContext* context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t output = 0U; output < kOutputCount; ++output) {
        const size_t input = kAliasInputs[output];
        if (context->GetInputShape(input) == nullptr ||
            context->GetOutputShape(output) == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *context->GetOutputShape(output) = *context->GetInputShape(input);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferA5P8ForwardControlAdvanceDataType(
    gert::InferDataTypeContext* context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(kForwardControlOut, ge::DT_INT64);
    context->SetOutputDataType(kP8ArenaHealthOut, ge::DT_INT32);
    context->SetOutputDataType(kTransitionResultOut, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

class A5P8ForwardControlAdvance : public OpDef {
public:
    explicit A5P8ForwardControlAdvance(const char* name) : OpDef(name) {
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        const std::vector<ge::DataType> int64_types = {ge::DT_INT64};
        const std::vector<ge::DataType> int32_types = {ge::DT_INT32};
        const char* input_names[kInputCount] = {
            "forward_control", "p8_arena_health", "expected_identity",
            "transition_program", "transition_result",
        };
        const char* output_names[kOutputCount] = {
            "forward_control", "p8_arena_health", "transition_result",
        };
        for (size_t input = 0U; input < kInputCount; ++input) {
            this->Input(input_names[input]).ParamType(REQUIRED)
                .DataType(input == kP8ArenaHealth ? int32_types : int64_types)
                .Format(formats);
        }
        for (size_t output = 0U; output < kOutputCount; ++output) {
            this->Output(output_names[output]).ParamType(REQUIRED)
                .DataType(output == kP8ArenaHealthOut ? int32_types
                                                       : int64_types)
                .Format(formats);
        }
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value", "a5_p8_forward_control_advance")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore()
            .SetTiling(optiling::TilingA5P8ForwardControlAdvance)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5P8ForwardControlAdvance);

IMPL_OP_INFERSHAPE(A5P8ForwardControlAdvance)
    .InferShape(InferA5P8ForwardControlAdvanceShape)
    .InferDataType(InferA5P8ForwardControlAdvanceDataType);
}  // namespace ops
