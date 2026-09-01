// P8 fixed-arena admission gate host ABI.
//
// A captured graph schedules this one-AIV control node before the dynamic P8
// POST workers.  It is a bounded safety scan, not a dedicated Get core: the
// node finishes before the normal dynamic queue-lease workers are launched.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "../op_kernel/a5_p8_arena_admission_gate_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
enum InputIndex : size_t {
    kForwardControl = 0,
    kP8ArenaHealth,
    kFusedGetControl,
    kP8WaveRecords,
    kP8PrivateRecords,
    kExpectedIdentity,
    kAdmissionResult,
    kInputCount,
};

enum OutputIndex : size_t {
    kForwardControlOut = 0,
    kP8ArenaHealthOut,
    kAdmissionResultOut,
    kOutputCount,
};

constexpr size_t kAliasInputs[kOutputCount] = {
    kForwardControl,
    kP8ArenaHealth,
    kAdmissionResult,
};

bool IsVectorOfLength(const gert::Shape& shape, int64_t length)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == length;
}

bool IsPositiveMatrix(const gert::Shape& shape, int64_t columns)
{
    return shape.GetDimNum() == 2 && shape.GetDim(0) > 0 &&
           shape.GetDim(1) == columns;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingA5P8ArenaAdmissionGate(
    gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType expected[kInputCount] = {
        ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64,
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    };
    const gert::Shape* shapes[kInputCount] = {};
    for (size_t index = 0U; index < kInputCount; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr ||
            context->GetInputDesc(index)->GetDataType() != expected[index]) {
            return ge::GRAPH_FAILED;
        }
        shapes[index] = &context->GetInputShape(index)->GetStorageShape();
    }
    const gert::Shape& control = *shapes[kForwardControl];
    const gert::Shape& health = *shapes[kP8ArenaHealth];
    const gert::Shape& fused = *shapes[kFusedGetControl];
    const gert::Shape& waves = *shapes[kP8WaveRecords];
    const gert::Shape& private_records = *shapes[kP8PrivateRecords];
    const gert::Shape& identity = *shapes[kExpectedIdentity];
    const gert::Shape& result = *shapes[kAdmissionResult];
    if (!IsVectorOfLength(control, kA5P8AdmissionControlWords) ||
        !IsVectorOfLength(health, 1) ||
        !IsVectorOfLength(fused, kA5P8AdmissionStreamingControlWords) ||
        !IsPositiveMatrix(waves, kA5P8AdmissionWaveRecordWords) ||
        !IsPositiveMatrix(private_records, kA5P8AdmissionPrivateRecordWords) ||
        !IsVectorOfLength(identity, kA5P8AdmissionIdentityWords) ||
        !IsVectorOfLength(result, kA5P8AdmissionResultWords)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t queue_count = waves.GetDim(0);
    const int64_t private_count = private_records.GetDim(0);
    if (queue_count <= 0 || private_count <= 0 ||
        queue_count > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        private_count > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
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
    auto* tiling = context->GetTilingData<A5P8ArenaAdmissionGateTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->queueCount = static_cast<uint32_t>(queue_count);
    tiling->privateRowCount = static_cast<uint32_t>(private_count);
    context->SetScheduleMode(1U);
    context->SetBlockDim(block_dim);
    if (context->GetWorkspaceSizes(1) != nullptr) {
        context->GetWorkspaceSizes(1)[0] = 0U;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferA5P8ArenaAdmissionGateShape(
    gert::InferShapeContext* context)
{
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

static ge::graphStatus InferA5P8ArenaAdmissionGateDataType(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(kForwardControlOut, ge::DT_INT64);
    context->SetOutputDataType(kP8ArenaHealthOut, ge::DT_INT32);
    context->SetOutputDataType(kAdmissionResultOut, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

class A5P8ArenaAdmissionGate : public OpDef {
public:
    explicit A5P8ArenaAdmissionGate(const char* name) : OpDef(name)
    {
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        const std::vector<ge::DataType> int32_types = {ge::DT_INT32};
        const std::vector<ge::DataType> int64_types = {ge::DT_INT64};
        const ge::DataType input_types[kInputCount] = {
            ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64,
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
        };
        const char* input_names[kInputCount] = {
            "forward_control", "p8_arena_health", "fused_get_control",
            "p8_wave_records", "p8_private_records", "expected_identity",
            "admission_result",
        };
        const ge::DataType output_types[kOutputCount] = {
            ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
        };
        const char* output_names[kOutputCount] = {
            "forward_control", "p8_arena_health", "admission_result",
        };
        for (size_t input = 0U; input < kInputCount; ++input) {
            const std::vector<ge::DataType>& types =
                input_types[input] == ge::DT_INT32 ? int32_types : int64_types;
            this->Input(input_names[input]).ParamType(REQUIRED)
                .DataType(types).Format(formats);
        }
        for (size_t output = 0U; output < kOutputCount; ++output) {
            const std::vector<ge::DataType>& types =
                output_types[output] == ge::DT_INT32 ? int32_types : int64_types;
            this->Output(output_names[output]).ParamType(REQUIRED)
                .DataType(types).Format(formats);
        }
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value", "a5_p8_arena_admission_gate")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore()
            .SetTiling(optiling::TilingA5P8ArenaAdmissionGate)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5P8ArenaAdmissionGate);

IMPL_OP_INFERSHAPE(A5P8ArenaAdmissionGate)
    .InferShape(InferA5P8ArenaAdmissionGateShape)
    .InferDataType(InferA5P8ArenaAdmissionGateDataType);
}  // namespace ops
