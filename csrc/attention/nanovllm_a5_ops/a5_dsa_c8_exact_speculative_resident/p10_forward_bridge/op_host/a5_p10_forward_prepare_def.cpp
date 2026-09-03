// Host definition for P10's graph-visible forward preparation node.

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include "../op_kernel/a5_p10_forward_prepare_abi.h"
#include "../op_kernel/a5_p10_forward_prepare_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
enum InputIndex : size_t {
    RECEIPT_STATE = 0,
    FORWARD_CONTROL,
    OBJECT_TO_FORWARD_SLOT,
    SOURCE_TO_FORWARD_SLOT,
    SPECULATIVE_SLOT_CURSOR,
    QUEUE_LEASES,
    RESIDENT_READY,
};

enum OutputIndex : size_t {
    RECEIPT_STATE_ALIAS = 0,
    FORWARD_CONTROL_ALIAS,
    OBJECT_TO_FORWARD_SLOT_ALIAS,
    SOURCE_TO_FORWARD_SLOT_ALIAS,
    SPECULATIVE_SLOT_CURSOR_ALIAS,
    QUEUE_LEASES_ALIAS,
    RESIDENT_READY_ALIAS,
};

bool HasInput(gert::TilingContext* context, size_t index)
{
    return context->GetInputShape(index) != nullptr &&
           context->GetInputDesc(index) != nullptr;
}

bool ShapeElements(const gert::Shape& shape, uint32_t* result)
{
    uint64_t elements = 1;
    for (size_t dim = 0; dim < shape.GetDimNum(); ++dim) {
        const int64_t extent = shape.GetDim(dim);
        if (extent <= 0 ||
            elements > std::numeric_limits<uint32_t>::max() /
                           static_cast<uint64_t>(extent)) {
            return false;
        }
        elements *= static_cast<uint64_t>(extent);
    }
    *result = static_cast<uint32_t>(elements);
    return true;
}

bool IsVectorLength(const gert::Shape& shape, int64_t length)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == length;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingA5P10ForwardPrepare(gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = RECEIPT_STATE; index <= RESIDENT_READY; ++index) {
        if (!HasInput(context, index) ||
            context->GetInputDesc(index)->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape receipt_state =
        context->GetInputShape(RECEIPT_STATE)->GetStorageShape();
    const gert::Shape forward_control =
        context->GetInputShape(FORWARD_CONTROL)->GetStorageShape();
    const gert::Shape object_to_forward_slot =
        context->GetInputShape(OBJECT_TO_FORWARD_SLOT)->GetStorageShape();
    const gert::Shape source_to_forward_slot =
        context->GetInputShape(SOURCE_TO_FORWARD_SLOT)->GetStorageShape();
    const gert::Shape speculative_slot_cursor =
        context->GetInputShape(SPECULATIVE_SLOT_CURSOR)->GetStorageShape();
    const gert::Shape queue_leases =
        context->GetInputShape(QUEUE_LEASES)->GetStorageShape();
    const gert::Shape resident_ready =
        context->GetInputShape(RESIDENT_READY)->GetStorageShape();
    if (receipt_state.GetDimNum() != 1 ||
        !IsVectorLength(forward_control,
                        A5_P10_FORWARD_PREPARE_CONTROL_WORDS) ||
        object_to_forward_slot.GetDimNum() != 2 ||
        source_to_forward_slot.GetDimNum() != 2 ||
        !IsVectorLength(speculative_slot_cursor, 1) ||
        queue_leases.GetDimNum() != 1 || resident_ready.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<A5P10ForwardPrepareTilingData>();
    if (tiling == nullptr || context->GetWorkspaceSizes(1) == nullptr ||
        !ShapeElements(receipt_state, &tiling->receiptStateElements) ||
        !ShapeElements(object_to_forward_slot,
                       &tiling->objectToForwardSlotElements) ||
        !ShapeElements(source_to_forward_slot,
                       &tiling->sourceToForwardSlotElements) ||
        !ShapeElements(queue_leases, &tiling->queueLeaseElements) ||
        !ShapeElements(resident_ready, &tiling->residentReadyElements)) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aiv_count = platform.GetCoreNumAiv();
    if (aiv_count == 0U) {
        return ge::GRAPH_FAILED;
    }
    // The preparation node uses a bounded logical number of vector blocks.
    // This is neither a C8 worker count nor an affinity for a physical core.
    constexpr uint32_t kLogicalPrepareBlockLimit = 8U;
    tiling->usedCoreNum =
        aiv_count < kLogicalPrepareBlockLimit ? aiv_count
                                               : kLogicalPrepareBlockLimit;
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    context->SetBlockDim(tiling->usedCoreNum);
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferA5P10ForwardPrepareShape(
    gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    constexpr size_t kInputForOutput[] = {
        RECEIPT_STATE, FORWARD_CONTROL, OBJECT_TO_FORWARD_SLOT,
        SOURCE_TO_FORWARD_SLOT, SPECULATIVE_SLOT_CURSOR, QUEUE_LEASES,
        RESIDENT_READY,
    };
    for (size_t output = RECEIPT_STATE_ALIAS; output <= RESIDENT_READY_ALIAS;
         ++output) {
        if (context->GetInputShape(kInputForOutput[output]) == nullptr ||
            context->GetOutputShape(output) == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *context->GetOutputShape(output) =
            *context->GetInputShape(kInputForOutput[output]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferA5P10ForwardPrepareDataType(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t output = RECEIPT_STATE_ALIAS; output <= RESIDENT_READY_ALIAS;
         ++output) {
        context->SetOutputDataType(output, ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}

class A5P10ForwardPrepare : public OpDef {
 public:
    explicit A5P10ForwardPrepare(const char* name) : OpDef(name)
    {
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        const std::vector<ge::DataType> int32s = {ge::DT_INT32};
        this->Input("receipt_state").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Input("forward_control").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Input("kv_object_to_forward_slot").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Input("source_to_forward_slot").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Input("speculative_slot_cursor").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Input("queue_leases").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Input("resident_ready").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);

        this->Output("receipt_state_alias").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Output("forward_control_alias").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Output("kv_object_to_forward_slot_alias").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Output("source_to_forward_slot_alias").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Output("speculative_slot_cursor_alias").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Output("queue_leases_alias").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Output("resident_ready_alias").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value", "a5_p10_forward_prepare")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore()
            .SetTiling(optiling::TilingA5P10ForwardPrepare)
            .AddConfig("ascend950", config);
    }
};

OP_ADD(A5P10ForwardPrepare);

IMPL_OP_INFERSHAPE(A5P10ForwardPrepare)
    .InferShape(InferA5P10ForwardPrepareShape)
    .InferDataType(InferA5P10ForwardPrepareDataType);
}  // namespace ops
