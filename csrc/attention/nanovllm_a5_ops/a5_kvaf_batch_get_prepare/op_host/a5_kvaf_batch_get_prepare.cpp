/** Host registration for the mask-compatible KVAF BatchGet input resolver. */

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include "../op_kernel/a5_kvaf_batch_get_prepare_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t SOURCE_TOKEN_IDS = 0;
constexpr size_t DESTINATION_SLOTS = 1;
constexpr size_t MISS_COUNTS = 2;
constexpr size_t RESIDENT_POOL_INDICES = 3;
constexpr size_t RESIDENT_BLOCK_TABLE = 4;
constexpr size_t CACHE_TOKENS = 5;
constexpr size_t CANDIDATE_LENS = 6;
constexpr size_t LOGICAL_SEQ_LENGTHS = 7;
constexpr size_t INDICES_BUFFER = 8;
constexpr size_t OFFSETS_BUFFER = 9;
constexpr size_t MASK_BUFFER = 10;
constexpr size_t OUTPUT_INDICES_BUFFER = 11;
constexpr size_t RESIDENT_SEQ_LENGTHS_BUFFER = 12;
constexpr size_t INPUT_COUNT = 13;

constexpr size_t LAYER_ID_ATTR = 0;
constexpr size_t NUM_LAYERS_ATTR = 1;
constexpr size_t MAX_NUM_REQS_ATTR = 2;
constexpr size_t QCAP_ATTR = 3;
constexpr int64_t BLOCK_SIZE = 128;

bool IsVector(const gert::Shape& shape, int64_t size)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == size;
}

bool IsRequestMatrix(const gert::Shape& shape, int64_t batch, int64_t capacity)
{
    return (shape.GetDimNum() == 2 && shape.GetDim(0) == batch &&
            shape.GetDim(1) == capacity) ||
        (shape.GetDimNum() == 3 && shape.GetDim(0) == batch &&
         shape.GetDim(1) == 1 && shape.GetDim(2) == capacity);
}
} // namespace

namespace optiling {
static ge::graphStatus TilingA5KvafBatchGetPrepare(
    gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr ||
        context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < INPUT_COUNT; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    for (size_t index = 0; index < INPUT_COUNT; ++index) {
        const ge::DataType expected =
            index == MASK_BUFFER ? ge::DT_UINT8 : ge::DT_INT32;
        if (context->GetInputDesc(index)->GetDataType() != expected) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape source =
        context->GetInputShape(SOURCE_TOKEN_IDS)->GetStorageShape();
    int64_t batch = 0;
    int64_t requestCapacity = 0;
    if (source.GetDimNum() == 2) {
        batch = source.GetDim(0);
        requestCapacity = source.GetDim(1);
    } else if (source.GetDimNum() == 3 && source.GetDim(1) == 1) {
        batch = source.GetDim(0);
        requestCapacity = source.GetDim(2);
    } else {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape residentTable =
        context->GetInputShape(RESIDENT_BLOCK_TABLE)->GetStorageShape();
    if (batch <= 0 || requestCapacity <= 0 ||
        residentTable.GetDimNum() != 2 || residentTable.GetDim(0) != batch ||
        residentTable.GetDim(1) <= 0 ||
        !IsRequestMatrix(
            context->GetInputShape(DESTINATION_SLOTS)->GetStorageShape(),
            batch, requestCapacity) ||
        !IsVector(
            context->GetInputShape(MISS_COUNTS)->GetStorageShape(), batch) ||
        !IsVector(
            context->GetInputShape(RESIDENT_POOL_INDICES)->GetStorageShape(),
            batch) ||
        !IsVector(
            context->GetInputShape(CACHE_TOKENS)->GetStorageShape(), batch) ||
        !IsVector(
            context->GetInputShape(CANDIDATE_LENS)->GetStorageShape(), batch) ||
        !IsVector(
            context->GetInputShape(LOGICAL_SEQ_LENGTHS)->GetStorageShape(),
            batch)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t flatCapacity = batch * requestCapacity;
    for (size_t index : {INDICES_BUFFER, OFFSETS_BUFFER,
                         OUTPUT_INDICES_BUFFER}) {
        if (!IsVector(
                context->GetInputShape(index)->GetStorageShape(),
                flatCapacity)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (!IsVector(
            context->GetInputShape(MASK_BUFFER)->GetStorageShape(),
            flatCapacity) ||
        !IsVector(
            context->GetInputShape(RESIDENT_SEQ_LENGTHS_BUFFER)
                ->GetStorageShape(),
            batch)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t* layerId =
        context->GetAttrs()->GetAttrPointer<int64_t>(LAYER_ID_ATTR);
    const int64_t* numLayers =
        context->GetAttrs()->GetAttrPointer<int64_t>(NUM_LAYERS_ATTR);
    const int64_t* maxNumReqs =
        context->GetAttrs()->GetAttrPointer<int64_t>(MAX_NUM_REQS_ATTR);
    const int64_t* qcap =
        context->GetAttrs()->GetAttrPointer<int64_t>(QCAP_ATTR);
    if (layerId == nullptr || numLayers == nullptr || maxNumReqs == nullptr ||
        qcap == nullptr || *layerId < 0 || *numLayers <= 0 ||
        *layerId >= *numLayers || *maxNumReqs <= 0 || *qcap <= 0 ||
        batch > *maxNumReqs ||
        *layerId > std::numeric_limits<uint32_t>::max() ||
        *numLayers > std::numeric_limits<uint32_t>::max() ||
        *maxNumReqs > std::numeric_limits<uint32_t>::max() ||
        *qcap > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(*maxNumReqs) *
                static_cast<uint64_t>(*numLayers) *
                static_cast<uint64_t>(*qcap) >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        static_cast<uint64_t>(residentTable.GetDim(1)) * BLOCK_SIZE >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        requestCapacity > std::numeric_limits<uint32_t>::max() ||
        flatCapacity > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aivCount == 0) {
        return ge::GRAPH_FAILED;
    }
    (void)aivCount;
    auto* tiling =
        context->GetTilingData<A5KvafBatchGetPrepareTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = 1;
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->requestCapacity = static_cast<uint32_t>(requestCapacity);
    tiling->residentBlockCapacity =
        static_cast<uint32_t>(residentTable.GetDim(1));
    tiling->layerId = static_cast<uint32_t>(*layerId);
    tiling->numLayers = static_cast<uint32_t>(*numLayers);
    tiling->maxNumReqs = static_cast<uint32_t>(*maxNumReqs);
    tiling->qcap = static_cast<uint32_t>(*qcap);
    context->SetBlockDim(tiling->usedCoreNum);
    if (context->GetWorkspaceSizes(1) != nullptr) {
        context->GetWorkspaceSizes(1)[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
static ge::graphStatus InferA5KvafBatchGetPrepareShape(
    gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t output = 0; output < 5; ++output) {
        const size_t input = INDICES_BUFFER + output;
        if (context->GetInputShape(input) == nullptr ||
            context->GetOutputShape(output) == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *context->GetOutputShape(output) = *context->GetInputShape(input);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferA5KvafBatchGetPrepareDataType(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_UINT8);
    context->SetOutputDataType(3, ge::DT_INT32);
    context->SetOutputDataType(4, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

class A5KvafBatchGetPrepare : public OpDef {
public:
    explicit A5KvafBatchGetPrepare(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> ints = {ge::DT_INT32};
        const std::vector<ge::DataType> bytes = {ge::DT_UINT8};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        this->Input("source_token_ids").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("destination_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("miss_counts").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("resident_pool_indices").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("resident_block_table").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("cache_tokens").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("candidate_lens").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("logical_seq_lengths").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("indices_buffer").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("offsets_buffer").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("mask_buffer").ParamType(REQUIRED).DataType(bytes).Format(formats);
        this->Input("output_indices_buffer").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("resident_seq_lengths_buffer").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Attr("layer_id").AttrType(REQUIRED).Int();
        this->Attr("num_layers").AttrType(REQUIRED).Int();
        this->Attr("max_num_reqs").AttrType(REQUIRED).Int();
        this->Attr("qcap").AttrType(REQUIRED).Int();
        this->Output("indices_out").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("offsets_out").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("mask_out").ParamType(REQUIRED).DataType(bytes).Format(formats);
        this->Output("output_indices_out").ParamType(REQUIRED).DataType(ints).Format(formats);
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
            .SetTiling(optiling::TilingA5KvafBatchGetPrepare)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5KvafBatchGetPrepare);

IMPL_OP_INFERSHAPE(A5KvafBatchGetPrepare)
    .InferShape(InferA5KvafBatchGetPrepareShape)
    .InferDataType(InferA5KvafBatchGetPrepareDataType);
} // namespace ops
