/**
 * Ascend 950 sparse-and-tail attention host registration.
 *
 * The shared tiler is derived from vLLM-Ascend 0.23 SparseFlashAttention.
 * Its first nine inputs intentionally keep the production SFA order. The
 * cache-token budget is an extra input consumed only by the custom kernel.
 */

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "a5_sfa_shared/sparse_flash_attention_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"

// Keep the imported production tiler in exactly one host translation unit.
#include "a5_sfa_shared/sparse_flash_attention_tiling.inc"

namespace optiling {
REGISTER_TILING_DATA_CLASS(
    A5SparseAndTailAttention,
    SparseFlashAttentionTilingDataMla)
} // namespace optiling

namespace {
constexpr size_t QUERY = 0;
constexpr size_t KEY = 1;
constexpr size_t VALUE = 2;
constexpr size_t SPARSE_SLOTS = 3;
constexpr size_t BLOCK_TABLE = 4;
constexpr size_t ACTUAL_Q = 5;
constexpr size_t ACTUAL_KV = 6;
constexpr size_t QUERY_ROPE = 7;
constexpr size_t KEY_ROPE = 8;
constexpr size_t CACHE_TOKENS = 9;

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t CKV_DIM = 512;
constexpr int64_t KPE_DIM = 64;
constexpr int64_t SPARSE_COUNT = 2048;
constexpr int64_t MAX_REGULAR_LOCAL_HEADS = 64;

bool IsShape(
    const gert::Shape &shape,
    std::initializer_list<int64_t> dims)
{
    if (shape.GetDimNum() != dims.size()) {
        return false;
    }
    size_t index = 0;
    for (const int64_t dim : dims) {
        if (dim >= 0 && shape.GetDim(index) != dim) {
            return false;
        }
        ++index;
    }
    return true;
}

ge::graphStatus CheckA5SparseTailInputs(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = QUERY; index <= CACHE_TOKENS; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape query =
        context->GetInputShape(QUERY)->GetStorageShape();
    const gert::Shape key =
        context->GetInputShape(KEY)->GetStorageShape();
    const gert::Shape value =
        context->GetInputShape(VALUE)->GetStorageShape();
    const gert::Shape slots =
        context->GetInputShape(SPARSE_SLOTS)->GetStorageShape();
    const gert::Shape blockTable =
        context->GetInputShape(BLOCK_TABLE)->GetStorageShape();
    const gert::Shape actualQ =
        context->GetInputShape(ACTUAL_Q)->GetStorageShape();
    const gert::Shape actualKv =
        context->GetInputShape(ACTUAL_KV)->GetStorageShape();
    const gert::Shape queryRope =
        context->GetInputShape(QUERY_ROPE)->GetStorageShape();
    const gert::Shape keyRope =
        context->GetInputShape(KEY_ROPE)->GetStorageShape();
    const gert::Shape cacheTokens =
        context->GetInputShape(CACHE_TOKENS)->GetStorageShape();

    if (!IsShape(query, {-1, -1, CKV_DIM}) ||
        query.GetDim(0) <= 0 ||
        query.GetDim(1) <= 0 ||
        query.GetDim(1) > MAX_REGULAR_LOCAL_HEADS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t totalQueryTokens = query.GetDim(0);
    const int64_t localHeads = query.GetDim(1);
    if (!IsShape(key, {-1, BLOCK_SIZE, 1, CKV_DIM}) ||
        !IsShape(value, {key.GetDim(0), BLOCK_SIZE, 1, CKV_DIM}) ||
        !IsShape(keyRope, {key.GetDim(0), BLOCK_SIZE, 1, KPE_DIM}) ||
        !IsShape(queryRope, {totalQueryTokens, localHeads, KPE_DIM}) ||
        !IsShape(slots, {totalQueryTokens, 1, SPARSE_COUNT}) ||
        !IsShape(blockTable, {-1, -1}) ||
        !IsShape(actualQ, {-1}) ||
        !IsShape(actualKv, {actualQ.GetDim(0)}) ||
        !IsShape(cacheTokens, {actualQ.GetDim(0)}) ||
        actualQ.GetDim(0) != totalQueryTokens ||
        blockTable.GetDim(0) != actualQ.GetDim(0) ||
        blockTable.GetDim(1) <= 0 ||
        blockTable.GetDim(1) * BLOCK_SIZE > (1 << 18)) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType floatingType =
        context->GetInputDesc(QUERY)->GetDataType();
    if (floatingType != ge::DT_BF16 &&
        floatingType != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index : {KEY, VALUE, QUERY_ROPE, KEY_ROPE}) {
        if (context->GetInputDesc(index)->GetDataType() != floatingType) {
            return ge::GRAPH_FAILED;
        }
    }
    for (size_t index :
         {SPARSE_SLOTS, BLOCK_TABLE, ACTUAL_Q, ACTUAL_KV, CACHE_TOKENS}) {
        if (context->GetInputDesc(index)->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingA5SparseAndTailAttention(
    gert::TilingContext *context)
{
    if (CheckA5SparseTailInputs(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return TilingSparseFlashAttention(context);
}
} // namespace optiling

namespace ops {
static ge::graphStatus InferShapeForA5SparseAndTailAttention(
    gert::InferShapeContext *context)
{
    if (context == nullptr ||
        context->GetInputShape(QUERY) == nullptr ||
        context->GetOutputShape(0) == nullptr ||
        context->GetOutputShape(1) == nullptr ||
        context->GetOutputShape(2) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *context->GetInputShape(QUERY);
    // torch_npu::OpCommand cannot pass a REQUIRED zero-storage output to GE.
    // These one-element placeholders are unused when return_softmax_lse=false.
    *context->GetOutputShape(1) = gert::Shape({1});
    *context->GetOutputShape(2) = gert::Shape({1});
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForA5SparseAndTailAttention(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(
        0, context->GetInputDataType(QUERY));
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

class A5SparseAndTailAttention : public OpDef {
public:
    explicit A5SparseAndTailAttention(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> floatTypes = {
            ge::DT_BF16, ge::DT_FLOAT16};
        const std::vector<ge::DataType> intTypes = {
            ge::DT_INT32, ge::DT_INT32};
        const std::vector<ge::DataType> fp32Types = {
            ge::DT_FLOAT, ge::DT_FLOAT};
        const std::vector<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("query").ParamType(REQUIRED).DataType(floatTypes).Format(formats);
        this->Input("key").ParamType(REQUIRED).DataType(floatTypes).Format(formats);
        this->Input("value").ParamType(REQUIRED).DataType(floatTypes).Format(formats);
        this->Input("sparse_indices").ParamType(REQUIRED).DataType(intTypes).Format(formats);
        this->Input("block_table").ParamType(OPTIONAL).DataType(intTypes).Format(formats);
        this->Input("actual_seq_lengths_query").ParamType(OPTIONAL).DataType(intTypes).Format(formats);
        this->Input("actual_seq_lengths_kv").ParamType(OPTIONAL).DataType(intTypes).Format(formats);
        this->Input("query_rope").ParamType(OPTIONAL).DataType(floatTypes).Format(formats);
        this->Input("key_rope").ParamType(OPTIONAL).DataType(floatTypes).Format(formats);
        this->Input("cache_tokens").ParamType(REQUIRED).DataType(intTypes).Format(formats);
        this->Output("attention_out").ParamType(REQUIRED).DataType(floatTypes).Format(formats);
        this->Output("softmax_max").ParamType(REQUIRED).DataType(fp32Types).Format(formats);
        this->Output("softmax_sum").ParamType(REQUIRED).DataType(fp32Types).Format(formats);

        this->Attr("scale_value").AttrType(REQUIRED).Float(1.0);
        this->Attr("sparse_block_size").AttrType(OPTIONAL).Int(1);
        this->Attr("layout_query").AttrType(OPTIONAL).String("TND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("PA_BSND");
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
        this->Attr("attention_mode").AttrType(OPTIONAL).Int(2);
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore()
            .SetTiling(optiling::TilingA5SparseAndTailAttention)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5SparseAndTailAttention);

IMPL_OP_INFERSHAPE(A5SparseAndTailAttention)
    .InferShape(InferShapeForA5SparseAndTailAttention)
    .InferDataType(InferDataTypeForA5SparseAndTailAttention);

} // namespace ops
