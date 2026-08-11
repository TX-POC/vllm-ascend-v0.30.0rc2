/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <register/op_impl_registry.h>
#include "a5_sfa_shared/ops_log_compat.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t CACHE_SLOTS_INDEX = 4;
constexpr int64_t DECODE_SPARSE_COUNT = 2048;

static ge::graphStatus InferShapeLightningIndexerDecodeUpdateA5(
    gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("LightningIndexerDecodeUpdateA5", "InferShapeContext is nullptr."),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    const gert::Shape *cacheShape = context->GetInputShape(CACHE_SLOTS_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, keyShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, cacheShape, return ge::GRAPH_FAILED);

    gert::Shape *sourceShape = context->GetOutputShape(0);
    gert::Shape *slotsShape = context->GetOutputShape(1);
    gert::Shape *missShape = context->GetOutputShape(2);
    gert::Shape *cacheAliasShape = context->GetOutputShape(3);
    OPS_LOG_E_IF_NULL(context, sourceShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, slotsShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, missShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, cacheAliasShape, return ge::GRAPH_FAILED);

    OPS_ERR_IF(queryShape->GetDimNum() != 3 || keyShape->GetDimNum() != 4,
               OPS_LOG_E(context, "query/key ranks must be 3/4."),
               return ge::GRAPH_FAILED);
    sourceShape->SetDimNum(3);
    sourceShape->SetDim(0, queryShape->GetDim(0));
    sourceShape->SetDim(1, keyShape->GetDim(2));
    sourceShape->SetDim(2, DECODE_SPARSE_COUNT);
    *slotsShape = *sourceShape;
    missShape->SetDimNum(1);
    missShape->SetDim(0, queryShape->GetDim(0));
    *cacheAliasShape = *cacheShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeLightningIndexerDecodeUpdateA5(
    gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("LightningIndexerDecodeUpdateA5", "InferDataTypeContext is nullptr."),
               return ge::GRAPH_FAILED);
    for (uint32_t index = 0; index < 4; ++index) {
        context->SetOutputDataType(index, ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LightningIndexerDecodeUpdateA5)
    .InferShape(InferShapeLightningIndexerDecodeUpdateA5)
    .InferDataType(InferDataTypeLightningIndexerDecodeUpdateA5);
} // namespace ops
