/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file a5_c8_resident_seed_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "err/ops_err.h"
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 2;
constexpr uint32_t ATTR_KV_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 4;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 9;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t SPARSE_INDICES_OUTPUT_INDEX = 0;
constexpr uint32_t SPARSE_VALUES_OUTPUT_INDEX = 1;
constexpr uint32_t RESIDENT_TAU_OUTPUT_INDEX = 2;
constexpr uint32_t TAU_VALID_OUTPUT_INDEX = 3;

static ge::graphStatus InferShapeA5C8ResidentSeed(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("A5C8ResidentSeed", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    gert::Shape *sparseIndicesShape = context->GetOutputShape(SPARSE_INDICES_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseIndicesShape);
    gert::Shape *sparseValuesShape = context->GetOutputShape(SPARSE_VALUES_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseValuesShape);
    gert::Shape *residentTauShape = context->GetOutputShape(RESIDENT_TAU_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, residentTauShape);
    gert::Shape *tauValidShape = context->GetOutputShape(TAU_VALID_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, tauValidShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutQueryPtr);
    const char *inputLayoutKeyPtr = attrs->GetAttrPointer<char>(ATTR_KV_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutKeyPtr);
    const int64_t *sparse_count = attrs->GetInt(ATTR_SPARSE_COUNT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparse_count);

    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    std::string inputLayoutKeyPtrStr = std::string(inputLayoutKeyPtr);
    if (inputLayoutQueryPtrStr != "TND" && inputLayoutQueryPtrStr != "BSND") {
        OP_LOGE(context, "The input layout query should be TND or BSND, but got %s.", inputLayoutQueryPtrStr.c_str());
        return GRAPH_FAILED;
    }

    int64_t keyHeadNum = (inputLayoutKeyPtrStr == "TND") ? keyShape->GetDim(1) : keyShape->GetDim(2);
    int64_t selectionRowCount = 0;
    if (inputLayoutQueryPtrStr == "BSND") {
        sparseIndicesShape->SetDimNum(DIM_NUM_4);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0));  // 0:Dim B
        sparseIndicesShape->SetDim(1, queryShape->GetDim(1));  // 1:Dim S
        sparseIndicesShape->SetDim(2, keyHeadNum);             // 2:Dim N
        sparseIndicesShape->SetDim(3, *sparse_count);          // 3:Dim K
        selectionRowCount = queryShape->GetDim(0) * queryShape->GetDim(1) * keyHeadNum;
    } else {
        sparseIndicesShape->SetDimNum(DIM_NUM_3);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0));  // 0:Dim T
        sparseIndicesShape->SetDim(1, keyHeadNum);             // 1:output shape's N Dim, 2: key shape's N Dim
        sparseIndicesShape->SetDim(2, *sparse_count);          // 2:Dim K
        selectionRowCount = queryShape->GetDim(0) * keyHeadNum;
    }
    const bool *return_value = attrs->GetAttrPointer<bool>(ATTR_RETURN_VALUE_INDEX);
    bool returnValueFlag = (return_value != nullptr) ? *return_value : false;
    if (returnValueFlag) {
        *sparseValuesShape = *sparseIndicesShape;
    } else {
        sparseValuesShape->SetDimNum(1);
        sparseValuesShape->SetDim(0, 0);
    }

    // Flat row-major companion ABI. Row r is exactly the same selection row
    // as sparse_indices.reshape(-1, 2048)[r], independent of BSND or TND.
    residentTauShape->SetDimNum(1);
    residentTauShape->SetDim(0, selectionRowCount);
    tauValidShape->SetDimNum(1);
    tauValidShape->SetDim(0, selectionRowCount);

    OP_LOGD(context->GetNodeName(), "A5C8ResidentSeed InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeA5C8ResidentSeed(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("A5C8ResidentSeed", "InferDataTypeContext context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Enter A5C8ResidentSeed InferDataType impl.");
    // default index data type is int32
    ge::DataType outputType = ge::DT_INT32;
    context->SetOutputDataType(SPARSE_INDICES_OUTPUT_INDEX, outputType);
    context->SetOutputDataType(RESIDENT_TAU_OUTPUT_INDEX, outputType);
    context->SetOutputDataType(TAU_VALID_OUTPUT_INDEX, outputType);
    OP_LOGD(context->GetNodeName(), "A5C8ResidentSeed InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(A5C8ResidentSeed)
    .InferShape(InferShapeA5C8ResidentSeed)
    .InferDataType(InferDataTypeA5C8ResidentSeed);
}  // namespace ops
