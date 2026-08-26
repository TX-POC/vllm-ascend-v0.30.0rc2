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
 * \file a5_c8_resident_seed.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#if (__CCE_AICORE__ != 310)
    #error "A5C8ResidentSeed is an Ascend-950-only source-derived C8 closure"
#endif
#include "arch35/quant_lightning_indexer_kernel.h"
#include "a5_c8_resident_seed_template_tiling_key.h"
using namespace QLIKernel;
using namespace optiling::detail;

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                         \
    do {                                                                                                     \
        templateClass<QLIType<__VA_ARGS__>> op;                                                              \
        GET_TILING_DATA_WITH_STRUCT(QLITilingData, tiling_data_in, tiling);                                  \
        const QLITilingData *__restrict tiling_data = &tiling_data_in;                                       \
        op.Init(query, residentIndexKey, weights, queryScale, residentIndexKeyDequantScale,                  \
                actualSeqLengthsQ, residentSeqLengths, residentIndexBlockTable, metadata, sparseIndices,     \
                residentTau, tauValid, user, tiling_data, &tPipe);                                           \
        op.Process();                                                                                        \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int Q_LAYOUT_T, int K_LAYOUT_T>
__global__ __aicore__ void a5_c8_resident_seed(
    __gm__ uint8_t *query, __gm__ uint8_t *residentIndexKey, __gm__ uint8_t *weights,
    __gm__ uint8_t *queryScale, __gm__ uint8_t *residentIndexKeyDequantScale,
    __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *residentSeqLengths,
    __gm__ uint8_t *residentIndexBlockTable, __gm__ uint8_t *metadata,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
    __gm__ uint8_t *residentTau, __gm__ uint8_t *tauValid,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    INVOKE_LI_NO_KFC_OP_IMPL(QLIPreload, fp8_e4m3fn_t, fp8_e4m3fn_t, float, uint16_t, int32_t,
                             PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T));
}
