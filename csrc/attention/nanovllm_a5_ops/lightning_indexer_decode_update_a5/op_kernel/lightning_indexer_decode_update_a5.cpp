/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_decode_update_a5.cpp
 * \brief Decode-only LightningIndexer update kernel.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lightning_indexer_decode_update_a5_template_tiling_key.h"
#include "lightning_indexer_decode_update_a5_kernel.h"

using namespace LIKernel;

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                                   \
    do {                                                                                                               \
        templateClass<LIType<__VA_ARGS__>> op;                                                                         \
        LI_COPY_TILING_DATA(LIA5TilingData, tiling);                                                                  \
        op.Init(query, key, weights, reqPoolEntries, cacheTokens, nullptr, candidateLens, blocktable,                 \
                sourceIds, destinationSlots, cacheSlotsPool, destinationSlots, missCounts,                            \
                user, tiling_data, &tPipe);                                                                            \
        op.Process();                                                                                                  \
    } while (0)

#define LI_COPY_TILING_DATA(tilingDataStruct, tiling)                                                                  \
    GET_TILING_DATA_WITH_STRUCT(tilingDataStruct, tiling_data_in, tiling);                                             \
    const tilingDataStruct *__restrict tiling_data = &tiling_data_in;


template <int DT>
__global__ __aicore__ void lightning_indexer_decode_update_a5(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                             __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheSlotsPool,
                                             __gm__ uint8_t *cacheTokens, __gm__ uint8_t *candidateLens,
                                             __gm__ uint8_t *blocktable, __gm__ uint8_t *sourceIds,
                                             __gm__ uint8_t *destinationSlots, __gm__ uint8_t *missCounts,
                                             __gm__ uint8_t *cacheSlotsAlias, __gm__ uint8_t *workspace,
                                             __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    (void)cacheSlotsAlias;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)
    if (ORIG_DTYPE_QUERY == DT_BF16) {
        INVOKE_LI_NO_KFC_OP_IMPL(LightningIndexerKernel, bfloat16_t, bfloat16_t, int32_t, true,
                                 LI_LAYOUT::BSND, LI_LAYOUT::PA_BSND, false);
    } else if (ORIG_DTYPE_QUERY == DT_FLOAT16) {
        INVOKE_LI_NO_KFC_OP_IMPL(LightningIndexerKernel, half, half, int32_t, true,
                                 LI_LAYOUT::BSND, LI_LAYOUT::PA_BSND, false);
    }
#else
    if constexpr (DT == LI_TPL_FP16) {
        INVOKE_LI_NO_KFC_OP_IMPL(LightningIndexerKernel, half, half, int32_t, true,
                                 LI_LAYOUT::BSND, LI_LAYOUT::PA_BSND, false);
    } else {
        INVOKE_LI_NO_KFC_OP_IMPL(LightningIndexerKernel, bfloat16_t, bfloat16_t, int32_t, true,
                                 LI_LAYOUT::BSND, LI_LAYOUT::PA_BSND, false);
    }
#endif
}
