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
 * \file a5_streaming_quant_li_get.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#if (__CCE_AICORE__ != 310)
    #error "A5StreamingQuantLiGet embeds the Ascend-950 C8 + Direct-HCCP helper and is ASCEND950-only"
#endif
#include "arch35/quant_lightning_indexer_kernel.h"
#include "a5_streaming_quant_li_get_template_tiling_key.h"
using namespace QLIKernel;
using namespace optiling::detail;

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                         \
    do {                                                                                                     \
        templateClass<QLIType<__VA_ARGS__>> op;                                                              \
        GET_TILING_DATA_WITH_STRUCT(QLITilingData, tiling_data_in, tiling);                                  \
        const QLITilingData *__restrict tiling_data = &tiling_data_in;                                       \
        op.Init(query, key, weights, queryScale, keyScale, actualSeqLengthsQ, actualSeqLengthsK, blocktable, \
                metadata, residentThreshold, residentThresholdValid, kvObjectSlotsSnapshot,                  \
                reqPoolEntries, kvObjectMap,                                                                   \
                privateSourceToSlot, privateReadyTable, privateArenaHead, privatePackedOutput,                \
                queueControl, queueLeases, kvObjectLens,                                                      \
                sparseIndices, sparseValues, privateSourceToSlotOut, privateReadyTableOut,                    \
                privateArenaHeadOut, privatePackedOutputOut, queueLeasesOut, telemetry,                       \
                scoreSourceLens,                                                                               \
                user, tiling_data, &tPipe);                                                                    \
        op.Process();                                                                                        \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int Q_LAYOUT_T, int K_LAYOUT_T>
__global__ __aicore__ void a5_streaming_quant_li_get(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                                   __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale,
                                                   __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK,
                                                   __gm__ uint8_t *blocktable, __gm__ uint8_t *metadata,
                                                   __gm__ uint8_t *residentThreshold,
                                                   __gm__ uint8_t *residentThresholdValid,
                                                   __gm__ uint8_t *kvObjectSlotsSnapshot,
                                                   __gm__ uint8_t *reqPoolEntries,
                                                   __gm__ uint8_t *kvObjectMap,
                                                   __gm__ uint8_t *privateSourceToSlot,
                                                   __gm__ uint8_t *privateReadyTable,
                                                   __gm__ uint8_t *privateArenaHead,
                                                   __gm__ uint8_t *privatePackedOutput,
                                                   __gm__ uint8_t *queueControl,
                                                   __gm__ uint8_t *queueLeases,
                                                   __gm__ uint8_t *kvObjectLens,
                                                   __gm__ uint8_t *sparseIndices,
                                                   __gm__ uint8_t *sparseValues,
                                                   __gm__ uint8_t *privateSourceToSlotOut,
                                                   __gm__ uint8_t *privateReadyTableOut,
                                                   __gm__ uint8_t *privateArenaHeadOut,
                                                   __gm__ uint8_t *privatePackedOutputOut,
                                                   __gm__ uint8_t *queueLeasesOut,
                                                   __gm__ uint8_t *telemetry,
                                                   __gm__ uint8_t *scoreSourceLens,
                                                   __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    INVOKE_LI_NO_KFC_OP_IMPL(QLIPreload, fp8_e4m3fn_t, fp8_e4m3fn_t, float, uint16_t, int32_t,
                             PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T));
}
