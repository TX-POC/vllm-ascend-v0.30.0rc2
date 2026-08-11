/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef A5_KVCACHE_SCATTER_COPY_TILING_H
#define A5_KVCACHE_SCATTER_COPY_TILING_H

#include <cstdint>

struct A5KvcacheScatterCopyTilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t copyCap;
    uint32_t hbmMaxBlockNum;
    uint32_t dramMaxBlockNum;
    uint32_t elementBytes;
    uint64_t totalPairSlots;
};

#endif // A5_KVCACHE_SCATTER_COPY_TILING_H
