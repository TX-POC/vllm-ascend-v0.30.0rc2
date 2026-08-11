/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef A5_FULL_BLOCK_DUMP_TILING_H
#define A5_FULL_BLOCK_DUMP_TILING_H

#include <cstdint>

struct A5FullBlockDumpTilingData {
    uint32_t usedCoreNum;
    uint32_t jobCount;
    uint32_t hbmBlockCount;
    uint32_t dramBlockCount;
    uint32_t elementBytes;
    uint64_t totalTokenSlots;
};

#endif // A5_FULL_BLOCK_DUMP_TILING_H
