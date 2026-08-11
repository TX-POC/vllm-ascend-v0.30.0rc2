/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef A5_PACKED_FULL_BLOCK_DUMP_TILING_H
#define A5_PACKED_FULL_BLOCK_DUMP_TILING_H

#include <cstdint>

struct A5PackedFullBlockDumpTilingData {
    uint32_t usedCoreNum;
    uint32_t jobCount;
    uint32_t hbmBlockCount;
    uint32_t dramBlockCount;
    uint64_t totalCopyChunks;
};

#endif // A5_PACKED_FULL_BLOCK_DUMP_TILING_H
