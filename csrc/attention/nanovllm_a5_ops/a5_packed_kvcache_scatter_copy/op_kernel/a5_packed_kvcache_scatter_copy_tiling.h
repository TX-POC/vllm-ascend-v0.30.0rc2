#ifndef A5_PACKED_KVCACHE_SCATTER_COPY_TILING_H
#define A5_PACKED_KVCACHE_SCATTER_COPY_TILING_H

#include <cstdint>

struct A5PackedKvcacheScatterCopyTilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t copyCap;
    uint32_t hbmMaxBlockNum;
    uint32_t dramMaxBlockNum;
    uint32_t packedRowBytes;
    uint32_t attentionCapacity;
    uint64_t totalPairSlots;
};

#endif
