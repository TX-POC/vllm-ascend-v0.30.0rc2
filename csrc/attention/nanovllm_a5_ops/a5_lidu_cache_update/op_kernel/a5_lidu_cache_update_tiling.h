#ifndef A5_LIDU_CACHE_UPDATE_TILING_H
#define A5_LIDU_CACHE_UPDATE_TILING_H

#include <cstdint>

struct A5LiduCacheUpdateTilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t poolSize;
    uint32_t sourceCapacity;
};

#endif
