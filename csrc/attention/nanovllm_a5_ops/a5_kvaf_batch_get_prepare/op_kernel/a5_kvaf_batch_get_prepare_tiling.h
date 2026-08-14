#ifndef A5_KVAF_BATCH_GET_PREPARE_TILING_H
#define A5_KVAF_BATCH_GET_PREPARE_TILING_H

#include <cstdint>

struct A5KvafBatchGetPrepareTilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t requestCapacity;
    uint32_t residentBlockCapacity;
    uint32_t layerId;
    uint32_t numLayers;
    uint32_t maxNumReqs;
    uint32_t qcap;
};

#endif
