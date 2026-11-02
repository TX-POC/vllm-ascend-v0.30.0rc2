#ifndef A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_TILING_H
#define A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_TILING_H

#include <cstdint>

// A 32-int64 lane record occupies 256 bytes, an exact multiple of a 64-byte
// GM cache line. Concurrent AIV profiling stores therefore never share a
// cache line.
constexpr uint32_t A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE_LANES = 8;
constexpr uint32_t A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE_FIELDS = 32;
static_assert(
    A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE_FIELDS * sizeof(int64_t) % 64 == 0,
    "each profiling lane must occupy complete GM cache lines");

struct A5LiduFusedBatchGetPrepareMtpTilingData {
    uint32_t usedCoreNum;
    uint32_t coresPerRequest;
    uint32_t batchSize;
    uint32_t packedQueryCount;
    uint32_t poolSize;
    uint32_t sourceCapacity;
    uint32_t residentCapacity;
    uint32_t residentBlockCapacity;
    uint32_t layerId;
    uint32_t numLayers;
    uint32_t maxNumReqs;
    uint32_t qcap;
};

#endif
