/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef lightning_indexer_decode_update_a5_TILING_H_
#define lightning_indexer_decode_update_a5_TILING_H_

#include "a5_sfa_shared/ops_log_compat.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct URequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct UTensorParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t WEIGHTS_INDEX = 2;
constexpr uint32_t REQ_POOL_ENTRIES_INDEX = 3;
constexpr uint32_t CACHE_SLOTS_INDEX = 4;
constexpr uint32_t CACHE_TOKENS_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 6;
constexpr uint32_t BLOCK_TABLE_INDEX = 7;
constexpr uint32_t TOPK_INDEX = 0;
constexpr uint32_t TOPK_SLOTS_INDEX = 1;
constexpr uint32_t MISS_COUNT_INDEX = 2;
constexpr uint32_t CACHE_SLOTS_OUT_INDEX = 3;

constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

constexpr uint32_t DECODE_N2 = 1;
constexpr uint32_t DECODE_G_SIZE_32 = 32;
constexpr uint32_t DECODE_G_SIZE_64 = 64;
constexpr uint32_t DECODE_HEAD_DIM = 128;
constexpr uint32_t DECODE_SPARSE_COUNT = 2048;
constexpr uint32_t MAX_CACHE_SLOTS_SIZE = 1U << 18;

BEGIN_TILING_DATA_DEF(LIA5TilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, poolSize)
TILING_DATA_FIELD_DEF(uint32_t, cacheSlotsSize)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(int64_t, preTokens)
TILING_DATA_FIELD_DEF(int64_t, nextTokens)
TILING_DATA_FIELD_DEF(uint32_t, returnValue)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(LightningIndexerDecodeUpdateA5, LIA5TilingData)

struct LIA5CompileInfo {};

struct LIA5ParaInfo {
    URequiredParaInfo query = {nullptr, nullptr};
    URequiredParaInfo key = {nullptr, nullptr};
    URequiredParaInfo weights = {nullptr, nullptr};
    UTensorParaInfo reqPoolEntries = {nullptr, nullptr};
    URequiredParaInfo cacheSlots = {nullptr, nullptr};
    UTensorParaInfo cacheTokens = {nullptr, nullptr};
    UTensorParaInfo actualSeqLengths = {nullptr, nullptr};
    UTensorParaInfo blockTable = {nullptr, nullptr};
    URequiredParaInfo topkIndexOut = {nullptr, nullptr};
    URequiredParaInfo topkSlotsOut = {nullptr, nullptr};
    URequiredParaInfo missCountOut = {nullptr, nullptr};
    URequiredParaInfo cacheSlotsOut = {nullptr, nullptr};
};

class LIA5TilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    LIA5ParaInfo opParamInfo;

    uint32_t bSize = 0;
    uint32_t n1Size = DECODE_G_SIZE_64;
    uint32_t n2Size = DECODE_N2;
    uint32_t s2Size = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t poolSize = 0;
    uint32_t cacheSlotsSize = 0;
    uint32_t usedCoreNum = 0;

    ge::DataType inputQType = ge::DT_FLOAT16;
};

class LightningIndexerDecodeUpdateA5Tiling {
public:
    explicit LightningIndexerDecodeUpdateA5Tiling(gert::TilingContext *context) : context_(context) {};
    ge::graphStatus ParseAndCheck(LIA5TilingInfo &tilingInfo);
    ge::graphStatus DoTiling(LIA5TilingInfo *tilingInfo);

private:
    ge::graphStatus GetNpuInfo(LIA5TilingInfo &tilingInfo) const;
    ge::graphStatus GetTensorInfo(LIA5TilingInfo &tilingInfo) const;
    ge::graphStatus CheckDtype(const LIA5TilingInfo &tilingInfo) const;
    ge::graphStatus CheckShape(LIA5TilingInfo &tilingInfo) const;

    gert::TilingContext *context_ = nullptr;
    LIA5TilingData tilingData_;
};

} // namespace optiling
#endif // lightning_indexer_decode_update_a5_TILING_H_
