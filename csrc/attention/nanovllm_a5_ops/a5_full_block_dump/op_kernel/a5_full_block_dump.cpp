/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "kernel_operator.h"
#include "a5_full_block_dump_tiling.h"

namespace {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 128;
constexpr uint32_t BLOCK_SHIFT = 7;
constexpr uint32_t BLOCK_MASK = BLOCK_SIZE - 1;
constexpr uint32_t K_ROPE_DIM = 64;
constexpr uint32_t KV_CACHE_DIM = 512;

class A5FullBlockDumpKernel {
public:
    __aicore__ inline A5FullBlockDumpKernel(
        TPipe* pipe,
        const A5FullBlockDumpTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR hbmKRoPE,
        GM_ADDR hbmKvCache,
        GM_ADDR dramKRoPE,
        GM_ADDR dramKvCache,
        GM_ADDR srcBlockIds,
        GM_ADDR dstBlockIds)
    {
        coreIdx_ = GetBlockIdx();
        kRopeBytes_ = K_ROPE_DIM * tiling_->elementBytes;
        kvCacheBytes_ = KV_CACHE_DIM * tiling_->elementBytes;
        pipe_->InitBuffer(copyQueue_, 2, kvCacheBytes_ + kRopeBytes_);

        hbmKRoPEGm_.SetGlobalBuffer((__gm__ uint8_t*)hbmKRoPE);
        hbmKvCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)hbmKvCache);
        dramKRoPEGm_.SetGlobalBuffer((__gm__ uint8_t*)dramKRoPE);
        dramKvCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)dramKvCache);
        srcBlockIdsGm_.SetGlobalBuffer((__gm__ int32_t*)srcBlockIds);
        dstBlockIdsGm_.SetGlobalBuffer((__gm__ int32_t*)dstBlockIds);
    }

    __aicore__ inline void Process()
    {
        for (uint64_t flatToken = coreIdx_;
             flatToken < tiling_->totalTokenSlots;
             flatToken += tiling_->usedCoreNum) {
            CopyAddress address;
            if (!ResolveAddress(flatToken, address)) {
                continue;
            }
            CopyOneToken(address);
        }
    }

private:
    struct CopyAddress {
        uint64_t srcKv = 0;
        uint64_t dstKv = 0;
        uint64_t srcRope = 0;
        uint64_t dstRope = 0;
    };

    __aicore__ inline bool ResolveAddress(
        uint64_t flatToken,
        CopyAddress& address)
    {
        const uint32_t jobIdx =
            static_cast<uint32_t>(flatToken >> BLOCK_SHIFT);
        const uint32_t tokenOffset =
            static_cast<uint32_t>(flatToken) & BLOCK_MASK;
        if (jobIdx >= tiling_->jobCount) {
            return false;
        }

        const int32_t srcBlock = srcBlockIdsGm_.GetValue(jobIdx);
        const int32_t dstBlock = dstBlockIdsGm_.GetValue(jobIdx);
        if (srcBlock < 0 || dstBlock < 0 ||
            static_cast<uint32_t>(srcBlock) >= tiling_->hbmBlockCount ||
            static_cast<uint32_t>(dstBlock) >= tiling_->dramBlockCount) {
            return false;
        }

        const uint64_t srcToken =
            static_cast<uint64_t>(srcBlock) * BLOCK_SIZE + tokenOffset;
        const uint64_t dstToken =
            static_cast<uint64_t>(dstBlock) * BLOCK_SIZE + tokenOffset;
        address.srcKv = srcToken * kvCacheBytes_;
        address.dstKv = dstToken * kvCacheBytes_;
        address.srcRope = srcToken * kRopeBytes_;
        address.dstRope = dstToken * kRopeBytes_;
        return true;
    }

    __aicore__ inline void CopyOneToken(const CopyAddress& address)
    {
        LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyExtParams kvParams{1, kvCacheBytes_, 0, 0, 0};
        DataCopyExtParams ropeParams{1, kRopeBytes_, 0, 0, 0};

        // HBM -> UB.  The destination tensors are swapped-memory GM mappings;
        // UB -> GM below is the A5 replacement for the legacy full-block ACLNN.
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            local, hbmKvCacheGm_[address.srcKv], kvParams, padParams);
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            local[kvCacheBytes_], hbmKRoPEGm_[address.srcRope],
            ropeParams, padParams);
        copyQueue_.EnQue<uint8_t>(local);

        local = copyQueue_.DeQue<uint8_t>();
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            dramKvCacheGm_[address.dstKv], local, kvParams);
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            dramKRoPEGm_[address.dstRope], local[kvCacheBytes_],
            ropeParams);
        copyQueue_.FreeTensor(local);
    }

private:
    TPipe* pipe_;
    const A5FullBlockDumpTilingData* tiling_;
    uint32_t coreIdx_ = 0;
    uint32_t kRopeBytes_ = 0;
    uint32_t kvCacheBytes_ = 0;
    GlobalTensor<uint8_t> hbmKRoPEGm_;
    GlobalTensor<uint8_t> hbmKvCacheGm_;
    GlobalTensor<uint8_t> dramKRoPEGm_;
    GlobalTensor<uint8_t> dramKvCacheGm_;
    GlobalTensor<int32_t> srcBlockIdsGm_;
    GlobalTensor<int32_t> dstBlockIdsGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
};
} // namespace

extern "C" __global__ __aicore__ void a5_full_block_dump(
    GM_ADDR hbmKRoPE,
    GM_ADDR hbmKvCache,
    GM_ADDR dramKRoPE,
    GM_ADDR dramKvCache,
    GM_ADDR srcBlockIds,
    GM_ADDR dstBlockIds,
    GM_ADDR dramKRoPEOut,
    GM_ADDR dramKvCacheOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5FullBlockDumpTilingData);
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    A5FullBlockDumpKernel op(&pipe, &tilingData);
    op.Init(
        hbmKRoPE,
        hbmKvCache,
        dramKRoPE,
        dramKvCache,
        srcBlockIds,
        dstBlockIds);
    op.Process();
}
