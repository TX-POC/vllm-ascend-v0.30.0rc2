/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "kernel_operator.h"
#include "a5_kvcache_scatter_copy_tiling.h"

namespace {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 128;
constexpr uint32_t BLOCK_SHIFT = 7;
constexpr uint32_t BLOCK_MASK = BLOCK_SIZE - 1;
constexpr uint32_t K_ROPE_DIM = 64;
constexpr uint32_t KV_CACHE_DIM = 512;

class A5KvcacheScatterCopyKernel {
public:
    __aicore__ inline A5KvcacheScatterCopyKernel(
        TPipe* pipe,
        const A5KvcacheScatterCopyTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR hbmKRoPE,
        GM_ADDR hbmKvCache,
        GM_ADDR dramKRoPE,
        GM_ADDR dramKvCache,
        GM_ADDR hbmBlockTable,
        GM_ADDR dramBlockTable,
        GM_ADDR srcTokenIds,
        GM_ADDR dstSlots,
        GM_ADDR copyCounts)
    {
        coreIdx_ = GetBlockIdx();
        kRopeBytes_ = K_ROPE_DIM * tiling_->elementBytes;
        kvCacheBytes_ = KV_CACHE_DIM * tiling_->elementBytes;
        pipe_->InitBuffer(copyQueue_, 2, kvCacheBytes_ + kRopeBytes_);
        hbmKRoPEGm_.SetGlobalBuffer((__gm__ uint8_t*)hbmKRoPE);
        hbmKvCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)hbmKvCache);
        dramKRoPEGm_.SetGlobalBuffer((__gm__ uint8_t*)dramKRoPE);
        dramKvCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)dramKvCache);
        hbmBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)hbmBlockTable);
        dramBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)dramBlockTable);
        srcTokenIdsGm_.SetGlobalBuffer((__gm__ int32_t*)srcTokenIds);
        dstSlotsGm_.SetGlobalBuffer((__gm__ int32_t*)dstSlots);
        copyCountsGm_.SetGlobalBuffer((__gm__ int32_t*)copyCounts);
    }

    __aicore__ inline void Process()
    {
        cachedBatchIdx_ = static_cast<uint32_t>(-1);
        cachedCopyCount_ = 0;

        uint64_t currentFlatPair = FindNextValidPair(coreIdx_);
        CopyAddress currentAddress;
        while (currentFlatPair < tiling_->totalPairSlots &&
               !ResolveAddress(currentFlatPair, currentAddress)) {
            currentFlatPair = FindNextValidPair(
                currentFlatPair + tiling_->usedCoreNum);
        }
        if (currentFlatPair >= tiling_->totalPairSlots) {
            return;
        }

        CopyIn(currentAddress);
        while (true) {
            uint64_t nextFlatPair = FindNextValidPair(
                currentFlatPair + tiling_->usedCoreNum);
            CopyAddress nextAddress;
            while (nextFlatPair < tiling_->totalPairSlots &&
                   !ResolveAddress(nextFlatPair, nextAddress)) {
                nextFlatPair = FindNextValidPair(
                    nextFlatPair + tiling_->usedCoreNum);
            }

            const bool hasNext =
                nextFlatPair < tiling_->totalPairSlots;
            if (hasNext) {
                CopyIn(nextAddress);
            }
            CopyOut(currentAddress);
            if (!hasNext) {
                break;
            }
            currentFlatPair = nextFlatPair;
            currentAddress = nextAddress;
        }
    }

private:
    struct CopyAddress {
        uint64_t srcKv = 0;
        uint64_t dstKv = 0;
        uint64_t srcRope = 0;
        uint64_t dstRope = 0;
    };

    __aicore__ inline uint64_t FirstFlatPairAtOrAfter(
        uint64_t start) const
    {
        if (start <= coreIdx_) {
            return coreIdx_;
        }
        const uint64_t distance = start - coreIdx_;
        const uint64_t steps =
            (distance + tiling_->usedCoreNum - 1) /
            tiling_->usedCoreNum;
        return coreIdx_ + steps * tiling_->usedCoreNum;
    }

    __aicore__ inline uint64_t FindNextValidPair(
        uint64_t flatPair)
    {
        while (flatPair < tiling_->totalPairSlots) {
            const uint32_t batchIdx = static_cast<uint32_t>(
                flatPair / tiling_->copyCap);
            const uint32_t copyIdx = static_cast<uint32_t>(
                flatPair -
                static_cast<uint64_t>(batchIdx) * tiling_->copyCap);
            if (batchIdx != cachedBatchIdx_) {
                cachedCopyCount_ = copyCountsGm_.GetValue(batchIdx);
                if (cachedCopyCount_ < 0) {
                    cachedCopyCount_ = 0;
                } else if (
                    cachedCopyCount_ >
                    static_cast<int32_t>(tiling_->copyCap)) {
                    cachedCopyCount_ =
                        static_cast<int32_t>(tiling_->copyCap);
                }
                cachedBatchIdx_ = batchIdx;
            }
            if (copyIdx < static_cast<uint32_t>(cachedCopyCount_)) {
                return flatPair;
            }
            flatPair = FirstFlatPairAtOrAfter(
                (static_cast<uint64_t>(batchIdx) + 1) *
                tiling_->copyCap);
        }
        return tiling_->totalPairSlots;
    }

    __aicore__ inline bool ResolveAddress(
        uint64_t flatPair,
        CopyAddress& address)
    {
        const uint32_t batchIdx = static_cast<uint32_t>(
            flatPair / tiling_->copyCap);
        const uint32_t copyIdx = static_cast<uint32_t>(
            flatPair -
            static_cast<uint64_t>(batchIdx) * tiling_->copyCap);
        const uint64_t pairOffset =
            static_cast<uint64_t>(batchIdx) * tiling_->copyCap +
            copyIdx;
        const int32_t srcTokenId =
            srcTokenIdsGm_.GetValue(pairOffset);
        const int32_t dstSlot = dstSlotsGm_.GetValue(pairOffset);
        if (srcTokenId < 0 || dstSlot < 0) {
            return false;
        }

        const uint32_t srcBlockCol =
            static_cast<uint32_t>(srcTokenId) >> BLOCK_SHIFT;
        const uint32_t dstBlockCol =
            static_cast<uint32_t>(dstSlot) >> BLOCK_SHIFT;
        if (srcBlockCol >= tiling_->dramMaxBlockNum ||
            dstBlockCol >= tiling_->hbmMaxBlockNum) {
            return false;
        }

        const int32_t srcPhysicalBlock =
            dramBlockTableGm_.GetValue(
                static_cast<uint64_t>(batchIdx) *
                    tiling_->dramMaxBlockNum +
                srcBlockCol);
        const int32_t dstPhysicalBlock =
            hbmBlockTableGm_.GetValue(
                static_cast<uint64_t>(batchIdx) *
                    tiling_->hbmMaxBlockNum +
                dstBlockCol);
        if (srcPhysicalBlock < 0 || dstPhysicalBlock < 0) {
            return false;
        }

        const uint64_t srcToken =
            static_cast<uint64_t>(srcPhysicalBlock) * BLOCK_SIZE +
            (static_cast<uint32_t>(srcTokenId) & BLOCK_MASK);
        const uint64_t dstToken =
            static_cast<uint64_t>(dstPhysicalBlock) * BLOCK_SIZE +
            (static_cast<uint32_t>(dstSlot) & BLOCK_MASK);
        address.srcKv = srcToken * kvCacheBytes_;
        address.dstKv = dstToken * kvCacheBytes_;
        address.srcRope = srcToken * kRopeBytes_;
        address.dstRope = dstToken * kRopeBytes_;
        return true;
    }

    __aicore__ inline void CopyIn(const CopyAddress& address)
    {
        LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyExtParams kvParams{1, kvCacheBytes_, 0, 0, 0};
        DataCopyExtParams ropeParams{1, kRopeBytes_, 0, 0, 0};

        // The source tensors may be NPU tensors backed by host DRAM through
        // torch_npu.empty_with_swapped_memory. GM->UB is therefore the exact
        // path this penetration experiment is intended to validate.
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            local, dramKvCacheGm_[address.srcKv], kvParams, padParams);
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            local[kvCacheBytes_], dramKRoPEGm_[address.srcRope],
            ropeParams, padParams);
        copyQueue_.EnQue<uint8_t>(local);
    }

    __aicore__ inline void CopyOut(const CopyAddress& address)
    {
        LocalTensor<uint8_t> local =
            copyQueue_.DeQue<uint8_t>();
        DataCopyExtParams kvParams{1, kvCacheBytes_, 0, 0, 0};
        DataCopyExtParams ropeParams{1, kRopeBytes_, 0, 0, 0};
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            hbmKvCacheGm_[address.dstKv], local, kvParams);
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            hbmKRoPEGm_[address.dstRope], local[kvCacheBytes_],
            ropeParams);
        copyQueue_.FreeTensor(local);
    }

private:
    TPipe* pipe_;
    const A5KvcacheScatterCopyTilingData* tiling_;
    uint32_t coreIdx_ = 0;
    uint32_t cachedBatchIdx_ = static_cast<uint32_t>(-1);
    int32_t cachedCopyCount_ = 0;
    uint32_t kRopeBytes_ = 0;
    uint32_t kvCacheBytes_ = 0;
    GlobalTensor<uint8_t> hbmKRoPEGm_;
    GlobalTensor<uint8_t> hbmKvCacheGm_;
    GlobalTensor<uint8_t> dramKRoPEGm_;
    GlobalTensor<uint8_t> dramKvCacheGm_;
    GlobalTensor<int32_t> hbmBlockTableGm_;
    GlobalTensor<int32_t> dramBlockTableGm_;
    GlobalTensor<int32_t> srcTokenIdsGm_;
    GlobalTensor<int32_t> dstSlotsGm_;
    GlobalTensor<int32_t> copyCountsGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
};
} // namespace

extern "C" __global__ __aicore__ void a5_kvcache_scatter_copy(
    GM_ADDR hbmKRoPE,
    GM_ADDR hbmKvCache,
    GM_ADDR dramKRoPE,
    GM_ADDR dramKvCache,
    GM_ADDR hbmBlockTable,
    GM_ADDR dramBlockTable,
    GM_ADDR srcTokenIds,
    GM_ADDR dstSlots,
    GM_ADDR copyCounts,
    GM_ADDR hbmKRoPEOut,
    GM_ADDR hbmKvCacheOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5KvcacheScatterCopyTilingData);
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    A5KvcacheScatterCopyKernel op(&pipe, &tilingData);
    op.Init(
        hbmKRoPE,
        hbmKvCache,
        dramKRoPE,
        dramKvCache,
        hbmBlockTable,
        dramBlockTable,
        srcTokenIds,
        dstSlots,
        copyCounts);
    op.Process();
}
