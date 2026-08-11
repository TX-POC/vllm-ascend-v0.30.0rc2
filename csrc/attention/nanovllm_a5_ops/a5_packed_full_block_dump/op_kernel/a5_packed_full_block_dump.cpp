/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "kernel_operator.h"
#include "a5_packed_full_block_dump_tiling.h"

namespace {
using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 128;
constexpr uint32_t PACKED_BYTES = 656;
constexpr uint32_t DMA_ALIGNMENT = 32;
constexpr uint32_t BLOCK_BYTES = BLOCK_SIZE * PACKED_BYTES; // 83,968 bytes.
constexpr uint32_t COPY_CHUNK_BYTES = 16 * 1024;
constexpr uint32_t FULL_CHUNKS_PER_BLOCK = BLOCK_BYTES / COPY_CHUNK_BYTES;
constexpr uint32_t TAIL_CHUNK_BYTES = BLOCK_BYTES % COPY_CHUNK_BYTES;
constexpr uint32_t CHUNKS_PER_BLOCK =
    FULL_CHUNKS_PER_BLOCK + (TAIL_CHUNK_BYTES == 0 ? 0 : 1);

static_assert(BLOCK_BYTES % DMA_ALIGNMENT == 0,
              "packed block byte size must be DMA aligned");
static_assert(COPY_CHUNK_BYTES % DMA_ALIGNMENT == 0,
              "copy chunk byte size must be DMA aligned");
static_assert(TAIL_CHUNK_BYTES == 0 ||
                  TAIL_CHUNK_BYTES % DMA_ALIGNMENT == 0,
              "tail chunk byte size must be DMA aligned");

class A5PackedFullBlockDumpKernel {
public:
    __aicore__ inline A5PackedFullBlockDumpKernel(
        TPipe* pipe,
        const A5PackedFullBlockDumpTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR hbmPackedKv,
        GM_ADDR dramPackedKv,
        GM_ADDR srcBlockIds,
        GM_ADDR dstBlockIds)
    {
        coreIdx_ = GetBlockIdx();
        // The packed row is 656 bytes, but a complete 128-token block is
        // 83,968 bytes and therefore 32-byte aligned.  Keep one AIV core as
        // the sole writer of one destination block and let that core publish
        // all aligned chunks sequentially.  This avoids both token-granular
        // partial-sector writes and concurrent writes from multiple cores into
        // the same swapped-memory block/page set.
        pipe_->InitBuffer(copyQueue_, 2, COPY_CHUNK_BYTES);
        hbmPackedKvGm_.SetGlobalBuffer((__gm__ uint8_t*)hbmPackedKv);
        dramPackedKvGm_.SetGlobalBuffer((__gm__ uint8_t*)dramPackedKv);
        srcBlockIdsGm_.SetGlobalBuffer((__gm__ int32_t*)srcBlockIds);
        dstBlockIdsGm_.SetGlobalBuffer((__gm__ int32_t*)dstBlockIds);
    }

    __aicore__ inline void Process()
    {
        // Ownership is job/block based, not chunk based.  Consequently one
        // destination DDR block is never written by more than one AIV core.
        for (uint32_t jobIdx = coreIdx_;
             jobIdx < tiling_->jobCount;
             jobIdx += tiling_->usedCoreNum) {
            const int32_t srcBlock = srcBlockIdsGm_.GetValue(jobIdx);
            const int32_t dstBlock = dstBlockIdsGm_.GetValue(jobIdx);
            // Graph replay can pad jobs with negative IDs. Invalid or padded
            // jobs are strict no-ops; host-side final-Prefill validation rejects
            // invalid real jobs before this kernel is launched.
            if (srcBlock < 0 || dstBlock < 0 ||
                static_cast<uint32_t>(srcBlock) >= tiling_->hbmBlockCount ||
                static_cast<uint32_t>(dstBlock) >= tiling_->dramBlockCount) {
                continue;
            }

            const uint64_t srcBlockBase =
                static_cast<uint64_t>(srcBlock) * BLOCK_BYTES;
            const uint64_t dstBlockBase =
                static_cast<uint64_t>(dstBlock) * BLOCK_BYTES;

            for (uint32_t chunkIdx = 0;
                 chunkIdx < CHUNKS_PER_BLOCK;
                 ++chunkIdx) {
                const uint32_t copyBytes = CopyBytesForChunk(chunkIdx);
                if (copyBytes == 0) {
                    continue;
                }
                const uint64_t chunkOffset =
                    static_cast<uint64_t>(chunkIdx) * COPY_CHUNK_BYTES;
                CopyAlignedChunk(
                    srcBlockBase + chunkOffset,
                    dstBlockBase + chunkOffset,
                    copyBytes);
            }
        }
    }

private:
    __aicore__ inline uint32_t CopyBytesForChunk(uint32_t chunkIdx) const
    {
        if (chunkIdx < FULL_CHUNKS_PER_BLOCK) {
            return COPY_CHUNK_BYTES;
        }
        if (chunkIdx == FULL_CHUNKS_PER_BLOCK && TAIL_CHUNK_BYTES != 0) {
            return TAIL_CHUNK_BYTES;
        }
        return 0;
    }

    __aicore__ inline void CopyAlignedChunk(
        uint64_t srcOffset,
        uint64_t dstOffset,
        uint32_t copyBytes)
    {
        LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyExtParams params{1, copyBytes, 0, 0, 0};

        DataCopyPad<uint8_t, PaddingMode::Normal>(
            local,
            hbmPackedKvGm_[srcOffset],
            params,
            padParams);
        copyQueue_.EnQue<uint8_t>(local);

        local = copyQueue_.DeQue<uint8_t>();
        DataCopyPad<uint8_t, PaddingMode::Normal>(
            dramPackedKvGm_[dstOffset],
            local,
            params);
        copyQueue_.FreeTensor(local);
    }

private:
    TPipe* pipe_;
    const A5PackedFullBlockDumpTilingData* tiling_;
    uint32_t coreIdx_ = 0;
    GlobalTensor<uint8_t> hbmPackedKvGm_;
    GlobalTensor<uint8_t> dramPackedKvGm_;
    GlobalTensor<int32_t> srcBlockIdsGm_;
    GlobalTensor<int32_t> dstBlockIdsGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
};
} // namespace

extern "C" __global__ __aicore__ void a5_packed_full_block_dump(
    GM_ADDR hbmPackedKv,
    GM_ADDR dramPackedKv,
    GM_ADDR srcBlockIds,
    GM_ADDR dstBlockIds,
    GM_ADDR dramPackedKvOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)dramPackedKvOut;
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5PackedFullBlockDumpTilingData);
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    A5PackedFullBlockDumpKernel op(&pipe, &tilingData);
    op.Init(hbmPackedKv, dramPackedKv, srcBlockIds, dstBlockIds);
    op.Process();
}
