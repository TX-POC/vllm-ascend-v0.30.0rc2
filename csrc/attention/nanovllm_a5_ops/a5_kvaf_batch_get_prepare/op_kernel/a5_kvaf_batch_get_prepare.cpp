/** Resolve LIDU output into the current mask-based KVAF BatchGet ABI. */

#include "kernel_operator.h"
#include "a5_kvaf_batch_get_prepare_tiling.h"

namespace {
using namespace AscendC;

constexpr uint32_t BLOCK_SHIFT = 7;
constexpr uint32_t BLOCK_SIZE = 1U << BLOCK_SHIFT;
constexpr uint32_t BLOCK_MASK = BLOCK_SIZE - 1U;
constexpr uint32_t PACKED_ROW_BYTES = 656U;

class A5KvafBatchGetPrepareKernel {
public:
    __aicore__ inline A5KvafBatchGetPrepareKernel(
        const A5KvafBatchGetPrepareTilingData* tiling)
        : tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR sourceTokenIds,
        GM_ADDR destinationSlots,
        GM_ADDR missCounts,
        GM_ADDR residentPoolIndices,
        GM_ADDR residentBlockTable,
        GM_ADDR cacheTokens,
        GM_ADDR candidateLens,
        GM_ADDR logicalSeqLengths,
        GM_ADDR indices,
        GM_ADDR offsets,
        GM_ADDR mask,
        GM_ADDR outputIndices,
        GM_ADDR residentSeqLengths)
    {
        coreIdx_ = GetBlockIdx();
        sourceTokenIdsGm_.SetGlobalBuffer((__gm__ int32_t*)sourceTokenIds);
        destinationSlotsGm_.SetGlobalBuffer((__gm__ int32_t*)destinationSlots);
        missCountsGm_.SetGlobalBuffer((__gm__ int32_t*)missCounts);
        residentPoolIndicesGm_.SetGlobalBuffer((__gm__ int32_t*)residentPoolIndices);
        residentBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)residentBlockTable);
        cacheTokensGm_.SetGlobalBuffer((__gm__ int32_t*)cacheTokens);
        candidateLensGm_.SetGlobalBuffer((__gm__ int32_t*)candidateLens);
        logicalSeqLengthsGm_.SetGlobalBuffer((__gm__ int32_t*)logicalSeqLengths);
        indicesGm_.SetGlobalBuffer((__gm__ int32_t*)indices);
        offsetsGm_.SetGlobalBuffer((__gm__ int32_t*)offsets);
        maskGm_.SetGlobalBuffer((__gm__ uint8_t*)mask);
        outputIndicesGm_.SetGlobalBuffer((__gm__ int32_t*)outputIndices);
        residentSeqLengthsGm_.SetGlobalBuffer((__gm__ int32_t*)residentSeqLengths);
    }

    __aicore__ inline void Process()
    {
        // Keep one AIV as the sole metadata writer. The uint8 mask and compact
        // int32 length vector otherwise allow different cores to false-share
        // one GM cache line on A5. This bring-up path favors a race-free ABI;
        // resolver parallelization belongs to the later optimized variant.
        if (coreIdx_ == 0U) {
            for (uint32_t batch = 0; batch < tiling_->batchSize; ++batch) {
                ProcessRow(batch);
                WriteResidentLength(batch);
            }
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t batch)
    {
        const int32_t rawMissCount = missCountsGm_.GetValue(batch);
        const uint32_t missCount =
            rawMissCount > 0 &&
                    rawMissCount <= static_cast<int32_t>(tiling_->requestCapacity)
                ? static_cast<uint32_t>(rawMissCount)
                : 0U;
        const int32_t poolRow = residentPoolIndicesGm_.GetValue(batch);
        const uint64_t flatBase =
            static_cast<uint64_t>(batch) * tiling_->requestCapacity;
        const uint64_t residentTableBase =
            static_cast<uint64_t>(batch) * tiling_->residentBlockCapacity;
        // AccessTable is row-major [pool_row][layer][logical_block]. Keep all
        // layers of one request row adjacent so Put/Get share the same layout.
        const uint64_t requestBase =
            static_cast<uint64_t>(poolRow) * tiling_->numLayers * tiling_->qcap;
        const uint64_t layerBase =
            requestBase + static_cast<uint64_t>(tiling_->layerId) * tiling_->qcap;

        for (uint32_t column = 0; column < tiling_->requestCapacity; ++column) {
            const uint64_t flat = flatBase + column;
            int32_t tableIndex = 0;
            int32_t byteOffset = 0;
            int32_t outputIndex = 0;
            uint8_t active = 0;
            if (column < missCount) {
                const int32_t sourceToken = sourceTokenIdsGm_.GetValue(flat);
                const int32_t destinationSlot =
                    destinationSlotsGm_.GetValue(flat);
                const uint32_t logicalBlock =
                    static_cast<uint32_t>(sourceToken) >> BLOCK_SHIFT;
                const uint32_t destinationBlock =
                    static_cast<uint32_t>(destinationSlot) >> BLOCK_SHIFT;
                const int32_t physicalBlock = residentBlockTableGm_.GetValue(
                    residentTableBase + destinationBlock);
                tableIndex = static_cast<int32_t>(layerBase + logicalBlock);
                byteOffset = static_cast<int32_t>(
                    (static_cast<uint32_t>(sourceToken) & BLOCK_MASK) *
                    PACKED_ROW_BYTES);
                outputIndex = static_cast<int32_t>(
                    static_cast<uint64_t>(physicalBlock) * BLOCK_SIZE +
                    (static_cast<uint32_t>(destinationSlot) & BLOCK_MASK));
                active = 1;
            }
            indicesGm_.SetValue(flat, tableIndex);
            offsetsGm_.SetValue(flat, byteOffset);
            maskGm_.SetValue(flat, active);
            outputIndicesGm_.SetValue(flat, outputIndex);
        }

    }

    __aicore__ inline void WriteResidentLength(uint32_t batch)
    {
        const int32_t cacheTokens = cacheTokensGm_.GetValue(batch);
        const int32_t candidateLen = candidateLensGm_.GetValue(batch);
        const int32_t logicalLen = logicalSeqLengthsGm_.GetValue(batch);
        int32_t residentLen = 0;
        if (cacheTokens >= static_cast<int32_t>(tiling_->requestCapacity) &&
            candidateLen >= static_cast<int32_t>(tiling_->requestCapacity) &&
            logicalLen >= candidateLen) {
            residentLen = cacheTokens + logicalLen - candidateLen;
        } else if (cacheTokens == 0 && logicalLen >= 0) {
            residentLen = logicalLen;
        }
        residentSeqLengthsGm_.SetValue(batch, residentLen);
    }

private:
    const A5KvafBatchGetPrepareTilingData* tiling_;
    uint32_t coreIdx_ = 0;
    GlobalTensor<int32_t> sourceTokenIdsGm_;
    GlobalTensor<int32_t> destinationSlotsGm_;
    GlobalTensor<int32_t> missCountsGm_;
    GlobalTensor<int32_t> residentPoolIndicesGm_;
    GlobalTensor<int32_t> residentBlockTableGm_;
    GlobalTensor<int32_t> cacheTokensGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> logicalSeqLengthsGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<int32_t> offsetsGm_;
    GlobalTensor<uint8_t> maskGm_;
    GlobalTensor<int32_t> outputIndicesGm_;
    GlobalTensor<int32_t> residentSeqLengthsGm_;
};
} // namespace

extern "C" __global__ __aicore__ void a5_kvaf_batch_get_prepare(
    GM_ADDR sourceTokenIds,
    GM_ADDR destinationSlots,
    GM_ADDR missCounts,
    GM_ADDR residentPoolIndices,
    GM_ADDR residentBlockTable,
    GM_ADDR cacheTokens,
    GM_ADDR candidateLens,
    GM_ADDR logicalSeqLengths,
    GM_ADDR indicesBuffer,
    GM_ADDR offsetsBuffer,
    GM_ADDR maskBuffer,
    GM_ADDR outputIndicesBuffer,
    GM_ADDR residentSeqLengthsBuffer,
    GM_ADDR indicesOut,
    GM_ADDR offsetsOut,
    GM_ADDR maskOut,
    GM_ADDR outputIndicesOut,
    GM_ADDR residentSeqLengthsOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)indicesBuffer;
    (void)offsetsBuffer;
    (void)maskBuffer;
    (void)outputIndicesBuffer;
    (void)residentSeqLengthsBuffer;
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5KvafBatchGetPrepareTilingData);
    GET_TILING_DATA(tilingData, tiling);
    A5KvafBatchGetPrepareKernel op(&tilingData);
    op.Init(
        sourceTokenIds, destinationSlots, missCounts, residentPoolIndices,
        residentBlockTable, cacheTokens, candidateLens, logicalSeqLengths,
        indicesOut, offsetsOut, maskOut, outputIndicesOut,
        residentSeqLengthsOut);
    op.Process();
}
