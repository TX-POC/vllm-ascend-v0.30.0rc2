/** Resolve LIDU output into the current mask-based KVAF BatchGet ABI. */

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "a5_kvaf_batch_get_prepare_tiling.h"

namespace {
using namespace AscendC;

constexpr uint32_t BLOCK_SHIFT = 7;
constexpr uint32_t BLOCK_SIZE = 1U << BLOCK_SHIFT;
constexpr uint32_t BLOCK_MASK = BLOCK_SIZE - 1U;
constexpr uint32_t PACKED_ROW_BYTES = 656U;
constexpr uint32_t MASK_CHUNK_BYTES = 2048U;
constexpr uint32_t RESIDENT_LENGTH_CHUNK = 512U;
constexpr uint32_t SIMT_BATCH_CHUNK = RESIDENT_LENGTH_CHUNK;
constexpr uint32_t SIMT_THREADS = 2048U;
constexpr uint32_t SIMT_MIN_WORK_ITEMS = 128U;
constexpr uint32_t ROW_OFFSETS_BUFFER_BYTES =
    ((SIMT_BATCH_CHUNK + 1U) * sizeof(uint32_t) + 31U) & ~31U;
constexpr uint32_t LAYER_BASES_BUFFER_BYTES =
    (SIMT_BATCH_CHUNK * sizeof(int32_t) + 31U) & ~31U;

// A 2048-thread VF has only 16 registers per thread. Scalar code resolves all
// UB metadata and GM row bases before launching one row-local VF per request.
// Keep source and destination temporaries in disjoint scopes so the compiler
// can reuse registers while generating all descriptors in that single VF.
__simt_vf__ __aicore__ LAUNCH_BOUND(2048)
inline void GenerateKvafDescriptorsSimt(
    __gm__ const int32_t* sourceTokenIds,
    __gm__ const int32_t* destinationSlots,
    __gm__ const int32_t* residentBlockTable,
    __gm__ int32_t* indices,
    __gm__ int32_t* offsets,
    __gm__ int32_t* outputIndices,
    int32_t layerBase,
    uint32_t missCount)
{
    for (uint32_t column = threadIdx.x; column < missCount;
         column += SIMT_THREADS) {
        {
            const uint32_t sourceToken =
                static_cast<uint32_t>(sourceTokenIds[column]);
            indices[column] = static_cast<int32_t>(
                static_cast<uint32_t>(layerBase) +
                (sourceToken >> BLOCK_SHIFT));
            offsets[column] = static_cast<int32_t>(
                (sourceToken & BLOCK_MASK) * PACKED_ROW_BYTES);
        }
        {
            const uint32_t destinationSlot =
                static_cast<uint32_t>(destinationSlots[column]);
            outputIndices[column] = static_cast<int32_t>(
                static_cast<uint32_t>(residentBlockTable[
                    destinationSlot >> BLOCK_SHIFT]) * BLOCK_SIZE +
                (destinationSlot & BLOCK_MASK));
        }
    }
}

class A5KvafBatchGetPrepareKernel {
public:
    __aicore__ inline A5KvafBatchGetPrepareKernel(
        TPipe* pipe,
        const A5KvafBatchGetPrepareTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
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
        pipe_->InitBuffer(zeroMaskBuf_, MASK_CHUNK_BYTES);
        pipe_->InitBuffer(oneMaskBuf_, MASK_CHUNK_BYTES);
        pipe_->InitBuffer(
            residentLengthBuf_, RESIDENT_LENGTH_CHUNK * sizeof(int32_t));
        pipe_->InitBuffer(rowOffsetsBuf_, ROW_OFFSETS_BUFFER_BYTES);
        pipe_->InitBuffer(layerBasesBuf_, LAYER_BASES_BUFFER_BYTES);
        sourceTokenIdsRaw_ = (__gm__ const int32_t*)sourceTokenIds;
        destinationSlotsRaw_ = (__gm__ const int32_t*)destinationSlots;
        residentBlockTableRaw_ = (__gm__ const int32_t*)residentBlockTable;
        indicesRaw_ = (__gm__ int32_t*)indices;
        offsetsRaw_ = (__gm__ int32_t*)offsets;
        outputIndicesRaw_ = (__gm__ int32_t*)outputIndices;
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
        // Keep one AIV as the sole metadata owner. Descriptor generation uses
        // intra-core SIMT threads; it does not change the one-core tiling or
        // introduce cross-core writes to compact metadata cache lines.
        if (coreIdx_ == 0U) {
            InitializeMaskTemplates();
            LocalTensor<int32_t> residentLengths =
                residentLengthBuf_.Get<int32_t>();
            LocalTensor<uint32_t> rowOffsets =
                rowOffsetsBuf_.Get<uint32_t>();
            LocalTensor<int32_t> layerBases =
                layerBasesBuf_.Get<int32_t>();
            for (uint32_t batchBase = 0U; batchBase < tiling_->batchSize;
                 batchBase += SIMT_BATCH_CHUNK) {
                const uint32_t batchCount = MinU32(
                    SIMT_BATCH_CHUNK, tiling_->batchSize - batchBase);
                const uint32_t totalMissCount = PrepareBatchChunk(
                    batchBase,
                    batchCount,
                    rowOffsets,
                    layerBases,
                    residentLengths);
                if (totalMissCount < SIMT_MIN_WORK_ITEMS) {
                    GenerateDescriptorsScalar(
                        batchBase, batchCount, rowOffsets, layerBases);
                } else {
                    LaunchDescriptorSimt(
                        batchBase,
                        batchCount,
                        rowOffsets,
                        layerBases);
                }
                WriteResidentLengths(
                    batchBase, batchCount, residentLengths);
            }
        }
    }

private:
    __aicore__ inline uint32_t MinU32(
        uint32_t left, uint32_t right) const
    {
        return left < right ? left : right;
    }

    __aicore__ inline void InitializeMaskTemplates()
    {
        LocalTensor<uint8_t> zeros = zeroMaskBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> ones = oneMaskBuf_.Get<uint8_t>();
        Duplicate(zeros, static_cast<uint8_t>(0), MASK_CHUNK_BYTES);
        Duplicate(ones, static_cast<uint8_t>(1), MASK_CHUNK_BYTES);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StoreMaskChunks(
        uint64_t outputBase,
        uint32_t count,
        LocalTensor<uint8_t> maskTemplate)
    {
        for (uint32_t offset = 0; offset < count;
             offset += MASK_CHUNK_BYTES) {
            const uint32_t chunk = MinU32(
                MASK_CHUNK_BYTES, count - offset);
            DataCopyExtParams params{1, chunk, 0, 0, 0};
            DataCopyPad<uint8_t, PaddingMode::Normal>(
                maskGm_[outputBase + offset], maskTemplate, params);
        }
    }

    __aicore__ inline void WriteMask(
        uint64_t outputBase, uint32_t missCount)
    {
        LocalTensor<uint8_t> zeros = zeroMaskBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> ones = oneMaskBuf_.Get<uint8_t>();
        if (missCount == tiling_->requestCapacity) {
            StoreMaskChunks(outputBase, missCount, ones);
            return;
        }

        // LIDU guarantees one contiguous miss prefix.  Publish a zero row
        // with MTE3 and overwrite only that prefix with the all-one template,
        // avoiding requestCapacity scalar GM stores.
        StoreMaskChunks(outputBase, tiling_->requestCapacity, zeros);
        if (missCount != 0U) {
            StoreMaskChunks(outputBase, missCount, ones);
        }
    }

    __aicore__ inline uint32_t SanitizeMissCount(uint32_t batch)
    {
        const int32_t rawMissCount = missCountsGm_.GetValue(batch);
        return rawMissCount > 0 &&
                rawMissCount <=
                    static_cast<int32_t>(tiling_->requestCapacity)
            ? static_cast<uint32_t>(rawMissCount)
            : 0U;
    }

    __aicore__ inline int32_t CalculateLayerBase(uint32_t batch)
    {
        const int32_t poolRow = residentPoolIndicesGm_.GetValue(batch);
        // AccessTable is row-major [pool_row][layer][logical_block]. Keep all
        // layers of one request row adjacent so Put/Get share the same layout.
        const uint64_t requestBase =
            static_cast<uint64_t>(poolRow) * tiling_->numLayers * tiling_->qcap;
        const uint64_t layerBase =
            requestBase + static_cast<uint64_t>(tiling_->layerId) * tiling_->qcap;
        return static_cast<int32_t>(layerBase);
    }

    __aicore__ inline uint32_t PrepareBatchChunk(
        uint32_t batchBase,
        uint32_t batchCount,
        LocalTensor<uint32_t> rowOffsets,
        LocalTensor<int32_t> layerBases,
        LocalTensor<int32_t> residentLengths)
    {
        uint32_t totalMissCount = 0U;
        rowOffsets.SetValue(0U, 0U);
        for (uint32_t localBatch = 0U; localBatch < batchCount;
             ++localBatch) {
            const uint32_t globalBatch = batchBase + localBatch;
            const uint32_t missCount = SanitizeMissCount(globalBatch);
            totalMissCount += missCount;
            rowOffsets.SetValue(localBatch + 1U, totalMissCount);
            layerBases.SetValue(
                localBatch, CalculateLayerBase(globalBatch));
            residentLengths.SetValue(
                localBatch, CalculateResidentLength(globalBatch));
            WriteMask(
                static_cast<uint64_t>(globalBatch) *
                    tiling_->requestCapacity,
                missCount);
        }
        return totalMissCount;
    }

    __aicore__ inline void GenerateDescriptorsScalar(
        uint32_t batchBase,
        uint32_t batchCount,
        LocalTensor<uint32_t> rowOffsets,
        LocalTensor<int32_t> layerBases)
    {
        for (uint32_t localBatch = 0U; localBatch < batchCount;
             ++localBatch) {
            const uint32_t globalBatch = batchBase + localBatch;
            const uint32_t missCount =
                rowOffsets.GetValue(localBatch + 1U) -
                rowOffsets.GetValue(localBatch);
            const int32_t layerBase = layerBases.GetValue(localBatch);
            const uint64_t flatBase =
                static_cast<uint64_t>(globalBatch) * tiling_->requestCapacity;
            const uint64_t residentTableBase =
                static_cast<uint64_t>(globalBatch) *
                tiling_->residentBlockCapacity;

            // Only the miss prefix is consumed by BatchGet. Leave inactive
            // descriptor entries caller-owned.
            for (uint32_t column = 0U; column < missCount; ++column) {
                const uint64_t flat = flatBase + column;
                const int32_t sourceToken = sourceTokenIdsGm_.GetValue(flat);
                const int32_t destinationSlot =
                    destinationSlotsGm_.GetValue(flat);
                const uint32_t logicalBlock =
                    static_cast<uint32_t>(sourceToken) >> BLOCK_SHIFT;
                const uint32_t destinationBlock =
                    static_cast<uint32_t>(destinationSlot) >> BLOCK_SHIFT;
                const int32_t physicalBlock =
                    residentBlockTableGm_.GetValue(
                        residentTableBase + destinationBlock);
                indicesGm_.SetValue(
                    flat,
                    static_cast<int32_t>(
                        static_cast<uint32_t>(layerBase) + logicalBlock));
                offsetsGm_.SetValue(
                    flat,
                    static_cast<int32_t>(
                        (static_cast<uint32_t>(sourceToken) & BLOCK_MASK) *
                        PACKED_ROW_BYTES));
                outputIndicesGm_.SetValue(
                    flat,
                    static_cast<int32_t>(
                        static_cast<uint64_t>(physicalBlock) * BLOCK_SIZE +
                        (static_cast<uint32_t>(destinationSlot) &
                         BLOCK_MASK)));
            }
        }
    }

    __aicore__ inline void LaunchDescriptorSimt(
        uint32_t batchBase,
        uint32_t batchCount,
        LocalTensor<uint32_t> rowOffsets,
        LocalTensor<int32_t> layerBases)
    {
        for (uint32_t localBatch = 0U; localBatch < batchCount;
             ++localBatch) {
            const uint32_t missCount =
                rowOffsets.GetValue(localBatch + 1U) -
                rowOffsets.GetValue(localBatch);
            if (missCount == 0U) {
                continue;
            }
            const uint32_t globalBatch = batchBase + localBatch;
            const uint64_t descriptorBase =
                static_cast<uint64_t>(globalBatch) *
                tiling_->requestCapacity;
            const uint64_t residentTableBase =
                static_cast<uint64_t>(globalBatch) *
                tiling_->residentBlockCapacity;
            asc_vf_call<GenerateKvafDescriptorsSimt>(
                dim3{SIMT_THREADS, 1U, 1U},
                sourceTokenIdsRaw_ + descriptorBase,
                destinationSlotsRaw_ + descriptorBase,
                residentBlockTableRaw_ + residentTableBase,
                indicesRaw_ + descriptorBase,
                offsetsRaw_ + descriptorBase,
                outputIndicesRaw_ + descriptorBase,
                layerBases.GetValue(localBatch),
                missCount);
        }
    }

    __aicore__ inline int32_t CalculateResidentLength(uint32_t batch)
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
        return residentLen;
    }

    __aicore__ inline void WriteResidentLengths(
        uint32_t outputOffset,
        uint32_t count,
        LocalTensor<int32_t> local)
    {
        const event_t scalarToMte3 = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(scalarToMte3);
        WaitFlag<HardEvent::S_MTE3>(scalarToMte3);
        DataCopyExtParams params{
            1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad<int32_t, PaddingMode::Normal>(
            residentSeqLengthsGm_[outputOffset], local, params);
        const event_t mte3ToScalar = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(mte3ToScalar);
        WaitFlag<HardEvent::MTE3_S>(mte3ToScalar);
    }

private:
    TPipe* pipe_;
    const A5KvafBatchGetPrepareTilingData* tiling_;
    uint32_t coreIdx_ = 0;
    __gm__ const int32_t* sourceTokenIdsRaw_;
    __gm__ const int32_t* destinationSlotsRaw_;
    __gm__ const int32_t* residentBlockTableRaw_;
    __gm__ int32_t* indicesRaw_;
    __gm__ int32_t* offsetsRaw_;
    __gm__ int32_t* outputIndicesRaw_;
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
    TBuf<TPosition::VECCALC> zeroMaskBuf_;
    TBuf<TPosition::VECCALC> oneMaskBuf_;
    TBuf<TPosition::VECCALC> residentLengthBuf_;
    TBuf<TPosition::VECCALC> rowOffsetsBuf_;
    TBuf<TPosition::VECCALC> layerBasesBuf_;
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
    TPipe pipe;
    A5KvafBatchGetPrepareKernel op(&pipe, &tilingData);
    op.Init(
        sourceTokenIds, destinationSlots, missCounts, residentPoolIndices,
        residentBlockTable, cacheTokens, candidateLens, logicalSeqLengths,
        indicesOut, offsetsOut, maskOut, outputIndicesOut,
        residentSeqLengthsOut);
    op.Process();
}
