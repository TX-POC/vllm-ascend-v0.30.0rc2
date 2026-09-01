// P7.1 standalone dynamic Direct-HCCP executor host contract.
//
// This op deliberately has no C8 input or source include.  Its only score
// boundary is CandidateSeed GM already published by an upstream exact-score
// consumer.  All mutable transport lifetime is supplied as graph-visible
// tensors so graph replay cannot inherit a hidden process-local queue state.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "../op_kernel/a5_score_receipt_speculative_executor_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
enum InputIndex : size_t {
    kCandidateSeeds = 0,
    kCandidateCounts,
    kCandidateReadyEpoch,
    kCandidateStatus,
    kStreamingEpoch,
    kSourceLens,
    kKvObjectAddressDomain,
    kReqPoolEntries,
    kKvObjectMap,
    kKvObjectLens,
    kKvObjectSlotsSnapshot,
    kPrivateSourceToSlot,
    kPrivateObjectToSlot,
    kPrivateReadyTable,
    kPrivateArenaHead,
    kPrivatePackedOutput,
    kFusedGetControl,
    kQueueLeases,
    kQueueEpoch,
    kQueueGeneration,
    kQueuePostSequence,
    kQueueState,
    kQueueQuarantine,
    kQueueRequestSourceOffsets,
    kQueueRequestObjectOffsets,
    kQueueRequestOutputIndices,
    kQueueRequestCount,
    kPrivateReadyEpoch,
    kPrivateReadyGeneration,
    kWorkerStats,
    kRowsPerEntry,
    kInputCount,
};

enum OutputIndex : size_t {
    kPrivateSourceToSlotOut = 0,
    kPrivateObjectToSlotOut,
    kPrivateReadyTableOut,
    kPrivateArenaHeadOut,
    kPrivatePackedOutputOut,
    kQueueLeasesOut,
    kQueueEpochOut,
    kQueueGenerationOut,
    kQueuePostSequenceOut,
    kQueueStateOut,
    kQueueQuarantineOut,
    kQueueRequestSourceOffsetsOut,
    kQueueRequestObjectOffsetsOut,
    kQueueRequestOutputIndicesOut,
    kQueueRequestCountOut,
    kPrivateReadyEpochOut,
    kPrivateReadyGenerationOut,
    kWorkerStatsOut,
    kOutputCount,
};

constexpr size_t kAliasInputs[kOutputCount] = {
    kPrivateSourceToSlot, kPrivateObjectToSlot, kPrivateReadyTable,
    kPrivateArenaHead, kPrivatePackedOutput, kQueueLeases, kQueueEpoch,
    kQueueGeneration, kQueuePostSequence, kQueueState, kQueueQuarantine,
    kQueueRequestSourceOffsets, kQueueRequestObjectOffsets,
    kQueueRequestOutputIndices, kQueueRequestCount, kPrivateReadyEpoch,
    kPrivateReadyGeneration, kWorkerStats,
};

constexpr uint32_t kControlWords = 16U;
constexpr uint32_t kPackedKvRowBytes = 656U;

// Keep the host-side failure receipt self-contained.  CANN reports every
// tiling callback failure to ACLNN as the same EZ1008 status, so returning
// GRAPH_FAILED without this context turns one concrete contract failure into
// a multi-round remote investigation.  The receipt is emitted only on an
// error and groups all 31 inputs into four short, copyable lines.
std::string ShapeDebugText(const gert::Shape* shape)
{
    if (shape == nullptr) {
        return "<null>";
    }
    std::string text{"["};
    for (int64_t dim = 0; dim < shape->GetDimNum(); ++dim) {
        if (dim != 0) {
            text += ",";
        }
        text += std::to_string(shape->GetDim(dim));
    }
    text += "]";
    return text;
}

void LogP7TilingInputs(const gert::TilingContext* context)
{
    if (context == nullptr) {
        std::fprintf(stderr, "P7_1_TILING_INPUTS context=<null>\n");
        std::fflush(stderr);
        return;
    }
    for (size_t begin = 0; begin < kInputCount; begin += 8U) {
        const size_t end = std::min(begin + 8U, kInputCount);
        std::fprintf(stderr, "P7_1_TILING_INPUTS range=%zu-%zu", begin,
                     end - 1U);
        for (size_t index = begin; index < end; ++index) {
            const auto* inputShape = context->GetInputShape(index);
            const auto* inputDesc = context->GetInputDesc(index);
            const int dtype = inputDesc == nullptr
                ? -1
                : static_cast<int>(inputDesc->GetDataType());
            const gert::Shape* storage = inputShape == nullptr
                ? nullptr
                : &inputShape->GetStorageShape();
            const std::string shape = ShapeDebugText(storage);
            std::fprintf(stderr, " i%zu=d%d:s%s", index, dtype,
                         shape.c_str());
        }
        std::fprintf(stderr, "\n");
    }
    std::fflush(stderr);
}

void LogP7TilingFailure(const char* stage, const gert::TilingContext* context,
                        size_t input, int expectedDtype, int actualDtype,
                        uint32_t aicCount, uint32_t aivCount,
                        uint32_t workerCount, uint32_t blockDim,
                        ge::graphStatus scheduleStatus)
{
    const char* node = context != nullptr && context->GetNodeName() != nullptr
        ? context->GetNodeName()
        : "A5ScoreReceiptSpeculativeExecutor";
    std::fprintf(
        stderr,
        "P7_1_TILING_DIAG stage=%s node=%s input=%zu expected_dtype=%d "
        "actual_dtype=%d aic=%u aiv=%u workers=%u block_dim=%u "
        "schedule_status=%d\n",
        stage, node, input, expectedDtype, actualDtype, aicCount, aivCount,
        workerCount, blockDim, static_cast<int>(scheduleStatus));
    LogP7TilingInputs(context);
}

bool IsPositiveVector(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) > 0;
}

bool IsPositiveMatrix(const gert::Shape& shape)
{
    return shape.GetDimNum() == 2 && shape.GetDim(0) > 0 &&
           shape.GetDim(1) > 0;
}

bool IsCandidateSeedTensor(const gert::Shape& shape, uint32_t* receipts)
{
    if (receipts == nullptr || shape.GetDimNum() != 3 ||
        shape.GetDim(0) <= 0 ||
        shape.GetDim(1) != kA5ScoreReceiptExecutorReceiptCapacity ||
        shape.GetDim(2) != kA5ScoreReceiptExecutorSeedWords ||
        static_cast<uint64_t>(shape.GetDim(0)) >
            std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *receipts = static_cast<uint32_t>(shape.GetDim(0));
    return true;
}

bool ValidateGeometry(const gert::Shape* const* inputs,
                      uint32_t* receiptCount, uint32_t* rowCount,
                      uint32_t* sourceCapacity, uint32_t* poolSize,
                      uint32_t* objectCapacity, uint32_t* queueCount,
                      uint32_t* privateArenaCapacity,
                      uint32_t* privatePackedRows,
                      uint32_t* workerStatsRows,
                      const char** failureStage)
{
    if (failureStage != nullptr) {
        *failureStage = "unknown";
    }
    if (inputs == nullptr || receiptCount == nullptr || rowCount == nullptr ||
        sourceCapacity == nullptr || poolSize == nullptr ||
        objectCapacity == nullptr || queueCount == nullptr ||
        privateArenaCapacity == nullptr || privatePackedRows == nullptr ||
        workerStatsRows == nullptr) {
        if (failureStage != nullptr) {
            *failureStage = "geometry_argument";
        }
        return false;
    }
    for (size_t index = 0; index < kInputCount; ++index) {
        if (inputs[index] == nullptr) {
            if (failureStage != nullptr) {
                *failureStage = "geometry_missing_shape";
            }
            return false;
        }
    }
    const gert::Shape& candidateSeeds = *inputs[kCandidateSeeds];
    const gert::Shape& candidateCounts = *inputs[kCandidateCounts];
    const gert::Shape& candidateReadyEpoch = *inputs[kCandidateReadyEpoch];
    const gert::Shape& candidateStatus = *inputs[kCandidateStatus];
    const gert::Shape& streamingEpoch = *inputs[kStreamingEpoch];
    const gert::Shape& sourceLens = *inputs[kSourceLens];
    const gert::Shape& addressDomain = *inputs[kKvObjectAddressDomain];
    const gert::Shape& reqPoolEntries = *inputs[kReqPoolEntries];
    const gert::Shape& kvObjectMap = *inputs[kKvObjectMap];
    const gert::Shape& kvObjectLens = *inputs[kKvObjectLens];
    const gert::Shape& snapshot = *inputs[kKvObjectSlotsSnapshot];
    const gert::Shape& privateSource = *inputs[kPrivateSourceToSlot];
    const gert::Shape& privateObject = *inputs[kPrivateObjectToSlot];
    const gert::Shape& privateReady = *inputs[kPrivateReadyTable];
    const gert::Shape& privateArenaHead = *inputs[kPrivateArenaHead];
    const gert::Shape& privateOutput = *inputs[kPrivatePackedOutput];
    const gert::Shape& control = *inputs[kFusedGetControl];
    const gert::Shape& queueLeases = *inputs[kQueueLeases];
    const gert::Shape& queueEpoch = *inputs[kQueueEpoch];
    const gert::Shape& queueGeneration = *inputs[kQueueGeneration];
    const gert::Shape& queuePostSequence = *inputs[kQueuePostSequence];
    const gert::Shape& queueState = *inputs[kQueueState];
    const gert::Shape& queueQuarantine = *inputs[kQueueQuarantine];
    const gert::Shape& requestSource = *inputs[kQueueRequestSourceOffsets];
    const gert::Shape& requestObject = *inputs[kQueueRequestObjectOffsets];
    const gert::Shape& requestOutput = *inputs[kQueueRequestOutputIndices];
    const gert::Shape& requestCount = *inputs[kQueueRequestCount];
    const gert::Shape& privateReadyEpoch = *inputs[kPrivateReadyEpoch];
    const gert::Shape& privateReadyGeneration = *inputs[kPrivateReadyGeneration];
    const gert::Shape& workerStats = *inputs[kWorkerStats];
    const gert::Shape& rowsPerEntry = *inputs[kRowsPerEntry];

    uint32_t receipts = 0U;
    if (!IsCandidateSeedTensor(candidateSeeds, &receipts) ||
        !IsPositiveVector(candidateCounts) ||
        !IsPositiveVector(candidateReadyEpoch) ||
        !IsPositiveVector(candidateStatus) ||
        candidateCounts.GetDim(0) != static_cast<int64_t>(receipts) ||
        candidateReadyEpoch.GetDim(0) != static_cast<int64_t>(receipts) ||
        candidateStatus.GetDim(0) != static_cast<int64_t>(receipts) ||
        !IsPositiveVector(streamingEpoch) || streamingEpoch.GetDim(0) != 1 ||
        !IsPositiveVector(sourceLens) ||
        !IsPositiveVector(addressDomain) || addressDomain.GetDim(0) != 1 ||
        !IsPositiveVector(reqPoolEntries) || !IsPositiveVector(kvObjectLens) ||
        !IsPositiveMatrix(kvObjectMap) || !IsPositiveMatrix(snapshot) ||
        !IsPositiveMatrix(privateSource) || !IsPositiveMatrix(privateObject) ||
        !IsPositiveMatrix(privateReady) || !IsPositiveVector(privateArenaHead) ||
        !IsPositiveMatrix(privateOutput) || !IsPositiveVector(control) ||
        control.GetDim(0) != static_cast<int64_t>(kControlWords) ||
        !IsPositiveVector(queueLeases) || !IsPositiveVector(queueEpoch) ||
        !IsPositiveVector(queueGeneration) ||
        !IsPositiveVector(queuePostSequence) || !IsPositiveVector(queueState) ||
        !IsPositiveVector(queueQuarantine) || !IsPositiveMatrix(requestSource) ||
        !IsPositiveMatrix(requestObject) || !IsPositiveMatrix(requestOutput) ||
        !IsPositiveVector(requestCount) || !IsPositiveMatrix(privateReadyEpoch) ||
        !IsPositiveMatrix(privateReadyGeneration) || !IsPositiveMatrix(workerStats) ||
        !IsPositiveVector(rowsPerEntry)) {
        if (failureStage != nullptr) {
            *failureStage = "geometry_rank_or_receipt";
        }
        return false;
    }

    const int64_t rows = reqPoolEntries.GetDim(0);
    const int64_t pools = kvObjectMap.GetDim(0);
    const int64_t sources = kvObjectMap.GetDim(1);
    const int64_t objects = snapshot.GetDim(1);
    const int64_t queues = privateArenaHead.GetDim(0);
    const int64_t arena = privateReady.GetDim(1);
    const int64_t statsRows = workerStats.GetDim(0);
    if (rows <= 0 || pools <= 0 || sources <= 0 || objects <= 0 || queues <= 0 ||
        arena <= 0 || statsRows <= 0 ||
        rows > std::numeric_limits<uint32_t>::max() ||
        pools > std::numeric_limits<uint32_t>::max() ||
        sources > std::numeric_limits<uint32_t>::max() ||
        objects > std::numeric_limits<uint32_t>::max() ||
        queues > std::numeric_limits<uint32_t>::max() ||
        arena > std::numeric_limits<uint32_t>::max() ||
        statsRows > std::numeric_limits<uint32_t>::max() ||
        sourceLens.GetDim(0) != rows || kvObjectLens.GetDim(0) != rows ||
        snapshot.GetDim(0) != pools ||
        privateSource.GetDim(0) != rows || privateSource.GetDim(1) != sources ||
        privateObject.GetDim(0) != pools || privateObject.GetDim(1) != objects ||
        privateReady.GetDim(0) != queues ||
        privateOutput.GetDim(0) != queues * arena ||
        privateOutput.GetDim(1) != static_cast<int64_t>(kPackedKvRowBytes) ||
        queueLeases.GetDim(0) != queues || queueEpoch.GetDim(0) != queues ||
        queueGeneration.GetDim(0) != queues ||
        queuePostSequence.GetDim(0) != queues || queueState.GetDim(0) != queues ||
        queueQuarantine.GetDim(0) != queues || requestSource.GetDim(0) != queues ||
        requestObject.GetDim(0) != queues || requestOutput.GetDim(0) != queues ||
        requestSource.GetDim(1) != kA5ScoreReceiptExecutorMicrobatchCapacity ||
        requestObject.GetDim(1) != kA5ScoreReceiptExecutorMicrobatchCapacity ||
        requestOutput.GetDim(1) != kA5ScoreReceiptExecutorMicrobatchCapacity ||
        requestCount.GetDim(0) != queues || privateReadyEpoch.GetDim(0) != queues ||
        privateReadyEpoch.GetDim(1) != arena || privateReadyGeneration.GetDim(0) != queues ||
        privateReadyGeneration.GetDim(1) != arena ||
        workerStats.GetDim(1) != kA5ScoreReceiptExecutorWorkerStatColumns ||
        rowsPerEntry.GetDim(0) != 1) {
        if (failureStage != nullptr) {
            *failureStage = "geometry_cross_tensor";
        }
        return false;
    }
    *receiptCount = receipts;
    *rowCount = static_cast<uint32_t>(rows);
    *sourceCapacity = static_cast<uint32_t>(sources);
    *poolSize = static_cast<uint32_t>(pools);
    *objectCapacity = static_cast<uint32_t>(objects);
    *queueCount = static_cast<uint32_t>(queues);
    *privateArenaCapacity = static_cast<uint32_t>(arena);
    *privatePackedRows = static_cast<uint32_t>(privateOutput.GetDim(0));
    *workerStatsRows = static_cast<uint32_t>(statsRows);
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingA5ScoreReceiptSpeculativeExecutor(
    gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        LogP7TilingFailure("bootstrap_context_or_platform", context, kInputCount,
                           -1, -1, 0U, 0U, 0U, 0U, ge::GRAPH_FAILED);
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* inputShapes[kInputCount] = {};
    const ge::DataType expected[kInputCount] = {
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
        ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_UINT8,
        ge::DT_INT64, ge::DT_INT32,
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32,
        ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32,
        ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
        ge::DT_INT32};
    for (size_t index = 0; index < kInputCount; ++index) {
        inputShapes[index] = context->GetInputShape(index) == nullptr
            ? nullptr : &context->GetInputShape(index)->GetStorageShape();
        if (inputShapes[index] == nullptr) {
            LogP7TilingFailure("missing_input_shape", context, index,
                               static_cast<int>(expected[index]), -1, 0U, 0U,
                               0U, 0U, ge::GRAPH_FAILED);
            return ge::GRAPH_FAILED;
        }
        if (context->GetInputDesc(index) == nullptr) {
            LogP7TilingFailure("missing_input_desc", context, index,
                               static_cast<int>(expected[index]), -1, 0U, 0U,
                               0U, 0U, ge::GRAPH_FAILED);
            return ge::GRAPH_FAILED;
        }
        const int actualDtype =
            static_cast<int>(context->GetInputDesc(index)->GetDataType());
        if (actualDtype != static_cast<int>(expected[index])) {
            LogP7TilingFailure("input_dtype", context, index,
                               static_cast<int>(expected[index]), actualDtype,
                               0U, 0U, 0U, 0U, ge::GRAPH_FAILED);
            return ge::GRAPH_FAILED;
        }
    }
    uint32_t receiptCount = 0U;
    uint32_t rowCount = 0U;
    uint32_t sourceCapacity = 0U;
    uint32_t poolSize = 0U;
    uint32_t objectCapacity = 0U;
    uint32_t queueCount = 0U;
    uint32_t privateArenaCapacity = 0U;
    uint32_t privatePackedRows = 0U;
    uint32_t workerStatsRows = 0U;
    const char* geometryFailure = "geometry_unknown";
    if (!ValidateGeometry(inputShapes, &receiptCount, &rowCount, &sourceCapacity,
                          &poolSize, &objectCapacity, &queueCount,
                          &privateArenaCapacity, &privatePackedRows,
                          &workerStatsRows, &geometryFailure)) {
        LogP7TilingFailure(geometryFailure, context, kInputCount, -1, -1,
                           0U, 0U, 0U, 0U, ge::GRAPH_FAILED);
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t availableAic = platform.GetCoreNumAic();
    const uint32_t availableAiv = platform.GetCoreNumAiv();
    // Candidate receipts are already compact.  Launching every available AIV
    // just to race a smaller dynamic HCCP queue pool adds lease/CAS churn and
    // does not create copy throughput.  This is a *logical task-count cap*,
    // not a static worker-to-queue affinity: any scheduled AIV can lease any queue,
    // and each worker walks multiple receipts with a stride of workerCount.
    uint32_t workerCount = availableAiv < receiptCount ? availableAiv : receiptCount;
    workerCount = workerCount < queueCount ? workerCount : queueCount;
    if (workerCount == 0U || workerStatsRows < workerCount) {
        LogP7TilingFailure("worker_capacity", context, kInputCount, -1, -1,
                           availableAic, availableAiv, workerCount, 0U,
                           ge::GRAPH_FAILED);
        return ge::GRAPH_FAILED;
    }

    // CANN 9.1 direct ACLNN requires the AIV-only execution geometry before
    // it materializes a typed tiling POD.  Requesting GetTilingData first
    // leaves the payload null and ACLNN reports only the opaque EZ1008
    // "tiling function" error.  This follows the known-good P5 ordering;
    // it is not a static Get-core partition.  `workerCount` is still the
    // dynamic number of receipts/queues to service in this launch.
    const uint32_t blockDim =
        platform.CalcTschBlockDim(workerCount, 0U, availableAiv);
    if (blockDim != workerCount) {
        LogP7TilingFailure("aiv_only_block_dim", context, kInputCount, -1,
                           -1, availableAic, availableAiv, workerCount,
                           blockDim, ge::GRAPH_FAILED);
        return ge::GRAPH_FAILED;
    }
    const ge::graphStatus scheduleStatus = context->SetScheduleMode(1U);
    context->SetBlockDim(blockDim);
    std::fprintf(
        stderr,
        "P7_1_TILING_PREALLOC aic=%u aiv=%u workers=%u block_dim=%u "
        "schedule_request=batch schedule_status=%d\n",
        availableAic, availableAiv, workerCount, blockDim,
        static_cast<int>(scheduleStatus));
    std::fflush(stderr);
    auto* tiling = context->GetTilingData<A5ScoreReceiptSpeculativeExecutorTilingData>();
    if (tiling == nullptr) {
        LogP7TilingFailure("typed_tiling_null", context, kInputCount, -1,
                           -1, availableAic, availableAiv, workerCount,
                           blockDim, scheduleStatus);
        return ge::GRAPH_FAILED;
    }
    tiling->workerCount = workerCount;
    tiling->receiptCount = receiptCount;
    tiling->rowCount = rowCount;
    tiling->sourceCapacity = sourceCapacity;
    tiling->poolSize = poolSize;
    tiling->kvObjectCapacity = objectCapacity;
    tiling->queueCount = queueCount;
    tiling->privateArenaCapacity = privateArenaCapacity;
    tiling->privatePackedRows = privatePackedRows;
    tiling->workerStatsRows = workerStatsRows;
    tiling->workerStatsColumns = kA5ScoreReceiptExecutorWorkerStatColumns;
    tiling->candidateSeedWords = kA5ScoreReceiptExecutorSeedWords;
    tiling->candidateCapacity = kA5ScoreReceiptExecutorReceiptCapacity;
    tiling->microbatchCapacity = kA5ScoreReceiptExecutorMicrobatchCapacity;
    tiling->microbatchSlots = kA5ScoreReceiptExecutorMicrobatchSlots;
    tiling->reserved = 0U;
    std::fprintf(
        stderr,
        "P7_1_TILING_READY aic=%u aiv=%u workers=%u block_dim=%u "
        "receipts=%u queues=%u rows=%u source=%u arena=%u\n",
        availableAic, availableAiv, workerCount, blockDim, receiptCount,
        queueCount, rowCount, sourceCapacity, privateArenaCapacity);
    std::fflush(stderr);
    if (context->GetWorkspaceSizes(1) != nullptr) {
        context->GetWorkspaceSizes(1)[0] = 0U;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferA5ScoreReceiptSpeculativeExecutorShape(
    gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t output = 0; output < kOutputCount; ++output) {
        const size_t input = kAliasInputs[output];
        if (context->GetInputShape(input) == nullptr ||
            context->GetOutputShape(output) == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *context->GetOutputShape(output) = *context->GetInputShape(input);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferA5ScoreReceiptSpeculativeExecutorDataType(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType outputTypes[kOutputCount] = {
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        ge::DT_UINT8, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64,
        ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64,
        ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64,
        ge::DT_INT64, ge::DT_INT64};
    for (size_t output = 0; output < kOutputCount; ++output) {
        context->SetOutputDataType(output, outputTypes[output]);
    }
    return ge::GRAPH_SUCCESS;
}

class A5ScoreReceiptSpeculativeExecutor : public OpDef {
public:
    explicit A5ScoreReceiptSpeculativeExecutor(const char* name) : OpDef(name)
    {
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        const std::vector<ge::DataType> int32Types = {ge::DT_INT32};
        const std::vector<ge::DataType> int64Types = {ge::DT_INT64};
        const std::vector<ge::DataType> uint8Types = {ge::DT_UINT8};
        const ge::DataType inputTypes[kInputCount] = {
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
            ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_UINT8,
            ge::DT_INT64, ge::DT_INT32,
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32,
            ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32,
            ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
            ge::DT_INT32};
        const char* inputNames[kInputCount] = {
            "candidate_seeds", "candidate_counts", "candidate_ready_epoch",
            "candidate_status", "streaming_epoch", "source_lens",
            "kv_object_address_domain", "req_pool_entries", "kv_object_map",
            "kv_object_lens", "kv_object_slots_snapshot",
            "private_source_to_slot", "private_object_to_slot",
            "private_ready_table", "private_arena_head",
            "private_packed_output", "fused_get_control", "queue_leases",
            "queue_epoch", "queue_generation", "queue_post_sequence",
            "queue_state", "queue_quarantine", "queue_request_source_offsets",
            "queue_request_object_offsets", "queue_request_output_indices",
            "queue_request_count", "private_ready_epoch",
            "private_ready_generation", "worker_stats", "rows_per_entry"};
        for (size_t index = 0; index < kInputCount; ++index) {
            const std::vector<ge::DataType>* types = &int32Types;
            if (inputTypes[index] == ge::DT_INT64) {
                types = &int64Types;
            } else if (inputTypes[index] == ge::DT_UINT8) {
                types = &uint8Types;
            }
            this->Input(inputNames[index]).ParamType(REQUIRED)
                .DataType(*types).Format(formats);
        }
        const ge::DataType outputTypes[kOutputCount] = {
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_UINT8, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64,
            ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64,
            ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64,
            ge::DT_INT64, ge::DT_INT64};
        // CANN ACLNN turns an input/output pair with the same name into one
        // mutable `...Ref` argument.  These are transaction-state refs, not
        // 18 independent outputs: keeping separate `_out` names produced a
        // 49-descriptor GetWorkspaceSize ABI and crashed before tiling.
        const char* outputNames[kOutputCount] = {
            "private_source_to_slot", "private_object_to_slot",
            "private_ready_table", "private_arena_head",
            "private_packed_output", "queue_leases", "queue_epoch",
            "queue_generation", "queue_post_sequence", "queue_state",
            "queue_quarantine", "queue_request_source_offsets",
            "queue_request_object_offsets", "queue_request_output_indices",
            "queue_request_count", "private_ready_epoch",
            "private_ready_generation", "worker_stats"};
        for (size_t output = 0; output < kOutputCount; ++output) {
            const std::vector<ge::DataType>* types = &int32Types;
            if (outputTypes[output] == ge::DT_INT64) {
                types = &int64Types;
            } else if (outputTypes[output] == ge::DT_UINT8) {
                types = &uint8Types;
            }
            this->Output(outputNames[output]).ParamType(REQUIRED)
                .DataType(*types).Format(formats);
        }
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value", "a5_score_receipt_speculative_executor")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore()
            .SetTiling(optiling::TilingA5ScoreReceiptSpeculativeExecutor)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5ScoreReceiptSpeculativeExecutor);

IMPL_OP_INFERSHAPE(A5ScoreReceiptSpeculativeExecutor)
    .InferShape(InferA5ScoreReceiptSpeculativeExecutorShape)
    .InferDataType(InferA5ScoreReceiptSpeculativeExecutorDataType);
}  // namespace ops
