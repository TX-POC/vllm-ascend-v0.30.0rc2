/**
 * Registration/tiling for the first real ExactScan + SpeculativeGet path.
 *
 * The P1 full-Get path contains no AIC work.  It therefore launches a single
 * 64-lane AIV task:
 *
 *   task lanes 0..1:  the two existing KVAF Direct-HCCP queue owners
 *   task lanes 2..63: ExactScan producers
 *
 * This preserves the task-lane positions used by the verified standalone
 * two-worker Get launch.  In P2 parallel mode each owner may compact the next
 * exact descriptor wave while its current one is in flight; strict serial
 * retains the blocking helper.  It is stronger than the old MIX logical layout but
 * is deliberately not described as a public physical-AIV affinity API.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "../op_kernel/a5_exact_scan_speculative_get_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
// Keep this positional ABI identical to the device-kernel signature.  The
// earlier P2 score/tables ABI deliberately does not coexist with this one:
// mixing the two would reinterpret C8 score workspace bytes as float scores.
constexpr size_t kScoreWorkspace = 0;
constexpr size_t kTileReceipts = 1;
constexpr size_t kStreamingEpoch = 2;
// C8 writes this per-row published source span only after the corresponding
// score-window receipt is valid.  P2 must consume it so stale/inactive
// receipt slots cannot be interpreted as valid score ranges.
constexpr size_t kScoreSourceLens = 3;
constexpr size_t kResidentThreshold = 4;
constexpr size_t kResidentThresholdValid = 5;
constexpr size_t kKvObjectSlotsSnapshot = 6;
constexpr size_t kReqPoolEntries = 7;
constexpr size_t kKvObjectMap = 8;
constexpr size_t kKvObjectLens = 9;
constexpr size_t kPrivateSourceToSlot = 10;
constexpr size_t kPrivateObjectToSlot = 11;
constexpr size_t kPrivateReady = 12;
constexpr size_t kPrivateArenaHead = 13;
constexpr size_t kPrivatePackedOutput = 14;
constexpr size_t kAddressTable = 15;
constexpr size_t kValidTable = 16;
constexpr size_t kFusedGetControl = 17;
constexpr size_t kPipelineState = 18;
constexpr size_t kExecutionMode = 19;
constexpr size_t kInputCount = 20;

constexpr size_t kPrivateSourceToSlotOut = 0;
constexpr size_t kPrivateObjectToSlotOut = 1;
constexpr size_t kPrivateReadyOut = 2;
constexpr size_t kPrivateArenaHeadOut = 3;
constexpr size_t kPrivatePackedOutputOut = 4;
constexpr size_t kCandidateSeeds = 5;
constexpr size_t kCandidateCounts = 6;
constexpr size_t kProducerStats = 7;
constexpr size_t kGetStats = 8;
constexpr size_t kRoleTrace = 9;
constexpr size_t kGetIndices = 10;
constexpr size_t kGetOffsets = 11;
constexpr size_t kGetOutputIndices = 12;
constexpr size_t kGetMask = 13;
constexpr size_t kGetCandidateSeedSlots = 14;
constexpr size_t kGetClaimNew = 15;
constexpr size_t kPipelineStateOut = 16;
constexpr size_t kOutputCount = 17;

constexpr uint32_t kAivCount = 64;
constexpr uint32_t kGetLaneCount = 2;
constexpr uint32_t kProducerCount = kAivCount - kGetLaneCount;
constexpr uint32_t kCacheLineWords = 8;
constexpr uint32_t kRingSlots = 2;
constexpr uint32_t kRecordsPerRingSlot = 32;
constexpr uint32_t kRecordWords = 3;
constexpr uint32_t kRingSlotRecordWord = 3 * kCacheLineWords;
constexpr uint32_t kRingWordsPerSlot =
    kRingSlotRecordWord + kRecordsPerRingSlot * kRecordWords;
constexpr uint32_t kPipelineWordsPerRing =
    kRingSlots * kRingWordsPerSlot + kCacheLineWords;
constexpr uint32_t kGetBootstrapRecords = 64;
constexpr uint32_t kGetWaveRecords = 256;
constexpr uint32_t kGetDescriptorBufferCount = 2;
constexpr uint32_t kFusedGetControlWords = 16;
constexpr uint32_t kPackedEntryBytes = 656;
constexpr uint32_t kRowsPerEntry = 128;
constexpr uint32_t kReceiptWords = 16;
constexpr uint32_t kReceiptTileSources = 128;
constexpr uint32_t kCandidateSeedWords = 5;
// ExactScan emits its normal five counters plus first-chunk, scan-done and
// both-rings-closed timestamps.  The final timestamp is what lets the strict
// serial reference prove that it waited for the terminal producer boundary.
// P5 adds receipt-poll and source-len skip accounting to the original
// producer telemetry.  Keep this width identical to the device writer and
// bridge allocation: producer_stats is [62, 10].
constexpr uint32_t kProducerStatColumns = 10;
// Get telemetry has the original eight lifecycle fields plus the strict
// serial-gate timestamp, the mode observed by the device, and a commutative
// descriptor fingerprint.  Words 14/15 additionally make the P2
// nonblocking CQ probes and alternate-buffer descriptor work observable.
constexpr uint32_t kGetStatColumns = 16;

bool IsVectorWithPositiveLength(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) > 0;
}

bool IsVectorOfLength(const gert::Shape& shape, int64_t length)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == length;
}

bool IsMatrix(const gert::Shape& shape, int64_t dim0, int64_t dim1)
{
    return shape.GetDimNum() == 2 && shape.GetDim(0) == dim0 &&
           shape.GetDim(1) == dim1;
}

bool ToU32(int64_t value, uint32_t* result)
{
    if (result == nullptr || value <= 0 ||
        value > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    *result = static_cast<uint32_t>(value);
    return true;
}

struct P5Geometry {
    uint32_t rowCount;
    uint32_t sourceCapacity;
    uint32_t poolSize;
    uint32_t kvObjectCapacity;
    uint32_t privateArenaCapacity;
    uint32_t tileReceiptCount;
    uint32_t candidateCapacityPerProducer;
    uint32_t scoreWorkspaceWords;
    uint32_t tableCapacity;
    uint32_t outputRows;
};

std::string ShapeDebugText(const gert::Shape& shape)
{
    std::string text{"["};
    for (int64_t dim = 0; dim < shape.GetDimNum(); ++dim) {
        if (dim != 0) {
            text += ",";
        }
        text += std::to_string(shape.GetDim(dim));
    }
    text += "]";
    return text;
}

// Host tiling failures otherwise arrive at Python as the undifferentiated
// EZ1008 error.  Keep the failure report short enough to paste back in one
// block, but complete enough to distinguish ABI, geometry and scheduler
// failures without another rebuild.  This deliberately uses libc stderr
// rather than the monorepo-only logging headers: the strict msopgen build of
// this standalone OPP has no dependency on the vLLM-Ascend logging closure.
void LogP5TilingBootstrapFailure(
    const char* stage, const gert::TilingContext* context, size_t input)
{
    const char* node =
        context != nullptr && context->GetNodeName() != nullptr
        ? context->GetNodeName()
        : "A5ExactScanSpeculativeGet";
    std::fprintf(stderr,
                 "P5_TILING_DIAG stage=%s node=%s input=%ld\\n", stage,
                 node, static_cast<long>(input));
    std::fflush(stderr);
}

void LogP5TilingContractFailure(
    const char* stage, const gert::TilingContext* context,
    const P5Geometry* geometry, uint32_t aicCount, uint32_t aivCount,
    uint32_t blockDim, ge::graphStatus scheduleStatus)
{
    const char* node = context->GetNodeName() == nullptr
        ? "A5ExactScanSpeculativeGet"
        : context->GetNodeName();
    const P5Geometry empty{};
    const P5Geometry& value = geometry == nullptr ? empty : *geometry;
    std::fprintf(
        stderr,
        "P5_TILING_DIAG stage=%s node=%s aic=%u aiv=%u requested_aiv=%u "
        "block_dim=%u schedule_status=%d rows=%u source=%u pool=%u object=%u "
        "arena=%u receipts=%u candidate_capacity=%u score_words=%u table=%u "
        "output_rows=%u\\n",
        stage, node, aicCount, aivCount, kAivCount, blockDim,
        static_cast<int>(scheduleStatus), value.rowCount, value.sourceCapacity,
        value.poolSize, value.kvObjectCapacity, value.privateArenaCapacity,
        value.tileReceiptCount, value.candidateCapacityPerProducer,
        value.scoreWorkspaceWords, value.tableCapacity, value.outputRows);
    std::fprintf(
        stderr,
        "P5_TILING_SHAPES_0_9 in0=%s in1=%s in2=%s in3=%s in4=%s in5=%s "
        "in6=%s in7=%s in8=%s in9=%s\\n",
        ShapeDebugText(context->GetInputShape(0)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(1)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(2)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(3)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(4)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(5)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(6)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(7)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(8)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(9)->GetStorageShape()).c_str());
    std::fprintf(
        stderr,
        "P5_TILING_SHAPES_10_19 in10=%s in11=%s in12=%s in13=%s in14=%s "
        "in15=%s in16=%s in17=%s in18=%s in19=%s\\n",
        ShapeDebugText(context->GetInputShape(10)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(11)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(12)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(13)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(14)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(15)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(16)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(17)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(18)->GetStorageShape()).c_str(),
        ShapeDebugText(context->GetInputShape(19)->GetStorageShape()).c_str());
    std::fprintf(
        stderr,
        "P5_TILING_DTYPES in0=%d,in1=%d,in2=%d,in3=%d,in4=%d,in5=%d,"
        "in6=%d,in7=%d,in8=%d,in9=%d,in10=%d,in11=%d,in12=%d,in13=%d,"
        "in14=%d,in15=%d,in16=%d,in17=%d,in18=%d,in19=%d\\n",
        static_cast<int>(context->GetInputDesc(0)->GetDataType()),
        static_cast<int>(context->GetInputDesc(1)->GetDataType()),
        static_cast<int>(context->GetInputDesc(2)->GetDataType()),
        static_cast<int>(context->GetInputDesc(3)->GetDataType()),
        static_cast<int>(context->GetInputDesc(4)->GetDataType()),
        static_cast<int>(context->GetInputDesc(5)->GetDataType()),
        static_cast<int>(context->GetInputDesc(6)->GetDataType()),
        static_cast<int>(context->GetInputDesc(7)->GetDataType()),
        static_cast<int>(context->GetInputDesc(8)->GetDataType()),
        static_cast<int>(context->GetInputDesc(9)->GetDataType()),
        static_cast<int>(context->GetInputDesc(10)->GetDataType()),
        static_cast<int>(context->GetInputDesc(11)->GetDataType()),
        static_cast<int>(context->GetInputDesc(12)->GetDataType()),
        static_cast<int>(context->GetInputDesc(13)->GetDataType()),
        static_cast<int>(context->GetInputDesc(14)->GetDataType()),
        static_cast<int>(context->GetInputDesc(15)->GetDataType()),
        static_cast<int>(context->GetInputDesc(16)->GetDataType()),
        static_cast<int>(context->GetInputDesc(17)->GetDataType()),
        static_cast<int>(context->GetInputDesc(18)->GetDataType()),
        static_cast<int>(context->GetInputDesc(19)->GetDataType()));
    std::fflush(stderr);
}

bool DeriveP5Geometry(
    const gert::Shape& scoreWorkspace, const gert::Shape& tileReceipts,
    const gert::Shape& streamingEpoch, const gert::Shape& scoreSourceLens,
    const gert::Shape& residentThreshold,
    const gert::Shape& residentThresholdValid,
    const gert::Shape& kvObjectSlotsSnapshot,
    const gert::Shape& reqPoolEntries, const gert::Shape& kvObjectMap,
    const gert::Shape& kvObjectLens,
    const gert::Shape& privateSourceToSlot,
    const gert::Shape& privateObjectToSlot,
    const gert::Shape& privateReady, const gert::Shape& privateArenaHead,
    const gert::Shape& privatePackedOutput, const gert::Shape& addressTable,
    const gert::Shape& validTable, const gert::Shape& control,
    const gert::Shape& pipelineState,
    const gert::Shape& executionMode, P5Geometry* geometry,
    const char** failureStage)
{
    const auto fail = [failureStage](const char* stage) {
        if (failureStage != nullptr) {
            *failureStage = stage;
        }
        return false;
    };
    // P5's input #3 is intentionally graph-static active-score metadata.
    // It classifies fixed receipt positions as inactive versus expected for
    // *this* forward before the consumer polls a ready epoch.  It must not
    // be wired to A5StreamingQuantLiGet's similarly shaped ProcessTopK
    // output, which is a terminal producer result and is too late for the
    // live executor protocol.
    if (geometry == nullptr || !IsVectorWithPositiveLength(scoreWorkspace) ||
        !IsVectorWithPositiveLength(residentThreshold) ||
        !IsVectorOfLength(scoreSourceLens, residentThreshold.GetDim(0)) ||
        !IsVectorOfLength(residentThresholdValid, residentThreshold.GetDim(0)) ||
        !IsVectorOfLength(reqPoolEntries, residentThreshold.GetDim(0)) ||
        !IsVectorOfLength(kvObjectLens, residentThreshold.GetDim(0)) ||
        kvObjectMap.GetDimNum() != 2 || kvObjectSlotsSnapshot.GetDimNum() != 2 ||
        privateSourceToSlot.GetDimNum() != 2 ||
        privateObjectToSlot.GetDimNum() != 2 || privateReady.GetDimNum() != 2 ||
        !IsVectorOfLength(privateArenaHead, kGetLaneCount) ||
        privatePackedOutput.GetDimNum() != 2 ||
        privatePackedOutput.GetDim(1) != static_cast<int64_t>(kPackedEntryBytes) ||
        !IsVectorWithPositiveLength(addressTable) ||
        !IsVectorOfLength(validTable, addressTable.GetDim(0)) ||
        !IsVectorOfLength(control, kFusedGetControlWords) ||
        !IsVectorOfLength(streamingEpoch, 1) ||
        !IsVectorOfLength(executionMode, 1) || pipelineState.GetDimNum() != 2 ||
        pipelineState.GetDim(0) !=
            static_cast<int64_t>(kProducerCount * kGetLaneCount) ||
        pipelineState.GetDim(1) !=
            static_cast<int64_t>(kPipelineWordsPerRing)) {
        return fail("geometry_basic");
    }

    P5Geometry value{};
    if (!ToU32(residentThreshold.GetDim(0), &value.rowCount) ||
        !ToU32(kvObjectMap.GetDim(1), &value.sourceCapacity) ||
        !ToU32(kvObjectMap.GetDim(0), &value.poolSize) ||
        !ToU32(kvObjectSlotsSnapshot.GetDim(1), &value.kvObjectCapacity) ||
        !ToU32(privateReady.GetDim(1), &value.privateArenaCapacity) ||
        !ToU32(scoreWorkspace.GetDim(0), &value.scoreWorkspaceWords) ||
        !ToU32(addressTable.GetDim(0), &value.tableCapacity) ||
        !ToU32(privatePackedOutput.GetDim(0), &value.outputRows)) {
        return fail("geometry_u32");
    }
    if (!IsMatrix(kvObjectSlotsSnapshot, value.poolSize, value.kvObjectCapacity) ||
        !IsMatrix(privateObjectToSlot, value.poolSize, value.kvObjectCapacity) ||
        !IsMatrix(privateSourceToSlot, value.rowCount, value.sourceCapacity) ||
        !IsMatrix(privateReady, kGetLaneCount, value.privateArenaCapacity) ||
        static_cast<uint64_t>(value.outputRows) !=
            static_cast<uint64_t>(value.privateArenaCapacity) * kGetLaneCount ||
        value.kvObjectCapacity == 0U || value.sourceCapacity == 0U ||
        value.poolSize == 0U || value.privateArenaCapacity == 0U) {
        return fail("geometry_matrix");
    }
    const uint64_t tilesPerRow =
        (static_cast<uint64_t>(value.sourceCapacity) + kReceiptTileSources - 1U) /
        kReceiptTileSources;
    const uint64_t receipts = static_cast<uint64_t>(value.rowCount) * tilesPerRow;
    // The graph carrier is int32, while C8 publishes two sortable uint16
    // score values per carrier word.  Receipt byte offsets are runtime
    // metadata, but the backing workspace must at least hold the full dense
    // `[row, source]` score surface before any receipt can be trusted.
    const uint64_t scoreValues =
        static_cast<uint64_t>(value.rowCount) * value.sourceCapacity;
    const uint64_t minimumScoreWords = (scoreValues + 1U) / 2U;
    const uint64_t candidatesPerProducer =
        ((receipts + kProducerCount - 1U) / kProducerCount) *
        kReceiptTileSources;
    if (tilesPerRow == 0U || receipts == 0U ||
        receipts > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
        minimumScoreWords > value.scoreWorkspaceWords ||
        candidatesPerProducer == 0U ||
        candidatesPerProducer >
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
        !IsMatrix(tileReceipts, static_cast<int64_t>(receipts), kReceiptWords)) {
        return fail("geometry_score_or_receipts");
    }
    value.tileReceiptCount = static_cast<uint32_t>(receipts);
    value.candidateCapacityPerProducer =
        static_cast<uint32_t>(candidatesPerProducer);
    *geometry = value;
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingA5ExactScanSpeculativeGet(
    gert::TilingContext* context)
{
    if (context == nullptr) {
        LogP5TilingBootstrapFailure("null_context", nullptr, 0U);
        return ge::GRAPH_FAILED;
    }
    if (context->GetPlatformInfo() == nullptr) {
        LogP5TilingBootstrapFailure("null_platform", context, 0U);
        return ge::GRAPH_FAILED;
    }
    for (size_t input = 0; input < kInputCount; ++input) {
        if (context->GetInputShape(input) == nullptr) {
            LogP5TilingBootstrapFailure("missing_shape", context, input);
            return ge::GRAPH_FAILED;
        }
        if (context->GetInputDesc(input) == nullptr) {
            LogP5TilingBootstrapFailure("missing_desc", context, input);
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape scoreWorkspace =
        context->GetInputShape(kScoreWorkspace)->GetStorageShape();
    const gert::Shape tileReceipts =
        context->GetInputShape(kTileReceipts)->GetStorageShape();
    const gert::Shape streamingEpoch =
        context->GetInputShape(kStreamingEpoch)->GetStorageShape();
    const gert::Shape scoreSourceLens =
        context->GetInputShape(kScoreSourceLens)->GetStorageShape();
    const gert::Shape residentThreshold =
        context->GetInputShape(kResidentThreshold)->GetStorageShape();
    const gert::Shape residentThresholdValid =
        context->GetInputShape(kResidentThresholdValid)->GetStorageShape();
    const gert::Shape kvObjectSlotsSnapshot =
        context->GetInputShape(kKvObjectSlotsSnapshot)->GetStorageShape();
    const gert::Shape reqPoolEntries =
        context->GetInputShape(kReqPoolEntries)->GetStorageShape();
    const gert::Shape kvObjectMap =
        context->GetInputShape(kKvObjectMap)->GetStorageShape();
    const gert::Shape kvObjectLens =
        context->GetInputShape(kKvObjectLens)->GetStorageShape();
    const gert::Shape privateSourceToSlot =
        context->GetInputShape(kPrivateSourceToSlot)->GetStorageShape();
    const gert::Shape privateObjectToSlot =
        context->GetInputShape(kPrivateObjectToSlot)->GetStorageShape();
    const gert::Shape privateReady =
        context->GetInputShape(kPrivateReady)->GetStorageShape();
    const gert::Shape privateArenaHead =
        context->GetInputShape(kPrivateArenaHead)->GetStorageShape();
    const gert::Shape privatePackedOutput =
        context->GetInputShape(kPrivatePackedOutput)->GetStorageShape();
    const gert::Shape addressTable =
        context->GetInputShape(kAddressTable)->GetStorageShape();
    const gert::Shape validTable =
        context->GetInputShape(kValidTable)->GetStorageShape();
    const gert::Shape control =
        context->GetInputShape(kFusedGetControl)->GetStorageShape();
    const gert::Shape pipelineState =
        context->GetInputShape(kPipelineState)->GetStorageShape();
    const gert::Shape executionMode =
        context->GetInputShape(kExecutionMode)->GetStorageShape();

    if (context->GetInputDesc(kScoreWorkspace)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(kTileReceipts)->GetDataType() != ge::DT_INT64 ||
        context->GetInputDesc(kStreamingEpoch)->GetDataType() != ge::DT_INT64 ||
        context->GetInputDesc(kScoreSourceLens)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(kResidentThreshold)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(kResidentThresholdValid)->GetDataType() !=
            ge::DT_INT32 ||
        context->GetInputDesc(kKvObjectSlotsSnapshot)->GetDataType() !=
            ge::DT_INT32 ||
        context->GetInputDesc(kReqPoolEntries)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(kKvObjectMap)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(kKvObjectLens)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(kPrivateSourceToSlot)->GetDataType() !=
            ge::DT_INT32 ||
        context->GetInputDesc(kPrivateObjectToSlot)->GetDataType() !=
            ge::DT_INT32 ||
        context->GetInputDesc(kPrivateReady)->GetDataType() != ge::DT_INT32 ||
        context->GetInputDesc(kPrivateArenaHead)->GetDataType() !=
            ge::DT_INT32 ||
        context->GetInputDesc(kPrivatePackedOutput)->GetDataType() !=
            ge::DT_UINT8 ||
        context->GetInputDesc(kAddressTable)->GetDataType() != ge::DT_INT64 ||
        context->GetInputDesc(kValidTable)->GetDataType() != ge::DT_UINT8 ||
        context->GetInputDesc(kFusedGetControl)->GetDataType() != ge::DT_INT64 ||
        context->GetInputDesc(kPipelineState)->GetDataType() != ge::DT_INT64 ||
        context->GetInputDesc(kExecutionMode)->GetDataType() != ge::DT_INT64) {
        LogP5TilingContractFailure("dtype", context, nullptr, 0U, 0U, 0U,
                                   ge::GRAPH_SUCCESS);
        return ge::GRAPH_FAILED;
    }

    P5Geometry geometry{};
    const char* geometryFailure = "geometry_unknown";
    if (!DeriveP5Geometry(
            scoreWorkspace, tileReceipts, streamingEpoch, scoreSourceLens,
            residentThreshold,
            residentThresholdValid, kvObjectSlotsSnapshot, reqPoolEntries,
            kvObjectMap, kvObjectLens, privateSourceToSlot,
            privateObjectToSlot, privateReady, privateArenaHead,
            privatePackedOutput, addressTable, validTable, control,
            pipelineState, executionMode, &geometry, &geometryFailure)) {
        LogP5TilingContractFailure(geometryFailure, context, nullptr, 0U, 0U,
                                   0U, ge::GRAPH_SUCCESS);
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aicCount = platform.GetCoreNumAic();
    const uint32_t aivCount = platform.GetCoreNumAiv();
    // P5's protocol has a fixed 64-lane logical ABI: two lanes own KVAF Get
    // queues and 62 produce candidate seeds.  A larger part can execute that
    // exact ABI, whereas a smaller part cannot.  Requiring equality here made
    // the same valid 64-lane launch fail on A5 variants exposing spare AIVs.
    if (aivCount < kAivCount) {
        LogP5TilingContractFailure("aiv_count", context, &geometry, aicCount,
                                   aivCount, 0U, ge::GRAPH_SUCCESS);
        return ge::GRAPH_FAILED;
    }

    const uint32_t aivOnlyBlockDim =
        platform.CalcTschBlockDim(kAivCount, 0U, aivCount);
    if (aivOnlyBlockDim != kAivCount) {
        LogP5TilingContractFailure("block_dim", context, &geometry, aicCount,
                                   aivCount, aivOnlyBlockDim,
                                   ge::GRAPH_SUCCESS);
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<A5ExactScanSpeculativeGetTilingData>();
    if (tiling == nullptr) {
        LogP5TilingContractFailure("null_tiling", context, &geometry, aicCount,
                                   aivCount, aivOnlyBlockDim,
                                   ge::GRAPH_SUCCESS);
        return ge::GRAPH_FAILED;
    }
    tiling->rowCount = geometry.rowCount;
    tiling->sourceCapacity = geometry.sourceCapacity;
    tiling->poolSize = geometry.poolSize;
    tiling->kvObjectCapacity = geometry.kvObjectCapacity;
    tiling->privateArenaCapacity = geometry.privateArenaCapacity;
    tiling->tileReceiptCount = geometry.tileReceiptCount;
    tiling->scoreWorkspaceWords = geometry.scoreWorkspaceWords;
    tiling->rowsPerEntry = kRowsPerEntry;
    tiling->candidateCapacityPerProducer =
        geometry.candidateCapacityPerProducer;
    tiling->candidateSeedWords = kCandidateSeedWords;
    tiling->tableCapacity = geometry.tableCapacity;
    tiling->outputRows = geometry.outputRows;
    tiling->controlWords = kFusedGetControlWords;
    tiling->producerCount = kProducerCount;
    tiling->getLaneCount = kGetLaneCount;
    tiling->ringSlots = kRingSlots;
    tiling->recordsPerRingSlot = kRecordsPerRingSlot;
    tiling->getBootstrapRecords = kGetBootstrapRecords;
    tiling->getWaveRecords = kGetWaveRecords;
    tiling->pipelineWordsPerRing = kPipelineWordsPerRing;
    tiling->reserved = 0;

    // The block dimension must be fixed before requesting an optional batch
    // schedule.  A GE/model launch accepts batch mode and provides the
    // cooperative start required by the target graph path.  CANN may reject
    // that request for an ACLNN direct single-op launch.  Treat that as a
    // launch-mode fallback, not as an ABI or arithmetic failure: the same
    // 64 AIV-only blocks still execute, and the test reports it explicitly.
    context->SetBlockDim(aivOnlyBlockDim);
    const ge::graphStatus scheduleStatus = context->SetScheduleMode(1U);
    const char* scheduleEffective =
        scheduleStatus == ge::GRAPH_SUCCESS ? "batch" : "default_aiv_only";
    std::fprintf(
        stderr,
        "P5_TILING_READY aic=%u aiv=%u requested_aiv=%u block_dim=%u "
        "schedule_request=batch schedule_status=%d schedule_effective=%s "
        "rows=%u source=%u receipts=%u candidate_capacity=%u\\n",
        aicCount, aivCount, kAivCount, aivOnlyBlockDim,
        static_cast<int>(scheduleStatus), scheduleEffective, geometry.rowCount,
        geometry.sourceCapacity, geometry.tileReceiptCount,
        geometry.candidateCapacityPerProducer);
    std::fflush(stderr);
    if (context->GetWorkspaceSizes(1) != nullptr) {
        context->GetWorkspaceSizes(1)[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferA5ExactScanSpeculativeGetShape(
    gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t input = 0; input < kInputCount; ++input) {
        if (context->GetInputShape(input) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    for (size_t output = 0; output < kOutputCount; ++output) {
        if (context->GetOutputShape(output) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    // InferShapeContext exposes gert::Shape directly on CANN 9.1.  The
    // GetStorageShape() wrapper exists on TilingContext's storage-shape view,
    // but not here; copying the Shape keeps this callback compatible with the
    // generated CANN 9.1 operator host ABI.
    const gert::Shape scoreWorkspace =
        *context->GetInputShape(kScoreWorkspace);
    const gert::Shape tileReceipts = *context->GetInputShape(kTileReceipts);
    const gert::Shape streamingEpoch =
        *context->GetInputShape(kStreamingEpoch);
    const gert::Shape scoreSourceLens =
        *context->GetInputShape(kScoreSourceLens);
    const gert::Shape residentThreshold =
        *context->GetInputShape(kResidentThreshold);
    const gert::Shape residentThresholdValid =
        *context->GetInputShape(kResidentThresholdValid);
    const gert::Shape kvObjectSlotsSnapshot =
        *context->GetInputShape(kKvObjectSlotsSnapshot);
    const gert::Shape reqPoolEntries =
        *context->GetInputShape(kReqPoolEntries);
    const gert::Shape kvObjectMap = *context->GetInputShape(kKvObjectMap);
    const gert::Shape kvObjectLens = *context->GetInputShape(kKvObjectLens);
    const gert::Shape privateSourceToSlot =
        *context->GetInputShape(kPrivateSourceToSlot);
    const gert::Shape privateObjectToSlot =
        *context->GetInputShape(kPrivateObjectToSlot);
    const gert::Shape privateReady = *context->GetInputShape(kPrivateReady);
    const gert::Shape privateArenaHead =
        *context->GetInputShape(kPrivateArenaHead);
    const gert::Shape privatePackedOutput =
        *context->GetInputShape(kPrivatePackedOutput);
    const gert::Shape addressTable = *context->GetInputShape(kAddressTable);
    const gert::Shape validTable = *context->GetInputShape(kValidTable);
    const gert::Shape control = *context->GetInputShape(kFusedGetControl);
    const gert::Shape pipelineState = *context->GetInputShape(kPipelineState);
    const gert::Shape executionMode =
        *context->GetInputShape(kExecutionMode);
    P5Geometry geometry{};
    if (!DeriveP5Geometry(
            scoreWorkspace, tileReceipts, streamingEpoch, scoreSourceLens,
            residentThreshold,
            residentThresholdValid, kvObjectSlotsSnapshot, reqPoolEntries,
            kvObjectMap, kvObjectLens, privateSourceToSlot,
            privateObjectToSlot, privateReady, privateArenaHead,
            privatePackedOutput, addressTable, validTable, control,
            pipelineState, executionMode, &geometry, nullptr)) {
        return ge::GRAPH_FAILED;
    }

    // These aliases describe transaction-local state only.  The persistent
    // resident mapping is intentionally not among this op's inputs or
    // outputs, so the consumer cannot publish an unfinished private read.
    *context->GetOutputShape(kPrivateSourceToSlotOut) = privateSourceToSlot;
    *context->GetOutputShape(kPrivateObjectToSlotOut) = privateObjectToSlot;
    *context->GetOutputShape(kPrivateReadyOut) = privateReady;
    *context->GetOutputShape(kPrivateArenaHeadOut) = privateArenaHead;
    *context->GetOutputShape(kPrivatePackedOutputOut) = privatePackedOutput;

    context->GetOutputShape(kCandidateSeeds)->SetDimNum(3);
    context->GetOutputShape(kCandidateSeeds)->SetDim(0, kProducerCount);
    context->GetOutputShape(kCandidateSeeds)->SetDim(
        1, geometry.candidateCapacityPerProducer);
    context->GetOutputShape(kCandidateSeeds)->SetDim(2, kCandidateSeedWords);
    context->GetOutputShape(kCandidateCounts)->SetDimNum(1);
    context->GetOutputShape(kCandidateCounts)->SetDim(0, kProducerCount);
    context->GetOutputShape(kProducerStats)->SetDimNum(2);
    context->GetOutputShape(kProducerStats)->SetDim(0, kProducerCount);
    context->GetOutputShape(kProducerStats)->SetDim(1, kProducerStatColumns);
    context->GetOutputShape(kGetStats)->SetDimNum(2);
    context->GetOutputShape(kGetStats)->SetDim(0, kGetLaneCount);
    context->GetOutputShape(kGetStats)->SetDim(1, kGetStatColumns);
    context->GetOutputShape(kRoleTrace)->SetDimNum(1);
    context->GetOutputShape(kRoleTrace)->SetDim(0, kAivCount);
    for (size_t output : {kGetIndices, kGetOffsets, kGetOutputIndices,
                          kGetMask, kGetCandidateSeedSlots, kGetClaimNew}) {
        context->GetOutputShape(output)->SetDimNum(3);
        context->GetOutputShape(output)->SetDim(0, kGetLaneCount);
        context->GetOutputShape(output)->SetDim(1, kGetDescriptorBufferCount);
        context->GetOutputShape(output)->SetDim(2, kGetWaveRecords);
    }
    *context->GetOutputShape(kPipelineStateOut) = pipelineState;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferA5ExactScanSpeculativeGetDataType(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(kPrivateSourceToSlotOut, ge::DT_INT32);
    context->SetOutputDataType(kPrivateObjectToSlotOut, ge::DT_INT32);
    context->SetOutputDataType(kPrivateReadyOut, ge::DT_INT32);
    context->SetOutputDataType(kPrivateArenaHeadOut, ge::DT_INT32);
    context->SetOutputDataType(kPrivatePackedOutputOut, ge::DT_UINT8);
    context->SetOutputDataType(kCandidateSeeds, ge::DT_INT32);
    context->SetOutputDataType(kCandidateCounts, ge::DT_INT32);
    context->SetOutputDataType(kProducerStats, ge::DT_INT64);
    context->SetOutputDataType(kGetStats, ge::DT_INT64);
    context->SetOutputDataType(kRoleTrace, ge::DT_INT32);
    context->SetOutputDataType(kGetIndices, ge::DT_INT32);
    context->SetOutputDataType(kGetOffsets, ge::DT_INT32);
    context->SetOutputDataType(kGetOutputIndices, ge::DT_INT32);
    context->SetOutputDataType(kGetMask, ge::DT_UINT8);
    context->SetOutputDataType(kGetCandidateSeedSlots, ge::DT_INT32);
    context->SetOutputDataType(kGetClaimNew, ge::DT_UINT8);
    context->SetOutputDataType(kPipelineStateOut, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

class A5ExactScanSpeculativeGet : public OpDef {
public:
    explicit A5ExactScanSpeculativeGet(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> int32Types = {ge::DT_INT32};
        const std::vector<ge::DataType> int64Types = {ge::DT_INT64};
        const std::vector<ge::DataType> uint8Types = {ge::DT_UINT8};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        this->Input("score_workspace").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Input("tile_receipts").ParamType(REQUIRED).DataType(int64Types).Format(formats);
        this->Input("streaming_epoch").ParamType(REQUIRED).DataType(int64Types).Format(formats);
        this->Input("score_source_lens").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Input("resident_threshold").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Input("resident_threshold_valid").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Input("kv_object_slots_snapshot").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Input("req_pool_entries").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Input("kv_object_map").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Input("kv_object_lens").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Input("private_source_to_slot").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Input("private_object_to_slot").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Input("private_ready_table").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Input("private_arena_head").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Input("private_packed_output").ParamType(REQUIRED).DataType(uint8Types).Format(formats);
        this->Input("address_table").ParamType(REQUIRED).DataType(int64Types).Format(formats);
        this->Input("valid_table").ParamType(REQUIRED).DataType(uint8Types).Format(formats);
        this->Input("fused_get_control").ParamType(REQUIRED).DataType(int64Types).Format(formats);
        this->Input("pipeline_state").ParamType(REQUIRED).DataType(int64Types).Format(formats);
        this->Input("execution_mode").ParamType(REQUIRED).DataType(int64Types).Format(formats);
        this->Output("private_source_to_slot_out").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("private_object_to_slot_out").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("private_ready_table_out").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("private_arena_head_out").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("private_packed_output_out").ParamType(REQUIRED).DataType(uint8Types).Format(formats);
        this->Output("candidate_seeds").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("candidate_counts").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("producer_stats").ParamType(REQUIRED).DataType(int64Types).Format(formats);
        this->Output("get_stats").ParamType(REQUIRED).DataType(int64Types).Format(formats);
        this->Output("role_trace").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("get_indices").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("get_offsets").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("get_output_indices").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("get_mask").ParamType(REQUIRED).DataType(uint8Types).Format(formats);
        this->Output("get_candidate_seed_slots").ParamType(REQUIRED).DataType(int32Types).Format(formats);
        this->Output("get_claim_new").ParamType(REQUIRED).DataType(uint8Types).Format(formats);
        this->Output("pipeline_state_out").ParamType(REQUIRED).DataType(int64Types).Format(formats);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore()
            .SetTiling(optiling::TilingA5ExactScanSpeculativeGet)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5ExactScanSpeculativeGet);

IMPL_OP_INFERSHAPE(A5ExactScanSpeculativeGet)
    .InferShape(InferA5ExactScanSpeculativeGetShape)
    .InferDataType(InferA5ExactScanSpeculativeGetDataType);
}  // namespace ops
