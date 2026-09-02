// P8 host ABI: one forward-owned state machine, four graph phases.
//
// The P8 phase is graph-visible in a fixed KVAF forward-control record, not
// an attribute or process-local switch.  Its four int64 words encode
// [forward_epoch, graph_replay_generation, packed(layer_control_index, phase),
// flags].  Device code always loads it through KVAF's field-wise ABI helper.
// This lets Post, Drain, and QSFA-terminal launches validate the same durable
// resource records across graph replay without changing C8's ABI.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "../op_kernel/a5_p8_forward_owned_executor_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
enum InputIndex : size_t {
    kCandidateSeeds = 0,
    kCandidateCounts,
    kCandidateReceiptIdentity,
    kCandidateStatus,
    kForwardControl,
    // Graph-stable, arena-wide sticky health bit.  This is intentionally
    // independent of the per-layer four-word forward-control record: an
    // in-flight resource quarantine must reject P8 admission for later
    // layer/forward invocations using the same fixed arena.
    kP8ArenaHealth,
    kSourceLens,
    kKvObjectAddressDomain,
    kReqPoolEntries,
    kKvObjectMap,
    kKvObjectLens,
    kKvObjectSlotsSnapshot,
    kPrivateSourceToSlot,
    kPrivateObjectToSlot,
    kPrivateReadyTable,
    kPrivateSlotCursor,
    kPrivatePackedOutput,
    kFusedGetControl,
    kP8WaveRecords,
    kP8PrivateRecords,
    kQueueRequestSourceOffsets,
    kQueueRequestObjectOffsets,
    kQueueRequestOutputIndices,
    kQueueRequestCount,
    kCandidateCursor,
    kCandidateCursorIdentity,
    kRowsPerEntry,
    kWorkerStats,
    kInputCount,
};

enum OutputIndex : size_t {
    // Forward-control is an output alias as well as an input.  P8 can set
    // the shared fail-closed abort gate on a device failure, and graph users
    // must observe that mutation rather than treating it as an invisible
    // side effect.
    kForwardControlOut = 0,
    kP8ArenaHealthOut,
    kPrivateSourceToSlotOut,
    kPrivateObjectToSlotOut,
    kPrivateReadyTableOut,
    kPrivateSlotCursorOut,
    kPrivatePackedOutputOut,
    kFusedGetControlOut,
    kP8WaveRecordsOut,
    kP8PrivateRecordsOut,
    kQueueRequestSourceOffsetsOut,
    kQueueRequestObjectOffsetsOut,
    kQueueRequestOutputIndicesOut,
    kQueueRequestCountOut,
    kCandidateCursorOut,
    kCandidateCursorIdentityOut,
    kWorkerStatsOut,
    kCandidateReceiptIdentityOut,
    kCandidateStatusOut,
    kOutputCount,
};

constexpr size_t kAliasInputs[kOutputCount] = {
    kForwardControl, kP8ArenaHealth, kPrivateSourceToSlot, kPrivateObjectToSlot,
    kPrivateReadyTable,
    kPrivateSlotCursor, kPrivatePackedOutput, kFusedGetControl,
    kP8WaveRecords, kP8PrivateRecords, kQueueRequestSourceOffsets,
    kQueueRequestObjectOffsets, kQueueRequestOutputIndices,
    kQueueRequestCount, kCandidateCursor, kCandidateCursorIdentity,
    kWorkerStats, kCandidateReceiptIdentity, kCandidateStatus,
};

// This literal is deliberately emitted by every Stage-B tiler diagnostic and
// is also checked in the freshly packaged liboptiling.so by the P8 runner.
// It closes the gap between "the source was staged" and "this exact host
// tiler was installed and selected" without adding a persistent artifact
// database or changing the P8 ABI.
constexpr char kP8TilingDiagnosticRevision[] =
    "P8_TILING_DIAG_REVISION=p8_tiling_diag_v1";

constexpr const char* kInputNames[kInputCount] = {
    "candidate_seeds", "candidate_counts", "candidate_receipt_identity",
    "candidate_status", "forward_control", "p8_arena_health",
    "source_lens", "kv_object_address_domain", "req_pool_entries",
    "kv_object_map", "kv_object_lens", "kv_object_slots_snapshot",
    "private_source_to_slot", "private_object_to_slot",
    "private_ready_table", "private_slot_cursor", "private_packed_output",
    "fused_get_control", "p8_wave_records", "p8_private_records",
    "queue_request_source_offsets", "queue_request_object_offsets",
    "queue_request_output_indices", "queue_request_count",
    "candidate_cursor", "candidate_cursor_identity", "rows_per_entry",
    "worker_stats",
};

void LogP8TilingShape(const gert::Shape* shape)
{
    if (shape == nullptr) {
        std::fprintf(stderr, " storage_rank=<null> storage_dims=<null>");
        return;
    }
    const int64_t rank = shape->GetDimNum();
    std::fprintf(stderr, " storage_rank=%lld storage_dims=[",
                 static_cast<long long>(rank));
    constexpr int64_t kMaxLoggedDims = 8;
    const int64_t shown = rank < kMaxLoggedDims ? rank : kMaxLoggedDims;
    for (int64_t dim = 0; dim < shown; ++dim) {
        std::fprintf(stderr, "%s%lld", dim == 0 ? "" : ",",
                     static_cast<long long>(shape->GetDim(dim)));
    }
    if (rank > shown) {
        std::fprintf(stderr, ",...");
    }
    std::fprintf(stderr, "]");
}

void LogP8TilingReject(const char* reason, size_t input,
                       const gert::Shape* shape, int expected_dtype,
                       int actual_dtype)
{
    std::fprintf(stderr,
                 "P8_TILING_REJECT marker=%s reason=%s input=%zu name=%s "
                 "expected_dtype=%d actual_dtype=%d",
                 kP8TilingDiagnosticRevision, reason, input,
                 input < kInputCount ? kInputNames[input] : "<none>",
                 expected_dtype, actual_dtype);
    LogP8TilingShape(shape);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

void LogP8TilingGeometry(const gert::Shape* const* shapes)
{
    for (size_t input = 0U; input < kInputCount; ++input) {
        std::fprintf(stderr,
                     "P8_TILING_GEOMETRY marker=%s input=%zu name=%s",
                     kP8TilingDiagnosticRevision, input, kInputNames[input]);
        LogP8TilingShape(shapes == nullptr ? nullptr : shapes[input]);
        std::fprintf(stderr, "\n");
    }
    std::fflush(stderr);
}

// P8's workers are dynamic *queue service tasks*, never a permanent Get-core
// partition.  Keep the host-side launch geometry visible in one compact line:
// this is the only place that can turn a bounded receipt/queue workload into
// a many-AIV launch before the device kernel gets control.
void LogP8Tiling(const char* stage, uint32_t aic, uint32_t aiv,
                 uint32_t receipts, uint32_t queues, uint32_t private_rows,
                 uint32_t stats_rows, uint32_t workers, uint32_t block_dim,
                 ge::graphStatus schedule_status)
{
    std::fprintf(
        stderr,
        "P8_TILING_%s marker=%s aic=%u aiv=%u receipts=%u queues=%u private_rows=%u "
        "stats_rows=%u workers=%u block_dim=%u schedule_status=%d\n",
        stage, kP8TilingDiagnosticRevision, aic, aiv, receipts, queues,
        private_rows, stats_rows,
        workers, block_dim, static_cast<int>(schedule_status));
    std::fflush(stderr);
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

bool ValidateGeometry(const gert::Shape* const* shapes, uint32_t* receipts,
                      uint32_t* rows, uint32_t* sources, uint32_t* pools,
                      uint32_t* objects, uint32_t* queues,
                      uint32_t* private_rows, uint32_t* worker_stat_rows)
{
    if (shapes == nullptr || receipts == nullptr || rows == nullptr ||
        sources == nullptr || pools == nullptr || objects == nullptr ||
        queues == nullptr || private_rows == nullptr ||
        worker_stat_rows == nullptr) {
        return false;
    }
    for (size_t index = 0U; index < kInputCount; ++index) {
        if (shapes[index] == nullptr) {
            return false;
        }
    }
    const gert::Shape& seeds = *shapes[kCandidateSeeds];
    const gert::Shape& counts = *shapes[kCandidateCounts];
    const gert::Shape& receipt_identity =
        *shapes[kCandidateReceiptIdentity];
    const gert::Shape& status = *shapes[kCandidateStatus];
    const gert::Shape& forward = *shapes[kForwardControl];
    const gert::Shape& arena_health = *shapes[kP8ArenaHealth];
    const gert::Shape& source_lens = *shapes[kSourceLens];
    const gert::Shape& domain = *shapes[kKvObjectAddressDomain];
    const gert::Shape& pool_entries = *shapes[kReqPoolEntries];
    const gert::Shape& object_map = *shapes[kKvObjectMap];
    const gert::Shape& object_lens = *shapes[kKvObjectLens];
    const gert::Shape& snapshot = *shapes[kKvObjectSlotsSnapshot];
    const gert::Shape& source_to_slot = *shapes[kPrivateSourceToSlot];
    const gert::Shape& object_to_slot = *shapes[kPrivateObjectToSlot];
    const gert::Shape& private_ready = *shapes[kPrivateReadyTable];
    const gert::Shape& private_cursor = *shapes[kPrivateSlotCursor];
    const gert::Shape& private_output = *shapes[kPrivatePackedOutput];
    const gert::Shape& control = *shapes[kFusedGetControl];
    const gert::Shape& waves = *shapes[kP8WaveRecords];
    const gert::Shape& private_records = *shapes[kP8PrivateRecords];
    const gert::Shape& request_source = *shapes[kQueueRequestSourceOffsets];
    const gert::Shape& request_object = *shapes[kQueueRequestObjectOffsets];
    const gert::Shape& request_output = *shapes[kQueueRequestOutputIndices];
    const gert::Shape& request_count = *shapes[kQueueRequestCount];
    const gert::Shape& candidate_cursor = *shapes[kCandidateCursor];
    const gert::Shape& candidate_cursor_identity =
        *shapes[kCandidateCursorIdentity];
    const gert::Shape& rows_per_entry = *shapes[kRowsPerEntry];
    const gert::Shape& stats = *shapes[kWorkerStats];

    if (seeds.GetDimNum() != 3 || seeds.GetDim(0) <= 0 ||
        seeds.GetDim(1) != kA5P8CandidateCapacity ||
        seeds.GetDim(2) != kA5P8CandidateSeedWords ||
        !IsPositiveVector(counts) || !IsPositiveMatrix(receipt_identity) ||
        !IsPositiveVector(status) || !IsPositiveVector(forward) ||
        forward.GetDim(0) != kA5P8ForwardControlWords ||
        !IsPositiveVector(arena_health) || arena_health.GetDim(0) != 1 ||
        !IsPositiveVector(source_lens) || !IsPositiveVector(domain) ||
        domain.GetDim(0) != 1 || !IsPositiveVector(pool_entries) ||
        !IsPositiveMatrix(object_map) || !IsPositiveVector(object_lens) ||
        !IsPositiveMatrix(snapshot) || !IsPositiveMatrix(source_to_slot) ||
        !IsPositiveMatrix(object_to_slot) || !IsPositiveVector(private_ready) ||
        !IsPositiveVector(private_cursor) || private_cursor.GetDim(0) != 1 ||
        private_output.GetDimNum() != 2 ||
        private_output.GetDim(1) != 656 || !IsPositiveVector(control) ||
        !IsPositiveMatrix(waves) ||
        waves.GetDim(1) != kA5P8WaveRecordWords ||
        !IsPositiveMatrix(private_records) ||
        private_records.GetDim(1) != kA5P8PrivateRecordWords ||
        !IsPositiveMatrix(request_source) || !IsPositiveMatrix(request_object) ||
        !IsPositiveMatrix(request_output) || !IsPositiveVector(request_count) ||
        !IsPositiveVector(candidate_cursor) ||
        !IsPositiveMatrix(candidate_cursor_identity) ||
        !IsPositiveVector(rows_per_entry) || rows_per_entry.GetDim(0) != 1 ||
        !IsPositiveMatrix(stats) ||
        stats.GetDim(1) != kA5P8WorkerStatColumns) {
        return false;
    }
    const int64_t receipt_count = seeds.GetDim(0);
    const int64_t row_count = source_lens.GetDim(0);
    const int64_t source_capacity = object_map.GetDim(1);
    const int64_t pool_count = object_map.GetDim(0);
    const int64_t object_capacity = snapshot.GetDim(1);
    const int64_t queue_count = waves.GetDim(0);
    const int64_t private_count = private_records.GetDim(0);
    const int64_t stat_rows = stats.GetDim(0);
    if (receipt_count > std::numeric_limits<uint32_t>::max() ||
        row_count <= 0 || source_capacity <= 0 || pool_count <= 0 ||
        object_capacity <= 0 || queue_count <= 0 || private_count <= 0 ||
        stat_rows <= 0 || row_count > std::numeric_limits<uint32_t>::max() ||
        source_capacity > std::numeric_limits<uint32_t>::max() ||
        pool_count > std::numeric_limits<uint32_t>::max() ||
        object_capacity > std::numeric_limits<uint32_t>::max() ||
        queue_count > std::numeric_limits<uint32_t>::max() ||
        private_count > std::numeric_limits<uint32_t>::max() ||
        stat_rows > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (counts.GetDim(0) != receipt_count ||
        receipt_identity.GetDim(0) != receipt_count ||
        receipt_identity.GetDim(1) != kA5P8CandidateReceiptIdentityWords ||
        status.GetDim(0) != receipt_count || candidate_cursor.GetDim(0) != receipt_count ||
        candidate_cursor_identity.GetDim(0) != receipt_count ||
        candidate_cursor_identity.GetDim(1) !=
            kA5P8CandidateReceiptIdentityWords ||
        pool_entries.GetDim(0) != row_count || object_lens.GetDim(0) != row_count ||
        snapshot.GetDim(0) != pool_count || source_to_slot.GetDim(0) != row_count ||
        source_to_slot.GetDim(1) != source_capacity ||
        object_to_slot.GetDim(0) != pool_count ||
        object_to_slot.GetDim(1) != object_capacity ||
        private_ready.GetDim(0) != private_count ||
        private_output.GetDim(0) != private_count ||
        request_source.GetDim(0) != queue_count ||
        request_object.GetDim(0) != queue_count ||
        request_output.GetDim(0) != queue_count ||
        request_source.GetDim(1) != kA5P8RequestCapacity ||
        request_object.GetDim(1) != kA5P8RequestCapacity ||
        request_output.GetDim(1) != kA5P8RequestCapacity ||
        request_count.GetDim(0) != queue_count) {
        return false;
    }
    *receipts = static_cast<uint32_t>(receipt_count);
    *rows = static_cast<uint32_t>(row_count);
    *sources = static_cast<uint32_t>(source_capacity);
    *pools = static_cast<uint32_t>(pool_count);
    *objects = static_cast<uint32_t>(object_capacity);
    *queues = static_cast<uint32_t>(queue_count);
    *private_rows = static_cast<uint32_t>(private_count);
    *worker_stat_rows = static_cast<uint32_t>(stat_rows);
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingA5P8ForwardOwnedExecutor(
    gert::TilingContext* context)
{
    if (context == nullptr) {
        LogP8TilingReject("null_context", kInputCount, nullptr, -1, -1);
        return ge::GRAPH_FAILED;
    }
    if (context->GetPlatformInfo() == nullptr) {
        LogP8TilingReject("null_platform", kInputCount, nullptr, -1, -1);
        return ge::GRAPH_FAILED;
    }
    std::fprintf(stderr, "P8_TILING_ENTER marker=%s inputs=%zu platform=1\n",
                 kP8TilingDiagnosticRevision, kInputCount);
    std::fflush(stderr);
    const ge::DataType expected[kInputCount] = {
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
        ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        ge::DT_UINT8, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32,
        ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64};
    const gert::Shape* shapes[kInputCount] = {};
    for (size_t index = 0U; index < kInputCount; ++index) {
        const auto* input_shape = context->GetInputShape(index);
        if (input_shape == nullptr) {
            LogP8TilingReject("null_input_shape", index, nullptr,
                              static_cast<int>(expected[index]), -1);
            return ge::GRAPH_FAILED;
        }
        const auto* input_desc = context->GetInputDesc(index);
        if (input_desc == nullptr) {
            LogP8TilingReject("null_input_desc", index,
                              &input_shape->GetStorageShape(),
                              static_cast<int>(expected[index]), -1);
            return ge::GRAPH_FAILED;
        }
        const ge::DataType actual = input_desc->GetDataType();
        if (actual != expected[index]) {
            LogP8TilingReject("input_dtype", index,
                              &input_shape->GetStorageShape(),
                              static_cast<int>(expected[index]),
                              static_cast<int>(actual));
            return ge::GRAPH_FAILED;
        }
        shapes[index] = &input_shape->GetStorageShape();
    }
    uint32_t receipts = 0U;
    uint32_t rows = 0U;
    uint32_t sources = 0U;
    uint32_t pools = 0U;
    uint32_t objects = 0U;
    uint32_t queues = 0U;
    uint32_t private_rows = 0U;
    uint32_t stats_rows = 0U;
    if (!ValidateGeometry(shapes, &receipts, &rows, &sources, &pools,
                          &objects, &queues, &private_rows, &stats_rows)) {
        LogP8TilingReject("geometry_contract", kInputCount, nullptr, -1,
                          -1);
        LogP8TilingGeometry(shapes);
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t available_aic = platform.GetCoreNumAic();
    const uint32_t available_aiv = platform.GetCoreNumAiv();
    // A worker walks receipts, queues and private rows with a stride.  Thus
    // private_rows is storage capacity, not a concurrency requirement.  A
    // larger worker count than receipt/queue service capacity only creates
    // empty AIV blocks that contend for the same graph launch resources.
    // Cap by both real sources of concurrent transport work; any scheduled
    // AIV can still lease any queue, so this is not a core-to-queue binding.
    uint32_t workers = available_aiv < receipts ? available_aiv : receipts;
    workers = workers < queues ? workers : queues;
    if (workers == 0U || stats_rows < workers) {
        LogP8Tiling("FAIL", available_aic, available_aiv, receipts, queues,
                    private_rows, stats_rows, workers, 0U,
                    ge::GRAPH_FAILED);
        return ge::GRAPH_FAILED;
    }
    const uint32_t block_dim = platform.CalcTschBlockDim(
        workers, 0U, available_aiv);
    if (block_dim != workers) {
        LogP8Tiling("FAIL", available_aic, available_aiv, receipts, queues,
                    private_rows, stats_rows, workers, block_dim,
                    ge::GRAPH_FAILED);
        return ge::GRAPH_FAILED;
    }
    const ge::graphStatus schedule_status = context->SetScheduleMode(1U);
    if (schedule_status != ge::GRAPH_SUCCESS) {
        LogP8Tiling("FAIL", available_aic, available_aiv, receipts, queues,
                    private_rows, stats_rows, workers, block_dim,
                    schedule_status);
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(block_dim);
    LogP8Tiling("PREALLOC", available_aic, available_aiv, receipts, queues,
                private_rows, stats_rows, workers, block_dim,
                schedule_status);
    auto* tiling = context->GetTilingData<A5P8ForwardOwnedExecutorTilingData>();
    if (tiling == nullptr) {
        LogP8Tiling("FAIL", available_aic, available_aiv, receipts, queues,
                    private_rows, stats_rows, workers, block_dim,
                    schedule_status);
        return ge::GRAPH_FAILED;
    }
    tiling->workerCount = workers;
    tiling->receiptCount = receipts;
    tiling->rowCount = rows;
    tiling->sourceCapacity = sources;
    tiling->poolSize = pools;
    tiling->kvObjectCapacity = objects;
    tiling->queueCount = queues;
    tiling->privateRowCount = private_rows;
    tiling->workerStatsRows = stats_rows;
    tiling->phase = 0U;
    tiling->candidateCapacity = kA5P8CandidateCapacity;
    tiling->requestCapacity = kA5P8RequestCapacity;
    tiling->reserved0 = 0U;
    tiling->reserved1 = 0U;
    if (context->GetWorkspaceSizes(1) != nullptr) {
        context->GetWorkspaceSizes(1)[0] = 0U;
    }
    LogP8Tiling("READY", available_aic, available_aiv, receipts, queues,
                private_rows, stats_rows, workers, block_dim,
                schedule_status);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferA5P8ForwardOwnedExecutorShape(
    gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t output = 0U; output < kOutputCount; ++output) {
        const size_t input = kAliasInputs[output];
        if (context->GetInputShape(input) == nullptr ||
            context->GetOutputShape(output) == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *context->GetOutputShape(output) = *context->GetInputShape(input);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferA5P8ForwardOwnedExecutorDataType(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType output_types[kOutputCount] = {
        ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
        ge::DT_INT32, ge::DT_INT32, ge::DT_UINT8, ge::DT_INT64,
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64,
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT32};
    for (size_t output = 0U; output < kOutputCount; ++output) {
        context->SetOutputDataType(output, output_types[output]);
    }
    return ge::GRAPH_SUCCESS;
}

class A5P8ForwardOwnedExecutor : public OpDef {
public:
    explicit A5P8ForwardOwnedExecutor(const char* name) : OpDef(name)
    {
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        const std::vector<ge::DataType> int32_types = {ge::DT_INT32};
        const std::vector<ge::DataType> int64_types = {ge::DT_INT64};
        const std::vector<ge::DataType> uint8_types = {ge::DT_UINT8};
        const ge::DataType input_types[kInputCount] = {
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
            ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_UINT8, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64};
        const char* input_names[kInputCount] = {
            "candidate_seeds", "candidate_counts", "candidate_receipt_identity",
            "candidate_status", "forward_control", "p8_arena_health",
            "source_lens",
            "kv_object_address_domain", "req_pool_entries", "kv_object_map",
            "kv_object_lens", "kv_object_slots_snapshot",
            "private_source_to_slot", "private_object_to_slot",
            "private_ready_table", "private_slot_cursor",
            "private_packed_output", "fused_get_control", "p8_wave_records",
            "p8_private_records", "queue_request_source_offsets",
            "queue_request_object_offsets", "queue_request_output_indices",
            "queue_request_count", "candidate_cursor",
            "candidate_cursor_identity", "rows_per_entry", "worker_stats"};
        for (size_t input = 0U; input < kInputCount; ++input) {
            const std::vector<ge::DataType>* types = &int32_types;
            if (input_types[input] == ge::DT_INT64) {
                types = &int64_types;
            } else if (input_types[input] == ge::DT_UINT8) {
                types = &uint8_types;
            }
            this->Input(input_names[input]).ParamType(REQUIRED)
                .DataType(*types).Format(formats);
        }
        const ge::DataType output_types[kOutputCount] = {
            ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT32, ge::DT_INT32, ge::DT_UINT8, ge::DT_INT64,
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64,
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT32};
        const char* output_names[kOutputCount] = {
            "forward_control", "p8_arena_health", "private_source_to_slot",
            "private_object_to_slot",
            "private_ready_table", "private_slot_cursor",
            "private_packed_output", "fused_get_control", "p8_wave_records",
            "p8_private_records", "queue_request_source_offsets",
            "queue_request_object_offsets", "queue_request_output_indices",
            "queue_request_count", "candidate_cursor",
            "candidate_cursor_identity", "worker_stats",
            "candidate_receipt_identity", "candidate_status"};
        for (size_t output = 0U; output < kOutputCount; ++output) {
            const std::vector<ge::DataType>* types = &int32_types;
            if (output_types[output] == ge::DT_INT64) {
                types = &int64_types;
            } else if (output_types[output] == ge::DT_UINT8) {
                types = &uint8_types;
            }
            this->Output(output_names[output]).ParamType(REQUIRED)
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
            .ExtendCfgInfo("opFile.value", "a5_p8_forward_owned_executor")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore()
            .SetTiling(optiling::TilingA5P8ForwardOwnedExecutor)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5P8ForwardOwnedExecutor);

IMPL_OP_INFERSHAPE(A5P8ForwardOwnedExecutor)
    .InferShape(InferA5P8ForwardOwnedExecutorShape)
    .InferDataType(InferA5P8ForwardOwnedExecutorDataType);
}  // namespace ops
