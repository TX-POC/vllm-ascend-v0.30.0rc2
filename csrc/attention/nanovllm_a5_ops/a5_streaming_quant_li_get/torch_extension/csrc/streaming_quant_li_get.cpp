// Typed, isolated ACLNN bridge for the source-derived C8 LI + score-time Get.
//
// This bridge deliberately has no vLLM policy registration.  It can launch a
// target validation case using the same C8 input tensors as the official op,
// but a product route remains disabled until target byte-equivalence and the
// private-view/QSFA terminal chain are both proven.

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>

#include <torch/extension.h>
#include <torch/library.h>

#include "op_api_common.h"

namespace {
constexpr int64_t kSparseCount = 2048;
constexpr int64_t kControlWords = 16;
// Keep this in lockstep with `streaming_li_get::kTelemetryColumns`.  The
// additional column is a C8-internal overlap witness, not a host timestamp.
constexpr int64_t kTelemetryColumns = 14;
constexpr int64_t kPackedKvRowBytes = 656;
constexpr int64_t kMaxSourceCapacity = 1 << 18;
constexpr int64_t kAicCoreCount = 36;
constexpr int64_t kScoreTileWidth = 128;
constexpr int64_t kTileReceiptWords = 16;
// Test-only marker.  The normal/product path keeps the output allocation
// unmodified.  When the environment switch below is enabled, the marker lets
// a target bring-up distinguish "the AICore never wrote out8" from "AICore
// wrote an unexpected value" without changing the operator ABI or input
// dataflow.
constexpr int32_t kScoreSourceLensDebugSentinel = -123456789;

enum class PostClosureMode {
    kFull,
    kEmptyShell,
};

using Result = std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

bool DebugEnabled()
{
    const char* value = std::getenv("NANOVLLM_A5_OPAPI_DEBUG");
    return value != nullptr && value[0] != '\0' &&
           !(value[0] == '0' && value[1] == '\0');
}

bool ScoreSourceLensDebugPrefillEnabled()
{
    const char* value = std::getenv(
        "NANOVLLM_A5_SCORE_SOURCE_LENS_DEBUG_PREFILL");
    return value != nullptr && value[0] != '\0' &&
           !(value[0] == '0' && value[1] == '\0');
}

PostClosureMode ResolvePostClosureMode()
{
    // The bridge cannot inspect an ACLNN DSO's compile-time macro.  Require
    // the caller to state the expected post artifact explicitly; the Python
    // adapter independently validates the matching provenance manifest before
    // this bridge is imported.
    const char* value = std::getenv("P3_POST_CLOSURE_MODE");
    if (value == nullptr || value[0] == '\0' || std::string(value) == "full") {
        return PostClosureMode::kFull;
    }
    if (std::string(value) == "empty_shell") {
        return PostClosureMode::kEmptyShell;
    }
    TORCH_CHECK(false,
                "P3_POST_CLOSURE_MODE must be unset, 'full', or 'empty_shell'; got ",
                value);
    return PostClosureMode::kFull;
}

void CheckStrictDecodeScoreSourceLensAlias(
    const at::Tensor& query, const at::Tensor& actualSeqLengthsK,
    const std::string& layoutQuery, int64_t cmpRatio, int64_t rowCount)
{
    TORCH_CHECK(layoutQuery == "TND",
                "empty-shell score_source_lens alias requires decode layout_query=TND.");
    TORCH_CHECK(cmpRatio == 1,
                "empty-shell score_source_lens alias requires cmp_ratio=1.");
    TORCH_CHECK(query.dim() == 3 && query.size(0) == rowCount,
                "empty-shell score_source_lens alias requires one TND decode row "
                "per sparse output row.");
    TORCH_CHECK(actualSeqLengthsK.dim() == 1 &&
                    actualSeqLengthsK.numel() == rowCount,
                "empty-shell score_source_lens alias requires "
                "actual_seq_lengths_key=int32[R], where R=ProductExceptLast(sparse_indices).");
}

std::string ShapeString(c10::IntArrayRef values)
{
    std::ostringstream stream;
    stream << '[';
    for (int64_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            stream << ',';
        }
        stream << values[index];
    }
    stream << ']';
    return stream.str();
}

void LogTensor(const char* role, const at::Tensor& tensor)
{
    if (!DebugEnabled()) {
        return;
    }
    const std::string shape = ShapeString(tensor.sizes());
    const std::string stride = ShapeString(tensor.strides());
    std::fprintf(
        stderr,
        "P3_LI_ACLNN_TENSOR role=%s dtype=%s device=%s shape=%s stride=%s "
        "contiguous=%d nbytes=%zu data=%p\n",
        role, c10::toString(tensor.scalar_type()), tensor.device().str().c_str(),
        shape.c_str(), stride.c_str(), tensor.is_contiguous() ? 1 : 0,
        tensor.nbytes(), tensor.data_ptr());
}

int64_t ProductExceptLast(const at::Tensor& tensor)
{
    TORCH_CHECK(tensor.dim() >= 1 && tensor.size(-1) == kSparseCount,
                "sparse output must have final dimension 2048.");
    int64_t rows = 1;
    for (int64_t dim = 0; dim + 1 < tensor.dim(); ++dim) {
        const int64_t value = tensor.size(dim);
        TORCH_CHECK(value > 0 && rows <=
                        std::numeric_limits<int64_t>::max() / value,
                    "sparse output leading dimensions have an invalid product.");
        rows *= value;
    }
    return rows;
}

int64_t ScoreWorkspaceWords(int64_t sourceCapacity)
{
    TORCH_CHECK(sourceCapacity > 0 && sourceCapacity <= kMaxSourceCapacity,
                "source capacity is outside the fixed A5 score-workspace range.");
    const int64_t alignedSource =
        ((sourceCapacity + kScoreTileWidth - 1) / kScoreTileWidth) *
        kScoreTileWidth;
    // The C8-compatible public ABI reserves score storage for at most 36 AICs
    // and four score rows per AIC.  Runtime tiling may use fewer active cores;
    // each int32 slot backs two uint16 official sortable scores.
    return (kAicCoreCount * 4 * alignedSource + 1) / 2;
}

std::vector<int64_t> SparseOutputShape(
    const at::Tensor& query, const at::Tensor& key,
    const std::string& layoutQuery, const std::string& layoutKey)
{
    TORCH_CHECK(layoutKey == "PA_BSND",
                "A5StreamingQuantLiGet only supports layout_key=PA_BSND.");
    TORCH_CHECK(key.dim() == 4 && key.size(2) == 1,
                "A5StreamingQuantLiGet requires PA_BSND key [blocks, page, 1, D].");
    const int64_t keyHeads = key.size(2);
    if (layoutQuery == "TND") {
        TORCH_CHECK(query.dim() == 3,
                    "TND query must have shape [T, N1, D].");
        return {query.size(0), keyHeads, kSparseCount};
    }
    if (layoutQuery == "BSND") {
        TORCH_CHECK(query.dim() == 4,
                    "BSND query must have shape [B, S1, N1, D].");
        return {query.size(0), query.size(1), keyHeads, kSparseCount};
    }
    TORCH_CHECK(false, "layout_query must be TND or BSND, got ", layoutQuery);
}

void CheckSameNpu(
    const at::Tensor& anchor, const at::Tensor& tensor, const char* name,
    bool requireContiguous)
{
    TORCH_CHECK(tensor.defined() && tensor.device() == anchor.device(), name,
                " must be defined on the same NPU as query.");
    if (requireContiguous) {
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
    }
}

void CheckInputs(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights,
    const at::Tensor& queryScale, const at::Tensor& keyScale,
    const at::Tensor& actualSeqLengthsQ, const at::Tensor& actualSeqLengthsK,
    const at::Tensor& blockTable, const at::Tensor& metadata,
    const at::Tensor& residentThreshold,
    const at::Tensor& residentThresholdValid,
    const at::Tensor& kvObjectSlotsSnapshot,
    const at::Tensor& reqPoolEntries, const at::Tensor& kvObjectMap,
    const at::Tensor& privateSourceToSlot,
    const at::Tensor& privateReadyTable, const at::Tensor& privateArenaHead,
    const at::Tensor& privatePackedOutput, const at::Tensor& queueControl,
    const at::Tensor& queueLeases, const at::Tensor& kvObjectLens,
    const at::Tensor& scoreWorkspace, const at::Tensor& tileReceipts,
    const at::Tensor& streamingEpoch,
    const std::string& layoutQuery, const std::string& layoutKey,
    int64_t sparseCount, int64_t cmpRatio, int64_t kvafRowsPerEntry,
    int64_t kvObjectDomain)
{
    TORCH_CHECK(query.device().is_privateuseone(),
                "streaming_quant_li_get requires NPU inputs.");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous.");
    // PA_BSND has physical page strides. Never materialize a contiguous copy:
    // the source-derived LI must read precisely the same key storage as the
    // official reference. The ACL tensor wrapper carries its original stride.
    CheckSameNpu(query, key, "key", false);
    for (const auto& pair : {
             std::pair<const at::Tensor*, const char*>{&weights, "weights"},
             {&queryScale, "query_dequant_scale"},
             {&keyScale, "key_dequant_scale"},
             {&actualSeqLengthsQ, "actual_seq_lengths_query"},
             {&actualSeqLengthsK, "actual_seq_lengths_key"},
             {&blockTable, "block_table"}, {&metadata, "metadata"},
             {&residentThreshold, "resident_threshold"},
             {&residentThresholdValid, "resident_threshold_valid"},
             {&kvObjectSlotsSnapshot, "kv_object_slots_snapshot"},
             {&reqPoolEntries, "req_pool_entries"},
             {&kvObjectMap, "kv_object_map"},
             {&privateSourceToSlot, "private_source_to_slot"},
             {&privateReadyTable, "private_ready_table"},
             {&privateArenaHead, "private_arena_head"},
             {&privatePackedOutput, "private_packed_output"},
             {&queueControl, "queue_control"}, {&queueLeases, "queue_leases"},
             {&kvObjectLens, "kv_object_lens"},
             {&scoreWorkspace, "score_workspace"},
             {&tileReceipts, "tile_receipts"},
             {&streamingEpoch, "streaming_epoch"},
         }) {
        CheckSameNpu(query, *pair.first, pair.second, true);
    }
    TORCH_CHECK(query.scalar_type() == at::kFloat8_e4m3fn &&
                    key.scalar_type() == at::kFloat8_e4m3fn &&
                    weights.scalar_type() == at::kFloat &&
                    queryScale.scalar_type() == at::kFloat &&
                    keyScale.scalar_type() == at::kFloat,
                "streaming C8 LI requires FP8 query/key and FP32 weights/scales.");
    for (const auto& pair : {
             std::pair<const at::Tensor*, const char*>{&actualSeqLengthsQ, "actual_seq_lengths_query"},
             {&actualSeqLengthsK, "actual_seq_lengths_key"},
             {&blockTable, "block_table"}, {&metadata, "metadata"},
             {&residentThreshold, "resident_threshold"},
             {&residentThresholdValid, "resident_threshold_valid"},
             {&kvObjectSlotsSnapshot, "kv_object_slots_snapshot"},
             {&reqPoolEntries, "req_pool_entries"},
             {&kvObjectMap, "kv_object_map"},
             {&privateSourceToSlot, "private_source_to_slot"},
             {&privateReadyTable, "private_ready_table"},
             {&privateArenaHead, "private_arena_head"},
             {&queueLeases, "queue_leases"}, {&kvObjectLens, "kv_object_lens"},
         }) {
        TORCH_CHECK(pair.first->scalar_type() == at::kInt, pair.second,
                    " must be int32.");
    }
    TORCH_CHECK(privatePackedOutput.scalar_type() == at::kByte &&
                    queueControl.scalar_type() == at::kLong,
                "private_packed_output must be uint8 and queue_control int64.");
    TORCH_CHECK(scoreWorkspace.scalar_type() == at::kInt,
                "score_workspace must be int32.");
    TORCH_CHECK(tileReceipts.scalar_type() == at::kLong &&
                    streamingEpoch.scalar_type() == at::kLong,
                "tile_receipts and streaming_epoch must be int64.");
    TORCH_CHECK(sparseCount == kSparseCount,
                "the source-derived C8 LI is fixed to sparse_count=2048.");
    TORCH_CHECK(cmpRatio == 1 || cmpRatio == 4 || cmpRatio == 128,
                "A5 C8 supports cmp_ratio 1, 4, or 128.");
    TORCH_CHECK(kvafRowsPerEntry > 0, "kvaf_rows_per_entry must be positive.");
    TORCH_CHECK(kvObjectDomain == 0 || kvObjectDomain == 1,
                "kv_object_domain must be 0 (RAW_TOKEN) or 1 "
                "(C8_COMPRESSED_OBJECT).");
    TORCH_CHECK(kvObjectDomain != 0 || cmpRatio == 1,
                "kv_object_domain=RAW_TOKEN requires cmp_ratio=1.");

    const std::vector<int64_t> sparseShape =
        SparseOutputShape(query, key, layoutQuery, layoutKey);
    const int64_t rowCount = [&]() {
        int64_t rows = 1;
        for (size_t index = 0; index + 1 < sparseShape.size(); ++index) {
            rows *= sparseShape[index];
        }
        return rows;
    }();
    TORCH_CHECK(residentThreshold.dim() == 1 &&
                    residentThreshold.numel() == rowCount &&
                    residentThresholdValid.dim() == 1 &&
                    residentThresholdValid.numel() == rowCount &&
                    reqPoolEntries.dim() == 1 && reqPoolEntries.numel() == rowCount &&
                    kvObjectLens.dim() == 1 && kvObjectLens.numel() == rowCount,
                "per-row streaming tensors must have R=ProductExceptLast(sparse_indices) elements.");
    TORCH_CHECK(kvObjectSlotsSnapshot.dim() == 2 &&
                    kvObjectSlotsSnapshot.size(0) > 0 &&
                    kvObjectSlotsSnapshot.size(1) > 0 &&
                    kvObjectSlotsSnapshot.size(1) <= kMaxSourceCapacity &&
                    kvObjectMap.dim() == 2 &&
                    kvObjectMap.size(0) == kvObjectSlotsSnapshot.size(0) &&
                    kvObjectMap.size(1) > 0 &&
                    kvObjectMap.size(1) <= kMaxSourceCapacity &&
                    privateSourceToSlot.dim() == 2 &&
                    privateSourceToSlot.size(0) == rowCount &&
                    privateSourceToSlot.size(1) == kvObjectMap.size(1),
                "KV-object/private source maps do not meet the S-to-object ABI.");
    TORCH_CHECK(privateReadyTable.dim() == 2 && privateReadyTable.size(0) > 0 &&
                    privateReadyTable.size(1) > 0 &&
                    privateArenaHead.dim() == 1 &&
                    privateArenaHead.numel() == privateReadyTable.size(0) &&
                    queueLeases.dim() == 1 &&
                    queueLeases.numel() == privateReadyTable.size(0) &&
                    privatePackedOutput.dim() == 2 &&
                    privatePackedOutput.size(1) == kPackedKvRowBytes &&
                    privatePackedOutput.size(0) == privateReadyTable.numel(),
                "private arena aliases do not meet the queue-by-row ABI.");
    TORCH_CHECK(queueControl.dim() == 1 && queueControl.numel() == kControlWords,
                "queue_control must be int64[16].");
    const int64_t sourceCapacity = kvObjectMap.size(1);
    const int64_t tileCount =
        (sourceCapacity + kScoreTileWidth - 1) / kScoreTileWidth;
    TORCH_CHECK(scoreWorkspace.dim() == 1 &&
                    scoreWorkspace.numel() >= ScoreWorkspaceWords(sourceCapacity),
                "score_workspace must be int32[>=36*4*aligned_source/2].");
    TORCH_CHECK(tileReceipts.dim() == 2 &&
                    tileReceipts.size(0) == rowCount * tileCount &&
                    tileReceipts.size(1) == kTileReceiptWords,
                "tile_receipts must be int64[R*ceil(S/128),16].");
    TORCH_CHECK(streamingEpoch.dim() == 1 && streamingEpoch.numel() == 1,
                "streaming_epoch must be int64[1].");
}

Result AllocateAndLaunch(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights,
    const at::Tensor& queryScale, const at::Tensor& keyScale,
    const at::Tensor& actualSeqLengthsQ, const at::Tensor& actualSeqLengthsK,
    const at::Tensor& blockTable, const at::Tensor& metadata,
    const at::Tensor& residentThreshold,
    const at::Tensor& residentThresholdValid,
    const at::Tensor& kvObjectSlotsSnapshot,
    const at::Tensor& reqPoolEntries, const at::Tensor& kvObjectMap,
    at::Tensor privateSourceToSlot,
    at::Tensor privateReadyTable, at::Tensor privateArenaHead,
    at::Tensor privatePackedOutput, const at::Tensor& queueControl,
    at::Tensor queueLeases, const at::Tensor& kvObjectLens,
    at::Tensor scoreWorkspace, at::Tensor tileReceipts,
    const at::Tensor& streamingEpoch,
    int64_t queryQuantMode, int64_t keyQuantMode,
    const std::string& layoutQuery, const std::string& layoutKey,
    int64_t sparseCount, int64_t sparseMode, int64_t preTokens,
    int64_t nextTokens, int64_t cmpRatio, bool returnValues,
    int64_t stride, int64_t scaleStride, int64_t kvafRowsPerEntry,
    int64_t kvObjectDomain)
{
    CheckInputs(query, key, weights, queryScale, keyScale, actualSeqLengthsQ,
                actualSeqLengthsK, blockTable, metadata, residentThreshold,
                residentThresholdValid, kvObjectSlotsSnapshot, reqPoolEntries,
                kvObjectMap, privateSourceToSlot, privateReadyTable,
                privateArenaHead, privatePackedOutput, queueControl, queueLeases,
                kvObjectLens, scoreWorkspace, tileReceipts, streamingEpoch,
                layoutQuery, layoutKey, sparseCount, cmpRatio,
                kvafRowsPerEntry, kvObjectDomain);
    const std::vector<int64_t> sparseShape =
        SparseOutputShape(query, key, layoutQuery, layoutKey);
    auto int32Options = query.options().dtype(at::kInt);
    auto floatOptions = query.options().dtype(at::kFloat);
    auto int64Options = query.options().dtype(at::kLong);
    auto sparseIndices = at::empty(sparseShape, int32Options);
    auto sparseValues = returnValues ? at::empty(sparseShape, floatOptions) :
                                      at::empty({0}, floatOptions);
    const int64_t rowCount = ProductExceptLast(sparseIndices);
    auto telemetry = at::zeros({queueLeases.numel(), kTelemetryColumns}, int64Options);
    const PostClosureMode postClosureMode = ResolvePostClosureMode();
    const bool scoreSourceLensAlias =
        postClosureMode == PostClosureMode::kEmptyShell;
    const bool scoreSourceLensDebugPrefill =
        ScoreSourceLensDebugPrefillEnabled();
    TORCH_CHECK(!scoreSourceLensAlias || !scoreSourceLensDebugPrefill,
                "NANOVLLM_A5_SCORE_SOURCE_LENS_DEBUG_PREFILL is incompatible "
                "with empty-shell input aliasing.");
    if (scoreSourceLensAlias) {
        CheckStrictDecodeScoreSourceLensAlias(
            query, actualSeqLengthsK, layoutQuery, cmpRatio, rowCount);
    }
    auto scoreSourceLens = scoreSourceLensAlias
        ? actualSeqLengthsK
        : (scoreSourceLensDebugPrefill
            ? at::full({rowCount}, kScoreSourceLensDebugSentinel, int32Options)
            : at::empty({rowCount}, int32Options));

    // Every receipt is intentionally the same physical allocation as its
    // matching mutable input. This is a graph ownership proof, not a copy.
    auto privateSourceToSlotOut = privateSourceToSlot;
    auto privateReadyTableOut = privateReadyTable;
    auto privateArenaHeadOut = privateArenaHead;
    auto privatePackedOutputOut = privatePackedOutput;
    auto queueLeasesOut = queueLeases;
    auto scoreWorkspaceOut = scoreWorkspace;
    auto tileReceiptsOut = tileReceipts;
    if (DebugEnabled()) {
        std::fprintf(stderr,
                     "P3_LI_ACLNN_CALL inputs=24 outputs=11 rows=%lld "
                     "sparse_shape=%s cmp_ratio=%lld return_values=%d\n",
                     static_cast<long long>(rowCount),
                     ShapeString(sparseIndices.sizes()).c_str(),
                     static_cast<long long>(cmpRatio), returnValues ? 1 : 0);
        std::fprintf(stderr,
                     "P3_LI_SCORE_SOURCE_LENS_PREFILL enabled=%d sentinel=%d\n",
                     scoreSourceLensDebugPrefill ? 1 : 0,
                     static_cast<int>(kScoreSourceLensDebugSentinel));
        std::fprintf(stderr,
                     "P3_LI_SCORE_SOURCE_LENS_ALIAS actual_seq_lengths_key=%d "
                     "post_closure_mode=%s\n",
                     scoreSourceLensAlias ? 1 : 0,
                     scoreSourceLensAlias ? "empty_shell" : "full");
        for (const auto& item : {
                 std::pair<const char*, const at::Tensor*>{"in0.query", &query},
                 {"in1.key", &key}, {"in2.weights", &weights},
                 {"in3.query_scale", &queryScale}, {"in4.key_scale", &keyScale},
                 {"in9.resident_threshold", &residentThreshold},
                 {"in10.resident_threshold_valid", &residentThresholdValid},
                 {"in11.kv_object_slots_snapshot", &kvObjectSlotsSnapshot},
                 {"in13.kv_object_map", &kvObjectMap},
                 {"in14.private_source_to_slot", &privateSourceToSlot},
                 {"in17.private_packed_output", &privatePackedOutput},
                 {"in20.kv_object_lens", &kvObjectLens},
                 {"in21.score_workspace", &scoreWorkspace},
                 {"in22.tile_receipts", &tileReceipts},
                 {"in23.streaming_epoch", &streamingEpoch},
                 {"out0.sparse_indices", &sparseIndices},
                 {"out7.telemetry", &telemetry},
                 {"out8.score_source_lens", &scoreSourceLens},
                 {"out9.score_workspace_out", &scoreWorkspaceOut},
                 {"out10.tile_receipts_out", &tileReceiptsOut},
             }) {
            LogTensor(item.first, *item.second);
        }
        std::fflush(stderr);
    }

    const auto keyArg = nanovllm_a5_opapi::AllowStridedAclTensor(key);
    const char* layoutQueryPtr = layoutQuery.c_str();
    const char* layoutKeyPtr = layoutKey.c_str();
    auto keepalive = std::make_tuple(
        query, key, weights, queryScale, keyScale, actualSeqLengthsQ,
        actualSeqLengthsK, blockTable, metadata, residentThreshold,
        residentThresholdValid, kvObjectSlotsSnapshot, reqPoolEntries,
        kvObjectMap, privateSourceToSlot, privateReadyTable,
        privateArenaHead, privatePackedOutput, queueControl, queueLeases,
        kvObjectLens, scoreWorkspace, tileReceipts, streamingEpoch,
        sparseIndices, sparseValues, privateSourceToSlotOut,
        privateReadyTableOut, privateArenaHeadOut, privatePackedOutputOut,
        queueLeasesOut, telemetry, scoreSourceLens, scoreWorkspaceOut,
        tileReceiptsOut);
    // The public Torch schema puts the required kv_object_domain before the
    // optional C8 attributes. The generated CANN ABI is separately fixed:
    // ACLNN ABI keeps kvObjectDomain after kvafRowsPerEntry. Keep this
    // reorder explicit; changing only this call without regenerating the
    // complete OPP would misinterpret the dynamically resolved signature.
    EXEC_NPU_CMD_ORDERED(
        aclnnA5StreamingQuantLiGet, keepalive,
        query, keyArg, weights, queryScale, keyScale, actualSeqLengthsQ,
        actualSeqLengthsK, blockTable, metadata, residentThreshold,
        residentThresholdValid, kvObjectSlotsSnapshot, reqPoolEntries,
        kvObjectMap, privateSourceToSlot, privateReadyTable,
        privateArenaHead, privatePackedOutput, queueControl, queueLeases,
        kvObjectLens, scoreWorkspace, tileReceipts, streamingEpoch,
        queryQuantMode, keyQuantMode, layoutQueryPtr,
        layoutKeyPtr, sparseCount, sparseMode, preTokens, nextTokens, cmpRatio,
        returnValues, stride, scaleStride, kvafRowsPerEntry, kvObjectDomain,
        sparseIndices,
        sparseValues, privateSourceToSlotOut, privateReadyTableOut,
        privateArenaHeadOut, privatePackedOutputOut, queueLeasesOut, telemetry,
        scoreSourceLens, scoreWorkspaceOut, tileReceiptsOut);
    return std::make_tuple(
        sparseIndices, sparseValues, privateSourceToSlotOut,
        privateReadyTableOut, privateArenaHeadOut, privatePackedOutputOut,
        queueLeasesOut, telemetry, scoreSourceLens, scoreWorkspaceOut,
        tileReceiptsOut);
}

Result StreamingQuantLiGetNpu(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights,
    const at::Tensor& queryScale, const at::Tensor& keyScale,
    const at::Tensor& actualSeqLengthsQ, const at::Tensor& actualSeqLengthsK,
    const at::Tensor& blockTable, const at::Tensor& metadata,
    const at::Tensor& residentThreshold,
    const at::Tensor& residentThresholdValid,
    const at::Tensor& kvObjectSlotsSnapshot,
    const at::Tensor& reqPoolEntries, const at::Tensor& kvObjectMap,
    at::Tensor privateSourceToSlot,
    at::Tensor privateReadyTable, at::Tensor privateArenaHead,
    at::Tensor privatePackedOutput, const at::Tensor& queueControl,
    at::Tensor queueLeases, const at::Tensor& kvObjectLens,
    at::Tensor scoreWorkspace, at::Tensor tileReceipts,
    const at::Tensor& streamingEpoch,
    int64_t kvObjectDomain, int64_t queryQuantMode, int64_t keyQuantMode, std::string layoutQuery,
    std::string layoutKey, int64_t sparseCount, int64_t sparseMode,
    int64_t preTokens, int64_t nextTokens, int64_t cmpRatio,
    bool returnValues, int64_t stride, int64_t scaleStride,
    int64_t kvafRowsPerEntry)
{
    return AllocateAndLaunch(
        query, key, weights, queryScale, keyScale, actualSeqLengthsQ,
        actualSeqLengthsK, blockTable, metadata, residentThreshold,
        residentThresholdValid, kvObjectSlotsSnapshot, reqPoolEntries,
        kvObjectMap, privateSourceToSlot, privateReadyTable,
        privateArenaHead, privatePackedOutput, queueControl, queueLeases,
        kvObjectLens, scoreWorkspace, tileReceipts, streamingEpoch,
        queryQuantMode, keyQuantMode, layoutQuery, layoutKey,
        sparseCount, sparseMode, preTokens, nextTokens, cmpRatio, returnValues,
        stride, scaleStride, kvafRowsPerEntry, kvObjectDomain);
}

Result StreamingQuantLiGetMeta(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights,
    const at::Tensor& queryScale, const at::Tensor& keyScale,
    const at::Tensor& actualSeqLengthsQ, const at::Tensor& actualSeqLengthsK,
    const at::Tensor& blockTable, const at::Tensor& metadata,
    const at::Tensor& residentThreshold,
    const at::Tensor& residentThresholdValid,
    const at::Tensor& kvObjectSlotsSnapshot,
    const at::Tensor& reqPoolEntries, const at::Tensor& kvObjectMap,
    at::Tensor privateSourceToSlot,
    at::Tensor privateReadyTable, at::Tensor privateArenaHead,
    at::Tensor privatePackedOutput, const at::Tensor& queueControl,
    at::Tensor queueLeases, const at::Tensor& kvObjectLens,
    at::Tensor scoreWorkspace, at::Tensor tileReceipts,
    const at::Tensor& streamingEpoch,
    int64_t kvObjectDomain, int64_t queryQuantMode, int64_t keyQuantMode, std::string layoutQuery,
    std::string layoutKey, int64_t sparseCount, int64_t sparseMode,
    int64_t preTokens, int64_t nextTokens, int64_t cmpRatio,
    bool returnValues, int64_t stride, int64_t scaleStride,
    int64_t kvafRowsPerEntry)
{
    // Meta validates the static shape/ABI but never resolves or launches ACLNN.
    CheckInputs(query, key, weights, queryScale, keyScale, actualSeqLengthsQ,
                actualSeqLengthsK, blockTable, metadata, residentThreshold,
                residentThresholdValid, kvObjectSlotsSnapshot, reqPoolEntries,
                kvObjectMap, privateSourceToSlot, privateReadyTable,
                privateArenaHead, privatePackedOutput, queueControl, queueLeases,
                kvObjectLens, scoreWorkspace, tileReceipts, streamingEpoch,
                layoutQuery, layoutKey, sparseCount, cmpRatio,
                kvafRowsPerEntry, kvObjectDomain);
    const std::vector<int64_t> sparseShape =
        SparseOutputShape(query, key, layoutQuery, layoutKey);
    auto int32Options = query.options().dtype(at::kInt);
    auto floatOptions = query.options().dtype(at::kFloat);
    auto int64Options = query.options().dtype(at::kLong);
    auto sparseIndices = at::empty(sparseShape, int32Options);
    auto sparseValues = returnValues ? at::empty(sparseShape, floatOptions) :
                                      at::empty({0}, floatOptions);
    const int64_t rowCount = ProductExceptLast(sparseIndices);
    const bool scoreSourceLensAlias =
        ResolvePostClosureMode() == PostClosureMode::kEmptyShell;
    if (scoreSourceLensAlias) {
        CheckStrictDecodeScoreSourceLensAlias(
            query, actualSeqLengthsK, layoutQuery, cmpRatio, rowCount);
    }
    return std::make_tuple(
        sparseIndices, sparseValues, privateSourceToSlot, privateReadyTable,
        privateArenaHead, privatePackedOutput, queueLeases,
        at::empty({queueLeases.numel(), kTelemetryColumns}, int64Options),
        scoreSourceLensAlias ? actualSeqLengthsK : at::empty({rowCount}, int32Options),
        scoreWorkspace, tileReceipts);
}
}  // namespace

TORCH_LIBRARY(a5_streaming_quant_li_get, m)
{
    m.def(
        "streaming_quant_li_get(Tensor query, Tensor key, Tensor weights, "
        "Tensor query_dequant_scale, Tensor key_dequant_scale, "
        "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_key, "
        "Tensor block_table, Tensor metadata, Tensor resident_threshold, "
        "Tensor resident_threshold_valid, Tensor kv_object_slots_snapshot, "
        "Tensor req_pool_entries, Tensor kv_object_map, "
        "Tensor private_source_to_slot, Tensor private_ready_table, "
        "Tensor private_arena_head, Tensor private_packed_output, "
        "Tensor queue_control, Tensor queue_leases, Tensor kv_object_lens, "
        "Tensor score_workspace, Tensor tile_receipts, Tensor streaming_epoch, "
        "int kv_object_domain, int query_quant_mode=0, int key_quant_mode=0, "
        "str layout_query='TND', str layout_key='PA_BSND', "
        "int sparse_count=2048, int sparse_mode=3, "
        "int pre_tokens=9223372036854775807, "
        "int next_tokens=9223372036854775807, int cmp_ratio=1, "
        "bool return_values=False, int stride=1, int scale_stride=1, "
        "int kvaf_rows_per_entry=128) -> (Tensor sparse_indices, "
        "Tensor sparse_values, Tensor private_source_to_slot_out, "
        "Tensor private_ready_table_out, Tensor private_arena_head_out, "
        "Tensor private_packed_output_out, Tensor queue_leases_out, "
        "Tensor telemetry, Tensor score_source_lens, Tensor score_workspace_out, "
        "Tensor tile_receipts_out)");
}

TORCH_LIBRARY_IMPL(a5_streaming_quant_li_get, PrivateUse1, m)
{
    m.impl("streaming_quant_li_get", &StreamingQuantLiGetNpu);
}

TORCH_LIBRARY_IMPL(a5_streaming_quant_li_get, Meta, m)
{
    m.impl("streaming_quant_li_get", &StreamingQuantLiGetMeta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
