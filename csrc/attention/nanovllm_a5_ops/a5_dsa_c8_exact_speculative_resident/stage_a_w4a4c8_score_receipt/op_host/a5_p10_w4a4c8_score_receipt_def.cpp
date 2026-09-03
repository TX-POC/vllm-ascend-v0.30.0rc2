// P10 Stage-A host registration and tiling.
//
// This is intentionally a full native W4A4C8 entrypoint rather than a
// wrapper around an existing C8 OPP.  The caller supplies the canonical
// Top-K buffer, full native score surface, receipt ring/state, and the shared
// forward-control alias before the graph starts.  Stage-B is launched from
// those cold-bound aliases independently; none of the trailing alias outputs
// is a Stage-A -> Stage-B completion dependency.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include "../op_kernel/a5_p10_w4a4c8_score_receipt_abi.h"
#include "../op_kernel/a5_p10_w4a4c8_score_receipt_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
enum InputIndex : size_t {
    QUERY = 0,
    KEY,
    WEIGHTS,
    QUERY_DEQUANT_SCALE,
    KEY_DEQUANT_SCALE,
    ACTUAL_SEQ_LENGTHS_QUERY,
    CANDIDATE_LENS,
    BLOCK_TABLE,
    CANONICAL_SPARSE_INDICES_BUFFER,
    CANONICAL_SCORE_SOURCE_LENS_BUFFER,
    SCORE_SURFACE_BUFFER,
    RECEIPT_RING_BUFFER,
    RECEIPT_STATE_BUFFER,
    FORWARD_CONTROL_BUFFER,
};

enum OutputIndex : size_t {
    CANONICAL_SPARSE_INDICES_ALIAS = 0,
    CANONICAL_SCORE_SOURCE_LENS_ALIAS,
    SCORE_SURFACE_ALIAS,
    RECEIPT_RING_ALIAS,
    RECEIPT_STATE_ALIAS,
    FORWARD_CONTROL_ALIAS,
};

constexpr int64_t kBlockSize = 128;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kTopK = A5_P10_SPARSE_COUNT;
constexpr int64_t kReceiptWords64 = A5_P10_RECEIPT_WORDS64;
constexpr int64_t kMaxSourceCapacity = 1 << 18;

bool IsShape(
    const gert::Shape &shape, std::initializer_list<int64_t> dimensions)
{
    if (shape.GetDimNum() != dimensions.size()) {
        return false;
    }
    size_t index = 0;
    for (const int64_t expected : dimensions) {
        if (expected >= 0 && shape.GetDim(index) != expected) {
            return false;
        }
        ++index;
    }
    return true;
}

bool HasInput(gert::TilingContext *context, size_t index)
{
    return context->GetInputShape(index) != nullptr &&
        context->GetInputDesc(index) != nullptr;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingA5P10W4A4C8ScoreReceipt(
    gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = QUERY; index <= FORWARD_CONTROL_BUFFER; ++index) {
        if (!HasInput(context, index)) {
            return ge::GRAPH_FAILED;
        }
    }

    const auto dtype = [context](size_t index) {
        return context->GetInputDesc(index)->GetDataType();
    };
    if (dtype(QUERY) != ge::DT_FLOAT8_E4M3FN ||
        dtype(KEY) != ge::DT_FLOAT8_E4M3FN ||
        dtype(WEIGHTS) != ge::DT_BF16 ||
        dtype(QUERY_DEQUANT_SCALE) != ge::DT_FLOAT ||
        dtype(KEY_DEQUANT_SCALE) != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index : {ACTUAL_SEQ_LENGTHS_QUERY, CANDIDATE_LENS,
                         BLOCK_TABLE, CANONICAL_SPARSE_INDICES_BUFFER,
                         CANONICAL_SCORE_SOURCE_LENS_BUFFER,
                         RECEIPT_STATE_BUFFER, FORWARD_CONTROL_BUFFER}) {
        if (dtype(index) != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }
    if (dtype(SCORE_SURFACE_BUFFER) != ge::DT_UINT16 ||
        dtype(RECEIPT_RING_BUFFER) != ge::DT_INT64) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape query =
        context->GetInputShape(QUERY)->GetStorageShape();
    const gert::Shape key = context->GetInputShape(KEY)->GetStorageShape();
    const gert::Shape weights =
        context->GetInputShape(WEIGHTS)->GetStorageShape();
    const gert::Shape queryScale =
        context->GetInputShape(QUERY_DEQUANT_SCALE)->GetStorageShape();
    const gert::Shape keyScale =
        context->GetInputShape(KEY_DEQUANT_SCALE)->GetStorageShape();
    const gert::Shape actualQ =
        context->GetInputShape(ACTUAL_SEQ_LENGTHS_QUERY)->GetStorageShape();
    const gert::Shape candidateLens =
        context->GetInputShape(CANDIDATE_LENS)->GetStorageShape();
    const gert::Shape blockTable =
        context->GetInputShape(BLOCK_TABLE)->GetStorageShape();
    const gert::Shape canonicalIndices =
        context->GetInputShape(CANONICAL_SPARSE_INDICES_BUFFER)
            ->GetStorageShape();
    const gert::Shape canonicalLens =
        context->GetInputShape(CANONICAL_SCORE_SOURCE_LENS_BUFFER)
            ->GetStorageShape();
    const gert::Shape scoreSurface =
        context->GetInputShape(SCORE_SURFACE_BUFFER)->GetStorageShape();
    const gert::Shape receiptRing =
        context->GetInputShape(RECEIPT_RING_BUFFER)->GetStorageShape();
    const gert::Shape receiptState =
        context->GetInputShape(RECEIPT_STATE_BUFFER)->GetStorageShape();
    const gert::Shape forwardControl =
        context->GetInputShape(FORWARD_CONTROL_BUFFER)->GetStorageShape();

    if (!IsShape(query, {-1, -1, kHeadDim}) ||
        !IsShape(key, {-1, kBlockSize, 1, kHeadDim}) ||
        !IsShape(weights, {-1, -1}) || !IsShape(queryScale, {-1, -1}) ||
        !IsShape(keyScale, {-1, kBlockSize, 1}) || !IsShape(actualQ, {-1}) ||
        !IsShape(candidateLens, {-1}) || !IsShape(blockTable, {-1, -1}) ||
        !IsShape(canonicalIndices, {-1, 1, kTopK}) ||
        !IsShape(canonicalLens, {-1}) || !IsShape(scoreSurface, {-1, -1, -1}) ||
        !IsShape(receiptRing, {-1, kReceiptWords64}) ||
        !IsShape(receiptState, {-1}) || !IsShape(forwardControl, {-1})) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = query.GetDim(0);
    const int64_t heads = query.GetDim(1);
    const int64_t sourceCapacity = blockTable.GetDim(1) * kBlockSize;
    const int64_t blocks = blockTable.GetDim(1);
    if (batch <= 0 || (heads != 32 && heads != 64) || key.GetDim(0) <= 0 ||
        weights.GetDim(0) != batch || weights.GetDim(1) != heads ||
        queryScale.GetDim(0) != batch || queryScale.GetDim(1) != heads ||
        keyScale.GetDim(0) != key.GetDim(0) || actualQ.GetDim(0) != batch ||
        candidateLens.GetDim(0) != batch || blockTable.GetDim(0) != batch ||
        blocks <= 0 || sourceCapacity <= 0 ||
        sourceCapacity > kMaxSourceCapacity || canonicalIndices.GetDim(0) != batch ||
        canonicalLens.GetDim(0) != batch ||
        forwardControl.GetDim(0) < A5_P10_FORWARD_CONTROL_MIN_WORDS) {
        return ge::GRAPH_FAILED;
    }

    const int64_t s1Base = (256 + heads - 1) / heads;
    const uint64_t scoreSliceElements =
        static_cast<uint64_t>(s1Base) * static_cast<uint64_t>(sourceCapacity);
    const uint64_t scoreSliceBytes =
        scoreSliceElements * sizeof(uint16_t);
    const uint64_t receiptCapacity64 =
        static_cast<uint64_t>(batch) * static_cast<uint64_t>(blocks);
    if (scoreSliceBytes > std::numeric_limits<uint32_t>::max() ||
        receiptCapacity64 == 0 ||
        receiptCapacity64 > std::numeric_limits<uint32_t>::max() ||
        scoreSurface.GetDim(0) != batch || scoreSurface.GetDim(1) != s1Base ||
        scoreSurface.GetDim(2) != sourceCapacity ||
        receiptRing.GetDim(0) != static_cast<int64_t>(receiptCapacity64) ||
        receiptState.GetDim(0) != static_cast<int64_t>(receiptCapacity64)) {
        return ge::GRAPH_FAILED;
    }

    // Full native stride is mandatory.  C8 Top-K reads this same GM layout;
    // accepting a compact score tile here would silently change the original
    // Top-K byte stream.
    const uint64_t scoreRowBytes =
        static_cast<uint64_t>(sourceCapacity) * sizeof(uint16_t);
    if (scoreRowBytes > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aicCount = platform.GetCoreNumAic();
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aicCount == 0 || aivCount < 2U) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t usedCoreNum = std::min<uint32_t>(
        static_cast<uint32_t>(batch), aicCount);
    if (usedCoreNum == 0U) {
        return ge::GRAPH_FAILED;
    }

    auto *tiling =
        context->GetTilingData<A5P10W4A4C8ScoreReceiptTilingData>();
    if (tiling == nullptr || context->GetWorkspaceSizes(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->sourceCapacity = static_cast<uint32_t>(sourceCapacity);
    tiling->indexHeads = static_cast<uint32_t>(heads);
    tiling->maxBlockNumPerBatch = static_cast<uint32_t>(blocks);
    tiling->maxCandidateLen = static_cast<uint32_t>(sourceCapacity);
    tiling->keyStride = kBlockSize * kHeadDim;
    tiling->scaleStride = kBlockSize;
    tiling->scoreWorkspaceStride = static_cast<uint32_t>(scoreSliceBytes);
    tiling->scoreSurfaceSliceStrideBytes = static_cast<uint32_t>(scoreSliceBytes);
    tiling->scoreSurfaceRowStrideBytes = static_cast<uint32_t>(scoreRowBytes);
    tiling->receiptTilesPerRow = static_cast<uint32_t>(blocks);
    tiling->receiptWords64 = A5_P10_RECEIPT_WORDS64;
    tiling->receiptCapacity = static_cast<uint32_t>(receiptCapacity64);

    // P10 eliminated the opaque per-core score workspace.  Only the regular
    // CANN library workspace remains; score surface/ring/state are explicit
    // graph-owned inputs and must never be placed in opaque workspace.
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    context->SetBlockDim(platform.CalcTschBlockDim(
        usedCoreNum * 2U, usedCoreNum, usedCoreNum * 2U));
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferA5P10W4A4C8ScoreReceiptShape(
    gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    constexpr size_t kInputForOutput[] = {
        CANONICAL_SPARSE_INDICES_BUFFER,
        CANONICAL_SCORE_SOURCE_LENS_BUFFER,
        SCORE_SURFACE_BUFFER,
        RECEIPT_RING_BUFFER,
        RECEIPT_STATE_BUFFER,
        FORWARD_CONTROL_BUFFER,
    };
    for (size_t output = 0; output <= FORWARD_CONTROL_ALIAS; ++output) {
        if (context->GetInputShape(kInputForOutput[output]) == nullptr ||
            context->GetOutputShape(output) == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *context->GetOutputShape(output) =
            *context->GetInputShape(kInputForOutput[output]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferA5P10W4A4C8ScoreReceiptDataType(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(CANONICAL_SPARSE_INDICES_ALIAS, ge::DT_INT32);
    context->SetOutputDataType(CANONICAL_SCORE_SOURCE_LENS_ALIAS,
                               ge::DT_INT32);
    context->SetOutputDataType(SCORE_SURFACE_ALIAS, ge::DT_UINT16);
    context->SetOutputDataType(RECEIPT_RING_ALIAS, ge::DT_INT64);
    context->SetOutputDataType(RECEIPT_STATE_ALIAS, ge::DT_INT32);
    context->SetOutputDataType(FORWARD_CONTROL_ALIAS, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

class A5P10W4A4C8ScoreReceipt : public OpDef {
 public:
    explicit A5P10W4A4C8ScoreReceipt(const char *name) : OpDef(name)
    {
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        const std::vector<ge::DataType> fp8 = {ge::DT_FLOAT8_E4M3FN};
        const std::vector<ge::DataType> bf16 = {ge::DT_BF16};
        const std::vector<ge::DataType> fp32 = {ge::DT_FLOAT};
        const std::vector<ge::DataType> int32s = {ge::DT_INT32};
        const std::vector<ge::DataType> int64s = {ge::DT_INT64};
        const std::vector<ge::DataType> uint16s = {ge::DT_UINT16};
        this->Input("query").ParamType(REQUIRED).DataType(fp8).Format(formats);
        this->Input("key").ParamType(REQUIRED).DataType(fp8).Format(formats);
        this->Input("weights").ParamType(REQUIRED).DataType(bf16).Format(formats);
        this->Input("query_dequant_scale").ParamType(REQUIRED)
            .DataType(fp32).Format(formats);
        this->Input("key_dequant_scale").ParamType(REQUIRED)
            .DataType(fp32).Format(formats);
        this->Input("actual_seq_lengths_query").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Input("candidate_lens").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Input("block_table").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Input("canonical_sparse_indices_buffer").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Input("canonical_score_source_lens_buffer").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Input("score_surface_buffer").ParamType(REQUIRED)
            .DataType(uint16s).Format(formats);
        this->Input("receipt_ring_buffer").ParamType(REQUIRED)
            .DataType(int64s).Format(formats);
        this->Input("receipt_state_buffer").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Input("forward_control_buffer").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);

        // These are aliases for graph visibility only.  Stage-B receives the
        // corresponding input buffers when it is launched; graph lowering
        // must not introduce a completion dependency through these handles.
        this->Output("canonical_sparse_indices_alias").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Output("canonical_score_source_lens_alias").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Output("score_surface_alias").ParamType(REQUIRED)
            .DataType(uint16s).Format(formats);
        this->Output("receipt_ring_alias").ParamType(REQUIRED)
            .DataType(int64s).Format(formats);
        this->Output("receipt_state_alias").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);
        this->Output("forward_control_alias").ParamType(REQUIRED)
            .DataType(int32s).Format(formats);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            // CANN derives this normalized stem from the operator name.  It
            // must name the generated/remapped kernel source exactly; the
            // checked-in source filename is only an authoring spelling.
            .ExtendCfgInfo("opFile.value", "a5_p10_w4_a4_c8_score_receipt")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore()
            .SetTiling(optiling::TilingA5P10W4A4C8ScoreReceipt)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5P10W4A4C8ScoreReceipt);

IMPL_OP_INFERSHAPE(A5P10W4A4C8ScoreReceipt)
    .InferShape(InferA5P10W4A4C8ScoreReceiptShape)
    .InferDataType(InferA5P10W4A4C8ScoreReceiptDataType);
}  // namespace ops
