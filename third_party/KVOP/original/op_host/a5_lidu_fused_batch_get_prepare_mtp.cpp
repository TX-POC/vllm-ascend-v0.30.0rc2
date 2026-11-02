/** Host definition and tiling for MTP fused LIDU + compact BatchGet prepare. */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "../op_kernel/a5_lidu_fused_batch_get_prepare_mtp_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t TOPK_INDICES = 0;
constexpr size_t ACTUAL_SEQ_LENGTHS_QUERY = 1;
constexpr size_t REQ_POOL_ENTRIES = 2;
constexpr size_t TOKEN_TO_SLOT_POOL = 3;
constexpr size_t SLOT_TO_TOKEN_POOL = 4;
constexpr size_t EVICTION_CURSORS = 5;
constexpr size_t CACHE_TOKENS = 6;
constexpr size_t CANDIDATE_LENS = 7;
constexpr size_t MISS_SLOTS = 8;
constexpr size_t MISS_SLOT_BITMAP = 9;
constexpr size_t RESIDENT_BLOCK_TABLE = 10;
constexpr size_t LOGICAL_SEQ_LENGTHS = 11;
constexpr size_t PROFILE_ENABLED = 12;
constexpr size_t INPUT_COUNT = 13;

constexpr size_t LAYER_ID_ATTR = 0;
constexpr size_t NUM_LAYERS_ATTR = 1;
constexpr size_t MAX_NUM_REQS_ATTR = 2;
constexpr size_t QCAP_ATTR = 3;

constexpr int64_t SPARSE_COUNT = 2048;
constexpr int64_t MIN_QUERIES_PER_REQUEST = 1;
constexpr int64_t MAX_QUERIES_PER_REQUEST = 4;
constexpr int64_t MTP_MISS_CAPACITY =
    SPARSE_COUNT * MAX_QUERIES_PER_REQUEST;
constexpr int64_t MAX_SOURCE_CAPACITY = (1 << 18) + 32;
constexpr uint32_t MAX_CACHE_TOKENS = (1U << 17) + 20U * 128U;
constexpr uint32_t MAX_CORES_PER_REQUEST = 8;
constexpr uint32_t TILING_CORE_CAP = MAX_CORES_PER_REQUEST;
constexpr uint32_t UB_BLOCK_BYTES = 32;
constexpr uint32_t UB_BLOCK_WORDS = UB_BLOCK_BYTES / sizeof(uint32_t);
constexpr uint32_t MISS_BITMAP_WORDS = (MAX_CACHE_TOKENS + 31) / 32;
constexpr uint32_t MISS_BITMAP_STORAGE_WORDS =
    (MISS_BITMAP_WORDS + UB_BLOCK_WORDS - 1) /
    UB_BLOCK_WORDS * UB_BLOCK_WORDS;
constexpr uint32_t SOURCE_BITMAP_WORDS = MAX_SOURCE_CAPACITY / 32;
constexpr uint32_t SLOT_SELECT_SCAN_PREFIX_OFFSET =
    (MISS_BITMAP_WORDS + 1 + UB_BLOCK_WORDS - 1) /
    UB_BLOCK_WORDS * UB_BLOCK_WORDS;
constexpr uint32_t SLOT_SELECT_SCRATCH_WORDS =
    SLOT_SELECT_SCAN_PREFIX_OFFSET + MISS_BITMAP_WORDS + 1;
constexpr uint32_t SOURCE_SCRATCH_STORAGE_WORDS =
    ((SLOT_SELECT_SCRATCH_WORDS > SOURCE_BITMAP_WORDS
          ? SLOT_SELECT_SCRATCH_WORDS : SOURCE_BITMAP_WORDS) + UB_BLOCK_WORDS - 1) /
    UB_BLOCK_WORDS * UB_BLOCK_WORDS;
constexpr uint32_t GLOBAL_CONTROL_WORDS = 16;
constexpr uint32_t REQUEST_CONTROL_WORDS = 16;
constexpr uint32_t DYNAMIC_UB_BYTES = static_cast<uint32_t>(
    MISS_BITMAP_STORAGE_WORDS * sizeof(uint32_t) +
    4 * SPARSE_COUNT * sizeof(int32_t) +
    REQUEST_CONTROL_WORDS * sizeof(int32_t) +
    SOURCE_SCRATCH_STORAGE_WORDS * sizeof(uint32_t) +
    MTP_MISS_CAPACITY * sizeof(int32_t));
constexpr uint32_t GM_CACHE_LINE_WORDS = 64 / sizeof(int32_t);
constexpr uint32_t CLASSIFICATION_OFFSET = 0;
constexpr uint32_t COMPACTED_TOKEN_OFFSET =
    CLASSIFICATION_OFFSET + MTP_MISS_CAPACITY;
constexpr uint32_t UNIQUE_SOURCE_OFFSET =
    COMPACTED_TOKEN_OFFSET + MTP_MISS_CAPACITY;
constexpr uint32_t RESOLVED_SLOT_OFFSET =
    UNIQUE_SOURCE_OFFSET + MTP_MISS_CAPACITY;
constexpr uint32_t ROW_CONTROL_OFFSET =
    RESOLVED_SLOT_OFFSET + MTP_MISS_CAPACITY;
constexpr uint32_t WORKSPACE_WORDS_PER_REQUEST =
    (ROW_CONTROL_OFFSET + REQUEST_CONTROL_WORDS + GM_CACHE_LINE_WORDS - 1) /
    GM_CACHE_LINE_WORDS * GM_CACHE_LINE_WORDS;
constexpr size_t WORKSPACE_ALIGNMENT = 512;
constexpr uint64_t BLOCK_SIZE = 128;

static_assert(GLOBAL_CONTROL_WORDS % GM_CACHE_LINE_WORDS == 0,
              "global control must occupy complete cache lines");
static_assert(DYNAMIC_UB_BYTES % UB_BLOCK_BYTES == 0,
              "dynamic UB allocation must be data-block aligned");
static_assert(CLASSIFICATION_OFFSET % GM_CACHE_LINE_WORDS == 0,
              "classification rows must be cache-line aligned");
static_assert(COMPACTED_TOKEN_OFFSET % GM_CACHE_LINE_WORDS == 0,
              "compacted rows must be cache-line aligned");
static_assert(UNIQUE_SOURCE_OFFSET % GM_CACHE_LINE_WORDS == 0,
              "unique sources must be cache-line aligned");
static_assert(RESOLVED_SLOT_OFFSET % GM_CACHE_LINE_WORDS == 0,
              "resolved slots must be cache-line aligned");
static_assert(ROW_CONTROL_OFFSET % GM_CACHE_LINE_WORDS == 0,
              "request control must be cache-line aligned");
static_assert(WORKSPACE_WORDS_PER_REQUEST % GM_CACHE_LINE_WORDS == 0,
              "request workspaces must not share cache lines");

size_t AlignWorkspace(size_t bytes)
{
    return (bytes + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT *
        WORKSPACE_ALIGNMENT;
}

bool IsVector(const gert::Shape &shape, int64_t size)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == size;
}

uint32_t SelectCoresPerRequest(uint32_t availableCores)
{
    if (availableCores >= 8) {
        return 8;
    }
    if (availableCores >= 4) {
        return 4;
    }
    if (availableCores >= 2) {
        return 2;
    }
    return 1;
}

}  // namespace

namespace optiling {
static ge::graphStatus TilingA5LiduFusedBatchGetPrepareMtp(
    gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr ||
        context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < INPUT_COUNT; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr ||
            context->GetInputDesc(index)->GetDataType() != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape topk =
        context->GetInputShape(TOPK_INDICES)->GetStorageShape();
    const gert::Shape actualQ =
        context->GetInputShape(ACTUAL_SEQ_LENGTHS_QUERY)->GetStorageShape();
    const gert::Shape req =
        context->GetInputShape(REQ_POOL_ENTRIES)->GetStorageShape();
    const gert::Shape tokenToSlot =
        context->GetInputShape(TOKEN_TO_SLOT_POOL)->GetStorageShape();
    const gert::Shape slotToToken =
        context->GetInputShape(SLOT_TO_TOKEN_POOL)->GetStorageShape();
    const gert::Shape cursors =
        context->GetInputShape(EVICTION_CURSORS)->GetStorageShape();
    const gert::Shape budgets =
        context->GetInputShape(CACHE_TOKENS)->GetStorageShape();
    const gert::Shape candidateLens =
        context->GetInputShape(CANDIDATE_LENS)->GetStorageShape();
    const gert::Shape missSlots =
        context->GetInputShape(MISS_SLOTS)->GetStorageShape();
    const gert::Shape missSlotBitmap =
        context->GetInputShape(MISS_SLOT_BITMAP)->GetStorageShape();
    const gert::Shape residentTable =
        context->GetInputShape(RESIDENT_BLOCK_TABLE)->GetStorageShape();
    const gert::Shape logicalLens =
        context->GetInputShape(LOGICAL_SEQ_LENGTHS)->GetStorageShape();
    const gert::Shape profileEnabled =
        context->GetInputShape(PROFILE_ENABLED)->GetStorageShape();

    if (topk.GetDimNum() != 3 || topk.GetDim(0) <= 0 ||
        topk.GetDim(1) != 1 || topk.GetDim(2) != SPARSE_COUNT ||
        actualQ.GetDimNum() != 1 || actualQ.GetDim(0) <= 0 ||
        tokenToSlot.GetDimNum() != 2 || tokenToSlot.GetDim(0) <= 0 ||
        tokenToSlot.GetDim(1) <= 0 ||
        tokenToSlot.GetDim(1) > MAX_SOURCE_CAPACITY ||
        slotToToken.GetDimNum() != 2 ||
        slotToToken.GetDim(0) != tokenToSlot.GetDim(0) ||
        slotToToken.GetDim(1) < SPARSE_COUNT ||
        cursors.GetDimNum() != 1 ||
        cursors.GetDim(0) != tokenToSlot.GetDim(0) ||
        missSlots.GetDimNum() != 2 ||
        missSlots.GetDim(1) != SPARSE_COUNT ||
        missSlotBitmap.GetDimNum() != 2 ||
        missSlotBitmap.GetDim(1) != MISS_BITMAP_STORAGE_WORDS ||
        residentTable.GetDimNum() != 2 || residentTable.GetDim(1) <= 0 ||
        profileEnabled.GetDimNum() != 1 || profileEnabled.GetDim(0) != 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = actualQ.GetDim(0);
    const int64_t packedQueries = topk.GetDim(0);
    if (!IsVector(req, batch) || !IsVector(budgets, batch) ||
        !IsVector(candidateLens, batch) || !IsVector(logicalLens, batch) ||
        missSlots.GetDim(0) != batch ||
        missSlotBitmap.GetDim(0) != batch ||
        residentTable.GetDim(0) != batch ||
        packedQueries < batch * MIN_QUERIES_PER_REQUEST ||
        packedQueries > batch * MAX_QUERIES_PER_REQUEST) {
        return ge::GRAPH_FAILED;
    }

    const int64_t *layerId =
        context->GetAttrs()->GetAttrPointer<int64_t>(LAYER_ID_ATTR);
    const int64_t *numLayers =
        context->GetAttrs()->GetAttrPointer<int64_t>(NUM_LAYERS_ATTR);
    const int64_t *maxNumReqs =
        context->GetAttrs()->GetAttrPointer<int64_t>(MAX_NUM_REQS_ATTR);
    const int64_t *qcap =
        context->GetAttrs()->GetAttrPointer<int64_t>(QCAP_ATTR);
    if (layerId == nullptr || numLayers == nullptr ||
        maxNumReqs == nullptr || qcap == nullptr ||
        *layerId < 0 || *numLayers <= 0 || *layerId >= *numLayers ||
        *maxNumReqs <= 0 || *qcap <= 0 || batch > *maxNumReqs ||
        *layerId > std::numeric_limits<uint32_t>::max() ||
        *numLayers > std::numeric_limits<uint32_t>::max() ||
        *maxNumReqs > std::numeric_limits<uint32_t>::max() ||
        *qcap > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(*maxNumReqs) *
                static_cast<uint64_t>(*numLayers) *
                static_cast<uint64_t>(*qcap) >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        static_cast<uint64_t>(residentTable.GetDim(1)) * BLOCK_SIZE >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        static_cast<uint64_t>(batch) * MTP_MISS_CAPACITY >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aivCount = platform.GetCoreNumAiv();
    if (aivCount == 0) {
        return ge::GRAPH_FAILED;
    }
    // Reuse every available AIV in sequential waves when one request per AIV
    // cannot cover the batch.
    const bool useSequentialBatchAssignment =
        batch > static_cast<int64_t>(aivCount);
    const uint32_t coresPerRequest = useSequentialBatchAssignment
        ? 1U
        : SelectCoresPerRequest(
              std::min(
                  TILING_CORE_CAP,
                  std::min(
                      MAX_CORES_PER_REQUEST,
                      std::max<uint32_t>(
                          1, aivCount / static_cast<uint32_t>(batch)))));
    const uint32_t usedCoreNum = useSequentialBatchAssignment
        ? aivCount
        : coresPerRequest * static_cast<uint32_t>(batch);

    auto *tiling =
        context->GetTilingData<A5LiduFusedBatchGetPrepareMtpTilingData>();
    if (tiling == nullptr || context->GetWorkspaceSizes(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->usedCoreNum = usedCoreNum;
    tiling->coresPerRequest = coresPerRequest;
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->packedQueryCount = static_cast<uint32_t>(packedQueries);
    tiling->poolSize = static_cast<uint32_t>(tokenToSlot.GetDim(0));
    tiling->sourceCapacity = static_cast<uint32_t>(tokenToSlot.GetDim(1));
    tiling->residentCapacity = static_cast<uint32_t>(slotToToken.GetDim(1));
    tiling->residentBlockCapacity =
        static_cast<uint32_t>(residentTable.GetDim(1));
    tiling->layerId = static_cast<uint32_t>(*layerId);
    tiling->numLayers = static_cast<uint32_t>(*numLayers);
    tiling->maxNumReqs = static_cast<uint32_t>(*maxNumReqs);
    tiling->qcap = static_cast<uint32_t>(*qcap);
    // The kernel uses launch-wide SyncAll barriers. Batch-mode scheduling is
    // required so every participating AIV can be resident concurrently; under
    // multi-stream ACLGraph replay the default schedule may otherwise leave a
    // subset of cores waiting forever at the barrier.
    if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(usedCoreNum);
    if (context->SetDynUBufSize(DYNAMIC_UB_BYTES) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const size_t userWorkspaceSize = AlignWorkspace(
        (GLOBAL_CONTROL_WORDS +
         static_cast<size_t>(WORKSPACE_WORDS_PER_REQUEST) * batch) *
        sizeof(int32_t));
    context->GetWorkspaceSizes(1)[0] =
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) +
        userWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferA5LiduFusedBatchGetPrepareMtpShape(
    gert::InferShapeContext *context)
{
    if (context == nullptr ||
        context->GetInputShape(TOPK_INDICES) == nullptr ||
        context->GetInputShape(ACTUAL_SEQ_LENGTHS_QUERY) == nullptr ||
        context->GetInputShape(TOKEN_TO_SLOT_POOL) == nullptr ||
        context->GetInputShape(SLOT_TO_TOKEN_POOL) == nullptr ||
        context->GetInputShape(EVICTION_CURSORS) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < 10; ++index) {
        if (context->GetOutputShape(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape *topk = context->GetInputShape(TOPK_INDICES);
    const int64_t batch =
        context->GetInputShape(ACTUAL_SEQ_LENGTHS_QUERY)->GetDim(0);
    *context->GetOutputShape(0) = *topk;
    context->GetOutputShape(1)->SetDimNum(1);
    context->GetOutputShape(1)->SetDim(0, batch + 1);
    for (size_t output = 2; output <= 4; ++output) {
        context->GetOutputShape(output)->SetDimNum(1);
        context->GetOutputShape(output)->SetDim(
            0, batch * MTP_MISS_CAPACITY);
    }
    context->GetOutputShape(5)->SetDimNum(1);
    context->GetOutputShape(5)->SetDim(0, batch);
    *context->GetOutputShape(6) =
        *context->GetInputShape(TOKEN_TO_SLOT_POOL);
    *context->GetOutputShape(7) =
        *context->GetInputShape(SLOT_TO_TOKEN_POOL);
    *context->GetOutputShape(8) =
        *context->GetInputShape(EVICTION_CURSORS);
    context->GetOutputShape(9)->SetDimNum(3);
    context->GetOutputShape(9)->SetDim(0, batch);
    context->GetOutputShape(9)->SetDim(
        1, A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE_LANES);
    context->GetOutputShape(9)->SetDim(
        2, A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE_FIELDS);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferA5LiduFusedBatchGetPrepareMtpDataType(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < 9; ++index) {
        context->SetOutputDataType(index, ge::DT_INT32);
    }
    context->SetOutputDataType(9, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

class A5LiduFusedBatchGetPrepareMtp : public OpDef {
public:
    explicit A5LiduFusedBatchGetPrepareMtp(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> ints = {ge::DT_INT32};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND};
        this->Input("topk_indices").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("actual_seq_lengths_query").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("req_pool_entries").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("token_to_slot_pool").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("slot_to_token_pool").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("eviction_cursors").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("cache_tokens").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("candidate_lens").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("miss_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("miss_slot_bitmap").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("resident_block_table").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("logical_seq_lengths").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Input("profile_enabled").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Attr("layer_id").AttrType(REQUIRED).Int();
        this->Attr("num_layers").AttrType(REQUIRED).Int();
        this->Attr("max_num_reqs").AttrType(REQUIRED).Int();
        this->Attr("qcap").AttrType(REQUIRED).Int();
        this->Output("destination_slots").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("miss_counts").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("indices").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("offsets").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("output_indices").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("resident_seq_lengths").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("token_to_slot_alias").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("slot_to_token_alias").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("eviction_cursors_alias").ParamType(REQUIRED).DataType(ints).Format(formats);
        this->Output("profiling").ParamType(REQUIRED)
            .DataType({ge::DT_INT64}).Format(formats);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore()
            .SetTiling(optiling::TilingA5LiduFusedBatchGetPrepareMtp)
            .AddConfig("ascend950", config);
    }
};
OP_ADD(A5LiduFusedBatchGetPrepareMtp);

IMPL_OP_INFERSHAPE(A5LiduFusedBatchGetPrepareMtp)
    .InferShape(InferA5LiduFusedBatchGetPrepareMtpShape)
    .InferDataType(InferA5LiduFusedBatchGetPrepareMtpDataType);
}  // namespace ops
