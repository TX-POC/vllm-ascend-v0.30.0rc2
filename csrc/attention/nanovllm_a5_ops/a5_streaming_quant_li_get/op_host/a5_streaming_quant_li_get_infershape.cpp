/**
 * Source-derived infer-shape for the isolated A5 streaming LI prototype.
 *
 * Output 0/1 intentionally follow VllmQuantLightningIndexer byte-for-byte
 * at the shape level. The remaining outputs expose transaction-local
 * receipts; they are not a cache-table publication mechanism.
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include <climits>

#include "err/ops_err.h"
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t RESIDENT_THRESHOLD_INDEX = 9;
constexpr uint32_t RESIDENT_THRESHOLD_VALID_INDEX = 10;
constexpr uint32_t KV_OBJECT_SLOTS_SNAPSHOT_INDEX = 11;
constexpr uint32_t REQ_POOL_ENTRIES_INDEX = 12;
constexpr uint32_t KV_OBJECT_MAP_INDEX = 13;
constexpr uint32_t PRIVATE_SOURCE_TO_SLOT_INDEX = 14;
constexpr uint32_t PRIVATE_READY_TABLE_INDEX = 15;
constexpr uint32_t PRIVATE_ARENA_HEAD_INDEX = 16;
constexpr uint32_t PRIVATE_PACKED_OUTPUT_INDEX = 17;
constexpr uint32_t QUEUE_CONTROL_INDEX = 18;
constexpr uint32_t QUEUE_LEASES_INDEX = 19;
constexpr uint32_t KV_OBJECT_LENS_INDEX = 20;
constexpr uint32_t SCORE_WORKSPACE_INDEX = 21;
constexpr uint32_t TILE_RECEIPTS_INDEX = 22;
constexpr uint32_t STREAMING_EPOCH_INDEX = 23;

constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 2;
constexpr uint32_t ATTR_KV_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 4;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 9;
constexpr uint32_t ATTR_KVAF_ROWS_PER_ENTRY_INDEX = 12;
constexpr uint32_t ATTR_CMP_RATIO_INDEX = 8;
constexpr uint32_t ATTR_KV_OBJECT_DOMAIN_INDEX = 13;
constexpr uint32_t SPARSE_INDICES_OUTPUT_INDEX = 0;
constexpr uint32_t SPARSE_VALUES_OUTPUT_INDEX = 1;
constexpr uint32_t PRIVATE_SOURCE_TO_SLOT_OUTPUT_INDEX = 2;
constexpr uint32_t PRIVATE_READY_TABLE_OUTPUT_INDEX = 3;
constexpr uint32_t PRIVATE_ARENA_HEAD_OUTPUT_INDEX = 4;
constexpr uint32_t PRIVATE_PACKED_OUTPUT_OUTPUT_INDEX = 5;
constexpr uint32_t QUEUE_LEASES_OUTPUT_INDEX = 6;
constexpr uint32_t TELEMETRY_OUTPUT_INDEX = 7;
constexpr uint32_t SCORE_SOURCE_LENS_OUTPUT_INDEX = 8;
constexpr uint32_t SCORE_WORKSPACE_OUTPUT_INDEX = 9;
constexpr uint32_t TILE_RECEIPTS_OUTPUT_INDEX = 10;
// Keep infer-shape, host tiling, bridge allocation, and the device hook on
// the same graph-static telemetry ABI.  Column 13 records a later exact C8
// score tile observed while a prior Direct-HCCP CQ was still pending.
constexpr uint32_t STREAMING_TELEMETRY_COLUMNS = 18;
constexpr uint32_t STREAMING_KVAF_CONTROL_WORDS = 16;
constexpr uint32_t STREAMING_PACKED_KV_ROW_BYTES = 656;
constexpr uint32_t STREAMING_TILE_RECEIPT_WORDS = 16;
constexpr uint32_t STREAMING_SCORE_TILE_WIDTH = 128;
// This is the fixed metadata/workspace capacity inherited from C8, not an
// assertion about the number of active AICs on the executing device.  Tiling
// derives the active geometry from PlatformAscendC at launch time.
constexpr int64_t STREAMING_A5_MAX_AIC_CORE_COUNT = 36;

static int64_t ProductExceptLast(const gert::Shape &shape)
{
    if (shape.GetDimNum() < 1) {
        return -1;
    }
    int64_t result = 1;
    for (int64_t i = 0; i + 1 < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim <= 0 || result > INT64_MAX / dim) {
            return -1;
        }
        result *= dim;
    }
    return result;
}

static ge::graphStatus InferShapeA5StreamingQuantLiGet(
    gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("A5StreamingQuantLiGet", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *query = context->GetInputShape(QUERY_INDEX);
    const gert::Shape *key = context->GetInputShape(KEY_INDEX);
    const gert::Shape *resident = context->GetInputShape(RESIDENT_THRESHOLD_INDEX);
    const gert::Shape *residentValid = context->GetInputShape(RESIDENT_THRESHOLD_VALID_INDEX);
    const gert::Shape *kvObjectSlots = context->GetInputShape(KV_OBJECT_SLOTS_SNAPSHOT_INDEX);
    const gert::Shape *poolEntries = context->GetInputShape(REQ_POOL_ENTRIES_INDEX);
    const gert::Shape *kvObjectMap = context->GetInputShape(KV_OBJECT_MAP_INDEX);
    const gert::Shape *sourceToSlot = context->GetInputShape(PRIVATE_SOURCE_TO_SLOT_INDEX);
    const gert::Shape *ready = context->GetInputShape(PRIVATE_READY_TABLE_INDEX);
    const gert::Shape *arenaHead = context->GetInputShape(PRIVATE_ARENA_HEAD_INDEX);
    const gert::Shape *privatePackedOutput = context->GetInputShape(PRIVATE_PACKED_OUTPUT_INDEX);
    const gert::Shape *control = context->GetInputShape(QUEUE_CONTROL_INDEX);
    const gert::Shape *leases = context->GetInputShape(QUEUE_LEASES_INDEX);
    const gert::Shape *kvObjectLens = context->GetInputShape(KV_OBJECT_LENS_INDEX);
    const gert::Shape *scoreWorkspace = context->GetInputShape(SCORE_WORKSPACE_INDEX);
    const gert::Shape *tileReceipts = context->GetInputShape(TILE_RECEIPTS_INDEX);
    const gert::Shape *streamingEpoch = context->GetInputShape(STREAMING_EPOCH_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, query);
    OP_CHECK_NULL_WITH_CONTEXT(context, key);
    OP_CHECK_NULL_WITH_CONTEXT(context, resident);
    OP_CHECK_NULL_WITH_CONTEXT(context, residentValid);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvObjectSlots);
    OP_CHECK_NULL_WITH_CONTEXT(context, poolEntries);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvObjectMap);
    OP_CHECK_NULL_WITH_CONTEXT(context, sourceToSlot);
    OP_CHECK_NULL_WITH_CONTEXT(context, ready);
    OP_CHECK_NULL_WITH_CONTEXT(context, arenaHead);
    OP_CHECK_NULL_WITH_CONTEXT(context, privatePackedOutput);
    OP_CHECK_NULL_WITH_CONTEXT(context, control);
    OP_CHECK_NULL_WITH_CONTEXT(context, leases);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvObjectLens);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoreWorkspace);
    OP_CHECK_NULL_WITH_CONTEXT(context, tileReceipts);
    OP_CHECK_NULL_WITH_CONTEXT(context, streamingEpoch);

    gert::Shape *sparseIndices = context->GetOutputShape(SPARSE_INDICES_OUTPUT_INDEX);
    gert::Shape *sparseValues = context->GetOutputShape(SPARSE_VALUES_OUTPUT_INDEX);
    gert::Shape *sourceToSlotOut = context->GetOutputShape(PRIVATE_SOURCE_TO_SLOT_OUTPUT_INDEX);
    gert::Shape *readyOut = context->GetOutputShape(PRIVATE_READY_TABLE_OUTPUT_INDEX);
    gert::Shape *arenaHeadOut = context->GetOutputShape(PRIVATE_ARENA_HEAD_OUTPUT_INDEX);
    gert::Shape *privatePackedOutputOut = context->GetOutputShape(PRIVATE_PACKED_OUTPUT_OUTPUT_INDEX);
    gert::Shape *leasesOut = context->GetOutputShape(QUEUE_LEASES_OUTPUT_INDEX);
    gert::Shape *telemetry = context->GetOutputShape(TELEMETRY_OUTPUT_INDEX);
    gert::Shape *scoreSourceLens = context->GetOutputShape(SCORE_SOURCE_LENS_OUTPUT_INDEX);
    gert::Shape *scoreWorkspaceOut = context->GetOutputShape(SCORE_WORKSPACE_OUTPUT_INDEX);
    gert::Shape *tileReceiptsOut = context->GetOutputShape(TILE_RECEIPTS_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseIndices);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseValues);
    OP_CHECK_NULL_WITH_CONTEXT(context, sourceToSlotOut);
    OP_CHECK_NULL_WITH_CONTEXT(context, readyOut);
    OP_CHECK_NULL_WITH_CONTEXT(context, arenaHeadOut);
    OP_CHECK_NULL_WITH_CONTEXT(context, privatePackedOutputOut);
    OP_CHECK_NULL_WITH_CONTEXT(context, leasesOut);
    OP_CHECK_NULL_WITH_CONTEXT(context, telemetry);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoreSourceLens);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoreWorkspaceOut);
    OP_CHECK_NULL_WITH_CONTEXT(context, tileReceiptsOut);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *queryLayoutAttr = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    const char *keyLayoutAttr = attrs->GetAttrPointer<char>(ATTR_KV_LAYOUT_INDEX);
    const int64_t *sparseCount = attrs->GetInt(ATTR_SPARSE_COUNT_INDEX);
    const int64_t *rowsPerEntry = attrs->GetAttrPointer<int64_t>(ATTR_KVAF_ROWS_PER_ENTRY_INDEX);
    const int64_t *cmpRatio = attrs->GetAttrPointer<int64_t>(ATTR_CMP_RATIO_INDEX);
    const int64_t *kvObjectDomain = attrs->GetAttrPointer<int64_t>(ATTR_KV_OBJECT_DOMAIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryLayoutAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyLayoutAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseCount);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowsPerEntry);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmpRatio);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvObjectDomain);

    const std::string queryLayout(queryLayoutAttr);
    const std::string keyLayout(keyLayoutAttr);
    if ((queryLayout != "TND" && queryLayout != "BSND") || *sparseCount <= 0 ||
        *rowsPerEntry <= 0 ||
        (*cmpRatio != 1 && *cmpRatio != 4 && *cmpRatio != 128) ||
        (*kvObjectDomain != 0 && *kvObjectDomain != 1) ||
        (*kvObjectDomain == 0 && *cmpRatio != 1)) {
        OP_LOGE(context,
                "invalid KV-object contract: cmp_ratio must be 1, 4, or 128; "
                "RAW_TOKEN requires cmp_ratio=1; "
                "C8_COMPRESSED_OBJECT is kv_object_domain=1.");
        return ge::GRAPH_FAILED;
    }
    const int64_t keyHeads = (keyLayout == "TND") ? key->GetDim(1) : key->GetDim(2);
    if (keyHeads <= 0) {
        OP_LOGE(context, "key head dimension is invalid.");
        return ge::GRAPH_FAILED;
    }

    // Preserve the official C8 output layout, including product TND behavior.
    if (queryLayout == "BSND") {
        sparseIndices->SetDimNum(4);
        sparseIndices->SetDim(0, query->GetDim(0));
        sparseIndices->SetDim(1, query->GetDim(1));
        sparseIndices->SetDim(2, keyHeads);
        sparseIndices->SetDim(3, *sparseCount);
    } else {
        sparseIndices->SetDimNum(3);
        sparseIndices->SetDim(0, query->GetDim(0));
        sparseIndices->SetDim(1, keyHeads);
        sparseIndices->SetDim(2, *sparseCount);
    }
    const bool *returnValues = attrs->GetAttrPointer<bool>(ATTR_RETURN_VALUE_INDEX);
    if (returnValues != nullptr && *returnValues) {
        *sparseValues = *sparseIndices;
    } else {
        sparseValues->SetDimNum(1);
        sparseValues->SetDim(0, 0);
    }

    const int64_t rowCount = ProductExceptLast(*sparseIndices);
    const int64_t sourceCapacity = (kvObjectMap->GetDimNum() == 2) ? kvObjectMap->GetDim(1) : -1;
    const int64_t tileCount = sourceCapacity > 0
        ? (sourceCapacity + STREAMING_SCORE_TILE_WIDTH - 1) /
            STREAMING_SCORE_TILE_WIDTH
        : -1;
    const int64_t alignedSource = tileCount > 0
        ? tileCount * static_cast<int64_t>(STREAMING_SCORE_TILE_WIDTH)
        : -1;
    const int64_t minScoreWorkspaceWords = alignedSource > 0 &&
        alignedSource <= INT64_MAX /
            (STREAMING_A5_MAX_AIC_CORE_COUNT * 4)
        ? (STREAMING_A5_MAX_AIC_CORE_COUNT * 4 * alignedSource + 1) / 2
        : -1;
    const int64_t queueCount = (leases->GetDimNum() == 1) ? leases->GetDim(0) : -1;
    const int64_t privateCapacity = (ready->GetDimNum() == 2) ? ready->GetDim(1) : -1;
    const bool validTransactionInterface =
        rowCount > 0 && sourceCapacity > 0 && queueCount > 0 && privateCapacity > 0 &&
        tileCount > 0 && rowCount <= INT64_MAX / tileCount &&
        resident->GetDimNum() == 1 && resident->GetDim(0) == rowCount &&
        residentValid->GetDimNum() == 1 && residentValid->GetDim(0) == rowCount &&
        poolEntries->GetDimNum() == 1 && poolEntries->GetDim(0) == rowCount &&
        kvObjectLens->GetDimNum() == 1 && kvObjectLens->GetDim(0) == rowCount &&
        kvObjectMap->GetDimNum() == 2 && kvObjectMap->GetDim(0) > 0 &&
        sourceToSlot->GetDimNum() == 2 && sourceToSlot->GetDim(0) == rowCount &&
        sourceToSlot->GetDim(1) == sourceCapacity &&
        kvObjectSlots->GetDimNum() == 2 && kvObjectSlots->GetDim(0) == kvObjectMap->GetDim(0) &&
        kvObjectSlots->GetDim(1) > 0 &&
        ready->GetDimNum() == 2 && ready->GetDim(0) == queueCount &&
        arenaHead->GetDimNum() == 1 && arenaHead->GetDim(0) == queueCount &&
        privatePackedOutput->GetDimNum() == 2 && privatePackedOutput->GetDim(0) > 0 &&
        privatePackedOutput->GetDim(1) == STREAMING_PACKED_KV_ROW_BYTES &&
        control->GetDimNum() == 1 && control->GetDim(0) == STREAMING_KVAF_CONTROL_WORDS &&
        leases->GetDimNum() == 1 && leases->GetDim(0) == queueCount &&
        scoreWorkspace->GetDimNum() == 1 &&
        scoreWorkspace->GetDim(0) >= minScoreWorkspaceWords &&
        tileReceipts->GetDimNum() == 2 &&
        tileReceipts->GetDim(0) == rowCount * tileCount &&
        tileReceipts->GetDim(1) == STREAMING_TILE_RECEIPT_WORDS &&
        streamingEpoch->GetDimNum() == 1 && streamingEpoch->GetDim(0) == 1;
    if (!validTransactionInterface) {
        OP_LOGE(context, "streaming transaction tensors do not match the TND/BSND row contract.");
        return ge::GRAPH_FAILED;
    }

    *sourceToSlotOut = *sourceToSlot;
    *readyOut = *ready;
    *arenaHeadOut = *arenaHead;
    *privatePackedOutputOut = *privatePackedOutput;
    *leasesOut = *leases;
    telemetry->SetDimNum(2);
    telemetry->SetDim(0, queueCount);
    telemetry->SetDim(1, STREAMING_TELEMETRY_COLUMNS);
    // This output is filled by the exact ProcessTopK valid-length formula on
    // device.  It is deliberately row-count shaped, matching sparse output
    // rows rather than cache-pool or KV-object capacity.
    scoreSourceLens->SetDimNum(1);
    scoreSourceLens->SetDim(0, rowCount);
    // These are mutation receipts. They alias their graph-static inputs, so
    // the terminal graph can observe them without making the Get sidecar a
    // post-C8 dependency; the sidecar polls tile_receipts concurrently.
    *scoreWorkspaceOut = *scoreWorkspace;
    *tileReceiptsOut = *tileReceipts;
    OP_LOGD(context->GetNodeName(), "A5StreamingQuantLiGet InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeA5StreamingQuantLiGet(
    gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("A5StreamingQuantLiGet", "InferDataTypeContext context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(SPARSE_INDICES_OUTPUT_INDEX, ge::DT_INT32);
    context->SetOutputDataType(SPARSE_VALUES_OUTPUT_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(PRIVATE_SOURCE_TO_SLOT_OUTPUT_INDEX, ge::DT_INT32);
    context->SetOutputDataType(PRIVATE_READY_TABLE_OUTPUT_INDEX, ge::DT_INT32);
    context->SetOutputDataType(PRIVATE_ARENA_HEAD_OUTPUT_INDEX, ge::DT_INT32);
    context->SetOutputDataType(PRIVATE_PACKED_OUTPUT_OUTPUT_INDEX, ge::DT_UINT8);
    context->SetOutputDataType(QUEUE_LEASES_OUTPUT_INDEX, ge::DT_INT32);
    context->SetOutputDataType(TELEMETRY_OUTPUT_INDEX, ge::DT_INT64);
    context->SetOutputDataType(SCORE_SOURCE_LENS_OUTPUT_INDEX, ge::DT_INT32);
    context->SetOutputDataType(SCORE_WORKSPACE_OUTPUT_INDEX, ge::DT_INT32);
    context->SetOutputDataType(TILE_RECEIPTS_OUTPUT_INDEX, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(A5StreamingQuantLiGet)
    .InferShape(InferShapeA5StreamingQuantLiGet)
    .InferDataType(InferDataTypeA5StreamingQuantLiGet);
}  // namespace ops
