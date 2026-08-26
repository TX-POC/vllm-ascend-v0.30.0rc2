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
constexpr uint32_t CACHE_SLOTS_SNAPSHOT_INDEX = 11;
constexpr uint32_t REQ_POOL_ENTRIES_INDEX = 12;
constexpr uint32_t LOGICAL_TOKEN_MAP_INDEX = 13;
constexpr uint32_t PRIVATE_SOURCE_TO_SLOT_INDEX = 14;
constexpr uint32_t PRIVATE_READY_TABLE_INDEX = 15;
constexpr uint32_t PRIVATE_ARENA_HEAD_INDEX = 16;
constexpr uint32_t PRIVATE_PACKED_OUTPUT_INDEX = 17;
constexpr uint32_t QUEUE_CONTROL_INDEX = 18;
constexpr uint32_t QUEUE_LEASES_INDEX = 19;

constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 2;
constexpr uint32_t ATTR_KV_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 4;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 9;
constexpr uint32_t ATTR_KVAF_ROWS_PER_ENTRY_INDEX = 12;
constexpr uint32_t SPARSE_INDICES_OUTPUT_INDEX = 0;
constexpr uint32_t SPARSE_VALUES_OUTPUT_INDEX = 1;
constexpr uint32_t PRIVATE_SOURCE_TO_SLOT_OUTPUT_INDEX = 2;
constexpr uint32_t PRIVATE_READY_TABLE_OUTPUT_INDEX = 3;
constexpr uint32_t PRIVATE_ARENA_HEAD_OUTPUT_INDEX = 4;
constexpr uint32_t PRIVATE_PACKED_OUTPUT_OUTPUT_INDEX = 5;
constexpr uint32_t QUEUE_LEASES_OUTPUT_INDEX = 6;
constexpr uint32_t TELEMETRY_OUTPUT_INDEX = 7;
constexpr uint32_t STREAMING_TELEMETRY_COLUMNS = 13;
constexpr uint32_t STREAMING_KVAF_CONTROL_WORDS = 16;
constexpr uint32_t STREAMING_PACKED_KV_ROW_BYTES = 656;

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
    const gert::Shape *cacheSlots = context->GetInputShape(CACHE_SLOTS_SNAPSHOT_INDEX);
    const gert::Shape *poolEntries = context->GetInputShape(REQ_POOL_ENTRIES_INDEX);
    const gert::Shape *logicalMap = context->GetInputShape(LOGICAL_TOKEN_MAP_INDEX);
    const gert::Shape *sourceToSlot = context->GetInputShape(PRIVATE_SOURCE_TO_SLOT_INDEX);
    const gert::Shape *ready = context->GetInputShape(PRIVATE_READY_TABLE_INDEX);
    const gert::Shape *arenaHead = context->GetInputShape(PRIVATE_ARENA_HEAD_INDEX);
    const gert::Shape *privatePackedOutput = context->GetInputShape(PRIVATE_PACKED_OUTPUT_INDEX);
    const gert::Shape *control = context->GetInputShape(QUEUE_CONTROL_INDEX);
    const gert::Shape *leases = context->GetInputShape(QUEUE_LEASES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, query);
    OP_CHECK_NULL_WITH_CONTEXT(context, key);
    OP_CHECK_NULL_WITH_CONTEXT(context, resident);
    OP_CHECK_NULL_WITH_CONTEXT(context, residentValid);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheSlots);
    OP_CHECK_NULL_WITH_CONTEXT(context, poolEntries);
    OP_CHECK_NULL_WITH_CONTEXT(context, logicalMap);
    OP_CHECK_NULL_WITH_CONTEXT(context, sourceToSlot);
    OP_CHECK_NULL_WITH_CONTEXT(context, ready);
    OP_CHECK_NULL_WITH_CONTEXT(context, arenaHead);
    OP_CHECK_NULL_WITH_CONTEXT(context, privatePackedOutput);
    OP_CHECK_NULL_WITH_CONTEXT(context, control);
    OP_CHECK_NULL_WITH_CONTEXT(context, leases);

    gert::Shape *sparseIndices = context->GetOutputShape(SPARSE_INDICES_OUTPUT_INDEX);
    gert::Shape *sparseValues = context->GetOutputShape(SPARSE_VALUES_OUTPUT_INDEX);
    gert::Shape *sourceToSlotOut = context->GetOutputShape(PRIVATE_SOURCE_TO_SLOT_OUTPUT_INDEX);
    gert::Shape *readyOut = context->GetOutputShape(PRIVATE_READY_TABLE_OUTPUT_INDEX);
    gert::Shape *arenaHeadOut = context->GetOutputShape(PRIVATE_ARENA_HEAD_OUTPUT_INDEX);
    gert::Shape *privatePackedOutputOut = context->GetOutputShape(PRIVATE_PACKED_OUTPUT_OUTPUT_INDEX);
    gert::Shape *leasesOut = context->GetOutputShape(QUEUE_LEASES_OUTPUT_INDEX);
    gert::Shape *telemetry = context->GetOutputShape(TELEMETRY_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseIndices);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseValues);
    OP_CHECK_NULL_WITH_CONTEXT(context, sourceToSlotOut);
    OP_CHECK_NULL_WITH_CONTEXT(context, readyOut);
    OP_CHECK_NULL_WITH_CONTEXT(context, arenaHeadOut);
    OP_CHECK_NULL_WITH_CONTEXT(context, privatePackedOutputOut);
    OP_CHECK_NULL_WITH_CONTEXT(context, leasesOut);
    OP_CHECK_NULL_WITH_CONTEXT(context, telemetry);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *queryLayoutAttr = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    const char *keyLayoutAttr = attrs->GetAttrPointer<char>(ATTR_KV_LAYOUT_INDEX);
    const int64_t *sparseCount = attrs->GetInt(ATTR_SPARSE_COUNT_INDEX);
    const int64_t *rowsPerEntry = attrs->GetAttrPointer<int64_t>(ATTR_KVAF_ROWS_PER_ENTRY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryLayoutAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyLayoutAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseCount);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowsPerEntry);

    const std::string queryLayout(queryLayoutAttr);
    const std::string keyLayout(keyLayoutAttr);
    if ((queryLayout != "TND" && queryLayout != "BSND") || *sparseCount <= 0 ||
        *rowsPerEntry <= 0) {
        OP_LOGE(context, "layout_query must be TND or BSND; sparse_count and kvaf_rows_per_entry must be positive.");
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
    const int64_t sourceCapacity = (logicalMap->GetDimNum() == 2) ? logicalMap->GetDim(1) : -1;
    const int64_t queueCount = (leases->GetDimNum() == 1) ? leases->GetDim(0) : -1;
    const int64_t privateCapacity = (ready->GetDimNum() == 2) ? ready->GetDim(1) : -1;
    const bool validTransactionInterface =
        rowCount > 0 && sourceCapacity > 0 && queueCount > 0 && privateCapacity > 0 &&
        resident->GetDimNum() == 1 && resident->GetDim(0) == rowCount &&
        residentValid->GetDimNum() == 1 && residentValid->GetDim(0) == rowCount &&
        poolEntries->GetDimNum() == 1 && poolEntries->GetDim(0) == rowCount &&
        logicalMap->GetDimNum() == 2 && logicalMap->GetDim(0) > 0 &&
        sourceToSlot->GetDimNum() == 2 && sourceToSlot->GetDim(0) == rowCount &&
        sourceToSlot->GetDim(1) == sourceCapacity &&
        cacheSlots->GetDimNum() == 2 && cacheSlots->GetDim(0) == logicalMap->GetDim(0) &&
        cacheSlots->GetDim(1) > 0 &&
        ready->GetDimNum() == 2 && ready->GetDim(0) == queueCount &&
        arenaHead->GetDimNum() == 1 && arenaHead->GetDim(0) == queueCount &&
        privatePackedOutput->GetDimNum() == 2 && privatePackedOutput->GetDim(0) > 0 &&
        privatePackedOutput->GetDim(1) == STREAMING_PACKED_KV_ROW_BYTES &&
        control->GetDimNum() == 1 && control->GetDim(0) == STREAMING_KVAF_CONTROL_WORDS &&
        leases->GetDimNum() == 1 && leases->GetDim(0) == queueCount;
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
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(A5StreamingQuantLiGet)
    .InferShape(InferShapeA5StreamingQuantLiGet)
    .InferDataType(InferDataTypeA5StreamingQuantLiGet);
}  // namespace ops
