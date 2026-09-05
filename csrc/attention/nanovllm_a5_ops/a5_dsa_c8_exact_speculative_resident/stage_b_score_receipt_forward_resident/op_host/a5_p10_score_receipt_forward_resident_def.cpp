// P10 Stage-B host registration, shape gate, and logical-worker tiling.
// Stage B is launched from cold-bound graph aliases before Stage A publishes
// receipts. Receipt state and producer seal, not a Stage-A completion edge,
// carry synchronization.

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

#include "../op_kernel/a5_p10_score_receipt_forward_resident_abi.h"
#include "../op_kernel/a5_p10_score_receipt_forward_resident_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
enum InputIndex : size_t {
  SCORE_SURFACE = 0, RECEIPT_RING, RECEIPT_STATE, FORWARD_CONTROL,
  ACTIVE_ROW_POOL_ENTRIES, KV_OBJECT_MAP, KV_OBJECT_SLOTS_SNAPSHOT,
  KV_OBJECT_TO_FORWARD_SLOT, SOURCE_TO_FORWARD_SLOT, SPECULATIVE_SLOT_BAND,
  SPECULATIVE_SLOT_CURSOR, RECALL_SLOT_BAND, CANONICAL_SPARSE_INDICES,
  CANONICAL_SCORE_SOURCE_LENS, KVAF_FORWARD_RESIDENT_CONTROL, QUEUE_LEASES,
  RESIDENT_PAYLOAD, RESIDENT_READY, RESIDENT_ROW_TO_QSFA_SLOT,
  ATTENTION_SLOTS_BUFFER, SELECTED_SLOT_LENGTHS_BUFFER,
  PREDICTED_SCORE_CUTOFF_U16,
};
enum OutputIndex : size_t {
  RECEIPT_STATE_ALIAS = 0, FORWARD_CONTROL_ALIAS,
  KV_OBJECT_TO_FORWARD_SLOT_ALIAS, SOURCE_TO_FORWARD_SLOT_ALIAS,
  SPECULATIVE_SLOT_CURSOR_ALIAS, QUEUE_LEASES_ALIAS, RESIDENT_PAYLOAD_ALIAS,
  RESIDENT_READY_ALIAS, ATTENTION_SLOTS, SELECTED_SLOT_LENGTHS,
  RESIDENT_SEQ_LENGTHS,
};
enum AttrIndex : size_t {
  NUM_LAYERS = 0, LAYER_ID, EXECUTOR_WORKER_COUNT, EXECUTOR_WORKER_LIMIT,
  RECEIPT_POLL_QUANTUM, BOUNDED_DRAIN_POLLS,
};
constexpr int64_t kRowsPerObject = 128;
constexpr int64_t kSparseCount = A5_P10_SPARSE_COUNT;
constexpr int64_t kReceiptWords = A5_P10_RECEIPT_WORDS64;
constexpr int64_t kForwardControlWords = A5_P10_STAGE_B_CONTROL_WORDS;
constexpr int64_t kKvafControlMinWords = 16;

bool IsShape(const gert::Shape& shape, std::initializer_list<int64_t> dims) {
  if (shape.GetDimNum() != dims.size()) return false;
  size_t index = 0;
  for (const int64_t expected : dims) {
    if (expected >= 0 && shape.GetDim(index) != expected) return false;
    ++index;
  }
  return true;
}
bool HasInput(gert::TilingContext* context, size_t index) {
  return context->GetInputShape(index) != nullptr &&
      context->GetInputDesc(index) != nullptr;
}
bool HasOutput(gert::TilingContext* context, size_t index) {
  return context->GetOutputShape(index) != nullptr &&
      context->GetOutputDesc(index) != nullptr;
}
bool IsInt32(const gert::TilingContext* context, size_t index) {
  return context->GetInputDesc(index)->GetDataType() == ge::DT_INT32;
}
}  // namespace

namespace optiling {
// This selects bounded logical graph workers, never physical AIV IDs. Both
// worker attributes must be strictly below available AIV capacity, leaving
// capacity for the simultaneous native W4A4C8 C8 producer.
static ge::graphStatus TilingA5P10ScoreReceiptForwardResident(
    gert::TilingContext* context) {
  if (context == nullptr || context->GetPlatformInfo() == nullptr) {
    return ge::GRAPH_FAILED;
  }
  for (size_t index = SCORE_SURFACE;
       index <= PREDICTED_SCORE_CUTOFF_U16; ++index) {
    if (!HasInput(context, index)) return ge::GRAPH_FAILED;
  }
  for (size_t index = RECEIPT_STATE_ALIAS;
       index <= RESIDENT_SEQ_LENGTHS; ++index) {
    if (!HasOutput(context, index)) return ge::GRAPH_FAILED;
  }
  const auto dtype = [context](size_t index) {
    return context->GetInputDesc(index)->GetDataType();
  };
  if (dtype(SCORE_SURFACE) != ge::DT_UINT16 ||
      dtype(PREDICTED_SCORE_CUTOFF_U16) != ge::DT_UINT16 ||
      dtype(RECEIPT_RING) != ge::DT_INT64 ||
      dtype(KVAF_FORWARD_RESIDENT_CONTROL) != ge::DT_INT64 ||
      dtype(RESIDENT_PAYLOAD) != ge::DT_UINT8) return ge::GRAPH_FAILED;
  for (size_t index : {RECEIPT_STATE, FORWARD_CONTROL,
                       ACTIVE_ROW_POOL_ENTRIES, KV_OBJECT_MAP,
                       KV_OBJECT_SLOTS_SNAPSHOT, KV_OBJECT_TO_FORWARD_SLOT,
                       SOURCE_TO_FORWARD_SLOT, SPECULATIVE_SLOT_BAND,
                       SPECULATIVE_SLOT_CURSOR, RECALL_SLOT_BAND,
                       CANONICAL_SPARSE_INDICES, CANONICAL_SCORE_SOURCE_LENS,
                       QUEUE_LEASES, RESIDENT_READY,
                       RESIDENT_ROW_TO_QSFA_SLOT, ATTENTION_SLOTS_BUFFER,
                       SELECTED_SLOT_LENGTHS_BUFFER}) {
    if (!IsInt32(context, index)) return ge::GRAPH_FAILED;
  }

  const gert::Shape score = context->GetInputShape(SCORE_SURFACE)->GetStorageShape();
  const gert::Shape ring = context->GetInputShape(RECEIPT_RING)->GetStorageShape();
  const gert::Shape state = context->GetInputShape(RECEIPT_STATE)->GetStorageShape();
  const gert::Shape control = context->GetInputShape(FORWARD_CONTROL)->GetStorageShape();
  const gert::Shape activeRows = context->GetInputShape(ACTIVE_ROW_POOL_ENTRIES)->GetStorageShape();
  const gert::Shape objectMap = context->GetInputShape(KV_OBJECT_MAP)->GetStorageShape();
  const gert::Shape objectSnapshot = context->GetInputShape(KV_OBJECT_SLOTS_SNAPSHOT)->GetStorageShape();
  const gert::Shape objectForward = context->GetInputShape(KV_OBJECT_TO_FORWARD_SLOT)->GetStorageShape();
  const gert::Shape sourceForward = context->GetInputShape(SOURCE_TO_FORWARD_SLOT)->GetStorageShape();
  const gert::Shape speculativeBand = context->GetInputShape(SPECULATIVE_SLOT_BAND)->GetStorageShape();
  const gert::Shape speculativeCursor = context->GetInputShape(SPECULATIVE_SLOT_CURSOR)->GetStorageShape();
  const gert::Shape recallBand = context->GetInputShape(RECALL_SLOT_BAND)->GetStorageShape();
  const gert::Shape canonicalIndices = context->GetInputShape(CANONICAL_SPARSE_INDICES)->GetStorageShape();
  const gert::Shape canonicalLens = context->GetInputShape(CANONICAL_SCORE_SOURCE_LENS)->GetStorageShape();
  const gert::Shape kvafControl = context->GetInputShape(KVAF_FORWARD_RESIDENT_CONTROL)->GetStorageShape();
  const gert::Shape queueLeases = context->GetInputShape(QUEUE_LEASES)->GetStorageShape();
  const gert::Shape residentPayload = context->GetInputShape(RESIDENT_PAYLOAD)->GetStorageShape();
  const gert::Shape residentReady = context->GetInputShape(RESIDENT_READY)->GetStorageShape();
  const gert::Shape residentToQsfa = context->GetInputShape(RESIDENT_ROW_TO_QSFA_SLOT)->GetStorageShape();
  const gert::Shape attention = context->GetInputShape(ATTENTION_SLOTS_BUFFER)->GetStorageShape();
  const gert::Shape selectedLengths = context->GetInputShape(SELECTED_SLOT_LENGTHS_BUFFER)->GetStorageShape();
  const gert::Shape predictedScoreCutoff = context->GetInputShape(PREDICTED_SCORE_CUTOFF_U16)->GetStorageShape();
  if (!IsShape(score, {-1, -1, -1}) || !IsShape(ring, {-1, kReceiptWords}) ||
      !IsShape(state, {-1}) || !IsShape(control, {-1}) ||
      !IsShape(activeRows, {-1}) || !IsShape(objectMap, {-1, -1}) ||
      !IsShape(objectSnapshot, {-1, -1}) || !IsShape(objectForward, {-1, -1}) ||
      !IsShape(sourceForward, {-1, -1}) ||
      !IsShape(speculativeBand, {A5_P10_SLOT_BAND_WORDS}) ||
      !IsShape(speculativeCursor, {1}) ||
      !IsShape(recallBand, {A5_P10_SLOT_BAND_WORDS}) ||
      !IsShape(canonicalIndices, {-1, 1, kSparseCount}) ||
      !IsShape(canonicalLens, {-1}) || !IsShape(kvafControl, {-1}) ||
      !IsShape(queueLeases, {-1}) || !IsShape(residentPayload, {-1, -1}) ||
      !IsShape(residentReady, {-1}) || !IsShape(residentToQsfa, {-1}) ||
      !IsShape(attention, {-1, 1, -1}) || !IsShape(selectedLengths, {-1})) {
    return ge::GRAPH_FAILED;
  }
  const int64_t batch = activeRows.GetDim(0);
  const int64_t poolCapacity = objectMap.GetDim(0);
  const int64_t sourceCapacity = objectMap.GetDim(1);
  const int64_t receiptCapacity = ring.GetDim(0);
  const int64_t residentRows = residentPayload.GetDim(0);
  if (batch <= 0 || poolCapacity <= 0 || sourceCapacity <= 0 ||
      sourceCapacity % kRowsPerObject != 0 || receiptCapacity <= 0 ||
      residentRows <= 0 || residentPayload.GetDim(1) <= 0 ||
      score.GetDim(0) != batch || score.GetDim(2) != sourceCapacity ||
      state.GetDim(0) != receiptCapacity ||
      control.GetDim(0) < kForwardControlWords ||
      objectSnapshot.GetDim(0) != poolCapacity ||
      objectSnapshot.GetDim(1) != sourceCapacity ||
      objectForward.GetDim(0) != poolCapacity ||
      objectForward.GetDim(1) != sourceCapacity ||
      sourceForward.GetDim(0) != batch ||
      sourceForward.GetDim(1) != sourceCapacity ||
      canonicalIndices.GetDim(0) != batch || canonicalLens.GetDim(0) != batch ||
      kvafControl.GetDim(0) < kKvafControlMinWords || queueLeases.GetDim(0) <= 0 ||
      residentReady.GetDim(0) != residentRows ||
      residentToQsfa.GetDim(0) != residentRows || attention.GetDim(0) != batch ||
      attention.GetDim(2) < kSparseCount || selectedLengths.GetDim(0) != batch ||
      !IsShape(predictedScoreCutoff, {-1}) ||
      predictedScoreCutoff.GetDim(0) != batch) {
    return ge::GRAPH_FAILED;
  }
  if (batch > std::numeric_limits<uint32_t>::max() ||
      poolCapacity > std::numeric_limits<uint32_t>::max() ||
      sourceCapacity > std::numeric_limits<uint32_t>::max() ||
      receiptCapacity > std::numeric_limits<uint32_t>::max() ||
      residentRows > std::numeric_limits<uint32_t>::max() ||
      attention.GetDim(2) > std::numeric_limits<uint32_t>::max()) {
    return ge::GRAPH_FAILED;
  }
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  if (attrs == nullptr) return ge::GRAPH_FAILED;
  const int64_t* numLayers = attrs->GetAttrPointer<int64_t>(NUM_LAYERS);
  const int64_t* layerId = attrs->GetAttrPointer<int64_t>(LAYER_ID);
  const int64_t* workerCount = attrs->GetAttrPointer<int64_t>(EXECUTOR_WORKER_COUNT);
  const int64_t* workerLimit = attrs->GetAttrPointer<int64_t>(EXECUTOR_WORKER_LIMIT);
  const int64_t* pollQuantum = attrs->GetAttrPointer<int64_t>(RECEIPT_POLL_QUANTUM);
  const int64_t* drainPolls = attrs->GetAttrPointer<int64_t>(BOUNDED_DRAIN_POLLS);
  if (numLayers == nullptr || layerId == nullptr || workerCount == nullptr ||
      workerLimit == nullptr || pollQuantum == nullptr || drainPolls == nullptr ||
      *numLayers <= 0 || *layerId < 0 || *layerId >= *numLayers ||
      *workerCount <= 0 || *workerLimit <= 0 || *pollQuantum <= 0 ||
      *drainPolls <= 0 || *workerCount > *workerLimit ||
      *numLayers > std::numeric_limits<uint32_t>::max() ||
      *layerId > std::numeric_limits<uint32_t>::max() ||
      *workerCount > std::numeric_limits<uint32_t>::max() ||
      *workerLimit > std::numeric_limits<uint32_t>::max() ||
      *pollQuantum > std::numeric_limits<uint32_t>::max() ||
      *drainPolls > std::numeric_limits<uint32_t>::max()) return ge::GRAPH_FAILED;
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  const uint32_t aivCount = platform.GetCoreNumAiv();
  if (aivCount < 2U ||
      static_cast<uint64_t>(*workerCount) >= aivCount ||
      static_cast<uint64_t>(*workerLimit) >= aivCount) return ge::GRAPH_FAILED;

  // Stage B is a SIMT/AIV Direct-HCCP executor.  It has no Cube/AIC data
  // plane, so a MIX_AIC_1_2 launch gives GetBlockIdx() mixed-lane semantics
  // and can silently drop an executor worker.  AIV_ONLY keeps the logical
  // worker id exact (0..workerCount-1); it is not a physical-core assignment,
  // because each worker dynamically leases any queue.
  const uint32_t blockDim = platform.CalcTschBlockDim(
      static_cast<uint32_t>(*workerCount), 0U, aivCount);
  if (blockDim != static_cast<uint32_t>(*workerCount)) {
    return ge::GRAPH_FAILED;
  }
  if (context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  context->SetBlockDim(blockDim);

  // CANN 9.1 direct ACLNN materializes typed tiling only after its AIV-only
  // launch geometry is declared.  Keep this order aligned with the proven
  // Direct-HCCP executor path; otherwise failures surface as opaque launch
  // behavior rather than a tiling rejection.
  auto* tiling = context->GetTilingData<A5P10ScoreReceiptForwardResidentTilingData>();
  if (tiling == nullptr || context->GetWorkspaceSizes(1) == nullptr) return ge::GRAPH_FAILED;
  tiling->executorWorkerCount = static_cast<uint32_t>(*workerCount);
  tiling->executorWorkerLimit = static_cast<uint32_t>(*workerLimit);
  tiling->receiptPollQuantum = static_cast<uint32_t>(*pollQuantum);
  tiling->activeRowCount = static_cast<uint32_t>(batch);
  tiling->receiptCapacity = static_cast<uint32_t>(receiptCapacity);
  tiling->sourceCapacity = static_cast<uint32_t>(sourceCapacity);
  tiling->numLayers = static_cast<uint32_t>(*numLayers);
  tiling->layerId = static_cast<uint32_t>(*layerId);
  tiling->qcap = static_cast<uint32_t>(sourceCapacity / kRowsPerObject);
  tiling->poolCapacity = static_cast<uint32_t>(poolCapacity);
  tiling->kvObjectCapacity = static_cast<uint32_t>(sourceCapacity);
  tiling->residentRowCount = static_cast<uint32_t>(residentRows);
  tiling->attentionSlotCapacity = static_cast<uint32_t>(attention.GetDim(2));
  tiling->sparseCount = static_cast<uint32_t>(kSparseCount);
  // predicted_score_cutoff_u16 is the immutable graph input authoritative for
  // ExactScoreScan; forward_control carries only trace diagnostics.
  tiling->boundedDrainPolls = static_cast<uint32_t>(*drainPolls);
  tiling->reserved = 0U;
  context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
static ge::graphStatus InferA5P10ScoreReceiptForwardResidentShape(
    gert::InferShapeContext* context) {
  if (context == nullptr) return ge::GRAPH_FAILED;
  constexpr size_t kInputForOutput[] = {
      RECEIPT_STATE, FORWARD_CONTROL, KV_OBJECT_TO_FORWARD_SLOT,
      SOURCE_TO_FORWARD_SLOT, SPECULATIVE_SLOT_CURSOR, QUEUE_LEASES,
      RESIDENT_PAYLOAD, RESIDENT_READY, ATTENTION_SLOTS_BUFFER,
      SELECTED_SLOT_LENGTHS_BUFFER, SELECTED_SLOT_LENGTHS_BUFFER};
  for (size_t output = RECEIPT_STATE_ALIAS; output <= RESIDENT_SEQ_LENGTHS;
       ++output) {
    if (context->GetInputShape(kInputForOutput[output]) == nullptr ||
        context->GetOutputShape(output) == nullptr) return ge::GRAPH_FAILED;
    *context->GetOutputShape(output) =
        *context->GetInputShape(kInputForOutput[output]);
  }
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferA5P10ScoreReceiptForwardResidentDataType(
    gert::InferDataTypeContext* context) {
  if (context == nullptr) return ge::GRAPH_FAILED;
  context->SetOutputDataType(RECEIPT_STATE_ALIAS, ge::DT_INT32);
  context->SetOutputDataType(FORWARD_CONTROL_ALIAS, ge::DT_INT32);
  context->SetOutputDataType(KV_OBJECT_TO_FORWARD_SLOT_ALIAS, ge::DT_INT32);
  context->SetOutputDataType(SOURCE_TO_FORWARD_SLOT_ALIAS, ge::DT_INT32);
  context->SetOutputDataType(SPECULATIVE_SLOT_CURSOR_ALIAS, ge::DT_INT32);
  context->SetOutputDataType(QUEUE_LEASES_ALIAS, ge::DT_INT32);
  context->SetOutputDataType(RESIDENT_PAYLOAD_ALIAS, ge::DT_UINT8);
  context->SetOutputDataType(RESIDENT_READY_ALIAS, ge::DT_INT32);
  context->SetOutputDataType(ATTENTION_SLOTS, ge::DT_INT32);
  context->SetOutputDataType(SELECTED_SLOT_LENGTHS, ge::DT_INT32);
  context->SetOutputDataType(RESIDENT_SEQ_LENGTHS, ge::DT_INT32);
  return ge::GRAPH_SUCCESS;
}

class A5P10ScoreReceiptForwardResident : public OpDef {
 public:
  explicit A5P10ScoreReceiptForwardResident(const char* name) : OpDef(name) {
    const std::vector<ge::Format> formats = {ge::FORMAT_ND};
    const std::vector<ge::DataType> int32s = {ge::DT_INT32};
    const std::vector<ge::DataType> int64s = {ge::DT_INT64};
    const std::vector<ge::DataType> uint8s = {ge::DT_UINT8};
    const std::vector<ge::DataType> uint16s = {ge::DT_UINT16};
    this->Input("score_surface").ParamType(REQUIRED).DataType(uint16s).Format(formats);
    this->Input("receipt_ring").ParamType(REQUIRED).DataType(int64s).Format(formats);
    this->Input("receipt_state").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("forward_control").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("active_row_pool_entries").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("kv_object_map").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("kv_object_slots_snapshot").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("kv_object_to_forward_slot").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("source_to_forward_slot").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("speculative_slot_band").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("speculative_slot_cursor").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("recall_slot_band").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("canonical_sparse_indices").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("canonical_score_source_lens").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("kvaf_forward_resident_control").ParamType(REQUIRED).DataType(int64s).Format(formats);
    this->Input("queue_leases").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("resident_payload").ParamType(REQUIRED).DataType(uint8s).Format(formats);
    this->Input("resident_ready").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("resident_row_to_qsfa_slot").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("attention_slots_buffer").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("selected_slot_lengths_buffer").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Input("predicted_score_cutoff_u16").ParamType(REQUIRED).DataType(uint16s).Format(formats);
    this->Output("receipt_state_alias").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Output("forward_control_alias").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Output("kv_object_to_forward_slot_alias").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Output("source_to_forward_slot_alias").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Output("speculative_slot_cursor_alias").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Output("queue_leases_alias").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Output("resident_payload_alias").ParamType(REQUIRED).DataType(uint8s).Format(formats);
    this->Output("resident_ready_alias").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Output("attention_slots").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Output("selected_slot_lengths").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Output("resident_seq_lengths").ParamType(REQUIRED).DataType(int32s).Format(formats);
    this->Attr("num_layers").AttrType(REQUIRED).Int(1);
    this->Attr("layer_id").AttrType(REQUIRED).Int(0);
    this->Attr("executor_worker_count").AttrType(REQUIRED).Int(1);
    this->Attr("executor_worker_limit").AttrType(REQUIRED).Int(1);
    this->Attr("receipt_poll_quantum").AttrType(REQUIRED).Int(1);
    this->Attr("bounded_drain_polls").AttrType(REQUIRED).Int(A5_P10_STAGE_B_FINAL_DRAIN_POLLS);
    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false).PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("opFile.value", "a5_p10_score_receipt_forward_resident")
        .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
    this->AICore().SetTiling(optiling::TilingA5P10ScoreReceiptForwardResident)
        .AddConfig("ascend950", config);
  }
};
OP_ADD(A5P10ScoreReceiptForwardResident);
IMPL_OP_INFERSHAPE(A5P10ScoreReceiptForwardResident)
    .InferShape(InferA5P10ScoreReceiptForwardResidentShape)
    .InferDataType(InferA5P10ScoreReceiptForwardResidentDataType);
}  // namespace ops
