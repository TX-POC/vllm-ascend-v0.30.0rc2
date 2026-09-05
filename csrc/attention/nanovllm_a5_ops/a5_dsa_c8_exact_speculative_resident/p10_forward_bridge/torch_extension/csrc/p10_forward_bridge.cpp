// P10 graph bridge: one preparation node and two independently launched
// device nodes.  This file has no candidate-loop, KVAF submission, physical
// core identifier, or host completion wait.  The generated P10 product OPP
// owns the device data plane; this bridge only converts the three graph nodes
// to ACLNN calls on the caller-selected NPU streams.

#include <array>
#include <tuple>
#include <utility>

#include <torch/extension.h>

// A P10 product assembly combines the prepare, score-receipt, and
// forward-resident operators into one source-provenanced custom OPP.  The
// generic launcher resolves all three symbols from this one DSO.  This avoids
// accidental resolution from an older separately-built package.
#define NANOVLLM_A5_OPAPI_ENV "NANOVLLM_A5_P10_PRODUCT_OPAPI_LIB"
#include "p10_direct_opapi.h"

namespace {

constexpr int64_t kPrepareInputCount = 7;
constexpr int64_t kStageAInputCount = 14;
constexpr int64_t kStageBInputCount = 22;
constexpr int64_t kForwardControlWords = 10;
constexpr int64_t kSparseCount = 2048;
constexpr int64_t kPackedKvRowBytes = 656;

void RequireSameNpu(const at::Tensor& anchor, const at::Tensor& value,
                    at::ScalarType dtype, const char* name,
                    bool contiguous = true)
{
    TORCH_CHECK(anchor.device().is_privateuseone(),
                "P10 forward bridge requires NPU tensors.");
    TORCH_CHECK(value.defined() && value.device() == anchor.device(), name,
                " must be on the same NPU as the graph anchor.");
    TORCH_CHECK(value.scalar_type() == dtype, name, " has unexpected dtype.");
    if (contiguous) {
        TORCH_CHECK(value.is_contiguous(), name, " must be contiguous.");
    }
}

void RequireFlatOrMatrix(const at::Tensor& value, const char* name)
{
    TORCH_CHECK(value.dim() >= 1 && value.numel() > 0, name,
                " must be a nonempty graph alias.");
}

void ValidatePrepare(const at::Tensor& receipt_state,
                     const at::Tensor& forward_control,
                     const at::Tensor& object_to_forward_slot,
                     const at::Tensor& source_to_forward_slot,
                     const at::Tensor& speculative_slot_cursor,
                     const at::Tensor& queue_leases,
                     const at::Tensor& resident_ready)
{
    RequireSameNpu(receipt_state, receipt_state, at::kInt, "receipt_state");
    for (const auto& item : {
             std::pair<const at::Tensor*, const char*>{&forward_control,
                                                        "forward_control"},
             {&object_to_forward_slot, "kv_object_to_forward_slot"},
             {&source_to_forward_slot, "source_to_forward_slot"},
             {&speculative_slot_cursor, "speculative_slot_cursor"},
             {&queue_leases, "queue_leases"},
             {&resident_ready, "resident_ready"},
         }) {
        RequireSameNpu(receipt_state, *item.first, at::kInt, item.second);
        RequireFlatOrMatrix(*item.first, item.second);
    }
    TORCH_CHECK(receipt_state.dim() == 1, "receipt_state must be int32[R].");
    TORCH_CHECK(forward_control.dim() == 1 &&
                    forward_control.numel() == kForwardControlWords,
                "forward_control must be int32[10], including P10's minimal "
                "producer-seal / preseal-ready overlap witness.");
    TORCH_CHECK(object_to_forward_slot.dim() == 2,
                "kv_object_to_forward_slot must be int32[P,S].");
    TORCH_CHECK(source_to_forward_slot.dim() == 2,
                "source_to_forward_slot must be int32[B,S].");
    TORCH_CHECK(speculative_slot_cursor.dim() == 1 &&
                    speculative_slot_cursor.numel() == 1,
                "speculative_slot_cursor must be int32[1].");
}

void ValidateStageA(
    const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& weights, const at::Tensor& query_dequant_scale,
    const at::Tensor& key_dequant_scale,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& candidate_lens, const at::Tensor& block_table,
    const at::Tensor& canonical_sparse_indices,
    const at::Tensor& canonical_score_source_lens,
    const at::Tensor& score_surface, const at::Tensor& receipt_ring,
    const at::Tensor& receipt_state, const at::Tensor& forward_control)
{
    RequireSameNpu(query, query, at::kFloat8_e4m3fn, "query");
    RequireSameNpu(query, key, at::kFloat8_e4m3fn, "key", false);
    RequireSameNpu(query, weights, at::kBFloat16, "weights");
    RequireSameNpu(query, query_dequant_scale, at::kFloat,
                   "query_dequant_scale");
    RequireSameNpu(query, key_dequant_scale, at::kFloat,
                   "key_dequant_scale");
    for (const auto& item : {
             std::pair<const at::Tensor*, const char*>{
                 &actual_seq_lengths_query, "actual_seq_lengths_query"},
             {&candidate_lens, "candidate_lens"},
             {&block_table, "block_table"},
             {&canonical_sparse_indices, "canonical_sparse_indices"},
             {&canonical_score_source_lens, "canonical_score_source_lens"},
             {&receipt_state, "receipt_state"},
             {&forward_control, "forward_control"},
         }) {
        RequireSameNpu(query, *item.first, at::kInt, item.second);
    }
    RequireSameNpu(query, score_surface, at::kUInt16, "score_surface");
    RequireSameNpu(query, receipt_ring, at::kLong, "receipt_ring");
    TORCH_CHECK(canonical_sparse_indices.dim() == 3 &&
                    canonical_sparse_indices.size(1) == 1 &&
                    canonical_sparse_indices.size(2) == kSparseCount,
                "canonical_sparse_indices must be int32[B,1,2048].");
    TORCH_CHECK(canonical_score_source_lens.dim() == 1 &&
                    canonical_score_source_lens.numel() == query.size(0),
                "canonical_score_source_lens must be int32[B].");
    TORCH_CHECK(score_surface.dim() == 3 && score_surface.size(0) == query.size(0),
                "score_surface must be uint16[B,S1,source].");
    TORCH_CHECK(receipt_ring.dim() == 2 && receipt_ring.size(1) == 5,
                "receipt_ring must be int64[R,5].");
}

void ValidateStageB(
    const at::Tensor& score_surface, const at::Tensor& receipt_ring,
    const at::Tensor& receipt_state, const at::Tensor& forward_control,
    const at::Tensor& active_row_pool_entries, const at::Tensor& kv_object_map,
    const at::Tensor& kv_object_slots_snapshot,
    const at::Tensor& kv_object_to_forward_slot,
    const at::Tensor& source_to_forward_slot,
    const at::Tensor& speculative_slot_band,
    const at::Tensor& speculative_slot_cursor,
    const at::Tensor& recall_slot_band,
    const at::Tensor& canonical_sparse_indices,
    const at::Tensor& canonical_score_source_lens,
    const at::Tensor& kvaf_forward_resident_control,
    const at::Tensor& queue_leases, const at::Tensor& resident_payload,
    const at::Tensor& resident_ready,
    const at::Tensor& resident_row_to_qsfa_slot,
    const at::Tensor& attention_slots,
    const at::Tensor& selected_slot_lengths,
    const at::Tensor& predicted_score_cutoff_u16,
    int64_t num_layers, int64_t layer_id,
    int64_t executor_worker_count, int64_t executor_worker_limit,
    int64_t receipt_poll_quantum, int64_t bounded_drain_polls)
{
    RequireSameNpu(score_surface, score_surface, at::kUInt16, "score_surface");
    for (const auto& item : {
             std::pair<const at::Tensor*, const char*>{&receipt_ring,
                                                        "receipt_ring"},
             {&kvaf_forward_resident_control,
              "kvaf_forward_resident_control"},
         }) {
        RequireSameNpu(score_surface, *item.first, at::kLong, item.second);
    }
    for (const auto& item : {
             std::pair<const at::Tensor*, const char*>{&receipt_state,
                                                        "receipt_state"},
             {&forward_control, "forward_control"},
             {&active_row_pool_entries, "active_row_pool_entries"},
             {&kv_object_map, "kv_object_map"},
             {&kv_object_slots_snapshot, "kv_object_slots_snapshot"},
             {&kv_object_to_forward_slot, "kv_object_to_forward_slot"},
             {&source_to_forward_slot, "source_to_forward_slot"},
             {&speculative_slot_band, "speculative_slot_band"},
             {&speculative_slot_cursor, "speculative_slot_cursor"},
             {&recall_slot_band, "recall_slot_band"},
             {&canonical_sparse_indices, "canonical_sparse_indices"},
             {&canonical_score_source_lens, "canonical_score_source_lens"},
             {&queue_leases, "queue_leases"},
             {&resident_ready, "resident_ready"},
             {&resident_row_to_qsfa_slot, "resident_row_to_qsfa_slot"},
             {&attention_slots, "attention_slots_buffer"},
             {&selected_slot_lengths, "selected_slot_lengths_buffer"},
         }) {
        RequireSameNpu(score_surface, *item.first, at::kInt, item.second);
    }
    RequireSameNpu(score_surface, resident_payload, at::kByte,
                   "resident_payload");
    RequireSameNpu(score_surface, predicted_score_cutoff_u16, at::kUInt16,
                   "predicted_score_cutoff_u16");
    TORCH_CHECK(kv_object_map.dim() == 2 &&
                    kv_object_to_forward_slot.dim() == 2 &&
                    kv_object_to_forward_slot.size(0) == kv_object_map.size(0) &&
                    kv_object_to_forward_slot.size(1) == kv_object_map.size(1),
                "kv_object_to_forward_slot must match kv_object_map[P,S].");
    TORCH_CHECK(source_to_forward_slot.dim() == 2 &&
                    source_to_forward_slot.size(0) == canonical_sparse_indices.size(0) &&
                    source_to_forward_slot.size(1) == kv_object_map.size(1),
                "source_to_forward_slot must be int32[B,S].");
    TORCH_CHECK(speculative_slot_band.numel() == 2 && recall_slot_band.numel() == 2,
                "slot bands must be int32[2].");
    TORCH_CHECK(speculative_slot_cursor.numel() == 1,
                "speculative_slot_cursor must be int32[1].");
    TORCH_CHECK(canonical_sparse_indices.dim() == 3 &&
                    canonical_sparse_indices.size(1) == 1 &&
                    canonical_sparse_indices.size(2) == kSparseCount,
                "canonical_sparse_indices must be int32[B,1,2048].");
    TORCH_CHECK(predicted_score_cutoff_u16.dim() == 1 &&
                    predicted_score_cutoff_u16.numel() ==
                        canonical_sparse_indices.size(0),
                "predicted_score_cutoff_u16 must be uint16[B].");
    TORCH_CHECK(attention_slots.dim() == 3 && attention_slots.size(0) == canonical_sparse_indices.size(0) &&
                    attention_slots.size(1) == 1 && attention_slots.size(2) >= kSparseCount,
                "attention_slots_buffer must be int32[B,1,2048+tail].");
    TORCH_CHECK(resident_payload.dim() == 2 && resident_payload.size(1) == kPackedKvRowBytes,
                "resident_payload must be uint8[resident_rows,656].");
    TORCH_CHECK(resident_row_to_qsfa_slot.dim() == 1 &&
                    resident_row_to_qsfa_slot.numel() == resident_payload.size(0),
                "resident_row_to_qsfa_slot must be int32[resident_rows]. "
                "P10 never assumes resident rows are QSFA logical slot IDs.");
    TORCH_CHECK(resident_ready.dim() == 1 &&
                    resident_ready.numel() == resident_payload.size(0),
                "resident_ready must be int32[resident_rows].");
    TORCH_CHECK(num_layers > 0 && layer_id >= 0 && layer_id < num_layers,
                "P10 Stage-B requires 0 <= layer_id < num_layers.");
    TORCH_CHECK(executor_worker_count > 0 && executor_worker_limit > 0 &&
                    executor_worker_count <= executor_worker_limit,
                "P10 Stage-B executor workers must be a positive bounded "
                "logical count; no physical AIV assignment is accepted.");
    TORCH_CHECK(receipt_poll_quantum > 0 && bounded_drain_polls > 0,
                "P10 Stage-B polling bounds must be positive.");
}

// Preparation is an actual ACLNN graph node.  The seven returned Ref aliases
// are intentionally discarded here: the caller passes the same pre-bound
// tensors to both concurrent stages rather than using a Stage-A alias as a
// Stage-B input.
void PrepareForward(const at::Tensor& receipt_state,
                    const at::Tensor& forward_control,
                    const at::Tensor& kv_object_to_forward_slot,
                    const at::Tensor& source_to_forward_slot,
                    const at::Tensor& speculative_slot_cursor,
                    const at::Tensor& queue_leases,
                    const at::Tensor& resident_ready)
{
    ValidatePrepare(receipt_state, forward_control, kv_object_to_forward_slot,
                    source_to_forward_slot, speculative_slot_cursor,
                    queue_leases, resident_ready);
    const auto keepalive = std::make_tuple(
        receipt_state, forward_control, kv_object_to_forward_slot,
        source_to_forward_slot, speculative_slot_cursor, queue_leases,
        resident_ready);
    P10_EXEC_NPU_CMD_DIRECT(
        aclnnA5P10ForwardPrepare, keepalive, receipt_state, forward_control,
        kv_object_to_forward_slot, source_to_forward_slot,
        speculative_slot_cursor, queue_leases, resident_ready, receipt_state,
        forward_control, kv_object_to_forward_slot, source_to_forward_slot,
        speculative_slot_cursor, queue_leases, resident_ready);
}

// This call is made on the producer stream selected by P10ForwardGraph.  It
// has 14 ABI inputs followed by six graph-visible aliases.  It does not
// create a completion event for the executor.
void LaunchStageA(
    const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& weights, const at::Tensor& query_dequant_scale,
    const at::Tensor& key_dequant_scale,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& candidate_lens, const at::Tensor& block_table,
    const at::Tensor& canonical_sparse_indices,
    const at::Tensor& canonical_score_source_lens,
    const at::Tensor& score_surface, const at::Tensor& receipt_ring,
    const at::Tensor& receipt_state, const at::Tensor& forward_control)
{
    ValidateStageA(query, key, weights, query_dequant_scale, key_dequant_scale,
                   actual_seq_lengths_query, candidate_lens, block_table,
                   canonical_sparse_indices, canonical_score_source_lens,
                   score_surface, receipt_ring, receipt_state, forward_control);
    const auto keepalive = std::make_tuple(
        query, key, weights, query_dequant_scale, key_dequant_scale,
        actual_seq_lengths_query, candidate_lens, block_table,
        canonical_sparse_indices, canonical_score_source_lens, score_surface,
        receipt_ring, receipt_state, forward_control);
    const auto key_arg = nanovllm_a5_opapi::AllowStridedAclTensor(key);
    P10_EXEC_NPU_CMD_DIRECT(
        aclnnA5P10W4A4C8ScoreReceipt, keepalive, query, key_arg, weights,
        query_dequant_scale, key_dequant_scale, actual_seq_lengths_query,
        candidate_lens, block_table, canonical_sparse_indices,
        canonical_score_source_lens, score_surface, receipt_ring,
        receipt_state, forward_control, canonical_sparse_indices,
        canonical_score_source_lens, score_surface, receipt_ring,
        receipt_state, forward_control);
}

// This call is made on the companion stream selected by P10ForwardGraph.  It
// reads the pre-bound aliases directly, polls receipt READY/producer-sealed
// on device, and writes only the existing QSFA aliases at terminal completion.
void LaunchStageB(
    const at::Tensor& score_surface, const at::Tensor& receipt_ring,
    const at::Tensor& receipt_state, const at::Tensor& forward_control,
    const at::Tensor& active_row_pool_entries, const at::Tensor& kv_object_map,
    const at::Tensor& kv_object_slots_snapshot,
    const at::Tensor& kv_object_to_forward_slot,
    const at::Tensor& source_to_forward_slot,
    const at::Tensor& speculative_slot_band,
    const at::Tensor& speculative_slot_cursor,
    const at::Tensor& recall_slot_band,
    const at::Tensor& canonical_sparse_indices,
    const at::Tensor& canonical_score_source_lens,
    const at::Tensor& kvaf_forward_resident_control,
    const at::Tensor& queue_leases, const at::Tensor& resident_payload,
    const at::Tensor& resident_ready,
    const at::Tensor& resident_row_to_qsfa_slot,
    const at::Tensor& attention_slots,
    const at::Tensor& selected_slot_lengths,
    const at::Tensor& predicted_score_cutoff_u16,
    int64_t num_layers, int64_t layer_id,
    int64_t executor_worker_count, int64_t executor_worker_limit,
    int64_t receipt_poll_quantum, int64_t bounded_drain_polls)
{
    ValidateStageB(
        score_surface, receipt_ring, receipt_state, forward_control,
        active_row_pool_entries, kv_object_map, kv_object_slots_snapshot,
        kv_object_to_forward_slot, source_to_forward_slot,
        speculative_slot_band, speculative_slot_cursor, recall_slot_band,
        canonical_sparse_indices, canonical_score_source_lens,
        kvaf_forward_resident_control, queue_leases, resident_payload,
        resident_ready, resident_row_to_qsfa_slot, attention_slots,
        selected_slot_lengths, predicted_score_cutoff_u16,
        num_layers, layer_id, executor_worker_count,
        executor_worker_limit, receipt_poll_quantum, bounded_drain_polls);
    const auto keepalive = std::make_tuple(
        score_surface, receipt_ring, receipt_state, forward_control,
        active_row_pool_entries, kv_object_map, kv_object_slots_snapshot,
        kv_object_to_forward_slot, source_to_forward_slot,
        speculative_slot_band, speculative_slot_cursor, recall_slot_band,
        canonical_sparse_indices, canonical_score_source_lens,
        kvaf_forward_resident_control, queue_leases, resident_payload,
        resident_ready, resident_row_to_qsfa_slot, attention_slots,
        selected_slot_lengths, predicted_score_cutoff_u16);
    P10_EXEC_NPU_CMD_DIRECT(
        aclnnA5P10ScoreReceiptForwardResident, keepalive, score_surface,
        receipt_ring, receipt_state, forward_control, active_row_pool_entries,
        kv_object_map, kv_object_slots_snapshot, kv_object_to_forward_slot,
        source_to_forward_slot, speculative_slot_band,
        speculative_slot_cursor, recall_slot_band, canonical_sparse_indices,
        canonical_score_source_lens, kvaf_forward_resident_control,
        queue_leases, resident_payload, resident_ready,
        resident_row_to_qsfa_slot, attention_slots, selected_slot_lengths,
        predicted_score_cutoff_u16,
        num_layers, layer_id, executor_worker_count, executor_worker_limit,
        receipt_poll_quantum, bounded_drain_polls,
        receipt_state, forward_control,
        kv_object_to_forward_slot, source_to_forward_slot,
        speculative_slot_cursor, queue_leases, resident_payload,
        resident_ready, attention_slots, selected_slot_lengths,
        selected_slot_lengths);
}

// Raw aclmdlRI graphs record the ACLNN workspace address used at capture.  A
// graph segment therefore owns one explicit workspace-retention scope from
// capture through graph destruction.  These helpers are deliberately not a
// forward control path: they carry no device tensors and cannot affect P10's
// C8/receipt/Direct-HCCP math.  They only prevent the host allocator from
// recycling a captured workspace address before the raw graph is reset.
int64_t BeginGraphWorkspaceCapture()
{
    return static_cast<int64_t>(
        nanovllm_a5_opapi::BeginP10GraphWorkspaceCapture());
}

int64_t EndGraphWorkspaceCapture(int64_t token)
{
    TORCH_CHECK(token > 0, "P10 graph workspace token must be positive");
    return nanovllm_a5_opapi::EndP10GraphWorkspaceCapture(
        static_cast<uint64_t>(token));
}

void ReleaseGraphWorkspaceCapture(int64_t token)
{
    TORCH_CHECK(token > 0, "P10 graph workspace token must be positive");
    nanovllm_a5_opapi::ReleaseP10GraphWorkspaceCapture(
        static_cast<uint64_t>(token));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("prepare_forward", &PrepareForward,
          "Enqueue the P10 graph-visible forward preparation node.");
    m.def("launch_stage_a", &LaunchStageA,
          "Enqueue native W4A4C8 score receipt production on the current stream.");
    m.def("launch_stage_b", &LaunchStageB,
          "Enqueue the P10 companion resident executor on the current stream.");
    m.def("begin_graph_workspace_capture", &BeginGraphWorkspaceCapture,
          "Begin one raw ACLGraph workspace-retention scope.");
    m.def("end_graph_workspace_capture", &EndGraphWorkspaceCapture,
          "Seal one raw ACLGraph workspace-retention scope and return its buffer count.");
    m.def("release_graph_workspace_capture", &ReleaseGraphWorkspaceCapture,
          "Release a sealed raw ACLGraph workspace-retention scope after graph reset.");
    m.def("abi_summary", [] {
        pybind11::dict value;
        value["prepare_inputs"] = kPrepareInputCount;
        value["stage_a_inputs"] = kStageAInputCount;
        value["stage_b_inputs"] = kStageBInputCount;
        value["stage_a_alias_outputs"] = 6;
        value["stage_b_alias_outputs"] = 11;
        value["candidate_submission"] = "device_only";
        value["physical_aiv_assignment"] = false;
        value["qsfa_boundary"] = "stage_b_complete_to_attention_slots_and_resident_seq_lengths";
        return value;
    });
}
