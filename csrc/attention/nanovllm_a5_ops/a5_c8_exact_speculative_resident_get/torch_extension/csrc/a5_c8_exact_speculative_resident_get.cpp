// Torch/ACLNN bridge for the atomic P9 C8 + canonical-resident Get ABI.
//
// This is intentionally a *dedicated* resolver and registration domain.  It
// borrows only the mechanical descriptor/launch discipline of
// a5_c8_resident_seed: it does not import that operator, P7, or P8.  The
// matching generated P9 OPP must expose the 21-input / 6-output ACLNN ABI
// below in one atomic host/kernel/bridge change.  In particular, the two
// trailing ACLNN outputs are caller-owned aliases; the public Torch result
// deliberately contains only the four non-alias outputs.

#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/extension.h>
#include <torch/library.h>

#define NANOVLLM_A5_OPAPI_ENV \
    "NANOVLLM_A5_EXACT_SPECULATIVE_RESIDENT_OPAPI_LIB"
#include "op_api_common.h"

namespace {

constexpr int64_t kMetadataWords = 1024;
constexpr int64_t kExactSparseCount = 2048;
constexpr int64_t kA5IndexHeads = 64;
constexpr int64_t kA5KeyHeads = 1;
constexpr int64_t kA5HeadDim = 128;
constexpr int64_t kCanonicalResidentLayoutWords = 4;
constexpr int64_t kLayerResidentGetControlWords = 16;
constexpr int64_t kTerminalPostStatWords = 7;

// The public API intentionally returns no aliases.  candidate_source_to_slot
// and candidate_slot_cursor are mutable, caller-owned graph tensors; CANN
// receives them again as the final two output aliases solely to make writes
// visible to graph scheduling without adding them to Python's public tuple.
using PublicResult =
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

struct P9ShapeInfo {
    std::vector<int64_t> sparse_shape;
    int64_t rows = 0;
    int64_t source_count = 0;
    int64_t pool_count = 0;
    int64_t object_count = 0;
};

void RequireSameDevice(const at::Tensor& anchor, const at::Tensor& value,
                       const char* name, bool require_contiguous)
{
    TORCH_CHECK(value.defined() && value.device() == anchor.device(), name,
                " must be on the same device as query.");
    if (require_contiguous) {
        TORCH_CHECK(value.is_contiguous(), name, " must be contiguous.");
    }
}

void RequireInt32(const at::Tensor& value, const char* name)
{
    TORCH_CHECK(value.scalar_type() == at::kInt, name, " must be int32.");
}

std::vector<int64_t> SparseOutputShape(const at::Tensor& query,
                                       const at::Tensor& key,
                                       const std::string& layout_query,
                                       const std::string& layout_key,
                                       int64_t sparse_count)
{
    TORCH_CHECK(sparse_count == kExactSparseCount,
                "A5C8ExactSpeculativeResidentGet requires sparse_count=2048 "
                "because it owns one canonical resident range per exact Top-K "
                "row.");
    TORCH_CHECK(layout_key == "PA_BSND" && key.dim() == 4,
                "A5C8ExactSpeculativeResidentGet requires PA_BSND key "
                "[blocks,page,N2,D].");
    TORCH_CHECK(key.size(2) == kA5KeyHeads && key.size(3) == kA5HeadDim,
                "A5C8ExactSpeculativeResidentGet requires key N2=1, D=128.");
    if (layout_query == "TND") {
        TORCH_CHECK(query.dim() == 3 && query.size(1) == kA5IndexHeads &&
                        query.size(2) == kA5HeadDim,
                    "TND query must be [T,64,128].");
        return {query.size(0), key.size(2), sparse_count};
    }
    if (layout_query == "BSND") {
        TORCH_CHECK(query.dim() == 4 && query.size(2) == kA5IndexHeads &&
                        query.size(3) == kA5HeadDim,
                    "BSND query must be [B,S1,64,128].");
        return {query.size(0), query.size(1), key.size(2), sparse_count};
    }
    TORCH_CHECK(false, "layout_query must be TND or BSND, got ", layout_query);
    return {};
}

int64_t SelectionRows(const std::vector<int64_t>& sparse_shape)
{
    TORCH_CHECK(!sparse_shape.empty() &&
                    sparse_shape.back() == kExactSparseCount,
                "P9 sparse output must end in 2048.");
    int64_t rows = 1;
    for (size_t index = 0; index + 1 < sparse_shape.size(); ++index) {
        TORCH_CHECK(sparse_shape[index] > 0 &&
                        rows <= std::numeric_limits<int64_t>::max() /
                                    sparse_shape[index],
                    "P9 selection-row product is invalid.");
        rows *= sparse_shape[index];
    }
    return rows;
}

void RequireVector(const at::Tensor& value, int64_t size, const char* name)
{
    TORCH_CHECK(value.dim() == 1 && value.size(0) == size, name,
                " must be int32[R] with R=", size, ".");
}

P9ShapeInfo ValidateInputs(
    const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& weights, const at::Tensor& query_scale,
    const at::Tensor& key_scale, const at::Tensor& actual_query_lengths,
    const at::Tensor& actual_key_lengths, const at::Tensor& block_table,
    const at::Tensor& metadata, const at::Tensor& runtime_enable,
    const at::Tensor& resident_threshold,
    const at::Tensor& resident_threshold_valid,
    const at::Tensor& req_pool_entries, const at::Tensor& kv_object_map,
    const at::Tensor& kv_object_lens,
    const at::Tensor& persistent_object_slots_snapshot,
    const at::Tensor& canonical_resident_layout,
    at::Tensor& candidate_source_to_slot, at::Tensor& candidate_slot_cursor,
    const at::Tensor& layer_resident_get_control,
    const at::Tensor& rows_per_entry, int64_t query_quant_mode,
    int64_t key_quant_mode, const std::string& layout_query,
    const std::string& layout_key, int64_t sparse_count, int64_t cmp_ratio,
    int64_t stride, int64_t scale_stride, bool require_npu)
{
    if (require_npu) {
        TORCH_CHECK(query.device().is_privateuseone(),
                    "A5C8ExactSpeculativeResidentGet requires NPU inputs.");
    } else {
        TORCH_CHECK(query.device().is_meta(),
                    "P9 Meta implementation requires Meta inputs.");
    }
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous.");
    // PA_BSND's physical page stride is part of C8's exact scoring contract;
    // it is deliberately not materialized into a contiguous view.
    RequireSameDevice(query, key, "key", false);
    for (const auto& item : {
             std::pair<const at::Tensor*, const char*>{&weights, "weights"},
             {&query_scale, "query_dequant_scale"},
             {&key_scale, "key_dequant_scale"},
             {&actual_query_lengths, "actual_seq_lengths_query"},
             {&actual_key_lengths, "actual_seq_lengths_key"},
             {&block_table, "block_table"},
             {&metadata, "metadata"},
             {&runtime_enable, "runtime_enable"},
             {&resident_threshold, "resident_threshold"},
             {&resident_threshold_valid, "resident_threshold_valid"},
             {&req_pool_entries, "req_pool_entries"},
             {&kv_object_map, "kv_object_map"},
             {&kv_object_lens, "kv_object_lens"},
             {&persistent_object_slots_snapshot,
              "persistent_object_slots_snapshot"},
             {&canonical_resident_layout, "canonical_resident_layout"},
             {&candidate_source_to_slot, "candidate_source_to_slot"},
             {&candidate_slot_cursor, "candidate_slot_cursor"},
             {&layer_resident_get_control, "layer_resident_get_control"},
             {&rows_per_entry, "rows_per_entry"},
         }) {
        RequireSameDevice(query, *item.first, item.second, true);
    }

    TORCH_CHECK(query.scalar_type() == at::kFloat8_e4m3fn &&
                    key.scalar_type() == at::kFloat8_e4m3fn,
                "P9 requires FP8 E4M3FN query/key.");
    TORCH_CHECK(weights.scalar_type() == at::kFloat &&
                    query_scale.scalar_type() == at::kFloat &&
                    key_scale.scalar_type() == at::kFloat,
                "P9 requires FP32 weights and dequant scales.");
    for (const auto& item : {
             std::pair<const at::Tensor*, const char*>{
                 &actual_query_lengths, "actual_seq_lengths_query"},
             {&actual_key_lengths, "actual_seq_lengths_key"},
             {&block_table, "block_table"},
             {&metadata, "metadata"},
             {&runtime_enable, "runtime_enable"},
             {&resident_threshold, "resident_threshold"},
             {&resident_threshold_valid, "resident_threshold_valid"},
             {&req_pool_entries, "req_pool_entries"},
             {&kv_object_map, "kv_object_map"},
             {&kv_object_lens, "kv_object_lens"},
             {&persistent_object_slots_snapshot,
              "persistent_object_slots_snapshot"},
             {&canonical_resident_layout, "canonical_resident_layout"},
             {&candidate_source_to_slot, "candidate_source_to_slot"},
             {&candidate_slot_cursor, "candidate_slot_cursor"},
             {&rows_per_entry, "rows_per_entry"},
         }) {
        RequireInt32(*item.first, item.second);
    }
    TORCH_CHECK(layer_resident_get_control.scalar_type() == at::kLong,
                "layer_resident_get_control must be int64[16].");

    TORCH_CHECK(metadata.numel() == kMetadataWords,
                "metadata must contain exactly 1024 int32 words.");
    TORCH_CHECK(runtime_enable.dim() == 1 && runtime_enable.size(0) == 1,
                "runtime_enable must be int32[1].");
    TORCH_CHECK(canonical_resident_layout.dim() == 1 &&
                    canonical_resident_layout.size(0) ==
                        kCanonicalResidentLayoutWords,
                "canonical_resident_layout must be int32[4]: "
                "[first_slot, slots_per_row, row_stride, row_count].");
    TORCH_CHECK(layer_resident_get_control.dim() == 1 &&
                    layer_resident_get_control.size(0) ==
                        kLayerResidentGetControlWords,
                "layer_resident_get_control must be int64[16].");
    TORCH_CHECK(rows_per_entry.dim() == 1 && rows_per_entry.size(0) == 1,
                "rows_per_entry must be int32[1].");

    TORCH_CHECK(query_quant_mode == 0 && key_quant_mode == 0 &&
                    cmp_ratio == 1,
                "P9 requires the official C8 decode score domain: "
                "query_quant_mode=0, key_quant_mode=0, cmp_ratio=1.");
    const auto sparse_shape =
        SparseOutputShape(query, key, layout_query, layout_key, sparse_count);
    const int64_t rows = SelectionRows(sparse_shape);
    TORCH_CHECK(actual_query_lengths.numel() > 0 &&
                    actual_key_lengths.numel() == rows,
                "actual_seq_lengths_key must have one item per C8 selection "
                "row; actual_seq_lengths_query must be nonempty.");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == rows,
                "block_table must be int32[R,max_blocks] for R=", rows, ".");
    RequireVector(resident_threshold, rows, "resident_threshold");
    RequireVector(resident_threshold_valid, rows, "resident_threshold_valid");
    RequireVector(req_pool_entries, rows, "req_pool_entries");
    RequireVector(kv_object_lens, rows, "kv_object_lens");
    RequireVector(candidate_slot_cursor, rows, "candidate_slot_cursor");
    TORCH_CHECK(kv_object_map.dim() == 2 && kv_object_map.size(0) > 0 &&
                    kv_object_map.size(1) > 0,
                "kv_object_map must be int32[P,S] with positive P and S.");
    const int64_t pool_count = kv_object_map.size(0);
    const int64_t source_count = kv_object_map.size(1);
    TORCH_CHECK(persistent_object_slots_snapshot.dim() == 2 &&
                    persistent_object_slots_snapshot.size(0) == pool_count &&
                    persistent_object_slots_snapshot.size(1) > 0,
                "persistent_object_slots_snapshot must be int32[P,O] with "
                "the same P as kv_object_map.");
    TORCH_CHECK(candidate_source_to_slot.dim() == 2 &&
                    candidate_source_to_slot.size(0) == rows &&
                    candidate_source_to_slot.size(1) == source_count,
                "candidate_source_to_slot must be int32[R,S], got [",
                candidate_source_to_slot.size(0), ",",
                candidate_source_to_slot.size(1), "] expected [", rows, ",",
                source_count, "].");

    const int64_t physical_stride = key.stride(0);
    const int64_t physical_scale_stride = key_scale.stride(0);
    TORCH_CHECK(physical_stride > 0 && physical_scale_stride > 0,
                "PA_BSND key and key-scale strides must be positive.");
    TORCH_CHECK(stride == physical_stride,
                "stride must equal key.stride(0): got ", stride,
                " expected ", physical_stride, ".");
    TORCH_CHECK(scale_stride == physical_scale_stride,
                "scale_stride must equal key_dequant_scale.stride(0): got ",
                scale_stride, " expected ", physical_scale_stride, ".");

    // The four layout words are dynamic, device-resident graph data.  Reading
    // them on the host would introduce a synchronization edge and defeat the
    // asynchronous path.  The generated P9 tiler/kernel must fail closed on
    // invalid values (including slots_per_row != 2048) before dereference;
    // this bridge validates their exact ABI shape and type only.
    return {sparse_shape, rows, source_count, pool_count,
            persistent_object_slots_snapshot.size(1)};
}

PublicResult AllocateAndLaunch(
    const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& weights, const at::Tensor& query_scale,
    const at::Tensor& key_scale, const at::Tensor& actual_query_lengths,
    const at::Tensor& actual_key_lengths, const at::Tensor& block_table,
    const at::Tensor& metadata, const at::Tensor& runtime_enable,
    const at::Tensor& resident_threshold,
    const at::Tensor& resident_threshold_valid,
    const at::Tensor& req_pool_entries, const at::Tensor& kv_object_map,
    const at::Tensor& kv_object_lens,
    const at::Tensor& persistent_object_slots_snapshot,
    const at::Tensor& canonical_resident_layout,
    at::Tensor candidate_source_to_slot, at::Tensor candidate_slot_cursor,
    const at::Tensor& layer_resident_get_control,
    const at::Tensor& rows_per_entry, int64_t query_quant_mode,
    int64_t key_quant_mode, const std::string& layout_query,
    const std::string& layout_key, int64_t sparse_count, int64_t sparse_mode,
    int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio,
    bool return_values, int64_t stride, int64_t scale_stride)
{
    const auto info = ValidateInputs(
        query, key, weights, query_scale, key_scale, actual_query_lengths,
        actual_key_lengths, block_table, metadata, runtime_enable,
        resident_threshold, resident_threshold_valid, req_pool_entries,
        kv_object_map, kv_object_lens, persistent_object_slots_snapshot,
        canonical_resident_layout, candidate_source_to_slot,
        candidate_slot_cursor, layer_resident_get_control, rows_per_entry,
        query_quant_mode, key_quant_mode, layout_query, layout_key,
        sparse_count, cmp_ratio, stride, scale_stride, true);

    auto int_options = query.options().dtype(at::kInt);
    auto float_options = query.options().dtype(at::kFloat);
    auto sparse_indices = at::empty(info.sparse_shape, int_options);
    auto sparse_values = return_values ? at::empty(info.sparse_shape, float_options)
                                       : at::empty({0}, float_options);
    auto attention_slots =
        at::empty({info.rows, kExactSparseCount}, int_options);
    auto terminal_post_stats =
        at::empty({info.rows, kTerminalPostStatWords}, int_options);

    const auto key_arg = nanovllm_a5_opapi::AllowStridedAclTensor(key);
    const char* layout_query_ptr = layout_query.c_str();
    const char* layout_key_ptr = layout_key.c_str();
    auto keepalive = std::make_tuple(
        query, key, weights, query_scale, key_scale, actual_query_lengths,
        actual_key_lengths, block_table, metadata, runtime_enable,
        resident_threshold, resident_threshold_valid, req_pool_entries,
        kv_object_map, kv_object_lens, persistent_object_slots_snapshot,
        canonical_resident_layout, candidate_source_to_slot,
        candidate_slot_cursor, layer_resident_get_control, rows_per_entry,
        sparse_indices, sparse_values, attention_slots, terminal_post_stats);

    // ABI order is frozen by stage_a_b_contract.json:
    // 9 official inputs, 12 P9 inputs, official attrs, four public outputs,
    // then two caller-owned output aliases.  Do not reorder the two aliases:
    // graph mutation visibility is part of the target ABI.
    EXEC_NPU_CMD_ORDERED(
        aclnnA5C8ExactSpeculativeResidentGet, keepalive,
        query, key_arg, weights, query_scale, key_scale, actual_query_lengths,
        actual_key_lengths, block_table, metadata, runtime_enable,
        resident_threshold, resident_threshold_valid, req_pool_entries,
        kv_object_map, kv_object_lens, persistent_object_slots_snapshot,
        canonical_resident_layout, candidate_source_to_slot,
        candidate_slot_cursor, layer_resident_get_control, rows_per_entry,
        query_quant_mode, key_quant_mode, layout_query_ptr, layout_key_ptr,
        sparse_count, sparse_mode, pre_tokens, next_tokens, cmp_ratio,
        return_values, stride, scale_stride, sparse_indices, sparse_values,
        attention_slots, terminal_post_stats, candidate_source_to_slot,
        candidate_slot_cursor);
    return std::make_tuple(sparse_indices, sparse_values, attention_slots,
                           terminal_post_stats);
}

PublicResult A5C8ExactSpeculativeResidentGetNpu(
    const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& weights, const at::Tensor& query_scale,
    const at::Tensor& key_scale, const at::Tensor& actual_query_lengths,
    const at::Tensor& actual_key_lengths, const at::Tensor& block_table,
    const at::Tensor& metadata, const at::Tensor& runtime_enable,
    const at::Tensor& resident_threshold,
    const at::Tensor& resident_threshold_valid,
    const at::Tensor& req_pool_entries, const at::Tensor& kv_object_map,
    const at::Tensor& kv_object_lens,
    const at::Tensor& persistent_object_slots_snapshot,
    const at::Tensor& canonical_resident_layout,
    at::Tensor candidate_source_to_slot, at::Tensor candidate_slot_cursor,
    const at::Tensor& layer_resident_get_control,
    const at::Tensor& rows_per_entry, int64_t query_quant_mode,
    int64_t key_quant_mode, std::string layout_query, std::string layout_key,
    int64_t sparse_count, int64_t sparse_mode, int64_t pre_tokens,
    int64_t next_tokens, int64_t cmp_ratio, bool return_values, int64_t stride,
    int64_t scale_stride)
{
    return AllocateAndLaunch(
        query, key, weights, query_scale, key_scale, actual_query_lengths,
        actual_key_lengths, block_table, metadata, runtime_enable,
        resident_threshold, resident_threshold_valid, req_pool_entries,
        kv_object_map, kv_object_lens, persistent_object_slots_snapshot,
        canonical_resident_layout, candidate_source_to_slot,
        candidate_slot_cursor, layer_resident_get_control, rows_per_entry,
        query_quant_mode, key_quant_mode, layout_query, layout_key,
        sparse_count, sparse_mode, pre_tokens, next_tokens, cmp_ratio,
        return_values, stride, scale_stride);
}

PublicResult A5C8ExactSpeculativeResidentGetMeta(
    const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& weights, const at::Tensor& query_scale,
    const at::Tensor& key_scale, const at::Tensor& actual_query_lengths,
    const at::Tensor& actual_key_lengths, const at::Tensor& block_table,
    const at::Tensor& metadata, const at::Tensor& runtime_enable,
    const at::Tensor& resident_threshold,
    const at::Tensor& resident_threshold_valid,
    const at::Tensor& req_pool_entries, const at::Tensor& kv_object_map,
    const at::Tensor& kv_object_lens,
    const at::Tensor& persistent_object_slots_snapshot,
    const at::Tensor& canonical_resident_layout,
    at::Tensor candidate_source_to_slot, at::Tensor candidate_slot_cursor,
    const at::Tensor& layer_resident_get_control,
    const at::Tensor& rows_per_entry, int64_t query_quant_mode,
    int64_t key_quant_mode, std::string layout_query, std::string layout_key,
    int64_t sparse_count, int64_t sparse_mode, int64_t pre_tokens,
    int64_t next_tokens, int64_t cmp_ratio, bool return_values, int64_t stride,
    int64_t scale_stride)
{
    const auto info = ValidateInputs(
        query, key, weights, query_scale, key_scale, actual_query_lengths,
        actual_key_lengths, block_table, metadata, runtime_enable,
        resident_threshold, resident_threshold_valid, req_pool_entries,
        kv_object_map, kv_object_lens, persistent_object_slots_snapshot,
        canonical_resident_layout, candidate_source_to_slot,
        candidate_slot_cursor, layer_resident_get_control, rows_per_entry,
        query_quant_mode, key_quant_mode, layout_query, layout_key,
        sparse_count, cmp_ratio, stride, scale_stride, false);
    auto int_options = query.options().dtype(at::kInt);
    auto float_options = query.options().dtype(at::kFloat);
    return std::make_tuple(
        at::empty(info.sparse_shape, int_options),
        return_values ? at::empty(info.sparse_shape, float_options)
                      : at::empty({0}, float_options),
        at::empty({info.rows, kExactSparseCount}, int_options),
        at::empty({info.rows, kTerminalPostStatWords}, int_options));
}

}  // namespace

TORCH_LIBRARY(a5_c8_exact_speculative_resident_get, m)
{
    m.def(
        "npu_a5_c8_exact_speculative_resident_get("
        "Tensor query, Tensor key, Tensor weights, "
        "Tensor query_dequant_scale, Tensor key_dequant_scale, "
        "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_key, "
        "Tensor block_table, Tensor metadata, Tensor runtime_enable, "
        "Tensor resident_threshold, Tensor resident_threshold_valid, "
        "Tensor req_pool_entries, Tensor kv_object_map, "
        "Tensor kv_object_lens, Tensor persistent_object_slots_snapshot, "
        "Tensor canonical_resident_layout, "
        "Tensor(a!) candidate_source_to_slot, "
        "Tensor(b!) candidate_slot_cursor, "
        "Tensor layer_resident_get_control, Tensor rows_per_entry, "
        "int query_quant_mode=0, int key_quant_mode=0, "
        "str layout_query='TND', str layout_key='PA_BSND', "
        "int sparse_count=2048, int sparse_mode=3, "
        "int pre_tokens=9223372036854775807, "
        "int next_tokens=9223372036854775807, int cmp_ratio=1, "
        "bool return_values=False, int stride=1, int scale_stride=1) "
        "-> (Tensor sparse_indices, Tensor sparse_values, "
        "Tensor attention_slots, Tensor terminal_post_stats)");
}

TORCH_LIBRARY_IMPL(a5_c8_exact_speculative_resident_get, PrivateUse1, m)
{
    m.impl("npu_a5_c8_exact_speculative_resident_get",
           &A5C8ExactSpeculativeResidentGetNpu);
}

TORCH_LIBRARY_IMPL(a5_c8_exact_speculative_resident_get, Meta, m)
{
    m.impl("npu_a5_c8_exact_speculative_resident_get",
           &A5C8ExactSpeculativeResidentGetMeta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("p9_bridge_contract", [] {
        pybind11::dict contract;
        contract["official_tensor_inputs"] = 9;
        contract["p9_tensor_inputs"] = 12;
        contract["public_outputs"] = 4;
        contract["mutable_output_aliases"] = 2;
        contract["sparse_count"] = kExactSparseCount;
        contract["attention_slot_width"] = kExactSparseCount;
        contract["terminal_post_stat_width"] = kTerminalPostStatWords;
        contract["fixed_get_core_partition"] = false;
        contract["private_arena"] = false;
        return contract;
    });
}
