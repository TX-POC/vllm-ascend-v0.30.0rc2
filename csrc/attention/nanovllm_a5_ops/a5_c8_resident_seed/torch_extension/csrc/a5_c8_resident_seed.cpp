// Standalone ACLNN bridge for the isolated A5C8ResidentSeed OPP.
//
// This bridge is intentionally validation-only.  It owns a resolver domain
// distinct from both the source-matched official C8 bridge and the P3/P4
// streaming bridge, so all three OPPs may coexist in one Python process
// without a stale libcust_opapi.so ABI being selected accidentally.

#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/extension.h>
#include <torch/library.h>

#define NANOVLLM_A5_OPAPI_ENV "NANOVLLM_A5_RESIDENT_SEED_OPAPI_LIB"
#include "op_api_common.h"

namespace {

constexpr int64_t kMetadataWords = 1024;
constexpr int64_t kExactSparseCount = 2048;
constexpr int64_t kA5IndexHeads = 64;
constexpr int64_t kA5KeyHeads = 1;
constexpr int64_t kA5HeadDim = 128;

using Result = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

void RequireSameNpu(const at::Tensor& anchor, const at::Tensor& value,
                    const char* name, bool require_contiguous)
{
    TORCH_CHECK(value.defined() && value.device() == anchor.device(), name,
                " must be on the same NPU as query.");
    if (require_contiguous) {
        TORCH_CHECK(value.is_contiguous(), name, " must be contiguous.");
    }
}

std::vector<int64_t> SparseOutputShape(const at::Tensor& query,
                                       const at::Tensor& resident_index_key,
                                       const std::string& layout_query,
                                       const std::string& layout_key,
                                       int64_t sparse_count)
{
    TORCH_CHECK(sparse_count == kExactSparseCount,
                "A5C8ResidentSeed requires sparse_count=2048 so resident_tau "
                "is the exact C8 Top-2048 cutoff.");
    TORCH_CHECK(layout_key == "PA_BSND" && resident_index_key.dim() == 4,
                "A5C8ResidentSeed requires PA_BSND resident_index_key "
                "[blocks,page,N2,D].");
    TORCH_CHECK(resident_index_key.size(2) == kA5KeyHeads &&
                    resident_index_key.size(3) == kA5HeadDim,
                "A5C8ResidentSeed requires N2=1 and D=128.");
    if (layout_query == "TND") {
        TORCH_CHECK(query.dim() == 3 && query.size(1) == kA5IndexHeads &&
                        query.size(2) == kA5HeadDim,
                    "TND query must be [T,64,128].");
        return {query.size(0), resident_index_key.size(2), sparse_count};
    }
    if (layout_query == "BSND") {
        TORCH_CHECK(query.dim() == 4 && query.size(2) == kA5IndexHeads &&
                        query.size(3) == kA5HeadDim,
                    "BSND query must be [B,S1,64,128].");
        return {query.size(0), query.size(1), resident_index_key.size(2),
                sparse_count};
    }
    TORCH_CHECK(false, "layout_query must be TND or BSND, got ", layout_query);
    return {};
}

int64_t SelectionRows(const std::vector<int64_t>& sparse_shape)
{
    TORCH_CHECK(!sparse_shape.empty() &&
                    sparse_shape.back() == kExactSparseCount,
                "A5C8ResidentSeed sparse output must end in 2048.");
    int64_t rows = 1;
    for (size_t index = 0; index + 1 < sparse_shape.size(); ++index) {
        TORCH_CHECK(sparse_shape[index] > 0 &&
                        rows <= std::numeric_limits<int64_t>::max() /
                                    sparse_shape[index],
                    "A5C8ResidentSeed selection-row product is invalid.");
        rows *= sparse_shape[index];
    }
    return rows;
}

void CheckInputs(
    const at::Tensor& query, const at::Tensor& resident_index_key,
    const at::Tensor& weights, const at::Tensor& query_scale,
    const at::Tensor& resident_index_key_scale,
    const at::Tensor& actual_query_lengths,
    const at::Tensor& resident_seq_lengths,
    const at::Tensor& resident_index_block_table, const at::Tensor& metadata,
    const std::string& layout_query, const std::string& layout_key,
    int64_t sparse_count)
{
    TORCH_CHECK(query.device().is_privateuseone(),
                "A5C8ResidentSeed requires NPU inputs.");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous.");
    // Preserve PA_BSND's physical page stride exactly, just as the official
    // reference bridge does.  Materializing it would invalidate the C8
    // score equivalence that ResidentSeed is meant to observe.
    RequireSameNpu(query, resident_index_key, "resident_index_key", false);
    for (const auto& item : {
             std::pair<const at::Tensor*, const char*>{&weights, "weights"},
             {&query_scale, "query_dequant_scale"},
             {&resident_index_key_scale,
              "resident_index_key_dequant_scale"},
             {&actual_query_lengths, "actual_seq_lengths_query"},
             {&resident_seq_lengths, "resident_seq_lengths"},
             {&resident_index_block_table, "resident_index_block_table"},
             {&metadata, "metadata"},
         }) {
        RequireSameNpu(query, *item.first, item.second, true);
    }
    TORCH_CHECK(query.scalar_type() == at::kFloat8_e4m3fn &&
                    resident_index_key.scalar_type() == at::kFloat8_e4m3fn,
                "A5C8ResidentSeed requires FP8 E4M3FN query/index-key.");
    TORCH_CHECK(weights.scalar_type() == at::kFloat &&
                    query_scale.scalar_type() == at::kFloat &&
                    resident_index_key_scale.scalar_type() == at::kFloat,
                "A5C8ResidentSeed requires FP32 weights/dequant scales.");
    for (const auto& item : {
             std::pair<const at::Tensor*, const char*>{
                 &actual_query_lengths, "actual_seq_lengths_query"},
             {&resident_seq_lengths, "resident_seq_lengths"},
             {&resident_index_block_table, "resident_index_block_table"},
             {&metadata, "metadata"},
         }) {
        TORCH_CHECK(item.first->scalar_type() == at::kInt, item.second,
                    " must be int32.");
    }
    TORCH_CHECK(metadata.numel() == kMetadataWords,
                "A5C8ResidentSeed metadata must contain 1024 int32 words.");
    const auto sparse_shape = SparseOutputShape(
        query, resident_index_key, layout_query, layout_key, sparse_count);
    const int64_t rows = SelectionRows(sparse_shape);
    TORCH_CHECK(actual_query_lengths.numel() > 0 &&
                    resident_seq_lengths.numel() == rows,
                "resident_seq_lengths must have one item per selection row; "
                "actual_seq_lengths_query must be nonempty.");
    TORCH_CHECK(resident_index_block_table.dim() == 2 &&
                    resident_index_block_table.size(0) == rows,
                "resident_index_block_table must be int32[R,max_blocks].");
}

Result AllocateAndLaunch(
    const at::Tensor& query, const at::Tensor& resident_index_key,
    const at::Tensor& weights, const at::Tensor& query_scale,
    const at::Tensor& resident_index_key_scale,
    const at::Tensor& actual_query_lengths,
    const at::Tensor& resident_seq_lengths,
    const at::Tensor& resident_index_block_table, const at::Tensor& metadata,
    int64_t query_quant_mode, int64_t key_quant_mode,
    const std::string& layout_query, const std::string& layout_key,
    int64_t sparse_count, int64_t sparse_mode, int64_t pre_tokens,
    int64_t next_tokens, int64_t cmp_ratio, bool return_values, int64_t stride,
    int64_t scale_stride)
{
    CheckInputs(query, resident_index_key, weights, query_scale,
                resident_index_key_scale, actual_query_lengths,
                resident_seq_lengths, resident_index_block_table, metadata,
                layout_query, layout_key, sparse_count);
    TORCH_CHECK(!return_values,
                "A5C8ResidentSeed does not expose sparse_values: its only "
                "score-domain output is resident_tau (sortable uint16 in "
                "int32 low bits).");
    TORCH_CHECK(cmp_ratio == 1,
                "A5C8ResidentSeed bridge is decode-only and requires "
                "cmp_ratio=1.");
    TORCH_CHECK(query_quant_mode == 0 && key_quant_mode == 0,
                "A5C8ResidentSeed bridge requires C8 quant modes 0/0.");
    const auto sparse_shape = SparseOutputShape(
        query, resident_index_key, layout_query, layout_key, sparse_count);
    const int64_t rows = SelectionRows(sparse_shape);
    auto int_options = query.options().dtype(at::kInt);
    auto float_options = query.options().dtype(at::kFloat);
    auto sparse_indices = at::empty(sparse_shape, int_options);
    auto sparse_values = at::empty({0}, float_options);
    auto resident_tau = at::empty({rows}, int_options);
    auto tau_valid = at::empty({rows}, int_options);

    const auto index_key_arg =
        nanovllm_a5_opapi::AllowStridedAclTensor(resident_index_key);
    const char* layout_query_ptr = layout_query.c_str();
    const char* layout_key_ptr = layout_key.c_str();
    auto keepalive = std::make_tuple(
        query, resident_index_key, weights, query_scale,
        resident_index_key_scale, actual_query_lengths, resident_seq_lengths,
        resident_index_block_table, metadata, sparse_indices, sparse_values,
        resident_tau, tau_valid);
    EXEC_NPU_CMD_ORDERED(
        aclnnA5C8ResidentSeed, keepalive,
        query, index_key_arg, weights, query_scale, resident_index_key_scale,
        actual_query_lengths, resident_seq_lengths, resident_index_block_table,
        metadata, query_quant_mode, key_quant_mode, layout_query_ptr,
        layout_key_ptr, sparse_count, sparse_mode, pre_tokens, next_tokens,
        cmp_ratio, return_values, stride, scale_stride, sparse_indices,
        sparse_values, resident_tau, tau_valid);
    return std::make_tuple(sparse_indices, sparse_values, resident_tau,
                           tau_valid);
}

Result A5C8ResidentSeedNpu(
    const at::Tensor& query, const at::Tensor& resident_index_key,
    const at::Tensor& weights, const at::Tensor& query_scale,
    const at::Tensor& resident_index_key_scale,
    const at::Tensor& actual_query_lengths,
    const at::Tensor& resident_seq_lengths,
    const at::Tensor& resident_index_block_table, const at::Tensor& metadata,
    int64_t query_quant_mode, int64_t key_quant_mode, std::string layout_query,
    std::string layout_key, int64_t sparse_count, int64_t sparse_mode,
    int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio,
    bool return_values, int64_t stride, int64_t scale_stride)
{
    return AllocateAndLaunch(
        query, resident_index_key, weights, query_scale,
        resident_index_key_scale, actual_query_lengths, resident_seq_lengths,
        resident_index_block_table, metadata, query_quant_mode, key_quant_mode,
        layout_query, layout_key, sparse_count, sparse_mode, pre_tokens,
        next_tokens, cmp_ratio, return_values, stride, scale_stride);
}

Result A5C8ResidentSeedMeta(
    const at::Tensor& query, const at::Tensor& resident_index_key,
    const at::Tensor& weights, const at::Tensor& query_scale,
    const at::Tensor& resident_index_key_scale,
    const at::Tensor& actual_query_lengths,
    const at::Tensor& resident_seq_lengths,
    const at::Tensor& resident_index_block_table, const at::Tensor& metadata,
    int64_t query_quant_mode, int64_t key_quant_mode, std::string layout_query,
    std::string layout_key, int64_t sparse_count, int64_t sparse_mode,
    int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio,
    bool return_values, int64_t stride, int64_t scale_stride)
{
    CheckInputs(query, resident_index_key, weights, query_scale,
                resident_index_key_scale, actual_query_lengths,
                resident_seq_lengths, resident_index_block_table, metadata,
                layout_query, layout_key, sparse_count);
    TORCH_CHECK(!return_values,
                "A5C8ResidentSeed does not expose sparse_values.");
    TORCH_CHECK(cmp_ratio == 1 && query_quant_mode == 0 && key_quant_mode == 0,
                "A5C8ResidentSeed meta bridge requires C8 decode attrs.");
    const auto sparse_shape = SparseOutputShape(
        query, resident_index_key, layout_query, layout_key, sparse_count);
    const int64_t rows = SelectionRows(sparse_shape);
    auto int_options = query.options().dtype(at::kInt);
    auto float_options = query.options().dtype(at::kFloat);
    return std::make_tuple(
        at::empty(sparse_shape, int_options),
        at::empty({0}, float_options),
        at::empty({rows}, int_options), at::empty({rows}, int_options));
}

}  // namespace

TORCH_LIBRARY(a5_c8_resident_seed, m)
{
    m.def(
        "npu_a5_c8_resident_seed("
        "Tensor query, Tensor resident_index_key, Tensor weights, "
        "Tensor query_dequant_scale, Tensor resident_index_key_dequant_scale, "
        "Tensor actual_seq_lengths_query, Tensor resident_seq_lengths, "
        "Tensor resident_index_block_table, Tensor metadata, "
        "int query_quant_mode=0, int key_quant_mode=0, "
        "str layout_query='TND', str layout_key='PA_BSND', "
        "int sparse_count=2048, int sparse_mode=3, "
        "int pre_tokens=9223372036854775807, "
        "int next_tokens=9223372036854775807, int cmp_ratio=1, "
        "bool return_values=False, int stride=1, int scale_stride=1) "
        "-> (Tensor sparse_indices, Tensor sparse_values, "
        "Tensor resident_tau, Tensor tau_valid)");
}

TORCH_LIBRARY_IMPL(a5_c8_resident_seed, PrivateUse1, m)
{
    m.impl("npu_a5_c8_resident_seed", &A5C8ResidentSeedNpu);
}

TORCH_LIBRARY_IMPL(a5_c8_resident_seed, Meta, m)
{
    m.impl("npu_a5_c8_resident_seed", &A5C8ResidentSeedMeta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
