// Standalone source-matched caller for the upstream A5 C8 ACLNN operators.
//
// The input/output allocation and ACLNN argument order below are derived from
// csrc/torch_binding.cpp::npu_vllm_quant_lightning_indexer_{npu,metadata_npu}.
// Keep this bridge validation-only: it owns no vLLM policy or product route.

#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/extension.h>
#include <torch/library.h>

#define NANOVLLM_A5_OPAPI_ENV "NANOVLLM_A5_REFERENCE_C8_OPAPI_LIB"
#include "op_api_common.h"

namespace {

constexpr int64_t kMetadataWords = 1024;
constexpr int64_t kDefaultSparseCount = 2048;

using IndexResult = std::tuple<at::Tensor, at::Tensor>;

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
                                       const at::Tensor& key,
                                       const std::string& layout_query,
                                       const std::string& layout_key,
                                       int64_t sparse_count)
{
    TORCH_CHECK(sparse_count > 0 && sparse_count <= kDefaultSparseCount,
                "sparse_count must be in [1, 2048].");
    if (layout_query == "TND") {
        TORCH_CHECK(query.dim() == 3, "TND query must be [T,N1,D].");
        TORCH_CHECK(layout_key == "PA_BSND" && key.dim() == 4,
                    "TND reference requires PA_BSND key [blocks,page,N2,D].");
        return {query.size(0), key.size(2), sparse_count};
    }
    if (layout_query == "BSND") {
        TORCH_CHECK(query.dim() == 4, "BSND query must be [B,S1,N1,D].");
        TORCH_CHECK(layout_key == "PA_BSND" && key.dim() == 4,
                    "BSND reference requires PA_BSND key [blocks,page,N2,D].");
        return {query.size(0), query.size(1), key.size(2), sparse_count};
    }
    TORCH_CHECK(false, "layout_query must be TND or BSND, got ", layout_query);
    return {};
}

void CheckIndexerInputs(const at::Tensor& query, const at::Tensor& key,
                        const at::Tensor& weights, const at::Tensor& query_scale,
                        const at::Tensor& key_scale,
                        const at::Tensor& actual_query_lengths,
                        const at::Tensor& actual_key_lengths,
                        const at::Tensor& block_table,
                        const at::Tensor& metadata,
                        const std::string& layout_query,
                        const std::string& layout_key,
                        int64_t sparse_count)
{
    TORCH_CHECK(query.device().is_privateuseone(),
                "official C8 reference requires NPU inputs.");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous.");
    // PA_BSND has a physical page stride.  The production bridge does not
    // materialize it, so neither can this source-equivalent reference.
    RequireSameNpu(query, key, "key", false);
    for (const auto& item : {
             std::pair<const at::Tensor*, const char*>{&weights, "weights"},
             {&query_scale, "query_dequant_scale"},
             {&key_scale, "key_dequant_scale"},
             {&actual_query_lengths, "actual_seq_lengths_query"},
             {&actual_key_lengths, "actual_seq_lengths_key"},
             {&block_table, "block_table"}, {&metadata, "metadata"},
         }) {
        RequireSameNpu(query, *item.first, item.second, true);
    }
    TORCH_CHECK(query.scalar_type() == at::kFloat8_e4m3fn &&
                    key.scalar_type() == at::kFloat8_e4m3fn,
                "official A5 C8 reference requires FP8 E4M3FN query/key.");
    TORCH_CHECK(weights.scalar_type() == at::kFloat &&
                    query_scale.scalar_type() == at::kFloat &&
                    key_scale.scalar_type() == at::kFloat,
                "official A5 C8 reference requires FP32 weights/scales.");
    TORCH_CHECK(actual_query_lengths.scalar_type() == at::kInt &&
                    actual_key_lengths.scalar_type() == at::kInt &&
                    block_table.scalar_type() == at::kInt &&
                    metadata.scalar_type() == at::kInt,
                "official A5 C8 reference metadata inputs must be int32.");
    TORCH_CHECK(metadata.numel() == kMetadataWords,
                "official C8 metadata must contain 1024 int32 words.");
    (void)SparseOutputShape(query, key, layout_query, layout_key, sparse_count);
}

at::Tensor OfficialMetadataNpu(
    int64_t num_heads_q, int64_t num_heads_k, int64_t head_dim,
    int64_t query_quant_mode, int64_t key_quant_mode,
    const at::Tensor& actual_query_lengths,
    const at::Tensor& actual_key_lengths, int64_t batch_size,
    int64_t max_seqlen_q, int64_t max_seqlen_k, std::string layout_query,
    std::string layout_key, int64_t sparse_count, int64_t sparse_mode,
    int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio,
    std::string device)
{
    TORCH_CHECK(actual_query_lengths.device().is_privateuseone(),
                "official C8 metadata requires an NPU sequence-length tensor.");
    RequireSameNpu(actual_query_lengths, actual_key_lengths,
                   "actual_seq_lengths_key", true);
    TORCH_CHECK(actual_query_lengths.scalar_type() == at::kInt &&
                    actual_key_lengths.scalar_type() == at::kInt,
                "official C8 metadata sequence lengths must be int32.");
    TORCH_CHECK(num_heads_q == 64 && num_heads_k == 1 && head_dim == 128,
                "A5 C8 reference requires N1=64, N2=1, D=128.");
    TORCH_CHECK(batch_size > 0 && sparse_count > 0 && max_seqlen_q > 0 &&
                    max_seqlen_k > 0,
                "official C8 metadata dimensions must be positive.");
    auto output = at::empty({kMetadataWords},
                            actual_query_lengths.options().dtype(at::kInt));
    char* layout_query_ptr = const_cast<char*>(layout_query.c_str());
    char* layout_key_ptr = const_cast<char*>(layout_key.c_str());
    auto keepalive = std::make_tuple(actual_query_lengths, actual_key_lengths,
                                     output);
    EXEC_NPU_CMD_ORDERED(
        aclnnVllmQuantLightningIndexerMetadata, keepalive,
        actual_query_lengths, actual_key_lengths, num_heads_q, num_heads_k,
        head_dim, query_quant_mode, key_quant_mode, batch_size, max_seqlen_q,
        max_seqlen_k, layout_query_ptr, layout_key_ptr, sparse_count,
        sparse_mode, pre_tokens, next_tokens, cmp_ratio, output);
    (void)device;
    return output;
}

IndexResult OfficialIndexerNpu(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights,
    const at::Tensor& query_scale, const at::Tensor& key_scale,
    int64_t query_quant_mode, int64_t key_quant_mode,
    const at::Tensor& actual_query_lengths,
    const at::Tensor& actual_key_lengths, const at::Tensor& block_table,
    const at::Tensor& metadata, std::string layout_query, std::string layout_key,
    int64_t sparse_count, int64_t sparse_mode, int64_t pre_tokens,
    int64_t next_tokens, int64_t cmp_ratio, bool return_value)
{
    CheckIndexerInputs(query, key, weights, query_scale, key_scale,
                       actual_query_lengths, actual_key_lengths, block_table,
                       metadata, layout_query, layout_key, sparse_count);
    const auto output_shape =
        SparseOutputShape(query, key, layout_query, layout_key, sparse_count);
    auto sparse_indices = at::empty(output_shape, query.options().dtype(at::kInt));
    auto sparse_values = return_value
        ? at::empty(output_shape, query.options().dtype(at::kFloat))
        : at::empty({0}, query.options().dtype(at::kFloat));
    char* layout_query_ptr = const_cast<char*>(layout_query.c_str());
    char* layout_key_ptr = const_cast<char*>(layout_key.c_str());
    const int64_t stride = key.stride(0);
    const int64_t scale_stride = key_scale.stride(0);
    const auto key_arg = nanovllm_a5_opapi::AllowStridedAclTensor(key);
    auto keepalive = std::make_tuple(
        query, key, weights, query_scale, key_scale, actual_query_lengths,
        actual_key_lengths, block_table, metadata, sparse_indices, sparse_values);
    EXEC_NPU_CMD_ORDERED(
        aclnnVllmQuantLightningIndexer, keepalive,
        query, key_arg, weights, query_scale, key_scale, actual_query_lengths,
        actual_key_lengths, block_table, metadata, query_quant_mode,
        key_quant_mode, layout_query_ptr, layout_key_ptr, sparse_count,
        sparse_mode, pre_tokens, next_tokens, cmp_ratio, return_value, stride,
        scale_stride, sparse_indices, sparse_values);
    return std::make_tuple(sparse_indices, sparse_values);
}

}  // namespace

TORCH_LIBRARY(a5_official_c8_reference, m)
{
    m.def(
        "npu_vllm_quant_lightning_indexer("
        "Tensor query, Tensor key, Tensor weights, "
        "Tensor query_dequant_scale, Tensor key_dequant_scale, "
        "int query_quant_mode, int key_quant_mode, "
        "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_key, "
        "Tensor block_table, Tensor metadata, "
        "str layout_query='BSND', str layout_key='BSND', "
        "int sparse_count=2048, int sparse_mode=3, "
        "int pre_tokens=9223372036854775807, "
        "int next_tokens=9223372036854775807, int cmp_ratio=1, "
        "bool return_value=False) -> (Tensor sparse_indices, Tensor sparse_values)");
    m.def(
        "npu_vllm_quant_lightning_indexer_metadata("
        "int num_heads_q, int num_heads_k, int head_dim, "
        "int query_quant_mode, int key_quant_mode, "
        "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_key, "
        "int batch_size=0, int max_seqlen_q=0, int max_seqlen_k=0, "
        "str layout_query='BSND', str layout_key='BSND', "
        "int sparse_count=2048, int sparse_mode=3, "
        "int pre_tokens=9223372036854775807, "
        "int next_tokens=9223372036854775807, int cmp_ratio=1, "
        "str device='npu') -> Tensor metadata");
}

TORCH_LIBRARY_IMPL(a5_official_c8_reference, PrivateUse1, m)
{
    m.impl("npu_vllm_quant_lightning_indexer", &OfficialIndexerNpu);
    m.impl("npu_vllm_quant_lightning_indexer_metadata", &OfficialMetadataNpu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
