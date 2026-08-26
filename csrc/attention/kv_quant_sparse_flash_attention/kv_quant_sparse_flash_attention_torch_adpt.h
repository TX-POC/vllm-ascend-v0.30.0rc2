/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_TORCH_ADPT_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_TORCH_ADPT_H

#include <limits>

namespace vllm_ascend {

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
construct_kv_quant_sparse_flash_attention_output_tensor(
    const at::Tensor &query,
    const at::Tensor &key,
    const std::string &layout_query_str,
    const std::string &layout_kv_str,
    int64_t rope_head_dim,
    bool return_softmax_lse)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    constexpr int64_t DIM_2 = 2;
    constexpr int64_t DIM_3 = 3;
    constexpr int64_t DIM_4 = 4;

    TORCH_CHECK(layout_query_str == "BSND" || layout_query_str == "TND",
                "The layout of query only support BSND and TND, but got ",
                layout_query_str);
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater than 0, but shape[",
                    i, "] is ", query.size(i));
    }

    at::SmallVector<int64_t, SIZE> output_size;
    if (layout_query_str == "BSND") {
        TORCH_CHECK(query.dim() == DIM_4,
                    "When the layout of query is BSND, the query dimension must be 4, but got ",
                    query.dim());
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2),
                       query.size(DIM_3) - rope_head_dim};
    } else {
        TORCH_CHECK(query.dim() == DIM_3,
                    "When the layout of query is TND, the query dimension must be 3, but got ",
                    query.dim());
        output_size = {query.size(DIM_0), query.size(DIM_1),
                       query.size(DIM_2) - rope_head_dim};
    }

    at::Tensor attention_output =
        at::empty(output_size, query.options().dtype(query.dtype()));
    at::SmallVector<int64_t, SIZE> softmax_size;
    if (return_softmax_lse) {
        if (query.dim() == DIM_3) {
            const int64_t kv_head_dim =
                layout_kv_str == "PA_BSND" ? key.size(DIM_2) : key.size(DIM_1);
            softmax_size = {kv_head_dim, query.size(DIM_0),
                            query.size(DIM_1) / kv_head_dim};
        } else {
            softmax_size = {query.size(DIM_0), key.size(DIM_2),
                            query.size(DIM_1),
                            query.size(DIM_2) / key.size(DIM_2)};
        }
    } else {
        softmax_size = {0};
    }

    at::Tensor softmax_max =
        at::empty(softmax_size, query.options().dtype(at::kFloat));
    at::Tensor softmax_sum =
        at::empty(softmax_size, query.options().dtype(at::kFloat));
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        attention_output, softmax_max, softmax_sum);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor>
npu_kv_quant_sparse_flash_attention(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &sparse_indices,
    double scale_value,
    int64_t key_quant_mode,
    int64_t value_quant_mode,
    const c10::optional<at::Tensor> &key_dequant_scale,
    const c10::optional<at::Tensor> &value_dequant_scale,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    int64_t sparse_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t attention_mode,
    int64_t quant_scale_repo_mode,
    int64_t tile_size,
    int64_t rope_head_dim,
    bool return_softmax_lse)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(value.numel() > 0, "Tensor value is empty.");
    TORCH_CHECK(sparse_indices.numel() > 0, "Tensor sparse_indices is empty.");

    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    auto output = construct_kv_quant_sparse_flash_attention_output_tensor(
        query, key, layout_query_str, layout_kv_str, rope_head_dim,
        return_softmax_lse);
    at::Tensor attention_output = std::get<0>(output);
    at::Tensor softmax_max = std::get<1>(output);
    at::Tensor softmax_sum = std::get<2>(output);

    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD(
        aclnnKvQuantSparseFlashAttention,
        query,
        key,
        value,
        sparse_indices,
        key_dequant_scale,
        value_dequant_scale,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        sparse_block_size,
        layout_query_ptr,
        layout_kv_ptr,
        sparse_mode,
        pre_tokens,
        next_tokens,
        attention_mode,
        quant_scale_repo_mode,
        tile_size,
        rope_head_dim,
        return_softmax_lse,
        attention_output,
        softmax_max,
        softmax_sum);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        attention_output, softmax_max, softmax_sum);
}

at::Tensor npu_dsa_a5_sparse_tail_attention_c8(
    const at::Tensor &query,
    const at::Tensor &packed_kv,
    const at::Tensor &sparse_and_tail_slots,
    double scale_value,
    const at::Tensor &block_table,
    const at::Tensor &actual_seq_lengths_query,
    const at::Tensor &resident_seq_lengths,
    const at::Tensor &selected_slot_lengths)
{
    constexpr int64_t C8_QUERY_DIM = 576;
    constexpr int64_t C8_OUTPUT_DIM = 512;
    constexpr int64_t C8_BLOCK_SIZE = 128;
    constexpr int64_t C8_PACKED_KV_DIM = 656;
    constexpr int64_t C8_SPARSE_COUNT = 2048;
    constexpr int64_t C8_ROPE_HEAD_DIM = 64;

    TORCH_CHECK(
        query.dim() == 3 && query.size(0) > 0 && query.size(1) >= 1 &&
            query.size(1) <= 64 && query.size(2) == C8_QUERY_DIM,
        "C8 sparse+tail attention query must be TND [T,Q_HEAD,576] "
        "with 1 <= Q_HEAD <= 64.");
    TORCH_CHECK(
        query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kHalf,
        "C8 sparse+tail attention query must be BF16 or FP16.");
    TORCH_CHECK(
        packed_kv.dim() == 4 && packed_kv.size(0) > 0 &&
            packed_kv.size(1) == C8_BLOCK_SIZE && packed_kv.size(2) == 1 &&
            packed_kv.size(3) == C8_PACKED_KV_DIM && packed_kv.element_size() == 1,
        "C8 sparse+tail attention packed KV must be a one-byte tensor "
        "[blocks,128,1,656].");
    TORCH_CHECK(
        packed_kv.scalar_type() == at::ScalarType::Float8_e4m3fn ||
            packed_kv.scalar_type() == at::kChar,
        "C8 sparse+tail attention packed KV must be FP8 E4M3FN or INT8.");
    TORCH_CHECK(
        sparse_and_tail_slots.dim() == 3 &&
            sparse_and_tail_slots.size(0) == query.size(0) &&
            sparse_and_tail_slots.size(1) == 1 &&
            sparse_and_tail_slots.size(2) >= C8_SPARSE_COUNT &&
            sparse_and_tail_slots.scalar_type() == at::kInt,
        "C8 sparse+tail slots must be int32 [T,1,2048+max_tail_tokens].");

    const int64_t batch = actual_seq_lengths_query.numel();
    TORCH_CHECK(
        batch > 0 && actual_seq_lengths_query.dim() == 1 &&
            resident_seq_lengths.dim() == 1 && resident_seq_lengths.numel() == batch &&
            block_table.dim() == 2 && block_table.size(0) == batch && block_table.size(1) > 0,
        "C8 sparse+tail attention batch metadata shapes are inconsistent.");
    TORCH_CHECK(
        block_table.scalar_type() == at::kInt &&
            actual_seq_lengths_query.scalar_type() == at::kInt &&
            resident_seq_lengths.scalar_type() == at::kInt &&
            selected_slot_lengths.scalar_type() == at::kInt,
        "C8 sparse+tail attention metadata must be int32.");
    TORCH_CHECK(
        selected_slot_lengths.dim() == 1 &&
            selected_slot_lengths.numel() == batch,
        "C8 sparse+tail attention selected_slot_lengths must be int32 [T].");
    TORCH_CHECK(
        query.is_privateuseone() && packed_kv.device() == query.device() &&
            sparse_and_tail_slots.device() == query.device() &&
            block_table.device() == query.device() &&
            actual_seq_lengths_query.device() == query.device() &&
            resident_seq_lengths.device() == query.device() &&
            selected_slot_lengths.device() == query.device(),
        "C8 sparse+tail attention inputs must be on one NPU.");
    TORCH_CHECK(
        query.is_contiguous() && packed_kv.is_contiguous() &&
            sparse_and_tail_slots.is_contiguous() && block_table.is_contiguous() &&
            actual_seq_lengths_query.is_contiguous() && resident_seq_lengths.is_contiguous() &&
            selected_slot_lengths.is_contiguous(),
        "C8 sparse+tail attention inputs must be contiguous.");

    at::Tensor attention_output = at::empty(
        {query.size(0), query.size(1), C8_OUTPUT_DIM}, query.options());
    at::Tensor softmax_max = at::empty({1}, query.options().dtype(at::kFloat));
    at::Tensor softmax_sum = at::empty({1}, query.options().dtype(at::kFloat));

    // Repository mode reads quant scales from packed_kv, but the A5 tiler
    // still requires non-null external scale inputs. The C8 A5 kernel does
    // not read key_dequant_scale as a quant scale in this mode; reuse that
    // otherwise-unused pointer to carry the per-row selected slot boundary.
    // This keeps graph's fixed sparse_indices shape from turning trailing
    // -1 sentinels into a partially uninitialized final BMM tile.
    const at::Tensor selected_slot_lengths_placeholder =
        selected_slot_lengths.view(at::kFloat);
    const at::Tensor dequant_scale_placeholder =
        resident_seq_lengths.view(at::kFloat);
    constexpr int64_t key_quant_mode = 2;
    constexpr int64_t value_quant_mode = 2;
    constexpr int64_t sparse_block_size = 1;
    std::string layout_query = "TND";
    std::string layout_kv = "PA_BSND";
    char *layout_query_ptr = layout_query.data();
    char *layout_kv_ptr = layout_kv.data();
    constexpr int64_t sparse_mode = 3;
    constexpr int64_t all_tokens = std::numeric_limits<int64_t>::max();
    constexpr int64_t attention_mode = 2;
    constexpr int64_t quant_scale_repo_mode = 1;
    constexpr int64_t tile_size = 128;
    constexpr bool return_softmax_lse = false;

    EXEC_NPU_CMD(
        aclnnA5SparseTailAttentionC8,
        query,
        packed_kv,
        packed_kv,
        sparse_and_tail_slots,
        selected_slot_lengths_placeholder,
        dequant_scale_placeholder,
        block_table,
        actual_seq_lengths_query,
        resident_seq_lengths,
        scale_value,
        key_quant_mode,
        value_quant_mode,
        sparse_block_size,
        layout_query_ptr,
        layout_kv_ptr,
        sparse_mode,
        all_tokens,
        all_tokens,
        attention_mode,
        quant_scale_repo_mode,
        tile_size,
        C8_ROPE_HEAD_DIM,
        return_softmax_lse,
        attention_output,
        softmax_max,
        softmax_sum);
    return attention_output;
}
}  // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, dsa_a5_sparse_tail_attention_c8_ops)
{
    dsa_a5_sparse_tail_attention_c8_ops.def(
        "npu_dsa_a5_sparse_tail_attention_c8("
        "Tensor query, Tensor packed_kv, Tensor sparse_and_tail_slots, "
        "float scale_value, Tensor block_table, "
        "Tensor actual_seq_lengths_query, Tensor resident_seq_lengths, "
        "Tensor selected_slot_lengths) -> Tensor");
}

TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, dsa_a5_sparse_tail_attention_c8_impl)
{
    dsa_a5_sparse_tail_attention_c8_impl.impl(
        "npu_dsa_a5_sparse_tail_attention_c8",
        &vllm_ascend::npu_dsa_a5_sparse_tail_attention_c8);
}

#endif  // KV_QUANT_SPARSE_FLASH_ATTENTION_TORCH_ADPT_H
