#include <initializer_list>
#include <limits>
#include <string>
#include <tuple>

#include <torch/extension.h>
#include <torch/library.h>

#include "op_api_common.h"

namespace {
constexpr int64_t kBlockSize = 128;
constexpr int64_t kKpeDim = 64;
constexpr int64_t kCkvDim = 512;
constexpr int64_t kIndexerDim = 128;
constexpr int64_t kSparseCount = 2048;
constexpr int64_t kPackedKvDim = 656;
constexpr int64_t kMaxSourceCapacity = 1 << 18;

void CheckOneDeviceAndContiguous(
    const at::Tensor& reference,
    std::initializer_list<const at::Tensor*> tensors,
    const char* op_name) {
  TORCH_CHECK(reference.device().is_privateuseone(), op_name, " inputs must be on NPU.");
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(
        tensor->device() == reference.device() && tensor->is_contiguous(),
        op_name, " inputs must be contiguous tensors on one NPU.");
  }
}

void CheckLiduCommon(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& req_pool_entries,
    const at::Tensor& cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& block_table) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 &&
          (query.size(1) == 32 || query.size(1) == 64) &&
          query.size(2) == kIndexerDim,
      "LIDU query must be [B,32|64,128].");
  const int64_t batch = query.size(0);
  TORCH_CHECK(
      key.dim() == 4 && key.size(0) > 0 && key.size(1) == kBlockSize &&
          key.size(2) == 1 && key.size(3) == kIndexerDim,
      "LIDU key must be [blocks,128,1,128].");
  TORCH_CHECK(
      weights.dim() == 2 && weights.size(0) == batch &&
          weights.size(1) == query.size(1),
      "LIDU weights must match query [B,N].");
  TORCH_CHECK(
      req_pool_entries.dim() == 1 && req_pool_entries.size(0) == batch &&
          cache_tokens.dim() == 1 && cache_tokens.size(0) == batch &&
          candidate_lens.dim() == 1 && candidate_lens.size(0) == batch,
      "LIDU request metadata must be int32[B].");
  TORCH_CHECK(
      cache_slots_pool.dim() == 2 && cache_slots_pool.size(0) > 0 &&
          cache_slots_pool.size(1) > 0 &&
          cache_slots_pool.size(1) <= kMaxSourceCapacity,
      "LIDU cache_slots_pool must be [pool_size,capacity], capacity <= 2^18.");
  TORCH_CHECK(
      block_table.dim() == 2 && block_table.size(0) == batch &&
          block_table.size(1) > 0 &&
          block_table.size(1) * kBlockSize == cache_slots_pool.size(1),
      "LIDU block_table capacity must equal cache_slots_pool.shape[1].");
  TORCH_CHECK(
      query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kHalf,
      "LIDU floating tensors must be bf16 or fp16.");
  TORCH_CHECK(
      key.scalar_type() == query.scalar_type() &&
          weights.scalar_type() == query.scalar_type(),
      "LIDU query/key/weights dtypes must match.");
  for (const at::Tensor* tensor :
       {&req_pool_entries, &cache_slots_pool, &cache_tokens,
        &candidate_lens, &block_table}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt, "LIDU metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &key, &weights, &req_pool_entries, &cache_slots_pool,
       &cache_tokens, &candidate_lens, &block_table},
      "LIDU");
}

void CheckLiduOutputs(
    const at::Tensor& query,
    const at::Tensor& source_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& miss_counts) {
  const int64_t batch = query.size(0);
  TORCH_CHECK(
      source_ids.dim() == 3 && source_ids.size(0) == batch &&
          source_ids.size(1) == 1 && source_ids.size(2) == kSparseCount &&
          destination_slots.sizes() == source_ids.sizes() &&
          miss_counts.dim() == 1 && miss_counts.size(0) == batch,
      "LIDU out buffers must be source/destination [B,1,2048], miss_counts [B].");
  for (const at::Tensor* tensor : {&source_ids, &destination_slots, &miss_counts}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt && tensor->device() == query.device() &&
            tensor->is_contiguous(),
        "LIDU out buffers must be contiguous int32 tensors on the query NPU.");
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
LiduDecodeUpdateOutNpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& block_table,
    at::Tensor source_ids,
    at::Tensor destination_slots,
    at::Tensor miss_counts) {
  CheckLiduCommon(
      query, key, weights, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, block_table);
  CheckLiduOutputs(query, source_ids, destination_slots, miss_counts);

  auto keepalive = std::make_tuple(
      query, key, weights, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, block_table, source_ids,
      destination_slots, miss_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnLightningIndexerDecodeUpdateA5,
      keepalive,
      query,
      key,
      weights,
      req_pool_entries,
      cache_slots_pool,
      cache_tokens,
      candidate_lens,
      block_table,
      source_ids,
      destination_slots,
      miss_counts,
      cache_slots_pool);
  return std::make_tuple(
      source_ids, destination_slots, miss_counts, cache_slots_pool);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
LiduDecodeUpdateNpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& block_table) {
  auto options = query.options().dtype(at::kInt);
  auto source_ids = at::empty({query.size(0), 1, kSparseCount}, options);
  auto destination_slots = at::empty_like(source_ids);
  auto miss_counts = at::empty({query.size(0)}, options);
  return LiduDecodeUpdateOutNpu(
      query, key, weights, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, block_table,
      source_ids, destination_slots, miss_counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
LiduDecodeUpdateMeta(
    const at::Tensor& query,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor cache_slots_pool,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  auto options = query.options().dtype(at::kInt);
  return std::make_tuple(
      at::empty({query.size(0), 1, kSparseCount}, options),
      at::empty({query.size(0), 1, kSparseCount}, options),
      at::empty({query.size(0)}, options),
      cache_slots_pool);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
LiduDecodeUpdateOutMeta(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor cache_slots_pool,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor source_ids,
    at::Tensor destination_slots,
    at::Tensor miss_counts) {
  return std::make_tuple(
      source_ids, destination_slots, miss_counts, cache_slots_pool);
}

void CheckLiduCacheUpdateCommon(
    const at::Tensor& topk_indices,
    const at::Tensor& req_pool_entries,
    const at::Tensor& cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens) {
  TORCH_CHECK(
      topk_indices.dim() == 3 && topk_indices.size(0) > 0 &&
          topk_indices.size(1) == 1 &&
          topk_indices.size(2) == kSparseCount,
      "C8 LIDU topk_indices must be int32 [B,1,2048].");
  const int64_t batch = topk_indices.size(0);
  TORCH_CHECK(
      req_pool_entries.dim() == 1 && req_pool_entries.size(0) == batch &&
          cache_tokens.dim() == 1 && cache_tokens.size(0) == batch &&
          candidate_lens.dim() == 1 && candidate_lens.size(0) == batch,
      "C8 LIDU metadata must be int32 [B].");
  TORCH_CHECK(
      cache_slots_pool.dim() == 2 && cache_slots_pool.size(0) > 0 &&
          cache_slots_pool.size(1) > 0 &&
          cache_slots_pool.size(1) <= kMaxSourceCapacity,
      "C8 LIDU cache_slots_pool must be [pool_size,capacity], capacity <= 2^18.");
  for (const at::Tensor* tensor :
       {&topk_indices, &req_pool_entries, &cache_slots_pool,
        &cache_tokens, &candidate_lens}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt,
        "C8 LIDU state-update tensors must be int32.");
  }
  CheckOneDeviceAndContiguous(
      topk_indices,
      {&topk_indices, &req_pool_entries, &cache_slots_pool,
       &cache_tokens, &candidate_lens},
      "C8 LIDU state update");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
LiduCacheUpdateOutNpu(
    const at::Tensor& topk_indices,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    at::Tensor source_ids,
    at::Tensor destination_slots,
    at::Tensor miss_counts) {
  CheckLiduCacheUpdateCommon(
      topk_indices, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens);
  CheckLiduOutputs(
      topk_indices, source_ids, destination_slots, miss_counts);
  auto keepalive = std::make_tuple(
      topk_indices, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens, source_ids,
      destination_slots, miss_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5LiduCacheUpdate,
      keepalive,
      topk_indices,
      req_pool_entries,
      cache_slots_pool,
      cache_tokens,
      candidate_lens,
      source_ids,
      destination_slots,
      miss_counts,
      cache_slots_pool);
  return std::make_tuple(
      source_ids, destination_slots, miss_counts, cache_slots_pool);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
LiduCacheUpdateNpu(
    const at::Tensor& topk_indices,
    const at::Tensor& req_pool_entries,
    at::Tensor cache_slots_pool,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens) {
  auto options = topk_indices.options().dtype(at::kInt);
  auto source_ids = at::empty(topk_indices.sizes(), options);
  auto destination_slots = at::empty(topk_indices.sizes(), options);
  auto miss_counts = at::empty({topk_indices.size(0)}, options);
  return LiduCacheUpdateOutNpu(
      topk_indices, req_pool_entries, cache_slots_pool,
      cache_tokens, candidate_lens,
      source_ids, destination_slots, miss_counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
LiduCacheUpdateMeta(
    const at::Tensor& topk_indices,
    const at::Tensor&,
    at::Tensor cache_slots_pool,
    const at::Tensor&,
    const at::Tensor&) {
  auto options = topk_indices.options().dtype(at::kInt);
  return std::make_tuple(
      at::empty(topk_indices.sizes(), options),
      at::empty(topk_indices.sizes(), options),
      at::empty({topk_indices.size(0)}, options),
      cache_slots_pool);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
LiduCacheUpdateOutMeta(
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor cache_slots_pool,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor source_ids,
    at::Tensor destination_slots,
    at::Tensor miss_counts) {
  return std::make_tuple(
      source_ids, destination_slots, miss_counts, cache_slots_pool);
}

void CheckScatterInputs(
    const at::Tensor& hbm_kpe,
    const at::Tensor& hbm_ckv,
    const at::Tensor& dram_kpe,
    const at::Tensor& dram_ckv,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts) {
  TORCH_CHECK(
      hbm_kpe.dim() == 3 && hbm_kpe.size(1) == kBlockSize &&
          hbm_kpe.size(2) == kKpeDim,
      "HBM KPE must be [blocks,128,64].");
  TORCH_CHECK(
      hbm_ckv.dim() == 3 && hbm_ckv.size(1) == kBlockSize &&
          hbm_ckv.size(2) == kCkvDim,
      "HBM CKV must be [blocks,128,512].");
  TORCH_CHECK(
      dram_kpe.dim() == 3 && dram_kpe.size(1) == kBlockSize &&
          dram_kpe.size(2) == kKpeDim &&
          dram_ckv.dim() == 3 && dram_ckv.size(1) == kBlockSize &&
          dram_ckv.size(2) == kCkvDim,
      "DRAM KPE/CKV must be [blocks,128,64/512].");
  TORCH_CHECK(
      hbm_kpe.size(0) == hbm_ckv.size(0) &&
          dram_kpe.size(0) == dram_ckv.size(0),
      "CKV/KPE block counts must agree in each memory tier.");
  TORCH_CHECK(
      hbm_block_table.dim() == 2 && dram_block_table.dim() == 2 &&
          source_token_ids.dim() == 2 && destination_slots.dim() == 2 &&
          copy_counts.dim() == 1 &&
          source_token_ids.sizes() == destination_slots.sizes() &&
          source_token_ids.size(0) == copy_counts.size(0) &&
          hbm_block_table.size(0) == copy_counts.size(0) &&
          dram_block_table.size(0) == copy_counts.size(0) &&
          hbm_block_table.size(1) > 0 && dram_block_table.size(1) > 0 &&
          source_token_ids.size(1) > 0 && source_token_ids.size(1) <= 65536 &&
          dram_block_table.size(1) * kBlockSize <= kMaxSourceCapacity,
      "SCATTER metadata batch dimensions are inconsistent.");
  const auto dtype = hbm_kpe.scalar_type();
  TORCH_CHECK(
      dtype == at::kBFloat16 || dtype == at::kHalf,
      "SCATTER supports bf16/fp16.");
  for (const at::Tensor* tensor : {&hbm_ckv, &dram_kpe, &dram_ckv}) {
    TORCH_CHECK(tensor->scalar_type() == dtype, "All SCATTER KV dtypes must match.");
  }
  for (const at::Tensor* tensor :
       {&hbm_block_table, &dram_block_table, &source_token_ids,
        &destination_slots, &copy_counts}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt, "SCATTER metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      hbm_kpe,
      {&hbm_kpe, &hbm_ckv, &dram_kpe, &dram_ckv, &hbm_block_table,
       &dram_block_table, &source_token_ids, &destination_slots, &copy_counts},
      "SCATTER");
}

std::tuple<at::Tensor, at::Tensor> ScatterCopyNpu(
    at::Tensor hbm_kpe,
    at::Tensor hbm_ckv,
    const at::Tensor& dram_kpe,
    const at::Tensor& dram_ckv,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts) {
  CheckScatterInputs(
      hbm_kpe, hbm_ckv, dram_kpe, dram_ckv, hbm_block_table,
      dram_block_table, source_token_ids, destination_slots, copy_counts);
  auto keepalive = std::make_tuple(
      hbm_kpe, hbm_ckv, dram_kpe, dram_ckv, hbm_block_table,
      dram_block_table, source_token_ids, destination_slots, copy_counts);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5KvcacheScatterCopy,
      keepalive,
      hbm_kpe,
      hbm_ckv,
      dram_kpe,
      dram_ckv,
      hbm_block_table,
      dram_block_table,
      source_token_ids,
      destination_slots,
      copy_counts,
      hbm_kpe,
      hbm_ckv);
  return std::make_tuple(hbm_kpe, hbm_ckv);
}

std::tuple<at::Tensor, at::Tensor> ScatterCopyMeta(
    at::Tensor hbm_kpe,
    at::Tensor hbm_ckv,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  return std::make_tuple(hbm_kpe, hbm_ckv);
}

int64_t CopyMetadataCapacity(const at::Tensor& tensor) {
  if (tensor.dim() == 2) {
    return tensor.size(1);
  }
  TORCH_CHECK(
      tensor.dim() == 3 && tensor.size(1) == 1,
      "packed SCATTER source/destination metadata must be [B,K] or [B,1,K].");
  return tensor.size(2);
}

int64_t AttentionCapacity(int64_t max_tail_tokens) {
  TORCH_CHECK(
      max_tail_tokens >= 0 &&
          max_tail_tokens <= kMaxSourceCapacity - kSparseCount,
      "packed SCATTER max_tail_tokens must be in [0,260096].");
  return kSparseCount + max_tail_tokens;
}

void CheckPackedScatterInputs(
    const at::Tensor& hbm_kv_bytes,
    const at::Tensor& dram_kv_bytes,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& actual_seq_lengths_kv) {
  TORCH_CHECK(
      hbm_kv_bytes.dim() == 4 && hbm_kv_bytes.size(0) > 0 &&
          hbm_kv_bytes.size(1) == kBlockSize &&
          hbm_kv_bytes.size(2) == 1 &&
          hbm_kv_bytes.size(3) == kPackedKvDim &&
          dram_kv_bytes.dim() == 4 && dram_kv_bytes.size(0) > 0 &&
          dram_kv_bytes.size(1) == kBlockSize &&
          dram_kv_bytes.size(2) == 1 &&
          dram_kv_bytes.size(3) == kPackedKvDim,
      "packed SCATTER KV byte views must be [blocks,128,1,656].");
  TORCH_CHECK(
      hbm_kv_bytes.scalar_type() == at::kChar &&
          dram_kv_bytes.scalar_type() == at::kChar,
      "packed SCATTER expects int8 byte views of C8 KV caches.");
  TORCH_CHECK(copy_counts.dim() == 1, "packed SCATTER copy_counts must be int32[B].");
  const int64_t batch = copy_counts.size(0);
  TORCH_CHECK(
      batch > 0 && cache_tokens.dim() == 1 &&
          cache_tokens.size(0) == batch && candidate_lens.dim() == 1 &&
          candidate_lens.size(0) == batch &&
          actual_seq_lengths_kv.dim() == 1 &&
          actual_seq_lengths_kv.size(0) == batch &&
          hbm_block_table.dim() == 2 &&
          hbm_block_table.size(0) == batch &&
          hbm_block_table.size(1) > 0 &&
          dram_block_table.dim() == 2 &&
          dram_block_table.size(0) == batch &&
          dram_block_table.size(1) > 0 &&
          source_token_ids.size(0) == batch &&
          destination_slots.size(0) == batch &&
          source_token_ids.sizes() == destination_slots.sizes() &&
          CopyMetadataCapacity(source_token_ids) == kSparseCount &&
          dram_block_table.size(1) * kBlockSize <= kMaxSourceCapacity,
      "packed SCATTER metadata shapes are inconsistent.");
  for (const at::Tensor* tensor :
       {&hbm_block_table, &dram_block_table, &source_token_ids,
        &destination_slots, &copy_counts, &cache_tokens,
        &candidate_lens, &actual_seq_lengths_kv}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt,
        "packed SCATTER metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      hbm_kv_bytes,
      {&hbm_kv_bytes, &dram_kv_bytes, &hbm_block_table,
       &dram_block_table, &source_token_ids, &destination_slots,
       &copy_counts, &cache_tokens, &candidate_lens,
       &actual_seq_lengths_kv},
      "packed SCATTER");
}

void CheckPackedScatterOutputs(
    const at::Tensor& hbm_kv_bytes,
    const at::Tensor& attention_slots,
    const at::Tensor& resident_seq_lengths,
    int64_t expected_batch,
    int64_t max_tail_tokens) {
  const int64_t expected_capacity = AttentionCapacity(max_tail_tokens);
  TORCH_CHECK(
      attention_slots.dim() == 3 &&
          attention_slots.size(0) == expected_batch &&
          attention_slots.size(1) == 1 &&
          attention_slots.size(2) == expected_capacity &&
          resident_seq_lengths.dim() == 1 &&
          resident_seq_lengths.size(0) == expected_batch,
      "packed SCATTER outputs must be attention_slots [B,1,2048+max_tail_tokens] "
      "and resident_seq_lengths [B].");
  for (const at::Tensor* tensor : {&attention_slots, &resident_seq_lengths}) {
    TORCH_CHECK(
        tensor->scalar_type() == at::kInt &&
            tensor->device() == hbm_kv_bytes.device() &&
            tensor->is_contiguous(),
        "packed SCATTER metadata outputs must be contiguous int32 tensors.");
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
PackedScatterCopyOutNpu(
    at::Tensor hbm_kv_bytes,
    const at::Tensor& dram_kv_bytes,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& actual_seq_lengths_kv,
    int64_t max_tail_tokens,
    at::Tensor attention_slots,
    at::Tensor resident_seq_lengths) {
  CheckPackedScatterInputs(
      hbm_kv_bytes, dram_kv_bytes, hbm_block_table, dram_block_table,
      source_token_ids, destination_slots, copy_counts, cache_tokens,
      candidate_lens, actual_seq_lengths_kv);
  CheckPackedScatterOutputs(
      hbm_kv_bytes, attention_slots, resident_seq_lengths,
      copy_counts.size(0), max_tail_tokens);
  auto keepalive = std::make_tuple(
      hbm_kv_bytes, dram_kv_bytes, hbm_block_table, dram_block_table,
      source_token_ids, destination_slots, copy_counts, cache_tokens,
      candidate_lens, actual_seq_lengths_kv, attention_slots,
      resident_seq_lengths);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5PackedKvcacheScatterCopy,
      keepalive,
      hbm_kv_bytes,
      dram_kv_bytes,
      hbm_block_table,
      dram_block_table,
      source_token_ids,
      destination_slots,
      copy_counts,
      cache_tokens,
      candidate_lens,
      actual_seq_lengths_kv,
      attention_slots,
      resident_seq_lengths,
      hbm_kv_bytes,
      attention_slots,
      resident_seq_lengths);
  return std::make_tuple(
      hbm_kv_bytes, attention_slots, resident_seq_lengths);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
PackedScatterCopyNpu(
    at::Tensor hbm_kv_bytes,
    const at::Tensor& dram_kv_bytes,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    const at::Tensor& source_token_ids,
    const at::Tensor& destination_slots,
    const at::Tensor& copy_counts,
    const at::Tensor& cache_tokens,
    const at::Tensor& candidate_lens,
    const at::Tensor& actual_seq_lengths_kv,
    int64_t max_tail_tokens) {
  auto options = copy_counts.options().dtype(at::kInt);
  auto attention_slots = at::empty(
      {copy_counts.size(0), 1, AttentionCapacity(max_tail_tokens)}, options);
  auto resident_seq_lengths = at::empty({copy_counts.size(0)}, options);
  return PackedScatterCopyOutNpu(
      hbm_kv_bytes, dram_kv_bytes, hbm_block_table, dram_block_table,
      source_token_ids, destination_slots, copy_counts, cache_tokens,
      candidate_lens, actual_seq_lengths_kv, max_tail_tokens,
      attention_slots, resident_seq_lengths);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
PackedScatterCopyMeta(
    at::Tensor hbm_kv_bytes,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor& copy_counts,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    int64_t max_tail_tokens) {
  auto options = copy_counts.options().dtype(at::kInt);
  return std::make_tuple(
      hbm_kv_bytes,
      at::empty(
          {copy_counts.size(0), 1, AttentionCapacity(max_tail_tokens)},
          options),
      at::empty({copy_counts.size(0)}, options));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
PackedScatterCopyOutMeta(
    at::Tensor hbm_kv_bytes,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor& copy_counts,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    int64_t max_tail_tokens,
    at::Tensor attention_slots,
    at::Tensor resident_seq_lengths) {
  CheckPackedScatterOutputs(
      hbm_kv_bytes, attention_slots, resident_seq_lengths,
      copy_counts.size(0), max_tail_tokens);
  return std::make_tuple(
      hbm_kv_bytes, attention_slots, resident_seq_lengths);
}

void CheckAttentionInputs(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_slots,
    const at::Tensor& cache_tokens,
    const at::Tensor& block_table,
    const at::Tensor& actual_q,
    const at::Tensor& actual_kv,
    const at::Tensor& query_rope,
    const at::Tensor& key_rope) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 && query.size(2) == kCkvDim &&
          query.size(1) >= 1 && query.size(1) <= 64,
      "SFA query must be [B,N,512] with 1 <= N <= 64.");
  const int64_t batch = query.size(0);
  const int64_t heads = query.size(1);
  TORCH_CHECK(
      key.dim() == 4 && key.size(1) == kBlockSize && key.size(2) == 1 &&
          key.size(3) == kCkvDim && value.sizes() == key.sizes(),
      "SFA key/value must be [blocks,128,1,512].");
  TORCH_CHECK(
      value.data_ptr() == key.data_ptr(),
      "SFA is the MLA path and requires value to alias key storage.");
  TORCH_CHECK(
      key_rope.dim() == 4 && key_rope.size(0) == key.size(0) &&
          key_rope.size(1) == kBlockSize && key_rope.size(2) == 1 &&
          key_rope.size(3) == kKpeDim &&
          query_rope.dim() == 3 && query_rope.size(0) == batch &&
          query_rope.size(1) == heads && query_rope.size(2) == kKpeDim,
      "SFA RoPE tensors must be [blocks,128,1,64] and [B,N,64].");
  TORCH_CHECK(
      sparse_slots.dim() == 3 && sparse_slots.size(0) == batch &&
          sparse_slots.size(1) == 1 && sparse_slots.size(2) == kSparseCount &&
          cache_tokens.dim() == 1 && cache_tokens.size(0) == batch &&
          actual_q.dim() == 1 && actual_q.size(0) == batch &&
          actual_kv.dim() == 1 && actual_kv.size(0) == batch &&
          block_table.dim() == 2 && block_table.size(0) == batch &&
          block_table.size(1) > 0 &&
          block_table.size(1) * kBlockSize <= kMaxSourceCapacity,
      "SFA metadata shapes are invalid.");
  const auto dtype = query.scalar_type();
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf, "SFA supports bf16/fp16.");
  for (const at::Tensor* tensor : {&key, &value, &query_rope, &key_rope}) {
    TORCH_CHECK(tensor->scalar_type() == dtype, "All SFA floating dtypes must match.");
  }
  for (const at::Tensor* tensor :
       {&sparse_slots, &cache_tokens, &block_table, &actual_q, &actual_kv}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt, "SFA metadata must be int32.");
  }
  CheckOneDeviceAndContiguous(
      query,
      {&query, &key, &value, &sparse_slots, &cache_tokens, &block_table,
       &actual_q, &actual_kv, &query_rope, &key_rope},
      "SFA");
}

at::Tensor SparseAndTailAttentionNpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_slots,
    const at::Tensor& cache_tokens,
    const at::Tensor& block_table,
    const at::Tensor& actual_q,
    const at::Tensor& actual_kv,
    const at::Tensor& query_rope,
    const at::Tensor& key_rope,
    double scale_value) {
  CheckAttentionInputs(
      query, key, value, sparse_slots, cache_tokens, block_table,
      actual_q, actual_kv, query_rope, key_rope);
  auto output = at::empty_like(query);
  auto softmax_max = at::empty({1}, query.options().dtype(at::kFloat));
  auto softmax_sum = at::empty({1}, query.options().dtype(at::kFloat));
  std::string query_layout = "TND";
  std::string kv_layout = "PA_BSND";
  char* query_layout_ptr = const_cast<char*>(query_layout.c_str());
  char* kv_layout_ptr = const_cast<char*>(kv_layout.c_str());
  constexpr int64_t kSparseBlockSize = 1;
  constexpr int64_t kSparseMode = 3;
  constexpr int64_t kAllTokens = std::numeric_limits<int64_t>::max();
  constexpr int64_t kAttentionMode = 2;
  constexpr bool kReturnSoftmaxLse = false;
  float scale_value_float = static_cast<float>(scale_value);
  auto keepalive = std::make_tuple(
      query, key, value, sparse_slots, block_table, actual_q, actual_kv,
      query_rope, key_rope, cache_tokens, output, softmax_max, softmax_sum);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5SparseAndTailAttention,
      keepalive,
      query,
      key,
      value,
      sparse_slots,
      block_table,
      actual_q,
      actual_kv,
      query_rope,
      key_rope,
      cache_tokens,
      scale_value_float,
      kSparseBlockSize,
      query_layout_ptr,
      kv_layout_ptr,
      kSparseMode,
      kAllTokens,
      kAllTokens,
      kAttentionMode,
      kReturnSoftmaxLse,
      output,
      softmax_max,
      softmax_sum);
  return output;
}

at::Tensor SparseAndTailAttentionMeta(
    const at::Tensor& query,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double) {
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) > 0 && query.size(2) == kCkvDim &&
          query.size(1) >= 1 && query.size(1) <= 64,
      "SFA query must be [B,N,512] with 1 <= N <= 64.");
  return at::empty_like(query);
}
}  // namespace

TORCH_LIBRARY(nanovllm_dsa, m) {
  m.def(
      "lidu_decode_update(Tensor query, Tensor key, Tensor weights, "
      "Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
      "Tensor cache_tokens, Tensor candidate_lens, Tensor block_table) "
      "-> (Tensor, Tensor, Tensor, Tensor(a!))");
  m.def(
      "lidu_decode_update_out(Tensor query, Tensor key, Tensor weights, "
      "Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
      "Tensor cache_tokens, Tensor candidate_lens, Tensor block_table, "
      "Tensor(b!) source_ids, Tensor(c!) destination_slots, "
      "Tensor(d!) miss_counts) "
      "-> (Tensor(b!), Tensor(c!), Tensor(d!), Tensor(a!))");
  m.def(
      "lidu_cache_update(Tensor topk_indices, Tensor req_pool_entries, "
      "Tensor(a!) cache_slots_pool, Tensor cache_tokens, "
      "Tensor candidate_lens) "
      "-> (Tensor, Tensor, Tensor, Tensor(a!))");
  m.def(
      "lidu_cache_update_out(Tensor topk_indices, "
      "Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
      "Tensor cache_tokens, Tensor candidate_lens, "
      "Tensor(b!) source_ids, Tensor(c!) destination_slots, "
      "Tensor(d!) miss_counts) "
      "-> (Tensor(b!), Tensor(c!), Tensor(d!), Tensor(a!))");
  m.def(
      "lidu_decode_update_c8(Tensor query, Tensor key, Tensor weights, "
      "Tensor query_dequant_scale, Tensor key_dequant_scale, "
      "Tensor actual_seq_lengths_query, Tensor req_pool_entries, "
      "Tensor(a!) cache_slots_pool, Tensor cache_tokens, "
      "Tensor candidate_lens, Tensor block_table) "
      "-> (Tensor, Tensor, Tensor, Tensor(a!))");
  m.def(
      "lidu_decode_update_c8_out(Tensor query, Tensor key, "
      "Tensor weights, Tensor query_dequant_scale, "
      "Tensor key_dequant_scale, Tensor actual_seq_lengths_query, "
      "Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
      "Tensor cache_tokens, Tensor candidate_lens, "
      "Tensor block_table, Tensor(b!) source_ids, "
      "Tensor(c!) destination_slots, Tensor(d!) miss_counts) "
      "-> (Tensor(b!), Tensor(c!), Tensor(d!), Tensor(a!))");
  m.def(
      "scatter_copy(Tensor(a!) hbm_kpe, Tensor(b!) hbm_ckv, "
      "Tensor dram_kpe, Tensor dram_ckv, Tensor hbm_block_table, "
      "Tensor dram_block_table, Tensor source_token_ids, "
      "Tensor destination_slots, Tensor copy_counts) "
      "-> (Tensor(a!), Tensor(b!))");
  m.def(
      "packed_scatter_copy(Tensor(a!) hbm_kv_bytes, Tensor dram_kv_bytes, "
      "Tensor hbm_block_table, Tensor dram_block_table, "
      "Tensor source_token_ids, Tensor destination_slots, "
      "Tensor copy_counts, Tensor cache_tokens, Tensor candidate_lens, "
      "Tensor actual_seq_lengths_kv, int max_tail_tokens) "
      "-> (Tensor(a!), Tensor, Tensor)");
  m.def(
      "packed_scatter_copy_out(Tensor(a!) hbm_kv_bytes, Tensor dram_kv_bytes, "
      "Tensor hbm_block_table, Tensor dram_block_table, "
      "Tensor source_token_ids, Tensor destination_slots, "
      "Tensor copy_counts, Tensor cache_tokens, Tensor candidate_lens, "
      "Tensor actual_seq_lengths_kv, int max_tail_tokens, "
      "Tensor(b!) attention_slots, Tensor(c!) resident_seq_lengths) "
      "-> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def(
      "sparse_and_tail_attention(Tensor query, Tensor key, Tensor value, "
      "Tensor sparse_slots, Tensor cache_tokens, Tensor block_table, "
      "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv, "
      "Tensor query_rope, Tensor key_rope, float scale_value) -> Tensor");
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl("lidu_decode_update", &LiduDecodeUpdateNpu);
  m.impl("lidu_decode_update_out", &LiduDecodeUpdateOutNpu);
  m.impl("lidu_cache_update", &LiduCacheUpdateNpu);
  m.impl("lidu_cache_update_out", &LiduCacheUpdateOutNpu);
  m.impl("scatter_copy", &ScatterCopyNpu);
  m.impl("packed_scatter_copy", &PackedScatterCopyNpu);
  m.impl("packed_scatter_copy_out", &PackedScatterCopyOutNpu);
  m.impl("sparse_and_tail_attention", &SparseAndTailAttentionNpu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl("lidu_decode_update", &LiduDecodeUpdateMeta);
  m.impl("lidu_decode_update_out", &LiduDecodeUpdateOutMeta);
  m.impl("lidu_cache_update", &LiduCacheUpdateMeta);
  m.impl("lidu_cache_update_out", &LiduCacheUpdateOutMeta);
  m.impl("scatter_copy", &ScatterCopyMeta);
  m.impl("packed_scatter_copy", &PackedScatterCopyMeta);
  m.impl("packed_scatter_copy_out", &PackedScatterCopyOutMeta);
  m.impl("sparse_and_tail_attention", &SparseAndTailAttentionMeta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
