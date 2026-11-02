#include <initializer_list>
#include <limits>
#include <tuple>

#include <torch/extension.h>
#include <torch/library.h>

#include "lidu_fused_batch_get_prepare_mtp_op_api.h"

namespace {
constexpr int64_t SPARSE_COUNT = 2048;
constexpr int64_t MIN_QUERIES_PER_REQUEST = 1;
constexpr int64_t MAX_QUERIES_PER_REQUEST = 4;
constexpr int64_t MTP_MISS_CAPACITY =
    SPARSE_COUNT * MAX_QUERIES_PER_REQUEST;
constexpr int64_t MAX_SOURCE_CAPACITY = (1 << 18) + 32;
constexpr int64_t MAX_CACHE_TOKENS = (1U << 17) + 20U * 128U;
constexpr int64_t BITMAP_BLOCK_WORDS = 8;
constexpr int64_t MISS_SLOT_BITMAP_WORDS =
    ((MAX_CACHE_TOKENS + 31) / 32 + BITMAP_BLOCK_WORDS - 1) /
    BITMAP_BLOCK_WORDS * BITMAP_BLOCK_WORDS;
constexpr int64_t PROFILE_LANES = 8;
constexpr int64_t PROFILE_FIELDS = 32;
constexpr int64_t BLOCK_SIZE = 128;

void CheckOneDeviceAndContiguous(
    const at::Tensor &reference,
    std::initializer_list<const at::Tensor *> tensors)
{
    TORCH_CHECK(
        reference.device().is_privateuseone(),
        "A5LiduFusedBatchGetPrepareMtp inputs must be on NPU");
    for (const at::Tensor *tensor : tensors) {
        TORCH_CHECK(
            tensor->device() == reference.device() && tensor->is_contiguous(),
            "A5LiduFusedBatchGetPrepareMtp tensors must be contiguous and "
            "on one NPU");
    }
}

void CheckInputs(
    const at::Tensor &topkIndices,
    const at::Tensor &actualSeqLengthsQuery,
    const at::Tensor &reqPoolEntries,
    const at::Tensor &tokenToSlotPool,
    const at::Tensor &slotToTokenPool,
    const at::Tensor &evictionCursors,
    const at::Tensor &cacheTokens,
    const at::Tensor &candidateLens,
    const at::Tensor &missSlots,
    const at::Tensor &missSlotBitmap,
    const at::Tensor &residentBlockTable,
    const at::Tensor &logicalSeqLengths,
    const at::Tensor &profileEnabled,
    int64_t layerId,
    int64_t numLayers,
    int64_t maxNumReqs,
    int64_t qcap)
{
    TORCH_CHECK(
        topkIndices.dim() == 3 && topkIndices.size(0) > 0 &&
        topkIndices.size(1) == 1 && topkIndices.size(2) == SPARSE_COUNT,
        "A5LiduFusedBatchGetPrepareMtp topk_indices must be int32 "
        "[T,1,2048]");
    TORCH_CHECK(
        actualSeqLengthsQuery.dim() == 1 &&
        actualSeqLengthsQuery.size(0) > 0,
        "A5LiduFusedBatchGetPrepareMtp actual_seq_lengths_query must be "
        "int32 [B]");
    const int64_t batch = actualSeqLengthsQuery.size(0);
    const int64_t packedQueries = topkIndices.size(0);
    TORCH_CHECK(
        packedQueries >= batch * MIN_QUERIES_PER_REQUEST &&
        packedQueries <= batch * MAX_QUERIES_PER_REQUEST,
        "A5LiduFusedBatchGetPrepareMtp requires 1--4 packed queries per "
        "request");
    TORCH_CHECK(
        reqPoolEntries.dim() == 1 && reqPoolEntries.size(0) == batch &&
        cacheTokens.dim() == 1 && cacheTokens.size(0) == batch &&
        candidateLens.dim() == 1 && candidateLens.size(0) == batch &&
        logicalSeqLengths.dim() == 1 &&
        logicalSeqLengths.size(0) == batch &&
        profileEnabled.dim() == 1 && profileEnabled.size(0) == 1,
        "A5LiduFusedBatchGetPrepareMtp request metadata must be int32 [B], "
        "with profile_enabled int32 [1]");
    TORCH_CHECK(
        missSlots.dim() == 2 && missSlots.size(0) == batch &&
        missSlots.size(1) == SPARSE_COUNT,
        "A5LiduFusedBatchGetPrepareMtp miss_slots must be int32 [B,2048]");
    TORCH_CHECK(
        missSlotBitmap.dim() == 2 && missSlotBitmap.size(0) == batch &&
        missSlotBitmap.size(1) == MISS_SLOT_BITMAP_WORDS,
        "A5LiduFusedBatchGetPrepareMtp miss_slot_bitmap must be int32 "
        "[B,4096]");
    TORCH_CHECK(
        tokenToSlotPool.dim() == 2 && tokenToSlotPool.size(0) > 0 &&
        tokenToSlotPool.size(1) > 0 &&
        tokenToSlotPool.size(1) <= MAX_SOURCE_CAPACITY,
        "A5LiduFusedBatchGetPrepareMtp token_to_slot_pool must be "
        "[pool,source_capacity]");
    TORCH_CHECK(
        slotToTokenPool.dim() == 2 &&
        slotToTokenPool.size(0) == tokenToSlotPool.size(0) &&
        slotToTokenPool.size(1) >= SPARSE_COUNT,
        "A5LiduFusedBatchGetPrepareMtp slot_to_token_pool must be "
        "[pool,resident_capacity >= 2048]");
    TORCH_CHECK(
        evictionCursors.dim() == 1 &&
        evictionCursors.size(0) == tokenToSlotPool.size(0),
        "A5LiduFusedBatchGetPrepareMtp eviction_cursors must be int32 [pool]");
    TORCH_CHECK(
        residentBlockTable.dim() == 2 &&
        residentBlockTable.size(0) == batch &&
        residentBlockTable.size(1) > 0,
        "A5LiduFusedBatchGetPrepareMtp resident_block_table must be "
        "int32 [B,resident_blocks]");
    TORCH_CHECK(
        layerId >= 0 && numLayers > 0 && layerId < numLayers &&
        maxNumReqs >= batch && qcap > 0 &&
        static_cast<uint64_t>(maxNumReqs) *
                static_cast<uint64_t>(numLayers) *
                static_cast<uint64_t>(qcap) <=
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) &&
        static_cast<uint64_t>(residentBlockTable.size(1)) * BLOCK_SIZE <=
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
        "A5LiduFusedBatchGetPrepareMtp AccessTable geometry is invalid");

    for (const at::Tensor *tensor : {
            &topkIndices, &actualSeqLengthsQuery, &reqPoolEntries,
            &tokenToSlotPool, &slotToTokenPool, &evictionCursors,
            &cacheTokens, &candidateLens, &missSlots, &missSlotBitmap,
            &residentBlockTable, &logicalSeqLengths, &profileEnabled}) {
        TORCH_CHECK(
            tensor->scalar_type() == at::kInt,
            "A5LiduFusedBatchGetPrepareMtp supports only int32 inputs");
    }
    CheckOneDeviceAndContiguous(
        topkIndices,
        {&topkIndices, &actualSeqLengthsQuery, &reqPoolEntries,
         &tokenToSlotPool, &slotToTokenPool, &evictionCursors,
         &cacheTokens, &candidateLens, &missSlots, &missSlotBitmap,
         &residentBlockTable, &logicalSeqLengths, &profileEnabled});
}

void CheckOutputs(
    const at::Tensor &topkIndices,
    int64_t batch,
    const at::Tensor &destinationSlots,
    const at::Tensor &missCounts,
    const at::Tensor &indices,
    const at::Tensor &offsets,
    const at::Tensor &outputIndices,
    const at::Tensor &residentSeqLengths,
    const at::Tensor &profiling)
{
    const int64_t descriptorCapacity = batch * MTP_MISS_CAPACITY;
    TORCH_CHECK(
        destinationSlots.sizes() == topkIndices.sizes(),
        "A5LiduFusedBatchGetPrepareMtp destination_slots must be "
        "int32 [T,1,2048]");
    TORCH_CHECK(
        missCounts.dim() == 1 && missCounts.size(0) == batch + 1 &&
        indices.dim() == 1 && indices.numel() == descriptorCapacity &&
        offsets.dim() == 1 && offsets.numel() == descriptorCapacity &&
        outputIndices.dim() == 1 &&
        outputIndices.numel() == descriptorCapacity &&
        residentSeqLengths.dim() == 1 &&
        residentSeqLengths.size(0) == batch,
        "A5LiduFusedBatchGetPrepareMtp descriptors must be int32 [B*8192], "
        "with miss_counts int32 [B+1] and resident_seq_lengths int32 [B]");
    for (const at::Tensor *tensor : {
            &destinationSlots, &missCounts, &indices, &offsets,
            &outputIndices, &residentSeqLengths}) {
        TORCH_CHECK(
            tensor->scalar_type() == at::kInt &&
            tensor->device() == topkIndices.device() &&
            tensor->is_contiguous(),
            "A5LiduFusedBatchGetPrepareMtp outputs must be contiguous int32 "
            "tensors on the input NPU");
    }
    TORCH_CHECK(
        profiling.dim() == 3 && profiling.size(0) == batch &&
        profiling.size(1) == PROFILE_LANES &&
        profiling.size(2) == PROFILE_FIELDS &&
        profiling.scalar_type() == at::kLong &&
        profiling.device() == topkIndices.device() &&
        profiling.is_contiguous(),
        "A5LiduFusedBatchGetPrepareMtp profiling must be contiguous int64 "
        "[B,8,32] on the input NPU");
}

using OutputTuple = std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

OutputTuple LiduFusedBatchGetPrepareMtpOutNpu(
    const at::Tensor &topkIndices,
    const at::Tensor &actualSeqLengthsQuery,
    const at::Tensor &reqPoolEntries,
    at::Tensor tokenToSlotPool,
    at::Tensor slotToTokenPool,
    at::Tensor evictionCursors,
    const at::Tensor &cacheTokens,
    const at::Tensor &candidateLens,
    const at::Tensor &missSlots,
    const at::Tensor &missSlotBitmap,
    const at::Tensor &residentBlockTable,
    const at::Tensor &logicalSeqLengths,
    const at::Tensor &profileEnabled,
    at::Tensor destinationSlots,
    at::Tensor missCounts,
    at::Tensor indices,
    at::Tensor offsets,
    at::Tensor outputIndices,
    at::Tensor residentSeqLengths,
    at::Tensor profiling,
    int64_t layerId,
    int64_t numLayers,
    int64_t maxNumReqs,
    int64_t qcap)
{
    CheckInputs(
        topkIndices, actualSeqLengthsQuery, reqPoolEntries,
        tokenToSlotPool, slotToTokenPool, evictionCursors, cacheTokens,
        candidateLens, missSlots, missSlotBitmap, residentBlockTable,
        logicalSeqLengths, profileEnabled, layerId, numLayers, maxNumReqs,
        qcap);
    const int64_t batch = actualSeqLengthsQuery.size(0);
    CheckOutputs(
        topkIndices, batch, destinationSlots, missCounts, indices, offsets,
        outputIndices, residentSeqLengths, profiling);
    auto keepalive = std::make_tuple(
        topkIndices, actualSeqLengthsQuery, reqPoolEntries, tokenToSlotPool,
        slotToTokenPool, evictionCursors, cacheTokens, candidateLens,
        missSlots, missSlotBitmap, residentBlockTable, logicalSeqLengths,
        profileEnabled, destinationSlots, missCounts, indices, offsets,
        outputIndices, residentSeqLengths, profiling);
    EXEC_LIDU_FUSED_BATCH_GET_PREPARE_MTP_OP(
        aclnnA5LiduFusedBatchGetPrepareMtp,
        keepalive,
        topkIndices,
        actualSeqLengthsQuery,
        reqPoolEntries,
        tokenToSlotPool,
        slotToTokenPool,
        evictionCursors,
        cacheTokens,
        candidateLens,
        missSlots,
        missSlotBitmap,
        residentBlockTable,
        logicalSeqLengths,
        profileEnabled,
        layerId,
        numLayers,
        maxNumReqs,
        qcap,
        destinationSlots,
        missCounts,
        indices,
        offsets,
        outputIndices,
        residentSeqLengths,
        tokenToSlotPool,
        slotToTokenPool,
        evictionCursors,
        profiling);
    return std::make_tuple(
        destinationSlots, missCounts, indices, offsets, outputIndices,
        residentSeqLengths);
}

OutputTuple LiduFusedBatchGetPrepareMtpNpu(
    const at::Tensor &topkIndices,
    const at::Tensor &actualSeqLengthsQuery,
    const at::Tensor &reqPoolEntries,
    at::Tensor tokenToSlotPool,
    at::Tensor slotToTokenPool,
    at::Tensor evictionCursors,
    const at::Tensor &cacheTokens,
    const at::Tensor &candidateLens,
    const at::Tensor &missSlots,
    const at::Tensor &missSlotBitmap,
    const at::Tensor &residentBlockTable,
    const at::Tensor &logicalSeqLengths,
    const at::Tensor &profileEnabled,
    int64_t layerId,
    int64_t numLayers,
    int64_t maxNumReqs,
    int64_t qcap)
{
    const auto intOptions = topkIndices.options().dtype(at::kInt);
    const int64_t batch = actualSeqLengthsQuery.size(0);
    const int64_t descriptorCapacity = batch * MTP_MISS_CAPACITY;
    auto profiling = at::zeros(
        {batch, PROFILE_LANES, PROFILE_FIELDS},
        topkIndices.options().dtype(at::kLong));
    return LiduFusedBatchGetPrepareMtpOutNpu(
        topkIndices, actualSeqLengthsQuery, reqPoolEntries,
        tokenToSlotPool, slotToTokenPool, evictionCursors, cacheTokens,
        candidateLens, missSlots, missSlotBitmap, residentBlockTable,
        logicalSeqLengths, profileEnabled,
        at::empty(topkIndices.sizes(), intOptions),
        at::empty({batch + 1}, intOptions),
        at::empty({descriptorCapacity}, intOptions),
        at::empty({descriptorCapacity}, intOptions),
        at::empty({descriptorCapacity}, intOptions),
        at::empty({batch}, intOptions), profiling, layerId, numLayers,
        maxNumReqs, qcap);
}

OutputTuple LiduFusedBatchGetPrepareMtpOutMeta(
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    at::Tensor destinationSlots,
    at::Tensor missCounts,
    at::Tensor indices,
    at::Tensor offsets,
    at::Tensor outputIndices,
    at::Tensor residentSeqLengths,
    at::Tensor,
    int64_t,
    int64_t,
    int64_t,
    int64_t)
{
    return std::make_tuple(
        destinationSlots, missCounts, indices, offsets, outputIndices,
        residentSeqLengths);
}

OutputTuple LiduFusedBatchGetPrepareMtpMeta(
    const at::Tensor &topkIndices,
    const at::Tensor &actualSeqLengthsQuery,
    const at::Tensor &,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    int64_t,
    int64_t,
    int64_t,
    int64_t)
{
    const auto options = topkIndices.options().dtype(at::kInt);
    const int64_t batch = actualSeqLengthsQuery.size(0);
    const int64_t descriptorCapacity = batch * MTP_MISS_CAPACITY;
    return std::make_tuple(
        at::empty(topkIndices.sizes(), options),
        at::empty({batch + 1}, options),
        at::empty({descriptorCapacity}, options),
        at::empty({descriptorCapacity}, options),
        at::empty({descriptorCapacity}, options),
        at::empty({batch}, options));
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(nanovllm_dsa, m) {
    m.def(
        "lidu_fused_batch_get_prepare_mtp(Tensor topk_indices, "
        "Tensor actual_seq_lengths_query, Tensor req_pool_entries, "
        "Tensor(a!) token_to_slot_pool, Tensor(b!) slot_to_token_pool, "
        "Tensor(c!) eviction_cursors, Tensor cache_tokens, "
        "Tensor candidate_lens, Tensor miss_slots, Tensor miss_slot_bitmap, "
        "Tensor resident_block_table, Tensor logical_seq_lengths, "
        "Tensor profile_enabled, int layer_id, int num_layers, "
        "int max_num_reqs, int qcap) -> "
        "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "lidu_fused_batch_get_prepare_mtp_out(Tensor topk_indices, "
        "Tensor actual_seq_lengths_query, Tensor req_pool_entries, "
        "Tensor(a!) token_to_slot_pool, Tensor(b!) slot_to_token_pool, "
        "Tensor(c!) eviction_cursors, Tensor cache_tokens, "
        "Tensor candidate_lens, Tensor miss_slots, Tensor miss_slot_bitmap, "
        "Tensor resident_block_table, Tensor logical_seq_lengths, "
        "Tensor profile_enabled, Tensor(d!) destination_slots, "
        "Tensor(e!) miss_counts, Tensor(f!) indices, Tensor(g!) offsets, "
        "Tensor(h!) output_indices, Tensor(i!) resident_seq_lengths, "
        "Tensor(j!) profiling, int layer_id, int num_layers, "
        "int max_num_reqs, int qcap) -> "
        "(Tensor(d!), Tensor(e!), Tensor(f!), Tensor(g!), Tensor(h!), "
        "Tensor(i!))");
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
    m.impl(
        "lidu_fused_batch_get_prepare_mtp",
        &LiduFusedBatchGetPrepareMtpNpu);
    m.impl(
        "lidu_fused_batch_get_prepare_mtp_out",
        &LiduFusedBatchGetPrepareMtpOutNpu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
    m.impl(
        "lidu_fused_batch_get_prepare_mtp",
        &LiduFusedBatchGetPrepareMtpMeta);
    m.impl(
        "lidu_fused_batch_get_prepare_mtp_out",
        &LiduFusedBatchGetPrepareMtpOutMeta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "set_opapi_path",
        &lidu_fused_batch_get_prepare_mtp_opapi::SetCustomOpApiPath);
}
