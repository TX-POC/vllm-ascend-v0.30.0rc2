// Prototype-only P3 bridge for transaction-local exact Top-K reconciliation.
//
// This extension has no dependency on the normal vLLM-Ascend registration
// path.  It is deliberately separate because this op only produces a plan:
// it does not launch Get, materialize private packed KV, call QSFA, or publish
// cache_slots.  The explicit ready/fallback outputs are the contract for a
// later transaction-terminal materialize/guaranteed-recall stage.

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>

#include <torch/extension.h>
#include <torch/library.h>

#include "op_api_common.h"

namespace {
constexpr int64_t kSparseCount = 2048;
constexpr int64_t kPlacementDeltaPlanes = 3;
constexpr int64_t kPrivateViewPlanes = 4;
constexpr int64_t kMaxSourceCapacity = 1 << 18;

using Result = std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor>;

bool DebugEnabled()
{
    const char* value = std::getenv("NANOVLLM_A5_OPAPI_DEBUG");
    return value != nullptr && value[0] != '\0' &&
           !(value[0] == '0' && value[1] == '\0');
}

std::string ShapeString(c10::IntArrayRef values)
{
    std::ostringstream stream;
    stream << '[';
    for (int64_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            stream << ',';
        }
        stream << values[index];
    }
    stream << ']';
    return stream.str();
}

void LogTensor(const char* role, const at::Tensor& tensor)
{
    if (!DebugEnabled()) {
        return;
    }
    const std::string shape = ShapeString(tensor.sizes());
    std::fprintf(
        stderr,
        "P3_RECONCILE_ACLNN_TENSOR role=%s dtype=%s device=%s shape=%s "
        "contiguous=%d nbytes=%zu data=%p\n",
        role, c10::toString(tensor.scalar_type()), tensor.device().str().c_str(),
        shape.c_str(), tensor.is_contiguous() ? 1 : 0, tensor.nbytes(),
        tensor.data_ptr());
}

void LogCall(
    const at::Tensor& exactTopkIndices, const at::Tensor& reqPoolEntries,
    const at::Tensor& kvObjectSlotsSnapshot, const at::Tensor& cacheTokens,
    const at::Tensor& kvObjectLens, const at::Tensor& scoreSourceLens,
    const at::Tensor& kvObjectMap,
    const at::Tensor& privateSourceToSlot, const at::Tensor& privateReady,
    const at::Tensor& kvObjectIds, const at::Tensor& destinationSlots,
    const at::Tensor& missCounts, const at::Tensor& placementDelta,
    const at::Tensor& privateViewDescriptors,
    const at::Tensor& readyPrivateKvObjectIds,
    const at::Tensor& readyPrivateSlots,
    const at::Tensor& readyPrivateDestinationSlots,
    const at::Tensor& readyPrivateCounts,
    const at::Tensor& fallbackKvObjectIds,
    const at::Tensor& fallbackDestinationSlots,
    const at::Tensor& fallbackCounts)
{
    if (!DebugEnabled()) {
        return;
    }
    std::fprintf(stderr, "P3_RECONCILE_ACLNN_CALL input_count=9 output_count=12\n");
    LogTensor("in0.exact_topk_indices", exactTopkIndices);
    LogTensor("in1.req_pool_entries", reqPoolEntries);
    LogTensor("in2.kv_object_slots_snapshot", kvObjectSlotsSnapshot);
    LogTensor("in3.cache_tokens", cacheTokens);
    LogTensor("in4.kv_object_lens", kvObjectLens);
    LogTensor("in5.score_source_lens", scoreSourceLens);
    LogTensor("in6.kv_object_map", kvObjectMap);
    LogTensor("in7.private_source_to_slot", privateSourceToSlot);
    LogTensor("in8.private_ready", privateReady);
    LogTensor("out0.kv_object_ids", kvObjectIds);
    LogTensor("out1.destination_slots", destinationSlots);
    LogTensor("out2.miss_counts", missCounts);
    LogTensor("out3.placement_delta", placementDelta);
    LogTensor("out4.private_view_descriptors", privateViewDescriptors);
    LogTensor("out5.ready_private_kv_object_ids", readyPrivateKvObjectIds);
    LogTensor("out6.ready_private_slots", readyPrivateSlots);
    LogTensor("out7.ready_private_destination_slots", readyPrivateDestinationSlots);
    LogTensor("out8.ready_private_counts", readyPrivateCounts);
    LogTensor("out9.fallback_kv_object_ids", fallbackKvObjectIds);
    LogTensor("out10.fallback_destination_slots", fallbackDestinationSlots);
    LogTensor("out11.fallback_counts", fallbackCounts);
    std::fflush(stderr);
}

bool IsVectorOfLength(const at::Tensor& tensor, int64_t length)
{
    return tensor.dim() == 1 && tensor.numel() == length;
}

int64_t ExactTopkRowCount(const at::Tensor& exactTopkIndices)
{
    TORCH_CHECK(exactTopkIndices.dim() >= 1 &&
                    exactTopkIndices.size(-1) == kSparseCount,
                "exact_topk_indices must have one or more dimensions and a "
                "final dimension of 2048.");
    int64_t rows = 1;
    for (int64_t dim = 0; dim + 1 < exactTopkIndices.dim(); ++dim) {
        const int64_t value = exactTopkIndices.size(dim);
        TORCH_CHECK(value > 0 && rows <=
                        std::numeric_limits<int64_t>::max() / value,
                    "exact_topk_indices leading dimensions must have a "
                    "positive, int64-safe product.");
        rows *= value;
    }
    return rows;
}

void CheckShapes(
    const at::Tensor& exactTopkIndices, const at::Tensor& reqPoolEntries,
    const at::Tensor& kvObjectSlotsSnapshot, const at::Tensor& cacheTokens,
    const at::Tensor& kvObjectLens, const at::Tensor& scoreSourceLens,
    const at::Tensor& kvObjectMap,
    const at::Tensor& privateSourceToSlot, const at::Tensor& privateReady)
{
    const int64_t rowCount = ExactTopkRowCount(exactTopkIndices);
    TORCH_CHECK(
        IsVectorOfLength(reqPoolEntries, rowCount) &&
            IsVectorOfLength(cacheTokens, rowCount) &&
            IsVectorOfLength(kvObjectLens, rowCount) &&
            IsVectorOfLength(scoreSourceLens, rowCount),
        "req_pool_entries, cache_tokens, kv_object_lens, and score_source_lens "
        "must have shape [R], where R=ProductExceptLast(exact_topk_indices)."
    );
    TORCH_CHECK(
        kvObjectSlotsSnapshot.dim() == 2 && kvObjectSlotsSnapshot.size(0) > 0 &&
            kvObjectSlotsSnapshot.size(1) > 0 &&
            kvObjectSlotsSnapshot.size(1) <= kMaxSourceCapacity,
        "kv_object_slots_snapshot must have shape [P, O], 0 < O <= ",
        kMaxSourceCapacity, ".");
    TORCH_CHECK(
        kvObjectMap.dim() == 2 &&
            kvObjectMap.size(0) == kvObjectSlotsSnapshot.size(0) &&
            kvObjectMap.size(1) > 0 &&
            kvObjectMap.size(1) <= kMaxSourceCapacity,
        "kv_object_map must have shape [P, S], share P with "
        "kv_object_slots_snapshot, and satisfy 0 < S <= ",
        kMaxSourceCapacity, ".  S is C8's score-source capacity and may "
        "differ from the KV-object directory width O.");
    TORCH_CHECK(
        privateSourceToSlot.dim() == 2 &&
            privateSourceToSlot.size(0) == rowCount &&
            privateSourceToSlot.size(1) == kvObjectMap.size(1),
        "private_source_to_slot must have shape [R, S].");
    TORCH_CHECK(
        privateReady.dim() == 2 && privateReady.size(0) > 0 &&
            privateReady.size(1) > 0 &&
            static_cast<uint64_t>(privateReady.size(0)) *
                    static_cast<uint64_t>(privateReady.size(1)) <=
                static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
        "private_ready must have shape [Q, R] and Q*R must fit int32.");
}

void CheckInputs(
    const at::Tensor& exactTopkIndices, const at::Tensor& reqPoolEntries,
    const at::Tensor& kvObjectSlotsSnapshot, const at::Tensor& cacheTokens,
    const at::Tensor& kvObjectLens, const at::Tensor& scoreSourceLens,
    const at::Tensor& kvObjectMap,
    const at::Tensor& privateSourceToSlot, const at::Tensor& privateReady,
    int64_t cmpRatio, int64_t kvObjectDomain)
{
    TORCH_CHECK(exactTopkIndices.device().is_privateuseone(),
                "exact_reconcile_private_view requires NPU inputs.");
    const at::Device device = exactTopkIndices.device();
    for (const auto& tensor : {reqPoolEntries, kvObjectSlotsSnapshot, cacheTokens,
                               kvObjectLens, scoreSourceLens, kvObjectMap,
                               privateSourceToSlot, privateReady}) {
        TORCH_CHECK(tensor.device() == device,
                    "exact_reconcile_private_view inputs must use one NPU.");
        TORCH_CHECK(tensor.is_contiguous(),
                    "exact_reconcile_private_view inputs must be contiguous.");
        TORCH_CHECK(tensor.scalar_type() == at::kInt,
                    "exact_reconcile_private_view inputs must be int32.");
    }
    TORCH_CHECK(exactTopkIndices.is_contiguous(),
                "exact_reconcile_private_view inputs must be contiguous.");
    TORCH_CHECK(exactTopkIndices.scalar_type() == at::kInt,
                "exact_reconcile_private_view inputs must be int32.");
    TORCH_CHECK(cmpRatio == 1 || cmpRatio == 4 || cmpRatio == 128,
                "cmp_ratio must be 1, 4, or 128.");
    TORCH_CHECK(kvObjectDomain == 0 || kvObjectDomain == 1,
                "kv_object_domain must be 0 (RAW_TOKEN) or 1 (C8_COMPRESSED_OBJECT).");
    TORCH_CHECK(kvObjectDomain != 0 || cmpRatio == 1,
                "kv_object_domain=RAW_TOKEN requires cmp_ratio=1.");
    CheckShapes(exactTopkIndices, reqPoolEntries, kvObjectSlotsSnapshot,
                cacheTokens, kvObjectLens, scoreSourceLens, kvObjectMap,
                privateSourceToSlot, privateReady);
}

Result AllocateOutputs(const at::Tensor& exactTopkIndices)
{
    const auto int32Options = exactTopkIndices.options().dtype(at::kInt);
    const int64_t rowCount = ExactTopkRowCount(exactTopkIndices);
    return std::make_tuple(
        at::empty_like(exactTopkIndices),
        at::empty_like(exactTopkIndices),
        at::empty({rowCount}, int32Options),
        at::empty({rowCount, kPlacementDeltaPlanes, kSparseCount}, int32Options),
        at::empty({rowCount, kPrivateViewPlanes, kSparseCount}, int32Options),
        at::empty_like(exactTopkIndices),
        at::empty_like(exactTopkIndices),
        at::empty_like(exactTopkIndices),
        at::empty({rowCount}, int32Options),
        at::empty_like(exactTopkIndices),
        at::empty_like(exactTopkIndices),
        at::empty({rowCount}, int32Options));
}

Result ExactReconcilePrivateViewNpu(
    const at::Tensor& exactTopkIndices, const at::Tensor& reqPoolEntries,
    const at::Tensor& kvObjectSlotsSnapshot, const at::Tensor& cacheTokens,
    const at::Tensor& kvObjectLens, const at::Tensor& scoreSourceLens,
    const at::Tensor& kvObjectMap,
    const at::Tensor& privateSourceToSlot, const at::Tensor& privateReady,
    int64_t cmpRatio, int64_t kvObjectDomain)
{
    CheckInputs(exactTopkIndices, reqPoolEntries, kvObjectSlotsSnapshot,
                cacheTokens, kvObjectLens, scoreSourceLens, kvObjectMap,
                privateSourceToSlot, privateReady, cmpRatio, kvObjectDomain);
    auto outputs = AllocateOutputs(exactTopkIndices);
    auto& kvObjectIds = std::get<0>(outputs);
    auto& destinationSlots = std::get<1>(outputs);
    auto& missCounts = std::get<2>(outputs);
    auto& placementDelta = std::get<3>(outputs);
    auto& privateViewDescriptors = std::get<4>(outputs);
    auto& readyPrivateKvObjectIds = std::get<5>(outputs);
    auto& readyPrivateSlots = std::get<6>(outputs);
    auto& readyPrivateDestinationSlots = std::get<7>(outputs);
    auto& readyPrivateCounts = std::get<8>(outputs);
    auto& fallbackKvObjectIds = std::get<9>(outputs);
    auto& fallbackDestinationSlots = std::get<10>(outputs);
    auto& fallbackCounts = std::get<11>(outputs);

    LogCall(
        exactTopkIndices, reqPoolEntries, kvObjectSlotsSnapshot, cacheTokens,
        kvObjectLens, scoreSourceLens, kvObjectMap, privateSourceToSlot,
        privateReady,
        kvObjectIds, destinationSlots, missCounts, placementDelta,
        privateViewDescriptors, readyPrivateKvObjectIds, readyPrivateSlots,
        readyPrivateDestinationSlots, readyPrivateCounts,
        fallbackKvObjectIds, fallbackDestinationSlots, fallbackCounts);

    auto keepalive = std::make_tuple(
        exactTopkIndices, reqPoolEntries, kvObjectSlotsSnapshot, cacheTokens,
        kvObjectLens, scoreSourceLens, kvObjectMap, privateSourceToSlot,
        privateReady,
        kvObjectIds, destinationSlots, missCounts, placementDelta,
        privateViewDescriptors, readyPrivateKvObjectIds, readyPrivateSlots,
        readyPrivateDestinationSlots, readyPrivateCounts,
        fallbackKvObjectIds, fallbackDestinationSlots, fallbackCounts);
    EXEC_NPU_CMD_ORDERED(
        aclnnA5ExactReconcilePrivateView, keepalive,
        exactTopkIndices, reqPoolEntries, kvObjectSlotsSnapshot, cacheTokens,
        kvObjectLens, scoreSourceLens, kvObjectMap, privateSourceToSlot,
        privateReady, cmpRatio, kvObjectDomain,
        kvObjectIds, destinationSlots, missCounts, placementDelta,
        privateViewDescriptors, readyPrivateKvObjectIds, readyPrivateSlots,
        readyPrivateDestinationSlots, readyPrivateCounts,
        fallbackKvObjectIds, fallbackDestinationSlots, fallbackCounts);
    return outputs;
}

Result ExactReconcilePrivateViewMeta(
    const at::Tensor& exactTopkIndices, const at::Tensor& reqPoolEntries,
    const at::Tensor& kvObjectSlotsSnapshot, const at::Tensor& cacheTokens,
    const at::Tensor& kvObjectLens, const at::Tensor& scoreSourceLens,
    const at::Tensor& kvObjectMap,
    const at::Tensor& privateSourceToSlot, const at::Tensor& privateReady,
    int64_t cmpRatio, int64_t kvObjectDomain)
{
    // Meta dispatch has no PrivateUse1 device.  Keep the exact shape contract
    // available for fake/graph execution without issuing NPU checks.
    TORCH_CHECK(
        exactTopkIndices.scalar_type() == at::kInt &&
            reqPoolEntries.scalar_type() == at::kInt &&
            kvObjectSlotsSnapshot.scalar_type() == at::kInt &&
            cacheTokens.scalar_type() == at::kInt &&
            kvObjectLens.scalar_type() == at::kInt &&
            scoreSourceLens.scalar_type() == at::kInt &&
            kvObjectMap.scalar_type() == at::kInt &&
            privateSourceToSlot.scalar_type() == at::kInt &&
            privateReady.scalar_type() == at::kInt,
        "exact_reconcile_private_view meta inputs must be int32.");
    TORCH_CHECK(cmpRatio == 1 || cmpRatio == 4 || cmpRatio == 128,
                "cmp_ratio must be 1, 4, or 128.");
    TORCH_CHECK(kvObjectDomain == 0 || kvObjectDomain == 1,
                "kv_object_domain must be 0 (RAW_TOKEN) or 1 (C8_COMPRESSED_OBJECT).");
    TORCH_CHECK(kvObjectDomain != 0 || cmpRatio == 1,
                "kv_object_domain=RAW_TOKEN requires cmp_ratio=1.");
    CheckShapes(exactTopkIndices, reqPoolEntries, kvObjectSlotsSnapshot,
                cacheTokens, kvObjectLens, scoreSourceLens, kvObjectMap,
                privateSourceToSlot, privateReady);
    return AllocateOutputs(exactTopkIndices);
}
}  // namespace

TORCH_LIBRARY(a5_exact_reconcile_private_view, m)
{
    m.def(
        "exact_reconcile_private_view(Tensor exact_topk_indices, "
        "Tensor req_pool_entries, Tensor kv_object_slots_snapshot, "
        "Tensor cache_tokens, Tensor kv_object_lens, "
        "Tensor score_source_lens, Tensor kv_object_map, "
        "Tensor private_source_to_slot, "
        "Tensor private_ready, int cmp_ratio, int kv_object_domain) -> (Tensor kv_object_ids, "
        "Tensor destination_slots, Tensor miss_counts, "
        "Tensor placement_delta, Tensor private_view_descriptors, "
        "Tensor ready_private_kv_object_ids, Tensor ready_private_slots, "
        "Tensor ready_private_destination_slots, Tensor ready_private_counts, "
        "Tensor fallback_kv_object_ids, Tensor fallback_destination_slots, "
        "Tensor fallback_counts)");
}

TORCH_LIBRARY_IMPL(a5_exact_reconcile_private_view, PrivateUse1, m)
{
    m.impl("exact_reconcile_private_view", &ExactReconcilePrivateViewNpu);
}

TORCH_LIBRARY_IMPL(a5_exact_reconcile_private_view, Meta, m)
{
    m.impl("exact_reconcile_private_view", &ExactReconcilePrivateViewMeta);
}

// Importing the extension is sufficient to register the dispatcher schema.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
