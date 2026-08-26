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
    const at::Tensor& cacheSlotsSnapshot, const at::Tensor& cacheTokens,
    const at::Tensor& candidateLens, const at::Tensor& logicalTokenMap,
    const at::Tensor& privateSourceToSlot, const at::Tensor& privateReady,
    const at::Tensor& sourceIds, const at::Tensor& destinationSlots,
    const at::Tensor& missCounts, const at::Tensor& placementDelta,
    const at::Tensor& privateViewDescriptors,
    const at::Tensor& readyPrivateSourceIds,
    const at::Tensor& readyPrivateSlots,
    const at::Tensor& readyPrivateDestinationSlots,
    const at::Tensor& readyPrivateCounts,
    const at::Tensor& fallbackLogicalTokenIds,
    const at::Tensor& fallbackDestinationSlots,
    const at::Tensor& fallbackCounts)
{
    if (!DebugEnabled()) {
        return;
    }
    std::fprintf(stderr, "P3_RECONCILE_ACLNN_CALL input_count=8 output_count=12\n");
    LogTensor("in0.exact_topk_indices", exactTopkIndices);
    LogTensor("in1.req_pool_entries", reqPoolEntries);
    LogTensor("in2.cache_slots_snapshot", cacheSlotsSnapshot);
    LogTensor("in3.cache_tokens", cacheTokens);
    LogTensor("in4.candidate_lens", candidateLens);
    LogTensor("in5.logical_token_map", logicalTokenMap);
    LogTensor("in6.private_source_to_slot", privateSourceToSlot);
    LogTensor("in7.private_ready", privateReady);
    LogTensor("out0.source_ids", sourceIds);
    LogTensor("out1.destination_slots", destinationSlots);
    LogTensor("out2.miss_counts", missCounts);
    LogTensor("out3.placement_delta", placementDelta);
    LogTensor("out4.private_view_descriptors", privateViewDescriptors);
    LogTensor("out5.ready_private_source_ids", readyPrivateSourceIds);
    LogTensor("out6.ready_private_slots", readyPrivateSlots);
    LogTensor("out7.ready_private_destination_slots", readyPrivateDestinationSlots);
    LogTensor("out8.ready_private_counts", readyPrivateCounts);
    LogTensor("out9.fallback_logical_token_ids", fallbackLogicalTokenIds);
    LogTensor("out10.fallback_destination_slots", fallbackDestinationSlots);
    LogTensor("out11.fallback_counts", fallbackCounts);
    std::fflush(stderr);
}

bool IsVectorOfLength(const at::Tensor& tensor, int64_t length)
{
    return tensor.dim() == 1 && tensor.numel() == length;
}

void CheckShapes(
    const at::Tensor& exactTopkIndices, const at::Tensor& reqPoolEntries,
    const at::Tensor& cacheSlotsSnapshot, const at::Tensor& cacheTokens,
    const at::Tensor& candidateLens, const at::Tensor& logicalTokenMap,
    const at::Tensor& privateSourceToSlot, const at::Tensor& privateReady)
{
    TORCH_CHECK(
        exactTopkIndices.dim() == 3 && exactTopkIndices.size(0) > 0 &&
            exactTopkIndices.size(1) == 1 &&
            exactTopkIndices.size(2) == kSparseCount,
        "exact_topk_indices must have shape [B, 1, 2048].");
    const int64_t batch = exactTopkIndices.size(0);
    TORCH_CHECK(
        IsVectorOfLength(reqPoolEntries, batch) &&
            IsVectorOfLength(cacheTokens, batch) &&
            IsVectorOfLength(candidateLens, batch),
        "req_pool_entries, cache_tokens, and candidate_lens must have shape [B].");
    TORCH_CHECK(
        cacheSlotsSnapshot.dim() == 2 && cacheSlotsSnapshot.size(0) > 0 &&
            cacheSlotsSnapshot.size(1) > 0 &&
            cacheSlotsSnapshot.size(1) <= kMaxSourceCapacity,
        "cache_slots_snapshot must have shape [P, S], 0 < S <= ",
        kMaxSourceCapacity, ".");
    TORCH_CHECK(
        logicalTokenMap.dim() == 2 &&
            logicalTokenMap.sizes() == cacheSlotsSnapshot.sizes(),
        "logical_token_map must have the same [P, S] shape as cache_slots_snapshot.");
    TORCH_CHECK(
        privateSourceToSlot.dim() == 2 &&
            privateSourceToSlot.size(0) == batch &&
            privateSourceToSlot.size(1) == cacheSlotsSnapshot.size(1),
        "private_source_to_slot must have shape [B, S].");
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
    const at::Tensor& cacheSlotsSnapshot, const at::Tensor& cacheTokens,
    const at::Tensor& candidateLens, const at::Tensor& logicalTokenMap,
    const at::Tensor& privateSourceToSlot, const at::Tensor& privateReady)
{
    TORCH_CHECK(exactTopkIndices.device().is_privateuseone(),
                "exact_reconcile_private_view requires NPU inputs.");
    const at::Device device = exactTopkIndices.device();
    for (const auto& tensor : {reqPoolEntries, cacheSlotsSnapshot, cacheTokens,
                               candidateLens, logicalTokenMap,
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
    CheckShapes(exactTopkIndices, reqPoolEntries, cacheSlotsSnapshot,
                cacheTokens, candidateLens, logicalTokenMap,
                privateSourceToSlot, privateReady);
}

Result AllocateOutputs(const at::Tensor& exactTopkIndices)
{
    const auto int32Options = exactTopkIndices.options().dtype(at::kInt);
    const int64_t batch = exactTopkIndices.size(0);
    return std::make_tuple(
        at::empty_like(exactTopkIndices),
        at::empty_like(exactTopkIndices),
        at::empty({batch}, int32Options),
        at::empty({batch, kPlacementDeltaPlanes, kSparseCount}, int32Options),
        at::empty({batch, kPrivateViewPlanes, kSparseCount}, int32Options),
        at::empty_like(exactTopkIndices),
        at::empty_like(exactTopkIndices),
        at::empty_like(exactTopkIndices),
        at::empty({batch}, int32Options),
        at::empty_like(exactTopkIndices),
        at::empty_like(exactTopkIndices),
        at::empty({batch}, int32Options));
}

Result ExactReconcilePrivateViewNpu(
    const at::Tensor& exactTopkIndices, const at::Tensor& reqPoolEntries,
    const at::Tensor& cacheSlotsSnapshot, const at::Tensor& cacheTokens,
    const at::Tensor& candidateLens, const at::Tensor& logicalTokenMap,
    const at::Tensor& privateSourceToSlot, const at::Tensor& privateReady)
{
    CheckInputs(exactTopkIndices, reqPoolEntries, cacheSlotsSnapshot,
                cacheTokens, candidateLens, logicalTokenMap,
                privateSourceToSlot, privateReady);
    auto outputs = AllocateOutputs(exactTopkIndices);
    auto& sourceIds = std::get<0>(outputs);
    auto& destinationSlots = std::get<1>(outputs);
    auto& missCounts = std::get<2>(outputs);
    auto& placementDelta = std::get<3>(outputs);
    auto& privateViewDescriptors = std::get<4>(outputs);
    auto& readyPrivateSourceIds = std::get<5>(outputs);
    auto& readyPrivateSlots = std::get<6>(outputs);
    auto& readyPrivateDestinationSlots = std::get<7>(outputs);
    auto& readyPrivateCounts = std::get<8>(outputs);
    auto& fallbackLogicalTokenIds = std::get<9>(outputs);
    auto& fallbackDestinationSlots = std::get<10>(outputs);
    auto& fallbackCounts = std::get<11>(outputs);

    LogCall(
        exactTopkIndices, reqPoolEntries, cacheSlotsSnapshot, cacheTokens,
        candidateLens, logicalTokenMap, privateSourceToSlot, privateReady,
        sourceIds, destinationSlots, missCounts, placementDelta,
        privateViewDescriptors, readyPrivateSourceIds, readyPrivateSlots,
        readyPrivateDestinationSlots, readyPrivateCounts,
        fallbackLogicalTokenIds, fallbackDestinationSlots, fallbackCounts);

    auto keepalive = std::make_tuple(
        exactTopkIndices, reqPoolEntries, cacheSlotsSnapshot, cacheTokens,
        candidateLens, logicalTokenMap, privateSourceToSlot, privateReady,
        sourceIds, destinationSlots, missCounts, placementDelta,
        privateViewDescriptors, readyPrivateSourceIds, readyPrivateSlots,
        readyPrivateDestinationSlots, readyPrivateCounts,
        fallbackLogicalTokenIds, fallbackDestinationSlots, fallbackCounts);
    EXEC_NPU_CMD_ORDERED(
        aclnnA5ExactReconcilePrivateView, keepalive,
        exactTopkIndices, reqPoolEntries, cacheSlotsSnapshot, cacheTokens,
        candidateLens, logicalTokenMap, privateSourceToSlot, privateReady,
        sourceIds, destinationSlots, missCounts, placementDelta,
        privateViewDescriptors, readyPrivateSourceIds, readyPrivateSlots,
        readyPrivateDestinationSlots, readyPrivateCounts,
        fallbackLogicalTokenIds, fallbackDestinationSlots, fallbackCounts);
    return outputs;
}

Result ExactReconcilePrivateViewMeta(
    const at::Tensor& exactTopkIndices, const at::Tensor& reqPoolEntries,
    const at::Tensor& cacheSlotsSnapshot, const at::Tensor& cacheTokens,
    const at::Tensor& candidateLens, const at::Tensor& logicalTokenMap,
    const at::Tensor& privateSourceToSlot, const at::Tensor& privateReady)
{
    // Meta dispatch has no PrivateUse1 device.  Keep the exact shape contract
    // available for fake/graph execution without issuing NPU checks.
    TORCH_CHECK(
        exactTopkIndices.scalar_type() == at::kInt &&
            reqPoolEntries.scalar_type() == at::kInt &&
            cacheSlotsSnapshot.scalar_type() == at::kInt &&
            cacheTokens.scalar_type() == at::kInt &&
            candidateLens.scalar_type() == at::kInt &&
            logicalTokenMap.scalar_type() == at::kInt &&
            privateSourceToSlot.scalar_type() == at::kInt &&
            privateReady.scalar_type() == at::kInt,
        "exact_reconcile_private_view meta inputs must be int32.");
    CheckShapes(exactTopkIndices, reqPoolEntries, cacheSlotsSnapshot,
                cacheTokens, candidateLens, logicalTokenMap,
                privateSourceToSlot, privateReady);
    return AllocateOutputs(exactTopkIndices);
}
}  // namespace

TORCH_LIBRARY(a5_exact_reconcile_private_view, m)
{
    m.def(
        "exact_reconcile_private_view(Tensor exact_topk_indices, "
        "Tensor req_pool_entries, Tensor cache_slots_snapshot, "
        "Tensor cache_tokens, Tensor candidate_lens, "
        "Tensor logical_token_map, Tensor private_source_to_slot, "
        "Tensor private_ready) -> (Tensor source_ids, "
        "Tensor destination_slots, Tensor miss_counts, "
        "Tensor placement_delta, Tensor private_view_descriptors, "
        "Tensor ready_private_source_ids, Tensor ready_private_slots, "
        "Tensor ready_private_destination_slots, Tensor ready_private_counts, "
        "Tensor fallback_logical_token_ids, Tensor fallback_destination_slots, "
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
