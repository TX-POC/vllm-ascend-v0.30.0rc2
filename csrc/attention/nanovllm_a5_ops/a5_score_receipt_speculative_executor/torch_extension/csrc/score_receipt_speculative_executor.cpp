// P7.1 typed bridge for the standalone CandidateSeed -> Direct-HCCP executor.
//
// This bridge intentionally accepts only already-published CandidateSeed GM.
// It does not import, call, or register C8.  All transport state is supplied
// by the graph as mutable tensors.  The Torch schema returns 18 aliases, but
// the ACLNN schema declares each pair with the same name, so CANN generates
// one mutable `Ref` descriptor rather than a duplicate input/output pair.
// This keeps the single-op ABI at 31 tensor descriptors (the CANN limit is
// 32), while preserving graph-visible in-place state across replay.

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <tuple>

#include <torch/extension.h>
#include <torch/library.h>

#include "op_api_common.h"

namespace {
constexpr int64_t kInputCount = 31;
constexpr int64_t kOutputCount = 18;
constexpr int64_t kCandidateCapacity = 128;
constexpr int64_t kCandidateSeedWords = 4;
constexpr int64_t kMicrobatchCapacity = 64;
constexpr int64_t kPackedKvRowBytes = 656;
constexpr int64_t kControlWords = 16;
constexpr int64_t kWorkerStatColumns = 16;

struct Geometry {
    int64_t receipts;
    int64_t rows;
    int64_t source_capacity;
    int64_t pools;
    int64_t objects;
    int64_t queues;
    int64_t private_capacity;
    int64_t workers;
};

bool DebugEnabled()
{
    const char* value = std::getenv("NANOVLLM_A5_OPAPI_DEBUG");
    return value != nullptr && value[0] != '\0' &&
           !(value[0] == '0' && value[1] == '\0');
}

void CheckVector(const at::Tensor& tensor, int64_t length, const char* name)
{
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == length,
                name, " must have shape [", length, "].");
}

void CheckMatrix(const at::Tensor& tensor, int64_t rows, int64_t columns,
                 const char* name)
{
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == rows &&
                    tensor.size(1) == columns,
                name, " must have shape [", rows, ", ", columns, "].");
}

Geometry ValidateShapes(
    const at::Tensor& candidateSeeds, const at::Tensor& candidateCounts,
    const at::Tensor& candidateReadyEpoch, const at::Tensor& candidateStatus,
    const at::Tensor& streamingEpoch, const at::Tensor& sourceLens,
    const at::Tensor& kvObjectAddressDomain,
    const at::Tensor& reqPoolEntries, const at::Tensor& kvObjectMap,
    const at::Tensor& kvObjectLens, const at::Tensor& kvObjectSlotsSnapshot,
    const at::Tensor& privateSourceToSlot,
    const at::Tensor& privateObjectToSlot,
    const at::Tensor& privateReadyTable, const at::Tensor& privateArenaHead,
    const at::Tensor& privatePackedOutput, const at::Tensor& fusedGetControl,
    const at::Tensor& queueLeases, const at::Tensor& queueEpoch,
    const at::Tensor& queueGeneration, const at::Tensor& queuePostSequence,
    const at::Tensor& queueState, const at::Tensor& queueQuarantine,
    const at::Tensor& queueRequestSourceOffsets,
    const at::Tensor& queueRequestObjectOffsets,
    const at::Tensor& queueRequestOutputIndices,
    const at::Tensor& queueRequestCount,
    const at::Tensor& privateReadyEpoch,
    const at::Tensor& privateReadyGeneration,
    const at::Tensor& workerStats, const at::Tensor& rowsPerEntry)
{
    TORCH_CHECK(candidateSeeds.dim() == 3 && candidateSeeds.size(0) > 0 &&
                    candidateSeeds.size(1) == kCandidateCapacity &&
                    candidateSeeds.size(2) == kCandidateSeedWords,
                "candidate_seeds must be int32[N, 128, 4], N > 0.");
    const int64_t receipts = candidateSeeds.size(0);
    CheckVector(candidateCounts, receipts, "candidate_counts");
    CheckVector(candidateReadyEpoch, receipts, "candidate_ready_epoch");
    CheckVector(candidateStatus, receipts, "candidate_status");
    CheckVector(streamingEpoch, 1, "streaming_epoch");
    CheckVector(kvObjectAddressDomain, 1, "kv_object_address_domain");
    TORCH_CHECK(sourceLens.dim() == 1 && sourceLens.numel() > 0,
                "source_lens must be int32[R], R > 0.");
    const int64_t rows = sourceLens.numel();
    CheckVector(reqPoolEntries, rows, "req_pool_entries");
    CheckVector(kvObjectLens, rows, "kv_object_lens");
    TORCH_CHECK(kvObjectMap.dim() == 2 && kvObjectMap.size(0) > 0 &&
                    kvObjectMap.size(1) > 0,
                "kv_object_map must be int32[P, S], P,S > 0.");
    const int64_t pools = kvObjectMap.size(0);
    const int64_t sourceCapacity = kvObjectMap.size(1);
    TORCH_CHECK(kvObjectSlotsSnapshot.dim() == 2 &&
                    kvObjectSlotsSnapshot.size(0) == pools &&
                    kvObjectSlotsSnapshot.size(1) > 0,
                "kv_object_slots_snapshot must be int32[P, O], P aligned to map.");
    const int64_t objects = kvObjectSlotsSnapshot.size(1);
    CheckMatrix(privateSourceToSlot, rows, sourceCapacity,
                "private_source_to_slot");
    CheckMatrix(privateObjectToSlot, pools, objects, "private_object_to_slot");
    TORCH_CHECK(privateReadyTable.dim() == 2 && privateReadyTable.size(0) > 0 &&
                    privateReadyTable.size(1) > 0,
                "private_ready_table must be int32[Q, A], Q,A > 0.");
    const int64_t queues = privateReadyTable.size(0);
    const int64_t privateCapacity = privateReadyTable.size(1);
    CheckVector(privateArenaHead, queues, "private_arena_head");
    TORCH_CHECK(queues <= std::numeric_limits<int64_t>::max() / privateCapacity,
                "private queue geometry overflows int64.");
    CheckMatrix(privatePackedOutput, queues * privateCapacity,
                kPackedKvRowBytes, "private_packed_output");
    CheckVector(fusedGetControl, kControlWords, "fused_get_control");
    CheckVector(queueLeases, queues, "queue_leases");
    CheckVector(queueEpoch, queues, "queue_epoch");
    CheckVector(queueGeneration, queues, "queue_generation");
    CheckVector(queuePostSequence, queues, "queue_post_sequence");
    CheckVector(queueState, queues, "queue_state");
    CheckVector(queueQuarantine, queues, "queue_quarantine");
    CheckMatrix(queueRequestSourceOffsets, queues, kMicrobatchCapacity,
                "queue_request_source_offsets");
    CheckMatrix(queueRequestObjectOffsets, queues, kMicrobatchCapacity,
                "queue_request_object_offsets");
    CheckMatrix(queueRequestOutputIndices, queues, kMicrobatchCapacity,
                "queue_request_output_indices");
    CheckVector(queueRequestCount, queues, "queue_request_count");
    CheckMatrix(privateReadyEpoch, queues, privateCapacity,
                "private_ready_epoch");
    CheckMatrix(privateReadyGeneration, queues, privateCapacity,
                "private_ready_generation");
    TORCH_CHECK(workerStats.dim() == 2 && workerStats.size(0) > 0 &&
                    workerStats.size(1) == kWorkerStatColumns,
                "worker_stats must be int64[W, 16], W > 0.");
    CheckVector(rowsPerEntry, 1, "rows_per_entry");
    return {receipts, rows, sourceCapacity, pools, objects, queues,
            privateCapacity, workerStats.size(0)};
}

void CheckNpuInputSet(const at::Tensor& candidateSeeds,
                      const std::array<const at::Tensor*, kInputCount>& tensors)
{
    TORCH_CHECK(candidateSeeds.device().is_privateuseone(),
                "score_receipt_speculative_executor requires NPU inputs.");
    const at::Device device = candidateSeeds.device();
    const std::array<at::ScalarType, kInputCount> types = {
        at::kInt, at::kInt, at::kLong, at::kInt, at::kLong, at::kInt,
        at::kInt, at::kInt, at::kInt, at::kInt, at::kInt, at::kInt,
        at::kInt, at::kInt, at::kInt, at::kByte, at::kLong, at::kInt,
        at::kLong, at::kLong, at::kLong, at::kInt, at::kInt, at::kLong,
        at::kLong, at::kInt, at::kInt, at::kLong, at::kLong, at::kLong,
        at::kInt};
    for (size_t index = 0; index < tensors.size(); ++index) {
        const at::Tensor& tensor = *tensors[index];
        TORCH_CHECK(tensor.device() == device && tensor.is_contiguous() &&
                        tensor.scalar_type() == types[index],
                    "P7.1 input ", index,
                    " must be contiguous, have its declared ABI dtype, and be on ",
                    device.str(), ".");
    }
}

using Result = std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor>;

Result RunNpu(
    const at::Tensor& candidateSeeds, const at::Tensor& candidateCounts,
    const at::Tensor& candidateReadyEpoch, const at::Tensor& candidateStatus,
    const at::Tensor& streamingEpoch, const at::Tensor& sourceLens,
    const at::Tensor& kvObjectAddressDomain,
    const at::Tensor& reqPoolEntries, const at::Tensor& kvObjectMap,
    const at::Tensor& kvObjectLens, const at::Tensor& kvObjectSlotsSnapshot,
    at::Tensor privateSourceToSlot, at::Tensor privateObjectToSlot,
    at::Tensor privateReadyTable, at::Tensor privateArenaHead,
    at::Tensor privatePackedOutput, const at::Tensor& fusedGetControl,
    at::Tensor queueLeases, at::Tensor queueEpoch, at::Tensor queueGeneration,
    at::Tensor queuePostSequence, at::Tensor queueState,
    at::Tensor queueQuarantine, at::Tensor queueRequestSourceOffsets,
    at::Tensor queueRequestObjectOffsets, at::Tensor queueRequestOutputIndices,
    at::Tensor queueRequestCount, at::Tensor privateReadyEpoch,
    at::Tensor privateReadyGeneration, at::Tensor workerStats,
    const at::Tensor& rowsPerEntry)
{
    const std::array<const at::Tensor*, kInputCount> inputs = {
        &candidateSeeds, &candidateCounts, &candidateReadyEpoch, &candidateStatus,
        &streamingEpoch, &sourceLens, &kvObjectAddressDomain, &reqPoolEntries,
        &kvObjectMap, &kvObjectLens, &kvObjectSlotsSnapshot,
        &privateSourceToSlot, &privateObjectToSlot, &privateReadyTable,
        &privateArenaHead, &privatePackedOutput, &fusedGetControl, &queueLeases,
        &queueEpoch, &queueGeneration, &queuePostSequence, &queueState,
        &queueQuarantine, &queueRequestSourceOffsets,
        &queueRequestObjectOffsets, &queueRequestOutputIndices,
        &queueRequestCount, &privateReadyEpoch, &privateReadyGeneration,
        &workerStats, &rowsPerEntry};
    const Geometry geometry = ValidateShapes(
        candidateSeeds, candidateCounts, candidateReadyEpoch, candidateStatus,
        streamingEpoch, sourceLens, kvObjectAddressDomain, reqPoolEntries,
        kvObjectMap, kvObjectLens, kvObjectSlotsSnapshot, privateSourceToSlot,
        privateObjectToSlot, privateReadyTable, privateArenaHead,
        privatePackedOutput, fusedGetControl, queueLeases, queueEpoch,
        queueGeneration, queuePostSequence, queueState, queueQuarantine,
        queueRequestSourceOffsets, queueRequestObjectOffsets,
        queueRequestOutputIndices, queueRequestCount, privateReadyEpoch,
        privateReadyGeneration, workerStats, rowsPerEntry);
    (void)geometry;
    CheckNpuInputSet(candidateSeeds, inputs);

    if (DebugEnabled()) {
        std::fprintf(stderr,
                     "P7_1_REF_ABI aclnn_tensor_args=31 logical_outputs=18 "
                     "mutable_refs=18 candidate_receipts=%lld queues=%lld private_capacity=%lld\n",
                     static_cast<long long>(geometry.receipts),
                     static_cast<long long>(geometry.queues),
                     static_cast<long long>(geometry.private_capacity));
        std::fflush(stderr);
    }

    auto keepalive = std::make_tuple(
        candidateSeeds, candidateCounts, candidateReadyEpoch, candidateStatus,
        streamingEpoch, sourceLens, kvObjectAddressDomain, reqPoolEntries,
        kvObjectMap, kvObjectLens, kvObjectSlotsSnapshot, privateSourceToSlot,
        privateObjectToSlot, privateReadyTable, privateArenaHead,
        privatePackedOutput, fusedGetControl, queueLeases, queueEpoch,
        queueGeneration, queuePostSequence, queueState, queueQuarantine,
        queueRequestSourceOffsets, queueRequestObjectOffsets,
        queueRequestOutputIndices, queueRequestCount, privateReadyEpoch,
        privateReadyGeneration, workerStats, rowsPerEntry);
    // P7.1 directly enqueues ACLNN on the caller's current stream. The device
    // kernel, not a torch_npu-private host task queue, owns PostOnly/CQ work.
    EXEC_NPU_CMD_P7_ZERO_WORKSPACE_DIRECT_ASYNC(
        aclnnA5ScoreReceiptSpeculativeExecutor, keepalive,
        candidateSeeds, candidateCounts, candidateReadyEpoch, candidateStatus,
        streamingEpoch, sourceLens, kvObjectAddressDomain, reqPoolEntries,
        kvObjectMap, kvObjectLens, kvObjectSlotsSnapshot, privateSourceToSlot,
        privateObjectToSlot, privateReadyTable, privateArenaHead,
        privatePackedOutput, fusedGetControl, queueLeases, queueEpoch,
        queueGeneration, queuePostSequence, queueState, queueQuarantine,
        queueRequestSourceOffsets, queueRequestObjectOffsets,
        queueRequestOutputIndices, queueRequestCount, privateReadyEpoch,
        privateReadyGeneration, workerStats, rowsPerEntry);
    return std::make_tuple(
        privateSourceToSlot, privateObjectToSlot, privateReadyTable,
        privateArenaHead, privatePackedOutput, queueLeases, queueEpoch,
        queueGeneration, queuePostSequence, queueState, queueQuarantine,
        queueRequestSourceOffsets, queueRequestObjectOffsets,
        queueRequestOutputIndices, queueRequestCount, privateReadyEpoch,
        privateReadyGeneration, workerStats);
}

Result RunMeta(
    const at::Tensor& candidateSeeds, const at::Tensor& candidateCounts,
    const at::Tensor& candidateReadyEpoch, const at::Tensor& candidateStatus,
    const at::Tensor& streamingEpoch, const at::Tensor& sourceLens,
    const at::Tensor& kvObjectAddressDomain,
    const at::Tensor& reqPoolEntries, const at::Tensor& kvObjectMap,
    const at::Tensor& kvObjectLens, const at::Tensor& kvObjectSlotsSnapshot,
    at::Tensor privateSourceToSlot, at::Tensor privateObjectToSlot,
    at::Tensor privateReadyTable, at::Tensor privateArenaHead,
    at::Tensor privatePackedOutput, const at::Tensor& fusedGetControl,
    at::Tensor queueLeases, at::Tensor queueEpoch, at::Tensor queueGeneration,
    at::Tensor queuePostSequence, at::Tensor queueState,
    at::Tensor queueQuarantine, at::Tensor queueRequestSourceOffsets,
    at::Tensor queueRequestObjectOffsets, at::Tensor queueRequestOutputIndices,
    at::Tensor queueRequestCount, at::Tensor privateReadyEpoch,
    at::Tensor privateReadyGeneration, at::Tensor workerStats,
    const at::Tensor& rowsPerEntry)
{
    (void)ValidateShapes(
        candidateSeeds, candidateCounts, candidateReadyEpoch, candidateStatus,
        streamingEpoch, sourceLens, kvObjectAddressDomain, reqPoolEntries,
        kvObjectMap, kvObjectLens, kvObjectSlotsSnapshot, privateSourceToSlot,
        privateObjectToSlot, privateReadyTable, privateArenaHead,
        privatePackedOutput, fusedGetControl, queueLeases, queueEpoch,
        queueGeneration, queuePostSequence, queueState, queueQuarantine,
        queueRequestSourceOffsets, queueRequestObjectOffsets,
        queueRequestOutputIndices, queueRequestCount, privateReadyEpoch,
        privateReadyGeneration, workerStats, rowsPerEntry);
    return std::make_tuple(
        privateSourceToSlot, privateObjectToSlot, privateReadyTable,
        privateArenaHead, privatePackedOutput, queueLeases, queueEpoch,
        queueGeneration, queuePostSequence, queueState, queueQuarantine,
        queueRequestSourceOffsets, queueRequestObjectOffsets,
        queueRequestOutputIndices, queueRequestCount, privateReadyEpoch,
        privateReadyGeneration, workerStats);
}
}  // namespace

TORCH_LIBRARY(a5_score_receipt_speculative_executor, m)
{
    m.def(
        "score_receipt_speculative_executor("
        "Tensor candidate_seeds, Tensor candidate_counts, "
        "Tensor candidate_ready_epoch, Tensor candidate_status, "
        "Tensor streaming_epoch, Tensor source_lens, "
        "Tensor kv_object_address_domain, Tensor req_pool_entries, "
        "Tensor kv_object_map, Tensor kv_object_lens, "
        "Tensor kv_object_slots_snapshot, "
        "Tensor(a!) private_source_to_slot, Tensor(b!) private_object_to_slot, "
        "Tensor(c!) private_ready_table, Tensor(d!) private_arena_head, "
        "Tensor(e!) private_packed_output, Tensor fused_get_control, "
        "Tensor(f!) queue_leases, Tensor(g!) queue_epoch, "
        "Tensor(h!) queue_generation, Tensor(i!) queue_post_sequence, "
        "Tensor(j!) queue_state, Tensor(k!) queue_quarantine, "
        "Tensor(l!) queue_request_source_offsets, "
        "Tensor(m!) queue_request_object_offsets, "
        "Tensor(n!) queue_request_output_indices, "
        "Tensor(o!) queue_request_count, Tensor(p!) private_ready_epoch, "
        "Tensor(q!) private_ready_generation, Tensor(r!) worker_stats, "
        "Tensor rows_per_entry) -> ("
        "Tensor(a!) private_source_to_slot, Tensor(b!) private_object_to_slot, "
        "Tensor(c!) private_ready_table, Tensor(d!) private_arena_head, "
        "Tensor(e!) private_packed_output, Tensor(f!) queue_leases, "
        "Tensor(g!) queue_epoch, Tensor(h!) queue_generation, "
        "Tensor(i!) queue_post_sequence, Tensor(j!) queue_state, "
        "Tensor(k!) queue_quarantine, Tensor(l!) queue_request_source_offsets, "
        "Tensor(m!) queue_request_object_offsets, "
        "Tensor(n!) queue_request_output_indices, Tensor(o!) queue_request_count, "
        "Tensor(p!) private_ready_epoch, Tensor(q!) private_ready_generation, "
        "Tensor(r!) worker_stats)");
}

TORCH_LIBRARY_IMPL(a5_score_receipt_speculative_executor, PrivateUse1, m)
{
    m.impl("score_receipt_speculative_executor", &RunNpu);
}

TORCH_LIBRARY_IMPL(a5_score_receipt_speculative_executor, Meta, m)
{
    m.impl("score_receipt_speculative_executor", &RunMeta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("p7_1_abi_contract", [] {
        pybind11::dict contract;
        contract["abi"] = "31tensor_18refout";
        contract["torch_schema"] = "31in_18aliasout";
        contract["candidate_capacity"] = kCandidateCapacity;
        contract["candidate_seed_words"] = kCandidateSeedWords;
        contract["local_microbatches"] = "2x64";
        contract["queue_lifecycle"] =
            "epoch,generation,post_sequence,state,quarantine";
        contract["mutable_ref_outputs"] = kOutputCount;
        contract["c8_dependency"] = false;
        contract["fixed_get_core"] = false;
        return contract;
    });
    m.def("p7_1_release_after_forward_terminal", [] {
        const int64_t released =
            nanovllm_a5_opapi::ReleaseP7DirectLaunchLeasesAtForwardTerminal();
        if (DebugEnabled()) {
            std::fprintf(stderr,
                         "P7_1_TERMINAL_LEASE released=%lld pending=%lld "
                         "terminal=caller_confirmed_stream_terminal\n",
                         static_cast<long long>(released),
                         static_cast<long long>(
                             nanovllm_a5_opapi::PendingP7DirectLaunchLeases()));
            std::fflush(stderr);
        }
        return released;
    });
    m.def("p7_1_pending_launch_leases", [] {
        return nanovllm_a5_opapi::PendingP7DirectLaunchLeases();
    });
}
