// Standalone P2 bridge for the device-resident ExactScoreScan +
// SpeculativeGet executor.  It consumes the score-window publication from
// A5StreamingQuantLiGet; it does not accept synthetic float LI scores.

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

#include <torch/extension.h>
#include <torch/library.h>

#include "op_api_common.h"

namespace {
constexpr int64_t kProducerCount = 62;
constexpr int64_t kGetLaneCount = 2;
constexpr int64_t kRingSlots = 2;
constexpr int64_t kRecordsPerRingSlot = 32;
constexpr int64_t kRecordWords = 3;
constexpr int64_t kCacheLineWords = 8;
constexpr int64_t kRingSlotRecordWord = 3 * kCacheLineWords;
constexpr int64_t kRingWordsPerSlot =
    kRingSlotRecordWord + kRecordsPerRingSlot * kRecordWords;
constexpr int64_t kPipelineWordsPerRing =
    kRingSlots * kRingWordsPerSlot + kCacheLineWords;
constexpr int64_t kPipelineRingCount = kProducerCount * kGetLaneCount;
constexpr int64_t kGetWaveRecords = 256;
constexpr int64_t kGetDescriptorBufferCount = 2;
// Must exactly match the P5 device writer: the final two words expose
// receipt/source-len polling decisions for each ExactScan producer.
constexpr int64_t kProducerStatColumns = 10;
constexpr int64_t kGetStatColumns = 16;
constexpr int64_t kControlWords = 16;
constexpr int64_t kPackedEntryBytes = 656;
constexpr int64_t kReceiptTileWidth = 128;
constexpr int64_t kReceiptWords = 16;
constexpr int64_t kCandidateSeedWords = 5;
// The bridge owns all dynamically allocated transaction buffers.  Keep this
// explicit ABI marker in the loaded extension so a newly built OPP cannot be
// accidentally paired with an older bridge that allocated a differently
// sized pipeline_state tensor.  In particular, ABI 3 carries the three
// cache-line ring cursors introduced with the exact descriptor-wave fix:
// [62 * 2, 248], not the legacy [124, 232].
constexpr int64_t kP5BridgeAbiVersion = 3;

struct P2Geometry {
    int64_t rowCount;
    int64_t sourceCapacity;
    int64_t poolSize;
    int64_t kvObjectCapacity;
    int64_t privateArenaCapacity;
    int64_t tileCount;
    int64_t tileReceiptCount;
    int64_t candidateCapacityPerProducer;
};

bool P2OpApiDebugEnabled()
{
    const char* value = std::getenv("NANOVLLM_A5_OPAPI_DEBUG");
    return value != nullptr && value[0] != '\0' &&
           !(value[0] == '0' && value[1] == '\0');
}

std::string TensorVectorString(c10::IntArrayRef values)
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

void LogP2AclnnTensor(const char* role, const at::Tensor& tensor)
{
    if (!P2OpApiDebugEnabled()) {
        return;
    }
    const std::string shape = TensorVectorString(tensor.sizes());
    const std::string stride = TensorVectorString(tensor.strides());
    const std::string dtype = c10::toString(tensor.scalar_type());
    const std::string device = tensor.device().str();
    std::fprintf(
        stderr,
        "P2_ACLNN_TENSOR role=%s dtype=%s device=%s shape=%s stride=%s "
        "contiguous=%d nbytes=%zu data=%p\n",
        role, dtype.c_str(), device.c_str(), shape.c_str(), stride.c_str(),
        tensor.is_contiguous() ? 1 : 0, tensor.nbytes(), tensor.data_ptr());
}

using NamedTensor = std::pair<const char*, const at::Tensor*>;

void LogP2AclnnCall(std::initializer_list<NamedTensor> inputs,
                    std::initializer_list<NamedTensor> outputs)
{
    if (!P2OpApiDebugEnabled()) {
        return;
    }
    std::fprintf(stderr, "P2_ACLNN_CALL input_count=%zu output_count=%zu\n",
                 inputs.size(), outputs.size());
    for (const auto& entry : inputs) {
        LogP2AclnnTensor(entry.first, *entry.second);
    }
    for (const auto& entry : outputs) {
        LogP2AclnnTensor(entry.first, *entry.second);
    }
    std::fflush(stderr);
}

void CheckVector(const at::Tensor& tensor, int64_t expected, const char* name)
{
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == expected,
                name, " must have shape [", expected, "].");
}

int64_t CheckedCeilDiv(int64_t numerator, int64_t denominator,
                       const char* description)
{
    TORCH_CHECK(numerator > 0 && denominator > 0,
                description, " requires positive operands.");
    return numerator / denominator + (numerator % denominator != 0 ? 1 : 0);
}

P2Geometry DeriveGeometry(const at::Tensor& residentThreshold,
                          const at::Tensor& kvObjectSlotsSnapshot,
                          const at::Tensor& kvObjectMap,
                          const at::Tensor& privateReady)
{
    TORCH_CHECK(residentThreshold.dim() == 1 && residentThreshold.numel() > 0,
                "resident_threshold must be int32[R], R > 0.");
    TORCH_CHECK(kvObjectSlotsSnapshot.dim() == 2 &&
                    kvObjectSlotsSnapshot.size(0) > 0 &&
                    kvObjectSlotsSnapshot.size(1) > 0,
                "kv_object_slots_snapshot must be int32[P, Ocap], both positive.");
    TORCH_CHECK(kvObjectMap.dim() == 2 && kvObjectMap.size(0) > 0 &&
                    kvObjectMap.size(1) > 0,
                "kv_object_map must be int32[P, S], both positive.");
    TORCH_CHECK(privateReady.dim() == 2 &&
                    privateReady.size(0) == kGetLaneCount &&
                    privateReady.size(1) > 0,
                "private_ready_table must be int32[", kGetLaneCount,
                ", A], A > 0 for the fixed two-lane P2 executor.");

    const int64_t rowCount = residentThreshold.numel();
    const int64_t poolSize = kvObjectMap.size(0);
    const int64_t sourceCapacity = kvObjectMap.size(1);
    const int64_t kvObjectCapacity = kvObjectSlotsSnapshot.size(1);
    const int64_t privateArenaCapacity = privateReady.size(1);
    TORCH_CHECK(kvObjectSlotsSnapshot.size(0) == poolSize,
                "kv_object_slots_snapshot and kv_object_map must agree on P: ",
                kvObjectSlotsSnapshot.size(0), " versus ", poolSize, ".");

    const int64_t tileCount = CheckedCeilDiv(
        sourceCapacity, kReceiptTileWidth, "P2 score-window tile count");
    TORCH_CHECK(rowCount <= std::numeric_limits<int64_t>::max() / tileCount,
                "rowCount * tileCount overflows int64.");
    const int64_t tileReceiptCount = rowCount * tileCount;
    const int64_t receiptStripes = CheckedCeilDiv(
        tileReceiptCount, kProducerCount, "P2 candidate receipt stripes");
    TORCH_CHECK(receiptStripes <=
                    std::numeric_limits<int64_t>::max() / kReceiptTileWidth,
                "candidate capacity overflows int64.");
    return {rowCount, sourceCapacity, poolSize, kvObjectCapacity,
            privateArenaCapacity, tileCount, tileReceiptCount,
            receiptStripes * kReceiptTileWidth};
}

P2Geometry ValidateShapeContract(
    const at::Tensor& scoreWorkspace, const at::Tensor& tileReceipts,
    const at::Tensor& streamingEpoch, const at::Tensor& scoreSourceLens,
    const at::Tensor& residentThreshold,
    const at::Tensor& residentThresholdValid,
    const at::Tensor& kvObjectSlotsSnapshot,
    const at::Tensor& reqPoolEntries, const at::Tensor& kvObjectMap,
    const at::Tensor& kvObjectLens, const at::Tensor& privateSourceToSlot,
    const at::Tensor& privateObjectToSlot, const at::Tensor& privateReady,
    const at::Tensor& privateArenaHead, const at::Tensor& privatePackedOutput,
    const at::Tensor& addressTable, const at::Tensor& validTable,
    const at::Tensor& fusedGetControl, const at::Tensor& pipelineState,
    const at::Tensor& executionMode)
{
    const P2Geometry geometry = DeriveGeometry(
        residentThreshold, kvObjectSlotsSnapshot, kvObjectMap, privateReady);
    TORCH_CHECK(scoreWorkspace.dim() == 1 && scoreWorkspace.numel() > 0,
                "score_workspace must be int32[W], W > 0; host tiling validates "
                "the C8 AIC-layout minimum.");
    TORCH_CHECK(tileReceipts.dim() == 2 &&
                    tileReceipts.size(0) == geometry.tileReceiptCount &&
                    tileReceipts.size(1) == kReceiptWords,
                "tile_receipts must be int64[R*ceil(S/128), ", kReceiptWords,
                "] with R=", geometry.rowCount, ", S=", geometry.sourceCapacity,
                ".");
    CheckVector(streamingEpoch, 1, "streaming_epoch");
    // This is graph-static `active_score_source_lens`, supplied before the
    // forward.  A zero span means the corresponding fixed receipt slots are
    // inactive, not zero-score KV entries.  It must not be connected to the
    // similarly shaped C8 ProcessTopK output because that output becomes
    // available only after the tile producer has already finished.
    CheckVector(scoreSourceLens, geometry.rowCount, "score_source_lens");
    CheckVector(residentThresholdValid, geometry.rowCount,
                "resident_threshold_valid");
    CheckVector(reqPoolEntries, geometry.rowCount, "req_pool_entries");
    CheckVector(kvObjectLens, geometry.rowCount, "kv_object_lens");
    TORCH_CHECK(privateSourceToSlot.dim() == 2 &&
                    privateSourceToSlot.size(0) == geometry.rowCount &&
                    privateSourceToSlot.size(1) == geometry.sourceCapacity,
                "private_source_to_slot must be int32[R, S].");
    TORCH_CHECK(privateObjectToSlot.dim() == 2 &&
                    privateObjectToSlot.size(0) == geometry.poolSize &&
                    privateObjectToSlot.size(1) == geometry.kvObjectCapacity,
                "private_object_to_slot must be int32[P, Ocap].");
    CheckVector(privateArenaHead, kGetLaneCount, "private_arena_head");
    TORCH_CHECK(privatePackedOutput.dim() == 2 &&
                    privatePackedOutput.size(0) ==
                        kGetLaneCount * geometry.privateArenaCapacity &&
                    privatePackedOutput.size(1) == kPackedEntryBytes,
                "private_packed_output must be uint8[", kGetLaneCount,
                "*A, ", kPackedEntryBytes, "].");
    TORCH_CHECK(addressTable.dim() == 1 && addressTable.numel() > 0,
                "address_table must be int64[Tcap], Tcap > 0.");
    CheckVector(validTable, addressTable.numel(), "valid_table");
    CheckVector(fusedGetControl, kControlWords, "fused_get_control");
    TORCH_CHECK(pipelineState.dim() == 2 &&
                    pipelineState.size(0) == kPipelineRingCount &&
                    pipelineState.size(1) == kPipelineWordsPerRing,
                "pipeline_state must be int64[", kPipelineRingCount, ", ",
                kPipelineWordsPerRing,
                "] and retain a stable address across graph replays.");
    CheckVector(executionMode, 1, "execution_mode");
    return geometry;
}

P2Geometry CheckInputs(
    const at::Tensor& scoreWorkspace, const at::Tensor& tileReceipts,
    const at::Tensor& streamingEpoch, const at::Tensor& scoreSourceLens,
    const at::Tensor& residentThreshold,
    const at::Tensor& residentThresholdValid,
    const at::Tensor& kvObjectSlotsSnapshot,
    const at::Tensor& reqPoolEntries, const at::Tensor& kvObjectMap,
    const at::Tensor& kvObjectLens, const at::Tensor& privateSourceToSlot,
    const at::Tensor& privateObjectToSlot, const at::Tensor& privateReady,
    const at::Tensor& privateArenaHead, const at::Tensor& privatePackedOutput,
    const at::Tensor& addressTable, const at::Tensor& validTable,
    const at::Tensor& fusedGetControl, const at::Tensor& pipelineState,
    const at::Tensor& executionMode)
{
    TORCH_CHECK(scoreWorkspace.device().is_privateuseone(),
                "exact_scan_speculative_get requires NPU inputs.");
    const at::Device device = scoreWorkspace.device();
    for (const at::Tensor& tensor : {
             tileReceipts, streamingEpoch, scoreSourceLens, residentThreshold,
             residentThresholdValid, kvObjectSlotsSnapshot, reqPoolEntries,
             kvObjectMap, kvObjectLens, privateSourceToSlot,
             privateObjectToSlot, privateReady, privateArenaHead,
             privatePackedOutput, addressTable, validTable, fusedGetControl,
             pipelineState, executionMode}) {
        TORCH_CHECK(tensor.device() == device,
                    "exact_scan_speculative_get inputs must use one NPU.");
        TORCH_CHECK(tensor.is_contiguous(),
                    "exact_scan_speculative_get inputs must be contiguous.");
    }
    TORCH_CHECK(scoreWorkspace.is_contiguous(),
                "exact_scan_speculative_get inputs must be contiguous.");
    TORCH_CHECK(
        scoreWorkspace.scalar_type() == at::kInt &&
            tileReceipts.scalar_type() == at::kLong &&
            streamingEpoch.scalar_type() == at::kLong &&
            scoreSourceLens.scalar_type() == at::kInt &&
            residentThreshold.scalar_type() == at::kInt &&
            residentThresholdValid.scalar_type() == at::kInt &&
            kvObjectSlotsSnapshot.scalar_type() == at::kInt &&
            reqPoolEntries.scalar_type() == at::kInt &&
            kvObjectMap.scalar_type() == at::kInt &&
            kvObjectLens.scalar_type() == at::kInt &&
            privateSourceToSlot.scalar_type() == at::kInt &&
            privateObjectToSlot.scalar_type() == at::kInt &&
            privateReady.scalar_type() == at::kInt &&
            privateArenaHead.scalar_type() == at::kInt &&
            privatePackedOutput.scalar_type() == at::kByte &&
            addressTable.scalar_type() == at::kLong &&
            validTable.scalar_type() == at::kByte &&
            fusedGetControl.scalar_type() == at::kLong &&
            pipelineState.scalar_type() == at::kLong &&
            executionMode.scalar_type() == at::kLong,
        "exact_scan_speculative_get input dtypes do not match its P2 ABI.");
    return ValidateShapeContract(
        scoreWorkspace, tileReceipts, streamingEpoch, scoreSourceLens,
        residentThreshold,
        residentThresholdValid, kvObjectSlotsSnapshot, reqPoolEntries,
        kvObjectMap, kvObjectLens, privateSourceToSlot, privateObjectToSlot,
        privateReady, privateArenaHead, privatePackedOutput, addressTable,
        validTable, fusedGetControl, pipelineState, executionMode);
}

using Result = std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor>;

Result ExactScanSpeculativeGetNpu(
    const at::Tensor& scoreWorkspace, const at::Tensor& tileReceipts,
    const at::Tensor& streamingEpoch, const at::Tensor& scoreSourceLens,
    const at::Tensor& residentThreshold,
    const at::Tensor& residentThresholdValid,
    const at::Tensor& kvObjectSlotsSnapshot,
    const at::Tensor& reqPoolEntries, const at::Tensor& kvObjectMap,
    const at::Tensor& kvObjectLens, at::Tensor privateSourceToSlot,
    at::Tensor privateObjectToSlot, at::Tensor privateReady,
    at::Tensor privateArenaHead, at::Tensor privatePackedOutput,
    const at::Tensor& addressTable, const at::Tensor& validTable,
    const at::Tensor& fusedGetControl, at::Tensor pipelineState,
    const at::Tensor& executionMode)
{
    const P2Geometry geometry = CheckInputs(
        scoreWorkspace, tileReceipts, streamingEpoch, scoreSourceLens,
        residentThreshold,
        residentThresholdValid, kvObjectSlotsSnapshot, reqPoolEntries,
        kvObjectMap, kvObjectLens, privateSourceToSlot, privateObjectToSlot,
        privateReady, privateArenaHead, privatePackedOutput, addressTable,
        validTable, fusedGetControl, pipelineState, executionMode);
    const auto int32Options = scoreWorkspace.options().dtype(at::kInt);
    const auto int64Options = scoreWorkspace.options().dtype(at::kLong);
    const auto uint8Options = scoreWorkspace.options().dtype(at::kByte);
    auto candidateSeeds = at::empty(
        {kProducerCount, geometry.candidateCapacityPerProducer,
         kCandidateSeedWords},
        int32Options);
    auto candidateCounts = at::empty({kProducerCount}, int32Options);
    auto producerStats = at::empty({kProducerCount, kProducerStatColumns},
                                   int64Options);
    auto getStats = at::empty({kGetLaneCount, kGetStatColumns}, int64Options);
    auto roleTrace = at::empty({kProducerCount + kGetLaneCount}, int32Options);
    auto getIndices = at::empty(
        {kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
        int32Options);
    auto getOffsets = at::empty(
        {kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
        int32Options);
    auto getOutputIndices = at::empty(
        {kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
        int32Options);
    auto getMask = at::empty(
        {kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
        uint8Options);
    auto getCandidateSeedSlots = at::empty(
        {kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
        int32Options);
    auto getClaimNew = at::empty(
        {kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
        uint8Options);

    // This bounded diagnostic bundle is deliberately exhaustive: a generated
    // ACLNN ABI mismatch can otherwise look like a device-kernel bug.
    LogP2AclnnCall(
        {{"in0.score_workspace", &scoreWorkspace},
         {"in1.tile_receipts", &tileReceipts},
         {"in2.streaming_epoch", &streamingEpoch},
         {"in3.score_source_lens", &scoreSourceLens},
         {"in4.resident_threshold", &residentThreshold},
         {"in5.resident_threshold_valid", &residentThresholdValid},
         {"in6.kv_object_slots_snapshot", &kvObjectSlotsSnapshot},
         {"in7.req_pool_entries", &reqPoolEntries},
         {"in8.kv_object_map", &kvObjectMap},
         {"in9.kv_object_lens", &kvObjectLens},
         {"in10.private_source_to_slot", &privateSourceToSlot},
         {"in11.private_object_to_slot", &privateObjectToSlot},
         {"in12.private_ready_table", &privateReady},
         {"in13.private_arena_head", &privateArenaHead},
         {"in14.private_packed_output", &privatePackedOutput},
         {"in15.address_table", &addressTable},
         {"in16.valid_table", &validTable},
         {"in17.fused_get_control", &fusedGetControl},
         {"in18.pipeline_state", &pipelineState},
         {"in19.execution_mode", &executionMode}},
        {{"out0.private_source_to_slot_out", &privateSourceToSlot},
         {"out1.private_object_to_slot_out", &privateObjectToSlot},
         {"out2.private_ready_table_out", &privateReady},
         {"out3.private_arena_head_out", &privateArenaHead},
         {"out4.private_packed_output_out", &privatePackedOutput},
         {"out5.candidate_seeds", &candidateSeeds},
         {"out6.candidate_counts", &candidateCounts},
         {"out7.producer_stats", &producerStats},
         {"out8.get_stats", &getStats},
         {"out9.role_trace", &roleTrace},
         {"out10.get_indices", &getIndices},
         {"out11.get_offsets", &getOffsets},
         {"out12.get_output_indices", &getOutputIndices},
         {"out13.get_mask", &getMask},
         {"out14.get_candidate_seed_slots", &getCandidateSeedSlots},
         {"out15.get_claim_new", &getClaimNew},
         {"out16.pipeline_state_out", &pipelineState}});

    auto keepalive = std::make_tuple(
        scoreWorkspace, tileReceipts, streamingEpoch, scoreSourceLens,
        residentThreshold,
        residentThresholdValid, kvObjectSlotsSnapshot, reqPoolEntries,
        kvObjectMap, kvObjectLens, privateSourceToSlot, privateObjectToSlot,
        privateReady, privateArenaHead, privatePackedOutput, addressTable,
        validTable, fusedGetControl, pipelineState, executionMode,
        candidateSeeds, candidateCounts, producerStats, getStats, roleTrace,
        getIndices, getOffsets, getOutputIndices, getMask,
        getCandidateSeedSlots, getClaimNew);
    EXEC_NPU_CMD_ORDERED(
        aclnnA5ExactScanSpeculativeGet, keepalive,
        scoreWorkspace, tileReceipts, streamingEpoch, scoreSourceLens,
        residentThreshold,
        residentThresholdValid, kvObjectSlotsSnapshot, reqPoolEntries,
        kvObjectMap, kvObjectLens, privateSourceToSlot, privateObjectToSlot,
        privateReady, privateArenaHead, privatePackedOutput, addressTable,
        validTable, fusedGetControl, pipelineState, executionMode,
        privateSourceToSlot, privateObjectToSlot, privateReady,
        privateArenaHead, privatePackedOutput, candidateSeeds, candidateCounts,
        producerStats, getStats, roleTrace, getIndices, getOffsets,
        getOutputIndices, getMask, getCandidateSeedSlots, getClaimNew,
        pipelineState);
    return std::make_tuple(
        privateSourceToSlot, privateObjectToSlot, privateReady,
        privateArenaHead, privatePackedOutput, candidateSeeds, candidateCounts,
        producerStats, getStats, roleTrace, getIndices, getOffsets,
        getOutputIndices, getMask, getCandidateSeedSlots, getClaimNew,
        pipelineState);
}

Result ExactScanSpeculativeGetMeta(
    const at::Tensor& scoreWorkspace, const at::Tensor& tileReceipts,
    const at::Tensor& streamingEpoch, const at::Tensor& scoreSourceLens,
    const at::Tensor& residentThreshold,
    const at::Tensor& residentThresholdValid,
    const at::Tensor& kvObjectSlotsSnapshot,
    const at::Tensor& reqPoolEntries, const at::Tensor& kvObjectMap,
    const at::Tensor& kvObjectLens, at::Tensor privateSourceToSlot,
    at::Tensor privateObjectToSlot, at::Tensor privateReady,
    at::Tensor privateArenaHead, at::Tensor privatePackedOutput,
    const at::Tensor& addressTable, const at::Tensor& validTable,
    const at::Tensor& fusedGetControl, at::Tensor pipelineState,
    const at::Tensor& executionMode)
{
    const P2Geometry geometry = ValidateShapeContract(
        scoreWorkspace, tileReceipts, streamingEpoch, scoreSourceLens,
        residentThreshold,
        residentThresholdValid, kvObjectSlotsSnapshot, reqPoolEntries,
        kvObjectMap, kvObjectLens, privateSourceToSlot, privateObjectToSlot,
        privateReady, privateArenaHead, privatePackedOutput, addressTable,
        validTable, fusedGetControl, pipelineState, executionMode);
    const auto int32Options = scoreWorkspace.options().dtype(at::kInt);
    const auto int64Options = scoreWorkspace.options().dtype(at::kLong);
    const auto uint8Options = scoreWorkspace.options().dtype(at::kByte);
    return std::make_tuple(
        privateSourceToSlot, privateObjectToSlot, privateReady,
        privateArenaHead, privatePackedOutput,
        at::empty({kProducerCount, geometry.candidateCapacityPerProducer,
                   kCandidateSeedWords}, int32Options),
        at::empty({kProducerCount}, int32Options),
        at::empty({kProducerCount, kProducerStatColumns}, int64Options),
        at::empty({kGetLaneCount, kGetStatColumns}, int64Options),
        at::empty({kProducerCount + kGetLaneCount}, int32Options),
        at::empty({kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
                  int32Options),
        at::empty({kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
                  int32Options),
        at::empty({kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
                  int32Options),
        at::empty({kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
                  uint8Options),
        at::empty({kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
                  int32Options),
        at::empty({kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
                  uint8Options),
        pipelineState);
}
}  // namespace

TORCH_LIBRARY(a5_exact_scan_get, m)
{
    m.def(
        "exact_scan_speculative_get("
        "Tensor score_workspace, Tensor tile_receipts, Tensor streaming_epoch, "
        "Tensor score_source_lens, Tensor resident_threshold, "
        "Tensor resident_threshold_valid, "
        "Tensor kv_object_slots_snapshot, Tensor req_pool_entries, "
        "Tensor kv_object_map, Tensor kv_object_lens, "
        "Tensor(a!) private_source_to_slot, "
        "Tensor(b!) private_object_to_slot, "
        "Tensor(c!) private_ready_table, Tensor(d!) private_arena_head, "
        "Tensor(e!) private_packed_output, Tensor address_table, "
        "Tensor valid_table, Tensor fused_get_control, "
        "Tensor(f!) pipeline_state, Tensor execution_mode) "
        "-> (Tensor(a!) private_source_to_slot, "
        "Tensor(b!) private_object_to_slot, "
        "Tensor(c!) private_ready_table, Tensor(d!) private_arena_head, "
        "Tensor(e!) private_packed_output, Tensor candidate_seeds, "
        "Tensor candidate_counts, Tensor producer_stats, Tensor get_stats, "
        "Tensor role_trace, Tensor get_indices, Tensor get_offsets, "
        "Tensor get_output_indices, Tensor get_mask, "
        "Tensor get_candidate_seed_slots, Tensor get_claim_new, "
        "Tensor(f!) pipeline_state)");
}

TORCH_LIBRARY_IMPL(a5_exact_scan_get, PrivateUse1, m)
{
    m.impl("exact_scan_speculative_get", &ExactScanSpeculativeGetNpu);
}

TORCH_LIBRARY_IMPL(a5_exact_scan_get, Meta, m)
{
    m.impl("exact_scan_speculative_get", &ExactScanSpeculativeGetMeta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // Expose a tiny, allocation-free contract probe.  The live integration
    // test calls it *before* it initializes Direct-HCCP, so a stale .so is
    // reported as a clear bridge/OPP ABI mismatch rather than an opaque host
    // tiling failure followed by runtime cleanup noise.
    m.def("p5_abi_contract", [] {
        pybind11::dict contract;
        contract["abi_version"] = kP5BridgeAbiVersion;
        contract["producer_count"] = kProducerCount;
        contract["get_lane_count"] = kGetLaneCount;
        contract["ring_slots"] = kRingSlots;
        contract["ring_slot_record_word"] = kRingSlotRecordWord;
        contract["records_per_ring_slot"] = kRecordsPerRingSlot;
        contract["record_words"] = kRecordWords;
        contract["pipeline_words_per_ring"] = kPipelineWordsPerRing;
        contract["pipeline_ring_count"] = kPipelineRingCount;
        contract["candidate_seed_words"] = kCandidateSeedWords;
        contract["producer_stat_columns"] = kProducerStatColumns;
        contract["get_stat_columns"] = kGetStatColumns;
        contract["packed_entry_bytes"] = kPackedEntryBytes;
        contract["get_wave_records"] = kGetWaveRecords;
        return contract;
    });
}
