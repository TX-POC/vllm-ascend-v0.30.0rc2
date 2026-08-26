// Standalone P1 bridge.  It deliberately lives outside vllm_ascend_C so the
// real device data path can be iterated without an editable vLLM rebuild.

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <tuple>

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
constexpr int64_t kRingSlotRecordWord = 2 * kCacheLineWords;
constexpr int64_t kRingWordsPerSlot =
    kRingSlotRecordWord + kRecordsPerRingSlot * kRecordWords;
constexpr int64_t kPipelineWordsPerRing =
    kRingSlots * kRingWordsPerSlot + kCacheLineWords;
constexpr int64_t kPipelineRingCount = kProducerCount * kGetLaneCount;
constexpr int64_t kGetWaveRecords = 256;
constexpr int64_t kGetDescriptorBufferCount = 2;
constexpr int64_t kProducerStatColumns = 8;
constexpr int64_t kGetStatColumns = 16;
constexpr int64_t kControlWords = 16;
constexpr int64_t kPackedEntryBytes = 656;

bool P1OpApiDebugEnabled()
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

void LogP1AclnnTensor(const char* role, const at::Tensor& tensor)
{
    if (!P1OpApiDebugEnabled()) {
        return;
    }
    const std::string shape = TensorVectorString(tensor.sizes());
    const std::string stride = TensorVectorString(tensor.strides());
    const std::string dtype = c10::toString(tensor.scalar_type());
    const std::string device = tensor.device().str();
    std::fprintf(
        stderr,
        "P1_ACLNN_TENSOR role=%s dtype=%s device=%s shape=%s stride=%s "
        "contiguous=%d nbytes=%zu data=%p\n",
        role, dtype.c_str(), device.c_str(), shape.c_str(), stride.c_str(),
        tensor.is_contiguous() ? 1 : 0, tensor.nbytes(), tensor.data_ptr());
}

void LogP1AclnnCall(
    const at::Tensor& liScores, const at::Tensor& threshold,
    const at::Tensor& resident, const at::Tensor& tableIndices,
    const at::Tensor& offsets, const at::Tensor& outputIndices,
    const at::Tensor& addressTable, const at::Tensor& validTable,
    const at::Tensor& packedOutput, const at::Tensor& fusedGetControl,
    const at::Tensor& frameEpoch, const at::Tensor& pipelineState,
    const at::Tensor& executionMode, const at::Tensor& candidateIndices,
    const at::Tensor& candidateKind, const at::Tensor& producerStats,
    const at::Tensor& getStats, const at::Tensor& roleTrace,
    const at::Tensor& getIndices, const at::Tensor& getOffsets,
    const at::Tensor& getOutputIndices, const at::Tensor& getMask)
{
    if (!P1OpApiDebugEnabled()) {
        return;
    }
    std::fprintf(stderr, "P1_ACLNN_CALL input_count=13 output_count=11\n");
    LogP1AclnnTensor("in0.li_scores", liScores);
    LogP1AclnnTensor("in1.threshold", threshold);
    LogP1AclnnTensor("in2.resident", resident);
    LogP1AclnnTensor("in3.table_indices", tableIndices);
    LogP1AclnnTensor("in4.offsets", offsets);
    LogP1AclnnTensor("in5.output_indices", outputIndices);
    LogP1AclnnTensor("in6.address_table", addressTable);
    LogP1AclnnTensor("in7.valid_table", validTable);
    LogP1AclnnTensor("in8.packed_output", packedOutput);
    LogP1AclnnTensor("in9.fused_get_control", fusedGetControl);
    LogP1AclnnTensor("in10.frame_epoch", frameEpoch);
    LogP1AclnnTensor("in11.pipeline_state", pipelineState);
    LogP1AclnnTensor("in12.execution_mode", executionMode);
    LogP1AclnnTensor("out0.packed_output_out", packedOutput);
    LogP1AclnnTensor("out1.candidate_indices", candidateIndices);
    LogP1AclnnTensor("out2.candidate_kind", candidateKind);
    LogP1AclnnTensor("out3.producer_stats", producerStats);
    LogP1AclnnTensor("out4.get_stats", getStats);
    LogP1AclnnTensor("out5.role_trace", roleTrace);
    LogP1AclnnTensor("out6.get_indices", getIndices);
    LogP1AclnnTensor("out7.get_offsets", getOffsets);
    LogP1AclnnTensor("out8.get_output_indices", getOutputIndices);
    LogP1AclnnTensor("out9.get_mask", getMask);
    LogP1AclnnTensor("out10.pipeline_state_out", pipelineState);
    std::fflush(stderr);
}

void CheckVector(const at::Tensor& tensor, int64_t expected, const char* name)
{
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == expected,
                name, " must have shape [", expected, "].");
}

void CheckInputs(
    const at::Tensor& liScores, const at::Tensor& threshold,
    const at::Tensor& resident, const at::Tensor& tableIndices,
    const at::Tensor& offsets, const at::Tensor& outputIndices,
    const at::Tensor& addressTable, const at::Tensor& validTable,
    const at::Tensor& packedOutput, const at::Tensor& fusedGetControl,
    const at::Tensor& frameEpoch, const at::Tensor& pipelineState,
    const at::Tensor& executionMode)
{
    TORCH_CHECK(liScores.device().is_privateuseone(),
                "exact_scan_speculative_get requires NPU inputs.");
    const at::Device device = liScores.device();
    for (const auto& tensor : {threshold, resident, tableIndices, offsets,
                               outputIndices, addressTable, validTable,
                               packedOutput, fusedGetControl, frameEpoch,
                               pipelineState, executionMode}) {
        TORCH_CHECK(tensor.device() == device,
                    "exact_scan_speculative_get inputs must use one NPU.");
        TORCH_CHECK(tensor.is_contiguous(),
                    "exact_scan_speculative_get inputs must be contiguous.");
    }
    TORCH_CHECK(liScores.is_contiguous(),
                "exact_scan_speculative_get inputs must be contiguous.");
    TORCH_CHECK(liScores.scalar_type() == at::kFloat &&
                    threshold.scalar_type() == at::kFloat &&
                    resident.scalar_type() == at::kByte &&
                    tableIndices.scalar_type() == at::kInt &&
                    offsets.scalar_type() == at::kInt &&
                    outputIndices.scalar_type() == at::kInt &&
                    addressTable.scalar_type() == at::kLong &&
                    validTable.scalar_type() == at::kByte &&
                    packedOutput.scalar_type() == at::kByte &&
                    fusedGetControl.scalar_type() == at::kLong &&
                    frameEpoch.scalar_type() == at::kLong &&
                    pipelineState.scalar_type() == at::kLong &&
                    executionMode.scalar_type() == at::kLong,
                "exact_scan_speculative_get input dtypes do not match its ABI.");
    TORCH_CHECK(liScores.dim() == 1 && liScores.numel() >= kProducerCount,
                "li_scores must be float32[N], N >= 62.");
    CheckVector(threshold, 1, "threshold");
    CheckVector(resident, liScores.numel(), "resident");
    CheckVector(tableIndices, liScores.numel(), "table_indices");
    CheckVector(offsets, liScores.numel(), "offsets");
    CheckVector(outputIndices, liScores.numel(), "output_indices");
    TORCH_CHECK(addressTable.dim() == 1 && addressTable.numel() > 0,
                "address_table must be int64[T], T > 0.");
    CheckVector(validTable, addressTable.numel(), "valid_table");
    TORCH_CHECK(packedOutput.dim() == 2 && packedOutput.size(0) > 0 &&
                    packedOutput.size(1) == kPackedEntryBytes,
                "packed_output must be uint8[output_rows, 656].");
    CheckVector(fusedGetControl, kControlWords, "fused_get_control");
    CheckVector(frameEpoch, 1, "frame_epoch");
    CheckVector(executionMode, 1, "execution_mode");
    TORCH_CHECK(pipelineState.dim() == 2 &&
                    pipelineState.size(0) == kPipelineRingCount &&
                    pipelineState.size(1) == kPipelineWordsPerRing,
                "pipeline_state must be int64[", kPipelineRingCount,
                ", ", kPipelineWordsPerRing,
                "] and retain a stable address across graph replays.");
}

using Result = std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor>;

Result ExactScanSpeculativeGetNpu(
    const at::Tensor& liScores, const at::Tensor& threshold,
    const at::Tensor& resident, const at::Tensor& tableIndices,
    const at::Tensor& offsets, const at::Tensor& outputIndices,
    const at::Tensor& addressTable, const at::Tensor& validTable,
    at::Tensor packedOutput, const at::Tensor& fusedGetControl,
    const at::Tensor& frameEpoch, at::Tensor pipelineState,
    const at::Tensor& executionMode)
{
    CheckInputs(liScores, threshold, resident, tableIndices, offsets,
                outputIndices, addressTable, validTable, packedOutput,
                fusedGetControl, frameEpoch, pipelineState, executionMode);
    const auto int32Options = liScores.options().dtype(at::kInt);
    const auto int64Options = liScores.options().dtype(at::kLong);
    const auto uint8Options = liScores.options().dtype(at::kByte);
    const int64_t n = liScores.numel();
    auto candidateIndices = at::empty({n}, int32Options);
    auto candidateKind = at::empty({n}, uint8Options);
    auto producerStats = at::empty({kProducerCount, kProducerStatColumns},
                                   int64Options);
    auto getStats = at::empty({kGetLaneCount, kGetStatColumns}, int64Options);
    auto roleTrace = at::empty({kProducerCount + kGetLaneCount}, int32Options);
    // P2 retains a posted wave's descriptor storage until its CQ terminal
    // state and compacts the next exact miss wave into the alternate buffer.
    auto getIndices = at::empty(
        {kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
        int32Options);
    auto getOffsets = at::empty(
        {kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
        int32Options);
    auto getOutputIndices =
        at::empty(
            {kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
            int32Options);
    auto getMask = at::empty(
        {kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
        uint8Options);

    // One opt-in, bounded diagnostic bundle for failures raised inside the
    // generated GetWorkspaceSize call.  It includes all 13 inputs and 11
    // outputs, so the remote log can be compared mechanically with ops.json
    // and the host registration without requesting another ad-hoc print.
    LogP1AclnnCall(
        liScores, threshold, resident, tableIndices, offsets, outputIndices,
        addressTable, validTable, packedOutput, fusedGetControl, frameEpoch,
        pipelineState, executionMode, candidateIndices, candidateKind,
        producerStats, getStats, roleTrace, getIndices, getOffsets,
        getOutputIndices, getMask);

    auto keepalive = std::make_tuple(
        liScores, threshold, resident, tableIndices, offsets, outputIndices,
        addressTable, validTable, packedOutput, fusedGetControl, frameEpoch,
        pipelineState, executionMode, candidateIndices, candidateKind, producerStats, getStats,
        roleTrace, getIndices, getOffsets, getOutputIndices, getMask);
    // CANN sees both mutable buffers as genuine outputs.  This is important:
    // the graph dataflow now contains the H2D-written packed KV and the
    // recurrent ring state instead of treating them as read-only inputs.
    EXEC_NPU_CMD_ORDERED(
        aclnnA5ExactScanSpeculativeGet, keepalive,
        liScores, threshold, resident, tableIndices, offsets, outputIndices,
        addressTable, validTable, packedOutput, fusedGetControl, frameEpoch,
        pipelineState, executionMode, packedOutput, candidateIndices, candidateKind,
        producerStats, getStats, roleTrace, getIndices, getOffsets,
        getOutputIndices, getMask, pipelineState);
    return std::make_tuple(
        packedOutput, candidateIndices, candidateKind, producerStats, getStats,
        roleTrace, getIndices, getOffsets, getOutputIndices, getMask,
        pipelineState);
}

Result ExactScanSpeculativeGetMeta(
    const at::Tensor& liScores, const at::Tensor& threshold,
    const at::Tensor& resident, const at::Tensor& tableIndices,
    const at::Tensor& offsets, const at::Tensor& outputIndices,
    const at::Tensor& addressTable, const at::Tensor& validTable,
    at::Tensor packedOutput, const at::Tensor& fusedGetControl,
    const at::Tensor& frameEpoch, at::Tensor pipelineState,
    const at::Tensor& executionMode)
{
    TORCH_CHECK(liScores.dim() == 1 && liScores.numel() >= kProducerCount &&
                    threshold.dim() == 1 && threshold.numel() == 1 &&
                    resident.dim() == 1 && resident.numel() == liScores.numel() &&
                    tableIndices.dim() == 1 &&
                    tableIndices.numel() == liScores.numel() &&
                    offsets.dim() == 1 && offsets.numel() == liScores.numel() &&
                    outputIndices.dim() == 1 &&
                    outputIndices.numel() == liScores.numel() &&
                    addressTable.dim() == 1 && addressTable.numel() > 0 &&
                    validTable.dim() == 1 &&
                    validTable.numel() == addressTable.numel() &&
                    packedOutput.dim() == 2 &&
                    packedOutput.size(1) == kPackedEntryBytes &&
                    fusedGetControl.dim() == 1 &&
                    fusedGetControl.numel() == kControlWords &&
                    frameEpoch.dim() == 1 && frameEpoch.numel() == 1 &&
                    executionMode.dim() == 1 && executionMode.numel() == 1 &&
                    pipelineState.dim() == 2 &&
                    pipelineState.size(0) == kPipelineRingCount &&
                    pipelineState.size(1) == kPipelineWordsPerRing,
                "exact_scan_speculative_get meta shapes are invalid.");
    const auto int32Options = liScores.options().dtype(at::kInt);
    const auto int64Options = liScores.options().dtype(at::kLong);
    const auto uint8Options = liScores.options().dtype(at::kByte);
    const int64_t n = liScores.numel();
    return std::make_tuple(
        packedOutput, at::empty({n}, int32Options), at::empty({n}, uint8Options),
        at::empty({kProducerCount, kProducerStatColumns}, int64Options),
        at::empty({kGetLaneCount, kGetStatColumns}, int64Options),
        at::empty({kProducerCount + kGetLaneCount}, int32Options),
        at::empty(
            {kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
            int32Options),
        at::empty(
            {kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
            int32Options),
        at::empty(
            {kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
            int32Options),
        at::empty(
            {kGetLaneCount, kGetDescriptorBufferCount, kGetWaveRecords},
            uint8Options),
        pipelineState);
}
}  // namespace

TORCH_LIBRARY(a5_exact_scan_get, m)
{
    m.def(
        "exact_scan_speculative_get(Tensor li_scores, Tensor threshold, "
        "Tensor resident, Tensor table_indices, Tensor offsets, "
        "Tensor output_indices, Tensor address_table, Tensor valid_table, "
        "Tensor(a!) packed_output, Tensor fused_get_control, Tensor frame_epoch, "
        "Tensor(b!) pipeline_state, Tensor execution_mode) "
        "-> (Tensor(a!) packed_output, Tensor candidate_indices, "
        "Tensor candidate_kind, Tensor producer_stats, Tensor get_stats, "
        "Tensor role_trace, Tensor get_indices, Tensor get_offsets, "
        "Tensor get_output_indices, Tensor get_mask, Tensor(b!) pipeline_state)");
}

TORCH_LIBRARY_IMPL(a5_exact_scan_get, PrivateUse1, m)
{
    m.impl("exact_scan_speculative_get", &ExactScanSpeculativeGetNpu);
}

TORCH_LIBRARY_IMPL(a5_exact_scan_get, Meta, m)
{
    m.impl("exact_scan_speculative_get", &ExactScanSpeculativeGetMeta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
