// P8 typed bridge: Stage-B CandidateSeed -> forward-owned Direct-HCCP state.
//
// Stage-A score-receipt scanning is deliberately a separately built,
// source-closed operator.  This bridge neither imports C8 nor P7.  It submits
// one graph-visible P8 phase at a time; the durable GM forward/wave/private
// records, rather than a host callback or Python lease, own Post/CQ/QSFA
// lifecycle state.

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <tuple>

#include <torch/extension.h>
#include <torch/library.h>

// Keep P8's custom OPP resolver disjoint from a Stage-A score-receipt OPP or
// any production C8 package loaded by the same Python process.
#define NANOVLLM_A5_OPAPI_ENV "NANOVLLM_A5_P8_FORWARD_OWNED_OPAPI"
#include "op_api_common.h"

namespace {

constexpr int64_t kInputCount = 27;
constexpr int64_t kOutputCount = 16;
constexpr int64_t kCandidateCapacity = 128;
constexpr int64_t kCandidateSeedWords = 4;
constexpr int64_t kForwardControlWords = 4;
constexpr int64_t kStreamingControlWords = 16;
constexpr int64_t kWaveRecordWords = 32;
constexpr int64_t kPrivateRecordWords = 10;
constexpr int64_t kRequestCapacity = 64;
constexpr int64_t kPackedKvRowBytes = 656;
constexpr int64_t kWorkerStatColumns = 16;
constexpr int64_t kAdvanceIdentityWords = 3;
constexpr int64_t kAdvanceProgramWords = 4;
constexpr int64_t kAdvanceResultWords = 4;

struct Geometry {
    int64_t receipts;
    int64_t rows;
    int64_t sources;
    int64_t pools;
    int64_t objects;
    int64_t queues;
    int64_t private_rows;
    int64_t workers;
};

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
    const at::Tensor& candidate_seeds, const at::Tensor& candidate_counts,
    const at::Tensor& candidate_ready_epoch,
    const at::Tensor& candidate_status, const at::Tensor& forward_control,
    const at::Tensor& source_lens, const at::Tensor& kv_object_address_domain,
    const at::Tensor& req_pool_entries, const at::Tensor& kv_object_map,
    const at::Tensor& kv_object_lens,
    const at::Tensor& kv_object_slots_snapshot,
    const at::Tensor& private_source_to_slot,
    const at::Tensor& private_object_to_slot,
    const at::Tensor& private_ready_table,
    const at::Tensor& private_slot_cursor,
    const at::Tensor& private_packed_output,
    const at::Tensor& fused_get_control, const at::Tensor& p8_wave_records,
    const at::Tensor& p8_private_records,
    const at::Tensor& queue_request_source_offsets,
    const at::Tensor& queue_request_object_offsets,
    const at::Tensor& queue_request_output_indices,
    const at::Tensor& queue_request_count,
    const at::Tensor& candidate_cursor,
    const at::Tensor& candidate_cursor_epoch,
    const at::Tensor& rows_per_entry, const at::Tensor& worker_stats)
{
    TORCH_CHECK(candidate_seeds.dim() == 3 && candidate_seeds.size(0) > 0 &&
                    candidate_seeds.size(1) == kCandidateCapacity &&
                    candidate_seeds.size(2) == kCandidateSeedWords,
                "candidate_seeds must be int32[N, 128, 4], N > 0.");
    const int64_t receipts = candidate_seeds.size(0);
    CheckVector(candidate_counts, receipts, "candidate_counts");
    CheckVector(candidate_ready_epoch, receipts, "candidate_ready_epoch");
    CheckVector(candidate_status, receipts, "candidate_status");
    CheckVector(forward_control, kForwardControlWords, "forward_control");
    TORCH_CHECK(source_lens.dim() == 1 && source_lens.numel() > 0,
                "source_lens must be int32[R], R > 0.");
    const int64_t rows = source_lens.numel();
    CheckVector(kv_object_address_domain, 1, "kv_object_address_domain");
    CheckVector(req_pool_entries, rows, "req_pool_entries");
    CheckVector(kv_object_lens, rows, "kv_object_lens");
    TORCH_CHECK(kv_object_map.dim() == 2 && kv_object_map.size(0) > 0 &&
                    kv_object_map.size(1) > 0,
                "kv_object_map must be int32[P, S], P,S > 0.");
    const int64_t pools = kv_object_map.size(0);
    const int64_t sources = kv_object_map.size(1);
    TORCH_CHECK(kv_object_slots_snapshot.dim() == 2 &&
                    kv_object_slots_snapshot.size(0) == pools &&
                    kv_object_slots_snapshot.size(1) > 0,
                "kv_object_slots_snapshot must be int32[P, O], P aligned to map.");
    const int64_t objects = kv_object_slots_snapshot.size(1);
    CheckMatrix(private_source_to_slot, rows, sources,
                "private_source_to_slot");
    CheckMatrix(private_object_to_slot, pools, objects,
                "private_object_to_slot");
    TORCH_CHECK(private_ready_table.dim() == 1 &&
                    private_ready_table.numel() > 0,
                "private_ready_table must be int32[A], A > 0.");
    const int64_t private_rows = private_ready_table.numel();
    CheckVector(private_slot_cursor, 1, "private_slot_cursor");
    CheckMatrix(private_packed_output, private_rows, kPackedKvRowBytes,
                "private_packed_output");
    CheckVector(fused_get_control, kStreamingControlWords,
                "fused_get_control");
    TORCH_CHECK(p8_wave_records.dim() == 2 && p8_wave_records.size(0) > 0 &&
                    p8_wave_records.size(1) == kWaveRecordWords,
                "p8_wave_records must be int64[Q, 32], Q > 0.");
    const int64_t queues = p8_wave_records.size(0);
    CheckMatrix(p8_private_records, private_rows, kPrivateRecordWords,
                "p8_private_records");
    CheckMatrix(queue_request_source_offsets, queues, kRequestCapacity,
                "queue_request_source_offsets");
    CheckMatrix(queue_request_object_offsets, queues, kRequestCapacity,
                "queue_request_object_offsets");
    CheckMatrix(queue_request_output_indices, queues, kRequestCapacity,
                "queue_request_output_indices");
    CheckVector(queue_request_count, queues, "queue_request_count");
    CheckVector(candidate_cursor, receipts, "candidate_cursor");
    CheckVector(candidate_cursor_epoch, receipts, "candidate_cursor_epoch");
    CheckVector(rows_per_entry, 1, "rows_per_entry");
    TORCH_CHECK(worker_stats.dim() == 2 && worker_stats.size(0) > 0 &&
                    worker_stats.size(1) == kWorkerStatColumns,
                "worker_stats must be int64[W, 16], W > 0.");
    return {receipts, rows, sources, pools, objects, queues, private_rows,
            worker_stats.size(0)};
}

void CheckNpuInputs(
    const at::Tensor& anchor,
    const std::array<const at::Tensor*, kInputCount>& inputs)
{
    TORCH_CHECK(anchor.device().is_privateuseone(),
                "p8_forward_owned_executor requires NPU inputs.");
    const at::Device device = anchor.device();
    const std::array<at::ScalarType, kInputCount> dtypes = {
        at::kInt,  at::kInt,  at::kLong, at::kInt,  at::kLong,
        at::kInt,  at::kInt,  at::kInt,  at::kInt,  at::kInt,
        at::kInt,  at::kInt,  at::kInt,  at::kInt,  at::kInt,
        at::kByte, at::kLong, at::kLong, at::kLong, at::kLong,
        at::kLong, at::kInt,  at::kInt,  at::kInt,  at::kLong,
        at::kInt,  at::kLong,
    };
    for (size_t index = 0; index < inputs.size(); ++index) {
        const at::Tensor& tensor = *inputs[index];
        TORCH_CHECK(tensor.device() == device && tensor.is_contiguous() &&
                        tensor.scalar_type() == dtypes[index],
                    "P8 input ", index,
                    " must be contiguous, have its declared ABI dtype, and be on ",
                    device.str(), ".");
    }
}

using Result = std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor>;

using AdvanceResult = std::tuple<at::Tensor, at::Tensor>;

void ValidateAdvanceInputs(const at::Tensor& forward_control,
                           const at::Tensor& expected_identity,
                           const at::Tensor& transition_program,
                           const at::Tensor& transition_result,
                           bool require_npu)
{
    CheckVector(forward_control, kForwardControlWords, "forward_control");
    CheckVector(expected_identity, kAdvanceIdentityWords,
                "expected_identity");
    CheckVector(transition_program, kAdvanceProgramWords,
                "transition_program");
    CheckVector(transition_result, kAdvanceResultWords,
                "transition_result");
    const std::array<const at::Tensor*, 4> tensors = {
        &forward_control, &expected_identity, &transition_program,
        &transition_result,
    };
    const at::Device device = forward_control.device();
    if (require_npu) {
        TORCH_CHECK(device.is_privateuseone(),
                    "p8 forward-control advance requires NPU tensors.");
    }
    for (size_t index = 0; index < tensors.size(); ++index) {
        const at::Tensor& tensor = *tensors[index];
        TORCH_CHECK(tensor.device() == device && tensor.is_contiguous() &&
                        tensor.scalar_type() == at::kLong,
                    "P8 forward-control advance input ", index,
                    " must be contiguous int64 on ", device.str(), ".");
    }
}

AdvanceResult AdvanceForwardControlNpu(
    at::Tensor forward_control, const at::Tensor& expected_identity,
    const at::Tensor& transition_program, at::Tensor transition_result)
{
    ValidateAdvanceInputs(forward_control, expected_identity,
                          transition_program, transition_result, true);
    auto keepalive = std::make_tuple(forward_control, expected_identity,
                                     transition_program, transition_result);
    EXEC_NPU_CMD_ORDERED(
        aclnnA5P8ForwardControlAdvance, keepalive, forward_control,
        expected_identity, transition_program, transition_result);
    return std::make_tuple(forward_control, transition_result);
}

AdvanceResult AdvanceForwardControlMeta(
    at::Tensor forward_control, const at::Tensor& expected_identity,
    const at::Tensor& transition_program, at::Tensor transition_result)
{
    ValidateAdvanceInputs(forward_control, expected_identity,
                          transition_program, transition_result, false);
    return std::make_tuple(forward_control, transition_result);
}

Result RunNpu(
    const at::Tensor& candidate_seeds, const at::Tensor& candidate_counts,
    const at::Tensor& candidate_ready_epoch,
    const at::Tensor& candidate_status, at::Tensor forward_control,
    const at::Tensor& source_lens, const at::Tensor& kv_object_address_domain,
    const at::Tensor& req_pool_entries, const at::Tensor& kv_object_map,
    const at::Tensor& kv_object_lens,
    const at::Tensor& kv_object_slots_snapshot,
    at::Tensor private_source_to_slot, at::Tensor private_object_to_slot,
    at::Tensor private_ready_table, at::Tensor private_slot_cursor,
    at::Tensor private_packed_output, at::Tensor fused_get_control,
    at::Tensor p8_wave_records, at::Tensor p8_private_records,
    at::Tensor queue_request_source_offsets,
    at::Tensor queue_request_object_offsets,
    at::Tensor queue_request_output_indices,
    at::Tensor queue_request_count, at::Tensor candidate_cursor,
    at::Tensor candidate_cursor_epoch, const at::Tensor& rows_per_entry,
    at::Tensor worker_stats)
{
    const std::array<const at::Tensor*, kInputCount> inputs = {
        &candidate_seeds, &candidate_counts, &candidate_ready_epoch,
        &candidate_status, &forward_control, &source_lens,
        &kv_object_address_domain, &req_pool_entries, &kv_object_map,
        &kv_object_lens, &kv_object_slots_snapshot, &private_source_to_slot,
        &private_object_to_slot, &private_ready_table, &private_slot_cursor,
        &private_packed_output, &fused_get_control, &p8_wave_records,
        &p8_private_records, &queue_request_source_offsets,
        &queue_request_object_offsets, &queue_request_output_indices,
        &queue_request_count, &candidate_cursor, &candidate_cursor_epoch,
        &rows_per_entry, &worker_stats,
    };
    const Geometry geometry = ValidateShapes(
        candidate_seeds, candidate_counts, candidate_ready_epoch,
        candidate_status, forward_control, source_lens,
        kv_object_address_domain, req_pool_entries, kv_object_map,
        kv_object_lens, kv_object_slots_snapshot, private_source_to_slot,
        private_object_to_slot, private_ready_table, private_slot_cursor,
        private_packed_output, fused_get_control, p8_wave_records,
        p8_private_records, queue_request_source_offsets,
        queue_request_object_offsets, queue_request_output_indices,
        queue_request_count, candidate_cursor, candidate_cursor_epoch,
        rows_per_entry, worker_stats);
    CheckNpuInputs(candidate_seeds, inputs);

    if (const char* debug = std::getenv("NANOVLLM_A5_OPAPI_DEBUG");
        debug != nullptr && debug[0] != '\0' &&
        !(debug[0] == '0' && debug[1] == '\0')) {
        std::fprintf(
            stderr,
            "P8_FORWARD_OWNED_BRIDGE_ABI tensors=27 mutable_refs=16 "
            "receipts=%lld queues=%lld private_rows=%lld phase_control=gm "
            "c8_dependency=0 fixed_get_core=0\n",
            static_cast<long long>(geometry.receipts),
            static_cast<long long>(geometry.queues),
            static_cast<long long>(geometry.private_rows));
        std::fflush(stderr);
    }

    // OpCommand retains this value-owned tuple until the framework-ordered
    // ACLNN launch completes.  Device Post/CQ/private lifetime is *not* a
    // host closure: it is the P8 GM state machine carried by these tensors.
    auto keepalive = std::make_tuple(
        candidate_seeds, candidate_counts, candidate_ready_epoch,
        candidate_status, forward_control, source_lens,
        kv_object_address_domain, req_pool_entries, kv_object_map,
        kv_object_lens, kv_object_slots_snapshot, private_source_to_slot,
        private_object_to_slot, private_ready_table, private_slot_cursor,
        private_packed_output, fused_get_control, p8_wave_records,
        p8_private_records, queue_request_source_offsets,
        queue_request_object_offsets, queue_request_output_indices,
        queue_request_count, candidate_cursor, candidate_cursor_epoch,
        rows_per_entry, worker_stats);
    EXEC_NPU_CMD_ORDERED(
        aclnnA5P8ForwardOwnedExecutor, keepalive, candidate_seeds,
        candidate_counts, candidate_ready_epoch, candidate_status,
        forward_control, source_lens, kv_object_address_domain,
        req_pool_entries, kv_object_map, kv_object_lens,
        kv_object_slots_snapshot, private_source_to_slot,
        private_object_to_slot, private_ready_table, private_slot_cursor,
        private_packed_output, fused_get_control, p8_wave_records,
        p8_private_records, queue_request_source_offsets,
        queue_request_object_offsets, queue_request_output_indices,
        queue_request_count, candidate_cursor, candidate_cursor_epoch,
        rows_per_entry, worker_stats);
    return std::make_tuple(
        forward_control, private_source_to_slot, private_object_to_slot,
        private_ready_table,
        private_slot_cursor, private_packed_output, fused_get_control,
        p8_wave_records, p8_private_records, queue_request_source_offsets,
        queue_request_object_offsets, queue_request_output_indices,
        queue_request_count, candidate_cursor, candidate_cursor_epoch,
        worker_stats);
}

Result RunMeta(
    const at::Tensor& candidate_seeds, const at::Tensor& candidate_counts,
    const at::Tensor& candidate_ready_epoch,
    const at::Tensor& candidate_status, at::Tensor forward_control,
    const at::Tensor& source_lens, const at::Tensor& kv_object_address_domain,
    const at::Tensor& req_pool_entries, const at::Tensor& kv_object_map,
    const at::Tensor& kv_object_lens,
    const at::Tensor& kv_object_slots_snapshot,
    at::Tensor private_source_to_slot, at::Tensor private_object_to_slot,
    at::Tensor private_ready_table, at::Tensor private_slot_cursor,
    at::Tensor private_packed_output, at::Tensor fused_get_control,
    at::Tensor p8_wave_records, at::Tensor p8_private_records,
    at::Tensor queue_request_source_offsets,
    at::Tensor queue_request_object_offsets,
    at::Tensor queue_request_output_indices,
    at::Tensor queue_request_count, at::Tensor candidate_cursor,
    at::Tensor candidate_cursor_epoch, const at::Tensor& rows_per_entry,
    at::Tensor worker_stats)
{
    (void)ValidateShapes(
        candidate_seeds, candidate_counts, candidate_ready_epoch,
        candidate_status, forward_control, source_lens,
        kv_object_address_domain, req_pool_entries, kv_object_map,
        kv_object_lens, kv_object_slots_snapshot, private_source_to_slot,
        private_object_to_slot, private_ready_table, private_slot_cursor,
        private_packed_output, fused_get_control, p8_wave_records,
        p8_private_records, queue_request_source_offsets,
        queue_request_object_offsets, queue_request_output_indices,
        queue_request_count, candidate_cursor, candidate_cursor_epoch,
        rows_per_entry, worker_stats);
    return std::make_tuple(
        forward_control, private_source_to_slot, private_object_to_slot,
        private_ready_table,
        private_slot_cursor, private_packed_output, fused_get_control,
        p8_wave_records, p8_private_records, queue_request_source_offsets,
        queue_request_object_offsets, queue_request_output_indices,
        queue_request_count, candidate_cursor, candidate_cursor_epoch,
        worker_stats);
}

}  // namespace

TORCH_LIBRARY(a5_p8_forward_owned_executor, m)
{
    m.def(
        "forward_owned_executor("
        "Tensor candidate_seeds, Tensor candidate_counts, "
        "Tensor candidate_ready_epoch, Tensor candidate_status, "
        "Tensor(a!) forward_control, Tensor source_lens, "
        "Tensor kv_object_address_domain, Tensor req_pool_entries, "
        "Tensor kv_object_map, Tensor kv_object_lens, "
        "Tensor kv_object_slots_snapshot, "
        "Tensor(b!) private_source_to_slot, Tensor(c!) private_object_to_slot, "
        "Tensor(d!) private_ready_table, Tensor(e!) private_slot_cursor, "
        "Tensor(f!) private_packed_output, Tensor(g!) fused_get_control, "
        "Tensor(h!) p8_wave_records, Tensor(i!) p8_private_records, "
        "Tensor(j!) queue_request_source_offsets, "
        "Tensor(k!) queue_request_object_offsets, "
        "Tensor(l!) queue_request_output_indices, "
        "Tensor(m!) queue_request_count, Tensor(n!) candidate_cursor, "
        "Tensor(o!) candidate_cursor_epoch, Tensor rows_per_entry, "
        "Tensor(p!) worker_stats) -> ("
        "Tensor(a!) forward_control, "
        "Tensor(b!) private_source_to_slot, Tensor(c!) private_object_to_slot, "
        "Tensor(d!) private_ready_table, Tensor(e!) private_slot_cursor, "
        "Tensor(f!) private_packed_output, Tensor(g!) fused_get_control, "
        "Tensor(h!) p8_wave_records, Tensor(i!) p8_private_records, "
        "Tensor(j!) queue_request_source_offsets, "
        "Tensor(k!) queue_request_object_offsets, "
        "Tensor(l!) queue_request_output_indices, "
        "Tensor(m!) queue_request_count, Tensor(n!) candidate_cursor, "
        "Tensor(o!) candidate_cursor_epoch, Tensor(p!) worker_stats)");
    m.def(
        "advance_forward_control("
        "Tensor(a!) forward_control, Tensor expected_identity, "
        "Tensor transition_program, Tensor(b!) transition_result) -> ("
        "Tensor(a!) forward_control, Tensor(b!) transition_result)");
}

TORCH_LIBRARY_IMPL(a5_p8_forward_owned_executor, PrivateUse1, m)
{
    m.impl("forward_owned_executor", &RunNpu);
    m.impl("advance_forward_control", &AdvanceForwardControlNpu);
}

TORCH_LIBRARY_IMPL(a5_p8_forward_owned_executor, Meta, m)
{
    m.impl("forward_owned_executor", &RunMeta);
    m.impl("advance_forward_control", &AdvanceForwardControlMeta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("p8_executor_contract", [] {
        pybind11::dict contract;
        contract["stage"] = 8;
        contract["tensor_args"] = kInputCount;
        contract["mutable_refs"] = kOutputCount;
        contract["candidate_seed_words"] = kCandidateSeedWords;
        contract["candidate_capacity"] = kCandidateCapacity;
        contract["request_capacity"] = kRequestCapacity;
        contract["phases"] = "post,drain,qsfa_terminal,forward_terminal";
        contract["graph_phase_advance"] = true;
        contract["advance_input_tensors"] = 4;
        contract["advance_mutable_refs"] = 2;
        contract["dynamic_queue_lease"] = true;
        contract["fixed_get_core"] = false;
        contract["c8_dependency"] = false;
        return contract;
    });
}
