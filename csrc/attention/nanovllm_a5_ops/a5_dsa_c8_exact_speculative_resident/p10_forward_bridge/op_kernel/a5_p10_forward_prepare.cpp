// P10 forward preparation node.
//
// This node is deliberately separate from both the native W4A4C8 score
// producer and the Direct-HCCP executor.  The graph uses its seven alias
// outputs as the common input provenance for both concurrent nodes.  There is
// consequently a prepare -> producer edge and a prepare -> executor edge, but
// no producer -> executor completion edge.

#include "kernel_operator.h"

#include "a5_p10_forward_prepare_abi.h"
#include "a5_p10_forward_prepare_tiling.h"

using namespace AscendC;

namespace {
template <int32_t kValue>
__aicore__ inline void FillShard(GlobalTensor<int32_t>& tensor,
                                 uint32_t element_count,
                                 uint32_t core_index,
                                 uint32_t core_count)
{
    for (uint32_t index = core_index; index < element_count;
         index += core_count) {
        tensor.SetValue(index, kValue);
    }
}
}  // namespace

extern "C" __global__ __aicore__ void a5_p10_forward_prepare(
    GM_ADDR receipt_state, GM_ADDR forward_control,
    GM_ADDR kv_object_to_forward_slot, GM_ADDR source_to_forward_slot,
    GM_ADDR speculative_slot_cursor, GM_ADDR queue_leases,
    GM_ADDR resident_ready, GM_ADDR receipt_state_alias,
    GM_ADDR forward_control_alias, GM_ADDR kv_object_to_forward_slot_alias,
    GM_ADDR source_to_forward_slot_alias,
    GM_ADDR speculative_slot_cursor_alias, GM_ADDR queue_leases_alias,
    GM_ADDR resident_ready_alias, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)receipt_state_alias;
    (void)forward_control_alias;
    (void)kv_object_to_forward_slot_alias;
    (void)source_to_forward_slot_alias;
    (void)speculative_slot_cursor_alias;
    (void)queue_leases_alias;
    (void)resident_ready_alias;
    (void)workspace;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5P10ForwardPrepareTilingData);
    GET_TILING_DATA(tiling_data, tiling);

    const uint32_t core_count = tiling_data.usedCoreNum;
    const uint32_t core_index = GetBlockIdx();
    if (core_count == 0U || core_index >= core_count) {
        return;
    }

    GlobalTensor<int32_t> receipt_state_gm;
    GlobalTensor<int32_t> forward_control_gm;
    GlobalTensor<int32_t> object_to_forward_slot_gm;
    GlobalTensor<int32_t> source_to_forward_slot_gm;
    GlobalTensor<int32_t> speculative_slot_cursor_gm;
    GlobalTensor<int32_t> queue_leases_gm;
    GlobalTensor<int32_t> resident_ready_gm;
    receipt_state_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(receipt_state),
        tiling_data.receiptStateElements);
    forward_control_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(forward_control),
        A5_P10_FORWARD_PREPARE_CONTROL_WORDS);
    object_to_forward_slot_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(kv_object_to_forward_slot),
        tiling_data.objectToForwardSlotElements);
    source_to_forward_slot_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(source_to_forward_slot),
        tiling_data.sourceToForwardSlotElements);
    speculative_slot_cursor_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(speculative_slot_cursor), 1U);
    queue_leases_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(queue_leases),
        tiling_data.queueLeaseElements);
    resident_ready_gm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(resident_ready),
        tiling_data.residentReadyElements);

    FillShard<A5_P10_FORWARD_PREPARE_RECEIPT_FREE>(
        receipt_state_gm, tiling_data.receiptStateElements, core_index,
        core_count);
    FillShard<A5_P10_FORWARD_PREPARE_UNMAPPED_SLOT>(
        object_to_forward_slot_gm, tiling_data.objectToForwardSlotElements,
        core_index, core_count);
    FillShard<A5_P10_FORWARD_PREPARE_UNMAPPED_SLOT>(
        source_to_forward_slot_gm, tiling_data.sourceToForwardSlotElements,
        core_index, core_count);
    FillShard<0>(queue_leases_gm, tiling_data.queueLeaseElements, core_index,
                 core_count);
    FillShard<0>(resident_ready_gm, tiling_data.residentReadyElements,
                 core_index, core_count);

    // Mutable control fields are initialized exactly once.  Tiling-owned
    // geometry/cutoff words [1..4,6] are pre-materialized by the graph and
    // intentionally survive this reset.
    if (core_index == 0U) {
        speculative_slot_cursor_gm.SetValue(0U, 0);
        forward_control_gm.SetValue(
            0U, A5_P10_FORWARD_PREPARE_PRODUCER_OPEN);
        // P10 Stage B uses word 1 only as a forward-local diagnostic ledger.
        // It is never a producer scheduling input or a receipt count.
        forward_control_gm.SetValue(1U, 0);
        forward_control_gm.SetValue(
            5U, A5_P10_FORWARD_PREPARE_FORWARD_INCOMPLETE);
        forward_control_gm.SetValue(7U, 0);
        forward_control_gm.SetValue(
            A5_P10_FORWARD_PREPARE_PRODUCER_SEAL_SEQUENCE_WORD, 0);
        forward_control_gm.SetValue(
            A5_P10_FORWARD_PREPARE_FIRST_PRESEAL_READY_CLAIM_WORD, 0);
    }
}
