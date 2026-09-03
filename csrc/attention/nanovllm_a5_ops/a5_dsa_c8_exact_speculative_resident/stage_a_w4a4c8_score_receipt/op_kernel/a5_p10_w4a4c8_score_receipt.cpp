// P10 Stage A: native W4A4C8 score production plus graph-visible receipts.
//
// This is a full numerical entrypoint, not a marker kernel.  Its only change
// to the imported C8 data path is the GM destination rebind performed by
// QuantLiPhase: ProcessVec1 writes the original uint16 score values directly
// into score_surface and ProcessTopK reads those same bytes unchanged.

#include "kernel_operator.h"

#include "a5_p10_w4a4c8_score_receipt_qli.h"
#include "a5_p10_w4a4c8_score_receipt_tiling.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void a5_p10_w4a4c8_score_receipt(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights,
    GM_ADDR query_dequant_scale, GM_ADDR key_dequant_scale,
    GM_ADDR actual_seq_lengths_query, GM_ADDR candidate_lens,
    GM_ADDR block_table, GM_ADDR canonical_sparse_indices_buffer,
    GM_ADDR canonical_score_source_lens_buffer, GM_ADDR score_surface_buffer,
    GM_ADDR receipt_ring_buffer, GM_ADDR receipt_state_buffer,
    GM_ADDR forward_control_buffer,
    GM_ADDR canonical_sparse_indices_alias,
    GM_ADDR canonical_score_source_lens_alias, GM_ADDR score_surface_alias,
    GM_ADDR receipt_ring_alias, GM_ADDR receipt_state_alias,
    GM_ADDR forward_control_alias,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // These five outputs are graph-visible mutation aliases of the caller
    // supplied buffers.  Stage B receives the same buffers directly; it does
    // not depend on this kernel's output handles or its completion event.
    (void)actual_seq_lengths_query;
    (void)canonical_sparse_indices_alias;
    (void)canonical_score_source_lens_alias;
    (void)score_surface_alias;
    (void)receipt_ring_alias;
    (void)receipt_state_alias;
    (void)forward_control_alias;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(A5P10W4A4C8ScoreReceiptTilingData);
    GET_TILING_DATA(tiling_data, tiling);

    TPipe pipe;
    GM_ADDR user_workspace = GetUserWorkspace(workspace);
    a5_p10_w4a4c8_score_receipt_impl::QuantLiPhase phase(
        &pipe, &tiling_data);
    phase.Init(
        query, key, weights, query_dequant_scale, key_dequant_scale,
        candidate_lens, block_table, canonical_sparse_indices_buffer,
        canonical_score_source_lens_buffer, score_surface_buffer,
        receipt_ring_buffer, receipt_state_buffer, forward_control_buffer,
        user_workspace);
    phase.Process();
}
