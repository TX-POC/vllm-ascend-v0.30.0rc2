// Dynamic Direct-HCCP executor for the P9 source-derived C8 operator.
//
// This is intentionally an invocation-local transport helper.  It has no
// private arena, no cross-forward directory, no permanent AIV role, and no
// resident-slot release operation.  A C8 worker compacts exact score seeds,
// wins a temporary queue lease if one is idle, posts one bounded wave into
// the row's caller-supplied direct-resident range, then continues scoring.
// The queue is returned only after its CQ is terminal; the resident payload
// remains at its fixed output row until QSFA consumes `attention_slots`.

#ifndef VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_POST_EXECUTOR_H_
#define VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_POST_EXECUTOR_H_

#include <climits>
#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/cpp/kernel_simt_warp_level_intf.h"

#include "a5_c8_exact_speculative_resident_contract.h"
#include "a5_c8_exact_speculative_resident_score_hook.h"
#include "kvaf_layer_resident/kvaf_layer_resident_get_device.h"

namespace QLIKernel {
namespace exact_speculative_resident {

using namespace AscendC;
using kvaf_layer_resident_get::AppendKvafLayerResidentRequest;
using kvaf_layer_resident_get::BeginKvafLayerResidentBatchGet;
using kvaf_layer_resident_get::FinishKvafLayerResidentBatchGet;
using kvaf_layer_resident_get::KvafLayerResidentBatchGetControl;
using kvaf_layer_resident_get::KvafLayerResidentBatchGetTerminal;
using kvaf_layer_resident_get::KvafLayerResidentBatchGetWave;
using kvaf_layer_resident_get::KvafLayerResidentQueueLease;
using kvaf_layer_resident_get::KvafLayerResidentQueueLeaseResult;
using kvaf_layer_resident_get::LoadKvafLayerResidentBatchGetControlFromGmForAicore;
using kvaf_layer_resident_get::LoadKvafLayerResidentQueueTemplate;
using kvaf_layer_resident_get::PublishKvafLayerResidentDescriptor;
using kvaf_layer_resident_get::ReturnKvafLayerResidentQueueFromAicore;
using kvaf_layer_resident_get::ReturnKvafLayerResidentUnpostedQueueFromAicore;
using kvaf_layer_resident_get::TryAcquireKvafLayerResidentQueueFromAicore;
using kvaf_layer_resident_get::TryRetireKvafLayerResidentBatchGet;
using vllm_ascend::a5_c8_exact_speculative_resident::IsSlotInCanonicalRow;
using vllm_ascend::a5_c8_exact_speculative_resident::kExactRecallWaveCapacity;

constexpr int32_t kCandidateSourceClaiming = -2;

static_assert(kExactRecallWaveCapacity ==
                  KVAF_AIV_BATCH_GET_DENSE_REQUEST_CAPACITY,
              "P9 Stage-B recall must match KVAF's dense descriptor capacity.");

// The SIMT reservation result is a tiny caller-supplied scalar hand-off.  It
// lives in pre-existing C8 scratch, next to the two CandidateSeed batches;
// it is not a TBuf, a ring in GM, or a cross-forward record.
struct CandidateSlotReservationResult {
    uint64_t sourceOffset;
    int32_t residentSlot;
    uint32_t success;
};
static_assert(sizeof(CandidateSlotReservationResult) == 16U,
              "P9 candidate reservation hand-off ABI changed");

// Reserve one canonical destination before descriptor assembly.  `-2` is a
// short pre-doorbell claim only: it becomes an absolute resident row after
// Begin accepted, or returns to -1 if this wave never reaches Begin.  The
// per-row cursor is monotonic for this invocation.  A failed reservation can
// burn a cursor value but can never point two requests at one output row;
// exact recall covers the resulting speculative miss.
__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void ReserveCandidateResidentSlotSimt(
    __gm__ int32_t* candidate_source_to_slot,
    __gm__ int32_t* candidate_slot_cursor,
    uint32_t row, uint32_t source, uint32_t source_capacity,
    int32_t first_slot, int32_t row_stride, uint32_t row_count,
    __ubuf__ CandidateSlotReservationResult* result)
{
    if (candidate_source_to_slot == nullptr || candidate_slot_cursor == nullptr ||
        result == nullptr || source_capacity == 0U || row >= row_count ||
        source >= source_capacity || first_slot < 0 ||
        row_stride < static_cast<int32_t>(kCanonicalResidentSlotCount)) {
        return;
    }

    const uint64_t source_offset =
        static_cast<uint64_t>(row) * source_capacity + source;
    result->sourceOffset = source_offset;
    result->residentSlot = kCandidateResidentSlotUnassigned;
    result->success = 0U;

    if (asc_atomic_cas(candidate_source_to_slot + source_offset,
                       kCandidateSourceUnmapped,
                       kCandidateSourceClaiming) != kCandidateSourceUnmapped) {
        return;
    }

    const uint32_t local_slot = asc_atomic_add(candidate_slot_cursor + row, 1U);
    if (local_slot >= kCanonicalResidentSlotCount) {
        (void)asc_atomic_cas(candidate_source_to_slot + source_offset,
                             kCandidateSourceClaiming,
                             kCandidateSourceUnmapped);
        return;
    }

    const int64_t absolute_slot = static_cast<int64_t>(first_slot) +
                                  static_cast<int64_t>(row) * row_stride +
                                  static_cast<int64_t>(local_slot);
    if (absolute_slot < 0 || absolute_slot > INT32_MAX ||
        local_slot >= kCanonicalResidentSlotCount) {
        (void)asc_atomic_cas(candidate_source_to_slot + source_offset,
                             kCandidateSourceClaiming,
                             kCandidateSourceUnmapped);
        return;
    }

    result->residentSlot = static_cast<int32_t>(absolute_slot);
    result->success = 1U;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void CommitCandidateResidentSlotSimt(
    __gm__ int32_t* candidate_source_to_slot,
    uint64_t source_offset, int32_t resident_slot,
    __ubuf__ CandidateSlotReservationResult* result)
{
    if (candidate_source_to_slot == nullptr || result == nullptr ||
        resident_slot < 0) {
        return;
    }
    result->sourceOffset = source_offset;
    result->residentSlot = resident_slot;
    result->success = 0U;

    // This is a post-acceptance receipt, not a payload-ready bit. Stage B
    // drains all initial waves before it can expose this fixed row to QSFA.
    // No inverse slot->source table is needed: Stage B derives a local
    // 2048-bit used-slot bitmap from the exact Top-K and may then reuse every
    // nonwinner row from the fixed canonical interval.
    AscendC::Simt::ThreadFence();
    if (asc_atomic_cas(candidate_source_to_slot + source_offset,
                       kCandidateSourceClaiming,
                       resident_slot) != kCandidateSourceClaiming) {
        return;
    }
    AscendC::Simt::ThreadFence();
    result->success = 1U;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void RollbackCandidateResidentSlotSimt(
    __gm__ int32_t* candidate_source_to_slot, uint64_t source_offset,
    __ubuf__ CandidateSlotReservationResult* result)
{
    if (candidate_source_to_slot == nullptr || result == nullptr) {
        return;
    }
    result->sourceOffset = source_offset;
    result->residentSlot = kCandidateResidentSlotUnassigned;
    result->success =
        asc_atomic_cas(candidate_source_to_slot + source_offset,
                       kCandidateSourceClaiming,
                       kCandidateSourceUnmapped) == kCandidateSourceClaiming
            ? 1U
            : 0U;
}

__aicore__ inline bool ReserveCandidateResidentSlotFromAicore(
    __gm__ int32_t* candidate_source_to_slot,
    __gm__ int32_t* candidate_slot_cursor,
    uint32_t row, uint32_t source, uint32_t source_capacity,
    CanonicalResidentLayout resident_layout,
    __ubuf__ CandidateSlotReservationResult* result, event_t simt_event)
{
    if (result == nullptr) {
        return false;
    }
    result->success = 0U;
    set_flag(PIPE_S, PIPE_V, simt_event);
    wait_flag(PIPE_S, PIPE_V, simt_event);
    AscendC::Simt::VF_CALL<ReserveCandidateResidentSlotSimt>(
        AscendC::Simt::Dim3{1U, 1U, 1U}, candidate_source_to_slot,
        candidate_slot_cursor, row, source, source_capacity,
        resident_layout.firstSlot, resident_layout.rowStride,
        resident_layout.rowCount, result);
    set_flag(PIPE_V, PIPE_S, simt_event);
    wait_flag(PIPE_V, PIPE_S, simt_event);
    return result->success != 0U;
}

__aicore__ inline bool CommitCandidateResidentSlotFromAicore(
    __gm__ int32_t* candidate_source_to_slot,
    uint64_t source_offset, int32_t resident_slot,
    __ubuf__ CandidateSlotReservationResult* result, event_t simt_event)
{
    if (result == nullptr) {
        return false;
    }
    result->success = 0U;
    set_flag(PIPE_S, PIPE_V, simt_event);
    wait_flag(PIPE_S, PIPE_V, simt_event);
    AscendC::Simt::VF_CALL<CommitCandidateResidentSlotSimt>(
        AscendC::Simt::Dim3{1U, 1U, 1U}, candidate_source_to_slot,
        source_offset, resident_slot, result);
    set_flag(PIPE_V, PIPE_S, simt_event);
    wait_flag(PIPE_V, PIPE_S, simt_event);
    return result->success != 0U;
}

__aicore__ inline void RollbackCandidateResidentSlotFromAicore(
    __gm__ int32_t* candidate_source_to_slot, uint64_t source_offset,
    __ubuf__ CandidateSlotReservationResult* result, event_t simt_event)
{
    if (result == nullptr) {
        return;
    }
    result->success = 0U;
    set_flag(PIPE_S, PIPE_V, simt_event);
    wait_flag(PIPE_S, PIPE_V, simt_event);
    AscendC::Simt::VF_CALL<RollbackCandidateResidentSlotSimt>(
        AscendC::Simt::Dim3{1U, 1U, 1U}, candidate_source_to_slot,
        source_offset, result);
    set_flag(PIPE_V, PIPE_S, simt_event);
    wait_flag(PIPE_V, PIPE_S, simt_event);
}

// One LI worker owns at most one active queue wave.  That is not a fixed Get
// core: any C8 worker can acquire any currently idle queue.  The one-wave
// rule is inherited from the existing KVAF Direct-HCCP queue ABI, whose
// descriptor/SQ/CQ record does not support two in-flight waves on one queue.
class C8ExactSpeculativeResidentPostExecutor {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t* layer_resident_get_control, uint32_t producer_id,
        __ubuf__ KvafLayerResidentQueueLeaseResult* lease_result,
        __ubuf__ CandidateSlotReservationResult* reservation_result,
        event_t simt_event)
    {
        initialized_ = false;
        healthy_ = true;
        has_wave_ = false;
        producer_id_ = producer_id;
        queue_hint_ = producer_id;
        control_gm_ = reinterpret_cast<__gm__ const KvafLayerResidentBatchGetControl*>(
            layer_resident_get_control);
        lease_result_ = lease_result;
        reservation_result_ = reservation_result;
        simt_event_ = simt_event;
        if (control_gm_ == nullptr || lease_result_ == nullptr ||
            reservation_result_ == nullptr ||
            !LoadKvafLayerResidentBatchGetControlFromGmForAicore(
                control_gm_, &control_) ||
            !kvaf_layer_resident_get::
                IsKvafLayerResidentBatchGetControlValidForAicore(control_)) {
            healthy_ = false;
            return;
        }
        initialized_ = true;
    }

    __aicore__ inline bool Enabled() const
    {
        return initialized_ && healthy_;
    }

    __aicore__ inline bool HasInFlightWave() const
    {
        return has_wave_ &&
               wave_.lease.state ==
                   kvaf_layer_resident_get::kLayerResidentQueuePosted;
    }

    // Nonblocking progress point.  The source-derived C8 seam calls this at
    // later score tiles; it never spins waiting for a CQ while score work
    // remains.  A terminal wave is finalized and its temporary queue lease
    // is returned, while its fixed resident payload stays intact.
    __aicore__ inline void TryRetireAtScoreBoundary()
    {
        // Once Begin has accepted the descriptor, terminal progress remains
        // mandatory even if a later bookkeeping check has made the executor
        // unhealthy.  `healthy_` controls whether we may issue *another*
        // wave; it must never suppress the only drain that can make the
        // already-posted fixed resident rows safe to expose or the temporary
        // queue safe to return.
        if (!initialized_ || !has_wave_) {
            return;
        }
        TryRetireKvafLayerResidentBatchGet(&wave_);
        if (wave_.lease.state ==
            kvaf_layer_resident_get::kLayerResidentQueueTerminal) {
            (void)FinalizeAndReturnTerminalWave();
        }
    }

    // Try to post one ready score microbatch.  A busy queue is ordinary
    // backpressure: the hook drops only that speculative opportunity and
    // Stage-B Guaranteed Recall remains exact.  Any protocol inconsistency
    // makes this executor unhealthy; it must not return a successful fused
    // operator result with an untracked resident write.
    __aicore__ inline bool TryPostReady(C8ExactSpeculativeResidentScoreHook* hook)
    {
        if (!Enabled() || hook == nullptr) {
            return false;
        }
        TryRetireAtScoreBoundary();
        if (!healthy_ || has_wave_) {
            return false;
        }

        CandidatePostView view{};
        uint32_t batch_token = 0U;
        if (!hook->TakeReadyPostView(&view, &batch_token)) {
            return false;
        }
        if (!ValidateView(view)) {
            hook->DeferPostView(batch_token);
            healthy_ = false;
            return false;
        }

        KvafLayerResidentQueueLease lease{};
        if (!TryAcquireKvafLayerResidentQueueFromAicore(
                control_gm_, producer_id_, queue_hint_, lease_result_, simt_event_,
                &lease)) {
            hook->DeferPostView(batch_token);
            return false;
        }
        queue_hint_ = lease.queue_id + 1U;

        KvafLayerResidentBatchGetDescriptorScratch descriptor{};
        if (!LoadKvafLayerResidentQueueTemplate(control_, lease, &descriptor)) {
            if (!ReturnKvafLayerResidentUnpostedQueueFromAicore(
                    control_gm_, lease_result_, simt_event_, &lease)) {
                healthy_ = false;
            }
            hook->DeferPostView(batch_token);
            return false;
        }

        uint32_t accepted_count = 0U;
        for (uint32_t seed_index = 0U; seed_index < view.count; ++seed_index) {
            CandidateSeed& seed = view.seeds[seed_index];
            if (!ReserveSeed(view, &seed) ||
                !AppendKvafLayerResidentRequest(
                    control_, &descriptor, static_cast<uint32_t>(seed.kvObject),
                    view.rowsPerEntry, seed.residentSlot)) {
                if (seed.residentSlot >= 0) {
                    RollbackCandidateResidentSlotFromAicore(
                        view.candidateSourceToSlot,
                        reservation_result_->sourceOffset, reservation_result_,
                        simt_event_);
                    seed.residentSlot = kCandidateResidentSlotUnassigned;
                }
                continue;
            }
            ++accepted_count;
        }

        if (accepted_count == 0U) {
            if (!ReturnKvafLayerResidentUnpostedQueueFromAicore(
                    control_gm_, lease_result_, simt_event_, &lease)) {
                healthy_ = false;
            }
            hook->DeferPostView(batch_token);
            return false;
        }

        if (!PublishKvafLayerResidentDescriptor(control_, lease, descriptor)) {
            RollbackViewClaims(view);
            if (!ReturnKvafLayerResidentUnpostedQueueFromAicore(
                    control_gm_, lease_result_, simt_event_, &lease)) {
                healthy_ = false;
            }
            hook->DeferPostView(batch_token);
            return false;
        }

        if (!BeginKvafLayerResidentBatchGet(control_, lease, &wave_)) {
            // Begin reports a terminal non-post failure through the ordinary
            // KVAF transaction.  Finish/terminal-return is mandatory here;
            // do not route this state through the Held-only unposted helper.
            KvafLayerResidentBatchGetTerminal ignored{};
            (void)FinishKvafLayerResidentBatchGet(&wave_, &ignored);
            if (!ReturnKvafLayerResidentQueueFromAicore(
                    control_gm_, lease_result_, simt_event_, &wave_)) {
                healthy_ = false;
            }
            RollbackViewClaims(view);
            hook->DeferPostView(batch_token);
            return false;
        }

        // At this point Direct-HCCP owns the fully published descriptor and
        // may write every selected fixed row.  Record the in-flight wave
        // *before* committing the invocation-local maps: a map-publication
        // fault after Begin must still have a reachable CQ owner for the
        // final operator drain.  It is never valid to lose that owner merely
        // because the fused invocation will subsequently fail closed.
        has_wave_ = true;

        // Publish both invocation-local maps, then release the UB seeds;
        // wave_ retains all post/CQ state.
        if (!CommitViewClaims(view)) {
            // No unsafe rollback after Begin.  Keep the wave owned and make
            // Stage-B fail closed after its terminal drain rather than reuse
            // an output row whose HCCP visibility can no longer be proved.
            healthy_ = false;
            return false;
        }
        hook->CompletePostView(batch_token);
        return true;
    }

    // Stage B uses this after official Top-K is final.  Its seeds already
    // name otherwise-unused rows in the caller's fixed canonical interval;
    // unlike score-time speculation, no source->slot reservation or map
    // publication is needed.  The call remains PostOnly: the finalizer drains
    // this one local wave before it reuses the seed scratch for the next
    // bounded recall microbatch or exposes attention_slots to QSFA.
    __aicore__ inline bool TryPostFixedRecallSeeds(
        __ubuf__ const CandidateSeed* seeds, uint32_t count,
        const CanonicalResidentLayout& resident_layout,
        uint32_t rows_per_entry)
    {
        if (!Enabled() || seeds == nullptr || count == 0U ||
            count > kExactRecallWaveCapacity || rows_per_entry == 0U ||
            !ValidateResidentLayout(resident_layout)) {
            return false;
        }
        TryRetireAtScoreBoundary();
        if (!healthy_ || has_wave_) {
            return false;
        }

        KvafLayerResidentQueueLease lease{};
        if (!TryAcquireKvafLayerResidentQueueFromAicore(
                control_gm_, producer_id_, queue_hint_, lease_result_, simt_event_,
                &lease)) {
            return false;
        }
        queue_hint_ = lease.queue_id + 1U;

        KvafLayerResidentBatchGetDescriptorScratch descriptor{};
        if (!LoadKvafLayerResidentQueueTemplate(control_, lease, &descriptor)) {
            if (!ReturnKvafLayerResidentUnpostedQueueFromAicore(
                    control_gm_, lease_result_, simt_event_, &lease)) {
                healthy_ = false;
            }
            return false;
        }

        uint32_t accepted_count = 0U;
        for (uint32_t index = 0U; index < count; ++index) {
            const CandidateSeed& seed = seeds[index];
            if (!IsFixedRecallSeedValid(seed, resident_layout) ||
                !AppendKvafLayerResidentRequest(
                    control_, &descriptor, static_cast<uint32_t>(seed.kvObject),
                    rows_per_entry, seed.residentSlot)) {
                healthy_ = false;
                break;
            }
            ++accepted_count;
        }

        if (!healthy_ || accepted_count != count) {
            if (!ReturnKvafLayerResidentUnpostedQueueFromAicore(
                    control_gm_, lease_result_, simt_event_, &lease)) {
                healthy_ = false;
            }
            return false;
        }
        if (!PublishKvafLayerResidentDescriptor(control_, lease, descriptor)) {
            if (!ReturnKvafLayerResidentUnpostedQueueFromAicore(
                    control_gm_, lease_result_, simt_event_, &lease)) {
                healthy_ = false;
            }
            return false;
        }
        if (!BeginKvafLayerResidentBatchGet(control_, lease, &wave_)) {
            KvafLayerResidentBatchGetTerminal ignored{};
            (void)FinishKvafLayerResidentBatchGet(&wave_, &ignored);
            if (!ReturnKvafLayerResidentQueueFromAicore(
                    control_gm_, lease_result_, simt_event_, &wave_)) {
                healthy_ = false;
            }
            return false;
        }
        // Doorbell is now visible: retain the sole temporary queue owner
        // before any later local validation can fail.
        has_wave_ = true;
        return true;
    }

    // Final Stage-B barrier.  It may wait only after official Top-K has been
    // computed and all remaining exact recall posts have been issued.  It is
    // the sole condition before the fused operator may expose attention slots
    // to QSFA.
    __aicore__ inline bool DrainAllBeforeReturn()
    {
        if (!initialized_) {
            return false;
        }
        if (!has_wave_) {
            return healthy_;
        }
        // A failed map publication after Begin still needs the hardware wave
        // drained and its temporary queue returned.  Preserve that cleanup
        // guarantee, then propagate the fail-closed health result.
        const bool terminal = FinalizeAndReturnTerminalWave();
        return terminal && healthy_;
    }

private:
    __aicore__ inline bool ValidateResidentLayout(
        const CanonicalResidentLayout& layout) const
    {
        if (!IsCanonicalLayout(layout)) {
            return false;
        }
        const int64_t range_end =
            static_cast<int64_t>(control_.resident_row_base) +
            control_.resident_row_count;
        const int64_t layout_end =
            static_cast<int64_t>(layout.firstSlot) +
            static_cast<int64_t>(layout.rowCount - 1U) * layout.rowStride +
            layout.slotsPerRow;
        return layout.firstSlot >= control_.resident_row_base &&
               layout_end <= range_end;
    }

    __aicore__ inline bool IsFixedRecallSeedValid(
        const CandidateSeed& seed,
        const CanonicalResidentLayout& resident_layout) const
    {
        return seed.row >= 0 && seed.source >= 0 && seed.kvObject >= 0 &&
               static_cast<uint32_t>(seed.row) < resident_layout.rowCount &&
               IsSlotInCanonicalRow(
                   resident_layout, static_cast<uint32_t>(seed.row),
                   seed.residentSlot);
    }

    __aicore__ inline bool ValidateView(const CandidatePostView& view) const
    {
        if (view.seeds == nullptr || view.count == 0U ||
            view.count > kCandidateMicrobatchCapacity ||
            view.candidateSourceToSlot == nullptr ||
            view.candidateSlotCursor == nullptr ||
            reinterpret_cast<__gm__ const uint8_t*>(
                view.layerResidentGetControl) !=
                reinterpret_cast<__gm__ const uint8_t*>(control_gm_) ||
            view.rowCount == 0U ||
            view.rowCount != view.residentLayout.rowCount ||
            view.sourceCapacity == 0U || view.rowsPerEntry == 0U ||
            !IsCanonicalLayout(view.residentLayout)) {
            return false;
        }
        return ValidateResidentLayout(view.residentLayout);
    }

    __aicore__ inline bool ReserveSeed(const CandidatePostView& view,
                                       CandidateSeed* seed)
    {
        if (seed == nullptr || seed->row < 0 || seed->source < 0 ||
            seed->kvObject < 0 ||
            static_cast<uint32_t>(seed->row) >= view.residentLayout.rowCount ||
            static_cast<uint32_t>(seed->source) >= view.sourceCapacity) {
            return false;
        }
        if (!ReserveCandidateResidentSlotFromAicore(
                view.candidateSourceToSlot, view.candidateSlotCursor,
                static_cast<uint32_t>(seed->row),
                static_cast<uint32_t>(seed->source), view.sourceCapacity,
                view.residentLayout, reservation_result_, simt_event_)) {
            return false;
        }
        seed->residentSlot = reservation_result_->residentSlot;
        return true;
    }

    __aicore__ inline void RollbackViewClaims(const CandidatePostView& view)
    {
        for (uint32_t seed_index = 0U; seed_index < view.count; ++seed_index) {
            const CandidateSeed& seed = view.seeds[seed_index];
            if (seed.residentSlot < 0 || seed.row < 0 || seed.source < 0) {
                continue;
            }
            const uint64_t source_offset =
                static_cast<uint64_t>(seed.row) * view.sourceCapacity + seed.source;
            RollbackCandidateResidentSlotFromAicore(
                view.candidateSourceToSlot, source_offset, reservation_result_,
                simt_event_);
        }
    }

    __aicore__ inline bool CommitViewClaims(const CandidatePostView& view)
    {
        for (uint32_t seed_index = 0U; seed_index < view.count; ++seed_index) {
            const CandidateSeed& seed = view.seeds[seed_index];
            if (seed.residentSlot < 0) {
                continue;
            }
            const uint32_t row = static_cast<uint32_t>(seed.row);
            if (!CommitCandidateResidentSlotFromAicore(
                    view.candidateSourceToSlot,
                    static_cast<uint64_t>(row) * view.sourceCapacity + seed.source,
                    seed.residentSlot, reservation_result_, simt_event_)) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool FinalizeAndReturnTerminalWave()
    {
        if (!has_wave_) {
            return true;
        }
        KvafLayerResidentBatchGetTerminal terminal{};
        if (!FinishKvafLayerResidentBatchGet(&wave_, &terminal) ||
            terminal.success == 0U ||
            !ReturnKvafLayerResidentQueueFromAicore(
                control_gm_, lease_result_, simt_event_, &wave_)) {
            healthy_ = false;
            return false;
        }
        has_wave_ = false;
        return true;
    }

    __gm__ const KvafLayerResidentBatchGetControl* control_gm_ = nullptr;
    KvafLayerResidentBatchGetControl control_{};
    __ubuf__ KvafLayerResidentQueueLeaseResult* lease_result_ = nullptr;
    __ubuf__ CandidateSlotReservationResult* reservation_result_ = nullptr;
    KvafLayerResidentBatchGetWave wave_{};
    // EVENT_ID0 is a mutable S<->V hand-off default.  CANN accepts this
    // established form at set_flag/wait_flag call sites; Init overwrites it
    // with the event selected by the actual source-derived C8 invocation.
    event_t simt_event_ = static_cast<event_t>(EVENT_ID0);
    uint32_t producer_id_ = 0U;
    uint32_t queue_hint_ = 0U;
    bool initialized_ = false;
    bool healthy_ = false;
    bool has_wave_ = false;
};

}  // namespace exact_speculative_resident
}  // namespace QLIKernel

#endif  // VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_POST_EXECUTOR_H_
