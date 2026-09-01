// P8 forward-owned score-receipt -> speculative Direct-HCCP executor.
//
// C8 is intentionally absent from this source closure.  C8 publishes score
// receipts; this operator owns only the bounded transport transaction that
// follows.  It has four explicit graph phases:
//   POST          CandidateSeed -> global private slot -> durable wave -> PostOnly
//   DRAIN         GM wave -> nonblocking CQ poll/Finish -> private READY/FAILED
//   QSFA_TERMINAL private READY/FAILED -> FREE after QSFA's final reader
//   FORWARD_TERM  quarantine anything unresolved; never free an in-flight DMA
//
// There is no 31+1 core reservation.  Every launched AIV executes the same
// dynamic queue-lease code.  A queue is released once CQ and private-state
// publication are terminal; a private HBM slot is retained separately until
// the QSFA-terminal phase.

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

#include "a5_p8_forward_owned_executor_tiling.h"
#include "kvaf_device/kvaf_fused_full_get_device.h"

using namespace AscendC;

namespace {
constexpr int32_t kUnmapped = -1;
constexpr int32_t kClaiming = -2;
constexpr int32_t kCandidateStatusOk = 1;
constexpr int32_t kMappedObjectAddressDomain = 1;
constexpr uint32_t kPackedKvRowBytes = 656U;

enum P8WorkerStat : uint32_t {
    kP8StatReceiptSeen = 0U,
    kP8StatCandidateScanned = 1U,
    kP8StatCandidateDeferred = 2U,
    kP8StatPrivateClaimed = 3U,
    kP8StatQueueLeased = 4U,
    kP8StatPostCommitted = 5U,
    kP8StatCqPoll = 6U,
    kP8StatCqTerminal = 7U,
    kP8StatQueueReleased = 8U,
    kP8StatPrivateReleased = 9U,
    kP8StatControlRejected = 10U,
    kP8StatQuarantined = 11U,
    kP8StatForwardTerminalWave = 12U,
    kP8StatForwardTerminalPrivate = 13U,
    kP8StatAbortPublished = 14U,
};

struct P8PrivateClaimResult {
    uint64_t source_offset;
    uint64_t object_offset;
    int32_t output_index;
    int32_t success;
};
static_assert(sizeof(P8PrivateClaimResult) == 24U,
              "P8 global-private claim ABI changed");

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void ClaimP8GlobalPrivateSlotSimt(
    __gm__ int32_t* source_to_slot, __gm__ int32_t* object_to_slot,
    __gm__ int32_t* private_ready, __gm__ int32_t* private_slot_cursor,
    __gm__ kvaf_fused_get::KvafP8PrivateRecord* private_records,
    uint32_t private_row_count, uint64_t source_offset,
    uint64_t object_offset, uint64_t forward_epoch,
    uint64_t replay_generation, uint32_t layer_control_index,
    uint32_t queue_id, uint64_t queue_generation, uint64_t post_sequence,
    int32_t owner_tag,
    __ubuf__ P8PrivateClaimResult* result)
{
    if (source_to_slot == nullptr || object_to_slot == nullptr ||
        private_ready == nullptr || private_slot_cursor == nullptr ||
        private_records == nullptr || result == nullptr ||
        private_row_count == 0U || forward_epoch == 0U ||
        queue_generation == 0U || post_sequence == 0U || owner_tag <= 0) {
        return;
    }
    result->source_offset = source_offset;
    result->object_offset = object_offset;
    result->output_index = kUnmapped;
    result->success = 0;

    // `-2` is only a short claim marker.  We never adopt a nonnegative old
    // mapping here: it might belong to another graph replay, and the exact
    // final Recall makes deferral safe.
    if (asc_atomic_cas(source_to_slot + source_offset, kUnmapped,
                       kClaiming) != kUnmapped) {
        return;
    }
    if (asc_atomic_cas(object_to_slot + object_offset, kUnmapped,
                       kClaiming) != kUnmapped) {
        asc_atomic_cas(source_to_slot + source_offset, kClaiming, kUnmapped);
        return;
    }

    uint32_t selected = private_row_count;
    for (uint32_t attempt = 0U; attempt < private_row_count; ++attempt) {
        const uint32_t cursor = asc_atomic_add(private_slot_cursor, 1U);
        const uint32_t slot = cursor % private_row_count;
        // `state` follows magic/version in the fixed 80-byte ABI.  Use a
        // scalar atomic rather than an aggregate GM write so two AIVs cannot
        // claim the same physical private output row.
        __gm__ int32_t* const state = reinterpret_cast<__gm__ int32_t*>(
            private_records + slot) + 3U;
        if (asc_atomic_cas(state, kvaf_fused_get::kKvafP8PrivateFree,
                           kvaf_fused_get::kKvafP8PrivatePending) ==
            kvaf_fused_get::kKvafP8PrivateFree) {
            selected = slot;
            break;
        }
    }
    if (selected == private_row_count) {
        asc_atomic_cas(object_to_slot + object_offset, kClaiming, kUnmapped);
        asc_atomic_cas(source_to_slot + source_offset, kClaiming, kUnmapped);
        return;
    }

    __gm__ kvaf_fused_get::KvafP8PrivateRecord* const record =
        private_records + selected;
    // The state CAS above makes this row exclusively ours.  Do not manually
    // publish a cross-address-space private-record aggregate here: the KVAF
    // helper owns the fixed ABI and field-wise GM store.  This keeps the
    // private record matched to the future durable wave identity before it
    // becomes discoverable through source/object maps.
    kvaf_fused_get::KvafP8WaveIdentity identity{};
    identity.forward_epoch = forward_epoch;
    identity.graph_replay_generation = replay_generation;
    identity.layer_control_index = layer_control_index;
    identity.queue_id = queue_id;
    identity.queue_generation = queue_generation;
    identity.post_sequence = post_sequence;
    identity.owner_tag = owner_tag;
    identity.reserved0 = 0U;
    kvaf_fused_get::KvafP8PrivateRecord prepared{};
    if (!kvaf_fused_get::PrepareKvafP8PrivateRecord(
            identity, source_offset, object_offset,
            static_cast<int32_t>(selected), &prepared) ||
        !kvaf_fused_get::StoreKvafP8PrivateRecordToGm(record, prepared)) {
        // Keep the row durably non-reusable.  The source/object claims are
        // rolled back because no descriptor can have observed this row yet.
        kvaf_fused_get::ResetKvafP8PrivateRecord(&prepared);
        prepared.state = kvaf_fused_get::kKvafP8PrivateQuarantined;
        (void)kvaf_fused_get::StoreKvafP8PrivateRecordToGm(record, prepared);
        private_ready[selected] = static_cast<int32_t>(
            kvaf_fused_get::kKvafP8PrivateQuarantined);
        asc_atomic_cas(object_to_slot + object_offset, kClaiming, kUnmapped);
        asc_atomic_cas(source_to_slot + source_offset, kClaiming, kUnmapped);
        return;
    }
    private_ready[selected] = static_cast<int32_t>(
        kvaf_fused_get::kKvafP8PrivatePending);
    AscendC::Simt::ThreadFence();

    if (asc_atomic_cas(object_to_slot + object_offset, kClaiming,
                       static_cast<int32_t>(selected)) != kClaiming ||
        asc_atomic_cas(source_to_slot + source_offset, kClaiming,
                       static_cast<int32_t>(selected)) != kClaiming) {
        // A partial publication cannot be silently recycled.  Quarantine the
        // row and leave final exact Recall responsible for correctness.
        prepared.state = kvaf_fused_get::kKvafP8PrivateQuarantined;
        (void)kvaf_fused_get::StoreKvafP8PrivateRecordToGm(record, prepared);
        private_ready[selected] = static_cast<int32_t>(
            kvaf_fused_get::kKvafP8PrivateQuarantined);
        asc_atomic_cas(source_to_slot + source_offset, kClaiming, kUnmapped);
        asc_atomic_cas(object_to_slot + object_offset, kClaiming, kUnmapped);
        return;
    }
    AscendC::Simt::ThreadFence();
    result->output_index = static_cast<int32_t>(selected);
    result->success = 1;
}

class A5P8ForwardOwnedExecutorKernel {
public:
    __aicore__ inline explicit A5P8ForwardOwnedExecutorKernel(
        TPipe* pipe, const A5P8ForwardOwnedExecutorTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR candidate_seeds, GM_ADDR candidate_counts,
        GM_ADDR candidate_ready_epoch, GM_ADDR candidate_status,
        GM_ADDR forward_control, GM_ADDR source_lens,
        GM_ADDR kv_object_address_domain, GM_ADDR req_pool_entries,
        GM_ADDR kv_object_map, GM_ADDR kv_object_lens,
        GM_ADDR kv_object_slots_snapshot, GM_ADDR private_source_to_slot,
        GM_ADDR private_object_to_slot, GM_ADDR private_ready_table,
        GM_ADDR private_slot_cursor, GM_ADDR private_packed_output,
        GM_ADDR fused_get_control, GM_ADDR p8_wave_records,
        GM_ADDR p8_private_records, GM_ADDR queue_request_source_offsets,
        GM_ADDR queue_request_object_offsets,
        GM_ADDR queue_request_output_indices, GM_ADDR queue_request_count,
        GM_ADDR candidate_cursor, GM_ADDR candidate_cursor_epoch,
        GM_ADDR rows_per_entry, GM_ADDR worker_stats)
    {
        candidateSeeds_ = reinterpret_cast<__gm__ int32_t*>(candidate_seeds);
        candidateCounts_ = reinterpret_cast<__gm__ int32_t*>(candidate_counts);
        candidateReadyEpoch_ = reinterpret_cast<__gm__ int64_t*>(candidate_ready_epoch);
        candidateStatus_ = reinterpret_cast<__gm__ int32_t*>(candidate_status);
        forwardControlGm_ = reinterpret_cast<__gm__
            kvaf_fused_get::KvafP8ForwardControl*>(forward_control);
        sourceLens_ = reinterpret_cast<__gm__ int32_t*>(source_lens);
        kvObjectAddressDomain_ = reinterpret_cast<__gm__ int32_t*>(kv_object_address_domain);
        reqPoolEntries_ = reinterpret_cast<__gm__ int32_t*>(req_pool_entries);
        kvObjectMap_ = reinterpret_cast<__gm__ int32_t*>(kv_object_map);
        kvObjectLens_ = reinterpret_cast<__gm__ int32_t*>(kv_object_lens);
        kvObjectSlotsSnapshot_ = reinterpret_cast<__gm__ int32_t*>(kv_object_slots_snapshot);
        privateSourceToSlot_ = reinterpret_cast<__gm__ int32_t*>(private_source_to_slot);
        privateObjectToSlot_ = reinterpret_cast<__gm__ int32_t*>(private_object_to_slot);
        privateReady_ = reinterpret_cast<__gm__ int32_t*>(private_ready_table);
        privateSlotCursor_ = reinterpret_cast<__gm__ int32_t*>(private_slot_cursor);
        privatePackedOutput_ = reinterpret_cast<__gm__ uint8_t*>(private_packed_output);
        controlGm_ = reinterpret_cast<__gm__ const kvaf_fused_get::KvafStreamingBatchGetControl*>(
            fused_get_control);
        waveRecords_ = reinterpret_cast<__gm__ kvaf_fused_get::KvafP8WaveRecord*>(
            p8_wave_records);
        privateRecords_ = reinterpret_cast<__gm__ kvaf_fused_get::KvafP8PrivateRecord*>(
            p8_private_records);
        queueRequestSourceOffsets_ = reinterpret_cast<__gm__ int64_t*>(
            queue_request_source_offsets);
        queueRequestObjectOffsets_ = reinterpret_cast<__gm__ int64_t*>(
            queue_request_object_offsets);
        queueRequestOutputIndices_ = reinterpret_cast<__gm__ int32_t*>(
            queue_request_output_indices);
        queueRequestCount_ = reinterpret_cast<__gm__ int32_t*>(queue_request_count);
        candidateCursor_ = reinterpret_cast<__gm__ int32_t*>(candidate_cursor);
        candidateCursorEpoch_ = reinterpret_cast<__gm__ int64_t*>(candidate_cursor_epoch);
        rowsPerEntry_ = reinterpret_cast<__gm__ int32_t*>(rows_per_entry);
        workerStats_ = reinterpret_cast<__gm__ int64_t*>(worker_stats);
        worker_ = GetBlockIdx();
        enabled_ = false;

        if (pipe_ == nullptr || tiling_ == nullptr ||
            worker_ >= tiling_->workerCount ||
            candidateSeeds_ == nullptr || candidateCounts_ == nullptr ||
            candidateReadyEpoch_ == nullptr || candidateStatus_ == nullptr ||
            forwardControlGm_ == nullptr || sourceLens_ == nullptr ||
            kvObjectAddressDomain_ == nullptr || reqPoolEntries_ == nullptr ||
            kvObjectMap_ == nullptr || kvObjectLens_ == nullptr ||
            kvObjectSlotsSnapshot_ == nullptr || privateSourceToSlot_ == nullptr ||
            privateObjectToSlot_ == nullptr || privateReady_ == nullptr ||
            privateSlotCursor_ == nullptr || privatePackedOutput_ == nullptr ||
            controlGm_ == nullptr || waveRecords_ == nullptr ||
            privateRecords_ == nullptr || queueRequestSourceOffsets_ == nullptr ||
            queueRequestObjectOffsets_ == nullptr ||
            queueRequestOutputIndices_ == nullptr || queueRequestCount_ == nullptr ||
            candidateCursor_ == nullptr || candidateCursorEpoch_ == nullptr ||
            rowsPerEntry_ == nullptr || workerStats_ == nullptr) {
            return;
        }
        if (!kvaf_fused_get::LoadKvafStreamingBatchGetControlFromGmForAicore(
                controlGm_, &control_) ||
            !kvaf_fused_get::IsKvafStreamingBatchGetControlValidForAicore(control_) ||
            control_.queue_count != tiling_->queueCount ||
            control_.output != reinterpret_cast<uint64_t>(privatePackedOutput_) ||
            control_.output_rows != tiling_->privateRowCount) {
            AddStat(kP8StatControlRejected, 1U);
            return;
        }
        if (!kvaf_fused_get::LoadKvafP8ForwardControlFromGm(
                forwardControlGm_, &forwardControl_) ||
            forwardControl_.phase < kA5P8PhasePost ||
            forwardControl_.phase > kA5P8PhaseForwardTerminal ||
            ReadGmByPassDCache<int32_t>(kvObjectAddressDomain_) !=
                kMappedObjectAddressDomain) {
            AddStat(kP8StatControlRejected, 1U);
            return;
        }
        forwardEpoch_ = forwardControl_.forward_epoch;
        replayGeneration_ = forwardControl_.graph_replay_generation;
        layerControlIndex_ = forwardControl_.layer_control_index;
        phase_ = forwardControl_.phase;
        pipe_->InitBuffer(leaseBuffer_, sizeof(kvaf_fused_get::KvafStreamingBatchGetLeaseResult));
        pipe_->InitBuffer(claimBuffer_, sizeof(P8PrivateClaimResult));
        pipe_->InitBuffer(drainClaimBuffer_,
                          sizeof(kvaf_fused_get::KvafP8WaveDrainClaimResult));
        pipe_->InitBuffer(abortBuffer_,
                          sizeof(kvaf_fused_get::KvafP8ForwardAbortResult));
        pipe_->InitBuffer(privateReleaseBuffer_,
                          sizeof(kvaf_fused_get::KvafP8PrivateReleaseResult));
        pipe_->InitBuffer(forwardTerminalBuffer_,
                          sizeof(kvaf_fused_get::KvafP8ForwardTerminalResult));
        leaseResult_ = (__ubuf__ kvaf_fused_get::KvafStreamingBatchGetLeaseResult*)
            leaseBuffer_.Get<uint8_t>().GetPhyAddr();
        claimResult_ = (__ubuf__ P8PrivateClaimResult*)
            claimBuffer_.Get<uint8_t>().GetPhyAddr();
        drainClaimResult_ = (__ubuf__ kvaf_fused_get::KvafP8WaveDrainClaimResult*)
            drainClaimBuffer_.Get<uint8_t>().GetPhyAddr();
        abortResult_ = (__ubuf__ kvaf_fused_get::KvafP8ForwardAbortResult*)
            abortBuffer_.Get<uint8_t>().GetPhyAddr();
        privateReleaseResult_ =
            (__ubuf__ kvaf_fused_get::KvafP8PrivateReleaseResult*)
                privateReleaseBuffer_.Get<uint8_t>().GetPhyAddr();
        forwardTerminalResult_ =
            (__ubuf__ kvaf_fused_get::KvafP8ForwardTerminalResult*)
                forwardTerminalBuffer_.Get<uint8_t>().GetPhyAddr();
        if (leaseResult_ == nullptr || claimResult_ == nullptr ||
            drainClaimResult_ == nullptr || abortResult_ == nullptr ||
            privateReleaseResult_ == nullptr || forwardTerminalResult_ == nullptr) {
            AddStat(kP8StatControlRejected, 1U);
            return;
        }
        enabled_ = true;
    }

    __aicore__ inline void Process()
    {
        if (!enabled_) {
            return;
        }
        if (phase_ == kA5P8PhasePost) {
            RunPostOnly();
        } else if (phase_ == kA5P8PhaseDrain) {
            RunDrain();
        } else if (phase_ == kA5P8PhaseQsfaTerminal) {
            RunQsfaTerminalRelease();
        } else {
            RunForwardTerminal();
        }
    }

private:
    __aicore__ inline void AddStat(uint32_t column, uint64_t delta)
    {
        if (workerStats_ == nullptr || column >= kA5P8WorkerStatColumns ||
            worker_ >= tiling_->workerStatsRows) {
            return;
        }
        const uint64_t offset = static_cast<uint64_t>(worker_) *
            kA5P8WorkerStatColumns + column;
        const int64_t old = ReadGmByPassDCache<int64_t>(workerStats_ + offset);
        WriteGmByPassDCache<int64_t>(workerStats_ + offset,
                                     old + static_cast<int64_t>(delta));
    }

    __aicore__ inline __gm__ int32_t* Candidate(uint32_t receipt,
                                                  uint32_t index) const
    {
        return candidateSeeds_ +
            (static_cast<uint64_t>(receipt) * tiling_->candidateCapacity + index) *
            kA5P8CandidateSeedWords;
    }

    __aicore__ inline uint64_t LedgerOffset(uint32_t queue, uint32_t request) const
    {
        return static_cast<uint64_t>(queue) * tiling_->requestCapacity + request;
    }

    __aicore__ inline bool SameForward(
        const kvaf_fused_get::KvafP8WaveIdentity& identity) const
    {
        return identity.forward_epoch == forwardEpoch_ &&
            identity.graph_replay_generation == replayGeneration_ &&
            identity.layer_control_index == layerControlIndex_;
    }

    __aicore__ inline bool SamePrivateForward(
        const kvaf_fused_get::KvafP8PrivateRecord& record) const
    {
        return kvaf_fused_get::IsKvafP8PrivateRecordValid(record) &&
            record.forward_epoch == forwardEpoch_ &&
            record.graph_replay_generation == replayGeneration_ &&
            record.layer_control_index == layerControlIndex_;
    }

    __aicore__ inline bool ClaimGlobalPrivate(
        uint64_t source_offset, uint64_t object_offset,
        const kvaf_fused_get::KvafP8WaveIdentity& identity)
    {
        claimResult_->success = 0;
        set_flag(PIPE_S, PIPE_V, simtEvent_);
        wait_flag(PIPE_S, PIPE_V, simtEvent_);
        AscendC::Simt::VF_CALL<ClaimP8GlobalPrivateSlotSimt>(
            AscendC::Simt::Dim3{1U, 1U, 1U}, privateSourceToSlot_,
            privateObjectToSlot_, privateReady_, privateSlotCursor_,
            privateRecords_, tiling_->privateRowCount, source_offset,
            object_offset, identity.forward_epoch,
            identity.graph_replay_generation, identity.layer_control_index,
            identity.queue_id, identity.queue_generation,
            identity.post_sequence, identity.owner_tag, claimResult_);
        set_flag(PIPE_V, PIPE_S, simtEvent_);
        wait_flag(PIPE_V, PIPE_S, simtEvent_);
        return claimResult_->success != 0;
    }

    __aicore__ inline bool ReleaseUnpostedLease(
        const kvaf_fused_get::KvafStreamingBatchGetLease& lease)
    {
        kvaf_fused_get::KvafStreamingBatchGetWave wave{};
        wave.lease = lease;
        wave.lease.state = kvaf_fused_get::kStreamingBatchGetLeaseTerminal;
        kvaf_fused_get::ResetDenseBatchGetTransaction(&wave.transaction);
        wave.transaction.finalized = 1U;
        return kvaf_fused_get::ReleaseKvafStreamingBatchGetQueueFromAicore(
            controlGm_, leaseResult_, simtEvent_, &wave);
    }

    // Each graph phase reloads the control record instead of trusting the
    // Init-time cache.  In particular POST reloads immediately before the
    // irreversible HCCP doorbell so a concurrent worker's monotonic ABORT
    // cannot be ignored or overwritten by stale control bytes.
    __aicore__ inline bool ReloadForwardControlForPhase(
        kvaf_fused_get::KvafP8ForwardPhase expected_phase,
        bool allow_abort)
    {
        kvaf_fused_get::KvafP8ForwardControl reloaded{};
        if (!kvaf_fused_get::LoadKvafP8ForwardControlFromGm(
                forwardControlGm_, &reloaded) ||
            reloaded.forward_epoch != forwardEpoch_ ||
            reloaded.graph_replay_generation != replayGeneration_ ||
            reloaded.layer_control_index != layerControlIndex_ ||
            reloaded.phase != expected_phase) {
            return false;
        }
        if (!allow_abort &&
            (reloaded.flags &
             (kvaf_fused_get::kKvafP8ForwardFlagAbort |
              kvaf_fused_get::kKvafP8ForwardFlagPrivateViewDisabled)) != 0U) {
            return false;
        }
        forwardControl_ = reloaded;
        phase_ = reloaded.phase;
        return true;
    }

    // P8 is fail-closed: a transport identity/terminal-publication error must
    // not leave an earlier private READY mapping visible to QSFA.  This is a
    // monotonic GM CAS, never a local read/modify/write.  Reloading afterward
    // makes a losing worker observe the same graph-visible gate.
    __aicore__ inline bool MarkForwardAbort()
    {
        abortResult_->success = 0U;
        abortResult_->observed_flags_low = 0U;
        if (!kvaf_fused_get::MarkKvafP8ForwardAbortFromAicore(
                forwardControlGm_, abortResult_, simtEvent_)) {
            AddStat(kP8StatControlRejected, 1U);
            return false;
        }
        AddStat(kP8StatAbortPublished, 1U);
        return ReloadForwardControlForPhase(
            static_cast<kvaf_fused_get::KvafP8ForwardPhase>(phase_), true);
    }

    // Before POST_INTENT, no HCCP DMA can have observed these rows.  We could
    // try to recover their maps locally, but a partial CAS is indistinguishable
    // from an interrupted graph invocation.  Keep them quarantined instead:
    // exact Recall preserves math and ForwardTerminal owns cold recovery.
    __aicore__ inline void QuarantineUnpostedRows(
        const kvaf_fused_get::KvafP8WaveIdentity& identity,
        uint32_t request_count)
    {
        const uint32_t queue = identity.queue_id;
        for (uint32_t request = 0U; request < request_count; ++request) {
            const uint64_t offset = LedgerOffset(queue, request);
            const int32_t output_index = ReadGmByPassDCache<int32_t>(
                queueRequestOutputIndices_ + offset);
            if (output_index < 0 || static_cast<uint32_t>(output_index) >=
                                   tiling_->privateRowCount) {
                continue;
            }
            kvaf_fused_get::KvafP8PrivateRecord record{};
            if (!kvaf_fused_get::LoadKvafP8PrivateRecordFromGm(
                    privateRecords_ + output_index, &record) ||
                !kvaf_fused_get::KvafP8PrivateRecordMatchesIdentity(record,
                                                                      identity)) {
                continue;
            }
            record.state = kvaf_fused_get::kKvafP8PrivateQuarantined;
            (void)kvaf_fused_get::StoreKvafP8PrivateRecordToGm(
                privateRecords_ + output_index, record);
            privateReady_[output_index] = static_cast<int32_t>(
                kvaf_fused_get::kKvafP8PrivateQuarantined);
            AddStat(kP8StatQuarantined, 1U);
        }
        WriteGmByPassDCache<int32_t>(queueRequestCount_ + queue, 0);
        DataSyncBarrier<MemDsbT::DDR>();
    }

    // A candidate that is malformed, resident, already claimed, or outside
    // this row's active prefix is intentionally *consumed but not posted*.
    // Exact Top-K plus Guaranteed Recall remains the authority for output.
    __aicore__ inline bool StageCandidate(
        __gm__ int32_t* seed,
        const kvaf_fused_get::KvafP8WaveIdentity& identity,
        const kvaf_fused_get::KvafStreamingBatchGetDescriptorScratch& descriptor,
        uint32_t* staged)
    {
        if (seed == nullptr || staged == nullptr ||
            *staged >= tiling_->requestCapacity) {
            return false;
        }
        const int32_t row_value = ReadGmByPassDCache<int32_t>(seed + 0U);
        const int32_t source_value = ReadGmByPassDCache<int32_t>(seed + 1U);
        const int32_t object_value = ReadGmByPassDCache<int32_t>(seed + 2U);
        AddStat(kP8StatCandidateScanned, 1U);
        if (row_value < 0 || source_value < 0 || object_value < 0 ||
            static_cast<uint32_t>(row_value) >= tiling_->rowCount ||
            static_cast<uint32_t>(source_value) >= tiling_->sourceCapacity) {
            return false;
        }
        const uint32_t row = static_cast<uint32_t>(row_value);
        const uint32_t source = static_cast<uint32_t>(source_value);
        const uint32_t object = static_cast<uint32_t>(object_value);
        const int32_t pool_value = ReadGmByPassDCache<int32_t>(
            reqPoolEntries_ + row);
        const int32_t source_length = ReadGmByPassDCache<int32_t>(
            sourceLens_ + row);
        const int32_t object_length = ReadGmByPassDCache<int32_t>(
            kvObjectLens_ + row);
        if (pool_value < 0 || static_cast<uint32_t>(pool_value) >=
                                 tiling_->poolSize ||
            source_length < 0 || static_cast<uint32_t>(source_length) >
                                 tiling_->sourceCapacity ||
            source >= static_cast<uint32_t>(source_length) ||
            object_length < 0 || static_cast<uint32_t>(object_length) >
                                 tiling_->kvObjectCapacity ||
            object >= static_cast<uint32_t>(object_length)) {
            return false;
        }
        const uint32_t pool = static_cast<uint32_t>(pool_value);
        if (ReadGmByPassDCache<int32_t>(kvObjectMap_ +
                static_cast<uint64_t>(pool) * tiling_->sourceCapacity + source) !=
                object_value ||
            ReadGmByPassDCache<int32_t>(kvObjectSlotsSnapshot_ +
                static_cast<uint64_t>(pool) * tiling_->kvObjectCapacity + object) >= 0) {
            return false;
        }
        const int32_t rows_per_entry = ReadGmByPassDCache<int32_t>(rowsPerEntry_);
        if (rows_per_entry <= 0) {
            return false;
        }
        const uint32_t table_index = object /
            static_cast<uint32_t>(rows_per_entry);
        if (table_index >= descriptor.table_capacity) {
            return false;
        }
        __gm__ int32_t* const indices = reinterpret_cast<__gm__ int32_t*>(
            descriptor.indices);
        __gm__ int32_t* const offsets = reinterpret_cast<__gm__ int32_t*>(
            descriptor.offsets);
        __gm__ int32_t* const output_indices =
            reinterpret_cast<__gm__ int32_t*>(descriptor.output_indices);
        __gm__ uint8_t* const mask = reinterpret_cast<__gm__ uint8_t*>(
            descriptor.mask);
        if (indices == nullptr || offsets == nullptr || output_indices == nullptr ||
            mask == nullptr) {
            return false;
        }
        const uint64_t source_offset = static_cast<uint64_t>(row) *
            tiling_->sourceCapacity + source;
        const uint64_t object_offset = static_cast<uint64_t>(pool) *
            tiling_->kvObjectCapacity + object;
        // The claim comes after every deterministic descriptor validation.
        // After this point the only work is writes into a prevalidated bounded
        // GM ledger, so we never clear a partially claimed row by guessing
        // whether a later PostOnly may have observed it.
        if (!ClaimGlobalPrivate(source_offset, object_offset, identity) ||
            claimResult_->output_index < 0) {
            AddStat(kP8StatCandidateDeferred, 1U);
            return false;
        }
        const uint32_t request = *staged;
        indices[request] = static_cast<int32_t>(table_index);
        offsets[request] = static_cast<int32_t>((object %
            static_cast<uint32_t>(rows_per_entry)) * kPackedKvRowBytes);
        output_indices[request] = claimResult_->output_index;
        mask[request] = 1U;
        const uint64_t ledger = LedgerOffset(identity.queue_id, request);
        WriteGmByPassDCache<int64_t>(queueRequestSourceOffsets_ + ledger,
            static_cast<int64_t>(source_offset));
        WriteGmByPassDCache<int64_t>(queueRequestObjectOffsets_ + ledger,
            static_cast<int64_t>(object_offset));
        WriteGmByPassDCache<int32_t>(queueRequestOutputIndices_ + ledger,
            claimResult_->output_index);
        ++(*staged);
        AddStat(kP8StatPrivateClaimed, 1U);
        return true;
    }

    __aicore__ inline bool PostReceipt(uint32_t receipt)
    {
        if (!ReloadForwardControlForPhase(
                kvaf_fused_get::kKvafP8ForwardPhaseAdmitPost, false)) {
            AddStat(kP8StatControlRejected, 1U);
            return false;
        }
        if (ReadGmByPassDCache<int64_t>(candidateReadyEpoch_ + receipt) !=
            static_cast<int64_t>(forwardEpoch_) ||
            ReadGmByPassDCache<int32_t>(candidateStatus_ + receipt) !=
                kCandidateStatusOk) {
            return false;
        }
        DataSyncBarrier<MemDsbT::DDR>();
        const int32_t count_value = ReadGmByPassDCache<int32_t>(
            candidateCounts_ + receipt);
        if (count_value <= 0 || static_cast<uint32_t>(count_value) >
                                   tiling_->candidateCapacity) {
            return false;
        }
        int32_t cursor = ReadGmByPassDCache<int32_t>(candidateCursor_ + receipt);
        if (ReadGmByPassDCache<int64_t>(candidateCursorEpoch_ + receipt) !=
            static_cast<int64_t>(forwardEpoch_)) {
            cursor = 0;
            WriteGmByPassDCache<int32_t>(candidateCursor_ + receipt, 0);
            WriteGmByPassDCache<int64_t>(candidateCursorEpoch_ + receipt,
                                         static_cast<int64_t>(forwardEpoch_));
            DataSyncBarrier<MemDsbT::DDR>();
        }
        if (cursor < 0 || static_cast<uint32_t>(cursor) >=
                              static_cast<uint32_t>(count_value)) {
            return false;
        }

        kvaf_fused_get::KvafStreamingBatchGetLease lease{};
        if (!kvaf_fused_get::TryAcquireKvafStreamingBatchGetQueueFromAicore(
                controlGm_, worker_, receipt, leaseResult_, simtEvent_, &lease)) {
            AddStat(kP8StatCandidateDeferred, 1U);
            return false;
        }
        AddStat(kP8StatQueueLeased, 1U);
        const uint32_t queue = lease.queue_id;
        kvaf_fused_get::KvafP8WaveRecord previous{};
        (void)kvaf_fused_get::LoadKvafP8WaveRecordFromGm(
            waveRecords_ + queue, &previous);
        if (kvaf_fused_get::IsKvafP8WaveRecordValid(previous) &&
            previous.state == kvaf_fused_get::kKvafP8WaveQuarantined) {
            (void)ReleaseUnpostedLease(lease);
            AddStat(kP8StatQuarantined, 1U);
            return false;
        }
        kvaf_fused_get::KvafP8WaveIdentity identity{};
        identity.forward_epoch = forwardEpoch_;
        identity.graph_replay_generation = replayGeneration_;
        identity.layer_control_index = layerControlIndex_;
        identity.queue_id = queue;
        identity.queue_generation = kvaf_fused_get::IsKvafP8WaveRecordValid(previous)
            ? previous.identity.queue_generation + 1U : 1U;
        identity.post_sequence = kvaf_fused_get::IsKvafP8WaveRecordValid(previous)
            ? previous.identity.post_sequence + 1U : 1U;
        identity.owner_tag = lease.owner_tag;
        identity.reserved0 = 0U;
        if (identity.queue_generation == 0U || identity.post_sequence == 0U ||
            identity.owner_tag <= 0) {
            (void)ReleaseUnpostedLease(lease);
            AddStat(kP8StatControlRejected, 1U);
            return false;
        }
        kvaf_fused_get::KvafStreamingBatchGetDescriptorScratch descriptor{};
        if (!kvaf_fused_get::LoadKvafStreamingBatchGetQueueTemplate(
                control_, lease, &descriptor) || descriptor.indices == 0U ||
            descriptor.offsets == 0U || descriptor.output_indices == 0U ||
            descriptor.mask == 0U || descriptor.table_capacity == 0U ||
            descriptor.request_capacity < tiling_->requestCapacity) {
            (void)ReleaseUnpostedLease(lease);
            AddStat(kP8StatControlRejected, 1U);
            return false;
        }
        uint32_t staged = 0U;
        uint32_t next = static_cast<uint32_t>(cursor);
        const uint32_t limit = static_cast<uint32_t>(count_value);
        while (next < limit && staged < tiling_->requestCapacity) {
            (void)StageCandidate(Candidate(receipt, next), identity, descriptor,
                                 &staged);
            ++next;
        }
        if (staged == 0U) {
            WriteGmByPassDCache<int32_t>(candidateCursor_ + receipt,
                                         static_cast<int32_t>(next));
            (void)ReleaseUnpostedLease(lease);
            return false;
        }
        descriptor.active_request_count = staged;
        WriteGmByPassDCache<int32_t>(queueRequestCount_ + queue,
                                     static_cast<int32_t>(staged));
        DataSyncBarrier<MemDsbT::DDR>();
        if (!kvaf_fused_get::PublishKvafP8BatchGetDescriptorScratch(
                control_, lease, identity, privateRecords_,
                tiling_->privateRowCount, descriptor)) {
            // No doorbell has run, so the queue lease is safe to release.
            // The private rows stay quarantined instead of attempting a local
            // directory rollback with only partial CAS evidence.
            QuarantineUnpostedRows(identity, staged);
            (void)ReleaseUnpostedLease(lease);
            (void)MarkForwardAbort();
            AddStat(kP8StatControlRejected, 1U);
            return false;
        }
        if (!ReloadForwardControlForPhase(
                kvaf_fused_get::kKvafP8ForwardPhaseAdmitPost, false)) {
            // Still before POST_INTENT: retain the private rows fail-closed,
            // return the non-posted queue, and make the forward gate sticky.
            QuarantineUnpostedRows(identity, staged);
            (void)ReleaseUnpostedLease(lease);
            (void)MarkForwardAbort();
            return false;
        }
        kvaf_fused_get::KvafP8WaveRecord wave{};
        if (!kvaf_fused_get::PrepareKvafP8WavePostIntent(
                control_, forwardControl_, lease, identity, staged,
                privateRecords_, tiling_->privateRowCount,
                queueRequestSourceOffsets_, queueRequestObjectOffsets_,
                queueRequestOutputIndices_, tiling_->requestCapacity, &wave)) {
            // Prepare rejects before PostOnly.  The descriptor had been
            // published, so preserve an explicit quarantined GM record and
            // fail the forward closed rather than guessing that it is safe to
            // recycle the queue/private rows in the POST graph phase.
            (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(
                waveRecords_ + queue, wave);
            DataSyncBarrier<MemDsbT::DDR>();
            MarkForwardAbort();
            return false;
        }
        (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(
            waveRecords_ + queue, wave);
        DataSyncBarrier<MemDsbT::DDR>();
        // POST_INTENT is durable.  Re-read the shared gate immediately
        // before HCCP PostOnly: another AIV may have aborted this forward
        // between descriptor publication and this irreversible operation.
        if (!ReloadForwardControlForPhase(
                kvaf_fused_get::kKvafP8ForwardPhaseAdmitPost, false)) {
            kvaf_fused_get::QuarantineKvafP8WaveRecord(
                &wave, kvaf_fused_get::kKvafP8WaveOutcomeIdentityQuarantined);
            (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(
                waveRecords_ + queue, wave);
            DataSyncBarrier<MemDsbT::DDR>();
            (void)MarkForwardAbort();
            return false;
        }
        const bool posted = kvaf_fused_get::PostKvafP8Wave(
            control_, forwardControl_, privateRecords_, tiling_->privateRowCount,
            &wave);
        (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(
            waveRecords_ + queue, wave);
        DataSyncBarrier<MemDsbT::DDR>();
        if (!posted || wave.state != kvaf_fused_get::kKvafP8WavePosted) {
            WriteGmByPassDCache<int32_t>(candidateCursor_ + receipt,
                                         static_cast<int32_t>(next));
            // There is one narrow non-error outcome: PostOnly deterministically
            // rejected *before* an SQ doorbell, and KVAF recorded the proof as
            // FINALIZED + POST_REJECTED + post_committed=0.  POST may not free
            // it: only a later DRAIN CAS-claims that GM record, releases every
            // PENDING private row through KVAF's verified map-clear protocol,
            // and then releases the held queue.  That avoids both an
            // unsafe POST_INTENT rollback and a permanent capacity leak on an
            // ordinary no-post rejection.
            if (wave.state == kvaf_fused_get::kKvafP8WaveFinalized &&
                wave.outcome == kvaf_fused_get::kKvafP8WaveOutcomePostRejected &&
                wave.post_committed == 0U && wave.transaction.finalized != 0U) {
                AddStat(kP8StatCandidateDeferred, 1U);
                return false;
            }
            // Every other !posted result remains ambiguous after durable
            // POST_INTENT.  Exact Recall preserves math while ForwardTerminal
            // quarantines the non-reusable transport record.
            (void)MarkForwardAbort();
            return false;
        }
        WriteGmByPassDCache<int32_t>(candidateCursor_ + receipt,
                                     static_cast<int32_t>(next));
        AddStat(kP8StatPostCommitted, 1U);
        return true;
    }

    __aicore__ inline void RunPostOnly()
    {
        for (uint32_t receipt = worker_; receipt < tiling_->receiptCount;
             receipt += tiling_->workerCount) {
            AddStat(kP8StatReceiptSeen, 1U);
            if (PostReceipt(receipt)) {
                // One outstanding P8 wave per AIV invocation keeps the GM
                // transaction explicit.  Later graph drain phases rehydrate
                // it; no Process()-tail drain is permitted.
                return;
            }
        }
    }

    __aicore__ inline bool PublishTerminalPrivateRows(
        kvaf_fused_get::KvafP8WaveRecord* wave)
    {
        if (wave == nullptr || wave->request_count == 0U ||
            wave->request_count > tiling_->requestCapacity) {
            return false;
        }
        const uint32_t queue = wave->identity.queue_id;
        if (queue >= tiling_->queueCount ||
            ReadGmByPassDCache<int32_t>(queueRequestCount_ + queue) !=
                static_cast<int32_t>(wave->request_count) ||
            !kvaf_fused_get::PublishKvafP8PrivateTerminalRecords(
                control_, forwardControl_, privateRecords_,
                tiling_->privateRowCount, wave)) {
            return false;
        }
        // The KVAF helper has already reloaded the descriptor and validated
        // every exact output index before setting private_ready_published.
        // Mirror that terminal state into the QSFA-facing readiness table;
        // QSFA must consume READY only, while FAILED remains available to the
        // exact Recall fallback until the QSFA-terminal phase frees it.
        for (uint32_t request = 0U; request < wave->request_count; ++request) {
            const int32_t output_index = ReadGmByPassDCache<int32_t>(
                queueRequestOutputIndices_ +
                LedgerOffset(wave->identity.queue_id, request));
            if (output_index < 0 || static_cast<uint32_t>(output_index) >=
                                   tiling_->privateRowCount) {
                return false;
            }
            kvaf_fused_get::KvafP8PrivateRecord private_record{};
            if (!kvaf_fused_get::LoadKvafP8PrivateRecordFromGm(
                    privateRecords_ + output_index, &private_record) ||
                (private_record.state != kvaf_fused_get::kKvafP8PrivateReady &&
                 private_record.state != kvaf_fused_get::kKvafP8PrivateFailed) ||
                !kvaf_fused_get::KvafP8PrivateRecordMatchesWave(
                    private_record, *wave)) {
                return false;
            }
            privateReady_[output_index] = static_cast<int32_t>(
                private_record.state);
        }
        DataSyncBarrier<MemDsbT::DDR>();
        return kvaf_fused_get::ValidateKvafP8PrivateTerminalPublication(
            control_, forwardControl_, privateRecords_, tiling_->privateRowCount,
            *wave);
    }

    __aicore__ inline bool IsDeterministicNoPostWave(
        uint32_t queue, const kvaf_fused_get::KvafP8WaveRecord& wave) const
    {
        // A drain claim protects the GM state word; this is the remaining
        // complete-identity proof before it clears any PENDING private map.
        // In particular, never treat a plain FINALIZED record as no-post:
        // a CQ-terminal real post is also FINALIZED but has post_committed=1.
        return queue < tiling_->queueCount &&
            kvaf_fused_get::IsKvafP8WaveRecordValid(wave) &&
            wave.state == kvaf_fused_get::kKvafP8WaveDrainClaimed &&
            wave.identity.queue_id == queue &&
            wave.identity.queue_generation != 0U &&
            wave.identity.post_sequence != 0U &&
            wave.identity.owner_tag > 0 &&
            SameForward(wave.identity) &&
            wave.post_committed == 0U &&
            wave.outcome == kvaf_fused_get::kKvafP8WaveOutcomePostRejected &&
            wave.transaction.finalized != 0U;
    }

    __aicore__ inline bool DrainDeterministicNoPostWave(
        uint32_t queue, kvaf_fused_get::KvafP8WaveRecord* wave)
    {
        if (wave == nullptr || !IsDeterministicNoPostWave(queue, *wave) ||
            !kvaf_fused_get::KvafP8ForwardControlMatchesIdentity(
                forwardControl_, wave->identity,
                kvaf_fused_get::kKvafP8ForwardPhaseDrain)) {
            if (wave != nullptr && kvaf_fused_get::IsKvafP8WaveRecordValid(*wave)) {
                kvaf_fused_get::QuarantineKvafP8WaveRecord(
                    wave, kvaf_fused_get::kKvafP8WaveOutcomeIdentityQuarantined);
                (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(
                    waveRecords_ + queue, *wave);
                DataSyncBarrier<MemDsbT::DDR>();
            }
            (void)MarkForwardAbort();
            return false;
        }

        // The exact no-post proof gives this DRAIN task exclusive authority
        // over these PENDING rows.  Each helper performs PENDING ->
        // RELEASE_CLAIMED -> verified source/object-map clear -> FREE; a
        // failed CAS quarantines rather than allowing a later wave to reuse a
        // partially released private row.
        const uint64_t ledger_base = LedgerOffset(queue, 0U);
        for (uint32_t request = 0U; request < wave->request_count; ++request) {
            const int32_t output_index = ReadGmByPassDCache<int32_t>(
                queueRequestOutputIndices_ + ledger_base + request);
            if (output_index < 0 || static_cast<uint32_t>(output_index) >=
                                   tiling_->privateRowCount ||
                !kvaf_fused_get::ReleaseKvafP8PendingPrivateRecordAfterNoPost(
                    privateRecords_ + output_index, forwardControl_,
                    wave->identity, privateSourceToSlot_, privateObjectToSlot_,
                    privateReady_, static_cast<uint64_t>(output_index),
                    privateReleaseResult_, simtEvent_)) {
                kvaf_fused_get::QuarantineKvafP8WaveRecord(
                    wave, kvaf_fused_get::kKvafP8WaveOutcomeIdentityQuarantined);
                (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(
                    waveRecords_ + queue, *wave);
                DataSyncBarrier<MemDsbT::DDR>();
                AddStat(kP8StatQuarantined, 1U);
                (void)MarkForwardAbort();
                return false;
            }
            AddStat(kP8StatPrivateReleased, 1U);
        }
        DataSyncBarrier<MemDsbT::DDR>();

        // Queue reuse is another irreversible ownership action.  Reload the
        // GM forward gate immediately before it; an independently published
        // ABORT turns this record into a terminal-sweep responsibility rather
        // than allowing a stale DRAIN worker to recycle a queue.
        if (!ReloadForwardControlForPhase(
                kvaf_fused_get::kKvafP8ForwardPhaseDrain, false) ||
            !kvaf_fused_get::ReleaseKvafP8ClaimedUnpostedWaveAfterPrivateCleanup(
                controlGm_, forwardControl_, privateRecords_,
                tiling_->privateRowCount,
                queueRequestSourceOffsets_ + ledger_base,
                queueRequestObjectOffsets_ + ledger_base,
                queueRequestOutputIndices_ + ledger_base,
                tiling_->requestCapacity, leaseResult_, simtEvent_, wave)) {
            if (kvaf_fused_get::IsKvafP8WaveRecordValid(*wave) &&
                wave->state != kvaf_fused_get::kKvafP8WaveQuarantined) {
                kvaf_fused_get::QuarantineKvafP8WaveRecord(
                    wave,
                    kvaf_fused_get::kKvafP8WaveOutcomeReleaseCasQuarantined);
            }
            (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(
                waveRecords_ + queue, *wave);
            DataSyncBarrier<MemDsbT::DDR>();
            AddStat(kP8StatQuarantined, 1U);
            (void)MarkForwardAbort();
            return false;
        }
        WriteGmByPassDCache<int32_t>(queueRequestCount_ + queue, 0);
        (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(
            waveRecords_ + queue, *wave);
        DataSyncBarrier<MemDsbT::DDR>();
        AddStat(kP8StatQueueReleased, 1U);
        return true;
    }

    __aicore__ inline void DrainOneQueue(uint32_t queue)
    {
        // This CAS is deliberately the *first* operation on a durable wave.
        // Loading a POSTED record before the claim recreates the same-owner
        // ABA bug: two dynamic AIVs could both observe it and later try to
        // finish/release it.  The GM state is the source of truth, not an
        // AIC-local hasWave flag.
        drainClaimResult_->success = 0U;
        drainClaimResult_->observed_state = kvaf_fused_get::kKvafP8WaveFree;
        const bool posted_claimed =
            kvaf_fused_get::TryClaimKvafP8WaveDrainFromAicore(
                waveRecords_ + queue, drainClaimResult_, simtEvent_);
        bool no_post_claimed = false;
        if (!posted_claimed) {
            drainClaimResult_->success = 0U;
            drainClaimResult_->observed_state = kvaf_fused_get::kKvafP8WaveFree;
            no_post_claimed =
                kvaf_fused_get::TryClaimKvafP8UnpostedWaveDrainFromAicore(
                    waveRecords_ + queue, drainClaimResult_, simtEvent_);
        }
        if (!posted_claimed && !no_post_claimed) {
            return;
        }
        AddStat(kP8StatCqPoll, 1U);

        kvaf_fused_get::KvafP8WaveRecord wave{};
        const bool loaded = kvaf_fused_get::LoadKvafP8WaveRecordFromGm(
            waveRecords_ + queue, &wave);
        const bool control_ready = ReloadForwardControlForPhase(
            kvaf_fused_get::kKvafP8ForwardPhaseDrain, false);
        if (!loaded || !control_ready ||
            !kvaf_fused_get::IsKvafP8WaveRecordValid(wave) ||
            wave.state != kvaf_fused_get::kKvafP8WaveDrainClaimed ||
            !SameForward(wave.identity) ||
            !kvaf_fused_get::KvafP8ForwardControlMatchesIdentity(
                forwardControl_, wave.identity,
                kvaf_fused_get::kKvafP8ForwardPhaseDrain)) {
            // A successful CAS followed by a mismatching re-load is an
            // identity violation, not a benign missed CQ.  Persist a durable
            // quarantine when there is a record to preserve, then disable the
            // private QSFA view for this forward.  We never manufacture a
            // local release after this point.
            if (loaded) {
                kvaf_fused_get::QuarantineKvafP8WaveRecord(
                    &wave, kvaf_fused_get::kKvafP8WaveOutcomeIdentityQuarantined);
                (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(
                    waveRecords_ + queue, wave);
                DataSyncBarrier<MemDsbT::DDR>();
            }
            MarkForwardAbort();
            return;
        }

        if (no_post_claimed) {
            (void)DrainDeterministicNoPostWave(queue, &wave);
            return;
        }

        // Nonblocking retire returns the durable state to POSTED when CQ is
        // still in flight.  Persist that state so a later graph drain phase,
        // potentially on another AIV, can claim it again.  A terminal CQ
        // keeps DRAIN_CLAIMED until Finish/private publication/release.
        (void)kvaf_fused_get::TryRetireKvafP8Wave(
            control_, forwardControl_, privateRecords_, tiling_->privateRowCount,
            &wave);
        if (wave.state == kvaf_fused_get::kKvafP8WavePosted) {
            (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(
                waveRecords_ + queue, wave);
            DataSyncBarrier<MemDsbT::DDR>();
            return;
        }
        if (wave.state == kvaf_fused_get::kKvafP8WaveQuarantined ||
            wave.state != kvaf_fused_get::kKvafP8WaveDrainClaimed) {
            (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(
                waveRecords_ + queue, wave);
            DataSyncBarrier<MemDsbT::DDR>();
            AddStat(kP8StatQuarantined, 1U);
            MarkForwardAbort();
            return;
        }

        (void)kvaf_fused_get::FinishKvafP8Wave(
            control_, forwardControl_, privateRecords_, tiling_->privateRowCount,
            &wave);
        if (wave.state == kvaf_fused_get::kKvafP8WavePosted) {
            // Finish may itself perform the nonblocking poll.  Honor the same
            // hand-off rule if that poll sees no terminal CQ.
            (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(
                waveRecords_ + queue, wave);
            DataSyncBarrier<MemDsbT::DDR>();
            return;
        }
        if (wave.state != kvaf_fused_get::kKvafP8WaveFinalized) {
            (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(
                waveRecords_ + queue, wave);
            DataSyncBarrier<MemDsbT::DDR>();
            AddStat(kP8StatQuarantined, 1U);
            MarkForwardAbort();
            return;
        }
        // Queue release is another ownership-changing operation.  Reload the
        // GM gate after Finish/private publication so a concurrent abort
        // cannot let this worker recycle a queue from the wrong forward.
        if (!ReloadForwardControlForPhase(
                kvaf_fused_get::kKvafP8ForwardPhaseDrain, false)) {
            kvaf_fused_get::QuarantineKvafP8WaveRecord(
                &wave, kvaf_fused_get::kKvafP8WaveOutcomeIdentityQuarantined);
            (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(
                waveRecords_ + queue, wave);
            DataSyncBarrier<MemDsbT::DDR>();
            (void)MarkForwardAbort();
            return;
        }
        if (!PublishTerminalPrivateRows(&wave)) {
            kvaf_fused_get::QuarantineKvafP8WaveRecord(
                &wave, kvaf_fused_get::kKvafP8WaveOutcomeIdentityQuarantined);
            (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(
                waveRecords_ + queue, wave);
            DataSyncBarrier<MemDsbT::DDR>();
            AddStat(kP8StatQuarantined, 1U);
            MarkForwardAbort();
            return;
        }
        (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(waveRecords_ + queue,
                                                         wave);
        DataSyncBarrier<MemDsbT::DDR>();
        const bool released = kvaf_fused_get::ReleaseKvafP8WaveAfterPrivateReady(
            controlGm_, forwardControl_, privateRecords_, tiling_->privateRowCount,
            leaseResult_, simtEvent_, &wave);
        if (released && wave.state == kvaf_fused_get::kKvafP8WaveFree) {
            WriteGmByPassDCache<int32_t>(queueRequestCount_ + queue, 0);
            AddStat(kP8StatQueueReleased, 1U);
            AddStat(kP8StatCqTerminal, 1U);
        } else {
            // A failed release CAS leaves the helper's record quarantined.
            // Keep that durable state and fail the complete forward closed;
            // terminal teardown owns cold recovery.
            if (wave.state != kvaf_fused_get::kKvafP8WaveQuarantined) {
                kvaf_fused_get::QuarantineKvafP8WaveRecord(
                    &wave,
                    kvaf_fused_get::kKvafP8WaveOutcomeReleaseCasQuarantined);
            }
            MarkForwardAbort();
        }
        (void)kvaf_fused_get::StoreKvafP8WaveRecordToGm(waveRecords_ + queue,
                                                         wave);
        DataSyncBarrier<MemDsbT::DDR>();
    }

    __aicore__ inline void RunDrain()
    {
        for (uint32_t queue = worker_; queue < tiling_->queueCount;
             queue += tiling_->workerCount) {
            DrainOneQueue(queue);
        }
    }

    __aicore__ inline void RunQsfaTerminalRelease()
    {
        // The per-layer QSFA terminal is allowed to recycle a private HBM
        // slot only after Exact Top-K *and* Guaranteed Recall are terminal.
        // A successful speculative CQ by itself is never sufficient proof
        // that QSFA consumed the exact input set.  Model-terminal teardown is
        // deliberately not on this hot path.
        if (!ReloadForwardControlForPhase(
                kvaf_fused_get::kKvafP8ForwardPhaseQsfaTerminal, false) ||
            !kvaf_fused_get::IsKvafP8PrivateViewReadyForQsfa(forwardControl_) ||
            (forwardControl_.flags &
             kvaf_fused_get::kKvafP8ForwardFlagQsfaTerminalObserved) == 0U) {
            AddStat(kP8StatControlRejected, 1U);
            return;
        }
        for (uint32_t slot = worker_; slot < tiling_->privateRowCount;
             slot += tiling_->workerCount) {
            kvaf_fused_get::KvafP8PrivateRecord record{};
            if (!kvaf_fused_get::LoadKvafP8PrivateRecordFromGm(
                    privateRecords_ + slot, &record) ||
                !SamePrivateForward(record) ||
                (record.state != kvaf_fused_get::kKvafP8PrivateReady &&
                 record.state != kvaf_fused_get::kKvafP8PrivateFailed)) {
                continue;
            }
            if (!kvaf_fused_get::ReleaseKvafP8PrivateRecordAfterQsfaTerminal(
                    privateRecords_ + slot, forwardControl_,
                    privateSourceToSlot_, privateObjectToSlot_, privateReady_,
                    slot, privateReleaseResult_, simtEvent_)) {
                // A second terminal graph invocation may observe a completed
                // FREE row after this worker's initial load.  That is safe;
                // every other outcome remains a fail-closed quarantine owned
                // by KVAF's verified RELEASE_CLAIMED protocol.
                kvaf_fused_get::KvafP8PrivateRecord after{};
                if (!kvaf_fused_get::LoadKvafP8PrivateRecordFromGm(
                        privateRecords_ + slot, &after) ||
                    !SamePrivateForward(after) ||
                    after.state != kvaf_fused_get::kKvafP8PrivateFree) {
                    (void)MarkForwardAbort();
                }
                continue;
            }
            AddStat(kP8StatPrivateReleased, 1U);
        }
        DataSyncBarrier<MemDsbT::DDR>();
    }

    __aicore__ inline void RunForwardTerminal()
    {
        // This is a bounded, graph-visible safety sweep.  It never performs a
        // CQ drain or queue release: POST_INTENT/POSTED/DRAIN_CLAIMED are
        // potentially in flight, so only cold recovery may return them to a
        // reusable pool.  The normal no-error path sees FREE records and
        // leaves the forward's private view untouched.
        if (!ReloadForwardControlForPhase(
                kvaf_fused_get::kKvafP8ForwardPhaseForwardTerminal, true)) {
            AddStat(kP8StatControlRejected, 1U);
            return;
        }
        bool abort_required = false;
        for (uint32_t queue = worker_; queue < tiling_->queueCount;
             queue += tiling_->workerCount) {
            kvaf_fused_get::KvafP8WaveRecord record{};
            if (!kvaf_fused_get::LoadKvafP8WaveRecordFromGm(
                    waveRecords_ + queue, &record) ||
                !kvaf_fused_get::IsKvafP8WaveRecordValid(record) ||
                !SameForward(record.identity)) {
                continue;
            }
            forwardTerminalResult_->success = 0U;
            forwardTerminalResult_->observed_state = record.state;
            if (!kvaf_fused_get::QuarantineKvafP8WaveAtForwardTerminal(
                    waveRecords_ + queue, forwardControl_,
                    forwardTerminalResult_, simtEvent_)) {
                // We own a matching record but cannot prove its transition;
                // fail closed without manufacturing a local release.
                abort_required = true;
                continue;
            }
            if (forwardTerminalResult_->observed_state !=
                    kvaf_fused_get::kKvafP8WaveFree &&
                forwardTerminalResult_->observed_state !=
                    kvaf_fused_get::kKvafP8WaveQuarantined) {
                abort_required = true;
                AddStat(kP8StatForwardTerminalWave, 1U);
            }
        }
        for (uint32_t slot = worker_; slot < tiling_->privateRowCount;
             slot += tiling_->workerCount) {
            kvaf_fused_get::KvafP8PrivateRecord record{};
            if (!kvaf_fused_get::LoadKvafP8PrivateRecordFromGm(
                    privateRecords_ + slot, &record) ||
                !kvaf_fused_get::IsKvafP8PrivateRecordValid(record) ||
                !SamePrivateForward(record)) {
                continue;
            }
            forwardTerminalResult_->success = 0U;
            forwardTerminalResult_->observed_state = record.state;
            if (!kvaf_fused_get::QuarantineKvafP8PrivateAtForwardTerminal(
                    privateRecords_ + slot, forwardControl_, privateReady_, slot,
                    forwardTerminalResult_, simtEvent_)) {
                abort_required = true;
                continue;
            }
            if (forwardTerminalResult_->observed_state !=
                    kvaf_fused_get::kKvafP8PrivateFree &&
                forwardTerminalResult_->observed_state !=
                    kvaf_fused_get::kKvafP8PrivateQuarantined) {
                abort_required = true;
                AddStat(kP8StatForwardTerminalPrivate, 1U);
            }
        }
        if (abort_required) {
            (void)MarkForwardAbort();
        }
        DataSyncBarrier<MemDsbT::DDR>();
    }

    TPipe* pipe_ = nullptr;
    const A5P8ForwardOwnedExecutorTilingData* tiling_ = nullptr;
    TBuf<TPosition::VECCALC> leaseBuffer_;
    TBuf<TPosition::VECCALC> claimBuffer_;
    TBuf<TPosition::VECCALC> drainClaimBuffer_;
    TBuf<TPosition::VECCALC> abortBuffer_;
    TBuf<TPosition::VECCALC> privateReleaseBuffer_;
    TBuf<TPosition::VECCALC> forwardTerminalBuffer_;
    __ubuf__ kvaf_fused_get::KvafStreamingBatchGetLeaseResult* leaseResult_ = nullptr;
    __ubuf__ P8PrivateClaimResult* claimResult_ = nullptr;
    __ubuf__ kvaf_fused_get::KvafP8WaveDrainClaimResult* drainClaimResult_ = nullptr;
    __ubuf__ kvaf_fused_get::KvafP8ForwardAbortResult* abortResult_ = nullptr;
    __ubuf__ kvaf_fused_get::KvafP8PrivateReleaseResult* privateReleaseResult_ = nullptr;
    __ubuf__ kvaf_fused_get::KvafP8ForwardTerminalResult* forwardTerminalResult_ = nullptr;
    event_t simtEvent_ = static_cast<event_t>(EVENT_ID0);

    __gm__ int32_t* candidateSeeds_ = nullptr;
    __gm__ int32_t* candidateCounts_ = nullptr;
    __gm__ int64_t* candidateReadyEpoch_ = nullptr;
    __gm__ int32_t* candidateStatus_ = nullptr;
    __gm__ kvaf_fused_get::KvafP8ForwardControl* forwardControlGm_ = nullptr;
    __gm__ int32_t* sourceLens_ = nullptr;
    __gm__ int32_t* kvObjectAddressDomain_ = nullptr;
    __gm__ int32_t* reqPoolEntries_ = nullptr;
    __gm__ int32_t* kvObjectMap_ = nullptr;
    __gm__ int32_t* kvObjectLens_ = nullptr;
    __gm__ int32_t* kvObjectSlotsSnapshot_ = nullptr;
    __gm__ int32_t* privateSourceToSlot_ = nullptr;
    __gm__ int32_t* privateObjectToSlot_ = nullptr;
    __gm__ int32_t* privateReady_ = nullptr;
    __gm__ int32_t* privateSlotCursor_ = nullptr;
    __gm__ uint8_t* privatePackedOutput_ = nullptr;
    __gm__ const kvaf_fused_get::KvafStreamingBatchGetControl* controlGm_ = nullptr;
    __gm__ kvaf_fused_get::KvafP8WaveRecord* waveRecords_ = nullptr;
    __gm__ kvaf_fused_get::KvafP8PrivateRecord* privateRecords_ = nullptr;
    __gm__ int64_t* queueRequestSourceOffsets_ = nullptr;
    __gm__ int64_t* queueRequestObjectOffsets_ = nullptr;
    __gm__ int32_t* queueRequestOutputIndices_ = nullptr;
    __gm__ int32_t* queueRequestCount_ = nullptr;
    __gm__ int32_t* candidateCursor_ = nullptr;
    __gm__ int64_t* candidateCursorEpoch_ = nullptr;
    __gm__ int32_t* rowsPerEntry_ = nullptr;
    __gm__ int64_t* workerStats_ = nullptr;

    kvaf_fused_get::KvafStreamingBatchGetControl control_{};
    kvaf_fused_get::KvafP8ForwardControl forwardControl_{};
    uint32_t worker_ = 0U;
    uint64_t forwardEpoch_ = 0U;
    uint64_t replayGeneration_ = 0U;
    uint32_t layerControlIndex_ = 0U;
    uint64_t phase_ = 0U;
    bool enabled_ = false;
};
}  // namespace

extern "C" __global__ __aicore__ void a5_p8_forward_owned_executor(
    GM_ADDR candidate_seeds, GM_ADDR candidate_counts,
    GM_ADDR candidate_ready_epoch, GM_ADDR candidate_status,
    GM_ADDR forward_control, GM_ADDR source_lens,
    GM_ADDR kv_object_address_domain, GM_ADDR req_pool_entries,
    GM_ADDR kv_object_map, GM_ADDR kv_object_lens,
    GM_ADDR kv_object_slots_snapshot, GM_ADDR private_source_to_slot,
    GM_ADDR private_object_to_slot, GM_ADDR private_ready_table,
    GM_ADDR private_slot_cursor, GM_ADDR private_packed_output,
    GM_ADDR fused_get_control, GM_ADDR p8_wave_records,
    GM_ADDR p8_private_records, GM_ADDR queue_request_source_offsets,
    GM_ADDR queue_request_object_offsets, GM_ADDR queue_request_output_indices,
    GM_ADDR queue_request_count, GM_ADDR candidate_cursor,
    GM_ADDR candidate_cursor_epoch, GM_ADDR rows_per_entry,
    GM_ADDR worker_stats,
    GM_ADDR forward_control_ref, GM_ADDR private_source_to_slot_ref,
    GM_ADDR private_object_to_slot_ref,
    GM_ADDR private_ready_table_ref, GM_ADDR private_slot_cursor_ref,
    GM_ADDR private_packed_output_ref, GM_ADDR fused_get_control_ref,
    GM_ADDR p8_wave_records_ref, GM_ADDR p8_private_records_ref,
    GM_ADDR queue_request_source_offsets_ref,
    GM_ADDR queue_request_object_offsets_ref,
    GM_ADDR queue_request_output_indices_ref, GM_ADDR queue_request_count_ref,
    GM_ADDR candidate_cursor_ref, GM_ADDR candidate_cursor_epoch_ref,
    GM_ADDR worker_stats_ref, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)forward_control_ref;
    (void)private_source_to_slot_ref;
    (void)private_object_to_slot_ref;
    (void)private_ready_table_ref;
    (void)private_slot_cursor_ref;
    (void)private_packed_output_ref;
    (void)fused_get_control_ref;
    (void)p8_wave_records_ref;
    (void)p8_private_records_ref;
    (void)queue_request_source_offsets_ref;
    (void)queue_request_object_offsets_ref;
    (void)queue_request_output_indices_ref;
    (void)queue_request_count_ref;
    (void)candidate_cursor_ref;
    (void)candidate_cursor_epoch_ref;
    (void)worker_stats_ref;
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5P8ForwardOwnedExecutorTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    A5P8ForwardOwnedExecutorKernel op(&pipe, &tilingData);
    op.Init(candidate_seeds, candidate_counts, candidate_ready_epoch,
            candidate_status, forward_control, source_lens,
            kv_object_address_domain, req_pool_entries, kv_object_map,
            kv_object_lens, kv_object_slots_snapshot, private_source_to_slot,
            private_object_to_slot, private_ready_table, private_slot_cursor,
            private_packed_output, fused_get_control, p8_wave_records,
            p8_private_records, queue_request_source_offsets,
            queue_request_object_offsets, queue_request_output_indices,
            queue_request_count, candidate_cursor, candidate_cursor_epoch,
            rows_per_entry, worker_stats);
    op.Process();
}
