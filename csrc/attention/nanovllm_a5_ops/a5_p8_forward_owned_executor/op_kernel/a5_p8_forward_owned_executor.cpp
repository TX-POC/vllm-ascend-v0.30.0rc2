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

#include "a5_p8_arena_tag.h"
#include "a5_p8_forward_owned_executor_tiling.h"
#include "kvaf_device/kvaf_fused_full_get_device.h"

using namespace AscendC;

namespace {
constexpr int32_t kUnmapped = -1;
constexpr int32_t kClaiming = -2;
constexpr int32_t kCandidateStatusOk = 1;
constexpr int32_t kCandidateStatusTombstoned = -6;
constexpr int32_t kMappedObjectAddressDomain = 1;
// A graph-stable tagged scalar shared by every P8 layer that reuses this
// bounded queue/private arena.  Its generation is an ABA guard, not a second
// ownership identity.  The graph reuse fence requires a later forward to
// wait for the prior FORWARD_TERMINAL node before reinitializing *any* P8 GM
// tensor (control, receipts/cursors, waves, private records or maps).
constexpr uint32_t kPackedKvRowBytes = 656U;
// The durable pre-Post state is supplied by KVAF as append-only
// `kKvafP8WaveFilling`.  Do not overload POST_INTENT here: POST_INTENT means
// that the descriptor is complete and can be observed by PostOnly, whereas a
// FILLING record means that a queue lease is owned but no descriptor/private
// record may be treated as postable yet.

// Stage A publishes this complete receipt identity with generation last.
// Epoch alone cannot close ABA across a graph replay or a later layer that
// reuses the same bounded receipt/cursor arrays.
struct P8CandidateReceiptIdentity {
    int64_t forward_epoch;
    int64_t graph_replay_generation;
    int64_t layer_control_index;
    int64_t receipt_generation;
};

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
    int32_t quarantined;
    int32_t reserved0;
};
static_assert(sizeof(P8PrivateClaimResult) == 32U,
              "P8 global-private claim ABI changed");

using P8ArenaLeaseResult = a5_p8_arena::ArenaResult;

__simt_callee__ inline void ClaimP8GlobalPrivateSlotFromSimt(
    __gm__ int32_t* private_slot_cursor,
    __gm__ kvaf_fused_get::KvafP8PrivateRecord* private_records,
    uint32_t private_row_count, uint64_t source_offset, uint64_t object_offset,
    uint64_t forward_epoch, uint64_t replay_generation,
    uint32_t layer_control_index, uint32_t queue_id,
    uint64_t queue_generation, uint64_t post_sequence, int32_t owner_tag,
    __ubuf__ P8PrivateClaimResult* result)
{
    if (private_slot_cursor == nullptr || private_records == nullptr ||
        result == nullptr ||
        private_row_count == 0U || forward_epoch == 0U ||
        queue_generation == 0U || post_sequence == 0U || owner_tag <= 0) {
        return;
    }
    result->source_offset = source_offset;
    result->object_offset = object_offset;
    result->output_index = kUnmapped;
    result->success = 0;
    result->quarantined = 0;
    result->reserved0 = 0;

    // Reserve a private physical row *before* touching the source/object
    // maps.  FREE -> ALLOCATING is durable but non-consumable: terminal can
    // quarantine it even while identity is being written.  In particular, no
    // map can be stranded at -2 before the terminal has a bounded row that it
    // can inspect and seal.
    uint32_t selected = private_row_count;
    for (uint32_t attempt = 0U; attempt < private_row_count; ++attempt) {
        const uint32_t cursor = asc_atomic_add(private_slot_cursor, 1U);
        const uint32_t slot = cursor % private_row_count;
        __gm__ kvaf_fused_get::KvafP8PrivateRecord* const record =
            private_records + slot;
        kvaf_fused_get::KvafP8WaveIdentity identity{};
        identity.forward_epoch = forward_epoch;
        identity.graph_replay_generation = replay_generation;
        identity.layer_control_index = layer_control_index;
        identity.queue_id = queue_id;
        identity.queue_generation = queue_generation;
        identity.post_sequence = post_sequence;
        identity.owner_tag = owner_tag;
        identity.reserved0 = 0U;
        if (kvaf_fused_get::BeginKvafP8PrivateAllocatingPayloadFromSimt(
                record, identity, source_offset, object_offset,
                static_cast<int32_t>(slot))) {
            selected = slot;
            break;
        }
    }
    if (selected == private_row_count) {
        return;
    }

    // The immutable payload is durable in ALLOCATING.  ClaimGlobalPrivate
    // executes its AICore DDR publication barrier before the map service
    // below may expose either directory entry.
    result->output_index = static_cast<int32_t>(selected);
    result->success = 1;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void ClaimP8GlobalPrivateSlotSimt(
    __gm__ int32_t* private_slot_cursor,
    __gm__ kvaf_fused_get::KvafP8PrivateRecord* private_records,
    uint32_t private_row_count, uint64_t source_offset, uint64_t object_offset,
    uint64_t forward_epoch, uint64_t replay_generation,
    uint32_t layer_control_index, uint32_t queue_id,
    uint64_t queue_generation, uint64_t post_sequence, int32_t owner_tag,
    __ubuf__ P8PrivateClaimResult* result)
{
    ClaimP8GlobalPrivateSlotFromSimt(
        private_slot_cursor, private_records, private_row_count, source_offset,
        object_offset, forward_epoch, replay_generation, layer_control_index,
        queue_id, queue_generation, post_sequence, owner_tag, result);
}

__simt_callee__ inline void CommitP8GlobalPrivateSlotMapsFromSimt(
    __gm__ int32_t* source_to_slot, __gm__ int32_t* object_to_slot,
    __gm__ int32_t* private_ready,
    __gm__ kvaf_fused_get::KvafP8PrivateRecord* private_records,
    uint32_t private_row_count, uint64_t source_offset,
    uint64_t object_offset, uint32_t selected,
    __ubuf__ P8PrivateClaimResult* result)
{
    if (source_to_slot == nullptr || object_to_slot == nullptr ||
        private_ready == nullptr || private_records == nullptr ||
        result == nullptr || selected >= private_row_count) {
        return;
    }
    result->source_offset = source_offset;
    result->object_offset = object_offset;
    result->output_index = static_cast<int32_t>(selected);
    result->success = 0;
    result->quarantined = 0;
    result->reserved0 = 0;
    __gm__ kvaf_fused_get::KvafP8PrivateRecord* const record =
        private_records + selected;

    // A transient `-2` is now paired with a durable ALLOCATING record.  If a
    // terminal races an interrupted producer it can quarantine the row and
    // roll back only `-2`; a positive map is never silently recycled.
    if (asc_atomic_cas(source_to_slot + source_offset, kUnmapped,
                       kClaiming) != kUnmapped ||
        asc_atomic_cas(object_to_slot + object_offset, kUnmapped,
                       kClaiming) != kUnmapped) {
        (void)kvaf_fused_get::TryQuarantineKvafP8PrivateAllocatingFromSimt(
            record);
        private_ready[selected] = static_cast<int32_t>(
            kvaf_fused_get::kKvafP8PrivateQuarantined);
        result->quarantined = 1;
        asc_atomic_cas(source_to_slot + source_offset, kClaiming, kUnmapped);
        asc_atomic_cas(object_to_slot + object_offset, kClaiming, kUnmapped);
        return;
    }

    // Make the producer the sole owner of the positive-map critical section.
    // Terminal may still seal ALLOCATING before this CAS, in which case only
    // our own -2 claims are rolled back.  Once MAP_PUBLISHING is durable,
    // terminal deliberately latches arena health rather than racing either
    // directory; an interruption stays bounded and unreusable.
    if (!kvaf_fused_get::TryClaimKvafP8PrivateMapPublishingFromSimt(record)) {
        asc_atomic_cas(source_to_slot + source_offset, kClaiming, kUnmapped);
        asc_atomic_cas(object_to_slot + object_offset, kClaiming, kUnmapped);
        result->quarantined = 1;
        return;
    }

    // MAP_PUBLISHING has no terminal mutator.  The two exact CAS operations
    // are therefore the only writers capable of replacing our -2 markers.
    if (asc_atomic_cas(source_to_slot + source_offset, kClaiming,
                       static_cast<int32_t>(selected)) != kClaiming ||
        asc_atomic_cas(object_to_slot + object_offset, kClaiming,
                       static_cast<int32_t>(selected)) != kClaiming ||
        !kvaf_fused_get::TryPromoteKvafP8PrivateMapPublishingToPendingFromSimt(
            record)) {
        // Do not roll back a positive map and do not full-store the record:
        // this is an uncertain bounded resource that ForwardTerminal/cold
        // recovery owns.  The caller latches the tagged arena unhealthy.
        result->quarantined = 1;
        return;
    }
    private_ready[selected] = static_cast<int32_t>(
        kvaf_fused_get::kKvafP8PrivatePending);
    AscendC::Simt::ThreadFence();
    result->success = 1;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void CommitP8GlobalPrivateSlotMapsSimt(
    __gm__ int32_t* source_to_slot, __gm__ int32_t* object_to_slot,
    __gm__ int32_t* private_ready,
    __gm__ kvaf_fused_get::KvafP8PrivateRecord* private_records,
    uint32_t private_row_count, uint64_t source_offset,
    uint64_t object_offset, uint32_t selected,
    __ubuf__ P8PrivateClaimResult* result)
{
    CommitP8GlobalPrivateSlotMapsFromSimt(
        source_to_slot, object_to_slot, private_ready, private_records,
        private_row_count, source_offset, object_offset, selected, result);
}

class A5P8ForwardOwnedExecutorKernel {
public:
    __aicore__ inline explicit A5P8ForwardOwnedExecutorKernel(
        TPipe* pipe, const A5P8ForwardOwnedExecutorTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR candidate_seeds, GM_ADDR candidate_counts,
        GM_ADDR candidate_receipt_identity, GM_ADDR candidate_status,
        GM_ADDR forward_control, GM_ADDR p8_arena_health, GM_ADDR source_lens,
        GM_ADDR kv_object_address_domain, GM_ADDR req_pool_entries,
        GM_ADDR kv_object_map, GM_ADDR kv_object_lens,
        GM_ADDR kv_object_slots_snapshot, GM_ADDR private_source_to_slot,
        GM_ADDR private_object_to_slot, GM_ADDR private_ready_table,
        GM_ADDR private_slot_cursor, GM_ADDR private_packed_output,
        GM_ADDR fused_get_control, GM_ADDR p8_wave_records,
        GM_ADDR p8_private_records, GM_ADDR queue_request_source_offsets,
        GM_ADDR queue_request_object_offsets,
        GM_ADDR queue_request_output_indices, GM_ADDR queue_request_count,
        GM_ADDR candidate_cursor, GM_ADDR candidate_cursor_identity,
        GM_ADDR rows_per_entry, GM_ADDR worker_stats)
    {
        candidateSeeds_ = reinterpret_cast<__gm__ int32_t*>(candidate_seeds);
        candidateCounts_ = reinterpret_cast<__gm__ int32_t*>(candidate_counts);
        candidateReceiptIdentity_ = reinterpret_cast<__gm__ int64_t*>(
            candidate_receipt_identity);
        candidateStatus_ = reinterpret_cast<__gm__ int32_t*>(candidate_status);
        forwardControlGm_ = reinterpret_cast<__gm__
            kvaf_fused_get::KvafP8ForwardControl*>(forward_control);
        arenaHealthGm_ = reinterpret_cast<__gm__ int32_t*>(p8_arena_health);
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
        candidateCursorIdentity_ = reinterpret_cast<__gm__ int64_t*>(
            candidate_cursor_identity);
        rowsPerEntry_ = reinterpret_cast<__gm__ int32_t*>(rows_per_entry);
        workerStats_ = reinterpret_cast<__gm__ int64_t*>(worker_stats);
        worker_ = GetBlockIdx();
        enabled_ = false;

        if (pipe_ == nullptr || tiling_ == nullptr ||
            worker_ >= tiling_->workerCount ||
            candidateSeeds_ == nullptr || candidateCounts_ == nullptr ||
            candidateReceiptIdentity_ == nullptr || candidateStatus_ == nullptr ||
            forwardControlGm_ == nullptr || arenaHealthGm_ == nullptr ||
            sourceLens_ == nullptr ||
            kvObjectAddressDomain_ == nullptr || reqPoolEntries_ == nullptr ||
            kvObjectMap_ == nullptr || kvObjectLens_ == nullptr ||
            kvObjectSlotsSnapshot_ == nullptr || privateSourceToSlot_ == nullptr ||
            privateObjectToSlot_ == nullptr || privateReady_ == nullptr ||
            privateSlotCursor_ == nullptr || privatePackedOutput_ == nullptr ||
            controlGm_ == nullptr || waveRecords_ == nullptr ||
            privateRecords_ == nullptr || queueRequestSourceOffsets_ == nullptr ||
            queueRequestObjectOffsets_ == nullptr ||
            queueRequestOutputIndices_ == nullptr || queueRequestCount_ == nullptr ||
            candidateCursor_ == nullptr || candidateCursorIdentity_ == nullptr ||
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
        // Admission stores the single tagged arena generation in the high
        // word of the existing graph-stable control record.  Every P8 phase
        // reuses this cached token; it is the ABA guard for the fixed arena,
        // not an expanded ownership ABI.
        arenaLeaseGeneration_ =
            kvaf_fused_get::GetKvafP8ForwardLeaseGeneration(forwardControl_);
        if (arenaLeaseGeneration_ == 0U ||
            arenaLeaseGeneration_ > a5_p8_arena::kArenaGenerationMax) {
            AddStat(kP8StatControlRejected, 1U);
            return;
        }
        arenaActiveTag_ = a5_p8_arena::MakeTag(
            arenaLeaseGeneration_, a5_p8_arena::kArenaActive);
        arenaTerminalTag_ = a5_p8_arena::MakeTag(
            arenaLeaseGeneration_, a5_p8_arena::kArenaTerminalClaimed);
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
        pipe_->InitBuffer(arenaLeaseBuffer_, sizeof(P8ArenaLeaseResult));
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
        arenaLeaseResult_ = (__ubuf__ P8ArenaLeaseResult*)
            arenaLeaseBuffer_.Get<uint8_t>().GetPhyAddr();
        if (leaseResult_ == nullptr || claimResult_ == nullptr ||
            drainClaimResult_ == nullptr || abortResult_ == nullptr ||
            privateReleaseResult_ == nullptr || forwardTerminalResult_ == nullptr ||
            arenaLeaseResult_ == nullptr) {
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

    __aicore__ inline __gm__ int64_t* CandidateReceiptIdentity(
        uint32_t receipt) const
    {
        return candidateReceiptIdentity_ +
            static_cast<uint64_t>(receipt) *
                kA5P8CandidateReceiptIdentityWords;
    }

    __aicore__ inline __gm__ int64_t* CandidateCursorIdentity(
        uint32_t receipt) const
    {
        return candidateCursorIdentity_ +
            static_cast<uint64_t>(receipt) *
                kA5P8CandidateReceiptIdentityWords;
    }

    __aicore__ inline bool LoadStableReceiptIdentity(
        __gm__ int64_t* base, P8CandidateReceiptIdentity* identity) const
    {
        if (base == nullptr || identity == nullptr) {
            return false;
        }
        const int64_t generation_before =
            ReadGmByPassDCache<int64_t>(base + 3U);
        if (generation_before <= 0) {
            return false;
        }
        identity->forward_epoch = ReadGmByPassDCache<int64_t>(base + 0U);
        identity->graph_replay_generation =
            ReadGmByPassDCache<int64_t>(base + 1U);
        identity->layer_control_index =
            ReadGmByPassDCache<int64_t>(base + 2U);
        const int64_t generation_after =
            ReadGmByPassDCache<int64_t>(base + 3U);
        identity->receipt_generation = generation_after;
        return generation_before == generation_after && generation_after > 0;
    }

    __aicore__ inline bool SameReceiptIdentity(
        const P8CandidateReceiptIdentity& lhs,
        const P8CandidateReceiptIdentity& rhs) const
    {
        return lhs.forward_epoch == rhs.forward_epoch &&
            lhs.graph_replay_generation == rhs.graph_replay_generation &&
            lhs.layer_control_index == rhs.layer_control_index &&
            lhs.receipt_generation == rhs.receipt_generation;
    }

    __aicore__ inline bool IsCurrentReceiptIdentity(
        const P8CandidateReceiptIdentity& identity) const
    {
        return identity.forward_epoch == static_cast<int64_t>(forwardEpoch_) &&
            identity.graph_replay_generation ==
                static_cast<int64_t>(replayGeneration_) &&
            identity.layer_control_index ==
                static_cast<int64_t>(layerControlIndex_) &&
            identity.receipt_generation > 0;
    }

    // Read status between two stable identity samples.  Terminal/replay
    // invalidates generation first, so this rejects a receipt that changed
    // while this AIV was deciding whether it could issue a PostOnly.
    __aicore__ inline bool LoadCurrentCandidateReceipt(
        uint32_t receipt, P8CandidateReceiptIdentity* identity) const
    {
        P8CandidateReceiptIdentity before{};
        P8CandidateReceiptIdentity after{};
        if (!LoadStableReceiptIdentity(CandidateReceiptIdentity(receipt),
                                       &before) ||
            !IsCurrentReceiptIdentity(before) ||
            ReadGmByPassDCache<int32_t>(candidateStatus_ + receipt) !=
                kCandidateStatusOk ||
            !LoadStableReceiptIdentity(CandidateReceiptIdentity(receipt),
                                       &after) ||
            !SameReceiptIdentity(before, after)) {
            return false;
        }
        if (identity != nullptr) {
            *identity = after;
        }
        return true;
    }

    __aicore__ inline void TombstoneReceiptIdentity(
        __gm__ int64_t* base) const
    {
        if (base == nullptr) {
            return;
        }
        // Generation is the sole release word.  Make it zero before touching
        // the prefix, then wait for visibility before any reuse can publish a
        // new identity.
        WriteGmByPassDCache<int64_t>(base + 3U, 0);
        DataSyncBarrier<MemDsbT::DDR>();
        WriteGmByPassDCache<int64_t>(base + 0U, 0);
        WriteGmByPassDCache<int64_t>(base + 1U, 0);
        WriteGmByPassDCache<int64_t>(base + 2U, 0);
        DataSyncBarrier<MemDsbT::DDR>();
    }

    __aicore__ inline void PublishCursorReceiptIdentity(
        uint32_t receipt, const P8CandidateReceiptIdentity& identity) const
    {
        __gm__ int64_t* const cursor_identity = CandidateCursorIdentity(receipt);
        TombstoneReceiptIdentity(cursor_identity);
        WriteGmByPassDCache<int64_t>(cursor_identity + 0U,
                                     identity.forward_epoch);
        WriteGmByPassDCache<int64_t>(cursor_identity + 1U,
                                     identity.graph_replay_generation);
        WriteGmByPassDCache<int64_t>(cursor_identity + 2U,
                                     identity.layer_control_index);
        DataSyncBarrier<MemDsbT::DDR>();
        WriteGmByPassDCache<int64_t>(cursor_identity + 3U,
                                     identity.receipt_generation);
        DataSyncBarrier<MemDsbT::DDR>();
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

    __aicore__ inline bool IsP8PrivateRecordAllZero(
        const kvaf_fused_get::KvafP8PrivateRecord& record) const
    {
        return record.magic == 0U && record.version == 0U &&
            record.state == 0U && record.forward_epoch == 0U &&
            record.graph_replay_generation == 0U &&
            record.layer_control_index == 0U && record.queue_id == 0U &&
            record.queue_generation == 0U && record.post_sequence == 0U &&
            record.source_offset == 0U && record.object_offset == 0U &&
            record.output_index == 0 && record.owner_tag == 0;
    }

    __aicore__ inline bool ClaimGlobalPrivate(
        uint64_t source_offset, uint64_t object_offset,
        const kvaf_fused_get::KvafP8WaveIdentity& identity)
    {
        claimResult_->success = 0;
        set_flag(PIPE_S, PIPE_V, simtEvent_);
        wait_flag(PIPE_S, PIPE_V, simtEvent_);
        AscendC::Simt::VF_CALL<ClaimP8GlobalPrivateSlotSimt>(
            AscendC::Simt::Dim3{1U, 1U, 1U}, privateSlotCursor_,
            privateRecords_, tiling_->privateRowCount, source_offset,
            object_offset, identity.forward_epoch,
            identity.graph_replay_generation, identity.layer_control_index,
            identity.queue_id, identity.queue_generation,
            identity.post_sequence, identity.owner_tag, claimResult_);
        set_flag(PIPE_V, PIPE_S, simtEvent_);
        wait_flag(PIPE_V, PIPE_S, simtEvent_);
        if (claimResult_->success == 0) {
            return false;
        }

        // Cross the only mixed-domain publication edge here.  The SIMT
        // reservation wrote immutable payload while state is ALLOCATING;
        // DDR visibility is required before the next SIMT VF may expose
        // either source/object directory entry or promote PENDING.
        DataSyncBarrier<MemDsbT::DDR>();
        const uint32_t selected = static_cast<uint32_t>(
            claimResult_->output_index);
        set_flag(PIPE_S, PIPE_V, simtEvent_);
        wait_flag(PIPE_S, PIPE_V, simtEvent_);
        AscendC::Simt::VF_CALL<CommitP8GlobalPrivateSlotMapsSimt>(
            AscendC::Simt::Dim3{1U, 1U, 1U}, privateSourceToSlot_,
            privateObjectToSlot_, privateReady_, privateRecords_,
            tiling_->privateRowCount, source_offset, object_offset, selected,
            claimResult_);
        set_flag(PIPE_V, PIPE_S, simtEvent_);
        wait_flag(PIPE_V, PIPE_S, simtEvent_);
        if (claimResult_->quarantined != 0) {
            // A partial map/private-record publication is persistent unsafe
            // evidence, not ordinary capacity contention.  Latch the arena
            // before the caller can consider another candidate or PostOnly.
            AddStat(kP8StatQuarantined, 1U);
            (void)MarkForwardAbort();
        }
        return claimResult_->success != 0;
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
            reloaded.phase != expected_phase ||
            kvaf_fused_get::GetKvafP8ForwardLeaseGeneration(reloaded) !=
                arenaLeaseGeneration_) {
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

    __aicore__ inline bool HasP8ArenaAdmissionLease() const
    {
        // A full tagged equality rejects a delayed node after another forward
        // acquired this fixed arena.  It is a lease check, never a fixed AIV
        // assignment.
        return arenaHealthGm_ != nullptr &&
            static_cast<uint32_t>(ReadGmByPassDCache<int32_t>(arenaHealthGm_))
                == arenaActiveTag_;
    }

    __aicore__ inline bool PoisonP8ArenaActive()
    {
        if (arenaHealthGm_ == nullptr || arenaLeaseResult_ == nullptr) {
            return false;
        }
        arenaLeaseResult_->success = 0U;
        arenaLeaseResult_->observed_tag = 0U;
        set_flag(PIPE_S, PIPE_V, simtEvent_);
        wait_flag(PIPE_S, PIPE_V, simtEvent_);
        AscendC::Simt::VF_CALL<a5_p8_arena::PoisonActiveLeaseSimtService>(
            AscendC::Simt::Dim3{1U, 1U, 1U}, arenaHealthGm_, arenaActiveTag_,
            arenaLeaseResult_);
        set_flag(PIPE_V, PIPE_S, simtEvent_);
        wait_flag(PIPE_V, PIPE_S, simtEvent_);
        DataSyncBarrier<MemDsbT::DDR>();
        return arenaLeaseResult_->success != 0U;
    }

    __aicore__ inline bool CompleteP8ArenaTerminal(bool release_clean)
    {
        if (arenaHealthGm_ == nullptr || arenaLeaseResult_ == nullptr) {
            return false;
        }
        arenaLeaseResult_->success = 0U;
        arenaLeaseResult_->observed_tag = 0U;
        set_flag(PIPE_S, PIPE_V, simtEvent_);
        wait_flag(PIPE_S, PIPE_V, simtEvent_);
        AscendC::Simt::VF_CALL<a5_p8_arena::CompleteTerminalSimtService>(
            AscendC::Simt::Dim3{1U, 1U, 1U}, arenaHealthGm_, arenaTerminalTag_,
            release_clean, arenaLeaseResult_);
        set_flag(PIPE_V, PIPE_S, simtEvent_);
        wait_flag(PIPE_V, PIPE_S, simtEvent_);
        DataSyncBarrier<MemDsbT::DDR>();
        return arenaLeaseResult_->success != 0U;
    }

    __aicore__ inline bool LatchP8ArenaUnhealthy()
    {
        const uint32_t unhealthy_tag = a5_p8_arena::MakeTag(
            arenaLeaseGeneration_, a5_p8_arena::kArenaUnhealthy);
        if (arenaHealthGm_ != nullptr &&
            static_cast<uint32_t>(ReadGmByPassDCache<int32_t>(arenaHealthGm_))
                == unhealthy_tag) {
            return true;
        }
        return PoisonP8ArenaActive();
    }

    __aicore__ inline bool ReleaseP8ArenaAfterCleanForwardTerminal()
    {
        return CompleteP8ArenaTerminal(true);
    }

    __aicore__ inline bool ClaimP8ArenaForForwardTerminal(
        bool* entered_unhealthy)
    {
        if (entered_unhealthy == nullptr || arenaHealthGm_ == nullptr ||
            arenaLeaseResult_ == nullptr) {
            return false;
        }
        *entered_unhealthy = false;
        arenaLeaseResult_->success = 0U;
        arenaLeaseResult_->observed_tag = 0U;
        set_flag(PIPE_S, PIPE_V, simtEvent_);
        wait_flag(PIPE_S, PIPE_V, simtEvent_);
        AscendC::Simt::VF_CALL<a5_p8_arena::ClaimTerminalSimtService>(
            AscendC::Simt::Dim3{1U, 1U, 1U}, arenaHealthGm_,
            arenaLeaseGeneration_, arenaLeaseResult_);
        set_flag(PIPE_V, PIPE_S, simtEvent_);
        wait_flag(PIPE_V, PIPE_S, simtEvent_);
        DataSyncBarrier<MemDsbT::DDR>();
        if (arenaLeaseResult_->success == 0U) {
            return false;
        }
        *entered_unhealthy = a5_p8_arena::StateOf(
            arenaLeaseResult_->observed_tag) == a5_p8_arena::kArenaUnhealthy;
        return true;
    }

    __aicore__ inline bool MarkForwardAbortAfterTerminalClaim()
    {
        if (!ReloadForwardControlForPhase(
                kvaf_fused_get::kKvafP8ForwardPhaseForwardTerminal, true)) {
            return false;
        }
        abortResult_->success = 0U;
        abortResult_->observed_flags_low = 0U;
        if (!kvaf_fused_get::MarkKvafP8ForwardAbortFromAicore(
                forwardControlGm_, abortResult_, simtEvent_)) {
            return false;
        }
        AddStat(kP8StatAbortPublished, 1U);
        const uint32_t unhealthy_tag = a5_p8_arena::MakeTag(
            arenaLeaseGeneration_, a5_p8_arena::kArenaUnhealthy);
        if (static_cast<uint32_t>(ReadGmByPassDCache<int32_t>(arenaHealthGm_))
            == unhealthy_tag) {
            return true;
        }
        return CompleteP8ArenaTerminal(false);
    }

    // P8 is fail-closed: a transport identity/terminal-publication error must
    // not leave an earlier private READY mapping visible to QSFA.  This is a
    // monotonic GM CAS, never a local read/modify/write.  Reloading afterward
    // makes a losing worker observe the same graph-visible gate.
    __aicore__ inline bool MarkForwardAbort()
    {
        // Do not allow an old graph node to poison a later forward: reload
        // the single generation token before either the arena transition or
        // the low-word abort CAS.  A mismatch is a no-op; admission/topology
        // owns recovery rather than mutating an unknown lease.
        if (!ReloadForwardControlForPhase(
                static_cast<kvaf_fused_get::KvafP8ForwardPhase>(phase_),
                true) || !LatchP8ArenaUnhealthy()) {
            AddStat(kP8StatControlRejected, 1U);
            return false;
        }
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

    // Once a queue lease has a durable P8 FILLING record, the GM state word
    // is the authority for every pre-doorbell failure path.  Never use a
    // full-record Store to mark a FILLING/POST_INTENT wave failed: a
    // ForwardTerminal worker may have concurrently won the same state word.
    // The narrow KVAF CAS helpers leave any losing worker with only one legal
    // outcome: latch forward abort and retain the resource for cold recovery.
    __aicore__ inline bool PersistWaveQuarantine(
        kvaf_fused_get::KvafP8WaveRecord* wave,
        kvaf_fused_get::KvafP8WaveOutcome outcome)
    {
        if (wave == nullptr || !kvaf_fused_get::IsKvafP8WaveRecordValid(*wave) ||
            wave->identity.queue_id >= tiling_->queueCount) {
            return false;
        }
        __gm__ kvaf_fused_get::KvafP8WaveRecord* const wave_gm =
            waveRecords_ + wave->identity.queue_id;
        bool quarantined = false;
        if (wave->state == kvaf_fused_get::kKvafP8WaveFilling ||
            wave->state ==
                kvaf_fused_get::kKvafP8WaveFillingReleaseClaimed) {
            quarantined = kvaf_fused_get::QuarantineKvafP8FillingStateFromAicore(
                wave_gm, wave->state, outcome, drainClaimResult_, simtEvent_,
                wave);
        } else if (wave->state == kvaf_fused_get::kKvafP8WavePostIntent) {
            // The only normal POST_INTENT failure before the doorbell is a
            // lost control/identity proof.  KVAF re-loads every identity word
            // and CASes the state, so a stale producer cannot poison a newer
            // queue generation that reused this dynamic queue id.
            quarantined =
                kvaf_fused_get::QuarantineKvafP8PostIntentStateFromAicore(
                    wave_gm, wave->identity, drainClaimResult_, simtEvent_,
                    wave);
        } else if (wave->state == kvaf_fused_get::kKvafP8WaveQuarantined) {
            // A helper may have won the CAS before returning false.  It is
            // already durable; do not rewrite it from this AIV-local copy.
            quarantined = true;
        }
        if (!quarantined) {
            return false;
        }
        AddStat(kP8StatQuarantined, 1U);
        return true;
    }

    __aicore__ inline void FailPostAfterFilling(
        kvaf_fused_get::KvafP8WaveRecord* wave,
        kvaf_fused_get::KvafP8WaveOutcome outcome)
    {
        (void)PersistWaveQuarantine(wave, outcome);
        (void)MarkForwardAbort();
    }

    // DRAIN and PostOnly never full-store a stale AIC-local wave record.  The
    // helper writes mutable payload, publishes it to DDR, then CASes the
    // exact expected GM state.  A ForwardTerminal winner therefore cannot be
    // resurrected as POSTED/FINALIZED/FREE by a late worker.
    __aicore__ inline bool CommitWaveState(
        kvaf_fused_get::KvafP8WaveRecord* wave, uint32_t expected_state,
        uint32_t desired_state)
    {
        if (wave == nullptr ||
            !kvaf_fused_get::IsKvafP8WaveRecordValid(*wave) ||
            wave->identity.queue_id >= tiling_->queueCount) {
            return false;
        }
        wave->state = desired_state;
        return kvaf_fused_get::CommitKvafP8WaveMutablePayloadAndStateFromAicore(
            waveRecords_ + wave->identity.queue_id, wave->identity,
            expected_state, desired_state, *wave, drainClaimResult_,
            simtEvent_, wave);
    }

    // The only normal pre-doorbell release path.  KVAF reloads FILLING from
    // GM, proves the no-work marker and exact owner CAS before it can write
    // FREE.  Any failed proof is already durable quarantine where possible;
    // make the graph-wide fallback gate sticky rather than attempting a
    // second local cleanup.
    __aicore__ inline bool ReleaseFillingAfterProvenNoWork(
        const kvaf_fused_get::KvafStreamingBatchGetLease& lease,
        kvaf_fused_get::KvafP8WaveRecord* wave)
    {
        if (wave == nullptr ||
            !ReloadForwardControlForPhase(
                kvaf_fused_get::kKvafP8ForwardPhaseAdmitPost, false) ||
            !kvaf_fused_get::ReleaseKvafP8FillingLeaseAfterProvenNoWork(
                controlGm_, forwardControl_, lease, leaseResult_,
                drainClaimResult_, simtEvent_, waveRecords_ + lease.queue_id,
                wave)) {
            if (wave != nullptr &&
                kvaf_fused_get::IsKvafP8WaveRecordValid(*wave) &&
                (wave->state == kvaf_fused_get::kKvafP8WaveFilling ||
                 wave->state ==
                     kvaf_fused_get::kKvafP8WaveFillingReleaseClaimed)) {
                (void)PersistWaveQuarantine(
                    wave, kvaf_fused_get::kKvafP8WaveOutcomeFillingQuarantined);
            }
            (void)MarkForwardAbort();
            return false;
        }
        AddStat(kP8StatQueueReleased, 1U);
        return true;
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
            // PENDING -> QUARANTINED is a state ownership transition, not a
            // best-effort diagnostic write.  A ForwardTerminal invocation
            // may have raced this POST worker, so KVAF validates every
            // identity word then CASes only PENDING.  A losing worker leaves
            // the row unavailable and the shared abort/arena-health gate
            // forces exact fallback; it must never restore a stale local
            // record with a full GM Store.
            if (kvaf_fused_get::QuarantineKvafP8PrivatePendingFromAicore(
                    privateRecords_ + output_index, identity, privateReady_,
                    static_cast<uint64_t>(output_index), forwardTerminalResult_,
                    simtEvent_)) {
                AddStat(kP8StatQuarantined, 1U);
            }
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
        const kvaf_fused_get::KvafStreamingBatchGetLease& lease,
        kvaf_fused_get::KvafP8WaveRecord* filling_wave,
        bool* external_writes_armed, bool* fatal, uint32_t* staged)
    {
        if (seed == nullptr || staged == nullptr || filling_wave == nullptr ||
            external_writes_armed == nullptr || fatal == nullptr ||
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
        // Avoid arming the FILLING external-write edge for an already
        // claimed candidate.  This is only a contention fast-path: the
        // subsequent CAS claim remains authoritative and still handles a
        // concurrent producer without ever recycling an ambiguous wave.
        if (ReadGmByPassDCache<int32_t>(privateSourceToSlot_ + source_offset) !=
                kUnmapped ||
            ReadGmByPassDCache<int32_t>(privateObjectToSlot_ + object_offset) !=
                kUnmapped) {
            return false;
        }
        // The FILLING no-work proof must be cleared and persisted before the
        // first descriptor/private/ledger byte becomes externally visible.
        // Keeping this edge adjacent to the first private claim means a
        // receipt containing only invalid/resident candidates can still use
        // KVAF's proven-no-work release path.
        if (!*external_writes_armed) {
            if (!ReloadForwardControlForPhase(
                    kvaf_fused_get::kKvafP8ForwardPhaseAdmitPost, false) ||
                !kvaf_fused_get::ArmKvafP8WaveFillingForExternalWrites(
                    waveRecords_ + identity.queue_id, control_, forwardControl_,
                    lease, identity, drainClaimResult_, simtEvent_,
                    filling_wave)) {
                *fatal = true;
                return false;
            }
            *external_writes_armed = true;
        }
        // The claim comes after every deterministic descriptor validation.
        // After this point the only work is writes into a prevalidated bounded
        // GM ledger, so we never clear a partially claimed row by guessing
        // whether a later PostOnly may have observed it.
        if (!ClaimGlobalPrivate(source_offset, object_offset, identity) ||
            claimResult_->output_index < 0) {
            if (claimResult_->quarantined != 0) {
                *fatal = true;
            }
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
        if (!HasP8ArenaAdmissionLease()) {
            // Do not overwrite this layer's control record merely because a
            // prior layer/forward quarantined the shared arena.  The sticky
            // scalar is graph-visible to the caller and forces the exact
            // fallback; terminal teardown remains separately runnable.
            AddStat(kP8StatControlRejected, 1U);
            return false;
        }
        if (!ReloadForwardControlForPhase(
                kvaf_fused_get::kKvafP8ForwardPhaseAdmitPost, false)) {
            AddStat(kP8StatControlRejected, 1U);
            return false;
        }
        P8CandidateReceiptIdentity receipt_identity{};
        if (!LoadCurrentCandidateReceipt(receipt, &receipt_identity)) {
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
        P8CandidateReceiptIdentity cursor_identity{};
        if (!LoadStableReceiptIdentity(CandidateCursorIdentity(receipt),
                                       &cursor_identity) ||
            !SameReceiptIdentity(cursor_identity, receipt_identity)) {
            cursor = 0;
            WriteGmByPassDCache<int32_t>(candidateCursor_ + receipt, 0);
            PublishCursorReceiptIdentity(receipt, receipt_identity);
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
        if (!kvaf_fused_get::LoadKvafP8WaveRecordFromGm(
                waveRecords_ + queue, &previous)) {
            // We now own a queue lease but cannot safely overwrite an
            // unreadable graph-stable wave slot.  Keep the lease held for
            // terminal/cold recovery and make P8 unavailable for this
            // forward; a speculative transport resource is never recycled
            // based on a local assumption.
            (void)MarkForwardAbort();
            return false;
        }
        const bool previous_valid =
            kvaf_fused_get::IsKvafP8WaveRecordValid(previous);
        kvaf_fused_get::KvafP8WaveIdentity identity{};
        identity.forward_epoch = forwardEpoch_;
        identity.graph_replay_generation = replayGeneration_;
        identity.layer_control_index = layerControlIndex_;
        identity.queue_id = queue;
        identity.queue_generation = previous_valid
            ? previous.identity.queue_generation + 1U : 1U;
        identity.post_sequence = previous_valid
            ? previous.identity.post_sequence + 1U : 1U;
        identity.owner_tag = lease.owner_tag;
        identity.reserved0 = 0U;
        if (identity.queue_generation == 0U || identity.post_sequence == 0U ||
            identity.owner_tag <= 0) {
            // No FILLING record can represent a wrapped identity.  Retain
            // the lease rather than trying a best-effort release from this
            // pre-publication edge.
            (void)MarkForwardAbort();
            AddStat(kP8StatControlRejected, 1U);
            return false;
        }
        // This is intentionally the first durable action after acquiring a
        // dynamic queue.  The KVAF helper accepts only all-zero or valid FREE
        // slots, writes FILLING, and performs the DDR publish barrier.  A
        // non-FREE/malformed prior slot is left untouched and the held lease
        // becomes terminal/cold-recovery state rather than an ABA overwrite.
        kvaf_fused_get::KvafP8WaveRecord wave{};
        if (!ReloadForwardControlForPhase(
                kvaf_fused_get::kKvafP8ForwardPhaseAdmitPost, false) ||
            !kvaf_fused_get::BeginAndPublishKvafP8WaveFilling(
                waveRecords_ + queue, control_, forwardControl_, lease,
                identity, drainClaimResult_, simtEvent_, &wave)) {
            (void)MarkForwardAbort();
            return false;
        }
        kvaf_fused_get::KvafStreamingBatchGetDescriptorScratch descriptor{};
        if (!kvaf_fused_get::LoadKvafStreamingBatchGetQueueTemplate(
                control_, lease, &descriptor) || descriptor.indices == 0U ||
            descriptor.offsets == 0U || descriptor.output_indices == 0U ||
            descriptor.mask == 0U || descriptor.table_capacity == 0U ||
            descriptor.request_capacity < tiling_->requestCapacity) {
            // Descriptor template loading has not exposed any descriptor,
            // private record, or ledger byte.  This route is safe only if
            // KVAF re-proves the durable FILLING no-work marker and wins the
            // owner-table CAS.
            (void)ReleaseFillingAfterProvenNoWork(lease, &wave);
            AddStat(kP8StatControlRejected, 1U);
            return false;
        }
        uint32_t staged = 0U;
        uint32_t next = static_cast<uint32_t>(cursor);
        const uint32_t limit = static_cast<uint32_t>(count_value);
        bool external_writes_armed = false;
        bool candidate_fatal = false;
        while (next < limit && staged < tiling_->requestCapacity) {
            (void)StageCandidate(Candidate(receipt, next), identity, descriptor,
                                 lease, &wave, &external_writes_armed,
                                 &candidate_fatal, &staged);
            ++next;
            if (candidate_fatal) {
                break;
            }
        }
        if (candidate_fatal) {
            if (staged != 0U) {
                QuarantineUnpostedRows(identity, staged);
            }
            FailPostAfterFilling(
                &wave, kvaf_fused_get::kKvafP8WaveOutcomeFillingQuarantined);
            return false;
        }
        if (staged == 0U) {
            WriteGmByPassDCache<int32_t>(candidateCursor_ + receipt,
                                         static_cast<int32_t>(next));
            if (external_writes_armed) {
                // The marker was durably armed before a candidate could be
                // claimed.  Even if all claims deferred, another invocation
                // cannot prove there was no externally visible partial
                // mutation, so retain/quarantine rather than recycle.
                FailPostAfterFilling(
                    &wave,
                    kvaf_fused_get::kKvafP8WaveOutcomeFillingQuarantined);
            } else {
                (void)ReleaseFillingAfterProvenNoWork(lease, &wave);
            }
            return false;
        }
        descriptor.active_request_count = staged;
        WriteGmByPassDCache<int32_t>(queueRequestCount_ + queue,
                                     static_cast<int32_t>(staged));
        DataSyncBarrier<MemDsbT::DDR>();
        if (!kvaf_fused_get::PublishKvafP8BatchGetDescriptorScratch(
                control_, lease, identity, privateRecords_,
                tiling_->privateRowCount, descriptor)) {
            // The external-write marker is durable, so no POST-side release
            // is legal even though the doorbell has not run.
            QuarantineUnpostedRows(identity, staged);
            FailPostAfterFilling(
                &wave, kvaf_fused_get::kKvafP8WaveOutcomeFillingQuarantined);
            AddStat(kP8StatControlRejected, 1U);
            return false;
        }
        if (!ReloadForwardControlForPhase(
                kvaf_fused_get::kKvafP8ForwardPhaseAdmitPost, false)) {
            // Still before POST_INTENT, but descriptor/private/ledger state
            // is already externally visible.  Preserve FILLING as durable
            // quarantine; do not perform a generic pre-doorbell release.
            QuarantineUnpostedRows(identity, staged);
            FailPostAfterFilling(
                &wave, kvaf_fused_get::kKvafP8WaveOutcomeFillingQuarantined);
            return false;
        }
        if (!kvaf_fused_get::PromoteKvafP8WaveFillingToPostIntent(
                waveRecords_ + queue, control_, forwardControl_, lease,
                identity, staged,
                privateRecords_, tiling_->privateRowCount,
                queueRequestSourceOffsets_, queueRequestObjectOffsets_,
                queueRequestOutputIndices_, tiling_->requestCapacity,
                drainClaimResult_, simtEvent_, &wave)) {
            // Promote reloads exact FILLING identity/lease from GM.  It
            // quarantines a matching failed record itself; retain all
            // descriptor/private state for ForwardTerminal and latch the
            // graph-wide fallback gate.
            QuarantineUnpostedRows(identity, staged);
            FailPostAfterFilling(
                &wave, kvaf_fused_get::kKvafP8WaveOutcomeFillingQuarantined);
            return false;
        }
        // POST_INTENT is durable.  Re-read the shared gate immediately
        // before HCCP PostOnly: another AIV may have aborted this forward
        // between descriptor publication and this irreversible operation.
        if (!ReloadForwardControlForPhase(
                kvaf_fused_get::kKvafP8ForwardPhaseAdmitPost, false)) {
            FailPostAfterFilling(
                &wave, kvaf_fused_get::kKvafP8WaveOutcomeIdentityQuarantined);
            return false;
        }
        const bool posted = kvaf_fused_get::PostKvafP8Wave(
            control_, forwardControl_, privateRecords_, tiling_->privateRowCount,
            &wave);
        uint32_t post_state = wave.state;
        if (post_state != kvaf_fused_get::kKvafP8WavePosted &&
            post_state != kvaf_fused_get::kKvafP8WaveFinalized &&
            post_state != kvaf_fused_get::kKvafP8WaveQuarantined) {
            wave.state = kvaf_fused_get::kKvafP8WaveQuarantined;
            wave.outcome = kvaf_fused_get::kKvafP8WaveOutcomeIdentityQuarantined;
            post_state = wave.state;
        }
        if (!CommitWaveState(&wave, kvaf_fused_get::kKvafP8WavePostIntent,
                             post_state)) {
            (void)MarkForwardAbort();
            return false;
        }
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
                tiling_->privateRowCount, privateReleaseResult_, simtEvent_,
                wave)) {
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
                (void)CommitWaveState(
                    wave, kvaf_fused_get::kKvafP8WaveDrainClaimed,
                    kvaf_fused_get::kKvafP8WaveQuarantined);
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
                (void)CommitWaveState(
                    wave, kvaf_fused_get::kKvafP8WaveDrainClaimed,
                    kvaf_fused_get::kKvafP8WaveQuarantined);
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
            (void)CommitWaveState(
                wave, kvaf_fused_get::kKvafP8WaveDrainClaimed,
                kvaf_fused_get::kKvafP8WaveQuarantined);
            AddStat(kP8StatQuarantined, 1U);
            (void)MarkForwardAbort();
            return false;
        }
        // The specialized KVAF endpoint releases the owner table first;
        // only its exact no-post state-last commit may make this DRAIN claim
        // FREE.  A generic DRAIN->FREE edge would be unsafe for real CQ work.
        if (!kvaf_fused_get::CommitKvafP8ClaimedUnpostedWaveReleaseFromAicore(
                waveRecords_ + queue, wave->identity, *wave,
                drainClaimResult_, simtEvent_, wave)) {
            (void)MarkForwardAbort();
            return false;
        }
        // The wave is now durably FREE through the sole verified no-post
        // edge.  Clearing the request count before that state commit would
        // make a failed CAS look like a reusable empty queue to diagnostics.
        WriteGmByPassDCache<int32_t>(queueRequestCount_ + queue, 0);
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
                (void)CommitWaveState(
                    &wave, kvaf_fused_get::kKvafP8WaveDrainClaimed,
                    kvaf_fused_get::kKvafP8WaveQuarantined);
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
            if (!CommitWaveState(&wave,
                                 kvaf_fused_get::kKvafP8WaveDrainClaimed,
                                 kvaf_fused_get::kKvafP8WavePosted)) {
                (void)MarkForwardAbort();
            }
            return;
        }
        if (wave.state == kvaf_fused_get::kKvafP8WaveQuarantined ||
            wave.state != kvaf_fused_get::kKvafP8WaveDrainClaimed) {
            wave.state = kvaf_fused_get::kKvafP8WaveQuarantined;
            if (wave.outcome == kvaf_fused_get::kKvafP8WaveOutcomeNone) {
                wave.outcome =
                    kvaf_fused_get::kKvafP8WaveOutcomeIdentityQuarantined;
            }
            (void)CommitWaveState(&wave,
                                  kvaf_fused_get::kKvafP8WaveDrainClaimed,
                                  kvaf_fused_get::kKvafP8WaveQuarantined);
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
            if (!CommitWaveState(&wave,
                                 kvaf_fused_get::kKvafP8WaveDrainClaimed,
                                 kvaf_fused_get::kKvafP8WavePosted)) {
                (void)MarkForwardAbort();
            }
            return;
        }
        if (wave.state != kvaf_fused_get::kKvafP8WaveFinalized) {
            wave.state = kvaf_fused_get::kKvafP8WaveQuarantined;
            if (wave.outcome == kvaf_fused_get::kKvafP8WaveOutcomeNone) {
                wave.outcome =
                    kvaf_fused_get::kKvafP8WaveOutcomeIdentityQuarantined;
            }
            (void)CommitWaveState(&wave,
                                  kvaf_fused_get::kKvafP8WaveDrainClaimed,
                                  kvaf_fused_get::kKvafP8WaveQuarantined);
            AddStat(kP8StatQuarantined, 1U);
            MarkForwardAbort();
            return;
        }
        // Finish has made a terminal CQ result local, but the next phase may
        // execute on another AIV.  Commit DRAIN_CLAIMED -> FINALIZED before
        // any private READY publication or queue release; otherwise a stale
        // terminal worker could still own the GM state word.
        if (!CommitWaveState(&wave,
                             kvaf_fused_get::kKvafP8WaveDrainClaimed,
                             kvaf_fused_get::kKvafP8WaveFinalized)) {
            (void)MarkForwardAbort();
            return;
        }
        // Queue release is another ownership-changing operation.  Reload the
        // GM gate after Finish/private publication so a concurrent abort
        // cannot let this worker recycle a queue from the wrong forward.
        if (!ReloadForwardControlForPhase(
                kvaf_fused_get::kKvafP8ForwardPhaseDrain, false)) {
            kvaf_fused_get::QuarantineKvafP8WaveRecord(
                &wave, kvaf_fused_get::kKvafP8WaveOutcomeIdentityQuarantined);
            (void)CommitWaveState(&wave,
                                  kvaf_fused_get::kKvafP8WaveFinalized,
                                  kvaf_fused_get::kKvafP8WaveQuarantined);
            (void)MarkForwardAbort();
            return;
        }
        if (!PublishTerminalPrivateRows(&wave)) {
            kvaf_fused_get::QuarantineKvafP8WaveRecord(
                &wave, kvaf_fused_get::kKvafP8WaveOutcomeIdentityQuarantined);
            (void)CommitWaveState(&wave,
                                  kvaf_fused_get::kKvafP8WaveFinalized,
                                  kvaf_fused_get::kKvafP8WaveQuarantined);
            AddStat(kP8StatQuarantined, 1U);
            MarkForwardAbort();
            return;
        }
        const bool released = kvaf_fused_get::ReleaseKvafP8WaveAfterPrivateReady(
            controlGm_, forwardControl_, privateRecords_, tiling_->privateRowCount,
            leaseResult_, simtEvent_, &wave);
        if (!(released && wave.state == kvaf_fused_get::kKvafP8WaveFree)) {
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
        const uint32_t release_state = released &&
                wave.state == kvaf_fused_get::kKvafP8WaveFree
            ? kvaf_fused_get::kKvafP8WaveFree
            : kvaf_fused_get::kKvafP8WaveQuarantined;
        if (!CommitWaveState(&wave,
                             kvaf_fused_get::kKvafP8WaveFinalized,
                             release_state)) {
            (void)MarkForwardAbort();
            return;
        }
        if (release_state == kvaf_fused_get::kKvafP8WaveFree) {
            // State-last wave release is durable before the ledger becomes
            // empty.  This keeps a failed final CAS fail-closed.
            WriteGmByPassDCache<int32_t>(queueRequestCount_ + queue, 0);
            AddStat(kP8StatQueueReleased, 1U);
            AddStat(kP8StatCqTerminal, 1U);
        }
    }

    __aicore__ inline void RunDrain()
    {
        if (!HasP8ArenaAdmissionLease()) {
            AddStat(kP8StatControlRejected, 1U);
            return;
        }
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
        if (!HasP8ArenaAdmissionLease() ||
            !ReloadForwardControlForPhase(
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

    __aicore__ inline void TombstoneCurrentCandidateReceipt(
        uint32_t receipt)
    {
        // ForwardTerminal is ordered by the shared forward-control graph
        // phase.  A new replay must not publish Stage-A receipts until this
        // terminal phase has completed; generation-first invalidation makes
        // that ordering visible to any in-flight Stage-B reader as well.
        P8CandidateReceiptIdentity candidate_identity{};
        if (LoadStableReceiptIdentity(CandidateReceiptIdentity(receipt),
                                      &candidate_identity) &&
            IsCurrentReceiptIdentity(candidate_identity)) {
            TombstoneReceiptIdentity(CandidateReceiptIdentity(receipt));
            WriteGmByPassDCache<int32_t>(candidateCounts_ + receipt, 0);
            WriteGmByPassDCache<int32_t>(candidateStatus_ + receipt,
                                         kCandidateStatusTombstoned);
        }
        P8CandidateReceiptIdentity cursor_identity{};
        if (LoadStableReceiptIdentity(CandidateCursorIdentity(receipt),
                                      &cursor_identity) &&
            IsCurrentReceiptIdentity(cursor_identity)) {
            TombstoneReceiptIdentity(CandidateCursorIdentity(receipt));
            WriteGmByPassDCache<int32_t>(candidateCursor_ + receipt, 0);
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
        // A terminal release of the global arena lease cannot be sharded
        // across AIV blocks: releasing after one shard finishes would admit a
        // new forward while another shard can still quarantine a resource.
        // This rare safety phase is therefore intentionally serialized on
        // block 0; normal PostOnly/Drain remain fully dynamic.
        if (worker_ != 0U) {
            return;
        }
        if (!ReloadForwardControlForPhase(
                kvaf_fused_get::kKvafP8ForwardPhaseForwardTerminal, true)) {
            AddStat(kP8StatControlRejected, 1U);
            return;
        }
        bool entered_unhealthy = false;
        if (!ClaimP8ArenaForForwardTerminal(&entered_unhealthy)) {
            // A stale terminal has no authority over a newer tagged lease.
            // It must not sweep, abort or free any resource.
            AddStat(kP8StatControlRejected, 1U);
            return;
        }
        // This closes the receipt ABA domain even when no wave was ever
        // posted: an abandoned receipt/cursor cannot be adopted by a later
        // replay that happens to share forward epoch-visible storage.
        for (uint32_t receipt = 0U; receipt < tiling_->receiptCount;
             ++receipt) {
            TombstoneCurrentCandidateReceipt(receipt);
        }
        bool abort_required = entered_unhealthy;
        for (uint32_t queue = 0U; queue < tiling_->queueCount; ++queue) {
            kvaf_fused_get::KvafP8WaveRecord record{};
            if (!kvaf_fused_get::LoadKvafP8WaveRecordFromGm(
                    waveRecords_ + queue, &record)) {
                abort_required = true;
                continue;
            }
            // FILLING_CLAIMED is intentionally valid even when magic and
            // identity remain zero: FREE/all-zero -> CLAIMED is the durable
            // admission reservation.  Terminal must seal it, never skip it
            // as an uninitialized FREE slot.
            if (record.state == kvaf_fused_get::kKvafP8WaveFillingClaimed) {
                forwardTerminalResult_->success = 0U;
                forwardTerminalResult_->observed_state = record.state;
                (void)kvaf_fused_get::QuarantineKvafP8WaveFillingClaimedFromAicore(
                    waveRecords_ + queue, drainClaimResult_, simtEvent_);
                abort_required = true;
                AddStat(kP8StatForwardTerminalWave, 1U);
                continue;
            }
            if (!kvaf_fused_get::IsKvafP8WaveRecordValid(record)) {
                if (!kvaf_fused_get::IsKvafP8WaveRecordAllZero(record)) {
                    abort_required = true;
                }
                continue;
            }
            if (record.state == kvaf_fused_get::kKvafP8WaveFree) {
                continue;
            }
            if (!SameForward(record.identity)) {
                // ACTIVE_FORWARD excludes an overlapping valid invocation,
                // so a non-free foreign identity is persistent arena damage.
                abort_required = true;
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
            // A matching durable QUARANTINED record is itself evidence that
            // this reusable arena cannot admit a later P8 forward.  Latch
            // the global health scalar for every non-FREE observation,
            // including a quarantine created by an earlier worker.
            if (forwardTerminalResult_->observed_state !=
                    kvaf_fused_get::kKvafP8WaveFree) {
                abort_required = true;
                AddStat(kP8StatForwardTerminalWave, 1U);
            }
        }
        for (uint32_t slot = 0U; slot < tiling_->privateRowCount; ++slot) {
            kvaf_fused_get::KvafP8PrivateRecord record{};
            if (!kvaf_fused_get::LoadKvafP8PrivateRecordFromGm(
                    privateRecords_ + slot, &record)) {
                abort_required = true;
                continue;
            }
            // ALLOCATING may intentionally have no valid identity yet.  It
            // is nevertheless a durable ownership claim, so ForwardTerminal
            // must quarantine it rather than skipping it as a malformed FREE
            // record.  The helper only rolls back transient -2 map markers
            // after it proves the immutable offsets were published.
            if (record.state == kvaf_fused_get::kKvafP8PrivateAllocating) {
                forwardTerminalResult_->success = 0U;
                forwardTerminalResult_->observed_state = record.state;
                if (!kvaf_fused_get::QuarantineKvafP8PrivateAllocatingAtForwardTerminal(
                            privateRecords_ + slot, forwardControl_,
                            privateSourceToSlot_,
                            static_cast<uint64_t>(tiling_->rowCount) *
                                tiling_->sourceCapacity,
                            privateObjectToSlot_,
                            static_cast<uint64_t>(tiling_->poolSize) *
                                tiling_->kvObjectCapacity,
                            privateReady_, slot, forwardTerminalResult_,
                            simtEvent_)) {
                    abort_required = true;
                } else {
                    abort_required = true;
                    AddStat(kP8StatForwardTerminalPrivate, 1U);
                }
                continue;
            }
            if (record.state ==
                kvaf_fused_get::kKvafP8PrivateMapPublishing) {
                // The producer exclusively owns its two -2 -> positive map
                // CAS operations in this state.  Touching the record or maps
                // here could create a stale positive mapping, so terminal
                // only makes the tagged arena fail closed.  Cold recovery is
                // the sole path that may reclaim this bounded uncertainty.
                abort_required = true;
                AddStat(kP8StatForwardTerminalPrivate, 1U);
                continue;
            }
            if (!kvaf_fused_get::IsKvafP8PrivateRecordValid(record)) {
                if (!IsP8PrivateRecordAllZero(record)) {
                    abort_required = true;
                }
                continue;
            }
            if (record.state == kvaf_fused_get::kKvafP8PrivateFree) {
                continue;
            }
            if (!SamePrivateForward(record)) {
                abort_required = true;
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
                    kvaf_fused_get::kKvafP8PrivateFree) {
                abort_required = true;
                AddStat(kP8StatForwardTerminalPrivate, 1U);
            }
        }
        if (abort_required ||
            (forwardControl_.flags &
             (kvaf_fused_get::kKvafP8ForwardFlagAbort |
              kvaf_fused_get::kKvafP8ForwardFlagPrivateViewDisabled)) != 0U) {
            (void)MarkForwardAbortAfterTerminalClaim();
        } else if (!ReleaseP8ArenaAfterCleanForwardTerminal()) {
            // A terminal node that cannot release its own ACTIVE_FORWARD
            // lease must poison the arena rather than leaving a later graph
            // to infer whether it overlaps this forward.
            (void)MarkForwardAbortAfterTerminalClaim();
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
    TBuf<TPosition::VECCALC> arenaLeaseBuffer_;
    __ubuf__ kvaf_fused_get::KvafStreamingBatchGetLeaseResult* leaseResult_ = nullptr;
    __ubuf__ P8PrivateClaimResult* claimResult_ = nullptr;
    __ubuf__ kvaf_fused_get::KvafP8WaveDrainClaimResult* drainClaimResult_ = nullptr;
    __ubuf__ kvaf_fused_get::KvafP8ForwardAbortResult* abortResult_ = nullptr;
    __ubuf__ kvaf_fused_get::KvafP8PrivateReleaseResult* privateReleaseResult_ = nullptr;
    __ubuf__ kvaf_fused_get::KvafP8ForwardTerminalResult* forwardTerminalResult_ = nullptr;
    __ubuf__ P8ArenaLeaseResult* arenaLeaseResult_ = nullptr;
    event_t simtEvent_ = static_cast<event_t>(EVENT_ID0);

    __gm__ int32_t* candidateSeeds_ = nullptr;
    __gm__ int32_t* candidateCounts_ = nullptr;
    __gm__ int64_t* candidateReceiptIdentity_ = nullptr;
    __gm__ int32_t* candidateStatus_ = nullptr;
    __gm__ kvaf_fused_get::KvafP8ForwardControl* forwardControlGm_ = nullptr;
    __gm__ int32_t* arenaHealthGm_ = nullptr;
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
    __gm__ int64_t* candidateCursorIdentity_ = nullptr;
    __gm__ int32_t* rowsPerEntry_ = nullptr;
    __gm__ int64_t* workerStats_ = nullptr;

    kvaf_fused_get::KvafStreamingBatchGetControl control_{};
    kvaf_fused_get::KvafP8ForwardControl forwardControl_{};
    uint32_t worker_ = 0U;
    uint64_t forwardEpoch_ = 0U;
    uint64_t replayGeneration_ = 0U;
    uint32_t layerControlIndex_ = 0U;
    uint64_t phase_ = 0U;
    uint32_t arenaLeaseGeneration_ = 0U;
    uint32_t arenaActiveTag_ = 0U;
    uint32_t arenaTerminalTag_ = 0U;
    bool enabled_ = false;
};
}  // namespace

extern "C" __global__ __aicore__ void a5_p8_forward_owned_executor(
    GM_ADDR candidate_seeds, GM_ADDR candidate_counts,
    GM_ADDR candidate_receipt_identity, GM_ADDR candidate_status,
    GM_ADDR forward_control, GM_ADDR p8_arena_health, GM_ADDR source_lens,
    GM_ADDR kv_object_address_domain, GM_ADDR req_pool_entries,
    GM_ADDR kv_object_map, GM_ADDR kv_object_lens,
    GM_ADDR kv_object_slots_snapshot, GM_ADDR private_source_to_slot,
    GM_ADDR private_object_to_slot, GM_ADDR private_ready_table,
    GM_ADDR private_slot_cursor, GM_ADDR private_packed_output,
    GM_ADDR fused_get_control, GM_ADDR p8_wave_records,
    GM_ADDR p8_private_records, GM_ADDR queue_request_source_offsets,
    GM_ADDR queue_request_object_offsets, GM_ADDR queue_request_output_indices,
    GM_ADDR queue_request_count, GM_ADDR candidate_cursor,
    GM_ADDR candidate_cursor_identity, GM_ADDR rows_per_entry,
    GM_ADDR worker_stats,
    GM_ADDR forward_control_ref, GM_ADDR p8_arena_health_ref,
    GM_ADDR private_source_to_slot_ref,
    GM_ADDR private_object_to_slot_ref,
    GM_ADDR private_ready_table_ref, GM_ADDR private_slot_cursor_ref,
    GM_ADDR private_packed_output_ref, GM_ADDR fused_get_control_ref,
    GM_ADDR p8_wave_records_ref, GM_ADDR p8_private_records_ref,
    GM_ADDR queue_request_source_offsets_ref,
    GM_ADDR queue_request_object_offsets_ref,
    GM_ADDR queue_request_output_indices_ref, GM_ADDR queue_request_count_ref,
    GM_ADDR candidate_cursor_ref, GM_ADDR candidate_cursor_identity_ref,
    GM_ADDR worker_stats_ref, GM_ADDR candidate_receipt_identity_ref,
    GM_ADDR candidate_status_ref, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)forward_control_ref;
    (void)p8_arena_health_ref;
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
    (void)candidate_cursor_identity_ref;
    (void)worker_stats_ref;
    (void)candidate_receipt_identity_ref;
    (void)candidate_status_ref;
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5P8ForwardOwnedExecutorTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    A5P8ForwardOwnedExecutorKernel op(&pipe, &tilingData);
    op.Init(candidate_seeds, candidate_counts, candidate_receipt_identity,
            candidate_status, forward_control, p8_arena_health, source_lens,
            kv_object_address_domain, req_pool_entries, kv_object_map,
            kv_object_lens, kv_object_slots_snapshot, private_source_to_slot,
            private_object_to_slot, private_ready_table, private_slot_cursor,
            private_packed_output, fused_get_control, p8_wave_records,
            p8_private_records, queue_request_source_offsets,
            queue_request_object_offsets, queue_request_output_indices,
            queue_request_count, candidate_cursor, candidate_cursor_identity,
            rows_per_entry, worker_stats);
    op.Process();
}
