// P7.1 standalone CandidateSeed -> Direct-HCCP Post/CQ executor.
//
// This file intentionally does not include any C8 implementation header.
// It receives only already-published CandidateSeed GM records and leaves C8
// Top-K completely untouched.  Every AIV task executes the same code and may
// dynamically lease any KVAF queue; no physical AIV is a permanent Get lane.

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

#include "a5_score_receipt_speculative_executor_tiling.h"
// build_nanovllm_ops.sh stages this exact helper from KVAF_FUSED_GET_SOURCE.
// BeginKvafStreamingBatchGet delegates to the real Direct-HCCP PostOnly path;
// TryRetire/Finish retain the existing CQ semantics.
#include "kvaf_device/kvaf_fused_full_get_device.h"

using namespace AscendC;

namespace {
constexpr uint32_t kPackedKvRowBytes = 656U;
constexpr int32_t kUnmapped = -1;
constexpr int32_t kClaiming = -2;
constexpr int32_t kPrivateFree = 0;
constexpr int32_t kPrivatePending = 1;
constexpr int32_t kPrivateReady = 2;
constexpr int32_t kPrivateFailed = 3;
constexpr int32_t kCandidateStatusOk = 1;
// P7.1 owns one deliberately narrow address ABI: CandidateSeed.O is the
// `kv_object_map[P, S]` object index consumed by the streaming KVAF
// descriptor.  Raw-token O would require a separate conversion/table
// contract, so accepting it here would make the address-domain input a
// misleading no-op.  Reject it until that contract exists.
constexpr int32_t kKvObjectAddressDomainMappedObject = 1;

enum QueueLifecycle : int32_t {
    kQueueFree = 0,
    kQueueFilling = 1,
    kQueuePosted = 2,
    kQueueCqTerminal = 3,
    kQueueQuarantined = 4,
};

enum WorkerStat : uint32_t {
    kStatReceipts = 0U,
    kStatSeedsLoaded = 1U,
    kStatSeedsDeferred = 2U,
    kStatLeaseAcquired = 3U,
    kStatQueueBusy = 4U,
    kStatPrivateNew = 5U,
    kStatPrivateShared = 6U,
    kStatDoorbellPosted = 7U,
    kStatCqPoll = 8U,
    kStatCqReady = 9U,
    kStatCqFailed = 10U,
    kStatLeaseReleased = 11U,
    kStatLeaseQuarantined = 12U,
    kStatControlRejected = 13U,
    kStatStaleReceipt = 14U,
    // Counts actual `FinishKvafStreamingBatchGet` calls, not merely a
    // request to drain.  The compact P7.1 test uses this with lease/post/
    // retire/release to prove a real Direct-HCCP terminal lifecycle.
    kStatFinish = 15U,
};

enum ClaimKind : uint32_t {
    kClaimNone = 0U,
    kClaimNew = 1U,
    kClaimShared = 2U,
};

struct PrivateClaimResult {
    uint64_t source_offset;
    uint64_t object_offset;
    uint32_t local_slot;
    int32_t output_index;
    uint32_t kind;
    uint32_t success;
};
static_assert(sizeof(PrivateClaimResult) == 32U,
              "P7.1 private claim SIMT ABI changed");

// Scalar `ReadGmByPassDCache` is not callable from a `__simt_vf__` service.
// These are the matching L2-coherent SIMT loads used by KVAF's WQE
// materializer.  Mapping identity is read through these helpers so a replayed
// worker cannot observe a DCache-stale prior-forward tag.
// A VF service may call only SIMT callees.  These helpers are intentionally
// SIMT callees (rather than nested VF launches): ClaimPrivateSlotSimt is the
// one VF launch and uses them for coherent identity loads.
__simt_callee__ inline int32_t LoadGmInt32ForSimt(
    __gm__ const int32_t* address)
{
    return static_cast<int32_t>(
        asc_ldcg(const_cast<__gm__ int32_t*>(address)));
}

__simt_callee__ inline int64_t LoadGmInt64ForSimt(
    __gm__ const int64_t* address)
{
    return static_cast<int64_t>(static_cast<uint64_t>(
        asc_ldcg(const_cast<__gm__ int64_t*>(address))));
}

// This one-thread SIMT service is the only place that performs concurrent
// private-source/private-object ownership changes.  A worker first claims
// source and object as -2; a successful Begin later turns both into the final
// private output row.  An already-pending object is intentionally deferred,
// not waited on, so a stalled producer cannot create an unbounded AIV spin.
__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void ClaimPrivateSlotSimt(
    __gm__ int32_t* source_to_slot, __gm__ int32_t* object_to_slot,
    __gm__ int32_t* private_ready, __gm__ int32_t* private_arena_head,
    __gm__ int64_t* private_ready_epoch,
    __gm__ int64_t* private_ready_generation,
    uint64_t source_offset, uint64_t object_offset, uint32_t queue_id,
    uint32_t private_capacity, uint32_t private_packed_rows,
    uint64_t current_epoch, uint64_t current_generation,
    __ubuf__ PrivateClaimResult* result)
{
    if (source_to_slot == nullptr || object_to_slot == nullptr ||
        private_ready == nullptr || private_arena_head == nullptr ||
        private_ready_epoch == nullptr || private_ready_generation == nullptr ||
        result == nullptr || private_capacity == 0U ||
        private_packed_rows == 0U || current_generation == 0U) {
        return;
    }
    result->source_offset = source_offset;
    result->object_offset = object_offset;
    result->local_slot = 0U;
    result->output_index = kUnmapped;
    result->kind = kClaimNone;
    result->success = 0U;
    const int32_t old_source = asc_atomic_cas(
        source_to_slot + source_offset, kUnmapped, kClaiming);
    if (old_source != kUnmapped) {
        // Acquire the publisher's private-tag stores after observing a
        // nonnegative mapping CAS.  This pairs with the release fence in
        // PublishPrivateClaimSimt and prevents an old tag line from being
        // mistaken for this forward's shared transaction.
        if (old_source >= 0) {
            AscendC::Simt::ThreadFence();
        }
        // A nonnegative source mapping is reusable only when its terminal
        // receipt belongs to this forward.  This prevents a graph replay
        // from binding a source to an old forward's private output row.
        if (old_source >= 0 && static_cast<uint32_t>(old_source) < private_packed_rows &&
            LoadGmInt64ForSimt(private_ready_epoch + old_source) ==
                static_cast<int64_t>(current_epoch) &&
            LoadGmInt64ForSimt(private_ready_generation + old_source) > 0 &&
            LoadGmInt32ForSimt(private_ready + old_source) != kPrivateFree) {
            // This source was already claimed by the same forward.  It is
            // deliberately a shared claim, not a second post.  The caller
            // records the share and leaves the original owner/CQ lifecycle
            // unchanged.
            result->output_index = old_source;
            result->kind = kClaimShared;
            result->success = 1U;
            return;
        }
        // Stale/invalid values are cleared with a CAS and this candidate is
        // deferred. The final exact Recall owns the retry, so no unbounded
        // local retry loop can form here.
        asc_atomic_cas(source_to_slot + source_offset, old_source, kUnmapped);
        return;
    }
    const int32_t old_object = asc_atomic_cas(object_to_slot + object_offset,
                                               kUnmapped, kClaiming);
    if (old_object >= 0) {
        // Same acquire edge for an object-level share.  A source mapping may
        // be installed from this object only after its private identity is
        // visible to this SIMT service.
        AscendC::Simt::ThreadFence();
        // A previous wave already owns the object.  Point this source at the
        // same *forward* transaction-private output row and let its ready
        // state gate QSFA; there is no duplicate H2D post.
        if (static_cast<uint32_t>(old_object) < private_packed_rows &&
            LoadGmInt64ForSimt(private_ready_epoch + old_object) ==
                static_cast<int64_t>(current_epoch) &&
            LoadGmInt64ForSimt(private_ready_generation + old_object) > 0 &&
            LoadGmInt32ForSimt(private_ready + old_object) != kPrivateFree &&
            asc_atomic_cas(source_to_slot + source_offset, kClaiming,
                           old_object) == kClaiming) {
            result->output_index = old_object;
            result->kind = kClaimShared;
            result->success = 1U;
        } else {
            // Do not adopt an old forward's object. Clear only the exact
            // value observed above and let Guaranteed Recall cover it.
            asc_atomic_cas(object_to_slot + object_offset, old_object, kUnmapped);
            asc_atomic_cas(source_to_slot + source_offset, kClaiming, kUnmapped);
        }
        return;
    }
    if (old_object != kUnmapped) {
        // Another producer is claiming this object (-2) or its state is
        // malformed.  Defer rather than speculate a duplicate request.
        asc_atomic_cas(source_to_slot + source_offset, kClaiming, kUnmapped);
        return;
    }
    const uint32_t local_slot =
        asc_atomic_add(private_arena_head + queue_id, 1U);
    if (local_slot >= private_capacity) {
        asc_atomic_cas(object_to_slot + object_offset, kClaiming, kUnmapped);
        asc_atomic_cas(source_to_slot + source_offset, kClaiming, kUnmapped);
        return;
    }
    const int32_t output_index = static_cast<int32_t>(
        static_cast<uint64_t>(queue_id) * private_capacity + local_slot);
    if (output_index < 0 || static_cast<uint32_t>(output_index) >=
                                private_packed_rows) {
        asc_atomic_cas(object_to_slot + object_offset, kClaiming, kUnmapped);
        asc_atomic_cas(source_to_slot + source_offset, kClaiming, kUnmapped);
        return;
    }
    private_ready[static_cast<uint64_t>(queue_id) * private_capacity +
                  local_slot] = kPrivatePending;
    private_ready_epoch[output_index] = static_cast<int64_t>(current_epoch);
    private_ready_generation[output_index] =
        static_cast<int64_t>(current_generation);
    // Publish the pending state and its forward identity before another
    // SIMT service can adopt this source/object as a shared claim.
    AscendC::Simt::ThreadFence();
    result->local_slot = local_slot;
    result->output_index = output_index;
    result->kind = kClaimNew;
    result->success = 1U;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void PublishPrivateClaimSimt(
    __gm__ int32_t* source_to_slot, __gm__ int32_t* object_to_slot,
    uint64_t source_offset, uint64_t object_offset, int32_t output_index,
    __ubuf__ PrivateClaimResult* result)
{
    if (source_to_slot == nullptr || object_to_slot == nullptr ||
        result == nullptr || output_index < 0) {
        return;
    }
    result->success = 0U;
    if (asc_atomic_cas(object_to_slot + object_offset, kClaiming,
                       output_index) != kClaiming) {
        return;
    }
    if (asc_atomic_cas(source_to_slot + source_offset, kClaiming,
                       output_index) != kClaiming) {
        // Object now points at a private row that no source can prove it
        // owns.  Do not overwrite another value; caller quarantines this
        // queue and final Guaranteed Recall preserves mathematical output.
        return;
    }
    // Both mapping CASes must be visible before the caller treats this as a
    // published pending private transaction.
    AscendC::Simt::ThreadFence();
    result->success = 1U;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void RollbackPrivateClaimSimt(
    __gm__ int32_t* source_to_slot, __gm__ int32_t* object_to_slot,
    __gm__ int32_t* private_ready, uint64_t source_offset,
    uint64_t object_offset, uint64_t ready_offset,
    __ubuf__ PrivateClaimResult* result)
{
    if (source_to_slot == nullptr || object_to_slot == nullptr ||
        private_ready == nullptr || result == nullptr) {
        return;
    }
    asc_atomic_cas(source_to_slot + source_offset, kClaiming, kUnmapped);
    asc_atomic_cas(object_to_slot + object_offset, kClaiming, kUnmapped);
    private_ready[ready_offset] = kPrivateFree;
    result->success = 1U;
}

// A failed posted wave may have published some mappings while leaving later
// claims at -2.  Cleanup is legal only after the CQ becomes terminal: before
// then Direct-HCCP can still be writing the associated private output row.
__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void ClearTerminalFailedMappingSimt(
    __gm__ int32_t* source_to_slot, __gm__ int32_t* object_to_slot,
    uint64_t source_offset, uint64_t object_offset, int32_t output_index,
    __ubuf__ PrivateClaimResult* result)
{
    if (source_to_slot == nullptr || object_to_slot == nullptr ||
        result == nullptr || output_index < 0) {
        return;
    }
    // The CAS forms make this cleanup queue-local: it cannot erase a mapping
    // that another forward or an independently valid wave has already
    // replaced after the terminal state was published.
    asc_atomic_cas(source_to_slot + source_offset, kClaiming, kUnmapped);
    asc_atomic_cas(source_to_slot + source_offset, output_index, kUnmapped);
    asc_atomic_cas(object_to_slot + object_offset, kClaiming, kUnmapped);
    asc_atomic_cas(object_to_slot + object_offset, output_index, kUnmapped);
    result->success = 1U;
}

class A5ScoreReceiptSpeculativeExecutorKernel {
public:
    __aicore__ inline explicit A5ScoreReceiptSpeculativeExecutorKernel(
        TPipe* pipe,
        const A5ScoreReceiptSpeculativeExecutorTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR candidate_seeds, GM_ADDR candidate_counts,
        GM_ADDR candidate_ready_epoch, GM_ADDR candidate_status,
        GM_ADDR streaming_epoch, GM_ADDR source_lens,
        GM_ADDR kv_object_address_domain, GM_ADDR req_pool_entries,
        GM_ADDR kv_object_map, GM_ADDR kv_object_lens,
        GM_ADDR kv_object_slots_snapshot, GM_ADDR private_source_to_slot,
        GM_ADDR private_object_to_slot, GM_ADDR private_ready_table,
        GM_ADDR private_arena_head, GM_ADDR private_packed_output,
        GM_ADDR fused_get_control, GM_ADDR queue_leases, GM_ADDR queue_epoch,
        GM_ADDR queue_generation, GM_ADDR queue_post_sequence,
        GM_ADDR queue_state, GM_ADDR queue_quarantine,
        GM_ADDR queue_request_source_offsets,
        GM_ADDR queue_request_object_offsets,
        GM_ADDR queue_request_output_indices, GM_ADDR queue_request_count,
        GM_ADDR private_ready_epoch, GM_ADDR private_ready_generation,
        GM_ADDR worker_stats, GM_ADDR rows_per_entry)
    {
        // The entry point declares KERNEL_TYPE_AIV_ONLY.  Keep this code free
        // of ASCEND_IS_AIV conditionals: that macro expands to a constexpr
        // statement (not a boolean expression), so negating it is invalid in
        // AscendC.  More importantly, P7.1 has no AIC-side state to gate.
        candidateSeeds_ = reinterpret_cast<__gm__ int32_t*>(candidate_seeds);
        candidateCounts_ = reinterpret_cast<__gm__ int32_t*>(candidate_counts);
        candidateReadyEpoch_ = reinterpret_cast<__gm__ int64_t*>(candidate_ready_epoch);
        candidateStatus_ = reinterpret_cast<__gm__ int32_t*>(candidate_status);
        streamingEpoch_ = reinterpret_cast<__gm__ int64_t*>(streaming_epoch);
        sourceLens_ = reinterpret_cast<__gm__ int32_t*>(source_lens);
        kvObjectAddressDomain_ = reinterpret_cast<__gm__ int32_t*>(kv_object_address_domain);
        reqPoolEntries_ = reinterpret_cast<__gm__ int32_t*>(req_pool_entries);
        kvObjectMap_ = reinterpret_cast<__gm__ int32_t*>(kv_object_map);
        kvObjectLens_ = reinterpret_cast<__gm__ int32_t*>(kv_object_lens);
        kvObjectSlotsSnapshot_ = reinterpret_cast<__gm__ int32_t*>(kv_object_slots_snapshot);
        privateSourceToSlot_ = reinterpret_cast<__gm__ int32_t*>(private_source_to_slot);
        privateObjectToSlot_ = reinterpret_cast<__gm__ int32_t*>(private_object_to_slot);
        privateReady_ = reinterpret_cast<__gm__ int32_t*>(private_ready_table);
        privateArenaHead_ = reinterpret_cast<__gm__ int32_t*>(private_arena_head);
        privatePackedOutput_ = reinterpret_cast<__gm__ uint8_t*>(private_packed_output);
        controlGm_ = reinterpret_cast<__gm__ const kvaf_fused_get::KvafStreamingBatchGetControl*>(fused_get_control);
        queueLeases_ = reinterpret_cast<__gm__ int32_t*>(queue_leases);
        queueEpoch_ = reinterpret_cast<__gm__ int64_t*>(queue_epoch);
        queueGeneration_ = reinterpret_cast<__gm__ int64_t*>(queue_generation);
        queuePostSequence_ = reinterpret_cast<__gm__ int64_t*>(queue_post_sequence);
        queueState_ = reinterpret_cast<__gm__ int32_t*>(queue_state);
        queueQuarantine_ = reinterpret_cast<__gm__ int32_t*>(queue_quarantine);
        queueRequestSourceOffsets_ = reinterpret_cast<__gm__ int64_t*>(queue_request_source_offsets);
        queueRequestObjectOffsets_ = reinterpret_cast<__gm__ int64_t*>(queue_request_object_offsets);
        queueRequestOutputIndices_ = reinterpret_cast<__gm__ int32_t*>(queue_request_output_indices);
        queueRequestCount_ = reinterpret_cast<__gm__ int32_t*>(queue_request_count);
        privateReadyEpoch_ = reinterpret_cast<__gm__ int64_t*>(private_ready_epoch);
        privateReadyGeneration_ = reinterpret_cast<__gm__ int64_t*>(private_ready_generation);
        workerStats_ = reinterpret_cast<__gm__ int64_t*>(worker_stats);
        rowsPerEntry_ = reinterpret_cast<__gm__ int32_t*>(rows_per_entry);

        for (uint32_t slot = 0U; slot < kA5ScoreReceiptExecutorMicrobatchSlots; ++slot) {
            batchState_[slot] = 0U;
            batchCount_[slot] = 0U;
            batchReceipt_[slot] = 0U;
        }
        hasWave_ = false;
        terminalPublished_ = false;
        retainedQuarantinedWave_ = false;
        enabled_ = false;
        worker_ = GetBlockIdx();
        if (worker_ >= tiling_->workerCount || worker_ >= tiling_->workerStatsRows ||
            candidateSeeds_ == nullptr || candidateCounts_ == nullptr ||
            candidateReadyEpoch_ == nullptr || candidateStatus_ == nullptr ||
            streamingEpoch_ == nullptr || sourceLens_ == nullptr ||
            kvObjectAddressDomain_ == nullptr || reqPoolEntries_ == nullptr ||
            kvObjectMap_ == nullptr || kvObjectLens_ == nullptr ||
            kvObjectSlotsSnapshot_ == nullptr || privateSourceToSlot_ == nullptr ||
            privateObjectToSlot_ == nullptr || privateReady_ == nullptr ||
            privateArenaHead_ == nullptr || privatePackedOutput_ == nullptr ||
            controlGm_ == nullptr || queueLeases_ == nullptr || queueEpoch_ == nullptr ||
            queueGeneration_ == nullptr || queuePostSequence_ == nullptr ||
            queueState_ == nullptr || queueQuarantine_ == nullptr ||
            queueRequestSourceOffsets_ == nullptr || queueRequestObjectOffsets_ == nullptr ||
            queueRequestOutputIndices_ == nullptr || queueRequestCount_ == nullptr ||
            privateReadyEpoch_ == nullptr || privateReadyGeneration_ == nullptr ||
            workerStats_ == nullptr || rowsPerEntry_ == nullptr) {
            return;
        }
        epoch_ = static_cast<uint64_t>(ReadGmByPassDCache<int64_t>(streamingEpoch_));
        addressDomain_ = ReadGmByPassDCache<int32_t>(kvObjectAddressDomain_);
        DataSyncBarrier<MemDsbT::DDR>();
        if (addressDomain_ != kKvObjectAddressDomainMappedObject) {
            AddStat(kStatControlRejected, 1U);
            return;
        }
        if (!kvaf_fused_get::LoadKvafStreamingBatchGetControlFromGmForAicore(
                controlGm_, &control_) ||
            !kvaf_fused_get::IsKvafStreamingBatchGetControlValidForAicore(control_) ||
            control_.queue_count != tiling_->queueCount ||
            control_.queue_leases != reinterpret_cast<uint64_t>(queueLeases_) ||
            control_.output != reinterpret_cast<uint64_t>(privatePackedOutput_) ||
            control_.output_rows != tiling_->privatePackedRows ||
            control_.private_output_stride_bytes / kPackedKvRowBytes !=
                tiling_->privateArenaCapacity) {
            AddStat(kStatControlRejected, 1U);
            return;
        }

        // Fail-closed ordering is deliberate.  A malformed control block is
        // an ABI/ownership rejection, not a request to initialize UB or touch
        // any SIMT/HCCP helper.  Keeping all four UB allocations below this
        // return makes the invalid-control test a genuine no-transport path.
        if (pipe_ == nullptr) {
            // This cannot occur at the normal entry point, which owns a local
            // TPipe and passes it explicitly.  Refuse the task rather than
            // constructing a partially initialized executor.
            AddStat(kStatControlRejected, 1U);
            return;
        }
        pipe_->InitBuffer(seedBuffers_[0], kA5ScoreReceiptExecutorMicrobatchCapacity *
                                          kA5ScoreReceiptExecutorSeedWords * sizeof(int32_t));
        pipe_->InitBuffer(seedBuffers_[1], kA5ScoreReceiptExecutorMicrobatchCapacity *
                                          kA5ScoreReceiptExecutorSeedWords * sizeof(int32_t));
        pipe_->InitBuffer(leaseBuffer_, sizeof(kvaf_fused_get::KvafStreamingBatchGetLeaseResult));
        pipe_->InitBuffer(claimBuffer_, sizeof(PrivateClaimResult));
        // TBuf::Get returns LocalTensor, not an address.  Convert its physical
        // UB address explicitly; direct LocalTensor-to-pointer conversion is
        // rejected by the AscendC compiler and was the prior P7.1 build
        // failure.  The two typed seed arrays remain exactly 2 x 64 x 4.
        localSeeds_[0] = (__ubuf__ int32_t*)seedBuffers_[0].Get<int32_t>().GetPhyAddr();
        localSeeds_[1] = (__ubuf__ int32_t*)seedBuffers_[1].Get<int32_t>().GetPhyAddr();
        leaseResult_ =
            (__ubuf__ kvaf_fused_get::KvafStreamingBatchGetLeaseResult*)
                leaseBuffer_.Get<uint8_t>().GetPhyAddr();
        claimResult_ = (__ubuf__ PrivateClaimResult*)
            claimBuffer_.Get<uint8_t>().GetPhyAddr();
        if (localSeeds_[0] == nullptr || localSeeds_[1] == nullptr ||
            leaseResult_ == nullptr || claimResult_ == nullptr) {
            // A valid transport control with no usable local state must
            // degrade to exact Guaranteed Recall; it must never enter a
            // partially materialized PostOnly transaction.
            AddStat(kStatControlRejected, 1U);
            return;
        }
        enabled_ = true;
    }

    __aicore__ inline void Process()
    {
        // P7.1 is an AIV-only entry point; see Init for why no
        // ASCEND_IS_AIV runtime/constexpr branch appears here.
        if (!enabled_ || retainedQuarantinedWave_) {
            return;
        }
        for (uint32_t receipt = worker_; receipt < tiling_->receiptCount;
             receipt += tiling_->workerCount) {
            Progress(false);
            if (retainedQuarantinedWave_) {
                break;
            }
            ConsumeReceipt(receipt);
            if (retainedQuarantinedWave_) {
                break;
            }
            Progress(false);
            if (retainedQuarantinedWave_) {
                break;
            }
            TryPostReadyBatch(receipt, false);
            // This worker owns an ambiguous post-publication wave.  Do not
            // consume more candidate receipts; drain its real CQ transaction
            // immediately.  Other workers remain free to use other queues.
            if (retainedQuarantinedWave_) {
                break;
            }
        }
        DrainTerminal();
    }

private:
    // P7.1 is a standalone task: EVENT_ID0 is reserved only for its serial
    // PIPE_S <-> PIPE_V SIMT acquire/claim/release services.  It is never
    // shared with C8 or with a score-pipeline dependency.
    //
    // CANN's `set_flag`/`wait_flag` validates the argument through its exact
    // expression type.  A `constexpr` class member is `const event_t` at the
    // call site and is rejected on A5; keep a non-const `event_t` field so
    // both direct VF hand-offs and the KVAF AIC wrappers receive the ABI type
    // CANN requires.
    event_t simtEvent_ = static_cast<event_t>(EVENT_ID0);
    // Local slot states are deliberately independent of queue ownership.
    static constexpr uint32_t kBatchFree = 0U;
    static constexpr uint32_t kBatchReady = 1U;

    __aicore__ inline uint64_t WorkerStatOffset(uint32_t column) const
    {
        return static_cast<uint64_t>(worker_) * tiling_->workerStatsColumns + column;
    }

    __aicore__ inline void AddStat(uint32_t column, uint64_t delta)
    {
        if (workerStats_ == nullptr || column >= tiling_->workerStatsColumns) {
            return;
        }
        const uint64_t offset = WorkerStatOffset(column);
        const int64_t current = ReadGmByPassDCache<int64_t>(workerStats_ + offset);
        WriteGmByPassDCache<int64_t>(workerStats_ + offset,
                                     current + static_cast<int64_t>(delta));
    }

    __aicore__ inline uint32_t FindFreeBatch() const
    {
        for (uint32_t slot = 0U; slot < kA5ScoreReceiptExecutorMicrobatchSlots; ++slot) {
            if (batchState_[slot] == kBatchFree) {
                return slot;
            }
        }
        return kA5ScoreReceiptExecutorMicrobatchSlots;
    }

    __aicore__ inline uint32_t FindReadyBatch() const
    {
        for (uint32_t slot = 0U; slot < kA5ScoreReceiptExecutorMicrobatchSlots; ++slot) {
            if (batchState_[slot] == kBatchReady && batchCount_[slot] != 0U) {
                return slot;
            }
        }
        return kA5ScoreReceiptExecutorMicrobatchSlots;
    }

    __aicore__ inline bool AnyQueueMayBeFree() const
    {
        for (uint32_t queue = 0U; queue < tiling_->queueCount; ++queue) {
            if (ReadGmByPassDCache<int32_t>(queueLeases_ + queue) == 0 &&
                ReadGmByPassDCache<int32_t>(queueQuarantine_ + queue) == 0) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline __gm__ int32_t* Candidate(uint32_t receipt,
                                                  uint32_t index) const
    {
        return candidateSeeds_ +
            (static_cast<uint64_t>(receipt) * tiling_->candidateCapacity + index) *
                tiling_->candidateSeedWords;
    }

    __aicore__ inline void ConsumeReceipt(uint32_t receipt)
    {
        AddStat(kStatReceipts, 1U);
        const uint64_t readyEpoch = static_cast<uint64_t>(
            ReadGmByPassDCache<int64_t>(candidateReadyEpoch_ + receipt));
        if (readyEpoch != epoch_) {
            AddStat(kStatStaleReceipt, 1U);
            return;
        }
        // CandidateSeed producer writes payload first and epoch last.  This
        // is the consumer acquire edge: no count/status/seed load may move
        // in front of the matching epoch observation. Recheck the epoch
        // after the barrier to reject a producer reuse race conservatively.
        DataSyncBarrier<MemDsbT::DDR>();
        if (static_cast<uint64_t>(ReadGmByPassDCache<int64_t>(
                candidateReadyEpoch_ + receipt)) != epoch_) {
            AddStat(kStatStaleReceipt, 1U);
            return;
        }
        const int32_t status = ReadGmByPassDCache<int32_t>(candidateStatus_ + receipt);
        const int32_t countValue = ReadGmByPassDCache<int32_t>(candidateCounts_ + receipt);
        if (status != kCandidateStatusOk ||
            countValue < 0 || static_cast<uint32_t>(countValue) > tiling_->candidateCapacity) {
            AddStat(kStatStaleReceipt, 1U);
            return;
        }
        uint32_t start = 0U;
        const uint32_t count = static_cast<uint32_t>(countValue);
        while (start < count) {
            uint32_t slot = FindFreeBatch();
            if (slot == kA5ScoreReceiptExecutorMicrobatchSlots) {
                Progress(false);
                TryPostReadyBatch(receipt, false);
                slot = FindFreeBatch();
            }
            if (slot == kA5ScoreReceiptExecutorMicrobatchSlots) {
                AddStat(kStatSeedsDeferred, count - start);
                return;
            }
            const uint32_t chunk =
                (count - start) < tiling_->microbatchCapacity
                    ? (count - start) : tiling_->microbatchCapacity;
            for (uint32_t item = 0U; item < chunk; ++item) {
                __gm__ int32_t* const source = Candidate(receipt, start + item);
                __ubuf__ int32_t* const destination = localSeeds_[slot] +
                    static_cast<uint64_t>(item) * tiling_->candidateSeedWords;
                for (uint32_t word = 0U; word < tiling_->candidateSeedWords; ++word) {
                    destination[word] = ReadGmByPassDCache<int32_t>(source + word);
                }
            }
            batchState_[slot] = kBatchReady;
            batchCount_[slot] = chunk;
            batchReceipt_[slot] = receipt;
            AddStat(kStatSeedsLoaded, chunk);
            start += chunk;
            // The second local batch may fill while the first HCCP wave is
            // in flight.  It is a bounded scheduling buffer, not a core role.
            TryPostReadyBatch(receipt, false);
        }
    }

    __aicore__ inline bool ClaimPrivateSlot(uint64_t sourceOffset,
                                             uint64_t objectOffset,
                                             uint32_t queueId)
    {
        claimResult_->success = 0U;
        set_flag(PIPE_S, PIPE_V, simtEvent_);
        wait_flag(PIPE_S, PIPE_V, simtEvent_);
        AscendC::Simt::VF_CALL<ClaimPrivateSlotSimt>(
            AscendC::Simt::Dim3{1U, 1U, 1U}, privateSourceToSlot_,
            privateObjectToSlot_, privateReady_, privateArenaHead_,
            privateReadyEpoch_, privateReadyGeneration_, sourceOffset,
            objectOffset, queueId, tiling_->privateArenaCapacity,
            tiling_->privatePackedRows, epoch_, waveGeneration_, claimResult_);
        set_flag(PIPE_V, PIPE_S, simtEvent_);
        wait_flag(PIPE_V, PIPE_S, simtEvent_);
        return claimResult_->success != 0U;
    }

    __aicore__ inline bool PublishPrivateClaim(uint64_t sourceOffset,
                                                uint64_t objectOffset,
                                                int32_t outputIndex)
    {
        claimResult_->success = 0U;
        set_flag(PIPE_S, PIPE_V, simtEvent_);
        wait_flag(PIPE_S, PIPE_V, simtEvent_);
        AscendC::Simt::VF_CALL<PublishPrivateClaimSimt>(
            AscendC::Simt::Dim3{1U, 1U, 1U}, privateSourceToSlot_,
            privateObjectToSlot_, sourceOffset, objectOffset, outputIndex,
            claimResult_);
        set_flag(PIPE_V, PIPE_S, simtEvent_);
        wait_flag(PIPE_V, PIPE_S, simtEvent_);
        return claimResult_->success != 0U;
    }

    __aicore__ inline void RollbackPrivateClaim(uint64_t sourceOffset,
                                                 uint64_t objectOffset,
                                                 int32_t outputIndex)
    {
        if (outputIndex < 0) {
            return;
        }
        claimResult_->success = 0U;
        set_flag(PIPE_S, PIPE_V, simtEvent_);
        wait_flag(PIPE_S, PIPE_V, simtEvent_);
        AscendC::Simt::VF_CALL<RollbackPrivateClaimSimt>(
            AscendC::Simt::Dim3{1U, 1U, 1U}, privateSourceToSlot_,
            privateObjectToSlot_, privateReady_, sourceOffset, objectOffset,
            static_cast<uint64_t>(outputIndex), claimResult_);
        set_flag(PIPE_V, PIPE_S, simtEvent_);
        wait_flag(PIPE_V, PIPE_S, simtEvent_);
        WriteGmByPassDCache<int64_t>(privateReadyEpoch_ + outputIndex, 0);
        WriteGmByPassDCache<int64_t>(privateReadyGeneration_ + outputIndex, 0);
    }

    __aicore__ inline bool ValidateDescriptor(
        const kvaf_fused_get::KvafStreamingBatchGetDescriptorScratch& descriptor,
        uint32_t queueId) const
    {
        const uint64_t privateRows =
            control_.private_output_stride_bytes / kPackedKvRowBytes;
        return descriptor.address_table != 0U && descriptor.valid_table != 0U &&
            descriptor.indices != 0U && descriptor.offsets != 0U &&
            descriptor.mask != 0U && descriptor.output_indices != 0U &&
            descriptor.table_capacity != 0U && descriptor.request_capacity >=
                tiling_->microbatchCapacity &&
            descriptor.request_capacity <= kvaf_fused_get::kDenseRequestCapacity &&
            control_.private_output_stride_bytes % kPackedKvRowBytes == 0U &&
            descriptor.private_output_row_base == queueId * privateRows;
    }

    __aicore__ inline void SetLifecycleFilling(uint32_t queueId)
    {
        waveEpoch_ = epoch_;
        waveGeneration_ = static_cast<uint64_t>(
            ReadGmByPassDCache<int64_t>(queueGeneration_ + queueId)) + 1U;
        wavePostSequence_ = static_cast<uint64_t>(
            ReadGmByPassDCache<int64_t>(queuePostSequence_ + queueId)) + 1U;
        WriteGmByPassDCache<int64_t>(queueEpoch_ + queueId,
                                     static_cast<int64_t>(waveEpoch_));
        WriteGmByPassDCache<int64_t>(queueGeneration_ + queueId,
                                     static_cast<int64_t>(waveGeneration_));
        WriteGmByPassDCache<int64_t>(queuePostSequence_ + queueId,
                                     static_cast<int64_t>(wavePostSequence_));
        WriteGmByPassDCache<int32_t>(queueState_ + queueId, kQueueFilling);
    }

    __aicore__ inline bool QueueLifecycleMatches(uint32_t queueId) const
    {
        return ReadGmByPassDCache<int64_t>(queueEpoch_ + queueId) ==
                   static_cast<int64_t>(waveEpoch_) &&
            ReadGmByPassDCache<int64_t>(queueGeneration_ + queueId) ==
                   static_cast<int64_t>(waveGeneration_) &&
            ReadGmByPassDCache<int64_t>(queuePostSequence_ + queueId) ==
                   static_cast<int64_t>(wavePostSequence_) &&
            ReadGmByPassDCache<int32_t>(queueState_ + queueId) == kQueuePosted;
    }

    // Quarantine is queue-local.  Only a post-publication ambiguity retains
    // this worker's local wave; a malformed free queue is released (if the
    // CAS still proves ownership) and other workers/queues keep progressing.
    __aicore__ inline void QuarantineQueue(uint32_t queueId,
                                            bool retain_local_wave)
    {
        WriteGmByPassDCache<int32_t>(queueQuarantine_ + queueId, 1);
        WriteGmByPassDCache<int32_t>(queueState_ + queueId, kQueueQuarantined);
        DataSyncBarrier<MemDsbT::DDR>();
        if (retain_local_wave) {
            retainedQuarantinedWave_ = true;
        }
        AddStat(kStatLeaseQuarantined, 1U);
    }

    __aicore__ inline bool ReleaseTerminalWave()
    {
        const uint32_t queueId = wave_.lease.queue_id;
        const bool alreadyQuarantined =
            ReadGmByPassDCache<int32_t>(queueQuarantine_ + queueId) != 0;
        // Publish FREE before releasing the KVAF owner word. A new worker
        // cannot acquire while the lease is nonzero, so this ordering avoids
        // a fresh owner observing stale CQ_TERMINAL state.
        if (!alreadyQuarantined) {
            WriteGmByPassDCache<int32_t>(queueState_ + queueId, kQueueFree);
            DataSyncBarrier<MemDsbT::DDR>();
        }
        if (kvaf_fused_get::ReleaseKvafStreamingBatchGetQueueFromAicore(
                controlGm_, leaseResult_, simtEvent_, &wave_)) {
            // A control/identity failure may deliberately quarantine an
            // otherwise releasable lease.  Releasing the KVAF owner does not
            // make that graph-visible queue trustworthy for this forward.
            if (ReadGmByPassDCache<int32_t>(queueQuarantine_ + queueId) != 0) {
                WriteGmByPassDCache<int32_t>(queueState_ + queueId,
                                             kQueueQuarantined);
            }
            AddStat(kStatLeaseReleased, 1U);
            return true;
        }
        // Failed release means the owner word cannot prove exclusive state.
        // Preserve the local wave and the nonzero KVAF lease for runtime
        // terminal drain; never silently clear `hasWave_` here.
        QuarantineQueue(queueId, true);
        return false;
    }

    __aicore__ inline bool ReleaseUnpostedLease(
        const kvaf_fused_get::KvafStreamingBatchGetLease& lease)
    {
        wave_.lease = lease;
        wave_.lease.state = kvaf_fused_get::kStreamingBatchGetLeaseTerminal;
        wave_.transaction.finalized = 1U;
        const bool released = ReleaseTerminalWave();
        if (released) {
            hasWave_ = false;
        }
        return released;
    }

    __aicore__ inline void RollbackLedger(uint32_t queueId, uint32_t count)
    {
        const uint64_t base = static_cast<uint64_t>(queueId) * tiling_->microbatchCapacity;
        for (uint32_t request = 0U; request < count; ++request) {
            const uint64_t sourceOffset = static_cast<uint64_t>(
                ReadGmByPassDCache<int64_t>(queueRequestSourceOffsets_ + base + request));
            const uint64_t objectOffset = static_cast<uint64_t>(
                ReadGmByPassDCache<int64_t>(queueRequestObjectOffsets_ + base + request));
            const int32_t outputIndex = ReadGmByPassDCache<int32_t>(
                queueRequestOutputIndices_ + base + request);
            RollbackPrivateClaim(sourceOffset, objectOffset, outputIndex);
        }
        WriteGmByPassDCache<int32_t>(queueRequestCount_ + queueId, 0);
    }

    __aicore__ inline bool StageOneSeed(
        __ubuf__ int32_t* seed, uint32_t queueId,
        const kvaf_fused_get::KvafStreamingBatchGetDescriptorScratch& descriptor,
        uint32_t* stagedCount)
    {
        if (seed == nullptr || stagedCount == nullptr) {
            return false;
        }
        const int32_t rowValue = seed[0];
        const int32_t sourceValue = seed[1];
        const int32_t objectValue = seed[2];
        if (rowValue < 0 || sourceValue < 0 || objectValue < 0 ||
            static_cast<uint32_t>(rowValue) >= tiling_->rowCount ||
            static_cast<uint32_t>(sourceValue) >= tiling_->sourceCapacity) {
            return false;
        }
        const uint32_t row = static_cast<uint32_t>(rowValue);
        const uint32_t source = static_cast<uint32_t>(sourceValue);
        const int32_t poolValue = ReadGmByPassDCache<int32_t>(reqPoolEntries_ + row);
        const int32_t sourceLen = ReadGmByPassDCache<int32_t>(sourceLens_ + row);
        const int32_t objectLen = ReadGmByPassDCache<int32_t>(kvObjectLens_ + row);
        // `source_lens[row]` is the active score-source prefix for this
        // forward; S may fit the graph-static capacity but still refer to a
        // stale padded tail on graph replay. P7.1 accepts only the explicit
        // mapped-object domain: O must equal kv_object_map[P,S]. It never
        // guesses a raw-token conversion before descriptor fill.
        if (addressDomain_ != kKvObjectAddressDomainMappedObject || sourceLen < 0 ||
            static_cast<uint32_t>(sourceLen) > tiling_->sourceCapacity ||
            source >= static_cast<uint32_t>(sourceLen) ||
            poolValue < 0 || static_cast<uint32_t>(poolValue) >= tiling_->poolSize ||
            objectLen < 0 || static_cast<uint32_t>(objectLen) > tiling_->kvObjectCapacity ||
            static_cast<uint32_t>(objectValue) >= static_cast<uint32_t>(objectLen) ||
            ReadGmByPassDCache<int32_t>(
                kvObjectMap_ + static_cast<uint64_t>(poolValue) *
                    tiling_->sourceCapacity + source) != objectValue ||
            ReadGmByPassDCache<int32_t>(
                kvObjectSlotsSnapshot_ + static_cast<uint64_t>(poolValue) *
                    tiling_->kvObjectCapacity + static_cast<uint32_t>(objectValue)) >= 0) {
            return false;
        }
        const uint64_t sourceOffset = static_cast<uint64_t>(row) *
            tiling_->sourceCapacity + source;
        const uint64_t objectOffset = static_cast<uint64_t>(poolValue) *
            tiling_->kvObjectCapacity + static_cast<uint32_t>(objectValue);
        if (!ClaimPrivateSlot(sourceOffset, objectOffset, queueId)) {
            return false;
        }
        if (claimResult_->kind == kClaimShared) {
            AddStat(kStatPrivateShared, 1U);
            return false;
        }
        if (claimResult_->kind != kClaimNew ||
            *stagedCount >= tiling_->microbatchCapacity) {
            if (claimResult_->kind == kClaimNew) {
                RollbackPrivateClaim(sourceOffset, objectOffset,
                                     claimResult_->output_index);
            }
            return false;
        }
        const int32_t rowsPerEntryValue =
            ReadGmByPassDCache<int32_t>(rowsPerEntry_);
        if (rowsPerEntryValue <= 0) {
            RollbackPrivateClaim(sourceOffset, objectOffset,
                                 claimResult_->output_index);
            return false;
        }
        const uint32_t rowsPerEntry = static_cast<uint32_t>(rowsPerEntryValue);
        const uint32_t tableIndex = static_cast<uint32_t>(objectValue) /
            rowsPerEntry;
        if (
            tableIndex >= descriptor.table_capacity) {
            RollbackPrivateClaim(sourceOffset, objectOffset,
                                 claimResult_->output_index);
            return false;
        }
        __gm__ int32_t* const indices = reinterpret_cast<__gm__ int32_t*>(descriptor.indices);
        __gm__ int32_t* const offsets = reinterpret_cast<__gm__ int32_t*>(descriptor.offsets);
        __gm__ int32_t* const outputIndices =
            reinterpret_cast<__gm__ int32_t*>(descriptor.output_indices);
        __gm__ uint8_t* const mask = reinterpret_cast<__gm__ uint8_t*>(descriptor.mask);
        const uint32_t request = *stagedCount;
        indices[request] = static_cast<int32_t>(tableIndex);
        offsets[request] = static_cast<int32_t>(
            (static_cast<uint32_t>(objectValue) %
             rowsPerEntry) *
            kPackedKvRowBytes);
        outputIndices[request] = claimResult_->output_index;
        mask[request] = 1U;
        const uint64_t ledger = static_cast<uint64_t>(queueId) *
            tiling_->microbatchCapacity + request;
        WriteGmByPassDCache<int64_t>(queueRequestSourceOffsets_ + ledger,
                                     static_cast<int64_t>(sourceOffset));
        WriteGmByPassDCache<int64_t>(queueRequestObjectOffsets_ + ledger,
                                     static_cast<int64_t>(objectOffset));
        WriteGmByPassDCache<int32_t>(queueRequestOutputIndices_ + ledger,
                                     claimResult_->output_index);
        ++(*stagedCount);
        AddStat(kStatPrivateNew, 1U);
        return true;
    }

    __aicore__ inline bool TryPostReadyBatch(uint32_t hint, bool terminal)
    {
        if (!enabled_ || retainedQuarantinedWave_ || hasWave_) {
            return false;
        }
        const uint32_t slot = FindReadyBatch();
        if (slot == kA5ScoreReceiptExecutorMicrobatchSlots) {
            return false;
        }
        if (!AnyQueueMayBeFree()) {
            AddStat(kStatQueueBusy, 1U);
            return false;
        }
        kvaf_fused_get::KvafStreamingBatchGetLease lease{};
        if (!kvaf_fused_get::TryAcquireKvafStreamingBatchGetQueueFromAicore(
                controlGm_, worker_, hint, leaseResult_, simtEvent_, &lease)) {
            AddStat(kStatQueueBusy, 1U);
            return false;
        }
        const uint32_t queueId = lease.queue_id;
        AddStat(kStatLeaseAcquired, 1U);
        if (ReadGmByPassDCache<int32_t>(queueQuarantine_ + queueId) != 0 ||
            ReadGmByPassDCache<int32_t>(queueState_ + queueId) != kQueueFree) {
            QuarantineQueue(queueId, false);
            ReleaseUnpostedLease(lease);
            return false;
        }
        SetLifecycleFilling(queueId);
        kvaf_fused_get::KvafStreamingBatchGetDescriptorScratch descriptor{};
        if (!kvaf_fused_get::LoadKvafStreamingBatchGetQueueTemplate(control_, lease,
                                                                      &descriptor) ||
            !ValidateDescriptor(descriptor, queueId)) {
            AddStat(kStatControlRejected, 1U);
            QuarantineQueue(queueId, false);
            ReleaseUnpostedLease(lease);
            return false;
        }
        uint32_t staged = 0U;
        for (uint32_t item = 0U; item < batchCount_[slot] &&
             staged < tiling_->microbatchCapacity; ++item) {
            StageOneSeed(localSeeds_[slot] +
                static_cast<uint64_t>(item) * tiling_->candidateSeedWords,
                queueId, descriptor, &staged);
        }
        batchState_[slot] = kBatchFree;
        batchCount_[slot] = 0U;
        if (staged == 0U) {
            WriteGmByPassDCache<int32_t>(queueRequestCount_ + queueId, 0);
            WriteGmByPassDCache<int32_t>(queueState_ + queueId, kQueueFree);
            ReleaseUnpostedLease(lease);
            return false;
        }
        descriptor.active_request_count = staged;
        WriteGmByPassDCache<int32_t>(queueRequestCount_ + queueId,
                                     static_cast<int32_t>(staged));
        // Ledger and descriptor payload are fully materialized before the
        // KVAF helper's own descriptor publication barrier and SQ doorbell.
        DataSyncBarrier<MemDsbT::DDR>();
        if (!kvaf_fused_get::PublishKvafStreamingBatchGetDescriptorScratch(
                control_, lease, descriptor) ||
            !kvaf_fused_get::BeginKvafStreamingBatchGet(control_, lease, &wave_)) {
            RollbackLedger(queueId, staged);
            if (wave_.lease.state == kvaf_fused_get::kStreamingBatchGetLeaseTerminal) {
                ReleaseTerminalWave();
            } else {
                ReleaseUnpostedLease(lease);
            }
            return false;
        }
        // PostOnly has handed the queue to Direct-HCCP.  From this point a
        // publication anomaly must retain a terminally drainable local wave;
        // it must never be mistaken for an unposted lease.
        hasWave_ = true;
        terminalPublished_ = false;
        waveFailurePending_ = false;
        for (uint32_t request = 0U; request < staged; ++request) {
            const uint64_t ledger = static_cast<uint64_t>(queueId) *
                tiling_->microbatchCapacity + request;
            const uint64_t sourceOffset = static_cast<uint64_t>(
                ReadGmByPassDCache<int64_t>(queueRequestSourceOffsets_ + ledger));
            const uint64_t objectOffset = static_cast<uint64_t>(
                ReadGmByPassDCache<int64_t>(queueRequestObjectOffsets_ + ledger));
            const int32_t outputIndex = ReadGmByPassDCache<int32_t>(
                queueRequestOutputIndices_ + ledger);
            if (!PublishPrivateClaim(sourceOffset, objectOffset, outputIndex)) {
                // The transfer may already be in flight.  Preserve its lease,
                // forbid reuse, and force later exact Recall rather than
                // publishing an unverifiable private mapping. Do not mark
                // private state FAILED or clear -2 claims until the real CQ
                // reaches terminal state below.
                waveFailurePending_ = true;
                QuarantineQueue(queueId, true);
                return false;
            }
        }
        WriteGmByPassDCache<int32_t>(queueState_ + queueId, kQueuePosted);
        DataSyncBarrier<MemDsbT::DDR>();
        AddStat(kStatDoorbellPosted, 1U);
        (void)terminal;
        return true;
    }

    __aicore__ inline void ClearFailedLedgerMappings()
    {
        const uint32_t queueId = wave_.lease.queue_id;
        const int32_t count = ReadGmByPassDCache<int32_t>(queueRequestCount_ + queueId);
        if (count < 0 || static_cast<uint32_t>(count) > tiling_->microbatchCapacity) {
            QuarantineQueue(queueId, true);
            return;
        }
        const uint64_t base = static_cast<uint64_t>(queueId) * tiling_->microbatchCapacity;
        for (uint32_t request = 0U; request < static_cast<uint32_t>(count); ++request) {
            const uint64_t sourceOffset = static_cast<uint64_t>(
                ReadGmByPassDCache<int64_t>(queueRequestSourceOffsets_ + base + request));
            const uint64_t objectOffset = static_cast<uint64_t>(
                ReadGmByPassDCache<int64_t>(queueRequestObjectOffsets_ + base + request));
            const int32_t outputIndex = ReadGmByPassDCache<int32_t>(
                queueRequestOutputIndices_ + base + request);
            claimResult_->success = 0U;
            set_flag(PIPE_S, PIPE_V, simtEvent_);
            wait_flag(PIPE_S, PIPE_V, simtEvent_);
            AscendC::Simt::VF_CALL<ClearTerminalFailedMappingSimt>(
                AscendC::Simt::Dim3{1U, 1U, 1U}, privateSourceToSlot_,
                privateObjectToSlot_, sourceOffset, objectOffset, outputIndex,
                claimResult_);
            set_flag(PIPE_V, PIPE_S, simtEvent_);
            wait_flag(PIPE_V, PIPE_S, simtEvent_);
        }
        DataSyncBarrier<MemDsbT::DDR>();
    }

    __aicore__ inline void PublishTerminalPrivateState(bool ready)
    {
        const uint32_t queueId = wave_.lease.queue_id;
        const int32_t count = ReadGmByPassDCache<int32_t>(queueRequestCount_ + queueId);
        if (count < 0 || static_cast<uint32_t>(count) > tiling_->microbatchCapacity) {
            QuarantineQueue(queueId, true);
            return;
        }
        const uint64_t base = static_cast<uint64_t>(queueId) * tiling_->microbatchCapacity;
        for (uint32_t request = 0U; request < static_cast<uint32_t>(count); ++request) {
            const int32_t outputIndex = ReadGmByPassDCache<int32_t>(
                queueRequestOutputIndices_ + base + request);
            if (outputIndex < 0 || static_cast<uint32_t>(outputIndex) >=
                                   tiling_->privatePackedRows) {
                QuarantineQueue(queueId, true);
                return;
            }
            WriteGmByPassDCache<int32_t>(privateReady_ + outputIndex,
                                         ready ? kPrivateReady : kPrivateFailed);
            WriteGmByPassDCache<int64_t>(privateReadyEpoch_ + outputIndex,
                                         static_cast<int64_t>(waveEpoch_));
            WriteGmByPassDCache<int64_t>(privateReadyGeneration_ + outputIndex,
                                         static_cast<int64_t>(waveGeneration_));
        }
        DataSyncBarrier<MemDsbT::DDR>();
        if (!ready) {
            ClearFailedLedgerMappings();
        }
    }

    __aicore__ inline void Progress(bool finish)
    {
        if (!hasWave_ || retainedQuarantinedWave_) {
            return;
        }
        if (finish) {
            AddStat(kStatFinish, 1U);
            kvaf_fused_get::FinishKvafStreamingBatchGet(&wave_);
        } else {
            AddStat(kStatCqPoll, 1U);
            kvaf_fused_get::TryRetireKvafStreamingBatchGet(&wave_);
        }
        if (wave_.lease.state != kvaf_fused_get::kStreamingBatchGetLeaseTerminal) {
            return;
        }
        const uint32_t queueId = wave_.lease.queue_id;
        if (!terminalPublished_) {
            if (!QueueLifecycleMatches(queueId)) {
                PublishTerminalPrivateState(false);
                QuarantineQueue(queueId, true);
                return;
            }
            kvaf_fused_get::KvafStreamingBatchGetReady ready{};
            const bool payloadReady = !waveFailurePending_ &&
                kvaf_fused_get::GetKvafStreamingBatchGetReady(wave_, &ready);
            WriteGmByPassDCache<int32_t>(queueState_ + queueId, kQueueCqTerminal);
            PublishTerminalPrivateState(payloadReady);
            terminalPublished_ = true;
            AddStat(payloadReady ? kStatCqReady : kStatCqFailed, 1U);
        }
        if (retainedQuarantinedWave_) {
            return;
        }
        if (ReleaseTerminalWave()) {
            hasWave_ = false;
            terminalPublished_ = false;
            WriteGmByPassDCache<int32_t>(queueRequestCount_ + queueId, 0);
        }
    }

    __aicore__ inline void DrainTerminal()
    {
        if (hasWave_ && retainedQuarantinedWave_) {
            // Quarantine prevents reuse/release, not terminal CQ draining.
            // Complete the real transaction so runtime teardown observes a
            // terminal failed/retired wave rather than an in-flight orphan.
            AddStat(kStatFinish, 1U);
            kvaf_fused_get::FinishKvafStreamingBatchGet(&wave_);
            // Quarantine can occur after PostOnly but before Progress()
            // normally publishes terminal state.  Conservatively fail every
            // private row in this wave after the real terminal drain; never
            // leave a PENDING row that QSFA could mistake for usable data.
            if (!terminalPublished_) {
                PublishTerminalPrivateState(false);
                terminalPublished_ = true;
                AddStat(kStatCqFailed, 1U);
            }
            WriteGmByPassDCache<int32_t>(queueState_ + wave_.lease.queue_id,
                                         kQueueQuarantined);
            DataSyncBarrier<MemDsbT::DDR>();
            return;
        }
        if (hasWave_) {
            Progress(true);
        }
        for (uint32_t attempts = 0U;
             attempts < kA5ScoreReceiptExecutorMicrobatchSlots && !retainedQuarantinedWave_;
             ++attempts) {
            const uint32_t ready = FindReadyBatch();
            if (ready == kA5ScoreReceiptExecutorMicrobatchSlots) {
                break;
            }
            if (!TryPostReadyBatch(batchReceipt_[ready], true)) {
                // No terminal spin: missing speculative work stays for exact
                // Guaranteed Recall.  The buffer is local and can be dropped.
                AddStat(kStatSeedsDeferred, batchCount_[ready]);
                batchState_[ready] = kBatchFree;
                batchCount_[ready] = 0U;
                continue;
            }
            Progress(true);
        }
    }

    // Follow the standard AscendC ownership pattern: the entry point owns a
    // stack-local TPipe for the invocation and the kernel object only borrows
    // it.  A class-owned TPipe is nonstandard and can leave UB bookkeeping
    // alive after a fail-closed early return.
    TPipe* pipe_ = nullptr;
    const A5ScoreReceiptSpeculativeExecutorTilingData* tiling_;
    TBuf<TPosition::VECCALC> seedBuffers_[kA5ScoreReceiptExecutorMicrobatchSlots];
    TBuf<TPosition::VECCALC> leaseBuffer_;
    TBuf<TPosition::VECCALC> claimBuffer_;
    __ubuf__ int32_t* localSeeds_[kA5ScoreReceiptExecutorMicrobatchSlots]{};
    __ubuf__ kvaf_fused_get::KvafStreamingBatchGetLeaseResult* leaseResult_ = nullptr;
    __ubuf__ PrivateClaimResult* claimResult_ = nullptr;

    __gm__ int32_t* candidateSeeds_ = nullptr;
    __gm__ int32_t* candidateCounts_ = nullptr;
    __gm__ int64_t* candidateReadyEpoch_ = nullptr;
    __gm__ int32_t* candidateStatus_ = nullptr;
    __gm__ int64_t* streamingEpoch_ = nullptr;
    __gm__ int32_t* sourceLens_ = nullptr;
    __gm__ int32_t* kvObjectAddressDomain_ = nullptr;
    __gm__ int32_t* reqPoolEntries_ = nullptr;
    __gm__ int32_t* kvObjectMap_ = nullptr;
    __gm__ int32_t* kvObjectLens_ = nullptr;
    __gm__ int32_t* kvObjectSlotsSnapshot_ = nullptr;
    __gm__ int32_t* privateSourceToSlot_ = nullptr;
    __gm__ int32_t* privateObjectToSlot_ = nullptr;
    __gm__ int32_t* privateReady_ = nullptr;
    __gm__ int32_t* privateArenaHead_ = nullptr;
    __gm__ uint8_t* privatePackedOutput_ = nullptr;
    __gm__ const kvaf_fused_get::KvafStreamingBatchGetControl* controlGm_ = nullptr;
    __gm__ int32_t* queueLeases_ = nullptr;
    __gm__ int64_t* queueEpoch_ = nullptr;
    __gm__ int64_t* queueGeneration_ = nullptr;
    __gm__ int64_t* queuePostSequence_ = nullptr;
    __gm__ int32_t* queueState_ = nullptr;
    __gm__ int32_t* queueQuarantine_ = nullptr;
    __gm__ int64_t* queueRequestSourceOffsets_ = nullptr;
    __gm__ int64_t* queueRequestObjectOffsets_ = nullptr;
    __gm__ int32_t* queueRequestOutputIndices_ = nullptr;
    __gm__ int32_t* queueRequestCount_ = nullptr;
    __gm__ int64_t* privateReadyEpoch_ = nullptr;
    __gm__ int64_t* privateReadyGeneration_ = nullptr;
    __gm__ int64_t* workerStats_ = nullptr;
    __gm__ int32_t* rowsPerEntry_ = nullptr;

    kvaf_fused_get::KvafStreamingBatchGetControl control_{};
    kvaf_fused_get::KvafStreamingBatchGetWave wave_{};
    uint32_t batchState_[kA5ScoreReceiptExecutorMicrobatchSlots]{};
    uint32_t batchCount_[kA5ScoreReceiptExecutorMicrobatchSlots]{};
    uint32_t batchReceipt_[kA5ScoreReceiptExecutorMicrobatchSlots]{};
    uint32_t worker_ = 0U;
    uint64_t epoch_ = 0U;
    int32_t addressDomain_ = -1;
    uint64_t waveEpoch_ = 0U;
    uint64_t waveGeneration_ = 0U;
    uint64_t wavePostSequence_ = 0U;
    bool enabled_ = false;
    bool hasWave_ = false;
    bool terminalPublished_ = false;
    bool waveFailurePending_ = false;
    bool retainedQuarantinedWave_ = false;
};
}  // namespace

extern "C" __global__ __aicore__ void a5_score_receipt_speculative_executor(
    GM_ADDR candidate_seeds, GM_ADDR candidate_counts,
    GM_ADDR candidate_ready_epoch, GM_ADDR candidate_status,
    GM_ADDR streaming_epoch, GM_ADDR source_lens,
    GM_ADDR kv_object_address_domain, GM_ADDR req_pool_entries, GM_ADDR kv_object_map,
    GM_ADDR kv_object_lens, GM_ADDR kv_object_slots_snapshot,
    GM_ADDR private_source_to_slot, GM_ADDR private_object_to_slot,
    GM_ADDR private_ready_table, GM_ADDR private_arena_head,
    GM_ADDR private_packed_output, GM_ADDR fused_get_control,
    GM_ADDR queue_leases, GM_ADDR queue_epoch, GM_ADDR queue_generation,
    GM_ADDR queue_post_sequence, GM_ADDR queue_state, GM_ADDR queue_quarantine,
    GM_ADDR queue_request_source_offsets, GM_ADDR queue_request_object_offsets,
    GM_ADDR queue_request_output_indices, GM_ADDR queue_request_count,
    GM_ADDR private_ready_epoch, GM_ADDR private_ready_generation,
    GM_ADDR worker_stats, GM_ADDR rows_per_entry,
    GM_ADDR private_source_to_slot_ref, GM_ADDR private_object_to_slot_ref,
    GM_ADDR private_ready_table_ref, GM_ADDR private_arena_head_ref,
    GM_ADDR private_packed_output_ref, GM_ADDR queue_leases_ref,
    GM_ADDR queue_epoch_ref, GM_ADDR queue_generation_ref,
    GM_ADDR queue_post_sequence_ref, GM_ADDR queue_state_ref,
    GM_ADDR queue_quarantine_ref, GM_ADDR queue_request_source_offsets_ref,
    GM_ADDR queue_request_object_offsets_ref,
    GM_ADDR queue_request_output_indices_ref, GM_ADDR queue_request_count_ref,
    GM_ADDR private_ready_epoch_ref, GM_ADDR private_ready_generation_ref,
    GM_ADDR worker_stats_ref, GM_ADDR workspace, GM_ADDR tiling)
{
    // CANN keeps Ref ports in the kernel ABI as trailing `*_ref` GM addresses,
    // even though GetWorkspaceSize receives only the corresponding mutable
    // input descriptor. They are the same physical state and this kernel
    // writes through the input positions, so the trailing aliases are only
    // retained to match the generated kernel calling convention.
    (void)private_source_to_slot_ref;
    (void)private_object_to_slot_ref;
    (void)private_ready_table_ref;
    (void)private_arena_head_ref;
    (void)private_packed_output_ref;
    (void)queue_leases_ref;
    (void)queue_epoch_ref;
    (void)queue_generation_ref;
    (void)queue_post_sequence_ref;
    (void)queue_state_ref;
    (void)queue_quarantine_ref;
    (void)queue_request_source_offsets_ref;
    (void)queue_request_object_offsets_ref;
    (void)queue_request_output_indices_ref;
    (void)queue_request_count_ref;
    (void)private_ready_epoch_ref;
    (void)private_ready_generation_ref;
    (void)worker_stats_ref;
    (void)workspace;
    // Queue lease CAS and Direct-HCCP helpers use a vector-side SIMT service;
    // they do not require a separate AIC lane.  AIV_ONLY prevents this
    // standalone executor from reserving otherwise-unused LI cube resources.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5ScoreReceiptSpeculativeExecutorTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    A5ScoreReceiptSpeculativeExecutorKernel op(&pipe, &tilingData);
    op.Init(candidate_seeds, candidate_counts, candidate_ready_epoch,
            candidate_status, streaming_epoch, source_lens,
            kv_object_address_domain, req_pool_entries, kv_object_map,
            kv_object_lens, kv_object_slots_snapshot, private_source_to_slot,
            private_object_to_slot, private_ready_table, private_arena_head,
            private_packed_output, fused_get_control, queue_leases, queue_epoch,
            queue_generation, queue_post_sequence, queue_state, queue_quarantine,
            queue_request_source_offsets, queue_request_object_offsets,
            queue_request_output_indices, queue_request_count, private_ready_epoch,
            private_ready_generation, worker_stats, rows_per_entry);
    op.Process();
}
