// Scan-time Direct-HCCP bridge for the source-derived A5 C8 LI operator.
//
// This hook consumes the official sortable-BF16 score tile directly after
// BatchMulWeightAndReduceSum. It never changes that tile, the score workspace,
// metadata, or ProcessTopK. A failed/no-capacity speculative request is only a
// suppressed prefetch: exact TopK and the later guaranteed-recall stage remain
// responsible for mathematical completeness.

#ifndef A5_STREAMING_QUANT_LI_GET_TILE_READY_H
#define A5_STREAMING_QUANT_LI_GET_TILE_READY_H

#include "kernel_operator.h"
#include "quant_lightning_indexer_common.h"
#include "../kvaf_device/kvaf_fused_full_get_device.h"

namespace QLIKernel {
namespace streaming_li_get {
using namespace AscendC;
using namespace QLICommon;

constexpr uint32_t kRowsPerKvafBlockDefault = 128U;
constexpr uint32_t kPackedKvRowBytes = 656U;
constexpr uint32_t kMaxWaveRequests = kvaf_fused_get::kDenseRequestCapacity;
constexpr int32_t kSourceUnmapped = -1;
constexpr int32_t kSourceClaiming = -2;
// P4.1 audit is a deliberately non-posting graph mode.  A caller marks every
// queue-lease cell with this otherwise-invalid owner value before launch.
// The hook then records score-qualified cold sources in the fixed
// [output-row, score-source] map below, but it never acquires a queue, writes
// a Direct-HCCP descriptor, rings a doorbell, or touches the private payload
// arena.  A negative owner is invalid for the transport ABI, so this cannot
// be confused with a real queue owner (which is strictly positive).
constexpr int32_t kAuditQueueLeaseSentinel = (-2147483647 - 1);
// P4.2 diagnostic mode is intentionally a different all-lease sentinel.  It
// never exists on the product path; it turns the existing audit-only [R,S]
// map into a complete score receipt so one launch can prove the seam between
// C8 reduction and the threshold/mapping predicate.
constexpr int32_t kAuditScoreReceiptQueueLeaseSentinel = -2147483647;
// `private_source_to_slot` is the existing graph-static [R,S] transaction
// map.  In P4.1 audit mode -3 is a candidate marker, not an HBM slot and not
// a published cache mapping.  It is intentionally separate from -1
// (unmapped) and -2 (a transient post-path claim).
constexpr int32_t kSourceAuditCandidate = -3;
// Audit score receipt encoding, stored only when every lease equals
// kAuditScoreReceiptQueueLeaseSentinel.  Bits [31:21] tag a written record;
// the remaining bits carry the exact sortable score and the decisions made
// at the device-side predicate.  No product/private-slot value can be
// confused with this representation because regular values are nonnegative
// slots, -1/-2 are local sentinels and -3 is candidate-only audit.
constexpr uint32_t kAuditScoreReceiptTag = 0x80000000U;
constexpr uint32_t kAuditScoreReceiptTagMask = 0xFFE00000U;
constexpr uint32_t kAuditScoreReceiptCandidateBit = 1U << 20;
constexpr uint32_t kAuditScoreReceiptMappingValidBit = 1U << 19;
constexpr uint32_t kAuditScoreReceiptColdBit = 1U << 18;
constexpr uint32_t kAuditScoreReceiptThresholdPassBit = 1U << 17;
constexpr uint32_t kAuditScoreReceiptSourceUnmappedBit = 1U << 16;
constexpr uint32_t kAuditScoreReceiptScoreMask = 0xFFFFU;
constexpr int32_t kPrivateReadyFree = 0;
constexpr int32_t kPrivateReadyPending = 1;
constexpr int32_t kPrivateReadyReady = 2;
constexpr int32_t kPrivateReadyFailed = 3;

// Each producer owns one telemetry row only while it holds that queue lease.
// Columns are intentionally compact enough for a graph-stable int64[Q,13].
enum TelemetryColumn : uint32_t {
    kTileReadyCalls = 0U,
    kScoresScanned = 1U,
    kThresholdPassed = 2U,
    kResidentHits = 3U,
    kPrivateAlreadyPresent = 4U,
    kPrivateClaims = 5U,
    kSuppressedNoQueueOrArena = 6U,
    kInvalidLogicalToken = 7U,
    kDoorbellPosted = 8U,
    kCqRetiredReady = 9U,
    kCqRetiredFailed = 10U,
    kLeaseReleased = 11U,
    kControlRejected = 12U,
};

// A source-map cell is first atomically changed -1 -> -2. Only its winning
// producer takes a bounded queue-local private slot. The caller publishes the
// final encoded slot only after Begin has posted a real Direct-HCCP wave;
// otherwise it restores -1. This is transaction-local state, never a managed
// HBM cache-slot/mapping update.
struct PrivateArenaClaimResult {
    uint64_t source_map_offset;
    uint32_t local_slot;
    int32_t encoded_slot;
    uint32_t success;
};
static_assert(sizeof(PrivateArenaClaimResult) == 24U,
              "private claim UB hand-off ABI changed");

// The candidate-marker audit and the full score-receipt audit must execute
// exactly the same threshold / S->O / resident test.  Keeping the decision in
// one helper prevents a diagnostic-only branch from accidentally proving a
// different predicate than the marker used by the staged product design.
struct AuditScoreDecision {
    uint32_t threshold_pass;
    uint32_t mapping_valid;
    uint32_t cold;
    uint32_t source_unmapped;
    uint32_t candidate;
};

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void ClaimStreamingPrivateArenaSimt(
    __gm__ int32_t *source_to_slot,
    __gm__ int32_t *private_ready,
    __gm__ int32_t *private_arena_head,
    uint64_t source_map_offset,
    uint32_t queue_id,
    uint32_t private_arena_capacity,
    __ubuf__ PrivateArenaClaimResult *result)
{
    if (source_to_slot == nullptr || private_ready == nullptr ||
        private_arena_head == nullptr || result == nullptr ||
        private_arena_capacity == 0U) {
        return;
    }
    result->source_map_offset = source_map_offset;
    result->local_slot = 0U;
    result->encoded_slot = kSourceUnmapped;
    result->success = 0U;
    if (asc_atomic_cas(source_to_slot + source_map_offset,
                       kSourceUnmapped, kSourceClaiming) != kSourceUnmapped) {
        return;
    }
    const uint32_t local_slot = asc_atomic_add(private_arena_head + queue_id, 1U);
    if (local_slot >= private_arena_capacity) {
        // This producer owns the -2 sentinel, so restoring it is safe.
        asc_atomic_cas(source_to_slot + source_map_offset,
                       kSourceClaiming, kSourceUnmapped);
        return;
    }
    private_ready[static_cast<uint64_t>(queue_id) * private_arena_capacity +
                  local_slot] = kPrivateReadyPending;
    result->local_slot = local_slot;
    result->encoded_slot = static_cast<int32_t>(
        static_cast<uint64_t>(queue_id) * private_arena_capacity + local_slot);
    result->success = 1U;
}

// ProcessVec1 runs on the LI vector/AIV side. CANN's own standalone AIV
// `a5_kvaf_batch_get_prepare` uses the same PIPE_S <-> PIPE_V + VF_CALL
// pattern, so the atomic SIMT service is explicitly routed through that
// supported MIX-kernel hand-off instead of a direct unsupported CAS call.
// EVENT_ID0 is only reused in the independent S/V event namespace; official
// LI uses the same numeric id for V/MTE hard events, which do not alias it.
__aicore__ inline bool ClaimStreamingPrivateArenaFromAiv(
    __gm__ int32_t *source_to_slot,
    __gm__ int32_t *private_ready,
    __gm__ int32_t *private_arena_head,
    uint64_t source_map_offset,
    uint32_t queue_id,
    uint32_t private_arena_capacity,
    __ubuf__ PrivateArenaClaimResult *result,
    event_t simt_event)
{
    if (result == nullptr) {
        return false;
    }
    result->success = 0U;
    set_flag(PIPE_S, PIPE_V, simt_event);
    wait_flag(PIPE_S, PIPE_V, simt_event);
    AscendC::Simt::VF_CALL<ClaimStreamingPrivateArenaSimt>(
        AscendC::Simt::Dim3{1U, 1U, 1U}, source_to_slot, private_ready,
        private_arena_head, source_map_offset, queue_id,
        private_arena_capacity, result);
    set_flag(PIPE_V, PIPE_S, simt_event);
    wait_flag(PIPE_V, PIPE_S, simt_event);
    return result->success != 0U;
}

class StreamingKvafTileReadyHook {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t *resident_threshold,
        __gm__ uint8_t *resident_threshold_valid,
        __gm__ uint8_t *kv_object_slots_snapshot,
        __gm__ uint8_t *req_pool_entries,
        __gm__ uint8_t *kv_object_lens,
        __gm__ uint8_t *kv_object_map,
        __gm__ uint8_t *private_source_to_slot,
        __gm__ uint8_t *private_ready_table,
        __gm__ uint8_t *private_arena_head,
        __gm__ uint8_t *private_packed_output,
        __gm__ uint8_t *queue_control,
        __gm__ uint8_t *queue_leases,
        __gm__ uint8_t *telemetry,
        const QLICommon::ConstInfo &const_info,
        uint32_t row_count,
        uint32_t source_capacity,
        uint32_t pool_size,
        uint32_t kv_object_capacity,
        uint32_t private_packed_rows,
        uint32_t private_arena_capacity,
        uint32_t queue_count,
        uint32_t rows_per_entry,
        __ubuf__ kvaf_fused_get::KvafStreamingBatchGetLeaseResult *lease_result,
        __ubuf__ PrivateArenaClaimResult *claim_result,
        __ubuf__ uint64_t *source_offsets,
        uint32_t producer_id)
    {
        residentThreshold_.SetGlobalBuffer((__gm__ int32_t *)resident_threshold);
        residentThresholdValid_.SetGlobalBuffer((__gm__ int32_t *)resident_threshold_valid);
        kvObjectSlots_.SetGlobalBuffer((__gm__ int32_t *)kv_object_slots_snapshot);
        reqPoolEntries_.SetGlobalBuffer((__gm__ int32_t *)req_pool_entries);
        kvObjectLens_.SetGlobalBuffer((__gm__ int32_t *)kv_object_lens);
        kvObjectMap_.SetGlobalBuffer((__gm__ int32_t *)kv_object_map);
        sourceToSlot_.SetGlobalBuffer((__gm__ int32_t *)private_source_to_slot);
        privateReady_.SetGlobalBuffer((__gm__ int32_t *)private_ready_table);
        privateArenaHead_.SetGlobalBuffer((__gm__ int32_t *)private_arena_head);
        queueLeases_.SetGlobalBuffer((__gm__ int32_t *)queue_leases);
        telemetry_.SetGlobalBuffer((__gm__ int64_t *)telemetry);
        controlGm_ = reinterpret_cast<__gm__ const kvaf_fused_get::KvafStreamingBatchGetControl *>(queue_control);
        sourceToSlotRaw_ = reinterpret_cast<__gm__ int32_t *>(private_source_to_slot);
        privateReadyRaw_ = reinterpret_cast<__gm__ int32_t *>(private_ready_table);
        privateArenaHeadRaw_ = reinterpret_cast<__gm__ int32_t *>(private_arena_head);
        privatePackedOutputRaw_ = private_packed_output;
        queueLeasesRaw_ = reinterpret_cast<__gm__ int32_t *>(queue_leases);
        leaseResult_ = lease_result;
        claimResult_ = claim_result;
        sourceOffsets_ = source_offsets;
        constInfo_ = const_info;
        rowCount_ = row_count;
        sourceCapacity_ = source_capacity;
        poolSize_ = pool_size;
        kvObjectCapacity_ = kv_object_capacity;
        privatePackedRows_ = private_packed_rows;
        privateArenaCapacity_ = private_arena_capacity;
        queueCount_ = queue_count;
        rowsPerEntry_ = rows_per_entry == 0U ? kRowsPerKvafBlockDefault : rows_per_entry;
        producerId_ = producer_id;
        enabled_ = false;
        auditMode_ = false;
        auditScoreReceiptMode_ = false;
        hasWave_ = false;
        stagedCount_ = 0U;
        if (resident_threshold == nullptr || resident_threshold_valid == nullptr ||
            kv_object_lens == nullptr ||
            controlGm_ == nullptr || leaseResult_ == nullptr || claimResult_ == nullptr ||
            sourceOffsets_ == nullptr || rowCount_ == 0U || sourceCapacity_ == 0U ||
            poolSize_ == 0U || kvObjectCapacity_ == 0U || privateArenaCapacity_ == 0U ||
            privatePackedOutputRaw_ == nullptr || privatePackedRows_ == 0U ||
            queueCount_ == 0U || rowsPerEntry_ == 0U) {
            return;
        }
        if (!kvaf_fused_get::LoadKvafStreamingBatchGetControlFromGmForAicore(
                controlGm_, &control_) ||
            !kvaf_fused_get::IsKvafStreamingBatchGetControlValidForAicore(control_) ||
            control_.queue_count != queueCount_ ||
            control_.queue_leases != reinterpret_cast<uint64_t>(queueLeasesRaw_) ||
            control_.output != reinterpret_cast<uint64_t>(privatePackedOutputRaw_) ||
            control_.output_rows != privatePackedRows_) {
            AddTelemetry(0U, kControlRejected, 1);
            return;
        }
        const uint64_t privateRows = control_.private_output_stride_bytes / kPackedKvRowBytes;
        // `private_source_to_slot` encodes queue * privateArenaCapacity +
        // local_slot, and reconcile later consumes that value as the physical
        // row in private_packed_output.  Therefore the queue stride in the
        // registered output must be *exactly* privateArenaCapacity, not just
        // large enough to contain it. Otherwise queue>0 would point at a
        // different physical row than the payload written by Direct-HCCP.
        if (privateRows == 0U || privateArenaCapacity_ != privateRows) {
            AddTelemetry(0U, kControlRejected, 1);
            return;
        }
        auditScoreReceiptMode_ = IsAuditScoreReceiptLeaseTable();
        auditMode_ = auditScoreReceiptMode_ || IsAuditLeaseTable();
#if A5_STREAMING_LI_GET_INLINE_MODE == 1
        // Audit is a separate binary, not a runtime promise.  Refuse to
        // enable this hook unless the caller supplied the full score-receipt
        // sentinel.  Therefore a mistaken queue table cannot turn a strict
        // equivalence run into a real Direct-HCCP post.
        if (!auditScoreReceiptMode_) {
            return;
        }
#endif
        enabled_ = true;
    }

    // Decide before the official score reduction whether this tile needs the
    // P3 observation seam at all.  The common saturated case has no local
    // in-flight wave and every lease is held by another producer.  It cannot
    // post a speculative Get, so it must also avoid adding a V-pipeline
    // barrier between the official reduction and score DMA.
    //
    // This is deliberately a conservative snapshot, not an ownership
    // decision.  If a lease changes after this read, we can miss one
    // *speculative* opportunity but Exact Top-K plus Guaranteed Recall still
    // supplies every required KV.  When a lease looks free, `Emit()` repeats
    // the real atomic acquire and remains the only ownership authority.
    __aicore__ inline bool MayNeedScoreObservation()
    {
        if (!enabled_) {
            return false;
        }
        // Audit must visit every raw-Q/K score tile even though all queue
        // leases deliberately look non-free.  The branch performs only
        // candidate-map writes and has no Direct-HCCP side effect.
        if (auditMode_) {
            return true;
        }
        // A locally posted wave must reach Emit even while all queues are
        // busy: Emit polls its CQ and may retire/release that lease.
        return hasWave_ || AnyQueueMayBeFree();
    }

    __aicore__ inline void Emit(const LocalTensor<uint16_t> &out_base,
                                const QLICommon::RunInfo &info,
                                int64_t cur_aiv_s1_idx,
                                int64_t cur_aiv_s1_proc_num,
                                int64_t cur_s2_idx)
    {
        if (!enabled_) {
            return;
        }
        if (auditMode_) {
            if (auditScoreReceiptMode_) {
                EmitAuditScoreReceipt(out_base, info, cur_aiv_s1_idx,
                                      cur_aiv_s1_proc_num, cur_s2_idx);
            } else {
                EmitAuditCandidates(out_base, info, cur_aiv_s1_idx,
                                    cur_aiv_s1_proc_num, cur_s2_idx);
            }
            return;
        }
        RetireAndReleaseIfTerminal(false);
        if (hasWave_) {
            // One Direct-HCCP wave per lease. The next tile remains exact scan
            // work and intentionally suppresses further speculation until CQ.
            AddTelemetry(wave_.lease.queue_id, kSuppressedNoQueueOrArena, 1);
            return;
        }
        // The no-idle-queue case is both common and intentionally harmless:
        // ExactScan still produces the unchanged official score/Top-K path,
        // while Guaranteed Recall later covers any cold item that was not
        // speculatively promoted.  Do this inexpensive GM snapshot before
        // entering the AIV<->SIMT atomic service.  In particular, an
        // all-busy graph replay must not invoke one VF_CALL per C8 score tile
        // merely to rediscover that every queue is already leased.
        //
        // This is deliberately only a conservative precheck.  A queue can
        // become free immediately after the read; then this tile skips a
        // *speculative* Get, never a required read.  If any queue appears
        // idle, the existing atomic acquire below remains the ownership
        // authority and preserves the real concurrent path.
        if (!AnyQueueMayBeFree()) {
            return;
        }
        kvaf_fused_get::KvafStreamingBatchGetLease lease{};
        if (!kvaf_fused_get::TryAcquireKvafStreamingBatchGetQueueFromAicore(
                controlGm_, producerId_, static_cast<uint32_t>(cur_s2_idx),
                leaseResult_, kStreamingSimtEvent, &lease)) {
            return;
        }
        const uint32_t queueId = lease.queue_id;
        // The cold host lifecycle preconfigures a graph-static template per
        // queue.  The active scratch is shared by the lease pool, so it is
        // populated only after the queue is exclusively held.  ExactScan may
        // modify only this queue's candidate arrays and active count.
        kvaf_fused_get::KvafStreamingBatchGetDescriptorScratch descriptor{};
        if (!kvaf_fused_get::LoadKvafStreamingBatchGetQueueTemplate(
                control_, lease, &descriptor)) {
            ReleaseUnpostedLease(lease);
            return;
        }
        if (!ValidateDescriptorTemplate(descriptor, queueId)) {
            AddTelemetry(queueId, kControlRejected, 1);
            ReleaseUnpostedLease(lease);
            return;
        }
        AddTelemetry(queueId, kTileReadyCalls, 1);
        stagedCount_ = 0U;
        const int64_t curS1ProcNum = cur_aiv_s1_proc_num;
        for (int64_t localRow = 0; localRow < curS1ProcNum && stagedCount_ < descriptor.request_capacity;
             ++localRow) {
            const uint64_t row = OutputRow(info, cur_aiv_s1_idx + localRow);
            if (row >= rowCount_) {
                continue;
            }
            const int32_t validInTile = ValidInTile(info, cur_aiv_s1_idx, localRow, cur_s2_idx);
            const uint64_t outRowOffset = static_cast<uint64_t>(localRow) *
                (UB_BANK_DEPTH_STRIDE / sizeof(uint16_t));
            for (int32_t inTile = 0; inTile < validInTile && stagedCount_ < descriptor.request_capacity;
                 ++inTile) {
                TryStageCandidate(out_base.GetValue(outRowOffset + inTile), row,
                                  static_cast<uint32_t>(cur_s2_idx + inTile),
                                  queueId, descriptor);
            }
        }
        if (stagedCount_ == 0U) {
            ReleaseUnpostedLease(lease);
            return;
        }
        descriptor.active_request_count = stagedCount_;
        if (!kvaf_fused_get::PublishKvafStreamingBatchGetDescriptorScratch(
                control_, lease, descriptor) ||
            !kvaf_fused_get::BeginKvafStreamingBatchGet(control_, lease, &wave_)) {
            RollBackUnpostedClaims(lease, descriptor);
            // Begin marks its local wave terminal even on validation failure.
            if (wave_.lease.state == kvaf_fused_get::kStreamingBatchGetLeaseTerminal) {
                kvaf_fused_get::ReleaseKvafStreamingBatchGetQueueFromAicore(
                    controlGm_, leaseResult_, kStreamingSimtEvent, &wave_);
            } else {
                ReleaseUnpostedLease(lease);
            }
            stagedCount_ = 0U;
            return;
        }
        FinalizePendingClaims();
        hasWave_ = true;
        AddTelemetry(queueId, kDoorbellPosted, 1);
    }

    __aicore__ inline void Finish()
    {
        if (!enabled_) {
            return;
        }
        RetireAndReleaseIfTerminal(true);
    }

private:
    static constexpr event_t kStreamingSimtEvent = EVENT_ID0;

    __aicore__ inline bool AnyQueueMayBeFree()
    {
        // `queueLeases_` is the graph-visible ownership table.  This scan is
        // bounded by the preconfigured Direct-HCCP queue count (at most the
        // existing KVAF AIV worker limit), so it is cheaper than a SIMT
        // service round trip on every rejected score tile.
        for (uint32_t queueId = 0U; queueId < queueCount_; ++queueId) {
            if (queueLeases_.GetValue(queueId) == 0) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline bool IsAuditLeaseTable()
    {
        // Audit is all-or-nothing.  A mixed table is never treated as audit:
        // it remains the normal post path where zero means a real free queue.
        for (uint32_t queueId = 0U; queueId < queueCount_; ++queueId) {
            if (queueLeases_.GetValue(queueId) != kAuditQueueLeaseSentinel) {
                return false;
            }
        }
        return queueCount_ != 0U;
    }

    __aicore__ inline bool IsAuditScoreReceiptLeaseTable()
    {
        // This separate sentinel makes the all-score receipt impossible to
        // activate accidentally from the normal candidate-only audit or from
        // the product post path.
        for (uint32_t queueId = 0U; queueId < queueCount_; ++queueId) {
            if (queueLeases_.GetValue(queueId) !=
                kAuditScoreReceiptQueueLeaseSentinel) {
                return false;
            }
        }
        return queueCount_ != 0U;
    }

    __aicore__ inline uint64_t OutputRow(const QLICommon::RunInfo &info,
                                          int64_t s1_index) const
    {
        // Same flattened output address arithmetic as official ProcessTopK.
        return info.indiceOutOffset / constInfo_.sparseCount +
               static_cast<uint64_t>(s1_index);
    }

    __aicore__ inline int32_t ValidInTile(const QLICommon::RunInfo &info,
                                           int64_t cur_aiv_s1_idx,
                                           int64_t local_row,
                                           int64_t cur_s2_idx) const
    {
        int64_t validS2 = info.actS2Size;
        if (constInfo_.attenMaskFlag) {
            const int64_t base = static_cast<int64_t>(info.actS2SizeOrig) -
                static_cast<int64_t>(info.actS1Size) + cur_aiv_s1_idx + 1;
            validS2 = (base + local_row) / constInfo_.cmpRatio;
        }
        const int64_t remain = validS2 - cur_s2_idx;
        if (remain <= 0) {
            return 0;
        }
        const int64_t sourceRemain = static_cast<int64_t>(sourceCapacity_) - cur_s2_idx;
        if (sourceRemain <= 0) {
            return 0;
        }
        const int64_t tile = remain < info.actualSingleProcessSInnerSize ?
            remain : info.actualSingleProcessSInnerSize;
        return static_cast<int32_t>(tile > sourceRemain ? sourceRemain : tile);
    }

    __aicore__ inline bool ValidateDescriptorTemplate(
        const kvaf_fused_get::KvafStreamingBatchGetDescriptorScratch &descriptor,
        uint32_t queue_id) const
    {
        return descriptor.address_table != 0U && descriptor.valid_table != 0U &&
               descriptor.indices != 0U && descriptor.offsets != 0U &&
               descriptor.mask != 0U && descriptor.output_indices != 0U &&
               descriptor.table_capacity > 0U &&
               descriptor.request_capacity > 0U &&
               descriptor.request_capacity <= kMaxWaveRequests &&
               descriptor.private_output_row_base == PrivateOutputRowBase(queue_id) &&
               queue_id < queueCount_;
    }

    // Audit is the score-time candidate contract, not a transport exercise.
    // This routine deliberately mirrors the threshold / S->O / resident-miss
    // predicates in TryStageCandidate(), but ends at the fixed source map.
    // Each (row, source_index) is owned by exactly one C8 score tile, so this
    // path does not need a queue claim or a private-arena allocation.
    __aicore__ inline AuditScoreDecision EvaluateAuditScore(
        uint16_t score, uint64_t row, uint32_t source_index)
    {
        AuditScoreDecision decision{0U, 0U, 0U, 0U, 0U};
        const uint16_t threshold = static_cast<uint16_t>(
            static_cast<uint32_t>(residentThreshold_.GetValue(row)) &
            kAuditScoreReceiptScoreMask);
        decision.threshold_pass =
            residentThresholdValid_.GetValue(row) == 1 && score >= threshold;
        const int32_t poolRow = reqPoolEntries_.GetValue(row);
        const int32_t kvObjectLen = kvObjectLens_.GetValue(row);
        if (poolRow < 0 || static_cast<uint32_t>(poolRow) >= poolSize_ ||
            kvObjectLen <= 0 ||
            static_cast<uint32_t>(kvObjectLen) > kvObjectCapacity_) {
            return decision;
        }
        const int32_t kvObject = kvObjectMap_.GetValue(
            static_cast<uint64_t>(poolRow) * sourceCapacity_ + source_index);
        if (kvObject < 0 ||
            static_cast<uint32_t>(kvObject) >=
                static_cast<uint32_t>(kvObjectLen)) {
            return decision;
        }
        decision.mapping_valid = 1U;
        decision.cold = kvObjectSlots_.GetValue(
            static_cast<uint64_t>(poolRow) * kvObjectCapacity_ +
            static_cast<uint32_t>(kvObject)) < 0;
        const uint64_t sourceOffset = row * sourceCapacity_ + source_index;
        decision.source_unmapped =
            sourceToSlot_.GetValue(sourceOffset) == kSourceUnmapped;
        decision.candidate =
            decision.threshold_pass != 0U && decision.cold != 0U &&
            decision.source_unmapped != 0U;
        return decision;
    }

    __aicore__ inline void TryRecordAuditCandidate(
        uint16_t score, uint64_t row, uint32_t source_index)
    {
        const AuditScoreDecision decision =
            EvaluateAuditScore(score, row, source_index);
        if (decision.candidate == 0U) {
            return;
        }
        const uint64_t sourceOffset = row * sourceCapacity_ + source_index;
        const int32_t existing = sourceToSlot_.GetValue(sourceOffset);
        if (existing == kSourceUnmapped) {
            // Audit candidates are a graph-visible record only.  They must
            // never be read by QSFA/reconcile as an HBM slot.
            sourceToSlot_.SetValue(sourceOffset, kSourceAuditCandidate);
            return;
        }
    }

    __aicore__ inline void EmitAuditCandidates(
        const LocalTensor<uint16_t> &out_base,
        const QLICommon::RunInfo &info,
        int64_t cur_aiv_s1_idx,
        int64_t cur_aiv_s1_proc_num,
        int64_t cur_s2_idx)
    {
        const int64_t curS1ProcNum = cur_aiv_s1_proc_num;
        for (int64_t localRow = 0; localRow < curS1ProcNum; ++localRow) {
            const uint64_t row = OutputRow(info, cur_aiv_s1_idx + localRow);
            if (row >= rowCount_) {
                continue;
            }
            const int32_t validInTile = ValidInTile(
                info, cur_aiv_s1_idx, localRow, cur_s2_idx);
            const uint64_t outRowOffset = static_cast<uint64_t>(localRow) *
                (UB_BANK_DEPTH_STRIDE / sizeof(uint16_t));
            for (int32_t inTile = 0; inTile < validInTile; ++inTile) {
                TryRecordAuditCandidate(
                    out_base.GetValue(outRowOffset + inTile), row,
                    static_cast<uint32_t>(cur_s2_idx + inTile));
            }
        }
        // Do not use the normal telemetry table here.  Its rows are owned by
        // a leased transport queue, while audit intentionally has no lease;
        // concurrent AIV read/modify/write telemetry updates would make a
        // count nondeterministic.  The exact audit count is instead the
        // host-visible number of -3 entries in this fixed [R,S] output map.
    }

    // P4.2 audit-only diagnostic.  This is deliberately not a second score
    // calculation and not a product state transition: it serializes the
    // value already produced by BatchMulWeightAndReduceSum plus each exact
    // branch decision into the existing fixed [R,S] audit receipt.  Every
    // valid score coordinate is owned by one LI tile, so non-atomic stores
    // are sufficient and make a skipped tile visible as the original -1.
    __aicore__ inline void EmitAuditScoreReceipt(
        const LocalTensor<uint16_t> &out_base,
        const QLICommon::RunInfo &info,
        int64_t cur_aiv_s1_idx,
        int64_t cur_aiv_s1_proc_num,
        int64_t cur_s2_idx)
    {
        const int64_t curS1ProcNum = cur_aiv_s1_proc_num;
        for (int64_t localRow = 0; localRow < curS1ProcNum; ++localRow) {
            const uint64_t row = OutputRow(info, cur_aiv_s1_idx + localRow);
            if (row >= rowCount_) {
                continue;
            }
            const int32_t validInTile = ValidInTile(
                info, cur_aiv_s1_idx, localRow, cur_s2_idx);
            const uint64_t outRowOffset = static_cast<uint64_t>(localRow) *
                (UB_BANK_DEPTH_STRIDE / sizeof(uint16_t));
            for (int32_t inTile = 0; inTile < validInTile; ++inTile) {
                const uint16_t score = out_base.GetValue(outRowOffset + inTile);
                const uint32_t sourceIndex = static_cast<uint32_t>(
                    cur_s2_idx + inTile);
                const AuditScoreDecision decision =
                    EvaluateAuditScore(score, row, sourceIndex);
                uint32_t receipt = kAuditScoreReceiptTag |
                    (static_cast<uint32_t>(score) & kAuditScoreReceiptScoreMask);
                if (decision.threshold_pass != 0U) {
                    receipt |= kAuditScoreReceiptThresholdPassBit;
                }
                if (decision.source_unmapped != 0U) {
                    receipt |= kAuditScoreReceiptSourceUnmappedBit;
                }
                if (decision.cold != 0U) {
                    receipt |= kAuditScoreReceiptColdBit;
                }
                if (decision.mapping_valid != 0U) {
                    receipt |= kAuditScoreReceiptMappingValidBit;
                }
                if (decision.candidate != 0U) {
                    receipt |= kAuditScoreReceiptCandidateBit;
                }
                const uint64_t sourceOffset = row * sourceCapacity_ + sourceIndex;
                // The tag always sets bit 31.  Converting that uint32_t
                // directly to int32_t is implementation-defined in C++17;
                // express the two's-complement bit pattern through int64_t
                // so the graph-visible int32 receipt is portable and exact.
                const int32_t signedReceipt = static_cast<int32_t>(
                    static_cast<int64_t>(receipt) - (1LL << 32));
                sourceToSlot_.SetValue(sourceOffset, signedReceipt);
            }
        }
    }

    __aicore__ inline uint32_t PrivateOutputRowBase(uint32_t queue_id) const
    {
        return static_cast<uint32_t>(queue_id *
            (control_.private_output_stride_bytes / kPackedKvRowBytes));
    }

    __aicore__ inline void TryStageCandidate(
        uint16_t score, uint64_t row, uint32_t source_index, uint32_t queue_id,
        const kvaf_fused_get::KvafStreamingBatchGetDescriptorScratch &descriptor)
    {
        AddTelemetry(queue_id, kScoresScanned, 1);
        // A zero validity bit is a sentinel from the resident-only C8 seed,
        // not a score threshold.  Suppressing speculation here preserves the
        // final exact path while avoiding an accidental all-score prefetch.
        if (residentThresholdValid_.GetValue(row) != 1) {
            return;
        }
        // ResidentSeed writes the official sortable uint16 C8 kth score into
        // the low 16 bits of an int32 output.  Keeping that storage type here
        // permits a direct graph edge from ResidentSeed with no cast/copy.
        const uint16_t threshold = static_cast<uint16_t>(
            static_cast<uint32_t>(residentThreshold_.GetValue(row)) & 0xffffU);
        if (score < threshold) {
            return;
        }
        AddTelemetry(queue_id, kThresholdPassed, 1);
        const int32_t poolRow = reqPoolEntries_.GetValue(row);
        if (poolRow < 0 || static_cast<uint32_t>(poolRow) >= poolSize_) {
            AddTelemetry(queue_id, kInvalidLogicalToken, 1);
            return;
        }
        const int32_t kvObject = kvObjectMap_.GetValue(
            static_cast<uint64_t>(poolRow) * sourceCapacity_ + source_index);
        const int32_t kvObjectLen = kvObjectLens_.GetValue(row);
        // `kv_object_map` is a static [pool,S_capacity] table, while a
        // forward's active QSFA-object prefix can be shorter.  Without this
        // check a graph replay could prefetch stale tail entries that are
        // outside the current request even though they fit static HBM shape.
        if (kvObjectLen <= 0 ||
            static_cast<uint32_t>(kvObjectLen) > kvObjectCapacity_ ||
            kvObject < 0 ||
            static_cast<uint32_t>(kvObject) >=
                static_cast<uint32_t>(kvObjectLen)) {
            AddTelemetry(queue_id, kInvalidLogicalToken, 1);
            return;
        }
        // The snapshot is indexed by the declared KV-object domain, never by
        // a guessed `source_index * cmp_ratio` raw-token conversion.
        if (kvObjectSlots_.GetValue(static_cast<uint64_t>(poolRow) * kvObjectCapacity_ +
                                    static_cast<uint32_t>(kvObject)) >= 0) {
            AddTelemetry(queue_id, kResidentHits, 1);
            return;
        }
        const uint64_t sourceOffset = row * sourceCapacity_ + source_index;
        if (sourceToSlot_.GetValue(sourceOffset) != kSourceUnmapped) {
            AddTelemetry(queue_id, kPrivateAlreadyPresent, 1);
            return;
        }
        if (!ClaimStreamingPrivateArenaFromAiv(sourceToSlotRaw_, privateReadyRaw_,
                                                privateArenaHeadRaw_, sourceOffset,
                                                queue_id, privateArenaCapacity_,
                                                claimResult_, kStreamingSimtEvent)) {
            AddTelemetry(queue_id, kSuppressedNoQueueOrArena, 1);
            return;
        }
        const uint32_t request = stagedCount_;
        const uint32_t object = static_cast<uint32_t>(kvObject);
        const uint32_t tableIndex = object / rowsPerEntry_;
        if (tableIndex >= descriptor.table_capacity) {
            sourceToSlot_.SetValue(sourceOffset, kSourceUnmapped);
            privateReady_.SetValue(static_cast<uint64_t>(queue_id) * privateArenaCapacity_ +
                                   claimResult_->local_slot, kPrivateReadyFree);
            AddTelemetry(queue_id, kInvalidLogicalToken, 1);
            return;
        }
        __gm__ int32_t *indices = reinterpret_cast<__gm__ int32_t *>(descriptor.indices);
        __gm__ int32_t *offsets = reinterpret_cast<__gm__ int32_t *>(descriptor.offsets);
        __gm__ int32_t *outputIndices = reinterpret_cast<__gm__ int32_t *>(descriptor.output_indices);
        __gm__ uint8_t *mask = reinterpret_cast<__gm__ uint8_t *>(descriptor.mask);
        indices[request] = static_cast<int32_t>(tableIndex);
        offsets[request] = static_cast<int32_t>((object % rowsPerEntry_) * kPackedKvRowBytes);
        outputIndices[request] = static_cast<int32_t>(PrivateOutputRowBase(queue_id) +
                                                       claimResult_->local_slot);
        if (mask != nullptr) {
            mask[request] = 1U;
        }
        sourceOffsets_[request] = sourceOffset;
        ++stagedCount_;
        AddTelemetry(queue_id, kPrivateClaims, 1);
    }

    __aicore__ inline void FinalizePendingClaims()
    {
        for (uint32_t request = 0U; request < stagedCount_; ++request) {
            const uint64_t sourceOffset = sourceOffsets_[request];
            // The encoded slot was produced by the atomic claim and is the
            // only transaction-visible mapping published before CQ.
            const int32_t outputIndex = reinterpret_cast<__gm__ int32_t *>(
                wave_.queue_args.output_indices)[request];
            const int32_t localSlot = outputIndex - static_cast<int32_t>(
                PrivateOutputRowBase(wave_.lease.queue_id));
            sourceToSlot_.SetValue(sourceOffset, static_cast<int32_t>(
                wave_.lease.queue_id * privateArenaCapacity_ + localSlot));
        }
    }

    __aicore__ inline void RollBackUnpostedClaims(
        const kvaf_fused_get::KvafStreamingBatchGetLease &lease,
        const kvaf_fused_get::KvafStreamingBatchGetDescriptorScratch &descriptor)
    {
        for (uint32_t request = 0U; request < stagedCount_; ++request) {
            const uint64_t sourceOffset = sourceOffsets_[request];
            const int32_t outputIndex = reinterpret_cast<__gm__ int32_t *>(
                descriptor.output_indices)[request];
            const int32_t localSlot = outputIndex - static_cast<int32_t>(
                PrivateOutputRowBase(lease.queue_id));
            if (localSlot >= 0 && static_cast<uint32_t>(localSlot) < privateArenaCapacity_) {
                privateReady_.SetValue(static_cast<uint64_t>(lease.queue_id) *
                    privateArenaCapacity_ + static_cast<uint32_t>(localSlot), kPrivateReadyFree);
            }
            sourceToSlot_.SetValue(sourceOffset, kSourceUnmapped);
        }
    }

    __aicore__ inline void PublishTerminalReadyState()
    {
        kvaf_fused_get::KvafStreamingBatchGetReady ready{};
        const bool payloadReady = kvaf_fused_get::GetKvafStreamingBatchGetReady(wave_, &ready);
        const uint32_t count = static_cast<uint32_t>(wave_.transaction.active_count);
        __gm__ int32_t *outputIndices = reinterpret_cast<__gm__ int32_t *>(
            wave_.queue_args.output_indices);
        const uint32_t base = PrivateOutputRowBase(wave_.lease.queue_id);
        for (uint32_t request = 0U; request < count; ++request) {
            const int32_t localSlot = outputIndices[request] - static_cast<int32_t>(base);
            if (localSlot >= 0 && static_cast<uint32_t>(localSlot) < privateArenaCapacity_) {
                privateReady_.SetValue(static_cast<uint64_t>(wave_.lease.queue_id) *
                    privateArenaCapacity_ + static_cast<uint32_t>(localSlot),
                    payloadReady ? kPrivateReadyReady : kPrivateReadyFailed);
            }
        }
        AddTelemetry(wave_.lease.queue_id,
                     payloadReady ? kCqRetiredReady : kCqRetiredFailed, 1);
    }

    __aicore__ inline void RetireAndReleaseIfTerminal(bool finish)
    {
        if (!hasWave_) {
            return;
        }
        if (finish) {
            kvaf_fused_get::FinishKvafStreamingBatchGet(&wave_);
        } else {
            kvaf_fused_get::TryRetireKvafStreamingBatchGet(&wave_);
        }
        if (wave_.lease.state != kvaf_fused_get::kStreamingBatchGetLeaseTerminal) {
            return;
        }
        PublishTerminalReadyState();
        if (kvaf_fused_get::ReleaseKvafStreamingBatchGetQueueFromAicore(
                controlGm_, leaseResult_, kStreamingSimtEvent, &wave_)) {
            AddTelemetry(wave_.lease.queue_id, kLeaseReleased, 1);
        }
        hasWave_ = false;
        stagedCount_ = 0U;
    }

    __aicore__ inline void ReleaseUnpostedLease(
        kvaf_fused_get::KvafStreamingBatchGetLease &lease)
    {
        // Convert a never-posted held lease into a terminal dummy wave so the
        // checked helper performs the same atomic ownership release.
        wave_.lease = lease;
        wave_.lease.state = kvaf_fused_get::kStreamingBatchGetLeaseTerminal;
        wave_.transaction.finalized = 1U;
        kvaf_fused_get::ReleaseKvafStreamingBatchGetQueueFromAicore(
            controlGm_, leaseResult_, kStreamingSimtEvent, &wave_);
    }

    __aicore__ inline void AddTelemetry(uint32_t queue_id, uint32_t column,
                                        int64_t value)
    {
        if (queue_id >= queueCount_ || column >= 13U) {
            return;
        }
        const uint64_t offset = static_cast<uint64_t>(queue_id) * 13U + column;
        telemetry_.SetValue(offset, telemetry_.GetValue(offset) + value);
    }

    GlobalTensor<int32_t> residentThreshold_;
    GlobalTensor<int32_t> residentThresholdValid_;
    GlobalTensor<int32_t> kvObjectSlots_;
    GlobalTensor<int32_t> reqPoolEntries_;
    GlobalTensor<int32_t> kvObjectLens_;
    GlobalTensor<int32_t> kvObjectMap_;
    GlobalTensor<int32_t> sourceToSlot_;
    GlobalTensor<int32_t> privateReady_;
    GlobalTensor<int32_t> privateArenaHead_;
    GlobalTensor<int32_t> queueLeases_;
    GlobalTensor<int64_t> telemetry_;
    __gm__ const kvaf_fused_get::KvafStreamingBatchGetControl *controlGm_ = nullptr;
    __gm__ int32_t *sourceToSlotRaw_ = nullptr;
    __gm__ int32_t *privateReadyRaw_ = nullptr;
    __gm__ int32_t *privateArenaHeadRaw_ = nullptr;
    __gm__ uint8_t *privatePackedOutputRaw_ = nullptr;
    __gm__ int32_t *queueLeasesRaw_ = nullptr;
    __ubuf__ kvaf_fused_get::KvafStreamingBatchGetLeaseResult *leaseResult_ = nullptr;
    __ubuf__ PrivateArenaClaimResult *claimResult_ = nullptr;
    __ubuf__ uint64_t *sourceOffsets_ = nullptr;
    kvaf_fused_get::KvafStreamingBatchGetControl control_{};
    kvaf_fused_get::KvafStreamingBatchGetWave wave_{};
    QLICommon::ConstInfo constInfo_{};
    uint32_t rowCount_ = 0U;
    uint32_t sourceCapacity_ = 0U;
    uint32_t poolSize_ = 0U;
    uint32_t kvObjectCapacity_ = 0U;
    uint32_t privatePackedRows_ = 0U;
    uint32_t privateArenaCapacity_ = 0U;
    uint32_t queueCount_ = 0U;
    uint32_t rowsPerEntry_ = kRowsPerKvafBlockDefault;
    uint32_t producerId_ = 0U;
    uint32_t stagedCount_ = 0U;
    bool enabled_ = false;
    bool auditMode_ = false;
    bool auditScoreReceiptMode_ = false;
    bool hasWave_ = false;
};

}  // namespace streaming_li_get
}  // namespace QLIKernel

#endif  // A5_STREAMING_QUANT_LI_GET_TILE_READY_H
