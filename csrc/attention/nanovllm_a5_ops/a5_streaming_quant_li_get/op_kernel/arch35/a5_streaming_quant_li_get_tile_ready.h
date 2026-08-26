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
        __gm__ uint8_t *cache_slots_snapshot,
        __gm__ uint8_t *req_pool_entries,
        __gm__ uint8_t *logical_token_map,
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
        uint32_t cache_token_capacity,
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
        cacheSlots_.SetGlobalBuffer((__gm__ int32_t *)cache_slots_snapshot);
        reqPoolEntries_.SetGlobalBuffer((__gm__ int32_t *)req_pool_entries);
        logicalTokenMap_.SetGlobalBuffer((__gm__ int32_t *)logical_token_map);
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
        cacheTokenCapacity_ = cache_token_capacity;
        privatePackedRows_ = private_packed_rows;
        privateArenaCapacity_ = private_arena_capacity;
        queueCount_ = queue_count;
        rowsPerEntry_ = rows_per_entry == 0U ? kRowsPerKvafBlockDefault : rows_per_entry;
        producerId_ = producer_id;
        enabled_ = false;
        hasWave_ = false;
        stagedCount_ = 0U;
        if (resident_threshold == nullptr || resident_threshold_valid == nullptr ||
            controlGm_ == nullptr || leaseResult_ == nullptr || claimResult_ == nullptr ||
            sourceOffsets_ == nullptr || rowCount_ == 0U || sourceCapacity_ == 0U ||
            poolSize_ == 0U || cacheTokenCapacity_ == 0U || privateArenaCapacity_ == 0U ||
            privatePackedOutputRaw_ == nullptr || privatePackedRows_ == 0U ||
            queueCount_ == 0U || rowsPerEntry_ == 0U) {
            return;
        }
        control_ = *controlGm_;
        if (!kvaf_fused_get::IsKvafStreamingBatchGetControlValidForAicore(control_) ||
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
        enabled_ = true;
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
        RetireAndReleaseIfTerminal(false);
        if (hasWave_) {
            // One Direct-HCCP wave per lease. The next tile remains exact scan
            // work and intentionally suppresses further speculation until CQ.
            AddTelemetry(wave_.lease.queue_id, kSuppressedNoQueueOrArena, 1);
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
        const int32_t logicalToken = logicalTokenMap_.GetValue(
            static_cast<uint64_t>(poolRow) * sourceCapacity_ + source_index);
        if (logicalToken < 0 || static_cast<uint32_t>(logicalToken) >= cacheTokenCapacity_) {
            AddTelemetry(queue_id, kInvalidLogicalToken, 1);
            return;
        }
        // Resident view is indexed by real logical token, not compressed S2.
        if (cacheSlots_.GetValue(static_cast<uint64_t>(poolRow) * cacheTokenCapacity_ +
                                 static_cast<uint32_t>(logicalToken)) >= 0) {
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
        const uint32_t logical = static_cast<uint32_t>(logicalToken);
        const uint32_t tableIndex = logical / rowsPerEntry_;
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
        offsets[request] = static_cast<int32_t>((logical % rowsPerEntry_) * kPackedKvRowBytes);
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
    GlobalTensor<int32_t> cacheSlots_;
    GlobalTensor<int32_t> reqPoolEntries_;
    GlobalTensor<int32_t> logicalTokenMap_;
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
    uint32_t cacheTokenCapacity_ = 0U;
    uint32_t privatePackedRows_ = 0U;
    uint32_t privateArenaCapacity_ = 0U;
    uint32_t queueCount_ = 0U;
    uint32_t rowsPerEntry_ = kRowsPerKvafBlockDefault;
    uint32_t producerId_ = 0U;
    uint32_t stagedCount_ = 0U;
    bool enabled_ = false;
    bool hasWave_ = false;
};

}  // namespace streaming_li_get
}  // namespace QLIKernel

#endif  // A5_STREAMING_QUANT_LI_GET_TILE_READY_H
