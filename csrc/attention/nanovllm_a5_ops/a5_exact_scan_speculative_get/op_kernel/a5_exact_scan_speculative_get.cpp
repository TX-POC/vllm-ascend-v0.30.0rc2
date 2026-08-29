/**
 * Device-resident ExactScoreScan + full SpeculativeGet.
 *
 * This is the P5 executor side of the C8 publication seam.  It consumes the
 * exact sortable uint16 score bytes already written by source-derived C8 in
 * `score_workspace`; it does not recreate Q/K math and it does not mutate
 * the official C8 Top-K state.  `tile_receipts` identifies the exact score
 * windows, and its epoch is the producer/consumer publication boundary.
 *
 * Task lanes 0 and 1 own the two fixed Direct-HCCP queues.  Task lanes 2
 * through 63 ExactScoreScan the C8 windows, compact cold candidates into
 * CandidateSeed records, then publish `(sourceOffset, objectOffset,
 * candidateSeedSlot)` to two SPSC rings.
 *
 *   compacted miss descriptors -> WQE/SQ -> doorbell -> CQ terminal wait.
 *
 * P2 changes only the parallel transport timing: each Get lane double-buffers
 * its compacted descriptor waves, posts one Direct-HCCP wave, then compacts
 * the next wave into the alternate buffer while that one is in flight.  The
 * one-in-flight-per-queue contract is retained.  A strict serial mode retains
 * the old blocking helper and therefore remains the same-engine reference.
 *
 * The fixed Get lane owns the private-arena allocation for every object routed
 * to it (`object & 1`).  Consequently identical compressed objects never
 * contend across lanes.  It posts a real Direct-HCCP Get without waiting, then
 * marks the corresponding private slot READY only after terminal CQ.  A failed
 * terminal CQ becomes FAILED so ExactTopK/GuaranteedRecall can supply the
 * required data; persistent resident mappings are never changed here.
 */

#include "kernel_operator.h"

#include "a5_exact_scan_speculative_get_tiling.h"
// This header is staged from the exact KVAF checkout by build_nanovllm_ops.sh.
// It contains the shared Direct-HCCP ABI and the current full Get device path.
#include "kvaf_fused_full_get_device.h"

using namespace AscendC;

namespace {
constexpr uint32_t kProducerCount = 62;
constexpr uint32_t kGetLaneCount = 2;
// The Direct-HCCP entities are created by the existing KVAF two-worker
// runtime.  Keep their device task lanes at 0 and 1, the same positions used
// by the standalone AIV-only Get kernel.  The remaining task lanes are
// disjoint ExactScan producers.  This is a task-lane partition, not an
// unsupported claim about public physical-core affinity.
constexpr uint32_t kProducerAivBegin = kGetLaneCount;
constexpr uint32_t kCacheLineBytes = 64;
constexpr uint32_t kCacheLineWords = kCacheLineBytes / sizeof(int64_t);
constexpr uint32_t kRingSlots = 2;
constexpr uint32_t kRecordsPerRingSlot = 32;
constexpr uint32_t kRecordWords = 3;
// A producer and its Get lane exchange these words through a tiny SPSC ring.
// These are cross-AIV handoff words, so they deliberately bypass DCache: a
// DCCI on one AIV must never publish an old cache line over another AIV's
// record/count/state update.  The fields still occupy separate cache lines so
// the compact layout remains safe if surrounding code later needs DCCI.
constexpr uint32_t kRingSlotStateWord = 0U;
constexpr uint32_t kRingSlotCountWord = kCacheLineWords;
// The consumer cursor has its own cache line.  A CandidateSeed chunk can
// contain aliases of the same compressed KV object, so the number of raw ring
// records is not the number of HCCP descriptors it produces.  Keeping a
// cursor lets a Get lane stop exactly at a descriptor-wave boundary without
// dropping the remaining records in a ready SPSC slot.
constexpr uint32_t kRingSlotCursorWord = 2U * kCacheLineWords;
constexpr uint32_t kRingSlotRecordWord = 3U * kCacheLineWords;
constexpr uint32_t kRingWordsPerSlot =
    kRingSlotRecordWord + kRecordsPerRingSlot * kRecordWords;
constexpr uint32_t kRingClosedWord = kRingSlots * kRingWordsPerSlot;
constexpr uint32_t kRingWords = kRingClosedWord + kCacheLineWords;
constexpr uint32_t kGetBootstrapRecords = 64;
constexpr uint32_t kGetWaveRecords = 256;
// P2 keeps each posted descriptor wave immutable until its CQ terminal state
// is observed.  The WQEs are already materialized into SQ at PostOnly return,
// but retaining the source buffer gives the fused path a simple auditable
// lifetime rule and lets the Get lane compact the next exact wave safely.
constexpr uint32_t kGetDescriptorBufferCount = 2;
// The final two words make live C8 receipt progress observable.  They are
// deliberately terminal telemetry rather than another cross-kernel control
// channel: [8] is the total ready-epoch polls; [9] is the number of expected
// score windows which never became ready before the bounded wait expired.
constexpr uint32_t kProducerStatColumns = 10;
constexpr uint32_t kGetStatColumns = 16;
constexpr uint32_t kPackedEntryBytes = 656;
constexpr uint32_t kCandidateSeedWords = 5;
constexpr uint32_t kReceiptWords = 16;
constexpr uint32_t kReceiptTileWidth = 128;
constexpr uint32_t kReceiptContractVersion = 1;
constexpr uint32_t kReceiptReadyEpochWord = 8;
constexpr uint32_t kReceiptByteOffsetWord = 0;
constexpr uint32_t kReceiptRowWord = 1;
constexpr uint32_t kReceiptSourceBeginWord = 2;
constexpr uint32_t kReceiptValidCountWord = 3;
constexpr uint32_t kReceiptScoreStrideWord = 4;
constexpr uint32_t kReceiptVersionWord = 7;
constexpr int32_t kPrivateSlotUnmapped = -1;
constexpr int32_t kCandidateSeedDeferred = -2;
constexpr int32_t kPrivateReadyFree = 0;
constexpr int32_t kPrivateReadyPending = 1;
constexpr int32_t kPrivateReadyReady = 2;
constexpr int32_t kPrivateReadyFailed = 3;
constexpr uint32_t kConsumerIdlePollLimit = 16U * 1024U * 1024U;
constexpr uint32_t kSerialGatePollLimit = 16U * 1024U * 1024U;
// P5 may launch beside C8 before every receipt is published.  This is a
// bounded *device* wait, not a host-side synchronization.  A timeout only
// suppresses a speculative request; Guaranteed Recall retains exactness.
constexpr uint32_t kReceiptReadyPollLimit = 16U * 1024U * 1024U;
constexpr uint32_t kReceiptPollBarrierStride = 256U;
// ExactScoreScan can produce one 128-source C8 tile before Direct-HCCP has
// retired the preceding 64-record SPSC burst.  In the normal parallel path a
// full producer slot is therefore backpressure, not a candidate-loss event.
// The wait remains bounded: a genuinely stalled Get lane must fail closed so
// GuaranteedRecall, rather than an unbounded AIV spin, preserves correctness.
constexpr uint32_t kProducerRingFreePollLimit = 16U * 1024U * 1024U;
constexpr uint32_t kProducerRingFreePollBarrierStride = 256U;
static_assert((kReceiptPollBarrierStride &
               (kReceiptPollBarrierStride - 1U)) == 0U,
              "receipt polling barrier stride must be a power of two");
static_assert((kProducerRingFreePollBarrierStride &
               (kProducerRingFreePollBarrierStride - 1U)) == 0U,
              "ring polling barrier stride must be a power of two");

constexpr uint64_t kPipelineModeParallel = 0U;
constexpr uint64_t kPipelineModeSerialWaitAllClosed = 1U;

static_assert(kRingWords % kCacheLineWords == 0U,
              "each SPSC ring must start on a cache-line boundary");

constexpr int32_t kRoleTraceExactScanBase = 1000;
constexpr int32_t kRoleTraceGetBase = 2000;
constexpr uint64_t kGetResultProtocolError = 0x1001U;
constexpr uint64_t kGetResultConsumerTimeout = 0x1002U;
constexpr uint64_t kGetResultRingRecordError = 0x1003U;
constexpr uint64_t kGetResultInvalidPipelineMode = 0x1004U;
constexpr uint64_t kGetResultSerialGateTimeout = 0x1005U;
constexpr uint64_t kGetResultReceiptError = 0x1006U;
constexpr uint64_t kGetResultPrivateStateError = 0x1007U;
// The descriptor tuple is the exact input to Direct-HCCP PostOnly.  Keep a
// distinct result so a malformed/stale descriptor can never be mistaken for
// a CQ failure or for the private-ready bookkeeping error above.
constexpr uint64_t kGetResultDescriptorStateError = 0x1008U;
// The producer and consumer both derive raw GM addresses from the ring
// geometry.  Rejecting a stale host tiler before either side touches the ring
// turns an OPP/bridge layout mismatch into observable telemetry instead of
// allowing it to reinterpret a 232-word ring as the current 248-word ring.
constexpr uint64_t kGetResultRingGeometryError = 0x1009U;

// CandidateSeed is intentionally an int32 record so it can be consumed by a
// later graph-visible reconcile stage without reconstructing a pointer-sized
// descriptor.  The producer writes fields 0..3, then the fixed Get lane fills
// field 4 with the transaction-private physical slot.
enum CandidateSeedWord : uint32_t {
    kCandidateSeedRow = 0U,
    kCandidateSeedSource = 1U,
    kCandidateSeedObject = 2U,
    kCandidateSeedScore = 3U,
    kCandidateSeedPrivateSlot = 4U,
};

struct FusedGetControlSnapshot {
    uint64_t magic;
    uint64_t version;
    uint64_t channelEntity;
    uint64_t queueState;
    uint64_t coreStatus;
    uint64_t output;
    uint64_t outputRows;
    uint64_t blockCount;
    uint64_t jettysPerCore;
    uint64_t blockBytes;
};

// `frameEpoch` and `fusedGetControl` are cold values written through the ACL
// runtime before this graph launch.  Do not use the normal scalar GM path for
// them: a DCCI clean of an AIV's old DCache line can write the allocation's
// earlier zero-fill back to HBM just before the read.  `ld_dev` (exposed by
// ReadGmByPassDCache) avoids that DCache ownership entirely.
__aicore__ inline uint64_t LoadColdU64(__gm__ uint64_t* word)
{
    return ReadGmByPassDCache<uint64_t>(word);
}

__aicore__ inline uint64_t LoadColdEpoch(__gm__ int64_t* word)
{
    const uint64_t value = static_cast<uint64_t>(
        ReadGmByPassDCache<int64_t>(word));
    DataSyncBarrier<MemDsbT::DDR>();
    return value;
}

__aicore__ inline void FlushGmRange(__gm__ uint8_t* address, uint64_t bytes)
{
    if (address == nullptr || bytes == 0U) {
        return;
    }
    const uint64_t first =
        reinterpret_cast<uint64_t>(address) & ~(kCacheLineBytes - 1U);
    const uint64_t last =
        (reinterpret_cast<uint64_t>(address) + bytes - 1U) &
        ~(kCacheLineBytes - 1U);
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(first));
    for (uint64_t offset = 0U; offset <= last - first;
         offset += kCacheLineBytes) {
        DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE,
                                 DcciDst::CACHELINE_OUT>(global[offset]);
    }
}

// Ring records are hot producer-to-consumer traffic, unlike the cold values
// above.  Each SPSC ring has one writer and one reader, but they run on
// different AIV task lanes.  Bypass loads/stores remove DCache ownership from
// that exchange; the explicit DataSyncBarrier calls at publish/acquire points
// establish the ordering contract.
__aicore__ inline uint64_t LoadRingWord(__gm__ int64_t* word)
{
    return static_cast<uint64_t>(ReadGmByPassDCache<int64_t>(word));
}

__aicore__ inline void StoreRingWord(__gm__ int64_t* word, uint64_t value)
{
    WriteGmByPassDCache<int64_t>(word, static_cast<int64_t>(value));
}

struct RingView {
    __gm__ int64_t* words;

    __aicore__ inline __gm__ int64_t* SlotState(uint32_t slot) const
    {
        return words + slot * kRingWordsPerSlot + kRingSlotStateWord;
    }

    __aicore__ inline __gm__ int64_t* SlotCount(uint32_t slot) const
    {
        return words + slot * kRingWordsPerSlot + kRingSlotCountWord;
    }

    __aicore__ inline __gm__ int64_t* SlotCursor(uint32_t slot) const
    {
        return words + slot * kRingWordsPerSlot + kRingSlotCursorWord;
    }

    __aicore__ inline __gm__ int64_t* RecordWord(
        uint32_t slot, uint32_t record, uint32_t word) const
    {
        return words + slot * kRingWordsPerSlot + kRingSlotRecordWord +
               record * kRecordWords + word;
    }

    __aicore__ inline __gm__ int64_t* ClosedEpoch() const
    {
        return words + kRingClosedWord;
    }
};

class A5ExactScanSpeculativeGetKernel {
public:
    __aicore__ inline explicit A5ExactScanSpeculativeGetKernel(
        const A5ExactScanSpeculativeGetTilingData* tiling)
        : tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR scoreWorkspace, GM_ADDR tileReceipts, GM_ADDR streamingEpoch,
        GM_ADDR scoreSourceLens, GM_ADDR residentThreshold,
        GM_ADDR residentThresholdValid,
        GM_ADDR kvObjectSlotsSnapshot, GM_ADDR reqPoolEntries,
        GM_ADDR kvObjectMap, GM_ADDR kvObjectLens,
        GM_ADDR privateSourceToSlot, GM_ADDR privateObjectToSlot,
        GM_ADDR privateReady, GM_ADDR privateArenaHead,
        GM_ADDR privatePackedOutput, GM_ADDR addressTable, GM_ADDR validTable,
        GM_ADDR fusedGetControl, GM_ADDR pipelineState,
        GM_ADDR executionMode, GM_ADDR candidateSeeds,
        GM_ADDR candidateCounts, GM_ADDR producerStats, GM_ADDR getStats,
        GM_ADDR roleTrace, GM_ADDR getIndices, GM_ADDR getOffsets,
        GM_ADDR getOutputIndices, GM_ADDR getMask,
        GM_ADDR getCandidateSeedSlots, GM_ADDR getClaimNew)
    {
        // `scoreWorkspace` is int32 only at the graph boundary.  C8 stores
        // two exact sortable uint16 scores in each carrier word.
        scoreWorkspaceU16_ = reinterpret_cast<__gm__ uint16_t*>(scoreWorkspace);
        tileReceipts_ = reinterpret_cast<__gm__ int64_t*>(tileReceipts);
        streamingEpoch_ = reinterpret_cast<__gm__ int64_t*>(streamingEpoch);
        scoreSourceLens_ = reinterpret_cast<__gm__ int32_t*>(scoreSourceLens);
        residentThreshold_ =
            reinterpret_cast<__gm__ int32_t*>(residentThreshold);
        residentThresholdValid_ =
            reinterpret_cast<__gm__ int32_t*>(residentThresholdValid);
        kvObjectSlotsSnapshot_ =
            reinterpret_cast<__gm__ int32_t*>(kvObjectSlotsSnapshot);
        reqPoolEntries_ = reinterpret_cast<__gm__ int32_t*>(reqPoolEntries);
        kvObjectMap_ = reinterpret_cast<__gm__ int32_t*>(kvObjectMap);
        kvObjectLens_ = reinterpret_cast<__gm__ int32_t*>(kvObjectLens);
        privateSourceToSlot_ =
            reinterpret_cast<__gm__ int32_t*>(privateSourceToSlot);
        privateObjectToSlot_ =
            reinterpret_cast<__gm__ int32_t*>(privateObjectToSlot);
        privateReady_ = reinterpret_cast<__gm__ int32_t*>(privateReady);
        privateArenaHead_ = reinterpret_cast<__gm__ int32_t*>(privateArenaHead);
        privatePackedOutput_ =
            reinterpret_cast<__gm__ uint8_t*>(privatePackedOutput);
        addressTable_ = reinterpret_cast<__gm__ uint64_t*>(addressTable);
        validTable_ = reinterpret_cast<__gm__ uint8_t*>(validTable);
        fusedGetControl_ =
            reinterpret_cast<__gm__ KvafFusedBatchGetControl*>(fusedGetControl);
        pipelineState_ = reinterpret_cast<__gm__ int64_t*>(pipelineState);
        executionMode_ = reinterpret_cast<__gm__ int64_t*>(executionMode);
        candidateSeeds_ = reinterpret_cast<__gm__ int32_t*>(candidateSeeds);
        candidateCounts_ = reinterpret_cast<__gm__ int32_t*>(candidateCounts);
        producerStats_ = reinterpret_cast<__gm__ int64_t*>(producerStats);
        getStats_ = reinterpret_cast<__gm__ int64_t*>(getStats);
        roleTrace_ = reinterpret_cast<__gm__ int32_t*>(roleTrace);
        getIndices_ = reinterpret_cast<__gm__ int32_t*>(getIndices);
        getOffsets_ = reinterpret_cast<__gm__ int32_t*>(getOffsets);
        getOutputIndices_ = reinterpret_cast<__gm__ int32_t*>(getOutputIndices);
        getMask_ = reinterpret_cast<__gm__ uint8_t*>(getMask);
        getCandidateSeedSlots_ =
            reinterpret_cast<__gm__ int32_t*>(getCandidateSeedSlots);
        getClaimNew_ = reinterpret_cast<__gm__ uint8_t*>(getClaimNew);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const uint32_t taskLane = GetBlockIdx();
            if (!HasExpectedRingGeometry()) {
                if (taskLane < kGetLaneCount) {
                    WriteRoleTrace(taskLane,
                                   kRoleTraceGetBase +
                                       static_cast<int32_t>(taskLane));
                    WriteGetStats(taskLane, 0U, 0U,
                                  kGetResultRingGeometryError, 0U, 0U, 0U,
                                  0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U);
                } else {
                    const uint32_t producer = taskLane - kProducerAivBegin;
                    if (producer < kProducerCount) {
                        WriteRoleTrace(
                            taskLane,
                            kRoleTraceExactScanBase +
                                static_cast<int32_t>(producer));
                        WriteGmByPassDCache<int32_t>(candidateCounts_ + producer,
                                                     0);
                        WriteProducerStats(
                            producer, 0U, 0U, 0U, 0U,
                            static_cast<uint64_t>(-1), 0U,
                            static_cast<uint64_t>(GetSystemCycle()), 0U, 0U,
                            0U);
                    }
                }
                return;
            }
            if (taskLane < kGetLaneCount) {
                ProcessFullGetLane(taskLane);
                return;
            }
            const uint32_t producer = taskLane - kProducerAivBegin;
            if (producer < tiling_->producerCount) {
                ProcessExactScanProducer(producer);
            }
        }
    }

private:
    __aicore__ inline bool HasExpectedRingGeometry() const
    {
        return tiling_->producerCount == kProducerCount &&
               tiling_->getLaneCount == kGetLaneCount &&
               tiling_->ringSlots == kRingSlots &&
               tiling_->recordsPerRingSlot == kRecordsPerRingSlot &&
               tiling_->pipelineWordsPerRing == kRingWords;
    }

    __aicore__ inline RingView Ring(uint32_t producer, uint32_t lane) const
    {
        const uint32_t ringIndex = producer * kGetLaneCount + lane;
        return RingView{pipelineState_ +
                        static_cast<uint64_t>(ringIndex) *
                            tiling_->pipelineWordsPerRing};
    }

    __aicore__ inline uint64_t DescriptorBufferOffset(
        uint32_t lane, uint32_t buffer) const
    {
        return (static_cast<uint64_t>(lane) * kGetDescriptorBufferCount +
                buffer) *
               kGetWaveRecords;
    }

    __aicore__ inline __gm__ int32_t* GetIndicesBuffer(
        uint32_t lane, uint32_t buffer) const
    {
        return getIndices_ + DescriptorBufferOffset(lane, buffer);
    }

    __aicore__ inline __gm__ int32_t* GetOffsetsBuffer(
        uint32_t lane, uint32_t buffer) const
    {
        return getOffsets_ + DescriptorBufferOffset(lane, buffer);
    }

    __aicore__ inline __gm__ int32_t* GetOutputIndicesBuffer(
        uint32_t lane, uint32_t buffer) const
    {
        return getOutputIndices_ + DescriptorBufferOffset(lane, buffer);
    }

    __aicore__ inline __gm__ uint8_t* GetMaskBuffer(
        uint32_t lane, uint32_t buffer) const
    {
        return getMask_ + DescriptorBufferOffset(lane, buffer);
    }

    __aicore__ inline __gm__ int32_t* GetCandidateSeedSlotsBuffer(
        uint32_t lane, uint32_t buffer) const
    {
        return getCandidateSeedSlots_ + DescriptorBufferOffset(lane, buffer);
    }

    __aicore__ inline __gm__ uint8_t* GetClaimNewBuffer(
        uint32_t lane, uint32_t buffer) const
    {
        return getClaimNew_ + DescriptorBufferOffset(lane, buffer);
    }

    __aicore__ inline uint64_t CandidateSeedOffset(
        uint32_t producer, uint32_t localIndex) const
    {
        return (static_cast<uint64_t>(producer) *
                    tiling_->candidateCapacityPerProducer +
                localIndex) *
               tiling_->candidateSeedWords;
    }

    __aicore__ inline __gm__ int32_t* CandidateSeed(
        uint32_t producer, uint32_t localIndex) const
    {
        return candidateSeeds_ + CandidateSeedOffset(producer, localIndex);
    }

    __aicore__ inline __gm__ int32_t* CandidateSeedByAbsolute(
        uint64_t absoluteSlot) const
    {
        return candidateSeeds_ +
               absoluteSlot * tiling_->candidateSeedWords;
    }

    __aicore__ inline void PublishRingSlot(
        const RingView& ring, uint32_t slot, uint32_t count, uint64_t epoch)
    {
        // The three record words are bypass-GM stores.  On the A5 path their
        // completion must be released *before* the count is published, then
        // the count must be released before `state = epoch` becomes visible.
        // `state` is the sole ready flag polled by the Get lane; collapsing
        // the two boundaries can therefore produce a successful Direct-HCCP
        // CQ for an old-but-valid descriptor triple.
        // A slot may have been partly consumed in a prior graph replay.  The
        // producer resets the private consumer cursor before publishing a new
        // payload; state remains the sole ready flag.
        StoreRingWord(ring.SlotCursor(slot), 0U);
        DataSyncBarrier<MemDsbT::DDR>();
        StoreRingWord(ring.SlotCount(slot), count);
        DataSyncBarrier<MemDsbT::DDR>();
        StoreRingWord(ring.SlotState(slot), epoch);
    }

    // The Get lane clears SlotState only after it has acquired the producer
    // payload and advanced its private cursor.  A producer may reuse that
    // slot only after seeing the clear and taking the corresponding DDR
    // acquire edge.  This wait must be disabled in serial_wait_all_closed:
    // that reference intentionally does not start its consumers until every
    // producer has closed, so waiting there would create a circular wait.
    __aicore__ inline bool WaitForRingWriteSlot(
        const RingView& ring, uint32_t slot, uint64_t epoch,
        bool allowConsumerProgress) const
    {
        if (LoadRingWord(ring.SlotState(slot)) != epoch) {
            DataSyncBarrier<MemDsbT::DDR>();
            return true;
        }
        if (!allowConsumerProgress) {
            return false;
        }
        for (uint32_t poll = 0U; poll < kProducerRingFreePollLimit; ++poll) {
            if (LoadRingWord(ring.SlotState(slot)) != epoch) {
                DataSyncBarrier<MemDsbT::DDR>();
                return true;
            }
            if ((poll & (kProducerRingFreePollBarrierStride - 1U)) ==
                (kProducerRingFreePollBarrierStride - 1U)) {
                DataSyncBarrier<MemDsbT::DDR>();
            }
        }
        return false;
    }

    __aicore__ inline bool TryAppendCandidate(
        const RingView& ring, uint32_t* writeSlot, uint32_t* writeCount,
        uint64_t epoch, uint64_t sourceOffset, uint64_t objectOffset,
        uint64_t candidateSlot, uint64_t* firstPublishCycle,
        bool allowConsumerProgress)
    {
        if (*writeCount == 0U &&
            !WaitForRingWriteSlot(ring, *writeSlot, epoch,
                                  allowConsumerProgress)) {
            return false;
        }
        // The executor deliberately does not let a producer allocate a
        // private slot.  These three immutable words are sufficient for the
        // fixed owner lane to deduplicate by compressed object, allocate its
        // private arena, materialize the legacy Direct-HCCP descriptor and
        // update CandidateSeed.private_slot.
        __gm__ int64_t* const sourceWord =
            ring.RecordWord(*writeSlot, *writeCount, 0);
        WriteGmByPassDCache<int64_t>(sourceWord,
                                     static_cast<int64_t>(sourceOffset));
        WriteGmByPassDCache<int64_t>(sourceWord + 1,
                                     static_cast<int64_t>(objectOffset));
        WriteGmByPassDCache<int64_t>(sourceWord + 2,
                                     static_cast<int64_t>(candidateSlot));
        ++(*writeCount);
        if (*writeCount == kRecordsPerRingSlot) {
            PublishRingSlot(ring, *writeSlot, *writeCount, epoch);
            if (*firstPublishCycle == 0U) {
                *firstPublishCycle = static_cast<uint64_t>(GetSystemCycle());
            }
            *writeSlot = 1U - *writeSlot;
            *writeCount = 0U;
        }
        return true;
    }

    __aicore__ inline void FlushProducerTail(
        const RingView& ring, uint32_t writeSlot, uint32_t writeCount,
        uint64_t epoch, uint64_t* firstPublishCycle, uint64_t* overflow,
        uint64_t* deferred, bool allowConsumerProgress)
    {
        if (writeCount != 0U) {
            if (!WaitForRingWriteSlot(ring, writeSlot, epoch,
                                      allowConsumerProgress)) {
                *overflow = 1U;
                for (uint32_t record = 0U; record < writeCount; ++record) {
                    const uint64_t candidateSlot = static_cast<uint64_t>(
                        LoadRingWord(ring.RecordWord(writeSlot, record, 2U)));
                    if (candidateSlot <
                        static_cast<uint64_t>(tiling_->producerCount) *
                            tiling_->candidateCapacityPerProducer) {
                        WriteGmByPassDCache<int32_t>(
                            CandidateSeedByAbsolute(candidateSlot) +
                                kCandidateSeedPrivateSlot,
                            kCandidateSeedDeferred);
                    }
                }
                *deferred += writeCount;
            } else {
                PublishRingSlot(ring, writeSlot, writeCount, epoch);
                if (*firstPublishCycle == 0U) {
                    *firstPublishCycle = static_cast<uint64_t>(GetSystemCycle());
                }
            }
        }
        // `closed` is also a release flag: do not let a Get lane conclude the
        // scan before every preceding ready-slot publication is visible.
        DataSyncBarrier<MemDsbT::DDR>();
        StoreRingWord(ring.ClosedEpoch(), epoch);
    }

    __aicore__ inline uint64_t LoadReceiptWord(
        uint64_t receipt, uint32_t word) const
    {
        return LoadRingWord(tileReceipts_ + receipt * kReceiptWords + word);
    }

    // A P5 consumer is allowed to start while C8 is still producing score
    // tiles.  The fixed receipt slot itself tells us *which* tile belongs at
    // this position; `score_source_lens[row]` tells us whether that slot is
    // expected to exist in this forward.  We must distinguish an inactive
    // fixed slot from an expected-but-not-yet-published one: the former is
    // skipped, while the latter gets a bounded device-side readiness wait.
    // A wait expiry suppresses only speculation.  ExactTopK/GuaranteedRecall
    // remains the mathematical authority for any data not made READY here.
    __aicore__ inline bool WaitForReceiptReady(
        uint64_t receipt, uint64_t epoch, uint64_t* readyPolls) const
    {
        if (receipt >= tiling_->tileReceiptCount) {
            return false;
        }
        for (uint32_t poll = 0U; poll < kReceiptReadyPollLimit; ++poll) {
            if (LoadReceiptWord(receipt, kReceiptReadyEpochWord) == epoch) {
                // Acquire after observing the producer's ready epoch.  The
                // publisher issues score-window DMA, immutable descriptor,
                // then epoch in that order.  The final cross-stream visibility
                // contract is separately gated by the physical-NPU test; this
                // barrier prevents this AIV's following bypass loads from
                // moving ahead of its own ready observation.
                DataSyncBarrier<MemDsbT::DDR>();
                return true;
            }
            if (readyPolls != nullptr) {
                ++(*readyPolls);
            }
            if ((poll & (kReceiptPollBarrierStride - 1U)) ==
                (kReceiptPollBarrierStride - 1U)) {
                DataSyncBarrier<MemDsbT::DDR>();
            }
        }
        return false;
    }

    __aicore__ inline bool IsReceiptValid(
        uint64_t receipt, uint64_t epoch, uint64_t* row,
        uint64_t* sourceBegin, uint64_t* validCount,
        uint64_t* scoreOffsetU16) const
    {
        if (row == nullptr || sourceBegin == nullptr || validCount == nullptr ||
            scoreOffsetU16 == nullptr || receipt >= tiling_->tileReceiptCount ||
            tiling_->scoreWorkspaceWords == 0U) {
            return false;
        }
        // Receipt line 0 is immutable descriptor data and line 1 carries the
        // ready epoch.  The graph/device acquire edge is supplied by the
        // caller; this read pair preserves the receipt's release ordering.
        if (LoadReceiptWord(receipt, kReceiptReadyEpochWord) != epoch ||
            LoadReceiptWord(receipt, kReceiptVersionWord) !=
                kReceiptContractVersion) {
            return false;
        }
        DataSyncBarrier<MemDsbT::DDR>();
        // The C8 publisher writes word 0 as a byte *offset* from the start
        // of the caller-owned score_workspace.  It is deliberately not a
        // device virtual address: graph replay may reuse the same logical
        // receipt with a different physical allocation.  Form the GM address
        // only after validating this relative offset below.
        const uint64_t byteOffset =
            LoadReceiptWord(receipt, kReceiptByteOffsetWord);
        *row = LoadReceiptWord(receipt, kReceiptRowWord);
        *sourceBegin = LoadReceiptWord(receipt, kReceiptSourceBeginWord);
        *validCount = LoadReceiptWord(receipt, kReceiptValidCountWord);
        const uint64_t scoreStride =
            LoadReceiptWord(receipt, kReceiptScoreStrideWord);
        if (*row >= tiling_->rowCount || *sourceBegin >= tiling_->sourceCapacity ||
            *validCount == 0U ||
            *validCount > tiling_->sourceCapacity - *sourceBegin ||
            scoreStride < tiling_->sourceCapacity ||
            (byteOffset & (sizeof(uint16_t) - 1U)) != 0U) {
            return false;
        }
        *scoreOffsetU16 = byteOffset / sizeof(uint16_t);
        const uint64_t workspaceScores =
            static_cast<uint64_t>(tiling_->scoreWorkspaceWords) * 2U;
        return *scoreOffsetU16 <= workspaceScores &&
               *validCount <= workspaceScores - *scoreOffsetU16;
    }

    __aicore__ inline void WriteCandidateSeed(
        uint32_t producer, uint32_t localIndex, uint32_t row,
        uint32_t source, int32_t kvObject, uint16_t score,
        int32_t privateSlot) const
    {
        __gm__ int32_t* const seed = CandidateSeed(producer, localIndex);
        WriteGmByPassDCache<int32_t>(seed + kCandidateSeedRow,
                                     static_cast<int32_t>(row));
        WriteGmByPassDCache<int32_t>(seed + kCandidateSeedSource,
                                     static_cast<int32_t>(source));
        WriteGmByPassDCache<int32_t>(seed + kCandidateSeedObject, kvObject);
        WriteGmByPassDCache<int32_t>(seed + kCandidateSeedScore,
                                     static_cast<int32_t>(score));
        WriteGmByPassDCache<int32_t>(seed + kCandidateSeedPrivateSlot,
                                     privateSlot);
    }

    __aicore__ inline void ProcessExactScanProducer(uint32_t producer)
    {
        const uint64_t epoch = LoadColdEpoch(streamingEpoch_);
        const bool allowConsumerProgress =
            LoadColdEpoch(executionMode_) == kPipelineModeParallel;
        WriteRoleTrace(kProducerAivBegin + producer,
                       kRoleTraceExactScanBase + static_cast<int32_t>(producer));
        if (epoch == 0U || tiling_->candidateSeedWords != kCandidateSeedWords ||
            tiling_->producerCount != kProducerCount ||
            tiling_->getLaneCount != kGetLaneCount ||
            tiling_->rowsPerEntry == 0U ||
            tiling_->privateArenaCapacity == 0U ||
            tiling_->outputRows <
                tiling_->privateArenaCapacity * kGetLaneCount) {
            WriteGmByPassDCache<int32_t>(candidateCounts_ + producer, 0);
            WriteProducerStats(producer, 0U, 0U, 0U, 0U,
                               static_cast<uint64_t>(-1), 0U,
                               static_cast<uint64_t>(GetSystemCycle()), 0U,
                               0U, 0U);
            return;
        }

        uint32_t writeSlot[kGetLaneCount] = {0U, 0U};
        uint32_t writeCount[kGetLaneCount] = {0U, 0U};
        uint64_t laneOverflow[kGetLaneCount] = {0U, 0U};
        uint64_t firstPublishCycle[kGetLaneCount] = {0U, 0U};
        uint64_t selected = 0U;
        uint64_t hit = 0U;
        uint64_t miss = 0U;
        uint64_t enqueued = 0U;
        uint64_t deferred = 0U;
        uint64_t receiptReadyPolls = 0U;
        uint64_t receiptReadyTimeouts = 0U;
        uint32_t candidateCount = 0U;

        const uint64_t tilesPerRow =
            (static_cast<uint64_t>(tiling_->sourceCapacity) +
             kReceiptTileWidth - 1U) /
            kReceiptTileWidth;
        if (tilesPerRow == 0U) {
            WriteGmByPassDCache<int32_t>(candidateCounts_ + producer, 0);
            WriteProducerStats(producer, 0U, 0U, 0U, 0U,
                               static_cast<uint64_t>(-1), 0U,
                               static_cast<uint64_t>(GetSystemCycle()), 0U,
                               receiptReadyPolls, receiptReadyTimeouts);
            return;
        }

        // Receipt stripe ownership is deterministic and independent of
        // runtime completion order.  A fixed receipt position gives the
        // expected [row, source) window; a ready receipt gives the exact C8
        // score bytes for that same window.  No producer recreates Q/K math.
        for (uint64_t receipt = producer; receipt < tiling_->tileReceiptCount;
             receipt += tiling_->producerCount) {
            const uint64_t expectedRow = receipt / tilesPerRow;
            const uint64_t expectedSourceBegin =
                (receipt % tilesPerRow) * kReceiptTileWidth;
            if (expectedRow >= tiling_->rowCount) {
                ++deferred;
                continue;
            }
            const int32_t sourceLen = ReadGmByPassDCache<int32_t>(
                scoreSourceLens_ + expectedRow);
            if (sourceLen < 0 ||
                static_cast<uint32_t>(sourceLen) > tiling_->sourceCapacity) {
                ++deferred;
                continue;
            }
            // The C8 graph has a fixed receipt allocation, but a decode row
            // may have fewer active score sources.  These inert slots are not
            // timeout/failure cases and must not become fake zero-score KV.
            if (expectedSourceBegin >= static_cast<uint32_t>(sourceLen)) {
                continue;
            }
            if (!WaitForReceiptReady(receipt, epoch, &receiptReadyPolls)) {
                ++deferred;
                ++receiptReadyTimeouts;
                continue;
            }
            uint64_t row64 = 0U;
            uint64_t sourceBegin64 = 0U;
            uint64_t validCount64 = 0U;
            uint64_t scoreOffsetU16 = 0U;
            if (!IsReceiptValid(receipt, epoch, &row64, &sourceBegin64,
                                &validCount64, &scoreOffsetU16)) {
                ++deferred;
                continue;
            }
            const uint64_t expectedValidCount =
                (static_cast<uint64_t>(sourceLen) - expectedSourceBegin) <
                        kReceiptTileWidth
                    ? static_cast<uint64_t>(sourceLen) - expectedSourceBegin
                    : kReceiptTileWidth;
            // Receipt content is part of the mathematical contract.  If an
            // epoch appears in the wrong fixed slot (or carries a partial
            // range inconsistent with C8's row length), do not speculate.
            // GuaranteedRecall makes this fail-closed rather than changing
            // the final Top-K/QSFA result.
            if (row64 != expectedRow || sourceBegin64 != expectedSourceBegin ||
                validCount64 != expectedValidCount) {
                ++deferred;
                continue;
            }
            const uint32_t row = static_cast<uint32_t>(row64);
            const int32_t thresholdValid =
                ReadGmByPassDCache<int32_t>(residentThresholdValid_ + row);
            if (thresholdValid == 0) {
                continue;
            }
            const uint16_t threshold = static_cast<uint16_t>(
                ReadGmByPassDCache<int32_t>(residentThreshold_ + row));
            const int32_t pool =
                ReadGmByPassDCache<int32_t>(reqPoolEntries_ + row);
            const int32_t objectLen =
                ReadGmByPassDCache<int32_t>(kvObjectLens_ + row);
            if (pool < 0 || static_cast<uint32_t>(pool) >= tiling_->poolSize ||
                objectLen < 0 ||
                static_cast<uint32_t>(objectLen) > tiling_->kvObjectCapacity) {
                ++deferred;
                continue;
            }

            for (uint64_t local = 0U; local < validCount64; ++local) {
                const uint32_t source = static_cast<uint32_t>(sourceBegin64 + local);
                const uint16_t score = ReadGmByPassDCache<uint16_t>(
                    scoreWorkspaceU16_ + scoreOffsetU16 + local);
                if (score < threshold) {
                    continue;
                }
                ++selected;
                const uint64_t mapSourceOffset =
                    static_cast<uint64_t>(pool) * tiling_->sourceCapacity +
                    source;
                // The map is [pool, source_capacity].  A source beyond the
                // mapped compressed-object capacity is invalid, not a legal
                // zero-valued object.
                if (source >= tiling_->sourceCapacity) {
                    ++deferred;
                    continue;
                }
                const int32_t kvObject =
                    ReadGmByPassDCache<int32_t>(kvObjectMap_ + mapSourceOffset);
                if (kvObject < 0 || static_cast<uint32_t>(kvObject) >=
                                        static_cast<uint32_t>(objectLen) ||
                    static_cast<uint32_t>(kvObject) >= tiling_->kvObjectCapacity) {
                    ++deferred;
                    continue;
                }
                const uint64_t compressedObjectOffset =
                    static_cast<uint64_t>(pool) * tiling_->kvObjectCapacity +
                    static_cast<uint32_t>(kvObject);
                if (ReadGmByPassDCache<int32_t>(kvObjectSlotsSnapshot_ +
                                                compressedObjectOffset) >= 0) {
                    ++hit;
                    continue;
                }
                ++miss;
                if (candidateCount >= tiling_->candidateCapacityPerProducer) {
                    ++deferred;
                    continue;
                }
                const uint64_t candidateSlot =
                    static_cast<uint64_t>(producer) *
                        tiling_->candidateCapacityPerProducer +
                    candidateCount;
                WriteCandidateSeed(producer, candidateCount, row, source,
                                   kvObject, score, kPrivateSlotUnmapped);
                ++candidateCount;
                // CandidateSeed must become visible before its ring ready
                // word.  CandidateSeed and the ring records are both bypass
                // GM stores; PublishRingSlot supplies their single release
                // boundary.  Do not DCCI-clean an unrelated stale cache line
                // after those bypass stores.

                const uint32_t lane =
                    static_cast<uint32_t>(kvObject) & (kGetLaneCount - 1U);
                if (laneOverflow[lane] != 0U ||
                    !TryAppendCandidate(Ring(producer, lane), &writeSlot[lane],
                                        &writeCount[lane], epoch,
                                        static_cast<uint64_t>(row) *
                                            tiling_->sourceCapacity + source,
                                        compressedObjectOffset, candidateSlot,
                                        &firstPublishCycle[lane],
                                        allowConsumerProgress)) {
                    laneOverflow[lane] = 1U;
                    WriteGmByPassDCache<int32_t>(
                        CandidateSeedByAbsolute(candidateSlot) +
                            kCandidateSeedPrivateSlot,
                        kCandidateSeedDeferred);
                    ++deferred;
                    continue;
                }
                ++enqueued;
            }
        }

        const uint64_t scanDone = static_cast<uint64_t>(GetSystemCycle());
        for (uint32_t lane = 0U; lane < kGetLaneCount; ++lane) {
            FlushProducerTail(Ring(producer, lane), writeSlot[lane],
                              writeCount[lane], epoch,
                              &firstPublishCycle[lane], &laneOverflow[lane],
                              &deferred, allowConsumerProgress);
        }
        WriteGmByPassDCache<int32_t>(candidateCounts_ + producer,
                                     static_cast<int32_t>(candidateCount));
        DataSyncBarrier<MemDsbT::DDR>();
        const uint64_t firstPublish = firstPublishCycle[0] != 0U
                                          ? firstPublishCycle[0]
                                          : firstPublishCycle[1];
        const uint64_t closedPublish = static_cast<uint64_t>(GetSystemCycle());
        WriteProducerStats(producer, selected, hit, miss, enqueued, deferred,
                           firstPublish, scanDone, closedPublish,
                           receiptReadyPolls, receiptReadyTimeouts);
    }

    __aicore__ inline bool IsValidDirectGetDescriptor(
        int32_t tableIndex, int32_t offset, int32_t outputIndex) const
    {
        if (tableIndex < 0 ||
            static_cast<uint32_t>(tableIndex) >= tiling_->tableCapacity ||
            offset < 0 || outputIndex < 0 ||
            static_cast<uint32_t>(outputIndex) >= tiling_->outputRows ||
            static_cast<uint32_t>(offset) % kPackedEntryBytes != 0U ||
            static_cast<uint64_t>(static_cast<uint32_t>(offset)) +
                    kPackedEntryBytes >
                KVAF_AIV_BATCH_GET_ADDRESS_TABLE_ENTRY_BYTES) {
            return false;
        }
        const uint32_t index = static_cast<uint32_t>(tableIndex);
        return validTable_[index] != 0U && addressTable_[index] != 0U;
    }

    __aicore__ inline void WriteProducerStats(
        uint32_t producer, uint64_t selected, uint64_t hit, uint64_t miss,
        uint64_t enqueued, uint64_t deferred, uint64_t firstPublish,
        uint64_t scanDone, uint64_t closedPublish, uint64_t receiptReadyPolls,
        uint64_t receiptReadyTimeouts)
    {
        __gm__ int64_t* stats = producerStats_ +
                                static_cast<uint64_t>(producer) *
                                    kProducerStatColumns;
        // Columns 0..7 are the original producer lifecycle.  Columns 8..9
        // make live C8 receipt pressure observable instead of silently
        // treating a late tile as an inactive one.  Bypassing DCache keeps
        // this terminal telemetry independent of the hot SPSC-ring protocol.
        WriteGmByPassDCache<int64_t>(stats + 0U,
                                     static_cast<int64_t>(selected));
        WriteGmByPassDCache<int64_t>(stats + 1U,
                                     static_cast<int64_t>(hit));
        WriteGmByPassDCache<int64_t>(stats + 2U,
                                     static_cast<int64_t>(miss));
        WriteGmByPassDCache<int64_t>(stats + 3U,
                                     static_cast<int64_t>(enqueued));
        WriteGmByPassDCache<int64_t>(stats + 4U,
                                     static_cast<int64_t>(deferred));
        WriteGmByPassDCache<int64_t>(stats + 5U,
                                     static_cast<int64_t>(firstPublish));
        WriteGmByPassDCache<int64_t>(stats + 6U,
                                     static_cast<int64_t>(scanDone));
        WriteGmByPassDCache<int64_t>(stats + 7U,
                                     static_cast<int64_t>(closedPublish));
        WriteGmByPassDCache<int64_t>(stats + 8U,
                                     static_cast<int64_t>(receiptReadyPolls));
        WriteGmByPassDCache<int64_t>(stats + 9U,
                                     static_cast<int64_t>(receiptReadyTimeouts));
        // This is the terminal publish for this producer.  The barrier waits
        // for its preceding bypass stores (including candidate output) before
        // the task reports completion to the stream/host.
        DataSyncBarrier<MemDsbT::DDR>();
    }

    __aicore__ inline void WriteRoleTrace(uint32_t taskLane, int32_t value)
    {
        // 16 int32 trace entries occupy one cache line.  This is terminal
        // telemetry, not a handoff, so direct scalar stores are sufficient
        // and avoid cross-AIV cache-line ownership entirely.
        WriteGmByPassDCache<int32_t>(roleTrace_ + taskLane, value);
    }

    __aicore__ inline bool AllRingsClosed(uint32_t lane, uint64_t epoch) const
    {
        for (uint32_t producer = 0U; producer < kProducerCount; ++producer) {
            if (LoadRingWord(Ring(producer, lane).ClosedEpoch()) != epoch) {
                return false;
            }
        }
        // `ClosedEpoch` is the producer's terminal release store: every
        // tail record/state publication precedes it.  A Get lane that sees
        // all rings closed must acquire that release *before* its final
        // ConsumeReadyRings pass.  Without this fence, a consumer can read
        // an earlier SlotState=0, subsequently observe ClosedEpoch=epoch,
        // and incorrectly exit with the producer's final partial slot still
        // unclaimed.  The normal streaming path remains nonblocking until
        // this terminal observation; this barrier applies only to the
        // closed-tail drain.
        DataSyncBarrier<MemDsbT::DDR>();
        return true;
    }

    // The strict serial reference is intentionally stronger than the normal
    // per-Get-lane completion condition: every producer must have published
    // the tail/closed word for *both* HCCP queues before either Get lane may
    // copy a descriptor or ring a doorbell.  Checking only `lane` would allow
    // lane 0 to start while a producer still has lane 1 open.
    __aicore__ inline bool AllProducerRingsClosed(uint64_t epoch) const
    {
        for (uint32_t producer = 0U; producer < kProducerCount; ++producer) {
            for (uint32_t lane = 0U; lane < kGetLaneCount; ++lane) {
                if (LoadRingWord(Ring(producer, lane).ClosedEpoch()) != epoch) {
                    return false;
                }
            }
        }
        DataSyncBarrier<MemDsbT::DDR>();
        return true;
    }

    __aicore__ inline uint32_t ConsumeReadyRings(
        uint32_t lane, uint32_t descriptorBuffer, uint64_t epoch, uint32_t queued,
        uint32_t consumeLimit, uint64_t* consumedChunks,
        uint64_t* firstConsumedChunkCycle, uint64_t* descriptorHashSum,
        uint64_t* descriptorHashXor, uint64_t* protocolError)
    {
        if (descriptorBuffer >= kGetDescriptorBufferCount) {
            *protocolError = kGetResultRingRecordError;
            return queued;
        }
        __gm__ int32_t* laneIndices =
            GetIndicesBuffer(lane, descriptorBuffer);
        __gm__ int32_t* laneOffsets =
            GetOffsetsBuffer(lane, descriptorBuffer);
        __gm__ int32_t* laneOutputIndices =
            GetOutputIndicesBuffer(lane, descriptorBuffer);
        __gm__ uint8_t* laneMask = GetMaskBuffer(lane, descriptorBuffer);
        __gm__ int32_t* laneCandidateSeedSlots =
            GetCandidateSeedSlotsBuffer(lane, descriptorBuffer);
        __gm__ uint8_t* laneClaimNew = GetClaimNewBuffer(lane, descriptorBuffer);

        for (uint32_t producer = 0U;
             producer < kProducerCount && queued < consumeLimit;
             ++producer) {
            const RingView ring = Ring(producer, lane);
            for (uint32_t slot = 0U;
                 slot < kRingSlots && queued < consumeLimit; ++slot) {
                if (LoadRingWord(ring.SlotState(slot)) != epoch) {
                    continue;
                }
                // Acquire after observing ready.  The producer's release
                // barrier makes the payload and count visible before this
                // point; this barrier prevents following reads from moving
                // ahead of the ready observation.
                DataSyncBarrier<MemDsbT::DDR>();
                const uint64_t count64 = LoadRingWord(ring.SlotCount(slot));
                const uint64_t cursor64 = LoadRingWord(ring.SlotCursor(slot));
                if (count64 == 0U || count64 > kRecordsPerRingSlot ||
                    cursor64 > count64) {
                    *protocolError = kGetResultRingRecordError;
                    StoreRingWord(ring.SlotCursor(slot), 0U);
                    StoreRingWord(ring.SlotState(slot), 0U);
                    continue;
                }
                const uint32_t count = static_cast<uint32_t>(count64);
                // A ring record describes a CandidateSeed, while `queued`
                // counts only new compressed objects that will receive an
                // HCCP descriptor.  The two units diverge when two sources
                // alias one KV object.  Consume only the prefix needed to
                // fill this descriptor wave and retain the remaining record
                // range in the slot for the next wave.
                uint32_t nextRecord = static_cast<uint32_t>(cursor64);
                // Advance the persistent cursor *before* processing a
                // record.  ``queued`` can reach ``consumeLimit`` inside the
                // body, in which case a conventional ``for (...;
                // ++nextRecord)`` would exit before its increment expression
                // runs and publish the already-consumed record as unread.
                // Keeping ``nextRecord`` equal to the first unread record is
                // the SPSC resume contract for the next descriptor wave.
                while (nextRecord < count && queued < consumeLimit) {
                    const uint32_t record = nextRecord++;
                    __gm__ int64_t* const words =
                        ring.RecordWord(slot, record, 0);
                    const uint64_t sourceOffset = LoadRingWord(words);
                    const uint64_t objectOffset = LoadRingWord(words + 1U);
                    const uint64_t candidateSlot = LoadRingWord(words + 2U);
                    const uint64_t candidateLimit =
                        static_cast<uint64_t>(tiling_->producerCount) *
                        tiling_->candidateCapacityPerProducer;
                    if (candidateSlot >= candidateLimit ||
                        sourceOffset >=
                            static_cast<uint64_t>(tiling_->rowCount) *
                                tiling_->sourceCapacity ||
                        objectOffset >=
                            static_cast<uint64_t>(tiling_->poolSize) *
                                tiling_->kvObjectCapacity) {
                        *protocolError = kGetResultRingRecordError;
                        continue;
                    }
                    __gm__ int32_t* const seed =
                        CandidateSeedByAbsolute(candidateSlot);
                    const int32_t row =
                        ReadGmByPassDCache<int32_t>(seed + kCandidateSeedRow);
                    const int32_t source = ReadGmByPassDCache<int32_t>(
                        seed + kCandidateSeedSource);
                    const int32_t kvObject = ReadGmByPassDCache<int32_t>(
                        seed + kCandidateSeedObject);
                    if (row < 0 || source < 0 || kvObject < 0 ||
                        static_cast<uint32_t>(row) >= tiling_->rowCount ||
                        static_cast<uint32_t>(source) >= tiling_->sourceCapacity ||
                        static_cast<uint32_t>(kvObject) >=
                            tiling_->kvObjectCapacity ||
                        sourceOffset != static_cast<uint64_t>(row) *
                                            tiling_->sourceCapacity +
                                        static_cast<uint32_t>(source)) {
                        WriteGmByPassDCache<int32_t>(
                            seed + kCandidateSeedPrivateSlot,
                            kCandidateSeedDeferred);
                        *protocolError = kGetResultRingRecordError;
                        continue;
                    }

                    // A source can have been seeded by an earlier C8 receipt
                    // only if it refers to the same logical source.  Reuse
                    // its private slot and do not submit a duplicate Get.
                    const int32_t sourceExisting = ReadGmByPassDCache<int32_t>(
                        privateSourceToSlot_ + sourceOffset);
                    if (sourceExisting >= 0) {
                        WriteGmByPassDCache<int32_t>(
                            seed + kCandidateSeedPrivateSlot, sourceExisting);
                        continue;
                    }
                    const int32_t objectExisting = ReadGmByPassDCache<int32_t>(
                        privateObjectToSlot_ + objectOffset);
                    if (objectExisting >= 0) {
                        WriteGmByPassDCache<int32_t>(
                            privateSourceToSlot_ + sourceOffset, objectExisting);
                        WriteGmByPassDCache<int32_t>(
                            seed + kCandidateSeedPrivateSlot, objectExisting);
                        continue;
                    }
                    if (objectExisting != kPrivateSlotUnmapped ||
                        sourceExisting != kPrivateSlotUnmapped) {
                        // `-2` is deliberately not a second ownership
                        // protocol here: a fixed lane is the sole object
                        // writer.  Seeing it means a caller violated the
                        // frame-reset contract; leave this item for fallback.
                        WriteGmByPassDCache<int32_t>(
                            seed + kCandidateSeedPrivateSlot,
                            kCandidateSeedDeferred);
                        continue;
                    }
                    const uint32_t localSlot = static_cast<uint32_t>(
                        ReadGmByPassDCache<int32_t>(privateArenaHead_ + lane));
                    if (localSlot >= tiling_->privateArenaCapacity) {
                        WriteGmByPassDCache<int32_t>(
                            seed + kCandidateSeedPrivateSlot,
                            kCandidateSeedDeferred);
                        continue;
                    }
                    const uint32_t outputIndex =
                        lane * tiling_->privateArenaCapacity + localSlot;
                    const int32_t tableIndex = static_cast<int32_t>(
                        static_cast<uint32_t>(kvObject) / tiling_->rowsPerEntry);
                    const int32_t offset = static_cast<int32_t>(
                        (static_cast<uint32_t>(kvObject) % tiling_->rowsPerEntry) *
                        kPackedEntryBytes);
                    if (!IsValidDirectGetDescriptor(tableIndex, offset,
                                                    static_cast<int32_t>(outputIndex))) {
                        WriteGmByPassDCache<int32_t>(
                            seed + kCandidateSeedPrivateSlot,
                            kCandidateSeedDeferred);
                        continue;
                    }
                    // This lane is the only writer of its arena head and all
                    // objects with `object & 1 == lane`; no global CAS is
                    // necessary.  Do not update persistent resident state.
                    WriteGmByPassDCache<int32_t>(privateArenaHead_ + lane,
                                                 static_cast<int32_t>(localSlot + 1U));
                    WriteGmByPassDCache<int32_t>(privateObjectToSlot_ + objectOffset,
                                                 static_cast<int32_t>(outputIndex));
                    WriteGmByPassDCache<int32_t>(privateSourceToSlot_ + sourceOffset,
                                                 static_cast<int32_t>(outputIndex));
                    WriteGmByPassDCache<int32_t>(privateReady_ + outputIndex,
                                                 kPrivateReadyPending);
                    WriteGmByPassDCache<int32_t>(
                        seed + kCandidateSeedPrivateSlot,
                        static_cast<int32_t>(outputIndex));
                    // These six arrays are the exact Direct-HCCP descriptor
                    // ledger.  The WQE builder and terminal READY publisher
                    // consume their GM addresses through bypass loads, so do
                    // not make an AIV DCache line owner here and then clean
                    // it later: on Ascend a clean of an old zero-filled line
                    // can overwrite a newer descriptor value.  Bypass
                    // stores plus the release barrier below give every
                    // reader the same HBM-resident tuple.
                    WriteGmByPassDCache<int32_t>(laneIndices + queued,
                                                 tableIndex);
                    WriteGmByPassDCache<int32_t>(laneOffsets + queued, offset);
                    WriteGmByPassDCache<int32_t>(
                        laneOutputIndices + queued,
                        static_cast<int32_t>(outputIndex));
                    WriteGmByPassDCache<uint8_t>(laneMask + queued, 1U);
                    WriteGmByPassDCache<int32_t>(
                        laneCandidateSeedSlots + queued,
                        static_cast<int32_t>(candidateSlot));
                    WriteGmByPassDCache<uint8_t>(laneClaimNew + queued, 1U);
                    const uint64_t fingerprint =
                        (static_cast<uint64_t>(static_cast<uint32_t>(tableIndex))
                         << 32U) ^
                        (static_cast<uint64_t>(static_cast<uint32_t>(offset))
                         << 1U) ^ static_cast<uint64_t>(outputIndex);
                    *descriptorHashSum += fingerprint;
                    *descriptorHashXor ^= fingerprint;
                    ++queued;
                }
                // Publish the next unread record before returning.  If this
                // slot was only partly consumed, state remains `epoch` and
                // the next descriptor wave resumes exactly here.  A fully
                // consumed slot is released to its SPSC producer only after
                // the cursor and all descriptor/private-state stores are
                // visible.
                DataSyncBarrier<MemDsbT::DDR>();
                StoreRingWord(ring.SlotCursor(slot), nextRecord);
                DataSyncBarrier<MemDsbT::DDR>();
                if (nextRecord == count) {
                    StoreRingWord(ring.SlotState(slot), 0U);
                    ++(*consumedChunks);
                    if (*firstConsumedChunkCycle == 0U) {
                        *firstConsumedChunkCycle =
                            static_cast<uint64_t>(GetSystemCycle());
                    }
                } else {
                    return queued;
                }
            }
        }
        return queued;
    }

    __aicore__ inline FusedGetControlSnapshot LoadFusedGetControl() const
    {
        // This 16-word ABI is populated by the cold KVAF bind.  Snapshot the
        // ten fields used by the device path through bypass loads once, then
        // use only registers for validation and WQE construction.
        __gm__ uint64_t* const words =
            reinterpret_cast<__gm__ uint64_t*>(fusedGetControl_);
        const FusedGetControlSnapshot control{
            LoadColdU64(words + 0U), LoadColdU64(words + 1U),
            LoadColdU64(words + 2U), LoadColdU64(words + 3U),
            LoadColdU64(words + 4U), LoadColdU64(words + 5U),
            LoadColdU64(words + 6U), LoadColdU64(words + 7U),
            LoadColdU64(words + 8U), LoadColdU64(words + 9U)};
        DataSyncBarrier<MemDsbT::DDR>();
        return control;
    }

    __aicore__ inline uint64_t ValidateControl(
        const FusedGetControlSnapshot& control, uint32_t lane) const
    {
        // Keep the low 16 bits compatible with the existing protocol error.
        // The upper bits make a remaining cold-bind mismatch diagnosable
        // without expanding any output tensor or KVAF ABI.
        uint64_t mismatch = 0U;
        mismatch |= control.magic != KVAF_FUSED_BATCH_GET_CONTROL_MAGIC
                        ? (1U << 0U)
                        : 0U;
        mismatch |= control.version != KVAF_FUSED_BATCH_GET_CONTROL_VERSION
                        ? (1U << 1U)
                        : 0U;
        mismatch |= control.channelEntity == 0U ? (1U << 2U) : 0U;
        mismatch |= control.queueState == 0U ? (1U << 3U) : 0U;
        mismatch |= control.coreStatus == 0U ? (1U << 4U) : 0U;
        mismatch |= control.output == 0U ? (1U << 5U) : 0U;
        mismatch |= control.output != reinterpret_cast<uint64_t>(privatePackedOutput_)
                        ? (1U << 6U)
                        : 0U;
        mismatch |= control.outputRows != tiling_->outputRows ? (1U << 7U)
                                                               : 0U;
        mismatch |= control.blockCount != kGetLaneCount ? (1U << 8U) : 0U;
        mismatch |= control.jettysPerCore != 1U ? (1U << 9U) : 0U;
        mismatch |= control.blockBytes != kPackedEntryBytes ? (1U << 10U)
                                                              : 0U;
        mismatch |= lane >= kGetLaneCount ? (1U << 11U) : 0U;
        return mismatch == 0U ? 0U
                              : kGetResultProtocolError | (mismatch << 16U);
    }

    // Materialize exactly the existing Direct-HCCP request ABI.  `buffer` is
    // part of the descriptor-storage lifetime only; it never changes the
    // logical source/table/offset/output tuple that ExactScan selected.
    __aicore__ inline uint64_t BuildFullGetArgs(
        uint32_t lane, uint32_t buffer, KvafExecuteBatchGetKernelArgs* args)
    {
        if (args == nullptr || buffer >= kGetDescriptorBufferCount) {
            return kGetResultProtocolError;
        }
        const FusedGetControlSnapshot control = LoadFusedGetControl();
        const uint64_t controlError = ValidateControl(control, lane);
        if (controlError != 0U) {
            return controlError;
        }
        // The lifecycle contract freezes this control block for the whole
        // graph replay.  Materialize only the scalar addresses needed below
        // from GM into AIV-local registers; do not form a non-GM struct
        // reference to HBM memory.
        const uint64_t outputAddress = control.output;
        const uint64_t coreStatusAddress = control.coreStatus;
        const uint64_t channelEntityAddress = control.channelEntity;
        const uint64_t queueStateAddress = control.queueState;
        __gm__ int32_t* laneIndices = GetIndicesBuffer(lane, buffer);
        __gm__ int32_t* laneOffsets = GetOffsetsBuffer(lane, buffer);
        __gm__ int32_t* laneOutputIndices =
            GetOutputIndicesBuffer(lane, buffer);
        __gm__ KvafBatchGetCoreStatus* status =
            reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(
                coreStatusAddress +
                static_cast<uint64_t>(lane) * sizeof(KvafBatchGetCoreStatus));
        // Retain the current KVAF profile layout. It yields an exact CQ wait
        // cycle count without introducing a host-side profiler in the hot path.
        status->profile.magic = KVAF_AIV_BATCH_GET_PROFILE_MAGIC;
        FlushGmRange(reinterpret_cast<__gm__ uint8_t*>(status),
                     sizeof(KvafBatchGetCoreStatus));

        *args = KvafExecuteBatchGetKernelArgs{};
        args->address_table = reinterpret_cast<uint64_t>(addressTable_);
        args->valid_table = reinterpret_cast<uint64_t>(validTable_);
        args->indices = reinterpret_cast<uint64_t>(laneIndices);
        args->offsets = reinterpret_cast<uint64_t>(laneOffsets);
        // The dense helper consumes only [0, requestCount).  `getMask_` is
        // retained as a debug output of the producer/consumer bridge, but it
        // is deliberately not part of this Direct-HCCP submission path.
        args->mask = 0U;
        args->output_indices = reinterpret_cast<uint64_t>(laneOutputIndices);
        args->output = outputAddress;
        args->core_status = reinterpret_cast<uint64_t>(status);
        args->channel_entity = channelEntityAddress +
                               static_cast<uint64_t>(lane) *
                                   sizeof(KvafBatchGetDirectUrmaEntity);
        args->queue_state = queueStateAddress +
                            static_cast<uint64_t>(lane) *
                                sizeof(KvafBatchGetQueueState);
        args->table_capacity = tiling_->tableCapacity;
        args->request_capacity = kGetWaveRecords;
        args->output_rows = tiling_->outputRows;
        args->block_bytes = kPackedEntryBytes;
        args->block_count = 1U;
        args->jettys_per_core = 1U;
        return 0U;
    }

    // Validate exactly the HBM descriptor ledger that PostOnly will consume.
    // This is deliberately just before Post, after the bypass-store release
    // in ConsumeReadyRings.  It turns a stale/incorrect descriptor into a
    // precise P5 error before an HCCP write can target a wrong private slot.
    __aicore__ inline uint64_t ValidateFullGetDescriptorArgs(
        uint32_t lane, uint32_t requestCount,
        const KvafExecuteBatchGetKernelArgs& args) const
    {
        if (lane >= kGetLaneCount || requestCount == 0U ||
            requestCount > kGetWaveRecords || args.indices == 0U ||
            args.offsets == 0U || args.output_indices == 0U) {
            return kGetResultDescriptorStateError;
        }
        const uint32_t laneBase = lane * tiling_->privateArenaCapacity;
        __gm__ int32_t* const indices =
            reinterpret_cast<__gm__ int32_t*>(args.indices);
        __gm__ int32_t* const offsets =
            reinterpret_cast<__gm__ int32_t*>(args.offsets);
        __gm__ int32_t* const outputIndices =
            reinterpret_cast<__gm__ int32_t*>(args.output_indices);
        DataSyncBarrier<MemDsbT::DDR>();
        for (uint32_t request = 0U; request < requestCount; ++request) {
            const int32_t tableIndex =
                ReadGmByPassDCache<int32_t>(indices + request);
            const int32_t offset =
                ReadGmByPassDCache<int32_t>(offsets + request);
            const int32_t outputIndex =
                ReadGmByPassDCache<int32_t>(outputIndices + request);
            if (outputIndex < 0 ||
                static_cast<uint32_t>(outputIndex) < laneBase ||
                static_cast<uint32_t>(outputIndex) >=
                    laneBase + tiling_->privateArenaCapacity ||
                !IsValidDirectGetDescriptor(tableIndex, offset, outputIndex)) {
                return kGetResultDescriptorStateError;
            }
        }
        return 0U;
    }

    // A Post failure has no terminal CQ that can make a private payload
    // observable.  Undo every *new* object claim in the failed descriptor
    // wave.  Object-to-slot is the dedup authority, but more than one source
    // may already point at the same object; clear every matching source entry
    // before returning the arena cell to FREE.  This bounded-on-error scan is
    // deliberately off the normal successful path.
    __aicore__ inline void RollBackPrivateClaimsForWave(
        uint32_t lane, uint32_t buffer, uint32_t requestCount)
    {
        if (buffer >= kGetDescriptorBufferCount) {
            return;
        }
        __gm__ int32_t* const outputIndices =
            GetOutputIndicesBuffer(lane, buffer);
        __gm__ int32_t* const candidateSlots =
            GetCandidateSeedSlotsBuffer(lane, buffer);
        __gm__ uint8_t* const claimNew = GetClaimNewBuffer(lane, buffer);
        const uint64_t sourceMapCount =
            static_cast<uint64_t>(tiling_->rowCount) * tiling_->sourceCapacity;
        for (uint32_t request = 0U; request < requestCount; ++request) {
            if (ReadGmByPassDCache<uint8_t>(claimNew + request) == 0U) {
                continue;
            }
            const int32_t outputIndex =
                ReadGmByPassDCache<int32_t>(outputIndices + request);
            const int32_t candidateSlot =
                ReadGmByPassDCache<int32_t>(candidateSlots + request);
            if (outputIndex < 0 || static_cast<uint32_t>(outputIndex) >=
                                       tiling_->outputRows ||
                candidateSlot < 0 || static_cast<uint64_t>(candidateSlot) >=
                                         static_cast<uint64_t>(tiling_->producerCount) *
                                             tiling_->candidateCapacityPerProducer) {
                continue;
            }
            __gm__ int32_t* const seed = CandidateSeedByAbsolute(
                static_cast<uint64_t>(candidateSlot));
            const int32_t row =
                ReadGmByPassDCache<int32_t>(seed + kCandidateSeedRow);
            const int32_t source =
                ReadGmByPassDCache<int32_t>(seed + kCandidateSeedSource);
            const int32_t kvObject =
                ReadGmByPassDCache<int32_t>(seed + kCandidateSeedObject);
            if (row >= 0 && source >= 0 && kvObject >= 0 &&
                static_cast<uint32_t>(row) < tiling_->rowCount &&
                static_cast<uint32_t>(source) < tiling_->sourceCapacity &&
                static_cast<uint32_t>(kvObject) < tiling_->kvObjectCapacity) {
                const int32_t pool = ReadGmByPassDCache<int32_t>(
                    reqPoolEntries_ + static_cast<uint32_t>(row));
                if (pool >= 0 && static_cast<uint32_t>(pool) < tiling_->poolSize) {
                    const uint64_t objectOffset =
                        static_cast<uint64_t>(pool) * tiling_->kvObjectCapacity +
                        static_cast<uint32_t>(kvObject);
                    if (ReadGmByPassDCache<int32_t>(
                            privateObjectToSlot_ + objectOffset) == outputIndex) {
                        WriteGmByPassDCache<int32_t>(
                            privateObjectToSlot_ + objectOffset,
                            kPrivateSlotUnmapped);
                    }
                }
            }
            for (uint64_t sourceOffset = 0U; sourceOffset < sourceMapCount;
                 ++sourceOffset) {
                if (ReadGmByPassDCache<int32_t>(
                        privateSourceToSlot_ + sourceOffset) == outputIndex) {
                    WriteGmByPassDCache<int32_t>(
                        privateSourceToSlot_ + sourceOffset,
                        kPrivateSlotUnmapped);
                }
            }
            WriteGmByPassDCache<int32_t>(privateReady_ + outputIndex,
                                         kPrivateReadyFree);
            WriteGmByPassDCache<int32_t>(seed + kCandidateSeedPrivateSlot,
                                         kCandidateSeedDeferred);
            WriteGmByPassDCache<uint8_t>(claimNew + request, 0U);
        }
        DataSyncBarrier<MemDsbT::DDR>();
    }

    // `FinishKvafDenseBatchGet` is the terminal CQ boundary.  Direct-HCCP has
    // completed the payload write before this routine is called, so private
    // ready transitions are published only after Finish and its status flush.
    //
    // `outputIndicesAddress` is deliberately taken from the immutable args
    // object that Begin used to materialize the WQEs.  Reconstructing the
    // active double-buffer from the next fill-buffer is not part of the
    // Direct-HCCP transaction contract: a stale reconstruction can mark (or
    // reject) a different descriptor ledger after a perfectly valid CQ.
    // Tying READY publication to the exact post pointer keeps the H2D
    // destination and its readiness state inseparable.
    __aicore__ inline uint64_t PublishPrivateReadyForWave(
        uint32_t lane, uint32_t requestCount, uint64_t outputIndicesAddress,
        bool terminalSuccess)
    {
        if (outputIndicesAddress == 0U) {
            return kGetResultPrivateStateError;
        }
        __gm__ int32_t* const outputIndices =
            reinterpret_cast<__gm__ int32_t*>(outputIndicesAddress);
        const uint32_t laneBase = lane * tiling_->privateArenaCapacity;
        for (uint32_t request = 0U; request < requestCount; ++request) {
            const int32_t outputIndex =
                ReadGmByPassDCache<int32_t>(outputIndices + request);
            if (outputIndex < 0 || static_cast<uint32_t>(outputIndex) < laneBase ||
                static_cast<uint32_t>(outputIndex) >=
                    laneBase + tiling_->privateArenaCapacity ||
                static_cast<uint32_t>(outputIndex) >= tiling_->outputRows) {
                return kGetResultPrivateStateError;
            }
            WriteGmByPassDCache<int32_t>(privateReady_ + outputIndex,
                                         terminalSuccess ? kPrivateReadyReady
                                                         : kPrivateReadyFailed);
        }
        DataSyncBarrier<MemDsbT::DDR>();
        return 0U;
    }

    // The strict serial reference retains the historical blocking helper.
    // It now explicitly names descriptor buffer zero only because the P2
    // parallel path owns a second buffer; no serial execution ordering changes.
    __aicore__ inline uint64_t IssueFullGet(
        uint32_t lane, uint32_t requestCount, uint64_t* firstDoorbell,
        uint64_t* completionCycle)
    {
        if (requestCount == 0U) {
            return 0U;
        }
        KvafExecuteBatchGetKernelArgs args{};
        const uint64_t buildError = BuildFullGetArgs(lane, 0U, &args);
        if (buildError != 0U) {
            return buildError;
        }
        const uint64_t descriptorError =
            ValidateFullGetDescriptorArgs(lane, requestCount, args);
        if (descriptorError != 0U) {
            return descriptorError;
        }

        kvaf_fused_get::RunKvafDenseBatchGetCore256(
            args, requestCount, 0U);
        __gm__ KvafBatchGetCoreStatus* status =
            reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args.core_status);
        FlushGmRange(reinterpret_cast<__gm__ uint8_t*>(status),
                     sizeof(KvafBatchGetCoreStatus));
        if (*firstDoorbell == 0U) {
            *firstDoorbell = status->profile.first_doorbell_cycle;
        }
        *completionCycle = status->profile.completion_done_cycle;
        const uint64_t readyError = PublishPrivateReadyForWave(
            lane, requestCount, args.output_indices, status->result == 0U);
        return status->result != 0U ? status->result : readyError;
    }

    // Post one wave, retain its descriptor buffer, then let the caller run
    // useful descriptor-compaction work until it observes a terminal CQE.
    // The transaction remains private to this lane, so the pre-existing
    // one-writer/one-in-flight queue contract is unchanged.
    __aicore__ inline uint64_t BeginFullGet(
        uint32_t lane, uint32_t buffer, uint32_t requestCount,
        KvafExecuteBatchGetKernelArgs* args,
        kvaf_fused_get::KvafDenseBatchGetTransaction* transaction,
        uint64_t* firstDoorbell)
    {
        if (requestCount == 0U || args == nullptr || transaction == nullptr) {
            return requestCount == 0U ? 0U : kGetResultProtocolError;
        }
        const uint64_t buildError = BuildFullGetArgs(lane, buffer, args);
        if (buildError != 0U) {
            // Build failure occurs before a Direct-HCCP post.  Finalize the
            // caller-local transaction explicitly so the error path can
            // safely publish FAILED/rollback state without dereferencing an
            // uninitialized queue/status address in Finish.
            kvaf_fused_get::ResetDenseBatchGetTransaction(transaction);
            transaction->result = buildError;
            transaction->state = kvaf_fused_get::kDenseBatchGetFailed;
            transaction->finalized = 1U;
            return buildError;
        }
        const uint64_t descriptorError =
            ValidateFullGetDescriptorArgs(lane, requestCount, *args);
        if (descriptorError != 0U) {
            // This is still before PostOnly: preserve the same terminal
            // contract as a Build failure, but keep the malformed descriptor
            // buffers visible to the bounded P5 test ledger.
            kvaf_fused_get::ResetDenseBatchGetTransaction(transaction);
            transaction->result = descriptorError;
            transaction->state = kvaf_fused_get::kDenseBatchGetFailed;
            transaction->finalized = 1U;
            return descriptorError;
        }
        kvaf_fused_get::BeginKvafDenseBatchGetPostOnly(
            *args, requestCount, 0U, transaction);
        if (*firstDoorbell == 0U && transaction->first_doorbell_cycle != 0U) {
            *firstDoorbell = transaction->first_doorbell_cycle;
        }
        return transaction->result;
    }

    __aicore__ inline uint64_t FinishFullGet(
        uint32_t lane, uint32_t requestCount,
        const KvafExecuteBatchGetKernelArgs& args,
        kvaf_fused_get::KvafDenseBatchGetTransaction* transaction,
        uint64_t* completionCycle)
    {
        if (transaction == nullptr) {
            return kGetResultProtocolError;
        }
        kvaf_fused_get::FinishKvafDenseBatchGet(args, transaction);
        __gm__ KvafBatchGetCoreStatus* status =
            reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args.core_status);
        if (status != nullptr) {
            FlushGmRange(reinterpret_cast<__gm__ uint8_t*>(status),
                         sizeof(KvafBatchGetCoreStatus));
        }
        if (transaction->profile_completion_done_cycle != 0U) {
            *completionCycle = transaction->profile_completion_done_cycle;
        } else if (status != nullptr) {
            *completionCycle = status->profile.completion_done_cycle;
        }
        const uint64_t readyError = PublishPrivateReadyForWave(
            lane, requestCount, args.output_indices,
            transaction->result == 0U);
        return transaction->result != 0U ? transaction->result : readyError;
    }

    __aicore__ inline void WriteGetStats(
        uint32_t lane, uint64_t requested, uint64_t batches, uint64_t error,
        uint64_t firstDoorbell, uint64_t completionCycle, uint64_t idlePolls,
        uint64_t consumedChunks, uint64_t pipelineMode,
        uint64_t serialGateOpenCycle, uint64_t firstConsumedChunkCycle,
        uint64_t serialGatePolls, uint64_t descriptorHashSum,
        uint64_t descriptorHashXor, uint64_t nonBlockingRetirePolls,
        uint64_t overlappedDescriptorRecords)
    {
        __gm__ int64_t* stats = getStats_ +
                                static_cast<uint64_t>(lane) *
                                    kGetStatColumns;
        WriteGmByPassDCache<int64_t>(stats + 0U,
                                     static_cast<int64_t>(requested));
        WriteGmByPassDCache<int64_t>(stats + 1U,
                                     static_cast<int64_t>(batches));
        WriteGmByPassDCache<int64_t>(stats + 2U,
                                     static_cast<int64_t>(error));
        WriteGmByPassDCache<int64_t>(stats + 3U,
                                     static_cast<int64_t>(firstDoorbell));
        WriteGmByPassDCache<int64_t>(stats + 4U,
                                     static_cast<int64_t>(completionCycle));
        WriteGmByPassDCache<int64_t>(stats + 5U,
                                     static_cast<int64_t>(idlePolls));
        WriteGmByPassDCache<int64_t>(stats + 6U,
                                     static_cast<int64_t>(consumedChunks));
        WriteGmByPassDCache<int64_t>(stats + 7U,
                                     static_cast<int64_t>(GetSystemCycle()));
        WriteGmByPassDCache<int64_t>(stats + 8U,
                                     static_cast<int64_t>(pipelineMode));
        WriteGmByPassDCache<int64_t>(stats + 9U,
                                     static_cast<int64_t>(serialGateOpenCycle));
        WriteGmByPassDCache<int64_t>(stats + 10U,
                                     static_cast<int64_t>(firstConsumedChunkCycle));
        WriteGmByPassDCache<int64_t>(stats + 11U,
                                     static_cast<int64_t>(serialGatePolls));
        WriteGmByPassDCache<int64_t>(stats + 12U,
                                     static_cast<int64_t>(descriptorHashSum));
        WriteGmByPassDCache<int64_t>(stats + 13U,
                                     static_cast<int64_t>(descriptorHashXor));
        // P2-only diagnostics.  Word 14 counts genuine nonblocking CQ
        // probes; word 15 counts descriptor records compacted into the
        // alternate HBM buffer while the preceding H2D wave was in flight.
        WriteGmByPassDCache<int64_t>(stats + 14U,
                                     static_cast<int64_t>(nonBlockingRetirePolls));
        WriteGmByPassDCache<int64_t>(stats + 15U,
                                     static_cast<int64_t>(overlappedDescriptorRecords));
        DataSyncBarrier<MemDsbT::DDR>();
    }

    // P2's parallel path owns exactly one Direct-HCCP transaction per lane.
    // `fillBuffer` is never the buffer referenced by `activeArgs`: while one
    // DMA uses buffer A, this lane compacts the next *already selected* miss
    // descriptors into buffer B.  The value-level candidate stream remains
    // owned by ProcessExactScanProducer and is unaffected by this overlap.
    __aicore__ inline void ProcessParallelFullGetLane(
        uint32_t lane, uint64_t epoch)
    {
        uint64_t requested = 0U;
        uint64_t batches = 0U;
        uint64_t error = epoch == 0U ? kGetResultProtocolError : 0U;
        uint64_t firstDoorbell = 0U;
        uint64_t completionCycle = 0U;
        uint64_t idlePolls = 0U;
        uint64_t consumedChunks = 0U;
        uint64_t firstConsumedChunkCycle = 0U;
        uint64_t descriptorHashSum = 0U;
        uint64_t descriptorHashXor = 0U;
        uint64_t nonBlockingRetirePolls = 0U;
        uint64_t overlappedDescriptorRecords = 0U;
        uint32_t fillBuffer = 0U;
        // This is only the owner-side lifetime record for rollback/debugging.
        // READY publication itself uses activeArgs.output_indices, which is
        // the same pointer the posted WQEs consumed.
        uint32_t activeBuffer = 0U;
        uint32_t queued = 0U;
        uint32_t activeRequestCount = 0U;
        bool inFlight = false;
        KvafExecuteBatchGetKernelArgs activeArgs{};
        kvaf_fused_get::KvafDenseBatchGetTransaction transaction{};

        while (error == 0U) {
            if (inFlight) {
                // A probe has no side effect while the CQ is not ready.  It
                // may therefore be issued around descriptor work without
                // changing this queue's WQE/CQE order.
                kvaf_fused_get::TryRetireKvafDenseBatchGet(
                    activeArgs, &transaction);
                ++nonBlockingRetirePolls;
                if (transaction.state != kvaf_fused_get::kDenseBatchGetInFlight) {
                    const uint64_t result = FinishFullGet(
                        lane, activeRequestCount, activeArgs,
                        &transaction, &completionCycle);
                    inFlight = false;
                    activeRequestCount = 0U;
                    if (result != 0U) {
                        error = result;
                        break;
                    }
                }
            }

            // The fixed [64, 256, ..., tail] sequence is retained exactly.
            // `batches` counts waves already posted, including an in-flight
            // wave, so the alternate buffer is always sized for the next
            // canonical wave.
            const uint32_t target = batches == 0U ? kGetBootstrapRecords
                                                   : kGetWaveRecords;
            // Observe terminal closure before consuming.  If the final
            // producer close becomes visible on this iteration,
            // AllRingsClosed performs the acquire required for the
            // immediately following tail drain.  If it is not yet closed,
            // ConsumeReadyRings retains the normal eager streaming behavior.
            const bool allClosed = AllRingsClosed(lane, epoch);
            const uint32_t before = queued;
            if (queued < target) {
                queued = ConsumeReadyRings(
                    lane, fillBuffer, epoch, queued, target, &consumedChunks,
                    &firstConsumedChunkCycle, &descriptorHashSum,
                    &descriptorHashXor, &error);
            }
            // Keep this accounting after the final acquire/drain as well:
            // descriptors compacted into the alternate buffer while the
            // preceding Direct-HCCP wave is in flight are genuine overlap.
            if (inFlight && queued > before) {
                overlappedDescriptorRecords += queued - before;
            }
            if (error != 0U) {
                break;
            }

            if (!inFlight &&
                (queued >= target ||
                 (allClosed && queued != 0U))) {
                const uint32_t requestCount = queued;
                const uint64_t result = BeginFullGet(
                    lane, fillBuffer, requestCount, &activeArgs, &transaction,
                    &firstDoorbell);
                requested += requestCount;
                ++batches;
                queued = 0U;
                activeBuffer = fillBuffer;
                fillBuffer = 1U - fillBuffer;
                if (result != 0U ||
                    transaction.state != kvaf_fused_get::kDenseBatchGetInFlight) {
                    const uint64_t finalResult = FinishFullGet(
                        lane, requestCount, activeArgs,
                        &transaction, &completionCycle);
                    if (result != 0U) {
                        RollBackPrivateClaimsForWave(lane, activeBuffer,
                                                     requestCount);
                    }
                    error = result != 0U ? result
                                          : (finalResult != 0U
                                                 ? finalResult
                                                 : kGetResultProtocolError);
                    break;
                }
                activeRequestCount = requestCount;
                inFlight = true;
                idlePolls = 0U;
                continue;
            }

            if (!inFlight && allClosed && queued == 0U) {
                break;
            }
            if (inFlight) {
                // CQ timeout is governed by the transaction's own exact
                // poll budget.  Do not label a live H2D wave as a consumer
                // idle timeout merely because the alternate buffer is full.
                idlePolls = 0U;
                continue;
            }
            if (queued == before) {
                ++idlePolls;
                if (idlePolls >= kConsumerIdlePollLimit) {
                    error = kGetResultConsumerTimeout;
                    break;
                }
            } else {
                idlePolls = 0U;
            }
        }

        // No caller may leave a posted wave unretired.  If a descriptor or
        // ring error happened after a successful post, drain that wave before
        // publishing the terminal P2 telemetry.
        if (inFlight) {
            const uint64_t result = FinishFullGet(
                lane, activeRequestCount, activeArgs,
                &transaction, &completionCycle);
            if (error == 0U && result != 0U) {
                error = result;
            }
        }
        WriteGetStats(lane, requested, batches, error, firstDoorbell,
                      completionCycle, idlePolls, consumedChunks,
                      kPipelineModeParallel, 0U, firstConsumedChunkCycle,
                      0U, descriptorHashSum, descriptorHashXor,
                      nonBlockingRetirePolls, overlappedDescriptorRecords);
    }

    __aicore__ inline void ProcessFullGetLane(uint32_t lane)
    {
        const uint64_t epoch = LoadColdEpoch(streamingEpoch_);
        const uint64_t pipelineMode = LoadColdEpoch(executionMode_);
        WriteRoleTrace(lane, kRoleTraceGetBase + static_cast<int32_t>(lane));
        uint64_t requested = 0U;
        uint64_t batches = 0U;
        uint64_t error = epoch == 0U ? kGetResultProtocolError : 0U;
        if (pipelineMode != kPipelineModeParallel &&
            pipelineMode != kPipelineModeSerialWaitAllClosed) {
            error = kGetResultInvalidPipelineMode;
        }
        if (error == 0U && pipelineMode == kPipelineModeParallel) {
            ProcessParallelFullGetLane(lane, epoch);
            return;
        }
        uint64_t firstDoorbell = 0U;
        uint64_t completionCycle = 0U;
        uint64_t idlePolls = 0U;
        uint64_t consumedChunks = 0U;
        uint64_t serialGateOpenCycle = 0U;
        uint64_t firstConsumedChunkCycle = 0U;
        uint64_t serialGatePolls = 0U;
        uint64_t descriptorHashSum = 0U;
        uint64_t descriptorHashXor = 0U;
        uint32_t queued = 0U;

        // `serial_wait_all_closed` is the strict reference, not a delayed
        // input fixture.  It shares the identical 64-AIV launch, score and
        // resident tensors, two Direct-HCCP queues, ring layout, and Get
        // segmentation with streaming.  The only changed causal edge is that
        // no Get descriptor can be consumed before all 124 producer rings
        // have been terminally closed for this epoch.
        if (error == 0U && pipelineMode == kPipelineModeSerialWaitAllClosed) {
            while (!AllProducerRingsClosed(epoch)) {
                ++serialGatePolls;
                if (serialGatePolls >= kSerialGatePollLimit) {
                    error = kGetResultSerialGateTimeout;
                    break;
                }
            }
            if (error == 0U) {
                serialGateOpenCycle = static_cast<uint64_t>(GetSystemCycle());
            }
        }

        while (error == 0U) {
            // This limit is intentionally selected before consuming.  It
            // fixes every replay to [64, 256, ..., tail] per Get lane, so the
            // serial reference cannot accidentally fuse a different WQE
            // shape merely because all rings are already ready.
            const uint32_t target = batches == 0U ? kGetBootstrapRecords
                                                   : kGetWaveRecords;
            const uint32_t before = queued;
            queued = ConsumeReadyRings(
                lane, 0U, epoch, queued, target, &consumedChunks,
                &firstConsumedChunkCycle, &descriptorHashSum,
                &descriptorHashXor, &error);
            const bool allClosed = AllRingsClosed(lane, epoch);
            if (queued >= target ||
                (allClosed && queued != 0U)) {
                const uint64_t result = IssueFullGet(
                    lane, queued, &firstDoorbell, &completionCycle);
                requested += queued;
                ++batches;
                queued = 0U;
                if (result != 0U) {
                    error = result;
                    break;
                }
                idlePolls = 0U;
                continue;
            }
            if (allClosed && queued == 0U) {
                break;
            }
            if (queued == before) {
                ++idlePolls;
                if (idlePolls >= kConsumerIdlePollLimit) {
                    error = kGetResultConsumerTimeout;
                    break;
                }
            } else {
                idlePolls = 0U;
            }
        }

        WriteGetStats(lane, requested, batches, error, firstDoorbell,
                      completionCycle, idlePolls, consumedChunks,
                      pipelineMode, serialGateOpenCycle,
                      firstConsumedChunkCycle, serialGatePolls,
                      descriptorHashSum, descriptorHashXor, 0U, 0U);
    }

private:
    const A5ExactScanSpeculativeGetTilingData* tiling_;
    __gm__ uint16_t* scoreWorkspaceU16_{nullptr};
    __gm__ int64_t* tileReceipts_{nullptr};
    __gm__ int64_t* streamingEpoch_{nullptr};
    __gm__ int32_t* scoreSourceLens_{nullptr};
    __gm__ int32_t* residentThreshold_{nullptr};
    __gm__ int32_t* residentThresholdValid_{nullptr};
    __gm__ int32_t* kvObjectSlotsSnapshot_{nullptr};
    __gm__ int32_t* reqPoolEntries_{nullptr};
    __gm__ int32_t* kvObjectMap_{nullptr};
    __gm__ int32_t* kvObjectLens_{nullptr};
    __gm__ int32_t* privateSourceToSlot_{nullptr};
    __gm__ int32_t* privateObjectToSlot_{nullptr};
    __gm__ int32_t* privateReady_{nullptr};
    __gm__ int32_t* privateArenaHead_{nullptr};
    __gm__ uint64_t* addressTable_{nullptr};
    __gm__ uint8_t* validTable_{nullptr};
    __gm__ uint8_t* privatePackedOutput_{nullptr};
    __gm__ KvafFusedBatchGetControl* fusedGetControl_{nullptr};
    __gm__ int32_t* candidateSeeds_{nullptr};
    __gm__ int32_t* candidateCounts_{nullptr};
    __gm__ int64_t* producerStats_{nullptr};
    __gm__ int64_t* getStats_{nullptr};
    __gm__ int32_t* roleTrace_{nullptr};
    __gm__ int32_t* getIndices_{nullptr};
    __gm__ int32_t* getOffsets_{nullptr};
    __gm__ int32_t* getOutputIndices_{nullptr};
    __gm__ uint8_t* getMask_{nullptr};
    __gm__ int32_t* getCandidateSeedSlots_{nullptr};
    __gm__ uint8_t* getClaimNew_{nullptr};
    __gm__ int64_t* pipelineState_{nullptr};
    __gm__ int64_t* executionMode_{nullptr};
};
}  // namespace

extern "C" __global__ __aicore__ void a5_exact_scan_speculative_get(
    GM_ADDR scoreWorkspace, GM_ADDR tileReceipts, GM_ADDR streamingEpoch,
    GM_ADDR scoreSourceLens, GM_ADDR residentThreshold,
    GM_ADDR residentThresholdValid,
    GM_ADDR kvObjectSlotsSnapshot, GM_ADDR reqPoolEntries,
    GM_ADDR kvObjectMap, GM_ADDR kvObjectLens,
    GM_ADDR privateSourceToSlot, GM_ADDR privateObjectToSlot,
    GM_ADDR privateReady, GM_ADDR privateArenaHead,
    GM_ADDR privatePackedOutput, GM_ADDR addressTable, GM_ADDR validTable,
    GM_ADDR fusedGetControl, GM_ADDR pipelineState, GM_ADDR executionMode,
    GM_ADDR privateSourceToSlotOut, GM_ADDR privateObjectToSlotOut,
    GM_ADDR privateReadyOut, GM_ADDR privateArenaHeadOut,
    GM_ADDR privatePackedOutputOut, GM_ADDR candidateSeeds,
    GM_ADDR candidateCounts, GM_ADDR producerStats, GM_ADDR getStats,
    GM_ADDR roleTrace, GM_ADDR getIndices, GM_ADDR getOffsets,
    GM_ADDR getOutputIndices, GM_ADDR getMask,
    GM_ADDR getCandidateSeedSlots, GM_ADDR getClaimNew,
    GM_ADDR pipelineStateOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    // The bridge aliases these transaction-local mutable inputs.  Keep named
    // output parameters so graph dataflow records the private-state publish;
    // no persistent KV resident mapping is among these aliases.
    (void)privateSourceToSlotOut;
    (void)privateObjectToSlotOut;
    (void)privateReadyOut;
    (void)privateArenaHeadOut;
    (void)privatePackedOutputOut;
    (void)pipelineStateOut;
    // ExactScan and the full HCCP Get path are both AIV work.  A MIX task
    // created 32 idle AIC lanes and moved the two KVAF workers away from the
    // standalone AIV-only task positions they were created for.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5ExactScanSpeculativeGetTilingData);
    GET_TILING_DATA(tilingData, tiling);
    A5ExactScanSpeculativeGetKernel op(&tilingData);
    op.Init(scoreWorkspace, tileReceipts, streamingEpoch, scoreSourceLens,
            residentThreshold, residentThresholdValid, kvObjectSlotsSnapshot,
            reqPoolEntries, kvObjectMap, kvObjectLens,
            privateSourceToSlot, privateObjectToSlot, privateReady,
            privateArenaHead, privatePackedOutput, addressTable, validTable,
            fusedGetControl, pipelineState, executionMode, candidateSeeds,
            candidateCounts, producerStats, getStats, roleTrace,
            getIndices, getOffsets, getOutputIndices, getMask,
            getCandidateSeedSlots, getClaimNew);
    op.Process();
}
