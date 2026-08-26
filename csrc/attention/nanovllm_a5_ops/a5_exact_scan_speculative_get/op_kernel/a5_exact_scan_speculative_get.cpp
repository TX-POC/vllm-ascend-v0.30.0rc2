/**
 * Device-resident P1 ExactScan + full SpeculativeGet.
 *
 * Inputs `li_scores` are deliberately the upstream Lightning score seam, not
 * a replacement implementation of official C8 LI.  Task lanes 0 and 1 own
 * the two Direct-HCCP queues.  Task lanes 2 through 63 perform the exact
 * threshold/resident decision and publish misses to two fixed SPSC rings.
 *
 *   compacted miss descriptors -> WQE/SQ -> doorbell -> CQ terminal wait.
 *
 * Therefore a Get AIV stays occupied while its H2D DMA is in flight.  The LI
 * producers do not wait for it: a full ring marks overflow and leaves final
 * correctness to later Guaranteed Recall rather than deadlocking the scan.
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
constexpr uint32_t kRingSlotRecordWord = 2U * kCacheLineWords;
constexpr uint32_t kRingWordsPerSlot =
    kRingSlotRecordWord + kRecordsPerRingSlot * kRecordWords;
constexpr uint32_t kRingClosedWord = kRingSlots * kRingWordsPerSlot;
constexpr uint32_t kRingWords = kRingClosedWord + kCacheLineWords;
constexpr uint32_t kGetBootstrapRecords = 64;
constexpr uint32_t kGetWaveRecords = 256;
constexpr uint32_t kProducerStatColumns = 7;
constexpr uint32_t kGetStatColumns = 8;
constexpr uint32_t kPackedEntryBytes = 656;
constexpr uint32_t kConsumerIdlePollLimit = 16U * 1024U * 1024U;

static_assert(kRingWords % kCacheLineWords == 0U,
              "each SPSC ring must start on a cache-line boundary");

constexpr int32_t kRoleTraceExactScanBase = 1000;
constexpr int32_t kRoleTraceGetBase = 2000;
constexpr uint64_t kGetResultProtocolError = 0x1001U;
constexpr uint64_t kGetResultConsumerTimeout = 0x1002U;
constexpr uint64_t kGetResultRingRecordError = 0x1003U;

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

// The compact candidate stream is consumed later by ExactTopK/Reconcile.
// A deferred miss is still part of that stream, but is deliberately not
// considered materialized by this speculative prefetch pass.  The later
// Guaranteed Recall owns it if it enters the final exact TopK.
constexpr uint8_t kCandidateHbmHit = 0U;
constexpr uint8_t kCandidateDdrEnqueued = 1U;
constexpr uint8_t kCandidateDdrDeferred = 2U;

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
        GM_ADDR liScores, GM_ADDR threshold, GM_ADDR resident,
        GM_ADDR tableIndices, GM_ADDR offsets, GM_ADDR outputIndices,
        GM_ADDR addressTable, GM_ADDR validTable, GM_ADDR packedOutput,
        GM_ADDR fusedGetControl, GM_ADDR frameEpoch,
        GM_ADDR pipelineState,
        GM_ADDR candidateIndices, GM_ADDR candidateKind,
        GM_ADDR producerStats, GM_ADDR getStats, GM_ADDR roleTrace,
        GM_ADDR getIndices, GM_ADDR getOffsets, GM_ADDR getOutputIndices,
        GM_ADDR getMask)
    {
        liScores_ = reinterpret_cast<__gm__ float*>(liScores);
        threshold_ = reinterpret_cast<__gm__ float*>(threshold);
        resident_ = reinterpret_cast<__gm__ uint8_t*>(resident);
        tableIndices_ = reinterpret_cast<__gm__ int32_t*>(tableIndices);
        offsets_ = reinterpret_cast<__gm__ int32_t*>(offsets);
        outputIndices_ = reinterpret_cast<__gm__ int32_t*>(outputIndices);
        addressTable_ = reinterpret_cast<__gm__ uint64_t*>(addressTable);
        validTable_ = reinterpret_cast<__gm__ uint8_t*>(validTable);
        packedOutput_ = reinterpret_cast<__gm__ uint8_t*>(packedOutput);
        fusedGetControl_ =
            reinterpret_cast<__gm__ KvafFusedBatchGetControl*>(fusedGetControl);
        frameEpoch_ = reinterpret_cast<__gm__ int64_t*>(frameEpoch);
        pipelineState_ = reinterpret_cast<__gm__ int64_t*>(pipelineState);
        candidateIndices_ = reinterpret_cast<__gm__ int32_t*>(candidateIndices);
        candidateKind_ = reinterpret_cast<__gm__ uint8_t*>(candidateKind);
        producerStats_ = reinterpret_cast<__gm__ int64_t*>(producerStats);
        getStats_ = reinterpret_cast<__gm__ int64_t*>(getStats);
        roleTrace_ = reinterpret_cast<__gm__ int32_t*>(roleTrace);
        getIndices_ = reinterpret_cast<__gm__ int32_t*>(getIndices);
        getOffsets_ = reinterpret_cast<__gm__ int32_t*>(getOffsets);
        getOutputIndices_ = reinterpret_cast<__gm__ int32_t*>(getOutputIndices);
        getMask_ = reinterpret_cast<__gm__ uint8_t*>(getMask);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const uint32_t taskLane = GetBlockIdx();
            if (taskLane < tiling_->getLaneCount) {
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
    __aicore__ inline RingView Ring(uint32_t producer, uint32_t lane) const
    {
        const uint32_t ringIndex = producer * kGetLaneCount + lane;
        return RingView{pipelineState_ +
                        static_cast<uint64_t>(ringIndex) *
                            tiling_->pipelineWordsPerRing};
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
        DataSyncBarrier<MemDsbT::DDR>();
        StoreRingWord(ring.SlotCount(slot), count);
        DataSyncBarrier<MemDsbT::DDR>();
        StoreRingWord(ring.SlotState(slot), epoch);
    }

    __aicore__ inline bool TryAppendMiss(
        const RingView& ring, uint32_t* writeSlot, uint32_t* writeCount,
        uint64_t epoch, int32_t tableIndex, int32_t offset,
        int32_t outputIndex, uint64_t candidateSlot,
        uint64_t* pendingCandidateSlots, uint64_t* firstPublishCycle)
    {
        if (*writeCount == 0U &&
            LoadRingWord(ring.SlotState(*writeSlot)) == epoch) {
            return false;
        }
        __gm__ int64_t* const tableWord =
            ring.RecordWord(*writeSlot, *writeCount, 0);
        WriteGmByPassDCache<int64_t>(tableWord,
                                     static_cast<int64_t>(tableIndex));
        WriteGmByPassDCache<int64_t>(tableWord + 1,
                                     static_cast<int64_t>(offset));
        WriteGmByPassDCache<int64_t>(tableWord + 2,
                                     static_cast<int64_t>(outputIndex));
        pendingCandidateSlots[*writeCount] = candidateSlot;
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
        uint64_t epoch, uint64_t* pendingCandidateSlots,
        uint64_t* firstPublishCycle, uint64_t* overflow,
        uint64_t* deferred)
    {
        if (writeCount != 0U) {
            if (LoadRingWord(ring.SlotState(writeSlot)) == epoch) {
                *overflow = 1U;
                for (uint32_t record = 0U; record < writeCount; ++record) {
                    WriteGmByPassDCache<uint8_t>(
                        candidateKind_ + pendingCandidateSlots[record],
                        kCandidateDdrDeferred);
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

    __aicore__ inline void ProcessExactScanProducer(uint32_t producer)
    {
        const uint64_t epoch = LoadColdEpoch(frameEpoch_);
        // Keep roleTrace in task-lane order so the P1 test can prove the two
        // Direct-HCCP workers occupy task lanes 0/1 rather than merely prove
        // that two workers existed somewhere in the launch.
        WriteRoleTrace(kProducerAivBegin + producer,
                       kRoleTraceExactScanBase + static_cast<int32_t>(producer));
        if (epoch == 0U) {
            WriteProducerStats(producer, 0U, 0U, 0U, 0U,
                               static_cast<uint64_t>(-1), 0U,
                               static_cast<uint64_t>(GetSystemCycle()));
            return;
        }

        const float threshold = threshold_[0];
        uint32_t writeSlot[kGetLaneCount] = {0U, 0U};
        uint32_t writeCount[kGetLaneCount] = {0U, 0U};
        uint64_t laneOverflow[kGetLaneCount] = {0U, 0U};
        uint64_t firstPublishCycle[kGetLaneCount] = {0U, 0U};
        uint64_t pendingCandidateSlots[kGetLaneCount][kRecordsPerRingSlot] = {};
        uint64_t selected = 0U;
        uint64_t hit = 0U;
        uint64_t miss = 0U;
        uint64_t enqueued = 0U;
        uint64_t deferred = 0U;

        for (uint64_t source = producer; source < tiling_->scoreCount;
             source += tiling_->producerCount) {
            if (liScores_[source] < threshold) {
                continue;
            }
            const uint64_t candidateSlot = selected * tiling_->producerCount +
                                           producer;
            WriteGmByPassDCache<int32_t>(
                candidateIndices_ + candidateSlot, static_cast<int32_t>(source));
            const bool isResident = resident_[source] != 0U;
            ++selected;
            if (isResident) {
                WriteGmByPassDCache<uint8_t>(candidateKind_ + candidateSlot,
                                              kCandidateHbmHit);
                ++hit;
                continue;
            }

            ++miss;
            const int32_t tableIndex = tableIndices_[source];
            const int32_t offset = offsets_[source];
            const int32_t outputIndex = outputIndices_[source];
            if (!IsValidDirectGetDescriptor(tableIndex, offset, outputIndex)) {
                WriteGmByPassDCache<uint8_t>(candidateKind_ + candidateSlot,
                                              kCandidateDdrDeferred);
                ++deferred;
                continue;
            }
            // This fixed route makes every producer/lane queue SPSC.  The
            // production C8 integration replaces source>>7 by row^block.
            const uint32_t lane = static_cast<uint32_t>((source >> 7U) & 1U);
            if (laneOverflow[lane] != 0U) {
                WriteGmByPassDCache<uint8_t>(candidateKind_ + candidateSlot,
                                              kCandidateDdrDeferred);
                ++deferred;
                continue;
            }
            const RingView ring = Ring(producer, lane);
            if (!TryAppendMiss(ring, &writeSlot[lane], &writeCount[lane], epoch,
                               tableIndex, offset, outputIndex,
                               candidateSlot, pendingCandidateSlots[lane],
                               &firstPublishCycle[lane])) {
                laneOverflow[lane] = 1U;
                WriteGmByPassDCache<uint8_t>(candidateKind_ + candidateSlot,
                                              kCandidateDdrDeferred);
                ++deferred;
            } else {
                WriteGmByPassDCache<uint8_t>(candidateKind_ + candidateSlot,
                                              kCandidateDdrEnqueued);
                ++enqueued;
            }
        }

        for (uint32_t lane = 0U; lane < kGetLaneCount; ++lane) {
            FlushProducerTail(Ring(producer, lane), writeSlot[lane],
                              writeCount[lane], epoch,
                              pendingCandidateSlots[lane],
                              &firstPublishCycle[lane], &laneOverflow[lane],
                              &deferred);
        }
        const uint64_t firstPublish = firstPublishCycle[0] != 0U
                                          ? firstPublishCycle[0]
                                          : firstPublishCycle[1];
        WriteProducerStats(producer, selected, hit, miss, enqueued, deferred,
                           firstPublish,
                           static_cast<uint64_t>(GetSystemCycle()));
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
        uint64_t scanDone)
    {
        __gm__ int64_t* stats = producerStats_ +
                                static_cast<uint64_t>(producer) *
                                    kProducerStatColumns;
        // Adjacent 7-word rows share 64-B lines.  Bypassing DCache makes each
        // scalar store independent, so a terminal write from one producer
        // cannot evict or overwrite a neighbouring producer's result.
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
        return true;
    }

    __aicore__ inline uint32_t ConsumeReadyRings(
        uint32_t lane, uint64_t epoch, uint32_t queued,
        uint64_t* consumedChunks, uint64_t* protocolError,
        bool* deferredReady)
    {
        __gm__ int32_t* laneIndices =
            getIndices_ + static_cast<uint64_t>(lane) * kGetWaveRecords;
        __gm__ int32_t* laneOffsets =
            getOffsets_ + static_cast<uint64_t>(lane) * kGetWaveRecords;
        __gm__ int32_t* laneOutputIndices =
            getOutputIndices_ + static_cast<uint64_t>(lane) * kGetWaveRecords;
        __gm__ uint8_t* laneMask =
            getMask_ + static_cast<uint64_t>(lane) * kGetWaveRecords;

        for (uint32_t producer = 0U;
             producer < kProducerCount && queued < kGetWaveRecords;
             ++producer) {
            const RingView ring = Ring(producer, lane);
            for (uint32_t slot = 0U;
                 slot < kRingSlots && queued < kGetWaveRecords; ++slot) {
                if (LoadRingWord(ring.SlotState(slot)) != epoch) {
                    continue;
                }
                // Acquire after observing ready.  The producer's release
                // barrier makes the payload and count visible before this
                // point; this barrier prevents following reads from moving
                // ahead of the ready observation.
                DataSyncBarrier<MemDsbT::DDR>();
                const uint64_t count64 = LoadRingWord(ring.SlotCount(slot));
                if (count64 == 0U || count64 > kRecordsPerRingSlot ||
                    queued + count64 > kGetWaveRecords) {
                    if (queued + count64 > kGetWaveRecords) {
                        *deferredReady = true;
                        return queued;  // issue current wave before consuming it
                    }
                    *protocolError = kGetResultRingRecordError;
                    StoreRingWord(ring.SlotState(slot), 0U);
                    continue;
                }
                const uint32_t count = static_cast<uint32_t>(count64);
                for (uint32_t record = 0U; record < count; ++record) {
                    __gm__ int64_t* const words = ring.RecordWord(slot, record, 0);
                    laneIndices[queued + record] = static_cast<int32_t>(
                        LoadRingWord(words));
                    laneOffsets[queued + record] = static_cast<int32_t>(
                        LoadRingWord(words + 1));
                    laneOutputIndices[queued + record] =
                        static_cast<int32_t>(LoadRingWord(words + 2));
                    laneMask[queued + record] = 1U;
                }
                // The descriptor copy is complete.  Only now can this SPSC
                // slot be reused by its producer.
                FlushGmRange(reinterpret_cast<__gm__ uint8_t*>(laneIndices + queued),
                             count * sizeof(int32_t));
                FlushGmRange(reinterpret_cast<__gm__ uint8_t*>(laneOffsets + queued),
                             count * sizeof(int32_t));
                FlushGmRange(
                    reinterpret_cast<__gm__ uint8_t*>(laneOutputIndices + queued),
                    count * sizeof(int32_t));
                FlushGmRange(reinterpret_cast<__gm__ uint8_t*>(laneMask + queued),
                             count * sizeof(uint8_t));
                // The consumer must finish its descriptor stores before it
                // returns this slot to the producer.  The same barrier also
                // completes the bypass record reads for this chunk.
                DataSyncBarrier<MemDsbT::DDR>();
                StoreRingWord(ring.SlotState(slot), 0U);
                queued += count;
                ++(*consumedChunks);
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
        mismatch |= control.output != reinterpret_cast<uint64_t>(packedOutput_)
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

    __aicore__ inline uint64_t IssueFullGet(
        uint32_t lane, uint32_t requestCount, uint64_t* firstDoorbell,
        uint64_t* completionCycle)
    {
        if (requestCount == 0U) {
            return 0U;
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
        __gm__ int32_t* laneIndices =
            getIndices_ + static_cast<uint64_t>(lane) * kGetWaveRecords;
        __gm__ int32_t* laneOffsets =
            getOffsets_ + static_cast<uint64_t>(lane) * kGetWaveRecords;
        __gm__ int32_t* laneOutputIndices =
            getOutputIndices_ + static_cast<uint64_t>(lane) * kGetWaveRecords;
        __gm__ KvafBatchGetCoreStatus* status =
            reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(
                coreStatusAddress +
                static_cast<uint64_t>(lane) * sizeof(KvafBatchGetCoreStatus));
        // Retain the current KVAF profile layout. It yields an exact CQ wait
        // cycle count without introducing a host-side profiler in the hot path.
        status->profile.magic = KVAF_AIV_BATCH_GET_PROFILE_MAGIC;
        FlushGmRange(reinterpret_cast<__gm__ uint8_t*>(status),
                     sizeof(KvafBatchGetCoreStatus));

        KvafExecuteBatchGetKernelArgs args{};
        args.address_table = reinterpret_cast<uint64_t>(addressTable_);
        args.valid_table = reinterpret_cast<uint64_t>(validTable_);
        args.indices = reinterpret_cast<uint64_t>(laneIndices);
        args.offsets = reinterpret_cast<uint64_t>(laneOffsets);
        // The dense helper consumes only [0, requestCount).  `getMask_` is
        // retained as a debug output of the producer/consumer bridge, but it
        // is deliberately not part of this Direct-HCCP submission path.
        args.mask = 0U;
        args.output_indices = reinterpret_cast<uint64_t>(laneOutputIndices);
        args.output = outputAddress;
        args.core_status = reinterpret_cast<uint64_t>(status);
        args.channel_entity = channelEntityAddress +
                              static_cast<uint64_t>(lane) *
                                  sizeof(KvafBatchGetDirectUrmaEntity);
        args.queue_state = queueStateAddress +
                           static_cast<uint64_t>(lane) *
                               sizeof(KvafBatchGetQueueState);
        args.table_capacity = tiling_->tableCapacity;
        args.request_capacity = kGetWaveRecords;
        args.output_rows = tiling_->outputRows;
        args.block_bytes = kPackedEntryBytes;
        args.block_count = 1U;
        args.jettys_per_core = 1U;

        kvaf_fused_get::RunKvafDenseBatchGetCore256(
            args, requestCount, 0U);
        FlushGmRange(reinterpret_cast<__gm__ uint8_t*>(status),
                     sizeof(KvafBatchGetCoreStatus));
        if (*firstDoorbell == 0U) {
            *firstDoorbell = status->profile.first_doorbell_cycle;
        }
        *completionCycle = status->profile.completion_done_cycle;
        return status->result;
    }

    __aicore__ inline void ProcessFullGetLane(uint32_t lane)
    {
        const uint64_t epoch = LoadColdEpoch(frameEpoch_);
        WriteRoleTrace(lane, kRoleTraceGetBase + static_cast<int32_t>(lane));
        __gm__ int64_t* stats = getStats_ +
                                static_cast<uint64_t>(lane) * kGetStatColumns;
        uint64_t requested = 0U;
        uint64_t batches = 0U;
        uint64_t error = epoch == 0U ? kGetResultProtocolError : 0U;
        uint64_t firstDoorbell = 0U;
        uint64_t completionCycle = 0U;
        uint64_t idlePolls = 0U;
        uint64_t consumedChunks = 0U;
        uint32_t queued = 0U;

        while (error == 0U) {
            const uint32_t before = queued;
            bool deferredReady = false;
            queued = ConsumeReadyRings(lane, epoch, queued, &consumedChunks,
                                       &error, &deferredReady);
            const uint32_t target = batches == 0U ? kGetBootstrapRecords
                                                   : kGetWaveRecords;
            const bool allClosed = AllRingsClosed(lane, epoch);
            if (queued >= target ||
                (allClosed && queued != 0U) ||
                (deferredReady && queued != 0U) ||
                (queued == kGetWaveRecords)) {
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
        // Same terminal-publish boundary for the Get lane's telemetry.
        DataSyncBarrier<MemDsbT::DDR>();
    }

private:
    const A5ExactScanSpeculativeGetTilingData* tiling_;
    __gm__ float* liScores_{nullptr};
    __gm__ float* threshold_{nullptr};
    __gm__ uint8_t* resident_{nullptr};
    __gm__ int32_t* tableIndices_{nullptr};
    __gm__ int32_t* offsets_{nullptr};
    __gm__ int32_t* outputIndices_{nullptr};
    __gm__ uint64_t* addressTable_{nullptr};
    __gm__ uint8_t* validTable_{nullptr};
    __gm__ uint8_t* packedOutput_{nullptr};
    __gm__ KvafFusedBatchGetControl* fusedGetControl_{nullptr};
    __gm__ int64_t* frameEpoch_{nullptr};
    __gm__ int32_t* candidateIndices_{nullptr};
    __gm__ uint8_t* candidateKind_{nullptr};
    __gm__ int64_t* producerStats_{nullptr};
    __gm__ int64_t* getStats_{nullptr};
    __gm__ int32_t* roleTrace_{nullptr};
    __gm__ int32_t* getIndices_{nullptr};
    __gm__ int32_t* getOffsets_{nullptr};
    __gm__ int32_t* getOutputIndices_{nullptr};
    __gm__ uint8_t* getMask_{nullptr};
    __gm__ int64_t* pipelineState_{nullptr};
};
}  // namespace

extern "C" __global__ __aicore__ void a5_exact_scan_speculative_get(
    GM_ADDR liScores, GM_ADDR threshold, GM_ADDR resident,
    GM_ADDR tableIndices, GM_ADDR offsets, GM_ADDR outputIndices,
    GM_ADDR addressTable, GM_ADDR validTable, GM_ADDR packedOutput,
    GM_ADDR fusedGetControl, GM_ADDR frameEpoch,
    GM_ADDR pipelineState,
    GM_ADDR packedOutputOut, GM_ADDR candidateIndices, GM_ADDR candidateKind,
    GM_ADDR producerStats,
    GM_ADDR getStats, GM_ADDR roleTrace, GM_ADDR getIndices, GM_ADDR getOffsets,
    GM_ADDR getOutputIndices, GM_ADDR getMask, GM_ADDR pipelineStateOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    // They alias the two mutable inputs by the ACLNN bridge.  Keep named
    // output parameters so CANN graph dataflow records both writes.
    (void)packedOutputOut;
    (void)pipelineStateOut;
    // ExactScan and the full HCCP Get path are both AIV work.  A MIX task
    // created 32 idle AIC lanes and moved the two KVAF workers away from the
    // standalone AIV-only task positions they were created for.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5ExactScanSpeculativeGetTilingData);
    GET_TILING_DATA(tilingData, tiling);
    A5ExactScanSpeculativeGetKernel op(&tilingData);
    op.Init(liScores, threshold, resident, tableIndices, offsets, outputIndices,
            addressTable, validTable, packedOutput, fusedGetControl, frameEpoch,
            pipelineState,
            candidateIndices, candidateKind, producerStats, getStats, roleTrace,
            getIndices, getOffsets, getOutputIndices, getMask);
    op.Process();
}
