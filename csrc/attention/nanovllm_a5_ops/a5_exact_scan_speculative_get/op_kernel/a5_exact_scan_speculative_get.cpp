/**
 * Device-resident P1 ExactScan + full SpeculativeGet.
 *
 * Inputs `li_scores` are deliberately the upstream Lightning score seam, not
 * a replacement implementation of official C8 LI.  Each of the first 62 AIV
 * lanes performs the exact threshold/resident decision and publishes only
 * misses to two fixed SPSC rings.  The final two AIV lanes own one Direct-HCCP
 * queue each and execute the current full KVAF Get transaction:
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
constexpr uint32_t kGetAivBegin = kProducerCount;
constexpr uint32_t kRingSlots = 2;
constexpr uint32_t kRecordsPerRingSlot = 32;
constexpr uint32_t kRecordWords = 3;
constexpr uint32_t kRingWordsPerSlot =
    2 + kRecordsPerRingSlot * kRecordWords;
constexpr uint32_t kRingWords = kRingSlots * kRingWordsPerSlot + 1;
constexpr uint32_t kRingClosedWord = kRingSlots * kRingWordsPerSlot;
constexpr uint32_t kGetBootstrapRecords = 64;
constexpr uint32_t kGetWaveRecords = 256;
constexpr uint32_t kProducerStatColumns = 7;
constexpr uint32_t kGetStatColumns = 8;
constexpr uint32_t kPackedEntryBytes = 656;
constexpr uint64_t kCacheLineBytes = 64;
constexpr uint32_t kConsumerIdlePollLimit = 16U * 1024U * 1024U;

constexpr int32_t kRoleTraceExactScanBase = 1000;
constexpr int32_t kRoleTraceGetBase = 2000;
constexpr uint64_t kGetResultProtocolError = 0x1001U;
constexpr uint64_t kGetResultConsumerTimeout = 0x1002U;
constexpr uint64_t kGetResultRingRecordError = 0x1003U;

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

__aicore__ inline uint64_t LoadEpochWord(__gm__ int64_t* word)
{
    FlushGmRange(reinterpret_cast<__gm__ uint8_t*>(word), sizeof(int64_t));
    __asm__ __volatile__("" ::: "memory");
    return static_cast<uint64_t>(*word);
}

__aicore__ inline void StoreEpochWord(__gm__ int64_t* word, uint64_t value)
{
    __asm__ __volatile__("" ::: "memory");
    *word = static_cast<int64_t>(value);
    FlushGmRange(reinterpret_cast<__gm__ uint8_t*>(word), sizeof(int64_t));
}

struct RingView {
    __gm__ int64_t* words;

    __aicore__ inline __gm__ int64_t* SlotState(uint32_t slot) const
    {
        return words + slot * kRingWordsPerSlot;
    }

    __aicore__ inline __gm__ int64_t* SlotCount(uint32_t slot) const
    {
        return SlotState(slot) + 1;
    }

    __aicore__ inline __gm__ int64_t* RecordWord(
        uint32_t slot, uint32_t record, uint32_t word) const
    {
        return SlotState(slot) + 2 + record * kRecordWords + word;
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
            const uint32_t logicalAiv = GetBlockIdx();
            if (logicalAiv < tiling_->producerCount) {
                ProcessExactScanProducer(logicalAiv);
            } else {
                const uint32_t lane = logicalAiv - kGetAivBegin;
                if (lane < tiling_->getLaneCount) {
                    ProcessFullGetLane(lane);
                }
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
        // Payload and count reach GM before the ready epoch.  The consumer
        // invalidates the ready line before dereferencing the payload.
        FlushGmRange(
            reinterpret_cast<__gm__ uint8_t*>(ring.RecordWord(slot, 0, 0)),
            static_cast<uint64_t>(count) * kRecordWords * sizeof(int64_t));
        *ring.SlotCount(slot) = static_cast<int64_t>(count);
        FlushGmRange(reinterpret_cast<__gm__ uint8_t*>(ring.SlotCount(slot)),
                     sizeof(int64_t));
        StoreEpochWord(ring.SlotState(slot), epoch);
    }

    __aicore__ inline bool TryAppendMiss(
        const RingView& ring, uint32_t* writeSlot, uint32_t* writeCount,
        uint64_t epoch, int32_t tableIndex, int32_t offset,
        int32_t outputIndex, uint64_t candidateSlot,
        uint64_t* pendingCandidateSlots, uint64_t* firstPublishCycle)
    {
        if (*writeCount == 0U && LoadEpochWord(ring.SlotState(*writeSlot)) == epoch) {
            return false;
        }
        __gm__ int64_t* const tableWord =
            ring.RecordWord(*writeSlot, *writeCount, 0);
        *tableWord = static_cast<int64_t>(tableIndex);
        *(tableWord + 1) = static_cast<int64_t>(offset);
        *(tableWord + 2) = static_cast<int64_t>(outputIndex);
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
            if (LoadEpochWord(ring.SlotState(writeSlot)) == epoch) {
                *overflow = 1U;
                for (uint32_t record = 0U; record < writeCount; ++record) {
                    candidateKind_[pendingCandidateSlots[record]] =
                        kCandidateDdrDeferred;
                }
                *deferred += writeCount;
            } else {
                PublishRingSlot(ring, writeSlot, writeCount, epoch);
                if (*firstPublishCycle == 0U) {
                    *firstPublishCycle = static_cast<uint64_t>(GetSystemCycle());
                }
            }
        }
        StoreEpochWord(ring.ClosedEpoch(), epoch);
    }

    __aicore__ inline void ProcessExactScanProducer(uint32_t producer)
    {
        const uint64_t epoch = static_cast<uint64_t>(frameEpoch_[0]);
        roleTrace_[producer] = kRoleTraceExactScanBase +
                               static_cast<int32_t>(producer);
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
            candidateIndices_[candidateSlot] = static_cast<int32_t>(source);
            const bool isResident = resident_[source] != 0U;
            ++selected;
            if (isResident) {
                candidateKind_[candidateSlot] = kCandidateHbmHit;
                ++hit;
                continue;
            }

            ++miss;
            const int32_t tableIndex = tableIndices_[source];
            const int32_t offset = offsets_[source];
            const int32_t outputIndex = outputIndices_[source];
            if (!IsValidDirectGetDescriptor(tableIndex, offset, outputIndex)) {
                candidateKind_[candidateSlot] = kCandidateDdrDeferred;
                ++deferred;
                continue;
            }
            // This fixed route makes every producer/lane queue SPSC.  The
            // production C8 integration replaces source>>7 by row^block.
            const uint32_t lane = static_cast<uint32_t>((source >> 7U) & 1U);
            if (laneOverflow[lane] != 0U) {
                candidateKind_[candidateSlot] = kCandidateDdrDeferred;
                ++deferred;
                continue;
            }
            const RingView ring = Ring(producer, lane);
            if (!TryAppendMiss(ring, &writeSlot[lane], &writeCount[lane], epoch,
                               tableIndex, offset, outputIndex,
                               candidateSlot, pendingCandidateSlots[lane],
                               &firstPublishCycle[lane])) {
                laneOverflow[lane] = 1U;
                candidateKind_[candidateSlot] = kCandidateDdrDeferred;
                ++deferred;
            } else {
                candidateKind_[candidateSlot] = kCandidateDdrEnqueued;
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
        stats[0] = static_cast<int64_t>(selected);
        stats[1] = static_cast<int64_t>(hit);
        stats[2] = static_cast<int64_t>(miss);
        stats[3] = static_cast<int64_t>(enqueued);
        stats[4] = static_cast<int64_t>(deferred);
        stats[5] = static_cast<int64_t>(firstPublish);
        stats[6] = static_cast<int64_t>(scanDone);
        FlushGmRange(reinterpret_cast<__gm__ uint8_t*>(stats),
                     kProducerStatColumns * sizeof(int64_t));
    }

    __aicore__ inline bool AllRingsClosed(uint32_t lane, uint64_t epoch) const
    {
        for (uint32_t producer = 0U; producer < kProducerCount; ++producer) {
            if (LoadEpochWord(Ring(producer, lane).ClosedEpoch()) != epoch) {
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
                if (LoadEpochWord(ring.SlotState(slot)) != epoch) {
                    continue;
                }
                const uint64_t count64 = LoadEpochWord(ring.SlotCount(slot));
                if (count64 == 0U || count64 > kRecordsPerRingSlot ||
                    queued + count64 > kGetWaveRecords) {
                    if (queued + count64 > kGetWaveRecords) {
                        *deferredReady = true;
                        return queued;  // issue current wave before consuming it
                    }
                    *protocolError = kGetResultRingRecordError;
                    StoreEpochWord(ring.SlotState(slot), 0U);
                    continue;
                }
                const uint32_t count = static_cast<uint32_t>(count64);
                FlushGmRange(
                    reinterpret_cast<__gm__ uint8_t*>(ring.RecordWord(slot, 0, 0)),
                    static_cast<uint64_t>(count) * kRecordWords * sizeof(int64_t));
                for (uint32_t record = 0U; record < count; ++record) {
                    __gm__ int64_t* const words = ring.RecordWord(slot, record, 0);
                    laneIndices[queued + record] = static_cast<int32_t>(words[0]);
                    laneOffsets[queued + record] = static_cast<int32_t>(words[1]);
                    laneOutputIndices[queued + record] =
                        static_cast<int32_t>(words[2]);
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
                StoreEpochWord(ring.SlotState(slot), 0U);
                queued += count;
                ++(*consumedChunks);
            }
        }
        return queued;
    }

    __aicore__ inline uint64_t ValidateControl(uint32_t lane) const
    {
        // fusedGetControl_ is an HBM (__gm__) object.  AscendC must load its
        // scalar fields through the GM pointer; binding it to an ordinary C++
        // reference would discard the address-space qualifier at compilation.
        if (fusedGetControl_->magic != KVAF_FUSED_BATCH_GET_CONTROL_MAGIC ||
            fusedGetControl_->version != KVAF_FUSED_BATCH_GET_CONTROL_VERSION ||
            fusedGetControl_->channel_entity == 0U ||
            fusedGetControl_->queue_state == 0U ||
            fusedGetControl_->core_status == 0U ||
            fusedGetControl_->output == 0U ||
            fusedGetControl_->output != reinterpret_cast<uint64_t>(packedOutput_) ||
            fusedGetControl_->output_rows != tiling_->outputRows ||
            fusedGetControl_->block_count != kGetLaneCount ||
            fusedGetControl_->jettys_per_core != 1U ||
            fusedGetControl_->block_bytes != kPackedEntryBytes ||
            lane >= kGetLaneCount) {
            return kGetResultProtocolError;
        }
        return 0U;
    }

    __aicore__ inline uint64_t IssueFullGet(
        uint32_t lane, uint32_t requestCount, uint64_t* firstDoorbell,
        uint64_t* completionCycle)
    {
        if (requestCount == 0U) {
            return 0U;
        }
        const uint64_t controlError = ValidateControl(lane);
        if (controlError != 0U) {
            return controlError;
        }
        // The lifecycle contract freezes this control block for the whole
        // graph replay.  Materialize only the scalar addresses needed below
        // from GM into AIV-local registers; do not form a non-GM struct
        // reference to HBM memory.
        const uint64_t outputAddress = fusedGetControl_->output;
        const uint64_t coreStatusAddress = fusedGetControl_->core_status;
        const uint64_t channelEntityAddress = fusedGetControl_->channel_entity;
        const uint64_t queueStateAddress = fusedGetControl_->queue_state;
        __gm__ int32_t* laneIndices =
            getIndices_ + static_cast<uint64_t>(lane) * kGetWaveRecords;
        __gm__ int32_t* laneOffsets =
            getOffsets_ + static_cast<uint64_t>(lane) * kGetWaveRecords;
        __gm__ int32_t* laneOutputIndices =
            getOutputIndices_ + static_cast<uint64_t>(lane) * kGetWaveRecords;
        __gm__ uint8_t* laneMask =
            getMask_ + static_cast<uint64_t>(lane) * kGetWaveRecords;
        for (uint32_t index = requestCount; index < kGetWaveRecords; ++index) {
            laneMask[index] = 0U;
        }
        FlushGmRange(reinterpret_cast<__gm__ uint8_t*>(laneMask),
                     kGetWaveRecords * sizeof(uint8_t));

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
        args.mask = reinterpret_cast<uint64_t>(laneMask);
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

        kvaf_fused_get::RunKvafFullBatchGetCore(args, 0U);
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
        const uint64_t epoch = static_cast<uint64_t>(frameEpoch_[0]);
        roleTrace_[kGetAivBegin + lane] =
            kRoleTraceGetBase + static_cast<int32_t>(lane);
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

        stats[0] = static_cast<int64_t>(requested);
        stats[1] = static_cast<int64_t>(batches);
        stats[2] = static_cast<int64_t>(error);
        stats[3] = static_cast<int64_t>(firstDoorbell);
        stats[4] = static_cast<int64_t>(completionCycle);
        stats[5] = static_cast<int64_t>(idlePolls);
        stats[6] = static_cast<int64_t>(consumedChunks);
        stats[7] = static_cast<int64_t>(GetSystemCycle());
        FlushGmRange(reinterpret_cast<__gm__ uint8_t*>(stats),
                     kGetStatColumns * sizeof(int64_t));
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
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
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
