// P9 Stage-B terminalizer for the source-derived A5 C8 operator.
//
// This is deliberately a tail after the official ProcessTopK writeback.  It
// reads the already-final local Top-2048 indices; it never changes a score,
// the official Top-K working storage, or the original sparse-index DMA.
// Its only job is to make the exact Top-K's KV locations concrete in the
// caller-owned normal-resident allocation before QSFA can read them:
//
//   1. discard never-posted score-time seed batches;
//   2. drain the one remaining initial Direct-HCCP wave, if any;
//   3. resolve each exact winner to a persistent slot, an already-terminal
//      speculative fixed slot, or a free fixed slot;
//   4. post exact recalls in bounded 256-request waves and drain each wave;
//   5. expose attention_slots only after every selected payload is terminal.
//
// The fixed slots belong to this one C8 invocation.  They are not a private
// cache, are never returned, and have no cross-invocation lifecycle.  The
// only temporary resource is the Direct-HCCP queue lease inside the executor.

#ifndef VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_FINALIZER_H_
#define VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_FINALIZER_H_

#include <climits>
#include <cstdint>

#include "kernel_operator.h"

#include "a5_c8_exact_speculative_resident_contract.h"
#include "a5_c8_exact_speculative_resident_post_executor.h"
#include "a5_c8_exact_speculative_resident_score_hook.h"

namespace QLIKernel {
namespace exact_speculative_resident {

using namespace AscendC;
using kvaf_layer_resident_get::KvafLayerResidentQueueLeaseResult;
using vllm_ascend::a5_c8_exact_speculative_resident::CandidateSeed;
using vllm_ascend::a5_c8_exact_speculative_resident::CanonicalResidentLayout;
using vllm_ascend::a5_c8_exact_speculative_resident::IsCanonicalLayout;
using vllm_ascend::a5_c8_exact_speculative_resident::IsSlotInCanonicalRow;
using vllm_ascend::a5_c8_exact_speculative_resident::kCanonicalResidentSlotCount;
using vllm_ascend::a5_c8_exact_speculative_resident::kExactRecallWaveCapacity;
using vllm_ascend::a5_c8_exact_speculative_resident::kExactRecallSeedScratchOffset;
using vllm_ascend::a5_c8_exact_speculative_resident::kExistingC8InvalidScratchBytes;
using vllm_ascend::a5_c8_exact_speculative_resident::kP9ScratchBytes;
using vllm_ascend::a5_c8_exact_speculative_resident::kStageBQueueLeaseReceiptScratchOffset;
using vllm_ascend::a5_c8_exact_speculative_resident::kStageBReservationScratchOffset;
using vllm_ascend::a5_c8_exact_speculative_resident::kStageBUsedSlotBitmapBytes;
using vllm_ascend::a5_c8_exact_speculative_resident::kStageBUsedSlotBitmapScratchOffset;

// The graph-visible diagnostic output is intentionally small and row-local.
// It does not participate in slot selection or transport correctness.
enum TerminalPostStatWord : uint32_t {
    kTerminalPersistentWinnerCount = 0U,
    kTerminalSpeculativeWinnerCount = 1U,
    kTerminalRecallWinnerCount = 2U,
    kTerminalRecallWaveCount = 3U,
    kTerminalQueueBusyRetryCount = 4U,
    kTerminalInvalidIndexCount = 5U,
    kTerminalFailure = 6U,
    kTerminalPostStatWordCount = 7U,
};

struct C8ExactSpeculativeResidentTerminalStats {
    uint32_t persistent_winners;
    uint32_t speculative_winners;
    uint32_t recall_winners;
    uint32_t recall_waves;
    uint32_t queue_busy_retries;
    uint32_t invalid_indices;
    uint32_t failure;
};

// Every pointer is graph-visible HBM.  `attentionSlots` is [R, 2048].  Its
// entries are graph-visible only after this operator returns: persistent slots
// are already ready, speculative slots are read only after the initial drain,
// and recall slots are terminal before the finalizer returns.  The finalizer
// may fill its output entry before the corresponding recall CQ terminal, but
// no downstream graph node is permitted to observe it at that point.
struct C8ExactSpeculativeResidentFinalizerConfig {
    __gm__ uint8_t* reqPoolEntries;
    __gm__ uint8_t* kvObjectMap;
    __gm__ uint8_t* kvObjectLens;
    __gm__ uint8_t* persistentObjectSlotsSnapshot;
    __gm__ uint8_t* candidateSourceToSlot;
    __gm__ uint8_t* attentionSlots;
    __gm__ uint8_t* terminalPostStats;
    __gm__ uint8_t* layerResidentGetControl;

    uint32_t rowCount;
    uint32_t sourceCapacity;
    uint32_t poolSize;
    uint32_t kvObjectCapacity;
    uint32_t rowsPerEntry;
    uint32_t producerId;
    CanonicalResidentLayout residentLayout;
};

// Queue contention is expected when multiple LI workers finish score scan at
// once.  A bounded retry is therefore distinct from a malformed request: a
// busy queue may be retried, whereas any other false result fails closed and
// cannot expose an incomplete attention_slots row.
constexpr uint32_t kExactRecallQueueRetryLimit = 128U;

class C8ExactSpeculativeResidentFinalizer {
public:
    __aicore__ inline void Init(
        const C8ExactSpeculativeResidentFinalizerConfig& config,
        LocalTensor<int32_t> existing_invalid_scratch,
        event_t simt_event)
    {
        config_ = config;
        valid_ = ValidateConfig(config_);
        simt_event_ = simt_event;
        if (!valid_) {
            return;
        }
        scratch_ = reinterpret_cast<__ubuf__ uint8_t*>(
            existing_invalid_scratch.GetPhyAddr());
        if (scratch_ == nullptr) {
            valid_ = false;
            return;
        }
        reqPoolEntries_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config_.reqPoolEntries));
        kvObjectMap_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config_.kvObjectMap));
        kvObjectLens_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config_.kvObjectLens));
        persistentObjectSlots_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(
                config_.persistentObjectSlotsSnapshot));
        candidateSourceToSlot_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config_.candidateSourceToSlot));
        attentionSlots_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config_.attentionSlots));
        if (config_.terminalPostStats != nullptr) {
            terminalPostStats_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t*>(config_.terminalPostStats));
        }
    }

    __aicore__ inline bool Enabled() const
    {
        return valid_;
    }

    // `row` is the flattened output row used by both the score seam and the
    // official indiceOutGm offset.  Call this exactly after that row's
    // unchanged official DataCopyPad(indiceOutGm, indicesOutLocal_) and its
    // original MTE3->V acknowledgement.
    __aicore__ inline bool FinalizeExactTopKRow(
        uint32_t row, LocalTensor<uint32_t> exact_topk_indices,
        C8ExactSpeculativeResidentScoreHook* score_hook,
        C8ExactSpeculativeResidentPostExecutor* initial_executor)
    {
        C8ExactSpeculativeResidentTerminalStats stats{};
        if (!valid_ || row >= config_.rowCount) {
            stats.failure = 1U;
            StoreStats(row, stats);
            return false;
        }

        // At this boundary an unposted score-time batch has no descriptor and
        // can safely be deferred.  A handed batch is intentionally untouched:
        // the initial executor owns it until CQ terminal.
        if (score_hook != nullptr) {
            score_hook->DeferAllReadyPostViews();
        }
        if (initial_executor != nullptr &&
            !initial_executor->DrainAllBeforeReturn()) {
            stats.failure = 1U;
            StoreStats(row, stats);
            return false;
        }

        // The initial executor is now terminal, so no DMA can still observe
        // the score-time seed bytes.  Rebind the existing 8 KiB invalid-index
        // scratch for Stage B: [0,4096) recall seeds, [4096,4352) bitmap,
        // then the two 16-byte SIMT receipts.  No P9 TBuf is introduced.
        __ubuf__ CandidateSeed* const recall_seeds =
            reinterpret_cast<__ubuf__ CandidateSeed*>(
                scratch_ + kExactRecallSeedScratchOffset);
        __ubuf__ uint8_t* const used_slots =
            scratch_ + kStageBUsedSlotBitmapScratchOffset;
        __ubuf__ KvafLayerResidentQueueLeaseResult* const lease_result =
            reinterpret_cast<__ubuf__ KvafLayerResidentQueueLeaseResult*>(
                scratch_ + kStageBQueueLeaseReceiptScratchOffset);
        __ubuf__ CandidateSlotReservationResult* const reservation_result =
            reinterpret_cast<__ubuf__ CandidateSlotReservationResult*>(
                scratch_ + kStageBReservationScratchOffset);
        ClearUsedSlots(used_slots);

        C8ExactSpeculativeResidentPostExecutor recall_executor;
        recall_executor.Init(config_.layerResidentGetControl, config_.producerId,
                             lease_result, reservation_result, simt_event_);
        if (!recall_executor.Enabled()) {
            stats.failure = 1U;
            StoreStats(row, stats);
            return false;
        }

        uint32_t next_free_slot = 0U;
        uint32_t recall_count = 0U;
        for (uint32_t topk_index = 0U;
             topk_index < kCanonicalResidentSlotCount; ++topk_index) {
            const int32_t source = static_cast<int32_t>(
                exact_topk_indices.GetValue(topk_index));
            if (source < 0) {
                StoreAttentionSlot(row, topk_index, -1);
                ++stats.invalid_indices;
                continue;
            }

            int32_t pool_row = -1;
            int32_t kv_object = -1;
            if (!ResolveKvObject(row, static_cast<uint32_t>(source), &pool_row,
                                 &kv_object)) {
                stats.failure = 1U;
                StoreStats(row, stats);
                return false;
            }

            const int32_t persistent_slot = persistentObjectSlots_.GetValue(
                static_cast<uint64_t>(pool_row) * config_.kvObjectCapacity +
                static_cast<uint32_t>(kv_object));
            if (persistent_slot >= 0) {
                if (IsSlotInCanonicalRow(config_.residentLayout, row,
                                         persistent_slot)) {
                    MarkUsedSlot(row, persistent_slot, used_slots);
                }
                StoreAttentionSlot(row, topk_index, persistent_slot);
                ++stats.persistent_winners;
                continue;
            }

            const int32_t candidate_slot = candidateSourceToSlot_.GetValue(
                static_cast<uint64_t>(row) * config_.sourceCapacity +
                static_cast<uint32_t>(source));
            if (candidate_slot == kCandidateSourceClaiming) {
                // Drain preceding the map read makes this a protocol error,
                // not an ordinary miss.  Returning a row that might still be
                // written would break the fixed-slot output contract.
                stats.failure = 1U;
                StoreStats(row, stats);
                return false;
            }
            if (candidate_slot >= 0) {
                if (!IsSlotInCanonicalRow(config_.residentLayout, row,
                                          candidate_slot)) {
                    stats.failure = 1U;
                    StoreStats(row, stats);
                    return false;
                }
                MarkUsedSlot(row, candidate_slot, used_slots);
                StoreAttentionSlot(row, topk_index, candidate_slot);
                ++stats.speculative_winners;
                continue;
            }
            if (candidate_slot != kCandidateSourceUnmapped) {
                stats.failure = 1U;
                StoreStats(row, stats);
                return false;
            }
            if (recall_count == kExactRecallWaveCapacity) {
                if (!DrainRecallWave(&recall_executor, recall_seeds,
                                      &recall_count, &stats)) {
                    stats.failure = 1U;
                    StoreStats(row, stats);
                    return false;
                }
            }

            int32_t recall_slot = -1;
            if (!AllocateUnusedSlot(row, used_slots, &next_free_slot,
                                    &recall_slot)) {
                stats.failure = 1U;
                StoreStats(row, stats);
                return false;
            }
            recall_seeds[recall_count].row = static_cast<int32_t>(row);
            recall_seeds[recall_count].source = source;
            recall_seeds[recall_count].kvObject = kv_object;
            recall_seeds[recall_count].residentSlot = recall_slot;
            ++recall_count;
            ++stats.recall_winners;
            StoreAttentionSlot(row, topk_index, recall_slot);

            if (recall_count == kExactRecallWaveCapacity &&
                !DrainRecallWave(&recall_executor, recall_seeds,
                                 &recall_count, &stats)) {
                stats.failure = 1U;
                StoreStats(row, stats);
                return false;
            }
        }

        if (!DrainRecallWave(&recall_executor, recall_seeds, &recall_count,
                              &stats)) {
            stats.failure = 1U;
            StoreStats(row, stats);
            return false;
        }
        StoreStats(row, stats);
        return true;
    }

private:
    __aicore__ inline bool ValidateConfig(
        const C8ExactSpeculativeResidentFinalizerConfig& config) const
    {
        return config.reqPoolEntries != nullptr && config.kvObjectMap != nullptr &&
               config.kvObjectLens != nullptr &&
               config.persistentObjectSlotsSnapshot != nullptr &&
               config.candidateSourceToSlot != nullptr &&
               config.attentionSlots != nullptr &&
               config.layerResidentGetControl != nullptr && config.rowCount != 0U &&
               config.sourceCapacity != 0U && config.poolSize != 0U &&
               config.kvObjectCapacity != 0U && config.rowsPerEntry != 0U &&
               IsCanonicalLayout(config.residentLayout) &&
               config.residentLayout.rowCount == config.rowCount &&
               kP9ScratchBytes <= kExistingC8InvalidScratchBytes;
    }

    __aicore__ inline void ClearUsedSlots(__ubuf__ uint8_t* used_slots) const
    {
        if (used_slots == nullptr) {
            return;
        }
        for (uint32_t byte = 0U; byte < kStageBUsedSlotBitmapBytes; ++byte) {
            used_slots[byte] = 0U;
        }
    }

    __aicore__ inline bool IsUsedSlot(uint32_t row, int32_t resident_slot,
                                      __ubuf__ const uint8_t* used_slots) const
    {
        if (used_slots == nullptr ||
            !IsSlotInCanonicalRow(config_.residentLayout, row,
                                  resident_slot)) {
            return true;
        }
        const int64_t first =
            static_cast<int64_t>(config_.residentLayout.firstSlot) +
            static_cast<int64_t>(row) * config_.residentLayout.rowStride;
        const uint32_t local = static_cast<uint32_t>(
            static_cast<int64_t>(resident_slot) - first);
        return (used_slots[local / 8U] & (1U << (local % 8U))) != 0U;
    }

    __aicore__ inline void MarkUsedSlot(uint32_t row, int32_t resident_slot,
                                        __ubuf__ uint8_t* used_slots) const
    {
        if (used_slots == nullptr ||
            !IsSlotInCanonicalRow(config_.residentLayout, row,
                                  resident_slot)) {
            return;
        }
        const int64_t first =
            static_cast<int64_t>(config_.residentLayout.firstSlot) +
            static_cast<int64_t>(row) * config_.residentLayout.rowStride;
        const uint32_t local = static_cast<uint32_t>(
            static_cast<int64_t>(resident_slot) - first);
        used_slots[local / 8U] |= static_cast<uint8_t>(1U << (local % 8U));
    }

    __aicore__ inline bool AllocateUnusedSlot(uint32_t row,
                                               __ubuf__ uint8_t* used_slots,
                                               uint32_t* next_free_slot,
                                               int32_t* resident_slot) const
    {
        if (used_slots == nullptr || next_free_slot == nullptr ||
            resident_slot == nullptr) {
            return false;
        }
        const int64_t first =
            static_cast<int64_t>(config_.residentLayout.firstSlot) +
            static_cast<int64_t>(row) * config_.residentLayout.rowStride;
        for (uint32_t probe = *next_free_slot;
             probe < kCanonicalResidentSlotCount; ++probe) {
            const int64_t candidate = first + probe;
            if (candidate < 0 || candidate > INT32_MAX ||
                IsUsedSlot(row, static_cast<int32_t>(candidate), used_slots)) {
                continue;
            }
            *resident_slot = static_cast<int32_t>(candidate);
            MarkUsedSlot(row, *resident_slot, used_slots);
            *next_free_slot = probe + 1U;
            return true;
        }
        return false;
    }

    __aicore__ inline bool ResolveKvObject(uint32_t row, uint32_t source,
                                           int32_t* pool_row,
                                           int32_t* kv_object) const
    {
        if (pool_row == nullptr || kv_object == nullptr ||
            row >= config_.rowCount || source >= config_.sourceCapacity) {
            return false;
        }
        const int32_t resolved_pool_row = reqPoolEntries_.GetValue(row);
        const int32_t object_len = kvObjectLens_.GetValue(row);
        if (resolved_pool_row < 0 ||
            static_cast<uint32_t>(resolved_pool_row) >= config_.poolSize ||
            object_len <= 0 ||
            static_cast<uint32_t>(object_len) > config_.kvObjectCapacity) {
            return false;
        }
        const int32_t resolved_object = kvObjectMap_.GetValue(
            static_cast<uint64_t>(resolved_pool_row) * config_.sourceCapacity +
            source);
        if (resolved_object < 0 || resolved_object >= object_len ||
            static_cast<uint32_t>(resolved_object) >= config_.kvObjectCapacity) {
            return false;
        }
        *pool_row = resolved_pool_row;
        *kv_object = resolved_object;
        return true;
    }

    __aicore__ inline bool DrainRecallWave(
        C8ExactSpeculativeResidentPostExecutor* executor,
        __ubuf__ CandidateSeed* recall_seeds, uint32_t* recall_count,
        C8ExactSpeculativeResidentTerminalStats* stats)
    {
        if (executor == nullptr || recall_seeds == nullptr ||
            recall_count == nullptr || stats == nullptr) {
            return false;
        }
        if (*recall_count == 0U) {
            return true;
        }
        bool posted = false;
        for (uint32_t retry = 0U; retry < kExactRecallQueueRetryLimit;
             ++retry) {
            if (executor->TryPostFixedRecallSeeds(
                    recall_seeds, *recall_count, config_.residentLayout,
                    config_.rowsPerEntry)) {
                posted = true;
                break;
            }
            if (!executor->Enabled()) {
                return false;
            }
            ++stats->queue_busy_retries;
        }
        if (!posted || !executor->DrainAllBeforeReturn()) {
            return false;
        }
        ++stats->recall_waves;
        *recall_count = 0U;
        return true;
    }

    __aicore__ inline void StoreAttentionSlot(uint32_t row, uint32_t index,
                                              int32_t slot)
    {
        attentionSlots_.SetValue(
            static_cast<uint64_t>(row) * kCanonicalResidentSlotCount + index,
            slot);
    }

    __aicore__ inline void StoreStats(
        uint32_t row, const C8ExactSpeculativeResidentTerminalStats& stats)
    {
        if (config_.terminalPostStats == nullptr || row >= config_.rowCount) {
            return;
        }
        const uint64_t base = static_cast<uint64_t>(row) *
            kTerminalPostStatWordCount;
        terminalPostStats_.SetValue(base + kTerminalPersistentWinnerCount,
                                    static_cast<int32_t>(stats.persistent_winners));
        terminalPostStats_.SetValue(base + kTerminalSpeculativeWinnerCount,
                                    static_cast<int32_t>(stats.speculative_winners));
        terminalPostStats_.SetValue(base + kTerminalRecallWinnerCount,
                                    static_cast<int32_t>(stats.recall_winners));
        terminalPostStats_.SetValue(base + kTerminalRecallWaveCount,
                                    static_cast<int32_t>(stats.recall_waves));
        terminalPostStats_.SetValue(base + kTerminalQueueBusyRetryCount,
                                    static_cast<int32_t>(stats.queue_busy_retries));
        terminalPostStats_.SetValue(base + kTerminalInvalidIndexCount,
                                    static_cast<int32_t>(stats.invalid_indices));
        terminalPostStats_.SetValue(base + kTerminalFailure,
                                    static_cast<int32_t>(stats.failure));
    }

    C8ExactSpeculativeResidentFinalizerConfig config_{};
    __ubuf__ uint8_t* scratch_ = nullptr;
    event_t simt_event_ = static_cast<event_t>(EVENT_ID0);
    GlobalTensor<int32_t> reqPoolEntries_;
    GlobalTensor<int32_t> kvObjectMap_;
    GlobalTensor<int32_t> kvObjectLens_;
    GlobalTensor<int32_t> persistentObjectSlots_;
    GlobalTensor<int32_t> candidateSourceToSlot_;
    GlobalTensor<int32_t> attentionSlots_;
    GlobalTensor<int32_t> terminalPostStats_;
    bool valid_ = false;
};

}  // namespace exact_speculative_resident
}  // namespace QLIKernel

#endif  // VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_FINALIZER_H_
