// P9 score-seam hook for the source-derived A5 C8 operator.
//
// This header is intentionally narrow.  It observes the exact uint16 score
// tile produced by the official LI reduction, compacts cold candidates into
// caller-owned 2 KiB scratch, and hands a batch to the *dynamic* PostOnly
// executor after C8 has released that score tile.  It does not own a Direct-
// HCCP queue, a resident slot lifetime, or a physical AIV role.
//
// Required source-derived insertion point in QLIVector::ProcessVec1:
//
//   BatchMulWeightAndReduceSum(outBase, ...);
//   // P9 V->S / S->V lifetime envelope, then CompactExactScoreTile(...)
//   DataCopyPad(scoreGm[...], outBase, ...);
//
// The caller may obtain a ready CandidatePostView only after the S->V
// acknowledgement and the normal C8 score publication hand-off.  The Post
// executor dynamically leases a KVAF queue, fills its descriptor, publishes
// it, rings PostOnly, and later drains CQ.  This keeps score-UB ownership and
// transport ownership separate without creating a sidecar score recompute.

#ifndef VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_HOOK_H_
#define VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_HOOK_H_

#include <cstdint>

#include "kernel_operator.h"
#include "quant_lightning_indexer_common.h"

#include "a5_c8_exact_speculative_resident_contract.h"

// The strict reference binary keeps this at zero.  A P9 target binary must
// explicitly select one, then still pass a graph-visible runtime enable word
// before this hook observes a score.  This two-stage gate makes the no-hook
// C8 bitwise control mechanically distinct from the resident-get closure.
#ifndef A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM
#define A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM 0
#endif

static_assert(A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM == 0 ||
                  A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM == 1,
              "P9 score seam must be disabled (0) or enabled (1).");

namespace QLIKernel {
namespace exact_speculative_resident {

using namespace AscendC;
using vllm_ascend::a5_c8_exact_speculative_resident::CandidateSeed;
using vllm_ascend::a5_c8_exact_speculative_resident::CanonicalResidentSlotTable;
using vllm_ascend::a5_c8_exact_speculative_resident::IsCanonicalResidentSlotTableValid;
using vllm_ascend::a5_c8_exact_speculative_resident::kCandidateMicrobatchCapacity;
using vllm_ascend::a5_c8_exact_speculative_resident::kCandidateMicrobatchCount;
using vllm_ascend::a5_c8_exact_speculative_resident::kCanonicalResidentSlotCount;

constexpr int32_t kCandidateSourceUnmapped = -1;
constexpr int32_t kCandidateResidentSlotUnassigned = -1;

enum class CandidateMicrobatchState : uint32_t {
    kFree = 0U,
    kReady = 1U,
    // Ownership changes to the dynamic Post/CQ executor only after TakeReady
    // succeeds.  The score seam must not reuse the batch until the executor
    // has copied it into the queue descriptor and accepted PostOnly.
    kHandedToExecutor = 2U,
};

// The hook exposes precisely the identities required by a Direct-HCCP
// PostOnly executor.  It deliberately contains no queue id or AIV id: queue
// acquisition is a dynamic operation over the control's shared lease pool.
struct CandidatePostView {
    __ubuf__ CandidateSeed* seeds;
    uint32_t count;
    uint32_t producerId;

    __gm__ int32_t* candidateSourceToSlot;
    __gm__ int32_t* candidateSlotCursor;
    __gm__ int32_t* reqPoolEntries;
    __gm__ int32_t* canonicalResidentSlots;
    __gm__ int32_t* residentBlockTable;
    __gm__ int32_t* persistentObjectSlotsSnapshot;
    __gm__ uint8_t* layerResidentGetControl;

    // Execution rows may be a permutation of resident-pool rows.  The
    // executor resolves row -> req_pool_entries[row] -> canonical table.
    uint32_t rowCount;
    uint32_t poolSize;
    uint32_t residentBlockCapacity;
    uint32_t kvObjectCapacity;
    uint32_t sourceCapacity;
    uint32_t rowsPerEntry;
    // Graph replay may pad execution rows by reusing a pool row.  P9 must
    // never turn that alias into a speculative post; only [0, activeRowCount)
    // belongs to this invocation.
    uint32_t activeRowCount;
};

static_assert(sizeof(CandidateSeed) == 4U * sizeof(int32_t),
              "P9 score seam expects [row, source, object, slot_ordinal].");

// Graph-visible inputs owned by the P9 operator.  All pointers are HBM
// tensors; no pointer refers to an implicit host callback or a transient
// slot-pool protocol.  `runtimeEnable` is int32[1] and must be exactly one to enable a
// compiled P9 hook for an invocation.
struct C8ExactSpeculativeResidentScoreHookConfig {
    __gm__ uint8_t* runtimeEnable;
    __gm__ uint8_t* residentThreshold;
    __gm__ uint8_t* residentThresholdValid;
    __gm__ uint8_t* reqPoolEntries;
    __gm__ uint8_t* kvObjectMap;
    __gm__ uint8_t* kvObjectLens;
    __gm__ uint8_t* persistentObjectSlotsSnapshot;
    __gm__ uint8_t* canonicalResidentSlots;
    __gm__ uint8_t* residentBlockTable;
    __gm__ uint8_t* activeRowCount;

    // `candidateSourceToSlot` has shape [rowCount, sourceCapacity], starts
    // at -1, and is committed by the Post executor only after PostOnly has
    // accepted the fully published descriptor. `candidateSlotCursor` has one
    // atomic cursor per row. No inverse slot->source map is needed: after the
    // initial CQ drain, Stage B derives a 2048-bit used-slot bitmap from the
    // exact Top-K and reuses every nonwinner fixed slot for recall.
    __gm__ uint8_t* candidateSourceToSlot;
    __gm__ uint8_t* candidateSlotCursor;
    __gm__ uint8_t* layerResidentGetControl;

    uint32_t rowCount;
    uint32_t sourceCapacity;
    uint32_t poolSize;
    uint32_t kvObjectCapacity;
    uint32_t residentBlockCapacity;
    uint32_t rowsPerEntry;
    uint32_t producerId;
};

// Two fixed microbatches are supplied by the source-derived vector service.
// They must alias the already-reserved 2 KiB C8 scratch, not a newly allocated
// TBuf.  This is the bounded backpressure point: if both are busy, the hook
// drops only a speculative opportunity; final Exact Top-K recall remains
// authoritative.
class C8ExactSpeculativeResidentScoreHook {
public:
    __aicore__ inline void Init(
        const C8ExactSpeculativeResidentScoreHookConfig& config,
        const QLICommon::ConstInfo& constInfo,
        __ubuf__ CandidateSeed* candidateScratch)
    {
        config_ = config;
        constInfo_ = constInfo;
        enabled_ = false;
        deferredCandidates_ = 0U;
        activeRowCountValue_ = 0U;
        lastTileFirstActiveRow_ = config.rowCount;
        for (uint32_t slot = 0U; slot < kCandidateMicrobatchCount; ++slot) {
            batches_[slot].seeds = candidateScratch == nullptr
                ? nullptr
                : candidateScratch + static_cast<uint64_t>(slot) *
                    kCandidateMicrobatchCapacity;
            batches_[slot].count = 0U;
            batches_[slot].state = CandidateMicrobatchState::kFree;
        }

#if A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM == 1
        // The disabled C8-equivalent path is deliberately a one-word read.
        // In particular it must not validate/dereference the table input,
        // which lets a graph preserve official C8 behaviour under
        // runtime_enable=0 even while P9 tensors are graph placeholders.
        if (candidateScratch == nullptr || config.runtimeEnable == nullptr ||
            constInfo_.sparseCount != kCanonicalResidentSlotCount) {
            return;
        }

        runtimeEnable_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.runtimeEnable));
        // A disabled invocation must be indistinguishable from the official
        // C8 path: it does not read score metadata or enter the V->S seam.
        if (runtimeEnable_.GetValue(0U) != 1) {
            return;
        }

        // An all-padding captured replay has no execution row authorized to
        // touch P9 maps, slots, or queue control.  Read this one graph-visible
        // gate before validating or binding any of those inputs.  This also
        // keeps the standalone hook safe if it is initialized outside the
        // source-derived vector wrapper.
        if (config.activeRowCount == nullptr || config.rowCount == 0U) {
            return;
        }
        activeRowCount_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.activeRowCount));
        const int32_t active_rows = activeRowCount_.GetValue(0U);
        if (active_rows <= 0 || static_cast<uint32_t>(active_rows) >
                                    config.rowCount) {
            return;
        }
        activeRowCountValue_ = static_cast<uint32_t>(active_rows);

        const CanonicalResidentSlotTable table{
            reinterpret_cast<__gm__ int32_t*>(config.canonicalResidentSlots),
            config.poolSize};
        if (config.residentThreshold == nullptr ||
            config.residentThresholdValid == nullptr ||
            config.reqPoolEntries == nullptr || config.kvObjectMap == nullptr ||
            config.kvObjectLens == nullptr ||
            config.persistentObjectSlotsSnapshot == nullptr ||
            config.canonicalResidentSlots == nullptr ||
            config.residentBlockTable == nullptr ||
            config.activeRowCount == nullptr ||
            config.candidateSourceToSlot == nullptr ||
            config.candidateSlotCursor == nullptr ||
            config.rowCount == 0U ||
            config.sourceCapacity == 0U || config.poolSize == 0U ||
            config.kvObjectCapacity == 0U || config.residentBlockCapacity == 0U ||
            !IsCanonicalResidentSlotTableValid(table)) {
            return;
        }
        residentThreshold_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.residentThreshold));
        residentThresholdValid_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.residentThresholdValid));
        reqPoolEntries_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.reqPoolEntries));
        kvObjectMap_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.kvObjectMap));
        kvObjectLens_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.kvObjectLens));
        persistentObjectSlots_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(
                config.persistentObjectSlotsSnapshot));
        candidateSourceToSlot_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.candidateSourceToSlot));
        candidateSlotCursor_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(config.candidateSlotCursor));
        enabled_ = true;
#endif
    }

    __aicore__ inline bool Enabled() const
    {
        return enabled_;
    }

    __aicore__ inline uint32_t DeferredCandidates() const
    {
        return deferredCandidates_;
    }

    // `active_row_count` is the sole graph-padding authority.  The vector
    // service uses this value to prove that it has an active execution row
    // before it initializes the Direct-HCCP executor (and therefore before
    // it reads the queue-control ABI).  Padding rows may alias a real pool
    // row, so req_pool_entries/canonical slots must not be used as a proxy.
    __aicore__ inline uint32_t ActiveRowCount() const
    {
        return activeRowCountValue_;
    }

    __aicore__ inline uint32_t LastTileFirstActiveRow() const
    {
        return lastTileFirstActiveRow_;
    }

    __aicore__ inline void ClearLastTileActiveRow()
    {
        lastTileFirstActiveRow_ = config_.rowCount;
    }

    __aicore__ inline bool HasReadyPostView() const
    {
        return FindReadyBatch() != kInvalidBatch;
    }

    // rows_per_entry belongs to the Direct-HCCP control contract, not score
    // compaction.  It is deliberately supplied only after the vector service
    // has observed an active row; this keeps an all-padding graph replay from
    // touching transport control merely because P9 was compiled in.
    __aicore__ inline void SetRowsPerEntry(uint32_t rowsPerEntry)
    {
#if A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM == 1
        if (enabled_ && rowsPerEntry != 0U) {
            config_.rowsPerEntry = rowsPerEntry;
        }
#else
        (void)rowsPerEntry;
#endif
    }

    // This function is the only score-UB consumer.  It must run under the
    // explicit V->S / S->V lifetime envelope at the exact source-derived
    // seam.  It neither acquires a queue nor writes any HCCP descriptor.
    __aicore__ inline void CompactExactScoreTile(
        const LocalTensor<uint16_t>& outBase,
        const QLICommon::RunInfo& info,
        int64_t curAivS1Idx,
        int64_t curAivS1ProcNum,
        int64_t curS2Idx)
    {
#if A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM == 1
        if (!enabled_) {
            return;
        }
        // Do this before checking/retiring a microbatch.  An all-padding
        // tile then returns without reading a pool/map/table or attempting a
        // transport operation in the caller.
        lastTileFirstActiveRow_ = FindFirstActiveRow(
            info, curAivS1Idx, curAivS1ProcNum);
        if (lastTileFirstActiveRow_ >= config_.rowCount) {
            return;
        }
        const uint32_t batchIndex = FindFreeBatch();
        if (batchIndex == kInvalidBatch) {
            CountDeferredCandidates(outBase, info, curAivS1Idx,
                                    curAivS1ProcNum, curS2Idx);
            return;
        }

        CandidateMicrobatch& batch = batches_[batchIndex];
        uint32_t count = 0U;
        for (int64_t localRow = 0; localRow < curAivS1ProcNum; ++localRow) {
            const uint64_t row = OutputRow(info, curAivS1Idx + localRow);
            if (row >= config_.rowCount) {
                continue;
            }
            const int32_t validInTile = ValidInTile(
                info, curAivS1Idx, localRow, curS2Idx);
            const uint64_t outRowOffset = static_cast<uint64_t>(localRow) *
                (UB_BANK_DEPTH_STRIDE / sizeof(uint16_t));
            for (int32_t inTile = 0; inTile < validInTile; ++inTile) {
                const uint32_t source =
                    static_cast<uint32_t>(curS2Idx + inTile);
                const uint16_t score = outBase.GetValue(outRowOffset + inTile);
                CandidateSeed seed{};
                if (!ResolveColdCandidate(score, static_cast<uint32_t>(row),
                                          source, &seed)) {
                    continue;
                }
                if (count == kCandidateMicrobatchCapacity) {
                    ++deferredCandidates_;
                    continue;
                }
                // AscendC does not permit the implicit C++ copy-assignment
                // operator to target an aggregate in __ubuf__ address space:
                // that operator is generated for local memory.  Write the
                // fixed four-word ABI explicitly instead.  `count` advances
                // only after the whole seed is materialized, so a later
                // hand-off can never observe a partially initialized entry.
                __ubuf__ CandidateSeed* const destination = batch.seeds + count;
                destination->row = seed.row;
                destination->source = seed.source;
                destination->kvObject = seed.kvObject;
                destination->residentSlotOrdinal = seed.residentSlotOrdinal;
                ++count;
            }
        }
        batch.count = count;
        batch.state = count == 0U ? CandidateMicrobatchState::kFree
                                  : CandidateMicrobatchState::kReady;
#else
        (void)outBase;
        (void)info;
        (void)curAivS1Idx;
        (void)curAivS1ProcNum;
        (void)curS2Idx;
#endif
    }

    // Returns one local seed batch to a later, dynamic PostOnly executor. It
    // intentionally says nothing about a queue id: every normal LI worker can
    // submit it if it wins a queue lease, and no worker is reserved for Get.
    __aicore__ inline bool TakeReadyPostView(CandidatePostView* view,
                                             uint32_t* batchToken)
    {
#if A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM == 1
        if (!enabled_ || config_.rowsPerEntry == 0U || view == nullptr ||
            batchToken == nullptr) {
            return false;
        }
        const uint32_t batchIndex = FindReadyBatch();
        if (batchIndex == kInvalidBatch) {
            return false;
        }
        CandidateMicrobatch& batch = batches_[batchIndex];
        batch.state = CandidateMicrobatchState::kHandedToExecutor;
        view->seeds = batch.seeds;
        view->count = batch.count;
        view->producerId = config_.producerId;
        view->candidateSourceToSlot = reinterpret_cast<__gm__ int32_t*>(
            config_.candidateSourceToSlot);
        view->candidateSlotCursor = reinterpret_cast<__gm__ int32_t*>(
            config_.candidateSlotCursor);
        view->reqPoolEntries = reinterpret_cast<__gm__ int32_t*>(
            config_.reqPoolEntries);
        view->canonicalResidentSlots = reinterpret_cast<__gm__ int32_t*>(
            config_.canonicalResidentSlots);
        view->residentBlockTable = reinterpret_cast<__gm__ int32_t*>(
            config_.residentBlockTable);
        view->persistentObjectSlotsSnapshot = reinterpret_cast<__gm__ int32_t*>(
            config_.persistentObjectSlotsSnapshot);
        view->layerResidentGetControl = config_.layerResidentGetControl;
        view->rowCount = config_.rowCount;
        view->poolSize = config_.poolSize;
        view->residentBlockCapacity = config_.residentBlockCapacity;
        view->kvObjectCapacity = config_.kvObjectCapacity;
        view->sourceCapacity = config_.sourceCapacity;
        view->rowsPerEntry = config_.rowsPerEntry;
        view->activeRowCount = activeRowCountValue_;
        *batchToken = batchIndex;
        return true;
#else
        (void)view;
        (void)batchToken;
        return false;
#endif
    }

    // The executor calls this after it has either rolled back a pre-doorbell
    // claim or accepted PostOnly and committed every source->fixed-slot map.
    // It does not wait for CQ: after PostOnly the queue descriptor, not this
    // UB seed array, owns the request.  Releasing a local microbatch has no
    // cache-slot lifecycle meaning; it only lets this worker compact a later
    // score tile while HCCP DMA runs.
    __aicore__ inline void CompletePostView(uint32_t batchToken)
    {
#if A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM == 1
        if (batchToken >= kCandidateMicrobatchCount) {
            return;
        }
        CandidateMicrobatch& batch = batches_[batchToken];
        if (batch.state == CandidateMicrobatchState::kHandedToExecutor) {
            batch.count = 0U;
            batch.state = CandidateMicrobatchState::kFree;
        }
#else
        (void)batchToken;
#endif
    }

    // If the dynamic pool is busy, this drops only a speculative batch.  No
    // resident destination was claimed by this hook, so Stage-B exact recall
    // can later assign all genuine Top-2048 misses without rollback work.
    __aicore__ inline void DeferPostView(uint32_t batchToken)
    {
        CompletePostView(batchToken);
    }

    // Stage B invokes this before it drains the last initial wave and
    // repurposes the official outInvalidBuf_ for its 256-entry exact-recall
    // descriptor.  A ready batch has never reached a queue descriptor, so
    // dropping it is semantically safe: final Exact Top-K recall will cover
    // every genuine winner.  A handed batch is deliberately left alone;
    // only its owning Post executor may make its CQ terminal.
    __aicore__ inline void DeferAllReadyPostViews()
    {
#if A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM == 1
        for (uint32_t slot = 0U; slot < kCandidateMicrobatchCount; ++slot) {
            if (batches_[slot].state == CandidateMicrobatchState::kReady) {
                batches_[slot].count = 0U;
                batches_[slot].state = CandidateMicrobatchState::kFree;
            }
        }
#endif
    }

private:
    static constexpr uint32_t kInvalidBatch = kCandidateMicrobatchCount;

    struct CandidateMicrobatch {
        __ubuf__ CandidateSeed* seeds = nullptr;
        uint32_t count = 0U;
        CandidateMicrobatchState state = CandidateMicrobatchState::kFree;
    };

    __aicore__ inline uint32_t FindFreeBatch() const
    {
        for (uint32_t slot = 0U; slot < kCandidateMicrobatchCount; ++slot) {
            if (batches_[slot].state == CandidateMicrobatchState::kFree) {
                return slot;
            }
        }
        return kInvalidBatch;
    }

    __aicore__ inline uint32_t FindReadyBatch() const
    {
        for (uint32_t slot = 0U; slot < kCandidateMicrobatchCount; ++slot) {
            if (batches_[slot].state == CandidateMicrobatchState::kReady &&
                batches_[slot].count != 0U) {
                return slot;
            }
        }
        return kInvalidBatch;
    }

    __aicore__ inline uint64_t OutputRow(const QLICommon::RunInfo& info,
                                          int64_t s1Index) const
    {
        if (constInfo_.sparseCount == 0U || s1Index < 0) {
            return config_.rowCount;
        }
        const uint64_t row = info.indiceOutOffset / constInfo_.sparseCount +
            static_cast<uint64_t>(s1Index);
        // This comparison happens before any P9 pool/table/map lookup.  A
        // graph-padding row can share `req_pool_entries` with a real request,
        // so a zero candidate length is not a sufficient guard.
        return row < activeRowCountValue_ ? row : config_.rowCount;
    }

    __aicore__ inline uint32_t FindFirstActiveRow(
        const QLICommon::RunInfo& info, int64_t curAivS1Idx,
        int64_t curAivS1ProcNum) const
    {
        for (int64_t localRow = 0; localRow < curAivS1ProcNum; ++localRow) {
            const uint64_t row = OutputRow(info, curAivS1Idx + localRow);
            if (row < config_.rowCount) {
                return static_cast<uint32_t>(row);
            }
        }
        return config_.rowCount;
    }

    __aicore__ inline int32_t ValidInTile(const QLICommon::RunInfo& info,
                                           int64_t curAivS1Idx,
                                           int64_t localRow,
                                           int64_t curS2Idx) const
    {
        int64_t validS2 = info.actS2Size;
        if (constInfo_.attenMaskFlag) {
            const int64_t base = static_cast<int64_t>(info.actS2SizeOrig) -
                static_cast<int64_t>(info.actS1Size) + curAivS1Idx + 1;
            validS2 = (base + localRow) / constInfo_.cmpRatio;
        }
        const int64_t remain = validS2 - curS2Idx;
        const int64_t sourceRemain =
            static_cast<int64_t>(config_.sourceCapacity) - curS2Idx;
        if (remain <= 0 || sourceRemain <= 0) {
            return 0;
        }
        const int64_t tile = remain < info.actualSingleProcessSInnerSize
            ? remain
            : info.actualSingleProcessSInnerSize;
        const int64_t bounded = tile < sourceRemain ? tile : sourceRemain;
        return static_cast<int32_t>(bounded);
    }

    // This is the mathematical score-time predicate only.  It reads the
    // persistent snapshot but does not modify it, does not touch C8 Top-K,
    // and does not reserve a canonical slot.  Reservation is deliberately
    // deferred until a dynamic queue lease has been won by the executor.
    __aicore__ inline bool ResolveColdCandidate(uint16_t score, uint32_t row,
                                                 uint32_t source,
                                                 CandidateSeed* seed) const
    {
        if (seed == nullptr || row >= activeRowCountValue_ ||
            source >= config_.sourceCapacity ||
            residentThresholdValid_.GetValue(row) != 1) {
            return false;
        }
        const uint16_t threshold = static_cast<uint16_t>(
            static_cast<uint32_t>(residentThreshold_.GetValue(row)) & 0xffffU);
        if (score < threshold) {
            return false;
        }
        const int32_t poolRow = reqPoolEntries_.GetValue(row);
        if (poolRow < 0 || static_cast<uint32_t>(poolRow) >= config_.poolSize) {
            return false;
        }
        const int32_t object = kvObjectMap_.GetValue(
            static_cast<uint64_t>(poolRow) * config_.sourceCapacity + source);
        const int32_t objectLen = kvObjectLens_.GetValue(row);
        if (object < 0 || objectLen <= 0 ||
            static_cast<uint32_t>(object) >= config_.kvObjectCapacity ||
            static_cast<uint32_t>(object) >= static_cast<uint32_t>(objectLen)) {
            return false;
        }
        if (persistentObjectSlots_.GetValue(
                static_cast<uint64_t>(poolRow) * config_.kvObjectCapacity +
                static_cast<uint32_t>(object)) >= 0) {
            return false;
        }
        // A terminally committed candidate needs no second speculative post.
        // A currently claimed value is also skipped; the executor resolves it
        // before Stage-B exact recall starts.
        const uint64_t sourceOffset =
            static_cast<uint64_t>(row) * config_.sourceCapacity + source;
        if (candidateSourceToSlot_.GetValue(sourceOffset) !=
            kCandidateSourceUnmapped) {
            return false;
        }
        seed->row = static_cast<int32_t>(row);
        seed->source = static_cast<int32_t>(source);
        seed->kvObject = object;
        seed->residentSlotOrdinal = kCandidateResidentSlotUnassigned;
        return true;
    }

    __aicore__ inline void CountDeferredCandidates(
        const LocalTensor<uint16_t>& outBase, const QLICommon::RunInfo& info,
        int64_t curAivS1Idx, int64_t curAivS1ProcNum, int64_t curS2Idx)
    {
        for (int64_t localRow = 0; localRow < curAivS1ProcNum; ++localRow) {
            const uint64_t row = OutputRow(info, curAivS1Idx + localRow);
            if (row >= config_.rowCount) {
                continue;
            }
            const int32_t validInTile = ValidInTile(
                info, curAivS1Idx, localRow, curS2Idx);
            const uint64_t outRowOffset = static_cast<uint64_t>(localRow) *
                (UB_BANK_DEPTH_STRIDE / sizeof(uint16_t));
            for (int32_t inTile = 0; inTile < validInTile; ++inTile) {
                CandidateSeed ignored{};
                if (ResolveColdCandidate(
                        outBase.GetValue(outRowOffset + inTile),
                        static_cast<uint32_t>(row),
                        static_cast<uint32_t>(curS2Idx + inTile), &ignored)) {
                    ++deferredCandidates_;
                }
            }
        }
    }

    C8ExactSpeculativeResidentScoreHookConfig config_{};
    QLICommon::ConstInfo constInfo_{};
    CandidateMicrobatch batches_[kCandidateMicrobatchCount]{};
    GlobalTensor<int32_t> runtimeEnable_;
    GlobalTensor<int32_t> residentThreshold_;
    GlobalTensor<int32_t> residentThresholdValid_;
    GlobalTensor<int32_t> reqPoolEntries_;
    GlobalTensor<int32_t> kvObjectMap_;
    GlobalTensor<int32_t> kvObjectLens_;
    GlobalTensor<int32_t> persistentObjectSlots_;
    GlobalTensor<int32_t> candidateSourceToSlot_;
    GlobalTensor<int32_t> candidateSlotCursor_;
    GlobalTensor<int32_t> activeRowCount_;
    bool enabled_ = false;
    uint32_t deferredCandidates_ = 0U;
    uint32_t activeRowCountValue_ = 0U;
    uint32_t lastTileFirstActiveRow_ = 0U;
};

}  // namespace exact_speculative_resident
}  // namespace QLIKernel

#endif  // VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_HOOK_H_
