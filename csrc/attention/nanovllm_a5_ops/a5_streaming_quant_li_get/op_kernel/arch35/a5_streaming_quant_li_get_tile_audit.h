// Compile-time audit-only score-tile observer for A5StreamingQuantLiGet.
//
// This header is intentionally separate from a5_streaming_quant_li_get_tile_ready.h.
// The audit OPP must prove the score/threshold/mapping decision at the exact
// LI seam while having no Direct-HCCP, SIMT, descriptor, CQ, or private-payload
// closure at all.  Keeping this small observer in a distinct translation-unit
// closure makes that property mechanically checkable in the generated OPP.

#ifndef A5_STREAMING_QUANT_LI_GET_TILE_AUDIT_H
#define A5_STREAMING_QUANT_LI_GET_TILE_AUDIT_H

#include "kernel_operator.h"
#include "quant_lightning_indexer_common.h"
#include "../a5_streaming_quant_li_get_hook_mode.h"

#if A5_STREAMING_LI_GET_INLINE_MODE == 1

namespace QLIKernel {
namespace streaming_li_get {
using namespace AscendC;
using namespace QLICommon;

constexpr int32_t kAuditSourceUnmapped = -1;
// P4 installs this invalid transport owner into every lease cell before the
// single full C8 scan.  No valid runtime queue state can accidentally enable
// this observer.
constexpr int32_t kAuditScoreReceiptQueueLeaseSentinel = -2147483647;
constexpr uint32_t kAuditScoreReceiptTag = 0x80000000U;
constexpr uint32_t kAuditScoreReceiptScoreMask = 0xFFFFU;
constexpr uint32_t kAuditScoreReceiptCandidateBit = 1U << 20;
constexpr uint32_t kAuditScoreReceiptMappingValidBit = 1U << 19;
constexpr uint32_t kAuditScoreReceiptColdBit = 1U << 18;
constexpr uint32_t kAuditScoreReceiptThresholdPassBit = 1U << 17;
constexpr uint32_t kAuditScoreReceiptSourceUnmappedBit = 1U << 16;

struct AuditScoreDecision {
    uint32_t thresholdPass;
    uint32_t mappingValid;
    uint32_t cold;
    uint32_t sourceUnmapped;
    uint32_t candidate;
};

// This observer consumes the exact uint16 tile produced by the official C8
// reduction.  It may write only the fixed [row, source] diagnostic receipt.
// In particular, its ABI deliberately excludes queue-control, descriptor,
// private-ready, private-arena, payload, telemetry and SIMT hand-off tensors.
class StreamingLiTileAuditHook {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t *residentThreshold,
        __gm__ uint8_t *residentThresholdValid,
        __gm__ uint8_t *kvObjectSlotsSnapshot,
        __gm__ uint8_t *reqPoolEntries,
        __gm__ uint8_t *kvObjectLens,
        __gm__ uint8_t *kvObjectMap,
        __gm__ uint8_t *privateSourceToSlot,
        __gm__ uint8_t *queueLeases,
        const QLICommon::ConstInfo &constInfo,
        uint32_t rowCount,
        uint32_t sourceCapacity,
        uint32_t poolSize,
        uint32_t kvObjectCapacity,
        uint32_t queueCount)
    {
        residentThreshold_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(residentThreshold));
        residentThresholdValid_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(residentThresholdValid));
        kvObjectSlots_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(kvObjectSlotsSnapshot));
        reqPoolEntries_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(reqPoolEntries));
        kvObjectLens_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(kvObjectLens));
        kvObjectMap_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(kvObjectMap));
        sourceToSlot_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(privateSourceToSlot));
        queueLeases_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(queueLeases));
        constInfo_ = constInfo;
        rowCount_ = rowCount;
        sourceCapacity_ = sourceCapacity;
        poolSize_ = poolSize;
        kvObjectCapacity_ = kvObjectCapacity;
        queueCount_ = queueCount;
        enabled_ = false;
        if (residentThreshold == nullptr || residentThresholdValid == nullptr ||
            kvObjectSlotsSnapshot == nullptr || reqPoolEntries == nullptr ||
            kvObjectLens == nullptr || kvObjectMap == nullptr ||
            privateSourceToSlot == nullptr || queueLeases == nullptr ||
            rowCount_ == 0U || sourceCapacity_ == 0U || poolSize_ == 0U ||
            kvObjectCapacity_ == 0U || queueCount_ == 0U) {
            return;
        }
        enabled_ = HasAuditReceiptLeaseTable();
    }

    __aicore__ inline bool MayNeedScoreObservation() const
    {
        // A successful audit must inspect every valid score coordinate; it
        // intentionally has no queue-capacity shortcut.
        return enabled_;
    }

    __aicore__ inline void Emit(const LocalTensor<uint16_t> &outBase,
                                const QLICommon::RunInfo &info,
                                int64_t curAivS1Idx,
                                int64_t curAivS1ProcNum,
                                int64_t curS2Idx)
    {
        if (!enabled_) {
            return;
        }
        for (int64_t localRow = 0; localRow < curAivS1ProcNum; ++localRow) {
            const uint64_t row = OutputRow(info, curAivS1Idx + localRow);
            if (row >= rowCount_) {
                continue;
            }
            const int32_t validInTile = ValidInTile(
                info, curAivS1Idx, localRow, curS2Idx);
            const uint64_t outRowOffset = static_cast<uint64_t>(localRow) *
                (UB_BANK_DEPTH_STRIDE / sizeof(uint16_t));
            for (int32_t inTile = 0; inTile < validInTile; ++inTile) {
                const uint16_t score = outBase.GetValue(outRowOffset + inTile);
                const uint32_t sourceIndex =
                    static_cast<uint32_t>(curS2Idx + inTile);
                const AuditScoreDecision decision =
                    Evaluate(score, row, sourceIndex);
                uint32_t receipt = kAuditScoreReceiptTag |
                    (static_cast<uint32_t>(score) & kAuditScoreReceiptScoreMask);
                if (decision.thresholdPass != 0U) {
                    receipt |= kAuditScoreReceiptThresholdPassBit;
                }
                if (decision.sourceUnmapped != 0U) {
                    receipt |= kAuditScoreReceiptSourceUnmappedBit;
                }
                if (decision.cold != 0U) {
                    receipt |= kAuditScoreReceiptColdBit;
                }
                if (decision.mappingValid != 0U) {
                    receipt |= kAuditScoreReceiptMappingValidBit;
                }
                if (decision.candidate != 0U) {
                    receipt |= kAuditScoreReceiptCandidateBit;
                }
                const uint64_t sourceOffset = row * sourceCapacity_ + sourceIndex;
                // `receipt` always has bit 31 set.  Materialize its intended
                // two's-complement representation without an implementation-
                // defined uint32_t -> int32_t conversion.
                sourceToSlot_.SetValue(sourceOffset, static_cast<int32_t>(
                    static_cast<int64_t>(receipt) - (1LL << 32)));
            }
        }
    }

private:
    __aicore__ inline bool HasAuditReceiptLeaseTable()
    {
        for (uint32_t queueId = 0U; queueId < queueCount_; ++queueId) {
            if (queueLeases_.GetValue(queueId) !=
                kAuditScoreReceiptQueueLeaseSentinel) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline uint64_t OutputRow(const QLICommon::RunInfo &info,
                                          int64_t s1Index) const
    {
        return info.indiceOutOffset / constInfo_.sparseCount +
               static_cast<uint64_t>(s1Index);
    }

    __aicore__ inline int32_t ValidInTile(const QLICommon::RunInfo &info,
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
        if (remain <= 0) {
            return 0;
        }
        const int64_t sourceRemain =
            static_cast<int64_t>(sourceCapacity_) - curS2Idx;
        if (sourceRemain <= 0) {
            return 0;
        }
        const int64_t tile = remain < info.actualSingleProcessSInnerSize ?
            remain : info.actualSingleProcessSInnerSize;
        return static_cast<int32_t>(tile > sourceRemain ? sourceRemain : tile);
    }

    __aicore__ inline AuditScoreDecision Evaluate(uint16_t score,
                                                   uint64_t row,
                                                   uint32_t sourceIndex) const
    {
        AuditScoreDecision decision{0U, 0U, 0U, 0U, 0U};
        const uint16_t threshold = static_cast<uint16_t>(
            static_cast<uint32_t>(residentThreshold_.GetValue(row)) &
            kAuditScoreReceiptScoreMask);
        decision.thresholdPass =
            residentThresholdValid_.GetValue(row) == 1 && score >= threshold;
        const int32_t poolRow = reqPoolEntries_.GetValue(row);
        const int32_t kvObjectLen = kvObjectLens_.GetValue(row);
        if (poolRow < 0 || static_cast<uint32_t>(poolRow) >= poolSize_ ||
            kvObjectLen <= 0 ||
            static_cast<uint32_t>(kvObjectLen) > kvObjectCapacity_) {
            return decision;
        }
        const int32_t kvObject = kvObjectMap_.GetValue(
            static_cast<uint64_t>(poolRow) * sourceCapacity_ + sourceIndex);
        if (kvObject < 0 ||
            static_cast<uint32_t>(kvObject) >=
                static_cast<uint32_t>(kvObjectLen)) {
            return decision;
        }
        decision.mappingValid = 1U;
        decision.cold = kvObjectSlots_.GetValue(
            static_cast<uint64_t>(poolRow) * kvObjectCapacity_ +
            static_cast<uint32_t>(kvObject)) < 0;
        const uint64_t sourceOffset = row * sourceCapacity_ + sourceIndex;
        decision.sourceUnmapped =
            sourceToSlot_.GetValue(sourceOffset) == kAuditSourceUnmapped;
        decision.candidate = decision.thresholdPass != 0U &&
            decision.cold != 0U && decision.sourceUnmapped != 0U;
        return decision;
    }

    GlobalTensor<int32_t> residentThreshold_;
    GlobalTensor<int32_t> residentThresholdValid_;
    GlobalTensor<int32_t> kvObjectSlots_;
    GlobalTensor<int32_t> reqPoolEntries_;
    GlobalTensor<int32_t> kvObjectLens_;
    GlobalTensor<int32_t> kvObjectMap_;
    GlobalTensor<int32_t> sourceToSlot_;
    GlobalTensor<int32_t> queueLeases_;
    QLICommon::ConstInfo constInfo_{};
    uint32_t rowCount_ = 0U;
    uint32_t sourceCapacity_ = 0U;
    uint32_t poolSize_ = 0U;
    uint32_t kvObjectCapacity_ = 0U;
    uint32_t queueCount_ = 0U;
    bool enabled_ = false;
};

}  // namespace streaming_li_get
}  // namespace QLIKernel

#endif  // A5_STREAMING_LI_GET_INLINE_MODE == 1

#endif  // A5_STREAMING_QUANT_LI_GET_TILE_AUDIT_H
