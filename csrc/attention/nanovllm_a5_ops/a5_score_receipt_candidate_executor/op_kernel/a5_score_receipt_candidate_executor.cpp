/**
 * P7 stage-1 device executor: published C8 score receipts -> CandidateSeed.
 *
 * This operator is intentionally a post-publisher consumer.  It must be
 * ordered after the score publisher on a stream/event with an established
 * device-memory visibility edge.  It does not poll a concurrently-running
 * C8 kernel, does not reserve AIVs for Get, and contains no Direct-HCCP/CQ
 * code.  That keeps this first executor stage both mathematically inert to
 * C8 and free of the invalid 31+1 static partition.
 */

#include "kernel_operator.h"

#include "a5_score_receipt_candidate_executor_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kScoreTileWidth = 128U;
constexpr uint32_t kReceiptWords = 16U;
constexpr uint32_t kReceiptContractVersion = 1U;
constexpr uint32_t kReceiptByteOffsetWord = 0U;
constexpr uint32_t kReceiptRowWord = 1U;
constexpr uint32_t kReceiptSourceBeginWord = 2U;
constexpr uint32_t kReceiptValidCountWord = 3U;
constexpr uint32_t kReceiptScoreStrideWord = 4U;
constexpr uint32_t kReceiptVersionWord = 7U;
constexpr uint32_t kReceiptReadyEpochWord = 8U;
constexpr uint32_t kCandidateSeedWords = 4U;
constexpr uint32_t kCandidateReceiptIdentityWords = 4U;

// Receipt status is deliberately terminal telemetry, not a retry protocol.
// Later Post/CQ stages will own retry/lease state.
constexpr int32_t kStatusOk = 1;
constexpr int32_t kStatusInactive = 2;
constexpr int32_t kStatusEpochMismatch = -1;
constexpr int32_t kStatusReceiptMalformed = -2;
constexpr int32_t kStatusRowMetadataInvalid = -3;
constexpr int32_t kStatusIdentityMalformed = -4;

// A receipt must never be identified by forward epoch alone: a graph replay
// or another layer can reuse the same epoch-visible storage.  Generation is
// the release word.  Writers publish it last; readers sample it twice.
struct CandidateReceiptIdentity {
    int64_t forward_epoch;
    int64_t graph_replay_generation;
    int64_t layer_control_index;
    int64_t receipt_generation;
};

class A5ScoreReceiptCandidateExecutorKernel {
public:
    __aicore__ inline explicit A5ScoreReceiptCandidateExecutorKernel(
        const A5ScoreReceiptCandidateExecutorTilingData* tiling)
        : tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR scoreWorkspace, GM_ADDR tileReceipts, GM_ADDR receiptIdentity,
        GM_ADDR scoreSourceLens, GM_ADDR residentThreshold,
        GM_ADDR residentThresholdValid, GM_ADDR kvObjectSlotsSnapshot,
        GM_ADDR reqPoolEntries, GM_ADDR kvObjectMap, GM_ADDR kvObjectLens,
        GM_ADDR candidateSeeds, GM_ADDR candidateCounts,
        GM_ADDR candidateReceiptIdentity, GM_ADDR receiptStatus,
        GM_ADDR receiptOwner)
    {
        scoreWorkspace_ = reinterpret_cast<__gm__ uint16_t*>(scoreWorkspace);
        tileReceipts_ = reinterpret_cast<__gm__ int64_t*>(tileReceipts);
        receiptIdentity_ = reinterpret_cast<__gm__ int64_t*>(receiptIdentity);
        scoreSourceLens_ = reinterpret_cast<__gm__ int32_t*>(scoreSourceLens);
        residentThreshold_ = reinterpret_cast<__gm__ int32_t*>(residentThreshold);
        residentThresholdValid_ =
            reinterpret_cast<__gm__ int32_t*>(residentThresholdValid);
        kvObjectSlotsSnapshot_ =
            reinterpret_cast<__gm__ int32_t*>(kvObjectSlotsSnapshot);
        reqPoolEntries_ = reinterpret_cast<__gm__ int32_t*>(reqPoolEntries);
        kvObjectMap_ = reinterpret_cast<__gm__ int32_t*>(kvObjectMap);
        kvObjectLens_ = reinterpret_cast<__gm__ int32_t*>(kvObjectLens);
        candidateSeeds_ = reinterpret_cast<__gm__ int32_t*>(candidateSeeds);
        candidateCounts_ = reinterpret_cast<__gm__ int32_t*>(candidateCounts);
        candidateReceiptIdentity_ =
            reinterpret_cast<__gm__ int64_t*>(candidateReceiptIdentity);
        receiptStatus_ = reinterpret_cast<__gm__ int32_t*>(receiptStatus);
        receiptOwner_ = reinterpret_cast<__gm__ int32_t*>(receiptOwner);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const uint32_t worker = GetBlockIdx();
            if (worker >= tiling_->workerCount) {
                return;
            }
            for (uint32_t receipt = worker; receipt < tiling_->tileReceiptCount;
                 receipt += tiling_->workerCount) {
                ProcessReceipt(receipt, worker);
            }
        }
    }

private:
    __aicore__ inline uint64_t ReceiptWord(uint32_t receipt,
                                            uint32_t word) const
    {
        return static_cast<uint64_t>(ReadGmByPassDCache<int64_t>(
            tileReceipts_ + static_cast<uint64_t>(receipt) * kReceiptWords +
            word));
    }

    __aicore__ inline __gm__ int32_t* Seed(uint32_t receipt,
                                            uint32_t index) const
    {
        return candidateSeeds_ +
               (static_cast<uint64_t>(receipt) * kScoreTileWidth + index) *
                   kCandidateSeedWords;
    }

    __aicore__ inline __gm__ int64_t* InputIdentity(uint32_t receipt) const
    {
        return receiptIdentity_ +
            static_cast<uint64_t>(receipt) * kCandidateReceiptIdentityWords;
    }

    __aicore__ inline __gm__ int64_t* OutputIdentity(uint32_t receipt) const
    {
        return candidateReceiptIdentity_ +
            static_cast<uint64_t>(receipt) * kCandidateReceiptIdentityWords;
    }

    __aicore__ inline bool LoadStableInputIdentity(
        uint32_t receipt, CandidateReceiptIdentity* identity) const
    {
        if (identity == nullptr) {
            return false;
        }
        __gm__ int64_t* const source = InputIdentity(receipt);
        const int64_t generation_before =
            ReadGmByPassDCache<int64_t>(source + 3U);
        if (generation_before <= 0) {
            return false;
        }
        identity->forward_epoch = ReadGmByPassDCache<int64_t>(source + 0U);
        identity->graph_replay_generation =
            ReadGmByPassDCache<int64_t>(source + 1U);
        identity->layer_control_index =
            ReadGmByPassDCache<int64_t>(source + 2U);
        const int64_t generation_after =
            ReadGmByPassDCache<int64_t>(source + 3U);
        identity->receipt_generation = generation_after;
        return generation_before == generation_after && generation_after > 0 &&
            identity->forward_epoch > 0 && identity->layer_control_index >= 0;
    }

    __aicore__ inline void TombstoneOutputIdentity(uint32_t receipt) const
    {
        // Invalidate first.  A concurrent downstream reader therefore cannot
        // accept an old release word while terminal/replay cleanup rewrites
        // the other three fields.
        __gm__ int64_t* const output = OutputIdentity(receipt);
        WriteGmByPassDCache<int64_t>(output + 3U, 0);
        DataSyncBarrier<MemDsbT::DDR>();
        WriteGmByPassDCache<int64_t>(output + 0U, 0);
        WriteGmByPassDCache<int64_t>(output + 1U, 0);
        WriteGmByPassDCache<int64_t>(output + 2U, 0);
        DataSyncBarrier<MemDsbT::DDR>();
    }

    __aicore__ inline void PublishOutputIdentity(
        uint32_t receipt, const CandidateReceiptIdentity& identity) const
    {
        __gm__ int64_t* const output = OutputIdentity(receipt);
        WriteGmByPassDCache<int64_t>(output + 0U, identity.forward_epoch);
        WriteGmByPassDCache<int64_t>(output + 1U,
                                     identity.graph_replay_generation);
        WriteGmByPassDCache<int64_t>(output + 2U,
                                     identity.layer_control_index);
        // All payload/status fields and identity prefix must be globally
        // visible before the nonzero generation makes this receipt readable.
        DataSyncBarrier<MemDsbT::DDR>();
        WriteGmByPassDCache<int64_t>(output + 3U,
                                     identity.receipt_generation);
        DataSyncBarrier<MemDsbT::DDR>();
    }

    __aicore__ inline void ClearSeeds(uint32_t receipt) const
    {
        for (uint32_t index = 0U; index < kScoreTileWidth; ++index) {
            __gm__ int32_t* const seed = Seed(receipt, index);
            WriteGmByPassDCache<int32_t>(seed + 0U, -1);
            WriteGmByPassDCache<int32_t>(seed + 1U, -1);
            WriteGmByPassDCache<int32_t>(seed + 2U, -1);
            WriteGmByPassDCache<int32_t>(seed + 3U, -1);
        }
    }

    __aicore__ inline void Complete(uint32_t receipt, uint32_t owner,
                                    const CandidateReceiptIdentity* identity,
                                    uint32_t count, int32_t status) const
    {
        WriteGmByPassDCache<int32_t>(candidateCounts_ + receipt,
                                     static_cast<int32_t>(count));
        WriteGmByPassDCache<int32_t>(receiptStatus_ + receipt, status);
        WriteGmByPassDCache<int32_t>(receiptOwner_ + receipt,
                                     static_cast<int32_t>(owner));
        if (identity != nullptr) {
            PublishOutputIdentity(receipt, *identity);
        }
    }

    __aicore__ inline void ProcessReceipt(uint32_t receipt,
                                          uint32_t owner) const
    {
        // The output generation is the Stage-A -> Stage-B release word.  It
        // must be tombstoned (and made DDR-visible) before *any* seed/count/
        // status/owner mutation: otherwise a P8 executor can accept the old
        // generation then observe a partially cleared receipt.  Completion
        // below remains the only generation-last publication edge.
        TombstoneOutputIdentity(receipt);
        ClearSeeds(receipt);
        CandidateReceiptIdentity identity{};
        if (!LoadStableInputIdentity(receipt, &identity)) {
            Complete(receipt, owner, nullptr, 0U, kStatusIdentityMalformed);
            return;
        }

        const uint32_t expectedRow = receipt / tiling_->tilesPerRow;
        const uint32_t expectedSource =
            (receipt % tiling_->tilesPerRow) * kScoreTileWidth;
        if (expectedRow >= tiling_->rowCount) {
            Complete(receipt, owner, &identity, 0U, kStatusReceiptMalformed);
            return;
        }
        const int32_t sourceLen =
            ReadGmByPassDCache<int32_t>(scoreSourceLens_ + expectedRow);
        if (sourceLen < 0 || static_cast<uint32_t>(sourceLen) >
                                 tiling_->sourceCapacity) {
            Complete(receipt, owner, &identity, 0U, kStatusRowMetadataInvalid);
            return;
        }
        if (expectedSource >= static_cast<uint32_t>(sourceLen)) {
            Complete(receipt, owner, &identity, 0U, kStatusInactive);
            return;
        }

        if (ReceiptWord(receipt, kReceiptReadyEpochWord) !=
            static_cast<uint64_t>(identity.forward_epoch)) {
            Complete(receipt, owner, &identity, 0U, kStatusEpochMismatch);
            return;
        }
        DataSyncBarrier<MemDsbT::DDR>();
        const uint64_t byteOffset = ReceiptWord(receipt, kReceiptByteOffsetWord);
        const uint64_t row = ReceiptWord(receipt, kReceiptRowWord);
        const uint64_t sourceBegin = ReceiptWord(receipt, kReceiptSourceBeginWord);
        const uint64_t validCount = ReceiptWord(receipt, kReceiptValidCountWord);
        const uint64_t scoreStride = ReceiptWord(receipt, kReceiptScoreStrideWord);
        const uint64_t version = ReceiptWord(receipt, kReceiptVersionWord);
        const uint64_t expectedCount =
            (static_cast<uint32_t>(sourceLen) - expectedSource) <
                    kScoreTileWidth
                ? static_cast<uint32_t>(sourceLen) - expectedSource
                : kScoreTileWidth;
        const uint64_t workspaceScores =
            static_cast<uint64_t>(tiling_->scoreWorkspaceWords) * 2U;
        if (version != kReceiptContractVersion || row != expectedRow ||
            sourceBegin != expectedSource || validCount != expectedCount ||
            scoreStride < tiling_->sourceCapacity ||
            (byteOffset & (sizeof(uint16_t) - 1U)) != 0U ||
            byteOffset / sizeof(uint16_t) > workspaceScores ||
            validCount > workspaceScores - byteOffset / sizeof(uint16_t)) {
            Complete(receipt, owner, &identity, 0U, kStatusReceiptMalformed);
            return;
        }

        const int32_t thresholdValid =
            ReadGmByPassDCache<int32_t>(residentThresholdValid_ + expectedRow);
        if (thresholdValid == 0) {
            Complete(receipt, owner, &identity, 0U, kStatusOk);
            return;
        }
        const int32_t pool =
            ReadGmByPassDCache<int32_t>(reqPoolEntries_ + expectedRow);
        const int32_t objectLen =
            ReadGmByPassDCache<int32_t>(kvObjectLens_ + expectedRow);
        if (pool < 0 || static_cast<uint32_t>(pool) >= tiling_->poolSize ||
            objectLen < 0 || static_cast<uint32_t>(objectLen) >
                                 tiling_->kvObjectCapacity) {
            Complete(receipt, owner, &identity, 0U, kStatusRowMetadataInvalid);
            return;
        }
        const uint16_t threshold = static_cast<uint16_t>(
            ReadGmByPassDCache<int32_t>(residentThreshold_ + expectedRow));
        const uint64_t scoreOffset = byteOffset / sizeof(uint16_t);
        uint32_t count = 0U;
        for (uint32_t local = 0U; local < static_cast<uint32_t>(validCount);
             ++local) {
            const uint16_t score =
                ReadGmByPassDCache<uint16_t>(scoreWorkspace_ + scoreOffset + local);
            if (score < threshold) {
                continue;
            }
            const uint32_t source = expectedSource + local;
            const int32_t object = ReadGmByPassDCache<int32_t>(
                kvObjectMap_ + static_cast<uint64_t>(pool) *
                                    tiling_->sourceCapacity + source);
            if (object < 0 || static_cast<uint32_t>(object) >=
                                  static_cast<uint32_t>(objectLen) ||
                static_cast<uint32_t>(object) >= tiling_->kvObjectCapacity) {
                continue;
            }
            if (ReadGmByPassDCache<int32_t>(
                    kvObjectSlotsSnapshot_ + static_cast<uint64_t>(pool) *
                                                  tiling_->kvObjectCapacity +
                                              static_cast<uint32_t>(object)) >= 0) {
                continue;
            }
            __gm__ int32_t* const seed = Seed(receipt, count);
            WriteGmByPassDCache<int32_t>(seed + 0U,
                                         static_cast<int32_t>(expectedRow));
            WriteGmByPassDCache<int32_t>(seed + 1U,
                                         static_cast<int32_t>(source));
            WriteGmByPassDCache<int32_t>(seed + 2U, object);
            WriteGmByPassDCache<int32_t>(seed + 3U,
                                         static_cast<int32_t>(score));
            ++count;
        }
        Complete(receipt, owner, &identity, count, kStatusOk);
    }

    const A5ScoreReceiptCandidateExecutorTilingData* tiling_;
    __gm__ uint16_t* scoreWorkspace_;
    __gm__ int64_t* tileReceipts_;
    __gm__ int64_t* receiptIdentity_;
    __gm__ int32_t* scoreSourceLens_;
    __gm__ int32_t* residentThreshold_;
    __gm__ int32_t* residentThresholdValid_;
    __gm__ int32_t* kvObjectSlotsSnapshot_;
    __gm__ int32_t* reqPoolEntries_;
    __gm__ int32_t* kvObjectMap_;
    __gm__ int32_t* kvObjectLens_;
    __gm__ int32_t* candidateSeeds_;
    __gm__ int32_t* candidateCounts_;
    __gm__ int64_t* candidateReceiptIdentity_;
    __gm__ int32_t* receiptStatus_;
    __gm__ int32_t* receiptOwner_;
};
}  // namespace

extern "C" __global__ __aicore__ void a5_score_receipt_candidate_executor(
    GM_ADDR scoreWorkspace, GM_ADDR tileReceipts, GM_ADDR receiptIdentity,
    GM_ADDR scoreSourceLens, GM_ADDR residentThreshold,
    GM_ADDR residentThresholdValid, GM_ADDR kvObjectSlotsSnapshot,
    GM_ADDR reqPoolEntries, GM_ADDR kvObjectMap, GM_ADDR kvObjectLens,
    GM_ADDR candidateSeeds, GM_ADDR candidateCounts,
    GM_ADDR candidateReceiptIdentity, GM_ADDR receiptStatus, GM_ADDR receiptOwner,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5ScoreReceiptCandidateExecutorTilingData);
    GET_TILING_DATA(tilingData, tiling);
    A5ScoreReceiptCandidateExecutorKernel op(&tilingData);
    op.Init(scoreWorkspace, tileReceipts, receiptIdentity, scoreSourceLens,
            residentThreshold, residentThresholdValid, kvObjectSlotsSnapshot,
            reqPoolEntries, kvObjectMap, kvObjectLens, candidateSeeds,
            candidateCounts, candidateReceiptIdentity, receiptStatus, receiptOwner);
    op.Process();
}
