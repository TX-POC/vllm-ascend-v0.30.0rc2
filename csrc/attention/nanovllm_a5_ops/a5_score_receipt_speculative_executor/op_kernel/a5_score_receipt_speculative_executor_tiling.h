// P7.1 standalone CandidateSeed -> KVAF Post/CQ executor tiling.
//
// This is deliberately independent of C8.  The C8 publisher remains an
// immutable producer of score receipts/Top-K; P7.1 consumes only compacted
// CandidateSeed records through graph-visible GM tensors.

#ifndef A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_TILING_H
#define A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_TILING_H

#include <cstdint>

// P7.1 intentionally consumes an already-published CandidateSeed GM tensor.
// There is no C8 tile/UB layout in this contract.  The fixed capacity belongs
// to the executor's own two local seed batches; it is not a permanent core
// assignment and it is not a transport queue count.
constexpr uint32_t kA5ScoreReceiptExecutorSeedWords = 4U;
constexpr uint32_t kA5ScoreReceiptExecutorReceiptCapacity = 128U;
constexpr uint32_t kA5ScoreReceiptExecutorMicrobatchCapacity = 64U;
constexpr uint32_t kA5ScoreReceiptExecutorMicrobatchSlots = 2U;
constexpr uint32_t kA5ScoreReceiptExecutorWorkerStatColumns = 16U;

struct A5ScoreReceiptSpeculativeExecutorTilingData {
    uint32_t workerCount;
    uint32_t receiptCount;
    uint32_t rowCount;
    uint32_t sourceCapacity;
    uint32_t poolSize;
    uint32_t kvObjectCapacity;
    uint32_t queueCount;
    uint32_t privateArenaCapacity;
    uint32_t privatePackedRows;
    uint32_t workerStatsRows;
    uint32_t workerStatsColumns;
    uint32_t candidateSeedWords;
    uint32_t candidateCapacity;
    uint32_t microbatchCapacity;
    uint32_t microbatchSlots;
    uint32_t reserved;
};

#endif  // A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_TILING_H
