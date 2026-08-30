/**
 * P7 stage-1 tiling contract: a symmetric AIV pool consumes published C8
 * score receipts and emits compact CandidateSeed records.  It intentionally
 * contains no KVAF/HCCP state and no C8 implementation state.
 */

#ifndef A5_SCORE_RECEIPT_CANDIDATE_EXECUTOR_TILING_H
#define A5_SCORE_RECEIPT_CANDIDATE_EXECUTOR_TILING_H

#include <cstdint>

struct A5ScoreReceiptCandidateExecutorTilingData {
    uint32_t workerCount;
    uint32_t rowCount;
    uint32_t sourceCapacity;
    uint32_t poolSize;
    uint32_t kvObjectCapacity;
    uint32_t tilesPerRow;
    uint32_t tileReceiptCount;
    uint32_t scoreWorkspaceWords;
};

#endif  // A5_SCORE_RECEIPT_CANDIDATE_EXECUTOR_TILING_H
