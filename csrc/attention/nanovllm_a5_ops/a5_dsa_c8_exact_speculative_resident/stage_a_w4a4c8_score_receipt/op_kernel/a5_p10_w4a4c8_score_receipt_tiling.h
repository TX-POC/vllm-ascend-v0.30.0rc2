#ifndef A5_P10_W4A4C8_SCORE_RECEIPT_TILING_H
#define A5_P10_W4A4C8_SCORE_RECEIPT_TILING_H

#include <cstdint>

// Static geometry retained from the direct W4A4C8 QuantLiPhase closure.
// `scoreWorkspaceStride` is kept solely because the imported QLI layout names
// that native stride.  P10 does not allocate an opaque score workspace: the
// exact same per-batch layout is rebound to graph-owned score_surface.
struct A5P10W4A4C8ScoreReceiptTilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t sourceCapacity;
    uint32_t indexHeads;
    uint32_t maxBlockNumPerBatch;
    uint32_t maxCandidateLen;
    uint32_t keyStride;
    uint32_t scaleStride;
    uint32_t scoreWorkspaceStride;
    uint32_t scoreSurfaceSliceStrideBytes;
    uint32_t scoreSurfaceRowStrideBytes;
    uint32_t receiptTilesPerRow;
    uint32_t receiptWords64;
    uint32_t receiptCapacity;
};

#endif  // A5_P10_W4A4C8_SCORE_RECEIPT_TILING_H
