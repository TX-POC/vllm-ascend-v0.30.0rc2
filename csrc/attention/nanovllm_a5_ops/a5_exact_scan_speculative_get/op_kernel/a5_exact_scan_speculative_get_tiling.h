// Fixed geometry for the device-resident ExactScoreScan + SpeculativeGet
// executor.
//
// The producer side no longer owns a synthetic floating-point LI score tensor.
// It consumes the exact sortable uint16 score bytes published by the source
// derived C8 operator.  `score_workspace` is an int32 graph carrier, but each
// carrier word contains two uint16 C8 scores; `tile_receipts` supplies the
// byte address, row, source range and ready epoch for every C8 score tile.
//
// The host tiler must keep this struct graph-static.  In particular it must
// derive `tileReceiptCount == rowCount * ceil(sourceCapacity / 128)` for the
// current C8 publisher and reserve a separate private HBM arena for each of
// the two fixed Direct-HCCP queues.

#ifndef A5_EXACT_SCAN_SPECULATIVE_GET_TILING_H
#define A5_EXACT_SCAN_SPECULATIVE_GET_TILING_H

#include <cstdint>

struct A5ExactScanSpeculativeGetTilingData {
    // Exact C8 score-publication contract.
    uint32_t rowCount;
    uint32_t sourceCapacity;
    uint32_t poolSize;
    uint32_t kvObjectCapacity;
    uint32_t privateArenaCapacity;
    uint32_t tileReceiptCount;
    uint32_t scoreWorkspaceWords;
    uint32_t rowsPerEntry;

    // CandidateSeed is `[row, source, kv_object, uint16_score,
    // private_slot]`.  Every ExactScan producer owns one fixed compact region
    // of this size, so producer-side publication needs no global atomic head.
    uint32_t candidateCapacityPerProducer;
    uint32_t candidateSeedWords;

    // Direct-HCCP full-get geometry.
    uint32_t tableCapacity;
    uint32_t outputRows;
    uint32_t controlWords;
    uint32_t producerCount;
    uint32_t getLaneCount;
    uint32_t ringSlots;
    uint32_t recordsPerRingSlot;
    uint32_t getBootstrapRecords;
    uint32_t getWaveRecords;
    uint32_t pipelineWordsPerRing;
    uint32_t reserved;
};

#endif  // A5_EXACT_SCAN_SPECULATIVE_GET_TILING_H
