// Fixed geometry for the first device-resident ExactScan + full SpeculativeGet
// prototype.  All sizes are graph-static and intentionally small enough to
// make the producer/consumer contract observable before C8 is embedded.

#ifndef A5_EXACT_SCAN_SPECULATIVE_GET_TILING_H
#define A5_EXACT_SCAN_SPECULATIVE_GET_TILING_H

#include <cstdint>

struct A5ExactScanSpeculativeGetTilingData {
    uint32_t scoreCount;
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
