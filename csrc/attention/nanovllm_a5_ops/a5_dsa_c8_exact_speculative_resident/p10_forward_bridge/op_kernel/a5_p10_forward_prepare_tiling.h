#ifndef A5_P10_FORWARD_PREPARE_TILING_H
#define A5_P10_FORWARD_PREPARE_TILING_H

#include <cstdint>

// All dimensions are graph-static aliases.  The preparation node has no
// knowledge of C8 tiles or HCCP queues beyond their caller-owned mutable
// buffers; it only establishes their forward-local initial state.
struct A5P10ForwardPrepareTilingData {
    uint32_t usedCoreNum;
    uint32_t receiptStateElements;
    uint32_t objectToForwardSlotElements;
    uint32_t sourceToForwardSlotElements;
    uint32_t queueLeaseElements;
    uint32_t residentReadyElements;
};

#endif  // A5_P10_FORWARD_PREPARE_TILING_H
