#ifndef A5_EXACT_RECONCILE_PRIVATE_VIEW_TILING_H
#define A5_EXACT_RECONCILE_PRIVATE_VIEW_TILING_H

#include <cstdint>

// All fields describe graph-stable tensor dimensions.  In particular, this
// operator never receives a writable persistent cache_slots tensor: the
// snapshot is an input-only directory and placement_delta is the only output
// that can later request a persistent publication.
struct A5ExactReconcilePrivateViewTilingData {
    uint32_t usedCoreNum;
    uint32_t batchSize;
    uint32_t poolSize;
    uint32_t sourceCapacity;
    uint32_t privateQueueCount;
    uint32_t privateRowsPerQueue;
};

#endif  // A5_EXACT_RECONCILE_PRIVATE_VIEW_TILING_H
