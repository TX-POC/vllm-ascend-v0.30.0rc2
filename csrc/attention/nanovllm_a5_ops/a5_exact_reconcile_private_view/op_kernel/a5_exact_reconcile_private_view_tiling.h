#ifndef A5_EXACT_RECONCILE_PRIVATE_VIEW_TILING_H
#define A5_EXACT_RECONCILE_PRIVATE_VIEW_TILING_H

#include <cstdint>

// All fields describe graph-stable tensor dimensions.  In particular, this
// operator never receives a writable persistent KV-object directory: the
// snapshot is an input-only directory and placement_delta is the only output
// that can later request a persistent publication.
struct A5ExactReconcilePrivateViewTilingData {
    uint32_t usedCoreNum;
    // Number of exact C8 vectors: ProductExceptLast(exact_topk_indices).
    // It is not necessarily the request-batch B (TND/MTP can have N2 > 1).
    uint32_t rowCount;
    uint32_t poolSize;
    // C8 emits indices in the score-source domain.  kv_object_map resolves
    // those to the declared QSFA-addressable object domain.  These two
    // capacities are deliberately separate when cmp_ratio > 1.
    uint32_t scoreSourceCapacity;
    uint32_t kvObjectCapacity;
    uint32_t privateQueueCount;
    uint32_t privateRowsPerQueue;
};

#endif  // A5_EXACT_RECONCILE_PRIVATE_VIEW_TILING_H
