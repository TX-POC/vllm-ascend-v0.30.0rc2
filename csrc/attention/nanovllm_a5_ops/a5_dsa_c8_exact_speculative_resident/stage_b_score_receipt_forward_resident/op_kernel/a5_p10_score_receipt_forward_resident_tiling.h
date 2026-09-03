#ifndef A5_P10_SCORE_RECEIPT_FORWARD_RESIDENT_TILING_H
#define A5_P10_SCORE_RECEIPT_FORWARD_RESIDENT_TILING_H

#include <cstdint>

// All fields are graph-static dimensions or bounded scheduling parameters.
// `executorWorkerCount` is a logical companion-work-item count selected by
// tiling. `executorWorkerLimit` is the graph scheduler's occupancy budget;
// neither field is a physical-AIV affinity, a request for all C8 AIV blocks,
// or a fixed Get-core partition. `receiptPollQuantum` bounds one polling turn
// before the executor reaches a device-side checkpoint.
struct A5P10ScoreReceiptForwardResidentTilingData {
  uint32_t executorWorkerCount;
  uint32_t executorWorkerLimit;
  uint32_t receiptPollQuantum;
  uint32_t activeRowCount;
  uint32_t receiptCapacity;
  uint32_t sourceCapacity;
  // Exact KVAF address-table geometry. qcap is logical 128-token blocks.
  // For source S in active pool P:
  // O_token = ((P * numLayers + layerId) * qcap * 128) + S.
  uint32_t numLayers;
  uint32_t layerId;
  uint32_t qcap;
  uint32_t poolCapacity;
  uint32_t kvObjectCapacity;
  uint32_t residentRowCount;
  uint32_t attentionSlotCapacity;
  uint32_t sparseCount;
  uint32_t scoreCutoffU16;
  uint32_t boundedDrainPolls;
  uint32_t reserved;
};

#endif  // A5_P10_SCORE_RECEIPT_FORWARD_RESIDENT_TILING_H
