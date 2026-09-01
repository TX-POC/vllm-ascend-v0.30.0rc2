// P8 forward-owned speculative Get executor tiling.
//
// This operator deliberately consumes score receipts already published by the
// C8 publisher.  It never includes or changes C8.  The four graph phases
// (PostOnly, CQ drain, QSFA terminal release, ForwardTerminal quarantine)
// share one fixed resource ABI.

#ifndef A5_P8_FORWARD_OWNED_EXECUTOR_TILING_H
#define A5_P8_FORWARD_OWNED_EXECUTOR_TILING_H

#include <cstdint>

constexpr uint32_t kA5P8CandidateSeedWords = 4U;
// [forward_epoch, graph_replay_generation, layer_control_index,
//  receipt_generation].  Generation is the release word and is always
// written/read last by the Stage-A -> Stage-B receipt seam.
constexpr uint32_t kA5P8CandidateReceiptIdentityWords = 4U;
constexpr uint32_t kA5P8CandidateCapacity = 128U;
constexpr uint32_t kA5P8WaveRecordWords = 32U;
constexpr uint32_t kA5P8PrivateRecordWords = 10U;
constexpr uint32_t kA5P8RequestCapacity = 64U;
// KVAF_P8_FORWARD_CONTROL_BYTES is 32 bytes.  Its four int64 words are not
// four independent counters: [epoch, replay, packed(layer, phase), flags].
// The device kernel must use KVAF's field-wise loader instead of reinterpreting
// word 2 as a layer and word 3 as a phase.
constexpr uint32_t kA5P8ForwardControlWords = 4U;
constexpr uint32_t kA5P8WorkerStatColumns = 16U;

enum A5P8ExecutionPhase : uint32_t {
    // Claim global private rows, persist a queue-owned wave record, then
    // issue PostOnly.  It deliberately does not wait for or drain CQ.
    kA5P8PhasePost = 1U,
    // Rehydrate durable wave records, poll/Finish CQ, publish private
    // READY/FAILED, and only then release the queue lease.
    kA5P8PhaseDrain = 2U,
    // Invoked after the matching QSFA terminal event.  It releases private
    // rows/mappings; it must never be used as a queue-release substitute.
    kA5P8PhaseQsfaTerminal = 3U,
    // A per-forward fail-closed sweep.  It quarantines unresolved wave/private
    // records and never frees a possibly in-flight Direct-HCCP resource.
    kA5P8PhaseForwardTerminal = 4U,
};

struct A5P8ForwardOwnedExecutorTilingData {
    uint32_t workerCount;
    uint32_t receiptCount;
    uint32_t rowCount;
    uint32_t sourceCapacity;
    uint32_t poolSize;
    uint32_t kvObjectCapacity;
    uint32_t queueCount;
    uint32_t privateRowCount;
    uint32_t workerStatsRows;
    uint32_t phase;
    uint32_t candidateCapacity;
    uint32_t requestCapacity;
    uint32_t reserved0;
    uint32_t reserved1;
};

#endif  // A5_P8_FORWARD_OWNED_EXECUTOR_TILING_H
