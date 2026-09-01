// P8 graph-visible arena admission gate.
//
// This is deliberately a one-AIV node placed before the multi-AIV POST node.
// It checks the fixed Direct-HCCP queue/private arena once per graph layer and
// latches a sticky health scalar before any dynamic producer can acquire a
// queue.  It is not a C8 operator and it never resets or recovers an arena.

#ifndef A5_P8_ARENA_ADMISSION_GATE_TILING_H
#define A5_P8_ARENA_ADMISSION_GATE_TILING_H

#include <cstdint>

constexpr uint32_t kA5P8AdmissionControlWords = 4U;
constexpr uint32_t kA5P8AdmissionIdentityWords = 3U;
constexpr uint32_t kA5P8AdmissionResultWords = 4U;
constexpr uint32_t kA5P8AdmissionStreamingControlWords = 16U;
constexpr uint32_t kA5P8AdmissionWaveRecordWords = 32U;
constexpr uint32_t kA5P8AdmissionPrivateRecordWords = 10U;

struct A5P8ArenaAdmissionGateTilingData {
    uint32_t queueCount;
    uint32_t privateRowCount;
};

#endif  // A5_P8_ARENA_ADMISSION_GATE_TILING_H
