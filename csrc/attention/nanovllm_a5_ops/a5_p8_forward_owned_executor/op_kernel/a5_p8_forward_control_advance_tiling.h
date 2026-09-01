// P8 graph-visible forward-control transition node.
//
// This tiny node is deliberately separate from the transport executor.  A
// captured graph places it between completed P8 phase nodes, so no Python/GIL
// callback mutates `KvafP8ForwardControl` per candidate.  Its inputs are
// fixed-size graph tensors and its sole stateful outputs alias the durable GM
// control/result records.

#ifndef A5_P8_FORWARD_CONTROL_ADVANCE_TILING_H
#define A5_P8_FORWARD_CONTROL_ADVANCE_TILING_H

#include <cstdint>

constexpr uint32_t kA5P8ControlWords = 4U;
constexpr uint32_t kA5P8IdentityWords = 3U;
constexpr uint32_t kA5P8TransitionProgramWords = 4U;
constexpr uint32_t kA5P8TransitionResultWords = 4U;

struct A5P8ForwardControlAdvanceTilingData {
    uint32_t reserved0;
    uint32_t reserved1;
};

#endif  // A5_P8_FORWARD_CONTROL_ADVANCE_TILING_H
