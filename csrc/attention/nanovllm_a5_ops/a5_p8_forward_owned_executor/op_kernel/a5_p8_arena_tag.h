#pragma once

// P8's graph-wide arena has exactly one atomic control word.  It is not a
// core assignment and it is not a second forward identity: the high bits are
// a monotonically increasing lease generation, while the low three bits are
// the lifecycle state.  Graph topology already forbids a new forward before
// the previous terminal.  The generation is only a cheap defensive CAS
// check: if that topology is ever violated, a delayed terminal cannot free a
// later lease of the same fixed queue/private arena.

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace a5_p8_arena {

enum ArenaState : uint32_t {
    kArenaIdle = 0U,
    kArenaUnhealthy = 1U,
    kArenaActive = 2U,
    kArenaClaiming = 3U,
    kArenaTerminalClaimed = 4U,
};

constexpr uint32_t kArenaStateBits = 3U;
constexpr uint32_t kArenaStateMask = (1U << kArenaStateBits) - 1U;
constexpr uint32_t kArenaGenerationMax = (1U << (32U - kArenaStateBits)) - 1U;

struct ArenaResult {
    uint32_t success;
    uint32_t observed_tag;
};
static_assert(sizeof(ArenaResult) == 8U,
              "P8 tagged arena result ABI changed");

// AICore graph nodes use these helpers directly.  They deliberately have a
// separate SIMT sibling below: AscendC does not permit a __simt_callee__
// function to call an __aicore__ helper, even when both bodies are trivially
// inline.  Keeping the two three-instruction encoders explicit is smaller
// and safer than relying on compiler-specific cross-domain inlining.
__aicore__ inline uint32_t MakeTag(uint32_t generation, ArenaState state)
{
    return (generation << kArenaStateBits) |
           static_cast<uint32_t>(state);
}

__aicore__ inline ArenaState StateOf(uint32_t tag)
{
    return static_cast<ArenaState>(tag & kArenaStateMask);
}

__aicore__ inline uint32_t GenerationOf(uint32_t tag)
{
    return tag >> kArenaStateBits;
}

// SIMT services run in a different callable domain from the surrounding
// AICore graph nodes.  These mirror the AICore helpers exactly; both families
// encode the same [generation | state] control word and do not change the P8
// arena state-machine semantics.
__simt_callee__ inline uint32_t SimtMakeTag(
    uint32_t generation, ArenaState state)
{
    return (generation << kArenaStateBits) |
           static_cast<uint32_t>(state);
}

__simt_callee__ inline ArenaState SimtStateOf(uint32_t tag)
{
    return static_cast<ArenaState>(tag & kArenaStateMask);
}

__simt_callee__ inline uint32_t SimtGenerationOf(uint32_t tag)
{
    return tag >> kArenaStateBits;
}

// A bounded CAS implementation that is used only by the admission node.
// Nothing else transitions IDLE, so an unsuccessful result means another
// forward owns or has quarantined the arena and this invocation must fall
// back to exact recall without mutating it.
__simt_callee__ inline bool TryClaimAdmissionFromSimt(
    __gm__ int32_t* health, __ubuf__ ArenaResult* result)
{
    if (health == nullptr || result == nullptr) {
        return false;
    }
    result->success = 0U;
    result->observed_tag = static_cast<uint32_t>(*health);
    const uint32_t expected = result->observed_tag;
    if (SimtStateOf(expected) != kArenaIdle ||
        SimtGenerationOf(expected) >= kArenaGenerationMax) {
        return false;
    }
    const uint32_t desired = SimtMakeTag(SimtGenerationOf(expected) + 1U,
                                         kArenaClaiming);
    const int32_t observed = asc_atomic_cas(
        health, static_cast<int32_t>(expected), static_cast<int32_t>(desired));
    result->observed_tag = static_cast<uint32_t>(observed);
    if (static_cast<uint32_t>(observed) != expected) {
        return false;
    }
    AscendC::Simt::ThreadFence();
    result->success = 1U;
    result->observed_tag = desired;
    return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void ClaimAdmissionSimtService(
    __gm__ int32_t* health, __ubuf__ ArenaResult* result)
{
    (void)TryClaimAdmissionFromSimt(health, result);
}

// The following named transitions are deliberately narrower than a generic
// state machine primitive.  P8 must never have an ACTIVE -> IDLE shortcut:
// only ForwardTerminal can release a tagged lease after its bounded sweep.
__simt_callee__ inline bool FinishAdmissionFromSimt(
    __gm__ int32_t* health, uint32_t claiming_tag, bool healthy,
    __ubuf__ ArenaResult* result)
{
    if (health == nullptr || result == nullptr ||
        SimtStateOf(claiming_tag) != kArenaClaiming ||
        SimtGenerationOf(claiming_tag) == 0U) {
        return false;
    }
    result->success = 0U;
    result->observed_tag = static_cast<uint32_t>(*health);
    const uint32_t desired = SimtMakeTag(
        SimtGenerationOf(claiming_tag),
        healthy ? kArenaActive : kArenaUnhealthy);
    const int32_t observed = asc_atomic_cas(
        health, static_cast<int32_t>(claiming_tag), static_cast<int32_t>(desired));
    result->observed_tag = static_cast<uint32_t>(observed);
    if (static_cast<uint32_t>(observed) != claiming_tag) {
        return false;
    }
    AscendC::Simt::ThreadFence();
    result->success = 1U;
    result->observed_tag = desired;
    return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void FinishAdmissionSimtService(
    __gm__ int32_t* health, uint32_t claiming_tag, bool healthy,
    __ubuf__ ArenaResult* result)
{
    (void)FinishAdmissionFromSimt(health, claiming_tag, healthy, result);
}

__simt_callee__ inline bool PoisonActiveLeaseFromSimt(
    __gm__ int32_t* health, uint32_t active_tag,
    __ubuf__ ArenaResult* result)
{
    if (health == nullptr || result == nullptr ||
        SimtStateOf(active_tag) != kArenaActive ||
        SimtGenerationOf(active_tag) == 0U) {
        return false;
    }
    result->success = 0U;
    result->observed_tag = static_cast<uint32_t>(*health);
    const uint32_t desired = SimtMakeTag(
        SimtGenerationOf(active_tag), kArenaUnhealthy);
    const int32_t observed = asc_atomic_cas(
        health, static_cast<int32_t>(active_tag), static_cast<int32_t>(desired));
    result->observed_tag = static_cast<uint32_t>(observed);
    if (static_cast<uint32_t>(observed) != active_tag) {
        return false;
    }
    AscendC::Simt::ThreadFence();
    result->success = 1U;
    result->observed_tag = desired;
    return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void PoisonActiveLeaseSimtService(
    __gm__ int32_t* health, uint32_t active_tag,
    __ubuf__ ArenaResult* result)
{
    (void)PoisonActiveLeaseFromSimt(health, active_tag, result);
}

__simt_callee__ inline bool CompleteTerminalFromSimt(
    __gm__ int32_t* health, uint32_t terminal_tag, bool release_clean,
    __ubuf__ ArenaResult* result)
{
    if (health == nullptr || result == nullptr ||
        SimtStateOf(terminal_tag) != kArenaTerminalClaimed ||
        SimtGenerationOf(terminal_tag) == 0U) {
        return false;
    }
    result->success = 0U;
    result->observed_tag = static_cast<uint32_t>(*health);
    const uint32_t desired = SimtMakeTag(SimtGenerationOf(terminal_tag),
        release_clean ? kArenaIdle : kArenaUnhealthy);
    const int32_t observed = asc_atomic_cas(
        health, static_cast<int32_t>(terminal_tag), static_cast<int32_t>(desired));
    result->observed_tag = static_cast<uint32_t>(observed);
    if (static_cast<uint32_t>(observed) != terminal_tag) {
        return false;
    }
    AscendC::Simt::ThreadFence();
    result->success = 1U;
    result->observed_tag = desired;
    return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void CompleteTerminalSimtService(
    __gm__ int32_t* health, uint32_t terminal_tag, bool release_clean,
    __ubuf__ ArenaResult* result)
{
    (void)CompleteTerminalFromSimt(health, terminal_tag, release_clean,
                                   result);
}

// A wrapped generation cannot safely distinguish a delayed old terminal from
// a future lease.  Admission turns this exact last idle tag permanently
// unhealthy instead of reusing generation zero.
__simt_callee__ inline bool QuarantineGenerationWrapFromSimt(
    __gm__ int32_t* health, uint32_t idle_max_tag,
    __ubuf__ ArenaResult* result)
{
    if (health == nullptr || result == nullptr ||
        SimtStateOf(idle_max_tag) != kArenaIdle ||
        SimtGenerationOf(idle_max_tag) != kArenaGenerationMax) {
        return false;
    }
    result->success = 0U;
    result->observed_tag = static_cast<uint32_t>(*health);
    const uint32_t desired = SimtMakeTag(kArenaGenerationMax,
                                         kArenaUnhealthy);
    const int32_t observed = asc_atomic_cas(
        health, static_cast<int32_t>(idle_max_tag), static_cast<int32_t>(desired));
    result->observed_tag = static_cast<uint32_t>(observed);
    if (static_cast<uint32_t>(observed) != idle_max_tag) {
        return false;
    }
    AscendC::Simt::ThreadFence();
    result->success = 1U;
    result->observed_tag = desired;
    return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void QuarantineGenerationWrapSimtService(
    __gm__ int32_t* health, uint32_t idle_max_tag,
    __ubuf__ ArenaResult* result)
{
    (void)QuarantineGenerationWrapFromSimt(health, idle_max_tag, result);
}

// Terminal can take exactly one ACTIVE(g) -> TERMINAL_CLAIMED(g) ownership
// transition.  A previously aborted forward is also terminal-sweepable from
// UNHEALTHY(g); it remains unhealthy after the sweep.
__simt_callee__ inline bool TryClaimTerminalFromSimt(
    __gm__ int32_t* health, uint32_t generation,
    __ubuf__ ArenaResult* result)
{
    if (health == nullptr || result == nullptr || generation == 0U ||
        generation > kArenaGenerationMax) {
        return false;
    }
    result->success = 0U;
    result->observed_tag = static_cast<uint32_t>(*health);
    const uint32_t observed_tag = result->observed_tag;
    const ArenaState state = SimtStateOf(observed_tag);
    if (SimtGenerationOf(observed_tag) != generation ||
        (state != kArenaActive && state != kArenaUnhealthy)) {
        return false;
    }
    const uint32_t desired = SimtMakeTag(generation,
                                         kArenaTerminalClaimed);
    const int32_t observed = asc_atomic_cas(
        health, static_cast<int32_t>(observed_tag), static_cast<int32_t>(desired));
    result->observed_tag = static_cast<uint32_t>(observed);
    if (static_cast<uint32_t>(observed) != observed_tag) {
        return false;
    }
    AscendC::Simt::ThreadFence();
    result->success = 1U;
    // Preserve the pre-claim state in the high-level caller by returning the
    // old tag; callers that need the terminal tag derive it from generation.
    return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void ClaimTerminalSimtService(
    __gm__ int32_t* health, uint32_t generation,
    __ubuf__ ArenaResult* result)
{
    (void)TryClaimTerminalFromSimt(health, generation, result);
}

}  // namespace a5_p8_arena
