// P8 graph-visible forward-control phase transition.
//
// This operator is intentionally not a host callback and not part of C8.
// A captured layer graph uses it at node boundaries:
//
//   P8 Post -> advance 1->2 -> bounded P8 Drain nodes
//   -> advance 2->2 (ExactInputReady) -> QSFA
//   -> advance 2->3 (QsfaTerminalObserved) -> P8 private release
//   -> advance 3->4 (ModelTerminalObserved) -> P8 terminal sweep.
//
// The phase/flag protocol is deliberately restrictive.  A stale graph replay
// cannot move a different forward because `expected_identity` is checked
// before the phase CAS; an invalid/mismatched program only writes a compact
// diagnostic result and never modifies the forward-control record.

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

#include "a5_p8_forward_control_advance_tiling.h"
#include "kvaf_device/kvaf_fused_full_get_device.h"

using namespace AscendC;

namespace {

enum P8TransitionError : uint32_t {
    kP8TransitionOk = 0U,
    kP8TransitionInvalidInput = 1U,
    kP8TransitionIdentityMismatch = 2U,
    kP8TransitionIllegalProgram = 3U,
    kP8TransitionAbortGate = 4U,
    kP8TransitionPhaseMismatch = 5U,
    kP8TransitionFlagMismatch = 6U,
    kP8TransitionFlagCasFailed = 7U,
};

struct P8TransitionServiceResult {
    uint32_t success;
    uint32_t observed_phase;
    uint32_t observed_flags_low;
    uint32_t error;
};
static_assert(sizeof(P8TransitionServiceResult) == 16U,
              "P8 control-advance SIMT hand-off ABI changed");

constexpr uint64_t kP8RuntimeSettableFlags =
    kvaf_fused_get::kKvafP8ForwardFlagExactInputReady |
    kvaf_fused_get::kKvafP8ForwardFlagQsfaTerminalObserved |
    kvaf_fused_get::kKvafP8ForwardFlagModelTerminalObserved;
constexpr uint64_t kP8AbortFlags =
    kvaf_fused_get::kKvafP8ForwardFlagAbort |
    kvaf_fused_get::kKvafP8ForwardFlagPrivateViewDisabled;

__simt_callee__ inline bool IsLegalP8Transition(
    uint32_t expected_phase, uint32_t next_phase,
    uint64_t required_flags, uint64_t set_flags, uint64_t current_flags) {
    if ((set_flags & ~kP8RuntimeSettableFlags) != 0U ||
        (required_flags & ~kP8RuntimeSettableFlags) != 0U ||
        (current_flags & kP8AbortFlags) != 0U) {
        return false;
    }
    // Post is followed by one or more graph-visible, nonblocking drains.
    if (expected_phase == kvaf_fused_get::kKvafP8ForwardPhaseAdmitPost &&
        next_phase == kvaf_fused_get::kKvafP8ForwardPhaseDrain) {
        return required_flags == 0U && set_flags == 0U;
    }
    // Exact Top-K + Guaranteed Recall are a graph predecessor of this
    // flag-only boundary.  Keeping phase=DRAIN permits additional bounded
    // drain nodes while exact work proceeds without a host-side mutation.
    if (expected_phase == kvaf_fused_get::kKvafP8ForwardPhaseDrain &&
        next_phase == kvaf_fused_get::kKvafP8ForwardPhaseDrain) {
        return required_flags == 0U &&
               set_flags == kvaf_fused_get::kKvafP8ForwardFlagExactInputReady;
    }
    // This boundary is emitted after QSFA itself, immediately before the P8
    // QSFA-terminal private-release node.
    if (expected_phase == kvaf_fused_get::kKvafP8ForwardPhaseDrain &&
        next_phase == kvaf_fused_get::kKvafP8ForwardPhaseQsfaTerminal) {
        return required_flags ==
                   kvaf_fused_get::kKvafP8ForwardFlagExactInputReady &&
               set_flags ==
                   kvaf_fused_get::kKvafP8ForwardFlagQsfaTerminalObserved;
    }
    if (expected_phase == kvaf_fused_get::kKvafP8ForwardPhaseQsfaTerminal &&
        next_phase == kvaf_fused_get::kKvafP8ForwardPhaseForwardTerminal) {
        return required_flags ==
                   (kvaf_fused_get::kKvafP8ForwardFlagExactInputReady |
                    kvaf_fused_get::kKvafP8ForwardFlagQsfaTerminalObserved) &&
               set_flags ==
                   kvaf_fused_get::kKvafP8ForwardFlagModelTerminalObserved;
    }
    return false;
}

__simt_callee__ inline bool IsAbortToForwardTerminal(
    uint32_t expected_phase, uint32_t next_phase,
    uint64_t required_flags, uint64_t set_flags, uint64_t current_flags) {
    return expected_phase >= kvaf_fused_get::kKvafP8ForwardPhaseAdmitPost &&
           expected_phase <= kvaf_fused_get::kKvafP8ForwardPhaseQsfaTerminal &&
           next_phase == kvaf_fused_get::kKvafP8ForwardPhaseForwardTerminal &&
           required_flags == 0U &&
           (set_flags == 0U ||
            set_flags == kvaf_fused_get::kKvafP8ForwardFlagModelTerminalObserved) &&
           (current_flags & kP8AbortFlags) != 0U;
}

__simt_callee__ inline bool AtomicOrP8RuntimeFlags(
    __gm__ kvaf_fused_get::KvafP8ForwardControl* control,
    uint32_t set_flags, P8TransitionServiceResult* result) {
    if (control == nullptr || result == nullptr) {
        return false;
    }
    __gm__ int32_t* const flags_low =
        reinterpret_cast<__gm__ int32_t*>(&(control->flags));
    int32_t expected = *flags_low;
    for (uint32_t attempt = 0U; attempt < 32U; ++attempt) {
        const int32_t desired = expected | static_cast<int32_t>(set_flags);
        const int32_t observed = asc_atomic_cas(flags_low, expected, desired);
        result->observed_flags_low = static_cast<uint32_t>(observed);
        if (observed == expected) {
            result->observed_flags_low = static_cast<uint32_t>(desired);
            AscendC::Simt::ThreadFence();
            return true;
        }
        expected = observed;
    }
    return false;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void P8ForwardControlAdvanceSimtService(
    __gm__ kvaf_fused_get::KvafP8ForwardControl* control,
    uint64_t expected_epoch, uint64_t expected_replay,
    uint32_t expected_layer, uint32_t expected_phase, uint32_t next_phase,
    uint64_t required_flags, uint64_t set_flags,
    __ubuf__ P8TransitionServiceResult* result) {
    if (result == nullptr) {
        return;
    }
    result->success = 0U;
    result->observed_phase = kvaf_fused_get::kKvafP8ForwardPhaseInvalid;
    result->observed_flags_low = 0U;
    result->error = kP8TransitionInvalidInput;
    if (control == nullptr || expected_epoch == 0U ||
        expected_phase < kvaf_fused_get::kKvafP8ForwardPhaseAdmitPost ||
        expected_phase > kvaf_fused_get::kKvafP8ForwardPhaseForwardTerminal ||
        next_phase < kvaf_fused_get::kKvafP8ForwardPhaseAdmitPost ||
        next_phase > kvaf_fused_get::kKvafP8ForwardPhaseForwardTerminal) {
        return;
    }
    kvaf_fused_get::KvafP8ForwardControl current{};
    current.forward_epoch = control->forward_epoch;
    current.graph_replay_generation = control->graph_replay_generation;
    current.layer_control_index = control->layer_control_index;
    current.phase = control->phase;
    current.flags = control->flags;
    result->observed_phase = current.phase;
    result->observed_flags_low = static_cast<uint32_t>(current.flags);
    if (!kvaf_fused_get::IsKvafP8ForwardControlValid(current) ||
        current.forward_epoch != expected_epoch ||
        current.graph_replay_generation != expected_replay ||
        current.layer_control_index != expected_layer) {
        result->error = kP8TransitionIdentityMismatch;
        return;
    }
    const bool aborted = (current.flags & kP8AbortFlags) != 0U;
    if ((!aborted && !IsLegalP8Transition(expected_phase, next_phase,
                                           required_flags, set_flags,
                                           current.flags)) ||
        (aborted && !IsAbortToForwardTerminal(expected_phase, next_phase,
                                               required_flags, set_flags,
                                               current.flags))) {
        result->error = aborted ? kP8TransitionAbortGate
                                : kP8TransitionIllegalProgram;
        return;
    }
    if ((current.flags & required_flags) != required_flags) {
        result->error = kP8TransitionFlagMismatch;
        return;
    }
    // A phase field CAS protects a stale graph replay from advancing a newer
    // forward's state.  No later operation depends on a process-local phase.
    __gm__ int32_t* const phase = reinterpret_cast<__gm__ int32_t*>(
        &(control->phase));
    const int32_t observed = asc_atomic_cas(
        phase, static_cast<int32_t>(expected_phase),
        static_cast<int32_t>(next_phase));
    result->observed_phase = static_cast<uint32_t>(observed);
    if (observed != static_cast<int32_t>(expected_phase)) {
        result->error = kP8TransitionPhaseMismatch;
        return;
    }
    // The phase CAS is intentionally before the OR: a stale phase node must
    // not publish `ExactInputReady`/`QsfaTerminalObserved` on its own.
    if (!AtomicOrP8RuntimeFlags(control, static_cast<uint32_t>(set_flags),
                                result)) {
        result->error = kP8TransitionFlagCasFailed;
        return;
    }
    result->success = 1U;
    result->observed_phase = next_phase;
    result->error = kP8TransitionOk;
}

class A5P8ForwardControlAdvanceKernel {
public:
    __aicore__ inline A5P8ForwardControlAdvanceKernel(
        TPipe* pipe, const A5P8ForwardControlAdvanceTilingData* tiling)
        : pipe_(pipe), tiling_(tiling) {}

    __aicore__ inline void Init(GM_ADDR forward_control,
                                GM_ADDR expected_identity,
                                GM_ADDR transition_program,
                                GM_ADDR transition_result) {
        control_ = reinterpret_cast<__gm__ kvaf_fused_get::KvafP8ForwardControl*>(
            forward_control);
        identity_ = reinterpret_cast<__gm__ int64_t*>(expected_identity);
        program_ = reinterpret_cast<__gm__ int64_t*>(transition_program);
        result_ = reinterpret_cast<__gm__ int64_t*>(transition_result);
        enabled_ = pipe_ != nullptr && tiling_ != nullptr && control_ != nullptr &&
                   identity_ != nullptr && program_ != nullptr && result_ != nullptr &&
                   GetBlockIdx() == 0U;
        if (!enabled_) {
            return;
        }
        pipe_->InitBuffer(serviceBuffer_, sizeof(P8TransitionServiceResult));
        pipe_->InitBuffer(abortBuffer_,
                          sizeof(kvaf_fused_get::KvafP8ForwardAbortResult));
        serviceResult_ = (__ubuf__ P8TransitionServiceResult*)
            serviceBuffer_.Get<uint8_t>().GetPhyAddr();
        abortResult_ = (__ubuf__ kvaf_fused_get::KvafP8ForwardAbortResult*)
            abortBuffer_.Get<uint8_t>().GetPhyAddr();
        enabled_ = serviceResult_ != nullptr && abortResult_ != nullptr;
    }

    __aicore__ inline void Process() {
        if (!enabled_) {
            return;
        }
        const int64_t epoch = ReadGmByPassDCache<int64_t>(identity_ + 0U);
        const int64_t replay = ReadGmByPassDCache<int64_t>(identity_ + 1U);
        const int64_t layer = ReadGmByPassDCache<int64_t>(identity_ + 2U);
        const int64_t expected = ReadGmByPassDCache<int64_t>(program_ + 0U);
        const int64_t next = ReadGmByPassDCache<int64_t>(program_ + 1U);
        const int64_t required = ReadGmByPassDCache<int64_t>(program_ + 2U);
        const int64_t set = ReadGmByPassDCache<int64_t>(program_ + 3U);
        serviceResult_->success = 0U;
        serviceResult_->observed_phase =
            kvaf_fused_get::kKvafP8ForwardPhaseInvalid;
        serviceResult_->observed_flags_low = 0U;
        serviceResult_->error = kP8TransitionInvalidInput;
        constexpr int64_t kUint32MaxAsInt64 = 0xFFFFFFFFLL;
        const bool scalar_valid = epoch > 0 && replay >= 0 && layer >= 0 &&
            layer <= kUint32MaxAsInt64 && expected >= 0 &&
            next >= 0 && required >= 0 && set >= 0;
        if (scalar_valid) {
            set_flag(PIPE_S, PIPE_V, simtEvent_);
            wait_flag(PIPE_S, PIPE_V, simtEvent_);
            AscendC::Simt::VF_CALL<P8ForwardControlAdvanceSimtService>(
                AscendC::Simt::Dim3{1U, 1U, 1U}, control_,
                static_cast<uint64_t>(epoch), static_cast<uint64_t>(replay),
                static_cast<uint32_t>(layer), static_cast<uint32_t>(expected),
                static_cast<uint32_t>(next), static_cast<uint64_t>(required),
                static_cast<uint64_t>(set), serviceResult_);
            set_flag(PIPE_V, PIPE_S, simtEvent_);
            wait_flag(PIPE_V, PIPE_S, simtEvent_);
        }
        // A failure after the phase CAS (the only possible partial mutation)
        // is made fail-closed.  Mismatch/illegal-program paths leave phase
        // unchanged and are diagnosed for the graph runtime to handle.
        if (serviceResult_->error == kP8TransitionFlagCasFailed) {
            (void)kvaf_fused_get::MarkKvafP8ForwardAbortFromAicore(
                control_, abortResult_, simtEvent_);
        }
        WriteGmByPassDCache<int64_t>(result_ + 0U,
            static_cast<int64_t>(serviceResult_->success));
        WriteGmByPassDCache<int64_t>(result_ + 1U,
            static_cast<int64_t>(serviceResult_->observed_phase));
        WriteGmByPassDCache<int64_t>(result_ + 2U,
            static_cast<int64_t>(serviceResult_->observed_flags_low));
        WriteGmByPassDCache<int64_t>(result_ + 3U,
            static_cast<int64_t>(serviceResult_->error));
        DataSyncBarrier<MemDsbT::DDR>();
    }

private:
    TPipe* pipe_ = nullptr;
    const A5P8ForwardControlAdvanceTilingData* tiling_ = nullptr;
    TBuf<TPosition::VECCALC> serviceBuffer_;
    TBuf<TPosition::VECCALC> abortBuffer_;
    __ubuf__ P8TransitionServiceResult* serviceResult_ = nullptr;
    __ubuf__ kvaf_fused_get::KvafP8ForwardAbortResult* abortResult_ = nullptr;
    event_t simtEvent_ = static_cast<event_t>(EVENT_ID0);
    __gm__ kvaf_fused_get::KvafP8ForwardControl* control_ = nullptr;
    __gm__ int64_t* identity_ = nullptr;
    __gm__ int64_t* program_ = nullptr;
    __gm__ int64_t* result_ = nullptr;
    bool enabled_ = false;
};

}  // namespace

extern "C" __global__ __aicore__ void a5_p8_forward_control_advance(
    GM_ADDR forward_control, GM_ADDR expected_identity,
    GM_ADDR transition_program, GM_ADDR transition_result,
    GM_ADDR forward_control_ref, GM_ADDR transition_result_ref,
    GM_ADDR workspace, GM_ADDR tiling) {
    (void)forward_control_ref;
    (void)transition_result_ref;
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5P8ForwardControlAdvanceTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    A5P8ForwardControlAdvanceKernel op(&pipe, &tilingData);
    op.Init(forward_control, expected_identity, transition_program,
            transition_result);
    op.Process();
}
