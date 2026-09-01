// P8 graph-stable fixed-arena admission gate.
//
// This is a tiny graph-control node, not a dedicated Get AIV.  It runs once
// before the normal dynamic POST workers, scans only the fixed wave/private
// arena plus KVAF queue leases, and latches `p8_arena_health` if it finds
// persistent unsafe state.  A later P8 node never clears that scalar; a new
// arena allocation is the only recovery boundary.
//
// The arena is acquired once per forward, not once per layer.  A captured
// graph must wait for the preceding P8 FORWARD_TERMINAL event before a *new
// forward* reuses any P8 GM tensor or invokes this gate.  A later layer in
// the same admitted forward does not re-admit; before it reuses a particular
// queue/private row it waits only for that prior layer's CQ and QSFA-terminal
// reader completion.  This is graph topology, not a permanent Get-core
// reservation.

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

#include "a5_p8_arena_tag.h"
#include "a5_p8_arena_admission_gate_tiling.h"
#include "kvaf_device/kvaf_fused_full_get_device.h"

using namespace AscendC;

namespace {
enum P8AdmissionReason : uint32_t {
    kP8AdmissionAdmitted = 0U,
    kP8AdmissionStickyUnhealthy = 1U,
    kP8AdmissionControlInvalid = 2U,
    kP8AdmissionIdentityOrPhaseMismatch = 3U,
    kP8AdmissionWaveNonFree = 4U,
    kP8AdmissionWaveCorrupt = 5U,
    kP8AdmissionPrivateNonFree = 6U,
    kP8AdmissionPrivateCorrupt = 7U,
    kP8AdmissionOrphanLease = 8U,
    kP8AdmissionAbortPublishFailed = 9U,
    // Another graph has the arena lease.  This is a normal P8 deferral, not
    // corruption; it must not poison the currently active forward.
    kP8AdmissionArenaBusy = 10U,
};

using P8ArenaHealthLatchResult = a5_p8_arena::ArenaResult;

class A5P8ArenaAdmissionGateKernel {
public:
    __aicore__ inline A5P8ArenaAdmissionGateKernel(
        TPipe* pipe, const A5P8ArenaAdmissionGateTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR forward_control, GM_ADDR p8_arena_health,
        GM_ADDR fused_get_control, GM_ADDR p8_wave_records,
        GM_ADDR p8_private_records, GM_ADDR expected_identity,
        GM_ADDR admission_result)
    {
        forwardControlGm_ = reinterpret_cast<__gm__
            kvaf_fused_get::KvafP8ForwardControl*>(forward_control);
        arenaHealthGm_ = reinterpret_cast<__gm__ int32_t*>(p8_arena_health);
        controlGm_ = reinterpret_cast<__gm__ const
            kvaf_fused_get::KvafStreamingBatchGetControl*>(fused_get_control);
        waveRecords_ = reinterpret_cast<__gm__
            kvaf_fused_get::KvafP8WaveRecord*>(p8_wave_records);
        privateRecords_ = reinterpret_cast<__gm__
            kvaf_fused_get::KvafP8PrivateRecord*>(p8_private_records);
        expectedIdentity_ = reinterpret_cast<__gm__ int64_t*>(expected_identity);
        result_ = reinterpret_cast<__gm__ int64_t*>(admission_result);
        hasArenaLease_ = false;
        initialArenaTag_ = 0U;
        claimArenaTag_ = 0U;
        activeArenaTag_ = 0U;
        enabled_ = pipe_ != nullptr && tiling_ != nullptr &&
            forwardControlGm_ != nullptr && arenaHealthGm_ != nullptr &&
            controlGm_ != nullptr && waveRecords_ != nullptr &&
            privateRecords_ != nullptr && expectedIdentity_ != nullptr &&
            result_ != nullptr && GetBlockIdx() == 0U;
        if (!enabled_) {
            return;
        }
        pipe_->InitBuffer(latchBuffer_, sizeof(P8ArenaHealthLatchResult));
        pipe_->InitBuffer(abortBuffer_,
                          sizeof(kvaf_fused_get::KvafP8ForwardAbortResult));
        latchResult_ = (__ubuf__ P8ArenaHealthLatchResult*)
            latchBuffer_.Get<uint8_t>().GetPhyAddr();
        abortResult_ = (__ubuf__ kvaf_fused_get::KvafP8ForwardAbortResult*)
            abortBuffer_.Get<uint8_t>().GetPhyAddr();
        enabled_ = latchResult_ != nullptr && abortResult_ != nullptr;
    }

    __aicore__ inline void Process()
    {
        if (!enabled_) {
            return;
        }
        WriteResult(0U, kP8AdmissionControlInvalid, -1, -1);
        DataSyncBarrier<MemDsbT::DDR>();

        kvaf_fused_get::KvafP8ForwardControl forward{};
        kvaf_fused_get::KvafStreamingBatchGetControl control{};
        if (!kvaf_fused_get::LoadKvafP8ForwardControlFromGm(
                forwardControlGm_, &forward) ||
            !kvaf_fused_get::LoadKvafStreamingBatchGetControlFromGmForAicore(
                controlGm_, &control) ||
            !kvaf_fused_get::IsKvafStreamingBatchGetControlValidForAicore(
                control) ||
            control.queue_count != tiling_->queueCount) {
            // A malformed/absent runtime control is an invocation setup
            // failure.  It must not poison an otherwise clean reusable arena.
            WriteResult(0U, kP8AdmissionControlInvalid, -1, -1);
            return;
        }
        const int64_t expected_epoch =
            ReadGmByPassDCache<int64_t>(expectedIdentity_ + 0U);
        const int64_t expected_replay =
            ReadGmByPassDCache<int64_t>(expectedIdentity_ + 1U);
        const int64_t expected_layer =
            ReadGmByPassDCache<int64_t>(expectedIdentity_ + 2U);
        if (!kvaf_fused_get::IsKvafP8ForwardControlValid(forward) ||
            expected_epoch <= 0 || expected_replay < 0 || expected_layer < 0 ||
            expected_layer > static_cast<int64_t>(
                ~static_cast<uint32_t>(0U)) ||
            forward.forward_epoch != static_cast<uint64_t>(expected_epoch) ||
            forward.graph_replay_generation !=
                static_cast<uint64_t>(expected_replay) ||
            forward.layer_control_index !=
                static_cast<uint32_t>(expected_layer) ||
            forward.phase != kvaf_fused_get::kKvafP8ForwardPhaseAdmitPost) {
            WriteResult(0U, kP8AdmissionIdentityOrPhaseMismatch, -1, -1);
            return;
        }

        // The graph topology forbids a next-forward admission/reuse until
        // the prior FORWARD_TERMINAL event has completed.  Scan first, then
        // acquire: this lets a clean idle arena stay untouched on malformed
        // inputs while the tagged CAS still rejects a contract violation.
        initialArenaTag_ = static_cast<uint32_t>(
            ReadGmByPassDCache<int32_t>(arenaHealthGm_));
        const a5_p8_arena::ArenaState initial_state =
            a5_p8_arena::StateOf(initialArenaTag_);
        if (initial_state != a5_p8_arena::kArenaIdle) {
            WriteResult(0U,
                        initial_state == a5_p8_arena::kArenaActive ||
                                initial_state == a5_p8_arena::kArenaClaiming ||
                                initial_state ==
                                    a5_p8_arena::kArenaTerminalClaimed
                            ? kP8AdmissionArenaBusy
                            : kP8AdmissionStickyUnhealthy,
                        -1, static_cast<int64_t>(initialArenaTag_));
            return;
        }
        if (a5_p8_arena::GenerationOf(initialArenaTag_) >=
            a5_p8_arena::kArenaGenerationMax) {
            (void)QuarantineGenerationWrap();
            WriteResult(0U, kP8AdmissionStickyUnhealthy, -1,
                        static_cast<int64_t>(initialArenaTag_));
            return;
        }

        for (uint32_t queue = 0U; queue < tiling_->queueCount; ++queue) {
            kvaf_fused_get::KvafP8WaveRecord wave{};
            if (!kvaf_fused_get::LoadKvafP8WaveRecordFromGm(
                    waveRecords_ + queue, &wave)) {
                FailClosed(kP8AdmissionWaveCorrupt,
                           static_cast<int64_t>(queue), -1);
                return;
            }
            if (kvaf_fused_get::IsKvafP8WaveRecordValid(wave)) {
                if (wave.state != kvaf_fused_get::kKvafP8WaveFree) {
                    FailClosed(kP8AdmissionWaveNonFree,
                               static_cast<int64_t>(queue),
                               static_cast<int64_t>(wave.state));
                    return;
                }
            } else if (!kvaf_fused_get::IsKvafP8WaveRecordAllZero(wave)) {
                FailClosed(kP8AdmissionWaveCorrupt,
                           static_cast<int64_t>(queue), -1);
                return;
            }
        }
        for (uint32_t slot = 0U; slot < tiling_->privateRowCount; ++slot) {
            kvaf_fused_get::KvafP8PrivateRecord record{};
            if (!kvaf_fused_get::LoadKvafP8PrivateRecordFromGm(
                    privateRecords_ + slot, &record)) {
                FailClosed(kP8AdmissionPrivateCorrupt,
                           static_cast<int64_t>(slot), -1);
                return;
            }
            if (kvaf_fused_get::IsKvafP8PrivateRecordValid(record)) {
                if (record.state != kvaf_fused_get::kKvafP8PrivateFree) {
                    FailClosed(kP8AdmissionPrivateNonFree,
                               static_cast<int64_t>(slot),
                               static_cast<int64_t>(record.state));
                    return;
                }
            } else if (!IsAllZero(reinterpret_cast<__gm__ int64_t*>(
                           privateRecords_ + slot),
                                   kA5P8AdmissionPrivateRecordWords)) {
                FailClosed(kP8AdmissionPrivateCorrupt,
                           static_cast<int64_t>(slot), -1);
                return;
            }
        }
        __gm__ const int32_t* const leases = reinterpret_cast<__gm__
            const int32_t*>(control.queue_leases);
        if (leases == nullptr) {
            WriteResult(0U, kP8AdmissionControlInvalid, -1, -1);
            return;
        }
        for (uint32_t queue = 0U; queue < tiling_->queueCount; ++queue) {
            const int32_t owner = ReadGmByPassDCache<int32_t>(leases + queue);
            if (owner != 0) {
                FailClosed(kP8AdmissionOrphanLease,
                           static_cast<int64_t>(queue), owner);
                return;
            }
        }

        // Only a completely clean fixed arena may obtain the dynamic P8
        // lease.  Claiming publishes generation g+1 to forward_control
        // before CLAIMING(g+1) -> ACTIVE(g+1); a later executor phase has one
        // atomically comparable ticket and never derives ownership from C8
        // identity fields.
        if (!TryAcquireArena()) {
            WriteResult(0U, kP8AdmissionArenaBusy, -1,
                        static_cast<int64_t>(latchResult_->observed_tag));
            return;
        }
        kvaf_fused_get::KvafP8ForwardControl reloaded{};
        if (!kvaf_fused_get::LoadKvafP8ForwardControlFromGm(
                forwardControlGm_, &reloaded) ||
            !kvaf_fused_get::IsKvafP8ForwardControlValid(reloaded) ||
            reloaded.forward_epoch != static_cast<uint64_t>(expected_epoch) ||
            reloaded.graph_replay_generation !=
                static_cast<uint64_t>(expected_replay) ||
            reloaded.layer_control_index !=
                static_cast<uint32_t>(expected_layer) ||
            reloaded.phase != kvaf_fused_get::kKvafP8ForwardPhaseAdmitPost ||
            kvaf_fused_get::GetKvafP8ForwardLeaseGeneration(reloaded) !=
                a5_p8_arena::GenerationOf(activeArenaTag_)) {
            FailClosed(kP8AdmissionIdentityOrPhaseMismatch, -1, -1);
            return;
        }
        WriteResult(1U, kP8AdmissionAdmitted, -1, 0);
        DataSyncBarrier<MemDsbT::DDR>();
    }

private:
    __aicore__ inline bool IsAllZero(__gm__ int64_t* words,
                                     uint32_t word_count) const
    {
        if (words == nullptr) {
            return false;
        }
        for (uint32_t word = 0U; word < word_count; ++word) {
            if (ReadGmByPassDCache<int64_t>(words + word) != 0) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void WriteResult(uint32_t success, uint32_t reason,
                                       int64_t index, int64_t observed) const
    {
        WriteGmByPassDCache<int64_t>(result_ + 0U,
                                     static_cast<int64_t>(success));
        WriteGmByPassDCache<int64_t>(result_ + 1U,
                                     static_cast<int64_t>(reason));
        WriteGmByPassDCache<int64_t>(result_ + 2U, index);
        WriteGmByPassDCache<int64_t>(result_ + 3U, observed);
    }

    __aicore__ inline bool QuarantineGenerationWrap()
    {
        latchResult_->success = 0U;
        latchResult_->observed_tag = 0U;
        set_flag(PIPE_S, PIPE_V, simtEvent_);
        wait_flag(PIPE_S, PIPE_V, simtEvent_);
        AscendC::Simt::VF_CALL<a5_p8_arena::QuarantineGenerationWrapSimtService>(
            AscendC::Simt::Dim3{1U, 1U, 1U}, arenaHealthGm_, initialArenaTag_,
            latchResult_);
        set_flag(PIPE_V, PIPE_S, simtEvent_);
        wait_flag(PIPE_V, PIPE_S, simtEvent_);
        DataSyncBarrier<MemDsbT::DDR>();
        return latchResult_->success != 0U;
    }

    __aicore__ inline bool FinishAdmission(bool healthy)
    {
        latchResult_->success = 0U;
        latchResult_->observed_tag = 0U;
        set_flag(PIPE_S, PIPE_V, simtEvent_);
        wait_flag(PIPE_S, PIPE_V, simtEvent_);
        AscendC::Simt::VF_CALL<a5_p8_arena::FinishAdmissionSimtService>(
            AscendC::Simt::Dim3{1U, 1U, 1U}, arenaHealthGm_, claimArenaTag_,
            healthy, latchResult_);
        set_flag(PIPE_V, PIPE_S, simtEvent_);
        wait_flag(PIPE_V, PIPE_S, simtEvent_);
        DataSyncBarrier<MemDsbT::DDR>();
        return latchResult_->success != 0U;
    }

    __aicore__ inline bool PoisonActiveArena()
    {
        latchResult_->success = 0U;
        latchResult_->observed_tag = 0U;
        set_flag(PIPE_S, PIPE_V, simtEvent_);
        wait_flag(PIPE_S, PIPE_V, simtEvent_);
        AscendC::Simt::VF_CALL<a5_p8_arena::PoisonActiveLeaseSimtService>(
            AscendC::Simt::Dim3{1U, 1U, 1U}, arenaHealthGm_, activeArenaTag_,
            latchResult_);
        set_flag(PIPE_V, PIPE_S, simtEvent_);
        wait_flag(PIPE_V, PIPE_S, simtEvent_);
        DataSyncBarrier<MemDsbT::DDR>();
        return latchResult_->success != 0U;
    }

    __aicore__ inline bool TryAcquireArena()
    {
        latchResult_->success = 0U;
        latchResult_->observed_tag = 0U;
        set_flag(PIPE_S, PIPE_V, simtEvent_);
        wait_flag(PIPE_S, PIPE_V, simtEvent_);
        AscendC::Simt::VF_CALL<a5_p8_arena::ClaimAdmissionSimtService>(
            AscendC::Simt::Dim3{1U, 1U, 1U}, arenaHealthGm_, latchResult_);
        set_flag(PIPE_V, PIPE_S, simtEvent_);
        wait_flag(PIPE_V, PIPE_S, simtEvent_);
        DataSyncBarrier<MemDsbT::DDR>();
        if (latchResult_->success == 0U) {
            return false;
        }
        claimArenaTag_ = latchResult_->observed_tag;
        const uint32_t generation = a5_p8_arena::GenerationOf(claimArenaTag_);
        if (!kvaf_fused_get::StoreKvafP8ForwardLeaseGenerationToGm(
                forwardControlGm_, generation)) {
            (void)FinishAdmission(false);
            return false;
        }
        if (!FinishAdmission(true)) {
            return false;
        }
        activeArenaTag_ = a5_p8_arena::MakeTag(
            generation, a5_p8_arena::kArenaActive);
        hasArenaLease_ = true;
        return true;
    }

    __aicore__ inline void FailClosed(uint32_t reason, int64_t index,
                                      int64_t observed)
    {
        // Scan-first admission has no lease yet.  Take and immediately poison
        // the tagged lease so an unsafe fixed arena cannot be silently
        // retried; if another graph won meanwhile, its tag is left untouched.
        if (!hasArenaLease_ && !TryAcquireArena()) {
            WriteResult(0U, kP8AdmissionArenaBusy, index, observed);
            DataSyncBarrier<MemDsbT::DDR>();
            return;
        }
        const bool poisoned = PoisonActiveArena();
        kvaf_fused_get::KvafP8ForwardControl reloaded{};
        const bool same_ticket = poisoned &&
            kvaf_fused_get::LoadKvafP8ForwardControlFromGm(
                forwardControlGm_, &reloaded) &&
            kvaf_fused_get::GetKvafP8ForwardLeaseGeneration(reloaded) ==
                a5_p8_arena::GenerationOf(activeArenaTag_);
        bool abort_published = false;
        if (same_ticket) {
            abort_published = kvaf_fused_get::MarkKvafP8ForwardAbortFromAicore(
                forwardControlGm_, abortResult_, simtEvent_);
        }
        WriteResult(0U,
                    abort_published ? reason : kP8AdmissionAbortPublishFailed,
                    index, observed);
        DataSyncBarrier<MemDsbT::DDR>();
    }

    TPipe* pipe_ = nullptr;
    const A5P8ArenaAdmissionGateTilingData* tiling_ = nullptr;
    TBuf<TPosition::VECCALC> latchBuffer_;
    TBuf<TPosition::VECCALC> abortBuffer_;
    __ubuf__ P8ArenaHealthLatchResult* latchResult_ = nullptr;
    __ubuf__ kvaf_fused_get::KvafP8ForwardAbortResult* abortResult_ = nullptr;
    event_t simtEvent_ = static_cast<event_t>(EVENT_ID0);

    __gm__ kvaf_fused_get::KvafP8ForwardControl* forwardControlGm_ = nullptr;
    __gm__ int32_t* arenaHealthGm_ = nullptr;
    __gm__ const kvaf_fused_get::KvafStreamingBatchGetControl* controlGm_ =
        nullptr;
    __gm__ kvaf_fused_get::KvafP8WaveRecord* waveRecords_ = nullptr;
    __gm__ kvaf_fused_get::KvafP8PrivateRecord* privateRecords_ = nullptr;
    __gm__ int64_t* expectedIdentity_ = nullptr;
    __gm__ int64_t* result_ = nullptr;
    uint32_t initialArenaTag_ = 0U;
    uint32_t claimArenaTag_ = 0U;
    uint32_t activeArenaTag_ = 0U;
    bool hasArenaLease_ = false;
    bool enabled_ = false;
};
}  // namespace

extern "C" __global__ __aicore__ void a5_p8_arena_admission_gate(
    GM_ADDR forward_control, GM_ADDR p8_arena_health,
    GM_ADDR fused_get_control, GM_ADDR p8_wave_records,
    GM_ADDR p8_private_records, GM_ADDR expected_identity,
    GM_ADDR admission_result, GM_ADDR forward_control_ref,
    GM_ADDR p8_arena_health_ref, GM_ADDR admission_result_ref,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)forward_control_ref;
    (void)p8_arena_health_ref;
    (void)admission_result_ref;
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5P8ArenaAdmissionGateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    A5P8ArenaAdmissionGateKernel op(&pipe, &tilingData);
    op.Init(forward_control, p8_arena_health, fused_get_control,
            p8_wave_records, p8_private_records, expected_identity,
            admission_result);
    op.Process();
}
