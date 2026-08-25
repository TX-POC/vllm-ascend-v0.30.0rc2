/**
 * First device-side ExactScan + SpeculativePost prototype.
 *
 * This kernel deliberately makes the 31+1 MIX role layout observable before
 * any Direct-HCCP ABI is introduced:
 *
 *   groups 0..30 : 31 LI/ExactScan groups (1 AIC + 2 AIV each)
 *   group 31     : AIC idle; its two AIV lanes own future PostOnly work
 *
 * `GetBlockIdx()` is a logical block number for this invocation, not a
 * physical AIV identifier.  The trace therefore proves logical role
 * isolation only; it makes no physical-affinity claim.
 */

#include "kernel_operator.h"

#include "a5_exact_scan_speculative_post_probe_tiling.h"

namespace {
using namespace AscendC;

constexpr uint32_t kLiAivCount = 62;
constexpr uint32_t kPostAivBegin = kLiAivCount;
constexpr uint32_t kPostAivCount = 2;
constexpr uint32_t kAivTraceBase = 0;
constexpr uint32_t kAicTraceBase = 64;
constexpr int32_t kLiAivTraceBase = 1000;
constexpr int32_t kPostAivTraceBase = 2000;
constexpr int32_t kLiAicTraceBase = 3000;
constexpr int32_t kIdleAicTrace = 4000;
constexpr int32_t kPostMarkerXor = 0x5A5A5A5A;

class A5ExactScanSpeculativePostProbeKernel {
public:
    __aicore__ inline explicit A5ExactScanSpeculativePostProbeKernel(
        const A5ExactScanSpeculativePostProbeTilingData* tiling)
        : tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR scanValues,
        GM_ADDR postSeed,
        GM_ADDR liChecksums,
        GM_ADDR postMarkers,
        GM_ADDR roleTrace)
    {
        scanValuesGm_.SetGlobalBuffer((__gm__ int32_t*)scanValues);
        postSeedGm_.SetGlobalBuffer((__gm__ int32_t*)postSeed);
        liChecksumsGm_.SetGlobalBuffer((__gm__ int64_t*)liChecksums);
        postMarkersGm_.SetGlobalBuffer((__gm__ int32_t*)postMarkers);
        roleTraceGm_.SetGlobalBuffer((__gm__ int32_t*)roleTrace);
    }

    __aicore__ inline void Process()
    {
        const uint32_t logicalBlock = GetBlockIdx();
        if ASCEND_IS_AIV {
            ProcessAiv(logicalBlock);
        } else {
            ProcessAic(logicalBlock);
        }
    }

private:
    __aicore__ inline void ProcessAiv(uint32_t logicalAiv)
    {
        if (logicalAiv < tiling_->liAivCount) {
            int64_t checksum = 0;
            for (uint32_t valueIndex = logicalAiv;
                 valueIndex < tiling_->scanValueCount;
                 valueIndex += tiling_->liAivCount) {
                checksum += static_cast<int64_t>(
                    scanValuesGm_.GetValue(valueIndex));
            }
            liChecksumsGm_.SetValue(logicalAiv, checksum);
            roleTraceGm_.SetValue(
                kAivTraceBase + logicalAiv,
                kLiAivTraceBase + static_cast<int32_t>(logicalAiv));
            return;
        }

        const uint32_t postLane = logicalAiv - kPostAivBegin;
        if (postLane < tiling_->postAivCount) {
            // The future implementation replaces this local, independent
            // marker with Direct-HCCP PostOnly.  It intentionally has no
            // dependency on the unfinished LI lanes in this P0 prototype.
            postMarkersGm_.SetValue(
                postLane,
                postSeedGm_.GetValue(postLane) ^ kPostMarkerXor);
            roleTraceGm_.SetValue(
                kAivTraceBase + logicalAiv,
                kPostAivTraceBase + static_cast<int32_t>(postLane));
        }
    }

    __aicore__ inline void ProcessAic(uint32_t logicalAic)
    {
        if (logicalAic < tiling_->liGroupCount) {
            roleTraceGm_.SetValue(
                kAicTraceBase + logicalAic,
                kLiAicTraceBase + static_cast<int32_t>(logicalAic));
            return;
        }
        if (logicalAic == tiling_->liGroupCount) {
            // Group 31 has no cube-side task in P0.  Keeping this explicit is
            // important: it prevents a later LI integration from entering its
            // 32-group cross-core protocol on the Post group by accident.
            roleTraceGm_.SetValue(
                kAicTraceBase + logicalAic, kIdleAicTrace);
        }
    }

private:
    const A5ExactScanSpeculativePostProbeTilingData* tiling_;
    GlobalTensor<int32_t> scanValuesGm_;
    GlobalTensor<int32_t> postSeedGm_;
    GlobalTensor<int64_t> liChecksumsGm_;
    GlobalTensor<int32_t> postMarkersGm_;
    GlobalTensor<int32_t> roleTraceGm_;
};
}  // namespace

extern "C" __global__ __aicore__ void a5_exact_scan_speculative_post_probe(
    GM_ADDR scanValues,
    GM_ADDR postSeed,
    GM_ADDR liChecksums,
    GM_ADDR postMarkers,
    GM_ADDR roleTrace,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(A5ExactScanSpeculativePostProbeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    A5ExactScanSpeculativePostProbeKernel op(&tilingData);
    op.Init(scanValues, postSeed, liChecksums, postMarkers, roleTrace);
    op.Process();
}
