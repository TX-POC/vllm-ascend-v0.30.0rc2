/**
 * Tiling contract for the first ExactScan + SpeculativePost prototype.
 *
 * This is deliberately fixed to the 32 AIC / 64 AIV geometry of Ascend 950DT.
 * It describes logical roles within one MIX invocation; it is not a physical
 * AIV-ID affinity contract.
 */

#ifndef A5_EXACT_SCAN_SPECULATIVE_POST_PROBE_TILING_H
#define A5_EXACT_SCAN_SPECULATIVE_POST_PROBE_TILING_H

#include <cstdint>

struct A5ExactScanSpeculativePostProbeTilingData {
    uint32_t liGroupCount;
    uint32_t liAivCount;
    uint32_t postAivCount;
    uint32_t scanValueCount;
};

#endif  // A5_EXACT_SCAN_SPECULATIVE_POST_PROBE_TILING_H
