// Compile-disabled P9 seam marker for the source-derived C8 binary.
//
// The first P9 OPP closure must prove that the official score path can be
// cloned, renamed, and observed at its exact reduction/publication seam
// without changing `ProcessTopK`.  It deliberately has no transport code:
// an enabled implementation must add the graph ABI and Direct-HCCP executor
// together, rather than accidentally enabling a partial closure.

#ifndef VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM_H_
#define VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM_H_

#ifndef A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM
#define A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM 0
#endif

#if A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM != 0
#error "The active P9 score seam requires the complete resident-get ABI and executor."
#endif

// This macro sits immediately after the completed official score reduction
// and before the normal score-GM DataCopyPad.  With the only supported value
// (zero), it generates no instructions and cannot alter score UB, score GM,
// Top-K storage, index ordering, or the C8 schedule.
#define A5_C8_EXACT_SPECULATIVE_RESIDENT_OBSERVE_SCORE_SEAM(...) \
    do {                                                        \
    } while (0)

#endif  // VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM_H_
