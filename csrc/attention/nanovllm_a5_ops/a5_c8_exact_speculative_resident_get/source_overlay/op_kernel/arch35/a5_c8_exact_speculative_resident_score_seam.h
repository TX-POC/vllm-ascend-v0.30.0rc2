// Active P9 score seam for the source-derived C8 binary.
//
// This is deliberately a *member call* rather than a free closure or a
// sidecar launch: QLIVector owns the existing C8 UB scratch, observes the
// completed uint16 score tile, and hands bounded CandidateSeed batches to its
// dynamic Direct-HCCP executor.  The graph-visible runtime-enable word still
// makes `runtime_enable=0` a no-observation C8 control path.

#ifndef VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM_H_
#define VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM_H_

#ifndef A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM
#define A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM 1
#endif

#include "a5_c8_exact_speculative_resident_contract.h"
#include "a5_c8_exact_speculative_resident_score_hook.h"
#include "a5_c8_exact_speculative_resident_post_executor.h"
#include "a5_c8_exact_speculative_resident_stage_b_finalizer.h"

#if A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM != 1
#error "P9 ships only as the complete active resident-get ABI."
#endif

// This macro sits immediately after the completed official score reduction
// and before the normal score-GM DataCopyPad.  It never writes score UB or
// Top-K storage; the member implementation only reads `out_base` and compacts
// a bounded candidate receipt.  Posting is intentionally deferred to the
// second macro below, after C8 has published the score tile to GM.
#define A5_C8_EXACT_SPECULATIVE_RESIDENT_OBSERVE_SCORE_SEAM( \
    self, out_base, info, cur_aiv_s1_idx, cur_aiv_s1_proc_num, cur_s2_idx) \
    do {                                                                  \
        (self)->P9ObserveScoreSeam((out_base), (info),                    \
                                   (cur_aiv_s1_idx),                       \
                                   (cur_aiv_s1_proc_num), (cur_s2_idx));   \
    } while (0)

// This is injected immediately after C8's original score-GM DataCopyPad and
// its matching V<-MTE3 acknowledgement.  It does not touch the score UB or
// any Top-K event.  It merely lets the dynamic queue executor retire a prior
// terminal CQ and post a fully compacted seed batch.
#define A5_C8_EXACT_SPECULATIVE_RESIDENT_POST_AFTER_SCORE_PUBLICATION(self) \
    do {                                                                     \
        (self)->P9PostAtScoreBoundary();                                     \
    } while (0)

#endif  // VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM_H_
