// Active finalizer tail for the source-derived P9 C8 operator.
//
// The P9 source overlay appends this macro only after each *official*
// `WaitFlag<V_MTE3>(TOPK_V_MTE3_EVENT)` and before its unchanged
// `indiceOutGm` writeback.  It provides a deliberately narrow, checkable
// place for the Stage-B terminalizer while V owns the completed local Top-K
// buffer.  The original MTE3->V event is neither waited nor re-signaled by
// P9, so the next untouched official Top-K row keeps its original token.

#ifndef VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER_H_
#define VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER_H_

#include "a5_c8_exact_speculative_resident_finalizer.h"

#ifndef A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER
#define A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER 1
#endif

#if A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER != 1
#error "P9 ships only with its complete Stage-B resident-get ABI."
#endif

// The tail is before the unchanged official sparse-index DMA.  The member
// method reads `indicesOutLocal_` while its owner is still V, then returns
// before MTE3 is handed that buffer.  Invalid rows naturally contain the
// original -1 indices.  The `row` is only the P9 destination-row identity;
// no output-GM offset or TOPK event is consumed here.
#define A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER_TAIL( \
    self, row)                                                     \
    do {                                                          \
        (self)->P9FinalizeExactTopKRow(                           \
            static_cast<uint32_t>(row));                           \
    } while (0)

#endif  // VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER_H_
