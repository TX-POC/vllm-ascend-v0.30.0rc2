// Compile-disabled finalizer tail for the source-derived P9 C8 operator.
//
// The P9 source overlay appends this macro only after each *official*
// `indiceOutGm` writeback and its matching `SetFlag<MTE3_V>`.  It provides a
// deliberately narrow, checkable place for the future Stage-B terminalizer:
// final Top-K indices are already in GM and the official MTE3->V dependency
// has already been restored.  The default emits no instructions, touches no
// graph ABI, and therefore cannot alter the original C8 Top-K algorithm,
// index writeback, or scheduling.

#ifndef VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER_H_
#define VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER_H_

// Keep the actual terminalizer in the CANN compile closure even while this
// insertion remains disabled.  That makes domain/API failures deterministic
// at build time instead of discovering them only in the later ABI enablement.
#include "a5_c8_exact_speculative_resident_finalizer.h"

#ifndef A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER
#define A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER 0
#endif

#if A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER != 0
#error "The active P9 Stage-B finalizer requires the complete graph ABI."
#endif

// Args deliberately remain unevaluated in the only supported configuration.
// In particular, this must not make an invalid-row branch observe or modify
// score/Top-K buffers that official C8 otherwise leaves untouched.
#define A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER_TAIL(...) \
    do {                                                           \
    } while (0)

#endif  // VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER_H_
