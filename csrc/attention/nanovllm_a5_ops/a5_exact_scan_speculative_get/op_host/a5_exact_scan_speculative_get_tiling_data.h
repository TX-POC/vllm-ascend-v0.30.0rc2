// Host-side CANN tiling schema for A5ExactScanSpeculativeGet.
//
// The device kernel intentionally consumes the compact POD declared in
// ../op_kernel/a5_exact_scan_speculative_get_tiling.h.  CANN's ACLNN package
// builder, however, discovers the raw tiling capacity only through a
// BEGIN_TILING_DATA_DEF registration.  Keep the field sequence here exactly
// identical to the device POD and let the host implementation reject any
// serialized-size drift before it launches the kernel.

#ifndef A5_EXACT_SCAN_SPECULATIVE_GET_HOST_TILING_DATA_H
#define A5_EXACT_SCAN_SPECULATIVE_GET_HOST_TILING_DATA_H

#include <cstdint>

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(A5ExactScanSpeculativeGetHostTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rowCount)
TILING_DATA_FIELD_DEF(uint32_t, sourceCapacity)
TILING_DATA_FIELD_DEF(uint32_t, poolSize)
TILING_DATA_FIELD_DEF(uint32_t, kvObjectCapacity)
TILING_DATA_FIELD_DEF(uint32_t, privateArenaCapacity)
TILING_DATA_FIELD_DEF(uint32_t, tileReceiptCount)
TILING_DATA_FIELD_DEF(uint32_t, scoreWorkspaceWords)
TILING_DATA_FIELD_DEF(uint32_t, rowsPerEntry)
TILING_DATA_FIELD_DEF(uint32_t, candidateCapacityPerProducer)
TILING_DATA_FIELD_DEF(uint32_t, candidateSeedWords)
TILING_DATA_FIELD_DEF(uint32_t, tableCapacity)
TILING_DATA_FIELD_DEF(uint32_t, outputRows)
TILING_DATA_FIELD_DEF(uint32_t, controlWords)
TILING_DATA_FIELD_DEF(uint32_t, producerCount)
TILING_DATA_FIELD_DEF(uint32_t, getLaneCount)
TILING_DATA_FIELD_DEF(uint32_t, ringSlots)
TILING_DATA_FIELD_DEF(uint32_t, recordsPerRingSlot)
TILING_DATA_FIELD_DEF(uint32_t, getBootstrapRecords)
TILING_DATA_FIELD_DEF(uint32_t, getWaveRecords)
TILING_DATA_FIELD_DEF(uint32_t, pipelineWordsPerRing)
TILING_DATA_FIELD_DEF(uint32_t, reserved)
TILING_DATA_FIELD_DEF(uint32_t, schemaAlignmentPadding)
END_TILING_DATA_DEF

// This macro is intentionally not replaced with a hand-written TilingDef.
// The CANN package builder queries the macro-recorded schema to emit the
// direct-ACLNN raw-tiling capacity.  A manual registration is visible too
// late and produces a zero-capacity TilingData buffer at runtime.
REGISTER_TILING_DATA_CLASS(A5ExactScanSpeculativeGet,
                           A5ExactScanSpeculativeGetHostTilingData)

}  // namespace optiling

#endif  // A5_EXACT_SCAN_SPECULATIVE_GET_HOST_TILING_DATA_H
