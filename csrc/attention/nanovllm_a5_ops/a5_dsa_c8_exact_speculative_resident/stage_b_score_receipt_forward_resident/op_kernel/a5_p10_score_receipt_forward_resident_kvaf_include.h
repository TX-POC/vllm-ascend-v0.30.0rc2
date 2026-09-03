#ifndef A5_P10_SCORE_RECEIPT_FORWARD_RESIDENT_KVAF_INCLUDE_H
#define A5_P10_SCORE_RECEIPT_FORWARD_RESIDENT_KVAF_INCLUDE_H

// The isolated Stage-B builder copies this four-file device closure only from
// the exact KVAF source revision named in P10_STAGE_PROVENANCE. Never let a
// caller redirect this macro to an installed, neighbouring, or stale KVAF
// header: its Direct-HCCP structs are part of the device ABI.
#ifdef P10_STAGE_B_KVAF_FORWARD_RESIDENT_DEVICE_HEADER
#error "P10 Stage B must use its provenance-staged KVAF device header"
#endif
#define P10_STAGE_B_KVAF_FORWARD_RESIDENT_DEVICE_HEADER \
  "kvaf_forward_resident/kvaf_forward_resident_get_device.h"

#endif  // A5_P10_SCORE_RECEIPT_FORWARD_RESIDENT_KVAF_INCLUDE_H
