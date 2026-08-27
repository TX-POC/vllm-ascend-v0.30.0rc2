// Build-time mode for the isolated P3 streaming-LI probe.
//
// The normal source tree defaults to the real streaming hook.  The isolated
// build script may replace this generated staging header with 0 in order to
// build a *true* no-hook control: the C8 score/TopK closure and the 21-input /
// 9-output ABI remain, while no KVAF helper, SIMT service, extra hook UB, or
// hook-only event/barrier is compiled into the target binary.
//
// This is deliberately compile-time rather than a runtime branch.  A runtime
// branch would still retain the Direct-HCCP/SIMT code and UB reservations, so
// it could not distinguish a C8-clone issue from a fused-hook issue.
#ifndef A5_STREAMING_LI_GET_ENABLE_HOOK
#define A5_STREAMING_LI_GET_ENABLE_HOOK 1
#endif
