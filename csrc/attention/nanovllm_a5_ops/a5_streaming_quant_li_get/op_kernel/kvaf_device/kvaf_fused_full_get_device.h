#ifndef KVAF_FUSED_FULL_GET_DEVICE_H_
#define KVAF_FUSED_FULL_GET_DEVICE_H_

// Device-side reuse boundary for a fused AIV operator.
//
// ExecuteBatchGet is currently built as a standalone AIV kernel in
// execute_batch_get_aiv.asc.  A MIX kernel cannot launch that global kernel
// as a child.  This header instead includes the same single-source device
// implementation with its standalone entry point suppressed.  A fused
// operator can either call the compatibility helper
// `RunKvafDenseBatchGetCore256(args, active_request_count, core)`, or use the
// explicit transaction sequence below:
//
//   KvafDenseBatchGetTransaction txn;
//   BeginKvafDenseBatchGetPostOnly(args, count, core, &txn);
//   // Run ExactScan tiles and occasionally TryRetireKvafDenseBatchGet(...).
//   FinishKvafDenseBatchGet(args, &txn);
//
// PostOnly has exactly the historical validation/WQE/SQ-doorbell semantics;
// only the CQ wait moves outside it.  There is deliberately one transaction
// per pre-bound queue: a caller must retire or finish a wave before posting a
// later wave on the same queue.  The existing queue ABI has no multi-flight
// CQ ledger, so relaxing that rule would make descriptor/WQE order ambiguous.
// Keeping one source of truth is deliberate: the Direct-HCCP WQE and CQ
// protocol must not drift between the normal BatchGet path and the fused
// path.
//
// The caller owns `core`: it is a KVAF transport-worker ordinal in
// [0, args.block_count), not a public physical-AIV identifier.  P1 uses an
// AIV-only task with lanes 0 and 1 as its two Get workers; the remaining task
// lanes run ExactScan producers.
//
// Streaming P2 uses the separate helpers below instead. There are no fixed
// Get-owner AIVs: every LI producer atomically leases an idle Direct-HCCP
// queue from an int32 queue-pool, posts one wave, continues scanning score
// tiles, probes completion, then releases the queue only after terminal CQ
// retirement. The pool has at most KVAF_AIV_BATCH_GET_MAX_AIV_CORES entries;
// it is a Direct-HCCP transport limit, not a count of LI AIV work items.

#ifndef KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED
#define KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED 1
#include "execute_batch_get_aiv.asc"
#undef KVAF_FUSED_FULL_GET_DEVICE_EMBEDDED
#endif

#include "simt_api/cpp/kernel_simt_warp_level_intf.h"

namespace kvaf_fused_get {

// Caller-local lease state. It is intentionally not stored in HBM: the
// globally shared state is only the int32 owner table referenced by the
// graph-stable KvafStreamingBatchGetControl. `owner_tag` is positive and
// equals 1 + the caller's stable LI producer id.
enum KvafStreamingBatchGetLeaseState : uint32_t {
  kStreamingBatchGetLeaseIdle = 0U,
  kStreamingBatchGetLeaseHeld = 1U,
  kStreamingBatchGetLeasePosted = 2U,
  kStreamingBatchGetLeaseTerminal = 3U,
};

struct KvafStreamingBatchGetLease {
  uint32_t queue_id;
  int32_t owner_tag;
  uint32_t state;
  uint32_t reserved0;
};

// UB hand-off record for the AIC-facing acquire/release wrappers. The raw
// atomics live in a one-thread SIMT service because CANN declares
// asc_atomic_cas as a SIMT callee in a MIX kernel. An AIC scalar caller must
// use the wrappers below rather than calling the raw SIMT helper directly.
struct KvafStreamingBatchGetLeaseResult {
  uint32_t queue_id;
  int32_t owner_tag;
  uint32_t success;
  uint32_t reserved0;
};
static_assert(sizeof(KvafStreamingBatchGetLeaseResult) == 16U,
              "streaming queue lease UB hand-off ABI changed");

// A caller-local wave combines the leased queue-specific argument view with
// the existing verified Direct-HCCP transaction. It is one active wave per
// lease; the existing SQ/CQ ABI deliberately does not support multiple
// in-flight waves on one queue.
struct KvafStreamingBatchGetWave {
  KvafStreamingBatchGetLease lease;
  KvafExecuteBatchGetKernelArgs queue_args;
  KvafDenseBatchGetTransaction transaction;
};

// This is deliberately caller-local ready information. It is valid only
// after Finish has made the CQ terminal; callers may then publish it to their
// own bounded source-map/ready arena. No helper here writes a persistent HBM
// block table, cache slot, or resident mapping.
struct KvafStreamingBatchGetReady {
  uint32_t queue_id;
  int32_t owner_tag;
  uint32_t active_request_count;
  uint32_t ready;
  uint64_t result;
};

__simt_callee__ inline bool IsKvafStreamingBatchGetControlValidForLease(
    const KvafStreamingBatchGetControl& control) {
  return control.magic == KVAF_STREAMING_BATCH_GET_CONTROL_MAGIC &&
         control.version == KVAF_STREAMING_BATCH_GET_CONTROL_VERSION &&
         control.channel_entity != 0U && control.queue_state != 0U &&
         control.core_status != 0U && control.output != 0U &&
         control.queue_leases != 0U && control.descriptor_scratch != 0U &&
         control.descriptor_templates != 0U &&
         control.queue_count != 0U &&
         control.queue_count <= KVAF_AIV_BATCH_GET_MAX_AIV_CORES &&
         control.descriptor_template_count == control.queue_count &&
         control.output_rows != 0U &&
         control.private_output_stride_bytes != 0U &&
         control.output_offset_bytes == 0U &&
         control.private_output_stride_bytes ==
             (control.output_rows /
              control.queue_count) * KVAF_AIV_BATCH_GET_WQE_DATA_BYTES &&
         control.output_rows % control.queue_count == 0U &&
         control.jettys_per_core == 1U &&
         control.block_bytes == KVAF_AIV_BATCH_GET_WQE_DATA_BYTES;
}

// The AIC scalar path cannot call a SIMT callee directly. Keep the same
// validation at that boundary rather than relying on a compiler-specific
// cross-pipeline call convention.
__aicore__ inline bool IsKvafStreamingBatchGetControlValidForAicore(
    const KvafStreamingBatchGetControl& control) {
  return control.magic == KVAF_STREAMING_BATCH_GET_CONTROL_MAGIC &&
         control.version == KVAF_STREAMING_BATCH_GET_CONTROL_VERSION &&
         control.channel_entity != 0U && control.queue_state != 0U &&
         control.core_status != 0U && control.output != 0U &&
         control.queue_leases != 0U && control.descriptor_scratch != 0U &&
         control.descriptor_templates != 0U &&
         control.queue_count != 0U &&
         control.queue_count <= KVAF_AIV_BATCH_GET_MAX_AIV_CORES &&
         control.descriptor_template_count == control.queue_count &&
         control.output_rows != 0U &&
         control.private_output_stride_bytes != 0U &&
         control.output_offset_bytes == 0U &&
         control.private_output_stride_bytes ==
             (control.output_rows /
              control.queue_count) * KVAF_AIV_BATCH_GET_WQE_DATA_BYTES &&
         control.output_rows % control.queue_count == 0U &&
         control.jettys_per_core == 1U &&
         control.block_bytes == KVAF_AIV_BATCH_GET_WQE_DATA_BYTES;
}

// TryAcquire is a SIMT-safe, non-blocking claim. The producer scans every
// queue at most once starting from `start_hint`; false means the pool is full
// and the LI kernel must retain/skip this speculative candidate rather than
// spin. `producer_id` is not a physical core id and may be any stable LI task
// ordinal below INT32_MAX - 1.
__simt_callee__ inline bool TryAcquireKvafStreamingBatchGetQueueFromSimt(
    const KvafStreamingBatchGetControl& control, uint32_t producer_id,
    uint32_t start_hint, KvafStreamingBatchGetLease* lease) {
  if (lease == nullptr ||
      !IsKvafStreamingBatchGetControlValidForLease(control) ||
      producer_id >= 0x7ffffffeU) {
    return false;
  }
  const int32_t owner_tag = static_cast<int32_t>(producer_id + 1U);
  const uint32_t queue_count = static_cast<uint32_t>(control.queue_count);
  const uint32_t first_queue = start_hint % queue_count;
  __gm__ int32_t* const owners =
      reinterpret_cast<__gm__ int32_t*>(control.queue_leases);
  for (uint32_t probe = 0U; probe < queue_count; ++probe) {
    const uint32_t queue_id = (first_queue + probe) % queue_count;
    if (asc_atomic_cas(owners + queue_id, 0, owner_tag) == 0) {
      // Pair the release fence before a previous owner's CAS-to-zero with
      // this producer's first access to the queue's state and CQ memory.
      AscendC::Simt::ThreadFence();
      lease->queue_id = queue_id;
      lease->owner_tag = owner_tag;
      lease->state = kStreamingBatchGetLeaseHeld;
      lease->reserved0 = 0U;
      return true;
    }
  }
  return false;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafStreamingBatchGetAcquireSimtService(
    __gm__ const KvafStreamingBatchGetControl* control_gm,
    uint32_t producer_id, uint32_t start_hint,
    __ubuf__ KvafStreamingBatchGetLeaseResult* result) {
  if (result == nullptr || control_gm == nullptr) {
    return;
  }
  const KvafStreamingBatchGetControl control = *control_gm;
  KvafStreamingBatchGetLease lease{};
  if (TryAcquireKvafStreamingBatchGetQueueFromSimt(
          control, producer_id, start_hint, &lease)) {
    result->queue_id = lease.queue_id;
    result->owner_tag = lease.owner_tag;
    result->success = 1U;
  }
}

// Supported AIC-scalar entry point for LI's ProcessVec1/ProcessTopK-side
// hook. `lease_result` must be caller-owned UB storage. `simt_event` must be
// an event id that the including LI operator reserves exclusively for this
// acquire/release service; it must not overlap a live vector/MTE dependency.
// The VF_CALL + PIPE_S/PIPE_V hand-off is required: a direct AIC call to
// asc_atomic_cas is not supported by the CANN MIX compilation model.
__aicore__ inline bool TryAcquireKvafStreamingBatchGetQueueFromAicore(
    __gm__ const KvafStreamingBatchGetControl* control_gm,
    uint32_t producer_id, uint32_t start_hint,
    __ubuf__ KvafStreamingBatchGetLeaseResult* lease_result,
    event_t simt_event, KvafStreamingBatchGetLease* lease) {
  if (control_gm == nullptr || lease_result == nullptr || lease == nullptr) {
    return false;
  }
  lease_result->queue_id = 0U;
  lease_result->owner_tag = 0;
  lease_result->success = 0U;
  lease_result->reserved0 = 0U;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafStreamingBatchGetAcquireSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, control_gm, producer_id, start_hint,
      lease_result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (lease_result->success == 0U || lease_result->owner_tag <= 0) {
    return false;
  }
  lease->queue_id = lease_result->queue_id;
  lease->owner_tag = lease_result->owner_tag;
  lease->state = kStreamingBatchGetLeaseHeld;
  lease->reserved0 = 0U;
  return true;
}

// Return the exclusively owned per-queue descriptor record. ExactScan writes
// its already-dense candidate arrays and final private-output indices here
// before Begin. It must not write a persistent HBM block table/cache slot;
// this record only drives a speculative private-slot Get wave.
__aicore__ inline __gm__ KvafStreamingBatchGetDescriptorScratch*
GetKvafStreamingBatchGetDescriptorScratch(
    const KvafStreamingBatchGetControl& control, uint32_t queue_id) {
  if (!IsKvafStreamingBatchGetControlValidForAicore(control) ||
      queue_id >= control.queue_count) {
    return nullptr;
  }
  return reinterpret_cast<__gm__ KvafStreamingBatchGetDescriptorScratch*>(
      control.descriptor_scratch) + queue_id;
}

// Return the immutable, layer-private template populated by the cold host
// ConfigureStreamingBatchGetQueueTemplates lifecycle call. The active
// descriptor_scratch table remains shared by the lease pool; therefore an LI
// producer must copy this template only after it has acquired its queue.
__aicore__ inline __gm__ const KvafStreamingBatchGetDescriptorScratch*
GetKvafStreamingBatchGetDescriptorTemplate(
    const KvafStreamingBatchGetControl& control, uint32_t queue_id) {
  if (!IsKvafStreamingBatchGetControlValidForAicore(control) ||
      queue_id >= control.descriptor_template_count) {
    return nullptr;
  }
  return reinterpret_cast<__gm__ const KvafStreamingBatchGetDescriptorScratch*>(
      control.descriptor_templates) + queue_id;
}

// Load a queue's fixed template into caller-local storage. ExactScan then
// writes only the template-owned candidate arrays and active_request_count;
// Publish verifies that those graph-stable pointer/capacity fields were not
// substituted on the replay path.
__aicore__ inline bool LoadKvafStreamingBatchGetQueueTemplate(
    const KvafStreamingBatchGetControl& control,
    const KvafStreamingBatchGetLease& lease,
    KvafStreamingBatchGetDescriptorScratch* descriptor) {
  if (descriptor == nullptr || lease.state != kStreamingBatchGetLeaseHeld ||
      lease.owner_tag <= 0 || lease.queue_id >= control.queue_count) {
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (owners[lease.queue_id] != lease.owner_tag) {
    return false;
  }
  __gm__ const KvafStreamingBatchGetDescriptorScratch* const source =
      GetKvafStreamingBatchGetDescriptorTemplate(control, lease.queue_id);
  if (source == nullptr) {
    return false;
  }
  *descriptor = *source;
  // A configured template intentionally has active_request_count == 0. It
  // becomes a valid wave only after the producer has written its own compact
  // candidate arrays and sets a nonzero count before Publish.
  return descriptor->address_table != 0U && descriptor->valid_table != 0U &&
         descriptor->indices != 0U && descriptor->offsets != 0U &&
         descriptor->mask != 0U && descriptor->output_indices != 0U &&
         descriptor->table_capacity != 0U &&
         descriptor->request_capacity != 0U &&
         descriptor->request_capacity <= kDenseRequestCapacity &&
         descriptor->active_request_count == 0U;
}

// Publish one dense descriptor record while holding its lease.
// `output_indices` must already contain absolute rows in this queue's private
// output arena; Begin does no late remapping. The expected row base is
// q * private_output_rows. Taking the lease rather than a bare queue id keeps
// a non-owner from overwriting another producer's descriptor scratch.
// `mask` is retained in the scratch ABI for diagnostics/compatibility but is
// not consumed by the dense post helper: ExactScan has already compacted the
// wave, so active_request_count is its exact valid count.
__aicore__ inline bool PublishKvafStreamingBatchGetDescriptorScratch(
    const KvafStreamingBatchGetControl& control,
    const KvafStreamingBatchGetLease& lease,
    const KvafStreamingBatchGetDescriptorScratch& descriptor) {
  if (!IsKvafStreamingBatchGetControlValidForAicore(control) ||
      lease.state != kStreamingBatchGetLeaseHeld || lease.owner_tag <= 0 ||
      lease.queue_id >= control.queue_count || descriptor.address_table == 0U ||
      descriptor.valid_table == 0U || descriptor.indices == 0U ||
      descriptor.offsets == 0U || descriptor.output_indices == 0U ||
      descriptor.table_capacity == 0U || descriptor.request_capacity == 0U ||
      descriptor.request_capacity > kDenseRequestCapacity ||
      descriptor.active_request_count == 0U ||
      descriptor.active_request_count > descriptor.request_capacity) {
    return false;
  }
  const uint32_t queue_id = lease.queue_id;
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  if (owners[queue_id] != lease.owner_tag) {
    return false;
  }
  __gm__ const KvafStreamingBatchGetDescriptorScratch* const template_record =
      GetKvafStreamingBatchGetDescriptorTemplate(control, queue_id);
  if (template_record == nullptr || template_record->address_table == 0U ||
      template_record->valid_table == 0U || template_record->indices == 0U ||
      template_record->offsets == 0U || template_record->mask == 0U ||
      template_record->output_indices == 0U ||
      template_record->table_capacity != descriptor.table_capacity ||
      template_record->request_capacity != descriptor.request_capacity ||
      template_record->address_table != descriptor.address_table ||
      template_record->valid_table != descriptor.valid_table ||
      template_record->indices != descriptor.indices ||
      template_record->offsets != descriptor.offsets ||
      template_record->mask != descriptor.mask ||
      template_record->output_indices != descriptor.output_indices ||
      template_record->private_output_row_base !=
          descriptor.private_output_row_base) {
    return false;
  }
  const uint64_t private_output_rows =
      control.private_output_stride_bytes /
      KVAF_AIV_BATCH_GET_WQE_DATA_BYTES;
  const uint64_t expected_private_base =
      static_cast<uint64_t>(queue_id) * private_output_rows;
  if (private_output_rows == 0U ||
      expected_private_base > control.output_rows ||
      private_output_rows > control.output_rows - expected_private_base ||
      descriptor.private_output_row_base != expected_private_base) {
    return false;
  }
  __gm__ KvafStreamingBatchGetDescriptorScratch* const scratch =
      GetKvafStreamingBatchGetDescriptorScratch(control, queue_id);
  if (scratch == nullptr) {
    return false;
  }
  *scratch = descriptor;
  // The following Begin may hand the queue to Direct-HCCP immediately. Keep
  // descriptor publication ordered before its SQ WQE materialization.
  AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
  return true;
}

// The legacy Direct-HCCP primitive validates an output index only against
// the whole registered output. Streaming adds this queue-private range check
// before the doorbell. It is intentionally a bounded scan over the already
// compacted wave (at most kDenseRequestCapacity entries), not a cache-table
// write or a persistent mapping update.
__aicore__ inline bool ValidateKvafStreamingBatchGetPrivateOutputIndices(
    const KvafStreamingBatchGetControl& control, uint32_t queue_id,
    const KvafStreamingBatchGetDescriptorScratch& descriptor) {
  const uint64_t private_output_rows =
      control.private_output_stride_bytes /
      KVAF_AIV_BATCH_GET_WQE_DATA_BYTES;
  const uint64_t private_base = descriptor.private_output_row_base;
  if (private_output_rows == 0U ||
      private_base != static_cast<uint64_t>(queue_id) * private_output_rows ||
      private_base > control.output_rows ||
      private_output_rows > control.output_rows - private_base) {
    return false;
  }
  __gm__ const int32_t* const output_indices =
      reinterpret_cast<__gm__ const int32_t*>(descriptor.output_indices);
  for (uint32_t request = 0U; request < descriptor.active_request_count;
       ++request) {
    const int32_t output_index = output_indices[request];
    if (output_index < 0 ||
        static_cast<uint64_t>(output_index) < private_base ||
        static_cast<uint64_t>(output_index) >=
            private_base + private_output_rows) {
      return false;
    }
  }
  return true;
}

// Build the existing single-queue Direct-HCCP view without changing the
// normal BatchGet kernel ABI. Descriptor pointers/count are read from the
// queue-private scratch record; output/entity/state/status are derived from
// the graph-stable streaming control plus the acquired queue id.
__aicore__ inline bool BuildKvafStreamingBatchGetQueueArgs(
    const KvafStreamingBatchGetControl& control, uint32_t queue_id,
    KvafExecuteBatchGetKernelArgs* queue_args,
    uint32_t* active_request_count) {
  if (queue_args == nullptr ||
      !IsKvafStreamingBatchGetControlValidForAicore(control) ||
      queue_id >= control.queue_count || active_request_count == nullptr) {
    return false;
  }
  __gm__ const KvafStreamingBatchGetDescriptorScratch* const scratch =
      GetKvafStreamingBatchGetDescriptorScratch(control, queue_id);
  if (scratch == nullptr || scratch->address_table == 0U ||
      scratch->valid_table == 0U || scratch->indices == 0U ||
      scratch->offsets == 0U || scratch->output_indices == 0U ||
      scratch->table_capacity == 0U || scratch->request_capacity == 0U ||
      scratch->request_capacity > kDenseRequestCapacity ||
      scratch->active_request_count == 0U ||
      scratch->active_request_count > scratch->request_capacity ||
      control.output_rows > UINT32_MAX ||
      !ValidateKvafStreamingBatchGetPrivateOutputIndices(
          control, queue_id, *scratch)) {
    return false;
  }

  queue_args->address_table = scratch->address_table;
  queue_args->valid_table = scratch->valid_table;
  queue_args->indices = scratch->indices;
  queue_args->offsets = scratch->offsets;
  queue_args->mask = scratch->mask;
  queue_args->output_indices = scratch->output_indices;
  // Keep the full registration base to satisfy the unchanged Direct-HCCP
  // entity check. The descriptor's validated absolute output indices select
  // this queue's private slot arena inside that allocation.
  queue_args->output = control.output;
  queue_args->core_status =
      control.core_status + queue_id * sizeof(KvafBatchGetCoreStatus);
  queue_args->channel_entity =
      control.channel_entity + queue_id * sizeof(KvafBatchGetDirectUrmaEntity);
  queue_args->queue_state =
      control.queue_state + queue_id * sizeof(KvafBatchGetQueueState);
  queue_args->table_capacity = scratch->table_capacity;
  queue_args->request_capacity = scratch->request_capacity;
  queue_args->output_rows = static_cast<uint32_t>(control.output_rows);
  queue_args->block_bytes = static_cast<uint32_t>(control.block_bytes);
  queue_args->block_count = 1U;
  queue_args->jettys_per_core = 1U;
  *active_request_count = scratch->active_request_count;
  return true;
}

// Begin a previously acquired lease. On failure it leaves the queue lease
// held but marks the wave terminal, so the caller can release it immediately
// through ReleaseKvafStreamingBatchGetQueue without risking an in-flight CQ.
__aicore__ inline bool BeginKvafStreamingBatchGet(
    const KvafStreamingBatchGetControl& control,
    const KvafStreamingBatchGetLease& lease,
    KvafStreamingBatchGetWave* wave) {
  if (wave == nullptr) {
    return false;
  }
  wave->lease = lease;
  ResetDenseBatchGetTransaction(&wave->transaction);
  if (lease.state != kStreamingBatchGetLeaseHeld || lease.owner_tag <= 0 ||
      lease.queue_id >= control.queue_count) {
    wave->transaction.result = kInvalidArgs;
    wave->transaction.state = kDenseBatchGetFailed;
    wave->transaction.finalized = 1U;
    wave->lease.state = kStreamingBatchGetLeaseTerminal;
    return false;
  }
  __gm__ const int32_t* const owners =
      reinterpret_cast<__gm__ const int32_t*>(control.queue_leases);
  uint32_t active_request_count = 0U;
  if (owners[lease.queue_id] != lease.owner_tag ||
      !BuildKvafStreamingBatchGetQueueArgs(control, lease.queue_id,
                                            &wave->queue_args,
                                            &active_request_count)) {
    wave->transaction.result = kInvalidArgs;
    wave->transaction.state = kDenseBatchGetFailed;
    wave->transaction.finalized = 1U;
    wave->lease.state = kStreamingBatchGetLeaseTerminal;
    return false;
  }
  BeginKvafDenseBatchGetPostOnly(wave->queue_args, active_request_count, 0U,
                                 &wave->transaction);
  wave->lease.state =
      wave->transaction.state == kDenseBatchGetInFlight
          ? kStreamingBatchGetLeasePosted
          : kStreamingBatchGetLeaseTerminal;
  return wave->transaction.state == kDenseBatchGetInFlight;
}

// Non-blocking completion probe for a posted streaming wave. It preserves the
// normal Direct-HCCP CQ semantics and gives the LI kernel a tile boundary at
// which to publish a successful speculative payload as ready.
__aicore__ inline void TryRetireKvafStreamingBatchGet(
    KvafStreamingBatchGetWave* wave) {
  if (wave == nullptr || wave->lease.state != kStreamingBatchGetLeasePosted) {
    return;
  }
  TryRetireKvafDenseBatchGet(wave->queue_args, &wave->transaction);
  if (wave->transaction.state == kDenseBatchGetRetired ||
      wave->transaction.state == kDenseBatchGetFailed) {
    wave->lease.state = kStreamingBatchGetLeaseTerminal;
  }
}

// Finish is the terminal drain. Call this once no further LI tile work can
// hide the CQ wait, before releasing the lease. It is mathematically the same
// post/CQ path as BatchGet because it delegates to the existing transaction.
__aicore__ inline void FinishKvafStreamingBatchGet(
    KvafStreamingBatchGetWave* wave) {
  if (wave == nullptr ||
      (wave->lease.state != kStreamingBatchGetLeasePosted &&
       wave->lease.state != kStreamingBatchGetLeaseTerminal)) {
    return;
  }
  FinishKvafDenseBatchGet(wave->queue_args, &wave->transaction);
  wave->lease.state = kStreamingBatchGetLeaseTerminal;
}

// Return a ready record only after Finish has observed a terminal CQ and
// published the normal KVAF status. `ready=1` means the private output arena
// contains the dense speculative payload; a failed/timeout CQ returns false
// so the caller can leave this candidate for ExactTopK's guaranteed fallback.
__aicore__ inline bool GetKvafStreamingBatchGetReady(
    const KvafStreamingBatchGetWave& wave,
    KvafStreamingBatchGetReady* ready) {
  if (ready == nullptr || wave.lease.state != kStreamingBatchGetLeaseTerminal ||
      wave.transaction.finalized == 0U) {
    return false;
  }
  ready->queue_id = wave.lease.queue_id;
  ready->owner_tag = wave.lease.owner_tag;
  ready->active_request_count =
      static_cast<uint32_t>(wave.transaction.active_count);
  ready->result = wave.transaction.result;
  ready->ready = wave.transaction.state == kDenseBatchGetRetired &&
                 wave.transaction.result == kSuccess
                     ? 1U
                     : 0U;
  return ready->ready != 0U;
}

// Release is SIMT-safe and deliberately refuses a non-terminal wave. This is
// the queue-pool ownership boundary: no subsequent LI producer can reuse the
// SQ/CQ pair until the former owner's CQ has been retired or terminally
// failed and Finish has published the normal KVAF status.
__simt_callee__ inline bool ReleaseKvafStreamingBatchGetQueueFromSimt(
    const KvafStreamingBatchGetControl& control,
    KvafStreamingBatchGetLease* lease) {
  if (lease == nullptr || lease->state != kStreamingBatchGetLeaseTerminal ||
      lease->owner_tag <= 0 || lease->queue_id >= control.queue_count ||
      !IsKvafStreamingBatchGetControlValidForLease(control)) {
    return false;
  }
  __gm__ int32_t* const owners =
      reinterpret_cast<__gm__ int32_t*>(control.queue_leases);
  // Ensure the preceding HCCP/CQ/status accesses complete before another LI
  // producer observes this queue as free.
  AscendC::Simt::ThreadFence();
  if (asc_atomic_cas(owners + lease->queue_id, lease->owner_tag, 0) !=
      lease->owner_tag) {
    return false;
  }
  AscendC::Simt::ThreadFence();
  lease->owner_tag = 0;
  lease->state = kStreamingBatchGetLeaseIdle;
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void KvafStreamingBatchGetReleaseSimtService(
    __gm__ const KvafStreamingBatchGetControl* control_gm,
    uint32_t queue_id, int32_t owner_tag,
    __ubuf__ KvafStreamingBatchGetLeaseResult* result) {
  if (result == nullptr || control_gm == nullptr) {
    return;
  }
  const KvafStreamingBatchGetControl control = *control_gm;
  KvafStreamingBatchGetLease lease{queue_id, owner_tag,
                                    kStreamingBatchGetLeaseTerminal, 0U};
  if (ReleaseKvafStreamingBatchGetQueueFromSimt(control, &lease)) {
    result->queue_id = queue_id;
    result->owner_tag = 0;
    result->success = 1U;
  }
}

// Supported AIC-scalar release companion. The caller must call
// FinishKvafStreamingBatchGet first; this wrapper enforces terminal local
// state before it asks the SIMT service to make the Direct-HCCP queue idle.
__aicore__ inline bool ReleaseKvafStreamingBatchGetQueueFromAicore(
    __gm__ const KvafStreamingBatchGetControl* control_gm,
    __ubuf__ KvafStreamingBatchGetLeaseResult* lease_result,
    event_t simt_event, KvafStreamingBatchGetWave* wave) {
  if (control_gm == nullptr || lease_result == nullptr || wave == nullptr ||
      wave->lease.state != kStreamingBatchGetLeaseTerminal ||
      wave->lease.owner_tag <= 0 || wave->transaction.finalized == 0U) {
    return false;
  }
  lease_result->queue_id = wave->lease.queue_id;
  lease_result->owner_tag = wave->lease.owner_tag;
  lease_result->success = 0U;
  lease_result->reserved0 = 0U;
  set_flag(PIPE_S, PIPE_V, simt_event);
  wait_flag(PIPE_S, PIPE_V, simt_event);
  AscendC::Simt::VF_CALL<KvafStreamingBatchGetReleaseSimtService>(
      AscendC::Simt::Dim3{1U, 1U, 1U}, control_gm, wave->lease.queue_id,
      wave->lease.owner_tag, lease_result);
  set_flag(PIPE_V, PIPE_S, simt_event);
  wait_flag(PIPE_V, PIPE_S, simt_event);
  if (lease_result->success == 0U) {
    return false;
  }
  wave->lease.owner_tag = 0;
  wave->lease.state = kStreamingBatchGetLeaseIdle;
  return true;
}

}  // namespace kvaf_fused_get

#endif  // KVAF_FUSED_FULL_GET_DEVICE_H_
