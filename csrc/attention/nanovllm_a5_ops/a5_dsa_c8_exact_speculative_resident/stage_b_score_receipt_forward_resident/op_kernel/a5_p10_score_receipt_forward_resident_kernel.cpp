// P10 Stage B: graph-owned score-receipt -> forward-resident Direct-HCCP.
//
// This source intentionally defines the target device data plane without
// registering an OPP package yet. Every AIV work-item runs the same
// receipt/scan/compact/post/drain logic and dynamically leases a fixed HCCP
// queue. There is no permanent Get core and no host submission.

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

#include "a5_p10_score_receipt_forward_resident_abi.h"
#include "a5_p10_score_receipt_forward_resident_tiling.h"
#include "a5_p10_score_receipt_forward_resident_kvaf_include.h"

#if !defined(P10_STAGE_B_KVAF_FORWARD_RESIDENT_DEVICE_HEADER)
#error "Define P10_STAGE_B_KVAF_FORWARD_RESIDENT_DEVICE_HEADER to the exact KVAF header"
#endif

#include P10_STAGE_B_KVAF_FORWARD_RESIDENT_DEVICE_HEADER

using namespace AscendC;

namespace a5_p10_stage_b {
namespace {

constexpr uint32_t kReceiptWords = A5_P10_RECEIPT_WORDS64;
constexpr uint32_t kSparseCount = A5_P10_SPARSE_COUNT;
constexpr uint32_t kNoBuffer = A5_P10_STAGE_B_LOCAL_WAVE_CAPACITY;
constexpr uint32_t kKvafRowsPerObject = 128U;

struct ReceiptClaimResult {
  uint32_t claimed;
  uint32_t observed_state;
  uint32_t reserved0;
  uint32_t reserved1;
};
static_assert(sizeof(ReceiptClaimResult) == 16U,
              "receipt claim SIMT ABI changed");

struct ReceiptDoneResult {
  uint32_t completed;
  uint32_t observed_state;
  uint32_t reserved0;
  uint32_t reserved1;
};
static_assert(sizeof(ReceiptDoneResult) == 16U,
              "receipt terminal SIMT ABI changed");

struct SlotClaimResult {
  int32_t resident_slot;
  int32_t observed_object_slot;
  uint32_t newly_claimed;
  uint32_t reserved;
};
static_assert(sizeof(SlotClaimResult) == 16U,
              "forward-slot claim SIMT ABI changed");

struct CompletionResult {
  uint32_t is_last;
  uint32_t observed;
  uint32_t reserved0;
  uint32_t reserved1;
};
static_assert(sizeof(CompletionResult) == 16U,
              "completion SIMT ABI changed");

// KVAF Dense PostOnly owns [0, 4 KiB) through get_imm(): two 32-WQE banks
// consumed by its vector-to-MTE3 submission path.  Stage B must not ask TPipe
// for an unrelated VECCALC allocation and hope that it lands outside those
// banks.  It owns the fixed subrange immediately after them for its two seed
// waves and SIMT result records.
//
// [4 KiB, ceiling) is generic-KVAF scratch only when generic compaction runs.
// P10 invokes Dense PostOnly exclusively, never generic/full BatchGet, so the
// subrange below is unshared for this closure.  The ceiling is a hard source
// guard against silently extending into KVAF's later reserved UB.  This is a
// resource contract, not a capacity tuning knob: 2 x 64 seeds stay fixed.
constexpr uint32_t AlignP10StageBUb(uint32_t bytes) {
  return (bytes + 63U) & ~63U;
}

constexpr uint32_t kP10StageBSeedBytes = AlignP10StageBUb(
    A5_P10_STAGE_B_MICROBATCH_CAPACITY * sizeof(A5P10ForwardResidentSeed));
constexpr uint32_t kP10StageBSeed0UbOffset =
    kvaf_fused_get::kDenseBatchGetReservedWqeUbBytes;
constexpr uint32_t kP10StageBSeed1UbOffset =
    kP10StageBSeed0UbOffset + kP10StageBSeedBytes;
constexpr uint32_t kP10StageBLeaseUbOffset =
    kP10StageBSeed1UbOffset + kP10StageBSeedBytes;
constexpr uint32_t kP10StageBReceiptClaimUbOffset =
    kP10StageBLeaseUbOffset +
    AlignP10StageBUb(sizeof(kvaf_forward_resident_get::
                          KvafForwardResidentQueueLeaseResult));
constexpr uint32_t kP10StageBReceiptDoneUbOffset =
    kP10StageBReceiptClaimUbOffset + AlignP10StageBUb(sizeof(ReceiptClaimResult));
constexpr uint32_t kP10StageBSlotClaimUbOffset =
    kP10StageBReceiptDoneUbOffset + AlignP10StageBUb(sizeof(ReceiptDoneResult));
constexpr uint32_t kP10StageBCompletionUbOffset =
    kP10StageBSlotClaimUbOffset + AlignP10StageBUb(sizeof(SlotClaimResult));
constexpr uint32_t kP10StageBScratchEnd =
    kP10StageBCompletionUbOffset + AlignP10StageBUb(sizeof(CompletionResult));

static_assert(kP10StageBSeed0UbOffset >=
                  kvaf_fused_get::kDenseBatchGetReservedWqeUbBytes,
              "P10 Stage B overlaps KVAF Dense PostOnly WQE banks");
static_assert(kP10StageBScratchEnd <=
                  kvaf_fused_get::kDenseBatchGetPostOnlyScratchCeiling,
              "P10 Stage B overlaps KVAF generic BatchGet scratch");

// A local wave owns one of the two local 64-seed buffers until CQ terminal.
// Its ownership persists for this invocation only, never across forwards.
struct StageBWave {
  kvaf_forward_resident_get::KvafForwardResidentBatchGetWave kvaf_wave;
  kvaf_forward_resident_get::KvafForwardResidentBatchGetTerminal terminal;
  __ubuf__ A5P10ForwardResidentSeed* seeds;
  uint32_t seed_count;
  uint32_t buffer_index;
  uint32_t posted;
  // Failure-only evidence retained until this local wave is reused.  It is
  // not graph-visible and never affects queue or resident-slot ownership.
  uint32_t failure_reason;
};

// HCCP PostOnly must not write a large dense transaction through this C8
// executor.  On 950DT that compound out-parameter can lose a nested writeback
// after the doorbell, leaving neither an explicit failure reason nor a CQ
// drain record.  KVAF now exposes a scalar Begin status; on success the exact
// GM descriptor that KVAF already published is reloaded into a small drain
// record below.  This changes representation only: queue lease, descriptor,
// resident row, and CQ lifecycle remain the verified KVAF protocol.
__aicore__ inline uint32_t BeginFailureReasonFromKvafStatus(uint32_t status) {
  switch (status) {
    case kvaf_forward_resident_get::kForwardResidentBeginInvalidArgs:
      return A5_P10_STAGE_B_POST_BEGIN_INVALID_ARGS;
    case kvaf_forward_resident_get::kForwardResidentBeginInvalidEntity:
      return A5_P10_STAGE_B_POST_BEGIN_INVALID_ENTITY;
    case kvaf_forward_resident_get::kForwardResidentBeginInvalidQueue:
      return A5_P10_STAGE_B_POST_BEGIN_INVALID_QUEUE;
    case kvaf_forward_resident_get::kForwardResidentBeginInvalidLookup:
      return A5_P10_STAGE_B_POST_BEGIN_INVALID_LOOKUP;
    case kvaf_forward_resident_get::kForwardResidentBeginQueueFull:
      return A5_P10_STAGE_B_POST_BEGIN_QUEUE_FULL;
    case kvaf_forward_resident_get::kForwardResidentBeginInvalidOutput:
      return A5_P10_STAGE_B_POST_BEGIN_INVALID_OUTPUT;
    case kvaf_forward_resident_get::kForwardResidentBeginStateIdle:
      return A5_P10_STAGE_B_POST_BEGIN_STATE_IDLE;
    case kvaf_forward_resident_get::kForwardResidentBeginStateRetired:
      return A5_P10_STAGE_B_POST_BEGIN_STATE_RETIRED;
    case kvaf_forward_resident_get::kForwardResidentBeginStateFailedSuccess:
      return A5_P10_STAGE_B_POST_BEGIN_STATE_FAILED_SUCCESS;
    case kvaf_forward_resident_get::kForwardResidentBeginStateUnknown:
      return A5_P10_STAGE_B_POST_BEGIN_STATE_UNKNOWN;
    case kvaf_forward_resident_get::kForwardResidentBeginResultUnknown:
    default:
      return A5_P10_STAGE_B_POST_BEGIN_RESULT_UNKNOWN;
  }
}

__aicore__ inline bool RehydrateForwardResidentWaveForDrain(
    __gm__ const KvafForwardResidentBatchGetControl* control_gm,
    __ubuf__ kvaf_forward_resident_get::KvafForwardResidentQueueLeaseResult*
        lease_result,
    uint32_t expected_active_request_count, StageBWave* wave) {
  if (control_gm == nullptr || lease_result == nullptr || wave == nullptr ||
      expected_active_request_count == 0U || lease_result->success == 0U ||
      lease_result->owner_tag <= 0) {
    return false;
  }
  kvaf_forward_resident_get::KvafForwardResidentBatchGetControl control{};
  if (!kvaf_forward_resident_get::LoadKvafForwardResidentBatchGetControlFromGmForAicore(
          control_gm, &control)) {
    return false;
  }
  kvaf_forward_resident_get::KvafForwardResidentQueueLease lease{};
  lease.queue_id = lease_result->queue_id;
  lease.owner_tag = lease_result->owner_tag;
  lease.state = kvaf_forward_resident_get::kForwardResidentQueueHeld;
  lease.reserved0 = 0U;
  return kvaf_forward_resident_get::
      InitializeKvafForwardResidentPostedWaveForDrain(
          control, lease, expected_active_request_count, &wave->kvaf_wave);
}

__aicore__ inline uint32_t CqFailureReasonFromKvafResult(uint64_t result) {
  switch (result) {
    case kvaf_fused_get::kCqTimeout:
      return A5_P10_STAGE_B_POST_CQ_TIMEOUT;
    case kvaf_fused_get::kCqError:
      return A5_P10_STAGE_B_POST_CQ_ERROR;
    default:
      return A5_P10_STAGE_B_POST_CQ_UNKNOWN;
  }
}

__simt_callee__ inline void ClaimReceiptFromSimt(
    __gm__ int32_t* receipt_state, __gm__ int32_t* forward_control,
    uint32_t receipt,
    __ubuf__ ReceiptClaimResult* result) {
  if (receipt_state == nullptr || forward_control == nullptr || result == nullptr) {
    return;
  }
  result->claimed = 0U;
  result->observed_state = static_cast<uint32_t>(
      asc_atomic_cas(receipt_state + receipt,
                     static_cast<int32_t>(A5P10ReceiptState::READY),
                     static_cast<int32_t>(A5P10ReceiptState::CLAIMED)));
  result->reserved0 = 0U;
  result->reserved1 = 0U;
  if (result->observed_state ==
      static_cast<uint32_t>(A5P10ReceiptState::READY)) {
    // Acquire Stage-A payload stores after READY is observed. Stage A writes
    // score bytes and descriptor first, then publishes READY last.
    AscendC::Simt::ThreadFence();
    result->claimed = 1U;
    // This atomic read/CAS linearizes the diagnostic against producer seal:
    // only a READY claim that observes OPEN may latch the first-preseal
    // sequence. It is a witness, never a scheduling or lifetime input.
    const int32_t observed_seal = asc_atomic_cas(
        forward_control + A5_P10_FORWARD_CONTROL_PRODUCER_SEALED_WORD,
        A5_P10_FORWARD_CONTROL_PRODUCER_OPEN,
        A5_P10_FORWARD_CONTROL_PRODUCER_OPEN);
    if (observed_seal == A5_P10_FORWARD_CONTROL_PRODUCER_OPEN) {
      const int32_t observed_first = asc_atomic_cas(
          forward_control +
              A5_P10_FORWARD_CONTROL_FIRST_PRESEAL_READY_CLAIM_WORD,
          0, static_cast<int32_t>(receipt + 1U));
      result->reserved0 = observed_first == 0 ? 1U : 0U;
    }
  }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void ClaimReceiptSimtService(__gm__ int32_t* receipt_state,
                                    __gm__ int32_t* forward_control,
                                    uint32_t receipt,
                                    __ubuf__ ReceiptClaimResult* result) {
  ClaimReceiptFromSimt(receipt_state, forward_control, receipt, result);
}

// Receipt completion is also state-last and atomic. Only the worker that
// successfully claimed READY may transition CLAIMED -> DONE; a stale worker
// may not overwrite a concurrently observed state with a raw GM store.
__simt_callee__ inline void FinishClaimedReceiptFromSimt(
    __gm__ int32_t* receipt_state, uint32_t receipt,
    __ubuf__ ReceiptDoneResult* result) {
  if (receipt_state == nullptr || result == nullptr) {
    return;
  }
  result->completed = 0U;
  result->observed_state = static_cast<uint32_t>(asc_atomic_cas(
      receipt_state + receipt, static_cast<int32_t>(A5P10ReceiptState::CLAIMED),
      static_cast<int32_t>(A5P10ReceiptState::DONE)));
  result->reserved0 = 0U;
  result->reserved1 = 0U;
  if (result->observed_state ==
      static_cast<uint32_t>(A5P10ReceiptState::CLAIMED)) {
    AscendC::Simt::ThreadFence();
    result->completed = 1U;
  }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void FinishClaimedReceiptSimtService(
    __gm__ int32_t* receipt_state, uint32_t receipt,
    __ubuf__ ReceiptDoneResult* result) {
  FinishClaimedReceiptFromSimt(receipt_state, receipt, result);
}

// Claim one fixed forward resident row. `cursor` is used only for the bounded
// speculative band. `fixed_slot` is the deterministic recall-band row.
// Neither branch frees or recycles a row within the forward.
__simt_callee__ inline void ClaimForwardSlotFromSimt(
    __gm__ int32_t* source_to_forward_slot,
    __gm__ int32_t* kv_object_to_forward_slot,
    __gm__ int32_t* speculative_slot_cursor,
    __gm__ int32_t* resident_ready, uint64_t source_offset,
    uint64_t object_offset,
    int32_t band_base, uint32_t band_capacity, int32_t fixed_slot,
    bool use_speculative_cursor, __ubuf__ SlotClaimResult* result) {
  if (source_to_forward_slot == nullptr || kv_object_to_forward_slot == nullptr ||
      resident_ready == nullptr ||
      result == nullptr || (use_speculative_cursor &&
                            speculative_slot_cursor == nullptr)) {
    return;
  }
  result->resident_slot = A5_P10_STAGE_B_UNMAPPED_SLOT;
  result->observed_object_slot = A5_P10_STAGE_B_UNMAPPED_SLOT;
  result->newly_claimed = 0U;
  result->reserved = 0U;

  // The source map is a fast BxS cache. The object map is authoritative for
  // transport ownership: two S values resolving to one (P,O) must share one
  // resident row and one PostOnly wave.
  const int32_t observed_source = asc_atomic_cas(
      source_to_forward_slot + source_offset, A5_P10_STAGE_B_UNMAPPED_SLOT,
      A5_P10_STAGE_B_CLAIMING_SLOT);
  if (observed_source >= 0) {
    result->resident_slot = observed_source;
    return;
  }
  if (observed_source != A5_P10_STAGE_B_UNMAPPED_SLOT) {
    return;
  }

  const int32_t observed_object = asc_atomic_cas(
      kv_object_to_forward_slot + object_offset,
      A5_P10_STAGE_B_UNMAPPED_SLOT, A5_P10_STAGE_B_CLAIMING_SLOT);
  result->observed_object_slot = observed_object;
  if (observed_object >= 0) {
    // Object was already assigned by another source. Publish its resolved
    // slot into this source cache only after the object map is readable.
    // The same object -> source release rule applies to this dedup path as to
    // a newly claimed object below: a positive source cache entry must never
    // outrun visibility of the authoritative object directory.
    AscendC::Simt::ThreadFence();
    (void)asc_atomic_cas(source_to_forward_slot + source_offset,
                         A5_P10_STAGE_B_CLAIMING_SLOT, observed_object);
    result->resident_slot = observed_object;
    return;
  }
  if (observed_object != A5_P10_STAGE_B_UNMAPPED_SLOT) {
    // A concurrent source is assigning this object. Do not make this source
    // appear mapped until the canonical object slot is final.
    (void)asc_atomic_cas(source_to_forward_slot + source_offset,
                         A5_P10_STAGE_B_CLAIMING_SLOT,
                         A5_P10_STAGE_B_UNMAPPED_SLOT);
    return;
  }

  int32_t selected = fixed_slot;
  if (use_speculative_cursor) {
    const uint32_t relative = asc_atomic_add(speculative_slot_cursor, 1U);
    if (relative >= band_capacity) {
      (void)asc_atomic_cas(kv_object_to_forward_slot + object_offset,
                           A5_P10_STAGE_B_CLAIMING_SLOT,
                           A5_P10_STAGE_B_UNMAPPED_SLOT);
      (void)asc_atomic_cas(source_to_forward_slot + source_offset,
                           A5_P10_STAGE_B_CLAIMING_SLOT,
                           A5_P10_STAGE_B_UNMAPPED_SLOT);
      return;
    }
    selected = band_base + static_cast<int32_t>(relative);
  }
  if (selected < 0) {
    (void)asc_atomic_cas(kv_object_to_forward_slot + object_offset,
                         A5_P10_STAGE_B_CLAIMING_SLOT,
                         A5_P10_STAGE_B_UNMAPPED_SLOT);
    (void)asc_atomic_cas(source_to_forward_slot + source_offset,
                         A5_P10_STAGE_B_CLAIMING_SLOT,
                         A5_P10_STAGE_B_UNMAPPED_SLOT);
    return;
  }

  // ForwardPrepare initializes the full fixed resident table to EMPTY before
  // either companion starts, and a row is never recycled in this forward.
  // Do not rewrite EMPTY here: an ordinary SIMT store could leave a dirty
  // cache line which races the later AICore bypass publication of IN_FLIGHT
  // or READY.  The claim CASes below are the durable admission transition;
  // the AICore PostOnly boundary publishes IN_FLIGHT.
  if (asc_atomic_cas(kv_object_to_forward_slot + object_offset,
                     A5_P10_STAGE_B_CLAIMING_SLOT, selected) !=
      A5_P10_STAGE_B_CLAIMING_SLOT) {
    (void)asc_atomic_cas(source_to_forward_slot + source_offset,
                         A5_P10_STAGE_B_CLAIMING_SLOT,
                         A5_P10_STAGE_B_UNMAPPED_SLOT);
    return;
  }
  // The object map is authoritative and the source map is only its fast
  // cache.  Publish the former before making the latter readable.  A fence
  // before the object CAS does not establish that release ordering: a
  // finalizer can otherwise observe source=selected while a bypass read of
  // the object map still returns the prior EMPTY value.  This is one SIMT
  // publication fence per successful admission, not a worker rendezvous.
  AscendC::Simt::ThreadFence();
  (void)asc_atomic_cas(source_to_forward_slot + source_offset,
                       A5_P10_STAGE_B_CLAIMING_SLOT, selected);
  result->resident_slot = selected;
  result->newly_claimed = 1U;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void ClaimForwardSlotSimtService(
    __gm__ int32_t* source_to_forward_slot,
    __gm__ int32_t* kv_object_to_forward_slot,
    __gm__ int32_t* speculative_slot_cursor,
    __gm__ int32_t* resident_ready, uint64_t source_offset,
    uint64_t object_offset,
    int32_t band_base, uint32_t band_capacity, int32_t fixed_slot,
    bool use_speculative_cursor, __ubuf__ SlotClaimResult* result) {
  ClaimForwardSlotFromSimt(source_to_forward_slot, kv_object_to_forward_slot,
                           speculative_slot_cursor, resident_ready,
                           source_offset, object_offset, band_base,
                           band_capacity, fixed_slot,
                           use_speculative_cursor, result);
}

__simt_callee__ inline void MarkWorkerCompleteFromSimt(
    __gm__ int32_t* progress_word, uint32_t worker_count,
    __ubuf__ CompletionResult* result) {
  if (progress_word == nullptr || result == nullptr || worker_count == 0U) {
    return;
  }
  const uint32_t observed = asc_atomic_add(progress_word, 1U);
  result->observed = observed;
  result->is_last = observed + 1U == worker_count ? 1U : 0U;
  result->reserved0 = 0U;
  result->reserved1 = 0U;
  AscendC::Simt::ThreadFence();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1)
inline void MarkWorkerCompleteSimtService(__gm__ int32_t* progress_word,
                                          uint32_t worker_count,
                                          __ubuf__ CompletionResult* result) {
  MarkWorkerCompleteFromSimt(progress_word, worker_count, result);
}

__aicore__ inline bool LoadReceiptDescriptor(
    __gm__ int64_t* receipt_ring, uint32_t receipt,
    A5P10ScoreReceiptDescriptor* descriptor) {
  if (receipt_ring == nullptr || descriptor == nullptr) {
    return false;
  }
  const uint64_t offset = static_cast<uint64_t>(
      ReadGmByPassDCache<int64_t>(receipt_ring + receipt * kReceiptWords + 0U));
  const uint64_t rows = static_cast<uint64_t>(
      ReadGmByPassDCache<int64_t>(receipt_ring + receipt * kReceiptWords + 1U));
  const uint64_t sources = static_cast<uint64_t>(
      ReadGmByPassDCache<int64_t>(receipt_ring + receipt * kReceiptWords + 2U));
  const uint64_t layout = static_cast<uint64_t>(
      ReadGmByPassDCache<int64_t>(receipt_ring + receipt * kReceiptWords + 3U));
  const uint64_t sequence = static_cast<uint64_t>(
      ReadGmByPassDCache<int64_t>(receipt_ring + receipt * kReceiptWords + 4U));
  descriptor->score_surface_offset_bytes = offset;
  descriptor->row_begin = static_cast<uint32_t>(rows);
  descriptor->row_count = static_cast<uint32_t>(rows >> 32U);
  descriptor->source_begin = static_cast<uint32_t>(sources);
  descriptor->source_count = static_cast<uint32_t>(sources >> 32U);
  descriptor->row_stride_bytes = static_cast<uint32_t>(layout);
  descriptor->score_element_bytes = static_cast<uint32_t>(layout >> 32U);
  descriptor->sequence = static_cast<uint32_t>(sequence);
  descriptor->reserved = static_cast<uint32_t>(sequence >> 32U);
  return descriptor->row_count != 0U && descriptor->source_count != 0U &&
         descriptor->score_element_bytes == A5_P10_SCORE_ELEMENT_BYTES &&
         descriptor->row_stride_bytes >=
             descriptor->source_count * A5_P10_SCORE_ELEMENT_BYTES &&
         descriptor->score_surface_offset_bytes %
                 A5_P10_SCORE_ELEMENT_BYTES ==
             0U;
}

// Begin is the unique descriptor publication / PostOnly boundary. KVAF owns
// its GM/DDR publish barrier and doorbell ordering; Stage B never duplicates
// either operation.
// Return a scalar failure reason rather than mutating a StageBWave during the
// Begin closure.  This function is deeply inlined with KVAF's descriptor/WQE
// construction.  Keeping every failure-side observable in a scalar return
// value makes the Begin boundary independent of the longer-lived executor
// object's address-space representation.  Zero means that PostOnly has
// completed; StartMicrobatch rehydrates its persistent CQ-drain wave from the
// descriptor that KVAF already published.
__aicore__ inline uint32_t BeginForwardResidentMicrobatch(
    __gm__ const KvafForwardResidentBatchGetControl* control_gm,
    uint32_t producer_id, uint32_t queue_hint,
    __ubuf__ A5P10ForwardResidentSeed* seeds, uint32_t seed_count,
    uint32_t rows_per_object,
    __ubuf__ kvaf_forward_resident_get::KvafForwardResidentQueueLeaseResult*
        lease_result,
    event_t simt_event) {
  if (control_gm == nullptr || seeds == nullptr || seed_count == 0U ||
      seed_count > A5_P10_STAGE_B_MICROBATCH_CAPACITY ||
      rows_per_object == 0U || lease_result == nullptr) {
    return A5_P10_STAGE_B_POST_INVALID_CALL;
  }

  kvaf_forward_resident_get::KvafForwardResidentBatchGetControl control{};
  if (!kvaf_forward_resident_get::LoadKvafForwardResidentBatchGetControlFromGmForAicore(
          control_gm, &control)) {
    return A5_P10_STAGE_B_POST_CONTROL_LOAD;
  }
  kvaf_forward_resident_get::KvafForwardResidentQueueLease lease{};
  // Bounded queue contention: final recall retries winners.
  if (!kvaf_forward_resident_get::TryAcquireKvafForwardResidentQueueFromAicore(
          control_gm, producer_id, queue_hint, lease_result, simt_event,
          &lease)) {
    return A5_P10_STAGE_B_POST_QUEUE_ACQUIRE;
  }

  kvaf_forward_resident_get::KvafForwardResidentBatchGetDescriptorScratch
      descriptor{};
  bool descriptor_ready =
      kvaf_forward_resident_get::LoadKvafForwardResidentQueueTemplate(
          control, lease, &descriptor);
  for (uint32_t index = 0U; descriptor_ready && index < seed_count; ++index) {
    // `seeds` lives in VECCALC UB.  AscendC does not permit binding an
    // ordinary C++ reference to a __ubuf__ object, so materialize only the
    // two descriptor fields needed by the PostOnly ABI as AIC scalar values.
    const int32_t kv_object = seeds[index].kv_object;
    const int32_t resident_slot = seeds[index].resident_slot;
    descriptor_ready = kv_object >= 0 && resident_slot >= 0 &&
        kvaf_forward_resident_get::AppendKvafForwardResidentRequest(
            control, &descriptor, static_cast<uint32_t>(kv_object),
            rows_per_object, resident_slot);
  }
  if (!descriptor_ready) {
    (void)kvaf_forward_resident_get::ReturnKvafForwardResidentUnpostedQueueFromAicore(
        control_gm, lease_result, simt_event, &lease);
    return A5_P10_STAGE_B_POST_DESCRIPTOR_BUILD;
  }
  const uint32_t begin_status =
      kvaf_forward_resident_get::BeginKvafForwardResidentBatchGetPostOnly(
          control, lease, descriptor);
  if (begin_status !=
      kvaf_forward_resident_get::kForwardResidentBeginPosted) {
    // Known dense validation failures are provably pre-doorbell and may
    // return the held lease.  Unknown scalar status means the compiler/hot
    // path could not prove whether the doorbell happened: preserve the lease
    // and fail closed instead of introducing an HCCP queue-reuse race.
    if (kvaf_forward_resident_get::KvafForwardResidentBeginHasNoPost(
            begin_status)) {
      (void)kvaf_forward_resident_get::
          ReturnKvafForwardResidentUnpostedQueueFromAicore(
              control_gm, lease_result, simt_event, &lease);
    }
    return BeginFailureReasonFromKvafStatus(begin_status);
  }
  return 0U;
}

// One TryRetire poll; false with terminal=false is normal nonblocking CQ
// progress. The fixed resident rows are never returned to any queue.
__aicore__ inline bool DrainForwardResidentMicrobatch(
    __gm__ const KvafForwardResidentBatchGetControl* control_gm,
    __ubuf__ kvaf_forward_resident_get::KvafForwardResidentQueueLeaseResult*
        lease_result,
    event_t simt_event, StageBWave* wave, bool* terminal) {
  if (control_gm == nullptr || lease_result == nullptr || wave == nullptr ||
      terminal == nullptr || wave->posted == 0U) {
    return false;
  }
  *terminal = false;
  kvaf_forward_resident_get::TryRetireKvafForwardResidentBatchGet(
      &wave->kvaf_wave);
  if (wave->kvaf_wave.lease.state !=
      kvaf_forward_resident_get::kForwardResidentQueueTerminal) {
    return true;
  }
  const bool success = kvaf_forward_resident_get::FinishKvafForwardResidentBatchGet(
      &wave->kvaf_wave, &wave->terminal);
  if (!success) {
    wave->failure_reason = CqFailureReasonFromKvafResult(wave->terminal.result);
  }
  if (!kvaf_forward_resident_get::ReturnKvafForwardResidentQueueFromAicore(
          control_gm, lease_result, simt_event, &wave->kvaf_wave)) {
    if (wave->failure_reason == 0U) {
      wave->failure_reason = A5_P10_STAGE_B_POST_QUEUE_RETURN;
    }
    return false;
  }
  wave->posted = 0U;
  *terminal = true;
  return success;
}

class A5P10ScoreReceiptForwardResidentKernel {
 public:
  __aicore__ inline A5P10ScoreReceiptForwardResidentKernel(
      const A5P10ScoreReceiptForwardResidentTilingData* tiling)
      : tiling_(tiling) {}

  __aicore__ inline void Init(
      GM_ADDR score_surface, GM_ADDR receipt_ring, GM_ADDR receipt_state,
      GM_ADDR forward_control, GM_ADDR active_row_pool_entries,
      GM_ADDR kv_object_map, GM_ADDR kv_object_slots_snapshot,
      GM_ADDR kv_object_to_forward_slot, GM_ADDR source_to_forward_slot,
      GM_ADDR speculative_slot_band,
      GM_ADDR speculative_slot_cursor, GM_ADDR recall_slot_band,
      GM_ADDR canonical_sparse_indices, GM_ADDR canonical_score_source_lens,
      GM_ADDR kvaf_forward_resident_control, GM_ADDR queue_leases,
      GM_ADDR resident_payload, GM_ADDR resident_ready,
      GM_ADDR resident_row_to_qsfa_slot, GM_ADDR attention_slots_buffer,
      GM_ADDR selected_slot_lengths_buffer) {
    scoreSurface_ = reinterpret_cast<__gm__ uint16_t*>(score_surface);
    receiptRing_ = reinterpret_cast<__gm__ int64_t*>(receipt_ring);
    receiptState_ = reinterpret_cast<__gm__ int32_t*>(receipt_state);
    forwardControl_ = reinterpret_cast<__gm__ int32_t*>(forward_control);
    activeRowPoolEntries_ = reinterpret_cast<__gm__ int32_t*>(active_row_pool_entries);
    kvObjectMap_ = reinterpret_cast<__gm__ int32_t*>(kv_object_map);
    kvObjectSlotsSnapshot_ =
        reinterpret_cast<__gm__ int32_t*>(kv_object_slots_snapshot);
    kvObjectToForwardSlot_ =
        reinterpret_cast<__gm__ int32_t*>(kv_object_to_forward_slot);
    sourceToForwardSlot_ =
        reinterpret_cast<__gm__ int32_t*>(source_to_forward_slot);
    speculativeSlotBand_ =
        reinterpret_cast<__gm__ int32_t*>(speculative_slot_band);
    speculativeSlotCursor_ =
        reinterpret_cast<__gm__ int32_t*>(speculative_slot_cursor);
    recallSlotBand_ = reinterpret_cast<__gm__ int32_t*>(recall_slot_band);
    canonicalSparseIndices_ =
        reinterpret_cast<__gm__ int32_t*>(canonical_sparse_indices);
    canonicalScoreSourceLens_ =
        reinterpret_cast<__gm__ int32_t*>(canonical_score_source_lens);
    controlGm_ = reinterpret_cast<__gm__ const KvafForwardResidentBatchGetControl*>(
        kvaf_forward_resident_control);
    queueLeases_ = reinterpret_cast<__gm__ int32_t*>(queue_leases);
    residentPayload_ = reinterpret_cast<__gm__ uint8_t*>(resident_payload);
    residentReady_ = reinterpret_cast<__gm__ int32_t*>(resident_ready);
    residentRowToQsfaSlot_ =
        reinterpret_cast<__gm__ int32_t*>(resident_row_to_qsfa_slot);
    attentionSlots_ = reinterpret_cast<__gm__ int32_t*>(attention_slots_buffer);
    selectedSlotLengths_ =
        reinterpret_cast<__gm__ int32_t*>(selected_slot_lengths_buffer);

    worker_ = GetBlockIdx();
    enabled_ = false;
    for (uint32_t index = 0U;
         index < A5_P10_STAGE_B_LOCAL_WAVE_CAPACITY; ++index) {
      waves_[index].seeds = nullptr;
      waves_[index].seed_count = 0U;
      waves_[index].buffer_index = index;
      waves_[index].posted = 0U;
      waves_[index].failure_reason = 0U;
      seedBufferBusy_[index] = false;
    }

    if (tiling_ == nullptr ||
        worker_ >= tiling_->executorWorkerCount ||
        scoreSurface_ == nullptr || receiptRing_ == nullptr ||
        receiptState_ == nullptr || forwardControl_ == nullptr ||
        activeRowPoolEntries_ == nullptr || kvObjectMap_ == nullptr ||
        kvObjectSlotsSnapshot_ == nullptr || kvObjectToForwardSlot_ == nullptr ||
        sourceToForwardSlot_ == nullptr ||
        speculativeSlotBand_ == nullptr || speculativeSlotCursor_ == nullptr ||
        recallSlotBand_ == nullptr || canonicalSparseIndices_ == nullptr ||
        canonicalScoreSourceLens_ == nullptr || controlGm_ == nullptr ||
        queueLeases_ == nullptr || residentPayload_ == nullptr ||
        residentReady_ == nullptr || residentRowToQsfaSlot_ == nullptr ||
        attentionSlots_ == nullptr ||
        selectedSlotLengths_ == nullptr ||
        tiling_->executorWorkerCount == 0U ||
        tiling_->executorWorkerLimit == 0U ||
        tiling_->executorWorkerCount > tiling_->executorWorkerLimit ||
        tiling_->receiptPollQuantum == 0U ||
        tiling_->activeRowCount == 0U ||
        tiling_->sparseCount != kSparseCount ||
        tiling_->boundedDrainPolls == 0U) {
      RecordFirstFailure(A5_P10_STAGE_B_DIAGNOSTIC_INIT_COMMON);
      return;
    }

    // receiptCapacity is graph-static and known before either companion is
    // launched. Do not read a dynamic producer count here: Stage A may not
    // have produced its first tile yet. FREE entries become exhausted only
    // after producer_sealed is visible.
    receiptCapacity_ = tiling_->receiptCapacity;
    activeRows_ = static_cast<uint32_t>(ReadGmByPassDCache<int32_t>(
        forwardControl_ + A5_P10_STAGE_B_ACTIVE_ROWS));
    rowsPerObject_ = static_cast<uint32_t>(ReadGmByPassDCache<int32_t>(
        forwardControl_ + A5_P10_STAGE_B_ROWS_PER_OBJECT));
    topkPerRow_ = static_cast<uint32_t>(ReadGmByPassDCache<int32_t>(
        forwardControl_ + A5_P10_STAGE_B_TOPK_PER_ROW));
    scoreCutoffU16_ = static_cast<uint32_t>(ReadGmByPassDCache<int32_t>(
        forwardControl_ + A5_P10_STAGE_B_SCORE_CUTOFF_U16)) & 0xffffU;
    speculativeBand_.base =
        ReadGmByPassDCache<int32_t>(speculativeSlotBand_ + 0U);
    speculativeBand_.capacity =
        ReadGmByPassDCache<int32_t>(speculativeSlotBand_ + 1U);
    recallBand_.base = ReadGmByPassDCache<int32_t>(recallSlotBand_ + 0U);
    recallBand_.capacity = ReadGmByPassDCache<int32_t>(recallSlotBand_ + 1U);

    const uint64_t expected_source_capacity =
        static_cast<uint64_t>(tiling_->qcap) * kKvafRowsPerObject;
    if (receiptCapacity_ == 0U || activeRows_ != tiling_->activeRowCount ||
        rowsPerObject_ == 0U || tiling_->numLayers == 0U ||
        tiling_->qcap == 0U || tiling_->layerId >= tiling_->numLayers ||
        rowsPerObject_ != kKvafRowsPerObject ||
        expected_source_capacity != tiling_->sourceCapacity ||
        tiling_->sourceCapacity > tiling_->kvObjectCapacity ||
        topkPerRow_ != kSparseCount || speculativeBand_.base < 0 ||
        speculativeBand_.capacity < 0 || recallBand_.base < 0 ||
        recallBand_.capacity < 0 ||
        static_cast<uint64_t>(recallBand_.capacity) <
            static_cast<uint64_t>(activeRows_) * topkPerRow_ ||
        static_cast<uint64_t>(speculativeBand_.base) +
                static_cast<uint32_t>(speculativeBand_.capacity) >
            tiling_->residentRowCount ||
        static_cast<uint64_t>(recallBand_.base) +
                static_cast<uint32_t>(recallBand_.capacity) >
            tiling_->residentRowCount) {
      RecordFirstFailure(A5_P10_STAGE_B_DIAGNOSTIC_INIT_GEOMETRY);
      return;
    }

    kvaf_forward_resident_get::KvafForwardResidentBatchGetControl control{};
    if (!kvaf_forward_resident_get::LoadKvafForwardResidentBatchGetControlFromGmForAicore(
            controlGm_, &control) ||
        !kvaf_forward_resident_get::IsKvafForwardResidentBatchGetControlValidForAicore(
            control) ||
        control.queue_leases != reinterpret_cast<uint64_t>(queueLeases_) ||
        control.output != reinterpret_cast<uint64_t>(residentPayload_) ||
        control.output_rows < tiling_->residentRowCount) {
      RecordFirstFailure(A5_P10_STAGE_B_DIAGNOSTIC_INIT_KVAF);
      return;
    }

    localSeeds_[0] = reinterpret_cast<__ubuf__ A5P10ForwardResidentSeed*>(
        get_imm(kP10StageBSeed0UbOffset));
    localSeeds_[1] = reinterpret_cast<__ubuf__ A5P10ForwardResidentSeed*>(
        get_imm(kP10StageBSeed1UbOffset));
    leaseResult_ = reinterpret_cast<__ubuf__ kvaf_forward_resident_get::
        KvafForwardResidentQueueLeaseResult*>(get_imm(kP10StageBLeaseUbOffset));
    receiptClaim_ = reinterpret_cast<__ubuf__ ReceiptClaimResult*>(
        get_imm(kP10StageBReceiptClaimUbOffset));
    receiptDone_ = reinterpret_cast<__ubuf__ ReceiptDoneResult*>(
        get_imm(kP10StageBReceiptDoneUbOffset));
    slotClaim_ = reinterpret_cast<__ubuf__ SlotClaimResult*>(
        get_imm(kP10StageBSlotClaimUbOffset));
    completion_ = reinterpret_cast<__ubuf__ CompletionResult*>(
        get_imm(kP10StageBCompletionUbOffset));
    if (localSeeds_[0] == nullptr || localSeeds_[1] == nullptr ||
        leaseResult_ == nullptr || receiptClaim_ == nullptr ||
        receiptDone_ == nullptr ||
        slotClaim_ == nullptr || completion_ == nullptr) {
      RecordFirstFailure(A5_P10_STAGE_B_DIAGNOSTIC_INIT_UB);
      return;
    }
    enabled_ = true;
    MarkRunning();
  }

  __aicore__ inline void Process() {
    if (!enabled_) {
      return;
    }
    // This companion starts from the forward's cold-bound mutable aliases
    // independently of Stage A. There is deliberately no Stage-A -> Stage-B
    // graph completion edge: C8 publishes each READY receipt while this loop
    // continues exact scans, dynamic Posts, and bounded nonblocking CQ polls.
    // Stage A may seal only after it has published all receipt_count records.
    bool producer_sealed = false;
    bool owned_receipts_done = false;
    uint32_t poll_turn = 0U;
    while (!producer_sealed || !owned_receipts_done) {
      producer_sealed = ProducerSealed();
      // Overlap witness: when producer_sealed is false this still claims any
      // READY receipt immediately; Stage B does not defer all work until C8
      // completes and seals its static receipt range.
      owned_receipts_done = ServiceOwnedReceipts(producer_sealed);
      (void)DrainLocalWaves(false);

      if (producer_sealed && !owned_receipts_done) {
        // After the seal, FREE/FILLING/CLAIMED may not wait indefinitely.
        // A bad producer/consumer protocol leaves forward_state incomplete;
        // it never lets QSFA observe an unfinished slot map.
        for (uint32_t poll = 0U; poll < tiling_->boundedDrainPolls &&
             !owned_receipts_done; ++poll) {
          owned_receipts_done = ServiceOwnedReceipts(true);
          (void)DrainLocalWaves(false);
        }
        if (!owned_receipts_done) {
          RecordFirstFailure(A5_P10_STAGE_B_DIAGNOSTIC_RECEIPT_TERMINAL);
          return;
        }
      }
      ++poll_turn;
      if (poll_turn == tiling_->receiptPollQuantum) {
        PollReceiptQuantumBoundary();
        poll_turn = 0U;
      }
    }
    // Only final exact winners may force transport progress. False-positive
    // speculative rows stay allocated but are never emitted.
    if (!FinalizeAssignedRows()) {
      RecordFirstFailure(A5_P10_STAGE_B_DIAGNOSTIC_FINALIZE);
      return;
    }
    if (!DrainLocalWaves(true)) {
      RecordFirstFailure(A5_P10_STAGE_B_DIAGNOSTIC_DRAIN);
      return;
    }
    MarkWorkerComplete();
  }

 private:
  // This is intentionally a tiny diagnostic ledger, not a second lifecycle
  // protocol.  It uses the existing forward_control[1] word, which no stage
  // reads for scheduling.  The first negative error wins; a later worker can
  // only replace the benign RUNNING marker, never erase an earlier failure.
  __aicore__ inline void MarkRunning() {
    if (forwardControl_ == nullptr) {
      return;
    }
    (void)AscendC::AtomicCas<uint32_t>(
        reinterpret_cast<__gm__ uint32_t*>(
            forwardControl_ + A5_P10_STAGE_B_DIAGNOSTIC),
        static_cast<uint32_t>(A5_P10_STAGE_B_DIAGNOSTIC_CLEAR),
        static_cast<uint32_t>(A5_P10_STAGE_B_DIAGNOSTIC_RUNNING));
  }

  __aicore__ inline void RecordFirstFailure(int32_t code) {
    if (forwardControl_ == nullptr || code >= 0) {
      return;
    }
    auto* const word = reinterpret_cast<__gm__ uint32_t*>(
        forwardControl_ + A5_P10_STAGE_B_DIAGNOSTIC);
    const uint32_t desired = static_cast<uint32_t>(code);
    // Do not depend on AtomicCas returning the observed value: the AscendC
    // interface is intentionally used here in the same fire-and-forget form
    // as the resident-map convergence CAS below.  The two legal benign
    // predecessors are CLEAR and RUNNING; once a negative code is installed,
    // neither CAS can overwrite it, so the first failure remains durable.
    (void)AscendC::AtomicCas<uint32_t>(
        word, static_cast<uint32_t>(A5_P10_STAGE_B_DIAGNOSTIC_CLEAR), desired);
    (void)AscendC::AtomicCas<uint32_t>(
        word, static_cast<uint32_t>(A5_P10_STAGE_B_DIAGNOSTIC_RUNNING), desired);
  }

  __aicore__ inline bool ProducerSealed() const {
    return ReadGmByPassDCache<int32_t>(
               forwardControl_ + A5_P10_STAGE_B_PRODUCER_SEALED) ==
           static_cast<int32_t>(A5P10ProducerSealState::SEALED);
  }

  // Each worker owns a deterministic stride of the graph-static receipt
  // capacity. READY is claimed atomically anyway so a future tiler may
  // repartition work without changing the receipt protocol. Before the seal,
  // FREE remains live because Stage A may publish it later. After the seal,
  // FREE is an unused tile slot and DONE is a consumed tile slot; both are
  // terminal for this forward.
  __aicore__ inline bool ServiceOwnedReceipts(bool producer_sealed) {
    bool all_done = true;
    for (uint32_t receipt = worker_; receipt < receiptCapacity_;
         receipt += tiling_->executorWorkerCount) {
      const int32_t state_before = ReadGmByPassDCache<int32_t>(
          receiptState_ + receipt);
      if (state_before == static_cast<int32_t>(A5P10ReceiptState::READY) &&
          TryClaimReceipt(receipt)) {
        ConsumeExactScoreReceipt(receipt);
      }
      const int32_t state_after = ReadGmByPassDCache<int32_t>(
          receiptState_ + receipt);
      if (producer_sealed &&
          state_after != static_cast<int32_t>(A5P10ReceiptState::DONE) &&
          state_after != static_cast<int32_t>(A5P10ReceiptState::FREE)) {
        all_done = false;
      }
    }
    return all_done;
  }

  __aicore__ inline bool TryClaimReceipt(uint32_t receipt) {
    receiptClaim_->claimed = 0U;
    set_flag(PIPE_S, PIPE_V, simtEvent_);
    wait_flag(PIPE_S, PIPE_V, simtEvent_);
    AscendC::Simt::VF_CALL<ClaimReceiptSimtService>(
        AscendC::Simt::Dim3{1U, 1U, 1U}, receiptState_, forwardControl_, receipt,
        receiptClaim_);
    set_flag(PIPE_V, PIPE_S, simtEvent_);
    wait_flag(PIPE_V, PIPE_S, simtEvent_);
    return receiptClaim_->claimed != 0U;
  }

  __aicore__ inline bool FinishClaimedReceipt(uint32_t receipt) {
    receiptDone_->completed = 0U;
    set_flag(PIPE_S, PIPE_V, simtEvent_);
    wait_flag(PIPE_S, PIPE_V, simtEvent_);
    AscendC::Simt::VF_CALL<FinishClaimedReceiptSimtService>(
        AscendC::Simt::Dim3{1U, 1U, 1U}, receiptState_, receipt,
        receiptDone_);
    set_flag(PIPE_V, PIPE_S, simtEvent_);
    wait_flag(PIPE_V, PIPE_S, simtEvent_);
    return receiptDone_->completed != 0U;
  }

  // A bounded quantum is an explicit device-side checkpoint between receipt
  // scans. This compile domain provides no portable instruction that yields a
  // running AIV to another kernel, so the product tiler must enforce the
  // logical executorWorkerLimit; this boundary is deliberately not misnamed
  // as a physical-core yield or an all-core launch.
  __aicore__ inline void PollReceiptQuantumBoundary() {
    DataSyncBarrier<MemDsbT::DDR>();
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline bool ResolveActiveRowMapping(
      uint32_t row, uint32_t source, A5P10ActiveRowMapping* mapping,
      bool require_canonical_source_length) const {
    if (mapping == nullptr || row >= activeRows_ ||
        source >= tiling_->sourceCapacity) {
      return false;
    }
    if (require_canonical_source_length) {
      const int32_t source_len = ReadGmByPassDCache<int32_t>(
          canonicalScoreSourceLens_ + row);
      if (source_len <= 0 || source >= static_cast<uint32_t>(source_len)) {
        return false;
      }
    }
    const int32_t pool = ReadGmByPassDCache<int32_t>(
        activeRowPoolEntries_ + row);
    if (pool < 0 || static_cast<uint32_t>(pool) >= tiling_->poolCapacity) {
      return false;
    }
    const uint64_t map_offset = static_cast<uint64_t>(pool) *
            tiling_->sourceCapacity + source;
    const uint64_t local_object = source;
    const uint64_t global_object =
        (static_cast<uint64_t>(pool) * tiling_->numLayers + tiling_->layerId) *
            tiling_->qcap * kKvafRowsPerObject + source;
    if (local_object >= tiling_->kvObjectCapacity ||
        global_object > static_cast<uint64_t>(INT32_MAX)) {
      return false;
    }
    // kv_object_map is a cold-bound witness, not an opportunity to remap S.
    // It must encode the exact token object used by KVAF Append:
    // ((P*numLayers+layerId)*qcap*128)+S. Append then derives the established
    // table index P*numLayers*qcap+layerId*qcap+(S>>7) and (S&127)*656.
    const int32_t materialized_object =
        ReadGmByPassDCache<int32_t>(kvObjectMap_ + map_offset);
    if (materialized_object != static_cast<int32_t>(global_object)) {
      return false;
    }
    const uint64_t slot_offset = static_cast<uint64_t>(pool) *
            tiling_->kvObjectCapacity + local_object;
    mapping->pool_entry = pool;
    mapping->kv_object = static_cast<int32_t>(global_object);
    mapping->object_local = static_cast<int32_t>(local_object);
    mapping->persistent_slot = ReadGmByPassDCache<int32_t>(
        kvObjectSlotsSnapshot_ + slot_offset);
    return true;
  }

  __aicore__ inline bool ClaimSpeculativeSlot(
      uint64_t source_offset, uint64_t object_offset, int32_t* slot,
      bool* newly_claimed) {
    if (slot == nullptr || newly_claimed == nullptr) {
      return false;
    }
    slotClaim_->resident_slot = A5_P10_STAGE_B_UNMAPPED_SLOT;
    slotClaim_->newly_claimed = 0U;
    set_flag(PIPE_S, PIPE_V, simtEvent_);
    wait_flag(PIPE_S, PIPE_V, simtEvent_);
    AscendC::Simt::VF_CALL<ClaimForwardSlotSimtService>(
        AscendC::Simt::Dim3{1U, 1U, 1U}, sourceToForwardSlot_,
        kvObjectToForwardSlot_, speculativeSlotCursor_, residentReady_,
        source_offset, object_offset,
        speculativeBand_.base, static_cast<uint32_t>(speculativeBand_.capacity),
        A5_P10_STAGE_B_UNMAPPED_SLOT, true, slotClaim_);
    set_flag(PIPE_V, PIPE_S, simtEvent_);
    wait_flag(PIPE_V, PIPE_S, simtEvent_);
    *slot = slotClaim_->resident_slot;
    *newly_claimed = slotClaim_->newly_claimed != 0U;
    return *slot >= 0;
  }

  __aicore__ inline bool ClaimRecallSlot(
      uint64_t source_offset, uint64_t object_offset, uint32_t row, uint32_t rank,
      int32_t* slot, bool* newly_claimed) {
    if (slot == nullptr || newly_claimed == nullptr || row >= activeRows_ ||
        rank >= topkPerRow_) {
      return false;
    }
    const uint64_t relative = static_cast<uint64_t>(row) * topkPerRow_ + rank;
    if (relative >= static_cast<uint32_t>(recallBand_.capacity)) {
      return false;
    }
    const int32_t fixed_slot = recallBand_.base + static_cast<int32_t>(relative);
    slotClaim_->resident_slot = A5_P10_STAGE_B_UNMAPPED_SLOT;
    slotClaim_->newly_claimed = 0U;
    set_flag(PIPE_S, PIPE_V, simtEvent_);
    wait_flag(PIPE_S, PIPE_V, simtEvent_);
    AscendC::Simt::VF_CALL<ClaimForwardSlotSimtService>(
        AscendC::Simt::Dim3{1U, 1U, 1U}, sourceToForwardSlot_,
        kvObjectToForwardSlot_, speculativeSlotCursor_, residentReady_,
        source_offset, object_offset, 0, 0U,
        fixed_slot, false, slotClaim_);
    set_flag(PIPE_V, PIPE_S, simtEvent_);
    wait_flag(PIPE_V, PIPE_S, simtEvent_);
    *slot = slotClaim_->resident_slot;
    *newly_claimed = slotClaim_->newly_claimed != 0U;
    return *slot >= 0;
  }

  __aicore__ inline uint32_t FindFreeSeedBuffer() const {
    for (uint32_t index = 0U;
         index < A5_P10_STAGE_B_LOCAL_WAVE_CAPACITY; ++index) {
      if (!seedBufferBusy_[index]) {
        return index;
      }
    }
    return kNoBuffer;
  }

  __aicore__ inline uint32_t FindFreeWave() const {
    for (uint32_t index = 0U;
         index < A5_P10_STAGE_B_LOCAL_WAVE_CAPACITY; ++index) {
      if (waves_[index].posted == 0U) {
        return index;
      }
    }
    return kNoBuffer;
  }

  // Return a scalar post result.  Do not use an AICore pointer out-parameter
  // for failure evidence: it was the final opaque writeback edge that could
  // erase a valid KVAF Begin diagnostic on a later final-recall wave.
  __aicore__ inline uint32_t StartMicrobatch(
      uint32_t buffer_index, uint32_t seed_count) {
    if (buffer_index >= A5_P10_STAGE_B_LOCAL_WAVE_CAPACITY ||
        seed_count == 0U || seed_count > A5_P10_STAGE_B_MICROBATCH_CAPACITY) {
      return A5_P10_STAGE_B_POST_INVALID_CALL;
    }
    uint32_t wave_index = FindFreeWave();
    if (wave_index == kNoBuffer) {
      (void)DrainLocalWaves(false);
      wave_index = FindFreeWave();
    }
    if (wave_index == kNoBuffer) {
      return A5_P10_STAGE_B_POST_NO_FREE_WAVE;
    }
    StageBWave* wave = waves_ + wave_index;
    wave->buffer_index = buffer_index;
    wave->posted = 0U;
    wave->seed_count = 0U;
    wave->seeds = nullptr;
    wave->failure_reason = 0U;
    // PostOnly returns one scalar status.  It cannot mutate a caller-owned
    // wave; only after that scalar proves POSTED do we reload the published
    // descriptor into this persistent CQ-drain record.
    const uint32_t begin_reason = BeginForwardResidentMicrobatch(
            controlGm_, worker_, worker_ + buffer_index, localSeeds_[buffer_index],
            seed_count, rowsPerObject_, leaseResult_, simtEvent_);
    if (begin_reason != 0U) {
      return begin_reason;
    }
    if (!RehydrateForwardResidentWaveForDrain(
            controlGm_, leaseResult_, seed_count, wave)) {
      // PostOnly is already durable here.  Do not release this lease from an
      // unproven state; preserve the failed forward's Direct-HCCP evidence.
      return A5_P10_STAGE_B_POST_BEGIN_REHYDRATE;
    }
    wave->seeds = localSeeds_[buffer_index];
    wave->seed_count = seed_count;
    wave->posted = 1U;
    // Begin returned successfully, so descriptors are durable and the fixed
    // rows may now be observed as in-flight by final recall.
    for (uint32_t index = 0U; index < seed_count; ++index) {
      // `resident_ready` crosses the AICore -> SIMT/host/QSFA boundary.  A
      // normal AICore GM store may remain only in DCache while a later
      // uncached consumer observes the old EMPTY value.  Publish the state
      // through the same bypass path used for the Direct-HCCP descriptor; the
      // following DSB orders it after the successful PostOnly boundary.
      WriteGmByPassDCache<int32_t>(
          residentReady_ + wave->seeds[index].resident_slot,
          static_cast<int32_t>(A5P10ResidentReadyState::IN_FLIGHT));
    }
    DataSyncBarrier<MemDsbT::DDR>();
    seedBufferBusy_[buffer_index] = true;
    return 0U;
  }

  __aicore__ inline void FlushSpeculativeSeeds(
      uint32_t buffer_index, uint32_t seed_count) {
    if (seed_count == 0U) {
      return;
    }
    // Busy queues only defer speculation. EMPTY rows are harmless and final
    // recall reuses the same fixed row for an actual exact winner.
    (void)StartMicrobatch(buffer_index, seed_count);
  }

  __aicore__ inline void ConsumeExactScoreReceipt(uint32_t receipt) {
    A5P10ScoreReceiptDescriptor descriptor{};
    if (!LoadReceiptDescriptor(receiptRing_, receipt, &descriptor) ||
        descriptor.row_begin >= activeRows_) {
      (void)FinishClaimedReceipt(receipt);
      return;
    }
    DataSyncBarrier<MemDsbT::DDR>();
    uint32_t buffer_index = FindFreeSeedBuffer();
    uint32_t seed_count = 0U;
    for (uint32_t row_delta = 0U;
         row_delta < descriptor.row_count &&
         descriptor.row_begin + row_delta < activeRows_; ++row_delta) {
      const uint32_t row = descriptor.row_begin + row_delta;
      for (uint32_t source_delta = 0U;
           source_delta < descriptor.source_count; ++source_delta) {
        const uint32_t source = descriptor.source_begin + source_delta;
        if (source >= tiling_->sourceCapacity || buffer_index == kNoBuffer) {
          continue;
        }
        const uint64_t score_byte_offset = descriptor.score_surface_offset_bytes +
            static_cast<uint64_t>(row_delta) * descriptor.row_stride_bytes +
            static_cast<uint64_t>(source_delta) * descriptor.score_element_bytes;
        const uint16_t score = ReadGmByPassDCache<uint16_t>(
            scoreSurface_ + score_byte_offset / A5_P10_SCORE_ELEMENT_BYTES);
        // ExactScoreScan's only numerical decision is in the native uint16
        // score domain. It neither changes nor reorders canonical Top-K.
        if (static_cast<uint32_t>(score) < scoreCutoffU16_) {
          continue;
        }
        A5P10ActiveRowMapping mapping{};
        // A READY receipt is published only after native C8 has written this
        // exact score range.  The final `canonicalScoreSourceLens_` is a
        // post-Top-K output and is intentionally unavailable while C8 is
        // still scanning.  Using it here suppressed every pre-seal candidate
        // without protecting correctness; the receipt's bounded source range
        // is the valid pre-seal authority.  Final Top-K materialization below
        // still requires the canonical length.
        if (!ResolveActiveRowMapping(row, source, &mapping, false) ||
            mapping.persistent_slot >= 0) {
          continue;
        }
        const uint64_t source_offset = static_cast<uint64_t>(row) *
            tiling_->sourceCapacity + source;
        const uint64_t object_offset = static_cast<uint64_t>(
            mapping.pool_entry) * tiling_->kvObjectCapacity +
            static_cast<uint32_t>(mapping.object_local);
        int32_t resident_slot = A5_P10_STAGE_B_UNMAPPED_SLOT;
        bool newly_claimed = false;
        if (!ClaimSpeculativeSlot(source_offset, object_offset, &resident_slot,
                                  &newly_claimed) || !newly_claimed) {
          continue;
        }
        // Aggregate/struct copies into __ubuf__ are not a supported AscendC
        // address-space operation.  Keep the 16-byte seed ABI, but issue its
        // four scalar UB stores explicitly.
        localSeeds_[buffer_index][seed_count].row = static_cast<int32_t>(row);
        localSeeds_[buffer_index][seed_count].source =
            static_cast<int32_t>(source);
        localSeeds_[buffer_index][seed_count].kv_object = mapping.kv_object;
        localSeeds_[buffer_index][seed_count].resident_slot = resident_slot;
        ++seed_count;
        if (seed_count == A5_P10_STAGE_B_MICROBATCH_CAPACITY) {
          FlushSpeculativeSeeds(buffer_index, seed_count);
          buffer_index = FindFreeSeedBuffer();
          seed_count = 0U;
        }
      }
    }
    FlushSpeculativeSeeds(buffer_index, seed_count);
    // Claim -> durable local copy/terminal decision -> DONE. Ring recycle is
    // intentionally outside this forward-owned kernel boundary.
    DataSyncBarrier<MemDsbT::DDR>();
    (void)FinishClaimedReceipt(receipt);
  }

  // `target_slot_ready` is a local completion witness for a final winner.
  // It is set only after the wave containing `target_slot` has reached a
  // successful CQ terminal, Stage B has written resident_ready=READY, and
  // the DDR publication barrier has executed.  This avoids treating a
  // same-kernel GM reload as the authority for a completion that this worker
  // has already observed directly from KVAF.  It is not a shortcut around
  // readiness: a failed/nonterminal CQ never sets this witness.
  __aicore__ inline bool DrainLocalWavesForTarget(
      bool terminal_mode, int32_t target_slot, bool* target_slot_ready,
      uint32_t* target_failure_reason = nullptr) {
    if (target_slot_ready != nullptr) {
      *target_slot_ready = false;
    }
    if (target_failure_reason != nullptr) {
      *target_failure_reason = 0U;
    }
    const uint32_t polls = terminal_mode ? tiling_->boundedDrainPolls : 1U;
    for (uint32_t poll = 0U; poll < polls; ++poll) {
      bool pending = false;
      for (uint32_t index = 0U;
           index < A5_P10_STAGE_B_LOCAL_WAVE_CAPACITY; ++index) {
        StageBWave* wave = waves_ + index;
        if (wave->posted == 0U) {
          continue;
        }
        pending = true;
        bool wave_has_target = false;
        if (target_slot >= 0 && wave->seeds != nullptr) {
          for (uint32_t seed = 0U; seed < wave->seed_count; ++seed) {
            wave_has_target = wave_has_target ||
                wave->seeds[seed].resident_slot == target_slot;
          }
        }
        bool terminal = false;
        const bool success = DrainForwardResidentMicrobatch(
            controlGm_, leaseResult_, simtEvent_, wave, &terminal);
        if (!terminal) {
          if (!success && wave_has_target && target_failure_reason != nullptr &&
              wave->failure_reason != 0U) {
            *target_failure_reason = wave->failure_reason;
          }
          continue;
        }
        // A terminal success alone makes a fixed resident slot READY. The
        // current target assumes successful posts, but this guard never
        // fabricates READY on a failed terminal.
        bool completed_target_slot = false;
        if (success) {
          for (uint32_t seed = 0U; seed < wave->seed_count; ++seed) {
            // CQ terminal is the release point for this fixed resident row.
            // It must be visible outside this AICore's DCache before either a
            // local target proof or the forward COMPLETE publication can let
            // QSFA consume the row.
            WriteGmByPassDCache<int32_t>(
                residentReady_ + wave->seeds[seed].resident_slot,
                static_cast<int32_t>(A5P10ResidentReadyState::READY));
            completed_target_slot = completed_target_slot ||
                wave->seeds[seed].resident_slot == target_slot;
          }
          DataSyncBarrier<MemDsbT::DDR>();
          if (completed_target_slot && target_slot_ready != nullptr) {
            *target_slot_ready = true;
          }
        } else if (wave_has_target && target_failure_reason != nullptr &&
                   wave->failure_reason != 0U) {
          *target_failure_reason = wave->failure_reason;
        }
        seedBufferBusy_[wave->buffer_index] = false;
        wave->seed_count = 0U;
        wave->seeds = nullptr;
      }
      if (!pending) {
        return true;
      }
      if (!terminal_mode) {
        return false;
      }
    }
    for (uint32_t index = 0U;
         index < A5_P10_STAGE_B_LOCAL_WAVE_CAPACITY; ++index) {
      if (waves_[index].posted != 0U) {
        return false;
      }
    }
    return true;
  }

  __aicore__ inline bool DrainLocalWaves(bool terminal_mode) {
    return DrainLocalWavesForTarget(
        terminal_mode, A5_P10_STAGE_B_UNMAPPED_SLOT, nullptr);
  }

  __aicore__ inline bool EnsureFinalWinnerReady(
      const A5P10ForwardResidentSeed& seed, uint32_t rank) {
    if (seed.resident_slot < 0) {
      return false;
    }
    uint32_t last_failure_reason = 0U;
    for (uint32_t attempt = 0U; attempt < tiling_->boundedDrainPolls;
         ++attempt) {
      const int32_t state = ReadGmByPassDCache<int32_t>(
          residentReady_ + seed.resident_slot);
      if (state == static_cast<int32_t>(A5P10ResidentReadyState::READY)) {
        return true;
      }
      if (state == static_cast<int32_t>(A5P10ResidentReadyState::EMPTY)) {
        const uint32_t buffer_index = FindFreeSeedBuffer();
        if (buffer_index != kNoBuffer) {
          // See the scalar UB-store rule in ConsumeExactScoreReceipt().
          localSeeds_[buffer_index][0].row = seed.row;
          localSeeds_[buffer_index][0].source = seed.source;
          localSeeds_[buffer_index][0].kv_object = seed.kv_object;
          localSeeds_[buffer_index][0].resident_slot = seed.resident_slot;
          const uint32_t post_failure_reason =
              StartMicrobatch(buffer_index, 1U);
          if (post_failure_reason != 0U && last_failure_reason == 0U) {
            last_failure_reason = post_failure_reason;
          }
        } else {
          if (last_failure_reason == 0U) {
            last_failure_reason = A5_P10_STAGE_B_POST_NO_FREE_WAVE;
          }
        }
      }
      // This is terminal-only progress; score scanning never waits for CQ.
      // A successful terminal for this exact slot is stronger than a second
      // same-kernel GM read: it proves HCCP completed and this worker has
      // published READY with a DDR barrier.  The latter GM read remains the
      // path for a slot completed by another worker.
      bool local_terminal_ready = false;
      uint32_t drain_failure_reason = 0U;
      (void)DrainLocalWavesForTarget(
          true, seed.resident_slot, &local_terminal_ready,
          &drain_failure_reason);
      if (local_terminal_ready) {
        return true;
      }
      if (drain_failure_reason != 0U && last_failure_reason == 0U) {
        last_failure_reason = drain_failure_reason;
      }
    }
    const int32_t final_state = ReadGmByPassDCache<int32_t>(
        residentReady_ + seed.resident_slot);
    if (final_state == static_cast<int32_t>(A5P10ResidentReadyState::READY)) {
      return true;
    }
    if (last_failure_reason == 0U) {
      last_failure_reason = final_state ==
              static_cast<int32_t>(A5P10ResidentReadyState::IN_FLIGHT)
          ? A5_P10_STAGE_B_POST_CQ_PENDING
          : A5_P10_STAGE_B_POST_BEGIN_UNKNOWN;
    }
    RecordFirstFailure(PostDiagnostic(static_cast<uint32_t>(seed.row), rank,
                                      last_failure_reason));
    return false;
  }

  // A second worker may have won the object claim while this worker reaches a
  // final canonical winner.  CLAIMING is therefore not a final miss: wait a
  // bounded number of device turns for its immutable object->resident-row
  // publication, then use that row.  This does not retry/recycle a slot and
  // does not turn a valid winner into a false failure merely due to a
  // speculative/recall race.
  __aicore__ inline bool ResolveConcurrentForwardSlot(
      uint64_t source_offset, uint64_t object_offset, int32_t* resident_slot) {
    if (resident_slot == nullptr) {
      return false;
    }
    for (uint32_t attempt = 0U; attempt < tiling_->boundedDrainPolls;
         ++attempt) {
      const int32_t source_value = ReadGmByPassDCache<int32_t>(
          sourceToForwardSlot_ + source_offset);
      if (source_value >= 0) {
        *resident_slot = source_value;
        return true;
      }
      const int32_t object_value = ReadGmByPassDCache<int32_t>(
          kvObjectToForwardSlot_ + object_offset);
      if (object_value >= 0) {
        // This is AICore code, whereas asc_atomic_cas is a SIMT-only GM
        // primitive.  On Ascend 950 scalar AIC AtomicCas supports uint32_t;
        // the signed slot states are stable two's-complement bit patterns and
        // the map is naturally four-byte aligned.  This is cache convergence
        // only: the positive object map was already published by the winner.
        (void)AscendC::AtomicCas<uint32_t>(
            reinterpret_cast<__gm__ uint32_t*>(
                sourceToForwardSlot_ + source_offset),
            static_cast<uint32_t>(A5_P10_STAGE_B_CLAIMING_SLOT),
            static_cast<uint32_t>(object_value));
        *resident_slot = object_value;
        return true;
      }
      // -1 means there is no in-flight owner to await. Anything other than
      // the one durable CLAIMING state is terminal for this bounded resolve.
      if (source_value != A5_P10_STAGE_B_CLAIMING_SLOT &&
          object_value != A5_P10_STAGE_B_CLAIMING_SLOT) {
        return false;
      }
      (void)DrainLocalWaves(false);
      PollReceiptQuantumBoundary();
    }
    return false;
  }

  // `attention_slots` carries QSFA logical slot ids, not physical resident
  // row numbers. The bridge cold-binds this exact W-entry mapping alongside
  // resident payload/ready tables; identity mode is deliberately forbidden.
  __aicore__ inline bool ResolveQsfaSlot(int32_t resident_row,
                                         int32_t* qsfa_slot) const {
    if (qsfa_slot == nullptr || resident_row < 0 ||
        static_cast<uint32_t>(resident_row) >= tiling_->residentRowCount) {
      return false;
    }
    const int32_t mapped = ReadGmByPassDCache<int32_t>(
        residentRowToQsfaSlot_ + resident_row);
    if (mapped < 0) {
      return false;
    }
    *qsfa_slot = mapped;
    return true;
  }

  __aicore__ inline int32_t FinalizeDiagnostic(
      uint32_t row, uint32_t rank,
      A5P10StageBFinalizeFailureReason reason) const {
    const uint64_t ordinal = static_cast<uint64_t>(row) * topkPerRow_ + rank;
    const uint64_t encoded = ordinal *
            A5_P10_STAGE_B_DIAGNOSTIC_FINALIZE_DETAIL_STRIDE +
        static_cast<uint32_t>(reason);
    // The graph ABI bounds B and K far below this limit.  Keep a generic
    // fail-closed code if a malformed dynamic input ever defeats that proof.
    if (encoded > 0x3fffffffULL) {
      return A5_P10_STAGE_B_DIAGNOSTIC_FINALIZE;
    }
    return A5_P10_STAGE_B_DIAGNOSTIC_FINALIZE_DETAIL_BASE -
        static_cast<int32_t>(encoded);
  }

  __aicore__ inline int32_t PostDiagnostic(uint32_t row, uint32_t rank,
                                            uint32_t reason) const {
    const uint64_t ordinal = static_cast<uint64_t>(row) * topkPerRow_ + rank;
    const uint64_t encoded = ordinal *
            A5_P10_STAGE_B_DIAGNOSTIC_POST_DETAIL_STRIDE +
        reason;
    if (reason == 0U ||
        reason >= A5_P10_STAGE_B_DIAGNOSTIC_POST_DETAIL_STRIDE ||
        encoded > 0x3fffffffULL) {
      return A5_P10_STAGE_B_DIAGNOSTIC_FINALIZE;
    }
    return A5_P10_STAGE_B_DIAGNOSTIC_POST_DETAIL_BASE -
        static_cast<int32_t>(encoded);
  }

  __aicore__ inline bool FinalizeAssignedRows() {
    for (uint32_t row = worker_; row < activeRows_;
         row += tiling_->executorWorkerCount) {
      const int32_t selected_length = ReadGmByPassDCache<int32_t>(
          selectedSlotLengths_ + row);
      if (selected_length < static_cast<int32_t>(topkPerRow_) ||
          static_cast<uint32_t>(selected_length) >
              tiling_->attentionSlotCapacity) {
        RecordFirstFailure(FinalizeDiagnostic(
            row, 0U, A5_P10_STAGE_B_FINALIZE_BAD_SELECTED_LENGTH));
        return false;
      }
      for (uint32_t rank = 0U; rank < topkPerRow_; ++rank) {
        const uint64_t canonical_offset = static_cast<uint64_t>(row) *
            topkPerRow_ + rank;
        const int32_t source_value = ReadGmByPassDCache<int32_t>(
            canonicalSparseIndices_ + canonical_offset);
        const uint64_t attention_offset = static_cast<uint64_t>(row) *
            tiling_->attentionSlotCapacity + rank;
        if (source_value < 0 || static_cast<uint32_t>(source_value) >=
                                    tiling_->sourceCapacity) {
          WriteGmByPassDCache<int32_t>(
              attentionSlots_ + attention_offset,
              A5_P10_STAGE_B_UNMAPPED_SLOT);
          continue;
        }
        const uint32_t source = static_cast<uint32_t>(source_value);
        A5P10ActiveRowMapping mapping{};
        if (!ResolveActiveRowMapping(row, source, &mapping, true)) {
          RecordFirstFailure(FinalizeDiagnostic(
              row, rank, A5_P10_STAGE_B_FINALIZE_ACTIVE_MAPPING));
          return false;
        }
        if (mapping.persistent_slot >= 0) {
          int32_t qsfa_slot = A5_P10_STAGE_B_UNMAPPED_SLOT;
          if (!ResolveQsfaSlot(mapping.persistent_slot, &qsfa_slot)) {
            RecordFirstFailure(FinalizeDiagnostic(
                row, rank, A5_P10_STAGE_B_FINALIZE_PERSISTENT_QSFA));
            return false;
          }
          WriteGmByPassDCache<int32_t>(attentionSlots_ + attention_offset,
                                        qsfa_slot);
          continue;
        }
        const uint64_t source_offset = static_cast<uint64_t>(row) *
            tiling_->sourceCapacity + source;
        const uint64_t object_offset = static_cast<uint64_t>(
            mapping.pool_entry) * tiling_->kvObjectCapacity +
            static_cast<uint32_t>(mapping.object_local);
        int32_t resident_slot = ReadGmByPassDCache<int32_t>(
            sourceToForwardSlot_ + source_offset);
        bool newly_claimed = false;
        if (resident_slot < 0 && !ClaimRecallSlot(source_offset, object_offset,
                                                   row, rank,
                                                   &resident_slot,
                                                   &newly_claimed)) {
          if (!ResolveConcurrentForwardSlot(source_offset, object_offset,
                                            &resident_slot)) {
            RecordFirstFailure(FinalizeDiagnostic(
                row, rank, A5_P10_STAGE_B_FINALIZE_SLOT_CLAIM));
            return false;
          }
        }
        A5P10ForwardResidentSeed seed{static_cast<int32_t>(row),
                                      static_cast<int32_t>(source),
                                      mapping.kv_object, resident_slot};
        if (!EnsureFinalWinnerReady(seed, rank)) {
          RecordFirstFailure(FinalizeDiagnostic(
              row, rank, A5_P10_STAGE_B_FINALIZE_WINNER_READY));
          return false;
        }
        int32_t qsfa_slot = A5_P10_STAGE_B_UNMAPPED_SLOT;
        if (!ResolveQsfaSlot(resident_slot, &qsfa_slot)) {
          RecordFirstFailure(FinalizeDiagnostic(
              row, rank, A5_P10_STAGE_B_FINALIZE_FINAL_QSFA));
          return false;
        }
        WriteGmByPassDCache<int32_t>(attentionSlots_ + attention_offset,
                                      qsfa_slot);
      }
      // [K,K+tail) remains the caller's native tail materialization. The
      // selected length is retained verbatim for the existing QSFA ABI.
    }
    DataSyncBarrier<MemDsbT::DDR>();
    return true;
  }

  __aicore__ inline void MarkWorkerComplete() {
    completion_->is_last = 0U;
    set_flag(PIPE_S, PIPE_V, simtEvent_);
    wait_flag(PIPE_S, PIPE_V, simtEvent_);
    AscendC::Simt::VF_CALL<MarkWorkerCompleteSimtService>(
        AscendC::Simt::Dim3{1U, 1U, 1U},
        forwardControl_ + A5_P10_STAGE_B_EXECUTOR_PROGRESS,
        tiling_->executorWorkerCount, completion_);
    set_flag(PIPE_V, PIPE_S, simtEvent_);
    wait_flag(PIPE_V, PIPE_S, simtEvent_);
    if (completion_->is_last != 0U) {
      DataSyncBarrier<MemDsbT::DDR>();
      // COMPLETE is the final release flag for the Stage-B -> QSFA graph
      // edge.  It must not become visible before the bypass-published ready
      // map and attention-slot prefix are globally ordered.
      WriteGmByPassDCache<int32_t>(
          forwardControl_ + A5_P10_STAGE_B_FORWARD_STATE,
          A5_P10_STAGE_B_FORWARD_STATE_COMPLETE);
      DataSyncBarrier<MemDsbT::DDR>();
    }
  }

  const A5P10ScoreReceiptForwardResidentTilingData* tiling_ = nullptr;
  __ubuf__ A5P10ForwardResidentSeed*
      localSeeds_[A5_P10_STAGE_B_LOCAL_WAVE_CAPACITY]{};
  __ubuf__ kvaf_forward_resident_get::KvafForwardResidentQueueLeaseResult*
      leaseResult_ = nullptr;
  __ubuf__ ReceiptClaimResult* receiptClaim_ = nullptr;
  __ubuf__ ReceiptDoneResult* receiptDone_ = nullptr;
  __ubuf__ SlotClaimResult* slotClaim_ = nullptr;
  __ubuf__ CompletionResult* completion_ = nullptr;
  StageBWave waves_[A5_P10_STAGE_B_LOCAL_WAVE_CAPACITY]{};
  bool seedBufferBusy_[A5_P10_STAGE_B_LOCAL_WAVE_CAPACITY]{};

  __gm__ uint16_t* scoreSurface_ = nullptr;
  __gm__ int64_t* receiptRing_ = nullptr;
  __gm__ int32_t* receiptState_ = nullptr;
  __gm__ int32_t* forwardControl_ = nullptr;
  __gm__ int32_t* activeRowPoolEntries_ = nullptr;
  __gm__ int32_t* kvObjectMap_ = nullptr;
  __gm__ int32_t* kvObjectSlotsSnapshot_ = nullptr;
  __gm__ int32_t* kvObjectToForwardSlot_ = nullptr;
  __gm__ int32_t* sourceToForwardSlot_ = nullptr;
  __gm__ int32_t* speculativeSlotBand_ = nullptr;
  __gm__ int32_t* speculativeSlotCursor_ = nullptr;
  __gm__ int32_t* recallSlotBand_ = nullptr;
  __gm__ int32_t* canonicalSparseIndices_ = nullptr;
  __gm__ int32_t* canonicalScoreSourceLens_ = nullptr;
  __gm__ const KvafForwardResidentBatchGetControl* controlGm_ = nullptr;
  __gm__ int32_t* queueLeases_ = nullptr;
  __gm__ uint8_t* residentPayload_ = nullptr;
  __gm__ int32_t* residentReady_ = nullptr;
  __gm__ int32_t* residentRowToQsfaSlot_ = nullptr;
  __gm__ int32_t* attentionSlots_ = nullptr;
  __gm__ int32_t* selectedSlotLengths_ = nullptr;

  A5P10ResidentSlotBand speculativeBand_{};
  A5P10ResidentSlotBand recallBand_{};
  uint32_t worker_ = 0U;
  uint32_t receiptCapacity_ = 0U;
  uint32_t activeRows_ = 0U;
  uint32_t rowsPerObject_ = 0U;
  uint32_t topkPerRow_ = 0U;
  uint32_t scoreCutoffU16_ = 0U;
  // Dense KVAF PostOnly reserves event IDs 0 and 1 for its two WQE banks;
  // generic KVAF compaction reserves 2.  Stage-B's SIMT handshakes therefore
  // use 3.  The event is per AIV invocation, not a fixed Get-core assignment.
  event_t simtEvent_ = static_cast<event_t>(EVENT_ID3);
  bool enabled_ = false;
};

}  // namespace
}  // namespace a5_p10_stage_b

extern "C" __global__ __aicore__ void a5_p10_score_receipt_forward_resident(
    GM_ADDR score_surface, GM_ADDR receipt_ring, GM_ADDR receipt_state,
    GM_ADDR forward_control, GM_ADDR active_row_pool_entries,
    GM_ADDR kv_object_map, GM_ADDR kv_object_slots_snapshot,
    GM_ADDR kv_object_to_forward_slot, GM_ADDR source_to_forward_slot,
    GM_ADDR speculative_slot_band,
    GM_ADDR speculative_slot_cursor, GM_ADDR recall_slot_band,
    GM_ADDR canonical_sparse_indices, GM_ADDR canonical_score_source_lens,
    GM_ADDR kvaf_forward_resident_control, GM_ADDR queue_leases,
    GM_ADDR resident_payload, GM_ADDR resident_ready,
    GM_ADDR resident_row_to_qsfa_slot, GM_ADDR attention_slots_buffer,
    GM_ADDR selected_slot_lengths_buffer,
    GM_ADDR receipt_state_alias, GM_ADDR forward_control_alias,
    GM_ADDR kv_object_to_forward_slot_alias,
    GM_ADDR source_to_forward_slot_alias, GM_ADDR speculative_slot_cursor_alias,
    GM_ADDR queue_leases_alias, GM_ADDR resident_payload_alias,
    GM_ADDR resident_ready_alias, GM_ADDR attention_slots,
    GM_ADDR selected_slot_lengths, GM_ADDR resident_seq_lengths,
    GM_ADDR workspace, GM_ADDR tiling) {
  (void)receipt_state_alias;
  (void)forward_control_alias;
  (void)kv_object_to_forward_slot_alias;
  (void)source_to_forward_slot_alias;
  (void)speculative_slot_cursor_alias;
  (void)queue_leases_alias;
  (void)resident_payload_alias;
  (void)resident_ready_alias;
  (void)attention_slots;
  (void)selected_slot_lengths;
  (void)resident_seq_lengths;
  (void)workspace;
  // Stage B owns SIMT/AIV scans, queue leases and Direct-HCCP progression;
  // it has no Cube/AIC branch.  This must be AIV_ONLY so GetBlockIdx() is the
  // exact logical executor worker id selected by its tiler.
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(A5P10ScoreReceiptForwardResidentTilingData);
  GET_TILING_DATA(tiling_data, tiling);
  a5_p10_stage_b::A5P10ScoreReceiptForwardResidentKernel op(&tiling_data);
  op.Init(score_surface, receipt_ring, receipt_state, forward_control,
          active_row_pool_entries, kv_object_map, kv_object_slots_snapshot,
          kv_object_to_forward_slot, source_to_forward_slot,
          speculative_slot_band,
          speculative_slot_cursor, recall_slot_band,
          canonical_sparse_indices, canonical_score_source_lens,
          kvaf_forward_resident_control, queue_leases, resident_payload,
          resident_ready, resident_row_to_qsfa_slot, attention_slots_buffer,
          selected_slot_lengths_buffer);
  op.Process();
}
