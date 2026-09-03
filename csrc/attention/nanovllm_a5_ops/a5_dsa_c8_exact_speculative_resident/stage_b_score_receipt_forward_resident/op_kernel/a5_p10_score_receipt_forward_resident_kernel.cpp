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

// A local wave owns one of the two local 64-seed buffers until CQ terminal.
// Its ownership persists for this invocation only, never across forwards.
struct StageBWave {
  kvaf_forward_resident_get::KvafForwardResidentBatchGetWave kvaf_wave;
  kvaf_forward_resident_get::KvafForwardResidentBatchGetTerminal terminal;
  __ubuf__ A5P10ForwardResidentSeed* seeds;
  uint32_t seed_count;
  uint32_t buffer_index;
  uint32_t posted;
};

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

  // The row is unique to this canonical object claim. EMPTY means it is
  // assigned but has not crossed the irreversible PostOnly boundary.
  resident_ready[selected] = static_cast<int32_t>(A5P10ResidentReadyState::EMPTY);
  AscendC::Simt::ThreadFence();
  if (asc_atomic_cas(kv_object_to_forward_slot + object_offset,
                     A5_P10_STAGE_B_CLAIMING_SLOT, selected) !=
      A5_P10_STAGE_B_CLAIMING_SLOT) {
    (void)asc_atomic_cas(source_to_forward_slot + source_offset,
                         A5_P10_STAGE_B_CLAIMING_SLOT,
                         A5_P10_STAGE_B_UNMAPPED_SLOT);
    return;
  }
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
    __gm__ const int64_t* receipt_ring, uint32_t receipt,
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
__aicore__ inline bool BeginForwardResidentMicrobatch(
    __gm__ const KvafForwardResidentBatchGetControl* control_gm,
    uint32_t producer_id, uint32_t queue_hint,
    __ubuf__ A5P10ForwardResidentSeed* seeds, uint32_t seed_count,
    uint32_t rows_per_object,
    __ubuf__ kvaf_forward_resident_get::KvafForwardResidentQueueLeaseResult*
        lease_result,
    event_t simt_event, StageBWave* output) {
  if (control_gm == nullptr || seeds == nullptr || seed_count == 0U ||
      seed_count > A5_P10_STAGE_B_MICROBATCH_CAPACITY ||
      rows_per_object == 0U || lease_result == nullptr || output == nullptr) {
    return false;
  }
  output->posted = 0U;
  output->seed_count = 0U;
  output->seeds = nullptr;

  kvaf_forward_resident_get::KvafForwardResidentBatchGetControl control{};
  if (!kvaf_forward_resident_get::LoadKvafForwardResidentBatchGetControlFromGmForAicore(
          control_gm, &control)) {
    return false;
  }
  kvaf_forward_resident_get::KvafForwardResidentQueueLease lease{};
  if (!kvaf_forward_resident_get::TryAcquireKvafForwardResidentQueueFromAicore(
          control_gm, producer_id, queue_hint, lease_result, simt_event,
          &lease)) {
    return false;  // bounded queue contention: final recall retries winners.
  }

  kvaf_forward_resident_get::KvafForwardResidentBatchGetDescriptorScratch
      descriptor{};
  bool descriptor_ready =
      kvaf_forward_resident_get::LoadKvafForwardResidentQueueTemplate(
          control, lease, &descriptor);
  for (uint32_t index = 0U; descriptor_ready && index < seed_count; ++index) {
    const A5P10ForwardResidentSeed& seed = seeds[index];
    descriptor_ready = seed.kv_object >= 0 && seed.resident_slot >= 0 &&
        kvaf_forward_resident_get::AppendKvafForwardResidentRequest(
            control, &descriptor, static_cast<uint32_t>(seed.kv_object),
            rows_per_object, seed.resident_slot);
  }
  if (!descriptor_ready) {
    (void)kvaf_forward_resident_get::ReturnKvafForwardResidentUnpostedQueueFromAicore(
        control_gm, lease_result, simt_event, &lease);
    return false;
  }
  if (!kvaf_forward_resident_get::BeginKvafForwardResidentBatchGet(
          control, lease, descriptor, &output->kvaf_wave)) {
    if (output->kvaf_wave.lease.state ==
        kvaf_forward_resident_get::kForwardResidentQueueTerminal) {
      (void)kvaf_forward_resident_get::FinishKvafForwardResidentBatchGet(
          &output->kvaf_wave, &output->terminal);
      (void)kvaf_forward_resident_get::ReturnKvafForwardResidentQueueFromAicore(
          control_gm, lease_result, simt_event, &output->kvaf_wave);
    }
    return false;
  }
  output->seeds = seeds;
  output->seed_count = seed_count;
  output->posted = 1U;
  return true;
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
  if (!kvaf_forward_resident_get::ReturnKvafForwardResidentQueueFromAicore(
          control_gm, lease_result, simt_event, &wave->kvaf_wave)) {
    return false;
  }
  wave->posted = 0U;
  *terminal = true;
  return success;
}

class A5P10ScoreReceiptForwardResidentKernel {
 public:
  __aicore__ inline A5P10ScoreReceiptForwardResidentKernel(
      TPipe* pipe, const A5P10ScoreReceiptForwardResidentTilingData* tiling)
      : pipe_(pipe), tiling_(tiling) {}

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
      seedBufferBusy_[index] = false;
    }

    if (pipe_ == nullptr || tiling_ == nullptr ||
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
      return;
    }

    pipe_->InitBuffer(seedBuffers_[0], A5_P10_STAGE_B_MICROBATCH_CAPACITY *
                                           sizeof(A5P10ForwardResidentSeed));
    pipe_->InitBuffer(seedBuffers_[1], A5_P10_STAGE_B_MICROBATCH_CAPACITY *
                                           sizeof(A5P10ForwardResidentSeed));
    pipe_->InitBuffer(leaseBuffer_, sizeof(
        kvaf_forward_resident_get::KvafForwardResidentQueueLeaseResult));
    pipe_->InitBuffer(receiptClaimBuffer_, sizeof(ReceiptClaimResult));
    pipe_->InitBuffer(receiptDoneBuffer_, sizeof(ReceiptDoneResult));
    pipe_->InitBuffer(slotClaimBuffer_, sizeof(SlotClaimResult));
    pipe_->InitBuffer(completionBuffer_, sizeof(CompletionResult));
    localSeeds_[0] = (__ubuf__ A5P10ForwardResidentSeed*)
        seedBuffers_[0].Get<uint8_t>().GetPhyAddr();
    localSeeds_[1] = (__ubuf__ A5P10ForwardResidentSeed*)
        seedBuffers_[1].Get<uint8_t>().GetPhyAddr();
    leaseResult_ = (__ubuf__ kvaf_forward_resident_get::
        KvafForwardResidentQueueLeaseResult*)
            leaseBuffer_.Get<uint8_t>().GetPhyAddr();
    receiptClaim_ = (__ubuf__ ReceiptClaimResult*)
        receiptClaimBuffer_.Get<uint8_t>().GetPhyAddr();
    receiptDone_ = (__ubuf__ ReceiptDoneResult*)
        receiptDoneBuffer_.Get<uint8_t>().GetPhyAddr();
    slotClaim_ = (__ubuf__ SlotClaimResult*)
        slotClaimBuffer_.Get<uint8_t>().GetPhyAddr();
    completion_ = (__ubuf__ CompletionResult*)
        completionBuffer_.Get<uint8_t>().GetPhyAddr();
    if (localSeeds_[0] == nullptr || localSeeds_[1] == nullptr ||
        leaseResult_ == nullptr || receiptClaim_ == nullptr ||
        receiptDone_ == nullptr ||
        slotClaim_ == nullptr || completion_ == nullptr) {
      return;
    }
    enabled_ = true;
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
    if (!FinalizeAssignedRows() || !DrainLocalWaves(true)) {
      return;
    }
    MarkWorkerComplete();
  }

 private:
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
      uint32_t row, uint32_t source, A5P10ActiveRowMapping* mapping) const {
    if (mapping == nullptr || row >= activeRows_ ||
        source >= tiling_->sourceCapacity) {
      return false;
    }
    const int32_t source_len = ReadGmByPassDCache<int32_t>(
        canonicalScoreSourceLens_ + row);
    if (source_len <= 0 || source >= static_cast<uint32_t>(source_len)) {
      return false;
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

  __aicore__ inline bool StartMicrobatch(
      uint32_t buffer_index, uint32_t seed_count) {
    if (buffer_index >= A5_P10_STAGE_B_LOCAL_WAVE_CAPACITY ||
        seed_count == 0U || seed_count > A5_P10_STAGE_B_MICROBATCH_CAPACITY) {
      return false;
    }
    uint32_t wave_index = FindFreeWave();
    if (wave_index == kNoBuffer) {
      (void)DrainLocalWaves(false);
      wave_index = FindFreeWave();
    }
    if (wave_index == kNoBuffer) {
      return false;
    }
    StageBWave* wave = waves_ + wave_index;
    wave->buffer_index = buffer_index;
    if (!BeginForwardResidentMicrobatch(
            controlGm_, worker_, worker_ + buffer_index, localSeeds_[buffer_index],
            seed_count, rowsPerObject_, leaseResult_, simtEvent_, wave)) {
      return false;
    }
    // Begin returned successfully, so descriptors are durable and the fixed
    // rows may now be observed as in-flight by final recall.
    for (uint32_t index = 0U; index < seed_count; ++index) {
      residentReady_[wave->seeds[index].resident_slot] =
          static_cast<int32_t>(A5P10ResidentReadyState::IN_FLIGHT);
    }
    DataSyncBarrier<MemDsbT::DDR>();
    seedBufferBusy_[buffer_index] = true;
    return true;
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
        if (!ResolveActiveRowMapping(row, source, &mapping) ||
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
        localSeeds_[buffer_index][seed_count++] = {
            static_cast<int32_t>(row), static_cast<int32_t>(source),
            mapping.kv_object, resident_slot};
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

  __aicore__ inline bool DrainLocalWaves(bool terminal_mode) {
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
        bool terminal = false;
        const bool success = DrainForwardResidentMicrobatch(
            controlGm_, leaseResult_, simtEvent_, wave, &terminal);
        if (!terminal) {
          continue;
        }
        // A terminal success alone makes a fixed resident slot READY. The
        // current target assumes successful posts, but this guard never
        // fabricates READY on a failed terminal.
        if (success) {
          for (uint32_t seed = 0U; seed < wave->seed_count; ++seed) {
            residentReady_[wave->seeds[seed].resident_slot] =
                static_cast<int32_t>(A5P10ResidentReadyState::READY);
          }
          DataSyncBarrier<MemDsbT::DDR>();
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

  __aicore__ inline bool EnsureFinalWinnerReady(
      const A5P10ForwardResidentSeed& seed) {
    if (seed.resident_slot < 0) {
      return false;
    }
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
          localSeeds_[buffer_index][0] = seed;
          (void)StartMicrobatch(buffer_index, 1U);
        }
      }
      // This is terminal-only progress; score scanning never waits for CQ.
      (void)DrainLocalWaves(true);
    }
    return ReadGmByPassDCache<int32_t>(residentReady_ + seed.resident_slot) ==
        static_cast<int32_t>(A5P10ResidentReadyState::READY);
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
        (void)asc_atomic_cas(sourceToForwardSlot_ + source_offset,
                             A5_P10_STAGE_B_CLAIMING_SLOT, object_value);
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

  __aicore__ inline bool FinalizeAssignedRows() {
    for (uint32_t row = worker_; row < activeRows_;
         row += tiling_->executorWorkerCount) {
      const int32_t selected_length = ReadGmByPassDCache<int32_t>(
          selectedSlotLengths_ + row);
      if (selected_length < static_cast<int32_t>(topkPerRow_) ||
          static_cast<uint32_t>(selected_length) >
              tiling_->attentionSlotCapacity) {
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
          attentionSlots_[attention_offset] = A5_P10_STAGE_B_UNMAPPED_SLOT;
          continue;
        }
        const uint32_t source = static_cast<uint32_t>(source_value);
        A5P10ActiveRowMapping mapping{};
        if (!ResolveActiveRowMapping(row, source, &mapping)) {
          return false;
        }
        if (mapping.persistent_slot >= 0) {
          int32_t qsfa_slot = A5_P10_STAGE_B_UNMAPPED_SLOT;
          if (!ResolveQsfaSlot(mapping.persistent_slot, &qsfa_slot)) {
            return false;
          }
          attentionSlots_[attention_offset] = qsfa_slot;
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
            return false;
          }
        }
        A5P10ForwardResidentSeed seed{static_cast<int32_t>(row),
                                      static_cast<int32_t>(source),
                                      mapping.kv_object, resident_slot};
        if (!EnsureFinalWinnerReady(seed)) {
          return false;
        }
        int32_t qsfa_slot = A5_P10_STAGE_B_UNMAPPED_SLOT;
        if (!ResolveQsfaSlot(resident_slot, &qsfa_slot)) {
          return false;
        }
        attentionSlots_[attention_offset] = qsfa_slot;
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
      forwardControl_[A5_P10_STAGE_B_FORWARD_STATE] =
          A5_P10_STAGE_B_FORWARD_STATE_COMPLETE;
      DataSyncBarrier<MemDsbT::DDR>();
    }
  }

  TPipe* pipe_ = nullptr;
  const A5P10ScoreReceiptForwardResidentTilingData* tiling_ = nullptr;
  TBuf<TPosition::VECCALC> seedBuffers_[A5_P10_STAGE_B_LOCAL_WAVE_CAPACITY];
  TBuf<TPosition::VECCALC> leaseBuffer_;
  TBuf<TPosition::VECCALC> receiptClaimBuffer_;
  TBuf<TPosition::VECCALC> receiptDoneBuffer_;
  TBuf<TPosition::VECCALC> slotClaimBuffer_;
  TBuf<TPosition::VECCALC> completionBuffer_;
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
  event_t simtEvent_ = static_cast<event_t>(EVENT_ID0);
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
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  REGISTER_TILING_DEFAULT(A5P10ScoreReceiptForwardResidentTilingData);
  GET_TILING_DATA(tiling_data, tiling);
  if ASCEND_IS_AIV {
    TPipe pipe;
    a5_p10_stage_b::A5P10ScoreReceiptForwardResidentKernel op(&pipe,
                                                               &tiling_data);
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
}
