#ifndef A5_P10_SCORE_RECEIPT_FORWARD_RESIDENT_ABI_H
#define A5_P10_SCORE_RECEIPT_FORWARD_RESIDENT_ABI_H

#include <cstdint>

#include "../../stage_a_w4a4c8_score_receipt/op_kernel/a5_p10_w4a4c8_score_receipt_abi.h"

// This ABI deliberately keeps transport ownership and resident placement
// separate.  Queue ownership is transient; resident rows are fixed for one
// forward and never released or remapped by Stage B.
constexpr uint32_t A5_P10_STAGE_B_CONTROL_WORDS = 10U;
constexpr uint32_t A5_P10_SLOT_BAND_WORDS = 2U;
constexpr uint32_t A5_P10_STAGE_B_MICROBATCH_CAPACITY = 64U;
constexpr uint32_t A5_P10_STAGE_B_LOCAL_WAVE_CAPACITY = 2U;
constexpr uint32_t A5_P10_STAGE_B_FINAL_DRAIN_POLLS = 1024U;

// The score threshold occupies only the low 16 bits of forward_control[6].
// The high diagnostic bit is opt-in and deliberately has no scheduling or
// numerical meaning.  It lets a failed performance investigation leave a
// compact per-receipt trail in the descriptor's otherwise-reserved high word,
// without adding a graph input, a C8 buffer, or a host-side synchronization.
constexpr uint32_t A5_P10_STAGE_B_SCORE_CONTROL_TRACE_ENABLE = 0x80000000U;

// Stage B owns a receipt after READY -> CLAIMED, so it alone may write this
// trace bitset.  Each bit says that at least one item in the receipt reached a
// particular gate; it is not a counter and it never participates in transport
// ownership, Top-K, resident placement, or QSFA.  The host reads it only after
// the Stage-B completion event in a trace-enabled diagnostic run.
enum A5P10StageBReceiptTraceBit : uint32_t {
  A5_P10_STAGE_B_TRACE_ENTERED = 1U << 0U,
  A5_P10_STAGE_B_TRACE_DESCRIPTOR_VALID = 1U << 1U,
  A5_P10_STAGE_B_TRACE_DESCRIPTOR_INVALID = 1U << 2U,
  A5_P10_STAGE_B_TRACE_SEED_BUFFER_UNAVAILABLE = 1U << 3U,
  A5_P10_STAGE_B_TRACE_SOURCE_SCANNED = 1U << 4U,
  A5_P10_STAGE_B_TRACE_SCORE_ELIGIBLE = 1U << 5U,
  A5_P10_STAGE_B_TRACE_MAPPING_VALID = 1U << 6U,
  A5_P10_STAGE_B_TRACE_PERSISTENT_HIT = 1U << 7U,
  A5_P10_STAGE_B_TRACE_MISS_CANDIDATE = 1U << 8U,
  A5_P10_STAGE_B_TRACE_CLAIM_ATTEMPTED = 1U << 9U,
  A5_P10_STAGE_B_TRACE_CLAIMED_NEW = 1U << 10U,
  A5_P10_STAGE_B_TRACE_POST_ATTEMPTED = 1U << 11U,
  A5_P10_STAGE_B_TRACE_POSTED = 1U << 12U,
  A5_P10_STAGE_B_TRACE_POST_DEFERRED = 1U << 13U,
  // These two bits are trace-only proof that a successful speculative
  // PostOnly wave reached CQ terminal.  The latter is set only when that
  // terminal was observed while Stage A was still OPEN, which is the minimum
  // evidence required before interpreting a latency delta as hidden HCCP
  // work rather than later final recall.
  A5_P10_STAGE_B_TRACE_CQ_READY = 1U << 14U,
  A5_P10_STAGE_B_TRACE_PRESEAL_CQ_READY = 1U << 15U,
};

constexpr int32_t A5_P10_STAGE_B_UNMAPPED_SLOT = -1;
constexpr int32_t A5_P10_STAGE_B_CLAIMING_SLOT = -2;
constexpr int32_t A5_P10_STAGE_B_FORWARD_STATE_INCOMPLETE = 0;
constexpr int32_t A5_P10_STAGE_B_FORWARD_STATE_COMPLETE = 1;

// Word 1 is intentionally not a producer->consumer scheduling input: Stage B
// always walks the graph-static tiling receiptCapacity rather than waiting for
// a dynamically published count.  Reuse that otherwise-reserved word as a
// compact, forward-local fail-closed ledger.  It is reset by ForwardPrepare,
// written only by Stage B, and never changes C8 scores, Top-K, queue ownership
// or resident-slot placement.  A negative value is the first terminal local
// failure observed by any logical Stage-B worker; RUNNING merely proves that
// Init completed and Process entered.
constexpr int32_t A5_P10_STAGE_B_DIAGNOSTIC_CLEAR = 0;
constexpr int32_t A5_P10_STAGE_B_DIAGNOSTIC_RUNNING = 1;
constexpr int32_t A5_P10_STAGE_B_DIAGNOSTIC_INIT_COMMON = -101;
constexpr int32_t A5_P10_STAGE_B_DIAGNOSTIC_INIT_GEOMETRY = -102;
constexpr int32_t A5_P10_STAGE_B_DIAGNOSTIC_INIT_KVAF = -103;
constexpr int32_t A5_P10_STAGE_B_DIAGNOSTIC_INIT_UB = -104;
constexpr int32_t A5_P10_STAGE_B_DIAGNOSTIC_RECEIPT_TERMINAL = -201;
constexpr int32_t A5_P10_STAGE_B_DIAGNOSTIC_FINALIZE = -202;
constexpr int32_t A5_P10_STAGE_B_DIAGNOSTIC_DRAIN = -203;

// A failed final winner must be actionable from one target run.  Keep the
// detail in the existing diagnostic ledger rather than adding a mutable
// tensor or another lifecycle state.  A detailed code is
// `BASE - (row * K + rank) * STRIDE - reason`; the host acceptance harness
// decodes it only after the forward has terminally failed.  This word remains
// diagnostic-only and never participates in C8, queue, or slot scheduling.
constexpr int32_t A5_P10_STAGE_B_DIAGNOSTIC_FINALIZE_DETAIL_BASE = -202000;
constexpr uint32_t A5_P10_STAGE_B_DIAGNOSTIC_FINALIZE_DETAIL_STRIDE = 8U;
enum A5P10StageBFinalizeFailureReason : uint32_t {
  A5_P10_STAGE_B_FINALIZE_BAD_SELECTED_LENGTH = 1U,
  A5_P10_STAGE_B_FINALIZE_ACTIVE_MAPPING = 2U,
  A5_P10_STAGE_B_FINALIZE_PERSISTENT_QSFA = 3U,
  A5_P10_STAGE_B_FINALIZE_SLOT_CLAIM = 4U,
  A5_P10_STAGE_B_FINALIZE_WINNER_READY = 5U,
  A5_P10_STAGE_B_FINALIZE_FINAL_QSFA = 6U,
};

// Post/CQ detail uses the same existing diagnostic word as finalization
// detail.  It is intentionally local to a failed forward: it does not add a
// graph input/output, a scheduling edge, or any transport lifecycle state.
// A detailed code is `BASE - (row * K + rank) * STRIDE - reason`.
constexpr int32_t A5_P10_STAGE_B_DIAGNOSTIC_POST_DETAIL_BASE = -210000;
constexpr uint32_t A5_P10_STAGE_B_DIAGNOSTIC_POST_DETAIL_STRIDE = 32U;
enum A5P10StageBPostFailureReason : uint32_t {
  A5_P10_STAGE_B_POST_INVALID_CALL = 1U,
  A5_P10_STAGE_B_POST_NO_FREE_WAVE = 2U,
  A5_P10_STAGE_B_POST_CONTROL_LOAD = 3U,
  A5_P10_STAGE_B_POST_QUEUE_ACQUIRE = 4U,
  A5_P10_STAGE_B_POST_DESCRIPTOR_BUILD = 5U,
  A5_P10_STAGE_B_POST_BEGIN_INVALID_ARGS = 6U,
  A5_P10_STAGE_B_POST_BEGIN_INVALID_ENTITY = 7U,
  A5_P10_STAGE_B_POST_BEGIN_INVALID_QUEUE = 8U,
  A5_P10_STAGE_B_POST_BEGIN_INVALID_LOOKUP = 9U,
  A5_P10_STAGE_B_POST_BEGIN_QUEUE_FULL = 10U,
  A5_P10_STAGE_B_POST_BEGIN_INVALID_OUTPUT = 11U,
  A5_P10_STAGE_B_POST_BEGIN_UNKNOWN = 12U,
  A5_P10_STAGE_B_POST_CQ_TIMEOUT = 13U,
  A5_P10_STAGE_B_POST_CQ_ERROR = 14U,
  A5_P10_STAGE_B_POST_CQ_UNKNOWN = 15U,
  A5_P10_STAGE_B_POST_QUEUE_RETURN = 16U,
  A5_P10_STAGE_B_POST_CQ_PENDING = 17U,
  // A dense Begin with a non-success result is classified above.  These
  // codes cover the otherwise-impossible case where Begin returns false but
  // does not leave an explicit dense failure.  It must fail closed: a caller
  // cannot prove whether the doorbell was reached, so it must not recycle the
  // queue from this path.
  A5_P10_STAGE_B_POST_BEGIN_STATE_IDLE = 18U,
  A5_P10_STAGE_B_POST_BEGIN_STATE_RETIRED = 19U,
  A5_P10_STAGE_B_POST_BEGIN_STATE_FAILED_SUCCESS = 20U,
  A5_P10_STAGE_B_POST_BEGIN_STATE_IN_FLIGHT = 21U,
  A5_P10_STAGE_B_POST_BEGIN_STATE_UNKNOWN = 22U,
  A5_P10_STAGE_B_POST_BEGIN_RESULT_UNKNOWN = 23U,
  // PostOnly was confirmed, but the companion could not reconstruct the
  // CQ-drain record from the immutable published descriptor.  This is an
  // ambiguous post state: preserve the queue and fail the forward rather
  // than treating it as a no-post or reusing the transport resource.
  A5_P10_STAGE_B_POST_BEGIN_REHYDRATE = 24U,
  // KVAF observed a nonzero but nonexact SQ producer-index delta. A doorbell
  // may have reached hardware, so this is quarantine/fail-closed rather than
  // a retryable queue-full result.
  A5_P10_STAGE_B_POST_BEGIN_SQ_PI_DELTA_MISMATCH = 25U,
};

// A persistent Stage-B companion starts from the same cold-bound mutable
// tensors as Stage A.  It must not be connected to Stage A through a graph
// completion edge: it polls receipt_state while this seal remains OPEN.
enum class A5P10ProducerSealState : int32_t {
  OPEN = 0,
  SEALED = 1,
};

enum A5P10StageBControlWord : uint32_t {
  A5_P10_STAGE_B_PRODUCER_SEALED = 0U,
  // The physical receipt range remains the graph-static tiling
  // receiptCapacity.  This word is deliberately diagnostic-only; it must not
  // become a dynamic receipt count or scheduling dependency.
  A5_P10_STAGE_B_DIAGNOSTIC = 1U,
  A5_P10_STAGE_B_TOPK_PER_ROW = 2U,
  A5_P10_STAGE_B_ROWS_PER_OBJECT = 3U,
  A5_P10_STAGE_B_ACTIVE_ROWS = 4U,
  A5_P10_STAGE_B_FORWARD_STATE = 5U,
  // Low 16 bits of the native uint16 score cut-off. Stage A does not read
  // this word: ExactScoreScan is wholly a Stage-B operation.
  A5_P10_STAGE_B_SCORE_CUTOFF_U16 = 6U,
  // A bounded Stage-B progress counter, never an arena generation or a
  // cross-forward lifetime token.
  A5_P10_STAGE_B_EXECUTOR_PROGRESS = 7U,
  // Diagnostic only: Stage A writes its final static receipt sequence before
  // it writes PRODUCER_SEALED. Stage B atomically latches the first READY
  // receipt sequence it claims while it observes OPEN. They prove real
  // pre-seal work without participating in control-flow or resource reuse.
  A5_P10_STAGE_B_PRODUCER_SEAL_SEQUENCE = 8U,
  A5_P10_STAGE_B_FIRST_PRESEAL_READY_CLAIM_SEQUENCE = 9U,
};
static_assert(A5_P10_STAGE_B_PRODUCER_SEAL_SEQUENCE ==
                  A5_P10_FORWARD_CONTROL_PRODUCER_SEAL_SEQUENCE_WORD,
              "Stage-A/Stage-B producer seal witness index drifted");
static_assert(A5_P10_STAGE_B_FIRST_PRESEAL_READY_CLAIM_SEQUENCE ==
                  A5_P10_FORWARD_CONTROL_FIRST_PRESEAL_READY_CLAIM_WORD,
              "Stage-A/Stage-B preseal claim witness index drifted");

// `[base, capacity]`, both absolute rows in the cold-bound resident buffer.
struct A5P10ResidentSlotBand {
  int32_t base;
  int32_t capacity;
};
static_assert(sizeof(A5P10ResidentSlotBand) ==
                  A5_P10_SLOT_BAND_WORDS * sizeof(int32_t),
              "resident slot-band ABI changed");

// A compact seed copied into one of Stage B's local 64-entry microbatches.
// `resident_slot` is fixed before PostOnly; it is never a queue-private row.
struct A5P10ForwardResidentSeed {
  int32_t row;
  int32_t source;
  int32_t kv_object;
  int32_t resident_slot;
};
static_assert(sizeof(A5P10ForwardResidentSeed) == 16U,
              "candidate seed ABI changed");

enum class A5P10ResidentReadyState : int32_t {
  EMPTY = 0,
  // A fixed resident row has an immutable source/object mapping and one
  // worker owns a seed in its private 2x64 UB staging buffer.  It is not
  // readable by QSFA and it is not yet a durable HCCP descriptor.  This
  // forward-local state closes the only cross-worker handoff gap: a second
  // worker must wait for PENDING to become EMPTY (discarded at seal),
  // IN_FLIGHT, or READY rather than submitting the same row a second time.
  // Keeping it negative preserves the established externally visible
  // EMPTY=0 / IN_FLIGHT=1 / READY=2 representation.
  PENDING = -3,
  IN_FLIGHT = 1,
  READY = 2,
};

// Device-side materialization resolves C8's row/source identity to the exact
// KVAF token-object domain; it must not invent an unrelated S -> O mapping.
// Let P be active_row_pool_entries[R], rows_per_object=128, and:
//
//   O_token = ((P * num_layers + layer_id) * qcap * rows_per_object) + S.
//
// kv_object_map[P,S] is a cold-bound materialized witness of O_token and the
// kernel rejects it unless it equals that formula. Snapshot and forward-slot
// directories are keyed by [P,S] for this one layer invocation. KVAF receives
// O_token and derives exactly:
// table_index = O_token / 128 = P*num_layers*qcap + layer_id*qcap + (S>>7)
// and byte_offset = (O_token % 128) * 656 = (S&127) * 656.
// This preserves the existing access-table domain while keeping a P-local
// de-dup directory. Earlier direct BxS object/slot tensors are deliberately
// not part of the P10 graph ABI.
struct A5P10ActiveRowMapping {
  int32_t pool_entry;
  int32_t kv_object;       // exact global KVAF token object O_token
  int32_t object_local;    // S, the one-layer [P,S] directory coordinate
  int32_t persistent_slot;
};
static_assert(sizeof(A5P10ActiveRowMapping) == 16U,
              "active-row mapping ABI changed");

// Stage B consumes a pre-materialized QSFA tail. It replaces only [0,K) in
// the caller-owned full [B,1,K+tail] attention-slot buffer and preserves the
// tail verbatim. `resident_row_to_qsfa_slot[W]` is a required cold-bound map:
// resident payload row ids are physical forward-resident rows, whereas
// attention_slots holds QSFA logical slots resolved through the downstream
// resident block table. Identity mapping is not assumed. selected_slot_lengths
// is the full selected length, not a Top-K-only count; resident_seq_lengths is
// its QSFA compatibility alias.
struct A5P10SelectedSlotView {
  uint32_t sparse_count;
  uint32_t attention_capacity;
  uint32_t selected_slot_length;
  uint32_t reserved;
};
static_assert(sizeof(A5P10SelectedSlotView) == 16U,
              "selected-slot view ABI changed");

// The host OpDef exposes the following graph-visible ABI. Shape symbols are
// intentional: B=batch, S=source capacity, R=exact forward receipt count,
// W=resident rows, K=native exact Top-K count. Stage B is launched from these
// cold-bound aliases independently of Stage A and polls receipt_state while
// producer_sealed is OPEN. Do not introduce a normal Stage-A -> Stage-B graph
// completion dependency; the graph dependency is Stage-B forward_state ->
// QSFA after the bounded terminal drain.
//
// Inputs
//   score_surface             uint16 [..]  (formal Stage-A GM surface)
//   receipt_ring              int64  [R, 5]
//   receipt_state             int32  [receiptCapacity], preinitialized before
//                             either Stage A or Stage B starts
//   forward_control           int32  [10], mutable; producer_sealed is
//                             written only after all published records; Stage
//                             B scans the static receiptCapacity range
//   active_row_pool_entries   int32  [B]       (row R -> pool P)
//   kv_object_map             int32  [P, S]    materialized exact O_token
//   kv_object_slots_snapshot  int32  [P, S]    ((P,S) -> persistent slot|-1)
//   kv_object_to_forward_slot  int32  [P, S], mutable de-dup table, initially -1
//   source_to_forward_slot    int32  [B, S], mutable, initially -1
//   speculative_slot_band     int32  [2]
//   speculative_slot_cursor   int32  [1], mutable
//   recall_slot_band          int32  [2]
//   canonical_sparse_indices  int32  [B, 1, K]
//   canonical_score_source_lens int32 [B]
//   kvaf_forward_resident_control int64 [..] (opaque cold-bound ABI)
//   queue_leases              int32  [Q], mutable graph-visible queue owners
//   resident_payload          uint8  [W, block_bytes], mutable
//   resident_ready            int32  [W], mutable
//   resident_row_to_qsfa_slot int32  [W], cold-bound; mandatory translation
//                             from resident row to QSFA logical slot
//   attention_slots_buffer    int32  [B,1,K+tail], mutable; its tail is
//                             materialized by the native downstream path
//   selected_slot_lengths_buffer int32 [B], mutable full QSFA lengths
//
// Outputs alias the mutable tensors and add QSFA-facing exact outputs:
//   kv_object_to_forward_slot_alias int32 [P, S]
//   attention_slots           int32 [B, 1, K+tail]
//   selected_slot_lengths     int32 [B]
//   resident_seq_lengths      int32 [B] (compatibility alias of selected
//                                        slot lengths for existing QSFA ABI)

#endif  // A5_P10_SCORE_RECEIPT_FORWARD_RESIDENT_ABI_H
