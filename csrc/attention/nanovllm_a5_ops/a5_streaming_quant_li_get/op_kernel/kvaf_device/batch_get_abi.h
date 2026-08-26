#ifndef KVAF_AIV_BATCH_GET_ABI_H_
#define KVAF_AIV_BATCH_GET_ABI_H_

#include <stdint.h>

// Shared host/AIV ABI for the graph-safe ExecuteBatchGet kernel.  These
// records are copied verbatim between host code, HBM and the AIV kernel.
enum {
  KVAF_AIV_BATCH_GET_STATUS_BYTES = 384,
  KVAF_AIV_BATCH_GET_QUEUE_STATE_BYTES = 64,
  KVAF_AIV_BATCH_GET_MAX_AIV_CORES = 48,
  KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE = 2,
  KVAF_AIV_BATCH_GET_WQE_DATA_BYTES = 656,
  // One streaming Direct-HCCP wave is deliberately bounded so that the
  // device-side fused helper reuses the verified dense post-only path.
  KVAF_AIV_BATCH_GET_DENSE_REQUEST_CAPACITY = 256,
  KVAF_AIV_BATCH_GET_ADDRESS_TABLE_ENTRY_COUNT = 128,
  KVAF_AIV_BATCH_GET_ADDRESS_TABLE_ENTRY_BYTES =
      KVAF_AIV_BATCH_GET_ADDRESS_TABLE_ENTRY_COUNT *
      KVAF_AIV_BATCH_GET_WQE_DATA_BYTES,
  // A graph-stable NPU control tensor for a fused ExactScan + Get operator.
  // It contains exactly 16 uint64 words so Python can expose it as
  // torch.int64[16] without an ABI-specific tensor dtype.
  KVAF_FUSED_BATCH_GET_CONTROL_WORDS = 16,
  KVAF_FUSED_BATCH_GET_CONTROL_BYTES =
      KVAF_FUSED_BATCH_GET_CONTROL_WORDS * sizeof(uint64_t),
  // A separate, graph-stable control ABI for the streaming ExactScan +
  // SpeculativeGet path.  It intentionally remains a plain uint64[16]
  // tensor so a captured LI kernel can dereference it directly from HBM.
  KVAF_STREAMING_BATCH_GET_CONTROL_WORDS = 16,
  KVAF_STREAMING_BATCH_GET_CONTROL_BYTES =
      KVAF_STREAMING_BATCH_GET_CONTROL_WORDS * sizeof(uint64_t),
  // One int32 owner word per Direct-HCCP queue: 0 means free and a positive
  // value is the producer-owned lease tag (normally 1 + LI producer id).
  KVAF_STREAMING_BATCH_GET_QUEUE_LEASE_BYTES = sizeof(int32_t),
  // Shared local-memory contract used by the AIV kernel and launcher.
  KVAF_AIV_BATCH_GET_LOCAL_MEMORY_BYTES = 160 * 1024,
};

static const uint64_t KVAF_AIV_BATCH_GET_STATUS_MAGIC = 0x4b5641464247444fULL;
static const uint64_t KVAF_AIV_BATCH_GET_PROFILE_MAGIC =
    0x4b56414650524f46ULL;  // "KVAFPROF"
static const uint64_t KVAF_FUSED_BATCH_GET_CONTROL_MAGIC =
    0x4b56414646555345ULL;  // "KVAFFUSE"
static const uint64_t KVAF_FUSED_BATCH_GET_CONTROL_VERSION = 1ULL;
static const uint64_t KVAF_STREAMING_BATCH_GET_CONTROL_MAGIC =
    0x4b5641465354524dULL;  // "KVAFSTRM"
static const uint64_t KVAF_STREAMING_BATCH_GET_CONTROL_VERSION = 1ULL;

// Optional, per-core kernel timeline. The host enables collection by placing
// KVAF_AIV_BATCH_GET_PROFILE_MAGIC in `magic` before launch. The kernel keeps
// the magic in its output, so repeated launches can reuse the same status
// buffer. GetSystemCycle frequency is platform-specific and the caller must
// supply the clock MHz when converting cycles to time.
typedef struct KvafBatchGetCoreProfile {
  uint64_t magic;
  uint64_t kernel_start_cycle;
  uint64_t setup_done_cycle;
  uint64_t compact_done_cycle;
  uint64_t first_doorbell_cycle;
  uint64_t post_done_cycle;
  uint64_t completion_done_cycle;
  uint64_t kernel_finish_cycle;
  uint64_t active_requests;
  uint64_t total_posts;
  uint64_t jetty_completion_wait_cycles[KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE];
  uint64_t jetty_completion_polls[KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE];
  uint64_t reserved[26];
} KvafBatchGetCoreProfile;

// Keep the historical 384-byte stride even though ExecuteBatchGet uses only
// the first cache line on the normal path. This keeps the verified host/device
// allocation ABI stable while using the reserved area for opt-in profiling.
typedef struct KvafBatchGetCoreStatus {
  uint64_t magic;
  uint64_t processed_request_count;
  uint64_t result;
  uint64_t completed_batch_count;
  // With multiple Jettys these two diagnostic indices report Jetty 0.
  uint64_t sq_producer_index;
  uint64_t cq_consumer_index;
  uint64_t failed_request_index;
  uint64_t last_wqe_protocol;
  KvafBatchGetCoreProfile profile;
} KvafBatchGetCoreStatus;

// Persistent, AIV-owned software indices for one USER_CTL SQ/CQ pair. Queue
// arrays are core-major: [block_count][jettys_per_core].
typedef struct KvafBatchGetQueueState {
  uint32_t magic;
  uint32_t initialized;
  uint64_t sq_producer_index;
  uint64_t cq_consumer_index;
  uint64_t completed_batch_count;
  uint64_t last_error;
  uint64_t reserved[3];
} KvafBatchGetQueueState;

// Device view of one Direct-HCCP transport plus one acquired local HBM
// buffer.  The Host-DDR/Jetty fields are transport-scoped; each successful
// LocalBufferAcquire creates its own device-resident entity table with the
// local HBM address/token fields below filled for that caller-owned buffer.
typedef struct KvafBatchGetDirectUrmaEntity {
  uint32_t magic;
  uint32_t version;
  uint64_t remote_host_ddr_address;
  uint64_t remote_host_ddr_size;
  uint32_t remote_token_id;
  uint32_t remote_token_value;
  uint64_t local_hbm_address;
  uint64_t local_hbm_size;
  uint32_t local_token_id;
  uint32_t local_token_value;
  uint64_t sq_va;
  uint64_t sq_db_va;
  uint32_t jetty_id;
  uint32_t sq_depth;
  uint32_t tp_id;
  uint32_t wqe_size;
  uint8_t remote_eid[16];
  uint64_t cq_va;
  uint64_t cq_db_va;
  uint32_t jfc_id;
  uint32_t cq_depth;
  uint32_t cqe_size;
  uint32_t reserved0;
  uint64_t reserved[7];
} KvafBatchGetDirectUrmaEntity;

// Fixed-shape graph ABI. `output` and `channel_entity` must identify the same
// LocalBufferAcquire result. During capture/replay both addresses stay fixed;
// address/valid/index/offset/mask/output-index contents may change between
// replays.
typedef struct KvafExecuteBatchGetKernelArgs {
  uint64_t address_table;       // uint64[table_capacity], absolute Host DDR UBA
  uint64_t valid_table;         // uint8[table_capacity]
  uint64_t indices;             // int32[request_capacity]
  uint64_t offsets;             // int32[request_capacity], byte offset in entry
  uint64_t mask;                // uint8[request_capacity]
  uint64_t output_indices;      // int32[request_capacity]
  uint64_t output;              // caller-owned HBM output; must match entity
  uint64_t core_status;         // KvafBatchGetCoreStatus[block_count]
  uint64_t channel_entity;      // Entity[block_count * jettys_per_core]
  uint64_t queue_state;         // QueueState[block_count * jettys_per_core]
  uint32_t table_capacity;
  uint32_t request_capacity;
  uint32_t output_rows;
  uint32_t block_bytes;
  uint32_t block_count;
  // Zero preserves the phase-one ABI and selects one Jetty per AIV core.
  uint32_t jettys_per_core;
} KvafExecuteBatchGetKernelArgs;

// Cold-path control block for a fused, full-completion BatchGet path.
//
// The block itself resides in caller-owned HBM.  KVAF fills it only after it
// has registered `output` with Direct-HCCP, so channel_entity carries valid
// SQ/CQ/token state for that exact output range.  Descriptor pointers are not
// stored here: the fused operator supplies its per-chunk descriptor buffers
// when it constructs a KvafExecuteBatchGetKernelArgs for one lane.
//
// Field order is part of the cross-repository ABI. Keep it at 16 uint64
// words: version 1 remains readable by the existing P1/P2 device helper.
// `block_count` is the explicitly selected queue_count, which may be smaller
// than the Runtime transport's total npu_core_count. A fused operator splits
// the queue_count worker tables by the ABI strides below and invokes the
// single-worker Begin/TryRetire/Finish helper with its chosen table slice.
typedef struct KvafFusedBatchGetControl {
  uint64_t magic;
  uint64_t version;
  uint64_t channel_entity;
  uint64_t queue_state;
  uint64_t core_status;
  uint64_t output;
  uint64_t output_rows;
  uint64_t block_count;
  uint64_t jettys_per_core;
  uint64_t block_bytes;
  uint64_t reserved[6];
} KvafFusedBatchGetControl;

// Cold-path control block for a streaming ExactScan + SpeculativeGet kernel.
//
// Unlike KvafFusedBatchGetControl, this ABI has an explicit device-owned
// queue-pool lease table.  Any LI AIV producer may atomically claim one of
// queue_count queues, post one dense wave, probe it between score tiles, then
// finish and release it.  No queue is permanently assigned to a particular
// LI AIV.  `queue_leases` is an int32[queue_count] in HBM: 0 is free and a
// positive tag is the owning producer's `1 + producer_id`.
//
// Each leased queue has a fixed HBM descriptor record. The LI producer fills
// only its own record before Begin; no two producers can write the same record
// because the int32 lease owns it. The normal Direct-HCCP dense helper then
// reads the exact same descriptor pointers/indices/offsets/output-indices it
// would receive from a standalone BatchGet launch.
typedef struct KvafStreamingBatchGetDescriptorScratch {
  uint64_t address_table;
  uint64_t valid_table;
  uint64_t indices;
  uint64_t offsets;
  uint64_t mask;
  uint64_t output_indices;
  uint32_t table_capacity;
  uint32_t request_capacity;
  uint32_t active_request_count;
  // Absolute row base of this queue's private slot arena within the layer's
  // one registered output allocation. `output_indices` must already contain
  // absolute (not queue-relative) rows in
  // [private_output_row_base,
  //  private_output_row_base + private_output_stride_bytes / 656).
  uint32_t private_output_row_base;
} KvafStreamingBatchGetDescriptorScratch;

// `channel_entity`, `queue_state`, and `core_status` are base addresses of
// standard one-Jetty KVAF arrays, so queue q uses the fixed existing ABI
// strides sizeof(Entity), sizeof(QueueState), and sizeof(CoreStatus).
// The existing Direct-HCCP ABI requires `args.output` to equal the base of
// its LocalBufferAcquire registration. Therefore `output` always remains the
// base of the complete layer allocation and `output_rows` is the complete
// layer row count. `private_output_stride_bytes` divides that one registered
// allocation into q disjoint graph-stable private-slot arenas. Queue q owns
// the absolute row interval
//   [q * private_output_stride_bytes / 656,
//    (q + 1) * private_output_stride_bytes / 656).
// Its descriptor's output_indices must be finalized as absolute rows in this
// interval before the doorbell. This prevents two leases from writing the
// same HBM row without violating the established Direct-HCCP entity contract.
//
// `output_offset_bytes` is reserved and must be zero in version 1. A future
// subrange needs its own LocalBufferAcquire registration, because the current
// Direct-HCCP entity validates the exact output base address.
//
// This object is layer-specific because each layer owns one registered HBM
// output/entity table.  The lease table and Runtime queue-state table are
// shared by every pre-bound layer in one streaming lifecycle, preventing two
// concurrent captured layers from using the same SQ/CQ pair.
typedef struct KvafStreamingBatchGetControl {
  uint64_t magic;
  uint64_t version;
  uint64_t channel_entity;
  uint64_t queue_state;
  uint64_t core_status;
  uint64_t output;
  uint64_t queue_leases;
  uint64_t descriptor_scratch;
  uint64_t queue_count;
  uint64_t output_rows;
  uint64_t private_output_stride_bytes;
  uint64_t output_offset_bytes;
  uint64_t jettys_per_core;
  uint64_t block_bytes;
  // Layer-private, graph-stable descriptor templates:
  // KvafStreamingBatchGetDescriptorScratch[queue_count].  Host configures
  // their pointer fields/capacities/private bases before capture. A leased LI
  // producer copies its queue template, fills that queue's candidate arrays,
  // sets active_request_count, and publishes the result into the shared
  // descriptor_scratch entry before ringing the Direct-HCCP doorbell.
  uint64_t descriptor_templates;
  uint64_t descriptor_template_count;
} KvafStreamingBatchGetControl;

#if defined(__cplusplus)
static_assert(sizeof(KvafBatchGetCoreStatus) == KVAF_AIV_BATCH_GET_STATUS_BYTES,
              "KvafBatchGetCoreStatus ABI changed");
static_assert(sizeof(KvafBatchGetCoreProfile) == 320,
              "KvafBatchGetCoreProfile ABI changed");
static_assert(sizeof(KvafBatchGetQueueState) == KVAF_AIV_BATCH_GET_QUEUE_STATE_BYTES,
              "KvafBatchGetQueueState ABI changed");
static_assert(sizeof(KvafBatchGetDirectUrmaEntity) == 192,
              "KvafBatchGetDirectUrmaEntity ABI changed");
static_assert(sizeof(KvafExecuteBatchGetKernelArgs) == 104,
              "KvafExecuteBatchGetKernelArgs ABI changed");
static_assert(sizeof(KvafFusedBatchGetControl) ==
                  KVAF_FUSED_BATCH_GET_CONTROL_BYTES,
              "KvafFusedBatchGetControl ABI changed");
static_assert(sizeof(KvafStreamingBatchGetControl) ==
                  KVAF_STREAMING_BATCH_GET_CONTROL_BYTES,
              "KvafStreamingBatchGetControl ABI changed");
static_assert(sizeof(KvafStreamingBatchGetDescriptorScratch) == 64,
              "KvafStreamingBatchGetDescriptorScratch ABI changed");
#endif

#endif  // KVAF_AIV_BATCH_GET_ABI_H_
