/**
 * Request-level MTP variant of fused LIDU and compact BatchGet prepare.
 *
 * A 1/2/4/8-AIV lane group owns one request containing 2--4 packed TopK
 * rows.  The request is processed transactionally in this order:
 *
 *   supplied miss-slot membership -> per-query classification -> per-query
 *   miss-first compaction -> unordered request-level miss deduplication -> map
 *   update -> miss-prefix refill -> compact descriptor generation.
 *
 * Miss-slot membership is supplied as a bitmap input. Classification and
 * compacted tokens occupy different GM workspace slices.
 * The unique-source, resolved-slot, and descriptor slices reserve 8192
 * entries per request. Requests with more than 2048 unique misses protect
 * every input miss slot and hit slot, then select additional eviction slots
 * from the remaining cache domain before any map is changed.
 */

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_functions.h"
#include "simt_api/device_sync_functions.h"
#include "a5_lidu_fused_batch_get_prepare_mtp_tiling.h"
#include "a5_lidu_fused_batch_get_prepare_mtp_vector.h"

namespace {
using namespace AscendC;

constexpr uint32_t SPARSE_COUNT = 2048;
constexpr uint32_t MIN_QUERIES_PER_REQUEST = 1;
constexpr uint32_t MAX_QUERIES_PER_REQUEST = 4;
constexpr uint32_t MTP_MISS_CAPACITY =
    SPARSE_COUNT * MAX_QUERIES_PER_REQUEST;
constexpr uint32_t MAX_SOURCE_CAPACITY = (1U << 18) + 32U;
constexpr uint32_t MAX_CACHE_TOKENS = (1U << 17) + 20U * 128U;
constexpr uint32_t MAX_CORES_PER_REQUEST = 8;
constexpr uint32_t SIMT_THREADS = 512;
constexpr uint32_t DESCRIPTOR_SIMT_THREADS = 512;
constexpr uint32_t BITMAP_WORD_BITS = 32;
constexpr uint32_t VECTOR_LOOPS =
    SPARSE_COUNT / A5LiduFusedBatchGetPrepareMtpVector::VECTOR_WIDTH;
constexpr uint32_t UB_BLOCK_BYTES = 32;
constexpr uint32_t UB_BLOCK_WORDS = UB_BLOCK_BYTES / sizeof(uint32_t);
constexpr uint32_t MISS_BITMAP_WORDS =
    (MAX_CACHE_TOKENS + BITMAP_WORD_BITS - 1) / BITMAP_WORD_BITS;
constexpr uint32_t MISS_BITMAP_STORAGE_WORDS =
    (MISS_BITMAP_WORDS + UB_BLOCK_WORDS - 1) /
    UB_BLOCK_WORDS * UB_BLOCK_WORDS;
constexpr uint32_t SOURCE_BITMAP_WORDS = MAX_SOURCE_CAPACITY / 32;
constexpr uint32_t SLOT_SELECT_SCAN_PREFIX_OFFSET =
    (MISS_BITMAP_WORDS + 1 + UB_BLOCK_WORDS - 1) /
    UB_BLOCK_WORDS * UB_BLOCK_WORDS;
constexpr uint32_t MAX_SLOT_SELECT_SCAN_WORDS = MISS_BITMAP_WORDS + 1;
constexpr uint32_t SLOT_SELECT_SCRATCH_WORDS =
    SLOT_SELECT_SCAN_PREFIX_OFFSET + MAX_SLOT_SELECT_SCAN_WORDS;
constexpr uint32_t SOURCE_SCRATCH_STORAGE_WORDS =
    ((SLOT_SELECT_SCRATCH_WORDS > SOURCE_BITMAP_WORDS
          ? SLOT_SELECT_SCRATCH_WORDS : SOURCE_BITMAP_WORDS) + UB_BLOCK_WORDS - 1) /
    UB_BLOCK_WORDS * UB_BLOCK_WORDS;
constexpr uint32_t GLOBAL_CONTROL_WORDS = 16;
constexpr uint32_t REQUEST_CONTROL_WORDS = 16;
constexpr uint32_t DYNAMIC_UB_BYTES =
    MISS_BITMAP_STORAGE_WORDS * sizeof(uint32_t) +
    4U * SPARSE_COUNT * sizeof(int32_t) +
    REQUEST_CONTROL_WORDS * sizeof(int32_t) +
    SOURCE_SCRATCH_STORAGE_WORDS * sizeof(uint32_t) +
    MTP_MISS_CAPACITY * sizeof(int32_t);
constexpr uint32_t GM_CACHE_LINE_WORDS = 64 / sizeof(int32_t);
constexpr uint32_t CLASSIFICATION_OFFSET = 0;
constexpr uint32_t COMPACTED_TOKEN_OFFSET =
    CLASSIFICATION_OFFSET + MTP_MISS_CAPACITY;
constexpr uint32_t UNIQUE_SOURCE_OFFSET =
    COMPACTED_TOKEN_OFFSET + MTP_MISS_CAPACITY;
constexpr uint32_t RESOLVED_SLOT_OFFSET =
    UNIQUE_SOURCE_OFFSET + MTP_MISS_CAPACITY;
constexpr uint32_t ROW_CONTROL_OFFSET =
    RESOLVED_SLOT_OFFSET + MTP_MISS_CAPACITY;
constexpr uint32_t WORKSPACE_WORDS_PER_REQUEST =
    (ROW_CONTROL_OFFSET + REQUEST_CONTROL_WORDS +
     GM_CACHE_LINE_WORDS - 1) /
    GM_CACHE_LINE_WORDS * GM_CACHE_LINE_WORDS;

constexpr uint32_t CONTROL_STATUS = 0;
constexpr uint32_t CONTROL_UNIQUE_MISS_COUNT = 1;
constexpr uint32_t CONTROL_QUERY_BEGIN = 2;
constexpr uint32_t CONTROL_QUERY_COUNT = 3;
constexpr uint32_t CONTROL_QUERY_MISS_BEGIN = 4;
constexpr uint32_t CONTROL_RAW_MISS_COUNT = 8;
constexpr uint32_t CONTROL_USES_RESOLVED_SLOTS = 9;
constexpr uint32_t CONTROL_PROPOSED_EVICTION_CURSOR = 10;
constexpr uint32_t CONTROL_EXTRA_SLOT_COUNT = 11;
constexpr uint32_t SLOT_SELECT_STATUS = 12;
constexpr uint32_t DEDUP_UNIQUE_COUNT = 8;
constexpr uint32_t DEDUP_STATUS = 9;

constexpr uint32_t BLOCK_SHIFT = 7;
constexpr uint32_t BLOCK_SIZE = 1U << BLOCK_SHIFT;
constexpr uint32_t BLOCK_MASK = BLOCK_SIZE - 1U;
constexpr uint32_t PACKED_ROW_BYTES = 656U;
constexpr uint32_t MAP_SHARD_WORDS = 2048;
constexpr uint32_t PROFILE_SEQUENTIAL_START =
    A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE_FIELDS - 1U;
constexpr uint32_t SLOT_SELECT_SCALAR_EXTRA_LIMIT = 32;

constexpr int32_t STATUS_OK = 0;
constexpr int32_t ERROR_QUERY_LAYOUT = -100;
constexpr int32_t ERROR_ROW_METADATA = -101;
constexpr int32_t ERROR_DUPLICATE_POOL_ROW = -102;
constexpr int32_t ERROR_TOPK_TOKEN = -103;
constexpr int32_t ERROR_COUNT_ACCOUNTING = -105;
constexpr int32_t ERROR_UNIQUE_MISS_CAPACITY = -106;
constexpr int32_t ERROR_INSUFFICIENT_EVICTION_SLOTS = -107;
constexpr int32_t ERROR_RESOLVED_SLOT = -109;
constexpr int32_t ERROR_DESTINATION_SLOT = -110;
constexpr int32_t ERROR_RESIDENT_BLOCK = -111;
constexpr int32_t VALIDATION_DESTINATION_SLOT = 1;
constexpr int32_t VALIDATION_RESIDENT_BLOCK = 2;
constexpr int32_t PROFILE_MODE_ENABLED = 1;
constexpr int32_t PROFILE_MODE_GRAPH_DUMMY = 2;

enum ProfileField : uint32_t {
    PROFILE_TOTAL = 0,
    PROFILE_MEMBERSHIP,
    PROFILE_CLASSIFY,
    PROFILE_COMPACT,
    PROFILE_DEDUP,
    PROFILE_CLEAR,
    PROFILE_INSTALL,
    PROFILE_REFILL,
    PROFILE_DESCRIPTOR,
    PROFILE_STATUS,
    PROFILE_UNIQUE_MISS_COUNT,
    PROFILE_RAW_MISS_COUNT,
    PROFILE_QUERY_BEGIN,
    PROFILE_QUERY_COUNT,
    PROFILE_QUERY0_MISS_COUNT,
    PROFILE_QUERY1_MISS_COUNT,
    PROFILE_QUERY2_MISS_COUNT,
    PROFILE_QUERY3_MISS_COUNT,
    PROFILE_SLOT_SELECT,
};

struct RequestTiming {
    uint64_t membership;
    uint64_t classify;
    uint64_t compact;
    uint64_t dedup;
    uint64_t slotSelect;
    uint64_t clear;
    uint64_t install;
    uint64_t refill;
    uint64_t descriptor;
};

static_assert(SOURCE_BITMAP_WORDS * 32 == MAX_SOURCE_CAPACITY,
              "source bitmap must cover the complete source domain");
static_assert(PROFILE_SEQUENTIAL_START > PROFILE_SLOT_SELECT,
              "sequential profile timestamp must use a reserved field");
static_assert(DYNAMIC_UB_BYTES % UB_BLOCK_BYTES == 0,
              "dynamic UB allocation must be data-block aligned");
static_assert(CLASSIFICATION_OFFSET % GM_CACHE_LINE_WORDS == 0,
              "classification rows must be cache-line aligned");
static_assert(COMPACTED_TOKEN_OFFSET % GM_CACHE_LINE_WORDS == 0,
              "compacted rows must be cache-line aligned");
static_assert(UNIQUE_SOURCE_OFFSET % GM_CACHE_LINE_WORDS == 0,
              "unique sources must be cache-line aligned");
static_assert(RESOLVED_SLOT_OFFSET % GM_CACHE_LINE_WORDS == 0,
              "resolved slots must be cache-line aligned");
static_assert(ROW_CONTROL_OFFSET % GM_CACHE_LINE_WORDS == 0,
              "request control must be cache-line aligned");
static_assert(WORKSPACE_WORDS_PER_REQUEST % GM_CACHE_LINE_WORDS == 0,
              "request workspaces must not share cache lines");
static_assert(
    SLOT_SELECT_SCAN_PREFIX_OFFSET + MAX_SLOT_SELECT_SCAN_WORDS <=
        SOURCE_SCRATCH_STORAGE_WORDS,
    "slot selection scratch must fit in the shared source buffer");
static_assert(SLOT_SELECT_SCAN_PREFIX_OFFSET >= MAX_SLOT_SELECT_SCAN_WORDS,
              "wrapped bitmap word counts must not overlap their prefixes");

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREADS) inline void
ClassifyTopkSimt(
    __ubuf__ int32_t *topk,
    __gm__ int32_t *tokenToSlot,
    __ubuf__ uint32_t *missSlotMembership,
    __gm__ int32_t *classification,
    int32_t candidateLen,
    int32_t residentBudget)
{
    for (uint32_t index = Simt::GetThreadIdx(); index < SPARSE_COUNT;
         index += Simt::GetThreadNum()) {
        const int32_t token = topk[index];
        int32_t slot = -1;
        if (token >= 0 && token < candidateLen) {
            const int32_t mapped = tokenToSlot[token];
            if (mapped >= 0 && mapped < residentBudget) {
                const uint32_t mappedSlot = static_cast<uint32_t>(mapped);
                const uint32_t word =
                    missSlotMembership[mappedSlot >> 5];
                if ((word & (1U << (mappedSlot & 31U))) == 0) {
                    slot = mapped;
                }
            }
        }
        classification[index] = slot;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREADS) inline void
DeduplicateOneQuerySimt(
    __ubuf__ const int32_t *queryTokens,
    __ubuf__ uint32_t *sourceMembership,
    __ubuf__ int32_t *uniqueSources,
    __ubuf__ int32_t *dedupControl,
    uint32_t missCount,
    int32_t candidateLen)
{
    const uint32_t threadIndex = Simt::GetThreadIdx();
    const uint32_t threadCount = Simt::GetThreadNum();
    for (uint32_t queryIndex = threadIndex; queryIndex < missCount;
         queryIndex += threadCount) {
        const int32_t tokenValue = queryTokens[queryIndex];
        if (tokenValue < 0 || tokenValue >= candidateLen) {
            asc_atomic_or(
                dedupControl + (DEDUP_STATUS - DEDUP_UNIQUE_COUNT),
                ERROR_TOPK_TOKEN);
            continue;
        }
        const uint32_t token = static_cast<uint32_t>(tokenValue);
        const uint32_t wordIndex = token >> 5;
        const uint32_t mask = 1U << (token & 31U);
        const uint32_t current = sourceMembership[wordIndex];
        if ((current & mask) != 0U) {
            continue;
        }
        const int32_t previous = asc_atomic_or(
            (__ubuf__ int32_t *)(sourceMembership + wordIndex),
            static_cast<int32_t>(mask));
        if ((static_cast<uint32_t>(previous) & mask) == 0U) {
            const int32_t ordinal = asc_atomic_add(dedupControl, 1);
            if (ordinal < static_cast<int32_t>(MTP_MISS_CAPACITY)) {
                uniqueSources[ordinal] = tokenValue;
            }
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREADS) inline void
ValidatePreparedDestinationsSimt(
    __ubuf__ const int32_t *slots,
    __ubuf__ const int32_t *physicalBlocks,
    __ubuf__ int32_t *validationFlags,
    uint32_t uniqueCount,
    uint32_t budget,
    uint32_t residentBlockCapacity)
{
    for (uint32_t index = Simt::GetThreadIdx(); index < uniqueCount;
         index += Simt::GetThreadNum()) {
        const int32_t slotValue = slots[index];
        if (slotValue < 0 || static_cast<uint32_t>(slotValue) >= budget) {
            asc_atomic_or(validationFlags, VALIDATION_DESTINATION_SLOT);
            continue;
        }
        const uint32_t block =
            static_cast<uint32_t>(slotValue) >> BLOCK_SHIFT;
        if (block >= residentBlockCapacity) {
            asc_atomic_or(validationFlags, VALIDATION_DESTINATION_SLOT);
            continue;
        }
        const int32_t physicalBlock = physicalBlocks[block];
        if (physicalBlock < 0 ||
            static_cast<uint32_t>(physicalBlock) >
                0x7FFFFFFFU / BLOCK_SIZE) {
            asc_atomic_or(validationFlags, VALIDATION_RESIDENT_BLOCK);
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREADS) inline void
MergeHitSlotsSimt(
    __gm__ int32_t *destinationSlots,
    __ubuf__ uint32_t *protectedSlots,
    __ubuf__ int32_t *status,
    uint32_t queryBase0,
    uint32_t hitCount0,
    uint32_t queryBase1,
    uint32_t hitCount1,
    uint32_t queryBase2,
    uint32_t hitCount2,
    uint32_t queryBase3,
    uint32_t hitCount3,
    uint32_t budget)
{
    const uint32_t threadIndex = Simt::GetThreadIdx();
    const uint32_t threadCount = Simt::GetThreadNum();
    const uint32_t totalHitCount =
        hitCount0 + hitCount1 + hitCount2 + hitCount3;
    for (uint32_t flatIndex = threadIndex; flatIndex < totalHitCount;
         flatIndex += threadCount) {
        uint32_t queryBase = queryBase0;
        uint32_t queryIndex = flatIndex;
        if (queryIndex >= hitCount0) {
            queryIndex -= hitCount0;
            queryBase = queryBase1;
            if (queryIndex >= hitCount1) {
                queryIndex -= hitCount1;
                queryBase = queryBase2;
                if (queryIndex >= hitCount2) {
                    queryIndex -= hitCount2;
                    queryBase = queryBase3;
                }
            }
        }

        const int32_t slotValue =
            asc_ldcg(destinationSlots + queryBase + queryIndex);
        if (slotValue < 0 ||
            slotValue >= static_cast<int32_t>(budget)) {
            asc_atomic_or(status, ERROR_RESOLVED_SLOT);
            continue;
        }
        const uint32_t slot = static_cast<uint32_t>(slotValue);
        const uint32_t wordIndex = slot >> 5;
        const int32_t mask = static_cast<int32_t>(1U << (slot & 31U));
        asc_atomic_or(
            (__ubuf__ int32_t *)(protectedSlots + wordIndex), mask);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREADS) inline void
CountAvailableBitmapWordsSimt(
    __ubuf__ const uint32_t *protectedSlots,
    __ubuf__ int32_t *wordCounts,
    uint32_t budget,
    uint32_t scanBegin,
    uint32_t firstWordCount,
    uint32_t secondWordCount)
{
    const uint32_t threadIndex = Simt::GetThreadIdx();
    const uint32_t threadCount = Simt::GetThreadNum();
    const uint32_t orderedWordCount = firstWordCount + secondWordCount;
    const uint32_t firstWord = scanBegin / BITMAP_WORD_BITS;
    for (uint32_t orderedWord = threadIndex;
         orderedWord < orderedWordCount;
         orderedWord += threadCount) {
        uint32_t wordIndex = 0;
        uint32_t segmentBegin = 0;
        uint32_t segmentEnd = scanBegin;
        if (orderedWord < firstWordCount) {
            wordIndex = firstWord + orderedWord;
            segmentBegin = scanBegin;
            segmentEnd = budget;
        } else {
            wordIndex = orderedWord - firstWordCount;
        }

        const uint32_t wordBase = wordIndex * BITMAP_WORD_BITS;
        uint32_t validMask = 0xFFFFFFFFU;
        if (segmentBegin > wordBase) {
            validMask &= 0xFFFFFFFFU << (segmentBegin - wordBase);
        }
        if (segmentEnd < wordBase + BITMAP_WORD_BITS) {
            const uint32_t validBits = segmentEnd - wordBase;
            validMask &= (1U << validBits) - 1U;
        }

        uint32_t available =
            ~protectedSlots[wordIndex] & validMask;
        uint32_t count = 0;
        while (available != 0U) {
            available &= available - 1U;
            ++count;
        }
        wordCounts[orderedWord] = static_cast<int32_t>(count);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREADS) inline void
WriteAvailableBitmapSlotsSimt(
    __ubuf__ const uint32_t *protectedSlots,
    __ubuf__ const int32_t *wordCounts,
    __ubuf__ const int32_t *wordOffsets,
    __ubuf__ int32_t *resolvedSlots,
    uint32_t budget,
    uint32_t scanBegin,
    uint32_t firstWordCount,
    uint32_t secondWordCount,
    uint32_t extraCount)
{
    const uint32_t threadIndex = Simt::GetThreadIdx();
    const uint32_t threadCount = Simt::GetThreadNum();
    const uint32_t orderedWordCount = firstWordCount + secondWordCount;
    const uint32_t firstWord = scanBegin / BITMAP_WORD_BITS;
    for (uint32_t orderedWord = threadIndex;
         orderedWord < orderedWordCount;
         orderedWord += threadCount) {
        const uint32_t prefix = static_cast<uint32_t>(
            wordOffsets[orderedWord]);
        const uint32_t count = static_cast<uint32_t>(
            wordCounts[orderedWord]);
        if (prefix >= extraCount || count == 0) {
            continue;
        }

        uint32_t wordIndex = 0;
        uint32_t segmentBegin = 0;
        uint32_t segmentEnd = scanBegin;
        if (orderedWord < firstWordCount) {
            wordIndex = firstWord + orderedWord;
            segmentBegin = scanBegin;
            segmentEnd = budget;
        } else {
            wordIndex = orderedWord - firstWordCount;
        }

        const uint32_t wordBase = wordIndex * BITMAP_WORD_BITS;
        uint32_t validMask = 0xFFFFFFFFU;
        if (segmentBegin > wordBase) {
            validMask &= 0xFFFFFFFFU << (segmentBegin - wordBase);
        }
        if (segmentEnd < wordBase + BITMAP_WORD_BITS) {
            const uint32_t validBits = segmentEnd - wordBase;
            validMask &= (1U << validBits) - 1U;
        }

        uint32_t available =
            ~protectedSlots[wordIndex] & validMask;
        uint32_t outputIndex = prefix;
        while (available != 0U && outputIndex < extraCount) {
            // Keep the original bitmap while locating the first set bit.
            // Clearing a bit from a progressively shifted value would leave
            // bits at positions >= 16 uncleared and repeat the same slot.
            uint32_t probe = available;
            uint32_t bit = 0;
            if ((probe & 0x0000FFFFU) == 0) {
                probe >>= 16;
                bit += 16;
            }
            if ((probe & 0x000000FFU) == 0) {
                probe >>= 8;
                bit += 8;
            }
            if ((probe & 0x0000000FU) == 0) {
                probe >>= 4;
                bit += 4;
            }
            if ((probe & 0x00000003U) == 0) {
                probe >>= 2;
                bit += 2;
            }
            if ((probe & 0x00000001U) == 0) {
                bit += 1;
            }
            resolvedSlots[SPARSE_COUNT + outputIndex] =
                static_cast<int32_t>(wordBase + bit);
            ++outputIndex;
            available &= ~(1U << bit);
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREADS) inline void
ClearVictimsSimt(
    __gm__ int32_t *inputMissSlots,
    __gm__ int32_t *resolvedMissSlots,
    __gm__ int32_t *tokenToSlot,
    __gm__ int32_t *slotToToken,
    uint32_t missCount,
    bool useResolvedSlots,
    uint32_t residentCapacity,
    uint32_t forwardBegin,
    uint32_t forwardEnd)
{
    const uint32_t threadCount = Simt::GetThreadNum();
    for (uint32_t index = Simt::GetThreadIdx(); index < missCount;
         index += threadCount) {
        const int32_t slotValue = useResolvedSlots
            ? resolvedMissSlots[index]
            : inputMissSlots[index];
        if (slotValue < 0 ||
            static_cast<uint32_t>(slotValue) >= residentCapacity) {
            continue;
        }
        const uint32_t slot = static_cast<uint32_t>(slotValue);
        const int32_t victim = slotToToken[slot];
        // Every lane scans the complete miss list, but only the lane owning
        // the victim's forward-map domain may update it.  The 2048-word
        // domains are cache-line aligned, so distinct AIVs never dirty
        // different words in the same tokenToSlot cache line.
        if (victim >= static_cast<int32_t>(forwardBegin) &&
            victim < static_cast<int32_t>(forwardEnd) &&
            tokenToSlot[victim] == static_cast<int32_t>(slot)) {
            // Keep the map handoff out of the SIMT Data Cache.  The cache may
            // still contain classification's old mapping across later VFs.
            asc_stcg(tokenToSlot + victim, static_cast<int32_t>(-1));
        }
    }
    // Publish every thread's forward-map stores before the caller performs
    // the launch-wide clear-to-install barrier.
    asc_threadfence();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREADS) inline void
InstallMapsSimt(
    __gm__ int32_t *sourceIds,
    __gm__ int32_t *inputMissSlots,
    __gm__ int32_t *resolvedMissSlots,
    __gm__ int32_t *tokenToSlot,
    __gm__ int32_t *slotToToken,
    uint32_t missCount,
    bool useResolvedSlots,
    uint32_t residentCapacity,
    uint32_t forwardBegin,
    uint32_t forwardEnd,
    uint32_t reverseBegin,
    uint32_t reverseEnd)
{
    const uint32_t threadIndex = Simt::GetThreadIdx();
    const uint32_t threadCount = Simt::GetThreadNum();
    for (uint32_t index = threadIndex; index < missCount;
         index += threadCount) {
        const int32_t token = sourceIds[index];
        const int32_t slot = useResolvedSlots
            ? resolvedMissSlots[index]
            : inputMissSlots[index];
        const bool slotValid = slot >= 0 &&
            static_cast<uint32_t>(slot) < residentCapacity;
        if (slotValid && token >= static_cast<int32_t>(forwardBegin) &&
            token < static_cast<int32_t>(forwardEnd)) {
            asc_stcg(tokenToSlot + token, slot);
        }
        if (slotValid && slot >= static_cast<int32_t>(reverseBegin) &&
            slot < static_cast<int32_t>(reverseEnd)) {
            asc_stcg(slotToToken + slot, token);
        }
    }
    // SIMT accesses GM through its dedicated Data Cache.  A scalar DCCI does
    // not publish these stores; fence them in the producing SIMT threads
    // before the caller synchronizes all AIVs and starts refill.
    asc_threadfence();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREADS) inline void
RefillMissPrefixesSimt(
    __gm__ int32_t *compactedTokens,
    __gm__ int32_t *tokenToSlot,
    __gm__ int32_t *destinationSlots,
    __ubuf__ int32_t *queryMissCounts,
    uint32_t queryCount,
    uint32_t lane,
    uint32_t laneCount)
{
    const uint32_t threadIndex = Simt::GetThreadIdx();
    const uint32_t threadCount = Simt::GetThreadNum();
    for (uint32_t localQuery = lane; localQuery < queryCount;
         localQuery += laneCount) {
        const uint32_t missCount = static_cast<uint32_t>(
            queryMissCounts[localQuery]);
        const uint32_t queryBase = localQuery * SPARSE_COUNT;
        for (uint32_t index = threadIndex; index < missCount;
             index += threadCount) {
            const int32_t token = compactedTokens[queryBase + index];
            // Classification's default GM loads may remain in the SIMT Data
            // Cache across VF switches.  Read L2 directly so refill observes
            // the cache-global stores published by install.
            destinationSlots[queryBase + index] =
                asc_ldcg(tokenToSlot + token);
        }
    }
}

// Descriptor tasks are partitioned by complete 64-byte cache lines.  Each
// task maps a compact ordinal back to its request through the common prefix.
__simt_vf__ __aicore__ LAUNCH_BOUND(DESCRIPTOR_SIMT_THREADS) inline void
GenerateCompactDescriptorsSimt(
    __gm__ const int32_t *workspace,
    __gm__ const int32_t *inputMissSlots,
    __gm__ const int32_t *residentBlockTable,
    __gm__ const int32_t *reqPoolEntries,
    __ubuf__ int32_t *compactPrefixes,
    __gm__ int32_t *indices,
    __gm__ int32_t *offsets,
    __gm__ int32_t *outputIndices,
    uint32_t taskBegin,
    uint32_t taskCount,
    uint32_t batchSize,
    uint32_t residentCapacity,
    uint32_t residentBlockCapacity,
    uint32_t numLayers,
    uint32_t layerId,
    uint32_t qcap)
{
    for (uint32_t taskColumn = Simt::GetThreadIdx();
         taskColumn < taskCount; taskColumn += Simt::GetThreadNum()) {
        const uint32_t ordinal = taskBegin + taskColumn;
        uint32_t lower = 0;
        uint32_t upper = batchSize;
        while (lower + 1 < upper) {
            const uint32_t middle = (lower + upper) >> 1;
            if (ordinal <
                static_cast<uint32_t>(compactPrefixes[middle])) {
                upper = middle;
            } else {
                lower = middle;
            }
        }
        const uint32_t row = lower;
        const uint32_t column = ordinal - static_cast<uint32_t>(
            compactPrefixes[row]);
        const uint64_t requestBase = GLOBAL_CONTROL_WORDS +
            static_cast<uint64_t>(row) * WORKSPACE_WORDS_PER_REQUEST;
        const uint32_t sourceToken = static_cast<uint32_t>(
            workspace[requestBase + UNIQUE_SOURCE_OFFSET + column]);
        const uint32_t poolRow = static_cast<uint32_t>(
            reqPoolEntries[row]);
        const uint64_t layerBase =
            (static_cast<uint64_t>(poolRow) * numLayers + layerId) * qcap;
        indices[taskColumn] = static_cast<int32_t>(
            static_cast<uint32_t>(layerBase) +
            (sourceToken >> BLOCK_SHIFT));
        offsets[taskColumn] = static_cast<int32_t>(
            (sourceToken & BLOCK_MASK) * PACKED_ROW_BYTES);

        const bool useResolvedSlots = workspace[
            requestBase + ROW_CONTROL_OFFSET +
            CONTROL_USES_RESOLVED_SLOTS] != 0;
        uint32_t destinationSlot = 0;
        if (useResolvedSlots) {
            destinationSlot = static_cast<uint32_t>(workspace[
                requestBase + RESOLVED_SLOT_OFFSET + column]);
        } else {
            const uint64_t destinationOffset =
                static_cast<uint64_t>(row) * SPARSE_COUNT + column;
            destinationSlot = static_cast<uint32_t>(
                inputMissSlots[destinationOffset]);
        }
        if (destinationSlot >= residentCapacity) {
            outputIndices[taskColumn] = -1;
            continue;
        }
        const uint32_t destinationBlock = destinationSlot >> BLOCK_SHIFT;
        if (destinationBlock >= residentBlockCapacity) {
            outputIndices[taskColumn] = -1;
            continue;
        }
        const uint64_t blockOffset =
            static_cast<uint64_t>(row) * residentBlockCapacity +
            destinationBlock;
        const int32_t physicalBlock = residentBlockTable[blockOffset];
        if (physicalBlock < 0 ||
            static_cast<uint32_t>(physicalBlock) >
                0x7FFFFFFFU / BLOCK_SIZE) {
            outputIndices[taskColumn] = -1;
            continue;
        }
        outputIndices[taskColumn] = physicalBlock * BLOCK_SIZE +
            static_cast<int32_t>(destinationSlot & BLOCK_MASK);
    }
}

class A5LiduFusedBatchGetPrepareMtpKernel {
public:
    __aicore__ inline A5LiduFusedBatchGetPrepareMtpKernel(
        TPipe *pipe,
        const A5LiduFusedBatchGetPrepareMtpTilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR topkIndices,
        GM_ADDR actualSeqLengthsQuery,
        GM_ADDR reqPoolEntries,
        GM_ADDR tokenToSlotPool,
        GM_ADDR slotToTokenPool,
        GM_ADDR evictionCursors,
        GM_ADDR cacheTokens,
        GM_ADDR candidateLens,
        GM_ADDR missSlots,
        GM_ADDR missSlotBitmap,
        GM_ADDR residentBlockTable,
        GM_ADDR logicalSeqLengths,
        GM_ADDR profileEnabled,
        GM_ADDR destinationSlots,
        GM_ADDR missCounts,
        GM_ADDR indices,
        GM_ADDR offsets,
        GM_ADDR outputIndices,
        GM_ADDR residentSeqLengths,
        GM_ADDR profiling,
        GM_ADDR workspace)
    {
        coreIdx_ = GetBlockIdx();
        topkIndicesGm_.SetGlobalBuffer(
            (__gm__ int32_t *)topkIndices,
            static_cast<uint64_t>(tiling_->packedQueryCount) * SPARSE_COUNT);
        actualSeqLengthsQueryGm_.SetGlobalBuffer(
            (__gm__ int32_t *)actualSeqLengthsQuery, tiling_->batchSize);
        reqPoolEntriesGm_.SetGlobalBuffer(
            (__gm__ int32_t *)reqPoolEntries, tiling_->batchSize);
        tokenToSlotPoolGm_.SetGlobalBuffer(
            (__gm__ int32_t *)tokenToSlotPool,
            static_cast<uint64_t>(tiling_->poolSize) *
                tiling_->sourceCapacity);
        slotToTokenPoolGm_.SetGlobalBuffer(
            (__gm__ int32_t *)slotToTokenPool,
            static_cast<uint64_t>(tiling_->poolSize) *
                tiling_->residentCapacity);
        evictionCursorsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)evictionCursors, tiling_->poolSize);
        cacheTokensGm_.SetGlobalBuffer(
            (__gm__ int32_t *)cacheTokens, tiling_->batchSize);
        candidateLensGm_.SetGlobalBuffer(
            (__gm__ int32_t *)candidateLens, tiling_->batchSize);
        missSlotsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)missSlots,
            static_cast<uint64_t>(tiling_->batchSize) * SPARSE_COUNT);
        missSlotBitmapGm_.SetGlobalBuffer(
            (__gm__ int32_t *)missSlotBitmap,
            static_cast<uint64_t>(tiling_->batchSize) *
                MISS_BITMAP_STORAGE_WORDS);
        residentBlockTableGm_.SetGlobalBuffer(
            (__gm__ int32_t *)residentBlockTable,
            static_cast<uint64_t>(tiling_->batchSize) *
                tiling_->residentBlockCapacity);
        logicalSeqLengthsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)logicalSeqLengths, tiling_->batchSize);
        profileEnabledGm_.SetGlobalBuffer(
            (__gm__ int32_t *)profileEnabled, 1);
        const int32_t profileMode = profileEnabledGm_.GetValue(0);
        profileEnabled_ = profileMode == PROFILE_MODE_ENABLED;
        graphDummy_ = profileMode == PROFILE_MODE_GRAPH_DUMMY;
        destinationSlotsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)destinationSlots,
            static_cast<uint64_t>(tiling_->packedQueryCount) * SPARSE_COUNT);
        missCountsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)missCounts, tiling_->batchSize + 1);
        const uint64_t descriptorCapacity =
            static_cast<uint64_t>(tiling_->batchSize) * MTP_MISS_CAPACITY;
        indicesGm_.SetGlobalBuffer((__gm__ int32_t *)indices,
                                   descriptorCapacity);
        offsetsGm_.SetGlobalBuffer((__gm__ int32_t *)offsets,
                                   descriptorCapacity);
        outputIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)outputIndices,
                                         descriptorCapacity);
        residentSeqLengthsGm_.SetGlobalBuffer(
            (__gm__ int32_t *)residentSeqLengths, tiling_->batchSize);
        profilingGm_.SetGlobalBuffer(
            (__gm__ int64_t *)profiling,
            static_cast<uint64_t>(tiling_->batchSize) *
                A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE_LANES *
                A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE_FIELDS);
        workspaceGm_.SetGlobalBuffer(
            (__gm__ int32_t *)GetUserWorkspace(workspace),
            GLOBAL_CONTROL_WORDS +
                static_cast<uint64_t>(tiling_->batchSize) *
                    WORKSPACE_WORDS_PER_REQUEST);

        // Host tiling publishes the complete TPipe allocation through
        // SetDynUBufSize, so SIMT can address the full TopK row in dynamic UB.
        // Keep VEC buffers before the scalar/MTE-only dedup buffers.
        pipe_->InitBuffer(
            missMembershipBuf_,
            MISS_BITMAP_STORAGE_WORDS * sizeof(uint32_t));
        pipe_->InitBuffer(topkBuf_, SPARSE_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(classificationBuf_, SPARSE_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(outputTokensBuf_, SPARSE_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(outputSlotsBuf_, SPARSE_COUNT * sizeof(int32_t));
        pipe_->InitBuffer(controlBuf_, REQUEST_CONTROL_WORDS * sizeof(int32_t));
        pipe_->InitBuffer(
            sourceMembershipBuf_, SOURCE_SCRATCH_STORAGE_WORDS * sizeof(uint32_t));
        pipe_->InitBuffer(
            uniqueSourcesBuf_, MTP_MISS_CAPACITY * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        if (tiling_->batchSize > tiling_->usedCoreNum) {
            ProcessSequentialBatch();
            return;
        }
        ProcessParallelBatch();
    }

    __aicore__ inline void ProcessParallelBatch()
    {
        const uint32_t row = coreIdx_ / tiling_->coresPerRequest;
        const uint32_t lane = coreIdx_ % tiling_->coresPerRequest;
        const uint64_t totalStart = ProfileStart();
        RequestTiming timing;
        if (profileEnabled_) {
            timing = RequestTiming{};
        }

        PublishQueryLayoutStatus();
        GlobalSync();
        const int32_t layoutStatus = workspaceGm_.GetValue(0);
        if (layoutStatus != STATUS_OK) {
            PublishAllInvalidQueryRows();
            GlobalSync();
            if (coreIdx_ == 0) {
                WriteBatchOutputs(true);
            }
            return;
        }

        const uint32_t queryBegin = row == 0
            ? 0U
            : static_cast<uint32_t>(
                  actualSeqLengthsQueryGm_.GetValue(row - 1));
        const uint32_t queryEnd = static_cast<uint32_t>(
            actualSeqLengthsQueryGm_.GetValue(row));
        const uint32_t queryCount = queryEnd - queryBegin;
        int32_t budget = 0;
        int32_t candidateLen = 0;
        int32_t poolRow = 0;
        const bool validRow = IsValidRow(
            row, budget, candidateLen, poolRow);
        const bool paddingRow = IsMtpGraphPaddingRow(
            row, budget, candidateLen, poolRow);
        int32_t status = (validRow || paddingRow)
            ? STATUS_OK
            : ERROR_ROW_METADATA;
        const uint64_t membershipStart = ProfileStart();
        if (status == STATUS_OK && !paddingRow) {
            LoadMissSlotMembership(row);
        }
        ProfileSet(timing.membership, membershipStart);

        const uint64_t classifyStart = ProfileStart();
        if (status == STATUS_OK && !paddingRow) {
            for (uint32_t localQuery = lane; localQuery < queryCount;
                 localQuery += tiling_->coresPerRequest) {
                ClassifyQuery(
                    row, localQuery, queryBegin + localQuery,
                    candidateLen, budget, poolRow);
            }
        }
        ProfileSet(timing.classify, classifyStart);
        GlobalSync();

        if (lane == 0) {
            if (paddingRow) {
                PrepareMtpGraphPaddingRow(
                    row, queryBegin, queryCount);
            } else {
                PrepareRequestAfterClassification(
                    row, status, queryBegin, queryCount,
                    candidateLen, static_cast<uint32_t>(budget), poolRow,
                    timing);
            }
        }
        GlobalSync();

        ClearPreparedRequest(row, lane, timing);
        // Preserve the base operator's clear-before-install rule globally,
        // including when two request entries unexpectedly alias one pool row.
        GlobalSync();
        InstallPreparedRequest(row, lane, timing);
        GlobalSync();
        RefillPreparedRequest(row, lane, timing);
        if (coreIdx_ == 0) {
            WriteBatchOutputs(false);
        }

        const uint64_t descriptorStart = ProfileStart();
        GenerateCompactDescriptorTask();
        ProfileSet(timing.descriptor, descriptorStart);
        if (profileEnabled_) {
            WriteProfile(row, lane, ReadCycle() - totalStart, timing);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessSequentialBatch()
    {
        // Finish preparation for every request before changing any map. This
        // preserves the batch transaction boundary across sequential waves.
        PublishQueryLayoutStatus();
        GlobalSync();
        if (workspaceGm_.GetValue(0) != STATUS_OK) {
            PublishAllInvalidQueryRows();
            GlobalSync();
            if (coreIdx_ == 0) {
                WriteBatchOutputs(true);
            }
            return;
        }

        for (uint32_t waveBegin = 0;
             waveBegin < tiling_->batchSize;
             waveBegin += tiling_->usedCoreNum) {
            const uint32_t row = waveBegin + coreIdx_;
            if (row >= tiling_->batchSize) {
                continue;
            }

            RequestTiming timing;
            const uint64_t totalStart = ReadCycle();
            if (profileEnabled_) {
                InitializeSequentialProfile(row, totalStart);
            }
            PrepareSequentialRow(row, timing);
            if (profileEnabled_) {
                StoreSequentialPreparationTiming(row, timing);
            }
        }
        GlobalSync();

        for (uint32_t waveBegin = 0;
             waveBegin < tiling_->batchSize;
             waveBegin += tiling_->usedCoreNum) {
            const uint32_t row = waveBegin + coreIdx_;
            if (row >= tiling_->batchSize) {
                continue;
            }
            RequestTiming timing;
            ClearPreparedRequest(row, 0, timing);
            if (profileEnabled_) {
                StoreSequentialStageTiming(row, PROFILE_CLEAR, timing.clear);
            }
        }
        GlobalSync();

        for (uint32_t waveBegin = 0;
             waveBegin < tiling_->batchSize;
             waveBegin += tiling_->usedCoreNum) {
            const uint32_t row = waveBegin + coreIdx_;
            if (row >= tiling_->batchSize) {
                continue;
            }
            RequestTiming timing;
            InstallPreparedRequest(row, 0, timing);
            if (profileEnabled_) {
                StoreSequentialStageTiming(
                    row, PROFILE_INSTALL, timing.install);
            }
        }
        GlobalSync();

        for (uint32_t waveBegin = 0;
             waveBegin < tiling_->batchSize;
             waveBegin += tiling_->usedCoreNum) {
            const uint32_t row = waveBegin + coreIdx_;
            if (row >= tiling_->batchSize) {
                continue;
            }
            RequestTiming timing;
            RefillPreparedRequest(row, 0, timing);
            if (profileEnabled_) {
                StoreSequentialStageTiming(row, PROFILE_REFILL, timing.refill);
            }
        }
        GlobalSync();

        if (coreIdx_ == 0) {
            WriteBatchOutputs(false);
        }
        GlobalSync();

        const uint64_t descriptorStart = ReadCycle();
        GenerateCompactDescriptorTask();
        const uint64_t descriptor = ReadCycle() - descriptorStart;
        if (profileEnabled_) {
            for (uint32_t waveBegin = 0;
                 waveBegin < tiling_->batchSize;
                 waveBegin += tiling_->usedCoreNum) {
                const uint32_t row = waveBegin + coreIdx_;
                if (row < tiling_->batchSize) {
                    FinalizeSequentialProfile(row, descriptor);
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline uint64_t ReadCycle() const
    {
        return static_cast<uint64_t>(AscendC::GetSystemCycle());
    }

    __aicore__ inline uint64_t ProfileStart() const
    {
        return profileEnabled_ ? ReadCycle() : 0;
    }

    __aicore__ inline void ProfileSet(
        uint64_t &field, uint64_t start) const
    {
        if (profileEnabled_) {
            field = ReadCycle() - start;
        }
    }

    __aicore__ inline void ProfileAdd(
        uint64_t &field, uint64_t start) const
    {
        if (profileEnabled_) {
            field += ReadCycle() - start;
        }
    }

    __aicore__ inline void GlobalSync()
    {
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PrepareSequentialRow(
        uint32_t row, RequestTiming &timing)
    {
        const uint32_t queryBegin = row == 0
            ? 0U
            : static_cast<uint32_t>(
                  actualSeqLengthsQueryGm_.GetValue(row - 1));
        const uint32_t queryEnd = static_cast<uint32_t>(
            actualSeqLengthsQueryGm_.GetValue(row));
        const uint32_t queryCount = queryEnd - queryBegin;
        int32_t budget = 0;
        int32_t candidateLen = 0;
        int32_t poolRow = 0;
        const bool validRow = IsValidRow(
            row, budget, candidateLen, poolRow);
        const bool paddingRow = IsMtpGraphPaddingRow(
            row, budget, candidateLen, poolRow);
        int32_t status = (validRow || paddingRow)
            ? STATUS_OK
            : ERROR_ROW_METADATA;

        const uint64_t membershipStart = ReadCycle();
        if (status == STATUS_OK && !paddingRow) {
            LoadMissSlotMembership(row);
        }
        timing.membership = ReadCycle() - membershipStart;

        const uint64_t classifyStart = ReadCycle();
        if (status == STATUS_OK && !paddingRow) {
            for (uint32_t localQuery = 0;
                 localQuery < queryCount;
                 ++localQuery) {
                ClassifyQuery(
                    row, localQuery, queryBegin + localQuery,
                    candidateLen, budget, poolRow);
            }
        }
        timing.classify = ReadCycle() - classifyStart;
        if (paddingRow) {
            PrepareMtpGraphPaddingRow(row, queryBegin, queryCount);
        } else {
            PrepareRequestAfterClassification(
                row, status, queryBegin, queryCount,
                candidateLen, static_cast<uint32_t>(budget), poolRow,
                timing);
        }
    }

    __aicore__ inline uint64_t RequestBase(uint32_t row) const
    {
        return GLOBAL_CONTROL_WORDS +
            static_cast<uint64_t>(row) * WORKSPACE_WORDS_PER_REQUEST;
    }

    __aicore__ inline uint64_t ClassificationBase(
        uint32_t row, uint32_t localQuery) const
    {
        return RequestBase(row) + CLASSIFICATION_OFFSET +
            static_cast<uint64_t>(localQuery) * SPARSE_COUNT;
    }

    __aicore__ inline uint64_t CompactedTokenBase(
        uint32_t row, uint32_t localQuery) const
    {
        return RequestBase(row) + COMPACTED_TOKEN_OFFSET +
            static_cast<uint64_t>(localQuery) * SPARSE_COUNT;
    }

    __aicore__ inline uint64_t UniqueSourceBase(uint32_t row) const
    {
        return RequestBase(row) + UNIQUE_SOURCE_OFFSET;
    }

    __aicore__ inline uint64_t ResolvedSlotBase(uint32_t row) const
    {
        return RequestBase(row) + RESOLVED_SLOT_OFFSET;
    }

    __aicore__ inline uint64_t RowControlBase(uint32_t row) const
    {
        return RequestBase(row) + ROW_CONTROL_OFFSET;
    }

    __aicore__ inline uint64_t QueryBase(uint32_t queryRow) const
    {
        return static_cast<uint64_t>(queryRow) * SPARSE_COUNT;
    }

    __aicore__ inline void StoreScalarControl(
        uint64_t destinationOffset, LocalTensor<int32_t> source,
        uint32_t count)
    {
        const DataCopyExtParams copy{
            1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad<int32_t, PaddingMode::Normal>(
            workspaceGm_[destinationOffset], source, copy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void PublishQueryLayoutStatus()
    {
        if (coreIdx_ != 0) {
            return;
        }
        int32_t status = STATUS_OK;
        int32_t previous = 0;
        for (uint32_t row = 0; row < tiling_->batchSize; ++row) {
            const int32_t end = actualSeqLengthsQueryGm_.GetValue(row);
            const int32_t queryCount = end - previous;
            if (previous < 0 || end < previous ||
                end > static_cast<int32_t>(tiling_->packedQueryCount) ||
                (queryCount < static_cast<int32_t>(MIN_QUERIES_PER_REQUEST) &&
                 !IsMtpGraphPaddingRow(row)) ||
                queryCount > static_cast<int32_t>(MAX_QUERIES_PER_REQUEST)) {
                status = ERROR_QUERY_LAYOUT;
            }
            previous = end;
        }
        // Captured TopK buffers can have an unused trailing query suffix.
        if (previous > static_cast<int32_t>(tiling_->packedQueryCount)) {
            status = ERROR_QUERY_LAYOUT;
        }
        for (uint32_t row = 0; row < tiling_->batchSize; ++row) {
            const int32_t poolRow = reqPoolEntriesGm_.GetValue(row);
            const bool paddingRow = IsMtpGraphPaddingRow(row);
            for (uint32_t previousRow = 0; previousRow < row; ++previousRow) {
                if (poolRow == reqPoolEntriesGm_.GetValue(previousRow) &&
                    !(paddingRow && IsMtpGraphPaddingRow(previousRow))) {
                    status = ERROR_DUPLICATE_POOL_ROW;
                }
            }
        }
        LocalTensor<int32_t> control = controlBuf_.Get<int32_t>();
        Duplicate(control, static_cast<int32_t>(0), REQUEST_CONTROL_WORDS);
        PipeBarrier<PIPE_V>();
        control.SetValue(0, status);
        StoreScalarControl(0, control, GLOBAL_CONTROL_WORDS);
    }

    __aicore__ inline void LoadTopk(
        uint32_t queryRow, LocalTensor<int32_t> destination)
    {
        const DataCopyExtParams copy{
            1, SPARSE_COUNT * sizeof(int32_t), 0, 0, 0};
        const DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad<int32_t, PaddingMode::Normal>(
            destination, topkIndicesGm_[QueryBase(queryRow)], copy, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    }

    __aicore__ inline void LoadWorkspaceVector(
        uint64_t sourceOffset, LocalTensor<int32_t> destination,
        uint32_t count)
    {
        const DataCopyExtParams copy{
            1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        const DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad<int32_t, PaddingMode::Normal>(
            destination, workspaceGm_[sourceOffset], copy, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    }

    __aicore__ inline void LoadWorkspaceScalar(
        uint64_t sourceOffset, LocalTensor<int32_t> destination,
        uint32_t count)
    {
        const DataCopyExtParams copy{
            1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        const DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad<int32_t, PaddingMode::Normal>(
            destination, workspaceGm_[sourceOffset], copy, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
    }

    __aicore__ inline void LoadInputMissSlots(
        uint32_t row, LocalTensor<int32_t> destination)
    {
        const DataCopyExtParams copy{
            1, SPARSE_COUNT * sizeof(int32_t), 0, 0, 0};
        const DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad<int32_t, PaddingMode::Normal>(
            destination,
            missSlotsGm_[static_cast<uint64_t>(row) * SPARSE_COUNT],
            copy, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
    }

    __aicore__ inline void LoadInputMissSlotsVector(
        uint32_t row, LocalTensor<int32_t> destination)
    {
        const DataCopyExtParams copy{
            1, SPARSE_COUNT * sizeof(int32_t), 0, 0, 0};
        const DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad<int32_t, PaddingMode::Normal>(
            destination,
            missSlotsGm_[static_cast<uint64_t>(row) * SPARSE_COUNT],
            copy, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    }

    __aicore__ inline void StoreWorkspaceVector(
        uint64_t destinationOffset, LocalTensor<int32_t> source,
        uint32_t count)
    {
        const DataCopyExtParams copy{
            1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        SetFlag<HardEvent::S_MTE3>(EVENT_ID5);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID5);
        DataCopyPad<int32_t, PaddingMode::Normal>(
            workspaceGm_[destinationOffset], source, copy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID5);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
    }

    __aicore__ inline void StorePreparedQuery(
        uint32_t row, uint32_t localQuery, uint32_t queryRow,
        LocalTensor<int32_t> compactedTokens,
        LocalTensor<int32_t> compactedSlots)
    {
        const DataCopyExtParams copy{
            1, SPARSE_COUNT * sizeof(int32_t), 0, 0, 0};
        SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
        DataCopyPad<int32_t, PaddingMode::Normal>(
            workspaceGm_[CompactedTokenBase(row, localQuery)],
            compactedTokens, copy);
        DataCopyPad<int32_t, PaddingMode::Normal>(
            destinationSlotsGm_[QueryBase(queryRow)], compactedSlots, copy);
        // This function is called repeatedly with the same output UB
        // buffers. Wait on the scalar side until both MTE3 copies finish so
        // the next query cannot overwrite them while DMA is still reading.
        SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
    }

    __aicore__ inline void StoreInvalidQueryRows(
        uint32_t queryBegin, uint32_t queryCount)
    {
        LocalTensor<int32_t> slots = outputSlotsBuf_.Get<int32_t>();
        Duplicate(slots, static_cast<int32_t>(-1), SPARSE_COUNT);
        PipeBarrier<PIPE_V>();
        const DataCopyExtParams copy{
            1, SPARSE_COUNT * sizeof(int32_t), 0, 0, 0};
        for (uint32_t localQuery = 0; localQuery < queryCount; ++localQuery) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
            DataCopyPad<int32_t, PaddingMode::Normal>(
                destinationSlotsGm_[QueryBase(queryBegin + localQuery)],
                slots, copy);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
        }
    }

    __aicore__ inline void PublishAllInvalidQueryRows()
    {
        LocalTensor<int32_t> slots = outputSlotsBuf_.Get<int32_t>();
        Duplicate(slots, static_cast<int32_t>(-1), SPARSE_COUNT);
        PipeBarrier<PIPE_V>();
        const DataCopyExtParams copy{
            1, SPARSE_COUNT * sizeof(int32_t), 0, 0, 0};
        for (uint32_t queryRow = coreIdx_;
             queryRow < tiling_->packedQueryCount;
             queryRow += tiling_->usedCoreNum) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
            DataCopyPad<int32_t, PaddingMode::Normal>(
                destinationSlotsGm_[QueryBase(queryRow)], slots, copy);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
        }
    }

    __aicore__ inline bool IsValidRow(
        uint32_t row, int32_t &budget, int32_t &candidateLen,
        int32_t &poolRow) const
    {
        budget = cacheTokensGm_.GetValue(row);
        candidateLen = candidateLensGm_.GetValue(row);
        poolRow = reqPoolEntriesGm_.GetValue(row);
        return budget >= static_cast<int32_t>(SPARSE_COUNT) &&
            budget <= static_cast<int32_t>(MAX_CACHE_TOKENS) &&
            budget <= static_cast<int32_t>(tiling_->residentCapacity) &&
            candidateLen >= static_cast<int32_t>(SPARSE_COUNT) &&
            candidateLen <= static_cast<int32_t>(tiling_->sourceCapacity) &&
            (static_cast<uint32_t>(candidateLen) + BLOCK_SIZE - 1) /
                    BLOCK_SIZE <= tiling_->qcap &&
            (static_cast<uint32_t>(budget) + BLOCK_SIZE - 1) /
                    BLOCK_SIZE <= tiling_->residentBlockCapacity &&
            poolRow >= 0 && poolRow < static_cast<int32_t>(tiling_->poolSize) &&
            poolRow < static_cast<int32_t>(tiling_->maxNumReqs);
    }

    // Native FULL graph replay may extend a uniform MTP cohort to the captured
    // request bucket. The extra rows use the single reserved pool entry and
    // must not take part in cache-map preparation. Keep the sentinel strict so
    // malformed real request metadata continues to fail closed.
    __aicore__ inline bool IsMtpGraphPaddingRow(
        uint32_t row, int32_t budget, int32_t candidateLen,
        int32_t poolRow) const
    {
        return budget == 0 && candidateLen == 0 &&
            logicalSeqLengthsGm_.GetValue(row) == 1 &&
            poolRow == static_cast<int32_t>(tiling_->maxNumReqs);
    }

    __aicore__ inline bool IsMtpGraphPaddingRow(uint32_t row) const
    {
        return IsMtpGraphPaddingRow(
            row, cacheTokensGm_.GetValue(row),
            candidateLensGm_.GetValue(row),
            reqPoolEntriesGm_.GetValue(row));
    }

    __aicore__ inline void LoadMissSlotMembership(uint32_t row)
    {
        LocalTensor<int32_t> membership =
            missMembershipBuf_.Get<int32_t>();
        if (graphDummy_) {
            Duplicate(
                membership, static_cast<int32_t>(0),
                MISS_BITMAP_STORAGE_WORDS);
            PipeBarrier<PIPE_V>();
            return;
        }
        const DataCopyExtParams copy{
            1,
            MISS_BITMAP_STORAGE_WORDS * sizeof(int32_t),
            0, 0, 0};
        const DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad<int32_t, PaddingMode::Normal>(
            membership,
            missSlotBitmapGm_[
                static_cast<uint64_t>(row) * MISS_BITMAP_STORAGE_WORDS],
            copy, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
    }

    __aicore__ inline void SanitizeGraphDummyTopk(
        LocalTensor<int32_t> topk, int32_t candidateLen)
    {
        if (!graphDummy_) {
            return;
        }
        SetFlag<HardEvent::V_S>(EVENT_ID4);
        WaitFlag<HardEvent::V_S>(EVENT_ID4);
        for (uint32_t index = 0; index < SPARSE_COUNT; ++index) {
            const int32_t token = topk.GetValue(index);
            if (token < 0 || token >= candidateLen) {
                topk.SetValue(index, 0);
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID4);
        WaitFlag<HardEvent::S_V>(EVENT_ID4);
    }

    __aicore__ inline void ClassifyQuery(
        uint32_t row, uint32_t localQuery, uint32_t queryRow,
        int32_t candidateLen, int32_t budget, int32_t poolRow)
    {
        LocalTensor<int32_t> topk = topkBuf_.Get<int32_t>();
        LocalTensor<uint32_t> membership =
            missMembershipBuf_.Get<uint32_t>();

        LoadTopk(queryRow, topk);
        SanitizeGraphDummyTopk(topk, candidateLen);
        const uint64_t tokenBase =
            static_cast<uint64_t>(poolRow) * tiling_->sourceCapacity;
        Simt::VF_CALL<ClassifyTopkSimt>(
            Simt::Dim3{SIMT_THREADS, 1, 1},
            (__ubuf__ int32_t *)topk.GetPhyAddr(),
            (__gm__ int32_t *)tokenToSlotPoolGm_[tokenBase].GetPhyAddr(),
            (__ubuf__ uint32_t *)membership.GetPhyAddr(),
            (__gm__ int32_t *)workspaceGm_[
                ClassificationBase(row, localQuery)].GetPhyAddr(),
            candidateLen, budget);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline uint32_t CompactQuery(
        uint32_t row, uint32_t localQuery, uint32_t queryRow,
        int32_t candidateLen, RequestTiming &timing)
    {
        LocalTensor<int32_t> topk = topkBuf_.Get<int32_t>();
        LocalTensor<int32_t> classification =
            classificationBuf_.Get<int32_t>();
        LocalTensor<int32_t> outputTokens = outputTokensBuf_.Get<int32_t>();
        LocalTensor<int32_t> outputSlots = outputSlotsBuf_.Get<int32_t>();

        const uint64_t compactStart = ProfileStart();
        // Classification for every packed query is already complete before
        // compaction starts. Reload this query's TopK row and consume only
        // its dedicated classification slice.
        LoadTopk(queryRow, topk);
        SanitizeGraphDummyTopk(topk, candidateLen);
        LoadWorkspaceVector(
            ClassificationBase(row, localQuery), classification,
            SPARSE_COUNT);
        Duplicate(outputTokens, static_cast<int32_t>(-1), SPARSE_COUNT);
        Duplicate(outputSlots, static_cast<int32_t>(-1), SPARSE_COUNT);
        PipeBarrier<PIPE_V>();
        __VEC_SCOPE__ {
            A5LiduFusedBatchGetPrepareMtpVector::SqueezeMissTokens(
                (__ubuf__ uint32_t *)outputTokens.GetPhyAddr(),
                (__ubuf__ uint32_t *)topk.GetPhyAddr(),
                (__ubuf__ uint32_t *)classification.GetPhyAddr(),
                VECTOR_LOOPS);
        }
        const uint32_t missCount = static_cast<uint32_t>(
            GetSpr<SpecialPurposeReg::AR>() / sizeof(uint32_t));
        if (missCount > SPARSE_COUNT) {
            return SPARSE_COUNT + 1;
        }
        PipeBarrier<PIPE_V>();
        __VEC_SCOPE__ {
            A5LiduFusedBatchGetPrepareMtpVector::SqueezeHitTokens(
                (__ubuf__ uint32_t *)outputTokens[missCount].GetPhyAddr(),
                (__ubuf__ uint32_t *)topk.GetPhyAddr(),
                (__ubuf__ uint32_t *)classification.GetPhyAddr(),
                VECTOR_LOOPS);
        }
        PipeBarrier<PIPE_V>();
        __VEC_SCOPE__ {
            A5LiduFusedBatchGetPrepareMtpVector::SqueezeHitSlots(
                (__ubuf__ uint32_t *)outputSlots[missCount].GetPhyAddr(),
                (__ubuf__ uint32_t *)classification.GetPhyAddr(),
                VECTOR_LOOPS);
        }
        PipeBarrier<PIPE_V>();
        StorePreparedQuery(
            row, localQuery, queryRow, outputTokens, outputSlots);
        ProfileAdd(timing.compact, compactStart);
        return missCount;
    }

    __aicore__ inline void InitializeDedup(
        int32_t candidateLen, RequestTiming &timing)
    {
        const uint64_t start = ProfileStart();
        const uint32_t sourceBitmapWords =
            ((static_cast<uint32_t>(candidateLen) + 31U) / 32U +
             UB_BLOCK_WORDS - 1U) /
            UB_BLOCK_WORDS * UB_BLOCK_WORDS;
        LocalTensor<uint32_t> sourceMembership =
            sourceMembershipBuf_.Get<uint32_t>();
        LocalTensor<int32_t> control = controlBuf_.Get<int32_t>();
        Duplicate(sourceMembership, static_cast<uint32_t>(0),
                  sourceBitmapWords);
        Duplicate(control, static_cast<int32_t>(0), REQUEST_CONTROL_WORDS);
        PipeBarrier<PIPE_V>();

        control.SetValue(DEDUP_UNIQUE_COUNT, 0);
        control.SetValue(DEDUP_STATUS, STATUS_OK);
        PipeBarrier<PIPE_ALL>();
        ProfileAdd(timing.dedup, start);
    }

    __aicore__ inline void DeduplicateCompactedQuery(
        LocalTensor<int32_t> compactedTokens, uint32_t missCount,
        int32_t candidateLen, RequestTiming &timing)
    {
        const uint64_t start = ProfileStart();
        LocalTensor<uint32_t> sourceMembership =
            sourceMembershipBuf_.Get<uint32_t>();
        LocalTensor<int32_t> uniqueSources =
            uniqueSourcesBuf_.Get<int32_t>();
        LocalTensor<int32_t> control = controlBuf_.Get<int32_t>();
        Simt::VF_CALL<DeduplicateOneQuerySimt>(
            Simt::Dim3{SIMT_THREADS, 1, 1},
            (__ubuf__ const int32_t *)compactedTokens.GetPhyAddr(),
            (__ubuf__ uint32_t *)sourceMembership.GetPhyAddr(),
            (__ubuf__ int32_t *)uniqueSources.GetPhyAddr(),
            (__ubuf__ int32_t *)
                control[DEDUP_UNIQUE_COUNT].GetPhyAddr(),
            missCount, candidateLen);
        PipeBarrier<PIPE_ALL>();
        ProfileAdd(timing.dedup, start);
    }

    __aicore__ inline int32_t FinishDedup(
        uint32_t row, uint32_t &uniqueCount, RequestTiming &timing)
    {
        const uint64_t start = ProfileStart();
        LocalTensor<int32_t> uniqueSources =
            uniqueSourcesBuf_.Get<int32_t>();
        LocalTensor<int32_t> control = controlBuf_.Get<int32_t>();
        uniqueCount = static_cast<uint32_t>(
            control.GetValue(DEDUP_UNIQUE_COUNT));
        const int32_t status = control.GetValue(DEDUP_STATUS);
        const uint32_t storedCount = uniqueCount > MTP_MISS_CAPACITY
            ? MTP_MISS_CAPACITY
            : uniqueCount;
        if (storedCount > 0) {
            StoreWorkspaceVector(
                UniqueSourceBase(row), uniqueSources, storedCount);
        }
        ProfileAdd(timing.dedup, start);
        if (status != STATUS_OK) {
            return status;
        }
        return uniqueCount > MTP_MISS_CAPACITY
            ? ERROR_UNIQUE_MISS_CAPACITY
            : STATUS_OK;
    }

    __aicore__ inline uint32_t FindFirstSetBit(uint32_t bits) const
    {
        uint32_t bit = 0;
        if ((bits & 0x0000FFFFU) == 0) {
            bits >>= 16;
            bit += 16;
        }
        if ((bits & 0x000000FFU) == 0) {
            bits >>= 8;
            bit += 8;
        }
        if ((bits & 0x0000000FU) == 0) {
            bits >>= 4;
            bit += 4;
        }
        if ((bits & 0x00000003U) == 0) {
            bits >>= 2;
            bit += 2;
        }
        if ((bits & 0x00000001U) == 0) {
            bit += 1;
        }
        return bit;
    }

    __aicore__ inline int32_t ResolveExtendedMissSlots(
        uint32_t row, uint32_t queryBegin, uint32_t queryCount,
        const uint32_t queryMissCounts[MAX_QUERIES_PER_REQUEST],
        uint32_t uniqueCount, uint32_t budget, int32_t poolRow,
        int32_t &proposedCursor, RequestTiming &timing)
    {
        const uint64_t start = ProfileStart();
        proposedCursor = -1;
        if (uniqueCount <= SPARSE_COUNT) {
            ProfileAdd(timing.slotSelect, start);
            return STATUS_OK;
        }
        if (uniqueCount > MTP_MISS_CAPACITY) {
            ProfileAdd(timing.slotSelect, start);
            return ERROR_UNIQUE_MISS_CAPACITY;
        }

        LocalTensor<uint32_t> protectedSlots =
            missMembershipBuf_.Get<uint32_t>();
        LocalTensor<int32_t> resolvedSlots =
            uniqueSourcesBuf_.Get<int32_t>();
        // Source deduplication is complete before slot selection starts.
        LocalTensor<int32_t> scanScratch = sourceMembershipBuf_.Get<int32_t>();

        // The first 2048 destinations preserve the input miss-slot order.
        // The supplied bitmap already protects those slots.
        LoadInputMissSlots(row, resolvedSlots);

        // Compact leaves every query's hit slots in destinationSlots after
        // its miss prefix. Merge all hit slots in one SIMT launch instead of
        // starting one MTE transfer and one scalar loop per query.
        const uint32_t hitBase0 = queryMissCounts[0];
        const uint32_t hitCount0 = queryCount > 0
            ? SPARSE_COUNT - queryMissCounts[0]
            : 0U;
        const uint32_t hitBase1 = SPARSE_COUNT + queryMissCounts[1];
        const uint32_t hitCount1 = queryCount > 1
            ? SPARSE_COUNT - queryMissCounts[1]
            : 0U;
        const uint32_t hitBase2 = 2U * SPARSE_COUNT + queryMissCounts[2];
        const uint32_t hitCount2 = queryCount > 2
            ? SPARSE_COUNT - queryMissCounts[2]
            : 0U;
        const uint32_t hitBase3 = 3U * SPARSE_COUNT + queryMissCounts[3];
        const uint32_t hitCount3 = queryCount > 3
            ? SPARSE_COUNT - queryMissCounts[3]
            : 0U;
        const uint32_t totalHitCount =
            hitCount0 + hitCount1 + hitCount2 + hitCount3;
        if (totalHitCount > 0) {
            LocalTensor<int32_t> control = controlBuf_.Get<int32_t>();
            control.SetValue(SLOT_SELECT_STATUS, STATUS_OK);
            PipeBarrier<PIPE_ALL>();
            Simt::VF_CALL<MergeHitSlotsSimt>(
                Simt::Dim3{SIMT_THREADS, 1, 1},
                (__gm__ int32_t *)destinationSlotsGm_[
                    QueryBase(queryBegin)].GetPhyAddr(),
                (__ubuf__ uint32_t *)protectedSlots.GetPhyAddr(),
                (__ubuf__ int32_t *)control[SLOT_SELECT_STATUS].GetPhyAddr(),
                hitBase0, hitCount0, hitBase1, hitCount1,
                hitBase2, hitCount2, hitBase3, hitCount3, budget);
            PipeBarrier<PIPE_ALL>();
            if (control.GetValue(SLOT_SELECT_STATUS) != STATUS_OK) {
                ProfileAdd(timing.slotSelect, start);
                return ERROR_RESOLVED_SLOT;
            }
        }

        const uint32_t extraCount = uniqueCount - SPARSE_COUNT;
        const int32_t cursorValue = evictionCursorsGm_.GetValue(poolRow);
        const uint32_t scanBegin = cursorValue >= 0
            ? static_cast<uint32_t>(cursorValue) % budget
            : 0U;
        const uint32_t firstWord = scanBegin / BITMAP_WORD_BITS;
        const uint32_t lastWord = (budget - 1U) / BITMAP_WORD_BITS;
        const uint32_t firstWordCount = lastWord - firstWord + 1U;
        const uint32_t secondWordCount = scanBegin == 0
            ? 0U
            : (scanBegin - 1U) / BITMAP_WORD_BITS + 1U;
        const uint32_t orderedWordCount =
            firstWordCount + secondWordCount;

        uint32_t selected = 0;
        if (extraCount <= SLOT_SELECT_SCALAR_EXTRA_LIMIT) {
            uint32_t lastSelected = scanBegin;
            // For a small replacement tail, avoid two extra VF launches.
            // Scan words in ring order and enumerate only the requested bits.
            for (uint32_t orderedWord = 0;
                 orderedWord < orderedWordCount &&
                     selected < extraCount;
                 ++orderedWord) {
                uint32_t wordIndex = 0;
                uint32_t segmentBegin = 0;
                uint32_t segmentEnd = scanBegin;
                if (orderedWord < firstWordCount) {
                    wordIndex = firstWord + orderedWord;
                    segmentBegin = scanBegin;
                    segmentEnd = budget;
                } else {
                    wordIndex = orderedWord - firstWordCount;
                }

                const uint32_t wordBase =
                    wordIndex * BITMAP_WORD_BITS;
                uint32_t validMask = 0xFFFFFFFFU;
                if (segmentBegin > wordBase) {
                    validMask &= 0xFFFFFFFFU
                        << (segmentBegin - wordBase);
                }
                if (segmentEnd < wordBase + BITMAP_WORD_BITS) {
                    const uint32_t validBits = segmentEnd - wordBase;
                    validMask &= (1U << validBits) - 1U;
                }

                uint32_t available =
                    ~protectedSlots.GetValue(wordIndex) & validMask;
                while (available != 0U && selected < extraCount) {
                    const uint32_t bit = FindFirstSetBit(available);
                    const uint32_t slot = wordBase + bit;
                    resolvedSlots.SetValue(SPARSE_COUNT + selected, slot);
                    lastSelected = slot;
                    ++selected;
                    available &= ~(1U << bit);
                }
            }
        } else {
            LocalTensor<int32_t> scanOffsets =
                scanScratch[SLOT_SELECT_SCAN_PREFIX_OFFSET];
            Simt::VF_CALL<CountAvailableBitmapWordsSimt>(
                Simt::Dim3{SIMT_THREADS, 1, 1},
                (__ubuf__ const uint32_t *)protectedSlots.GetPhyAddr(),
                (__ubuf__ int32_t *)scanScratch.GetPhyAddr(),
                budget, scanBegin, firstWordCount, secondWordCount);
            PipeBarrier<PIPE_ALL>();

            for (uint32_t orderedWord = 0;
                 orderedWord < orderedWordCount;
                 ++orderedWord) {
                const int32_t countValue =
                    scanScratch.GetValue(orderedWord);
                scanOffsets.SetValue(
                    orderedWord, static_cast<int32_t>(selected));
                if (selected >= extraCount || countValue <= 0) {
                    continue;
                }
                const uint32_t count =
                    static_cast<uint32_t>(countValue);
                const uint32_t remaining = extraCount - selected;
                selected += count < remaining ? count : remaining;
            }
            if (selected == extraCount) {
                PipeBarrier<PIPE_ALL>();
                Simt::VF_CALL<WriteAvailableBitmapSlotsSimt>(
                    Simt::Dim3{SIMT_THREADS, 1, 1},
                    (__ubuf__ const uint32_t *)protectedSlots.GetPhyAddr(),
                    (__ubuf__ const int32_t *)scanScratch.GetPhyAddr(),
                    (__ubuf__ const int32_t *)
                        scanOffsets.GetPhyAddr(),
                    (__ubuf__ int32_t *)resolvedSlots.GetPhyAddr(),
                    budget, scanBegin, firstWordCount,
                    secondWordCount, extraCount);
                PipeBarrier<PIPE_ALL>();
            }
        }
        if (selected != extraCount) {
            ProfileAdd(timing.slotSelect, start);
            return ERROR_INSUFFICIENT_EVICTION_SLOTS;
        }

        const uint32_t lastSelected = static_cast<uint32_t>(
            resolvedSlots.GetValue(SPARSE_COUNT + extraCount - 1U));
        proposedCursor = static_cast<int32_t>(
            lastSelected + 1U == budget ? 0U : lastSelected + 1U);
        StoreWorkspaceVector(
            ResolvedSlotBase(row), resolvedSlots, uniqueCount);
        ProfileAdd(timing.slotSelect, start);
        return STATUS_OK;
    }

    __aicore__ inline int32_t ValidatePreparedDestinations(
        uint32_t row, uint32_t uniqueCount, bool useResolvedSlots,
        uint32_t budget)
    {
        if (uniqueCount == 0) {
            return STATUS_OK;
        }
        if (uniqueCount > MTP_MISS_CAPACITY ||
            budget > tiling_->residentCapacity) {
            return ERROR_DESTINATION_SLOT;
        }
        LocalTensor<int32_t> slots = uniqueSourcesBuf_.Get<int32_t>();
        if (useResolvedSlots) {
            LoadWorkspaceVector(ResolvedSlotBase(row), slots, uniqueCount);
        } else {
            if (uniqueCount > SPARSE_COUNT) {
                return ERROR_DESTINATION_SLOT;
            }
            LoadInputMissSlotsVector(row, slots);
        }
        const uint32_t residentBlockCount =
            (budget + BLOCK_MASK) >> BLOCK_SHIFT;
        const uint64_t tableBase =
            static_cast<uint64_t>(row) * tiling_->residentBlockCapacity;
        LocalTensor<int32_t> physicalBlocks = topkBuf_.Get<int32_t>();
        const DataCopyExtParams copy{
            1,
            static_cast<uint32_t>(residentBlockCount * sizeof(int32_t)),
            0, 0, 0};
        const DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad<int32_t, PaddingMode::Normal>(
            physicalBlocks, residentBlockTableGm_[tableBase], copy, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        LocalTensor<int32_t> control = controlBuf_.Get<int32_t>();
        control.SetValue(SLOT_SELECT_STATUS, 0);
        PipeBarrier<PIPE_ALL>();
        Simt::VF_CALL<ValidatePreparedDestinationsSimt>(
            Simt::Dim3{SIMT_THREADS, 1, 1},
            (__ubuf__ const int32_t *)slots.GetPhyAddr(),
            (__ubuf__ const int32_t *)physicalBlocks.GetPhyAddr(),
            (__ubuf__ int32_t *)
                control[SLOT_SELECT_STATUS].GetPhyAddr(),
            uniqueCount, budget, tiling_->residentBlockCapacity);
        PipeBarrier<PIPE_ALL>();
        const int32_t validationFlags =
            control.GetValue(SLOT_SELECT_STATUS);
        if ((validationFlags & VALIDATION_DESTINATION_SLOT) != 0) {
            return ERROR_DESTINATION_SLOT;
        }
        if ((validationFlags & VALIDATION_RESIDENT_BLOCK) != 0) {
            return ERROR_RESIDENT_BLOCK;
        }
        return STATUS_OK;
    }

    __aicore__ inline void StoreRowControl(
        uint32_t row, int32_t status, uint32_t uniqueCount,
        uint32_t queryBegin, uint32_t queryCount,
        const uint32_t queryMissCounts[MAX_QUERIES_PER_REQUEST],
        bool usesResolvedSlots, int32_t proposedCursor)
    {
        LocalTensor<int32_t> control = controlBuf_.Get<int32_t>();
        Duplicate(control, static_cast<int32_t>(0), REQUEST_CONTROL_WORDS);
        PipeBarrier<PIPE_V>();
        control.SetValue(CONTROL_STATUS, status);
        control.SetValue(
            CONTROL_UNIQUE_MISS_COUNT, static_cast<int32_t>(uniqueCount));
        control.SetValue(
            CONTROL_QUERY_BEGIN, static_cast<int32_t>(queryBegin));
        control.SetValue(
            CONTROL_QUERY_COUNT, static_cast<int32_t>(queryCount));
        uint32_t rawMissCount = 0;
        for (uint32_t localQuery = 0;
             localQuery < MAX_QUERIES_PER_REQUEST; ++localQuery) {
            control.SetValue(
                CONTROL_QUERY_MISS_BEGIN + localQuery,
                static_cast<int32_t>(queryMissCounts[localQuery]));
            rawMissCount += queryMissCounts[localQuery];
        }
        control.SetValue(
            CONTROL_RAW_MISS_COUNT, static_cast<int32_t>(rawMissCount));
        control.SetValue(
            CONTROL_USES_RESOLVED_SLOTS, usesResolvedSlots ? 1 : 0);
        control.SetValue(
            CONTROL_PROPOSED_EVICTION_CURSOR, proposedCursor);
        control.SetValue(
            CONTROL_EXTRA_SLOT_COUNT,
            usesResolvedSlots
                ? static_cast<int32_t>(uniqueCount - SPARSE_COUNT)
                : 0);
        StoreScalarControl(
            RowControlBase(row), control, REQUEST_CONTROL_WORDS);
    }

    __aicore__ inline void PrepareMtpGraphPaddingRow(
        uint32_t row, uint32_t queryBegin, uint32_t queryCount)
    {
        // The captured MTP SFA node still consumes every fixed graph query.
        // Mark padding queries as invalid slots, but keep row status OK and
        // zero misses so subsequent map phases and descriptor generation are
        // true no-ops. The reserved pool row is never dereferenced here.
        const uint32_t queryMissCounts[MAX_QUERIES_PER_REQUEST] = {
            0, 0, 0, 0};
        StoreInvalidQueryRows(queryBegin, queryCount);
        StoreRowControl(
            row, STATUS_OK, 0, queryBegin, queryCount,
            queryMissCounts, false, -1);
    }

    __aicore__ inline void ClearVictims(
        uint32_t row, uint32_t lane, uint32_t uniqueCount,
        bool useResolvedSlots, int32_t candidateLen, int32_t poolRow,
        RequestTiming &timing)
    {
        const uint64_t start = ProfileStart();
        const uint64_t missBase =
            static_cast<uint64_t>(row) * SPARSE_COUNT;
        const uint64_t resolvedBase = ResolvedSlotBase(row);
        const uint64_t tokenBase =
            static_cast<uint64_t>(poolRow) * tiling_->sourceCapacity;
        const uint64_t slotBase =
            static_cast<uint64_t>(poolRow) * tiling_->residentCapacity;
        const uint32_t candidateCount =
            static_cast<uint32_t>(candidateLen);
        const uint32_t forwardShardCount =
            (candidateCount + MAP_SHARD_WORDS - 1) / MAP_SHARD_WORDS;
        const uint32_t forwardShardBegin =
            lane * forwardShardCount / tiling_->coresPerRequest;
        const uint32_t forwardShardEnd =
            (lane + 1) * forwardShardCount / tiling_->coresPerRequest;
        const uint32_t forwardBegin =
            forwardShardBegin * MAP_SHARD_WORDS;
        uint32_t forwardEnd = forwardShardEnd * MAP_SHARD_WORDS;
        if (forwardEnd > candidateCount) {
            forwardEnd = candidateCount;
        }
        Simt::VF_CALL<ClearVictimsSimt>(
            Simt::Dim3{SIMT_THREADS, 1, 1},
            (__gm__ int32_t *)missSlotsGm_[missBase].GetPhyAddr(),
            (__gm__ int32_t *)workspaceGm_[resolvedBase].GetPhyAddr(),
            (__gm__ int32_t *)
                tokenToSlotPoolGm_[tokenBase].GetPhyAddr(),
            (__gm__ int32_t *)
                slotToTokenPoolGm_[slotBase].GetPhyAddr(),
            uniqueCount, useResolvedSlots, tiling_->residentCapacity,
            forwardBegin, forwardEnd);
        PipeBarrier<PIPE_V>();
        const event_t clearDone = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(clearDone);
        WaitFlag<HardEvent::V_S>(clearDone);
        ProfileAdd(timing.clear, start);
    }

    __aicore__ inline void InstallMaps(
        uint32_t row, uint32_t lane, uint32_t uniqueCount,
        bool useResolvedSlots, uint32_t budget, uint32_t candidateLen,
        int32_t poolRow, RequestTiming &timing)
    {
        const uint64_t start = ProfileStart();
        const uint64_t missBase =
            static_cast<uint64_t>(row) * SPARSE_COUNT;
        const uint64_t resolvedBase = ResolvedSlotBase(row);
        const uint64_t tokenBase =
            static_cast<uint64_t>(poolRow) * tiling_->sourceCapacity;
        const uint64_t slotBase =
            static_cast<uint64_t>(poolRow) * tiling_->residentCapacity;
        const uint64_t sourceBase = UniqueSourceBase(row);
        const uint32_t forwardShardCount =
            (candidateLen + MAP_SHARD_WORDS - 1) / MAP_SHARD_WORDS;
        const uint32_t forwardShardBegin =
            lane * forwardShardCount / tiling_->coresPerRequest;
        const uint32_t forwardShardEnd =
            (lane + 1) * forwardShardCount / tiling_->coresPerRequest;
        const uint32_t forwardBegin = forwardShardBegin * MAP_SHARD_WORDS;
        uint32_t forwardEnd = forwardShardEnd * MAP_SHARD_WORDS;
        if (forwardEnd > candidateLen) {
            forwardEnd = candidateLen;
        }

        const uint32_t reverseShardCount =
            (budget + MAP_SHARD_WORDS - 1) / MAP_SHARD_WORDS;
        const uint32_t reverseShardBegin =
            lane * reverseShardCount / tiling_->coresPerRequest;
        const uint32_t reverseShardEnd =
            (lane + 1) * reverseShardCount / tiling_->coresPerRequest;
        const uint32_t reverseBegin = reverseShardBegin * MAP_SHARD_WORDS;
        uint32_t reverseEnd = reverseShardEnd * MAP_SHARD_WORDS;
        if (reverseEnd > budget) {
            reverseEnd = budget;
        }

        Simt::VF_CALL<InstallMapsSimt>(
            Simt::Dim3{SIMT_THREADS, 1, 1},
            (__gm__ int32_t *)workspaceGm_[sourceBase].GetPhyAddr(),
            (__gm__ int32_t *)missSlotsGm_[missBase].GetPhyAddr(),
            (__gm__ int32_t *)workspaceGm_[resolvedBase].GetPhyAddr(),
            (__gm__ int32_t *)
                tokenToSlotPoolGm_[tokenBase].GetPhyAddr(),
            (__gm__ int32_t *)
                slotToTokenPoolGm_[slotBase].GetPhyAddr(),
            uniqueCount, useResolvedSlots, tiling_->residentCapacity,
            forwardBegin, forwardEnd,
            reverseBegin, reverseEnd);
        PipeBarrier<PIPE_V>();
        const event_t installDone = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(installDone);
        WaitFlag<HardEvent::V_S>(installDone);
        ProfileAdd(timing.install, start);
    }

    __aicore__ inline void RefillMissPrefixes(
        uint32_t row, uint32_t queryBegin, uint32_t queryCount,
        uint32_t lane, int32_t poolRow,
        LocalTensor<int32_t> queryMissCounts, RequestTiming &timing)
    {
        const uint64_t start = ProfileStart();
        const uint64_t tokenBase =
            static_cast<uint64_t>(poolRow) * tiling_->sourceCapacity;
        Simt::VF_CALL<RefillMissPrefixesSimt>(
            Simt::Dim3{SIMT_THREADS, 1, 1},
            (__gm__ int32_t *)workspaceGm_[
                CompactedTokenBase(row, 0)].GetPhyAddr(),
            (__gm__ int32_t *)
                tokenToSlotPoolGm_[tokenBase].GetPhyAddr(),
            (__gm__ int32_t *)destinationSlotsGm_[
                QueryBase(queryBegin)].GetPhyAddr(),
            (__ubuf__ int32_t *)queryMissCounts.GetPhyAddr(),
            queryCount, lane, tiling_->coresPerRequest);
        PipeBarrier<PIPE_V>();
        const event_t refillDone = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(refillDone);
        WaitFlag<HardEvent::V_S>(refillDone);
        ProfileAdd(timing.refill, start);
    }

    __aicore__ inline void PrepareRequestAfterClassification(
        uint32_t row, int32_t status, uint32_t queryBegin,
        uint32_t queryCount, int32_t candidateLen, uint32_t budget,
        int32_t poolRow, RequestTiming &timing)
    {
        uint32_t queryMissCounts[MAX_QUERIES_PER_REQUEST] = {0, 0, 0, 0};
        uint32_t uniqueCount = 0;
        bool usesResolvedSlots = false;
        bool dedupInitialized = false;
        int32_t proposedCursor = -1;
        LocalTensor<int32_t> compactedTokens =
            outputTokensBuf_.Get<int32_t>();
        if (status == STATUS_OK) {
            for (uint32_t localQuery = 0;
                 localQuery < queryCount; ++localQuery) {
                const uint32_t missCount = CompactQuery(
                    row, localQuery, queryBegin + localQuery,
                    candidateLen, timing);
                if (missCount > SPARSE_COUNT) {
                    status = ERROR_COUNT_ACCOUNTING;
                    break;
                }
                queryMissCounts[localQuery] = missCount;
                if (missCount > 0) {
                    if (!dedupInitialized) {
                        InitializeDedup(candidateLen, timing);
                        dedupInitialized = true;
                    }
                    DeduplicateCompactedQuery(
                        compactedTokens, missCount, candidateLen, timing);
                }
            }
        }
        if (status == STATUS_OK && dedupInitialized) {
            status = FinishDedup(row, uniqueCount, timing);
        }
        if (status == STATUS_OK && uniqueCount > SPARSE_COUNT) {
            status = ResolveExtendedMissSlots(
                row, queryBegin, queryCount, queryMissCounts,
                uniqueCount, budget, poolRow, proposedCursor, timing);
            usesResolvedSlots = status == STATUS_OK;
        }
        if (status == STATUS_OK) {
            status = ValidatePreparedDestinations(
                row, uniqueCount, usesResolvedSlots, budget);
        }

        if (status != STATUS_OK) {
            StoreInvalidQueryRows(queryBegin, queryCount);
        }
        StoreRowControl(
            row, status, uniqueCount, queryBegin, queryCount,
            queryMissCounts, usesResolvedSlots, proposedCursor);
    }

    __aicore__ inline void ClearPreparedRequest(
        uint32_t row, uint32_t lane, RequestTiming &timing)
    {
        LocalTensor<int32_t> control = controlBuf_.Get<int32_t>();
        LoadRowControl(row, control);
        const int32_t status = control.GetValue(CONTROL_STATUS);
        const int32_t uniqueCount = control.GetValue(
            CONTROL_UNIQUE_MISS_COUNT);
        if (status != STATUS_OK || uniqueCount <= 0 ||
            uniqueCount > static_cast<int32_t>(MTP_MISS_CAPACITY)) {
            return;
        }
        const bool useResolvedSlots = control.GetValue(
            CONTROL_USES_RESOLVED_SLOTS) != 0;
        ClearVictims(
            row, lane, static_cast<uint32_t>(uniqueCount),
            useResolvedSlots, candidateLensGm_.GetValue(row),
            reqPoolEntriesGm_.GetValue(row), timing);
    }

    __aicore__ inline void InstallPreparedRequest(
        uint32_t row, uint32_t lane, RequestTiming &timing)
    {
        LocalTensor<int32_t> control = controlBuf_.Get<int32_t>();
        LoadRowControl(row, control);
        const int32_t status = control.GetValue(CONTROL_STATUS);
        const int32_t uniqueCountValue = control.GetValue(
            CONTROL_UNIQUE_MISS_COUNT);
        if (status != STATUS_OK || uniqueCountValue <= 0 ||
            uniqueCountValue > static_cast<int32_t>(MTP_MISS_CAPACITY)) {
            return;
        }
        const uint32_t uniqueCount = static_cast<uint32_t>(uniqueCountValue);
        const bool useResolvedSlots = control.GetValue(
            CONTROL_USES_RESOLVED_SLOTS) != 0;
        const int32_t poolRow = reqPoolEntriesGm_.GetValue(row);
        InstallMaps(
            row, lane, uniqueCount, useResolvedSlots,
            static_cast<uint32_t>(cacheTokensGm_.GetValue(row)),
            static_cast<uint32_t>(candidateLensGm_.GetValue(row)),
            poolRow, timing);
    }

    __aicore__ inline void RefillPreparedRequest(
        uint32_t row, uint32_t lane, RequestTiming &timing)
    {
        LocalTensor<int32_t> control = controlBuf_.Get<int32_t>();
        LoadRowControl(row, control);
        const int32_t status = control.GetValue(CONTROL_STATUS);
        const int32_t uniqueCountValue = control.GetValue(
            CONTROL_UNIQUE_MISS_COUNT);
        if (status != STATUS_OK || uniqueCountValue <= 0 ||
            uniqueCountValue > static_cast<int32_t>(MTP_MISS_CAPACITY)) {
            return;
        }
        const uint32_t queryBegin = static_cast<uint32_t>(
            control.GetValue(CONTROL_QUERY_BEGIN));
        const uint32_t queryCount = static_cast<uint32_t>(
            control.GetValue(CONTROL_QUERY_COUNT));
        const int32_t poolRow = reqPoolEntriesGm_.GetValue(row);
        RefillMissPrefixes(
            row, queryBegin, queryCount, lane, poolRow,
            control[CONTROL_QUERY_MISS_BEGIN], timing);
    }

    __aicore__ inline int32_t CalculateResidentLength(uint32_t row) const
    {
        const int32_t cacheTokens = cacheTokensGm_.GetValue(row);
        const int32_t candidateLen = candidateLensGm_.GetValue(row);
        const int32_t logicalLen = logicalSeqLengthsGm_.GetValue(row);
        if (cacheTokens >= static_cast<int32_t>(SPARSE_COUNT) &&
            candidateLen >= static_cast<int32_t>(SPARSE_COUNT) &&
            logicalLen >= candidateLen) {
            return cacheTokens + logicalLen - candidateLen;
        }
        if (cacheTokens == 0 && logicalLen >= 0) {
            return logicalLen;
        }
        return 0;
    }

    __aicore__ inline void LoadRowControl(
        uint32_t row, LocalTensor<int32_t> control)
    {
        LoadWorkspaceScalar(
            RowControlBase(row), control, REQUEST_CONTROL_WORDS);
    }

    __aicore__ inline void WriteBatchOutputs(bool layoutInvalid)
    {
        LocalTensor<int32_t> counts = outputTokensBuf_.Get<int32_t>();
        LocalTensor<int32_t> resident = topkBuf_.Get<int32_t>();
        LocalTensor<int32_t> control = controlBuf_.Get<int32_t>();
        uint32_t totalMissCount = 0;
        for (uint32_t offset = 0; offset < tiling_->batchSize;
             offset += SPARSE_COUNT) {
            uint32_t count = tiling_->batchSize - offset;
            if (count > SPARSE_COUNT) {
                count = SPARSE_COUNT;
            }
            SetFlag<HardEvent::V_S>(EVENT_ID4);
            WaitFlag<HardEvent::V_S>(EVENT_ID4);
            for (uint32_t index = 0; index < count; ++index) {
                int32_t status = ERROR_QUERY_LAYOUT;
                int32_t uniqueCount = 0;
                if (!layoutInvalid) {
                    LoadRowControl(offset + index, control);
                    status = control.GetValue(CONTROL_STATUS);
                    uniqueCount = control.GetValue(
                        CONTROL_UNIQUE_MISS_COUNT);
                }
                if (status == STATUS_OK && uniqueCount > 0 &&
                    uniqueCount <= static_cast<int32_t>(MTP_MISS_CAPACITY)) {
                    totalMissCount += static_cast<uint32_t>(uniqueCount);
                }
                if (status == STATUS_OK &&
                    control.GetValue(CONTROL_USES_RESOLVED_SLOTS) != 0) {
                    const int32_t poolRow = reqPoolEntriesGm_.GetValue(
                        offset + index);
                    const int32_t proposedCursor = control.GetValue(
                        CONTROL_PROPOSED_EVICTION_CURSOR);
                    if (poolRow >= 0 && proposedCursor >= 0) {
                        evictionCursorsGm_.SetValue(poolRow, proposedCursor);
                    }
                }
                counts.SetValue(
                    index, status == STATUS_OK ? uniqueCount : -1);
                resident.SetValue(
                    index, status == STATUS_OK
                        ? CalculateResidentLength(offset + index)
                        : 0);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID5);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID5);
            const DataCopyExtParams copy{
                1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad<int32_t, PaddingMode::Normal>(
                missCountsGm_[offset], counts, copy);
            DataCopyPad<int32_t, PaddingMode::Normal>(
                residentSeqLengthsGm_[offset], resident, copy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID5);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
        }
        counts.SetValue(0, static_cast<int32_t>(totalMissCount));
        SetFlag<HardEvent::S_MTE3>(EVENT_ID5);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID5);
        const DataCopyExtParams totalCopy{
            1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad<int32_t, PaddingMode::Normal>(
            missCountsGm_[tiling_->batchSize], counts, totalCopy);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID5);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
    }

    __aicore__ inline uint32_t LoadCompactPrefixes(
        LocalTensor<int32_t> prefixes)
    {
        uint32_t total = 0;
        prefixes.SetValue(0, 0);
        LocalTensor<int32_t> control = controlBuf_.Get<int32_t>();
        for (uint32_t row = 0; row < tiling_->batchSize; ++row) {
            LoadRowControl(row, control);
            const int32_t status = control.GetValue(CONTROL_STATUS);
            const int32_t count = control.GetValue(
                CONTROL_UNIQUE_MISS_COUNT);
            if (status == STATUS_OK && count > 0 &&
                count <= static_cast<int32_t>(MTP_MISS_CAPACITY)) {
                total += static_cast<uint32_t>(count);
            }
            prefixes.SetValue(row + 1, static_cast<int32_t>(total));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID4);
        WaitFlag<HardEvent::S_V>(EVENT_ID4);
        return total;
    }

    __aicore__ inline void GenerateCompactDescriptorTask()
    {
        LocalTensor<int32_t> prefixes =
            missMembershipBuf_.Get<uint32_t>()
                .template ReinterpretCast<int32_t>();
        const uint32_t total = LoadCompactPrefixes(prefixes);
        if (total == 0) {
            return;
        }
        const uint32_t totalLines =
            (total + GM_CACHE_LINE_WORDS - 1) / GM_CACHE_LINE_WORDS;
        const uint32_t lineBegin = static_cast<uint32_t>(
            static_cast<uint64_t>(totalLines) * coreIdx_ /
            tiling_->usedCoreNum);
        const uint32_t lineEnd = static_cast<uint32_t>(
            static_cast<uint64_t>(totalLines) * (coreIdx_ + 1) /
            tiling_->usedCoreNum);
        if (lineBegin >= lineEnd) {
            return;
        }
        const uint32_t taskBegin = lineBegin * GM_CACHE_LINE_WORDS;
        uint32_t taskEnd = lineEnd * GM_CACHE_LINE_WORDS;
        if (taskEnd > total) {
            taskEnd = total;
        }
        Simt::VF_CALL<GenerateCompactDescriptorsSimt>(
            Simt::Dim3{DESCRIPTOR_SIMT_THREADS, 1, 1},
            (__gm__ const int32_t *)workspaceGm_[0].GetPhyAddr(),
            (__gm__ const int32_t *)missSlotsGm_[0].GetPhyAddr(),
            (__gm__ const int32_t *)
                residentBlockTableGm_[0].GetPhyAddr(),
            (__gm__ const int32_t *)reqPoolEntriesGm_[0].GetPhyAddr(),
            (__ubuf__ int32_t *)prefixes.GetPhyAddr(),
            (__gm__ int32_t *)indicesGm_[taskBegin].GetPhyAddr(),
            (__gm__ int32_t *)offsetsGm_[taskBegin].GetPhyAddr(),
            (__gm__ int32_t *)outputIndicesGm_[taskBegin].GetPhyAddr(),
            taskBegin, taskEnd - taskBegin, tiling_->batchSize,
            tiling_->residentCapacity, tiling_->residentBlockCapacity,
            tiling_->numLayers,
            tiling_->layerId, tiling_->qcap);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline uint64_t ProfileBase(uint32_t row) const
    {
        return (static_cast<uint64_t>(row) *
                A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE_LANES) *
            A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE_FIELDS;
    }

    __aicore__ inline void InitializeSequentialProfile(
        uint32_t row, uint64_t totalStart)
    {
        const uint64_t base = ProfileBase(row);
        for (uint32_t field = 0;
             field < A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE_FIELDS;
             ++field) {
            profilingGm_.SetValue(base + field, 0);
        }
        // Keep the timestamp in a reserved field until all phase passes have
        // completed. FinalizeSequentialProfile converts it into PROFILE_TOTAL.
        profilingGm_.SetValue(
            base + PROFILE_SEQUENTIAL_START,
            static_cast<int64_t>(totalStart));
    }

    __aicore__ inline void StoreSequentialStageTiming(
        uint32_t row, uint32_t field, uint64_t value)
    {
        profilingGm_.SetValue(
            ProfileBase(row) + field, static_cast<int64_t>(value));
    }

    __aicore__ inline void StoreSequentialPreparationTiming(
        uint32_t row, const RequestTiming &timing)
    {
        const uint64_t base = ProfileBase(row);
        profilingGm_.SetValue(
            base + PROFILE_MEMBERSHIP,
            static_cast<int64_t>(timing.membership));
        profilingGm_.SetValue(
            base + PROFILE_CLASSIFY,
            static_cast<int64_t>(timing.classify));
        profilingGm_.SetValue(
            base + PROFILE_COMPACT,
            static_cast<int64_t>(timing.compact));
        profilingGm_.SetValue(
            base + PROFILE_DEDUP,
            static_cast<int64_t>(timing.dedup));
        profilingGm_.SetValue(
            base + PROFILE_SLOT_SELECT,
            static_cast<int64_t>(timing.slotSelect));
    }

    __aicore__ inline void FinalizeSequentialProfile(
        uint32_t row, uint64_t descriptor)
    {
        const uint64_t base = ProfileBase(row);
        const uint64_t totalStart = static_cast<uint64_t>(
            profilingGm_.GetValue(base + PROFILE_SEQUENTIAL_START));
        RequestTiming timing;
        timing.membership = static_cast<uint64_t>(
            profilingGm_.GetValue(base + PROFILE_MEMBERSHIP));
        timing.classify = static_cast<uint64_t>(
            profilingGm_.GetValue(base + PROFILE_CLASSIFY));
        timing.compact = static_cast<uint64_t>(
            profilingGm_.GetValue(base + PROFILE_COMPACT));
        timing.dedup = static_cast<uint64_t>(
            profilingGm_.GetValue(base + PROFILE_DEDUP));
        timing.slotSelect = static_cast<uint64_t>(
            profilingGm_.GetValue(base + PROFILE_SLOT_SELECT));
        timing.clear = static_cast<uint64_t>(
            profilingGm_.GetValue(base + PROFILE_CLEAR));
        timing.install = static_cast<uint64_t>(
            profilingGm_.GetValue(base + PROFILE_INSTALL));
        timing.refill = static_cast<uint64_t>(
            profilingGm_.GetValue(base + PROFILE_REFILL));
        timing.descriptor = descriptor;
        WriteProfile(row, 0, ReadCycle() - totalStart, timing);
    }

    __aicore__ inline void WriteProfile(
        uint32_t row, uint32_t lane, uint64_t total,
        const RequestTiming &timing)
    {
        LocalTensor<int32_t> control = controlBuf_.Get<int32_t>();
        LoadRowControl(row, control);
        const uint64_t base =
            (static_cast<uint64_t>(row) *
                 A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE_LANES + lane) *
            A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE_FIELDS;
        for (uint32_t field = 0;
             field < A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_PROFILE_FIELDS;
             ++field) {
            profilingGm_.SetValue(base + field, 0);
        }
        profilingGm_.SetValue(base + PROFILE_TOTAL,
                              static_cast<int64_t>(total));
        profilingGm_.SetValue(base + PROFILE_MEMBERSHIP,
                              static_cast<int64_t>(timing.membership));
        profilingGm_.SetValue(base + PROFILE_CLASSIFY,
                              static_cast<int64_t>(timing.classify));
        profilingGm_.SetValue(base + PROFILE_COMPACT,
                              static_cast<int64_t>(timing.compact));
        profilingGm_.SetValue(base + PROFILE_DEDUP,
                              static_cast<int64_t>(timing.dedup));
        profilingGm_.SetValue(base + PROFILE_CLEAR,
                              static_cast<int64_t>(timing.clear));
        profilingGm_.SetValue(base + PROFILE_INSTALL,
                              static_cast<int64_t>(timing.install));
        profilingGm_.SetValue(base + PROFILE_REFILL,
                              static_cast<int64_t>(timing.refill));
        profilingGm_.SetValue(base + PROFILE_DESCRIPTOR,
                              static_cast<int64_t>(timing.descriptor));
        profilingGm_.SetValue(base + PROFILE_STATUS,
                              control.GetValue(CONTROL_STATUS));
        profilingGm_.SetValue(
            base + PROFILE_UNIQUE_MISS_COUNT,
            control.GetValue(CONTROL_UNIQUE_MISS_COUNT));
        profilingGm_.SetValue(base + PROFILE_RAW_MISS_COUNT,
                              control.GetValue(CONTROL_RAW_MISS_COUNT));
        profilingGm_.SetValue(base + PROFILE_QUERY_BEGIN,
                              control.GetValue(CONTROL_QUERY_BEGIN));
        profilingGm_.SetValue(base + PROFILE_QUERY_COUNT,
                              control.GetValue(CONTROL_QUERY_COUNT));
        for (uint32_t query = 0; query < MAX_QUERIES_PER_REQUEST; ++query) {
            profilingGm_.SetValue(
                base + PROFILE_QUERY0_MISS_COUNT + query,
                control.GetValue(CONTROL_QUERY_MISS_BEGIN + query));
        }
        profilingGm_.SetValue(base + PROFILE_SLOT_SELECT,
                              static_cast<int64_t>(timing.slotSelect));
    }

private:
    TPipe *pipe_;
    const A5LiduFusedBatchGetPrepareMtpTilingData *tiling_;
    uint32_t coreIdx_ = 0;
    GlobalTensor<int32_t> topkIndicesGm_;
    GlobalTensor<int32_t> actualSeqLengthsQueryGm_;
    GlobalTensor<int32_t> reqPoolEntriesGm_;
    GlobalTensor<int32_t> tokenToSlotPoolGm_;
    GlobalTensor<int32_t> slotToTokenPoolGm_;
    GlobalTensor<int32_t> evictionCursorsGm_;
    GlobalTensor<int32_t> cacheTokensGm_;
    GlobalTensor<int32_t> candidateLensGm_;
    GlobalTensor<int32_t> missSlotsGm_;
    GlobalTensor<int32_t> missSlotBitmapGm_;
    GlobalTensor<int32_t> residentBlockTableGm_;
    GlobalTensor<int32_t> logicalSeqLengthsGm_;
    GlobalTensor<int32_t> profileEnabledGm_;
    GlobalTensor<int32_t> destinationSlotsGm_;
    GlobalTensor<int32_t> missCountsGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<int32_t> offsetsGm_;
    GlobalTensor<int32_t> outputIndicesGm_;
    GlobalTensor<int32_t> residentSeqLengthsGm_;
    GlobalTensor<int64_t> profilingGm_;
    GlobalTensor<int32_t> workspaceGm_;
    TBuf<TPosition::VECCALC> missMembershipBuf_;
    TBuf<TPosition::VECCALC> sourceMembershipBuf_;
    TBuf<TPosition::VECCALC> uniqueSourcesBuf_;
    TBuf<TPosition::VECCALC> topkBuf_;
    TBuf<TPosition::VECCALC> classificationBuf_;
    TBuf<TPosition::VECCALC> outputTokensBuf_;
    TBuf<TPosition::VECCALC> outputSlotsBuf_;
    TBuf<TPosition::VECCALC> controlBuf_;
    bool profileEnabled_ = false;
    bool graphDummy_ = false;
};
}  // namespace

extern "C" __global__ __aicore__ void a5_lidu_fused_batch_get_prepare_mtp(
    GM_ADDR topkIndices,
    GM_ADDR actualSeqLengthsQuery,
    GM_ADDR reqPoolEntries,
    GM_ADDR tokenToSlotPool,
    GM_ADDR slotToTokenPool,
    GM_ADDR evictionCursors,
    GM_ADDR cacheTokens,
    GM_ADDR candidateLens,
    GM_ADDR missSlots,
    GM_ADDR missSlotBitmap,
    GM_ADDR residentBlockTable,
    GM_ADDR logicalSeqLengths,
    GM_ADDR profileEnabled,
    GM_ADDR destinationSlots,
    GM_ADDR missCounts,
    GM_ADDR indices,
    GM_ADDR offsets,
    GM_ADDR outputIndices,
    GM_ADDR residentSeqLengths,
    GM_ADDR tokenToSlotAlias,
    GM_ADDR slotToTokenAlias,
    GM_ADDR evictionCursorsAlias,
    GM_ADDR profiling,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)tokenToSlotAlias;
    (void)slotToTokenAlias;
    (void)evictionCursorsAlias;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(A5LiduFusedBatchGetPrepareMtpTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    A5LiduFusedBatchGetPrepareMtpKernel op(&pipe, &tilingData);
    op.Init(
        topkIndices, actualSeqLengthsQuery, reqPoolEntries,
        tokenToSlotPool, slotToTokenPool, evictionCursors, cacheTokens,
        candidateLens, missSlots, missSlotBitmap, residentBlockTable,
        logicalSeqLengths, profileEnabled, destinationSlots, missCounts,
        indices, offsets, outputIndices, residentSeqLengths, profiling,
        workspace);
    op.Process();
}
