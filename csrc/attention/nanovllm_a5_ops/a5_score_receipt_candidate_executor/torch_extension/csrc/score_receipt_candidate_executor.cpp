// P7 stage-1 bridge.  The executor is a standalone, after-publisher score
// consumer; it has no C8 implementation dependency and no transport ABI.

#include <limits>
#include <tuple>

#include <torch/extension.h>
#include <torch/library.h>

#include "op_api_common.h"

namespace {
constexpr int64_t kReceiptWords = 16;
constexpr int64_t kTileWidth = 128;
constexpr int64_t kSeedWords = 4;
constexpr int64_t kReceiptIdentityWords = 4;

struct Geometry {
    int64_t rows;
    int64_t sources;
    int64_t pools;
    int64_t objects;
    int64_t tilesPerRow;
    int64_t receipts;
};

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    TORCH_CHECK(value > 0 && divisor > 0, "positive geometry is required");
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

void CheckNpuTensor(const at::Tensor& tensor, at::ScalarType type,
                    const at::Device& device, const char* name)
{
    TORCH_CHECK(tensor.device() == device && tensor.is_contiguous() &&
                    tensor.scalar_type() == type,
                name, " must be a contiguous ", c10::toString(type),
                " tensor on the score workspace NPU.");
}

Geometry Validate(
    const at::Tensor& scoreWorkspace, const at::Tensor& tileReceipts,
    const at::Tensor& receiptIdentity, const at::Tensor& scoreSourceLens,
    const at::Tensor& residentThreshold,
    const at::Tensor& residentThresholdValid,
    const at::Tensor& kvObjectSlotsSnapshot,
    const at::Tensor& reqPoolEntries, const at::Tensor& kvObjectMap,
    const at::Tensor& kvObjectLens, bool requireNpu)
{
    TORCH_CHECK(scoreWorkspace.dim() == 1 && scoreWorkspace.numel() > 0,
                "score_workspace must be int32[W], W > 0.");
    TORCH_CHECK(scoreSourceLens.dim() == 1 && scoreSourceLens.numel() > 0,
                "score_source_lens must be int32[R], R > 0.");
    TORCH_CHECK(kvObjectMap.dim() == 2 && kvObjectMap.size(0) > 0 &&
                    kvObjectMap.size(1) > 0,
                "kv_object_map must be int32[P,S], P,S > 0.");
    TORCH_CHECK(kvObjectSlotsSnapshot.dim() == 2 &&
                    kvObjectSlotsSnapshot.size(0) == kvObjectMap.size(0) &&
                    kvObjectSlotsSnapshot.size(1) > 0,
                "kv_object_slots_snapshot must be int32[P,O], P aligned to map.");
    const int64_t rows = scoreSourceLens.numel();
    const int64_t sources = kvObjectMap.size(1);
    const int64_t pools = kvObjectMap.size(0);
    const int64_t objects = kvObjectSlotsSnapshot.size(1);
    const int64_t tiles = CeilDiv(sources, kTileWidth);
    TORCH_CHECK(rows <= std::numeric_limits<int64_t>::max() / tiles,
                "receipt geometry overflows int64.");
    const int64_t receipts = rows * tiles;
    TORCH_CHECK(tileReceipts.dim() == 2 && tileReceipts.size(0) == receipts &&
                    tileReceipts.size(1) == kReceiptWords,
                "tile_receipts must be int64[R*ceil(S/128),16].");
    TORCH_CHECK(receiptIdentity.dim() == 2 &&
                    receiptIdentity.size(0) == receipts &&
                    receiptIdentity.size(1) == kReceiptIdentityWords &&
                    residentThreshold.dim() == 1 &&
                    residentThreshold.numel() == rows &&
                    residentThresholdValid.dim() == 1 &&
                    residentThresholdValid.numel() == rows &&
                    reqPoolEntries.dim() == 1 && reqPoolEntries.numel() == rows &&
                    kvObjectLens.dim() == 1 && kvObjectLens.numel() == rows,
                "per-row score metadata must be [R].");
    TORCH_CHECK(scoreWorkspace.numel() >= (rows * tiles * kTileWidth + 1) / 2,
                "score_workspace cannot hold C8's padded uint16 score rows.");
    if (requireNpu) {
        TORCH_CHECK(scoreWorkspace.device().is_privateuseone(),
                    "score_receipt_candidate_executor requires NPU inputs.");
        const at::Device device = scoreWorkspace.device();
        CheckNpuTensor(scoreWorkspace, at::kInt, device, "score_workspace");
        CheckNpuTensor(tileReceipts, at::kLong, device, "tile_receipts");
        CheckNpuTensor(receiptIdentity, at::kLong, device, "receipt_identity");
        CheckNpuTensor(scoreSourceLens, at::kInt, device, "score_source_lens");
        CheckNpuTensor(residentThreshold, at::kInt, device, "resident_threshold");
        CheckNpuTensor(residentThresholdValid, at::kInt, device,
                       "resident_threshold_valid");
        CheckNpuTensor(kvObjectSlotsSnapshot, at::kInt, device,
                       "kv_object_slots_snapshot");
        CheckNpuTensor(reqPoolEntries, at::kInt, device, "req_pool_entries");
        CheckNpuTensor(kvObjectMap, at::kInt, device, "kv_object_map");
        CheckNpuTensor(kvObjectLens, at::kInt, device, "kv_object_lens");
    }
    return {rows, sources, pools, objects, tiles, receipts};
}

using Result = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                          at::Tensor>;

Result RunNpu(
    const at::Tensor& scoreWorkspace, const at::Tensor& tileReceipts,
    const at::Tensor& receiptIdentity, const at::Tensor& scoreSourceLens,
    const at::Tensor& residentThreshold,
    const at::Tensor& residentThresholdValid,
    const at::Tensor& kvObjectSlotsSnapshot,
    const at::Tensor& reqPoolEntries, const at::Tensor& kvObjectMap,
    const at::Tensor& kvObjectLens)
{
    const Geometry geometry = Validate(
        scoreWorkspace, tileReceipts, receiptIdentity, scoreSourceLens,
        residentThreshold, residentThresholdValid, kvObjectSlotsSnapshot,
        reqPoolEntries, kvObjectMap, kvObjectLens, true);
    const auto int32Options = scoreWorkspace.options().dtype(at::kInt);
    const auto int64Options = scoreWorkspace.options().dtype(at::kLong);
    auto candidateSeeds = at::empty({geometry.receipts, kTileWidth, kSeedWords},
                                    int32Options);
    auto candidateCounts = at::empty({geometry.receipts}, int32Options);
    auto candidateReceiptIdentity = at::empty(
        {geometry.receipts, kReceiptIdentityWords}, int64Options);
    auto receiptStatus = at::empty({geometry.receipts}, int32Options);
    auto receiptOwner = at::empty({geometry.receipts}, int32Options);
    auto keepalive = std::make_tuple(
        scoreWorkspace, tileReceipts, receiptIdentity, scoreSourceLens,
        residentThreshold, residentThresholdValid, kvObjectSlotsSnapshot,
        reqPoolEntries, kvObjectMap, kvObjectLens, candidateSeeds,
        candidateCounts, candidateReceiptIdentity, receiptStatus, receiptOwner);
    EXEC_NPU_CMD_ORDERED(
        aclnnA5ScoreReceiptCandidateExecutor, keepalive, scoreWorkspace,
        tileReceipts, receiptIdentity, scoreSourceLens, residentThreshold,
        residentThresholdValid, kvObjectSlotsSnapshot, reqPoolEntries,
        kvObjectMap, kvObjectLens, candidateSeeds, candidateCounts,
        candidateReceiptIdentity, receiptStatus, receiptOwner);
    return std::make_tuple(candidateSeeds, candidateCounts,
                           candidateReceiptIdentity, receiptStatus, receiptOwner);
}

Result RunMeta(
    const at::Tensor& scoreWorkspace, const at::Tensor& tileReceipts,
    const at::Tensor& receiptIdentity, const at::Tensor& scoreSourceLens,
    const at::Tensor& residentThreshold,
    const at::Tensor& residentThresholdValid,
    const at::Tensor& kvObjectSlotsSnapshot,
    const at::Tensor& reqPoolEntries, const at::Tensor& kvObjectMap,
    const at::Tensor& kvObjectLens)
{
    const Geometry geometry = Validate(
        scoreWorkspace, tileReceipts, receiptIdentity, scoreSourceLens,
        residentThreshold, residentThresholdValid, kvObjectSlotsSnapshot,
        reqPoolEntries, kvObjectMap, kvObjectLens, false);
    const auto int32Options = scoreWorkspace.options().dtype(at::kInt);
    const auto int64Options = scoreWorkspace.options().dtype(at::kLong);
    return std::make_tuple(
        at::empty({geometry.receipts, kTileWidth, kSeedWords}, int32Options),
        at::empty({geometry.receipts}, int32Options),
        at::empty({geometry.receipts, kReceiptIdentityWords}, int64Options),
        at::empty({geometry.receipts}, int32Options),
        at::empty({geometry.receipts}, int32Options));
}
}  // namespace

TORCH_LIBRARY(a5_score_receipt_executor, m)
{
    m.def(
        "score_receipt_candidate_executor("
        "Tensor score_workspace, Tensor tile_receipts, Tensor receipt_identity, "
        "Tensor score_source_lens, Tensor resident_threshold, "
        "Tensor resident_threshold_valid, Tensor kv_object_slots_snapshot, "
        "Tensor req_pool_entries, Tensor kv_object_map, Tensor kv_object_lens) "
        "-> (Tensor candidate_seeds, Tensor candidate_counts, "
        "Tensor candidate_receipt_identity, Tensor receipt_status, "
        "Tensor receipt_owner)");
}

TORCH_LIBRARY_IMPL(a5_score_receipt_executor, PrivateUse1, m)
{
    m.impl("score_receipt_candidate_executor", &RunNpu);
}

TORCH_LIBRARY_IMPL(a5_score_receipt_executor, Meta, m)
{
    m.impl("score_receipt_candidate_executor", &RunMeta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("p7_executor_contract", [] {
        pybind11::dict contract;
        contract["stage"] = 1;
        contract["receipt_words"] = kReceiptWords;
        contract["tile_width"] = kTileWidth;
        contract["seed_words"] = kSeedWords;
        contract["post_cq"] = false;
        contract["static_core_partition"] = false;
        return contract;
    });
}
