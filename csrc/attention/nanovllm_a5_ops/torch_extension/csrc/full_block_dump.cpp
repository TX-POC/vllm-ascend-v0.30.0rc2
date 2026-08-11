#include <tuple>

#include <torch/extension.h>
#include <torch/library.h>

#include "op_api_common.h"

namespace {
constexpr int64_t kBlockSize = 128;
constexpr int64_t kKpeDim = 64;
constexpr int64_t kCkvDim = 512;

void CheckCacheShape(
    const at::Tensor& tensor,
    int64_t dim,
    const char* name) {
  TORCH_CHECK(
      tensor.dim() == 3 && tensor.size(0) > 0 &&
          tensor.size(1) == kBlockSize && tensor.size(2) == dim,
      name, " must be [blocks,128,", dim, "].");
}

void CheckFullBlockDumpInputs(
    const at::Tensor& hbm_kpe,
    const at::Tensor& hbm_ckv,
    const at::Tensor& dram_kpe,
    const at::Tensor& dram_ckv,
    const at::Tensor& src_block_ids,
    const at::Tensor& dst_block_ids) {
  CheckCacheShape(hbm_kpe, kKpeDim, "full_block_dump HBM KPE");
  CheckCacheShape(hbm_ckv, kCkvDim, "full_block_dump HBM CKV");
  CheckCacheShape(dram_kpe, kKpeDim, "full_block_dump DRAM KPE");
  CheckCacheShape(dram_ckv, kCkvDim, "full_block_dump DRAM CKV");
  TORCH_CHECK(
      hbm_kpe.size(0) == hbm_ckv.size(0),
      "full_block_dump HBM CKV/KPE block counts must match.");
  TORCH_CHECK(
      dram_kpe.size(0) == dram_ckv.size(0),
      "full_block_dump DRAM CKV/KPE block counts must match.");

  const auto dtype = hbm_kpe.scalar_type();
  TORCH_CHECK(
      dtype == at::kBFloat16 || dtype == at::kHalf,
      "full_block_dump supports bf16/fp16 only.");
  for (const at::Tensor* tensor : {&hbm_ckv, &dram_kpe, &dram_ckv}) {
    TORCH_CHECK(
        tensor->scalar_type() == dtype,
        "full_block_dump cache dtypes must match.");
  }

  TORCH_CHECK(
      src_block_ids.dim() == 1 && dst_block_ids.dim() == 1 &&
          src_block_ids.numel() > 0 &&
          src_block_ids.sizes() == dst_block_ids.sizes(),
      "full_block_dump block ids must be matching non-empty int32[J].");
  TORCH_CHECK(
      src_block_ids.scalar_type() == at::kInt &&
          dst_block_ids.scalar_type() == at::kInt,
      "full_block_dump block ids must be int32.");

  TORCH_CHECK(
      hbm_kpe.device().is_privateuseone(),
      "full_block_dump tensors must be on NPU.");
  for (const at::Tensor* tensor :
       {&hbm_kpe, &hbm_ckv, &dram_kpe, &dram_ckv,
        &src_block_ids, &dst_block_ids}) {
    TORCH_CHECK(
        tensor->device() == hbm_kpe.device() && tensor->is_contiguous(),
        "full_block_dump inputs must be contiguous tensors on one NPU.");
  }
}

std::tuple<at::Tensor, at::Tensor> FullBlockDumpNpu(
    const at::Tensor& hbm_kpe,
    const at::Tensor& hbm_ckv,
    at::Tensor dram_kpe,
    at::Tensor dram_ckv,
    const at::Tensor& src_block_ids,
    const at::Tensor& dst_block_ids) {
  CheckFullBlockDumpInputs(
      hbm_kpe, hbm_ckv, dram_kpe, dram_ckv,
      src_block_ids, dst_block_ids);
  auto keepalive = std::make_tuple(
      hbm_kpe, hbm_ckv, dram_kpe, dram_ckv,
      src_block_ids, dst_block_ids);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5FullBlockDump,
      keepalive,
      hbm_kpe,
      hbm_ckv,
      dram_kpe,
      dram_ckv,
      src_block_ids,
      dst_block_ids,
      dram_kpe,
      dram_ckv);
  return std::make_tuple(dram_kpe, dram_ckv);
}

std::tuple<at::Tensor, at::Tensor> FullBlockDumpMeta(
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor dram_kpe,
    at::Tensor dram_ckv,
    const at::Tensor&,
    const at::Tensor&) {
  return std::make_tuple(dram_kpe, dram_ckv);
}
} // namespace

TORCH_LIBRARY_FRAGMENT(nanovllm_dsa, m) {
  m.def(
      "full_block_dump(Tensor hbm_kpe, Tensor hbm_ckv, "
      "Tensor(a!) dram_kpe, Tensor(b!) dram_ckv, "
      "Tensor src_block_ids, Tensor dst_block_ids) "
      "-> (Tensor(a!), Tensor(b!))");
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl("full_block_dump", &FullBlockDumpNpu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl("full_block_dump", &FullBlockDumpMeta);
}
