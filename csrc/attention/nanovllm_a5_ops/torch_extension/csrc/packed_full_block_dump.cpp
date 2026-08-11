#include <tuple>

#include <torch/extension.h>
#include <torch/library.h>

#include "op_api_common.h"

namespace {
constexpr int64_t kBlockSize = 128;
constexpr int64_t kPackedKvDim = 656;

void CheckPackedCache(
    const at::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(
      tensor.dim() == 4 && tensor.size(0) > 0 &&
          tensor.size(1) == kBlockSize && tensor.size(2) == 1 &&
          tensor.size(3) == kPackedKvDim,
      name, " must be [blocks,128,1,656].");
  TORCH_CHECK(
      tensor.element_size() == 1,
      name, " must use one-byte storage.");
}

void CheckPackedFullBlockDumpInputs(
    const at::Tensor& hbm_packed_kv,
    const at::Tensor& dram_packed_kv,
    const at::Tensor& src_block_ids,
    const at::Tensor& dst_block_ids) {
  CheckPackedCache(hbm_packed_kv, "packed_full_block_dump HBM KV");
  CheckPackedCache(dram_packed_kv, "packed_full_block_dump DRAM KV");
  TORCH_CHECK(
      hbm_packed_kv.scalar_type() == at::kChar &&
          dram_packed_kv.scalar_type() == at::kChar,
      "packed_full_block_dump byte views must be int8.");
  TORCH_CHECK(
      src_block_ids.dim() == 1 && dst_block_ids.dim() == 1 &&
          src_block_ids.numel() > 0 &&
          src_block_ids.sizes() == dst_block_ids.sizes(),
      "packed_full_block_dump ids must be matching non-empty int32[J].");
  TORCH_CHECK(
      src_block_ids.scalar_type() == at::kInt &&
          dst_block_ids.scalar_type() == at::kInt,
      "packed_full_block_dump ids must be int32.");
  TORCH_CHECK(
      hbm_packed_kv.device().is_privateuseone(),
      "packed_full_block_dump tensors must be on NPU.");
  for (const at::Tensor* tensor : {
           &hbm_packed_kv,
           &dram_packed_kv,
           &src_block_ids,
           &dst_block_ids}) {
    TORCH_CHECK(
        tensor->device() == hbm_packed_kv.device() &&
            tensor->is_contiguous(),
        "packed_full_block_dump inputs must be contiguous on one NPU.");
  }
}

at::Tensor PackedFullBlockDumpNpu(
    const at::Tensor& hbm_packed_kv,
    at::Tensor dram_packed_kv,
    const at::Tensor& src_block_ids,
    const at::Tensor& dst_block_ids) {
  CheckPackedFullBlockDumpInputs(
      hbm_packed_kv,
      dram_packed_kv,
      src_block_ids,
      dst_block_ids);
  auto keepalive = std::make_tuple(
      hbm_packed_kv,
      dram_packed_kv,
      src_block_ids,
      dst_block_ids);
  EXEC_NPU_CMD_ORDERED(
      aclnnA5PackedFullBlockDump,
      keepalive,
      hbm_packed_kv,
      dram_packed_kv,
      src_block_ids,
      dst_block_ids,
      dram_packed_kv);
  return dram_packed_kv;
}

at::Tensor PackedFullBlockDumpMeta(
    const at::Tensor&,
    at::Tensor dram_packed_kv,
    const at::Tensor&,
    const at::Tensor&) {
  return dram_packed_kv;
}
} // namespace

TORCH_LIBRARY_FRAGMENT(nanovllm_dsa, m) {
  m.def(
      "packed_full_block_dump(Tensor hbm_packed_kv, "
      "Tensor(a!) dram_packed_kv, Tensor src_block_ids, "
      "Tensor dst_block_ids) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, PrivateUse1, m) {
  m.impl("packed_full_block_dump", &PackedFullBlockDumpNpu);
}

TORCH_LIBRARY_IMPL(nanovllm_dsa, Meta, m) {
  m.impl("packed_full_block_dump", &PackedFullBlockDumpMeta);
}
