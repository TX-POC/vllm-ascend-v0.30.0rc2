// Minimal prototype-only ACLNN bridge.  It owns a namespace separate from
// nanovllm_dsa, so it can be built and loaded without recompiling or
// conflicting with vllm_ascend_C.

#include <tuple>

#include <torch/extension.h>
#include <torch/library.h>

#include "op_api_common.h"

namespace {
constexpr int64_t kLiAivCount = 62;
constexpr int64_t kPostAivCount = 2;
constexpr int64_t kRoleTraceCount = 96;

void CheckInputs(
    const at::Tensor& scanValues,
    const at::Tensor& postSeed)
{
    TORCH_CHECK(
        scanValues.device().is_privateuseone() &&
            postSeed.device() == scanValues.device(),
        "exact_scan_speculative_post_probe inputs must be on one NPU.");
    TORCH_CHECK(
        scanValues.is_contiguous() && postSeed.is_contiguous(),
        "exact_scan_speculative_post_probe inputs must be contiguous.");
    TORCH_CHECK(
        scanValues.scalar_type() == at::kInt &&
            postSeed.scalar_type() == at::kInt,
        "exact_scan_speculative_post_probe inputs must be int32.");
    TORCH_CHECK(
        scanValues.dim() == 1 && scanValues.numel() >= kLiAivCount,
        "scan_values must be int32[N] with N >= 62.");
    TORCH_CHECK(
        postSeed.dim() == 1 && postSeed.numel() == kPostAivCount,
        "post_seed must be int32[2], one value per Post AIV lane.");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ProbeNpu(
    const at::Tensor& scanValues,
    const at::Tensor& postSeed)
{
    CheckInputs(scanValues, postSeed);
    const auto int32Options = scanValues.options().dtype(at::kInt);
    const auto int64Options = scanValues.options().dtype(at::kLong);
    auto liChecksums = at::empty({kLiAivCount}, int64Options);
    auto postMarkers = at::empty({kPostAivCount}, int32Options);
    auto roleTrace = at::empty({kRoleTraceCount}, int32Options);
    auto keepalive = std::make_tuple(
        scanValues, postSeed, liChecksums, postMarkers, roleTrace);
    EXEC_NPU_CMD_ORDERED(
        aclnnA5ExactScanSpeculativePostProbe,
        keepalive,
        scanValues,
        postSeed,
        liChecksums,
        postMarkers,
        roleTrace);
    return std::make_tuple(liChecksums, postMarkers, roleTrace);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ProbeMeta(
    const at::Tensor& scanValues,
    const at::Tensor& postSeed)
{
    // Meta tensors do not have a PrivateUse1 device.  Keep the shape contract
    // available to graph/fake execution without applying the NPU-only checks
    // used by the real launcher above.
    TORCH_CHECK(
        scanValues.dim() == 1 && scanValues.numel() >= kLiAivCount &&
            postSeed.dim() == 1 && postSeed.numel() == kPostAivCount,
        "exact_scan_speculative_post_probe meta shapes are invalid.");
    const auto int32Options = scanValues.options().dtype(at::kInt);
    const auto int64Options = scanValues.options().dtype(at::kLong);
    return std::make_tuple(
        at::empty({kLiAivCount}, int64Options),
        at::empty({kPostAivCount}, int32Options),
        at::empty({kRoleTraceCount}, int32Options));
}
}  // namespace

TORCH_LIBRARY(a5_exact_scan_probe, m)
{
    m.def(
        "exact_scan_speculative_post_probe(Tensor scan_values, "
        "Tensor post_seed) -> (Tensor li_checksums, Tensor post_markers, "
        "Tensor role_trace)");
}

TORCH_LIBRARY_IMPL(a5_exact_scan_probe, PrivateUse1, m)
{
    m.impl("exact_scan_speculative_post_probe", &ProbeNpu);
}

TORCH_LIBRARY_IMPL(a5_exact_scan_probe, Meta, m)
{
    m.impl("exact_scan_speculative_post_probe", &ProbeMeta);
}

// The TORCH_LIBRARY blocks register schemas during shared-library loading,
// but Python still requires this extension-module entry point before it can
// load ``nanovllm_exact_scan_probe._C``.  It intentionally exports no Python
// methods: importing it is sufficient to run the Torch registrations above.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
