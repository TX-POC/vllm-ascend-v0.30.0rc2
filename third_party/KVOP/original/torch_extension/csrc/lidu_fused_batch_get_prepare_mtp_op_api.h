#pragma once

// ACLNN launcher for packaged and standalone
// A5LiduFusedBatchGetPrepareMtp builds.

#include <cstdlib>
#include <dlfcn.h>

#include <initializer_list>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;

namespace lidu_fused_batch_get_prepare_mtp_opapi {
using AclCreateTensorFn = aclTensor *(*) (
    const int64_t *, uint64_t, aclDataType, const int64_t *, int64_t,
    aclFormat, const int64_t *, uint64_t, void *);
using AclDestroyTensorFn = int (*)(const aclTensor *);
using OpApiLaunchFn = int (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using InitHugeMemThreadLocalFn = int (*)(void *, bool);
using UnInitHugeMemThreadLocalFn = void (*)(void *, bool);
using ReleaseHugeMemFn = void (*)(void *, bool);

inline std::string &CustomOpApiPath()
{
    static std::string path;
    return path;
}

inline void SetCustomOpApiPath(const std::string &path)
{
    TORCH_CHECK(!path.empty(), "A5LiduFusedBatchGetPrepareMtp op-api path is empty");
    std::string &current = CustomOpApiPath();
    TORCH_CHECK(
        current.empty() || current == path,
        "A5LiduFusedBatchGetPrepareMtp was initialized with two different OPP paths: ",
        current, " and ", path);
    current = path;
}

inline void *CustomOpApiHandle()
{
    static void *handle = []() -> void * {
        std::string &path = CustomOpApiPath();
        if (path.empty()) {
            const char *packagedPath = std::getenv("NANOVLLM_CUST_OPAPI_LIB");
            if (packagedPath != nullptr && packagedPath[0] != '\0') {
                path = packagedPath;
            }
        }
        TORCH_CHECK(
            !path.empty(),
            "A5LiduFusedBatchGetPrepareMtp OPP is not initialized. Import ",
            "vllm_ascend.a5_dsa_registration or "
            "nanovllm_dsa_lidu_fused_batch_get_prepare_mtp before calling "
            "this torch op.");
        void *opened = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        const char *error = dlerror();
        TORCH_CHECK(
            opened != nullptr,
            "cannot load A5LiduFusedBatchGetPrepareMtp libcust_opapi.so: ", path,
            "; dlerror=", error == nullptr ? "<none>" : error);
        return opened;
    }();
    return handle;
}

inline void *SystemOpApiHandle()
{
    static void *handle = dlopen("libopapi.so", RTLD_NOW | RTLD_GLOBAL);
    return handle;
}

inline void *ResolveSymbol(const char *name)
{
    if (void *address = dlsym(CustomOpApiHandle(), name); address != nullptr) {
        return address;
    }
    if (void *system = SystemOpApiHandle(); system != nullptr) {
        if (void *address = dlsym(system, name); address != nullptr) {
            return address;
        }
    }
    return nullptr;
}

inline aclDataType ToAclDataType(at::ScalarType dtype)
{
    switch (dtype) {
        case at::kInt:
            return ACL_INT32;
        case at::kLong:
            return ACL_INT64;
        default:
            TORCH_CHECK(
                false,
                "A5LiduFusedBatchGetPrepareMtp supports only int32 state tensors and "
                "int64 profiling, got ", dtype);
    }
}

inline aclTensor *ConvertType(const at::Tensor &tensor)
{
    TORCH_CHECK(tensor.defined() && tensor.is_contiguous(),
        "A5LiduFusedBatchGetPrepareMtp requires contiguous tensors");
    auto createTensor = reinterpret_cast<AclCreateTensorFn>(
        ResolveSymbol("aclCreateTensor"));
    TORCH_CHECK(createTensor != nullptr, "aclCreateTensor is unavailable");
    std::vector<int64_t> storageDims(
        tensor.sizes().begin(), tensor.sizes().end());
    return createTensor(
        tensor.sizes().data(), static_cast<uint64_t>(tensor.dim()),
        ToAclDataType(tensor.scalar_type()), tensor.strides().data(), 0,
        ACL_FORMAT_ND, storageDims.data(),
        static_cast<uint64_t>(storageDims.size()), tensor.data_ptr());
}

template <typename T>
inline T ConvertType(T value)
{
    return value;
}

inline void Release(aclTensor *tensor)
{
    if (tensor == nullptr) {
        return;
    }
    auto destroyTensor = reinterpret_cast<AclDestroyTensorFn>(
        ResolveSymbol("aclDestroyTensor"));
    if (destroyTensor != nullptr) {
        (void)destroyTensor(tensor);
    }
}

template <typename T>
inline void Release(T)
{}

template <typename... Ts>
inline auto ConvertTypes(Ts &...args)
{
    return std::make_tuple(ConvertType(args)...);
}

template <typename Function, typename Tuple, size_t... I>
inline auto CallTuple(Function fn, Tuple &args, std::index_sequence<I...>)
{
    return fn(std::get<I>(args)...);
}

template <typename Function, typename Tuple>
inline auto CallTuple(Function fn, Tuple &args)
{
    return CallTuple(
        fn, args,
        std::make_index_sequence<std::tuple_size<Tuple>::value>{});
}

template <typename Tuple, size_t... I>
inline void ReleaseTuple(Tuple &args, std::index_sequence<I...>)
{
    (void)std::initializer_list<int>{(Release(std::get<I>(args)), 0)...};
}

template <typename Tuple>
inline void ReleaseTuple(Tuple &args)
{
    ReleaseTuple(args, std::make_index_sequence<std::tuple_size<Tuple>::value>{});
}

template <typename Tuple, size_t... I>
inline auto WorkspaceFunction(
    Tuple &args, void *address, std::index_sequence<I...>)
{
    using Function = int (*) (
        typename std::decay<decltype(std::get<I>(args))>::type...);
    return reinterpret_cast<Function>(address);
}

template <typename Tuple>
inline auto WorkspaceFunction(Tuple &args, void *address)
{
    return WorkspaceFunction(
        args, address,
        std::make_index_sequence<std::tuple_size<Tuple>::value>{});
}

inline const char *RecentAclError()
{
    const char *message = aclGetRecentErrMsg();
    return message == nullptr ? "<no ACL error message>" : message;
}
}  // namespace lidu_fused_batch_get_prepare_mtp_opapi

#define EXEC_LIDU_FUSED_BATCH_GET_PREPARE_MTP_OP(                           \
    aclnn_api, tensor_keepalive, ...)                                        \
  do {                                                                        \
    using namespace lidu_fused_batch_get_prepare_mtp_opapi;                               \
    static void *workspaceAddrFn = ResolveSymbol(                             \
        #aclnn_api "GetWorkspaceSize");                                      \
    static void *launchAddrFn = ResolveSymbol(#aclnn_api);                    \
    TORCH_CHECK(                                                              \
        workspaceAddrFn != nullptr && launchAddrFn != nullptr,                \
        #aclnn_api " or " #aclnn_api "GetWorkspaceSize is missing from ",  \
        "A5LiduFusedBatchGetPrepareMtp libcust_opapi.so");                             \
    auto initHugeMem = reinterpret_cast<InitHugeMemThreadLocalFn>(            \
        ResolveSymbol("InitHugeMemThreadLocal"));                            \
    auto uninitHugeMem = reinterpret_cast<UnInitHugeMemThreadLocalFn>(        \
        ResolveSymbol("UnInitHugeMemThreadLocal"));                          \
    auto releaseHugeMem = reinterpret_cast<ReleaseHugeMemFn>(                 \
        ResolveSymbol("ReleaseHugeMem"));                                   \
    if (initHugeMem != nullptr) {                                             \
      (void)initHugeMem(nullptr, false);                                      \
    }                                                                         \
    uint64_t workspaceSize = 0;                                               \
    uint64_t *workspaceSizePtr = &workspaceSize;                              \
    aclOpExecutor *executor = nullptr;                                        \
    aclOpExecutor **executorPtr = &executor;                                  \
    auto converted = ConvertTypes(                                            \
        __VA_ARGS__, workspaceSizePtr, executorPtr);                          \
    auto workspaceFn = WorkspaceFunction(converted, workspaceAddrFn);         \
    const int workspaceStatus = CallTuple(workspaceFn, converted);            \
    TORCH_CHECK(                                                              \
        workspaceStatus == 0, "call " #aclnn_api                             \
        "GetWorkspaceSize failed: ", RecentAclError());                      \
    at::Tensor workspace;                                                     \
    void *workspacePtr = nullptr;                                             \
    if (workspaceSize != 0) {                                                 \
      const auto &reference = std::get<0>(tensor_keepalive);                  \
      workspace = at::empty(                                                 \
          {static_cast<int64_t>(workspaceSize)},                              \
          reference.options().dtype(at::kByte));                              \
      workspacePtr = workspace.data_ptr();                                   \
    }                                                                         \
    auto npuStream = c10_npu::getCurrentNPUStream();                          \
    /* Submit queued Torch inputs before the direct ACLNN launch. */         \
    /* This drains host submissions, not device execution. */                \
    auto aclStream = npuStream.stream();                                     \
    const int launchStatus = reinterpret_cast<OpApiLaunchFn>(launchAddrFn)(   \
        workspacePtr, workspaceSize, executor, aclStream);                    \
    TORCH_CHECK(launchStatus == 0, "call " #aclnn_api " failed: ",          \
        RecentAclError());                                                    \
    ReleaseTuple(converted);                                                  \
    if (releaseHugeMem != nullptr) {                                          \
      releaseHugeMem(nullptr, false);                                         \
    }                                                                         \
    if (uninitHugeMem != nullptr) {                                           \
      uninitHugeMem(nullptr, false);                                          \
    }                                                                         \
    (void)workspace;                                                          \
    (void)tensor_keepalive;                                                   \
  } while (false)
