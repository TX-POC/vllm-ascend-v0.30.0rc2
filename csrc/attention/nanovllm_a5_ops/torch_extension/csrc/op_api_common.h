#pragma once

// Minimal, self-contained ACLNN launcher for nanovllm_dsa_a5.
//
// Do NOT include torch_npu's OpCommand/OpPreparation helpers here. Those are
// private C++ implementation details whose exported ABI varies across
// torch_npu releases. This adapter only depends on ATen, ACL runtime and the
// public NPU stream wrapper, while resolving ACLNN symbols dynamically from
// the repository-local libcust_opapi.so.

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

namespace nanovllm_a5_opapi {

using AclCreateTensorFn = aclTensor *(*)(
    const int64_t *view_dims,
    uint64_t view_dims_num,
    aclDataType data_type,
    const int64_t *stride,
    int64_t offset,
    aclFormat format,
    const int64_t *storage_dims,
    uint64_t storage_dims_num,
    void *tensor_data);
using AclDestroyTensorFn = int (*)(const aclTensor *tensor);
using OpApiLaunchFn = int (*)(
    void *workspace,
    uint64_t workspace_size,
    aclOpExecutor *executor,
    aclrtStream stream);
using InitHugeMemThreadLocalFn = int (*)(void *, bool);
using UnInitHugeMemThreadLocalFn = void (*)(void *, bool);
using ReleaseHugeMemFn = void (*)(void *, bool);

inline void *OpenLibraryOnce(const char *path) {
  if (path == nullptr || path[0] == '\0') {
    return nullptr;
  }
  return dlopen(path, RTLD_NOW | RTLD_GLOBAL);
}

inline void *RepositoryCustomHandle() {
  static void *handle = []() -> void * {
    const char *path = std::getenv("NANOVLLM_CUST_OPAPI_LIB");
    if (path == nullptr || path[0] == '\0') {
      return nullptr;
    }
    void *h = OpenLibraryOnce(path);
    const char *error = dlerror();
    TORCH_CHECK(
        h != nullptr,
        "cannot load repository-local libcust_opapi.so: ",
        path,
        "; dlerror=",
        error == nullptr ? "<none>" : error);
    return h;
  }();
  return handle;
}

inline void *SystemOpApiHandle() {
  static void *handle = OpenLibraryOnce("libopapi.so");
  return handle;
}

inline void *ResolveSymbol(const char *name) {
  if (void *custom = RepositoryCustomHandle(); custom != nullptr) {
    if (void *addr = dlsym(custom, name); addr != nullptr) {
      return addr;
    }
  }
  if (void *system = SystemOpApiHandle(); system != nullptr) {
    if (void *addr = dlsym(system, name); addr != nullptr) {
      return addr;
    }
  }
  return nullptr;
}

inline aclDataType ToAclDataType(at::ScalarType dtype) {
  switch (dtype) {
    case at::kByte:
      return ACL_UINT8;
    case at::kChar:
      return ACL_INT8;
    case at::kShort:
      return ACL_INT16;
    case at::kInt:
      return ACL_INT32;
    case at::kLong:
      return ACL_INT64;
    case at::kHalf:
      return ACL_FLOAT16;
    case at::kFloat:
      return ACL_FLOAT;
    case at::kDouble:
      return ACL_DOUBLE;
    case at::kBool:
      return ACL_BOOL;
    case at::kBFloat16:
      return ACL_BF16;
    default:
      TORCH_CHECK(
          false,
          "unsupported tensor dtype in nanovllm A5 ACLNN launcher: ",
          dtype);
  }
}

inline aclTensor *ConvertType(const at::Tensor &tensor) {
  if (!tensor.defined()) {
    return nullptr;
  }
  TORCH_CHECK(
      tensor.is_contiguous(),
      "nanovllm A5 ACLNN launcher requires contiguous tensors");
  auto create_tensor =
      reinterpret_cast<AclCreateTensorFn>(ResolveSymbol("aclCreateTensor"));
  TORCH_CHECK(create_tensor != nullptr, "aclCreateTensor is unavailable");

  // Every checked-in custom A5 operator schema uses ND. data_ptr() already
  // points at the logical first element, so ACL offset is zero. Keeping the
  // storage shape equal to the logical shape also avoids any dependency on
  // torch_npu's private NPUStorageImpl ABI.
  std::vector<int64_t> storage_dims(
      tensor.sizes().begin(), tensor.sizes().end());
  return create_tensor(
      tensor.sizes().data(),
      static_cast<uint64_t>(tensor.dim()),
      ToAclDataType(tensor.scalar_type()),
      tensor.strides().data(),
      0,
      ACL_FORMAT_ND,
      storage_dims.data(),
      static_cast<uint64_t>(storage_dims.size()),
      tensor.data_ptr());
}

template <typename T>
inline T ConvertType(T value) {
  return value;
}

inline void Release(aclTensor *tensor) {
  if (tensor == nullptr) {
    return;
  }
  auto destroy_tensor = reinterpret_cast<AclDestroyTensorFn>(
      ResolveSymbol("aclDestroyTensor"));
  if (destroy_tensor != nullptr) {
    (void)destroy_tensor(tensor);
  }
}

template <typename T>
inline void Release(T) {}

template <typename... Ts>
inline auto ConvertTypes(Ts &...args) {
  return std::make_tuple(ConvertType(args)...);
}

template <typename Function, typename Tuple, size_t... I>
inline auto CallTuple(
    Function fn,
    Tuple &args,
    std::index_sequence<I...>) {
  return fn(std::get<I>(args)...);
}

template <typename Function, typename Tuple>
inline auto CallTuple(Function fn, Tuple &args) {
  constexpr size_t kSize = std::tuple_size<Tuple>::value;
  return CallTuple(fn, args, std::make_index_sequence<kSize>{});
}

template <typename Tuple, size_t... I>
inline void ReleaseTuple(Tuple &args, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{(Release(std::get<I>(args)), 0)...};
}

template <typename Tuple>
inline void ReleaseTuple(Tuple &args) {
  constexpr size_t kSize = std::tuple_size<Tuple>::value;
  ReleaseTuple(args, std::make_index_sequence<kSize>{});
}

template <typename Tuple, size_t... I>
inline auto WorkspaceFunction(
    Tuple &args,
    void *address,
    std::index_sequence<I...>) {
  using Fn = int (*)(
      typename std::decay<decltype(std::get<I>(args))>::type...);
  return reinterpret_cast<Fn>(address);
}

template <typename Tuple>
inline auto WorkspaceFunction(Tuple &args, void *address) {
  constexpr size_t kSize = std::tuple_size<Tuple>::value;
  return WorkspaceFunction(
      args, address, std::make_index_sequence<kSize>{});
}

inline const char *RecentAclError() {
  const char *message = aclGetRecentErrMsg();
  return message == nullptr ? "<no ACL error message>" : message;
}

}  // namespace nanovllm_a5_opapi

#define EXEC_NPU_CMD_ORDERED(aclnn_api, tensor_keepalive, ...)                \
  do {                                                                         \
    using namespace nanovllm_a5_opapi;                                         \
    static void *workspace_addr_fn =                                           \
        ResolveSymbol(#aclnn_api "GetWorkspaceSize");                         \
    static void *launch_addr_fn = ResolveSymbol(#aclnn_api);                   \
    TORCH_CHECK(                                                               \
        workspace_addr_fn != nullptr && launch_addr_fn != nullptr,             \
        #aclnn_api,                                                            \
        " or ",                                                               \
        #aclnn_api "GetWorkspaceSize is missing from repository-local ",      \
        "libcust_opapi.so");                                                 \
                                                                               \
    auto init_huge_mem = reinterpret_cast<InitHugeMemThreadLocalFn>(           \
        ResolveSymbol("InitHugeMemThreadLocal"));                             \
    auto uninit_huge_mem = reinterpret_cast<UnInitHugeMemThreadLocalFn>(       \
        ResolveSymbol("UnInitHugeMemThreadLocal"));                           \
    auto release_huge_mem = reinterpret_cast<ReleaseHugeMemFn>(                \
        ResolveSymbol("ReleaseHugeMem"));                                     \
    if (init_huge_mem != nullptr) {                                            \
      (void)init_huge_mem(nullptr, false);                                     \
    }                                                                          \
                                                                               \
    uint64_t workspace_size = 0;                                               \
    uint64_t *workspace_size_ptr = &workspace_size;                            \
    aclOpExecutor *executor = nullptr;                                         \
    aclOpExecutor **executor_ptr = &executor;                                  \
    auto converted = ConvertTypes(                                             \
        __VA_ARGS__, workspace_size_ptr, executor_ptr);                        \
    auto workspace_fn = WorkspaceFunction(converted, workspace_addr_fn);       \
    int workspace_status = CallTuple(workspace_fn, converted);                 \
    TORCH_CHECK(                                                               \
        workspace_status == 0,                                                 \
        "call " #aclnn_api "GetWorkspaceSize failed: ",                     \
        RecentAclError());                                                     \
                                                                               \
    at::Tensor workspace;                                                      \
    void *workspace_ptr = nullptr;                                             \
    if (workspace_size != 0) {                                                 \
      const auto &_workspace_ref_tensor = std::get<0>(tensor_keepalive);       \
      workspace = at::empty(                                                   \
          {static_cast<int64_t>(workspace_size)},                              \
          _workspace_ref_tensor.options().dtype(at::kByte));                   \
      workspace_ptr = workspace.data_ptr();                                    \
    }                                                                          \
                                                                               \
    auto npu_stream = c10_npu::getCurrentNPUStream();                          \
    aclrtStream acl_stream = npu_stream.stream(false);                         \
    auto launch_fn = reinterpret_cast<OpApiLaunchFn>(launch_addr_fn);          \
    int launch_status =                                                        \
        launch_fn(workspace_ptr, workspace_size, executor, acl_stream);        \
    TORCH_CHECK(                                                               \
        launch_status == 0,                                                    \
        "call " #aclnn_api " failed: ", RecentAclError());                  \
                                                                               \
    ReleaseTuple(converted);                                                   \
    if (release_huge_mem != nullptr) {                                         \
      release_huge_mem(nullptr, false);                                        \
    }                                                                          \
    if (uninit_huge_mem != nullptr) {                                          \
      uninit_huge_mem(nullptr, false);                                         \
    }                                                                          \
    (void)workspace;                                                           \
    (void)tensor_keepalive;                                                    \
  } while (false)
