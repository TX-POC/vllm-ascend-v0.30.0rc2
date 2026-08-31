#pragma once

// Minimal, self-contained ACLNN launcher for nanovllm_dsa_a5.
//
// Most bridges below use torch_npu's OpCommand task-queue contract.  That is
// the existing contract for their independently validated launch paths.  P7.1
// is deliberately different: its CANN ABI has zero workspace and its actual
// PostOnly/CQ asynchronous work is already device-side, so it uses the same
// direct-current-stream ACLNN form as the repository's original standalone
// baseline.  Keep that exception local to P7.1 rather than changing the
// ownership rule for every other bridge.

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

// Each standalone validation bridge owns a distinct resolver domain.  The
// production/P3 bridge keeps the historical default, while the source-matched
// C8 reference defines this macro before including the header.  Do not share
// one process-global environment slot: the two bridges intentionally load
// different OPP DSOs in the same equivalence process.
#ifndef NANOVLLM_A5_OPAPI_ENV
#define NANOVLLM_A5_OPAPI_ENV "NANOVLLM_CUST_OPAPI_LIB"
#endif

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;

namespace nanovllm_a5_opapi {

// ``ACL_FLOAT8_E4M3FN`` is an aclDataType *enumerator*, not a preprocessor
// macro.  Testing it with ``#ifdef`` therefore always evaluates false and
// used to rewrite a valid CANN 9.1 FP8 enum to ACL_DT_UNDEFINED.  That made
// the first official C8 ACLNN call fail before tiling.  A5 is intentionally
// built against CANN 9.1, where this enumerator is a required public ABI.
inline constexpr aclDataType kAclFloat8E4M3Fn = ACL_FLOAT8_E4M3FN;
static_assert(kAclFloat8E4M3Fn != ACL_DT_UNDEFINED,
              "CANN must expose ACL_FLOAT8_E4M3FN for A5 C8 LI.");

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
using InitPtaCacheThreadLocalFn = void (*)();
using SetPtaHashKeyFn = void (*)(uint64_t);
using UnInitPtaCacheThreadLocalFn = void (*)();
using AclUseStreamResFn = aclError (*)(aclrtStream);

inline void *OpenLibraryOnce(const char *path) {
  if (path == nullptr || path[0] == '\0') {
    return nullptr;
  }
  // Keep a custom OPP in its own dynamic-loader scope.  The checked-in CANN
  // adapter uses the default local scope (dlopen(..., RTLD_LAZY)); exporting
  // every generated custom OPP globally is both unnecessary for dlsym(handle,
  // ...) and unsafe when several custom packages coexist in one Python
  // process.  In particular, RTLD_GLOBAL can interpose helper symbols used by
  // a subsequently loaded OPP.  RTLD_NOW retains the earlier fail-fast
  // dependency check, while RTLD_LOCAL restores the important isolation.
  return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}

inline void *RepositoryCustomHandle() {
  static void *handle = []() -> void * {
    const char *path = std::getenv(NANOVLLM_A5_OPAPI_ENV);
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

inline bool OpApiDebugEnabled() {
  const char *value = std::getenv("NANOVLLM_A5_OPAPI_DEBUG");
  return value != nullptr && value[0] != '\0' &&
      !(value[0] == '0' && value[1] == '\0');
}

inline const char *DsoNameForAddress(void *address) {
  Dl_info info{};
  return address != nullptr && dladdr(address, &info) != 0 &&
          info.dli_fname != nullptr
      ? info.dli_fname
      : "<unknown>";
}

// aclTensor descriptors must be destroyed by the same resolver domain that
// created them.  Resolving create and destroy independently permits an
// accidental half-match if a custom OPP exports only one of the two symbols
// while another loaded provider exports the other.  It is a small but real
// source of allocator corruption after an otherwise successful ACLNN call.
// Resolve an all-or-nothing pair once from a single handle instead.
struct AclTensorApiPair {
  AclCreateTensorFn create = nullptr;
  AclDestroyTensorFn destroy = nullptr;
  const char *provider = "<unresolved>";
};

inline const AclTensorApiPair &GetAclTensorApiPair() {
  static const AclTensorApiPair pair = []() {
    const struct Provider {
      const char *name;
      void *handle;
    } providers[] = {
        {"repository_custom", RepositoryCustomHandle()},
        {"system_opapi", SystemOpApiHandle()},
    };
    for (const Provider &provider : providers) {
      if (provider.handle == nullptr) {
        continue;
      }
      auto create = reinterpret_cast<AclCreateTensorFn>(
          dlsym(provider.handle, "aclCreateTensor"));
      auto destroy = reinterpret_cast<AclDestroyTensorFn>(
          dlsym(provider.handle, "aclDestroyTensor"));
      if (create != nullptr && destroy != nullptr) {
        if (OpApiDebugEnabled()) {
          std::fprintf(
              stderr,
              "P7_1_ACL_DSO role=acl_tensor_pair provider=%s "
              "loader=RTLD_NOW|RTLD_LOCAL create_dso=%s destroy_dso=%s "
              "create_addr=%p destroy_addr=%p\n",
              provider.name,
              DsoNameForAddress(reinterpret_cast<void *>(create)),
              DsoNameForAddress(reinterpret_cast<void *>(destroy)),
              reinterpret_cast<void *>(create),
              reinterpret_cast<void *>(destroy));
          std::fflush(stderr);
        }
        return AclTensorApiPair{create, destroy, provider.name};
      }
    }
    return AclTensorApiPair{};
  }();
  TORCH_CHECK(
      pair.create != nullptr && pair.destroy != nullptr,
      "aclCreateTensor/aclDestroyTensor are unavailable from one resolver "
      "provider; refusing a cross-DSO descriptor lifecycle");
  return pair;
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
    case at::kFloat8_e4m3fn:
      return kAclFloat8E4M3Fn;
    default:
      TORCH_CHECK(
          false,
          "unsupported tensor dtype in nanovllm A5 ACLNN launcher: ",
          dtype);
  }
}

struct AclTensorArgument {
  const at::Tensor &tensor;
  bool require_contiguous;
};

inline AclTensorArgument AllowStridedAclTensor(const at::Tensor &tensor) {
  return {tensor, false};
}

inline aclTensor *ConvertTensor(
    const at::Tensor &tensor, bool require_contiguous) {
  if (!tensor.defined()) {
    return nullptr;
  }
  if (require_contiguous) {
    TORCH_CHECK(
        tensor.is_contiguous(),
        "nanovllm A5 ACLNN launcher requires contiguous tensors");
  }
  const auto &acl_tensor_api = GetAclTensorApiPair();

  // Every checked-in custom A5 operator schema uses ND. data_ptr() already
  // points at the logical first element, so ACL offset is zero. Keeping the
  // storage shape equal to the logical shape also avoids any dependency on
  // torch_npu's private NPUStorageImpl ABI.
  const aclDataType acl_dtype = ToAclDataType(tensor.scalar_type());
  TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED,
              "NPU ACL tensor dtype resolved to ACL_DT_UNDEFINED: ",
              tensor.scalar_type());
  if (OpApiDebugEnabled() && tensor.scalar_type() == at::kFloat8_e4m3fn) {
    std::fprintf(stderr,
                 "NANOVLLM_A5_ACL_TENSOR aten_dtype=%s acl_dtype=%d "
                 "shape_rank=%zu data=%p\n",
                 c10::toString(tensor.scalar_type()),
                 static_cast<int>(acl_dtype),
                 static_cast<size_t>(tensor.dim()), tensor.data_ptr());
    std::fflush(stderr);
  }
  std::vector<int64_t> storage_dims(
      tensor.sizes().begin(), tensor.sizes().end());
  return acl_tensor_api.create(
      tensor.sizes().data(),
      static_cast<uint64_t>(tensor.dim()),
      acl_dtype,
      tensor.strides().data(),
      0,
      ACL_FORMAT_ND,
      storage_dims.data(),
      static_cast<uint64_t>(storage_dims.size()),
      tensor.data_ptr());
}

inline aclTensor *ConvertType(const at::Tensor &tensor) {
  return ConvertTensor(tensor, true);
}

inline aclTensor *ConvertType(const AclTensorArgument &argument) {
  return ConvertTensor(argument.tensor, argument.require_contiguous);
}

template <typename T>
inline T ConvertType(T value) {
  return value;
}

inline void Release(aclTensor *tensor) {
  if (tensor == nullptr) {
    return;
  }
  (void)GetAclTensorApiPair().destroy(tensor);
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

// P7.1 is the only bridge that launches ACLNN directly on the caller's
// current stream.  CANN's generated examples retain aclTensor descriptors
// until the stream is terminal, so a direct bridge must not destroy its 31
// descriptors merely because aclnn(...) has returned to the host.
//
// This small registry is deliberately *not* a second execution queue: the
// kernel is already enqueued on the NPU stream.  It only owns host-side
// descriptors and Tensor handles until the graph Runtime calls
// ReleaseP7DirectLaunchLeasesAtForwardTerminal(token) after the matching
// forward's final stream terminal.  `token` is created at direct launch and
// is bound to the exact aclrtStream that owns the queued ACLNN work.  It is
// deliberately not a host-thread identity: graph terminal callbacks may run
// on a different runtime thread from the submitter.
//
// The bounded reservation fails closed if that terminal is not being
// consumed, rather than allowing unbounded host-lifetime leakage.
class P7DirectLaunchLeaseBase {
 public:
  virtual ~P7DirectLaunchLeaseBase() = default;
};

template <typename ConvertedTuple, typename TensorKeepaliveTuple>
class P7DirectLaunchLease final : public P7DirectLaunchLeaseBase {
 public:
  P7DirectLaunchLease(ConvertedTuple converted,
                      TensorKeepaliveTuple tensor_keepalive)
      : converted_(std::move(converted)),
        tensor_keepalive_(std::move(tensor_keepalive)) {}

  ~P7DirectLaunchLease() override { ReleaseTuple(converted_); }

 private:
  ConvertedTuple converted_;
  // The tuple keeps every input, including the 13 read-only tensors that are
  // not represented by a returned Ref alias, alive through stream terminal.
  TensorKeepaliveTuple tensor_keepalive_;
};

class P7DirectLaunchLeaseRegistry {
 private:
  struct PendingLease {
    uint64_t token;
    uintptr_t stream_key;
    std::shared_ptr<P7DirectLaunchLeaseBase> lease;
  };

 public:
  static constexpr size_t kMaxPendingLeases = 16;

  P7DirectLaunchLeaseRegistry() { pending_.reserve(kMaxPendingLeases); }

  bool Reserve() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (pending_.size() + reservations_ >= kMaxPendingLeases) {
      return false;
    }
    ++reservations_;
    return true;
  }

  void CancelReservation() {
    std::lock_guard<std::mutex> guard(mutex_);
    TORCH_CHECK(reservations_ > 0,
                "P7 direct launch lease reservation underflow");
    --reservations_;
  }

  uint64_t Commit(std::shared_ptr<P7DirectLaunchLeaseBase> lease,
                  aclrtStream stream) {
    TORCH_CHECK(lease != nullptr, "P7 direct launch lease is null");
    TORCH_CHECK(stream != nullptr,
                "P7 direct launch lease requires a non-null ACL stream");
    std::lock_guard<std::mutex> guard(mutex_);
    TORCH_CHECK(reservations_ > 0,
                "P7 direct launch lease commit without reservation");
    TORCH_CHECK(next_token_ != 0,
                "P7 direct launch token space exhausted");
    const uint64_t token = next_token_++;
    --reservations_;
    pending_.push_back(
        {token, reinterpret_cast<uintptr_t>(stream), std::move(lease)});
    return token;
  }

  int64_t ReleaseAtForwardTerminal(uint64_t terminal_token) {
    std::vector<std::shared_ptr<P7DirectLaunchLeaseBase>> completed;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      TORCH_CHECK(terminal_token != 0,
                  "P7 forward-terminal token must be non-zero");
      const auto terminal = std::find_if(
          pending_.begin(), pending_.end(),
          [terminal_token](const PendingLease& pending) {
            return pending.token == terminal_token;
          });
      TORCH_CHECK(
          terminal != pending_.end(),
          "P7 forward-terminal token is not pending; it was either already "
          "retired or belongs to a different process: ",
          terminal_token);
      const uintptr_t stream_key = terminal->stream_key;
      std::vector<PendingLease> remaining;
      remaining.reserve(pending_.size());
      for (auto& pending : pending_) {
        // A terminal for this stream proves every earlier launch on the same
        // stream is no longer read by the device.  It says nothing about a
        // concurrent stream, even if that stream was submitted by the same
        // host thread, so those descriptors remain leased.
        if (pending.stream_key == stream_key &&
            pending.token <= terminal_token) {
          completed.push_back(std::move(pending.lease));
        } else {
          remaining.push_back(std::move(pending));
        }
      }
      pending_.swap(remaining);
    }
    const int64_t released = static_cast<int64_t>(completed.size());
    // Destruction releases aclTensor descriptors and Tensor keepalives only
    // after the caller has established the forward stream terminal.
    completed.clear();
    return released;
  }

  int64_t PendingCount() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return static_cast<int64_t>(pending_.size());
  }

 private:
  mutable std::mutex mutex_;
  size_t reservations_ = 0;
  uint64_t next_token_ = 1;
  std::vector<PendingLease> pending_;
};

inline P7DirectLaunchLeaseRegistry &GetP7DirectLaunchLeaseRegistry() {
  static P7DirectLaunchLeaseRegistry registry;
  return registry;
}

template <typename ConvertedTuple, typename TensorKeepaliveTuple>
inline std::shared_ptr<P7DirectLaunchLeaseBase> MakeP7DirectLaunchLease(
    ConvertedTuple converted, TensorKeepaliveTuple tensor_keepalive) {
  return std::make_shared<
      P7DirectLaunchLease<ConvertedTuple, TensorKeepaliveTuple>>(
      std::move(converted), std::move(tensor_keepalive));
}

// The dispatch bridge consumes this immediately after one P7 direct launch
// and stores it in the forward's RuntimeResources.  It is thread-local only
// for hand-off from the synchronous dispatch return to that RuntimeResources
// object; terminal release itself is intentionally allowed from any thread.
inline uint64_t& LastP7DirectLaunchTokenForCurrentThread() {
  static thread_local uint64_t token = 0;
  return token;
}

inline void ClearP7DirectLaunchTokenForCurrentThread() {
  LastP7DirectLaunchTokenForCurrentThread() = 0;
}

inline void PublishP7DirectLaunchTokenForCurrentThread(uint64_t token) {
  TORCH_CHECK(token != 0, "P7 direct launch token must be non-zero");
  LastP7DirectLaunchTokenForCurrentThread() = token;
}

inline uint64_t TakeP7DirectLaunchTokenForCurrentThread() {
  uint64_t& token = LastP7DirectLaunchTokenForCurrentThread();
  const uint64_t result = token;
  token = 0;
  return result;
}

inline int64_t ReleaseP7DirectLaunchLeasesAtForwardTerminal(
    uint64_t terminal_token) {
  return GetP7DirectLaunchLeaseRegistry().ReleaseAtForwardTerminal(
      terminal_token);
}

inline int64_t PendingP7DirectLaunchLeases() {
  return GetP7DirectLaunchLeaseRegistry().PendingCount();
}

inline void CollectAclTensorDescriptor(
    aclTensor *descriptor, std::vector<void *> *descriptors) {
  if (descriptor != nullptr) {
    descriptors->push_back(descriptor);
  }
}

template <typename T>
inline void CollectAclTensorDescriptor(T, std::vector<void *> *) {}

template <typename Tuple, size_t... I>
inline void LogConvertedDescriptorSetImpl(
    const char *api, Tuple &args, std::index_sequence<I...>) {
  if (!OpApiDebugEnabled()) {
    return;
  }
  std::vector<void *> descriptors;
  descriptors.reserve(sizeof...(I));
  (void)std::initializer_list<int>{
      (CollectAclTensorDescriptor(std::get<I>(args), &descriptors), 0)...};
  std::unordered_set<void *> unique(descriptors.begin(), descriptors.end());
  std::fprintf(
      stderr,
      "P7_1_DESCRIPTOR_LEDGER api=%s tuple_args=%zu acl_tensors=%zu unique_acl_tensors=%zu\n",
      api,
      sizeof...(I),
      descriptors.size(),
      unique.size());
  std::fflush(stderr);
}

// The launcher also carries two non-descriptor trailing pointers (workspace
// size and executor).  Keeping the full tuple count separate from the ACL
// tensor count makes a malformed 31-in/18-out P7 bridge immediately visible
// without dumping 49 host addresses into an uncopyable log.
template <typename Tuple>
inline void LogConvertedDescriptorSet(const char *api, Tuple &args) {
  constexpr size_t kSize = std::tuple_size<Tuple>::value;
  LogConvertedDescriptorSetImpl(
      api, args, std::make_index_sequence<kSize>{});
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

inline AclUseStreamResFn GetUseStreamResFn() {
  static void *handle = OpenLibraryOnce("libascendcl.so");
  static auto use_stream_res =
      handle == nullptr
          ? nullptr
          : reinterpret_cast<AclUseStreamResFn>(
                dlsym(handle, "aclrtUseStreamResInCurrentThread"));
  return use_stream_res;
}

inline void UseStreamResIfNeeded(aclrtStream stream, bool enqueue) {
  const bool needed = enqueue ? c10_npu::check_enqueue_need_use(stream)
                              : c10_npu::check_dequeue_need_use(stream);
  if (!needed) {
    return;
  }
  auto use_stream_res = GetUseStreamResFn();
  TORCH_CHECK(
      use_stream_res != nullptr,
      "aclrtUseStreamResInCurrentThread is unavailable for ordered ACLNN launch");
  const aclError status = use_stream_res(stream);
  TORCH_CHECK(
      status == ACL_SUCCESS,
      "aclrtUseStreamResInCurrentThread failed: ",
      status,
      "; ",
      RecentAclError());
}

inline void LogOpApiLifecycle(const char *api, const char *stage) {
  if (!OpApiDebugEnabled()) {
    return;
  }
  std::fprintf(stderr, "NANOVLLM_A5_OPAPI_LIFECYCLE api=%s stage=%s\n", api, stage);
  std::fflush(stderr);
}

// P7.1 has a process-exit heap abort even when its device kernel rejects an
// invalid control block before it initializes UB or touches Direct-HCCP.  The
// normal production path below must keep its complete cleanup contract.  For
// that one bounded diagnostic only, allow the P7 runner to enable four
// *cumulative* cleanup checkpoints in separate Python processes:
//
//   phase2_noop        launch phase 2, then no explicit host-side release;
//   release_converted  additionally destroy converted aclTensor descriptors;
//   release_huge_mem   additionally call ReleaseHugeMem;
//   uninit_huge_mem    additionally call UnInitHugeMemThreadLocal.
//
// `GetWorkspaceSize` is not a valid boundary without the host TLS allocator
// setup used by the normal ACLNN path. A generated custom ACLNN host stub may
// dereference PTA thread-local state while it creates an executor. Every
// diagnostic mode therefore initializes PTA before the workspace query. The
// modes differ only after phase 2 has run: a non-repeatable aclOpExecutor is
// owned and automatically released by that phase. A GetWorkspaceSize-only
// case is deliberately not a checkpoint: manually destroying that executor
// is API-invalid unless the operator first supports and enters repeatable
// mode, which is neither the production path nor a valid P7 gate. The normal
// default is `full` and remains byte-for-byte equivalent to the existing
// production cleanup sequence.
enum class OrderedCleanupMode {
  kFull,
  kPhase2Noop,
  kReleaseConverted,
  kReleaseHugeMem,
  kUninitHugeMem,
  kPtaCache,
};

struct OrderedCleanupPlan {
  OrderedCleanupMode mode;
  bool bisect;
  bool release_converted;
  bool release_huge_mem;
  bool uninit_huge_mem;
  bool uninit_pta_cache;
};

inline const char *OrderedCleanupModeName(OrderedCleanupMode mode) {
  switch (mode) {
    case OrderedCleanupMode::kFull:
      return "full";
    case OrderedCleanupMode::kPhase2Noop:
      return "phase2_noop";
    case OrderedCleanupMode::kReleaseConverted:
      return "release_converted";
    case OrderedCleanupMode::kReleaseHugeMem:
      return "release_huge_mem";
    case OrderedCleanupMode::kUninitHugeMem:
      return "uninit_huge_mem";
    case OrderedCleanupMode::kPtaCache:
      return "pta_cache";
  }
  return "invalid";
}

inline bool IsEnabledEnvironmentFlag(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' &&
      !(value[0] == '0' && value[1] == '\0');
}

inline OrderedCleanupPlan OrderedCleanupPlanFromEnvironment() {
  // No normal caller can accidentally change cleanup timing just by setting
  // the mode variable.  Both variables are required and the P7-only runner
  // starts a fresh process for every bisect case.
  if (!IsEnabledEnvironmentFlag("NANOVLLM_A5_OPAPI_CLEANUP_BISECT")) {
    return {OrderedCleanupMode::kFull, false, true, true, true, true};
  }

  const char *value = std::getenv("NANOVLLM_A5_OPAPI_CLEANUP_MODE");
  TORCH_CHECK(value != nullptr && value[0] != '\0',
              "NANOVLLM_A5_OPAPI_CLEANUP_MODE is required when "
              "NANOVLLM_A5_OPAPI_CLEANUP_BISECT=1");
  if (std::strcmp(value, "phase2_noop") == 0) {
    return {OrderedCleanupMode::kPhase2Noop, true, false, false, false, false};
  }
  if (std::strcmp(value, "release_converted") == 0) {
    return {OrderedCleanupMode::kReleaseConverted, true, true, false, false, false};
  }
  if (std::strcmp(value, "release_huge_mem") == 0) {
    return {OrderedCleanupMode::kReleaseHugeMem, true, true, true, false, false};
  }
  if (std::strcmp(value, "uninit_huge_mem") == 0) {
    return {OrderedCleanupMode::kUninitHugeMem, true, true, true, true, false};
  }
  if (std::strcmp(value, "pta_cache") == 0) {
    return {OrderedCleanupMode::kPtaCache, true, true, true, true, true};
  }
  TORCH_CHECK(false,
              "unsupported NANOVLLM_A5_OPAPI_CLEANUP_MODE=", value,
              "; expected phase2_noop, release_converted, release_huge_mem, "
              "uninit_huge_mem, or pta_cache");
}

inline void LogOrderedCleanup(
    const char *api,
    const OrderedCleanupPlan &plan,
    const char *stage) {
  if (!plan.bisect && !OpApiDebugEnabled()) {
    return;
  }
  std::fprintf(
      stderr,
      "P7_1_CLEANUP api=%s mode=%s stage=%s bisect=%d\n",
      api,
      OrderedCleanupModeName(plan.mode),
      stage,
      plan.bisect ? 1 : 0);
  std::fflush(stderr);
}

// This is the same host-side cache lifecycle used by the checked-in CANN
// ACLNN adapter around GetWorkspaceSize.  Keep the three functions as a
// complete optional triplet: calling only init/set or only uninit would make
// an older libopapi's host allocator state less safe, not more safe.
inline bool InitPtaCacheForWorkspace() {
  static auto init = reinterpret_cast<InitPtaCacheThreadLocalFn>(
      ResolveSymbol("InitPTACacheThreadLocal"));
  static auto set_hash = reinterpret_cast<SetPtaHashKeyFn>(
      ResolveSymbol("SetPTAHashKey"));
  static auto uninit = reinterpret_cast<UnInitPtaCacheThreadLocalFn>(
      ResolveSymbol("UnInitPTACacheThreadLocal"));
  if (init == nullptr || set_hash == nullptr || uninit == nullptr) {
    return false;
  }
  init();
  set_hash(0);
  return true;
}

inline void UnInitPtaCacheForWorkspace(bool active) {
  if (!active) {
    return;
  }
  static auto uninit = reinterpret_cast<UnInitPtaCacheThreadLocalFn>(
      ResolveSymbol("UnInitPTACacheThreadLocal"));
  if (uninit != nullptr) {
    uninit();
  }
}

// An ACLNN custom operator may have the same exported symbol name in several
// local OPP packages.  When that happens, a stale ABI often fails later with a
// generic "dtype or format" message.  This one compact line identifies the
// exact DSO chosen by dlopen/dlsym before the generated API is called.
inline void LogResolvedOpApi(
    const char *api, void *workspace_addr, void *launch_addr) {
  if (!OpApiDebugEnabled()) {
    return;
  }
  Dl_info workspace_info{};
  Dl_info launch_info{};
  const int workspace_ok = dladdr(workspace_addr, &workspace_info);
  const int launch_ok = dladdr(launch_addr, &launch_info);
  const char *custom = std::getenv(NANOVLLM_A5_OPAPI_ENV);
  std::fprintf(
      stderr,
      "NANOVLLM_A5_OPAPI_RESOLVE api=%s custom_env_name=%s custom_env=%s workspace_dso=%s "
      "launch_dso=%s workspace_addr=%p launch_addr=%p\n",
      api,
      NANOVLLM_A5_OPAPI_ENV,
      custom == nullptr || custom[0] == '\0' ? "<unset>" : custom,
      workspace_ok && workspace_info.dli_fname != nullptr
          ? workspace_info.dli_fname : "<unknown>",
      launch_ok && launch_info.dli_fname != nullptr
          ? launch_info.dli_fname : "<unknown>",
      workspace_addr, launch_addr);
  std::fflush(stderr);
}

// Keep this deliberately compact: it records every host-side allocator /
// descriptor boundary that can corrupt the process after a device-side
// fail-closed result.  The P7 cleanup matrix below consumes this one line;
// it is not a per-tensor trace.
inline void LogOrderedHostDso(
    const char *api,
    void *workspace_addr,
    void *launch_addr,
    InitHugeMemThreadLocalFn init_huge_mem,
    ReleaseHugeMemFn release_huge_mem,
    UnInitHugeMemThreadLocalFn uninit_huge_mem) {
  if (!OpApiDebugEnabled()) {
    return;
  }
  std::fprintf(
      stderr,
      "P7_1_HOST_DSO api=%s loader=RTLD_NOW|RTLD_LOCAL "
      "workspace_dso=%s launch_dso=%s init_huge_dso=%s "
      "release_huge_dso=%s uninit_huge_dso=%s\n",
      api,
      DsoNameForAddress(workspace_addr),
      DsoNameForAddress(launch_addr),
      DsoNameForAddress(reinterpret_cast<void *>(init_huge_mem)),
      DsoNameForAddress(reinterpret_cast<void *>(release_huge_mem)),
      DsoNameForAddress(reinterpret_cast<void *>(uninit_huge_mem)));
  std::fflush(stderr);
}

}  // namespace nanovllm_a5_opapi

#define EXEC_NPU_CMD_ORDERED(aclnn_api, tensor_keepalive, ...)                \
  do {                                                                         \
    using namespace nanovllm_a5_opapi;                                         \
    const OrderedCleanupPlan ordered_cleanup_plan =                            \
        OrderedCleanupPlanFromEnvironment();                                   \
    LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan, "config");           \
    static void *workspace_addr_fn =                                           \
        ResolveSymbol(#aclnn_api "GetWorkspaceSize");                         \
    static void *launch_addr_fn = ResolveSymbol(#aclnn_api);                   \
    TORCH_CHECK(                                                               \
        workspace_addr_fn != nullptr && launch_addr_fn != nullptr,             \
        #aclnn_api,                                                            \
        " or ",                                                               \
        #aclnn_api "GetWorkspaceSize is missing from repository-local ",      \
        "libcust_opapi.so");                                                 \
    LogResolvedOpApi(#aclnn_api, workspace_addr_fn, launch_addr_fn);            \
                                                                               \
    auto npu_stream = c10_npu::getCurrentNPUStream();                          \
    aclrtStream acl_stream = npu_stream.stream(false);                         \
    if (!ordered_cleanup_plan.bisect) {                                        \
      UseStreamResIfNeeded(acl_stream, true);                                  \
      LogOpApiLifecycle(#aclnn_api, "enqueue_ready");                         \
    } else {                                                                   \
      LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan, "direct_stream"); \
    }                                                                          \
    /* PTA TLS is a GetWorkspaceSize precondition, not optional cleanup. */    \
    const bool pta_cache_active = InitPtaCacheForWorkspace();                  \
    LogOpApiLifecycle(#aclnn_api,                                              \
                      pta_cache_active ? "pta_cache_ready"                    \
                                       : "pta_cache_unavailable");             \
    LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan, "pta_cache_enabled");\
    auto init_huge_mem = reinterpret_cast<InitHugeMemThreadLocalFn>(           \
        ResolveSymbol("InitHugeMemThreadLocal"));                             \
    auto uninit_huge_mem = reinterpret_cast<UnInitHugeMemThreadLocalFn>(       \
        ResolveSymbol("UnInitHugeMemThreadLocal"));                           \
    auto release_huge_mem = reinterpret_cast<ReleaseHugeMemFn>(                \
        ResolveSymbol("ReleaseHugeMem"));                                     \
    LogOrderedHostDso(#aclnn_api, workspace_addr_fn, launch_addr_fn,           \
                      init_huge_mem, release_huge_mem, uninit_huge_mem);       \
    if (init_huge_mem != nullptr) {                                            \
      (void)init_huge_mem(nullptr, false);                                     \
      LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan, "init_huge_mem_done");\
    }                                                                          \
                                                                               \
    uint64_t workspace_size = 0;                                               \
    uint64_t *workspace_size_ptr = &workspace_size;                            \
    aclOpExecutor *executor = nullptr;                                         \
    aclOpExecutor **executor_ptr = &executor;                                  \
    auto converted = ConvertTypes(                                             \
        __VA_ARGS__, workspace_size_ptr, executor_ptr);                        \
    LogConvertedDescriptorSet(#aclnn_api, converted);                          \
    LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan,                        \
                      "workspace_call_begin");                               \
    auto workspace_fn = WorkspaceFunction(converted, workspace_addr_fn);       \
    int workspace_status = CallTuple(workspace_fn, converted);                 \
    TORCH_CHECK(                                                               \
        workspace_status == 0,                                                 \
        "call " #aclnn_api "GetWorkspaceSize failed: ",                     \
        RecentAclError());                                                     \
    LogOpApiLifecycle(#aclnn_api, "workspace_ready");                        \
                                                                               \
    at::Tensor workspace;                                                      \
    void *workspace_ptr = nullptr;                                             \
    if (workspace_size != 0) {                                                \
      const auto &_workspace_ref_tensor = std::get<0>(tensor_keepalive);       \
      workspace = at::empty(                                                   \
          {static_cast<int64_t>(workspace_size)},                              \
          _workspace_ref_tensor.options().dtype(at::kByte));                   \
      workspace_ptr = workspace.data_ptr();                                    \
    }                                                                          \
                                                                               \
    if (ordered_cleanup_plan.bisect) {                                        \
      /* P7 test-only matrix: direct launch plus stream synchronization       \
       * removes OpCommand callback/destructor timing from the equation.      \
       * It is unreachable in normal production execution. */                 \
      auto launch_fn = reinterpret_cast<OpApiLaunchFn>(launch_addr_fn);        \
      LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan,                     \
                        "direct_launch_begin");                              \
      const int launch_status =                                                \
          launch_fn(workspace_ptr, workspace_size, executor, acl_stream);     \
      TORCH_CHECK(                                                             \
          launch_status == 0,                                                  \
          "call " #aclnn_api " failed: ", RecentAclError());                \
      LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan,                     \
                        "direct_launch_done");                               \
      const aclError sync_status = aclrtSynchronizeStream(acl_stream);        \
      TORCH_CHECK(                                                             \
          sync_status == ACL_SUCCESS,                                         \
          "synchronize " #aclnn_api " failed: ", sync_status, "; ",       \
          RecentAclError());                                                   \
      LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan,                     \
                        "phase2_executor_autorelease_done");                 \
      LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan,                     \
                        "direct_stream_sync_done");                          \
      if (ordered_cleanup_plan.release_converted) {                            \
        ReleaseTuple(converted);                                               \
        LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan,                   \
                          "release_converted_done");                         \
      } else {                                                                 \
        LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan,                   \
                          "release_converted_held");                         \
      }                                                                        \
      if (ordered_cleanup_plan.release_huge_mem &&                             \
          release_huge_mem != nullptr) {                                      \
        release_huge_mem(nullptr, false);                                      \
        LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan,                   \
                          "release_huge_mem_done");                          \
      } else {                                                                 \
        LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan,                   \
                          "release_huge_mem_held");                          \
      }                                                                        \
      if (ordered_cleanup_plan.uninit_huge_mem &&                              \
          uninit_huge_mem != nullptr) {                                       \
        uninit_huge_mem(nullptr, false);                                       \
        LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan,                   \
                          "uninit_huge_mem_done");                           \
      } else {                                                                 \
        LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan,                   \
                          "uninit_huge_mem_held");                           \
      }                                                                        \
      if (ordered_cleanup_plan.uninit_pta_cache) {                             \
        UnInitPtaCacheForWorkspace(pta_cache_active);                          \
        LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan,                   \
                          "pta_uninit_done");                                \
      } else {                                                                 \
        LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan,                   \
                          "pta_uninit_held");                                \
      }                                                                        \
      LogOrderedCleanup(#aclnn_api, ordered_cleanup_plan, "direct_return"); \
    } else {                                                                   \
      auto ordered_tensor_keepalive = tensor_keepalive;                        \
      auto ordered_acl_call =                                                   \
          [converted, workspace, workspace_ptr, workspace_size, acl_stream,   \
           executor, ordered_tensor_keepalive, release_huge_mem]() mutable -> int { \
        (void)workspace;                                                       \
        (void)ordered_tensor_keepalive;                                        \
        UseStreamResIfNeeded(acl_stream, false);                               \
        LogOpApiLifecycle(#aclnn_api, "launch_begin");                       \
        auto launch_fn = reinterpret_cast<OpApiLaunchFn>(launch_addr_fn);      \
        const int launch_status =                                              \
            launch_fn(workspace_ptr, workspace_size, executor, acl_stream);   \
        TORCH_CHECK(                                                           \
            launch_status == 0,                                                \
            "call " #aclnn_api " failed: ", RecentAclError());              \
        LogOpApiLifecycle(#aclnn_api, "launch_done");                        \
        ReleaseTuple(converted);                                               \
        if (release_huge_mem != nullptr) {                                     \
          release_huge_mem(nullptr, false);                                    \
        }                                                                      \
        LogOpApiLifecycle(#aclnn_api, "descriptor_release_done");            \
        return launch_status;                                                  \
      };                                                                       \
      /* RunOpApi takes the closure by value.  We choose it to make that      \
       * ownership boundary explicit; this is not a claim that V2 caused P7. */\
      LogOpApiLifecycle(#aclnn_api, "handoff_runopapi_by_value");            \
      at_npu::native::OpCommand::RunOpApi(#aclnn_api, ordered_acl_call);      \
      if (uninit_huge_mem != nullptr) {                                        \
        uninit_huge_mem(nullptr, false);                                       \
      }                                                                        \
      UnInitPtaCacheForWorkspace(pta_cache_active);                            \
      LogOpApiLifecycle(#aclnn_api,                                            \
                        pta_cache_active ? "pta_cache_uninit"                \
                                         : "pta_cache_not_active");           \
      LogOpApiLifecycle(#aclnn_api, "submit_done");                          \
    }                                                                          \
  } while (false)

// P7.1 is a standalone device-side PostOnly/CQ executor. Its host launch uses
// the public ACLNN stream ABI directly: ACLNN enqueues work on the caller's
// current NPU stream and returns without waiting for DMA or CQ. The 31 input
// tensors and 18 mutable Ref tensors are graph RuntimeResources: their
// lifetime is required to cover this forward's stream terminal. This macro
// deliberately introduces no temporary workspace Tensor whose lifetime would
// end at the host return boundary.
//
// Do not route P7.1 through torch_npu::OpCommand. OpCommand is a private host
// task queue, while P7.1's real asynchronous work already lives on the device
// stream. A P7.1 fail-closed launch reached `submit_done` and then crashed in
// that second hand-off. The direct path below matches the repository's direct
// ACLNN baseline: create descriptors, query workspace, enqueue, then retain
// host-only descriptors until the graph forward terminal. PTA/HugeMem setup
// remains a GetWorkspaceSize precondition for this CANN release.
//
// This is P7-specific on purpose: existing operators retain their independently
// validated EXEC_NPU_CMD_ORDERED contracts. P7's OpDef declares zero workspace;
// fail closed if that changes instead of silently introducing an untracked
// temporary tensor lifetime into the asynchronous device path.
#define EXEC_NPU_CMD_P7_ZERO_WORKSPACE_DIRECT_ASYNC(aclnn_api, tensor_keepalive, ...) \
  do {                                                                           \
    using namespace nanovllm_a5_opapi;                                           \
    /* Do not let a prior failed/successful dispatch leak a stale token. */     \
    ClearP7DirectLaunchTokenForCurrentThread();                                  \
    static void *workspace_addr_fn =                                             \
        ResolveSymbol(#aclnn_api "GetWorkspaceSize");                          \
    static void *launch_addr_fn = ResolveSymbol(#aclnn_api);                    \
    TORCH_CHECK(                                                                 \
        workspace_addr_fn != nullptr && launch_addr_fn != nullptr,               \
        #aclnn_api,                                                              \
        " or ",                                                                \
        #aclnn_api "GetWorkspaceSize is missing from repository-local ",        \
        "libcust_opapi.so");                                                   \
    LogResolvedOpApi(#aclnn_api, workspace_addr_fn, launch_addr_fn);             \
    auto npu_stream = c10_npu::getCurrentNPUStream();                            \
    const aclrtStream acl_stream = npu_stream.stream(false);                     \
    /* Keep torch_npu core-control stream ownership in the direct path. */      \
    UseStreamResIfNeeded(acl_stream, true);                                      \
    LogOpApiLifecycle(#aclnn_api, "direct_async_enqueue_ready");             \
                                                                               \
    const bool pta_cache_active = InitPtaCacheForWorkspace();                    \
    LogOpApiLifecycle(#aclnn_api,                                                \
                      pta_cache_active ? "direct_async_pta_cache_ready"        \
                                       : "direct_async_pta_cache_unavailable"); \
    auto init_huge_mem = reinterpret_cast<InitHugeMemThreadLocalFn>(             \
        ResolveSymbol("InitHugeMemThreadLocal"));                               \
    auto uninit_huge_mem = reinterpret_cast<UnInitHugeMemThreadLocalFn>(         \
        ResolveSymbol("UnInitHugeMemThreadLocal"));                             \
    auto release_huge_mem = reinterpret_cast<ReleaseHugeMemFn>(                  \
        ResolveSymbol("ReleaseHugeMem"));                                       \
    LogOrderedHostDso(#aclnn_api, workspace_addr_fn, launch_addr_fn,             \
                      init_huge_mem, release_huge_mem, uninit_huge_mem);         \
    if (init_huge_mem != nullptr) {                                              \
      (void)init_huge_mem(nullptr, false);                                       \
    }                                                                            \
                                                                               \
    uint64_t workspace_size = 0;                                                 \
    uint64_t *workspace_size_ptr = &workspace_size;                              \
    aclOpExecutor *executor = nullptr;                                           \
    aclOpExecutor **executor_ptr = &executor;                                    \
    auto converted = ConvertTypes(                                               \
        __VA_ARGS__, workspace_size_ptr, executor_ptr);                          \
    LogConvertedDescriptorSet(#aclnn_api, converted);                            \
    auto direct_async_tls_cleanup = [&]() {                                     \
      if (release_huge_mem != nullptr) {                                        \
        release_huge_mem(nullptr, false);                                        \
      }                                                                          \
      if (uninit_huge_mem != nullptr) {                                         \
        uninit_huge_mem(nullptr, false);                                         \
      }                                                                          \
      UnInitPtaCacheForWorkspace(pta_cache_active);                              \
    };                                                                           \
    LogOpApiLifecycle(#aclnn_api, "direct_async_workspace_call_begin");       \
    auto workspace_fn = WorkspaceFunction(converted, workspace_addr_fn);         \
    const int workspace_status = CallTuple(workspace_fn, converted);             \
    if (workspace_status != 0) {                                                 \
      ReleaseTuple(converted);                                                   \
      direct_async_tls_cleanup();                                                \
      TORCH_CHECK(false,                                                         \
                  "call " #aclnn_api "GetWorkspaceSize failed: ",             \
                  RecentAclError());                                             \
    }                                                                            \
    LogOpApiLifecycle(#aclnn_api, "direct_async_workspace_ready");           \
                                                                               \
    if (workspace_size != 0) {                                                   \
      ReleaseTuple(converted);                                                   \
      direct_async_tls_cleanup();                                                \
      TORCH_CHECK(false,                                                         \
                  "P7 direct async ABI requires zero CANN workspace; nonzero ", \
                  "workspace needs an explicit stream-owned lifetime contract");\
    }                                                                            \
    void *workspace_ptr = nullptr;                                               \
                                                                               \
    auto direct_async_lease = MakeP7DirectLaunchLease(                          \
        std::move(converted), tensor_keepalive);                                 \
    if (!GetP7DirectLaunchLeaseRegistry().Reserve()) {                           \
      direct_async_tls_cleanup();                                                \
      TORCH_CHECK(false,                                                         \
                  "P7 direct launch lease capacity exhausted; forward terminal ",\
                  "did not release prior P7 leases");                         \
    }                                                                            \
    auto launch_fn = reinterpret_cast<OpApiLaunchFn>(launch_addr_fn);            \
    /* Mirror the checked torch_npu launch contract before direct ACLNN. */     \
    UseStreamResIfNeeded(acl_stream, false);                                     \
    LogOpApiLifecycle(#aclnn_api, "direct_async_launch_begin");               \
    const int launch_status =                                                     \
        launch_fn(workspace_ptr, workspace_size, executor, acl_stream);          \
    if (launch_status != 0) {                                                    \
      GetP7DirectLaunchLeaseRegistry().CancelReservation();                      \
      direct_async_tls_cleanup();                                                \
      TORCH_CHECK(false,                                                         \
                  "call " #aclnn_api " failed: ", RecentAclError());          \
    }                                                                            \
    LogOpApiLifecycle(#aclnn_api, "direct_async_launch_done");                \
                                                                               \
    const uint64_t direct_async_token =                                         \
        GetP7DirectLaunchLeaseRegistry().Commit(                                 \
            std::move(direct_async_lease), acl_stream);                          \
    PublishP7DirectLaunchTokenForCurrentThread(direct_async_token);              \
    LogOpApiLifecycle(#aclnn_api, "direct_async_descriptor_lease_held");     \
    direct_async_tls_cleanup();                                                  \
    LogOpApiLifecycle(#aclnn_api,                                                \
                      pta_cache_active ? "direct_async_pta_cache_uninit"       \
                                       : "direct_async_pta_cache_not_active");  \
    LogOpApiLifecycle(#aclnn_api, "direct_async_submit_done");                \
  } while (false)
