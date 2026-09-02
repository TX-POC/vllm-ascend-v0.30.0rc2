#pragma once

// Minimal, self-contained ACLNN launcher for nanovllm_dsa_a5.
//
// Generic bridges below use torch_npu's OpCommand task-queue contract.  The
// P8-only direct macro is the explicit TQ0 acceptance path: the caller
// submits all graph nodes directly to one current stream, then performs its
// own ACLNN host lifecycle without putting a bridge closure into ReleaseQueue.
// Device-side PostOnly/CQ work remains asynchronous.

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
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
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
  // `aclTensor` keeps the tensor description until GetWorkspaceSize has
  // consumed it.  The previous implementation constructed `storage_dims` as
  // a local vector and passed its data pointer to aclCreateTensor.  That
  // pointer was already invalid by the time the deferred task-queue consumer
  // called GetWorkspaceSize.  It can surface as a late, input-specific
  // `storage_rank=<null>` tiler failure (P8's rows_per_entry was the first
  // reproducible instance), even though the source at::Tensor is valid.
  //
  // This launcher intentionally models ND storage as the logical tensor
  // shape.  `sizes` is backed by TensorImpl, and the value-owned Tensor
  // keepalive remains live through the deferred WorkspaceSize/launch call.
  // Reuse that stable storage instead of manufacturing a short-lived shape
  // buffer.  Do not replace this with `tensor.storage()` metadata: that would
  // reintroduce the private NPUStorageImpl dependency this adapter avoids.
  const auto sizes = tensor.sizes();
  const auto strides = tensor.strides();
  aclTensor *descriptor = acl_tensor_api.create(
      sizes.data(),
      static_cast<uint64_t>(sizes.size()),
      acl_dtype,
      strides.data(),
      0,
      ACL_FORMAT_ND,
      sizes.data(),
      static_cast<uint64_t>(sizes.size()),
      tensor.data_ptr());
  // Do not turn a failed descriptor allocation into a later opaque tiler
  // error.  In particular, a null descriptor at P8 input 26 is otherwise
  // indistinguishable from a descriptor whose storage shape was lost after
  // construction.  Keep this check here so every bridge gets the same
  // fail-fast boundary before GetWorkspaceSize.
  TORCH_CHECK(descriptor != nullptr,
              "aclCreateTensor returned null: aten_dtype=",
              c10::toString(tensor.scalar_type()),
              " rank=", sizes.size(),
              " device=", tensor.device().str());
  return descriptor;
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

// RunOpApi's task consumer executes the launch closure, but torch_npu moves
// the closure into a separate ReleaseQueue before it destroys the closure.
// `torch.npu.synchronize()` drains the launch queue and the device stream; it
// does not make that later host-side closure destruction an ATen-Tensor
// lifetime boundary. Record each Tensor storage on the actual NPU stream
// immediately after a successful launch, then the closure can discard its
// Tensor values before it reaches ReleaseQueue. This keeps device memory alive
// through the stream without leaving TensorImpl destruction to a
// teardown-order-dependent background thread.
inline void RecordTensorStorageOnNpuStream(
    const at::Tensor &tensor, c10_npu::NPUStream stream) {
  if (!tensor.defined()) {
    return;
  }
  c10_npu::NPUCachingAllocator::recordStream(
      tensor.storage().data_ptr(), stream);
}

template <typename Tuple, size_t... I>
inline void RecordTensorTupleOnNpuStream(
    Tuple &args, c10_npu::NPUStream stream, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{
      (RecordTensorStorageOnNpuStream(std::get<I>(args), stream), 0)...};
}

template <typename Tuple>
inline void RecordTensorTupleOnNpuStream(
    Tuple &args, c10_npu::NPUStream stream) {
  constexpr size_t kSize = std::tuple_size<Tuple>::value;
  RecordTensorTupleOnNpuStream(args, stream,
                               std::make_index_sequence<kSize>{});
}

template <typename Tuple, size_t... I>
inline void ClearTensorTuple(
    Tuple &args, std::index_sequence<I...>) noexcept {
  (void)std::initializer_list<int>{
      ((std::get<I>(args) = at::Tensor()), 0)...};
}

template <typename Tuple>
inline void ClearTensorTuple(Tuple &args) noexcept {
  constexpr size_t kSize = std::tuple_size<Tuple>::value;
  ClearTensorTuple(args, std::make_index_sequence<kSize>{});
}

// P7.1 launches one ACLNN executor through torch_npu's ordinary host task
// queue.  There are two ownership domains which must remain separate:
//
//  * aclTensor descriptors are phase-1/phase-2 *host* objects.  The checked-in
//    CANN adapter destroys them immediately after aclnn(...) returns and
//    before it tears down HugeMem/PTA TLS.  Keeping them alive across that TLS
//    teardown is invalid: an aclTensor may still reference allocator-local
//    bookkeeping even though the device kernel has already received its
//    addresses.
//  * the ATen tensors are graph resources.  They may still be read by the
//    enqueued AIV kernel, so the bridge keeps those handles until the matching
//    forward stream terminal.  Production RuntimeResources owns this terminal
//    hand-off; the bounded registry below is only the bridge-side carrier.
//
// This registry is deliberately *not* an execution queue. The device kernel
// already owns CandidateSeed -> PostOnly -> DMA -> CQ. Its token is bound to
// the actual aclrtStream, rather than to a host thread, because a graph
// terminal callback may execute on a different runtime thread from the
// submitter. The registry tracks only host ownership of Tensor handles:
//
//   HOST_ENQUEUED -> HOST_HANDOFF_ACCEPTED -> FORWARD_TERMINAL_RELEASED
//                        \-> HOST_FAILED (kept until abort/terminal)
//
// It intentionally does not duplicate the device queue state machine. In
// particular, HOST_HANDOFF_ACCEPTED means torch_npu accepted one executor
// closure for ordered launch; it does not claim that ACLNN, HCCP DMA, or CQ
// has completed.
class P7DirectLaunchLeaseBase {
 public:
  virtual ~P7DirectLaunchLeaseBase() = default;
};

template <typename TensorKeepaliveTuple>
class P7DirectLaunchLease final : public P7DirectLaunchLeaseBase {
 public:
  explicit P7DirectLaunchLease(TensorKeepaliveTuple tensor_keepalive)
      : tensor_keepalive_(std::move(tensor_keepalive)) {}

 private:
  // The tuple keeps every input, including the 13 read-only tensors that are
  // not represented by a returned Ref alias, alive through stream terminal.
  TensorKeepaliveTuple tensor_keepalive_;
};

class P7DirectLaunchLeaseRegistry {
 private:
  enum class HostSubmissionState : uint8_t {
    kEnqueued = 1,
    kHandoffAccepted = 2,
    kFailed = 3,
  };

  struct PendingLease {
    uint64_t token;
    uintptr_t stream_key;
    HostSubmissionState state;
    std::shared_ptr<P7DirectLaunchLeaseBase> lease;
  };

 public:
  // This result is used only while RunOpApi itself throws before RunNpu can
  // publish the token into RuntimeResources.  The ordinary reclaiming case is
  // deliberately narrow:
  //
  // * kCancelledEnqueued: the producer failed before the consumer owned the
  //   closure, so removing the pre-reserved lease is ordinary cancellation.
  // * kFailedBeforePublishQuarantined: this should be unreachable after the
  //   bridge rejects task-queue-disabled and sync-launch streams.  Do not
  //   release it here: an unexpected consumer may have partially submitted
  //   device work.  Retain/quarantine it and fail the dispatch instead.
  //
  // A handoff-accepted token is never reclaimed here: a post-return consumer
  // failure must remain owned by the forward terminal.
  enum class PrePublishAbortAction : uint8_t {
    kCancelledEnqueued = 1,
    kFailedBeforePublishQuarantined = 2,
    kUnexpectedHandoffAccepted = 3,
    kNotPending = 4,
    kWrongStream = 5,
    kInternalFailure = 6,
  };

  static constexpr size_t kMaxPendingLeases = 16;

  P7DirectLaunchLeaseRegistry() { pending_.reserve(kMaxPendingLeases); }

  // Allocate and retain the terminal token *before* RunOpApi hands the
  // closure to torch_npu. The consumer may launch before the producer returns
  // to Python, so a post-callback Commit would race capture_forward_terminal.
  uint64_t ReserveEnqueued(std::shared_ptr<P7DirectLaunchLeaseBase> lease,
                           aclrtStream stream) {
    TORCH_CHECK(lease != nullptr, "P7 direct launch lease is null");
    TORCH_CHECK(stream != nullptr,
                "P7 direct launch lease requires a non-null ACL stream");
    std::lock_guard<std::mutex> guard(mutex_);
    TORCH_CHECK(pending_.size() < kMaxPendingLeases,
                "P7 ordered launch lease capacity exhausted; forward terminal ",
                "did not release prior P7 leases");
    TORCH_CHECK(next_token_ != 0,
                "P7 direct launch token space exhausted");
    const uint64_t token = next_token_++;
    pending_.push_back(
        {token, reinterpret_cast<uintptr_t>(stream),
         HostSubmissionState::kEnqueued, std::move(lease)});
    return token;
  }

  // Returns false only when the queue-consumer closure had already failed and
  // quarantined this exact token before the producer observed RunOpApi's
  // successful enqueue return.  That race is valid with an eager task queue:
  // the failure must remain terminal-owned, not be overwritten as accepted.
  bool MarkHostHandoffAccepted(uint64_t token, aclrtStream stream) {
    TORCH_CHECK(stream != nullptr,
                "P7 ordered launch state update requires a non-null stream");
    std::lock_guard<std::mutex> guard(mutex_);
    const auto pending = FindPendingLocked(token);
    TORCH_CHECK(pending != pending_.end(),
                "P7 ordered launch token is not pending during host handoff "
                "accepted: ", token);
    TORCH_CHECK(pending->stream_key == reinterpret_cast<uintptr_t>(stream),
                "P7 ordered launch token belongs to a different stream: ",
                token);
    if (pending->state == HostSubmissionState::kFailed) {
      return false;
    }
    TORCH_CHECK(pending->state == HostSubmissionState::kEnqueued,
                "P7 ordered launch token has invalid host state during "
                "handoff: ", token);
    pending->state = HostSubmissionState::kHandoffAccepted;
    return true;
  }

  // This runs while the torch_npu queue consumer is already unwinding an
  // exception.  It is therefore intentionally noexcept: a bookkeeping fault
  // must never replace the original ACLNN / queue-consumer exception with a
  // process-terminating second exception.  A false result still leaves the
  // original consumer failure visible to torch_npu.
  bool MarkHostFailedNoThrow(uint64_t token, aclrtStream stream) noexcept {
    if (token == 0 || stream == nullptr) {
      return false;
    }
    try {
      std::lock_guard<std::mutex> guard(mutex_);
      const auto pending = FindPendingLocked(token);
      if (pending == pending_.end() ||
          pending->stream_key != reinterpret_cast<uintptr_t>(stream)) {
        return false;
      }
      // A late observer after terminal release is not an ownership error: the
      // terminal was already established.  Otherwise retain the tensor lease
      // for forward abort/terminal cleanup instead of silently releasing it.
      pending->state = HostSubmissionState::kFailed;
      return true;
    } catch (...) {
      return false;
    }
  }

  // Resolve the producer-side exception boundary before Publish token.  It is
  // noexcept because it runs while preserving the original RunOpApi error.
  // Tensor handles are deliberately destroyed after releasing mutex_: neither
  // a Tensor destructor nor Python refcount work may run under the registry
  // lock.  This function never releases a failed or handoff-accepted token:
  // those states may have device work and must remain quarantined rather than
  // losing keepalives before a proven terminal.
  PrePublishAbortAction ResolvePrePublishAbortNoThrow(
      uint64_t token, aclrtStream stream) noexcept {
    std::shared_ptr<P7DirectLaunchLeaseBase> released;
    PrePublishAbortAction action = PrePublishAbortAction::kInternalFailure;
    try {
      if (token == 0 || stream == nullptr) {
        return PrePublishAbortAction::kInternalFailure;
      }
      {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto pending = FindPendingLocked(token);
        if (pending == pending_.end()) {
          action = PrePublishAbortAction::kNotPending;
        } else if (pending->stream_key != reinterpret_cast<uintptr_t>(stream)) {
          action = PrePublishAbortAction::kWrongStream;
        } else if (pending->state == HostSubmissionState::kEnqueued) {
          // No consumer closure owns this lease.  This is ordinary synchronous
          // producer cancellation, not a queue quarantine.
          released = std::move(pending->lease);
          pending_.erase(pending);
          action = PrePublishAbortAction::kCancelledEnqueued;
        } else if (pending->state == HostSubmissionState::kFailed) {
          // Never free this lease from an exception path.  A consumer that
          // reached this state may have submitted part of an ACLNN launch.
          // With the strict task-queue gate below this is an invariant failure,
          // not an alternate cleanup path: preserve the quarantine and let the
          // original dispatch failure abort the forward/process path.
          action = PrePublishAbortAction::kFailedBeforePublishQuarantined;
        } else {
          // Do not free a token that reached handoff.  If this invariant is
          // ever observed, retaining the bounded lease is safer than allowing
          // a live asynchronous launch to lose its Tensor keepalives.
          action = PrePublishAbortAction::kUnexpectedHandoffAccepted;
        }
      }
      released.reset();
      return action;
    } catch (...) {
      return PrePublishAbortAction::kInternalFailure;
    }
  }

  static const char *PrePublishAbortActionName(
      PrePublishAbortAction action) noexcept {
    switch (action) {
      case PrePublishAbortAction::kCancelledEnqueued:
        return "cancel_enqueued";
      case PrePublishAbortAction::kFailedBeforePublishQuarantined:
        return "failed_pre_publish_quarantined_fatal";
      case PrePublishAbortAction::kUnexpectedHandoffAccepted:
        return "unexpected_handoff_retained";
      case PrePublishAbortAction::kNotPending:
        return "not_pending";
      case PrePublishAbortAction::kWrongStream:
        return "wrong_stream";
      case PrePublishAbortAction::kInternalFailure:
        return "internal_failure";
    }
    return "unknown";
  }

  int64_t ReleaseAtForwardTerminal(uint64_t terminal_token) {
    std::vector<std::shared_ptr<P7DirectLaunchLeaseBase>> completed;
    // Exact-token release moves one shared_ptr while holding the registry
    // mutex. Reserve first so allocation cannot throw after that move and
    // strand a lease in neither owner.
    completed.reserve(1);
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
      TORCH_CHECK(
          terminal->state == HostSubmissionState::kHandoffAccepted ||
              terminal->state == HostSubmissionState::kFailed,
          "P7 forward-terminal token was never handed to the ordered "
          "consumer: ", terminal_token);
      // Do not use a same-stream `token <= terminal_token` shortcut here.
      // A later producer can enqueue/terminal ahead of an earlier producer
      // whose token was reserved before RunOpApi hand-off.  Releasing that
      // earlier lease would allow its tensors to die before its consumer
      // closure reaches ACLNN.  RuntimeResources owns one exact token, so the
      // only sound terminal operation is exact-token destruction.
      completed.push_back(std::move(terminal->lease));
      pending_.erase(terminal);
    }
    const int64_t released = static_cast<int64_t>(completed.size());
    // Destruction releases only Tensor keepalives after the caller has
    // established the forward stream terminal.  aclTensor descriptors were
    // already released immediately after phase 2, before PTA/HugeMem cleanup.
    completed.clear();
    return released;
  }

  int64_t PendingCount() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return static_cast<int64_t>(pending_.size());
  }

 private:
  std::vector<PendingLease>::iterator FindPendingLocked(
      uint64_t token) {
    return std::find_if(
        pending_.begin(), pending_.end(),
        [token](const PendingLease& pending) { return pending.token == token; });
  }

  mutable std::mutex mutex_;
  uint64_t next_token_ = 1;
  std::vector<PendingLease> pending_;
};

inline P7DirectLaunchLeaseRegistry &GetP7DirectLaunchLeaseRegistry() {
  // This bridge is loaded into CPython beside Torch/CANN.  A normal forward
  // terminal must leave `pending_` empty, but destroying this mutex/vector
  // during extension unload can still race an external runtime's teardown
  // order.  Keep the bounded registry process-lifetime instead: it changes no
  // forward semantics or memory bound (16 pending leases), and removes this
  // bridge-owned non-trivial static destructor from that order.
  static auto *registry = new P7DirectLaunchLeaseRegistry();
  return *registry;
}

// An observer is carried in a bridge's existing tensor keepalive tuple.  The
// generic ordered launcher knows nothing about P7 state, but on a consumer
// exception it can notify the observer before it rethrows into torch_npu's
// task queue.  Other bridges simply do not carry one and pay no behavioural
// cost.
class OrderedLaunchFailureObserver {
 public:
  virtual ~OrderedLaunchFailureObserver() = default;
  virtual void OnOrderedLaunchFailure() noexcept = 0;
};

inline void NotifyOrderedLaunchFailure(
    const std::shared_ptr<OrderedLaunchFailureObserver> &observer) noexcept {
  if (observer != nullptr) {
    observer->OnOrderedLaunchFailure();
  }
}

template <typename T>
inline void NotifyOrderedLaunchFailure(const T &) noexcept {}

template <typename Tuple, size_t... I>
inline void NotifyOrderedLaunchFailureTuple(
    const Tuple &tuple, std::index_sequence<I...>) noexcept {
  (void)std::initializer_list<int>{
      (NotifyOrderedLaunchFailure(std::get<I>(tuple)), 0)...};
}

template <typename Tuple>
inline void NotifyOrderedLaunchFailureTuple(const Tuple &tuple) noexcept {
  constexpr size_t kSize = std::tuple_size<Tuple>::value;
  NotifyOrderedLaunchFailureTuple(tuple, std::make_index_sequence<kSize>{});
}

class P7OrderedLaunchFailureObserver final
    : public OrderedLaunchFailureObserver {
 public:
  P7OrderedLaunchFailureObserver(uint64_t token, aclrtStream stream)
      : token_(token), stream_(stream) {}

  void OnOrderedLaunchFailure() noexcept override {
    (void)GetP7DirectLaunchLeaseRegistry().MarkHostFailedNoThrow(token_, stream_);
  }

 private:
  uint64_t token_;
  aclrtStream stream_;
};

inline std::shared_ptr<OrderedLaunchFailureObserver>
MakeP7OrderedLaunchFailureObserver(uint64_t token, aclrtStream stream) {
  TORCH_CHECK(token != 0 && stream != nullptr,
              "P7 ordered launch failure observer requires token and stream");
  return std::make_shared<P7OrderedLaunchFailureObserver>(token, stream);
}

template <typename TensorKeepaliveTuple>
inline std::shared_ptr<P7DirectLaunchLeaseBase> MakeP7DirectLaunchLease(
    TensorKeepaliveTuple tensor_keepalive) {
  return std::make_shared<
      P7DirectLaunchLease<TensorKeepaliveTuple>>(
      std::move(tensor_keepalive));
}

// The dispatch bridge consumes this immediately after one ordered P7 host
// submission
// and stores it in the forward's RuntimeResources.  It is thread-local only
// for hand-off from the synchronous dispatch return to that RuntimeResources
// object; terminal release itself is intentionally allowed from any thread.
inline uint64_t& LastP7DirectLaunchTokenForCurrentThread() {
  static thread_local uint64_t token = 0;
  return token;
}

inline void ClearP7DirectLaunchTokenForCurrentThread() {
  uint64_t &token = LastP7DirectLaunchTokenForCurrentThread();
  TORCH_CHECK(
      token == 0,
      "P7 previous ordered-submission token was not captured before the next "
      "submission; refusing to discard terminal ownership: ", token);
}

inline void PublishP7DirectLaunchTokenForCurrentThread(uint64_t token) {
  TORCH_CHECK(token != 0, "P7 ordered submission token must be non-zero");
  uint64_t &previous = LastP7DirectLaunchTokenForCurrentThread();
  TORCH_CHECK(previous == 0,
              "P7 ordered submission token would overwrite uncaptured token: ",
              previous);
  previous = token;
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

inline void LogConvertedDescriptorArgument(
    const char *api, size_t index, aclTensor *descriptor) {
  if (!OpApiDebugEnabled()) {
    return;
  }
  // This is deliberately one line per ACL tensor, rather than a host-address
  // dump of every tuple member.  It lets a P8 run identify whether its
  // zero-based input 26 (`rows_per_entry`) failed descriptor creation or
  // survived creation and later lost shape materialization in CANN.
  std::fprintf(stderr,
               "P7_1_DESCRIPTOR_ARG api=%s index=%zu descriptor=%p\n",
               api, index, static_cast<void *>(descriptor));
  std::fflush(stderr);
}

template <typename T>
inline void LogConvertedDescriptorArgument(const char *, size_t, T) {}

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
  (void)std::initializer_list<int>{
      (LogConvertedDescriptorArgument(api, I, std::get<I>(args)), 0)...};
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

// P8 deliberately does not hand its ACLNN host lifecycle to torch_npu's
// task queue.  On this torch_npu revision an executed RunOpApi callback is
// moved to a private ReleaseQueue, whose destruction is not covered by
// torch.npu.synchronize().  A bridge-local callback can therefore outlive
// Python/DSO teardown even after all device work is complete.
//
// There is no public torch_npu ABI that drains its host task queue and then
// returns a raw ACL stream for bridge-local ACLNN.  P8 must not emulate that
// boundary with header-only c10_npu internals: the deployed runtime need not
// export them.  The direct acceptance path therefore requires task queue
// disabled before Python imports torch_npu.  C8, Stage-A and P8 are submitted
// by one caller in program order to the same current stream.
//
// This is deliberately not the future TQ1 product contract.  TQ1 will use a
// framework-ordered trampoline with an explicit lifecycle owner; it may not
// use this raw direct path.
inline void RequireP8Tq0DirectAsyncOpApiContract() {
  const char* task_queue = std::getenv("TASK_QUEUE_ENABLE");
  const char* launch_blocking = std::getenv("ASCEND_LAUNCH_BLOCKING");
  TORCH_CHECK(
      task_queue != nullptr && std::strcmp(task_queue, "0") == 0 &&
          (launch_blocking == nullptr ||
           std::strcmp(launch_blocking, "0") == 0),
      "P8 TQ0 direct ACLNN requires TASK_QUEUE_ENABLE=0 and "
      "ASCEND_LAUNCH_BLOCKING=0 before torch_npu import. "
      "TASK_QUEUE_ENABLE=",
      task_queue == nullptr ? "<unset>" : task_queue,
      " ASCEND_LAUNCH_BLOCKING=",
      launch_blocking == nullptr ? "<unset>" : launch_blocking);
  if (OpApiDebugEnabled()) {
    std::fprintf(
        stderr,
        "P8_TQ0_DIRECT_HOST_GATE task_queue_enable=0 "
        "launch_blocking=0 host_lifecycle=caller_thread "
        "queue_order=caller_submission_tq0 "
        "device_launch=async\n");
    std::fflush(stderr);
  }
}

// Keep the call in the Python bridge's caller thread.  Unlike generic
// OpCommand, P8 deliberately retains the GIL over this short host-only
// region.  TQ0 makes the caller's graph submission order authoritative, and
// the launch still returns as soon as its device work is enqueued.
//
// P8's current acceptance contract has one Python graph submitter on this
// stream.  A future TQ1 path will use framework ordering plus explicit event
// edges, rather than pretending a bridge-local mutex can serialize foreign
// task-queue producers.
template <typename Function>
inline int RunP8Tq0CallerThreadHost(Function&& function) {
  return std::forward<Function>(function)();
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
        try {                                                                  \
          (void)workspace;                                                     \
          (void)ordered_tensor_keepalive;                                      \
          UseStreamResIfNeeded(acl_stream, false);                             \
          LogOpApiLifecycle(#aclnn_api, "launch_begin");                     \
          auto launch_fn = reinterpret_cast<OpApiLaunchFn>(launch_addr_fn);    \
          const int launch_status =                                            \
              launch_fn(workspace_ptr, workspace_size, executor, acl_stream); \
          TORCH_CHECK(                                                         \
              launch_status == 0,                                              \
              "call " #aclnn_api " failed: ", RecentAclError());            \
          LogOpApiLifecycle(#aclnn_api, "launch_done");                      \
          ReleaseTuple(converted);                                             \
          if (release_huge_mem != nullptr) {                                   \
            release_huge_mem(nullptr, false);                                  \
          }                                                                    \
          LogOpApiLifecycle(#aclnn_api, "descriptor_release_done");          \
          return launch_status;                                                \
        } catch (...) {                                                        \
          /* P7 optionally carries one observer in this tuple.  Mark that     \
           * exact terminal lease failed before torch_npu receives the        \
           * original exception. Other bridges carry no observer. */          \
          NotifyOrderedLaunchFailureTuple(ordered_tensor_keepalive);           \
          LogOpApiLifecycle(#aclnn_api, "consumer_failed_quarantined");     \
          throw;                                                               \
        }                                                                      \
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

// P7's ordered launch has a stricter host-lifetime requirement than the
// historical generic launcher above.  `RunOpApi` hands a closure from the
// Python submitter thread to a torch_npu task-queue consumer thread.  PTA and
// HugeMem explicitly have ThreadLocal in their CANN ABI names, and aclTensor
// descriptors / aclOpExecutor are allocated from that host-side lifecycle.
//
// Therefore this P7-only helper deliberately crosses the queue boundary with
// *only* value-owned ATen tensors plus the graph-terminal keepalive tuple.  It
// constructs and destroys all ACLNN host objects in the one consumer closure
// that launches the operation.  Do not generalize this by blindly changing
// EXEC_NPU_CMD_ORDERED: other callers may pass AclTensorArgument, which holds
// a reference rather than a value.  P7 passes plain at::Tensor values only.
//
// Producer thread:
//   framework enqueue hand-off -> RunOpApi
// Consumer thread:
//   PTA/HugeMem -> descriptors -> GetWorkspaceSize -> launch -> descriptor
//   release -> HugeMem/PTA teardown
//
// The P7 kernel owns CandidateSeed -> PostOnly -> HCCP DMA -> CQ afterwards;
// this helper only fixes the host ACLNN hand-off and never waits for DMA/CQ.
#define EXEC_NPU_CMD_P7_ORDERED_DEFERRED(aclnn_api, tensor_keepalive, ...)    \
  do {                                                                         \
    using namespace nanovllm_a5_opapi;                                        \
    static void *p7_workspace_addr_fn =                                       \
        ResolveSymbol(#aclnn_api "GetWorkspaceSize");                        \
    static void *p7_launch_addr_fn = ResolveSymbol(#aclnn_api);               \
    TORCH_CHECK(                                                               \
        p7_workspace_addr_fn != nullptr && p7_launch_addr_fn != nullptr,      \
        #aclnn_api,                                                            \
        " or ",                                                               \
        #aclnn_api "GetWorkspaceSize is missing from repository-local ",     \
        "libcust_opapi.so");                                                 \
    LogResolvedOpApi(#aclnn_api, p7_workspace_addr_fn, p7_launch_addr_fn);    \
    auto p7_npu_stream = c10_npu::getCurrentNPUStream();                       \
    aclrtStream p7_acl_stream = p7_npu_stream.stream(false);                  \
    /* This producer-side call is the torch_npu queue hand-off pair. */       \
    UseStreamResIfNeeded(p7_acl_stream, true);                                \
    LogOpApiLifecycle(#aclnn_api, "producer_enqueue_ready");                 \
    /* std::make_tuple decays P7's at::Tensor arguments to value ownership. */\
    auto p7_deferred_args = std::make_tuple(__VA_ARGS__);                      \
    auto p7_ordered_tensor_keepalive = tensor_keepalive;                       \
    auto p7_ordered_acl_call =                                                 \
        [p7_deferred_args = std::move(p7_deferred_args),                       \
         p7_ordered_tensor_keepalive = std::move(p7_ordered_tensor_keepalive),\
         p7_acl_stream, p7_workspace_addr_fn, p7_launch_addr_fn]() mutable    \
        -> int {                                                               \
      bool p7_pta_cache_active = false;                                       \
      bool p7_huge_mem_active = false;                                        \
      auto p7_init_huge_mem =                                                  \
          reinterpret_cast<InitHugeMemThreadLocalFn>(                         \
              ResolveSymbol("InitHugeMemThreadLocal"));                      \
      auto p7_uninit_huge_mem =                                                \
          reinterpret_cast<UnInitHugeMemThreadLocalFn>(                       \
              ResolveSymbol("UnInitHugeMemThreadLocal"));                    \
      auto p7_release_huge_mem =                                               \
          reinterpret_cast<ReleaseHugeMemFn>(ResolveSymbol("ReleaseHugeMem"));\
      auto p7_cleanup_consumer_tls = [&]() noexcept {                          \
        if (p7_huge_mem_active) {                                              \
          if (p7_release_huge_mem != nullptr) {                               \
            p7_release_huge_mem(nullptr, false);                              \
          }                                                                    \
          if (p7_uninit_huge_mem != nullptr) {                                \
            p7_uninit_huge_mem(nullptr, false);                               \
          }                                                                    \
          p7_huge_mem_active = false;                                         \
        }                                                                      \
        if (p7_pta_cache_active) {                                            \
          UnInitPtaCacheForWorkspace(true);                                   \
          p7_pta_cache_active = false;                                        \
        }                                                                      \
        LogOpApiLifecycle(#aclnn_api, "consumer_tls_uninit");                \
      };                                                                       \
      try {                                                                    \
        (void)p7_ordered_tensor_keepalive;                                    \
        /* This is the consumer-side mate of producer_enqueue_ready. */       \
        UseStreamResIfNeeded(p7_acl_stream, false);                           \
        LogOpApiLifecycle(#aclnn_api, "consumer_stream_ready");              \
        p7_pta_cache_active = InitPtaCacheForWorkspace();                     \
        LogOpApiLifecycle(                                                    \
            #aclnn_api,                                                        \
            p7_pta_cache_active ? "consumer_pta_cache_ready"                 \
                                : "consumer_pta_cache_unavailable");          \
        LogOrderedHostDso(#aclnn_api, p7_workspace_addr_fn, p7_launch_addr_fn,\
                          p7_init_huge_mem, p7_release_huge_mem,              \
                          p7_uninit_huge_mem);                                \
        if (p7_init_huge_mem != nullptr) {                                    \
          (void)p7_init_huge_mem(nullptr, false);                             \
          p7_huge_mem_active = true;                                          \
          LogOpApiLifecycle(#aclnn_api, "consumer_huge_mem_ready");          \
        }                                                                      \
        uint64_t p7_workspace_size = 0;                                       \
        uint64_t *p7_workspace_size_ptr = &p7_workspace_size;                 \
        aclOpExecutor *p7_executor = nullptr;                                 \
        aclOpExecutor **p7_executor_ptr = &p7_executor;                       \
        auto p7_converted = std::apply(                                       \
            [&](auto &...p7_args) {                                           \
              return ConvertTypes(p7_args..., p7_workspace_size_ptr,          \
                                  p7_executor_ptr);                            \
            },                                                                 \
            p7_deferred_args);                                                \
        LogConvertedDescriptorSet(#aclnn_api, p7_converted);                  \
        bool p7_converted_released = false;                                   \
        try {                                                                  \
          auto p7_workspace_fn =                                              \
              WorkspaceFunction(p7_converted, p7_workspace_addr_fn);          \
          const int p7_workspace_status =                                    \
              CallTuple(p7_workspace_fn, p7_converted);                       \
          TORCH_CHECK(                                                        \
              p7_workspace_status == 0,                                      \
              "call " #aclnn_api "GetWorkspaceSize failed: ",              \
              RecentAclError());                                              \
          /* P7's hand-written AIV-only kernel has no CANN workspace ABI. */  \
          TORCH_CHECK(                                                        \
              p7_workspace_size == 0,                                        \
              "P7.1 requires zero ACLNN workspace; got ",                   \
              p7_workspace_size);                                            \
          LogOpApiLifecycle(#aclnn_api, "consumer_workspace_ready");         \
          auto p7_launch_fn =                                                 \
              reinterpret_cast<OpApiLaunchFn>(p7_launch_addr_fn);             \
          LogOpApiLifecycle(#aclnn_api, "launch_begin");                     \
          const int p7_launch_status =                                        \
              p7_launch_fn(nullptr, p7_workspace_size, p7_executor,          \
                           p7_acl_stream);                                    \
          TORCH_CHECK(                                                        \
              p7_launch_status == 0,                                         \
              "call " #aclnn_api " failed: ", RecentAclError());           \
          LogOpApiLifecycle(#aclnn_api, "launch_done");                      \
          ReleaseTuple(p7_converted);                                        \
          p7_converted_released = true;                                       \
          LogOpApiLifecycle(#aclnn_api, "descriptor_release_done");          \
          p7_cleanup_consumer_tls();                                          \
          return p7_launch_status;                                            \
        } catch (...) {                                                        \
          if (!p7_converted_released) {                                      \
            ReleaseTuple(p7_converted);                                      \
          }                                                                   \
          throw;                                                              \
        }                                                                      \
      } catch (...) {                                                          \
        p7_cleanup_consumer_tls();                                            \
        /* Mark the exact forward lease before torch_npu propagates error. */ \
        NotifyOrderedLaunchFailureTuple(p7_ordered_tensor_keepalive);         \
        LogOpApiLifecycle(#aclnn_api, "consumer_failed_quarantined");        \
        throw;                                                                 \
      }                                                                        \
    };                                                                         \
    LogOpApiLifecycle(#aclnn_api, "handoff_runopapi_p7_deferred");           \
    at_npu::native::OpCommand::RunOpApi(#aclnn_api, p7_ordered_acl_call);     \
    /* Producer owns no ACLNN descriptor/PTA/HugeMem state after hand-off. */ \
    LogOpApiLifecycle(#aclnn_api, "submit_done");                            \
  } while (false)

// This is the narrow reusable form of P7's deferred host-lifetime contract.
// It is intentionally *not* a replacement for EXEC_NPU_CMD_ORDERED: callers
// may use it only when every ACLNN argument is a value-ownable at::Tensor and
// the operator has a zero-workspace ABI.  Those two restrictions mean that
// the producer may hand only Tensor values to the torch_npu queue consumer;
// the consumer then owns the complete TLS -> descriptor -> executor -> launch
// -> descriptor/TLS teardown interval.  In particular, no aclTensor or
// aclOpExecutor constructed on the producer thread can outlive that thread's
// HugeMem/PTA state.
//
// This retains framework ordering but never waits for device work.  It is
// suitable for P8's Admission, Advance and executor graph nodes because their
// generated host tilers explicitly declare zero workspace.  Do not relax the
// workspace check without first defining an explicit stream-lifetime owner
// for a consumer-created workspace Tensor.
#define EXEC_NPU_CMD_TENSOR_OWNED_DEFERRED_ZERO_WORKSPACE_IMPL(               \
    aclnn_api, workspace_function_expression, tensor_keepalive, ...)         \
  do {                                                                         \
    using namespace nanovllm_a5_opapi;                                        \
    static void *deferred_workspace_addr_fn =                                 \
        ResolveSymbol(#aclnn_api "GetWorkspaceSize");                        \
    static void *deferred_launch_addr_fn = ResolveSymbol(#aclnn_api);         \
    TORCH_CHECK(                                                               \
        deferred_workspace_addr_fn != nullptr &&                              \
            deferred_launch_addr_fn != nullptr,                               \
        #aclnn_api,                                                            \
        " or ",                                                               \
        #aclnn_api "GetWorkspaceSize is missing from repository-local ",     \
        "libcust_opapi.so");                                                 \
    LogResolvedOpApi(#aclnn_api, deferred_workspace_addr_fn,                  \
                     deferred_launch_addr_fn);                                \
    auto deferred_npu_stream = c10_npu::getCurrentNPUStream();                \
    aclrtStream deferred_acl_stream = deferred_npu_stream.stream(false);     \
    /* Producer-side stream ownership is the framework queue hand-off. */    \
    UseStreamResIfNeeded(deferred_acl_stream, true);                          \
    LogOpApiLifecycle(#aclnn_api, "producer_enqueue_ready");                \
    /* std::make_tuple decays every at::Tensor argument to value ownership. */\
    auto deferred_args = std::make_tuple(__VA_ARGS__);                        \
    auto deferred_tensor_keepalive = tensor_keepalive;                        \
    auto deferred_acl_call =                                                  \
        [deferred_args = std::move(deferred_args),                            \
         deferred_tensor_keepalive = std::move(deferred_tensor_keepalive),   \
         deferred_npu_stream, deferred_acl_stream,                            \
         deferred_workspace_addr_fn,                                          \
         deferred_launch_addr_fn]() mutable -> int {                          \
      bool deferred_pta_cache_active = false;                                 \
      bool deferred_huge_mem_active = false;                                  \
      auto deferred_init_huge_mem =                                           \
          reinterpret_cast<InitHugeMemThreadLocalFn>(                         \
              ResolveSymbol("InitHugeMemThreadLocal"));                      \
      auto deferred_uninit_huge_mem =                                         \
          reinterpret_cast<UnInitHugeMemThreadLocalFn>(                       \
              ResolveSymbol("UnInitHugeMemThreadLocal"));                    \
      auto deferred_release_huge_mem =                                        \
          reinterpret_cast<ReleaseHugeMemFn>(ResolveSymbol("ReleaseHugeMem"));\
      auto deferred_cleanup_consumer_tls = [&]() noexcept {                   \
        if (deferred_huge_mem_active) {                                       \
          if (deferred_release_huge_mem != nullptr) {                         \
            deferred_release_huge_mem(nullptr, false);                        \
          }                                                                    \
          if (deferred_uninit_huge_mem != nullptr) {                          \
            deferred_uninit_huge_mem(nullptr, false);                         \
          }                                                                    \
          deferred_huge_mem_active = false;                                   \
        }                                                                      \
        if (deferred_pta_cache_active) {                                      \
          UnInitPtaCacheForWorkspace(true);                                    \
          deferred_pta_cache_active = false;                                  \
        }                                                                      \
        LogOpApiLifecycle(#aclnn_api, "consumer_tls_uninit");               \
      };                                                                       \
      try {                                                                    \
        (void)deferred_tensor_keepalive;                                      \
        UseStreamResIfNeeded(deferred_acl_stream, false);                     \
        LogOpApiLifecycle(#aclnn_api, "consumer_stream_ready");             \
        deferred_pta_cache_active = InitPtaCacheForWorkspace();               \
        LogOpApiLifecycle(                                                     \
            #aclnn_api,                                                        \
            deferred_pta_cache_active ? "consumer_pta_cache_ready"          \
                                      : "consumer_pta_cache_unavailable");   \
        LogOrderedHostDso(#aclnn_api, deferred_workspace_addr_fn,             \
                          deferred_launch_addr_fn, deferred_init_huge_mem,    \
                          deferred_release_huge_mem,                           \
                          deferred_uninit_huge_mem);                          \
        if (deferred_init_huge_mem != nullptr) {                              \
          (void)deferred_init_huge_mem(nullptr, false);                       \
          deferred_huge_mem_active = true;                                    \
          LogOpApiLifecycle(#aclnn_api, "consumer_huge_mem_ready");         \
        }                                                                      \
        uint64_t deferred_workspace_size = 0;                                 \
        uint64_t *deferred_workspace_size_ptr = &deferred_workspace_size;    \
        aclOpExecutor *deferred_executor = nullptr;                           \
        aclOpExecutor **deferred_executor_ptr = &deferred_executor;           \
        auto deferred_converted = std::apply(                                 \
            [&](auto &...deferred_args_ref) {                                 \
              return ConvertTypes(deferred_args_ref...,                       \
                                  deferred_workspace_size_ptr,                 \
                                  deferred_executor_ptr);                      \
            },                                                                 \
            deferred_args);                                                    \
        LogConvertedDescriptorSet(#aclnn_api, deferred_converted);            \
        bool deferred_converted_released = false;                             \
        try {                                                                  \
          auto deferred_workspace_fn = workspace_function_expression;         \
          LogOpApiLifecycle(#aclnn_api, "consumer_workspace_fn_ready");     \
          LogOpApiLifecycle(#aclnn_api, "consumer_workspace_call_begin");  \
          const int deferred_workspace_status =                               \
              CallTuple(deferred_workspace_fn, deferred_converted);          \
          LogOpApiLifecycle(#aclnn_api, "consumer_workspace_call_end");    \
          TORCH_CHECK(                                                        \
              deferred_workspace_status == 0,                                 \
              "call " #aclnn_api "GetWorkspaceSize failed: ",              \
              RecentAclError());                                              \
          TORCH_CHECK(                                                        \
              deferred_workspace_size == 0,                                   \
              #aclnn_api " requires zero ACLNN workspace for the deferred ",\
              "tensor-owned launch path; got ", deferred_workspace_size);  \
          LogOpApiLifecycle(#aclnn_api, "consumer_workspace_ready");        \
          auto deferred_launch_fn =                                           \
              reinterpret_cast<OpApiLaunchFn>(deferred_launch_addr_fn);      \
          LogOpApiLifecycle(#aclnn_api, "launch_begin");                    \
          const int deferred_launch_status =                                  \
              deferred_launch_fn(nullptr, deferred_workspace_size,            \
                                 deferred_executor, deferred_acl_stream);     \
          TORCH_CHECK(                                                        \
              deferred_launch_status == 0,                                    \
              "call " #aclnn_api " failed: ", RecentAclError());           \
          LogOpApiLifecycle(#aclnn_api, "launch_done");                     \
          RecordTensorTupleOnNpuStream(                                       \
              deferred_args, deferred_npu_stream);                            \
          LogOpApiLifecycle(#aclnn_api, "tensor_lifetime_recorded");        \
          ReleaseTuple(deferred_converted);                                   \
          deferred_converted_released = true;                                 \
          LogOpApiLifecycle(#aclnn_api, "descriptor_release_done");         \
          /* `deferred_args` is the only P8 Tensor capture. After stream     \
           * recording, clear it on the launch consumer instead of leaving   \
           * TensorImpl destruction to torch_npu's asynchronous ReleaseQueue. */\
          ClearTensorTuple(deferred_args);                                    \
          LogOpApiLifecycle(#aclnn_api, "tensor_keepalive_released");       \
          deferred_cleanup_consumer_tls();                                    \
          return deferred_launch_status;                                      \
        } catch (...) {                                                        \
          if (!deferred_converted_released) {                                 \
            ReleaseTuple(deferred_converted);                                 \
          }                                                                    \
          throw;                                                               \
        }                                                                      \
      } catch (...) {                                                          \
        deferred_cleanup_consumer_tls();                                      \
        NotifyOrderedLaunchFailureTuple(deferred_tensor_keepalive);           \
        LogOpApiLifecycle(#aclnn_api, "consumer_failed_quarantined");       \
        throw;                                                                 \
      }                                                                        \
    };                                                                         \
    LogOpApiLifecycle(#aclnn_api, "handoff_runopapi_tensor_owned_deferred");\
    at_npu::native::OpCommand::RunOpApi(#aclnn_api, deferred_acl_call);       \
    /* The producer never owns ACLNN TLS or descriptors on this path. */     \
    LogOpApiLifecycle(#aclnn_api, "submit_done");                            \
  } while (false)

// The common form derives the signature from the converted ACL descriptor
// tuple.  It remains the right choice for operators whose source tree does
// not ship an OPP-generated public ACLNN header.
#define EXEC_NPU_CMD_TENSOR_OWNED_DEFERRED_ZERO_WORKSPACE(                    \
    aclnn_api, tensor_keepalive, ...)                                         \
  EXEC_NPU_CMD_TENSOR_OWNED_DEFERRED_ZERO_WORKSPACE_IMPL(                     \
      aclnn_api,                                                              \
      WorkspaceFunction(deferred_converted, deferred_workspace_addr_fn),     \
      tensor_keepalive, __VA_ARGS__)

// Operators with a freshly generated ACLNN header may bind the exact C ABI
// instead of recreating it from a tuple.  This is intentionally an opt-in
// narrow variant: it keeps existing bridges source-compatible while making
// a generated signature mismatch a compile-time failure in the new bridge.
// `workspace_function_type` should be
// `decltype(&aclnn<Op>GetWorkspaceSize)` from that generated header.
#define EXEC_NPU_CMD_TENSOR_OWNED_DEFERRED_ZERO_WORKSPACE_TYPED(              \
    aclnn_api, workspace_function_type, tensor_keepalive, ...)               \
  EXEC_NPU_CMD_TENSOR_OWNED_DEFERRED_ZERO_WORKSPACE_IMPL(                     \
      aclnn_api,                                                              \
      reinterpret_cast<workspace_function_type>(deferred_workspace_addr_fn), \
      tensor_keepalive, __VA_ARGS__)

// P8-only caller-thread counterpart of the deferred launcher above.  It is
// intentionally a separate macro rather than a mode switch in the generic
// P7 launcher: P7 still needs task-queue ownership, whereas P8 must not place
// a bridge closure into torch_npu's private ReleaseQueue.  The only allowed
// ABI is value-owned at::Tensor arguments and zero ACLNN workspace.
//
// This TQ0 acceptance macro uses `NPUStream::stream(false)`: the raw current
// stream is safe only because every graph node is submitted directly by the
// same caller.  It preserves FIFO stream order and keeps device launch
// asynchronous; it never calls aclrtSynchronizeStream.
//
// It must not call RunOpApi.  That API transfers its handler to torch_npu's
// private ReleaseQueue after execution; a bridge-local std::function can then
// survive Python/DSO teardown even after torch.npu.synchronize() returns.
#define EXEC_NPU_CMD_TENSOR_OWNED_DRAINED_DIRECT_ASYNC_ZERO_WORKSPACE_IMPL(    \
    aclnn_api, workspace_function_expression, ...)                             \
  do {                                                                          \
    using namespace nanovllm_a5_opapi;                                         \
    static void *inline_workspace_addr_fn =                                    \
        ResolveSymbol(#aclnn_api "GetWorkspaceSize");                         \
    static void *inline_launch_addr_fn = ResolveSymbol(#aclnn_api);            \
    TORCH_CHECK(                                                                \
        inline_workspace_addr_fn != nullptr && inline_launch_addr_fn != nullptr,\
        #aclnn_api,                                                             \
        " or ",                                                                \
        #aclnn_api "GetWorkspaceSize is missing from repository-local ",      \
        "libcust_opapi.so");                                                  \
    LogResolvedOpApi(#aclnn_api, inline_workspace_addr_fn,                     \
                     inline_launch_addr_fn);                                   \
    auto inline_npu_stream = c10_npu::getCurrentNPUStream();                   \
    RequireP8Tq0DirectAsyncOpApiContract();                                    \
    aclrtStream inline_acl_stream = inline_npu_stream.stream(false);           \
    TORCH_CHECK(inline_acl_stream != nullptr,                                  \
                #aclnn_api " received a null current ACL stream");            \
    /* P8 owns no CANN stream-resource quota.  Its scheduler is the device    \
     * queue lease, not a caller-thread AIV binding: binding a quota here      \
     * would both revive the rejected fixed-core design and import             \
     * torch_npu's non-exported UseStreamResInCurrentThread symbol into the    \
     * bridge DSO.  The existing stream remains intact; P8 neither sets,       \
     * binds, resets, nor unbinds stream resources. */                         \
    LogOpApiLifecycle(#aclnn_api, "caller_thread_tq0_stream_ready");         \
    /* std::make_tuple decays every at::Tensor argument to value ownership. */ \
    auto inline_args = std::make_tuple(__VA_ARGS__);                            \
    auto inline_acl_call =                                                      \
        [inline_args = std::move(inline_args),                                 \
         inline_npu_stream, inline_acl_stream]() mutable -> int {             \
      bool inline_pta_cache_active = false;                                    \
      bool inline_huge_mem_active = false;                                     \
      auto inline_init_huge_mem =                                               \
          reinterpret_cast<InitHugeMemThreadLocalFn>(                          \
              ResolveSymbol("InitHugeMemThreadLocal"));                       \
      auto inline_uninit_huge_mem =                                             \
          reinterpret_cast<UnInitHugeMemThreadLocalFn>(                        \
              ResolveSymbol("UnInitHugeMemThreadLocal"));                     \
      auto inline_release_huge_mem =                                           \
          reinterpret_cast<ReleaseHugeMemFn>(ResolveSymbol("ReleaseHugeMem"));\
      auto inline_cleanup_host_tls = [&]() noexcept {                          \
        if (inline_huge_mem_active) {                                          \
          if (inline_release_huge_mem != nullptr) {                            \
            inline_release_huge_mem(nullptr, false);                           \
          }                                                                     \
          if (inline_uninit_huge_mem != nullptr) {                             \
            inline_uninit_huge_mem(nullptr, false);                            \
          }                                                                     \
          inline_huge_mem_active = false;                                      \
        }                                                                       \
        if (inline_pta_cache_active) {                                         \
          UnInitPtaCacheForWorkspace(true);                                    \
          inline_pta_cache_active = false;                                     \
        }                                                                       \
        LogOpApiLifecycle(#aclnn_api, "caller_thread_tls_uninit");           \
      };                                                                        \
      try {                                                                     \
        /* The pinned torch_npu ABI exposes no public deterministic-state     \
         * snapshot/apply API.  Do not emulate one with private symbols: P8's \
         * deterministic behavior comes from its graph-visible control and    \
         * state tensors, while this raw launch preserves the existing stream \
         * ordering without mutating process-global framework state. */       \
        inline_pta_cache_active = InitPtaCacheForWorkspace();                  \
        LogOpApiLifecycle(                                                      \
            #aclnn_api,                                                         \
            inline_pta_cache_active ? "caller_thread_pta_cache_ready"        \
                                    : "caller_thread_pta_cache_unavailable"); \
        LogOrderedHostDso(#aclnn_api, inline_workspace_addr_fn,                \
                          inline_launch_addr_fn, inline_init_huge_mem,         \
                          inline_release_huge_mem,                              \
                          inline_uninit_huge_mem);                              \
        if (inline_init_huge_mem != nullptr) {                                 \
          (void)inline_init_huge_mem(nullptr, false);                          \
          inline_huge_mem_active = true;                                       \
          LogOpApiLifecycle(#aclnn_api, "caller_thread_huge_mem_ready");     \
        }                                                                       \
        uint64_t inline_workspace_size = 0;                                    \
        uint64_t *inline_workspace_size_ptr = &inline_workspace_size;          \
        aclOpExecutor *inline_executor = nullptr;                              \
        aclOpExecutor **inline_executor_ptr = &inline_executor;                \
        auto inline_converted = std::apply(                                    \
            [&](auto &...inline_args_ref) {                                    \
              return ConvertTypes(inline_args_ref...,                          \
                                  inline_workspace_size_ptr,                    \
                                  inline_executor_ptr);                         \
            },                                                                  \
            inline_args);                                                       \
        LogConvertedDescriptorSet(#aclnn_api, inline_converted);               \
        bool inline_converted_released = false;                                \
        try {                                                                   \
          auto inline_workspace_fn = workspace_function_expression;            \
          LogOpApiLifecycle(#aclnn_api, "caller_thread_workspace_fn_ready"); \
          LogOpApiLifecycle(#aclnn_api, "caller_thread_workspace_call_begin");\
          const int inline_workspace_status =                                  \
              CallTuple(inline_workspace_fn, inline_converted);                \
          LogOpApiLifecycle(#aclnn_api, "caller_thread_workspace_call_end");\
          TORCH_CHECK(                                                         \
              inline_workspace_status == 0,                                    \
              "call " #aclnn_api "GetWorkspaceSize failed: ",               \
              RecentAclError());                                               \
          TORCH_CHECK(                                                         \
              inline_workspace_size == 0,                                      \
              #aclnn_api " requires zero ACLNN workspace for the P8 inline ",\
              "asynchronous launch path; got ", inline_workspace_size);      \
          LogOpApiLifecycle(#aclnn_api, "caller_thread_workspace_ready");   \
          auto inline_launch_fn =                                              \
              reinterpret_cast<OpApiLaunchFn>(inline_launch_addr_fn);          \
          LogOpApiLifecycle(#aclnn_api, "launch_begin");                     \
          const int inline_launch_status =                                     \
              inline_launch_fn(nullptr, inline_workspace_size,                 \
                               inline_executor, inline_acl_stream);            \
          TORCH_CHECK(                                                         \
              inline_launch_status == 0,                                       \
              "call " #aclnn_api " failed: ", RecentAclError());            \
          LogOpApiLifecycle(#aclnn_api, "launch_done");                      \
          ReleaseTuple(inline_converted);                                      \
          inline_converted_released = true;                                    \
          LogOpApiLifecycle(#aclnn_api, "descriptor_release_done");          \
          /* TQ0 dispatches on this caller thread.  It does not use          \
           * RunOpApi or torch_npu's delayed ReleaseQueue, so it must not     \
           * reuse the deferred bridge's allocator recordStream workaround.   \
           * That extra same-stream retirement edge can corrupt the           \
           * direct-launch synchronize path.  The graph owns P8 input/output  \
           * tensors; drop only this local duplicate tuple before TLS teardown \
           * so its TensorImpl destructors cannot run after HugeMem/PTA is     \
           * dismantled. */                                                    \
          ClearTensorTuple(inline_args);                                       \
          LogOpApiLifecycle(#aclnn_api, "caller_thread_tensor_captures_released"); \
          inline_cleanup_host_tls();                                            \
          return inline_launch_status;                                         \
        } catch (...) {                                                         \
          if (!inline_converted_released) {                                    \
            ReleaseTuple(inline_converted);                                    \
          }                                                                     \
          throw;                                                                \
        }                                                                       \
      } catch (...) {                                                           \
        inline_cleanup_host_tls();                                              \
        LogOpApiLifecycle(#aclnn_api, "caller_thread_failed_quarantined");  \
        throw;                                                                  \
      }                                                                         \
    };                                                                          \
    /* Do not delegate `inline_acl_call` to RunOpApi: doing so would enqueue  \
     * this bridge closure in torch_npu's private ReleaseQueue.  This direct  \
     * call retains the caller's GIL only through the host handler; the ACL   \
     * launch above remains asynchronous on the NPU. */                         \
    LogOpApiLifecycle(#aclnn_api, "caller_thread_direct_dispatch_begin");    \
    const int inline_direct_status =                                           \
        RunP8Tq0CallerThreadHost(std::move(inline_acl_call));                  \
    TORCH_CHECK(inline_direct_status == 0,                                    \
                "call " #aclnn_api " direct host dispatch failed: ",          \
                RecentAclError());                                            \
    LogOpApiLifecycle(#aclnn_api, "caller_thread_direct_dispatch_done");     \
    LogOpApiLifecycle(#aclnn_api, "submit_done");                             \
  } while (false)

#define EXEC_NPU_CMD_TENSOR_OWNED_DRAINED_DIRECT_ASYNC_ZERO_WORKSPACE(        \
    aclnn_api, ...)                                                            \
  EXEC_NPU_CMD_TENSOR_OWNED_DRAINED_DIRECT_ASYNC_ZERO_WORKSPACE_IMPL(          \
      aclnn_api,                                                               \
      WorkspaceFunction(inline_converted, inline_workspace_addr_fn),          \
      __VA_ARGS__)

#define EXEC_NPU_CMD_TENSOR_OWNED_DRAINED_DIRECT_ASYNC_ZERO_WORKSPACE_TYPED(  \
    aclnn_api, workspace_function_type, ...)                                  \
  EXEC_NPU_CMD_TENSOR_OWNED_DRAINED_DIRECT_ASYNC_ZERO_WORKSPACE_IMPL(          \
      aclnn_api,                                                               \
      reinterpret_cast<workspace_function_type>(inline_workspace_addr_fn),   \
      __VA_ARGS__)
