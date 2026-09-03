// P10 caller-thread ACLNN submission for graph nodes.
//
// This is intentionally separate from the historical framework task-queue
// launchers.  P10 has exactly three coarse graph-node submissions per forward
// (prepare, native W4A4C8 producer, resident executor), not a host submission
// per candidate.  Submit each on the current caller-selected device stream,
// record every tensor storage on that stream, then return without a host wait.
// The bridge owns ACL tensor descriptors and the optional HugeMem scope, but
// it deliberately does *not* initialize or tear down torch_npu's PTA TLS:
// direct caller-stream dispatch is already inside the framework's allocator
// ownership domain.  Taking PTA ownership here would reproduce the prior
// post-submit teardown race.  This closes the descriptor/HugeMem and tensor
// lifetime boundaries without importing an old queue callback or a
// physical-core policy.

#pragma once

#include "op_api_common.h"

// P10 deliberately has no TQ1 fallback. A raw ACLNN caller-stream launch
// bypasses torch_npu's task queue, so framework-order ownership is proven only
// with task queue disabled before the process imports torch_npu. The product
// runner emits the same compact receipt before it starts Python; this repeated
// C++ gate prevents an embedded caller from silently changing that contract.
inline void RequireP10DirectDispatchContract()
{
    const char* task_queue = std::getenv("TASK_QUEUE_ENABLE");
    const char* launch_blocking = std::getenv("ASCEND_LAUNCH_BLOCKING");
    TORCH_CHECK(
        task_queue != nullptr && std::strcmp(task_queue, "0") == 0 &&
            (launch_blocking == nullptr || std::strcmp(launch_blocking, "0") == 0),
        "P10 direct caller-stream dispatch requires TASK_QUEUE_ENABLE=0 and "
        "ASCEND_LAUNCH_BLOCKING=0 before torch_npu initialization; "
        "TASK_QUEUE_ENABLE=", task_queue == nullptr ? "<unset>" : task_queue,
        " ASCEND_LAUNCH_BLOCKING=",
        launch_blocking == nullptr ? "<unset>" : launch_blocking);
    if (nanovllm_a5_opapi::OpApiDebugEnabled()) {
        std::fprintf(
            stderr,
            "P10_DIRECT_DISPATCH_CONFIG task_queue_enable=0 "
            "launch_blocking=%s owner=caller_stream "
            "task_queue_fallback=forbidden\n",
            launch_blocking == nullptr ? "<unset>" : launch_blocking);
        std::fflush(stderr);
    }
}

#define P10_EXEC_NPU_CMD_DIRECT(aclnn_api, tensor_keepalive, ...)             \
  do {                                                                          \
    using namespace nanovllm_a5_opapi;                                         \
    RequireP10DirectDispatchContract();                                         \
    static void *p10_workspace_addr_fn =                                       \
        ResolveSymbol(#aclnn_api "GetWorkspaceSize");                         \
    static void *p10_launch_addr_fn = ResolveSymbol(#aclnn_api);               \
    TORCH_CHECK(                                                                \
        p10_workspace_addr_fn != nullptr && p10_launch_addr_fn != nullptr,     \
        #aclnn_api,                                                             \
        " or ",                                                                \
        #aclnn_api "GetWorkspaceSize is missing from the exact P10 product " \
                   "libcust_opapi.so");                                      \
    LogResolvedOpApi(#aclnn_api, p10_workspace_addr_fn,                         \
                     p10_launch_addr_fn);                                      \
    auto p10_stream = c10_npu::getCurrentNPUStream();                           \
    aclrtStream p10_acl_stream = p10_stream.stream(false);                     \
    TORCH_CHECK(p10_acl_stream != nullptr,                                     \
                #aclnn_api " received a null current NPU stream");           \
    auto p10_keepalive = tensor_keepalive;                                     \
    auto p10_init_huge = reinterpret_cast<InitHugeMemThreadLocalFn>(           \
        ResolveSymbol("InitHugeMemThreadLocal"));                              \
    auto p10_uninit_huge = reinterpret_cast<UnInitHugeMemThreadLocalFn>(       \
        ResolveSymbol("UnInitHugeMemThreadLocal"));                            \
    auto p10_release_huge = reinterpret_cast<ReleaseHugeMemFn>(                \
        ResolveSymbol("ReleaseHugeMem"));                                      \
    bool p10_huge_active = false;                                              \
    auto p10_cleanup_huge_mem = [&]() noexcept {                               \
      if (p10_huge_active) {                                                   \
        if (p10_release_huge != nullptr) p10_release_huge(nullptr, false);     \
        if (p10_uninit_huge != nullptr) p10_uninit_huge(nullptr, false);       \
        p10_huge_active = false;                                               \
      }                                                                         \
    };                                                                          \
    if (p10_init_huge != nullptr) {                                            \
      (void)p10_init_huge(nullptr, false);                                     \
      p10_huge_active = true;                                                  \
    }                                                                           \
    try {                                                                       \
      uint64_t p10_workspace_size = 0;                                         \
      aclOpExecutor *p10_executor = nullptr;                                  \
      uint64_t *p10_workspace_size_ptr = &p10_workspace_size;                  \
      aclOpExecutor **p10_executor_ptr = &p10_executor;                        \
      auto p10_converted = ConvertTypes(                                       \
          __VA_ARGS__, p10_workspace_size_ptr, p10_executor_ptr);              \
      bool p10_converted_released = false;                                     \
      try {                                                                     \
        auto p10_workspace_fn = WorkspaceFunction(                             \
            p10_converted, p10_workspace_addr_fn);                             \
        const int p10_workspace_status = CallTuple(                            \
            p10_workspace_fn, p10_converted);                                  \
        TORCH_CHECK(p10_workspace_status == 0,                                \
                    "call " #aclnn_api "GetWorkspaceSize failed: ",         \
                    RecentAclError());                                          \
        at::Tensor p10_workspace;                                              \
        void *p10_workspace_ptr = nullptr;                                     \
        if (p10_workspace_size != 0) {                                         \
          const auto &p10_anchor = std::get<0>(p10_keepalive);                 \
          p10_workspace = at::empty(                                           \
              {static_cast<int64_t>(p10_workspace_size)},                      \
              p10_anchor.options().dtype(at::kByte));                          \
          p10_workspace_ptr = p10_workspace.data_ptr();                        \
        }                                                                       \
        auto p10_launch_fn = reinterpret_cast<OpApiLaunchFn>(p10_launch_addr_fn);\
        const int p10_launch_status = p10_launch_fn(                           \
            p10_workspace_ptr, p10_workspace_size, p10_executor, p10_acl_stream);\
        TORCH_CHECK(p10_launch_status == 0,                                    \
                    "call " #aclnn_api " failed: ", RecentAclError());      \
        RecordTensorTupleOnNpuStream(p10_keepalive, p10_stream);               \
        RecordTensorStorageOnNpuStream(p10_workspace, p10_stream);             \
        ReleaseTuple(p10_converted);                                           \
        p10_converted_released = true;                                         \
        ClearTensorTuple(p10_keepalive);                                       \
      } catch (...) {                                                          \
        if (!p10_converted_released) ReleaseTuple(p10_converted);              \
        throw;                                                                 \
      }                                                                         \
      p10_cleanup_huge_mem();                                                  \
    } catch (...) {                                                            \
      p10_cleanup_huge_mem();                                                  \
      throw;                                                                   \
    }                                                                           \
  } while (false)
