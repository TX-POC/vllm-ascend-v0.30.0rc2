#pragma once

// When A5 DSA registrations are embedded in the main vllm_ascend_C module,
// route their ACLNN launches through vLLM-Ascend's normal op-api adapter.
//
// The normal adapter snapshots ASCEND_CUSTOM_OPP_PATH in namespace-scope
// globals when a translation unit is loaded.  A5 DSA may be initialized after
// another vLLM-Ascend path has already imported vllm_ascend_C, so relying only
// on that snapshot can make packaged A5 libcust_opapi.so invisible even though
// a5_extension.py publishes the correct path before the first DSA invocation.
//
// Keep all tensor conversion/workspace/stream behavior from the native
// vLLM-Ascend adapter, but wrap only its symbol resolver for these A5 binding
// translation units.  The wrapper checks the explicit packaged A5 op-api and
// the *current* ASCEND_CUSTOM_OPP_PATH on every first API resolution, then
// falls back to the unmodified vLLM-Ascend resolver.

#define GetOpApiFuncAddr VllmAscendBaseGetOpApiFuncAddr
#include "aclnn_torch_adapter/op_api_common.h"
#undef GetOpApiFuncAddr

inline void *GetOpApiFuncAddr(const char *apiName)
{
    // a5_extension.py sets this to the exact packaged A5 libcust_opapi.so.
    // Prefer it so A5 DSA resolution is independent of import ordering.
    const char *explicitOpApi = std::getenv("NANOVLLM_CUST_OPAPI_LIB");
    if (explicitOpApi != nullptr && explicitOpApi[0] != '\0') {
        const auto explicitPath = real_path(explicitOpApi);
        if (!explicitPath.empty()) {
            auto handler = GetOpApiLibHandler(explicitPath.c_str());
            if (handler != nullptr) {
                auto funcAddr = GetOpApiFuncAddrInLib(
                    handler, GetCustOpApiLibName(), apiName);
                if (funcAddr != nullptr) {
                    return funcAddr;
                }
            }
        }
    }

    // Re-read the environment instead of using g_custom_lib_path, which is a
    // load-time snapshot in the original adapter.
    const auto liveCustomLibPaths = get_custom_lib_path();
    for (const auto &libDir : liveCustomLibPaths) {
        const auto customOpApi =
            real_path(libDir + "/" + GetCustOpApiLibName());
        if (customOpApi.empty()) {
            continue;
        }
        auto handler = GetOpApiLibHandler(customOpApi.c_str());
        if (handler == nullptr) {
            continue;
        }
        auto funcAddr = GetOpApiFuncAddrInLib(
            handler, GetCustOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }

    return VllmAscendBaseGetOpApiFuncAddr(apiName);
}

// The standalone nanovllm operator-development extension still includes its
// local op_api_common.h directly and therefore keeps its self-contained
// launcher.  In the main package build we include that header here once, then
// replace only EXEC_NPU_CMD_ORDERED.  Because the local header is guarded by
// #pragma once, the later include from ops.cpp/full_block_dump.cpp is skipped
// in this translation unit.
#include "attention/nanovllm_a5_ops/torch_extension/csrc/op_api_common.h"

#undef EXEC_NPU_CMD_ORDERED
#define EXEC_NPU_CMD_ORDERED(aclnn_api, tensor_keepalive, ...) \
  do {                                                         \
    (void)(tensor_keepalive);                                  \
    EXEC_NPU_CMD(aclnn_api, __VA_ARGS__);                      \
  } while (false)
