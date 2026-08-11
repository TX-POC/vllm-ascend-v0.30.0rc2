#ifndef NANOVLLM_DSA_A5_OPS_LOG_COMPAT_H
#define NANOVLLM_DSA_A5_OPS_LOG_COMPAT_H

#include <cstdio>

// The upstream vLLM-Ascend tiler uses logging helpers from its monorepo build.
// Keep this standalone operator project independent from those private headers.
#ifndef OP_LOGI
#define OP_LOGI(op_name, ...) \
    do {                       \
        (void)(op_name);       \
    } while (0)
#endif

#ifndef OP_LOGW
#define OP_LOGW(op_name, ...)          \
    do {                               \
        (void)(op_name);               \
        std::fprintf(stderr, "[WARN] "); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n");    \
    } while (0)
#endif

#ifndef OP_LOGE
#define OP_LOGE(op_name, ...)           \
    do {                                \
        (void)(op_name);                \
        std::fprintf(stderr, "[ERROR] "); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n");     \
    } while (0)
#endif

#ifndef OPS_REPORT_VECTOR_INNER_ERR
#define OPS_REPORT_VECTOR_INNER_ERR(op_name, ...) \
    OP_LOGE(op_name, __VA_ARGS__)
#endif

#ifndef OP_CHECK_IF
#define OP_CHECK_IF(condition, log_statement, action) \
    do {                                               \
        if (condition) {                               \
            log_statement;                             \
            action;                                    \
        }                                              \
    } while (0)
#endif

// Some reference kernels use the older OPS_* spellings from
// error/ops_error.h.  That private header is not shipped by every CANN 9.1
// Ascend 950 package, so keep the small subset needed by this standalone
// project local.
#ifndef OPS_LOG_E
#define OPS_LOG_E(op_name, ...) OP_LOGE(op_name, __VA_ARGS__)
#endif

#ifndef OPS_ERR_IF
#define OPS_ERR_IF(condition, log_statement, action) \
    OP_CHECK_IF(condition, log_statement, action)
#endif

#ifndef OPS_LOG_E_IF_NULL
#define OPS_LOG_E_IF_NULL(context, value, action)              \
    do {                                                       \
        if ((value) == nullptr) {                              \
            OP_LOGE(context, "%s is nullptr.", #value);      \
            action;                                            \
        }                                                      \
    } while (0)
#endif

#endif
