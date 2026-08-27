/**
 * Standalone host-side compatibility layer for A5StreamingQuantLiGet.
 *
 * The normal vLLM-Ascend build supplies err/ops_err.h and log/log.h through
 * its repository-wide include graph.  CANN's msopgen project deliberately
 * does not inherit that graph.  This header is staged under both include
 * paths by build_nanovllm_ops.sh, so the isolated OPP keeps the same host
 * validation control flow without importing unrelated project headers.
 *
 * It is intentionally limited to diagnostics and guard macros.  It does not
 * alter the official C8 score calculation, tiling decisions, Top-K ordering,
 * or any device-side KVAF operation.
 */

#ifndef NANOVLLM_A5_STREAMING_QUANT_LI_GET_HOST_COMPAT_H
#define NANOVLLM_A5_STREAMING_QUANT_LI_GET_HOST_COMPAT_H

#include <cstdio>

// Match the upstream host-op contract: information/debug logs are silent in
// the isolated build, while warnings and errors remain visible in build/test
// logs.  Keeping INFO silent avoids flooding logs from the source-derived
// tiler without weakening an error return path.
#ifndef OP_LOGI
#define OP_LOGI(op_name, ...) \
    do {                       \
        (void)(op_name);       \
    } while (0)
#endif

#ifndef OP_LOGD
#define OP_LOGD(op_name, ...) \
    do {                       \
        (void)(op_name);       \
    } while (0)
#endif

#ifndef OP_LOGW
#define OP_LOGW(op_name, ...)           \
    do {                                \
        (void)(op_name);                \
        std::fprintf(stderr, "[WARN] "); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fputc('\n', stderr);      \
    } while (0)
#endif

#ifndef OP_LOGE_WITHOUT_REPORT
#define OP_LOGE_WITHOUT_REPORT(op_name, ...) \
    do {                                        \
        (void)(op_name);                        \
        std::fprintf(stderr, "[ERRORx] ");     \
        std::fprintf(stderr, __VA_ARGS__);      \
        std::fputc('\n', stderr);              \
    } while (0)
#endif

#ifndef OP_LOGE
#define OP_LOGE(op_name, ...)          \
    do {                               \
        (void)(op_name);               \
        std::fprintf(stderr, "[ERROR] "); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fputc('\n', stderr);     \
    } while (0)
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

#ifndef OP_TILING_CHECK
#define OP_TILING_CHECK(condition, log_statement, action) \
    OP_CHECK_IF(condition, log_statement, action)
#endif

// The source-derived C8 infer-shape path uses this macro as an early return.
// Keep that return exactly: only the unavailable monorepo logger is replaced.
#ifndef OP_CHECK_NULL_WITH_CONTEXT
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)       \
    do {                                                \
        (void)(context);                                \
        if ((ptr) == nullptr) {                         \
            OP_LOGE("A5StreamingQuantLiGet", "%s is null", #ptr); \
            return ge::GRAPH_FAILED;                    \
        }                                               \
    } while (0)
#endif

// These aliases close the error-reporting surface used by the official C8
// host sources.  The current P3 files only need the OP_* guards above, but
// keeping the corresponding OPS_* spelling self-contained prevents a future
// source-derived C8 update from pulling in another private logging header.
#ifndef REPORT_INNER_ERR_MSG
#define REPORT_INNER_ERR_MSG(err_code, ...)       \
    do {                                           \
        (void)(err_code);                          \
        std::fprintf(stderr, "[P3][REPORT] ");    \
        std::fprintf(stderr, __VA_ARGS__);         \
        std::fputc('\n', stderr);                 \
    } while (0)
#endif

#ifndef OpLogSub
#define OpLogSub(module, level, op_name, ...)      \
    do {                                           \
        (void)(op_name);                           \
        std::fprintf(stderr, "[P3][ERROR] ");     \
        std::fprintf(stderr, __VA_ARGS__);         \
        std::fputc('\n', stderr);                 \
    } while (0)
#endif

#ifndef OPS_INNER_ERR_STUB
#define OPS_INNER_ERR_STUB(err_code, op_name, ...)             \
    do {                                                        \
        OpLogSub(OP, DLOG_ERROR, op_name, __VA_ARGS__);        \
        REPORT_INNER_ERR_MSG(err_code, __VA_ARGS__);           \
    } while (0)
#endif

#ifndef OPS_REPORT_VECTOR_INNER_ERR
#define OPS_REPORT_VECTOR_INNER_ERR(op_name, ...) \
    OPS_INNER_ERR_STUB("E89999", op_name, __VA_ARGS__)
#endif

#ifndef OPS_REPORT_CUBE_INNER_ERR
#define OPS_REPORT_CUBE_INNER_ERR(op_name, ...) \
    OPS_INNER_ERR_STUB("E69999", op_name, __VA_ARGS__)
#endif

#ifndef VECTOR_INNER_ERR_REPORT_TILIING
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__)
#endif

#ifndef OPS_LOG_E
#define OPS_LOG_E(op_name, ...) OP_LOGE(op_name, __VA_ARGS__)
#endif

#ifndef OPS_ERR_IF
#define OPS_ERR_IF(condition, log_statement, action) \
    OP_CHECK_IF(condition, log_statement, action)
#endif

#ifndef OP_CHECK
#define OP_CHECK(condition, log_statement, action) \
    OP_CHECK_IF(condition, log_statement, action)
#endif

#ifndef OPS_LOG_E_IF_NULL
#define OPS_LOG_E_IF_NULL(context, value, action)      \
    do {                                                \
        if ((value) == nullptr) {                       \
            OP_LOGE(context, "%s is null", #value);   \
            action;                                     \
        }                                               \
    } while (0)
#endif

#endif  // NANOVLLM_A5_STREAMING_QUANT_LI_GET_HOST_COMPAT_H
