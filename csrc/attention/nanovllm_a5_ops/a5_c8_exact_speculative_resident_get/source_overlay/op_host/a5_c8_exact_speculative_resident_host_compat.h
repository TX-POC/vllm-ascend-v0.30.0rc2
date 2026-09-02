// Isolated P9 host build compatibility surface.
//
// CANN's msopgen project does not inherit the repository-wide err/log include
// graph used by the source-derived C8 host code.  These diagnostics-only
// definitions preserve all original failure branches without importing an
// unrelated operator closure or changing C8 tiling/math.

#ifndef VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_HOST_COMPAT_H_
#define VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_HOST_COMPAT_H_

#include <cstdio>

#ifndef OP_LOGI
#define OP_LOGI(op_name, ...) \
    do {                      \
        (void)(op_name);      \
    } while (0)
#endif

#ifndef OP_LOGD
#define OP_LOGD(op_name, ...) \
    do {                      \
        (void)(op_name);      \
    } while (0)
#endif

#ifndef OP_LOGW
#define OP_LOGW(op_name, ...)               \
    do {                                    \
        (void)(op_name);                    \
        std::fprintf(stderr, "[WARN] ");   \
        std::fprintf(stderr, __VA_ARGS__);  \
        std::fputc('\n', stderr);          \
    } while (0)
#endif

#ifndef OP_LOGE_WITHOUT_REPORT
#define OP_LOGE_WITHOUT_REPORT(op_name, ...) \
    do {                                      \
        (void)(op_name);                      \
        std::fprintf(stderr, "[ERRORx] ");  \
        std::fprintf(stderr, __VA_ARGS__);    \
        std::fputc('\n', stderr);            \
    } while (0)
#endif

#ifndef OP_LOGE
#define OP_LOGE(op_name, ...)                \
    do {                                     \
        (void)(op_name);                     \
        std::fprintf(stderr, "[ERROR] ");  \
        std::fprintf(stderr, __VA_ARGS__);   \
        std::fputc('\n', stderr);           \
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

#ifndef OP_CHECK_NULL_WITH_CONTEXT
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)             \
    do {                                                      \
        (void)(context);                                      \
        if ((ptr) == nullptr) {                               \
            OP_LOGE("A5C8ExactSpeculativeResidentGet",       \
                    "%s is null", #ptr);                    \
            return ge::GRAPH_FAILED;                          \
        }                                                     \
    } while (0)
#endif

#ifndef REPORT_INNER_ERR_MSG
#define REPORT_INNER_ERR_MSG(err_code, ...)                   \
    do {                                                      \
        (void)(err_code);                                     \
        std::fprintf(stderr, "[P9][REPORT] ");               \
        std::fprintf(stderr, __VA_ARGS__);                    \
        std::fputc('\n', stderr);                            \
    } while (0)
#endif

#ifndef OpLogSub
#define OpLogSub(module, level, op_name, ...)                 \
    do {                                                      \
        (void)(module);                                       \
        (void)(level);                                        \
        (void)(op_name);                                      \
        std::fprintf(stderr, "[P9][ERROR] ");                \
        std::fprintf(stderr, __VA_ARGS__);                    \
        std::fputc('\n', stderr);                            \
    } while (0)
#endif

#ifndef OPS_INNER_ERR_STUB
#define OPS_INNER_ERR_STUB(err_code, op_name, ...)            \
    do {                                                      \
        OpLogSub(OP, DLOG_ERROR, op_name, __VA_ARGS__);       \
        REPORT_INNER_ERR_MSG(err_code, __VA_ARGS__);          \
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
#define OPS_LOG_E_IF_NULL(context, value, action)             \
    do {                                                      \
        if ((value) == nullptr) {                             \
            OP_LOGE(context, "%s is null", #value);         \
            action;                                           \
        }                                                     \
    } while (0)
#endif

#endif  // VLLM_ASCEND_A5_C8_EXACT_SPECULATIVE_RESIDENT_HOST_COMPAT_H_
