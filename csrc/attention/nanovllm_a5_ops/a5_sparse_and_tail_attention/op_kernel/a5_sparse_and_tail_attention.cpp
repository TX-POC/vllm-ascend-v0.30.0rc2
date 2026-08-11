/**
 * Ascend 950 sparse-and-tail attention kernel entry.
 */

#include "kernel_operator.h"
#include "a5_sparse_and_tail_attention_template_tiling_key.h"
#include "a5_sfa/arch35/sparse_flash_attention_kernel_mla.h"

using namespace AscendC;

#if defined(__DAV_C310_CUBE__)
#define A5_SPARSE_TAIL_IMPL(...)                                                  \
    do {                                                                          \
        using CubeBlockType = typename std::conditional<                           \
            g_coreType == AscendC::AIC,                                            \
            BaseApi::SFAMatmulService<__VA_ARGS__>,                                \
            BaseApi::SFAMatmulServiceDummy<__VA_ARGS__>>::type;                    \
        using VecBlockType = typename std::conditional<                            \
            g_coreType == AscendC::AIC,                                            \
            BaseApi::SFAVectorServiceDummy<__VA_ARGS__>,                           \
            BaseApi::SFAVectorService<__VA_ARGS__>>::type;                         \
        BaseApi::SparseFlashAttentionKernelMla<                                    \
            CubeBlockType, VecBlockType> op;                                       \
        GET_TILING_DATA_WITH_STRUCT(                                               \
            SparseFlashAttentionTilingDataMla, tilingDataIn, tiling);              \
        op.Init(                                                                   \
            query, key, value, sparseIndices, cacheTokens,                         \
            actualSeqLengthsQuery, actualSeqLengthsKv, blockTable,                 \
            queryRope, keyRope, attentionOut, softmaxMax, softmaxSum,              \
            userWorkspace, nullptr, tiling, &pipe);                                \
        op.Process();                                                              \
    } while (0)
#else
#define A5_SPARSE_TAIL_IMPL(...)                                                  \
    do {                                                                          \
        using CubeBlockType = typename std::conditional<                           \
            g_coreType == AscendC::AIC,                                            \
            BaseApi::SFAMatmulService<__VA_ARGS__>,                                \
            BaseApi::SFAMatmulServiceDummy<__VA_ARGS__>>::type;                    \
        using VecBlockType = typename std::conditional<                            \
            g_coreType == AscendC::AIC,                                            \
            BaseApi::SFAVectorServiceDummy<__VA_ARGS__>,                           \
            BaseApi::SFAVectorService<__VA_ARGS__>>::type;                         \
        BaseApi::SparseFlashAttentionKernelMla<                                    \
            CubeBlockType, VecBlockType> op;                                       \
        GET_TILING_DATA_WITH_STRUCT(                                               \
            SparseFlashAttentionTilingDataMla, tilingDataIn, tiling);              \
        const SparseFlashAttentionTilingDataMla *__restrict tilingData =            \
            &tilingDataIn;                                                         \
        op.Init(                                                                   \
            query, key, value, sparseIndices, cacheTokens,                         \
            actualSeqLengthsQuery, actualSeqLengthsKv, blockTable,                 \
            queryRope, keyRope, attentionOut, softmaxMax, softmaxSum,              \
            userWorkspace, tilingData, tiling, &pipe);                             \
        op.Process();                                                              \
    } while (0)
#endif

template <
    int FLASH_DECODE,
    int PAGE_ATTENTION,
    int LAYOUT_T,
    int KV_LAYOUT_T,
    int TEMPLATE_MODE,
    int IS_SPLIT_G>
__global__ __aicore__ void a5_sparse_and_tail_attention(
    GM_ADDR query,
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR sparseIndices,
    GM_ADDR blockTable,
    GM_ADDR actualSeqLengthsQuery,
    GM_ADDR actualSeqLengthsKv,
    GM_ADDR queryRope,
    GM_ADDR keyRope,
    GM_ADDR cacheTokens,
    GM_ADDR attentionOut,
    GM_ADDR softmaxMax,
    GM_ADDR softmaxSum,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

    if constexpr (
        ORIG_DTYPE_QUERY == DT_BF16 &&
        ORIG_DTYPE_KEY == DT_BF16 &&
        ORIG_DTYPE_ATTENTION_OUT == DT_BF16) {
        A5_SPARSE_TAIL_IMPL(
            bfloat16_t, bfloat16_t, float, bfloat16_t,
            FLASH_DECODE, PAGE_ATTENTION,
            static_cast<SFA_LAYOUT>(LAYOUT_T),
            static_cast<SFA_LAYOUT>(KV_LAYOUT_T),
            static_cast<SFATemplateMode>(TEMPLATE_MODE),
            IS_SPLIT_G);
    } else {
        A5_SPARSE_TAIL_IMPL(
            half, half, float, half,
            FLASH_DECODE, PAGE_ATTENTION,
            static_cast<SFA_LAYOUT>(LAYOUT_T),
            static_cast<SFA_LAYOUT>(KV_LAYOUT_T),
            static_cast<SFATemplateMode>(TEMPLATE_MODE),
            IS_SPLIT_G);
    }
}
