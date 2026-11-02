#ifndef A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_VECTOR_H
#define A5_LIDU_FUSED_BATCH_GET_PREPARE_MTP_VECTOR_H

#include "kernel_operator.h"

namespace A5LiduFusedBatchGetPrepareMtpVector {
using namespace AscendC;

constexpr uint32_t VECTOR_WIDTH = 64;

__simd_vf__ void SqueezeMissTokens(
    __ubuf__ uint32_t *outputTokens,
    __ubuf__ uint32_t *inputTokens,
    __ubuf__ uint32_t *inputSlots,
    uint32_t vectorLoops)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> invalidSlot;
    MicroAPI::Duplicate(invalidSlot, 0xffffffffU);
    MicroAPI::ClearSpr<SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore tokenStore;
    for (uint32_t loop = 0; loop < vectorLoops; ++loop) {
        MicroAPI::RegTensor<uint32_t> token;
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::RegTensor<uint32_t> compactToken;
        MicroAPI::MaskReg isMiss;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            token, inputTokens + loop * VECTOR_WIDTH);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            slot, inputSlots + loop * VECTOR_WIDTH);
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(
            isMiss, slot, invalidSlot, all);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(
            compactToken, token, isMiss);
        MicroAPI::StoreUnAlign<uint32_t,
                                MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            outputTokens, compactToken, tokenStore);
    }
    MicroAPI::StoreUnAlignPost(outputTokens, tokenStore);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE,
                          MicroAPI::MemType::VEC_LOAD>();
}

__simd_vf__ void SqueezeHitTokens(
    __ubuf__ uint32_t *outputTokens,
    __ubuf__ uint32_t *inputTokens,
    __ubuf__ uint32_t *inputSlots,
    uint32_t vectorLoops)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> invalidSlot;
    MicroAPI::Duplicate(invalidSlot, 0xffffffffU);
    MicroAPI::ClearSpr<SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore tokenStore;
    for (uint32_t loop = 0; loop < vectorLoops; ++loop) {
        MicroAPI::RegTensor<uint32_t> token;
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::RegTensor<uint32_t> compactToken;
        MicroAPI::MaskReg isMiss;
        MicroAPI::MaskReg isHit;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            token, inputTokens + loop * VECTOR_WIDTH);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            slot, inputSlots + loop * VECTOR_WIDTH);
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(
            isMiss, slot, invalidSlot, all);
        MicroAPI::Xor(isHit, isMiss, all, all);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(
            compactToken, token, isHit);
        MicroAPI::StoreUnAlign<uint32_t,
                                MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            outputTokens, compactToken, tokenStore);
    }
    MicroAPI::StoreUnAlignPost(outputTokens, tokenStore);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE,
                          MicroAPI::MemType::VEC_LOAD>();
}

__simd_vf__ void SqueezeHitSlots(
    __ubuf__ uint32_t *outputSlots,
    __ubuf__ uint32_t *inputSlots,
    uint32_t vectorLoops)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> invalidSlot;
    MicroAPI::Duplicate(invalidSlot, 0xffffffffU);
    MicroAPI::ClearSpr<SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore slotStore;
    for (uint32_t loop = 0; loop < vectorLoops; ++loop) {
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::RegTensor<uint32_t> compactSlot;
        MicroAPI::MaskReg isMiss;
        MicroAPI::MaskReg isHit;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            slot, inputSlots + loop * VECTOR_WIDTH);
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(
            isMiss, slot, invalidSlot, all);
        MicroAPI::Xor(isHit, isMiss, all, all);
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(
            compactSlot, slot, isHit);
        MicroAPI::StoreUnAlign<uint32_t,
                                MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            outputSlots, compactSlot, slotStore);
    }
    MicroAPI::StoreUnAlignPost(outputSlots, slotStore);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE,
                          MicroAPI::MemType::VEC_LOAD>();
}

}  // namespace A5LiduFusedBatchGetPrepareMtpVector

#endif
