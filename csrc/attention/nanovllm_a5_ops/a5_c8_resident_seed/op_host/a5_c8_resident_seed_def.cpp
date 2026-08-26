/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file a5_c8_resident_seed_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class A5C8ResidentSeed : public OpDef {
public:
    explicit A5C8ResidentSeed(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        // This is the separate FP8 index-key view used by C8.  It is not the
        // packed 656-byte KV payload consumed later by QSFA.
        this->Input("resident_index_key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("query_dequant_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("resident_index_key_dequant_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("actual_seq_lengths_query")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("resident_seq_lengths")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("resident_index_block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("metadata")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("sparse_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("sparse_values").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        // Graph ABI: resident_tau is int32 with the sortable uint16 C8 score
        // in its low 16 bits. tau_valid is 1 iff that row had >=2048 valid
        // resident index keys, otherwise resident_tau is 0 and tau_valid=0.
        this->Output("resident_tau").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("tau_valid").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Attr("query_quant_mode").AttrType(REQUIRED).Int(0);  // 0: 默认值，per-token-head
        this->Attr("key_quant_mode").AttrType(REQUIRED).Int(0);    // 0: 默认值，per-token-head
        // Decode resident scan uses the same TND + paged index-view contract
        // as the A5 C8 path; callers may still explicitly request official
        // compatible layouts accepted by the copied tiling validator.
        this->Attr("layout_query").AttrType(OPTIONAL).String("TND");
        this->Attr("layout_key").AttrType(OPTIONAL).String("PA_BSND");
        this->Attr("sparse_count").AttrType(OPTIONAL).Int(2048);  // 2048: 默认值，筛选前2048
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);      // 3: 默认值，只计算下三角
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(9223372036854775807);  // 9223372036854775807: 默认值，int64的最大值
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(9223372036854775807); // 9223372036854775807: 默认值，int64的最大值
        this->Attr("cmp_ratio").AttrType(OPTIONAL).Int(1);          // 1: 压缩率
        this->Attr("return_values").AttrType(OPTIONAL).Bool(false); // 是否返回sparse_values
        this->Attr("stride").AttrType(OPTIONAL).Int(1);             // stride参数
        this->Attr("scale_stride").AttrType(OPTIONAL).Int(1);       // scaleStride参数
        // Deliberately do not register a 910B/910_93 kernel: this isolated
        // prototype embeds the Ascend-950 C8 uint16 sortable-score closure.

        OpAICoreConfig aicore_config_950;
        aicore_config_950.Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("resident_index_key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicore_config_950.Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("query_dequant_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("resident_index_key_dequant_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicore_config_950.Input("actual_seq_lengths_query")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("resident_seq_lengths")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("resident_index_block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("metadata")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Output("sparse_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        aicore_config_950.Output("sparse_values").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        aicore_config_950.Output("resident_tau").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        aicore_config_950.Output("tau_valid").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        aicore_config_950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value", "a5_c8_resident_seed")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend950", aicore_config_950);
    }
};
OP_ADD(A5C8ResidentSeed);
}  // namespace ops
