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
 * \file a5_streaming_quant_li_get_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class A5StreamingQuantLiGet : public OpDef {
public:
    explicit A5StreamingQuantLiGet(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key")
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
        this->Input("key_dequant_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("actual_seq_lengths_query")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_key")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("metadata")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        // Appended streaming inputs. They are all graph-static views owned by
        // the runtime transaction; none replaces an official LI input.
        this->Input("resident_threshold")
            .ParamType(REQUIRED)
            // Preserve the resident seed's int32 ABI.  Its low 16 bits are
            // the official C8 sortable score, so graph capture can wire the
            // seed output here without a cast/copy allocation.
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        // A zero means the resident-only C8 seed has fewer than sparse_count
        // valid candidates.  In that case this operator must not reinterpret
        // the seed's zero tau as a real score threshold and flood the private
        // Get arena; it simply disables speculation for that row.
        this->Input("resident_threshold_valid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        // Object-directory snapshot.  Its index domain is declared by the
        // required graph-static kv_object_domain attribute below.
        this->Input("kv_object_slots_snapshot")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("req_pool_entries")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        // Maps official C8 score-source S to a QSFA-addressable KV object.
        this->Input("kv_object_map")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("private_source_to_slot")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("private_ready_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("private_arena_head")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        // Explicit graph-visible owner of the one registered KVAF output.
        // The device hook verifies it is exactly control.output before it
        // posts; queue-private rows are selected only by output_indices.
        this->Input("private_packed_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("queue_control")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("queue_leases")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        // Per C8 sparse-output row, this is the active KV-object prefix.
        // It is distinct from output score_source_lens (the compressed C8
        // score-source S prefix). The hook needs both before it may dereference
        // kv_object_map or kv_object_slots_snapshot.
        this->Input("kv_object_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        // C8 writes exact uint16 scores through this int32-backed
        // graph-static allocation. It is intentionally separate from CANN's
        // private workspace so an independent Get sidecar can consume the
        // same raw-score representation without recomputation.
        this->Input("score_workspace")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        // One fixed [16]int64 slot per (output row, score tile). The producer
        // writes its epoch in the second cache line only after score and
        // descriptor lines are enqueued on MTE3.
        this->Input("tile_receipts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("streaming_epoch")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("sparse_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("sparse_values").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        // These two outputs are transaction receipts, aliasing the caller's
        // mutable private state.  They are deliberately not a persistent
        // cache mapping and never change a managed HBM slot.
        this->Output("private_source_to_slot_out").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("private_ready_table_out").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("private_arena_head_out").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("private_packed_output_out").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
        this->Output("queue_leases_out").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("telemetry").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        // The exact C8 ProcessTopK path is the authoritative producer of
        // this per-output-row score-source length.  Reconcile must consume
        // it rather than infer compressed-S2 lengths on the host.
        this->Output("score_source_lens").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("score_workspace_out").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("tile_receipts_out").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Attr("query_quant_mode").AttrType(REQUIRED).Int(0);  // 0: 默认值，per-token-head
        this->Attr("key_quant_mode").AttrType(REQUIRED).Int(0);    // 0: 默认值，per-token-head
        // Product decode calls C8 with TND query plus paged PA_BSND key.
        // The copied tiling still accepts the official BSND/TND variants when
        // callers explicitly request them, but the isolated P3 default must
        // not silently select the non-product BSND baseline.
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
        // Direct-HCCP address-table granularity. This is a transport ABI
        // property, not a LI score/topk parameter; it is kept explicit so
        // the derived op never silently assumes 128-token pages.
        this->Attr("kvaf_rows_per_entry").AttrType(OPTIONAL).Int(128);
        // 0 = RAW_TOKEN: valid only when cmp_ratio == 1.
        // 1 = C8_COMPRESSED_OBJECT: values name the exact cmp_kv objects
        // consumed by the final QSFA path; cmp_ratio may be 1, 4, or 128.
        // Required intentionally: omitting this semantic choice must fail
        // graph construction rather than silently assume raw-token mapping.
        this->Attr("kv_object_domain").AttrType(REQUIRED).Int();
        // This derivative embeds the A5 C310 score tile and its Direct-HCCP
        // device helper.  Do not advertise a 910B/910_93 configuration: the
        // kernel intentionally has no compatible implementation for them.
        OpAICoreConfig aicore_config_950;
        aicore_config_950.Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("key")
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
        aicore_config_950.Input("key_dequant_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicore_config_950.Input("actual_seq_lengths_query")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("actual_seq_lengths_key")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("metadata")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("resident_threshold")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("resident_threshold_valid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("kv_object_slots_snapshot")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("req_pool_entries")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("kv_object_map")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("private_source_to_slot")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("private_ready_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("private_arena_head")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("private_packed_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("queue_control")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("queue_leases")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("kv_object_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("score_workspace")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("tile_receipts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Input("streaming_epoch")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_950.Output("sparse_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        aicore_config_950.Output("sparse_values").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        aicore_config_950.Output("private_source_to_slot_out").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        aicore_config_950.Output("private_ready_table_out").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        aicore_config_950.Output("private_arena_head_out").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        aicore_config_950.Output("private_packed_output_out").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
        aicore_config_950.Output("queue_leases_out").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        aicore_config_950.Output("telemetry").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        aicore_config_950.Output("score_source_lens").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        aicore_config_950.Output("score_workspace_out").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        aicore_config_950.Output("tile_receipts_out").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        aicore_config_950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value", "a5_streaming_quant_li_get")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend950", aicore_config_950);
    }
};
OP_ADD(A5StreamingQuantLiGet);
}  // namespace ops
