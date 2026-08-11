// SPDX-License-Identifier: Apache-2.0
// Build the A5 full-block dump torch registration as part of vllm_ascend_C.

#define IsOpInputBaseFormat nanovllm_a5_full_block_dump_IsOpInputBaseFormat
#include "nanovllm_a5_vllm_op_api_bridge.h"
#undef IsOpInputBaseFormat

#include "attention/nanovllm_a5_ops/torch_extension/csrc/full_block_dump.cpp"
