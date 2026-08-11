# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA offload 框架与算子共享的静态接口契约。

本文件只放不会随请求或 decode step 变化的 ABI 常量。框架配置校验和
LIDU/KSC/SFA-Offload 算子适配层共同引用这些定义，
避免同一约束在多处以魔法数字重复出现。
"""

DSA_SFA_COMPUTE_TOPK = 2048
# Legacy/default LIDU keeps the original fixed output capacity. The A5 backend
# uses DSA_SFA_COMPUTE_TOPK directly so caller-owned buffers are physically
# contiguous [B,1,2048] and never need a decode-hot-path copy.
DSA_LIDU_OUTPUT_CAPACITY = 16384
DSA_A5_LIDU_OUTPUT_CAPACITY = DSA_SFA_COMPUTE_TOPK
DSA_LIDU_TOKEN_CAPACITY = 1 << 18
DSA_LIDU_CACHE_ROW_ALIGNMENT = 256
DSA_LIDU_SUPPORTED_RESIDENT_BUDGETS = (6144, 10240, 12288)
DSA_REQUIRED_CACHE_BLOCK_SIZE = 128
DSA_PACKED_KV_DIM = 656
DSA_KV_LORA_RANK = 512
DSA_QK_ROPE_HEAD_DIM = 64
DSA_INDEX_HEAD_DIM = 128
DSA_SUPPORTED_INDEX_HEAD_COUNTS = (32, 64)

# LIDU/KSC/SFA-Offload 的逐行执行模式。ENTER 与 steady SPARSE 在设备
# 数据面上都属于 SPARSE；请求生命周期阶段仍由 request_cache_layout 管理。
DSA_ROW_MODE_PAD = 0
DSA_ROW_MODE_DENSE = 1
DSA_ROW_MODE_SPARSE = 2

# 通用满块复制算子只把目的 block id=-1 解释为空转。DRAM 逻辑块表的 0
# 仍保留为空映射，因此有效物理 DRAM block id 从 1 开始。
DSA_DUMP_NOOP_DST_BLOCK_ID = -1
DSA_DRAM_NULL_BLOCK_ID = 0

DSA_OP_BACKEND_DEFAULT = "default"
DSA_OP_BACKEND_NANOVLLM_A5 = "nanovllm_a5"
