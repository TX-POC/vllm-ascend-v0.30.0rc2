# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        a5_sparse_and_tail_attention.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME A5SparseAndTailAttention
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -fpermissive
        --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(
        OPTYPE a5_sparse_and_tail_attention
        ACLNNTYPE aclnn_inner
    )
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/a5_sfa_shared
    )
endif()
