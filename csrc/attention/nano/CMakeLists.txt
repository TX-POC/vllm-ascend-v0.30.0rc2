set(_a5_sparse_tail_attention_c8_dir
    "${CMAKE_CURRENT_SOURCE_DIR}/sparse_tail_attention_c8")
set(_a5_sparse_tail_attention_c8_host_dir
    "${_a5_sparse_tail_attention_c8_dir}/op_host")
set(a5_sparse_tail_attention_c8_dir
    "${_a5_sparse_tail_attention_c8_dir}"
    CACHE INTERNAL "A5 sparse-tail attention kernel source directory")

set(COMPILED_OPS
    ${COMPILED_OPS} sparse_tail_attention_c8
    CACHE STRING "Compiled Ops" FORCE)
set(COMPILED_OP_DIRS
    ${COMPILED_OP_DIRS} "${_a5_sparse_tail_attention_c8_dir}"
    CACHE STRING "Compiled Ops Dirs" FORCE)

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        "${_a5_sparse_tail_attention_c8_host_dir}/a5_sparse_tail_attention_c8.cpp"
    )
endif()

add_ops_compile_options(
    OP_NAME A5SparseTailAttentionC8
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    set(_a5_sparse_tail_attention_c8_compat_include_dir
        "${CMAKE_CURRENT_BINARY_DIR}/a5_sparse_tail_attention_c8_compat")
    file(MAKE_DIRECTORY
        "${_a5_sparse_tail_attention_c8_compat_include_dir}/a5_sfa_shared")
    file(WRITE
        "${_a5_sparse_tail_attention_c8_compat_include_dir}/a5_sfa_shared/ops_log_compat.h"
        "#pragma once\n#include \"err/ops_err.h\"\n")

    target_sources(${OPHOST_NAME}_opdef_aclnn_obj INTERFACE
        "${_a5_sparse_tail_attention_c8_host_dir}/a5_sparse_tail_attention_c8.cpp"
    )

    add_infer_modules()
    target_include_directories(${OPHOST_NAME}_infer_obj PRIVATE
        "${_a5_sparse_tail_attention_c8_compat_include_dir}"
    )
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE
        "${_a5_sparse_tail_attention_c8_host_dir}/kv_quant_sparse_flash_attention_infershape.cpp"
    )

    add_tiling_modules()
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        "${_a5_sparse_tail_attention_c8_compat_include_dir}"
        "${OPS_TRANSFORMER_DIR}/attention/kv_quant_sparse_flash_attention/op_host"
    )
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        "${_a5_sparse_tail_attention_c8_host_dir}/kv_quant_sparse_flash_attention_tiling.cpp"
    )
endif()
