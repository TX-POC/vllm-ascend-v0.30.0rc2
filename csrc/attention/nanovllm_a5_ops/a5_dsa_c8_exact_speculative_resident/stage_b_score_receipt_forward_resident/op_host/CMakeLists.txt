add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
  target_sources(op_host_aclnn PRIVATE
    a5_p10_score_receipt_forward_resident_def.cpp
  )
endif()

add_ops_compile_options(
  OP_NAME A5P10ScoreReceiptForwardResident
  OPTIONS
    --cce-auto-sync=off
    -Wno-deprecated-declarations
    -mllvm -cce-vf-remove-membar=false
)

if(NOT BUILD_OPS_RTY_KERNEL)
  add_modules_sources(
    OPTYPE a5_p10_score_receipt_forward_resident
    ACLNNTYPE aclnn
  )
  target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
endif()
