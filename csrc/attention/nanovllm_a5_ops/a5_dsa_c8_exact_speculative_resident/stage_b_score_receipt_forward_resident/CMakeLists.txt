cmake_minimum_required(VERSION 3.19)

# The executable CANN package route is build_stage_b_opp.sh: it creates a
# fresh msopgen project, stages the exact KVAF device closure, and replaces
# only that project's generated stub with this distinct P10 MIX kernel. This
# file keeps the checked-in source manifest-friendly for a CANN superproject;
# it never reaches into an ambient KVAF build directory.
if(COMMAND add_op_to_compiled_list)
  add_subdirectory(op_host)
else()
  add_custom_target(p10_stage_b_source
    SOURCES
      op_host/a5_p10_score_receipt_forward_resident_def.cpp
      op_kernel/a5_p10_score_receipt_forward_resident_kernel.cpp
      op_kernel/a5_p10_score_receipt_forward_resident_abi.h
      op_kernel/a5_p10_score_receipt_forward_resident_tiling.h
      op_kernel/a5_p10_score_receipt_forward_resident_kvaf_include.h
      ops.json
  )
  message(STATUS
    "P10 Stage-B source manifest loaded; use build_stage_b_opp.sh for the "
    "fresh provenance-checked CANN package build.")
endif()
