// SPDX-License-Identifier: Apache-2.0
// Embed the existing A5 DSA torch registrations into vllm_ascend_C.
//
// The standalone nanovllm_dsa_a5 extension used PYBIND11_MODULE only to make
// torch load the translation unit.  vllm_ascend_C already owns the Python
// module entry point, so suppress the nested module definition while keeping
// the TORCH_LIBRARY(nanovllm_dsa, ...) registrations unchanged.

#include <torch/extension.h>

// op_api_common.h contains one non-inline helper definition.  Give it a
// translation-unit-local unique symbol name so embedding the same native
// adapter in the other A5 binding TUs cannot create duplicate linker symbols.
#define IsOpInputBaseFormat nanovllm_a5_ops_IsOpInputBaseFormat
#include "nanovllm_a5_vllm_op_api_bridge.h"
#undef IsOpInputBaseFormat

#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable)                                      \
  static void nanovllm_a5_embedded_pybind_stub(pybind11::module_ &variable)

#include "attention/nanovllm_a5_ops/torch_extension/csrc/ops.cpp"
