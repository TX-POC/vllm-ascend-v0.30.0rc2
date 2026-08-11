# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from types import SimpleNamespace

import torch

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.dsa_offload import a5_c8_precision_policy as policy
from vllm_ascend.utils import AscendDeviceType


def test_linear_quant_method_unwraps_adapter_and_direct_method():
    quant_method = object()

    direct_layer = SimpleNamespace(quant_method=quant_method)
    wrapped_layer = SimpleNamespace(
        quant_method=SimpleNamespace(quant_method=quant_method)
    )

    assert policy._layer_quant_method(direct_layer) is quant_method
    assert policy._layer_quant_method(wrapped_layer) is quant_method


def test_target_layer_requires_enabled_dsa_a5_c8_w4a4(monkeypatch):
    quant_method = object()
    impl = SimpleNamespace(
        enable_sparse_sfa_c8=True,
        fused_qkv_a_proj=SimpleNamespace(
            quant_method=SimpleNamespace(quant_method=quant_method)
        ),
        q_proj=None,
    )
    dsa_config = SimpleNamespace(enabled=True)
    monkeypatch.setattr(
        policy,
        "get_ascend_device_type",
        lambda: AscendDeviceType.A5,
    )
    monkeypatch.setattr(
        policy,
        "get_ascend_config",
        lambda: SimpleNamespace(dsa_offload_config=dsa_config),
    )
    monkeypatch.setattr(
        policy,
        "_is_w4a4_mxfp4",
        lambda method: method is quant_method,
    )

    assert policy._is_target_layer(impl)
    dsa_config.enabled = False
    assert not policy._is_target_layer(impl)


def test_precision_guard_disables_mlapo_before_native_packed_path(monkeypatch):
    impl = SimpleNamespace(
        enable_mlapo=True,
        enable_sfa_prolog_v3=True,
        fused_qkv_a_proj=None,
        q_proj=None,
        layer_name="model.layers.0.self_attn.attn",
    )
    monkeypatch.setattr(policy, "_is_target_layer", lambda _: True)
    monkeypatch.setattr(policy.logger, "warning_once", lambda *args: None)

    assert policy.apply_a5_c8_precision_guard(impl)
    assert impl.enable_mlapo is False
    assert impl.enable_sfa_prolog_v3 is False
    assert impl._a5_c8_force_native_packed is True
    assert impl.c8_k_cache_dtype == torch.float8_e4m3fn
    assert impl.c8_k_scale_cache_dtype == torch.float32


def test_precision_guard_runs_before_float_mlapo_weight_transform():
    source = inspect.getsource(AscendSFAImpl.process_weights_after_loading)

    assert source.index("apply_a5_c8_precision_guard(self)") < source.index(
        "self._process_weights_for_fused_mlapo_a5_float(act_dtype)"
    )


def test_pack_c8_rows_preserves_raw_rope_and_scale_bytes():
    row_count = 2
    k_nope = torch.arange(row_count * 512, dtype=torch.int16).to(torch.int8)
    k_nope = k_nope.view(row_count, 1, 512)
    k_pe = torch.arange(row_count * 64, dtype=torch.bfloat16)
    k_pe = k_pe.view(row_count, 1, 64)
    knope_scale = torch.tensor(
        [[1.0, -2.0, 3.5, -4.5], [5.0, -6.0, 7.5, -8.5]],
        dtype=torch.float32,
    )

    packed = policy._pack_c8_rows_as_bytes(
        k_nope,
        k_pe,
        knope_scale,
        row_count,
    )
    expected = torch.cat(
        [
            k_nope.contiguous().view(torch.int8).view(row_count, -1),
            k_pe.contiguous().view(torch.int8).view(row_count, -1),
            knope_scale.contiguous().view(torch.int8).view(row_count, -1),
        ],
        dim=-1,
    )

    assert packed.shape == (row_count, 656)
    assert torch.equal(packed, expected)
