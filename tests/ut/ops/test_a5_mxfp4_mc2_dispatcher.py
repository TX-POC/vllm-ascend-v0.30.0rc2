# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.ops.fused_moe.a5_mxfp4_mc2_dispatcher import (
    requires_unquantized_a5_mxfp4_mc2_transport,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType


def test_a5_mxfp4_tp2_uses_unquantized_mc2_transport():
    assert requires_unquantized_a5_mxfp4_mc2_transport(
        device_type=AscendDeviceType.A5,
        tp_size=2,
        quant_type=QuantType.MXFP4,
    )


def test_a5_mxfp4_tp1_keeps_native_quantized_transport():
    assert not requires_unquantized_a5_mxfp4_mc2_transport(
        device_type=AscendDeviceType.A5,
        tp_size=1,
        quant_type=QuantType.MXFP4,
    )


def test_a5_mxfp8_tp2_keeps_native_transport():
    assert not requires_unquantized_a5_mxfp4_mc2_transport(
        device_type=AscendDeviceType.A5,
        tp_size=2,
        quant_type=QuantType.MXFP8,
    )


def test_non_a5_mxfp4_tp2_keeps_native_transport():
    assert not requires_unquantized_a5_mxfp4_mc2_transport(
        device_type=AscendDeviceType.A3,
        tp_size=2,
        quant_type=QuantType.MXFP4,
    )
