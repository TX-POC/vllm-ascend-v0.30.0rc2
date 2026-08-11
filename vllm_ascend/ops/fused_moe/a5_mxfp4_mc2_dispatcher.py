# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A5 TP-aware MC2 compatibility for MXFP4 token transport.

The model/MLP remains W4A4 MXFP4. This module only adapts the representation
used while tokens pass through MC2 on A5 with TP>1, where the current
MoeDistributeDispatchV3 tiling rejects the packed FP4 expand_x logical width.

The implementation subclasses the native dispatcher so HCCL groups, routing,
operator ABI and all non-affected quantization modes stay on the upstream path.
"""

from __future__ import annotations

from dataclasses import replace

from vllm.config import get_current_vllm_config

from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMC2CombineMetadata,
    MoETokenDispatchInput,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def requires_unquantized_a5_mxfp4_mc2_transport(
    *,
    device_type: AscendDeviceType,
    tp_size: int,
    quant_type: QuantType,
) -> bool:
    """Return whether MC2 must transport MXFP4 activations unquantized.

    This is deliberately a transport-capability decision. It does not alter
    ``QuantType.MXFP4`` or the precision used by the expert MLP.
    """

    return (
        device_type == AscendDeviceType.A5
        and tp_size > 1
        and quant_type == QuantType.MXFP4
    )


class A5Mxfp4CompatibleTokenDispatcherWithMC2(TokenDispatcherWithMC2):
    """Native MC2 dispatcher with a narrow A5 TP>1 MXFP4 transport fix."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vllm_config = get_current_vllm_config()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.device_type = get_ascend_device_type()

    def _requires_unquantized_transport(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> bool:
        return requires_unquantized_a5_mxfp4_mc2_transport(
            device_type=self.device_type,
            tp_size=self.tp_size,
            quant_type=token_dispatch_input.quant.quant_type,
        )

    def get_dispatch_mc2_kwargs(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        kwargs_mc2 = super().get_dispatch_mc2_kwargs(token_dispatch_input)
        if self._requires_unquantized_transport(token_dispatch_input):
            # Keep x/expand_x in BF16/FP16 during MC2 transport. The model
            # remains MXFP4 and the existing MLP path dynamically quantizes
            # expand_x immediately before expert computation.
            kwargs_mc2["quant_mode"] = 0
            kwargs_mc2.pop("y_dtype", None)
        return kwargs_mc2

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        unquantized_transport = self._requires_unquantized_transport(
            token_dispatch_input
        )
        output = super().token_dispatch(token_dispatch_input)
        if not unquantized_transport:
            return output

        # quant_mode=0 may still expose a placeholder dynamic_scale on some
        # PTA/CANN combinations. Clear it explicitly so the MXFP4 MLP follows
        # its existing dynamic-quant path for the unquantized expand_x tensor.
        combine_metadata = replace(
            output.combine_metadata,
            dispatch_quantized=False,
        )
        return replace(
            output,
            dynamic_scale=None,
            combine_metadata=combine_metadata,
        )

    def get_combine_mc_kwargs(
        self,
        hidden_states,
        combine_metadata: MoEMC2CombineMetadata,
    ):
        kwargs_mc2 = super().get_combine_mc_kwargs(
            hidden_states,
            combine_metadata,
        )
        if combine_metadata.dispatch_quantized is False:
            # Native combine drops tp_recv_counts when the model supports
            # quantized dispatch. Here the *actual* dispatch transport was
            # unquantized, so retain the real counts and keep combine transport
            # unquantized as well.
            kwargs_mc2["tp_send_counts"] = combine_metadata.tp_recv_counts
            kwargs_mc2["comm_quant_mode"] = 0
        return kwargs_mc2


__all__ = [
    "A5Mxfp4CompatibleTokenDispatcherWithMC2",
    "requires_unquantized_a5_mxfp4_mc2_transport",
]
