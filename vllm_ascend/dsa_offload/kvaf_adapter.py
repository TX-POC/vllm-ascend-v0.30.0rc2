# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA KVAF 适配层（本模块只落地 C8 Attention 索引语义部分）。

范围：只实现 ``DsaKvafAdapter::BuildC8AttentionMetadata`` —— 读取已由子图 C
resolver 验证的 destination slots / R/P/S，写出 ``attention_slots``。
不访问 DDR、不发 URMA、也不读写 KV payload。

KVAF 的 Put / AccessTable / BatchGetSlice / BatchFree 属后续模块，本文件
仅为这些后续职责预留结构，不实现具体 KVAF 调用。

依赖前项：子图 C resolver 已把 ``resident_seq_lengths`` 写入（唯一 writer）。
单元测试中可先构造该值。
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_ascend.dsa_offload.ops_nanovllm import build_c8_attention_metadata


class DsaKvafAdapter:
    """管理 KVAF 生命周期，并生成 C8 Attention 索引语义。

    本模块只实现 ``build_c8_attention_metadata``（索引侧）。
    """

    def __init__(self, device: str | None = None) -> None:
        self._device = device or ("npu" if torch.npu.is_available() else "cpu")

    @property
    def device(self) -> str:
        return self._device

    def build_c8_attention_metadata(
        self,
        *,
        destination_slots: torch.Tensor,
        cache_tokens: torch.Tensor,
        candidate_lens: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        max_tail_tokens: int,
        resident_seq_lengths: torch.Tensor,
        attention_slots: torch.Tensor,
    ) -> torch.Tensor:
        """写 ``attention_slots``；不访问 DDR、不发 URMA、不读写 KV payload。

        委托 ``ops_nanovllm.build_c8_attention_metadata``（纯索引）。

        调用方（``execute_decode_selection``）保证：
        - 子图 C resolver 已写好 ``resident_seq_lengths``（本方法只读）；
        - ``complete_direct_get_across_ranks`` 已在当前 stream 建立必要的
          stream/event 依赖（本方法不做同步）。
        """
        return build_c8_attention_metadata(
            destination_slots=destination_slots,
            cache_tokens=cache_tokens,
            candidate_lens=candidate_lens,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            max_tail_tokens=max_tail_tokens,
            resident_seq_lengths=resident_seq_lengths,
            attention_slots=attention_slots,
        )

    def complete_direct_get_across_ranks(self, *args: Any, **kwargs: Any) -> None:
        """（子图 C 实现）在调用 QSFA 前建立 stream/event 依赖。

        本模块只保证不默认同步；真正的 happens-before 依赖由子图 C 的
        KVAF BatchGetSlice / Direct Get 回填流程建立。
        """
        raise NotImplementedError(
            "DsaKvafAdapter.complete_direct_get_across_ranks is owned by "
            "subgraph C (KVAF backfill); not implemented in this module."
        )
