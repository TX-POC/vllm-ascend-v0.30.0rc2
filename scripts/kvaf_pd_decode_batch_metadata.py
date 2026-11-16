# SPDX-License-Identifier: Apache-2.0
"""Opt-in metadata extension; the ordinary ReceiveRequest layout is unchanged."""

from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kvaf.pd.scheduler import ReceiveRequest


@dataclass(frozen=True)
class BatchReceiveRequest(ReceiveRequest):
    decode_batch: dict
