# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for the native synchronized Decode receive metadata."""

from vllm_ascend.distributed.kv_transfer.kvaf.pd.scheduler import BatchReceiveRequest

__all__ = ["BatchReceiveRequest"]
