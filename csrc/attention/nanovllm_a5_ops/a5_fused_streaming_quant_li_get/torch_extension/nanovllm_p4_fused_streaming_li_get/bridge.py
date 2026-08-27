"""P4.1 single-scan bridge: score-time candidate map plus optional Get post.

This is intentionally a thin policy boundary, not a second LI invocation.
It calls the isolated P3 raw-Q/K operator exactly once and exposes its
graph-static candidate-map receipt.  Its C8 Top-K is *non-authoritative*
until a target bitwise comparison admits it; official C8 remains the only
QSFA authority during bring-up.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


AUDIT_QUEUE_LEASE_SENTINEL = -(1 << 31)
AUDIT_CANDIDATE_MARKER = -3
_AUDIT = "audit"
_POST = "post"
_TRANSACTION_KEYS = (
    "resident_threshold",
    "resident_threshold_valid",
    "kv_object_slots_snapshot",
    "req_pool_entries",
    "kv_object_map",
    "private_source_to_slot",
    "private_ready_table",
    "private_arena_head",
    "private_packed_output",
    "queue_control",
    "queue_leases",
    "kv_object_lens",
)


def _allow_post() -> bool:
    return os.getenv("P4_FUSED_ALLOW_POST", "") == "1"


def _require_transaction(transaction: Mapping[str, Any]) -> None:
    missing = tuple(name for name in _TRANSACTION_KEYS if name not in transaction)
    if missing:
        raise ValueError(f"P4.1 transaction is missing keys: {missing}")


def _require_backing_bridge() -> Any:
    # Importing it is deliberate: the P3 bridge contains the typed ACLNN
    # binding.  P4.1 must not copy or resolve a second, possibly stale ABI.
    import nanovllm_streaming_quant_li_get  # noqa: F401
    import torch

    namespace = getattr(torch.ops, "a5_streaming_quant_li_get", None)
    op = getattr(namespace, "streaming_quant_li_get", None) if namespace else None
    if op is None:
        raise RuntimeError(
            "P4.1 requires the isolated A5StreamingQuantLiGet bridge/OPP; "
            "build it with NANOVLLM_A5_OPS_ONLY=A5StreamingQuantLiGet first"
        )
    return op


def launch(
    inputs: Mapping[str, Any],
    transaction: Mapping[str, Any],
    *,
    mode: str,
    kv_object_domain: int = 0,
    query_quant_mode: int = 0,
    key_quant_mode: int = 0,
    layout_query: str = "TND",
    layout_key: str = "PA_BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    pre_tokens: int = (1 << 63) - 1,
    next_tokens: int = (1 << 63) - 1,
    cmp_ratio: int = 1,
    return_values: bool = False,
    stride: int = 1,
    scale_stride: int = 1,
    kvaf_rows_per_entry: int = 128,
) -> dict[str, Any]:
    """Launch exactly one raw-Q/K scan and return non-authoritative receipts.

    Audit is selected solely by the caller's graph-static lease input: every
    lease cell must already equal ``AUDIT_QUEUE_LEASE_SENTINEL``.  The bridge
    intentionally does not perform a host-side fill or synchronization, since
    that would hide a graph dependency and turn the test into two executions.
    ``post`` uses the normal lease table but remains fail-closed unless the
    explicit environment gate is enabled.
    """
    if mode not in (_AUDIT, _POST):
        raise ValueError(f"P4.1 mode must be {_AUDIT!r} or {_POST!r}, got {mode!r}")
    _require_transaction(transaction)
    if mode == _POST and not _allow_post():
        raise RuntimeError(
            "P4.1 post is disabled: set P4_FUSED_ALLOW_POST=1 only after "
            "the Direct-HCCP payload path has a separate correctness proof"
        )

    op = _require_backing_bridge()
    outputs = op(
        inputs["query"],
        inputs["key"],
        inputs["weights"],
        inputs["query_dequant_scale"],
        inputs["key_dequant_scale"],
        inputs["actual_seq_lengths_query"],
        inputs["actual_seq_lengths_key"],
        inputs["block_table"],
        inputs["metadata"],
        transaction["resident_threshold"],
        transaction["resident_threshold_valid"],
        transaction["kv_object_slots_snapshot"],
        transaction["req_pool_entries"],
        transaction["kv_object_map"],
        transaction["private_source_to_slot"],
        transaction["private_ready_table"],
        transaction["private_arena_head"],
        transaction["private_packed_output"],
        transaction["queue_control"],
        transaction["queue_leases"],
        transaction["kv_object_lens"],
        kv_object_domain=kv_object_domain,
        query_quant_mode=query_quant_mode,
        key_quant_mode=key_quant_mode,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        cmp_ratio=cmp_ratio,
        return_values=return_values,
        stride=stride,
        scale_stride=scale_stride,
        kvaf_rows_per_entry=kvaf_rows_per_entry,
    )
    if not isinstance(outputs, tuple) or len(outputs) != 9:
        raise RuntimeError("P4.1 backing scan returned an unexpected ABI")

    return {
        # This result intentionally cannot be fed to QSFA by this bridge.
        "non_authoritative_sparse_indices": outputs[0],
        "non_authoritative_sparse_values": outputs[1],
        # In audit mode the fixed [R,S] receipt contains -3 candidates;
        # in post mode it retains the existing private-slot receipt ABI.
        "candidate_source_to_slot": outputs[2],
        "private_ready_table": outputs[3],
        "private_arena_head": outputs[4],
        "private_packed_output": outputs[5],
        "queue_leases": outputs[6],
        "telemetry": outputs[7],
        "score_source_lens": outputs[8],
        "mode": mode,
        "candidate_marker": AUDIT_CANDIDATE_MARKER,
        "audit_lease_sentinel": AUDIT_QUEUE_LEASE_SENTINEL,
        "topk_authority": "non_authoritative_pending_strict_equivalence",
        "h2d_claim": "none" if mode == _AUDIT else "payload_unverified",
    }
