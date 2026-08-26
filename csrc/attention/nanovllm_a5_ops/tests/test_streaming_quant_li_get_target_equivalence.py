#!/usr/bin/env python3
"""Fail-closed target-runtime equivalence harness for A5StreamingQuantLiGet.

This file intentionally contains no invented target adapter.  The isolated
OPP bridge is only a launch path; a host-side score model is not evidence that
the A5 C8 implementation is equivalent.  Pass an adapter in
``module:function`` form.  The adapter must run
one *identical* target-NPU case through both the official C8 op and the derived
op and return the two output tensors.

Required adapter result:

    {
      "official_indices": tensor_or_array,
      "derived_indices": tensor_or_array,
      # Both launches must consume the same physical official-C8 inputs.  The
      # adapter supplies these pairs rather than merely asserting that its
      # Python values happen to compare equal after a copy.
      "input_aliases": {
        "query": (official_input, derived_input),
        "key": (official_input, derived_input),
        "weights": (official_input, derived_input),
        "query_dequant_scale": (official_input, derived_input),
        "key_dequant_scale": (official_input, derived_input),
        "actual_seq_lengths_query": (official_input, derived_input),
        "actual_seq_lengths_key": (official_input, derived_input),
        "block_table": (official_input, derived_input),
        "metadata": (official_input, derived_input),
      },
      # Required graph-capture proof. Each entry is a (input, output) tensor
      # pair for exactly the same physical allocation:
      "transaction_aliases": {
        "private_source_to_slot": (input_tensor, output_tensor),
        "private_ready_table": (input_tensor, output_tensor),
        "private_arena_head": (input_tensor, output_tensor),
        "private_packed_output": (input_tensor, output_tensor),
        "queue_leases": (input_tensor, output_tensor),
      },
      # Optional only when the actual official configuration exposes values:
      "official_values": tensor_or_array_or_none,
      "derived_values": tensor_or_array_or_none,
      # Required: an independent host reconstruction using the same official
      # C8 actual-sequence/metadata rules, not a copy of the derived output.
      "expected_score_source_lens": tensor_or_array,
      "derived_score_source_lens": tensor_or_array,
    }

The harness compares shape, dtype, and raw contiguous bytes.  Any mismatch is
an equivalence failure and must keep the product gate closed.  Current official
A5 C8 tiling rejects ``return_values=true``; therefore the first real target
case is expected to compare indices only.  Values become mandatory if/when the
official reference configuration exposes them.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any, Callable


def _resolve_callable(spec: str) -> Callable[[], dict[str, Any]]:
    module_name, separator, callable_name = spec.partition(":")
    if not separator or not module_name or not callable_name:
        raise ValueError("--adapter must be module:function")
    callback = getattr(importlib.import_module(module_name), callable_name)
    if not callable(callback):
        raise TypeError(f"adapter {spec!r} is not callable")
    return callback


def _canonical(tensor: Any, name: str) -> tuple[tuple[int, ...], str, bytes]:
    """Return a strict host-byte representation without assuming torch/numpy."""
    if tensor is None:
        raise AssertionError(f"{name} is missing")
    # Target adapters normally return torch tensors.  Synchronize and move to
    # host before materializing bytes, but intentionally do not change dtype.
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "contiguous"):
        tensor = tensor.contiguous()

    shape = tuple(int(value) for value in getattr(tensor, "shape", ()))
    dtype = str(getattr(tensor, "dtype", type(tensor).__name__))
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    if hasattr(tensor, "tobytes"):
        raw = tensor.tobytes()
    else:
        raise TypeError(f"{name} has no byte serialization method")
    return shape, dtype, raw


def _assert_bitwise_equal(lhs: Any, rhs: Any, label: str) -> None:
    lhs_shape, lhs_dtype, lhs_bytes = _canonical(lhs, f"official_{label}")
    rhs_shape, rhs_dtype, rhs_bytes = _canonical(rhs, f"derived_{label}")
    assert lhs_shape == rhs_shape, (
        f"{label} shape mismatch: official={lhs_shape}, derived={rhs_shape}"
    )
    assert lhs_dtype == rhs_dtype, (
        f"{label} dtype mismatch: official={lhs_dtype}, derived={rhs_dtype}"
    )
    assert lhs_bytes == rhs_bytes, f"{label} byte mismatch"


def _data_pointer(value: Any, label: str) -> int:
    """Extract a device address without accepting a copied host value."""
    if isinstance(value, int):
        return value
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        return int(data_ptr())
    raise TypeError(f"{label} has no device data_ptr()")


def _assert_transaction_aliases(result: dict[str, Any]) -> None:
    aliases = result.get("transaction_aliases")
    if not isinstance(aliases, dict):
        raise AssertionError("adapter did not return transaction_aliases")
    for name in (
        "private_source_to_slot",
        "private_ready_table",
        "private_arena_head",
        "private_packed_output",
        "queue_leases",
    ):
        pair = aliases.get(name)
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise AssertionError(f"missing input/output alias pair for {name}")
        input_ptr = _data_pointer(pair[0], f"{name}.input")
        output_ptr = _data_pointer(pair[1], f"{name}.output")
        assert input_ptr == output_ptr, (
            f"{name} is not a graph-visible physical alias: "
            f"input=0x{input_ptr:x}, output=0x{output_ptr:x}"
        )


def _assert_official_input_aliases(result: dict[str, Any]) -> None:
    """Prevent a target adapter from comparing different C8 input cases."""
    aliases = result.get("input_aliases")
    if not isinstance(aliases, dict):
        raise AssertionError("adapter did not return input_aliases")
    for name in (
        "query",
        "key",
        "weights",
        "query_dequant_scale",
        "key_dequant_scale",
        "actual_seq_lengths_query",
        "actual_seq_lengths_key",
        "block_table",
        "metadata",
    ):
        pair = aliases.get(name)
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise AssertionError(f"missing official/derived input pair for {name}")
        official_ptr = _data_pointer(pair[0], f"{name}.official")
        derived_ptr = _data_pointer(pair[1], f"{name}.derived")
        assert official_ptr == derived_ptr, (
            f"{name} does not use the same physical C8 input: "
            f"official=0x{official_ptr:x}, derived=0x{derived_ptr:x}"
        )


def _assert_source_index_domain(result: dict[str, Any]) -> None:
    """Require the derived S-domain bound and reject out-of-domain TopK ids.

    C8 sparse indices are score-source indices S.  The later Reconcile maps
    S to logical KV L, so treating a cache-slot capacity or logical sequence
    length as this bound would be a mathematical bug when cmp_ratio > 1.
    """
    _assert_bitwise_equal(
        result.get("expected_score_source_lens"),
        result.get("derived_score_source_lens"),
        "score_source_lens",
    )

    indices = result.get("derived_indices")
    lengths = result.get("derived_score_source_lens")
    if hasattr(indices, "detach"):
        indices = indices.detach()
    if hasattr(lengths, "detach"):
        lengths = lengths.detach()
    if hasattr(indices, "cpu"):
        indices = indices.cpu()
    if hasattr(lengths, "cpu"):
        lengths = lengths.cpu()
    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    if hasattr(lengths, "tolist"):
        lengths = lengths.tolist()

    def flatten_rows(value: Any) -> list[list[int]]:
        if not isinstance(value, list):
            raise TypeError("derived_indices must be list-convertible")
        if value and isinstance(value[0], int):
            return [value]
        rows: list[list[int]] = []
        for child in value:
            rows.extend(flatten_rows(child))
        return rows

    rows = flatten_rows(indices)
    if not isinstance(lengths, list):
        raise TypeError("derived_score_source_lens must be list-convertible")
    flat_lengths: list[int] = []
    pending = list(lengths)
    while pending:
        item = pending.pop(0)
        if isinstance(item, list):
            pending[:0] = item
        else:
            flat_lengths.append(int(item))
    if len(rows) != len(flat_lengths):
        raise AssertionError(
            "score_source_lens row count mismatches sparse_indices: "
            f"rows={len(rows)} lengths={len(flat_lengths)}"
        )
    for row, source_len in zip(rows, flat_lengths, strict=True):
        if source_len < 0:
            raise AssertionError(f"negative score-source length {source_len}")
        for source_index in row:
            if source_index != -1 and not 0 <= int(source_index) < source_len:
                raise AssertionError(
                    "derived sparse index exceeds its exact score-source "
                    f"length: index={source_index} length={source_len}"
                )


def _assert_actual_c8_output_domain(result: dict[str, Any]) -> None:
    """Reject a fictitious N2/C output axis before calling it C8 equivalent.

    A5 C8 accepts N2 == 1.  C=4 is its internal S1 score tile, so target
    output must be either TND [T, 1, 2048] or BSND [B, S1, 1, 2048].  This is
    deliberately checked on both official and derived output, not inferred
    from a test adapter's description.
    """
    for side in ("official_indices", "derived_indices"):
        shape, _, _ = _canonical(result.get(side), side)
        if len(shape) not in (3, 4) or shape[-2:] != (1, 2048):
            raise AssertionError(
                f"{side} is not an actual A5 C8 output shape: {shape}; "
                "expected [T,1,2048] or [B,S1,1,2048]"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        required=True,
        help="target runner module:function; must launch official and derived C8",
    )
    args = parser.parse_args()

    result = _resolve_callable(args.adapter)()
    if not isinstance(result, dict):
        raise TypeError("target adapter must return a dict")
    _assert_official_input_aliases(result)
    _assert_transaction_aliases(result)
    _assert_bitwise_equal(
        result.get("official_indices"), result.get("derived_indices"), "indices"
    )
    _assert_actual_c8_output_domain(result)
    _assert_source_index_domain(result)

    official_values = result.get("official_values")
    derived_values = result.get("derived_values")
    if (official_values is None) != (derived_values is None):
        raise AssertionError("only one side returned sparse_values")
    if official_values is not None:
        _assert_bitwise_equal(official_values, derived_values, "values")

    print(
        "P3_STREAMING_LI_GET_TARGET_EQ_PASS "
        f"values_compared={int(official_values is not None)} "
        "math_gate=bitwise_official_c8 source_domain=verified c8_n2_one=verified input_aliases=verified "
        "graph_aliases=verified"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # Fail closed; do not turn a missing bridge into PASS.
        print(
            "P3_STREAMING_LI_GET_TARGET_EQ_FAIL "
            f"type={type(error).__name__} text={error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
