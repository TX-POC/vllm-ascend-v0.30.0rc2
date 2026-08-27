#!/usr/bin/env python3
"""Strictly verify that an advisory raw-Q/K mirror cannot alter exact C8.

This intentionally does not compare the mirror clone's sparse indices with
official C8: P3 proved that those indices are not safe as an authority once
the prefetch hook is compiled in.  The only assertion is the product rule:
the exact official C8 result is byte-for-byte identical with and without an
independent mirror branch in flight.
"""

from __future__ import annotations

import argparse
import hashlib
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


def _canonical(value: Any, label: str) -> tuple[tuple[int, ...], str, bytes]:
    if value is None:
        raise AssertionError(f"{label} is missing")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "contiguous"):
        value = value.contiguous()
    shape = tuple(int(item) for item in getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", type(value).__name__))
    if hasattr(value, "numpy"):
        value = value.numpy()
    if not hasattr(value, "tobytes"):
        raise TypeError(f"{label} has no byte serialization method")
    return shape, dtype, value.tobytes()


def _assert_bitwise_equal(baseline: Any, concurrent: Any) -> None:
    left_shape, left_dtype, left_bytes = _canonical(baseline, "baseline")
    right_shape, right_dtype, right_bytes = _canonical(concurrent, "concurrent")
    if left_shape != right_shape or left_dtype != right_dtype:
        raise AssertionError(
            "P4_EXACT_ISOLATION_DIFF metadata "
            f"baseline_shape={left_shape} concurrent_shape={right_shape} "
            f"baseline_dtype={left_dtype} concurrent_dtype={right_dtype}"
        )
    if left_bytes == right_bytes:
        return
    first = next(
        offset
        for offset, (left, right) in enumerate(
            zip(left_bytes, right_bytes, strict=True)
        )
        if left != right
    )
    mismatches = sum(
        left != right
        for left, right in zip(left_bytes, right_bytes, strict=True)
    )
    detail = (
        "P4_EXACT_ISOLATION_DIFF "
        f"shape={left_shape} dtype={left_dtype} "
        f"total_bytes={len(left_bytes)} differing_bytes={mismatches} "
        f"first_byte={first} "
        f"baseline_sha256={hashlib.sha256(left_bytes).hexdigest()[:16]} "
        f"concurrent_sha256={hashlib.sha256(right_bytes).hexdigest()[:16]}"
    )
    if left_dtype.endswith("int32") and first + 4 <= len(left_bytes):
        element = first // 4
        base = element * 4
        detail += (
            f" first_int32_flat={element} "
            "baseline_value="
            f"{int.from_bytes(left_bytes[base:base + 4], sys.byteorder, signed=True)} "
            "concurrent_value="
            f"{int.from_bytes(right_bytes[base:base + 4], sys.byteorder, signed=True)}"
        )
    print(detail, file=sys.stderr, flush=True)
    raise AssertionError("official exact C8 changed while advisory mirror was in flight")


def _data_pointer(value: Any, label: str) -> int:
    pointer = getattr(value, "data_ptr", None)
    if not callable(pointer):
        raise TypeError(f"{label} has no data_ptr")
    return int(pointer())


def _assert_aliases(result: dict[str, Any], name: str) -> None:
    aliases = result.get(name)
    if not isinstance(aliases, dict):
        raise AssertionError(f"missing {name}")
    for label, pair in aliases.items():
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise AssertionError(f"invalid {name}[{label}]")
        if _data_pointer(pair[0], f"{name}[{label}].left") != _data_pointer(
            pair[1], f"{name}[{label}].right"
        ):
            raise AssertionError(f"{name}[{label}] is not a physical alias")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default=(
            "streaming_quant_li_get_advisory_mirror_adapter:"
            "run_advisory_mirror_isolation"
        ),
        help="advisory mirror runner in module:function form",
    )
    args = parser.parse_args()
    result = _resolve_callable(args.adapter)()
    if not isinstance(result, dict):
        raise TypeError("adapter must return a dictionary")
    _assert_aliases(result, "input_aliases")
    _assert_aliases(result, "transaction_aliases")
    _assert_bitwise_equal(
        result.get("baseline_indices"), result.get("concurrent_exact_indices")
    )
    print(
        "P4_ADVISORY_MIRROR_EXACT_ISOLATION_PASS "
        "official_c8_authority=1 mirror_topk_authority=0 "
        "exact_indices=bitwise_equal no_h2d_claim=1",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            "P4_ADVISORY_MIRROR_EXACT_ISOLATION_FAIL "
            f"type={type(error).__name__} text={error}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
