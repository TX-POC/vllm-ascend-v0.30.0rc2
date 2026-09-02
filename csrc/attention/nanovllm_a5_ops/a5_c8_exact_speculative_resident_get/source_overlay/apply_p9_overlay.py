#!/usr/bin/env python3
"""Apply the minimal, compile-disabled P9 observation seam.

This tool runs *only* on a fresh copy of the exact official C8 source tree.
It intentionally owns no transport ABI: its job is to prove that the P9
operator can preserve the official score and Top-K source while adding one
disabled observation point at the score publication seam.

The transformations are deliberately anchor-based instead of broad patches.
If upstream C8 changes the relevant source shape, this exits before CANN is
invoked rather than producing a subtly different LI binary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable


OFFICIAL_OPERATOR = "VllmQuantLightningIndexer"
P9_OPERATOR = "A5C8ExactSpeculativeResidentGet"
OFFICIAL_SLUG = "vllm_quant_lightning_indexer"
P9_SLUG = "a5_c8_exact_speculative_resident_get"
VECTOR_RELATIVE = Path("op_kernel/arch35/quant_lightning_indexer_service_vector.h")
SEAM_HEADER_RELATIVE = Path(
    "op_kernel/arch35/a5_c8_exact_speculative_resident_score_seam.h"
)
COMPAT_HEADER_RELATIVE = Path(
    "op_host/a5_c8_exact_speculative_resident_host_compat.h"
)
OPS_RELATIVE = Path("ops.json")
SEAM_INCLUDE = '#include "a5_c8_exact_speculative_resident_score_seam.h"\n'
SEAM_MARKER = "A5_C8_EXACT_SPECULATIVE_RESIDENT_OBSERVE_SCORE_SEAM("


class OverlayError(RuntimeError):
    """An exact-source prerequisite was not met."""


def fail(reason: str) -> "None":
    raise OverlayError(reason)


def checked_file(path: Path, label: str) -> Path:
    if not path.is_file():
        fail(f"missing_{label}:path={path}")
    return path


def manifest_entries(manifest: Path, overlay_root: Path) -> list[str]:
    entries: list[str] = []
    for raw in checked_file(manifest, "overlay_manifest").read_text(
        encoding="utf-8"
    ).splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            fail(f"unsafe_manifest_path:value={value}")
        checked_file(overlay_root / candidate, "manifest_entry")
        entries.append(value)
    if not entries:
        fail("empty_overlay_manifest")
    return entries


def function_body(source: str, signature: str) -> str:
    """Return one C++ function definition, including its outer braces.

    The current official function has no raw-string literal containing braces.
    A compact scanner is still safer than a greedy regex and intentionally
    fails closed if the source loses this structure.
    """

    start = source.find(signature)
    if start < 0:
        fail(f"function_signature_missing:value={signature}")
    brace = source.find("{", start)
    if brace < 0:
        fail(f"function_open_brace_missing:value={signature}")
    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
            if depth < 0:
                break
    fail(f"function_close_brace_missing:value={signature}")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def replace_exactly_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        fail(f"{label}_anchor_count:actual={count}")
    return source.replace(old, new, 1)


def rename_source_files(staged_root: Path) -> None:
    """Rename only files whose names encode the original C8 slug."""

    candidates = sorted(
        (
            path
            for path in staged_root.rglob(f"*{OFFICIAL_SLUG}*")
            if path.is_file()
        ),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    if not candidates:
        fail("official_slug_source_files_missing")
    for old_path in candidates:
        new_name = old_path.name.replace(OFFICIAL_SLUG, P9_SLUG)
        new_path = old_path.with_name(new_name)
        if new_path.exists():
            fail(f"renamed_source_already_exists:path={new_path}")
        old_path.rename(new_path)


def replace_operator_identifiers(staged_root: Path) -> None:
    """Rename the cloned operator without touching the frozen Top-K body."""

    source_files = sorted(
        path
        for root in (staged_root / "op_host", staged_root / "op_kernel")
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix in {".h", ".hpp", ".cpp", ".cc", ".cxx", ".asc", ".cmake"}
            or path.name == "CMakeLists.txt"
        )
    )
    source_files.append(staged_root / "CMakeLists.txt")
    for path in source_files:
        checked_file(path, "staged_c8_source")
        text = path.read_text(encoding="utf-8")
        text = text.replace(OFFICIAL_OPERATOR, P9_OPERATOR)
        text = text.replace(OFFICIAL_SLUG, P9_SLUG)
        path.write_text(text, encoding="utf-8")


def inject_score_seam(vector_path: Path) -> tuple[str, str]:
    source = checked_file(vector_path, "official_vector_source").read_text(
        encoding="utf-8"
    )
    topk_signature = "QLIVector<QLIT>::ProcessTopK"
    topk_before = function_body(source, topk_signature)

    source = replace_exactly_once(
        source,
        '#include "quant_lightning_indexer_common.h"\n',
        '#include "quant_lightning_indexer_common.h"\n' + SEAM_INCLUDE,
        "score_seam_include",
    )

    reduction_pattern = re.compile(
        r"(vector1::BatchMulWeightAndReduceSum\(outBase,[\s\S]*?"
        r"gSize_, curAivS1ProcNum\);\n)"
        r"(\s*SetFlag<HardEvent::V_MTE2>\(VEC1_V_MTE2_EVENT \+ pingpong\);)",
        flags=re.MULTILINE,
    )
    matches = list(reduction_pattern.finditer(source))
    if len(matches) != 1:
        fail(f"score_reduction_anchor_count:actual={len(matches)}")
    match = matches[0]
    observation = (
        "    A5_C8_EXACT_SPECULATIVE_RESIDENT_OBSERVE_SCORE_SEAM(\n"
        "        outBase, info, curAivS1Idx, curAivS1ProcNum, curS2Idx);\n"
    )
    source = source[: match.end(1)] + observation + source[match.end(1) :]

    # The source ordering is the substantive seam proof: reduction -> marker
    # -> official V/MTE hand-off -> score GM publication -> untouched Top-K.
    reduction_index = source.index("vector1::BatchMulWeightAndReduceSum")
    marker_index = source.index(SEAM_MARKER, reduction_index)
    publication_index = source.index("DataCopyPad(scoreGm", marker_index)
    topk_index = source.index(topk_signature)
    if not (reduction_index < marker_index < publication_index < topk_index):
        fail("score_seam_source_order_invalid")

    topk_after = function_body(source, topk_signature)
    if topk_before != topk_after:
        fail(
            "official_process_topk_changed:before="
            f"{sha256_text(topk_before)}:after={sha256_text(topk_after)}"
        )
    vector_path.write_text(source, encoding="utf-8")
    return sha256_text(topk_before), sha256_text(source)


def copy_overlay_input(overlay_root: Path, staged_root: Path) -> None:
    shutil.copy2(
        checked_file(overlay_root / OPS_RELATIVE, "overlay_ops_json"),
        staged_root / OPS_RELATIVE,
    )
    seam_destination = staged_root / SEAM_HEADER_RELATIVE
    seam_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        checked_file(overlay_root / SEAM_HEADER_RELATIVE, "overlay_seam_header"),
        seam_destination,
    )

    compat_source = checked_file(
        overlay_root / COMPAT_HEADER_RELATIVE, "overlay_host_compat"
    )
    for destination in (
        staged_root / "op_host/err/ops_err.h",
        staged_root / "op_host/log/log.h",
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(compat_source, destination)


def validate_output(staged_root: Path, topk_hash: str) -> None:
    vector_path = staged_root / VECTOR_RELATIVE
    vector = checked_file(vector_path, "staged_vector_source").read_text(
        encoding="utf-8"
    )
    if vector.count(SEAM_INCLUDE.rstrip("\n")) != 1:
        fail("staged_score_seam_include_count_invalid")
    if vector.count(SEAM_MARKER) != 1:
        fail("staged_score_seam_marker_count_invalid")
    if sha256_text(function_body(vector, "QLIVector<QLIT>::ProcessTopK")) != topk_hash:
        fail("staged_process_topk_digest_changed")

    ops = json.loads(checked_file(staged_root / OPS_RELATIVE, "staged_ops_json").read_text(encoding="utf-8"))
    if len(ops) != 1 or ops[0].get("op") != P9_OPERATOR:
        fail("staged_ops_json_operator_invalid")
    for relative in (
        "op_host/err/ops_err.h",
        "op_host/log/log.h",
        "op_host/a5_c8_exact_speculative_resident_get_def.cpp",
        "op_kernel/a5_c8_exact_speculative_resident_get.cpp",
        str(SEAM_HEADER_RELATIVE),
    ):
        checked_file(staged_root / relative, "staged_overlay_output")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged-c8-root", required=True, type=Path)
    parser.add_argument("--overlay-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--operator", default=P9_OPERATOR)
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    staged_root = args.staged_c8_root.resolve()
    overlay_root = args.overlay_root.resolve()
    if args.operator != P9_OPERATOR:
        fail(f"unexpected_operator:value={args.operator}")
    checked_file(staged_root / "CMakeLists.txt", "staged_c8_root")
    manifest_entries(args.manifest.resolve(), overlay_root)

    copy_overlay_input(overlay_root, staged_root)
    topk_hash, vector_hash = inject_score_seam(staged_root / VECTOR_RELATIVE)
    rename_source_files(staged_root)
    replace_operator_identifiers(staged_root)
    validate_output(staged_root, topk_hash)

    print(
        "P9_OVERLAY_APPLY_PASS "
        f"operator={P9_OPERATOR} seam=compile_disabled "
        f"topk_sha256={topk_hash} vector_sha256={vector_hash} "
        f"staged_c8={staged_root}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except OverlayError as error:
        print(f"P9_OVERLAY_APPLY_FAIL reason={error}", file=sys.stderr)
        raise SystemExit(2)
