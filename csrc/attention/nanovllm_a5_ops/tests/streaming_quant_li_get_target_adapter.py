#!/usr/bin/env python3
"""Target-only adapter for the A5StreamingQuantLiGet equivalence harness.

This module deliberately does not register, monkey-patch, or select any
vLLM product route.  It performs one controlled target-NPU comparison:

1. run the real A5 ``VllmQuantLightningIndexerMetadata`` companion and the
   real A5 ``VllmQuantLightningIndexer`` reference;
2. switch the *isolated bridge resolver only* to the separately built
   ``A5StreamingQuantLiGet`` OPP;
3. run the derived 24-input / 11-output operator with the very same nine C8
   input allocations; and
4. return the strict dictionary consumed by
   ``test_streaming_quant_li_get_target_equivalence.py``.

For the P5 integration gate, the returned dictionary also exposes the three
*same-allocation* publication tensors.  This is intentionally a narrow ABI
handoff: P5 receives the C8-written score workspace and fixed tile receipts
without repacking scores through Python or manufacturing a second receipt
surface.  The static score-source lengths remain a separate scheduler-owned
input on the P5 side; P5 must not use the derived terminal output as a live
producer dependency.

The P3 operator has two independent build-time selectors.  Publication modes
are exercised through ``P3_PUBLICATION_MODE``:

* ``off``: original private score workspace, no externally visible change;
* ``score_workspace``: score DMA targets caller-owned workspace, no receipt;
* ``publisher``: score workspace plus a fixed 16-word receipt for each score
  tile, published after that tile's score DMA.

The real score-tile inline modes are selected through ``P3_INLINE_MODE``:

* ``off``: no KVAF/SIMT closure; this adapter's strict C8 control/publisher
  proof;
* ``audit``: tile-local score/threshold/mapping audit, exercised only by the
  P4 audit adapter; and
* ``post``: native-enqueue Direct-HCCP, not accepted by this correctness-only
  adapter until payload correctness is separately established.

This adapter proves only official C8 Top-K equivalence, output aliases,
exact score-source lengths, and the selected publication contract.  It does
not claim that a later sidecar has posted an H2D request or that H2D payloads
are correct.

Before importing this module, build the two isolated OPPs and set:

* ``NANOVLLM_A5_REFERENCE_C8_OPP`` to the source-matched official C8 OPP;
* ``NANOVLLM_A5_STREAMING_LI_OPP`` to the derived P3 OPP; and
* ``P3_REFERENCE_EXPECTED_SOURCE_HEAD`` to the current P3 source revision.

The reference is then launched through a standalone ACLNN bridge compiled
from this worktree's ``csrc/torch_binding.cpp`` contract.  In particular, this
adapter does **not** import ``vllm_ascend`` or reuse an editable package from
another worktree.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from time import monotonic
from pathlib import Path
from typing import Any

import torch
import torch_npu  # noqa: F401  # Registers the NPU backend and FP8 quantizer.


BLOCK_SIZE = 128
HEAD_DIM = 128
# This fixture must be accepted by the real official A5 C8 tiler before it
# can serve as a reference for the derived operator.  For PA_BSND, that
# tiler requires the exact grouping N1 / N2 == 64; it also currently accepts
# only N2 == 1.  Keep these two dimensions explicit so a future fixture edit
# fails at construction time instead of after an expensive target launch.
INDEX_HEADS = 64
KEY_HEADS = 1
A5_C8_QUERY_TO_KEY_GROUP = 64
SPARSE_COUNT = 2048
BATCH_SIZE = 2
SOURCE_TOKENS = 4096
METADATA_WORDS = 1024
PACKED_KV_ROW_BYTES = 656
QUEUE_COUNT = 1
PRIVATE_ARENA_CAPACITY = 1
STREAMING_CONTROL_WORDS = 16
STREAMING_CONTROL_MAGIC = 0x4B5641465354524D  # Legacy ABI placeholder only.
STREAMING_CONTROL_VERSION = 1
INT64_MAX = (1 << 63) - 1
REFERENCE_OPP_ENV = "NANOVLLM_A5_REFERENCE_C8_OPP"
TARGET_OPP_ENV = "NANOVLLM_A5_STREAMING_LI_OPP"
REFERENCE_EXPECTED_HEAD_ENV = "P3_REFERENCE_EXPECTED_SOURCE_HEAD"
TARGET_EXPECTED_HEAD_ENV = "P3_TARGET_EXPECTED_SOURCE_HEAD"
KVAF_EXPECTED_HEAD_ENV = "P3_KVAF_EXPECTED_SOURCE_HEAD"
REFERENCE_PROVENANCE_NAME = "P3_REFERENCE_PROVENANCE"
TARGET_PROVENANCE_NAME = "P3_STREAMING_LI_PROVENANCE"
REFERENCE_OPAPI_ENV = "NANOVLLM_A5_REFERENCE_C8_OPAPI_LIB"
TARGET_NPU_ENV = "P3_NPU_ID"
PHYSICAL_TARGET_NPU_ENV = "P3_PHYSICAL_NPU_ID"
PUBLICATION_MODE_ENV = "P3_PUBLICATION_MODE"
PUBLICATION_MODES = ("off", "score_workspace", "publisher")
PUBLICATION_MODE_TO_BUILD_VALUE = {
    "off": "0",
    "score_workspace": "1",
    "publisher": "2",
}
INLINE_MODE_ENV = "P3_INLINE_MODE"
INLINE_MODES = ("off", "audit", "post")
INLINE_MODE_TO_BUILD_VALUE = {
    "off": "0",
    "audit": "1",
    "post": "2",
}
POST_CLOSURE_MODE_ENV = "P3_POST_CLOSURE_MODE"
POST_CLOSURE_MODE_TO_BUILD_VALUE = {
    "not_applicable": "0",
    "empty_shell": "0",
    "full": "1",
}
SCORE_WORKSPACE_SENTINEL = -0x3A5A5A5A
TILE_RECEIPT_SENTINEL = -1
TILE_RECEIPT_WORDS = 16
TILE_SCORE_WIDTH = 128
TILE_RECEIPT_CONTRACT_VERSION = 1
STREAMING_EPOCH = 0x50330001
# C8 reserves score-workspace metadata for up to 36 AICs.  This is a capacity,
# not the number of active AICs on every 950DT: runtime tiling reads that
# geometry from PlatformAscendC.  Keeping the maximum here prevents a smaller
# active geometry from under-allocating the graph-visible workspace.
A5_MAX_AIC_CORE_COUNT = 36
# Compatibility exports for legacy P4 scripts. They are deliberately no
# longer consumed by this strict P3 adapter.
HOOK_CONTROL_MODE_ENV = "P3_HOOK_CONTROL_MODE"
HOOK_CONTROL_MODE_ALL_BUSY = "valid_all_busy"
HOOK_CONTROL_MODE_REJECT_BUSY = "reject_all_busy"
ALLOW_HOOK_OFF_BASELINE_ENV = "P3_ALLOW_HOOK_OFF_BASELINE"
NO_POST_TRANSACTION_NAMES = (
    "private_source_to_slot",
    "private_ready_table",
    "private_arena_head",
    "private_packed_output",
    "queue_leases",
)
_SELECTED_PHYSICAL_NPU_ID: int | None = None


def _parse_nonnegative_npu_id(name: str, value: str | None) -> int:
    if value is None:
        raise RuntimeError(f"set {name} to a non-negative integer")
    try:
        parsed = int(value)
    except ValueError as error:
        raise RuntimeError(
            f"{name} must be a non-negative integer, got {value!r}"
        ) from error
    if parsed < 0:
        raise RuntimeError(
            f"{name} must be a non-negative integer, got {parsed}"
        )
    return parsed


def _visible_physical_npus() -> tuple[str, tuple[int, ...]] | None:
    """Return the explicit logical-to-physical mask, if one is present.

    ``ASCEND_RT_VISIBLE_DEVICES=7`` deliberately makes physical NPU 7 appear
    as logical ``npu:0``.  The strict runner may use that isolation, but its
    proof must still state the physical card rather than relabelling it as 0.
    """
    raw_rt = os.getenv("ASCEND_RT_VISIBLE_DEVICES")
    raw_legacy = os.getenv("ASCEND_VISIBLE_DEVICES")
    raw = raw_rt if raw_rt else raw_legacy
    if not raw:
        return None
    source = "ASCEND_RT_VISIBLE_DEVICES" if raw_rt else "ASCEND_VISIBLE_DEVICES"
    fields = tuple(part.strip() for part in raw.split(","))
    if not fields or any(not field for field in fields):
        raise RuntimeError(f"{source} has an invalid empty device entry: {raw!r}")
    try:
        physical = tuple(int(field) for field in fields)
    except ValueError as error:
        raise RuntimeError(
            f"{source} must be a comma-separated physical NPU-id list; got {raw!r}"
        ) from error
    if any(value < 0 for value in physical) or len(set(physical)) != len(physical):
        raise RuntimeError(f"{source} has invalid physical NPU ids: {raw!r}")
    return source, physical


def _require_a5_npu() -> torch.device:
    global _SELECTED_PHYSICAL_NPU_ID
    requested_text = os.getenv(TARGET_NPU_ENV)
    if requested_text is None:
        raise RuntimeError(
            f"set {TARGET_NPU_ENV} to the logical A5 NPU id visible to this "
            "process; this target test must not silently use npu:0"
        )
    requested_id = _parse_nonnegative_npu_id(TARGET_NPU_ENV, requested_text)
    mapping = _visible_physical_npus()
    explicit_physical_text = os.getenv(PHYSICAL_TARGET_NPU_ENV)
    explicit_physical = (
        _parse_nonnegative_npu_id(PHYSICAL_TARGET_NPU_ENV, explicit_physical_text)
        if explicit_physical_text is not None
        else None
    )
    if mapping is None:
        physical_id = requested_id
        if explicit_physical is not None and explicit_physical != physical_id:
            raise RuntimeError(
                f"{PHYSICAL_TARGET_NPU_ENV}={explicit_physical} disagrees with "
                f"unmasked {TARGET_NPU_ENV}={requested_id}"
            )
        mapping_text = "<unset>"
    else:
        mapping_env, mapped_physical = mapping
        if requested_id >= len(mapped_physical):
            raise RuntimeError(
                f"{TARGET_NPU_ENV}={requested_id} is outside the logical range "
                f"[0,{len(mapped_physical)}) selected by {mapping_env}"
            )
        physical_id = mapped_physical[requested_id]
        if explicit_physical is not None and explicit_physical != physical_id:
            raise RuntimeError(
                f"{PHYSICAL_TARGET_NPU_ENV}={explicit_physical} disagrees with "
                f"{mapping_env} logical={requested_id} -> physical={physical_id}"
            )
        mapping_text = f"{mapping_env}={','.join(str(value) for value in mapped_physical)}"

    device_count = int(torch.npu.device_count())
    # A caller that asks for an explicit physical proof must use the strict
    # one-card runtime mapping.  Without this, a multi-card mask can still
    # print the correct first physical id while leaving the process capable of
    # submitting work to another card later in the test.
    if explicit_physical is not None and mapping is not None:
        if (
            mapping_env != "ASCEND_RT_VISIBLE_DEVICES"
            or requested_id != 0
            or mapped_physical != (explicit_physical,)
            or device_count != 1
        ):
            raise RuntimeError(
                "explicit physical P3 proof requires exactly one runtime-visible "
                "card: use ASCEND_RT_VISIBLE_DEVICES=<physical>, P3_NPU_ID=0, "
                "P3_PHYSICAL_NPU_ID=<same physical>; got "
                f"mapping={mapping_text!r} logical={requested_id} "
                f"physical={explicit_physical} device_count={device_count}"
            )
    if requested_id >= device_count:
        raise RuntimeError(
            f"{TARGET_NPU_ENV}={requested_id} is outside the visible logical "
            f"NPU range [0,{device_count})"
        )
    torch.npu.set_device(requested_id)
    current_id = int(torch.npu.current_device())
    if current_id != requested_id:
        raise RuntimeError(
            "failed to select requested logical NPU: "
            f"requested={requested_id} current={current_id}"
        )
    device = torch.device(f"npu:{requested_id}")
    getter = getattr(torch.npu, "get_device_name", torch_npu.npu.get_device_name)
    name = str(getter(requested_id))
    if "950" not in name.lower():
        raise RuntimeError(
            "target equivalence requires an Ascend 950 NPU, "
            f"got {name!r}"
        )
    _SELECTED_PHYSICAL_NPU_ID = physical_id
    print(
        "P3_TARGET_NPU "
        f"requested_logical={requested_id} requested_physical={physical_id} "
        f"physical_npu={physical_id} current_device={current_id} "
        f"device={device} visible_mapping={mapping_text!r} device_name={name!r}",
        flush=True,
    )
    return device


def _quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the C8 FP8 tensor and the FP32 per-token/head scale view."""
    quantized, scale = torch_npu.npu_dynamic_quant(
        tensor, dst_type=torch.float8_e4m3fn
    )
    return (
        quantized.contiguous(),
        scale.view(tensor.shape[:-1]).to(torch.float32).contiguous(),
    )


def _resolve_single_opp(root_value: str | None, label: str) -> tuple[Path, Path]:
    """Return exactly one ``(opapi, vendor)`` pair below an isolated OPP root."""
    if not root_value:
        raise RuntimeError(
            f"set {REFERENCE_OPP_ENV if label == 'reference' else TARGET_OPP_ENV} "
            f"to the built isolated {label} OPP root or vendor directory"
        )
    root = Path(root_value).expanduser().resolve()
    candidates = [root / "op_api" / "lib" / "libcust_opapi.so"]
    candidates.extend(root.glob("vendors/*/op_api/lib/libcust_opapi.so"))
    opapis = tuple(
        sorted({candidate.resolve() for candidate in candidates if candidate.is_file()})
    )
    if len(opapis) != 1:
        raise RuntimeError(
            f"expected exactly one {label} libcust_opapi.so below {root}, "
            f"found={len(opapis)}"
        )
    opapi = opapis[0]
    vendor = opapi.parents[2]
    if not vendor.is_dir():
        raise RuntimeError(f"{label} OPP vendor path is invalid: {vendor}")
    return opapi, vendor


def _prepend_unique_path(name: str, paths: tuple[Path, ...]) -> None:
    """Install both isolated vendors before the first ACLNN custom-op call."""
    existing = [entry for entry in os.getenv(name, "").split(":") if entry]
    ordered = [str(path) for path in paths]
    os.environ[name] = ":".join(
        [*ordered, *(entry for entry in existing if entry not in ordered)]
    )


def _read_reference_provenance(vendor: Path) -> dict[str, str]:
    path = vendor / REFERENCE_PROVENANCE_NAME
    if not path.is_file():
        raise RuntimeError(
            "source-matched reference OPP has no provenance manifest: " f"{path}"
        )
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            values[key] = value
    required = ("kind", "source_root", "source_head", "operators")
    missing = tuple(key for key in required if not values.get(key))
    if missing:
        raise RuntimeError(
            "source-matched reference provenance is incomplete: "
            f"path={path} missing={missing}"
        )
    if values["kind"] != "source_matched_official_c8":
        raise RuntimeError(
            "reference OPP provenance kind is not source-matched official C8: "
            f"got={values['kind']!r}"
        )
    expected_head = os.getenv(REFERENCE_EXPECTED_HEAD_ENV)
    if expected_head and values["source_head"] != expected_head:
        raise RuntimeError(
            "reference OPP was built from a different source revision: "
            f"actual={values['source_head']} expected={expected_head}"
        )
    return values


def _expected_target_hook_mode() -> tuple[str, bool]:
    """Return the fail-closed hook expectation for this target process.

    ``hook_mode=off`` is never a product option.  It is accepted only for a
    fresh-process mathematical baseline with an explicit ``=1`` opt-in, so an
    old diagnostic environment cannot silently turn a normal P3 proof into a
    hook-free one.
    """
    raw_value = os.getenv(ALLOW_HOOK_OFF_BASELINE_ENV)
    if raw_value in (None, "", "0"):
        return "on", False
    if raw_value == "1":
        return "off", True
    raise RuntimeError(
        f"{ALLOW_HOOK_OFF_BASELINE_ENV} must be unset, '0', or '1'; "
        f"got {raw_value!r}"
    )


def _resolve_publication_mode() -> str:
    """Return the explicit build-time publication mode under test.

    The mode belongs to the separately built OPP binary, not to a runtime
    branch. The provenance manifest must match this value exactly.
    """
    mode = os.getenv(PUBLICATION_MODE_ENV)
    if mode not in PUBLICATION_MODES:
        raise RuntimeError(
            f"set {PUBLICATION_MODE_ENV} to one of {PUBLICATION_MODES}; "
            f"got {mode!r}"
        )
    return mode


def _resolve_inline_mode() -> str:
    """Return the explicit score-tile inline mode selected for this process."""
    mode = os.getenv(INLINE_MODE_ENV)
    if mode not in INLINE_MODES:
        raise RuntimeError(
            f"set {INLINE_MODE_ENV} to one of {INLINE_MODES}; got {mode!r}"
        )
    return mode


def _resolve_post_closure_mode(inline_mode: str) -> str:
    """Return the expected compile-time post closure for this test process."""
    raw_mode = os.getenv(POST_CLOSURE_MODE_ENV)
    if inline_mode != "post":
        if raw_mode not in (None, "", "not_applicable"):
            raise RuntimeError(
                f"{POST_CLOSURE_MODE_ENV} is valid only with "
                f"P3_INLINE_MODE=post; got {raw_mode!r}"
            )
        return "not_applicable"
    mode = raw_mode or "full"
    if mode not in ("empty_shell", "full"):
        raise RuntimeError(
            f"set {POST_CLOSURE_MODE_ENV} to empty_shell or full for "
            f"P3_INLINE_MODE=post; got {mode!r}"
        )
    return mode


def _read_target_provenance(
    vendor: Path, expected_publication_mode: str, expected_inline_mode: str
) -> dict[str, str]:
    """Reject a stale P3 OPP or a build-mode mismatch.

    A P3 artifact's source revision alone is insufficient: publication and
    inline KVAF closures can produce the same public 24/11 ABI while having
    intentionally different score-tile execution semantics.
    """
    if expected_publication_mode not in PUBLICATION_MODES:
        raise RuntimeError(
            "P3 publication provenance is invalid; use P3_PUBLICATION_MODE=off, "
            "score_workspace, or publisher"
        )
    if expected_inline_mode not in INLINE_MODES:
        raise RuntimeError(
            "P3 inline provenance is invalid; use P3_INLINE_MODE=off, audit, or post"
        )
    path = vendor / TARGET_PROVENANCE_NAME
    if not path.is_file():
        raise RuntimeError(
            "isolated P3 target OPP has no provenance manifest: " f"{path}"
        )
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            values[key] = value
    required = (
        "format",
        "kind",
        "source_root",
        "source_head",
        "source_clean",
        "operators",
        "publication_mode",
        "publication_mode_value",
        "inline_mode",
        "inline_mode_value",
        "post_closure_mode",
        "post_closure_mode_value",
        "kvaf_closure",
        "kvaf_source_head",
        "abi",
    )
    missing = tuple(key for key in required if not values.get(key))
    if missing:
        raise RuntimeError(
            "isolated P3 target provenance is incomplete: "
            f"path={path} missing={missing}; rebuild with the publication "
            "aware P3 OPP build script"
        )
    if values["kind"] != "source_derived_streaming_quant_li_get":
        raise RuntimeError(
            "isolated P3 target provenance has an unexpected kind: "
            f"got={values['kind']!r}"
        )
    if values["format"] != "1" or values["abi"] != "24in_11out":
        raise RuntimeError(
            "isolated P3 target provenance has an incompatible manifest ABI: "
            f"format={values['format']!r} abi={values['abi']!r}"
        )
    if values["operators"] != "A5StreamingQuantLiGet":
        raise RuntimeError(
            "isolated P3 target provenance contains an unexpected operator set: "
            f"operators={values['operators']!r}"
        )
    expected_head = os.getenv(TARGET_EXPECTED_HEAD_ENV)
    if expected_head and values["source_head"] != expected_head:
        raise RuntimeError(
            "isolated P3 target OPP was built from a different source revision: "
            f"actual={values['source_head']} expected={expected_head}"
        )
    if values["source_clean"] != "1":
        raise RuntimeError(
            "isolated P3 target OPP was not built from a tracked-clean source "
            f"worktree: source_clean={values['source_clean']!r}"
        )
    if values["publication_mode"] != expected_publication_mode:
        raise RuntimeError(
            "isolated P3 target OPP has an unexpected publication mode: "
            f"actual={values['publication_mode']!r} "
            f"expected={expected_publication_mode!r}"
        )
    build_value = values["publication_mode_value"]
    expected_value = PUBLICATION_MODE_TO_BUILD_VALUE[expected_publication_mode]
    if build_value != expected_value:
        raise RuntimeError(
            "isolated P3 target provenance maps publication mode to the "
            "wrong compile-time value: "
            f"mode={expected_publication_mode!r} actual={build_value!r} "
            f"expected={expected_value!r}"
        )
    if values["inline_mode"] != expected_inline_mode:
        raise RuntimeError(
            "isolated P3 target OPP has an unexpected inline mode: "
            f"actual={values['inline_mode']!r} expected={expected_inline_mode!r}"
        )
    inline_value = values["inline_mode_value"]
    expected_inline_value = INLINE_MODE_TO_BUILD_VALUE[expected_inline_mode]
    if inline_value != expected_inline_value:
        raise RuntimeError(
            "isolated P3 target provenance maps inline mode to the wrong "
            f"compile-time value: mode={expected_inline_mode!r} "
            f"actual={inline_value!r} expected={expected_inline_value!r}"
        )
    expected_post_closure_mode = _resolve_post_closure_mode(expected_inline_mode)
    if values["post_closure_mode"] != expected_post_closure_mode:
        raise RuntimeError(
            "isolated P3 target OPP has an unexpected post closure mode: "
            f"inline_mode={expected_inline_mode!r} "
            f"actual={values['post_closure_mode']!r} "
            f"expected={expected_post_closure_mode!r}"
        )
    expected_post_closure_value = POST_CLOSURE_MODE_TO_BUILD_VALUE[
        expected_post_closure_mode
    ]
    if values["post_closure_mode_value"] != expected_post_closure_value:
        raise RuntimeError(
            "isolated P3 target provenance maps post closure mode to the "
            "wrong compile-time value: "
            f"mode={expected_post_closure_mode!r} "
            f"actual={values['post_closure_mode_value']!r} "
            f"expected={expected_post_closure_value!r}"
        )
    expected_kvaf_closure = (
        "staged" if expected_post_closure_mode == "full" else "absent"
    )
    if values["kvaf_closure"] != expected_kvaf_closure:
        raise RuntimeError(
            "isolated P3 target has an unexpected KVAF transport closure: "
            f"inline_mode={expected_inline_mode!r} "
            f"post_closure_mode={expected_post_closure_mode!r} "
            f"actual={values['kvaf_closure']!r} "
            f"expected={expected_kvaf_closure!r}"
        )
    if expected_kvaf_closure == "absent" and values["kvaf_source_head"] != "not_applicable":
        raise RuntimeError(
            "isolated P3 no-transport artifact unexpectedly records a KVAF "
            f"source: kvaf_source_head={values['kvaf_source_head']!r}"
        )
    if expected_kvaf_closure == "staged":
        kvaf_head = values["kvaf_source_head"]
        if not re.fullmatch(r"[0-9a-f]{40}", kvaf_head):
            raise RuntimeError(
                "isolated P3 post artifact has no exact KVAF source revision: "
                f"kvaf_source_head={kvaf_head!r}"
            )
        expected_kvaf_head = os.getenv(KVAF_EXPECTED_HEAD_ENV)
        if expected_kvaf_head and kvaf_head != expected_kvaf_head:
            raise RuntimeError(
                "isolated P3 post artifact was built from a different KVAF "
                f"revision: actual={kvaf_head} expected={expected_kvaf_head}"
            )
    return values


def _require_source_matched_official_c8_ops(
    reference_opapi: Path, reference_vendor: Path
) -> tuple[Any, Any, dict[str, str]]:
    """Load the current-source official C8 bridge, never ``vllm_ascend``."""
    provenance = _read_reference_provenance(reference_vendor)
    if "nanovllm_official_c8_reference" in sys.modules:
        raise RuntimeError(
            "run target equivalence in a fresh Python process: the source-matched "
            "official C8 bridge resolver was already initialized"
        )
    os.environ[REFERENCE_OPAPI_ENV] = str(reference_opapi)
    importlib.import_module("nanovllm_official_c8_reference")
    namespace = getattr(torch.ops, "a5_official_c8_reference", None)
    reference = (
        getattr(namespace, "npu_vllm_quant_lightning_indexer", None)
        if namespace is not None
        else None
    )
    metadata = (
        getattr(namespace, "npu_vllm_quant_lightning_indexer_metadata", None)
        if namespace is not None
        else None
    )
    if reference is None or metadata is None:
        raise RuntimeError(
            "standalone source-matched official C8 bridge did not register both "
            "VllmQuantLightningIndexer operators"
        )
    print(
        "P3_REFERENCE_PROVENANCE "
        "implementation=standalone_source_opp "
        f"source_root={provenance['source_root']} "
        f"source_head={provenance['source_head']} "
        f"expected_head={os.getenv(REFERENCE_EXPECTED_HEAD_ENV, '<unset>')} "
        "ready=1 "
        f"vendor={reference_vendor} opapi={reference_opapi}",
        flush=True,
    )
    return reference, metadata, provenance


def _build_official_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    """Create one real TND/PA_BSND C8 case shared byte-for-byte by both ops."""
    if SOURCE_TOKENS % BLOCK_SIZE:
        raise AssertionError("SOURCE_TOKENS must be page aligned")
    if KEY_HEADS != 1 or INDEX_HEADS != KEY_HEADS * A5_C8_QUERY_TO_KEY_GROUP:
        raise AssertionError(
            "official A5 C8 PA_BSND fixture requires N2=1 and N1/N2=64; "
            f"got N1={INDEX_HEADS}, N2={KEY_HEADS}"
        )
    blocks = SOURCE_TOKENS // BLOCK_SIZE
    torch.manual_seed(20260826)
    torch.npu.manual_seed_all(20260826)

    query_fp = torch.empty(
        (BATCH_SIZE, INDEX_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-1.0, 1.0)
    key_fp = torch.empty(
        (blocks, BLOCK_SIZE, KEY_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-1.0, 1.0)
    query, query_scale = _quantize_fp8(query_fp)
    key, key_scale = _quantize_fp8(key_fp)
    weights = torch.empty(
        (BATCH_SIZE, INDEX_HEADS), dtype=torch.float32, device=device
    ).uniform_(0.01, 1.0).contiguous()

    # TND decode has one query row per request.  The final row has a shorter
    # logical/source prefix, exercising that score_source_lens is per row.
    candidate_lengths = (SOURCE_TOKENS, SOURCE_TOKENS - BLOCK_SIZE)
    actual_q = torch.arange(1, BATCH_SIZE + 1, dtype=torch.int32, device=device)
    actual_k = torch.tensor(candidate_lengths, dtype=torch.int32, device=device)
    block_table = torch.stack(
        [torch.randperm(blocks, dtype=torch.int64).to(torch.int32)
         for _ in range(BATCH_SIZE)]
    ).to(device)
    return {
        "query": query,
        "key": key,
        "weights": weights,
        "query_dequant_scale": query_scale,
        "key_dequant_scale": key_scale,
        "actual_seq_lengths_query": actual_q,
        "actual_seq_lengths_key": actual_k,
        "block_table": block_table,
    }


def _official_metadata(metadata_op: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Use the registered official metadata companion, never a host stand-in."""
    metadata = metadata_op(
        num_heads_q=INDEX_HEADS,
        num_heads_k=KEY_HEADS,
        head_dim=HEAD_DIM,
        query_quant_mode=0,
        key_quant_mode=0,
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_key=inputs["actual_seq_lengths_key"],
        batch_size=BATCH_SIZE,
        max_seqlen_q=1,
        max_seqlen_k=SOURCE_TOKENS,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        cmp_ratio=1,
        device=str(inputs["query"].device),
    )
    if (
        not isinstance(metadata, torch.Tensor)
        or metadata.dtype != torch.int32
        or metadata.numel() != METADATA_WORDS
        or metadata.device != inputs["query"].device
    ):
        raise RuntimeError(
            "official C8 metadata companion returned an unexpected tensor: "
            f"type={type(metadata).__name__} "
            f"shape={getattr(metadata, 'shape', None)} "
            f"dtype={getattr(metadata, 'dtype', None)}"
        )
    return metadata.contiguous()


def _call_official(reference_op: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    result = reference_op(
        query=inputs["query"],
        key=inputs["key"],
        weights=inputs["weights"],
        query_dequant_scale=inputs["query_dequant_scale"],
        key_dequant_scale=inputs["key_dequant_scale"],
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_key=inputs["actual_seq_lengths_key"],
        block_table=inputs["block_table"],
        metadata=inputs["metadata"],
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        cmp_ratio=1,
        return_value=False,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "official A5 C8 LI did not return its registered two-output ABI"
        )
    indices = result[0]
    if not isinstance(indices, torch.Tensor) or indices.dtype != torch.int32:
        raise RuntimeError("official A5 C8 LI returned no int32 sparse_indices")
    expected_shape = (BATCH_SIZE, KEY_HEADS, SPARSE_COUNT)
    if tuple(indices.shape) != expected_shape:
        raise RuntimeError(
            "official A5 C8 LI sparse_indices shape mismatch: "
            f"got={tuple(indices.shape)} expected={expected_shape}"
        )
    return indices


def _bind_isolated_target_opp(opapi: Path) -> Path:
    """Switch only the target bridge's resolver after the reference launch."""
    # ``nanovllm_a5_opapi::RepositoryCustomHandle`` is static per extension,
    # not process-global.  The source-matched bridge already cached its DSO,
    # so this cannot redirect the completed reference call.
    if "nanovllm_streaming_quant_li_get" in sys.modules:
        raise RuntimeError(
            "run target equivalence in a fresh Python process: the isolated "
            "P3 bridge resolver was already initialized"
        )
    os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(opapi)
    return opapi


def _resolve_hook_control_mode() -> str:
    mode = os.getenv(HOOK_CONTROL_MODE_ENV, HOOK_CONTROL_MODE_ALL_BUSY)
    if mode not in (HOOK_CONTROL_MODE_ALL_BUSY, HOOK_CONTROL_MODE_REJECT_BUSY):
        raise RuntimeError(
            f"{HOOK_CONTROL_MODE_ENV} must be one of "
            f"{HOOK_CONTROL_MODE_ALL_BUSY!r}, {HOOK_CONTROL_MODE_REJECT_BUSY!r}; "
            f"got {mode!r}"
        )
    return mode


def _make_no_post_transaction(
    device: torch.device,
    row_count: int,
    control_mode: str,
    *,
    object_mapping: str = "identity",
) -> dict[str, torch.Tensor]:
    """Build a valid-shape transaction whose queue cannot be leased.

    A full Direct-HCCP control must be cold-bound by KVAF; this isolated math
    comparator intentionally does not claim to reproduce that lifecycle.  A
    busy lease table makes ``TryAcquire...`` fail before any descriptor pointer
    is dereferenced, while every shape and mandatory control invariant seen by
    the A5 hook remains explicit and testable.
    """
    pool_rows = BATCH_SIZE
    if object_mapping not in ("identity", "permuted_pool"):
        raise ValueError(
            "object_mapping must be 'identity' or 'permuted_pool', got "
            f"{object_mapping!r}"
        )
    if row_count != BATCH_SIZE or pool_rows != BATCH_SIZE:
        raise RuntimeError(
            "the controlled C8 fixture requires exactly its fixed batch size: "
            f"rows={row_count} batch={BATCH_SIZE}"
        )
    kv_object_lens = torch.tensor(
        (SOURCE_TOKENS, SOURCE_TOKENS - BLOCK_SIZE),
        dtype=torch.int32,
        device=device,
    )
    if object_mapping == "identity":
        req_pool_entries = torch.arange(
            row_count, dtype=torch.int32, device=device
        )
        kv_object_map = torch.arange(
            SOURCE_TOKENS, dtype=torch.int32, device=device
        ).repeat(pool_rows, 1).contiguous()
    else:
        # Exercise the same row -> pool -> KV-object indirection as the real
        # transaction.  This is deliberately not a source-index identity
        # map: an audit that accidentally reads slots[row, source] must fail.
        # Each active row remains a bijection over its own active object
        # prefix, while the unused static tail stays invalid.
        req_pool_entries = torch.arange(
            row_count, dtype=torch.int32, device=device
        ).flip(0)
        kv_object_map = torch.full(
            (pool_rows, SOURCE_TOKENS), -1, dtype=torch.int32, device=device
        )
        for row, object_len in enumerate((SOURCE_TOKENS, SOURCE_TOKENS - BLOCK_SIZE)):
            pool_row = int(req_pool_entries[row].item())
            source = torch.arange(object_len, dtype=torch.int32, device=device)
            # A reverse permutation is easy to audit in a compact log while
            # ensuring both source zero and the request pool are nonidentity.
            kv_object_map[pool_row, :object_len] = object_len - 1 - source
    # A sentinel makes the no-post assertion meaningful: a mistaken write is
    # visible even if the accelerator allocator would otherwise return a
    # zeroed allocation.
    private_packed_output = torch.full(
        (QUEUE_COUNT * PRIVATE_ARENA_CAPACITY, PACKED_KV_ROW_BYTES),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )
    # Keep the lease busy in both diagnostic modes. This remains safe even if
    # a future control validator regresses: no fabricated descriptor can be
    # acquired or posted. The reject mode differs only in that Init must set
    # enabled_=false and increment kControlRejected telemetry.
    queue_leases = torch.ones(QUEUE_COUNT, dtype=torch.int32, device=device)
    safe_nonzero_pointer = int(private_packed_output.data_ptr())
    control_magic = (
        STREAMING_CONTROL_MAGIC
        if control_mode == HOOK_CONTROL_MODE_ALL_BUSY
        else 0
    )
    control_words = [
        control_magic,
        STREAMING_CONTROL_VERSION,
        safe_nonzero_pointer,  # channel_entity, never dereferenced: queue busy
        safe_nonzero_pointer,  # queue_state, never dereferenced: queue busy
        safe_nonzero_pointer,  # core_status, never dereferenced: queue busy
        safe_nonzero_pointer,  # output: must exactly match packed output
        int(queue_leases.data_ptr()),
        safe_nonzero_pointer,  # descriptor_scratch, never dereferenced
        QUEUE_COUNT,
        QUEUE_COUNT * PRIVATE_ARENA_CAPACITY,
        PRIVATE_ARENA_CAPACITY * PACKED_KV_ROW_BYTES,
        0,
        1,
        PACKED_KV_ROW_BYTES,
        safe_nonzero_pointer,  # descriptor_templates, never dereferenced
        QUEUE_COUNT,
    ]
    return {
        "resident_threshold": torch.zeros(
            row_count, dtype=torch.int32, device=device
        ),
        # A valid resident tau makes this a stronger no-post guard: the queue
        # saturation, rather than an invalid tau sentinel, prevents H2D.
        "resident_threshold_valid": torch.ones(
            row_count, dtype=torch.int32, device=device
        ),
        "kv_object_slots_snapshot": torch.full(
            (pool_rows, SOURCE_TOKENS), -1, dtype=torch.int32, device=device
        ),
        "req_pool_entries": req_pool_entries,
        "kv_object_map": kv_object_map,
        "private_source_to_slot": torch.full(
            (row_count, SOURCE_TOKENS), -1, dtype=torch.int32, device=device
        ),
        "private_ready_table": torch.zeros(
            (QUEUE_COUNT, PRIVATE_ARENA_CAPACITY), dtype=torch.int32, device=device
        ),
        "private_arena_head": torch.zeros(
            QUEUE_COUNT, dtype=torch.int32, device=device
        ),
        "private_packed_output": private_packed_output,
        "queue_control": torch.tensor(
            control_words, dtype=torch.int64, device=device
        ),
        "queue_leases": queue_leases,
        "kv_object_lens": kv_object_lens,
    }


def _snapshot_no_post_transaction(
    transaction: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Copy the small graph-visible mutable state before the derived call."""
    return {
        name: transaction[name].detach().cpu().contiguous().clone()
        for name in NO_POST_TRANSACTION_NAMES
    }


def _assert_no_post_transaction_unchanged(
    transaction: dict[str, torch.Tensor],
    snapshot: dict[str, torch.Tensor],
) -> None:
    changed = [
        name
        for name in NO_POST_TRANSACTION_NAMES
        if not torch.equal(transaction[name].detach().cpu().contiguous(), snapshot[name])
    ]
    if changed:
        raise RuntimeError(
            "no-post transaction mutated graph-visible state: "
            f"changed={','.join(changed)}"
        )


def _prove_safe_hook_control(
    outputs: tuple[torch.Tensor, ...],
    transaction: dict[str, torch.Tensor],
    snapshot: dict[str, torch.Tensor],
    control_mode: str,
    expected_hook_mode: str,
) -> None:
    """Print one compact proof after the target stream has completed."""
    _assert_no_post_transaction_unchanged(transaction, snapshot)
    telemetry = outputs[7].detach().cpu().contiguous()
    expected_shape = (QUEUE_COUNT, 13)
    if tuple(telemetry.shape) != expected_shape:
        raise RuntimeError(
            "unexpected P3 telemetry shape: "
            f"got={tuple(telemetry.shape)} expected={expected_shape}"
        )
    control_rejected = int(telemetry[:, 12].sum().item())
    doorbell_posted = int(telemetry[:, 8].sum().item())
    if doorbell_posted != 0:
        raise RuntimeError(
            "safe no-post control unexpectedly posted Direct-HCCP: "
            f"doorbell_posted={doorbell_posted}"
        )
    if expected_hook_mode == "off":
        if control_mode != HOOK_CONTROL_MODE_ALL_BUSY:
            raise RuntimeError(
                "the hook-off mathematical baseline requires "
                f"{HOOK_CONTROL_MODE_ENV}={HOOK_CONTROL_MODE_ALL_BUSY!r}; "
                f"got {control_mode!r}"
            )
        if control_rejected != 0:
            raise RuntimeError(
                "hook-off mathematical baseline unexpectedly wrote hook "
                f"telemetry: control_rejected={control_rejected}"
            )
        hook_state = "compiled_off_baseline"
    elif control_mode == HOOK_CONTROL_MODE_REJECT_BUSY:
        if control_rejected < 1:
            raise RuntimeError(
                "reject_all_busy did not observe hook control rejection: "
                f"control_rejected={control_rejected}"
            )
        hook_state = "control_rejected"
    else:
        if control_rejected != 0:
            raise RuntimeError(
                "valid_all_busy unexpectedly rejected the hook control: "
                f"control_rejected={control_rejected}"
            )
        hook_state = "enabled_all_busy"
    print(
        "P3_HOOK_CONTROL_PROOF "
        f"mode={control_mode} hook_state={hook_state} "
        f"control_rejected={control_rejected} doorbell_posted={doorbell_posted} "
        "transaction_unchanged=1 packed_output_prefill=0xa5 "
        "h2d_result=not_exercised",
        flush=True,
    )


def _make_static_transaction(
    device: torch.device, row_count: int
) -> dict[str, torch.Tensor]:
    """Allocate shape-valid sidecar state without an inline transport contract.

    The P3 score producer does not read or write these resources.  They remain
    appended graph ABI inputs for the later independent Get sidecar, so this
    test supplies stable allocations and checks their aliases, but never
    manufactures a queue descriptor or treats them as a no-post proof.
    """
    if row_count != BATCH_SIZE:
        raise RuntimeError(
            f"fixed C8 fixture requires row_count={BATCH_SIZE}, got {row_count}"
        )
    pool_rows = BATCH_SIZE
    return {
        "resident_threshold": torch.zeros(row_count, dtype=torch.int32, device=device),
        "resident_threshold_valid": torch.zeros(
            row_count, dtype=torch.int32, device=device
        ),
        "kv_object_slots_snapshot": torch.full(
            (pool_rows, SOURCE_TOKENS), -1, dtype=torch.int32, device=device
        ),
        "req_pool_entries": torch.arange(
            row_count, dtype=torch.int32, device=device
        ),
        "kv_object_map": torch.arange(
            SOURCE_TOKENS, dtype=torch.int32, device=device
        ).repeat(pool_rows, 1).contiguous(),
        "private_source_to_slot": torch.full(
            (row_count, SOURCE_TOKENS), -1, dtype=torch.int32, device=device
        ),
        "private_ready_table": torch.zeros(
            (QUEUE_COUNT, PRIVATE_ARENA_CAPACITY), dtype=torch.int32, device=device
        ),
        "private_arena_head": torch.zeros(QUEUE_COUNT, dtype=torch.int32, device=device),
        "private_packed_output": torch.full(
            (QUEUE_COUNT * PRIVATE_ARENA_CAPACITY, PACKED_KV_ROW_BYTES),
            0xA5,
            dtype=torch.uint8,
            device=device,
        ),
        "queue_control": torch.zeros(
            STREAMING_CONTROL_WORDS, dtype=torch.int64, device=device
        ),
        "queue_leases": torch.zeros(QUEUE_COUNT, dtype=torch.int32, device=device),
        "kv_object_lens": torch.tensor(
            (SOURCE_TOKENS, SOURCE_TOKENS - BLOCK_SIZE),
            dtype=torch.int32,
            device=device,
        ),
    }


def _score_workspace_words(source_tokens: int) -> int:
    aligned = ((source_tokens + TILE_SCORE_WIDTH - 1) // TILE_SCORE_WIDTH) * TILE_SCORE_WIDTH
    return (A5_MAX_AIC_CORE_COUNT * 4 * aligned + 1) // 2


def _make_publication_buffers(device: torch.device) -> dict[str, torch.Tensor]:
    score_words = _score_workspace_words(SOURCE_TOKENS)
    receipt_rows = BATCH_SIZE * ((SOURCE_TOKENS + TILE_SCORE_WIDTH - 1) // TILE_SCORE_WIDTH)
    return {
        "score_workspace": torch.full(
            (score_words,), SCORE_WORKSPACE_SENTINEL, dtype=torch.int32, device=device
        ),
        "tile_receipts": torch.full(
            (receipt_rows, TILE_RECEIPT_WORDS),
            TILE_RECEIPT_SENTINEL,
            dtype=torch.int64,
            device=device,
        ),
        "streaming_epoch": torch.tensor(
            (STREAMING_EPOCH,), dtype=torch.int64, device=device
        ),
    }


def _assert_publication_contract(
    mode: str, publication: dict[str, torch.Tensor]
) -> dict[str, int | str]:
    """Validate the score workspace and fixed-slot publication ABI after sync."""
    score = publication["score_workspace"].detach().cpu().contiguous()
    receipts = publication["tile_receipts"].detach().cpu().contiguous()
    epoch = int(publication["streaming_epoch"].detach().cpu().item())
    if epoch <= 0:
        raise RuntimeError(
            "streaming_epoch must be a nonzero forward generation; a zero or "
            "reused epoch can make a stale receipt appear ready"
        )
    score_changed = int(bool(torch.any(score != SCORE_WORKSPACE_SENTINEL).item()))
    ready = receipts[:, 8] == epoch
    ready_count = int(torch.count_nonzero(ready).item())
    untouched = receipts[~ready]

    if mode == "off":
        if score_changed or ready_count:
            raise RuntimeError(
                "publication=off changed graph-visible score/receipt buffers: "
                f"score_changed={score_changed} ready_count={ready_count}"
            )
        if not bool(torch.all(untouched == TILE_RECEIPT_SENTINEL).item()):
            raise RuntimeError("publication=off changed an unwritten receipt slot")
        return {
            "mode": mode,
            "score_workspace_changed": score_changed,
            "receipt_ready_count": ready_count,
            "receipt_expected_active": 0,
        }

    if not score_changed:
        raise RuntimeError(f"publication={mode} did not write score_workspace")
    if mode == "score_workspace":
        if ready_count:
            raise RuntimeError(
                "publication=score_workspace wrote a tile receipt: "
                f"ready_count={ready_count}"
            )
        if not bool(torch.all(untouched == TILE_RECEIPT_SENTINEL).item()):
            raise RuntimeError("publication=score_workspace changed an unwritten receipt")
        return {
            "mode": mode,
            "score_workspace_changed": score_changed,
            "receipt_ready_count": ready_count,
            "receipt_expected_active": 0,
        }

    if mode != "publisher":
        raise AssertionError(f"unexpected publication mode {mode!r}")
    source_lens = (SOURCE_TOKENS, SOURCE_TOKENS - BLOCK_SIZE)
    tiles_per_row = SOURCE_TOKENS // TILE_SCORE_WIDTH
    expected_active = sum(
        (source_len + TILE_SCORE_WIDTH - 1) // TILE_SCORE_WIDTH
        for source_len in source_lens
    )
    full_grid = BATCH_SIZE * tiles_per_row
    if ready_count not in (expected_active, full_grid):
        raise RuntimeError(
            "publisher receipt count is neither active-tile nor full-grid: "
            f"ready_count={ready_count} active={expected_active} "
            f"grid={full_grid}"
        )
    # Some official C8 schedules iterate the fixed max-length tile grid and
    # therefore publish a ready descriptor with ``valid=0`` for a shorter
    # row's terminal tile.  That is acceptable only as a complete, fixed-grid
    # convention: every score-bearing slot is still present exactly once, and
    # the zero-valid descriptor itself must carry the same ABI contract.  A
    # partial mixture would make the publisher impossible to audit safely.
    for row, source_len in enumerate(source_lens):
        for tile in range(tiles_per_row):
            slot = row * tiles_per_row + tile
            record = receipts[slot]
            expected_valid = max(
                0, min(TILE_SCORE_WIDTH, source_len - tile * TILE_SCORE_WIDTH)
            )
            is_ready = int(record[8].item()) == epoch
            if expected_valid > 0 and not is_ready:
                raise RuntimeError(
                    "publisher omitted a required fixed receipt slot: "
                    f"row={row} tile={tile} slot={slot}"
                )
            if expected_valid == 0 and not is_ready:
                if not bool(torch.all(record == TILE_RECEIPT_SENTINEL).item()):
                    raise RuntimeError(
                        "publisher changed an inactive receipt without the expected epoch: "
                        f"row={row} tile={tile} slot={slot} record={record.tolist()}"
                    )
                continue
            # A ready inactive slot is allowed only in the full-grid form.
            if expected_valid == 0 and ready_count != full_grid:
                raise RuntimeError(
                    "publisher mixed active-only and full-grid receipt styles: "
                    f"row={row} tile={tile} slot={slot} ready_count={ready_count}"
                )
            if (
                int(record[1].item()) != row
                or int(record[2].item()) != tile * TILE_SCORE_WIDTH
                or int(record[3].item()) != expected_valid
                or int(record[4].item()) != SOURCE_TOKENS
                or int(record[7].item()) != TILE_RECEIPT_CONTRACT_VERSION
            ):
                raise RuntimeError(
                    "publisher receipt descriptor mismatch: "
                    f"slot={slot} record={record.tolist()}"
                )
            byte_offset = int(record[0].item())
            if byte_offset < 0 or byte_offset % 2 or byte_offset >= score.numel() * 4:
                raise RuntimeError(
                    "publisher receipt score byte offset is out of score_workspace: "
                    f"slot={slot} offset={byte_offset} bytes={score.numel() * 4}"
                )
            if not bool(torch.all(record[9:] == 0).item()):
                raise RuntimeError(
                    f"publisher receipt has a nonzero ready-line tail at slot={slot}")
    return {
        "mode": mode,
        "score_workspace_changed": score_changed,
        "receipt_ready_count": ready_count,
        "receipt_expected_active": expected_active,
    }


def _expected_score_source_lens(device: torch.device) -> torch.Tensor:
    """Independent decode-TND reconstruction of official C8's valid S bound.

    Each request has exactly one query token and this adapter fixes
    ``cmp_ratio=1``.  The unmodified official causal formula is therefore
    ``(L - 1 + 0 + 1) / 1 == L`` per output row.  Constructing a fresh tensor
    from host scalars—not copying the target output—keeps the harness's
    score-source check independent.
    """
    return torch.tensor(
        [(SOURCE_TOKENS - 1 + 0 + 1) // 1,
         (SOURCE_TOKENS - BLOCK_SIZE - 1 + 0 + 1) // 1],
        dtype=torch.int32,
        device=device,
    )


def run_target_equivalence() -> dict[str, Any]:
    """Run one real official/derived C8 comparison for the harness."""
    reference_opapi, reference_vendor = _resolve_single_opp(
        os.getenv(REFERENCE_OPP_ENV), "reference"
    )
    target_opapi, target_vendor = _resolve_single_opp(
        os.getenv(TARGET_OPP_ENV), "target"
    )
    publication_mode = _resolve_publication_mode()
    inline_mode = _resolve_inline_mode()
    post_closure_mode = _resolve_post_closure_mode(inline_mode)
    is_empty_post_shell = (
        inline_mode == "post" and post_closure_mode == "empty_shell"
    )
    if inline_mode != "off" and not is_empty_post_shell:
        raise RuntimeError(
            "this strict P3 publication adapter requires P3_INLINE_MODE=off, "
            "or P3_INLINE_MODE=post with P3_POST_CLOSURE_MODE=empty_shell; "
            "use fused_streaming_quant_li_get_audit_adapter.py for inline=audit"
        )
    target_provenance = _read_target_provenance(
        target_vendor, publication_mode, inline_mode
    )
    # CANN discovers the binary config for an op on its first ACLNN call.  Both
    # OPP vendors must be visible before either the reference or P3 target
    # executes; switching this path after the reference is too late.
    _prepend_unique_path(
        "ASCEND_CUSTOM_OPP_PATH", (reference_vendor, target_vendor)
    )
    official_op, metadata_op, _ = _require_source_matched_official_c8_ops(
        reference_opapi, reference_vendor
    )
    print(
        "P3_TARGET_PROVENANCE "
        f"source_root={target_provenance['source_root']} "
        f"source_head={target_provenance['source_head']} "
        f"expected_head={os.getenv(TARGET_EXPECTED_HEAD_ENV, '<unset>')} "
        f"source_clean={target_provenance['source_clean']} "
        f"publication_mode={target_provenance['publication_mode']} "
        f"inline_mode={target_provenance['inline_mode']} "
        f"post_closure_mode={target_provenance['post_closure_mode']} "
        f"kvaf_closure={target_provenance['kvaf_closure']} "
        "inline_hccp=absent ready=1 "
        f"vendor={target_vendor} opapi={target_opapi}",
        flush=True,
    )
    device = _require_a5_npu()
    inputs = _build_official_inputs(device)
    inputs["metadata"] = _official_metadata(metadata_op, inputs)
    official_indices = _call_official(official_op, inputs)
    # Fix the normal reference result before the isolated bridge chooses its
    # own libcust_opapi DSO.  This is validation-only synchronization, not a
    # product execution dependency.
    torch.npu.synchronize()

    target_opapi = _bind_isolated_target_opp(target_opapi)
    importlib.import_module("nanovllm_streaming_quant_li_get")
    namespace = getattr(torch.ops, "a5_streaming_quant_li_get", None)
    derived_op = (
        getattr(namespace, "streaming_quant_li_get", None)
        if namespace is not None
        else None
    )
    if derived_op is None:
        raise RuntimeError(
            "isolated StreamingQuantLiGet bridge did not register "
            "torch.ops.a5_streaming_quant_li_get.streaming_quant_li_get"
        )

    transaction = _make_static_transaction(device, BATCH_SIZE)
    publication = _make_publication_buffers(device)
    static_sidecar_before = {
        name: transaction[name].detach().cpu().contiguous().clone()
        for name in NO_POST_TRANSACTION_NAMES
    }
    print(
        "P3_STREAMING_LI_GET_TARGET_ADAPTER "
        f"device={device} physical_npu={_SELECTED_PHYSICAL_NPU_ID} "
        f"batch={BATCH_SIZE} source_tokens={SOURCE_TOKENS} "
        f"heads={INDEX_HEADS} metadata_words={inputs['metadata'].numel()} "
        f"reference_opapi={reference_opapi} target_opapi={target_opapi} "
        f"publication_mode={publication_mode} "
        f"inline_mode={inline_mode} "
        f"post_closure_mode={post_closure_mode} "
        f"score_workspace_words={publication['score_workspace'].numel()} "
        f"tile_receipt_rows={publication['tile_receipts'].shape[0]} "
        "product_route=disabled h2d_result=not_exercised",
        flush=True,
    )
    derived_started = monotonic()
    print("P3_DERIVED_STAGE stage=call_begin", flush=True)
    outputs = derived_op(
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
        publication["score_workspace"],
        publication["tile_receipts"],
        publication["streaming_epoch"],
        kv_object_domain=0,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        cmp_ratio=1,
        return_values=False,
        stride=int(inputs["key"].stride(0)),
        scale_stride=int(inputs["key_dequant_scale"].stride(0)),
        kvaf_rows_per_entry=BLOCK_SIZE,
    )
    print(
        "P3_DERIVED_STAGE stage=call_return "
        f"elapsed_ms={(monotonic() - derived_started) * 1000.0:.3f}",
        flush=True,
    )
    if not isinstance(outputs, tuple) or len(outputs) != 11:
        raise RuntimeError(
            "derived A5StreamingQuantLiGet did not return its 11-output ABI; "
            "rebuild the isolated target bridge for 24 inputs / 11 outputs"
        )
    print("P3_DERIVED_STAGE stage=synchronize_begin", flush=True)
    torch.npu.synchronize()
    print(
        "P3_DERIVED_STAGE stage=synchronize_end "
        f"elapsed_ms={(monotonic() - derived_started) * 1000.0:.3f}",
        flush=True,
    )
    changed_static = [
        name
        for name, before in static_sidecar_before.items()
        if not torch.equal(transaction[name].detach().cpu().contiguous(), before)
    ]
    if changed_static:
        raise RuntimeError(
            "C8 score producer changed a static sidecar allocation: "
            f"changed={','.join(changed_static)}"
        )
    publication_summary = _assert_publication_contract(publication_mode, publication)
    print(
        "P3_PUBLICATION_PROOF "
        f"mode={publication_summary['mode']} "
        f"score_workspace_changed={publication_summary['score_workspace_changed']} "
        f"receipt_ready_count={publication_summary['receipt_ready_count']} "
        f"receipt_expected_active={publication_summary['receipt_expected_active']} "
        "post_kernel_receipt=verified inline_hccp=absent",
        flush=True,
    )

    return {
        "official_indices": official_indices,
        "derived_indices": outputs[0],
        "input_aliases": {
            name: (inputs[name], inputs[name])
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
            )
        },
        "transaction_aliases": {
            "private_source_to_slot": (
                transaction["private_source_to_slot"], outputs[2]
            ),
            "private_ready_table": (
                transaction["private_ready_table"], outputs[3]
            ),
            "private_arena_head": (
                transaction["private_arena_head"], outputs[4]
            ),
            "private_packed_output": (
                transaction["private_packed_output"], outputs[5]
            ),
            "queue_leases": (transaction["queue_leases"], outputs[6]),
        },
        "publication_aliases": {
            "score_workspace": (publication["score_workspace"], outputs[9]),
            "tile_receipts": (publication["tile_receipts"], outputs[10]),
        },
        # The P5 live-publisher harness consumes these *exact* C8 output
        # allocations.  Keeping them in addition to ``publication_aliases``
        # makes the zero-copy cross-op edge explicit instead of recovering
        # tensors by fragile output-position knowledge in a second module.
        # ``streaming_epoch`` is graph-static forward metadata, not a C8
        # terminal output; its identity is still part of the publication ABI.
        "publication_tensors": {
            "score_workspace": publication["score_workspace"],
            "tile_receipts": publication["tile_receipts"],
            "streaming_epoch": publication["streaming_epoch"],
        },
        "publication_summary": publication_summary,
        # A5 C8 rejects return_value=True in the reference tiling, so this
        # first real target proof deliberately compares only sparse indices.
        "official_values": None,
        "derived_values": None,
        "expected_score_source_lens": _expected_score_source_lens(device),
        "derived_score_source_lens": outputs[8],
    }


__all__ = ["run_target_equivalence"]
