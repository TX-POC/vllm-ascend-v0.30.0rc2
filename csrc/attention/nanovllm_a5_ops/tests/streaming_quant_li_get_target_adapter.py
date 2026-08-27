#!/usr/bin/env python3
"""Target-only adapter for the A5StreamingQuantLiGet equivalence harness.

This module deliberately does not register, monkey-patch, or select any
vLLM product route.  It performs one controlled target-NPU comparison:

1. run the real A5 ``VllmQuantLightningIndexerMetadata`` companion and the
   real A5 ``VllmQuantLightningIndexer`` reference;
2. switch the *isolated bridge resolver only* to the separately built
   ``A5StreamingQuantLiGet`` OPP;
3. run the derived 21-input / 9-output operator with the very same nine C8
   input allocations; and
4. return the strict dictionary consumed by
   ``test_streaming_quant_li_get_target_equivalence.py``.

The default transaction control deliberately marks its one queue busy.  That
gives the device hook a structurally valid, graph-shaped transaction ABI while
preventing it from dereferencing an invented KVAF transport descriptor or
posting a fabricated H2D request. ``P3_HOOK_CONTROL_MODE=reject_all_busy``
uses the same safe busy lease table but deliberately invalidates the control
magic; this proves that hook initialization reached its reject path without
ever allowing a fabricated H2D post. Therefore this adapter proves target C8
score/Top-K equivalence, output-alias wiring, exact score-source lengths, and
the selected safe hook-control path; it does *not* claim Direct-HCCP
correctness or H2D performance.

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
STREAMING_CONTROL_MAGIC = 0x4B5641465354524D  # "KVAFSTRM"
STREAMING_CONTROL_VERSION = 1
INT64_MAX = (1 << 63) - 1
REFERENCE_OPP_ENV = "NANOVLLM_A5_REFERENCE_C8_OPP"
TARGET_OPP_ENV = "NANOVLLM_A5_STREAMING_LI_OPP"
REFERENCE_EXPECTED_HEAD_ENV = "P3_REFERENCE_EXPECTED_SOURCE_HEAD"
TARGET_EXPECTED_HEAD_ENV = "P3_TARGET_EXPECTED_SOURCE_HEAD"
REFERENCE_PROVENANCE_NAME = "P3_REFERENCE_PROVENANCE"
TARGET_PROVENANCE_NAME = "P3_STREAMING_LI_PROVENANCE"
REFERENCE_OPAPI_ENV = "NANOVLLM_A5_REFERENCE_C8_OPAPI_LIB"
TARGET_NPU_ENV = "P3_NPU_ID"
HOOK_CONTROL_MODE_ENV = "P3_HOOK_CONTROL_MODE"
HOOK_CONTROL_MODE_ALL_BUSY = "valid_all_busy"
HOOK_CONTROL_MODE_REJECT_BUSY = "reject_all_busy"
# This is intentionally opt-in and diagnostic-only.  Normal P3 validation
# must load a hook-on OPP because the product path needs score-time Get
# observation.  The one hook-off artifact is a binary-isolation tool: it tells
# us whether merely compiling the hook changes official C8 math, even when no
# queue is ever leased and no H2D can be posted.
ALLOW_HOOK_OFF_BASELINE_ENV = "P3_ALLOW_HOOK_OFF_BASELINE"
NO_POST_TRANSACTION_NAMES = (
    "private_source_to_slot",
    "private_ready_table",
    "private_arena_head",
    "private_packed_output",
    "queue_leases",
)


def _require_a5_npu() -> torch.device:
    requested_text = os.getenv(TARGET_NPU_ENV)
    if requested_text is None:
        raise RuntimeError(
            f"set {TARGET_NPU_ENV} to the explicit physical A5 NPU id; "
            "this target test must not silently use logical npu:0"
        )
    try:
        requested_id = int(requested_text)
    except ValueError as error:
        raise RuntimeError(
            f"{TARGET_NPU_ENV} must be a non-negative integer, "
            f"got {requested_text!r}"
        ) from error
    if requested_id < 0:
        raise RuntimeError(
            f"{TARGET_NPU_ENV} must be a non-negative integer, "
            f"got {requested_id}"
        )

    # A single-card ASCEND_VISIBLE_DEVICES mapping makes physical card 7 look
    # like logical npu:0.  That is useful for product isolation but unsuitable
    # for this bring-up: it obscures which card actually executed the C8 proof.
    # Fail closed instead of printing an ambiguous logical id in the evidence.
    visible_mapping = os.getenv("ASCEND_VISIBLE_DEVICES")
    if visible_mapping:
        raise RuntimeError(
            "P3 target equivalence requires an unmasked physical device; "
            "unset ASCEND_VISIBLE_DEVICES and use "
            f"{TARGET_NPU_ENV}={requested_id}, got "
            f"ASCEND_VISIBLE_DEVICES={visible_mapping!r}"
        )

    device_count = int(torch.npu.device_count())
    if requested_id >= device_count:
        raise RuntimeError(
            f"{TARGET_NPU_ENV}={requested_id} is outside the visible NPU "
            f"range [0,{device_count})"
        )
    torch.npu.set_device(requested_id)
    current_id = int(torch.npu.current_device())
    if current_id != requested_id:
        raise RuntimeError(
            "failed to select requested physical NPU: "
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
    print(
        "P3_TARGET_NPU "
        f"requested_physical={requested_id} current_device={current_id} "
        f"device={device} visible_mapping=<unset> device_name={name!r}",
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


def _read_target_provenance(
    vendor: Path, expected_hook_mode: str
) -> dict[str, str]:
    """Reject a stale P3 OPP or an unexpected hook compilation mode."""
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
        "kind",
        "source_root",
        "source_head",
        "source_clean",
        "hook_mode",
        "operators",
    )
    missing = tuple(key for key in required if not values.get(key))
    if missing:
        raise RuntimeError(
            "isolated P3 target provenance is incomplete: "
            f"path={path} missing={missing}"
        )
    if values["kind"] != "source_derived_streaming_quant_li_get":
        raise RuntimeError(
            "isolated P3 target provenance has an unexpected kind: "
            f"got={values['kind']!r}"
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
    if values["hook_mode"] != expected_hook_mode:
        raise RuntimeError(
            "isolated P3 target OPP has an unexpected hook compilation mode: "
            f"actual={values['hook_mode']!r} expected={expected_hook_mode!r}; "
            f"set {ALLOW_HOOK_OFF_BASELINE_ENV}=1 only for the diagnostic "
            "hook-off mathematical baseline"
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
    expected_hook_mode, hook_off_baseline = _expected_target_hook_mode()
    target_provenance = _read_target_provenance(
        target_vendor, expected_hook_mode
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
        f"hook_mode={target_provenance['hook_mode']} "
        f"baseline_only={int(hook_off_baseline)} ready=1 "
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

    row_count = BATCH_SIZE
    control_mode = _resolve_hook_control_mode()
    transaction = _make_no_post_transaction(device, row_count, control_mode)
    transaction_snapshot = _snapshot_no_post_transaction(transaction)
    print(
        "P3_STREAMING_LI_GET_TARGET_ADAPTER "
        f"device={device} physical_npu={device.index} "
        f"batch={BATCH_SIZE} source_tokens={SOURCE_TOKENS} "
        f"heads={INDEX_HEADS} metadata_words={inputs['metadata'].numel()} "
        f"reference_opapi={reference_opapi} target_opapi={target_opapi} "
        f"queue_mode={control_mode} "
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
    if not isinstance(outputs, tuple) or len(outputs) != 9:
        raise RuntimeError(
            "derived A5StreamingQuantLiGet did not return its 9-output ABI"
        )
    print("P3_DERIVED_STAGE stage=synchronize_begin", flush=True)
    torch.npu.synchronize()
    print(
        "P3_DERIVED_STAGE stage=synchronize_end "
        f"elapsed_ms={(monotonic() - derived_started) * 1000.0:.3f}",
        flush=True,
    )
    _prove_safe_hook_control(
        outputs,
        transaction,
        transaction_snapshot,
        control_mode,
        expected_hook_mode,
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
        # A5 C8 rejects return_value=True in the reference tiling, so this
        # first real target proof deliberately compares only sparse indices.
        "official_values": None,
        "derived_values": None,
        "expected_score_source_lens": _expected_score_source_lens(device),
        "derived_score_source_lens": outputs[8],
    }


__all__ = ["run_target_equivalence"]
