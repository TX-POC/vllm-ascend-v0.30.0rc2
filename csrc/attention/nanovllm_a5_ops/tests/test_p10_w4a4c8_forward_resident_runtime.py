#!/usr/bin/env python3
"""P10 NPU acceptance: W4A4C8 math + Direct-HCCP resident closure.

This is a deliberately small *real* forward fixture, not a synthetic receipt
oracle.  It cold-binds local KVAF Direct-HCCP state, launches the one P10 graph
through its forward bridge, then waits only in this validation process after
the Stage-B -> QSFA event has been inserted.  The product graph itself has no
Python candidate loop, host Get submit, or physical-core assignment.

The fixture uses a single active W4A4C8 row and 4,096 source tokens.  This is
deliberately larger than Top-K: the run must perform a real ordered selection,
then prove that speculative candidates and final guaranteed recall leave every
one of the 2,048 true winners in a fixed resident row before QSFA proceeds.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from typing import Any


BATCH = 1
INDEX_HEADS = 64
KEY_HEADS = 1
HEAD_DIM = 128
BLOCK_TOKENS = 128
SOURCE_TOKENS = 4096
TOPK = 2048
PACKED_KV_ROW_BYTES = 656
ENTRY_BYTES = BLOCK_TOKENS * PACKED_KV_ROW_BYTES
QUEUE_COUNT = 1
RECEIPT_WORDS = 5
SPECULATIVE_ROWS = 64
RESIDENT_ROWS = SPECULATIVE_ROWS + BATCH * TOPK
ATTENTION_TAIL = 8
ATTENTION_WIDTH = TOPK + ATTENTION_TAIL
CONTROL_WORDS = 10
MAX_REQUESTS_PER_WAVE = 64


class P10Blocked(RuntimeError):
    """A missing target-runtime prerequisite, reported without a false pass."""


class P10SemanticSignature(tuple):
    """Rank-ordered P10 result semantics, independent of private slot IDs.

    Speculative admission can legitimately select a different physical
    resident row in the serial and overlap schedules.  This compact signature
    instead represents what QSFA can actually consume: exact canonical Top-K,
    source lengths, selected lengths, non-sparse tail, and rank-ordered KV
    bytes after logical-QSFA-slot translation.
    """

    __slots__ = ()

    def __new__(
        cls,
        canonical_sha16: str,
        source_lens: tuple[int, ...],
        selected_lens: tuple[int, ...],
        tail_sha16: str,
        ranked_payload_sha16: str,
    ) -> "P10SemanticSignature":
        return tuple.__new__(
            cls,
            (
                canonical_sha16,
                source_lens,
                selected_lens,
                tail_sha16,
                ranked_payload_sha16,
            ),
        )


class P10TimingSample(tuple):
    """One complete, already-validated P10 forward timing sample."""

    __slots__ = ()

    def __new__(
        cls,
        device_us: float,
        host_us: float,
        signature: P10SemanticSignature,
        speculative_slots: int,
        mapped_sources: int,
    ) -> "P10TimingSample":
        return tuple.__new__(
            cls,
            (device_us, host_us, signature, speculative_slots, mapped_sources),
        )

    @property
    def device_us(self) -> float:
        return float(self[0])

    @property
    def host_us(self) -> float:
        return float(self[1])

    @property
    def signature(self) -> P10SemanticSignature:
        return self[2]

    @property
    def speculative_slots(self) -> int:
        return int(self[3])

    @property
    def mapped_sources(self) -> int:
        return int(self[4])


def _blocked(reason: str) -> None:
    raise P10Blocked(reason)


def _sha16(value: Any) -> str:
    """Hash the tensor's storage bytes without converting its product dtype.

    NumPy has no ``Float8_e4m3fn`` scalar type, so ``tensor.numpy()`` rejects
    the exact W4A4C8 Q/K tensors used by this fixture.  A dtype *view* to
    uint8 is a one-to-one storage reinterpretation, not a quantization or
    numerical cast: it preserves the FP8 bits that the product operator sees.
    """

    import torch

    tensor = value.detach().cpu().contiguous().reshape(-1)
    raw_bytes = tensor.view(torch.uint8)
    return hashlib.sha256(raw_bytes.numpy().tobytes()).hexdigest()[:16]


def _require_file(environment: str) -> Path:
    path = Path(os.environ.get(environment, "")).expanduser()
    if not path.is_file() or path.stat().st_size == 0:
        _blocked(f"required_file_missing:{environment}:{path}")
    return path.resolve()


def _require_directory(environment: str) -> Path:
    path = Path(os.environ.get(environment, "")).expanduser()
    if not path.is_dir():
        _blocked(f"required_directory_missing:{environment}:{path}")
    return path.resolve()


def _require_logical_npu() -> Any:
    if os.environ.get("P10_REQUIRE_REAL_HCCP") != "1":
        _blocked("P10_REQUIRE_REAL_HCCP_must_be_1")
    if os.environ.get("P10_LOGICAL_NPU_ID", "0") != "0":
        _blocked("logical_npu_must_be_zero")
    physical = os.environ.get("P10_PHYSICAL_NPU_ID", "")
    if not physical or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != physical:
        _blocked(
            "physical_logical_mapping_invalid:"
            f"physical={physical or '<unset>'}:"
            f"visible={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<unset>')}"
        )
    if os.environ.get("SOC_VERSION"):
        _blocked("legacy_SOC_VERSION_must_be_unset")

    import torch
    import torch_npu  # noqa: F401  -- initializes the Ascend device provider.

    torch.npu.set_device(0)
    value = torch.ones(1, dtype=torch.float32, device="npu:0").cpu().item()
    print(
        "P10_RUNTIME_NPU "
        f"physical={physical} logical=0 device={torch.npu.get_device_name(0)!r} "
        f"value={value}",
        flush=True,
    )
    return torch.device("npu:0")


def _quantize_fp8(value: Any) -> tuple[Any, Any]:
    import torch
    import torch_npu

    quantized, scale = torch_npu.npu_dynamic_quant(
        value, dst_type=torch.float8_e4m3fn
    )
    return (
        quantized.contiguous(),
        scale.view(value.shape[:-1]).to(dtype=torch.float32).contiguous(),
    )


def _build_w4a4c8_inputs(device: Any) -> dict[str, Any]:
    """Allocate the exact W4A4C8 frontend ABI, with no FP32 weight widening."""

    import torch

    blocks = SOURCE_TOKENS // BLOCK_TOKENS
    torch.manual_seed(20260903)
    torch.npu.manual_seed_all(20260903)
    query_bf16 = torch.empty(
        (BATCH, INDEX_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-1.0, 1.0)
    key_bf16 = torch.empty(
        (blocks, BLOCK_TOKENS, KEY_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-1.0, 1.0)
    query, query_scale = _quantize_fp8(query_bf16)
    key, key_scale = _quantize_fp8(key_bf16)
    weights = torch.empty(
        (BATCH, INDEX_HEADS), dtype=torch.bfloat16, device=device
    ).uniform_(0.01, 1.0).contiguous()
    actual_query = torch.ones(BATCH, dtype=torch.int32, device=device)
    candidate_lens = torch.full(
        (BATCH,), SOURCE_TOKENS, dtype=torch.int32, device=device
    )
    # Ascending blocks keep the test's S -> O token witness directly visible.
    block_table = torch.arange(
        blocks, dtype=torch.int32, device=device
    ).reshape(BATCH, blocks)
    return {
        "query": query,
        "key": key,
        "weights": weights,
        "query_dequant_scale": query_scale,
        "key_dequant_scale": key_scale,
        "actual_seq_lengths_query": actual_query,
        "candidate_lens": candidate_lens,
        "block_table": block_table,
    }


def _reference_w4a4c8(inputs: dict[str, Any]) -> Any:
    """Call only the installed product W4A4C8 frontend as math oracle."""

    import torch
    import torch_npu

    op = getattr(torch_npu, "npu_quant_lightning_indexer", None)
    if not callable(op):
        _blocked("installed_product_w4a4c8_callable_missing")
    result = op(
        query=inputs["query"],
        key=inputs["key"],
        weights=inputs["weights"],
        query_dequant_scale=inputs["query_dequant_scale"],
        key_dequant_scale=inputs["key_dequant_scale"],
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_key=inputs["candidate_lens"],
        block_table=inputs["block_table"],
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    topk = result[0] if isinstance(result, tuple) else result
    if not isinstance(topk, torch.Tensor) or topk.dtype != torch.int32:
        raise RuntimeError("product W4A4C8 frontend did not return int32 Top-K")
    if topk.numel() != BATCH * TOPK:
        raise RuntimeError(
            "product W4A4C8 Top-K element count mismatch: "
            f"shape={tuple(topk.shape)} expected={BATCH * TOPK}"
        )
    topk = topk.reshape(BATCH, 1, TOPK).contiguous()
    print(
        "P10_RUNTIME_REFERENCE "
        f"product=torch_npu.npu_quant_lightning_indexer "
        f"sparse_sha16={_sha16(topk)} dtype={topk.dtype}",
        flush=True,
    )
    return topk


def _payload(entries: int) -> Any:
    import torch

    entry = torch.arange(entries, dtype=torch.int64).reshape(entries, 1)
    byte = torch.arange(ENTRY_BYTES, dtype=torch.int64).reshape(1, ENTRY_BYTES)
    return ((entry * 29 + byte * 7 + (entry >> 2) + 17) & 0xFF).to(torch.uint8)


class _ForwardResidentKvaf:
    """Cold-bound local Direct-HCCP resources for one P10 forward.

    The only data-plane submissions after :meth:`bind` occur inside Stage B.
    This class is intentionally a fixture boundary, not a host Get fallback.
    """

    def __init__(self, device: Any) -> None:
        try:
            self.kvaf = importlib.import_module("kvaf")
        except Exception as error:  # noqa: BLE001
            _blocked(f"kvaf_import_failed:{type(error).__name__}:{error}")
        expected_build = _require_directory("P10_KVAF_BUILD")
        module_path = Path(str(getattr(self.kvaf, "__file__", ""))).resolve()
        try:
            module_path.relative_to(expected_build / "python")
        except ValueError as error:
            raise RuntimeError(
                "kvaf import does not originate from the exact P10 build: "
                f"module={module_path} expected_root={expected_build / 'python'}"
            ) from error
        self.device = device
        self.entries = SOURCE_TOKENS // BLOCK_TOKENS
        self.runtime = self.kvaf.NpuRuntime(
            str(_require_file("P10_KVAF_BATCH_GET_KERNEL")),
            0,
            QUEUE_COUNT,
            str(_require_file("P10_KVAF_BATCH_PUT_KERNEL")),
            1,
        )
        required = (
            "initialize",
            "batch_put",
            "prepare_access_table",
            "access_table_view",
            "bind_layer_resident_batch_get_control",
            "configure_layer_resident_batch_get_queue_templates",
            "release_layer_resident_batch_get_control",
            "layer_resident_batch_get_control_diagnostic",
            "release_access_table",
            "shutdown",
        )
        missing = [name for name in required if not callable(getattr(self.runtime, name, None))]
        if missing:
            _blocked("kvaf_forward_resident_api_missing:" + ",".join(missing))
        if callable(getattr(self.runtime, "link_to", None)):
            _blocked("kvaf_fixture_requires_local_direct_hccp_not_remote_service")
        self.table: Any | None = None
        self.bound = False
        self.terminal = False
        self.data = _payload(self.entries)
        self.resident_payload: Any | None = None
        self.control: Any | None = None
        self.leases: Any | None = None

    def bind(self) -> tuple[Any, Any, Any]:
        import torch

        self.runtime.initialize(self.entries * ENTRY_BYTES, ENTRY_BYTES)
        source = self.data.to(self.device).reshape(-1).contiguous()
        keys = list(range(1, self.entries + 1))
        self.runtime.batch_put(
            keys,
            source,
            [index * ENTRY_BYTES for index in range(self.entries)],
            [ENTRY_BYTES] * self.entries,
        )
        torch.npu.synchronize()
        self.table = self.runtime.prepare_access_table(keys)
        access = self.runtime.access_table_view(
            self.table, device="npu", capacity=self.entries
        )
        torch.npu.synchronize()

        resident_payload = torch.zeros(
            (RESIDENT_ROWS, PACKED_KV_ROW_BYTES),
            dtype=torch.uint8,
            device=self.device,
        )
        control = torch.zeros(16, dtype=torch.int64, device=self.device)
        leases = torch.zeros(QUEUE_COUNT, dtype=torch.int32, device=self.device)
        descriptors = {
            "indices": [torch.zeros(MAX_REQUESTS_PER_WAVE, dtype=torch.int32, device=self.device)],
            "offsets": [torch.zeros(MAX_REQUESTS_PER_WAVE, dtype=torch.int32, device=self.device)],
            "masks": [torch.zeros(MAX_REQUESTS_PER_WAVE, dtype=torch.uint8, device=self.device)],
            "outputs": [torch.zeros(MAX_REQUESTS_PER_WAVE, dtype=torch.int32, device=self.device)],
        }
        # The process has exactly one logical visible device (physical NPU is
        # mapped before launch), so asking the active provider stream avoids a
        # torch_npu-version-specific ``device=`` overload.
        stream = torch.npu.current_stream()
        self.runtime.bind_layer_resident_batch_get_control(
            0,
            resident_payload,
            control,
            0,
            RESIDENT_ROWS,
            leases,
            QUEUE_COUNT,
            stream,
        )
        self.bound = True
        self.runtime.configure_layer_resident_batch_get_queue_templates(
            [0],
            [access.addresses],
            [access.valid],
            descriptors["indices"],
            descriptors["offsets"],
            descriptors["masks"],
            descriptors["outputs"],
            stream,
        )
        torch.npu.synchronize()
        words = [int(word) & ((1 << 64) - 1) for word in control.cpu().tolist()]
        if (
            len(words) < 15
            or words[0] == 0
            or words[5] != (resident_payload.data_ptr() & ((1 << 64) - 1))
            or words[6] != (leases.data_ptr() & ((1 << 64) - 1))
            or words[8] != QUEUE_COUNT
            or words[9] != RESIDENT_ROWS
            or words[10] != 0
            or words[11] != RESIDENT_ROWS
            or words[13] != PACKED_KV_ROW_BYTES
        ):
            raise RuntimeError(f"invalid local forward-resident KVAF control: {words}")
        self.resident_payload = resident_payload
        self.control = control
        self.leases = leases
        print(
            "P10_RUNTIME_HCCP_BIND "
            f"entries={self.entries} queues={QUEUE_COUNT} resident_rows={RESIDENT_ROWS} "
            f"control_magic={words[0]} diagnostic={self.runtime.layer_resident_batch_get_control_diagnostic()}",
            flush=True,
        )
        return resident_payload, control, leases

    def expected_row(self, source: int) -> Any:
        if source < 0 or source >= SOURCE_TOKENS:
            raise AssertionError(f"source outside KVAF direct table: {source}")
        return self.data[source // BLOCK_TOKENS].reshape(
            BLOCK_TOKENS, PACKED_KV_ROW_BYTES
        )[source % BLOCK_TOKENS]

    def close(self) -> None:
        """Release only after all P10 final slots are known terminal-ready."""

        if not self.terminal:
            print(
                "P10_RUNTIME_CLEANUP safe=0 action=process_exit_preserve_direct_hccp_diagnostics",
                flush=True,
            )
            return
        import torch

        try:
            torch.npu.synchronize()
            if self.bound:
                self.runtime.release_layer_resident_batch_get_control()
                self.bound = False
            if self.table is not None:
                self.runtime.release_access_table(self.table)
                self.table = None
            if getattr(self.runtime, "initialized", False):
                self.runtime.shutdown()
            print("P10_RUNTIME_CLEANUP safe=1 action=released_after_forward", flush=True)
        except Exception as error:  # noqa: BLE001
            print(
                "P10_RUNTIME_CLEANUP safe=0 action=process_exit_preserve_direct_hccp_diagnostics "
                f"type={type(error).__name__}",
                flush=True,
            )


def _build_forward_inputs(
    device: Any, base: dict[str, Any], fixture: _ForwardResidentKvaf
) -> tuple[Any, dict[str, Any]]:
    import torch
    from nanovllm_p10_forward_bridge.p10_forward_graph import P10ForwardInputs

    blocks = SOURCE_TOKENS // BLOCK_TOKENS
    score_slices = (256 + INDEX_HEADS - 1) // INDEX_HEADS
    receipt_capacity = BATCH * blocks
    canonical_sparse = torch.full(
        (BATCH, 1, TOPK), -1, dtype=torch.int32, device=device
    )
    canonical_lens = torch.zeros(BATCH, dtype=torch.int32, device=device)
    score_surface = torch.zeros(
        (BATCH, score_slices, SOURCE_TOKENS), dtype=torch.uint16, device=device
    )
    receipt_ring = torch.zeros(
        (receipt_capacity, RECEIPT_WORDS), dtype=torch.int64, device=device
    )
    receipt_state = torch.zeros(receipt_capacity, dtype=torch.int32, device=device)
    # Fields [2..4,6] are cold geometry/cutoff.  Word 1 is the Stage-B
    # diagnostic ledger, reset by prepare; a zero score cutoff gives Stage B
    # a real speculative workload while final recall still guarantees all
    # canonical winners.
    forward_control = torch.tensor(
        [0, 0, TOPK, BLOCK_TOKENS, BATCH, 0, 0, 0, 0, 0],
        dtype=torch.int32,
        device=device,
    )
    active_pool = torch.zeros(BATCH, dtype=torch.int32, device=device)
    object_map = torch.arange(SOURCE_TOKENS, dtype=torch.int32, device=device).reshape(1, -1)
    object_snapshot = torch.full(
        (1, SOURCE_TOKENS), -1, dtype=torch.int32, device=device
    )
    object_to_slot = torch.full(
        (1, SOURCE_TOKENS), -1, dtype=torch.int32, device=device
    )
    source_to_slot = torch.full(
        (BATCH, SOURCE_TOKENS), -1, dtype=torch.int32, device=device
    )
    speculative_band = torch.tensor(
        [0, SPECULATIVE_ROWS], dtype=torch.int32, device=device
    )
    speculative_cursor = torch.zeros(1, dtype=torch.int32, device=device)
    recall_band = torch.tensor(
        [SPECULATIVE_ROWS, BATCH * TOPK], dtype=torch.int32, device=device
    )
    resident_ready = torch.zeros(RESIDENT_ROWS, dtype=torch.int32, device=device)
    # A nonidentity bijection proves Stage B translates physical resident rows
    # into downstream QSFA slots instead of treating them as interchangeable.
    # 5 is coprime with RESIDENT_ROWS (= 2112), so this is a real nonidentity
    # permutation.  A multiplier of 3 would silently alias three resident
    # rows onto each logical QSFA slot and make the integration fixture weaker
    # than the production mapping contract it is meant to prove.
    resident_to_qsfa = (
        torch.arange(RESIDENT_ROWS, dtype=torch.int32, device=device) * 5
    ) % RESIDENT_ROWS
    attention_slots = torch.full(
        (BATCH, 1, ATTENTION_WIDTH), -77, dtype=torch.int32, device=device
    )
    selected_lengths = torch.full(
        (BATCH,), ATTENTION_WIDTH, dtype=torch.int32, device=device
    )
    if fixture.resident_payload is None or fixture.control is None or fixture.leases is None:
        raise AssertionError("KVAF fixture was not bound before P10 input construction")
    inputs = P10ForwardInputs(
        **base,
        canonical_sparse_indices=canonical_sparse,
        canonical_score_source_lens=canonical_lens,
        score_surface=score_surface,
        receipt_ring=receipt_ring,
        receipt_state=receipt_state,
        forward_control=forward_control,
        active_row_pool_entries=active_pool,
        kv_object_map=object_map,
        kv_object_slots_snapshot=object_snapshot,
        kv_object_to_forward_slot=object_to_slot,
        source_to_forward_slot=source_to_slot,
        speculative_slot_band=speculative_band,
        speculative_slot_cursor=speculative_cursor,
        recall_slot_band=recall_band,
        kvaf_forward_resident_control=fixture.control,
        queue_leases=fixture.leases,
        resident_payload=fixture.resident_payload,
        resident_ready=resident_ready,
        resident_row_to_qsfa_slot=resident_to_qsfa,
        attention_slots_buffer=attention_slots,
        selected_slot_lengths_buffer=selected_lengths,
        num_layers=1,
        layer_id=0,
        executor_worker_count=1,
        executor_worker_limit=2,
        receipt_poll_quantum=1,
        bounded_drain_polls=4096,
    )
    tensors = {
        "canonical_sparse": canonical_sparse,
        "canonical_lens": canonical_lens,
        "receipt_state": receipt_state,
        "forward_control": forward_control,
        "object_to_slot": object_to_slot,
        "source_to_slot": source_to_slot,
        "speculative_cursor": speculative_cursor,
        "resident_ready": resident_ready,
        "resident_to_qsfa": resident_to_qsfa,
        "attention_slots": attention_slots,
        "selected_lengths": selected_lengths,
    }
    return inputs, tensors


_STAGE_B_DIAGNOSTIC_NAMES = {
    0: "clear_or_not_entered",
    1: "running",
    -101: "init_common",
    -102: "init_geometry",
    -103: "init_kvaf_control",
    -104: "init_ub",
    -201: "receipt_terminal",
    -202: "finalize",
    -203: "drain",
}

_STAGE_B_FINALIZE_DETAIL_BASE = -202000
_STAGE_B_FINALIZE_DETAIL_STRIDE = 8
_STAGE_B_FINALIZE_DETAIL_NAMES = {
    1: "bad_selected_length",
    2: "active_mapping",
    3: "persistent_qsfa",
    4: "slot_claim",
    5: "winner_ready",
    6: "final_qsfa",
}


def _emit_terminal_diagnostic(
    fixture: _ForwardResidentKvaf,
    tensors: dict[str, Any],
    forward: list[int],
) -> None:
    """Emit one bounded post-event state snapshot when P10 cannot complete.

    This runs only in the NPU acceptance harness after Stage-B's completion
    event has been observed.  It does not add a product dependency, mutate an
    operator input, or inspect a second transport path.  The point is to make
    one failed target run distinguish an executor launch/Init rejection from a
    receipt, Post/CQ, or final-recall failure.
    """

    import torch

    receipt = tensors["receipt_state"].detach().cpu()
    canonical_sparse = tensors["canonical_sparse"].detach().cpu()
    source_slots = tensors["source_to_slot"].detach().cpu()
    object_slots = tensors["object_to_slot"].detach().cpu()
    ready = tensors["resident_ready"].detach().cpu()
    cursor = tensors["speculative_cursor"].detach().cpu()
    attention = tensors["attention_slots"].detach().cpu()
    leases = fixture.leases.detach().cpu() if fixture.leases is not None else None

    def count_eq(value: Any, expected: int) -> int:
        return int(torch.count_nonzero(value == expected).item())

    diagnostic = int(forward[1]) if len(forward) > 1 else 0
    prefix = [int(value) for value in attention.reshape(-1)[:8].tolist()]
    lease_values = [] if leases is None else [int(value) for value in leases.tolist()]
    detail = "none"
    if diagnostic <= _STAGE_B_FINALIZE_DETAIL_BASE - 1:
        encoded = _STAGE_B_FINALIZE_DETAIL_BASE - diagnostic
        ordinal, reason = divmod(encoded, _STAGE_B_FINALIZE_DETAIL_STRIDE)
        row, rank = divmod(ordinal, TOPK)
        if 0 <= row < canonical_sparse.size(0) and 0 <= rank < TOPK:
            source = int(canonical_sparse[row, 0, rank].item())
            source_slot = (
                int(source_slots[row, source].item())
                if 0 <= source < source_slots.size(1)
                else -999
            )
            object_slot = (
                int(object_slots[0, source].item())
                if 0 <= source < object_slots.size(1)
                else -999
            )
            resident_state = (
                int(ready[source_slot].item())
                if 0 <= source_slot < ready.numel()
                else -999
            )
            qsfa_value = (
                int(tensors["resident_to_qsfa"][source_slot].detach().cpu().item())
                if 0 <= source_slot < ready.numel()
                else -999
            )
            attention_value = int(attention[row, 0, rank].item())
            detail = (
                f"reason={_STAGE_B_FINALIZE_DETAIL_NAMES.get(reason, 'unknown')} "
                f"row={row} rank={rank} source={source} source_slot={source_slot} "
                f"object_slot={object_slot} resident_state={resident_state} "
                f"qsfa_value={qsfa_value} attention_value={attention_value}"
            )
        else:
            detail = f"reason={_STAGE_B_FINALIZE_DETAIL_NAMES.get(reason, 'unknown')} ordinal={ordinal} out_of_fixture_range=1"
    print(
        "P10_RUNTIME_TERMINAL_DIAG "
        f"control={forward} stage_b_diag={diagnostic} "
        f"stage_b_diag_name={_STAGE_B_DIAGNOSTIC_NAMES.get(diagnostic, 'unknown')} "
        f"receipt_free={count_eq(receipt, 0)} receipt_filling={count_eq(receipt, 1)} "
        f"receipt_ready={count_eq(receipt, 2)} receipt_claimed={count_eq(receipt, 3)} "
        f"receipt_done={count_eq(receipt, 4)} source_mapped={int(torch.count_nonzero(source_slots >= 0).item())} "
        f"object_mapped={int(torch.count_nonzero(object_slots >= 0).item())} "
        f"resident_empty={count_eq(ready, 0)} resident_in_flight={count_eq(ready, 1)} "
        f"resident_ready={count_eq(ready, 2)} speculative_cursor={int(cursor.reshape(-1)[0].item())} "
        f"queue_leases={lease_values} attention_prefix={prefix} finalize_detail={detail}",
        flush=True,
    )


def _assert_terminal(
    reference: Any,
    fixture: _ForwardResidentKvaf,
    tensors: dict[str, Any],
    launch: Any,
    *,
    aliases: Any | None = None,
    require_preseal_overlap: bool = True,
    emit_summary: bool = True,
) -> P10SemanticSignature:
    import torch

    # This synchronization is a validation-only observation after the bridge
    # has already inserted Stage-B COMPLETE -> QSFA. It is never a product
    # host dependency.
    if aliases is None:
        aliases = launch.bind_qsfa()
    launch.stage_b_complete.synchronize()
    torch.npu.synchronize()
    actual = tensors["canonical_sparse"].detach().cpu()
    expected = reference.detach().cpu()
    if not torch.equal(actual, expected):
        mismatch = int(torch.count_nonzero(actual != expected).item())
        raise AssertionError(
            "strict W4A4C8 canonical Top-K mismatch: "
            f"reference_sha16={_sha16(expected)} actual_sha16={_sha16(actual)} mismatch={mismatch}"
        )
    actual_lens = tensors["canonical_lens"].detach().cpu()
    if actual_lens.tolist() != [SOURCE_TOKENS]:
        raise AssertionError(f"canonical score-source lengths mismatch: {actual_lens.tolist()}")
    forward = tensors["forward_control"].detach().cpu().tolist()
    if int(forward[5]) != 1:
        _emit_terminal_diagnostic(fixture, tensors, forward)
        raise AssertionError(f"Stage B did not reach forward COMPLETE: control={forward}")
    states = tensors["receipt_state"].detach().cpu()
    if int(torch.count_nonzero(states != 4).item()) != 0:
        raise AssertionError(f"not every produced receipt reached DONE: states={states.tolist()}")
    leases = fixture.leases.detach().cpu()
    if int(torch.count_nonzero(leases).item()) != 0:
        raise AssertionError(f"Direct-HCCP queue lease leaked: {leases.tolist()}")

    source_to_slot = tensors["source_to_slot"].detach().cpu()
    object_to_slot = tensors["object_to_slot"].detach().cpu()
    ready = tensors["resident_ready"].detach().cpu()
    translation = tensors["resident_to_qsfa"].detach().cpu()
    if int(torch.unique(translation).numel()) != RESIDENT_ROWS or torch.equal(
        translation, torch.arange(RESIDENT_ROWS, dtype=torch.int32)
    ):
        raise AssertionError(
            "resident_row_to_qsfa_slot is not a nonidentity permutation"
        )
    attention = aliases.attention_slots.detach().cpu()
    lengths = aliases.resident_seq_lengths.detach().cpu()
    payload = fixture.resident_payload.detach().cpu()
    if lengths.tolist() != [ATTENTION_WIDTH]:
        raise AssertionError(f"QSFA selected length/tail contract broke: {lengths.tolist()}")
    if not torch.equal(attention[0, 0, TOPK:], torch.full((ATTENTION_TAIL,), -77, dtype=torch.int32)):
        raise AssertionError("Stage B modified caller-owned QSFA tail")

    mismatches: list[str] = []
    used_slots: set[int] = set()
    ranked_payload = hashlib.sha256()
    for rank, source_value in enumerate(expected.reshape(-1).tolist()):
        source = int(source_value)
        slot = int(source_to_slot[0, source])
        if slot < 0 or slot >= RESIDENT_ROWS:
            mismatches.append(f"slot_range:rank={rank}:source={source}:slot={slot}")
            break
        if int(object_to_slot[0, source]) != slot:
            mismatches.append(f"object_dedup:rank={rank}:source={source}:slot={slot}")
            break
        if int(ready[slot]) != 2:
            mismatches.append(f"not_ready:rank={rank}:source={source}:slot={slot}:state={int(ready[slot])}")
            break
        if int(attention[0, 0, rank]) != int(translation[slot]):
            mismatches.append(f"qsfa_translation:rank={rank}:slot={slot}")
            break
        if not torch.equal(payload[slot], fixture.expected_row(source)):
            # One bounded CPU-side diagnostic after the product graph has
            # completed.  It adds no device synchronization or work and
            # distinguishes a bad WQE offset, a stale source row, and an
            # inbound-DMA/cache-visibility failure in one rerun.
            actual_payload = payload[slot].contiguous()
            expected_payload = fixture.expected_row(source).contiguous()
            byte_diff = actual_payload != expected_payload
            first_byte = int(torch.nonzero(byte_diff, as_tuple=False)[0].item())
            candidate_match = {
                candidate: int(
                    torch.equal(actual_payload, fixture.expected_row(candidate))
                )
                for candidate in (0, 1, 2)
            }
            print(
                "P10_RUNTIME_PAYLOAD_DIAG "
                f"rank={rank} source={source} slot={slot} "
                f"actual_sha16={_sha16(actual_payload)} "
                f"expected_sha16={_sha16(expected_payload)} "
                f"mismatch_bytes={int(torch.count_nonzero(byte_diff).item())} "
                f"first_byte={first_byte} "
                f"actual_first16={actual_payload[:16].tolist()} "
                f"expected_first16={expected_payload[:16].tolist()} "
                f"match_source0={candidate_match[0]} "
                f"match_source1={candidate_match[1]} "
                f"match_source2={candidate_match[2]} "
                f"all_zero={int(torch.count_nonzero(actual_payload).item()) == 0}",
                flush=True,
            )
            mismatches.append(f"payload:rank={rank}:source={source}:slot={slot}")
            break
        used_slots.add(slot)
        ranked_payload.update(payload[slot].contiguous().numpy().tobytes())
    if mismatches:
        raise AssertionError("P10 forward-resident closure mismatch: " + ",".join(mismatches))
    if len(used_slots) != TOPK:
        raise AssertionError(f"final winners alias resident rows: unique={len(used_slots)} expected={TOPK}")
    witness = launch.read_overlap_witness_after_completion()
    if require_preseal_overlap:
        if not witness.overlapped:
            raise AssertionError(
                "P10 did not observe receipt consumption before producer seal: "
                f"producer_seal={witness.producer_seal_sequence} "
                f"first_preseal_claim={witness.first_preseal_ready_claim_sequence}"
            )
    elif (
        witness.producer_seal_sequence <= 0
        or witness.first_preseal_ready_claim_sequence != 0
    ):
        raise AssertionError(
            "P10 serial reference did not remain Stage-A-before-Stage-B: "
            f"producer_seal={witness.producer_seal_sequence} "
            f"first_preseal_claim={witness.first_preseal_ready_claim_sequence}"
        )
    signature = P10SemanticSignature(
        canonical_sha16=_sha16(actual),
        source_lens=tuple(int(value) for value in actual_lens.tolist()),
        selected_lens=tuple(int(value) for value in lengths.tolist()),
        tail_sha16=_sha16(attention[0, 0, TOPK:]),
        ranked_payload_sha16=ranked_payload.hexdigest()[:16],
    )
    fixture.terminal = True
    if emit_summary:
        print(
            "P10_RUNTIME_PARITY "
            f"canonical_topk=1 score_source_lens=1 sha16={_sha16(actual)}",
            flush=True,
        )
        print(
            "P10_RUNTIME_HCCP "
            f"final_winners={TOPK} resident_ready=1 payload=1 queue_leases_zero=1 unique_slots={len(used_slots)}",
            flush=True,
        )
        print(
            "P10_RUNTIME_QSFA "
            f"stage_b_to_qsfa_event=1 nonidentity_resident_translation=1 tail_preserved=1 "
            f"preseal_overlap={int(witness.overlapped)} producer_seal={witness.producer_seal_sequence} "
            f"first_preseal_claim={witness.first_preseal_ready_claim_sequence}",
            flush=True,
        )
    return signature


def _verify_runtime_provenance() -> None:
    path = _require_file("P10_PRODUCT_RUNTIME_PROVENANCE")
    values = dict(
        line.split("=", 1)
        for line in path.read_text().splitlines()
        if "=" in line
    )
    required = {
        "kind": "p10_w4a4c8_forward_resident_runtime",
        "vllm_source_head": os.environ.get("P10_VLLM_EXPECTED_SHA", ""),
        "kvaf_source_head": os.environ.get("P10_KVAF_EXPECTED_SHA", ""),
        "physical_npu": os.environ.get("P10_PHYSICAL_NPU_ID", ""),
        "logical_npu": "0",
    }
    bad = [key for key, expected in required.items() if not expected or values.get(key) != expected]
    if bad:
        _blocked("runtime_provenance_mismatch:" + ",".join(bad))
    if values.get("product_opapi") != os.environ.get("NANOVLLM_A5_P10_PRODUCT_OPAPI_LIB"):
        _blocked("runtime_provenance_product_opapi_mismatch")
    if values.get("product_vendor") != os.environ.get("NANOVLLM_A5_P10_PRODUCT_OPP_VENDOR"):
        _blocked("runtime_provenance_product_vendor_mismatch")
    print(
        "P10_RUNTIME_PROVENANCE "
        f"file={path} vllm={values['vllm_source_head']} kvaf={values['kvaf_source_head']} "
        f"opapi_sha256={values.get('product_opapi_sha256', '<missing>')}",
        flush=True,
    )


def _load_product_runtime() -> tuple[Any, Any]:
    """Resolve the one source-keyed product bridge before any P10 work.

    Both the normal acceptance case and the paired performance case pass
    through this exact function.  That keeps the comparison honest: there is
    no second bridge, no separately resolved Stage A, and no host-side Get
    baseline hidden behind the benchmark label.
    """

    _verify_runtime_provenance()
    device = _require_logical_npu()
    bridge_root = _require_directory("P10_PRODUCT_BRIDGE_PYTHON")
    opapi = _require_file("NANOVLLM_A5_P10_PRODUCT_OPAPI_LIB")
    vendor = _require_directory("NANOVLLM_A5_P10_PRODUCT_OPP_VENDOR")
    if os.environ.get("ASCEND_CUSTOM_OPP_PATH") != str(vendor):
        _blocked("combined_vendor_is_not_the_only_custom_opp_path")
    if opapi.parent.parent.parent != vendor:
        _blocked("opapi_vendor_relationship_invalid")

    module = importlib.import_module("nanovllm_p10_forward_bridge")
    module_file_text = getattr(module, "__file__", None)
    if not module_file_text:
        _blocked("product_bridge_package_file_missing")
    module_file = Path(module_file_text).resolve()
    try:
        module_file.relative_to(bridge_root)
    except ValueError as error:
        raise RuntimeError(
            f"P10 bridge imported from an unexpected path: module={module_file} expected={bridge_root}"
        ) from error
    from nanovllm_p10_forward_bridge.p10_forward_graph import launch_p10_forward

    return device, launch_p10_forward


def _positive_environment_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise P10Blocked(f"invalid_{name.lower()}:{raw}") from error
    if value <= 0:
        _blocked(f"invalid_{name.lower()}:{raw}")
    return value


def _immutable_input_digests(base: dict[str, Any]) -> dict[str, str]:
    """Return a bounded identity check for the product's read-only W4A4C8 ABI."""

    return {
        name: _sha16(base[name])
        for name in (
            "query",
            "key",
            "weights",
            "query_dequant_scale",
            "key_dequant_scale",
            "actual_seq_lengths_query",
            "candidate_lens",
            "block_table",
        )
    }


def _require_immutable_inputs(
    before: dict[str, str], base: dict[str, Any], *, context: str
) -> None:
    after = _immutable_input_digests(base)
    changed = [name for name, digest in before.items() if after.get(name) != digest]
    if changed:
        raise AssertionError(
            "W4A4C8 product input mutated during "
            f"{context}:" + ",".join(changed)
        )


def _percentile(samples: list[float], percentile: float) -> float:
    """Linear percentile with deterministic behavior for the short A5 run."""

    if not samples:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(samples)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    fraction = index - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _emit_perf_result(operation: str, values: list[float], *, warmup: int, metric: str) -> None:
    print(
        "PERF_RESULT "
        f"operation={operation} samples={len(values)} warmup_excluded={warmup} "
        f"metric={metric} mean_us={statistics.mean(values):.3f} "
        f"p50_us={_percentile(values, 50.0):.3f} "
        f"p90_us={_percentile(values, 90.0):.3f} "
        f"p99_us={_percentile(values, 99.0):.3f}",
        flush=True,
    )


def _reset_benchmark_destination(fixture: _ForwardResidentKvaf, producer_stream: Any) -> None:
    """Equalize the only persistent writable payload outside either timing interval.

    The Direct-HCCP control/table stay cold-bound throughout the paired run;
    that is intentional product setup.  The resident payload itself is
    cleared only after the preceding forward reached terminal state and its
    queue lease was proven zero.  It prevents an old row from being mistaken
    for a newly transferred winner if a future regression skips a copy.
    """

    import torch

    if fixture.resident_payload is None:
        raise AssertionError("P10 benchmark destination is not cold-bound")
    with torch.npu.stream(producer_stream):
        fixture.resident_payload.zero_()
    torch.npu.synchronize()


def _run_timed_forward(
    *,
    mode: str,
    device: Any,
    base: dict[str, Any],
    reference: Any,
    fixture: _ForwardResidentKvaf,
    launch_p10_forward: Any,
    immutable_digests: dict[str, str],
) -> P10TimingSample:
    """Run one fully terminal P10 forward with an exact device-side boundary.

    The compared interval begins on the producer stream immediately before
    ``ForwardPrepare`` and ends on that same stream after it has waited for
    Stage B's completion event.  It therefore includes native W4A4C8 score,
    receipt consumption, PostOnly/CQ drain, final recall, and the precise
    Stage-B -> QSFA readiness edge.  Allocation, cold binding, destination
    reset, host validation, and teardown are deliberately outside the timed
    interval for both modes.
    """

    import torch

    if mode not in ("serial", "overlap"):
        raise ValueError(f"unsupported P10 benchmark mode: {mode}")

    # The runner exposes one logical device only.  Use the active provider
    # stream rather than a version-specific ``device=`` overload, matching
    # the cold KVAF bind path above.
    producer = torch.npu.current_stream()
    executor = torch.npu.Stream(device=device)
    _reset_benchmark_destination(fixture, producer)
    fixture.terminal = False
    inputs, tensors = _build_forward_inputs(device, base, fixture)

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    host_start_ns = time.perf_counter_ns()
    with torch.npu.stream(producer):
        start.record()
    launch = launch_p10_forward(
        inputs,
        producer_stream=producer,
        executor_stream=executor,
        execution_mode=mode,
    )
    aliases = launch.bind_qsfa(producer)
    with torch.npu.stream(producer):
        end.record()
    end.synchronize()
    host_us = (time.perf_counter_ns() - host_start_ns) / 1000.0
    device_us = start.elapsed_time(end) * 1000.0

    signature = _assert_terminal(
        reference,
        fixture,
        tensors,
        launch,
        aliases=aliases,
        require_preseal_overlap=(mode == "overlap"),
        emit_summary=False,
    )
    _require_immutable_inputs(immutable_digests, base, context=f"P10 {mode} run")
    speculative_slots = int(
        tensors["speculative_cursor"].detach().cpu().reshape(-1)[0].item()
    )
    mapped_sources = int(
        torch.count_nonzero(tensors["source_to_slot"].detach().cpu() >= 0).item()
    )
    return P10TimingSample(
        device_us=device_us,
        host_us=host_us,
        signature=signature,
        speculative_slots=speculative_slots,
        mapped_sources=mapped_sources,
    )


def _run_benchmark() -> int:
    """Compare serial and overlapped P10 using the identical product route.

    This is intentionally a B=1, 4,096-token acceptance geometry: it proves
    that the first production-shaped forward has an exact serial reference
    before a later capacity/saturation sweep changes batch geometry.  It is
    not a claim that the numbers characterize a fully saturated decode batch.
    """

    warmup = _positive_environment_int("P10_PERF_WARMUP", 1)
    iterations = _positive_environment_int("P10_PERF_ITERS", 5)
    device, launch_p10_forward = _load_product_runtime()
    base = _build_w4a4c8_inputs(device)
    immutable_digests = _immutable_input_digests(base)
    reference = _reference_w4a4c8(base)
    _require_immutable_inputs(immutable_digests, base, context="product oracle")
    fixture = _ForwardResidentKvaf(device)
    samples: dict[str, list[P10TimingSample]] = {"serial": [], "overlap": []}

    try:
        fixture.bind()
        # Warm both modes once, then alternate order in measured trials.  The
        # alternation prevents Direct-HCCP and cache warm state from always
        # favouring whichever mode happens to run second.
        for iteration in range(warmup):
            for mode in (("serial", "overlap") if iteration % 2 == 0 else ("overlap", "serial")):
                _run_timed_forward(
                    mode=mode,
                    device=device,
                    base=base,
                    reference=reference,
                    fixture=fixture,
                    launch_p10_forward=launch_p10_forward,
                    immutable_digests=immutable_digests,
                )
        print(
            "P10_PERF_WARMUP "
            f"pairs={warmup} serial_terminal=1 overlap_preseal=1",
            flush=True,
        )

        for iteration in range(iterations):
            order = ("serial", "overlap") if iteration % 2 == 0 else ("overlap", "serial")
            for mode in order:
                sample = _run_timed_forward(
                    mode=mode,
                    device=device,
                    base=base,
                    reference=reference,
                    fixture=fixture,
                    launch_p10_forward=launch_p10_forward,
                    immutable_digests=immutable_digests,
                )
                samples[mode].append(sample)
                print(
                    "P10_PERF_SAMPLE "
                    f"iteration={iteration} mode={mode} device_us={sample.device_us:.3f} "
                    f"host_us={sample.host_us:.3f} speculative_slots={sample.speculative_slots} "
                    f"mapped_sources={sample.mapped_sources}",
                    flush=True,
                )
    finally:
        fixture.close()

    serial_signatures = [sample.signature for sample in samples["serial"]]
    overlap_signatures = [sample.signature for sample in samples["overlap"]]
    if not serial_signatures or not overlap_signatures:
        raise AssertionError("P10 benchmark produced no paired timing samples")
    expected_signature = serial_signatures[0]
    if any(signature != expected_signature for signature in serial_signatures):
        raise AssertionError("P10 serial benchmark semantics varied between samples")
    if any(signature != expected_signature for signature in overlap_signatures):
        raise AssertionError("P10 serial/overlap semantic signature mismatch")

    serial_device = [sample.device_us for sample in samples["serial"]]
    overlap_device = [sample.device_us for sample in samples["overlap"]]
    serial_host = [sample.host_us for sample in samples["serial"]]
    overlap_host = [sample.host_us for sample in samples["overlap"]]
    _emit_perf_result(
        "p10_w4a4c8_forward_resident_serial",
        serial_device,
        warmup=warmup,
        metric="npu_event_forward_prepare_to_qsfa_ready",
    )
    _emit_perf_result(
        "p10_w4a4c8_forward_resident_overlap",
        overlap_device,
        warmup=warmup,
        metric="npu_event_forward_prepare_to_qsfa_ready",
    )
    _emit_perf_result(
        "p10_w4a4c8_forward_resident_serial_host_diagnostic",
        serial_host,
        warmup=warmup,
        metric="host_submit_to_terminal_diagnostic",
    )
    _emit_perf_result(
        "p10_w4a4c8_forward_resident_overlap_host_diagnostic",
        overlap_host,
        warmup=warmup,
        metric="host_submit_to_terminal_diagnostic",
    )

    serial_p50 = _percentile(serial_device, 50.0)
    overlap_p50 = _percentile(overlap_device, 50.0)
    serial_p90 = _percentile(serial_device, 90.0)
    overlap_p90 = _percentile(overlap_device, 90.0)
    saved_p50 = serial_p50 - overlap_p50
    saved_p90 = serial_p90 - overlap_p90
    print(
        "P10_PERF_EQ "
        "product_w4a4c8_oracle=1 canonical_topk=1 score_source_lens=1 "
        "ranked_resident_payload=1 qsfa_lengths_and_tail=1 "
        "serial_terminal_after_seal=1 overlap_preseal_claim=1",
        flush=True,
    )
    print(
        "P10_PERF_RESULT "
        f"serial_p50_us={serial_p50:.3f} overlap_p50_us={overlap_p50:.3f} "
        f"saved_p50_us={saved_p50:.3f} "
        f"saved_p50_pct={100.0 * saved_p50 / serial_p50:.3f} "
        f"serial_p90_us={serial_p90:.3f} overlap_p90_us={overlap_p90:.3f} "
        f"saved_p90_us={saved_p90:.3f} "
        f"saved_p90_pct={100.0 * saved_p90 / serial_p90:.3f}",
        flush=True,
    )
    for mode in ("serial", "overlap"):
        speculative = [sample.speculative_slots for sample in samples[mode]]
        mapped = [sample.mapped_sources for sample in samples[mode]]
        print(
            "P10_PERF_WORK "
            f"mode={mode} speculative_slots_min={min(speculative)} "
            f"speculative_slots_max={max(speculative)} mapped_sources_min={min(mapped)} "
            f"mapped_sources_max={max(mapped)}",
            flush=True,
        )
    print(
        "P10_RUNTIME_PASS "
        "route=w4a4c8_score_receipt_direct_hccp_qsfa benchmark=serial_vs_overlap",
        flush=True,
    )
    return 0


def _run_acceptance() -> int:
    device, launch_p10_forward = _load_product_runtime()
    base = _build_w4a4c8_inputs(device)
    reference = _reference_w4a4c8(base)
    fixture = _ForwardResidentKvaf(device)
    try:
        fixture.bind()
        inputs, tensors = _build_forward_inputs(device, base, fixture)
        launch = launch_p10_forward(inputs)
        _assert_terminal(reference, fixture, tensors, launch)
    finally:
        fixture.close()
    print("P10_RUNTIME_PASS route=w4a4c8_score_receipt_direct_hccp_qsfa", flush=True)
    return 0


def _main() -> int:
    runtime_case = os.environ.get("P10_RUNTIME_CASE", "acceptance")
    if runtime_case == "acceptance":
        return _run_acceptance()
    if runtime_case == "benchmark":
        return _run_benchmark()
    _blocked(f"invalid_p10_runtime_case:{runtime_case}")
    raise AssertionError("unreachable")


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except P10Blocked as error:
        print(f"P10_RUNTIME_BLOCKED reason={error}", flush=True)
        raise SystemExit(2)
    except Exception as error:  # noqa: BLE001 -- retain concise precise failure marker.
        print(f"P10_RUNTIME_FAIL type={type(error).__name__} text={error}", flush=True)
        traceback.print_exc(limit=24)
        raise
