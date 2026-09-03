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
import os
from pathlib import Path
import sys
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


def _blocked(reason: str) -> None:
    raise P10Blocked(reason)


def _sha16(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()[:16]


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
        f"queue_leases={lease_values} attention_prefix={prefix}",
        flush=True,
    )


def _assert_terminal(
    reference: Any,
    fixture: _ForwardResidentKvaf,
    tensors: dict[str, Any],
    launch: Any,
) -> None:
    import torch

    # This synchronization is a validation-only observation after the bridge
    # has already inserted Stage-B COMPLETE -> QSFA. It is never a product
    # host dependency.
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
            mismatches.append(f"payload:rank={rank}:source={source}:slot={slot}")
            break
        used_slots.add(slot)
    if mismatches:
        raise AssertionError("P10 forward-resident closure mismatch: " + ",".join(mismatches))
    if len(used_slots) != TOPK:
        raise AssertionError(f"final winners alias resident rows: unique={len(used_slots)} expected={TOPK}")
    witness = launch.read_overlap_witness_after_completion()
    if not witness.overlapped:
        raise AssertionError(
            "P10 did not observe receipt consumption before producer seal: "
            f"producer_seal={witness.producer_seal_sequence} "
            f"first_preseal_claim={witness.first_preseal_ready_claim_sequence}"
        )
    fixture.terminal = True
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
        f"preseal_overlap=1 producer_seal={witness.producer_seal_sequence} "
        f"first_preseal_claim={witness.first_preseal_ready_claim_sequence}",
        flush=True,
    )


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


def _main() -> int:
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
