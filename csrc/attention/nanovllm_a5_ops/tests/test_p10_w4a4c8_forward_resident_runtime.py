#!/usr/bin/env python3
"""P10 NPU acceptance: W4A4C8 math + Direct-HCCP resident closure.

This is a deliberately small *real* forward fixture, not a synthetic receipt
oracle.  It cold-binds local KVAF Direct-HCCP state, launches the one P10 graph
through its forward bridge, then waits only in this validation process after
the Stage-B -> QSFA event has been inserted.  The product graph itself has no
Python candidate loop, host Get submit, or physical-core assignment.

The default fixture uses a single active W4A4C8 row and 4,096 source tokens.
``P10_BENCH_PROFILE=b16_s128k_miss5`` selects the capacity profile used for
the graph-compatible latency comparison: 16 rows, 128 Ki source tokens per
row, and a deterministic five-percent KV-residency miss mask.  It is not a
different algorithm or a host-Get baseline; the same native W4A4C8 inputs,
Direct-HCCP Stage-B operator, final recall, and QSFA boundary are used for
both serial and overlapping schedules.
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


P10_BENCH_PROFILE = os.environ.get("P10_BENCH_PROFILE", "acceptance")

# The capacity profile deliberately uses an all-source miss mask rather than
# marking a hand-picked subset of final winners as misses.  Thus Stage B sees
# the same pre-existing residency decision while it scans receipts as it sees
# during exact Top-K recall.  1/20 is 5% up to the unavoidable final partial
# group of a 128 Ki row.
if P10_BENCH_PROFILE == "acceptance":
    BATCH = 1
    SOURCE_TOKENS = 4096
    KV_MISS_DENOMINATOR = 1
    KV_MISS_NUMERATOR = 1
elif P10_BENCH_PROFILE == "b16_s128k_miss5":
    BATCH = 16
    SOURCE_TOKENS = 131072
    KV_MISS_DENOMINATOR = 20
    KV_MISS_NUMERATOR = 1
else:
    raise RuntimeError(
        "unsupported P10_BENCH_PROFILE:"
        f"{P10_BENCH_PROFILE}; expected acceptance or b16_s128k_miss5"
    )

INDEX_HEADS = 64
KEY_HEADS = 1
HEAD_DIM = 128
BLOCK_TOKENS = 128
TOPK = 2048
PACKED_KV_ROW_BYTES = 656
ENTRY_BYTES = BLOCK_TOKENS * PACKED_KV_ROW_BYTES
QUEUE_COUNT = int(os.environ.get("P10_BENCH_QUEUE_COUNT", "1"))
RECEIPT_WORDS = 5
SPECULATIVE_ROWS = 64
RECALL_ROWS = BATCH * TOPK
PERSISTENT_BASE = SPECULATIVE_ROWS + RECALL_ROWS
# The acceptance profile has no preloaded hits.  The B16 profile reserves a
# distinct persistent band for the actual final hit payloads so that hits can
# never collide with Direct-HCCP speculative/recall rows.
PERSISTENT_ROWS = RECALL_ROWS if KV_MISS_NUMERATOR < KV_MISS_DENOMINATOR else 0
RESIDENT_ROWS = PERSISTENT_BASE + PERSISTENT_ROWS
ATTENTION_TAIL = 8
ATTENTION_WIDTH = TOPK + ATTENTION_TAIL
CONTROL_WORDS = 10
MAX_REQUESTS_PER_WAVE = 64
EXECUTOR_WORKERS = int(
    os.environ.get("P10_BENCH_EXECUTOR_WORKERS", str(min(BATCH, 8)))
)
EXECUTOR_WORKER_LIMIT = int(os.environ.get("P10_BENCH_EXECUTOR_WORKER_LIMIT", "16"))
STAGE_B_RECEIPT_TRACE = os.environ.get("P10_STAGE_B_RECEIPT_TRACE", "0") == "1"
# ``forward_control`` is a signed int32 tensor.  C++ reads this exact word as
# uint32_t, so use the signed representation of bit 31 rather than trying to
# construct an out-of-range Python ``0x80000000`` int32 literal.
_STAGE_B_RECEIPT_TRACE_ENABLE = -(1 << 31)
_STAGE_B_RECEIPT_TRACE_BITS = {
    "entered": 1 << 0,
    "descriptor_valid": 1 << 1,
    "descriptor_invalid": 1 << 2,
    "seed_buffer_unavailable": 1 << 3,
    "source_scanned": 1 << 4,
    "score_eligible": 1 << 5,
    "mapping_valid": 1 << 6,
    "persistent_hit": 1 << 7,
    "miss_candidate": 1 << 8,
    "claim_attempted": 1 << 9,
    "claimed_new": 1 << 10,
    "post_attempted": 1 << 11,
    "posted": 1 << 12,
    "post_deferred": 1 << 13,
    "cq_ready": 1 << 14,
    "preseal_cq_ready": 1 << 15,
    "post_preseal": 1 << 16,
}


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
        speculative_admitted_sources: int,
        speculative_final_winner_hits: int,
        speculative_false_positive_sources: int,
        speculative_cursor_attempts: int,
        final_recall_sources: int,
        mapped_sources: int,
    ) -> "P10TimingSample":
        return tuple.__new__(
            cls,
            (
                device_us,
                host_us,
                signature,
                speculative_admitted_sources,
                speculative_final_winner_hits,
                speculative_false_positive_sources,
                speculative_cursor_attempts,
                final_recall_sources,
                mapped_sources,
            ),
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
        """Compatibility alias for actual admitted speculative sources.

        The old metric exposed the raw cursor, which counts failed capacity
        attempts too. Keep the property name for older result parsers, but
        make it report the only quantity that can represent early admission.
        """

        return int(self[3])

    @property
    def speculative_admitted_sources(self) -> int:
        return int(self[3])

    @property
    def speculative_final_winner_hits(self) -> int:
        return int(self[4])

    @property
    def speculative_false_positive_sources(self) -> int:
        return int(self[5])

    @property
    def speculative_cursor_attempts(self) -> int:
        return int(self[6])

    @property
    def final_recall_sources(self) -> int:
        return int(self[7])

    @property
    def mapped_sources(self) -> int:
        return int(self[8])


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
    # Every active row owns a distinct physical block range.  Reusing one
    # range for every B row would make the Direct-HCCP S->O map look
    # multi-row while W4A4C8 still reads a single-row physical cache.
    physical_blocks = BATCH * blocks
    torch.manual_seed(20260903)
    torch.npu.manual_seed_all(20260903)
    query_bf16 = torch.empty(
        (BATCH, INDEX_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    ).uniform_(-1.0, 1.0)
    key_bf16 = torch.empty(
        (physical_blocks, BLOCK_TOKENS, KEY_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-1.0, 1.0)
    query, query_scale = _quantize_fp8(query_bf16)
    key, key_scale = _quantize_fp8(key_bf16)
    weights = torch.empty(
        (BATCH, INDEX_HEADS), dtype=torch.bfloat16, device=device
    ).uniform_(0.01, 1.0).contiguous()
    # ``layout_query=TND`` consumes cumulative query lengths, not one
    # per-row length.  B=1 accidentally makes both encodings identical, but
    # a B16 tensor with sixteen ``1`` values is non-monotonic and leaves the
    # product W4A4C8 frontend free to produce invalid source indices.  The
    # fixture has exactly one query token per request, hence [1, 2, ..., B].
    actual_query = torch.arange(
        1, BATCH + 1, dtype=torch.int32, device=device
    )
    candidate_lens = torch.full(
        (BATCH,), SOURCE_TOKENS, dtype=torch.int32, device=device
    )
    # ``row * blocks + block`` matches Stage-B's pool-major KVAF object map.
    block_table = torch.arange(
        physical_blocks, dtype=torch.int32, device=device
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
    # This is a fail-closed product-ABI gate, deliberately before Direct-HCCP
    # setup.  A corrupt W4A4C8 reference must never be allowed to turn into a
    # later fixture gather fault or an invalid descriptor.  Do not clamp or
    # otherwise repair indices here: P10 mathematical equivalence requires
    # the product frontend itself to return valid source coordinates.
    topk_cpu = topk.detach().cpu()
    invalid = (topk_cpu < 0) | (topk_cpu >= SOURCE_TOKENS)
    invalid_count = int(torch.count_nonzero(invalid).item())
    if invalid_count:
        first = torch.nonzero(invalid, as_tuple=False)[0].tolist()
        first_value = int(topk_cpu[tuple(first)].item())
        raise RuntimeError(
            "product W4A4C8 Top-K source index outside the logical row: "
            f"invalid={invalid_count} first_coord={first} first_value={first_value} "
            f"allowed=[0,{SOURCE_TOKENS}) min={int(topk_cpu.min().item())} "
            f"max={int(topk_cpu.max().item())}"
        )
    print(
        "P10_RUNTIME_REFERENCE "
        f"product=torch_npu.npu_quant_lightning_indexer "
        f"sparse_sha16={_sha16(topk)} dtype={topk.dtype} "
        f"source_range=[{int(topk_cpu.min().item())},{int(topk_cpu.max().item())}]",
        flush=True,
    )
    return topk


def _payload(entries: int) -> Any:
    """Build the local-DDR witness without an enormous int64 broadcast temp.

    B16/S128K has 16,384 packed KV blocks (about 1.28 GiB).  The original
    expression first formed an eleven-GiB int64 tensor.  This preserves the
    exact byte formula while bounding the largest temporary to about 22 MiB.
    """

    import torch

    payload = torch.empty((entries, ENTRY_BYTES), dtype=torch.uint8)
    byte = torch.arange(ENTRY_BYTES, dtype=torch.int32).reshape(1, ENTRY_BYTES)
    for start in range(0, entries, 64):
        stop = min(entries, start + 64)
        entry = torch.arange(start, stop, dtype=torch.int32).reshape(-1, 1)
        values = (entry * 29 + byte * 7 + (entry >> 2) + 17) & 0xFF
        payload[start:stop].copy_(values.to(torch.uint8))
    return payload


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
        self.blocks = SOURCE_TOKENS // BLOCK_TOKENS
        self.entries = BATCH * self.blocks
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
            "batch_get",
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
        self.residency_stats: dict[str, int] = {}

    def bind(self) -> tuple[Any, Any, Any]:
        import torch

        self.runtime.initialize(self.entries * ENTRY_BYTES, ENTRY_BYTES)
        source = self.data.to(self.device).reshape(-1).contiguous()
        # ``source`` is the local-HBM producer for the cold Direct-HCCP Put
        # bootstrap.  The transfer is intentionally outside every P10 timing
        # interval, so make the producer/consumer boundary explicit instead
        # of relying on an implicit interaction between torch's current
        # stream and the eager KVAF stream resolver.
        torch.npu.synchronize()
        print(
            "P10_BOOTSTRAP_SOURCE_READY "
            f"bytes={source.numel()} producer=cpu_to_npu current_stream_drained=1 "
            "timed_interval=excluded",
            flush=True,
        )
        keys = list(range(1, self.entries + 1))
        offsets = [index * ENTRY_BYTES for index in range(self.entries)]
        lengths = [ENTRY_BYTES] * self.entries

        # BatchPut is a cold fixture bootstrap, before either serial or
        # overlap timing starts.  The eager AIV BatchPut ABI has one terminal
        # CQ wait per invocation.  Submitting all B16/S128K blocks in a single
        # 1.281-GiB wave makes that terminal wait exceed its bounded device
        # polling window even though every descriptor is valid.  Keep every
        # source byte, key, offset and destination identical, but bound each
        # pre-timing transport wave to the known single-wave fixture capacity.
        # This is deliberately not a host-side Get fallback and is outside
        # the measured P10 forward interval.
        bootstrap_rows = _positive_environment_int(
            "P10_BOOTSTRAP_PUT_ROWS", MAX_REQUESTS_PER_WAVE
        )
        if bootstrap_rows > MAX_REQUESTS_PER_WAVE:
            _blocked(
                "invalid_p10_bootstrap_put_rows:"
                f"{bootstrap_rows}; max={MAX_REQUESTS_PER_WAVE}"
            )
        bootstrap_waves = math.ceil(self.entries / bootstrap_rows)
        print(
            "P10_BOOTSTRAP_PUT_CONFIG "
            f"entries={self.entries} entry_bytes={ENTRY_BYTES} "
            f"rows_per_wave={bootstrap_rows} waves={bootstrap_waves} "
            f"timed_interval=excluded",
            flush=True,
        )
        for start in range(0, self.entries, bootstrap_rows):
            stop = min(start + bootstrap_rows, self.entries)
            wave = start // bootstrap_rows + 1
            try:
                self.runtime.batch_put(
                    keys[start:stop],
                    source,
                    offsets[start:stop],
                    lengths[start:stop],
                )
            except Exception as error:  # noqa: BLE001
                print(
                    "P10_BOOTSTRAP_PUT_FAIL "
                    f"wave={wave}/{bootstrap_waves} entry_begin={start} "
                    f"entry_end={stop} bytes={(stop - start) * ENTRY_BYTES} "
                    f"type={type(error).__name__} text={error}",
                    flush=True,
                )
                raise
        print(
            "P10_BOOTSTRAP_PUT_PASS "
            f"waves={bootstrap_waves} entries={self.entries} "
            f"bytes={self.entries * ENTRY_BYTES}",
            flush=True,
        )
        torch.npu.synchronize()
        self.table = self.runtime.prepare_access_table(keys)
        access = self.runtime.access_table_view(
            self.table, device="npu", capacity=self.entries
        )
        torch.npu.synchronize()
        # This is a cold data-plane gate, not a host Get fallback.  It
        # validates the exact KVAF table, source DDR objects, Direct-HCCP Get
        # kernel and 656-B packed-row ABI before Layer-resident binding
        # forbids ordinary eager Gets.  Therefore an all-zero Stage-B result
        # can be classified immediately as either a Put/DDR baseline fault
        # or a Stage-B-only descriptor/executor fault; no benchmark result is
        # accepted on an unproven source payload.
        self._prove_bootstrap_direct_hccp_payload(access, torch.npu.current_stream())

        # This tensor is an inbound Direct-HCCP destination.  Its transient
        # bands must never receive a torch producer write after registration:
        # a dirty DCache line owned by a different producer AIV can write back
        # after KVAF's executor-side pre-doorbell clean and overwrite a CQ-
        # successful DMA result.  ForwardPrepare clears reachability, not
        # payload bytes; final misses overwrite their assigned row before it
        # can be published READY.  Persistent hits are populated separately
        # in their disjoint, aligned band below.
        resident_payload = torch.empty(
            (RESIDENT_ROWS, PACKED_KV_ROW_BYTES),
            dtype=torch.uint8,
            device=self.device,
        )
        control = torch.zeros(16, dtype=torch.int64, device=self.device)
        leases = torch.zeros(QUEUE_COUNT, dtype=torch.int32, device=self.device)
        # These are queue-local Direct-HCCP descriptor *outputs* of Stage B,
        # not host protocol inputs.  On each active wave KVAF first evicts its
        # local descriptor-cache range, then writes every entry below the
        # published request count with bypass stores and performs the DDR
        # publish barrier before doorbell.  A torch zero-fill here would add a
        # second AIV writer whose delayed dirty-line writeback could corrupt a
        # valid descriptor after that publish.  Leave only the consumed range
        # to the device executor; control and leases above intentionally stay
        # zero-initialized because they *are* protocol inputs.
        descriptors = {
            "indices": [torch.empty(MAX_REQUESTS_PER_WAVE, dtype=torch.int32, device=self.device)],
            "offsets": [torch.empty(MAX_REQUESTS_PER_WAVE, dtype=torch.int32, device=self.device)],
            "masks": [torch.empty(MAX_REQUESTS_PER_WAVE, dtype=torch.uint8, device=self.device)],
            "outputs": [torch.empty(MAX_REQUESTS_PER_WAVE, dtype=torch.int32, device=self.device)],
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
            f"entries={self.entries} blocks_per_row={self.blocks} queues={QUEUE_COUNT} "
            f"resident_rows={RESIDENT_ROWS} source_gib={self.entries * ENTRY_BYTES / (1 << 30):.3f} "
            f"control_magic={words[0]} diagnostic={self.runtime.layer_resident_batch_get_control_diagnostic()}",
            flush=True,
        )
        return resident_payload, control, leases

    def _prove_bootstrap_direct_hccp_payload(self, access: Any, stream: Any) -> None:
        """Fail closed unless cold Direct-HCCP covers targeted table seams.

        This is not a host-Get fallback: it is the same KVAF Direct-HCCP
        source-table/offset/output ABI that Stage B will use after the
        layer-resident control is bound.  Keep this cold gate before the
        product graph so a missing table entry or a bad 128-row boundary is
        reported as a transport bootstrap fault rather than being misread as
        a graph or C8 math failure.  In particular source 380 is the first
        observed B16 failing final-recall source (table 2, offset 124 * 656).
        """

        import torch

        # Deduplicate the small-profile clamps while covering: first row,
        # both sides of two 128-row table seams, the observed source-380
        # failure, and the final table entry.  The entire vector remains far
        # below the fixed 64-request KVAF direct-Get capacity.
        probe_sources = tuple(
            dict.fromkeys(
                min(source, SOURCE_TOKENS - 1)
                for source in (0, 127, 128, 240, 255, 256, 380, SOURCE_TOKENS - 1)
            )
        )
        probe_indices = torch.tensor(
            [source // BLOCK_TOKENS for source in probe_sources],
            dtype=torch.int32,
            device=self.device,
        )
        probe_offsets = torch.tensor(
            [(source % BLOCK_TOKENS) * PACKED_KV_ROW_BYTES for source in probe_sources],
            dtype=torch.int32,
            device=self.device,
        )
        probe_mask = torch.ones(len(probe_sources), dtype=torch.uint8, device=self.device)
        probe_output_indices = torch.arange(
            len(probe_sources), dtype=torch.int32, device=self.device
        )
        # Like the product resident payload, this is strictly an inbound DMA
        # destination.  Initializing it with a torch kernel would let the
        # cold source-plane classifier create the same cross-AIV dirty-cache
        # overwrite it is supposed to diagnose.  Every checked row is fully
        # written by this Direct-HCCP Get before it is observed.
        probe_output = torch.empty(
            (len(probe_sources), PACKED_KV_ROW_BYTES),
            dtype=torch.uint8,
            device=self.device,
        )
        try:
            self.runtime.batch_get(
                access,
                probe_indices,
                probe_offsets,
                probe_mask,
                probe_output_indices,
                probe_output,
                stream=stream,
                synchronize=True,
            )
            torch.npu.synchronize()
        except Exception as error:  # noqa: BLE001
            print(
                "P10_BOOTSTRAP_ROUNDTRIP_FAIL "
                f"stage=submit type={type(error).__name__} text={error}",
                flush=True,
            )
            raise

        for output_row, source in enumerate(probe_sources):
            actual = probe_output[output_row].detach().cpu().contiguous()
            expected = self.expected_row(0, source).detach().cpu().contiguous()
            if not torch.equal(actual, expected):
                mismatch = torch.nonzero(actual != expected, as_tuple=False)
                first_byte = int(mismatch[0].item()) if mismatch.numel() else -1
                print(
                    "P10_BOOTSTRAP_ROUNDTRIP_FAIL "
                    f"stage=payload row=0 source={source} "
                    f"table_index={source // BLOCK_TOKENS} "
                    f"byte_offset={(source % BLOCK_TOKENS) * PACKED_KV_ROW_BYTES} "
                    f"actual_sha16={_sha16(actual)} expected_sha16={_sha16(expected)} "
                    f"mismatch_bytes={int(torch.count_nonzero(actual != expected).item())} "
                    f"first_byte={first_byte} "
                    f"actual_first16={actual[:16].tolist()} "
                    f"expected_first16={expected[:16].tolist()} "
                    f"all_zero={bool(torch.count_nonzero(actual).item() == 0)}",
                    flush=True,
                )
                raise AssertionError(
                    "P10 cold KVAF Direct-HCCP roundtrip mismatch: "
                    f"source={source}"
                )
        print(
            "P10_BOOTSTRAP_ROUNDTRIP_PASS "
            f"probes={len(probe_sources)} sources={','.join(map(str, probe_sources))} "
            f"bytes={probe_output.numel()} timed_interval=excluded",
            flush=True,
        )

    def expected_row(self, row: int, source: int) -> Any:
        if row < 0 or row >= BATCH:
            raise AssertionError(f"row outside KVAF direct table: {row}")
        if source < 0 or source >= SOURCE_TOKENS:
            raise AssertionError(f"source outside KVAF direct table: {source}")
        entry = row * self.blocks + source // BLOCK_TOKENS
        return self.data[entry].reshape(
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
    # Fields [2..4,6] are cold geometry/cutoff. Word 1 is the Stage-B
    # diagnostic ledger, reset by prepare. The low 16 bits of word 6 select
    # only speculative admission; Guaranteed Recall still materializes every
    # canonical winner, so a cutoff sweep cannot change P10 mathematics.
    score_cutoff_u16 = _score_cutoff_u16()
    score_control = score_cutoff_u16 | (
        _STAGE_B_RECEIPT_TRACE_ENABLE if STAGE_B_RECEIPT_TRACE else 0
    )
    forward_control = torch.tensor(
        [
            0,
            0,
            TOPK,
            BLOCK_TOKENS,
            BATCH,
            0,
            score_control,
            0,
            0,
            0,
        ],
        dtype=torch.int32,
        device=device,
    )
    # One pool entry per logical row.  Stage B checks this exact pool-major
    # object identity before it can form a Direct-HCCP descriptor.
    active_pool = torch.arange(BATCH, dtype=torch.int32, device=device)
    object_map = torch.arange(
        BATCH * SOURCE_TOKENS, dtype=torch.int32, device=device
    ).reshape(BATCH, SOURCE_TOKENS)

    # This is a real all-source residency mask, not a post-Top-K miss list.
    # A nonnegative snapshot value means that source is already resident, so
    # receipt scanning skips HCCP for it.  Final hit sources receive unique
    # slots below after the native W4A4C8 oracle is known; non-final hits use
    # slot zero only as a compact fixture witness and are never read by QSFA.
    # This is fixed fixture metadata, not P10 data-plane work.  Keep its
    # deterministic integer expression on CPU and upload the resulting
    # immutable mask once.  The A5 vector implementation of this particular
    # large int32 broadcast/remainder expression has raised a device 334
    # before any P10 operator runs; moving the pre-timing construction to CPU
    # avoids that unrelated provider kernel without changing a single mask
    # bit, source-object identity, or timed operation.
    source_ids = torch.arange(SOURCE_TOKENS, dtype=torch.int32)
    row_ids = torch.arange(BATCH, dtype=torch.int32).reshape(-1, 1)
    if KV_MISS_NUMERATOR == KV_MISS_DENOMINATOR:
        miss_mask_cpu = torch.ones((BATCH, SOURCE_TOKENS), dtype=torch.bool)
    else:
        miss_mask_cpu = (
            (source_ids.reshape(1, -1) + row_ids * 7) % KV_MISS_DENOMINATOR
        ) < KV_MISS_NUMERATOR
    object_snapshot_cpu = torch.zeros((BATCH, SOURCE_TOKENS), dtype=torch.int32)
    object_snapshot_cpu.masked_fill_(miss_mask_cpu, -1)
    miss_mask = miss_mask_cpu.to(device=device).contiguous()
    object_snapshot = object_snapshot_cpu.to(device=device).contiguous()
    torch.npu.synchronize()
    print(
        "P10_RESIDENCY_MASK_READY "
        f"construction=cpu upload=npu misses={int(miss_mask_cpu.sum().item())} "
        f"total={BATCH * SOURCE_TOKENS}",
        flush=True,
    )
    # Object ownership is pool-major, just like ``object_snapshot``.  This
    # was accidentally B=1-shaped in the original fixture; Stage B indexes it
    # as ``pool * sourceCapacity + source``, so a B16 case would otherwise
    # read/write beyond the alias for every row after row zero.
    object_to_slot = torch.full(
        (BATCH, SOURCE_TOKENS), -1, dtype=torch.int32, device=device
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
    # It is fixture geometry, not part of the P10 data plane.  In particular,
    # do not create this int32 multiply/remainder as an ordinary NPU vector
    # kernel after Direct-HCCP has been bound: on the affected A5 runtime it
    # can surface the unrelated AIVEC 334 fault before Stage A has run.  The
    # CPU formula is exactly the same int32 permutation, and the uploaded NPU
    # tensor remains the alias consumed by Stage B and QSFA.
    # 7 remains coprime for both the small and B16 resident-row capacities.
    resident_to_qsfa_cpu = (
        torch.arange(RESIDENT_ROWS, dtype=torch.int32) * 7
    ) % RESIDENT_ROWS
    resident_to_qsfa = resident_to_qsfa_cpu.to(device=device).contiguous()
    torch.npu.synchronize()
    print(
        "P10_FIXTURE_STATIC_METADATA_READY "
        f"construction=cpu upload=npu resident_rows={RESIDENT_ROWS}",
        flush=True,
    )
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
        executor_worker_count=EXECUTOR_WORKERS,
        executor_worker_limit=EXECUTOR_WORKER_LIMIT,
        receipt_poll_quantum=1,
        bounded_drain_polls=4096,
    )
    tensors = {
        "canonical_sparse": canonical_sparse,
        "canonical_lens": canonical_lens,
        "receipt_ring": receipt_ring,
        "receipt_state": receipt_state,
        "forward_control": forward_control,
        "object_snapshot": object_snapshot,
        "object_snapshot_cpu": object_snapshot_cpu,
        "object_to_slot": object_to_slot,
        "source_to_slot": source_to_slot,
        "miss_mask": miss_mask,
        "miss_mask_cpu": miss_mask_cpu,
        "speculative_cursor": speculative_cursor,
        "resident_ready": resident_ready,
        "resident_to_qsfa": resident_to_qsfa,
        "attention_slots": attention_slots,
        "selected_lengths": selected_lengths,
    }
    return inputs, tensors


def _install_final_hit_payloads(
    reference: Any, fixture: _ForwardResidentKvaf, tensors: dict[str, Any]
) -> None:
    """Materialize only already-resident final KV rows for the test fixture.

    The resident/miss decision itself was made for *every* source token in
    :func:`_build_forward_inputs`, before P10 starts.  The native W4A4C8 oracle
    is used here only to give the final resident hits distinct payload rows so
    that QSFA can validate their bytes.  It does not alter the miss mask,
    candidate admission, score surface, or canonical Top-K output.
    """

    import torch

    # This is untimed fixture setup, so use the CPU-resident immutable mask
    # and derive the final-hit mapping there.  Apart from avoiding a second
    # ordinary AIVEC gather/count kernel after Direct-HCCP initialization,
    # this makes the single upload explicit: ``snapshot`` remains the same
    # NPU tensor alias passed to Stage B.
    snapshot = tensors["object_snapshot"]
    snapshot_cpu = tensors["object_snapshot_cpu"].clone()
    miss_mask_cpu = tensors["miss_mask_cpu"]
    if fixture.resident_payload is None:
        raise AssertionError("resident payload is not cold-bound")

    ranked_sources_cpu = (
        reference.detach()
        .to(device="cpu", dtype=torch.int64)
        .reshape(BATCH, TOPK)
        .contiguous()
    )
    final_miss_cpu = miss_mask_cpu.gather(1, ranked_sources_cpu)
    final_hit_cpu = ~final_miss_cpu
    final_miss_count = int(torch.count_nonzero(final_miss_cpu).item())
    final_hit_count = int(torch.count_nonzero(final_hit_cpu).item())
    source_miss_count = int(torch.count_nonzero(miss_mask_cpu).item())

    rows_cpu = torch.arange(BATCH, dtype=torch.int64).reshape(-1, 1)
    ranks_cpu = torch.arange(TOPK, dtype=torch.int64).reshape(1, -1)
    persistent_slots_cpu = PERSISTENT_BASE + rows_cpu * TOPK + ranks_cpu

    if final_hit_count:
        if PERSISTENT_ROWS < BATCH * TOPK:
            raise AssertionError("profile has final hits but no persistent slot band")
        snapshot_cpu.scatter_(
            1,
            ranked_sources_cpu,
            torch.where(
                final_hit_cpu,
                persistent_slots_cpu.to(dtype=torch.int32),
                -torch.ones_like(persistent_slots_cpu, dtype=torch.int32),
            ),
        )

        # Gather the fixed resident-hit rows from the same local DDR witness
        # that Direct-HCCP reads for misses.  This setup is complete before
        # the timer starts and corresponds to pre-existing HBM cache content.
        row_cpu = rows_cpu.expand(BATCH, TOPK)[final_hit_cpu]
        source_cpu = ranked_sources_cpu[final_hit_cpu]
        entry_cpu = row_cpu * fixture.blocks + source_cpu // BLOCK_TOKENS
        token_cpu = source_cpu % BLOCK_TOKENS
        source_rows = fixture.data.reshape(
            fixture.entries, BLOCK_TOKENS, PACKED_KV_ROW_BYTES
        )[entry_cpu, token_cpu].contiguous()
        slot_device = persistent_slots_cpu[final_hit_cpu].to(
            device=fixture.device, dtype=torch.int64
        )
        fixture.resident_payload.index_copy_(0, slot_device, source_rows.to(fixture.device))

    # Preserve the original NPU object-snapshot storage identity; Stage B
    # receives exactly this alias from P10ForwardInputs.
    snapshot.copy_(snapshot_cpu.to(device=snapshot.device))
    torch.npu.synchronize()
    print(
        "P10_FINAL_HIT_METADATA_READY "
        "construction=cpu upload=npu "
        f"final_miss={final_miss_count} final_hit={final_hit_count}",
        flush=True,
    )
    print(
        "P10_FINAL_HIT_PAYLOAD_READY "
        "device=npu "
        f"hits={final_hit_count} bytes={final_hit_count * PACKED_KV_ROW_BYTES}",
        flush=True,
    )

    # The all-miss acceptance profile has no persistent rows and therefore
    # must never accidentally acquire a synthetic hit.
    if PERSISTENT_ROWS == 0 and final_hit_count:
        raise AssertionError("all-miss profile unexpectedly contains a hit")
    torch.npu.synchronize()
    fixture.residency_stats = {
        "source_miss_count": source_miss_count,
        "source_total": BATCH * SOURCE_TOKENS,
        "final_miss_count": final_miss_count,
        "final_hit_count": final_hit_count,
    }
    print(
        "P10_RUNTIME_RESIDENCY "
        f"profile={P10_BENCH_PROFILE} source_misses={source_miss_count} "
        f"source_total={BATCH * SOURCE_TOKENS} "
        f"source_miss_pct={100.0 * source_miss_count / (BATCH * SOURCE_TOKENS):.4f} "
        f"final_winner_misses={final_miss_count} final_winner_hits={final_hit_count} "
        f"persistent_base={PERSISTENT_BASE} persistent_rows={PERSISTENT_ROWS}",
        flush=True,
    )


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

_STAGE_B_POST_DETAIL_BASE = -210000
_STAGE_B_POST_DETAIL_STRIDE = 32
_STAGE_B_POST_DETAIL_NAMES = {
    1: "invalid_call",
    2: "no_free_wave",
    3: "control_load",
    4: "queue_acquire",
    5: "descriptor_build",
    6: "begin_invalid_args",
    7: "begin_invalid_entity",
    8: "begin_invalid_queue",
    9: "begin_invalid_lookup",
    10: "begin_queue_full",
    11: "begin_invalid_output",
    12: "begin_unknown",
    13: "cq_timeout",
    14: "cq_error",
    15: "cq_unknown",
    16: "queue_return",
    17: "cq_pending",
    18: "begin_state_idle",
    19: "begin_state_retired",
    20: "begin_state_failed_success",
    21: "begin_state_in_flight_returned_false",
    22: "begin_state_unknown",
    23: "begin_result_unknown",
    24: "begin_rehydrate",
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
    object_snapshot = tensors["object_snapshot"].detach().cpu()
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
    diagnostic_name = _STAGE_B_DIAGNOSTIC_NAMES.get(diagnostic, "unknown")
    if diagnostic <= _STAGE_B_POST_DETAIL_BASE - 1:
        encoded = _STAGE_B_POST_DETAIL_BASE - diagnostic
        ordinal, reason = divmod(encoded, _STAGE_B_POST_DETAIL_STRIDE)
        diagnostic_name = "post_detail"
        detail_name = _STAGE_B_POST_DETAIL_NAMES.get(reason, "unknown")
    elif diagnostic <= _STAGE_B_FINALIZE_DETAIL_BASE - 1:
        encoded = _STAGE_B_FINALIZE_DETAIL_BASE - diagnostic
        ordinal, reason = divmod(encoded, _STAGE_B_FINALIZE_DETAIL_STRIDE)
        detail_name = _STAGE_B_FINALIZE_DETAIL_NAMES.get(reason, "unknown")
    else:
        ordinal = 0
        reason = 0
        detail_name = "none"
    if diagnostic <= _STAGE_B_FINALIZE_DETAIL_BASE - 1:
        row, rank = divmod(ordinal, TOPK)
        if 0 <= row < canonical_sparse.size(0) and 0 <= rank < TOPK:
            source = int(canonical_sparse[row, 0, rank].item())
            source_slot = (
                int(source_slots[row, source].item())
                if 0 <= source < source_slots.size(1)
                else -999
            )
            object_slot = (
                int(object_slots[row, source].item())
                if 0 <= source < object_slots.size(1)
                else -999
            )
            persistent_slot = (
                int(object_snapshot[row, source].item())
                if 0 <= source < object_snapshot.size(1)
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
                f"reason={detail_name} "
                f"row={row} rank={rank} source={source} source_slot={source_slot} "
                f"object_slot={object_slot} persistent_slot={persistent_slot} resident_state={resident_state} "
                f"qsfa_value={qsfa_value} attention_value={attention_value}"
            )
        else:
            detail = f"reason={detail_name} ordinal={ordinal} out_of_fixture_range=1"
    print(
        "P10_RUNTIME_TERMINAL_DIAG "
        f"control={forward} stage_b_diag={diagnostic} "
        f"stage_b_diag_name={diagnostic_name} "
        f"receipt_free={count_eq(receipt, 0)} receipt_filling={count_eq(receipt, 1)} "
        f"receipt_ready={count_eq(receipt, 2)} receipt_claimed={count_eq(receipt, 3)} "
        f"receipt_done={count_eq(receipt, 4)} source_mapped={int(torch.count_nonzero(source_slots >= 0).item())} "
        f"object_mapped={int(torch.count_nonzero(object_slots >= 0).item())} "
        f"resident_empty={count_eq(ready, 0)} resident_in_flight={count_eq(ready, 1)} "
        f"resident_ready={count_eq(ready, 2)} speculative_cursor={int(cursor.reshape(-1)[0].item())} "
        f"queue_leases={lease_values} attention_prefix={prefix} stage_b_detail={detail}",
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
    if actual_lens.tolist() != [SOURCE_TOKENS] * BATCH:
        raise AssertionError(f"canonical score-source lengths mismatch: {actual_lens.tolist()}")
    forward = tensors["forward_control"].detach().cpu().tolist()
    if int(forward[5]) != 1:
        _emit_terminal_diagnostic(fixture, tensors, forward)
        raise AssertionError(f"Stage B did not reach forward COMPLETE: control={forward}")
    states = tensors["receipt_state"].detach().cpu()
    if int(torch.count_nonzero(states != 4).item()) != 0:
        raise AssertionError(
            "not every produced receipt reached DONE: "
            f"free={int(torch.count_nonzero(states == 0).item())} "
            f"filling={int(torch.count_nonzero(states == 1).item())} "
            f"ready={int(torch.count_nonzero(states == 2).item())} "
            f"claimed={int(torch.count_nonzero(states == 3).item())} "
            f"done={int(torch.count_nonzero(states == 4).item())}"
        )
    leases = fixture.leases.detach().cpu()
    if int(torch.count_nonzero(leases).item()) != 0:
        raise AssertionError(f"Direct-HCCP queue lease leaked: {leases.tolist()}")

    source_to_slot = tensors["source_to_slot"].detach().cpu()
    object_to_slot = tensors["object_to_slot"].detach().cpu()
    object_snapshot = tensors["object_snapshot"].detach().cpu()
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
    if lengths.tolist() != [ATTENTION_WIDTH] * BATCH:
        raise AssertionError(f"QSFA selected length/tail contract broke: {lengths.tolist()}")
    if not torch.equal(
        attention[:, 0, TOPK:],
        torch.full((BATCH, ATTENTION_TAIL), -77, dtype=torch.int32),
    ):
        raise AssertionError("Stage B modified caller-owned QSFA tail")

    mismatches: list[str] = []
    used_slots: set[int] = set()
    ranked_payload = hashlib.sha256()
    final_miss_count = 0
    final_hit_count = 0
    for row in range(BATCH):
        for rank, source_value in enumerate(expected[row, 0].tolist()):
            source = int(source_value)
            persistent_slot = int(object_snapshot[row, source])
            if persistent_slot >= 0:
                final_hit_count += 1
                slot = persistent_slot
                if slot < PERSISTENT_BASE or slot >= RESIDENT_ROWS:
                    mismatches.append(
                        f"persistent_slot_range:row={row}:rank={rank}:source={source}:slot={slot}"
                    )
                    break
            else:
                final_miss_count += 1
                slot = int(source_to_slot[row, source])
                if slot < 0 or slot >= PERSISTENT_BASE:
                    mismatches.append(
                        f"miss_slot_range:row={row}:rank={rank}:source={source}:slot={slot}"
                    )
                    break
                object_slot = int(object_to_slot[row, source])
                if object_slot != slot:
                    mismatches.append(
                        "object_dedup:"
                        f"row={row}:rank={rank}:source={source}:"
                        f"source_slot={slot}:object_slot={object_slot}"
                    )
                    break
                if int(ready[slot]) != 2:
                    mismatches.append(
                        f"not_ready:row={row}:rank={rank}:source={source}:slot={slot}:state={int(ready[slot])}"
                    )
                    break
            if int(attention[row, 0, rank]) != int(translation[slot]):
                mismatches.append(f"qsfa_translation:row={row}:rank={rank}:slot={slot}")
                break
            if not torch.equal(payload[slot], fixture.expected_row(row, source)):
                # One bounded CPU-side diagnostic after the product graph has
                # completed.  It adds no device synchronization or work and
                # distinguishes a bad WQE offset, a stale source row, and an
                # inbound-DMA/cache-visibility failure in one rerun.
                actual_payload = payload[slot].contiguous()
                expected_payload = fixture.expected_row(row, source).contiguous()
                byte_diff = actual_payload != expected_payload
                first_byte = int(torch.nonzero(byte_diff, as_tuple=False)[0].item())
                candidate_match = {
                    candidate: int(
                        torch.equal(actual_payload, fixture.expected_row(row, candidate))
                    )
                    for candidate in (0, 1, 2)
                }
                print(
                    "P10_RUNTIME_PAYLOAD_DIAG "
                    f"row={row} rank={rank} source={source} slot={slot} "
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
                mismatches.append(
                    f"payload:row={row}:rank={rank}:source={source}:slot={slot}"
                )
                break
            used_slots.add(slot)
            ranked_payload.update(payload[slot].contiguous().numpy().tobytes())
        if mismatches:
            break
    if mismatches:
        raise AssertionError("P10 forward-resident closure mismatch: " + ",".join(mismatches))
    if len(used_slots) != BATCH * TOPK:
        raise AssertionError(
            f"final winners alias resident rows: unique={len(used_slots)} "
            f"expected={BATCH * TOPK}"
        )
    if fixture.residency_stats and (
        final_miss_count != fixture.residency_stats["final_miss_count"]
        or final_hit_count != fixture.residency_stats["final_hit_count"]
    ):
        raise AssertionError(
            "final residency changed after setup: "
            f"expected_miss={fixture.residency_stats['final_miss_count']} actual_miss={final_miss_count} "
            f"expected_hit={fixture.residency_stats['final_hit_count']} actual_hit={final_hit_count}"
        )
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
        tail_sha16=_sha16(attention[:, 0, TOPK:]),
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
            f"final_winners={BATCH * TOPK} final_misses={final_miss_count} "
            f"final_hits={final_hit_count} resident_ready=1 payload=1 "
            f"queue_leases_zero=1 unique_slots={len(used_slots)}",
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


def _collect_speculation_telemetry(
    reference: Any, tensors: dict[str, Any]
) -> dict[str, int]:
    """Report admission quality after the already-terminal exact proof.

    These counters are observations, not scheduling inputs.  In particular,
    ``speculative_cursor`` is an admission-attempt counter because the device
    increments it before capacity rejection; it must never be reported as a
    successful early Get count.
    """

    import torch

    source_to_slot = tensors["source_to_slot"].detach().cpu()
    object_snapshot = tensors["object_snapshot"].detach().cpu()
    canonical = reference.detach().to(device="cpu", dtype=torch.int64).reshape(
        BATCH, TOPK
    )
    if source_to_slot.shape != (BATCH, SOURCE_TOKENS):
        raise AssertionError(
            "unexpected source_to_slot telemetry shape: "
            f"actual={tuple(source_to_slot.shape)}"
        )
    if object_snapshot.shape != (BATCH, SOURCE_TOKENS):
        raise AssertionError(
            "unexpected object_snapshot telemetry shape: "
            f"actual={tuple(object_snapshot.shape)}"
        )

    final_winner_miss_mask = torch.zeros_like(source_to_slot, dtype=torch.bool)
    final_miss = object_snapshot.gather(1, canonical) < 0
    winner_positions = torch.nonzero(final_miss, as_tuple=False)
    if winner_positions.numel():
        final_winner_miss_mask[
            winner_positions[:, 0],
            canonical[winner_positions[:, 0], winner_positions[:, 1]],
        ] = True

    early_mask = (source_to_slot >= 0) & (source_to_slot < SPECULATIVE_ROWS)
    admitted = int(torch.count_nonzero(early_mask).item())
    final_winner_hits = int(
        torch.count_nonzero(early_mask & final_winner_miss_mask).item()
    )
    cursor_attempts = int(
        tensors["speculative_cursor"].detach().cpu().reshape(-1)[0].item()
    )
    return {
        "speculative_admitted_sources": admitted,
        "speculative_final_winner_hits": final_winner_hits,
        "speculative_false_positive_sources": admitted - final_winner_hits,
        "speculative_cursor_attempts": cursor_attempts,
        "final_recall_sources": int(
            torch.count_nonzero(final_winner_miss_mask & ~early_mask).item()
        ),
        "mapped_sources": int(torch.count_nonzero(source_to_slot >= 0).item()),
    }


def _make_timing_sample(
    *,
    device_us: float,
    host_us: float,
    signature: P10SemanticSignature,
    reference: Any,
    tensors: dict[str, Any],
) -> P10TimingSample:
    telemetry = _collect_speculation_telemetry(reference, tensors)
    return P10TimingSample(
        device_us=device_us,
        host_us=host_us,
        signature=signature,
        **telemetry,
    )


def _speculation_metrics_text(sample: P10TimingSample) -> str:
    return (
        f"speculative_admitted_sources={sample.speculative_admitted_sources} "
        f"speculative_final_winner_hits={sample.speculative_final_winner_hits} "
        f"speculative_false_positive_sources={sample.speculative_false_positive_sources} "
        f"speculative_cursor_attempts={sample.speculative_cursor_attempts} "
        f"final_recall_sources={sample.final_recall_sources} "
        f"mapped_sources={sample.mapped_sources}"
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


def _load_graph_product_runtime() -> tuple[Any, Any, Any]:
    """Resolve the public ACLGraph adapter from the same product bridge.

    This deliberately layers on top of :func:`_load_product_runtime`: graph
    replay is not allowed to import a second bridge, a separately built C8
    OPP, or a host Get path just to make capture easier.
    """

    device, launch_p10_forward = _load_product_runtime()
    module = importlib.import_module("nanovllm_p10_forward_bridge")
    capture_p10_forward_graph = getattr(module, "capture_p10_forward_graph", None)
    if not callable(capture_p10_forward_graph):
        _blocked("product_bridge_aclgraph_adapter_missing")
    return device, launch_p10_forward, capture_p10_forward_graph


def _positive_environment_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise P10Blocked(f"invalid_{name.lower()}:{raw}") from error
    if value <= 0:
        _blocked(f"invalid_{name.lower()}:{raw}")
    return value


def _score_cutoff_u16() -> int:
    """Return the test-only ExactScoreScan cutoff without touching C8 math.

    Stage B consumes the low 16 bits of ``forward_control[6]`` as an unsigned
    score threshold. Bit 31 remains the independent receipt-trace flag.
    Fail closed rather than silently truncating an operator experiment into a
    different threshold.
    """

    raw = os.environ.get("P10_SCORE_CUTOFF_U16", "0")
    try:
        value = int(raw, 0)
    except ValueError as error:
        raise P10Blocked(f"invalid_p10_score_cutoff_u16:{raw}") from error
    if not 0 <= value <= 0xFFFF:
        _blocked(f"invalid_p10_score_cutoff_u16:{raw}")
    return value


def _validate_workload_profile() -> None:
    """Fail before allocating a multi-GiB fixture on a malformed profile."""

    if SOURCE_TOKENS <= 0 or SOURCE_TOKENS % BLOCK_TOKENS != 0:
        _blocked(f"invalid_source_tokens:{SOURCE_TOKENS}")
    if TOPK <= 0 or TOPK > SOURCE_TOKENS:
        _blocked(f"invalid_topk:{TOPK}")
    if not (0 < KV_MISS_NUMERATOR <= KV_MISS_DENOMINATOR):
        _blocked(
            "invalid_kv_miss_ratio:"
            f"{KV_MISS_NUMERATOR}/{KV_MISS_DENOMINATOR}"
        )
    # The current exact KVAF fixture initializes the one queue template used
    # by the product route.  Do not silently claim a multi-queue experiment
    # until the template construction is expanded too.
    if QUEUE_COUNT != 1:
        _blocked(f"unsupported_queue_count_for_exact_fixture:{QUEUE_COUNT}")
    if EXECUTOR_WORKERS <= 0 or EXECUTOR_WORKER_LIMIT < EXECUTOR_WORKERS:
        _blocked(
            "invalid_executor_workers:"
            f"count={EXECUTOR_WORKERS}:limit={EXECUTOR_WORKER_LIMIT}"
        )
    print(
        "P10_SCORE_CUTOFF_CONFIG "
        f"cutoff_u16={_score_cutoff_u16()} trace={int(STAGE_B_RECEIPT_TRACE)}",
        flush=True,
    )
    print(
        "P10_BENCH_WORKLOAD "
        f"profile={P10_BENCH_PROFILE} batch={BATCH} source_tokens={SOURCE_TOKENS} "
        f"topk={TOPK} kv_miss_ratio={KV_MISS_NUMERATOR}/{KV_MISS_DENOMINATOR} "
        f"executor_workers={EXECUTOR_WORKERS} queue_count={QUEUE_COUNT} "
        f"graph_capture=none_for_direct_hccp",
        flush=True,
    )


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
    """Reset forward metadata without writing an inbound-DMA destination.

    ``ForwardPrepare`` owns the per-forward reset of the slot maps,
    ``resident_ready``, queue leases, receipt state, and completion control.
    It makes every speculative/recall row unreachable until this forward's
    Direct-HCCP CQ has terminally published ``READY``.  Consequently the
    payload itself has no reset semantic: every final miss overwrites its
    fixed row before QSFA may read it, while final hits live in their separate
    persistent band.

    Do *not* use a torch kernel to zero the Direct-HCCP destination after it
    has been registered.  A producer-stream AIV may retain dirty zero cache
    lines that are invisible to the executor AIV's pre-doorbell clean; a
    later writeback can overwrite an otherwise successful inbound DMA.  The
    old zero_ therefore created exactly the invalid "CQ success + READY +
    zero payload" state that this graph-compatible path is meant to reject.
    """

    if fixture.resident_payload is None:
        raise AssertionError("P10 benchmark destination is not cold-bound")
    # Keep the parameter in the call boundary: it documents that this reset
    # occurs before work is placed on the stable producer stream, but there
    # is intentionally no payload write on any stream.
    del producer_stream
    print(
        "P10_BENCH_DESTINATION_RESET "
        "authority=forward_prepare payload_write=none "
        "resident_ready=empty_before_stage_b",
        flush=True,
    )


def _emit_stage_b_receipt_trace(
    tensors: dict[str, Any], *, context: str
) -> None:
    """Print one bounded, post-terminal explanation of the receipt path.

    This is intentionally a *diagnostic mode*, selected with
    ``P10_STAGE_B_RECEIPT_TRACE=1``.  Stage B then writes a bitset into the
    reserved high half of each receipt's fifth word after it owns the record;
    the source-owned low sequence half is preserved exactly.  No extra graph
    input, C8 workspace, scheduling edge, or synchronization is introduced.

    The result is an OR-style per-gate count rather than a verbose per-source
    dump.  It answers the one question needed to interpret a zero speculative
    cursor: did a receipt fail descriptor validation, have no eligible source,
    resolve only resident hits, lose slot admission, or defer its Post?
    """

    if not STAGE_B_RECEIPT_TRACE:
        return

    import torch

    ring = tensors["receipt_ring"].detach().cpu().reshape(-1, RECEIPT_WORDS)
    states = tensors["receipt_state"].detach().cpu().reshape(-1)
    # The trace lives in the high 32 bits.  Mask after the signed right shift
    # so a high diagnostic bit remains an ordinary unsigned bit pattern.
    trace_words = torch.bitwise_and(
        torch.bitwise_right_shift(ring[:, 4].to(dtype=torch.int64), 32),
        0xFFFFFFFF,
    )
    bit_counts = {
        name: int(torch.count_nonzero(torch.bitwise_and(trace_words, bit)).item())
        for name, bit in _STAGE_B_RECEIPT_TRACE_BITS.items()
    }
    # Decode the source-owned descriptor fields independently of the device
    # trace.  This makes a malformed Stage-A publication distinguishable from
    # a valid descriptor rejected later by Stage B.
    rows = ring[:, 1].to(dtype=torch.int64)
    sources = ring[:, 2].to(dtype=torch.int64)
    layout = ring[:, 3].to(dtype=torch.int64)
    row_begin = torch.bitwise_and(rows, 0xFFFFFFFF)
    row_count = torch.bitwise_and(torch.bitwise_right_shift(rows, 32), 0xFFFFFFFF)
    source_begin = torch.bitwise_and(sources, 0xFFFFFFFF)
    source_count = torch.bitwise_and(
        torch.bitwise_right_shift(sources, 32), 0xFFFFFFFF
    )
    row_stride = torch.bitwise_and(layout, 0xFFFFFFFF)
    score_bytes = torch.bitwise_and(
        torch.bitwise_right_shift(layout, 32), 0xFFFFFFFF
    )
    host_layout_valid = (
        (row_count > 0)
        & (source_count > 0)
        & (score_bytes == 2)
        & (row_stride >= source_count * 2)
        & (row_begin < BATCH)
        & (source_begin < SOURCE_TOKENS)
    )
    # Match the Stage-B predicate rather than calling a post-terminal host
    # layout check "descriptor_valid".  The device also rejects an odd byte
    # offset and a row outside the graph's active-row extent.  Those two
    # checks caught the shared-scratch overwrite that otherwise looked like a
    # contradictory host/device diagnosis.
    offsets = ring[:, 0].to(dtype=torch.int64)
    active_rows = int(tensors["forward_control"].detach().cpu()[4].item())
    offset_even = torch.bitwise_and(offsets, 1) == 0
    row_in_active = row_begin < active_rows
    device_descriptor_valid = host_layout_valid & offset_even & row_in_active
    state_counts = {
        f"state_{state}": int(torch.count_nonzero(states == state).item())
        for state in range(5)
    }
    print(
        "P10_STAGE_B_RECEIPT_TRACE "
        f"context={context} records={ring.size(0)} "
        f"trace_records={int(torch.count_nonzero(trace_words).item())} "
        f"host_layout_valid={int(torch.count_nonzero(host_layout_valid).item())} "
        f"device_descriptor_valid={int(torch.count_nonzero(device_descriptor_valid).item())} "
        f"device_descriptor_invalid={int(torch.count_nonzero(~device_descriptor_valid).item())} "
        f"offset_even={int(torch.count_nonzero(offset_even).item())} "
        f"row_in_active={int(torch.count_nonzero(row_in_active).item())} "
        f"active_rows={active_rows} "
        f"row_begin_min={int(row_begin.min().item())} "
        f"row_begin_max={int(row_begin.max().item())} "
        f"source_begin_min={int(source_begin.min().item())} "
        f"source_begin_max={int(source_begin.max().item())} "
        + " ".join(f"{name}={count}" for name, count in state_counts.items())
        + " "
        + " ".join(f"{name}={count}" for name, count in bit_counts.items()),
        flush=True,
    )


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
    _install_final_hit_payloads(reference, fixture, tensors)

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
    _emit_stage_b_receipt_trace(tensors, context=f"timed_{mode}")
    _require_immutable_inputs(immutable_digests, base, context=f"P10 {mode} run")
    return _make_timing_sample(
        device_us=device_us,
        host_us=host_us,
        signature=signature,
        reference=reference,
        tensors=tensors,
    )


def _run_timed_static_forward(
    *,
    mode: str,
    base: dict[str, Any],
    reference: Any,
    fixture: _ForwardResidentKvaf,
    launch_p10_forward: Any,
    inputs: Any,
    tensors: dict[str, Any],
    producer_stream: Any,
    executor_stream: Any,
    immutable_digests: dict[str, str],
) -> P10TimingSample:
    """Time one graph-compatible P10 replay with stable aliases and streams.

    Direct-HCCP is deliberately *not* captured with raw ACLGraph: that route
    has already shown capture-time state surviving while inbound DMA payloads
    do not.  The surrounding inference graph can retain its normal static
    buffers; P10 is a graph-break/escape sequence that reuses those same
    buffers and logical streams on every replay.  This is the only honest
    graph-mode measurement until CANN provides a replay contract for HCCP.
    """

    import torch

    if mode not in ("serial", "overlap"):
        raise ValueError(f"unsupported P10 graph-compatible mode: {mode}")
    _reset_benchmark_destination(fixture, producer_stream)
    fixture.terminal = False
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    host_start_ns = time.perf_counter_ns()
    with torch.npu.stream(producer_stream):
        start.record()
    launch = launch_p10_forward(
        inputs,
        producer_stream=producer_stream,
        executor_stream=executor_stream,
        execution_mode=mode,
    )
    aliases = launch.bind_qsfa(producer_stream)
    with torch.npu.stream(producer_stream):
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
    _emit_stage_b_receipt_trace(tensors, context=f"static_{mode}")
    _require_immutable_inputs(
        immutable_digests, base, context=f"P10 graph-compatible {mode} replay"
    )
    return _make_timing_sample(
        device_us=device_us,
        host_us=host_us,
        signature=signature,
        reference=reference,
        tensors=tensors,
    )


def _run_timed_graph_forward(
    *,
    mode: str,
    base: dict[str, Any],
    reference: Any,
    fixture: _ForwardResidentKvaf,
    graph: Any,
    inputs: Any,
    tensors: dict[str, Any],
    producer_stream: Any,
    immutable_digests: dict[str, str],
) -> P10TimingSample:
    """Time one ACLGraph replay through the identical P10 terminal boundary.

    Capture has already completed before this function is called.  The timer
    therefore measures only graph replay through the Stage-B -> QSFA event,
    directly comparable with :func:`_run_timed_forward`.  The host waits only
    after the end event to collect a validation result; it is not part of the
    forward dependency graph.
    """

    import torch

    if mode not in ("serial", "overlap"):
        raise ValueError(f"unsupported P10 ACLGraph benchmark mode: {mode}")
    if getattr(graph, "execution_mode", None) != mode:
        raise AssertionError(
            "P10 ACLGraph benchmark mode mismatch: "
            f"graph={getattr(graph, 'execution_mode', None)!r} requested={mode!r}"
        )

    _reset_benchmark_destination(fixture, producer_stream)
    fixture.terminal = False
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    host_start_ns = time.perf_counter_ns()
    with torch.npu.stream(producer_stream):
        start.record()
    launch = graph.replay(inputs)
    aliases = launch.bind_qsfa(producer_stream)
    with torch.npu.stream(producer_stream):
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
    _require_immutable_inputs(
        immutable_digests, base, context=f"P10 ACLGraph {mode} replay"
    )
    return _make_timing_sample(
        device_us=device_us,
        host_us=host_us,
        signature=signature,
        reference=reference,
        tensors=tensors,
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
                    f"host_us={sample.host_us:.3f} {_speculation_metrics_text(sample)}",
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
        speculative = [sample.speculative_admitted_sources for sample in samples[mode]]
        winner_hits = [sample.speculative_final_winner_hits for sample in samples[mode]]
        false_positives = [sample.speculative_false_positive_sources for sample in samples[mode]]
        cursor_attempts = [sample.speculative_cursor_attempts for sample in samples[mode]]
        final_recall = [sample.final_recall_sources for sample in samples[mode]]
        mapped = [sample.mapped_sources for sample in samples[mode]]
        print(
            "P10_PERF_WORK "
            f"mode={mode} speculative_admitted_sources_min={min(speculative)} "
            f"speculative_admitted_sources_max={max(speculative)} "
            f"speculative_final_winner_hits_min={min(winner_hits)} "
            f"speculative_final_winner_hits_max={max(winner_hits)} "
            f"speculative_false_positive_sources_min={min(false_positives)} "
            f"speculative_false_positive_sources_max={max(false_positives)} "
            f"speculative_cursor_attempts_min={min(cursor_attempts)} "
            f"speculative_cursor_attempts_max={max(cursor_attempts)} "
            f"final_recall_sources_min={min(final_recall)} "
            f"final_recall_sources_max={max(final_recall)} "
            f"mapped_sources_min={min(mapped)} mapped_sources_max={max(mapped)}",
            flush=True,
        )
    print(
        "P10_RUNTIME_PASS "
        "route=w4a4c8_score_receipt_direct_hccp_qsfa benchmark=serial_vs_overlap",
        flush=True,
    )
    return 0


def _run_graph_compatible_benchmark() -> int:
    """Compare serial and fused P10 in the valid graph-inference topology.

    This is intentionally not the old raw-ACLGraph benchmark.  Raw capture
    currently lacks a Direct-HCCP replay guarantee, so it cannot yield an
    interpretable payload or latency result.  Instead this path freezes all
    P10 tensor addresses and logical streams once, then invokes the exact
    product operators as the graph-compatible custom-op escape on each replay.
    The surrounding graph can therefore be stable without treating the HCCP
    doorbell as a capture-safe node.
    """

    import torch

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
        # These aliases and streams are held for the entire benchmark just as
        # vLLM's graph cache holds a replay slot.  The direct HCCP executor is
        # launched into the stable companion stream, never raw-captured.
        inputs, tensors = _build_forward_inputs(device, base, fixture)
        _install_final_hit_payloads(reference, fixture, tensors)
        producer = torch.npu.Stream(device=device)
        executor = torch.npu.Stream(device=device)
        print(
            "P10_GRAPH_COMPAT_CONFIG "
            "outer_graph=static_aliases p10_capture=none "
            "direct_hccp=direct_relaunch stable_streams=1 "
            "serial_added_edge=stage_a_complete_to_stage_b "
            f"profile={P10_BENCH_PROFILE} batch={BATCH} source_tokens={SOURCE_TOKENS}",
            flush=True,
        )

        for iteration in range(warmup):
            order = ("serial", "overlap") if iteration % 2 == 0 else ("overlap", "serial")
            for mode in order:
                _run_timed_static_forward(
                    mode=mode,
                    base=base,
                    reference=reference,
                    fixture=fixture,
                    launch_p10_forward=launch_p10_forward,
                    inputs=inputs,
                    tensors=tensors,
                    producer_stream=producer,
                    executor_stream=executor,
                    immutable_digests=immutable_digests,
                )
        print(
            "P10_GRAPH_COMPAT_WARMUP "
            f"pairs={warmup} static_aliases=1 terminal=1",
            flush=True,
        )

        for iteration in range(iterations):
            order = ("serial", "overlap") if iteration % 2 == 0 else ("overlap", "serial")
            for mode in order:
                sample = _run_timed_static_forward(
                    mode=mode,
                    base=base,
                    reference=reference,
                    fixture=fixture,
                    launch_p10_forward=launch_p10_forward,
                    inputs=inputs,
                    tensors=tensors,
                    producer_stream=producer,
                    executor_stream=executor,
                    immutable_digests=immutable_digests,
                )
                samples[mode].append(sample)
                print(
                    "P10_GRAPH_COMPAT_SAMPLE "
                    f"iteration={iteration} mode={mode} device_us={sample.device_us:.3f} "
                    f"host_us={sample.host_us:.3f} {_speculation_metrics_text(sample)}",
                    flush=True,
                )
    finally:
        fixture.close()

    serial = samples["serial"]
    overlap = samples["overlap"]
    if not serial or not overlap:
        raise AssertionError("P10 graph-compatible benchmark produced no paired samples")
    expected_signature = serial[0].signature
    if any(sample.signature != expected_signature for sample in serial + overlap):
        raise AssertionError("P10 graph-compatible serial/overlap semantic signature mismatch")

    # A fused timing only has an overlap interpretation if Stage B admitted a
    # pre-final-recall *final winner*.  Mere slot admission can consist wholly
    # of false positives; the later exact winner path then performs the same
    # final recall in both schedules and a saved-us number would be launch or
    # polling noise rather than HCCP hiding.  The separate one-iteration
    # trace also proves Post/CQ; this non-trace benchmark intentionally gates
    # only candidate usefulness, not a fabricated transport completion.
    overlap_speculative_min = min(
        sample.speculative_admitted_sources for sample in overlap
    )
    overlap_winner_hit_min = min(
        sample.speculative_final_winner_hits for sample in overlap
    )
    if overlap_speculative_min == 0 or overlap_winner_hit_min == 0:
        serial_speculative_min = min(
            sample.speculative_admitted_sources for sample in serial
        )
        serial_winner_hit_min = min(
            sample.speculative_final_winner_hits for sample in serial
        )
        print(
            "P10_GRAPH_COMPAT_INCONCLUSIVE "
            "reason=overlap_has_no_useful_pre_final_winner_candidate_do_not_claim_hccp_hiding_gain "
            f"serial_speculative_admitted_sources_min={serial_speculative_min} "
            f"overlap_speculative_admitted_sources_min={overlap_speculative_min} "
            f"serial_speculative_final_winner_hits_min={serial_winner_hit_min} "
            f"overlap_speculative_final_winner_hits_min={overlap_winner_hit_min} "
            "next=P10_STAGE_B_RECEIPT_TRACE",
            flush=True,
        )
        raise AssertionError(
            "P10 graph-compatible overlap had no useful pre-final winner "
            "candidate; inspect P10_STAGE_B_RECEIPT_TRACE before interpreting latency"
        )

    serial_device = [sample.device_us for sample in serial]
    overlap_device = [sample.device_us for sample in overlap]
    serial_host = [sample.host_us for sample in serial]
    overlap_host = [sample.host_us for sample in overlap]
    _emit_perf_result(
        "p10_w4a4c8_graph_compatible_serial",
        serial_device,
        warmup=warmup,
        metric="npu_event_forward_prepare_to_qsfa_ready",
    )
    _emit_perf_result(
        "p10_w4a4c8_graph_compatible_fused",
        overlap_device,
        warmup=warmup,
        metric="npu_event_forward_prepare_to_qsfa_ready",
    )
    _emit_perf_result(
        "p10_w4a4c8_graph_compatible_serial_host_diagnostic",
        serial_host,
        warmup=warmup,
        metric="host_submit_to_terminal_diagnostic",
    )
    _emit_perf_result(
        "p10_w4a4c8_graph_compatible_fused_host_diagnostic",
        overlap_host,
        warmup=warmup,
        metric="host_submit_to_terminal_diagnostic",
    )
    serial_p50 = _percentile(serial_device, 50.0)
    fused_p50 = _percentile(overlap_device, 50.0)
    serial_p90 = _percentile(serial_device, 90.0)
    fused_p90 = _percentile(overlap_device, 90.0)
    print(
        "P10_GRAPH_COMPAT_EQ "
        "product_w4a4c8_oracle=1 canonical_topk=1 score_source_lens=1 "
        "ranked_resident_payload=1 qsfa_lengths_and_tail=1 "
        "same_source_miss_mask=1 static_aliases=1 raw_aclgraph_capture=0 "
        "candidate_winner_coverage=1 trace_required_for_hccp_completion=1",
        flush=True,
    )
    print(
        "P10_GRAPH_COMPAT_RESULT "
        f"serial_p50_us={serial_p50:.3f} fused_p50_us={fused_p50:.3f} "
        f"saved_p50_us={serial_p50 - fused_p50:.3f} "
        f"saved_p50_pct={100.0 * (serial_p50 - fused_p50) / serial_p50:.3f} "
        f"serial_p90_us={serial_p90:.3f} fused_p90_us={fused_p90:.3f} "
        f"saved_p90_us={serial_p90 - fused_p90:.3f} "
        f"saved_p90_pct={100.0 * (serial_p90 - fused_p90) / serial_p90:.3f}",
        flush=True,
    )
    for mode, values in (("serial", serial), ("fused", overlap)):
        speculative = [sample.speculative_admitted_sources for sample in values]
        winner_hits = [sample.speculative_final_winner_hits for sample in values]
        false_positives = [sample.speculative_false_positive_sources for sample in values]
        cursor_attempts = [sample.speculative_cursor_attempts for sample in values]
        final_recall = [sample.final_recall_sources for sample in values]
        mapped = [sample.mapped_sources for sample in values]
        print(
            "P10_GRAPH_COMPAT_WORK "
            f"mode={mode} speculative_admitted_sources_min={min(speculative)} "
            f"speculative_admitted_sources_max={max(speculative)} "
            f"speculative_final_winner_hits_min={min(winner_hits)} "
            f"speculative_final_winner_hits_max={max(winner_hits)} "
            f"speculative_false_positive_sources_min={min(false_positives)} "
            f"speculative_false_positive_sources_max={max(false_positives)} "
            f"speculative_cursor_attempts_min={min(cursor_attempts)} "
            f"speculative_cursor_attempts_max={max(cursor_attempts)} "
            f"final_recall_sources_min={min(final_recall)} "
            f"final_recall_sources_max={max(final_recall)} "
            f"mapped_sources_min={min(mapped)} mapped_sources_max={max(mapped)}",
            flush=True,
        )
    print(
        "P10_RUNTIME_PASS "
        "route=w4a4c8_score_receipt_direct_hccp_qsfa "
        "benchmark=graph_compatible_serial_vs_fused",
        flush=True,
    )
    return 0


def _run_graph_benchmark() -> int:
    """Compare eager and ACLGraph replay on the exact P10 product closure.

    The only baseline difference is launch form and the one explicit
    Stage-A-complete edge in serial mode.  All four cases use the same
    W4A4C8 inputs, resident mappings, Direct-HCCP control, terminal recall,
    and QSFA aliases.  Capture is performed before warmup and is deliberately
    excluded from timing: steady-state inference replays an already captured
    graph rather than recapturing each forward.
    """

    # Direct-HCCP's raw ACLGraph replay contract is not established.  The
    # small historical graph fixture remains available for capture debugging,
    # but it is explicitly not a valid capacity/latency result for the real
    # B16 workload.  Route that workload through the fixed-alias, direct-HCCP
    # graph-compatible path instead of silently measuring a poisoned replay.
    if P10_BENCH_PROFILE != "acceptance":
        _blocked(
            "raw_aclgraph_benchmark_not_valid_for_direct_hccp:"
            f"profile={P10_BENCH_PROFILE}:use_graph_compatible_benchmark"
        )

    import torch

    warmup = _positive_environment_int("P10_PERF_WARMUP", 1)
    iterations = _positive_environment_int("P10_PERF_ITERS", 5)
    device, launch_p10_forward, capture_p10_forward_graph = _load_graph_product_runtime()
    base = _build_w4a4c8_inputs(device)
    immutable_digests = _immutable_input_digests(base)
    reference = _reference_w4a4c8(base)
    _require_immutable_inputs(immutable_digests, base, context="product oracle")
    fixture = _ForwardResidentKvaf(device)
    cases = (
        "eager_serial",
        "eager_overlap",
        "graph_serial",
        "graph_overlap",
    )
    samples: dict[str, list[P10TimingSample]] = {case: [] for case in cases}
    graph_serial: Any | None = None
    graph_overlap: Any | None = None

    try:
        fixture.bind()

        # Each ACLGraph owns persistent, logical streams.  These are not an
        # AIV partition: both streams submit the normal dynamic P10 workers.
        # Separate input/output aliases are required because graph replay is
        # intentionally fail-closed on any captured-address drift.
        serial_producer = torch.npu.Stream(device=device)
        serial_executor = torch.npu.Stream(device=device)
        serial_inputs, serial_tensors = _build_forward_inputs(device, base, fixture)
        graph_serial = capture_p10_forward_graph(
            serial_inputs,
            producer_stream=serial_producer,
            executor_stream=serial_executor,
            execution_mode="serial",
        )

        overlap_producer = torch.npu.Stream(device=device)
        overlap_executor = torch.npu.Stream(device=device)
        overlap_inputs, overlap_tensors = _build_forward_inputs(device, base, fixture)
        graph_overlap = capture_p10_forward_graph(
            overlap_inputs,
            producer_stream=overlap_producer,
            executor_stream=overlap_executor,
            execution_mode="overlap",
        )
        print(
            "P10_GRAPH_CAPTURE "
            "segments=2 backend=aclmdlRI_raw_direct_aclnn "
            "captured=prepare,stage_a stage_b=direct_aclnn_relaunch "
            "direct_hccp_captured=0 "
            "graph_serial=1 graph_overlap=1 static_alias_gate=1 "
            f"captured_segments_serial={graph_serial.captured_segment_count} "
            f"captured_segments_overlap={graph_overlap.captured_segment_count} "
            f"workspace_buffers_serial={sum(graph_serial.workspace_counts)} "
            f"workspace_buffers_overlap={sum(graph_overlap.workspace_counts)}",
            flush=True,
        )

        def run_case(case: str) -> P10TimingSample:
            if case == "eager_serial":
                return _run_timed_forward(
                    mode="serial",
                    device=device,
                    base=base,
                    reference=reference,
                    fixture=fixture,
                    launch_p10_forward=launch_p10_forward,
                    immutable_digests=immutable_digests,
                )
            if case == "eager_overlap":
                return _run_timed_forward(
                    mode="overlap",
                    device=device,
                    base=base,
                    reference=reference,
                    fixture=fixture,
                    launch_p10_forward=launch_p10_forward,
                    immutable_digests=immutable_digests,
                )
            if case == "graph_serial":
                return _run_timed_graph_forward(
                    mode="serial",
                    base=base,
                    reference=reference,
                    fixture=fixture,
                    graph=graph_serial,
                    inputs=serial_inputs,
                    tensors=serial_tensors,
                    producer_stream=serial_producer,
                    immutable_digests=immutable_digests,
                )
            if case == "graph_overlap":
                return _run_timed_graph_forward(
                    mode="overlap",
                    base=base,
                    reference=reference,
                    fixture=fixture,
                    graph=graph_overlap,
                    inputs=overlap_inputs,
                    tensors=overlap_tensors,
                    producer_stream=overlap_producer,
                    immutable_digests=immutable_digests,
                )
            raise AssertionError(f"unknown P10 graph benchmark case: {case}")

        # Rotate every round so queue/cache state cannot consistently favor
        # either eager or graph replay, or serial versus overlap.
        for iteration in range(warmup):
            order = cases[iteration % len(cases):] + cases[:iteration % len(cases)]
            for case in order:
                run_case(case)
        print(
            "P10_GRAPH_WARMUP "
            f"rounds={warmup} cases={','.join(cases)} terminal=1",
            flush=True,
        )

        for iteration in range(iterations):
            order = cases[iteration % len(cases):] + cases[:iteration % len(cases)]
            for case in order:
                sample = run_case(case)
                samples[case].append(sample)
                print(
                    "P10_GRAPH_PERF_SAMPLE "
                    f"iteration={iteration} case={case} device_us={sample.device_us:.3f} "
                    f"host_us={sample.host_us:.3f} {_speculation_metrics_text(sample)}",
                    flush=True,
                )
    finally:
        try:
            # Raw ACL graphs retain their ACLNN workspaces explicitly. Tear
            # them down only after each timed sample synchronized Stage B's
            # terminal event, and always before KVAF releases Direct-HCCP.
            for graph in (graph_overlap, graph_serial):
                if graph is not None:
                    graph.close()
        finally:
            fixture.close()

    expected_signature = samples["eager_serial"][0].signature if samples["eager_serial"] else None
    if expected_signature is None or any(not samples[case] for case in cases):
        raise AssertionError("P10 ACLGraph benchmark produced incomplete timing samples")
    for case in cases:
        if any(sample.signature != expected_signature for sample in samples[case]):
            raise AssertionError(
                "P10 ACLGraph semantic signature mismatch: "
                f"case={case} expected={expected_signature!r}"
            )

    device_values = {case: [sample.device_us for sample in samples[case]] for case in cases}
    host_values = {case: [sample.host_us for sample in samples[case]] for case in cases}
    for case, operation in (
        ("eager_serial", "p10_w4a4c8_forward_resident_eager_serial"),
        ("eager_overlap", "p10_w4a4c8_forward_resident_eager_overlap"),
        ("graph_serial", "p10_w4a4c8_forward_resident_graph_serial"),
        ("graph_overlap", "p10_w4a4c8_forward_resident_graph_overlap"),
    ):
        _emit_perf_result(
            operation,
            device_values[case],
            warmup=warmup,
            metric="npu_event_forward_prepare_to_qsfa_ready",
        )
        _emit_perf_result(
            f"{operation}_host_diagnostic",
            host_values[case],
            warmup=warmup,
            metric="host_submit_to_terminal_diagnostic",
        )

    p50 = {case: _percentile(device_values[case], 50.0) for case in cases}
    p90 = {case: _percentile(device_values[case], 90.0) for case in cases}
    eager_serial_p50 = p50["eager_serial"]
    eager_serial_p90 = p90["eager_serial"]
    overlap_speculative = [
        sample.speculative_admitted_sources for sample in samples["graph_overlap"]
    ]
    print(
        "P10_GRAPH_PERF_EQ "
        "product_w4a4c8_oracle=1 canonical_topk=1 score_source_lens=1 "
        "ranked_resident_payload=1 qsfa_lengths_and_tail=1 "
        "eager_serial_terminal=1 graph_serial_terminal=1 "
        "graph_overlap_preseal_claim=1 aclgraph_segments=2 "
        "stage_b_direct_relaunch=1 direct_hccp_captured=0",
        flush=True,
    )
    print(
        "P10_GRAPH_PERF_RESULT "
        f"eager_serial_p50_us={eager_serial_p50:.3f} "
        f"eager_overlap_p50_us={p50['eager_overlap']:.3f} "
        f"graph_serial_p50_us={p50['graph_serial']:.3f} "
        f"graph_overlap_p50_us={p50['graph_overlap']:.3f} "
        f"graph_overlap_saved_vs_eager_serial_p50_us={eager_serial_p50 - p50['graph_overlap']:.3f} "
        f"graph_overlap_saved_vs_eager_serial_p50_pct={100.0 * (eager_serial_p50 - p50['graph_overlap']) / eager_serial_p50:.3f} "
        f"eager_serial_p90_us={eager_serial_p90:.3f} "
        f"graph_overlap_p90_us={p90['graph_overlap']:.3f} "
        f"graph_overlap_saved_vs_eager_serial_p90_us={eager_serial_p90 - p90['graph_overlap']:.3f}",
        flush=True,
    )
    for case in cases:
        speculative = [sample.speculative_admitted_sources for sample in samples[case]]
        winner_hits = [sample.speculative_final_winner_hits for sample in samples[case]]
        false_positives = [sample.speculative_false_positive_sources for sample in samples[case]]
        cursor_attempts = [sample.speculative_cursor_attempts for sample in samples[case]]
        final_recall = [sample.final_recall_sources for sample in samples[case]]
        mapped = [sample.mapped_sources for sample in samples[case]]
        print(
            "P10_GRAPH_PERF_WORK "
            f"case={case} speculative_admitted_sources_min={min(speculative)} "
            f"speculative_admitted_sources_max={max(speculative)} "
            f"speculative_final_winner_hits_min={min(winner_hits)} "
            f"speculative_final_winner_hits_max={max(winner_hits)} "
            f"speculative_false_positive_sources_min={min(false_positives)} "
            f"speculative_false_positive_sources_max={max(false_positives)} "
            f"speculative_cursor_attempts_min={min(cursor_attempts)} "
            f"speculative_cursor_attempts_max={max(cursor_attempts)} "
            f"final_recall_sources_min={min(final_recall)} "
            f"final_recall_sources_max={max(final_recall)} "
            f"mapped_sources_min={min(mapped)} mapped_sources_max={max(mapped)}",
            flush=True,
        )
    if min(overlap_speculative) == 0:
        print(
            "P10_GRAPH_PERF_INCONCLUSIVE "
            "reason=graph_overlap_has_no_speculative_admission_do_not_claim_hccp_hiding_gain",
            flush=True,
        )
    print(
        "P10_RUNTIME_PASS "
        "route=w4a4c8_score_receipt_direct_hccp_qsfa_aclgraph "
        "benchmark=eager_serial_vs_eager_overlap_vs_graph_serial_vs_graph_overlap",
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
        _install_final_hit_payloads(reference, fixture, tensors)
        launch = launch_p10_forward(inputs)
        _assert_terminal(reference, fixture, tensors, launch)
    finally:
        fixture.close()
    print("P10_RUNTIME_PASS route=w4a4c8_score_receipt_direct_hccp_qsfa", flush=True)
    return 0


def _main() -> int:
    _validate_workload_profile()
    runtime_case = os.environ.get("P10_RUNTIME_CASE", "acceptance")
    if runtime_case == "acceptance":
        return _run_acceptance()
    if runtime_case == "benchmark":
        return _run_benchmark()
    if runtime_case == "graph_benchmark":
        return _run_graph_benchmark()
    if runtime_case == "graph_compatible_benchmark":
        return _run_graph_compatible_benchmark()
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
