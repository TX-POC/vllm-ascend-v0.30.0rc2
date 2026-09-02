#!/usr/bin/env python3
"""Apply the atomic active P9 source-derived C8 overlay.

The overlay is deliberately mechanical: it extends the official C8 ABI from
9/2 to P9's exact 23/6 schema, wires those values through the official kernel
wrapper and QLIPreload/QLIVector path, then inserts only two bounded source
seams.  It never rewrites C8's score reduction or Top-K selection.  Every
transformation is an exact anchor; an upstream drift fails before CANN is
invoked instead of producing a subtly mixed binary.
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
PRELOAD_RELATIVE = Path("op_kernel/arch35/quant_lightning_indexer_kernel.h")
KERNEL_WRAPPER_RELATIVE = Path("op_kernel/vllm_quant_lightning_indexer.cpp")
TILING_HEADER_RELATIVE = Path("op_host/vllm_quant_lightning_indexer_tiling.h")
TILING_SOURCE_RELATIVE = Path("op_host/vllm_quant_lightning_indexer_tiling.cpp")
HOST_DEF_RELATIVE = Path("op_host/vllm_quant_lightning_indexer_def.cpp")
INFER_SHAPE_RELATIVE = Path("op_host/vllm_quant_lightning_indexer_infershape.cpp")
SEAM_HEADER_RELATIVE = Path(
    "op_kernel/arch35/a5_c8_exact_speculative_resident_score_seam.h"
)
COMPAT_HEADER_RELATIVE = Path(
    "op_host/a5_c8_exact_speculative_resident_host_compat.h"
)
OPS_RELATIVE = Path("ops.json")
SEAM_INCLUDE = '#include "a5_c8_exact_speculative_resident_score_seam.h"\n'
SEAM_MARKER = "A5_C8_EXACT_SPECULATIVE_RESIDENT_OBSERVE_SCORE_SEAM("
POST_MARKER = "A5_C8_EXACT_SPECULATIVE_RESIDENT_POST_AFTER_SCORE_PUBLICATION("
FINALIZER_MARKER = "A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER_TAIL("

# These are the two exact official Top-K ownership boundaries.  P9's
# terminalizer runs after V has finished producing `indicesOutLocal_` but
# *before* MTE3 begins the unchanged local-to-GM sparse-index DMA.  This is
# intentional: it is the only legal way to inspect local indices without
# consuming or trying to restore the official TOPK_MTE3_V token.  The
# invalid-row path continues after the original writeback; the normal path
# closes the row loop.  Literal anchors make an upstream drift fail closed.
_TOPK_WRITEBACK = (
    "AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + "
    "(curS1Idx + rowIdx) * topkCount_], "
    "indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);\n"
)
_TOPK_MTE3_V = "SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);"
_TOPK_V_MTE3_WAIT = "WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);"
_TOPK_INVALID_FINALIZER_ANCHOR = (
    "            " + _TOPK_V_MTE3_WAIT + "\n            " + _TOPK_WRITEBACK
)
_TOPK_NORMAL_FINALIZER_ANCHOR = (
    "        " + _TOPK_V_MTE3_WAIT + "\n        " + _TOPK_WRITEBACK
)
_TOPK_INVALID_FINALIZER_TAIL = (
    "            A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER_TAIL(\n"
    "                this, curS1Idx + rowIdx);\n"
)
_TOPK_NORMAL_FINALIZER_TAIL = (
    "        A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER_TAIL(\n"
    "            this, curS1Idx + rowIdx);\n"
)


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


def strip_stage_b_finalizer_tails(topk: str) -> str:
    """Remove only the two controlled P9 tail insertions from ProcessTopK.

    The result must be byte-for-byte the official body.  This is stronger
    than checking an output property: it prevents a future overlay edit from
    changing an official calculation, writeback, or event order under cover
    of the active finalizer.
    """

    stripped = topk
    for label, tail in (
        ("invalid", _TOPK_INVALID_FINALIZER_TAIL),
        ("normal", _TOPK_NORMAL_FINALIZER_TAIL),
    ):
        count = stripped.count(tail)
        if count != 1:
            fail(f"stage_b_finalizer_{label}_tail_count:actual={count}")
        stripped = stripped.replace(tail, "", 1)
    return stripped


def inject_stage_b_finalizer_tail(source: str, topk_before: str) -> str:
    """Insert the active finalizer at the local-buffer ownership boundary.

    There is intentionally no insertion inside C8's Top-K selection/reduction
    loops and no change to either official GM writeback/event chain.  The
    finalizer receives `indicesOutLocal_` only after the original V->MTE3
    hand-off has completed and returns before original MTE3 DMA begins.
    Deleting the two known tail snippets must recover ProcessTopK exactly.
    """

    source = replace_exactly_once(
        source,
        _TOPK_INVALID_FINALIZER_ANCHOR,
        "            " + _TOPK_V_MTE3_WAIT + "\n" + _TOPK_INVALID_FINALIZER_TAIL
        + "            " + _TOPK_WRITEBACK,
        "topk_invalid_local_index_boundary",
    )
    source = replace_exactly_once(
        source,
        _TOPK_NORMAL_FINALIZER_ANCHOR,
        "        " + _TOPK_V_MTE3_WAIT + "\n" + _TOPK_NORMAL_FINALIZER_TAIL
        + "        " + _TOPK_WRITEBACK,
        "topk_normal_local_index_boundary",
    )

    topk_after = function_body(source, "QLIVector<QLIT>::ProcessTopK")
    if topk_after.count(FINALIZER_MARKER) != 2:
        fail(
            "stage_b_finalizer_marker_count:actual="
            f"{topk_after.count(FINALIZER_MARKER)}"
        )
    if strip_stage_b_finalizer_tails(topk_after) != topk_before:
        fail(
            "official_process_topk_changed_outside_finalizer:before="
            f"{sha256_text(topk_before)}:after="
            f"{sha256_text(strip_stage_b_finalizer_tails(topk_after))}"
        )
    return source


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


def inject_kernel_wrapper_abi(wrapper_path: Path) -> None:
    """Extend the device entrypoint and its one QLIPreload call atomically.

    The six CANN outputs remain ordered as official sparse outputs 0/1,
    P9 public outputs 2/3, and the two graph-visible aliases of inputs 17/18
    at 4/5.  The alias pointers are intentionally passed through the wrapper
    but are not used as a second directory by the device implementation.
    """

    source = checked_file(wrapper_path, "official_kernel_wrapper").read_text(
        encoding="utf-8"
    )
    old_init = (
        "        op.Init(query, key, weights, queryScale, keyScale, actualSeqLengthsQ, actualSeqLengthsK, blocktable, \\\n"
        "                metadata, sparseIndices, user, tiling_data, &tPipe);"
    )
    new_init = (
        "        op.Init(query, key, weights, queryScale, keyScale, actualSeqLengthsQ, actualSeqLengthsK, blocktable, \\\n"
        "                metadata, runtimeEnable, residentThreshold, residentThresholdValid, reqPoolEntries, kvObjectMap, \\\n"
        "                kvObjectLens, persistentObjectSlotsSnapshot, canonicalResidentSlots, candidateSourceToSlot, \\\n"
        "                candidateSlotCursor, layerResidentGetControl, rowsPerEntry, residentBlockTable, activeRowCount, \\\n"
        "                sparseIndices, attentionSlots, \\\n"
        "                terminalPostStats, candidateSourceToSlotRef, candidateSlotCursorRef, user, tiling_data, &tPipe);"
    )
    source = replace_exactly_once(source, old_init, new_init, "kernel_wrapper_init")

    old_args = (
        "                                                   __gm__ uint8_t *blocktable, __gm__ uint8_t *metadata,\n"
        "                                                   __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,\n"
        "                                                   __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)"
    )
    new_args = (
        "                                                   __gm__ uint8_t *blocktable, __gm__ uint8_t *metadata,\n"
        "                                                   __gm__ uint8_t *runtimeEnable, __gm__ uint8_t *residentThreshold,\n"
        "                                                   __gm__ uint8_t *residentThresholdValid, __gm__ uint8_t *reqPoolEntries,\n"
        "                                                   __gm__ uint8_t *kvObjectMap, __gm__ uint8_t *kvObjectLens,\n"
        "                                                   __gm__ uint8_t *persistentObjectSlotsSnapshot, __gm__ uint8_t *canonicalResidentSlots,\n"
        "                                                   __gm__ uint8_t *candidateSourceToSlot, __gm__ uint8_t *candidateSlotCursor,\n"
        "                                                   __gm__ uint8_t *layerResidentGetControl, __gm__ uint8_t *rowsPerEntry,\n"
        "                                                   __gm__ uint8_t *residentBlockTable, __gm__ uint8_t *activeRowCount,\n"
        "                                                   __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,\n"
        "                                                   __gm__ uint8_t *attentionSlots, __gm__ uint8_t *terminalPostStats,\n"
        "                                                   __gm__ uint8_t *candidateSourceToSlotRef, __gm__ uint8_t *candidateSlotCursorRef,\n"
        "                                                   __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)"
    )
    source = replace_exactly_once(source, old_args, new_args, "kernel_wrapper_signature")
    source = replace_exactly_once(
        source,
        "    __gm__ uint8_t *user = GetUserWorkspace(workspace);\n",
        "    // P9 aliases 17/18 are graph-visible outputs of their input storage; "
        "the C8 device path owns only the input pointers.\n"
        "    (void)sparseValues;\n"
        "    (void)candidateSourceToSlotRef;\n"
        "    (void)candidateSlotCursorRef;\n"
        "    __gm__ uint8_t *user = GetUserWorkspace(workspace);\n",
        "kernel_wrapper_alias_comment",
    )
    wrapper_path.write_text(source, encoding="utf-8")


def inject_preload_abi(preload_path: Path) -> None:
    """Pass the 14 P9 inputs into QLIVector after its official UB setup.

    `InitP9Runtime` deliberately runs *after* `InitBuffers()` and only on an
    AIV.  It borrows C8's existing invalid-index scratch, so calling it when
    the original `QLIVector::InitBuffers` has not initialized that local
    tensor would create an uninitialized-UB bug even on a disabled runtime
    path.  The source anchor below makes that ordering part of the overlay
    contract rather than an accidental property of the current source.
    """

    source = checked_file(preload_path, "official_preload_source").read_text(
        encoding="utf-8"
    )
    old_decl = (
        "                                __gm__ uint8_t *metadata, __gm__ uint8_t *sparseIndices,\n"
        "                                __gm__ uint8_t *workspace, const QLITilingData *__restrict tiling, TPipe *tPipe);"
    )
    new_decl = (
        "                                __gm__ uint8_t *metadata, __gm__ uint8_t *runtimeEnable,\n"
        "                                __gm__ uint8_t *residentThreshold, __gm__ uint8_t *residentThresholdValid,\n"
        "                                __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *kvObjectMap,\n"
        "                                __gm__ uint8_t *kvObjectLens, __gm__ uint8_t *persistentObjectSlotsSnapshot,\n"
        "                                __gm__ uint8_t *canonicalResidentSlots, __gm__ uint8_t *candidateSourceToSlot,\n"
        "                                __gm__ uint8_t *candidateSlotCursor, __gm__ uint8_t *layerResidentGetControl,\n"
        "                                __gm__ uint8_t *rowsPerEntry, __gm__ uint8_t *residentBlockTable,\n"
        "                                __gm__ uint8_t *activeRowCount, __gm__ uint8_t *sparseIndices,\n"
        "                                __gm__ uint8_t *attentionSlots, __gm__ uint8_t *terminalPostStats,\n"
        "                                __gm__ uint8_t *candidateSourceToSlotRef, __gm__ uint8_t *candidateSlotCursorRef,\n"
        "                                __gm__ uint8_t *workspace, const QLITilingData *__restrict tiling, TPipe *tPipe);"
    )
    source = replace_exactly_once(source, old_decl, new_decl, "preload_declaration")

    old_impl = (
        "                                              __gm__ uint8_t *blockTable, __gm__ uint8_t *metadata,\n"
        "                                              __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,\n"
        "                                              const QLITilingData *__restrict tiling, TPipe *tPipe)"
    )
    new_impl = (
        "                                              __gm__ uint8_t *blockTable, __gm__ uint8_t *metadata,\n"
        "                                              __gm__ uint8_t *runtimeEnable, __gm__ uint8_t *residentThreshold,\n"
        "                                              __gm__ uint8_t *residentThresholdValid, __gm__ uint8_t *reqPoolEntries,\n"
        "                                              __gm__ uint8_t *kvObjectMap, __gm__ uint8_t *kvObjectLens,\n"
        "                                              __gm__ uint8_t *persistentObjectSlotsSnapshot, __gm__ uint8_t *canonicalResidentSlots,\n"
        "                                              __gm__ uint8_t *candidateSourceToSlot, __gm__ uint8_t *candidateSlotCursor,\n"
        "                                              __gm__ uint8_t *layerResidentGetControl, __gm__ uint8_t *rowsPerEntry,\n"
        "                                              __gm__ uint8_t *residentBlockTable, __gm__ uint8_t *activeRowCount,\n"
        "                                              __gm__ uint8_t *sparseIndices, __gm__ uint8_t *attentionSlots,\n"
        "                                              __gm__ uint8_t *terminalPostStats, __gm__ uint8_t *candidateSourceToSlotRef,\n"
        "                                              __gm__ uint8_t *candidateSlotCursorRef, __gm__ uint8_t *workspace,\n"
        "                                              const QLITilingData *__restrict tiling, TPipe *tPipe)"
    )
    source = replace_exactly_once(source, old_impl, new_impl, "preload_definition")

    init_tail = "    InitBuffers();\n}\n"
    p9_init_tail = (
        "    InitBuffers();\n"
        "    // The P9 vector state borrows outInvalidLocal_, which the official\n"
        "    // AIV-only InitBuffers call has initialized immediately above.\n"
        "    if ASCEND_IS_AIV {\n"
        "        vectorService.InitP9Runtime(runtimeEnable, residentThreshold, residentThresholdValid,\n"
        "                                       reqPoolEntries, kvObjectMap, kvObjectLens,\n"
        "                                       persistentObjectSlotsSnapshot, canonicalResidentSlots,\n"
        "                                       candidateSourceToSlot, candidateSlotCursor,\n"
        "                                       layerResidentGetControl, rowsPerEntry, residentBlockTable, activeRowCount,\n"
        "                                       attentionSlots,\n"
        "                                       terminalPostStats, tiling);\n"
        "    }\n"
        "    // Outputs 4/5 are ABI aliases of inputs 17/18.  They deliberately\n"
        "    // have no second device directory or copy path.\n"
        "    (void)candidateSourceToSlotRef;\n"
        "    (void)candidateSlotCursorRef;\n"
        "}\n"
    )
    source = replace_exactly_once(
        source, init_tail, p9_init_tail, "preload_p9_after_vector_buffers"
    )
    preload_path.write_text(source, encoding="utf-8")


def inject_vector_runtime(vector_path: Path) -> None:
    """Add P9-owned member methods without changing official C8 methods.

    P9 uses only C8's already allocated `outInvalidBuf_`.  The finalizer is
    invoked at the local-index ownership boundary before C8 submits its
    original MTE3 DMA, so it never consumes or re-signals TOPK_MTE3_V.  The
    disabled control path returns before any P9 seam/finalizer event action.
    """

    source = checked_file(vector_path, "official_vector_runtime_source").read_text(
        encoding="utf-8"
    )
    declaration_anchor = (
        "    __aicore__ inline void InitVecInputTensor(GlobalTensor<float> weightsGm, GlobalTensor<float> qScaleGm,\n"
        "                                              GlobalTensor<float> kScaleGm, GlobalTensor<int32_t> indiceOutGm,\n"
        "                                              GlobalTensor<int32_t> blockTableGm);\n"
    )
    declaration_insert = declaration_anchor + (
        "    __aicore__ inline void InitP9Runtime(\n"
        "        __gm__ uint8_t* runtimeEnable, __gm__ uint8_t* residentThreshold,\n"
        "        __gm__ uint8_t* residentThresholdValid, __gm__ uint8_t* reqPoolEntries,\n"
        "        __gm__ uint8_t* kvObjectMap, __gm__ uint8_t* kvObjectLens,\n"
        "        __gm__ uint8_t* persistentObjectSlotsSnapshot, __gm__ uint8_t* canonicalResidentSlots,\n"
        "        __gm__ uint8_t* candidateSourceToSlot, __gm__ uint8_t* candidateSlotCursor,\n"
        "        __gm__ uint8_t* layerResidentGetControl, __gm__ uint8_t* rowsPerEntry,\n"
        "        __gm__ uint8_t* residentBlockTable, __gm__ uint8_t* activeRowCount,\n"
        "        __gm__ uint8_t* attentionSlots, __gm__ uint8_t* terminalPostStats,\n"
        "        const QLITilingData* tilingData);\n"
        "    __aicore__ inline void P9ObserveScoreSeam(const LocalTensor<uint16_t>& outBase,\n"
        "                                                   const QLICommon::RunInfo& info, int64_t curAivS1Idx,\n"
        "                                                   int64_t curAivS1ProcNum, int64_t curS2Idx);\n"
        "    __aicore__ inline void P9PostAtScoreBoundary();\n"
        "    __aicore__ inline void P9FinalizeExactTopKRow(uint32_t row);\n"
        "    __aicore__ inline bool EnsureP9PostExecutor(uint32_t activeRow);\n"
    )
    source = replace_exactly_once(
        source, declaration_anchor, declaration_insert, "vector_p9_declarations"
    )
    field_anchor = "    LocalTensor<int32_t> outInvalidLocal_;\n"
    field_insert = field_anchor + (
        "\n"
        "    // P9 state is invocation-local and reuses only outInvalidBuf_.  It is\n"
        "    // not a private arena and does not reserve an AIV for Get.\n"
        "    bool p9RuntimeActive_ = false;\n"
        "    // A Direct-HCCP executor is constructed lazily only after an active\n"
        "    // execution row has crossed the graph-padding gate.\n"
        "    bool p9PostExecutorInitialized_ = false;\n"
        "    __gm__ uint8_t* p9RowsPerEntry_ = nullptr;\n"
        "    exact_speculative_resident::C8ExactSpeculativeResidentScoreHook p9ScoreHook_;\n"
        "    exact_speculative_resident::C8ExactSpeculativeResidentPostExecutor p9PostExecutor_;\n"
        "    exact_speculative_resident::C8ExactSpeculativeResidentScoreHookConfig p9ScoreConfig_{};\n"
        "    exact_speculative_resident::C8ExactSpeculativeResidentFinalizerConfig p9FinalizerConfig_{};\n"
    )
    source = replace_exactly_once(source, field_anchor, field_insert, "vector_p9_fields")

    init_signature = "QLIVector<QLIT>::InitVecInputTensor"
    init_body = function_body(source, init_signature)
    p9_methods = r'''

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitP9Runtime(
    __gm__ uint8_t* runtimeEnable, __gm__ uint8_t* residentThreshold,
    __gm__ uint8_t* residentThresholdValid, __gm__ uint8_t* reqPoolEntries,
    __gm__ uint8_t* kvObjectMap, __gm__ uint8_t* kvObjectLens,
    __gm__ uint8_t* persistentObjectSlotsSnapshot, __gm__ uint8_t* canonicalResidentSlots,
    __gm__ uint8_t* candidateSourceToSlot, __gm__ uint8_t* candidateSlotCursor,
    __gm__ uint8_t* layerResidentGetControl, __gm__ uint8_t* rowsPerEntry,
    __gm__ uint8_t* residentBlockTable, __gm__ uint8_t* activeRowCount,
    __gm__ uint8_t* attentionSlots, __gm__ uint8_t* terminalPostStats,
    const QLITilingData* tilingData)
{
    p9RuntimeActive_ = false;
    p9PostExecutorInitialized_ = false;
    p9RowsPerEntry_ = nullptr;
    if (tilingData == nullptr || runtimeEnable == nullptr ||
        !vllm_ascend::a5_c8_exact_speculative_resident::IsP9ExistingScratchLayoutValid(
            topkCount_ * sizeof(int32_t))) {
        return;
    }

    // runtime_enable is the only P9 control read permitted on the C8-equivalent
    // disabled path.  Do not validate or touch table/map/queue state until
    // it is one; graph placeholders remain legal under runtime_enable=0.
    GlobalTensor<int32_t> runtimeEnableGm;
    runtimeEnableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(runtimeEnable));
    if (runtimeEnableGm.GetValue(0U) != 1) {
        return;
    }

    // A captured all-padding replay is another exact-C8 control case.  Read
    // only its graph-visible row count before requiring any map/table/control
    // pointer: there is no active execution row that could legally consume
    // those P9 inputs.  Per-worker padding is gated again at the score seam.
    if (activeRowCount == nullptr || tilingData->p9RowCount == 0U) {
        return;
    }
    GlobalTensor<int32_t> activeRowCountGm;
    activeRowCountGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(activeRowCount));
    const int32_t activeRowCountValue = activeRowCountGm.GetValue(0U);
    if (activeRowCountValue <= 0 ||
        static_cast<uint32_t>(activeRowCountValue) > tilingData->p9RowCount) {
        return;
    }

    if (residentThreshold == nullptr || residentThresholdValid == nullptr ||
        reqPoolEntries == nullptr || kvObjectMap == nullptr || kvObjectLens == nullptr ||
        persistentObjectSlotsSnapshot == nullptr || canonicalResidentSlots == nullptr ||
        candidateSourceToSlot == nullptr || candidateSlotCursor == nullptr ||
        residentBlockTable == nullptr ||
        attentionSlots == nullptr ||
        tilingData->p9PoolSize == 0U || tilingData->p9SourceCapacity == 0U ||
        tilingData->p9KvObjectCapacity == 0U || tilingData->p9RowCount == 0U ||
        tilingData->p9ResidentBlockCapacity == 0U) {
        return;
    }

    p9ScoreConfig_.runtimeEnable = runtimeEnable;
    p9ScoreConfig_.residentThreshold = residentThreshold;
    p9ScoreConfig_.residentThresholdValid = residentThresholdValid;
    p9ScoreConfig_.reqPoolEntries = reqPoolEntries;
    p9ScoreConfig_.kvObjectMap = kvObjectMap;
    p9ScoreConfig_.kvObjectLens = kvObjectLens;
    p9ScoreConfig_.persistentObjectSlotsSnapshot = persistentObjectSlotsSnapshot;
    p9ScoreConfig_.canonicalResidentSlots = canonicalResidentSlots;
    p9ScoreConfig_.residentBlockTable = residentBlockTable;
    p9ScoreConfig_.activeRowCount = activeRowCount;
    p9ScoreConfig_.candidateSourceToSlot = candidateSourceToSlot;
    p9ScoreConfig_.candidateSlotCursor = candidateSlotCursor;
    p9ScoreConfig_.layerResidentGetControl = layerResidentGetControl;
    p9RowsPerEntry_ = rowsPerEntry;
    p9ScoreConfig_.rowCount = tilingData->p9RowCount;
    p9ScoreConfig_.sourceCapacity = tilingData->p9SourceCapacity;
    p9ScoreConfig_.poolSize = tilingData->p9PoolSize;
    p9ScoreConfig_.kvObjectCapacity = tilingData->p9KvObjectCapacity;
    p9ScoreConfig_.residentBlockCapacity = tilingData->p9ResidentBlockCapacity;
    // Do not read rows_per_entry or layer_resident_get_control here.  A
    // graph replay may have no active rows; the lazy executor below reads
    // transport control only after a concrete row survives active_row_count.
    p9ScoreConfig_.rowsPerEntry = 0U;
    p9ScoreConfig_.producerId = static_cast<uint32_t>(blockId_);

    __ubuf__ uint8_t* const scratch = reinterpret_cast<__ubuf__ uint8_t*>(outInvalidLocal_.GetPhyAddr());
    if (scratch == nullptr) {
        return;
    }
    __ubuf__ exact_speculative_resident::CandidateSeed* const candidateScratch =
        reinterpret_cast<__ubuf__ exact_speculative_resident::CandidateSeed*>(
        scratch + vllm_ascend::a5_c8_exact_speculative_resident::kCandidateSeedScratchOffset);
    p9ScoreHook_.Init(p9ScoreConfig_, constInfo_, candidateScratch);
    if (!p9ScoreHook_.Enabled()) {
        return;
    }

    p9FinalizerConfig_.reqPoolEntries = reqPoolEntries;
    p9FinalizerConfig_.kvObjectMap = kvObjectMap;
    p9FinalizerConfig_.kvObjectLens = kvObjectLens;
    p9FinalizerConfig_.persistentObjectSlotsSnapshot = persistentObjectSlotsSnapshot;
    p9FinalizerConfig_.canonicalResidentSlots = canonicalResidentSlots;
    p9FinalizerConfig_.residentBlockTable = residentBlockTable;
    p9FinalizerConfig_.activeRowCount = activeRowCount;
    p9FinalizerConfig_.candidateSourceToSlot = candidateSourceToSlot;
    p9FinalizerConfig_.attentionSlots = attentionSlots;
    p9FinalizerConfig_.terminalPostStats = terminalPostStats;
    p9FinalizerConfig_.layerResidentGetControl = layerResidentGetControl;
    p9FinalizerConfig_.rowCount = tilingData->p9RowCount;
    p9FinalizerConfig_.sourceCapacity = tilingData->p9SourceCapacity;
    p9FinalizerConfig_.poolSize = tilingData->p9PoolSize;
    p9FinalizerConfig_.kvObjectCapacity = tilingData->p9KvObjectCapacity;
    p9FinalizerConfig_.residentBlockCapacity = tilingData->p9ResidentBlockCapacity;
    p9FinalizerConfig_.rowsPerEntry = 0U;
    p9FinalizerConfig_.producerId = p9ScoreConfig_.producerId;
    p9RuntimeActive_ = true;
}

template <typename QLIT>
__aicore__ inline bool QLIVector<QLIT>::EnsureP9PostExecutor(uint32_t activeRow)
{
    // This is the only entry point that reads Direct-HCCP control or creates
    // an executor.  Keeping the active-row check adjacent to the first
    // control read prevents graph-padding rows (which can alias a live pool
    // row) from acquiring queues, polling CQ, or publishing descriptors.
    if (!p9RuntimeActive_ || activeRow >= p9ScoreHook_.ActiveRowCount()) {
        return false;
    }
    if (p9PostExecutorInitialized_) {
        return p9PostExecutor_.Enabled();
    }
    if (p9RowsPerEntry_ == nullptr ||
        p9ScoreConfig_.layerResidentGetControl == nullptr) {
        return false;
    }

    GlobalTensor<int32_t> rowsPerEntryGm;
    rowsPerEntryGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(p9RowsPerEntry_));
    const int32_t rowsPerEntryValue = rowsPerEntryGm.GetValue(0U);
    if (rowsPerEntryValue <= 0) {
        return false;
    }

    __ubuf__ uint8_t* const scratch =
        reinterpret_cast<__ubuf__ uint8_t*>(outInvalidLocal_.GetPhyAddr());
    if (scratch == nullptr) {
        return false;
    }
    __ubuf__ kvaf_layer_resident_get::KvafLayerResidentQueueLeaseResult* const leaseResult =
        reinterpret_cast<__ubuf__ kvaf_layer_resident_get::KvafLayerResidentQueueLeaseResult*>(
            scratch + vllm_ascend::a5_c8_exact_speculative_resident::kQueueLeaseReceiptScratchOffset);
    __ubuf__ exact_speculative_resident::CandidateSlotReservationResult* const reservationResult =
        reinterpret_cast<__ubuf__ exact_speculative_resident::CandidateSlotReservationResult*>(
            scratch + vllm_ascend::a5_c8_exact_speculative_resident::kCandidateReservationScratchOffset);
    p9ScoreHook_.SetRowsPerEntry(static_cast<uint32_t>(rowsPerEntryValue));
    p9FinalizerConfig_.rowsPerEntry = static_cast<uint32_t>(rowsPerEntryValue);
    p9PostExecutor_.Init(p9ScoreConfig_.layerResidentGetControl,
                         p9ScoreConfig_.producerId, leaseResult,
                         reservationResult, static_cast<event_t>(EVENT_ID0));
    if (!p9PostExecutor_.Enabled()) {
        return false;
    }
    p9PostExecutorInitialized_ = true;
    return true;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::P9ObserveScoreSeam(
    const LocalTensor<uint16_t>& outBase, const QLICommon::RunInfo& info,
    int64_t curAivS1Idx, int64_t curAivS1ProcNum, int64_t curS2Idx)
{
    if (!p9RuntimeActive_) {
        return;
    }
    p9ScoreHook_.CompactExactScoreTile(outBase, info, curAivS1Idx,
                                       curAivS1ProcNum, curS2Idx);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::P9PostAtScoreBoundary()
{
    if (!p9RuntimeActive_) {
        return;
    }
    const uint32_t activeRow = p9ScoreHook_.LastTileFirstActiveRow();
    // Every score-publication hook consumes exactly one observation marker.
    // Clearing it makes a future accidental unpaired post hook fail closed
    // instead of reusing an older active-row permission.
    p9ScoreHook_.ClearLastTileActiveRow();
    if (activeRow >= p9ScoreHook_.ActiveRowCount()) {
        return;
    }
    // Do not touch queue/CQ control for an active score tile that has no
    // postable receipt until an earlier active tile has initialized it.
    if (!p9PostExecutorInitialized_ && !p9ScoreHook_.HasReadyPostView()) {
        return;
    }
    if (!EnsureP9PostExecutor(activeRow)) {
        p9RuntimeActive_ = false;
        return;
    }
    p9PostExecutor_.TryRetireAtScoreBoundary();
    (void)p9PostExecutor_.TryPostReady(&p9ScoreHook_);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::P9FinalizeExactTopKRow(uint32_t row)
{
    // runtime_enable=0 reaches this fast return before P9 observes an event
    // or P9 GM state.  Active P9 runs before C8 submits its sparse-index DMA:
    // V owns indicesOutLocal_ after the unchanged TOPK_V_MTE3 wait, and C8
    // retains its original MTE3->V token after this method returns.
    if (!p9RuntimeActive_ || row >= p9ScoreHook_.ActiveRowCount()) {
        return;
    }
    if (!EnsureP9PostExecutor(row)) {
        p9RuntimeActive_ = false;
        return;
    }
    exact_speculative_resident::C8ExactSpeculativeResidentFinalizer finalizer;
    finalizer.Init(p9FinalizerConfig_, outInvalidLocal_, static_cast<event_t>(EVENT_ID0));
    const bool terminal = finalizer.Enabled() && finalizer.FinalizeExactTopKRow(
        row, indicesOutLocal_, &p9ScoreHook_, &p9PostExecutor_);
    if (!terminal) {
        p9RuntimeActive_ = false;
    }
}
'''
    source = replace_exactly_once(
        source, init_body, init_body + p9_methods, "vector_p9_method_definitions"
    )
    vector_path.write_text(source, encoding="utf-8")


def inject_tiling_abi(tiling_header_path: Path, tiling_source_path: Path) -> None:
    """Carry map dimensions through the existing official C8 tiling record.

    P9 does not add host-visible scalar tensors merely to describe existing
    tensor shapes.  The host validates the relevant P9 input shapes once and
    stores their three bounded dimensions plus flattened C8 row count in the
    source-derived tiling.  The device then uses those values for every map
    bound check.
    """

    header = checked_file(tiling_header_path, "official_tiling_header").read_text(
        encoding="utf-8"
    )
    header = replace_exactly_once(
        header,
        "constexpr uint32_t METADATA_INDEX = 8;\n",
        "constexpr uint32_t METADATA_INDEX = 8;\n"
        "// P9 keeps the official C8 0..8 prefix intact.\n"
        "constexpr uint32_t P9_RUNTIME_ENABLE_INDEX = 9;\n"
        "constexpr uint32_t P9_RESIDENT_THRESHOLD_INDEX = 10;\n"
        "constexpr uint32_t P9_RESIDENT_THRESHOLD_VALID_INDEX = 11;\n"
        "constexpr uint32_t P9_REQ_POOL_ENTRIES_INDEX = 12;\n"
        "constexpr uint32_t P9_KV_OBJECT_MAP_INDEX = 13;\n"
        "constexpr uint32_t P9_KV_OBJECT_LENS_INDEX = 14;\n"
        "constexpr uint32_t P9_PERSISTENT_OBJECT_SLOTS_INDEX = 15;\n"
        "constexpr uint32_t P9_CANONICAL_RESIDENT_SLOTS_INDEX = 16;\n"
        "constexpr uint32_t P9_CANDIDATE_SOURCE_TO_SLOT_INDEX = 17;\n"
        "constexpr uint32_t P9_CANDIDATE_SLOT_CURSOR_INDEX = 18;\n"
        "constexpr uint32_t P9_LAYER_RESIDENT_GET_CONTROL_INDEX = 19;\n"
        "constexpr uint32_t P9_ROWS_PER_ENTRY_INDEX = 20;\n"
        "constexpr uint32_t P9_RESIDENT_BLOCK_TABLE_INDEX = 21;\n"
        "constexpr uint32_t P9_ACTIVE_ROW_COUNT_INDEX = 22;\n",
        "tiling_p9_input_indices",
    )
    header = replace_exactly_once(
        header,
        "TILING_DATA_FIELD_DEF(int64_t, scaleStride)\n",
        "TILING_DATA_FIELD_DEF(int64_t, scaleStride)\n"
        "// Dimensions of existing P9 graph inputs; no hidden host callback.\n"
        "TILING_DATA_FIELD_DEF(uint32_t, p9RowCount)\n"
        "TILING_DATA_FIELD_DEF(uint32_t, p9PoolSize)\n"
        "TILING_DATA_FIELD_DEF(uint32_t, p9SourceCapacity)\n"
        "TILING_DATA_FIELD_DEF(uint32_t, p9KvObjectCapacity)\n"
        "TILING_DATA_FIELD_DEF(uint32_t, p9ResidentBlockCapacity)\n",
        "tiling_p9_fields",
    )
    tiling_header_path.write_text(header, encoding="utf-8")

    source = checked_file(tiling_source_path, "official_tiling_source").read_text(
        encoding="utf-8"
    )
    source = replace_exactly_once(
        source,
        '#include "vllm_quant_lightning_indexer_tiling.h"\n',
        '#include "vllm_quant_lightning_indexer_tiling.h"\n#include <climits>\n',
        "tiling_p9_climits_include",
    )
    tiling_anchor = "    tilingData_.set_scaleStride(tilingInfo->scaleStride);\n"
    tiling_block = tiling_anchor + r'''    // P9 dimensions are derived from graph tensor shapes, not from a hidden
    // control tensor.  This keeps the 23-input ABI explicit while preserving
    // device-side bounds for req_pool_entries -> kv_object_map -> snapshot.
    const auto* p9SparseStorage = context_->GetOutputShape(0);
    const auto* p9RuntimeEnableStorage = context_->GetInputShape(P9_RUNTIME_ENABLE_INDEX);
    const auto* p9ThresholdStorage = context_->GetInputShape(P9_RESIDENT_THRESHOLD_INDEX);
    const auto* p9ThresholdValidStorage = context_->GetInputShape(P9_RESIDENT_THRESHOLD_VALID_INDEX);
    const auto* p9ReqStorage = context_->GetInputShape(P9_REQ_POOL_ENTRIES_INDEX);
    const auto* p9MapStorage = context_->GetInputShape(P9_KV_OBJECT_MAP_INDEX);
    const auto* p9LensStorage = context_->GetInputShape(P9_KV_OBJECT_LENS_INDEX);
    const auto* p9SnapshotStorage = context_->GetInputShape(P9_PERSISTENT_OBJECT_SLOTS_INDEX);
    const auto* p9SourceToSlotStorage = context_->GetInputShape(P9_CANDIDATE_SOURCE_TO_SLOT_INDEX);
    const auto* p9CursorStorage = context_->GetInputShape(P9_CANDIDATE_SLOT_CURSOR_INDEX);
    const auto* p9CanonicalSlotsStorage = context_->GetInputShape(P9_CANONICAL_RESIDENT_SLOTS_INDEX);
    const auto* p9RowsStorage = context_->GetInputShape(P9_ROWS_PER_ENTRY_INDEX);
    const auto* p9ControlStorage = context_->GetInputShape(P9_LAYER_RESIDENT_GET_CONTROL_INDEX);
    const auto* p9ResidentBlockTableStorage = context_->GetInputShape(P9_RESIDENT_BLOCK_TABLE_INDEX);
    const auto* p9ActiveRowCountStorage = context_->GetInputShape(P9_ACTIVE_ROW_COUNT_INDEX);
    if (p9SparseStorage == nullptr || p9RuntimeEnableStorage == nullptr ||
        p9ThresholdStorage == nullptr || p9ThresholdValidStorage == nullptr ||
        p9ReqStorage == nullptr || p9MapStorage == nullptr || p9LensStorage == nullptr ||
        p9SnapshotStorage == nullptr || p9SourceToSlotStorage == nullptr ||
        p9CursorStorage == nullptr || p9CanonicalSlotsStorage == nullptr ||
        p9RowsStorage == nullptr || p9ControlStorage == nullptr ||
        p9ResidentBlockTableStorage == nullptr || p9ActiveRowCountStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape p9Sparse = p9SparseStorage->GetStorageShape();
    const gert::Shape p9RuntimeEnable = p9RuntimeEnableStorage->GetStorageShape();
    const gert::Shape p9Threshold = p9ThresholdStorage->GetStorageShape();
    const gert::Shape p9ThresholdValid = p9ThresholdValidStorage->GetStorageShape();
    const gert::Shape p9Req = p9ReqStorage->GetStorageShape();
    const gert::Shape p9Map = p9MapStorage->GetStorageShape();
    const gert::Shape p9Lens = p9LensStorage->GetStorageShape();
    const gert::Shape p9Snapshot = p9SnapshotStorage->GetStorageShape();
    const gert::Shape p9SourceToSlot = p9SourceToSlotStorage->GetStorageShape();
    const gert::Shape p9Cursor = p9CursorStorage->GetStorageShape();
    const gert::Shape p9CanonicalSlots = p9CanonicalSlotsStorage->GetStorageShape();
    const gert::Shape p9Rows = p9RowsStorage->GetStorageShape();
    const gert::Shape p9Control = p9ControlStorage->GetStorageShape();
    const gert::Shape p9ResidentBlockTable = p9ResidentBlockTableStorage->GetStorageShape();
    const gert::Shape p9ActiveRowCount = p9ActiveRowCountStorage->GetStorageShape();
    int64_t p9RowCount = 1;
    if (p9Sparse.GetDimNum() < 1 || p9Sparse.GetDim(p9Sparse.GetDimNum() - 1) != 2048 ||
        p9Map.GetDimNum() != 2 || p9Map.GetDim(0) <= 0 || p9Map.GetDim(1) <= 0 ||
        p9Snapshot.GetDimNum() != 2 || p9Snapshot.GetDim(0) != p9Map.GetDim(0) ||
        p9Snapshot.GetDim(1) <= 0 || p9CanonicalSlots.GetDimNum() != 2 ||
        p9CanonicalSlots.GetDim(0) != p9Map.GetDim(0) ||
        p9CanonicalSlots.GetDim(1) != 2048 ||
        p9Rows.GetDimNum() != 1 || p9Rows.GetDim(0) != 1 ||
        p9Control.GetDimNum() != 1 || p9Control.GetDim(0) != 16 ||
        p9ActiveRowCount.GetDimNum() != 1 || p9ActiveRowCount.GetDim(0) != 1) {
        return ge::GRAPH_FAILED;
    }
    for (size_t dim = 0; dim + 1 < p9Sparse.GetDimNum(); ++dim) {
        const int64_t value = p9Sparse.GetDim(dim);
        if (value <= 0 || p9RowCount > static_cast<int64_t>(UINT32_MAX) / value) {
            return ge::GRAPH_FAILED;
        }
        p9RowCount *= value;
    }
    if (p9RuntimeEnable.GetDimNum() != 1 || p9RuntimeEnable.GetDim(0) != 1 ||
        p9Threshold.GetDimNum() != 1 || p9Threshold.GetDim(0) != p9RowCount ||
        p9ThresholdValid.GetDimNum() != 1 || p9ThresholdValid.GetDim(0) != p9RowCount ||
        p9Req.GetDimNum() != 1 || p9Req.GetDim(0) != p9RowCount ||
        p9Lens.GetDimNum() != 1 || p9Lens.GetDim(0) != p9RowCount ||
        p9Cursor.GetDimNum() != 1 || p9Cursor.GetDim(0) != p9RowCount ||
        p9SourceToSlot.GetDimNum() != 2 || p9SourceToSlot.GetDim(0) != p9RowCount ||
        p9SourceToSlot.GetDim(1) != p9Map.GetDim(1) ||
        p9ResidentBlockTable.GetDimNum() != 2 ||
        p9ResidentBlockTable.GetDim(0) != p9RowCount ||
        p9ResidentBlockTable.GetDim(1) <= 0 ||
        p9Map.GetDim(0) > static_cast<int64_t>(UINT32_MAX) ||
        p9Map.GetDim(1) > static_cast<int64_t>(UINT32_MAX) ||
        p9Snapshot.GetDim(1) > static_cast<int64_t>(UINT32_MAX) ||
        p9ResidentBlockTable.GetDim(1) > static_cast<int64_t>(UINT32_MAX)) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.set_p9RowCount(static_cast<uint32_t>(p9RowCount));
    tilingData_.set_p9PoolSize(static_cast<uint32_t>(p9Map.GetDim(0)));
    tilingData_.set_p9SourceCapacity(static_cast<uint32_t>(p9Map.GetDim(1)));
    tilingData_.set_p9KvObjectCapacity(static_cast<uint32_t>(p9Snapshot.GetDim(1)));
    tilingData_.set_p9ResidentBlockCapacity(
        static_cast<uint32_t>(p9ResidentBlockTable.GetDim(1)));
'''
    source = replace_exactly_once(source, tiling_anchor, tiling_block, "tiling_p9_dimensions")
    tiling_source_path.write_text(source, encoding="utf-8")


def p9_input_chain(receiver: str) -> str:
    """Return the host OpDef/AICore P9 input chain in ABI order."""

    names_and_types = (
        ("runtime_enable", "ge::DT_INT32"),
        ("resident_threshold", "ge::DT_INT32"),
        ("resident_threshold_valid", "ge::DT_INT32"),
        ("req_pool_entries", "ge::DT_INT32"),
        ("kv_object_map", "ge::DT_INT32"),
        ("kv_object_lens", "ge::DT_INT32"),
        ("persistent_object_slots_snapshot", "ge::DT_INT32"),
        ("canonical_resident_slots", "ge::DT_INT32"),
        ("candidate_source_to_slot", "ge::DT_INT32"),
        ("candidate_slot_cursor", "ge::DT_INT32"),
        ("layer_resident_get_control", "ge::DT_INT64"),
        ("rows_per_entry", "ge::DT_INT32"),
        ("resident_block_table", "ge::DT_INT32"),
        ("active_row_count", "ge::DT_INT32"),
    )
    return "".join(
        f'        {receiver}Input("{name}")\n'
        "            .ParamType(REQUIRED)\n"
        f"            .DataType({{{dtype}}})\n"
        "            .Format({ge::FORMAT_ND})\n"
        "            .AutoContiguous();\n"
        for name, dtype in names_and_types
    )


def p9_output_chain(receiver: str) -> str:
    return (
        f'        {receiver}Output("attention_slots").ParamType(REQUIRED).DataType({{ge::DT_INT32}}).Format({{ge::FORMAT_ND}});\n'
        f'        {receiver}Output("terminal_post_stats").ParamType(REQUIRED).DataType({{ge::DT_INT32}}).Format({{ge::FORMAT_ND}});\n'
        f'        {receiver}Output("candidate_source_to_slot_ref").ParamType(REQUIRED).DataType({{ge::DT_INT32}}).Format({{ge::FORMAT_ND}});\n'
        f'        {receiver}Output("candidate_slot_cursor_ref").ParamType(REQUIRED).DataType({{ge::DT_INT32}}).Format({{ge::FORMAT_ND}});\n'
    )


def inject_host_abi(host_def_path: Path, infer_shape_path: Path) -> None:
    """Extend source-derived OpDef and infer-shape source with the P9 ABI."""

    source = checked_file(host_def_path, "official_host_def").read_text(encoding="utf-8")
    metadata_base = (
        '        this->Input("metadata")\n'
        "            .ParamType(OPTIONAL)\n"
        "            .DataType({ge::DT_INT32})\n"
        "            .Format({ge::FORMAT_ND})\n"
        "            .AutoContiguous();\n"
    )
    source = replace_exactly_once(
        source, metadata_base, metadata_base + "        // P9 active ABI inputs 9..22.\n" + p9_input_chain("this->"),
        "host_def_p9_base_inputs",
    )
    sparse_values_base = (
        '        this->Output("sparse_values").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});\n'
    )
    source = replace_exactly_once(
        source, sparse_values_base, sparse_values_base + "        // Outputs 4/5 alias inputs 17/18; no duplicate resident payload output exists.\n" + p9_output_chain("this->"),
        "host_def_p9_base_outputs",
    )
    metadata_950 = (
        '        aicore_config_950.Input("metadata")\n'
        "            .ParamType(OPTIONAL)\n"
        "            .DataType({ge::DT_INT32})\n"
        "            .Format({ge::FORMAT_ND})\n"
        "            .AutoContiguous();\n"
    )
    source = replace_exactly_once(
        source, metadata_950, metadata_950 + p9_input_chain("aicore_config_950."),
        "host_def_p9_950_inputs",
    )
    sparse_values_950 = (
        '        aicore_config_950.Output("sparse_values").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});\n'
    )
    source = replace_exactly_once(
        source, sparse_values_950, sparse_values_950 + p9_output_chain("aicore_config_950."),
        "host_def_p9_950_outputs",
    )
    # The active P9 kernel source is arch35/Ascend950-only.  Leaving the
    # official 910B/910_93 AICore configs in the cloned OpDef would advertise
    # the expanded 23/6 schema for an unchanged 9/2 device entrypoint.  Do
    # not make that a late runtime dispatch failure: P9 has one supported
    # target configuration, Ascend950.
    unsupported_config_block = (
        "        OpAICoreConfig aicore_config;\n"
        "        aicore_config.DynamicCompileStaticFlag(true)\n"
        "            .DynamicFormatFlag(true)\n"
        "            .DynamicRankSupportFlag(true)\n"
        "            .DynamicShapeSupportFlag(true)\n"
        "            .NeedCheckSupportFlag(false)\n"
        "            .PrecisionReduceFlag(true);\n"
        "        this->AICore().AddConfig(\"ascend910b\", aicore_config);\n"
        "        this->AICore().AddConfig(\"ascend910_93\", aicore_config);\n\n"
    )
    source = replace_exactly_once(
        source,
        unsupported_config_block,
        "        // P9 is source-derived from the Ascend950 C8 path only.\n"
        "        // Do not advertise its 23/6 ABI for legacy 9/2 kernels.\n\n",
        "host_def_remove_unsupported_aicore_configs",
    )
    host_def_path.write_text(source, encoding="utf-8")

    infer = checked_file(infer_shape_path, "official_infer_shape").read_text(encoding="utf-8")
    infer = replace_exactly_once(
        infer,
        "#include <register/op_impl_registry.h>\n",
        "#include <register/op_impl_registry.h>\n\n#include <climits>\n",
        "infer_p9_climits_include",
    )
    output_anchor = (
        "    gert::Shape *sparseValuesShape = context->GetOutputShape(1);\n"
    )
    output_insert = output_anchor + r'''    gert::Shape *attentionSlotsShape = context->GetOutputShape(2);
    gert::Shape *terminalPostStatsShape = context->GetOutputShape(3);
    gert::Shape *candidateSourceToSlotRefShape = context->GetOutputShape(4);
    gert::Shape *candidateSlotCursorRefShape = context->GetOutputShape(5);
    const gert::Shape *candidateSourceToSlotShape = context->GetInputShape(17);
    const gert::Shape *candidateSlotCursorShape = context->GetInputShape(18);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseValuesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionSlotsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, terminalPostStatsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, candidateSourceToSlotRefShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, candidateSlotCursorRefShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, candidateSourceToSlotShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, candidateSlotCursorShape);
'''
    infer = replace_exactly_once(infer, output_anchor, output_insert, "infer_p9_output_handles")
    infer_anchor = "    OP_LOGD(context->GetNodeName(), \"VllmQuantLightningIndexer InferShape end.\");\n"
    infer_block = r'''    int64_t p9RowCount = 1;
    for (size_t dim = 0; dim + 1 < sparseIndicesShape->GetDimNum(); ++dim) {
        const int64_t value = sparseIndicesShape->GetDim(dim);
        OP_CHECK_IF(value <= 0 || p9RowCount > INT64_MAX / value,
                    OP_LOGE(context, "P9 sparse row shape is invalid."),
                    return GRAPH_FAILED);
        p9RowCount *= value;
    }
    attentionSlotsShape->SetDimNum(2);
    attentionSlotsShape->SetDim(0, p9RowCount);
    attentionSlotsShape->SetDim(1, *sparse_count);
    terminalPostStatsShape->SetDimNum(2);
    terminalPostStatsShape->SetDim(0, p9RowCount);
    terminalPostStatsShape->SetDim(1, 7);
    *candidateSourceToSlotRefShape = *candidateSourceToSlotShape;
    *candidateSlotCursorRefShape = *candidateSlotCursorShape;
    // P9 outputs 4/5 are aliases by ABI; the runtime bridge passes exactly
    // inputs 17/18 at those output positions and the kernel owns no copy.
'''
    infer = replace_exactly_once(infer, infer_anchor, infer_block + infer_anchor, "infer_p9_shapes")
    dtype_anchor = "    context->SetOutputDataType(0, outputType);\n"
    dtype_insert = dtype_anchor + (
        "    context->SetOutputDataType(2, ge::DT_INT32);\n"
        "    context->SetOutputDataType(3, ge::DT_INT32);\n"
        "    context->SetOutputDataType(4, ge::DT_INT32);\n"
        "    context->SetOutputDataType(5, ge::DT_INT32);\n"
    )
    infer = replace_exactly_once(infer, dtype_anchor, dtype_insert, "infer_p9_dtypes")
    infer_shape_path.write_text(infer, encoding="utf-8")


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
        "        this, outBase, info, curAivS1Idx, curAivS1ProcNum, curS2Idx);\n"
    )
    source = source[: match.end(1)] + observation + source[match.end(1) :]

    # Candidate receipts become postable only after C8 has issued its normal
    # score-GM publication and the matching MTE3->V signal.  This adds no
    # score-buffer wait/ownership operation: P9 merely asks its dynamic
    # executor to retire a prior CQ and submit a completed bounded receipt.
    score_publication_anchor = (
        "    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, copyOutParams);\n"
        "    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);"
    )
    source = replace_exactly_once(
        source,
        score_publication_anchor,
        score_publication_anchor
        + "\n    A5_C8_EXACT_SPECULATIVE_RESIDENT_POST_AFTER_SCORE_PUBLICATION(this);",
        "score_publication_post_boundary",
    )

    # The source ordering is the substantive seam proof: reduction -> marker
    # -> official V/MTE hand-off -> score GM publication -> untouched Top-K.
    reduction_index = source.index("vector1::BatchMulWeightAndReduceSum")
    marker_index = source.index(SEAM_MARKER, reduction_index)
    publication_index = source.index("DataCopyPad(scoreGm", marker_index)
    topk_index = source.index(topk_signature)
    post_index = source.index(POST_MARKER, publication_index)
    if not (
        reduction_index < marker_index < publication_index < post_index < topk_index
    ):
        fail("score_seam_source_order_invalid")

    source = inject_stage_b_finalizer_tail(source, topk_before)
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
    if vector.count(POST_MARKER) != 1:
        fail("staged_score_post_marker_count_invalid")
    for member in (
        "InitP9Runtime(",
        "P9ObserveScoreSeam(",
        "P9PostAtScoreBoundary()",
        "P9FinalizeExactTopKRow(",
    ):
        if member not in vector:
            fail(f"staged_vector_p9_member_missing:value={member}")
    topk = function_body(vector, "QLIVector<QLIT>::ProcessTopK")
    if topk.count(FINALIZER_MARKER) != 2:
        fail("staged_stage_b_finalizer_marker_count_invalid")
    if sha256_text(strip_stage_b_finalizer_tails(topk)) != topk_hash:
        fail("staged_process_topk_digest_changed_outside_finalizer")

    ops = json.loads(
        checked_file(staged_root / OPS_RELATIVE, "staged_ops_json").read_text(
            encoding="utf-8"
        )
    )
    if len(ops) != 1 or ops[0].get("op") != P9_OPERATOR:
        fail("staged_ops_json_operator_invalid")
    expected_inputs = (
        ("query", "int8"),
        ("key", "int8"),
        ("weights", "float"),
        ("query_dequant_scale", "float"),
        ("key_dequant_scale", "float"),
        ("actual_seq_lengths_query", "int32"),
        ("actual_seq_lengths_key", "int32"),
        ("block_table", "int32"),
        ("metadata", "int32"),
        ("runtime_enable", "int32"),
        ("resident_threshold", "int32"),
        ("resident_threshold_valid", "int32"),
        ("req_pool_entries", "int32"),
        ("kv_object_map", "int32"),
        ("kv_object_lens", "int32"),
        ("persistent_object_slots_snapshot", "int32"),
        ("canonical_resident_slots", "int32"),
        ("candidate_source_to_slot", "int32"),
        ("candidate_slot_cursor", "int32"),
        ("layer_resident_get_control", "int64"),
        ("rows_per_entry", "int32"),
        ("resident_block_table", "int32"),
        ("active_row_count", "int32"),
    )
    expected_outputs = (
        ("sparse_indices", "int32"),
        ("sparse_values", "float"),
        ("attention_slots", "int32"),
        ("terminal_post_stats", "int32"),
        ("candidate_source_to_slot_ref", "int32"),
        ("candidate_slot_cursor_ref", "int32"),
    )
    actual_inputs = tuple(
        (entry.get("name"), entry.get("type", [None])[0])
        for entry in ops[0].get("input_desc", [])
    )
    actual_outputs = tuple(
        (entry.get("name"), entry.get("type", [None])[0])
        for entry in ops[0].get("output_desc", [])
    )
    if actual_inputs != expected_inputs or actual_outputs != expected_outputs:
        fail(
            "staged_ops_json_abi_invalid:"
            f"inputs={actual_inputs}:outputs={actual_outputs}"
        )
    if len(actual_inputs) != 23 or len(actual_outputs) != 6:
        fail("staged_ops_json_23_6_abi_invalid")
    if "resident_packed_output" in (staged_root / OPS_RELATIVE).read_text(
        encoding="utf-8"
    ):
        fail("staged_ops_json_redundant_resident_output")

    # Mechanical generated-source checks.  The Python applier is the only
    # safe place to catch a plausible-looking 23/6 JSON paired with a C++
    # receiver typo or an early P9 scratch borrow, before a remote CANN build
    # consumes tens of minutes.
    wrapper = checked_file(
        staged_root / "op_kernel/a5_c8_exact_speculative_resident_get.cpp",
        "staged_kernel_wrapper",
    ).read_text(encoding="utf-8")
    preload = checked_file(
        staged_root / PRELOAD_RELATIVE, "staged_preload_source"
    ).read_text(encoding="utf-8")
    host_def = checked_file(
        staged_root / "op_host/a5_c8_exact_speculative_resident_get_def.cpp",
        "staged_host_def",
    ).read_text(encoding="utf-8")
    tiling = checked_file(
        staged_root / "op_host/a5_c8_exact_speculative_resident_get_tiling.cpp",
        "staged_tiling_source",
    ).read_text(encoding="utf-8")
    if "this.Input(" in host_def or "this.Output(" in host_def:
        fail("staged_host_def_invalid_this_receiver")
    for token in (
        'this->Input("runtime_enable")',
        'this->Output("attention_slots")',
        'aicore_config_950.Input("runtime_enable")',
        'aicore_config_950.Output("attention_slots")',
    ):
        if token not in host_def:
            fail(f"staged_host_def_p9_receiver_missing:value={token}")
    if 'AddConfig("ascend910b"' in host_def or 'AddConfig("ascend910_93"' in host_def:
        fail("staged_host_def_advertises_legacy_9_2_kernel")
    for token in (
        "runtimeEnable", "candidateSourceToSlotRef", "candidateSlotCursorRef",
        "attentionSlots", "terminalPostStats", "canonicalResidentSlots",
        "residentBlockTable", "activeRowCount",
    ):
        if token not in wrapper:
            fail(f"staged_wrapper_p9_argument_missing:value={token}")
    # Match the actual Init definition, not InitTilingData/InitBuffers.
    preload_init = function_body(preload, "QLIPreload<QLIT>::Init(__gm__")
    p9_init_position = preload_init.find("vectorService.InitP9Runtime(")
    buffer_position = preload_init.find("InitBuffers();")
    if p9_init_position < 0 or buffer_position < 0 or p9_init_position <= buffer_position:
        fail("staged_preload_p9_init_not_after_buffers")
    if "if ASCEND_IS_AIV {\n        vectorService.InitP9Runtime(" not in preload_init:
        fail("staged_preload_p9_init_not_aiv_guarded")
    p9_runtime = function_body(vector, "QLIVector<QLIT>::InitP9Runtime")
    runtime_gate = p9_runtime.find("runtimeEnableGm.GetValue(0U)")
    scratch_borrow = p9_runtime.find("outInvalidLocal_.GetPhyAddr()")
    if runtime_gate < 0 or scratch_borrow < 0 or runtime_gate >= scratch_borrow:
        fail("staged_runtime_enable_not_before_scratch_borrow")
    for forbidden in ("canonicalResidentLayout", "CanonicalResidentLayout",
                      "IsCanonicalLayout"):
        if forbidden in p9_runtime:
            fail(f"staged_runtime_legacy_affine_layout:value={forbidden}")
    active_gate = p9_runtime.find("activeRowCount == nullptr")
    if active_gate < 0 or active_gate <= runtime_gate:
        fail("staged_runtime_active_row_gate_missing_or_before_enable")
    for token in (
        "P9_KV_OBJECT_LENS_INDEX",
        "P9_CANONICAL_RESIDENT_SLOTS_INDEX",
        "P9_RESIDENT_BLOCK_TABLE_INDEX",
        "P9_ACTIVE_ROW_COUNT_INDEX",
        "p9RuntimeEnableStorage == nullptr",
        "p9ThresholdStorage == nullptr",
        "p9ThresholdValidStorage == nullptr",
        "p9LensStorage == nullptr",
        "p9RuntimeEnable.GetDim(0) != 1",
        "p9CanonicalSlots.GetDim(1) != 2048",
        "p9ResidentBlockTable.GetDim(0) != p9RowCount",
        "p9ActiveRowCount.GetDim(0) != 1",
    ):
        if token not in tiling:
            fail(f"staged_tiling_p9_validation_missing:value={token}")
    for relative in (
        "op_host/err/ops_err.h",
        "op_host/log/log.h",
        "op_host/a5_c8_exact_speculative_resident_get_def.cpp",
        "op_host/a5_c8_exact_speculative_resident_get_infershape.cpp",
        "op_host/a5_c8_exact_speculative_resident_get_tiling.cpp",
        "op_host/a5_c8_exact_speculative_resident_get_tiling.h",
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
    # Keep this ordering atomic: every signature/host/tiling/vector change is
    # made before the two active score/Top-K seams are inserted.  A missing
    # anchor therefore aborts staging rather than producing a 23/6 JSON paired
    # with an old 9/2 kernel path.
    inject_kernel_wrapper_abi(staged_root / KERNEL_WRAPPER_RELATIVE)
    inject_preload_abi(staged_root / PRELOAD_RELATIVE)
    inject_vector_runtime(staged_root / VECTOR_RELATIVE)
    inject_tiling_abi(
        staged_root / TILING_HEADER_RELATIVE,
        staged_root / TILING_SOURCE_RELATIVE,
    )
    inject_host_abi(
        staged_root / HOST_DEF_RELATIVE,
        staged_root / INFER_SHAPE_RELATIVE,
    )
    topk_hash, vector_hash = inject_score_seam(staged_root / VECTOR_RELATIVE)
    rename_source_files(staged_root)
    replace_operator_identifiers(staged_root)
    validate_output(staged_root, topk_hash)

    print(
        "P9_OVERLAY_APPLY_PASS "
        f"operator={P9_OPERATOR} seam=active_abi "
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
