#!/usr/bin/env python3
"""Offline atomic-overlay gate for the active P9 direct-resident operator.

This suite intentionally runs without CANN.  It takes a fresh copy of the
official C8 source, applies the checked-in overlay, then proves three things
before an expensive OPP build is allowed:

* the public schema is exactly P9's 23-input / 6-output ABI;
* the official score and Top-K implementation are preserved byte-for-byte
  except for anchored P9 calls at legal boundaries; and
* all P9 inputs reach the source-derived wrapper, preload, and vector path in
  one atomic transform.  A partial 9/2 closure is not a valid P9 artifact.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest


OPS_ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = OPS_ROOT / "a5_c8_exact_speculative_resident_get"
OVERLAY_ROOT = TARGET_ROOT / "source_overlay"
OFFICIAL_C8_ROOT = OPS_ROOT.parent / "vllm_quant_lightning_indexer"
APPLIER_PATH = OVERLAY_ROOT / "apply_p9_overlay.py"

OFFICIAL_INPUTS = [
    ("query", "int8"),
    ("key", "int8"),
    ("weights", "float"),
    ("query_dequant_scale", "float"),
    ("key_dequant_scale", "float"),
    ("actual_seq_lengths_query", "int32"),
    ("actual_seq_lengths_key", "int32"),
    ("block_table", "int32"),
    ("metadata", "int32"),
]
P9_INPUTS = [
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
]
OUTPUTS = [
    ("sparse_indices", "int32"),
    ("sparse_values", "float"),
    ("attention_slots", "int32"),
    ("terminal_post_stats", "int32"),
    ("candidate_source_to_slot_ref", "int32"),
    ("candidate_slot_cursor_ref", "int32"),
]


def load_applier():
    spec = importlib.util.spec_from_file_location("p9_overlay_applier", APPLIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_digest(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return hashlib.sha256(source[start : index + 1].encode()).hexdigest()
    raise AssertionError("unbalanced function body")


def tensor_schema(descriptors: list[dict[str, object]]) -> list[tuple[str, str]]:
    return [(entry["name"], entry["type"][0]) for entry in descriptors]


class C8ExactSpeculativeResidentOverlayTest(unittest.TestCase):
    def test_manifest_is_a_small_closed_overlay_input_set(self) -> None:
        entries = [
            line.strip()
            for line in (OVERLAY_ROOT / "P9_OVERLAY_MANIFEST").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(
            entries,
            [
                "apply_p9_overlay.py",
                "ops.json",
                "op_host/a5_c8_exact_speculative_resident_host_compat.h",
                "op_kernel/arch35/a5_c8_exact_speculative_resident_score_seam.h",
            ],
        )
        for relative in entries:
            self.assertTrue((OVERLAY_ROOT / relative).is_file(), relative)

    def test_ops_schema_is_exact_active_23_input_6_output_abi(self) -> None:
        ops = json.loads((OVERLAY_ROOT / "ops.json").read_text(encoding="utf-8"))
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["op"], "A5C8ExactSpeculativeResidentGet")
        self.assertEqual(tensor_schema(ops[0]["input_desc"]), OFFICIAL_INPUTS + P9_INPUTS)
        self.assertEqual(tensor_schema(ops[0]["output_desc"]), OUTPUTS)
        self.assertEqual(len(ops[0]["input_desc"]), 23)
        self.assertEqual(len(ops[0]["output_desc"]), 6)
        self.assertNotIn(
            "resident_packed_output",
            [entry["name"] for entry in ops[0]["input_desc"] + ops[0]["output_desc"]],
        )

    def test_fresh_clone_receives_atomic_active_overlay_without_c8_math_delta(self) -> None:
        applier = load_applier()
        original_vector = (
            OFFICIAL_C8_ROOT
            / "op_kernel/arch35/quant_lightning_indexer_service_vector.h"
        ).read_text(encoding="utf-8")
        original_topk = function_digest(original_vector, "QLIVector<QLIT>::ProcessTopK")

        with tempfile.TemporaryDirectory(prefix="p9-overlay-") as temporary:
            staged_root = Path(temporary) / "vllm_quant_lightning_indexer"
            shutil.copytree(OFFICIAL_C8_ROOT, staged_root)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = applier.main(
                    [
                        "--staged-c8-root",
                        str(staged_root),
                        "--overlay-root",
                        str(OVERLAY_ROOT),
                        "--manifest",
                        str(OVERLAY_ROOT / "P9_OVERLAY_MANIFEST"),
                        "--operator",
                        "A5C8ExactSpeculativeResidentGet",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertIn("P9_OVERLAY_APPLY_PASS", output.getvalue())
            self.assertIn("seam=active_abi", output.getvalue())
            self.assertNotIn("compile_disabled", output.getvalue())

            vector_path = (
                staged_root
                / "op_kernel/arch35/quant_lightning_indexer_service_vector.h"
            )
            staged_vector = vector_path.read_text(encoding="utf-8")
            staged_topk = applier.function_body(
                staged_vector, "QLIVector<QLIT>::ProcessTopK"
            )
            # Removing only the two anchored tail calls must recover the
            # original C8 Top-K function exactly.  In particular, P9 cannot
            # rewrite score reduction, candidate ordering, or index DMA.
            self.assertEqual(
                hashlib.sha256(
                    applier.strip_stage_b_finalizer_tails(staged_topk).encode()
                ).hexdigest(),
                original_topk,
            )
            self.assertEqual(
                staged_topk.count(
                    "A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER_TAIL("
                ),
                2,
            )
            invalid_tail = staged_topk.index(applier._TOPK_INVALID_FINALIZER_TAIL)
            normal_tail = staged_topk.index(applier._TOPK_NORMAL_FINALIZER_TAIL)
            # Stage B consumes the completed local Top-K indices at the legal
            # V->MTE3 boundary.  It must run before C8 starts its index DMA;
            # P9 must neither wait on nor relay TOPK_MTE3_V a second time.
            for tail_offset in (invalid_tail, normal_tail):
                self.assertTrue(
                    staged_topk[:tail_offset]
                    .rstrip()
                    .endswith("WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);")
                )
                after_tail = staged_topk[tail_offset:]
                writeback = after_tail.index("AscendC::DataCopyPad(indiceOutGm")
                completion = after_tail.index(
                    "SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);"
                )
                self.assertGreater(writeback, 0)
                self.assertGreater(completion, writeback)

            reduction = staged_vector.index("vector1::BatchMulWeightAndReduceSum")
            observation = staged_vector.index(
                "A5_C8_EXACT_SPECULATIVE_RESIDENT_OBSERVE_SCORE_SEAM", reduction
            )
            publication = staged_vector.index("DataCopyPad(scoreGm", observation)
            post = staged_vector.index(
                "A5_C8_EXACT_SPECULATIVE_RESIDENT_POST_AFTER_SCORE_PUBLICATION",
                publication,
            )
            self.assertLess(reduction, observation)
            self.assertLess(observation, publication)
            self.assertLess(publication, post)
            self.assertIn(
                "A5_C8_EXACT_SPECULATIVE_RESIDENT_OBSERVE_SCORE_SEAM(\n        this,",
                staged_vector,
            )
            self.assertIn(
                "A5_C8_EXACT_SPECULATIVE_RESIDENT_POST_AFTER_SCORE_PUBLICATION(this)",
                staged_vector,
            )
            self.assertIn("void InitP9Runtime(", staged_vector)
            self.assertIn("void P9ObserveScoreSeam(", staged_vector)
            self.assertIn("void P9PostAtScoreBoundary()", staged_vector)
            self.assertIn("void P9FinalizeExactTopKRow(", staged_vector)
            # P9 may not consume the official Top-K MTE3->V event a second
            # time; the tail must use a separately legal, anchored boundary.
            p9_finalizer = applier.function_body(
                staged_vector, "QLIVector<QLIT>::P9FinalizeExactTopKRow"
            )
            self.assertNotIn(
                "WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT)",
                p9_finalizer,
            )

            wrapper = (
                staged_root / "op_kernel/a5_c8_exact_speculative_resident_get.cpp"
            ).read_text(encoding="utf-8")
            preload = (
                staged_root
                / "op_kernel/arch35/quant_lightning_indexer_kernel.h"
            ).read_text(encoding="utf-8")
            host_def = (
                staged_root
                / "op_host/a5_c8_exact_speculative_resident_get_def.cpp"
            ).read_text(encoding="utf-8")
            tiling = (
                staged_root
                / "op_host/a5_c8_exact_speculative_resident_get_tiling.cpp"
            ).read_text(encoding="utf-8")
            for camel_name in (
                "runtimeEnable",
                "residentThreshold",
                "residentThresholdValid",
                "reqPoolEntries",
                "kvObjectMap",
                "kvObjectLens",
                "persistentObjectSlotsSnapshot",
                "canonicalResidentSlots",
                "candidateSourceToSlot",
                "candidateSlotCursor",
                "layerResidentGetControl",
                "rowsPerEntry",
                "residentBlockTable",
                "activeRowCount",
                "attentionSlots",
                "terminalPostStats",
                "candidateSourceToSlotRef",
                "candidateSlotCursorRef",
            ):
                self.assertIn(camel_name, wrapper)
            self.assertIn("InitP9Runtime(runtimeEnable", preload)
            self.assertIn("candidateSourceToSlotRef", preload)
            self.assertIn("candidateSlotCursorRef", preload)

            # The host schema must compile as a real C++ OpDef.  In
            # particular, the generic builder receiver is `this->`, whereas
            # the 950 builder receiver already ends in `.`.  A previous
            # mechanical expansion emitted `this.Input`, which passed JSON
            # checks but failed only after the expensive CANN build.
            self.assertNotIn("this.Input(", host_def)
            self.assertNotIn("this.Output(", host_def)
            for token in (
                'this->Input("runtime_enable")',
                'this->Input("canonical_resident_slots")',
                'this->Input("resident_block_table")',
                'this->Input("active_row_count")',
                'this->Output("attention_slots")',
                'this->Output("candidate_source_to_slot_ref")',
                'this->Output("candidate_slot_cursor_ref")',
                'aicore_config_950.Input("runtime_enable")',
                'aicore_config_950.Input("canonical_resident_slots")',
                'aicore_config_950.Input("resident_block_table")',
                'aicore_config_950.Input("active_row_count")',
                'aicore_config_950.Output("attention_slots")',
                'aicore_config_950.Output("candidate_source_to_slot_ref")',
                'aicore_config_950.Output("candidate_slot_cursor_ref")',
            ):
                self.assertIn(token, host_def)
            self.assertNotIn('AddConfig("ascend910b"', host_def)
            self.assertNotIn('AddConfig("ascend910_93"', host_def)

            # P9 borrows C8's existing invalid-index scratch.  It must not
            # run before the official AIV InitBuffers call establishes that
            # local tensor.  The disabled path is intentionally an official
            # C8 execution plus one runtime-enable word read: no borrowed UB,
            # P9 slot/table/control GM, or P9 event hand-off is allowed before
            # that gate.
            preload_init = applier.function_body(
                preload, "QLIPreload<QLIT>::Init(__gm__"
            )
            self.assertLess(
                preload_init.index("InitBuffers();"),
                preload_init.index("vectorService.InitP9Runtime("),
            )
            self.assertIn(
                "if ASCEND_IS_AIV {\n"
                "        vectorService.InitP9Runtime(",
                preload_init,
            )
            p9_runtime = applier.function_body(
                staged_vector, "QLIVector<QLIT>::InitP9Runtime"
            )
            runtime_gate = p9_runtime.index("runtimeEnableGm.GetValue(0U)")
            scratch_borrow = p9_runtime.index("outInvalidLocal_.GetPhyAddr()")
            self.assertLess(runtime_gate, scratch_borrow)
            disabled_prefix = p9_runtime[:runtime_gate]
            self.assertNotIn("outInvalidLocal_.GetPhyAddr()", disabled_prefix)
            self.assertNotIn("SetFlag<", disabled_prefix)
            self.assertNotIn("WaitFlag<", disabled_prefix)
            self.assertLess(
                runtime_gate,
                p9_runtime.index("activeRowCount == nullptr"),
            )
            # Graph padding may reuse a live pool row.  Queue/control setup
            # is therefore lazy: InitP9Runtime installs only score state; the
            # first Direct-HCCP control read occurs in a helper guarded by a
            # concrete row that passed active_row_count.
            self.assertNotIn("p9PostExecutor_.Init(", p9_runtime)
            ensure_executor = applier.function_body(
                staged_vector, "QLIVector<QLIT>::EnsureP9PostExecutor"
            )
            self.assertLess(
                ensure_executor.index("activeRow >= p9ScoreHook_.ActiveRowCount()"),
                ensure_executor.index("rowsPerEntryGm.GetValue(0U)"),
            )
            post_boundary = applier.function_body(
                staged_vector, "QLIVector<QLIT>::P9PostAtScoreBoundary"
            )
            self.assertLess(
                post_boundary.index("activeRow >= p9ScoreHook_.ActiveRowCount()"),
                post_boundary.index("EnsureP9PostExecutor(activeRow)"),
            )

            # Tiling validates all P9 shapes before GetStorageShape() and
            # carries the exact 23-input ABI's index 14 (`kv_object_lens`).
            # This makes an ABI mismatch fail during host tiling rather than
            # at an unchecked device GM access.
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
                "p9Lens.GetDim(0) != p9RowCount",
                "p9CanonicalSlots.GetDim(1) != 2048",
                "p9ResidentBlockTable.GetDim(0) != p9RowCount",
                "p9ActiveRowCount.GetDim(0) != 1",
            ):
                self.assertIn(token, tiling)
            first_shape_read = tiling.index("p9SparseStorage->GetStorageShape()")
            self.assertLess(
                tiling.index("p9SparseStorage == nullptr"), first_shape_read
            )

            staged_ops = json.loads((staged_root / "ops.json").read_text())
            self.assertEqual(
                tensor_schema(staged_ops[0]["input_desc"]), OFFICIAL_INPUTS + P9_INPUTS
            )
            self.assertEqual(tensor_schema(staged_ops[0]["output_desc"]), OUTPUTS)
            self.assertNotIn("resident_packed_output", (staged_root / "ops.json").read_text())
            self.assertTrue(
                (
                    staged_root
                    / "op_host/a5_c8_exact_speculative_resident_get_def.cpp"
                ).is_file()
            )
            self.assertTrue((staged_root / "op_host/err/ops_err.h").is_file())
            self.assertTrue((staged_root / "op_host/log/log.h").is_file())

            # The applier must remain one atomic transform.  Keeping these
            # calls ordered in `main()` prevents a future refactor from
            # staging an active score seam around a partial 9/2 source tree.
            main_source = APPLIER_PATH.read_text(encoding="utf-8")
            main_body = main_source[main_source.index("\ndef main(") :]
            atomic_steps = (
                "copy_overlay_input(overlay_root, staged_root)",
                "inject_kernel_wrapper_abi(staged_root / KERNEL_WRAPPER_RELATIVE)",
                "inject_preload_abi(staged_root / PRELOAD_RELATIVE)",
                "inject_vector_runtime(staged_root / VECTOR_RELATIVE)",
                "inject_tiling_abi(",
                "inject_host_abi(",
                "inject_score_seam(staged_root / VECTOR_RELATIVE)",
                "rename_source_files(staged_root)",
                "replace_operator_identifiers(staged_root)",
                "validate_output(staged_root, topk_hash)",
            )
            positions = [main_body.index(step) for step in atomic_steps]
            self.assertEqual(positions, sorted(positions))

    def test_active_headers_reject_compile_disabled_closure(self) -> None:
        seam = (
            OVERLAY_ROOT
            / "op_kernel/arch35/a5_c8_exact_speculative_resident_score_seam.h"
        ).read_text(encoding="utf-8")
        finalizer = (
            TARGET_ROOT
            / "op_kernel/arch35/a5_c8_exact_speculative_resident_stage_b_finalizer.h"
        ).read_text(encoding="utf-8")
        self.assertIn("A5_C8_EXACT_SPECULATIVE_RESIDENT_SCORE_SEAM 1", seam)
        self.assertIn("A5_C8_EXACT_SPECULATIVE_RESIDENT_STAGE_B_FINALIZER 1", finalizer)
        self.assertNotIn("compile-disabled", seam)
        self.assertNotIn("compile-disabled", finalizer)
        self.assertIn("P9ObserveScoreSeam", seam)
        self.assertIn("P9PostAtScoreBoundary", seam)
        self.assertIn("P9FinalizeExactTopKRow", finalizer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
