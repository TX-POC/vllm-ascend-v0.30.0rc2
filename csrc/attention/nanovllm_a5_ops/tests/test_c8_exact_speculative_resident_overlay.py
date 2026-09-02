#!/usr/bin/env python3
"""Offline closure test for P9's fresh source-overlay applier.

No CANN toolchain is needed here.  The test exercises the exact operation the
remote builder performs on a newly copied official C8 source tree and rejects
any overlay that changes `ProcessTopK`.
"""

from __future__ import annotations

import hashlib
import importlib.util
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

    def test_fresh_clone_gets_only_a_disabled_score_observer(self) -> None:
        applier = load_applier()
        original_vector = (
            OFFICIAL_C8_ROOT
            / "op_kernel/arch35/quant_lightning_indexer_service_vector.h"
        ).read_text(encoding="utf-8")
        original_topk = function_digest(original_vector, "QLIVector<QLIT>::ProcessTopK")

        with tempfile.TemporaryDirectory(prefix="p9-overlay-") as temporary:
            staged_root = Path(temporary) / "vllm_quant_lightning_indexer"
            shutil.copytree(OFFICIAL_C8_ROOT, staged_root)
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

            staged_vector = (
                staged_root / "op_kernel/arch35/quant_lightning_indexer_service_vector.h"
            ).read_text(encoding="utf-8")
            self.assertEqual(
                function_digest(staged_vector, "QLIVector<QLIT>::ProcessTopK"),
                original_topk,
            )
            reduction = staged_vector.index("vector1::BatchMulWeightAndReduceSum")
            marker = staged_vector.index(
                "A5_C8_EXACT_SPECULATIVE_RESIDENT_OBSERVE_SCORE_SEAM", reduction
            )
            publication = staged_vector.index("DataCopyPad(scoreGm", marker)
            self.assertLess(reduction, marker)
            self.assertLess(marker, publication)
            self.assertIn(
                '#include "a5_c8_exact_speculative_resident_score_seam.h"',
                staged_vector,
            )

            ops = json.loads((staged_root / "ops.json").read_text(encoding="utf-8"))
            self.assertEqual(ops[0]["op"], "A5C8ExactSpeculativeResidentGet")
            self.assertTrue(
                (
                    staged_root
                    / "op_host/a5_c8_exact_speculative_resident_get_def.cpp"
                ).is_file()
            )
            self.assertTrue(
                (
                    staged_root
                    / "op_kernel/a5_c8_exact_speculative_resident_get.cpp"
                ).is_file()
            )
            self.assertTrue((staged_root / "op_host/err/ops_err.h").is_file())
            self.assertTrue((staged_root / "op_host/log/log.h").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
