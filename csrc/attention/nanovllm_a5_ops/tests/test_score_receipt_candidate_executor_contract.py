#!/usr/bin/env python3
"""Pure-source closure test for the isolated P7.0 candidate oracle.

This intentionally needs neither CANN nor torch.  It prevents P7.0 from
silently acquiring a second OpDef registration or an accidental C8/KVAF/SIMT
implementation dependency while it remains a completed-C8 score oracle.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[1]
P7_ROOT = OPS_ROOT / "a5_score_receipt_candidate_executor"
HOST = P7_ROOT / "op_host" / "a5_score_receipt_candidate_executor.cpp"
KERNEL = P7_ROOT / "op_kernel" / "a5_score_receipt_candidate_executor.cpp"
TILING = P7_ROOT / "op_kernel" / "a5_score_receipt_candidate_executor_tiling.h"
OPS_JSON = P7_ROOT / "ops.json"
SETUP = P7_ROOT / "torch_extension" / "score_receipt_candidate_executor_setup.py"
PACKAGE_INIT = (
    P7_ROOT
    / "torch_extension"
    / "nanovllm_score_receipt_executor"
    / "__init__.py"
)
ORACLE_TEST = OPS_ROOT / "tests" / "test_score_receipt_candidate_executor.py"
BUILD_SCRIPT = OPS_ROOT / "build_nanovllm_ops.sh"


def fail(message: str) -> None:
    print(f"P7_SOURCE_CONTRACT_FAIL {message}")
    raise SystemExit(1)


for path in (
    HOST,
    KERNEL,
    TILING,
    OPS_JSON,
    SETUP,
    PACKAGE_INIT,
    ORACLE_TEST,
    BUILD_SCRIPT,
):
    if not path.is_file():
        fail(f"missing={path}")

host_text = HOST.read_text(encoding="utf-8")
kernel_text = KERNEL.read_text(encoding="utf-8")
tiling_text = TILING.read_text(encoding="utf-8")
ops_text = OPS_JSON.read_text(encoding="utf-8")
build_text = BUILD_SCRIPT.read_text(encoding="utf-8")
oracle_text = ORACLE_TEST.read_text(encoding="utf-8")

op_adds = re.findall(
    r"OP_ADD\s*\(\s*A5ScoreReceiptCandidateExecutor\s*\)", host_text
)
if len(op_adds) != 1:
    fail(f"op_add_count={len(op_adds)} expected=1")

for required, text, label in (
    ('"op": "A5ScoreReceiptCandidateExecutor"', ops_text, "ops_json"),
    ("REGISTER_TILING_DEFAULT(A5ScoreReceiptCandidateExecutorTilingData)", kernel_text, "kernel"),
    ("GetTilingData<", host_text, "host"),
    ("A5ScoreReceiptCandidateExecutorTilingData", tiling_text, "tiling"),
    ("A5ScoreReceiptCandidateExecutor", build_text, "build_script"),
):
    if required not in text:
        fail(f"missing_required={label}:{required}")

code_text = "\n".join((host_text, kernel_text, tiling_text))
for forbidden in (
    r"^\s*#\s*include\s*[<\"](?:[^>\"]*/)?kvaf",
    r"^\s*#\s*include\s*[<\"](?:[^>\"]*/)?simt_api",
    r"^\s*#\s*include\s*[<\"][^>\"]*(?:quant_lightning|a5_streaming_quant)",
    r"\bkvaf_fused_get\s*::",
    r"\b(?:TryAcquire|Begin|TryRetire|Finish)KvafStreamingBatchGet\b",
    r"\b(?:DirectHccp|DirectHCCP|URMA|urma_)\b",
):
    if re.search(forbidden, code_text, flags=re.MULTILINE):
        fail(f"forbidden_dependency={forbidden}")

for required in (
    "STAGE_SCORE_RECEIPT_CANDIDATE_EXECUTOR_ONLY",
    "strict P7.0 staging",
    "P7 host registration closure",
):
    if required not in build_text:
        fail(f"build_guard_missing={required}")

for required in (
    "P7_ORACLE_CONFIG",
    "P7_ORACLE_RESULT",
    "P7_ORACLE_PASS",
    "torch.ops.a5_score_receipt_executor.score_receipt_candidate_executor",
):
    if required not in oracle_text:
        fail(f"oracle_gate_missing={required}")

# The Stage-A candidate receipt is not an epoch-only ready bit.  This source
# closure deliberately proves the four-word identity ABI and its release
# ordering without pretending this host-only test can observe a device race.
for required, text, label in (
    ('"receipt_identity"', ops_text, "ops_json"),
    ('"candidate_receipt_identity"', ops_text, "ops_json"),
    ("kCandidateReceiptIdentityWords = 4U", kernel_text, "kernel"),
    ("LoadStableInputIdentity", kernel_text, "kernel"),
    ("TombstoneOutputIdentity", kernel_text, "kernel"),
    ("PublishOutputIdentity", kernel_text, "kernel"),
    ("RECEIPT_IDENTITY_WORDS = 4", oracle_text, "oracle"),
):
    if required not in text:
        fail(f"receipt_identity_missing={label}:{required}")
publish_start = kernel_text.find("void PublishOutputIdentity")
publish_end = kernel_text.find("void ClearSeeds", publish_start)
if publish_start < 0 or publish_end < 0:
    fail("receipt_identity_publish_function_missing")
publish = kernel_text[publish_start:publish_end]
prefix = publish.find("output + 0U")
barrier = publish.find("DataSyncBarrier<MemDsbT::DDR>()", prefix)
generation = publish.find("output + 3U", barrier)
if min(prefix, barrier, generation) < 0 or not prefix < barrier < generation:
    fail("receipt_identity_generation_not_published_last")
for forbidden in (
    r"^\s*(?:from|import)\s+.*a5_streaming_quant_li_get",
    r"^\s*(?:from|import)\s+.*kvaf",
    r"torch\.ops\.(?:a5_streaming_quant_li_get|kvaf)",
    r"\b(?:DirectHccp|DirectHCCP|URMA|urma_)\b",
):
    if re.search(forbidden, oracle_text, flags=re.MULTILINE):
        fail(f"oracle_dependency={forbidden}")

print(
    "P7_SOURCE_CONTRACT_PASS "
    "op_add=1 c8_dependency=0 kvaf_dependency=0 simt_dependency=0 "
    "post_cq=0"
)
