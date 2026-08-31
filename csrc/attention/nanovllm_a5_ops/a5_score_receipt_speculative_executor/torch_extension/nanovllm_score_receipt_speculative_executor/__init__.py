"""Bind exactly one fresh P7.1 OPP before loading the isolated bridge.

P7.1 is C8-free by construction.  Importing it must never resolve a P3/P6
OPP left in ``ASCEND_CUSTOM_OPP_PATH``; the runner supplies a newly built OPP
root and this module resets the custom-op resolver to that single vendor.
"""

from __future__ import annotations

import os
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OPP = OPS_ROOT / "_custom_opp_score_receipt_speculative_executor"
OPP_ROOT = Path(
    os.getenv(
        "NANOVLLM_A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_OPP", str(DEFAULT_OPP)
    )
).resolve()
MANIFESTS = tuple(
    (OPP_ROOT / "vendors").glob(
        "*/P7_1_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_PROVENANCE"
    )
)
if len(MANIFESTS) != 1:
    raise RuntimeError(
        "P7.1 requires exactly one fresh provenance manifest under "
        f"{OPP_ROOT}/vendors; set "
        "NANOVLLM_A5_SCORE_RECEIPT_SPECULATIVE_EXECUTOR_OPP to the current build."
    )

MANIFEST = MANIFESTS[0].resolve()
FIELDS: dict[str, str] = {}
for line in MANIFEST.read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        FIELDS[key] = value

REQUIRED = {
    "kind": "standalone_score_receipt_speculative_executor",
    "abi": "31tensor_18refout",
    "torch_schema": "31in_18aliasout",
    "local_microbatches": "2x64",
    "queue_lifecycle": "epoch,generation,post_sequence,state,quarantine",
    "post_cq": "real_kvaf_streaming_postonly_tryretire_finish",
    "c8_source_closure": "absent",
    "fixed_get_core": "absent",
}
for key, expected in REQUIRED.items():
    if FIELDS.get(key) != expected:
        raise RuntimeError(
            f"P7.1 OPP manifest mismatch for {key}: "
            f"actual={FIELDS.get(key)!r} expected={expected!r}; manifest={MANIFEST}"
        )

expected_source = os.getenv("P7_EXPECTED_VLLM_SHA")
if expected_source and FIELDS.get("source_head") != expected_source:
    raise RuntimeError(
        "P7.1 OPP source revision mismatch: "
        f"actual={FIELDS.get('source_head')} expected={expected_source}"
    )
expected_kvaf = os.getenv("P7_EXPECTED_KVAF_SHA")
if expected_kvaf and FIELDS.get("kvaf_source_head") != expected_kvaf:
    raise RuntimeError(
        "P7.1 OPP KVAF revision mismatch: "
        f"actual={FIELDS.get('kvaf_source_head')} expected={expected_kvaf}"
    )

OPAPI = MANIFEST.parent / "op_api" / "lib" / "libcust_opapi.so"
if not OPAPI.is_file():
    raise RuntimeError(f"P7.1 OPP has no op_api library: {OPAPI}")

# Do not append: an older P3/P6 vendor can otherwise win symbol resolution.
os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(MANIFEST.parent)
os.environ["NANOVLLM_CUST_OPAPI_LIB"] = str(OPAPI)

if os.getenv("NANOVLLM_A5_OPAPI_DEBUG") not in (None, "", "0"):
    print(
        "P7_1_OPAPI_BIND "
        f"opp_root={OPP_ROOT} vendor={MANIFEST.parent} opapi={OPAPI} "
        f"source_head={FIELDS.get('source_head')} "
        f"kvaf_head={FIELDS.get('kvaf_source_head')} abi=31tensor_18refout",
        flush=True,
    )

from . import _C  # noqa: E402,F401

__all__ = []
