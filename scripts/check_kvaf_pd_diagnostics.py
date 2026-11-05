# SPDX-License-Identifier: Apache-2.0
"""Audit bounded PD diagnostic logs; this is not a model-accuracy certificate."""

import argparse
import json
from pathlib import Path


def audit(records, require_ddr=False, require_compute=False):
    if any(record.get("status") == "FAIL" for record in records):
        raise ValueError("diagnostic FAIL found; HTTP success does not override a failed KV check")
    ready = [record for record in records if record.get("event") == "DECODE_READY"]
    if not ready:
        raise ValueError("no DECODE_READY evidence; collect D logs with verify_kv enabled")
    paths = set()
    for gate in ready:
        same = [
            record
            for record in records
            if record.get("role") == "D"
            and (record.get("worker"), record.get("request")) == (gate.get("worker"), gate.get("request"))
        ]
        checks = [
            record
            for record in same
            if record.get("event") == "KV_CHECK"
            and record.get("generation") == gate.get("generation")
            and record.get("path") in ("npu", "cpu")
        ]
        replicas = [
            record
            for record in same
            if record.get("event") == "MAIN_REPLICA_CHECK" and record.get("generation") == gate.get("generation")
        ]
        if not gate.get("verify_kv") or gate.get("contributions") != gate.get("expected"):
            raise ValueError("Decode gate lacks byte verification or complete TP contributions")
        if not checks or len(checks) != gate.get("verified_batches"):
            raise ValueError("missing transfer byte-check records; log set may be incomplete")
        if not replicas or len(replicas) != gate.get("replica_checks"):
            raise ValueError(
                "missing P TP main-replica evidence: "
                f"worker={gate.get('worker')} request={gate.get('request')} "
                f"expected={gate.get('replica_checks')} observed={len(replicas)}"
            )
        if any(record.get("status") != "PASS" for record in checks + replicas):
            raise ValueError("KV or main-replica evidence is not PASS")
        current = {record["path"] for record in checks}
        if "npu" not in current or gate.get("full_blocks", 0) > 0 and "cpu" not in current:
            raise ValueError("required HBM/DDR path has no byte-check evidence")
        paths.update(current)
        if require_compute:
            phases = {record.get("event") for record in same}
            graph_returned = any(
                record.get("event") == "MODEL_FORWARD_DONE" and record.get("graph_mode") == "FULL" for record in same
            )
            if (
                "MODEL_FORWARD" not in phases
                or not ("SFA_DONE" in phases or graph_returned)
                or not phases.intersection({"MTP_PROPOSE_DONE", "MTP_SKIP"})
            ):
                raise ValueError("missing actual D model/SFA/MTP compute evidence")
            if gate.get("tail_tokens", 0) and "ABC_IMPORT" not in phases:
                raise ValueError("missing ordinary tail to ABC handoff evidence")
    if require_ddr and "cpu" not in paths:
        raise ValueError("no DDR main-block verification; send a prompt with at least 128 actual tokens")
    return len(ready), sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--require-ddr", action="store_true")
    parser.add_argument("--require-compute", action="store_true")
    args = parser.parse_args()
    records, unique = [], set()
    try:
        for path in args.logs:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if "KVAF_PD_DIAG " not in line:
                    continue
                raw = line.split("KVAF_PD_DIAG ", 1)[1]
                if raw not in unique:
                    records.append(json.loads(raw))
                    unique.add(raw)
        count, paths = audit(records, args.require_ddr, args.require_compute)
    except (OSError, ValueError) as error:
        parser.exit(1, f"PD_DIAGNOSTICS_FAIL: {error}\n")
    print(f"PD_DIAGNOSTICS_PASS completed_receives={count} paths={paths}; model accuracy requires a separate baseline")


if __name__ == "__main__":
    main()
