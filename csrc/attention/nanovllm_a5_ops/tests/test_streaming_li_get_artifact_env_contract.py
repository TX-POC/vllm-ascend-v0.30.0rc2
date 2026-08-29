#!/usr/bin/env python3
"""Regression test for independent OPP and runner provenance identities."""

from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
HELPER = OPS_ROOT / "tests" / "resolve_streaming_li_get_artifacts.sh"
REFERENCE_HEAD = "1" * 40
TARGET_HEAD = "2" * 40
KVAF_HEAD = "3" * 40


def _write_manifest(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="p3_artifact_env_") as temporary:
        temporary_root = Path(temporary)
        reference_root = temporary_root / "reference"
        target_root = temporary_root / "target"
        _write_manifest(
            reference_root / "vendors/ref/P3_REFERENCE_PROVENANCE",
            f"kind=source_matched_official_c8\nsource_head={REFERENCE_HEAD}\n",
        )
        _write_manifest(
            target_root / "vendors/target/P3_STREAMING_LI_PROVENANCE",
            "kind=source_derived_streaming_quant_li_get\n"
            f"source_head={TARGET_HEAD}\n"
            f"kvaf_source_head={KVAF_HEAD}\n",
        )
        command = "; ".join(
            (
                "export P3_REFERENCE_EXPECTED_SOURCE_HEAD=stale_reference",
                "P3_TARGET_EXPECTED_SOURCE_HEAD=stale_target",
                f"source {shlex.quote(str(HELPER))} {shlex.quote(str(reference_root))} {shlex.quote(str(target_root))} {shlex.quote(str(REPO_ROOT))}",
                'printf "P3_ARTIFACT_TEST ref=%s target=%s runner=%s kvaf=%s legacy_ref=%s legacy_target=%s\\n" '
                '"$P3_REFERENCE_EXPECTED_OPP_SOURCE_HEAD" '
                '"$P3_TARGET_EXPECTED_OPP_SOURCE_HEAD" '
                '"$P3_RUNNER_EXPECTED_HEAD" '
                '"${P3_KVAF_EXPECTED_SOURCE_HEAD:-<unset>}" '
                '"${P3_REFERENCE_EXPECTED_SOURCE_HEAD:-<unset>}" '
                '"${P3_TARGET_EXPECTED_SOURCE_HEAD:-<unset>}"',
            )
        )
        result = subprocess.run(
            ["bash", "-lc", command],
            check=False,
            capture_output=True,
            text=True,
        )

    assert result.returncode == 0, result.stderr
    expected_runner = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    assert (
        f"P3_ARTIFACT_ENV reference_opp_source_head={REFERENCE_HEAD} "
        f"target_opp_source_head={TARGET_HEAD} runner_head={expected_runner} "
        f"kvaf_source_head={KVAF_HEAD}"
    ) in result.stdout
    assert (
        f"P3_ARTIFACT_TEST ref={REFERENCE_HEAD} target={TARGET_HEAD} "
        f"runner={expected_runner} kvaf={KVAF_HEAD} "
        "legacy_ref=<unset> legacy_target=<unset>"
    ) in result.stdout
    print("P3_ARTIFACT_ENV_CONTRACT_PASS runner_and_opp_source_heads_independent=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
