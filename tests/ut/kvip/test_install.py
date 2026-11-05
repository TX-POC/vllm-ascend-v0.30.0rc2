# SPDX-License-Identifier: Apache-2.0
"""Installer regression checks without network, compilation or NPU access."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class ExternalKvipInstallTest(unittest.TestCase):
    def run_install(self, dirty=False, source_only=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            shutil.copy(ROOT / "scripts/install_kvip.sh", root / "scripts")
            checkout = root / "third_party/KVIP"
            (checkout / "scripts").mkdir(parents=True)
            (checkout / "scripts/build.sh").write_text('printf "BUILD %s\\n" "$@"\n')
            binaries = root / "bin"
            binaries.mkdir()
            git = binaries / "git"
            git.write_text(
                '#!/bin/bash\ncase "$*" in\n'
                '  *"status --porcelain"*) echo "$FAKE_DIRTY" ;;\n'
                '  *"checkout --detach"*) echo "CHECKOUT $*" ;;\n'
                "esac\n"
            )
            git.chmod(0o755)
            env = dict(os.environ, PATH=f"{binaries}:{os.environ['PATH']}", FAKE_DIRTY=" M local.py" if dirty else "")
            args = ["bash", str(root / "scripts/install_kvip.sh")]
            args += ["--source-only"] if source_only else ["--skip-kvaf-build"]
            return subprocess.run(args, env=env, capture_output=True, text=True)

    def test_build_uses_external_checkout_and_explicit_kvaf_root(self):
        result = self.run_install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("46c03fcf889a3ab56e62c1dbf5acb60580120142", result.stdout)
        self.assertIn("BUILD --kvaf-root", result.stdout)
        self.assertIn("/third_party/KVAF", result.stdout)
        self.assertIn("BUILD --skip-kvaf-build", result.stdout)

    def test_source_only_does_not_build(self):
        result = self.run_install(source_only=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("BUILD", result.stdout)

    def test_dirty_checkout_is_not_overwritten(self):
        result = self.run_install(dirty=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("local changes", result.stderr)
        self.assertNotIn("CHECKOUT", result.stdout)
        self.assertNotIn("BUILD", result.stdout)
