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
    def run_install(self, dirty=False, source_only=False, old_kvaf=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            shutil.copy(ROOT / "scripts/install_kvip.sh", root / "scripts")
            checkout = root / "third_party/KVIP"
            (checkout / "scripts").mkdir(parents=True)
            (checkout / ".git").mkdir()
            (checkout / "scripts/build.sh").write_text('printf "BUILD %s\\n" "$@"\n')
            if dirty:
                (checkout / "local.py").write_text("local edits")
            if old_kvaf:
                (root / "third_party/KVAF").mkdir()
                (root / "third_party/KVAF/local.txt").write_text("preserve")
            binaries = root / "bin"
            binaries.mkdir()
            git = binaries / "git"
            template = root / "build-template.sh"
            template.write_text('printf "BUILD %s\\n" "$@"\n')
            git.write_text(
                '#!/bin/bash\ncase "$*" in\n'
                '  clone*) target="${@: -1}"; mkdir -p "$target/scripts"; '
                'cp "$BUILD_TEMPLATE" "$target/scripts/build.sh"; echo "CLONE $*" ;;\n'
                '  *"rev-parse HEAD"*) echo "$TEST_KVAF_REVISION" ;;\n'
                '  *"checkout --detach"*) echo "CHECKOUT $*" ;;\n'
                "esac\n"
            )
            git.chmod(0o755)
            python = binaries / "python3"
            python.write_text('#!/bin/bash\necho "URMA_BUILD $*"\n')
            python.chmod(0o755)
            env = dict(
                os.environ,
                PATH=f"{binaries}:{os.environ['PATH']}",
                BUILD_TEMPLATE=str(template),
                TEST_KVAF_REVISION="old",
            )
            args = ["bash", str(root / "scripts/install_kvip.sh")]
            args += ["--source-only"] if source_only else ["--skip-kvaf-build"]
            result = subprocess.run(args, env=env, capture_output=True, text=True)
            backups = list((root / "third_party").glob("KVIP-backup.*/KVIP/local.py"))
            self.assertEqual(len(backups), int(dirty))
            if dirty:
                self.assertEqual(backups[0].read_text(), "local edits")
            if old_kvaf:
                backups = list((root / "third_party").glob("KVAF-backup.*/KVAF/local.txt"))
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_text(), "preserve")
            return result

    def test_build_uses_external_checkout_and_explicit_kvaf_root(self):
        result = self.run_install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("clone --no-checkout --branch main https://github.com/TX-POC/KVAF-KVIP.git", result.stdout)
        self.assertIn("clone --no-checkout --branch main https://github.com/TX-POC/KVAF-Develop.git", result.stdout)
        self.assertIn("checkout --detach d6cfd392f722dd3f96ef6e05175f503aac511dbb", result.stdout)
        self.assertIn("checkout --detach 95a14f1cfaa6d7c76666f5d692df4c4f182360ec", result.stdout)
        self.assertIn("BUILD --kvaf-root", result.stdout)
        self.assertIn("/third_party/KVAF", result.stdout)
        self.assertNotIn("BUILD --skip-kvaf-build", result.stdout)
        self.assertIn("BUILD ubmem", result.stdout)
        self.assertIn("URMA_BUILD", result.stdout)

    def test_upgrade_preserves_checkout_and_forces_rebuild(self):
        result = self.run_install(old_kvaf=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Previous KVAF preserved", result.stdout)
        self.assertNotIn("BUILD --skip-kvaf-build", result.stdout)

    def test_source_only_prepares_without_compilation(self):
        result = self.run_install(source_only=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("BUILD --prepare-only", result.stdout)
        self.assertNotIn("URMA_BUILD", result.stdout)

    def test_dirty_checkout_is_preserved_and_built(self):
        result = self.run_install(dirty=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Previous KVIP preserved", result.stdout)
        self.assertIn("BUILD", result.stdout)
