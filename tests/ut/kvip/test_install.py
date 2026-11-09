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
    def run_install(self, dirty=False, source_only=False, old_kvaf=False, experimental=False):
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
            if old_kvaf or experimental:
                (root / "third_party/KVAF").mkdir()
                (root / "third_party/KVAF/local.txt").write_text("preserve")
            if experimental:
                cmake = root / "third_party/KVAF/access_operator/demo/aiv_batch_get/kernel/CMakeLists.txt"
                cmake.parent.mkdir(parents=True)
                cmake.write_text("function(kvaf_build_isolated_urma_get)\n")
            binaries = root / "bin"
            binaries.mkdir()
            git = binaries / "git"
            template = root / "build-template.sh"
            template.write_text('printf "BUILD %s\\n" "$@"\n')
            git.write_text(
                '#!/bin/bash\ncase "$*" in\n'
                '  clone*) target="${@: -1}"; mkdir -p "$target/scripts"; '
                'cp "$BUILD_TEMPLATE" "$target/scripts/build.sh"; echo "CLONE $*" ;;\n'
                '  *"status --porcelain"*) if [ "$TEST_DIRTY" = 1 ]; then echo " M local.py"; fi ;;\n'
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
                TEST_DIRTY="1" if dirty else "0",
                TEST_KVAF_REVISION="1c1310387ebfb6776a07316874e924265b209ada" if experimental else "old",
            )
            args = ["bash", str(root / "scripts/install_kvip.sh")]
            args += ["--source-only"] if source_only else ["--skip-kvaf-build"]
            result = subprocess.run(args, env=env, capture_output=True, text=True)
            backups = list((root / "third_party").glob("KVIP-backup.*/KVIP/local.py"))
            self.assertEqual(len(backups), int(dirty))
            if dirty:
                self.assertEqual(backups[0].read_text(), "local edits")
            if old_kvaf or experimental:
                backups = list((root / "third_party").glob("KVAF-backup.*/KVAF/local.txt"))
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_text(), "preserve")
            return result

    def test_build_uses_external_checkout_and_explicit_kvaf_root(self):
        result = self.run_install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("81b1472809b7a5f3469e26467feee0ca1c61d71a", result.stdout)
        self.assertIn("BUILD --kvaf-root", result.stdout)
        self.assertIn("/third_party/KVAF", result.stdout)
        self.assertIn("Previous UBmem KVAF build not verified; rebuilding KVAF.", result.stdout)
        self.assertNotIn("BUILD --skip-kvaf-build", result.stdout)
        self.assertIn("BUILD ubmem", result.stdout)
        self.assertIn("URMA_BUILD", result.stdout)

    def test_upgrade_preserves_checkout_and_forces_rebuild(self):
        result = self.run_install(old_kvaf=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Previous KVAF preserved", result.stdout)
        self.assertNotIn("BUILD --skip-kvaf-build", result.stdout)

    def test_experimental_checkout_is_preserved_and_rebuilt(self):
        result = self.run_install(experimental=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Previous KVAF preserved", result.stdout)
        self.assertNotIn("BUILD --skip-kvaf-build", result.stdout)

    def test_source_only_does_not_build(self):
        result = self.run_install(source_only=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("BUILD", result.stdout)

    def test_dirty_checkout_is_preserved_and_built(self):
        result = self.run_install(dirty=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Previous KVIP preserved", result.stdout)
        self.assertIn("BUILD", result.stdout)
