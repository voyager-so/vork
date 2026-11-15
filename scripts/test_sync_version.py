"""Release identities must agree across Cargo, SDKs, installers and tags."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent


class VersionTests(unittest.TestCase):
    def test_packaging_rejects_a_binary_with_the_wrong_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/package-release'),
                                     '--binary', sys.executable, '--platform', 'linux-x64',
                                     '--output', tmp, '--slang-toolchain', tmp, '--oidn-toolchain', tmp],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('does not match Cargo version', result.stderr)

    def test_release_channels_and_tag_drift(self):
        for version, python in [('1.2.3', '1.2.3'), ('0.1.0-beta.7', '0.1.0b7'),
                                ('0.1.0-alpha.2', '0.1.0a2'), ('0.1.0-rc.1', '0.1.0rc1'),
                                ('0.1.0-internal.9', '0.1.0.dev9')]:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name in ['scripts/sync-version', 'docs/help/index.md',
                             'packages/ts/package.json',
                             'packages/ts/package-lock.json',
                             'install.sh', 'install.ps1', 'packages/python/pyproject.toml', 'packages/python/vork/_version.py']:
                    dest = root / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / name, dest)
                (root / 'Cargo.toml').write_text(f'[workspace]\nmembers = []\n[workspace.package]\nversion = "{version}"\n')
                command = [sys.executable, str(root / 'scripts/sync-version')]
                subprocess.run(command, check=True, capture_output=True)
                subprocess.run(command + ['--check', '--tag', 'v' + version], check=True, capture_output=True)
                self.assertEqual(json.loads((root / 'packages/ts/package.json').read_text())['version'], version)
                self.assertIn(f'VERSION = {version!r}', (root / 'packages/python/vork/_version.py').read_text())
                self.assertIn(f'version = "{python}"', (root / 'packages/python/pyproject.toml').read_text())
                schema_url = f'https://raw.githubusercontent.com/voyager-so/vork/v{version}/packages/ts/scene.schema.json'
                for name in ['docs/help/index.md']:
                    self.assertIn(schema_url, (root / name).read_text())
                self.assertIn('VERSION:-' + version, (root / 'install.sh').read_text())
                self.assertIn("else { '" + version + "' }", (root / 'install.ps1').read_text())
                self.assertNotEqual(subprocess.run(command + ['--check', '--tag', 'v9.9.9'], capture_output=True).returncode, 0)
                (root / 'install.sh').write_text('drift')
                self.assertNotEqual(subprocess.run(command + ['--check'], capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
