"""The Mac rebuild uses its own pinned URL without weakening digest checks."""
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name('install-slang')


class InstallSlangTests(unittest.TestCase):
    def test_custom_archive_is_verified_before_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'archive.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                member = tarfile.TarInfo('bin/slangc')
                member.size = 8
                member.mode = 0o755
                tar.addfile(member, io.BytesIO(b'compiler'))
            pin = {'version': '2026.21', 'archives': {'Darwin-arm64': {
                'name': 'macos-aarch64-vork', 'url': archive.as_uri(),
                'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}}}
            original = Path.read_text
            def read(path, *args, **kwargs):
                return json.dumps(pin) if path.name == 'slang.json' else original(path, *args, **kwargs)
            with patch.object(Path, 'read_text', read), patch('platform.system', return_value='Darwin'), patch('platform.machine', return_value='arm64'):
                destination = root / 'installed'
                with patch.object(sys, 'argv', [str(SCRIPT), str(destination)]):
                    runpy.run_path(str(SCRIPT), run_name='__main__')
                self.assertEqual((destination / 'bin/slangc').read_bytes(), b'compiler')
                self.assertTrue((destination / 'bin/slangc').stat().st_mode & 0o111)
                pin['archives']['Darwin-arm64']['sha256'] = '0' * 64
                bad = root / 'bad'
                with patch.object(sys, 'argv', [str(SCRIPT), str(bad)]):
                    with self.assertRaisesRegex(SystemExit, 'SHA256 mismatch'):
                        runpy.run_path(str(SCRIPT), run_name='__main__')
                self.assertFalse(bad.exists())


    def test_public_archive_does_not_use_environment_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode='w:gz') as tar:
                member = tarfile.TarInfo('bin/slangc')
                member.size = 8
                tar.addfile(member, io.BytesIO(b'compiler'))
            payload = data.getvalue()
            url = json.loads((SCRIPT.parent.parent / 'dist/slang.json').read_text())['archives']['Darwin-arm64']['url']
            pin = {'version': '2026.21', 'archives': {'Darwin-arm64': {
                'name': 'macos-aarch64-vork', 'url': url,
                'sha256': hashlib.sha256(payload).hexdigest()}}}
            original = Path.read_text
            def read(path, *args, **kwargs):
                return json.dumps(pin) if path.name == 'slang.json' else original(path, *args, **kwargs)
            def download(actual_url, destination):
                self.assertEqual(actual_url, url)
                self.assertTrue(url.startswith('https://github.com/voyager-so/vork/releases/download/'))
                Path(destination).write_bytes(payload)
            with patch.object(Path, 'read_text', read), patch('platform.system', return_value='Darwin'), patch('platform.machine', return_value='arm64'), patch.dict(os.environ, GH_TOKEN='test-only-token', GITHUB_TOKEN='test-only-token'):
                with patch.object(sys, 'argv', [str(SCRIPT), str(root / 'installed')]), patch('urllib.request.urlretrieve', side_effect=download):
                    runpy.run_path(str(SCRIPT), run_name='__main__')
                self.assertEqual((root / 'installed/bin/slangc').read_bytes(), b'compiler')


if __name__ == '__main__':
    unittest.main()
