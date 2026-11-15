"""Regression: an older checkout must not inherit another checkout's freshness."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'cargo-build-root.py'
spec = importlib.util.spec_from_file_location('cargo_build_root', SCRIPT)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class CheckoutTests(unittest.TestCase):
    def test_unowned_populated_target_and_other_checkout_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / 'old'; old.mkdir(); (old / 'release').mkdir()
            with self.assertRaisesRegex(RuntimeError, 'ownership is unknown'):
                b.claim_target(old, root / 'a')
            fresh = root / 'fresh'
            b.claim_target(fresh, root / 'a')
            b.claim_target(fresh, root / 'a')
            with self.assertRaisesRegex(RuntimeError, 'another checkout'):
                b.claim_target(fresh, root / 'b')
            self.assertTrue((old / 'release').is_dir())

    def test_checkout_target_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / 'a'; a.mkdir()
            alias = root / 'alias'; alias.symlink_to(a)
            self.assertEqual(b.target_for_checkout(root, a), b.target_for_checkout(root, alias))
            self.assertNotEqual(b.target_for_checkout(root, a), b.target_for_checkout(root, root / 'b'))

    @unittest.skipUnless(shutil.which('bwrap') and (Path.home() / '.cargo/bin/cargo').exists(), 'requires Linux bwrap and Cargo')
    def test_older_checkouts_build_their_own_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Both trees predate the first build: mtimes alone cannot distinguish them.
            for name, value in [('a', 111), ('b', 222)]:
                source = root / name; (source / 'src').mkdir(parents=True)
                (source / 'Cargo.toml').write_text('[package]\nname="freshness_probe"\nversion="0.1.0"\nedition="2021"\n')
                (source / 'src/main.rs').write_text('fn main(){println!("' + str(value) + '");}\n')
                subprocess.run(['git', 'init', '-q', str(source)], check=True)
            env = os.environ.copy()
            env.update(CARGO_TARGET_DIR=str(root / 'target'), RUSTC_WRAPPER='', CARGO_INCREMENTAL='0')
            env.pop('SCCACHE_LOG', None); env.pop('SCCACHE_ERROR_LOG', None)
            for name, expected in [('a', '111'), ('b', '222'), ('a', '111')]:
                subprocess.run([str(SCRIPT), 'build', '--release', '--offline'], cwd=root / name, env=env,
                               check=True, capture_output=True)
                binary = b.target_for_checkout(root / 'target', root / name) / 'release/freshness_probe'
                self.assertEqual(subprocess.check_output([str(binary)], text=True).strip(), expected)


if __name__ == '__main__':
    unittest.main()
