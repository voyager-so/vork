"""Runtime selection must not silently borrow a different installation."""
import sys
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from vork._engine import engine, _npm_engine
from vork._version import VERSION


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from test_support import write_executable


def npm_installation(binary, root, *, global_install=False, nested=False):
    modules = root / 'node_modules'
    package = modules / '@voyager-so' / 'vork'
    package.mkdir(parents=True)
    (package / 'package.json').write_text(json.dumps({
        'name': '@voyager-so/vork', 'exports': {'./package.json': './package.json'},
    }))
    native = (package / 'node_modules' if nested else modules) / '@voyager-so' / 'vork-windows-x64' / 'runtime' / 'vork.exe'
    native.parent.mkdir(parents=True)
    shutil.copy2(binary, native)
    (package / 'cli.cjs').write_text(
        "const {spawnSync} = require('node:child_process');\n"
        "const p = require.resolve('@voyager-so/vork-windows-x64/runtime/vork.exe');\n"
        "process.exit(spawnSync(p, process.argv.slice(2), {stdio:'inherit'}).status);\n")
    launch_dir = root if global_install else modules / '.bin'
    launch_dir.mkdir(exist_ok=True)
    launcher = launch_dir / 'vork.CMD'
    relative = 'node_modules' if global_install else '..'
    launcher.write_text('@echo off\nnode "%~dp0' + relative +
                        '/@voyager-so/vork/cli.cjs" %*\n')
    return launcher, native


@unittest.skipIf(os.name == 'nt', 'POSIX executable fixtures')
class EngineDiscovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {'PATH': str(self.root), 'VORK_BIN': ''})
        self.env.start()
        self.addCleanup(self.env.stop)

    def binary(self, name, version=VERSION):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_executable(path, f'#!/bin/sh\necho "vork {version} (commit abcdef123456; API v2)"\n')
        return str(path)

    def test_path_is_resolved_and_invocable(self):
        expected = self.binary('vork')
        self.assertEqual(engine(), expected)
        subprocess.run([engine(), '--version'], check=True, capture_output=True)

    def test_override_precedes_path(self):
        self.binary('vork', '0.1.0-internal.5')
        expected = self.binary('override')
        with patch.dict(os.environ, {'VORK_BIN': expected}):
            self.assertEqual(engine(), expected)

    def test_stale_path_is_refused(self):
        self.binary('vork', '0.1.0-internal.5')
        with self.assertRaisesRegex(RuntimeError, 'renderer.*SDK'):
            engine()

    def test_native_precedes_override_and_path(self):
        expected = self.binary('native/runtime/vork')
        init = self.root / 'native/__init__.py'
        init.touch()
        with patch('vork._engine.find_spec', return_value=SimpleNamespace(origin=str(init))), patch.dict(os.environ, {'VORK_BIN': '/missing'}):
            self.assertEqual(engine(), expected)

    def test_broken_native_does_not_fall_back(self):
        self.binary('vork')
        with patch('vork._engine.find_spec', return_value=SimpleNamespace(origin=str(self.root / 'native/__init__.py'))):
            with self.assertRaisesRegex(RuntimeError, 'not found'):
                engine()

    def test_wrong_identity_and_api_are_refused(self):
        path = Path(self.binary('vork'))
        for output in (f'other {VERSION}', f'vork {VERSION} (commit abcdef; API v1)'):
            write_executable(path, f'#!/bin/sh\necho "{output}"\n')
            with self.subTest(output=output), self.assertRaisesRegex(RuntimeError, 'does not match SDK'):
                engine()

    def test_source_archive_identity_is_supported(self):
        path = Path(self.binary('vork'))
        write_executable(path, path.read_text().replace('abcdef123456', 'source-archive'))
        self.assertEqual(engine(), str(path))

    def test_failed_version_probe_is_refused(self):
        path = Path(self.binary('vork'))
        write_executable(path, path.read_text() + 'exit 1\n')
        with self.assertRaisesRegex(RuntimeError, 'does not match SDK'):
            engine()

    def test_windows_extension_resolution_is_used(self):
        with patch('vork._engine.find_spec', return_value=None), patch('vork._engine.shutil.which', return_value=str(self.root / 'vork.CMD')) as which, patch('vork._engine.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=f'vork {VERSION} (commit abcdef; API v2)')) as run:
            self.assertEqual(engine(), str(self.root / 'vork.CMD'))
            which.assert_called_once_with('vork')
            self.assertEqual(run.call_args.args[0], [str(self.root / 'vork.CMD'), '--version'])


@unittest.skipUnless(os.name == 'nt', 'Windows subprocess/PATHEXT contract')
class WindowsLauncher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='vork-native-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        source = cls.root / 'fixture.rs'
        source.write_text('''
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args == ["--version"] {
        println!("vork VERSION (commit abcdef; API v2)");
    } else {
        std::fs::write(std::env::var("VORK_TEST_ARGS").unwrap(), args.join("\\n")).unwrap();
        println!("[]");
    }
}
'''.replace('VERSION', VERSION))
        cls.binary = cls.root / 'fixture.exe'
        subprocess.run(['rustc', str(source), '-o', str(cls.binary)], check=True)

    def test_batch_reproducer_and_native_argument_control(self):
        with tempfile.TemporaryDirectory(prefix='vork-cmd-') as directory:
            root = Path(directory)
            launcher, native = npm_installation(self.binary, root / 'npm')
            record = root / 'args.txt'
            args = ['render', 'a&b.json', '--out', 'a&b.png']
            with patch.dict(os.environ, {'VORK_TEST_ARGS': str(record)}):
                before = subprocess.run([str(launcher), *args], cwd=root, capture_output=True, text=True)
                batch_args = record.read_text(encoding='utf-8').splitlines()
                self.assertNotEqual(batch_args, args)
                subprocess.run([str(native), *args], cwd=root, check=True, capture_output=True)
                self.assertEqual(record.read_text(encoding='utf-8').splitlines(), args)
                print(f'Windows batch before: exit={before.returncode}, argv={batch_args!r}; native after: argv={args!r}')

    def test_npm_native_preserves_metacharacters_for_sdk_calls(self):
        from vork import render, export_asset
        from vork.ask import ask
        for layout in ('local', 'nested', 'global'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory(prefix='vork-npm-') as directory:
                root = Path(directory)
                launcher, native = npm_installation(self.binary, root / 'project with spaces & unicode é',
                                                     global_install=layout == 'global',
                                                     nested=layout == 'nested')
                record = root / 'args.txt'
                env = {'PATH': str(launcher.parent) + os.pathsep + os.environ['PATH'],
                       'VORK_BIN': '', 'VORK_TEST_ARGS': str(record)}
                with patch.dict(os.environ, env), patch('vork._engine.find_spec', return_value=None):
                    self.assertEqual(Path(engine()), native)
                    for name in ('a&b', 'a^b', 'a%b', 'a!b', 'a(b)', 'a&b é with spaces'):
                        scene = str(root / (name + '.json'))
                        Path(scene).write_text('{}')
                        output = str(root / (name + '.png'))
                        Path(output + '.meta.json').write_text('{"warnings": []}')
                        render(scene, output)
                        self.assertEqual(record.read_text(encoding='utf-8').splitlines(), ['render', scene, '--out', output])
                        self.assertEqual(ask(scene, []), [])
                        self.assertEqual(record.read_text(encoding='utf-8').splitlines(), ['ask', scene, '-'])
                        asset = str(root / (name + '.glb'))
                        Path(root / (name + '.export.json')).write_text('{}')
                        export_asset(scene, asset)
                        self.assertEqual(record.read_text(encoding='utf-8').splitlines(), ['export', scene, '--out', asset])
                    # An explicit npm launcher follows the same safe route.
                    with patch.dict(os.environ, {'VORK_BIN': str(launcher)}):
                        self.assertEqual(Path(engine()), native)
                    native.unlink()
                    with self.assertRaisesRegex(RuntimeError, 'include=optional'):
                        engine()

    def test_native_override_does_not_require_node(self):
        with patch.dict(os.environ, {'PATH': '', 'VORK_BIN': str(self.binary)}), patch('vork._engine.find_spec', return_value=None):
            self.assertEqual(Path(engine()), self.binary)


class NpmResolution(unittest.TestCase):
    def test_selected_installation_and_nested_dependencies(self):
        node = shutil.which('node')
        self.assertIsNotNone(node, 'npm runtime discovery requires Node')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.binary = root / 'fixture.exe'
            self.binary.touch()
            for layout in ('local', 'nested', 'global'):
                with self.subTest(layout=layout):
                    launcher, native = npm_installation(
                        self.binary, root / layout, global_install=layout == 'global', nested=layout == 'nested')
                    with patch('vork._engine.shutil.which', return_value=node):
                        self.assertEqual(Path(_npm_engine(str(launcher))), native)
                        native.unlink()
                        with self.assertRaisesRegex(RuntimeError, 'include=optional'):
                            _npm_engine(str(launcher))

    def test_missing_node_is_actionable(self):
        with patch('vork._engine.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'node.exe'):
                _npm_engine('vork.CMD')

    def test_node_failure_is_actionable(self):
        for error in (OSError('cannot launch'), subprocess.TimeoutExpired('node.exe', 30)):
            with self.subTest(error=error), patch('vork._engine.shutil.which', return_value='node.exe'), patch('vork._engine.subprocess.run', side_effect=error):
                with self.assertRaisesRegex(RuntimeError, 'Cannot resolve npm vork renderer'):
                    _npm_engine('vork.CMD')
