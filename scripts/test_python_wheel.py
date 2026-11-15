#!/usr/bin/env python3
"""Install the SDK wheel in a clean venv and test its public import/runtime contract."""
import argparse
import importlib.machinery
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import tomllib
import venv

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('wheel', type=Path)
a = p.parse_args()
version = tomllib.loads((ROOT / 'packages/python/pyproject.toml').read_text())['project']['version']
assert (ROOT / 'packages/python/LICENSE').read_bytes() == (ROOT / 'LICENSE').read_bytes()
# The release packager must preserve the same SDK-only contract as pyproject.
loader = importlib.machinery.SourceFileLoader('package_registries', str(ROOT / 'scripts/package-registries'))
spec = importlib.util.spec_from_loader(loader.name, loader)
module = importlib.util.module_from_spec(spec)
import sys
sys.path.insert(0, str(ROOT / 'scripts'))
loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix='vork-wheel-') as directory:
    work = Path(directory)
    python_files = {f'vork/{f.name}': f for f in (ROOT / 'packages/python/vork').iterdir() if f.is_file()}
    module.wheel(work, 'vork', version, 'any', python_files)
    # Test each packaging route in its own fresh environment.
    for index, wheel in enumerate((a.wheel.resolve(), next(work.glob('*.whl')))):
        env = work / str(index)
        venv.EnvBuilder(with_pip=True).create(env)
        python = env / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', str(wheel)], check=True)
        subprocess.run([str(python), '-I', '-c', '''
from importlib.metadata import distribution
from pathlib import Path
import os, sys, tempfile
import vork
from vork import Scene, std, render, write_scene
package = distribution('vork')
assert package.version == sys.argv[1], package.version
assert Path(vork.__file__).is_relative_to(sys.prefix), vork.__file__
assert 'Root-Is-Purelib: true' in package.read_text('WHEEL')
assert 'Tag: py3-none-any' in package.read_text('WHEEL')
assert not package.requires, package.requires
assert not package.entry_points, package.entry_points
assert 'npm install -g @voyager-so/vork' in package.read_text('METADATA')
assert any(str(f).endswith('licenses/LICENSE') for f in package.files)
assert not (Path(__import__('sys').executable).parent / ('vork.exe' if os.name == 'nt' else 'vork')).exists()
with tempfile.TemporaryDirectory() as empty:
    os.environ.pop('VORK_BIN', None)
    os.environ['PATH'] = empty
    scene: Scene = {'version': 2, 'nodes': []}
    write_scene(scene, str(Path(empty) / 'scene.json'))
    for call in (lambda: std('rounded_box'), lambda: render('scene.json', 'out.png')):
        try:
            call()
        except RuntimeError as error:
            message = str(error)
            for text in ('vork renderer not found', 'VORK_BIN', 'npm install -g @voyager-so/vork', 'Curl installer: https://github.com/voyager-so/vork#install'):
                assert text in message, message
            print(message)
        else:
            raise AssertionError('missing renderer must explain installation')
print('Installed SDK import, scene writing, metadata and missing-renderer checks passed')
''', version], cwd=work, check=True)
