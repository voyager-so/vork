#!/usr/bin/env python3
"""Build the pinned Mac compiler from source; never publishes an archive."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
pin = json.loads((ROOT / 'dist/slang.json').read_text())
recipe = pin['macos_source']
assert recipe['deployment_target'] == (ROOT / 'dist/macos-minimum.txt').read_text().strip()
source = a.source.resolve()
output = a.output.resolve()
output.mkdir(parents=True)
def run(*args):
    subprocess.run(args, cwd=source, check=True)
def capture(*args):
    return subprocess.check_output(args, cwd=source, text=True).strip()
assert capture('git', 'rev-parse', 'HEAD') == recipe['commit'], 'wrong Slang source commit'
assert not capture('git', 'status', '--porcelain', '--untracked-files=no'), 'modified Slang source'
assert capture('uname', '-m') == 'arm64'
assert capture('xcodebuild', '-version') == recipe['xcode'], 'select the pinned Xcode'
# Use the pinned Python-distributed tools even when Homebrew precedes them on PATH.
import cmake
assert importlib.metadata.version('cmake') == recipe['cmake']
assert importlib.metadata.version('ninja') == recipe['ninja']
ninja = importlib.metadata.distribution('ninja')
ninja_binary = next(ninja.locate_file(f).resolve() for f in ninja.files if f.name == 'ninja')
os.environ['PATH'] = os.pathsep.join((cmake.CMAKE_BIN_DIR, str(ninja_binary.parent), os.environ['PATH']))
assert capture('cmake', '--version').splitlines()[0] == 'cmake version ' + recipe['cmake']
os.environ['CC'] = capture('xcrun', '--find', 'clang')
os.environ['CXX'] = capture('xcrun', '--find', 'clang++')
os.environ['MACOSX_DEPLOYMENT_TARGET'] = recipe['deployment_target']
flags = recipe['cmake_flags']
run('cmake', '--preset', 'default', '--fresh', *flags,
    '-DCMAKE_OSX_ARCHITECTURES=arm64',
    '-DCMAKE_OSX_DEPLOYMENT_TARGET=' + recipe['deployment_target'])
run('cmake', '--build', 'build', '--config', 'Release', '--parallel', '3')
stage = output / 'toolchain'
run('cmake', '--install', 'build', '--config', 'Release', '--prefix', str(stage))
run('cmake', '--install', 'build', '--config', 'Release', '--component', 'metadata', '--prefix', str(stage))
# Record the actual SDK and all recursive dependency commits, not just a tag.
provenance = dict(recipe=recipe, submodules=capture('git', 'submodule', 'status', '--recursive'),
                  compiler=capture('clang', '--version'), sdk=capture('xcrun', '--show-sdk-version'),
                  cmake=capture('cmake', '--version'), ninja=capture('ninja', '--version'))
(stage / 'vork-build.json').write_text(json.dumps(provenance, indent=2) + '\n')
# Normalize container metadata; compiler reproducibility is checked separately
# by comparing generated shader bytes against upstream's official compiler.
epoch = int(capture('git', 'show', '-s', '--format=%ct', 'HEAD'))
archive = output / f"slang-{pin['version']}-macos-aarch64-vork.tar.gz"
with archive.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=epoch) as zipped:
    with tarfile.open(fileobj=zipped, mode='w') as tar:
        for path in sorted(stage.rglob('*')):
            info = tar.gettarinfo(str(path), str(path.relative_to(stage)))
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = epoch
            if info.isfile():
                with path.open('rb') as data:
                    tar.addfile(info, data)
            else:
                tar.addfile(info)
checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
(output / 'SHA256SUMS').write_text(f'{checksum}  {archive.name}\n')
print(checksum, archive)
