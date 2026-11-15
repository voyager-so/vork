"""Resolve and verify the renderer before starting SDK work."""
from importlib.util import find_spec
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from ._version import VERSION


def engine():
    target = {
        ('linux', 'x86_64'): 'linux_x64',
        ('darwin', 'arm64'): 'darwin_arm64',
        ('win32', 'amd64'): 'windows_x64',
    }.get((sys.platform, platform.machine().lower()))
    spec = find_spec('vork_native_' + target) if target else None
    if spec is not None:
        executable = str(Path(spec.origin).parent / 'runtime' /
                         ('vork.exe' if os.name == 'nt' else 'vork'))
    else:
        requested = os.environ.get('VORK_BIN') or 'vork'
        # Resolve PATHEXT on Windows, including npm's vork.CMD launcher.
        executable = shutil.which(requested)
        if executable is None:
            raise _missing(requested)
        executable = os.path.abspath(executable)
        if os.name == 'nt' and Path(executable).suffix.lower() in ('.cmd', '.bat'):
            executable = _npm_engine(executable)
    try:
        result = subprocess.run([executable, '--version'], capture_output=True,
                                text=True, timeout=30, check=False)
    except FileNotFoundError:
        raise _missing(executable) from None
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f'Cannot verify vork renderer {executable!r}: {error}') from error
    identity = re.fullmatch(r'vork (\S+) \(commit [^;\s]+; API v2\)', result.stdout.strip())
    if result.returncode or identity is None or identity[1] != VERSION:
        raise RuntimeError(
            f'vork renderer {executable!r} does not match SDK {VERSION}: '
            f'{result.stdout.strip()!r}. Install the matching renderer release.'
        )
    return executable


def _npm_engine(launcher):
    # Resolve from the selected npm installation, including nested optional
    # dependencies and global installs. Never send SDK arguments through cmd.exe.
    node = shutil.which('node.exe')
    if node is None:
        raise _missing('node.exe')
    script = """
const {createRequire} = require('node:module');
const path = require('node:path');
const manifest = require.resolve('@voyager-so/vork/package.json', {
  paths: [path.dirname(process.argv[1])]
});
process.stdout.write(createRequire(manifest).resolve(
  '@voyager-so/vork-windows-x64/runtime/vork.exe'));
"""
    try:
        result = subprocess.run([node, '-e', script, launcher], capture_output=True,
                                text=True, encoding='utf-8', timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f'Cannot resolve npm vork renderer {launcher!r}: {error}') from error
    if result.returncode or not result.stdout.strip():
        raise RuntimeError(
            f'Cannot resolve npm vork renderer {launcher!r}. Reinstall '
            '@voyager-so/vork with --include=optional or set VORK_BIN to '
            'the full path of the native vork.exe.'
        )
    return result.stdout.strip()


def _missing(executable):
    return RuntimeError(
        f'vork renderer not found: {executable!r}. Install it with:\n'
        'npm install -g @voyager-so/vork\n'
        'Or set VORK_BIN to the vork executable. '
        'Curl installer: https://github.com/voyager-so/vork#install'
    )


def run_engine(args, **kwargs):
    executable = engine()
    try:
        return subprocess.run([executable, *args], **kwargs)
    except FileNotFoundError:
        raise _missing(executable) from None
