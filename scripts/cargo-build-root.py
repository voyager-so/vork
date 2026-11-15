#!/usr/bin/env python3
"""Run Cargo with stable private mounts and checkout-owned freshness records."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


COMPILE_COMMANDS = {'build', 'b', 'check', 'c', 'test', 't', 'run', 'r', 'bench',
                    'clippy', 'rustc', 'rustdoc', 'doc', 'd', 'fix'}


def target_for_checkout(target, source):
    # Cargo's timestamp checks cannot distinguish older checkouts mounted at the
    # same source path. Never share its target/fingerprint state between them.
    key = hashlib.sha256(os.fsencode(source.resolve())).hexdigest()
    return target.resolve() / 'checkouts' / key


def claim_target(target, source):
    """Fail closed when a legacy target's source ownership cannot be proved."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    owner = target / '.vork-source.json'
    with (target / '.vork-source.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        expected = {'format': 1, 'source': str(source.resolve())}
        if owner.exists():
            if json.loads(owner.read_text()) != expected:
                raise RuntimeError('target belongs to another checkout; set CARGO_TARGET_DIR to a fresh directory')
        else:
            if any(p.name != '.vork-source.lock' for p in target.iterdir()):
                raise RuntimeError('target ownership is unknown; set CARGO_TARGET_DIR to a fresh directory')
            temporary = target / '.vork-source.tmp'
            temporary.write_text(json.dumps(expected) + '\n')
            temporary.replace(owner)


def main():
    real = str(Path.home() / '.cargo/bin/cargo')
    args = sys.argv[1:]
    if not os.environ.get('CARGO_TARGET_DIR'):
        os.execv(real, [real, *args])
    result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
    if result.returncode:
        os.execv(real, [real, *args])
    source = Path(result.stdout.strip()).resolve()
    if not (source / 'Cargo.toml').is_file():
        os.execv(real, [real, *args])
    target = target_for_checkout(Path(os.environ['CARGO_TARGET_DIR']), source)
    claim_target(target, source)
    if not any(arg in COMPILE_COMMANDS for arg in args):
        os.environ['CARGO_TARGET_DIR'] = str(target)
        os.execv(real, [real, *args])
    root = Path.home() / '.cache/vork-build-root'
    (root / 'source').mkdir(parents=True, exist_ok=True)
    (root / 'target').mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(CARGO_TARGET_DIR=str(root / 'target'), SCCACHE_CLIENT_SIDE='1')
    if 'SCCACHE_LOG' in env or env.get('SCCACHE_ERROR_LOG'):
        sys.exit('cargo-build-root: unset SCCACHE_LOG and SCCACHE_ERROR_LOG before using private mounts')
    command = ['bwrap', '--die-with-parent', '--bind', '/', '/', '--dev-bind', '/dev', '/dev',
               '--bind', str(source), str(root / 'source'), '--bind', str(target), str(root / 'target'),
               '--chdir', str(root / 'source' / Path.cwd().resolve().relative_to(source)), real, *args]
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit('cargo-build-root: ' + str(error))
