#!/usr/bin/env python3
"""Run every Python example contract with VORK_BIN pointing to the built CLI.

Murmuration fetches the setting assets pinned in its assets.lock.json. No GPU
is needed. Discover each directory separately: examples need not be packages,
and their names may contain hyphens.
"""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    directories = sorted({path.parent for path in (root / 'examples').rglob('test_*.py')})
    if not directories:
        raise SystemExit('No Python example tests found')
    failed = False
    for directory in directories:
        print(f'Testing {directory.relative_to(root)}', flush=True)
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover',
                                 '-v', '-s', str(directory), '-p', 'test_*.py'], cwd=root)
        failed |= result.returncode != 0
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
