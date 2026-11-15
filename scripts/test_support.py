"""Executable fixtures must not leave writable descriptors in concurrent forks."""
from pathlib import Path
import subprocess
import sys


def write_executable(path, contents, mode=0o755):
    # Only this child opens the executable for writing. Waiting for it to exit
    # closes every writer, including copies inherited before CLOEXEC takes effect.
    # A close followed by rename in this process would preserve the busy inode.
    subprocess.run([sys.executable, '-c',
                    'import sys; open(sys.argv[1], "wb").write(sys.stdin.buffer.read())',
                    str(path)], input=contents.encode() if isinstance(contents, str) else contents,
                   check=True)
    Path(path).chmod(mode)
