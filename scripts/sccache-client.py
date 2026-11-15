#!/usr/bin/env python3
"""Keep compiler input reads inside the caller's private mount namespace."""
import os
import shutil
import sys


def main():
    binary = os.environ.get('VORK_SCCACHE')
    if not binary:
        sys.exit('sccache-client: set VORK_SCCACHE to the sccache executable')
    binary = shutil.which(binary) or binary
    env = os.environ.copy()
    # Cargo config can inject these after the outer launcher checks its env.
    # SCCACHE_LOG silently disables client-side mode in sccache 0.18.
    env.pop('SCCACHE_LOG', None)
    env.pop('SCCACHE_ERROR_LOG', None)
    env['SCCACHE_CLIENT_SIDE'] = '1'
    # A user dist.scheduler_url also disables client-side mode. This wrapper's
    # cache backend/credentials are supplied explicitly through the environment.
    env['SCCACHE_CONF'] = os.devnull
    os.execve(binary, [binary, *sys.argv[1:]], env)


if __name__ == '__main__':
    main()
