#!/usr/bin/env python3
"""Plan registry publication from tested files, with opt-in npm upload and PyPI staging.

Shared by `scripts/cut-release publish --dry-run` and the registry-publish
workflow. Every registry file is checked by sha256 against the tested manifest;
a changed, missing or extra file is a refusal (exit 2). The plan lists package,
version, registry, file and sha256 in publication order: native packages before
the wrappers that depend on them.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import subprocess
import time
from urllib.parse import quote

NPM_REGISTRY = 'https://registry.npmjs.org'
PYPI_REGISTRY = 'https://upload.pypi.org/legacy/'
NPM_VISIBILITY_TIMEOUT = 2700
REGISTRY_REQUEST_TIMEOUT = 60
NPM_PACKAGES = ('@voyager-so/vork-linux-x64', '@voyager-so/vork-darwin-arm64',
                '@voyager-so/vork-windows-x64', '@voyager-so/vork', 'create-vork')
PYPI_PROJECTS = ('vork',)
PACKAGES = NPM_PACKAGES + PYPI_PROJECTS


class Mismatch(Exception):
    """Publication cannot proceed with the available files or registry data."""


class RegistryUnavailable(Exception):
    """A transient registry response."""


def classify(name, version):
    """(registry, package) for a registry artifact of this version, else None."""
    for package in NPM_PACKAGES:
        if name == package.lstrip('@').replace('/', '-') + f'-{version}.tgz':
            return 'npm', package
    match = re.fullmatch(r'([A-Za-z0-9_]+)-' + re.escape(version) + r'-py3-none-[A-Za-z0-9_.]+\.whl', name)
    if match and match[1].replace('_', '-') in PYPI_PROJECTS:
        return 'pypi', match[1].replace('_', '-')
    return None


def plan(directory, manifest, version):
    """Verified rows for every tested registry file in directory, in publication order."""
    directory = Path(directory)
    if manifest.get('version') != version:
        raise Mismatch(f'tested manifest is for version {manifest.get("version")}, not {version}')
    tested = {n: h for n, h in manifest.get('files', {}).items() if classify(n, version)}
    present = {p.name for p in directory.iterdir() if classify(p.name, version)}
    problems = [f'{n}: present but not in the tested manifest' for n in sorted(present - set(tested))]
    problems += [f'{n}: tested but missing' for n in sorted(set(tested) - present)]
    named = {classify(n, version)[1] for n in tested}
    problems += [f'{p}: the tested manifest has no file for this package' for p in PACKAGES if p not in named]
    rows = []
    for name, expected in sorted(tested.items()):
        if name not in present:
            continue
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            problems.append(f'{name}: sha256 {actual} is not the tested {expected}')
            continue
        registry, package = classify(name, version)
        rows.append(dict(package=package, version=version, file=name, sha256=actual,
                         registry=NPM_REGISTRY if registry == 'npm' else PYPI_REGISTRY))
    if problems:
        raise Mismatch('Registry files are not the tested files:\n  ' + '\n  '.join(problems))
    rows.sort(key=lambda row: PACKAGES.index(row['package']))
    return rows


def render(rows):
    return '\n'.join(f'{row["package"]} {row["version"]} -> {row["registry"]}\n'
                     f'    {row["file"]}  sha256 {row["sha256"]}' for row in rows)


def registry_get(url, timeout, *, head=False):
    command = ['curl', '--silent', '--show-error', '--location',
               '--proto', '=https', '--proto-redir', '=https',
               '--header', 'Cache-Control: no-cache', '--write-out', r'\n%{http_code}']
    if head:
        command += ['--head', '--output', '/dev/null']
    result = subprocess.run(command + [url], check=True, capture_output=True, text=True, timeout=timeout)
    body, status = result.stdout.rsplit('\n', 1)
    if status == '404':
        return None
    if status in ('408', '429') or status.startswith('5'):
        raise RegistryUnavailable(f'HTTP {status} from {url}')
    if status != '200':
        raise Mismatch(f'Unexpected HTTP {status} from {url}')
    return body


def registry_json(url, timeout=REGISTRY_REQUEST_TIMEOUT):
    body = registry_get(url, timeout)
    if body is None:
        return None
    try:
        metadata = json.loads(body)
    except ValueError as error:
        raise Mismatch(f'Invalid registry JSON from {url}: {error}') from error
    if not isinstance(metadata, dict):
        raise Mismatch(f'Invalid registry metadata from {url}')
    return metadata


def npm_dist(row, timeout=REGISTRY_REQUEST_TIMEOUT):
    package = quote(row['package'], safe='@')
    metadata = registry_json(f'{NPM_REGISTRY}/{package}/{row["version"]}', timeout)
    if metadata is None:
        return None
    dist = metadata.get('dist')
    if (metadata.get('name') != row['package'] or metadata.get('version') != row['version']
            or not isinstance(dist, dict)):
        raise Mismatch(f'Invalid npm metadata for {row["package"]}@{row["version"]}')
    return dist


def npm_digests(file):
    data = file.read_bytes()
    digests = {algorithm: hashlib.new(algorithm, data) for algorithm in ('sha1', 'sha256', 'sha384', 'sha512')}
    integrity = {f'{algorithm}-{base64.b64encode(digest.digest()).decode("ascii")}'
                 for algorithm, digest in digests.items()}
    return digests['sha1'].hexdigest(), integrity


def verify_npm_digest(spec, dist, digests):
    expected_shasum, expected_integrity = digests
    shasum = dist.get('shasum')
    integrity = dist.get('integrity')
    if shasum is None and integrity is None:
        raise Mismatch(f'npm has no digest for {spec}')
    if shasum is not None and shasum != expected_shasum:
        raise Mismatch(f'npm shasum for {spec} differs from the tested tarball')
    if integrity is not None and (not isinstance(integrity, str) or integrity not in expected_integrity):
        raise Mismatch(f'npm integrity for {spec} differs from the tested tarball')


def wait_npm_visible(row, digests, timeout=NPM_VISIBILITY_TIMEOUT, interval=15):
    """Require matching exact-version metadata and a reachable tarball."""
    spec = f'{row["package"]}@{row["version"]}'
    deadline = time.monotonic() + timeout
    last_error = 'no successful registry response'

    def request_timeout():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('visibility deadline reached')
        return min(REGISTRY_REQUEST_TIMEOUT, remaining)

    while time.monotonic() < deadline:
        try:
            dist = npm_dist(row, request_timeout())
            # npm can return 404 during propagation even after accepting the upload,
            # and its tarball can return 404 long after the version metadata is visible.
            if dist is None:
                last_error = 'HTTP 404 for version metadata (waiting for propagation)'
            else:
                verify_npm_digest(spec, dist, digests)
                tarball = dist.get('tarball')
                if not isinstance(tarball, str) or not tarball.startswith('https://'):
                    raise Mismatch(f'Invalid npm tarball URL for {spec}: {tarball!r}')
                if registry_get(tarball, request_timeout(), head=True) is not None:
                    print(f'VISIBLE: {spec} (metadata and tarball)', flush=True)
                    return
                last_error = 'HTTP 404 for tarball (waiting for propagation)'
        except (RegistryUnavailable, subprocess.SubprocessError, TimeoutError) as error:
            last_error = f'transient request failure: {error}'
        remaining = deadline - time.monotonic()
        if remaining > 0:
            print(f'Waiting for {spec}: {last_error}', flush=True)
            time.sleep(min(interval, remaining))
    raise Mismatch(f'npm visibility timed out after {timeout}s for {spec}: {last_error}')


def publish_npm(rows, directory, visibility_timeout=NPM_VISIBILITY_TIMEOUT):
    """Publish in dependency order, waiting before any dependant can upload."""
    npm_rows = sorted((row for row in rows if row['registry'] == NPM_REGISTRY),
                      key=lambda row: NPM_PACKAGES.index(row['package']))
    digests_by_file = {}
    directory = Path(directory).resolve()
    for row in npm_rows:
        spec = f'{row["package"]}@{row["version"]}'
        file = directory / row['file']
        digests = digests_by_file[row['file']] = npm_digests(file)
        dist = npm_dist(row)
        if dist is None:
            subprocess.run(['npm', 'publish', str(file), '--ignore-scripts', '--access', 'public',
                            '--tag', 'latest', '--provenance', '--registry', NPM_REGISTRY], check=True)
        else:
            verify_npm_digest(spec, dist, digests)
            print(f'ALREADY PUBLISHED: {spec} (matches tested tarball)', flush=True)
        # Also wait for the SDK before create-vork, which pins it exactly.
        wait_npm_visible(row, digests, visibility_timeout)
    missing = []
    for row in npm_rows:
        try:
            wait_npm_visible(row, digests_by_file[row['file']], visibility_timeout)
        except Mismatch as error:
            missing.append(str(error))
    if missing:
        raise Mismatch('Final npm visibility check failed:\n  ' + '\n  '.join(missing))
    print(f'Final npm visibility check passed: {len(npm_rows)} packages.', flush=True)


def prepare_pypi(rows, directory, output):
    """Stage verified wheels whose exact filenames are absent from PyPI."""
    pending = []
    for row in rows:
        if row['registry'] != PYPI_REGISTRY:
            continue
        spec = f'{row["package"]}=={row["version"]}'
        metadata = registry_json(f'https://pypi.org/pypi/{row["package"]}/{row["version"]}/json')
        if metadata is None:
            pending.append(row['file'])
            continue
        files = metadata.get('urls')
        if not isinstance(files, list) or any(not isinstance(file, dict) for file in files):
            raise Mismatch(f'Invalid PyPI metadata for {spec}')
        existing = next((file for file in files if file.get('filename') == row['file']), None)
        if existing is None:
            pending.append(row['file'])
            continue
        digests = existing.get('digests')
        if not isinstance(digests, dict) or digests.get('sha256') != row['sha256']:
            raise Mismatch(f'PyPI sha256 for {row["file"]} differs from the tested wheel')
        print(f'ALREADY PUBLISHED: {row["file"]} (matches tested wheel)', flush=True)
    if pending:
        output.mkdir()
        for name in pending:
            shutil.copyfile(Path(directory) / name, output / name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dist', type=Path, required=True, help='directory holding the registry files')
    p.add_argument('--manifest', type=Path, required=True, help='tested manifest with sha256 per file')
    p.add_argument('--version', required=True)
    p.add_argument('--format', choices=('text', 'json'), default='text')
    action = p.add_mutually_exclusive_group()
    action.add_argument('--publish-npm', action='store_true',
                        help='publish verified npm files and wait for registry visibility')
    action.add_argument('--prepare-pypi', type=Path, metavar='DIRECTORY',
                        help='stage verified wheels that still need to be uploaded to PyPI')
    p.add_argument('--npm-visibility-timeout', type=int, default=NPM_VISIBILITY_TIMEOUT,
                   help=f'seconds to wait for each npm package to become visible (default: {NPM_VISIBILITY_TIMEOUT})')
    a = p.parse_args()
    if a.npm_visibility_timeout <= 0:
        p.error('--npm-visibility-timeout must be positive')
    try:
        rows = plan(a.dist, json.loads(a.manifest.read_text()), a.version)
        if a.publish_npm:
            publish_npm(rows, a.dist, a.npm_visibility_timeout)
            return
        if a.prepare_pypi:
            prepare_pypi(rows, a.dist, a.prepare_pypi)
            return
    except (Mismatch, RegistryUnavailable, subprocess.SubprocessError, OSError) as error:
        print(f'REFUSED: {error}', file=sys.stderr)
        sys.exit(2)
    if a.format == 'json':
        print(json.dumps(rows, indent=2))
    else:
        print(render(rows))
        print(f'{len(rows)} registry files are the tested files. Nothing uploaded.')


if __name__ == '__main__':
    main()
