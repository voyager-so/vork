#!/usr/bin/env python3
"""Opt-in, exact-commit release workspace binaries. Never adopts lane targets."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(args, **kwargs):
    return subprocess.check_output([str(x) for x in args], text=True, **kwargs).strip()


def tree_digest(paths, cache):
    """Memoize byte hashes only while every input's stat identity is unchanged."""
    files = []
    for label, root in paths:
        root = Path(root)
        for p in sorted(root.rglob('*')) if root.is_dir() else [root]:
            if p.is_file():
                s = p.stat()
                files.append((label + '/' + str(p.relative_to(root)) if root.is_dir() else label,
                              str(p), [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]))
    signature = hashlib.sha256(encoded(files)).hexdigest()
    memo = cache / (signature + '.json')
    if memo.exists():
        return json.loads(memo.read_text())
    result = hashlib.sha256(encoded([(label, digest(p)) for label, p, _ in files])).hexdigest()
    # An in-place toolchain change must not produce a memo under old stats.
    for _, p, expected in files:
        s = Path(p).stat()
        if [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns] != expected:
            raise RuntimeError('toolchain changed while fingerprinting')
    cache.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=cache)
    with os.fdopen(fd, 'w') as f:
        json.dump(result, f)
    os.replace(temporary, memo)
    return result


class Cloud:
    def __init__(self, bucket):
        self.bucket = bucket
        self.token_time = float('-inf')

    def request(self, method, name, data=None, generation=None):
        if time.monotonic() - self.token_time > 1800:
            self.token = run(['gcloud', 'auth', 'print-access-token'])
            self.token_time = time.monotonic()
        bucket = urllib.parse.quote(self.bucket, safe='')
        if method == 'POST':
            url = f'https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name=' + urllib.parse.quote(name, safe='')
        else:
            url = f'https://storage.googleapis.com/storage/v1/b/{bucket}/o/' + urllib.parse.quote(name, safe='')
            if method == 'GET':
                url += '?alt=media'
        if generation is not None:
            url += ('&' if '?' in url else '?') + 'ifGenerationMatch=' + str(generation)
        request = urllib.request.Request(url, data=data, method=method,
            headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/octet-stream'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code == 412 or (e.code == 404 and method in ('GET', 'DELETE')):
                return None
            raise

    def get(self, name):
        data = self.request('GET', name)
        return json.loads(data) if data is not None else None

    @contextlib.contextmanager
    def lock(self, key, timeout):
        name = 'locks/' + key
        deadline = time.monotonic() + timeout
        announced = False
        while True:
            result = self.request('POST', name, encoded(dict(host=os.uname().nodename,
                pid=os.getpid(), nonce=uuid.uuid4().hex, started=time.time())), generation=0)
            if result is not None:
                generation = json.loads(result)['generation']
                break
            if not announced:
                print('shared-build: waiting for producer ' + key, file=sys.stderr)
                announced = True
            if time.monotonic() >= deadline:
                raise RuntimeError('producer lock is busy; inspect gs://' + self.bucket + '/' + name + '; no automatic stale-lock takeover')
            time.sleep(10)
        try:
            yield
        finally:
            # Only the generation this process created can be removed.
            self.request('DELETE', name, generation=generation)

    def copy(self, source, target):
        subprocess.run(['gcloud', 'storage', 'cp', '--quiet', str(source), str(target)], check=True,
                       stdout=sys.stderr)


def validate(directory, identity):
    if (directory / 'receipt.json').is_symlink():
        raise ValueError('artifact receipt is a symlink')
    receipt = json.loads((directory / 'receipt.json').read_text())
    if receipt['identity'] != identity:
        raise ValueError('artifact identity mismatch')
    files = receipt['files']
    if not files or 'vork' not in files or 'vork-slang-filesystem' not in files:
        raise ValueError('artifact is missing required workspace binaries')
    if set(p.name for p in directory.iterdir()) != set(files) | {'receipt.json'}:
        raise ValueError('unexpected artifact files')
    for name, expected in files.items():
        if Path(name).name != name or name in ('.', '..', 'receipt.json'):
            raise ValueError('unsafe artifact name')
        p = directory / name
        if p.is_symlink() or not p.is_file() or digest(p) != expected:
            raise ValueError('artifact checksum mismatch: ' + name)
    return receipt


def unpack(archive, directory):
    with tarfile.open(archive) as tf:
        names = set()
        for member in tf.getmembers():
            if member.name in names:
                raise ValueError('duplicate archive member')
            names.add(member.name)
            if not member.isfile() or Path(member.name).name != member.name or member.name in ('.', '..'):
                raise ValueError('archive must contain only flat regular files')
        tf.extractall(directory, filter='data')


def environment(a, sha, target):
    # Do not inherit lane feature flags, profile overrides, native flags or wrappers.
    env = {k: os.environ[k] for k in ('HOME', 'USER', 'LOGNAME', 'TMPDIR', 'SSL_CERT_FILE') if k in os.environ}
    toolchain = tomllib.loads(run(['git', 'show', sha + ':rust-toolchain.toml'], cwd=a.repo))['toolchain']['channel']
    cargo_home = Path.home() / '.cache/vork-buildshare/cargo-home'
    cargo_home.mkdir(parents=True, exist_ok=True)
    for name in ('registry', 'git'):
        shared = Path.home() / '.cargo' / name
        link = cargo_home / name
        if shared.exists() and not link.exists():
            try:
                link.symlink_to(shared)
            except FileExistsError:
                if link.resolve() != shared.resolve():
                    raise RuntimeError('unexpected Cargo registry link')
    env.update(RUSTUP_TOOLCHAIN=toolchain, PATH=str(Path(a.cargo).parent) + ':' + str(Path.home() / '.cargo/bin') + ':' + str(Path.home() / '.local/bin') + ':/usr/local/bin:/usr/bin:/bin',
        CARGO_HOME=str(cargo_home), RUSTUP_HOME=str(Path.home() / '.rustup'),
        CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS=str(a.jobs), CARGO_INCREMENTAL='0',
        VORK_BUILD_COMMIT=sha, VORK_SLANGC=str(a.slang), VORK_SHADER_TARGETS='spirv,metal',
        RUSTC_WRAPPER=str(Path(__file__).with_name('sccache-client.py')) if a.sccache else '',
        VORK_SCCACHE=a.sccache, SCCACHE_CLIENT_SIDE='1',
        SCCACHE_SERVER_PORT=str(a.cache_port), SCCACHE_GCS_BUCKET=a.bucket,
        SCCACHE_GCS_KEY_PREFIX='rust/v1/', SCCACHE_GCS_RW_MODE='READ_WRITE', SCCACHE_IDLE_TIMEOUT='0')
    if a.credentials:
        env['SCCACHE_GCS_KEY_PATH'] = str(a.credentials)
    return env


def native_packages():
    if not shutil.which('dpkg-query'):
        return None
    # Kernel, desktop and operator tools do not define the native compiler ABI.
    prefixes = ('gcc', 'g++', 'binutils', 'libc6', 'libgcc', 'libstdc++',
                'libclang', 'clang', 'llvm', 'libtinfo', 'libzstd', 'zlib1g')
    packages = run(['dpkg-query', '-W', '-f=${binary:Package}=${Version}\n'])
    return [line for line in packages.splitlines() if line.startswith(prefixes)]


def cargo_config_fingerprints(cargo_home, roots):
    # Cargo also discovers .cargo/config in ancestors of its working directory,
    # even with a separate CARGO_HOME. The checkout config is covered by its tree.
    directories = {Path(cargo_home)}
    for root in roots:
        directories.update(directory / '.cargo' for directory in (root, *root.parents))
    return {str(p): digest(p) for directory in sorted(directories)
            for p in (directory / 'config', directory / 'config.toml') if p.is_file()}


def identity(a, sha):
    env = environment(a, sha, a.work / 'target')
    pin = json.loads(run(['git', 'show', sha + ':dist/slang.json'], cwd=a.repo))['version']
    slang_version = run([a.slang, '-version'], stderr=subprocess.STDOUT)
    if not re.search(r'(?<![0-9.])' + re.escape(pin) + r'(?![0-9.])', slang_version):
        raise ValueError('commit requires Slang ' + pin + '; choose its pinned toolchain')
    rustc = run(['rustup', 'which', 'rustc'], cwd=a.repo, env=env)
    sysroot = run([rustc, '--print', 'sysroot'], env=env)
    with a.slang.open('rb') as f:
        wrapped = f.read(2) == b'#!'
    if wrapped and not a.guard:
        raise ValueError('wrapped Slang requires --guard pointing at its cache-key implementation')
    paths = [('cargo-driver', Path(a.cargo)), ('rust', Path(sysroot) / 'bin'), ('rust-lib', Path(sysroot) / 'lib'),
             ('slang-bin', a.slang.parent), ('slang-lib', a.slang.parent.parent / 'lib'),
             ('slang-include', a.slang.parent.parent / 'include')]
    for name in ('cc', 'c++', 'ld', 'ar'):
        paths.append((name, Path(shutil.which(name, path=env['PATH'])).resolve()))
    result = dict(format=1, recipe=digest(__file__),
        compiler_wrapper=digest(Path(__file__).with_name('sccache-client.py')), commit=sha,
        tree=run(['git', 'rev-parse', sha + '^{tree}'], cwd=a.repo),
        rust=run([rustc, '-vV'], env=env), native=run(['c++', '-v'], env=env, stderr=subprocess.STDOUT),
        os=Path('/etc/os-release').read_text(), libc=run(['ldd', '--version'], env=env),
        toolchain=tree_digest(paths, a.work / 'identities'),
        shader_guard=digest(a.guard) if a.guard else None,
        native_packages=native_packages(),
        cargo_config=cargo_config_fingerprints(env['CARGO_HOME'],
            [a.work / 'sources', Path.home() / '.cache/vork-build-root']),
        features=sorted(set(a.features.split(','))) if a.features else [],
        default_features=not a.no_default_features, all_features=a.all_features,
        profile='release', workspace=True, locked=True, shader_targets=['spirv', 'metal'])
    return result


def cache_ready(a, env, source):
    if not a.sccache:
        raise RuntimeError('install sccache or use --fetch-only')
    stats = json.loads(run([a.sccache, '--show-stats', '--stats-format=json'], cwd=source, env=env))
    location = stats.get('cache_location', '')
    if not (location.startswith('gcs,') and 'name: ' + a.bucket + ',' in location
            and 'prefix: /rust/v1/' in location):
        raise RuntimeError('sccache endpoint uses another backend; select an unused --cache-port; do not restart a running job cache')


def build_paths(work, sha, expected):
    # Immutable identities never inherit Cargo freshness records from another tree.
    build_key = hashlib.sha256(encoded(expected)).hexdigest()
    return work / 'sources' / sha, work / 'targets' / build_key


def build(a, sha, expected, stage):
    source, target = build_paths(a.work, sha, expected)
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        subprocess.run(['git', 'worktree', 'add', '--detach', str(source), sha], cwd=a.repo, check=True, stdout=sys.stderr)
    else:
        if run(['git', 'status', '--porcelain', '--untracked-files=all', '--ignored'], cwd=source):
            raise RuntimeError('owned producer source is dirty; inspect ' + str(source))
        subprocess.run(['git', 'checkout', '--detach', sha], cwd=source, check=True, stdout=sys.stderr)
    env = environment(a, sha, target)
    cache_ready(a, env, source)
    args = [a.cargo, 'build', '--release', '--workspace', '--locked', '--message-format=json']
    if a.features:
        args += ['--features', a.features]
    if a.no_default_features:
        args += ['--no-default-features']
    if a.all_features:
        args += ['--all-features']
    print('shared-build: building ' + sha + '; log ' + str(a.work / 'build.log'), file=sys.stderr)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    with (a.work / 'build.jsonl').open('w') as output, (a.work / 'build.log').open('w') as log:
        result = subprocess.run(args, cwd=source, env=env, stdout=output, stderr=log)
        if result.returncode:
            raise RuntimeError('build failed; inspect ' + str(a.work / 'build.log'))
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    # Recheck toolchain bytes/config before publishing; inputs cannot drift mid-build.
    if (run(['git', 'rev-parse', 'HEAD'], cwd=source) != sha
            or run(['git', 'status', '--porcelain', '--untracked-files=all', '--ignored'], cwd=source)):
        raise RuntimeError('producer source changed during build; artifact not published')
    if identity(a, sha) != expected:
        raise RuntimeError('toolchain/config changed during build; artifact not published')
    binaries = set()
    cargo_fresh = {'fresh': 0, 'rebuilt': 0}
    for line in (a.work / 'build.jsonl').read_text().splitlines():
        message = json.loads(line)
        if message.get('reason') == 'compiler-artifact':
            cargo_fresh['fresh' if message.get('fresh') else 'rebuilt'] += 1
        if message.get('reason') == 'compiler-artifact' and message.get('executable'):
            executable = Path(message['executable'])
            if executable.parent.name != 'release':
                raise ValueError('unexpected workspace executable layout')
            # A launcher may bind-mount the private target at a stable path.
            binaries.add(target / 'release' / executable.name)
    binaries.add(target / 'release/vork-slang-filesystem')
    for binary in binaries:
        shutil.copy2(binary, stage / binary.name)
    receipt = dict(identity=expected, files={p.name: digest(p) for p in stage.iterdir()},
        build_seconds=time.monotonic() - start,
        build_cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        cpu_note='waited children; sccache server work excluded', cargo_fresh=cargo_fresh, created=time.time())
    (stage / 'receipt.json').write_bytes(encoded(receipt))
    validate(stage, expected)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('commit')
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--store', type=Path, default=Path(os.environ.get('VORK_SHARED_BINARIES', str(Path.home() / '.cache/vork-binaries'))) / 'buildshare-v1')
    p.add_argument('--work', type=Path, required=True, help='dedicated disposable producer workspace; never a lane target')
    p.add_argument('--bucket', required=True)
    p.add_argument('--slang', type=Path, required=True)
    p.add_argument('--cargo', default=shutil.which('cargo') or str(Path.home() / '.cargo/bin/cargo'))
    p.add_argument('--guard', type=Path, help='cache-key implementation for a wrapped Slang compiler')
    p.add_argument('--sccache', default=shutil.which('sccache') or '')
    p.add_argument('--credentials', type=Path, help='existing GCS credential file; referenced, never copied')
    p.add_argument('--cache-port', type=int, default=43319)
    p.add_argument('--jobs', type=int, default=6)
    p.add_argument('--features', default='')
    p.add_argument('--no-default-features', action='store_true')
    p.add_argument('--all-features', action='store_true')
    p.add_argument('--fetch-only', action='store_true')
    p.add_argument('--print-identity', action='store_true', help='print the exact key and provenance without building or fetching')
    p.add_argument('--lock-timeout', type=int, default=3600)
    a = p.parse_args()
    if sys.platform != 'linux':
        p.error('shared workspace producer currently requires Linux')
    a.repo = a.repo.resolve(); a.work = a.work.resolve(); a.store = a.store.resolve(); a.slang = a.slang.resolve()
    a.work.mkdir(parents=True, exist_ok=True); a.store.mkdir(parents=True, exist_ok=True)
    sha = run(['git', 'rev-parse', '--verify', '--end-of-options', a.commit + '^{commit}'], cwd=a.repo)
    expected = identity(a, sha)
    key = sha + '/' + hashlib.sha256(encoded(expected)).hexdigest()
    if a.print_identity:
        print(encoded(dict(key=key, identity=expected)).decode())
        return
    final = a.store / key
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        validate(final, expected); print(final); return
    # Per-key publication does not block fetches behind an unrelated build.
    with final.with_suffix('.lock').open('a') as local_lock:
        fcntl.flock(local_lock, fcntl.LOCK_EX)
        if final.exists():
            validate(final, expected); print(final); return
        cloud = Cloud(a.bucket)
        manifest_name = 'binaries/v1/' + key + '/receipt.json'
        archive_name = 'binaries/v1/' + key + '/binary.tar'
        with cloud.lock(key.replace('/', '-'), a.lock_timeout):
            manifest = cloud.get(manifest_name)
            with tempfile.TemporaryDirectory(prefix='artifact-', dir=a.work) as temp:
                temp = Path(temp); stage = temp / 'stage'; stage.mkdir(); archive = temp / 'binary.tar'
                if manifest:
                    if manifest['identity'] != expected:
                        raise ValueError('remote artifact identity mismatch')
                    print('shared-build: fetching ' + key, file=sys.stderr)
                    cloud.copy('gs://' + a.bucket + '/' + archive_name, archive)
                    if digest(archive) != manifest['archive_sha256']:
                        raise ValueError('archive checksum mismatch')
                    unpack(archive, stage); validate(stage, expected)
                else:
                    if a.fetch_only:
                        raise RuntimeError('exact artifact not built yet')
                    # Only builds serialize access to the mutable source/target.
                    with (a.work / 'producer.lock').open('a') as producer_lock:
                        fcntl.flock(producer_lock, fcntl.LOCK_EX)
                        manifest = build(a, sha, expected, stage)
                    with tarfile.open(archive, 'w') as tf:
                        for path in sorted(stage.iterdir()):
                            tf.add(path, arcname=path.name)
                    manifest['archive_sha256'] = digest(archive)
                    cloud.copy(archive, 'gs://' + a.bucket + '/' + archive_name)
                    if cloud.request('POST', manifest_name, encoded(manifest), generation=0) is None:
                        raise RuntimeError('immutable manifest already exists; inspect producer lock')
                # Stage on the destination filesystem; atomic rename is the ready marker.
                staging = Path(tempfile.mkdtemp(prefix='.publishing-', dir=final.parent))
                try:
                    shutil.copytree(stage, staging, dirs_exist_ok=True)
                    validate(staging, expected)
                    staging.rename(final)
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
    print(final)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        sys.exit('shared-build: ' + str(error))
