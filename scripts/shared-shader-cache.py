#!/usr/bin/env python3
"""Share successful content-keyed VM Slang entries, without changing its guard."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
import re
import shutil
import tarfile
import tempfile

spec = importlib.util.spec_from_file_location('shared_build', Path(__file__).with_name('shared-build.py'))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
KEY = re.compile(r'[0-9a-f]{64}')


def export_entry(root, key, stage):
    if not KEY.fullmatch(key):
        raise ValueError('invalid shader cache key')
    entry = root / key
    with (root / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (entry / 'ready').is_file():
            return None
        files = {}
        for p in sorted(entry.iterdir()):
            if p.name != 'ready' and not p.name.isdecimal():
                raise ValueError('unrecognized shader cache layout')
            if not p.is_file() or p.is_symlink():
                raise ValueError('shader cache entry is not a regular file')
            shutil.copyfile(p, stage / p.name)
            files[p.name] = b.digest(stage / p.name)
        return dict(key=key, files=files)


def import_entry(root, key, stage, manifest):
    if not KEY.fullmatch(key) or manifest['key'] != key or 'ready' not in manifest['files']:
        raise ValueError('shader key/receipt mismatch')
    for name, sha in manifest['files'].items():
        if name != 'ready' and not name.isdecimal():
            raise ValueError('invalid shader output name')
        p = stage / name
        if p.is_symlink() or not p.is_file() or b.digest(p) != sha:
            raise ValueError('shader checksum mismatch')
    if set(p.name for p in stage.iterdir()) != set(manifest['files']):
        raise ValueError('unexpected shader files')
    with (root / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / key / 'ready').exists():
            return False
        if (root / key).exists():
            raise ValueError('incomplete local entry; inspect without replacing it')
        # Caller stages on this filesystem. The existing guard sees only ready entries.
        stage.rename(root / key)
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['push', 'pull'])
    p.add_argument('--root', type=Path, required=True, help='existing shader-cache-v1 directory')
    p.add_argument('--bucket', required=True)
    p.add_argument('--guard', type=Path, required=True, help='slang-cache.py that owns this cache format')
    p.add_argument('--limit', type=int, default=100)
    a = p.parse_args()
    if a.limit < 1:
        p.error('--limit must be positive')
    a.root.mkdir(parents=True, exist_ok=True)
    cloud = b.Cloud(a.bucket)
    # Cache producer implementation belongs in the namespace: changing key semantics cannot reuse old entries.
    prefix = 'shaders/v1/' + b.digest(a.guard) + '/'
    if a.action == 'push':
        keys = [p.name for p in a.root.iterdir() if KEY.fullmatch(p.name) and (p / 'ready').is_file()]
    else:
        listing = subprocess.run(['gcloud', 'storage', 'ls', 'gs://' + a.bucket + '/' + prefix + '*.json'], capture_output=True, text=True)
        if listing.returncode and 'matched no objects' not in listing.stderr:
            raise RuntimeError('cannot list shared shader cache: ' + listing.stderr.strip())
        keys = [Path(line).stem for line in listing.stdout.splitlines()]
    completed = 0
    for key in sorted(keys):
        if completed >= a.limit:
            break
        if a.action == 'pull' and (a.root / key / 'ready').exists():
            continue
        manifest = cloud.get(prefix + key + '.json')
        if a.action == 'push' and manifest is not None:
            continue
        with tempfile.TemporaryDirectory(prefix='.transfer-', dir=a.root) as temp:
            temp = Path(temp); stage = temp / 'entry'; stage.mkdir(); archive = temp / 'entry.tar'
            if a.action == 'push':
                manifest = export_entry(a.root, key, stage)
                if manifest is None:
                    continue
                with tarfile.open(archive, 'w') as tf:
                    for path in stage.iterdir():
                        tf.add(path, arcname=path.name)
                data = archive.read_bytes(); manifest['archive_sha256'] = hashlib.sha256(data).hexdigest()
                # Content-address the payload as well as the input key. Racing uploads cannot overwrite.
                blob = prefix + manifest['archive_sha256'] + '.tar'
                cloud.request('POST', blob, data, generation=0)
                cloud.request('POST', prefix + key + '.json', b.encoded(manifest), generation=0)
            else:
                if manifest is None:
                    raise ValueError('missing shader manifest')
                blob = prefix + manifest['archive_sha256'] + '.tar'
                data = cloud.request('GET', blob)
                if data is None or hashlib.sha256(data).hexdigest() != manifest['archive_sha256']:
                    raise ValueError('shader archive checksum mismatch')
                archive.write_bytes(data); b.unpack(archive, stage)
                import_entry(a.root, key, stage, manifest)
            completed += 1
    print(json.dumps(dict(action=a.action, transferred=completed, namespace=prefix)))


if __name__ == '__main__':
    main()
