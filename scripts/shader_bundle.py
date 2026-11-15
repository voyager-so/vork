"""Release assets from the renderer's portable, content-addressed shader entries.

A bundle holds each entry as one XZ stream, `<entry name>.xz` (disk_cache.rs
reads it): a render expands only the few entries it needs.
"""
import hashlib
import json
import lzma
from pathlib import Path
import re
import shutil

ENTRY = re.compile(r'[0-9a-f]{64}\.(?:slang-variant-v1|slang-program-v2|slang-resolve-v2|aov-resolve-v1|slang-closed-input-v1)\.entry')
PACKED = re.compile(ENTRY.pattern + r'\.xz')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(binary, caches, output):
    # Preparation needs BLAKE3 to verify the renderer's existing cache envelope.
    # Packaging remains stdlib-only: its SHA-256 manifest binds these verified bytes.
    try:
        from blake3 import blake3
    except ImportError as error:
        raise RuntimeError('shader preparation requires blake3; run with uv run --with blake3==1.0.8 python scripts/prepare-shaders') from error
    entries = {}
    corrupt = {}
    for cache in caches:
        # A store holds entries as they are; a bundle beside a packaged binary holds them packed.
        for entry in sorted([*Path(cache).glob('*.entry'), *Path(cache).glob('*.entry.xz')]):
            packed = entry.suffix == '.xz'
            name = entry.name.removesuffix('.xz')
            if not ENTRY.fullmatch(name):
                continue
            data = entry.read_bytes()
            if packed:
                try:
                    data = lzma.decompress(data, format=lzma.FORMAT_XZ)
                except lzma.LZMAError:
                    data = b''
            if len(data) <= 32 or data[:32] != blake3(b'vork-cache-entry-v2\0' + name.encode() + b'\0' + data[32:]).digest():
                corrupt.setdefault(name, entry)
                continue
            entries[name] = data
    # A valid duplicate wins in either store order; a corrupt unique entry fails
    # preparation before any output is published, rather than silently losing it.
    for name, entry in corrupt.items():
        if name not in entries:
            raise ValueError(f'corrupt shader artifact: {entry}')
    if not entries:
        raise ValueError('setup produced no portable shader artifacts')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, data in entries.items():
        # CRC64 and LZMA2 alone: what the renderer's decoder reads.
        packed = lzma.compress(data, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=6)
        (output / (name + '.xz')).write_bytes(packed)
        files[name + '.xz'] = hashlib.sha256(packed).hexdigest()
    (output / 'manifest.json').write_text(json.dumps({
        'version': 2, 'binary_sha256': digest(binary), 'files': files,
    }, indent=2) + '\n')


def package(binary, source, output):
    source = Path(source)
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest.get('version') != 2 or manifest.get('binary_sha256') != digest(binary):
        raise ValueError('shader bundle must be prepared with the packaged binary')
    files = manifest['files']
    if not files:
        raise ValueError('empty shader bundle')
    for name, expected in files.items():
        if not PACKED.fullmatch(name) or digest(source / name) != expected:
            raise ValueError(f'invalid shader bundle entry: {name}')
    output = Path(output)
    output.mkdir()
    for name in [*files, 'manifest.json']:
        shutil.copy2(source / name, output / name)
