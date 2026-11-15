#!/usr/bin/env python3
"""Compile real vork entries and the specialization regression; compare exact bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--compiler', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--reference', type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
samples = [('scripts/toolchains/address-space-dedup.slang', 'main')]
samples += [('crates/vork-render/shaders/present.slang', 'fs'),
            ('crates/vork-render/shaders/mip.slang', 'vs'),
            ('crates/vork-render/shaders/mip.slang', 'fs')]
rows = {}
for source, entry in samples:
    for target in ('spirv', 'metal'):
        name = f'{Path(source).stem}-{entry}.{target}'
        output = a.output / name
        subprocess.run([str(a.compiler.resolve()), source, '-entry', entry,
                        '-target', target, '-O2', '-o', str(output.resolve())]
                       + (['-DVORK_METAL=1'] if target == 'metal' else []), cwd=ROOT, check=True)
        data = output.read_bytes()
        assert data, name
        rows[name] = hashlib.sha256(data).hexdigest()
        if a.reference:
            assert data == (a.reference / name).read_bytes(), f'compiler output differs: {name}'
        print(name, rows[name], 'identical' if a.reference else 'compiled')
(a.output / 'sha256.json').write_text(json.dumps(rows, indent=2) + '\n')
