#!/usr/bin/env python3
"""65 rate/destination contracts; requires FFmpeg. VORK_GPU selects a device (default: hardware)."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
import time

RATES = Path(__file__).resolve().parents[2] / 'vork-encode/tests/fixtures/movie-rates.tsv'
FIRST, COUNT = 7, 32


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def probe(path, rate):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=avg_frame_rate,r_frame_rate,time_base,nb_frames,duration_ts:packet=pts,dts,duration',
        '-of', 'json', str(path)], check=True, capture_output=True, text=True, timeout=60)
    data = json.loads(result.stdout)
    path.with_suffix(path.suffix + '.probe.json').write_text(result.stdout)
    stream, = data['streams']
    assert Fraction(stream['time_base']) == Fraction(1, rate.numerator), stream
    for field in ('avg_frame_rate', 'r_frame_rate'):
        assert Fraction(stream[field]) == rate, stream
    assert int(stream['nb_frames']) == COUNT and stream['duration_ts'] == COUNT * rate.denominator, stream
    assert len(data['packets']) == COUNT
    for index, packet in enumerate(data['packets']):
        assert packet['pts'] == packet['dts'] == index * rate.denominator, (index, packet)
        assert packet['duration'] == rate.denominator, (index, packet)


def metadata(path, rate, directory):
    meta = path / 'meta.json' if directory else path.with_name(path.name + '.meta.json')
    value = json.loads(meta.read_text(), parse_float=str)  # exact authored decimal identity
    assert Fraction(str(value['fps'])) == rate, value['fps']
    assert [f['index'] for f in value['frames']] == list(range(FIRST, FIRST + COUNT))
    for frame in value['frames']:
        expected = float(Fraction(frame['index']) / rate)
        # Evaluation seconds have the existing f32 ABI; FPS + index is exact.
        assert struct.pack('<f', float(frame['time'])) == struct.pack('<f', expected), frame
    if directory:
        assert sorted(p.name for p in path.glob('*.png')) == [f'{i:06}.png' for i in range(FIRST, FIRST + COUNT)]
        for frame in path.glob('*.png'):
            assert frame.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'), frame


def run(binary, out):
    binary, out = binary.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    receipt = dict(protocol='exact-movie-rates-v1', passed=False, cells=[],
        binary_sha256=sha256(binary), fixture_sha256=sha256(RATES), first_frame=FIRST, frames=COUNT,
        version=subprocess.check_output([str(binary), '--version'], text=True).strip(),
        ffmpeg=subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0])

    def save():
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

    def cell(authored, kind, refused, collection, prefix=''):
        rate = Fraction(authored)
        folder = out / (prefix + authored.replace('/', '_')) / kind
        folder.mkdir(parents=True, exist_ok=True)
        scene = folder / 'scene.json'
        duration = Fraction(FIRST + COUNT) / rate
        value = dict(version=2, nodes=[dict(type='camera', translation=[0, 0, 4], target=[0, 0, 0])],
            render=dict(resolution=[64, 32], duration=f'{duration.numerator}/{duration.denominator}', background=[0.12, 0.35, 0.65]))
        token = json.dumps(authored) if '/' in authored else authored
        scene.write_text(json.dumps(value).replace('"resolution":', '"fps":' + token + ',"resolution":', 1))
        output = folder / ('frames' if kind == 'png' else 'clip.' + kind)
        row = dict(fps=authored, fraction=str(rate), format=kind, expected='refused' if refused else 'exact', evidence=str(folder.relative_to(out)))
        try:
            outputs, preserved = [output], []
            if refused:
                stills = folder / 'stills'
                stills.mkdir(exist_ok=True)
                outputs.append(stills)
                preserved = [output, output.with_name(output.name + '.meta.json'), stills / '000007.png', stills / 'meta.json']
                for path in preserved:
                    path.write_bytes(b'preserve existing output and metadata')
            command = [str(binary), 'render', str(scene), '--profile', 'std/previs',
                '--set', 'samples=1', '--set', 'denoise=false', '--frame-start', str(FIRST), '--frame-end', str(FIRST + COUNT - 1)]
            for path in outputs:
                command += ['--out', str(path)]
            (folder / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
            start = time.monotonic()
            with (folder / 'stdout.log').open('w') as stdout, (folder / 'stderr.log').open('w') as stderr:
                result = subprocess.run(command, stdout=stdout, stderr=stderr,
                    env=dict(os.environ, LP_NUM_THREADS='4', VORK_NO_UPDATE_CHECK='1'), timeout=600)
            row.update(exit=result.returncode, seconds=time.monotonic() - start)
            stderr = (folder / 'stderr.log').read_text()
            if refused:
                assert result.returncode == 2, (result.returncode, stderr)
                errors = [line for line in stderr.splitlines() if line.startswith('Error:')]
                assert len(errors) == 1, stderr
                if kind in ('mp4', 'mov'):
                    assert errors[0].endswith(f'{kind.upper()} cannot represent fps {rate}; use fps "1113587715/197197819"'), errors
                else:
                    assert 'unrecognized extension' in errors[0], errors
                for path in preserved:
                    assert path.read_bytes() == b'preserve existing output and metadata', path
            else:
                assert result.returncode == 0, (result.returncode, stderr)
                metadata(output, rate, kind == 'png')
                if kind != 'png':
                    probe(output, rate)
            row['passed'] = True
        except Exception as error:
            row.update(passed=False, error=str(error))
        collection.append(row)
        save()
        print(json.dumps(row), flush=True)

    with RATES.open() as stream:
        rates = list(csv.DictReader(stream, delimiter='\t'))
    for definition in rates:
        assert Fraction(definition['fps']) == Fraction(definition['reduced'])
        for kind in ('mp4', 'mov', 'png', 'webm', 'mkv'):
            refused = kind in ('webm', 'mkv') or (kind in ('mp4', 'mov') and definition['movie'] == 'refused')
            cell(definition['fps'], kind, refused, receipt['cells'])
    receipt['passed'] = len(receipt['cells']) == 65 and all(r['passed'] for r in receipt['cells'])
    receipt['verdict'] = 'complete' if receipt['passed'] else 'failed'
    save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    return 0 if run(args.binary, args.out)['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
