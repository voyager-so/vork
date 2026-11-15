"""Exact packet and ISO BMFF extent oracle shared by encoder tests and Greenlight."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import struct
import subprocess


def atom(data, kind):
    offset = 0
    while offset < len(data):
        size, name = struct.unpack_from('>I4s', data, offset)
        header = 8
        if size == 1:
            size, = struct.unpack_from('>Q', data, offset + 8)
            header = 16
        elif size == 0:
            size = len(data) - offset
        assert header <= size <= len(data) - offset, (name, size)
        if name == kind:
            return data[offset + header:offset + size]
        offset += size
    raise AssertionError(f'missing atom {kind!r}')


def extent(header):
    assert header[0] in (0, 1), header[0]
    return struct.unpack_from('>IQ' if header[0] else '>II', header, 20 if header[0] else 12)


def check(path, rate, frames):
    expected = frames / rate
    movie = atom(path.read_bytes(), b'moov')
    scale, duration = extent(atom(movie, b'mvhd'))
    assert Fraction(duration, scale) == expected, ('mvhd', duration, scale, expected)
    track = atom(movie, b'trak')
    header = atom(track, b'tkhd')
    assert header[0] in (0, 1), header[0]
    duration, = struct.unpack_from('>Q' if header[0] else '>I', header, 28 if header[0] else 20)
    assert Fraction(duration, scale) == expected, ('tkhd', duration, scale, expected)
    media = atom(track, b'mdia')
    media_scale, duration = extent(atom(media, b'mdhd'))
    assert Fraction(duration, media_scale) == expected, ('mdhd', duration, media_scale, expected)
    edits = atom(atom(track, b'edts'), b'elst')
    assert edits[0] in (0, 1), edits[0]
    count, = struct.unpack_from('>I', edits, 4)
    assert count == 1, ('elst count', count)
    duration, start, speed, fraction = struct.unpack_from('>Qqhh' if edits[0] else '>Iihh', edits, 8)
    assert Fraction(duration, scale) == expected, ('elst', duration, scale, expected)
    assert (start, speed, fraction) == (0, 1, 0), ('elst media start/rate', start, speed, fraction)
    timing = atom(atom(atom(media, b'minf'), b'stbl'), b'stts')
    count, = struct.unpack_from('>I', timing, 4)
    entries = [struct.unpack_from('>II', timing, 8 + i * 8) for i in range(count)]
    assert sum(n for n, _ in entries) == frames, entries
    assert all(Fraction(delta, media_scale) == 1 / rate for _, delta in entries), entries

    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=avg_frame_rate,r_frame_rate,time_base,nb_frames,duration_ts,width,height:packet=pts,dts,duration',
        '-of', 'json', str(path)], check=True, capture_output=True, text=True, timeout=60)
    data = json.loads(result.stdout)
    stream, = data['streams']
    assert Fraction(stream['time_base']) == Fraction(1, rate.numerator), stream
    for field in ('avg_frame_rate', 'r_frame_rate'):
        assert Fraction(stream[field]) == rate, stream
    assert int(stream['nb_frames']) == frames and stream['duration_ts'] == frames * rate.denominator, stream
    assert len(data['packets']) == frames
    for index, packet in enumerate(data['packets']):
        assert packet['pts'] == packet['dts'] == index * rate.denominator, (index, packet)
        assert packet['duration'] == rate.denominator, (index, packet)
    return data


def identities(path, data, first):
    stream, = data['streams']
    width, height = stream['width'], stream['height']
    # FFmpeg 5.1+ spells passthrough as -fps_mode; later releases removed -vsync.
    help_text = subprocess.run(['ffmpeg', '-hide_banner', '-h', 'full'], capture_output=True, text=True, timeout=60).stdout
    passthrough = ['-fps_mode', 'passthrough'] if any(line.split()[:1] and line.split()[0].split('[')[0] == '-fps_mode'
        for line in help_text.splitlines()) else ['-vsync', '0']
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), *passthrough,
        '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1'], timeout=60)
    size = width * height
    assert len(raw) == len(data['packets']) * size
    for index in range(len(data['packets'])):
        identity = sum((raw[index * size + width * (height // 2) + 4 * bit + 2] > 127) << bit for bit in range(16))
        assert identity == first + index, ('decoded identity', index, identity, first + index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('movie', type=Path)
    parser.add_argument('rate', type=Fraction)
    parser.add_argument('frames', type=int)
    parser.add_argument('--identity-start', type=int)
    args = parser.parse_args()
    data = check(args.movie, args.rate, args.frames)
    if args.identity_start is not None:
        identities(args.movie, data, args.identity_start)
