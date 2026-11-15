"""Qualify the OS floor of the complete native runtime, including companions."""
import hashlib
import json
from pathlib import Path
import re
import struct

MACOS_MINIMUM = tuple(map(int, (Path(__file__).resolve().parents[1] / 'dist/macos-minimum.txt').read_text().strip().split('.'))) + (0,)
FLOORS = {'linux-x64': (2, 35), 'darwin-arm64': MACOS_MINIMUM}


# FAT headers have a 32- or 64-bit slice directory in either byte order.
FAT_FORMATS = {
    b'\xca\xfe\xba\xbe': ('>', 'IIIII'),
    b'\xbe\xba\xfe\xca': ('<', 'IIIII'),
    b'\xca\xfe\xba\xbf': ('>', 'IIQQII'),
    b'\xbf\xba\xfe\xca': ('<', 'IIQQII'),
}
MACH_MAGICS = {b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xce',
               b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf'}


def _macho_floor(data, source, *, is_slice=False, slices=None):
    magic = data[:4]
    required = (0, 0, 0)
    if magic in FAT_FORMATS:
        if is_slice:
            raise ValueError(f'nested universal Mach-O slice: {source}')
        endian, fields = FAT_FORMATS[magic]
        entry_size = struct.calcsize(endian + fields)
        if len(data) < 8:
            raise ValueError(f'truncated universal Mach-O header: {source}')
        count = struct.unpack_from(endian + 'I', data, 4)[0]
        table_end = 8 + count * entry_size
        if not count or table_end > len(data):
            raise ValueError(f'invalid universal Mach-O slice directory: {source}')
        for index in range(count):
            entry = struct.unpack_from(endian + fields, data, 8 + index * entry_size)
            offset, size = entry[2:4]
            if offset < table_end or size < 4 or offset + size > len(data):
                raise ValueError(f'invalid universal Mach-O slice {index}: {source}')
            required = max(required, _macho_floor(data[offset:offset + size], source, is_slice=True, slices=slices))
        return required
    if magic != b'\xcf\xfa\xed\xfe':
        if is_slice or magic in MACH_MAGICS:
            raise ValueError(f'unsupported Mach-O slice: {source}; use a little-endian 64-bit runtime')
        return required  # Nonbinary resources do not impose a deployment target.
    if len(data) < 32:
        raise ValueError(f'truncated Mach-O header: {source}')
    offset = 32
    found = False
    for _ in range(struct.unpack_from('<I', data, 16)[0]):
        if offset + 8 > len(data):
            raise ValueError(f'truncated Mach-O load command: {source}')
        command, size = struct.unpack_from('<II', data, offset)
        if size < 8 or offset + size > len(data):
            raise ValueError(f'invalid Mach-O load command: {source}')
        if command in (0x32, 0x24):
            field = 12 if command == 0x32 else 8
            if size < field + 4:
                raise ValueError(f'truncated Mach-O deployment target: {source}')
            if command == 0x32 and struct.unpack_from('<I', data, offset + 8)[0] != 1:
                raise ValueError(f'non-macOS Mach-O platform: {source}')
            found = True
            value = struct.unpack_from('<I', data, offset + field)[0]
            required = max(required, (value >> 16, (value >> 8) & 255, value & 255))
        offset += size
    if not found:
        raise ValueError(f'Mach-O has no deployment target: {source}')
    if slices is not None:
        slices.append(dict(cpu_type=struct.unpack_from('<I', data, 4)[0],
                           minos='.'.join(map(str, required))))
    return required


def elf_soname(data):
    """The DT_SONAME of a 64-bit little-endian ELF shared library, or None."""
    if data[:6] != b'\x7fELF\x02\x01':
        return None
    section_offset, = struct.unpack_from('<Q', data, 0x28)
    entry_size, count = struct.unpack_from('<HH', data, 0x3a)
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, section_offset + i * entry_size) for i in range(count)]
    for _, kind, _, _, offset, size, link, *_ in sections:
        if kind != 6:  # SHT_DYNAMIC
            continue
        strings = sections[link][4]
        for at in range(offset, offset + size, 16):
            tag, value = struct.unpack_from('<qQ', data, at)
            if tag == 0:
                break
            if tag == 14:  # DT_SONAME
                return data[strings + value:data.index(b'\0', strings + value)].decode()
    return None


def redundant_library_names(runtime):
    """Names in `runtime` (packaged name -> file) that repeat a shared library.

    A release tree names a library several ways through links (libtbb.so,
    libtbb.so.12, libtbb.so.12.19). The engine and every other library ask the
    loader for it by its SONAME, so a package, which cannot hold links, carries
    that one name. A file with one name, or no SONAME, is left as it is.
    """
    aliases = {}
    for name, source in runtime.items():
        aliases.setdefault(source.resolve(), []).append(name)
    redundant = []
    for target, names in aliases.items():
        soname = elf_soname(target.read_bytes()) if len(names) > 1 else None
        if soname is None:
            continue
        kept = [name for name in names if Path(name).name == soname]
        if len(kept) != 1:
            raise ValueError(f'{target}: none of its names {sorted(names)} is its SONAME {soname}')
        redundant += [name for name in names if name != kept[0]]
    return redundant


def check_runtime_platform(files, platform):
    minimum = (11, 0, 0) if platform == 'darwin-arm64' else (2, 0)
    for source in files:
        data = source.read_bytes()
        required = minimum
        if platform == 'linux-x64' and data[:4] == b'\x7fELF':
            versions = [tuple(map(int, v)) for v in re.findall(rb'GLIBC_(\d+)\.(\d+)', data)]
            required = max(versions, default=minimum)
        elif platform == 'darwin-arm64':
            required = _macho_floor(data, source)
        if platform in FLOORS and required > FLOORS[platform]:
            label = 'glibc' if platform == 'linux-x64' else 'macOS'
            expected = '.'.join(map(str, FLOORS[platform]))
            actual = '.'.join(map(str, required))
            raise ValueError(f'{source} requires {label} {actual}, exceeding the {expected} package baseline')
        minimum = max(minimum, required)
    return minimum


def macos_manifest(root):
    """Validate and record every Mach-O slice, including extensionless helpers."""
    paths = sorted(root.rglob('*'))
    for path in paths:
        if path.is_symlink() and (not path.is_file() or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError(f'runtime link must target an internal regular file: {path}')
    files = [p for p in paths if p.is_file()]
    minimum = check_runtime_platform(files, 'darwin-arm64')
    rows = []
    for path in files:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'runtime symlink escapes tree: {path}')
        data = path.read_bytes()
        slices = []
        _macho_floor(data, path, slices=slices)
        if slices:
            rows.append(dict(path=path.relative_to(root).as_posix(),
                             sha256=hashlib.sha256(data).hexdigest(), slices=slices))
    if not rows:
        raise ValueError('runtime has no Mach-O files')
    return dict(minimum_macos='.'.join(map(str, MACOS_MINIMUM)),
                maximum_minos='.'.join(map(str, minimum)), files=rows)


def verify_macos_manifest(root):
    recorded = json.loads((root / 'macos-runtime.json').read_text())
    if recorded != macos_manifest(root):
        raise ValueError('macos-runtime.json does not match bundled Mach-O files')
