"""Scene input types, the shared vork standard library, scene writing and rendering."""
from ._call_keys import CALL_KEYS as _CALL_KEYS
import copy
import hashlib
import json
import math
import os
from ._engine import run_engine as _run_engine
import struct
import stat
import tempfile
import uuid
import re
from typing import Any, Optional, Sequence, Union, TypedDict
from .types import Scene
from ._validate import helper_options as _helper_options
from .session import Session, SessionError
from .ask import ask

# SDK authoring choice shared with @voyager-so/vork: externalize buffers of at least 4 KiB.
BINARY_BUFFER_MIN_BYTES = 4096
_WIDTHS = {'f32': 1, 'i32': 1, 'u32': 1, 'vec2': 2, 'vec3': 3, 'vec4': 4, 'uvec4': 4, 'mat4': 16}


def _identity(path: str) -> str:
    """The physical path, resolving the nearest existing ancestor of a missing file."""
    try:
        return os.path.realpath(path, strict=True)
    except FileNotFoundError:
        parent = os.path.dirname(path)
        return path if parent == path else os.path.join(_identity(parent), os.path.basename(path))


def _scene_buffers(value, path='scene', seen=None):
    """Visit each buffer object once, preserving shared-reference path semantics."""
    if seen is None:
        seen = set()
    if isinstance(value, (dict, list, tuple)):
        if id(value) in seen:
            return
        seen.add(id(value))
    if isinstance(value, dict):
        if isinstance(value.get('type'), str) and value['type'] in _WIDTHS and (isinstance(value.get('data'), (list, tuple)) or isinstance(value.get('file'), str)):
            yield path, value
        else:
            for key, child in value.items():
                yield from _scene_buffers(child, f'{path}.{key}', seen)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _scene_buffers(child, f'{path}[{index}]', seen)


def _external_buffers(scene: dict, file: str) -> Optional[bytes]:
    """Pack inline buffers of at least 4 KiB into one little-endian sidecar (mutates the copy)."""
    parts = []
    offset = 0
    shared = {}
    for name, buffer in _scene_buffers(scene):
        data = buffer.get('data')
        if data is None:
            continue
        width = _WIDTHS[buffer['type']]
        size = len(data) * width * 4
        if size < BINARY_BUFFER_MIN_BYTES:
            continue
        if buffer.get('file') or (buffer.get('offset') or 0) != 0 or buffer.get('count') is not None:
            raise ValueError(f'buffer {name}: inline data cannot have file, offset or count')
        dims = buffer.get('dims')
        if dims is not None and (len(dims) != 3 or any(type(n) is not int or n <= 0 for n in dims)
                                 or dims[0] * dims[1] * dims[2] != len(data)):
            raise ValueError(f'buffer {name}: data length must match positive dims product')
        kind = buffer['type']
        code = {'u32': 'I', 'uvec4': 'I', 'i32': 'i'}.get(kind, 'f')
        packed = bytearray()
        for row in data:
            values = []
            stack = [row]
            while stack:
                value = stack.pop()
                if isinstance(value, (list, tuple)):
                    stack.extend(reversed(value))
                else:
                    values.append(value)
            if len(values) != width:
                raise ValueError(f'buffer {name}: element width must be {width}')
            for value in values:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f'buffer {name}: expected finite numbers')
                if code != 'f' and not float(value).is_integer():
                    raise ValueError(f'buffer {name}: expected {"u32" if code == "I" else "i32"}')
                try:
                    packed += struct.pack('<' + code, int(value) if code != 'f' else value)
                except (struct.error, OverflowError):
                    raise ValueError(f'buffer {name}: expected {"u32" if code == "I" else "i32" if code == "i" else "finite f32"}') from None
        packed = bytes(packed)
        key = hashlib.sha256(packed).digest()
        previous = shared.get(key)
        same = previous is not None and previous[0] == packed
        at_offset = previous[1] if same else offset
        del buffer['data']
        buffer.update(file=file, offset=at_offset, count=len(data))
        if not same:
            shared[key] = (packed, offset)
            parts.append(packed)
            offset += size
    return b''.join(parts) if parts else None


def _previous_sidecars(path):
    try:
        with open(path, encoding='utf-8') as f:
            previous = json.load(f)
    except (FileNotFoundError, ValueError):
        return set()
    pattern = re.escape(os.path.basename(path)) + r'\.buffers\.[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.bin'
    return {os.path.join(os.path.dirname(path), buffer['file']) for _, buffer in _scene_buffers(previous)
            if isinstance(buffer.get('file'), str) and re.fullmatch(pattern, buffer['file'])}


def write_scene(scene: Scene, path: str, base_dir: Optional[str] = None) -> str:
    """Write scene JSON with relative resource paths and a binary sidecar for large buffers.

    Mirrors writeScene in @voyager-so/vork: resource paths resolve against base_dir (default:
    the working directory) and are written relative to the scene's directory; inline buffers
    of at least 4 KiB go to an immutable PATH.buffers.UUID.bin sidecar. The JSON is
    atomically replaced after its sidecar is complete; obsolete owned sidecars are removed. The input is not
    modified. Serialize saves and buffer reads at one destination; copy JSON and
    its resources for a lasting snapshot. Returns the absolute scene path.
    """
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.islink(destination):
        raise ValueError('write_scene destination must not be a symbolic link')
    scene_dir = os.path.realpath(os.path.dirname(destination))
    physical = os.path.join(scene_dir, os.path.basename(destination))
    base = os.path.abspath(base_dir if base_dir is not None else os.getcwd())

    def asset_path(file: str) -> str:
        source = os.path.join(base, file)
        physical_source = os.path.join(_identity(os.path.dirname(source)), os.path.basename(source))
        try:
            return os.path.relpath(physical_source, scene_dir)
        except ValueError:  # another Windows drive: keep it absolute, as Node's path.relative does
            return physical_source

    result = copy.deepcopy(scene)

    # Generated from the Rust scene schema, shared with the TypeScript writer.
    from importlib.resources import files
    graph = json.loads(files(__package__).joinpath('_resource_slots.json').read_text())
    slots = {}

    def walk(value, node, parent=None, key=None):
        if node.get('file') and isinstance(value, str) and parent is not None:
            slots[(id(parent), key)] = (parent, key)
        if 'ref' in node:
            walk(value, graph['defs'][node['ref']])
        for branch in node.get('branches', []):
            walk(value, branch, parent, key)
        if isinstance(value, dict):
            for field, child in node.get('properties', {}).items():
                if field in value:
                    walk(value[field], child, value, field)
            if 'values' in node:
                for field, child in value.items():
                    walk(child, node['values'], value, field)
        elif isinstance(value, (list, tuple)) and 'items' in node:
            for index, child in enumerate(value):
                walk(child, node['items'], value, index)

    walk(result, graph['root'])
    for parent, key in slots.values():
        parent[key] = asset_path(parent[key])
    previous = _previous_sidecars(physical)
    # Publish a complete pair before retiring obsolete generated siblings.
    binary_path = f'{physical}.buffers.{uuid.uuid4()}.bin'
    binary = _external_buffers(result, os.path.relpath(binary_path, scene_dir))
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    temporary = None
    binary_created = False
    temporary_created = False
    try:
        if binary is not None:
            handle = os.open(binary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            binary_created = True
            with os.fdopen(handle, 'wb') as f:
                f.write(binary)
        temporary = f'{physical}.{uuid.uuid4()}.tmp'
        handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        temporary_created = True
        with os.fdopen(handle, 'wb') as f:
            f.write(text.encode('utf-8'))
        try:
            existing = os.lstat(physical)
        except FileNotFoundError:
            existing = None
        if existing is not None and stat.S_ISREG(existing.st_mode):
            os.chmod(temporary, stat.S_IMODE(existing.st_mode))
        # Creation respects umask; replacement JSON preserves its existing mode.
        os.replace(temporary, physical)
    except BaseException:
        if temporary_created and os.path.exists(temporary):
            os.unlink(temporary)
        if binary_created:
            os.unlink(binary_path)
        raise
    retained = {os.path.realpath(os.path.join(scene_dir, parent[key])) for parent, key in slots.values()}
    for old in previous:
        if os.path.realpath(old) not in retained:
            try:
                if stat.S_ISREG(os.lstat(old).st_mode):
                    os.unlink(old)
            except FileNotFoundError:
                pass
    return destination


def std(function: str, argument: Any = None, **options: Any) -> Any:
    """Evaluate a shared std helper; pass either a JSON argument or keyword options.

    VORK_BIN selects the executable. Failures raise RuntimeError carrying the CLI's message.
    """
    if argument is not None and options:
        raise TypeError('pass an argument or keyword options')
    result = _run_engine(
        ['std', function, '-'],
        input=json.dumps(options if argument is None else argument, allow_nan=False),
        text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f'vork std {function}: {result.stderr.strip() or f"exit {result.returncode}"}')
    return json.loads(result.stdout)


@_helper_options
def _render_arguments(scene: str, out: Union[str, dict, Sequence[Union[str, dict]]], *, at: Optional[float] = None,
                     frame_start: Optional[int] = None, frame_end: Optional[int] = None,
                     profile: Optional[str] = None,
                     set: Optional[dict] = None,
                     channels: Optional[Sequence[str]] = None) -> list:
    """The `vork render` command line; channels default to color, ["all"] selects all.

    set accepts resolution, fps, seed, samples and denoise only.
    """
    args = ['render', str(scene)]
    for path in [out] if isinstance(out, (str, dict)) else out:
        args += ['--out', json.dumps(path, allow_nan=False, separators=(',', ':')) if isinstance(path, dict) else str(path)]
    if at is not None:
        args += ['--at', str(at)]
    if frame_start is not None:
        args += ['--frame-start', str(frame_start)]
    if frame_end is not None:
        args += ['--frame-end', str(frame_end)]
    if profile is not None:
        args += ['--profile', str(profile)]
    for key, value in (set or {}).items():
        if key not in _CALL_KEYS:
            raise TypeError('unknown render-call field: ' + key)
        args += ['--set', key + '=' + json.dumps(value, allow_nan=False, separators=(',', ':'))]
    if channels:
        args += ['--channels', ','.join(channels)]
    return args


class ExposureSummary(TypedDict):
    frames: int
    coverage: float
    clipped: Optional[float]
    near_black: Optional[float]
    mean: Optional[float]
    worst_clipped: Optional[dict[str, Union[float, int]]]
    worst_near_black: Optional[dict[str, Union[float, int]]]


class ExposureOutput(TypedDict):
    output: str
    exposure: ExposureSummary


class RenderWarning(TypedDict):
    code: str
    requested: dict[str, Any]
    dropped: str


class RenderResult(TypedDict):
    warnings: list[RenderWarning]
    exposure: list[ExposureOutput]


def render(scene: str, out: Union[str, dict, Sequence[Union[str, dict]]], **options: Any) -> RenderResult:
    """Render, print the CLI exposure report, and return full-precision statistics per output.

    Options are render-time choices (at, frame_start, frame_end, profile,
    set, channels). EXR and control-only outputs have no exposure entry.
    VORK_BIN selects the executable; the scene file is unchanged.
    """
    _run_engine(_render_arguments(scene, out, **options),
                   check=True)
    exposure = []
    warnings = []
    for output in [out] if isinstance(out, (str, dict)) else out:
        output = output["file"] if isinstance(output, dict) else output
        directory = output.endswith(os.sep) or output.endswith('/')
        file = not directory and os.path.splitext(output)[1].lower() in ('.png', '.mp4', '.mov', '.exr')
        name = os.path.basename(output)
        if '%06d' in name:
            before, after = name.split('%06d', 1)
            name = before.rstrip('._-') + after
        sidecar = 'meta.json' if name.startswith('.') and '.' not in name[1:] else name + '.meta.json'
        path = os.path.join(os.path.dirname(output), sidecar) if file else os.path.join(output, 'meta.json')
        with open(path, encoding='utf-8') as f:
            metadata = json.load(f)
        for warning in metadata['warnings']:
            if warning not in warnings:
                warnings.append(warning)
        if metadata.get('exposure') is not None:
            exposure.append({'output': output, 'exposure': metadata['exposure']})
    return {'exposure': exposure, 'warnings': warnings}


@_helper_options
def environment_image(file: str, output: str, *, name: str = 'environment', threshold: float = 10,
                      color_space: Optional[str] = None, fps: float = 24) -> dict:
    """Preprocess HDR lighting into an ordinary image program and declared sun."""
    options = dict(file=file, output=output, name=name, threshold=threshold, fps=fps)
    if color_space is not None:
        options['color_space'] = color_space
    return std('environment', options)


@_helper_options
def sky(*, name: str = 'sky', sun_elevation: float = 0.261799, sun_rotation: float = 0,
        altitude: float = 100, air_density: float = 1, aerosol_density: float = 1,
        ozone_density: float = 1, sun_size: float = 0.00951204, sun_intensity: float = 1) -> dict:
    """Nishita atmosphere and an ordinary solar emitter."""
    return std('sky', dict(name=name, sun_elevation=sun_elevation, sun_rotation=sun_rotation,
                          altitude=altitude, air_density=air_density, aerosol_density=aerosol_density,
                          ozone_density=ozone_density, sun_size=sun_size, sun_intensity=sun_intensity))


def export_asset(scene: str, out: str, *, root: Optional[str] = None,
                 at: Optional[Union[float, str]] = None,
                 frame_start: Optional[int] = None, frame_end: Optional[int] = None,
                 allow_losses: Optional[Sequence[str]] = None) -> dict:
    """Export an evaluated glTF/GLB snapshot or frame-range clip and return NAME.export.json.

    Each allowed loss names one diagnostic code; unsupported stages still fail.
    """
    args = ['export', str(scene), '--out', str(out)]
    if root is not None:
        args += ['--root', root]
    if at is not None:
        args += ['--at', str(at)]
    if frame_start is not None:
        args += ['--frame-start', str(frame_start)]
    if frame_end is not None:
        args += ['--frame-end', str(frame_end)]
    for code in allow_losses or []:
        args += ['--allow-loss', code]
    _run_engine(args, check=True)
    with open(os.path.splitext(out)[0] + '.export.json', encoding='utf-8') as file:
        return json.load(file)
