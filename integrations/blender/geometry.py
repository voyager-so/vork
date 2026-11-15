"""Retain Blender UV identities alongside glTF's ordinal texture coordinates.

Aliases share accessors with the original TEXCOORD streams. Surface programs
bind names, so one material also works on meshes with different UV layer order.
"""
import json
from pathlib import Path
import struct

_METADATA = '_vork_uv_bindings'
_BAKE_SOURCE = '_vork_bake_object'


def uv_attribute(name=None):
    return '_VORK_UV_ACTIVE' if name is None else '_VORK_UV_' + name.encode('utf8').hex().upper()


def retain_uv_bindings(document):
    """Consume temporary node metadata and add stable per-primitive aliases."""
    for node in document.get('nodes', []):
        extras = node.get('extras', {})
        metadata = extras.pop(_METADATA, None) if isinstance(extras, dict) else None
        if metadata is None:
            continue
        if not extras:
            node.pop('extras', None)
        names, active, source = json.loads(metadata)
        if source is not None:
            node.setdefault('extras', {})[_BAKE_SOURCE] = source
        if 'mesh' not in node:
            continue
        for primitive in document['meshes'][node['mesh']]['primitives']:
            attributes = primitive['attributes']
            for index, name in enumerate(names):
                accessor = attributes.get(f'TEXCOORD_{index}')
                if accessor is None:
                    continue  # Missing source data stays missing; installation diagnoses a read.
                aliases = [uv_attribute(name)]
                if index == active:
                    aliases.append(uv_attribute())
                for alias in aliases:
                    if alias in attributes and attributes[alias] != accessor:
                        raise ValueError(f'conflicting Blender UV alias {name!r}')
                    attributes[alias] = accessor


def export_gltf(*, bake_sources=False, **options):
    """Export evaluated geometry with active/named UV identities, without source edits."""
    import bpy
    deps = bpy.context.evaluated_depsgraph_get()
    tagged = []
    from .visibility import METADATA as ray_metadata, flags
    from .instances import SOURCE
    try:
        for obj in bpy.data.objects:
            if obj.type not in ('MESH', 'CURVE', 'SURFACE', 'META', 'FONT'):
                continue
            if _METADATA in obj or ray_metadata in obj:
                raise ValueError(f'object {obj.name!r}: reserved bridge metadata property already exists')
            if bake_sources and _BAKE_SOURCE in obj:
                raise ValueError(f'object {obj.name!r}: reserved bake metadata property already exists')
            tagged.append(obj)
            obj[ray_metadata] = json.dumps(dict(object=obj.name, flags=flags(obj),
                                                 visible_volume_scatter=bool(obj.visible_volume_scatter)))
            if obj.type == 'MESH':
                mesh = obj.evaluated_get(deps).data if options.get('export_apply', False) else obj.data
                layers = mesh.uv_layers
                active = next((i for i, layer in enumerate(layers) if layer.active_render), None)
                obj[_METADATA] = json.dumps([[layer.name for layer in layers], active, obj.get(SOURCE, obj.name) if bake_sources else None])
        options['export_extras'] = True
        result = bpy.ops.export_scene.gltf(**options)
        if 'FINISHED' not in result:
            raise ValueError(f'glTF export did not finish: {result}')
    finally:
        for obj in tagged:
            for key in (_METADATA, ray_metadata):
                if key in obj: del obj[key]
    path = Path(options['filepath'])
    data = path.read_bytes()
    if data[:4] == b'glTF':
        size, kind = struct.unpack_from('<II', data, 12)
        if kind != 0x4E4F534A:
            raise ValueError('glTF binary has no initial JSON chunk')
        document = json.loads(data[20:20 + size])
        for scene in document.get('scenes', []):
            scene.setdefault('nodes', [])
        retain_uv_bindings(document)
        encoded = json.dumps(document, separators=(',', ':')).encode('utf8')
        encoded += b' ' * (-len(encoded) % 4)
        tail = data[20 + size:]
        path.write_bytes(struct.pack('<III', 0x46546C67, 2, 20 + len(encoded) + len(tail))
                         + struct.pack('<II', len(encoded), kind) + encoded + tail)
    else:
        document = json.loads(data)
        for scene in document.get('scenes', []):
            scene.setdefault('nodes', [])
        retain_uv_bindings(document)
        path.write_text(json.dumps(document, separators=(',', ':')) + '\n')


def filter_material_overrides(filepath, bindings):
    """Keep overrides present in the exported asset, reporting every missing slot.

    Blender may drop unused slots (or all faces that use one). Such slots cannot
    receive an override. Filter before lowering their graphs, so a dropped,
    unsupported graph cannot prevent the remaining exported scene from loading.
    The renderer still rejects unknown bindings in authored scene documents.
    """
    import warnings
    path = Path(filepath)
    data = path.read_bytes()
    if data[:4] == b'glTF':
        size, kind = struct.unpack_from('<II', data, 12)
        if kind != 0x4E4F534A:
            raise ValueError('glTF binary has no initial JSON chunk')
        document = json.loads(data[20:20 + size])
    else:
        document = json.loads(data)
    materials = document.get('materials', [])
    names = {material['name'] for material in materials if 'name' in material}
    kept = {}
    for key, value in bindings.items():
        # Match Rust usize parsing: a leading '+' is accepted; overflow and
        # nonnumeric suffixes fall back to a literal imported material name.
        index = key[1:].removeprefix('+') if key.startswith('#') else ''
        digits = index.lstrip('0') or '0'
        maximum = str((1 << (8 * struct.calcsize('P'))) - 1)
        numeric = (index.isascii() and index.isdecimal()
                   and (len(digits), digits) <= (len(maximum), maximum))
        present = int(digits) < len(materials) if numeric else key in names
        if key == '*' or present:
            kept[key] = value
        else:
            warnings.warn(f'{path}: material override {key!r} skipped: no matching '
                          'material in exported glTF (Blender may have dropped the slot or its faces)',
                          stacklevel=2)
    return kept
