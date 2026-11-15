"""Enumerate file slots independently from the full published schema, not the writer graph."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from vork import write_scene

SCHEMA = json.loads((Path(__file__).parent.parent / 'ts/scene.schema.json').read_text())


def resource_paths(node, seen=frozenset(), path=()):
    if not isinstance(node, dict):
        return
    if '$ref' in node:
        ref = node['$ref'].split('/')[-1]
        if ref not in seen:
            yield from resource_paths(SCHEMA['$defs'][ref], seen | {ref}, path)
    for key in ('anyOf', 'oneOf', 'allOf'):
        for child in node.get(key, []):
            yield from resource_paths(child, seen, path)
    for key, child in node.get('properties', {}).items():
        if key in ('file', 'source_origin') and 'string' in str(child.get('type')):
            yield path + (key,)
        else:
            yield from resource_paths(child, seen, path + (key,))
    if 'items' in node:
        yield from resource_paths(node['items'], seen, path + (0,))
    if isinstance(node.get('additionalProperties'), dict):
        # Opaque table/input names deliberately collide with a structural field.
        yield from resource_paths(node['additionalProperties'], seen, path + ('file',))


def cases():
    paths = set(resource_paths(SCHEMA))
    paths |= {('nodes', 0, 'children', 0) + p[2:] for p in paths if p[:1] == ('nodes',)}
    paths |= {('images', 'nested') + p for p in paths if p[:2] == ('composition', 'compositor')}
    return sorted(paths, key=str)


def scene_at(path):
    value = 'resource.bin'
    for key in reversed(path):
        value = [value] if isinstance(key, int) else {key: value}
    return {'version': 2, 'nodes': [], **value}


class ResourceSlots(unittest.TestCase):
    def test_all_schema_resource_slots_preserve_source_identity(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'source'
            source.mkdir()
            ((source / 'resource.bin').resolve()).write_bytes(b'resource')
            for path in cases():
                with self.subTest(path=path):
                    scene = scene_at(path)
                    original = copy.deepcopy(scene)
                    output = Path(write_scene(scene, str(Path(root) / 'export/scene.json'), str(source)))
                    saved = json.loads(output.read_text())
                    value = saved
                    for key in path:
                        value = value[key]
                    self.assertEqual((output.parent / value).resolve(), (source / 'resource.bin').resolve())
                    self.assertEqual(scene, original)


if __name__ == '__main__':
    # TypeScript consumes the same exhaustive cases, while using its own writer.
    print(json.dumps([{'path': p, 'scene': scene_at(p)} for p in cases()]))
