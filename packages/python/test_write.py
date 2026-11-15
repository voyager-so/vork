import copy
from pathlib import Path
import subprocess
import json
import os
import struct
import tempfile
import unittest
from vork import std, write_scene


class WriteScene(unittest.TestCase):
    """Mirrors packages/ts/test/serialization.test.mjs for writeScene."""

    def test_text_font_paths_survive_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'fonts').mkdir()
            for name in ['a.ttf', 'b.otf']:
                (root / 'fonts' / name).write_bytes(name.encode())
            for font in ['fonts/a.ttf', ['fonts/a.ttf', 'fonts/b.otf']]:
                with self.subTest(font=font):
                    text = {'type': 'text', 'content': 'Hello', 'font': font}
                    run = {'content': 'Hello', 'font': font}
                    scene = {'version': 2, 'composition': {'space': '2d'},
                             'nodes': [text, {'children': [{**text, 'content': [run]}]}]}
                    original = copy.deepcopy(scene)
                    path = Path(write_scene(scene, str(root / 'out/deep/scene.json'), directory))
                    saved = self.read(path)
                    expected = ['../../' + f for f in font] if isinstance(font, list) else '../../' + font
                    nested = saved['nodes'][1]['children'][0]
                    for node in [saved['nodes'][0], nested, nested['content'][0]]:
                        self.assertEqual(node['font'], expected)
                        for file in node['font'] if isinstance(node['font'], list) else [node['font']]:
                            self.assertEqual((path.parent / file).read_bytes(), Path(file).name.encode())
                    self.assertEqual(scene, original)

    def test_dense_fields_round_trip_across_binary_threshold(self):
        binary = os.environ.get('VORK_BIN', os.path.join(os.environ.get('CARGO_TARGET_DIR', str(Path(__file__).resolve().parents[2] / 'target')), 'release', 'vork'))
        for x in (15, 16, 17):
            with self.subTest(x=x), tempfile.TemporaryDirectory() as directory:
                data = [float(i % x) - (x - 1) / 2 for i in range(x * 8 * 8)]
                scene = {'version': 2, 'buffers': {'field': {'type': 'f32', 'dims': [x, 8, 8], 'data': data}},
                         'geometry': {'shape': {'topology': 'implicit', 'attributes': {'field': 'field'}}},
                         'nodes': [{'type': 'mesh', 'geometry': 'shape'}, {'type': 'camera'}]}
                path = write_scene(scene, os.path.join(directory, 'scene.json'))
                saved = self.read(path)['buffers']['field']
                self.assertEqual(saved['dims'], [x, 8, 8])
                if x < 16:
                    self.assertEqual(saved['data'], data)
                else:
                    self.assertEqual(saved['count'], len(data))
                    self.assertEqual(saved['offset'], 0)
                    with open(os.path.join(directory, saved['file']), 'rb') as f:
                        self.assertEqual(f.read(), struct.pack('<' + 'f' * len(data), *data))
                result = subprocess.run([binary, 'check', path], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(scene['buffers']['field']['data'], data)
        with tempfile.TemporaryDirectory() as directory:
            scene['buffers']['field']['dims'] = [2, 2, 2]
            with self.assertRaisesRegex(ValueError, 'dims product'):
                write_scene(scene, os.path.join(directory, 'invalid.json'))

    def test_geometry_stage_order_is_preserved(self):
        scene={'version':2,'nodes':[],'geometry_stages':['displace','deform','compute']}
        with tempfile.TemporaryDirectory() as directory:
            saved=self.read(write_scene(scene,os.path.join(directory,'scene.json')))
            self.assertEqual(saved['geometry_stages'],scene['geometry_stages'])

    def test_numeric_input_tracks_preserve_shared_scene_ir(self):
        animation = [{'tracks': [
            {'path': ['materials', 'paint', 'inputs', 'emission_color'],
             'keys': [[0, [1, 0, 0]], [1, [0, 0, 1]]], 'interpolation': 'LINEAR'},
            {'path': ['materials', 'sky', 'inputs', 'sun_elevation'], 'keys': [[0, 0.2], [1, 0.8]]},
            {'path': ['materials', 'fog', 'inputs', 'density'],
             'keys': [[0, 0.1], [1, 0.5]], 'interpolation': 'STEP'},
        ]}]
        scene = {'version': 2, 'nodes': [], 'materials': {
            'paint': {}, 'sky': {'program': 'std/sky'}, 'fog': {'program': 'std/fog'},
        }, 'animation': animation}
        with tempfile.TemporaryDirectory() as directory:
            saved = self.read(write_scene(scene, os.path.join(directory, 'scene.json')))
            self.assertNotIn('$schema', saved)
            self.assertNotIn('$schema', scene)
            annotated = dict(scene, **{'$schema': 'https://voyager.so/vork/schema/scene.schema.json'})
            annotated_path = write_scene(annotated, os.path.join(directory, 'annotated.json'), directory)
            with open(annotated_path) as f:
                self.assertEqual(json.load(f)['$schema'], annotated['$schema'])
            self.assertEqual(saved['animation'], animation)
            self.assertEqual(saved['materials'], scene['materials'])

    def test_inline_resource_paths_survive_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            os.makedirs(os.path.join(directory, 'textures'))
            os.makedirs(os.path.join(directory, 'shaders'))
            for file, data in [('textures/a.png', b'image'), ('shaders/main.slang', b'import vork;'),
                               ('positions.bin', struct.pack('<9f', 0, 0, 0, 1, 0, 0, 0, 1, 0))]:
                with open(os.path.join(directory, file), 'wb') as f:
                    f.write(data)
            for program in ({'file': 'shaders/main.slang'},
                            {'source': 'import vork;', 'source_origin': 'shaders/main.slang'}):
                scene = {'version': 2, 'nodes': [{'type': 'camera', 'name': 'camera'},
                    {'type': 'node', 'children': [{'type': 'mesh',
                        'geometry': {'attributes': {'POSITION': {'type': 'vec3', 'file': 'positions.bin', 'count': 3}}},
                        'materials': [{'program': program, 'inputs': {'texture': {'image': {'file': 'textures/a.png'}},
                                                                    'file': 'opaque_token'}}]}]}]}
                original = copy.deepcopy(scene)
                path = write_scene(scene, os.path.join(directory, 'out/deep/scene.json'), directory)
                saved = self.read(path)
                mesh = saved['nodes'][1]['children'][0]
                self.assertEqual(mesh['materials'][0]['inputs']['file'], 'opaque_token')
                fields = [(mesh['geometry']['attributes']['POSITION']['file'], 'positions.bin'),
                          (mesh['materials'][0]['inputs']['texture']['image']['file'], 'textures/a.png')]
                for field in ('file', 'source_origin'):
                    if field in program:
                        fields.append((mesh['materials'][0]['program'][field], 'shaders/main.slang'))
                for file, expected in fields:
                    self.assertEqual(os.path.realpath(os.path.join(os.path.dirname(path), file)),
                                     os.path.realpath(os.path.join(directory, expected)))
                self.assertEqual(scene, original)

    def test_resource_slots_match_inline_lowering(self):
        buffer = {'type': 'f32', 'file': 'data.bin', 'count': 1}
        image = {'file': 'texture.png'}
        material = {'program': {'file': 'shader.slang', 'bindings': {'data': buffer}},
                    'inputs': {'texture': {'image': image}, 'density': {'grid': buffer}}}
        scene = {'version': 2, 'nodes': [
            {'type': 'mesh', 'geometry': {'indices': buffer, 'widths': buffer, 'curve_vertex_counts': buffer,
                'attributes': {'POSITION': buffer}, 'morphs': [{'POSITION': buffer, 'NORMAL': buffer}]},
             'materials': [material], 'instances': {'count': buffer, 'attributes': {'x': buffer}},
             'skin': {'inverse_bind_matrices': buffer}},
            {'type': 'light', 'image': image, 'medium': material, 'material': material},
            {'type': 'asset', 'file': 'asset.glb', 'materials': {'slot': material}}],
            'images': {'nested': {'composition':{'compositor':[material]},'render': {}}},
            'composition':{'space':'3d','compositor':[material]},'render': {'compositor_per_step': [material]},
            'animation': [{'tracks': [{'keys': {'times': buffer, 'values': buffer}}]}]}
        with tempfile.TemporaryDirectory() as directory:
            saved = self.read(write_scene(scene, os.path.join(directory, 'out/scene.json'), directory))
            def check(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key == 'file':
                            self.assertTrue(child.startswith('../'), child)
                            self.assertFalse(child.startswith('../../'), child)
                        else:
                            check(child)
                elif isinstance(value, list):
                    for child in value:
                        check(child)
            check(saved)

    def test_nested_sidecar_is_rebased_when_rewriting_in_another_directory(self):
        source = os.path.join(self.directory, 'a')
        destination = os.path.join(self.directory, 'b')
        scene = {'version': 2, 'nodes': [{'type': 'node', 'children': [
            {'type': 'mesh', 'geometry': {'attributes': {'POSITION':
                {'type': 'vec3', 'data': [[i, 0, 0] for i in range(342)]}}}}]}]}
        first = self.read(write_scene(scene, os.path.join(source, 'shot.json')))
        original = first['nodes'][0]['children'][0]['geometry']['attributes']['POSITION']
        second = self.read(write_scene(first, os.path.join(destination, 'shot.json'), base_dir=source))
        moved = second['nodes'][0]['children'][0]['geometry']['attributes']['POSITION']
        self.assertEqual(moved['file'], '../a/' + original['file'])
        with open(os.path.join(destination, moved['file']), 'rb') as a, open(os.path.join(source, original['file']), 'rb') as b:
            self.assertEqual(a.read(), b.read())

    def test_shared_nested_file_buffer_is_rebased_only_once(self):
        source = os.path.join(self.directory, 'input')
        destination = os.path.join(self.directory, 'output', 'deep')
        os.makedirs(source)
        with open(os.path.join(source, 'positions.bin'), 'wb') as f:
            f.write(bytes(36))
        buffer = {'type': 'vec3', 'file': 'positions.bin', 'count': 3}
        scene = {'version': 2, 'nodes': [{'type': 'mesh', 'geometry': {'attributes': {'POSITION': buffer}}} for _ in range(2)]}
        saved = self.read(write_scene(scene, os.path.join(destination, 'shot.json'), base_dir=source))
        for node in saved['nodes']:
            file = node['geometry']['attributes']['POSITION']['file']
            self.assertEqual(file, '../../input/positions.bin')
            self.assertEqual(os.path.getsize(os.path.join(destination, file)), 36)
        self.assertEqual(buffer['file'], 'positions.bin')

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.directory = os.path.realpath(self._directory.name)

    def tearDown(self):
        self._directory.cleanup()

    def read(self, path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def test_replacement_preserves_permissions_without_touching_umask(self):
        from unittest.mock import patch
        import stat
        path = os.path.join(self.directory, 'private.json')
        sidecar = path + '.buffers.bin'
        for file in [path, sidecar]:
            with open(file, 'w') as f:
                f.write('private')
            os.chmod(file, 0o600)
        # Fail deterministically if saving ever queries or changes global umask.
        with patch('os.umask', side_effect=AssertionError('process umask changed')):
            write_scene({'version': 2, 'nodes': [], 'buffers': {'x': {'type': 'f32', 'data': [1] * 1024}}}, path)
        for file in [path, sidecar]:
            self.assertEqual(stat.S_IMODE(os.stat(file).st_mode), 0o600)

    def test_atomic_publication_and_failure_keep_current_scene(self):
        from unittest.mock import patch
        from concurrent.futures import ThreadPoolExecutor
        path = os.path.join(self.directory, 'shot.json')
        def scene(n):
            return {'version': 2, 'nodes': [{'type': 'node', 'name': str(n)}],
                    'buffers': {'x': {'type': 'f32', 'data': [n] * 1024}}}
        write_scene(scene(0), path)
        snapshots = [self.read(path)]
        before_files = sorted(os.listdir(self.directory))
        with patch('os.replace', side_effect=OSError('publication failed')):
            with self.assertRaisesRegex(OSError, 'publication failed'):
                write_scene(scene(99), path)
        self.assertEqual(self.read(path), snapshots[0])
        self.assertEqual(sorted(os.listdir(self.directory)), before_files)
        with ThreadPoolExecutor(max_workers=4) as pool:
            writes = [pool.submit(write_scene, scene(n), path) for n in range(1, 21)]
            while not all(f.done() for f in writes):
                snapshots.append(self.read(path))
            for f in writes:
                f.result()
        snapshots.append(self.read(path))
        for snapshot in snapshots[-1:]:
            with open(os.path.join(self.directory, snapshot['buffers']['x']['file']), 'rb') as f:
                values = struct.unpack('<1024f', f.read())
            self.assertEqual(set(values), {float(snapshot['nodes'][0]['name'])})

    def test_resource_paths_are_relative_to_the_scene_and_input_is_unchanged(self):
        scene = {'version': 2, 'images': {'sky': {'file': 'studio.hdr'}},
                 'programs': {'p': {'source': 'import vork;', 'source_origin': 'shaders/main.slang'}},
                 'animation': [{'clip': {'target': 'actor', 'file': 'walk.glb'}}],
                 'nodes': [{'type': 'node', 'children': [{'type': 'asset', 'file': 'model.glb'}]}]}
        path = write_scene(scene, os.path.join(self.directory, 'scenes/shot.json'), self.directory)
        saved = self.read(path)
        self.assertEqual(saved['nodes'][0]['children'][0]['file'], '../model.glb')
        self.assertEqual(saved['images']['sky']['file'], '../studio.hdr')
        self.assertEqual(saved['programs']['p']['source_origin'], '../shaders/main.slang')
        self.assertEqual(saved['animation'][0]['clip']['file'], '../walk.glb')
        self.assertEqual(scene['nodes'][0]['children'][0]['file'], 'model.glb')

    def test_large_buffers_share_one_little_endian_sidecar(self):
        matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 3, 4, 1]
        scene = {'version': 2, 'nodes': [], 'buffers': {
            'signed': {'type': 'i32', 'data': [-1] * 1024},
            'unsigned': {'type': 'uvec4', 'data': [[0xffffffff, 0, 17, 1]] * 256},
            'matrices': {'type': 'mat4', 'data': [matrix] * 64},
            'positions': {'type': 'vec3', 'data': [[0.1, -2, 3]] * 342},
            'small': {'type': 'f32', 'data': [0.5]},
        }}
        path = write_scene(scene, os.path.join(self.directory, 'shot.json'))
        saved = self.read(path)
        self.assertEqual(saved['buffers']['small'], scene['buffers']['small'])
        self.assertEqual([saved['buffers'][k]['offset'] for k in ('signed', 'unsigned', 'matrices', 'positions')],
                         [0, 4096, 8192, 12288])
        sidecar = saved['buffers']['signed']['file']
        self.assertRegex(sidecar, r'^shot\.json\.buffers\.[0-9a-f-]+\.bin$')
        with open(os.path.join(self.directory, sidecar), 'rb') as f:
            binary = f.read()
        self.assertEqual(len(binary), 16392)
        self.assertEqual(struct.unpack_from('<i', binary, 0)[0], -1)
        self.assertEqual(struct.unpack_from('<I', binary, 4096)[0], 0xffffffff)
        self.assertEqual(struct.unpack_from('<f', binary, 8192 + 12 * 4)[0], 2)
        self.assertEqual(struct.unpack_from('<f', binary, 12288)[0], struct.unpack('<f', struct.pack('<f', 0.1))[0])
        # A reused buffer stays alive until it is no longer referenced.
        write_scene(saved, path, self.directory)
        self.assertEqual(os.path.getsize(os.path.join(self.directory, sidecar)), len(binary))
        write_scene({'version': 2, 'nodes': []}, path)
        self.assertFalse(os.path.exists(os.path.join(self.directory, sidecar)))

    def test_invalid_packed_scalars_are_rejected(self):
        for kind, row in (('u32', -1), ('i32', 2147483648), ('f32', 1e100), ('vec3', [0, 1]), ('u32', 0.5)):
            with self.assertRaises(ValueError, msg=kind):
                write_scene({'version': 2, 'nodes': [], 'buffers': {'bad': {'type': kind, 'data': [row] * 1024}}},
                            os.path.join(self.directory, 'bad.json'))

    def test_std_shapes_go_to_the_sidecar(self):
        shape = std('rounded_box', name='ball')
        scene = {'version': 2, **shape, 'nodes': [{'type': 'mesh', 'geometry': 'ball'}]}
        saved = self.read(write_scene(scene, os.path.join(self.directory, 'ball.json')))
        self.assertEqual(len({b['file'] for b in saved['buffers'].values()}), 1)
        self.assertRegex(next(iter(saved['buffers'].values()))['file'], r'^ball\.json\.buffers\.[0-9a-f-]+\.bin$')
        self.assertNotIn('data', json.dumps(saved))

    def test_std_failure_carries_the_cli_message(self):
        with self.assertRaisesRegex(RuntimeError, r'vork std rounded_box: .*radius must be finite'):
            std('rounded_box', name='well', size=[0.3, 0.002, 0.1], radius=-0.0015)

    def test_symbolic_link_destination_is_refused(self):
        target = os.path.join(self.directory, 'real.json')
        open(target, 'w').close()
        link = os.path.join(self.directory, 'link.json')
        os.symlink(target, link)
        with self.assertRaisesRegex(ValueError, 'symbolic link'):
            write_scene({'version': 2, 'nodes': []}, link)

    def test_unrelated_sidecar_symlink_is_untouched(self):
        other = os.path.join(self.directory, 'other.json.buffers.bin')
        with open(other, 'wb') as f:
            f.write(b'keep')
        os.symlink(other, os.path.join(self.directory, 'shot.json.buffers.bin'))
        write_scene({'version': 2, 'nodes': [], 'buffers': {'x': {'type': 'f32', 'data': [1] * 1024}}},
                    os.path.join(self.directory, 'shot.json'))
        with open(other, 'rb') as f:
            self.assertEqual(f.read(), b'keep')
        self.assertTrue(os.path.islink(os.path.join(self.directory, 'shot.json.buffers.bin')))

    def test_failed_serialization_keeps_the_previous_scene(self):
        path = os.path.join(self.directory, 'shot.json')
        write_scene({'version': 2, 'nodes': [], 'buffers': {'x': {'type': 'f32', 'data': [1] * 1024}}}, path)
        sidecar = os.path.join(self.directory, self.read(path)['buffers']['x']['file'])
        before = (self.read(path), os.path.getsize(sidecar))
        files = sorted(os.listdir(self.directory))
        with self.assertRaises(ValueError):
            write_scene({'version': 2, 'nodes': [], 'buffers': {'x': {'type': 'f32', 'data': [float('nan')]}}}, path)
        self.assertEqual((self.read(path), os.path.getsize(sidecar)), before)
        self.assertEqual(sorted(os.listdir(self.directory)), files)

    def test_sidecar_never_replaces_or_removes_a_named_resource(self):
        shader = os.path.join(self.directory, 'shot.json.buffers.bin')
        with open(shader, 'w') as f:
            f.write('import vork;')
        path = os.path.join(self.directory, 'shot.json')
        for buffers in ({}, {'x': {'type': 'f32', 'data': [1] * 1024}}):
            saved = self.read(write_scene({'version': 2, 'nodes': [], 'buffers': buffers,
                                           'programs': {'p': {'file': 'shot.json.buffers.bin'}}}, path, self.directory))
            with open(shader) as f:
                self.assertEqual(f.read(), 'import vork;')
        self.assertNotEqual(saved['buffers']['x']['file'], 'shot.json.buffers.bin')

    def test_regeneration_cleans_owned_buffers_and_preserves_assets(self):
        path = os.path.join(self.directory, 'shot.json')
        unrelated = path + '.buffers.00000000-0000-0000-0000-000000000000.bin'
        with open(unrelated, 'w') as f:
            f.write('unrelated')
        write_scene({'version': 2, 'nodes': [], 'programs': {'keep': {'file': unrelated}}}, path, self.directory)
        for n in range(3):
            write_scene({'version': 2, 'nodes': [], 'buffers': {'x': {'type': 'f32', 'data': [n] * 1024}}}, path)
            self.assertEqual(len([f for f in os.listdir(self.directory) if f.endswith('.bin')]), 2)
        saved = self.read(path)
        sidecar = saved['buffers']['x']['file']
        os.symlink(sidecar, os.path.join(self.directory, 'buffer-alias.bin'))
        write_scene({'version': 2, 'nodes': [], 'images': {'keep': {'file': 'buffer-alias.bin'}}}, path, self.directory)
        self.assertEqual(os.path.getsize(os.path.join(self.directory, sidecar)), 4096)
        with open(unrelated) as f:
            self.assertEqual(f.read(), 'unrelated')

    @unittest.skipIf(os.name == 'nt', 'POSIX permissions')
    def test_created_files_respect_umask(self):
        import subprocess, sys
        script = """
import os
from vork import write_scene
os.umask(0o022)
write_scene({'version':2,'nodes':[],'buffers':{'x':{'type':'f32','data':[1]*1024}}}, 'mode.json')
"""
        subprocess.run([sys.executable, '-c', script], cwd=self.directory, check=True,
                       env={**os.environ, 'PYTHONPATH': os.path.dirname(os.path.dirname(__import__('vork').__file__))})
        saved = self.read(os.path.join(self.directory, 'mode.json'))
        for file in ['mode.json', saved['buffers']['x']['file']]:
            self.assertEqual(os.stat(os.path.join(self.directory, file)).st_mode & 0o777, 0o644)


if __name__ == '__main__':
    unittest.main()
