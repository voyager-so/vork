import unittest
from typing import get_args, get_type_hints, is_typeddict
from vork import std, _render_arguments as render_arguments
import vork.types as types


class GeneratedTypes(unittest.TestCase):
    def test_compositor_step_binding_replaces_node_binding(self):
        self.assertEqual(set(get_type_hints(types.StepBinding)), {"step"})
        self.assertEqual(types.StepBinding.__required_keys__, {"step"})
        self.assertFalse(hasattr(types, "NodeBinding"))

    def test_argv_builder_is_private(self):
        import vork
        self.assertFalse(hasattr(vork, "render_arguments"))

    def test_all_references_resolve(self):
        for value in vars(types).values():
            if is_typeddict(value):
                get_type_hints(value)
        self.assertIn('nodes', types.Scene.__required_keys__)

    def test_object_unions_preserve_shared_fields_and_branch_requirements(self):
        self.assertNotIn('passes', get_type_hints(types.FileProgram))
        self.assertNotIn('passes', get_type_hints(types.SourceProgram))
        self.assertIn('bindings', get_type_hints(types.FileProgram))
        self.assertIn('type', types.InlineBuffer.__required_keys__)
        self.assertIn('file', types.FileProgram.__required_keys__)
        self.assertNotIn('source', types.FileProgram.__required_keys__)
        self.assertIn('pass_index', get_type_hints(types.TransformNode))
        self.assertIn('blend', get_type_hints(types.TrackLayer))

    def test_public_variants_have_semantic_names(self):
        for name in ('TransformNode', 'MeshNode', 'AssetNode', 'CameraNode', 'LightNode',
                     'Settings', 'ProfileFile', 'InlineBuffer', 'FileBuffer',
                     'FileProgram', 'SourceProgram', 'TrackLayer', 'ClipLayer'):
            self.assertTrue(is_typeddict(getattr(types, name)), name)

    def test_only_seed_samples_and_denoise_are_public_settings(self):
        self.assertEqual(set(get_type_hints(types.Settings)), {'seed', 'samples', 'denoise'})
        for internal in ['RasterIntegrator', 'PathIntegrator', 'Denoiser', 'FastGi', 'Probes']:
            self.assertFalse(hasattr(types, internal), internal)

    def test_geometry_includes_kernel_buffers_and_authoring_shorthand(self):
        fields = get_type_hints(types.BufferGeometry)
        self.assertIn('attributes', types.BufferGeometry.__required_keys__)
        self.assertNotIn('std', fields)
        self.assertIn('std', types.StdGeometry.__required_keys__)
        self.assertIn('font', get_type_hints(types.StdGeometry))
        for name in ['StdSphereGeometry', 'StdTextGeometry', 'StdExtrudeGeometry']:
            self.assertFalse(hasattr(types, name), name)

    def test_std_uses_shared_cli(self):
        self.assertEqual(std('rounded_box', name='ball')['geometry']['ball']
                         ['attributes']['POSITION'], 'ball_position')

    def test_removed_render_aliases_fail(self):
        for option in ['samples', 'resolution', 'previs']:
            with self.assertRaisesRegex(TypeError, 'unknown option'):
                render_arguments('shot.json', 'a.png', **{option: 1})

    def test_render_channels_default_and_all(self):
        self.assertEqual(render_arguments('shot.json', 'a.png'), ['render', 'shot.json', '--out', 'a.png'])
        self.assertEqual(render_arguments('shot.json', 'a.png', channels=['all']),
                         ['render', 'shot.json', '--out', 'a.png', '--channels', 'all'])

    def test_render_options_map_to_cli_flags(self):
        self.assertEqual(render_arguments('shot.json', 'a.png', at=0, set={'resolution': [64, 32]}),
                         ['render', 'shot.json', '--out', 'a.png', '--at', '0', '--set', 'resolution=[64,32]'])
        for key in ['render.camera', 'camera', 'render.background', 'render.resolution', 'settings.seed', 'exposure']:
            with self.assertRaisesRegex(TypeError, 'unknown render-call field'):
                render_arguments('shot.json', 'a.png', set={key: 1})
        self.assertEqual(render_arguments('shot.json', 'a.png', set={'samples': 7}),
                         ['render', 'shot.json', '--out', 'a.png', '--set', 'samples=7'])
        self.assertNotIn('--previs', render_arguments('shot.json', ['a.mp4', 'b']))
        self.assertEqual(render_arguments('shot.json', 'a.png', profile='std/draft', set={'seed': 4}),
                         ['render', 'shot.json', '--out', 'a.png', '--profile', 'std/draft', '--set', 'seed=4'])
        for key in ('frame_start', 'frame_end'):
            with self.assertRaises(TypeError):
                render_arguments('shot.json', 'frames/', set={key: 1})
        self.assertEqual(render_arguments('shot.json', 'frames/', frame_start=3, frame_end=7),
                         ['render', 'shot.json', '--out', 'frames/', '--frame-start', '3', '--frame-end', '7'])


if __name__ == '__main__':
    unittest.main()
