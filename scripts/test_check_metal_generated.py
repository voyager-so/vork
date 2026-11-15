import runpy
from pathlib import Path
import unittest
import subprocess
import sys
import tempfile
from unittest.mock import patch

scan = runpy.run_path(str(Path(__file__).with_name('check-metal-generated')))['file_scope_acceleration_structures']
AS = 'metal::raytracing::acceleration_structure<metal::raytracing::instancing>'


class ScopeScan(unittest.TestCase):
    def test_illegal_global(self):
        self.assertEqual(len(scan(f'{AS} ray_scene_0;')), 1)
        self.assertEqual(len(scan(f'struct Context {{ {AS} ray_scene_1; }};\n{AS}\nray_scene_0;')), 1)

    def test_initialized_and_qualified_globals(self):
        self.assertEqual(scan(f'{AS} const global = {{}}; {AS} *pointer;'), ['global', 'pointer'])

    def test_legal_context_and_parameter(self):
        self.assertEqual(scan(f'struct Context {{ {AS} ray_scene_0; }};\nvoid entry({AS} ray_scene_0) {{}}'), [])

    def test_comments_and_strings(self):
        self.assertEqual(scan(f'// {AS} bad;\n/* {{ */\nconst char* s="{{}}";\n{AS} bad;'), ['bad'])

    def test_preprocessor_does_not_hide_global(self):
        self.assertEqual(len(scan(f'#include <metal_stdlib>\n{AS} bad;')), 1)

    def test_function_prototype_is_not_variable(self):
        self.assertEqual(scan(f'void helper({AS} ray_scene_0);'), [])


class ScanCommand(unittest.TestCase):
    def test_producer_executable_runs_without_cargo_in_renderer_cwd(self):
        script = Path(__file__).with_name('check-metal-generated')
        main = runpy.run_path(str(script))['main']
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'modules'
            executable = Path(directory) / 'renderer-tests'
            def capture(command, **kwargs):
                self.assertEqual(command, [str(executable.resolve()),
                    'metal_generated_tests::installed_scene_modules', '--exact', '--ignored', '--nocapture'])
                self.assertEqual(kwargs['cwd'], script.resolve().parent.parent / 'crates/vork-render')
                self.assertEqual(kwargs['env']['VORK_INSTALLED_METAL_DIR'], str(output.resolve()))
                output.joinpath('scene.metal').write_text('kernel void scene() {}')
            with patch.object(sys, 'argv', [str(script), '--executable', str(executable), '--output', str(output)]), \
                 patch.object(subprocess, 'run', side_effect=capture) as run:
                self.assertFalse(main())
                run.assert_called_once()

    def test_empty_module_is_not_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'placeholder.metal').touch()
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('check-metal-generated')),
                '--scan', directory], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('empty MSL module', result.stderr)

    def test_stale_output_is_refused_before_building(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'old.metal').write_text('kernel void old() {}')
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('check-metal-generated')),
                '--output', directory], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('output already contains MSL', result.stderr)


if __name__ == '__main__':
    unittest.main()
