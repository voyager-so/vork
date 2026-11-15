import json
import os
import tempfile
import unittest
from unittest.mock import patch
from vork import render


class RenderExposure(unittest.TestCase):
    def test_rational_output_rates_reach_the_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, 'frame.png')
            with open(output + '.meta.json', 'w') as stream:
                json.dump({'warnings': []}, stream)
            with patch('vork._engine.engine', return_value='vork'), patch('subprocess.run') as run:
                render('shot.json', output, at=0.5, set={'fps': '30000/1001'})
            self.assertEqual(run.call_args.args[0][1:],
                             ['render', 'shot.json', '--out', output, '--at', '0.5', '--set', 'fps="30000/1001"'])

    def test_returns_full_precision_per_output_and_keeps_cli_stdout_visible(self):
        summary = {'frames': 2, 'coverage': 0.5, 'clipped': 0.123456789, 'near_black': 0.2, 'mean': 0.41,
                   'worst_clipped': {'share': 0.25, 'frame': 7},
                   'worst_near_black': {'share': 0.3, 'frame': 8}}
        empty = dict(frames=1, coverage=0, clipped=None, near_black=None, mean=None,
                     worst_clipped=None, worst_near_black=None)
        summaries = [summary, empty, summary]
        degradation = dict(code="nonfinite_path_samples", requested={"frame": 795, "dropped_path_samples": 4294967299}, dropped="zero samples remain in average; check inputs and rerender")
        warning = dict(code="additional_environment", requested={"node": {"name": "second", "energy": 2}}, dropped="second environment; retaining first")
        with tempfile.TemporaryDirectory() as directory:
            outputs = [os.path.join(directory, p) for p in ['still.png', 'clip.MOV', 'frames.png/', 'controls', 'exr/%06d.exr']]
            paths = ['still.png.meta.json', 'clip.MOV.meta.json', 'frames.png/meta.json', 'controls/meta.json', 'exr/meta.json']
            for i, path in enumerate(paths):
                path = os.path.join(directory, path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w') as f:
                    json.dump({'exposure': summaries[i], 'warnings': [warning, degradation]} if i < 3 else {'warnings': [warning, degradation]}, f)
            with patch('vork._engine.engine', return_value='vork'), patch('subprocess.run') as run:
                result = render('shot.json', outputs)
                self.assertEqual(run.call_args.kwargs, {'check': True})
            self.assertEqual(result, {'exposure': [{'output': out, 'exposure': summaries[i]} for i, out in enumerate(outputs[:3])], 'warnings': [warning, degradation]})



if __name__ == '__main__':
    unittest.main()
