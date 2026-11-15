import os
import tempfile
import unittest
from unittest.mock import patch
from vork import std, render


class MissingEngine(unittest.TestCase):
    def test_missing_path_and_explicit_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            for executable in ('vork', os.path.join(directory, 'missing-vork')):
                for call in (lambda: std('rounded_box'), lambda: render('scene.json', 'out.png')):
                    with self.subTest(executable=executable, call=call), patch.dict(os.environ, {'VORK_BIN': executable, 'PATH': directory}):
                        with self.assertRaisesRegex(RuntimeError, 'vork renderer not found:.*Install it with:\\nnpm install -g @voyager-so/vork\\nOr set VORK_BIN'):
                            call()
