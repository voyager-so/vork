import json
from fractions import Fraction
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import captions
import sequences
import spring


class RecipeTests(unittest.TestCase):
    def test_spring_initial_rest_and_settling(self):
        self.assertEqual(spring.spring(0), 0)
        self.assertLess(abs(spring.spring(0.000001)), 1e-9)
        self.assertGreater(max(spring.spring(f / 30) for f in range(61)), 1)
        self.assertAlmostEqual(spring.spring(2), 1, delta=1e-5)

    def test_sequence_frame_boundaries(self):
        scene = sequences.scene()
        cards = scene['nodes']
        tick = Fraction(1001, 30000)
        for frame, active in [(47, [True, False]), (48, [True, True]),
                              (59, [True, True]), (60, [False, True]),
                              (107, [False, True]), (108, [False, False])]:
            self.assertEqual([Fraction(n['active'][0]) <= frame * tick < Fraction(n['active'][1])
                              for n in cards], active)
        self.assertEqual(Fraction(scene['render']['duration']) / tick, 108)
        self.assertEqual(Fraction(cards[1]['animation'][0]['tracks'][0]['keys'][-1][0]) / tick, 12)

    def test_zero_overlap_is_a_hard_cut(self):
        with patch.object(sequences, 'OVERLAP', 0):
            scene = sequences.scene()
        self.assertNotIn('animation', scene['nodes'][1])
        self.assertEqual(scene['nodes'][0]['active'][1], scene['nodes'][1]['active'][0])
        self.assertEqual(Fraction(scene['render']['duration']) / Fraction(1001, 30000), 120)
        with patch.object(sequences, 'OVERLAP', 61), self.assertRaises(ValueError):
            sequences.scene()

    def test_srt_adjacency_gaps_newlines_and_bom(self):
        text = '\ufeff1\r\n00:00:00,500 --> 00:00:01,000\r\nA\r\nB\r\n\r\n2\r\n00:00:01,000 --> 00:00:01,500\r\nC\r\n'
        cues = captions.parse_srt(text)
        self.assertEqual(cues, [(500, 1000, 'A\nB'), (1000, 1500, 'C')])
        keys = captions.scene(cues)['animation'][0]['tracks'][0]['keys']
        self.assertEqual(keys, [['0/1000', ''], ['500/1000', 'A\nB'],
                                ['1000/1000', 'C'], ['1500/1000', '']])
        keys = captions.scene([(500, 1000, 'A'), (1500, 2000, 'B')])['animation'][0]['tracks'][0]['keys']
        self.assertIn(['1000/1000', ''], keys)
        self.assertIn(['1500/1000', 'B'], keys)

    def test_bad_srt_refuses(self):
        for text in ['', 'nonsense', '1\n00:60:00,000 --> 00:61:00,000\nA',
                     '1\n00:00:01,000 --> 00:00:01,000\nA',
                     '1\n00:00:01,000 --> 00:00:00,500\nA',
                     '1\n00:00:00,000 --> 00:00:01,000\nA\n\n2\n00:00:00,500 --> 00:00:02,000\nB']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                captions.parse_srt(text)

    @unittest.skipUnless(os.environ.get('VORK_BIN'), 'requires built CLI')
    def test_scenes_check_with_cli(self):
        cues = captions.parse_srt(Path(captions.__file__).with_suffix('.srt').read_text())
        for scene in [spring.scene(), sequences.scene(), captions.scene(cues)]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'scene.json'
                path.write_text(json.dumps(scene))
                subprocess.run([os.environ['VORK_BIN'], 'check', str(path)], check=True)


if __name__ == '__main__':
    unittest.main()
