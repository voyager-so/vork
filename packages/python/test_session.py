import sys
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from vork import Session, SessionError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from test_support import write_executable

WORKER = '''#!/usr/bin/env python3
import json, os, sys
from vork._version import VERSION
if '--version' in sys.argv:
    print(f'vork {VERSION} (commit abcdef123456; API v2)'); sys.exit(0)
requests, revision = [], 0
for line in sys.stdin:
    r = json.loads(line); requests.append(r)
    if r['op'] == 'open':
        if r.get('document', {}).get('nodes', [{}])[0].get('energy') == 'bright':
            reply = {'id': r['id'], 'ok': False, 'kind': 'scene', 'error': 'load scene (document): nodes.key.energy: invalid type: string'}
        else:
            revision += 1
            reply = {'id': r['id'], 'ok': True, 'revision': revision, 'reuse': 'reinstalled' if revision == 1 else 'updated',
                     'changed': [] if revision == 1 else [['nodes', 'key', 'energy']]}
    elif r['op'] == 'render':
        if r['out'][0].endswith('.jpg'):
            reply = {'id': r['id'], 'ok': False, 'kind': 'usage', 'error': 'unrecognized extension'}
        else:
            renders = sum(1 for q in requests if q['op'] == 'render')
            reply = {'id': r['id'], 'ok': True, 'revision': revision, 'reuse': 'reinstalled' if renders == 1 else 'updated', 'installs': 1,
                     'provisional': r.get('provisional') is True, 'result': {'warnings': [{'code': 'x', 'requested': {}, 'dropped': 'y'}], 'exposure': [{'output': r['out'][0], 'exposure': {'frames': 1}}], 'frames': 1}}
    else:
        reply = {'id': r['id'], 'ok': True}
    sys.stdout.write(json.dumps(reply) + '\\n'); sys.stdout.flush()
    if r['op'] == 'close':
        with open(os.environ['VORK_ARGS_CAPTURE'], 'w') as f:
            json.dump({'argv': sys.argv[1:], 'requests': requests}, f)
        break
'''


class SessionProtocol(unittest.TestCase):
    def test_invalid_json_input_preserves_error_and_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = os.path.join(directory, 'vork')
            capture = os.path.join(directory, 'requests.json')
            write_executable(executable, WORKER)
            with patch.dict(os.environ, {'VORK_BIN': executable, 'VORK_ARGS_CAPTURE': capture}):
                with Session() as session:
                    cycle = {}
                    cycle['cycle'] = cycle
                    for invalid in (float('nan'), float('inf'), -float('inf')):
                        with self.assertRaisesRegex(ValueError, 'Out of range float'):
                            session.render('frame.png', at=invalid)
                    with self.assertRaisesRegex(ValueError, 'Circular reference'):
                        session.open(cycle)
                    with self.assertRaises(TypeError):
                        session.open({'value': object()})
                    self.assertIsNone(session._process.poll())
                    self.assertEqual(session.open('valid.json')['revision'], 1)
            with open(capture) as stream:
                requests = json.load(stream)['requests']
            self.assertEqual([r['op'] for r in requests], ['open', 'close'])
            self.assertEqual([r['id'] for r in requests], [1, 2])

    def test_requests_and_replies(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = os.path.join(directory, 'vork')
            capture = os.path.join(directory, 'requests.json')
            write_executable(executable, WORKER)
            with patch.dict(os.environ, {'VORK_BIN': executable, 'VORK_ARGS_CAPTURE': capture}):
                with Session(gpu='cpu') as s:
                    self.assertEqual(s.open('shot.json'), {'revision': 1, 'reuse': 'reinstalled', 'changed': []})
                    first = s.render('a.png', at=0.5, profile='std/draft', set={'samples': 4, 'denoise': False})
                    self.assertEqual(first, {'warnings': [{'code': 'x', 'requested': {}, 'dropped': 'y'}],
                                             'exposure': [{'output': 'a.png', 'exposure': {'frames': 1}}],
                                             'revision': 1, 'reuse': 'reinstalled', 'installs': 1, 'provisional': False})
                    self.assertEqual(s.open({'version': 2, 'nodes': [{'name': 'key', 'type': 'light', 'energy': 1400}]}, base='shots'),
                                     {'revision': 2, 'reuse': 'updated', 'changed': [['nodes', 'key', 'energy']]})
                    self.assertEqual(s.render(['b.png', 'c.mov'], frame_start=2, frame_end=5, channels=['all'])['reuse'], 'updated')
                    # The first picture available is labelled; the full render after it is not.
                    self.assertTrue(s.render('early.png', provisional=True)['provisional'])
                    self.assertFalse(s.render('final.png')['provisional'])
                    with self.assertRaises(SessionError) as refused:
                        s.open({'version': 2, 'nodes': [{'name': 'key', 'type': 'light', 'energy': 'bright'}]})
                    self.assertEqual(refused.exception.kind, 'scene')
                    self.assertIn('nodes.key.energy', str(refused.exception))
                    with self.assertRaises(TypeError):
                        s.open('shot.json', base='shots')
                    with self.assertRaises(SessionError) as usage:
                        s.render('x.jpg')
                    self.assertEqual(usage.exception.kind, 'usage')
                    with self.assertRaises(TypeError):
                        s.render('x.png', set={'sample': 1})
                s.close()
                with self.assertRaises(RuntimeError):
                    s.open('again.json')
            with open(capture) as f:
                captured = json.load(f)
            self.assertEqual(captured['argv'], ['session', '--gpu', 'cpu'])
            self.assertEqual(captured['requests'], [
                {'id': 1, 'op': 'open', 'scene': 'shot.json'},
                {'id': 2, 'op': 'render', 'out': ['a.png'], 'at': 0.5, 'profile': 'std/draft', 'set': {'samples': 4, 'denoise': False}},
                {'id': 3, 'op': 'open', 'document': {'version': 2, 'nodes': [{'name': 'key', 'type': 'light', 'energy': 1400}]}, 'base': 'shots'},
                {'id': 4, 'op': 'render', 'out': ['b.png', 'c.mov'], 'frame_start': 2, 'frame_end': 5, 'channels': ['all']},
                {'id': 5, 'op': 'render', 'out': ['early.png'], 'provisional': True},
                {'id': 6, 'op': 'render', 'out': ['final.png']},
                {'id': 7, 'op': 'open', 'document': {'version': 2, 'nodes': [{'name': 'key', 'type': 'light', 'energy': 'bright'}]}},
                {'id': 8, 'op': 'render', 'out': ['x.jpg']},
                {'id': 9, 'op': 'close'},
            ])

    def test_close_releases_both_pipes_while_sessions_remain_reachable(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = os.path.join(directory, 'vork')
            write_executable(executable, WORKER)
            capture = os.path.join(directory, 'requests.json')
            with patch.dict(os.environ, {'VORK_BIN': executable, 'VORK_ARGS_CAPTURE': capture}):
                sessions = []
                for failure in (False, True):
                    session = Session()
                    sessions.append(session)
                    if failure:
                        session._process.terminate()
                        session._process.wait()
                        with self.assertRaisesRegex(RuntimeError, 'ended'):
                            session.close()
                    else:
                        session.close()
                    session.close()
                    self.assertTrue(session._process.stdout.closed)
                for session in sessions:
                    self.assertIsNotNone(session._process.returncode)
                    self.assertTrue(session._process.stdin.closed)
                    self.assertTrue(session._process.stdout.closed)

    def test_dead_worker_is_an_error_not_a_hang(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = os.path.join(directory, 'vork')
            write_executable(executable, "#!/usr/bin/env python3\nimport sys\nfrom vork._version import VERSION\nif '--version' in sys.argv: print(f'vork {VERSION} (commit abcdef123456; API v2)'); sys.exit(0)\nsys.stdin.readline()\nsys.exit(3)\n")
            with patch.dict(os.environ, {'VORK_BIN': executable}):
                s = Session()
                with self.assertRaisesRegex(RuntimeError, 'ended .*before replying to open'):
                    s.open('shot.json')
                with self.assertRaises(RuntimeError):
                    s.close()


if __name__ == '__main__':
    unittest.main()
