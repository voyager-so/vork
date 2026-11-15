import sys
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vork import ask, Session, SessionError


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from test_support import write_executable

ENGINE = '''#!/usr/bin/env python3
import json, sys
from vork._version import VERSION
if '--version' in sys.argv:
    print(f'vork {VERSION} (commit abcdef123456; API v2)'); sys.exit(0)
def answers(q):
    if any('bad' in x for x in q): raise ValueError('ask: unknown address')
    return [x.get('at', 0) for x in q]
if sys.argv[1] == 'ask':
    assert sys.argv[2:] == ['scene with spaces.json', '-']
    try: print(json.dumps(answers(json.load(sys.stdin))))
    except ValueError as e: print(e, file=sys.stderr); sys.exit(3)
else:
    revision = 0
    for line in sys.stdin:
        r = json.loads(line)
        reply = {'id':r['id'], 'ok':True}
        if r['op'] == 'open': revision += 1; reply.update(revision=revision,reuse='updated',changed=[])
        elif r['op'] == 'ask':
            try: reply.update(answers=answers(r['questions']),revision=revision)
            except ValueError as e: reply.update(ok=False,kind='scene',error=str(e))
        print(json.dumps(reply), flush=True)
        if r['op'] == 'close': break
'''


class AskProtocol(unittest.TestCase):
    def test_batch_and_session_order_times_and_refusal_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'vork'
            write_executable(binary, ENGINE, mode=0o700)
            with patch.dict(os.environ, {'VORK_BIN': str(binary)}):
                questions = [{'field':'nodes.hero.translation','at':['1/3',2]}, {'buffer':'p'}]
                self.assertEqual(ask('scene with spaces.json', questions), [['1/3',2],0])
                with self.assertRaisesRegex(RuntimeError, 'ask: unknown address'):
                    ask('scene with spaces.json', [{'bad':True}])
                with Session() as session:
                    session.open('scene with spaces.json')
                    self.assertEqual(session.ask(questions), [['1/3',2],0])
                    with self.assertRaises(SessionError): session.ask([{'bad':True}])
                    self.assertEqual(session.ask([]), [])
                with self.assertRaisesRegex(RuntimeError, 'closed'): session.ask(questions)


if __name__ == '__main__': unittest.main()
